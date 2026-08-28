# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep more than one validation batch in flight, retire them in order.

A validation batch is a turn loop: tokenise, generate, step the environment,
repeat. Only the generate is on the GPU. Measured on the multitask SFT arm after
the retriever was batched, one search batch is 11.87 s of generation against
1.20 s of tokenising and 1.17 s of environment -- so 16.5% of the run is the GPU
waiting for work that cannot be brought forward, because the next turn's prompt
is the environment's answer to this turn.

Nothing inside one batch can fill that. Another batch can: batches are
independent, so while one waits on its environment the other can generate. The
worker group is a Ray actor and runs one call at a time, so the two generations
serialise on their own -- what overlaps is one batch's environment and
tokenising against the other's generation, which is exactly the gap.

Order is preserved. Results are yielded in submission order, and the caller does
its accumulation on the calling thread, so nothing downstream of this sees a
concurrent run: the same batches produce the same rows in the same sequence.

A slot carries the resources one batch needs to itself -- its own environment
manager and trajectory collector, since both hold per-rollout state. Slots may
be restricted to certain tasks: alfworld draws its episodes from a seeded
game-file cycle indexed by position within its manager, so a second manager
would score different games and it does not get one. Search has no such state
(each row carries its own question), which is the 411 batches of 413 that
matter.
"""

import contextlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as futures_wait
from typing import Any, Callable, Iterable, List, Optional


# How often the coverage summary is printed, in retired batches. A summary only
# at the end needs the whole validation to finish -- fifty minutes to answer a
# question a tenth of that would settle.
_REPORT_EVERY = int(os.environ.get("VAL_PIPELINE_REPORT_EVERY", "25"))


def _coverage(intervals):
    """Total wall covered by at least one interval, and the span they lie in."""
    if not intervals:
        return 0.0, 0.0
    ordered = sorted(intervals)
    covered, cur_start, cur_end = 0.0, ordered[0][0], ordered[0][1]
    for start, end in ordered[1:]:
        if start > cur_end:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    covered += cur_end - cur_start
    return covered, ordered[-1][1] - ordered[0][0]


class Slot:
    """One batch's worth of dedicated resources.

    ``tasks=None`` means the slot can run any batch; a set restricts it. Two
    batches must never share a slot at the same time -- the environment manager
    keeps per-rollout state (observation history, per-env step counters) that a
    second concurrent rollout would corrupt.
    """

    __slots__ = ("name", "tasks", "envs", "collector")

    def __init__(self, name: str, envs: Any, collector: Any, tasks: Optional[Iterable[str]] = None):
        self.name = name
        self.envs = envs
        self.collector = collector
        self.tasks = None if tasks is None else set(tasks)

    def accepts(self, task: Optional[str]) -> bool:
        return self.tasks is None or task in self.tasks

    def __repr__(self):
        return f"Slot({self.name}, tasks={'any' if self.tasks is None else sorted(self.tasks)})"


@contextlib.contextmanager
def _gauge(name, n=1):
    """Hold n on a profiler gauge, if profiling is on."""
    try:
        from verl.utils import gpu_profiler
    except Exception:  # pragma: no cover - profiler is optional
        yield
        return
    with gpu_profiler.inflight(name, n):
        yield


@contextlib.contextmanager
def _activity(name):
    """Count the CALLING thread in the activity census, if profiling is on.

    The census tagged the slot threads and nothing else, so the thread that
    loads, prepares and scores -- which holds the GIL for all of it -- showed up
    nowhere. That is what "0.8 of 3 slots in no tagged phase" was made of, and
    it was invisible in exactly the samples where every card was empty.
    """
    try:
        from verl.utils import gpu_profiler
    except Exception:  # pragma: no cover - profiler is optional
        yield
        return
    with gpu_profiler.activity(name):
        yield


def _residency_over(spans):
    """The NVML residency line for the span the launches cover, if sampling is on."""
    if not spans:
        return ""
    try:
        from verl.utils import gpu_profiler
    except Exception:  # pragma: no cover - profiler is optional
        return ""
    # The spans are perf_counter and the sampler stamps monotonic. Those are the
    # same clock on Linux and NOT on every platform, and getting it wrong yields
    # an empty window rather than an error -- which is to say, silence that reads
    # exactly like "the GPU was never idle". Measure the offset instead.
    offset = time.monotonic() - time.perf_counter()
    res = gpu_profiler.residency_between(
        min(start for start, _end in spans) + offset,
        max(end for _start, end in spans) + offset,
    )
    return gpu_profiler.format_residency(res)


def run_pipelined(
    items: Iterable[Any],
    prepare: Callable[[Any], Any],
    task_of: Callable[[Any], Optional[str]],
    launch: Callable[[Any, Slot], Any],
    slots: List[Slot],
):
    """Yield ``(prepared, result)`` in item order, several launches in flight.

    ``prepare`` runs on the calling thread, in item order: it is the part that
    touches shared state (tokeniser, trainer fields) and must stay sequential.
    ``launch`` runs on a worker thread with a slot to itself.

    With one slot nothing is submitted to a thread at all -- the call is inline,
    so a run that does not ask for a pipeline behaves exactly as it did before.
    """
    if len(slots) <= 1:
        slot = slots[0]
        for item in items:
            prepared = prepare(item)
            yield prepared, launch(prepared, slot)
        return

    executor = ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="val-slot")
    # A SLOT IS RELEASED WHEN ITS OWN BATCH FINISHES, NOT WHEN ITS TURN COMES.
    #
    # This used to be a list retired oldest-first: retire() popped inflight[0]
    # and blocked on that future, so slots belonging to batches that had already
    # finished stayed out of `free` until every older batch was collected. Head
    # of line blocking, and not a small one -- an alfworld batch runs 239.7 s
    # (126 rows to 50 turns) against a search batch's ~65 s, so while alfworld
    # sat at the head, three finished search slots were unusable for as much as
    # 175 s and a four-slot pipeline ran as one. That is the signature the
    # profiler kept reporting: TAIL_BLOCKS_READY, ready=1, gen_inflight=1..4 --
    # a handful of requests being alfworld's late turns, with everything else
    # idle behind it.
    #
    # WHAT DOES NOT CHANGE: results are still handed back in submission order,
    # the caller still accumulates on this thread, and a slot is still released
    # only after its own future has RESOLVED -- never while its rollout is
    # running, which would let two batches share one environment manager. The
    # rows scored, and their order, are identical.
    inflight = {}     # seq -> (prepared, future, slot); insertion order = submission order
    finished = {}     # seq -> (prepared, result), resolved but not yet handed back
    free = list(slots)
    next_seq = 0      # the next batch to submit
    next_out = 0      # the next batch to hand back

    # Where the wall clock goes. The per-batch tables measure inside a rollout
    # and the occupancy ratio averages the slots, so neither can see the state
    # that matters here: EVERY slot finished and none resubmitted, because the
    # calling thread is between them. NVML reported 23% of an evaluation's
    # samples at exactly zero on all three cards; a union of the launch
    # intervals is what says whether that is this.
    spans = []  # (start, end) per launch, appended from the worker threads
    clock = {"prepare": 0.0, "consumer": 0.0, "retire_wait": 0.0, "dataload": 0.0, "dataload_max": 0.0}

    def timed_launch(prepared, slot):
        started = time.perf_counter()
        try:
            return launch(prepared, slot)
        finally:
            spans.append((started, time.perf_counter()))

    def harvest():
        """Free the slot of every batch whose future has resolved.

        future.result() is called here rather than at hand-back time, so a
        rollout that raised raises on this thread as it did before -- just
        earlier, and still before anything after it is yielded.
        """
        for seq in [s for s, (_p, fut, _s) in inflight.items() if fut.done()]:
            prepared, future, slot = inflight.pop(seq)
            finished[seq] = (prepared, future.result())
            free.append(slot)

    def wait_for_one():
        """Block until at least one more batch finishes, then free its slot."""
        pending = [fut for _p, fut, _s in inflight.values()]
        waited = time.perf_counter()
        try:
            # Counted, because the stack sampler found this thread parked here
            # while every card was idle and the census could not see it: it is
            # in no tagged phase and burns no CPU.
            with _gauge("future_wait"):
                futures_wait(pending, return_when=FIRST_COMPLETED)
        finally:
            clock["retire_wait"] += time.perf_counter() - waited
        harvest()

    @contextlib.contextmanager
    def _scoring():
        """The caller's accumulation, which runs on this thread between yields."""
        with _activity("scoring"):
            yield

    def report(final):
        covered, span = _coverage(spans)
        if span <= 0:
            return
        idle = span - covered
        tag = "final" if final else f"after {len(spans)}"
        print(
            f"[val-pipeline] {tag}: {len(spans)} batches over {span:.1f}s: at least one slot "
            f"running {covered:.1f}s ({100 * covered / span:.1f}%), NOTHING running {idle:.1f}s "
            f"({100 * idle / span:.1f}%). Calling thread: dataload {clock['dataload']:.1f}s "
            f"(worst single wait {clock['dataload_max']:.1f}s), prepare {clock['prepare']:.1f}s, "
            f"scoring {clock['consumer']:.1f}s, waiting on a slot {clock['retire_wait']:.1f}s.",
            flush=True,
        )
        # The line above answers "was a slot running", which is NOT "was a GPU
        # running": a slot blocked in env.step is running by that measure while
        # every card is empty. It said NOTHING running 0.1% for a run in which
        # NVML saw 285 s of node-wide idle, and that reading is what made the
        # async pump look like it was chasing 0.4%. Print the device's answer
        # next to it so the two can never be confused again.
        residency = _residency_over(spans)
        if residency:
            print(residency, flush=True)

    def maybe_report():
        if _REPORT_EVERY > 0 and len(spans) and len(spans) % _REPORT_EVERY == 0:
            report(final=False)

    # next(items) is the loader, and it was the one thing on the calling thread
    # that nothing timed: with num_workers>0 torch hands batches back in strict
    # worker order, so one slow worker stalls the loop every num_workers
    # batches -- the shape of the periodic stalls NVML shows.
    iterator = iter(items)
    try:
        while True:
            _t = time.perf_counter()
            try:
                with _activity("dataload"):
                    item = next(iterator)
            except StopIteration:
                break
            waited = time.perf_counter() - _t
            clock["dataload"] += waited
            clock["dataload_max"] = max(clock["dataload_max"], waited)

            _t = time.perf_counter()
            with _activity("prepare"):
                prepared = prepare(item)
            clock["prepare"] += time.perf_counter() - _t
            task = task_of(prepared)

            # THE GAUGE THE CLASSIFIER TURNS ON. From here until the batch is
            # submitted there is a whole batch of work that exists and is not
            # with the engine. If the cards go idle in this window it is not the
            # retriever's fault, however busy the retriever happens to be --
            # that is the difference between RETRIEVER DEPENDENCY and SCHEDULER
            # STARVATION, and the two look identical on the device.
            with _gauge("ready"):
                slot = None
                while slot is None:
                    harvest()
                    # A batch whose task only one slot serves waits for that
                    # slot specifically, which is why this looks for an
                    # acceptor rather than taking whatever is free.
                    slot = next((candidate for candidate in free if candidate.accepts(task)), None)
                    if slot is not None:
                        break
                    if not inflight:
                        raise RuntimeError(f"no slot can run task {task!r}: {slots}")
                    # Hand back whatever is already in order before blocking:
                    # the caller's scoring then runs while the slots are busy
                    # rather than after they have gone idle.
                    while next_out in finished:
                        payload = finished.pop(next_out)
                        next_out += 1
                        resumed = time.perf_counter()
                        with _scoring():
                            yield payload
                        clock["consumer"] += time.perf_counter() - resumed
                        maybe_report()
                    if free and any(candidate.accepts(task) for candidate in free):
                        continue
                    if inflight:
                        wait_for_one()
            free.remove(slot)
            inflight[next_seq] = (prepared, executor.submit(timed_launch, prepared, slot), slot)
            next_seq += 1

            # Submitted first, scored second: the new batch is already running
            # while the caller accumulates the finished ones.
            while next_out in finished:
                payload = finished.pop(next_out)
                next_out += 1
                resumed = time.perf_counter()
                with _scoring():
                    yield payload
                clock["consumer"] += time.perf_counter() - resumed
                maybe_report()

        while inflight or finished:
            harvest()
            while next_out in finished:
                payload = finished.pop(next_out)
                next_out += 1
                resumed = time.perf_counter()
                with _scoring():
                    yield payload
                clock["consumer"] += time.perf_counter() - resumed
                maybe_report()
            if inflight and next_out not in finished:
                wait_for_one()
    finally:
        report(final=True)
        # Never leave a rollout running into the caller's next phase; a batch
        # still generating would be holding the worker group and the engine.
        for _prepared, future, _slot in inflight.values():
            future.cancel()
        executor.shutdown(wait=True)
