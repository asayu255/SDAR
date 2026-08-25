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

import os
import time
from concurrent.futures import ThreadPoolExecutor
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
    inflight = []  # [(prepared, future, slot)] oldest first -- also the retirement order
    free = list(slots)

    # Where the wall clock goes. The per-batch tables measure inside a rollout
    # and the occupancy ratio averages the slots, so neither can see the state
    # that matters here: EVERY slot finished and none resubmitted, because the
    # calling thread is between them. NVML reported 23% of an evaluation's
    # samples at exactly zero on all three cards; a union of the launch
    # intervals is what says whether that is this.
    spans = []  # (start, end) per launch, appended from the worker threads
    clock = {"prepare": 0.0, "consumer": 0.0, "retire_wait": 0.0}

    def timed_launch(prepared, slot):
        started = time.perf_counter()
        try:
            return launch(prepared, slot)
        finally:
            spans.append((started, time.perf_counter()))

    def retire():
        """Finish the oldest batch, hand it back, and free its slot."""
        prepared, future, slot = inflight.pop(0)
        waited = time.perf_counter()
        try:
            result = future.result()
        finally:
            clock["retire_wait"] += time.perf_counter() - waited
            free.append(slot)
        return prepared, result

    def handed_back():
        """Yield one retired batch and charge the caller's time to the caller."""
        payload = retire()
        resumed = time.perf_counter()
        return payload, resumed

    def report(final):
        covered, span = _coverage(spans)
        if span <= 0:
            return
        idle = span - covered
        tag = "final" if final else f"after {len(spans)}"
        print(
            f"[val-pipeline] {tag}: {len(spans)} batches over {span:.1f}s: at least one slot "
            f"running {covered:.1f}s ({100 * covered / span:.1f}%), NOTHING running {idle:.1f}s "
            f"({100 * idle / span:.1f}%). Calling thread: prepare {clock['prepare']:.1f}s, "
            f"scoring {clock['consumer']:.1f}s, waiting on a slot {clock['retire_wait']:.1f}s.",
            flush=True,
        )

    def maybe_report():
        if _REPORT_EVERY > 0 and len(spans) and len(spans) % _REPORT_EVERY == 0:
            report(final=False)

    try:
        for item in items:
            _t = time.perf_counter()
            prepared = prepare(item)
            clock["prepare"] += time.perf_counter() - _t
            task = task_of(prepared)
            # Retire until a slot this batch can use is free. A batch whose task
            # only one slot serves waits for that slot specifically, which is why
            # this drains rather than picking any free one.
            while True:
                slot = next((candidate for candidate in free if candidate.accepts(task)), None)
                if slot is not None:
                    break
                if not inflight:
                    raise RuntimeError(f"no slot can run task {task!r}: {slots}")
                payload, resumed = handed_back()
                yield payload
                clock["consumer"] += time.perf_counter() - resumed
                maybe_report()
            free.remove(slot)
            inflight.append((prepared, executor.submit(timed_launch, prepared, slot), slot))
        while inflight:
            payload, resumed = handed_back()
            yield payload
            clock["consumer"] += time.perf_counter() - resumed
            maybe_report()
    finally:
        report(final=True)
        # Never leave a rollout running into the caller's next phase; a batch
        # still generating would be holding the worker group and the engine.
        for _, future, _ in inflight:
            future.cancel()
        executor.shutdown(wait=True)
