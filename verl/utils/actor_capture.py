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
"""A few micro-batches of every rank, traced, with the phases named.

``gpu_profiler`` has reached its resolution. Its sampler cannot go below ~330 ms
(an NVML read of three devices costs ~130 ms on top of the interval), and one
micro-batch's forward is ~1.14 s of about 500 kernels. So a stall of 0.3-1.0 s
-- a quarter to the whole of one forward, on one rank -- lands inside a single
``actor.fwd`` sample and cannot be placed any more precisely than "somewhere in
that phase". Two questions it therefore cannot answer, and both decide what to
fix:

  * **Which way does the straggler point?** The signature is one card at 0 and
    two at 99-100, but NCCL's spin kernels count as busy in NVML, so "the other
    two are computing" and "the other two are blocked waiting for this one" read
    identically. In a kernel timeline the collectives appear by name
    (``ncclDevKernel_AllGather...``) with their durations, which settles it.
  * **Is it even a gap?** NVML reports "no kernel resident". That is equally
    consistent with an empty stream (the host stopped submitting -- a CPU-side
    stall), a kernel too short for the sampler to catch, and a wait on an event.
    Those want completely different fixes.

Two backends answer those, over the same window, and neither is on unless its
own count is set:

  ``ACTOR_NSYS_MICRO``    Nsight Systems. The better instrument -- it sees the
                          driver and the OS runtime, not just the parts of the
                          process torch knows about. It needs Ray's ``_nsight``
                          plugin attached at ``ray.init`` and an offline
                          ``.qdstrm`` -> ``.nsys-rep`` conversion afterwards.
  ``ACTOR_TORCH_MICRO``   ``torch.profiler``. Strictly less than Nsight sees,
                          and its own CUPTI start-up perturbs the first captured
                          micro-batch. It earns its place by having no moving
                          parts outside the process: it writes a finished Chrome
                          trace from inside the rank, so nothing downstream can
                          fail to convert it, and ``record_shapes`` puts the
                          per-op tensor shapes in the file -- which is the direct
                          test of whether a slow rank was handed more tokens.
                          ``scripts/actor_trace_summary.py`` reads the result.

Everything is a no-op unless one of those is set: no NVTX pushes, no profiler
calls, one bool per micro-batch. Unlike ``gpu_profiler``'s worker-side stages
this runs on EVERY rank, because comparing ranks is the entire point.

Env vars
--------
  ACTOR_NSYS_MICRO=20   capture this many micro-batches under Nsight, then stop.
                        Events run about one per 5.5 micro-batches, so 20 should
                        hold 3-4.
  ACTOR_NSYS_SKIP=40    micro-batches to let past first. The allocator settles
                        and the first mini-batch pays warm-up costs that are not
                        what is being chased.
  ACTOR_NSYS_TRACE=...  the ``-t`` list; see ``nsight_runtime_env``.
  ACTOR_TORCH_MICRO=20  the same window for ``torch.profiler``. Independent of
                        the Nsight one: either, both, or neither.
  ACTOR_TORCH_SKIP=40
  ACTOR_TORCH_DIR=...   where the per-rank Chrome traces are written
                        (default /tmp/actor_trace).

How to run the Nsight one
-------------------------
``torch.cuda.profiler.start()`` only opens a capture range; something has to be
attached to the process to honour it, and the ranks are Ray actors rather than
children of the launcher, so wrapping the launch command reaches nothing. Ray's
``_nsight`` runtime-env plugin wraps each worker instead, and
``main_sft_multitask`` fills it in when ACTOR_NSYS_MICRO is set. One report per
process lands in the Ray session's ``logs/nsight`` directory.
"""

import os
import time
from contextlib import contextmanager

__all__ = [
    "enabled",
    "nsys_enabled",
    "torch_enabled",
    "nsight_runtime_env",
    "phase",
    "micro_batch",
    "iter_micro_batches",
]


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


_NSYS_MICRO = _int_env("ACTOR_NSYS_MICRO", 0)
_NSYS_SKIP = _int_env("ACTOR_NSYS_SKIP", 40)
# Overridable because osrt is the one entry here that is suspected of breaking
# the import: the first real capture died in QdstrmImporter with "Wrong event
# order has been detected", and the offending records were TraceProcessEvents
# carrying syscall return values -- osrt's shape. Dropping it costs the host-side
# detail and keeps everything else, so it is worth being able to try without a
# code change: ACTOR_NSYS_TRACE=cuda,cudnn,cublas,nvtx
_NSYS_TRACE = os.environ.get("ACTOR_NSYS_TRACE", "cuda,cudnn,cublas,nvtx,osrt").strip()

# The stall watch is on by default, unlike the two capture backends, because
# the thing it is looking for is rare: five events in 68 minutes of training,
# at unpredictable positions inside their steps. A capture window fixed at
# micro-batch 40 of step 1 has essentially no chance of containing one -- which
# is why the first Nsight capture said nothing about the spike, and why timing
# every micro-batch of every step is the instrument this needs instead. Two
# CUDA event records per micro-batch is a few microseconds against a ~1 s body.
_STALL_FACTOR = float(os.environ.get("ACTOR_STALL_FACTOR", "3.0") or 0)
_STALL_MIN_MS = float(os.environ.get("ACTOR_STALL_MIN_MS", "500") or 0)
# Every line also goes to a per-rank file, because the console loses two of the
# three ranks. Ray's log dedup matches on the message with numbers substituted,
# so "[step-gpu] rank 0 step 7: ..." and "[step-gpu] rank 1 step 7: ..." are one
# pattern to it: it prints one and collapses the rest to "[repeated 2x across
# cluster]". Comparing ranks is the entire point of this instrument, so it cannot
# depend on RAY_DEDUP_LOGS being remembered on the command line.
_STALL_DIR = os.environ.get("ACTOR_STALL_DIR", "/tmp/actor_stall")

_TORCH_MICRO = _int_env("ACTOR_TORCH_MICRO", 0)
_TORCH_SKIP = _int_env("ACTOR_TORCH_SKIP", 40)
_TORCH_DIR = os.environ.get("ACTOR_TORCH_DIR", "/tmp/actor_trace")

_seen = 0
_nsys_running = False
_nsys_finished = False
_torch_prof = None
_torch_finished = False

_watch = None       # _StallWatch, built on the first micro-batch that has a device


def nsys_enabled() -> bool:
    return _NSYS_MICRO > 0


def torch_enabled() -> bool:
    return _TORCH_MICRO > 0


def enabled() -> bool:
    return nsys_enabled() or torch_enabled()


def stall_watch_enabled() -> bool:
    return _STALL_FACTOR > 0


def nsight_runtime_env() -> dict:
    """The ``_nsight`` runtime-env value that makes the capture ranges real.

    ``capture-range=cudaProfilerApi`` keeps the report to the window this module
    opens rather than the whole run, which is what makes tracing a multi-hundred
    -second step affordable. ``nvtx`` in the trace list is what carries the phase
    names across; without it the timeline is anonymous kernels.
    """
    return {
        # osrt earns its place on this host: nsys reports "CPU IP/backtrace
        # sampling not supported" here, so there are no Python stacks to say
        # what the host was doing during a gap. The CUDA API trace still shows
        # whether it stopped submitting or blocked in a synchronize, and osrt is
        # what turns "blocked" into which call -- a futex, a read, a condvar.
        # ACTOR_NSYS_TRACE drops it if it turns out to be what breaks the import.
        "t": _NSYS_TRACE,
        "o": "'actor_rank_%p'",
        "capture-range": "cudaProfilerApi",
        # stop-shutdown, not stop: end the session at cudaProfilerStop instead of
        # leaving it attached for the rest of the run. The report is then written
        # at that moment (so a later Ctrl-C cannot lose it), and nothing after the
        # window can reach the event stream -- which is the shape of the
        # "Wrong event order has been detected" import failure this hit.
        "capture-range-end": "stop-shutdown",
        "stop-on-exit": "true",
    }


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return -1


def _torch_trace_path() -> str:
    """One file per process. The rank is the useful name and the pid is the
    unique one, so the file carries both: two runs into the same directory then
    do not overwrite each other, and a stale file is obvious by its pid."""
    return os.path.join(_TORCH_DIR, f"actor_rank{_rank()}_pid{os.getpid()}.json")


def _torch_start():
    """Open the torch capture. A failure here must not take the run down: this
    is a diagnostic, and CUPTI is the part of the stack most likely to refuse."""
    global _torch_prof, _torch_finished
    import torch

    try:
        _torch_prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # The direct test of "was this rank handed more tokens": the shapes
            # are in the file next to the durations, so the answer does not
            # depend on trusting a separate log.
            record_shapes=True,
            # Python stacks would multiply the file size for a question that is
            # about the device timeline, and this host cannot sample them anyway.
            with_stack=False,
        )
        _torch_prof.start()
        print(f"[torch-capture] started at micro-batch {_seen} "
              f"(rank {_rank()}, pid {os.getpid()})", flush=True)
    except Exception as exc:   # pragma: no cover - needs a broken CUPTI
        print(f"[torch-capture] disabled: {exc!r}", flush=True)
        _torch_prof = None
        _torch_finished = True


def _torch_stop():
    """Close it and write the trace. Same rule: never raise into training."""
    global _torch_prof, _torch_finished
    prof, _torch_prof, _torch_finished = _torch_prof, None, True
    try:
        prof.stop()
        os.makedirs(_TORCH_DIR, exist_ok=True)
        path = _torch_trace_path()
        prof.export_chrome_trace(path)
        size = os.path.getsize(path) / (1 << 20)
        print(f"[torch-capture] wrote {path} ({size:.1f} MiB, "
              f"{_TORCH_MICRO} micro-batches)", flush=True)
    except Exception as exc:   # pragma: no cover - needs a full disk
        print(f"[torch-capture] trace lost: {exc!r}", flush=True)


class _StallWatch:
    """Times every micro-batch of every step, on every rank, and reports outliers.

    Built for the shape of the thing being chased. The dips are rare (five in 68
    minutes), several seconds long, land at unpredictable positions inside a
    step, and show on all three cards at once. None of that survives a capture
    window pinned to micro-batch 40 of step 1, so the only instrument that can
    see one is one cheap enough to leave on for the whole run.

    Two CUDA events per micro-batch, read back only once they have completed --
    ``query()`` never blocks, so the host is not synchronised and the timing does
    not create the stall it is looking for. Three numbers come out and they
    separate the hypotheses:

      gpu_ms   the micro-batch's span on the device timeline.
      gap_ms   device-timeline idle between the previous micro-batch ending and
               this one starting. A stall here is BETWEEN micro-batches -- the
               optimizer step, the gradient reduce, the next batch's H2D.
      host_ms  the same span in host wall time. host ~= gpu means the host was
               blocked too, so whatever stopped is upstream of the device; host
               << gpu means the host ran ahead and the device itself was slow,
               which is what a collective wait looks like.
    """

    _KEEP = 64          # samples behind the running median

    def __init__(self, device_module):
        self._cuda = device_module
        self._log = self._open_log()
        self._pending = []            # (counter, kind, start_ev, end_ev, host_start, host_end)
        self._recent = []             # completed gpu_ms, for the median
        # Gaps are not one population. The gap before the first micro-batch of a
        # step contains everything between two update_policy calls -- the return
        # to the driver, its logging, the next batch's dispatch and H2D -- and the
        # gap before the first micro-batch of a mini-batch contains the gradient
        # reduce and the optimizer step. Both are structurally larger than an
        # interior gap and both happen on a fixed schedule, so judging them
        # against one median flags them every single time: 300 lines a run, with
        # the five events that matter buried in them.
        self._recent_gap = {"step": [], "mini": [], "interior": []}
        self._prev_end = None         # to measure the gap across micro-batches
        self._new_step = True
        self._reported = 0
        self._step = 0
        self._acc = None
        self._reset_acc()

    def _reset_acc(self):
        # partial: this step's first gap had no predecessor to measure from, so
        # its before-step number is absent rather than zero. True only for the
        # run's first step, and reporting 0.00 there reads as "the boundary is
        # free" -- which is exactly the wrong conclusion to hand someone.
        self._acc = {"n": 0, "gpu": 0.0, "step": 0.0, "mini": 0.0, "interior": 0.0,
                     "partial": self._prev_end is None}

    def new_step(self):
        """Called once per update_policy.

        Two jobs. The step-boundary gap gets judged against other step-boundary
        gaps rather than against a forward. And the step that just ended gets a
        one-line summary, which is the half of this the outlier detector cannot
        do: an outlier needs the loss to be concentrated in one micro-batch, and
        the same seconds spread thinly over sixty-six of them is invisible to it
        while being exactly as expensive. The summary is a running total, so it
        sees both.
        """
        self._drain()
        if self._acc["n"]:
            a = self._acc
            idle = a["step"] + a["mini"] + a["interior"]
            span = a["gpu"] + idle
            before_step = "n/a" if a["partial"] else f"{a['step'] / 1e3:.2f}"
            self._say(
                f"[step-gpu] rank {_rank()} step {self._step}: {a['n']} micro-batches, "
                f"busy {a['gpu'] / 1e3:.1f} s, idle {idle / 1e3:.2f} s "
                f"(before-step {before_step}, before-mini {a['mini'] / 1e3:.2f}, "
                f"interior {a['interior'] / 1e3:.2f}) = {100 * idle / span:.2f}% of "
                f"{span / 1e3:.1f} s")
        self._reset_acc()
        self._step += 1
        self._new_step = True

    @staticmethod
    def _open_log():
        try:
            os.makedirs(_STALL_DIR, exist_ok=True)
            path = os.path.join(_STALL_DIR, f"rank{_rank()}_pid{os.getpid()}.log")
            handle = open(path, "a", buffering=1)      # line-buffered: readable live
            print(f"[stall-watch] rank {_rank()} logging to {path}", flush=True)
            return handle
        except OSError as exc:      # a diagnostic must never take the run down
            print(f"[stall-watch] no file log: {exc!r}", flush=True)
            return None

    def _say(self, line):
        print(line, flush=True)
        if self._log is not None:
            try:
                self._log.write(line + "\n")
            except (OSError, ValueError):
                self._log = None

    @staticmethod
    def _median(values):
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def open(self, counter, index):
        kind = "step" if self._new_step else ("mini" if index == 0 else "interior")
        self._new_step = False
        event = self._cuda.Event(enable_timing=True)
        event.record()
        return [counter, kind, event, None, time.time(), None]

    def close(self, entry):
        entry[3] = self._cuda.Event(enable_timing=True)
        entry[3].record()
        entry[5] = time.time()
        self._pending.append(entry)
        self._drain()

    def _drain(self):
        """Read back whatever has finished. Never waits: an event that is not
        ready yet is simply left for the next micro-batch to pick up."""
        while self._pending and self._pending[0][3].query():
            counter, kind, start_ev, end_ev, host_start, host_end = self._pending.pop(0)
            gpu_ms = start_ev.elapsed_time(end_ev)
            gap_ms = self._prev_end.elapsed_time(start_ev) if self._prev_end is not None else 0.0
            self._prev_end = end_ev
            self._judge(counter, kind, gpu_ms, gap_ms, (host_end - host_start) * 1e3, host_end)
            self._acc["n"] += 1
            self._acc["gpu"] += gpu_ms
            self._acc[kind] += gap_ms
            self._recent.append(gpu_ms)
            self._recent_gap[kind].append(gap_ms)
            del self._recent[: -self._KEEP]
            del self._recent_gap[kind][: -self._KEEP]

    _WARMUP = 8         # micro-batches to establish a baseline before judging

    def _judge(self, counter, kind, gpu_ms, gap_ms, host_ms, at):
        # The first micro-batches of a run are all outliers -- allocator growth,
        # cudnn autotune, the first all-gather -- and reporting them buries the
        # rare mid-run event this exists to catch under startup noise.
        if len(self._recent) < self._WARMUP:
            return
        # Against the running median rather than a fixed threshold: a micro-batch
        # is ~1 s here but that is a property of this batch size and this model,
        # and a number baked in for one arm is a number that silently reports
        # everything or nothing on the next.
        median = self._median(self._recent)
        peers = self._recent_gap[kind]
        median_gap = self._median(peers)
        slow = median and gpu_ms > _STALL_FACTOR * median and gpu_ms - median > _STALL_MIN_MS
        # A step-boundary gap is compared with other step-boundary gaps, and only
        # once there are enough of them to have a baseline -- otherwise the first
        # of each kind is reported for being the first of its kind.
        gapped = (len(peers) >= self._WARMUP and gap_ms > _STALL_FACTOR * median_gap
                  and gap_ms - median_gap > _STALL_MIN_MS)
        if not (slow or gapped):
            return
        self._reported += 1
        self._say(
            f"[stall] rank {_rank()} micro {counter} at {at:.3f}: "
            f"gpu {gpu_ms:.0f} ms (median {median:.0f}), "
            f"gap/{kind} {gap_ms:.0f} ms (median {median_gap:.0f}), "
            f"host {host_ms:.0f} ms -> "
            f"{'BEFORE this micro-batch' if gapped else 'inside the micro-batch'}, "
            f"{'host blocked too' if host_ms > 0.8 * gpu_ms else 'host ran ahead'}")


def new_step():
    """Tell the watch a new update_policy has begun.

    Free and safe to call when nothing is watching; without it the gap that
    spans two steps is judged against the gaps inside one, which flags it every
    step for being structurally larger.
    """
    if _watch is not None:
        _watch.new_step()


def _watch_for(module):
    """One watch per process, built lazily so a CPU-only import stays free."""
    global _watch
    if _watch is None:
        _watch = _StallWatch(module.cuda)
    return _watch


@contextmanager
def phase(name: str):
    """Name a stretch of the timeline, so a gap can be attributed to a phase.

    Both backends need naming and neither reads the other's: NVTX ranges do not
    reach a Chrome trace, and ``record_function`` does not reach an Nsight
    report. So whichever is running gets its own marker, and with both on the
    two timelines carry the same names and can be laid side by side.
    """
    if not enabled():
        yield
        return
    import torch

    if nsys_enabled():
        torch.cuda.nvtx.range_push(name)
    try:
        if _torch_prof is not None:
            with torch.profiler.record_function(name):
                yield
        else:
            yield
    finally:
        if nsys_enabled():
            torch.cuda.nvtx.range_pop()


@contextmanager
def micro_batch(index: int):
    """One micro-batch: the stall watch, and each capture backend's start/stop.

    The counter is per process and spans mini-batches, so SKIP and MICRO are
    counted in micro-batches of the step rather than in anything the caller has
    to track. ``index`` is the position within the current mini-batch and only
    labels the range. The two backends share the counter but not their
    thresholds, so they can cover the same window or different ones.

    The stall watch is independent of both and on by default -- it is what runs
    for the whole 300 steps, and the capture backends are what get aimed once it
    has said where to aim them.
    """
    global _seen, _nsys_running, _nsys_finished
    if not enabled() and not stall_watch_enabled():
        yield
        return
    import torch

    watching = stall_watch_enabled() and torch.cuda.is_available()
    if enabled():
        if nsys_enabled() and not _nsys_running and not _nsys_finished and _seen >= _NSYS_SKIP:
            torch.cuda.profiler.start()
            _nsys_running = True
            print(f"[nsys] capture started at micro-batch {_seen} (pid {os.getpid()})", flush=True)
        if torch_enabled() and _torch_prof is None and not _torch_finished and _seen >= _TORCH_SKIP:
            _torch_start()

    started_at = _seen
    _seen += 1
    label = f"micro/{started_at}/{index}"
    entry = _watch_for(torch).open(started_at, index) if watching else None
    if _nsys_running:
        torch.cuda.nvtx.range_push(label)
    try:
        if _torch_prof is not None:
            with torch.profiler.record_function(label):
                yield
        else:
            yield
    finally:
        if entry is not None:
            # Before the capture bookkeeping, so a stopped capture's report write
            # does not land inside the micro-batch this is timing.
            _watch_for(torch).close(entry)
        if _nsys_running:
            torch.cuda.nvtx.range_pop()
            if _seen >= _NSYS_SKIP + _NSYS_MICRO:
                torch.cuda.profiler.stop()
                _nsys_running = False
                _nsys_finished = True
                print(f"[nsys] capture stopped after {_NSYS_MICRO} micro-batches "
                      f"(pid {os.getpid()})", flush=True)
        if _torch_prof is not None and _seen >= _TORCH_SKIP + _TORCH_MICRO:
            _torch_stop()


def iter_micro_batches(micro_batches):
    """``enumerate(micro_batches)`` with each iteration inside its own window.

    Wrapping the iterator rather than the loop body is what keeps this a
    one-line change at the call site: a generator's ``with`` block spans from
    the ``yield`` until the consumer asks for the next item, which is exactly
    one iteration of the caller's loop. A ``break`` or an exception closes the
    generator, so the range is popped and the capture closed either way.
    """
    for index, micro_batch_data in enumerate(micro_batches):
        with micro_batch(index):
            yield index, micro_batch_data
