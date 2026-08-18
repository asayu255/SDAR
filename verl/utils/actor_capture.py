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

_TORCH_MICRO = _int_env("ACTOR_TORCH_MICRO", 0)
_TORCH_SKIP = _int_env("ACTOR_TORCH_SKIP", 40)
_TORCH_DIR = os.environ.get("ACTOR_TORCH_DIR", "/tmp/actor_trace")

_seen = 0
_nsys_running = False
_nsys_finished = False
_torch_prof = None
_torch_finished = False


def nsys_enabled() -> bool:
    return _NSYS_MICRO > 0


def torch_enabled() -> bool:
    return _TORCH_MICRO > 0


def enabled() -> bool:
    return nsys_enabled() or torch_enabled()


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
    """One micro-batch, and each backend's start/stop.

    The counter is per process and spans mini-batches, so SKIP and MICRO are
    counted in micro-batches of the step rather than in anything the caller has
    to track. ``index`` is the position within the current mini-batch and only
    labels the range. The two backends share the counter but not their
    thresholds, so they can cover the same window or different ones.
    """
    global _seen, _nsys_running, _nsys_finished
    if not enabled():
        yield
        return
    import torch

    if nsys_enabled() and not _nsys_running and not _nsys_finished and _seen >= _NSYS_SKIP:
        torch.cuda.profiler.start()
        _nsys_running = True
        print(f"[nsys] capture started at micro-batch {_seen} (pid {os.getpid()})", flush=True)
    if torch_enabled() and _torch_prof is None and not _torch_finished and _seen >= _TORCH_SKIP:
        _torch_start()

    started_at = _seen
    _seen += 1
    label = f"micro/{started_at}/{index}"
    if _nsys_running:
        torch.cuda.nvtx.range_push(label)
    try:
        if _torch_prof is not None:
            with torch.profiler.record_function(label):
                yield
        else:
            yield
    finally:
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
