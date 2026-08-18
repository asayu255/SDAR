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
"""A few micro-batches of every rank, under Nsight, with the phases named.

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
    identically. In an Nsight timeline the collectives appear by name
    (``ncclDevKernel_AllGather...``) with their durations, which settles it.
  * **Is it even a gap?** NVML reports "no kernel resident". That is equally
    consistent with an empty stream (the host stopped submitting -- a CPU-side
    stall), a kernel too short for the sampler to catch, and a wait on an event.
    Those want completely different fixes.

Everything here is a no-op unless ``ACTOR_NSYS_MICRO`` is set: no NVTX pushes, no
profiler calls, one bool per micro-batch. Unlike ``gpu_profiler``'s worker-side
stages this runs on EVERY rank, because comparing ranks is the entire point.

Env vars
--------
  ACTOR_NSYS_MICRO=20   capture this many micro-batches, then stop. Events run
                        about one per 5.5 micro-batches, so 20 should hold 3-4.
  ACTOR_NSYS_SKIP=40    micro-batches to let past first. The allocator settles
                        and the first mini-batch pays warm-up costs that are not
                        what is being chased.

How to run it
-------------
``torch.cuda.profiler.start()`` only opens a capture range; something has to be
attached to the process to honour it, and the ranks are Ray actors rather than
children of the launcher, so wrapping the launch command reaches nothing. Ray's
``_nsight`` runtime-env plugin wraps each worker instead, and
``main_sft_multitask`` fills it in when ACTOR_NSYS_MICRO is set. One report per
process lands in the Ray session's ``logs/nsight`` directory.
"""

import os
from contextlib import contextmanager

__all__ = ["enabled", "nsight_runtime_env", "phase", "micro_batch", "iter_micro_batches"]


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


_MICRO = _int_env("ACTOR_NSYS_MICRO", 0)
_SKIP = _int_env("ACTOR_NSYS_SKIP", 40)

_seen = 0
_running = False
_finished = False


def enabled() -> bool:
    return _MICRO > 0


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
        "t": "cuda,cudnn,cublas,nvtx,osrt",
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


@contextmanager
def phase(name: str):
    """Name a stretch of the timeline, so a gap can be attributed to a phase."""
    if not enabled():
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def micro_batch(index: int):
    """One micro-batch, and the window's start/stop.

    The counter is per process and spans mini-batches, so SKIP and MICRO are
    counted in micro-batches of the step rather than in anything the caller has
    to track. ``index`` is the position within the current mini-batch and only
    labels the range.
    """
    global _seen, _running, _finished
    if not enabled() or _finished:
        yield
        return
    import torch

    if not _running and _seen >= _SKIP:
        torch.cuda.profiler.start()
        _running = True
        print(f"[nsys] capture started at micro-batch {_seen} (pid {os.getpid()})", flush=True)

    started_at = _seen
    _seen += 1
    if _running:
        torch.cuda.nvtx.range_push(f"micro/{started_at}/{index}")
    try:
        yield
    finally:
        if _running:
            torch.cuda.nvtx.range_pop()
            if _seen >= _SKIP + _MICRO:
                torch.cuda.profiler.stop()
                _running = False
                _finished = True
                print(f"[nsys] capture stopped after {_MICRO} micro-batches "
                      f"(pid {os.getpid()})", flush=True)


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
