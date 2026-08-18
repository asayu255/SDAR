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
"""The capture window: off by default, and exactly where it says it is.

A capture that opens on the wrong micro-batch or never closes turns a 300 s
step into an unusable multi-gigabyte report, and one that quietly does nothing
costs a whole run to discover. Both are cheap to pin down here; no GPU is
needed, since the profiler and NVTX calls are the only device-touching part and
they are recorded through a stub.

Two backends share the window. The pair that matters most is the pair a test
can check without hardware: turning the torch one on must not drag the workers
under Nsight, and turning the Nsight one on must not start a torch profiler --
either mistake costs a run to notice and the ranks are Ray actors, so neither
shows up in the launcher's log.
"""

import importlib
import os
import sys

import pytest


def _fresh(monkeypatch, **env):
    for key in ("ACTOR_NSYS_MICRO", "ACTOR_NSYS_SKIP", "ACTOR_NSYS_TRACE",
                "ACTOR_TORCH_MICRO", "ACTOR_TORCH_SKIP", "ACTOR_TORCH_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("verl.utils.actor_capture", None)
    return importlib.import_module("verl.utils.actor_capture")


class _Calls:
    """Stands in for the parts of torch both backends touch, recording the order.

    The order is the whole assertion: "started before the first micro-batch it
    claims to cover" and "stopped after the last" are not visible in any single
    call, only in their sequence.
    """

    def __init__(self, rank=0, export_raises=None, profile_raises=None):
        self.log = []
        self.rank = rank
        self.exported = []
        self._export_raises = export_raises
        self._profile_raises = profile_raises

    def _profile(self, **kwargs):
        if self._profile_raises is not None:
            raise self._profile_raises
        self.log.append(f"torch.profile(record_shapes={kwargs.get('record_shapes')})")
        calls = self

        class _Prof:
            def start(self):
                calls.log.append("torch.start")

            def stop(self):
                calls.log.append("torch.stop")

            def export_chrome_trace(self, path):
                if calls._export_raises is not None:
                    raise calls._export_raises
                calls.log.append("torch.export")
                calls.exported.append(path)
                with open(path, "w") as handle:      # the caller stats it
                    handle.write("{}")

        return _Prof()

    def install(self, monkeypatch, mod):
        import types
        from contextlib import contextmanager

        @contextmanager
        def _record_function(name):
            self.log.append(f"rf:{name}")
            try:
                yield
            finally:
                self.log.append("rf-end")

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            profiler=types.SimpleNamespace(
                start=lambda: self.log.append("start"),
                stop=lambda: self.log.append("stop"),
            ),
            nvtx=types.SimpleNamespace(
                range_push=lambda name: self.log.append(f"push:{name}"),
                range_pop=lambda: self.log.append("pop"),
            ),
        )
        torch.profiler = types.SimpleNamespace(
            profile=self._profile,
            record_function=_record_function,
            ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
        )
        distributed = types.ModuleType("torch.distributed")
        distributed.is_available = lambda: True
        distributed.is_initialized = lambda: True
        distributed.get_rank = lambda: self.rank
        torch.distributed = distributed
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
        return self


def test_off_by_default(monkeypatch):
    mod = _fresh(monkeypatch)
    assert not mod.enabled()


def test_disabled_costs_nothing_and_touches_no_torch(monkeypatch):
    """The no-op path must not even import torch, so it is safe to leave in the
    hot loop of every rank."""
    mod = _fresh(monkeypatch)
    calls = _Calls().install(monkeypatch, mod)

    with mod.phase("actor.fwd"):
        pass
    for _ in mod.iter_micro_batches(range(50)):
        pass

    assert calls.log == []


def test_the_wrapper_is_enumerate(monkeypatch):
    """It replaces enumerate() at the call site, so it has to behave like it --
    including under a break, where the range still has to be popped."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="4", ACTOR_NSYS_SKIP="0")
    _Calls().install(monkeypatch, mod)
    items = ["a", "b", "c"]

    assert list(mod.iter_micro_batches(items)) == list(enumerate(items))

    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="4", ACTOR_NSYS_SKIP="0")
    calls = _Calls().install(monkeypatch, mod)
    for i, _ in mod.iter_micro_batches(items):
        if i == 0:
            break
    assert calls.log.count("push:micro/0/0") == 1
    assert "pop" in calls.log


def test_the_window_starts_after_skip_and_stops_after_micro(monkeypatch):
    """SKIP lets the allocator settle and the first mini-batch pay its warm-up;
    MICRO bounds the report. Counting is in micro-batches and spans mini-batches,
    so neither depends on where a step happens to begin."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="3", ACTOR_NSYS_SKIP="2")
    calls = _Calls().install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(10)):
        pass

    assert calls.log.count("start") == 1
    assert calls.log.count("stop") == 1
    starts = calls.log.index("start")
    stops = calls.log.index("stop")
    # nothing pushed before the window opens: the first two micro-batches are skipped
    assert "push:micro/0/0" not in calls.log
    assert "push:micro/1/1" not in calls.log
    # ...and the window covers exactly three
    pushed = [c for c in calls.log if c.startswith("push:micro/")]
    assert pushed == ["push:micro/2/2", "push:micro/3/3", "push:micro/4/4"]
    assert starts < calls.log.index(pushed[0])
    assert stops > calls.log.index(pushed[-1])


def test_the_window_never_reopens(monkeypatch):
    """update_policy is called once per step; a window that reopened would trace
    the whole run."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="2", ACTOR_NSYS_SKIP="0")
    calls = _Calls().install(monkeypatch, mod)

    for _ in range(5):                      # five steps' worth of micro-batches
        for _ in mod.iter_micro_batches(range(4)):
            pass

    assert calls.log.count("start") == 1
    assert calls.log.count("stop") == 1


def test_phases_are_named_on_the_timeline(monkeypatch):
    """Without NVTX the trace is anonymous kernels, so this is what makes a gap
    attributable to actor.fwd rather than to 'the step'."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="1", ACTOR_NSYS_SKIP="0")
    calls = _Calls().install(monkeypatch, mod)

    with mod.phase("actor.fwd"):
        pass

    assert calls.log == ["push:actor.fwd", "pop"]


def test_the_runtime_env_limits_the_report_to_the_capture_range(monkeypatch):
    """A full-run trace of a 300 s step is unusable; capture-range is what keeps
    it to the handful of micro-batches this module opens. nvtx has to be in the
    trace list or the phase names never reach the report."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="1")
    env = mod.nsight_runtime_env()

    assert env["capture-range"] == "cudaProfilerApi"
    # shutdown, so the report is written at the window's end and nothing after
    # it can reach the event stream
    assert env["capture-range-end"] == "stop-shutdown"
    assert "nvtx" in env["t"]
    assert "cuda" in env["t"]
    # This host cannot do CPU IP sampling ("not supported, disabling"), so osrt
    # is the only remaining way to see what the host was blocked on.
    assert "osrt" in env["t"]


def test_the_trace_list_can_drop_osrt_without_a_code_change(monkeypatch):
    """osrt is the suspect for the "Wrong event order" import failure, and
    ruling it in or out should not cost a commit and a pull."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="1",
                 ACTOR_NSYS_TRACE="cuda,cudnn,cublas,nvtx")

    assert mod.nsight_runtime_env()["t"] == "cuda,cudnn,cublas,nvtx"
    # the rest of the config is unchanged, so the two runs stay comparable
    assert mod.nsight_runtime_env()["capture-range-end"] == "stop-shutdown"


# --- the torch backend -------------------------------------------------------
#
# It exists because the Nsight one has a step outside the process: collection
# writes a .qdstrm and a separate QdstrmImporter turns it into a report. On this
# host that importer fails ("Wrong event order has been detected"), leaving three
# intact 39 MB captures that nothing can read. torch.profiler sees strictly less
# -- no driver, no OS runtime -- but it writes a finished Chrome trace from
# inside the rank, so there is no conversion left to fail.


def test_the_two_backends_are_independent(monkeypatch):
    """Turning on the torch one must not drag every worker under Nsight: the
    plugin is attached at ray.init off nsys_enabled(), and a worker launched
    under `nsys profile` that never opens a capture range pays the interception
    cost for an empty report."""
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="2")
    assert mod.torch_enabled()
    assert not mod.nsys_enabled()
    assert mod.enabled()

    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="2")
    assert mod.nsys_enabled()
    assert not mod.torch_enabled()
    assert mod.enabled()


def test_nsight_alone_never_starts_a_torch_profiler(monkeypatch):
    """CUPTI cannot be attached twice. A torch profiler started underneath an
    nsys session is at best duplicated overhead and at worst a failed capture,
    so the Nsight path has to leave torch's profiler untouched."""
    mod = _fresh(monkeypatch, ACTOR_NSYS_MICRO="2", ACTOR_NSYS_SKIP="0")
    calls = _Calls().install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(4)):
        pass

    assert "torch.profile(record_shapes=True)" not in calls.log
    assert not calls.exported


def test_the_torch_window_starts_after_skip_and_writes_one_trace(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="3", ACTOR_TORCH_SKIP="2",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls().install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(10)):
        pass

    assert calls.log.count("torch.start") == 1
    assert calls.log.count("torch.stop") == 1
    assert calls.log.count("torch.export") == 1
    # The record_function labels are the same strings as the NVTX ones, so a
    # torch trace and an Nsight report of the same run line up by name.
    labelled = [c for c in calls.log if c.startswith("rf:micro/")]
    assert labelled == ["rf:micro/2/2", "rf:micro/3/3", "rf:micro/4/4"]
    assert calls.log.index("torch.start") < calls.log.index("rf:micro/2/2")
    assert calls.log.index("torch.stop") > calls.log.index("rf:micro/4/4")
    # and no NVTX: nothing is listening for it
    assert not [c for c in calls.log if c.startswith("push:")]


def test_the_torch_trace_names_the_rank_and_the_pid(monkeypatch, tmp_path):
    """Every rank writes into one directory, so the files have to differ. The
    rank is the name the comparison is made in; the pid is what keeps a second
    run from overwriting the first."""
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="1", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls(rank=2).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(2)):
        pass

    assert len(calls.exported) == 1
    name = os.path.basename(calls.exported[0])
    assert name == f"actor_rank2_pid{os.getpid()}.json"


def test_the_torch_window_never_reopens(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="2", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls().install(monkeypatch, mod)

    for _ in range(5):
        for _ in mod.iter_micro_batches(range(4)):
            pass

    assert calls.log.count("torch.start") == 1
    assert calls.log.count("torch.export") == 1


def test_phases_are_named_for_the_torch_backend_too(monkeypatch, tmp_path):
    """NVTX ranges do not reach a Chrome trace. Without record_function the
    torch timeline is anonymous kernels and a gap cannot be pinned to actor.fwd
    -- which is the only reason to have opened it."""
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="2", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls().install(monkeypatch, mod)

    it = mod.iter_micro_batches(range(2))
    next(it)
    with mod.phase("actor.fwd"):
        pass

    assert "rf:actor.fwd" in calls.log


def test_a_profiler_that_refuses_to_start_does_not_take_the_run_down(monkeypatch, tmp_path):
    """CUPTI is the part of the stack most likely to refuse -- another profiler
    attached, a permissions setting, a driver mismatch. Losing the diagnostic is
    acceptable; losing the run to it is not."""
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="2", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls(profile_raises=RuntimeError("CUPTI_ERROR_NOT_INITIALIZED"))
    calls.install(monkeypatch, mod)

    items = list(mod.iter_micro_batches(range(6)))

    assert items == list(enumerate(range(6)))     # training carried on
    assert not calls.exported
    # and it gave up rather than retrying on every micro-batch for the run's life
    assert mod._torch_finished


def test_a_trace_that_cannot_be_written_does_not_take_the_run_down(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="1", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    calls = _Calls(export_raises=OSError("No space left on device"))
    calls.install(monkeypatch, mod)

    items = list(mod.iter_micro_batches(range(3)))

    assert items == list(enumerate(range(3)))
    assert mod._torch_prof is None                # released, not left recording
