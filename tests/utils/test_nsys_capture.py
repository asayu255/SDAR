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
"""The Nsight window: off by default, and exactly where it says it is.

A capture that opens on the wrong micro-batch or never closes turns a 300 s
step into an unusable multi-gigabyte report, and one that quietly does nothing
costs a whole run to discover. Both are cheap to pin down here; no GPU is
needed, since torch's profiler and NVTX calls are the only device-touching part
and they are recorded through a stub.
"""

import importlib
import sys

import pytest


def _fresh(monkeypatch, **env):
    for key in ("ACTOR_NSYS_MICRO", "ACTOR_NSYS_SKIP", "ACTOR_NSYS_TRACE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("verl.utils.nsys_capture", None)
    return importlib.import_module("verl.utils.nsys_capture")


class _Calls:
    """Stands in for torch.cuda.{profiler,nvtx}, recording the call order."""

    def __init__(self):
        self.log = []

    def install(self, monkeypatch, mod):
        import types

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
        monkeypatch.setitem(sys.modules, "torch", torch)
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
