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
                "ACTOR_TORCH_MICRO", "ACTOR_TORCH_SKIP", "ACTOR_TORCH_DIR",
                "ACTOR_STALL_FACTOR", "ACTOR_STALL_MIN_MS", "ACTOR_STALL_DIR"):
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

    def __init__(self, rank=0, export_raises=None, profile_raises=None, cuda=False,
                 timings=(), gaps=(), ready=True, deltas=None):
        self.log = []
        self.cuda = cuda
        self.timings = list(timings)     # gpu_ms per micro-batch, in order
        self.gaps = list(gaps)           # device-idle ms before each micro-batch
        self.ready = ready               # what Event.query() answers
        # An explicit ms-per-record timeline, for tests whose event order is not
        # a plain open/close alternation -- a named span inside a gap records two
        # more events between one micro-batch's close and the next one's open.
        self.deltas = list(deltas) if deltas is not None else None
        self.events = []
        self.rank = rank
        self.exported = []
        self._export_raises = export_raises
        self._profile_raises = profile_raises

    def _clock(self):
        """Advance by the next requested micro-batch duration on every second
        record(), so open/close pairs come out as the caller specified."""
        if not hasattr(self, "_t"):
            self._t, self._i = 0.0, 0
        if self.deltas is not None:
            self._t += self.deltas[self._i] if self._i < len(self.deltas) else 0.0
            self._i += 1
            return self._t
        if len(self.events) % 2 == 1:                 # closing an open pair
            self._t += self.timings[self._i] if self._i < len(self.timings) else 1000.0
            self._i += 1
        else:
            self._t += self.gaps[self._i] if self._i < len(self.gaps) else 1.0
        return self._t

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

        calls = self

        class _Event:
            """torch.cuda.Event, with the timeline the test asked for.

            elapsed_time is driven off the recorded order rather than a clock, so
            a run of micro-batches with one slow one in it is exactly
            reproducible -- which is the only way to assert on a detector whose
            whole job is picking an outlier out of a distribution.
            """

            def __init__(self, enable_timing=False):
                self.at = None

            def record(self):
                self.at = calls._clock()
                calls.events.append(self)

            def query(self):
                return calls.ready

            def elapsed_time(self, other):
                return other.at - self.at

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: self.cuda,
            Event=_Event,
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


# --- the stall watch ---------------------------------------------------------
#
# The capture backends above are aimed instruments: they cover a fixed window,
# and the first Nsight capture proved what that costs. The dips being chased are
# five events in 68 minutes of training, several seconds long, at unpredictable
# positions inside their steps -- a window pinned to micro-batch 40 of step 1 has
# essentially no chance of holding one, and the capture itself became the largest
# stall in the run. So the watch is the opposite: cheap enough to leave on for
# the whole 300 steps, and it says where to aim the others.


def test_the_watch_is_on_by_default_and_the_captures_are_not(monkeypatch):
    """Inverted defaults, deliberately. Leaving a capture on costs a run; leaving
    the watch off costs the one event it existed to see."""
    mod = _fresh(monkeypatch)

    assert mod.stall_watch_enabled()
    assert not mod.enabled()


def test_it_can_be_turned_off(monkeypatch):
    mod = _fresh(monkeypatch, ACTOR_STALL_FACTOR="0")
    calls = _Calls(cuda=True).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(30)):
        pass

    assert calls.events == []          # not even an Event was constructed


def test_a_steady_run_reports_nothing(monkeypatch, capsys):
    """A detector that fires on ordinary jitter is a detector nobody reads."""
    mod = _fresh(monkeypatch)
    _Calls(cuda=True, timings=[1000, 1050, 980, 1010, 1100, 990] * 8).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(40)):
        pass

    assert "[stall]" not in capsys.readouterr().out


def test_the_first_micro_batches_are_not_reported(monkeypatch, capsys):
    """Warm-up is all outliers -- allocator growth, cudnn autotune, the first
    all-gather. Reporting those buries the rare mid-run event under startup."""
    mod = _fresh(monkeypatch)
    _Calls(cuda=True, timings=[9000, 8000, 7000] + [1000] * 30).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(20)):
        pass

    assert "[stall]" not in capsys.readouterr().out


def test_one_slow_micro_batch_is_found_in_a_run_of_normal_ones(monkeypatch, capsys):
    """The whole point: pick the outlier out of the distribution, without having
    been told in advance where in the run to look."""
    mod = _fresh(monkeypatch)
    timings = [1000] * 20
    timings[15] = 6000                        # the event
    _Calls(cuda=True, timings=timings).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(20)):
        pass

    out = [line for line in capsys.readouterr().out.splitlines() if "[stall]" in line]
    assert len(out) == 1, out
    assert "micro 15" in out[0]
    assert "gpu 6000 ms (median 1000)" in out[0]
    assert "inside the micro-batch" in out[0]


def test_a_gap_between_micro_batches_is_told_apart_from_a_slow_one(monkeypatch, capsys):
    """Different fixes. Inside the micro-batch points at the forward; between
    them points at the optimizer step, the gradient reduce, or the next batch's
    H2D -- and only the device timeline can tell them apart, since both look the
    same from the step's wall time."""
    mod = _fresh(monkeypatch)
    gaps = [1] * 20
    gaps[14] = 4000                           # device idle BEFORE micro 14
    _Calls(cuda=True, timings=[1000] * 20, gaps=gaps).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(20)):
        pass

    out = [line for line in capsys.readouterr().out.splitlines() if "[stall]" in line]
    assert len(out) == 1, out
    assert "gap/interior 4000 ms" in out[0]
    assert "BEFORE this micro-batch" in out[0]


def test_it_says_whether_the_host_was_blocked_too(monkeypatch, capsys):
    """The question the dips turn on. A host that ran ahead means the device
    itself was slow -- a collective wait. A host blocked for the same span means
    whatever stopped is upstream of the device entirely, and no amount of
    rebalancing GPU work will touch it."""
    mod = _fresh(monkeypatch)
    timings = [1000] * 20
    timings[15] = 6000
    _Calls(cuda=True, timings=timings).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(20)):
        pass

    # the stub's host clock is real time, so the body returns instantly: the host
    # ran far ahead of a 6 s device span, and the line has to say so
    assert "host ran ahead" in capsys.readouterr().out


def test_an_event_that_has_not_finished_is_left_alone(monkeypatch, capsys):
    """query() is used rather than synchronize() precisely so the watch cannot
    create the stall it is looking for. An unfinished event must therefore wait,
    not be forced."""
    mod = _fresh(monkeypatch)
    calls = _Calls(cuda=True, timings=[1000] * 10 + [9000] + [1000] * 10, ready=False)
    calls.install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(21)):
        pass

    assert "[stall]" not in capsys.readouterr().out     # nothing read back yet
    assert calls.events                                  # but the timing was recorded


def test_a_host_without_a_gpu_costs_nothing_and_does_not_crash(monkeypatch):
    """dp_actor's loop runs wherever the tests do."""
    mod = _fresh(monkeypatch)
    calls = _Calls(cuda=False).install(monkeypatch, mod)

    items = list(mod.iter_micro_batches(range(10)))

    assert items == list(enumerate(range(10)))
    assert calls.events == []


def test_the_watch_runs_even_with_a_capture_backend_on(monkeypatch, capsys, tmp_path):
    """They answer different questions and neither replaces the other: the watch
    says which micro-batch, the capture says what was in it."""
    mod = _fresh(monkeypatch, ACTOR_TORCH_MICRO="2", ACTOR_TORCH_SKIP="0",
                 ACTOR_TORCH_DIR=str(tmp_path))
    timings = [1000] * 20
    timings[15] = 6000
    _Calls(cuda=True, timings=timings).install(monkeypatch, mod)

    for _ in mod.iter_micro_batches(range(20)):
        pass

    assert "[stall]" in capsys.readouterr().out


def test_the_step_boundary_gap_is_not_reported_every_step(monkeypatch, capsys):
    """The hole this had before new_step().

    Gaps come from three structurally different places: inside a mini-batch,
    between mini-batches (the gradient reduce and optimizer step), and between
    steps (the return to the driver, its logging, the next batch's H2D). The
    last two are larger by construction and happen on a fixed schedule, so one
    shared median flags them every single time -- 300 lines a run with the five
    events that matter buried in them.
    """
    mod = _fresh(monkeypatch)
    # four "steps" of five micro-batches: a 900 ms boundary gap, then 1 ms ones
    gaps, timings = [], []
    for _ in range(8):
        gaps += [900, 1, 1, 1, 1]
        timings += [1000] * 5
    _Calls(cuda=True, timings=timings, gaps=gaps).install(monkeypatch, mod)

    for step in range(8):
        mod.new_step()
        for _ in mod.iter_micro_batches(range(5)):
            pass

    assert "[stall]" not in capsys.readouterr().out


def test_a_step_boundary_that_is_slow_FOR_A_BOUNDARY_still_reports(monkeypatch, capsys):
    """...and separating the populations must not blind it to a real one."""
    mod = _fresh(monkeypatch)
    gaps, timings = [], []
    for step in range(12):
        gaps += [12000 if step == 10 else 900, 1, 1, 1, 1]
        timings += [1000] * 5
    _Calls(cuda=True, timings=timings, gaps=gaps).install(monkeypatch, mod)

    for step in range(12):
        mod.new_step()
        for _ in mod.iter_micro_batches(range(5)):
            pass

    out = [l for l in capsys.readouterr().out.splitlines() if "[stall]" in l]
    assert len(out) == 1, out
    assert "gap/step 12000 ms (median 900)" in out[0]
    assert "BEFORE this micro-batch" in out[0]


def test_the_gradient_reduce_gap_is_its_own_population_too(monkeypatch, capsys):
    """With micro < mini there are two micro-batches per mini-batch, so every
    other gap holds the optimizer step. Judged together they are bimodal and the
    larger half is flagged forever."""
    mod = _fresh(monkeypatch)
    gaps, timings = [], []
    for _ in range(20):
        gaps += [700, 1]          # index 0 carries the reduce, index 1 does not
        timings += [1000, 1000]
    _Calls(cuda=True, timings=timings, gaps=gaps).install(monkeypatch, mod)

    mod.new_step()
    for _ in range(20):
        for _ in mod.iter_micro_batches(range(2)):
            pass

    assert "[stall]" not in capsys.readouterr().out


def test_new_step_is_free_when_nothing_is_watching(monkeypatch):
    """dp_actor calls it unconditionally at the top of update_policy."""
    mod = _fresh(monkeypatch, ACTOR_STALL_FACTOR="0")
    mod.new_step()          # must not raise with no watch built


def test_every_step_gets_a_summary_even_when_no_outlier_fires(monkeypatch, capsys):
    """The other half of the instrument, and the one that survives a loss that is
    not concentrated. An outlier detector needs the seconds to be in one
    micro-batch; the same seconds spread over sixty-six are exactly as expensive
    and trip nothing. The running total sees both, and splits the idle into the
    three places it can come from."""
    mod = _fresh(monkeypatch)
    gaps, timings = [], []
    for _ in range(3):
        gaps += [500, 100, 100, 100]          # 0.5 s boundary + 0.3 s interior
        timings += [1000] * 4
    _Calls(cuda=True, timings=timings, gaps=gaps).install(monkeypatch, mod)

    for _ in range(3):
        mod.new_step()
        for _ in mod.iter_micro_batches(range(4)):
            pass
    mod.new_step()                             # flushes the last step

    lines = [l for l in capsys.readouterr().out.splitlines() if "[step-gpu]" in l]
    assert len(lines) == 3, lines
    assert "4 micro-batches" in lines[-1]
    assert "in-micro 4.0 s" in lines[-1]
    assert "outside 0.80 s" in lines[-1]                      # 500 + 3x100
    assert "before-step 0.50" in lines[-1]
    assert "before-step n/a" in lines[0]        # the run's first step cannot know it
    assert "interior 0.30" in lines[-1]
    assert "16.67% idle of 4.8 s" in lines[-1]   # nothing named, so all of it


def test_a_loss_spread_over_every_micro_batch_shows_in_the_summary(monkeypatch, capsys):
    """The case that defeats the outlier detector by construction: no single
    micro-batch is anywhere near 3x the median, and the step still loses four
    seconds."""
    mod = _fresh(monkeypatch)
    _Calls(cuda=True, timings=[1000] * 40,
           gaps=[1] * 20 + [201] * 20).install(monkeypatch, mod)

    for step in range(2):
        mod.new_step()
        for _ in mod.iter_micro_batches(range(20)):
            pass
    mod.new_step()

    out = capsys.readouterr().out
    assert "[stall]" not in out                            # nothing is an outlier
    lines = [l for l in out.splitlines() if "[step-gpu]" in l]
    assert "outside 0.02 s" in lines[0]                    # ...and the two steps
    assert "outside 4.02 s" in lines[1]                    # are plainly different


def test_the_first_step_does_not_claim_the_boundary_was_free(monkeypatch, capsys):
    """The run's first micro-batch has no predecessor, so its before-step gap is
    0.0 by construction. Printing that as a number reads as "the step boundary
    costs nothing on the device" -- which is a conclusion, and the wrong one."""
    mod = _fresh(monkeypatch)
    _Calls(cuda=True, timings=[1000] * 12, gaps=[400] + [1] * 11).install(monkeypatch, mod)

    for _ in range(3):
        mod.new_step()
        for _ in mod.iter_micro_batches(range(4)):
            pass
    mod.new_step()

    lines = [l for l in capsys.readouterr().out.splitlines() if "[step-gpu]" in l]
    assert "before-step n/a" in lines[0]
    assert "before-step n/a" not in lines[1]     # and only the first


def test_every_line_also_reaches_a_per_rank_file(monkeypatch, tmp_path, capsys):
    """The console loses two ranks of three.

    Ray's log dedup matches on the message with numbers substituted, so the three
    ranks' [step-gpu] lines are one pattern to it: it prints one and collapses the
    others into "[repeated 2x across cluster]". Comparing ranks is the entire
    point of the instrument, so it cannot depend on RAY_DEDUP_LOGS being
    remembered on a command line.
    """
    mod = _fresh(monkeypatch, ACTOR_STALL_DIR=str(tmp_path))
    timings = [1000] * 20
    timings[15] = 6000
    _Calls(cuda=True, rank=2, timings=timings).install(monkeypatch, mod)

    mod.new_step()
    for _ in mod.iter_micro_batches(range(20)):
        pass
    mod.new_step()

    written = list(tmp_path.glob("rank2_pid*.log"))
    assert len(written) == 1, list(tmp_path.iterdir())
    body = written[0].read_text()
    assert "[stall] rank 2 micro 15" in body
    assert "[step-gpu] rank 2 step 0" in body
    # and the console still gets them, so nothing has to be tailed to notice
    out = capsys.readouterr().out
    assert "[stall] rank 2 micro 15" in out


def test_an_unwritable_log_directory_does_not_take_the_run_down(monkeypatch, tmp_path, capsys):
    mod = _fresh(monkeypatch, ACTOR_STALL_DIR=str(tmp_path / "f" / "nope"))
    (tmp_path / "f").write_text("not a directory")
    _Calls(cuda=True, timings=[1000] * 12).install(monkeypatch, mod)

    mod.new_step()
    items = list(mod.iter_micro_batches(range(12)))
    mod.new_step()

    assert items == list(enumerate(range(12)))
    assert "[step-gpu]" in capsys.readouterr().out      # console still works


def test_the_optimizer_step_is_not_counted_as_idle(monkeypatch, capsys):
    """The correction that decides what the baseline means.

    The window between two micro-batches holds the gradient reduce and the
    optimizer step, and on a 1.7B model sharded three ways that is 50-80 ms of
    real kernels per mini-batch -- 5.75 s of a 346 s step, reported as "idle".
    A stall then has to clear an invented 1.7% noise floor to look like anything.
    Naming the span subtracts it, and what is left is time the device had nothing
    to do.
    """
    mod = _fresh(monkeypatch)
    # per record(): micro opens (700 ms since the last close), micro runs 1000 ms,
    # the optimizer starts at once and runs 600 ms. Four mini-batches of one.
    calls = _Calls(cuda=True, deltas=[700, 1000, 0, 600] * 4)
    calls.install(monkeypatch, mod)

    mod.new_step()
    for _ in range(4):
        for _ in mod.iter_micro_batches(range(1)):
            pass
        with mod.span("optim"):          # after the micro-batch, as dp_actor does
            pass
    mod.new_step()

    line = [l for l in capsys.readouterr().out.splitlines() if "[step-gpu]" in l][0]
    assert "in-micro 4.0 s" in line                  # 4 x 1000 ms of forward/backward
    assert "outside 3.90 s" in line                  # 3 gaps of 600 + 700, plus the first
    assert "optim 2.40," in line                     # 4 x 600 ms, named
    assert "unaccounted 1.50 s" in line              # what the device really had nothing to do


def test_span_is_free_when_nothing_is_watching(monkeypatch):
    """dp_actor wraps the optimizer step unconditionally."""
    mod = _fresh(monkeypatch, ACTOR_STALL_FACTOR="0")
    with mod.span("optim"):
        pass
