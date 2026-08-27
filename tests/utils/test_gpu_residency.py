"""The instrument that says how many cards had work at once.

Two instruments already existed and both missed this, in opposite directions:
``[val-pipeline]`` counts a slot blocked in ``env.step`` as running while every
GPU is empty, and ``genGPU%`` only looks inside ``generate`` so it cannot see a
gap between generates at all. Between them they produced a run reported as
"NOTHING running 0.1%" in which NVML saw 285 s of node-wide idle.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl.utils import gpu_profiler as gp  # noqa: E402


class _FakeSampler:
    """Just enough of Sampler for residency_between: a lock and a util trace."""

    def __init__(self, trace, cpu=None, act=None):
        import threading

        self._lock = threading.Lock()
        self._util_trace = list(trace)
        self._cpu_trace = list(cpu or [])
        self._act_trace = list(act or [])

    residency_between = gp._Sampler.residency_between


def _trace(rows, interval=0.3, t0=100.0):
    return [(t0 + i * interval, list(vals)) for i, vals in enumerate(rows)]


def test_counts_how_many_gpus_were_busy():
    sampler = _FakeSampler(_trace([
        [90, 90, 90],   # 3 busy
        [90, 90, 90],
        [90, 0, 0],     # 1 busy
        [0, 0, 0],      # empty
    ]))
    res = sampler.residency_between(0, 1e9)
    assert res["n_gpus"] == 3
    assert res["samples"] == 4
    assert res["counts"] == {0: 1, 1: 1, 2: 0, 3: 2}
    assert res["pct"][3] == 50.0 and res["pct"][0] == 25.0


def test_the_empty_share_is_the_number_that_matters():
    """Zero cards busy is recoverable by another batch; three-at-87% is not."""
    sampler = _FakeSampler(_trace([[0, 0, 0]] * 3 + [[88, 88, 88]] * 7))
    res = sampler.residency_between(0, 1e9)
    assert res["pct"][0] == pytest.approx(30.0)
    assert res["pct"][3] == pytest.approx(70.0)


def test_a_partly_busy_sample_is_not_counted_as_empty():
    """One card working is a load-balance problem, not an idle node.

    Lumping it with zero would say "another batch would fill this", which is the
    wrong prescription: the work is there, it is on one rank.
    """
    sampler = _FakeSampler(_trace([[89, 0, 0], [0, 85, 0]]))
    res = sampler.residency_between(0, 1e9)
    assert res["counts"][0] == 0
    assert res["counts"][1] == 2


def test_the_busy_threshold_is_the_profilers_idle_threshold(monkeypatch):
    monkeypatch.setattr(gp, "_IDLE_THRESH", 30.0)
    sampler = _FakeSampler(_trace([[29, 31, 100]]))
    assert sampler.residency_between(0, 1e9)["counts"][2] == 1
    assert sampler.residency_between(0, 1e9, busy_thresh=99)["counts"][1] == 1


def test_a_sampler_gap_is_not_charged_as_idle_wall():
    """A stopped sampler leaves a hole; charging the whole hole invents idle time."""
    trace = _trace([[0, 0, 0], [0, 0, 0]], interval=0.3)
    trace.append((trace[-1][0] + 600.0, [0, 0, 0]))  # sampler was off for 10 min
    sampler = _FakeSampler(trace)
    res = sampler.residency_between(0, 1e9)
    assert res["counts"][0] == 3
    assert res["wall"] < 5.0, f"a 600 s hole was charged as idle: {res['wall']}"


def test_a_window_with_no_samples_returns_none():
    """Silence must be distinguishable from "the GPU was never idle"."""
    assert _FakeSampler(_trace([[90, 90, 90]])).residency_between(0, 1) is None


def test_none_entries_do_not_count_as_busy():
    """A card whose read failed is unknown, not working."""
    sampler = _FakeSampler(_trace([[90, None, None]]))
    assert sampler.residency_between(0, 1e9)["counts"][1] == 1


def test_the_line_names_the_empty_share():
    sampler = _FakeSampler(_trace([[0, 0, 0]] * 2 + [[90, 90, 90]] * 8))
    line = gp.format_residency(sampler.residency_between(0, 1e9))
    assert "[gpu-residency]" in line
    assert "EMPTY 20.0%" in line, line
    assert "3gpu 80.0%" in line, line


def test_formatting_nothing_is_empty_not_a_crash():
    assert gp.format_residency(None) == ""


def test_the_module_wrapper_is_quiet_when_profiling_is_off(monkeypatch):
    monkeypatch.setattr(gp, "_sampler", None)
    assert gp.residency_between(0, 1e9) is None


# --------------------------------------------------------------------------- #
# the pipeline prints it next to the slot-coverage line it was misread against
# --------------------------------------------------------------------------- #
def _live_profiler():
    """The gpu_profiler object the code under test will actually import.

    test_gpu_profiler_trace.py pops verl.utils.gpu_profiler out of sys.modules
    and re-imports it, so this file's module-level ``gp`` can be a STALE object
    by the time these run. Patching the stale one and calling code that imports
    the fresh one leaves the fake sampler unreachable -- and residency_between
    then returns None, which formats to "", which reads exactly like "the GPU
    was never idle". Resolve it at call time.
    """
    import importlib

    return importlib.import_module("verl.utils.gpu_profiler")


def test_the_pipeline_reads_residency_over_the_span_its_launches_covered(monkeypatch):
    """Spans are perf_counter, the sampler stamps monotonic.

    Those are the same clock on Linux and not everywhere, and getting it wrong
    returns an empty window rather than an error -- silence that reads exactly
    like "the GPU was never idle". The offset conversion is the thing under test.
    """
    import threading
    import time

    from verl.utils import val_pipeline as vp

    live = _live_profiler()

    class _LiveFakeSampler:
        residency_between = live._Sampler.residency_between

        def __init__(self, trace, cpu=None, act=None):
            self._lock = threading.Lock()
            self._util_trace = list(trace)
            self._cpu_trace = list(cpu or [])
            self._act_trace = list(act or [])

    start = time.perf_counter()
    end = start + 3.0
    # Stamped on the sampler's clock, which is what the conversion has to hit.
    offset = time.monotonic() - time.perf_counter()
    trace = _trace([[0, 0, 0]] * 2 + [[90, 90, 90]] * 8, interval=0.3, t0=start + offset)
    monkeypatch.setattr(live, "_sampler", _LiveFakeSampler(trace))
    monkeypatch.setattr(live, "enabled", lambda: True)
    # Pinned rather than inherited. These are module globals read at call time,
    # and test_gpu_profiler_trace.py pops gpu_profiler out of sys.modules and
    # re-imports it under GPU_PROFILER_INTERVAL=0.02 -- monkeypatch restores the
    # env var but not the module that already baked it in. Which module ran
    # first then decides whether this test's percentages hold, and it failed
    # only in the full three-directory suite because of it.
    monkeypatch.setattr(live, "_IDLE_THRESH", 30.0)
    monkeypatch.setattr(live, "_INTERVAL", 0.3)

    line = vp._residency_over([(start, end)])
    assert "[gpu-residency]" in line, line
    assert "EMPTY 20.0%" in line, line
    assert "3gpu 80.0%" in line, line


def test_the_pipeline_says_nothing_extra_when_profiling_is_off(capsys, monkeypatch):
    from verl.utils import val_pipeline as vp

    monkeypatch.setattr(_live_profiler(), "_sampler", None)
    monkeypatch.setattr(vp, "_REPORT_EVERY", 1)
    slots = [vp.Slot("a", envs=None, collector=None), vp.Slot("b", envs=None, collector=None)]
    list(vp.run_pipelined([1, 2], prepare=lambda x: x, task_of=lambda x: None,
                          launch=lambda prepared, slot: prepared, slots=slots))
    out = capsys.readouterr().out
    assert "[val-pipeline]" in out
    assert "[gpu-residency]" not in out, "an unmeasured residency must not print a made-up one"


# --------------------------------------------------------------------------- #
# partial != empty, and the line has to say which
# --------------------------------------------------------------------------- #
def test_the_line_separates_empty_from_partial():
    """They need different fixes, so one number for both prescribes the wrong one.

    EMPTY is every slot outside the GPU: another batch in flight fills it, and
    the empty share falls as p^depth. PARTIAL is a rank that finished its chunk
    of a collective generate and is waiting for the slowest rank -- depth cannot
    fill that, because the worker group runs one call at a time.
    """
    sampler = _FakeSampler(_trace([[0, 0, 0]] * 2 + [[90, 0, 0]] * 2 + [[90, 90, 90]] * 16))
    line = gp.format_residency(sampler.residency_between(0, 1e9))
    assert "EMPTY 10.0%" in line, line
    assert "PARTIAL 10.0%" in line, line


def test_the_line_reports_the_per_gpu_spread():
    """A lopsided column is a rank idling, not the engine's duty cycle."""
    sampler = _FakeSampler(_trace([[90, 30, 30]] * 10))
    res = sampler.residency_between(0, 1e9)
    assert res["per_gpu"] == pytest.approx([90.0, 30.0, 30.0])
    line = gp.format_residency(res)
    assert "per-gpu 90 30 30" in line, line
    assert "spread 60 pt" in line, line


def test_an_unreadable_card_does_not_fake_a_spread():
    sampler = _FakeSampler(_trace([[90, None, 90]] * 4))
    res = sampler.residency_between(0, 1e9)
    assert res["per_gpu"][1] is None
    assert "spread 0 pt" in gp.format_residency(res)


def test_a_single_gpu_box_prints_no_spread():
    sampler = _FakeSampler(_trace([[90]] * 4))
    assert "per-gpu" not in gp.format_residency(sampler.residency_between(0, 1e9))


# --------------------------------------------------------------------------- #
# WHY the cards were empty: the two causes look identical on the GPU
# --------------------------------------------------------------------------- #
def _cpu(trace, pct):
    """Driver cpu_pct stamped on the same instants as the util trace."""
    return [(ts, pct(vals)) for ts, vals in trace]


def test_cpu_alone_no_longer_names_a_cause():
    """WITHDRAWN: this used to read cpu >= 60% as "Python holding the GIL".

    It cannot. cpu_pct says the process was running, not that it was running
    Python under the GIL -- the Rust tokeniser releases it and still reads
    busy, and native BLAS reads as several cores. The number is still printed,
    because "blocked" versus "running" is real evidence, but the cause now
    comes from the activity census (which slots were in which phase), which is
    a direct observation rather than an inference from a rate.
    """
    trace = _trace([[0, 0, 0]] * 4 + [[90, 90, 90]] * 6)
    cpu = _cpu(trace, lambda v: 190.0 if max(v) == 0 else 40.0)
    line = gp.format_residency(_FakeSampler(trace, cpu).residency_between(0, 1e9))
    assert "driver CPU 190% of one core while EMPTY" in line, line
    assert "(running during EMPTY)" in line, line
    assert "EMPTY is" not in line, "with no census there is no cause to name"


def test_a_blocked_process_is_reported_as_blocked():
    """A process waiting on a socket burns no CPU. That much cpu_pct can say.

    MEASURED on the pump run's deep-idle samples: GPU 1.5%, memory controller
    0.8%, power 88 W against 288 W, host CPU and thread count indistinguishable
    from a busy sample, disk at zero. Nothing anywhere was working.
    """
    trace = _trace([[0, 0, 0]] * 4 + [[90, 90, 90]] * 6)
    cpu = _cpu(trace, lambda v: 3.0 if max(v) == 0 else 150.0)
    line = gp.format_residency(_FakeSampler(trace, cpu).residency_between(0, 1e9))
    assert "(blocked during EMPTY)" in line, line


def test_no_cpu_samples_means_no_verdict_rather_than_a_guess():
    """psutil may be absent. Silence beats naming a cause from no evidence."""
    trace = _trace([[0, 0, 0]] * 2 + [[90, 90, 90]] * 8)
    line = gp.format_residency(_FakeSampler(trace).residency_between(0, 1e9))
    assert "driver CPU" not in line, line
    assert "EMPTY 20.0%" in line, "the rest of the line still has to print"


def test_partial_samples_are_charged_to_neither_side():
    """One card working is a third case; averaging it into either verdict lies."""
    trace = _trace([[0, 0, 0]] * 2 + [[90, 0, 0]] * 2 + [[90, 90, 90]] * 6)
    cpu = _cpu(trace, lambda v: 5.0 if max(v) == 0 else (999.0 if v[1] == 0 else 100.0))
    res = _FakeSampler(trace, cpu).residency_between(0, 1e9)
    assert res["cpu_when_empty"] == pytest.approx(5.0)
    assert res["cpu_when_busy"] == pytest.approx(100.0), "the 1-busy samples must not leak in"


# --------------------------------------------------------------------------- #
# the activity census: what the slots were DOING when the cards were empty
# --------------------------------------------------------------------------- #
def test_the_census_counts_threads_not_a_stack():
    """Three pipeline slots run concurrently; a stack reports the last mover.

    push_phase/pop_phase keep one list, which is right for a single training
    thread and meaningless here -- and the question that decides the remaining
    idle is "how many slots were in each activity at once", which a stack
    cannot answer at all.
    """
    with gp.activity("envstep"):
        with gp.activity("envstep"):
            with gp.activity("preproc"):
                assert gp.activity_snapshot() == {"envstep": 2, "preproc": 1}
        assert gp.activity_snapshot() == {"envstep": 1}
    assert gp.activity_snapshot() == {}


def test_the_census_survives_an_exception():
    """A leaked count would make every later sample report a phantom slot."""
    try:
        with gp.activity("envstep"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gp.activity_snapshot() == {}


def test_concurrent_threads_are_counted_together():
    import threading

    started, release = threading.Barrier(4), threading.Event()
    seen = []

    def worker():
        with gp.activity("envstep"):
            started.wait(5)
            release.wait(5)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    started.wait(5)
    seen.append(gp.activity_snapshot())
    release.set()
    for t in threads:
        t.join(5)
    assert seen[0] == {"envstep": 3}
    assert gp.activity_snapshot() == {}


def _act(trace, fn):
    return [(ts, fn(vals)) for ts, vals in trace]


def test_all_slots_in_envstep_is_named_the_environment():
    trace = _trace([[0, 0, 0]] * 3 + [[90, 90, 90]] * 7)
    act = _act(trace, lambda v: {"envstep": 3} if max(v) == 0 else {"gen": 2, "envstep": 1})
    line = gp.format_residency(_FakeSampler(trace, act=act).residency_between(0, 1e9))
    assert "while EMPTY the slots were in: envstep 3.0" in line, line
    assert "EMPTY is the ENVIRONMENT" in line, line


def test_all_slots_in_preproc_is_named_the_driver():
    trace = _trace([[0, 0, 0]] * 3 + [[90, 90, 90]] * 7)
    act = _act(trace, lambda v: {"preproc": 3} if max(v) == 0 else {"gen": 3})
    line = gp.format_residency(_FakeSampler(trace, act=act).residency_between(0, 1e9))
    assert "EMPTY is the DRIVER's own Python (tokenising)" in line, line


def test_a_split_census_refuses_to_name_one():
    """Half in envstep and half in preproc is not evidence for either."""
    trace = _trace([[0, 0, 0]] * 3 + [[90, 90, 90]] * 7)
    act = _act(trace, lambda v: {"envstep": 0.4, "preproc": 0.4} if max(v) == 0 else {"gen": 3})
    line = gp.format_residency(_FakeSampler(trace, act=act).residency_between(0, 1e9))
    assert "no single activity dominates" in line, line


def test_cpu_is_reported_but_no_longer_used_as_the_verdict():
    """A driver using CPU is not the same claim as a driver holding the GIL.

    A Rust tokeniser releases it and still reads busy; native BLAS reads as
    several cores. The number stays, the conclusion moves to the census.
    """
    trace = _trace([[0, 0, 0]] * 3 + [[90, 90, 90]] * 7)
    cpu = [(ts, 4.0 if max(vals) == 0 else 160.0) for ts, vals in trace]
    line = gp.format_residency(_FakeSampler(trace, cpu=cpu).residency_between(0, 1e9))
    assert "driver CPU 4% of one core while EMPTY" in line, line
    assert "(blocked during EMPTY)" in line, line
    assert "EMPTY is Python on the driver" not in line, "cpu alone must not name a cause"


# --------------------------------------------------------------------------- #
# the duty cycle: the only number an engine setting can move
# --------------------------------------------------------------------------- #
def test_the_duty_cycle_is_measured_over_all_cards_busy_only():
    """Node util cannot stand in for it.

    Node util moves whenever EMPTY or PARTIAL move, and multi-step scheduling
    and async_scheduling touch neither -- they hide the engine's own per-step
    host work, which only shows in the samples where every card already had
    something to do. Reading node util for that experiment reads the wrong thing.
    """
    trace = _trace([[0, 0, 0]] * 4 + [[89, 90, 88]] * 6)
    res = _FakeSampler(trace).residency_between(0, 1e9)
    assert res["util_when_busy"] == pytest.approx(89.0)
    assert res["util_when_busy"] > 80.0, "the empty samples must not drag it down"


def test_partial_samples_are_excluded_from_the_duty_cycle():
    """A card at zero inside a partial sample is a scheduling gap, not duty."""
    trace = _trace([[90, 0, 0]] * 5 + [[90, 90, 90]] * 5)
    assert _FakeSampler(trace).residency_between(0, 1e9)["util_when_busy"] == pytest.approx(90.0)


def test_the_line_names_it_as_the_engine_setting_target():
    trace = _trace([[0, 0, 0]] * 2 + [[92, 92, 92]] * 8)
    line = gp.format_residency(_FakeSampler(trace).residency_between(0, 1e9))
    assert "reading 92.0%" in line, line
    assert "engine's duty cycle" in line, line


def test_no_fully_busy_sample_means_no_duty_cycle_rather_than_zero():
    """Early in a run there may be none, and 0.0% would read as a catastrophe."""
    trace = _trace([[90, 0, 0]] * 4)
    res = _FakeSampler(trace).residency_between(0, 1e9)
    assert res["util_when_busy"] is None
    assert "reading" not in gp.format_residency(res)
