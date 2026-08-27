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

    def __init__(self, trace, cpu=None, act=None, stacks=None):
        import threading

        self._lock = threading.Lock()
        self._util_trace = list(trace)
        self._cpu_trace = list(cpu or [])
        self._act_trace = list(act or [])
        self._stack_trace = list(stacks or [])

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
    # approx, not ==: the shares divide summed float seconds now, not sample
    # counts, so an exact half arrives as 49.999999999999993.
    assert res["pct"][3] == pytest.approx(50.0) and res["pct"][0] == pytest.approx(25.0)


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

        def __init__(self, trace, cpu=None, act=None, stacks=None):
            self._lock = threading.Lock()
            self._util_trace = list(trace)
            self._cpu_trace = list(cpu or [])
            self._act_trace = list(act or [])
            self._stack_trace = list(stacks or [])

    start = time.perf_counter()
    end = start + 3.0
    # Stamped on the sampler's clock, which is what the conversion has to hit.
    offset = time.monotonic() - time.perf_counter()
    # Started inside the window, not on its edge: _residency_over recomputes
    # the offset, and a sample stamped exactly at the boundary can land a
    # microsecond outside it and drop out of the count.
    trace = _trace([[0, 0, 0]] * 2 + [[90, 90, 90]] * 8, interval=0.3, t0=start + offset + 0.05)
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


def test_cpu_decides_whether_and_the_census_decides_where():
    """Two swings, and the rule that survived both.

    First version: cpu >= 60 meant "Python holding the GIL". Withdrawn -- cpu
    says the process was running, not that it was running Python under the GIL.
    Second version: the census alone named the cause. Withdrawn too -- it named
    the retriever on a run whose driver was burning 82% of a core.

    What each one can actually answer: cpu says WHETHER the driver was working,
    because a process blocked on a socket burns none; the census says WHERE,
    when a phase actually dominates the slots. The verdict needs both.
    """
    trace = _trace([[0, 0, 0]] * 4 + [[90, 90, 90]] * 6)
    running = _cpu(trace, lambda v: 190.0 if max(v) == 0 else 40.0)
    line = gp.format_residency(_FakeSampler(trace, running).residency_between(0, 1e9))
    assert "DRIVER RUNNING PYTHON" in line, line

    blocked = _cpu(trace, lambda v: 3.0 if max(v) == 0 else 150.0)
    line = gp.format_residency(_FakeSampler(trace, blocked).residency_between(0, 1e9))
    assert "WAIT OFF THE BOX" in line, line


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


# --------------------------------------------------------------------------- #
# the verdict is what the two readings AGREE on
# --------------------------------------------------------------------------- #
def _both(trace, cpu_empty, cpu_busy, act_empty, act_busy=None):
    cpu = [(ts, cpu_empty if max(v) == 0 else cpu_busy) for ts, v in trace]
    act = [(ts, dict(act_empty) if max(v) == 0 else dict(act_busy or {"gen": 3})) for ts, v in trace]
    return _FakeSampler(trace, cpu, act).residency_between(0, 1e9)


def test_the_real_run_that_broke_the_old_rule():
    """MEASURED, V0 + num_scheduler_steps=4, 2026-08-27:

        while EMPTY the slots were in: envstep 1.1, preproc 0.6, gen 0.5
          -> EMPTY is the ENVIRONMENT (retriever round trip)
        driver CPU 82% of one core while EMPTY vs 15% while busy

    The two lines contradict each other and the old rule believed the wrong
    one. A process waiting on a retriever burns no CPU; 82% of a core while the
    cards are empty, against 15% while they are busy, is the driver working.
    And 1.1 of three slots is a third, not a majority -- the old threshold
    compared it against the tagged total instead of the slot count.
    """
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    res = _both(trace, 82.0, 15.0, {"envstep": 1.1, "preproc": 0.6, "gen": 0.5})
    line = gp.format_residency(res)
    assert "DRIVER RUNNING PYTHON" in line, line
    assert "ENVIRONMENT" not in line, "the old rule named the retriever here"
    assert "no single phase dominates" in line, line
    assert "0.8 in no tagged phase" in line, "the unaccounted slots have to be visible"


def test_a_blocked_driver_with_envstep_on_top_is_the_environment():
    """Both readings agreeing is what a retriever verdict needs."""
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 3.0, 150.0, {"envstep": 2.8}))
    assert "WAIT OFF THE BOX" in line, line


def test_a_blocked_driver_whose_top_phase_is_not_envstep_says_they_disagree():
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 3.0, 150.0, {"preproc": 2.8}))
    assert "WAIT OFF THE BOX" in line and "these disagree" in line, line


def test_a_running_driver_dominated_by_envstep_says_it_is_not_the_retriever():
    """envstep counts the driver's work around the call as well as the wait."""
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"envstep": 2.8}))
    assert "DRIVER RUNNING PYTHON" in line and "mostly in envstep" in line, line
    assert "not the retriever" in line, line


def test_the_ambiguous_band_refuses_to_decide():
    """Between blocked and running is not evidence for either."""
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 40.0, 100.0, {"envstep": 2.8}))
    assert "UNRESOLVED" in line, line


def test_dominance_is_measured_against_the_slot_count():
    """1.1 of 3 slots is a third even when it is half of what was tagged."""
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    res = _both(trace, 82.0, 15.0, {"envstep": 1.1, "preproc": 0.6, "gen": 0.5},
                act_busy={"gen": 3})
    assert res["slots_seen"] == 3
    assert "no single phase dominates" in gp.format_residency(res)


def test_with_no_cpu_sample_a_dominant_phase_is_still_named_but_hedged():
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    act = [(ts, {"envstep": 3} if max(v) == 0 else {"gen": 3}) for ts, v in trace]
    line = gp.format_residency(_FakeSampler(trace, act=act).residency_between(0, 1e9))
    assert "mostly envstep" in line and "no CPU sample to corroborate" in line, line


# --------------------------------------------------------------------------- #
# the phases that were in no phase
# --------------------------------------------------------------------------- #
def test_the_calling_threads_phases_are_nameable():
    """dataload, prepare and scoring run on the thread nobody counted.

    The census tagged the slot threads only, so the thread that loads, prepares
    and accumulates -- holding the GIL for all of it -- appeared nowhere. That
    is what "0.8 of 3 slots in no tagged phase" was made of, and it was missing
    from exactly the samples where every card was empty.
    """
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"scoring": 2.6}))
    assert "reward accumulation on the calling thread" in line, line


def test_the_between_phases_glue_is_nameable():
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"glue": 2.6}))
    assert "padding and DataProto union" in line, line


def test_envstep_under_a_running_driver_still_says_it_is_not_the_retriever():
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"envstep": 2.6}))
    assert "not the retriever" in line, line


def test_the_per_turn_bookkeeping_is_nameable():
    """MEASURED: "1.5 of 3 slots in no tagged phase", half the driver's time.

    to_list_of_dict expands the whole 252-row DataProto into dicts on every
    turn, and gather_rollout_data pads every turn of every trajectory after the
    last generate returns -- both on the driver, both with the cards empty, and
    neither was in any phase.
    """
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"record": 2.6}))
    assert "to_list_of_dict expands the whole batch" in line, line


def test_the_final_assembly_is_nameable():
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    line = gp.format_residency(_both(trace, 90.0, 20.0, {"assemble": 2.6}))
    assert "gather_rollout_data" in line, line


def test_a_large_untagged_remainder_is_the_headline_number():
    """Half in no phase means the tagging is the thing to fix, not the phases."""
    trace = _trace([[0, 0, 0]] * 3 + [[92, 92, 92]] * 7)
    res = _both(trace, 80.0, 11.0, {"gen": 0.8, "preproc": 0.5, "scoring": 0.1})
    assert res["slots_seen"] == 3
    line = gp.format_residency(res)
    assert "1.6 in no tagged phase" in line, line
    assert "no single phase dominates" in line, line


# --------------------------------------------------------------------------- #
# a mean over two modes describes neither
# --------------------------------------------------------------------------- #
def _bimodal(trace, low, high, busy_cpu, act_empty):
    cpu = []
    i = 0
    for ts, v in trace:
        if max(v) == 0:
            cpu.append((ts, low if i % 2 else high))
            i += 1
        else:
            cpu.append((ts, busy_cpu))
    act = [(ts, dict(act_empty) if max(v) == 0 else {"gen": 3}) for ts, v in trace]
    return _FakeSampler(trace, cpu, act).residency_between(0, 1e9)


def test_a_two_mode_empty_is_split_rather_than_averaged():
    """MEASURED, pump + multi-step: "driver CPU 52% of one core ... UNRESOLVED".

    52% is the mean of samples at ~5% (blocked on a future) and ~100%
    (working). It is true of the mean and true of nothing that happened, and
    the verdict then refused to decide -- correctly, on a number that should
    never have been a single number.
    """
    trace = _trace([[0, 0, 0]] * 6 + [[92, 92, 92]] * 6)
    res = _bimodal(trace, 5.0, 99.0, 23.0, {"envstep": 1.1, "gen": 0.6})
    line = gp.format_residency(res)
    assert "EMPTY SPLITS" in line, line
    assert "BLOCKED (cpu<20%) for 50%" in line, line
    assert "RUNNING (cpu>=60%) for 50%" in line, line
    assert "UNRESOLVED" not in line, "a split is an answer; UNRESOLVED was the absence of one"


def test_the_split_says_what_the_working_half_was_doing():
    """Only the running half makes "where" a meaningful question."""
    trace = _trace([[0, 0, 0]] * 6 + [[92, 92, 92]] * 6)
    res = _bimodal(trace, 5.0, 99.0, 23.0, {"envstep": 1.1, "gen": 0.6})
    assert res["activity_when_empty_running"], "the running samples need their own census"
    assert "while RUNNING-and-empty the slots were in" in gp.format_residency(res)


def test_a_genuinely_middling_cpu_still_refuses():
    """Every sample at 45% is one mode, and it is not evidence for either."""
    trace = _trace([[0, 0, 0]] * 6 + [[92, 92, 92]] * 6)
    cpu = _cpu(trace, lambda v: 45.0 if max(v) == 0 else 20.0)
    line = gp.format_residency(_FakeSampler(trace, cpu).residency_between(0, 1e9))
    assert "UNRESOLVED" in line, line
    assert "EMPTY SPLITS" not in line, line


def test_the_bands_are_shares_of_the_empty_samples():
    trace = _trace([[0, 0, 0]] * 4 + [[92, 92, 92]] * 6)
    cpu = _cpu(trace, lambda v: 3.0 if max(v) == 0 else 90.0)
    res = _FakeSampler(trace, cpu).residency_between(0, 1e9)
    assert res["empty_cpu_bands"]["blocked"] == pytest.approx(100.0)
    assert res["empty_cpu_bands"]["running"] == pytest.approx(0.0)


def test_the_shares_are_weighted_by_WALL_not_by_sample_count():
    """The bucket whose samples are stretched is the bucket being measured.

    While every card is EMPTY the driver is running Python holding the GIL, the
    sampler thread is starved, and the interval between its samples grows. A
    share computed over sample COUNT therefore under-reports EMPTY by exactly
    the amount the driver stole from the sampler -- the one direction that
    flatters the result. Measured on a real run: 312 s of 3541 s is 8.8% of the
    seconds and printed as 6.8% of the samples.

    Every fixture above spaces its samples evenly, where the two definitions
    coincide. That is why this was invisible: the instrument agreed with itself
    on every test and disagreed with itself in every log, where the seconds
    printed next to the percentage did not divide into it.
    """
    interval, cap = gp._INTERVAL, gp._INTERVAL * gp._CONTIGUITY_SLACK
    trace, t = [], 100.0
    for _ in range(8):  # eight busy samples, one interval apart
        t += interval
        trace.append((t, [90, 90, 90]))
    for _ in range(2):  # two empty samples, at the widest gap that still counts
        t += cap
        trace.append((t, [0, 0, 0]))

    res = _FakeSampler(trace).residency_between(0, 1e9)
    assert res["counts"] == {0: 2, 1: 0, 2: 0, 3: 8}

    # 2 of 10 samples, but 2*0.6s of 8*0.3s + 2*0.6s = 3.6s of wall.
    by_samples = 100.0 * 2 / 10
    by_wall = 100.0 * res["wall_by_count"][0] / res["wall"]
    assert by_wall == pytest.approx(100.0 * 1.2 / 3.6)
    assert by_wall > by_samples
    assert res["pct"][0] == pytest.approx(by_wall)

    # And the printed percentage divides the printed seconds, which is the
    # property a reader assumes and the old one did not have.
    for k, seconds in res["wall_by_count"].items():
        assert res["pct"][k] == pytest.approx(100.0 * seconds / res["wall"])


# --------------------------------------------------------------------------- #
# Asking the interpreter instead of the tags
# --------------------------------------------------------------------------- #
def test_a_thread_waiting_inside_repo_code_names_both_frames():
    """Our frame says whose fault it is; the innermost says what it is doing.

    They differ in exactly the case that matters -- our code blocked inside
    somebody else's -- and a key that carried only one of them would answer
    either "val_pipeline" (so what) or "queue.py: get" (whose queue?).
    """
    import queue
    import threading
    import time

    from verl.utils.val_pipeline import Slot, run_pipelined

    gate, seen = queue.Queue(), {}

    def sampler():
        time.sleep(0.25)
        seen["keys"] = gp.stack_snapshot()
        gate.put(None)

    threading.Thread(target=sampler, name="probe", daemon=True).start()
    slots = [Slot(f"s{i}", envs=None, collector=None) for i in range(2)]
    list(run_pipelined([1], lambda x: x, lambda _p: None,
                       lambda _p, _s: gate.get(timeout=5), slots))

    keys = seen["keys"]
    # The worker: repo code blocked in the standard library. Both halves,
    # because "threading.py: wait" alone would not say who is waiting -- and
    # Queue.get parks in a Condition, so the innermost frame is threading's,
    # not queue's. That is the point of keeping our frame too.
    joined = [k for k in keys if "<-" in k]
    assert joined, keys
    assert any(k.split(" <- ")[1].startswith("threading.py") for k in joined), keys
    assert any("test_gpu_residency.py" in k.split(" <- ")[0] for k in joined), keys
    # The calling thread, which the census could never see: parked in retire()
    # waiting on the future. This is a real finding, not scaffolding -- an
    # untagged thread that is blocked, not running.
    assert any("val_pipeline.py" in k and "retire" in k for k in keys), keys


def test_the_stacks_are_aggregated_over_the_empty_samples_only():
    """Mean threads per sample, the same unit as the census beside it."""
    ts = [100.0 + i * 0.3 for i in range(4)]
    trace = list(zip(ts, [[0, 0, 0], [0, 0, 0], [90, 90, 90], [90, 90, 90]]))
    stacks = [
        (ts[0], ["rollout_loop.py:1 a", "rollout_loop.py:2 b"]),
        (ts[1], ["rollout_loop.py:1 a"]),
        (ts[2], ["rollout_loop.py:9 busy"]),
        (ts[3], ["rollout_loop.py:9 busy"]),
    ]
    res = _FakeSampler(trace, stacks=stacks).residency_between(0, 1e9)
    empty = dict(res["stacks_when_empty"])
    assert empty["rollout_loop.py:1 a"] == pytest.approx(1.0)   # in both empty samples
    assert empty["rollout_loop.py:2 b"] == pytest.approx(0.5)   # in one of two
    assert "rollout_loop.py:9 busy" not in empty                # that one was busy
    assert dict(res["stacks_when_busy"])["rollout_loop.py:9 busy"] == pytest.approx(1.0)


def test_the_frames_print_when_the_tags_leave_a_third_of_a_thread_unclaimed():
    """The line only earns its space when the census has failed to explain."""
    ts = [100.0 + i * 0.3 for i in range(2)]
    trace = list(zip(ts, [[0, 0, 0], [0, 0, 0]]))
    # slots_seen is the most threads ever tagged at once, so the fixture has to
    # show three before one of them can be missing.
    act = [(ts[0], {"gen": 3}), (ts[1], {"gen": 1})]
    stacks = [(t, ["rollout_loop.py:1408 _scatter_active_to_full"] * 2) for t in ts]
    out = gp.format_residency(_FakeSampler(trace, act=act, stacks=stacks).residency_between(0, 1e9))
    assert "in NO tagged phase" in out
    assert "_scatter_active_to_full" in out

    # Everything tagged -> nothing to explain, no frame dump.
    act_full = [(t, {"gen": 3}) for t in ts]
    out = gp.format_residency(_FakeSampler(trace, act=act_full, stacks=stacks).residency_between(0, 1e9))
    assert "in NO tagged phase" not in out
