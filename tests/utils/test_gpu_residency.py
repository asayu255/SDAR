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

    def __init__(self, trace):
        import threading

        self._lock = threading.Lock()
        self._util_trace = list(trace)

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

        def __init__(self, trace):
            self._lock = threading.Lock()
            self._util_trace = list(trace)

    start = time.perf_counter()
    end = start + 3.0
    # Stamped on the sampler's clock, which is what the conversion has to hit.
    offset = time.monotonic() - time.perf_counter()
    trace = _trace([[0, 0, 0]] * 2 + [[90, 90, 90]] * 8, interval=0.3, t0=start + offset)
    monkeypatch.setattr(live, "_sampler", _LiveFakeSampler(trace))
    monkeypatch.setattr(live, "enabled", lambda: True)

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
