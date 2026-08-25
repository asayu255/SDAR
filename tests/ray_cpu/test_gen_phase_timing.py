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
"""Attributing the fixed cost inside one generate_sequences call.

The driver times the call as a single span, and the NVML sampler says the GPU
sits idle for about 0.65 s of it regardless of what the call contains -- the
signature of a per-call cost rather than one that scales with the data. From
the driver there is no way to tell the Ray round trip from the sharding
manager's reshaping from the engine call itself; they are one opaque span.

The worker can see four of those five legs, so it sums them and prints the mean
every N calls. The fifth -- the round trip -- is what the driver's number minus
this one leaves, which is the only way either side can get at it.

These tests hold the accumulation and the printing cadence: a per-call print
across 413 batches would be its own problem, and an accumulator that resets
would quietly report one call's noise as the mean.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

import verl.workers.fsdp_workers as fsdp_workers  # noqa: E402


class _Worker:
    """Just the fields _record_gen_phases touches."""

    _record_gen_phases = fsdp_workers.ActorRolloutRefWorker._record_gen_phases

    def __init__(self, rank=0):
        self._rank = rank


def _call(worker, generate=1.0, **rest):
    marks = {"to_device": 0.01, "preprocess": 0.1, "generate": generate, "postprocess": 0.05, "to_cpu": 0.02}
    marks.update(rest)
    worker._record_gen_phases(marks)


def test_nothing_is_printed_before_the_period_is_up(monkeypatch, capsys):
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "50")
    worker = _Worker()
    for _ in range(49):
        _call(worker)
    assert capsys.readouterr().out == ""


def test_the_mean_is_printed_on_the_period(monkeypatch, capsys):
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "10")
    worker = _Worker()
    for _ in range(10):
        _call(worker, generate=2.0)
    out = capsys.readouterr().out
    assert "[gen-phases] rank 0, mean over 10 calls" in out
    assert "generate 2.00" in out
    assert "total 2.18" in out


def test_the_mean_is_over_every_call_not_the_last_period(monkeypatch, capsys):
    """A counter that reset each period would report noise as the mean."""
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "10")
    worker = _Worker()
    for _ in range(10):
        _call(worker, generate=1.0)
    capsys.readouterr()
    for _ in range(10):
        _call(worker, generate=3.0)
    out = capsys.readouterr().out
    assert "mean over 20 calls" in out
    assert "generate 2.00" in out  # (10*1 + 10*3) / 20


def test_only_rank_zero_prints(monkeypatch, capsys):
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "5")
    worker = _Worker(rank=2)
    for _ in range(5):
        _call(worker)
    assert capsys.readouterr().out == ""
    # but it still accumulated, so the rank is a print filter and not a gate
    assert worker._gen_phase_timer.calls == 5


def test_a_zero_period_turns_the_print_off(monkeypatch, capsys):
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "0")
    worker = _Worker()
    for _ in range(100):
        _call(worker)
    assert capsys.readouterr().out == ""


def test_a_missing_phase_counts_as_zero(monkeypatch, capsys):
    """The no-session branch and the session branch record the same names, but a
    partial call must not raise inside the thing being measured."""
    monkeypatch.setenv("ROLLOUT_GEN_PHASE_EVERY", "1")
    worker = _Worker()
    worker._record_gen_phases({"generate": 1.5})
    out = capsys.readouterr().out
    assert "generate 1.50" in out
    assert "preprocess 0.00" in out


def test_mark_records_the_span_and_advances_the_cursor():
    marks = {}
    start = 100.0
    after = fsdp_workers._mark(marks, "preprocess", start)
    assert after > start
    assert marks["preprocess"] == pytest.approx(after - start, abs=1e-6)


def test_mark_is_a_no_op_when_timing_is_off():
    """marks is None on a run without ROLLOUT_TURN_TIMING; the cursor must be
    returned unchanged so the call site stays branch-free."""
    assert fsdp_workers._mark(None, "preprocess", 100.0) == 100.0


# --------------------------------------------------------------------------- #
# The accumulator is only as good as the call sites. generate_sequences has two
# branches -- inside a rollout session and outside it -- and a phase dropped
# from one of them does not raise: PhaseTimer.record treats an unreached phase
# as zero, so the mean silently halves and the missing time reappears in the
# driver's number as "the Ray round trip". Pinned at the source for that
# reason.
# --------------------------------------------------------------------------- #


def _generate_sequences_source():
    import inspect
    import textwrap

    return textwrap.dedent(
        inspect.getsource(fsdp_workers.ActorRolloutRefWorker.generate_sequences)
    )


def test_every_declared_phase_is_actually_recorded():
    src = _generate_sequences_source()
    for phase in fsdp_workers._GEN_PHASES:
        assert f'"{phase}"' in src, (
            f"{phase} is in _GEN_PHASES but nothing records it; it would report "
            "as a constant zero rather than as missing"
        )


def test_both_branches_time_the_three_phases_they_share():
    """to_device and to_cpu are outside the branch, so they are recorded once.
    preprocess/generate/postprocess are inside it and must appear on both
    sides -- twice each, once per branch."""
    src = _generate_sequences_source()
    for phase in ("preprocess", "generate", "postprocess"):
        assert src.count(f'_mark(marks, "{phase}"') == 2, (
            f"{phase} is marked {src.count(chr(34) + phase + chr(34))} times; "
            "the session and no-session branches must each time it"
        )
    for phase in ("to_device", "to_cpu"):
        assert src.count(f'marks["{phase}"]') == 1


def test_the_timing_is_off_unless_asked_for():
    """A run that is not being profiled should not carry even the dict."""
    assert fsdp_workers._GEN_PHASE_TIMING is False or os.environ.get("ROLLOUT_TURN_TIMING")
