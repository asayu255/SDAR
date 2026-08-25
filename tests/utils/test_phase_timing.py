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
"""Running means of a call's phases.

Two callers need this -- the worker splitting generate_sequences, and the vllm
rollout splitting the engine call from the Python around it -- and neither can
print per call: a generate happens once per turn, tens of thousands of times in
an evaluation. It lives outside both so it can be tested without a GPU or a
vllm install, which is exactly what the two copies of it could not be.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from verl.utils.phase_timing import PhaseTimer, mark  # noqa: E402


def _timer(every=2, phases=("a", "b"), rank=0, note=""):
    lines = []
    timer = PhaseTimer("t", phases, every=every, note=note, rank=lambda: rank, printer=lines.append)
    return timer, lines


def test_nothing_prints_before_the_period():
    timer, lines = _timer(every=50)
    for _ in range(49):
        timer.record({"a": 1.0, "b": 2.0})
    assert lines == []


def test_the_mean_is_over_every_call_not_the_period():
    """A counter that reset each period would report noise as the mean."""
    timer, lines = _timer(every=10)
    for _ in range(10):
        timer.record({"a": 1.0})
    for _ in range(10):
        timer.record({"a": 3.0})
    assert "mean over 20 calls" in lines[-1]
    assert "a 2.000" in lines[-1]


def test_a_missing_phase_counts_as_zero():
    """Two code paths record the same names; a partial call must not raise
    inside the thing being measured."""
    timer, lines = _timer(every=1)
    timer.record({"a": 1.5})
    assert "a 1.500" in lines[-1] and "b 0.000" in lines[-1]


def test_the_total_is_the_sum_of_the_phases():
    timer, lines = _timer(every=1)
    timer.record({"a": 1.0, "b": 2.0})
    assert "total 3.000" in lines[-1]


def test_only_rank_zero_prints_but_every_rank_accumulates():
    timer, lines = _timer(every=2, rank=3)
    for _ in range(4):
        timer.record({"a": 1.0})
    assert lines == []
    assert timer.calls == 4


def test_a_rank_that_cannot_be_determined_still_prints():
    """Single-process runs have no distributed rank, and silence there would
    read as a measurement that came back empty."""
    timer, lines = _timer(every=1, rank=None)
    timer.record({"a": 1.0})
    assert lines


def test_a_zero_period_accumulates_without_printing():
    timer, lines = _timer(every=0)
    for _ in range(100):
        timer.record({"a": 1.0})
    assert lines == []
    assert timer.calls == 100
    assert "a 1.000" in timer.line()  # still available on demand


def test_the_note_is_carried_into_the_line():
    timer, lines = _timer(every=1, note="(engine = vllm)")
    timer.record({"a": 1.0})
    assert lines[-1].endswith("(engine = vllm)")


def test_line_before_any_call_does_not_divide_by_zero():
    timer, _ = _timer(every=1)
    assert "mean over 0 calls" in timer.line()


def test_mark_records_the_span_and_advances():
    marks = {}
    after = mark(marks, "a", 100.0)
    assert after > 100.0
    assert marks["a"] == pytest.approx(after - 100.0, abs=1e-6)


def test_mark_is_a_no_op_when_timing_is_off():
    """marks is None on a run without timing; the cursor must come back
    unchanged so the call site stays branch-free."""
    assert mark(None, "a", 100.0) == 100.0
