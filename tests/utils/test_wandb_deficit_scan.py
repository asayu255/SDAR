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
"""The three pieces of wandb_deficit_scan that decide what a number means.

The tool splits a run's GPU idle into checkpoint / step boundary / ambient. Its
arithmetic is a mean, and a mean cannot be wrong -- what can be wrong is which
samples went into which bucket, and every way that has gone wrong so far is in
this file:

  * **The clocks.** ``_runtime`` is not shared between the training history and
    the system stream: a resumed run continues the previous run's clock in one
    and restarts at zero in the other (307 s apart on run 2ynp20j6, a whole
    step). Aligning on ``_runtime`` charges a checkpoint's idle window to the
    wrong step and clears the checkpoint of a cost it caused.
  * **The startup.** The first logged step's timer reaches back over the model
    load, where the GPUs are idle for minutes on purpose. Counted as training,
    it appears as a pile of "discrete events inside a step" -- which is exactly
    the signature being hunted, arriving from the one place it means nothing.
  * **The save-vs-boundary order.** Every save step's end is also a step
    boundary. Testing the boundary first would file the checkpoint's own idle
    under the driver.
"""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "wandb_deficit_scan.py"
_spec = importlib.util.spec_from_file_location("wandb_deficit_scan", _PATH)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)


def _row(timestamp, runtime, util=100.0, cards=3):
    row = {"_timestamp": timestamp, "_runtime": runtime}
    for k in range(cards):
        row[f"system.gpu.{k}.gpu"] = util
    return row


def test_epoch_comes_from_the_system_stream_not_the_history():
    """A resumed run's system stream restarts at _runtime 0 while its history
    keeps counting. The epoch has to be read off the stream being converted."""
    system = [_row(1_700_000_307.0, 15.0), _row(1_700_000_322.0, 30.0)]
    assert scan.clock_epoch(system) == pytest.approx(1_700_000_292.0)


def test_epoch_survives_a_resume_offset():
    """The failure this guards: a history row at _runtime 1129 and a system row
    at _runtime 822 that are the SAME instant. Converting both through the wall
    clock has to bring them together."""
    epoch_sys = 1_700_000_000.0
    system = [_row(epoch_sys + 822.0, 822.0)]
    history_timestamp = epoch_sys + 822.0  # same instant, _runtime would say 1129
    assert history_timestamp - scan.clock_epoch(system) == pytest.approx(822.0)


def test_training_starts_at_the_first_busy_window():
    epoch = 0.0
    system = [_row(t, t, util=0.0) for t in range(0, 300, 15)] + [_row(t, t, util=100.0) for t in range(300, 600, 15)]
    # the first step's timer opens at 0, long before anything ran
    assert scan.training_start(system, epoch, earliest=0.0, cards=3) == 300.0


def test_training_start_never_moves_earlier_than_the_first_step():
    """A busy window from BEFORE the first logged step is not this run's
    training -- on a resume it is the tail of whatever ran previously."""
    epoch = 0.0
    system = [_row(t, t, util=100.0) for t in range(0, 600, 15)]
    assert scan.training_start(system, epoch, earliest=200.0, cards=3) == 210.0


def test_training_start_falls_back_when_nothing_is_busy():
    epoch = 0.0
    system = [_row(t, t, util=40.0) for t in range(0, 300, 15)]
    assert scan.training_start(system, epoch, earliest=45.0, cards=3) == 45.0


def test_a_save_window_is_a_save_not_a_boundary():
    """Both predicates match at a save step's end; save has to win, or the
    checkpoint's cost lands in the driver's column."""
    assert scan.classify(1000.0, saves=[1000.0], bounds=[1000.0]) == "save"


def test_the_save_window_is_wider_than_the_boundary_window():
    """The background write outlives the staging, so the save's skirt has to
    reach past the driver's tail."""
    assert scan.SAVE_PAD_S > scan.BOUNDARY_PAD_S
    assert scan.classify(1015.0, saves=[1000.0], bounds=[1000.0]) == "save"
    assert scan.classify(1005.0, saves=[], bounds=[1000.0]) == "boundary"


def test_interior_is_what_is_near_neither():
    assert scan.classify(1200.0, saves=[1000.0], bounds=[1000.0, 1300.0]) == "interior"


def test_classification_is_symmetric_around_a_boundary():
    """A save's idle can precede its logged end (the staging runs before the
    log call) as easily as follow it."""
    for offset in (-24.0, 24.0):
        assert scan.classify(1000.0 + offset, saves=[1000.0], bounds=[]) == "save"
    for offset in (-26.0, 26.0):
        assert scan.classify(1000.0 + offset, saves=[1000.0], bounds=[]) == "interior"


def test_deep_threshold_is_above_what_launch_gaps_can_reach():
    """A 15 s window reading 90% needs 1.5 s of idle in it. Nothing that big
    comes from the microsecond gaps between kernel launches, so anything under
    the line is an event with a cause, not dust."""
    assert scan.DEEP_UTIL >= 90.0
