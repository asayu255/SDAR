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
"""One update_actor stays queued behind the running one.

At 0.23 s resolution the training phase is at 98.4% and every dip together is
0.46% of the wall clock -- and the longest of them are all the same thing: a
once-per-step window where the phase stack is empty and all three GPUs read 0,
while the driver reduces the finished step's metrics and re-serialises ~480 MB
of the next batch. A Ray actor runs its calls one at a time and in order, so
dispatching step k+1 before waiting on step k lets the workers start it the
instant k returns.

Nothing about the arithmetic moves: the actor still cannot begin k+1 until k has
returned. What the pipeline must not do is run ahead across a step whose
retirement reads worker state -- a checkpoint taken with a call already queued
would hold the weights of a later step under this step's name. That is what
``_step_has_barrier`` is for, and most of this file is about it.
"""

import pytest

try:
    from omegaconf import OmegaConf

    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


class _Trainer:
    """Just enough trainer for the barrier predicate."""

    def __init__(self, save_freq=25, test_freq=-1, total=300, val_reward_fn=object(), test_start_step=0):
        self.config = OmegaConf.create({
            "trainer": {"save_freq": save_freq, "test_freq": test_freq, "test_start_step": test_start_step}
        })
        self.total_training_steps = total
        self.val_reward_fn = val_reward_fn

    def barrier(self, step):
        return OffPolicyOPDRayTrainer._step_has_barrier(self, step)


def test_a_save_step_is_a_barrier():
    """_save_checkpoint reads the worker's weights and names the file after this
    step. A queued update_actor would have moved them on by then."""
    t = _Trainer(save_freq=25)

    assert t.barrier(25)
    assert t.barrier(50)
    assert not t.barrier(24)
    assert not t.barrier(26)


def test_the_last_step_is_a_barrier_whatever_its_number():
    t = _Trainer(save_freq=25, total=301)

    assert t.barrier(301)      # saves because it is last, not because 301 % 25
    assert not t.barrier(299)  # 300 is a save step in its own right, 299 is not


def test_a_validation_step_is_a_barrier_too():
    """Validation generates from the same weights and has the same problem."""
    t = _Trainer(save_freq=-1, test_freq=150)

    assert t.barrier(150)
    assert t.barrier(300)
    assert not t.barrier(151)


def test_this_arm_has_no_validation_barriers():
    """The SFT arm runs test_freq=-1: validation happens after the run, so the
    only barriers are the twelve saves."""
    t = _Trainer(save_freq=25, test_freq=-1)

    assert sum(t.barrier(s) for s in range(1, 301)) == 12


def test_validation_barriers_respect_test_start_step():
    t = _Trainer(save_freq=-1, test_freq=50, test_start_step=100)

    assert not t.barrier(50)
    assert t.barrier(100)
    assert t.barrier(150)


def test_no_reward_function_means_no_validation_barrier():
    t = _Trainer(save_freq=-1, test_freq=150, val_reward_fn=None)

    assert not t.barrier(150)


def _run_pipeline(n_steps, depth, is_barrier):
    """Replay fit()'s fill/retire discipline and record the interleaving.

    Returns the event log: ("launch", step) and ("retire", step).
    """
    from collections import deque

    events, inflight, launch_step, exhausted = [], deque(), 1, False
    while True:
        while not exhausted and len(inflight) < depth:
            if inflight and is_barrier(inflight[0]):
                break
            if launch_step > n_steps:
                exhausted = True
                break
            inflight.append(launch_step)
            events.append(("launch", launch_step))
            launch_step += 1
        if not inflight:
            break
        step = inflight.popleft()
        events.append(("retire", step))
    return events


def test_every_step_runs_once_and_in_order():
    events = _run_pipeline(10, depth=2, is_barrier=lambda s: False)

    assert [s for kind, s in events if kind == "launch"] == list(range(1, 11))
    assert [s for kind, s in events if kind == "retire"] == list(range(1, 11))


def test_the_next_step_is_launched_before_the_current_one_is_retired():
    """The point of the whole thing: when step k is retired, k+1 is already with
    the workers, so they start it without waiting for the driver."""
    events = _run_pipeline(10, depth=2, is_barrier=lambda s: False)

    for k in range(1, 10):
        launch_next = events.index(("launch", k + 1))
        retire_this = events.index(("retire", k))
        assert launch_next < retire_this, f"step {k+1} was not queued before step {k} retired"


def test_nothing_is_launched_across_a_barrier():
    """A save at step 5: step 6 must not be with the workers when 5 is retired."""
    events = _run_pipeline(10, depth=2, is_barrier=lambda s: s == 5)

    assert events.index(("retire", 5)) < events.index(("launch", 6))
    # and the pipeline picks straight back up afterwards
    assert events.index(("launch", 7)) < events.index(("retire", 6))


def test_depth_one_is_the_old_serial_order():
    """With the pipeline off, every launch is immediately followed by its retire
    -- the behaviour the flag defaults to."""
    events = _run_pipeline(6, depth=1, is_barrier=lambda s: False)

    assert events == [e for s in range(1, 7) for e in (("launch", s), ("retire", s))]


def test_back_to_back_barriers_still_drain():
    events = _run_pipeline(6, depth=2, is_barrier=lambda s: s in (3, 4))

    assert [s for kind, s in events if kind == "retire"] == list(range(1, 7))
    assert events.index(("retire", 3)) < events.index(("launch", 4))
    assert events.index(("retire", 4)) < events.index(("launch", 5))
