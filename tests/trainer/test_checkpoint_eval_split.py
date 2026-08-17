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
"""Validation happens after the run, not inside it.

The multitask SFT arm sets ``trainer.test_freq=-1`` and scores its checkpoints
afterwards with ``examples/sft_trainer/eval_checkpoints.sh``, one process per
checkpoint. Measured on the run that validated in-loop (wandb x7g9r7bx): two
passes took 7.6 h of a 42.5 h run and 68.6% of all the GPU time it left idle,
because validation is a full agentic rollout rather than this arm's loss.

These tests gate the three things that split depends on:

* a ``val_only`` process does not pay for the training pool it never draws from;
* ``val_only`` is sufficient on its own to make fit() validate, which is what
  keeps the eval command to keys the run script does not already pass;
* the run script and the eval driver still agree about which keys those are.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "examples" / "sft_trainer" / "run_multitask_sft_qwen3.sh"
EVAL_SCRIPT = REPO / "examples" / "sft_trainer" / "eval_checkpoints.sh"

# Keys the eval driver appends. The no-duplicate-override design is only correct
# while the run script passes none of them itself.
EVAL_ONLY_KEYS = ("trainer.val_only", "trainer.resume_mode", "trainer.resume_from_path")


def _command_lines(script: Path):
    """The script's lines with comments and blanks dropped.

    A key named in a comment is documentation, not an argument, and both files
    document the keys they do not pass.
    """
    return [
        line for line in script.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class _Trainer:
    """The parts of OffPolicyOPDRayTrainer these two methods touch, and no more.

    Binding the real methods onto a stub keeps this a unit test of the decisions
    rather than of Ray, FSDP and vLLM.
    """

    def __init__(self, val_only=False, val_before_train=False, val_reward_fn=object()):
        from omegaconf import OmegaConf

        self.config = OmegaConf.create(
            {"trainer": {"val_only": val_only, "val_before_train": val_before_train}}
        )
        self.val_reward_fn = val_reward_fn
        self.teacher_data_dir = "/pool/that/is/never/read"
        self.loads = 0

    def _load_offpolicy_data(self):
        self.loads += 1

    def _is_val_only(self):
        from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

        return OffPolicyOPDRayTrainer._is_val_only(self)

    def maybe_load(self):
        from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

        return OffPolicyOPDRayTrainer._maybe_load_offpolicy_data(self)

    def should_validate(self):
        from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

        return OffPolicyOPDRayTrainer._should_validate_before_training(self)


@pytest.fixture
def needs_verl():
    """Only the trainer tests need the stack; the script-contract tests below are
    plain text and must still run where torch is not installed."""
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    try:
        import verl.trainer.ppo.opd_offpolicy_ray_trainer  # noqa: F401
    except Exception as e:  # pragma: no cover - environment without full deps
        pytest.skip(f"verl import unavailable: {e}")


def test_a_val_only_process_does_not_load_the_teacher_pool(needs_verl):
    """136.5 GiB and a multi-minute read for a process that never draws a batch.

    Paying it per checkpoint is what would make per-checkpoint evaluation too
    expensive to be worth splitting out.
    """
    trainer = _Trainer(val_only=True)

    trainer.maybe_load()

    assert trainer.loads == 0


def test_a_training_process_still_loads_it(needs_verl):
    """The guard must not disarm the load for the case that needs it."""
    trainer = _Trainer(val_only=False)

    trainer.maybe_load()

    assert trainer.loads == 1


def test_drawing_a_batch_without_the_pool_says_why(needs_verl):
    """A val_only process returns before the training loop, so reaching the draw
    means the skip fired for a run that does train. Without this the failure is an
    AttributeError on an attribute whose absence explains nothing."""
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

    trainer = _Trainer(val_only=True)
    trainer.maybe_load()

    with pytest.raises(AssertionError, match="trainer.val_only"):
        next(OffPolicyOPDRayTrainer._offpolicy_batch_iter(trainer))


@pytest.mark.parametrize(
    "val_only,val_before_train,expected",
    [
        # The eval path: the run script pins val_before_train=False, and the driver
        # adds only val_only. If that did not imply validating, the process would
        # load a checkpoint, do nothing and exit 0 -- a silent no-op.
        (True, False, True),
        (True, True, True),
        # Training unchanged: val_before_train alone still decides.
        (False, False, False),
        (False, True, True),
    ],
)
def test_val_only_is_enough_to_make_fit_validate(needs_verl, val_only, val_before_train, expected):
    assert _Trainer(val_only=val_only, val_before_train=val_before_train).should_validate() is expected


def test_no_validation_without_a_reward_function(needs_verl):
    assert _Trainer(val_only=True, val_reward_fn=None).should_validate() is False


def test_the_training_run_never_validates_in_loop():
    """test_freq=-1 is the training half of the split. Its positive value is what
    cost the measured 7.6 h; the expectations file pins it, and this states it at
    the level of the script so a reader of either file finds it."""
    lines = _command_lines(RUN_SCRIPT)

    assert any("trainer.test_freq=-1" in line for line in lines)
    assert not any(re.search(r"trainer\.test_freq=(?!-1)", line) for line in lines)


def test_the_run_script_leaves_the_eval_keys_alone():
    """The eval driver appends its three keys to this script's own command line.

    Hydra is handed each key once, which is only true while the run script passes
    none of them. If one is ever added there, the driver has to stop passing it
    rather than pass it a second time.
    """
    lines = _command_lines(RUN_SCRIPT)

    for key in EVAL_ONLY_KEYS:
        assert not any(f"{key}=" in line for line in lines), (
            f"{key} is now set by the run script; eval_checkpoints.sh would pass it twice"
        )


def test_the_eval_driver_passes_exactly_those_keys():
    lines = _command_lines(EVAL_SCRIPT)

    for key in EVAL_ONLY_KEYS:
        assert any(f"{key}=" in line for line in lines), f"{key} is no longer passed by the eval driver"


def test_the_eval_driver_runs_the_training_script_rather_than_repeating_it():
    """The validation protocol (tasks, episode caps, val_kwargs_by_task, the
    retriever) has to be the training run's. Invoking that script is what makes it
    so by construction; a second copy of the arguments could drift from it."""
    lines = _command_lines(EVAL_SCRIPT)

    assert any("run_multitask_sft_qwen3.sh" in line for line in lines)
    # The one scientific knob a copy would be tempted to restate.
    assert not any("val_kwargs_by_task" in line for line in lines)
