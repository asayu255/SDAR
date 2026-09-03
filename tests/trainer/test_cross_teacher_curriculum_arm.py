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
"""The curriculum arm must differ from its comparators in the mechanism and nothing else.

Composed configs rather than script text, for the reason the other two pairing
tests give: an injection in main_opd or a default moved in ppo_trainer.yaml never
shows in a diff of the arguments.

This arm carries one obligation the tilt arm does not. Its prediction is that the
END of the run agrees with the control, so a schedule that does not fit inside
the run, or a lock that lets the schedule drift, would not weaken the experiment
-- it would delete it. That is what the schedule tests below are for.
"""
import os

import pytest

pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from tests.trainer.test_signweight_arms_match_the_control import (
    IDENTITY,
    _differing,
    _effective,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CURRICULUM = "examples/opd_grpo_trainer/run_multitask_cross_teacher_curriculum_qwen3.sh"
TARGET = "examples/opd_grpo_trainer/run_multitask_cross_teacher_target_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"

_MECHANISMS = ("algorithm.opd.cross_teacher_target.",
               "algorithm.opd.cross_teacher_kl_weight.",
               "actor_rollout_ref.actor.cross_teacher_target.",
               "actor_rollout_ref.actor.cross_teacher_kl_weight.")


def _mechanism_or_identity(differing):
    allowed = set(IDENTITY)
    allowed |= {k for k in differing if k.startswith(_MECHANISMS)}
    allowed |= {k for k in differing if "micro_batch_size" in k}
    return allowed


@pytest.mark.parametrize("other,name", [(CONTROL, "control"), (TARGET, "tilt arm")])
def test_the_curriculum_arm_differs_only_in_the_mechanism(other, name):
    a = _effective(CURRICULUM)
    b = _effective(other)
    differing = _differing(a, b)
    assert differing <= _mechanism_or_identity(differing), sorted(
        differing - _mechanism_or_identity(differing)
    )
    assert any(k.startswith("algorithm.opd.cross_teacher_target.") for k in differing), (
        f"the curriculum arm and the {name} are identical"
    )


def test_the_arms_do_not_share_a_directory():
    """Shared default_local_dir + resume_mode=auto means the arm started second
    RESUMES FROM the first and reports it under its own name."""
    arm = _effective(CURRICULUM)
    for other in (CONTROL, TARGET):
        b = _effective(other)
        for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir",
                    "trainer.sign_token_dump_dir", "trainer.project_name"):
            assert arm[key] != b[key], (key, other)


def test_the_mode_is_what_separates_this_arm_from_the_tilt_one():
    """Both arms are cross_teacher_target.enable=true. If the mode were equal the
    two scripts would run the SAME mechanism under two names, which is the one
    confusion no metric would reveal."""
    arm = _effective(CURRICULUM)
    tilt = _effective(TARGET)
    assert arm["algorithm.opd.cross_teacher_target.enable"] is True
    assert tilt["algorithm.opd.cross_teacher_target.enable"] is True
    assert arm["algorithm.opd.cross_teacher_target.mode"] == "curriculum"
    assert tilt.get("algorithm.opd.cross_teacher_target.mode", "tilt") in ("tilt", "<absent>")


def test_the_curriculum_carries_no_exponent_scale():
    """The layers are in the on-task teacher's own nats, so the key has nothing to
    convert -- and the actor asserts it is absent rather than ignoring it. A
    script that reintroduced it would fail at startup; this test says so earlier."""
    arm = _effective(CURRICULUM)
    assert arm.get("algorithm.opd.cross_teacher_target.exponent_scale", "<absent>") == "<absent>"
    lock = yaml.safe_load(open(os.path.join(REPO, arm["trainer.expected_config"])))
    assert not any("exponent_scale" in k for k in lock)


def test_the_lock_pins_the_schedule_which_is_the_only_free_parameter():
    arm = _effective(CURRICULUM)
    lock = yaml.safe_load(open(os.path.join(REPO, arm["trainer.expected_config"])))
    for key in ("enable", "mode", "stage_steps", "ramp_steps", "base_path"):
        assert f"algorithm.opd.cross_teacher_target.{key}" in lock, key
    assert lock["algorithm.opd.cross_teacher_target.mode"] == "curriculum"
    assert lock["algorithm.opd.cross_teacher_target.stage_steps"] == [40, 80]
    assert lock["algorithm.opd.cross_teacher_target.ramp_steps"] == 10


def test_the_schedule_finishes_well_inside_the_run():
    """The arm's prediction is about the FULLY RELEASED stage, so the run has to
    contain one. It also has to contain enough of one to validate in: the release
    ends at 90 and the run is 300 steps, so the last 210 steps are the control's
    own objective."""
    arm = _effective(CURRICULUM)
    s1, s2 = arm["algorithm.opd.cross_teacher_target.stage_steps"]
    ramp = arm["algorithm.opd.cross_teacher_target.ramp_steps"]
    total = arm["trainer.total_training_steps"]
    assert s1 + ramp <= s2, "the two ramps must not overlap or there is no stage 2"
    assert s2 + ramp < total, "the run must contain a fully-released stage"
    assert (total - (s2 + ramp)) / total > 0.5, (
        "over half the run should be fully released, or 'the endpoint agrees with the "
        "control' is a claim about a few steps"
    )


def test_the_intermediate_checkpoints_exist_to_be_validated():
    """The arm predicts a difference only in the intermediate steps, so a run that
    saves at 150 and 300 alone cannot test it."""
    arm = _effective(CURRICULUM)
    save_freq = arm["trainer.save_freq"]
    s1, s2 = arm["algorithm.opd.cross_teacher_target.stage_steps"]
    ramp = arm["algorithm.opd.cross_teacher_target.ramp_steps"]
    assert 0 < save_freq <= 25, save_freq
    # at least one checkpoint inside each stage, and one right after full release
    assert save_freq <= s1, "stage 1 must contain a checkpoint"
    assert save_freq <= s2 - (s1 + ramp), "stage 2 must contain a checkpoint"


def test_the_support_matches_the_comparators():
    for script in (CURRICULUM, TARGET, CONTROL):
        assert _effective(script)["actor_rollout_ref.actor.student_indexed_topk"] is True, script


def test_the_teacher_coefficient_matches_the_comparators():
    """beta = 0.01 on all three. The arm changes WHAT is distilled and WHEN, not
    how much the distillation weighs against the reward -- mixing the two would
    make a difference unattributable."""
    want = _effective(CONTROL)["actor_rollout_ref.actor.teacher_kl_loss_coef"]
    for script in (CURRICULUM, TARGET):
        assert _effective(script)["actor_rollout_ref.actor.teacher_kl_loss_coef"] == want


def test_the_curriculum_gate_line_is_printed_and_nothing_aborts_on_it():
    """Advisory, like the tilt arm's: printed every step, acted on by a human."""
    src = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    block = src[src.index("[cross_teacher_curriculum] rho="):]
    block = block[: block.index("flush=True")]
    for key in ("target/tv", "target/live_frac", "target/abs_dkl_mean",
                "target/layer/shared/mass_share", "target/layer/pair/mass_share",
                "target/layer/shared/role/structural_share",
                "target/layer/pair/role/content_share",
                "target/retained_shuffled_ratio",
                "target/stage_kl/shared", "target/stage_kl/own",
                "target/entropy_delta", "target/tag_share",
                "target/max_abs_log_w", "target/mass_error_max"):
        assert key in block, key
    assert "raise" not in block and "assert" not in block


def test_the_arm_starts_against_its_own_lock():
    """Composed config + main_opd's injection + the lock, which is exactly the
    sequence a launch performs. The pairing tests above compare two composes and
    would not notice a lock that no run can satisfy -- and this arm's lock names
    three keys that reach the actor only through the injection, so it is the one
    that could.
    """
    from hydra import compose, initialize_config_dir
    from verl.trainer.main_opd import inject_distillation_config
    from verl.utils.expected_config import check_expected_config

    from tests.trainer.test_run_script_overrides_compose import _overrides

    arm = _effective(CURRICULUM)
    cfg_dir = os.path.join(REPO, "verl", "trainer", "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=_overrides(CURRICULUM))
    inject_distillation_config(config)
    mismatches = check_expected_config(
        config, os.path.join(REPO, arm["trainer.expected_config"]))
    assert mismatches == [], mismatches


def test_the_driver_ships_the_schedule_rather_than_the_actor_counting_steps():
    """Resume correctness, as a property of the code and not of a comment. If the
    actor ever derives the stage from a counter of its own, a run resumed at step
    150 re-runs stage 1 against a 150-step student and nothing says so."""
    driver = open(os.path.join(REPO, "verl", "trainer", "ppo", "opd_ray_trainer.py")).read()
    actor = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    assert 'batch.meta_info["cross_teacher_curriculum_rho"]' in driver
    assert "curriculum_rho(" in driver
    # The actor READS the schedule and cannot compute one: the function is not
    # among what it imports from the mechanism module. Checked on the import
    # block rather than on the whole file, because the name legitimately appears
    # in the assertion message that tells a future reader where rho comes from.
    import re

    imported = re.search(
        r"from verl\.trainer\.ppo\.cross_teacher_target import \(([^)]*)\)", actor
    ).group(1)
    assert 'meta_info.get("cross_teacher_curriculum_rho"' in actor
    assert "curriculum_rho" not in imported, imported
