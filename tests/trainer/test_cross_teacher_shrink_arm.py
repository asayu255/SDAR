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
"""The shrink arm must differ from its comparators in the mechanism and nothing else.

Composed configs rather than script text, for the reason the other pairing tests
give: an injection in main_opd or a default moved in ppo_trainer.yaml never shows
in a diff of the arguments.

This arm's obligation is the opposite of the curriculum arm's. That one predicts
the END of the run agrees with the control, so its schedule has to fit inside the
run. This one MOVES THE FIXED POINT, so what has to be pinned is the single
number that says how far: lambda'.
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
SHRINK = "examples/opd_grpo_trainer/run_multitask_cross_teacher_shrink_qwen3.sh"
CURRICULUM = "examples/opd_grpo_trainer/run_multitask_cross_teacher_curriculum_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"

_MECHANISMS = ("algorithm.opd.cross_teacher_target.",
               "algorithm.opd.cross_teacher_kl_weight.",
               "actor_rollout_ref.actor.cross_teacher_target.",
               "actor_rollout_ref.actor.cross_teacher_kl_weight.")
# Where a run STOPS is operational, not scientific: total_training_steps is what
# defines the schedule and it is pinned. The lock deliberately does not pin this.
_OPERATIONAL = ("trainer.stop_after_steps", "env.search.search_url")


def _allowed(differing):
    allowed = set(IDENTITY) | set(_OPERATIONAL)
    allowed |= {k for k in differing if k.startswith(_MECHANISMS)}
    allowed |= {k for k in differing if "micro_batch_size" in k}
    return allowed


@pytest.mark.parametrize("other,name", [(CONTROL, "control"), (CURRICULUM, "curriculum arm")])
def test_the_shrink_arm_differs_only_in_the_mechanism(other, name):
    a = _effective(SHRINK)
    b = _effective(other)
    differing = _differing(a, b)
    assert differing <= _allowed(differing), sorted(differing - _allowed(differing))
    assert any(k.startswith("algorithm.opd.cross_teacher_target.") for k in differing), (
        f"the shrink arm and the {name} are identical"
    )


def test_the_arms_do_not_share_a_directory():
    """Shared default_local_dir + resume_mode=auto means the arm started second
    RESUMES FROM the first and reports it under its own name."""
    arm = _effective(SHRINK)
    for other in (CONTROL, CURRICULUM):
        b = _effective(other)
        for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir",
                    "trainer.sign_token_dump_dir", "trainer.project_name",
                    "trainer.experiment_name"):
            assert arm[key] != b[key], (key, other)


def test_the_mode_separates_it_from_the_other_target_mode():
    """Both arms are cross_teacher_target.enable=true. Equal modes would run the
    SAME mechanism under two names -- the one confusion no metric would reveal.
    They are opposite in sign: curriculum SUBTRACTS uncorroborated on-task shift
    and injects nothing, shrink ADDS the off-task voices."""
    arm = _effective(SHRINK)
    cur = _effective(CURRICULUM)
    assert arm["algorithm.opd.cross_teacher_target.enable"] is True
    assert arm["algorithm.opd.cross_teacher_target.mode"] == "shrink"
    assert cur["algorithm.opd.cross_teacher_target.mode"] == "curriculum"


def test_the_shrink_arm_carries_no_exponent_scale_and_no_schedule():
    """The tilt is already in the destination teacher's own nats, so the unit
    conversion has nothing to convert and the actor ASSERTS the key is absent.
    The schedule keys belong to the other mode and would be silently ignored."""
    arm = _effective(SHRINK)
    for key in ("exponent_scale", "stage_steps", "ramp_steps"):
        assert arm.get(f"algorithm.opd.cross_teacher_target.{key}", "<absent>") == "<absent>", key
    lock = yaml.safe_load(open(os.path.join(REPO, arm["trainer.expected_config"])))
    for key in ("exponent_scale", "stage_steps", "ramp_steps"):
        assert not any(key in k for k in lock), key


def test_the_lock_pins_lambda_prime_which_is_the_only_free_parameter():
    arm = _effective(SHRINK)
    lock = yaml.safe_load(open(os.path.join(REPO, arm["trainer.expected_config"])))
    for key in ("enable", "mode", "lambda_prime", "base_path"):
        assert f"algorithm.opd.cross_teacher_target.{key}" in lock, key
        # ...and on the actor, which is where the target is actually built
        assert f"actor_rollout_ref.actor.cross_teacher_target.{key}" in lock, key
    assert lock["algorithm.opd.cross_teacher_target.mode"] == "shrink"
    assert lock["algorithm.opd.cross_teacher_target.lambda_prime"] == 0.6
    assert lock["actor_rollout_ref.actor.cross_teacher_target.lambda_prime"] == 0.6


def test_lambda_prime_is_inside_the_admissible_range():
    """[1/K, 1] with K = 3 teachers, from the hierarchical model. Below 1/K the
    on-task teacher would carry LESS weight than an off-task one, which no
    posterior does -- it is the only teacher that observed this task's own
    component. lambda' = 1 would be the control under a different name."""
    arm = _effective(SHRINK)
    n_teachers = len([k for k in arm if k.startswith("algorithm.opd.teacher_paths.")])
    assert n_teachers == 3, n_teachers
    lam = arm["algorithm.opd.cross_teacher_target.lambda_prime"]
    assert 1.0 / n_teachers <= lam < 1.0, lam


def test_the_arm_starts_against_its_own_lock():
    """Composed config + main_opd's injection + the lock, which is exactly the
    sequence a launch performs. The pairing tests above compare two composes and
    would not notice a lock no run can satisfy -- and lambda_prime reaches the
    actor only through the injection, so this is the test that could."""
    from hydra import compose, initialize_config_dir
    from verl.trainer.main_opd import inject_distillation_config
    from verl.utils.expected_config import check_expected_config

    from tests.trainer.test_run_script_overrides_compose import _overrides

    arm = _effective(SHRINK)
    cfg_dir = os.path.join(REPO, "verl", "trainer", "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=_overrides(SHRINK))
    inject_distillation_config(config)
    mismatches = check_expected_config(
        config, os.path.join(REPO, arm["trainer.expected_config"]))
    assert mismatches == [], mismatches


def test_the_run_stops_on_a_checkpoint_step():
    """stop_after_steps is refused by the driver unless it is a multiple of
    save_freq: a run that stopped between checkpoints would have to redo the
    interval, and the arm is compared at the step it stopped on."""
    arm = _effective(SHRINK)
    stop = int(arm["trainer.stop_after_steps"])
    save = int(arm["trainer.save_freq"])
    assert stop > 0 and stop % save == 0, (stop, save)
    assert stop <= int(arm["trainer.total_training_steps"])


def test_the_driver_needs_no_per_step_state_for_this_mode():
    """lambda' is static, so -- unlike the curriculum's rho -- nothing has to
    ride in meta_info and a resumed run cannot restart a schedule it is halfway
    through. Checked as a property of the code, not of a comment."""
    driver = open(os.path.join(REPO, "verl", "trainer", "ppo", "opd_ray_trainer.py")).read()
    assert "shrink" not in driver
    actor = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    assert "lambda_prime" in actor


def test_the_actor_hands_this_mode_the_roles_it_splits_by():
    """The regression the first launch found. `roles` was gated on curriculum
    mode, so shrink got None and target/shrink/role/*_share came out of a live
    run as exactly 0.000 -- a structurally zero column reads as a measurement.
    Checked on the call site, because the unit test can only see what it passes.
    """
    src = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    call = src[src.index("xtt_stats.update("):]
    call = call[: call.index("xtt_token_stats is not None")]
    block = call[call.index("roles=("):]
    block = block[: block.index("),")]
    assert "shrink" in block, block
    assert "curriculum" in block, block
