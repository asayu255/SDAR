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

"""The three arms, and the claim that they differ in nothing but b.

The comparison is only worth running if that is true, and it is the kind of
thing that stops being true one careless edit later -- so it is checked here
rather than asserted in a comment.
"""

import os
import re

import pytest

from verl.trainer.main_opd import KL_COEF_BY_TASK_BOX, validate_kl_coef_by_task
from verl.utils.expected_config import load_expectations

ARMS = ("control", "uniform", "redistribute")
EXPECT = "examples/opd_grpo_trainer/expected_multitask_opd_coef_{arm}_config.yaml"
SCRIPT = "examples/opd_grpo_trainer/run_multitask_opd_coef_qwen3.sh"

# The design's numbers (docs/opd_coefficient_arm_design.md, step 300, N=8).
B = {
    "control": None,
    "uniform": {"alfworld": 1.110833, "search": 1.110833, "webshop": 1.110833},
    "redistribute": {"alfworld": 1.076431, "search": 1.191101, "webshop": 0.5},
}
CALIBRATION_D = {"alfworld": 0.4563, "search": 0.8859, "webshop": 0.4084}


def _flat(arm):
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    return load_expectations(EXPECT.format(arm=arm))


def _is_coef_key(key):
    # the parent key AND its per-task children, on both sides of the injection
    return "kl_loss_coef_by_task" in key


# ---------------------------------------------------------------------------
# 1. the arms differ in b and in their own name, and in nothing else


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_the_lock_files_differ_in_nothing_but_the_coefficient(arm):
    control, other = _flat("control"), _flat(arm)
    allowed = {"trainer.experiment_name"}
    differing = {
        k for k in set(control) | set(other)
        if control.get(k, "<absent>") != other.get(k, "<absent>")
    }
    unexpected = {k for k in differing if k not in allowed and not _is_coef_key(k)}
    assert not unexpected, f"{arm} differs from control outside b: {sorted(unexpected)}"
    # and it really does differ in b -- a test that passes because both files
    # are identical would be worse than no test.
    assert any(_is_coef_key(k) for k in differing)


def test_the_control_arm_leaves_the_key_unset_rather_than_setting_it_to_one():
    control = _flat("control")
    assert control["actor_rollout_ref.actor.teacher_kl_loss_coef_by_task"] is None
    assert control["algorithm.opd.kl_loss_coef_by_task"] is None
    assert not any(k.endswith((".alfworld", ".search", ".webshop")) and _is_coef_key(k)
                   for k in control)


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_the_pinned_coefficients_are_the_design_numbers_on_both_sides(arm):
    flat = _flat(arm)
    for task, want in B[arm].items():
        assert flat[f"actor_rollout_ref.actor.teacher_kl_loss_coef_by_task.{task}"] == want
        assert flat[f"algorithm.opd.kl_loss_coef_by_task.{task}"] == want


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_pins_the_readout_and_the_step_count(arm):
    flat = _flat(arm)
    assert flat["algorithm.opd.task_diag"] is True
    assert flat["actor_rollout_ref.actor.teacher_kl_task_diag"] is True
    assert flat["trainer.total_training_steps"] == 150
    # unchanged from the control recipe this arm is derived from
    assert flat["actor_rollout_ref.actor.teacher_kl_loss_coef"] == 0.01
    assert flat["algorithm.opd.normalize_loss_by_task"] is True
    assert flat["actor_rollout_ref.actor.cross_teacher_kl_weight.enable"] is False


# ---------------------------------------------------------------------------
# 2. the rule's own arithmetic, on the numbers the files pin


def test_the_redistributed_vector_holds_the_calibration_budget():
    """sum_j q_j b_j = 1, with q the gradient-norm shares. The property the
    whole design is built on, checked against the pinned values rather than
    against a note in a document."""
    d = CALIBRATION_D
    total = sum(d.values())
    q = {k: v / total for k, v in d.items()}
    b = B["redistribute"]
    assert sum(q[k] * b[k] for k in d) == pytest.approx(1.0, abs=5e-4)
    # the linear budget is the same statement in unnormalised form
    assert sum(b[k] * d[k] for k in d) == pytest.approx(sum(d.values()), rel=5e-4)


def test_the_uniform_arm_does_not_hold_that_budget_and_is_not_meant_to():
    d = CALIBRATION_D
    b = B["uniform"]
    ratio = sum(b[k] * d[k] for k in d) / sum(d.values())
    assert ratio == pytest.approx(1.110833, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. the run script


def _script():
    return open(SCRIPT).read()


@pytest.mark.parametrize("arm", ARMS)
def test_the_script_offers_exactly_these_arms(arm):
    s = _script()
    assert re.search(rf"^\s+{arm}\)\s*$", s, re.M), f"ARM={arm} has no case branch"


def test_the_script_refuses_an_unknown_arm():
    s = _script()
    assert "ARM must be control | uniform | redistribute" in s
    assert "exit 1" in s.split("esac")[0]


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_the_script_passes_the_pinned_vector(arm):
    """Parsed and compared as numbers, so a lost decimal cannot pass as a match."""
    branch = _script().split(f"    {arm})")[1].split(";;")[0]
    m = re.search(r"kl_loss_coef_by_task=\{([^}]*)\}", branch)
    assert m, f"{arm}: no coefficient vector in its branch"
    got = {k.strip(): float(v) for k, v in
           (pair.split(":") for pair in m.group(1).split(","))}
    assert got == B[arm]
    assert got == {t: _flat(arm)[f"algorithm.opd.kl_loss_coef_by_task.{t}"] for t in got}


def test_the_script_selects_the_lock_by_arm_and_turns_the_readout_on():
    s = _script()
    assert 'expected_multitask_opd_coef_${ARM}_config.yaml' in s
    assert "+algorithm.opd.task_diag=True" in s
    assert '"${OPD_COEF_ARGS[@]}"' in s
    # the control branch must pass nothing at all
    control_branch = s.split("    control)")[1].split(";;")[0]
    assert "OPD_COEF_ARGS=()" in control_branch
    assert "kl_loss_coef_by_task" not in control_branch.replace(
        "teacher_kl_loss_coef_by_task stays null", "")


def test_the_script_and_the_lock_agree_on_the_step_count():
    s = _script()
    assert "trainer.total_training_steps=150" in s
    # and the data prep is deliberately NOT re-cut at 150
    assert "--total_training_steps 300" in s


# ---------------------------------------------------------------------------
# 4. the startup gate


@pytest.mark.parametrize("arm", ("uniform", "redistribute"))
def test_startup_validation_accepts_the_arms(arm):
    assert validate_kl_coef_by_task(B[arm]) is not None


def test_startup_validation_accepts_unset():
    assert validate_kl_coef_by_task(None) is None


@pytest.mark.parametrize("bad", [
    {"alfworld": 1.6},          # above the box
    {"alfworld": 0.4},          # below it
    {"alfworld": float("nan")},
    {"alfworld": float("inf")},
    {"alfworld": "1.0"},        # a string coefficient would reach the row tensor
    {"alfworld": True},         # bool is an int in python, and is not a coefficient
    {},                         # an empty map is not the uniform arm
])
def test_startup_validation_refuses_what_would_become_a_different_arm(bad):
    with pytest.raises(ValueError):
        validate_kl_coef_by_task(bad)


def test_the_box_is_the_designs_box_and_not_a_mean_one_constraint():
    assert KL_COEF_BY_TASK_BOX == (0.5, 1.5)
    # sum(b) = 2.767 for the redistributed arm: a mean-1 check would reject it
    assert sum(B["redistribute"].values()) == pytest.approx(2.767532, abs=1e-5)
    assert validate_kl_coef_by_task(B["redistribute"]) is not None
