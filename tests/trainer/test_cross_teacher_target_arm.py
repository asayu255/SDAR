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
"""The target arm must differ from its comparators in the mechanism and nothing else.

Composed configs, not script text, for the reason the signweight pairing test
gives: an injection in main_opd or a default moved in ppo_trainer.yaml never
shows in a diff of the arguments. The audit's costliest finding was a
control/treatment pair that was not a pair (docs/cross_teacher_kl_weight_offline_audit.md
section 0.2); these tests are that finding, made permanent.
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
TARGET = "examples/opd_grpo_trainer/run_multitask_cross_teacher_target_qwen3.sh"
KLW = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"

_MECHANISMS = ("algorithm.opd.cross_teacher_target.",
               "algorithm.opd.cross_teacher_kl_weight.",
               "actor_rollout_ref.actor.cross_teacher_target.",
               "actor_rollout_ref.actor.cross_teacher_kl_weight.")


def _mechanism_or_identity(differing):
    allowed = set(IDENTITY)
    allowed |= {k for k in differing if k.startswith(_MECHANISMS)}
    allowed |= {k for k in differing if "micro_batch_size" in k}
    # There is NO exception for student_indexed_topk any more. The arm carried
    # one while design C7 held (a teacher-indexed support, so p_tilde would not
    # depend on the student); C7 was withdrawn on 2026-09-02 and the arm is back
    # on the student's top-k, which both comparators already use. The difference
    # is therefore the mechanism keys alone -- and this test failing because
    # somebody re-flipped the support is the point of not re-adding the
    # exception.
    return allowed


@pytest.mark.parametrize("other,name", [(CONTROL, "control"), (KLW, "klw arm")])
def test_the_target_arm_differs_only_in_the_mechanism(other, name):
    a = _effective(TARGET)
    b = _effective(other)
    differing = _differing(a, b)
    assert differing <= _mechanism_or_identity(differing), sorted(
        differing - _mechanism_or_identity(differing)
    )
    assert any(k.startswith("algorithm.opd.cross_teacher_target.") for k in differing), (
        f"the target arm and the {name} are identical"
    )


def test_the_arms_do_not_share_a_directory():
    """Shared default_local_dir + resume_mode=auto means the arm started second
    RESUMES FROM the first and reports it under its own name."""
    target = _effective(TARGET)
    for other in (CONTROL, KLW):
        b = _effective(other)
        for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir",
                    "trainer.sign_token_dump_dir", "trainer.project_name"):
            assert target[key] != b[key], (key, other)


def test_the_lock_pins_every_knob_of_the_mechanism():
    """The mechanism has exactly one knob and the lock must hold it. A knob the
    lock does not pin is one a future edit can move silently -- and this arm's
    whole claim of being parameter-light rests on the knob being visible."""
    arm = _effective(TARGET)
    lock = yaml.safe_load(open(os.path.join(REPO, arm["trainer.expected_config"])))
    for key in ("enable", "base_path", "exponent_scale"):
        assert f"algorithm.opd.cross_teacher_target.{key}" in lock, key
    assert lock["algorithm.opd.cross_teacher_target.enable"] is True
    # 1.0. It was 2.148 for one commit, on an offline estimate that was retracted
    # (the yaml records why); pinning the value here is what makes a repeat of
    # that round trip a deliberate edit rather than a drift.
    assert lock["algorithm.opd.cross_teacher_target.exponent_scale"] == 1.0


def test_the_support_matches_the_comparators():
    """The mechanism reads every model at the STUDENT's top-k (C7 withdrawn), and
    the arm is only comparable to the klw arm and the control if all three agree
    on it -- the support decides what the coarse-grained KL is exact on, so a
    mismatch compares two different objectives."""
    for script in (TARGET, KLW, CONTROL):
        assert _effective(script)["actor_rollout_ref.actor.student_indexed_topk"] is True, script


def test_the_two_mechanisms_cannot_be_enabled_together():
    a = _effective(TARGET)
    assert a.get("algorithm.opd.cross_teacher_target.enable") is True
    assert a.get("algorithm.opd.cross_teacher_kl_weight.enable", False) in (False, None, "<absent>")


def test_the_gate_line_is_printed_and_nothing_aborts_on_it():
    """The gates are advisory (decided 2026-09-02): printed every step, acted on
    by a human. A future edit that turns one back into a raise would make the run
    stop on a number the design says to read, not to obey."""
    src = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    block = src[src.index("[cross_teacher_target] gates (advisory)"):]
    block = block[: block.index("flush=True")]
    for key in ("target/tv", "target/shuffled_tv_ratio", "target/acted_novelty",
                "target/tag_share", "target/max_abs_log_w", "target/mass_error_max",
                # The revision's own readings. abs_dkl decides the scale,
                # entropy_delta is the only channel that tracked the accuracy
                # gain, dkl_kl_corr is the focal test the predecessor failed, and
                # clamped_mass_frac is the saturation reading the count ratio
                # understates -- none of them is useful in wandb alone.
                "target/abs_dkl_mean", "target/entropy_delta",
                "target/dkl_kl_corr", "target/clamped_mass_frac"):
        assert key in block, key
    # Nothing in the arm raises or exits on a gate value.
    where = src.index("if xtt_on:")
    tail = src[where : where + 4000]
    for banned in ("raise RuntimeError", "sys.exit", "assert_all_finite({\n                \"target"):
        assert banned not in tail, banned
