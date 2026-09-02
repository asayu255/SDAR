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
    # The support choice is part of the mechanism, not a drift: design C7 puts
    # the target arm on the ON-TASK TEACHER's top-k so p_tilde is a
    # student-independent fixed point, while both comparators run
    # student-indexed. This is a real second difference against the control and
    # it is accepted on a measurement, not on faith: the teachertopk sign arm
    # changed exactly this key against ITS student-indexed sibling and every one
    # of the seven state fractions and mass fractions moved by less than 0.006
    # (docs/multitask_signweight_teachertopk_150step_report.md), with the
    # training-rollout episode metrics indistinguishable.
    allowed |= {k for k in differing if k.endswith("student_indexed_topk")}
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
    assert lock["algorithm.opd.cross_teacher_target.exponent_scale"] == 1.0


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
                "target/tag_share", "target/max_abs_log_w", "target/mass_error_max"):
        assert key in block, key
    # Nothing in the arm raises or exits on a gate value.
    where = src.index("if xtt_on:")
    tail = src[where : where + 4000]
    for banned in ("raise RuntimeError", "sys.exit", "assert_all_finite({\n                \"target"):
        assert banned not in tail, banned
