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
"""The two notice arms must differ from each other in the text alone, and from
the control in the notice and the ladder alone.

Composed configs, not script text, for the reason the other pairing tests give.
This contrast has one obligation the others do not: its control is a PLACEBO
carrying the same avoid-list, so the only admissible difference between the two
arms is `variant`, the three text hashes and the run identity. Anything else
differing would mean the contrast measures something other than the framing.
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
from verl.trainer.ppo import privileged_notice as pn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAMED = "examples/opd_grpo_trainer/run_multitask_privileged_notice_named_qwen3.sh"
PLACEBO = "examples/opd_grpo_trainer/run_multitask_privileged_notice_placebo_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"

_NOTICE = ("algorithm.opd.privileged_notice.", "actor_rollout_ref.actor.privileged_notice.")
_LADDER = ("algorithm.opd.transfer_ladder.", "actor_rollout_ref.actor.transfer_ladder.")


def _allowed(differing, prefixes):
    allowed = set(IDENTITY)
    allowed |= {k for k in differing if k.startswith(prefixes)}
    allowed |= {k for k in differing if "micro_batch_size" in k}
    return allowed


def test_named_and_placebo_differ_in_the_text_and_identity_only():
    a, b = _effective(NAMED), _effective(PLACEBO)
    differing = _differing(a, b)
    allowed = _allowed(differing, _NOTICE)
    assert differing <= allowed, sorted(differing - allowed)
    # and WITHIN the notice block, only variant and the hashes may differ
    for k in differing:
        if k.startswith(_NOTICE):
            leaf = k.split(".privileged_notice.")[1]
            assert leaf == "variant" or leaf.startswith("doc_sha256"), k
    assert a["algorithm.opd.privileged_notice.variant"] == "named"
    assert b["algorithm.opd.privileged_notice.variant"] == "placebo"


@pytest.mark.parametrize("arm", [NAMED, PLACEBO])
def test_each_arm_differs_from_the_control_only_in_notice_and_ladder(arm):
    a, c = _effective(arm), _effective(CONTROL)
    differing = _differing(a, c)
    allowed = _allowed(differing, _NOTICE + _LADDER)
    assert differing <= allowed, sorted(differing - allowed)
    assert any(k.startswith(_NOTICE) for k in differing), "the arm and the control are identical"


def test_the_arms_do_not_share_a_directory():
    arms = {p: _effective(p) for p in (NAMED, PLACEBO, CONTROL)}
    for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir",
                "trainer.sign_token_dump_dir", "trainer.experiment_name"):
        vals = [arms[p][key] for p in arms]
        assert len(set(vals)) == 3, (key, vals)


def test_no_context_cap_was_moved():
    """Design section 3.2, as corrected at implementation: the per-turn cap is the
    global one and no override changes between the arms and the control."""
    c = _effective(CONTROL)
    for arm in (NAMED, PLACEBO):
        a = _effective(arm)
        for key in ("data.max_prompt_length", "data.truncation",
                    "data.task_overrides.alfworld.max_prompt_length",
                    "data.task_overrides.webshop.max_prompt_length",
                    "data.task_overrides.search.max_prompt_length"):
            assert a[key] == c[key], (arm, key)


def test_the_locks_pin_the_hashes_of_the_texts_in_code():
    for arm, variant in ((NAMED, "named"), (PLACEBO, "placebo")):
        eff = _effective(arm)
        lock = yaml.safe_load(open(os.path.join(REPO, eff["trainer.expected_config"])))
        pinned = lock["algorithm.opd.privileged_notice.doc_sha256"]
        assert set(pinned) == set(pn.TASKS)
        for task, pfx in pinned.items():
            assert pn.notice_sha256(variant, task).startswith(str(pfx)), (variant, task)
        assert lock["algorithm.opd.privileged_notice.variant"] == variant
        assert lock["algorithm.opd.privileged_notice.apply_to"] == ["student"]
        assert lock["algorithm.opd.transfer_ladder.enable"] is True
        # the weighting arms stay OFF on both
        assert lock["algorithm.opd.cross_teacher_kl_weight.enable"] is False


def test_the_script_hashes_match_the_locks():
    """The script passes the pins as a Hydra dict; the lock pins the same dict.
    A drift between the two would be caught at launch, but here is cheaper."""
    for arm in (NAMED, PLACEBO):
        eff = _effective(arm)
        lock = yaml.safe_load(open(os.path.join(REPO, eff["trainer.expected_config"])))
        for task in pn.TASKS:
            assert str(eff[f"algorithm.opd.privileged_notice.doc_sha256.{task}"]) == str(
                lock["algorithm.opd.privileged_notice.doc_sha256"][task]), (arm, task)


@pytest.mark.parametrize("arm", [NAMED, PLACEBO])
def test_the_arm_starts_against_its_own_lock(arm):
    """Composed config + main_opd's injection + the lock: the launch sequence."""
    from hydra import compose, initialize_config_dir
    from verl.trainer.main_opd import inject_distillation_config
    from verl.utils.expected_config import check_expected_config

    from tests.trainer.test_run_script_overrides_compose import _overrides

    eff = _effective(arm)
    cfg_dir = os.path.join(REPO, "verl", "trainer", "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=_overrides(arm))
    inject_distillation_config(config)
    mismatches = check_expected_config(config, os.path.join(REPO, eff["trainer.expected_config"]))
    assert mismatches == [], mismatches
    # and the injected actor copy parses as a valid notice config
    nc = pn.parse_notice_config(config.actor_rollout_ref.actor.privileged_notice)
    assert nc is not None and nc.to_student and not nc.to_teacher
    pn.verify_doc_hashes(nc)


def test_the_diagnostics_are_wired_and_named_as_the_design_says():
    actor = open(os.path.join(REPO, "verl", "workers", "actor", "dp_actor.py")).read()
    driver = open(os.path.join(REPO, "verl", "trainer", "ppo", "opd_ray_trainer.py")).read()
    assert 'metrics["notice/effect_kl"]' in actor          # section 4, diagnostic 1
    assert 'f"notice/leak_rate/{task}"' in driver          # section 4, diagnostic 2
    assert 'out["notice/truncated_frac"]' in driver         # section 3.2's floor
    assert "ladder_stats is not None and ladder_enabled and not sign_enabled" in actor
    # teacher mode is plumbed even though student runs first: both on-task call
    # sites go through the prefix, and the worker corrects the fingerprint.
    assert driver.count("self._with_teacher_notice(sub, task)") == 2
    worker = open(os.path.join(REPO, "verl", "workers", "fsdp_workers.py")).read()
    assert "fingerprints - fingerprint_adjust" in worker
