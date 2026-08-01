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
"""CPU-only tests for the expected-config fail-fast validator.

Covers verl/utils/expected_config.py plus the self-consistency of the
committed OPD+GRPO expectations file: the file must validate against a config
assembled from its own keys (typo detection) and must fail loudly when exactly
one knob drifts — the low_var_kl-instead-of-topk_kl scenario.

Run:  python tests/trainer/test_expected_config.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf

from verl.utils.expected_config import (
    check_expected_config,
    enforce_expected_config,
    load_expectations,
)

_EXPECT_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "opd_grpo_trainer",
        "expected_multitask_config.yaml",
    )
)


def _config_from_expectations(expect_file):
    """Assemble a nested OmegaConf config out of the flat dotted expectations."""
    config = OmegaConf.create({})
    for dotted_key, value in load_expectations(expect_file).items():
        OmegaConf.update(config, dotted_key, value, force_add=True)
    return config


def test_matching_config_passes(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        expect_file = os.path.join(tmp, "expect.yaml")
        with open(expect_file, "w") as f:
            f.write(
                '"a.b.loss_type": topk_kl\n'
                '"a.b.coef": 1.0\n'
                '"a.mini_batch": 60\n'
                '"a.flag": true\n'
                '"a.tasks": [x, y]\n'
            )
        config = OmegaConf.create(
            {"a": {"b": {"loss_type": "topk_kl", "coef": 1.0}, "mini_batch": 60, "flag": True, "tasks": ["x", "y"]}}
        )
        n = enforce_expected_config(config, expect_file, tag="test")
        assert n == 5
        # int/float equivalence: 60 vs 60.0 must not be a mismatch
        config.a.mini_batch = 60.0
        assert check_expected_config(config, expect_file) == []
    print("PASS: matching config passes (incl. int/float and list handling)")


def test_single_mismatch_fails_and_lists_key():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        expect_file = os.path.join(tmp, "expect.yaml")
        with open(expect_file, "w") as f:
            f.write('"algo.kl_loss_type": topk_kl\n"algo.coef": 1.0\n')
        # The original mishap: intent says topk_kl, effective config says low_var_kl.
        config = OmegaConf.create({"algo": {"kl_loss_type": "low_var_kl", "coef": 1.0}})
        try:
            enforce_expected_config(config, expect_file, tag="test")
        except AssertionError as e:
            msg = str(e)
            assert "algo.kl_loss_type" in msg and "low_var_kl" in msg and "topk_kl" in msg
        else:
            raise AssertionError("mismatch must raise")
    print("PASS: single drifted knob aborts with the key named")


def test_missing_key_fails():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        expect_file = os.path.join(tmp, "expect.yaml")
        with open(expect_file, "w") as f:
            f.write('"algo.kl_loss_type": topk_kl\n"algo.not_set_anywhere": 3\n')
        config = OmegaConf.create({"algo": {"kl_loss_type": "topk_kl"}})
        mismatches = check_expected_config(config, expect_file)
        assert len(mismatches) == 1 and mismatches[0][0] == "algo.not_set_anywhere"
    print("PASS: expectation key absent from config is a hard mismatch")


def test_committed_expectations_file_self_consistent():
    """The committed OPD+GRPO expectations file must (a) load, (b) validate
    against a config assembled from itself, and (c) fail if the KL type drifts."""
    config = _config_from_expectations(_EXPECT_FILE)
    assert check_expected_config(config, _EXPECT_FILE) == []

    # Sanity: the file pins the knob that caused the original mishap, and the
    # algorithm-level and injected actor-level values agree with each other.
    # (The run name itself is also pinned as a plain expectation key, so it is
    # allowed to be any string — drift protection comes from the file.)
    expectations = load_expectations(_EXPECT_FILE)
    kl_type = expectations["algorithm.opd.kl_loss_type"]
    assert kl_type == expectations["actor_rollout_ref.actor.teacher_kl_loss_type"]
    assert "trainer.experiment_name" in expectations

    # Drift the effective KL type (to whichever value is NOT committed) -> both
    # the algorithm and actor keys must trip.
    drift_type = "low_var_kl" if kl_type == "topk_kl" else "topk_kl"
    config.algorithm.opd.kl_loss_type = drift_type
    mismatched_keys = {key for key, _, _ in check_expected_config(config, _EXPECT_FILE)}
    assert mismatched_keys == {"algorithm.opd.kl_loss_type"}
    config.actor_rollout_ref.actor.teacher_kl_loss_type = drift_type
    mismatched_keys = {key for key, _, _ in check_expected_config(config, _EXPECT_FILE)}
    assert mismatched_keys == {
        "algorithm.opd.kl_loss_type",
        "actor_rollout_ref.actor.teacher_kl_loss_type",
    }
    print("PASS: committed expectations file is self-consistent and catches KL drift")


if __name__ == "__main__":
    test_matching_config_passes()
    test_single_mismatch_fails_and_lists_key()
    test_missing_key_fails()
    test_committed_expectations_file_self_consistent()
    print("\nALL EXPECTED-CONFIG TESTS PASSED")


def test_expectation_paths_expand_the_home_directory(tmp_path, monkeypatch):
    """A pinned teacher path must survive machines whose $HOME differs.

    The lock asserts *which* checkpoint the run distils from, not where the home
    directory happens to live, so the file may write "$HOME/..." and still match.
    """
    from verl.utils.expected_config import load_expectations

    monkeypatch.setenv("HOME", "/somewhere/else")
    expect = tmp_path / "expect.yaml"
    expect.write_text('"algorithm.opd.teacher_paths.alfworld": "$HOME/checkpoints/teachers/alfworld_step300"\n')

    loaded = load_expectations(str(expect))
    assert loaded["algorithm.opd.teacher_paths.alfworld"] == "/somewhere/else/checkpoints/teachers/alfworld_step300"

