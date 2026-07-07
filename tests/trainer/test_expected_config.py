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

Covers verl/utils/expected_config.py plus the self-consistency of EVERY
committed expectations file (discovered by glob under examples/): each file
must validate against a config assembled from its own keys (typo detection)
and must fail loudly when a knob drifts — the low_var_kl-instead-of-topk_kl
scenario. Branch-portable: the discovery makes this file identical across the
method branches regardless of which expectations files each carries.

Run:  python tests/trainer/test_expected_config.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf

from verl.utils.expected_config import (
    check_expected_config,
    enforce_expected_config,
    load_expectations,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXPECT_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "examples", "*", "expected_*.yaml")))


def _config_from_expectations(expect_file):
    """Assemble a nested OmegaConf config out of the flat dotted expectations."""
    config = OmegaConf.create({})
    for dotted_key, value in load_expectations(expect_file).items():
        OmegaConf.update(config, dotted_key, value, force_add=True)
    return config


def test_matching_config_passes():
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


def test_committed_expectations_files_self_consistent():
    """Every committed expectations file must (a) load, (b) validate against a
    config assembled from itself, and (c) fail if any knob drifts."""
    assert _EXPECT_FILES, "no expectations files found under examples/*/expected_*.yaml"
    for expect_file in _EXPECT_FILES:
        rel = os.path.relpath(expect_file, _REPO_ROOT)
        config = _config_from_expectations(expect_file)
        assert check_expected_config(config, expect_file) == [], rel

        expectations = load_expectations(expect_file)

        # OPD-family coupling: the algorithm-level KL type and the injected
        # actor-level value must agree with each other when both are pinned.
        kl_key = "algorithm.opd.kl_loss_type"
        actor_kl_key = "actor_rollout_ref.actor.teacher_kl_loss_type"
        if kl_key in expectations and actor_kl_key in expectations:
            assert expectations[kl_key] == expectations[actor_kl_key], rel

        # Drift ANY scalar key to a sentinel -> exactly that key must trip.
        drift_key = next(
            key for key, val in expectations.items() if not isinstance(val, (dict, list))
        )
        OmegaConf.update(config, drift_key, "___DRIFTED___", force_add=True)
        mismatched = {key for key, _, _ in check_expected_config(config, expect_file)}
        assert mismatched == {drift_key}, (rel, mismatched)
        print(f"PASS: {rel} self-consistent ({len(expectations)} keys), drift detected on {drift_key}")


if __name__ == "__main__":
    test_matching_config_passes()
    test_single_mismatch_fails_and_lists_key()
    test_missing_key_fails()
    test_committed_expectations_files_self_consistent()
    print("\nALL EXPECTED-CONFIG TESTS PASSED")
