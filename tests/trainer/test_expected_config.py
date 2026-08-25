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

import contextlib
import glob
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf

from verl.utils.expected_config import (
    check_expected_config,
    enforce_expected_config,
    load_expectations,
    waived_keys,
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


def test_a_null_expectation_is_not_confused_with_an_absent_key():
    """``env.search.max_retries: null`` is a pinned value, not a missing one.

    ``OmegaConf.select`` returns its ``default`` when a key is absent, so a
    sentinel that happened to be ``None`` would make "pinned to null" and "not in
    the config at all" indistinguishable -- and the second is exactly the failure
    the lock exists to catch.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        expect_file = os.path.join(tmp, "expect.yaml")
        with open(expect_file, "w") as f:
            f.write('"env.search.max_retries": null\n')

        assert check_expected_config(OmegaConf.create({"env": {"search": {"max_retries": None}}}), expect_file) == []

        drifted = check_expected_config(OmegaConf.create({"env": {"search": {"max_retries": 10}}}), expect_file)
        assert len(drifted) == 1 and drifted[0][1] == 10

        absent = check_expected_config(OmegaConf.create({"env": {"search": {}}}), expect_file)
        assert len(absent) == 1, "an absent key must still be reported"
    print("PASS: null expectation distinguishes pinned-null from absent")


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

        # Same coupling for the sign-weight knobs. They are authored under
        # algorithm.opd with the other scientific settings and copied onto the
        # actor by main_opd, because the weights are built inside the training
        # forward; the two copies disagreeing would mean the run used one value
        # and the lock recorded the other.
        for knob in ("enable", "mode", "agree_weight", "agree_neg_weight",
                     "disagree_weight", "deadzone"):
            a = f"algorithm.opd.sign_weight.{knob}"
            b = f"actor_rollout_ref.actor.sign_weight.{knob}"
            if a in expectations and b in expectations:
                assert expectations[a] == expectations[b], (rel, knob)

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


def _waive(keys):
    """Set EXPECTED_CONFIG_WAIVE for one call; restore whatever was there.

    Plain os.environ rather than monkeypatch so these run under the __main__
    runner below as well as under pytest, which is the convention this file
    already follows.
    """
    previous = os.environ.get("EXPECTED_CONFIG_WAIVE")
    if keys is None:
        os.environ.pop("EXPECTED_CONFIG_WAIVE", None)
    else:
        os.environ["EXPECTED_CONFIG_WAIVE"] = keys
    return previous


def _restore(previous):
    if previous is None:
        os.environ.pop("EXPECTED_CONFIG_WAIVE", None)
    else:
        os.environ["EXPECTED_CONFIG_WAIVE"] = previous


def _waiver_fixture(tmp_dir, **over):
    """A two-key expectations file and a config that matches it unless told not to."""
    path = os.path.join(tmp_dir, "expect.yaml")
    with open(path, "w") as handle:
        handle.write('"trainer.save_freq": 25\n"trainer.total_training_steps": 300\n')
    trainer = {"save_freq": 25, "total_training_steps": 300}
    trainer.update(over)
    return OmegaConf.create({"trainer": trainer}), path


def test_waived_key_no_longer_fails():
    """A probe may move a pinned knob when it names it.

    The lock exists so a comparison run cannot be silently invalidated, and
    editing the expectations file to unblock a probe is the failure it is meant
    to prevent -- that edit is the production run's intent, and it is the kind
    somebody forgets to revert. Naming the key in the invocation keeps the
    exception in one shell line and in the log the run gets judged from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config, expect_file = _waiver_fixture(tmp, save_freq=1)
        previous = _waive("trainer.save_freq")
        try:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                checked = enforce_expected_config(config, expect_file)
        finally:
            _restore(previous)

    out = captured.getvalue()
    assert "WAIVED trainer.save_freq" in out, out
    assert "running with 1" in out and "says 25" in out, out
    assert checked == 1, "the waived key must not count as one still being checked"
    print("PASS: a named waiver is honoured and printed")


def test_waiver_does_not_disarm_other_keys():
    """Otherwise a probe escape becomes a blanket off switch."""
    with tempfile.TemporaryDirectory() as tmp:
        config, expect_file = _waiver_fixture(tmp, save_freq=1, total_training_steps=10)
        previous = _waive("trainer.save_freq")
        try:
            try:
                enforce_expected_config(config, expect_file)
            except AssertionError as exc:
                assert "total_training_steps" in str(exc), str(exc)
                assert "save_freq" not in str(exc), str(exc)
            else:
                raise AssertionError("a drifted key survived alongside a waiver")
        finally:
            _restore(previous)
    print("PASS: waiving one key still enforces the rest")


def test_waiver_for_an_unpinned_key_is_an_error():
    """A waiver that protects nothing is a typo, and belongs to the same family
    as the typo'd ``+key=`` override this module already catches."""
    with tempfile.TemporaryDirectory() as tmp:
        config, expect_file = _waiver_fixture(tmp, save_freq=1)
        previous = _waive("trainer.save_freq,trainer.tpyo")
        try:
            try:
                enforce_expected_config(config, expect_file)
            except AssertionError as exc:
                assert "tpyo" in str(exc), str(exc)
            else:
                raise AssertionError("a waiver naming an unpinned key was accepted")
        finally:
            _restore(previous)
    print("PASS: a waiver naming an unpinned key is rejected")


def test_no_waiver_is_unchanged_behaviour():
    with tempfile.TemporaryDirectory() as tmp:
        previous = _waive(None)
        try:
            drifted, expect_file = _waiver_fixture(tmp, save_freq=1)
            try:
                enforce_expected_config(drifted, expect_file)
            except AssertionError as exc:
                assert "save_freq" in str(exc), str(exc)
            else:
                raise AssertionError("drift passed with no waiver set")

            clean, _ = _waiver_fixture(tmp)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                assert enforce_expected_config(clean, expect_file) == 2
            assert "waived" not in captured.getvalue()
        finally:
            _restore(previous)
    print("PASS: with no waiver set the validator behaves exactly as before")


def test_waivers_accept_commas_or_spaces():
    for raw in ("a.b c.d", "a.b,c.d", " a.b , c.d "):
        previous = _waive(raw)
        try:
            assert waived_keys() == ["a.b", "c.d"], raw
        finally:
            _restore(previous)
    print("PASS: waiver lists accept commas or spaces")
