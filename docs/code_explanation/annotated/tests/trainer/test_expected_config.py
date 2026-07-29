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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""CPU-only tests for the expected-config fail-fast validator.

Covers verl/utils/expected_config.py plus the self-consistency of EVERY
committed expectations file (discovered by glob under examples/): each file
must validate against a config assembled from its own keys (typo detection)
and must fail loudly when a knob drifts — the low_var_kl-instead-of-topk_kl
scenario. Branch-portable: the discovery makes this file identical across the
method branches regardless of which expectations files each carries.

Run:  python tests/trainer/test_expected_config.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import glob
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.expected_config import (
    check_expected_config,
    enforce_expected_config,
    load_expectations,
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_EXPECT_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "examples", "*", "expected_*.yaml")))


# [EXPLAIN] `_config_from_expectations` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _config_from_expectations(expect_file):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Assemble a nested OmegaConf config out of the flat dotted expectations."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = OmegaConf.create({})
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dotted_key, value in load_expectations(expect_file).items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        OmegaConf.update(config, dotted_key, value, force_add=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return config


# [EXPLAIN] `test_matching_config_passes` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_matching_config_passes():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import tempfile

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tempfile.TemporaryDirectory() as tmp:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expect_file = os.path.join(tmp, "expect.yaml")
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(expect_file, "w") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write(
                '"a.b.loss_type": topk_kl\n'
                '"a.b.coef": 1.0\n'
                '"a.mini_batch": 60\n'
                '"a.flag": true\n'
                '"a.tasks": [x, y]\n'
            )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = OmegaConf.create(
            {"a": {"b": {"loss_type": "topk_kl", "coef": 1.0}, "mini_batch": 60, "flag": True, "tasks": ["x", "y"]}}
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n = enforce_expected_config(config, expect_file, tag="test")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert n == 5
        # int/float equivalence: 60 vs 60.0 must not be a mismatch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config.a.mini_batch = 60.0
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert check_expected_config(config, expect_file) == []
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: matching config passes (incl. int/float and list handling)")


# [EXPLAIN] `test_single_mismatch_fails_and_lists_key` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_single_mismatch_fails_and_lists_key():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import tempfile

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tempfile.TemporaryDirectory() as tmp:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expect_file = os.path.join(tmp, "expect.yaml")
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(expect_file, "w") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write('"algo.kl_loss_type": topk_kl\n"algo.coef": 1.0\n')
        # The original mishap: intent says topk_kl, effective config says low_var_kl.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = OmegaConf.create({"algo": {"kl_loss_type": "low_var_kl", "coef": 1.0}})
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            enforce_expected_config(config, expect_file, tag="test")
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except AssertionError as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            msg = str(e)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert "algo.kl_loss_type" in msg and "low_var_kl" in msg and "topk_kl" in msg
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise AssertionError("mismatch must raise")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: single drifted knob aborts with the key named")


# [EXPLAIN] `test_missing_key_fails` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_missing_key_fails():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import tempfile

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tempfile.TemporaryDirectory() as tmp:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expect_file = os.path.join(tmp, "expect.yaml")
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(expect_file, "w") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write('"algo.kl_loss_type": topk_kl\n"algo.not_set_anywhere": 3\n')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = OmegaConf.create({"algo": {"kl_loss_type": "topk_kl"}})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mismatches = check_expected_config(config, expect_file)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(mismatches) == 1 and mismatches[0][0] == "algo.not_set_anywhere"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: expectation key absent from config is a hard mismatch")


# [EXPLAIN] `test_committed_expectations_files_self_consistent` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_committed_expectations_files_self_consistent():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Every committed expectations file must (a) load, (b) validate against a
    config assembled from itself, and (c) fail if any knob drifts."""
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert _EXPECT_FILES, "no expectations files found under examples/*/expected_*.yaml"
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for expect_file in _EXPECT_FILES:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rel = os.path.relpath(expect_file, _REPO_ROOT)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = _config_from_expectations(expect_file)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert check_expected_config(config, expect_file) == [], rel

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expectations = load_expectations(expect_file)

        # OPD-family coupling: the algorithm-level KL type and the injected
        # actor-level value must agree with each other when both are pinned.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kl_key = "algorithm.opd.kl_loss_type"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_kl_key = "actor_rollout_ref.actor.teacher_kl_loss_type"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if kl_key in expectations and actor_kl_key in expectations:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert expectations[kl_key] == expectations[actor_kl_key], rel

        # Drift ANY scalar key to a sentinel -> exactly that key must trip.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        drift_key = next(
            key for key, val in expectations.items() if not isinstance(val, (dict, list))
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        OmegaConf.update(config, drift_key, "___DRIFTED___", force_add=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mismatched = {key for key, _, _ in check_expected_config(config, expect_file)}
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mismatched == {drift_key}, (rel, mismatched)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"PASS: {rel} self-consistent ({len(expectations)} keys), drift detected on {drift_key}")


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_matching_config_passes()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_single_mismatch_fails_and_lists_key()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_missing_key_fails()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_committed_expectations_files_self_consistent()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\nALL EXPECTED-CONFIG TESTS PASSED")
