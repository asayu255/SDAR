# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from types import SimpleNamespace  # Or use a mock object library

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import update_model_config


# Parametrize with different override scenarios
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
# [EXPLAIN] `test_update_model_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_update_model_config(override_kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_config = SimpleNamespace(param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me")
    # Apply the updates using the parametrized override_kwargs
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_nested = override_kwargs["nested_params"]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:  # Case 1: Plain override (nested params untouched)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"
