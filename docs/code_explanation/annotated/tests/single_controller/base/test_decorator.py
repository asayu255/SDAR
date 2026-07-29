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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.single_controller.base.decorator as decorator_module
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import DISPATCH_MODE_FN_REGISTRY, Dispatch, _check_dispatch_mode, get_predefined_dispatch_fn, register_dispatch_mode, update_dispatch_mode


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.fixture
# [EXPLAIN] `reset_dispatch_registry` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reset_dispatch_registry():
    # Store original state
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    original_registry = DISPATCH_MODE_FN_REGISTRY.copy()
    # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
    yield
    # Reset registry after test
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    decorator_module.DISPATCH_MODE_FN_REGISTRY.clear()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    decorator_module.DISPATCH_MODE_FN_REGISTRY.update(original_registry)


# [EXPLAIN] `test_register_new_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_register_new_dispatch_mode(reset_dispatch_registry):
    # Test registration
    # [EXPLAIN] `dummy_dispatch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dummy_dispatch(worker_group, *args, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return args, kwargs

    # [EXPLAIN] `dummy_collect` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dummy_collect(worker_group, output):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    register_dispatch_mode("TEST_MODE", dummy_dispatch, dummy_collect)

    # Verify enum extension
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _check_dispatch_mode(Dispatch.TEST_MODE)

    # Verify registry update
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert get_predefined_dispatch_fn(Dispatch.TEST_MODE) == {"dispatch_fn": dummy_dispatch, "collect_fn": dummy_collect}
    # Clean up
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.remove("TEST_MODE")


# [EXPLAIN] `test_update_existing_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_update_existing_dispatch_mode(reset_dispatch_registry):
    # Store original implementation
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    original_mode = Dispatch.ONE_TO_ALL

    # New implementations
    # [EXPLAIN] `new_dispatch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def new_dispatch(worker_group, *args, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return args, kwargs

    # [EXPLAIN] `new_collect` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def new_collect(worker_group, output):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # Test update=
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    update_dispatch_mode(original_mode, new_dispatch, new_collect)

    # Verify update
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert get_predefined_dispatch_fn(original_mode)["dispatch_fn"] == new_dispatch
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert get_predefined_dispatch_fn(original_mode)["collect_fn"] == new_collect
