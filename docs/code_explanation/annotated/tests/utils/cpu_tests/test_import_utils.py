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
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.import_utils import load_extern_type

# Path to the test module
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), "test_module.py")


# [EXPLAIN] `test_load_extern_type_class` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_class():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test loading a class from an external file"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    TestClass = load_extern_type(TEST_MODULE_PATH, "TestClass")

    # Verify the class was loaded correctly
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert TestClass is not None
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert TestClass.__name__ == "TestClass"

    # Test instantiation and functionality
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    instance = TestClass()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert instance.value == "default"

    # Test with a custom value
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    custom_instance = TestClass("custom")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert custom_instance.get_value() == "custom"


# [EXPLAIN] `test_load_extern_type_function` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_function():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test loading a function from an external file"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_function = load_extern_type(TEST_MODULE_PATH, "test_function")

    # Verify the function was loaded correctly
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert test_function is not None
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert callable(test_function)

    # Test function execution
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result = test_function()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result == "test_function_result"


# [EXPLAIN] `test_load_extern_type_constant` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_constant():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test loading a constant from an external file"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    constant = load_extern_type(TEST_MODULE_PATH, "TEST_CONSTANT")

    # Verify the constant was loaded correctly
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert constant is not None
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert constant == "test_constant_value"


# [EXPLAIN] `test_load_extern_type_nonexistent_file` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_nonexistent_file():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test behavior when file doesn't exist"""
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(FileNotFoundError):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        load_extern_type("/nonexistent/path.py", "SomeType")


# [EXPLAIN] `test_load_extern_type_nonexistent_type` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_nonexistent_type():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test behavior when type doesn't exist in the file"""
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(AttributeError):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        load_extern_type(TEST_MODULE_PATH, "NonExistentType")


# [EXPLAIN] `test_load_extern_type_none_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_none_path():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test behavior when file path is None"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result = load_extern_type(None, "SomeType")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result is None


# [EXPLAIN] `test_load_extern_type_invalid_module` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_load_extern_type_invalid_module():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test behavior when module has syntax errors"""
    # Create a temporary file with syntax errors
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import tempfile

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as temp_file:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        temp_file.write("This is not valid Python syntax :")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temp_path = temp_file.name

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with pytest.raises(RuntimeError):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_extern_type(temp_path, "SomeType")
    # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
    finally:
        # Clean up the temporary file
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.exists(temp_path):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.remove(temp_path)
