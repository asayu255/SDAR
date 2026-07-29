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
import multiprocessing
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from concurrent.futures import ProcessPoolExecutor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# Import the function to be tested
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness

# Get SANDBOX_URL from environment variable
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SANDBOX_URL = os.environ.get("SANDBOX_FUSION_URL")
# Define skip condition and reason
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
skip_reason = "SANDBOX_FUSION_URL environment variable not set"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
skip_condition = not SANDBOX_URL

# --- Test code (for real API calls) ---
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CODE_SUCCESS = """
import sys
data = sys.stdin.read()
if data == 'input1':
    print('output1\\n', end='')
elif data == 'input2':
    print('output2\\n', end='')
else:
    print('unexpected input', end='')
"""

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CODE_WRONG_OUTPUT = """
print('wrong_output\\n', end='')
"""

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CODE_COMPILE_ERROR = """
a=b
"""

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CODE_RUNTIME_ERROR = """
import sys
print("About to raise error", file=sys.stderr)
raise ValueError("This is a runtime error")
"""

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CODE_TIMEOUT = """
import time
import sys
print("Sleeping...", file=sys.stderr)
time.sleep(10) # Sleep time should be longer than the timeout set in the test
print("Finished sleeping", file=sys.stderr)
"""

# --- Test input/output data ---
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INPUT_OUTPUT_VALID = {"inputs": ["input1", "input2"], "outputs": ["output1\n", "output2\n"]}

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INPUT_OUTPUT_SINGLE = {"inputs": ["input1"], "outputs": ["output1\n"]}

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INPUT_OUTPUT_MISMATCH = {"inputs": ["input1"], "outputs": ["output1\n", "output2\n"]}

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INPUT_OUTPUT_INVALID_MISSING_KEY = {"inputs": ["input1"]}

# --- Integration test cases (calling real API) ---


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_success_correct` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_success_correct():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Integration test: Code is correct, output is correct"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_SUCCESS)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [True, True]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["stdout"] == "output1\n"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["stdout"] == "output2\n"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_success_wrong_output` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_success_wrong_output():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Integration test: Code runs successfully, but output is wrong"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_WRONG_OUTPUT)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [False, False]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "wrong_answer"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["stdout"] == "wrong_output\n"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["status"] == "wrong_answer"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_compile_error` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_compile_error():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Integration test: Code causes compile error"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_COMPILE_ERROR, language="cpp")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-4, -4]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "compile_error"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["status"] == "compile_error"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_runtime_error` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_runtime_error():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Integration test: Code causes runtime error"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_RUNTIME_ERROR)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-2]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "runtime_error"
    # More assertions can be added based on the actual API response, e.g., exit_code, stderr


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_runtime_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_runtime_timeout():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Integration test: Code causes runtime timeout"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_timeout = 5  # Set a timeout shorter than the sleep time in CODE_TIMEOUT
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_TIMEOUT, timeout=test_timeout)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-3]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "timeout"
    # More assertions can be added based on the actual API response, e.g., run_status


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_concurrency_high_load` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_concurrency_high_load():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Integration test: High concurrency (100 cases) against real API with mixed results (success, wrong answer, timeout)"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    concurrency_level = 100
    # Indices for different expected outcomes
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wrong_answer_indices = {10, 25, 50}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout_indices = {5, 30, 60, 90}  # Indices where we expect a timeout

    # Generate 100 input/output pairs and code
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    high_load_inputs = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    high_load_outputs = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_results_map = {}  # Store expected result for each index

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(concurrency_level):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if i in timeout_indices:
            # Use a special input to trigger timeout in the code
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_inputs.append(f"input_timeout_{i}")
            # Output doesn't matter for timeout, but keep it consistent
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_outputs.append(f"output_{i}\n")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expected_results_map[i] = -3  # Expect timeout
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif i in wrong_answer_indices:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_inputs.append(f"input_{i}")
            # Intentionally set wrong expected output
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_outputs.append(f"wrong_output_{i}\n")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expected_results_map[i] = False  # Expect wrong answer
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_inputs.append(f"input_{i}")
            # Correct expected output
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            high_load_outputs.append(f"output_{i}\n")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expected_results_map[i] = True  # Expect success

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    high_load_in_outs = {"inputs": high_load_inputs, "outputs": high_load_outputs}

    # Code that handles normal inputs, and sleeps on specific "timeout" inputs
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    code_mixed_concurrent = """
import sys
import time
data = sys.stdin.read()
if data.startswith('input_timeout_'):
    time.sleep(20) # Sleep longer than the test timeout
    print(f"output_{data.split('_')[-1]}\\n", end='') # Still print something in case it finishes early
elif data.startswith('input_'):
    print(f"output_{data.split('_')[-1]}\\n", end='')
else:
    print("unknown_input\\n", end='')
"""
    # Set a reasonable timeout per case (must be less than the sleep time in the code)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_timeout = 15  # Allow slightly more time due to potential API load, but less than 20s sleep

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_time = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(
        SANDBOX_URL,
        high_load_in_outs,
        code_mixed_concurrent,  # Use the new code
        timeout=test_timeout,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    end_time = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    duration = end_time - start_time
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"\nHigh concurrency test ({concurrency_level} cases with {len(wrong_answer_indices)} wrong answers, {len(timeout_indices)} timeouts) duration: {duration:.2f} seconds")

    # Verify results against the expected map
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    correct_count = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wrong_count = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout_count = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unexpected_results = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, r in enumerate(results):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected = expected_results_map[i]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if r == expected:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if expected is True:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                correct_count += 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif expected is False:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                wrong_count += 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif expected == -3:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                timeout_count += 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            unexpected_results.append((i, r, f"Expected {expected}"))

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Correct results (True): {correct_count}/{concurrency_level - len(wrong_answer_indices) - len(timeout_indices)}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Expected wrong answers (False, correctly identified): {wrong_count}/{len(wrong_answer_indices)}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Expected timeouts (-3, correctly identified): {timeout_count}/{len(timeout_indices)}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if unexpected_results:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Unexpected results found:")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, res, expected_str in unexpected_results[:10]:  # Print first 10 unexpected
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"  Index {idx}: Got {res}, {expected_str}. Metadata: {metadata_list[idx]}")
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AssertionError(f"Found {len(unexpected_results)} unexpected results.")

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert correct_count == concurrency_level - len(wrong_answer_indices) - len(timeout_indices), "Incorrect number of successful results"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert wrong_count == len(wrong_answer_indices), "Incorrect number of identified wrong answers"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert timeout_count == len(timeout_indices), "Incorrect number of identified timeouts"

    # Verify metadata count and basic status of one of each type
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(metadata_list) == concurrency_level
    # Find the first correct index
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first_correct_index = next(i for i in range(concurrency_level) if i not in wrong_answer_indices and i not in timeout_indices)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_correct_index]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_correct_index]["stdout"] == f"output_{first_correct_index}\n"

    # Check the status of the first intentionally wrong case
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first_wrong_index = min(wrong_answer_indices)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_wrong_index]["status"] == "wrong_answer"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_wrong_index]["stdout"] == f"output_{first_wrong_index}\n"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_wrong_index]["expected_output"] == f"wrong_output_{first_wrong_index}\n"

    # Check the status of the first intentionally timeout case
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first_timeout_index = min(timeout_indices)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[first_timeout_index]["status"] == "timeout"
    # For timeout, stdout might be None or empty depending on when the timeout occurred
    # assert metadata_list[first_timeout_index]["stdout"] is None or metadata_list[first_timeout_index]["stdout"] == ""


# --- Unit test cases (using mock) ---


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
# [EXPLAIN] `test_unit_concurrency_order` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_unit_concurrency_order(mock_call_sandbox_api):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sandbox_url = "mock_url"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generation = "print(input())"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    language = "python"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout = 5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    in_outs = {"inputs": ["input1", "input2", "input3"], "outputs": ["output1", "output2", "output3"]}

    # [EXPLAIN] `side_effect` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def side_effect(*args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        stdin = kwargs.get("stdin")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if stdin == "input1":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif stdin == "input2":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(0.1)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output2", "return_code": 0}}, None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif stdin == "input3":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (None, "Unknown input in mock")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_call_sandbox_api.side_effect = side_effect

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [True, True, True]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(metadata_list) == 3
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["case_index"] == 0
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["case_index"] == 1
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[2]["case_index"] == 2
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[2]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert mock_call_sandbox_api.call_count == 3


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
# [EXPLAIN] `test_unit_api_timeout_error_concurrent` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_unit_api_timeout_error_concurrent(mock_call_sandbox_api):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sandbox_url = "mock_url"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generation = "print(input())"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    language = "python"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout = 5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    in_outs = {"inputs": ["input1", "input2_timeout", "input3"], "outputs": ["output1", "output2", "output3"]}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    api_error_message = "API Call Failed: Gateway Timeout (504) on attempt 3/3"

    # [EXPLAIN] `side_effect` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def side_effect(*args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        stdin = kwargs.get("stdin")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if stdin == "input1":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif stdin == "input2_timeout":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (None, api_error_message)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif stdin == "input3":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (None, "Unknown input in mock")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_call_sandbox_api.side_effect = side_effect

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [True, -1, True]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(metadata_list) == 3
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["status"] == "api_error"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[1]["api_request_error"] == api_error_message
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[2]["status"] == "success"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert mock_call_sandbox_api.call_count == 3


# --- Constants for the new concurrency test ---
# Define a low global concurrency limit to test the semaphore's effect
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_GLOBAL_CONCURRENCY_LIMIT_TEST = 5
# Define the number of processes used in the test
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
NUM_PROCESSES_TEST = 4
# Define the number of tasks processed by check_correctness in each process (i.e., internal ThreadPoolExecutor's concurrency potential)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
NUM_TASKS_PER_PROCESS_TEST = 3
# Simulate API call duration to ensure calls can overlap
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SIMULATED_API_CALL_DURATION_TEST = 0.2  # seconds


# --- Mock API call function for concurrency tracking ---
# This function will replace the real call_sandbox_api and use shared variables to track concurrency
# [EXPLAIN] `_mock_api_call_for_concurrency_tracking` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _mock_api_call_for_concurrency_tracking(
    active_calls_counter,  # multiprocessing.Value
    max_calls_tracker,  # multiprocessing.Value
    call_lock,  # multiprocessing.Lock
    # Standard call_sandbox_api parameters
    sandbox_fusion_url,
    code,
    stdin,
    compile_timeout,
    run_timeout,
    language,
):
    # entry_time = time.time() # For detailed logging
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with call_lock:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        active_calls_counter.value += 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if active_calls_counter.value > max_calls_tracker.value:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_calls_tracker.value = active_calls_counter.value
        # Optional debug log:
        # print(f"[PID:{os.getpid()}-TID:{threading.get_ident()}] API Call Start. Active: {active_calls_counter.value}, Max Observed: {max_calls_tracker.value}, Input: {stdin}")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(SIMULATED_API_CALL_DURATION_TEST)  # Simulate actual work duration

    # exit_time = time.time() # For detailed logging
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with call_lock:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        active_calls_counter.value -= 1
        # Optional debug log:
        # print(f"[PID:{os.getpid()}-TID:{threading.get_ident()}] API Call End. Active: {active_calls_counter.value}, Input: {stdin}, Duration: {exit_time - entry_time:.2f}s")

    # Return a simulated successful API response
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {"status": "Success", "run_result": {"status": "Finished", "stdout": f"mock_output_for_{stdin}", "return_code": 0}}, None


# --- Worker function for ProcessPoolExecutor ---
# This function runs in each child process of ProcessPoolExecutor
# [EXPLAIN] `_process_pool_worker_for_concurrency_test` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _process_pool_worker_for_concurrency_test(sandbox_url, in_outs, generation, language, timeout, mp_semaphore_for_check_correctness, active_calls_counter, max_calls_tracker, call_lock):
    # Corrected lambda to accept keyword arguments matching call_sandbox_api's usage
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    curried_mock_api_call = lambda sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, language: _mock_api_call_for_concurrency_tracking(active_calls_counter, max_calls_tracker, call_lock, sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, language)

    # ---- START DEBUG PRINTS ----
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import os

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import verl.utils.reward_score.sandbox_fusion.utils

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"[Worker PID:{os.getpid()}] Original call_sandbox_api: {verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}", flush=True)
    # ---- END DEBUG PRINTS ----

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api", side_effect=curried_mock_api_call) as mock_obj:
        # ---- START DEBUG PRINTS ----
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Worker PID:{os.getpid()}] Patched call_sandbox_api: {verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}", flush=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[Worker PID:{os.getpid()}] Mock object: {mock_obj}", flush=True)
        # ---- END DEBUG PRINTS ----
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_url,
            in_outs=in_outs,
            generation=generation,
            timeout=timeout,
            language=language,
            concurrent_semaphore=mp_semaphore_for_check_correctness,  # Pass multiprocessing.Semaphore
        )
        # print(f"Process {os.getpid()} finished check_correctness. Processed {len(results)} tasks.")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return len(results)  # Return the number of processed tasks for basic validation


# --- The actual test case for multiprocess concurrency control ---
# [EXPLAIN] `test_multiprocess_global_concurrency_limit_with_semaphore` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_multiprocess_global_concurrency_limit_with_semaphore():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tests that the global concurrent_semaphore (multiprocessing.Semaphore)
    correctly limits the number of concurrent calls to call_sandbox_api
    across multiple processes, each potentially running multiple threads
    via check_correctness's internal ThreadPoolExecutor.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    manager = multiprocessing.Manager()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    active_calls_counter = manager.Value("i", 0)  # Current active mock API calls
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_calls_tracker = manager.Value("i", 0)  # Observed maximum concurrent mock API calls
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    call_lock = manager.Lock()  # Lock to protect counters

    # Create a multiprocessing.Semaphore instance, this is the global semaphore we are testing.
    # It will be passed to check_correctness and used by _process_single_case to limit calls to call_sandbox_api.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_mp_semaphore = manager.Semaphore(MAX_GLOBAL_CONCURRENCY_LIMIT_TEST)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_sandbox_url = "mock_url_for_concurrency_test"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_generation = "pass"  # Specific code content is not important as API call is mocked
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_language = "python"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mock_timeout = 5  # Timeout setting, not critical for mock calls

    # Input/output data for each process
    # NUM_TASKS_PER_PROCESS_TEST tasks will be handled by check_correctness's internal ThreadPoolExecutor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    process_in_outs = {"inputs": [f"task_input_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)], "outputs": [f"task_output_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)]}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    futures = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_tasks_expected_to_run = NUM_PROCESSES_TEST * NUM_TASKS_PER_PROCESS_TEST

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_start_time = time.time()

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES_TEST) as executor:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(NUM_PROCESSES_TEST):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = executor.submit(
                _process_pool_worker_for_concurrency_test,  # Worker function
                mock_sandbox_url,
                process_in_outs,
                mock_generation,
                mock_language,
                mock_timeout,
                global_mp_semaphore,  # Global semaphore to test
                active_calls_counter,  # Shared variables for tracking
                max_calls_tracker,
                call_lock,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

    # Wait for all processes to complete and collect results
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_tasks_processed_per_worker = [f.result() for f in futures]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_end_time = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_execution_time = test_end_time - test_start_time

    # Print some test statistics for debugging and validation
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\n--- Global Concurrency Test Stats ---")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Semaphore Limit (MAX_GLOBAL_CONCURRENCY_LIMIT_TEST): {MAX_GLOBAL_CONCURRENCY_LIMIT_TEST}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Number of Processes (NUM_PROCESSES_TEST): {NUM_PROCESSES_TEST}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Tasks per Process (NUM_TASKS_PER_PROCESS_TEST): {NUM_TASKS_PER_PROCESS_TEST}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Total Tasks Submitted: {total_tasks_expected_to_run}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Simulated API Call Duration: {SIMULATED_API_CALL_DURATION_TEST}s")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Total Test Execution Time: {total_execution_time:.2f}s")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Max Concurrent Mock API Calls Observed: {max_calls_tracker.value}")
    # print(f"Tasks processed per worker: {num_tasks_processed_per_worker}")

    # Verify that all submitted tasks have been processed
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert sum(num_tasks_processed_per_worker) == total_tasks_expected_to_run, "Mismatch in the number of tasks processed."

    # Verify that the mock API was called at least once
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_calls_tracker.value > 0, "The mocked API call_sandbox_api was not called."

    # Core assertion: Observed maximum concurrent calls should not exceed the semaphore's limit
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_calls_tracker.value <= MAX_GLOBAL_CONCURRENCY_LIMIT_TEST, f"Observed concurrency ({max_calls_tracker.value}) exceeded semaphore limit ({MAX_GLOBAL_CONCURRENCY_LIMIT_TEST})."

    # Optional: Rough check on execution time to verify semaphore is working to limit concurrency
    # Theoretical minimum execution time = (Total tasks / Concurrency limit) * Single task duration
    # Actual time will be longer due to various overheads
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_expected_duration = (total_tasks_expected_to_run * SIMULATED_API_CALL_DURATION_TEST) / MAX_GLOBAL_CONCURRENCY_LIMIT_TEST
    # print(f"Minimum Expected Execution Time (approx): {min_expected_duration:.2f}s")
    # Allow some margin, e.g., 80% of theoretical minimum time
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert total_execution_time >= min_expected_duration * 0.8, f"Total execution time ({total_execution_time:.2f}s) was unexpectedly short, suggesting the semaphore might not be effectively limiting concurrency as expected (min expected: {min_expected_duration * 0.8:.2f}s)."


# Ensure there is no more code after this point if these were the last functions.
# If there was other code, it would follow here.
# [EXPLAIN] `test_unit_invalid_input_format` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_unit_invalid_input_format():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Unit test: Invalid in_outs format passed"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, None, CODE_SUCCESS)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-1]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["error"] == "Invalid input/output data"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, {}, CODE_SUCCESS)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-1]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["error"] == "Invalid input/output data"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_INVALID_MISSING_KEY, CODE_SUCCESS)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-1]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["error"] == "Invalid input/output data"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_unit_input_output_mismatch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_unit_input_output_mismatch():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Unit test: Mismatch between the number of inputs and outputs"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_MISMATCH, CODE_SUCCESS)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [-1]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(metadata_list) == 1
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["error"] == "Input/output count mismatch"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_integration_concurrency_all_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_integration_concurrency_all_timeout():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Integration test: High concurrency (100 cases) against real API, all causing timeout"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    concurrency_level = 100
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    code_infinite_loop = """
def knight_moves(X, Y):
    MOD = 10**9 + 7
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    dp[0][0] = 1
    for i in range(1, X + 1):
        for j in range(1, Y + 1):
            dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD
    return dp[X][Y]

def solve():
    X, Y = map(int, input().split())
    print(knight_moves(X, Y))

if __name__ == "__main__":
    solve()
    """

    # Generate 100 simple input/output pairs (content doesn't matter)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout_inputs = ["324 384429" for i in range(concurrency_level)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout_outputs = [f"output_{i}\n" for i in range(concurrency_level)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timeout_in_outs = {"inputs": timeout_inputs, "outputs": timeout_outputs}

    # Set a timeout for the test cases
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_timeout = 10  # Set a timeout value

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_time = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, timeout_in_outs, code_infinite_loop, timeout=test_timeout)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    end_time = time.time()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    duration = end_time - start_time
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"\nHigh concurrency all timeout test ({concurrency_level} cases) duration: {duration:.2f} seconds")

    # Verify all results are -3 (timeout)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_timed_out = all(r == -3 for r in results)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not all_timed_out:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_timeout_indices = [i for i, r in enumerate(results) if r != -3]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Indices that did not time out: {non_timeout_indices}")
        # Print metadata for the first few non-timeout cases for debugging
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in non_timeout_indices[:5]:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Metadata for non-timeout case {i}: {metadata_list[i]}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert all_timed_out, f"Not all {concurrency_level} concurrent tests resulted in timeout (-3). Results: {results}"

    # Verify metadata count and status of the first case
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(metadata_list) == concurrency_level
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0]["status"] == "timeout"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(skip_condition, reason=skip_reason)
# [EXPLAIN] `test_fn_name_success_single_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_fn_name_success_single_case():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests successful execution for a single test case with fn_name.
    from livecodebench/code_generation_lite test 510
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generation_code = """
class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)

        x_positions = positions[x]
        answer = []
        for k in queries:
            if k > len(x_positions):
                answer.append(-1)
            else:
                answer.append(x_positions[k-1])
        return answer
"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    in_outs = {
        "fn_name": "occurrencesOfElement",
        "inputs": ["[1, 3, 1, 7]\n[1, 3, 2, 4]\n1", "[1, 2, 3]\n[10]\n5"],
        "outputs": ["[0, -1, 2, -1]", "[-1]"],
    }

    # Use a short timeout for fast tests
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results, metadata_list = check_correctness(SANDBOX_URL, in_outs, generation_code, timeout=5)
    # from verl.utils.reward_score.prime_code import apps_check_correctness
    # results, metadata_list = apps_check_correctness(in_outs=in_outs, generation=generation_code, timeout=50000, debug=True)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert results == [True, True]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "error" not in metadata_list[0]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0].get("status") != "compilation error"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert metadata_list[0].get("status") != "runtime error"
