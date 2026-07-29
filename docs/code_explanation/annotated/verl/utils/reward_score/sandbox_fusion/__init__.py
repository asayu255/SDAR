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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import traceback

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .utils import check_correctness

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(sandbox_fusion_url, concurrent_semaphore, completion, test_cases, continuous=False, timeout=10):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    solution = completion
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "```python" in completion:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        solution = completion.split("```python")[-1].split("```")[0]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parts = completion.split("```")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(parts) >= 2:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "\n" in solution:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                first_line, rest = solution.split("\n", 1)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if first_line.strip().isalpha():  # Simple check for language name
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    solution = rest
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(test_cases, dict):
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                test_cases = json.loads(test_cases)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except json.JSONDecodeError as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.error(f"Failed to parse test_cases JSON: {e}")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error("Invalid test_cases structure.")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res_list, metadata_list = check_correctness(sandbox_fusion_url=sandbox_fusion_url, in_outs=test_cases, generation=solution, timeout=timeout, concurrent_semaphore=concurrent_semaphore)

        # Calculate score
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not res_list:  # If there are no results (e.g., invalid input)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0.0, metadata_list

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if continuous:
            # Calculate pass rate for the first N (e.g., 10) test cases
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_to_consider = min(len(res_list), 10)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if num_to_consider == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score = 0.0
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score = passed_count / num_to_consider
            # Return all metadata, even if score is based on the first N
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_metadata = metadata_list
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # Calculate pass rate for all test cases
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            passed_count = sum(1 for r in res_list if r is True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_cases = len(res_list)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            score = passed_count / total_cases if total_cases > 0 else 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_metadata = metadata_list

    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Error during compute_score: {e}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        traceback.print_exc()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Ensure float and list are returned
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
