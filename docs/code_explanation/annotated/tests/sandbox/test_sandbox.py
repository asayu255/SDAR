# Copyright 2024 PRIME team and/or its affiliates
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
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score import default_compute_score, prime_code, sandbox_fusion
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score.prime_code import apps_check_correctness
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.reward_manager.prime import parallel_compute_score_async

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
prime_math_answers = [
    """\\begin{bmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19 \n \\end{bmatrix}""",
    """\\frac{\\sqrt{505}}{7}""",
    """x^2 + y^2 + 4x - 6y + 13""",
]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
prime_math_gts = [
    """\\begin{pmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19\n \\end{pmatrix}""",  # mat test
    """\\frac{\\sqrt{505}}{7}""",  # frac test
    """(x + 2)^2 + (y - 3)^2 """,  # symbolic test
]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
prime_code_answers = [
    """import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    
    # Read start and target positions
    x0, y0, x1, y1 = int(next(it)), int(next(it)), int(next(it)), int(next(it))
    
    n = int(next(it))
    allowed = set()
    # The total number of allowed cells is at most 10^5.
    for _ in range(n):
        r = int(next(it))
        a = int(next(it))
        b = int(next(it))
        for c in range(a, b + 1):
            allowed.add((r, c))
    
    # Directions for the king (8 neighboring cells)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    start = (x0, y0)
    target = (x1, y1)
    
    # BFS initialization
    queue = deque()
    queue.append((x0, y0, 0))
    # Mark the starting cell as visited by removing it from allowed set.
    allowed.discard(start)
    
    while queue:
        x, y, moves = queue.popleft()
        if (x, y) == target:
            print(moves)
            return
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in allowed:
                allowed.remove((nx, ny))
                queue.append((nx, ny, moves + 1))
    
    print(-1)

if __name__ == '__main__':
    main()
"""
] * 2
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
prime_code_gts = [
    """{\n \"inputs\": [\n \"5 7 6 11\\n3\\n5 3 8\\n6 7 11\\n5 2 5\\n\",\n \"3 4 3 10\\n3\\n3 1 4\\n4 5 9\\n3 10 10\\n\",\n \"1 1 2 10\\n2\\n1 1 3\\n2 6 10\\n\",\n \"9 8 7 8\\n9\\n10 6 6\\n10 6 6\\n7 7 8\\n9 5 6\\n8 9 9\\n9 5 5\\n9 8 8\\n8 5 6\\n9 10 10\\n\",\n \"6 15 7 15\\n9\\n6 15 15\\n7 14 14\\n6 15 15\\n9 14 14\\n7 14 16\\n6 15 15\\n6 15 15\\n7 14 14\\n8 15 15\\n\",\n \"13 16 20 10\\n18\\n13 16 16\\n20 10 10\\n19 10 10\\n12 15 15\\n20 10 10\\n18 11 11\\n19 10 10\\n19 10 10\\n20 10 10\\n19 10 10\\n20 10 10\\n20 10 10\\n19 10 10\\n18 11 11\\n13 16 16\\n12 15 15\\n19 10 10\\n19 10 10\\n\",\n \"89 29 88 30\\n16\\n87 31 31\\n14 95 95\\n98 88 89\\n96 88 88\\n14 97 97\\n13 97 98\\n100 88 88\\n88 32 32\\n99 88 89\\n90 29 29\\n87 31 31\\n15 94 96\\n89 29 29\\n88 32 32\\n97 89 89\\n88 29 30\\n\",\n \"30 14 39 19\\n31\\n35 7 11\\n37 11 12\\n32 13 13\\n37 5 6\\n46 13 13\\n37 14 14\\n31 13 13\\n43 13 19\\n45 15 19\\n46 13 13\\n32 17 17\\n41 14 19\\n30 14 14\\n43 13 17\\n34 16 18\\n44 11 19\\n38 13 13\\n40 12 20\\n37 16 18\\n46 16 18\\n34 10 14\\n36 9 10\\n36 15 19\\n38 15 19\\n42 13 19\\n33 14 15\\n35 15 19\\n33 17 18\\n39 12 20\\n36 5 7\\n45 12 12\\n\",\n \"2 1 1 1\\n2\\n1 1 2\\n2 1 2\\n\",\n \"1 1 1 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\",\n \"1 1 1000000000 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\"\n ],\n \"outputs\": [\n \"4\\n\",\n \"6\\n\",\n \"-1\\n\",\n \"2\\n\",\n \"1\\n\",\n \"-1\\n\",\n \"1\\n\",\n \"9\\n\",\n \"1\\n\",\n \"1\\n\",\n \"-1\\n\"\n ]\n}""",  # A correct sample # noqa: E501
    """{\n \"inputs\": [\n \"5 7 6 11\\n3\\n5 3 8\\n6 7 11\\n5 2 5\\n\",\n \"3 4 3 10\\n3\\n3 1 4\\n4 5 9\\n3 10 10\\n\",\n \"1 1 2 10\\n2\\n1 1 3\\n2 6 10\\n\",\n \"9 8 7 8\\n9\\n10 6 6\\n10 6 6\\n7 7 8\\n9 5 6\\n8 9 9\\n9 5 5\\n9 8 8\\n8 5 6\\n9 10 10\\n\",\n \"6 15 7 15\\n9\\n6 15 15\\n7 14 14\\n6 15 15\\n9 14 14\\n7 14 16\\n6 15 15\\n6 15 15\\n7 14 14\\n8 15 15\\n\",\n \"13 16 20 10\\n18\\n13 16 16\\n20 10 10\\n19 10 10\\n12 15 15\\n20 10 10\\n18 11 11\\n19 10 10\\n19 10 10\\n20 10 10\\n19 10 10\\n20 10 10\\n20 10 10\\n19 10 10\\n18 11 11\\n13 16 16\\n12 15 15\\n19 10 10\\n19 10 10\\n\",\n \"89 29 88 30\\n16\\n87 31 31\\n14 95 95\\n98 88 89\\n96 88 88\\n14 97 97\\n13 97 98\\n100 88 88\\n88 32 32\\n99 88 89\\n90 29 29\\n87 31 31\\n15 94 96\\n89 29 29\\n88 32 32\\n97 89 89\\n88 29 30\\n\",\n \"30 14 39 19\\n31\\n35 7 11\\n37 11 12\\n32 13 13\\n37 5 6\\n46 13 13\\n37 14 14\\n31 13 13\\n43 13 19\\n45 15 19\\n46 13 13\\n32 17 17\\n41 14 19\\n30 14 14\\n43 13 17\\n34 16 18\\n44 11 19\\n38 13 13\\n40 12 20\\n37 16 18\\n46 16 18\\n34 10 14\\n36 9 10\\n36 15 19\\n38 15 19\\n42 13 19\\n33 14 15\\n35 15 19\\n33 17 18\\n39 12 20\\n36 5 7\\n45 12 12\\n\",\n \"2 1 1 1\\n2\\n1 1 2\\n2 1 2\\n\",\n \"1 1 1 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\",\n \"1 1 1000000000 2\\n5\\n1000000000 1 10000\\n19920401 1188 5566\\n1000000000 1 10000\\n1 1 10000\\n5 100 200\\n\"\n ],\n \"outputs\": [\n \"4\\n\",\n \"6\\n\",\n \"-1\\n\",\n \"-1\\n\",\n \"1\\n\",\n \"-1\\n\",\n \"1\\n\",\n \"9\\n\",\n \"1\\n\",\n \"1\\n\",\n \"-1\\n\"\n ]\n}""",  # noqa: E501
]  # A failed sample with first several in-out passed

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
prime_code_scores = [1.0, 0.9]


# [EXPLAIN] `test_parallelism` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_parallelism():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Test if process pool works properly
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequences_str = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_sources = []
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while len(sequences_str) < 32:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sequences_str.extend(prime_code_answers)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ground_truth.extend(prime_code_gts)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data_sources.extend(["codecontests"] * len(prime_code_answers))

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sequences_str.extend(prime_math_answers)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ground_truth.extend(prime_math_gts)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data_sources.extend(["numina_aops_forum"] * len(prime_math_answers))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = asyncio.run(parallel_compute_score_async(default_compute_score, sequences_str, ground_truth, data_sources, num_processes=16))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(scores)


# [EXPLAIN] `test_prime_code` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prime_code():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Test PRIME code sandbox.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source = "codecontests"
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for completion, ground_truth, score_ in zip(prime_code_answers, prime_code_gts, prime_code_scores):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        score = default_compute_score(data_source, completion, ground_truth)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert float(score) == score_


# Use the pytest.mark.skipif decorator to skip the test
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(not os.environ.get("SANDBOX_FUSION_URL"), reason="SANDBOX_FUSION_URL environment variable not set")
# [EXPLAIN] `test_prime_code_sandbox_fusion` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prime_code_sandbox_fusion():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Test PRIME code on sandbox fusion. Skips if SANDBOX_FUSION_URL is not set.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source = "codecontests"
    # Get the URL from the environment variable, as skipif ensures it is set at this point
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sandbox_fusion_url = os.environ.get("SANDBOX_FUSION_URL")
    # Removed the previous 'if not sandbox_url' check block

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for completion, ground_truth, score_ in zip(prime_code_answers, prime_code_gts, prime_code_scores):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        score = default_compute_score(data_source, completion, ground_truth, extra_info={"sandbox_fusion_url": sandbox_fusion_url})  # <-- Use the URL obtained from the environment variable
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert float(score) == score_


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skipif(not os.environ.get("SANDBOX_FUSION_URL"), reason="SANDBOX_FUSION_URL environment variable not set")
# [EXPLAIN] `test_continuous_score_consistency` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_continuous_score_consistency():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Verify that continuous score calculation is consistent between prime_code and sandbox_fusion.
    Uses a test case where the first 9 out of 11 sub-cases pass (expected score 0.9).
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    completion = prime_code_answers[1]  # Use the second sample
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = prime_code_gts[1]  # Use the second sample (9/11 pass, first 9 pass)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_continuous_score = 0.9

    # 1. Calculate score using prime_code (default) with continuous=True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prime_score, _ = sandbox_fusion.compute_score(os.environ.get("SANDBOX_FUSION_URL"), None, completion, ground_truth, continuous=True)

    # 2. Calculate score using sandbox_fusion with continuous=True
    # Ensure the extra_info key triggers the sandbox_fusion path in default_compute_score
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fusion_score, _ = prime_code.compute_score(completion, ground_truth, continuous=True)

    # 3. Assert scores are equal (using pytest.approx for float comparison)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert float(prime_score) == pytest.approx(expected_continuous_score)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert float(fusion_score) == pytest.approx(expected_continuous_score)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert float(prime_score) == pytest.approx(float(fusion_score))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Continuous Score (Prime Code): {prime_score}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Continuous Score (Sandbox Fusion): {fusion_score}")


# [EXPLAIN] `test_check_correctness` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_check_correctness():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    completion = prime_code_answers[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = json.loads(prime_code_gts[0])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_single = {"inputs": ground_truth["inputs"][:1], "outputs": ground_truth["outputs"][:1]}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    res, meta = apps_check_correctness(in_outs=ground_truth_single, generation=completion, timeout=5, debug=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(res, meta)


# [EXPLAIN] `test_prime_math` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prime_math():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source = "numina_aops_forum"
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for completion, ground_truth in zip(prime_math_answers, prime_math_gts):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        score = default_compute_score(data_source, completion, ground_truth)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert float(score) == 1.0
