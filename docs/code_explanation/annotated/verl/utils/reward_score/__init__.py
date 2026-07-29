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
# from . import gsm8k, math, prime_math, prime_code

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.import_utils import deprecated


# [EXPLAIN] `default_compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if data_source == "openai/gsm8k":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import gsm8k

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = gsm8k.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import math

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import math_dapo

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = math_dapo.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import prime_math

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = prime_math.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sandbox_fusion_url:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in ["hiyouga/geometry3k"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import geo3k

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = geo3k.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from . import search_r1_like_qa_em

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(res, dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return res
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(res, (int, float, bool)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return float(res)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return float(res[0])


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@deprecated("verl.utils.reward_score.default_compute_score")
# [EXPLAIN] `_default_compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["default_compute_score"]
