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
import re


# [EXPLAIN] `extract_solution` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def extract_solution(solution_str, method="strict"):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert method in ["strict", "flexible"]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if method == "strict":
        # this also tests the formatting of the model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if solution is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_answer = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_answer = solution.group(0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif method == "flexible":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_answer = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(answer) == 0:
            # no reward is there is no answer
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            invalid_str = ["", "."]
            # find the last number that is not '.'
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for final_answer in reversed(answer):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if final_answer not in invalid_str:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return final_answer


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer = extract_solution(solution_str=solution_str, method=method)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if answer is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if answer == ground_truth:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return score
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return format_score
