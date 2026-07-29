# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import string


# [EXPLAIN] `normalize_answer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_answer(s):
    # [EXPLAIN] `remove_articles` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def remove_articles(text):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return re.sub(r"\b(a|an|the)\b", " ", text)

    # [EXPLAIN] `white_space_fix` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def white_space_fix(text):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return " ".join(text.split())

    # [EXPLAIN] `remove_punc` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def remove_punc(text):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        exclude = set(string.punctuation)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "".join(ch for ch in text if ch not in exclude)

    # [EXPLAIN] `lower` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def lower(text):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return text.lower()

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# [EXPLAIN] `em_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def em_check(prediction, golden_answers):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(golden_answers, str):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        golden_answers = [golden_answers]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    normalized_prediction = normalize_answer(prediction)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    score = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for golden_answer in golden_answers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        golden_answer = normalize_answer(golden_answer)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if golden_answer == normalized_prediction:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            score = 1
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return score


# [EXPLAIN] `subem_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def subem_check(prediction, golden_answers):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(golden_answers, str):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        golden_answers = [golden_answers]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    normalized_prediction = normalize_answer(prediction)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    score = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for golden_answer in golden_answers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        golden_answer = normalize_answer(golden_answer)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if golden_answer in normalized_prediction:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            score = 1
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return score


# [EXPLAIN] `extract_solution` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def extract_solution(solution_str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer_pattern = r"<answer>(.*?)</answer>"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    matches = list(match)

    # If there are 0  matches, return None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(matches) < 1:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # If there are 2 or more matches, return the last one
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return matches[-1].group(1).strip()


# [EXPLAIN] `count_answer_tags` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def count_answer_tags(text):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    opening_tags = text.count("<answer>")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    closing_tags = text.count("</answer>")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return opening_tags, closing_tags


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer = extract_solution(solution_str=solution_str)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    open_count, close_count = count_answer_tags(solution_str)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    do_print = random.randint(1, 64) == 1

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if do_print:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("--------------------------------")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Golden answers: {ground_truth['target']}")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if answer is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Extracted answer is not None: {answer}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Extracted answer: None!")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Solution string: {solution_str}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if answer is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if em_check(answer, ground_truth["target"]):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score = score / 4
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return score
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return score
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return format_score


# [EXPLAIN] `compute_score_subem` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer = extract_solution(solution_str=solution_str)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    do_print = random.randint(1, 64) == 1

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if do_print:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("--------------------------------")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Golden answers: {ground_truth['target']}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Extracted answer: {answer}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Solution string: {solution_str}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if answer is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if subem_check(answer, ground_truth["target"]):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return score
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return format_score
