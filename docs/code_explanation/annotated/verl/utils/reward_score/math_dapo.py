# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional


# [EXPLAIN] `last_boxed_only_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def last_boxed_only_string(string: str) -> Optional[str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx = string.rfind("\\boxed{")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if idx < 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    i = idx
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    right_brace_idx = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_left_braces_open = 0

    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while i < len(string):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if string[i] == "{":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            num_left_braces_open += 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if string[i] == "}":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            num_left_braces_open -= 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if num_left_braces_open == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                right_brace_idx = i
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        i += 1

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


# [EXPLAIN] `remove_boxed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_boxed(s: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    left = "\\boxed{"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert s[: len(left)] == left, f"box error: {s}"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert s[-1] == "}", f"box error: {s}"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return s[len(left) : -1]



# Constants for normalization
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


# [EXPLAIN] `normalize_final_answer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_final_answer(final_answer: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for before, after in SUBSTITUTIONS:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_answer = final_answer.replace(before, after)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for expr in REMOVED_EXPRESSIONS:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if final_answer.replace(",", "").isdigit():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_answer = final_answer.replace(",", "")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return final_answer.strip()


# [EXPLAIN] `is_correct_minerva` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> tuple[bool, str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    match = re.findall(answer_pattern, solution_str)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    extracted_answer = match[-1] if match else "[INVALID]"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if gt_need_extract:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gt = normalize_final_answer(gt)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (pred == gt), pred


# [EXPLAIN] `is_correct_strict_box` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pause_tokens_index is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(pause_tokens_index) == 4
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred = pred[pause_tokens_index[-1] - 100 :]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred = pred[-100:]

    # Extract and check the boxed answer
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    boxed_pred = last_boxed_only_string(pred)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return 1 if (extracted_pred == gt) else -1, extracted_pred


# [EXPLAIN] `verify` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def verify(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if strict_box_verify:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return correct == 1, pred

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    correct, pred = is_correct_minerva(solution_str, answer)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return correct, pred


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward = 1.0 if correct else -1.0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    acc = correct

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
