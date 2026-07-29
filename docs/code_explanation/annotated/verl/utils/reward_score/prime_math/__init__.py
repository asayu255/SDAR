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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).

FROM: https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import contextlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sympy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pylatexenc import latex2text
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sympy.parsing import sympy_parser

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import timeout_limit

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from . import math_normalize
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .grader import math_equal

# import math_normalize
# from grader import math_equal

# sympy might hang -- we don't care about trying to be lenient in these cases
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BAD_SUBSTRINGS = ["^{", "^("]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TUPLE_CHARS = "()[]"

# [EXPLAIN] `_sympy_parse` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _sympy_parse(expr: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Parses an expression with sympy."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    py_expr = expr.replace("^", "**")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


# [EXPLAIN] `_parse_latex` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _parse_latex(expr: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Attempts to parse latex to an expression sympy can read."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("\\tfrac", "\\frac")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("\\dfrac", "\\frac")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("√", "sqrt")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("π", "pi")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("∞", "inf")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("∪", "U")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("·", "*")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("×", "*")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return expr.strip()


# [EXPLAIN] `_is_float` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _is_float(num: str) -> bool:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        float(num)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ValueError:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] `_is_int` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _is_int(x: float) -> bool:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return abs(x - int(round(x))) <= 1e-7
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] `_is_frac` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _is_frac(expr: str) -> bool:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


# [EXPLAIN] `_str_is_int` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _str_is_int(x: str) -> bool:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = _strip_properly_formatted_commas(x)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = float(x)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return abs(x - int(round(x))) <= 1e-7
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] `_str_to_int` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _str_to_int(x: str) -> bool:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x = x.replace(",", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x = float(x)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return int(x)


# [EXPLAIN] `_inject_implicit_mixed_number` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _inject_implicit_mixed_number(step: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    p1 = re.compile("([0-9]) +([0-9])")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return step


# [EXPLAIN] `_strip_properly_formatted_commas` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while True:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_expr = p1.sub("\\1\\3\\4", expr)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if next_expr == expr:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = next_expr
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return next_expr


# [EXPLAIN] `_normalize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _normalize(expr: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Normalize answer expressions."""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if expr is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # Remove enclosing `\text{}`.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if m is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = m.group("text")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("\\%", "%")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("\\$", "$")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("$", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("%", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace(" or ", " , ")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace(" and ", " , ")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("million", "*10^6")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("billion", "*10^9")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("trillion", "*10^12")

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = re.sub("\^ *\\\\circ", "", expr)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = expr[1:-1]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = re.sub(",\\\\! *", "", expr)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _is_float(expr) and _is_int(float(expr)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = str(int(round(float(expr))))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\\" in expr:
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with contextlib.suppress(Exception):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expr = _parse_latex(expr)

    # edge case with mixed numbers and negative signs
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = re.sub("- *", "-", expr)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = _inject_implicit_mixed_number(expr)

    # don't be case sensitive for text answers
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.lower()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _str_is_int(expr):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = str(_str_to_int(expr))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return expr


# [EXPLAIN] `count_unknown_letters_in_expr` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def count_unknown_letters_in_expr(expr: str):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("sqrt", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = expr.replace("frac", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    letters_in_expr = set([x for x in expr if x.isalpha()])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return len(letters_in_expr)


# [EXPLAIN] `should_allow_eval` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if count_unknown_letters_in_expr(expr) > 2:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for bad_string in BAD_SUBSTRINGS:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if bad_string in expr:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return all(re.search(bad_regex, expr) is None for bad_regex in BAD_REGEXES)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@timeout_limit(seconds=10)
# [EXPLAIN] `are_equal_under_sympy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    are_equal = False
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expr = f"({ground_truth_normalized})-({given_normalized})"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if should_allow_eval(expr):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sympy_diff = _sympy_parse(expr)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            simplified = sympy.simplify(sympy_diff)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if simplified == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                are_equal = True
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return are_equal


# [EXPLAIN] `split_tuple` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def split_tuple(expr: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expr = _strip_properly_formatted_commas(expr)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(expr) == 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS and all([ch not in expr[1:-1] for ch in TUPLE_CHARS]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        elems = [expr]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return elems


# [EXPLAIN] `grade_answer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def grade_answer(given_answer: str, ground_truth: str) -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if given_answer is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_normalized = _normalize(ground_truth)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    given_normalized = _normalize(given_answer)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ground_truth_normalized is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ground_truth_normalized == given_normalized:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(given_normalized) == 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_elems = split_tuple(ground_truth_normalized)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    given_elems = split_tuple(given_normalized)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(ground_truth_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]) or len(ground_truth_elems) != len(given_elems):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_correct = False
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                is_correct = ground_truth_elem == given_elem
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                is_correct = False
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # if there's an error, we'll just say it's not correct
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    is_correct = False
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Error: {e} from are_equal_under_sympy, {ground_truth_elem}, {given_elem}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not is_correct:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return is_correct


# [EXPLAIN] `remove_boxed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_boxed(s):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    left = "\\boxed{"
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert s[: len(left)] == left
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert s[-1] == "}"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return s[len(left) : -1]
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None


# [EXPLAIN] `_last_boxed_only_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _last_boxed_only_string(string):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx = string.rfind("\\boxed")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if idx < 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = string.rfind("\\fbox")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if idx < 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    i = idx
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    left_brace_idx = None
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
            if left_brace_idx is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                left_brace_idx = i
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif string[i] == "}":
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

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if left_brace_idx is None or right_brace_idx is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return string[left_brace_idx + 1 : right_brace_idx].strip()


# [EXPLAIN] `match_answer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def match_answer(response):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    is_matched = False
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for ans_marker in ["answer:", "answer is", "answers are"]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ans_idx = response.lower().rfind(ans_marker)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ans_idx != -1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_matched = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = response[ans_idx + len(ans_marker) :].strip()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response.endswith("\n"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                response = response[:-2]

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ans_idx = response.lower().rfind(ans_marker)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ans_idx != -1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_matched = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = response[:ans_idx].strip()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response.endswith("\n"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                response = response[:-2]

    # Find boxed
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ans_boxed = _last_boxed_only_string(response)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ans_boxed:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_matched = True
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response = ans_boxed

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ". " in response:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dot_idx = response.lower().rfind(". ")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dot_idx != -1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = response[:dot_idx].strip()

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ans_idx = response.lower().rfind(ans_marker)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ans_idx != -1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_matched = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = response[ans_idx + len(ans_marker) :].strip()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response.endswith("\n"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                response = response[:-2]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    is_matched = is_matched if any([c.isdigit() for c in response]) else False  # answer must have a digit
    # Grade
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return is_matched, response


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(model_output: str, ground_truth: str) -> bool:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_output = str(model_output)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = str(ground_truth)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    is_matched, extracted_model_output = match_answer(model_output)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_correctness = "Step 2:" in model_output and "\\box" in model_output

    # grade simple algebra questions. if succeeded, return; otherwise, proceed to more complex grading
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if grade_answer(extracted_model_output, ground_truth):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True, True, extracted_model_output

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "\pi" in extracted_model_output or "\pi" in ground_truth:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            equivs = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for pi in [math.pi, 3.14]:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_correct = any(equivs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_correct = math_equal(extracted_model_output, ground_truth, timeout=True)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_correct = False

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return is_correct, format_correctness, extracted_model_output
