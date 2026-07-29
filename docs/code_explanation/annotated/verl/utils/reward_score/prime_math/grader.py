# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2023 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2021 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import contextlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from math import isclose
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Union

# sympy related
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sympy import N, simplify
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sympy.parsing.latex import parse_latex
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sympy.parsing.sympy_parser import parse_expr

# verl related
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import timeout_limit


# [EXPLAIN] `is_digit` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_digit(s):
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "{,}" in str(s):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num = float(str(s).replace("{,}", ""))
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True, num

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num = float(str(s).replace(",", ""))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True, num
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ValueError:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False, None


# [EXPLAIN] `normalize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize(answer, pi) -> str:
    # checking if answer is $<number> and removing $ in that case to compare
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(answer, str) and (bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return answer.replace("\\%", "").replace("%", "")

    # handle base
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer = handle_base(answer)

    # handle pi
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    answer = handle_pi(answer, pi)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return answer


# [EXPLAIN] `handle_base` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def handle_base(x) -> str:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, str) and "_" in x:
        # Due to base
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = x.split("_")[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = float(x)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return int(x)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x


# [EXPLAIN] `handle_pi` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def handle_pi(string, pi):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(string, str) and "\pi" in string:
        # Find the first occurrence of "\pi"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = string.find("\pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while idx != -1:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]

            # Find the next occurrence of "\pi"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx = string.find("\pi", idx + 1)

        # Evaluate the expression using eval() function
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with contextlib.suppress(Exception):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            string = eval(string)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return string


# [EXPLAIN] `math_equal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prediction = normalize(prediction, pi)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reference = normalize(reference, pi)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prediction = prediction[:1000]

    # 0. string comparison
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(prediction, str) and isinstance(reference, str):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if prediction.strip().lower() == reference.strip().lower():
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:  # 1. numerical equal
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prediction = is_digit(prediction)[1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reference = is_digit(reference)[1]
            # number questions
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for item in gt_result:
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isclose(item, prediction, rel_tol=tolerance):
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return True
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not prediction and prediction not in [0, False]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # 2. symbolic equal
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reference = str(reference).strip()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prediction = format_intervals(prediction)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pred_str, ref_str = prediction, reference
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred_str = pred_str.strip("[]()")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_str = ref_str.strip("[]()")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for s in ["{", "}", "(", ")"]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_str = ref_str.replace(s, "")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred_str = pred_str.replace(s, "")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pred_str == ref_str:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if prediction and reference and prediction[0] in "([" and prediction[-1] in ")]" and prediction[0] == reference[0] and prediction[-1] == reference[-1]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred_parts = prediction[1:-1].split(",")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_parts = reference[1:-1].split(",")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(pred_parts) == len(ref_parts) and all([math_equal(pred_pt, ref_pt, include_percentage, tolerance) for pred_pt, ref_pt in zip(pred_parts, ref_parts)]):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "," in prediction and "," in reference:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred_parts = [item.strip() for item in prediction.split(",")]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_parts = [item.strip() for item in reference.split(",")]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(pred_parts) == len(ref_parts):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return bool(all([math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance) for i in range(len(pred_parts))]))

    # if we have point == tuple of values
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_parts = reference[1:-1].split(",")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(pred_parts) == len(ref_parts) and all([math_equal(pred_pt, ref_pt, include_percentage, tolerance) for pred_pt, ref_pt in zip(pred_parts, ref_parts)]):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True

    # if reference is a matrix
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pred_matrix = parse_expr(prediction)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ref_matrix_items = reference.split()[1:-1:2]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(pred_matrix) == len(ref_matrix_items) and all([math_equal(pred, ref, include_percentage, tolerance) for ref, pred in zip(ref_matrix_items, pred_matrix)]):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return True
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(eval(prediction), list):
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pred_matrix = eval(prediction)
                # ref_matrix_items = reference.split()[1:-1:2]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_matrix_items = reference.lstrip("\\begin{pmatrix}").lstrip("\begin{pmatrix}").rstrip("\\end{pmatrix}").rstrip("\end{pmatrix}")  # noqa: B005
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_matrix_items = ref_matrix_items.split("\\")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_matrix_items = [row.split("&") if "&" in row else row for row in ref_matrix_items]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(pred_matrix) == len(ref_matrix_items) and all([math_equal(pred, ref, include_percentage, tolerance) for ref, pred in zip(ref_matrix_items, pred_matrix)]):
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return True
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return symbolic_equal(prediction, reference, tolerance, timeout)


# [EXPLAIN] `symbolic_equal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def symbolic_equal(a, b, tolerance, timeout=10.0):
    # [EXPLAIN] `_parse` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _parse(s):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for f in [parse_expr, parse_latex]:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with timeout_limit(seconds=timeout):
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return f(s)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except TimeoutError:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"Parsing timed out for {s}")
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return s

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = _parse(a)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b = _parse(b)

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with timeout_limit(seconds=timeout):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if simplify(a - b) == 0:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return True
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except TimeoutError:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Simplification timed out for {a} - {b}") 
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with timeout_limit(seconds=timeout):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isclose(N(a), N(b), rel_tol=tolerance):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return True
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except TimeoutError:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Numerical evaluation timed out for {a}, {b}")
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return False

# [EXPLAIN] `format_intervals` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def format_intervals(prediction):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, pattern in patterns.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        match = re.match(pattern, prediction)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if match:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inner_content = match.group(1)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key == "Interval(":  # Intarval(a, b) == [a, b]
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return f"[{inner_content}]"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return f"[{inner_content})"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return f"({inner_content}]"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return f"({inner_content})"

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return prediction
