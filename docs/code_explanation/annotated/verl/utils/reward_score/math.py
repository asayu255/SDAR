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


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(solution_str, ground_truth) -> float:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    retval = 0.0
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string_in_last_boxed = last_boxed_only_string(solution_str)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if string_in_last_boxed is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            answer = remove_boxed(string_in_last_boxed)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_equiv(answer, ground_truth):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                retval = 1.0
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(e)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
# [EXPLAIN] `is_equiv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_equiv(str1, str2, verbose=False):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if str1 is None and str2 is None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("WARNING: Both None")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if str1 is None or str2 is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ss1 = strip_string(str1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ss2 = strip_string(str2)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(ss1, ss2)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ss1 == ss2
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return str1 == str2


# [EXPLAIN] `remove_boxed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_boxed(s):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\\boxed " in s:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        left = "\\boxed "
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert s[: len(left)] == left
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return s[len(left) :]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    left = "\\boxed{"

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert s[: len(left)] == left
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert s[-1] == "}"

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return s[len(left) : -1]


# [EXPLAIN] `last_boxed_only_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def last_boxed_only_string(string):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx = string.rfind("\\boxed")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\\boxed " in string:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
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

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return retval


# [EXPLAIN] `fix_fracs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fix_fracs(string):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    substrs = string.split("\\frac")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_str = substrs[0]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(substrs) > 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        substrs = substrs[1:]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for substr in substrs:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_str += "\\frac"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if substr[0] == "{":
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                new_str += substr
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert len(substr) >= 2
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except:  # noqa: E722
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return string
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                a = substr[0]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                b = substr[1]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if b != "{":
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if len(substr) > 2:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        post_substr = substr[2:]
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        new_str += "{" + a + "}{" + b + "}"
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if len(substr) > 2:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        post_substr = substr[2:]
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        new_str += "{" + a + "}" + b + post_substr
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        new_str += "{" + a + "}" + b
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = new_str
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return string


# [EXPLAIN] `fix_a_slash_b` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fix_a_slash_b(string):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(string.split("/")) != 2:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return string
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = string.split("/")[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b = string.split("/")[1]
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        a = int(a)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        b = int(b)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert string == "{}/{}".format(a, b)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return new_string
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except:  # noqa: E722
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return string


# [EXPLAIN] `remove_right_units` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\\text{ " in string:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splits = string.split("\\text{ ")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(splits) == 2
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return splits[0]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return string


# [EXPLAIN] `fix_sqrt` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fix_sqrt(string):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "\\sqrt" not in string:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return string
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splits = string.split("\\sqrt")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_string = splits[0]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for split in splits[1:]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if split[0] != "{":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            a = split[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_substr = "\\sqrt" + split
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        new_string += new_substr
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return new_string


# [EXPLAIN] `strip_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def strip_string(string):
    # linebreaks
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\n", "")

    # remove inverse spaces
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\!", "")

    # replace \\ with \
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("tfrac", "frac")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\left", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("^{\\circ}", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("^\\circ", "")

    # remove dollar signs
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\$", "")

    # remove units (on the right)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = remove_right_units(string)

    # remove percentage
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\\%", "")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace(" .", " 0.")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(string) == 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return string
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if string[0] == ".":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = fix_sqrt(string)

    # remove spaces
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if string == "0.5":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    string = fix_a_slash_b(string)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return string
