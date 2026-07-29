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
import ast
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import faulthandler
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import platform

# to run the solution files we're using a timing based approach
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import signal
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import traceback

# used for debugging to time steps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datetime import datetime
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum

# for capturing the stdout
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from io import StringIO

# used for testing the code that reads from input
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import mock_open, patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pyext import RuntimeModule


# [EXPLAIN] `truncatefn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def truncatefn(s, length=300):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(s, str)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(s) <= length:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return s

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


# [EXPLAIN] `CODE_TYPE` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CODE_TYPE(Enum):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    call_based = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    standard_input = 1

# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
# [EXPLAIN] `Capturing` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Capturing(list):
    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._stdout = sys.stdout
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._stringio.close = lambda x: 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self

    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, *args):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.append(self._stringio.getvalue())
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del self._stringio  # free up some memory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sys.stdout = self._stdout


# [EXPLAIN] `only_int_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def only_int_check(val):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return isinstance(val, int)


# [EXPLAIN] `string_int_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def string_int_check(val):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return isinstance(val, str) and val.isdigit()


# [EXPLAIN] `combined_int_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def combined_int_check(val):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return only_int_check(val) or string_int_check(val)


# [EXPLAIN] `clean_traceback` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def clean_traceback(error_traceback):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    file_start = error_traceback.find('File "<string>"')
    # print(file_start)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    error_traceback = "Traceback (most recent call last):\n  " + error_traceback[file_start:]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return error_traceback


# [EXPLAIN] `run_test` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_test(in_outs, test=None, debug=False, timeout=15):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    reliability_guard()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if debug:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"start = {datetime.now().time()}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if in_outs:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if in_outs.get("fn_name") is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            which_type = CODE_TYPE.standard_input  # Standard input
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method_name = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            which_type = CODE_TYPE.call_based  # Call-based
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method_name = in_outs["fn_name"]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if debug:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"loaded input_output = {datetime.now().time()}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if test is None:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AssertionError("should not happen: test code is none")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif test is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"  # noqa: E501
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if debug:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"loading test code = {datetime.now().time()}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if which_type == CODE_TYPE.call_based:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            sol += test
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if debug:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"sol = {sol}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            signal.alarm(timeout)
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(0)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                error_traceback = traceback.format_exc()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"type 0 compilation error = {e}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                results.append(-2)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return results, {
                    "error": repr(e),
                    # "error_code": -1,
                    # "error_message": "Compilation Error",
                    "traceback": clean_traceback(error_traceback),
                }
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            signal.alarm(0)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                astree = ast.parse(test)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                last_block = astree.body[-1]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(last_block, ast.If):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    condition = last_block.test
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if ast.unparse(condition).strip() == "__name__ == '__main__'":
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_test = test.split("\n")

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_test = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for x in tmp_test:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    new_test.append("\t" + x + "\n")
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    new_test.append(x + "\n")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_test = new_test

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_test = ""
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            started = False
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in tmp_test:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if i.startswith("\t") and not started:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    new_test += "def code():\n"
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_test += i
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    started = True
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_test += "\t" + i
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_test += i
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_test = new_test

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            sol += tmp_test
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if debug:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"sol = {sol}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method_name = "code"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            signal.alarm(timeout)
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tmp = tmp_sol
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(0)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                error_traceback = traceback.format_exc()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"type 1 compilation error = {e}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                results.append(-2)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return results, {
                    "error": repr(e),
                    # "error_code": -1,
                    # "error_message": "Compilation Error",
                    "traceback": clean_traceback(error_traceback),
                }
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            signal.alarm(0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if debug:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"get method = {datetime.now().time()}")

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            signal.alarm(0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            error_traceback = traceback.format_exc()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            error_info = sys.exc_info()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"unable to get function error = {error_info}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            results.append(-2)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return results, {
                "error": repr(error_info),
                # "error_code": -1,
                # "error_message": "Unable to extract code",
                "traceback": clean_traceback(error_traceback),
            }

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for index, inputs in enumerate(in_outs["inputs"]):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_inputs = inputs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_outputs = in_outs["outputs"][index]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if which_type == CODE_TYPE.call_based:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                inputs = [json.loads(line) for line in inputs.split("\n")]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                in_outs["outputs"][index] = json.loads(in_outs["outputs"][index])

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                truncate_line_size = 300 // (raw_inputs.count("\n") + 1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_inputs = "\n".join([truncatefn(line, truncate_line_size) for line in raw_inputs.strip().split("\n")])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_outputs = truncatefn(raw_outputs, 200)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_inputs = truncatefn(raw_inputs)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_outputs = truncatefn(raw_outputs, 200)
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(inputs[0], dict):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(in_outs["outputs"][index], dict):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(in_outs["outputs"][index][0], dict):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if debug:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if which_type == CODE_TYPE.call_based:  # Call-based
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(timeout)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                faulthandler.enable()
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = method(*inputs)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    raw_true_output = output

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    raw_true_output_copy = json.dumps(output)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    raw_true_output_copy = truncatefn(raw_true_output_copy, 200)

                    # ground truth sequences are not tuples
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(output, tuple):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        output = list(output)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = output == in_outs["outputs"][index]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                    try:
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if isinstance(output[0], tuple):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                    except Exception:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        pass
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if tmp_result is not True:
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return results, {
                            "output": raw_true_output_copy,
                            "expected": raw_outputs,
                            "inputs": raw_inputs,
                            # "error_code": -2,
                            "error_message": "Wrong Answer",
                        }
                    # reset the alarm
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    signal.alarm(0)
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    signal.alarm(0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    error_traceback = traceback.format_exc()
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    faulthandler.disable()
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Standard input runtime error or time limit exceeded error = {e}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(-1)
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return results, {
                        "error": repr(e),
                        "traceback": clean_traceback(error_traceback),
                    }
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                faulthandler.disable()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(0)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                faulthandler.enable()
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                passed = False

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(inputs, list):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    inputs = "\n".join(inputs)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(in_outs["outputs"][index], list):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.alarm(timeout)
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with Capturing() as output:
                    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                    try:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        call_method(method, inputs)
                        # reset the alarm
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        signal.alarm(0)
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        passed = True
                    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                    except Exception as e:
                        # runtime error or took too long
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        signal.alarm(0)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        error_traceback = traceback.format_exc()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        results.append(-1)
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return results, {
                            "error": repr(e),
                            "traceback": clean_traceback(error_traceback),
                        }
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    signal.alarm(0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_true_output = output[0]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_true_output_copy = truncatefn(raw_true_output, 200)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = raw_true_output.splitlines()
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not passed:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        nl = "\n"
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if not isinstance(inputs, list):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if passed and debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if custom_compare_(output, in_outs["outputs"][index]):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = True
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # ground truth sequences are expressed as lists not tuples
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(output, tuple):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = list(output)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tmp_result = False
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = output == [in_outs["outputs"][index]]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(in_outs["outputs"][index], list):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if isinstance(output[0], str):
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Failed check1 exception = {e}")
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is True:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # try one more time without \n
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(in_outs["outputs"][index], list):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = output == [in_outs["outputs"][index]]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(in_outs["outputs"][index], list):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Failed check2 exception = {e}")
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is True:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # try by converting the output into a split up list too
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(output, list):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = list(filter(len, output))

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    nl = "\n"
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not isinstance(inputs, list):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}")
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}")

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @a")

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = output == [in_outs["outputs"][index]]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(in_outs["outputs"][index], list):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Failed check3 exception = {e}")
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @b")

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    all_ints = all(combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output, in_outs["outputs"][index]))
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not all_ints:
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if debug:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print([combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output, in_outs["outputs"][index])])
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        output_float = [float(e) for e in output]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        gt_float = [float(e) for e in in_outs["outputs"][index]]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @c")

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(output[0], list):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        all_ints = all(combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output[0], in_outs["outputs"][index]))
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if not all_ints:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            output_float = [float(e) for e in output[0]]
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is True:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @d")
                # try by converting the stuff into split up list
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(in_outs["outputs"][index], list):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @e")

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tmp_result = output == in_outs["outputs"][index]
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if debug:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Failed check4 exception = {e}")
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is True:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    results.append(tmp_result)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @f")

                # try by converting the output into a split up list too
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(output, list):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for tmp_index, i in enumerate(output):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        output[tmp_index] = i.split()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = list(filter(len, output))
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for tmp_index, i in enumerate(output):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        output[tmp_index] = set(i)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = output.split()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = list(filter(len, output))
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = set(output)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"{tmp_result=} @g")

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is True and debug:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("PASSED")

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                results.append(tmp_result)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tmp_result is not True:
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return results, {
                        "output": raw_true_output_copy,
                        "expected": raw_outputs,
                        "inputs": raw_inputs,
                        # "error_code": -2,
                        "error_message": "Wrong Answer",
                    }

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if debug:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    nl = "\n"
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not isinstance(inputs, list):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"results = {results}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return results, {}


# [EXPLAIN] `custom_compare_` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def custom_compare_(output, ground_truth):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(output, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_1 = "\n".join(output)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if stripped_string_compare(output_1, ground_truth):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(output, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_2 = [o.lstrip().rstrip() for o in output]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_2 = "\n".join(output_2)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if stripped_string_compare(output_2, ground_truth):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return True

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return False


# [EXPLAIN] `stripped_string_compare` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def stripped_string_compare(s1, s2):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s1 = s1.lstrip().rstrip()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s2 = s2.lstrip().rstrip()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return s1 == s2


# [EXPLAIN] `call_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def call_method(method, inputs):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(inputs, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs = "\n".join(inputs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("builtins.open", mock_open(read_data=inputs))
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("sys.stdin", StringIO(inputs))
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    # [EXPLAIN] `_inner_call_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _inner_call_method(_method):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return _method()
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except SystemExit:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _inner_call_method(method)


# [EXPLAIN] `reliability_guard` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reliability_guard(maximum_memory_bytes=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if maximum_memory_bytes is not None:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import resource

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if platform.uname().system != "Darwin":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    faulthandler.disable()

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import builtins

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    builtins.exit = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    builtins.quit = None

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import os

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["OMP_NUM_THREADS"] = "1"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.kill = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.system = None  # 防止干扰repl评测
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.putenv = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.remove = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.removedirs = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.rmdir = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.fchdir = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.setuid = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.fork = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.forkpty = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.killpg = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.rename = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.renames = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.truncate = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.replace = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.unlink = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.fchmod = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.fchown = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.chmod = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.chown = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.chroot = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.lchflags = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.lchmod = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.lchown = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.getcwd = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    os.chdir = None

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import shutil

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shutil.rmtree = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shutil.move = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shutil.chown = None

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import subprocess

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    subprocess.Popen = None  # type: ignore

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    __builtins__["help"] = None

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import sys

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sys.modules["ipdb"] = None
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sys.modules["joblib"] = None
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sys.modules["resource"] = None
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sys.modules["psutil"] = None
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sys.modules["tkinter"] = None
