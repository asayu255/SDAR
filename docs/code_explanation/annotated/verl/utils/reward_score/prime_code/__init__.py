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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import traceback

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .utils import check_correctness as apps_check_correctness


# [EXPLAIN] `compute_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    solution = completion.split("```python")[-1].split("```")[0]
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not isinstance(test_cases, dict):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                test_cases = json.loads(test_cases)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metadata = dict(enumerate(metadata))[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            success = all(map(lambda x: x is True, res))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if success:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return success, metadata
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        test_cases_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs = test_cases["inputs"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = test_cases["outputs"]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(inputs)):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metadata_list = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res_list = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for test_case_id, test_case in enumerate(test_cases_list):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    metadata = {}
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metadata["test_case"] = {}
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metadata["test_case"]["res"] = str(res)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metadata_list.append(metadata)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                res_list.extend(res)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if test_case_id >= 9:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res_count = len(res_list) if len(res_list) > 0 else 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            success = sum(map(lambda x: x is True, res_list)) / res_count
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        traceback.print_exc(10)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        success = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metadata_list = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return success, metadata_list
