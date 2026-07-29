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
from argparse import ArgumentParser
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_bytedance = "Copyright 2024 Bytedance Ltd. and/or its affiliates"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_bytedance_25 = "Copyright 2025 Bytedance Ltd. and/or its affiliates"
# Add custom license headers below
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_prime = "Copyright 2024 PRIME team and/or its affiliates"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_individual = "Copyright 2025 Individual Contributor:"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_sglang = "Copyright 2023-2024 SGLang Team"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_head_modelbest = "Copyright 2025 ModelBest Inc. and/or its affiliates"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
license_headers = [
    license_head_bytedance,
    license_head_bytedance_25,
    license_head_prime,
    license_head_individual,
    license_head_sglang,
    license_head_modelbest,
]


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = ArgumentParser()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--directory", "-d", required=True, type=str)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    directory_in_str = args.directory

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pathlist = Path(directory_in_str).glob("**/*.py")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for path in pathlist:
        # because path is object not string
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path_in_str = str(path.absolute())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(path_in_str)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(path_in_str, encoding="utf-8") as f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_content = f.read()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            has_license = False
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for lh in license_headers:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if lh in file_content:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    has_license = True
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert has_license, f"file {path_in_str} does not contain license"
