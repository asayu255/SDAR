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
import argparse

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np


# [EXPLAIN] `extract_reward_from_line` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def extract_reward_from_line(line):
    # TODO: this function needs error handling
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_vals = line.split(" - ")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key_val in key_vals:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            key, val = key_val.split(":")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key == "critic/rewards/mean":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward = float(val)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return reward
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return -np.inf
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return -np.inf


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--output_file", required=True, type=str)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--target", type=float, default=0.2, help="target reward score")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(args.output_file) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = f.read().split("\n")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    best_reward = -np.inf
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for line in output:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if line.startswith("step"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = extract_reward_from_line(line)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if reward > best_reward:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                best_reward = reward

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Best reward is {best_reward}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert best_reward > args.target, f"Best reward must be greater than {args.target}. best_reward: {best_reward}"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Check passes")
