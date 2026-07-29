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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
- Preprocess data and split the training set into 75% for training RM and 25% for validting RM.
- All the training data is used to train SFT and RL.
- Both chosen and rejected is used to train SFT
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datasets import load_dataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm.auto import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy, makedirs


# [EXPLAIN] `generate_sft_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_sft_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlh/sft"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = load_dataset("Dahoas/full-hh-rlhf")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = {"prompt": [], "response": []}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in tqdm(dataset["train"]):
        # add chosen
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output["prompt"].append(data["prompt"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output["response"].append(data["chosen"])

        # add rejection
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output["prompt"].append(data["prompt"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output["response"].append(data["rejected"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    df = pd.DataFrame(output)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_dir = os.path.expanduser(local_dir)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(local_dir, exist_ok=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = os.path.join(local_dir, "train.parquet")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    df.to_parquet(path=local_path)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if target_hdfs_path_dir is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        makedirs(hdfs_dir)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        copy(local_path, hdfs_dir)


# [EXPLAIN] `generate_rm_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_rm_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlh/rm"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[:75%]")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[-25%:]")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_dir = os.path.expanduser(local_dir)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(local_dir, exist_ok=True)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dataset, name in zip([train_dataset, test_dataset], ["train", "test"]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = {"prompt": [], "chosen": [], "rejected": []}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for data in tqdm(dataset):
            # add chosen
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output["prompt"].append(data["prompt"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output["chosen"].append(data["chosen"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output["rejected"].append(data["rejected"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        df = pd.DataFrame(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = os.path.join(local_dir, name + ".parquet")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        df.to_parquet(path=local_path)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if target_hdfs_path_dir is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hdfs_dir = target_hdfs_path_dir + "/" + name + ".parquet"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            makedirs(hdfs_dir)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            copy(local_path, hdfs_dir)


# [EXPLAIN] `generate_rl_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_rl_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlhf/rl"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = load_dataset("Dahoas/full-hh-rlhf")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = dataset["train"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source = "Dahoas/full-hh-rlhf"

    # add a row to each data item that represents a unique id
    # [EXPLAIN] `make_map_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def make_map_fn(split):
        # [EXPLAIN] `process_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def process_fn(example, idx):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt = example.pop("prompt")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = example.pop("response")

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": response,  # should not be used
                },
                "extra_info": {"split": split, "index": idx},
            }
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return process_fn

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_dir = os.path.expanduser(local_dir)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = os.path.join(local_dir, "train.parquet")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    train_dataset.to_parquet(local_path)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if target_hdfs_path_dir is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        makedirs(hdfs_dir)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        copy(local_path, hdfs_dir)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--split", type=str, choices=["sft", "rm", "rl"], required=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--local_dir", type=str, default="~/data/full_hh_rlhf")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--hdfs_dir", type=str, required=False, default=None)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if args.split == "sft":
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        generate_sft_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif args.split == "rm":
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        generate_rm_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif args.split == "rl":
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        generate_rl_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError
