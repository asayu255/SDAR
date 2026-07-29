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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hydra
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.reward import get_custom_reward_fn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `process_item` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_item(reward_fn, data_source, response_lst, reward_data):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth = reward_data["ground_truth"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data_source, np.mean(score_lst)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@hydra.main(config_path="config", config_name="evaluation", version_base=None)
# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main(config):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = pd.read_parquet(local_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    responses = dataset[config.data.response_key]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_sources = dataset[config.data.data_source_key]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_model_data = dataset[config.data.reward_model_key]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total = len(dataset)

    # Initialize Ray
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not ray.is_initialized():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_source_reward = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tqdm(total=total) as pbar:
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            done_ids, remote_tasks = ray.wait(remote_tasks)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for result_id in done_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_source, score = ray.get(result_id)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                data_source_reward[data_source].append(score)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                pbar.update(1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metric_dict = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data_source, rewards in data_source_reward.items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(metric_dict)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
