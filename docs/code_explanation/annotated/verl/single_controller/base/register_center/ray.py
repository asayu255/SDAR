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
from typing import Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `WorkerGroupRegisterCenter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WorkerGroupRegisterCenter:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, rank_zero_info):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank_zero_info = rank_zero_info
        # rank -> node_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers_info: Dict[int, str] = {}

    # [EXPLAIN] `get_rank_zero_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rank_zero_info(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.rank_zero_info

    # [EXPLAIN] `set_worker_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_worker_info(self, rank, node_id) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers_info[rank] = node_id

    # [EXPLAIN] `get_worker_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_worker_info(self) -> Dict[int, str]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.workers_info


# [EXPLAIN] `create_worker_group_register_center` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_worker_group_register_center(name, info):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return WorkerGroupRegisterCenter.options(name=name).remote(info)
