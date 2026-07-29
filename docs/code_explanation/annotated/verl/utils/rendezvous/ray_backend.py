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
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from cupy.cuda.nccl import NcclCommunicator, get_unique_id
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray.util import list_named_actors


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `NCCLIDStore` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class NCCLIDStore:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, nccl_id):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._nccl_id = nccl_id

    # [EXPLAIN] `get` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._nccl_id


# [EXPLAIN] `get_nccl_id_store_by_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_nccl_id_store_by_name(name):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_actors = list_named_actors(all_namespaces=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(matched_actors) == 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor = matched_actors[0]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.get_actor(**actor)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif len(matched_actors) > 1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logging.warning("multiple actors with same name found: %s", matched_actors)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif len(matched_actors) == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logging.info("failed to get any actor named %s", name)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return None


# [EXPLAIN] `create_nccl_communicator_in_ray` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_nccl_communicator_in_ray(rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rank == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nccl_id = get_unique_id()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        communicator = NcclCommunicator(
            ndev=world_size,
            commId=nccl_id,
            rank=0,
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return communicator
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(max_retries):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if nccl_id_store is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logging.info("nccl_id_store %s got", group_name)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                nccl_id = ray.get(nccl_id_store.get.remote())
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                communicator = NcclCommunicator(
                    ndev=world_size,
                    commId=nccl_id,
                    rank=rank,
                )
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return communicator
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(interval_s)
