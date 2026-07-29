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
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.worker import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["RAY_DEDUP_LOGS"] = "0"
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["NCCL_DEBUG"] = "WARN"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `ModelActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ModelActor(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass


# [EXPLAIN] `HackSelf` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class HackSelf:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass


# [EXPLAIN] `get_aux_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_aux_metrics(self, test_proto):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_ids = test_proto.batch["sequence_ids"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    decode_count = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(sequence_ids.size(0)):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        decode_count.append(len(sequence_ids[i].tolist()))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret_proto = DataProto(batch=TensorDict({"sequence_ids": sequence_ids, "decode_count": torch.tensor(decode_count)}, batch_size=sequence_ids.size(0)))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ret_proto


# [EXPLAIN] `test` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test():
    # construct model
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init()

    # create 2 workers, each hold a GPU
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([2], use_gpu=True, name_prefix="a")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_with_args = RayClassWithInitArgs(cls=ModelActor)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shard_wg = RayWorkerGroup(resource_pool, class_with_args)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_bs = 8
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_proto = DataProto(
        TensorDict(
            {
                "sequence_ids": torch.ones([test_bs, 2048], dtype=torch.int64),
            },
            batch_size=test_bs,
        ),
        meta_info={"query_length": 1536},
    )

    # Sharding among different ranks
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret_proto1 = shard_wg.execute_with_func_generator(get_aux_metrics, test_proto)

    # compare execute on driver
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hs = HackSelf()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret_proto2 = get_aux_metrics(hs, test_proto)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(ret_proto1.batch["decode_count"], ret_proto2.batch["decode_count"])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()
