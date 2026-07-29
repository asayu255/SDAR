# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProtoConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

# or set env var VERL_AUTO_PADDING = "1" / "true"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DataProtoConfig.auto_padding = True


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `Actor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Actor(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `add` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def add(self, data: DataProto):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["a"] += self.rank
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data


# [EXPLAIN] `test_auto_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_auto_padding():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(num_cpus=100)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    chunk_size = 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_cls = RayClassWithInitArgs(cls=Actor)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool(process_on_nodes=[chunk_size], use_gpu=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_cls)

    # test locally first
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for test_size in range(4, 20):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_data = DataProto.from_dict({"a": torch.zeros(test_size)}, {"na": np.zeros(test_size, dtype=object)})
        # print(f"before padding, local_data = {local_data}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_size = (chunk_size - (test_size % chunk_size)) if (test_size % chunk_size > 0) else 0
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        local_data.padding(padding_size)
        # print(f"after padding, local_data = {local_data}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(local_data) == len(local_data) + len(local_data) % chunk_size, f"expecting padded length to be {len(local_data) + len(local_data) % chunk_size}, but got {len(local_data)}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunked = local_data.chunk(chunk_size)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(chunked) == chunk_size, f"during test_size = {test_size}, expecting {chunk_size}, got {chunked}"
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for dp in chunked:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(dp) == test_size // chunk_size + bool(test_size % chunk_size), f"test size = {test_size}, expecting dp to be length of {test_size // chunk_size + bool(test_size % chunk_size)}, but got {len(dp)}: {dp} {chunked}"

    # test with RayWorkerGroup method decorated as dispatch_mode=Dispatch.DP_COMPUTE_PROTO
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 10

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict({"a": torch.zeros(1)}, {"na": np.array([str(i) for i in range(1)], dtype=object)})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 1

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict({"a": torch.zeros(8)}, {"na": np.array([str(i) for i in range(8)], dtype=object)})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 8

    # test data proto specific config
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DataProtoConfig.auto_padding = False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)}, auto_padding=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 10

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_single_dict({"a": torch.zeros(1), "na": np.array([str(i) for i in range(1)], dtype=object)}, auto_padding=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 1

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_single_dict({"a": torch.zeros(8), "na": np.array([str(i) for i in range(8)], dtype=object)})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_wg.add(data)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output.batch["a"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == 8

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_auto_padding()
