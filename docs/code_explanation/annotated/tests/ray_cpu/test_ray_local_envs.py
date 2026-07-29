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
e2e test verl.single_controller.ray
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.worker import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `TestActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestActor(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

    # [EXPLAIN] `getenv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def getenv(self, key):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val = os.getenv(key, f"{key} not set")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return val


# [EXPLAIN] `test_basics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_basics():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([4], use_gpu=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert output == ["4", "4", "4", "4"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_RANK")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert set(output) == set(["0", "1", "2", "3"])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_basics()
