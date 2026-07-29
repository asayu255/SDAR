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
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.worker import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, merge_resource_pool


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `TestActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, cuda_visible_devices=None) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(cuda_visible_devices)

    # [EXPLAIN] `get_node_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_node_id(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.get_runtime_context().get_node_id()


# [EXPLAIN] `test` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init()

    # test single-node-no-partition
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("test single-node-no-partition")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([8], use_gpu=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("create actor worker group")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_actor")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("create critic worker group")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    critic_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="hight_level_api_critic")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("create rm worker group")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rm_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_rm")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("create ref worker group")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ref_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_ref")

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]

    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del actor_wg
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del critic_wg
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del rm_wg
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del ref_wg

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    [ray.util.remove_placement_group(pg) for pg in resource_pool.get_placement_groups()]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("wait 5s to remove placemeng_group")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(5)
    # test single-node-multi-partition

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("test single-node-multi-partition")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rm_resource_pool = RayResourcePool([4], use_gpu=True, name_prefix="rm")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ref_resource_pool = RayResourcePool([4], use_gpu=True, name_prefix="ref")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_resource_pool = merge_resource_pool(rm_resource_pool, ref_resource_pool)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rm_resource_pool.world_size == 4
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ref_resource_pool.world_size == 4
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert total_resource_pool.world_size == 8

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_wg = RayWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_actor")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    critic_wg = RayWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_critic")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rm_wg = RayWorkerGroup(rm_resource_pool, class_with_args, name_prefix="high_level_api_rm")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ref_wg = RayWorkerGroup(ref_resource_pool, class_with_args, name_prefix="high_level_api_ref")

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4)]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4, 8)]

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()
