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
from typing import Dict, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.megatron.worker import DistGlobalInfo, DistRankInfo
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# NOTE(sgm): for open-source megatron-core
# [EXPLAIN] `NVMegatronRayWorkerGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Initialize the NVMegatronRayWorkerGroup.

        Args:
            resource_pool (RayResourcePool): The resource pool containing worker resources
            ray_cls_with_init (RayClassWithInitArgs): The Ray class with initialization arguments
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._megatron_global_info: DistGlobalInfo = ray.get(self.execute_rank_zero_async(method_name="get_megatron_global_info"))


# [EXPLAIN] `MegatronRayWorkerGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
        default_megatron_kwargs: Dict = None,
        **kwargs,
    ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            default_megatron_kwargs=default_megatron_kwargs,
            **kwargs,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._megatron_global_info: DistGlobalInfo = ray.get(self.execute_rank_zero_async(method_name="get_megatron_global_info"))

    # [EXPLAIN] `init_megatron` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_megatron(self, default_megatron_kwargs: Optional[Dict] = None):
        # after super, we will call init of each worker
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self._is_init_with_detached_workers:
            # only init_megatron if the WorkerGroup is created from scratch
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.execute_all_sync(method_name="init_megatron", default_megatron_kwargs=default_megatron_kwargs)
