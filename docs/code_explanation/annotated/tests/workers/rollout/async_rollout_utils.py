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
from typing import Any, Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import create_colocated_worker_cls
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.async_server import AsyncLLMServerManager


# [EXPLAIN] `init_async_rollout_manager` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_async_rollout_manager(config: DictConfig, scheduler_kwargs: Dict[str, Any] = None) -> AsyncLLMServerManager:
    # make openai client happy
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["no_proxy"] = ""
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["http_proxy"] = ""
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["https_proxy"] = ""

    # =========================== 1. Create hybrid ActorRollout workers ===========================
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_pool_id = "global_pool"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    resource_pool_manager.create_resource_pool()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_wg = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for resource_pool, class_dict in resource_pool_to_cls.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_wg.update(spawn_wg)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_rollout_wg = all_wg["actor_rollout"]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    actor_rollout_wg.init_model()

    # =========================== 2. Create AsyncLLMServerManager  ===========================
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_rollout_manager = AsyncLLMServerManager(
        config=config.actor_rollout_ref,
        worker_group=actor_rollout_wg,
        scheduler_kwargs=scheduler_kwargs,
    )

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return async_rollout_manager
