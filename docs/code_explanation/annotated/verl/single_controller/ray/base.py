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
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List, Optional, Tuple
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray.experimental.state.api import get_actor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray.util import list_named_actors
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray.util.placement_group import PlacementGroup, placement_group
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto, _padding_size_key
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["Worker"]


# [EXPLAIN] `get_random_string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_random_string(length: int) -> str:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import random
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import string

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    letters_digits = string.ascii_letters + string.digits
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "".join(random.choice(letters_digits) for _ in range(length))


# [EXPLAIN] `func_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    # [EXPLAIN] `Functor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class Functor:
        # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __call__(this, *args, **kwargs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padding_count = kwargs.pop(_padding_size_key, 0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = execute_fn(method_name, *args, **kwargs)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if blocking:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = ray.get(output)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = collect_fn(self, output)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if padding_count > 0:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(output, DataProto):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    indices = [i for i in range(len(output))][:-padding_count]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = output.select_idxs(indices)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif isinstance(output, list):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = output[:-padding_count]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return output

    # use class type to pass the method_name to get a better observability
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return type(method_name, (Functor,), {})()


# [EXPLAIN] `sort_placement_group_by_node_ip` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def sort_placement_group_by_node_ip(pgs: List[PlacementGroup]) -> List[PlacementGroup]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_ip = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for pg in pgs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        node_id = specs["bundles_to_node_id"][0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg_ip[pg.id] = node_ip[node_id]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


# [EXPLAIN] `RayResourcePool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RayResourcePool(ResourcePool):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        process_on_nodes: Optional[List[int]] = None,
        use_gpu: bool = True,
        name_prefix: str = "",
        max_colocate_count: int = 10,
        detached=False,
        accelerator_type: Optional[str] = None,
    ) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(process_on_nodes, max_colocate_count)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name_prefix = name_prefix
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pgs = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.detached = detached
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.accelerator_type = accelerator_type

    # [EXPLAIN] `get_placement_groups` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.pgs is not None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.pgs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg_name_prefix = name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # print(f"pg_name_prefix = {pg_name_prefix}")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if device_name == "npu":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            device_name = "NPU"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif device_name == "cuda":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            device_name = "GPU"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bundle = {"CPU": self.max_colocate_count}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_gpu:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            bundle[device_name] = 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.accelerator_type is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                bundle[self.accelerator_type] = 1e-4
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lifetime = "detached" if self.detached else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pgs = [placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime) for idx, bundles in enumerate(pg_scheme)]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get([pg.ready() for pg in pgs])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pgs = pgs
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return pgs


# [EXPLAIN] `extract_pg_from_exist` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def extract_pg_from_exist(resource_pools: Dict[str, RayResourcePool], src_role_names: List[str], resource_pool: RayResourcePool) -> List:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_pgs = [pg for role_name, resource_pool in resource_pools.items() for pg in resource_pool.get_placement_groups() if role_name in src_role_names]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unsorted_pgs: List[Tuple[int, PlacementGroup]] = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    searching_idx = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for request_process, original_idx in sorted_process_on_nodes:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, f"requesting {request_process} processes, bundle count cannot satisfy"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        searching_idx += 1

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [pg for _, pg in sorted(unsorted_pgs)]


# [EXPLAIN] `merge_resource_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rp1.use_gpu == rp2.use_gpu, "Both RayResourcePool must either use_gpu or not"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rp1.max_colocate_count == rp2.max_colocate_count, "Both RayResourcePool must has the same max_colocate_count"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayResourcePool must has the same n_gpus_per_node"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_store = rp1.store + rp2.store

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merged = type(rp1)(new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return merged


# [EXPLAIN] `RayClassWithInitArgs` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RayClassWithInitArgs(ClassWithInitArgs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(cls, *args, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._options = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._additional_resource = {}

    # [EXPLAIN] `set_additional_resource` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_additional_resource(self, additional_resource):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._additional_resource = additional_resource

    # [EXPLAIN] `update_options` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_options(self, options: Dict):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._options.update(options)

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, placement_group, placement_group_bundle_idx, use_gpu: bool = True, num_gpus=1, sharing_with=None, device_name="cuda") -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with
            device_name: Device for training

        Returns:
            A Ray actor handle with the configured options
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sharing_with is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=cuda_visible_devices, **self.kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        options = {"scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx)}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        options.update(self._options)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_gpu and device_name == "cuda":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            options["num_gpus"] = num_gpus
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_gpu and device_name == "npu":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            options["resources"] = {"NPU": num_gpus}

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(self._additional_resource) > 1:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k, v in self._additional_resource.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


# [EXPLAIN] `RayWorkerGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RayWorkerGroup(WorkerGroup):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        worker_handles: List[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        device_name="cuda",
        **kwargs,
    ) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize a RayWorkerGroup.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            name_prefix: Prefix for worker names
            detached: Whether workers should be detached
            worker_names: Names of existing workers to attach to
            ray_wait_register_center_timeout: Timeout for waiting on register center
            **kwargs: Additional keyword arguments
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(resource_pool=resource_pool, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ray_cls_with_init = ray_cls_with_init
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout
        # Whether the WorkerGroup is a Colocate WorkerGroup created by FusedWorker.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.fused_worker_used = ray_cls_with_init.fused_worker_used
        # if a WorkerGroup is spawned from Colocate WorkerGroup, this indicates which sub-class is binded to this WorkerGroup.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sub_cls_name = ""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_name = device_name

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if worker_names is not None and (not self.fused_worker_used):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self._is_init_with_detached_workers
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._worker_names = worker_names

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_init_with_detached_workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._init_with_detached_workers(worker_names=worker_names, worker_handles=worker_handles)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._init_with_resource_pool(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, bin_pack=bin_pack, detached=detached)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ray_cls_with_init is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.wg_dict = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.method_names = []

    # [EXPLAIN] `_is_worker_alive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        worker_state_dict = get_actor(worker._actor_id.hex())
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    # [EXPLAIN] `_init_with_detached_workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_with_detached_workers(self, worker_names, worker_handles):
        # ray.get_actor holds a weak reference to the actor, which causes actors garbage collected unexpectedly
        # if we only hold spawn RayWorkerGroup. By passing actor handle explicitly, spawn RayWorkerGroup have
        # strong reference to these actors.
        # https://github.com/ray-project/ray/pull/45699
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._workers = workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._world_size = len(worker_names)

    # [EXPLAIN] `_init_with_resource_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_gpu = resource_pool.use_gpu

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        strategy = "PACK"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if bin_pack:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            strategy = "STRICT_PACK"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = resource_pool.world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_gpus = 1 / resource_pool.max_colocate_count

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank = -1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_world_size = resource_pool.store[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for local_rank in range(local_world_size):
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                rank += 1

                # we pass in environment variable at option so that Worker can use environment variable to set
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "WG_PREFIX": self.name_prefix,
                    "WG_BACKEND": "ray",
                    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                    "RAY_LOCAL_RANK": str(local_rank),
                }
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if rank != 0:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    env_vars["MASTER_ADDR"] = self._master_addr
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    env_vars["MASTER_PORT"] = self._master_port

                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                import re

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cia_name = type(ray_cls_with_init.cls).__name__
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if detached:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    ray_cls_with_init.update_options({"lifetime": "detached"})

                # create a worker
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                worker = ray_cls_with_init(placement_group=pg, placement_group_bundle_idx=local_rank, use_gpu=use_gpu, num_gpus=num_gpus, device_name=self.device_name)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._workers.append(worker)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._worker_names.append(name)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if rank == 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    register_center_actor = None
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    actor_name = f"{self.name_prefix}_register_center"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    start_time = time.time()

                    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
                    while time.time() - start_time < self._ray_wait_register_center_timeout:
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if actor_name in list_named_actors():
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            register_center_actor = ray.get_actor(actor_name)
                            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                            break

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        elapsed = int(time.time() - start_time)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if elapsed % 30 == 0:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            logging.warning(
                                "Waiting for register center actor %s to be ready. Elapsed time: %s seconds out of %s seconds.",
                                actor_name,
                                elapsed,
                                self._ray_wait_register_center_timeout,
                            )
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        time.sleep(1)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if register_center_actor is None:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise TimeoutError(
                            f"Failed to get register_center_actor {actor_name} "
                            f"in {list_named_actors(all_namespaces=True)} "
                            f"for {self._ray_wait_register_center_timeout} seconds. "
                            "Ensure that any lingering Ray resources from previous "
                            "runs are cleaned up (e.g., by restarting the Ray cluster), "
                            "or adjust the waiting time by modifying the config "
                            "`trainer.ray_wait_register_center_timeout`."
                        )

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self._master_addr, self._master_port = rank_zero_info["MASTER_ADDR"], rank_zero_info["MASTER_PORT"]
                    # print(f"rank_zero_info: {rank_zero_info}")
                    # print(f"master_addr: {self._master_addr}, master_port: {self._master_port}")

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `worker_names` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def worker_names(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._worker_names

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_detached` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_detached(
        cls,
        name_prefix,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            ray_cls_with_init: Class with initialization arguments for workers

        Returns:
            A new RayWorkerGroup instance
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        worker_group = cls(resource_pool=None, ray_cls_with_init=ray_cls_with_init, name_prefix=name_prefix, worker_names=worker_names, worker_handles=worker_handles)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return worker_group

    # [EXPLAIN] `spawn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def spawn(self, prefix_set):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.fused_worker_used:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.spawn_fused(prefix_set)

        # [EXPLAIN] `_rebind_actor_methods` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def _rebind_actor_methods(worker_group, actor_name):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix: str = actor_name + "_"
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for method_name in dir(worker_group):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if method_name.startswith(prefix):
                    # only valid when Python >= 3.9
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    original_method_name = method_name.removeprefix(prefix)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    method = getattr(worker_group, method_name)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(worker_group, original_method_name, method)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_worker_group_dict = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for prefix in prefix_set:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_worker_group = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._workers,
                ray_cls_with_init=self.ray_cls_with_init,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _rebind_actor_methods(new_worker_group, prefix)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_worker_group_dict[prefix] = new_worker_group
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return new_worker_group_dict

    # [EXPLAIN] `spawn_fused` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def spawn_fused(self, prefix_set):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wg_dict = dict()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in prefix_set:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_wg = deepcopy(self)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_wg.sub_cls_name = key
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            wg_dict[key] = new_wg
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return wg_dict

    # [EXPLAIN] `fuse` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fuse(self, prefix_set):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.wg_dict is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.wg_dict = self.spawn(prefix_set)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for role_name, role_wg in self.wg_dict.items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(self, role_name, role_wg)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    # [EXPLAIN] `_execute_remote_single_worker` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.fused_worker_used and method_name not in self.method_names:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            remote_call = getattr(worker, self.fused_worker_execute_fn_name)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
        # fused worker not used
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remote_call = getattr(worker, method_name)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return remote_call.remote(*args, **kwargs)

    # [EXPLAIN] `execute_rank_zero_sync` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    # [EXPLAIN] `execute_rank_zero_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    # [EXPLAIN] `execute_rank_zero` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    # [EXPLAIN] `execute_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_all(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.execute_all_async(method_name, *args, **kwargs)

    # [EXPLAIN] `execute_all_sync` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_all_sync(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    # [EXPLAIN] `execute_all_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_all_async(self, method_name: str, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        length = len(self._workers)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i in range(length):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    sliced_args = tuple(arg[i] for arg in args)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    result.append(self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return result

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `master_address` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def master_address(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._master_addr

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `master_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def master_port(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._master_port

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def workers(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._workers

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._world_size


# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Utilities that enables creating workers inside the same ray.Actor, 
with code written in separate ray.Actors.
"""


# deprecated, switching to FusedWorker
# [EXPLAIN] `_bind_workers_method_to_parent` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for method_name in dir(user_defined_cls):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method = getattr(user_defined_cls, method_name)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(method, MAGIC_ATTR):

            # [EXPLAIN] `generate_function` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def generate_function(name, key=key):
                # [EXPLAIN] `func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                # [EXPLAIN] `async_func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
                async def async_func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return wrapper

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attrs = getattr(method, MAGIC_ATTR)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(func, MAGIC_ATTR, attrs)
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # bind direct rollout method to class without prefix
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert not hasattr(cls, method_name), f"conflict direct rollout method {method_name} with role {key}"
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(cls, method_name, func)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"bind role {key} method {method_name} to class {cls}")
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    method_name_with_prefix = key + "_" + method_name
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(cls, method_name_with_prefix, func)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Fail to set method_name {method_name}") from e


# [EXPLAIN] `_unwrap_ray_remote` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _unwrap_ray_remote(cls):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if hasattr(cls, "__ray_actor_class__"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls = cls.__ray_actor_class__
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return cls


# [EXPLAIN] `_determine_fsdp_megatron_base_class` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _determine_fsdp_megatron_base_class(mros: List):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for cls in mros[0]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cls.__name__ == "MegatronWorker":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cls.__name__ == "Worker":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
# [EXPLAIN] `create_colocated_worker_cls` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_dict = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_args_dict = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_cls = _determine_fsdp_megatron_base_class([cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"colocated worker base class {worker_cls}")

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, cls in class_dict.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls_dict[key] = cls.cls
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    # [EXPLAIN] `WorkerDict` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class WorkerDict(worker_cls):
        # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __init__(self):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            super().__init__()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.worker_dict = {}
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, user_defined_cls in cls_dict.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with patch.dict(os.environ, {"DISABLE_WORKER_INIT": "1"}):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {}))

    # now monkey-patch the methods from inner class to WorkerDict
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, user_defined_cls in cls_dict.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remote_cls = ray.remote(WorkerDict)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return remote_cls


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
FusedWorkerCLSName = "FusedWorker"


# [EXPLAIN] `create_colocated_worker_raw_cls` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    # [EXPLAIN] `FusedWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class FusedWorker(Worker):
        # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __init__(self, *args, **kwargs):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            super().__init__(*args, **kwargs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.cls_names = cls_names
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.raw_cls_dict = raw_cls_dict
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.init_args_dict = init_args_dict
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.init_kwargs_dict = init_kwargs_dict

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for cls_name, udc, ud_args, ud_kwargs in zip(self.cls_names, self.raw_cls_dict.values(), self.init_args_dict.values(), self.init_kwargs_dict.values()):
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with patch.dict(os.environ, {"DISABLE_WORKER_INIT": "1"}):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _, worker in self.fused_worker_dict.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        # [EXPLAIN] `_fuw_execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            names = method_name.split("_fwmn_")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cls_name = names[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            method_name = names[1]

            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert cls_name in self.fused_worker_dict, f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return udc_method(*args, **kwargs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    renamed_fused_worker_cls.is_fused_worker = True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return renamed_fused_worker_cls


# [EXPLAIN] `create_colocated_worker_cls_fused` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remote_cls = ray.remote(raw_colocated_worker_cls)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cia = RayClassWithInitArgs(cls=remote_cls)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cia.fused_worker_used = True

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return cia
