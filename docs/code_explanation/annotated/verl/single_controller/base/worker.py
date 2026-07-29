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
the class for Worker
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import socket
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .decorator import Dispatch, Execute, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import (
    get_torch_device,
    get_visible_devices_keyword,
    is_npu_available,
)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `DistRankInfo` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DistRankInfo:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    tp_rank: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    dp_rank: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    pp_rank: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    cp_rank: int


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `DistGlobalInfo` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DistGlobalInfo:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    tp_size: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    dp_size: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    pp_size: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    cp_size: int


# [EXPLAIN] `WorkerHelper` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WorkerHelper:
    # [EXPLAIN] `_get_node_ip` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_node_ip(self):
        # [EXPLAIN] `get_node_ip_by_sdk` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def get_node_ip_by_sdk():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if os.getenv("WG_BACKEND", None) == "ray":
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                import ray

                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return ray._private.services.get_node_ip_address()
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError("WG_BACKEND now just support ray mode.")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        host_ipv4 = os.getenv("MY_HOST_IP", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        host_ip_by_env = host_ipv4 or host_ipv6
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        host_ip_by_sdk = get_node_ip_by_sdk()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        host_ip = host_ip_by_env or host_ip_by_sdk
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return host_ip

    # [EXPLAIN] `_get_free_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_free_port(self):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with socket.socket() as sock:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            sock.bind(("", 0))
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return sock.getsockname()[1]

    # [EXPLAIN] `get_availale_master_addr_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_availale_master_addr_port(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_node_ip(), str(self._get_free_port())

    # [EXPLAIN] `_get_pid` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_pid(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return os.getpid()


# we assume that in each WorkerGroup, there is a Master Worker
# [EXPLAIN] `Worker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Worker(WorkerHelper):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A distributed worker that handles initialization and configuration for distributed training.

    This class manages worker initialization, configuration, and provides methods for executing
    distributed operations. It handles communication settings, device configuration, and worker
    metadata management.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fused_worker_attr_name = "fused_worker_dict"

    # [EXPLAIN] `__new__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __new__(cls, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a new Worker instance with proper initialization based on environment settings."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        instance = super().__new__(cls)

        # note that here we use int to distinguish
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        disable_worker_init = int(os.environ.get("DISABLE_WORKER_INIT", 0))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if disable_worker_init:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return instance

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank = os.environ.get("RANK", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if None not in [rank, worker_group_prefix] and "ActorClass(" not in cls.__name__:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return instance

    # [EXPLAIN] `_configure_before_init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _configure_before_init(self, register_center_name: str, rank: int):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Configure worker settings before initialization.

        Args:
            register_center_name (str):
                Name of the register center Ray actor for worker coordination
            rank (int):
                Rank of the worker in the distributed setup
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rank == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            master_addr, master_port = self.get_availale_master_addr_port()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if os.getenv("WG_BACKEND", None) == "ray":
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.register_center = create_worker_group_register_center(name=register_center_name, info=rank_zero_info)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.environ.update(rank_zero_info)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.register_center = ray.get_actor(register_center_name)

        # set worker info for node affinity scheduling
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get(self.register_center.set_worker_info.remote(rank, ray.get_runtime_context().get_node_id()))

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `env_keys` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def env_keys(cls):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """The keys of the environment variables that are used to configure the Worker."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            get_visible_devices_keyword().upper(),
        ]

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, cuda_visible_devices=None) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize the worker with environment settings and device configuration.

        Args:
            cuda_visible_devices (str, optional):
                CUDA visible devices configuration. Defaults to None.
        """
        # construct a meta from environment variable. Note that the import must be inside the class because it is executed remotely
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import os

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._setup_env_cuda_visible_devices()

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from packaging import version
        ###
        # [SUPPORT AMD: torch]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.cuda.is_available() and "AMD" in get_torch_device().get_device_name() and version.parse(ray.__version__) < version.parse("2.45.0"):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("ROCR_VISIBLE_DEVICES")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.environ["LOCAL_RANK"] = os.environ.get("RAY_LOCAL_RANK")
        ###

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = int(os.environ["WORLD_SIZE"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank = int(os.environ["RANK"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._rank = rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._world_size = world_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        master_addr = os.environ["MASTER_ADDR"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        master_port = os.environ["MASTER_PORT"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        ###
        # [SUPPORT AMD: torch]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.cuda.is_available() and "AMD" in get_torch_device().get_device_name() and version.parse(ray.__version__) < version.parse("2.45.0"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.local_rank = int(os.environ["LOCAL_RANK"])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cuda_visible_devices = str(local_rank)
        ###

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        store = {
            "_world_size": world_size,
            "_rank": rank,
            "_local_world_size": local_world_size,
            "_local_rank": local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cuda_visible_devices is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            store["_cuda_visible_devices"] = cuda_visible_devices

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._configure_with_store(store=store)

        ###
        # [SUPPORT AMD: torch]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.cuda.is_available() and "AMD" in get_torch_device().get_device_name() and version.parse(ray.__version__) < version.parse("2.45.0"):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().set_device(int(cuda_visible_devices))
        ###

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.fused_worker_dict = {}

    # [EXPLAIN] `get_fused_worker_by_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_fused_worker_by_name(self, worker_name: str):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get a fused worker by its name.

        Args:
            worker_name (str):
                Name of the worker to retrieve
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.fused_worker_dict.get(worker_name, None)

    # [EXPLAIN] `_setup_env_cuda_visible_devices` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _setup_env_cuda_visible_devices(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.ray_utils import ray_noset_visible_devices

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_ray_noset_visible_devices = ray_noset_visible_devices()

        # Prevent use of clashing `{CUDA/HIP/ROCR}_VISIBLE_DEVICES``
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hip_val:
            # Switch the use of HIP_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES for consistency.
            # Make sure that the HIP_VISIBLE_DEVICES is set to the same value as CUDA_VISIBLE_DEVICES
            # at this point.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hip_val = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cuda_val:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert val == cuda_val, (
                    f"Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values "
                    f"found: {val} and {cuda_val}."
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cuda_val = val
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                # os.environ["HIP_VISIBLE_DEVICES"] = val

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rocr_val:
            # You must take care if both HIP/CUDA and ROCR env vars are set as they have
            # different meanings. Both env vars accept either a list of ints or a
            # list of UUIDs. The ROCR env var is processed first which then reduces
            # the number of GPUs that HIP can select from.
            # https://github.com/pytorch/pytorch/pull/144026
            # To avoid the complexity of this, we simply gives out error if both are set
            # (Also to keep consistency with ray's practice with 2.45.0).
            # Otherwise, we will set ROCR_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES
            # and remove ROCR_VISIBLE_DEVICES.
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cuda_val:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rocr_val = None

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_ray_noset_visible_devices:
            # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
            # environment variable for each actor, unless
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set,
            # so we need to set local rank when the flag is set.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            device_name = "NPU" if is_npu_available else "GPU"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            os.environ["LOCAL_RANK"] = local_rank
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().set_device(int(local_rank))

    # [EXPLAIN] `_configure_with_store` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _configure_with_store(self, store: Dict):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        This function should only be called inside by WorkerGroup
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.__dict__.update(store_env_dict)  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in type(self).env_keys():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val = self.__dict__.get(f"_{key.lower()}", None)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if val is not None:
                # print(f"set {key} to {val}")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                os.environ[key] = str(val)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""

    # [EXPLAIN] `get_master_addr_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_master_addr_port(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the master address and port for distributed communication."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._master_addr, self._master_port

    # [EXPLAIN] `get_cuda_visible_devices` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_cuda_visible_devices(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the CUDA visible devices configuration."""
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import os

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cuda_visible_devices

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the total number of workers in the distributed setup."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._world_size

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def rank(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the rank of this worker in the distributed setup."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._rank

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    # [EXPLAIN] `execute_with_func_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_with_func_generator(self, func, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ret_proto = func(self, *args, **kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ret_proto

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    # [EXPLAIN] `execute_func_rank_zero` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_func_rank_zero(self, func, *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a function in rank zero execution mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = func(*args, **kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result
