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
The main entry point to run the PPO algorithm
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import psutil
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from codetiming import Timer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig, open_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import init_device_mesh

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import convert_to_regular_types
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.transformers.monkey_patch import apply_monkey_patch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_processor, hf_tokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.activation_offload import enable_activation_offloading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import log_gpu_memory_usage
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.flops_counter import FlopsCounter
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    layered_summon_lora_params,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.import_utils import import_external_libs
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from peft import LoraConfig, TaskType, get_peft_model
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from codetiming import Timer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from peft import PeftModel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from safetensors.torch import save_file
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import asdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
device_name = get_device_name()


# [EXPLAIN] `create_device_mesh` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_device_mesh(world_size, fsdp_size):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fsdp_size < 0 or fsdp_size >= world_size:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return device_mesh


# [EXPLAIN] `get_sharding_strategy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_sharding_strategy(device_mesh):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.fsdp import ShardingStrategy

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if device_mesh.ndim == 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharding_strategy = ShardingStrategy.FULL_SHARD
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif device_mesh.ndim == 2:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sharding_strategy


# [EXPLAIN] `ActorRolloutRefWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ActorRolloutRefWorker(Worker):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: DictConfig, role: str):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch.distributed

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ.get("RANK", 0))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", rank=rank, world_size=world_size)

        # build device mesh for FSDP
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_device_mesh = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp = world_size // self.ulysses_sequence_parallel_size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.ulysses_sequence_parallel_size > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._lora_rank = self.config.model.get('lora_rank', 0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_lora = self._lora_rank > 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.role = role
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_param = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_optimizer = False
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.actor.ppo_mini_batch_size > 0, f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.actor.ppo_micro_batch_size is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    # [EXPLAIN] `_build_model_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch import optim
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_dtypes import PrecisionType

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert role in ["actor", "ref"]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_dtype = fsdp_config.get("model_dtype", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch_dtype is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
                
        # patch for kimi-vl
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_model_config.text_config.topk_method = "greedy"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        override_config_kwargs.update(override_model_config)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with init_context(), warnings.catch_warnings():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.simplefilter("ignore")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actor_module_class = AutoModelForVision2Seq
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actor_module_class = AutoModelForCausalLM

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if use_liger:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _apply_liger_kernel_to_instance(model=actor_module)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fused_kernels_backend = "torch"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            actor_module.to(torch_dtype)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if enable_gradient_checkpointing:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_lora:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("Applying LoRA to actor module")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_model_size(actor_module)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mixed_precision_config is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_dtype = torch.bfloat16
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reduce_dtype = torch.float32
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            buffer_dtype = torch.float32

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get('lora_rank', 0) > 0)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            auto_wrap_policy = None

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"wrap_policy: {auto_wrap_policy}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_mesh = self.device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        # [EXPLAIN] OPD teacher は `role="ref"` でこの分岐へ入り、actor と異なり parameter CPUOffload が有効になる。
        # [EXPLAIN] task 別 teacher worker group は同じ resource pool に colocate されるが、常時 GPU 常駐ではない。
        # [EXPLAIN] forward 時の FSDP AllGather/一時 load、計算後の reshard/CPU 戻しにより GPU を逐次共有する。
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_strategy = self.config.actor.strategy
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if fsdp_strategy == "fsdp":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif fsdp_strategy == "fsdp2":
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if role == "actor" and fsdp_config.offload_policy:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._is_offload_param = False
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._is_offload_optimizer = False
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_state = actor_module.state_dict()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_module_fsdp = actor_module
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if enable_activation_offload:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if role == "actor" and optim_config is not None:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_steps = optim_config.get("total_training_steps", 0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            warmup_style = optim_config.get("warmup_style", "constant")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_cycles = optim_config.get("num_cycles", 0.5)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if num_warmup_steps < 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if warmup_style == "constant":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif warmup_style == "cosine":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_optimizer = None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_lr_scheduler = None

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    # [EXPLAIN] `_build_rollout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_rollout(self, trust_remote_code=False):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infer_tp = self.config.rollout.tensor_model_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp = self.world_size // infer_tp
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout_name = self.config.rollout.name
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rollout_name == "hf":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout import HFRollout
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.sharding_manager.base import BaseShardingManager

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif rollout_name == "vllm":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lora_kwargs = {'lora_kwargs': {"enable_lora":True, "max_loras":1, "max_lora_rank":self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if vllm_mode == "customized":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif vllm_mode == "spmd":
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_params = torch.distributed.get_world_size() == 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get('layered_summon', False),
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif rollout_name in ["sglang", "sglang_async"]:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if rollout_name == "sglang_async":
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. "
                    "Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.distributed.get_world_size() == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.rollout.load_format = "dummy_hf"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rollout, rollout_sharding_manager

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        import_external_libs(self.config.model.get("external_lib", None))

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_shm = self.config.model.get('use_shm', False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_actor:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                optim_config = self.config.actor.optim
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                fsdp_config = self.config.actor.fsdp_config
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                optim_config = None
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                fsdp_config = OmegaConf.create()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fsdp_version(self.actor_module_fsdp) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_param:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_optimizer:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            OmegaConf.set_struct(self.config.actor, True)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with open_dict(self.config.actor):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.actor.use_remove_padding = use_remove_padding
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.actor.use_fused_kernels = use_fused_kernels
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_rollout:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_ref:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            OmegaConf.set_struct(self.config.ref, True)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with open_dict(self.config.ref):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.ref.use_remove_padding = use_remove_padding
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.ref.use_fused_kernels = use_fused_kernels
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.flops_counter = FlopsCounter(self.actor_model_config)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] student actor の micro-batch forward、teacher 分布との KL、gradient accumulation、optimizer step を実行する。
    def update_actor(self, data: DataProto):
        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_actor
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_torch_device().current_device())

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with Timer(name="update_policy", logger=None) as timer:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metrics = self.actor.update_policy(data=data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            delta_time = timer.last
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_num_tokens = data.meta_info["global_token_num"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics["actor/lr"] = lr
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto(meta_info={"metrics": metrics})

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = output.to("cpu")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = prompts.to(get_torch_device().current_device())

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_rollout

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        prompts.meta_info.update(meta_info)

        # ROLLOUT_KEEP_VLLM_AWAKE session mode: when begin_rollout_session() has
        # entered the sharding manager for the whole multi-turn rollout, skip the
        # per-turn __enter__/__exit__ (which would re-sync the *unchanged* actor
        # weights to vLLM and sleep/wake the engine on every turn). The actor
        # weights are frozen during a rollout, so a single sync at session start
        # produces the identical weights for every turn -> generation is unchanged.
        # Only the per-turn data sharding (preprocess/postprocess) still runs.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if getattr(self, "_rollout_session_active", False):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.rollout.generate_sequences(prompts=prompts)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.rollout_sharding_manager.postprocess_data(output)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with self.rollout_sharding_manager:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.rollout.generate_sequences(prompts=prompts)

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After rollout generation", logger=logger)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.rollout_sharding_manager.postprocess_data(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")

        # clear kv cache
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `begin_rollout_session` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def begin_rollout_session(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Open the rollout sharding manager once for an entire multi-turn rollout.

        Without this, generate_sequences() re-enters the sharding manager every
        turn, which re-gathers the FSDP state_dict, re-syncs the full model weights
        to vLLM, and wakes/sleeps the engine — ~50x per rollout, all redundant
        because the actor weights are frozen during a rollout. Hoisting the
        enter/exit to the rollout boundary keeps vLLM awake and the weights synced
        once. Accuracy-safe: identical (frozen) weights are used on every turn, so
        the generated tokens are unchanged. No-op unless the vLLM sharding manager
        is in use. Paired with end_rollout_session() in a finally block.
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_rollout
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(self.rollout_sharding_manager, FSDPVLLMShardingManager):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if getattr(self, "_rollout_session_active", False):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.rollout_sharding_manager.__enter__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._rollout_session_active = True

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `end_rollout_session` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def end_rollout_session(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Close the sharding manager opened by begin_rollout_session(), restoring
        exactly the post-generation state of the non-session path (sleep/offload
        vLLM, restore RNG + train mode, empty cache)."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not getattr(self, "_rollout_session_active", False):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._rollout_session_active = False
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.rollout_sharding_manager.__exit__(None, None, None)



    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `compute_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_actor
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from contextlib import nullcontext
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_lora = data.meta_info.pop("is_lora", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with adapter_ctx:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.actor.actor_module._handle.reshard(True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `compute_ref_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_ref_log_prob(self, data: DataProto):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            data.meta_info['is_lora'] = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = DataProto.from_dict(tensors={'ref_log_prob': data.batch['old_log_probs']})
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = micro_batch_size
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["temperature"] = self.config.rollout.temperature
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.ref_policy.actor_module._handle.reshard(True)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `compute_ref_topk_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_ref_topk_log_prob(self, data: DataProto):
        # [EXPLAIN] teacher forward は ref policy の no-grad 経路で実行し、response 各位置について
        # [EXPLAIN] top-k token ID と full-vocabulary log-softmax から gather した値を CPU DataProto で返す。
        # [EXPLAIN] ID は整数精度が必要で bf16 に格納できないため、DP padding を通る shared Tensor 経路では
        # [EXPLAIN] float32 を介した後に long へ戻し、最終 shape (batch,response_length,k) を維持する。
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Teacher-side top-k log-probs for OPD distillation: per response token,
        the teacher's top-k token ids and the teacher's full-vocab log-softmax at
        those ids. ``topk_k`` is taken from data.meta_info (default 20)."""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_ref
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_k = int(data.meta_info.get("topk_k", 20))
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["temperature"] = self.config.rollout.temperature
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            topk_logprob, topk_ids = self.ref_policy.compute_topk_log_prob(data=data, topk_k=topk_k)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto.from_dict(
                tensors={"teacher_topk_logprobs": topk_logprob, "teacher_topk_ids": topk_ids},
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.ref_policy.actor_module._handle.reshard(True)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_actor

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.barrier()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lora_save_path = os.path.join(local_path, "lora_adapter")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            peft_config = {}
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dist.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.makedirs(lora_save_path, exist_ok=True)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                peft_config = asdict(self.actor_module.peft_config.get('default', {}))
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                peft_config['task_type'] = peft_config['task_type'].value
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                peft_config['peft_type'] = peft_config['peft_type'].value
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                peft_config['target_modules'] = list(peft_config['target_modules'])
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(self.actor_module_fsdp, FSDP):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.actor_module_fsdp = self.actor_module_fsdp.cuda()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if dist.get_rank() == 0:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding='utf-8') as f:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if dist.get_rank() == 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"[rank-{self.rank}]: Save LoRA Adapter Error ({e})")

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.barrier()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dist.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_optimizer(self.actor_optimizer)


# [EXPLAIN] `CriticWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CriticWorker(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch.distributed

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = torch.distributed.get_world_size()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.device_mesh import init_device_mesh

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_size = self.config.model.fsdp_config.fsdp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_device_mesh = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp = world_size // self.ulysses_sequence_parallel_size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.ulysses_sequence_parallel_size > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.ppo_micro_batch_size is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.ppo_micro_batch_size_per_gpu is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_lora = self.config.model.get('lora_rank', 0) > 0

    # [EXPLAIN] `_build_critic_model_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch import optim
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import MixedPrecision

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.model import print_model_size
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_dtypes import PrecisionType

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_shm = config.model.get('use_shm', False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        override_config_kwargs.update(override_config)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Critic overriding config {override_config_kwargs}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers import AutoConfig, AutoModelForTokenClassification

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation="flash_attention_2", trust_remote_code=config.model.get("trust_remote_code", False))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_model_config.text_config.topk_method = "greedy"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with init_context(), warnings.catch_warnings():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.simplefilter("ignore")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_model_config.classifier_dropout = 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_model_config.hidden_dropout = "0"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            use_remove_padding = config.model.get("use_remove_padding", False)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            critic_module.to(torch_dtype)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.model.get("enable_gradient_checkpointing", False):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_lora:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Applying LoRA to critic module")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lora_config = {
                'task_type': TaskType.CAUSAL_LM,
                'r': self.config.model.lora_rank,
                'lora_alpha': self.config.model.lora_alpha,
                'target_modules': convert_to_regular_types(self.config.model.target_modules),
                'bias': "none",
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_model_size(critic_module)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_model_config = critic_model_config

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_config = self.config.model.fsdp_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mixed_precision_config is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_dtype = torch.bfloat16
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reduce_dtype = torch.float32
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            buffer_dtype = torch.float32

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.get('lora_rank', 0) > 0)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before critic FSDP", logger=None)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_mesh = self.device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.strategy == "fsdp":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=False,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif config.strategy == "fsdp2":
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            offload_policy = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fsdp_config.offload_policy:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._is_offload_param = False
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._is_offload_optimizer = False
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_state = critic_module.state_dict()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.model.get("enable_activation_offload", False):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After critic FSDP", logger=None)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_steps = config.optim.get("total_training_steps", 0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        warmup_style = config.optim.get("warmup_style", "constant")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if num_warmup_steps < 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if warmup_style == "constant":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif warmup_style == "cosine":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return critic_module, critic_optimizer, critic_lr_scheduler

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        import_external_libs(self.config.model.get("external_lib", None))

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.critic import DataParallelPPOCritic

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.critic_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.flops_counter = FlopsCounter(self.critic_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `compute_values` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_values(self, data: DataProto):
        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.critic_module)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = micro_batch_size
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = self.critic.compute_values(data=data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto.from_dict(tensors={"values": values})
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.critic_module)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `update_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_critic(self, data: DataProto):
        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_torch_device().current_device())

        # perform forward computation
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with Timer(name="update_critic", logger=None) as timer:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metrics = self.critic.update_critic(data=data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            delta_time = timer.last

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_num_tokens = data.meta_info["global_token_num"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_lr_scheduler.step()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics["critic/lr"] = lr

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto(batch=None, meta_info={"metrics": metrics})
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.critic_module)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.critic_module)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.critic_module)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.critic_module)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
# [EXPLAIN] `RewardModelWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RewardModelWorker(Worker):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch.distributed

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = torch.distributed.get_world_size()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.device_mesh import init_device_mesh

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_size = self.config.model.fsdp_config.fsdp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_device_mesh = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp = world_size // self.ulysses_sequence_parallel_size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.ulysses_sequence_parallel_size > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.micro_batch_size is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    # [EXPLAIN] `_build_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_model(self, config):
        # the following line is necessary
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import CPUOffload
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers import AutoConfig, AutoModelForTokenClassification

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_shm = config.model.get('use_shm', False)
        # download the checkpoint from hdfs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.input_tokenizer is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._do_switch_chat_template = False
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._do_switch_chat_template = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trust_remote_code = config.model.get("trust_remote_code", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with init_context(), warnings.catch_warnings():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.simplefilter("ignore")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_config.classifier_dropout = 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_module.to(torch.bfloat16)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_mesh = self.device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.strategy == "fsdp":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=False,
                device_mesh=self.device_mesh,
            )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif config.strategy == "fsdp2":
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_state = reward_module.state_dict()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return reward_module

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        import_external_libs(self.config.model.get("external_lib", None))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_module = self._build_model(config=self.config)

    # [EXPLAIN] `_forward_micro_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_micro_batch(self, micro_batch):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_cuda_available:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif is_npu_available:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = micro_batch["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_size, seqlen = input_ids.shape
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = micro_batch["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = micro_batch["position_ids"]

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.use_remove_padding:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.ulysses_sequence_parallel_size > 1:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward_rmpad = output.logits
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.ulysses_sequence_parallel_size > 1:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_score = output.logits  # (batch_size, seq_len, 1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return rm_score

    # [EXPLAIN] `_expand_to_token_level` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = data.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = data.batch["position_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = data.batch["responses"].shape[-1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_scores = token_level_scores[:, -response_length:]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return token_level_scores

    # [EXPLAIN] `_switch_chat_template` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _switch_chat_template(self, data: DataProto):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_max_length = data.batch["attention_mask"].shape[-1]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_tokenizer = self.input_tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        target_tokenizer = self.tokenizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_input_ids = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_attention_mask = []

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_ids = data.batch["responses"][i]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_length = response_ids.shape[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = response.replace(src_tokenizer.eos_token, "")

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            chat.append({"role": "assistant", "content": response})

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0 and i == 0:
                # for debugging purpose
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_length = self.config.get("max_length", src_max_length)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if max_length is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                max_length = src_max_length

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rm_input_ids.append(input_ids)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rm_attention_mask.append(attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.from_dict(rm_inputs)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `compute_rm_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_rm_score(self, data: DataProto):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import itertools

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(get_torch_device().current_device())
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._do_switch_chat_template:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_data = self._switch_chat_template(data)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_input_ids = data.batch["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_attention_mask = data.batch["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_position_ids = data.batch["position_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_data.batch = rm_data.batch.to(get_torch_device().current_device())

        # perform forward computation
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.ulysses_sharding_manager:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            use_dynamic_bsz = self.config.use_dynamic_bsz
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if use_dynamic_bsz:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for micro_batch in micro_batches:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_score = self._forward_micro_batch(micro_batch)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                output.append(rm_score)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores = torch.cat(output, dim=0)  # (batch_size)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if use_dynamic_bsz:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                indices = list(itertools.chain.from_iterable(indices))
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores = scores[revert_indices]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.reward_module._handle.reshard(True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output


# ================================= Async related workers =================================
# [EXPLAIN] `AsyncActorRolloutRefWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    # [EXPLAIN] `_build_rollout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_rollout(self, trust_remote_code=False):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout.sharding_manager = rollout_sharding_manager

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rollout, rollout_sharding_manager

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    # [EXPLAIN] `execute_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.vllm_tp_rank == 0 and method != "execute_model":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.rollout.execute_method(method, *args, **kwargs)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    # [EXPLAIN] `resume` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def resume(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.rollout.resume()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    # [EXPLAIN] `offload` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.rollout.offload()
