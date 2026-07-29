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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["NCCL_DEBUG"] = "WARN"
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import nullcontext

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hydra
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from peft import LoraConfig, TaskType, get_peft_model
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn, optim
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import DataLoader, Dataset, DistributedSampler
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.hdfs_io as hdfs_io
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.dataset import SFTDataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import log_gpu_memory_usage
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import convert_to_regular_types
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.tracking import Tracking
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_cuda_available:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
elif is_npu_available:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `extract_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def extract_step(path):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    match = re.search(r"global_step_(\d+)", path)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if match:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return int(match.group(1))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return None


# [EXPLAIN] `FSDPSFTTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPSFTTrainer:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_device_mesh = ulysses_device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.data.chat_template is not None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._normalize_config_bsz()

        # Set sequence parallel size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Using remove padding: {self.use_remove_padding}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._build_dataloader(train_dataset, val_dataset)
        # build model
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(self.config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_name = get_device_name()

    # [EXPLAIN] `_normalize_config_bsz` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _normalize_config_bsz(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Normalize batch size by dp {dp_size}")

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.config.data.train_batch_size //= dp_size

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    # [EXPLAIN] `_build_dataloader` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = self.config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.ulysses_sequence_parallel_size > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            world_size = self.ulysses_device_mesh.size(0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.ulysses_device_mesh.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = self.device_mesh.get_rank()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            world_size = self.device_mesh.size()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    # [EXPLAIN] `_build_model_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import importlib

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            importlib.import_module(self.config.model.external_lib)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before model allocation", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.ulysses_sequence_parallel_size > 1:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with init_context():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.model.get("use_liger", False):
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _apply_liger_kernel_to_instance(model=self.model)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.model.get("lora_rank", 0) > 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.enable_gradient_checkpointing:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After model allocation", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(auto_wrap_policy)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.config.model.fsdp_config.cpu_offload:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cpu_offload = None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_strategy = self.config.model.strategy
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if fsdp_strategy == "fsdp":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
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
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_state = self.model.state_dict()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.fsdp_model = self.model
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.steps_per_epoch = len(self.train_dataloader)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.optim.lr_scheduler == "wsd":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    # [EXPLAIN] `_compute_loss_and_backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _compute_loss_and_backward(self, batch, do_backward=True):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Compute loss with optional sequence parallelism and remove padding features"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = batch["input_ids"].to(self.device_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = batch["attention_mask"].to(self.device_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = batch["position_ids"].to(self.device_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        context = self.sharding_manager if use_sp else nullcontext()
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not use_sp:
                # Standard forward pass without sequence parallel
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                labels = input_ids[:, 1:].contiguous()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits = output.logits

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shift_logits = logits[..., :-1, :].contiguous()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shift_labels = labels.contiguous()
                # Flatten the tokens
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shift_labels = shift_labels.to(shift_logits.device)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = loss_fct(shift_logits, shift_labels)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = loss * loss_mask.to(loss.device)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                batch_size, seqlen = input_ids.shape
                # Remove padding
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits_rmpad = output.logits.squeeze(0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_loss = full_loss.reshape(-1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_mask = loss_mask.to(full_loss.device)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss = full_loss * loss_mask

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_token_this_rank = torch.sum(loss_mask)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.data.balance_dp_token:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_reduce(valid_token_this_rank)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dp_size = 1

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if do_backward:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                loss.backward()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return loss

    # [EXPLAIN] `training_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def training_step(self, batch: TensorDict):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.fsdp_model.train()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.optimizer.zero_grad()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_micro_batches = len(micro_batches)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        step_loss = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for micro_batch in micro_batches:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            step_loss += loss.item()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.strategy == 'fsdp':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.model.strategy == 'fsdp2':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.isfinite(grad_norm):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.optimizer.zero_grad()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.optimizer.step()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After optimizer step", logger=logger)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.lr_scheduler.step()

        # reduce loss across dp ranks
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lr = self.lr_scheduler.get_last_lr()[0]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After offload weights", logger=logger)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        step_loss = torch.tensor(step_loss).to(self.device_name)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_cuda_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif is_npu_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.all_reduce(step_loss)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            step_loss /= self.ulysses_device_mesh.size(0)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    # [EXPLAIN] `validation_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def validation_step(self, batch: TensorDict):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.fsdp_model.eval()
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_cuda_available:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif is_npu_available:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_reduce(loss)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                loss /= self.ulysses_device_mesh.size(0)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return loss

    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, step):
        # save checkpoint
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_strategy = self.config.model.strategy
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.device_mesh.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.makedirs(path, exist_ok=True)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.model.save_pretrained(path, state_dict=state_dict)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.tokenizer.save_pretrained(path)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.device_mesh.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.makedirs(path, exist_ok=True)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.model.save_pretrained(path, state_dict=state_dict)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.model_config.save_pretrained(path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.tokenizer.save_pretrained(path)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()

    # [EXPLAIN] rollout 生成から teacher forward、actor 更新、検証、checkpoint までの学習 phase を順序付ける trainer loop である。
    def fit(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rank == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.trainer.total_training_steps is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_training_steps = self.config.trainer.total_training_steps

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.total_training_steps = total_training_steps
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for epoch in range(self.config.trainer.total_epochs):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.train_sampler.set_epoch(epoch=epoch)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0
            ):
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                global_step += 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metric = self.training_step(data)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if rank == 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    tracking.log(data=metric, step=global_step)

                # for early exit validation
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    val_losses = []
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for val_data in self.val_dataloader:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        val_loss = self.validation_step(val_data)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        val_losses.append(val_loss)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if rank == 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        metric = {"val/loss": avg_val_loss.detach().item()}
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        tracking.log(data=metric, step=global_step)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    torch.distributed.barrier()

                    # Save final checkpoint
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self.save_checkpoint(step=global_step)
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return

            # validation
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_losses = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for data in self.val_dataloader:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                val_loss = self.validation_step(data)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                val_losses.append(val_loss)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if rank == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                val_loss = torch.mean(torch.stack(val_losses))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metric = {"val/loss": val_loss.detach().item()}
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tracking.log(data=metric, step=global_step)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.barrier()

            # save checkpoint
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.save_checkpoint(step=global_step)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main(config):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_name = get_device_name()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank, rank, world_size = initialize_global_process_group()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_size = world_size // config.ulysses_sequence_parallel_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils import hf_tokenizer

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    trainer.fit()


# [EXPLAIN] `create_sft_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_sft_dataset(data_paths, data_config, tokenizer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if data_config.custom_cls.get("path", None):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.import_utils import load_extern_type

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_config.get("multiturn", {}).get("enable", False):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dataset


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
