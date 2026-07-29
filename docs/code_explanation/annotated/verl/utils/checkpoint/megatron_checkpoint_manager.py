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
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import mpu, tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.dist_checkpointing.mapping import ShardedObject
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import GenerationConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.weight_loader_registry import get_weight_saver
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import is_non_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import (
    get_hf_config_and_tokenizer_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_model_checkpoint_path,
    get_optimizer_checkpoint_path,
    get_rng_states_checkpoint_path,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .checkpoint_manager import BaseCheckpointManager


# [EXPLAIN] `MegatronCheckpointManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronCheckpointManager(BaseCheckpointManager):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        config,
        model_config,
        role,
        model: torch.nn.ModuleList,
        arch: str,
        hf_config,
        param_dtype: torch.dtype,
        share_embeddings_and_output_weights: bool,
        tokenizer,
        optimizer,
        use_distributed_optimizer: bool,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if checkpoint_contents is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            checkpoint_contents = ["model", "optimizer", "extra"]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            model,
            optimizer=optimizer,
            lr_scheduler=None,
            processing_class=tokenizer,
            checkpoint_contents=checkpoint_contents,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.arch = arch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.role = role
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.is_value_model = False
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.role in ["reward", "critic"]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.is_value_model = True
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.param_dtype = param_dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_path = self.config.model.path
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_distributed_optimizer = use_distributed_optimizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank = torch.distributed.get_rank()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weight_saver = get_weight_saver(self.arch)

    # [EXPLAIN] `get_rng_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rng_state(self, use_dist_ckpt: bool = False, data_parallel_random_init: bool = False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """collect rng state across data parallel ranks"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng_state_list = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state_list = [rng_state]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dist_ckpt:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tp_rank = mpu.get_tensor_model_parallel_rank()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tp_size = mpu.get_tensor_model_parallel_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cp_rank = mpu.get_context_parallel_rank()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cp_size = mpu.get_context_parallel_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size, cp_size),
                (pp_rank, tp_rank, cp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rng_state_list

    # [EXPLAIN] `get_checkpoint_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_checkpoint_name(
        self,
        checkpoints_path,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        cp_rank=None,
        expert_parallel=None,
        expert_rank=None,
        return_base_dir=True,
        basename="model.pt",
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pipeline_parallel is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor_rank is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pipeline_rank is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cp_rank is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cp_rank = mpu.get_context_parallel_rank()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if expert_parallel is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if expert_rank is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.

        # due to the fact that models are identical across cp ranks, cp rank is not used in the checkpoint path
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not pipeline_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if expert_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            common_path = common_path + f"_{expert_rank:03d}"

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(common_path, exist_ok=True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if return_base_dir:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return common_path
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return os.path.join(common_path, basename)

    # [EXPLAIN] `load_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_optimizer(self, ckpt_path):
        # TODO: Check Optimizer format and distributed optimizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optimizer_path = get_optimizer_checkpoint_path(ckpt_path)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Loading optimizer from {optimizer_path}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.optimizer.load_parameter_state(optimizer_path)

    # [EXPLAIN] `load_rng_states` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_rng_states(self, ckpt_path, data_parallel_random_init=False, use_dist_ckpt=False):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng_state_path = get_rng_states_checkpoint_path(ckpt_path, only_rank0_save=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Loading rng states from {rng_state_path}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng_state = torch.load(rng_state_path, weights_only=False)
        # access rng_state for data parallel rank
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not use_dist_ckpt:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state = rng_state[mpu.get_data_parallel_rank()] if data_parallel_random_init else rng_state[0]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.setstate(rng_state["random_rng_state"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.set_state(rng_state["np_rng_state"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.set_rng_state(rng_state["torch_rng_state"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
        # Check for empty states array
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not rng_state["rng_tracker_states"]:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise KeyError
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])

    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_path is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "model" in self.checkpoint_contents:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_path = get_model_checkpoint_path(local_path)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ckpt_name = self.get_checkpoint_name(model_path, return_base_dir=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dicts = torch.load(os.path.join(ckpt_name), weights_only=False)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(state_dicts) == len(self.model), f"state_dicts length: {len(state_dicts)} mismatch with model length: {len(self.model)}"
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for vpp_rank, (state_dict, model) in enumerate(zip(state_dicts, self.model)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                model.load_state_dict(state_dict)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Loaded sharded model checkpoint from {model_path}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "optimizer" in self.checkpoint_contents:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.load_optimizer(local_path)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "extra" in self.checkpoint_contents:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.load_rng_states(local_path)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if del_local_after_load:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.remove(local_path) if is_non_local(local_path) else None
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        # record the previous global step
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.previous_global_step = global_step

        # remove previous local_path
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = self.local_mkdir(local_path)

        # Save Model
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "model" in self.checkpoint_contents and mpu.get_data_parallel_rank() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dicts = []

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for vpp_rank, model in enumerate(self.model):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state_dict = model.state_dict()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                state_dicts.append(state_dict)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saving sharded model checkpoint to {local_path}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_ckpt_path = get_model_checkpoint_path(local_path)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hf_config_and_tokenizer_path = get_hf_config_and_tokenizer_checkpoint_path(local_path)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ckpt_name = self.get_checkpoint_name(model_ckpt_path, return_base_dir=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.save(state_dicts, os.path.join(ckpt_name))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saved checkpoint to {model_ckpt_path}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.processing_class.save_pretrained(hf_config_and_tokenizer_path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.hf_config.save_pretrained(hf_config_and_tokenizer_path)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                    try:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        generation_config.save_pretrained(hf_config_and_tokenizer_path)
                    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        pass
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hdfs_path is not None:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Uploading checkpoint to {hdfs_path}")
                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                    from verl.utils import hdfs_io

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    hdfs_io.copy(src=model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    hdfs_io.copy(src=hf_config_and_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "hf_model" in self.checkpoint_contents:
            # wait for everyone to dump to local
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = self.weight_saver(
                self.model,
                self.hf_config,
                dtype=self.param_dtype,
                is_value_model=self.is_value_model,
                tie_word_embeddings=self.share_embeddings_and_output_weights,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.barrier()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"self.param_dtype: {self.param_dtype}")
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key in state_dict.keys():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"state_dict[key].dtype: {key} {state_dict[key].dtype}")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                import warnings

                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from accelerate import init_empty_weights

                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with init_empty_weights(), warnings.catch_warnings():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    warnings.simplefilter("ignore")
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if "mistral7b-rm" in self.config.model.path:
                        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                        from transformers import MistralForSequenceClassification

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        model = MistralForSequenceClassification.from_pretrained(self.config.model.path)  # use score head instead of lm_head
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        state_dict["score.weight"] = state_dict["score.weight"]
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                        from transformers import AutoModelForCausalLM

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        model = AutoModelForCausalLM.from_pretrained(self.config.model.path, torch_dtype="auto")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.processing_class.save_pretrained(hf_model_ckpt_path)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hdfs_path is not None:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Uploading checkpoint to {hdfs_path}")
                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                    from verl.utils import hdfs_io

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    hdfs_io.copy(src=hf_model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)

        # Save Optimizer
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "optimizer" in self.checkpoint_contents:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.barrier()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            optimizer_path = get_optimizer_checkpoint_path(local_path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.optimizer.save_parameter_state(optimizer_path)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"saving optimizer state to {optimizer_path}")

        # Save RNG States
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "extra" in self.checkpoint_contents:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.barrier()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state_path = get_rng_states_checkpoint_path(local_path, only_rank0_save=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rng_state = self.get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.save(rng_state, rng_state_path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Rank {self.rank} saving rng states to {rng_state_path}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.previous_saved_paths.append(local_path)
