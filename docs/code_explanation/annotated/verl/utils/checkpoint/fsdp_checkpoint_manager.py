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
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from accelerate import init_empty_weights
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import is_cuda_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local, is_non_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .checkpoint_manager import BaseCheckpointManager


# [EXPLAIN] `FSDPCheckpointManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPCheckpointManager(BaseCheckpointManager):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if checkpoint_contents is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            checkpoint_contents = ["model", "optimizer", "extra"]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if processing_class is None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            processing_class = kwargs.pop("tokenizer")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_path is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # every rank download its own checkpoint
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = copy_to_local(remote_model_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_optim_path = copy_to_local(remote_optim_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_state_dict = torch.load(local_model_path, weights_only=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if del_local_after_load:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.model.load_state_dict(model_state_dict)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.optimizer is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.load_rng_state(extra_state_dict["rng"])

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.lr_scheduler is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_path is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

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
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with warnings.catch_warnings():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.simplefilter("ignore")
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                model_state_dict = self.model.state_dict()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.save(model_state_dict, model_path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.save(extra_state_dict, extra_path)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fsdp_version(self.model) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                unwrap_model = self.model._fsdp_wrapped_module
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                unwrap_model = self.model

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_config = unwrap_model.config
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                generation_config.save_pretrained(local_path)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                generation_config = None

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model_config.save_pretrained(local_path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "hf_model" in self.checkpoint_contents:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hf_local_path = os.path.join(local_path, "huggingface")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state_dict = self.model.state_dict()

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "ForTokenClassification" in model_config.architectures[0]:
                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                    from transformers import AutoModelForTokenClassification

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    auto_model_cls = AutoModelForTokenClassification
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif "ForCausalLM" in model_config.architectures[0]:
                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                    from transformers import AutoModelForCausalLM

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    auto_model_cls = AutoModelForCausalLM
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                    from transformers import AutoModelForVision2Seq

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    auto_model_cls = AutoModelForVision2Seq
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with init_empty_weights():
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                save_model.to_empty(device="cpu")

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if save_model.can_generate():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if generation_config is not None:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        save_model.generation_config = generation_config
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.processing_class.save_pretrained(hf_local_path)
                # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
                del state_dict
                # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
                del save_model

            # wait for rank0 to dump hf_model to local
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.barrier()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.previous_saved_paths.append(local_path)
