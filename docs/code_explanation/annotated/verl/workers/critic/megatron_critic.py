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
Implement a multiprocess PPOCritic
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Iterable

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.pipeline_parallel import get_forward_backward_func
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo import core_algos
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron.pipeline_parallel import make_batch_generator
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import append_to_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import broadcast_dict_tensor, masked_mean
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.critic import BasePPOCritic

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `MegatronPPOCritic` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronPPOCritic(BasePPOCritic):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        critic_module: nn.ModuleList,
        critic_optimizer: DistributedOptimizer,
        critic_optimizer_config: OptimizerConfig,
    ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config=config)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._validate_config(config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config  # huggingface config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tf_config = tf_config  # mcore transformer config

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_module = critic_module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_optimizer = critic_optimizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_optimizer_config = critic_optimizer_config

        # we create a separate nametuple for optimizer step so that global args won't affect it.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

    # [EXPLAIN] `_validate_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _validate_config(self, config) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Validate config options not implemented for Megatron backend"""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.shuffle:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.megatron.tensor_model_parallel_size == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("[Warining] Because critic tp size == 1, set sp to False")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config.megatron.sequence_parallel = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger("megatron critic", logger=logger)
    # [EXPLAIN] `compute_values` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_values(self, data: DataProto) -> DataProto:
        # data.batch = data.batch.to(self.critic_module.module.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        responses = data.batch["responses"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = data.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_token_len = data.meta_info.get("max_token_len", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert micro_batch_size is not None, "micro batch size is needed for forward compute"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = responses.size(1)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.forward_backward_batch(data=data, forward_only=True, use_dynamic_bsz=use_dynamic_bsz, micro_batch_size=micro_batch_size, max_token_len=max_token_len, mini_batch_size=None)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                values = [o["vpreds"] for o in output["output"]]  # (bs, seq_size, vocal_size)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                values = torch.cat(values, dim=0).to(torch.float32)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if use_dynamic_bsz:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    indices = output["indices"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    indices = list(itertools.chain.from_iterable(indices))
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    values = values[revert_indices]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # each tp ranks should contain the same value
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = values * attention_mask
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = values[:, -response_length - 1 : -1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = values.contiguous()

            # sync among pp ranks
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.broadcast(
                tensor=values,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        # add empty cache after each compute
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return values

    # [EXPLAIN] `make_minibatch_iterator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.select(batch_keys=select_keys)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    # [EXPLAIN] `forward_backward_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward_backward_batch(self, data: DataProto, forward_only=False, use_dynamic_bsz=False, micro_batch_size=None, max_token_len=None, mini_batch_size=None):
        # broadcast from last pp rank to all other pp ranks
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mini_batch = data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mini_batch.batch = mini_batch.batch.contiguous()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        broadcast_dict_tensor(mini_batch.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indices = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if vpp_size is not None and vpp_size > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, num_batches_divided_by=microbatch_group_size_per_vp_stage, max_token_len=max_token_len)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_seqlen = max_token_len
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert micro_batch_size is not None, "micro_batch_size is needed to be passed in when not using dynamic batch size"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = mini_batch.batch.split(micro_batch_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            seq_len = micro_batches[0]["input_ids"].shape[1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_seqlen = micro_batch_size * seq_len
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_micro_batch = len(micro_batches)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_backward_func = get_forward_backward_func()

        # [EXPLAIN] `loss_func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def loss_func(output, data, meta_info):
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            nonlocal use_dynamic_bsz

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if forward_only:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return torch.tensor(1.0, device=output.device), {"vpreds": output}

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            responses = data["responses"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = data["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = data["values"]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            returns = data["returns"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_length = responses.size(1)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_mask = attention_mask[:, -response_length:]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cliprange_value = self.config.cliprange_value

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vpreds = output  # (bs, sequence_length)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vpreds = vpreds[:, -response_length - 1 : -1]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange_value,
                loss_agg_mode=self.config.loss_agg_mode,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stats = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return vf_loss, stats

        # [EXPLAIN] `forward_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def forward_step(batch_iter, model):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch = next(batch_iter)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = batch["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = batch["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = batch["position_ids"]
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import get_mcore_forward_fn

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            forward_fn = get_mcore_forward_fn(self.hf_config)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return output, partial(loss_func, data=batch, meta_info={})

        # batch should be a list of batches inside micro-batches
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.critic_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        losses_reduced = {"output": losses_reduced}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            losses_reduced["indices"] = indices
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return losses_reduced

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger("megatron critic", logger=logger)
    # [EXPLAIN] `update_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_critic(self, dataloader: Iterable[DataProto]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {}

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for data in dataloader:
            # data = data.batch.to(self.critic_module.device)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for chunk in self.critic_module:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                chunk.zero_grad_buffer()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.use_dynamic_bsz:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metric_micro_batch = self.forward_backward_batch(data, forward_only=False, use_dynamic_bsz=self.config.use_dynamic_bsz, micro_batch_size=micro_batch_size, max_token_len=max_token_len, mini_batch_size=self.config.ppo_mini_batch_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metric_micro_batch = metric_micro_batch["output"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            learning_rate = self.critic_optimizer.param_groups[-1]["lr"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = {"critic/grad_norm": grad_norm, "critic/lr": learning_rate}
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            append_to_dict(metrics, data)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for metric in metric_micro_batch:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        # add empty cache after each compute
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return metrics
