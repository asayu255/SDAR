# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.core_algos import agg_loss, agg_loss_with_sample_weights, compute_policy_loss, compute_policy_loss_gspo, kl_penalty, topk_kl_per_token
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import iter_task_row_masks
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import append_to_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import logprobs_from_logits
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.actor import BasePPOActor

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_cuda_available:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
elif is_npu_available:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["DataParallelPPOActor"]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `DataParallelPPOActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DataParallelPPOActor(BasePPOActor):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """When optimizer is None, it is Reference Policy"""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.actor_module = actor_module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.actor_optimizer = actor_optimizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_name = get_device_name()

    # [EXPLAIN] `_forward_micro_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, topk_k=None, topk_ids=None) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_out: optional top-k output for distillation KL. None unless
                topk_k or topk_ids is given. When ``topk_k`` is set (teacher mode):
                a tuple (topk_logprob, topk_ids) each (bs, response_len, k), where
                topk_logprob are full-vocab log-softmax values at the model's own
                top-k ids. When ``topk_ids`` is given (student mode): a tensor
                (bs, response_len, k) of this model's full-vocab log-softmax values
                gathered at the provided ids (carries gradient).
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_out = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (topk_k is not None or topk_ids is not None) and self.use_fused_kernels:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError("top-k KL forward is not supported with fused kernels")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = micro_batch["responses"].size(-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        multi_modal_inputs = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "multi_modal_inputs" in micro_batch:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key in micro_batch["multi_modal_inputs"][0].keys():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = micro_batch["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_size, seqlen = input_ids.shape
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = micro_batch["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = micro_batch["position_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            entropy = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if position_ids.dim() == 3:  # qwen2vl mrope
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.use_remove_padding:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if position_ids.dim() == 3:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_ulysses_sp:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                extra_args = {}
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_fused_kernels:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    extra_args["temperature"] = temperature
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    extra_args["return_dict"] = True

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_fused_kernels:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    inplace_backward = True
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if calculate_entropy or topk_k is not None or topk_ids is not None:
                        # top-k KL reads logits_rmpad after this call, so don't mutate it.
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        inplace_backward = False
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if calculate_entropy:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if calculate_entropy:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if calculate_entropy:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if calculate_entropy:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # top-k distillation log-probs (teacher: own top-k; student: gather at given ids)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if topk_k is not None or topk_ids is not None:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.use_fused_kernels or self.use_ulysses_sp:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise NotImplementedError("top-k KL forward is not supported with fused kernels or ulysses SP")
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lse = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)  # (total_nnz, 1)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if topk_k is not None:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        tvals, tids = torch.topk(logits_rmpad, k=topk_k, dim=-1)  # (total_nnz, k)
                        # Use float32 for pad_input: bf16 cannot represent vocab ids
                        # (>256) exactly, and float32 keeps log-probs precise.
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        t_lp_rmpad = (tvals - lse).float()
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        full_t_lp = pad_input(t_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        full_t_id = pad_input(tids.float(), indices=indices, batch=batch_size, seqlen=seqlen)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        topk_out = (
                            full_t_lp[:, -response_length - 1 : -1, :],
                            full_t_id[:, -response_length - 1 : -1, :].round().long(),
                        )
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        k = topk_ids.size(-1)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        full_ids = torch.zeros((batch_size, seqlen, k), dtype=torch.long, device=logits_rmpad.device)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        full_ids[:, -response_length - 1 : -1, :] = topk_ids
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        ids_rmpad = index_first_axis(rearrange(full_ids, "b s k -> (b s) k"), indices)  # (total_nnz, k)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        s_lp_rmpad = (logits_rmpad.gather(-1, ids_rmpad) - lse).float()  # (total_nnz, k), keeps grad
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        full_s_lp = pad_input(s_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        topk_out = full_s_lp[:, -response_length - 1 : -1, :]

            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:  # not using rmpad and no ulysses sp
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                extra_args = {}
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_fused_kernels:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    extra_args["temperature"] = temperature
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.use_fused_kernels:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    logits = output.logits

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logits.div_(temperature)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if calculate_entropy:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if topk_k is not None or topk_ids is not None:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (bsz, response_length, 1)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if topk_k is not None:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            tvals, tids = torch.topk(logits, k=topk_k, dim=-1)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            topk_out = ((tvals - lse).float(), tids.long())
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            topk_out = (logits.gather(-1, topk_ids) - lse).float()

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return entropy, log_probs, topk_out

    # [EXPLAIN] `_optimizer_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _optimizer_step(self):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.config.grad_clip is not None

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(self.actor_module, FSDP):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(self.actor_module, FSDPModule):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.isfinite(grad_norm):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.actor_optimizer.zero_grad()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.actor_optimizer.step()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return grad_norm

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="dp actor", logger=logger)
    # [EXPLAIN] `compute_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_module.eval()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = data.meta_info["micro_batch_size"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = data.select(batch_keys=select_keys).batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if has_multi_modal_inputs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_select_keys = ["multi_modal_inputs"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif use_dynamic_bsz:
            # split using dynamic bsz
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = batch.split(micro_batch_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropy_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for micro_batch in micro_batches:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(micro_batch, DataProto):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with torch.no_grad():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_probs_lst.append(log_probs)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if calculate_entropy:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                entropy_lst.append(entropy)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = torch.concat(log_probs_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropys = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if calculate_entropy:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            entropys = torch.concat(entropy_lst, dim=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices = list(itertools.chain.from_iterable(indices))
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_probs = log_probs[revert_indices]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return log_probs, entropys

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="dp actor", logger=logger)
    # [EXPLAIN] `compute_topk_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_topk_log_prob(self, data: DataProto, topk_k: int):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Teacher-side: per response token, the teacher's top-k token ids and the
        teacher's full-vocab log-softmax values at those ids.

        Returns:
            topk_logprob: (bs, response_length, k)
            topk_ids:     (bs, response_length, k) int64
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_module.eval()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = data.meta_info["micro_batch_size"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temperature = data.meta_info["temperature"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = data.select(batch_keys=select_keys).batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert not has_multi_modal_inputs, "top-k KL is not supported for multi-modal inputs"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = batch.split(micro_batch_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_logprob_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_ids_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for micro_batch in micro_batches:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(micro_batch, DataProto):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with torch.no_grad():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _, _, topk_out = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=False, topk_k=topk_k)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tlp, tids = topk_out
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            topk_logprob_lst.append(tlp)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            topk_ids_lst.append(tids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_logprob = torch.concat(topk_logprob_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        topk_ids = torch.concat(topk_ids_lst, dim=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            indices = list(itertools.chain.from_iterable(indices))
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(indices) == topk_logprob.size(0), f"{len(indices)} vs. {topk_logprob.size()}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            topk_logprob = topk_logprob[revert_indices]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            topk_ids = topk_ids[revert_indices]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return topk_logprob, topk_ids

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="dp actor", logger=logger)
    # [EXPLAIN] `update_policy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_policy(self, data: DataProto):
        # [EXPLAIN] Pure OPD の実効設定では global mini-batch 60 を rank/micro-batch へ分割し、
        # [EXPLAIN] 1 GPU あたり micro-batch 2 の forward/backward を gradient accumulation する。
        # [EXPLAIN] `response_mask` は (micro_batch,response_length) で、multi-turn の無効 token と padding を
        # [EXPLAIN] scalar loss の token-mean 分母・分子の双方から除外する。
        # make sure we are in training mode
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_module.train()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        multi_turn = data.meta_info.get("multi_turn", False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_teacher_kl_loss = self.config.get("use_teacher_kl_loss", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_kl_loss_type = self.config.get("teacher_kl_loss_type", "low_var_kl")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        teacher_topk_kl = use_teacher_kl_loss and teacher_kl_loss_type == "topk_kl"
        # Exact token-mean under dynamic bsz: scale each micro-batch by its valid-token
        # share of the mini-batch (token-weighted) instead of by sample count, so the
        # objective is grouping-invariant and matches the true global token-mean.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dynamic_bsz_token_scale = (
            self.config.use_dynamic_bsz
            and self.config.get("dynamic_bsz_token_scale", False)
            and self.config.loss_agg_mode == "token-mean"
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # advantages / old_log_probs are only needed by the policy-gradient (and SDL) paths.
        # Pure teacher-KL distillation (pg_loss_coef==0) does not produce them, so don't require them.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pg_loss_coef != 0:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            select_keys += ["old_log_probs", "advantages"]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.get("use_sdl_loss", False):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            select_keys.append("old_log_probs")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if multi_turn:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            select_keys.append("loss_mask")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.use_kl_loss:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            select_keys.append("ref_log_prob")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "kl_loss_coef" in data.batch:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                select_keys.append("kl_loss_coef")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False) or (use_teacher_kl_loss and not teacher_topk_kl):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            select_keys.append("teacher_log_probs")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if teacher_topk_kl:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]
        # Multitask runs tag every row with its task id (see RayPPOTrainer._attach_task_ids)
        # so the loss metrics below can also be reported per task. Absent in single-task runs.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_id_names = data.meta_info.get("task_id_names", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "task_ids" in data.batch.keys():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            select_keys.append("task_ids")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = data.select(batch_keys=select_keys).batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if has_multi_modal_inputs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_select_keys = ["multi_modal_inputs"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for epoch in range(self.config.ppo_epochs):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                mini_batch = data
                # Total valid response tokens in this mini-batch (for exact token-mean
                # scaling under dynamic bsz). Same mask rule as the per-micro loss below.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                minibatch_valid_tokens = None
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if dynamic_bsz_token_scale:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    _resp_len = mini_batch["responses"].size(1)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    _mb_mask = mini_batch["loss_mask"][:, -_resp_len:] if multi_turn else mini_batch["attention_mask"][:, -_resp_len:]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    minibatch_valid_tokens = float(_mb_mask.sum().clamp(min=1))
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if has_multi_modal_inputs:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif self.config.use_dynamic_bsz:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.actor_optimizer.zero_grad()

                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for data in micro_batches:
                    # Support all hardwares
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if isinstance(data, DataProto):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    responses = data["responses"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    response_length = responses.size(1)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    attention_mask = data["attention_mask"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    task_ids = data.get("task_ids", None) if task_id_names else None
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if multi_turn:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_mask = data["loss_mask"][:, -response_length:]
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        response_mask = attention_mask[:, -response_length:]

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    clip_ratio = self.config.clip_ratio
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    entropy_coeff = self.config.entropy_coeff
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    calculate_entropy = False
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if entropy_coeff != 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        calculate_entropy = True
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    fwd_topk_ids = data["teacher_topk_ids"] if teacher_topk_kl else None
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    entropy, log_prob, student_topk_logprobs = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, topk_ids=fwd_topk_ids)
                    
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if loss_mode == "vanilla":
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss_fn = compute_policy_loss
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif loss_mode == "gspo":
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss_fn = compute_policy_loss_gspo
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if pg_loss_coef != 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob = data["old_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        advantages = data["advantages"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # Pure teacher-KL distillation: no policy-gradient signal.
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_log_prob = data.get("old_log_probs", None)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        zero = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        pg_loss = pg_clipfrac = ppo_kl = pg_clipfrac_lower = zero

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if entropy_coeff != 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss = pg_loss * pg_loss_coef

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.use_kl_loss:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        kl_loss_coef = data.get("kl_loss_coef", None)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if kl_loss_coef is None:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            weighted_kl_loss = agg_loss_with_sample_weights(
                                loss_mat=kld,
                                loss_mask=response_mask,
                                sample_weights=kl_loss_coef,
                                loss_agg_mode=loss_agg_mode,
                            )
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            policy_loss = policy_loss + weighted_kl_loss
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics["actor/kl_coef"] = kl_loss_coef.float().mean().detach().item()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.get("use_sdl_loss", False):
                        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_log_probs = data["teacher_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["actor/sdl_coef"] = sdl_coef

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.get("use_sdar_loss", False):
                        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_log_probs = data["teacher_log_probs"]
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics.update(sdar_metrics)
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["sdar/coef"] = sdar_coef

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if use_teacher_kl_loss:
                        # [EXPLAIN] entry point が `pg_loss_coef=0`、entropy=0、reference-KL/SDL/SDAR=false を注入するため、
                        # [EXPLAIN] `teacher_kl_loss * teacher_kl_coef` が実効 scalar loss の唯一の非ゼロ項になる。
                        # [EXPLAIN] monitoring reward や invalid-action penalty が reward Tensor を変えても、
                        # [EXPLAIN] thin loop は advantage/returns を作らないため、この加算式へは到達しない。
                        # On-policy distillation: KL between student and a (per-task) teacher,
                        # evaluated on the student's own on-policy responses. Only the student
                        # log-probs carry gradients; teacher values are detached upstream.
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if teacher_topk_kl:
                            # [EXPLAIN] student/teacher top-k log-prob はともに
                            # [EXPLAIN] (micro_batch,response_length,k) で、前者だけが gradient を保持する。
                            # Dense reverse KL over the teacher's top-k support (+ tail bucket).
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            teacher_kld = topk_kl_per_token(
                                student_topk_logprob=student_topk_logprobs,
                                teacher_topk_logprob=data["teacher_topk_logprobs"],
                            )
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # Single-sampled-token estimator (low_var_kl / kl / mse / abs).
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            teacher_kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=data["teacher_log_probs"],
                                kl_penalty=teacher_kl_loss_type,
                            )
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_kl_loss = agg_loss(loss_mat=teacher_kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # [EXPLAIN] token-mean により (micro_batch,response_length) を1 scalar へ集約し、
                        # [EXPLAIN] padding/非応答 token は response_mask で除外する。
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        teacher_kl_coef = self.config.get("teacher_kl_loss_coef", 1.0)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        metrics["actor/teacher_kl_loss"] = teacher_kl_loss.detach().item()
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metrics["actor/teacher_kl_coef"] = teacher_kl_coef

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.use_dynamic_bsz:
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if minibatch_valid_tokens is not None:
                            # Exact global token-mean: policy_loss is a token-mean
                            # (denominator = this micro-batch's valid tokens), so
                            # policy_loss * micro_valid_tokens = token-sum, and dividing
                            # by the mini-batch's total valid tokens makes the per-token
                            # weight independent of how tokens were grouped into
                            # micro-batches (unlike the sample-count factor below).
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            micro_valid_tokens = response_mask.sum()
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            loss = policy_loss * (micro_valid_tokens / minibatch_valid_tokens)
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # relative to the dynamic bsz (sample-count reweighting)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        loss = policy_loss / self.gradient_accumulation
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    loss.backward()

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if task_ids is not None:
                        # Same losses, re-aggregated over the rows of one task at a
                        # time. Diagnostics only: nothing here touches the graph the
                        # optimizer step above was built from.
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with torch.no_grad():
                            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                            for task, rows in iter_task_row_masks(task_ids, task_id_names):
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                task_response_mask = response_mask[rows]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                task_metrics = {}

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if pg_loss_coef != 0:
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    task_pg_loss, task_pg_clipfrac, task_ppo_kl, task_pg_clipfrac_lower = policy_loss_fn(
                                        old_log_prob=old_log_prob[rows],
                                        log_prob=log_prob[rows],
                                        advantages=advantages[rows],
                                        response_mask=task_response_mask,
                                        cliprange=clip_ratio,
                                        cliprange_low=clip_ratio_low,
                                        cliprange_high=clip_ratio_high,
                                        clip_ratio_c=clip_ratio_c,
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                                else:
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    task_pg_loss = task_pg_clipfrac = task_ppo_kl = task_pg_clipfrac_lower = zero
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                task_metrics[f"actor/pg_loss/{task}"] = task_pg_loss.detach().item()
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                task_metrics[f"actor/pg_clipfrac/{task}"] = task_pg_clipfrac.detach().item()
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                task_metrics[f"actor/ppo_kl/{task}"] = task_ppo_kl.detach().item()
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                task_metrics[f"actor/pg_clipfrac_lower/{task}"] = task_pg_clipfrac_lower.detach().item()

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if entropy_coeff != 0:
                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    task_metrics[f"actor/entropy_loss/{task}"] = (
                                        agg_loss(loss_mat=entropy[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.config.use_kl_loss:
                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    task_metrics[f"actor/kl_loss/{task}"] = (
                                        agg_loss(loss_mat=kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )
                                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                    if kl_loss_coef is not None:
                                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                        task_metrics[f"actor/kl_coef/{task}"] = kl_loss_coef[rows].float().mean().detach().item()

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.config.get("use_sdl_loss", False):
                                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                                    from verl.trainer.ppo.skillsd_utils import compute_sdl_loss

                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    task_metrics[f"actor/sdl_loss/{task}"] = compute_sdl_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        old_log_probs=old_log_prob[rows],
                                        response_mask=task_response_mask,
                                        loss_agg_mode=loss_agg_mode,
                                    ).detach().item()

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.config.get("use_sdar_loss", False):
                                    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                                    from verl.trainer.ppo.sdar_utils import compute_sdar_loss

                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    _, task_sdar_metrics = compute_sdar_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        response_mask=task_response_mask,
                                        gate_beta=self.config.get("sdar_gate_beta", 5.0),
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    task_metrics.update({f"{name}/{task}": value for name, value in task_sdar_metrics.items()})

                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if use_teacher_kl_loss:
                                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                    task_metrics[f"actor/teacher_kl_loss/{task}"] = (
                                        agg_loss(loss_mat=teacher_kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                append_to_dict(metrics, task_metrics)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    append_to_dict(metrics, data)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                grad_norm = self._optimizer_step()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = {"actor/grad_norm": grad_norm.detach().item()}
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                append_to_dict(metrics, data)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_optimizer.zero_grad()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return metrics
