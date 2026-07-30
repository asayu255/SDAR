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
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, agg_loss_with_sample_weights, compute_policy_loss, compute_policy_loss_gspo, kl_penalty, topk_kl_per_token
from verl.trainer.ppo.metric_utils import iter_task_row_masks
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

# With pg_loss_coef == 0 (pure distillation / SFT) the policy-gradient statistics are
# a single device-side zero tensor. Reading it back with .item() synchronises the
# stream, and update_policy would do so four times per micro-batch plus four more per
# task -- so name the constants instead.
_ZERO_PG_METRICS = {
    "actor/pg_loss": 0.0,
    "actor/pg_clipfrac": 0.0,
    "actor/ppo_kl": 0.0,
    "actor/pg_clipfrac_lower": 0.0,
}


def _ZERO_PG_METRICS_BY_TASK(task: str) -> dict:
    return {f"{name}/{task}": value for name, value in _ZERO_PG_METRICS.items()}

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, topk_k=None, topk_ids=None) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
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
        topk_out = None
        if (topk_k is not None or topk_ids is not None) and self.use_fused_kernels:
            raise NotImplementedError("top-k KL forward is not supported with fused kernels")
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or topk_k is not None or topk_ids is not None:
                        # top-k KL reads logits_rmpad after this call, so don't mutate it.
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # top-k distillation log-probs (teacher: own top-k; student: gather at given ids)
                if topk_k is not None or topk_ids is not None:
                    if self.use_fused_kernels or self.use_ulysses_sp:
                        raise NotImplementedError("top-k KL forward is not supported with fused kernels or ulysses SP")
                    lse = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)  # (total_nnz, 1)
                    if topk_k is not None:
                        tvals, tids = torch.topk(logits_rmpad, k=topk_k, dim=-1)  # (total_nnz, k)
                        # Use float32 for pad_input: bf16 cannot represent vocab ids
                        # (>256) exactly, and float32 keeps log-probs precise.
                        t_lp_rmpad = (tvals - lse).float()
                        full_t_lp = pad_input(t_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        full_t_id = pad_input(tids.float(), indices=indices, batch=batch_size, seqlen=seqlen)
                        topk_out = (
                            full_t_lp[:, -response_length - 1 : -1, :],
                            full_t_id[:, -response_length - 1 : -1, :].round().long(),
                        )
                    else:
                        k = topk_ids.size(-1)
                        full_ids = torch.zeros((batch_size, seqlen, k), dtype=torch.long, device=logits_rmpad.device)
                        full_ids[:, -response_length - 1 : -1, :] = topk_ids
                        ids_rmpad = index_first_axis(rearrange(full_ids, "b s k -> (b s) k"), indices)  # (total_nnz, k)
                        s_lp_rmpad = (logits_rmpad.gather(-1, ids_rmpad) - lse).float()  # (total_nnz, k), keeps grad
                        full_s_lp = pad_input(s_lp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                        topk_out = full_s_lp[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                    if topk_k is not None or topk_ids is not None:
                        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (bsz, response_length, 1)
                        if topk_k is not None:
                            tvals, tids = torch.topk(logits, k=topk_k, dim=-1)
                            topk_out = ((tvals - lse).float(), tids.long())
                        else:
                            topk_out = (logits.gather(-1, topk_ids) - lse).float()

            return entropy, log_probs, topk_out

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
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
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_topk_log_prob(self, data: DataProto, topk_k: int):
        """Teacher-side: per response token, the teacher's top-k token ids and the
        teacher's full-vocab log-softmax values at those ids.

        Returns:
            topk_logprob: (bs, response_length, k)
            topk_ids:     (bs, response_length, k) int64
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        assert not has_multi_modal_inputs, "top-k KL is not supported for multi-modal inputs"

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        topk_logprob_lst = []
        topk_ids_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, _, topk_out = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=False, topk_k=topk_k)
            tlp, tids = topk_out
            topk_logprob_lst.append(tlp)
            topk_ids_lst.append(tids)

        topk_logprob = torch.concat(topk_logprob_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == topk_logprob.size(0), f"{len(indices)} vs. {topk_logprob.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            topk_logprob = topk_logprob[revert_indices]
            topk_ids = topk_ids[revert_indices]

        return topk_logprob, topk_ids

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
        use_teacher_kl_loss = self.config.get("use_teacher_kl_loss", False)
        teacher_kl_loss_type = self.config.get("teacher_kl_loss_type", "low_var_kl")
        teacher_topk_kl = use_teacher_kl_loss and teacher_kl_loss_type == "topk_kl"
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # advantages / old_log_probs are only needed by the policy-gradient (and SDL) paths.
        # Pure teacher-KL distillation (pg_loss_coef==0) does not produce them, so don't require them.
        if pg_loss_coef != 0:
            select_keys += ["old_log_probs", "advantages"]
        elif self.config.get("use_sdl_loss", False):
            select_keys.append("old_log_probs")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
            if "kl_loss_coef" in data.batch:
                select_keys.append("kl_loss_coef")
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False) or (use_teacher_kl_loss and not teacher_topk_kl):
            select_keys.append("teacher_log_probs")
        if teacher_topk_kl:
            select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]
        # Multitask runs tag every row with its task id (see RayPPOTrainer._attach_task_ids)
        # so the loss metrics below can also be reported per task. Absent in single-task runs.
        task_id_names = data.meta_info.get("task_id_names", None)
        if "task_ids" in data.batch.keys():
            select_keys.append("task_ids")
        # Per-task normalised SFT loss: the driver attaches a per-row weight (see
        # OffPolicyOPDRayTrainer._attach_sft_loss_weights). Absent -> plain token-mean.
        weighted_sft = self.config.get("use_sft_loss", False) and "sft_loss_weight" in data.batch.keys()
        if weighted_sft:
            select_keys.append("sft_loss_weight")
            # The weights encode the whole normalisation, which these three would
            # each re-scale out from under it.
            assert not self.config.use_dynamic_bsz, (
                "per-task normalised SFT loss is incompatible with use_dynamic_bsz "
                "(its mini-batch scaling assumes an unweighted token-mean)"
            )
            assert self.ulysses_sequence_parallel_size == 1, (
                "per-task normalised SFT loss assumes DP size == world size; "
                f"got ulysses_sequence_parallel_size={self.ulysses_sequence_parallel_size}"
            )
            assert self.config.ppo_epochs == 1, (
                "per-task normalised SFT loss assumes one pass over each mini-batch; "
                f"got ppo_epochs={self.config.ppo_epochs}"
            )
            # Any other loss term would be added to a differently-normalised number.
            for other in ("use_kl_loss", "use_sdl_loss", "use_sdar_loss", "use_teacher_kl_loss"):
                assert not self.config.get(other, False), (
                    f"per-task normalised SFT loss must be the only loss term, but {other} is set"
                )
            assert pg_loss_coef == 0 and self.config.entropy_coeff == 0, (
                "per-task normalised SFT loss must be the only loss term, but "
                f"pg_loss_coef={pg_loss_coef} entropy_coeff={self.config.entropy_coeff}"
            )
            self.sft_dp_world_size = (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    task_ids = data.get("task_ids", None) if task_id_names else None
                    sft_loss_weight = data["sft_loss_weight"] if weighted_sft else None
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    fwd_topk_ids = data["teacher_topk_ids"] if teacher_topk_kl else None
                    entropy, log_prob, student_topk_logprobs = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, topk_ids=fwd_topk_ids)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    if pg_loss_coef != 0:
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]
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
                    else:
                        # Pure teacher-KL distillation: no policy-gradient signal.
                        old_log_prob = data.get("old_log_probs", None)
                        zero = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
                        pg_loss = pg_clipfrac = ppo_kl = pg_clipfrac_lower = zero

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        kl_loss_coef = data.get("kl_loss_coef", None)
                        if kl_loss_coef is None:
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        else:
                            weighted_kl_loss = agg_loss_with_sample_weights(
                                loss_mat=kld,
                                loss_mask=response_mask,
                                sample_weights=kl_loss_coef,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + weighted_kl_loss
                            metrics["actor/kl_coef"] = kl_loss_coef.float().mean().detach().item()
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if use_teacher_kl_loss:
                        # On-policy distillation: KL between student and a (per-task) teacher,
                        # evaluated on the student's own on-policy responses. Only the student
                        # log-probs carry gradients; teacher values are detached upstream.
                        if teacher_topk_kl:
                            # Dense reverse KL over the teacher's top-k support (+ tail bucket).
                            teacher_kld = topk_kl_per_token(
                                student_topk_logprob=student_topk_logprobs,
                                teacher_topk_logprob=data["teacher_topk_logprobs"],
                            )
                        else:
                            # Single-sampled-token estimator (low_var_kl / kl / mse / abs).
                            teacher_kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=data["teacher_log_probs"],
                                kl_penalty=teacher_kl_loss_type,
                            )
                        teacher_kl_loss = agg_loss(loss_mat=teacher_kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        teacher_kl_coef = self.config.get("teacher_kl_loss_coef", 1.0)
                        policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef
                        metrics["actor/teacher_kl_loss"] = teacher_kl_loss.detach().item()
                        metrics["actor/teacher_kl_coef"] = teacher_kl_coef

                    if self.config.get("use_sft_loss", False):
                        # Off-policy SFT / behaviour cloning: maximize the student's
                        # log-prob of the (teacher-generated) response tokens, i.e.
                        # cross-entropy / NLL on hard targets. This is the degenerate
                        # (one-hot teacher) limit of the teacher-KL distillation.
                        sft_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        sft_coef = self.config.get("sft_loss_coef", 1.0)
                        if sft_loss_weight is None:
                            policy_loss = policy_loss + sft_loss * sft_coef
                        else:
                            # Per-task normalised variant: the driver put a weight on
                            # every row such that summing weight * row-NLL over the
                            # whole step gives each task an equal share of the loss
                            # (see _attach_sft_loss_weights). The two divisions this
                            # sum must survive are undone here rather than by
                            # special-casing the shared scaling below: FSDP averages
                            # gradients across the DP ranks, and the mini-batch loss
                            # is divided by gradient_accumulation, but the weights
                            # already carry the full normalisation.
                            row_nll = (-log_prob * response_mask).sum(-1)
                            weighted = (row_nll * sft_loss_weight).sum()
                            weighted = weighted * (self.sft_dp_world_size * self.gradient_accumulation)
                            policy_loss = policy_loss + weighted * sft_coef
                            metrics["actor/sft_loss_weighted"] = weighted.detach().item()
                        # Kept unweighted so it stays comparable with runs that do not
                        # normalise per task.
                        metrics["actor/sft_loss"] = sft_loss.detach().item()
                        metrics["actor/sft_coef"] = sft_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    if task_ids is not None:
                        # Same losses, re-aggregated over the rows of one task at a
                        # time. Diagnostics only: nothing here touches the graph the
                        # optimizer step above was built from.
                        with torch.no_grad():
                            for task, rows in iter_task_row_masks(task_ids, task_id_names):
                                task_response_mask = response_mask[rows]
                                task_metrics = {}

                                if pg_loss_coef != 0:
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
                                    task_metrics[f"actor/pg_loss/{task}"] = task_pg_loss.detach().item()
                                    task_metrics[f"actor/pg_clipfrac/{task}"] = task_pg_clipfrac.detach().item()
                                    task_metrics[f"actor/ppo_kl/{task}"] = task_ppo_kl.detach().item()
                                    task_metrics[f"actor/pg_clipfrac_lower/{task}"] = task_pg_clipfrac_lower.detach().item()
                                else:
                                    # Reading four device-side constants back per task
                                    # per micro-batch costs a stream sync each; the
                                    # values are known.
                                    task_metrics.update(_ZERO_PG_METRICS_BY_TASK(task))

                                if entropy_coeff != 0:
                                    task_metrics[f"actor/entropy_loss/{task}"] = (
                                        agg_loss(loss_mat=entropy[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                if self.config.use_kl_loss:
                                    task_metrics[f"actor/kl_loss/{task}"] = (
                                        agg_loss(loss_mat=kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )
                                    if kl_loss_coef is not None:
                                        task_metrics[f"actor/kl_coef/{task}"] = kl_loss_coef[rows].float().mean().detach().item()

                                if self.config.get("use_sdl_loss", False):
                                    from verl.trainer.ppo.skillsd_utils import compute_sdl_loss

                                    task_metrics[f"actor/sdl_loss/{task}"] = compute_sdl_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        old_log_probs=old_log_prob[rows],
                                        response_mask=task_response_mask,
                                        loss_agg_mode=loss_agg_mode,
                                    ).detach().item()

                                if self.config.get("use_sdar_loss", False):
                                    from verl.trainer.ppo.sdar_utils import compute_sdar_loss

                                    _, task_sdar_metrics = compute_sdar_loss(
                                        student_log_probs=log_prob[rows],
                                        teacher_log_probs=teacher_log_probs[rows],
                                        response_mask=task_response_mask,
                                        gate_beta=self.config.get("sdar_gate_beta", 5.0),
                                        loss_agg_mode=loss_agg_mode,
                                    )
                                    task_metrics.update({f"{name}/{task}": value for name, value in task_sdar_metrics.items()})

                                if use_teacher_kl_loss:
                                    task_metrics[f"actor/teacher_kl_loss/{task}"] = (
                                        agg_loss(loss_mat=teacher_kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                if self.config.get("use_sft_loss", False):
                                    task_metrics[f"actor/sft_loss/{task}"] = (
                                        agg_loss(loss_mat=-log_prob[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode).detach().item()
                                    )

                                append_to_dict(metrics, task_metrics)

                    if pg_loss_coef != 0:
                        data = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    else:
                        data = dict(_ZERO_PG_METRICS)
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
