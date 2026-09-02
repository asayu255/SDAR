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
from contextlib import contextmanager
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, agg_loss_with_sample_weights, compute_policy_loss, compute_policy_loss_gspo, kl_penalty, topk_kl_per_token
from verl.trainer.ppo.metric_utils import iter_task_row_masks
from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY
from verl.utils.debug import GPUMemoryLogger
from verl.utils import gpu_profiler
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

# With pg_loss_coef == 0 (pure distillation) the policy-gradient statistics are a
# single device-side zero tensor. Reading it back with .item() synchronises the
# stream, and update_policy would do so four times per micro-batch plus four more per
# task -- so name the constants instead.
# Worker-side stage profiling. Rank 0 only, and only when the profiler is on --
# see _actor_phase. Read once at import so the instrumentation cannot change
# halfway through a run.
_PROFILE_STAGES = False   # set in DataParallelPPOActor.__init__ once the rank is known
_SYNC_PHASES = os.environ.get("GPU_PROFILER_SYNC_PHASES", "0").strip().lower() in ("1", "true", "yes", "on")

_ZERO_PG_METRICS = {
    "actor/pg_loss": 0.0,
    "actor/pg_clipfrac": 0.0,
    "actor/ppo_kl": 0.0,
    "actor/pg_clipfrac_lower": 0.0,
}


def _ZERO_PG_METRICS_BY_TASK(task: str) -> dict:
    return {f"{name}/{task}": value for name, value in _ZERO_PG_METRICS.items()}


@contextmanager
def _actor_phase(name: str):
    """Tag GPU samples inside update_policy with the stage that produced them.

    The gpu_profiler's phase stack is driven by ray_trainer._timer, which runs in
    the DRIVER. update_policy runs in a worker, so from the driver's side the
    whole call is one opaque ``update_actor`` bucket. Pushing here starts a second
    sampler in the worker process and gives that bucket an interior: NVML reads
    are device-wide, so a worker-side sampler sees the same GPUs and the phase tag
    is what makes the samples attributable.

    Only rank 0 pushes. Three concurrent samplers would triple the NVML polling
    and print three interleaved reports for readings that are already device-wide
    and therefore identical.

    A caveat the numbers carry: kernel launches are asynchronous, so a phase's
    wall clock is when its work was *issued*, not when the GPU finished it. The
    boundaries smear by roughly one launch queue. GPU_PROFILER_SYNC_PHASES=1 adds
    a device synchronize at each boundary, which makes the split exact at the cost
    of serializing what the run would otherwise overlap -- read those numbers as
    an attribution, not as the run's real speed.

    A no-op unless GPU_PROFILER=1: push_phase/pop_phase return immediately.
    """
    if not _PROFILE_STAGES:
        yield
        return
    if _SYNC_PHASES:
        get_torch_device().synchronize()
    gpu_profiler.push_phase(name)
    try:
        yield
    finally:
        if _SYNC_PHASES:
            get_torch_device().synchronize()
        gpu_profiler.pop_phase(name)


def response_row_selection(indices: torch.Tensor, seqlen: int, response_length: int):
    """Which unpadded rows land in the response slice, and where.

    ``indices`` is what ``unpad_input`` returns: for each row of the packed
    (total_nnz, ...) batch, its position in the flattened (batch * seqlen) grid.
    The distillation path only ever reads ``[:, -response_length - 1 : -1]`` of
    the re-padded result, so every row outside that window is vocab-sized work
    thrown away -- see the caller in ``_forward_micro_batch``.

    The window is offset by one because a logit at position ``t`` predicts the
    token at ``t + 1``: predicting the ``response_length`` response tokens needs
    positions ``seqlen - response_length - 1 .. seqlen - 2``.

    Returns:
        sel: row numbers into the packed batch, ascending.
        sel_indices: ``indices[sel]``, i.e. their flattened grid positions, so
            ``pad_input`` can scatter results straight back.
        sel_slot: each selected row's column in a (bs, response_length, ...)
            tensor, for reading per-response-token inputs without first
            scattering them across the full sequence.
    """
    seq_pos = indices % seqlen
    lo = seqlen - response_length - 1
    sel = torch.nonzero((seq_pos >= lo) & (seq_pos < seqlen - 1), as_tuple=True)[0]
    return sel, indices[sel], seq_pos[sel] - lo


def _grad_sync_context(module, active: bool):
    """FSDP's no_sync() for one micro-batch, or None when it does not apply.

    Returned rather than yielded because it has to be entered and exited at
    micro-batch *boundaries*: FSDP1's no_sync() asserts the module is IDLE, so it
    cannot be wrapped around the backward alone -- by then the forward has moved
    the module into FORWARD_BACKWARD.

    Returns None when there is nothing to do, so the caller can skip the enter
    entirely on the single-micro-batch path.
    """
    if not active:
        return None
    if isinstance(module, FSDP):
        return module.no_sync()
    if isinstance(module, FSDPModule):
        return _fsdp2_no_sync(module)
    return None


@contextmanager
def _fsdp2_no_sync(module):
    """no_sync() for FSDP2, which spells it as a setter rather than a context."""
    module.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        module.set_requires_gradient_sync(True)


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        # Stage profiling on rank 0 only: NVML is device-wide, so one sampler
        # already sees every GPU and three would just triple the polling and
        # interleave three identical reports.
        global _PROFILE_STAGES
        try:
            _PROFILE_STAGES = gpu_profiler.enabled() and torch.distributed.get_rank() == 0
        except Exception:
            _PROFILE_STAGES = False

        # Accumulate gradients locally across a mini-batch's micro-batches and
        # reduce once, instead of reducing after every one. Off by default: it
        # changes the order the partial sums are reduced in, so gradients differ
        # in their last bits (the expectation is identical -- summing then
        # averaging across ranks and averaging then summing are the same number).
        self.no_sync_grad_accum = bool(self.config.get("no_sync_grad_accum", False))
        if self.no_sync_grad_accum:
            print("Actor no_sync_grad_accum=True (one gradient reduce per mini-batch)")

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

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, topk_k=None, topk_ids=None, return_lse=False) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
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

                With ``return_lse`` the teacher mode appends a third element: the
                packed-row map and the full-vocabulary normaliser, which is what
                lets a caller keep this forward's output usable at ids nobody has
                chosen yet (see verl/workers/teacher_cache.py). Only the last
                gather depends on the ids, so everything else here is already the
                whole answer.
        """
        topk_out = None
        if (topk_k is not None or topk_ids is not None) and self.use_fused_kernels:
            raise NotImplementedError("top-k KL forward is not supported with fused kernels")
        if return_lse and topk_ids is not None:
            # Student mode returns one tensor where teacher mode returns a pair,
            # and the return_lse packing below unpacks a pair. Nothing asks for
            # this combination -- only the teacher path caches -- so refuse it
            # rather than let it drop the student's values on the floor.
            raise NotImplementedError("return_lse is for the teacher path (topk_k), not for topk_ids")
        if return_lse and not self.use_remove_padding:
            # The hidden states come off a forward hook on the packed sequence, so
            # the row map that pairs them with a (bs, response_length) grid only
            # exists on the remove-padding path. Refuse rather than hand back a
            # normaliser the caller cannot align anything to.
            raise NotImplementedError(
                "return_lse needs use_remove_padding: the caller pairs the normaliser with the body's "
                "packed hidden states, and there is no row map without it"
            )
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
                    if calculate_entropy or topk_k is not None or topk_ids is not None or return_lse:
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
                if topk_k is not None or topk_ids is not None or return_lse:
                    if self.use_fused_kernels or self.use_ulysses_sp:
                        raise NotImplementedError("top-k KL forward is not supported with fused kernels or ulysses SP")
                    # Restrict the vocab-sized work to the rows that survive the slice
                    # at the end of this block. logits_rmpad covers prompt AND response,
                    # and on this mixture the mean prompt is ~3x the mean response, so
                    # roughly three quarters of the rows were being logsumexp'd, top-k'd
                    # and gathered only for pad_input to scatter them somewhere the
                    # slice drops. On the student side that waste sat inside the
                    # autograd graph, so the backward paid for it a second time.
                    #
                    # Values are unchanged. A row outside the window can only land
                    # outside the slice, and a window position missing from rmpad
                    # (attention_mask==0, i.e. response padding) is zero-filled by
                    # pad_input whether or not any row was scattered elsewhere.
                    #
                    # Cost: one (n_resp, vocab) copy of the selected logits. It is a
                    # quarter of logits_rmpad, which is live either way.
                    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
                    logits_resp = logits_rmpad[sel]  # (n_resp, vocab)
                    lse = torch.logsumexp(logits_resp, dim=-1, keepdim=True)  # (n_resp, 1)
                    if topk_k is not None:
                        tvals, tids = torch.topk(logits_resp, k=topk_k, dim=-1)  # (n_resp, k)
                        # Use float32 for pad_input: bf16 cannot represent vocab ids
                        # (>256) exactly, and float32 keeps log-probs precise.
                        t_lp_rmpad = (tvals - lse).float()
                        full_t_lp = pad_input(t_lp_rmpad, indices=sel_indices, batch=batch_size, seqlen=seqlen)
                        full_t_id = pad_input(tids.float(), indices=sel_indices, batch=batch_size, seqlen=seqlen)
                        topk_out = (
                            full_t_lp[:, -response_length - 1 : -1, :],
                            full_t_id[:, -response_length - 1 : -1, :].round().long(),
                        )
                    elif topk_ids is not None:
                        # Read topk_ids (bs, response_len, k) directly at each selected
                        # row's (sample, response-slot). The old path scattered it into a
                        # (bs, seqlen, k) zero tensor and gathered that back through
                        # `indices`, which is the same map computed the long way round.
                        ids_resp = topk_ids[sel_indices // seqlen, sel_slot]  # (n_resp, k)
                        s_lp_rmpad = (logits_resp.gather(-1, ids_resp) - lse).float()  # (n_resp, k), keeps grad
                        full_s_lp = pad_input(s_lp_rmpad, indices=sel_indices, batch=batch_size, seqlen=seqlen)
                        topk_out = full_s_lp[:, -response_length - 1 : -1, :]
                    if return_lse:
                        # The caller wants to evaluate this model at ids nobody has
                        # chosen yet, which needs the normaliser and the packed row
                        # map -- the projection itself it can redo for 2*H*k. Handed
                        # back PACKED, not scattered, because it is paired with the
                        # body's hidden states, which the forward hook also sees
                        # packed; `sel` indexes those and `sel_indices` scatters
                        # either of them back.
                        #
                        # With topk_k None the model's OWN top-k is not built at
                        # all, which is a full-vocabulary selection and two
                        # scatters saved on every micro-batch that is not the
                        # witness sample.
                        tlp, tids_out = topk_out if topk_k is not None else (None, None)
                        topk_out = (
                            tlp,
                            tids_out,
                            {
                                "lse": lse.float(),
                                "sel": sel,
                                "sel_indices": sel_indices,
                                "batch_size": batch_size,
                                "seqlen": seqlen,
                                "response_length": response_length,
                            },
                        )

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
    @contextmanager
    def _capture_last_hidden(self, sink: dict):
        """Grab the transformer body's output without asking for all 29 layers.

        ``output_hidden_states=True`` would return every layer -- on a teacher
        micro-batch that is gigabytes of tensors, 28/29 of them discarded. A
        forward hook on the base model takes only what the projection consumes.
        Registered on the unwrapped module so FSDP's own forward still runs
        normally around it; a model whose body cannot be located falls back to
        yielding nothing and the caller raises with a readable message.
        """
        inner = self.actor_module
        for _ in range(8):
            nxt = getattr(inner, "_fsdp_wrapped_module", None) or getattr(inner, "_orig_mod", None)
            if nxt is None or nxt is inner:
                break
            inner = nxt
        body = getattr(inner, "model", None)
        if body is None:
            yield
            return

        def _hook(_module, _inputs, output):
            sink["h"] = output[0] if isinstance(output, tuple) else output.last_hidden_state

        handle = body.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def _teacher_logprobs_at(self, cache_ids, ids, input_ids=None, attention_mask=None):
        """Teacher log-probs at ids the student just chose.

        The teacher's hidden states were cached wherever its forward ran, which is
        not where this micro-batch is being trained: between the two calls the rows
        are padded by ``adjust_batch`` and reordered by ``_balance_batch`` to
        equalise tokens per rank. ``exchange_teacher_logprobs`` therefore asks every
        rank and sums the one answer that exists, raising if a row is unanswered or
        answered twice. See verl/workers/teacher_cache.py.

        Both sides work per ROW: one key locates the row's whole
        (response_length, hidden) block, and ``ids`` stays (bs, response_length, k).
        """
        from verl.workers.teacher_cache import exchange_teacher_logprobs, get_teacher_cache, row_fingerprint

        if cache_ids is None:
            raise ValueError(
                "student_indexed_topk needs a `teacher_cache_ids` column locating each row's cached "
                "teacher hidden states; the batch has none."
            )
        # Derived from the rows being trained RIGHT HERE, so a key column shifted
        # against its batch is caught. The key alone would resolve cleanly and
        # return a real teacher log-prob for somebody else's sample.
        fingerprints = None
        if input_ids is not None and attention_mask is not None:
            fingerprints = row_fingerprint(input_ids, attention_mask)
        return exchange_teacher_logprobs(get_teacher_cache(), cache_ids, ids, fingerprints=fingerprints)

    def compute_topk_log_prob(
        self, data: DataProto, topk_k: int, return_hidden: bool = False, witness_micro_batches=None
    ):
        """Teacher-side: per response token, the teacher's top-k token ids and the
        teacher's full-vocab log-softmax values at those ids.

        Args:
            return_hidden: also return the body's hidden states and the
                full-vocabulary logsumexp at the scored positions, so the caller
                can evaluate this teacher at ids chosen later (see
                verl/workers/teacher_cache.py). The expensive work is unchanged --
                only the last gather depends on the ids.
            witness_micro_batches: build the teacher's OWN top-k for that many
                leading micro-batches only, and return it as a sample instead of a
                full column. In this mode nothing downstream reads it: the support
                comes from the student, and the teacher's top-k exists only to
                check the cache against the forward that filled it. Building it is
                a selection over the whole vocabulary and a scatter back to
                (bs, response_length, k), so running it on every row is the
                expensive way to do a spot check. Requires ``return_hidden``.

        Returns:
            witness_micro_batches is None:
                topk_logprob: (bs, response_length, k)
                topk_ids:     (bs, response_length, k) int64
                hidden, lse:  (bs, response_length, hidden) / (bs, response_length),
                              only when ``return_hidden``
            otherwise:
                hidden, lse, witness_rows, witness_ids, witness_logprob -- where
                ``witness_rows`` indexes the ORIGINAL row order (so it survives the
                dynamic-batching reorder) and the other two are
                (len(witness_rows), response_length, k).
        """
        self.actor_module.eval()
        sampled_witness = witness_micro_batches is not None
        if sampled_witness and not return_hidden:
            raise ValueError("witness_micro_batches only makes sense with return_hidden=True")

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
        hidden_lst = []
        lse_lst = []
        witness_rows_lst = []
        row_cursor = 0
        for mb_i, micro_batch in enumerate(micro_batches):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            n_rows = micro_batch["responses"].size(0)
            # In sampled-witness mode the teacher's own top-k is a spot check, not
            # a column: a full-vocabulary selection plus two scatters per row, for
            # values nothing downstream reads once the support is the student's.
            want_topk = (not sampled_witness) or mb_i < witness_micro_batches
            sink = {}
            with torch.no_grad(), self._capture_last_hidden(sink if return_hidden else {}):
                _, _, topk_out = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=False,
                    topk_k=topk_k if want_topk else None,
                    return_lse=return_hidden,
                )
            if return_hidden:
                tlp, tids, extras = topk_out
                if "h" not in sink:
                    raise RuntimeError(
                        "student_indexed_topk needs the teacher's hidden states, but the forward hook did "
                        "not fire -- the model's transformer body could not be located under its wrappers."
                    )
                # Select the same packed rows the projection used, then scatter both
                # back into (bs, response_length, ·) so they line up row-for-row with
                # the top-k the caller also gets.
                h_resp = sink["h"].squeeze(0)[extras["sel"]]  # (n_resp, hidden)
                bs_, sl_, rl_ = extras["batch_size"], extras["seqlen"], extras["response_length"]
                hidden_lst.append(
                    pad_input(h_resp, indices=extras["sel_indices"], batch=bs_, seqlen=sl_)[:, -rl_ - 1 : -1, :]
                )
                lse_lst.append(
                    pad_input(extras["lse"], indices=extras["sel_indices"], batch=bs_, seqlen=sl_)[:, -rl_ - 1 : -1, 0]
                )
            else:
                tlp, tids = topk_out
            if want_topk:
                topk_logprob_lst.append(tlp)
                topk_ids_lst.append(tids)
                if sampled_witness:
                    # Original row numbers: rearrange_micro_batches hands back, per
                    # micro-batch, where each of its rows came from, so the sample
                    # needs no revert of its own.
                    witness_rows_lst.append(
                        torch.as_tensor(indices[mb_i], dtype=torch.long)
                        if use_dynamic_bsz
                        else torch.arange(row_cursor, row_cursor + n_rows, dtype=torch.long)
                    )
            row_cursor += n_rows

        hidden = torch.concat(hidden_lst, dim=0) if return_hidden else None
        lse = torch.concat(lse_lst, dim=0) if return_hidden else None
        if not sampled_witness:
            topk_logprob = torch.concat(topk_logprob_lst, dim=0)
            topk_ids = torch.concat(topk_ids_lst, dim=0)

        if use_dynamic_bsz:
            flat = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(flat), dtype=torch.long)
            if not sampled_witness:
                assert len(flat) == topk_logprob.size(0), f"{len(flat)} vs. {topk_logprob.size()}"
                topk_logprob = topk_logprob[revert_indices]
                topk_ids = topk_ids[revert_indices]
            if return_hidden:
                # The hidden states are per row like the top-k, so they follow the
                # same reordering; skipping this would file every entry under a
                # neighbour's key, which is exactly what the witness catches.
                assert len(flat) == hidden.size(0), f"{len(flat)} vs. {hidden.size()}"
                hidden = hidden[revert_indices]
                lse = lse[revert_indices]

        if sampled_witness:
            if witness_rows_lst:
                w_rows = torch.cat(witness_rows_lst)
                w_ids = torch.concat(topk_ids_lst, dim=0)
                w_lp = torch.concat(topk_logprob_lst, dim=0)
            else:
                w_rows = torch.empty(0, dtype=torch.long)
                w_ids = w_lp = None
            return hidden, lse, w_rows, w_ids, w_lp
        if return_hidden:
            return topk_logprob, topk_ids, hidden, lse
        return topk_logprob, topk_ids

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # A call that died between entering and leaving no_sync would leave the
        # flag off, and every later step would then silently skip its gradient
        # reduce. Cheap to make each call start from a known state.
        if self.no_sync_grad_accum and isinstance(self.actor_module, FSDP):
            for m in self.actor_module.modules():
                if isinstance(m, FSDP):
                    m._sync_gradients = True

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
        use_teacher_kl_loss = self.config.get("use_teacher_kl_loss", False)
        teacher_kl_loss_type = self.config.get("teacher_kl_loss_type", "low_var_kl")
        teacher_topk_kl = use_teacher_kl_loss and teacher_kl_loss_type == "topk_kl"
        # Whose top-k defines the KL's support. Teacher-indexed (the default) reads
        # the ids the teacher already chose off the batch. Student-indexed asks
        # this forward for the student's OWN top-k and resolves the teacher at
        # those ids afterwards, out of the hidden states the teacher left behind --
        # see verl/workers/teacher_cache.py for why that does not force the teacher
        # to run second. Both are lower bounds on the same full reverse KL; the
        # student-indexed one is the tighter of the two, because the mass the bound
        # drops is weighted by the STUDENT's tail.
        student_indexed_topk = teacher_topk_kl and bool(self.config.get("student_indexed_topk", False))
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
            if student_indexed_topk:
                # The support is chosen here, so the teacher's own top-k is not an
                # input any more -- only the key locating its cached hidden states.
                # The pool still carries the top-k columns (they are what a
                # teacher-indexed control arm trains on, and what the Stage-2
                # teacher can be checked against), they are just not read here.
                select_keys.append("teacher_cache_ids")
            else:
                select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]
        # Multitask runs tag every row with its task id (see RayPPOTrainer._attach_task_ids)
        # so the loss metrics below can also be reported per task. Absent in single-task runs.
        task_id_names = data.meta_info.get("task_id_names", None)
        if "task_ids" in data.batch.keys():
            select_keys.append("task_ids")
        # Per-task normalised distillation loss: the driver attaches a per-row weight
        # (see verl/trainer/ppo/task_loss_weights.py) that makes each task's share of
        # the loss 1/num_tasks instead of its share of the response tokens. Absent ->
        # plain token-mean, i.e. nothing below changes. Both distillation terms (the
        # top-k KL and the hard-label CE) are weighted by it, so their sum stays the
        # objective the run declares.
        task_weighted = TASK_LOSS_WEIGHT_KEY in data.batch.keys()
        if task_weighted:
            select_keys.append(TASK_LOSS_WEIGHT_KEY)
            # The weights encode the whole normalisation, which these three would
            # each re-scale out from under it.
            assert not self.config.use_dynamic_bsz, (
                "per-task loss normalisation is incompatible with use_dynamic_bsz "
                "(its mini-batch scaling assumes an unweighted token-mean)"
            )
            assert self.ulysses_sequence_parallel_size == 1, (
                "per-task loss normalisation assumes DP size == world size; "
                f"got ulysses_sequence_parallel_size={self.ulysses_sequence_parallel_size}"
            )
            assert self.config.ppo_epochs == 1, (
                "per-task loss normalisation assumes one pass over each mini-batch; "
                f"got ppo_epochs={self.config.ppo_epochs}"
            )
            # Any term that is not weighted the same way would be added to a
            # differently-normalised number.
            assert use_teacher_kl_loss or self.config.get("use_sft_loss", False), (
                "per-task loss normalisation has nothing to normalise: neither "
                "use_teacher_kl_loss nor use_sft_loss is set"
            )
            for other in ("use_kl_loss", "use_sdl_loss", "use_sdar_loss"):
                assert not self.config.get(other, False), (
                    f"per-task loss normalisation requires the distillation terms to be "
                    f"the only loss terms, but {other} is set"
                )
            assert pg_loss_coef == 0 and self.config.entropy_coeff == 0, (
                "per-task loss normalisation requires the distillation terms to be the "
                f"only loss terms, but pg_loss_coef={pg_loss_coef} "
                f"entropy_coeff={self.config.entropy_coeff}"
            )
            self.task_dp_world_size = (
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
        # Scalar metrics that are only read by the logger are kept as 0-d GPU
        # tensors until the end of the call. Reading them per micro-batch
        # (.item()) blocks the host until the device drains -- with three tasks
        # that was hundreds of forced syncs per step. Deferring changes nothing
        # about the values: the single read at the end takes the same mean over
        # micro-batches that append_to_dict + reduce_metrics would have.
        deferred_metrics = {}

        def _defer(name, value):
            deferred_metrics.setdefault(name, []).append(value.detach())

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

                # One reduce per mini-batch instead of one per micro-batch. The
                # context is entered before the first micro-batch's forward and
                # left before the last one's, both points where FSDP is IDLE --
                # no_sync() asserts that, so it cannot wrap the backward alone.
                # The last micro-batch then runs with sync on and reduces
                # everything accumulated.
                #
                # Under SHARD_GRAD_OP this removes the all-gather as well, not
                # just the reduce-scatter: _should_free_in_backward returns
                # `state._sync_gradients or strategy in
                # RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES`, and SHARD_GRAD_OP is
                # not in that set -- so with sync off the parameters are left
                # unsharded and the next micro-batch does not re-gather them.
                # That is why this pairs with sharding_strategy=shard_grad_op.
                n_micro = len(micro_batches)
                accum_ctx = None
                for micro_idx, data in enumerate(micro_batches):
                    if self.no_sync_grad_accum:
                        if micro_idx == 0 and n_micro > 1:
                            accum_ctx = _grad_sync_context(self.actor_module, True)
                            if accum_ctx is not None:
                                accum_ctx.__enter__()
                        elif micro_idx == n_micro - 1 and accum_ctx is not None:
                            accum_ctx.__exit__(None, None, None)
                            accum_ctx = None
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    task_ids = data.get("task_ids", None) if task_id_names else None
                    task_loss_weight = data[TASK_LOSS_WEIGHT_KEY] if task_weighted else None
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    def _task_weighted(loss_mat):
                        """The per-task normalised counterpart of ``agg_loss``.

                        The driver put a weight on every row such that summing
                        weight * row-loss over the whole step gives each task an
                        equal share (see attach_task_loss_weights). The two
                        divisions that sum must survive are undone here rather
                        than by special-casing the shared scaling below: FSDP
                        averages gradients across the DP ranks, and the mini-batch
                        loss is divided by gradient_accumulation, but the weights
                        already carry the full normalisation.
                        """
                        row_loss = (loss_mat * response_mask).sum(-1)
                        weighted = (row_loss * task_loss_weight).sum()
                        return weighted * (self.task_dp_world_size * self.gradient_accumulation)

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
                    # Teacher-indexed hands the forward the ids the teacher already
                    # chose; student-indexed asks it for the student's own top-k
                    # and resolves the teacher at those ids below.
                    fwd_topk_ids = fwd_topk_k = None
                    if teacher_topk_kl:
                        if student_indexed_topk:
                            fwd_topk_k = int(self.config.get("teacher_kl_topk", 20))
                        else:
                            fwd_topk_ids = data["teacher_topk_ids"]
                    with _actor_phase("actor.fwd"):
                        entropy, log_prob, student_topk_out = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, topk_ids=fwd_topk_ids, topk_k=fwd_topk_k)
                    if student_indexed_topk:
                        # One logits tensor produced both: the values (which carry
                        # the gradient -- topk's backward is the same scatter a
                        # gather at those ids would give) and the ids that chose
                        # them. There is no second student forward here.
                        student_topk_logprobs, student_topk_ids = student_topk_out
                        with _actor_phase("actor.teacher_lookup"):
                            fwd_teacher_topk_logprobs = self._teacher_logprobs_at(
                                cache_ids=data.get("teacher_cache_ids", None),
                                ids=student_topk_ids,
                                input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"],
                            )
                    else:
                        student_topk_logprobs = student_topk_out
                        fwd_teacher_topk_logprobs = None
                    
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
                                teacher_topk_logprob=(
                                    fwd_teacher_topk_logprobs
                                    if student_indexed_topk
                                    else data["teacher_topk_logprobs"]
                                ),
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
                        if task_loss_weight is None:
                            policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef
                        else:
                            weighted_teacher_kl = _task_weighted(teacher_kld)
                            policy_loss = policy_loss + weighted_teacher_kl * teacher_kl_coef
                            # Appended, not assigned: assignment kept only the LAST
                            # micro-batch, which after _balance_batch's reorder is often
                            # entirely adjust_batch padding -- weight 0, so the logged
                            # value sat at 0.000 while the real loss was fine.
                            _defer("actor/teacher_kl_loss_weighted", weighted_teacher_kl)
                        # Deferred, and appended rather than assigned: assignment kept
                        # only the LAST micro-batch, which after _balance_batch's
                        # reorder is often entirely adjust_batch padding.
                        #
                        # Kept unweighted so it stays comparable with runs that do not
                        # normalise per task -- and so the KL/CE ratio below is read on
                        # one consistent scale.
                        _defer("actor/teacher_kl_loss", teacher_kl_loss)
                        metrics["actor/teacher_kl_coef"] = teacher_kl_coef

                    if self.config.get("use_sft_loss", False):
                        # Hard-label cross-entropy on the teacher's own sampled tokens:
                        # the degenerate (one-hot teacher) limit of the distillation
                        # term above. Added to it rather than replacing it, so the
                        # signal is kl_coef * KL(teacher_topk || student) + sft_coef * CE.
                        #
                        # Both terms are aggregated the same way over the same mask, so
                        # the sum is well defined -- but they are not on the same scale
                        # and do not stay in the same ratio. The KL goes to 0 as the
                        # student matches the teacher; the CE cannot, because it is
                        # bounded below by the teacher's own entropy on those tokens. A
                        # fixed sft_loss_coef therefore weights the CE more and more
                        # heavily as training converges, which is a property of the
                        # objective, not a bug -- see actor/sft_loss vs
                        # actor/teacher_kl_loss in wandb to see the ratio move.
                        sft_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        sft_coef = self.config.get("sft_loss_coef", 1.0)
                        if task_loss_weight is None:
                            policy_loss = policy_loss + sft_loss * sft_coef
                        else:
                            # Same weights as the KL term above, so normalising per task
                            # does not disturb the kl_coef : sft_coef ratio the run pins.
                            weighted_sft = _task_weighted(-log_prob)
                            policy_loss = policy_loss + weighted_sft * sft_coef
                            _defer("actor/sft_loss_weighted", weighted_sft)
                        # Unweighted, for the same reason as actor/teacher_kl_loss.
                        _defer("actor/sft_loss", sft_loss)
                        metrics["actor/sft_coef"] = sft_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    with _actor_phase("actor.bwd"):
                        loss.backward()

                    if task_ids is not None:
                        # Same losses, re-aggregated over the rows of one task at a
                        # time. Diagnostics only: nothing here touches the graph the
                        # optimizer step above was built from, and the results are
                        # deferred GPU tensors -- no host sync happens here. Timed
                        # separately anyway: this phase is where the backward's
                        # queued reduce-scatter tail drains.
                        with _actor_phase("actor.task_metrics"), torch.no_grad():
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
                                    _defer(
                                        f"actor/teacher_kl_loss/{task}",
                                        agg_loss(loss_mat=teacher_kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode),
                                    )

                                if self.config.get("use_sft_loss", False):
                                    _defer(
                                        f"actor/sft_loss/{task}",
                                        agg_loss(loss_mat=-log_prob[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode),
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

                if student_indexed_topk:
                    # Every row the exchange was asked about must have been
                    # answered by exactly one rank, and by the row it was asked
                    # for. Checked here rather than inside the exchange because
                    # reading the tally synchronises and the exchange runs once
                    # per micro-batch -- and this is still the last point before
                    # the weights move, so a row that went unresolved cannot
                    # reach them.
                    from verl.workers.teacher_cache import assert_rows_were_owned_once

                    assert_rows_were_owned_once()

                with _actor_phase("actor.optim"):
                    grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        # The one read for everything deferred above. torch.stack forces a single
        # host sync here instead of one per micro-batch.
        for name, values in deferred_metrics.items():
            metrics[name] = torch.stack(values).mean().item()
        if _PROFILE_STAGES:
            # One table per update_policy call. The driver's boundary phase
            # ("step") never pops in this process, so the report is asked for
            # explicitly rather than falling out of the phase stack.
            gpu_profiler.report_and_reset(label="update_policy stages (rank 0)")
        return metrics
