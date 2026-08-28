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
import functools
import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, agg_loss_by_task_weights, agg_loss_with_sample_weights, compute_policy_loss, compute_policy_loss_gspo, compute_policy_loss_per_token, kl_penalty, policy_loss_gradient_coef, topk_kl_per_token
from verl.trainer.ppo.metric_utils import iter_task_row_masks
from verl.trainer.ppo.sign_weights import (
    REWRITE_TERMS,
    OffTaskLadderStats,
    ScopeTermStats,
    PairEventSamples,
    SignEventSamples,
    SignPairTokens,
    SignPairCounts,
    SignWeightStats,
    TokenStateCounts,
    rewrite_decomposition_terms,
    rewrite_ratio_metrics,
    candidate_effect,
    candidate_weights,
    normalize_per_task,
    position_weights,
    POSITION_TERMS,
    position_decomposition_terms,
    position_ratio_metrics,
    ROLE_NAMES,
    RoleTokenCounts,
    token_roles,
    turn_index,
    reweight_teacher_logprobs,
)
from verl.trainer.ppo.cross_teacher_kl_weight import (
    POSITION_TERMS as XT_POSITION_TERMS,
    CHANNEL_PROBES as XT_CHANNEL_PROBES,
    PROBE_ALPHAS as XT_PROBE_ALPHAS,
    STATE_TERMS as XT_STATE_TERMS,
    AdvantageReliabilityStats,
    CumulativePolicyShiftRMS,
    LogitPushTokens,
    OutcomeEffectStats,
    PairEvidenceStats,
    PairStateEvidenceStats,
    PositionScopeTermStats,
    PreviousStepTaskKLWeightedMean,
    SourceOutcomeStats,
    WeightShiftHistogram,
    opd_logit_push,
    pair_state_index,
    ROLE_CUT_SUFFIXES as XT_ROLE_CUT_SUFFIXES,
    ROLE_SCOPE_NAMES as XT_ROLE_SCOPE_NAMES,
    TURN_CUT_SUFFIXES as XT_TURN_CUT_SUFFIXES,
    select_metrics as xt_select_metrics,
    TURN_BUCKETS as XT_TURN_BUCKETS,
    TURN_SCOPE_NAMES as XT_TURN_SCOPE_NAMES,
    build_position_weight,
    compute_raw_policy_shifts,
    decompose_common_residual,
    load_sidecar_state,
    GRAD_TERMS as XT_GRAD_TERMS,
    assert_all_finite,
    gradient_metrics,
    logit_gradient_terms,
    per_candidate_shift,
    position_terms as xt_position_terms,
    position_weight_metrics,
    probe_name,
    residual_support_score,
    standardize_policy_shifts,
    state_shift_metrics,
    state_shift_terms as xt_state_shift_terms,
)
from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY
from verl.utils.debug import GPUMemoryLogger
from verl.utils import actor_capture, gpu_profiler
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

# Worker-side stage profiling. Rank 0 only, and only when the profiler is on --
# see _actor_phase. Read once at import so the instrumentation cannot change
# halfway through a run.
_PROFILE_STAGES = False   # set in DataParallelPPOActor.__init__ once the rank is known
_SYNC_PHASES = os.environ.get("GPU_PROFILER_SYNC_PHASES", "0").strip().lower() in ("1", "true", "yes", "on")

# With pg_loss_coef == 0 (pure distillation) the policy-gradient statistics are a
# single device-side zero tensor. Reading it back with .item() synchronises the
# stream, and update_policy would do so four times per micro-batch plus four more
# per task -- so name the constants instead.
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
    # The NVTX range is separate from the sampler and is pushed on EVERY rank:
    # the question a kernel trace answers is which rank the other two are
    # waiting for, and a timeline labelled on rank 0 alone cannot answer it.
    # Both are no-ops unless their own env var is set.
    with actor_capture.phase(name):
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


_VARLEN_KWARGS = os.environ.get("ACTOR_PASS_CU_SEQLENS", "1").strip().lower() in (
    "1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def _flash_attention_takes_varlen_kwargs() -> bool:
    """Does this transformers accept cu_seqlens at the flash-attention entry?

    Checked once, by signature, because the alternative is a silent wrong
    answer: an older ``_flash_attention_forward`` swallows unknown keywords into
    **kwargs and passes them to flash-attn, which does not know them either.
    """
    try:
        import inspect

        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except Exception:  # noqa: BLE001 - any import shape means "do not risk it"
        return False
    params = inspect.signature(_flash_attention_forward).parameters
    return all(name in params for name in
               ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"))


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


def check_task_weighting_supported(config, *, use_teacher_kl_loss: bool, ulysses_sequence_parallel_size: int):
    """Refuse a configuration whose loss the per-task row weights do not describe.

    The weights carry the WHOLE normalisation of the loss (see
    ``verl/trainer/ppo/task_loss_weights.py``), which makes them fragile in one
    specific way: anything that re-scales the loss afterwards, aggregates it
    differently, or adds a term that was normalised some other way does not
    produce a wrong-looking number -- it produces a plausible one, with the task
    weighting quietly undone. Every check below is that failure mode.

    Note what is NOT here: ``pg_loss_coef``. Pure OPD required it to be 0 because
    it only knew how to weight the teacher KL, so a live policy gradient would
    have been added to a differently-normalised number. This actor weights the
    policy-gradient, entropy and teacher-KL terms with the same row weights, so
    OPD+GRPO can run with the policy gradient on and the coefficients between the
    terms keep meaning what they say.

    Split out of ``update_policy`` so it can be tested without standing up FSDP.
    """
    assert not config.use_dynamic_bsz, (
        "per-task loss normalisation is incompatible with use_dynamic_bsz "
        "(its mini-batch scaling assumes an unweighted token-mean)"
    )
    assert ulysses_sequence_parallel_size == 1, (
        "per-task loss normalisation assumes DP size == world size; "
        f"got ulysses_sequence_parallel_size={ulysses_sequence_parallel_size}"
    )
    assert config.ppo_epochs == 1, (
        "per-task loss normalisation assumes one pass over each mini-batch; "
        f"got ppo_epochs={config.ppo_epochs}"
    )
    # The weighted path replaces the token-mean with a weighted token-sum, so a
    # configured loss_agg_mode would be silently ignored rather than honoured.
    # Only the mode the weights were derived against is allowed.
    agg_mode = config.loss_agg_mode
    assert agg_mode == "token-mean", (
        "per-task loss normalisation replaces the token-mean with a weighted "
        f"token-sum; loss_agg_mode={agg_mode!r} would be ignored"
    )
    # GSPO aggregates a sequence-level ratio with seq-mean-token-mean, a different
    # normalisation that these row weights do not describe.
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    assert loss_mode == "vanilla", (
        f"per-task loss normalisation is only derived for the vanilla policy loss; "
        f"got policy_loss.loss_mode={loss_mode!r}"
    )
    assert use_teacher_kl_loss, (
        "per-task loss normalisation is for the distillation loss, but "
        "use_teacher_kl_loss is off"
    )
    for other in ("use_kl_loss", "use_sdl_loss", "use_sdar_loss"):
        assert not config.get(other, False), (
            f"per-task loss normalisation weights the policy-gradient, entropy and "
            f"teacher-KL terms; {other} is set and would keep the plain token-mean"
        )


def _unwrap_module(module):
    """The module that actually defines ``forward``, under FSDP / torch.compile.

    The loop is bounded because a broken wrapper chain must not hang startup.
    """
    inner = module
    for _ in range(8):
        nxt = None
        for attr in ("_fsdp_wrapped_module", "_orig_mod", "module"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and candidate is not inner:
                nxt = candidate
                break
        if nxt is None:
            break
        inner = nxt
    return inner


def model_vocab_size(module) -> Optional[int]:
    """The wrapped model's vocabulary size, or None if it does not say.

    Needed to size the per-token diagnostic's dense accumulator. None rather than
    a guess: an accumulator too small would index out of bounds on a real token,
    and one sized from the largest id seen so far would silently change shape
    between steps. The caller turns None into "run without the diagnostic".
    """
    cfg = getattr(_unwrap_module(module), "config", None)
    size = getattr(cfg, "vocab_size", None)
    if size is None:
        size = getattr(getattr(cfg, "text_config", None), "vocab_size", None)
    return int(size) if size else None


def _supports_logits_to_keep(module) -> bool:
    """Does the wrapped HF model's forward take ``logits_to_keep``?

    Checked once at construction so ``response_only_logits`` fails at worker init
    rather than in the first micro-batch.
    """
    import inspect

    inner = _unwrap_module(module)
    try:
        params = inspect.signature(inner.forward).parameters
    except (TypeError, ValueError):
        return False
    return "logits_to_keep" in params or "num_logits_to_keep" in params


def _xt_normalizer_mu(pre_weight, snapshot, task_ids):
    """The divisor that reproduces a probe's applied weight, as a tensor.

    Where the snapshot has no valid mean the arm runs at ``W = 1`` rather than
    at the raw ``W~``, so the divisor that reproduces THAT is ``pre`` itself,
    not 1. Returning it keeps ``pre / mu`` and :func:`_xt_apply_normalizer` the
    same number at every position -- which is what lets the probe's state
    partition, built from this ``mu``, sum to the probe's own ``(W - 1) D``
    instead of to the live arm's.
    """
    floor = pre_weight.clamp(min=1e-12)
    if snapshot is None or task_ids is None:
        return floor
    dst = task_ids.reshape(-1).to(torch.long).clamp(min=0)
    mean = snapshot["mean"].to(pre_weight.device, pre_weight.dtype)
    valid = snapshot["valid"].to(pre_weight.device)
    mu = mean[dst].reshape(-1, 1).expand_as(pre_weight)
    return torch.where(valid[dst].reshape(-1, 1), mu.clamp(min=1e-12), floor)


def _xt_apply_normalizer(pre_weight, snapshot, task_ids):
    """A probe's applied weight: the same divisor rule the training path uses.

    The probes exist to bracket the go/no-go, so they have to be measured the
    way the arm is measured -- a raw W~ has a different spread from a normalised
    one, and the threshold is on the normalised quantity.
    """
    return pre_weight / _xt_normalizer_mu(pre_weight, snapshot, task_ids)


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

        # position-mode sign weighting divides by the previous call's per-task
        # mean; None until one has been measured, which is what makes the first
        # step run unnormalised rather than by a guess.
        self._sign_position_means = None
        # The parameter-free cross-teacher arm's one-step-lag state. Cumulative
        # across the whole run for the RMS and the reliability, per step for the
        # normaliser; all three are built lazily because they are indexed by task
        # and the task list arrives with the first batch.
        self._xt_rms = None
        self._xt_mean = None
        self._xt_adv = None
        self._xt_rms_snapshot = None       # (diag, diag_valid) from the previous step
        self._xt_mean_snapshot = None      # {"mean", "valid"} from the previous step
        self._xt_alpha = None              # (T, T) applied reliability
        self._xt_probe_mean = {}           # one normaliser per probe alpha
        self._xt_channel_mean = {}         # and one per channel counterfactual
        # The previous step's top-N token ids per scope, so turnover has
        # something to compare against. Not in the sidecar: a resumed run losing
        # one step of turnover is cheaper than a resume that silently compares
        # against a table from a different teacher set.
        self._xt_token_prev = None
        self._xt_pair_token_prev = None
        # Counts update_policy calls, so the dense-token stride is a function of
        # something every rank shares. Incremented once per call, at the end.
        self._xt_step_index = 0

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

        # Run lm_head on the response rows only. Everything _forward_micro_batch
        # returns is sliced to [:, -response_length-1:-1] before it leaves, so the
        # prompt rows' logits are computed, materialized at (rows, vocab) and then
        # dropped -- and on this mixture prompts are ~3x responses, so that is
        # about three quarters of the vocab projection, forward and backward, plus
        # the largest activation in the step. The transformer body still runs on
        # every token (causal attention: a response position reads the prompt's
        # KV); only the projection moves behind the row selection.
        self.response_only_logits = bool(self.config.get("response_only_logits", False))
        if self.response_only_logits:
            if not self.use_remove_padding:
                raise ValueError("response_only_logits requires use_remove_padding=True")
            if self.use_fused_kernels:
                raise ValueError("response_only_logits and use_fused_kernels are mutually exclusive")
            if self.use_ulysses_sp:
                raise ValueError("response_only_logits is not supported with ulysses sequence parallel")
            if not _supports_logits_to_keep(actor_module):
                raise ValueError(
                    "response_only_logits needs a model whose forward accepts `logits_to_keep` "
                    "with a tensor index (transformers >= 4.51 for Qwen3). Set it to False, or "
                    "upgrade transformers. Failing here rather than silently falling back: the "
                    "fallback is the slow path this flag exists to avoid."
                )
            print("Actor response_only_logits=True (lm_head on response rows only)")

    def _topk_from_response_logits(
        self, logits_resp, sel_indices, sel_slot, batch_size, seqlen, response_length, topk_k, topk_ids,
        lse=None,
    ):
        """Top-k distillation outputs from logits that already cover only the
        response rows.

        ``sel_indices`` are those rows' positions in the flattened (batch, seqlen)
        grid, so ``pad_input`` scatters them straight back into the window the
        caller slices. A window position missing from the packed batch (response
        padding) is zero-filled, which is what the full-logits path produced there
        too.

        ``lse`` is the full-vocabulary normaliser. It is a parameter because the
        caller may already need it for its own reasons; computing it twice is a
        second full reduction over (n_resp, vocab), which is the widest tensor in
        the step.
        """
        if lse is None:
            lse = torch.logsumexp(logits_resp, dim=-1, keepdim=True)  # (n_resp, 1)
        if topk_k is not None:
            # sorted=False: the KL sums over the support, so the order within the
            # k is never read.
            tvals, tids = torch.topk(logits_resp, k=topk_k, dim=-1, sorted=False)  # (n_resp, k)
            # Use float32 for pad_input: bf16 cannot represent vocab ids
            # (>256) exactly, and float32 keeps log-probs precise.
            t_lp_rmpad = (tvals - lse).float()
            full_t_lp = pad_input(t_lp_rmpad, indices=sel_indices, batch=batch_size, seqlen=seqlen)
            full_t_id = pad_input(tids.float(), indices=sel_indices, batch=batch_size, seqlen=seqlen)
            return (
                full_t_lp[:, -response_length - 1 : -1, :],
                full_t_id[:, -response_length - 1 : -1, :].round().long(),
            )
        # Read topk_ids (bs, response_len, k) directly at each selected row's
        # (sample, response-slot). The original path scattered it into a
        # (bs, seqlen, k) zero tensor and gathered that back through `indices`,
        # which is the same map computed the long way round.
        ids_resp = topk_ids[sel_indices // seqlen, sel_slot]  # (n_resp, k)
        s_lp_rmpad = (logits_resp.gather(-1, ids_resp) - lse).float()  # (n_resp, k), keeps grad
        full_s_lp = pad_input(s_lp_rmpad, indices=sel_indices, batch=batch_size, seqlen=seqlen)
        return full_s_lp[:, -response_length - 1 : -1, :]

    def _cross_teacher_planes(self, data, support_ids):
        """``(base, off_teachers)`` log-probs at ``support_ids``.

        The same exchange the on-task teacher went through, once per model: the
        base policy in column 0 of ``sign_cache_ids``, then this row's off-task
        teachers. They were cached by the driver on the rows they are off-task
        for, at the same granularity, so the lookup is the same shape.

        Shared by both cross-teacher arms because both need exactly these four
        models on exactly one support; a second copy is how the two would come
        to read different things and still be called the same experiment.

        One exchange for all of them, not one each. They share ``support_ids``,
        and that is the largest thing the exchange all-gathers -- (n, resp, k)
        int64 against (n,) keys -- so asking per plane sent it three times and
        paid three ``all_reduce`` pairs and, offloaded, three device-to-host
        copies of the gathered keys. See
        :func:`~verl.workers.teacher_cache.exchange_teacher_logprobs_multi`,
        which is bit-identical to the per-plane calls.
        """
        planes = self._teacher_logprobs_at_planes(
            cache_ids=data["sign_cache_ids"],
            ids=support_ids,
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
        )
        return planes[0], torch.stack(planes[1:], dim=-1)

    def _xt_token_tables(
        self, *, built, teacher_kl, data, support_ids, student_topk_logprob,
        on_task_logprob, response_mask, task_ids, report_epsilon, tables, roles,
        planes, raw_shifts=None, alpha_table=None, row_reward=None,
        row_advantage=None, push=None,
    ):
        """Name the tokens the weighting acted on, and who supplied the evidence.

        Three tables off one quantity: ``per_candidate_shift`` is each
        candidate's share of the nats the position moved, and the per-state
        table, the per-token table and the source table all decompose exactly
        that. Computing it once is what stops three views of one number from
        drifting apart.

        The two AGGREGATE tables are fed the STANDARDIZED shifts against a zero
        base, so their state labels and their deadzone are in RMS units and
        comparable across teachers -- the same trick build_position_weight uses.
        The teacher's probability travels separately, since a standardized shift
        does not exponentiate to one.

        The EVENT dump is fed the real log-probs. It is the one table whose
        columns are named for what the four models said, and it exponentiates
        them: handed the shifts it would write exp(delta_hat) into p_on and
        exp(0) = 1 into p_base, which is not what either name means and is not
        detectable downstream. The shifts ride in their own columns beside them,
        which is what makes a row readable as "the teacher put this much mass
        here, and that was N RMS above the base".

        Args:
            planes: ``(base_logprob, off_task_logprobs)`` at ``support_ids`` --
                the raw ones the shifts were computed from.
        """
        base_plane, off_planes = planes
        token_stats, pair_token_stats, event_stats, pair_event_stats = tables
        if all(t is None for t in tables):
            return
        shift = per_candidate_shift(built, teacher_kl)
        zero_base = torch.zeros_like(built["hat_on"])
        # The same nats, split over the sources that caused them. W~ - 1 is
        # sum_v p_on(v)[ |c(v)| + sum_m a_m |d_m(v)| ], so a source's share of
        # the position's shift is its own term over mu times D -- which is
        # exactly evidence_by_source scaled the way per_candidate_shift scales
        # the total. Filing the TOTAL against every source instead would let a
        # source with alpha = 0 appear to have moved the KL, and would count the
        # corroboration term, which no single source caused, once per source.
        inv_mu = (1.0 / built["mu"].clamp(min=1e-12)).unsqueeze(-1)
        source_shift = built["evidence_by_source"] * (inv_mu * teacher_kl.detach().to(
            built["evidence_by_source"].dtype
        ).unsqueeze(-1)).unsqueeze(-1)

        if token_stats is not None:
            token_stats.update(
                support_ids=support_ids, state=built["state"],
                weight=1.0 + built["evidence"], on_task_logprob=on_task_logprob,
                response_mask=response_mask, task_ids=task_ids, effect=shift,
            )
        if pair_token_stats is not None and task_ids is not None:
            pair_token_stats.update(
                support_ids=support_ids,
                on_task_logprob=built["hat_on"], off_task_logprobs=built["hat_off"],
                base_logprob=zero_base, response_mask=response_mask, task_ids=task_ids,
                off_plane_tasks=data["sign_off_tasks"], deadzone=report_epsilon,
                # Per source. The corroboration term reaches no column here on
                # purpose: it is what ALL the teachers agreed on, so charging it
                # to one of them is the error this table exists to avoid. It is
                # reported whole by evidence/shared_* and per token by the state
                # table, and source_shift + shared is the position's total.
                effect=source_shift, mass=built["teacher_prob"],
            )
        if event_stats is not None:
            row_scores = data.get("token_level_scores", None)
            event_stats.update(
                support_ids=support_ids, state=built["state"],
                weight=1.0 + built["evidence"], effect=shift,
                on_task_logprob=on_task_logprob, off_task_logprobs=off_planes,
                base_logprob=base_plane, student_logprob=student_topk_logprob,
                response_mask=response_mask, responses=data["responses"],
                norm=built["mu"], teacher_kl=teacher_kl, task_ids=task_ids,
                roles=(token_roles(data["responses"], roles) if roles else None),
                reward=(row_scores.sum(dim=-1) if row_scores is not None else None),
                shift_on=built["hat_on"], shift_off=built["hat_off"],
            )
        if pair_event_stats is not None and push is not None:
            self._xt_pair_events(
                stats=pair_event_stats, built=built, teacher_kl=teacher_kl, data=data,
                support_ids=support_ids, student_topk_logprob=student_topk_logprob,
                on_task_logprob=on_task_logprob, response_mask=response_mask,
                task_ids=task_ids, report_epsilon=report_epsilon, roles=roles,
                planes=planes, raw_shifts=raw_shifts, alpha_table=alpha_table,
                row_reward=row_reward, row_advantage=row_advantage, push=push,
            )

    def _xt_pair_events(
        self, *, stats, built, teacher_kl, data, support_ids, student_topk_logprob,
        on_task_logprob, response_mask, task_ids, report_epsilon, roles, planes,
        raw_shifts, alpha_table, row_reward, row_advantage, push,
    ):
        """One row per (candidate, source), with every column the schema names.

        Assembled here rather than inside the sampler because every input is
        already resident at this call site and passing eighteen tensors through
        a second function is how two of them come to be the wrong pair.
        """
        base_plane, off_planes = planes
        bs, resp, k = support_ids.shape
        n_off = off_planes.size(-1)
        dev = support_ids.device
        # Three shapes reach this table and each needs its own broadcast. Named
        # for what they come in as, so a column cannot be widened by the wrong
        # one and land silently transposed.
        k4 = lambda t: t.unsqueeze(-1).expand(bs, resp, k, n_off)          # (bs, resp, k)
        p4 = lambda t: t.reshape(bs, resp, 1, 1).expand(bs, resp, k, n_off)  # (bs, resp)

        state = pair_state_index(
            hat_on=built["hat_on"], hat_off=built["hat_off"], deadzone=report_epsilon
        )
        dst = task_ids.reshape(-1, 1, 1, 1).expand(bs, resp, k, n_off)
        src = data["sign_off_tasks"].reshape(bs, 1, 1, n_off).expand(bs, resp, k, n_off)
        alpha = (
            torch.zeros(bs, n_off, device=dev)
            if alpha_table is None
            else alpha_table.to(dev)[
                task_ids.reshape(-1).clamp(min=0).unsqueeze(-1),
                data["sign_off_tasks"].to(torch.long).clamp(min=0),
            ]
        )
        inv_mu = (1.0 / built["mu"].clamp(min=1e-12)).unsqueeze(-1)
        kl32 = teacher_kl.detach().to(torch.float32)
        src_shift = built["evidence_by_source"] * (inv_mu * kl32.unsqueeze(-1)).unsqueeze(-1)
        extra = ((built["weight"] - 1.0).unsqueeze(-1) * push["g0"])
        weighted = built["weight"].unsqueeze(-1) * push["g0"]
        # The same added push, split by what caused it. g0 is a property of the
        # candidate and the three shares are properties of the position, so the
        # product is exact rather than apportioned -- and the per-source column
        # is the one that differs across the source rows of a candidate, which
        # extra_logit_push does not.
        g0 = push["g0"]
        extra_src = built["push_by_source"].unsqueeze(2) * g0.unsqueeze(-1)
        extra_src_all = (built["push_by_source"].sum(dim=-1)).unsqueeze(-1) * g0
        extra_shared = built["push_shared"].unsqueeze(-1) * g0
        extra_norm = built["push_normalizer"].unsqueeze(-1) * g0
        y1 = data["responses"].unsqueeze(-1)
        sampled = (support_ids == y1)
        turn = turn_index(response_mask)
        role = (
            torch.zeros(bs, resp, dtype=torch.long, device=dev) if not roles
            else token_roles(data["responses"], roles)
        )
        raw_on = (raw_shifts or {}).get("on", torch.zeros_like(built["hat_on"]))
        raw_off = (raw_shifts or {}).get("off", torch.zeros_like(built["hat_off"]))
        zero_row = torch.zeros(bs, device=dev)
        rew = zero_row if row_reward is None else row_reward.detach().reshape(-1).to(torch.float32)
        adv = zero_row if row_advantage is None else row_advantage.detach().reshape(-1).to(torch.float32)
        row4 = lambda t: t.reshape(-1, 1, 1, 1).expand(bs, resp, k, n_off)

        columns = {
            "token_id": k4(support_ids),
            "dst": dst, "src": src, "pair_state": state,
            "state": k4(built["state"]),
            "role": p4(role), "turn": p4(turn),
            "position": p4(torch.arange(resp, device=dev).reshape(1, resp).expand(bs, resp)),
            "row_len": row4(response_mask.to(torch.long).sum(dim=1)),
            "is_sampled": k4(sampled.to(torch.long)),
            "p_base": k4(base_plane.detach().to(torch.float32).exp()),
            "p_on": k4(on_task_logprob.detach().to(torch.float32).exp()),
            "p_source": off_planes.detach().to(torch.float32).exp(),
            "p_student": k4(student_topk_logprob.detach().to(torch.float32).exp()),
            "delta_on_raw": k4(raw_on), "delta_source_raw": raw_off,
            "delta_on_std": k4(built["hat_on"]), "delta_source_std": built["hat_off"],
            "alpha_source": alpha.reshape(bs, 1, 1, n_off).expand(bs, resp, k, n_off),
            "shared_evidence": k4(built["teacher_prob"] * built["common"].abs()),
            "source_evidence": built["evidence_by_source"],
            "pre_weight": p4(built["pre_weight"]), "applied_weight": p4(built["weight"]),
            "teacher_kl": p4(kl32),
            "source_attributed_kl_shift": src_shift,
            "weighted_logit_push": k4(weighted), "extra_logit_push": k4(extra),
            "extra_push_source": extra_src,
            "extra_push_sources_all": k4(extra_src_all),
            "extra_push_shared": k4(extra_shared),
            "extra_push_normalizer": k4(extra_norm),
            "advantage": row4(adv), "reward": row4(rew),
        }
        width = stats.context
        pad = torch.nn.functional.pad(data["responses"], (width, width))
        ctx = pad.unfold(1, 2 * width + 1, 1)[:, :resp]          # (bs, resp, 2w+1)
        stats.update(
            columns=columns,
            group=(
                stats._pair_index(dst.clamp(min=0), src.clamp(min=0)).clamp(min=0)
                * stats.n_states + state
            ),
            valid=(response_mask.to(torch.bool).reshape(bs, resp, 1, 1).expand(bs, resp, k, n_off)
                   & (dst >= 0) & (src >= 0)),
            context_ids=ctx.reshape(bs, resp, 1, 1, -1).expand(bs, resp, k, n_off, 2 * width + 1),
            shift=src_shift,
            push=k4(extra),
        )

    def _read_sidecar_on_rank_zero(self, path):
        """One reader, then a broadcast. Returns the state, or None if absent.

        Rank 0 writes the sidecar, so rank 0 reads it. Letting every rank open
        the same file works on a shared filesystem and silently does not
        elsewhere: a rank that reads a stale or missing copy continues from a
        DIFFERENT accumulated scale than its neighbours, and since the weight is
        built rank-locally nothing downstream compares them. The broadcast makes
        that impossible rather than unlikely.

        The presence decision is broadcast too. If rank 0 has no file and the
        others do, they must all agree to cold-start, or the ranks that restored
        would be one step ahead in every statistic.
        """
        dist_on = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if dist_on else 0
        payload = [None]
        if rank == 0 and path and os.path.exists(path):
            payload[0] = torch.load(path, map_location="cpu", weights_only=False)
        if dist_on:
            torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def _xt_accumulate_reliability(
        self, *, data, built, student_topk_logprob, support_ids, response_mask,
        task_ids, diag, outside_counter,
    ):
        """This step's rows into the reliability statistic, read one step later.

        The score is the source's RESIDUAL opinion at the token the student
        actually emitted, measured against what that opinion was worth on
        average under the student's own distribution. The residual rather than
        the full shift so that the part every teacher shares -- generically good
        tokens -- cannot inflate one source's credit; the FULL shift is what the
        alpha this produces then gates, and keeping the two apart is the whole
        reason the evidence function does not take a residual.

        The emitted token is usually in the support but need not be, so it is
        looked up on its own: one extra id per model, resolved from the same
        cached hidden states, with no extra forward and -- deliberately -- no
        change to the KL's support. Widening the support to include it would
        move the loss to make a diagnostic tidier.

        Does nothing on an arm with no advantages, which leaves alpha at zero
        and the corroboration channel running alone. That is the pure-OPD case
        and it is a configuration, not a failure.
        """
        row_adv = data.get("adv_row_value", None)
        informative = data.get("adv_group_informative", None)
        if row_adv is None or informative is None or task_ids is None:
            return

        y1 = data["responses"].unsqueeze(-1)
        on_y = self._teacher_logprobs_at(
            cache_ids=data["teacher_cache_ids"], ids=y1,
            input_ids=data["input_ids"], attention_mask=data["attention_mask"],
        )
        base_y, off_y = self._cross_teacher_planes(data, y1)
        std_y = standardize_policy_shifts(
            shifts=compute_raw_policy_shifts(
                on_task_logprob=on_y, off_task_logprobs=off_y, base_logprob=base_y
            ),
            diag=diag[0], diag_valid=diag[1],
            task_ids=task_ids, off_plane_tasks=data["sign_off_tasks"],
        )
        keep = built["available"].reshape(-1, 1, 1).to(std_y["on"].dtype)
        hat_on_y, hat_off_y = std_y["on"] * keep, std_y["off"] * keep.unsqueeze(-1)
        dec_y = decompose_common_residual(hat_on=hat_on_y, hat_off=hat_off_y)

        score = residual_support_score(
            residual_at_sampled=dec_y["residual"][:, :, 0, :],
            residual=built["residual"],
            student_logprob=student_topk_logprob,
            response_mask=response_mask,
        )
        # The same score for the row's OWN teacher, carried only so the
        # diagnostics can partial it out: "this source agrees with the row's own
        # teacher" is the obvious confound for "this source predicts reward".
        on_score = residual_support_score(
            residual_at_sampled=(hat_on_y - dec_y["common"])[:, :, :1],
            residual=(built["hat_on"] - built["common"]).unsqueeze(-1),
            student_logprob=student_topk_logprob,
            response_mask=response_mask,
        ).reshape(-1)

        self._xt_adv.update(
            advantage=row_adv,
            support_score=score,
            on_support_score=on_score,
            length=response_mask.to(torch.float32).sum(dim=1),
            informative=informative,
            task_ids=task_ids,
            off_plane_tasks=data["sign_off_tasks"],
            group_ids=data.get("adv_group_id", None),
        )
        # Coverage of the centring approximation: the expectation the score is
        # measured against runs over the top-k with a zero tail residual, so an
        # emitted token outside it is not in its own baseline.
        inside = (support_ids == y1).any(dim=-1)
        valid = response_mask.to(torch.bool)
        outside_counter[0] += ((~inside) & valid).sum().to(torch.float64)
        outside_counter[1] += valid.sum().to(torch.float64)

    def _xt_rms_metrics(self, task_id_names) -> dict:
        """The scale itself, and how far out of its domain each teacher reaches.

        ``off_to_in_domain_ratio`` is the direct measurement of the thing the
        DIAGONAL divisor exists to preserve: how much less a teacher moves on
        another task's states than on its own. Dividing by the
        destination-conditioned RMS instead would force this to 1 by
        construction and call the resulting noise a full unit of signal.
        """
        snap = self._xt_rms.snapshot()
        # This step's rows alone. The weight divides by the cumulative sigma; the
        # current one is what says whether that divisor still describes the run.
        # Over 150 steps the student moves, so the teachers' shifts on its states
        # are not stationary, and a scale that averages step 1 into step 150 is a
        # transfer strength nobody chose. The drift ratio is the reading: at 1
        # the cumulative divisor is current, and away from it by a lot the arm is
        # standardising against a policy that no longer exists.
        cur = self._xt_rms.snapshot(scope="current")
        name = lambda t: task_id_names[t] if task_id_names and t < len(task_id_names) else f"task{t}"
        out = {}
        for d in range(self._xt_rms.n_tasks):
            if float(snap["n"][d]) <= 0:
                continue
            for m in range(self._xt_rms.n_tasks):
                if not bool(snap["valid"][d, m]):
                    continue
                head = f"kl_weight/rms/{name(m)}__on__{name(d)}"
                out[f"{head}/cumulative"] = float(snap["sigma"][d, m])
                if bool(cur["valid"][d, m]):
                    out[f"{head}/current"] = float(cur["sigma"][d, m])
                    if float(snap["sigma"][d, m]) > 0:
                        # Named both ways: drift_ratio is what a dashboard
                        # watches, current_to_cumulative is what the write-up
                        # calls it. One expression, so they cannot disagree.
                        ratio = float(cur["sigma"][d, m]) / float(snap["sigma"][d, m])
                        out[f"{head}/drift_ratio"] = ratio
                        out[f"{head}/current_to_cumulative"] = ratio
                if d != m and bool(snap["valid"][m, m]) and float(snap["sigma"][m, m]) > 0:
                    out[f"{head}/off_to_in_domain_ratio"] = (
                        float(snap["sigma"][d, m]) / float(snap["sigma"][m, m])
                    )
            out[f"kl_weight/rms/{name(d)}/n_positions"] = float(snap["n"][d])
            out[f"kl_weight/rms/{name(d)}/n_positions_current"] = float(cur["n"][d])
        return out

    def _xt_reliability_metrics(self, task_id_names) -> dict:
        """alpha, and everything that says whether to believe it.

        ``rho_lcb95`` and the two partials are reported and never multiplied
        into anything: a confidence level is a knob, and so is a choice of which
        confound to trust. What they are for is reading a positive alpha that
        the rectifier's own small-sample bias could have produced on its own.
        """
        out = {}
        # This step's rows alone, on the same estimator. rho_cumulative is what
        # alpha is built from; rho_current is the only thing that can say it has
        # gone stale, and a pair whose two disagree in SIGN is one where the
        # applied alpha and the step's own evidence point opposite ways.
        current = self._xt_adv.alpha(task_names=task_id_names, scope="current")
        disagree = total = 0
        for (dst, src), row in self._xt_adv.alpha(task_names=task_id_names).items():
            head = f"kl_weight/adv/{src}__on__{dst}"
            now = current.get((dst, src), {})
            out[f"{head}/alpha_applied"] = row["alpha"]
            out[f"{head}/n_rollouts_cumulative"] = row["n"]
            out[f"{head}/n_rollouts_current"] = now.get("n", 0.0)
            out[f"{head}/veto_rate"] = float(row["rho"] is not None and row["rho"] < 0)
            for key in ("rho", "rho_lcb95", "rho_length_controlled", "rho_length_on_controlled"):
                if row[key] is not None:
                    out[f"{head}/{key}"] = row[key]
            # Why an alpha is small. GRPO is group-relative, so a prompt whose
            # rollouts all scored the same gives every row zero advantage and
            # carries no information -- without this, "the source does not
            # predict reward" and "there was nothing to predict" are one number.
            for scope, src_row in (("cumulative", row), ("current", now)):
                frac = src_row.get("informative_group_frac", None)
                if frac is not None:
                    out[f"{head}/informative_group_frac_{scope}"] = frac
            # The two spreads the correlation is a ratio of, this step only. A
            # rho that collapsed because the ADVANTAGE stopped varying is a
            # different event from one that collapsed because the source stopped
            # speaking, and rho alone reports them identically.
            for key in ("adv_std", "support_score_std", "n_grouped"):
                if now.get(key, None) is not None:
                    out[f"{head}/{key}_current"] = now[key]
                if row.get(key, None) is not None:
                    out[f"{head}/{key}_cumulative"] = row[key]
            rho_now = now.get("rho", None)
            if rho_now is not None:
                out[f"{head}/rho_current"] = rho_now
            if row["rho"] is not None:
                out[f"{head}/rho_cumulative"] = row["rho"]
            if rho_now is not None and row["rho"] is not None:
                out[f"{head}/rho_current_minus_cumulative"] = rho_now - row["rho"]
                # What the applied alpha WOULD have been on this step's evidence
                # alone. The loss never sees it; the gap is what says the applied
                # one is running on a relationship that has moved.
                out[f"{head}/alpha_delta"] = max(0.0, rho_now) - row["alpha"]
                total += 1
                if rho_now * row["rho"] < 0:
                    disagree += 1
        if total:
            # One number for "is the reliability estimate still describing this
            # run": the fraction of ordered pairs whose step-local rho has the
            # opposite sign to the cumulative one the loss is using. Pairs where
            # either is undefined are excluded rather than counted as agreeing.
            out["kl_weight/adv/rho_sign_disagree"] = disagree / total
        return out

    def _refresh_sign_position_means(self, stats, task_id_names):
        """Per-task mean position weight, pooled over ranks, for the NEXT call.

        ``position`` mode has to divide by a per-task mean or it cannot be told
        apart from a larger ``teacher_kl_loss_coef``: the weight table has no
        entry below 1.0, so the mean sits above 1 whenever anything agrees, and an
        un-normalised arm would simply have distilled harder.

        The mean it needs is step-global, and the weight only exists inside the
        training forward, which sees one micro-batch at a time. Two ways out are
        worse than this one: a micro-batch's own mean makes the objective depend
        on how the batch was split, and a second forward to measure it first costs
        the forward. The mean is a slow quantity -- the agreement rate moved from
        0.26 to 0.17 over 150 steps -- so the previous call's value is the right
        constant, and being a constant it cannot bias a gradient, only the
        effective coefficient, by a fraction of a percent. The first step runs
        unnormalised and ``sign_weight/*/w_mean_pre_norm`` says by how much.

        Pooled across ranks because every rank has to divide by the SAME number:
        rank-local means would scale each rank's share of the loss differently,
        which is a change to the objective rather than to its normalisation.
        """
        import torch.distributed as dist

        n = len(task_id_names) if task_id_names else 0
        if n == 0:
            return
        acc = torch.zeros((n, 2), dtype=torch.float64, device=get_torch_device().current_device())
        for tid, total in stats.pos_w.items():
            if tid is None or not (0 <= tid < n):
                continue
            acc[tid, 0] = total
            acc[tid, 1] = stats.pos_n.get(tid, 0.0)
        if dist.is_initialized():
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        means = {tid: float(acc[tid, 0] / acc[tid, 1]) for tid in range(n) if acc[tid, 1] > 0}
        self._sign_position_means = means or None

    def _teacher_logprobs_at(self, cache_ids, ids, input_ids=None, attention_mask=None):
        """Teacher log-probs at ids the student just chose.

        The teacher's hidden states were cached wherever its forward ran, which is
        not where this micro-batch is being trained: between the two calls the rows
        are regrouped by task, padded, and reordered by ``_balance_batch`` to
        equalise tokens per rank. ``exchange_teacher_logprobs`` therefore asks every
        rank and sums the one answer that exists, raising if a row is unanswered or
        answered twice. See verl/workers/teacher_cache.py.

        Both sides work per ROW: one key locates the row's whole
        (response_length, hidden) block, and ``ids`` stays (bs, response_length, k).
        """
        if cache_ids is None:
            raise ValueError(
                "student_indexed_topk needs a `teacher_cache_ids` column locating each row's cached "
                "teacher hidden states; the batch has none."
            )
        return self._teacher_logprobs_at_planes(
            cache_ids=cache_ids.unsqueeze(1), ids=ids,
            input_ids=input_ids, attention_mask=attention_mask,
        )[0]

    def _teacher_logprobs_at_planes(self, cache_ids, ids, input_ids=None, attention_mask=None):
        """The same lookup for several models at once, over one support.

        ``cache_ids`` is (bs, P), one key column per model; the return is a list
        of P (bs, response_length, k) tensors in that column order. Everything
        the single-plane form documents applies unchanged -- this is where it is
        implemented, and the batching is a transport change the values do not
        see.
        """
        from verl.workers.teacher_cache import (
            exchange_teacher_logprobs_multi,
            get_teacher_cache,
            row_fingerprint,
        )

        # Derived from the rows being trained RIGHT HERE, so a key column shifted
        # against its batch is caught. The key alone would resolve cleanly and
        # return a real teacher log-prob for somebody else's sample. One row is
        # one row whichever model is read off it, so the same column checks every
        # plane.
        fingerprints = None
        if input_ids is not None and attention_mask is not None:
            fingerprints = row_fingerprint(input_ids, attention_mask)
        return exchange_teacher_logprobs_multi(
            get_teacher_cache(), cache_ids, ids, fingerprints=fingerprints
        )

    def _varlen_kwargs(self, cu_seqlens, max_seqlen_in_batch) -> dict:
        """The packed-sequence boundaries, handed over instead of re-derived.

        With ``use_remove_padding`` the model is given ``position_ids`` and HF's
        flash-attention path works the boundaries out itself: whether the
        sequences are packed at all, and how long the longest is. Both decisions
        are made on the device and read on the host -- flash-attn needs Python
        ints -- so each is a device-to-host sync, ONCE PER LAYER PER FORWARD.
        Twenty-eight layers, doubled because gradient checkpointing recomputes
        the forward inside the backward.

        ``unpad_input`` has already computed both, once, for the whole
        micro-batch. Passing them makes _flash_attention_forward skip the
        position_ids path entirely, and the values are the same ones it would
        have derived -- both come from the same attention_mask.

        Off in two cases:

        * Ulysses SP > 1. The sequence is split across ranks after this point,
          so boundaries computed on the unsplit batch describe a different
          tensor than the attention sees. verl's monkey_patch all-gathers
          position_ids for exactly that reason; handing it stale cu_seqlens
          would be wrong rather than merely slower.
        * A transformers whose entry point does not name the kwargs. It would
          take them into **kwargs and pass them to flash-attn, which does not
          know them either -- so the check is by signature, not by version.
        """
        if cu_seqlens is None or not _VARLEN_KWARGS or self.use_ulysses_sp:
            return {}
        if not _flash_attention_takes_varlen_kwargs():
            return {}
        return {
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlen_in_batch,
            "max_length_k": max_seqlen_in_batch,
        }

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, topk_k=None, topk_ids=None,
        need_log_prob=True, return_lse=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
        """
        Args:
            return_lse: alongside the top-k, hand back the full-vocabulary
                logsumexp and the packed-row map, so a caller can evaluate this
                model at ids chosen later without re-running it. Only on the
                ``response_only_logits`` path -- the row map is what it produces.
            need_log_prob: when False the sampled-token log-prob is not computed and
                ``None`` is returned in its place. Only honoured on the
                ``response_only_logits`` path (elsewhere it is a by-product of work
                that happens anyway). Pure top-k distillation never reads it: the KL
                is built from the top-k gather, and with pg_loss_coef=0 /
                use_kl_loss=False nothing else in update_policy touches it.

        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len), or None when need_log_prob=False
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
                # cu_seqlens and max_seqlen are kept, not discarded: handing them
                # to the attention entry saves a device-to-host sync per layer per
                # forward (see _varlen_kwargs). Older flash_attn returns fewer
                # values, so the unpack is by length rather than by position.
                unpadded = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad, indices = unpadded[0], unpadded[1]  # input_ids_rmpad (total_nnz, ...)
                cu_seqlens, max_seqlen_in_batch = (
                    (unpadded[2], unpadded[3]) if len(unpadded) >= 4 else (None, None)
                )
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

                # Response-only vocab projection. `sel` indexes the packed rows that
                # survive the final slice; handing it to the model as logits_to_keep
                # makes lm_head run on those hidden states only. The transformer body
                # is untouched -- it must still see every token.
                resp_only = self.response_only_logits and not multi_modal_inputs
                if resp_only:
                    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
                    extra_args["logits_to_keep"] = sel

                extra_args.update(self._varlen_kwargs(cu_seqlens, max_seqlen_in_batch))

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if resp_only:
                    # (n_resp, vocab): already restricted, so nothing below needs to
                    # re-select. Values are identical to indexing the full logits --
                    # lm_head is a per-position linear map, so selecting the rows
                    # before or after it is the same arithmetic.
                    logits_resp = output.logits.squeeze(0)
                    logits_resp.div_(temperature)

                    log_probs = None
                    if need_log_prob:
                        log_probs_resp = logprobs_from_logits(
                            logits=logits_resp,
                            labels=input_ids_rmpad_rolled[sel],
                            inplace_backward=False,  # topk/entropy below read logits_resp
                        )
                        full_log_probs = pad_input(
                            hidden_states=log_probs_resp.unsqueeze(-1),
                            indices=sel_indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

                    if calculate_entropy:
                        entropy_resp = self.compute_entropy_from_logits(logits_resp)
                        full_entropy = pad_input(
                            hidden_states=entropy_resp.unsqueeze(-1),
                            indices=sel_indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

                    if topk_k is not None or topk_ids is not None or return_lse:
                        # One reduction over (n_resp, vocab), shared. This is the
                        # widest tensor in the step, so computing the normaliser
                        # here and again for `lse` below was a second full pass
                        # over it for nothing.
                        lse_resp = torch.logsumexp(logits_resp, dim=-1, keepdim=True)
                        if topk_k is not None or topk_ids is not None:
                            topk_out = self._topk_from_response_logits(
                                logits_resp=logits_resp,
                                sel_indices=sel_indices,
                                sel_slot=sel_slot,
                                batch_size=batch_size,
                                seqlen=seqlen,
                                response_length=response_length,
                                topk_k=topk_k,
                                topk_ids=topk_ids,
                                lse=lse_resp,
                            )
                        if return_lse:
                            # The caller wants to evaluate this model at ids chosen
                            # later, which needs the normaliser and the row map --
                            # the projection itself it can redo for 2*H*k. With
                            # topk_k None the model's OWN top-k is not built at
                            # all, which is a full-vocabulary selection saved.
                            tlp, tids = topk_out if topk_k is not None else (None, None)
                            topk_out = (
                                tlp,
                                tids,
                                {
                                    "lse": lse_resp.float(),
                                    "sel": sel,
                                    "sel_indices": sel_indices,
                                    "batch_size": batch_size,
                                    "seqlen": seqlen,
                                    "response_length": response_length,
                                },
                            )
                    return entropy, log_probs, topk_out

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
                    #
                    # response_only_logits goes one step further and never builds the
                    # prompt rows' logits at all; this path stays for the runs that
                    # do not enable it (and for ulysses/multimodal, which it excludes).
                    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
                    topk_out = self._topk_from_response_logits(
                        logits_resp=logits_rmpad[sel],
                        sel_indices=sel_indices,
                        sel_slot=sel_slot,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        response_length=response_length,
                        topk_k=topk_k,
                        topk_ids=topk_ids,
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
        if return_hidden and not self.response_only_logits:
            raise ValueError(
                "student_indexed_topk needs ref.response_only_logits=True: the hidden states are handed "
                "back on the packed rows that path selects, and there is no row map without it."
            )
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
            want_topk = (not sampled_witness) or mb_i < witness_micro_batches
            sink = {}
            with torch.no_grad(), self._capture_last_hidden(sink if return_hidden else {}):
                _, _, topk_out = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=False,
                    topk_k=topk_k if want_topk else None,
                    return_lse=return_hidden,
                    # The teacher's own sampled-token log-prob is a log-softmax and
                    # a gather over the widest tensor in the step, and this call
                    # DISCARDS it -- the unpacking above keeps only topk_out. The
                    # actor path already worked this out for itself
                    # (need_log_prob above); the teacher path never passed the
                    # argument, so it defaulted to True and paid for it on every
                    # micro-batch of every frozen model. On this arm that is four
                    # models a row.
                    need_log_prob=False,
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

        # The stall watch measures the gap before each micro-batch, and the one
        # that spans two update_policy calls is a different animal from the ones
        # inside a step: it holds the return to the driver, its logging, and the
        # next batch's dispatch and H2D. Told where the boundary is, it compares
        # like with like instead of flagging that gap every step.
        actor_capture.new_step()

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
        # Exact token-mean under dynamic bsz: scale each micro-batch by its valid-token
        # share of the mini-batch (token-weighted) instead of by sample count, so the
        # objective is grouping-invariant and matches the true global token-mean.
        dynamic_bsz_token_scale = (
            self.config.use_dynamic_bsz
            and self.config.get("dynamic_bsz_token_scale", False)
            and self.config.loss_agg_mode == "token-mean"
        )
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
        # Whose top-k the KL's support comes from. Student-indexed resolves the
        # teacher from cached hidden states at update time, so the pre-scored
        # columns are replaced by the cache key that locates them.
        student_indexed_topk = teacher_topk_kl and bool(self.config.get("student_indexed_topk", False))
        if teacher_topk_kl:
            if student_indexed_topk:
                select_keys.append("teacher_cache_ids")
            else:
                select_keys += ["teacher_topk_logprobs", "teacher_topk_ids"]
        # Cross-teacher sign agreement: the base policy and the row's off-task
        # teachers, cached by the driver on the same rows and read here at the ids
        # the student is about to pick. Absent -> the weighting never runs, which
        # is what makes enable=false identical to the plain arm.
        sign_cfg = self.config.get("sign_weight", None)
        # Split deliberately. sign_enabled is config AND batch content, because a
        # batch without the cached planes cannot build weights. The accumulators
        # below run COLLECTIVES, so they must be constructed and reduced on the
        # config alone: a rank whose batch happened to lack the key would
        # otherwise skip an all_reduce and hang every other rank.
        sign_cfg_on = bool(sign_cfg and sign_cfg.get("enable", False))
        sign_enabled = sign_cfg_on and "sign_cache_ids" in data.batch.keys()
        # The parameter-free arm. It reads the same four models on the same
        # support and reaches the loss through the same one line, so it shares
        # every gate the sign arm has and differs only in how the scalar is
        # built. Both at once would train an arm that is neither.
        xt_cfg = self.config.get("cross_teacher_kl_weight", None)
        xt_cfg_on = bool(xt_cfg and xt_cfg.get("enable", False))
        assert not (sign_cfg_on and xt_cfg_on), (
            "sign_weight and cross_teacher_kl_weight are two mechanisms for one signal "
            "and both multiply the same teacher KL; enable one"
        )
        xt_enabled = xt_cfg_on and "sign_cache_ids" in data.batch.keys()
        xt_report_eps = float((xt_cfg or {}).get("report_epsilon", 0.1))
        if xt_enabled:
            assert teacher_topk_kl and use_teacher_kl_loss, (
                "cross_teacher_kl_weight scales the teacher KL over a shared top-k "
                "support; it needs kl_loss_type=topk_kl and use_teacher_kl_loss=true"
            )
            assert student_indexed_topk, (
                "cross_teacher_kl_weight measures every model on the STUDENT's top-k "
                "(support: student_topk); set actor.student_indexed_topk=true"
            )
            select_keys += ["sign_cache_ids", "sign_off_tasks"]
            for key in ("adv_row_value", "adv_group_informative", "adv_group_id"):
                if key in data.batch.keys():
                    select_keys.append(key)
            # The row's episode score. Two readers now: the event dump, where
            # "the weighting fires here" and "the weighting fires here on rows
            # that went on to score" are different findings, and the outcome
            # statistics, where it is what splits the reward buckets -- the
            # advantage cannot, being group-relative. Selected whenever the arm
            # is on rather than on the dump's switch, so turning the dump off
            # does not silently empty those buckets. Uniform across ranks
            # (the driver builds one batch), so it desynchronises nothing.
            if "token_level_scores" in data.batch.keys():
                select_keys.append("token_level_scores")
        if sign_enabled:
            # Either support works -- the student's top-k, resolved above, or the
            # teacher's own, already selected into select_keys by the branch above.
            # What the weights cannot be built from is the single-token estimator,
            # which produces no candidate set for the four models to share.
            assert teacher_topk_kl, (
                "sign weighting needs a top-k support for the four models to share; "
                "set algorithm.opd.kl_loss_type=topk_kl"
            )
            # The weighting reaches the loss ONLY through the teacher KL -- in
            # both modes -- and its diagnostics are read off that KL, so with the
            # term switched off the whole mechanism is three frozen forwards
            # producing nothing. Refuse rather than run inert.
            assert use_teacher_kl_loss, (
                "sign weighting only acts on the teacher KL, which is off; set "
                "actor.use_teacher_kl_loss=true or algorithm.opd.sign_weight.enable=false"
            )
            select_keys += ["sign_cache_ids", "sign_off_tasks"]
            # The event dump reports the row's episode score beside the
            # candidate, because "the weighting fires here" and "the weighting
            # fires here on rows that went on to score" are different findings.
            # Config-gated, and the presence check is uniform across ranks (the
            # driver builds one batch), so it cannot desynchronise anything.
            if bool((sign_cfg.get("event_dump", None) or {}).get("enable", False)) and (
                "token_level_scores" in data.batch.keys()
            ):
                select_keys.append("token_level_scores")
        sign_mode = str(sign_cfg.get("mode", "target")) if sign_enabled else None
        sign_agree = float(sign_cfg.get("agree_weight", 1.25)) if sign_enabled else 1.0
        sign_agree_neg = float(sign_cfg.get("agree_neg_weight", 0.75)) if sign_enabled else 1.0
        sign_disagree = float(sign_cfg.get("disagree_weight", 1.0)) if sign_enabled else 1.0
        sign_deadzone = float(sign_cfg.get("deadzone", 0.1)) if sign_enabled else 0.0
        # Multitask runs tag every row with its task id (see RayPPOTrainer._attach_task_ids)
        # so the loss metrics below can also be reported per task. Absent in single-task runs.
        task_id_names = data.meta_info.get("task_id_names", None)
        if "task_ids" in data.batch.keys():
            select_keys.append("task_ids")
        # Per-task normalised distillation loss: the driver attaches a per-row weight
        # (see verl/trainer/ppo/task_loss_weights.py) that makes each task's share of
        # the loss 1/num_tasks instead of its share of the response tokens. Absent ->
        # plain token-mean, i.e. nothing below changes.
        task_weighted = TASK_LOSS_WEIGHT_KEY in data.batch.keys()
        if task_weighted:
            select_keys.append(TASK_LOSS_WEIGHT_KEY)
            check_task_weighting_supported(
                self.config,
                use_teacher_kl_loss=use_teacher_kl_loss,
                ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
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
        # that was ~450 forced syncs per step, timed as actor.task_metrics.
        # Deferring changes nothing about the values: the single read at the end
        # takes the same mean over micro-batches that append_to_dict +
        # reduce_metrics would have.
        deferred_metrics = {}

        def _defer(name, value, weight=None):
            deferred_metrics.setdefault(name, []).append(
                (value.detach(), None if weight is None else weight.detach())
            )

        def _defer_task(name, loss_mat, mask):
            """A task's token-mean for this micro-batch, deferred with its presence.

            Written out rather than calling agg_loss so an absent task yields 0
            instead of 0/0: the read at the end divides the sum of the values by
            the number of micro-batches the task appeared in, which is the same
            average the per-task metric has always reported -- it just no longer
            needs the device to tell the host which micro-batches those were.

            The formula is token-mean, which is why sync_free_task_metrics below
            also requires loss_agg_mode to be token-mean. Under seq-mean-* this
            would be a different quantity reported under the same name, and
            nothing would say so.
            """
            den = mask.sum()
            value = (loss_mat * mask).sum() / den.clamp(min=1)
            _defer(name, value, weight=(den > 0).to(value.dtype))

        # Pooled across every micro-batch of this call and rendered once at the
        # end. Ratios cannot be emitted per micro-batch: the reducer would average
        # them, and a micro-batch with fewer valid tokens would weigh as much as a
        # full one.
        # Whether the per-task metric loop can run over ALL tasks, including the
        # ones with no rows in this micro-batch, and so skip the device read that
        # would say which those are -- torch.unique(...).tolist(), one host sync
        # per micro-batch. It can exactly when every metric the loop computes is
        # deferred: the branches below that call .item() inside the loop would
        # turn an absent task into a NaN, and would be paying a sync each anyway.
        # That is the pure-OPD shape, which inject_opd_config forces and the
        # intent locks pin -- teacher-KL only, no policy gradient, entropy,
        # reference KL or SD terms.
        #
        # loss_agg_mode is in the condition because _defer_task hard-codes the
        # token-mean formula; the aggregation is not a term that can be switched
        # off, so it has to be checked rather than assumed.
        sync_free_task_metrics = (
            pg_loss_coef == 0
            and self.config.entropy_coeff == 0
            and not self.config.use_kl_loss
            and not self.config.get("use_sdl_loss", False)
            and not self.config.get("use_sdar_loss", False)
            and self.config.loss_agg_mode == "token-mean"
        )

        sign_stats = SignWeightStats(task_names=task_id_names) if sign_enabled else None
        # ---- transfer measurement (opt-in, target mode) --------------------
        # Answers what nothing else here does: whether the STUDENT ended up
        # carrying anything from the teachers it is not trained on. Every other
        # number in this file describes the teachers' agreement or the size of
        # the rewrite. Gated on the CONFIG only -- see sign_cfg_on above.
        sign_dev = next(self.actor_module.parameters()).device
        n_task = len(task_id_names or [])
        target_mode = sign_cfg_on and str((sign_cfg or {}).get("mode", "target")) == "target"
        transfer_on = sign_cfg_on and bool(
            ((sign_cfg or {}).get("transfer_stats", None) or {}).get("enable", False)
        )
        rewrite_on = transfer_on and target_mode
        pair_on = sign_cfg_on and bool(
            ((sign_cfg or {}).get("pair_stats", None) or {}).get("enable", False)
        )
        # The two halves of transfer_stats have different prerequisites and are
        # gated separately. rewrite_stats DECOMPOSES a rewrite of the teacher's
        # probabilities, so it needs target mode -- there is nothing to decompose
        # in position mode, which scales the KL term instead. The ladder needs no
        # rewrite at all: it measures where the student sits relative to the
        # off-task teachers, which is a question about four frozen-or-not models
        # and is exactly as meaningful on a position arm. Gating them together
        # silently gave the position arm no ladder while its script passed
        # transfer_stats.enable=True.
        rewrite_stats = ScopeTermStats(names=REWRITE_TERMS, n_tasks=n_task, device=sign_dev) if rewrite_on else None
        # The position arm's counterpart, and NOT gated behind transfer_stats.
        # rewrite_stats is a decomposition of an optional diagnostic; this is the
        # only readout of what the direct-KL weighting did at all -- the arm has
        # until now reported one number, w_mean_pre_norm, which cannot tell a
        # redistribution from a larger teacher_kl_loss_coef. It costs a
        # (1+T, len(POSITION_TERMS)+1) float64 buffer and one all-reduce of it.
        position_stats = (
            ScopeTermStats(names=POSITION_TERMS, n_tasks=n_task, device=sign_dev)
            if (sign_cfg_on and not target_mode)
            else None
        )
        # The table's own range, so the concentration bands below are cuts of
        # something fixed rather than of the observed weights: a moving quantile
        # would report the same share at every step by construction.
        sign_weight_range = (
            (
                min(1.0, float((sign_cfg or {}).get("agree_weight", 1.0)), float((sign_cfg or {}).get("disagree_weight", 1.0))),
                max(1.0, float((sign_cfg or {}).get("agree_weight", 1.0)), float((sign_cfg or {}).get("disagree_weight", 1.0))),
            )
            if sign_cfg_on
            else (1.0, 1.0)
        )
        ladder_stats = OffTaskLadderStats(n_tasks=n_task, device=sign_dev) if (transfer_on and n_task) else None
        pair_stats = SignPairCounts(n_tasks=n_task, device=sign_dev) if (pair_on and n_task) else None
        student_resid_deadzone = float((sign_cfg or {}).get("student_resid_deadzone", 0.0)) if sign_cfg_on else 0.0
        # The parameter-free arm's three accumulators, built on the config alone
        # so every rank runs the same collectives whatever its micro-batch holds.
        # The RMS and the reliability are cumulative across the run; the
        # normaliser is a per-step mean, reset below once its snapshot is taken.
        if xt_cfg_on:
            # Every matrix here is indexed by task. Without task ids there is no
            # destination, no source and no ordered pair, so the arm would run
            # its three extra forwards and weight nothing -- and say so nowhere.
            assert n_task >= 3, (
                "cross_teacher_kl_weight is indexed by task and needs at least three "
                f"of them (a destination and two corroborating sources); the batch names {n_task}. "
                "Multitask routing attaches task_ids -- see RayPPOTrainer._attach_task_ids."
            )
            if self._xt_rms is None:
                self._xt_rms = CumulativePolicyShiftRMS(n_tasks=n_task, device=sign_dev)
                self._xt_mean = PreviousStepTaskKLWeightedMean(n_tasks=n_task, device=sign_dev)
                self._xt_adv = AdvantageReliabilityStats(
                    n_tasks=n_task, device=sign_dev,
                    max_groups=int((xt_cfg or {}).get("max_groups", 512)),
                )
                self._xt_alpha = torch.zeros((n_task, n_task), dtype=torch.float32)
                self._xt_probe_mean = {
                    probe_name(a): PreviousStepTaskKLWeightedMean(n_tasks=n_task, device=sign_dev)
                    for a in XT_PROBE_ALPHAS
                }
                self._xt_channel_mean = {
                    c: PreviousStepTaskKLWeightedMean(n_tasks=n_task, device=sign_dev)
                    for c in XT_CHANNEL_PROBES
                }
                # A checkpoint's accumulated state, if this process was resumed.
                # Restored here rather than in load_checkpoint because the
                # accumulators are indexed by task and do not exist until the
                # first batch names them.
                pending = getattr(self, "cross_teacher_sidecar_path", None)
                blob = self._read_sidecar_on_rank_zero(pending)
                if blob is not None:
                    restored = load_sidecar_state(
                        blob,
                        rms=self._xt_rms, mean=self._xt_mean, adv=self._xt_adv,
                        identity=getattr(self, "cross_teacher_identity", {}) or {},
                    )
                    self._xt_rms_snapshot = self._xt_rms.diagonal()
                    if restored is not None:
                        self._xt_alpha = restored.to(torch.float32)
                    print(f"[cross_teacher] resumed accumulated state from {pending}", flush=True)
                self.cross_teacher_sidecar_path = None
        # {tag: [ids]}, set by the worker at startup -- this process has no
        # tokenizer. Absent means the role column reports "format" throughout,
        # which is honest: nothing was classified. Read here rather than beside
        # the sign arm's tables because the role-keyed accumulators below are
        # built or skipped on whether it exists.
        sign_role_tags = getattr(self, "sign_role_tag_ids", None)
        xt_on = xt_cfg_on and self._xt_rms is not None
        xt_position_stats = (
            ScopeTermStats(names=XT_POSITION_TERMS, n_tasks=n_task, device=sign_dev) if xt_on else None
        )
        xt_state_stats = (
            ScopeTermStats(names=XT_STATE_TERMS, n_tasks=n_task, device=sign_dev) if xt_on else None
        )
        xt_pair_stats = PairEvidenceStats(n_tasks=n_task, device=sign_dev) if xt_on else None
        xt_grad_stats = (
            ScopeTermStats(names=XT_GRAD_TERMS, n_tasks=n_task, device=sign_dev) if xt_on else None
        )
        # WHERE in the text the arm acts, and WHERE in the episode. Both axes
        # vary WITHIN a row -- one response walks through <think>, <action> and
        # the scaffolding between them, and a multi-turn row spans several turns
        # -- so they need the per-position accumulator rather than the per-row
        # one every other table here uses.
        #
        # These are the two cuts an ablation cannot reconstruct from what was
        # already logged. "The arm moved 3% of the OPD budget" is the same
        # number whether it moved it into the tokens the environment executes or
        # into whitespace, and those are not the same finding.
        xt_role_position_stats = (
            PositionScopeTermStats(names=XT_POSITION_TERMS, n_scopes=len(ROLE_NAMES), device=sign_dev)
            if xt_on else None
        )
        xt_role_state_stats = (
            PositionScopeTermStats(names=XT_STATE_TERMS, n_scopes=len(ROLE_NAMES), device=sign_dev)
            if xt_on else None
        )
        # The reading the norm ratio and the pooled cosine cannot give: whether
        # the budget the arm ADDED at a kind of position pulls with the reward
        # gradient THERE. A pooled cosine of 0.1 is consistent with strong
        # agreement inside <action> and strong disagreement in the scaffolding.
        xt_role_grad_stats = (
            PositionScopeTermStats(names=XT_GRAD_TERMS, n_scopes=len(ROLE_NAMES), device=sign_dev)
            if xt_on else None
        )
        xt_turn_stats = (
            PositionScopeTermStats(names=XT_POSITION_TERMS, n_scopes=XT_TURN_BUCKETS, device=sign_dev)
            if xt_on else None
        )
        # The shape of W. w_cv cannot tell "1.02 nearly everywhere" from "1.00 at
        # 99% of positions and 3.0 at the rest", and those are the two mechanisms
        # the arm is being tested for.
        xt_weight_hist = WeightShiftHistogram(n_tasks=n_task, device=sign_dev) if xt_on else None
        # When Search DISAGREED with AlfWorld's own teacher, what did the arm do.
        # The state table sums the sources out and the source table sums the
        # states out, so the case the mechanism exists to arbitrate is in
        # neither. 144 cells at three tasks.
        xt_pair_state_stats = PairStateEvidenceStats(n_tasks=n_task, device=sign_dev) if xt_on else None
        # Per TRAJECTORY, not per position: "the arm moved 3% of the budget" and
        # "the arm moved 3% of the budget, almost all of it on rollouts that
        # failed" are the same number and opposite findings.
        xt_outcome_stats = OutcomeEffectStats(n_tasks=n_task, device=sign_dev) if xt_on else None
        # The same rows keyed by SOURCE as well as by outcome -- the join the two
        # single-axis tables cannot make between "who supplied the push" and
        # "which rollouts it went to".
        xt_source_outcome_stats = (
            SourceOutcomeStats(n_tasks=n_task, device=sign_dev) if xt_on else None
        )
        # The per-token side, on the arm's own switches rather than the sign
        # arm's. Without these the run can say a source raised a task's KL by so
        # many nats and cannot name one token it did it at, which is most of
        # what a write-up needs.
        xt_token_cfg = (xt_cfg.get("token_stats", None) or {}) if xt_cfg_on else {}
        xt_event_cfg = (xt_cfg.get("event_dump", None) or {}) if xt_cfg_on else {}
        xt_token_stats = xt_pair_token_stats = xt_event_stats = None
        xt_role_token_stats = xt_push_token_stats = None
        # The dense tables are ~300 MB of vocabulary-wide buffers and one host
        # read each. Their content is a slow quantity -- which tokens the
        # teachers agree about does not turn over between adjacent steps -- so
        # they run on a stride and are simply not allocated in between. Gated on
        # the STEP COUNT, which every rank shares, never on batch content.
        # turnover then compares the last two COLLECTED steps, which is the
        # comparison it was always making.
        xt_token_every = max(1, int(xt_token_cfg.get("every", 1)))
        xt_token_due = (self._xt_step_index % xt_token_every) == 0
        if xt_on and xt_token_due and (
            xt_token_cfg.get("enable", False) or xt_event_cfg.get("enable", False)
        ):
            _vocab = model_vocab_size(self.actor_module)
            if _vocab is None:
                print(
                    "[cross_teacher] a per-token table was requested but the model does not "
                    "report a vocab_size; running without it",
                    flush=True,
                )
            elif xt_token_cfg.get("enable", False):
                xt_token_stats = TokenStateCounts(
                    vocab_size=_vocab, n_tasks=n_task, device=sign_dev,
                    top_n=int(xt_token_cfg.get("top_n", 64)),
                    # The effect column is nats of weighted KL here, as in the
                    # sign arm's position mode, and is passed in rather than
                    # recomputed so all three tables share one definition.
                    mode="position",
                )
                xt_pair_token_stats = SignPairTokens(
                    n_tasks=n_task, vocab_size=_vocab, device=sign_dev,
                    top_n=int(xt_token_cfg.get("top_n", 64)),
                )
                # The vocabulary cut by what was being WRITTEN. Its own switch
                # because it is the one table here whose cost is not free --
                # n_roles * V is 22 MB at Qwen3's vocabulary -- and because it
                # is useless without the tag ids, which a worker that never set
                # them does not have.
                if sign_role_tags and bool(xt_token_cfg.get("roles", True)):
                    xt_role_token_stats = RoleTokenCounts(
                        vocab_size=_vocab, device=sign_dev,
                        top_n=int(xt_token_cfg.get("role_top_n", 32)),
                    )
                # The tokens whose LOGIT the weight moved, which is a different
                # set from the tokens whose evidence justified it: W is a scalar
                # on the position, so every candidate's push is scaled, including
                # ones no teacher spoke at.
                if bool(xt_token_cfg.get("logit_push", True)):
                    xt_push_token_stats = LogitPushTokens(
                        vocab_size=_vocab, n_tasks=n_task, device=sign_dev,
                        top_n=int(xt_token_cfg.get("push_top_n", 32)),
                    )
        xt_pair_event_stats = None
        if xt_on and xt_token_due and xt_event_cfg.get("enable", False):
            xt_event_stats = SignEventSamples(
                capacity=int(xt_event_cfg.get("per_step", 128)),
                context=int(xt_event_cfg.get("context", 16)),
                device=sign_dev,
            )
            # The pair-stratified companion. A global top-N is dominated by
            # whichever ordered pair is loudest, and the event this arm exists
            # to find -- a source acting where it DISAGREES with the on-task
            # teacher -- is structurally a minority of a minority. Sampling per
            # (dst, src, pair_state) and gathering across ranks is the
            # difference between "we looked and found none" and "we never
            # looked".
            if bool(xt_event_cfg.get("pair_strata", True)):
                xt_pair_event_stats = PairEventSamples(
                    n_tasks=n_task,
                    per_group=int(xt_event_cfg.get("per_group", 4)),
                    context=int(xt_event_cfg.get("context", 16)),
                    device=sign_dev,
                )
        xt_probe_stats = (
            {
                probe_name(a): ScopeTermStats(names=XT_POSITION_TERMS, n_tasks=n_task, device=sign_dev)
                for a in XT_PROBE_ALPHAS
            }
            if xt_on
            else {}
        )
        # The alpha series is the arm's own ablation, run inside the arm. Until
        # now it reported only how BIG the counterfactual weight would be; the
        # state table is what says whether raising alpha moves budget toward the
        # corroborated positions or just scales everything, and that is the
        # question the series exists to answer.
        xt_probe_state_stats = (
            {
                probe_name(a): ScopeTermStats(names=XT_STATE_TERMS, n_tasks=n_task, device=sign_dev)
                for a in XT_PROBE_ALPHAS
            }
            if xt_on
            else {}
        )
        # The channel counterfactuals: not "what if alpha were different" but
        # "what if this channel were not there". Same lagged-normaliser
        # machinery as the alpha probes, because a raw W~ has a different spread
        # from a normalised one and the comparison has to be like for like.
        xt_channel_stats = (
            {
                c: ScopeTermStats(names=XT_POSITION_TERMS, n_tasks=n_task, device=sign_dev)
                for c in XT_CHANNEL_PROBES
            }
            if xt_on
            else {}
        )
        xt_channel_state_stats = (
            {
                c: ScopeTermStats(names=XT_STATE_TERMS, n_tasks=n_task, device=sign_dev)
                for c in XT_CHANNEL_PROBES
            }
            if xt_on
            else {}
        )
        # The alpha the loss uses is the PREVIOUS step's: this step's rows are
        # still being scored, and reading them here would make the objective
        # depend on the order the micro-batches ran in.
        xt_alpha_snapshot = self._xt_alpha if xt_on else None
        xt_rms_snapshot = self._xt_rms_snapshot if xt_on else None
        xt_outside_topk = torch.zeros(2, dtype=torch.float64, device=sign_dev) if xt_on else None
        # Non-finite tallies, accumulated on the device so counting them costs no
        # sync inside the micro-batch loop, and read once at the step boundary.
        # Slot 0 is the weight path, slot 1 the accumulated scale, slot 2 the
        # normaliser, slot 3 the teacher KL the weight multiplies -- which the
        # first three cannot cover, since replacing W by 1 leaves 1 * NaN.
        xt_nonfinite = torch.zeros(4, dtype=torch.float64, device=sign_dev) if xt_on else None
        # The normaliser is a per-STEP mean: taken here from what the last call
        # accumulated, then cleared so this call's rows become the next one's
        # divisor. Reading this step's own would make the objective depend on
        # how the batch was split into micro-batches. Absent -- the first step
        # that has a scale but no mean yet -- means W is exactly 1, not the raw
        # weight and not a within-micro-batch mean.
        xt_mean_snapshot = None
        xt_probe_snapshots = {}
        xt_channel_snapshots = {}
        if xt_on:
            snap = self._xt_mean.snapshot()
            xt_nonfinite[2] += float(snap["nonfinite"])
            xt_mean_snapshot = snap if bool(snap["valid"].any()) else None
            self._xt_mean.reset()
            for _name, _st in self._xt_probe_mean.items():
                _snap = _st.snapshot()
                xt_probe_snapshots[_name] = _snap if bool(_snap["valid"].any()) else None
                _st.reset()
            for _name, _st in self._xt_channel_mean.items():
                _snap = _st.snapshot()
                xt_channel_snapshots[_name] = _snap if bool(_snap["valid"].any()) else None
                _st.reset()

        # Individual candidates, with the tokens around them. Every other table
        # here is an aggregate, and an aggregate cannot be read for a mechanism:
        # "the weighting acts on the same forty tokens" and "it acts on
        # <think>'s connectives" produce the same top-N list.
        event_cfg = (sign_cfg.get("event_dump", None) or {}) if sign_cfg_on else {}
        event_stats = (
            SignEventSamples(
                capacity=int(event_cfg.get("per_step", 128)),
                context=int(event_cfg.get("context", 16)),
                device=sign_dev,
            )
            if (sign_cfg_on and bool(event_cfg.get("enable", False)))
            else None
        )
        # The observer arm: measure everything, change nothing. NOT the same as
        # setting all three weights to 1.0 -- reweight_teacher_logprobs still
        # subtracts a log z that differs from 0 by float error, and its tail
        # clamp can bind at teacher_coverage 1.000, so over 150 steps that is a
        # different trajectory and would not be a control.
        sign_measure_only = bool((sign_cfg or {}).get("measure_only", False)) if sign_cfg_on else False
        # Which vocabulary tokens the weighting acts on. Off unless asked for:
        # it costs a dense (scopes x states x V) accumulator and one 34 MB
        # device-to-host read per call, which is nothing next to the step but is
        # not worth paying on an arm nobody is going to read it for.
        token_stats = None
        pair_token_stats = None
        token_cfg = (sign_cfg.get("token_stats", None) or {}) if sign_enabled else {}
        pair_cfg = (sign_cfg.get("pair_stats", None) or {}) if sign_enabled else {}
        # Both tables are vocabulary-wide, so they share one lookup and one
        # refusal. Resolved once whether or not either is on, because asking the
        # model for its vocab size is free and branching on it twice is how the
        # two tables end up disagreeing about V.
        want_tokens = sign_enabled and bool(token_cfg.get("enable", False))
        want_pair_tokens = sign_enabled and bool(pair_cfg.get("tokens", False)) and n_task >= 2
        if want_tokens or want_pair_tokens:
            vocab = model_vocab_size(self.actor_module)
            if vocab is None:
                print(
                    "[sign_weight] a per-token table was requested but the model does not "
                    "report a vocab_size; running without it",
                    flush=True,
                )
            else:
                if want_tokens:
                    token_stats = TokenStateCounts(
                        vocab_size=vocab,
                        n_tasks=len(task_id_names or []),
                        device=sign_dev,
                        top_n=int(token_cfg.get("top_n", 64)),
                        # The effect column means different things in the two
                        # modes -- a probability displacement against nats of
                        # weighted KL -- so the mode is what decides the formula
                        # AND the name it is reported under.
                        mode="target" if target_mode else "position",
                    )
                if want_pair_tokens:
                    # The vocabulary axis on the pair family: which tokens each
                    # off-task teacher sends into each other task's states. Costs
                    # T*(T-1)*3*V cells -- 55 MB at T=3, i.e. less than the table
                    # above -- because the (on-task sign, src sign) contingency
                    # is collapsed to the three classes that carry a src opinion
                    # and the structurally empty src == dst diagonal is skipped.
                    pair_token_stats = SignPairTokens(
                        n_tasks=n_task,
                        vocab_size=vocab,
                        device=sign_dev,
                        top_n=int(pair_cfg.get("top_n", int(token_cfg.get("top_n", 64)))),
                    )

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                # Total valid response tokens in this mini-batch (for exact token-mean
                # scaling under dynamic bsz). Same mask rule as the per-micro loss below.
                minibatch_valid_tokens = None
                if dynamic_bsz_token_scale:
                    _resp_len = mini_batch["responses"].size(1)
                    _mb_mask = mini_batch["loss_mask"][:, -_resp_len:] if multi_turn else mini_batch["attention_mask"][:, -_resp_len:]
                    minibatch_valid_tokens = float(_mb_mask.sum().clamp(min=1))
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
                # enumerate(), plus the capture window and one named range per
                # micro-batch. Wrapping the iterator leaves the body untouched;
                # a no-op unless ACTOR_NSYS_MICRO or ACTOR_TORCH_MICRO is set.
                for micro_idx, data in actor_capture.iter_micro_batches(micro_batches):
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
                    # Whose top-k defines the KL's support. Teacher-indexed (the
                    # default) hands the forward the ids the teacher already chose.
                    # Student-indexed asks the forward for the student's OWN top-k
                    # and resolves the teacher at those ids afterwards, from cached
                    # hidden states -- see verl/workers/teacher_cache.py for why
                    # that does not force the teacher to run second.
                    fwd_topk_ids = None
                    fwd_topk_k = None
                    if teacher_topk_kl:
                        if student_indexed_topk:
                            fwd_topk_k = int(self.config.get("teacher_kl_topk", 20))
                        else:
                            fwd_topk_ids = data["teacher_topk_ids"]
                    # The sampled-token log-prob is dead weight in pure top-k
                    # distillation: the KL is built from the top-k gather, and every
                    # other consumer here (policy gradient, reference KL, sdl, sdar,
                    # the single-token teacher estimator) is switched off. Computing
                    # it means a log-softmax + gather over the full vocabulary for
                    # every row, whose only surviving use below is reading .device
                    # and .dtype off the result.
                    need_log_prob = not (
                        pg_loss_coef == 0
                        and teacher_topk_kl
                        and use_teacher_kl_loss
                        and not self.config.use_kl_loss
                        and not self.config.get("use_sdl_loss", False)
                        and not self.config.get("use_sdar_loss", False)
                    )
                    with _actor_phase("actor.fwd"):
                        entropy, log_prob, student_topk_out = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            topk_ids=fwd_topk_ids,
                            topk_k=fwd_topk_k,
                            need_log_prob=need_log_prob,
                        )
                    if student_indexed_topk and teacher_topk_kl:
                        # The forward returned the student's own top-k: values (with
                        # gradient) and the ids that chose them, from one logits
                        # tensor -- there is no second student forward here.
                        student_topk_logprobs, student_topk_ids = student_topk_out
                        with _actor_phase("actor.teacher_lookup"):
                            fwd_teacher_topk_logprobs = self._teacher_logprobs_at(
                                cache_ids=data.get("teacher_cache_ids", None),
                                ids=student_topk_ids,
                                input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"],
                            )
                        sign_support_ids = student_topk_ids
                        sign_on_task_logprobs = fwd_teacher_topk_logprobs
                    else:
                        student_topk_logprobs = student_topk_out
                        fwd_teacher_topk_logprobs = None
                        # Teacher-indexed: the support and the on-task teacher's
                        # values at it are both already on the batch, so the sign
                        # weights need no lookup to build -- only the other three
                        # models do. The support is then a function of the frozen
                        # teacher alone and does not drift as the student moves,
                        # which is the whole reason to run this variant.
                        sign_support_ids = data.get("teacher_topk_ids", None) if teacher_topk_kl else None
                        sign_on_task_logprobs = (
                            data.get("teacher_topk_logprobs", None) if teacher_topk_kl else None
                        )

                    # ---- cross-teacher sign agreement --------------------- #
                    sign_position_weight = None
                    sign_target_inputs = None
                    sign_position_inputs = None
                    sign_cand_inputs = None
                    sign_base_logprob = None
                    if sign_enabled:
                        # Refuse rather than skip. This block used to be guarded on
                        # fwd_teacher_topk_logprobs, which is None whenever
                        # student_indexed_topk is off -- so a teacher-indexed arm
                        # with sign_weight.enable=true ran the driver's three extra
                        # frozen forwards (a quarter of the step) and then silently
                        # trained plain OPD, with no sign_weight/* metrics to say so.
                        assert sign_on_task_logprobs is not None and sign_support_ids is not None, (
                            "sign weighting needs a top-k support and the on-task teacher's "
                            "log-probs at it; got neither. It requires teacher_kl_loss_type=topk_kl."
                        )
                        with _actor_phase("actor.sign_weight"):
                            base_logprob, off_logprobs = self._cross_teacher_planes(
                                data, sign_support_ids
                            )
                            # The rewrite decomposition runs further down, past the
                            # end of this block, and needs the base to measure the
                            # teacher's own travel against.
                            sign_base_logprob = base_logprob
                            candidate_weight, sign_state = candidate_weights(
                                sign_on_task_logprobs,
                                off_logprobs,
                                base_logprob,
                                mode=sign_mode,
                                agree_weight=sign_agree,
                                agree_neg_weight=sign_agree_neg,
                                disagree_weight=sign_disagree,
                                deadzone=sign_deadzone,
                            )
                            sign_stats.update_candidates(
                                state=sign_state,
                                on_task_logprob=sign_on_task_logprobs,
                                off_task_logprobs=off_logprobs,
                                base_logprob=base_logprob,
                                response_mask=response_mask,
                                deadzone=sign_deadzone,
                                task_ids=task_ids,
                                off_plane_tasks=data.get("sign_off_tasks", None),
                            )
                            if pair_stats is not None:
                                # The (on-task, off-task) sign contingency table
                                # per ordered pair, from which the pair
                                # association, the gate's leave-one-out and the
                                # blind-spot census are all read.
                                pair_stats.update(
                                    on_task_logprob=sign_on_task_logprobs,
                                    off_task_logprobs=off_logprobs,
                                    base_logprob=base_logprob,
                                    student_logprob=student_topk_logprobs,
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                    off_plane_tasks=data["sign_off_tasks"],
                                    deadzone=sign_deadzone,
                                    student_deadzone=student_resid_deadzone,
                                )
                            if ladder_stats is not None:
                                # How far the student travelled toward the
                                # teachers it is NOT trained on, against where
                                # the base started and where its own teacher
                                # sits. The headline transfer measurement.
                                ladder_stats.update(
                                    student_logprob=student_topk_logprobs,
                                    on_task_logprob=sign_on_task_logprobs,
                                    base_logprob=base_logprob,
                                    off_task_logprobs=off_logprobs,
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                    off_plane_tasks=data["sign_off_tasks"],
                                )
                            # The per-token tables and the position family both
                            # need the KL, which is not built until the loss
                            # below, so what happens here is only to stash the
                            # tensors they read. sign_support_ids is whichever
                            # model nominated the support, i.e. exactly the set
                            # the weights were computed at.
                            sign_cand_inputs = {
                                "support_ids": sign_support_ids,
                                "state": sign_state,
                                "weight": candidate_weight,
                                "on_task_logprob": sign_on_task_logprobs,
                                "base_logprob": base_logprob,
                                "off_task_logprobs": off_logprobs,
                                "off_plane_tasks": data["sign_off_tasks"],
                            }
                            if sign_mode == "target":
                                # Keep the original: the diagnostics below measure
                                # how far the rewrite moved the target, which is a
                                # statement about the pair.
                                sign_target_inputs = (sign_on_task_logprobs, candidate_weight, sign_state)
                                if not sign_measure_only:
                                    # Assigned to fwd_teacher_topk_logprobs (not to
                                    # the teacher-indexed column it may have come
                                    # from) because that is the variable the loss
                                    # reads below, and it takes precedence over
                                    # data[...] there. One path for both supports.
                                    fwd_teacher_topk_logprobs = reweight_teacher_logprobs(
                                        sign_on_task_logprobs, candidate_weight
                                    )
                            else:
                                pos_w = position_weights(candidate_weight, sign_on_task_logprobs)
                                sign_stats.update_position(
                                    position_weight=pos_w,
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                )
                                # By the PREVIOUS call's per-task means: this one's
                                # are not known until every micro-batch has run, and
                                # normalising by a micro-batch's own mean would make
                                # the objective depend on how the batch was split.
                                # Before the first call there is no such mean, and
                                # the micro-batch's own is exactly the wrong
                                # fallback, so the first step runs unnormalised --
                                # sign_weight/*/w_mean_pre_norm records by how much.
                                pos_w_norm = (
                                    normalize_per_task(
                                        pos_w, response_mask, task_ids,
                                        means=self._sign_position_means,
                                    )
                                    if self._sign_position_means is not None
                                    else pos_w
                                )
                                # Computed either way, applied only when the arm
                                # is live: an observer arm still has to report
                                # the weights it declined to spend, and gating
                                # the arithmetic instead of the assignment would
                                # leave measure_only with nothing to measure.
                                sign_position_inputs = (pos_w, pos_w_norm)
                                if not sign_measure_only:
                                    sign_position_weight = pos_w_norm
                    
                    xt_built = None
                    # Per micro-batch, so the readers below cannot see a
                    # previous one's roles when this one produced none.
                    xt_roles_mb = None
                    # Likewise: absent means the policy term was switched off or
                    # ran on a path that did not produce it, and the gradient
                    # comparison is skipped rather than run against a stale one.
                    xt_pg_grad_coef = None
                    # The row-level columns two blocks below both read. Resolved
                    # once, here, so the outcome statistics and the event rows
                    # cannot disagree about what this row scored.
                    _scores = data.get("token_level_scores", None)
                    _row_adv = data.get("adv_row_value", None)
                    _push_for_events = None
                    if xt_enabled:
                        assert sign_support_ids is not None and sign_on_task_logprobs is not None, (
                            "cross_teacher_kl_weight needs the student's top-k support and the "
                            "on-task teacher's log-probs at it"
                        )
                        # The weight is needed on every PPO epoch; the
                        # STATISTICS are collected on the first only. Later
                        # epochs re-visit the same rows against a student that
                        # has already moved, so folding them in would count each
                        # trajectory once per epoch and mix two policies into one
                        # cumulative scale.
                        xt_collect = epoch == 0
                        with _actor_phase("actor.cross_teacher"):
                            base_logprob, off_logprobs = self._cross_teacher_planes(
                                data, sign_support_ids
                            )
                            xt_shifts = compute_raw_policy_shifts(
                                on_task_logprob=sign_on_task_logprobs,
                                off_task_logprobs=off_logprobs,
                                base_logprob=base_logprob,
                            )
                            # This step's contribution to the CUMULATIVE scale.
                            # Read one step later, so nothing here reaches the
                            # weight built below.
                            if xt_collect:
                                self._xt_rms.update(
                                    shifts=xt_shifts,
                                    student_logprob=student_topk_logprobs,
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                    off_plane_tasks=data["sign_off_tasks"],
                                )
                            if xt_rms_snapshot is None:
                                # Step 0: no scale exists, so no weight does
                                # either. Not the raw W~ -- that would be a
                                # silent increase in distillation strength for
                                # as long as the RMS takes to appear.
                                xt_built = None
                            else:
                                xt_built = build_position_weight(
                                    shifts=xt_shifts,
                                    on_task_logprob=sign_on_task_logprobs,
                                    task_ids=task_ids,
                                    off_plane_tasks=data["sign_off_tasks"],
                                    diag=xt_rms_snapshot[0],
                                    diag_valid=xt_rms_snapshot[1],
                                    alpha_table=xt_alpha_snapshot,
                                    normalizer=xt_mean_snapshot,
                                    report_epsilon=xt_report_eps,
                                )
                                xt_nonfinite[0] += xt_built["nonfinite"]
                                if xt_collect:
                                    self._xt_accumulate_reliability(
                                        data=data,
                                        built=xt_built,
                                        student_topk_logprob=student_topk_logprobs,
                                        support_ids=sign_support_ids,
                                        response_mask=response_mask,
                                        task_ids=task_ids,
                                        diag=xt_rms_snapshot,
                                        outside_counter=xt_outside_topk,
                                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    # Under per-task normalisation every term is aggregated by the
                    # same row weights instead of by the token-mean, so the
                    # coefficients between them keep meaning what they say. The
                    # weights already carry the full normalisation, so the two
                    # divisions the sum has to survive are undone once here: FSDP
                    # averages gradients across the DP ranks, and the mini-batch
                    # loss is divided by gradient_accumulation below.
                    task_agg_scale = None
                    if task_loss_weight is not None:
                        task_agg_scale = task_loss_weight * (
                            self.task_dp_world_size * self.gradient_accumulation
                        )

                    def _task_agg(loss_mat):
                        return agg_loss_by_task_weights(
                            loss_mat=loss_mat, loss_mask=response_mask, row_weights=task_agg_scale
                        )

                    if pg_loss_coef != 0:
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]
                        if task_agg_scale is None:
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
                            pg_term = pg_loss
                        else:
                            # Same clipped objective, aggregated by the row weights.
                            # loss_mode is pinned to vanilla for this path (asserted
                            # in check_task_weighting_supported), so the per-token
                            # split of compute_policy_loss is the right one to use.
                            pg_losses, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_per_token(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                            )
                            pg_term = _task_agg(pg_losses)
                            # Reported unweighted so it stays comparable with runs
                            # that do not normalise per task; the weighted number is
                            # deferred separately below.
                            pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            _defer("actor/pg_loss_weighted", pg_term)
                        if xt_grad_stats is not None:
                            # d(pg_losses)/d(log_prob), from the SAME inputs the
                            # loss above was built from rather than from a copy
                            # reconstructed in the diagnostic. Outside the
                            # task-weighting branch, because both paths minimise
                            # the same clipped objective and only differ in how
                            # they aggregate it. Read in the cross-teacher block
                            # below, which is a sibling and cannot see these
                            # names.
                            xt_pg_grad_coef = policy_loss_gradient_coef(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                            ).detach()
                    else:
                        # Pure teacher-KL distillation: no policy-gradient signal.
                        # Take device/dtype from whichever tensor the forward
                        # actually produced -- log_prob is None when it was skipped.
                        old_log_prob = data.get("old_log_probs", None)
                        probe = log_prob if log_prob is not None else student_topk_logprobs
                        zero = torch.zeros((), device=probe.device, dtype=probe.dtype)
                        pg_loss = pg_clipfrac = ppo_kl = pg_clipfrac_lower = zero
                        pg_term = zero

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        entropy_term = entropy_loss if task_agg_scale is None else _task_agg(entropy)

                        # compute policy loss
                        policy_loss = pg_term * pg_loss_coef - entropy_term * entropy_coeff
                    else:
                        policy_loss = pg_term * pg_loss_coef

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
                            # Dense reverse KL over the top-k support (+ tail bucket).
                            # Both sides are full-vocabulary log-softmax values at the
                            # SAME ids, whichever model chose them, so the tail is
                            # 1 - sum in both cases and the formula is unchanged.
                            teacher_topk_lp = (
                                fwd_teacher_topk_logprobs
                                if fwd_teacher_topk_logprobs is not None
                                else data["teacher_topk_logprobs"]
                            )
                            teacher_kld = topk_kl_per_token(
                                student_topk_logprob=student_topk_logprobs,
                                teacher_topk_logprob=teacher_topk_lp,
                            )
                        else:
                            # Single-sampled-token estimator (low_var_kl / kl / mse / abs).
                            teacher_kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=data["teacher_log_probs"],
                                kl_penalty=teacher_kl_loss_type,
                            )
                        if xt_nonfinite is not None:
                            # W = 1 saves nothing when the KL is already NaN --
                            # 1 * NaN is NaN, and so is 0 * NaN at a masked
                            # position, so agg_loss carries it into backward and
                            # the optimizer steps before the step-end check
                            # fires. A NaN on-task teacher log-prob would destroy
                            # the weights and only then be reported.
                            #
                            # Zero everywhere so nothing propagates; count only
                            # inside the mask, because a padded position is not a
                            # teacher failure and killing the step for one would
                            # be a false alarm. The step still dies at the
                            # synchronised point rather than here: raising inside
                            # the micro-batch loop would leave the other ranks in
                            # a collective.
                            finite_kl = torch.isfinite(teacher_kld)
                            xt_nonfinite[3] += (
                                (~finite_kl) & response_mask.to(torch.bool)
                            ).sum().to(torch.float64)
                            teacher_kld = torch.where(
                                finite_kl, teacher_kld, torch.zeros_like(teacher_kld)
                            )
                        # Read BEFORE the position weight multiplies it. The
                        # unweighted KL is what makes w_kl/kl the factor the arm
                        # applied to the total; taking the weighted one would
                        # make that ratio 1 by construction, and the per-token
                        # effect table would be crediting tokens with a cost the
                        # weighting had already inflated.
                        if sign_position_inputs is not None and position_stats is not None:
                            pre_w, applied_w = sign_position_inputs
                            position_stats.update(
                                position_decomposition_terms(
                                    position_weight=pre_w,
                                    applied_weight=applied_w,
                                    candidate_weight=sign_cand_inputs["weight"],
                                    state=sign_cand_inputs["state"],
                                    on_task_logprob=sign_cand_inputs["on_task_logprob"],
                                    teacher_kl=teacher_kld,
                                    weight_range=sign_weight_range,
                                ),
                                response_mask=response_mask,
                                task_ids=task_ids,
                            )
                        if sign_cand_inputs is not None and (
                            token_stats is not None or pair_token_stats is not None
                        ):
                            # The same candidates the stats above summarise, kept
                            # under their vocabulary ids. In position mode the
                            # per-task normaliser is recovered as pre/applied
                            # rather than plumbed separately: it is one number per
                            # task and the two tensors that carry it are already
                            # here.
                            extra = {}
                            if sign_position_inputs is not None:
                                pre_w, applied_w = sign_position_inputs
                                extra = {
                                    "position_scale": pre_w / applied_w.clamp(min=1e-8),
                                    "teacher_kl": teacher_kld,
                                }
                            # Once, for both tables. Two calls would let the two
                            # rankings disagree about what "effect" means.
                            cand_effect = candidate_effect(
                                mode="target" if target_mode else "position",
                                on_task_logprob=sign_cand_inputs["on_task_logprob"],
                                weight=sign_cand_inputs["weight"],
                                **extra,
                            )
                            if token_stats is not None:
                                token_stats.update(
                                    support_ids=sign_cand_inputs["support_ids"],
                                    state=sign_cand_inputs["state"],
                                    weight=sign_cand_inputs["weight"],
                                    on_task_logprob=sign_cand_inputs["on_task_logprob"],
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                    effect=cand_effect,
                                )
                            if event_stats is not None:
                                # The normaliser the effect column was divided
                                # by, so a row can be read without knowing which
                                # mode produced it: Z in target mode, the applied
                                # position weight in position mode. position_weights
                                # rebuilds Z exactly -- it IS sum_v p w + tail.
                                norm = (
                                    sign_position_inputs[1]
                                    if sign_position_inputs is not None
                                    else position_weights(
                                        sign_cand_inputs["weight"],
                                        sign_cand_inputs["on_task_logprob"],
                                    )
                                )
                                responses_mb = data["responses"]
                                # .get, not `in data.batch`: by this point the
                                # micro-batch is a TensorDict (or a plain dict on
                                # the multi-modal path) and has no .batch. The
                                # column is absent on an arm whose reward manager
                                # failed, which is a monitoring failure and must
                                # not take the step down.
                                row_scores = data.get("token_level_scores", None)
                                event_stats.update(
                                    support_ids=sign_cand_inputs["support_ids"],
                                    state=sign_cand_inputs["state"],
                                    weight=sign_cand_inputs["weight"],
                                    effect=cand_effect,
                                    on_task_logprob=sign_cand_inputs["on_task_logprob"],
                                    off_task_logprobs=sign_cand_inputs["off_task_logprobs"],
                                    base_logprob=sign_cand_inputs["base_logprob"],
                                    student_logprob=student_topk_logprobs,
                                    response_mask=response_mask,
                                    responses=responses_mb,
                                    norm=norm,
                                    teacher_kl=teacher_kld,
                                    task_ids=task_ids,
                                    roles=(
                                        token_roles(responses_mb, sign_role_tags)
                                        if sign_role_tags
                                        else None
                                    ),
                                    reward=(row_scores.sum(dim=-1) if row_scores is not None else None),
                                )
                            if pair_token_stats is not None and task_ids is not None:
                                # Which tokens each off-task teacher sends into
                                # THIS task's states. The counts answer it on
                                # their own; the effect column says whether the
                                # weighting acted on what was sent.
                                pair_token_stats.update(
                                    support_ids=sign_cand_inputs["support_ids"],
                                    on_task_logprob=sign_cand_inputs["on_task_logprob"],
                                    off_task_logprobs=sign_cand_inputs["off_task_logprobs"],
                                    base_logprob=sign_cand_inputs["base_logprob"],
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                    off_plane_tasks=sign_cand_inputs["off_plane_tasks"],
                                    deadzone=sign_deadzone,
                                    effect=cand_effect,
                                )
                        if xt_built is not None and xt_position_stats is not None and xt_collect:
                            # Read BEFORE the weight multiplies it, everywhere.
                            # The normaliser composes with itself if it is fed
                            # the weighted KL -- and since the first step runs at
                            # W = 1 it would be right exactly once and drift from
                            # the second, where no metric is looking. kl_scale
                            # would also be 1 by construction rather than by
                            # measurement.
                            self._xt_mean.update(
                                pre_weight=xt_built["pre_weight"], teacher_kl=teacher_kld,
                                response_mask=response_mask, task_ids=task_ids,
                                row_weights=task_loss_weight,
                            )
                            # Computed once and handed to four accumulators:
                            # the task cut, the role cut, the turn cut and the
                            # histogram all decompose the SAME columns, and three
                            # copies of the arithmetic is how three views of one
                            # number come to disagree.
                            xt_pos_cols = xt_position_terms(xt_built, teacher_kld)
                            xt_state_cols = xt_state_shift_terms(xt_built, teacher_kld)
                            xt_position_stats.update(
                                xt_pos_cols, response_mask=response_mask, task_ids=task_ids,
                            )
                            xt_state_stats.update(
                                xt_state_cols, response_mask=response_mask, task_ids=task_ids,
                            )
                            xt_weight_hist.update(
                                weight=xt_built["weight"], teacher_kl=teacher_kld,
                                response_mask=response_mask, task_ids=task_ids,
                            )
                            # Roles are absent when the worker never handed down
                            # the tag ids. Filing everything under "format" would
                            # be a table that looks populated and says nothing, so
                            # the accumulator is simply not fed.
                            xt_roles_mb = (
                                token_roles(data["responses"], sign_role_tags)
                                if sign_role_tags
                                else None
                            )
                            if xt_roles_mb is not None:
                                xt_role_position_stats.update(
                                    xt_pos_cols, response_mask=response_mask, scope_ids=xt_roles_mb,
                                )
                                xt_role_state_stats.update(
                                    xt_state_cols, response_mask=response_mask, scope_ids=xt_roles_mb,
                                )
                            xt_turn_stats.update(
                                xt_pos_cols, response_mask=response_mask,
                                scope_ids=turn_index(response_mask).clamp(max=XT_TURN_BUCKETS - 1),
                            )
                            # Each source's share of W~ - 1, and of the nats that
                            # share went on to move.
                            src_evidence = xt_built["evidence_by_source"].sum(dim=2)
                            xt_src_kl = (
                                teacher_kld / xt_built["mu"].clamp(min=1e-12)
                            ).unsqueeze(-1)
                            xt_pair_stats.update(
                                evidence=src_evidence,
                                shift=src_evidence * xt_src_kl,
                                response_mask=response_mask, task_ids=task_ids,
                                off_plane_tasks=data["sign_off_tasks"],
                                # How much of each source teacher's probability
                                # lands inside the student's top-k at all. Free:
                                # the log-probs are already gathered there.
                                support_mass=off_logprobs.detach().to(
                                    torch.float32
                                ).exp().sum(dim=-2),
                                # The same evidence with alpha divided out. Both
                                # near zero means the source had nothing to say;
                                # this large and the shift near zero means alpha
                                # refused what it did say.
                                activity=xt_built["activity_by_source"].sum(dim=2),
                            )
                            # The same nats, cut by what the source was doing
                            # relative to the on-task teacher.
                            xt_pair_state_stats.update(
                                state=pair_state_index(
                                    hat_on=xt_built["hat_on"], hat_off=xt_built["hat_off"],
                                    deadzone=xt_report_eps,
                                ),
                                evidence=xt_built["evidence_by_source"],
                                shift=xt_built["evidence_by_source"]
                                * xt_src_kl.unsqueeze(-1),
                                response_mask=response_mask, task_ids=task_ids,
                                off_plane_tasks=data["sign_off_tasks"],
                                activity=xt_built["activity_by_source"],
                            )
                            # Per trajectory. The row score is what separates
                            # "spent on rollouts that worked" from "spent on the
                            # ones that did not", which the advantage alone
                            # cannot say -- it is group-relative.
                            xt_outcome_stats.update(
                                weight=xt_built["weight"], teacher_kl=teacher_kld,
                                response_mask=response_mask,
                                advantage=(
                                    _row_adv
                                    if _row_adv is not None
                                    else torch.zeros(
                                        teacher_kld.size(0), device=teacher_kld.device
                                    )
                                ),
                                task_ids=task_ids,
                                reward=(None if _scores is None else _scores.sum(dim=-1)),
                            )
                            # The same rows, cut by SOURCE. The table above sums
                            # the sources out and the evidence table sums the
                            # outcome out, so "Search moved 4% of AlfWorld's
                            # budget" and "the budget went to the rollouts that
                            # scored" cannot be joined without this.
                            xt_source_outcome_stats.update(
                                push_by_source=xt_built["push_by_source"],
                                teacher_kl=teacher_kld, response_mask=response_mask,
                                advantage=(
                                    _row_adv
                                    if _row_adv is not None
                                    else torch.zeros(
                                        teacher_kld.size(0), device=teacher_kld.device
                                    )
                                ),
                                task_ids=task_ids,
                                off_plane_tasks=data["sign_off_tasks"],
                                reward=(None if _scores is None else _scores.sum(dim=-1)),
                            )
                            for name, pre in xt_built["probe_pre_weight"].items():
                                self._xt_probe_mean[name].update(
                                    pre_weight=pre, teacher_kl=teacher_kld,
                                    response_mask=response_mask, task_ids=task_ids,
                                    row_weights=task_loss_weight,
                                )
                                snap = xt_probe_snapshots.get(name, None)
                                probe_mu = _xt_normalizer_mu(pre, snap, task_ids)
                                probe = {
                                    "weight": pre / probe_mu,
                                    "pre_weight": pre,
                                    "mu": probe_mu,
                                    # alpha changes the evidence and nothing
                                    # else: the state labels, the teacher's
                                    # probability and the availability mask are
                                    # the live ones by construction.
                                    "evidence": xt_built["probe_evidence"][name],
                                    "state": xt_built["state"],
                                    "teacher_prob": xt_built["teacher_prob"],
                                    "available": xt_built["available"],
                                    "evidence_shared": xt_built["evidence_shared"],
                                    "evidence_shared_offtask_only": xt_built[
                                        "evidence_shared_offtask_only"
                                    ],
                                }
                                xt_probe_stats[name].update(
                                    xt_position_terms(probe, teacher_kld),
                                    response_mask=response_mask, task_ids=task_ids,
                                )
                                # The counterfactual's own state partition,
                                # built from the probe's OWN mu and evidence so
                                # its columns sum to the probe's (W - 1) D and
                                # not to the shipped arm's. The state labels and
                                # the teacher's probability are shared because
                                # alpha does not move them -- which is what makes
                                # the series an alpha ablation rather than three
                                # unrelated weightings.
                                xt_probe_state_stats[name].update(
                                    xt_state_shift_terms(probe, teacher_kld),
                                    response_mask=response_mask, task_ids=task_ids,
                                )
                            for name, pre in xt_built["channel_pre_weight"].items():
                                self._xt_channel_mean[name].update(
                                    pre_weight=pre, teacher_kl=teacher_kld,
                                    response_mask=response_mask, task_ids=task_ids,
                                    row_weights=task_loss_weight,
                                )
                                snap = xt_channel_snapshots.get(name, None)
                                mu_c = _xt_normalizer_mu(pre, snap, task_ids)
                                chan = {
                                    "weight": pre / mu_c, "pre_weight": pre, "mu": mu_c,
                                    "evidence": xt_built["channel_evidence"][name],
                                    "state": xt_built["state"],
                                    "teacher_prob": xt_built["teacher_prob"],
                                    "available": xt_built["available"],
                                    "evidence_shared": xt_built["evidence_shared"],
                                    "evidence_shared_offtask_only": xt_built[
                                        "evidence_shared_offtask_only"
                                    ],
                                }
                                xt_channel_stats[name].update(
                                    xt_position_terms(chan, teacher_kld),
                                    response_mask=response_mask, task_ids=task_ids,
                                )
                                xt_channel_state_stats[name].update(
                                    xt_state_shift_terms(chan, teacher_kld),
                                    response_mask=response_mask, task_ids=task_ids,
                                )
                        # sign_support_ids / sign_on_task_logprobs directly, NOT
                        # sign_cand_inputs: that dict is the SIGN arm's stash and is
                        # None whenever sign_weight.enable is off. This arm is a
                        # different mechanism and runs with it off, so reading it here
                        # crashed the cross-teacher arm at step 2 -- step 1 survives
                        # only because no RMS exists yet, xt_built is None, and this
                        # block never opens. The two names it needs are the ones
                        # xt_enabled already asserts are present, a hundred lines up.
                        if xt_built is not None and xt_collect:
                            # Built once here whether or not the dense push
                            # table is on: the event rows carry the same two
                            # columns and must not compute them a second way.
                            _push_for_events = (
                                opd_logit_push(
                                    student_logprob=student_topk_logprobs,
                                    teacher_logprob=sign_on_task_logprobs,
                                    teacher_kl=teacher_kld,
                                    coef=float(self.config.get("teacher_kl_loss_coef", 1.0)),
                                )
                                if (xt_push_token_stats is not None or xt_pair_event_stats is not None)
                                else None
                            )
                            self._xt_token_tables(
                                built=xt_built, teacher_kl=teacher_kld, data=data,
                                support_ids=sign_support_ids,
                                student_topk_logprob=student_topk_logprobs,
                                on_task_logprob=sign_on_task_logprobs,
                                response_mask=response_mask, task_ids=task_ids,
                                report_epsilon=xt_report_eps,
                                tables=(
                                    xt_token_stats, xt_pair_token_stats,
                                    xt_event_stats, xt_pair_event_stats,
                                ),
                                roles=sign_role_tags,
                                planes=(base_logprob, off_logprobs),
                                raw_shifts=xt_shifts,
                                alpha_table=xt_alpha_snapshot,
                                row_reward=(None if _scores is None else _scores.sum(dim=-1)),
                                row_advantage=_row_adv,
                                push=_push_for_events,
                            )
                            if xt_role_token_stats is not None and xt_roles_mb is not None:
                                xt_role_token_stats.update(
                                    support_ids=sign_support_ids,
                                    roles=xt_roles_mb,
                                    effect=per_candidate_shift(xt_built, teacher_kld),
                                    response_mask=response_mask,
                                )
                            if xt_push_token_stats is not None:
                                # The OTHER token table. The one above names the
                                # candidates whose evidence justified the weight;
                                # this one names the tokens whose logit the
                                # weight then moved, which is every token in the
                                # support. Conflating them is how "Search
                                # reinforced retrieve" gets written about a
                                # position where what it reinforced is the
                                # suppression of something else.
                                _y1 = data["responses"].unsqueeze(-1)
                                _push = _push_for_events
                                xt_push_token_stats.update(
                                    support_ids=sign_support_ids,
                                    g0=_push["g0"],
                                    weight=xt_built["weight"],
                                    coef_applied_weight=xt_built["weight"],
                                    response_mask=response_mask, task_ids=task_ids,
                                    sampled_onehot=(
                                        sign_support_ids == _y1
                                    ).to(teacher_kld.dtype),
                                    p_student=_push["p_student"],
                                    # The tail bucket has no token to be filed
                                    # under, so it never reaches a row above --
                                    # and without it the token ranking is quoted
                                    # with an unstated denominator.
                                    g0_tail=_push["g0_tail"],
                                )
                            if xt_grad_stats is not None and xt_pg_grad_coef is not None:
                                # Analytic, so the diagnostic cannot perturb the
                                # update it describes. The policy side is the
                                # real clipped objective's per-token derivative,
                                # not A: with 360 rows at a mini-batch of 60,
                                # five of the six mini-batches in an epoch run at
                                # a ratio the optimizer has already moved, and a
                                # bound clip branch has no gradient at all.
                                y1 = data["responses"].unsqueeze(-1)
                                xt_grad_cols = logit_gradient_terms(
                                    student_logprob=student_topk_logprobs,
                                    teacher_logprob=sign_on_task_logprobs,
                                    weight=xt_built["weight"],
                                    teacher_kl=teacher_kld,
                                    pg_grad_coef=xt_pg_grad_coef,
                                    sampled_onehot=(
                                        sign_support_ids == y1
                                    ).to(teacher_kld.dtype),
                                    coef=float(self.config.get("teacher_kl_loss_coef", 1.0)),
                                    pg_coef=float(pg_loss_coef),
                                    # Both terms carry it in the loss, so a pooled
                                    # norm ratio that omits it is the ratio of a
                                    # different objective's gradients.
                                    row_weight=task_loss_weight,
                                )
                                xt_grad_stats.update(
                                    xt_grad_cols, response_mask=response_mask, task_ids=task_ids,
                                )
                                if xt_roles_mb is not None:
                                    xt_role_grad_stats.update(
                                        xt_grad_cols, response_mask=response_mask,
                                        scope_ids=xt_roles_mb,
                                    )
                        if xt_built is not None:
                            # The one line the whole module exists to reach.
                            teacher_kld = teacher_kld * xt_built["weight"].to(teacher_kld.dtype)
                        if sign_position_weight is not None:
                            # position mode: a positive per-token scalar, computed
                            # from frozen models, so it scales the gradient at this
                            # position without moving what the loss is minimised by.
                            # target mode needs nothing here -- it rewrote the
                            # teacher's own values above and reaches the loss
                            # through the line that built teacher_kld.
                            teacher_kld = teacher_kld * sign_position_weight.to(teacher_kld.dtype)
                        if sign_target_inputs is not None:
                            sign_stats.update_target(
                                on_task_logprob=sign_target_inputs[0],
                                candidate_weight=sign_target_inputs[1],
                                response_mask=response_mask,
                                task_ids=task_ids,
                                teacher_kl=teacher_kld,
                            )
                            if rewrite_stats is not None and sign_base_logprob is not None:
                                # The same rewrite, measured at the STUDENT's own
                                # distribution instead of the teacher's. target_kl
                                # says how far the target moved; these say whether
                                # that displacement reached the student, and what
                                # it cost the loss at the states actually visited.
                                #
                                # teacher_kld is passed as it stands, so under
                                # measure_only it is the KL to the UNREWRITTEN
                                # teacher and cf_clamp_resid picks up the whole
                                # rewrite instead of the clamp -- which is the
                                # correct reading for an arm whose loss the rewrite
                                # never entered.
                                rewrite_stats.update(
                                    rewrite_decomposition_terms(
                                        student_logprob=student_topk_logprobs,
                                        on_task_logprob=sign_target_inputs[0],
                                        base_logprob=sign_base_logprob,
                                        candidate_weight=sign_target_inputs[1],
                                        teacher_kl=teacher_kld,
                                        state=sign_target_inputs[2],
                                    ),
                                    response_mask=response_mask,
                                    task_ids=task_ids,
                                )
                        teacher_kl_loss = agg_loss(loss_mat=teacher_kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        teacher_kl_coef = self.config.get("teacher_kl_loss_coef", 1.0)
                        if task_loss_weight is None:
                            policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef
                        else:
                            # Per-task normalised variant: the driver put a weight on
                            # every row such that summing weight * row-KL over the whole
                            # step gives each task an equal share of the loss (see
                            # attach_task_loss_weights). The two divisions this sum must
                            # survive are undone here rather than by special-casing the
                            # shared scaling below: FSDP averages gradients across the DP
                            # ranks, and the mini-batch loss is divided by
                            # gradient_accumulation, but the weights already carry the
                            # full normalisation.
                            row_kl = (teacher_kld * response_mask).sum(-1)
                            weighted_teacher_kl = (row_kl * task_loss_weight).sum()
                            weighted_teacher_kl = weighted_teacher_kl * (
                                self.task_dp_world_size * self.gradient_accumulation
                            )
                            policy_loss = policy_loss + weighted_teacher_kl * teacher_kl_coef
                            _defer("actor/teacher_kl_loss_weighted", weighted_teacher_kl)
                        # Deferred, and appended rather than assigned: assignment kept
                        # only the LAST micro-batch, which after _balance_batch's
                        # reorder is often entirely adjust_batch padding.
                        #
                        # Kept unweighted so it stays comparable with runs that do not
                        # normalise per task.
                        _defer("actor/teacher_kl_loss", teacher_kl_loss)
                        metrics["actor/teacher_kl_coef"] = teacher_kl_coef

                    if self.config.use_dynamic_bsz:
                        if minibatch_valid_tokens is not None:
                            # Exact global token-mean: policy_loss is a token-mean
                            # (denominator = this micro-batch's valid tokens), so
                            # policy_loss * micro_valid_tokens = token-sum, and dividing
                            # by the mini-batch's total valid tokens makes the per-token
                            # weight independent of how tokens were grouped into
                            # micro-batches (unlike the sample-count factor below).
                            micro_valid_tokens = response_mask.sum()
                            loss = policy_loss * (micro_valid_tokens / minibatch_valid_tokens)
                        else:
                            # relative to the dynamic bsz (sample-count reweighting)
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
                            for task, rows in iter_task_row_masks(
                                task_ids, task_id_names, include_absent=sync_free_task_metrics
                            ):
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
                                    if sync_free_task_metrics:
                                        # rows may select nothing here; _defer_task
                                        # carries the presence so an absent task
                                        # contributes 0 rather than NaN.
                                        _defer_task(
                                            f"actor/teacher_kl_loss/{task}",
                                            teacher_kld[rows],
                                            task_response_mask,
                                        )
                                    else:
                                        _defer(
                                            f"actor/teacher_kl_loss/{task}",
                                            agg_loss(loss_mat=teacher_kld[rows], loss_mask=task_response_mask, loss_agg_mode=loss_agg_mode),
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

                if student_indexed_topk or sign_enabled:
                    # Every row the exchange was asked about must have been
                    # answered by exactly one rank. Also when only the sign
                    # weights used it: a teacher-indexed arm reads base and the
                    # off-task teachers out of the same cache. Checked here rather than in the
                    # exchange itself: reading the tally synchronises, and this is
                    # the last point before the weights move, so a row that went
                    # unresolved still cannot reach them.
                    from verl.workers.teacher_cache import assert_rows_were_owned_once

                    assert_rows_were_owned_once()

                with _actor_phase("actor.optim"):
                    # Named separately because it runs in the window between two
                    # micro-batches, which the stall watch would otherwise report
                    # as idle -- a reduce-scatter plus an Adam update over 570M
                    # parameters is real kernels, and calling that idle puts a
                    # noise floor under the stalls being looked for.
                    with actor_capture.span("optim"):
                        grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if sign_stats is not None:
            metrics.update(sign_stats.metrics())
            metrics["sign_weight/mode_is_target"] = float(sign_mode == "target")
            if sign_mode == "position":
                self._refresh_sign_position_means(sign_stats, task_id_names)
        # Reduced before it is rendered: a top-N list has no meaningful per-rank
        # average, so unlike the pooled ratios above it has to be made global
        # first. The ranked rows go on the actor for the worker to decode -- this
        # process has no tokenizer -- while the shape metrics are scalars and
        # ride back with everything else.
        # Reduced before rendering, for the same reason the token table is: these
        # are ratios over sparse per-(dst, src) cells, and a mean of per-rank
        # ratios is not the pooled ratio. Unconditional on batch content -- the
        # collective must not depend on it.
        if rewrite_stats is not None:
            rewrite_stats.all_reduce()
            metrics.update(rewrite_stats.metrics(task_names=task_id_names))
            metrics.update(rewrite_ratio_metrics(rewrite_stats.sums(task_names=task_id_names)))
        if xt_on:
            # Unconditional and in a fixed order: gated on the config alone, so a
            # rank whose micro-batches held no informative group still runs every
            # collective its neighbours do.
            self._xt_rms.all_reduce()
            self._xt_rms_snapshot = self._xt_rms.diagonal()
            self._xt_mean.all_reduce()
            self._xt_adv.all_reduce()
            for _st in self._xt_probe_mean.values():
                _st.all_reduce()
            for _st in self._xt_channel_mean.values():
                _st.all_reduce()
            # Every rank derives the same table from the same reduced moments,
            # so no broadcast is needed to keep them agreeing.
            self._xt_alpha = self._xt_adv.alpha_table()
            # The accumulated scale, checked once it is global.
            snap = self._xt_rms.snapshot()
            xt_nonfinite[1] += float((~torch.isfinite(snap["sigma"])).sum())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                for _t in (xt_outside_topk, xt_nonfinite):
                    torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.SUM)
            # After the collectives, so every rank raises together or none does:
            # an exception thrown before one would leave the others waiting in a
            # collective and hang the job instead of ending it.
            assert_all_finite({
                "weight": xt_nonfinite[0],
                "cumulative_scale": xt_nonfinite[1],
                "normaliser": xt_nonfinite[2],
                "teacher_kl": xt_nonfinite[3],
            })

            xt_position_stats.all_reduce()
            xt_state_stats.all_reduce()
            xt_pair_stats.all_reduce()
            metrics.update(position_weight_metrics(xt_position_stats.sums(task_names=task_id_names)))
            metrics.update(state_shift_metrics(xt_state_stats.sums(task_names=task_id_names)))
            metrics.update(xt_pair_stats.metrics(task_names=task_id_names))
            xt_pair_state_stats.all_reduce()
            metrics.update(xt_pair_state_stats.metrics(task_names=task_id_names))
            xt_outcome_stats.all_reduce()
            metrics.update(xt_outcome_stats.metrics(task_names=task_id_names))
            xt_source_outcome_stats.all_reduce()
            metrics.update(xt_source_outcome_stats.metrics(task_names=task_id_names))
            if xt_grad_stats is not None:
                xt_grad_stats.all_reduce()
                metrics.update(gradient_metrics(xt_grad_stats.sums(task_names=task_id_names)))
            # The two per-position cuts. Rendered by the SAME functions as the
            # task cut -- the accumulators return the same shape of sums -- so a
            # role row and a task row of the same metric are the same
            # arithmetic over different positions, never two definitions.
            xt_weight_hist.all_reduce()
            metrics.update(xt_weight_hist.metrics(task_names=task_id_names))
            xt_turn_stats.all_reduce()
            metrics.update(xt_select_metrics(
                position_weight_metrics(
                    xt_turn_stats.sums(scope_names=XT_TURN_SCOPE_NAMES), prefix="kl_weight/turn"
                ),
                XT_TURN_CUT_SUFFIXES,
            ))
            # Curated, not truncated: the accumulators hold every column, and
            # what a cut publishes is a decision about the dashboard rather than
            # about the measurement. Six roles of all thirty-four columns is
            # three hundred series a step, which is a worse analysis than five
            # that are read.
            for _acc, _render in (
                (xt_role_position_stats, position_weight_metrics),
                (xt_role_state_stats, state_shift_metrics),
                (xt_role_grad_stats, gradient_metrics),
            ):
                _acc.all_reduce()
                metrics.update(xt_select_metrics(
                    _render(_acc.sums(scope_names=XT_ROLE_SCOPE_NAMES), prefix="kl_weight/role"),
                    XT_ROLE_CUT_SUFFIXES,
                ))
            # The rows are stashed and published below, beside the sign arm's:
            # the reports are reset to None there, so assigning them here would
            # be undone a few lines later by a branch that never runs on this arm.
            if xt_token_stats is not None:
                xt_token_stats.all_reduce()
                metrics.update(xt_token_stats.scalar_metrics(task_names=task_id_names))
                # The one reading that needs two steps: n_distinct and
                # top_share are both within-step, and a set of forty tokens
                # replaced wholesale every step reads exactly like a stable
                # forty in both of them.
                _turnover, _token_ids = xt_token_stats.turnover(
                    previous=self._xt_token_prev,
                    task_names=task_id_names,
                    prefix="kl_weight",
                )
                metrics.update(_turnover)
                # Only on a step that had tokens: an empty table must not erase
                # the reference a later step would have compared against.
                if _token_ids:
                    self._xt_token_prev = _token_ids
            if xt_pair_token_stats is not None:
                xt_pair_token_stats.all_reduce()
                metrics.update(xt_pair_token_stats.scalar_metrics(task_names=task_id_names))
                # Per pair, because the pooled turnover cannot tell "a stable
                # set" from "each source contributes a stable set and they are
                # different sets" -- and the second is this arm's whole claim.
                _pt, _pt_ids = xt_pair_token_stats.turnover(
                    previous=self._xt_pair_token_prev,
                    task_names=task_id_names, prefix="kl_weight",
                )
                metrics.update(_pt)
                if _pt_ids:
                    self._xt_pair_token_prev = _pt_ids
            if xt_role_token_stats is not None:
                xt_role_token_stats.all_reduce()
                metrics.update(xt_role_token_stats.scalar_metrics(prefix="kl_weight"))
            if xt_push_token_stats is not None:
                xt_push_token_stats.all_reduce()
                metrics.update(xt_push_token_stats.scalar_metrics(task_names=task_id_names))
            metrics.update(self._xt_rms_metrics(task_id_names))
            metrics.update(self._xt_reliability_metrics(task_id_names))
            def _publish_probe(name, rendered, suffixes):
                for key, value in xt_select_metrics(rendered, suffixes).items():
                    metrics[key.replace("kl_weight/", f"kl_weight/probe/{name}/", 1)] = value

            for _name, _st in xt_probe_stats.items():
                _st.all_reduce()
                _sums = _st.sums(task_names=task_id_names)
                _publish_probe(
                    _name, position_weight_metrics(_sums), ("/w_cv", "/kl_shift_gross_frac")
                )
                # Pooled only. w_pre_mean and shared_share joined the two size
                # readings because an alpha series that reports only how big the
                # weight got cannot say whether the extra size came from the
                # corroboration channel -- which is the channel alpha scales --
                # but that is a statement about the mechanism, not about a task,
                # and three copies of it per probe is three copies.
                if None in _sums:
                    _publish_probe(
                        _name,
                        position_weight_metrics({None: _sums[None]}),
                        ("/w_pre_mean", "/shared_share"),
                    )
            for _name, _st in xt_channel_stats.items():
                _st.all_reduce()
                _sums = _st.sums(task_names=task_id_names)
                # Per task as well as pooled: "is the corroboration channel
                # carrying this" can have different answers on three tasks, and
                # the arm is one mechanism across all three.
                for key, value in xt_select_metrics(
                    position_weight_metrics(_sums),
                    ("/kl_shift_gross_frac", "/kl_scale", "/w_cv"),
                ).items():
                    metrics[key.replace("kl_weight/", f"kl_weight/channel/{_name}/", 1)] = value
            for _name, _st in xt_channel_state_stats.items():
                _st.all_reduce()
                _sums = _st.sums(task_names=task_id_names)
                if None not in _sums:
                    continue
                for key, value in xt_select_metrics(
                    state_shift_metrics({None: _sums[None]}), ("/gross_share",)
                ).items():
                    metrics[key.replace("kl_weight/", f"kl_weight/channel/{_name}/", 1)] = value
            for _name, _st in xt_probe_state_stats.items():
                _st.all_reduce()
                _sums = _st.sums(task_names=task_id_names)
                if None not in _sums:
                    continue
                # gross_share only, pooled only: the per-state net at a
                # counterfactual alpha is a number of nats no arm ever paid,
                # while the share is a composition and reads across the series.
                _publish_probe(
                    _name, state_shift_metrics({None: _sums[None]}), ("/gross_share",)
                )
            total = float(xt_outside_topk[1])
            if total > 0:
                metrics["kl_weight/adv/frac_sampled_outside_topk"] = float(xt_outside_topk[0]) / total
            metrics["kl_weight/cold_start_state"] = float(
                0 if xt_rms_snapshot is None else (1 if xt_mean_snapshot is None else 2)
            )
            # What the worker writes beside the actor checkpoint. Held by
            # reference: the objects keep accumulating and the save reads them
            # at whatever step it happens on.
            self.cross_teacher_state = {
                "rms": self._xt_rms, "mean": self._xt_mean,
                "adv": self._xt_adv, "alpha": self._xt_alpha,
            }
            self.cross_teacher_task_order = list(task_id_names or [])
            # One per call, after everything that read it. A plain python int on
            # every rank, incremented in lockstep, so the dense-token stride
            # cannot make two ranks disagree about whether a table exists.
            self._xt_step_index += 1
        if position_stats is not None:
            # Only the ratios: the raw per-position means would put w_sq and
            # kl_sq in the log, which are not quantities anyone reads. The
            # ratios are formed from sums for the reason rewrite_ratio_metrics
            # is -- a mean of per-position quotients weights a position that
            # carried none of the KL the same as one that carried it all.
            position_stats.all_reduce()
            metrics.update(position_ratio_metrics(position_stats.sums(task_names=task_id_names)))
        if ladder_stats is not None:
            ladder_stats.all_reduce()
            metrics.update(ladder_stats.metrics(task_names=task_id_names))
        if pair_stats is not None:
            pair_stats.all_reduce()
            metrics.update(pair_stats.metrics(task_names=task_id_names))
        if sign_measure_only:
            # So a reader of the logs can tell an observer arm from a live one
            # without going back to the launch command.
            metrics["sign_weight/measure_only"] = 1.0
        self.last_token_report = None
        if token_stats is not None:
            token_stats.all_reduce()
            metrics.update(token_stats.scalar_metrics(task_names=task_id_names))
            self.last_token_report = token_stats.top_tokens(task_names=task_id_names)
        elif xt_token_stats is not None:
            # Same worker path, same dump file: the two arms never run together,
            # so one channel to the tokenizer serves both.
            self.last_token_report = xt_token_stats.top_tokens(task_names=task_id_names)
            if xt_role_token_stats is not None:
                # Appended rather than given a file of its own: the rows carry
                # scope="role:<name>", which is what already tells the two apart,
                # and a third file is a third thing a reader has to join.
                self.last_token_report += xt_role_token_stats.top_tokens()
            if xt_push_token_stats is not None:
                # Same file, and told apart by ranked_by="extra_logit_push" plus
                # the direction_class column no other row carries. A reader
                # filtering on either gets exactly the affected-logit table.
                self.last_token_report += xt_push_token_stats.top_tokens(
                    task_names=task_id_names
                )
        # A second file rather than a discriminator column in the first: the two
        # tables are keyed differently (scope/state against dst/src/class) and
        # merging them would make every row carry the other's empty columns.
        # Its own file: the rows are keyed by (dst, src, pair_state) and carry
        # nineteen float columns no other table has, so merging them into the
        # candidate dump would give every other row nineteen empty ones.
        self.last_pair_event_report = None
        if xt_pair_event_stats is not None:
            self.last_pair_event_report = xt_pair_event_stats.rows(task_names=task_id_names)
        self.last_pair_token_report = None
        if pair_token_stats is not None:
            pair_token_stats.all_reduce()
            metrics.update(pair_token_stats.scalar_metrics(task_names=task_id_names))
            self.last_pair_token_report = pair_token_stats.top_tokens(task_names=task_id_names)
        elif xt_pair_token_stats is not None:
            self.last_pair_token_report = xt_pair_token_stats.top_tokens(task_names=task_id_names)
        # NOT all-reduced, and it cannot be: a sum is a sum but a sample is not,
        # and gathering variable-length selections across ranks to re-sample them
        # would be a collective whose size depends on batch content. What lands
        # on disk is a sample of rank 0's shard -- itself a random shard of the
        # batch, so unbiased, but world_size times smaller than it looks.
        self.last_event_report = None
        if event_stats is not None:
            self.last_event_report = event_stats.rows(task_names=task_id_names)
        elif xt_event_stats is not None:
            self.last_event_report = xt_event_stats.rows(task_names=task_id_names)
        # The one read for everything deferred above. torch.stack forces a single
        # host sync here instead of one per micro-batch. Entries carrying a
        # presence weight are summed and divided by how many micro-batches
        # actually held the task, which is the mean the unweighted path took over
        # exactly those.
        for name, entries in deferred_metrics.items():
            values = torch.stack([value for value, _ in entries])
            if entries[0][1] is None:
                metrics[name] = values.mean().item()
            else:
                present = torch.stack([weight for _, weight in entries])
                metrics[name] = (values.sum() / present.sum().clamp(min=1)).item()
        if _PROFILE_STAGES:
            # One table per update_policy call. The driver's boundary phase
            # ("step") never pops in this process, so the report is asked for
            # explicitly rather than falling out of the phase stack.
            gpu_profiler.report_and_reset(label="update_policy stages (rank 0)")
        return metrics
