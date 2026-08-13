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
"""Cross-teacher sign-agreement weights for multitask on-policy distillation.

The multitask arms distil each sample from the teacher of its own task (the
*on-task* teacher) and never consult the other two (the *off-task* teachers), so
nothing in the loss can carry what task A's RL learned into task B's states.
This module builds the smallest signal that can: whether the off-task teachers
*agree with* or *contradict* the on-task teacher about a given candidate token.

Every teacher is a single-task RL fine-tune of one shared base policy, so what
task m's RL wrote into the model at a state is the policy shift

    delta_m(v) = log pi_m(v | s) - log pi_0(v | s)

on candidate token ``v``. ``delta_m > 0`` means "task m's RL raised this token",
``delta_m < 0`` means it suppressed it, and ``|delta_m| ~ 0`` means task m's RL
never touched it. The sign is what travels across tasks: magnitudes carry each
teacher's own KL coefficient and step count (the search teacher was trained at
``kl_loss_coef=0.001`` against 0.01 for the other two), so they are not on a
common footing, while "raised or lowered" is.

Two ways to spend that signal, selected by ``mode``:

``position``
    One scalar per token position multiplies the whole per-token KL. The weight
    is a positive constant with respect to the student, so the minimiser of the
    weighted loss is still the on-task teacher's distribution: this reallocates
    *how hard* each position is learned and cannot move the target. Safe by
    construction -- no off-task opinion can enter what the student is asked to
    become -- but for the same reason it cannot transfer knowledge, only
    sample efficiency.

``target``
    The on-task teacher's distribution itself is reweighted candidate-wise and
    renormalised, and the student is distilled to *that*. The fixed point moves
    to ``p~(v) ~ w(v) p_teacher(v)``: tokens both sides endorse end up with more
    mass than the on-task teacher alone assigned them. This is the variant that
    can actually inject something, and correspondingly the one that can inject
    something wrong.

Note what ``target`` deliberately does NOT do: multiply the individual terms of
the KL sum. The distillation loss here is a *reverse* KL,
``sum_v p_student(v) (log p_student(v) - log p_teacher(v))``, whose per-candidate
term is a cost the student pays for its own mass at ``v``. Scaling that term up
makes the student put *less* mass on ``v`` -- the exact opposite of "both
teachers like this token, learn it harder". Reweighting the target instead gets
the intended direction and keeps the loss a proper divergence (non-negative,
zero only at the target), which the term-wise product is not.

All quantities here are functions of frozen models (three teachers and the base)
and the sampled trajectory only. Nothing depends on the student, so the weights
are constants for the backward pass and stay fixed across the whole run for a
given state.
"""

from typing import Optional

import torch

__all__ = [
    "SIGN_WEIGHT_KEY",
    "candidate_weights",
    "position_weights",
    "reweight_teacher_logprobs",
    "normalize_per_task",
    "sign_state_metrics",
]

# Column the driver writes and ``DataParallelPPOActor.update_policy`` reads in
# ``position`` mode. Its absence leaves the KL untouched, which is what makes
# ``enable=false`` bit-identical to the plain OPD+GRPO arm.
SIGN_WEIGHT_KEY = "teacher_kl_token_weight"

# State ids produced by :func:`candidate_weights`, reported by
# :func:`sign_state_metrics`. Ordered so the two "nothing happened" states sit
# last: their share is the first number to read when the mechanism looks inert.
STATE_AGREE_POS = 0  # on-task raises, off-task consensus raises
STATE_AGREE_NEG = 1  # on-task lowers, off-task consensus lowers
STATE_CONFLICT_ON_POS = 2  # on-task raises, off-task consensus lowers
STATE_CONFLICT_ON_NEG = 3  # on-task lowers, off-task consensus raises
STATE_NEUTRAL_ON = 4  # on-task teacher inside the deadzone (no opinion)
STATE_NEUTRAL_OFF = 5  # on-task has an opinion, off-task teachers split or silent

STATE_NAMES = {
    STATE_AGREE_POS: "agree_pos",
    STATE_AGREE_NEG: "agree_neg",
    STATE_CONFLICT_ON_POS: "conflict_on_pos",
    STATE_CONFLICT_ON_NEG: "conflict_on_neg",
    STATE_NEUTRAL_ON: "neutral_on_task_silent",
    STATE_NEUTRAL_OFF: "neutral_off_task_split",
}


def _deadzoned_sign(delta: torch.Tensor, deadzone: float) -> torch.Tensor:
    """``sign(delta)`` with everything smaller than ``deadzone`` mapped to 0.

    Without the deadzone this is the mechanism's main failure mode rather than a
    refinement. Most candidate tokens are ones the teacher's RL never moved, so
    their ``delta`` is drift noise around zero; a bare ``sign()`` turns that noise
    into a confident +1/-1, two independent teachers then "agree" on it half the
    time, and the loss gets reweighted on coin flips. ``deadzone`` is in nats, so
    0.1 means "ignore anything that changed the token's probability by less than
    about 10%".
    """
    return torch.where(
        delta > deadzone,
        torch.ones_like(delta),
        torch.where(delta < -deadzone, -torch.ones_like(delta), torch.zeros_like(delta)),
    )


def candidate_weights(
    on_task_logprob: torch.Tensor,
    off_task_logprobs: torch.Tensor,
    base_logprob: torch.Tensor,
    *,
    agree_weight: float,
    disagree_weight: float,
    deadzone: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-candidate weight and state from the teachers' shift signs.

    Args:
        on_task_logprob: (bs, resp, k) on-task teacher log-probs at its own top-k
            ids (full-vocab log-softmax, so ``exp(.).sum(-1) <= 1``).
        off_task_logprobs: (bs, resp, k, n_off) the off-task teachers' log-probs
            at those same ids.
        base_logprob: (bs, resp, k) the base policy's log-probs at those ids.
        agree_weight: weight where on-task and off-task consensus share a sign.
        disagree_weight: weight where their signs oppose.
        deadzone: |delta| below this counts as "no opinion" (nats).

    Returns:
        weight: (bs, resp, k) float32, one of
            ``{agree_weight, disagree_weight, 1.0}``.
        state: (bs, resp, k) int64 state id from ``STATE_*``.

    The off-task teachers only speak with one voice: their consensus sign is
    defined only when *every* off-task teacher is outside the deadzone and they
    all share a sign. A split -- one raises, one lowers, or one is silent --
    yields no weighting at all, on the grounds that "the other tasks disagree
    about this token" is not evidence of shared structure in either direction.
    """
    assert off_task_logprobs.dim() == on_task_logprob.dim() + 1, (
        f"off_task_logprobs must be (bs, resp, k, n_off); got {tuple(off_task_logprobs.shape)}"
    )
    n_off = off_task_logprobs.size(-1)
    assert n_off >= 1, "sign weighting needs at least one off-task teacher"

    sign_on = _deadzoned_sign(on_task_logprob - base_logprob, deadzone)  # (bs, resp, k)
    sign_off = _deadzoned_sign(off_task_logprobs - base_logprob.unsqueeze(-1), deadzone)

    # Unanimity test. Each off-task sign is in {-1, 0, +1}, so the sum reaches
    # +-n_off only when all of them are non-zero and identical; anything else
    # (a split, or any teacher inside the deadzone) falls short in absolute value
    # and is treated as no consensus.
    off_sum = sign_off.sum(dim=-1)  # (bs, resp, k)
    unanimous = off_sum.abs() == n_off
    consensus = torch.where(unanimous, torch.sign(off_sum), torch.zeros_like(off_sum))

    on_silent = sign_on == 0
    off_silent = (~on_silent) & (consensus == 0)
    agree = (~on_silent) & (consensus == sign_on)
    conflict = (~on_silent) & (consensus != 0) & (consensus != sign_on)

    weight = torch.ones_like(on_task_logprob, dtype=torch.float32)
    weight = torch.where(agree, torch.full_like(weight, float(agree_weight)), weight)
    weight = torch.where(conflict, torch.full_like(weight, float(disagree_weight)), weight)

    state = torch.full(on_task_logprob.shape, STATE_NEUTRAL_ON, dtype=torch.long, device=on_task_logprob.device)
    state = torch.where(off_silent, torch.full_like(state, STATE_NEUTRAL_OFF), state)
    state = torch.where(agree & (sign_on > 0), torch.full_like(state, STATE_AGREE_POS), state)
    state = torch.where(agree & (sign_on < 0), torch.full_like(state, STATE_AGREE_NEG), state)
    state = torch.where(conflict & (sign_on > 0), torch.full_like(state, STATE_CONFLICT_ON_POS), state)
    state = torch.where(conflict & (sign_on < 0), torch.full_like(state, STATE_CONFLICT_ON_NEG), state)

    return weight, state


def position_weights(
    candidate_weight: torch.Tensor,
    on_task_logprob: torch.Tensor,
) -> torch.Tensor:
    """Collapse per-candidate weights to one weight per token position.

    The k candidates at a position do not matter equally: the teacher's own
    probability says how much of the KL each of them accounts for. Averaging by
    that mass makes the position weight the expected candidate weight under the
    teacher, which keeps a position dominated by one confident token from being
    swung by nineteen tokens the teacher has all but ruled out.

    The tail (the vocabulary outside the top-k) enters at weight 1.0: no teacher
    reported a shift there, so there is no sign to read, and holding it neutral
    means a position whose top-k covers little mass is modulated correspondingly
    little.

    Args:
        candidate_weight: (bs, resp, k) from :func:`candidate_weights`.
        on_task_logprob: (bs, resp, k) on-task teacher log-probs at its top-k.

    Returns:
        (bs, resp) float32 in ``[min(w), max(w)]``.
    """
    p = on_task_logprob.detach().exp().to(torch.float32)  # (bs, resp, k)
    tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0, max=1.0)  # (bs, resp)
    return (p * candidate_weight).sum(dim=-1) + tail


def normalize_per_task(
    weight: torch.Tensor,
    response_mask: torch.Tensor,
    task_ids: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rescale ``weight`` so its masked mean is 1.0 within every task.

    Without this the mechanism cannot be told apart from a change to
    ``teacher_kl_loss_coef``: if agreement happens to be common, the mean weight
    drifts above 1 and the arm has simply distilled harder. Normalising leaves
    only the *redistribution* across tokens, which is the thing being tested.

    Per task rather than per batch because the arm runs with
    ``normalize_loss_by_task=true``, i.e. each task is supposed to own exactly a
    third of the loss. A single batch-wide mean would let a task whose tokens
    agree more often keep a larger share, quietly undoing that split.

    Rows with no task id (padding, or ``task_ids`` absent) are normalised
    together as one group.
    """
    weight = weight.to(torch.float32)
    mask = response_mask.to(weight.dtype)
    out = weight.clone()

    if task_ids is None:
        groups = [torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)]
    else:
        task_ids = task_ids.reshape(-1)
        groups = [task_ids == t for t in torch.unique(task_ids)]

    for rows in groups:
        if not bool(rows.any()):
            continue
        m = mask[rows]
        denom = m.sum()
        if denom < eps:
            continue
        mean = (weight[rows] * m).sum() / denom
        if mean.abs() < eps:
            continue
        out[rows] = weight[rows] / mean
    return out


def reweight_teacher_logprobs(
    on_task_logprob: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reweight the teacher's top-k distribution and renormalise it (``target`` mode).

    Builds ``p~(v) ~ w(v) p_teacher(v)`` over the top-k, with the tail held at
    weight 1.0 for the same reason as in :func:`position_weights` -- and, here,
    for a second one: the tail is what anchors the *scale* of ``w``. Scaling all
    top-k weights up would otherwise just move mass out of the tail. With the
    neutral weight equal to the tail's 1.0, a position whose candidates are all
    neutral comes back exactly unchanged.

    The result is a full-vocab log-softmax like its input (``exp(.).sum(-1) <= 1``
    with the leftover being the renormalised tail), so it drops straight into
    ``topk_kl_per_token`` in place of the original teacher values and the loss
    stays a proper reverse KL -- to a different, deliberately chosen target.

    Args:
        on_task_logprob: (bs, resp, k) teacher log-probs at its own top-k ids.
        candidate_weight: (bs, resp, k) from :func:`candidate_weights`.

    Returns:
        (bs, resp, k) log-probs of the reweighted target at the same ids.
    """
    p = on_task_logprob.detach().exp().to(torch.float32)
    tail = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)  # (bs, resp, 1)
    z = (p * candidate_weight).sum(dim=-1, keepdim=True) + tail  # (bs, resp, 1)
    z = z.clamp(min=eps)
    return on_task_logprob.detach().to(torch.float32) + torch.log(candidate_weight.clamp(min=eps)) - torch.log(z)


def sign_state_metrics(
    state: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    prefix: str = "sign_weight",
) -> dict:
    """Share of each ``STATE_*`` among the candidates of valid response tokens.

    This is the diagnostic that says whether the mechanism did anything at all:
    if ``neutral_*`` dominates, the weights were 1.0 nearly everywhere and a null
    result says nothing about cross-task transfer, only about the deadzone.
    """
    mask = response_mask.to(torch.bool).unsqueeze(-1).expand_as(state)
    total = int(mask.sum())
    if total == 0:
        return {}
    out = {}
    for sid, name in STATE_NAMES.items():
        out[f"{prefix}/frac_{name}"] = float(((state == sid) & mask).sum()) / total
    return out
