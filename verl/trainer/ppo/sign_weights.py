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

The support every model is scored on is the STUDENT's top-k, not the on-task
teacher's. Nothing here depends on that choice -- ``delta_m(v)`` is a property of
the models at ``v``, whoever nominated ``v`` -- but two things follow from it and
belong in any write-up of a result:

* the candidate set is no longer a function of frozen models alone. The weights
  are still constants for the backward pass, but the same state gets a different
  candidate set as the student drifts, so this is a feedback loop rather than a
  fixed annotation of the state space.
* the off-task teachers are no longer restricted to tokens the on-task teacher
  happened to rank in its own top-k, which is the structural reason the earlier
  teacher-indexed form could not inject anything its on-task teacher had not
  already nominated.

Two ways to spend the signal, selected by ``mode``. THEY DO NOT SHARE A WEIGHT
TABLE, and that is not an oversight:

``position``
    One scalar per token position multiplies the whole per-token KL -- the
    weighting applied to the DIVERGENCE rather than to a probability. The weight
    is a positive constant with respect to the student, so the minimiser of the
    weighted loss is still the on-task teacher's distribution: this reallocates
    *how hard* each position is learned and cannot move the target. Exactly, at
    a position weighted by ``W``, every student logit's gradient is the
    unweighted one times ``W`` -- the direction is untouched and only the step
    length at that position changes. Direction is meaningless here -- a KL term
    has no sign to follow -- so agreement counts the same whether the teachers
    agreed to raise a token or to lower it.

``target``
    The on-task teacher's distribution itself is reweighted candidate-wise and
    renormalised, and the student is distilled to *that*. Here the weight
    multiplies a PROBABILITY, so direction is the whole point: tokens the
    teachers agreed to raise get more mass (``agree_weight`` > 1) and tokens they
    agreed to lower get less (``agree_neg_weight`` < 1). Weighting both
    agreements up -- which is right for ``position`` -- would raise the target
    probability of tokens every teacher agreed to suppress, i.e. undo the very
    edit the agreement is evidence for.

Note what NEITHER mode does: multiply the individual terms of the KL sum. That
third option is the other thing "weight the KL directly" could mean, and it is
the one deliberately not taken. The distillation loss here is a *reverse* KL,
``sum_v p_student(v) (log p_student(v) - log p_teacher(v))``, whose per-candidate
term is a cost the student pays for its own mass at ``v``. Writing
``f_v = log p_student(v) - log p_teacher(v)``, the descent direction on logit
``j`` under term weights ``w`` is

    p_student(j) * ( E_student[w (f + 1)] - w_j (f_j + 1) )

so raising ``w_j`` pushes token ``j`` DOWN whenever ``f_j > -1``, i.e. at every
candidate where the teacher holds less than ``e`` times the student's mass --
the exact opposite of "both teachers like this token, learn it harder". Measured
on a five-candidate example: at ``w_j = 1.5`` on a token the student already
matches the teacher on, the update moves from ``0`` to ``-0.120``; at ``w_j =
2.0``, to ``-0.240``. The sign only flips in the far-gap regime ``f_j < -1``,
which makes the weight's meaning depend on how far the student currently is
rather than on what the teachers said. On top of that the term-wise product is
not a divergence at all -- it can go negative, and its minimum is not the
teacher -- whereas ``position`` scales a whole non-negative KL and ``target``
distils to a genuine distribution. Both of those keep the property that the loss
is zero only at the target; the term-wise product does not.
"""

import math
from typing import Optional

import torch

__all__ = [
    "SIGN_WEIGHT_KEY",
    "SIGN_BASE_TASK",
    "STATE_NAMES",
    "ACTED_STATES",
    "candidate_weights",
    "position_weights",
    "reweight_teacher_logprobs",
    "normalize_per_task",
    "SignWeightStats",
    "TokenStateCounts",
    "POSITION_TERMS",
    "position_decomposition_terms",
    "position_ratio_metrics",
    "candidate_effect",
    "PAIR_TOKEN_CLASSES",
    "SignPairTokens",
    "TAG_ROLES",
    "ROLE_NAMES",
    "token_roles",
    "turn_index",
    "RoleTokenCounts",
    "PAIR_STATES",
    "PAIR_EVENT_INTS",
    "PAIR_EVENT_FLOATS",
    "PairEventSamples",
    "SignEventSamples",
]

# Column the driver writes and the actor reads in ``position`` mode when the
# weights are built driver-side. Its absence leaves the KL untouched, which is
# what makes ``enable=false`` bit-identical to the plain arm.
SIGN_WEIGHT_KEY = "teacher_kl_token_weight"

# The task label the base policy's cached hidden states are filed under. Not a
# real task: it must never collide with one, because the teacher cache picks an
# output projection by this string and the routing picks a teacher by task name.
SIGN_BASE_TASK = "__sign_base__"

# State ids produced by :func:`candidate_weights`, reported by
# :func:`sign_state_metrics`. Ordered so the three "nothing happened" states sit
# last: their share is the first number to read when the mechanism looks inert.
STATE_AGREE_POS = 0  # on-task raises, off-task consensus raises
STATE_AGREE_NEG = 1  # on-task lowers, off-task consensus lowers
STATE_CONFLICT_ON_POS = 2  # on-task raises, off-task consensus lowers
STATE_CONFLICT_ON_NEG = 3  # on-task lowers, off-task consensus raises
STATE_NEUTRAL_ON = 4  # on-task teacher inside the deadzone (no opinion)
STATE_NEUTRAL_OFF_SPLIT = 5  # off-task teachers all spoke, and disagreed with each other
STATE_NEUTRAL_OFF_SILENT = 6  # at least one off-task teacher inside the deadzone

STATE_NAMES = {
    STATE_AGREE_POS: "agree_pos",
    STATE_AGREE_NEG: "agree_neg",
    STATE_CONFLICT_ON_POS: "conflict_on_pos",
    STATE_CONFLICT_ON_NEG: "conflict_on_neg",
    STATE_NEUTRAL_ON: "neutral_on_task_silent",
    STATE_NEUTRAL_OFF_SPLIT: "neutral_off_task_split",
    STATE_NEUTRAL_OFF_SILENT: "neutral_off_task_silent",
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
    mode: str,
    agree_weight: float,
    agree_neg_weight: float,
    disagree_weight: float,
    deadzone: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-candidate weight and state from the teachers' shift signs.

    Args:
        on_task_logprob: (bs, resp, k) on-task teacher log-probs at the support
            (full-vocab log-softmax, so ``exp(.).sum(-1) <= 1``).
        off_task_logprobs: (bs, resp, k, n_off) the off-task teachers' log-probs
            at those same ids.
        base_logprob: (bs, resp, k) the base policy's log-probs at those ids.
        mode: ``"position"`` or ``"target"``; they use different tables, see the
            module docstring.
        agree_weight: agreement to RAISE. In ``position`` mode, agreement in
            either direction.
        agree_neg_weight: agreement to LOWER. ``target`` mode only.
        disagree_weight: on-task and off-task consensus oppose each other.
        deadzone: |delta| below this counts as "no opinion" (nats).

    Returns:
        weight: (bs, resp, k) float32.
        state: (bs, resp, k) int64 state id from ``STATE_*``.

    The off-task teachers only speak with one voice: their consensus sign is
    defined only when *every* off-task teacher is outside the deadzone and they
    all share a sign. A split -- one raises, one lowers -- or a silence yields no
    consensus, on the grounds that "the other tasks disagree about this token" is
    not evidence of shared structure in either direction. The two are separate
    states because they are separate diagnoses: a split says the tasks really do
    pull apart here, a silence says the deadzone swallowed the evidence.
    """
    assert mode in ("position", "target"), f"mode must be 'position' or 'target', got {mode!r}"
    assert off_task_logprobs.dim() == on_task_logprob.dim() + 1, (
        f"off_task_logprobs must be (bs, resp, k, n_off); got {tuple(off_task_logprobs.shape)}"
    )
    n_off = off_task_logprobs.size(-1)
    assert n_off >= 1, "sign weighting needs at least one off-task teacher"
    if mode == "target":
        # A conflict weight in target mode multiplies a probability, so it has to
        # say which way to push, and one number cannot: pulling back toward the
        # objecting teachers means LOWERING a token the on-task teacher raised and
        # RAISING one it lowered. Applying a single factor below 1 to both, which
        # is what ``position`` correctly does, would drive the (-,+) case further
        # in the on-task teacher's direction at exactly the candidates the other
        # tasks objected to. Rather than pick a direction by default, refuse: the
        # arms that act on conflict have to add the knob deliberately.
        assert float(disagree_weight) == 1.0, (
            "target mode has no direction-aware conflict weight yet, so "
            "disagree_weight must be 1.0 (got "
            f"{disagree_weight}); use position mode, or add the second knob first"
        )

    sign_on = _deadzoned_sign(on_task_logprob - base_logprob, deadzone)  # (bs, resp, k)
    sign_off = _deadzoned_sign(off_task_logprobs - base_logprob.unsqueeze(-1), deadzone)

    # Unanimity test. Each off-task sign is in {-1, 0, +1}, so the sum reaches
    # +-n_off only when all of them are non-zero and identical; anything else
    # (a split, or any teacher inside the deadzone) falls short in absolute value
    # and is treated as no consensus.
    off_sum = sign_off.sum(dim=-1)  # (bs, resp, k)
    unanimous = off_sum.abs() == n_off
    consensus = torch.where(unanimous, torch.sign(off_sum), torch.zeros_like(off_sum))
    any_off_silent = (sign_off == 0).any(dim=-1)

    on_silent = sign_on == 0
    no_consensus = (~on_silent) & (consensus == 0)
    off_silent = no_consensus & any_off_silent
    off_split = no_consensus & (~any_off_silent)
    agree = (~on_silent) & (consensus == sign_on)
    conflict = (~on_silent) & (consensus != 0) & (consensus != sign_on)

    agree_pos = agree & (sign_on > 0)
    agree_neg = agree & (sign_on < 0)
    conflict_pos = conflict & (sign_on > 0)
    conflict_neg = conflict & (sign_on < 0)

    weight = torch.ones_like(on_task_logprob, dtype=torch.float32)

    def _set(where, value):
        return torch.where(where, torch.full_like(weight, float(value)), weight)

    if mode == "position":
        # Direction-agnostic: both agreements say "there is shared structure at
        # this candidate", and what the weight buys is learning effort.
        weight = _set(agree, agree_weight)
        weight = _set(conflict, disagree_weight)
    else:
        weight = _set(agree_pos, agree_weight)
        weight = _set(agree_neg, agree_neg_weight)
        weight = _set(conflict, disagree_weight)

    state = torch.full(on_task_logprob.shape, STATE_NEUTRAL_ON, dtype=torch.long, device=on_task_logprob.device)
    state = torch.where(off_split, torch.full_like(state, STATE_NEUTRAL_OFF_SPLIT), state)
    state = torch.where(off_silent, torch.full_like(state, STATE_NEUTRAL_OFF_SILENT), state)
    state = torch.where(agree_pos, torch.full_like(state, STATE_AGREE_POS), state)
    state = torch.where(agree_neg, torch.full_like(state, STATE_AGREE_NEG), state)
    state = torch.where(conflict_pos, torch.full_like(state, STATE_CONFLICT_ON_POS), state)
    state = torch.where(conflict_neg, torch.full_like(state, STATE_CONFLICT_ON_NEG), state)

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

    The tail (the vocabulary outside the support) enters at weight 1.0: no
    teacher reported a shift there, so there is no sign to read, and holding it
    neutral means a position whose support covers little mass is modulated
    correspondingly little.

    Args:
        candidate_weight: (bs, resp, k) from :func:`candidate_weights`.
        on_task_logprob: (bs, resp, k) on-task teacher log-probs at the support.

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
    means: Optional[dict] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rescale ``weight`` so its masked mean is 1.0 within every task.

    Without this the mechanism cannot be told apart from a change to
    ``teacher_kl_loss_coef``: the weight table has no value below 1.0, so the
    mean is above 1 whenever anything agrees and the arm has simply distilled
    harder. Normalising leaves only the *redistribution* across tokens, which is
    the thing being tested.

    Per task rather than per batch because the arm runs with
    ``normalize_loss_by_task=true``, i.e. each task is supposed to own exactly a
    third of the loss. A single batch-wide mean would let a task whose tokens
    agree more often keep a larger share, quietly undoing that split.

    Args:
        means: ``{task_id: mean}`` to divide by, instead of the mean of this
            call's own rows. The weights only exist inside the training forward,
            which sees one micro-batch at a time, and normalising by a
            micro-batch's own mean would make the objective depend on how the
            batch was split. The caller therefore passes the PREVIOUS step's
            per-task means, which are a step-global quantity and move slowly
            (the agreement rate drifted from 0.26 to 0.17 over 150 steps). A task
            absent from the dict is left unscaled.
    """
    weight = weight.to(torch.float32)
    mask = response_mask.to(weight.dtype)
    out = weight.clone()

    if task_ids is None:
        groups = [(None, torch.ones(weight.size(0), dtype=torch.bool, device=weight.device))]
    else:
        task_ids = task_ids.reshape(-1)
        groups = [(int(t), task_ids == t) for t in torch.unique(task_ids)]

    for tid, rows in groups:
        if not bool(rows.any()):
            continue
        if means is not None:
            mean = means.get(tid, None)
            if mean is None or abs(float(mean)) < eps:
                continue
            out[rows] = weight[rows] / float(mean)
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
    """Reweight the teacher's distribution over the support and renormalise it.

    Builds ``p~(v) ~ w(v) p_teacher(v)`` over the support, with the tail held at
    weight 1.0 for the same reason as in :func:`position_weights` -- and, here,
    for a second one: the tail is what anchors the *scale* of ``w``. Scaling all
    the weights up would otherwise just move mass out of the tail. With the
    neutral weight equal to the tail's 1.0, a position whose candidates are all
    neutral comes back exactly unchanged.

    The renormalised tail is not returned and does not need to be: the result is
    a full-vocab log-softmax like its input (``exp(.).sum(-1) <= 1``), so
    ``topk_kl_per_token`` recovers ``tail~ = 1 - sum = tail / Z`` from it, which
    is the renormalised tail exactly. That is also where the tail's own mass
    changes -- weight 1.0 fixes its numerator, not its share.

    Args:
        on_task_logprob: (bs, resp, k) teacher log-probs at the support.
        candidate_weight: (bs, resp, k) from :func:`candidate_weights`.

    Returns:
        (bs, resp, k) log-probs of the reweighted target at the same ids.
    """
    p = on_task_logprob.detach().exp().to(torch.float32)
    tail = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)  # (bs, resp, 1)
    z = (p * candidate_weight).sum(dim=-1, keepdim=True) + tail  # (bs, resp, 1)
    z = z.clamp(min=eps)
    return on_task_logprob.detach().to(torch.float32) + torch.log(candidate_weight.clamp(min=eps)) - torch.log(z)


# --------------------------------------------------------------------------- #
# Diagnostics
#
# Everything below accumulates SUMS across micro-batches and turns them into
# ratios once, at the end of the update. Emitting a ratio per micro-batch and
# letting the metric reducer average those would weight every micro-batch
# equally, and they do not carry equal numbers of valid tokens -- a padded-short
# micro-batch would count as much as a full one. Pooling numerator and
# denominator first is the same number the driver-side version used to report.
# --------------------------------------------------------------------------- #


class SignWeightStats:
    """Pooled counters for one ``update_policy`` call.

    Held by the actor across its micro-batch loop, then rendered once. All
    counters are plain Python floats: they are read on the host anyway, and
    keeping tensors here would pin one micro-batch's memory per entry.
    """

    def __init__(self, task_names: Optional[list] = None):
        self.task_names = list(task_names) if task_names else []
        self.n_cand = {}  # (task_id | None, state) -> count
        self.mass = {}  # (task_id | None, state) -> sum of teacher probability
        self.n_tok = {}  # task_id | None -> valid (row, position) count
        self.mass_tot = {}  # task_id | None -> sum of teacher top-k mass
        # Ordered teacher pairs: how often the off-task teacher m agrees with the
        # on-task teacher t, among candidates where both are outside the deadzone.
        self.pair_both = {}  # (t, m) -> count
        self.pair_same = {}  # (t, m) -> count
        # Per teacher, pooled over both roles it plays in a batch.
        self.shift_abs = {}  # task_id -> sum |delta|
        self.shift_n = {}  # task_id -> count
        self.shift_dead = {}  # task_id -> count inside the deadzone
        # Mode-specific. Each family carries its own token denominator: they are
        # filled by different calls, and a family whose counter was never touched
        # has to report nothing rather than divide by another family's count.
        self.pos_w = {}  # task_id | None -> sum of the position weight, PRE-normalisation
        self.pos_n = {}  # task_id | None -> valid (row, position) count
        self.tgt = {}  # (task_id | None, name) -> sum over valid positions
        self.tgt_n = {}  # task_id | None -> valid (row, position) count

    # -- accumulation ------------------------------------------------------ #

    def _add(self, d, key, value):
        d[key] = d.get(key, 0.0) + float(value)

    def update_candidates(
        self,
        *,
        state: torch.Tensor,
        on_task_logprob: torch.Tensor,
        off_task_logprobs: torch.Tensor,
        base_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        deadzone: float,
        task_ids: Optional[torch.Tensor] = None,
        off_plane_tasks: Optional[torch.Tensor] = None,
    ) -> None:
        """Counts, probability mass, pairwise agreement and per-teacher shifts."""
        valid = response_mask.to(torch.bool)  # (bs, resp)
        cand_valid = valid.unsqueeze(-1).expand_as(state)  # (bs, resp, k)
        p = on_task_logprob.detach().exp().to(torch.float32)

        rows = [(None, torch.ones(state.size(0), dtype=torch.bool, device=state.device))]
        if task_ids is not None:
            ids = task_ids.reshape(-1)
            rows += [(int(t), ids == t) for t in torch.unique(ids) if int(t) >= 0]

        for tid, sel in rows:
            if not bool(sel.any()):
                continue
            m = cand_valid[sel]
            self._add(self.n_tok, tid, int(valid[sel].sum()))
            self._add(self.mass_tot, tid, float((p[sel] * m).sum()))
            s = state[sel]
            for sid in STATE_NAMES:
                hit = (s == sid) & m
                self._add(self.n_cand, (tid, sid), int(hit.sum()))
                self._add(self.mass, (tid, sid), float((p[sel] * hit).sum()))

        # Per-teacher shift magnitudes and deadzone occupancy, pooled over the
        # on-task role (this row's own teacher) and the off-task roles.
        sign_on = _deadzoned_sign(on_task_logprob - base_logprob, deadzone)
        d_off = off_task_logprobs - base_logprob.unsqueeze(-1)  # (bs, resp, k, n_off)
        sign_off = _deadzoned_sign(d_off, deadzone)
        if task_ids is not None:
            ids = task_ids.reshape(-1)
            d_on = (on_task_logprob - base_logprob).abs()
            for t in torch.unique(ids):
                t = int(t)
                if t < 0:
                    continue
                sel = ids == t
                m = cand_valid[sel]
                self._add(self.shift_abs, t, float((d_on[sel] * m).sum()))
                self._add(self.shift_n, t, int(m.sum()))
                self._add(self.shift_dead, t, int(((sign_on[sel] == 0) & m).sum()))

        if off_plane_tasks is not None and task_ids is not None:
            ids = task_ids.reshape(-1)
            n_off = off_task_logprobs.size(-1)
            for plane in range(n_off):
                planes = off_plane_tasks[:, plane].reshape(-1)
                for m_task in torch.unique(planes):
                    m_task = int(m_task)
                    if m_task < 0:
                        continue
                    sel = planes == m_task
                    if not bool(sel.any()):
                        continue
                    mask = cand_valid[sel]
                    self._add(self.shift_abs, m_task, float((d_off[sel, :, :, plane].abs() * mask).sum()))
                    self._add(self.shift_n, m_task, int(mask.sum()))
                    so = sign_off[sel, :, :, plane]
                    self._add(self.shift_dead, m_task, int(((so == 0) & mask).sum()))
                    # Agreement with the on-task teacher of the rows this plane
                    # sits on, split by which task those rows are.
                    on = sign_on[sel]
                    row_tasks = ids[sel]
                    for t in torch.unique(row_tasks):
                        t = int(t)
                        if t < 0:
                            continue
                        r = row_tasks == t
                        both = (on[r] != 0) & (so[r] != 0) & mask[r]
                        self._add(self.pair_both, (t, m_task), int(both.sum()))
                        self._add(self.pair_same, (t, m_task), int(((on[r] == so[r]) & both).sum()))

    def update_position(
        self,
        *,
        position_weight: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """The position weight BEFORE per-task normalisation.

        This is the number that says whether the arm is distinguishable from a
        larger ``teacher_kl_loss_coef``: the table has no entry below 1.0, so a
        mean of 1.15 means the un-normalised arm would have distilled 15% harder
        for reasons that have nothing to do with redistribution.
        """
        valid = response_mask.to(torch.float32)
        w = position_weight.detach().to(torch.float32)
        rows = [(None, torch.ones(w.size(0), dtype=torch.bool, device=w.device))]
        if task_ids is not None:
            ids = task_ids.reshape(-1)
            rows += [(int(t), ids == t) for t in torch.unique(ids) if int(t) >= 0]
        for tid, sel in rows:
            if not bool(sel.any()):
                continue
            self._add(self.pos_w, tid, float((w[sel] * valid[sel]).sum()))
            self._add(self.pos_n, tid, float(valid[sel].sum()))

    def update_target(
        self,
        *,
        on_task_logprob: torch.Tensor,
        candidate_weight: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        teacher_kl: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        """How far the rewrite moved the target, and in what shape.

        Four numbers, because a gain from this arm invites four different
        objections and they have to be separable.

        ``target_kl`` is ``KL(p_teacher || p~)`` over the k+1 categories, in the
        same units as ``actor/teacher_kl_loss``; their ratio says what fraction of
        the student's remaining distance the rewrite is responsible for. It has a
        closed form -- ``log Z - sum_v p(v) log w(v)`` -- which is both cheaper
        than rebuilding ``p~`` and better behaved, since it never takes the log of
        a reweighted probability that underflowed.

        ``inv_z`` is ``1/Z = tail~ / tail``: the factor the tail's share is
        multiplied by. Agreement to raise sits on high-probability candidates and
        agreement to lower on low-probability ones, so the two do NOT cancel in
        ``Z = 1 + sum_v (w-1) p``; the residual is a systematic sharpening of the
        target, and this is it, read directly rather than inferred.

        ``target_entropy_delta`` is the same thing in nats, and the one a
        sign-shuffle control cannot stand in for: shuffling destroys the
        sharpening along with the sign content, so both hypotheses predict the
        same null. A near-zero delta is what licenses reading a shuffle as a test
        of the sign content specifically.

        ``target_tv`` is the total variation distance, which unlike the KL is
        bounded in [0, 1] and so comparable across steps and arms.
        """
        p = on_task_logprob.detach().exp().to(torch.float32)
        w = candidate_weight.detach().to(torch.float32)
        tail = (1.0 - p.sum(dim=-1)).clamp(min=eps, max=1.0)  # (bs, resp)
        z = ((p * w).sum(dim=-1) + tail).clamp(min=eps)  # (bs, resp)

        log_z = z.log()
        kl = log_z - (p * w.clamp(min=eps).log()).sum(dim=-1)
        inv_z = 1.0 / z

        q = p * w / z.unsqueeze(-1)
        q_tail = (tail / z).clamp(min=eps)

        def _entropy(r, t):
            return -(r * r.clamp(min=eps).log()).sum(dim=-1) - t * t.log()

        delta_h = _entropy(q, q_tail) - _entropy(p, tail)
        tv = 0.5 * ((q - p).abs().sum(dim=-1) + (q_tail - tail).abs())

        valid = response_mask.to(torch.float32)
        terms = {"target_kl": kl, "inv_z": inv_z, "target_entropy_delta": delta_h, "target_tv": tv}
        if teacher_kl is not None:
            terms["teacher_kl"] = teacher_kl.detach().to(torch.float32)
        for name, t in terms.items():
            self._add(self.tgt, (None, name), float((t * valid).sum()))
        self._add(self.tgt_n, None, float(valid.sum()))
        if task_ids is not None:
            ids = task_ids.reshape(-1)
            for t_id in torch.unique(ids):
                t_id = int(t_id)
                if t_id < 0:
                    continue
                sel = ids == t_id
                for name, t in terms.items():
                    self._add(self.tgt, (t_id, name), float((t[sel] * valid[sel]).sum()))
                self._add(self.tgt_n, t_id, float(valid[sel].sum()))

    # -- rendering --------------------------------------------------------- #

    def _name(self, tid) -> Optional[str]:
        if tid is None:
            return None
        if 0 <= tid < len(self.task_names):
            return self.task_names[tid]
        return f"task{tid}"

    def metrics(self, prefix: str = "sign_weight") -> dict:
        """Pooled ratios, plus the raw denominators the ratios were taken over.

        The counts are kept because the metric reducer averages across ranks, and
        the mean of per-rank ratios is only the pooled ratio when the ranks hold
        equal numbers of valid tokens. They do, near enough, after
        ``_balance_batch`` -- but the exact number is then still recoverable from
        the two counts, and that is cheaper than being wrong about it later.
        """
        out = {}
        total_cand = {tid: sum(self.n_cand.get((tid, s), 0.0) for s in STATE_NAMES) for tid in self.n_tok}
        for tid, n_tok in self.n_tok.items():
            scope = self._name(tid)
            head = prefix if scope is None else f"{prefix}/{scope}"
            denom = total_cand.get(tid, 0.0)
            mass_denom = self.mass_tot.get(tid, 0.0)
            if denom <= 0:
                continue
            for sid, sname in STATE_NAMES.items():
                out[f"{head}/frac_{sname}"] = self.n_cand.get((tid, sid), 0.0) / denom
                if mass_denom > 0:
                    out[f"{head}/mass_frac_{sname}"] = self.mass.get((tid, sid), 0.0) / mass_denom
            out[f"{head}/n_candidates"] = denom
            out[f"{head}/n_tokens"] = n_tok
            # Mean teacher mass covered by the support, per token. THE ceiling on
            # target mode's leverage: everything the rewrite can move lives inside
            # this, and 1 - it is the tail's share of Z. The mass_frac_* above are
            # shares OF this covered mass, so without it they overstate a
            # mechanism whose support covers little of the teacher.
            if n_tok > 0:
                out[f"{head}/teacher_coverage"] = mass_denom / n_tok

        for tid, n in self.pos_n.items():
            if n <= 0:
                continue
            scope = self._name(tid)
            head = prefix if scope is None else f"{prefix}/{scope}"
            out[f"{head}/w_mean_pre_norm"] = self.pos_w.get(tid, 0.0) / n

        for (t, m), both in self.pair_both.items():
            if both <= 0:
                continue
            on_name, off_name = self._name(t), self._name(m)
            out[f"{prefix}/agree_rate/{off_name}__on__{on_name}"] = self.pair_same.get((t, m), 0.0) / both

        for t, n in self.shift_n.items():
            if n <= 0:
                continue
            name = self._name(t)
            out[f"{prefix}/abs_delta_mean/{name}"] = self.shift_abs.get(t, 0.0) / n
            out[f"{prefix}/deadzone_frac/{name}"] = self.shift_dead.get(t, 0.0) / n

        for (tid, name), total in self.tgt.items():
            n = self.tgt_n.get(tid, 0.0)
            if n <= 0:
                continue
            scope = self._name(tid)
            head = prefix if scope is None else f"{prefix}/{scope}"
            out[f"{head}/{name}"] = total / n
            # Against the loss it sits inside: what fraction of the student's
            # remaining distance to its target the rewrite is responsible for.
            if name == "target_kl":
                tkl = self.tgt.get((tid, "teacher_kl"), 0.0)
                if tkl:
                    out[f"{head}/target_kl_ratio"] = total / tkl
        return out


# --------------------------------------------------------------------------- #
# Per-vocabulary-token diagnostics.
# --------------------------------------------------------------------------- #
# States worth naming a token for. The three neutral ones are "the mechanism did
# nothing here", and they hold the overwhelming majority of candidates, so a
# top-N list over them would just be a list of the most frequent tokens in the
# corpus and would say nothing about the mechanism.
ACTED_STATES = (STATE_AGREE_POS, STATE_AGREE_NEG, STATE_CONFLICT_ON_POS, STATE_CONFLICT_ON_NEG)

EFFECT_KIND = {"target": "dq", "position": "dkl_nats"}


def candidate_effect(
    *,
    mode: str,
    on_task_logprob: torch.Tensor,
    weight: torch.Tensor,
    position_scale: Optional[torch.Tensor] = None,
    teacher_kl: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """(bs, resp, k) post-normalisation effect of the weighting at each candidate.

    One formula, two normalisers, because the two modes act on different objects
    and a single column holding both would be nats and probabilities in one
    series:

    ``target``   ``dq(v) = p_T(v) (w(v)/Z - 1)`` with ``Z = sum_v p_T(v) w(v) +
                 tail``, the per-position renormaliser the rewrite divides by.
                 A change to the distribution the student is distilled toward.
    ``position`` ``dkl(v) = p_T(v) (w(v)/m - 1) * KL``, in nats, with ``m`` the
                 PER-TASK normalising mean -- nothing is renormalised per
                 position in this mode -- times the KL the position weight
                 multiplies. The KL factor is not decoration: a weight of 1.25 at
                 a position whose KL is zero changes the loss by zero, and the
                 same table without it would rank that candidate first.

    In both cases the per-candidate values sum, over the support, to the change
    at that position less the tail's own share, which has no token id to be
    filed under.

    Lives here rather than inside an accumulator because three of them need the
    same number and computing it three times would let them disagree.

    Args:
        position_scale: (bs, resp) ``position`` mode only, REQUIRED there. Not
            rebuildable from a micro-batch: it is the PREVIOUS step's mean over
            the whole batch.
        teacher_kl: (bs, resp) ``position`` mode only, REQUIRED there -- the
            per-token KL BEFORE the weight multiplied it.
    """
    assert mode in EFFECT_KIND, f"mode must be one of {sorted(EFFECT_KIND)}, got {mode!r}"
    p_k = on_task_logprob.detach().to(torch.float32).exp()
    w_k = weight.detach().to(torch.float32)
    if mode == "target":
        tail = (1.0 - p_k.sum(dim=-1)).clamp(min=eps, max=1.0)
        z = ((p_k * w_k).sum(dim=-1) + tail).clamp(min=eps)
        scale = torch.ones_like(z)
    else:
        # Refuse rather than substitute: falling back on the target-mode Z here
        # is exactly the mislabelling this split exists to remove.
        assert position_scale is not None and teacher_kl is not None, (
            "position mode needs position_scale and teacher_kl; the target-mode "
            "normaliser is a different quantity"
        )
        z = position_scale.detach().to(torch.float32).clamp(min=eps)
        scale = teacher_kl.detach().to(torch.float32)
    return (
        p_k.to(torch.float64)
        * (w_k.to(torch.float64) / z.unsqueeze(-1).to(torch.float64) - 1.0)
        * scale.unsqueeze(-1).to(torch.float64)
    )


class TokenStateCounts:
    """Which vocabulary tokens the sign weighting actually acts on.

    Everything else in this module reports the mechanism with the vocabulary
    summed out: ``frac_agree_pos`` says a fifth of candidates were reinforced,
    and cannot say whether that is the same twenty tokens every step or a
    different thousand. Those are different mechanisms with the same summary, and
    the difference decides what a gain would mean -- one is "the tasks share a
    small stable set of moves", the other is "the tasks share a broad statistical
    tendency". This class keeps the token identity.

    Three quantities per token, because the three answer different questions:

    * ``n`` -- how OFTEN the token was a candidate in each state. The count of
      shared-structure events.
    * ``mass`` -- the on-task teacher's probability summed over those events. A
      token can be reinforced constantly and carry no mass, which means the
      reinforcement reached nothing the teacher was actually going to say.
    * ``eff_pos`` / ``eff_neg`` -- the POST-NORMALISATION effect the weighting had
      at that token, accumulated separately by sign. WHAT that effect is depends
      on the mode, because the two modes act on different objects, and reporting
      one formula under both would put nats and probabilities in the same column:

      ``target``   ``dq(v) = p_T(v) * (w(v)/Z - 1)``, the change to the target
                   distribution -- what the student is actually distilled toward.
                   It is not a rescaling of the pre-normalisation ``(w-1) p_T``:
                   since ``dq = (w-1)p_T/Z - p_T (Z-1)/Z``, the second term moves
                   EVERY token in proportion to its own mass, so a
                   high-probability token the weights never touched still loses
                   mass when the rewrite raises others. A quantity built from
                   ``w - 1`` alone reports zero there and misses the whole
                   redistribution.
      ``position`` ``dkl(v) = p_T(v) * (w(v)/m - 1) * KL``, in NATS: the target
                   does not move here, so the only thing a token can be credited
                   with is its share of the cost the weighting added to this
                   position. The same formula with the per-task normalising mean
                   ``m`` where target mode has ``Z``, times the KL the weight
                   multiplies -- because a weight of 1.25 at a position whose KL
                   is zero changes the loss by zero, and a table without the
                   factor would rank that token first. Summed over tokens and
                   positions it is the nats the arm added, up to the tail's own
                   ``tail * (1/m - 1) * KL``, which no token can be charged for.

      Split by sign, and filed per STATE, because both cancellations are real: a
      token reinforced in one context and suppressed in another nets toward zero,
      and so does one whose own weight is 1 while the normaliser varies. Net is
      ``eff_pos + eff_neg``; gross is ``eff_pos - eff_neg``. Reporting only the
      net drops a token that matters in both directions out of the ranking
      entirely.

    Accumulated dense and sync-free. The alternative -- ``torch.unique`` per
    micro-batch, keyed into a Python dict -- reads the device inside the
    micro-batch loop thousands of times a step, which is the run-ahead this
    actor's whole design protects. A dense ``index_add_`` costs one kernel and
    (scopes * states * V) of memory: 4 * 7 * 151,936 is 34 MB at int64, next to
    a teacher output projection of 622 MB.

    Scope 0 is the pooled batch; scope ``1 + t`` is task ``t``. Rows whose task
    is unknown contribute to the pooled scope only.
    """

    # What the effect columns hold, by mode. Carried into the metric names and
    # into every dumped row, so a table can never be read as the other quantity.
    EFFECT_KIND = EFFECT_KIND

    def __init__(self, *, vocab_size: int, n_tasks: int, device, top_n: int = 64, mode: str = "target"):
        assert mode in self.EFFECT_KIND, f"mode must be one of {sorted(self.EFFECT_KIND)}, got {mode!r}"
        self.vocab_size = int(vocab_size)
        self.n_states = len(STATE_NAMES)
        self.n_scopes = 1 + int(n_tasks)
        self.top_n = int(top_n)
        self.mode = mode
        self.effect = self.EFFECT_KIND[mode]
        cells = self.n_scopes * self.n_states * self.vocab_size
        self.n = torch.zeros(cells, dtype=torch.int64, device=device)
        self.mass = torch.zeros(cells, dtype=torch.float32, device=device)
        # Post-normalisation effect, split by sign, on the SAME
        # (scope, state, token) cells as the two above. float64 because these are
        # ~1e-3..1e-6 per event summed over millions of atomic adds into a few
        # cells: in float32 the later increments fall below the running sum's
        # last bit and are silently dropped.
        self.eff_pos = torch.zeros(cells, dtype=torch.float64, device=device)
        self.eff_neg = torch.zeros(cells, dtype=torch.float64, device=device)
        self._cpu_cache = None

    # -- accumulation ------------------------------------------------------ #

    def update(
        self,
        *,
        support_ids: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        on_task_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        position_scale: Optional[torch.Tensor] = None,
        teacher_kl: Optional[torch.Tensor] = None,
        effect: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        """Fold one micro-batch in.

        Args:
            support_ids: (bs, resp, k) vocabulary ids of the support. Whichever
                model nominated them -- the student's top-k or the on-task
                teacher's -- these are the tokens the weights were computed at.
            state: (bs, resp, k) ``STATE_*``.
            weight: (bs, resp, k) the candidate weight that was applied.
            on_task_logprob: (bs, resp, k) on-task teacher log-probs at the support.
            response_mask: (bs, resp).
            task_ids: (bs,) or None.
            position_scale: (bs, resp) ``position`` mode only, REQUIRED there --
                the per-task mean the position weight was divided by. Not
                rebuildable here: it is the PREVIOUS step's mean over the whole
                batch, and a micro-batch cannot see it.
            teacher_kl: (bs, resp) ``position`` mode only, REQUIRED there -- the
                per-token KL BEFORE the weight multiplied it.
            effect: (bs, resp, k) from :func:`candidate_effect`, when the caller
                has already built it for another table. Passed rather than
                recomputed so two rankings cannot disagree about what "effect"
                means; computed here from the two arguments above when absent.

        The validity mask is folded into the VALUES rather than into the indices,
        so every entry is scattered and the invalid ones add zero. Selecting the
        valid entries first would need their count on the host.
        """
        # Any accumulation invalidates a rendering taken before it. Cheap
        # insurance against a caller that reads, then keeps folding.
        self._cpu_cache = None
        ids = support_ids.reshape(-1).to(torch.long)
        st = state.reshape(-1).to(torch.long)
        valid = response_mask.to(torch.bool).unsqueeze(-1).expand_as(state).reshape(-1)

        p_k = on_task_logprob.detach().to(torch.float32).exp()   # (bs, resp, k)
        eff_k = (
            effect.detach().to(torch.float64)
            if effect is not None
            else candidate_effect(
                mode=self.mode,
                on_task_logprob=on_task_logprob,
                weight=weight,
                position_scale=position_scale,
                teacher_kl=teacher_kl,
                eps=eps,
            )
        )

        p = p_k.reshape(-1)
        dq = eff_k.reshape(-1)
        v_f = valid.to(torch.float32)
        v_d = valid.to(torch.float64)
        v_i = valid.to(torch.int64)
        V, S = self.vocab_size, self.n_states

        flat = st * V + ids  # scope 0 starts at offset 0
        self.n.index_add_(0, flat, v_i)
        self.mass.index_add_(0, flat, p * v_f)
        self.eff_pos.index_add_(0, flat, dq.clamp(min=0) * v_d)
        self.eff_neg.index_add_(0, flat, dq.clamp(max=0) * v_d)

        if task_ids is None:
            return
        t = task_ids.reshape(-1, 1, 1).expand_as(state).reshape(-1).to(torch.long)
        known = (t >= 0) & valid
        k_f, k_d = known.to(torch.float32), known.to(torch.float64)
        scope = t.clamp(min=0) + 1
        flat_t = (scope * S + st) * V + ids
        self.n.index_add_(0, flat_t, known.to(torch.int64))
        self.mass.index_add_(0, flat_t, p * k_f)
        self.eff_pos.index_add_(0, flat_t, dq.clamp(min=0) * k_d)
        self.eff_neg.index_add_(0, flat_t, dq.clamp(max=0) * k_d)

    def all_reduce(self) -> None:
        """Sum the three arrays across the DP group.

        Rendered from the reduced arrays on every rank, so the numbers the metric
        reducer averages are already identical and the mean is a no-op -- unlike
        the pooled ratios elsewhere in this module, a top-N list has no meaningful
        per-rank average, so it has to be made global before it is taken.
        """
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for t in (self.n, self.mass, self.eff_pos, self.eff_neg):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    # -- rendering --------------------------------------------------------- #

    def _cpu(self):
        """One device-to-host transfer, shared by both renderings.

        34 MB over PCIe once per ``update_policy`` is the whole host cost of this
        diagnostic; doing it per rendering would double it for nothing.
        """
        if self._cpu_cache is None:
            V, S = self.vocab_size, self.n_states
            self._cpu_cache = (
                self.n.detach().to("cpu").view(self.n_scopes, S, V),
                self.mass.detach().to("cpu").view(self.n_scopes, S, V),
                self.eff_pos.detach().to("cpu").view(self.n_scopes, S, V),
                self.eff_neg.detach().to("cpu").view(self.n_scopes, S, V),
            )
        return self._cpu_cache

    def _scope_name(self, scope: int, task_names) -> Optional[str]:
        if scope == 0:
            return None
        t = scope - 1
        if task_names and 0 <= t < len(task_names):
            return task_names[t]
        return f"task{t}"

    def scalar_metrics(self, task_names=None, prefix: str = "sign_weight") -> dict:
        """The shape of the token distribution, and the size of the real change.

        ``n_distinct`` and ``top_share`` together separate the two mechanisms in
        the class docstring: a small stable set gives few distinct tokens and a
        high top-N share, a broad tendency gives the opposite. Neither is
        derivable from the existing ``frac_*``.

        The effect family reports the POST-NORMALISATION change, positive and
        negative halves separately, because the point of the normaliser is that
        they do not cancel. Per acted state as well as pooled, so a token
        reinforced in one context and suppressed in another is not netted away
        before it is ever seen. Its key carries the mode's own name for the
        quantity -- ``dq`` (probability) or ``dkl_nats`` (nats) -- so the two
        arms never write different things into one series.
        """
        out = {}
        n, _mass, eff_pos, eff_neg = self._cpu()
        e = self.effect
        N = min(self.top_n, self.vocab_size)
        for scope in range(self.n_scopes):
            scope_name = self._scope_name(scope, task_names)
            head = prefix if scope_name is None else f"{prefix}/{scope_name}"
            for sid in ACTED_STATES:
                counts = n[scope, sid]
                total = int(counts.sum())
                if total <= 0:
                    continue
                sname = STATE_NAMES[sid]
                out[f"{head}/token/n_distinct/{sname}"] = float((counts > 0).sum())
                out[f"{head}/token/top{N}_share/{sname}"] = float(torch.topk(counts, N).values.sum()) / total
                # The gross effect of this state, by sign. A state whose two
                # halves are large and nearly equal has been doing work that a
                # net figure reports as nothing.
                out[f"{head}/token/{e}_pos/{sname}"] = float(eff_pos[scope, sid].sum())
                out[f"{head}/token/{e}_neg/{sname}"] = float(eff_neg[scope, sid].sum())

            pos = float(eff_pos[scope].sum())
            neg = float(eff_neg[scope].sum())
            if pos == 0.0 and neg == 0.0:
                continue
            out[f"{head}/token/{e}_pos_sum"] = pos
            out[f"{head}/token/{e}_neg_sum"] = neg
            # Gross, not |net|: the two halves are summed after taking absolute
            # values, so mutual cancellation is excluded rather than hidden.
            out[f"{head}/token/{e}_abs_sum"] = pos - neg
            # Per token, net across states, then ranked by magnitude. The
            # difference between this and the gross sum is exactly how much of
            # the movement cancels within a single token.
            per_tok = (eff_pos[scope] + eff_neg[scope]).sum(0).abs()
            tot = float(per_tok.sum())
            if tot > 0:
                out[f"{head}/token/{e}_abs_top{N}_share"] = float(torch.topk(per_tok, N).values.sum()) / tot
                out[f"{head}/token/{e}_net_over_gross"] = tot / (pos - neg) if (pos - neg) > 0 else 0.0
        return out

    def top_tokens(self, task_names=None) -> list:
        """The ranked lists themselves, as plain rows for the caller to decode.

        Token ids rather than strings: this runs inside the actor, which has no
        tokenizer. The worker that owns one turns them into text.

        Three rankings per scope, because they answer three different questions
        and a token can top one while being absent from the others:

        ``count``    -- which tokens do the teachers keep agreeing about. The
                        shared structure, named.
        ``mass``     -- which of them the on-task teacher was actually going to
                        say. A rare candidate at high probability never reaches
                        the count list and, if its weight is 1, never reaches the
                        effect list either -- yet it is where the objective lives.
        ``abs_effect`` -- which tokens the weighting actually moved, after
                        normalisation. Pooled over states, since a token has one
                        net displacement and no single state. What "moved" means
                        is the mode's: a probability in ``target``, nats of
                        weighted KL in ``position``. Every row carries
                        ``effect_kind`` so the two are never read as one series.
        """
        rows = []
        n, mass, eff_pos, eff_neg = self._cpu()
        kind = self.effect
        N = min(self.top_n, self.vocab_size)
        for scope in range(self.n_scopes):
            scope_name = self._scope_name(scope, task_names) or "__pooled__"
            n_any = n[scope].sum(0)
            mass_any = mass[scope].sum(0)
            eff_net_any = (eff_pos[scope] + eff_neg[scope]).sum(0)
            eff_gross_any = (eff_pos[scope] - eff_neg[scope]).sum(0)

            def _row(ranked_by, sid, rank, tok):
                """One table row. ``sid`` is None for the state-pooled ranking."""
                if sid is None:
                    return {
                        "scope": scope_name, "ranked_by": ranked_by, "state": "__any__",
                        "rank": rank, "token_id": int(tok),
                        "count": int(n_any[tok]), "mass": float(mass_any[tok]),
                        "effect_kind": kind,
                        "effect_net": float(eff_net_any[tok]),
                        "effect_gross": float(eff_gross_any[tok]),
                    }
                return {
                    "scope": scope_name, "ranked_by": ranked_by, "state": STATE_NAMES[sid],
                    "rank": rank, "token_id": int(tok),
                    "count": int(n[scope, sid, tok]), "mass": float(mass[scope, sid, tok]),
                    "effect_kind": kind,
                    "effect_net": float(eff_pos[scope, sid, tok] + eff_neg[scope, sid, tok]),
                    "effect_gross": float(eff_pos[scope, sid, tok] - eff_neg[scope, sid, tok]),
                }

            for sid in ACTED_STATES:
                if int(n[scope, sid].sum()) <= 0:
                    continue
                for ranked_by, series in (("count", n[scope, sid].to(torch.float64)),
                                          ("mass", mass[scope, sid].to(torch.float64))):
                    vals, idx = torch.topk(series, N)
                    for rank, (v, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                        if v <= 0:
                            break
                        rows.append(_row(ranked_by, sid, rank, tok))

            absd = eff_net_any.abs()
            if float(absd.sum()) <= 0:
                continue
            vals, idx = torch.topk(absd, N)
            for rank, (a, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                if a <= 0:
                    break
                rows.append(_row("abs_effect", None, rank, tok))
        return rows


    def turnover(self, previous=None, task_names=None, prefix: str = "sign_weight"):
        """Is it the SAME tokens each step, or a different set with the same shape?

        The class docstring names two mechanisms -- "the tasks share a small
        stable set of moves" and "the tasks share a broad statistical tendency"
        -- and says ``n_distinct`` with ``top_share`` separates them. It does
        not, quite: both are within-step, so a set of forty tokens that is
        completely replaced every step reads exactly like a stable forty. Only a
        comparison ACROSS steps closes that, and this is it.

        Two numbers, because either alone can be gamed by the other:

        ``topN_jaccard``     overlap of the two ranked sets. Set membership only,
                             so a token that stayed in the list but stopped
                             carrying anything still counts.
        ``effect_carryover`` the share of THIS step's gross per-token effect that
                             landed on tokens the PREVIOUS step had ranked. Mass,
                             not membership: high with a low Jaccard means a
                             stable core with a churning tail, which is the
                             common and benign case.

        Returns ``(metrics, state)``. ``state`` is what the next call passes as
        ``previous``; the first call returns no metrics, which is honest -- there
        is nothing to compare against.
        """
        _n, _mass, eff_pos, eff_neg = self._cpu()
        N = min(self.top_n, self.vocab_size)
        out, state = {}, {}
        for scope in range(self.n_scopes):
            scope_name = self._scope_name(scope, task_names)
            key = scope_name or "__pooled__"
            per_tok = (eff_pos[scope] + eff_neg[scope]).sum(0).abs()
            tot = float(per_tok.sum())
            if tot <= 0:
                continue
            vals, idx = torch.topk(per_tok, N)
            cur = [int(t) for v, t in zip(vals.tolist(), idx.tolist()) if v > 0]
            if not cur:
                continue
            state[key] = cur
            prev = (previous or {}).get(key)
            if not prev:
                continue
            head = prefix if scope_name is None else f"{prefix}/{scope_name}"
            a, b = set(cur), set(prev)
            out[f"{head}/token/turnover/top{N}_jaccard"] = len(a & b) / len(a | b)
            keep = torch.zeros(self.vocab_size, dtype=torch.bool)
            keep[torch.tensor(sorted(b), dtype=torch.long)] = True
            out[f"{head}/token/turnover/effect_carryover"] = float(per_tok[keep].sum()) / tot
        return out, state


class ScopeTermStats:
    """Sync-free device accumulator for per-POSITION scalars, keyed by scope.

    The counterpart of :class:`TokenStateCounts` for ``(bs, resp)`` quantities.
    It exists because :meth:`SignWeightStats._add` is ``float(value)`` -- one
    device-to-host read per term per scope per micro-batch, and ``update_target``
    already pays about twenty-eight of them. Nothing new joins that debt: every
    term here lands in one ``index_add_`` and is read once per ``update_policy``.

    Layout is ``(1 + n_tasks, n_terms + 1)``. Scope 0 is the pooled batch, scope
    ``1 + t`` is task ``t``, and the trailing column is the valid-position count
    every ratio divides by -- shared rather than per term, since they all run
    over the same mask.

    float64 throughout. These cells receive millions of atomic adds each and CUDA
    ``index_add_`` does not promise an order, so float32 would make the last bits
    of every reported ratio depend on the scheduler.

    The all_reduce is over WORLD. Under ``ulysses_sequence_parallel_size > 1`` a
    position is held by several ranks, so counts come out multiplied by that
    factor -- benign for every ratio here, since the factor cancels, and wrong
    for any absolute sum. Said here so nobody "fixes" it into a mean.
    """

    def __init__(self, *, names, n_tasks: int, device):
        self.names = list(names)
        self.n_scopes = 1 + int(n_tasks)
        self.buf = torch.zeros(
            (self.n_scopes, len(self.names) + 1), dtype=torch.float64, device=device
        )
        self._cpu_cache = None

    def update(self, terms: dict, response_mask: torch.Tensor, task_ids=None) -> None:
        """Fold one micro-batch in.

        Args:
            terms: ``{name: (bs, resp) tensor}``; every name given at
                construction must be present.
            response_mask: (bs, resp).
            task_ids: (bs,) or None. Rows with a negative id reach the pooled
                scope only -- adjust_batch's padding and any untagged row are
                real positions, but filing them under a task would invent one.
        """
        self._cpu_cache = None
        m = response_mask.to(torch.float64)
        cols = [terms[name].detach().to(torch.float64) * m for name in self.names]
        cols.append(m)
        # (bs, n_terms + 1): each row's contribution, already masked.
        vals = torch.stack(cols, dim=-1).sum(dim=1)
        self.buf[0] += vals.sum(0)
        if task_ids is None:
            return
        t = task_ids.reshape(-1).to(torch.long)
        known = (t >= 0).to(torch.float64).unsqueeze(-1)
        self.buf.index_add_(0, t.clamp(min=0) + 1, vals * known)

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu")
        return self._cpu_cache

    def sums(self, task_names=None) -> dict:
        """``{scope_name_or_None: {term: total}}`` plus ``"n"``, the denominator.

        Handed to callers that need to build a ratio ACROSS terms (``a / b``)
        rather than a mean; taking two separately-rendered means and dividing
        them is the same number only when both ran over the same positions, and
        several of the ratios below deliberately do not.
        """
        buf = self._cpu()
        out = {}
        for scope in range(self.n_scopes):
            n = float(buf[scope, -1])
            if n <= 0:
                continue
            name = None if scope == 0 else (
                task_names[scope - 1] if task_names and scope - 1 < len(task_names) else f"task{scope - 1}"
            )
            out[name] = {t: float(buf[scope, i]) for i, t in enumerate(self.names)}
            out[name]["n"] = n
        return out

    def metrics(self, task_names=None, prefix: str = "sign_weight") -> dict:
        """Every term as a mean over the valid positions it was accumulated on."""
        out = {}
        for scope_name, totals in self.sums(task_names).items():
            head = prefix if scope_name is None else f"{prefix}/{scope_name}"
            n = totals["n"]
            for term in self.names:
                out[f"{head}/{term}"] = totals[term] / n
        return out


def _dist_parts(logprob: torch.Tensor, eps: float = 1e-8):
    """``(p, tail)`` for a full-vocab log-softmax gathered at a top-k support.

    Hoisted out of the KL below because the ladder scores five distributions
    against each other and ``topk_kl_per_token`` would recompute both
    exponentials and both tails on every pairing -- about thirty passes over
    ``(bs, resp, k)`` where sixteen do.
    """
    p = logprob.detach().to(torch.float32).exp()
    tail = (1.0 - p.sum(dim=-1)).clamp(min=eps, max=1.0)
    return p, tail


def _topk_kl_from_parts(lp_a, p_a, tail_a, lp_b, tail_b) -> torch.Tensor:
    """``KL(a || b)`` over the k+1 categories, from parts computed once.

    Numerically identical to ``core_algos.topk_kl_per_token(a, b)``; it is
    written out here only so the shared exponentials can be reused.
    """
    return (p_a * (lp_a.detach().to(torch.float32) - lp_b.detach().to(torch.float32))).sum(-1) + tail_a * (
        tail_a.log() - tail_b.log()
    )


class OffTaskLadderStats:
    """How far the student travelled toward the teachers it is NOT trained on.

    This is the measurement the mechanism was built for and that nothing in the
    arm currently makes. Every other number describes the TEACHERS' agreement
    structure or the SIZE of the rewrite; none says whether the student ended up
    carrying anything from the off-task teachers.

    Three rungs, all against the same off-task teacher ``pi_src`` on the same
    support, at the same positions:

        kl_base = KL(pi_0        || pi_src)   where the student started
        kl_on   = KL(pi_dst      || pi_src)   where on-task distillation alone lands
        kl_stu  = KL(pi_student  || pi_src)   where the student actually is

    and the reading is the ratio

        off_travel = (kl_stu - kl_base) / (kl_on - kl_base)

    which is 0 at initialisation by construction (the student IS the base: the
    lock pins model.path == sign_weight.base_path == Qwen/Qwen3-1.7B) and 1 when
    the student has moved exactly as far toward pi_src as its own teacher sits.

    Why the ratio and not the difference. The three teachers were RL-trained with
    different KL coefficients -- search at 0.001 against 0.01 for the other two,
    which shows up as abs_delta_mean 7.29 / 5.42 / 1.98 nats -- so any absolute
    distance ranks teachers by how far they drifted, and a raw KL to a far-drifted
    teacher is large for reasons that have nothing to do with the student.

    What the ratio buys is exact ANCHORS, not scale invariance. Both endpoints are
    distances to the same pi_src over the same positions and support, so 0 and 1
    mean the same two things whatever pi_src is and however far it drifted. Between
    the anchors the mapping is not scale-free -- KL is not linear, so rescaling
    pi_src does move an intermediate value -- and the number must be read as "where
    between not-moved and teacher-matched", never as a quantity comparable in
    magnitude across pairs.

    What it cannot say on its own: ``kl_stu < kl_on`` is passed for free by a
    student that has simply not converged, and the report puts search's
    distillation at 0.023 nats residual. Read off_travel against
    ``on_travel = 1 - teacher_kl_now / teacher_kl_step0``, never alone.

    Keyed by the ORDERED pair (dst = the row's own task, src = the off-task
    teacher), because "what alfworld picked up from search" and "what search
    picked up from alfworld" are different quantities and the existing pairwise
    agreement matrix is already asymmetric (0.49 vs 0.38 on the shipped run).
    """

    TERMS = ("kl_base", "kl_on", "kl_stu", "cov_off")

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = int(n_tasks)
        self.buf = torch.zeros(
            (self.n_tasks * self.n_tasks, len(self.TERMS) + 1), dtype=torch.float64, device=device
        )
        self._cpu_cache = None

    def update(
        self,
        *,
        student_logprob: torch.Tensor,
        on_task_logprob: torch.Tensor,
        base_logprob: torch.Tensor,
        off_task_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: torch.Tensor,
        off_plane_tasks: torch.Tensor,
    ) -> None:
        """Fold one micro-batch in.

        ``student_logprob`` must be detached by the caller's convention; it is
        detached again here because letting a graph reach a diagnostic buffer
        would hold the whole micro-batch's activations alive.
        """
        self._cpu_cache = None
        m = response_mask.to(torch.float64)
        T = self.n_tasks

        lp_s, lp_on, lp_0 = student_logprob, on_task_logprob, base_logprob
        p_s, tail_s = _dist_parts(lp_s)
        p_on, tail_on = _dist_parts(lp_on)
        p_0, tail_0 = _dist_parts(lp_0)

        dst = task_ids.reshape(-1).to(torch.long)
        for c in range(off_task_logprobs.size(-1)):
            lp_off = off_task_logprobs[..., c]
            _p_off, tail_off = _dist_parts(lp_off)
            terms = [
                _topk_kl_from_parts(lp_0, p_0, tail_0, lp_off, tail_off),
                _topk_kl_from_parts(lp_on, p_on, tail_on, lp_off, tail_off),
                _topk_kl_from_parts(lp_s, p_s, tail_s, lp_off, tail_off),
                1.0 - tail_off,  # how much of pi_src the support covers
            ]
            vals = torch.stack([t.to(torch.float64) * m for t in terms] + [m], dim=-1).sum(dim=1)
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            # sign_off_tasks is legitimately -1 when the driver could not name the
            # plane's task, and task_ids is -1 on padding. Clamp the INDEX, fold
            # the validity into the VALUE -- selecting rows out would need their
            # count on the host.
            ok = ((dst >= 0) & (src >= 0)).to(torch.float64).unsqueeze(-1)
            flat = dst.clamp(min=0) * T + src.clamp(min=0)
            self.buf.index_add_(0, flat, vals * ok)

    def all_reduce(self) -> None:
        """Summed across ranks before any ratio is taken.

        Required, not optional: these cells are sparse per (dst, src) and a mean
        of per-rank ratios is not the pooled ratio when the ranks hold different
        numbers of positions for a pair.
        """
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu").view(self.n_tasks, self.n_tasks, len(self.TERMS) + 1)
        return self._cpu_cache

    def metrics(self, task_names=None, prefix: str = "transfer", min_positions: float = 1.0) -> dict:
        """The three rungs, the travel ratio, and the coverage that bounds them."""
        buf = self._cpu()
        out = {}

        def _name(i):
            if task_names and i < len(task_names):
                return task_names[i]
            return f"task{i}"

        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if dst == src:
                    continue  # the on-task teacher is not an off-task plane
                n = float(buf[dst, src, -1])
                if n < min_positions:
                    continue
                pair = f"{_name(dst)}__vs__{_name(src)}"
                vals = {t: float(buf[dst, src, i]) / n for i, t in enumerate(self.TERMS)}
                for rung in ("base", "on", "stu"):
                    out[f"{prefix}/kl_to_off/{rung}/{pair}"] = vals[f"kl_{rung}"]
                out[f"{prefix}/support_coverage/off/{pair}"] = vals["cov_off"]
                out[f"{prefix}/kl_to_off/n_positions/{pair}"] = n
                span = vals["kl_on"] - vals["kl_base"]
                # A span at or below noise makes the ratio meaningless rather than
                # large: the denominator is "how far on-task distillation moves the
                # model toward this teacher", and where that is zero the question
                # "how far along that has the student come" has no answer.
                if abs(span) > 1e-6:
                    out[f"{prefix}/off_travel/{pair}"] = (vals["kl_stu"] - vals["kl_base"]) / span
                out[f"{prefix}/off_span/{pair}"] = span
        return out


# Terms produced by :func:`rewrite_decomposition_terms`, in the order a
# ScopeTermStats should be constructed with.
REWRITE_TERMS = (
    "cf_cost",
    "control_teacher_kl",
    "rewrite_align",
    "rewrite_span",
    "rewrite_fisher",
    "log_z",
    "log_z_sq",
    "cf_clamp_resid",
    "student_tail_mass",
    # Where the probability the rewrite moves actually goes. Every term below is
    # a POST-normalisation displacement, so together they are what the student is
    # distilled toward -- unlike log_z and the alignment terms, which describe the
    # tilt before Z divides it back out.
    "mass_shift_agree_pos",
    "mass_shift_agree_neg",
    "mass_shift_conflict",
    "mass_shift_neutral",
    "mass_shift_tail",
    "mass_shift_abs",
    "mass_conservation_error",
    "tail_tv_share",
)


def rewrite_decomposition_terms(
    *,
    student_logprob: torch.Tensor,
    on_task_logprob: torch.Tensor,
    base_logprob: torch.Tensor,
    candidate_weight: torch.Tensor,
    teacher_kl: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> dict:
    """What the rewrite cost the student, decomposed exactly.

    ``target_kl`` says how far the rewrite moved the TARGET. It cannot say
    whether that displacement reached the student, because it is a statement
    about ``p`` and ``p~`` only. These terms are the same rewrite measured at the
    student's own distribution, at the states the student actually visited.

    The central quantity is

        cf_cost = log Z - sum_S p_s log w  ==  KL(p_s||p~) - KL(p_s||p)

    and the identity is exact, not an approximation: substituting
    ``log p~(v) = log p(v) + log w(v) - log Z`` into ``KL(p_s||p~)`` leaves
    ``KL(p_s||p) - sum_S p_s log w + log Z``, because ``p_s`` sums to one over the
    k+1 categories. So cf_cost is signed -- POSITIVE means the rewrite moved the
    target away from where the student already was (it made the loss harder),
    NEGATIVE that it moved toward it. Neither sign licenses a story about which
    tokens were "confirmed" or "corrected"; it is a decomposition, not a test.

    ``rewrite_align = target_kl - cf_cost = sum_S (p_s - p) log w`` is how far
    along the rewrite direction the student already sits, relative to the plain
    teacher. ``rewrite_span = sum_S (p - p_0) log w`` is the same projection for
    the teacher's own travel from the base, which is what makes their ratio
    (``rewrite_progress``, formed at render) a dimensionless number: -1 when the
    student sits at the base, 0 when it has matched the teacher's travel along
    log w, positive on overshoot.

    ``rewrite_fisher = Var_{p_s}(log w)`` is the squared Fisher norm of the extra
    gradient this arm adds. It is exactly zero when nothing fires, and -- unlike
    every distance here -- it carries none of the teachers' KL coefficients,
    which differ by 3.7x in measured drift.

    The variance is over the k+1 categories, with the TAIL counted as a category
    at ``log w = 0``: the rewrite does not touch the tail, so ``Z`` does not undo
    a uniform weight and a weight that is constant across the support still tilts
    the support against the tail. Read it as "how much structure the rewrite has
    within this position", not as "how large the rewrite is" -- those come apart
    exactly here, and ``cf_cost`` is the one that answers the second.

    ``control_teacher_kl`` is computed DIRECTLY against the unrewritten teacher
    rather than derived as ``teacher_kl - cf_cost``: only the direct form shares
    the loss's own 1e-8 tail clamp, and the difference between the two is
    precisely what ``cf_clamp_resid`` reports. That residual is the identity
    ``teacher_kl == control_teacher_kl + cf_cost`` measured rather than assumed;
    read it before trusting cf_cost on a task whose teacher_coverage is 1.000,
    where the clamp binds.

    Args:
        student_logprob: (bs, resp, k) the student at the support. Detached here.
        on_task_logprob: (bs, resp, k) the unrewritten on-task teacher.
        base_logprob: (bs, resp, k) the shared base policy.
        candidate_weight: (bs, resp, k) the weight the rewrite applied.
        teacher_kl: (bs, resp) the loss's own per-token KL to the REWRITTEN
            teacher, i.e. what update_target already receives.

    Returns:
        ``{name: (bs, resp) tensor}`` for every name in :data:`REWRITE_TERMS`.
    """
    from verl.trainer.ppo.core_algos import topk_kl_per_token

    lp_s = student_logprob.detach().to(torch.float32)
    p = on_task_logprob.detach().to(torch.float32).exp()
    p_s = lp_s.exp()
    p_0 = base_logprob.detach().to(torch.float32).exp()
    w = candidate_weight.detach().to(torch.float32)

    tail = (1.0 - p.sum(dim=-1)).clamp(min=eps, max=1.0)
    z = ((p * w).sum(dim=-1) + tail).clamp(min=eps)
    log_z = z.log()
    log_w = w.clamp(min=eps).log()

    a_s = (p_s * log_w).sum(dim=-1)
    cf_cost = log_z - a_s
    target_kl = log_z - (p * log_w).sum(dim=-1)
    control_kl = topk_kl_per_token(lp_s, on_task_logprob.detach())

    # ---- where the moved probability lands -------------------------------
    # dq(v) = p(v) (w(v)/Z - 1). NOT a rescaling of (w-1)p: the second term of
    # dq = (w-1)p/Z - p(Z-1)/Z moves every token in proportion to its own mass,
    # so a high-probability token the weights never touched still loses mass when
    # the rewrite raises others. Reporting only (w-1)p misses that redistribution
    # entirely, and it is the part the student is actually trained on.
    dq = p * (w / z.unsqueeze(-1) - 1.0)                      # (bs, resp, k)
    dq_tail = tail * (1.0 / z - 1.0)                          # (bs, resp)

    def _shift(mask):
        return (dq * mask.to(dq.dtype)).sum(dim=-1) if mask is not None else dq.sum(dim=-1)

    if state is None:
        zeros = torch.zeros_like(dq_tail)
        shift_pos = shift_neg = shift_conf = shift_neu = zeros
    else:
        st = state.detach()
        shift_pos = _shift(st == STATE_AGREE_POS)
        shift_neg = _shift(st == STATE_AGREE_NEG)
        shift_conf = _shift((st == STATE_CONFLICT_ON_POS) | (st == STATE_CONFLICT_ON_NEG))
        # Everything the weight table left at 1. Non-zero anyway, and that is the
        # whole point: these tokens move only because Z moved.
        shift_neu = _shift(
            (st == STATE_NEUTRAL_ON) | (st == STATE_NEUTRAL_OFF_SPLIT) | (st == STATE_NEUTRAL_OFF_SILENT)
        )

    abs_shift = dq.abs().sum(dim=-1) + dq_tail.abs()

    return {
        "cf_cost": cf_cost,
        "control_teacher_kl": control_kl,
        "rewrite_align": target_kl - cf_cost,
        "rewrite_span": ((p - p_0) * log_w).sum(dim=-1),
        "rewrite_fisher": (p_s * log_w.square()).sum(dim=-1) - a_s.square(),
        "log_z": log_z,
        "log_z_sq": log_z.square(),
        "cf_clamp_resid": teacher_kl.detach().to(torch.float32) - control_kl - cf_cost,
        # Not bounded by teacher_coverage, which is the ON-TASK TEACHER's mass on
        # the support. The student's own leftover is a different number and the
        # one that says how much of its distribution these terms never saw.
        "student_tail_mass": (1.0 - p_s.sum(dim=-1)).clamp(min=0.0, max=1.0),
        "mass_shift_agree_pos": shift_pos,
        "mass_shift_agree_neg": shift_neg,
        "mass_shift_conflict": shift_conf,
        "mass_shift_neutral": shift_neu,
        "mass_shift_tail": dq_tail,
        # Gross movement, the denominator the shares above are shares of. Twice
        # the total variation between p and the rewritten target.
        "mass_shift_abs": abs_shift,
        # Identically zero: dq sums to (sum_S p w + tail)/Z - 1 = 0 over the k+1
        # categories. Reported so the wiring is checked by the run rather than by
        # this docstring -- a non-zero value means the tail clamp bound, and the
        # shares above are then shares of a quantity that does not conserve.
        "mass_conservation_error": dq.sum(dim=-1) + dq_tail,
        # How much of the movement is the tail bucket absorbing or releasing. The
        # tail is one lumped category, so a large share here says the rewrite is
        # mostly trading against tokens the support never names.
        "tail_tv_share": dq_tail.abs() / abs_shift.clamp(min=eps),
    }


def rewrite_ratio_metrics(sums: dict, prefix: str = "sign_weight") -> dict:
    """Ratios that must be formed from SUMS, not from means of ratios.

    ``rewrite_progress`` and ``cf_cost_ratio`` are quotients of two totals over
    the same positions. Averaging a per-position quotient instead would weight
    every position equally regardless of how much of the rewrite it carried, and
    would divide by zero at the many positions where nothing fired.

    Args:
        sums: the mapping :meth:`ScopeTermStats.sums` returns.
    """
    out = {}
    for scope_name, tot in sums.items():
        head = prefix if scope_name is None else f"{prefix}/{scope_name}"
        span = tot.get("rewrite_span", 0.0)
        if abs(span) > 1e-12:
            out[f"{head}/rewrite_progress"] = tot["rewrite_align"] / span
        # Unlike target_kl_ratio this one can be negative: cf_cost is signed.
        tkl = tot.get("control_teacher_kl", 0.0)
        if abs(tkl) > 1e-12:
            out[f"{head}/cf_cost_ratio"] = tot["cf_cost"] / tkl
        n = tot["n"]
        mean_lz = tot["log_z"] / n
        out[f"{head}/log_z_mean"] = mean_lz
        # E[Z] and E[1/Z] are different functionals of the same distribution and
        # the arm has only ever reported the second. The variance is what says
        # how far apart they are allowed to be.
        out[f"{head}/log_z_var"] = max(tot["log_z_sq"] / n - mean_lz * mean_lz, 0.0)
    return out


# --------------------------------------------------------------------------- #
# position mode: the weight that multiplies the KL itself
# --------------------------------------------------------------------------- #
# The four disjoint groups the position weight decomposes over. ``w - 1`` is
# exactly ``sum_v p(v) (w(v) - 1)`` -- the tail contributes nothing because its
# weight is 1 -- so grouping the candidates partitions the whole departure from
# neutrality with no residual. Conflict and the three neutral states are pooled
# because position mode gives every member of each the same weight, so splitting
# them further would produce columns that are equal by construction.
POSITION_STATE_GROUPS = (
    ("agree_pos", (STATE_AGREE_POS,)),
    ("agree_neg", (STATE_AGREE_NEG,)),
    ("conflict", (STATE_CONFLICT_ON_POS, STATE_CONFLICT_ON_NEG)),
    ("neutral", (STATE_NEUTRAL_ON, STATE_NEUTRAL_OFF_SPLIT, STATE_NEUTRAL_OFF_SILENT)),
)

# Quantiles of the weight TABLE's range, not of the observed weights. A fixed
# cut is what makes the band comparable across steps: a moving quantile would
# report the same share at every step by construction, which is the one thing
# these are meant to detect.
POSITION_BANDS = (0.25, 0.50, 0.75)

POSITION_TERMS = (
    "w_pre",
    "w_pre_sq",
    "w",
    "w_sq",
    "kl",
    "kl_sq",
    "w_kl",
    "kl_shift_abs",
) + tuple(f"w_from_{name}" for name, _ in POSITION_STATE_GROUPS) + tuple(
    f"band{int(q * 100):02d}_{suffix}" for q in POSITION_BANDS for suffix in ("n", "kl")
)


def position_decomposition_terms(
    *,
    position_weight: torch.Tensor,
    applied_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
    state: torch.Tensor,
    on_task_logprob: torch.Tensor,
    teacher_kl: torch.Tensor,
    weight_range: tuple,
) -> dict:
    """Per-position terms for the direct-KL (``position``) arm.

    ``target`` mode has :func:`rewrite_decomposition_terms` because its weight
    moves a distribution and the question is where the probability went. Here the
    weight moves nothing: it multiplies a scalar cost, so the question is a
    different one -- how many nats the weighting added or removed, and whether it
    spent them where the student was already wrong or where it was already right.
    Nothing in the ``target`` family answers that, and the position arm has until
    now reported a single number for the whole mechanism
    (``w_mean_pre_norm``), which cannot tell a redistribution from a coefficient
    change.

    Every term is ``(bs, resp)`` and goes straight into :class:`ScopeTermStats`.

    Args:
        position_weight: (bs, resp) BEFORE per-task normalisation, i.e. the raw
            ``sum_v p(v) w(v) + tail``. On a fixed scale set by the weight table,
            which is what makes the bands below comparable across steps.
        applied_weight: (bs, resp) AFTER normalisation -- the number that actually
            multiplied the KL. Pass the same tensor twice on a step that ran
            unnormalised, and pass what WOULD have been applied under
            ``measure_only``: an observer arm still has to report the weights it
            declined to use.
        candidate_weight: (bs, resp, k).
        state: (bs, resp, k) ``STATE_*``.
        on_task_logprob: (bs, resp, k) on-task teacher log-probs at the support.
        teacher_kl: (bs, resp) the per-token KL BEFORE the weight multiplies it.
            The unweighted one on purpose: ``w_kl / kl`` is then the factor the
            arm applied to the total, and taking the already-weighted KL would
            make that ratio 1 by construction.
        weight_range: ``(lo, hi)`` of the weight table, used only to place a
            position inside the band it belongs to.
    """
    p = on_task_logprob.detach().to(torch.float32).exp()
    w_k = candidate_weight.detach().to(torch.float32)
    # Sums over v to position_weight - 1 exactly: the tail's weight is 1, so it
    # drops out of the excess rather than being neglected.
    excess = p * (w_k - 1.0)

    kl = teacher_kl.detach().to(torch.float32)
    w = applied_weight.detach().to(torch.float32)
    wp = position_weight.detach().to(torch.float32)

    terms = {
        "w_pre": wp,
        "w_pre_sq": wp * wp,
        "w": w,
        "w_sq": w * w,
        "kl": kl,
        "kl_sq": kl * kl,
        # What the loss saw. The difference from ``kl`` is the whole effect of
        # the arm on the objective's size, and it is signed.
        "w_kl": w * kl,
        # Gross rather than net: an arm that adds a nat here and removes one
        # there reports zero net and is doing the redistribution it exists to do.
        "kl_shift_abs": (w - 1.0).abs() * kl,
    }
    for name, sids in POSITION_STATE_GROUPS:
        sel = torch.zeros_like(state, dtype=torch.bool)
        for sid in sids:
            sel = sel | (state == sid)
        terms[f"w_from_{name}"] = (excess * sel.to(excess.dtype)).sum(dim=-1)

    lo = min(float(weight_range[0]), 1.0)
    hi = max(float(weight_range[1]), 1.0)
    span = hi - lo
    if span <= 0:
        frac = torch.zeros_like(wp)
    else:
        frac = ((wp - lo) / span).clamp(min=0.0, max=1.0)
    for q in POSITION_BANDS:
        band = (frac >= q).to(kl.dtype)
        tag = f"band{int(q * 100):02d}"
        terms[f"{tag}_n"] = band
        terms[f"{tag}_kl"] = band * kl
    return terms


def position_ratio_metrics(sums: dict, prefix: str = "sign_weight") -> dict:
    """What the direct-KL weighting did, from sums rather than means of ratios.

    Five readings, in the order a write-up needs them.

    ``w_mean`` / ``w_std`` / ``w_cv`` -- the applied weight's distribution.
    ``w_mean`` is the one number that says whether the normalisation is holding:
    it is 1 by construction only if the per-task means it divides by were exact,
    and they are the PREVIOUS step's, so the drift from 1 is how stale they were.
    ``w_cv`` is the spread that drift is measured against -- a 1% drift matters
    at a 2% spread and does not at 20%.

    ``kl_scale`` = ``sum w*kl / sum kl`` -- the factor the arm applied to the
    total teacher KL. This is the number that separates the mechanism from a
    change to ``teacher_kl_loss_coef``: at exactly 1 the arm only redistributed,
    and any departure is distillation strength that the weighting bought without
    saying so. Note it is NOT ``w_mean``: they coincide only when the weight and
    the KL are uncorrelated, which is precisely what the arm hopes is false.

    ``kl_shift_net`` / ``kl_shift_gross`` -- nats per token added, and nats per
    token moved. Their ratio is the redistribution measure: near 0 the arm
    shuffled effort between positions, near 1 it scaled every position the same
    way and is a coefficient change wearing a mechanism's clothes.

    ``w_kl_lift`` = ``E[w*kl] / (E[w] E[kl])`` and ``w_kl_corr``, its
    correlation form. THE headline for the position arm: it says where the
    weighting spent its budget. Above 1 the extra effort landed on positions the
    student was ALREADY far from its teacher on -- the weighting piled onto
    tokens ordinary distillation was going to fix anyway. Below 1 it landed where
    the student had nearly converged, which is the only regime where a
    redistribution can change the outcome rather than the schedule. The lift is
    scale-free in both arguments; the correlation is the same statement bounded
    to [-1, 1], and needs the two variances, which is why they are accumulated.

    ``band*_kl_share`` vs ``band*_n_share`` -- how concentrated the weighting is
    and whether the concentrated part carries the KL. The bands are cuts of the
    weight TABLE's range, so a share that moves across steps is the arm changing
    and not the cut.

    Args:
        sums: the mapping :meth:`ScopeTermStats.sums` returns.
    """
    out = {}
    for scope_name, tot in sums.items():
        head = prefix if scope_name is None else f"{prefix}/{scope_name}"
        n = tot["n"]
        if n <= 0:
            continue
        w_mean = tot["w"] / n
        w_var = max(tot["w_sq"] / n - w_mean * w_mean, 0.0)
        kl_mean = tot["kl"] / n
        kl_var = max(tot["kl_sq"] / n - kl_mean * kl_mean, 0.0)
        wpre_mean = tot["w_pre"] / n
        wpre_var = max(tot["w_pre_sq"] / n - wpre_mean * wpre_mean, 0.0)

        out[f"{head}/pos/w_mean"] = w_mean
        out[f"{head}/pos/w_std"] = math.sqrt(w_var)
        out[f"{head}/pos/w_cv"] = math.sqrt(w_var) / w_mean if abs(w_mean) > 1e-12 else 0.0
        # Signed, and not folded into w_mean: "the normaliser was stale by +0.4%"
        # and "the weights average 1.004" are the same number and only the first
        # is a sentence about the run.
        out[f"{head}/pos/w_norm_drift"] = w_mean - 1.0
        out[f"{head}/pos/w_pre_mean"] = wpre_mean
        out[f"{head}/pos/w_pre_std"] = math.sqrt(wpre_var)

        if abs(tot["kl"]) > 1e-12:
            out[f"{head}/pos/kl_scale"] = tot["w_kl"] / tot["kl"]
        out[f"{head}/pos/kl_shift_net"] = (tot["w_kl"] - tot["kl"]) / n
        gross = tot["kl_shift_abs"] / n
        out[f"{head}/pos/kl_shift_gross"] = gross
        if gross > 1e-12:
            out[f"{head}/pos/kl_shift_net_over_gross"] = ((tot["w_kl"] - tot["kl"]) / n) / gross

        denom = w_mean * kl_mean
        if abs(denom) > 1e-12:
            out[f"{head}/pos/w_kl_lift"] = (tot["w_kl"] / n) / denom
        if w_var > 1e-24 and kl_var > 1e-24:
            cov = tot["w_kl"] / n - w_mean * kl_mean
            out[f"{head}/pos/w_kl_corr"] = cov / math.sqrt(w_var * kl_var)

        # The exact partition of w_pre - 1. Reported as a share so it is readable
        # next to the others, and only when there is something to divide by --
        # at a table that never fires the shares are 0/0, not 0.
        excess = wpre_mean - 1.0
        for name, _ in POSITION_STATE_GROUPS:
            out[f"{head}/pos/w_from/{name}"] = tot[f"w_from_{name}"] / n
            if abs(excess) > 1e-12:
                out[f"{head}/pos/w_share/{name}"] = (tot[f"w_from_{name}"] / n) / excess

        for q in POSITION_BANDS:
            tag = f"band{int(q * 100):02d}"
            n_share = tot[f"{tag}_n"] / n
            out[f"{head}/pos/{tag}_n_share"] = n_share
            if abs(tot["kl"]) > 1e-12:
                kl_share = tot[f"{tag}_kl"] / tot["kl"]
                out[f"{head}/pos/{tag}_kl_share"] = kl_share
                # >1: the heavily weighted positions are also the expensive ones.
                if n_share > 1e-12:
                    out[f"{head}/pos/{tag}_kl_lift"] = kl_share / n_share
    return out


class SignPairCounts:
    """The (on-task, off-task) sign contingency table, per ordered task pair.

    One accumulator, five readings. It exists because the questions "is what
    transfers common knowledge", "whose knowledge is it", and "what does the gate
    throw away" are all questions about the same joint distribution of signs, and
    building three accumulators for it would let them disagree.

    Cell axes, in index order:

    ``dst``  the task of the ROW -- whose states these are, i.e. who would receive.
    ``src``  the task of the off-task PLANE -- whose teacher is being consulted.
    ``i``    ``sign(delta_on) + 1`` for the row's own teacher: 0 lower, 1 silent, 2 raise.
    ``j``    ``sign(delta_src) + 1`` for that off-task teacher.
    ``l``    ``sign(student - on_task_teacher) + 1``: which side of its own target
             the student sits on at this candidate. The residual, not the shift
             from base -- the student is TRAINED toward the on-task teacher, so
             its shift from base is dominated by that and says little.
    ``o``    do the OTHER off-task planes all carry the same sign as the on-task
             teacher. Only meaningful when ``i != 1``; it is what makes the
             leave-one-out reading possible (would the gate still pass without
             ``src``).
    ``h``    headroom: is a ``+deadzone`` raise arithmetically possible at all,
             i.e. ``-base_logprob > deadzone``. Without this axis a large
             "the off-task teacher had an opinion the on-task one did not" mass
             cannot be told from the log-space ceiling: at ``p_0 > 0.905`` no
             teacher CAN raise a token by 0.1 nats, so its silence there is
             arithmetic, not ignorance.

    Three arrays over those cells: candidate counts, on-task teacher probability
    mass, and the SQUARE of that mass. The third is not decoration -- the mass
    weight here is extremely heavy-tailed (the shipped run puts 64% of teacher
    mass on 4.2% of candidates), so ``(sum p)^2 / sum p^2`` is the effective
    sample size, and without it a mass-weighted rate has no error bar and cannot
    be told from step noise.

    float64, because millions of atomic adds concentrate into a few hundred cells
    and CUDA's ``index_add_`` does not promise an order.

    All-reduced before rendering: these cells are sparse per pair and a mean of
    per-rank ratios is not the pooled ratio. Note that this makes ``agree_count``
    here a GLOBAL number while the shipped ``sign_weight/agree_rate`` is rank-0
    local -- they agree only at world_size 1, and both are reported so the
    difference is visible rather than assumed away.
    """

    N_I = N_J = N_L = 3
    N_O = N_H = 2

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.n_cells = T * T * self.N_I * self.N_J * self.N_L * self.N_O * self.N_H
        self.count = torch.zeros(self.n_cells, dtype=torch.float64, device=device)
        self.mass = torch.zeros(self.n_cells, dtype=torch.float64, device=device)
        self.mass_sq = torch.zeros(self.n_cells, dtype=torch.float64, device=device)
        # Population totals per receiving task, accumulated ONCE per micro-batch.
        # Inside the per-plane loop every candidate's mass would be added once per
        # plane, so the denominator would read half its true value at n_off == 2 --
        # and it is the number that says which population a rate is a rate OF.
        self.totals = torch.zeros((T, 2), dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(
        self,
        *,
        on_task_logprob: torch.Tensor,
        off_task_logprobs: torch.Tensor,
        base_logprob: torch.Tensor,
        student_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: torch.Tensor,
        off_plane_tasks: torch.Tensor,
        deadzone: float,
        student_deadzone: float = 0.0,
    ) -> None:
        """Fold one micro-batch in.

        The signs are recomputed here rather than returned from
        :func:`candidate_weights`. That function is on the loss path and its
        arity is depended on at eight call sites; two subtractions and two
        comparisons are the cheaper trade, and ``SignWeightStats.update_candidates``
        already recomputes exactly the same tensors.
        """
        self._cpu_cache = None
        T = self.n_tasks
        valid = response_mask.to(torch.bool).unsqueeze(-1).expand_as(on_task_logprob)

        p_on = on_task_logprob.detach().to(torch.float32).exp()
        sign_on = _deadzoned_sign(on_task_logprob.detach() - base_logprob.detach(), deadzone)
        sign_off = _deadzoned_sign(
            off_task_logprobs.detach() - base_logprob.detach().unsqueeze(-1), deadzone
        )
        sign_res = _deadzoned_sign(
            student_logprob.detach().to(on_task_logprob.dtype) - on_task_logprob.detach(),
            student_deadzone,
        )
        # Can any model raise this token past the deadzone at all? log p <= 0, so
        # a +deadzone raise needs -log p_0 > deadzone.
        headroom = (-base_logprob.detach() > deadzone).long()

        dst = task_ids.reshape(-1).to(torch.long)
        n_off = off_task_logprobs.size(-1)
        # How many planes carry the on-task sign, for the leave-one-out axis.
        same_as_on = (sign_off == sign_on.unsqueeze(-1))
        n_same = same_as_on.sum(dim=-1)

        i = (sign_on + 1).long()
        l = (sign_res + 1).long()
        dst_b = dst.view(-1, 1, 1)

        for c in range(n_off):
            j = (sign_off[..., c] + 1).long()
            others = ((n_same - same_as_on[..., c].long()) == (n_off - 1)).long()
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            # -1 is legitimate on both (an unnamed plane, a padding row). Clamp the
            # INDEX and fold validity into the VALUE; selecting rows out first
            # would need their count on the host.
            ok = valid & (dst_b >= 0) & (src.view(-1, 1, 1) >= 0)
            flat = (
                ((((((dst.clamp(min=0).view(-1, 1, 1) * T + src.clamp(min=0).view(-1, 1, 1))
                     * self.N_I + i) * self.N_J + j) * self.N_L + l) * self.N_O + others)
                 * self.N_H + headroom)
            ).reshape(-1)
            okf = ok.to(torch.float64).reshape(-1)
            m = (p_on.to(torch.float64) * ok).reshape(-1)
            self.count.index_add_(0, flat, okf)
            self.mass.index_add_(0, flat, m)
            self.mass_sq.index_add_(0, flat, m * m)

        # Once per micro-batch, outside the plane loop.
        ok_row = valid & (dst_b >= 0)
        tot = torch.stack(
            [ok_row.to(torch.float64).sum(dim=(1, 2)), (p_on.to(torch.float64) * ok_row).sum(dim=(1, 2))],
            dim=-1,
        )
        self.totals.index_add_(0, dst.clamp(min=0), tot * (dst >= 0).to(torch.float64).unsqueeze(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for t in (self.count, self.mass, self.mass_sq, self.totals):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            T = self.n_tasks
            shape = (T, T, self.N_I, self.N_J, self.N_L, self.N_O, self.N_H)
            self._cpu_cache = (
                self.count.detach().to("cpu").view(shape),
                self.mass.detach().to("cpu").view(shape),
                self.mass_sq.detach().to("cpu").view(shape),
                self.totals.detach().to("cpu"),
            )
        return self._cpu_cache

    # -- readings ---------------------------------------------------------- #

    def _pair_name(self, i, task_names):
        if task_names and i < len(task_names):
            return task_names[i]
        return f"task{i}"

    def metrics(self, task_names=None, prefix: str = "sign_weight", min_count: float = 1000.0) -> dict:
        """Every reading this table supports, rendered host-side.

        Three families, all over the ordered pair ``src -> dst`` ("what src's
        teacher says at dst's states"):

        ``pair/``       how associated the two teachers' opinions are.
        ``gate/``       what the unanimity rule does with that association.
        ``blindspot/``  what src says where dst's own teacher says nothing.

        Cells with too few candidates are omitted rather than reported noisy: a
        rate over a handful of candidates goes through ``reduce_metrics`` looking
        exactly like a rate over millions.
        """
        count, mass, mass_sq, totals = self._cpu()
        out = {}
        T = self.n_tasks
        LOW, SIL, HIGH = 0, 1, 2  # i / j values: lower, silent, raise

        for dst in range(T):
            for src in range(T):
                if dst == src:
                    continue
                c = count[dst, src]
                if float(c.sum()) < min_count:
                    continue
                m = mass[dst, src]
                msq = mass_sq[dst, src]
                pair = f"{self._pair_name(src, task_names)}__on__{self._pair_name(dst, task_names)}"

                # ---- pair association ----------------------------------- #
                # The 2x2 over non-silent signs. n[a][b] = both spoke, on-task
                # said a, off-task said b.
                n = {(a, b): float(c[a, b].sum()) for a in (LOW, HIGH) for b in (LOW, HIGH)}
                n_tot = sum(n.values())
                if n_tot > 0:
                    agree = n[(HIGH, HIGH)] + n[(LOW, LOW)]
                    out[f"{prefix}/pair/agree_count/{pair}"] = agree / n_tot
                    # Haldane-corrected log odds ratio. THE headline of this
                    # family: unlike an agreement rate it is invariant to each
                    # teacher's own propensity to raise rather than lower, which
                    # is exactly what an agreement rate confounds with
                    # association. The teachers differ 3.7x in measured drift, so
                    # that confound is not hypothetical here.
                    lor = math.log(
                        ((n[(HIGH, HIGH)] + 0.5) * (n[(LOW, LOW)] + 0.5))
                        / ((n[(HIGH, LOW)] + 0.5) * (n[(LOW, HIGH)] + 0.5))
                    )
                    out[f"{prefix}/pair/lor/{pair}"] = lor
                    # The marginals the odds ratio divides out, reported so a
                    # near-zero lor can be told from a degenerate table.
                    out[f"{prefix}/pair/sign_bias_on/{pair}"] = (n[(HIGH, HIGH)] + n[(HIGH, LOW)]) / n_tot
                    out[f"{prefix}/pair/sign_bias_off/{pair}"] = (n[(HIGH, HIGH)] + n[(LOW, HIGH)]) / n_tot

                # Mass-weighted agreement, and the population it is a rate of.
                # Never quote one without the other: the count-weighted matrix
                # can describe a tail population the objective barely touches.
                m_both = {(a, b): float(m[a, b].sum()) for a in (LOW, HIGH) for b in (LOW, HIGH)}
                m_tot = sum(m_both.values())
                if m_tot > 0:
                    out[f"{prefix}/pair/agree_mass/{pair}"] = (
                        m_both[(HIGH, HIGH)] + m_both[(LOW, LOW)]
                    ) / m_tot
                    dst_mass = float(totals[dst, 1])
                    if dst_mass > 0:
                        out[f"{prefix}/pair/agree_pop_mass/{pair}"] = m_tot / dst_mass
                    sq = float(msq[[LOW, HIGH]][:, [LOW, HIGH]].sum())
                    if sq > 0:
                        # (sum p)^2 / sum p^2 -- how many equally-weighted
                        # candidates this mass-weighted rate is really worth.
                        out[f"{prefix}/pair/agree_ess/{pair}"] = (m_tot * m_tot) / sq

                # ---- the gate's leave-one-out ---------------------------- #
                # Restricted to i != silent, since the gate never fires there.
                # o == 1 means every OTHER off-task plane already agrees with the
                # on-task teacher, so src alone decides whether the gate passes.
                for label, sel in (
                    ("concur", [(HIGH, HIGH), (LOW, LOW)]),
                    ("veto_silent", [(HIGH, SIL), (LOW, SIL)]),
                    ("veto_dissent", [(HIGH, LOW), (LOW, HIGH)]),
                ):
                    mm = sum(float(m[a, b, :, 1].sum()) for a, b in sel)
                    cc = sum(float(c[a, b, :, 1].sum()) for a, b in sel)
                    out[f"{prefix}/gate/{label}_mass/{pair}"] = mm
                    out[f"{prefix}/gate/{label}_count/{pair}"] = cc
                gate_pop = float(m[[LOW, HIGH], :, :, 1].sum())
                out[f"{prefix}/gate/pop_mass/{pair}"] = gate_pop
                if gate_pop > 0:
                    for label, sel in (
                        ("concur", [(HIGH, HIGH), (LOW, LOW)]),
                        ("veto_silent", [(HIGH, SIL), (LOW, SIL)]),
                        ("veto_dissent", [(HIGH, LOW), (LOW, HIGH)]),
                    ):
                        mm = sum(float(m[a, b, :, 1].sum()) for a, b in sel)
                        out[f"{prefix}/gate/{label}_frac/{pair}"] = mm / gate_pop

                # ---- the blind spot -------------------------------------- #
                # i == SIL: the row's own teacher had no opinion. Whatever src
                # says here is knowledge the distillation target cannot carry,
                # because the target is the on-task teacher and it did not move.
                # This is where "specialist" knowledge would have to live.
                for label, jj in (("pos", HIGH), ("neg", LOW)):
                    total_m = float(m[SIL, jj].sum())
                    out[f"{prefix}/blindspot/off_opinion_mass_{label}/{pair}"] = total_m
                    # Split by headroom. Silence at h == 0 is arithmetic, not
                    # ignorance: no model can raise a token whose base
                    # probability already exceeds exp(-deadzone).
                    if total_m > 0:
                        out[f"{prefix}/blindspot/off_opinion_h1_frac_{label}/{pair}"] = (
                            float(m[SIL, jj, :, :, 1].sum()) / total_m
                        )
                dst_mass = float(totals[dst, 1])
                if dst_mass > 0:
                    out[f"{prefix}/blindspot/silent_pop_mass/{pair}"] = float(m[SIL].sum()) / dst_mass

                # ---- does the student take the off-task side? ------------ #
                # Among candidates where the on-task teacher was SILENT and src
                # had an opinion: does the student sit on src's side of its own
                # target? The on-task teacher said nothing, so a lean here cannot
                # be attributed to the distillation target -- which is the whole
                # reason this cell is the one worth reading.
                for label, jj in (("pos", HIGH), ("neg", LOW)):
                    lead = float(c[SIL, jj, jj].sum())   # student residual sign == src's sign
                    opp = float(c[SIL, jj, 2 - jj].sum())
                    if lead + opp >= min_count:
                        out[f"{prefix}/blindspot/student_follows_{label}/{pair}"] = lead / (lead + opp)
        return out


# --------------------------------------------------------------------------- #
# src -> dst -> token
# --------------------------------------------------------------------------- #
# The three classes a candidate can fall into that carry an OPINION FROM src.
# The full (on-task sign, src sign) contingency has nine cells, but four of them
# are "src said nothing" and two more are "the on-task teacher said nothing and
# neither did src" -- none of which has a token identity worth a vocabulary-wide
# array. Collapsing to these three is what makes the table affordable: at
# T*T*3*3*V the cells are 12.3M and at T*(T-1)*3*V they are 2.7M.
PAIR_TOKEN_CLASSES = {
    0: "agree",       # src moved the token the same way the on-task teacher did
    1: "conflict",    # src moved it the opposite way
    2: "blindspot",   # the on-task teacher is silent and src is not
}


class SignPairTokens:
    """Which vocabulary tokens each teacher sends to each other task's states.

    :class:`SignPairCounts` answers "how often does src agree with dst's teacher"
    with the vocabulary summed out, and :class:`TokenStateCounts` answers "which
    tokens" with the SOURCE summed out. Neither can say whether the tokens
    alfworld's teacher pushes into search's states are the same ones webshop's
    teacher pushes there -- which is the difference between "the tasks share a
    common surface vocabulary" and "each teacher contributes its own", and that
    difference is the whole content of a transfer claim.

    Cell axes: ``(pair, cls, token)``.

    ``pair`` is the ORDERED (dst, src) pair with the diagonal compressed out.
    ``dst`` is the task of the row -- whose states these are, who receives --
    and ``src`` is the task of the off-task plane, whose teacher is speaking.
    ``src == dst`` cannot occur (a teacher is never off-task on its own rows), so
    a square ``T x T`` layout would spend a third of the array on cells that are
    structurally zero: at T=3 that is 27 MB of 82. The index is therefore
    ``dst * (T-1) + src - (src > dst)``, which is exactly invertible.

    ``cls`` is :data:`PAIR_TOKEN_CLASSES`. ``blindspot`` is the population the
    distillation target structurally cannot carry -- src has an opinion where
    dst's own teacher has none -- and is the one this table exists for.

    Three arrays over those cells:

    * ``n`` -- how often. The event count.
    * ``mass`` -- the on-task teacher's probability there. A token src keeps
      voting on that dst's teacher was never going to say reaches nothing.
    * ``eff`` -- the post-normalisation effect the weighting actually had at that
      candidate, signed, in whatever unit the mode measures effects in (see
      :func:`candidate_effect`). Under the unanimity gate every off-task teacher
      is one of the voices behind a fired weight, so each src is credited with
      the whole effect rather than a share of it: the question this answers is
      "was src one of the voices", not "how much of the vote was src's".

    Sizing, at T=3 and V=151,936: 2.7M cells at int64 + float32 + float64 is
    54.7 MB, one all-reduce and one device-to-host read of that per
    ``update_policy``. For scale, :class:`TokenStateCounts` is already 119 MB on
    the same run and a teacher output projection is 622 MB.

    Dense and sync-free for the reason :class:`TokenStateCounts` is: a
    ``torch.unique`` per micro-batch would read the device thousands of times a
    step, which is the run-ahead this actor's design protects.
    """

    N_CLS = len(PAIR_TOKEN_CLASSES)

    def __init__(self, *, n_tasks: int, vocab_size: int, device, top_n: int = 64):
        assert n_tasks >= 2, "a pair table needs at least two tasks"
        self.n_tasks = T = int(n_tasks)
        self.vocab_size = V = int(vocab_size)
        self.n_pairs = T * (T - 1)
        self.top_n = int(top_n)
        cells = self.n_pairs * self.N_CLS * V
        self.n = torch.zeros(cells, dtype=torch.int64, device=device)
        self.mass = torch.zeros(cells, dtype=torch.float32, device=device)
        self.eff = torch.zeros(cells, dtype=torch.float64, device=device)
        self._cpu_cache = None

    # -- accumulation ------------------------------------------------------ #

    def update(
        self,
        *,
        support_ids: torch.Tensor,
        on_task_logprob: torch.Tensor,
        off_task_logprobs: torch.Tensor,
        base_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: torch.Tensor,
        off_plane_tasks: torch.Tensor,
        deadzone: float,
        effect: Optional[torch.Tensor] = None,
        mass: Optional[torch.Tensor] = None,
    ) -> None:
        """Fold one micro-batch in.

        Args:
            support_ids: (bs, resp, k) vocabulary ids of the support.
            mass: (bs, resp, k) the on-task teacher's probability, when the
                caller's ``on_task_logprob`` is not a log-probability. The
                parameter-free arm passes STANDARDIZED shifts against a zero
                base -- which is what makes its deadzone comparable across
                teachers -- and those do not exponentiate to a probability.
                Defaults to ``exp(on_task_logprob)``.
            effect: (bs, resp, k) from :func:`candidate_effect`, or
                (bs, resp, k, n_off) when the arm can say which SOURCE caused
                which part of it, or None on an arm that applies no weight --
                the counts and the mass are still the answer to "who names
                what", and ``eff`` simply stays zero.

                The three-dimensional form files the SAME total against every
                source that spoke at the candidate. That is right for the sign
                arm, whose weight table is a function of the sign PATTERN and
                cannot be decomposed over the teachers that produced it. It is
                wrong for a weight built as a sum over sources: there,
                "what did Search bring to AlfWorld" reads back as Search's share
                plus WebShop's plus the part neither caused, and a source with
                alpha = 0 shows up carrying effect it contributed none of. Pass
                the four-dimensional form and each source gets its own column.

        The signs are recomputed here rather than taken from
        :func:`candidate_weights`, for the reason :class:`SignPairCounts` does
        it: that function is on the loss path and its arity is depended on at
        eight call sites.
        """
        self._cpu_cache = None
        T = self.n_tasks
        V = self.vocab_size
        valid = response_mask.to(torch.bool).unsqueeze(-1).expand_as(on_task_logprob)

        p_on = (
            on_task_logprob.detach().to(torch.float32).exp()
            if mass is None
            else mass.detach().to(torch.float32)
        )
        sign_on = _deadzoned_sign(on_task_logprob.detach() - base_logprob.detach(), deadzone)
        sign_off = _deadzoned_sign(
            off_task_logprobs.detach() - base_logprob.detach().unsqueeze(-1), deadzone
        )
        eff = (
            torch.zeros_like(p_on, dtype=torch.float64)
            if effect is None
            else effect.detach().to(torch.float64)
        )
        # (bs, resp, k) shared by every source, or (bs, resp, k, n_off) sliced
        # per source in the loop below.
        eff_per_source = eff.dim() == on_task_logprob.dim() + 1
        if eff_per_source:
            assert eff.size(-1) == off_task_logprobs.size(-1), (
                f"per-source effect has {eff.size(-1)} columns for "
                f"{off_task_logprobs.size(-1)} off-task teachers"
            )
        dst = task_ids.reshape(-1).to(torch.long)
        dst_b = dst.view(-1, 1, 1)
        on_silent = sign_on == 0

        for c in range(off_task_logprobs.size(-1)):
            s = sign_off[..., c]
            speaks = s != 0
            agree = (~on_silent) & (s == sign_on)
            conflict = (~on_silent) & speaks & (s != sign_on)
            blind = on_silent & speaks
            # -1 for "src had nothing to say here", folded into the mask below
            # rather than into a fourth class: a class with no opinion in it
            # would be a vocabulary-wide array of the whole support.
            cls = torch.full_like(sign_on, -1, dtype=torch.long)
            cls = torch.where(agree, torch.zeros_like(cls), cls)
            cls = torch.where(conflict, torch.ones_like(cls), cls)
            cls = torch.where(blind, torch.full_like(cls, 2), cls)

            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            src_b = src.view(-1, 1, 1)
            ok = valid & (dst_b >= 0) & (src_b >= 0) & (cls >= 0)
            # The diagonal is structurally absent, so the pair index skips it.
            # clamp before the comparison: a -1 src would otherwise decide the
            # (src > dst) branch on a row that is masked out anyway.
            d_c, s_c = dst_b.clamp(min=0), src_b.clamp(min=0)
            pair = d_c * (T - 1) + s_c - (s_c > d_c).long()
            flat = ((pair * self.N_CLS + cls.clamp(min=0)) * V + support_ids.to(torch.long)).reshape(-1)

            self.n.index_add_(0, flat, ok.reshape(-1).to(torch.int64))
            self.mass.index_add_(0, flat, (p_on * ok).reshape(-1))
            eff_c = eff[..., c] if eff_per_source else eff
            self.eff.index_add_(0, flat, (eff_c * ok).reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for t in (self.n, self.mass, self.eff):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    # -- rendering --------------------------------------------------------- #

    def _cpu(self):
        if self._cpu_cache is None:
            shape = (self.n_pairs, self.N_CLS, self.vocab_size)
            self._cpu_cache = (
                self.n.detach().to("cpu").view(shape),
                self.mass.detach().to("cpu").view(shape),
                self.eff.detach().to("cpu").view(shape),
            )
        return self._cpu_cache

    def _pair_index(self, dst: int, src: int) -> int:
        return dst * (self.n_tasks - 1) + src - (1 if src > dst else 0)

    def _pairs(self, task_names=None):
        """``(pair_index, dst_name, src_name)`` for every ordered pair."""
        def name(t):
            return task_names[t] if task_names and t < len(task_names) else f"task{t}"
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                yield self._pair_index(dst, src), name(dst), name(src)

    def scalar_metrics(self, task_names=None, prefix: str = "sign_weight") -> dict:
        """Shape per pair, and the overlap BETWEEN the pairs sharing a receiver.

        ``n_distinct`` and ``top_share`` say whether src's contribution to dst is
        a small stable vocabulary or a broad one -- the same question
        :class:`TokenStateCounts` asks, but now attributable to a sender.

        ``token_overlap`` is the reading neither existing accumulator can give:
        the weighted Jaccard between the two srcs' token vectors at one dst. Near
        1 the off-task teachers are naming the same tokens, so what crosses is
        common surface vocabulary and no individual sender is necessary; near 0
        each sender contributes its own, and the unanimity gate is passing on the
        intersection of two different vocabularies. Weighted rather than
        set-valued because a token seen once and a token seen ten thousand times
        are not the same claim, and a set Jaccard scores them identically.
        """
        out = {}
        n, mass, eff = self._cpu()
        N = min(self.top_n, self.vocab_size)
        for p, dst, src in self._pairs(task_names):
            for cid, cname in PAIR_TOKEN_CLASSES.items():
                counts = n[p, cid]
                total = int(counts.sum())
                if total <= 0:
                    continue
                head = f"{prefix}/pair/token/{cname}/{src}__on__{dst}"
                out[f"{head}/n_distinct"] = float((counts > 0).sum())
                out[f"{head}/top{N}_share"] = float(torch.topk(counts, N).values.sum()) / total
                out[f"{head}/mass"] = float(mass[p, cid].sum())
                out[f"{head}/eff_net"] = float(eff[p, cid].sum())

        # One overlap per (dst, class, unordered src pair).
        for dst in range(self.n_tasks):
            srcs = [s for s in range(self.n_tasks) if s != dst]
            dname = task_names[dst] if task_names and dst < len(task_names) else f"task{dst}"
            for a in range(len(srcs)):
                for b in range(a + 1, len(srcs)):
                    pa, pb = self._pair_index(dst, srcs[a]), self._pair_index(dst, srcs[b])
                    an = task_names[srcs[a]] if task_names and srcs[a] < len(task_names) else f"task{srcs[a]}"
                    bn = task_names[srcs[b]] if task_names and srcs[b] < len(task_names) else f"task{srcs[b]}"
                    for cid, cname in PAIR_TOKEN_CLASSES.items():
                        x = n[pa, cid].to(torch.float64)
                        y = n[pb, cid].to(torch.float64)
                        union = float(torch.maximum(x, y).sum())
                        if union <= 0:
                            continue
                        out[f"{prefix}/pair/token_overlap/{cname}/{an}__and__{bn}__on__{dname}"] = (
                            float(torch.minimum(x, y).sum()) / union
                        )
        return out

    def turnover(self, previous=None, task_names=None, prefix: str = "sign_weight"):
        """Per PAIR, is it the same vocabulary each step?

        The pooled turnover cannot answer this. "The arm acts on a stable set of
        tokens" and "each source contributes a stable set, and they are different
        sets" produce the same pooled Jaccard, and the second is the claim this
        arm is for -- that the tasks share structure a particular OTHER task can
        supply. A pair whose set churns while the pooled one is stable is a pair
        contributing noise into a stable total.

        Same two readings as :meth:`TokenStateCounts.turnover`: set membership,
        and the share of THIS step's attributed nats that landed on tokens the
        previous step had ranked. Returns ``(metrics, state)``; the first call
        returns no metrics because there is nothing to compare against.
        """
        _n, _mass, eff = self._cpu()
        N = min(self.top_n, self.vocab_size)
        out, state = {}, {}
        for pair, dst, src in self._pairs(task_names):
            per_tok = eff[pair].sum(0).abs()
            tot = float(per_tok.sum())
            if tot <= 0:
                continue
            vals, idx = torch.topk(per_tok, N)
            cur = [int(t) for v, t in zip(vals.tolist(), idx.tolist()) if v > 0]
            if not cur:
                continue
            key = f"{src}__on__{dst}"
            state[key] = cur
            prev = (previous or {}).get(key)
            if not prev:
                continue
            a, b = set(cur), set(prev)
            head = f"{prefix}/pair_token/{key}/turnover"
            out[f"{head}/top{N}_jaccard"] = len(a & b) / len(a | b)
            keep = torch.zeros(self.vocab_size, dtype=torch.bool)
            keep[torch.tensor(sorted(b), dtype=torch.long)] = True
            out[f"{head}/effect_carryover"] = float(per_tok[keep].sum()) / tot
        return out, state

    def top_tokens(self, task_names=None) -> list:
        """The ranked rows themselves, ids not strings -- the actor has no tokenizer.

        Three rankings per (pair, class), for the reason
        :meth:`TokenStateCounts.top_tokens` has three: how often src named it,
        how much of dst's teacher's mass sits there, and how much the weighting
        moved because of it. A token can top one and be absent from the others.
        """
        rows = []
        n, mass, eff = self._cpu()
        N = min(self.top_n, self.vocab_size)
        for p, dst, src in self._pairs(task_names):
            for cid, cname in PAIR_TOKEN_CLASSES.items():
                if int(n[p, cid].sum()) <= 0:
                    continue
                series = (
                    ("count", n[p, cid].to(torch.float64)),
                    ("mass", mass[p, cid].to(torch.float64)),
                    ("abs_effect", eff[p, cid].abs()),
                )
                for ranked_by, values in series:
                    vals, idx = torch.topk(values, N)
                    for rank, (v, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                        if v <= 0:
                            break
                        rows.append({
                            "table": "pair_token",
                            "dst": dst, "src": src, "cls": cname,
                            "ranked_by": ranked_by, "rank": rank, "token_id": int(tok),
                            "count": int(n[p, cid, tok]),
                            "mass": float(mass[p, cid, tok]),
                            "effect_net": float(eff[p, cid, tok]),
                        })
        return rows


# --------------------------------------------------------------------------- #
# what kind of token a position is
# --------------------------------------------------------------------------- #
# Every scalar in this module is an average over positions of very different
# kinds. A weight that fires almost entirely inside <think> is a claim about
# reasoning style; the same number concentrated on the contents of <action> is a
# claim about which moves the tasks share. Those are different findings with the
# same summary, and nothing so far can tell them apart.
ROLE_FORMAT = 0       # between the tagged spans: whitespace, chat scaffolding
ROLE_REASONING = 1    # inside <think> ... </think>
ROLE_ENV_ACTION = 2   # inside <action> ... </action>, the move the env executes
ROLE_TOOL_CALL = 3    # inside <search> / <answer>, a call or a final answer
ROLE_ENV_OBS = 4      # inside <information>, returned BY the env
ROLE_TAG = 5          # the tag tokens themselves -- pure syntax

ROLE_NAMES = {
    ROLE_FORMAT: "format",
    ROLE_REASONING: "reasoning",
    ROLE_ENV_ACTION: "env_action",
    ROLE_TOOL_CALL: "tool_call",
    ROLE_ENV_OBS: "env_obs",
    ROLE_TAG: "tag",
}

# The tag vocabulary these environments actually use (agent_system/environments/
# prompts/*.py): alfworld / sokoban / webshop reason in <think> and act in
# <action>; search reasons in <think>, calls in <search> and finishes in
# <answer>, with the retriever's reply wrapped in <information>. A closing tag
# returns the stream to ROLE_FORMAT rather than to whatever was open before it,
# because these spans do not nest here.
TAG_ROLES = {
    "<think>": ROLE_REASONING,
    "</think>": ROLE_FORMAT,
    "<action>": ROLE_ENV_ACTION,
    "</action>": ROLE_FORMAT,
    "<search>": ROLE_TOOL_CALL,
    "</search>": ROLE_FORMAT,
    "<answer>": ROLE_TOOL_CALL,
    "</answer>": ROLE_FORMAT,
    "<information>": ROLE_ENV_OBS,
    "</information>": ROLE_FORMAT,
}


def token_roles(responses: torch.Tensor, tag_ids: dict) -> torch.Tensor:
    """(bs, resp) role code per position, from the tag ids alone.

    Done on the token ids rather than on decoded text because this runs inside
    the actor, which has no tokenizer -- the worker tokenises the ten tag strings
    once at startup and hands the ids down. A tag is usually several tokens
    ("<", "think", ">"), so each one is matched as a SEQUENCE: L shifted
    comparisons per tag, ten tags, over a (bs, resp) int tensor. That is about
    fifty elementwise compares per micro-batch, against a forward pass.

    The scan is a ``cummax`` over ``position * n_tags + tag_code`` at the
    positions where a tag ends, which makes "the most recently opened tag" a
    single kernel rather than a loop over the response length.

    Args:
        responses: (bs, resp) token ids.
        tag_ids: ``{tag_string: [ids]}``; a tag absent from the dict, or one
            whose ids are empty, is simply never matched. Tag strings outside
            :data:`TAG_ROLES` are ignored rather than refused, so a caller can
            hand over its whole special-token table.

    Returns:
        (bs, resp) int64 in :data:`ROLE_NAMES`. Positions before any tag are
        ``ROLE_FORMAT``: nothing has been opened, which is what that means.
    """
    bs, T = responses.shape
    device = responses.device
    role = torch.full((bs, T), ROLE_FORMAT, dtype=torch.long, device=device)
    if T == 0:
        return role

    tags = [(t, ids) for t, ids in tag_ids.items() if t in TAG_ROLES and len(ids or ())]
    if not tags:
        return role
    K = len(tags)

    pos = torch.arange(T, device=device).unsqueeze(0)
    # -1 = "no tag ends here". Encoded together with the position so one cummax
    # recovers both which tag was last and that it was in fact seen.
    event = torch.full((bs, T), -1, dtype=torch.long, device=device)
    is_tag = torch.zeros((bs, T), dtype=torch.bool, device=device)

    for code, (_tag, ids) in enumerate(tags):
        L = len(ids)
        if L > T:
            continue
        match = torch.ones((bs, T - L + 1), dtype=torch.bool, device=device)
        for i, tid in enumerate(ids):
            match = match & (responses[:, i : T - L + 1 + i] == int(tid))
        ends = torch.zeros((bs, T), dtype=torch.bool, device=device)
        ends[:, L - 1 :] = match
        event = torch.where(ends, pos * K + code, event)
        # The tag's own span, so its tokens are reported as syntax rather than
        # as the first tokens of what they open.
        for back in range(L):
            shifted = torch.zeros((bs, T), dtype=torch.bool, device=device)
            if back == 0:
                shifted = ends
            else:
                shifted[:, :-back] = ends[:, back:]
            is_tag = is_tag | shifted

    last = torch.cummax(event, dim=1).values
    seen = last >= 0
    code = last.clamp(min=0) % K
    # code -> role, as a small gather rather than K comparisons.
    lookup = torch.tensor(
        [TAG_ROLES[tag] for tag, _ in tags], dtype=torch.long, device=device
    )
    role = torch.where(seen, lookup[code], role)
    return torch.where(is_tag, torch.full_like(role, ROLE_TAG), role)


class RoleTokenCounts:
    """Which tokens the weighting moved, cut by WHAT WAS BEING WRITTEN.

    :class:`TokenStateCounts` cuts the vocabulary by task and by agreement
    state. Neither says whether a token was moved while the model was
    reasoning, while it was emitting the action the environment executes, or in
    the scaffolding between them -- and those are different claims about the
    mechanism. "The arm reinforced ``go`` and ``take`` inside ``<action>``" and
    "the arm reinforced ``go`` and ``take`` inside ``<think>``" produce the same
    row in every table this module already has.

    Deliberately thinner than ``TokenStateCounts``: no state axis and no mass
    column, so the buffer is ``n_roles * V`` rather than
    ``(1 + n_tasks) * n_states * V``. At Qwen3's 151,936 that is 22 MB against
    the other table's ~100, and it buys the one axis the aggregates cannot
    reconstruct.

    ``effect`` is the same ``per_candidate_shift`` the other two tables
    decompose, so the three sum to the same nats. Split by sign for the reason
    ``TokenStateCounts`` splits: a token raised in one context and lowered in
    another nets to nothing and is the most interesting row in the table.
    """

    def __init__(self, *, vocab_size: int, device, top_n: int = 32):
        self.vocab_size = int(vocab_size)
        self.n_roles = len(ROLE_NAMES)
        self.top_n = int(top_n)
        cells = self.n_roles * self.vocab_size
        self.n = torch.zeros(cells, dtype=torch.int64, device=device)
        # float64 for the reason TokenStateCounts uses it: millions of atomic
        # adds of ~1e-6 into a few cells lose their tail in float32.
        self.eff_pos = torch.zeros(cells, dtype=torch.float64, device=device)
        self.eff_neg = torch.zeros(cells, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, support_ids, roles, effect, response_mask) -> None:
        """Fold one micro-batch in.

        Args:
            support_ids: (bs, resp, k) the candidate token ids.
            roles: (bs, resp) from :func:`token_roles`.
            effect: (bs, resp, k) the post-normalisation nats each candidate
                moved -- ``per_candidate_shift``.
            response_mask: (bs, resp).
        """
        self._cpu_cache = None
        V = self.vocab_size
        r = roles.clamp(min=0, max=self.n_roles - 1).unsqueeze(-1)
        idx = (r * V + support_ids.clamp(min=0, max=V - 1)).reshape(-1)
        m = response_mask.unsqueeze(-1).expand_as(support_ids)
        e = (effect.detach().to(torch.float64) * m.to(torch.float64)).reshape(-1)
        self.n.index_add_(0, idx, m.reshape(-1).to(torch.int64))
        self.eff_pos.index_add_(0, idx, e.clamp(min=0.0))
        self.eff_neg.index_add_(0, idx, e.clamp(max=0.0))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for buf in (self.n, self.eff_pos, self.eff_neg):
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = tuple(
                b.detach().to("cpu").view(self.n_roles, self.vocab_size)
                for b in (self.n, self.eff_pos, self.eff_neg)
            )
        return self._cpu_cache

    def scalar_metrics(self, prefix: str = "kl_weight") -> dict:
        """Per role: the share of the nats, and how spread over the vocabulary.

        ``shift_gross_share`` is the headline -- it is the same partition the
        role-scoped position accumulator reports, recomputed here from the token
        table so a disagreement between the two is visible rather than assumed
        away.
        """
        n, eff_pos, eff_neg = self._cpu()
        gross_all = float((eff_pos - eff_neg).sum())
        N = min(self.top_n, self.vocab_size)
        out = {}
        for rid in range(self.n_roles):
            rname = ROLE_NAMES[rid]
            counts = n[rid]
            total = int(counts.sum())
            if total <= 0:
                continue
            head = f"{prefix}/role/{rname}"
            gross = float((eff_pos[rid] - eff_neg[rid]).sum())
            # Three columns, not six. The signed halves and the count-ranked
            # share are already answered for this role by the role-scoped
            # position cut and by the ranked rows below; what only this table
            # can say is how many DISTINCT tokens the role's nats went to and
            # how concentrated they were.
            out[f"{head}/token/n_distinct"] = float((counts > 0).sum())
            if gross_all > 0:
                out[f"{head}/token/shift_gross_share"] = gross / gross_all
            per_tok = (eff_pos[rid] + eff_neg[rid]).abs()
            tot = float(per_tok.sum())
            if tot > 0:
                out[f"{head}/token/dkl_nats_abs_top{N}_share"] = (
                    float(torch.topk(per_tok, N).values.sum()) / tot
                )
        return out

    def top_tokens(self) -> list:
        """Ranked rows, in the schema the worker's decoder expects.

        Appended to the same report the other tables go to rather than given a
        file of its own: ``scope`` already distinguishes them, and a third file
        is a third thing a reader has to join.
        """
        n, eff_pos, eff_neg = self._cpu()
        N = min(self.top_n, self.vocab_size)
        rows = []
        for rid in range(self.n_roles):
            rname = ROLE_NAMES[rid]
            if int(n[rid].sum()) <= 0:
                continue
            net = eff_pos[rid] + eff_neg[rid]
            gross = eff_pos[rid] - eff_neg[rid]
            for ranked_by, series in (
                ("count", n[rid].to(torch.float64)),
                ("abs_effect", net.abs()),
            ):
                if float(series.sum()) <= 0:
                    continue
                vals, idx = torch.topk(series, N)
                for rank, (v, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                    if v <= 0:
                        break
                    rows.append({
                        "scope": f"role:{rname}", "ranked_by": ranked_by, "state": "__any__",
                        "rank": rank, "token_id": int(tok), "count": int(n[rid, tok]),
                        "effect_kind": EFFECT_KIND["position"],
                        "effect_net": float(net[tok]), "effect_gross": float(gross[tok]),
                    })
        return rows


# The four things a source can be doing at a candidate, relative to the on-task
# teacher. The existing state table sums the sources out and the existing source
# table sums the states out, so "what did the arm do when Search DISAGREED with
# AlfWorld's own teacher" -- which is the case the mechanism exists to arbitrate
# -- is in neither. Defined here rather than beside the accumulator that fills
# it, because the event sampler below needs the names and lives on this side of
# the import edge.
PAIR_STATES = ("agree", "conflict", "on_silent_source_active", "source_silent")


# One row per (candidate, SOURCE), against SignEventSamples' one row per
# candidate. The extra axis is the whole point: "what did Search bring to
# AlfWorld" needs Search's own probability, Search's own shift and Search's own
# alpha on the row, and a per-candidate row can only carry the min and max over
# whichever off-task teachers happened to speak.
PAIR_EVENT_INTS = (
    "token_id", "dst", "src", "pair_state", "state", "role", "turn",
    "position", "row_len", "is_sampled",
)
PAIR_EVENT_FLOATS = (
    # what the four models said, as probabilities
    "p_base", "p_on", "p_source", "p_student",
    # and the shifts, raw nats and RMS units, never mixed with the above
    "delta_on_raw", "delta_source_raw", "delta_on_std", "delta_source_std",
    # the mechanism at this candidate
    "alpha_source", "shared_evidence", "source_evidence",
    "pre_weight", "applied_weight",
    # what it cost and what it moved
    "teacher_kl", "source_attributed_kl_shift",
    "weighted_logit_push", "extra_logit_push",
    # THE ADDED PUSH, SPLIT BY WHAT CAUSED IT. extra_logit_push is (W - 1) g0,
    # and W - 1 is B_shared/mu + sum_m B_m/mu + (1/mu - 1) exactly, so the same
    # split carries through to the logit:
    #
    #     extra_push_sources_all + extra_push_shared + extra_push_normalizer
    #         == extra_logit_push
    #
    # holds on every row (test_the_event_push_columns_add_to_the_total). Without
    # it the dump can say "the arm moved this student token" and "Search
    # supplied evidence at that position", and cannot join them: extra_logit_push
    # is the whole mechanism's effect and is identical across the source rows of
    # one candidate. extra_push_source is THIS source's own share of it, which is
    # the column the sentence "Search raised this AlfWorld token" needs.
    #
    # The normaliser term is nobody's: mu is a whole-task divisor, so a position
    # with no evidence at all still moves by (1/mu - 1) g0. It gets its own
    # column rather than being spread over the sources, because attributing it
    # would credit a source for an effect it had no part in.
    "extra_push_source", "extra_push_sources_all",
    "extra_push_shared", "extra_push_normalizer",
    # and what happened to the rollout it sat in
    "advantage", "reward",
)


class PairEventSamples:
    """Individual (candidate, source) events, stratified so the rare ones survive.

    Two problems with a single global top-N, both of which this fixes.

    STRATIFICATION. A global ranking by ``|effect|`` is dominated by whichever
    ordered pair happens to be loudest, and the interesting event -- Search's
    evidence acting on an AlfWorld state where the two teachers DISAGREE -- is
    structurally a minority of a minority. Sampling per ``(dst, src, pair_state)``
    guarantees every cell that occurred at all is represented, which is the
    difference between "we looked and found none" and "we never looked".

    CROSS-RANK. ``SignEventSamples`` is rank-0 local, so what lands on disk is a
    sample of one shard -- unbiased, but ``world_size`` times smaller than a
    reader assumes, and for a cell that fires a handful of times a step that is
    the difference between a few examples and none. Here each rank selects its
    own fixed-size slate and they are ``all_gather``ed: the shape is decided by
    the config (groups x strata x per_group), never by batch content, so the
    collective is uniform. About 140 KB a rank at three tasks.

    Three strata per cell, because they answer different questions and the same
    row rarely tops two of them:

    ``top_shift``  the largest ``|source_attributed_kl_shift|`` -- where this
                   source moved the most of this task's KL budget.
    ``top_push``   the largest ``|extra_logit_push|`` -- where the weighting
                   changed a student logit the most. NOT the same rows: the
                   first is about the evidence, the second about the effect on
                   the student, and keeping both is what lets the write-up say
                   which token supplied the reason and which token moved.
    ``spread``     a deterministic hash sample, so the median event is
                   represented and a mechanism whose extremes are
                   unrepresentative is visible as such.
    """

    STRATA = ("top_shift", "top_push", "spread")
    _HASH = 2654435761

    def __init__(self, *, n_tasks: int, per_group: int = 4, context: int = 8, device=None):
        self.n_tasks = T = int(n_tasks)
        self.n_states = len(PAIR_STATES)
        self.n_groups = max(T * (T - 1), 1) * self.n_states
        self.per_group = int(per_group)
        self.context = int(context)
        self.device = device
        self._ints, self._floats, self._ctx, self._scores = {}, {}, {}, {}
        for s in self.STRATA:
            self._ints[s], self._floats[s], self._ctx[s], self._scores[s] = [], [], [], []
        self._seen = 0

    def _pair_index(self, dst, src):
        """(dst, src) -> 0..T*(T-1)-1, skipping the structurally empty diagonal."""
        return dst * (self.n_tasks - 1) + src - (src > dst).to(torch.long)

    def update(self, *, columns: dict, group: torch.Tensor, valid: torch.Tensor,
               context_ids: torch.Tensor, shift: torch.Tensor, push: torch.Tensor) -> None:
        """Take this micro-batch's slate.

        Args:
            columns: every name in :data:`PAIR_EVENT_INTS` and
                :data:`PAIR_EVENT_FLOATS`, each broadcastable to
                ``(bs, resp, k, n_off)``.
            group: (bs, resp, k, n_off) the cell index.
            valid: (bs, resp, k, n_off) bool.
            context_ids: (bs, resp, k, n_off, 2 * context + 1).
            shift: the ``top_shift`` ranking key.
            push: the ``top_push`` ranking key.

        Sync-free: ``per_group`` topk calls over a masked score, and small device
        tensors appended to a list. Nothing is read to the host until
        :meth:`rows`.
        """
        n = group.numel()
        idx = torch.arange(self._seen, self._seen + n, device=group.device, dtype=torch.long)
        self._seen += n
        g = group.reshape(-1)
        ok = valid.reshape(-1)
        ints = torch.stack(
            [columns[c].expand_as(group).reshape(-1).to(torch.long) for c in PAIR_EVENT_INTS],
            dim=-1,
        )
        floats = torch.stack(
            [columns[c].expand_as(group).reshape(-1).to(torch.float64) for c in PAIR_EVENT_FLOATS],
            dim=-1,
        )
        ctx = context_ids.reshape(n, -1)
        keys = {
            "top_shift": shift.reshape(-1).abs().to(torch.float64),
            "top_push": push.reshape(-1).abs().to(torch.float64),
            "spread": ((idx * self._HASH) % 2147483647).to(torch.float64),
        }
        take = min(self.per_group, n)
        for name, key in keys.items():
            # -1 sorts below every real score and is dropped at render time, so a
            # cell with fewer than per_group events keeps only what it had.
            for cell in range(self.n_groups):
                score = torch.where(ok & (g == cell), key, torch.full_like(key, -1.0))
                _v, sel = torch.topk(score, take)
                self._scores[name].append(score[sel])
                self._ints[name].append(ints[sel])
                self._floats[name].append(floats[sel])
                self._ctx[name].append(ctx[sel])

    def _slate(self):
        """This rank's final selection, as fixed-shape padded tensors."""
        cap = self.n_groups * self.per_group
        n_i, n_f = len(PAIR_EVENT_INTS), len(PAIR_EVENT_FLOATS)
        width = 2 * self.context + 1
        out = {}
        for name in self.STRATA:
            ints = torch.zeros((cap, n_i), dtype=torch.long)
            floats = torch.zeros((cap, n_f), dtype=torch.float64)
            ctx = torch.zeros((cap, width), dtype=torch.long)
            score = torch.full((cap,), -1.0, dtype=torch.float64)
            if self._scores[name]:
                s = torch.cat(self._scores[name])
                i_all = torch.cat(self._ints[name])
                f_all = torch.cat(self._floats[name])
                c_all = torch.cat(self._ctx[name])
                # Re-rank per cell over the concatenation, so the result is what
                # one pass over the whole mini-batch would have produced.
                cells = self._cell_of(i_all)
                for cell in range(self.n_groups):
                    pick = torch.where(cells == cell, s, torch.full_like(s, -1.0))
                    take = min(self.per_group, pick.numel())
                    v, sel = torch.topk(pick, take)
                    lo = cell * self.per_group
                    ints[lo : lo + take] = i_all[sel].to("cpu")
                    floats[lo : lo + take] = f_all[sel].to("cpu")
                    ctx[lo : lo + take] = c_all[sel].to("cpu")
                    # The MASKED score, not the original. topk on an all-masked
                    # cell still returns indices -- of rows belonging to other
                    # cells -- and writing their real scores here would let a
                    # cell that never fired hand its slots to a cell that fired
                    # a lot, which is exactly the per-cell guarantee this class
                    # exists to make.
                    score[lo : lo + take] = v.to("cpu")
            out[name] = (ints, floats, ctx, score)
        return out

    def _cell_of(self, ints: torch.Tensor) -> torch.Tensor:
        dst = ints[:, PAIR_EVENT_INTS.index("dst")]
        src = ints[:, PAIR_EVENT_INTS.index("src")]
        st = ints[:, PAIR_EVENT_INTS.index("pair_state")]
        pair = self._pair_index(dst.clamp(min=0), src.clamp(min=0))
        return pair.clamp(min=0) * self.n_states + st.clamp(min=0)

    def rows(self, task_names=None) -> list:
        """Gather every rank's slate, re-select per cell, and render on rank 0.

        The gather is FIXED SHAPE -- ``groups * per_group`` rows whatever the
        batch held -- so it is a collective on the config and cannot hang on a
        rank whose micro-batches happened to hold nothing.
        """
        dist_on = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if dist_on else 1
        rank = torch.distributed.get_rank() if dist_on else 0
        mine = self._slate()

        out = []
        for name in self.STRATA:
            ints, floats, ctx, score = mine[name]
            if dist_on and world > 1:
                dev = self.device or ints.device
                packed = [
                    torch.cat(
                        [ints.to(torch.float64), floats, ctx.to(torch.float64),
                         score.unsqueeze(-1)],
                        dim=-1,
                    ).to(dev)
                ]
                buf = [torch.zeros_like(packed[0]) for _ in range(world)]
                torch.distributed.all_gather(buf, packed[0])
                if rank != 0:
                    continue
                merged = torch.cat(buf, dim=0).to("cpu")
                n_i, n_f = len(PAIR_EVENT_INTS), len(PAIR_EVENT_FLOATS)
                w = 2 * self.context + 1
                ints = merged[:, :n_i].to(torch.long)
                floats = merged[:, n_i : n_i + n_f]
                ctx = merged[:, n_i + n_f : n_i + n_f + w].to(torch.long)
                score = merged[:, -1]
                cells = self._cell_of(ints)
                sel = []
                for cell in range(self.n_groups):
                    pick = torch.where(cells == cell, score, torch.full_like(score, -1.0))
                    take = min(self.per_group, pick.numel())
                    v, s = torch.topk(pick, take)
                    # Filter on the MASKED value, for the reason above: a cell
                    # with nothing in it must contribute nothing, not the best
                    # rows of whichever cell was loudest.
                    sel.append(s[v >= 0])
                if not sel:
                    continue
                sel = torch.cat(sel)
                ints, floats, ctx = ints[sel], floats[sel], ctx[sel]
            elif rank != 0:
                continue
            else:
                keep = score >= 0
                ints, floats, ctx = ints[keep], floats[keep], ctx[keep]

            name_of = lambda t: (
                task_names[t] if task_names and 0 <= t < len(task_names)
                else (None if t < 0 else f"task{t}")
            )
            for iv, fv, cv in zip(ints.tolist(), floats.tolist(), ctx.tolist()):
                row = {"table": "pair_event", "stratum": name}
                row.update(dict(zip(PAIR_EVENT_INTS, (int(x) for x in iv))))
                row.update(dict(zip(PAIR_EVENT_FLOATS, (float(x) for x in fv))))
                row["dst"] = name_of(row["dst"])
                row["src"] = name_of(row["src"])
                row["pair_state"] = PAIR_STATES[row["pair_state"]] if 0 <= row["pair_state"] < len(PAIR_STATES) else str(row["pair_state"])
                row["state"] = STATE_NAMES.get(row["state"], str(row["state"]))
                row["role"] = ROLE_NAMES.get(row["role"], str(row["role"]))
                row["context_ids"] = [int(x) for x in cv]
                out.append(row)
        return out


def turn_index(response_mask: torch.Tensor) -> torch.Tensor:
    """(bs, resp) which generated segment each position belongs to, 0-based.

    ``response_mask`` is the multi-turn loss mask: 1 on tokens the model
    produced, 0 on the environment's replies spliced in between. A turn is
    therefore a maximal run of ones, and its index is the running count of
    0 -> 1 transitions. Positions inside an environment reply carry the index of
    the turn they follow, which is what a reader wants -- the observation that
    preceded turn n+1 belongs with turn n's consequences.

    On a single-turn arm the mask is all ones and every position is turn 0,
    which is the correct answer rather than a degenerate one.
    """
    m = response_mask.to(torch.bool)
    prev = torch.zeros_like(m)
    prev[:, 1:] = m[:, :-1]
    starts = m & (~prev)
    return starts.to(torch.long).cumsum(dim=1).clamp(min=1) - 1


# --------------------------------------------------------------------------- #
# individual events, with the text around them
# --------------------------------------------------------------------------- #
# Ints and floats are kept in separate tensors because a token id must survive
# the round trip exactly and a float64 mantissa stops being able to promise that
# at 2^53 -- which a vocabulary of 151,936 is nowhere near, but a position
# encoded into the same tensor as a probability is a bug waiting for a bigger
# model. Column order is the schema; both lists are the schema's only definition.
EVENT_INTS = ("token_id", "task_id", "state", "role", "turn", "position", "row_len")
EVENT_FLOATS = (
    "p_base",       # pi_0 at the candidate: what the shared base said
    "p_on",         # the on-task teacher
    "p_student",    # the student, i.e. whether the edit has landed yet
    "p_off_lo",     # the least enthusiastic off-task teacher
    "p_off_hi",     # the most
    "weight",       # the candidate weight the table assigned
    "effect",       # candidate_effect: the post-normalisation change, signed
    "norm",         # Z (target) or the applied position weight (position)
    "teacher_kl",   # the position's per-token KL, before any weighting
    "reward",       # the row's episode score, or nan when the arm has none
    # The parameter-free arm's own quantity: delta / sigma, the teacher's move
    # away from the base in units of its OWN in-domain RMS. A SEPARATE column
    # from the four probabilities rather than a substitute for them -- a
    # standardized shift does not exponentiate to a probability, and a dump that
    # put one in p_on would report exp(delta_hat) under a name every reader
    # takes for pi_teacher(v). nan on an arm that has no standardization.
    "shift_on",
    "shift_off_lo",
    "shift_off_hi",
)


class SignEventSamples:
    """A bounded sample of individual candidates, with the tokens around them.

    Everything else in this module is an aggregate. Aggregates are what a claim
    is made of, but they cannot be read for a MECHANISM: "the weighting acts on
    the same forty tokens every step" and "it acts on <think>'s connectives"
    produce identical top-N tables, and only looking at instances separates them.
    This class is the instances -- one row per sampled candidate, carrying the
    four models' probabilities at it, the weight, the effect, where in the
    episode it sat and what was being written around it.

    Two strata, because either alone is misleading:

    ``top``     the largest ``|effect|`` seen. Where the mechanism is loudest,
                and the natural thing to quote -- but a table of extremes says
                nothing about the median event, and a mechanism whose extremes
                are unrepresentative is exactly the failure mode worth catching.
    ``spread``  a pseudo-random sample. The key is a multiplicative hash of a
                running index, not an RNG: a generator would either need its
                state synchronised across ranks or produce a sample nobody can
                reproduce, and the hash is deterministic given the batch order.

    RANK-0 LOCAL, unlike every other table here. The aggregates are all-reduced
    before rendering because a sum is a sum; a sample is not, and gathering
    variable-length selections across ranks to re-sample them would be a
    collective whose size depends on batch content. What lands on disk is
    therefore a sample of ONE rank's shard -- which is itself a random shard of
    the batch, so the sample is unbiased, but its size is world_size times
    smaller than a reader might assume. Said here because the file cannot say it.

    Sync-free in the micro-batch loop: each call does one ``topk`` per stratum
    and appends small device tensors to a list. Nothing is read to the host until
    :meth:`rows`, once per ``update_policy``.
    """

    STRATA = ("top", "spread")
    # Knuth's multiplicative constant. Any odd 32-bit multiplier with a
    # well-mixed bit pattern does; this one is only here to make the choice
    # explicit rather than magic.
    _HASH = 2654435761

    def __init__(self, *, capacity: int = 128, context: int = 8, device=None):
        self.capacity = int(capacity)
        self.context = int(context)
        self.device = device
        self._ints = {k: [] for k in self.STRATA}
        self._floats = {k: [] for k in self.STRATA}
        self._ctx = {k: [] for k in self.STRATA}
        self._scores = {k: [] for k in self.STRATA}
        self._seen = 0   # a python int: no device read, and stable across calls

    def update(
        self,
        *,
        support_ids: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        effect: torch.Tensor,
        on_task_logprob: torch.Tensor,
        off_task_logprobs: torch.Tensor,
        base_logprob: torch.Tensor,
        student_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        responses: torch.Tensor,
        norm: torch.Tensor,
        teacher_kl: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        roles: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        shift_on: Optional[torch.Tensor] = None,
        shift_off: Optional[torch.Tensor] = None,
    ) -> None:
        """Sample this micro-batch's candidates into both strata.

        Args:
            norm: (bs, resp) the per-position normaliser -- ``Z`` in target mode,
                the applied position weight in position mode. One column rather
                than two because a row already says which mode it came from
                through the arm it was dumped by, and two columns of which one is
                always empty is how a schema rots.
            roles: (bs, resp) from :func:`token_roles`, or None when the tag ids
                were not available -- the row then reports ``format`` for
                everything, which is honest: nothing was classified.
            reward: (bs,) the row's episode score, or None on an arm that has
                none. Absent becomes NaN in the column rather than 0.0, which is
                a score.
        """
        bs, resp, k = support_ids.shape
        n = bs * resp * k
        dev = support_ids.device
        valid = response_mask.to(torch.bool).unsqueeze(-1).expand_as(support_ids).reshape(-1)

        eff = effect.detach().to(torch.float64).reshape(-1)
        idx = torch.arange(self._seen, self._seen + n, device=dev, dtype=torch.long)
        self._seen += n
        scores = {
            # -1 sorts below every real score, so invalid entries are taken only
            # when there is nothing else -- and they are dropped at render time.
            "top": torch.where(valid, eff.abs(), torch.full_like(eff, -1.0)),
            "spread": torch.where(
                valid,
                ((idx * self._HASH) % 2147483647).to(torch.float64),
                torch.full_like(eff, -1.0),
            ),
        }

        p_off = off_task_logprobs.detach().to(torch.float32).exp()
        cols_f = [
            base_logprob.detach().to(torch.float32).exp(),
            on_task_logprob.detach().to(torch.float32).exp(),
            student_logprob.detach().to(torch.float32).exp(),
            p_off.min(dim=-1).values,
            p_off.max(dim=-1).values,
            weight.detach().to(torch.float32),
            # float64 all the way: this column is the ranking key, and rounding
            # it to float32 on the way out would make the value a row is chosen
            # for disagree with the value the row reports.
            effect.detach().to(torch.float64),
            norm.detach().to(torch.float32).unsqueeze(-1).expand(bs, resp, k),
            teacher_kl.detach().to(torch.float32).unsqueeze(-1).expand(bs, resp, k),
        ]
        rew = (
            torch.full((bs,), float("nan"), device=dev)
            if reward is None
            else reward.detach().to(torch.float32).reshape(-1)
        )
        cols_f.append(rew.view(bs, 1, 1).expand(bs, resp, k))
        # nan, not zero: an arm without a standardization has not measured this,
        # and zero is a value the column can legitimately take.
        nan_col = torch.full((bs, resp, k), float("nan"), device=dev)
        cols_f.append(nan_col if shift_on is None else shift_on.detach().to(torch.float32))
        if shift_off is None:
            cols_f.extend([nan_col, nan_col])
        else:
            s_off = shift_off.detach().to(torch.float32)
            cols_f.extend([s_off.min(dim=-1).values, s_off.max(dim=-1).values])
        flat_f = torch.stack([c.reshape(-1).to(torch.float64) for c in cols_f], dim=-1)

        role = (
            torch.zeros((bs, resp), dtype=torch.long, device=dev) if roles is None else roles.to(torch.long)
        )
        tid = (
            torch.full((bs,), -1, dtype=torch.long, device=dev)
            if task_ids is None
            else task_ids.reshape(-1).to(torch.long)
        )
        row_len = response_mask.to(torch.long).sum(dim=1)
        pos = torch.arange(resp, device=dev).view(1, resp).expand(bs, resp)
        cols_i = [
            support_ids.to(torch.long),
            tid.view(bs, 1, 1).expand(bs, resp, k),
            state.to(torch.long),
            role.unsqueeze(-1).expand(bs, resp, k),
            turn_index(response_mask).unsqueeze(-1).expand(bs, resp, k),
            pos.unsqueeze(-1).expand(bs, resp, k),
            row_len.view(bs, 1, 1).expand(bs, resp, k),
        ]
        flat_i = torch.stack([c.reshape(-1) for c in cols_i], dim=-1)

        # The window around each event, gathered once per stratum below. Built
        # here so the clamped index arithmetic is written once.
        w = 2 * self.context + 1
        take = min(self.capacity, n)
        for name, score in scores.items():
            vals, sel = torch.topk(score, take)
            self._scores[name].append(vals)
            self._ints[name].append(flat_i[sel])
            self._floats[name].append(flat_f[sel])
            if responses.size(1) == resp:
                # Advanced indexing rather than gather: the row index and the
                # column window are both per-EVENT here, and gather would want
                # an index shaped like `responses` instead.
                row = torch.div(sel, resp * k, rounding_mode="floor")          # (take,)
                col = torch.div(sel, k, rounding_mode="floor") % resp          # (take,)
                span = col.unsqueeze(-1) + torch.arange(
                    -self.context, self.context + 1, device=dev
                )                                                              # (take, w)
                # Clamped, not padded: at the ends of a response the window
                # repeats the edge token, which reads as an edge in the dump and
                # costs no sentinel value that a decoder would have to know.
                self._ctx[name].append(responses[row.unsqueeze(-1), span.clamp(min=0, max=resp - 1)])
            else:
                self._ctx[name].append(torch.zeros((take, w), dtype=responses.dtype, device=dev))

    def rows(self, task_names=None) -> list:
        """Merge the per-micro-batch selections and read them once.

        The merge re-runs the same ``topk`` over the concatenation, so the result
        is what a single pass over the whole mini-batch would have produced --
        for ``top`` exactly, and for ``spread`` exactly as well, since the hash
        is a function of a running index and not of the split.
        """
        out = []
        for name in self.STRATA:
            if not self._scores[name]:
                continue
            scores = torch.cat(self._scores[name])
            ints = torch.cat(self._ints[name])
            floats = torch.cat(self._floats[name])
            ctx = torch.cat(self._ctx[name])
            take = min(self.capacity, scores.numel())
            _v, sel = torch.topk(scores, take)
            keep = scores[sel] >= 0
            sel = sel[keep]
            ints_c = ints[sel].to("cpu").tolist()
            floats_c = floats[sel].to("cpu").tolist()
            ctx_c = ctx[sel].to("cpu").tolist()
            for rank, (iv, fv, cv) in enumerate(zip(ints_c, floats_c, ctx_c)):
                row = {"table": "event", "stratum": name, "rank": rank}
                row.update(dict(zip(EVENT_INTS, (int(x) for x in iv))))
                row.update(dict(zip(EVENT_FLOATS, (float(x) for x in fv))))
                t = row.pop("task_id")
                row["task"] = (
                    task_names[t] if task_names and 0 <= t < len(task_names) else (None if t < 0 else f"task{t}")
                )
                row["state"] = STATE_NAMES.get(row["state"], str(row["state"]))
                row["role"] = ROLE_NAMES.get(row["role"], str(row["role"]))
                row["context_ids"] = [int(x) for x in cv]
                out.append(row)
        return out
