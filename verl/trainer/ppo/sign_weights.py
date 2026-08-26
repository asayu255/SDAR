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
    One scalar per token position multiplies the whole per-token KL. The weight
    is a positive constant with respect to the student, so the minimiser of the
    weighted loss is still the on-task teacher's distribution: this reallocates
    *how hard* each position is learned and cannot move the target. Direction is
    meaningless here -- a KL term has no sign to follow -- so agreement counts
    the same whether the teachers agreed to raise a token or to lower it.

``target``
    The on-task teacher's distribution itself is reweighted candidate-wise and
    renormalised, and the student is distilled to *that*. Here the weight
    multiplies a PROBABILITY, so direction is the whole point: tokens the
    teachers agreed to raise get more mass (``agree_weight`` > 1) and tokens they
    agreed to lower get less (``agree_neg_weight`` < 1). Weighting both
    agreements up -- which is right for ``position`` -- would raise the target
    probability of tokens every teacher agreed to suppress, i.e. undo the very
    edit the agreement is evidence for.

Note what ``target`` deliberately does NOT do: multiply the individual terms of
the KL sum. The distillation loss here is a *reverse* KL,
``sum_v p_student(v) (log p_student(v) - log p_teacher(v))``, whose per-candidate
term is a cost the student pays for its own mass at ``v``. Scaling that term up
makes the student put *less* mass on ``v`` -- the exact opposite of "both
teachers like this token, learn it harder". Reweighting the target instead gets
the intended direction and keeps the loss a proper divergence (non-negative,
zero only at the target), which the term-wise product is not.
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
    * ``dw`` -- the signed sum of ``(w - 1) * p``, i.e. the token's own
      contribution to ``Z - 1``. This is the one ranked list that is a statement
      about the OBJECTIVE rather than about the diagnosis: it is exactly how much
      of the target's rewrite this token is responsible for, in the same
      decomposition ``target_kl`` and ``inv_z`` are read from. Summing it over
      the vocabulary reproduces ``Z - 1``.

    Accumulated dense and sync-free. The alternative -- ``torch.unique`` per
    micro-batch, keyed into a Python dict -- reads the device inside the
    micro-batch loop thousands of times a step, which is the run-ahead this
    actor's whole design protects. A dense ``index_add_`` costs one kernel and
    (scopes * states * V) of memory: 4 * 7 * 151,936 is 34 MB at int64, next to
    a teacher output projection of 622 MB.

    Scope 0 is the pooled batch; scope ``1 + t`` is task ``t``. Rows whose task
    is unknown contribute to the pooled scope only.
    """

    def __init__(self, *, vocab_size: int, n_tasks: int, device, top_n: int = 64):
        self.vocab_size = int(vocab_size)
        self.n_states = len(STATE_NAMES)
        self.n_scopes = 1 + int(n_tasks)
        self.top_n = int(top_n)
        cells = self.n_scopes * self.n_states * self.vocab_size
        self.n = torch.zeros(cells, dtype=torch.int64, device=device)
        self.mass = torch.zeros(cells, dtype=torch.float32, device=device)
        self.dw = torch.zeros(self.n_scopes * self.vocab_size, dtype=torch.float32, device=device)
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
        p = on_task_logprob.detach().to(torch.float32).exp().reshape(-1)
        dw = (weight.detach().to(torch.float32) - 1.0).reshape(-1) * p

        v_f = valid.to(torch.float32)
        v_i = valid.to(torch.int64)
        V, S = self.vocab_size, self.n_states

        flat = st * V + ids  # scope 0 starts at offset 0
        self.n.index_add_(0, flat, v_i)
        self.mass.index_add_(0, flat, p * v_f)
        self.dw.index_add_(0, ids, dw * v_f)

        if task_ids is None:
            return
        t = task_ids.reshape(-1, 1, 1).expand_as(state).reshape(-1).to(torch.long)
        known = (t >= 0) & valid
        scope = t.clamp(min=0) + 1
        flat_t = (scope * S + st) * V + ids
        self.n.index_add_(0, flat_t, known.to(torch.int64))
        self.mass.index_add_(0, flat_t, p * known.to(torch.float32))
        self.dw.index_add_(0, scope * V + ids, dw * known.to(torch.float32))

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
        for t in (self.n, self.mass, self.dw):
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
                self.dw.detach().to("cpu").view(self.n_scopes, V),
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
        """The shape of the token distribution, as numbers a run can be plotted by.

        ``n_distinct`` and ``top_share`` together separate the two mechanisms in
        the class docstring: a small stable set gives few distinct tokens and a
        high top-N share, a broad tendency gives the opposite. Neither is
        derivable from the existing ``frac_*``.

        ``dw_pos_sum`` / ``dw_neg_sum`` are the two halves of ``Z - 1``. They are
        reported separately because the whole point of ``inv_z`` is that they do
        NOT cancel -- reporting only the residual hides how much was pushed each
        way to produce it.
        """
        out = {}
        n, _mass, dw = self._cpu()
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
            d = dw[scope]
            pos, neg = float(d.clamp(min=0).sum()), float(d.clamp(max=0).sum())
            if pos == 0.0 and neg == 0.0:
                continue
            out[f"{head}/token/dw_pos_sum"] = pos
            out[f"{head}/token/dw_neg_sum"] = neg
            absd = d.abs()
            tot_abs = float(absd.sum())
            if tot_abs > 0:
                out[f"{head}/token/dw_abs_top{N}_share"] = float(torch.topk(absd, N).values.sum()) / tot_abs
        return out

    def top_tokens(self, task_names=None) -> list:
        """The ranked lists themselves, as plain rows for the caller to decode.

        Token ids rather than strings: this runs inside the actor, which has no
        tokenizer. The worker that owns one turns them into text.

        Two rankings per scope, because they answer different questions. By
        ``count`` within each acted state: which tokens do the teachers keep
        agreeing (or disagreeing) about -- the shared structure, named. By
        ``abs_dw`` pooled: which tokens actually moved the target. A token can top
        one list and be absent from the other, and that gap is itself the finding
        -- a token reinforced constantly at negligible probability is shared
        structure the objective never sees.
        """
        rows = []
        n, mass, dw = self._cpu()
        N = min(self.top_n, self.vocab_size)
        for scope in range(self.n_scopes):
            scope_name = self._scope_name(scope, task_names) or "__pooled__"
            n_any = n[scope].sum(0)
            mass_any = mass[scope].sum(0)
            for sid in ACTED_STATES:
                counts = n[scope, sid]
                if int(counts.sum()) <= 0:
                    continue
                masses = mass[scope, sid]
                vals, idx = torch.topk(counts, N)
                for rank, (c, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                    if c <= 0:
                        break
                    rows.append({
                        "scope": scope_name,
                        "ranked_by": "count",
                        "state": STATE_NAMES[sid],
                        "rank": rank,
                        "token_id": int(tok),
                        "count": int(c),
                        "mass": float(masses[tok]),
                        "dw": float(dw[scope, tok]),
                    })
            absd = dw[scope].abs()
            if float(absd.sum()) <= 0:
                continue
            vals, idx = torch.topk(absd, N)
            for rank, (a, tok) in enumerate(zip(vals.tolist(), idx.tolist())):
                if a <= 0:
                    break
                rows.append({
                    "scope": scope_name,
                    "ranked_by": "abs_dw",
                    # Pooled over states on purpose: dw is the token's net effect,
                    # and a token that is reinforced in some rows and suppressed in
                    # others has one net effect and no single state.
                    "state": "__any__",
                    "rank": rank,
                    "token_id": int(tok),
                    "count": int(n_any[tok]),
                    "mass": float(mass_any[tok]),
                    "dw": float(dw[scope, tok]),
                })
        return rows


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
)


def rewrite_decomposition_terms(
    *,
    student_logprob: torch.Tensor,
    on_task_logprob: torch.Tensor,
    base_logprob: torch.Tensor,
    candidate_weight: torch.Tensor,
    teacher_kl: torch.Tensor,
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
