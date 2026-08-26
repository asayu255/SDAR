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
