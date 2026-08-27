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
"""Parameter-free cross-teacher weighting of the on-task OPD KL.

The multitask arms distil each sample from the teacher of its own task and never
consult the other two, so nothing in the loss can carry what task A's RL learned
into task B's states. :mod:`verl.trainer.ppo.sign_weights` built the first
signal that could -- whether the off-task teachers agree with the on-task one
about a candidate -- but spent it through a hand-set table (``agree_weight``
1.25, ``deadzone`` 0.1 nats). Every number in that table is a knob that decides
how strong the mechanism is, which makes "does cross-teacher structure help"
inseparable from "was 1.25 the right number".

This module answers the same question with no such knob. It produces exactly one
thing: a positive scalar ``W`` per token position that multiplies the whole
per-token KL,

    L_OPD = Agg( W[i,t] * KL(pi_student || pi_on_task_teacher)[i,t] )

and NOTHING else. The teacher distribution is untouched, so ``W`` cannot move
what the student converges to -- at a position weighted by ``W`` every student
logit's gradient is the unweighted one times ``W``, direction included. What it
reallocates is EFFORT: which positions inside a task get more of that task's OPD
budget. Read ``sign_weights``' module docstring for why the two other things one
could do with the signal (rewriting the teacher's probabilities, weighting the
individual terms of the KL sum) are not done here.

The chain, and where each link's scale comes from rather than from a config:

``delta``      ``log pi_m - log pi_0``, the base-relative policy shift. Every
               teacher is a single-task RL fine-tune of one shared base, so this
               is what that task's RL wrote into the model at this candidate.
``delta_hat``  ``delta_m / sigma_{m<-m}``: each teacher divided by ITS OWN
               in-domain RMS shift, accumulated over the run. The teachers were
               trained at KL coefficients differing 10x and drifted 3.7x, so raw
               nats are not on a common footing; one RMS of the teacher's own
               typical shift is. Crucially the denominator is the DIAGONAL --
               teacher m measured on task m's states -- not ``sigma_{d<-m}``.
               Dividing by the destination-conditioned RMS would stretch the
               noise of a teacher that barely moves out of domain up to one full
               unit, which is the very thing the deadzone used to suppress.
``c_ev``       corroboration: the off-task teachers' unanimous minimum. A
               continuous min rather than a sign test, so a shift that is nearly
               zero contributes nearly nothing and no deadzone is needed.
``alpha``      per ordered task pair, the correlation between that source's
               RESIDUAL support for the tokens the student actually generated
               and the GRPO advantage of the trajectory it generated them in.
               Rectified at zero: a source that anti-correlates is vetoed, not
               inverted.
``e``          ``|c_ev| + sum_m alpha_m |delta_hat_m|``, the candidate evidence.
``W~``         ``1 + sum_v p_teacher(v) e(v)``, the position's raw weight.
``W``          ``W~ / mu_d``, where ``mu_d`` is the PREVIOUS step's KL-weighted
               per-task mean. That divisor is what keeps this a redistribution
               instead of a larger ``teacher_kl_loss_coef``.

Two properties of ``e`` are load-bearing and both are pinned by tests.

**It is monotone in corroboration.** An earlier draft used
``|c| + sum_m alpha_m |delta_hat_m - c|``, subtracting the common part from each
source. That is right for a target rewrite, where double-adding a shift in log
space really would double-count probability mass, and wrong here: agreement is
credited once and debited ``n_off`` times, so for ``alpha > 1/n_off`` a
CONFLICTING position scores HIGHER than an agreeing one with the same shift
magnitudes. At ``alpha=1`` the whole thing collapses to
``sum|delta_hat| - (n_off-1)|c|``, i.e. corroboration is penalised. ``W`` is a
dimensionless effort scalar with no conservation law to respect, so the
subtraction buys nothing; dropping it makes
``e(off-task unanimous) - e(off-task split) = |c_ev| >= 0`` hold for every
``alpha`` in [0, 1].

**Corroboration is measured among the OFF-TASK teachers only.** Including the
on-task teacher in the minimum -- ``c = sign(d) min_j |delta_hat_j|`` over
``{d} u off`` -- caps the bonus at the on-task teacher's own shift, and the
on-task teacher is silent at 64% of teacher mass (the shipped run's
``neutral_on_task_silent`` state). The corroboration channel would then be dead
across two thirds of the mass this arm is trying to say something about. Note
what that choice does and does not buy: flipping an OFF-TASK sign breaks the
unanimity and costs ``|c_ev|``, but flipping the ON-TASK sign does not, so a
position where the other two tasks unanimously oppose the on-task teacher scores
the same as one where they agree with it. That is deliberate -- a KL term has no
direction to disagree with, and "both other tasks moved decisively here" is a
statement about the position, not about who was right -- but it is exactly why
``kl_shift_by_state`` is a required metric rather than an optional one. The
signed, on-task-inclusive ``c`` is still computed, and is what the residual, the
reliability estimate and the attribution tables are built from.
"""

import math
from typing import Optional

import torch

__all__ = [
    "compute_raw_policy_shifts",
    "tail_logprob",
    "CumulativePolicyShiftRMS",
    "standardize_policy_shifts",
    "decompose_common_residual",
    "candidate_kl_evidence",
    "position_pre_weight",
    "PreviousStepTaskKLWeightedMean",
    "group_center",
    "ADV_MOMENTS",
    "AdvantageReliabilityStats",
]


# --------------------------------------------------------------------------- #
# policy shifts
# --------------------------------------------------------------------------- #
def tail_logprob(logprob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``log(1 - sum_v exp(logprob))`` over the support's complement.

    The support is a top-k of the FULL vocabulary's log-softmax, so what is left
    over is real probability mass and not a rounding error. Treating it as one
    lumped category is what makes every quantity here a proper expectation over
    a distribution that sums to 1 rather than over a truncation that does not.

    ``eps`` is numerical safety only -- it binds when the top-k covers the whole
    distribution to float precision -- and is never a knob on the mechanism.
    """
    tail = (1.0 - logprob.detach().to(torch.float32).exp().sum(dim=-1)).clamp(min=eps, max=1.0)
    return tail.log()


def compute_raw_policy_shifts(
    *,
    on_task_logprob: torch.Tensor,
    off_task_logprobs: torch.Tensor,
    base_logprob: torch.Tensor,
    eps: float = 1e-8,
) -> dict:
    """``delta = log pi_teacher - log pi_base`` on the support and on the tail.

    Args:
        on_task_logprob: (bs, resp, k) the row's own teacher at the support.
        off_task_logprobs: (bs, resp, k, n_off) the other teachers at the same ids.
        base_logprob: (bs, resp, k) the shared base policy at those ids.

    Returns:
        ``{"on", "off", "tail_on", "tail_off"}``. The tail entries are
        ``log p_teacher(tail) - log p_base(tail)`` over the SAME support's
        complement, which is what makes them comparable to the per-candidate
        shifts and legitimate to put in the same expectation.

    Everything is detached: the weight this feeds is a constant with respect to
    the student, and that is the property that keeps the KL's gradient direction
    untouched.
    """
    on = on_task_logprob.detach()
    off = off_task_logprobs.detach()
    base = base_logprob.detach()
    tail_base = tail_logprob(base, eps)
    return {
        "on": on - base,
        "off": off - base.unsqueeze(-1),
        "tail_on": tail_logprob(on, eps) - tail_base,
        "tail_off": torch.stack(
            [tail_logprob(off[..., c], eps) for c in range(off.size(-1))], dim=-1
        ) - tail_base.unsqueeze(-1),
    }


# --------------------------------------------------------------------------- #
# the cumulative RMS matrix
# --------------------------------------------------------------------------- #
class CumulativePolicyShiftRMS:
    """Each teacher's typical shift magnitude, per ordered (destination, teacher).

    Cell ``(d, m)`` is the running second moment of teacher ``m``'s
    base-relative shift, measured on the states task ``d``'s rollouts visit and
    weighted by the STUDENT's probability there:

        Q[d, m] += sum_positions [ sum_v p_student(v) delta_m(v)^2
                                   + p_student(tail) delta_m(tail)^2 ]
        N[d]    += sum_positions 1
        sigma[d, m] = sqrt(Q[d, m] / N[d])

    Weighted by the student rather than counted per slot because the support is
    a top-k: twenty candidates the student has all but ruled out would otherwise
    swamp the one it is about to emit. The tail is in the sum for the same
    reason it is in the KL -- leaving it out would make the RMS a statistic of a
    truncation.

    ONLY THE DIAGONAL SETS A SCALE. ``sigma[m, m]`` is teacher ``m`` measured
    where ``m`` operates, i.e. one unit of "typical for that teacher", and that
    is what :func:`standardize_policy_shifts` divides by. The off-diagonal is
    accumulated anyway and reported as ``off_to_in_domain_ratio``: it is the
    direct measurement of how much less a teacher moves out of its own domain,
    which is a finding in its own right and is precisely the signal that
    dividing by ``sigma[d, m]`` would erase.

    float64 and ``index_add_`` throughout, read once per ``update_policy``: a
    host sync inside the micro-batch loop is the run-ahead this actor's whole
    design protects.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.q = torch.zeros(T * T, dtype=torch.float64, device=device)
        self.n = torch.zeros(T, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(
        self,
        *,
        shifts: dict,
        student_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: torch.Tensor,
        off_plane_tasks: torch.Tensor,
        eps: float = 1e-8,
    ) -> None:
        """Fold one micro-batch in.

        Args:
            shifts: the mapping :func:`compute_raw_policy_shifts` returns.
            student_logprob: (bs, resp, k) the student at the support. The
                expectation's measure, so it is the student's and not a
                teacher's.
            task_ids: (bs,) destination task per row; negative rows (padding, or
                an untagged row) reach no cell.
            off_plane_tasks: (bs, n_off) which task each off-task plane holds.

        The validity mask is folded into the VALUES, never into the indices:
        selecting the valid rows out first would need their count on the host.
        """
        self._cpu_cache = None
        T = self.n_tasks
        p = student_logprob.detach().to(torch.float64).exp()          # (bs, resp, k)
        p_tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0, max=1.0)        # (bs, resp)
        mask = response_mask.to(torch.float64)                        # (bs, resp)

        d_on = shifts["on"].to(torch.float64)
        d_off = shifts["off"].to(torch.float64)
        t_on = shifts["tail_on"].to(torch.float64)
        t_off = shifts["tail_off"].to(torch.float64)

        # (bs, resp) and (bs, resp, n_off): the per-position second moment.
        m_on = (p * d_on.square()).sum(dim=-1) + p_tail * t_on.square()
        m_off = (p.unsqueeze(-1) * d_off.square()).sum(dim=-2) + p_tail.unsqueeze(-1) * t_off.square()

        dst = task_ids.reshape(-1).to(torch.long)
        row_ok = (dst >= 0).to(torch.float64)
        dst_c = dst.clamp(min=0)

        # Diagonal: the row's own teacher, on its own task's states.
        self.q.index_add_(0, dst_c * T + dst_c, (m_on * mask).sum(dim=1) * row_ok)
        self.n.index_add_(0, dst_c, (mask.sum(dim=1)) * row_ok)

        # Off-diagonal: each plane's teacher on THIS destination's states.
        for c in range(d_off.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = row_ok * (src >= 0).to(torch.float64)
            self.q.index_add_(0, dst_c * T + src.clamp(min=0), (m_off[..., c] * mask).sum(dim=1) * ok)

    def all_reduce(self) -> None:
        """Sum across the DP group. Unconditional: gated on config, never on data."""
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for t in (self.q, self.n):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            T = self.n_tasks
            self._cpu_cache = (
                self.q.detach().to("cpu").view(T, T),
                self.n.detach().to("cpu"),
            )
        return self._cpu_cache

    def snapshot(self) -> dict:
        """``{"sigma": (T, T), "valid": (T, T), "n": (T,)}`` on the host.

        A cell is valid only with positive count, positive sigma and a finite
        value. An invalid cell is never patched with an epsilon: a denominator
        invented to avoid a division is a transfer strength nobody chose.
        """
        q, n = self._cpu()
        denom = n.clamp(min=1.0).unsqueeze(-1)
        sigma = (q / denom).sqrt()
        valid = (n > 0).unsqueeze(-1) & (sigma > 0) & torch.isfinite(sigma)
        return {"sigma": sigma, "valid": valid, "n": n}

    def diagonal(self, snapshot: Optional[dict] = None):
        """``(diag, diag_valid)`` -- the only part the training weight reads.

        Availability is a property of this GLOBAL, all-reduced snapshot and of
        nothing else. A rank must not decide it from the tasks its own
        micro-batch happens to hold, or two ranks take different branches and
        the collectives below them stop lining up.
        """
        snap = self.snapshot() if snapshot is None else snapshot
        idx = torch.arange(self.n_tasks)
        return snap["sigma"][idx, idx].clone(), snap["valid"][idx, idx].clone()

    def state_dict(self) -> dict:
        return {"n_tasks": self.n_tasks, "q": self.q.detach().to("cpu"), "n": self.n.detach().to("cpu")}

    def load_state_dict(self, state: dict) -> None:
        assert int(state["n_tasks"]) == self.n_tasks, (
            f"task count changed across resume: {state['n_tasks']} -> {self.n_tasks}; "
            "the RMS matrix is indexed by task and cannot be reinterpreted"
        )
        self.q.copy_(state["q"].to(self.q.device, self.q.dtype))
        self.n.copy_(state["n"].to(self.n.device, self.n.dtype))
        self._cpu_cache = None


# --------------------------------------------------------------------------- #
# standardisation and decomposition
# --------------------------------------------------------------------------- #
def standardize_policy_shifts(
    *,
    shifts: dict,
    diag: torch.Tensor,
    diag_valid: torch.Tensor,
    task_ids: torch.Tensor,
    off_plane_tasks: torch.Tensor,
) -> dict:
    """Divide every teacher by its OWN in-domain RMS.

    Args:
        diag: (n_tasks,) ``sigma[m, m]`` from :meth:`CumulativePolicyShiftRMS.diagonal`.
        diag_valid: (n_tasks,) bool, from the same global snapshot.

    Returns:
        ``{"on", "off", "row_available"}``. ``row_available`` is (bs,) and false
        for a row whose own teacher or any of whose sources has no usable
        diagonal yet; the caller leaves those positions at ``W = 1`` rather than
        weighting them from a scale it does not have.

    The result is dimensionless: "how many of that teacher's own typical units
    did it move here". A teacher that barely moves out of its domain keeps a
    small standardized shift, which is the whole point of using the diagonal --
    see the module docstring.
    """
    dst = task_ids.reshape(-1).to(torch.long)
    src = off_plane_tasks.to(torch.long)                              # (bs, n_off)
    dev = shifts["on"].device
    d = diag.to(dev, torch.float32)
    v = diag_valid.to(dev)

    # Clamp the INDEX and carry the invalidity in the mask: a negative task id is
    # legitimate (padding) and must not index anything.
    s_on = d[dst.clamp(min=0)].view(-1, 1, 1)
    s_off = d[src.clamp(min=0)].view(src.size(0), 1, 1, src.size(1))
    ok_on = v[dst.clamp(min=0)] & (dst >= 0)
    ok_off = v[src.clamp(min=0)] & (src >= 0)

    # A guard on the DIVISOR only. Rows it fires on are excluded by
    # row_available, so the value it produces is never read; without it the
    # unread lane would still be an inf and would poison a later reduction.
    return {
        "on": shifts["on"].to(torch.float32) / s_on.clamp(min=1e-12),
        "off": shifts["off"].to(torch.float32) / s_off.clamp(min=1e-12),
        "row_available": ok_on & ok_off.all(dim=-1),
    }


def decompose_common_residual(*, hat_on: torch.Tensor, hat_off: torch.Tensor) -> dict:
    """Corroboration and residual, from the standardized shifts.

    Returns three things, and they are NOT interchangeable:

    ``common``     ``1[all teachers incl. the on-task one share a sign] *
                   sign(hat_on) * min_j |hat_j|``. Signed, on-task inclusive,
                   and bounded by ``|hat_on|``. This is the one the residual is
                   defined against, so it is what the reliability estimate and
                   the attribution tables see.
    ``common_ev``  ``1[the OFF-TASK teachers share a sign] * min_m |hat_m|``,
                   the corroboration bonus the KL evidence uses. Off-task only,
                   because the on-task teacher is silent at 64% of teacher mass
                   and including it would kill the channel across two thirds of
                   what this arm measures. Zero when there is fewer than one
                   other voice to corroborate with -- one teacher agreeing with
                   itself is not corroboration.
    ``residual``   ``hat_m - common``. What is left of source ``m`` once the
                   part every teacher shares is taken out; the reliability
                   correlation is measured on THIS so that generally-good
                   tokens cannot inflate a single source's credit.

    No deadzone. The minimum self-attenuates -- a shift near zero drags the
    whole corroboration to near zero whatever its sign does -- so noise costs a
    small number instead of tripping a fixed gate.
    """
    n_off = hat_off.size(-1)
    sign_on = torch.sign(hat_on)
    sign_off = torch.sign(hat_off)

    # Every teacher shares the on-task sign. The j == d term is hat_on^2 > 0,
    # i.e. "the on-task teacher said something at all".
    all_same = (sign_off == sign_on.unsqueeze(-1)).all(dim=-1) & (sign_on != 0)
    min_all = torch.minimum(hat_on.abs(), hat_off.abs().amin(dim=-1))
    common = torch.where(all_same, sign_on * min_all, torch.zeros_like(hat_on))

    if n_off >= 2:
        first = sign_off[..., :1]
        off_same = (sign_off == first).all(dim=-1) & (first.squeeze(-1) != 0)
        common_ev = torch.where(off_same, hat_off.abs().amin(dim=-1), torch.zeros_like(hat_on))
    else:
        common_ev = torch.zeros_like(hat_on)

    return {"common": common, "common_ev": common_ev, "residual": hat_off - common.unsqueeze(-1)}


def candidate_kl_evidence(
    *,
    common_ev: torch.Tensor,
    hat_off: torch.Tensor,
    source_alpha: torch.Tensor,
) -> torch.Tensor:
    """``e = |c_ev| + sum_m alpha_m |hat_delta_m|``. (bs, resp, k), non-negative.

    Args:
        common_ev: (bs, resp, k) from :func:`decompose_common_residual`.
        hat_off: (bs, resp, k, n_off) standardized source shifts. The FULL
            shift, not the residual -- see the module docstring for why
            subtracting the common part here makes corroboration score lower
            than conflict once ``alpha`` passes ``1/n_off``.
        source_alpha: (bs, n_off) or broadcastable, the reliability of each
            plane's teacher for this row's destination, in [0, 1].

    The residual is deliberately absent from this signature. ``alpha`` is
    ESTIMATED from the residual and APPLIED to the full shift, and keeping the
    residual out of the call is what stops the two from being confused at a call
    site.
    """
    a = source_alpha.to(hat_off.dtype)
    while a.dim() < hat_off.dim():
        a = a.unsqueeze(1)
    return common_ev.abs() + (a * hat_off.abs()).sum(dim=-1)


def position_pre_weight(*, evidence: torch.Tensor, on_task_logprob: torch.Tensor) -> torch.Tensor:
    """``W~ = sum_v p_teacher(v) [1 + e(v)] + p_teacher(tail)``, i.e. ``1 + E[e]``.

    Collapsed to the second form because the two are algebraically identical
    once the tail is the support's complement, and the second cannot drift from
    1 by float error in the way summing k+1 probabilities can.

    The candidates do not matter equally: the on-task teacher's own probability
    says how much of the KL each one accounts for, so evidence at a token the
    teacher has all but ruled out moves the position barely at all. The tail
    enters at evidence 0 -- no teacher was read there, so there is nothing to
    corroborate -- which means a position whose support covers little mass is
    modulated correspondingly little.

    The ``1`` is a multiplicative identity, not a coefficient: with no evidence
    the position's KL is exactly what it was.
    """
    p = on_task_logprob.detach().to(torch.float32).exp()
    return 1.0 + (p * evidence).sum(dim=-1)


# --------------------------------------------------------------------------- #
# the previous step's normaliser
# --------------------------------------------------------------------------- #
class PreviousStepTaskKLWeightedMean:
    """Per task, the KL-WEIGHTED mean of the raw position weight.

        mu_d = sum mask * w_row * W~ * D  /  sum mask * w_row * D

    ``W~`` is at least 1 everywhere, so applying it unnormalised would be
    indistinguishable from raising ``teacher_kl_loss_coef``: the arm would have
    distilled harder for reasons that have nothing to do with redistribution.
    Dividing by ``mu_d`` removes exactly that and leaves the reallocation, which
    is the thing being tested.

    KL-weighted, not a plain mean, and that is the whole point of this class.
    The invariant the arm claims is that it does not change how much OPD each
    task gets -- and how much it gets is ``sum W*D``, not ``sum W``. Those two
    agree only when the weight and the KL are uncorrelated, which is precisely
    what this arm hopes is false: evidence is large where the teachers moved,
    and the student tends to be further from its teacher there. With the plain
    mean a run can show ``w_mean = 1.000`` and still have multiplied its total
    teacher KL by 1.02; with this one, on the snapshot it was built from,

        sum W*D / sum D == 1

    holds by construction. ``w_mean`` is then NOT 1 in general, and does not
    need to be: it is a diagnostic of the W-D correlation, and ``kl_scale`` is
    the invariant.

    ``D`` IS ALWAYS THE UNWEIGHTED KL -- the raw ``topk_kl_per_token`` output,
    read before ``teacher_kld *= W``. Feeding the weighted one back in makes
    ``mu`` compose with itself every step and drift; step 1 runs at ``W = 1`` so
    it is correct exactly once, and the damage starts at step 2 where no metric
    is looking.

    Weighted by the same per-row weights the loss aggregation uses, so the
    invariant holds for the quantity that actually enters the objective and not
    for an unweighted stand-in.

    By the PREVIOUS step's value, never this one's: the weights exist only
    inside the training forward, which sees one micro-batch at a time, and
    normalising by a micro-batch's own mean would make the objective depend on
    how the batch was split.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = int(n_tasks)
        self.num = torch.zeros(self.n_tasks, dtype=torch.float64, device=device)
        self.den = torch.zeros(self.n_tasks, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(
        self,
        *,
        pre_weight: torch.Tensor,
        teacher_kl: torch.Tensor,
        response_mask: torch.Tensor,
        task_ids: torch.Tensor,
        row_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Fold one micro-batch in.

        Args:
            pre_weight: (bs, resp) ``W~``, BEFORE normalisation.
            teacher_kl: (bs, resp) the per-token KL BEFORE the weight multiplies
                it. Passing the weighted one is the failure this class's
                docstring is about.
            row_weights: (bs,) the same per-row weights the loss aggregation
                applies, or None for the token-mean path.
        """
        self._cpu_cache = None
        m = response_mask.to(torch.float64)
        w = pre_weight.detach().to(torch.float64)
        d = teacher_kl.detach().to(torch.float64)
        if row_weights is not None:
            m = m * row_weights.reshape(-1, 1).to(torch.float64)
        num = (w * d * m).sum(dim=1)
        den = (d * m).sum(dim=1)
        t = task_ids.reshape(-1).to(torch.long)
        ok = (t >= 0).to(torch.float64)
        self.num.index_add_(0, t.clamp(min=0), num * ok)
        self.den.index_add_(0, t.clamp(min=0), den * ok)

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for t in (self.num, self.den):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    def snapshot(self) -> dict:
        """``{"mean": (T,), "valid": (T,)}`` on the host.

        Invalid means "no KL mass was seen for that task yet", which is a real
        cold-start state and is answered with ``W = 1``. A mean that IS observed
        and comes out zero or non-finite is not a cold start -- it is a bug --
        and the caller fails the step rather than neutralising it.
        """
        if self._cpu_cache is None:
            self._cpu_cache = (self.num.detach().to("cpu"), self.den.detach().to("cpu"))
        num, den = self._cpu_cache
        mean = num / den.clamp(min=1e-30)
        return {"mean": mean, "valid": den > 0, "den": den}

    def reset(self) -> None:
        """Start the next step's accumulation. The snapshot is a per-STEP mean."""
        self.num.zero_()
        self.den.zero_()
        self._cpu_cache = None

    def state_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "num": self.num.detach().to("cpu"),
            "den": self.den.detach().to("cpu"),
        }

    def load_state_dict(self, state: dict) -> None:
        assert int(state["n_tasks"]) == self.n_tasks, "task count changed across resume"
        self.num.copy_(state["num"].to(self.num.device, self.num.dtype))
        self.den.copy_(state["den"].to(self.den.device, self.den.dtype))
        self._cpu_cache = None


# --------------------------------------------------------------------------- #
# advantage-calibrated reliability
# --------------------------------------------------------------------------- #
def group_center(values: torch.Tensor, group_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Subtract each prompt group's own mean, over the valid rows only.

    GRPO's advantage is already group-relative, so the support score has to be
    too: a prompt every rollout finds hard shifts the whole group's score, and
    correlating an un-centred score against a centred advantage reads that
    common offset as between-group variance in the denominator. It cannot create
    a spurious correlation on its own, but it deflates a real one toward zero,
    which on a statistic this noisy is the difference between a signal and a
    shrug.
    """
    g = group_ids.reshape(-1).to(torch.long)
    v = valid.reshape(-1).to(values.dtype)
    n_groups = int(g.max().item()) + 1 if g.numel() and int(g.max().item()) >= 0 else 1
    shape = (n_groups,) + tuple(values.shape[1:])
    total = torch.zeros(shape, dtype=values.dtype, device=values.device)
    count = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    idx = g.clamp(min=0)
    vv = v.reshape((-1,) + (1,) * (values.dim() - 1))
    total.index_add_(0, idx, values * vv)
    count.index_add_(0, idx, v)
    mean = total / count.clamp(min=1.0).reshape((-1,) + (1,) * (values.dim() - 1))
    return values - mean[idx]


# The variables the reliability correlation is formed over, and every second
# moment among them. ``a`` is the GRPO advantage, ``s`` the source's residual
# support score, ``l`` the trajectory's valid length and ``o`` the ON-TASK
# teacher's support score. The last two are carried only so the diagnostics can
# partial them out: length because the score is a SUM over tokens and any
# residual per-token bias in the centring accumulates with it, and the on-task
# score because "this source agrees with the row's own teacher" is the obvious
# confound for "this source predicts reward".
ADV_VARS = ("a", "s", "l", "o")
ADV_MOMENTS = ("n",) + ADV_VARS + tuple(
    f"{x}{y}" for i, x in enumerate(ADV_VARS) for y in ADV_VARS[i:]
)


class AdvantageReliabilityStats:
    """How much each source teacher's residual predicts the destination's reward.

    One float64 cell block per ordered pair ``(d, m)``, holding the moments in
    :data:`ADV_MOMENTS`, from which

        rho[d, m] = Corr(A, S_residual)   over the rows of task d
        alpha[d, m] = max(0, rho[d, m])

    The rectifier is a veto, not an inversion: a source whose residual
    anti-correlates with reward has its evidence dropped, and its shift is NOT
    re-used with the sign flipped -- nothing says a reversed policy shift points
    anywhere useful.

    WHAT IT IS AND IS NOT. The advantage is observed only for the token the
    student actually emitted; no counterfactual reward exists for the rest of
    the support. So this does not certify that a source's tokens are correct. It
    calibrates whether a source's SOURCE-SPECIFIC opinion tracks the
    destination's successful rollouts, and the direction the weight is applied
    in comes from the KL, which always points at the on-task teacher. Estimated
    on the RESIDUAL rather than the full shift so that the part every teacher
    shares -- generically good tokens -- cannot inflate one source's credit.

    Only informative groups contribute. GRPO is group-relative, so a prompt
    whose rollouts all scored the same gives every row an advantage of zero;
    folding those in adds variance to ``S`` against no variance in ``A`` and
    drags every correlation toward zero for a reason that has nothing to do with
    the teachers.

    ``max(0, rho)`` carries a small positive bias under the null -- about
    ``0.4/sqrt(N)`` -- because rectifying a symmetric estimator keeps only its
    positive half. It is left in rather than replaced by a confidence bound,
    because a confidence level is a knob and this arm has none; ``rho_lcb95`` is
    reported beside ``alpha`` so a reader can see when the two are not
    distinguishable from zero. It matters most right after cold start, which is
    also when the student is most plastic.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.n_moments = len(ADV_MOMENTS)
        self.buf = torch.zeros(T * T * self.n_moments, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(
        self,
        *,
        advantage: torch.Tensor,
        support_score: torch.Tensor,
        on_support_score: torch.Tensor,
        length: torch.Tensor,
        informative: torch.Tensor,
        task_ids: torch.Tensor,
        off_plane_tasks: torch.Tensor,
    ) -> None:
        """Fold one batch of ROWS in. Trajectory-level, not per position.

        Args:
            advantage: (bs,) the row's GRPO advantage.
            support_score: (bs, n_off) group-centred residual support score.
            on_support_score: (bs,) group-centred on-task support score.
            length: (bs,) valid response tokens.
            informative: (bs,) bool -- the row's prompt group had a spread of
                advantages. Padding copies are false.
        """
        self._cpu_cache = None
        T, M = self.n_tasks, self.n_moments
        a = advantage.reshape(-1).to(torch.float64)
        l = length.reshape(-1).to(torch.float64)
        o = on_support_score.reshape(-1).to(torch.float64)
        dst = task_ids.reshape(-1).to(torch.long)
        keep_row = informative.reshape(-1).to(torch.bool) & (dst >= 0)

        for c in range(support_score.size(-1)):
            s = support_score[:, c].reshape(-1).to(torch.float64)
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = (keep_row & (src >= 0)).to(torch.float64)
            v = {"a": a, "s": s, "l": l, "o": o}
            cols = [ok]
            cols += [v[x] * ok for x in ADV_VARS]
            cols += [
                v[x] * v[y] * ok
                for i, x in enumerate(ADV_VARS)
                for y in ADV_VARS[i:]
            ]
            vals = torch.stack(cols, dim=-1)                     # (bs, M)
            base = (dst.clamp(min=0) * T + src.clamp(min=0)) * M
            flat = (base.unsqueeze(-1) + torch.arange(M, device=vals.device)).reshape(-1)
            self.buf.index_add_(0, flat, vals.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            T, M = self.n_tasks, self.n_moments
            self._cpu_cache = self.buf.detach().to("cpu").view(T, T, M)
        return self._cpu_cache

    def _cell(self, dst: int, src: int) -> dict:
        buf = self._cpu()
        return {name: float(buf[dst, src, i]) for i, name in enumerate(ADV_MOMENTS)}

    @staticmethod
    def _cov(cell: dict, x: str, y: str) -> float:
        n = cell["n"]
        if n <= 1:
            return 0.0
        key = f"{x}{y}" if f"{x}{y}" in cell else f"{y}{x}"
        return cell[key] / n - (cell[x] / n) * (cell[y] / n)

    @classmethod
    def _corr(cls, cell: dict, x: str, y: str) -> Optional[float]:
        vx, vy = cls._cov(cell, x, x), cls._cov(cell, y, y)
        if cell["n"] < 3 or vx <= 0 or vy <= 0:
            return None
        r = cls._cov(cell, x, y) / math.sqrt(vx * vy)
        return max(-1.0, min(1.0, r)) if math.isfinite(r) else None

    @classmethod
    def _partial(cls, cell: dict, x: str, y: str, given) -> Optional[float]:
        """Correlation of x and y with ``given`` regressed out, via the precision matrix."""
        names = [x, y] + list(given)
        if cell["n"] < len(names) + 2:
            return None
        cov = [[cls._cov(cell, i, j) for j in names] for i in names]
        try:
            import numpy as np

            p = np.linalg.inv(np.array(cov, dtype=float))
        except Exception:
            return None
        denom = p[0, 0] * p[1, 1]
        if not (denom > 0):
            return None
        r = -p[0, 1] / math.sqrt(denom)
        return max(-1.0, min(1.0, r)) if math.isfinite(r) else None

    def alpha(self, *, task_names=None) -> dict:
        """``{(dst_name, src_name): {...}}`` -- the applied alpha and its diagnostics.

        ``alpha`` is the only field the loss may read. ``rho_lcb95`` and the two
        partials are reported next to it and never multiplied into anything: the
        first would introduce a confidence level, and the last two a choice of
        which confound to trust, and both are knobs.
        """
        out = {}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                cell = self._cell(dst, src)
                if cell["n"] <= 0:
                    continue
                rho = self._corr(cell, "a", "s")
                row = {
                    "n": cell["n"],
                    "rho": rho,
                    "alpha": 0.0 if rho is None else max(0.0, rho),
                    "rho_length_controlled": self._partial(cell, "a", "s", ["l"]),
                    "rho_length_on_controlled": self._partial(cell, "a", "s", ["l", "o"]),
                    "rho_lcb95": None,
                }
                if rho is not None and cell["n"] > 3 and abs(rho) < 1.0:
                    se = 1.0 / math.sqrt(cell["n"] - 3.0)
                    row["rho_lcb95"] = math.tanh(math.atanh(rho) - 1.96 * se)
                name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
                out[(name(dst), name(src))] = row
        return out

    def alpha_table(self) -> torch.Tensor:
        """(T, T) float32 of the applied alphas, zero on the diagonal and where undefined.

        Derived from the all-reduced moments, so every rank computes the same
        table from the same numbers and no broadcast is needed.
        """
        table = torch.zeros((self.n_tasks, self.n_tasks), dtype=torch.float32)
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                cell = self._cell(dst, src)
                if cell["n"] <= 0:
                    continue
                rho = self._corr(cell, "a", "s")
                if rho is not None:
                    table[dst, src] = max(0.0, rho)
        return table

    def state_dict(self) -> dict:
        return {"n_tasks": self.n_tasks, "moments": ADV_MOMENTS, "buf": self.buf.detach().to("cpu")}

    def load_state_dict(self, state: dict) -> None:
        assert int(state["n_tasks"]) == self.n_tasks, "task count changed across resume"
        assert tuple(state["moments"]) == ADV_MOMENTS, (
            "the moment layout changed; a resumed buffer would be read column-wise wrong"
        )
        self.buf.copy_(state["buf"].to(self.buf.device, self.buf.dtype))
        self._cpu_cache = None
