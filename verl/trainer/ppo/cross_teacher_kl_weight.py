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
``c``          corroboration: the minimum every teacher IN THE RUN guarantees,
               the on-task one included in both the unanimity test and the
               minimum. A continuous min rather than a sign test, so a shift
               that is nearly zero contributes nearly nothing and no deadzone is
               needed.
``alpha``      per ordered task pair, the correlation between that source's
               RESIDUAL support for the tokens the student actually generated
               and the GRPO advantage of the trajectory it generated them in.
               Rectified at zero: a source that anti-correlates is vetoed, not
               inverted.
``e``          ``|c| + sum_m alpha_m |delta_hat_m|``, the candidate evidence.
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
``e(unanimous) - e(split) = |c| >= 0`` hold for every ``alpha`` in [0, 1] and
for a sign flipped on ANY teacher, the on-task one included.

**Corroboration requires every teacher, not just the off-task ones.** ``c`` is
``1[all teachers share a sign] * sign(d) * min_j |delta_hat_j|`` over
``{d} u off``, so a candidate earns the bonus only where the whole run moved it
the same way. The alternative -- unanimity among the off-task teachers alone --
was tried and is not what this arm runs, because it credits a position where the
other two tasks decisively OPPOSE the on-task teacher exactly as it credits one
where they agree with it. The KL that then gets strengthened points at the
on-task teacher, i.e. at the very opinion the sources contradicted, and it does
so with no advantage-side evidence behind it: at ``alpha = 0`` a
``(-1, +3, +2)`` candidate would score 2 instead of 0.

The price of requiring the on-task teacher to have spoken is real and is
measured rather than assumed. ``c`` is capped by ``|delta_hat_d|`` and the
on-task teacher is silent at about 64% of teacher mass (the shipped run's
``neutral_on_task_silent`` state), so the corroboration channel is quiet across
most of the support and the ``alpha``-gated source term carries the arm there.
:func:`decompose_common_residual` therefore also returns the off-task-only
``common_ev`` and the metrics report what it WOULD have contributed --
``evidence/shared_offtask_only_*`` beside the applied ``evidence/shared_*`` --
so the size of the quiet region is a number in the logs. Nothing in the loss
reads it.

The signed ``c`` is also what the residual, the reliability estimate and the
attribution tables are built from, so one decomposition serves all four.
"""

import math
from typing import Optional

import torch

from verl.trainer.ppo.sign_weights import STATE_NAMES as _STATE_NAMES

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
    "SIDECAR_NAME",
    "sidecar_state",
    "load_sidecar_state",
    "ADV_MOMENTS",
    "AdvantageReliabilityStats",
    "residual_support_score",
    "POSITION_TERMS",
    "STATE_TERMS",
    "PROBE_ALPHAS",
    "probe_name",
    "PairEvidenceStats",
    "build_position_weight",
    "assert_all_finite",
    "position_terms",
    "state_shift_terms",
    "position_weight_metrics",
    "state_shift_metrics",
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

    ACCUMULATION = """Why a delta buffer, and not just one cumulative one.

    ``all_reduce`` is called once per step, and the buffer it reduces must hold
    THIS STEP's contribution only. Reducing the running total instead gives

        Q_n = R * Q_{n-1} + sum_r delta_{n,r}

    at world size R -- every rank already holds the previous global total, so the
    sum counts it R times. The failure is silent and, worse, looks like success:
    Q and N inflate together so sigma does not blow up, it FREEZES. At R=2 and
    step 8, step 1 carries 50% of the cumulative and step 8 carries 0.4%, so the
    scale stops tracking the run while reporting a reassuringly stable number.
    The same recursion hits every moment of the reliability statistic equally,
    which leaves alpha pinned at its step-1 estimate.

    So: ``update`` writes into ``delta``, ``all_reduce`` reduces ``delta``, folds
    it into ``total`` once, and zeroes it. ``total`` is never reduced.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        # Cumulative across the run, already global. Never all-reduced.
        self.q = torch.zeros(T * T, dtype=torch.float64, device=device)
        self.n = torch.zeros(T, dtype=torch.float64, device=device)
        # This step's rank-local contribution. See ACCUMULATION.
        self.dq = torch.zeros(T * T, dtype=torch.float64, device=device)
        self.dn = torch.zeros(T, dtype=torch.float64, device=device)
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
        self.dq.index_add_(0, dst_c * T + dst_c, (m_on * mask).sum(dim=1) * row_ok)
        self.dn.index_add_(0, dst_c, (mask.sum(dim=1)) * row_ok)

        # Off-diagonal: each plane's teacher on THIS destination's states.
        for c in range(d_off.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = row_ok * (src >= 0).to(torch.float64)
            self.dq.index_add_(0, dst_c * T + src.clamp(min=0), (m_off[..., c] * mask).sum(dim=1) * ok)

    def all_reduce(self) -> None:
        """Reduce THIS STEP's delta, fold it in once, and clear it.

        Unconditional: gated on config, never on data, so a rank whose
        micro-batches held nothing still runs the same collective its neighbours
        do. Reducing ``total`` here instead is the bug ACCUMULATION describes.
        """
        self._cpu_cache = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (self.dq, self.dn):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        self.q += self.dq
        self.n += self.dn
        self.dq.zero_()
        self.dn.zero_()

    def _cpu(self):
        if self._cpu_cache is None:
            T = self.n_tasks
            # The pending delta travels with them so the check below costs no
            # extra transfer. Rendering a total that has not absorbed this step
            # yet is a silently stale number, and the whole reason the delta
            # exists is that silently wrong statistics are the failure mode here.
            self._cpu_cache = (
                self.q.detach().to("cpu").view(T, T),
                self.n.detach().to("cpu"),
                float(self.dn.detach().to("cpu").abs().sum()),
            )
        q, n, pending = self._cpu_cache
        assert pending == 0.0, (
            "CumulativePolicyShiftRMS was rendered with an unreduced step delta; "
            "call all_reduce() at the step boundary before reading the scale"
        )
        return q, n

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
        # The delta is empty between steps -- checkpoints are taken there -- but
        # it travels anyway rather than being asserted away: a resume that
        # silently dropped a partial step would be indistinguishable from one
        # that never had it.
        return {
            "n_tasks": self.n_tasks,
            "q": self.q.detach().to("cpu"), "n": self.n.detach().to("cpu"),
            "dq": self.dq.detach().to("cpu"), "dn": self.dn.detach().to("cpu"),
        }

    def load_state_dict(self, state: dict) -> None:
        assert int(state["n_tasks"]) == self.n_tasks, (
            f"task count changed across resume: {state['n_tasks']} -> {self.n_tasks}; "
            "the RMS matrix is indexed by task and cannot be reinterpreted"
        )
        self.q.copy_(state["q"].to(self.q.device, self.q.dtype))
        self.n.copy_(state["n"].to(self.n.device, self.n.dtype))
        self.dq.copy_(state.get("dq", torch.zeros_like(self.dq)).to(self.dq.device, self.dq.dtype))
        self.dn.copy_(state.get("dn", torch.zeros_like(self.dn)).to(self.dn.device, self.dn.dtype))
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
                   and bounded by ``|hat_on|``. THIS is the corroboration the KL
                   evidence uses, and it is also what the residual is defined
                   against, so the reliability estimate and the attribution
                   tables see the same decomposition the weight does.
    ``common_ev``  ``1[the OFF-TASK teachers share a sign] * min_m |hat_m|``,
                   the same quantity with the on-task teacher left out. NOT used
                   by the loss -- see the module docstring for why crediting a
                   position where the sources contradict the on-task teacher is
                   the wrong bonus -- and reported as a counterfactual so the
                   cost of requiring the on-task teacher to have spoken is a
                   measurement. Zero with fewer than two off-task voices: one
                   teacher agreeing with itself is not corroboration.
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
    common: torch.Tensor,
    hat_off: torch.Tensor,
    source_alpha: torch.Tensor,
) -> torch.Tensor:
    """``e = |c| + sum_m alpha_m |hat_delta_m|``. (bs, resp, k), non-negative.

    ``c`` is the ALL-TEACHER corroboration -- the on-task teacher included in
    both the unanimity test and the minimum. A candidate earns the bonus only
    where every teacher in the run moved it the same way, which is what makes
    the bonus a statement about shared structure rather than about the other
    tasks agreeing with each other over the on-task teacher's objection.

    The consequence is worth stating because it is the cost of that choice: the
    on-task teacher is silent at about 64% of teacher mass, and ``c`` is capped
    by its shift, so the corroboration channel is quiet across most of the
    support. :func:`decompose_common_residual` still returns the off-task-only
    ``common_ev``, which is reported as a counterfactual so the size of that
    quiet region is a measurement rather than an assumption -- but nothing in
    the loss reads it.

    Args:
        common: (bs, resp, k) the on-task-inclusive corroboration from
            :func:`decompose_common_residual`. Signed there, absolute here: a KL
            term has no direction to carry.
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
    return common.abs() + (a * hat_off.abs()).sum(dim=-1)


SIDECAR_NAME = "cross_teacher_kl_weight_state.pt"


def sidecar_state(*, rms, mean, adv, alpha, identity: dict) -> dict:
    """Everything the arm needs to resume as if it had never stopped.

    The cumulative RMS, the previous step's normaliser and the reliability
    moments are TRAINING STATE, not diagnostics. Restoring the actor's
    parameters and starting these from zero would put the run back at cold start
    -- every weight 1 for two steps, then a scale rebuilt from a handful of
    positions -- while the logs show a step number in the hundreds. Nothing in
    the metrics distinguishes that from the mechanism having stopped working.

    ``identity`` pins what the numbers mean: the base checkpoint the shifts are
    measured against, the teachers, the temperature the log-probs were taken at,
    and the task order the matrices are indexed by. A resume that disagrees on
    any of them is not this run continued, and is refused rather than blended.
    """
    return {
        "version": 1,
        "identity": dict(identity),
        "rms": rms.state_dict(),
        "position_weight": mean.state_dict(),
        "advantage": adv.state_dict(),
        "last_alpha": alpha.detach().to("cpu") if alpha is not None else None,
    }


def load_sidecar_state(state: dict, *, rms, mean, adv, identity: dict):
    """Restore, after checking the identity. Returns the stored alpha table.

    Every key the caller names must be PRESENT in the checkpoint and must match.
    An absent key used to pass, which made the check strictly weaker the older
    the checkpoint was -- exactly backwards, since an older checkpoint is the one
    most likely to have been written under a different base or teacher set.
    """
    assert int(state.get("version", 0)) == 1, f"unknown sidecar version {state.get('version')}"
    stored = dict(state.get("identity", {}))
    for key, value in identity.items():
        if key not in stored:
            raise AssertionError(
                f"cross_teacher_kl_weight resume: the checkpoint's identity does not record "
                f"{key!r}, so it cannot be shown to describe this run. The accumulated scale "
                "and reliability are only meaningful against a known base, teacher set and "
                "temperature; start the arm fresh rather than assume."
            )
        if stored[key] != value:
            raise AssertionError(
                f"cross_teacher_kl_weight resume mismatch on {key!r}: checkpoint has "
                f"{stored[key]!r}, this run has {value!r}. The accumulated scale and "
                "reliability are measured against that, so they cannot be carried over."
            )
    rms.load_state_dict(state["rms"])
    mean.load_state_dict(state["position_weight"])
    adv.load_state_dict(state["advantage"])
    return state.get("last_alpha", None)


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
        # Positive AND finite. Without the second half a non-finite numerator
        # passes ``den > 0``, and the NaN then rides mu into the weight and out
        # into the loss -- clamp propagates NaN and torch.where selects the
        # valid branch. The caller reads ``nonfinite`` and fails the step; the
        # LOSS is protected here, so the failure is loud instead of a poisoned
        # optimizer state.
        finite = torch.isfinite(mean) & (mean > 0)
        return {
            "mean": mean,
            "valid": (den > 0) & finite,
            "den": den,
            "nonfinite": int(((den > 0) & ~finite).sum()),
        }

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

    def __init__(self, *, n_tasks: int, device, max_groups: int = 512):
        self.n_tasks = T = int(n_tasks)
        self.n_moments = len(ADV_MOMENTS)
        self.max_groups = G = int(max_groups)
        # Cumulative and already global; never all-reduced. See
        # CumulativePolicyShiftRMS.ACCUMULATION for why the delta exists.
        self.buf = torch.zeros(T * T * self.n_moments, dtype=torch.float64, device=device)
        self.dbuf = torch.zeros_like(self.buf)
        # The between-group sum of squares, accumulated as a CORRECTED SCALAR per
        # pair rather than rebuilt from group cells. Each prompt group belongs to
        # exactly one step, so sum_g (sum_g S)^2 / n_g decomposes over steps and
        # this column is exact.
        self.between = torch.zeros(T * T, dtype=torch.float64, device=device)
        # How many rows ever carried a usable group id, per pair. It is what
        # distinguishes "the groups explain none of the spread" from "no group
        # was ever named", and those two must not both read as zero.
        self.grouped = torch.zeros(T * T, dtype=torch.float64, device=device)
        # Per (pair, prompt group): the group's summed support score and its row
        # count. This is what makes the group centring EXACT without a second
        # pass over the step. A group's rollouts land in different micro-batches
        # and on different ranks, so centring locally would centre a fragment;
        # index_add_ pools them here and the all-reduce finishes the job.
        #
        # The advantage is already group-relative, so its group means are zero
        # and the covariance needs no correction. Only the support score's
        # variance does, and
        #     sum (S - S_g)^2 = sum S^2 - sum_g (sum_g S)^2 / n_g
        # turns these two columns into that.
        # STEP-LOCAL, and that is the whole point. ``adv_group_id`` is dense and
        # re-issued from zero every step, so a buffer that survived the step
        # would add step 1's group 0 to step 2's unrelated group 0 and then
        # square the sum. Merging groups understates the between-group term
        # (Cauchy-Schwarz), which overstates the within-group variance and pulls
        # every correlation toward zero -- undoing the centring this exists for.
        self.group = torch.zeros(T * T * G * 2, dtype=torch.float64, device=device)
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
        group_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Fold one batch of ROWS in. Trajectory-level, not per position.

        Args:
            advantage: (bs,) the row's GRPO advantage.
            support_score: (bs, n_off) group-centred residual support score.
            on_support_score: (bs,) group-centred on-task support score.
            length: (bs,) valid response tokens.
            informative: (bs,) bool -- the row's prompt group had a spread of
                advantages. Padding copies are false.
            group_ids: (bs,) dense prompt-group index, or None to skip the
                group centring. Out-of-range ids are dropped from the centring
                rather than wrapped, so an oversized batch loses statistical
                power instead of mixing two prompts into one group.
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
            pair = dst.clamp(min=0) * T + src.clamp(min=0)
            flat = (pair.unsqueeze(-1) * M + torch.arange(M, device=vals.device)).reshape(-1)
            self.dbuf.index_add_(0, flat, vals.reshape(-1))

            if group_ids is not None:
                g = group_ids.reshape(-1).to(torch.long)
                g_ok = ok * ((g >= 0) & (g < self.max_groups)).to(torch.float64)
                cell = (pair * self.max_groups + g.clamp(min=0, max=self.max_groups - 1)) * 2
                idx = torch.stack([cell, cell + 1], dim=-1).reshape(-1)
                self.group.index_add_(0, idx, torch.stack([s * g_ok, g_ok], dim=-1).reshape(-1))

    def all_reduce(self) -> None:
        """Reduce this step's moments and groups, fold both in, and clear.

        The group cells are reduced and then COLLAPSED here, into one
        between-group sum of squares per pair, because that is the only form
        that survives the step boundary: the ids naming the cells are re-issued
        next step and mean something else.
        """
        self._cpu_cache = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (self.dbuf, self.group):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        self.buf += self.dbuf
        self.dbuf.zero_()

        T, G = self.n_tasks, self.max_groups
        grp = self.group.view(T * T, G, 2)
        totals, counts = grp[..., 0], grp[..., 1]
        self.between += torch.where(
            counts > 0, totals.square() / counts.clamp(min=1.0), torch.zeros_like(totals)
        ).sum(dim=-1)
        self.grouped += counts.sum(dim=-1)
        self.group.zero_()

    def _cpu(self):
        if self._cpu_cache is None:
            T, M = self.n_tasks, self.n_moments
            self._cpu_cache = (
                self.buf.detach().to("cpu").view(T, T, M),
                self.between.detach().to("cpu").view(T, T),
                self.grouped.detach().to("cpu").view(T, T),
                float(self.dbuf.detach().to("cpu").abs().sum()),
            )
        buf, between, grouped, pending = self._cpu_cache
        assert pending == 0.0, (
            "AdvantageReliabilityStats was rendered with an unreduced step delta; "
            "call all_reduce() at the step boundary before reading alpha"
        )
        return buf, between, grouped

    def _cell(self, dst: int, src: int) -> dict:
        buf, between, grouped = self._cpu()
        cell = {name: float(buf[dst, src, i]) for i, name in enumerate(ADV_MOMENTS)}
        # The between-group part of the support score's spread, subtracted below
        # so the correlation is against the WITHIN-group score. A prompt every
        # rollout finds hard shifts the whole group, and the advantage it is
        # correlated with has already had exactly that removed.
        #
        # None, not 0.0, when no group was ever named: 0.0 would mean "the groups
        # explain none of the spread", which is a claim, and would silently turn
        # the variance below into an uncentred second moment. ``grouped`` counts
        # the rows that carried a usable group id, which is what tells the two
        # apart.
        cell["_s_between"] = float(between[dst, src]) if float(grouped[dst, src]) > 0 else None
        return cell

    @staticmethod
    def _cov(cell: dict, x: str, y: str) -> float:
        n = cell["n"]
        if n <= 1:
            return 0.0
        key = f"{x}{y}" if f"{x}{y}" in cell else f"{y}{x}"
        raw = cell[key] / n - (cell[x] / n) * (cell[y] / n)
        between = cell.get("_s_between", None)
        if x == "s" and y == "s" and between is not None:
            # Within-group variance: sum (S - S_g)^2 = sum S^2 - sum_g (sum S_g)^2/n_g.
            # The covariance needs no such correction -- the advantage is
            # group-centred already, so its group means are zero and the cross
            # term vanishes. One group holding every row reproduces the ordinary
            # variance exactly, which is what makes this a refinement and not a
            # different statistic.
            return max((cell["ss"] - between) / n, 0.0)
        return raw

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
        return {
            "n_tasks": self.n_tasks,
            "moments": ADV_MOMENTS,
            "max_groups": self.max_groups,
            "buf": self.buf.detach().to("cpu"),
            "dbuf": self.dbuf.detach().to("cpu"),
            "between": self.between.detach().to("cpu"),
            "grouped": self.grouped.detach().to("cpu"),
            "group": self.group.detach().to("cpu"),
        }

    def load_state_dict(self, state: dict) -> None:
        assert int(state["n_tasks"]) == self.n_tasks, "task count changed across resume"
        assert tuple(state["moments"]) == ADV_MOMENTS, (
            "the moment layout changed; a resumed buffer would be read column-wise wrong"
        )
        assert int(state["max_groups"]) == self.max_groups, (
            "max_groups changed; the per-group buffer would be re-indexed onto other prompts"
        )
        self.buf.copy_(state["buf"].to(self.buf.device, self.buf.dtype))
        for name, dst in (
            ("dbuf", self.dbuf), ("between", self.between),
            ("grouped", self.grouped), ("group", self.group),
        ):
            src = state.get(name, None)
            dst.copy_(torch.zeros_like(dst) if src is None else src.to(dst.device, dst.dtype))
        self._cpu_cache = None


def residual_support_score(
    *,
    residual_at_sampled: torch.Tensor,
    residual: torch.Tensor,
    student_logprob: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """(bs, n_off) how much each source's residual backed the tokens actually emitted.

    Per position, ``z = r(y) - E_{pi_student}[r]``: the source's source-specific
    opinion at the token the student chose, measured against what that opinion
    was worth on average under the student's own distribution. Centring against
    the student rather than against zero is what makes ``z`` a statement about
    the CHOICE and not about how opinionated the source is at this state.

    Summed over valid positions rather than averaged. The policy gradient adds
    each valid token to the loss, so a sum is what corresponds to the local
    gradient contribution; the length-normalised version is carried as a
    diagnostic through the ``l`` moment instead of replacing this.

    The expectation runs over the top-k with the tail's residual taken as zero:
    no teacher was read outside the support, so there is no residual there to
    average. When the emitted token is itself outside the top-k the expectation
    therefore excludes it, which is a real approximation and is why
    ``frac_sampled_outside_topk`` is a required metric rather than a nicety.

    Args:
        residual_at_sampled: (bs, resp, n_off) the residual at the emitted token.
        residual: (bs, resp, k, n_off) the residual over the support.
    """
    p = student_logprob.detach().to(residual.dtype).exp()
    expect = (p.unsqueeze(-1) * residual).sum(dim=-2)
    z = residual_at_sampled - expect
    return (z * response_mask.to(z.dtype).unsqueeze(-1)).sum(dim=1)


# --------------------------------------------------------------------------- #
# putting it together, and reporting what it did
# --------------------------------------------------------------------------- #
# The per-position scalars the effect metrics are formed from. ``available`` is
# in the list rather than inferred so the cold-start share is a number and not a
# gap in the series.
POSITION_TERMS = (
    "w",
    "w_sq",
    "w_pre",
    "w_pre_sq",
    "kl",
    "w_kl",
    "kl_shift_abs",
    "evidence",
    "evidence_shared",
    "evidence_shared_offtask_only",
    "available",
)

# The exact partition of the KL each position had moved. ``W - 1`` splits into a
# part proportional to the evidence, which candidates can be charged for, and the
# normaliser's own offset ``1/mu - 1``, which no candidate caused:
#
#     (W - 1) D = [ sum_v p_d(v) e(v) / mu ] D  +  (1/mu - 1) D
#
# so the seven state columns plus ``shift_norm_offset`` sum to the total shift
# with no residual. Reported because the arm's whole effect is WHICH positions
# it moved budget between, and the states are what distinguishes "it backed the
# corroborated moves" from "it starved the on-task teacher's own specialism".
STATE_TERMS = tuple(f"shift_{n}" for n in _STATE_NAMES.values()) + ("shift_norm_offset",)

# What a probe series is for. The training path runs at whatever alpha the
# reliability statistics produced; these run the same arithmetic at fixed alphas
# and never touch the loss. Without the top of the bracket a Phase-2 go/no-go
# would be measuring the corroboration channel alone -- which is structurally the
# minority one, since the on-task teacher is silent at 64% of teacher mass -- and
# would say nothing about what the finished mechanism can do.
PROBE_ALPHAS = (0.0, 0.1, 1.0)


def probe_name(alpha: float) -> str:
    return f"alpha{int(round(float(alpha) * 100)):03d}"


class PairEvidenceStats:
    """Per ordered (destination, source), how much evidence that source supplied.

    ``evidence`` is that source's share of ``W~ - 1``; ``shift`` is its share of
    the nats the weighting actually moved. Kept apart from the per-state table
    because they answer different questions -- "who supplied it" against "what
    kind of position received it" -- and a single table keyed by both would be
    mostly empty.
    """

    TERMS = ("evidence", "shift", "n")

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.buf = torch.zeros(T * T * len(self.TERMS), dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, evidence, shift, response_mask, task_ids, off_plane_tasks) -> None:
        """``evidence`` and ``shift`` are (bs, resp, n_off)."""
        self._cpu_cache = None
        T, K = self.n_tasks, len(self.TERMS)
        m = response_mask.to(torch.float64).unsqueeze(-1)
        e = evidence.detach().to(torch.float64) * m
        s = shift.detach().to(torch.float64) * m
        dst = task_ids.reshape(-1).to(torch.long)
        for c in range(e.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = ((dst >= 0) & (src >= 0)).to(torch.float64)
            vals = torch.stack(
                [e[..., c].sum(dim=1) * ok, s[..., c].sum(dim=1) * ok,
                 (response_mask.to(torch.float64).sum(dim=1)) * ok],
                dim=-1,
            )
            base = (dst.clamp(min=0) * T + src.clamp(min=0)) * K
            flat = (base.unsqueeze(-1) + torch.arange(K, device=vals.device)).reshape(-1)
            self.buf.index_add_(0, flat, vals.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu").view(self.n_tasks, self.n_tasks, len(self.TERMS))
        buf = self._cpu_cache
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst or float(buf[dst, src, 2]) <= 0:
                    continue
                n = float(buf[dst, src, 2])
                head = f"{prefix}/evidence/{name(src)}__on__{name(dst)}"
                out[f"{head}/source_shift_mean"] = float(buf[dst, src, 0]) / n
                out[f"{head}/kl_shift_attributed"] = float(buf[dst, src, 1]) / n
        return out


def build_position_weight(
    *,
    shifts: dict,
    on_task_logprob: torch.Tensor,
    task_ids: torch.Tensor,
    off_plane_tasks: torch.Tensor,
    diag: torch.Tensor,
    diag_valid: torch.Tensor,
    alpha_table: torch.Tensor,
    normalizer: Optional[dict] = None,
    report_epsilon: float = 0.1,
    probe_alphas=PROBE_ALPHAS,
) -> dict:
    """Everything from raw shifts to the scalar that multiplies the KL.

    One function because the pieces have to see the same standardized shifts:
    the evidence, the reliability's residual, the state labels and the
    attribution are four readings of one decomposition, and computing them at
    four call sites is how they start disagreeing.

    Args:
        normalizer: the mapping :meth:`PreviousStepTaskKLWeightedMean.snapshot`
            returns, or None before one exists. None means cold start and every
            weight is exactly 1 -- NOT the raw ``W~``, which would be an
            unannounced increase in distillation strength on the arm's first
            steps, and not a within-micro-batch mean, which would make the
            objective depend on how the batch was split.
        report_epsilon: in RMS units, and used ONLY to bucket a candidate into a
            state for the report. It reaches no weight and no loss; the
            mechanism itself has no deadzone.

    Returns a mapping whose ``weight`` is the only field the loss may read.
    ``pre_weight`` feeds the next step's normaliser, and everything else is
    measurement.

    A row whose destination or any of whose sources has no RMS yet is left at
    ``weight = 1`` and its standardized shifts are zeroed, so no metric reports
    a number divided by a scale that does not exist. Availability is a property
    of the task, not of the row, so a whole task cold-starts together and its
    normaliser comes out at 1 as well.
    """
    from verl.trainer.ppo.sign_weights import candidate_weights

    std = standardize_policy_shifts(
        shifts=shifts, diag=diag, diag_valid=diag_valid,
        task_ids=task_ids, off_plane_tasks=off_plane_tasks,
    )
    avail = std["row_available"]
    keep = avail.reshape(-1, 1, 1).to(std["on"].dtype)
    hat_on = std["on"] * keep
    hat_off = std["off"] * keep.unsqueeze(-1)

    dec = decompose_common_residual(hat_on=hat_on, hat_off=hat_off)
    alpha = alpha_table.to(hat_off.device)
    dst = task_ids.reshape(-1).to(torch.long).clamp(min=0)
    src = off_plane_tasks.to(torch.long).clamp(min=0)
    row_alpha = alpha[dst.unsqueeze(-1), src]                       # (bs, n_off)

    evidence = candidate_kl_evidence(
        common=dec["common"], hat_off=hat_off, source_alpha=row_alpha
    )
    # The off-task-only variant, computed and never applied. It answers the one
    # question the all-teacher rule cannot: how much corroboration the on-task
    # teacher's silence is costing, at the 64% of teacher mass where it says
    # nothing. A counterfactual, so it is reported beside the applied share and
    # reaches no weight.
    evidence_offtask_only = candidate_kl_evidence(
        common=dec["common_ev"], hat_off=hat_off, source_alpha=row_alpha
    )
    pre = position_pre_weight(evidence=evidence, on_task_logprob=on_task_logprob)
    pre = torch.where(avail.reshape(-1, 1), pre, torch.ones_like(pre))
    # A teacher log-prob that arrived non-finite reaches here as a non-finite
    # weight, and a non-finite weight multiplied into the KL is a non-finite
    # gradient and a poisoned optimizer state. Neutralise the position and COUNT
    # it: the count is all-reduced at the step boundary and fails the step
    # there, which is a synchronised point every rank reaches. Silently
    # continuing at W = 1 would leave a corrupted teacher looking like a quiet
    # mechanism.
    finite_pre = torch.isfinite(pre)
    pre = torch.where(finite_pre, pre, torch.ones_like(pre))

    # mu, and the weight. Gathered per row from a per-task snapshot, so a task
    # whose normaliser is not observed yet keeps weight 1 while the others do not
    # have to wait for it.
    if normalizer is None:
        mu = torch.ones_like(pre)
        mu_valid = torch.zeros_like(avail)
    else:
        mean = normalizer["mean"].to(pre.device, pre.dtype)
        valid = normalizer["valid"].to(pre.device)
        mu = mean[dst].reshape(-1, 1).expand_as(pre)
        mu_valid = valid[dst] & avail
    weight = torch.where(mu_valid.reshape(-1, 1), pre / mu.clamp(min=1e-12), torch.ones_like(pre))
    finite_w = torch.isfinite(weight)
    weight = torch.where(finite_w, weight, torch.ones_like(weight))
    nonfinite = (~finite_pre).sum() + (~finite_w).sum()

    # Labels only. candidate_weights is called on the STANDARDIZED shifts with a
    # zero base, so report_epsilon is in RMS units and comparable across
    # teachers, and its weight output (all ones) is discarded: reusing the
    # function is what keeps the state definition from drifting away from the
    # tables the existing arms are reported with.
    _ones, state = candidate_weights(
        hat_on, hat_off, torch.zeros_like(hat_on),
        mode="position", agree_weight=1.0, agree_neg_weight=1.0,
        disagree_weight=1.0, deadzone=float(report_epsilon),
    )

    probes = {}
    for a in probe_alphas:
        e_a = candidate_kl_evidence(
            common=dec["common"], hat_off=hat_off,
            source_alpha=torch.full_like(row_alpha, float(a)),
        )
        p_a = position_pre_weight(evidence=e_a, on_task_logprob=on_task_logprob)
        probes[probe_name(a)] = torch.where(avail.reshape(-1, 1), p_a, torch.ones_like(p_a))

    p_teacher = on_task_logprob.detach().to(torch.float32).exp()
    return {
        "weight": weight,
        "pre_weight": pre,
        # A device scalar, so counting costs no host sync inside the loop.
        "nonfinite": nonfinite,
        "mu": mu,
        "available": avail,
        "hat_on": hat_on,
        "hat_off": hat_off,
        "common": dec["common"],
        "common_ev": dec["common_ev"],
        "residual": dec["residual"],
        "evidence": evidence,
        "state": state,
        "teacher_prob": p_teacher,
        # sum_v p_d(v) |c(v)| -- the corroboration channel's share of W~ - 1.
        "evidence_shared": (p_teacher * dec["common"].abs()).sum(dim=-1),
        # The same thing the off-task-only rule would have produced. Reported,
        # never applied: the gap between the two IS the price of requiring the
        # on-task teacher to have spoken.
        "evidence_shared_offtask_only": (p_teacher * dec["common_ev"].abs()).sum(dim=-1),
        "evidence_offtask_only": (p_teacher * evidence_offtask_only).sum(dim=-1),
        # (bs, resp, n_off): each source's share of the same quantity.
        "evidence_by_source": p_teacher.unsqueeze(-1) * row_alpha.unsqueeze(1).unsqueeze(1) * hat_off.abs(),
        "probe_pre_weight": probes,
    }


def assert_all_finite(counts: dict) -> None:
    """Fail the step on any non-finite the arm saw, at a synchronised point.

    Every caller reaches this line, so raising here cannot deadlock the way an
    exception inside the micro-batch loop would -- one rank leaving a collective
    its neighbours are still waiting on hangs the job instead of ending it.

    Failing rather than neutralising is the point. A non-finite teacher log-prob,
    a normaliser that went to NaN, a scale that came back inf: each of those has
    a cause, and each looks from the metrics exactly like "the mechanism found
    nothing to do". The weights are already forced to 1 by the time this runs,
    so the loss for the failing step was never corrupted; what this stops is the
    NEXT thousand steps continuing on a silently inert arm.
    """
    bad = {name: int(value) for name, value in counts.items() if int(value) > 0}
    if bad:
        raise AssertionError(
            "cross_teacher_kl_weight saw non-finite values this step: "
            + ", ".join(f"{name}={n}" for name, n in sorted(bad.items()))
            + ". The weights were forced to 1 so the loss is intact, but the cause is "
            "upstream -- check the teacher/base log-probs and the accumulated scale."
        )


def position_terms(built: dict, teacher_kl: torch.Tensor) -> dict:
    """The (bs, resp) columns :data:`POSITION_TERMS` names."""
    w = built["weight"].detach().to(torch.float32)
    pre = built["pre_weight"].detach().to(torch.float32)
    kl = teacher_kl.detach().to(torch.float32)
    return {
        "w": w,
        "w_sq": w * w,
        "w_pre": pre,
        "w_pre_sq": pre * pre,
        "kl": kl,
        "w_kl": w * kl,
        "kl_shift_abs": (w - 1.0).abs() * kl,
        "evidence": pre - 1.0,
        "evidence_shared": built["evidence_shared"],
        "evidence_shared_offtask_only": built["evidence_shared_offtask_only"],
        "available": built["available"].reshape(-1, 1).expand_as(w).to(torch.float32),
    }


def state_shift_terms(built: dict, teacher_kl: torch.Tensor) -> dict:
    """The exact partition of ``(W - 1) D`` over the seven states plus the offset.

    The per-candidate part is ``p_d(v) e(v) / mu * D`` and the leftover is the
    normaliser's ``(1/mu - 1) D``, which belongs to no candidate. Their sum is
    the position's whole shift, which is what makes the columns a decomposition
    rather than a set of correlated summaries.
    """
    kl = teacher_kl.detach().to(torch.float32)
    inv_mu = 1.0 / built["mu"].clamp(min=1e-12)
    per_cand = built["teacher_prob"] * built["evidence"] * inv_mu.unsqueeze(-1) * kl.unsqueeze(-1)
    state = built["state"]
    out = {}
    for sid, name in _STATE_NAMES.items():
        out[f"shift_{name}"] = (per_cand * (state == sid).to(per_cand.dtype)).sum(dim=-1)
    out["shift_norm_offset"] = (inv_mu - 1.0) * kl
    return out


def position_weight_metrics(sums: dict, prefix: str = "kl_weight") -> dict:
    """The readings, formed from sums rather than from means of ratios.

    ``kl_scale`` is the invariant, not ``w_mean``. What a task's OPD budget IS
    is ``sum W*D``; the two coincide only when the weight and the KL are
    uncorrelated, and this arm exists on the bet that they are not. On the
    snapshot the normaliser was built from ``kl_scale`` is 1 by construction, so
    what it reports on the live batch is the one-step lag plus the step's own
    distribution shift.

    ``kl_shift_gross_frac`` is what the Phase-2 go/no-go is judged on: the
    fraction of the OPD term the arm moved between positions. Dimensionless, so
    a threshold on it means the same thing at any teacher_kl_loss_coef, and
    unlike ``w_cv`` it is in the units the objective is actually in.
    """
    out = {}
    for scope, tot in sums.items():
        head = prefix if scope is None else f"{prefix}/{scope}"
        n = tot["n"]
        if n <= 0:
            continue
        w_mean = tot["w"] / n
        w_var = max(tot["w_sq"] / n - w_mean * w_mean, 0.0)
        pre_mean = tot["w_pre"] / n
        pre_var = max(tot["w_pre_sq"] / n - pre_mean * pre_mean, 0.0)
        out[f"{head}/position/w_mean"] = w_mean
        out[f"{head}/position/w_std"] = math.sqrt(w_var)
        out[f"{head}/position/w_cv"] = math.sqrt(w_var) / w_mean if abs(w_mean) > 1e-12 else 0.0
        out[f"{head}/position/w_pre_mean"] = pre_mean
        out[f"{head}/position/w_pre_std"] = math.sqrt(pre_var)
        out[f"{head}/position/available_frac"] = tot["available"] / n
        out[f"{head}/evidence/total_mean"] = tot["evidence"] / n
        out[f"{head}/evidence/shared_mean"] = tot["evidence_shared"] / n
        out[f"{head}/evidence/shared_offtask_only_mean"] = tot["evidence_shared_offtask_only"] / n
        # The counterfactual, never applied. Above 1 by a lot means the on-task
        # teacher's silence is where the corroboration channel goes quiet, which
        # is the known cost of the all-teacher rule and the number that says how
        # large it actually is on this run.
        if abs(tot["evidence_shared"]) > 1e-12:
            out[f"{head}/evidence/shared_offtask_only_ratio"] = (
                tot["evidence_shared_offtask_only"] / tot["evidence_shared"]
            )
        if abs(tot["evidence"]) > 1e-12:
            # Which channel is carrying the mechanism. Near 1 the corroboration
            # bonus IS the arm; near 0 it is decoration and what the run tests is
            # "reliable source activity", not agreement.
            out[f"{head}/evidence/shared_share"] = tot["evidence_shared"] / tot["evidence"]
        if abs(tot["kl"]) > 1e-12:
            out[f"{head}/effect/kl_scale"] = tot["w_kl"] / tot["kl"]
            out[f"{head}/effect/kl_shift_gross_frac"] = tot["kl_shift_abs"] / tot["kl"]
            out[f"{head}/effect/kl_scale_lag_error"] = tot["w_kl"] / tot["kl"] - 1.0
        out[f"{head}/effect/kl_unweighted"] = tot["kl"] / n
        out[f"{head}/effect/kl_weighted"] = tot["w_kl"] / n
        out[f"{head}/effect/kl_shift_net"] = (tot["w_kl"] - tot["kl"]) / n
        gross = tot["kl_shift_abs"] / n
        out[f"{head}/effect/kl_shift_gross"] = gross
        if gross > 1e-12:
            out[f"{head}/effect/redistribution_ratio"] = ((tot["w_kl"] - tot["kl"]) / n) / gross
    return out


def state_shift_metrics(sums: dict, prefix: str = "kl_weight") -> dict:
    """Which kinds of position the budget moved between.

    THE analysis reading for this arm. The weight scales the whole KL toward the
    on-task teacher, so the only thing the mechanism can do is decide which
    states get more of it -- and ``neutral_off_task_silent``, where the on-task
    teacher is the only one with an opinion, is where it takes the most away.
    That is the on-task teacher's own specialism, so a gain and a loss have to
    be read against this table or neither can be attributed.
    """
    out = {}
    for scope, tot in sums.items():
        head = prefix if scope is None else f"{prefix}/{scope}"
        total = sum(abs(tot[t]) for t in STATE_TERMS)
        for term in STATE_TERMS:
            out[f"{head}/effect/kl_shift_by_state/{term[len('shift_'):]}/net"] = tot[term] / tot["n"]
            if total > 1e-12:
                out[f"{head}/effect/kl_shift_by_state/{term[len('shift_'):]}/gross_share"] = (
                    abs(tot[term]) / total
                )
    return out
