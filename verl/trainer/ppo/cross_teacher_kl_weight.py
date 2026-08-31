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
``c``          corroboration: ``delta_hat_d * f``, the on-task teacher's own
               shift graded by ``f``, the fraction of off-task mass moving the
               same way. The on-task teacher is the DIRECTION and the CEILING;
               the other teachers vote on how much of it to believe. A
               continuous vote rather than a unanimity, and a ceiling rather
               than a minimum -- see below for what the hard forms cost.
``x_m``        ``relu(|delta_hat_m| - |delta_hat_d|)``: what source ``m`` says
               BEYOND what the on-task teacher already says. The boundary
               between the two channels.
``q``          ``2 sum_{m<m'} delta_hat_m delta_hat_m' / [(M-1) sum_m
               delta_hat_m^2]``, clamped at zero: one minus the share of the
               off-task teachers' energy that is disagreement. The source
               channel's reliability, computed at the candidate it is applied
               at.
``e``          ``|c| + q * sum_m x_m``, the candidate evidence.
``W~``         ``1 + sum_v p_teacher(v) e(v)``, the position's raw weight.
``W``          ``W~ / mu_d``, where ``mu_d`` is the PREVIOUS step's KL-weighted
               per-task mean. That divisor is what keeps this a redistribution
               instead of a larger ``teacher_kl_loss_coef``.

Three properties of ``e`` are load-bearing and all three are pinned by tests.

**It is monotone in corroboration.** An earlier draft used
``|c| + sum_m alpha_m |delta_hat_m - c|``, subtracting the common part from each
source. That is right for a target rewrite, where double-adding a shift in log
space really would double-count probability mass, and wrong here: agreement is
credited once and debited ``n_off`` times, so a CONFLICTING position scores
HIGHER than an agreeing one with the same shift magnitudes. ``W`` is a
dimensionless effort scalar with no conservation law to respect, so the
subtraction buys nothing; without it a unanimity always outscores a split, and
flipping the ON-TASK teacher's sign costs exactly ``|c|`` -- neither source
factor reads that sign.

**The two channels partition the source shift rather than sharing it.** The
identity is

    |delta_hat_m| = min(|delta_hat_m|, |delta_hat_d|) + relu(|delta_hat_m| - |delta_hat_d|)
                    \____ what ``c`` is capped at ____/   \_________ ``x_m`` _________/

so each teacher's volume is spent once. This is the correction the shipped run
forced. There, ``e`` was ``|c| + sum_m alpha_m |delta_hat_m|`` -- the source term
took the FULL shift at every candidate, with no condition on the on-task teacher
at all -- and the measurement said what that meant: **53.3% of the source
channel's mass sat in the ``agree`` state**, the corroboration's own territory,
against 13.3% in ``on_silent_source_active``, the state the channel was designed
for. A second copy of the channel beside it, by mass. The split is by MAGNITUDE
and not by agreement: a candidate where the teachers conflict and the off-task
one is louder still has an excess and still gets it, because "the on-task shift
is small and the off-task shift is large" cannot be narrowed to on-task silence
without a threshold, and ``|delta_hat_d|`` as a continuous ceiling is what
avoids one.

**The reliability is measured where it is applied.** ``q`` replaces ``alpha``,
which was the correlation between a source's residual support for the sampled
tokens and the GRPO advantage of the trajectory they came from -- estimated over
ROWS, accumulated over the RUN, and applied per TOKEN. Three granularities, and
the run showed the cost of all three: per-step ``rho`` had a larger standard
deviation than its own mean at every pair and flipped sign between adjacent
steps 33-51% of the time at four of the six; two thirds of the two largest
values survived a length control; ``alpha`` was identically zero for two of the
six pairs across the entire 131-step window that has a control arm, and reached
0.24% of its own headroom over that window. An estimator that needs 130 steps to
leave zero cannot be evaluated in a 300-step run. ``q`` carries no state, needs
no warm-up, and is a function of this candidate's shifts alone.

``q`` is a normalized inner product and NOT the sign-agreement ratio
``|sum_m delta_hat_m| / sum_m |delta_hat_m|``, for two reasons the ratio cannot
fix. A silent teacher scores the ratio a full 1 -- one teacher agreeing with
itself, credited as corroboration, the exact case :func:`decompose_common_residual`
has to special-case away for ``common_ev``; here the numerator is a product over
PAIRS, so silence sends it to zero and the ``n_off < 2`` branch falls out of the
algebra. And the ratio ignores magnitude entirely, scoring 1 whenever the signs
agree however lopsided the two shifts are, while ``q`` is ``2r/(1+r^2)`` in
their size ratio -- 0.60 at 3x, 0.20 at 10x. That matters on this run because
the off-task pair is systematically lopsided at two of the three destinations
(``bottleneck_share`` 0.66/0.11 and 0.80/0.15).

**THE ADVANTAGE REACHES NO WEIGHT.** ``alpha_table`` is still threaded through
:func:`build_position_weight` and the reliability accumulator still runs, so
``corr_adv_source_effect`` and the GRPO gradient cosines still report whether the
reward-free rule happened to land where the reward would have. That is now a
measurement of the mechanism rather than an input to it -- and it buys a source
channel that is live from step 1 at every destination, where the correlation
would still be off for the ~70% of ``search`` groups whose rollouts all score
the same (``informative_group_frac`` 0.302, against 0.937 on ``alfworld``).

Both channels are counterfactualled rather than argued. ``no_shared`` is the
source alone and ``offtask_shared`` the off-task-only corroboration; the two the
new form adds are ``ungated_source`` (``q`` forced to 1, so the gate's cost is a
number) and ``shuffled_gate`` (``q`` recomputed on
:func:`decorrelated_off_shifts`, the teachers slid past each other). The last is
the one the design most needs: every teacher here is the same base after RL on a
different task with the same recipe, so two of them agreeing can be one shared
generation grammar answering twice rather than two independent findings. Where
the shuffled gate matches the live one, the independence ``q`` assumes does not
hold -- and a structural-token mask, if one is wanted, comes out of that ratio
instead of being written by hand.

The old hard forms are kept as measurement. ``common`` (the all-teacher
unanimity over a minimum) still feeds the attribution tables and the residual;
``common_ev`` (the off-task-only version) is still reported beside the applied
share. Neither reaches the loss. The two are no longer nested, so
``shared_offtask_only_ratio`` is a comparison rather than a bound.

The signed ``c`` is also what the residual, the reliability estimate and the
attribution tables are built from, so one decomposition serves all four.
"""

import math
from typing import Optional

import torch

from verl.trainer.ppo.sign_weights import PAIR_STATES
from verl.trainer.ppo.sign_weights import ROLE_NAMES as _ROLE_NAMES
from verl.trainer.ppo.sign_weights import STATE_NAMES as _STATE_NAMES

__all__ = [
    "compute_raw_policy_shifts",
    "tail_logprob",
    "CumulativePolicyShiftRMS",
    "standardize_policy_shifts",
    "corroboration_attribution",
    "decompose_common_residual",
    "candidate_kl_evidence",
    "position_pre_weight",
    "PreviousStepTaskKLWeightedMean",
    "group_center",
    "SIDECAR_NAME",
    "sidecar_state",
    "load_sidecar_state",
    "resume_identity",
    "ADV_MOMENTS",
    "AdvantageReliabilityStats",
    "residual_support_score",
    "teacher_similarity",
    "source_exclusive_shift",
    "decorrelated_off_shifts",
    "POSITION_TERMS",
    "STATE_TERMS",
    "PROBE_ALPHAS",
    "CHANNEL_PROBES",
    "probe_name",
    "PairEvidenceStats",
    "PAIR_STATES",
    "pair_state_index",
    "PairStateEvidenceStats",
    "CorroborationAttributionStats",
    "build_position_weight",
    "assert_all_finite",
    "position_terms",
    "per_candidate_shift",
    "GRAD_TERMS",
    "opd_logit_push",
    "PUSH_CLASSES",
    "push_direction_class",
    "LogitPushTokens",
    "OUTCOME_BUCKETS",
    "OutcomeEffectStats",
    "SourceOutcomeStats",
    "SOURCE_OUTCOME_TERMS",
    "logit_gradient_terms",
    "gradient_metrics",
    "state_shift_terms",
    "position_weight_metrics",
    "state_shift_metrics",
    "PositionScopeTermStats",
    "WEIGHT_BUCKET_EDGES",
    "WEIGHT_THRESHOLDS",
    "WeightShiftHistogram",
    "TURN_BUCKETS",
    "TURN_SCOPE_NAMES",
    "ROLE_SCOPE_NAMES",
    "ROLE_CUT_SUFFIXES",
    "TURN_CUT_SUFFIXES",
    "select_metrics",
    "BELOW_ONE_EDGE",
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
        # Set by all_reduce: the delta it just folded. None before the first.
        self._step = None
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
        self._cpu_cache = {}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (self.dq, self.dn):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        # This step's reduced delta, kept before it is cleared. The cumulative
        # sigma is what the weight divides by and what has the sample size; the
        # step's own is the only thing that can say the cumulative one has gone
        # stale. Over 150 steps the student moves, so a teacher's shift on its
        # own task is not a stationary quantity, and a divisor that averages the
        # first fifty steps into the last is a scale nobody chose.
        self._step = (self.dq.detach().clone(), self.dn.detach().clone())
        self.q += self.dq
        self.n += self.dn
        self.dq.zero_()
        self.dn.zero_()

    SCOPES = ("cumulative", "current")

    def _cpu(self, scope: str = "cumulative"):
        assert scope in self.SCOPES, scope
        if not isinstance(self._cpu_cache, dict):
            self._cpu_cache = {}
        if scope not in self._cpu_cache:
            T = self.n_tasks
            if scope == "current":
                q, n = self._step if self._step is not None else (
                    torch.zeros(T * T, dtype=torch.float64), torch.zeros(T, dtype=torch.float64)
                )
                self._cpu_cache[scope] = (
                    q.detach().to("cpu").view(T, T), n.detach().to("cpu"), 0.0
                )
            else:
                # The pending delta travels with them so the check below costs no
                # extra transfer. Rendering a total that has not absorbed this step
                # yet is a silently stale number, and the whole reason the delta
                # exists is that silently wrong statistics are the failure mode here.
                self._cpu_cache[scope] = (
                    self.q.detach().to("cpu").view(T, T),
                    self.n.detach().to("cpu"),
                    float(self.dn.detach().to("cpu").abs().sum()),
                )
        q, n, pending = self._cpu_cache[scope]
        assert pending == 0.0, (
            "CumulativePolicyShiftRMS was rendered with an unreduced step delta; "
            "call all_reduce() at the step boundary before reading the scale"
        )
        return q, n

    def snapshot(self, scope: str = "cumulative") -> dict:
        """``{"sigma": (T, T), "valid": (T, T), "n": (T,)}`` on the host.

        A cell is valid only with positive count, positive sigma and a finite
        value. An invalid cell is never patched with an epsilon: a denominator
        invented to avoid a division is a transfer strength nobody chose.

        ``scope="current"`` renders the same quantities from THIS step's rows
        alone. Diagnostic only -- :meth:`diagonal`, which is what the weight
        divides by, takes the cumulative snapshot and nothing else.
        """
        q, n = self._cpu(scope)
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


def decompose_common_residual(
    *, hat_on: torch.Tensor, hat_off: torch.Tensor, eps: float = 1e-12
) -> dict:
    """Corroboration and residual, from the standardized shifts.

    Returns four things, and they are NOT interchangeable:

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
    ``common_soft`` ``hat_on * f``, ``f = sum_m relu(sign(hat_on) hat_m) /
                   sum_m |hat_m|``. THE CORROBORATION THE EVIDENCE USES. Same
                   two properties as ``common`` -- signed with the on-task
                   teacher, bounded by ``|hat_on|`` -- reached without either of
                   the two hard gates ``common`` is built from:

                   the UNANIMITY becomes a fraction. One dissenting teacher
                   lowers ``f`` by its share of the off-task mass instead of
                   zeroing the candidate, so the run's own measurement of what
                   the veto costs (``suppression_ratio`` 1.5-4.7x, one teacher
                   at a time) stops being a counterfactual.

                   the MINIMUM becomes a ceiling. ``|hat_on|`` caps ``c``, not
                   ``min_j |hat_j|``, so a quiet off-task teacher can no longer
                   hold the corroboration down to its own volume. What it
                   exceeds the ceiling by is not discarded either: it is exactly
                   :func:`source_exclusive_shift`, which the source channel
                   picks up, so the two channels partition
                   ``|hat_m| = min(|hat_m|, |hat_on|) + relu(|hat_m| - |hat_on|)``
                   between them rather than competing for the same mass.

                   Silence still zeroes it, twice over: ``sign(hat_on) = 0``
                   drives ``f`` to 0 AND ``hat_on`` multiplies through. That is
                   what keeps ``f``'s numerical noise at a silent on-task
                   teacher from reaching the weight -- it multiplies something
                   that is already zero.

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

    # The graded rule. ``agree_mass`` is the off-task mass moving the on-task
    # teacher's way; over the total off-task mass it is the fraction of the
    # other teachers that corroborate, which is 1 under unanimity and falls
    # smoothly rather than to zero under partial dissent.
    off_mass = hat_off.abs().sum(dim=-1)
    agree_mass = (sign_on.unsqueeze(-1) * hat_off).clamp(min=0.0).sum(dim=-1)
    frac = torch.where(
        off_mass > eps, agree_mass / off_mass.clamp(min=eps), torch.zeros_like(agree_mass)
    )
    return {
        "common": common,
        "common_ev": common_ev,
        "common_soft": hat_on * frac,
        "residual": hat_off - common.unsqueeze(-1),
    }


def teacher_similarity(hat_off: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """How alike the OFF-TASK teachers' shifts are here. (bs, resp, k) in [0, 1].

        q = 1 - sum_{m<m'} (hat_m - hat_m')^2 / [ (M-1) sum_m hat_m^2 ]
          = 2 sum_{m<m'} hat_m hat_m'  / [ (M-1) sum_m hat_m^2 ]

    "One minus the share of the teachers' total energy that is disagreement",
    clamped at zero. Both forms are the same expression; the second is what is
    computed, from ``(sum hat)^2 - sum hat^2``, in one pass.

    It replaces the reliability the advantage used to supply, and it is chosen
    over the obvious sign-agreement ratio ``|sum hat| / sum |hat|`` because that
    ratio has two failures this one does not:

    a SILENT teacher scores full agreement. With ``hat_2 = 0`` the ratio is
    ``|hat_1| / |hat_1| = 1``: one teacher agreeing with itself is credited as
    corroboration, which is the exact case :func:`decompose_common_residual`
    has to special-case away for ``common_ev``. Here the numerator is a product
    over PAIRS, so a silent teacher sends it to 0 and the ``n_off < 2`` branch
    falls out of the algebra instead of being written.

    MAGNITUDE is ignored. The ratio is 1 whenever the signs agree, however
    lopsided the two shifts are; this one is ``2r/(1+r^2)`` in the ratio ``r`` of
    their sizes -- 1.0 at parity, 0.60 at 3x, 0.20 at 10x. That matters on this
    run because the off-task pair is systematically lopsided at two of the three
    destinations (``bottleneck_share`` 0.66/0.11 and 0.80/0.15), and a rule that
    calls those full agreement is not measuring agreement.

    ``q <= 1`` is exact, by AM-GM on each pair; the clamp is float hygiene, not
    a rule. The floor at 0 is the rule: a negative ``q`` is teachers pointing
    opposite ways, and the source channel has nothing to weight there.

    Not a correlation over rows or over history. It is computed at the candidate
    it is applied at, from the same tensors the weight is built from, which is
    the property the advantage-based reliability could not have.
    """
    n_off = hat_off.size(-1)
    if n_off < 2:
        return torch.zeros(hat_off.shape[:-1], dtype=hat_off.dtype, device=hat_off.device)
    total = hat_off.sum(dim=-1)
    energy = (hat_off * hat_off).sum(dim=-1)
    cross = total * total - energy
    q = cross / (float(n_off - 1) * energy).clamp(min=eps)
    return torch.where(energy > eps, q, torch.zeros_like(q)).clamp(min=0.0, max=1.0)


def source_exclusive_shift(*, hat_on: torch.Tensor, hat_off: torch.Tensor) -> torch.Tensor:
    """``relu(|hat_m| - |hat_on|)``. (bs, resp, k, n_off), non-negative.

    What source ``m`` says here BEYOND what the on-task teacher already says.
    The boundary between the two channels, and the reason they can be added
    without counting the same evidence twice:

        |hat_m| = min(|hat_m|, |hat_on|) + relu(|hat_m| - |hat_on|)
                  \_ the corroboration's _/   \_ the source channel's _/

    The left part is what ``common_soft`` is already capped at; only the right
    part reaches the source term. On the measured state distribution that is the
    difference between a source channel with 53.3% of its mass sitting in
    ``agree`` -- a second copy of the corroboration -- and one that keeps the
    mass the on-task teacher is not covering.

    A magnitude split, not an agreement split: a candidate where the teachers
    CONFLICT and the off-task one is louder still has an excess, and still gets
    it. That is the specification -- "the on-task shift is small and the
    off-task shift is large" -- and it cannot be narrowed to on-task silence
    without a threshold, which is what ``|hat_on|`` as a continuous ceiling
    exists to avoid.
    """
    return (hat_off.abs() - hat_on.abs().unsqueeze(-1)).clamp(min=0.0)


def decorrelated_off_shifts(hat_off: torch.Tensor) -> torch.Tensor:
    """The off-task planes slid past each other along the response axis.

    The placebo for :func:`teacher_similarity`. Every teacher here is the same
    base model after RL on a different task with the same recipe, so two of them
    agreeing is not evidence of two independent findings -- it can be the shared
    generation grammar answering twice. Rolling each plane by a different offset
    destroys the position-to-position correspondence while leaving each
    teacher's own distribution of shifts untouched, so:

        q_sim(rolled) / q_sim  ~ 0   agreement that needed the two teachers to
                                     be looking at the SAME candidate
        q_sim(rolled) / q_sim  ~ 1   agreement that survives being paired with
                                     an arbitrary other position -- grammar, not
                                     content

    The second is the region where the independence the gate assumes does not
    hold, and it is measured rather than assumed: a structural-token mask, if
    one is wanted later, comes out of this ratio instead of being written by
    hand.

    A within-row roll, not a permutation: deterministic, no generator to thread
    through, no extra forward. It rolls over padded positions too, which is why
    this is a diagnostic and never a weight.
    """
    n_off, resp = hat_off.size(-1), hat_off.size(1)
    if n_off < 2 or resp < 2:
        return hat_off
    return torch.stack(
        [
            torch.roll(hat_off[..., m], shifts=int(((m + 1) * resp) // (n_off + 1)), dims=1)
            for m in range(n_off)
        ],
        dim=-1,
    )


def corroboration_attribution(*, hat_on: torch.Tensor, hat_off: torch.Tensor) -> dict:
    """Which teacher decides ``c``, and how much each one is suppressing.

    ``c = 1[all agree] * sign * min_j |hat_j|`` has two ways for a single
    teacher to control it, and the shared channel carries ~98% of this arm's
    evidence, so "the teachers corroborated" is a claim about ONE of them and
    the logs never said which.

    ``bottleneck``   ``argmin_j |hat_j|`` over ``{on} u off``, so column 0 is the
                     on-task teacher. Only meaningful where ``c != 0``; the
                     caller weights by ``p_on |c|``, which is zero elsewhere.
                     The module docstring already asserts that ``c`` is capped
                     by ``|hat_on|`` and that the on-task teacher is silent at
                     ~64% of teacher mass -- this measures how often that cap is
                     the binding one rather than assuming it.
    ``without``      ``|c_{-j}|`` for each teacher ``j`` dropped in turn, same
                     column order. Leave-one-out on the EVIDENCE rather than on
                     the applied weight: no counterfactual normaliser, no second
                     pass, and with the shared channel at 98% the two answer the
                     same question. ``|c_{-j}| - |c| >= 0`` always, because
                     dropping a teacher can only raise the minimum or restore a
                     unanimity it was breaking -- so one number covers both ways
                     a teacher holds the bonus down, and a teacher that is
                     neither reads as exactly 0.
    ``margin``       ``second_smallest - smallest`` of ``|hat_j|`` where ``c``
                     is live. Near zero the argmin is a coin flip between two
                     teachers and the attribution above is fragile; the caller
                     reports the fraction under a small threshold rather than
                     letting a tie read as a decision.

    Dropping the on-task teacher is column 0 and is exactly the ``common_ev``
    :func:`decompose_common_residual` already returns, recomputed here so the
    columns come from one expression instead of two.
    """
    stack = torch.cat([hat_on.abs().unsqueeze(-1), hat_off.abs()], dim=-1)  # (..., 1+n_off)
    signs = torch.cat([torch.sign(hat_on).unsqueeze(-1), torch.sign(hat_off)], dim=-1)
    n_all = stack.size(-1)

    def _common_over(keep_mask):
        """|c| over the teachers ``keep_mask`` selects. (...,)"""
        first = signs[..., :1]
        agree = ((signs == first) | ~keep_mask).all(dim=-1) & (first.squeeze(-1) != 0)
        big = torch.finfo(stack.dtype).max
        mag = torch.where(keep_mask, stack, torch.full_like(stack, big)).amin(dim=-1)
        return torch.where(agree, mag, torch.zeros_like(mag))

    ones = torch.ones_like(stack, dtype=torch.bool)
    # ``_common_over`` judges unanimity against the ON-TASK sign, which is what
    # the live rule does and is right for every column but its own. Dropping the
    # on-task teacher has to switch the reference to the off-task teachers'
    # own -- otherwise a position where it is SILENT reads as "no unanimity" in
    # the one column that exists to say what the others agree on without it, and
    # that is the 64% of teacher mass the column was built for. It is also what
    # makes column 0 equal common_ev exactly rather than approximately.
    without = []
    for j in range(n_all):
        keep = ones.clone()
        keep[..., j] = False
        if j == 0:
            # Off-task-only unanimity, judged against the first off-task sign.
            off_first = signs[..., 1:2]
            agree = (signs[..., 1:] == off_first).all(dim=-1) & (off_first.squeeze(-1) != 0)
            mag = stack[..., 1:].amin(dim=-1)
            without.append(torch.where(agree, mag, torch.zeros_like(mag)))
        else:
            without.append(_common_over(keep))

    two_smallest = stack.topk(min(2, n_all), dim=-1, largest=False).values
    margin = (two_smallest[..., -1] - two_smallest[..., 0]) if n_all >= 2 else torch.zeros_like(hat_on)
    return {
        "bottleneck": stack.argmin(dim=-1),
        "without": torch.stack(without, dim=-1),
        "margin": margin,
    }


def candidate_kl_evidence(
    *,
    common: torch.Tensor,
    source_gate: torch.Tensor,
    exclusive: torch.Tensor,
    source_scale: float = 1.0,
) -> torch.Tensor:
    """``e = |c| + q * sum_m relu(|hat_m| - |hat_on|)``. (bs, resp, k), non-negative.

    TWO channels, two factors each, and nothing else. The evidence is the whole
    of what the weight is built from, so every gate in it multiplies every other
    one; the arm's own history is the argument for keeping the count down. The
    shipped mechanism ran ``|c|`` behind a unanimity AND a minimum AND an
    advantage correlation, and the source channel it was supposed to test
    reached 0.24% of its own headroom over the 131 steps that have a control.
    Five [0, 1] gates multiplied together is that failure again with more
    places for it to hide.

    ``common``       the corroboration, ``common_soft`` from
                     :func:`decompose_common_residual`. Signed there, absolute
                     here: a KL term has no direction to carry.
    ``source_gate``  (bs, resp, k), :func:`teacher_similarity`. WHETHER the
                     off-task teachers are worth believing at this candidate.
    ``exclusive``    (bs, resp, k, n_off), :func:`source_exclusive_shift`. WHAT
                     they add beyond the on-task teacher.
    ``source_scale`` a plain multiplier on the source channel, for the probe
                     series only. 1.0 is the shipped value; the training path
                     never passes anything else.

    The two source factors are not two views of one quantity -- the gate is a
    second-order statement about how alike the teachers' shifts are, the
    exclusive shift is a first-order statement about their level -- so a lone
    loud teacher and two concordant ones are told apart rather than summed.

    NO ADVANTAGE. The reliability the row's reward used to supply is gone from
    this function and from the loss; ``corr_adv_source_effect`` and the GRPO
    gradient cosines still report it, which makes "did the reward-free rule
    happen to agree with the reward" a measurement instead of the rule's own
    input. What that buys, beyond not fitting the gate to the objective it is
    supposed to be independent of, is a source channel that is live from step 1
    at every destination -- the advantage rule was identically zero for two of
    the six pairs across the whole control window, and would be off for the ~70%
    of ``search`` groups whose rollouts all score the same.

    The residual is deliberately absent from this signature, as it was from the
    advantage version: it is what the reliability DIAGNOSTICS are still measured
    on, and keeping it out of the call is what stops the two from being confused
    at a call site.
    """
    return common.abs() + float(source_scale) * source_gate * exclusive.sum(dim=-1)


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


def resume_identity(snapshot: Optional[dict], task_order) -> dict:
    """The identity to check a sidecar against, completed where it can be.

    The worker snapshots what it knows when it loads the checkpoint -- the base,
    the temperature, the teachers. All three are CONFIG, available before a
    single batch has been seen. The task order is not: it is whatever the first
    batch names, and at load time no batch has arrived, so the snapshot recorded
    an empty list. Compared against a checkpoint written by a run that had
    completed a step, that could never match, and every resume of this arm died
    on ``task_order`` after the models were built and the checkpoint located.

    The order is known where the accumulators are built, because it is the axis
    they are indexed by -- which is also where the restore happens, for the same
    reason. Filling it in there is what makes the check test THIS RUN against the
    checkpoint rather than the snapshot's age against it. It does not weaken the
    check: a resume into a different task set, or the same tasks in a different
    order, still mismatches and is still refused.
    """
    out = dict(snapshot or {})
    out["task_order"] = list(task_order or [])
    return out


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
        # Rows offered to the pair, informative or not. Cumulative, with its own
        # step delta, on the same rule as everything else here.
        self.offered = torch.zeros(T * T, dtype=torch.float64, device=device)
        self.doffered = torch.zeros_like(self.offered)
        # Set by all_reduce: (moments, between, grouped, offered) for the step it
        # just folded. None before the first one.
        self._step = None
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
            # Every row the pair was OFFERED, before the informative filter. Its
            # own accumulator rather than a moment, because it must not be
            # divided by ``n`` the way the moments are -- ``n`` is what it is the
            # denominator of. A step where GRPO found no spread in any group
            # contributes rows here and none there, which is the one case a
            # correlation cannot report on itself: it simply goes missing.
            offered = ((dst >= 0) & (src >= 0)).to(torch.float64)
            self.doffered.index_add_(0, dst.clamp(min=0) * T + src.clamp(min=0), offered)
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
        self._cpu_cache = {}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (self.dbuf, self.group, self.doffered):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        T, G = self.n_tasks, self.max_groups
        grp = self.group.view(T * T, G, 2)
        totals, counts = grp[..., 0], grp[..., 1]
        step_between = torch.where(
            counts > 0, totals.square() / counts.clamp(min=1.0), torch.zeros_like(totals)
        ).sum(dim=-1)
        step_grouped = counts.sum(dim=-1)
        # This step alone, kept before the delta is cleared. rho estimated on
        # the cumulative moments is the one the loss uses and the one with the
        # sample size; rho on this step alone is the only thing that can say the
        # cumulative one has gone stale, which over 150 steps of a moving
        # student is the question. Each prompt group belongs to exactly one
        # step, so the between-group correction decomposes over steps and this
        # is an exact statistic rather than an approximation of one.
        self._step = (
            self.dbuf.detach().clone(), step_between.detach().clone(),
            step_grouped.detach().clone(), self.doffered.detach().clone(),
        )

        self.buf += self.dbuf
        self.dbuf.zero_()
        self.between += step_between
        self.grouped += step_grouped
        self.offered += self.doffered
        self.doffered.zero_()
        self.group.zero_()

    SCOPES = ("cumulative", "current")

    def _cpu(self, scope: str = "cumulative"):
        assert scope in self.SCOPES, scope
        if not isinstance(self._cpu_cache, dict):
            self._cpu_cache = {}
        if scope not in self._cpu_cache:
            T, M = self.n_tasks, self.n_moments
            if scope == "current":
                # The step's own contribution, stashed by all_reduce before it
                # cleared the delta. Empty before the first reduce, which renders
                # as no rows and therefore no rho -- the honest answer.
                if self._step is None:
                    zeros = torch.zeros(T * T * M, dtype=torch.float64)
                    step = (zeros, torch.zeros(T * T, dtype=torch.float64),
                            torch.zeros(T * T, dtype=torch.float64),
                            torch.zeros(T * T, dtype=torch.float64))
                else:
                    step = self._step
                self._cpu_cache[scope] = (
                    step[0].detach().to("cpu").view(T, T, M),
                    step[1].detach().to("cpu").view(T, T),
                    step[2].detach().to("cpu").view(T, T),
                    step[3].detach().to("cpu").view(T, T),
                    0.0,
                )
            else:
                self._cpu_cache[scope] = (
                    self.buf.detach().to("cpu").view(T, T, M),
                    self.between.detach().to("cpu").view(T, T),
                    self.grouped.detach().to("cpu").view(T, T),
                    self.offered.detach().to("cpu").view(T, T),
                    float(self.dbuf.detach().to("cpu").abs().sum()),
                )
        buf, between, grouped, offered, pending = self._cpu_cache[scope]
        assert pending == 0.0, (
            "AdvantageReliabilityStats was rendered with an unreduced step delta; "
            "call all_reduce() at the step boundary before reading alpha"
        )
        return buf, between, grouped, offered

    def _cell(self, dst: int, src: int, scope: str = "cumulative") -> dict:
        buf, between, grouped, offered = self._cpu(scope)
        cell = {name: float(buf[dst, src, i]) for i, name in enumerate(ADV_MOMENTS)}
        cell["_offered"] = float(offered[dst, src])
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
        cell["_grouped"] = float(grouped[dst, src])
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

    def alpha(self, *, task_names=None, scope: str = "cumulative") -> dict:
        """``{(dst_name, src_name): {...}}`` -- the applied alpha and its diagnostics.

        ``alpha`` is the only field the loss may read, and only at the default
        scope: ``current`` is this step's rows alone, which is a diagnostic of
        whether the cumulative estimate has gone stale and has neither the
        sample size nor the stability to weight anything.

        ``rho_lcb95`` and the two partials are reported next to it and never
        multiplied into anything: the first would introduce a confidence level,
        and the last two a choice of which confound to trust, and both are knobs.
        """
        out = {}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                cell = self._cell(dst, src, scope)
                if cell["n"] <= 0 and cell["_offered"] <= 0:
                    continue
                rho = self._corr(cell, "a", "s") if cell["n"] > 0 else None
                row = {
                    "n": cell["n"],
                    # The two spreads the correlation is a ratio of. A rho that
                    # collapses because the ADVANTAGE stopped varying is a
                    # different event from one that collapses because the source
                    # stopped saying anything, and the correlation alone reports
                    # them identically.
                    "adv_std": math.sqrt(max(self._cov(cell, "a", "a"), 0.0)),
                    "support_score_std": math.sqrt(max(self._cov(cell, "s", "s"), 0.0)),
                    # Rows that carried a usable prompt-group id, i.e. how much
                    # of the group centring the within-group variance actually
                    # rests on.
                    "n_grouped": cell.get("_grouped", 0.0),
                    # What fraction of the rows offered to this pair had a prompt
                    # group with a spread of advantages. GRPO is group-relative,
                    # so a group whose rollouts all scored the same gives every
                    # row zero advantage and contributes no information; a low
                    # fraction is why an alpha is small, and without it the two
                    # readings "the source does not predict reward" and "there
                    # was nothing to predict" are the same number.
                    "informative_group_frac": (
                        cell["n"] / cell["_offered"] if cell["_offered"] > 0 else None
                    ),
                    "rho": rho,
                    "alpha": 0.0 if rho is None else max(0.0, rho),
                    "rho_length_controlled": (
                        self._partial(cell, "a", "s", ["l"]) if cell["n"] > 0 else None
                    ),
                    "rho_length_on_controlled": (
                        self._partial(cell, "a", "s", ["l", "o"]) if cell["n"] > 0 else None
                    ),
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
            # Rows offered, so informative_group_frac survives a resume with the
            # denominator its numerator was accumulated against. A sidecar
            # written before this key existed loads as zero, which renders the
            # fraction as absent rather than as a ratio over a partial count.
            "offered": self.offered.detach().to("cpu"),
            "doffered": self.doffered.detach().to("cpu"),
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
            ("offered", self.offered), ("doffered", self.doffered),
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
    "kl_sq",
    "w_kl",
    "kl_shift_abs",
    "evidence",
    "evidence_shared",
    "evidence_shared_offtask_only",
    # The two channels as APPLIED budget rather than as evidence, plus the three
    # second moments their co-location needs. evidence/shared_share already says
    # which channel is larger; these say whether they are aiming at the same
    # positions, which is the only way two non-negative channels can compete.
    "push_shared",
    "push_source",
    "push_shared_sq",
    "push_source_sq",
    "push_cross",
    # The two source gates as MASS rather than as a rate, so "the mechanism is
    # quiet here" and "the gates closed here" stop being the same reading. Both
    # are sum_v p(v) * (...), the same expectation ``evidence`` is, so their
    # ratios against it and against each other are the pass rates of the two
    # stages the source now goes through.
    "source_gross",            # sum_v p sum_m |hat_m|      before either gate
    "source_exclusive_gross",  # sum_v p sum_m relu(|hat_m| - |hat_on|)
    "gate_mass",               # sum_v p q_sim
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

# What a probe series is for: the size of the source channel, held next to the
# size it is. The names are unchanged from the advantage era -- ``alpha000`` is
# still the corroboration channel alone and still the reading a control arm
# reproduces -- but the number is now a plain multiplier on the GATED source
# term, not a reliability. ``alpha100`` is therefore the shipped weight rather
# than an upper bracket; what the bracket used to say is now
# ``ungated_source``'s job, which is a counterfactual about the gate and not
# about the channel's size.
PROBE_ALPHAS = (0.0, 0.1, 1.0)

# The channel counterfactuals. Named rather than numbered because they are not a
# series: each removes a different part of the evidence, and reading them as a
# ladder would suggest an ordering they do not have.
#
#   no_shared       the source channel alone.
#   offtask_shared  the off-task-only agreement rule for the corroboration.
#   ungated_source  q_sim forced to 1. What the similarity gate COSTS, which is
#                   the number that says whether the gate is selecting or just
#                   attenuating.
#   shuffled_gate   q_sim recomputed on :func:`decorrelated_off_shifts`. What
#                   the gate would have found if the teachers had never been
#                   looking at the same candidate -- the null the shared base
#                   and the shared RL recipe can produce on their own. Its ratio
#                   to the live gate is the only evidence available here that
#                   agreement means two findings rather than one grammar.
CHANNEL_PROBES = ("no_shared", "offtask_shared", "ungated_source", "shuffled_gate")

# The per-position scopes. Role comes from the tag scan; turn from the runs of
# ones in the multi-turn loss mask.
#
# The last turn bucket is open-ended because the tail is thin and uneven -- a
# search episode that needed nine retrieval rounds is one row, and a scope with
# one row in it publishes a ratio built from one row. Eight is where the
# multitask rollout's turn counts stop being dense.
TURN_BUCKETS = 6
TURN_SCOPE_NAMES = tuple(f"turn{i}" for i in range(TURN_BUCKETS - 1)) + (
    f"turn{TURN_BUCKETS - 1}plus",
)
ROLE_SCOPE_NAMES = tuple(_ROLE_NAMES[i] for i in sorted(_ROLE_NAMES))

# What a per-position cut PUBLISHES, against everything its accumulator holds.
#
# position_weight_metrics renders eighteen columns and state_shift_metrics
# sixteen. Six roles and six turns of all of them is over three hundred series a
# step, which is not a richer analysis than the arm already had -- it is an
# unreadable one, and every extra series is another thing a reader has to decide
# is not the finding. The accumulators keep every column; these are the ones
# that answer the cut's own question, and the token and event dumps carry the
# depth that does not fit in a column.
ROLE_CUT_SUFFIXES = (
    "/effect/kl_unweighted",        # where this role's OPD budget IS
    "/effect/kl_shift_gross_frac",  # how much of it the arm moved here
    "/effect/kl_shift_net",         # and in which direction
    "/position/w_mean",
    "/evidence/shared_share",       # which channel carried it here
    "/grpo/grad_cosine",            # does the budget moved here pull with the reward
    "/grpo/shared_grad_cosine",     # and which CHANNEL's budget, at this role
    "/grpo/source_grad_cosine",
    "/grpo/grad_norm_ratio",
    "/gross_share",                 # the state composition; shares, never nats
)

# Turn is the thinner cut: what is wanted is whether the arm is front-loaded,
# not a second copy of the role analysis indexed differently.
TURN_CUT_SUFFIXES = (
    "/effect/kl_unweighted",
    "/effect/kl_shift_gross_frac",
    "/position/w_mean",
    "/position/available_frac",
)


def select_metrics(rendered: dict, suffixes) -> dict:
    """The published subset of a rendered cut. Suffixes, so a scope name in the
    middle of the key cannot make a series appear or disappear."""
    keep = tuple(suffixes)
    return {k: v for k, v in rendered.items() if k.endswith(keep)}


def probe_name(alpha: float) -> str:
    return f"alpha{int(round(float(alpha) * 100)):03d}"




def pair_state_index(*, hat_on, hat_off, deadzone: float) -> torch.Tensor:
    """(bs, resp, k, n_off) index into :data:`PAIR_STATES`.

    Labels only, on the STANDARDIZED shifts against a zero base, so the deadzone
    is in RMS units and means the same thing for every teacher -- the same rule
    the state table and the token tables use, and the reason they can be read
    side by side.
    """
    on = torch.sign(hat_on.detach()) * (hat_on.detach().abs() > deadzone)
    off = torch.sign(hat_off.detach()) * (hat_off.detach().abs() > deadzone)
    on_c = on.unsqueeze(-1)
    speaks = off != 0
    on_silent = on_c == 0
    agree = (~on_silent) & speaks & (off == on_c)
    conflict = (~on_silent) & speaks & (off != on_c)
    blind = on_silent & speaks
    idx = torch.full_like(off, PAIR_STATES.index("source_silent"), dtype=torch.long)
    idx = torch.where(agree, torch.full_like(idx, PAIR_STATES.index("agree")), idx)
    idx = torch.where(conflict, torch.full_like(idx, PAIR_STATES.index("conflict")), idx)
    idx = torch.where(
        blind, torch.full_like(idx, PAIR_STATES.index("on_silent_source_active")), idx
    )
    return idx


class PairStateEvidenceStats:
    """Per ordered (destination, source) AND per disagreement state.

    The two tables it sits between each collapse the axis the other keeps:
    ``kl_shift_by_state`` sums the sources out, ``evidence/{src}__on__{dst}``
    sums the states out. Neither can answer "when Search disagreed with
    AlfWorld's own teacher, what did the arm do" -- and arbitrating exactly that
    is what the mechanism is for, so it is the table the write-up is built on.

    ``(T, T, 4, 5)`` float64 for the candidate table plus ``(T, T, 1 + 4)`` for
    the position one -- 234 cells at three tasks. The cost is one ``index_add_``
    over the candidate axis per source per micro-batch, and one more over the
    positions.

    TWO COUNTING UNITS, kept apart because they are routinely confused. The
    candidate table counts TOP-K CANDIDATES: a position with k = 20 contributes
    twenty of them, filed under whatever state each one is in, so a share out of
    it answers "of all the candidate slots, how many were conflicts". The
    position table counts POSITIONS, filed by whether ANY candidate there was in
    the state, and answers "at how many positions did this source disagree at
    all". A position is in exactly one candidate state only by accident; it is
    routinely in several at once, so the position shares do NOT sum to 1 and are
    not meant to.

    Naming the first one ``position_frac``, as this class did until the fraction
    was checked against the code, states the second and reports the first.
    """

    TERMS = ("n", "evidence", "shift", "shift_abs", "activity")

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.n_states = S = len(PAIR_STATES)
        self.buf = torch.zeros(T * T * S * len(self.TERMS), dtype=torch.float64, device=device)
        # [n_positions, any_state_0 .. any_state_{S-1}] per ordered pair.
        self.pos_buf = torch.zeros(T * T * (1 + S), dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, state, evidence, shift, response_mask, task_ids, off_plane_tasks,
               activity=None) -> None:
        """``state``/``evidence``/``shift``/``activity`` are (bs, resp, k, n_off).

        ``activity`` is the source's PRE-ALPHA evidence, ``sum_v p(v)|dhat_m(v)|``
        per candidate. None leaves the column at zero rather than reusing
        ``evidence``, which would make a vetoed source indistinguishable from a
        silent one -- the exact reading the column exists to provide.
        """
        self._cpu_cache = None
        T, S, K = self.n_tasks, self.n_states, len(self.TERMS)
        # (bs, resp, 1): the mask is per position, and every quantity below is
        # sliced to (bs, resp, k) before it is used.
        m = response_mask.to(torch.float64).unsqueeze(-1)
        e = evidence.detach().to(torch.float64) * m.unsqueeze(-1)
        s = shift.detach().to(torch.float64) * m.unsqueeze(-1)
        a = (
            torch.zeros_like(e) if activity is None
            else activity.detach().to(torch.float64) * m.unsqueeze(-1)
        )
        dst = task_ids.reshape(-1, 1, 1).expand_as(state[..., 0])
        mp = response_mask.to(torch.float64)                       # (bs, resp)
        dst_p = task_ids.reshape(-1, 1).expand_as(mp)
        for c in range(state.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1, 1, 1).expand_as(state[..., c])
            ok = ((dst >= 0) & (src >= 0)).to(torch.float64)
            cell = ((dst.clamp(min=0) * T + src.clamp(min=0)) * S + state[..., c]) * K
            vals = torch.stack(
                [m.expand_as(e[..., c]) * ok, e[..., c] * ok, s[..., c] * ok,
                 s[..., c].abs() * ok, a[..., c] * ok],
                dim=-1,
            )
            flat = (cell.unsqueeze(-1) + torch.arange(K, device=vals.device)).reshape(-1)
            self.buf.index_add_(0, flat, vals.reshape(-1))

            # The position cut. ``any`` over the candidate axis, so a position
            # where one candidate conflicted and nineteen were silent counts
            # once for conflict -- which is the sentence the write-up makes and
            # the candidate table above cannot support.
            src_p = off_plane_tasks[:, c].reshape(-1, 1).expand_as(mp)
            ok_p = ((dst_p >= 0) & (src_p >= 0)).to(torch.float64) * mp
            base_p = (dst_p.clamp(min=0) * T + src_p.clamp(min=0)) * (1 + S)
            cols = [ok_p] + [
                ((state[..., c] == st).any(dim=-1).to(torch.float64) * ok_p)
                for st in range(S)
            ]
            vals_p = torch.stack(cols, dim=-1)
            flat_p = (base_p.unsqueeze(-1) + torch.arange(1 + S, device=vals_p.device)).reshape(-1)
            self.pos_buf.index_add_(0, flat_p, vals_p.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.pos_buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = (
                self.buf.detach().to("cpu").view(
                    self.n_tasks, self.n_tasks, self.n_states, len(self.TERMS)
                ),
                self.pos_buf.detach().to("cpu").view(
                    self.n_tasks, self.n_tasks, 1 + self.n_states
                ),
            )
        return self._cpu_cache

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        buf, pos = self._cpu()
        i = {t: j for j, t in enumerate(self.TERMS)}
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                cells = buf[dst, src]
                total_n = float(cells[:, i["n"]].sum())
                total_abs = float(cells[:, i["shift_abs"]].sum())
                if total_n <= 0:
                    continue
                head = f"{prefix}/pair_state/{name(src)}__on__{name(dst)}"
                out[f"{head}/candidate_count"] = total_n
                n_pos = float(pos[dst, src, 0])
                if n_pos > 0:
                    out[f"{head}/n_positions"] = n_pos
                for st in range(self.n_states):
                    sname = PAIR_STATES[st]
                    n = float(cells[st, i["n"]])
                    # Of the CANDIDATE slots, not of the positions. The two
                    # differ by a factor of k and answer different questions;
                    # the position one is the ``position_any_frac`` below.
                    out[f"{head}/{sname}/candidate_frac"] = n / total_n
                    if n_pos > 0:
                        out[f"{head}/{sname}/position_any_frac"] = (
                            float(pos[dst, src, 1 + st]) / n_pos
                        )
                    if n <= 0:
                        continue
                    out[f"{head}/{sname}/source_evidence_mean"] = (
                        float(cells[st, i["evidence"]]) / n
                    )
                    # The same evidence with alpha divided out. Where this is
                    # large and source_evidence_mean is near zero, the source
                    # spoke and the reliability gate refused it -- which is not
                    # the same finding as the source having said nothing, and is
                    # invisible in every other column here.
                    out[f"{head}/{sname}/source_activity_pre_alpha_mean"] = (
                        float(cells[st, i["activity"]]) / n
                    )
                    out[f"{head}/{sname}/kl_shift_net"] = float(cells[st, i["shift"]]) / n
                    if total_abs > 0:
                        out[f"{head}/{sname}/kl_shift_gross_share"] = (
                            float(cells[st, i["shift_abs"]]) / total_abs
                        )
        return out


class CorroborationAttributionStats:
    """Per destination, which teacher decided the corroboration and which one capped it.

    The shared channel carries ~98% of this arm's evidence on the live run, and
    every existing table sums the corroboration into ONE anonymous column --
    ``evidence_shared``, ``push_shared``. That is not an oversight of the
    per-source tables: ``c = 1[all agree] * sign * min_j |hat_j|`` is a minimum
    over a unanimity, so it is NOT additive over sources and there is no share
    of it to hand to a per-source accumulator. Attribution has to be a
    counterfactual, which is what :func:`corroboration_attribution` computes and
    this class accumulates.

    Two readings per (destination, teacher), and they answer opposite questions:

    ``bottleneck_share``   of the corroboration mass ``sum_v p(v)|c(v)|`` that
                           the arm ACTUALLY applied, how much this teacher was
                           the binding minimum for. A teacher at 0.8 here is
                           the one setting the bonus; a teacher at 0.02 agreed
                           and was never consulted about the size.
    ``suppression_ratio``  ``sum_v p(v)(|c_-j(v)| - |c(v)|)`` over the SAME
                           total -- the corroboration that does not exist
                           because this teacher is in the vote, whether it
                           capped the minimum or vetoed the unanimity outright.
                           A teacher can be 0 on the first and dominant on the
                           second: that is a teacher who never sets the bonus
                           and frequently cancels it, which no current metric
                           can express.

    The on-task teacher gets a slot of its own (``on_task``) rather than a task
    name, because in that cell "destination" and "teacher" are the same task and
    a ``{src}__on__{dst}`` label would read as a self-pair. It is the cell the
    module docstring's claim rests on -- ``c`` is capped by ``|hat_on|``, and
    the on-task teacher is silent at ~64% of teacher mass -- so it is the one
    that has to be legible.

    ``near_tie_share`` guards the first reading. ``argmin`` names a teacher even
    when two are within floating-point noise of each other, and a share built
    out of coin flips looks exactly like a share built out of decisions. Where
    this is high the bottleneck column is not a finding.

    ``(T, T, 5)`` float64 plus ``(T, 2)``: 51 cells at three tasks, one
    ``index_add_`` per column per micro-batch.
    """

    TERMS = ("bottleneck_mass", "tie_mass", "suppression", "n_bottleneck", "n_seen")
    TOTALS = ("shared_mass", "n_candidates")

    def __init__(self, *, n_tasks: int, device, tie_epsilon: float = 0.05):
        self.n_tasks = T = int(n_tasks)
        # ``tie_epsilon`` is in RMS units: the margin is a difference of
        # standardized |hat|, so one threshold is comparable across teachers and
        # across steps, which a raw-nats one would not be.
        self.tie_epsilon = float(tie_epsilon)
        self.buf = torch.zeros(T * T * len(self.TERMS), dtype=torch.float64, device=device)
        self.tot = torch.zeros(T * len(self.TOTALS), dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, attribution: dict, common, teacher_prob, response_mask,
               task_ids, off_plane_tasks) -> None:
        """``common``/``teacher_prob`` are (bs, resp, k); ``attribution`` is what
        :func:`corroboration_attribution` returned for the same call.

        Weighted by ``p(v)|c(v)|``, the per-candidate term of ``evidence_shared``
        -- so a share out of it is a share of the corroboration the objective
        applied, not of the candidate slots. Weighting by candidate count would
        let a million near-zero bonuses outvote the ones that moved the loss.
        """
        self._cpu_cache = None
        T, K = self.n_tasks, len(self.TERMS)
        m = response_mask.to(torch.float64).unsqueeze(-1)              # (bs, resp, 1)
        p = teacher_prob.detach().to(torch.float64) * m
        c_abs = common.detach().to(torch.float64).abs()
        w = p * c_abs                                                  # (bs, resp, k)
        # Where c is 0 there is no argmin to attribute -- w is already 0 there,
        # but the COUNT would otherwise credit a teacher for winning a minimum
        # that was never used.
        live = common.detach() != 0
        bott = attribution["bottleneck"]
        near = (attribution["margin"].detach().to(torch.float64) < self.tie_epsilon)
        without = attribution["without"].detach().to(torch.float64)
        n_all = without.size(-1)

        dst = task_ids.reshape(-1).to(torch.long)
        # Column 0 is the on-task teacher, so its slot IS the destination task.
        cols = torch.cat([dst.reshape(-1, 1), off_plane_tasks.to(torch.long)], dim=1)
        dst_e = dst.reshape(-1, 1, 1).expand_as(c_abs)
        ok_dst = (dst_e >= 0)

        for j in range(n_all):
            tea = cols[:, j].reshape(-1, 1, 1).expand_as(c_abs)
            ok = (ok_dst & (tea >= 0)).to(torch.float64)
            is_bn = ((bott == j) & live).to(torch.float64) * ok
            vals = torch.stack(
                [
                    w * is_bn,
                    w * is_bn * near.to(torch.float64),
                    # Over EVERY candidate, not just the live ones: a teacher
                    # that vetoed the unanimity leaves c = 0, which is exactly
                    # where its suppression is largest and where the bottleneck
                    # column above is blind.
                    p * (without[..., j] - c_abs).clamp(min=0.0) * ok,
                    m.expand_as(c_abs) * is_bn,
                    # Candidates where this teacher was one of THIS row's
                    # planes at all. The gate the metrics use, so a teacher that
                    # was consulted everywhere and suppressed nothing reports a
                    # zero -- which is a finding -- instead of vanishing from
                    # the table the way a task pair that never occurred does.
                    m.expand_as(c_abs) * ok,
                ],
                dim=-1,
            )
            cell = (dst_e.clamp(min=0) * T + tea.clamp(min=0)) * K
            flat = (cell.unsqueeze(-1) + torch.arange(K, device=vals.device)).reshape(-1)
            self.buf.index_add_(0, flat, vals.reshape(-1))

        ok_d = ok_dst.to(torch.float64)
        tot_vals = torch.stack([w * ok_d, m.expand_as(c_abs) * ok_d], dim=-1)
        base = dst_e.clamp(min=0) * len(self.TOTALS)
        flat_t = (base.unsqueeze(-1) + torch.arange(len(self.TOTALS), device=w.device)).reshape(-1)
        self.tot.index_add_(0, flat_t, tot_vals.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.tot, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = (
                self.buf.detach().to("cpu").view(self.n_tasks, self.n_tasks, len(self.TERMS)),
                self.tot.detach().to("cpu").view(self.n_tasks, len(self.TOTALS)),
            )
        return self._cpu_cache

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        buf, tot = self._cpu()
        i = {t: j for j, t in enumerate(self.TERMS)}
        g = {t: j for j, t in enumerate(self.TOTALS)}
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for dst in range(self.n_tasks):
            shared = float(tot[dst, g["shared_mass"]])
            n_cand = float(tot[dst, g["n_candidates"]])
            if n_cand <= 0:
                continue
            head = f"{prefix}/corroboration/{name(dst)}"
            out[f"{head}/shared_mass_mean"] = shared / n_cand
            if shared <= 0:
                continue
            for tea in range(self.n_tasks):
                cells = buf[dst, tea]
                n_seen = float(cells[i["n_seen"]])
                if n_seen <= 0:
                    continue
                bn = float(cells[i["bottleneck_mass"]])
                sup = float(cells[i["suppression"]])
                n_bn = float(cells[i["n_bottleneck"]])
                slot = "on_task" if tea == dst else name(tea)
                out[f"{head}/{slot}/bottleneck_share"] = bn / shared
                out[f"{head}/{slot}/bottleneck_candidate_frac"] = n_bn / n_seen
                # Of the applied corroboration mass, how much MORE there would
                # be without this teacher. Above 1 means the teacher is
                # cancelling more than the arm is applying.
                out[f"{head}/{slot}/suppression_ratio"] = sup / shared
                if bn > 0:
                    out[f"{head}/{slot}/near_tie_share"] = (
                        float(cells[i["tie_mass"]]) / bn
                    )
        return out


class PairEvidenceStats:
    """Per ordered (destination, source), how much evidence that source supplied.

    ``evidence`` is that source's share of ``W~ - 1``; ``shift`` is its share of
    the nats the weighting actually moved. Kept apart from the per-state table
    because they answer different questions -- "who supplied it" against "what
    kind of position received it" -- and a single table keyed by both would be
    mostly empty.
    """

    # ``support_mass`` is the source teacher's probability that lands INSIDE the
    # student's top-k at all. Without it, "Search contributed little to
    # AlfWorld" has three explanations that read identically -- the signal is
    # weak, alpha is low, or Search's vocabulary is simply not in the support
    # the whole mechanism is measured on -- and only the third is a measurement
    # artefact rather than a finding. Free: the log-probs are already gathered.
    # ``activity`` is the SAME evidence with alpha taken out. Its whole purpose
    # is the case where ``evidence`` is zero: alpha = 0 makes a source that
    # never spoke and a source whose every word the reliability gate refused
    # produce the identical column, and those are opposite findings about the
    # mechanism. Every planned change to how alpha is estimated is an experiment
    # on the gap between these two numbers.
    TERMS = ("evidence", "shift", "support_mass", "n", "activity")

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.buf = torch.zeros(T * T * len(self.TERMS), dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, evidence, shift, response_mask, task_ids, off_plane_tasks,
               support_mass=None, activity=None) -> None:
        """``evidence`` and ``shift`` are (bs, resp, n_off).

        Args:
            support_mass: (bs, resp, n_off) the source teacher's probability
                summed over the student's top-k, or None to leave the column at
                zero rather than guessing a coverage that was not measured.
            activity: (bs, resp, n_off) the pre-alpha evidence, or None to leave
                that column at zero. NOT defaulted to ``evidence``: the column
                exists precisely to differ from it.
        """
        self._cpu_cache = None
        T, K = self.n_tasks, len(self.TERMS)
        m = response_mask.to(torch.float64).unsqueeze(-1)
        e = evidence.detach().to(torch.float64) * m
        s = shift.detach().to(torch.float64) * m
        cov = (
            torch.zeros_like(e) if support_mass is None
            else support_mass.detach().to(torch.float64) * m
        )
        act = (
            torch.zeros_like(e) if activity is None
            else activity.detach().to(torch.float64) * m
        )
        dst = task_ids.reshape(-1).to(torch.long)
        for c in range(e.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = ((dst >= 0) & (src >= 0)).to(torch.float64)
            vals = torch.stack(
                [e[..., c].sum(dim=1) * ok, s[..., c].sum(dim=1) * ok,
                 cov[..., c].sum(dim=1) * ok,
                 (response_mask.to(torch.float64).sum(dim=1)) * ok,
                 act[..., c].sum(dim=1) * ok],
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
        i = {t: j for j, t in enumerate(self.TERMS)}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst or float(buf[dst, src, i["n"]]) <= 0:
                    continue
                n = float(buf[dst, src, i["n"]])
                head = f"{prefix}/evidence/{name(src)}__on__{name(dst)}"
                out[f"{head}/source_shift_mean"] = float(buf[dst, src, i["evidence"]]) / n
                out[f"{head}/kl_shift_attributed"] = float(buf[dst, src, i["shift"]]) / n
                # Read the two together. Both near zero: this source has nothing
                # to say to this destination. Activity large, shift near zero:
                # it had plenty and alpha refused it.
                out[f"{head}/source_activity_pre_alpha_mean"] = (
                    float(buf[dst, src, i["activity"]]) / n
                )
                mass = float(buf[dst, src, i["support_mass"]]) / n
                if mass > 0:
                    out[f"{head}/support_mass"] = mass
                    # What the mechanism CANNOT see of this source, because the
                    # measurement runs on the student's top-k and the teacher's
                    # mass is elsewhere.
                    out[f"{head}/tail_mass"] = 1.0 - mass
        return out


class PositionScopeTermStats:
    """:class:`ScopeTermStats`, keyed per POSITION instead of per row.

    The task scope is a property of the row, so ``ScopeTermStats`` sums a row's
    positions first and files the total once. Role is a property of the POSITION
    -- one response walks through ``<think>``, ``<action>`` and the tag tokens
    between them -- so the same layout with a per-position index is a different
    class rather than a flag: filing a whole row under the role of its first
    token would report the arm acting on reasoning wherever a response happened
    to open with ``<think>``.

    Layout, all_reduce and float64 are ``ScopeTermStats``'s, and :meth:`sums`
    returns the same shape, so :func:`position_weight_metrics`,
    :func:`state_shift_metrics` and :func:`gradient_metrics` render this without
    knowing which of the two produced it.

    The pooled scope is accumulated but NOT rendered by default: it is the same
    number the task-scoped accumulator already publishes at the top level, and
    two keys for one quantity is how two series come to disagree after a
    refactor. It is kept in the buffer because every ratio here divides by the
    scope's own count and a reader checking that the parts sum to the whole
    needs the whole.
    """

    def __init__(self, *, names, n_scopes: int, device):
        self.names = list(names)
        self.n_scopes = 1 + int(n_scopes)
        self.buf = torch.zeros(
            (self.n_scopes, len(self.names) + 1), dtype=torch.float64, device=device
        )
        self._cpu_cache = None

    def update(self, terms: dict, response_mask: torch.Tensor, scope_ids: torch.Tensor) -> None:
        """Fold one micro-batch in.

        Args:
            terms: ``{name: (bs, resp) tensor}``; every name given at
                construction must be present.
            response_mask: (bs, resp).
            scope_ids: (bs, resp) int, or None. A negative id, or one past the
                last scope, reaches the pooled scope only -- the same rule
                ``ScopeTermStats`` applies to an untagged row.
        """
        self._cpu_cache = None
        m = response_mask.to(torch.float64)
        cols = [terms[name].detach().to(torch.float64) * m for name in self.names]
        cols.append(m)
        vals = torch.stack(cols, dim=-1).reshape(-1, len(self.names) + 1)
        self.buf[0] += vals.sum(0)
        if scope_ids is None or self.n_scopes <= 1:
            return
        s = scope_ids.reshape(-1).to(torch.long)
        known = ((s >= 0) & (s < self.n_scopes - 1)).to(torch.float64).unsqueeze(-1)
        self.buf.index_add_(0, s.clamp(min=0, max=self.n_scopes - 2) + 1, vals * known)

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu")
        return self._cpu_cache

    def sums(self, scope_names=None, include_pooled: bool = False) -> dict:
        """``{scope_name_or_None: {term: total, "n": count}}``."""
        buf = self._cpu()
        out = {}
        for scope in range(0 if include_pooled else 1, self.n_scopes):
            n = float(buf[scope, -1])
            if n <= 0:
                continue
            name = None if scope == 0 else (
                scope_names[scope - 1]
                if scope_names and scope - 1 < len(scope_names)
                else f"scope{scope - 1}"
            )
            out[name] = {t: float(buf[scope, i]) for i, t in enumerate(self.names)}
            out[name]["n"] = n
        return out


# Where the applied weight sits. Chosen so that the two readings a threshold has
# to answer exactly -- "how often did the arm take budget AWAY" (W < 1, which
# only the normaliser can produce, since W~ >= 1 by construction) and "how often
# did it move a position by more than a few percent" -- fall on bucket
# boundaries rather than inside a bucket, where they would have to be
# interpolated and would stop being counts.
WEIGHT_BUCKET_EDGES = (0.5, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0)

# The cut points reported as exact shares. Each is an entry of
# WEIGHT_BUCKET_EDGES, so "above t" is a sum of whole buckets. 0.99 is not among
# them because the below-one pair already reports that side, and a series and its
# complement are one reading in two columns.
WEIGHT_THRESHOLDS = (1.05, 1.25, 2.0)
BELOW_ONE_EDGE = 0.99


def _threshold_name(t: float) -> str:
    return f"{int(round(float(t) * 100)):03d}"


class WeightShiftHistogram:
    """How the applied weight is DISTRIBUTED, and where the moved nats sit in it.

    ``w_cv`` is the only shape reading the arm has, and a coefficient of
    variation cannot tell a weight that is 1.02 nearly everywhere from one that
    is 1.00 at 99% of positions and 3.0 at the rest. Those are different
    mechanisms: the first is a slightly larger ``teacher_kl_loss_coef`` wearing
    a disguise, the second is a genuine redistribution, and an ablation that
    cannot separate them cannot attribute a gain to either.

    Three columns per bucket, because the question is not only how the weight is
    spread but whether the KL was where the weight was:

    ``n``          positions in the bucket.
    ``kl``         the UNWEIGHTED OPD term they carried. A weight of 3.0 at a
                   position whose KL is zero changes the loss by zero, so a
                   histogram of ``n`` alone can report a large tail that moved
                   nothing.
    ``shift_abs``  ``|W - 1| * D``, the gross nats those positions moved. Its
                   share above a threshold is what "the mechanism is carried by
                   the tail" means as a number.

    Cheap by construction: one ``bucketize`` and two ``index_add_`` per
    micro-batch into a ``(1 + n_tasks, 14, 3)`` buffer, and no host read until
    the step boundary.
    """

    TERMS = ("n", "kl", "shift_abs")

    def __init__(self, *, n_tasks: int, device):
        self.n_scopes = 1 + int(n_tasks)
        self.n_buckets = len(WEIGHT_BUCKET_EDGES) + 1
        self.edges = torch.tensor(WEIGHT_BUCKET_EDGES, dtype=torch.float64, device=device)
        self.buf = torch.zeros(
            (self.n_scopes, self.n_buckets, len(self.TERMS)), dtype=torch.float64, device=device
        )
        self._cpu_cache = None

    def update(self, *, weight, teacher_kl, response_mask, task_ids=None) -> None:
        """``weight`` and ``teacher_kl`` are (bs, resp); ``task_ids`` is (bs,)."""
        self._cpu_cache = None
        m = response_mask.to(torch.float64)
        w = weight.detach().to(torch.float64)
        kl = teacher_kl.detach().to(torch.float64) * m
        # right=False makes bucket j the half-open (edges[j-1], edges[j]], so a
        # position exactly AT a threshold falls below it and "above t" stays a
        # sum of whole buckets.
        b = torch.bucketize(w, self.edges).reshape(-1)
        # Masked positions contribute exactly zero to all three columns -- their
        # bucket index is arbitrary and harmless, which is why the weight does
        # not need masking first.
        vals = torch.stack([m, kl, (w - 1.0).abs() * kl], dim=-1).reshape(-1, len(self.TERMS))
        flat = self.buf.view(-1, len(self.TERMS))
        flat.index_add_(0, b, vals)
        if task_ids is None or self.n_scopes <= 1:
            return
        t = task_ids.reshape(-1, 1).expand(-1, w.size(1)).reshape(-1).to(torch.long)
        known = ((t >= 0) & (t < self.n_scopes - 1)).to(torch.float64).unsqueeze(-1)
        idx = (t.clamp(min=0, max=self.n_scopes - 2) + 1) * self.n_buckets + b
        flat.index_add_(0, idx, vals * known)

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu")
        return self._cpu_cache

    @staticmethod
    def _quantile(counts, q: float) -> float:
        """Linear interpolation inside the bucket the quantile lands in.

        The top bucket is unbounded, so a quantile that lands there is reported
        as its lower edge -- an understatement, and the ``frac_w_gt_*`` series
        beside it is what says how much mass is out there. Said in the name:
        nothing here is called a maximum.
        """
        total = sum(counts)
        if total <= 0:
            return float("nan")
        target = q * total
        run = 0.0
        n_edges = len(WEIGHT_BUCKET_EDGES)
        for i, c in enumerate(counts):
            if c > 0 and run + c >= target:
                lo = 0.0 if i == 0 else WEIGHT_BUCKET_EDGES[i - 1]
                hi = WEIGHT_BUCKET_EDGES[i] if i < n_edges else WEIGHT_BUCKET_EDGES[-1]
                return lo + (hi - lo) * ((target - run) / c)
            run += c
        return float(WEIGHT_BUCKET_EDGES[-1])

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        buf = self._cpu()
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for scope in range(self.n_scopes):
            counts = [float(v) for v in buf[scope, :, 0]]
            total = sum(counts)
            if total <= 0:
                continue
            head = prefix if scope == 0 else f"{prefix}/{name(scope - 1)}"
            kl = [float(v) for v in buf[scope, :, 1]]
            shift = [float(v) for v in buf[scope, :, 2]]
            kl_total, shift_total = sum(kl), sum(shift)
            for q in (0.5, 0.9, 0.99):
                out[f"{head}/shape/w_q{int(round(q * 100)):02d}"] = self._quantile(counts, q)
            for t in WEIGHT_THRESHOLDS:
                i = WEIGHT_BUCKET_EDGES.index(t)
                tag = _threshold_name(t)
                out[f"{head}/shape/frac_w_gt_{tag}"] = sum(counts[i + 1 :]) / total
                if kl_total > 0:
                    out[f"{head}/shape/kl_share_w_gt_{tag}"] = sum(kl[i + 1 :]) / kl_total
                if shift_total > 0:
                    out[f"{head}/shape/shift_share_w_gt_{tag}"] = sum(shift[i + 1 :]) / shift_total
            # The one direction W~ cannot reach on its own: below 1 is the
            # normaliser taking budget away, and how often that happens is what
            # separates "the arm added distillation" from "the arm moved it".
            below = WEIGHT_BUCKET_EDGES.index(BELOW_ONE_EDGE) + 1
            out[f"{head}/shape/frac_w_below_one"] = sum(counts[:below]) / total
            if shift_total > 0:
                out[f"{head}/shape/shift_share_w_below_one"] = sum(shift[:below]) / shift_total
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
    # Who decided ``c``, and who is holding it down. Same tensors, no extra
    # forward: with the shared channel at ~98% of this arm's evidence, an
    # unattributed ``min`` over a unanimity is most of the mechanism reported
    # as a single anonymous number.
    attribution = corroboration_attribution(hat_on=hat_on, hat_off=hat_off)
    # Hoisted above the weight so the W - 1 decomposition and the returned
    # per-candidate evidence are built from ONE tensor. Two exp() calls of the
    # same log-probs agree to the last bit today and are an invitation to
    # diverge the moment one of them grows a mask.
    p_teacher = on_task_logprob.detach().to(torch.float32).exp()
    alpha = alpha_table.to(hat_off.device)
    dst = task_ids.reshape(-1).to(torch.long).clamp(min=0)
    src = off_plane_tasks.to(torch.long).clamp(min=0)
    # KEPT AND NOT APPLIED. The advantage reliability is a diagnostic now: it is
    # returned so the outcome tables can still ask whether the reward-free gate
    # happened to land where the reward would have, and it reaches no evidence,
    # no probe and no weight in this function.
    row_alpha = alpha[dst.unsqueeze(-1), src]                       # (bs, n_off)

    # The two source factors, at the candidate. Both from the same standardized
    # shifts everything else here is built from -- no accumulator, no history,
    # nothing that has to warm up before the channel is live.
    q_sim = teacher_similarity(hat_off)
    exclusive = source_exclusive_shift(hat_on=hat_on, hat_off=hat_off)

    evidence = candidate_kl_evidence(
        common=dec["common_soft"], source_gate=q_sim, exclusive=exclusive
    )
    # The off-task-only variant, computed and never applied. It answers the one
    # question the all-teacher rule cannot: how much corroboration the on-task
    # teacher's silence is costing, at the 64% of teacher mass where it says
    # nothing. A counterfactual, so it is reported beside the applied share and
    # reaches no weight.
    evidence_offtask_only = candidate_kl_evidence(
        common=dec["common_ev"], source_gate=q_sim, exclusive=exclusive
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

    # W - 1 SPLIT EXACTLY THREE WAYS, made here so every reader gets the same
    # split. W~ = 1 + B_shared + sum_m B_m with
    #
    #     B_shared = sum_v p(v) |c(v)|                      corroboration
    #     B_m      = sum_v p(v) q(v) relu(|dhat_m(v)| - |dhat_on(v)|)   source m
    #
    # and W = W~/mu, so
    #
    #     W - 1 = B_shared/mu + sum_m B_m/mu + (1/mu - 1)
    #
    # The last term is not a rounding artefact: mu is a whole-task divisor, so a
    # position with NO evidence at all still moves by (1/mu - 1) and the arm has
    # to own that. Charging it to a source, or leaving it out so the parts do not
    # add up, are the two ways this gets misreported.
    #
    # The gates make the identity hold in all four states rather than only the
    # healthy one. Where pre was replaced by 1 the evidence terms are zero and
    # the normaliser term alone is W - 1; where the weight was replaced by 1 all
    # three are zero, which is what W - 1 is there.
    inv_mu = 1.0 / mu.clamp(min=1e-12)
    ok_evidence = (mu_valid.reshape(-1, 1) & finite_pre & finite_w).to(pre.dtype)
    ok_normalizer = (mu_valid.reshape(-1, 1) & finite_w).to(pre.dtype)
    evidence_shared_sum = (p_teacher * dec["common_soft"].abs()).sum(dim=-1)
    evidence_by_source_cand = p_teacher.unsqueeze(-1) * q_sim.unsqueeze(-1) * exclusive
    push_shared = evidence_shared_sum * inv_mu * ok_evidence
    push_by_source = (
        evidence_by_source_cand.sum(dim=2) * inv_mu.unsqueeze(-1) * ok_evidence.unsqueeze(-1)
    )
    push_normalizer = (inv_mu - 1.0) * ok_normalizer

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

    # CHANNEL counterfactuals, against the alpha series' MAGNITUDE ones. The
    # probes ask "what if the advantage channel were scaled differently"; these
    # ask "what if a channel were not there at all", which is the ablation the
    # design decisions were actually made on:
    #
    #   no_shared       the advantage channel alone -- is the corroboration term
    #                   carrying the mechanism, or is it decoration on top of
    #                   "reliable source activity"?
    #   offtask_shared  the off-task-only agreement rule, the D-2 option this
    #                   arm chose against. Its cost is the on-task teacher's
    #                   silence at 64% of teacher mass, and this is that cost as
    #                   a weight rather than as an evidence total.
    #
    # Both reuse this call's alphas, its standardization and its availability
    # mask, so the only difference from the shipped weight is the channel.
    channels = {
        "no_shared": candidate_kl_evidence(
            common=torch.zeros_like(dec["common_soft"]),
            source_gate=q_sim, exclusive=exclusive,
        ),
        "offtask_shared": evidence_offtask_only,
        "ungated_source": candidate_kl_evidence(
            common=dec["common_soft"],
            source_gate=torch.ones_like(q_sim), exclusive=exclusive,
        ),
        "shuffled_gate": candidate_kl_evidence(
            common=dec["common_soft"],
            source_gate=teacher_similarity(decorrelated_off_shifts(hat_off)),
            exclusive=exclusive,
        ),
    }
    channel_pre = {}
    channel_evidence = {}
    for cname, e_c in channels.items():
        p_c = position_pre_weight(evidence=e_c, on_task_logprob=on_task_logprob)
        channel_pre[cname] = torch.where(avail.reshape(-1, 1), p_c, torch.ones_like(p_c))
        channel_evidence[cname] = torch.where(
            avail.reshape(-1, 1, 1), e_c, torch.zeros_like(e_c)
        )

    probes = {}
    probe_evidence = {}
    for a in probe_alphas:
        e_a = candidate_kl_evidence(
            common=dec["common_soft"], source_gate=q_sim,
            exclusive=exclusive, source_scale=float(a),
        )
        p_a = position_pre_weight(evidence=e_a, on_task_logprob=on_task_logprob)
        probes[probe_name(a)] = torch.where(avail.reshape(-1, 1), p_a, torch.ones_like(p_a))
        # Kept rather than discarded: without the per-candidate evidence the
        # probe can only be reported as a size, and a size cannot say WHICH
        # positions a larger alpha would have moved budget to -- which is the
        # only thing the series is for. Zeroed where the position is
        # unavailable, matching the pre-weight beside it.
        probe_evidence[probe_name(a)] = torch.where(
            avail.reshape(-1, 1, 1), e_a, torch.zeros_like(e_a)
        )

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
        "common_soft": dec["common_soft"],
        # (bs, resp, k) and (bs, resp, k, n_off): the two source factors, kept
        # so a caller can cut them the same ways the evidence is cut.
        "q_sim": q_sim,
        "source_exclusive": exclusive,
        # (bs, n_off). The advantage reliability, carried for the diagnostics
        # and applied to nothing -- see the note at its assignment.
        "row_alpha": row_alpha,
        # (bs, resp, k) long / (bs, resp, k, 1 + n_off) / (bs, resp, k).
        # Column 0 of ``without`` is the on-task teacher, columns 1.. are the
        # off-task planes in ``off_plane_tasks`` order -- the same layout
        # ``hat_off`` and ``evidence_by_source`` already use.
        "attribution": attribution,
        "residual": dec["residual"],
        "evidence": evidence,
        "state": state,
        "teacher_prob": p_teacher,
        # sum_v p_d(v) |c(v)| -- the corroboration channel's share of W~ - 1.
        "evidence_shared": evidence_shared_sum,
        # The same thing the off-task-only rule would have produced. Reported,
        # never applied: the gap between the two IS the price of requiring the
        # on-task teacher to have spoken.
        "evidence_shared_offtask_only": (p_teacher * dec["common_ev"].abs()).sum(dim=-1),
        "evidence_offtask_only": (p_teacher * evidence_offtask_only).sum(dim=-1),
        # (bs, resp, k, n_off): each source's share of the same quantity.
        "evidence_by_source": evidence_by_source_cand,
        # The SAME quantity with alpha taken out: sum_v p_d(v) |dhat_m(v)|, per
        # candidate. Reported because alpha = 0 collapses evidence_by_source to
        # zero, and "the source teacher had nothing to say here" and "the source
        # teacher spoke loudly and the advantage correlation vetoed it" are then
        # the same reading. They are opposite findings about the mechanism, and
        # every future change to how alpha is estimated -- no veto, partial
        # correlation, a per-source gate -- is an experiment on exactly the
        # difference between them.
        "activity_by_source": p_teacher.unsqueeze(-1) * hat_off.abs(),
        # W - 1, split. The three add to it EXACTLY (test_the_logit_push_
        # decomposition_is_exact), which is what lets the per-source logit push
        # be attributed rather than approximated.
        "push_shared": push_shared,                 # (bs, resp)
        "push_by_source": push_by_source,           # (bs, resp, n_off)
        "push_normalizer": push_normalizer,         # (bs, resp)
        # THE TWO STAGES THE SOURCE PASSES THROUGH, as position mass, so each
        # one's cost is a ratio and not an inference. All three are the same
        # expectation ``evidence`` is, over the same p(v):
        #   source_gross            everything the off-task teachers said
        #   source_exclusive_gross  what survived the on-task ceiling
        #   (evidence - shared)     what survived the similarity gate as well
        "source_gross": (p_teacher.unsqueeze(-1) * hat_off.abs()).sum(dim=(2, 3)),
        "source_exclusive_gross": (p_teacher.unsqueeze(-1) * exclusive).sum(dim=(2, 3)),
        "gate_mass": (p_teacher * q_sim).sum(dim=-1),
        "probe_pre_weight": probes,
        "probe_evidence": probe_evidence,
        "channel_pre_weight": channel_pre,
        "channel_evidence": channel_evidence,
    }


class LogitPushTokens:
    """Which STUDENT tokens the weighting amplified or damped, and by how much.

    NOT the same table as :class:`TokenStateCounts`, and the difference is the
    one most likely to be misread. That table names the tokens whose EVIDENCE
    justified the weight -- the candidates a source spoke at. This one names the
    tokens whose LOGIT the weight then moved, which is every token in the
    support, because W is a scalar on the position and the OPD term already had
    a direction at each of them.

    A token can be in one and not the other. Search's evidence at ``retrieve``
    can raise W at a position whose OPD term is pushing ``the`` down harder than
    anything else, and the sentence "Search reinforced ``retrieve``" would be
    false: what Search reinforced there is the suppression of ``the``. Reporting
    one table under both readings is the error this class exists to prevent.

    The unweighted descent direction on logit u is

        g0(u) = coef * p_student(u) * (D - f(u))

    the applied one is ``W * g0(u)``, and what the mechanism ADDED is
    ``(W - 1) * g0(u)``. Filed by :data:`PUSH_CLASSES`, because sign alone is
    not the reading: a weight above 1 at a token the term was pushing DOWN
    amplifies the suppression, and a table that called that "reinforced" would
    invert the claim.

    Columns per (scope, class, token):

    ``n``          candidate occurrences.
    ``extra``      sum of ``(W - 1) * g0``, signed. The mechanism's own addition.
    ``extra_abs``  the same in absolute value, so a token pushed both ways
                   across contexts is not netted out of its own ranking.
    ``weighted``   sum of ``W * g0``, the push the objective actually applied.
    ``mass``       the student's probability there, so "the arm amplified this
                   token" can be read against whether the student was ever going
                   to say it.
    ``sampled``    how often it was the token actually emitted.
    ``base``       sum of ``g0``, signed -- the push the UNWEIGHTED OPD term
                   would have applied at the same positions.
    ``base_abs``   the same in absolute value.
    ``kl_mass``    sum of ``p_student(u) * f(u)``, the candidate's own share of
                   the position's ``D``. Nats of evidence, not of push. Signed,
                   and genuinely negative at candidates the teacher is more
                   confident about than the student -- only the position's whole
                   ``D`` is bounded below, not each candidate's share of it.
    ``kl_mass_abs``the same in absolute value, for the same reason ``extra_abs``
                   exists: a token that carries the KL one way in one context and
                   the other way in another nets to nothing and is the most
                   interesting row in the table.

    THE LAST THREE COLUMNS CONTAIN NO ``W``. That is their point. They are the
    same quantity whether this table runs inside the weighted arm or inside a
    control that never builds a weight, so "which tokens does the plain OPD term
    push hardest" is one ranking with one definition across both arms -- and,
    inside the weighted arm, a counterfactual taken on the SAME policy, which is
    stronger than the same question asked of a different run.

    They are summed over the class axis when ranked, deliberately: the four
    classes are a fact about ``W``, and splitting an unweighted series by them
    would put a control's whole vocabulary in the two ``_damped`` cells (``W = 1``
    is not ``> 1``) and make the two arms' tables incomparable by construction.

    Memory is ``(1 + n_tasks) * 4 * V * len(TERMS)`` cells. At Qwen3's vocabulary
    and three tasks that is 24.3M cells and about 194 MB, which is why the whole
    dense-token family is stride-gated: see ``token_stats.every``.
    """

    TERMS = ("n", "extra", "extra_abs", "weighted", "mass", "sampled",
             "base", "base_abs", "kl_mass", "kl_mass_abs")

    def __init__(self, *, vocab_size: int, n_tasks: int, device, top_n: int = 32):
        self.vocab_size = V = int(vocab_size)
        self.n_scopes = S = 1 + int(n_tasks)
        self.n_classes = C = len(PUSH_CLASSES)
        self.top_n = int(top_n)
        # float64 for the reason every other table here uses it: millions of
        # atomic adds of ~1e-6 lose their tail in float32, and index_add_ does
        # not promise an order.
        self.buf = torch.zeros(S * C * V * len(self.TERMS), dtype=torch.float64, device=device)
        # HOW MUCH OF THE PUSH THIS TABLE CAN NAME. Every row above is a token in
        # the student's top-k, but the OPD term also acts on the tail bucket,
        # which has no token to be filed under. Without these four sums the token
        # ranking is quoted with an unstated denominator: "the arm amplified
        # these tokens" reads the same whether the named tokens carry 92% of the
        # added push or 35% of it, and only the first supports the sentence.
        # [support_extra_abs, tail_extra_abs, support_weighted_abs,
        #  tail_weighted_abs, support_base_abs, tail_base_abs] per scope. The
        # last two carry no W, so the base ranking below is quoted against its
        # own denominator rather than against the weighted one -- which in the
        # arm is a different number and in a control is the same number by
        # accident, and an accident is not a definition.
        self.tail_buf = torch.zeros(S * 6, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, support_ids, g0, weight, coef_applied_weight, response_mask,
               task_ids=None, sampled_onehot=None, p_student=None, g0_tail=None,
               gap=None) -> None:
        """Fold one micro-batch in.

        Args:
            g0: (bs, resp, k) the UNWEIGHTED descent direction, from
                :func:`opd_logit_push`.
            weight: (bs, resp) the applied W. Pass ones on an arm that builds no
                weight: ``extra`` is then identically zero and the three
                weight-free columns carry the whole table.
            coef_applied_weight: (bs, resp) the same W -- passed separately only
                so a caller cannot silently hand the pre-normalisation weight to
                one argument and the applied one to the other.
            sampled_onehot: (bs, resp, k) one at the emitted token, or None.
            p_student: (bs, resp, k), or None to take it from nothing -- the mass
                column then stays zero rather than guessing.
            g0_tail: (bs, resp) the unweighted push on the TAIL bucket, from
                :func:`opd_logit_push`. None leaves the coverage shares
                unreported rather than implying the tail was empty.
            gap: (bs, resp, k) ``f = log p_student - log p_on``, from
                :func:`opd_logit_push`. Needed for ``kl_mass`` and for nothing
                else; None leaves that column at zero rather than deriving it
                from ``g0``, which would divide by a probability.
        """
        assert weight is coef_applied_weight or torch.equal(weight, coef_applied_weight), (
            "LogitPushTokens takes the APPLIED weight for both the class and the "
            "magnitude; two different weights would file a token under one and "
            "measure it under the other"
        )
        self._cpu_cache = None
        V, C, K = self.vocab_size, self.n_classes, len(self.TERMS)
        m = response_mask.unsqueeze(-1).to(torch.float64)
        g = g0.detach().to(torch.float64)
        w = weight.detach().to(torch.float64).unsqueeze(-1)
        extra = (w - 1.0) * g
        cls = push_direction_class(g0, weight)
        tok = support_ids.clamp(min=0, max=V - 1).to(torch.long)

        ps = torch.zeros_like(g) if p_student is None else p_student.detach().to(torch.float64)
        klm = (torch.zeros_like(g) if gap is None
               else ps * gap.detach().to(torch.float64)) * m
        cols = [
            m.expand_as(g),
            extra * m,
            extra.abs() * m,
            w * g * m,
            ps * m,
            (torch.zeros_like(g) if sampled_onehot is None
             else sampled_onehot.detach().to(torch.float64)) * m,
            # No W below this line. See the class docstring.
            g * m,
            g.abs() * m,
            klm,
            klm.abs(),
        ]
        vals = torch.stack(cols, dim=-1).reshape(-1, K)
        base = (cls * V + tok).reshape(-1) * K
        offs = torch.arange(K, device=vals.device)
        # Pooled scope first, then the row's task -- the same two-pass rule the
        # other dense tables use, so an untagged row still reaches the pooled
        # table instead of inventing a task.
        self.buf.index_add_(0, (base.unsqueeze(-1) + offs).reshape(-1), vals.reshape(-1))
        if task_ids is None:
            return
        t = task_ids.reshape(-1, 1, 1).expand_as(cls).reshape(-1).to(torch.long)
        known = ((t >= 0) & (t < self.n_scopes - 1)).to(torch.float64).unsqueeze(-1)
        scoped = (t.clamp(min=0, max=self.n_scopes - 2) + 1) * (C * V) * K + base
        self.buf.index_add_(0, (scoped.unsqueeze(-1) + offs).reshape(-1), (vals * known).reshape(-1))
        self._fold_tail(
            g0=g0, g0_tail=g0_tail, weight=weight, response_mask=response_mask,
            task_ids=task_ids,
        )

    def _fold_tail(self, *, g0, g0_tail, weight, response_mask, task_ids) -> None:
        """The support/tail coverage sums, per scope. No-op without ``g0_tail``."""
        if g0_tail is None:
            return
        mp = response_mask.to(torch.float64)
        w = weight.detach().to(torch.float64)
        gs = g0.detach().to(torch.float64).abs().sum(dim=-1)
        gt = g0_tail.detach().to(torch.float64).abs()
        wm1 = (w - 1.0).abs()
        cols = torch.stack(
            [wm1 * gs * mp, wm1 * gt * mp, w.abs() * gs * mp, w.abs() * gt * mp,
             gs * mp, gt * mp], dim=-1
        )                                                     # (bs, resp, 6)
        self.tail_buf[:6] += cols.sum(dim=(0, 1))
        if task_ids is None:
            return
        t = task_ids.reshape(-1).to(torch.long)
        known = ((t >= 0) & (t < self.n_scopes - 1)).to(torch.float64).reshape(-1, 1)
        per_row = cols.sum(dim=1) * known                     # (bs, 6)
        base = (t.clamp(min=0, max=max(self.n_scopes - 2, 0)) + 1) * 6
        flat = (base.unsqueeze(-1) + torch.arange(6, device=per_row.device)).reshape(-1)
        self.tail_buf.index_add_(0, flat, per_row.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.tail_buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu").view(
                self.n_scopes, self.n_classes, self.vocab_size, len(self.TERMS)
            )
        return self._cpu_cache

    def _scope_name(self, scope, task_names):
        if scope == 0:
            return None
        t = scope - 1
        return task_names[t] if task_names and t < len(task_names) else f"task{t}"

    def scalar_metrics(self, task_names=None, prefix: str = "kl_weight",
                       weight_free: bool = True) -> dict:
        """How the added push splits over the four directions, and how spread.

        ``share`` is of the GROSS added push, so the four sum to 1 and a class
        that cancels itself across contexts is not reported as inactive.

        ``weight_free`` publishes the columns that contain no ``W``. It is a
        rendering choice, not a measurement one -- the buffer holds them either
        way -- and it exists because a weighted arm runs a SECOND instance of
        this class at ``W = 1`` to own exactly those columns. Two instances
        publishing them would be two identical series under one key, which is
        the same as one series that a reader has to double-count.
        """
        buf = self._cpu()
        tail = self.tail_buf.detach().to("cpu").view(self.n_scopes, 6)
        out = {}
        idx = {t: i for i, t in enumerate(self.TERMS)}
        N = min(self.top_n, self.vocab_size)
        for scope in range(self.n_scopes):
            name = self._scope_name(scope, task_names)
            head = prefix if name is None else f"{prefix}/{name}"
            gross_all = float(buf[scope, :, :, idx["extra_abs"]].sum())
            # How much of the added push the rows above can NAME. The support
            # and the tail are the whole of it, so the two extra shares sum to 1
            # and a token ranking can be quoted with its denominator attached.
            sup_x, tail_x, sup_w, tail_w, sup_b, tail_b = (float(v) for v in tail[scope])
            if sup_x + tail_x > 0:
                out[f"{head}/push/support_extra_abs_share"] = sup_x / (sup_x + tail_x)
                out[f"{head}/push/tail_extra_abs_share"] = tail_x / (sup_x + tail_x)
            if sup_w + tail_w > 0:
                out[f"{head}/push/tail_weighted_abs_share"] = tail_w / (sup_w + tail_w)
            # THE WEIGHT-FREE HALF, emitted before the early return below rather
            # than after it. That return is taken on the added push being zero,
            # which is exactly the case an arm with no weight is in -- putting
            # these keys after it would mean the control, the one run they exist
            # for, is the one run that never publishes them.
            if weight_free and sup_b + tail_b > 0:
                out[f"{head}/push/tail_base_abs_share"] = tail_b / (sup_b + tail_b)
            base_cell = buf[scope].sum(dim=0)         # summed over the four W classes
            base_abs = base_cell[:, idx["base_abs"]]
            base_gross = float(base_abs.sum()) if weight_free else 0.0
            if base_gross > 0:
                out[f"{head}/push/base_abs_total"] = base_gross
                out[f"{head}/push/base_net_total"] = float(base_cell[:, idx["base"]].sum())
                out[f"{head}/push/base_n_distinct"] = float((base_cell[:, idx["n"]] > 0).sum())
                out[f"{head}/push/base_top{N}_share"] = (
                    float(torch.topk(base_abs, N).values.sum()) / base_gross
                )
            kl_gross = float(base_cell[:, idx["kl_mass_abs"]].sum()) if weight_free else 0.0
            if kl_gross > 0:
                # Nats of the position's own D, filed by candidate. Its top-N
                # share answers the question the push ranking cannot: whether
                # the distillation signal itself is concentrated, independently
                # of what any weight did with it. Concentration is quoted on the
                # GROSS, because the signed sum has cancellation in it and a
                # share of a cancelled total is not a share of anything.
                out[f"{head}/push/kl_mass_total"] = float(base_cell[:, idx["kl_mass"]].sum())
                out[f"{head}/push/kl_mass_abs_total"] = kl_gross
                out[f"{head}/push/kl_mass_abs_top{N}_share"] = (
                    float(torch.topk(base_cell[:, idx["kl_mass_abs"]], N).values.sum()) / kl_gross
                )
            if gross_all <= 0:
                continue
            out[f"{head}/push/extra_abs_total"] = gross_all
            out[f"{head}/push/extra_net_total"] = float(buf[scope, :, :, idx["extra"]].sum())
            for c, cname in enumerate(PUSH_CLASSES):
                cell = buf[scope, c]
                gross = float(cell[:, idx["extra_abs"]].sum())
                if gross <= 0:
                    continue
                h = f"{head}/push/{cname}"
                out[f"{h}/gross_share"] = gross / gross_all
                out[f"{h}/net"] = float(cell[:, idx["extra"]].sum())
                out[f"{h}/n_distinct"] = float((cell[:, idx["n"]] > 0).sum())
                per_tok = cell[:, idx["extra_abs"]]
                out[f"{h}/top{N}_share"] = float(torch.topk(per_tok, N).values.sum()) / gross
                # Was the arm amplifying tokens the student was going to say, or
                # ones it was not? The two are different mechanisms with the same
                # nats.
                n_cell = float(cell[:, idx["n"]].sum())
                if n_cell > 0:
                    out[f"{h}/mean_p_student"] = float(cell[:, idx["mass"]].sum()) / n_cell
                    out[f"{h}/sampled_frac"] = float(cell[:, idx["sampled"]].sum()) / n_cell
        return out

    def top_tokens(self, task_names=None, weight_free: bool = True) -> list:
        """Ranked rows in the shared dump schema, one ranking per class.

        Plus two rankings that carry no ``W``: ``base_logit_push`` (the plain
        OPD term's push) and ``kl_mass`` (the candidate's own share of ``D``).
        Both are taken on the class-SUMMED cell, so they mean the same thing in
        an arm that builds a weight and in one that does not -- which is what
        makes "these tokens rank top without weighting" a comparison rather than
        two tables with the same column names. ``weight_free=False`` withholds
        them; see :meth:`scalar_metrics` for when that is the right call.

        The weighted rows carry ``base_logit_push`` as a column regardless, so
        "the arm amplified this token" is always readable against how hard the
        unweighted term was pushing it in the first place.
        """
        buf = self._cpu()
        idx = {t: i for i, t in enumerate(self.TERMS)}
        N = min(self.top_n, self.vocab_size)
        rows = []
        for scope in range(self.n_scopes):
            name = self._scope_name(scope, task_names) or "__pooled__"
            if weight_free:
                rows += self._base_rows(buf[scope].sum(dim=0), idx, N, name)
            for c, cname in enumerate(PUSH_CLASSES):
                cell = buf[scope, c]
                series = cell[:, idx["extra_abs"]]
                if float(series.sum()) <= 0:
                    continue
                vals, ids = torch.topk(series, N)
                for rank, (v, tok) in enumerate(zip(vals.tolist(), ids.tolist())):
                    if v <= 0:
                        break
                    n_tok = float(cell[tok, idx["n"]])
                    rows.append({
                        "scope": name, "ranked_by": "extra_logit_push",
                        "direction_class": cname, "rank": rank, "token_id": int(tok),
                        "count": int(n_tok),
                        "extra_logit_push": float(cell[tok, idx["extra"]]),
                        "extra_logit_push_abs": v,
                        "weighted_logit_push": float(cell[tok, idx["weighted"]]),
                        # In THIS class cell, so it is the unweighted push at the
                        # subset of occurrences that landed in this direction
                        # class -- not the token's whole base push, which the
                        # base ranking reports. Carried so a row saying "the arm
                        # amplified this token" is readable against how hard the
                        # plain term was already pushing it.
                        "base_logit_push": float(cell[tok, idx["base"]]),
                        "kl_mass": float(cell[tok, idx["kl_mass"]]),
                        "p_student_mean": (float(cell[tok, idx["mass"]]) / n_tok) if n_tok else 0.0,
                        "sampled_count": int(cell[tok, idx["sampled"]]),
                    })
        return rows

    def _base_rows(self, cell, idx, N, scope_name) -> list:
        """The two weight-free rankings for one scope, class axis already summed.

        ``direction`` is the sign of the token's NET unweighted push, not a
        :data:`PUSH_CLASSES` label: those four are a fact about ``W``, and at
        ``W = 1`` every token in the vocabulary falls in the two ``_damped``
        cells, which would read as a finding and is an artefact of ``1 > 1``
        being false.
        """
        rows = []
        for ranked_by, series in (
            ("base_logit_push", cell[:, idx["base_abs"]]),
            ("kl_mass", cell[:, idx["kl_mass_abs"]]),
        ):
            if float(series.sum()) <= 0:
                continue
            vals, ids = torch.topk(series, N)
            for rank, (v, tok) in enumerate(zip(vals.tolist(), ids.tolist())):
                if v <= 0:
                    break
                n_tok = float(cell[tok, idx["n"]])
                net = float(cell[tok, idx["base"]])
                rows.append({
                    "scope": scope_name, "ranked_by": ranked_by,
                    "direction_class": "base_up" if net > 0 else "base_down",
                    "rank": rank, "token_id": int(tok), "count": int(n_tok),
                    "base_logit_push": net,
                    "base_logit_push_abs": float(cell[tok, idx["base_abs"]]),
                    "kl_mass": float(cell[tok, idx["kl_mass"]]),
                    "kl_mass_abs": float(cell[tok, idx["kl_mass_abs"]]),
                    # Carried on these rows too so a reader comparing the two
                    # arms never has to join back to the weighted ranking, whose
                    # top-N is a different set of tokens.
                    "extra_logit_push": float(cell[tok, idx["extra"]]),
                    "weighted_logit_push": float(cell[tok, idx["weighted"]]),
                    "p_student_mean": (float(cell[tok, idx["mass"]]) / n_tok) if n_tok else 0.0,
                    "sampled_count": int(cell[tok, idx["sampled"]]),
                })
        return rows


# Which trajectories the arm spent its budget on. The buckets are outcome
# facts about the ROW, not about the position, so a row lands in "all" and in
# exactly one of each opposed pair -- the pairs are reported as a ratio and the
# "all" bucket is the denominator that makes each pair's shares add up.
OUTCOME_BUCKETS = ("all", "adv_positive", "adv_negative", "reward_positive", "reward_nonpositive")

# Per-row moments. g is the row's gross effect fraction, a its advantage: the
# correlation between them is the one number that says whether the mechanism
# tracks the reward signal at the TRAJECTORY level, which no per-position
# reading can answer.
OUTCOME_TERMS = ("n", "gross", "net", "kl", "g", "gg", "a", "aa", "ag")


class OutcomeEffectStats:
    """Did the weighting spend its KL budget on the rollouts that worked?

    Every other reading here is per position or per token. This one is per
    TRAJECTORY, because "the arm moved 3% of the budget" and "the arm moved 3%
    of the budget, almost all of it on rollouts that failed" are the same
    number and opposite findings -- and the second is the one a reviewer asks
    about.

    For row i,

        G_i = sum_t |W - 1| D  /  sum_t D

    the fraction of its own OPD budget the arm redistributed. Reported per task
    and per outcome bucket, plus ``Corr(A, G)`` over rows, which is the
    trajectory-level companion to the position-level ``weight_kl_corr``.

    NOT a causal claim, and the metric names avoid implying one: the advantage
    is observed, the weight is built from frozen teachers, and nothing here
    randomises anything. What it establishes is which side of the reward signal
    the budget went to, which is a fact about the run rather than an inference
    from it.

    THE REWARD BUCKETS ARE NAMED FOR THE SCORE, NOT FOR SUCCESS, and the
    distinction is not pedantry here. The split is ``sum(token_level_scores) >
    0``, and that tensor is where ``apply_invalid_action_penalty`` writes: a
    solved episode that emitted enough malformed actions sums to zero or below
    and lands in ``reward_nonpositive``. On search the score is graded as well,
    so "scored at all" and "solved the task" come apart there for a second,
    independent reason. Calling the bucket ``success`` -- as this class did --
    puts a claim in the write-up that the number does not support, and no
    per-row success flag exists in the batch to support it with.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.n_buckets = B = len(OUTCOME_BUCKETS)
        self.n_terms = K = len(OUTCOME_TERMS)
        self.buf = torch.zeros((1 + T) * B * K, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, weight, teacher_kl, response_mask, advantage,
               task_ids=None, reward=None) -> None:
        """Fold one micro-batch of ROWS in.

        Args:
            weight: (bs, resp) the applied W.
            teacher_kl: (bs, resp) the UNWEIGHTED per-token KL.
            advantage: (bs,) the row's GRPO advantage.
            reward: (bs,) the row's episode score INCLUDING any invalid-action
                penalty, or None -- the two reward buckets then stay empty
                rather than being filled from the advantage, which is
                group-relative and says nothing about what the episode scored.
        """
        self._cpu_cache = None
        B, K = self.n_buckets, self.n_terms
        m = response_mask.to(torch.float64)
        d = teacher_kl.detach().to(torch.float64) * m
        gross_row = ((weight.detach().to(torch.float64) - 1.0).abs() * d).sum(dim=1)
        net_row = ((weight.detach().to(torch.float64) - 1.0) * d).sum(dim=1)
        kl_row = d.sum(dim=1)
        live = kl_row > 0
        g = torch.where(live, gross_row / kl_row.clamp(min=1e-12), torch.zeros_like(kl_row))
        a = advantage.detach().reshape(-1).to(torch.float64)
        ok = live.to(torch.float64)

        vals = torch.stack(
            [ok, gross_row * ok, net_row * ok, kl_row * ok,
             g * ok, g * g * ok, a * ok, a * a * ok, a * g * ok],
            dim=-1,
        )                                                        # (bs, K)
        sel = [
            torch.ones_like(ok),
            (a > 0).to(torch.float64),
            (a < 0).to(torch.float64),
        ]
        if reward is None:
            sel += [torch.zeros_like(ok), torch.zeros_like(ok)]
        else:
            r = reward.detach().reshape(-1).to(torch.float64)
            sel += [(r > 0).to(torch.float64), (r <= 0).to(torch.float64)]

        for b, pick in enumerate(sel):
            row = vals * pick.unsqueeze(-1)
            self.buf[b * K : (b + 1) * K] += row.sum(dim=0)
            if task_ids is None:
                continue
            t = task_ids.reshape(-1).to(torch.long)
            known = ((t >= 0) & (t < self.n_tasks)).to(torch.float64).unsqueeze(-1)
            base = (t.clamp(min=0, max=max(self.n_tasks - 1, 0)) + 1) * B * K + b * K
            flat = (base.unsqueeze(-1) + torch.arange(K, device=row.device)).reshape(-1)
            self.buf.index_add_(0, flat, (row * known).reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu").view(
                1 + self.n_tasks, self.n_buckets, self.n_terms
            )
        return self._cpu_cache

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        buf = self._cpu()
        i = {t: j for j, t in enumerate(OUTCOME_TERMS)}
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for scope in range(1 + self.n_tasks):
            head = prefix if scope == 0 else f"{prefix}/{name(scope - 1)}"
            cells = {}
            for b, bucket in enumerate(OUTCOME_BUCKETS):
                cell = buf[scope, b]
                n = float(cell[i["n"]])
                if n <= 0:
                    continue
                kl = float(cell[i["kl"]])
                cells[bucket] = gross = (float(cell[i["gross"]]) / kl) if kl > 0 else 0.0
                out[f"{head}/outcome/{bucket}/gross_effect"] = gross
                out[f"{head}/outcome/{bucket}/net_effect"] = (
                    (float(cell[i["net"]]) / kl) if kl > 0 else 0.0
                )
                out[f"{head}/outcome/{bucket}/n_rows"] = n
            # The two contrasts. A ratio rather than a difference so it reads the
            # same at any teacher_kl_loss_coef, and only when both sides exist --
            # a step with no failures is not a step with an infinite ratio.
            for hi, lo, label in (
                ("adv_positive", "adv_negative", "adv_positive_to_negative"),
                ("reward_positive", "reward_nonpositive", "reward_positive_to_nonpositive"),
            ):
                if hi in cells and lo in cells and cells[lo] > 1e-12:
                    out[f"{head}/outcome/{label}_effect_ratio"] = cells[hi] / cells[lo]
            # Corr(A, G) over rows, from the pooled bucket's moments.
            cell = buf[scope, 0]
            n = float(cell[i["n"]])
            if n >= 3:
                va = float(cell[i["aa"]]) / n - (float(cell[i["a"]]) / n) ** 2
                vg = float(cell[i["gg"]]) / n - (float(cell[i["g"]]) / n) ** 2
                if va > 0 and vg > 0:
                    cov = float(cell[i["ag"]]) / n - (float(cell[i["a"]]) / n) * (
                        float(cell[i["g"]]) / n
                    )
                    out[f"{head}/outcome/corr_adv_gross_effect"] = cov / math.sqrt(va * vg)
        return out


# Per row AND per source. OutcomeEffectStats collapses the source axis and the
# evidence table collapses the outcome axis, so between them "Search moved 4% of
# AlfWorld's budget" and "the arm spent its budget on the rollouts that scored"
# are two facts that cannot be joined. This is the join.
SOURCE_OUTCOME_TERMS = ("n", "effect", "kl", "g", "gg", "a", "aa", "ag")


class SourceOutcomeStats:
    """Which trajectories each SOURCE's own contribution went to.

    For row i and source m,

        G_i^(m) = sum_t push_by_source[i, t, m] * D[i, t]  /  sum_t D[i, t]

    the fraction of the row's OPD budget that this one source's channel moved.
    ``push_by_source`` is ``B_m/mu`` from :func:`build_position_weight`, so the
    sources and the corroboration channel and the normaliser offset add up to
    the row's whole ``(W - 1) D`` -- these numbers are shares of a partition,
    not correlated summaries of one.

    Non-negative by construction: alpha, the standardized magnitude and the
    teacher's probability are all non-negative, so a source can only ever ADD
    weight. Which is why the outcome cut matters -- the question is never
    whether a source pushed, it is which rollouts it pushed on.

    ``(T, T, 5, 8)`` float64: 360 cells at three tasks, one ``index_add_`` per
    source per micro-batch over rows rather than positions.
    """

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = T = int(n_tasks)
        self.n_buckets = B = len(OUTCOME_BUCKETS)
        self.n_terms = K = len(SOURCE_OUTCOME_TERMS)
        self.buf = torch.zeros(T * T * B * K, dtype=torch.float64, device=device)
        self._cpu_cache = None

    def update(self, *, push_by_source, teacher_kl, response_mask, advantage,
               task_ids, off_plane_tasks, reward=None) -> None:
        """Fold one micro-batch of ROWS in.

        Args:
            push_by_source: (bs, resp, n_off) each source's share of ``W - 1``.
            teacher_kl: (bs, resp) the UNWEIGHTED per-token KL.
        """
        self._cpu_cache = None
        T, B, K = self.n_tasks, self.n_buckets, self.n_terms
        m = response_mask.to(torch.float64)
        d = teacher_kl.detach().to(torch.float64) * m
        kl_row = d.sum(dim=1)
        live = kl_row > 0
        ok_row = live.to(torch.float64)
        a = advantage.detach().reshape(-1).to(torch.float64)
        eff = (push_by_source.detach().to(torch.float64) * d.unsqueeze(-1)).sum(dim=1)  # (bs, n_off)

        sel = [torch.ones_like(ok_row), (a > 0).to(torch.float64), (a < 0).to(torch.float64)]
        if reward is None:
            sel += [torch.zeros_like(ok_row), torch.zeros_like(ok_row)]
        else:
            r = reward.detach().reshape(-1).to(torch.float64)
            sel += [(r > 0).to(torch.float64), (r <= 0).to(torch.float64)]

        dst = task_ids.reshape(-1).to(torch.long)
        for c in range(eff.size(-1)):
            src = off_plane_tasks[:, c].reshape(-1).to(torch.long)
            ok = ((dst >= 0) & (src >= 0)).to(torch.float64) * ok_row
            e = eff[:, c]
            g = torch.where(live, e / kl_row.clamp(min=1e-12), torch.zeros_like(kl_row))
            vals = torch.stack(
                [ok, e * ok, kl_row * ok, g * ok, g * g * ok, a * ok, a * a * ok, a * g * ok],
                dim=-1,
            )                                                    # (bs, K)
            pair = (dst.clamp(min=0) * T + src.clamp(min=0)) * B * K
            for b, pick in enumerate(sel):
                row = vals * pick.unsqueeze(-1)
                flat = ((pair + b * K).unsqueeze(-1)
                        + torch.arange(K, device=row.device)).reshape(-1)
                self.buf.index_add_(0, flat, row.reshape(-1))

    def all_reduce(self) -> None:
        self._cpu_cache = None
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.buf, op=torch.distributed.ReduceOp.SUM)

    def _cpu(self):
        if self._cpu_cache is None:
            self._cpu_cache = self.buf.detach().to("cpu").view(
                self.n_tasks, self.n_tasks, self.n_buckets, self.n_terms
            )
        return self._cpu_cache

    def metrics(self, *, task_names=None, prefix: str = "kl_weight") -> dict:
        buf = self._cpu()
        i = {t: j for j, t in enumerate(SOURCE_OUTCOME_TERMS)}
        name = lambda t: task_names[t] if task_names and t < len(task_names) else f"task{t}"
        out = {}
        for dst in range(self.n_tasks):
            for src in range(self.n_tasks):
                if src == dst:
                    continue
                cells = buf[dst, src]
                if float(cells[0, i["n"]]) <= 0:
                    continue
                head = f"{prefix}/source_outcome/{name(src)}__on__{name(dst)}"
                shares = {}
                for b, bucket in enumerate(OUTCOME_BUCKETS):
                    cell = cells[b]
                    n = float(cell[i["n"]])
                    if n <= 0:
                        continue
                    kl = float(cell[i["kl"]])
                    shares[bucket] = e = (float(cell[i["effect"]]) / kl) if kl > 0 else 0.0
                    out[f"{head}/{bucket}/effect"] = e
                    out[f"{head}/{bucket}/n_rows"] = n
                # Ratios, and only where both sides exist -- a step with no
                # negative-advantage rows is not a step with an infinite ratio.
                for hi, lo, label in (
                    ("adv_positive", "adv_negative", "adv_positive_to_negative"),
                    ("reward_positive", "reward_nonpositive", "reward_positive_to_nonpositive"),
                ):
                    if hi in shares and lo in shares and shares[lo] > 1e-12:
                        out[f"{head}/{label}_effect_ratio"] = shares[hi] / shares[lo]
                cell = cells[0]
                n = float(cell[i["n"]])
                if n >= 3:
                    va = float(cell[i["aa"]]) / n - (float(cell[i["a"]]) / n) ** 2
                    vg = float(cell[i["gg"]]) / n - (float(cell[i["g"]]) / n) ** 2
                    if va > 0 and vg > 0:
                        cov = float(cell[i["ag"]]) / n - (float(cell[i["a"]]) / n) * (
                            float(cell[i["g"]]) / n
                        )
                        out[f"{head}/corr_adv_source_effect"] = cov / math.sqrt(va * vg)
        return out


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


def _optional_column(built: dict, name: str, like: torch.Tensor) -> torch.Tensor:
    """A (bs, resp) column the shipped arm has and a counterfactual scope does not."""
    if name not in built:
        return torch.zeros_like(like)
    return built[name].detach().to(torch.float32)


def position_terms(built: dict, teacher_kl: torch.Tensor) -> dict:
    """The (bs, resp) columns :data:`POSITION_TERMS` names."""
    w = built["weight"].detach().to(torch.float32)
    pre = built["pre_weight"].detach().to(torch.float32)
    kl = teacher_kl.detach().to(torch.float32)
    # S and R exist only for the SHIPPED arm. The probe and channel scopes pass
    # a hand-built mapping -- a counterfactual pre-weight, the live state labels
    # and evidence -- and have no partition of their own: at another alpha the
    # split is a different one, and the no_shared channel has no shared term at
    # all. They are handed zeros rather than a partition borrowed from the live
    # arm, and position_weight_metrics renders no channel reading from a scope
    # whose columns are all zero.
    if "push_shared" in built:
        s_push = built["push_shared"].detach().to(torch.float32)
        r_push = built["push_by_source"].detach().to(torch.float32).sum(dim=-1)
    else:
        s_push = torch.zeros_like(w)
        r_push = torch.zeros_like(w)
    return {
        "w": w,
        "w_sq": w * w,
        "w_pre": pre,
        "w_pre_sq": pre * pre,
        "kl": kl,
        "kl_sq": kl * kl,
        "w_kl": w * kl,
        "kl_shift_abs": (w - 1.0).abs() * kl,
        "evidence": pre - 1.0,
        "evidence_shared": built["evidence_shared"],
        "evidence_shared_offtask_only": built["evidence_shared_offtask_only"],
        # S and R: the corroboration channel's and the source channels' EXACT
        # shares of W - 1, from the partition build_position_weight already
        # makes. Both are non-negative by construction -- alpha, |c|, |dhat| and
        # p are all non-negative -- so no position is ever pushed in opposite
        # directions by the two, and the "opposition" worth measuring is which
        # positions each one picks, not which way it pushes.
        "push_shared": s_push,
        "push_source": r_push,
        "push_shared_sq": s_push * s_push,
        "push_source_sq": r_push * r_push,
        "push_cross": s_push * r_push,
        # Same guard as the partition above, and for the same reason: a
        # counterfactual scope is handed a pre-weight and an evidence tensor,
        # never the gate masses, and a scope that published 0.0 for them would
        # read as "both gates closed here" rather than "not measured here".
        "source_gross": _optional_column(built, "source_gross", w),
        "source_exclusive_gross": _optional_column(built, "source_exclusive_gross", w),
        "gate_mass": _optional_column(built, "gate_mass", w),
        "available": built["available"].reshape(-1, 1).expand_as(w).to(torch.float32),
    }


def per_candidate_shift(built: dict, teacher_kl: torch.Tensor) -> torch.Tensor:
    """(bs, resp, k) each candidate's share of the nats the weighting moved.

    ``(W - 1) D`` splits into a part the candidates caused and the normaliser's
    own offset:

        (W - 1) D  =  [ sum_v p_d(v) e(v) / mu ] D  +  (1/mu - 1) D

    and this is the summand of the first term. It is the ONE definition the
    per-state table, the per-token table and the source attribution all read, so
    three tables that are meant to decompose the same quantity cannot come to
    disagree about what it is.
    """
    inv_mu = 1.0 / built["mu"].clamp(min=1e-12)
    kl = teacher_kl.detach().to(torch.float32)
    return built["teacher_prob"] * built["evidence"] * inv_mu.unsqueeze(-1) * kl.unsqueeze(-1)


# The three per-position scalars the gradient-interference metrics are built
# from. Analytic, in LOGIT space, over the top-k plus the tail bucket: the two
# gradients are known in closed form there, so no second backward is needed and
# the metric cannot perturb the one the optimizer takes.
GRAD_TERMS = (
    "g_opd_sq", "g_grpo_sq", "g_dot",
    # The same three, per CHANNEL. The applied W mixes the corroboration
    # channel, the source channels and the base OPD direction, so the pooled
    # cosine above cannot say which of the two the reward signal disagrees with
    # -- which is the question an arm choice rests on.
    "g_shared_sq", "g_shared_dot", "g_source_sq", "g_source_dot", "g_cross_dot",
)


def opd_logit_push(
    *,
    student_logprob: torch.Tensor,
    teacher_logprob: torch.Tensor,
    teacher_kl: torch.Tensor,
    coef: float,
    eps: float = 1e-8,
) -> dict:
    """The UNWEIGHTED OPD descent direction on each logit, and the pieces of it.

        g0(u) = coef * p_student(u) * (D - f(u)),   f = log p_student - log p_on

    One definition, read by two things that must not disagree: the norm-and-
    cosine metric, which weights it and squares it, and the per-token push table,
    which classifies its sign against W. A second copy is how "the arm amplified
    this token" and "the arm's gradient norm" come to describe different
    quantities under one name.

    Sign convention is DESCENT: positive means the objective is pushing this
    logit up. Checked against autograd in the tests.
    """
    lp_s = student_logprob.detach().to(torch.float32)
    lp_t = teacher_logprob.detach().to(torch.float32)
    p_s = lp_s.exp()
    tail_s = (1.0 - p_s.sum(dim=-1)).clamp(min=eps, max=1.0)
    f = lp_s - lp_t
    f_tail = tail_s.log() - tail_logprob(lp_t, eps)
    d = teacher_kl.detach().to(torch.float32).unsqueeze(-1)
    return {
        "g0": coef * p_s * (d - f),
        "g0_tail": coef * tail_s * (d.squeeze(-1) - f_tail),
        "p_student": p_s,
        "tail_student": tail_s,
        "gap": f,
    }


# What the weighting did to a logit the OPD term was already pushing. The four
# cells are the product of two binary facts -- which way the unweighted term was
# pushing, and whether the arm scaled that push up or down -- and they are the
# only honest way to say "the mechanism reinforced this token", because a weight
# above 1 on a position whose OPD term was pushing a token DOWN reinforces the
# suppression, not the token.
PUSH_CLASSES = (
    "push_down_damped",       # g0 < 0, W < 1
    "push_down_amplified",    # g0 < 0, W > 1
    "push_up_damped",         # g0 > 0, W < 1
    "push_up_amplified",      # g0 > 0, W > 1
)


def push_direction_class(g0: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(bs, resp, k) index into :data:`PUSH_CLASSES`."""
    up = (g0 > 0).to(torch.long)
    amp = (weight.unsqueeze(-1) > 1.0).to(torch.long).expand_as(up)
    return up * 2 + amp


def logit_gradient_terms(
    *,
    student_logprob: torch.Tensor,
    teacher_logprob: torch.Tensor,
    weight: torch.Tensor,
    teacher_kl: torch.Tensor,
    pg_grad_coef: torch.Tensor,
    sampled_onehot: torch.Tensor,
    coef: float,
    pg_coef: float = 1.0,
    row_weight: Optional[torch.Tensor] = None,
    push_shared: Optional[torch.Tensor] = None,
    push_source: Optional[torch.Tensor] = None,
    push: Optional[dict] = None,
    eps: float = 1e-8,
) -> dict:
    """How the weighted OPD term and the policy gradient push the same logits.

    For the reverse KL the descent direction on logit ``v`` is

        g_opd(v) = coef * W * p_student(v) * (D - f(v)),   f = log p_s - log p_t

    and for the policy term, whose per-token loss derivative
    ``dL/d log pi(y)`` the caller supplies,

        g_grpo(v) = -pg_coef * (dL/d log pi(y)) * (1[v = y] - p_student(v)).

    THE COEFFICIENT IS NOT ``A``. It is ``-A*r`` outside the clip and exactly
    zero inside a bound clip branch, and ``r`` is 1 only for the first
    mini-batch of the first PPO epoch -- every ``ppo_mini_batch_size`` rows take
    an optimizer step. :func:`core_algos.policy_loss_gradient_coef` is that
    derivative in closed form, differentiated from the loss the run actually
    minimises and pinned against autograd; passing ``advantages`` here instead
    would report the gradient of a different objective, at full magnitude on
    positions the real one has stopped pushing at all.

    Both are exact on the support the KL already runs over, so the norms and the
    cosine come out of tensors that are already resident. A second backward
    would cost a full pass and, worse, would make a diagnostic capable of
    changing the run it describes.

    Args:
        pg_grad_coef: (bs, resp) ``d(pg_losses)/d(log_prob)``.
        sampled_onehot: (bs, resp, k) one at the emitted token's slot in the
            support, zero elsewhere -- and all-zero when the emitted token is
            outside the top-k, in which case its spike is folded into the tail
            bucket. That is an approximation, and it is why
            ``frac_sampled_outside_topk`` is reported next to these.
        coef: ``teacher_kl_loss_coef``.
        pg_coef: ``pg_loss_coef``. Both terms carry their coefficient because
            the ratio is between what the two contribute to ONE objective, and
            at 0.01 against 1.0 the coefficients are most of the answer.
        row_weight: (bs,) the per-task loss weight, when the run normalises the
            loss per task. It multiplies BOTH terms -- the loss applies it to
            both -- so it cannot change a per-row ratio or cosine, but it does
            change how much each row contributes to the POOLED ones, which is
            what these are. Absent -> every row counts once, which is what an
            unweighted run does.
        push_shared, push_source: (bs, resp) the corroboration channel's and the
            source channels' EXACT shares of ``W - 1``. The mechanism's own
            addition to the logit push is ``(W - 1) g0``, and that partition
            splits it, so ``S g0`` and ``R g0`` are each channel's added
            gradient with no counterfactual normaliser and no second pass. Both
            absent leaves the channel columns at zero and no cosine is rendered.

    ``S g0`` and ``R g0`` scale the SAME direction ``g0``, so their mutual
    cosine is non-negative for any S, R >= 0 -- the two channels cannot pull
    against each other at a logit, only allocate to different positions. It is
    reported because its MAGNITUDE says how redundant they are, and named here
    so a positive value is not read as evidence that they agree. Their cosines
    with the policy gradient carry no such constraint: each can take either
    sign, and their disagreeing is the finding that separates the arms.

    ``push`` is an already-built :func:`opd_logit_push` on the SAME arguments,
    for a caller that needs it for something else too. It is an optimisation
    only -- the values are identical -- and it exists so the one definition of
    ``g0`` is computed once per micro-batch rather than once per reader.
    """
    if push is None:
        push = opd_logit_push(
            student_logprob=student_logprob, teacher_logprob=teacher_logprob,
            teacher_kl=teacher_kl, coef=coef, eps=eps,
        )
    p_s, tail_s = push["p_student"], push["tail_student"]
    w = weight.detach().to(torch.float32).unsqueeze(-1)
    g_opd = w * push["g0"]
    g_opd_tail = w.squeeze(-1) * push["g0_tail"]

    # Descent, to match g_opd's sign: the caller's coefficient is the LOSS
    # derivative, and (bs, resp, 1) so it broadcasts over the candidates.
    a = -float(pg_coef) * pg_grad_coef.detach().to(torch.float32).unsqueeze(-1)
    # 1[v = y] lives at the emitted token. Inside the support that is one slot;
    # outside it, the tail bucket carries the whole spike.
    onehot = sampled_onehot.to(torch.float32)
    in_support = onehot.sum(dim=-1).clamp(max=1.0)
    g_grpo = a * (onehot - p_s)
    g_grpo_tail = a.squeeze(-1) * ((1.0 - in_support) - tail_s)

    # Per position, the three reductions every channel column is built from.
    # (W - 1) g0 is the mechanism's addition; a channel owning share C of W - 1
    # owns C * g0 of it, so a channel's norm and its dot with the policy
    # gradient are C^2 |g0|^2 and C (g0 . g_grpo) -- no extra reduction over the
    # support per channel, and none of the tensors below is new.
    g0, g0_tail = push["g0"], push["g0_tail"]
    g0_sq = (g0 * g0).sum(dim=-1) + g0_tail * g0_tail
    g0_grpo = (g0 * g_grpo).sum(dim=-1) + g0_tail * g_grpo_tail
    zero = torch.zeros_like(g0_sq)
    sh = zero if push_shared is None else push_shared.detach().to(torch.float32)
    sr = zero if push_source is None else push_source.detach().to(torch.float32)

    out = {
        "g_opd_sq": (g_opd * g_opd).sum(dim=-1) + g_opd_tail * g_opd_tail,
        "g_grpo_sq": (g_grpo * g_grpo).sum(dim=-1) + g_grpo_tail * g_grpo_tail,
        "g_dot": (g_opd * g_grpo).sum(dim=-1) + g_opd_tail * g_grpo_tail,
        "g_shared_sq": sh * sh * g0_sq,
        "g_shared_dot": sh * g0_grpo,
        "g_source_sq": sr * sr * g0_sq,
        "g_source_dot": sr * g0_grpo,
        "g_cross_dot": sh * sr * g0_sq,
    }
    if row_weight is not None:
        # Squared for the squared terms AND for the cross term, so the reported
        # norms are those of the weighted gradient rather than of the weighted
        # squared gradient, and every cosine below stays a ratio of like for
        # like.
        rw = row_weight.detach().to(torch.float32).reshape(-1, 1)
        rw2 = rw * rw
        out = {name: col * rw2 for name, col in out.items()}
    return out


def gradient_metrics(sums: dict, prefix: str = "kl_weight") -> dict:
    """Norm ratio and cosine between the two terms that share the logits.

    The ratio says how much of the update the OPD term is responsible for at
    all -- at ``teacher_kl_loss_coef = 0.01`` that is the first thing a reader
    wants and the last thing a loss curve shows. The cosine says whether the two
    are pulling the same way; a persistently negative one means the weighting is
    spending its budget against the reward signal, which is a finding and not a
    bug.
    """
    out = {}
    for scope, tot in sums.items():
        head = prefix if scope is None else f"{prefix}/{scope}"
        n = tot["n"]
        if n <= 0:
            continue
        opd = math.sqrt(max(tot["g_opd_sq"], 0.0))
        grpo = math.sqrt(max(tot["g_grpo_sq"], 0.0))
        out[f"{head}/grpo/grad_norm_opd"] = opd
        out[f"{head}/grpo/grad_norm_grpo"] = grpo
        if grpo > 1e-12:
            out[f"{head}/grpo/grad_norm_ratio"] = opd / grpo
        if opd > 1e-12 and grpo > 1e-12:
            out[f"{head}/grpo/grad_cosine"] = tot["g_dot"] / (opd * grpo)

        # WHICH CHANNEL the reward signal disagrees with. The cosine above is
        # taken on the applied W, which is the base OPD direction plus both
        # channels, and it is dominated by the base -- so it stays near whatever
        # the unweighted term does however the channels are allocated. These two
        # are on each channel's ADDITION alone.
        sh = math.sqrt(max(tot.get("g_shared_sq", 0.0), 0.0))
        src = math.sqrt(max(tot.get("g_source_sq", 0.0), 0.0))
        if sh <= 0.0 and src <= 0.0:
            # No partition was supplied -- an unweighted caller, where W - 1 is
            # identically zero and there is no addition to split. Two keys
            # pinned at 0.0 forever would read as "the channels did nothing",
            # which is a finding this caller is in no position to make.
            continue
        out[f"{head}/channel/shared_grad_norm"] = sh
        out[f"{head}/channel/source_grad_norm"] = src
        if grpo > 1e-12:
            if sh > 1e-12:
                out[f"{head}/grpo/shared_grad_cosine"] = tot["g_shared_dot"] / (sh * grpo)
            if src > 1e-12:
                out[f"{head}/grpo/source_grad_cosine"] = tot["g_source_dot"] / (src * grpo)
        if sh > 1e-12 and src > 1e-12:
            # >= 0 for any allocation, because both channels scale the SAME g0.
            # Read it as redundancy -- near 1 the two are the same mechanism
            # twice -- and never as "the channels agree", which it cannot deny.
            out[f"{head}/channel/shared_source_grad_cosine"] = tot["g_cross_dot"] / (sh * src)
    return out


# The arm-independent cut. Every other terms tuple in this module names a
# quantity built out of W; these are the columns that exist whether or not a
# weight was ever computed, so one accumulator with one definition runs in the
# weighted arm and in the control that has no off-task teachers at all.
#
# GRAD_TERMS' channel columns come along and stay at zero: the partition of
# W - 1 is empty when W is 1. gradient_metrics declines to render them rather
# than publishing two series pinned at 0.0, which would read as a measurement.
OPD_TERMS = GRAD_TERMS + ("d_kl", "push_abs")

# The role cut of the unweighted term, curated for the same reason
# ROLE_CUT_SUFFIXES is: six roles times every column opd_attribution_metrics can
# render is a hundred series a step, which is a worse analysis than four that
# are read. These four are the ones the weighted role cut has no answer for --
# where the plain OPD term pushes, how much of the KL sits there, and whether it
# pulls with the reward THERE rather than pooled.
OPD_ROLE_CUT_SUFFIXES = (
    "/kl_share", "/push_abs_share", "/grpo/grad_cosine", "/grpo/grad_norm_ratio",
)


def opd_attribution_terms(
    *,
    student_logprob: torch.Tensor,
    teacher_logprob: torch.Tensor,
    teacher_kl: torch.Tensor,
    pg_grad_coef: torch.Tensor,
    sampled_onehot: torch.Tensor,
    coef: float,
    pg_coef: float = 1.0,
    row_weight: Optional[torch.Tensor] = None,
    push: Optional[dict] = None,
    eps: float = 1e-8,
) -> dict:
    """:func:`logit_gradient_terms` at ``W = 1``, plus the two size columns.

    WHAT THE UNWEIGHTED OPD TERM DOES, measured on inputs a run needs no
    cross-teacher machinery to have: the student's own top-k, the on-task
    teacher's log-probs at it, and the policy-gradient coefficient. A control
    arm has all three and has none of the rest, which is why this is a separate
    entry point rather than a flag on the weighted one -- ``build_position_weight``
    would have nothing to read.

    In the weighted arm it is the SAME positions the weighted terms are taken
    on, so ``opd/grpo/grad_cosine`` beside ``kl_weight/grpo/grad_cosine`` is the
    weighting's effect on the gradient geometry with the policy held fixed. That
    is a stronger comparison than the same two numbers from two runs, whose
    policies have diverged by the step you read them.

    The two extra columns are LINEAR in the row weight where the gradient
    columns are quadratic (they are norms of a weighted gradient), so the
    scaling is applied to each half separately. One dict, two scalings, said
    here because a reader summing them would otherwise be right to expect one.

    ``d_kl``      the position's OPD KL, exactly what the loss carries there.
    ``push_abs``  ``sum_v |g0(v)| + |g0_tail|``, the gross size of the descent
                  direction. Its cut by role answers "where does the plain OPD
                  term push hardest", which is the denominator every claim about
                  where the WEIGHT moved budget is quoted against.

    ``push`` is the caller's already-built :func:`opd_logit_push`, when the
    per-token table has one; passing it is what keeps ``g0`` computed once per
    micro-batch and, more to the point, defined once.
    """
    if push is None:
        push = opd_logit_push(
            student_logprob=student_logprob, teacher_logprob=teacher_logprob,
            teacher_kl=teacher_kl, coef=coef, eps=eps,
        )
    ones = torch.ones_like(teacher_kl, dtype=torch.float32)
    out = logit_gradient_terms(
        student_logprob=student_logprob, teacher_logprob=teacher_logprob,
        weight=ones, teacher_kl=teacher_kl, pg_grad_coef=pg_grad_coef,
        sampled_onehot=sampled_onehot, coef=coef, pg_coef=pg_coef,
        row_weight=row_weight, push=push, eps=eps,
    )
    lin = {
        "d_kl": teacher_kl.detach().to(torch.float32),
        "push_abs": push["g0"].abs().sum(dim=-1) + push["g0_tail"].abs(),
    }
    if row_weight is not None:
        rw = row_weight.detach().to(torch.float32).reshape(-1, 1)
        lin = {name: col * rw for name, col in lin.items()}
    out.update(lin)
    return out


def opd_attribution_metrics(sums: dict, prefix: str = "opd") -> dict:
    """:func:`gradient_metrics` plus where the unweighted OPD budget sits.

    The share is over the scopes the accumulator actually holds, taken against
    the pooled row when it is present and against the sum of the parts when it
    is not -- which is the difference between the task cut (whose scopes are the
    whole batch) and the role cut (whose scopes are positions, and whose pooled
    row a caller may or may not have asked for).
    """
    out = gradient_metrics(sums, prefix=prefix)
    pooled = sums.get(None, None)
    parts = {k: v for k, v in sums.items() if k is not None}
    for term, label in (("d_kl", "kl"), ("push_abs", "push_abs")):
        total = (
            float(pooled[term]) if pooled is not None and term in pooled
            else sum(float(v[term]) for v in parts.values() if term in v)
        )
        for scope, tot in sums.items():
            if term not in tot or tot["n"] <= 0:
                continue
            head = prefix if scope is None else f"{prefix}/{scope}"
            out[f"{head}/{label}_mean"] = float(tot[term]) / tot["n"]
            if scope is not None and total > 0:
                out[f"{head}/{label}_share"] = float(tot[term]) / total
    return out


def state_shift_terms(built: dict, teacher_kl: torch.Tensor) -> dict:
    """The exact partition of ``(W - 1) D`` over the seven states plus the offset.

    The per-candidate part is ``p_d(v) e(v) / mu * D`` and the leftover is the
    normaliser's ``(1/mu - 1) D``, which belongs to no candidate. Their sum is
    the position's whole shift, which is what makes the columns a decomposition
    rather than a set of correlated summaries.
    """
    kl = teacher_kl.detach().to(torch.float32)
    inv_mu = 1.0 / built["mu"].clamp(min=1e-12)
    per_cand = per_candidate_shift(built, teacher_kl)
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
        # THE SOURCE CHANNEL'S TWO GATES, each as the share it let through of
        # what reached it. Reported as a chain rather than as one number because
        # they fail differently and the fix is different: a low
        # ``exclusive_pass_rate`` means the on-task teacher already covers what
        # the sources are saying -- the channel is redundant, not gated -- while
        # a low ``gate_pass_rate`` means the sources are saying things they do
        # not agree with each other about, which is the gate working.
        if tot["source_gross"] > 1e-12:
            out[f"{head}/evidence/exclusive_pass_rate"] = (
                tot["source_exclusive_gross"] / tot["source_gross"]
            )
            # Inside the same guard, so a counterfactual scope -- which is
            # handed no gate masses at all -- publishes nothing here instead of
            # a 0.0 that reads as a closed gate.
            out[f"{head}/evidence/gate_mean"] = tot["gate_mass"] / n
        if tot["source_exclusive_gross"] > 1e-12:
            out[f"{head}/evidence/gate_pass_rate"] = (
                tot["evidence"] - tot["evidence_shared"]
            ) / tot["source_exclusive_gross"]
        # WHERE EACH CHANNEL PUTS ITS BUDGET, against where the other one does.
        # Both channels only ever ADD weight, so they cannot fight at a
        # position; they compete for the fixed per-task mean mu preserves, which
        # makes "do they pick the same positions" the whole question.
        s_mean, r_mean = tot["push_shared"] / n, tot["push_source"] / n
        s_var = max(tot["push_shared_sq"] / n - s_mean * s_mean, 0.0)
        r_var = max(tot["push_source_sq"] / n - r_mean * r_mean, 0.0)
        # Gated on there being a partition at all, means included. A scope that
        # was handed none -- every counterfactual one is -- would otherwise
        # publish a channel push of exactly 0.0, which reads as "this scope
        # allocated nothing" rather than "this scope has no channels".
        if tot["push_shared_sq"] > 0.0 or tot["push_source_sq"] > 0.0:
            out[f"{head}/channel/shared_push_mean"] = s_mean
            out[f"{head}/channel/source_push_mean"] = r_mean
        if tot["push_shared_sq"] > 1e-24 and tot["push_source_sq"] > 1e-24:
            # Uncentered: how much the two channels' mass overlaps at all. In
            # [0, 1] because neither channel is ever negative, so a LOW value is
            # the finding -- it means the arm is running two mechanisms that
            # touch different text.
            out[f"{head}/channel/allocation_cosine"] = tot["push_cross"] / math.sqrt(
                tot["push_shared_sq"] * tot["push_source_sq"]
            )
        if s_var > 0 and r_var > 0:
            # Centered, and the one that can go negative. Below zero the
            # channels systematically pick different positions RELATIVE to their
            # own means: given a task budget that mu holds fixed, that is them
            # taking it from each other, which the uncentered cosine cannot say.
            out[f"{head}/channel/allocation_corr"] = (
                tot["push_cross"] / n - s_mean * r_mean
            ) / math.sqrt(s_var * r_var)
        if abs(tot["kl"]) > 1e-12:
            kl_scale = tot["w_kl"] / tot["kl"]
            out[f"{head}/effect/kl_scale"] = kl_scale
            out[f"{head}/effect/kl_shift_gross_frac"] = tot["kl_shift_abs"] / tot["kl"]
            out[f"{head}/effect/kl_scale_lag_error"] = kl_scale - 1.0
            # DID THE WEIGHT LAND WHERE THE KL WAS. kl_scale is the KL-weighted
            # mean of W and w_mean is the plain one, so their ratio is the lift a
            # position gets for carrying KL. Above 1 the arm put its budget on
            # the positions that already had the most to distil; at exactly 1 the
            # weight and the KL are uncorrelated and the arm is a scalar on the
            # whole term, i.e. teacher_kl_loss_coef with extra steps. This is the
            # cheapest statement of what the mechanism is doing and it costs no
            # accumulation -- both sums were already here.
            if abs(w_mean) > 1e-12:
                out[f"{head}/effect/weight_kl_lift"] = kl_scale / w_mean
            # The same question as a correlation, which unlike the lift is
            # bounded and comparable across runs. kl_sq is the one column added
            # for it.
            kl_var = max(tot["kl_sq"] / n - (tot["kl"] / n) ** 2, 0.0)
            if w_var > 0 and kl_var > 0:
                cov = tot["w_kl"] / n - w_mean * (tot["kl"] / n)
                out[f"{head}/effect/weight_kl_corr"] = cov / math.sqrt(w_var * kl_var)
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
