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

"""Precision-weighted combination of the RL and OPD signals, per vocabulary id.

See ``docs/opd_logit_precision_weighting_design.md``. The short version:

WHERE THE SIGNALS MEET. In one step the shared parameters receive six signals --
three policy gradients and three teacher-KL gradients. They do not meet at a
POSITION: task i's positions carry only RL_i and OPD_i. They meet at a shared
VOCABULARY ID, because every position's logit for id v is produced by the same
output row. So "conflict" is a statement about an id, and the natural coordinate
for a mechanism is the id -- which is also the gradient of a hypothetical shared
output bias b_v, split by source:

    a_i[v] = sum_{t in T_i} rho_t c_t (1[y_t = v] - pi_t[v])        RL,  3 of them
    d_j[v] = sum_{t in T_j} rho_t g_opd(t)[v]                       OPD, 3 of them

WHAT IS OPTIMAL. Model a_i[v] = theta_i[v] + eps (unbiased, variance sigma_i^2)
and d_j[v] = lambda_j theta_j[v] + zeta (unknown scale, error variance varsigma^2),
where theta_i is task i's true objective gradient at that coordinate. Among all
linear combinations that are unbiased for theta_i, the minimum-variance one is
the precision-weighted average -- Gauss-Markov, no Gaussian assumption needed:

    w_i = (1/sigma^2) / (1/sigma^2 + lambda^2/varsigma^2)
    u_i = (lambda/varsigma^2) / (1/sigma^2 + lambda^2/varsigma^2)

WHAT THAT DOES AND DOES NOT ESTABLISH. Gauss-Markov makes this the
minimum-variance unbiased estimate OF THE PER-ID BIAS GRADIENT. It does not
follow that it is the best cotangent to push through the network, and the
tempting argument that "Adam normalises each coordinate so only direction and
SNR are left" does not close the gap: Adam normalises per PARAMETER, while
g_z[v] flows into the output row w_v (2048 numbers, and tied to the input
embedding on this model) and into the trunk, where every id's contribution is
mixed before any normalisation happens. The id is where the signals MEET, which
is what makes the combination well posed; it is not a basis in which the
optimiser is diagonal.

WHAT lambda MEASURES, WHICH IS NOT TRUST. The model says the teacher's push is a
scaled noisy copy of the reward's, so anything the teacher pushes that is
ORTHOGONAL to the current reward gradient -- which is where a teacher's value
is supposed to live -- lands in varsigma^2 and attenuates it. lambda/varsigma^2
is therefore a measure of REDUNDANCY WITH THE REWARD, not of how good the
teacher is. A teacher that knows something the reward has not found yet scores
low. That is a real limitation of this estimator and not a wording problem;
see the design doc's limitations.

A second layer shrinks the combined estimate toward zero by its own reliability
(James-Stein). It is computed and reported but NOT applied -- see fit_weights
for the failure mode that decides that.

There is NO cross-task gate. Teacher j is evidence about task j's objective and
about nothing else, so the tasks meet only in the final sum -- which is what
plain OPD+GRPO already does. The mechanism is therefore SINGLE-TASK, applied
once per task; see the design doc section 10.

WHAT TODAY'S CODE IS, IN THESE TERMS. ``loss = pg + beta * teacher_kl`` with
beta = 0.01 fixes w_i = 1 and u_i = beta for every task and every id. Since
u/w = lambda sigma^2 / varsigma^2, the constant beta is an unmeasured claim that
this ratio is 0.01 everywhere. This module measures it instead.

WHY THIS COSTS NO EXTRA FORWARD OR BACKWARD. Both signals are already separate
in logit space -- it is the loss that adds them. Weighting them before the sum
and pushing a single cotangent through the trunk is one backward, the one the
step already pays. Nothing here needs the parameter-space gradients separated,
which is the thing that cannot be done for free.

THE RANK-1 STRUCTURE IS WHAT MAKES THE AGGREGATION CHEAP. Both logit gradients
are (something sparse) + (a scalar times pi_t):

    g_rl(t)[u]  = c_t * (1[u = y_t] - pi_t[u])
    g_opd(t)[u] = pi_t[u] * kappa_t  +  1[u in A_t] * pi_t[u] * (F_t - f_t(u))

so summing them over a task's positions needs only scatter-adds over the k
support plus a few weighted column sums of pi. No (tokens x vocab) intermediate
has to be kept, and the pi sums are taken in chunks over positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

__all__ = [
    "LogitPrecisionConfig",
    "topk_kl_logit_grad",
    "StepAccumulator",
    "fit_weights",
    "TaskFit",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LogitPrecisionConfig:
    """Knobs. Everything that changes a number the mechanism reports lives here.

    ``observe_only`` is the pilot: the statistics and the weights are computed
    and logged, and the loss is untouched. It is the default because the weights
    have never been measured, and the design doc's go/no-go is stated in terms
    of what they turn out to be.
    """

    enable: bool = False
    # Compute and log, do not apply. The pilot.
    observe_only: bool = True
    # EMA over steps, applied to the ACCUMULATORS rather than to the fitted
    # values: the fit is a ratio of moments, and smoothing the moments keeps it
    # a ratio of smoothed moments instead of an average of noisy ratios.
    ema_decay: float = 0.8
    # Activity strata for the prior variance. An id that appears constantly has
    # both a larger signal and a larger noise floor than a rare one, so one
    # prior variance across the whole vocabulary fits neither.
    n_strata: int = 10
    # Ids whose activity is below this are dropped from the fit entirely (they
    # carry no information and would dominate the strata boundaries).
    min_activity: float = 1e-6
    # Floor on the permutation variance, as a fraction of that stratum's mean.
    # Without it an id that occurs in exactly one rollout gets sigma^2 = 0 and
    # therefore infinite precision.
    sigma2_floor_frac: float = 1e-3
    # Positions per chunk when summing pi. Trades peak memory for launches:
    # a chunk holds (rows x chunk_tokens x vocab) floats.
    chunk_tokens: int = 32
    # A negative lambda says the teacher pushes AGAINST what improves its own
    # task. The model's answer is to flip the teacher's sign, which is a large
    # claim to act on from one regression; clamped to zero by default, and the
    # unclamped value is reported either way.
    allow_negative_lambda: bool = False
    # Minimum tokens a task must contribute before its fit is used at all.
    min_tokens: int = 64
    # Effective number of smoothed steps required before sigma^2 is taken as the
    # ACROSS-STEP variance of the RL push. Below it the permutation null is used
    # instead and the fit is flagged `warming_up`, which the apply path refuses.
    # See the note on sigma^2 in fit_weights for why the two differ and why the
    # across-step one is the quantity the model actually assumes.
    min_eff_steps: float = 3.0
    # Floor on the teacher's residual variance, as a fraction of the variance of
    # its own push. NUMERICAL SAFETY ONLY, which is why it is small: the
    # errors-in-variables correction divides by an ESTIMATED signal variance, so
    # when that estimate is low the fit can credit the teacher with explaining
    # more than all of its own variance and hand back a negative residual, at
    # which point the precision lambda^2/varsigma^2 diverges. A negative
    # residual means the prior was underestimated, not that the teacher is
    # useless, so the floor caps rather than switches off. Set large enough to
    # distort a genuinely excellent teacher and it becomes the mechanism.
    min_residual_frac: float = 1e-3
    # Hard cap on the per-id teacher coefficient, as a multiple of the run's
    # base kl_loss_coef. Deliberately loose: it exists so a degenerate fit
    # cannot put an unbounded coefficient on the teacher, NOT to shape the
    # answer. ``beta_at_cap_frac`` is logged so a cap that starts binding is
    # visible rather than silently becoming the mechanism -- which is what
    # lambda_max turned out to be in the cross-gate strength arm.
    max_beta_ratio: float = 1000.0

    def validate(self) -> None:
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {self.n_strata}")
        if self.chunk_tokens < 1:
            raise ValueError(f"chunk_tokens must be >= 1, got {self.chunk_tokens}")
        if not (0.0 < self.min_residual_frac <= 1.0):
            raise ValueError(f"min_residual_frac must be in (0, 1], got {self.min_residual_frac}")
        if not (self.max_beta_ratio > 0.0) or not math.isfinite(self.max_beta_ratio):
            raise ValueError(f"max_beta_ratio must be finite and > 0, got {self.max_beta_ratio}")
        if not (self.min_tokens >= 1):
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")
        if not (self.min_activity >= 0.0) or not math.isfinite(self.min_activity):
            raise ValueError(f"min_activity must be finite and >= 0, got {self.min_activity}")
        if not (self.min_eff_steps >= 1.0) or not math.isfinite(self.min_eff_steps):
            raise ValueError(f"min_eff_steps must be finite and >= 1, got {self.min_eff_steps}")
        if not (self.sigma2_floor_frac >= 0.0) or not math.isfinite(self.sigma2_floor_frac):
            raise ValueError(f"sigma2_floor_frac must be finite and >= 0, got {self.sigma2_floor_frac}")


# ---------------------------------------------------------------------------
# The OPD logit gradient, in closed form
# ---------------------------------------------------------------------------


def topk_kl_logit_grad(
    student_topk_logprob: torch.Tensor,
    teacher_topk_logprob: torch.Tensor,
    eps: float = 1e-8,
):
    """``-d/d(logits)`` of :func:`core_algos.topk_kl_per_token`, factorised.

    That loss is a reverse KL over the student's top-k support plus one tail
    bucket::

        L = sum_{v in A} p_s(v) [log p_s(v) - log p_t(v)]  +  tail_s (log tail_s - log tail_t)

    Differentiating through the softmax gives, for EVERY id u (not just the ones
    in the support)::

        -dL/dz_u = pi[u] * kappa  +  1[u in A] * pi[u] * (F - f(u))
        f(v)  = log p_s(v) - log p_t(v)          on the support
        F     = log tail_s - log tail_t
        S1    = sum_{v in A} p_s(v) f(v)
        P_A   = sum_{v in A} p_s(v)
        kappa = S1 - P_A * F

    The off-support part is therefore proportional to ``pi`` with one scalar per
    position -- rank one in the vocabulary. That is what lets a whole task's
    OPD push be aggregated into id space with a scatter over k ids plus one
    weighted column sum of pi, instead of a (tokens x vocab) intermediate.

    A test differentiates ``topk_kl_per_token`` with autograd and compares, so a
    change to the loss that is not mirrored here fails rather than silently
    reporting the gradient of the old objective.

    Returns:
        ``(support_term, kappa)`` where ``support_term`` is
        ``pi[u] (F - f(u))`` at the k support ids, shape (bs, resp, k), and
        ``kappa`` is (bs, resp). The caller multiplies both by ``pi`` -- the
        support term already carries its own ``pi`` factor, ``kappa`` does not.
    """
    ps = student_topk_logprob.exp()
    p_a = ps.sum(dim=-1)
    raw_tail_s = 1.0 - p_a
    tail_s = raw_tail_s.clamp(min=eps, max=1.0)
    tail_t = (1.0 - teacher_topk_logprob.exp().sum(dim=-1)).clamp(min=eps, max=1.0)

    f = student_topk_logprob - teacher_topk_logprob            # (bs, resp, k)
    big_f = tail_s.log() - tail_t.log()                        # (bs, resp)
    # WHERE THE TAIL CLAMP BINDS, tail_s is the constant eps and carries no
    # gradient, so the tail term drops out of dL/dz. Algebraically that is the
    # same expression with F replaced by -1: the unclamped result is
    #   p_s(u) {1[u in A] f(u) - S1 - F (1[u in A] - P_A)}
    # and dropping the tail term leaves
    #   p_s(u) {1[u in A] (f(u) + 1) - S1 - P_A},
    # which is the first line at F = -1. Substituting keeps one code path.
    # It is not a rare branch: the design's own note records tail_mass_mean at
    # 0.000-0.001 on this mixture, so a student top-20 that captures the whole
    # mass to within 1e-8 does occur.
    big_f = torch.where(raw_tail_s > eps, big_f, torch.full_like(big_f, -1.0))
    s1 = (ps * f).sum(dim=-1)                                  # (bs, resp)
    kappa = s1 - p_a * big_f                                   # (bs, resp)

    support_term = ps * (big_f.unsqueeze(-1) - f)              # (bs, resp, k)
    return support_term, kappa


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


class StepAccumulator:
    """The six pushes and the permutation null, summed over a step.

    Everything is (n_tasks, vocab) or (n_tasks,), so the whole state is a few
    tens of megabytes and all-reduces as plain sums. Nothing here is per
    position or per row once a micro-batch has been consumed.

    THE NULL. ``sigma2`` is the variance of ``a_i[v]`` under permuting the
    advantages across the task's rows, which is the distribution of the RL push
    when the id's occurrence carries no information about reward. GRPO centres
    the advantages within each prompt group, so the task mean is exactly zero and
    the permuted mean is exactly zero too. The classical permutation variance is

        Var = [sum_r (A_r - Abar)^2] [sum_r (m_r[v] - mbar[v])^2] / (N - 1)

    with ``m_r`` the row's advantage-free id profile. It is a closed form; no
    resampling is done. Permuting across the task's rows rather than within each
    prompt group is a slightly wider null -- it destroys group identity as well
    as the advantage association -- so it errs toward a LARGER noise floor and
    therefore more shrinkage, which is the safe direction.

    The profile ``m_r`` uses the PPO ratio without the clip branches, while
    ``a_i`` uses the exact coefficient. The measured clip rate over this run's
    first 150 steps is 1.28 percent pooled (alfworld 1.37, webshop 1.11, search
    0.49), so the two agree to about that order. The point estimate stays exact
    -- it uses the real coefficient -- and only the noise floor carries the
    approximation. (An earlier version of this note quoted 0.085 percent, which
    is the term probe's alfworld stage-0 figure and not what the run does.)
    """

    def __init__(self, n_tasks: int, vocab: int, *, device=None, dtype=torch.float32):
        z2 = lambda: torch.zeros(n_tasks, vocab, device=device, dtype=dtype)
        z1 = lambda: torch.zeros(n_tasks, device=device, dtype=dtype)
        self.n_tasks, self.vocab = int(n_tasks), int(vocab)
        self.a = z2()          # RL push
        self.d = z2()          # OPD push
        self.act = z2()        # activity, sum_t rho_t pi_t[v] -- the strata key
        self.sm = z2()         # sum_r m_r[v]
        self.sm2 = z2()        # sum_r m_r[v]^2
        self.sa2 = z1()        # sum_r A_r^2
        self.n_rows = z1()
        self.n_tokens = z1()

    # -- plumbing ---------------------------------------------------------

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def zero_(self):
        for v in self.__dict__.values():
            if torch.is_tensor(v):
                v.zero_()
        return self

    def all_reduce_(self):
        """Sums across data-parallel ranks. Every field is a plain sum."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self
        for v in self.__dict__.values():
            if torch.is_tensor(v):
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
        return self

    def ema_(self, other: "StepAccumulator", decay: float):
        """``self <- decay * self + (1 - decay) * other``, field by field."""
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                v.mul_(decay).add_(getattr(other, k).to(v.device), alpha=1.0 - decay)
        return self

    def state_dict(self) -> dict:
        out = {"n_tasks": self.n_tasks, "vocab": self.vocab}
        out.update({k: v.detach().cpu() for k, v in self.__dict__.items() if torch.is_tensor(v)})
        return out

    def load_state_dict(self, sd: dict):
        if int(sd.get("n_tasks", -1)) != self.n_tasks or int(sd.get("vocab", -1)) != self.vocab:
            raise ValueError(
                f"logit-precision state is for n_tasks={sd.get('n_tasks')} vocab={sd.get('vocab')}, "
                f"this run has n_tasks={self.n_tasks} vocab={self.vocab}"
            )
        for k, v in sd.items():
            cur = getattr(self, k, None)
            if torch.is_tensor(cur):
                cur.copy_(v.to(cur.device, cur.dtype))
        return self

    # -- the two methods that read a micro-batch --------------------------

    @torch.no_grad()
    def moments(self, cfg: "LogitPrecisionConfig") -> "StepMoments":
        """This step's per-id second moments and noise floor, ready to smooth.

        The permutation variance is formed HERE, from this step's own row sums,
        because it is not linear in them: ``sum_r m_r^2 - (sum_r m_r)^2 / N``
        mixes degrees, and smoothing the three sums separately and combining
        them afterwards is a different quantity. See :class:`StepMoments`.
        """
        out = StepMoments(self.n_tasks, self.vocab, device=self.a.device, dtype=self.a.dtype)
        n = self.n_rows.clamp(min=1.0).unsqueeze(-1)
        spread = (self.sm2 - self.sm * self.sm / n).clamp(min=0.0)
        denom = (self.n_rows - 1.0).clamp(min=1.0).unsqueeze(-1)
        out.sig2.copy_(spread * (self.sa2.unsqueeze(-1) / denom))
        out.a.copy_(self.a)
        out.d.copy_(self.d)
        out.a2.copy_(self.a * self.a)
        out.ad.copy_(self.a * self.d)
        out.d2.copy_(self.d * self.d)
        out.act.copy_(self.act)
        out.n_tokens.copy_(self.n_tokens)
        out.n_rows.copy_(self.n_rows)
        out.w.fill_(1.0)
        out.w2.fill_(1.0)
        return out

    @torch.no_grad()
    def add_packed(
        self,
        *,
        logits: torch.Tensor,          # (T, V) response-token logits, PACKED
        token_row: torch.Tensor,       # (T,) which row of this micro-batch
        token_task: torch.Tensor,      # (T,) which task
        sampled_ids: torch.Tensor,     # (T,) the id that was sampled
        w_rl: torch.Tensor,            # (T,) rho * c_t
        w_opd: torch.Tensor,           # (T,) rho * kappa_t
        w_act: torch.Tensor,           # (T,) rho
        w_prof: torch.Tensor,          # (T,) rho * ratio_t, for the null
        row_task: torch.Tensor,        # (R,) task of each row, -1 to skip
        row_adv: torch.Tensor,         # (R,) the row's advantage
        topk_ids: Optional[torch.Tensor] = None,      # (T, k)
        support_term: Optional[torch.Tensor] = None,  # (T, k), RAW from topk_kl_logit_grad
        token_valid: Optional[torch.Tensor] = None,   # (T,) 1 where the token counts
        chunk_tokens: int = 32,
    ):
        """Fold one micro-batch in, taking the logits in the packed layout.

        PACKED IS THE LAYOUT THAT EXISTS. With ``use_remove_padding`` and
        ``response_only_logits`` the forward produces (total response tokens,
        vocab) and never builds the padded (rows, response, vocab) cube. Asking
        for the cube back would allocate the padding -- on this mixture the
        response lengths run from tens to 512 -- so the accumulation reads what
        the forward already has.

        ``pi`` is rebuilt a chunk of tokens at a time and dropped. The largest
        intermediate is (chunk_tokens, vocab), which at the default is 19 MB at
        this vocabulary. Nothing of size (tokens, vocab) is ever held.

        Rows whose ``row_task`` is negative contribute nothing: their tokens'
        weights are zeroed here rather than filtered, so the packed indices stay
        valid and no branch depends on how many such rows there are.

        ``support_term`` is the RAW output of :func:`topk_kl_logit_grad`, with no
        row weight applied. It is weighted here by ``w_act``, which is exactly
        ``rho_t`` on the kept, unmasked tokens -- the same factor ``w_opd``
        already carries on the dense half of the same gradient. Making the
        caller apply it to one half and not the other is how the two halves of
        one gradient end up on different scales.
        """
        if logits.dim() != 2:
            raise ValueError(f"add_packed wants (T, V) logits, got {tuple(logits.shape)}")
        n_tok, vocab = logits.shape
        if vocab != self.vocab:
            raise ValueError(f"vocab mismatch: accumulator {self.vocab}, logits {vocab}")
        dev = logits.device
        work = torch.promote_types(logits.dtype, torch.float32)
        acc_dtype = self.a.dtype
        n_rows_mb = int(row_task.numel())

        row_keep = (row_task >= 0).to(dev)
        # A packed caller has already dropped the padding, so validity defaults
        # to all-ones. The padded wrapper passes the response mask, and without
        # it ``n_tokens`` would count padding positions -- which are weighted to
        # zero everywhere else and so would silently inflate only the count that
        # the min_tokens gate reads.
        tok_keep = row_keep[token_row.to(dev)].to(work)
        if token_valid is not None:
            tok_keep = tok_keep * token_valid.to(dev, work)
        # A row that contributes no weighted token contributes nothing to a, d,
        # sm or sm2 -- adjust_batch padding has task_loss_weight 0 -- so letting
        # it into n_rows and sa2 would put the numerator and the denominator of
        # sigma^2 on different populations, and would break the sum_r A_r = 0
        # the permutation null rests on. Measured at 1 row in ~7000, so this is
        # exactness rather than a correction.
        row_mass = torch.zeros(n_rows_mb, device=dev, dtype=work)
        row_mass.index_add_(0, token_row.to(dev, torch.int64), tok_keep)
        row_keep = row_keep & (row_mass > 0)
        tok_keep = tok_keep * row_keep[token_row.to(dev)].to(work)
        trow = token_row.to(dev, torch.int64)
        ttask = token_task.to(dev, torch.int64).clamp(min=0)
        sid = sampled_ids.to(dev, torch.int64)

        w_rl = w_rl.to(dev, work) * tok_keep
        w_opd = w_opd.to(dev, work) * tok_keep
        w_act = w_act.to(dev, work) * tok_keep
        w_prof = w_prof.to(dev, work) * tok_keep

        # Dense parts: both logit gradients are (something sparse) + (a scalar
        # times pi), so these three weighted column sums of pi are the whole
        # dense contribution. The profile needs per-ROW resolution because the
        # permutation null squares it; the others only need per-task.
        dense_rl = torch.zeros(self.n_tasks, vocab, device=dev, dtype=work)
        dense_opd = torch.zeros_like(dense_rl)
        dense_act = torch.zeros_like(dense_rl)
        prof_dense = torch.zeros(n_rows_mb, vocab, device=dev, dtype=work)
        step = max(1, int(chunk_tokens))
        for c0 in range(0, n_tok, step):
            c1 = min(n_tok, c0 + step)
            pi = torch.softmax(logits[c0:c1].to(work), dim=-1)
            tt, tr = ttask[c0:c1], trow[c0:c1]
            dense_rl.index_add_(0, tt, w_rl[c0:c1, None] * pi)
            dense_opd.index_add_(0, tt, w_opd[c0:c1, None] * pi)
            dense_act.index_add_(0, tt, w_act[c0:c1, None] * pi)
            prof_dense.index_add_(0, tr, w_prof[c0:c1, None] * pi)
            del pi

        # Sparse parts, as flat scatters so one call covers every (group, id).
        def _flat(n_groups, group, ids, vals):
            out = torch.zeros(n_groups * vocab, device=dev, dtype=work)
            out.scatter_add_(0, (group * vocab + ids).reshape(-1), vals.reshape(-1))
            return out.view(n_groups, vocab)

        sampled_rl = _flat(self.n_tasks, ttask, sid, w_rl)
        prof_sampled = _flat(n_rows_mb, trow, sid, w_prof)
        opd_support = torch.zeros_like(dense_opd)
        if topk_ids is not None and support_term is not None:
            k = topk_ids.shape[-1]
            opd_support = _flat(
                self.n_tasks,
                ttask.unsqueeze(-1).expand(-1, k),
                topk_ids.to(dev, torch.int64),
                support_term.to(dev, work) * w_act.unsqueeze(-1),
            )

        self.a += (sampled_rl - dense_rl).to(acc_dtype)
        self.d += (opd_support + dense_opd).to(acc_dtype)
        self.act += dense_act.to(acc_dtype)

        # Per-row moments of the advantage-free profile, then into the task.
        m_row = prof_sampled - prof_dense
        keepf = row_keep.to(work).unsqueeze(-1)
        m_row = m_row * keepf
        rtask = row_task.to(dev, torch.int64).clamp(min=0)
        self.sm.index_add_(0, rtask, m_row.to(acc_dtype))
        self.sm2.index_add_(0, rtask, (m_row * m_row).to(acc_dtype))
        adv = row_adv.to(dev, work) * row_keep.to(work)
        self.sa2.index_add_(0, rtask, (adv * adv).to(acc_dtype))
        self.n_rows.index_add_(0, rtask, row_keep.to(acc_dtype))
        tok_per_task = torch.zeros(self.n_tasks, device=dev, dtype=work)
        tok_per_task.index_add_(0, ttask, tok_keep)
        self.n_tokens += tok_per_task.to(acc_dtype)
        return self

    @torch.no_grad()
    def add_micro_batch(
        self,
        *,
        logits: torch.Tensor,             # (bs, resp, V)
        responses: torch.Tensor,          # (bs, resp)
        response_mask: torch.Tensor,      # (bs, resp)
        task_ids: torch.Tensor,           # (bs,), -1 to skip the row
        row_weight: torch.Tensor,         # (bs,) rho_r
        pg_coef: torch.Tensor,            # (bs, resp) c_t
        ratio: torch.Tensor,              # (bs, resp) the PPO ratio
        row_advantage: torch.Tensor,      # (bs,)
        topk_ids: Optional[torch.Tensor] = None,
        student_topk_logprob: Optional[torch.Tensor] = None,
        teacher_topk_logprob: Optional[torch.Tensor] = None,
        chunk_tokens: int = 32,
    ):
        """The padded-layout entry point: flattens and calls :meth:`add_packed`.

        The trainer uses the packed one. This exists because a (rows, response)
        view is what a reader can check by hand, so the tests drive the same
        arithmetic through a shape they can write a brute-force loop against.
        """
        bs, resp, vocab = logits.shape
        dev = logits.device
        work = torch.promote_types(logits.dtype, torch.float32)
        mask = response_mask.to(dev, work)
        rho = row_weight.to(dev, work).unsqueeze(-1)

        support_flat = None
        kappa = torch.zeros(bs, resp, device=dev, dtype=work)
        if topk_ids is not None and student_topk_logprob is not None and teacher_topk_logprob is not None:
            support, kappa = topk_kl_logit_grad(
                student_topk_logprob.to(work), teacher_topk_logprob.to(work)
            )
            support_flat = support.reshape(bs * resp, -1)

        rows = torch.arange(bs, device=dev).unsqueeze(-1).expand(bs, resp).reshape(-1)
        return self.add_packed(
            logits=logits.reshape(bs * resp, vocab),
            token_row=rows,
            token_task=task_ids.to(dev, torch.int64).clamp(min=0).unsqueeze(-1).expand(bs, resp).reshape(-1),
            sampled_ids=responses.reshape(-1),
            w_rl=(pg_coef.to(dev, work) * mask * rho).reshape(-1),
            w_opd=(kappa * mask * rho).reshape(-1),
            w_act=(mask * rho).reshape(-1),
            w_prof=(ratio.to(dev, work) * mask * rho).reshape(-1),
            token_valid=mask.reshape(-1),
            row_task=task_ids,
            row_adv=row_advantage,
            topk_ids=None if topk_ids is None else topk_ids.reshape(bs * resp, -1),
            support_term=support_flat,
            chunk_tokens=chunk_tokens,
        )



class StepMoments:
    """The quantities the fit consumes, averaged over steps the RIGHT way.

    WHY THIS CLASS EXISTS, WHICH IS A BUG THE PILOT FOUND IN ITS FIRST STEP.
    The first version smoothed the RAW accumulators -- sums over rows -- and
    then formed the fit from them. That is wrong, and not subtly: an EMA that
    starts at zero scales every accumulator by c = 1 - decay^t, and the prior
    variance

        tau^2 = mean_v(a[v]^2) - mean_v(sigma^2[v])

    is QUADRATIC in c through the first term and LINEAR through the second, so
    what the fit actually saw at step t was

        tau^2 = c * ( c * mean(a^2) - mean(sigma^2) )

    which is negative unless the reliability already exceeds 1 - c. At step 1,
    c = 0.2, so it needed 0.8. The run duly reported tau^2 = 0,
    no_rl_signal_frac = 1.0 and "rl_push_is_all_noise" for all three tasks --
    an artefact of the smoother, reported as a measurement.

    The fix is to smooth quantities of MATCHING DEGREE: the per-id second
    moments themselves (a^2, a*d, d^2), the per-id noise floor, and the
    activity. Each is averaged as itself, so the fit is a ratio of averages
    rather than an average of ratios of mismatched powers.

    The average is also BIAS CORRECTED. ``w`` accumulates the same way the
    fields do, and dividing by it makes the estimate the weighted mean of the
    per-step values at EVERY step rather than only asymptotically -- so step 1
    gives exactly the single-step fit instead of one fifth of it.
    """

    _PER_ID = ("a", "d", "a2", "ad", "d2", "sig2", "act")
    _PER_TASK = ("n_tokens", "n_rows")
    # w is the sum of the EMA weights and w2 the sum of their squares, so
    # w^2 / w2 is the effective number of steps the average rests on -- 1 after
    # one step, rising to (1 + decay) / (1 - decay) in the limit. The fit needs
    # it because the ACROSS-STEP variance of the push cannot be estimated from
    # fewer than two effective steps, and is worthless from three.
    _WEIGHTS = ("w", "w2")

    def __init__(self, n_tasks: int, vocab: int, *, device=None, dtype=torch.float32):
        self.n_tasks, self.vocab = int(n_tasks), int(vocab)
        for name in self._PER_ID:
            setattr(self, name, torch.zeros(n_tasks, vocab, device=device, dtype=dtype))
        for name in self._PER_TASK:
            setattr(self, name, torch.zeros(n_tasks, device=device, dtype=dtype))
        for name in self._WEIGHTS:
            setattr(self, name, torch.zeros((), device=device, dtype=dtype))

    def _fields(self):
        return self._PER_ID + self._PER_TASK + self._WEIGHTS

    def n_eff_steps(self) -> float:
        """``w^2 / w2``: how many independent steps this average is worth."""
        w, w2 = float(self.w), float(self.w2)
        return (w * w / w2) if w2 > 0 else 0.0

    def to(self, device):
        for name in self._fields():
            setattr(self, name, getattr(self, name).to(device))
        return self

    def ema_(self, other: "StepMoments", decay: float):
        """``self <- decay * self + (1 - decay) * other``.

        ``w2`` is the only field that does not follow that rule: the sum of
        SQUARED weights composes as ``decay^2 * w2 + (1 - decay)^2``, which is
        what makes ``w^2 / w2`` the effective step count.
        """
        d = float(decay)
        for name in self._PER_ID + self._PER_TASK + ("w",):
            cur = getattr(self, name)
            cur.mul_(d).add_(getattr(other, name).to(cur.device, cur.dtype), alpha=1.0 - d)
        self.w2.mul_(d * d).add_(other.w2.to(self.w2.device, self.w2.dtype), alpha=(1.0 - d) ** 2)
        return self

    def corrected(self) -> "StepMoments":
        """The bias-corrected average: every field divided by the weight sum."""
        out = StepMoments(self.n_tasks, self.vocab, device=self.w.device, dtype=self.w.dtype)
        wv = float(self.w)
        scale = 1.0 / wv if wv > 0 else 0.0
        for name in self._PER_ID + self._PER_TASK:
            getattr(out, name).copy_(getattr(self, name) * scale)
        # Normalised so the effective step count survives: w = 1 and
        # w2 = w2 / w^2, hence n_eff = w^2 / w2 is unchanged.
        out.w.fill_(1.0 if wv > 0 else 0.0)
        out.w2.copy_(self.w2 * (scale * scale))
        return out

    def state_dict(self) -> dict:
        out = {"n_tasks": self.n_tasks, "vocab": self.vocab}
        out.update({n: getattr(self, n).detach().cpu() for n in self._fields()})
        return out

    def load_state_dict(self, sd: dict):
        if int(sd.get("n_tasks", -1)) != self.n_tasks or int(sd.get("vocab", -1)) != self.vocab:
            raise ValueError(
                f"logit-precision moments are for n_tasks={sd.get('n_tasks')} "
                f"vocab={sd.get('vocab')}, this run has n_tasks={self.n_tasks} vocab={self.vocab}"
            )
        for name in self._fields():
            cur = getattr(self, name)
            if name in sd:
                cur.copy_(sd[name].to(cur.device, cur.dtype))
        return self


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


_REASON_CODE = {
    "ok": 0,
    "warming_up": 6,
    "residual_floored": 1,
    "teacher_uninformative": 2,
    "too_few_tokens": 3,
    "too_few_ids": 4,
    "rl_push_is_all_noise": 5,
}


@dataclass
class TaskFit:
    """One task's estimated weights, plus everything needed to audit them."""

    task: str
    valid: bool
    reason: str
    # Scalars
    tau2: float = 0.0          # prior variance of the true signal, pooled
    lam: float = 0.0           # teacher scale, after the sign clamp
    lam_raw: float = 0.0       # before the clamp -- a negative here is the finding
    varsigma2: float = 0.0     # teacher error variance
    r2: float = 0.0            # fraction of the teacher's push the reward explains
    lam_se: float = float("nan")   # jackknife-over-ids standard error of lam_raw
    n_eff_steps: float = 0.0       # how many independent steps the average rests on
    sigma2_is_across_step: bool = False   # False = the permutation null (warm-up only)
    sigma2_perm_median: float = 0.0
    n_eff: float = 0.0             # Kish effective number of ids deciding lam
    n_ids: int = 0
    n_tokens: float = 0.0
    at_cap_frac: float = 0.0
    # Share of kept ids in a stratum where the RL push is indistinguishable
    # from its own permutation null. There the teacher is the only source, and
    # a mechanism that shrinks by an RL-estimated prior would zero them out.
    no_rl_signal_frac: float = 0.0
    # Per id, or None when the fit is not valid
    sigma2: Optional[torch.Tensor] = None
    implied_beta: Optional[torch.Tensor] = None   # omega_D / omega_R -- the per-id beta
    shrink: Optional[torch.Tensor] = None         # k[v], the James-Stein factor
    keep: Optional[torch.Tensor] = None           # ids that entered the fit

    def metrics(self, prefix: str = "logit_prec") -> dict:
        m = {
            f"{prefix}/{self.task}/valid": float(self.valid),
            f"{prefix}/{self.task}/lam": self.lam,
            f"{prefix}/{self.task}/lam_raw": self.lam_raw,
            f"{prefix}/{self.task}/varsigma2": self.varsigma2,
            f"{prefix}/{self.task}/tau2": self.tau2,
            f"{prefix}/{self.task}/teacher_r2": self.r2,
            f"{prefix}/{self.task}/lam_se": self.lam_se,
            f"{prefix}/{self.task}/lam_t": (
                self.lam_raw / self.lam_se
                if self.lam_se and math.isfinite(self.lam_se) and self.lam_se > 0
                else float("nan")
            ),
            f"{prefix}/{self.task}/n_eff_ids": self.n_eff,
            f"{prefix}/{self.task}/n_eff_steps": self.n_eff_steps,
            f"{prefix}/{self.task}/sigma2_across_step": float(self.sigma2_is_across_step),
            f"{prefix}/{self.task}/sigma2_perm_median": self.sigma2_perm_median,
            f"{prefix}/{self.task}/n_ids": float(self.n_ids),
            f"{prefix}/{self.task}/beta_at_cap_frac": self.at_cap_frac,
            f"{prefix}/{self.task}/no_rl_signal_frac": self.no_rl_signal_frac,
            f"{prefix}/{self.task}/reason": float(_REASON_CODE.get(self.reason, -1)),
        }
        if self.implied_beta is not None and self.keep is not None and bool(self.keep.any()):
            b = self.implied_beta[self.keep]
            w = self.shrink[self.keep] if self.shrink is not None else None
            m[f"{prefix}/{self.task}/beta_median"] = float(b.median())
            m[f"{prefix}/{self.task}/beta_p10"] = float(b.quantile(0.10))
            m[f"{prefix}/{self.task}/beta_p90"] = float(b.quantile(0.90))
            # The headline: today's single constant is 0.01 for every id.
            m[f"{prefix}/{self.task}/beta_log10_median"] = float(b.clamp(min=1e-30).log10().median())
            if w is not None:
                m[f"{prefix}/{self.task}/shrink_mean"] = float(w.mean())
        return m


def _strata(activity: torch.Tensor, keep: torch.Tensor, n_strata: int) -> torch.Tensor:
    """Bucket the kept ids into activity quantiles. Returns -1 for dropped ids.

    An id that is almost always available and one that appears twice a step have
    signal and noise of completely different magnitudes, so a single prior
    variance across the vocabulary fits neither. Quantiles rather than fixed
    edges because the activity distribution is extremely skewed.
    """
    out = torch.full_like(activity, -1, dtype=torch.int64)
    idx = keep.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return out
    order = torch.argsort(activity[idx])
    n = idx.numel()
    edges = torch.linspace(0, n, n_strata + 1, device=activity.device).round().to(torch.int64)
    for s in range(n_strata):
        lo, hi = int(edges[s]), int(edges[s + 1])
        if hi > lo:
            out[idx[order[lo:hi]]] = s
    return out


def fit_weights(
    mom: "StepMoments",
    task_names,
    cfg: LogitPrecisionConfig,
    *,
    base_beta: float = 0.01,
):
    """Turn the accumulated pushes into a per-id teacher weight for each task.

    WHAT THE OUTPUT IS. ``implied_beta[v]`` is the coefficient that should
    multiply the teacher-KL gradient at id ``v``, in exactly the units of the
    single constant the run uses today (``algorithm.opd.kl_loss_coef``, 0.01).
    So the whole mechanism is readable as "beta, but measured, and per id".

    WHY THE SHRINKAGE IS REPORTED SEPARATELY AND NOT FOLDED IN. Adam divides
    each coordinate by its own running magnitude, so multiplying coordinate v's
    gradient by a factor that is CONSTANT OVER TIME changes nothing at all:
    ``m`` and ``sqrt(v)`` scale together and the step is unchanged. The
    James-Stein factor ``k[v]`` is close to constant over time, so as a
    per-coordinate rescaling it is nearly a no-op -- what survives is (a) its
    time variation and (b) its effect through the GLOBAL gradient-norm clip,
    which is active on 79 percent of this workload's steps and does couple the
    coordinates. The RATIO between the two sources at one coordinate is a
    different matter: it changes the direction of the combination and is not
    normalised away. That ratio is ``implied_beta``, and it is the mechanism.

    Returns:
        ``{task_name: TaskFit}``.
    """
    cfg.validate()
    out = {}
    mom = mom.corrected()
    n = min(len(task_names), mom.n_tasks)
    for i in range(n):
        name = str(task_names[i])
        a, d = mom.a[i], mom.d[i]
        a2, ad, d2 = mom.a2[i], mom.ad[i], mom.d2[i]
        act, sigma2 = mom.act[i], mom.sig2[i]
        n_tok = float(mom.n_tokens[i])

        if n_tok < cfg.min_tokens or float(mom.n_rows[i]) < 2:
            out[name] = TaskFit(name, False, "too_few_tokens", n_tokens=n_tok)
            continue

        # WHICH sigma^2, AND WHY THE OBVIOUS CHOICE IS WRONG.
        #
        # The model in section 2 says a[v] = theta[v] + eps with variance
        # sigma^2[v], so sigma^2 is the ESTIMATOR'S sampling variance. The
        # permutation null is not that: it conditions on the advantages the step
        # actually drew and asks how much of the push is attributable to an
        # id-reward association. The two come apart exactly where this mechanism
        # is supposed to earn its keep.
        #
        # Concretely, sigma^2_perm[v] = spread[v] * sum_r A_r^2 / (N - 1), and
        # only spread[v] depends on the id -- so the id profile of the measured
        # coefficient is the ROW-TO-ROW VARIABILITY OF THE ID'S OCCURRENCE and
        # nothing else. Worse, as the advantages vanish sum_r A_r^2 goes to zero
        # together with a[v] itself, so the BLUE hands a push that is exactly
        # zero an infinite precision and gives the teacher none. The model reads
        # "not measured" as "measured to be zero". On webshop, where 62 percent
        # of tokens carry no advantage, that is the common case and it is
        # backwards.
        #
        # The across-step variance has none of that: it is the variance of the
        # push over the steps the EMA covers, so it includes the advantage
        # draw's own variability and cannot collapse when one step happens to
        # be flat. E[a2] = theta^2 + sigma^2 and E[abar^2] = theta^2 +
        # sigma^2/n, so (a2 - abar^2) n/(n-1) estimates sigma^2 and leaves
        # tau^2 = mean(a2) - mean(sigma^2) unchanged in form.
        #
        # It needs steps, though. Below min_eff_steps the permutation null is
        # all there is; the fit is then flagged `warming_up` and the apply path
        # declines to use it.
        n_eff_steps = mom.n_eff_steps()
        sig2_perm = sigma2
        warming = n_eff_steps < cfg.min_eff_steps
        if not warming:
            k_n = n_eff_steps / (n_eff_steps - 1.0)
            sigma2 = ((a2 - a * a) * k_n).clamp(min=0.0)

        keep = (act > cfg.min_activity) & torch.isfinite(a2) & torch.isfinite(d2) & torch.isfinite(sigma2)
        n_ids = int(keep.sum())
        if n_ids < 2 * cfg.n_strata:
            out[name] = TaskFit(name, False, "too_few_ids", n_tokens=n_tok, n_ids=n_ids)
            continue

        # Floor sigma^2 so an id seen in a single rollout does not claim
        # infinite precision. The floor is relative to this task's own scale.
        pos = sigma2[keep]
        floor = cfg.sigma2_floor_frac * float(pos[pos > 0].mean()) if bool((pos > 0).any()) else 0.0
        sigma2 = sigma2.clamp(min=max(floor, torch.finfo(sigma2.dtype).tiny))

        # WHICH IDS CAN IDENTIFY lambda, AND WHICH ONLY CONSUME IT.
        #
        # lambda is identified by regressing the teacher's push on the reward's,
        # so it can only be learned where the reward HAS a push above its own
        # permutation null. Pooling every id into one regression lets a large
        # noise-dominated region drive the estimated prior to zero and refuse
        # the whole task -- and that region is not hypothetical: this run
        # records 62 percent zero-advantage tokens on webshop, the task whose
        # teacher matters most.
        #
        # So the fit is identified on the strata that carry signal, and the
        # resulting lambda and varsigma are then applied to EVERY kept id. At
        # a silent coordinate that gives a large implied_beta, because sigma^2
        # is large there -- the teacher is the only source and is weighted
        # accordingly. That is the answer the model gives; suppressing those
        # coordinates instead would be the mechanism deciding, not the data.
        strat = _strata(act, keep, cfg.n_strata)
        signal = torch.zeros_like(keep)
        tau2_by_stratum = {}
        for st in range(cfg.n_strata):
            sel = strat == st
            if not bool(sel.any()):
                continue
            t2s = float(a2[sel].mean()) - float(sigma2[sel].mean())
            tau2_by_stratum[st] = t2s
            if t2s > 0.0:
                signal |= sel
        no_rl_frac = float((keep & ~signal).sum()) / max(n_ids, 1)
        n_sig = int(signal.sum())
        if n_sig < 2 * cfg.n_strata:
            out[name] = TaskFit(name, False, "rl_push_is_all_noise", tau2=0.0,
                                n_tokens=n_tok, n_ids=n_ids, no_rl_signal_frac=no_rl_frac)
            continue

        a2k, adk, d2k, s2k = a2[signal], ad[signal], d2[signal], sigma2[signal]
        # Both pushes sum to exactly zero over the vocabulary, so the second
        # moments are already central and nothing is subtracted here.
        mean_a2 = float(a2k.mean())
        mean_s2 = float(s2k.mean())
        mean_d2 = float(d2k.mean())
        mean_ad = float(adk.mean())

        tau2 = max(0.0, mean_a2 - mean_s2)
        if tau2 <= 0.0:
            out[name] = TaskFit(name, False, "rl_push_is_all_noise", tau2=0.0,
                                n_tokens=n_tok, n_ids=n_ids, no_rl_signal_frac=no_rl_frac)
            continue

        # Errors-in-variables: a carries measurement error, so the denominator
        # is the signal variance, not the observed variance.
        lam_raw = mean_ad / tau2
        varsigma2 = mean_d2 - lam_raw * lam_raw * tau2
        r2 = 0.0 if mean_d2 <= 0 else max(0.0, 1.0 - varsigma2 / mean_d2)

        # HOW MANY IDS ACTUALLY DECIDE lambda. The OPD push is extremely
        # concentrated -- a handful of format tokens carry most of it -- so a
        # regression nominally over thousands of ids can still be settled by
        # ten of them. n_eff = (sum |w|)^2 / sum w^2 over the per-id products
        # says how many, and the jackknife says what that costs. It is Kish's
        # ratio with absolute values in the numerator, because the products are
        # SIGNED and Kish's (sum w)^2 would let a positive and a negative id
        # cancel into an effective size of zero. Reporting lambda without either
        # would repeat the error the sign gates were faulted for: acting on a
        # statistic whose noise was never measured.
        prod = adk
        n_eff = float((prod.abs().sum() ** 2) / (prod * prod).sum().clamp(min=1e-300))
        nk = float(n_sig)
        s_ad, s_a2, s_s2 = float(prod.sum()), float(a2k.sum()), float(s2k.sum())
        den_j = (s_a2 - a2k) - (s_s2 - s2k)
        lam_j = torch.where(den_j.abs() > 0, (s_ad - prod) / den_j,
                            torch.full_like(den_j, float("nan")))
        ok_j = torch.isfinite(lam_j)
        n_ok = int(ok_j.sum())
        if n_ok > 2:
            lj = lam_j[ok_j]
            # (n-1)/n over the leave-one-out values that EXIST: using the full
            # id count here would scale the standard error by the share that
            # was dropped for a vanishing denominator.
            lam_se = float((((n_ok - 1.0) / n_ok) * ((lj - lj.mean()) ** 2).sum()).clamp(min=0.0).sqrt())
        else:
            lam_se = float("nan")

        lam = lam_raw if cfg.allow_negative_lambda else max(0.0, lam_raw)
        if not math.isfinite(lam) or not math.isfinite(varsigma2) or lam == 0.0:
            # Either the teacher's push carries no linear information about what
            # improves its own task, or it carries information with the WRONG
            # sign and the run has not opted into acting on that. Either way the
            # answer is to stop listening to it, which is beta = 0 -- not a
            # failure, so the fit stays valid and the metric shows lam_raw.
            out[name] = TaskFit(name, True, "teacher_uninformative", tau2=tau2, lam=0.0,
                                lam_raw=lam_raw, varsigma2=max(varsigma2, 0.0), r2=r2,
                                lam_se=lam_se, n_eff=n_eff, no_rl_signal_frac=no_rl_frac,
                                n_eff_steps=n_eff_steps, sigma2_is_across_step=not warming,
                                sigma2_perm_median=float(sig2_perm[keep].median()),
                                n_ids=n_ids, n_tokens=n_tok, sigma2=sigma2,
                                implied_beta=torch.zeros_like(a2), keep=keep,
                                shrink=torch.ones_like(a2))
            continue

        reason = "warming_up" if warming else "ok"
        floor = cfg.min_residual_frac * mean_d2
        if varsigma2 < floor:
            varsigma2 = floor
            reason = "warming_up" if warming else "residual_floored"

        prec_d = lam * lam / varsigma2                 # the teacher's precision
        implied_beta = lam * sigma2 / varsigma2        # omega_D / omega_R, per id
        s2_blue = 1.0 / (1.0 / sigma2 + prec_d)        # BLUE variance

        # The James-Stein factor, DIAGNOSTIC ONLY.
        #
        # It is deliberately not folded into implied_beta. The prior variance
        # behind it is estimated from the RL push alone, so at a coordinate
        # where the reward is silent it collapses to zero -- and multiplying by
        # it would switch the teacher off exactly where the teacher is the only
        # gradient there is. The prior is circular in that region: it asks the
        # one source with no information how much signal is present, while the
        # other source is saying it is not zero.
        #
        # An earlier version of this note also argued that applying it would be
        # nearly free because Adam normalises each coordinate over time. That
        # argument is withdrawn: Adam normalises per PARAMETER, and a logit
        # coordinate flows into a 2048-wide output row -- tied to the input
        # embedding on this model -- and into the trunk, where every id is mixed
        # before any normalisation. The reason it is not applied is the failure
        # mode above and nothing else.
        shrink = torch.ones_like(a2)
        for st, t2s in tau2_by_stratum.items():
            sel = strat == st
            shrink[sel] = 0.0 if t2s <= 0.0 else (t2s / (t2s + s2_blue[sel]))

        cap = cfg.max_beta_ratio * abs(base_beta)
        at_cap = implied_beta.abs() > cap
        implied_beta = implied_beta.clamp(min=-cap, max=cap)
        implied_beta = torch.where(keep, implied_beta, torch.full_like(a2, base_beta))
        shrink = torch.where(keep, shrink, torch.ones_like(a2))
        out[name] = TaskFit(name, True, reason, tau2=tau2, lam=lam, lam_raw=lam_raw,
                            varsigma2=varsigma2, r2=r2, lam_se=lam_se, n_eff=n_eff,
                            n_eff_steps=n_eff_steps, sigma2_is_across_step=not warming,
                            sigma2_perm_median=float(sig2_perm[keep].median()),
                            n_ids=n_ids, n_tokens=n_tok, no_rl_signal_frac=no_rl_frac,
                            sigma2=sigma2, implied_beta=implied_beta, shrink=shrink,
                            keep=keep, at_cap_frac=float((at_cap & keep).sum()) / max(n_ids, 1))
    return out


# ---------------------------------------------------------------------------
# Applying the weights
# ---------------------------------------------------------------------------


def reweighted_opd_surrogate(opd_scalar, student_topk_logprob, weight_topk):
    """A scalar whose backward is the OPD gradient reweighted per vocabulary id.

    WHY NOT JUST BUILD THE COTANGENT. Multiplying the (tokens x vocab) logit
    gradient by a per-id weight needs that tensor materialised -- at this
    micro-batch shape and vocabulary, 3.1 GB in float32, against roughly 4 GB of
    headroom. This gets the same result without it.

    THE STRUCTURE THAT MAKES IT POSSIBLE. The OPD loss reaches the logits only
    through the student's top-k log-probs ``s``, a (tokens, k) gather of the
    log-softmax. For any such loss,

        dL/dz_u = 1[u in A] (dL/ds_u)  -  pi[u] sum_j (dL/ds_j)

    -- sparse on the support, plus one rank-one background. So reweighting the k
    COEFFICIENTS and letting the log-softmax backward run as usual gives

        1[u in A] w_u (dL/ds_u)  -  pi[u] sum_j w_j (dL/ds_j)

    which applies ``w`` exactly on the support, applies the support's weighted
    average to the background, and -- because it is still the pullback of a
    (tokens, k) cotangent through the same log-softmax -- automatically sums to
    zero over the vocabulary. A cotangent that did not would push the overall
    logit level, which the policy ignores and the parameters do not.

    Weighting the background by an average rather than per id is the honest
    reading of what the background is: the normaliser's recoil, carrying no
    per-id preference of the teacher's to weight.

    ``dL/ds`` costs one backward over the KL head alone -- exp, two sums, the
    aggregation -- and does not enter the trunk. The returned scalar is then
    added to the policy loss in place of the OPD term, and the step's single
    backward carries it.

    Args:
        opd_scalar: exactly the scalar the OPD term would have contributed,
            coefficient and aggregation included.
        student_topk_logprob: the (bs, resp, k) tensor it flows through.
        weight_topk: (bs, resp, k) per-id weight, gathered at the support ids
            and already divided by the base coefficient the scalar carries.

    Returns:
        A scalar with the reweighted gradient, and ``weight_topk == 1``
        reproduces ``opd_scalar``'s own gradient exactly.
    """
    (gcoef,) = torch.autograd.grad(
        opd_scalar, student_topk_logprob, retain_graph=True, create_graph=False
    )
    return (student_topk_logprob * (weight_topk.to(gcoef.dtype) * gcoef).detach()).sum()


def beta_ratio_at(fit: "TaskFit", topk_ids: torch.Tensor, base_beta: float) -> torch.Tensor:
    """``implied_beta / base_beta`` gathered at the support ids, or ones.

    The surrogate multiplies a scalar that ALREADY carries ``base_beta``, so the
    weight it needs is the ratio rather than the coefficient. An invalid fit
    gives ones, which is the run's present behaviour -- never zero, because a
    measurement that failed must not silently switch the teacher off.
    """
    if fit is None or not fit.valid or fit.implied_beta is None:
        return torch.ones_like(topk_ids, dtype=torch.float32)
    base = abs(float(base_beta))
    if base <= 0.0:
        return torch.ones_like(topk_ids, dtype=torch.float32)
    ratio = (fit.implied_beta / base).to(torch.float32)
    return ratio.to(topk_ids.device).gather(0, topk_ids.reshape(-1).to(torch.int64)).view_as(topk_ids)


def beta_ratio_matrix(fits, task_names, vocab, base_beta, *, device=None) -> torch.Tensor:
    """``implied_beta / base_beta`` as (n_tasks, vocab), ones where unmeasured.

    Ones rather than zeros in every fallback -- a task whose fit was refused,
    an id below the activity floor, a base coefficient of zero -- so a failed
    measurement leaves the run at the coefficient it already uses instead of
    silently switching the teacher off.
    """
    names = list(task_names)
    out = torch.ones(len(names), int(vocab), device=device, dtype=torch.float32)
    base = abs(float(base_beta))
    if base <= 0.0:
        return out
    for i, name in enumerate(names):
        fit = fits.get(str(name)) if fits else None
        if fit is None or not fit.valid or fit.implied_beta is None:
            continue
        if not fit.sigma2_is_across_step:
            # A warm-up fit rests on the permutation null, which is the wrong
            # variance and collapses where the reward is flat. Leave the run at
            # its own coefficient until enough steps have accumulated.
            continue
        out[i] = (fit.implied_beta / base).to(device=out.device, dtype=out.dtype)
    return out
