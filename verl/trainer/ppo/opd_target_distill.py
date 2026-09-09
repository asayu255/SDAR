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

"""MOPD v3: build the distillation target from the RL direction; distil to it, and only that.

Design: ``docs/opd_target_distillation_design.md``. The loss is

    L = beta * KL( p_theta || sg(q*) ),        q* = softmax( log q + c )

with NO GRPO term beside it. The reward enters only through c, which is the
token's own policy-gradient direction weighted by how hard the teacher is
pushing there, by whether the teacher backs the sampled token, and by a
per-(task, role) cross-task coefficient:

    c_{i,t} = eta * alpha_{i,c(t)} * f_{i,t} * e_{i,t} * r_{i,t}      (re-centred)

WHY A TARGET AND NOT A COEFFICIENT. ``cross_teacher_target``'s audit settled
this: a positive scalar on KL(p || q) cannot inject anything, because the
minimiser is p = q whatever the scalar is. Every coefficient arm this repo has
run -- the self gate, the cross gate v1/v2, the static per-task b -- can only
change how fast the student is pulled to the teacher, never where it lands.
Moving the target is the only way to put the reward's direction into the
distillation channel.

WHAT IS ACTUALLY INJECTED. Not c. For q* = softmax(log q + c),

    -grad_z [ beta KL(p_theta || q*) ] |_{p_0}  =  d  +  beta * F_{p0} c,
    F_p = diag(p) - p p^T,   i.e.  beta * p .* (c - E_p[c])

(checked against autograd to 5e-19). Dropping the mean-subtraction is a 24%
error, so every metric and the cross-task constraint below are written on
``beta F c`` and not on c. The same identity is why c is re-centred on the
support: r = c_t (e_a - p) sums to c_t * tail_mass over the support, not zero.

WHAT THIS IS NOT.

  * NOT equal to pure OPD+GRPO at eta = 1. That equality needs the 1/p_0 form
    of the exponent, no clamp, and no support approximation. The 1/p_0 form
    reaches exponents of [-37, +22] on measured data, and clamping it costs
    70-91% of the identity it exists to provide -- so this uses the bounded
    log-space tilt and gives the equality up. The control is the control arm.
  * NOT a fixed point one can name in advance. q* depends on the student, the
    rollouts and the advantages, so it is self-consistent, not specified.
  * NOT a trust region. KL(p || q*) is a distance to a target; PPO's clip bounds
    the ratio to the OLD POLICY. Different objects.

WHAT DOES CARRY THROUGH: the clip itself. r is built from
``policy_loss_gradient_coef``, which is zero inside a clipped branch, so there
c = 0 and q* = q exactly. Rebuilding the target every forward keeps that step's
clip branches in the target.

WHAT IS HONEST ABOUT THE SCOPE. Folding the loss into one term removes the
double count, and nothing else. The reward is still there, the OPD/RL strength
trade-off is still a design decision -- it has moved from the loss coefficient
to the target construction (eta, f, e). No new reward information and no new
teacher knowledge is created by the change of variables.
"""

from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from verl.trainer.ppo.cross_teacher_target import normalized_weight
from verl.trainer.ppo.sign_weights import ROLE_NAMES

__all__ = [
    "needs_policy_gradient_inputs",
    "TargetDistillConfig",
    "TargetDistillController",
    "TargetDistillRefs",
    "TargetDistillStats",
    "rl_support_direction",
    "opd_strength_ratio",
    "rlsd_support_factor",
    "build_target",
    "solve_alpha",
    "fisher_apply",
    "N_ROLES",
]

def needs_policy_gradient_inputs(cfg_map) -> bool:
    """Does this arm need the reward's inputs even though pg_loss_coef is 0?

    Yes, and it is the whole point: v3 takes the GRPO TERM away but keeps the
    reward, routing it through the target instead. The actor has three fast
    paths that key off ``pg_loss_coef == 0`` meaning "no policy-gradient signal
    at all" -- it drops ``advantages`` and ``old_log_probs`` from select_keys,
    it skips the full-vocabulary log_prob, and it computes
    ``policy_loss_gradient_coef`` only inside the pg branch. With all three
    taken, r = 0 and e = 1, so c = 0 and q* = q: the arm runs as pure OPD while
    still emitting every metric. That is silent, so the condition lives here,
    once, and the three sites call it.
    """
    return bool(cfg_map) and bool(dict(cfg_map).get("enable", False))


N_ROLES = len(ROLE_NAMES)
ROLE_ID = {name: rid for rid, name in ROLE_NAMES.items()}

# Why a reference (i, c) cannot be used this step. 0 = usable.
UNUSABLE_NONE = 0
UNUSABLE_NEVER_SEEN = 1
UNUSABLE_FEW_TOKENS = 2
UNUSABLE_STALE = 3
UNUSABLE_NONFINITE = 4
UNUSABLE_ZERO_NORM = 5
UNUSABLE_NAMES = {
    UNUSABLE_NONE: "ok", UNUSABLE_NEVER_SEEN: "never_seen", UNUSABLE_FEW_TOKENS: "few_tokens",
    UNUSABLE_STALE: "stale", UNUSABLE_NONFINITE: "nonfinite", UNUSABLE_ZERO_NORM: "zero_norm",
}
# Why alpha fell back to 1 rather than being solved. 0 = solved.
FALLBACK_NONE = 0
FALLBACK_NO_RECEIVER = 1
FALLBACK_NONFINITE = 2
FALLBACK_UNSOLVED = 3
FALLBACK_NAMES = {FALLBACK_NONE: "solved", FALLBACK_NO_RECEIVER: "no_receiver",
                  FALLBACK_NONFINITE: "nonfinite", FALLBACK_UNSOLVED: "unsolved"}


@dataclass
class TargetDistillConfig:
    enable: bool = False
    # Scale on the tilt. NOT a control point: see the module docstring. An
    # experimental condition, fixed at 1.0 for the first arm.
    eta: float = 1.0
    # RLSD token-weight clip, the same epsilon_w as rlsd_utils.
    epsilon_w: float = 0.2
    # Bound on |c| in nats. The identity is already given up (see docstring), so
    # this is chosen to bound how far the target may leave the teacher, not to
    # preserve a gradient equality.
    clamp: float = 2.0
    # Cross-task integration. Off => alpha == 1 everywhere, which is arm B.
    integrate: bool = False
    # EMA for sbar (the OPD strength scale), the references and the Gram terms.
    ema_decay: float = 0.8
    # A reference (i, c) is usable after this many control tokens in the window
    # and while it is no more than max_staleness steps old.
    window_steps: int = 8
    min_tokens: int = 64
    max_staleness: int = 2
    delta: float = 1.0e-30
    roles: tuple = ("format", "env_action")
    solver_iters: int = 500
    solver_tol: float = 1.0e-12

    @classmethod
    def from_mapping(cls, m) -> TargetDistillConfig:
        if not m:
            return cls()
        m = dict(m)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        bad = sorted(set(m) - allowed)
        if bad:
            raise ValueError(f"target_distill: unknown keys {bad}; allowed {sorted(allowed)}")
        kw = {}
        for k, v in m.items():
            if k in ("enable", "integrate"):
                kw[k] = bool(v)
            elif k == "roles":
                kw[k] = tuple(str(x) for x in (list(v) if not isinstance(v, str) else v.split(",")))
            elif k in ("window_steps", "min_tokens", "max_staleness", "solver_iters"):
                kw[k] = int(v)
            else:
                kw[k] = float(v)
        return cls(**kw)

    def validate(self) -> None:
        if self.eta < 0.0:
            raise ValueError(f"target_distill.eta must be >= 0, got {self.eta}")
        if not (0.0 <= self.epsilon_w < 1.0):
            raise ValueError(f"target_distill.epsilon_w must be in [0, 1), got {self.epsilon_w}")
        if self.clamp <= 0.0:
            raise ValueError(f"target_distill.clamp must be > 0, got {self.clamp}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"target_distill.ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.window_steps < 1 or self.min_tokens < 1 or self.max_staleness < 0:
            raise ValueError("target_distill.window_steps/min_tokens must be >= 1, max_staleness >= 0")
        if self.delta <= 0.0:
            raise ValueError("target_distill.delta must be > 0")
        unknown = [r for r in self.roles if r not in ROLE_ID]
        if unknown:
            raise ValueError(f"target_distill.roles has unknown roles {unknown}; known {sorted(ROLE_ID)}")
        if self.solver_iters < 1 or self.solver_tol <= 0.0:
            raise ValueError("target_distill.solver_iters must be >= 1 and solver_tol > 0")

    def control_role_mask(self) -> torch.Tensor:
        m = torch.zeros(N_ROLES, dtype=torch.bool)
        for r in self.roles:
            m[ROLE_ID[r]] = True
        return m


# --------------------------------------------------------------------------- #
# The three per-token pieces


def rl_support_direction(*, p_s: torch.Tensor, sampled_onehot: torch.Tensor,
                         pg_grad_coef: torch.Tensor | None) -> torch.Tensor:
    """``r = -dL_pg/dz`` on the top-k support: ``c_t (e_a - p)``, descent convention.

    ``pg_grad_coef`` is ``dL_pg/dlog p`` WITH the clip branches
    (:func:`core_algos.policy_loss_gradient_coef`), never the advantage: -A is
    the coefficient only at ratio 1 and outside the clip, which holds for the
    first mini-batch of six. Inside a clipped branch it is exactly zero, so r is
    zero there and the target is left at the teacher -- which is how the clip
    reaches the target at all.
    """
    if pg_grad_coef is None:
        return torch.zeros_like(p_s)
    scale = -pg_grad_coef.detach().to(p_s.dtype)            # descent convention
    return scale.unsqueeze(-1) * (sampled_onehot.to(p_s.dtype) - p_s)


def opd_strength_ratio(*, d_norm: torch.Tensor, sbar: torch.Tensor,
                       delta: float = 1e-30) -> torch.Tensor:
    """``f = 2 s / (s + sbar)`` in [0, 2): 1 at the running mean, 0 where OPD is silent.

    A ratio rather than a raw norm so it is scale-free in beta and comparable
    across tasks and roles; bounded above by 2 so one loud position cannot
    dominate. ``sbar`` is the previous steps' EMA of ``s`` for this (task, role).
    """
    s = d_norm.detach()
    return 2.0 * s / (s + sbar + float(delta))


def rlsd_support_factor(*, teacher_logprob_a: torch.Tensor, student_logprob_a: torch.Tensor,
                        advantages: torch.Tensor | None, epsilon_w: float) -> torch.Tensor:
    """``e = clip(exp(sign(A) * (log q(a) - log p(a))), 1-eps, 1+eps)``.

    The same expression as :func:`rlsd_utils.compute_rlsd_token_advantage`'s
    ``w_t``; only the injection point differs -- there it multiplies the
    advantage, here the logit-space RL direction. On a FAILED trajectory
    (A < 0) at a token the teacher backs (q(a) > p(a)), the exponent is negative
    and e < 1, so the penalty on that token is softened. That is the direct
    answer to what the pushback review §9.3 measured: GRPO gives the correct
    intermediate steps of a failed rollout the same negative advantage as the
    rest. It does NOT establish that the teacher is right there.
    """
    if advantages is None:
        return torch.ones_like(teacher_logprob_a)
    d = (teacher_logprob_a - student_logprob_a).detach()
    sign_a = torch.sign(advantages.detach()).to(d.dtype)
    return torch.exp(sign_a * d).clamp(1.0 - float(epsilon_w), 1.0 + float(epsilon_w))


def fisher_apply(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``F_p c = p .* (c - E_p[c])`` -- the logit direction a tilt c actually injects.

    ``-grad_z[beta KL(p||softmax(log q + c))]|_p = d + beta F_p c``. The
    mean-subtraction is not decoration: without it the direction is wrong by 24%
    on a realistic support, and the whole point of the arm is which direction is
    injected.
    """
    return p * (c - (p * c).sum(dim=-1, keepdim=True))


def build_target(
    *,
    student_topk_logprob: torch.Tensor,
    teacher_topk_logprob: torch.Tensor,
    topk_ids: torch.Tensor,
    response_ids: torch.Tensor,
    pg_grad_coef: torch.Tensor | None,
    advantages: torch.Tensor | None,
    teacher_kl_base: torch.Tensor,
    task_ids: torch.Tensor,
    roles: torch.Tensor,
    refs: "TargetDistillRefs",
    cfg: TargetDistillConfig,
) -> dict:
    """The tilted target and everything the step's statistics need. All detached.

    Shapes: log-probs and ids ``(bs, T, k)``; ``response_ids``, ``roles``,
    ``pg_grad_coef``, ``advantages``, ``opd_push_sq`` ``(bs, T)``; ``task_ids`` ``(bs,)``.
    ``teacher_kl_base`` is the UNTILTED per-token KL to the on-task teacher; the
    OPD direction ``g_opd = p (D - (log p - log q))`` and its norm are derived
    from it here. Taking ||d|| from ``opd_task_diag`` instead would be circular:
    that module computes it from the teacher KL, which is the quantity this
    function is about to rewrite.

    Returns ``target_logprob`` (what the loss distils to) plus ``r``, ``c``,
    ``inject`` (= F_p c), ``f``, ``e``, ``alpha_t``, ``tv``, ``clamped``,
    ``recenter_residual`` and the masks the accumulator needs.
    """
    with torch.no_grad():
        dt = torch.promote_types(student_topk_logprob.dtype, torch.float32)
        lp_s = student_topk_logprob.detach().to(dt)
        lp_t = teacher_topk_logprob.detach().to(dt)
        p_s = lp_s.exp()
        bs, T, k = lp_s.shape
        dev = lp_s.device
        nT, nR = refs.alpha.shape

        hit = (topk_ids == response_ids.unsqueeze(-1))
        in_support = hit.any(dim=-1)
        hit_f = hit.to(dt)

        # --- r, the clipped PG direction on the support ---------------------
        r = rl_support_direction(p_s=p_s, sampled_onehot=hit_f, pg_grad_coef=pg_grad_coef)
        # A token whose sampled id is outside the support has no e_a there; its
        # -p-only vector is not the PG direction, so it is excluded rather than
        # injected as a bias.
        r = r * in_support.unsqueeze(-1).to(dt)

        # --- d and its norm, from the UNTILTED teacher ----------------------
        # g_opd(v) = p(v) (D - f(v)), f = log p - log q. Same expression as
        # opd_task_diag / opd_logit_push at coefficient 1; beta is left out
        # because f below is a RATIO and beta cancels in it.
        D = teacher_kl_base.detach().to(dt).unsqueeze(-1)
        g_opd = p_s * (D - (lp_s - lp_t))
        d_norm = (g_opd * g_opd).sum(dim=-1).clamp(min=0.0).sqrt()

        # --- f, the OPD strength ratio --------------------------------------
        tid = task_ids.reshape(-1)
        tid = tid.round().to(torch.long) if tid.is_floating_point() else tid.to(torch.long)
        tid_ok = (tid >= 0) & (tid < nT)
        tid_c = tid.clamp(min=0, max=max(nT - 1, 0))
        rol = roles.to(torch.long).clamp(min=0, max=nR - 1)
        sbar_t = refs.sbar.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]
        f = opd_strength_ratio(d_norm=d_norm, sbar=sbar_t, delta=cfg.delta)

        # --- e, the RLSD support factor -------------------------------------
        lp_t_a = (hit_f * lp_t).sum(dim=-1)
        lp_s_a = (hit_f * lp_s).sum(dim=-1)
        e = rlsd_support_factor(teacher_logprob_a=lp_t_a, student_logprob_a=lp_s_a,
                                advantages=advantages, epsilon_w=cfg.epsilon_w)
        e = torch.where(in_support, e, torch.ones_like(e))

        # --- alpha, per (sender task, role), fixed for the step --------------
        alpha_t = refs.alpha.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]
        ctrl = refs.control_roles.to(dev)[rol]
        # Off a controlled role, or on a padding row, the target is the teacher.
        live = ctrl & tid_ok.view(bs, 1) & in_support
        alpha_t = torch.where(live, alpha_t, torch.zeros_like(alpha_t))

        # --- c, re-centred on the support ------------------------------------
        c_raw = float(cfg.eta) * (alpha_t * f * e).unsqueeze(-1) * r
        # r sums to c_t * tail_mass over the support, not to zero; F_p c drops
        # any constant, but the CLAMP does not, so re-centre before clamping.
        recenter = c_raw.mean(dim=-1, keepdim=True)
        c = c_raw - recenter

        built = normalized_weight(c=c, p_on=lp_t.exp(), clamp=float(cfg.clamp))
        target_logprob = lp_t + built["log_w"].to(lp_t.dtype)

        # --- what is actually injected ---------------------------------------
        c_eff = built["c_eff"].to(dt)
        inject = fisher_apply(p_s, c_eff)          # F_p c ; beta is applied by the loss
        return {
            "target_logprob": target_logprob,
            "r": r, "c": c, "c_eff": c_eff, "inject": inject,
            "f": f, "e": e, "alpha_t": alpha_t, "d_norm": d_norm,
            "in_support": in_support.to(dt), "live": live.to(dt),
            "tv": built["moved"].to(dt),
            "clamped": built["clamped"].to(dt).mean(dim=-1),
            "recenter_residual": recenter.squeeze(-1).abs() * k,
            "rtilde": (f * e).unsqueeze(-1) * r,   # the candidate direction, alpha-free
        }


# --------------------------------------------------------------------------- #
# Per-step accumulators

_TOK_COLS = (
    "n_tok", "n_live", "d_norm_sum", "d_sq_sum",
    "f_sum", "e_sum", "fe_sum", "e_sum_adv_neg", "n_adv_neg", "e_sum_adv_pos", "n_adv_pos",
    "inject_sq_sum", "r_sq_sum", "rtilde_sq_sum", "inject_dot_r_sum", "inject_norm_sum",
    "r_norm_sum", "tv_sum", "clamped_sum", "recenter_sum", "c_abs_sum", "alpha_sum",
)
_TOK_IDX = {n: i for i, n in enumerate(_TOK_COLS)}


class TargetDistillStats:
    """Sums for one update, on the device, reduced once. Built on the config alone
    so every rank runs the same collectives whatever its micro-batch holds."""

    def __init__(self, n_tasks: int, vocab_size: int, device):
        self.n_tasks, self.V = int(n_tasks), int(vocab_size)
        nT, nR = self.n_tasks, N_ROLES
        self.tok = torch.zeros(nT, nR, len(_TOK_COLS), dtype=torch.float64, device=device)
        # v = E[r] and g = E[rtilde] per (task, role), over the vocabulary.
        self.v_sum = torch.zeros(nT, nR, self.V, dtype=torch.float32, device=device)
        self.g_sum = torch.zeros(nT, nR, self.V, dtype=torch.float32, device=device)
        # The same two after F_p, which is what the constraint is written on.
        self.fv_sum = torch.zeros(nT, nR, self.V, dtype=torch.float32, device=device)
        self.fg_sum = torch.zeros(nT, nR, self.V, dtype=torch.float32, device=device)

    def update(self, *, built: dict, task_ids: torch.Tensor, roles: torch.Tensor,
               response_mask: torch.Tensor, topk_ids: torch.Tensor,
               advantages: torch.Tensor | None, row_basis: torch.Tensor | None,
               p_s: torch.Tensor) -> None:
        with torch.no_grad():
            nT, nR = self.n_tasks, N_ROLES
            bs, T = response_mask.shape
            mask = response_mask.to(torch.float32)
            if row_basis is not None:
                # A duplicated padding row carries basis 0 and is not evidence.
                mask = mask * (row_basis.detach().reshape(-1).to(torch.float32) > 0).to(torch.float32).unsqueeze(-1)
            tid = task_ids.reshape(-1)
            tid = tid.round().to(torch.long) if tid.is_floating_point() else tid.to(torch.long)
            tid_ok = (tid >= 0) & (tid < nT)
            mask = mask * tid_ok.to(torch.float32).unsqueeze(-1)
            tid_c = tid.clamp(min=0, max=max(nT - 1, 0))
            rol = roles.to(torch.long).clamp(min=0, max=nR - 1)
            cell = (tid_c.unsqueeze(-1) * nR + rol)
            ncell = nT * nR
            live = built["live"].to(torch.float32) * mask

            r = built["r"].to(torch.float32)
            rt = built["rtilde"].to(torch.float32)
            inj = built["inject"].to(torch.float32)
            r_sq = (r * r).sum(-1)
            rt_sq = (rt * rt).sum(-1)
            inj_sq = (inj * inj).sum(-1)

            def add(name, val):
                self.tok.view(ncell, -1)[:, _TOK_IDX[name]].index_add_(
                    0, cell.reshape(-1), (val * mask).reshape(-1).to(torch.float64))
            one = torch.ones_like(mask)
            add("n_tok", one)
            add("n_live", built["live"].to(torch.float32))
            add("d_norm_sum", built["d_norm"].to(torch.float32))
            add("d_sq_sum", built["d_norm"].to(torch.float32) ** 2)
            add("f_sum", built["f"].to(torch.float32))
            add("e_sum", built["e"].to(torch.float32))
            add("fe_sum", (built["f"] * built["e"]).to(torch.float32))
            add("inject_sq_sum", inj_sq)
            add("r_sq_sum", r_sq)
            add("rtilde_sq_sum", rt_sq)
            add("inject_dot_r_sum", (inj * r).sum(-1))
            add("inject_norm_sum", inj_sq.clamp(min=0).sqrt())
            add("r_norm_sum", r_sq.clamp(min=0).sqrt())
            add("tv_sum", built["tv"].to(torch.float32))
            add("clamped_sum", built["clamped"].to(torch.float32))
            add("recenter_sum", built["recenter_residual"].to(torch.float32))
            add("c_abs_sum", built["c_eff"].abs().sum(-1).to(torch.float32))
            add("alpha_sum", built["alpha_t"].to(torch.float32))
            if advantages is not None:
                neg = (advantages.detach() < 0).to(torch.float32)
                pos = (advantages.detach() > 0).to(torch.float32)
                add("e_sum_adv_neg", built["e"].to(torch.float32) * neg)
                add("n_adv_neg", neg)
                add("e_sum_adv_pos", built["e"].to(torch.float32) * pos)
                add("n_adv_pos", pos)

            # --- the vocabulary-space means, on LIVE tokens only -------------
            ids = topk_ids.to(torch.long).clamp(min=0, max=self.V - 1)
            flat = (cell.unsqueeze(-1) * self.V + ids).reshape(-1)
            lw = live.unsqueeze(-1)
            fv = fisher_apply(p_s.to(torch.float32), r)
            fg = fisher_apply(p_s.to(torch.float32), rt)
            for buf, val in ((self.v_sum, r), (self.g_sum, rt), (self.fv_sum, fv), (self.fg_sum, fg)):
                buf.view(-1).index_add_(0, flat, (val * lw).reshape(-1).to(torch.float32))

    def reduced(self) -> dict:
        out = {}
        for name in ("tok", "v_sum", "g_sum", "fv_sum", "fg_sum"):
            buf = getattr(self, name)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                buf = buf.clone()
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            out[name] = buf.detach().cpu()
        return out


# --------------------------------------------------------------------------- #
# The per-role constrained integration


def solve_alpha(K: np.ndarray, Gvg: np.ndarray, valid_recv: np.ndarray, *,
                delta: float = 1e-30, iters: int = 500, tol: float = 1e-12) -> dict:
    """min 1/2 sum_j K_j (1 - a_j)^2  s.t.  sum_j a_j Gvg[i, j] >= 0 for valid i, 0 <= a <= 1.

    ``Gvg[i, j] = (F v_i)^T (F g_j)`` -- the constraint is written on the
    directions the distillation actually injects, not on the raw reference inner
    products, because those differ by F_p (design §0).

    SOLVED EXACTLY BY ACTIVE-SET ENUMERATION, not by a penalty. The problem is a
    strictly convex QP whose feasible set is a polytope with at most n_task + 2
    n_task faces, so for the three or four tasks this ever runs on, every
    candidate KKT point can be enumerated: pick a subset of constraints to hold
    with equality, solve the resulting linear system, keep it if it is feasible,
    and take the best. A penalty method was tried first and returned points 60%
    worse than a coarse grid -- it drove alpha down to satisfy the constraint and
    had no way back up.

    a = 1 is what the objective wants and is returned whenever it is feasible.
    Returns ``alpha``, the constraint value at the solution and at a = 1, and a
    fallback code: a caller that gets anything but FALLBACK_NONE must use
    alpha = 1, NOT 0 -- with distillation-only, alpha = 0 removes the RL signal
    entirely (design §2.3).
    """
    n = int(np.asarray(K).reshape(-1).shape[0])
    K = np.asarray(K, dtype=np.float64).reshape(n)
    G = np.asarray(Gvg, dtype=np.float64).reshape(n, n)
    valid = np.asarray(valid_recv, dtype=bool).reshape(n)
    one = np.ones(n)
    if not (np.isfinite(K).all() and np.isfinite(G).all()):
        return {"alpha": one, "slack": G @ one, "slack_at_one": G @ one,
                "fallback": FALLBACK_NONFINITE, "converged": False}
    if not valid.any():
        return {"alpha": one, "slack": G @ one, "slack_at_one": G @ one,
                "fallback": FALLBACK_NO_RECEIVER, "converged": True}

    s1 = G @ one
    if (s1[valid] >= 0.0).all():
        return {"alpha": one, "slack": s1, "slack_at_one": s1,
                "fallback": FALLBACK_NONE, "converged": True}

    # A sender with no observed signal has K_j = 0 and the objective would not
    # care where its alpha lands. Ridge it so the tie is broken toward 1, which
    # is the direction that keeps the RL signal.
    scale = max(float(np.abs(K).max()), 1.0)
    Kp = np.maximum(K, 0.0) + 1e-9 * scale

    # rows of  A a >= b :  the receivers' conditions, then 0 <= a <= 1
    rows = [G[i] for i in np.nonzero(valid)[0]] + \
           [np.eye(n)[j] for j in range(n)] + [-np.eye(n)[j] for j in range(n)]
    rhs = [0.0] * int(valid.sum()) + [0.0] * n + [-1.0] * n
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    m = A.shape[0]

    def feasible(a):
        return bool(np.isfinite(a).all() and (A @ a >= b - 1e-9 * max(scale, 1.0)).all())

    def obj(a):
        return 0.5 * float((Kp * (1.0 - a) ** 2).sum())

    best_a, best_f = None, np.inf
    invK = 1.0 / Kp
    from itertools import combinations
    for size in range(0, n + 1):
        for S in combinations(range(m), size):
            if size == 0:
                a = one.copy()
            else:
                As, bs = A[list(S)], b[list(S)]
                # a = 1 + K^-1 As^T lam ,  As a = bs
                M = As @ (invK[:, None] * As.T)
                try:
                    lam = np.linalg.solve(M, bs - As @ one)
                except np.linalg.LinAlgError:
                    continue
                a = one + invK * (As.T @ lam)
            if not feasible(a):
                continue
            f = obj(a)
            if f < best_f - 1e-15:
                best_a, best_f = np.clip(a, 0.0, 1.0), f
    if best_a is None:
        return {"alpha": one, "slack": s1, "slack_at_one": s1,
                "fallback": FALLBACK_UNSOLVED, "converged": False}
    return {"alpha": best_a, "slack": G @ best_a, "slack_at_one": s1,
            "fallback": FALLBACK_NONE, "converged": True}


# --------------------------------------------------------------------------- #
# What the loss reads, and the controller that produces it


@dataclass
class TargetDistillRefs:
    """Previous-step quantities in THIS batch's task order, on the device."""
    alpha: torch.Tensor          # (nT, nR) cross-task coefficient, 1 where not integrated
    sbar: torch.Tensor           # (nT, nR) EMA of ||d||, the scale f is relative to
    control_roles: torch.Tensor  # (nR,) bool


@dataclass
class _RefState:
    fv: torch.Tensor | None = None      # (V,) EMA of E[F_p r]
    fg: torch.Tensor | None = None      # (V,) EMA of E[F_p rtilde]
    sbar: float = 0.0
    k: float = 0.0                      # EMA of E||rtilde||^2
    n_obs: int = 0
    last_step: int = -1
    tokens: collections.deque = None


class TargetDistillController:
    """Holds sbar, the references and alpha; survives a checkpoint.

    Same lifecycle as the cross gate's: the values the loss reads are the
    previous step's, the step's own reduced sums produce the next ones, and the
    solve runs on every rank from the all-reduced table so no broadcast is
    needed (deterministic arithmetic, same iterate order).
    """

    def __init__(self, cfg: TargetDistillConfig, vocab_size: int, task_names=()):
        cfg.validate()
        self.cfg = cfg
        self.V = int(vocab_size)
        self.step = 0
        self.refs: dict = {}                # (task, role) -> _RefState
        self.alpha: dict = {}               # (task, role) -> float
        self.usable: dict = {}              # (task, role) -> (ok, reason, n_steps)
        self.last_solve: dict = {}
        self.tasks: list = []
        self._ensure(task_names)

    def _ensure(self, names) -> None:
        for n in names or ():
            n = str(n)
            if n not in self.tasks:
                self.tasks.append(n)
                self.tasks.sort()

    def _ref(self, task, role) -> _RefState:
        key = (str(task), int(role))
        st = self.refs.get(key)
        if st is None:
            st = _RefState(tokens=collections.deque())
            self.refs[key] = st
        return st

    def refs_to_device(self, names, device, dtype=torch.float32) -> TargetDistillRefs:
        self._ensure(names)
        names = [str(n) for n in names]
        nT, nR = len(names), N_ROLES
        alpha = torch.ones(nT, nR, dtype=dtype)
        sbar = torch.zeros(nT, nR, dtype=dtype)
        for ti, n in enumerate(names):
            for c in range(nR):
                alpha[ti, c] = float(self.alpha.get((n, c), 1.0))
                sbar[ti, c] = float(self._ref(n, c).sbar)
        ctrl = self.cfg.control_role_mask()
        return TargetDistillRefs(alpha=alpha.to(device), sbar=sbar.to(device),
                                 control_roles=ctrl.to(device))

    def update(self, names, reduced: dict) -> dict:
        """Fold this step's reduced sums in; solve alpha for the NEXT step."""
        cfg = self.cfg
        names = [str(n) for n in names]
        self._ensure(names)
        nT, nR = len(names), N_ROLES
        tok, fv_s, fg_s = reduced["tok"], reduced["fv_sum"], reduced["fg_sum"]
        metrics: dict = {}
        self.step += 1
        step = self.step
        dec = cfg.ema_decay

        def g(ti, c, col):
            return float(tok[ti, c, _TOK_IDX[col]])

        # 1. sbar, the references and K. Missing is not zero: a (task, role) with
        #    no tokens this step holds its state and ages.
        for ti, n in enumerate(names):
            for c in range(nR):
                st = self._ref(n, c)
                n_tok, n_live = g(ti, c, "n_tok"), g(ti, c, "n_live")
                st.tokens.append(int(n_live))
                while len(st.tokens) > cfg.window_steps:
                    st.tokens.popleft()
                if n_tok > 0:
                    sb = g(ti, c, "d_norm_sum") / n_tok
                    st.sbar = sb if (st.n_obs == 0 or dec == 0.0) else dec * st.sbar + (1 - dec) * sb
                if n_live > 0:
                    mfv = (fv_s[ti, c] / n_live).to(torch.float32)
                    mfg = (fg_s[ti, c] / n_live).to(torch.float32)
                    kk = g(ti, c, "rtilde_sq_sum") / n_live
                    if st.n_obs == 0 or dec == 0.0 or st.fv is None:
                        st.fv, st.fg, st.k = mfv.clone(), mfg.clone(), kk
                    else:
                        st.fv = dec * st.fv + (1 - dec) * mfv
                        st.fg = dec * st.fg + (1 - dec) * mfg
                        st.k = dec * st.k + (1 - dec) * kk
                    st.n_obs += 1
                    st.last_step = step

        # 2. usability per (i, c)
        for n in names:
            for c in range(nR):
                st = self._ref(n, c)
                reason = UNUSABLE_NONE
                if st.n_obs == 0 or st.fv is None:
                    reason = UNUSABLE_NEVER_SEEN
                elif not (torch.isfinite(st.fv).all() and torch.isfinite(st.fg).all()):
                    reason = UNUSABLE_NONFINITE
                elif float(st.fv.double().norm()) <= 0.0:
                    reason = UNUSABLE_ZERO_NORM
                elif step - st.last_step > cfg.max_staleness:
                    reason = UNUSABLE_STALE
                elif sum(st.tokens) < cfg.min_tokens:
                    reason = UNUSABLE_FEW_TOKENS
                prev = self.usable.get((n, c), (False, UNUSABLE_NEVER_SEEN, 0))
                self.usable[(n, c)] = (reason == UNUSABLE_NONE, reason,
                                       0 if reason == UNUSABLE_NONE else prev[2] + 1)

        # 3. solve alpha per controlled role
        ctrl = cfg.control_role_mask()
        for c in range(nR):
            rn = ROLE_NAMES[c]
            if not bool(ctrl[c]) or not cfg.integrate:
                for n in names:
                    self.alpha[(n, c)] = 1.0
                if bool(ctrl[c]):
                    metrics[f"actor/target/alpha_fallback/{rn}"] = float(FALLBACK_NONE)
                continue
            K = np.array([self._ref(n, c).k for n in names], dtype=np.float64)
            G = np.zeros((nT, nT))
            for ti, i in enumerate(names):
                a = self._ref(i, c).fv
                for tj, j in enumerate(names):
                    b = self._ref(j, c).fg
                    if a is not None and b is not None:
                        G[ti, tj] = float((a.double() * b.double()).sum())
            valid = np.array([self.usable.get((i, c), (False, 0, 0))[0] for i in names])
            sol = solve_alpha(K, G, valid, delta=cfg.delta, iters=cfg.solver_iters, tol=cfg.solver_tol)
            self.last_solve[c] = sol
            for tj, j in enumerate(names):
                self.alpha[(j, c)] = float(sol["alpha"][tj])
            metrics[f"actor/target/alpha_fallback/{rn}"] = float(sol["fallback"])
            metrics[f"actor/target/alpha_converged/{rn}"] = 1.0 if sol["converged"] else 0.0
            for ti, i in enumerate(names):
                metrics[f"actor/target/constraint_slack/{i}/{rn}"] = float(sol["slack"][ti])
                metrics[f"actor/target/constraint_slack_at_one/{i}/{rn}"] = float(sol["slack_at_one"][ti])
                for tj, j in enumerate(names):
                    metrics[f"actor/target/gram_vg/{i}/{j}/{rn}"] = float(G[ti, tj])

        # 4. metrics
        for ti, n in enumerate(names):
            for c in range(nR):
                rn = ROLE_NAMES[c]
                st = self._ref(n, c)
                n_tok, n_live = g(ti, c, "n_tok"), g(ti, c, "n_live")
                ok, reason, nsteps = self.usable.get((n, c), (False, UNUSABLE_NEVER_SEEN, 0))
                metrics[f"actor/target/usable/{n}/{rn}"] = 1.0 if ok else 0.0
                metrics[f"actor/target/unusable_reason/{n}/{rn}"] = float(reason)
                metrics[f"actor/target/unusable_steps/{n}/{rn}"] = float(nsteps)
                metrics[f"actor/target/alpha/{n}/{rn}"] = float(self.alpha.get((n, c), 1.0))
                metrics[f"actor/target/sbar/{n}/{rn}"] = float(st.sbar)
                metrics[f"actor/target/K/{n}/{rn}"] = float(st.k)
                if n_tok <= 0:
                    continue
                metrics[f"actor/target/live_frac/{n}/{rn}"] = n_live / n_tok
                metrics[f"actor/target/f_mean/{n}/{rn}"] = g(ti, c, "f_sum") / n_tok
                metrics[f"actor/target/e_mean/{n}/{rn}"] = g(ti, c, "e_sum") / n_tok
                metrics[f"actor/target/fe_mean/{n}/{rn}"] = g(ti, c, "fe_sum") / n_tok
                metrics[f"actor/target/tv_qstar_q/{n}/{rn}"] = g(ti, c, "tv_sum") / n_tok
                metrics[f"actor/target/clamped_frac/{n}/{rn}"] = g(ti, c, "clamped_sum") / n_tok
                metrics[f"actor/target/c_absmean/{n}/{rn}"] = g(ti, c, "c_abs_sum") / n_tok
                metrics[f"actor/target/recenter_residual/{n}/{rn}"] = g(ti, c, "recenter_sum") / n_tok
                metrics[f"actor/target/alpha_applied/{n}/{rn}"] = g(ti, c, "alpha_sum") / n_tok
                nn_ = g(ti, c, "n_adv_neg")
                if nn_ > 0:
                    metrics[f"actor/target/e_mean_adv_neg/{n}/{rn}"] = g(ti, c, "e_sum_adv_neg") / nn_
                np_ = g(ti, c, "n_adv_pos")
                if np_ > 0:
                    metrics[f"actor/target/e_mean_adv_pos/{n}/{rn}"] = g(ti, c, "e_sum_adv_pos") / np_
                if n_live > 0:
                    inj = g(ti, c, "inject_norm_sum") / n_live
                    rn_ = g(ti, c, "r_norm_sum") / n_live
                    dn = g(ti, c, "d_norm_sum") / max(n_tok, 1)
                    metrics[f"actor/target/inject_norm/{n}/{rn}"] = inj
                    metrics[f"actor/target/inject_over_r/{n}/{rn}"] = inj / (rn_ + cfg.delta)
                    metrics[f"actor/target/inject_over_d/{n}/{rn}"] = inj / (dn + cfg.delta)
                    isq, rsq = g(ti, c, "inject_sq_sum"), g(ti, c, "r_sq_sum")
                    dot = g(ti, c, "inject_dot_r_sum")
                    if isq > 0 and rsq > 0:
                        metrics[f"actor/target/cos_inject_r/{n}/{rn}"] = dot / math.sqrt(isq * rsq)
                if st.fv is not None and st.fg is not None:
                    a, b = st.fv.double(), st.fg.double()
                    na, nb = float(a.norm()), float(b.norm())
                    metrics[f"actor/target/ref_norm_fv/{n}/{rn}"] = na
                    metrics[f"actor/target/ref_norm_fg/{n}/{rn}"] = nb
                    if na > 0 and nb > 0:
                        metrics[f"actor/target/cos_fv_fg/{n}/{rn}"] = float((a * b).sum()) / (na * nb)
        return metrics

    # ---- checkpoint ----------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "version": 1,
            "cfg": asdict(self.cfg),
            "vocab_size": self.V,
            "step": self.step,
            "tasks": list(self.tasks),
            "refs": {f"{t}|{c}": {
                "fv": None if st.fv is None else st.fv.detach().cpu(),
                "fg": None if st.fg is None else st.fg.detach().cpu(),
                "sbar": float(st.sbar), "k": float(st.k), "n_obs": int(st.n_obs),
                "last_step": int(st.last_step), "tokens": list(st.tokens),
            } for (t, c), st in self.refs.items()},
            "alpha": {f"{t}|{c}": float(v) for (t, c), v in self.alpha.items()},
            "usable": {f"{t}|{c}": list(v) for (t, c), v in self.usable.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        if sd.get("version") != 1:
            raise ValueError(f"target_distill state version {sd.get('version')} is not 1")
        saved = dict(sd.get("cfg", {}))
        saved["roles"] = tuple(saved.get("roles", ()))
        live = asdict(self.cfg)
        drift = {k: (saved.get(k), live[k]) for k in live if saved.get(k) != live[k]}
        if drift:
            raise ValueError(f"target_distill config changed across resume: {drift}")
        if int(sd.get("vocab_size", self.V)) != self.V:
            raise ValueError("target_distill: vocabulary size changed across resume")
        self.step = int(sd.get("step", 0))
        self._ensure(sd.get("tasks", []))
        self.refs = {}
        for key, d in sd.get("refs", {}).items():
            t, c = key.split("|")
            st = _RefState(
                fv=None if d["fv"] is None else d["fv"].to(torch.float32),
                fg=None if d["fg"] is None else d["fg"].to(torch.float32),
                sbar=float(d["sbar"]), k=float(d["k"]), n_obs=int(d["n_obs"]),
                last_step=int(d["last_step"]), tokens=collections.deque(int(x) for x in d.get("tokens", [])))
            self.refs[(t, int(c))] = st
        self.alpha = {}
        for key, v in sd.get("alpha", {}).items():
            t, c = key.split("|")
            self.alpha[(t, int(c))] = float(v)
        self.usable = {}
        for key, v in sd.get("usable", {}).items():
            t, c = key.split("|")
            self.usable[(t, int(c))] = (bool(v[0]), int(v[1]), int(v[2]))
