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
    "lambda_at",
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


# The tail floor. MUST match topk_kl_per_token's eps: the loss and the target
# have to agree about a bucket that carries real mass, or the KL is taken against
# a distribution the target generator never built.
_TAIL_EPS = 1.0e-8

N_ROLES = len(ROLE_NAMES)
ROLE_ID = {name: rid for rid, name in ROLE_NAMES.items()}

# Why a reference (i, c) cannot be used this step. 0 = usable.
UNUSABLE_NONE = 0
UNUSABLE_NEVER_SEEN = 1
UNUSABLE_FEW_TOKENS = 2
UNUSABLE_STALE = 3
UNUSABLE_NONFINITE = 4
UNUSABLE_ZERO_NORM = 5
UNUSABLE_FEW_PROMPTS = 6
UNUSABLE_FEW_PG_PROMPTS = 7
UNUSABLE_NAMES = {
    UNUSABLE_NONE: "ok", UNUSABLE_NEVER_SEEN: "never_seen", UNUSABLE_FEW_TOKENS: "few_tokens",
    UNUSABLE_STALE: "stale", UNUSABLE_NONFINITE: "nonfinite", UNUSABLE_ZERO_NORM: "zero_norm",
    UNUSABLE_FEW_PROMPTS: "few_prompts", UNUSABLE_FEW_PG_PROMPTS: "few_pg_prompts",
}
# Why alpha fell back to 1 rather than being solved. 0 = solved.
#
# EVERY ONE OF THESE FALLS BACK TO alpha = 1, NEVER 0. Under distillation-only
# there is no GRPO term beside the KL, so alpha = 0 does not mean "no cross-task
# adjustment", it means NO REWARD AT ALL for that (task, role) -- the arm
# silently becomes pure OPD. 1 is the value the objective wants; the codes exist
# so the metric can say which of these happened rather than leaving one number.
FALLBACK_NONE = 0
FALLBACK_NO_RECEIVER = 1
FALLBACK_NONFINITE = 2
FALLBACK_UNSOLVED = 3          # the QP itself failed: no KKT point was feasible
FALLBACK_NO_SIGNAL = 4         # every K_j = 0, so the objective cannot rank points
FALLBACK_NO_COMMON_DIR = 5     # feasible only at alpha = 0: no non-zero direction
                               # satisfies every receiver at once
FALLBACK_NAMES = {FALLBACK_NONE: "solved", FALLBACK_NO_RECEIVER: "no_receiver",
                  FALLBACK_NONFINITE: "nonfinite", FALLBACK_UNSOLVED: "unsolved",
                  FALLBACK_NO_SIGNAL: "no_signal", FALLBACK_NO_COMMON_DIR: "no_common_dir"}

# Two disjoint halves of the prompts, as in the cross gate: a reference built on
# one half is applied to the other, so the Gram is never an inner product of a
# direction with the very tokens that produced it. Off the diagonal the halves
# are disjoint anyway (a prompt belongs to one task), but G[i, i] is a real
# self-confirmation risk and the same rule is applied everywhere rather than
# only there.
N_SIDES = 2
# Columns in the per-(task, role, side) prompt bitmaps. The dense prompt index
# comes from the driver's stable anchor key; anything past this cannot be
# counted for validity and is reported as unkeyed, never hidden.
MAX_PROMPTS = 1024
# f in [0, 2] by construction (opd_strength_ratio), so a fixed grid gives exact
# percentiles to one bin without a sort or a host sync.
F_BINS = 40


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
    # beta, the teacher-KL loss coefficient. NOT used to build the target -- it
    # cancels in f, which is a ratio -- but the injected direction is beta F_p c
    # and every magnitude metric is reported on that, so the readout needs it.
    # Passed from the actor, never set independently of teacher_kl_loss_coef.
    beta: float = 1.0
    # A reference (i, c, side) is usable after this many prompts, PG prompts and
    # control tokens in the window, and while it is no more than max_staleness
    # steps old. Same conditions as the cross gate (design section 8).
    window_steps: int = 8
    min_prompts: int = 4
    min_pg_prompts: int = 2
    min_tokens: int = 64
    max_staleness: int = 2
    delta: float = 1.0e-30
    # WHERE THE TARGET IS REWRITTEN AT ALL. Every generated role by default: with
    # pg_loss_coef = 0 there is no other path for the reward, so a role left out
    # of this set gets NO reward signal whatsoever and becomes pure OPD.
    target_roles: tuple = ("format", "reasoning", "env_action", "tool_call", "tag")
    # WHERE THE CROSS-TASK COEFFICIENT IS SOLVED. A subset. Only roles that can
    # carry a shared reference belong here: tool_call exists for search alone and
    # has no cross-task partner, so it stays at alpha = 1 and keeps its own
    # task's RL rather than being zeroed by a solve it cannot participate in.
    roles: tuple = ("format", "env_action")
    solver_iters: int = 500
    solver_tol: float = 1.0e-12
    # LAMBDA DECAY (design section 12): move the target's base between the
    # teacher and the student's own detached distribution. OFF by default, and
    # when off lambda == 1 takes the EXISTING target path unchanged, so arms B
    # and C do not move by a single bit.
    #
    # ONE SCALAR FOR EVERY TASK, deliberately. A per-task lambda_i is a new
    # task-priority knob; choosing it from success rates or cosines brings back
    # the self gate's credit-assignment problem.
    lambda_decay: bool = False
    lambda_min: float = 0.1
    lambda_begin_step: int = 50
    lambda_end_step: int = 250
    # Seed of the prompt -> side hash. The driver reads it off whichever arm is
    # enabled and passes it to cross_gate_prompt_columns, so the two halves are
    # built the same way here as in the cross gate.
    split_seed: int = 1

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
            if k in ("enable", "integrate", "lambda_decay"):
                kw[k] = bool(v)
            elif k in ("roles", "target_roles"):
                kw[k] = tuple(str(x) for x in (list(v) if not isinstance(v, str) else v.split(",")))
            elif k in ("window_steps", "min_tokens", "max_staleness", "solver_iters",
                       "min_prompts", "min_pg_prompts", "split_seed",
                       "lambda_begin_step", "lambda_end_step"):
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
        if self.min_prompts < 1 or self.min_pg_prompts < 0:
            raise ValueError("target_distill.min_prompts must be >= 1 and min_pg_prompts >= 0")
        if self.beta < 0.0:
            raise ValueError(f"target_distill.beta must be >= 0, got {self.beta}")
        if not (0.0 <= self.lambda_min <= 1.0):
            raise ValueError(f"target_distill.lambda_min must be in [0, 1], got {self.lambda_min}")
        if self.lambda_decay and self.lambda_min <= 0.0:
            raise ValueError(
                "target_distill.lambda_min must be > 0: at lambda = 0 a token whose injection is "
                "also zero (clipped, off-support, f = 0) gets no gradient at all. Design 12.5."
            )
        if self.lambda_begin_step < 0 or self.lambda_end_step <= self.lambda_begin_step:
            raise ValueError(
                "target_distill needs 0 <= lambda_begin_step < lambda_end_step, got "
                f"{self.lambda_begin_step} / {self.lambda_end_step}"
            )
        if self.delta <= 0.0:
            raise ValueError("target_distill.delta must be > 0")
        for name, val in (("roles", self.roles), ("target_roles", self.target_roles)):
            unknown = [r for r in val if r not in ROLE_ID]
            if unknown:
                raise ValueError(f"target_distill.{name} has unknown roles {unknown}; "
                                 f"known {sorted(ROLE_ID)}")
        extra = [r for r in self.roles if r not in self.target_roles]
        if extra:
            raise ValueError(
                f"target_distill.roles {extra} are integrated but not in target_roles, so their "
                f"alpha would be solved and then never applied. Integration is a subset of rewriting."
            )
        if self.solver_iters < 1 or self.solver_tol <= 0.0:
            raise ValueError("target_distill.solver_iters must be >= 1 and solver_tol > 0")

    def control_role_mask(self) -> torch.Tensor:
        """Roles whose alpha is SOLVED. A subset of :meth:`target_role_mask`."""
        m = torch.zeros(N_ROLES, dtype=torch.bool)
        for r in self.roles:
            m[ROLE_ID[r]] = True
        return m

    def target_role_mask(self) -> torch.Tensor:
        """Roles where the target is rewritten at all. Outside it the reward is
        gone entirely, because pg_loss_coef = 0 leaves no other path."""
        m = torch.zeros(N_ROLES, dtype=torch.bool)
        for r in self.target_roles:
            m[ROLE_ID[r]] = True
        return m


# --------------------------------------------------------------------------- #
# The three per-token pieces


def lambda_at(step: int, cfg: TargetDistillConfig) -> float:
    """The step's lambda: 1 -> lambda_min, linearly, between the two step bounds.

    A PRE-FIXED schedule, not a controller. Nothing about the run feeds into it:
    a lambda chosen from success rates or from a measured cosine is the self
    gate's credit-assignment problem again, and this arm exists partly because
    that failed. One scalar for every task, so no new task priority is introduced.

    THE FLOOR IS PROVISIONAL. The RL share of the update is about
    eps / (lambda + eps) with eps = ||beta F C|| / ||d||, so the crossover sits at
    lambda ~ eps and a floor of 0.1 does almost nothing if eps is 1e-3. eps is
    exactly what arm B's ``inject_over_d`` reports, so the floor is meant to be
    revised once that is read (design 12.4).
    """
    if not cfg.lambda_decay:
        return 1.0
    t0, t1 = int(cfg.lambda_begin_step), int(cfg.lambda_end_step)
    if step <= t0:
        return 1.0
    if step >= t1:
        return float(cfg.lambda_min)
    frac = (float(step) - t0) / float(t1 - t0)
    return 1.0 - (1.0 - float(cfg.lambda_min)) * frac


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
    """``F_p c = p .* (c - <p, c>)`` -- the logit direction a tilt c actually injects.

    ``-grad_z[beta KL(p||softmax(log q + c))]|_p = d + beta F_p c``. The
    mean-subtraction is not decoration: without it the direction is wrong by 24%
    on a realistic support, and the whole point of the arm is which direction is
    injected.

    ``p`` IS THE UNNORMALISED full-vocabulary softmax at the support, summing to
    m < 1, because the loss keeps the tail as its own bucket. Two consequences,
    both of which a "Fisher matrix" reading gets wrong:

      * ``<p, c>`` is not an expectation -- it is m times one.
      * **F_p DOES NOT KILL A CONSTANT.** ``F_p 1 = p(1 - m)`` exactly, so adding
        a constant to c is in principle a change of mechanism, not a
        re-parameterisation. HOW MUCH IT MATTERS DEPENDS ENTIRELY ON THE PINNED
        SUPPORT, and at this arm's pin it is negligible: ``student_indexed_topk``
        is true, so S is the student's own top-k and the measured tail mass is
        0.000-0.002 (``actor/opd_diag/tail_mass_mean``, cross2 run). At m = 0.999
        re-centring moves the injected direction by 0.02%; it would be 23% at
        m = 0.5, which is what a TEACHER-indexed support could give and the lock
        forbids. So the re-centring below is NOT justified by "F_p kills it
        anyway" -- that reasoning is unsound -- but neither is it a large
        correction here: its real job is to keep the clamp from acting on an
        arbitrary offset. Two earlier versions of this file got this wrong in
        opposite directions: the first claimed F_p kills a constant (its test had
        normalised p, the one case that never occurs), the second claimed the
        effect was several times the direction's own norm (it quoted a support
        mass that is not this quantity).

    Everything here is the component INSIDE the support. The tilt does not reach
    the tail (it has no per-symbol c), so this is not the whole-vocabulary
    gradient of the loss -- only the part the arm can steer.
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
    cfg_lambda: float = 1.0,
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
        # THE SAME DIRECTION IN THE (k+1) SPACE, which is what the lambda
        # decomposition -grad = lam d + beta F C is written on (design 12.2).
        # d_norm above is the k-only norm and is a DIFFERENT NUMBER; it stays as
        # it is because f is defined on it and changing f would make this arm
        # incomparable with the one already specified. Reported side by side.
        _tp = (1.0 - p_s.sum(dim=-1, keepdim=True)).clamp(min=_TAIL_EPS, max=1.0)
        _tq = (1.0 - lp_t.exp().sum(dim=-1, keepdim=True)).clamp(min=_TAIL_EPS, max=1.0)
        _P = torch.cat([p_s, _tp], dim=-1)
        _dlog = torch.cat([lp_t - lp_s, _tq.log() - _tp.log()], dim=-1)
        _d_kp1 = fisher_apply(_P, _dlog)
        d_norm_kp1 = (_d_kp1 * _d_kp1).sum(dim=-1).clamp(min=0.0).sqrt()
        q_tail = _tq.squeeze(-1)

        # --- f, the OPD strength ratio --------------------------------------
        tid = task_ids.reshape(-1)
        tid = tid.round().to(torch.long) if tid.is_floating_point() else tid.to(torch.long)
        tid_ok = (tid >= 0) & (tid < nT)
        tid_c = tid.clamp(min=0, max=max(nT - 1, 0))
        rol = roles.to(torch.long).clamp(min=0, max=nR - 1)
        sbar_t = refs.sbar.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]
        ready = refs.sbar_ready.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]
        # Until sbar exists, f = 1 (neutral). With sbar = 0 the ratio would be 2
        # for every token -- the cap -- so the first step would inject at double
        # strength. Initialising from the first MICRO-batch instead would make
        # the value depend on row order and rank placement, so the controller
        # seeds the EMA from the step's all-reduced aggregate at the end.
        f = torch.where(ready, opd_strength_ratio(d_norm=d_norm, sbar=sbar_t, delta=cfg.delta),
                        torch.ones_like(d_norm))

        # --- e, the RLSD support factor -------------------------------------
        lp_t_a = (hit_f * lp_t).sum(dim=-1)
        lp_s_a = (hit_f * lp_s).sum(dim=-1)
        e = rlsd_support_factor(teacher_logprob_a=lp_t_a, student_logprob_a=lp_s_a,
                                advantages=advantages, epsilon_w=cfg.epsilon_w)
        e = torch.where(in_support, e, torch.ones_like(e))
        # At a bound the RLSD factor stopped carrying the teacher/student gap and
        # became a constant: how often that happens decides whether e is a weight
        # or a switch.
        _eb = 1.0e-6 * max(float(cfg.epsilon_w), 1.0e-6)
        e_clipped = in_support & ((e <= 1.0 - float(cfg.epsilon_w) + _eb)
                                 | (e >= 1.0 + float(cfg.epsilon_w) - _eb))

        # --- where the target is rewritten, and where alpha is solved --------
        tgt = refs.target_roles.to(dev)[rol]
        ctrl = refs.control_roles.to(dev)[rol]
        # OUTSIDE target_roles there is no reward at all: pg_loss_coef = 0 leaves
        # no other path, so a role left out becomes pure OPD.
        live = tgt & tid_ok.view(bs, 1) & in_support
        alpha_t = refs.alpha.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]
        # A role that is rewritten but NOT integrated keeps its own task's RL:
        # alpha = 1, not 0. tool_call is the case that matters -- search alone
        # has it, so it can never have a cross-task partner.
        alpha_t = torch.where(ctrl, alpha_t, torch.ones_like(alpha_t))
        alpha_t = torch.where(live, alpha_t, torch.zeros_like(alpha_t))

        # --- the BASE tilt: centred and clamped ONCE, before alpha ------------
        # alpha multiplies AFTER the clamp so that the injected direction is
        # linear in it: inject = alpha * F_p c_base. Clamping alpha*c instead
        # (the first version) made the integration statistics describe a
        # direction the loss never took, because the clamp is not homogeneous.
        #
        # The centring is a deliberate change of direction, not a normalisation:
        # F_p does not kill a constant on a partial support (see fisher_apply).
        # It is applied so the clamp acts on a quantity with no arbitrary offset,
        # and the SAME centred, clamped base is what the statistics accumulate.
        c_base_raw = float(cfg.eta) * (f * e).unsqueeze(-1) * r
        centre = c_base_raw.mean(dim=-1, keepdim=True)
        c_base = (c_base_raw - centre) * live.unsqueeze(-1).to(dt)
        c_base = c_base.clamp(min=-float(cfg.clamp), max=float(cfg.clamp))
        clamped_frac = (c_base_raw - centre).abs().gt(float(cfg.clamp)).to(dt).mean(dim=-1)

        c = alpha_t.unsqueeze(-1) * c_base

        # --- the target -------------------------------------------------------
        # The (k+1) categories the loss actually uses: the support, plus one tail
        # bucket. lambda moves the BASE between the teacher and the student's own
        # detached distribution (design 12.2):
        #
        #   Q*_lam = softmax( (1-lam) log P_0 + lam log Q + C ),  C_tail = 0
        #
        # This is NOT the full-vocabulary mixture aggregated into a bucket -- it
        # does not use the shape inside the tail, and the two do not agree. Only
        # this one is computable from top-k, so it is the definition, not an
        # approximation. The tail floor is the loss's own (topk_kl_per_token's
        # eps), or the two would disagree about a bucket that carries real mass.
        lam = float(cfg_lambda)
        if lam == 1.0:
            # EXACTLY the existing path, called unchanged. Algebra says the
            # general form reduces to it here; bit-identity does not follow from
            # algebra, and arms B and C must not move because this code was added.
            built = normalized_weight(c=c, p_on=lp_t.exp(), clamp=None)
            target_logprob = lp_t + built["log_w"].to(lp_t.dtype)
            tv = built["moved"].to(dt)
        else:
            t_p = (1.0 - p_s.sum(dim=-1, keepdim=True)).clamp(min=_TAIL_EPS, max=1.0)
            t_q = (1.0 - lp_t.exp().sum(dim=-1, keepdim=True)).clamp(min=_TAIL_EPS, max=1.0)
            z_s = (1.0 - lam) * lp_s + lam * lp_t + c            # (bs, T, k)
            z_t = (1.0 - lam) * t_p.log() + lam * t_q.log()      # (bs, T, 1), C_tail = 0
            log_z = torch.logsumexp(torch.cat([z_s, z_t], dim=-1), dim=-1, keepdim=True)
            target_logprob = (z_s - log_z).to(lp_t.dtype)
            # TV over the whole (k+1) space, tail included, so it is comparable
            # with the lam = 1 branch's "moved".
            tv = 0.5 * ((target_logprob.to(dt).exp() - lp_t.exp()).abs().sum(dim=-1)
                        + ((z_t - log_z).exp().squeeze(-1) - t_q.squeeze(-1)).abs())

        # --- what is actually injected ---------------------------------------
        c_eff = c
        inject = fisher_apply(p_s, c_eff)          # F_p c ; beta is applied by the loss
        inject_base = fisher_apply(p_s, c_base)    # the alpha = 1 direction, for the Gram
        return {
            "target_logprob": target_logprob,
            "r": r, "c": c, "c_eff": c_eff, "inject": inject,
            "f": f, "e": e, "alpha_t": alpha_t, "d_norm": d_norm,
            "d_norm_kp1": d_norm_kp1, "q_tail": q_tail, "lam": lam,
            "in_support": in_support.to(dt), "live": live.to(dt),
            "tv": tv,
            "clamped": clamped_frac,
            "recenter_residual": centre.squeeze(-1).abs() * k,
            "c_base": c_base,
            "inject_base": inject_base,
            "e_clipped": e_clipped.to(dt),
            "rtilde": (f * e).unsqueeze(-1) * r,   # the candidate direction, alpha-free
        }


# --------------------------------------------------------------------------- #
# Per-step accumulators

_TOK_COLS = (
    "n_tok", "n_live", "n_live_pg", "n_inject_nz", "d_norm_sum", "d_norm_sum_live", "d_sq_sum",
    "d_kp1_norm_sum", "q_tail_sum",
    "f_sum", "e_sum", "fe_sum", "e_sum_adv_neg", "n_adv_neg", "e_sum_adv_pos", "n_adv_pos",
    "e_clip_sum", "inject_sq_sum", "r_sq_sum", "rtilde_sq_sum", "inject_dot_r_sum",
    "inject_norm_sum", "inject_base_norm_sum", "inject_base_sq_sum", "removed_norm_sum",
    "r_norm_sum", "tv_sum", "clamped_sum", "recenter_sum", "c_abs_sum", "alpha_sum",
)
_TOK_IDX = {n: i for i, n in enumerate(_TOK_COLS)}


class TargetDistillStats:
    """Sums for one update, on the device, reduced once. Built on the config alone
    so every rank runs the same collectives whatever its micro-batch holds."""

    def __init__(self, n_tasks: int, vocab_size: int, device):
        self.n_tasks, self.V = int(n_tasks), int(vocab_size)
        nT, nR, nS = self.n_tasks, N_ROLES, N_SIDES
        self.tok = torch.zeros(nT, nR, len(_TOK_COLS), dtype=torch.float64, device=device)
        # The references, per (task, role, SIDE), in the space the constraint is
        # written on -- after F_p, and on c_base rather than on rtilde, so they
        # describe the direction the loss actually injected at alpha = 1 (the
        # centring and the clamp are both inside c_base). The raw pre-F sums are
        # NOT kept: nothing read them, and at (nT, nR, V) each cost 10 MiB of
        # device memory and one all-reduce per step.
        self.fv_sum = torch.zeros(nT, nR, nS, self.V, dtype=torch.float32, device=device)
        self.fg_sum = torch.zeros(nT, nR, nS, self.V, dtype=torch.float32, device=device)
        self.side_tok = torch.zeros(nT, nR, nS, dtype=torch.float64, device=device)
        # Which prompts fed each (task, role, side) this step, and which of them
        # carried a non-zero PG direction. Counted as a bitmap so the window can
        # take a union across steps rather than adding duplicates.
        self.prompts = torch.zeros(nT, nR, nS, MAX_PROMPTS, dtype=torch.float32, device=device)
        self.pg_prompts = torch.zeros(nT, nR, nS, MAX_PROMPTS, dtype=torch.float32, device=device)
        self.unkeyed = torch.zeros(1, dtype=torch.float64, device=device)
        # f's distribution, for the percentiles. f in [0, 2] by construction.
        self.fhist = torch.zeros(nT, nR, F_BINS, dtype=torch.float64, device=device)

    def update(self, *, built: dict, task_ids: torch.Tensor, roles: torch.Tensor,
               response_mask: torch.Tensor, topk_ids: torch.Tensor,
               advantages: torch.Tensor | None, row_basis: torch.Tensor | None,
               p_s: torch.Tensor, side: torch.Tensor | None = None,
               prompt_idx: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            nT, nR, nS = self.n_tasks, N_ROLES, N_SIDES
            bs, T = response_mask.shape
            dev = response_mask.device
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
            inj_b = built["inject_base"].to(torch.float32)
            r_sq = (r * r).sum(-1)
            rt_sq = (rt * rt).sum(-1)
            inj_sq = (inj * inj).sum(-1)
            inj_b_norm = (inj_b * inj_b).sum(-1).clamp(min=0).sqrt()
            has_pg = (r_sq > 0).to(torch.float32)

            def add(name, val):
                self.tok.view(ncell, -1)[:, _TOK_IDX[name]].index_add_(
                    0, cell.reshape(-1), (val * mask).reshape(-1).to(torch.float64))
            one = torch.ones_like(mask)
            add("n_tok", one)
            add("n_live", built["live"].to(torch.float32))
            add("n_live_pg", built["live"].to(torch.float32) * has_pg)
            # NOT the same as n_live_pg. r can be non-zero while the injection is
            # zero -- off target_roles, outside the support, f = 0, alpha = 0. At
            # a low lambda a token with no injection has no gradient at all, so
            # this is the count that decides whether the floor is safe (12.6).
            add("n_inject_nz", (inj_sq > 0).to(torch.float32))
            add("d_kp1_norm_sum", built["d_norm_kp1"].to(torch.float32))
            add("q_tail_sum", built["q_tail"].to(torch.float32))
            add("d_norm_sum", built["d_norm"].to(torch.float32))
            add("d_norm_sum_live", built["d_norm"].to(torch.float32) * built["live"].to(torch.float32))
            add("d_sq_sum", built["d_norm"].to(torch.float32) ** 2)
            add("f_sum", built["f"].to(torch.float32))
            add("e_sum", built["e"].to(torch.float32))
            add("fe_sum", (built["f"] * built["e"]).to(torch.float32))
            add("e_clip_sum", built["e_clipped"].to(torch.float32))
            add("inject_sq_sum", inj_sq)
            add("r_sq_sum", r_sq)
            add("rtilde_sq_sum", rt_sq)
            add("inject_dot_r_sum", (inj * r).sum(-1))
            add("inject_norm_sum", inj_sq.clamp(min=0).sqrt())
            add("inject_base_norm_sum", inj_b_norm)
            add("inject_base_sq_sum", inj_b_norm * inj_b_norm)
            # ||F(c_1 - c_alpha)|| = (1 - alpha) ||F c_base||, exactly, because
            # c_alpha = alpha c_base after the A2 fix and F is linear.
            add("removed_norm_sum", (1.0 - built["alpha_t"].to(torch.float32)) * inj_b_norm)
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

            # --- f's histogram, on the same all-token basis as f_mean ---------
            fb = (built["f"].to(torch.float32) * (F_BINS / 2.0)).floor().to(torch.long).clamp(0, F_BINS - 1)
            self.fhist.view(ncell * F_BINS).index_add_(
                0, (cell * F_BINS + fb).reshape(-1),
                mask.reshape(-1).to(torch.float64))

            # --- the vocabulary-space references, per side, on LIVE tokens ----
            if side is None:
                sd = torch.zeros(bs, dtype=torch.long, device=dev)
                keyed = torch.zeros(bs, dtype=torch.bool, device=dev)
            else:
                sd = side.detach().reshape(-1).to(torch.long)
                keyed = (sd >= 0) & (sd < nS)
                sd = sd.clamp(0, nS - 1)
            self.unkeyed += (~keyed & (mask.sum(-1) > 0)).sum().to(torch.float64)
            klive = live * keyed.to(torch.float32).unsqueeze(-1)
            cell_s = (tid_c.unsqueeze(-1) * nR + rol) * nS + sd.unsqueeze(-1)
            self.side_tok.view(-1).index_add_(
                0, cell_s.reshape(-1), klive.reshape(-1).to(torch.float64))
            ids = topk_ids.to(torch.long).clamp(min=0, max=self.V - 1)
            flat = (cell_s.unsqueeze(-1) * self.V + ids).reshape(-1)
            lw = klive.unsqueeze(-1)
            fv = fisher_apply(p_s.to(torch.float32), r)
            fg = built["inject_base"].to(torch.float32)   # = F_p c_base
            for buf, val in ((self.fv_sum, fv), (self.fg_sum, fg)):
                buf.view(-1).index_add_(0, flat, (val * lw).reshape(-1).to(torch.float32))

            # --- the prompt bitmaps -------------------------------------------
            if prompt_idx is not None:
                pidx = prompt_idx.detach().reshape(-1).to(torch.long)
                pok = keyed & (pidx >= 0) & (pidx < MAX_PROMPTS)
                pidx = pidx.clamp(0, MAX_PROMPTS - 1)
                rowok = pok.to(torch.float32).unsqueeze(-1)
                pflat = (cell_s * MAX_PROMPTS + pidx.unsqueeze(-1)).reshape(-1)
                self.prompts.view(-1).index_add_(
                    0, pflat, (live * rowok).reshape(-1).to(torch.float32))
                self.pg_prompts.view(-1).index_add_(
                    0, pflat, (live * rowok * has_pg).reshape(-1).to(torch.float32))

    def reduced(self) -> dict:
        out = {}
        for name in ("tok", "fv_sum", "fg_sum", "side_tok", "prompts", "pg_prompts",
                     "unkeyed", "fhist"):
            buf = getattr(self, name)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                buf = buf.clone()
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            out[name] = buf.detach().cpu()
        # A prompt is present or it is not; the sums above counted its tokens.
        out["prompts"] = (out["prompts"] > 0)
        out["pg_prompts"] = (out["pg_prompts"] > 0)
        return out


# --------------------------------------------------------------------------- #
# The per-role constrained integration


# Below this an alpha is "off". The QP lives on [0, 1] and 0 is always feasible
# (G 0 = 0 >= 0), so a solution at 0 does not mean 0 is good -- it means nothing
# else was feasible.
_ALPHA_ZERO = 1.0e-9
_RIDGE_REL = 1.0e-9


def solve_alpha(K: np.ndarray, Gvg: np.ndarray, valid_recv: np.ndarray, *,
                delta: float = 1e-30, iters: int = 500, tol: float = 1e-12) -> dict:
    """min 1/2 sum_j K_j (1 - a_j)^2  s.t.  sum_j a_j Gvg[i, j] >= 0 for valid i, 0 <= a <= 1.

    ``Gvg[i, j] = (F v_i)^T (F g_j)`` -- the constraint is written on the
    directions the distillation actually injects, not on the raw reference inner
    products, because those differ by F_p (design section 0).

    SOLVED EXACTLY BY ACTIVE-SET ENUMERATION, not by a penalty. The problem is a
    strictly convex QP whose feasible set is a polytope with at most n_task + 2
    n_task faces, so for the three or four tasks this ever runs on, every
    candidate KKT point can be enumerated: pick a subset of constraints to hold
    with equality, solve the resulting linear system, keep it if it is feasible,
    and take the best. A penalty method was tried first and returned points 60%
    worse than a coarse grid -- it drove alpha down to satisfy the constraint and
    had no way back up.

    SCALE. The receiver rows are inner products of two EMAs of F_p c and carry
    whatever units those have -- measured at 1e-12 and below, so an absolute
    feasibility slack would accept every point and an absolute ridge would swamp
    K. Both are therefore relative: each receiver row is normalised to unit
    length before the enumeration (which changes no constraint, since its
    right-hand side is 0), feasibility is judged on that normalised residual
    against ``tol``, and the ridge on K is a fraction of ``max|K|``. The bound
    rows are already unit-norm with O(1) right-hand sides.

    a = 1 is what the objective wants and is returned whenever it is feasible.
    Returns ``alpha``, the constraint value at the solution and at a = 1, and a
    fallback code: A CALLER THAT GETS ANYTHING BUT FALLBACK_NONE MUST USE
    alpha = 1, NOT 0 -- with distillation-only, alpha = 0 removes the RL signal
    entirely (design section 2.3). The codes separate the two ways this ends
    badly: FALLBACK_UNSOLVED means the arithmetic failed and the answer is
    unknown, FALLBACK_NO_COMMON_DIR means the arithmetic succeeded and its answer
    was "switch every sender off" -- there is no non-zero combination that no
    valid receiver objects to. They are different findings about the run and the
    metric must not blur them, even though the action taken is the same.
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
    # The rows as the enumeration sees them: unit length, so the tolerance below
    # is a relative one. A receiver whose row is identically zero constrains
    # nothing and is dropped rather than normalised by 0.
    rown = np.linalg.norm(G, axis=1)
    act = np.nonzero(valid & (rown > 0.0))[0]
    Gn = np.zeros_like(G)
    Gn[act] = G[act] / rown[act, None]
    if (s1[act] >= -float(tol) * rown[act]).all():
        return {"alpha": one, "slack": s1, "slack_at_one": s1,
                "fallback": FALLBACK_NONE, "converged": True}

    scale = float(np.abs(K).max())
    if not (scale > 0.0):
        # No sender has any measured magnitude, so the objective cannot rank the
        # feasible points at all. Any answer would be an artefact of the ridge.
        return {"alpha": one, "slack": s1, "slack_at_one": s1,
                "fallback": FALLBACK_NO_SIGNAL, "converged": False}
    # A sender with no observed signal has K_j = 0 and the objective would not
    # care where its alpha lands -- but its column of G is zero as well (K = 0
    # means every c_base was 0, hence g_j = 0), so it appears in no constraint
    # and the ridge leaves it at 1, which is the direction that keeps the RL.
    #
    # DIVIDED BY ITS OWN SCALE. A positive factor on the objective moves no
    # argmin, but it decides whether the arithmetic below means anything: K is
    # measured at 1e-24 and smaller, so the "is this candidate better" test had
    # been comparing objective values of that size against an absolute 1e-15 and
    # was keeping whichever feasible point the enumeration reached FIRST. At
    # beta^2 ||F c||^2 scale that silently returned alpha = 0 where the true
    # optimum was 0.5. It also conditions the KKT systems, whose matrices are
    # built from 1/Kp.
    Kp = (np.maximum(K, 0.0) + _RIDGE_REL * scale) / scale

    # rows of  A a >= b :  the receivers' normalised conditions, then 0 <= a <= 1
    rows = [Gn[i] for i in act] + [np.eye(n)[j] for j in range(n)] + [-np.eye(n)[j] for j in range(n)]
    rhs = [0.0] * int(act.size) + [0.0] * n + [-1.0] * n
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    m = A.shape[0]

    def feasible(a):
        return bool(np.isfinite(a).all() and (A @ a >= b - float(tol)).all())

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
    if float(np.abs(best_a).max()) <= _ALPHA_ZERO:
        # The solve worked and said: turn everything off. Under distillation-only
        # that is not a weaker intervention, it is no reward at all, so the arm
        # keeps alpha = 1 and the metric records that the constraint could not be
        # met by any non-zero combination.
        return {"alpha": one, "slack": s1, "slack_at_one": s1,
                "fallback": FALLBACK_NO_COMMON_DIR, "converged": True}
    return {"alpha": best_a, "slack": G @ best_a, "slack_at_one": s1,
            "fallback": FALLBACK_NONE, "converged": True}


# --------------------------------------------------------------------------- #
# What the loss reads, and the controller that produces it


@dataclass
class TargetDistillRefs:
    """Previous-step quantities in THIS batch's task order, on the device."""
    alpha: torch.Tensor          # (nT, nR) cross-task coefficient, 1 where not integrated
    sbar: torch.Tensor           # (nT, nR) EMA of ||d||, the scale f is relative to
    sbar_ready: torch.Tensor     # (nT, nR) bool -- false until the first step's aggregate
    control_roles: torch.Tensor  # (nR,) bool -- where alpha is solved
    target_roles: torch.Tensor   # (nR,) bool -- where the target is rewritten
    lam: float = 1.0             # this step's lambda, one scalar for every task


@dataclass
class _RefState:
    """One (task, role, side). The two sides never share a prompt."""
    fv: torch.Tensor | None = None      # (V,) EMA of E[F_p r]         (receiver)
    fg: torch.Tensor | None = None      # (V,) EMA of E[F_p c_base]    (sender)
    k: float = 0.0                      # EMA of E||F_p c_base||^2
    n_obs: int = 0
    last_step: int = -1
    prompts: collections.deque = None
    pg_prompts: collections.deque = None
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
        self.refs: dict = {}                # (task, role, side) -> _RefState
        self.sbar: dict = {}                # (task, role) -> float
        self.sbar_n: dict = {}              # (task, role) -> observations folded in
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

    def _ref(self, task, role, side) -> _RefState:
        key = (str(task), int(role), int(side))
        st = self.refs.get(key)
        if st is None:
            st = _RefState(prompts=collections.deque(), pg_prompts=collections.deque(),
                           tokens=collections.deque())
            self.refs[key] = st
        return st

    def refs_to_device(self, names, device, dtype=torch.float32) -> TargetDistillRefs:
        self._ensure(names)
        names = [str(n) for n in names]
        nT, nR = len(names), N_ROLES
        alpha = torch.ones(nT, nR, dtype=dtype)
        sbar = torch.zeros(nT, nR, dtype=dtype)
        ready = torch.zeros(nT, nR, dtype=torch.bool)
        for ti, n in enumerate(names):
            for c in range(nR):
                alpha[ti, c] = float(self.alpha.get((n, c), 1.0))
                sbar[ti, c] = float(self.sbar.get((n, c), 0.0))
                ready[ti, c] = int(self.sbar_n.get((n, c), 0)) > 0
        return TargetDistillRefs(alpha=alpha.to(device), sbar=sbar.to(device),
                                 sbar_ready=ready.to(device),
                                 control_roles=self.cfg.control_role_mask().to(device),
                                 target_roles=self.cfg.target_role_mask().to(device),
                                 lam=lambda_at(self.step, self.cfg))

    def update(self, names, reduced: dict) -> dict:
        """Fold this step's reduced sums in; solve alpha for the NEXT step."""
        cfg = self.cfg
        names = [str(n) for n in names]
        self._ensure(names)
        nT, nR, nS = len(names), N_ROLES, N_SIDES
        tok = reduced["tok"]
        fv_s, fg_s, side_tok = reduced["fv_sum"], reduced["fg_sum"], reduced["side_tok"]
        prm, pgp = reduced["prompts"], reduced["pg_prompts"]
        fhist = reduced["fhist"]
        beta = float(cfg.beta)
        metrics: dict = {}
        self.step += 1
        step = self.step
        dec = cfg.ema_decay

        def g(ti, c, col):
            return float(tok[ti, c, _TOK_IDX[col]])

        # 1. sbar, per (task, role). SEEDED FROM THE STEP AGGREGATE, never from a
        #    micro-batch: the all-reduced sum is the same on every rank and does
        #    not depend on which rows landed where, and until it exists f is held
        #    at 1 by build_target rather than at the cap of 2 that sbar = 0 gives.
        for ti, n in enumerate(names):
            for c in range(nR):
                n_tok = g(ti, c, "n_tok")
                if n_tok <= 0:
                    continue          # missing is not zero: hold the value and age
                sb = g(ti, c, "d_norm_sum") / n_tok
                seen = int(self.sbar_n.get((n, c), 0))
                self.sbar[(n, c)] = sb if (seen == 0 or dec == 0.0) else \
                    dec * float(self.sbar[(n, c)]) + (1 - dec) * sb
                self.sbar_n[(n, c)] = seen + 1

        # 2. the references and K, per (task, role, side)
        for ti, n in enumerate(names):
            for c in range(nR):
                for sd in range(nS):
                    st = self._ref(n, c, sd)
                    n_live = float(side_tok[ti, c, sd])
                    st.tokens.append(int(n_live))
                    st.prompts.append(frozenset(torch.nonzero(prm[ti, c, sd]).reshape(-1).tolist()))
                    st.pg_prompts.append(frozenset(torch.nonzero(pgp[ti, c, sd]).reshape(-1).tolist()))
                    while len(st.tokens) > cfg.window_steps:
                        st.tokens.popleft(); st.prompts.popleft(); st.pg_prompts.popleft()
                    if n_live <= 0:
                        continue
                    mfv = (fv_s[ti, c, sd] / n_live).to(torch.float32)
                    mfg = (fg_s[ti, c, sd] / n_live).to(torch.float32)
                    # K weighs how much is lost by turning sender j down, so it is
                    # the magnitude of the very thing alpha scales: ||F c_base||^2.
                    # Split across the sides only by which tokens fed it.
                    denom = max(g(ti, c, "n_live"), 1.0)
                    kk = g(ti, c, "inject_base_sq_sum") / denom
                    if st.n_obs == 0 or dec == 0.0 or st.fv is None:
                        st.fv, st.fg, st.k = mfv.clone(), mfg.clone(), kk
                    else:
                        st.fv = dec * st.fv + (1 - dec) * mfv
                        st.fg = dec * st.fg + (1 - dec) * mfg
                        st.k = dec * st.k + (1 - dec) * kk
                    st.n_obs += 1
                    st.last_step = step

        # 3. usability per (i, c): BOTH sides must pass, on the same conditions
        #    as the cross gate (design section 8) -- 4 prompts, 2 of them with a
        #    non-zero PG direction, 64 tokens, over an 8-step window, no more
        #    than 2 steps stale.
        for n in names:
            for c in range(nR):
                reason = UNUSABLE_NONE
                for sd in range(nS):
                    st = self._ref(n, c, sd)
                    if st.n_obs == 0 or st.fv is None or st.fg is None:
                        reason = UNUSABLE_NEVER_SEEN; break
                    if not (torch.isfinite(st.fv).all() and torch.isfinite(st.fg).all()
                            and math.isfinite(st.k)):
                        reason = UNUSABLE_NONFINITE; break
                    if float(st.fv.double().norm()) <= 0.0:
                        reason = UNUSABLE_ZERO_NORM; break
                    if step - st.last_step > cfg.max_staleness:
                        reason = UNUSABLE_STALE; break
                    if len(set().union(*st.prompts) if st.prompts else set()) < cfg.min_prompts:
                        reason = UNUSABLE_FEW_PROMPTS; break
                    if len(set().union(*st.pg_prompts) if st.pg_prompts else set()) < cfg.min_pg_prompts:
                        reason = UNUSABLE_FEW_PG_PROMPTS; break
                    if sum(st.tokens) < cfg.min_tokens:
                        reason = UNUSABLE_FEW_TOKENS; break
                prev = self.usable.get((n, c), (False, UNUSABLE_NEVER_SEEN, 0))
                self.usable[(n, c)] = (reason == UNUSABLE_NONE, reason,
                                       0 if reason == UNUSABLE_NONE else prev[2] + 1)

        # 4. solve alpha per controlled role
        ctrl = cfg.control_role_mask()
        for c in range(nR):
            rn = ROLE_NAMES[c]
            if not bool(ctrl[c]) or not cfg.integrate:
                for n in names:
                    self.alpha[(n, c)] = 1.0
                if bool(ctrl[c]):
                    metrics[f"actor/target/alpha_fallback/{rn}"] = float(FALLBACK_NONE)
                continue
            # K on the injected scale, so the objective and the constraint are in
            # the same units: both carry beta^2.
            K = np.array([beta * beta * sum(self._ref(n, c, sd).k for sd in range(nS)) / nS
                          for n in names], dtype=np.float64)
            # CROSS-SIDE ONLY. G[i, j] pairs receiver i's reference from one half
            # of the prompts with sender j's from the other, averaged over the two
            # pairings. Off the diagonal the halves are disjoint anyway (a prompt
            # belongs to one task); G[i, i] is where it matters, and the same rule
            # is applied everywhere rather than only there.
            G = np.zeros((nT, nT))
            for ti, i in enumerate(names):
                for tj, j in enumerate(names):
                    acc, cnt = 0.0, 0
                    for sd in range(nS):
                        a = self._ref(i, c, sd).fv
                        b = self._ref(j, c, 1 - sd).fg
                        if a is not None and b is not None:
                            acc += float((a.double() * b.double()).sum()); cnt += 1
                    G[ti, tj] = beta * beta * acc / cnt if cnt else 0.0
            valid = np.array([self.usable.get((i, c), (False, 0, 0))[0] for i in names])
            sol = solve_alpha(K, G, valid, delta=cfg.delta, iters=cfg.solver_iters, tol=cfg.solver_tol)
            self.last_solve[c] = sol
            for tj, j in enumerate(names):
                self.alpha[(j, c)] = float(sol["alpha"][tj])
            metrics[f"actor/target/alpha_fallback/{rn}"] = float(sol["fallback"])
            metrics[f"actor/target/alpha_converged/{rn}"] = 1.0 if sol["converged"] else 0.0
            _al = np.asarray(sol["alpha"], dtype=np.float64)
            metrics[f"actor/target/alpha_at_bound_frac/{rn}"] = float(
                np.mean((_al <= 1e-9) | (_al >= 1.0 - 1e-9)))
            for ti, i in enumerate(names):
                metrics[f"actor/target/constraint_slack/{i}/{rn}"] = float(sol["slack"][ti])
                metrics[f"actor/target/constraint_slack_at_one/{i}/{rn}"] = float(sol["slack_at_one"][ti])
                for tj, j in enumerate(names):
                    metrics[f"actor/target/gram_vg/{i}/{j}/{rn}"] = float(G[ti, tj])
                    # The same Gram as an angle, so a number at 1e-12 can be read
                    # as "aligned but tiny" or "orthogonal" rather than only small.
                    na = math.sqrt(sum(float(self._ref(i, c, sd).fv.double().pow(2).sum())
                                       for sd in range(nS) if self._ref(i, c, sd).fv is not None) / nS)
                    nb = math.sqrt(sum(float(self._ref(j, c, sd).fg.double().pow(2).sum())
                                       for sd in range(nS) if self._ref(j, c, sd).fg is not None) / nS)
                    if na > 0 and nb > 0:
                        metrics[f"actor/target/cos_vg/{i}/{j}/{rn}"] = float(
                            G[ti, tj] / (beta * beta * na * nb))

        # 5. metrics
        edges = torch.linspace(0.0, 2.0, F_BINS + 1, dtype=torch.float64)
        for ti, n in enumerate(names):
            for c in range(nR):
                rn = ROLE_NAMES[c]
                n_tok, n_live = g(ti, c, "n_tok"), g(ti, c, "n_live")
                ok, reason, nsteps = self.usable.get((n, c), (False, UNUSABLE_NEVER_SEEN, 0))
                metrics[f"actor/target/valid/{n}/{rn}"] = 1.0 if ok else 0.0
                metrics[f"actor/target/invalid_reason/{n}/{rn}"] = float(reason)
                metrics[f"actor/target/invalid_steps/{n}/{rn}"] = float(nsteps)
                metrics[f"actor/target/alpha/{n}/{rn}"] = float(self.alpha.get((n, c), 1.0))
                metrics[f"actor/target/sbar/{n}/{rn}"] = float(self.sbar.get((n, c), 0.0))
                metrics[f"actor/target/sbar_ready/{n}/{rn}"] = float(int(self.sbar_n.get((n, c), 0)) > 0)
                for sd in range(nS):
                    st = self._ref(n, c, sd)
                    metrics[f"actor/target/K_side{sd + 1}/{n}/{rn}"] = beta * beta * float(st.k)
                    metrics[f"actor/target/prompts_side{sd + 1}/{n}/{rn}"] = float(
                        len(set().union(*st.prompts)) if st.prompts else 0)
                    metrics[f"actor/target/pg_prompts_side{sd + 1}/{n}/{rn}"] = float(
                        len(set().union(*st.pg_prompts)) if st.pg_prompts else 0)
                    metrics[f"actor/target/tokens_window_side{sd + 1}/{n}/{rn}"] = float(sum(st.tokens))
                    metrics[f"actor/target/staleness_side{sd + 1}/{n}/{rn}"] = float(
                        step - st.last_step if st.last_step >= 0 else -1)
                    if st.fv is not None and st.fg is not None:
                        metrics[f"actor/target/ref_norm_fv_side{sd + 1}/{n}/{rn}"] = \
                            beta * float(st.fv.double().norm())
                        metrics[f"actor/target/ref_norm_fg_side{sd + 1}/{n}/{rn}"] = \
                            beta * float(st.fg.double().norm())
                        metrics[f"actor/target/n_distinct_side{sd + 1}/{n}/{rn}"] = float((st.fg != 0).sum())
                        a1 = float(st.fg.double().abs().sum())
                        a2 = float(st.fg.double().pow(2).sum())
                        # A participation ratio, NOT a sample count: it says how
                        # many coordinates carry the mass, and nothing about how
                        # many independent observations produced it.
                        metrics[f"actor/target/n_eff_side{sd + 1}/{n}/{rn}"] = (a1 * a1 / a2) if a2 > 0 else 0.0
                # THE reproducibility check the design asks for: the same
                # reference built on two disjoint halves of the prompts. If this
                # is near 0 the direction is noise and a constraint written on it
                # says nothing, however tight the slack looks.
                for nm, sel in (("fv", lambda t: t.fv), ("fg", lambda t: t.fg)):
                    v0, v1 = sel(self._ref(n, c, 0)), sel(self._ref(n, c, 1))
                    if v0 is not None and v1 is not None:
                        n0, n1 = float(v0.double().norm()), float(v1.double().norm())
                        if n0 > 0 and n1 > 0:
                            metrics[f"actor/target/ref_cos_sides_{nm}/{n}/{rn}"] = \
                                float((v0.double() * v1.double()).sum()) / (n0 * n1)
                if n_tok <= 0:
                    continue
                metrics[f"actor/target/live_frac/{n}/{rn}"] = n_live / n_tok
                # Of the rewritten tokens, how many carried a non-zero PG direction
                # at all: the rest sit inside a clip branch or in a group whose
                # rewards all tied, and their target is the teacher exactly.
                metrics[f"actor/target/pg_live_frac/{n}/{rn}"] = g(ti, c, "n_live_pg") / n_tok
                # ||d||'s spread, so sbar can be read as a scale rather than a point
                _d1 = g(ti, c, "d_norm_sum") / n_tok
                _d2 = g(ti, c, "d_sq_sum") / n_tok
                metrics[f"actor/target/d_norm_sd/{n}/{rn}"] = math.sqrt(max(_d2 - _d1 * _d1, 0.0))
                metrics[f"actor/target/f_mean/{n}/{rn}"] = g(ti, c, "f_sum") / n_tok
                metrics[f"actor/target/e_mean/{n}/{rn}"] = g(ti, c, "e_sum") / n_tok
                metrics[f"actor/target/fe_mean/{n}/{rn}"] = g(ti, c, "fe_sum") / n_tok
                metrics[f"actor/target/e_clip_frac/{n}/{rn}"] = g(ti, c, "e_clip_sum") / n_tok
                metrics[f"actor/target/tv_qstar_q/{n}/{rn}"] = g(ti, c, "tv_sum") / n_tok
                metrics[f"actor/target/clamped_frac/{n}/{rn}"] = g(ti, c, "clamped_sum") / n_tok
                metrics[f"actor/target/c_absmean/{n}/{rn}"] = g(ti, c, "c_abs_sum") / n_tok
                metrics[f"actor/target/recenter_residual/{n}/{rn}"] = g(ti, c, "recenter_sum") / n_tok
                metrics[f"actor/target/alpha_applied/{n}/{rn}"] = g(ti, c, "alpha_sum") / n_tok
                h = fhist[ti, c].double()
                tot = float(h.sum())
                if tot > 0:
                    cdf = torch.cumsum(h, dim=0) / tot
                    for q, nm in ((0.10, "f_p10"), (0.50, "f_p50"), (0.90, "f_p90")):
                        idx = int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)).item())
                        metrics[f"actor/target/{nm}/{n}/{rn}"] = float(edges[min(idx + 1, F_BINS)])
                nn_ = g(ti, c, "n_adv_neg")
                if nn_ > 0:
                    metrics[f"actor/target/e_mean_adv_neg/{n}/{rn}"] = g(ti, c, "e_sum_adv_neg") / nn_
                np_ = g(ti, c, "n_adv_pos")
                if np_ > 0:
                    metrics[f"actor/target/e_mean_adv_pos/{n}/{rn}"] = g(ti, c, "e_sum_adv_pos") / np_
                # THE MAGNITUDE METRICS ARE beta-INCLUSIVE. What the update sees
                # is d + beta F_p c, so beta F_p c is the intervention and
                # ||F_p c|| alone is off by a factor of 100 at beta = 0.01. The
                # beta-free form is kept under its own name for the arithmetic.
                inj_all = beta * g(ti, c, "inject_norm_sum") / n_tok
                d_all = g(ti, c, "d_norm_sum") / n_tok
                metrics[f"actor/target/inject_norm/{n}/{rn}"] = inj_all
                metrics[f"actor/target/inject_norm_nobeta/{n}/{rn}"] = g(ti, c, "inject_norm_sum") / n_tok
                # PRIMARY: both sides over EVERY loss token, the population the
                # loss is averaged over. A non-live token contributes 0 to the
                # numerator and its real ||d|| to the denominator, which is the
                # honest statement of how much of the update the arm touches.
                metrics[f"actor/target/inject_over_d/{n}/{rn}"] = inj_all / (d_all + cfg.delta)
                # The (k+1) OPD norm, which is what the lambda decomposition is
                # written on -- a DIFFERENT number from d_all, which is k-only and
                # is what f is defined on. Both are reported so neither is read as
                # the other (design 12.2).
                _lam = lambda_at(step - 1, cfg)
                d_kp1 = g(ti, c, "d_kp1_norm_sum") / n_tok
                metrics[f"actor/target/d_norm_kp1/{n}/{rn}"] = d_kp1
                metrics[f"actor/target/teacher_contrib/{n}/{rn}"] = _lam * d_kp1
                metrics[f"actor/target/inject_over_teacher/{n}/{rn}"] = \
                    inj_all / (_lam * d_kp1 + cfg.delta)
                metrics[f"actor/target/inject_nz_frac/{n}/{rn}"] = g(ti, c, "n_inject_nz") / n_tok
                metrics[f"actor/target/q_tail_mass/{n}/{rn}"] = g(ti, c, "q_tail_sum") / n_tok
                if n_live > 0:
                    inj_live = beta * g(ti, c, "inject_norm_sum") / n_live
                    d_live = g(ti, c, "d_norm_sum_live") / n_live
                    rn_ = g(ti, c, "r_norm_sum") / n_live
                    # AUXILIARY: the same ratio restricted to the tokens the arm
                    # actually rewrote, numerator and denominator on that one mask.
                    metrics[f"actor/target/inject_over_d_live/{n}/{rn}"] = inj_live / (d_live + cfg.delta)
                    metrics[f"actor/target/inject_over_r/{n}/{rn}"] = inj_live / (rn_ + cfg.delta)
                    ib = g(ti, c, "inject_base_norm_sum")
                    # what the centring and the clamp cost: the candidate's RMS
                    # before either, against the injected magnitude after both
                    metrics[f"actor/target/rtilde_rms/{n}/{rn}"] = math.sqrt(
                        max(g(ti, c, "rtilde_sq_sum") / n_live, 0.0))
                    metrics[f"actor/target/inject_base_norm/{n}/{rn}"] = beta * ib / n_live
                    metrics[f"actor/target/removed_by_integration/{n}/{rn}"] = \
                        beta * g(ti, c, "removed_norm_sum") / n_live
                    metrics[f"actor/target/removed_frac/{n}/{rn}"] = \
                        g(ti, c, "removed_norm_sum") / (ib + cfg.delta)
                    isq, rsq = g(ti, c, "inject_sq_sum"), g(ti, c, "r_sq_sum")
                    dot = g(ti, c, "inject_dot_r_sum")
                    if isq > 0 and rsq > 0:
                        metrics[f"actor/target/cos_inject_r/{n}/{rn}"] = dot / math.sqrt(isq * rsq)
        # The lambda the loss USED this step -- read before the increment above,
        # like alpha, so the number names the step it acted on.
        metrics["actor/target/lambda_now"] = lambda_at(step - 1, cfg)
        metrics["actor/target/unkeyed_rows"] = float(reduced["unkeyed"].reshape(-1)[0])
        return metrics

    # ---- checkpoint ----------------------------------------------------
    # VERSION 2. Version 1 held one _RefState per (task, role) whose ``fg`` was
    # the mean of rtilde; the references are now per (task, role, SIDE) and fg is
    # the mean of F_p c_base -- centred, clamped, after F. The numbers are not
    # comparable, so a v1 state is refused rather than reinterpreted. No run has
    # produced one.
    STATE_VERSION = 2

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "cfg": asdict(self.cfg),
            "vocab_size": self.V,
            "step": self.step,
            "tasks": list(self.tasks),
            "refs": {f"{t}|{c}|{sd}": {
                "fv": None if st.fv is None else st.fv.detach().cpu(),
                "fg": None if st.fg is None else st.fg.detach().cpu(),
                "k": float(st.k), "n_obs": int(st.n_obs), "last_step": int(st.last_step),
                "tokens": list(st.tokens),
                "prompts": [sorted(x) for x in st.prompts],
                "pg_prompts": [sorted(x) for x in st.pg_prompts],
            } for (t, c, sd), st in self.refs.items()},
            "sbar": {f"{t}|{c}": float(v) for (t, c), v in self.sbar.items()},
            "sbar_n": {f"{t}|{c}": int(v) for (t, c), v in self.sbar_n.items()},
            "alpha": {f"{t}|{c}": float(v) for (t, c), v in self.alpha.items()},
            "usable": {f"{t}|{c}": list(v) for (t, c), v in self.usable.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        if sd.get("version") != self.STATE_VERSION:
            raise ValueError(
                f"target_distill state version {sd.get('version')} is not {self.STATE_VERSION}")
        saved = dict(sd.get("cfg", {}))
        for _k in ("roles", "target_roles"):
            if _k in saved:
                saved[_k] = tuple(saved[_k])
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
            t, c, side = key.split("|")
            st = _RefState(
                fv=None if d["fv"] is None else d["fv"].to(torch.float32),
                fg=None if d["fg"] is None else d["fg"].to(torch.float32),
                k=float(d["k"]), n_obs=int(d["n_obs"]), last_step=int(d["last_step"]),
                tokens=collections.deque(int(x) for x in d.get("tokens", [])),
                prompts=collections.deque(frozenset(x) for x in d.get("prompts", [])),
                pg_prompts=collections.deque(frozenset(x) for x in d.get("pg_prompts", [])),
            )
            self.refs[(t, int(c), int(side))] = st
        self.sbar, self.sbar_n, self.alpha, self.usable = {}, {}, {}, {}
        for key, v in sd.get("sbar", {}).items():
            t, c = key.split("|"); self.sbar[(t, int(c))] = float(v)
        for key, v in sd.get("sbar_n", {}).items():
            t, c = key.split("|"); self.sbar_n[(t, int(c))] = int(v)
        for key, v in sd.get("alpha", {}).items():
            t, c = key.split("|"); self.alpha[(t, int(c))] = float(v)
        for key, v in sd.get("usable", {}).items():
            t, c = key.split("|"); self.usable[(t, int(c))] = (bool(v[0]), int(v[1]), int(v[2]))
