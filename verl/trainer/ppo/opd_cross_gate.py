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

"""Cross-task soft gate on the OPD term, driven by OTHER tasks' role-wise RL references.

MOPD v1 (docs/opd_output_space_cross_gate_design.md). On top of pure OPD+GRPO
-- no self gate, no self-conflict condition, no per-task coefficient -- the
teacher-KL term of task j at token t is multiplied by

    w_{j,t} = 1 - lambda_{j,c(t)} * h_{j,t},        0 <= lambda <= lambda_max

where c(t) is the token's role (sign_weights.token_roles) and h in [0, 1] is a
soft gate built ONLY from other tasks' references:

    q_{ij,t} = min_k [ -(v^(k)_{i,c})^T d_{j,t} ]_+ / ( sqrt(R^(k)_{i,c}) ||d_{j,t}|| + delta )
    h_{j,t}  = mean over valid receivers i != j of q_{ij,t}

v^(k)_{i,c} and R^(k)_{i,c} are the previous step's EMA of E_{i,c}[r] and
E_{i,c}[||r||^2] on two PROMPT-DISJOINT halves k = 1, 2 (the side is a stable
hash of the prompt, so the same prompt never feeds both). d = beta * g_opd is
the OPD descent direction on the token's top-k support, and
r = c_t (e_a - p) the policy gradient's, with c_t the CLIPPED coefficient
(core_algos.policy_loss_gradient_coef), never the advantage.

By Jensen ||v||^2 = ||E r||^2 <= E ||r||^2 = R whenever v and R are updated
with the same weights, so q <= min_k sqrt(kappa^(k)) <= 1 with
kappa = ||v||^2 / R. A reference with no consistent direction cannot drive a
large intervention; that is a property of the definition, not a tuning.

lambda is chosen once per step, per role, from the reduced statistics:

    min_{0 <= lambda <= lambda_max}  sum_j K_{j,c} lambda_j^2
                                     + rho * sum_{i valid} ( s_i(lambda) / (R_{i,c} + delta) )^2
    s_i(lambda) = [ -eps R_{i,c} - sum_{j != i} B_{ijc} + sum_{j != i} lambda_j D_{ijc} ]_+

with B = E_{j,c}[v_i . d], D = E_{j,c}[h v_i . d] the SIGNED cross quantities
(so a removal that helps receiver i by taking away a positive contribution to
receiver k shows up as cost), and K the squared norm of what the gate removes,
normalised per sender. The slack is eliminated analytically; what remains is a
box-constrained convex piecewise-quadratic in at most n_task variables per
role, solved by EXACT CYCLIC COORDINATE DESCENT in float64 on the CPU (see
:func:`solve_role`, which also records why projected gradient was dropped).

There is NO broadcast, and none is needed: the solver's inputs are the
all-reduced sums (the module's only collective, in
:meth:`CrossGateStats.reduced`) plus the driver's meta_info, so every rank
solves the same problem, and the descent is deterministic -- same iterate order,
same breakpoints, same float64 arithmetic -- so every rank lands on the same
lambda. A broadcast would be the alternative to that determinism, not an
addition to it.

What follows from the definitions: if every valid receiver's condition is
already met at lambda = 0 (s_i(0) = 0), lambda = 0 is optimal. What does NOT
follow: "a noise reference is inert" -- a finite-sample B can be negative by
chance and the gate's own selection can bias D -- and release is a tendency
under EMA refresh, not a monotone guarantee.

Nothing here needs a second forward or backward. The (bs, T, k) intermediates
live inside :func:`cross_gate_forward` and die there; the reference is a
(task, role, side, vocab) buffer scattered into once per micro-batch and
all-reduced once per step.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from verl.trainer.ppo.sign_weights import ROLE_NAMES

__all__ = [
    "CrossGateConfig",
    "CrossGateController",
    "CrossGateRefs",
    "CrossGateStats",
    "cross_gate_forward",
    "cross_gate_prompt_columns",
    "prompt_side",
    "solve_role",
    "N_SIDES",
    "SPLIT_VERSION",
    "MAX_PROMPTS",
]

N_SIDES = 2
# Bumping this changes every prompt's side. It is part of the hash input so a
# resumed run with a different split scheme is refused rather than blended.
SPLIT_VERSION = 1
# Columns of the per-step prompt bitmap; batch-local dense ids index it.
MAX_PROMPTS = 4096
N_ROLES = len(ROLE_NAMES)
ROLE_ID = {name: rid for rid, name in ROLE_NAMES.items()}

# Why a reference (i, c) is invalid this step. Coded so it can be logged as a
# number and read back; 0 means valid.
INVALID_NONE = 0
INVALID_NEVER_SEEN = 1
INVALID_FEW_PROMPTS = 2
INVALID_FEW_PG_PROMPTS = 3
INVALID_FEW_TOKENS = 4
INVALID_STALE = 5
INVALID_NONFINITE = 6
INVALID_ZERO_R = 7
INVALID_NAMES = {
    INVALID_NONE: "valid",
    INVALID_NEVER_SEEN: "never_seen",
    INVALID_FEW_PROMPTS: "few_prompts",
    INVALID_FEW_PG_PROMPTS: "few_pg_prompts",
    INVALID_FEW_TOKENS: "few_tokens",
    INVALID_STALE: "stale",
    INVALID_NONFINITE: "nonfinite",
    INVALID_ZERO_R: "zero_R",
}


@dataclass
class CrossGateConfig:
    enable: bool = False
    # 1 = the gate as first run: q scaled by sqrt(kappa) through the sqrt(R)
    # denominator, penalty (s/R)^2. 2 = MOPD v2: q = gamma * [negative cosine],
    # penalty (s/S)^2 against the OBSERVED cross magnitude. Selectable so the
    # v1 run stays reproducible from this file; the arms differ in their locks.
    gate_version: int = 1
    # Tolerated NEGATIVE net cross contribution, as a fraction of the receiver's
    # R_{i,c}. 0 makes any net negative contribution the object of control.
    eps_cross: float = 0.0
    # Weight on the squared normalised slack. 1/beta_0^2 corrects the one known
    # scale gap: at eps_cross = 0 the slack is linear in beta while the removal
    # cost is beta-free (design doc §8.2). Not a balance claim.
    rho: float = 1.0e4
    # v2's penalty weight, against s/S instead of s/R. 1.0 because S already IS
    # the scale of the quantity being constrained, so the two terms of the
    # objective are comparable without a conversion factor -- unlike rho above,
    # which exists to bridge beta's scale gap between s/R and K.
    rho_rel: float = 1.0
    # Cap on the per-token attenuation. An experimental condition the constraint
    # is not allowed to override.
    lambda_max: float = 0.2
    # STRENGTH KNOB. Applied to the DIRECTION RESPONSE a_i = min_s[-cos]_+,
    # before gamma and before omega pools it:
    #     a_tilde_i = min(1, q_scale * a_i),  q_tilde_i = gamma_i * a_tilde_i
    # so q_tilde_i <= gamma_i at every strength -- amplifying the response to an
    # opposed direction never amplifies a low RELIABILITY. (min(1, k*gamma*a)
    # would: gamma 0.01 with a 0.5 is q 0.005, and k = 100 makes it 0.5, a
    # decision to trust an unreliable reference dressed as a strength change.)
    # h = sum_i omega_i q_tilde_i still satisfies h <= 1 and is still the
    # omega-weighted mean the receiver attribution divides by.
    #
    # Why here and not on lambda_max: 1 - w = lambda * h_tilde <= lambda_max
    # whatever q_scale is, so the per-token floor (80% of the teacher signal at
    # lambda_max = 0.2) survives any value of this knob, while raising
    # lambda_max toward 1 would let a single token's OPD vanish. In the MEAN the
    # two are the same amplification while the box binds; they differ in the
    # tail, and the tail is what the cap is for.
    #
    # Why it is not a post-hoc multiply on the loss: D = E[h x] and
    # K = E[h^2 d^2] feed the solver, and both are accumulated from the h this
    # returns. Scaling q here re-aggregates them by construction, so the
    # intervention the solver prices is the one that executes. The consequence
    # is that the solver will pull lambda back down where the box is not
    # binding -- the objective is invariant under (D -> kD, K -> k^2 K,
    # lambda -> lambda/k) -- so the realised strength is min(lambda_hat,
    # q_scale * lambda_max) * h, NOT q_scale times the old attenuation. Read
    # actor/cross/attenuation_mean, never q_scale, for what actually happened.
    #
    # It also amplifies a SMALL gamma: q carries gamma as a factor, so a large
    # q_scale weakens the reliability suppression rather than only raising the
    # strength. That is the reason not to jump to a value large enough to
    # saturate everything.
    q_scale: float = 1.0
    # One decay for the references, R, B, D and K's two halves.
    ema_decay: float = 0.8
    # Reference validity, judged over a window of steps, per side.
    window_steps: int = 8
    min_prompts: int = 4
    min_pg_prompts: int = 2
    min_tokens: int = 64
    max_staleness: int = 2
    # Numerical guard only. The zero cases are branched on before it matters.
    delta: float = 1.0e-30
    # The prompt split.
    split_seed: int = 1
    # Roles the gate acts on. Others keep w = 1 and are only observed.
    roles: tuple = ("format", "env_action")
    # Solver.
    solver_iters: int = 500
    solver_tol: float = 1.0e-12

    @classmethod
    def from_mapping(cls, m) -> CrossGateConfig:
        if not m:
            return cls()
        m = dict(m)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        bad = sorted(set(m) - allowed)
        if bad:
            raise ValueError(f"cross_gate: unknown keys {bad}; allowed {sorted(allowed)}")
        kw = {}
        for k, v in m.items():
            if k == "enable":
                kw[k] = bool(v)
            elif k == "roles":
                kw[k] = tuple(str(x) for x in (list(v) if not isinstance(v, str) else v.split(",")))
            elif k in ("gate_version", "window_steps", "min_prompts", "min_pg_prompts",
                       "min_tokens", "max_staleness", "split_seed", "solver_iters"):
                kw[k] = int(v)
            else:
                kw[k] = float(v)
        return cls(**kw)

    def validate(self) -> None:
        if self.eps_cross < 0.0:
            raise ValueError(f"cross_gate.eps_cross must be >= 0, got {self.eps_cross}")
        if self.rho < 0.0:
            raise ValueError(f"cross_gate.rho must be >= 0, got {self.rho}")
        if int(self.gate_version) not in (1, 2):
            raise ValueError(f"cross_gate.gate_version must be 1 or 2, got {self.gate_version}")
        if self.rho_rel < 0.0:
            raise ValueError(f"cross_gate.rho_rel must be >= 0, got {self.rho_rel}")
        if not (0.0 <= self.lambda_max <= 1.0):
            raise ValueError(f"cross_gate.lambda_max must be in [0, 1], got {self.lambda_max}")
        if not (self.q_scale >= 1.0) or not math.isfinite(self.q_scale):
            # Below 1 would be a WEAKER gate wearing a strength knob's name, and
            # the arm it would produce is already available as lambda_max.
            raise ValueError(f"cross_gate.q_scale must be finite and >= 1, got {self.q_scale}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"cross_gate.ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.window_steps < 1 or self.min_prompts < 1 or self.min_tokens < 1:
            raise ValueError("cross_gate.window_steps, min_prompts, min_tokens must be >= 1")
        if self.min_pg_prompts < 0 or self.max_staleness < 0:
            raise ValueError("cross_gate.min_pg_prompts and max_staleness must be >= 0")
        if self.delta <= 0.0:
            raise ValueError("cross_gate.delta must be > 0")
        unknown = [r for r in self.roles if r not in ROLE_ID]
        if unknown:
            raise ValueError(f"cross_gate.roles has unknown roles {unknown}; known {sorted(ROLE_ID)}")
        if self.solver_iters < 1 or self.solver_tol <= 0.0:
            raise ValueError("cross_gate.solver_iters must be >= 1 and solver_tol > 0")

    def control_role_mask(self) -> torch.Tensor:
        m = torch.zeros(N_ROLES, dtype=torch.bool)
        for r in self.roles:
            m[ROLE_ID[r]] = True
        return m


# --------------------------------------------------------------------------- #
# The prompt split (driver side)


def _canonical(anchor) -> str:
    """One string per prompt anchor, whatever container the env used."""
    if anchor is None:
        return ""
    if isinstance(anchor, bytes):
        return anchor.decode("utf-8", errors="replace")
    if isinstance(anchor, np.ndarray):
        return _canonical(anchor.tolist())
    if isinstance(anchor, (list, tuple)):
        return "\x1f".join(_canonical(a) for a in anchor)
    return str(anchor)


def prompt_side(task: str, anchor_text: str, *, seed: int, version: int = SPLIT_VERSION):
    """``(key, side)``: a process-stable key for the prompt and its half.

    SHA-256, not ``hash()``: Python's string hash is salted per process, so a
    fixed seed alone would not give every rank -- or a resumed run -- the same
    split. The rank is deliberately NOT an input.
    """
    h = hashlib.sha256()
    h.update(f"{int(version)}|{int(seed)}|{task}|".encode("utf-8"))
    h.update(anchor_text.encode("utf-8", errors="replace"))
    digest = h.digest()
    key = digest.hex()
    side = int.from_bytes(digest[:8], "big") % N_SIDES
    return key, side


def cross_gate_prompt_columns(
    *,
    uids,
    turn_steps,
    anchors,
    task_names,
    real,
    seed: int,
    version: int = SPLIT_VERSION,
) -> dict:
    """Per-row side and dense prompt index, from the group's turn-0 anchor.

    The training batch carries no stable prompt id: its ``index`` is the row's
    position in the generation batch. What IS stable is the environment's
    initial observation -- the game, the shopping goal, the question -- which
    every rollout of a GRPO group shares (the group's envs are seeded together)
    and which sits on the turn-0 row as ``anchor_obs``. So: group rows by
    ``uid``, take the anchor of any turn-0 row in the group, hash it with the
    task. A group with no turn-0 row in the batch gets side -1: it can still be
    gated (that needs only the receivers' references) but it feeds no reference
    and counts for no prompt.

    ``real`` masks out padding rows (adjust_batch copies, which carry their
    original's uid and must not be counted twice).

    Returns ``side`` (n,) long in {0, 1, -1}, ``prompt_idx`` (n,) long dense in
    0..P-1 or -1, ``keys`` a list of P stable keys (dense index -> key), and
    ``unkeyed_rows`` the number of real rows that got -1.
    """
    n = len(uids)
    uids = [None if u is None else str(u) for u in list(uids)[:n]]
    turn_steps = list(turn_steps)[:n]
    anchors = list(anchors)[:n]
    task_names = list(task_names)[:n]
    real = [bool(r) for r in list(real)[:n]]

    # turn-0 anchor per uid, from the first real turn-0 row seen
    anchor_of: dict = {}
    for i in range(n):
        if not real[i] or uids[i] is None:
            continue
        try:
            t0 = int(turn_steps[i]) == 0
        except (TypeError, ValueError):
            t0 = False
        if t0 and uids[i] not in anchor_of:
            anchor_of[uids[i]] = (str(task_names[i]), _canonical(anchors[i]))

    side = torch.full((n,), -1, dtype=torch.long)
    prompt_idx = torch.full((n,), -1, dtype=torch.long)
    keys: list = []
    dense: dict = {}
    unkeyed = 0
    for i in range(n):
        if not real[i] or uids[i] is None or uids[i] not in anchor_of:
            if real[i]:
                unkeyed += 1
            continue
        task, text = anchor_of[uids[i]]
        key, s = prompt_side(task, text, seed=seed, version=version)
        if key not in dense:
            dense[key] = len(keys)
            keys.append(key)
        d = dense[key]
        if d >= MAX_PROMPTS:
            # More distinct prompts than the bitmap has columns: this row still
            # gets a side (the gate needs none of this), but it cannot be
            # counted for validity. Reported, not hidden.
            side[i] = s
            unkeyed += 1
            continue
        side[i] = s
        prompt_idx[i] = d
    return {"side": side, "prompt_idx": prompt_idx, "keys": keys, "unkeyed_rows": int(unkeyed)}


# --------------------------------------------------------------------------- #
# What the loss reads: the references, on the device, fixed for one step


@dataclass
class CrossGateRefs:
    """Previous-step references in THIS batch's task order, on the device."""
    v: torch.Tensor          # (nT, nR, 2, V) float32 -- E_{i,c}[r] per side
    R: torch.Tensor          # (nT, nR, 2) float32 -- E_{i,c}[||r||^2] per side
    side_w: torch.Tensor     # (nT, nR, 2) float32 -- pooling weights over sides (sum 1 or 0)
    valid: torch.Tensor      # (nT, nR) bool -- both sides usable this step
    lam: torch.Tensor        # (nT, nR) float32 -- lambda applied this step (0 off control roles)
    control_roles: torch.Tensor  # (nR,) bool
    # v2 only, and defaulted so a v1 construction stays valid: the reference
    # norms it divides by, and the two halves' agreement it scales by. When
    # absent they are derived from v (norms) and left at 0 (gamma), which makes
    # a v2 gate inert rather than wrong if someone builds refs by hand.
    v_norm: torch.Tensor | None = None
    gamma: torch.Tensor | None = None

    def __post_init__(self):
        if self.v_norm is None:
            self.v_norm = self.v.double().norm(dim=-1).to(self.R.dtype)
        if self.gamma is None:
            self.gamma = torch.zeros(self.v.shape[0], self.v.shape[1],
                                     dtype=self.R.dtype, device=self.R.device)


def cross_gate_forward(
    *,
    student_topk_logprob: torch.Tensor,
    teacher_topk_logprob: torch.Tensor,
    teacher_kl: torch.Tensor,
    topk_ids: torch.Tensor,
    response_ids: torch.Tensor,
    pg_grad_coef: torch.Tensor | None,
    opd_coef: float,
    task_ids: torch.Tensor,
    roles: torch.Tensor,
    refs: CrossGateRefs,
    delta: float,
    gate_version: int = 1,
    q_scale: float = 1.0,
) -> dict:
    """Per-token gate and everything the step's statistics need, detached.

    Shapes: log-probs and ids ``(bs, T, k)``; ``teacher_kl``, ``response_ids``,
    ``roles`` ``(bs, T)``; ``pg_grad_coef`` ``(bs, T)`` or None (pure
    distillation: r = 0 everywhere, the gate still runs); ``task_ids`` ``(bs,)``.

    Returns (all detached):
      ``w`` (bs, T)            1 - lambda h, the OPD weight
      ``h`` (bs, T)            the soft gate
      ``q`` (bs, T, nT)        per-receiver gate before averaging (0 where not valid)
      ``omega`` (bs, T, nT)    the receiver weights actually used (0/1 over valid, / n_valid)
      ``x`` (bs, T, nT)        pooled-reference inner product v_i . d (signed)
      ``d_sq``, ``r_sq``, ``self_dot`` (bs, T)
      ``in_support`` (bs, T)   sampled id inside the support (the reference population)
      ``r`` (bs, T, k)         the PG direction on the support (for the reference scatter)
      ``d`` (bs, T, k)         the OPD direction on the support
      ``overlap`` (bs, T, nT)  share of the token's support where v_i is non-zero
      ``energy_shared`` (bs, T, nT)  sum of d^2 over that shared support
      ``lam_t`` (bs, T)        lambda at the token
    """
    with torch.no_grad():
        dt = torch.promote_types(student_topk_logprob.dtype, torch.float32)
        lp_s = student_topk_logprob.detach().to(dt)
        lp_t = teacher_topk_logprob.detach().to(dt)
        D = teacher_kl.detach().to(dt).unsqueeze(-1)
        bs, T, k = lp_s.shape
        nT, nR, nS, V = refs.v.shape
        dev = lp_s.device

        p_s = lp_s.exp()
        # OPD descent direction on the support, beta folded in, lambda not.
        d = float(opd_coef) * p_s * (D - (lp_s - lp_t))
        d_sq = (d * d).sum(dim=-1)
        d_norm = d_sq.clamp(min=0.0).sqrt()

        hit = (topk_ids == response_ids.unsqueeze(-1))
        in_support = hit.any(dim=-1)
        hit_f = hit.to(dt)
        if pg_grad_coef is None:
            pg_scale = torch.zeros(bs, T, device=dev, dtype=dt)
        else:
            # Descent convention: positive pushes the sampled logit up.
            pg_scale = -pg_grad_coef.detach().to(dt)
        r = pg_scale.unsqueeze(-1) * (hit_f - p_s)
        # A token whose sampled id is outside the support has no e_a on it;
        # it is not part of the reference population, so its r is zeroed here
        # rather than left as a -p-only vector that would bias the mean.
        r = r * in_support.unsqueeze(-1).to(dt)
        r_sq = (r * r).sum(dim=-1)
        self_dot = (r * d).sum(dim=-1)

        # --- gather the references at the token's support ------------------
        tid = task_ids.reshape(-1)
        tid = tid.round().to(torch.long) if tid.is_floating_point() else tid.to(torch.long)
        tid_ok = (tid >= 0) & (tid < nT)
        tid_c = tid.clamp(min=0, max=max(nT - 1, 0))
        rol = roles.to(torch.long).clamp(min=0, max=nR - 1)                       # (bs, T)
        ids = topk_ids.to(torch.long).clamp(min=0, max=V - 1)                     # (bs, T, k)
        vflat = refs.v.reshape(-1)
        Rr = refs.R                                                                # (nT, nR, 2)
        VNr = refs.v_norm                                                          # (nT, nR, 2)
        sw = refs.side_w                                                           # (nT, nR, 2)

        x_side = torch.zeros(bs, T, nT, nS, device=dev, dtype=dt)
        nonzero_share = torch.zeros(bs, T, nT, device=dev, dtype=dt)
        energy_shared = torch.zeros(bs, T, nT, device=dev, dtype=dt)
        R_tok = torch.zeros(bs, T, nT, nS, device=dev, dtype=dt)
        v_norm_tok = torch.zeros(bs, T, nT, nS, device=dev, dtype=dt)
        for i in range(nT):
            for s in range(nS):
                base = ((i * nR + rol) * nS + s) * V                                # (bs, T)
                vg = vflat[base.unsqueeze(-1) + ids]                                # (bs, T, k)
                x_side[:, :, i, s] = (vg * d).sum(dim=-1)
                R_tok[:, :, i, s] = Rr[i][rol, s]
                v_norm_tok[:, :, i, s] = VNr[i][rol, s]
                nz = (vg != 0).to(dt)
                # pooled support overlap / energy, weighted by the side weights
                wgt = sw[i][rol, s].unsqueeze(-1)                                  # (bs, T, 1)
                nonzero_share[:, :, i] += (nz.mean(dim=-1) * wgt.squeeze(-1))
                energy_shared[:, :, i] += ((nz * d * d).sum(dim=-1) * wgt.squeeze(-1))
        # pooled (signed) inner product: sum_s side_w * x_s. Linear in v, so this
        # IS the inner product with the pooled reference. The pooling weight is
        # the RECEIVER's (i, role), which is why it is built per receiver.
        x = torch.zeros(bs, T, nT, device=dev, dtype=dt)
        for i in range(nT):
            for s in range(nS):
                x[:, :, i] += x_side[:, :, i, s] * sw[i][rol, s]

        # --- the soft gate --------------------------------------------------
        # q per side, then the weaker: zero unless BOTH sides oppose.
        neg = (-x_side).clamp(min=0.0)                                             # (bs, T, nT, nS)
        if int(gate_version) >= 2:
            # v2. The denominator is the reference's own norm, so q is a NEGATIVE
            # COSINE: bounded by 1 through Cauchy-Schwarz whatever the support's
            # diversity. v1 divided by sqrt(R) instead, which bounds q by
            # sqrt(kappa) = ||v||/sqrt(R) and so shrank the gate for a reference
            # merely spread over many vocabulary items -- diversity, not
            # unreliability. Reliability is carried separately, by gamma.
            scale_tok = v_norm_tok
        else:
            scale_tok = R_tok.clamp(min=0.0).sqrt()
        denom = scale_tok * d_norm.unsqueeze(-1).unsqueeze(-1) + float(delta)
        q_side = neg / denom
        q = q_side.min(dim=-1).values                                              # (bs, T, nT)
        # Zero out where the side has no reference to speak of: the formula would
        # divide by delta and manufacture a gate out of nothing. Branched on the
        # same quantity the denominator uses.
        q = torch.where((scale_tok > 0).all(dim=-1), q, torch.zeros_like(q))

        # THE STRENGTH KNOB, applied here: to a_i = min_s [-cos(v_i^s, d)]_+, the
        # DIRECTION response, BEFORE gamma multiplies it.
        #
        #     a_tilde_i = min(1, q_scale * a_i),   q_tilde_i = gamma_i a_tilde_i
        #
        # so q_tilde_i <= gamma_i at every strength: amplifying the response to
        # an opposed direction never amplifies a LOW RELIABILITY. Applying it
        # after gamma instead -- min(1, k gamma a) -- would: gamma = 0.01 with a
        # = 0.5 gives q = 0.005, and k = 100 turns that into 0.5, which is not a
        # strength change but a decision to trust an unreliable reference. In v1
        # there is no gamma and the two placements coincide.
        #
        # Still before omega pools q into h, which is what keeps D = E[h x] and
        # K = E[h^2 d^2] re-aggregated from the SAME gate the loss applies, and
        # keeps h the omega-weighted mean that lost_by_recv divides by.
        if float(q_scale) != 1.0:
            q = (q * float(q_scale)).clamp(min=0.0, max=1.0)

        if int(gate_version) >= 2:
            # gamma_{i,c}: the two prompt-disjoint halves' agreement. Opposed
            # halves give 0 and the receiver is inert; agreeing halves that both
            # oppose the sender's OPD are what the gate acts on.
            gam = torch.stack([refs.gamma.to(dev)[i][rol] for i in range(nT)], dim=-1)
            q = q * gam.clamp(min=0.0, max=1.0)
        # THE LOAD-BEARING GUARD. w = 1 - lambda*h only attenuates while
        # h in [0, 1], and h inherits that from q: the [.]_+ above is exactly
        # redundant with this line's lower clamp (min_k [a_k]_+ == [min_k a_k]_+
        # for every a), and the clamp on h below is redundant with its upper one
        # (h is a convex combination of the q's). So the other two can be removed
        # with no observable effect and this one cannot -- do not "simplify" it
        # away by analogy with them. The upper clamp is not implied by Jensen
        # either: q <= sqrt(kappa) <= 1 holds only while v and R are updated over
        # the same population with the same weights, which a hand-set or
        # part-resumed reference need not respect.
        # tests/trainer/test_opd_cross_gate.py section 6 asserts the invariant.
        q = q.clamp(min=0.0, max=1.0)

        # valid receivers: (i, c(t)) usable, i != sender, and the role is controlled
        valid_ic = refs.valid.to(dev)                                              # (nT, nR)
        ctrl = refs.control_roles.to(dev)[rol]                                     # (bs, T)
        recv_ok = torch.stack([valid_ic[i][rol] for i in range(nT)], dim=-1)       # (bs, T, nT)
        # i != sender's task
        sender = tid_c.unsqueeze(-1).unsqueeze(-1).expand(bs, T, 1)
        not_self = torch.ones(bs, T, nT, device=dev, dtype=torch.bool)
        not_self.scatter_(2, sender, False)
        recv_ok = recv_ok & not_self & ctrl.unsqueeze(-1) & tid_ok.view(bs, 1, 1)
        n_valid = recv_ok.to(dt).sum(dim=-1)                                       # (bs, T)
        omega = recv_ok.to(dt) / n_valid.clamp(min=1.0).unsqueeze(-1)
        q = q * recv_ok.to(dt)
        h = (omega * q).sum(dim=-1).clamp(min=0.0, max=1.0)

        lam_t = refs.lam.to(dev)[tid_c.unsqueeze(-1).expand(bs, T), rol]          # (bs, T)
        lam_t = torch.where(tid_ok.view(bs, 1) & ctrl, lam_t, torch.zeros_like(lam_t))
        w = 1.0 - lam_t * h

        return {
            "w": w, "h": h, "q": q, "omega": omega, "x": x,
            "d_sq": d_sq, "r_sq": r_sq, "self_dot": self_dot,
            "in_support": in_support.to(dt), "r": r, "d": d,
            "overlap": nonzero_share, "energy_shared": energy_shared,
            "lam_t": lam_t, "n_valid": n_valid,
        }


# --------------------------------------------------------------------------- #
# Per-step accumulators (device), one all-reduce each

_SIDE_COLS = ("n_tok", "r_sq_sum")
_SIDE_IDX = {n: i for i, n in enumerate(_SIDE_COLS)}
_CROSS_COLS = (
    "n_tok",           # tokens of (j, c) the receiver i was evaluated on
    "B_sum",           # sum v_i . d            (signed)
    "D_sum",           # sum h v_i . d          (signed)
    # D SPLIT BY THE SIGN OF THE CROSS EFFECT, and D_sum = Dpos_sum - Dneg_sum.
    # D is what the solver prices attenuation by, and it pools two opposite
    # things at token level: where x_i < 0 the attenuation HELPS receiver i,
    # where x_i > 0 (a token some OTHER receiver raised h on) it destroys a
    # POSITIVE transfer to i. A net D that looks like help can still be paying
    # for it out of transfer that was working. Only the split can say so:
    #   lost positive    = sum_j lambda_j Dpos_ij
    #   removed negative = sum_j lambda_j Dneg_ij
    # Diagnostics in this version -- NOT a constraint. A "lose no positive
    # transfer at all" rule would stop nearly every intervention whose
    # receivers disagree in sign, which is the road back to a mechanism that
    # does nothing. And these are OUTPUT-SPACE proxies: Dpos > 0 is not
    # evidence that useful knowledge was transferred.
    "Dpos_sum",        # sum h [v_i . d]_+
    "Dneg_sum",        # sum h [-v_i . d]_+
    "Aneg_sum",        # sum [-v_i . d]_+       (diagnostic only)
    "realized_sum",    # sum w v_i . d          (after the gate applied this step)
    "lost_by_recv",    # sum (omega_i q_i / h) (1 - w^2) ||d||^2  -- receiver i's share of removed strength
    "overlap_sum",     # sum share of support where v_i != 0
    "energy_shared_sum",
)
_CROSS_IDX = {n: i for i, n in enumerate(_CROSS_COLS)}
_SEND_COLS = (
    "n_tok",
    "K_num",           # sum h^2 ||d||^2
    "d_sq_sum",        # sum ||d||^2
    "kl_sum",
    "kl_lost",         # sum (1 - w) KL
    "strength_lost",   # sum (1 - w^2) ||d||^2
    "strength_lost_self_aligned",   # ... restricted to r . d > 0
    # THE SENDER'S OWN COST, which nothing in the objective sees. T_j =
    # E[h r_j . d_j] is the OPD component aligned with this task's OWN RL that
    # the attenuation is about to remove: at lambda_j the predicted self-cost is
    # lambda_j T_j. Split by sign because the signed mean pools "removing OPD
    # that pushes against this task's RL" (sd < 0, a gain) with "removing OPD
    # that agrees with it" (sd > 0, a loss), and strength_lost_self_aligned
    # already shows those are both present -- it is a MAGNITUDE share, so it
    # cannot give the sign of the net.
    #
    # Measured, not constrained, and deliberately NOT wired to token selection:
    # choosing tokens by self-conflict is the old self gate, which this arm
    # replaced. The cross gate still picks the tokens; this only prices what
    # picking them costs the sender.
    "T_num",           # sum h (r . d)          (signed)
    "T_pos",           # sum h [r . d]_+
    "T_neg",           # sum h [-(r . d)]_+
    "h_sum",
    "h_nonzero",
    "w_sum",
    "n_valid_recv_sum",
    "n_unkeyed",       # tokens on rows whose side is -1
    # The TAIL of the attenuation, not just its mean. 1 - w = lambda * h, so at
    # a fixed lambda these three are quantile crossings of the attenuation
    # itself: h >= 1 is the token sitting at the lambda_max cap. A strength
    # experiment that moves the mean by raising q_scale moves these too, and by
    # more -- the 80% per-token floor is unchanged but the number of tokens
    # pressed against it is not, so "same floor" is not "same risk".
    "h_ge_half",       # tokens with h >= 0.5   (1 - w >= lambda/2)
    "h_ge_9_10",       # tokens with h >= 0.9
    "h_sat",           # tokens with h >= 1 - 1e-6, i.e. attenuated at lambda
)
_SEND_IDX = {n: i for i, n in enumerate(_SEND_COLS)}


class CrossGateStats:
    """Sums for one update, on the device, reduced once. Built on the config
    alone so every rank runs the same collectives whatever its batch holds."""

    def __init__(self, n_tasks: int, vocab_size: int, device, n_prompts: int = MAX_PROMPTS):
        self.n_tasks, self.V = int(n_tasks), int(vocab_size)
        self.n_prompts = int(n_prompts)
        nT, nR, nS = self.n_tasks, N_ROLES, N_SIDES
        # float32: entries are O(1) contributions into O(1e5) cells; the float64
        # the scalar tables use would double a 22 MB buffer for no gain.
        self.ref_sum = torch.zeros(nT, nR, nS, self.V, dtype=torch.float32, device=device)
        self.side = torch.zeros(nT, nR, nS, len(_SIDE_COLS), dtype=torch.float64, device=device)
        self.pbm_contrib = torch.zeros(nT, nR, nS, self.n_prompts, dtype=torch.float32, device=device)
        self.pbm_pg = torch.zeros(nT, nR, nS, self.n_prompts, dtype=torch.float32, device=device)
        self.cross = torch.zeros(nT, nT, nR, len(_CROSS_COLS), dtype=torch.float64, device=device)
        self.send = torch.zeros(nT, nR, len(_SEND_COLS), dtype=torch.float64, device=device)

    def update(
        self,
        *,
        fwd: dict,
        task_ids: torch.Tensor,
        roles: torch.Tensor,
        response_mask: torch.Tensor,
        teacher_kl: torch.Tensor,
        topk_ids: torch.Tensor,
        side: torch.Tensor | None,
        prompt_idx: torch.Tensor | None,
        row_basis: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            dev = self.ref_sum.device
            nT, nR, nS = self.n_tasks, N_ROLES, N_SIDES
            bs, T = response_mask.shape
            mask = response_mask.to(torch.float32)
            if row_basis is not None:
                # duplicated padding rows carry basis 0 and are not evidence
                mask = mask * (row_basis.detach().reshape(-1).to(torch.float32) > 0).to(torch.float32).unsqueeze(-1)
            tid = task_ids.reshape(-1)
            tid = tid.round().to(torch.long) if tid.is_floating_point() else tid.to(torch.long)
            tid_ok = (tid >= 0) & (tid < nT)
            mask = mask * tid_ok.to(torch.float32).unsqueeze(-1)
            tid_c = tid.clamp(min=0, max=max(nT - 1, 0))
            rol = roles.to(torch.long).clamp(min=0, max=nR - 1)
            kl = teacher_kl.detach().to(torch.float32) * mask

            w, h, q, om, x = fwd["w"], fwd["h"], fwd["q"], fwd["omega"], fwd["x"]
            d_sq, r_sq, sd = fwd["d_sq"], fwd["r_sq"], fwd["self_dot"]
            in_sup = fwd["in_support"] * mask
            r, d_vec = fwd["r"], fwd["d"]

            # ---- reference population: (task, role, side) ------------------
            if side is not None:
                sd_row = side.reshape(-1).to(torch.long)
            else:
                sd_row = torch.full((bs,), -1, dtype=torch.long, device=dev)
            side_ok = (sd_row >= 0) & (sd_row < nS)
            sdc = sd_row.clamp(min=0, max=nS - 1)
            # float32 from here on whatever the forward's dtype was (float64 in
            # the CPU tests): the tables are float64 and the bitmaps float32.
            pop = (in_sup * side_ok.to(torch.float32).unsqueeze(-1)).to(torch.float32)   # (bs, T)
            r_sq = r_sq.to(torch.float32)
            # cell id per token for the (task, role, side) tables
            cell = (tid_c.unsqueeze(-1) * nR + rol) * nS + sdc.unsqueeze(-1)     # (bs, T)
            ncell = nT * nR * nS
            self.side.view(ncell, -1)[:, _SIDE_IDX["n_tok"]].index_add_(
                0, cell.reshape(-1), pop.reshape(-1).to(torch.float64))
            self.side.view(ncell, -1)[:, _SIDE_IDX["r_sq_sum"]].index_add_(
                0, cell.reshape(-1), (r_sq * pop).reshape(-1).to(torch.float64))
            # scatter r over the vocabulary
            k = topk_ids.shape[-1]
            ids = topk_ids.to(torch.long).clamp(min=0, max=self.V - 1)
            flat_idx = (cell.unsqueeze(-1) * self.V + ids).reshape(-1)
            vals = (r * pop.unsqueeze(-1)).reshape(-1).to(torch.float32)
            self.ref_sum.view(-1).index_add_(0, flat_idx, vals)
            # prompt bitmaps
            if prompt_idx is not None:
                pidx = prompt_idx.reshape(-1).to(torch.long)
                p_ok = side_ok & (pidx >= 0) & (pidx < self.n_prompts) & tid_ok
                if bool(p_ok.any()):
                    # contributed: any population token in (row, role)
                    contrib = torch.zeros(bs, nR, device=dev, dtype=torch.float32)
                    contrib.scatter_add_(1, rol, pop)
                    pg = torch.zeros(bs, nR, device=dev, dtype=torch.float32)
                    pg.scatter_add_(1, rol, pop * (r_sq > 0).to(torch.float32))
                    rows = torch.nonzero(p_ok).reshape(-1)
                    for rr in range(nR):
                        c_rows = rows[contrib[rows, rr] > 0]
                        if c_rows.numel():
                            self.pbm_contrib[tid_c[c_rows], rr, sdc[c_rows], pidx[c_rows]] = 1.0
                        g_rows = rows[pg[rows, rr] > 0]
                        if g_rows.numel():
                            self.pbm_pg[tid_c[g_rows], rr, sdc[g_rows], pidx[g_rows]] = 1.0

            # ---- sender table: (j, c) ----------------------------------------
            scell = tid_c.unsqueeze(-1) * nR + rol                                # (bs, T)
            nsc = nT * nR
            def _add_send(name, val):
                self.send.view(nsc, -1)[:, _SEND_IDX[name]].index_add_(
                    0, scell.reshape(-1), (val * mask).reshape(-1).to(torch.float64))
            one = torch.ones_like(mask)
            _add_send("n_tok", one)
            _add_send("K_num", h * h * d_sq)
            _add_send("d_sq_sum", d_sq)
            _add_send("kl_sum", kl)
            _add_send("kl_lost", (1.0 - w) * kl)
            _add_send("strength_lost", (1.0 - w * w) * d_sq)
            _add_send("strength_lost_self_aligned", (1.0 - w * w) * d_sq * (sd > 0).to(torch.float32))
            _add_send("T_num", h * sd)
            _add_send("T_pos", h * sd.clamp(min=0.0))
            _add_send("T_neg", h * (-sd).clamp(min=0.0))
            _add_send("h_sum", h)
            _add_send("h_nonzero", (h > 0).to(torch.float32))
            _add_send("h_ge_half", (h >= 0.5).to(torch.float32))
            _add_send("h_ge_9_10", (h >= 0.9).to(torch.float32))
            _add_send("h_sat", (h >= 1.0 - 1e-6).to(torch.float32))
            _add_send("w_sum", w)
            _add_send("n_valid_recv_sum", fwd["n_valid"])
            _add_send("n_unkeyed", (~side_ok).to(torch.float32).unsqueeze(-1).expand(bs, T) * one)

            # ---- cross table: (i, j, c) -------------------------------------
            ncc = nT * nT * nR
            removed = (1.0 - w * w) * d_sq                                        # (bs, T)
            h_safe = h.clamp(min=1e-30)
            for i in range(nT):
                ccell = (i * nT + tid_c.unsqueeze(-1)) * nR + rol                 # (bs, T)
                xi, qi, oi = x[:, :, i], q[:, :, i], om[:, :, i]
                def _add_cross(name, val):
                    self.cross.view(ncc, -1)[:, _CROSS_IDX[name]].index_add_(
                        0, ccell.reshape(-1), (val * mask).reshape(-1).to(torch.float64))
                _add_cross("n_tok", one)
                _add_cross("B_sum", xi)
                _add_cross("D_sum", h * xi)
                _add_cross("Dpos_sum", h * xi.clamp(min=0.0))
                _add_cross("Dneg_sum", h * (-xi).clamp(min=0.0))
                _add_cross("Aneg_sum", (-xi).clamp(min=0.0))
                _add_cross("realized_sum", w * xi)
                _add_cross("lost_by_recv", (oi * qi / h_safe) * removed * (h > 0).to(torch.float32))
                _add_cross("overlap_sum", fwd["overlap"][:, :, i])
                _add_cross("energy_shared_sum", fwd["energy_shared"][:, :, i])

    def reduced(self) -> dict:
        """All-rank sums, on the CPU. One collective per buffer, all ranks."""
        out = {}
        for name in ("ref_sum", "side", "pbm_contrib", "pbm_pg", "cross", "send"):
            buf = getattr(self, name)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                buf = buf.clone()
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            out[name] = buf.detach().cpu()
        return out


# --------------------------------------------------------------------------- #
# The per-role optimisation


def solve_role(
    K: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    valid_recv: np.ndarray,
    *,
    eps: float,
    rho: float,
    scale: np.ndarray | None = None,
    lam_max: float,
    delta: float,
    iters: int = 500,
    tol: float = 1e-12,
    gamma: np.ndarray | None = None,
) -> dict:
    """min sum_j K_j l_j^2 + rho sum_i g_i (s_i(l)/(scale_i+delta))^2, 0 <= l <= lam_max.

    ``scale`` defaults to ``R`` (v1: the receiver's own reward energy, so the
    rule reads "leave a cross effect alone while it is small against RL"). v2
    passes S = sum_{j!=i} E_{j,c}|v_i . d| -- the OBSERVED cross magnitude -- so
    the rule reads "control by what share of the cross effect is net conflict".
    That is a change of control basis, not a unit fix: a cross effect that is
    absolutely tiny but relatively adverse becomes an object of control.

    ``gamma`` (n,) is the RECEIVER'S RELIABILITY, the same [cos(v^1, v^2)]_+ the
    gate multiplies q by, and it weights that receiver's demand here. Default
    ones, which is v1 (no gamma exists there) and is also what every caller
    that predates this argument gets. It sits OUTSIDE the square on purpose:
    inside the numerator it would weight by gamma^2, a different design. This
    is a design choice -- "how much of a receiver's demand to believe, given how
    far its two prompt-disjoint halves agree" -- not a derived optimum. Before
    it, a reference too unreliable to attenuate a single token (gamma = 0 makes
    q = 0) still carried its full B, D and S into the decision of WHOSE lambda
    to raise.

    ``K`` (n,), ``B``/``D`` (n_i, n_j) with the sender on the second axis,
    ``R`` (n,), ``valid_recv`` (n,) bool. Diagonal entries of B and D are
    ignored (a task is never its own receiver). Returns ``lam`` (n,), the slack
    ``s`` (n,) at the solution, ``s0`` at lambda = 0, ``f``, the optimality
    certificate ``opt_gap``, the gradient, the partials of any coordinate
    sitting at lambda_max, ``converged`` and ``reason``.

    SOLVED BY ACTIVE-SET ENUMERATION FOR A STARTING POINT, THEN EXACT CYCLIC
    COORDINATE DESCENT, AND ACCEPTED ONLY ON A CERTIFICATE.

    The objective is convex and C^1: each s_i is a positive part, so s_i^2 is
    continuously differentiable (d/dx [x]_+^2 = 2[x]_+), and K >= 0. On a fixed
    active set A = {i : s_i > 0} it is a plain convex quadratic, so the problem
    is a finite union of box-constrained least-squares problems -- 2^n active
    sets by 3^n box faces, 216 of them at n = 3.

    ENUMERATION ALONE IS NOT ENOUGH, and that is measured rather than assumed.
    Scoring every (active set, face) candidate on the true objective looks like
    it must return the optimum, since the optimum's own pair is among them. It
    does not, because the per-face least-squares solve is ILL-CONDITIONED
    exactly where this problem lives: a valid receiver with S = 0 gives
    w = rho/delta^2 ~ 1e60, its rows enter at 1e30, and lstsq's relative
    singular-value cutoff then discards the sqrt(K) ~ 1e-4 rows as noise -- the
    removal cost drops out of the solve. On 300 random problems drawn at the
    conditioning the live run shows, enumeration alone returned a point WORSE
    than a 41^3 grid search on one of them, with a relative certificate gap up
    to 3e10. Column scaling improves this and does not fix it.

    Coordinate descent does not have that failure mode: each step is a 1-D
    exact minimisation of a convex piecewise quadratic -- its breakpoints are
    where a receiver's slack crosses zero, and each piece has a closed-form
    minimiser -- so no matrix is ever inverted. That is why the OLD solver
    reached the optimum on all 300 saved steps despite its broken stopping test.
    So enumeration is kept for what it is good at, a globally informed starting
    point that a cyclic path can otherwise need many sweeps to reach, and the
    descent does the conditioning-sensitive work. Neither is trusted: the
    certificate decides.

    WHY NOT THE OLD CYCLIC DESCENT PLUS POLISH. Two defects, both found by
    re-solving the 150 saved steps against an independent minimiser:

      - ``converged`` meant "the polish improved the objective by more than
        1e-18", not "this point is optimal". On a fixture with the iteration
        budget cut to one, it returned converged=True on a point 27.2% above
        the optimum.
      - the polish step was 1/L with L built from ``min(scale)`` over ALL
        receivers, INVALID ONES INCLUDED. One receiver with S = 0 -- which is
        the ordinary state of a task whose reference is not usable -- put
        1/(0+1e-30)^2 in L and drove the step to zero. That was live: in 150 of
        the 300 saved (step, role) cases the polish could not move at all.
        Enumeration has no step size, and invalid receivers are dropped from
        ``wgt`` before any reduction, so neither can recur.

    The old code blamed the 0.5% stall on non-smoothness ("a coordinate-wise
    optimum need not be a stationary point"). That reasoning was wrong -- the
    objective is differentiable and coordinate-wise optimality of a convex C^1
    function on a box IS global optimality. The stall was numerical, in the
    stopping test or the iteration budget, and the exact cause is not
    established here; enumeration removes the question rather than answering it.

    ``iters`` is accepted and ignored: there is no iteration to budget. ``tol``
    is now a RELATIVE optimality tolerance on ``opt_gap``, not a displacement
    threshold.
    """
    n = int(K.shape[0])
    K = np.asarray(K, dtype=np.float64).reshape(n)
    B = np.asarray(B, dtype=np.float64).reshape(n, n)
    D = np.asarray(D, dtype=np.float64).reshape(n, n)
    R = np.asarray(R, dtype=np.float64).reshape(n)
    sc = R if scale is None else np.asarray(scale, dtype=np.float64).reshape(n)
    sc = np.asarray(sc, dtype=np.float64).reshape(n)
    valid = np.asarray(valid_recv, dtype=bool).reshape(n)
    gam = np.ones(n, dtype=np.float64) if gamma is None else np.asarray(gamma, dtype=np.float64).reshape(n)
    off = ~np.eye(n, dtype=bool)
    Bo, Do = B * off, D * off
    lam0 = np.zeros(n)
    # the slack's constant part: eps * R stays on the slack (the tolerance is a
    # fraction of R by definition); only the PENALTY's denominator moves to `scale`.
    a0 = -eps * R - Bo.sum(axis=1)

    # Per-receiver penalty weight. An INVALID receiver is zeroed here and so
    # never reaches any reduction -- and its `scale` is replaced before the
    # division, because np.where evaluates both branches and 0/0 would seed a
    # nan that a later multiply by zero does not clear.
    den = np.where(valid, sc, 1.0) + delta
    wgt = np.where(valid, rho * np.maximum(gam, 0.0) / (den * den), 0.0)

    def slack(l):
        return np.maximum(a0 + Do @ l, 0.0) * valid

    def f(l):
        s = slack(l)
        return float((K * l * l).sum() + (wgt * s * s).sum())

    def grad(l):
        s = slack(l)
        return 2.0 * K * l + Do.T @ (2.0 * wgt * s)

    def _kkt_res(l, g=None):
        """How far lambda is from optimal, IN LAMBDA UNITS.

        The projected-gradient residual with each coordinate's own curvature
        alpha_j = K_j + sum_i w_i D_ij^2 as the step:

            r_j = l_j - clip(l_j - g_j / alpha_j, 0, lam_max)

        Zero exactly at a KKT point of the box problem, and -- because the
        gradient is divided by the curvature -- it is a DISPLACEMENT, so it
        carries lambda's units and is comparable across the eight orders of
        magnitude w spans here. The objective gap is reported too, but it is
        not what convergence is judged on: at w ~ 1e60 the gap's natural scale
        is ~1e60 as well, so a threshold relative to it passes anything.
        """
        g = grad(l) if g is None else g
        # alpha_j = K_j + sum_i w_i D_ij^2. The weight indexes the RECEIVER, so
        # it has to be broadcast down the rows -- `wgt * (Do * Do)` would scale
        # column j by w_j instead, which understates the curvature of any
        # coordinate whose receivers are not itself and turns a converged point
        # into a false KKT violation.
        alpha = K + (wgt[:, None] * (Do * Do)).sum(axis=0)
        step = np.where(alpha > 0.0, g / np.where(alpha > 0.0, alpha, 1.0), 0.0)
        r = l - np.clip(l - step, 0.0, lam_max)
        return float(np.max(np.abs(r))) if np.isfinite(r).all() else float("inf")

    def _out(lam, converged, reason, extra_gap=None):
        g = grad(lam)
        # THE CERTIFICATE. For a convex f on a box, the linearisation gap
        #   f(l) - min_{u in box} [f(l) + g.(u - l)]
        #     = sum_j [g_j]_+ l_j + sum_j [-g_j]_+ (lam_max - l_j)
        # is >= f(l) - f* and vanishes exactly at a KKT point. Non-negative by
        # construction, so it is reported as-is rather than as an absolute
        # value, and it is what `converged` is judged on.
        gap = float((np.maximum(g, 0.0) * lam).sum()
                    + (np.maximum(-g, 0.0) * (lam_max - lam)).sum()) if extra_gap is None else extra_gap
        cap = np.where(lam >= lam_max - 1e-15, g, np.nan)
        return {"lam": lam, "s": slack(lam), "s0": slack(lam0), "converged": converged,
                "reason": reason, "f": f(lam), "rho": float(rho),
                "scale": np.array(sc, dtype=np.float64),
                "gamma": np.array(gam, dtype=np.float64),
                "opt_gap": gap, "kkt_res": _kkt_res(lam, g), "grad": g, "cap_partials": cap}

    finite_in = (np.isfinite(K).all() and np.isfinite(Bo).all() and np.isfinite(Do).all()
                 and np.isfinite(R).all() and np.isfinite(wgt).all() and np.isfinite(gam).all())
    if not finite_in:
        return {"lam": lam0, "s": slack(lam0), "s0": slack(lam0), "converged": False,
                "reason": "nonfinite_input", "f": float("nan"), "rho": float(rho),
                "scale": np.array(sc, dtype=np.float64), "gamma": np.array(gam, dtype=np.float64),
                "opt_gap": float("nan"), "kkt_res": float("nan"),
                "grad": np.full(n, np.nan), "cap_partials": np.full(n, np.nan)}
    if not valid.any() or lam_max <= 0.0:
        return _out(lam0, True, "no_valid_receiver" if not valid.any() else "lambda_max_zero", extra_gap=0.0)

    # ---- the enumeration -------------------------------------------------
    # rows of the least-squares problem, per active set:
    #   sqrt(K_j) * l_j                      (n rows, the removal cost)
    #   sqrt(w_i) * (a0_i + (Do l)_i)        (one row per ACTIVE receiver)
    sqrtK = np.sqrt(np.maximum(K, 0.0))
    sqrtW = np.sqrt(np.maximum(wgt, 0.0))
    faces = list(itertools.product((0, 1, 2), repeat=n))       # 0 -> 0, 1 -> free, 2 -> lam_max
    best_lam, best_f = lam0.copy(), f(lam0)
    for bits in range(1 << n):
        act = np.array([bool((bits >> i) & 1) and valid[i] for i in range(n)], dtype=bool)
        rows_pen = np.nonzero(act)[0]
        # M l = y  in the least-squares sense
        M = np.zeros((n + rows_pen.size, n), dtype=np.float64)
        y = np.zeros(n + rows_pen.size, dtype=np.float64)
        M[:n, :] = np.diag(sqrtK)
        for r, i in enumerate(rows_pen):
            M[n + r, :] = sqrtW[i] * Do[i, :]
            y[n + r] = -sqrtW[i] * a0[i]
        for face in faces:
            free = [j for j in range(n) if face[j] == 1]
            lam = np.array([0.0 if face[j] == 0 else (lam_max if face[j] == 2 else 0.0)
                            for j in range(n)], dtype=np.float64)
            if free:
                fixed = [j for j in range(n) if face[j] != 1]
                rhs = y - (M[:, fixed] @ lam[fixed] if fixed else 0.0)
                try:
                    sol, *_ = np.linalg.lstsq(M[:, free], rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.isfinite(sol).all():
                    continue
                # clipped to stay feasible: every candidate is scored on the
                # TRUE objective, so an out-of-box face minimiser can only ever
                # contribute a worse feasible point, never a wrong answer
                lam[free] = np.clip(sol, 0.0, lam_max)
            fc = f(lam)
            if not np.isfinite(fc):
                continue
            # smaller lambda wins a tie, as in every earlier version
            if fc < best_f - 1e-18 or (abs(fc - best_f) <= 1e-18 and lam.sum() < best_lam.sum()):
                best_lam, best_f = lam, fc

    # ---- exact cyclic coordinate descent from there -----------------------
    # Along one coordinate the objective is a convex piecewise quadratic whose
    # breakpoints are the lambda_j at which some receiver's slack crosses zero.
    # Each piece has a closed-form minimiser, so the 1-D step is exact and
    # needs no matrix -- which is the whole reason this stage exists next to the
    # enumeration. Sweeps run until the CERTIFICATE is met, not until the
    # iterate stops moving: at rho/S^2 ~ 1e8 a displacement test fires while
    # the objective is still falling, which is what let the old solver report
    # convergence 27.2% above the optimum.
    lam = best_lam.copy()
    for _ in range(max(int(iters), 1)):
        for j in range(n):
            dj = Do[:, j]
            touch = valid & off[:, j] & (dj != 0.0) & (wgt > 0.0)
            # the slack's affine part with lambda_j's own contribution removed
            a = a0 + Do @ lam - dj * lam[j]
            pts = {0.0, float(lam_max)}
            for i in np.nonzero(touch)[0]:
                bp = -a[i] / dj[i]
                if 0.0 < bp < lam_max:
                    pts.add(float(bp))
            best_x, best_fx = lam[j], None
            for lo, hi in zip(sorted(pts)[:-1], sorted(pts)[1:]):
                mid = 0.5 * (lo + hi)
                active = touch & (a + dj * mid > 0.0)
                wa = wgt * active
                alpha = float(K[j] + (wa * dj * dj).sum())
                beta = float(2.0 * (wa * a * dj).sum())
                x = min(max(-beta / (2.0 * alpha), lo), hi) if alpha > 0.0 else (lo if beta >= 0.0 else hi)
                cand = lam.copy()
                cand[j] = x
                fc = f(cand)
                if best_fx is None or fc < best_fx - 1e-18 or (abs(fc - best_fx) <= 1e-18 and x < best_x):
                    best_fx, best_x = fc, x
            if np.isfinite(best_x):
                lam[j] = best_x
        if _kkt_res(lam) <= max(float(tol), 0.0):
            break
    if np.isfinite(lam).all() and np.isfinite(f(lam)) and f(lam) <= best_f + 1e-18:
        best_lam, best_f = lam, f(lam)

    # ---- the certificate -------------------------------------------------
    out = _out(best_lam, False, "not_optimal")
    if out["kkt_res"] <= max(float(tol), 0.0):
        out["converged"], out["reason"] = True, "ok"
    return out


# --------------------------------------------------------------------------- #
# The controller: CPU state, one update per step, identical on every rank


@dataclass
class _RefSide:
    v: torch.Tensor | None = None       # (V,) float32 EMA of E[r]
    R: float = 0.0                       # EMA of E[||r||^2]
    n_obs: int = 0
    last_step: int = -1                  # last step with an observation
    prompts: collections.deque = field(default_factory=collections.deque)      # window of sets
    pg_prompts: collections.deque = field(default_factory=collections.deque)
    tokens: collections.deque = field(default_factory=collections.deque)       # window of counts


@dataclass
class _Ema:
    val: float = 0.0
    n_obs: int = 0


class CrossGateController:
    """Holds the references, the cross statistics and lambda; survives a checkpoint."""

    def __init__(self, cfg: CrossGateConfig, vocab_size: int, task_names=()):
        cfg.validate()
        self.cfg = cfg
        self.V = int(vocab_size)
        self.step = 0
        self.gamma: dict = {}
        # keyed by (task, role, side) -> _RefSide
        self.refs: dict = {}
        # (i, j, role) -> {"B","D","Aneg","Dpos","Dneg"} EMAs
        self.cross: dict = {}
        # (j, role) -> K numerator EMA ; j -> d_sq EMA (K denominator)
        self.k_num: dict = {}
        self.d_sq: dict = {}
        # (j, role, {"T","Tpos","Tneg"}) -> the SENDER'S OWN cost EMAs. Reported,
        # never in the objective: the diagonal of B and D is masked out, so what
        # attenuating task j costs task j is invisible to the solver by
        # construction. Recording it is the prerequisite for ever pricing it.
        self.self_cost: dict = {}
        # (j, role) -> lambda applied NEXT step
        self.lam: dict = {}
        # (i, role) -> (valid, reason, invalid_steps)
        self.validity: dict = {}
        self.last_solver: dict = {}
        self.tasks: list = []
        self._ensure(task_names)

    # ---- bookkeeping ---------------------------------------------------
    def _ensure(self, names) -> None:
        for n in names or ():
            n = str(n)
            if n not in self.tasks:
                self.tasks.append(n)
                self.tasks.sort()

    def _ref(self, task, role, side) -> _RefSide:
        key = (str(task), int(role), int(side))
        st = self.refs.get(key)
        if st is None:
            st = _RefSide()
            self.refs[key] = st
        return st

    def _ema_scalar(self, d: dict, key, value: float) -> None:
        e = d.get(key)
        if e is None:
            e = _Ema()
            d[key] = e
        dec = self.cfg.ema_decay
        e.val = float(value) if (e.n_obs == 0 or dec == 0.0) else dec * e.val + (1.0 - dec) * float(value)
        e.n_obs += 1

    # ---- what the loss reads -------------------------------------------
    def refs_to_device(self, names, device, dtype=torch.float32) -> CrossGateRefs:
        self._ensure(names)
        names = [str(n) for n in names]
        nT, nR, nS = len(names), N_ROLES, N_SIDES
        v = torch.zeros(nT, nR, nS, self.V, dtype=dtype)
        R = torch.zeros(nT, nR, nS, dtype=dtype)
        v_norm = torch.zeros(nT, nR, nS, dtype=dtype)
        gamma = torch.zeros(nT, nR, dtype=dtype)
        side_w = torch.zeros(nT, nR, nS, dtype=dtype)
        valid = torch.zeros(nT, nR, dtype=torch.bool)
        lam = torch.zeros(nT, nR, dtype=dtype)
        for ti, n in enumerate(names):
            for c in range(nR):
                seen = []
                for s in range(nS):
                    st = self.refs.get((n, c, s))
                    if st is not None and st.v is not None and st.n_obs > 0:
                        v[ti, c, s] = st.v.to(dtype)
                        R[ti, c, s] = float(st.R)
                        v_norm[ti, c, s] = float(st.v.double().norm())
                        seen.append(s)
                for s in seen:
                    side_w[ti, c, s] = 1.0 / len(seen)
                valid[ti, c] = bool(self.validity.get((n, c), (False, INVALID_NEVER_SEEN, 0))[0])
                lam[ti, c] = float(self.lam.get((n, c), 0.0))
                # gamma: how far the two prompt-disjoint halves agree on a
                # direction. [.]_+ so opposed halves are inert rather than
                # sign-flipped. A HEURISTIC reliability weight, NOT a
                # significance test: no null is subtracted, because
                # cos(v1, v2) ~ N(0, 1/n_eff) has no basis for correlated,
                # frequency-skewed, sparse vocabulary gradients -- and n_eff
                # here is a participation ratio (||v||_1^2 / ||v||_2^2), which
                # measures how spread the vocabulary components are, not a count
                # of independent directions or of samples. A bias common to both
                # halves also raises gamma. This removes the v1 gate's DIRECT
                # penalty on support diversity; it does not solve reliability.
                if len(seen) == nS:
                    a = self.refs[(n, c, 0)].v.double()
                    b = self.refs[(n, c, 1)].v.double()
                    na, nb = float(a.norm()), float(b.norm())
                    if na > 0.0 and nb > 0.0:
                        cs = float((a @ b).item() / (na * nb))
                        gamma[ti, c] = max(0.0, cs) if math.isfinite(cs) else 0.0
                self.gamma[(n, c)] = float(gamma[ti, c])
        ctrl = self.cfg.control_role_mask()
        lam = lam * ctrl.to(dtype).unsqueeze(0)
        return CrossGateRefs(
            v=v.to(device), R=R.to(device), v_norm=v_norm.to(device),
            gamma=gamma.to(device), side_w=side_w.to(device),
            valid=valid.to(device), lam=lam.to(device), control_roles=ctrl.to(device),
        )

    # ---- once per step -------------------------------------------------
    def update(self, names, reduced: dict, prompt_keys, *, unkeyed_rows: int = 0) -> dict:
        """Fold this step's reduced sums in; solve lambda for the NEXT step.

        ``names`` is this batch's task order (the tables are indexed by it),
        ``reduced`` the dict from :meth:`CrossGateStats.reduced`, ``prompt_keys``
        the driver's dense-index -> stable-key list for this step.
        """
        cfg = self.cfg
        names = [str(n) for n in names]
        self._ensure(names)
        nT, nR, nS = len(names), N_ROLES, N_SIDES
        ref_sum, side = reduced["ref_sum"], reduced["side"]
        pbm_c, pbm_g = reduced["pbm_contrib"], reduced["pbm_pg"]
        cross, send = reduced["cross"], reduced["send"]
        keys = list(prompt_keys or [])
        metrics: dict = {}
        self.step += 1
        step = self.step

        # 1. references: mean of this step's sums, into the EMAs; missing != zero
        for ti, n in enumerate(names):
            for c in range(nR):
                for s in range(nS):
                    st = self._ref(n, c, s)
                    n_tok = float(side[ti, c, s, _SIDE_IDX["n_tok"]])
                    if n_tok > 0:
                        mean_r = (ref_sum[ti, c, s] / n_tok).to(torch.float32)
                        mean_rsq = float(side[ti, c, s, _SIDE_IDX["r_sq_sum"]]) / n_tok
                        dec = cfg.ema_decay
                        if st.n_obs == 0 or dec == 0.0 or st.v is None:
                            st.v, st.R = mean_r.clone(), mean_rsq
                        else:
                            st.v = dec * st.v + (1.0 - dec) * mean_r
                            st.R = dec * st.R + (1.0 - dec) * mean_rsq
                        st.n_obs += 1
                        st.last_step = step
                    # window bookkeeping -- always appended, so the window is in
                    # STEPS, and a step with no rows of this task contributes an
                    # empty set rather than being skipped
                    cols_c = torch.nonzero(pbm_c[ti, c, s] > 0).reshape(-1).tolist()
                    cols_g = torch.nonzero(pbm_g[ti, c, s] > 0).reshape(-1).tolist()
                    st.prompts.append({keys[d] for d in cols_c if d < len(keys)})
                    st.pg_prompts.append({keys[d] for d in cols_g if d < len(keys)})
                    st.tokens.append(int(n_tok))
                    while len(st.prompts) > cfg.window_steps:
                        st.prompts.popleft(); st.pg_prompts.popleft(); st.tokens.popleft()

        # 2. validity per (i, c), both sides
        for ti, n in enumerate(names):
            for c in range(nR):
                reason = INVALID_NONE
                for s in range(nS):
                    st = self._ref(n, c, s)
                    if st.n_obs == 0 or st.v is None:
                        reason = INVALID_NEVER_SEEN; break
                    if not (torch.isfinite(st.v).all() and math.isfinite(st.R)):
                        reason = INVALID_NONFINITE; break
                    if st.R <= 0.0:
                        reason = INVALID_ZERO_R; break
                    if step - st.last_step > cfg.max_staleness:
                        reason = INVALID_STALE; break
                    n_prompts = len(set().union(*st.prompts)) if st.prompts else 0
                    n_pg = len(set().union(*st.pg_prompts)) if st.pg_prompts else 0
                    if n_prompts < cfg.min_prompts:
                        reason = INVALID_FEW_PROMPTS; break
                    if n_pg < cfg.min_pg_prompts:
                        reason = INVALID_FEW_PG_PROMPTS; break
                    if sum(st.tokens) < cfg.min_tokens:
                        reason = INVALID_FEW_TOKENS; break
                prev = self.validity.get((n, c), (False, INVALID_NEVER_SEEN, 0))
                inv_steps = 0 if reason == INVALID_NONE else prev[2] + 1
                self.validity[(n, c)] = (reason == INVALID_NONE, reason, inv_steps)

        # 3. cross / sender EMAs (present iff the cell had tokens)
        for ti, i in enumerate(names):
            for tj, j in enumerate(names):
                if i == j:
                    continue
                for c in range(nR):
                    n_tok = float(cross[ti, tj, c, _CROSS_IDX["n_tok"]])
                    if n_tok <= 0:
                        continue
                    for col, nm in (("B_sum", "B"), ("D_sum", "D"), ("Aneg_sum", "Aneg"),
                                    ("Dpos_sum", "Dpos"), ("Dneg_sum", "Dneg")):
                        self._ema_scalar(self.cross, (i, j, c, nm), float(cross[ti, tj, c, _CROSS_IDX[col]]) / n_tok)
        for tj, j in enumerate(names):
            tot = float(send[tj, :, _SEND_IDX["n_tok"]].sum())
            if tot > 0:
                self._ema_scalar(self.d_sq, j, float(send[tj, :, _SEND_IDX["d_sq_sum"]].sum()) / tot)
                for c in range(nR):
                    # K's numerator carries the role share: sum over (j, c) / n_j
                    self._ema_scalar(self.k_num, (j, c), float(send[tj, c, _SEND_IDX["K_num"]]) / tot)
                    # the sender's own cost, on the same EMA and the same share
                    # basis as K so lambda_j T_j and K_j lambda_j^2 are readable
                    # against each other
                    for col, nm in (("T_num", "T"), ("T_pos", "Tpos"), ("T_neg", "Tneg")):
                        self._ema_scalar(self.self_cost, (j, c, nm),
                                         float(send[tj, c, _SEND_IDX[col]]) / tot)

        # 4. solve per controlled role
        ctrl = cfg.control_role_mask()
        new_lam: dict = {}
        for c in range(nR):
            if not bool(ctrl[c]):
                for j in names:
                    new_lam[(j, c)] = 0.0
                continue
            K = np.array([
                (self.k_num.get((j, c), _Ema()).val / (self.d_sq.get(j, _Ema()).val + cfg.delta))
                if self.d_sq.get(j) is not None else 0.0
                for j in names], dtype=np.float64)
            Bm = np.zeros((nT, nT)); Dm = np.zeros((nT, nT)); Am = np.zeros((nT, nT))
            Dp = np.zeros((nT, nT)); Dn = np.zeros((nT, nT))
            for ti, i in enumerate(names):
                for tj, j in enumerate(names):
                    if i == j:
                        continue
                    Bm[ti, tj] = self.cross.get((i, j, c, "B"), _Ema()).val
                    Dm[ti, tj] = self.cross.get((i, j, c, "D"), _Ema()).val
                    Am[ti, tj] = self.cross.get((i, j, c, "Aneg"), _Ema()).val
                    Dp[ti, tj] = self.cross.get((i, j, c, "Dpos"), _Ema()).val
                    Dn[ti, tj] = self.cross.get((i, j, c, "Dneg"), _Ema()).val
            Rv = np.array([
                float(np.mean([self._ref(i, c, s).R for s in range(nS)]))
                for i in names], dtype=np.float64)
            # v2's penalty scale: the OBSERVED cross magnitude per receiver.
            #   E|x| = E[x] + 2 E[[-x]_+]  =  B + 2 A^-      (|x| = x + 2[-x]_+)
            # so S_i = sum_{j != i} (B_ij + 2 A^-_ij) needs no new statistic.
            # A^- only sets the SCALE here; what decides firing is still the
            # signed sum B, so this is not a return to "any negative part
            # attenuates". Clamped at 0 because a float sum of a non-negative
            # quantity can go slightly negative.
            off = ~np.eye(nT, dtype=bool)
            Sv = np.maximum(((Bm + 2.0 * Am) * off).sum(axis=1), 0.0)
            valid = np.array([bool(self.validity.get((i, c), (False, 0, 0))[0]) for i in names])
            _v2 = int(cfg.gate_version) >= 2
            # THE RECEIVER'S RELIABILITY, now on both sides of the mechanism.
            #
            # gamma_{i,c} = [cos(v_i^1, v_i^2)]_+ already multiplies q in the
            # gate, so gamma = 0 means "this reference may not attenuate a
            # single token". Until this line it did NOT weight the same
            # reference's demand in the objective: a receiver whose two
            # prompt-disjoint halves disagree still carried its full B, D and S
            # into the choice of WHOSE lambda to raise, and so could have its
            # demand met through the gate some OTHER receiver opened. Passing it
            # here makes the two agree on what is believable.
            #
            # gamma_USED, not gamma_next: this is the gamma that produced the h
            # in this step's forward and therefore the B, D and S being read.
            # Those are multi-step EMAs while gamma is this step's value, so the
            # pairing is an APPROXIMATION, not an identity -- the weight is
            # current and the statistics it weights are not. v1 has no gamma and
            # passes None, which is ones.
            gam_used = np.array([float(self.gamma.get((i, c), 0.0)) for i in names], dtype=np.float64) \
                if _v2 else None
            sol = solve_role(K, Bm, Dm, Rv, valid, eps=cfg.eps_cross,
                             rho=(cfg.rho_rel if _v2 else cfg.rho),
                             scale=(Sv if _v2 else None),
                             lam_max=cfg.lambda_max, delta=cfg.delta,
                             iters=cfg.solver_iters, tol=cfg.solver_tol,
                             gamma=gam_used)
            self.last_solver[c] = sol
            # Both scales, always: s/S is what v2 controls on, s/R is the v1
            # basis kept so the two arms are readable against each other. Also
            # gamma and S themselves, since they are what v2 added.
            for ti, i in enumerate(names):
                rn = ROLE_NAMES[c]
                _s, _s0 = float(sol["s"][ti]), float(sol["s0"][ti])
                metrics[f"actor/cross/S/{i}/{rn}"] = float(Sv[ti])
                metrics[f"actor/cross/gamma/{i}/{rn}"] = float(self.gamma.get((i, c), 0.0))
                metrics[f"actor/cross/slack_over_S/{i}/{rn}"] = _s / (float(Sv[ti]) + cfg.delta)
                metrics[f"actor/cross/slack_over_R/{i}/{rn}"] = _s / (float(Rv[ti]) + cfg.delta)
                metrics[f"actor/cross/slack0_over_S/{i}/{rn}"] = _s0 / (float(Sv[ti]) + cfg.delta)
                metrics[f"actor/cross/slack0_over_R/{i}/{rn}"] = _s0 / (float(Rv[ti]) + cfg.delta)
            for tj, j in enumerate(names):
                # AN UNCERTIFIED SOLVE ATTENUATES NOTHING. The old code kept a
                # "max_iters" iterate, which was safe only because `converged`
                # meant "improved a little" and so almost never came back false.
                # Now it means "the KKT residual is within tol", and a solve
                # that cannot say that has not established which lambda is
                # right -- so lambda is 0 and solver_reason_code says why. The
                # cost of the conservative branch is a step with no cross
                # attenuation; the cost of the other is attenuating on a number
                # nothing vouches for.
                new_lam[(j, c)] = (float(sol["lam"][tj])
                                   if sol["reason"] in ("ok", "no_valid_receiver", "lambda_max_zero")
                                   else 0.0)
            rn = ROLE_NAMES[c]
            metrics[f"actor/cross/solver_converged/{rn}"] = 1.0 if sol["converged"] else 0.0
            metrics[f"actor/cross/solver_reason_code/{rn}"] = float(
                {"ok": 0, "not_optimal": 1, "no_valid_receiver": 2, "lambda_max_zero": 3,
                 "nonfinite_input": 4, "nonfinite_iterate": 5}.get(sol["reason"], 9))
            # the certificate itself, so a run can be audited without re-solving
            metrics[f"actor/cross/solver_kkt_res/{rn}"] = float(sol.get("kkt_res", float("nan")))
            metrics[f"actor/cross/solver_opt_gap/{rn}"] = float(sol.get("opt_gap", float("nan")))
            metrics[f"actor/cross/solver_f/{rn}"] = float(sol.get("f", float("nan")))
            for ti, i in enumerate(names):
                metrics[f"actor/cross/slack/{i}/{rn}"] = float(sol["s"][ti])
                metrics[f"actor/cross/slack_at_zero/{i}/{rn}"] = float(sol["s0"][ti])
                # predicted post-control quantity for receiver i under the NEW lambda
                pred = float(sum(Bm[ti, tj] - sol["lam"][tj] * Dm[ti, tj] for tj in range(nT) if tj != ti))
                metrics[f"actor/cross/predicted/{i}/{rn}"] = pred
                # THE TWO HALVES OF THAT ONE NUMBER, which it cannot show.
                # `predicted` moving the right way is compatible with paying for
                # it out of transfer that was working: D pools E[h [x]_+] with
                # E[h [-x]_+]. Under the new lambda, receiver i is predicted to
                # LOSE sum_j lambda_j Dpos_ij of positive cross contribution and
                # to have sum_j lambda_j Dneg_ij of negative contribution
                # REMOVED. Reported, not constrained: a "lose nothing positive"
                # rule would stop almost every intervention whose receivers
                # disagree in sign. Output-space proxies either way.
                metrics[f"actor/cross/pred_lost_positive/{i}/{rn}"] = float(
                    sum(sol["lam"][tj] * Dp[ti, tj] for tj in range(nT) if tj != ti))
                metrics[f"actor/cross/pred_removed_negative/{i}/{rn}"] = float(
                    sum(sol["lam"][tj] * Dn[ti, tj] for tj in range(nT) if tj != ti))
        self.lam.update(new_lam)

        # 5. metrics
        metrics["actor/cross/unkeyed_rows"] = float(unkeyed_rows)
        for ti, n in enumerate(names):
            for c in range(nR):
                rn = ROLE_NAMES[c]
                v_ok, reason, inv_steps = self.validity.get((n, c), (False, INVALID_NEVER_SEEN, 0))
                metrics[f"actor/cross/valid/{n}/{rn}"] = 1.0 if v_ok else 0.0
                metrics[f"actor/cross/invalid_reason/{n}/{rn}"] = float(reason)
                metrics[f"actor/cross/invalid_steps/{n}/{rn}"] = float(inv_steps)
                metrics[f"actor/cross/lambda_next/{n}/{rn}"] = float(self.lam.get((n, c), 0.0))
                vs = []
                for s in range(nS):
                    st = self._ref(n, c, s)
                    if st.v is not None and st.n_obs > 0:
                        vn2 = float((st.v.double() ** 2).sum())
                        kappa = vn2 / st.R if st.R > 0 else 0.0
                        metrics[f"actor/cross/kappa_{s + 1}/{n}/{rn}"] = kappa
                        metrics[f"actor/cross/R_{s + 1}/{n}/{rn}"] = float(st.R)
                        a1 = float(st.v.double().abs().sum())
                        metrics[f"actor/cross/n_eff_{s + 1}/{n}/{rn}"] = (a1 * a1 / vn2) if vn2 > 0 else 0.0
                        metrics[f"actor/cross/n_distinct_{s + 1}/{n}/{rn}"] = float((st.v != 0).sum())
                        vs.append(st.v.double())
                    metrics[f"actor/cross/prompts_side{s + 1}/{n}/{rn}"] = float(
                        len(set().union(*st.prompts)) if st.prompts else 0)
                    metrics[f"actor/cross/pg_prompts_side{s + 1}/{n}/{rn}"] = float(
                        len(set().union(*st.pg_prompts)) if st.pg_prompts else 0)
                    metrics[f"actor/cross/tokens_window_side{s + 1}/{n}/{rn}"] = float(sum(st.tokens))
                    metrics[f"actor/cross/staleness_side{s + 1}/{n}/{rn}"] = float(
                        step - st.last_step if st.last_step >= 0 else -1)
                if len(vs) == nS:
                    na, nb = float(vs[0].norm()), float(vs[1].norm())
                    metrics[f"actor/cross/ref_cos_sides/{n}/{rn}"] = (
                        float((vs[0] * vs[1]).sum()) / (na * nb) if na > 0 and nb > 0 else 0.0)
                # sender-side, this step
                n_tok = float(send[ti, c, _SEND_IDX["n_tok"]])
                if n_tok > 0:
                    g = lambda col: float(send[ti, c, _SEND_IDX[col]])
                    metrics[f"actor/cross/h_mean/{n}/{rn}"] = g("h_sum") / n_tok
                    metrics[f"actor/cross/h_nonzero_frac/{n}/{rn}"] = g("h_nonzero") / n_tok
                    metrics[f"actor/cross/w_mean/{n}/{rn}"] = g("w_sum") / n_tok
                    # THE READOUT OF THE STRENGTH EXPERIMENT. Redundant with
                    # 1 - w_mean by construction and recorded anyway, because
                    # q_scale is not the realised strength: the solver rescales
                    # lambda against the amplified D and K, so this is the only
                    # number that says what the intervention actually was.
                    metrics[f"actor/cross/attenuation_mean/{n}/{rn}"] = 1.0 - g("w_sum") / n_tok
                    metrics[f"actor/cross/atten_ge_half_frac/{n}/{rn}"] = g("h_ge_half") / n_tok
                    metrics[f"actor/cross/atten_ge_9_10_frac/{n}/{rn}"] = g("h_ge_9_10") / n_tok
                    metrics[f"actor/cross/atten_at_cap_frac/{n}/{rn}"] = g("h_sat") / n_tok
                    metrics[f"actor/cross/n_valid_receivers/{n}/{rn}"] = g("n_valid_recv_sum") / n_tok
                    metrics[f"actor/cross/kl_lost_frac/{n}/{rn}"] = g("kl_lost") / max(g("kl_sum"), 1e-30)
                    metrics[f"actor/cross/strength_lost_frac/{n}/{rn}"] = g("strength_lost") / max(g("d_sq_sum"), 1e-30)
                    metrics[f"actor/cross/strength_lost_self_aligned_frac/{n}/{rn}"] = (
                        g("strength_lost_self_aligned") / max(g("strength_lost"), 1e-30))
                    metrics[f"actor/cross/unkeyed_token_frac/{n}/{rn}"] = g("n_unkeyed") / n_tok
                    K_here = self.k_num.get((n, c), _Ema()).val / (self.d_sq.get(n, _Ema()).val + cfg.delta) \
                        if self.d_sq.get(n) is not None else 0.0
                    metrics[f"actor/cross/K/{n}/{rn}"] = float(K_here)
                    # THE SENDER'S OWN COST, which the objective does not see.
                    # T_j = E[h r_j . d_j] is the OPD component aligned with
                    # task j's OWN RL that the attenuation removes; at the new
                    # lambda_j the predicted self-cost is lambda_j T_j. Split
                    # because the signed mean pools a gain (removing OPD that
                    # pushes against this task's RL, sd < 0) with a loss
                    # (removing OPD that agrees, sd > 0), and the existing
                    # strength_lost_self_aligned share is a MAGNITUDE, so it
                    # cannot give the sign of the net. Diagonal-masked B and D
                    # mean none of this reaches the solver: measured first, on
                    # purpose, and not wired to token selection -- choosing
                    # tokens by self-conflict is the old self gate.
                    T_j = float(self.self_cost.get((n, c, "T"), _Ema()).val)
                    metrics[f"actor/cross/self_T/{n}/{rn}"] = T_j
                    metrics[f"actor/cross/self_T_pos/{n}/{rn}"] = float(
                        self.self_cost.get((n, c, "Tpos"), _Ema()).val)
                    metrics[f"actor/cross/self_T_neg/{n}/{rn}"] = float(
                        self.self_cost.get((n, c, "Tneg"), _Ema()).val)
                    metrics[f"actor/cross/pred_self_cost/{n}/{rn}"] = float(
                        new_lam.get((n, c), 0.0)) * T_j
        for ti, i in enumerate(names):
            for tj, j in enumerate(names):
                if i == j:
                    continue
                for c in range(nR):
                    rn = ROLE_NAMES[c]
                    n_tok = float(cross[ti, tj, c, _CROSS_IDX["n_tok"]])
                    if n_tok <= 0:
                        continue
                    g = lambda col: float(cross[ti, tj, c, _CROSS_IDX[col]])
                    metrics[f"actor/cross/B/{i}/{j}/{rn}"] = g("B_sum") / n_tok
                    metrics[f"actor/cross/D/{i}/{j}/{rn}"] = g("D_sum") / n_tok
                    metrics[f"actor/cross/A_neg/{i}/{j}/{rn}"] = g("Aneg_sum") / n_tok
                    metrics[f"actor/cross/D_pos/{i}/{j}/{rn}"] = g("Dpos_sum") / n_tok
                    metrics[f"actor/cross/D_neg/{i}/{j}/{rn}"] = g("Dneg_sum") / n_tok
                    metrics[f"actor/cross/realized/{i}/{j}/{rn}"] = g("realized_sum") / n_tok
                    metrics[f"actor/cross/lost_by_receiver/{j}/{i}/{rn}"] = g("lost_by_recv") / n_tok
                    metrics[f"actor/cross/support_overlap/{i}/{j}/{rn}"] = g("overlap_sum") / n_tok
                    metrics[f"actor/cross/energy_on_shared/{i}/{j}/{rn}"] = (
                        g("energy_shared_sum") / max(float(send[tj, c, _SEND_IDX["d_sq_sum"]]), 1e-30))
        return metrics

    # ---- checkpoint ----------------------------------------------------
    def state_dict(self) -> dict:
        refs = {}
        for (t, c, s), st in self.refs.items():
            refs[f"{t}|{c}|{s}"] = {
                "v": None if st.v is None else st.v.detach().cpu(),
                "R": float(st.R), "n_obs": int(st.n_obs), "last_step": int(st.last_step),
                "prompts": [sorted(p) for p in st.prompts],
                "pg_prompts": [sorted(p) for p in st.pg_prompts],
                "tokens": list(st.tokens),
            }
        return {
            "version": 1,
            "split_version": SPLIT_VERSION,
            "cfg": asdict(self.cfg),
            "vocab_size": self.V,
            "step": self.step,
            "tasks": list(self.tasks),
            "refs": refs,
            "cross": {f"{i}|{j}|{c}|{nm}": asdict(e) for (i, j, c, nm), e in self.cross.items()},
            "k_num": {f"{j}|{c}": asdict(e) for (j, c), e in self.k_num.items()},
            "self_cost": {f"{j}|{c}|{nm}": asdict(e) for (j, c, nm), e in self.self_cost.items()},
            "d_sq": {j: asdict(e) for j, e in self.d_sq.items()},
            "lam": {f"{j}|{c}": float(v) for (j, c), v in self.lam.items()},
            "validity": {f"{i}|{c}": list(v) for (i, c), v in self.validity.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        if sd.get("version") != 1:
            raise ValueError(f"cross_gate state version {sd.get('version')} is not 1")
        if int(sd.get("split_version", -1)) != SPLIT_VERSION:
            raise ValueError("cross_gate: the prompt split scheme changed across resume; refusing to blend")
        saved_cfg = dict(sd.get("cfg", {}))
        live_cfg = asdict(self.cfg)
        saved_cfg["roles"] = tuple(saved_cfg.get("roles", ()))
        drift = {k: (saved_cfg.get(k), live_cfg[k]) for k in live_cfg if saved_cfg.get(k) != live_cfg[k]}
        if drift:
            raise ValueError(f"cross_gate config changed across resume: {drift}")
        if int(sd.get("vocab_size", self.V)) != self.V:
            raise ValueError("cross_gate: vocabulary size changed across resume")
        self.step = int(sd.get("step", 0))
        self._ensure(sd.get("tasks", []))
        self.refs = {}
        for key, d in sd.get("refs", {}).items():
            t, c, s = key.split("|")
            st = _RefSide(v=None if d["v"] is None else d["v"].to(torch.float32), R=float(d["R"]),
                          n_obs=int(d["n_obs"]), last_step=int(d["last_step"]))
            st.prompts = collections.deque(set(p) for p in d.get("prompts", []))
            st.pg_prompts = collections.deque(set(p) for p in d.get("pg_prompts", []))
            st.tokens = collections.deque(int(x) for x in d.get("tokens", []))
            self.refs[(t, int(c), int(s))] = st
        self.cross = {}
        for key, d in sd.get("cross", {}).items():
            i, j, c, nm = key.split("|")
            self.cross[(i, j, int(c), nm)] = _Ema(**d)
        self.k_num = {}
        for key, d in sd.get("k_num", {}).items():
            j, c = key.split("|")
            self.k_num[(j, int(c))] = _Ema(**d)
        self.d_sq = {j: _Ema(**d) for j, d in sd.get("d_sq", {}).items()}
        # absent in checkpoints written before the self-cost accounting existed;
        # an empty dict rebuilds from the next step's observations, which is
        # correct for a REPORTED quantity and would not be for one the solver reads
        self.self_cost = {}
        for key, d in sd.get("self_cost", {}).items():
            j, c, nm = key.split("|")
            self.self_cost[(j, int(c), nm)] = _Ema(**d)
        self.lam = {}
        for key, v in sd.get("lam", {}).items():
            j, c = key.split("|")
            self.lam[(j, int(c))] = float(v)
        self.validity = {}
        for key, v in sd.get("validity", {}).items():
            i, c = key.split("|")
            self.validity[(i, int(c))] = (bool(v[0]), int(v[1]), int(v[2]))
