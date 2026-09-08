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
role, solved by projected gradient with backtracking in float64 on the CPU,
on one rank, and broadcast.

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
    # Tolerated NEGATIVE net cross contribution, as a fraction of the receiver's
    # R_{i,c}. 0 makes any net negative contribution the object of control.
    eps_cross: float = 0.0
    # Weight on the squared normalised slack. 1/beta_0^2 corrects the one known
    # scale gap: at eps_cross = 0 the slack is linear in beta while the removal
    # cost is beta-free (design doc §8.2). Not a balance claim.
    rho: float = 1.0e4
    # Cap on the per-token attenuation. An experimental condition the constraint
    # is not allowed to override.
    lambda_max: float = 0.2
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
            elif k in ("window_steps", "min_prompts", "min_pg_prompts", "min_tokens",
                       "max_staleness", "split_seed", "solver_iters"):
                kw[k] = int(v)
            else:
                kw[k] = float(v)
        return cls(**kw)

    def validate(self) -> None:
        if self.eps_cross < 0.0:
            raise ValueError(f"cross_gate.eps_cross must be >= 0, got {self.eps_cross}")
        if self.rho < 0.0:
            raise ValueError(f"cross_gate.rho must be >= 0, got {self.rho}")
        if not (0.0 <= self.lambda_max <= 1.0):
            raise ValueError(f"cross_gate.lambda_max must be in [0, 1], got {self.lambda_max}")
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
        sw = refs.side_w                                                           # (nT, nR, 2)

        x_side = torch.zeros(bs, T, nT, nS, device=dev, dtype=dt)
        nonzero_share = torch.zeros(bs, T, nT, device=dev, dtype=dt)
        energy_shared = torch.zeros(bs, T, nT, device=dev, dtype=dt)
        R_tok = torch.zeros(bs, T, nT, nS, device=dev, dtype=dt)
        for i in range(nT):
            for s in range(nS):
                base = ((i * nR + rol) * nS + s) * V                                # (bs, T)
                vg = vflat[base.unsqueeze(-1) + ids]                                # (bs, T, k)
                x_side[:, :, i, s] = (vg * d).sum(dim=-1)
                R_tok[:, :, i, s] = Rr[i][rol, s]
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
        denom = R_tok.clamp(min=0.0).sqrt() * d_norm.unsqueeze(-1).unsqueeze(-1) + float(delta)
        q_side = neg / denom
        q = q_side.min(dim=-1).values                                              # (bs, T, nT)
        # Zero out where the side's R is 0 (no reference energy): the formula
        # would divide by delta and manufacture a gate out of nothing.
        q = torch.where((R_tok > 0).all(dim=-1), q, torch.zeros_like(q))
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
    "h_sum",
    "h_nonzero",
    "w_sum",
    "n_valid_recv_sum",
    "n_unkeyed",       # tokens on rows whose side is -1
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
            _add_send("h_sum", h)
            _add_send("h_nonzero", (h > 0).to(torch.float32))
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
    lam_max: float,
    delta: float,
    iters: int = 500,
    tol: float = 1e-12,
) -> dict:
    """min sum_j K_j l_j^2 + rho sum_{i valid} (s_i(l)/(R_i+delta))^2, 0 <= l <= lam_max.

    ``K`` (n,), ``B``/``D`` (n_i, n_j) with the sender on the second axis,
    ``R`` (n,), ``valid_recv`` (n,) bool. Diagonal entries of B and D are
    ignored (a task is never its own receiver). Returns ``lam`` (n,), the slack
    ``s`` (n,) at the solution, ``s0`` at lambda = 0, ``converged`` and ``reason``.

    Convex: each s_i is the positive part of an affine function of lambda, its
    square is convex, and K >= 0. Piecewise quadratic, so no single closed form
    -- and one sender's lambda enters several receivers' conditions, so it is
    not a single-multiplier problem either.

    SOLVED BY EXACT CYCLIC COORDINATE DESCENT, not projected gradient. Along one
    coordinate the objective is a convex piecewise quadratic whose breakpoints
    are where a receiver's slack crosses zero; each piece has a closed-form
    minimiser, so the 1-D step is exact. Projected gradient was tried first and
    needs O(rho / R^2) ~ 1e4 iterations on this objective; this converges in a
    handful of sweeps. A coordinate whose objective is flat (unobserved, or no
    receiver it can help) is placed at 0 -- the smaller lambda wins a tie.
    """
    n = int(K.shape[0])
    K = np.asarray(K, dtype=np.float64).reshape(n)
    B = np.asarray(B, dtype=np.float64).reshape(n, n)
    D = np.asarray(D, dtype=np.float64).reshape(n, n)
    R = np.asarray(R, dtype=np.float64).reshape(n)
    valid = np.asarray(valid_recv, dtype=bool).reshape(n)
    off = ~np.eye(n, dtype=bool)
    Bo, Do = B * off, D * off
    lam0 = np.zeros(n)

    def slack(l):
        return np.maximum(-eps * R - Bo.sum(axis=1) + Do @ l, 0.0) * valid

    def f(l):
        s = slack(l)
        return float((K * l * l).sum() + rho * ((s / (R + delta)) ** 2).sum())

    def grad(l):
        s = slack(l)
        g = 2.0 * K * l
        coef = 2.0 * rho * s / (R + delta) ** 2          # (n_i,)
        return g + Do.T @ coef

    finite_in = np.isfinite(K).all() and np.isfinite(Bo).all() and np.isfinite(Do).all() and np.isfinite(R).all()
    if not finite_in:
        return {"lam": lam0, "s": slack(lam0), "s0": slack(lam0), "converged": False,
                "reason": "nonfinite_input", "f": float("nan")}
    if not valid.any() or lam_max <= 0.0:
        return {"lam": lam0, "s": slack(lam0), "s0": slack(lam0), "converged": True,
                "reason": "no_valid_receiver" if not valid.any() else "lambda_max_zero", "f": f(lam0)}

    wgt_all = rho / (R + delta) ** 2 * valid          # per receiver
    lam = lam0.copy()
    fx = f(lam)
    converged = False
    for _ in range(int(iters)):
        max_move = 0.0
        for j in range(n):
            dj = Do[:, j]
            # receivers this coordinate can touch: valid, not itself, D != 0
            touch = valid & off[:, j] & (dj != 0.0)
            # the slack's affine part with lambda_j's own contribution removed
            a = -eps * R - Bo.sum(axis=1) + Do @ lam - dj * lam[j]
            pts = {0.0, float(lam_max)}
            for i in np.nonzero(touch)[0]:
                bp = -a[i] / dj[i]
                if 0.0 < bp < lam_max:
                    pts.add(float(bp))
            pts = sorted(pts)
            best_x, best_f = lam[j], None
            for lo, hi in zip(pts[:-1], pts[1:]):
                mid = 0.5 * (lo + hi)
                active = touch & (a + dj * mid > 0.0)
                w = wgt_all * active
                alpha = float(K[j] + (w * dj * dj).sum())
                beta = float(2.0 * (w * a * dj).sum())
                if alpha > 0.0:
                    x = min(max(-beta / (2.0 * alpha), lo), hi)
                else:
                    # flat or linear on this piece: the smaller lambda on a tie
                    x = lo if beta >= 0.0 else hi
                cand = lam.copy()
                cand[j] = x
                fc = f(cand)
                if best_f is None or fc < best_f - 1e-18 or (abs(fc - best_f) <= 1e-18 and x < best_x):
                    best_f, best_x = fc, x
            max_move = max(max_move, abs(best_x - lam[j]))
            lam[j] = best_x
        fx = f(lam)
        if not np.isfinite(fx):
            break
        if max_move < tol:
            converged = True
            break
    if not np.isfinite(lam).all() or not np.isfinite(fx):
        return {"lam": lam0, "s": slack(lam0), "s0": slack(lam0), "converged": False,
                "reason": "nonfinite_iterate", "f": float("nan")}
    return {"lam": lam, "s": slack(lam), "s0": slack(lam0), "converged": converged,
            "reason": "ok" if converged else "max_iters", "f": fx}


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
        # keyed by (task, role, side) -> _RefSide
        self.refs: dict = {}
        # (i, j, role) -> {"B","D","Aneg"} EMAs
        self.cross: dict = {}
        # (j, role) -> K numerator EMA ; j -> d_sq EMA (K denominator)
        self.k_num: dict = {}
        self.d_sq: dict = {}
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
                        seen.append(s)
                for s in seen:
                    side_w[ti, c, s] = 1.0 / len(seen)
                valid[ti, c] = bool(self.validity.get((n, c), (False, INVALID_NEVER_SEEN, 0))[0])
                lam[ti, c] = float(self.lam.get((n, c), 0.0))
        ctrl = self.cfg.control_role_mask()
        lam = lam * ctrl.to(dtype).unsqueeze(0)
        return CrossGateRefs(
            v=v.to(device), R=R.to(device), side_w=side_w.to(device),
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
                    for col, nm in (("B_sum", "B"), ("D_sum", "D"), ("Aneg_sum", "Aneg")):
                        self._ema_scalar(self.cross, (i, j, c, nm), float(cross[ti, tj, c, _CROSS_IDX[col]]) / n_tok)
        for tj, j in enumerate(names):
            tot = float(send[tj, :, _SEND_IDX["n_tok"]].sum())
            if tot > 0:
                self._ema_scalar(self.d_sq, j, float(send[tj, :, _SEND_IDX["d_sq_sum"]].sum()) / tot)
                for c in range(nR):
                    # K's numerator carries the role share: sum over (j, c) / n_j
                    self._ema_scalar(self.k_num, (j, c), float(send[tj, c, _SEND_IDX["K_num"]]) / tot)

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
            Bm = np.zeros((nT, nT)); Dm = np.zeros((nT, nT))
            for ti, i in enumerate(names):
                for tj, j in enumerate(names):
                    if i == j:
                        continue
                    Bm[ti, tj] = self.cross.get((i, j, c, "B"), _Ema()).val
                    Dm[ti, tj] = self.cross.get((i, j, c, "D"), _Ema()).val
            Rv = np.array([
                float(np.mean([self._ref(i, c, s).R for s in range(nS)]))
                for i in names], dtype=np.float64)
            valid = np.array([bool(self.validity.get((i, c), (False, 0, 0))[0]) for i in names])
            sol = solve_role(K, Bm, Dm, Rv, valid, eps=cfg.eps_cross, rho=cfg.rho,
                             lam_max=cfg.lambda_max, delta=cfg.delta,
                             iters=cfg.solver_iters, tol=cfg.solver_tol)
            self.last_solver[c] = sol
            for tj, j in enumerate(names):
                new_lam[(j, c)] = float(sol["lam"][tj]) if sol["reason"] in ("ok", "max_iters", "no_valid_receiver", "lambda_max_zero") else 0.0
                if sol["reason"] not in ("ok", "no_valid_receiver", "lambda_max_zero"):
                    # max_iters keeps the (feasible) iterate; non-finite falls to 0
                    if sol["reason"].startswith("nonfinite"):
                        new_lam[(j, c)] = 0.0
            rn = ROLE_NAMES[c]
            metrics[f"actor/cross/solver_converged/{rn}"] = 1.0 if sol["converged"] else 0.0
            metrics[f"actor/cross/solver_reason_code/{rn}"] = float(
                {"ok": 0, "max_iters": 1, "no_valid_receiver": 2, "lambda_max_zero": 3,
                 "nonfinite_input": 4, "nonfinite_iterate": 5}.get(sol["reason"], 9))
            for ti, i in enumerate(names):
                metrics[f"actor/cross/slack/{i}/{rn}"] = float(sol["s"][ti])
                metrics[f"actor/cross/slack_at_zero/{i}/{rn}"] = float(sol["s0"][ti])
                # predicted post-control quantity for receiver i under the NEW lambda
                pred = float(sum(Bm[ti, tj] - sol["lam"][tj] * Dm[ti, tj] for tj in range(nT) if tj != ti))
                metrics[f"actor/cross/predicted/{i}/{rn}"] = pred
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
                    metrics[f"actor/cross/n_valid_receivers/{n}/{rn}"] = g("n_valid_recv_sum") / n_tok
                    metrics[f"actor/cross/kl_lost_frac/{n}/{rn}"] = g("kl_lost") / max(g("kl_sum"), 1e-30)
                    metrics[f"actor/cross/strength_lost_frac/{n}/{rn}"] = g("strength_lost") / max(g("d_sq_sum"), 1e-30)
                    metrics[f"actor/cross/strength_lost_self_aligned_frac/{n}/{rn}"] = (
                        g("strength_lost_self_aligned") / max(g("strength_lost"), 1e-30))
                    metrics[f"actor/cross/unkeyed_token_frac/{n}/{rn}"] = g("n_unkeyed") / n_tok
                    K_here = self.k_num.get((n, c), _Ema()).val / (self.d_sq.get(n, _Ema()).val + cfg.delta) \
                        if self.d_sq.get(n) is not None else 0.0
                    metrics[f"actor/cross/K/{n}/{rn}"] = float(K_here)
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
        self.lam = {}
        for key, v in sd.get("lam", {}).items():
            j, c = key.split("|")
            self.lam[(j, int(c))] = float(v)
        self.validity = {}
        for key, v in sd.get("validity", {}).items():
            i, c = key.split("|")
            self.validity[(i, int(c))] = (bool(v[0]), int(v[1]), int(v[2]))
