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

"""Per-task, per-step OPD retention, applied only where the teacher pushes back.

The static coefficient arm scaled a whole task's teacher-KL term by one number,
calibrated once at step 300 and applied from step 0. Two things were wrong with
that at once: the coefficient was stale (the signal it answered to had moved by
step 6), and it was indiscriminate (webshop's OPD has a part that helps webshop
and a part that fights its current reward, and halving both was the crude part).

This replaces it with a rule that has no calibration step and no cross-task
transfer:

    w_t = a_i  if the token is in the control population AND u_R . u_D < 0
        = 1    otherwise

    L = L_GRPO + beta * Agg[ w_t * L_OPD,t ]

with a_i in [0, 1] a per-task RETENTION on the conflicting tokens, chosen as the
smallest attenuation that keeps the teacher's pushback within a fraction eps of
the reward's own descent:

    min  1/2 (a - 1)^2   s.t.  a * C_i^- <= eps * R_i,   0 <= a <= 1
    =>   a_i* = 1                         if C_i^- = 0
              = min(1, eps * R_i / C_i^-)  otherwise

where, on ONE population with ONE weighting and the teacher term at its BASE
coefficient beta (never at a -- a rule that reads its own output back is a
feedback loop, not a measurement):

    R_i   = sum_t ||u_R,t||^2                  the reward's own descent
    C_i^- = sum_t [ -u_R,t . u_D,t ]_+         what the teacher cancels of it

C^- takes the negative part PER TOKEN before summing. sum_t [-X_t]_+ is not
[-sum_t X_t]_+: a signed sum lets one token's helpful distillation hide another
token's strong conflict, which is exactly the thing a token-selective rule exists
to see.

WHY THE GATE IS BINARY IN THIS VERSION. The closed form above is the solution
when every conflicting token gets the same retention a. A soft gate
w_t = 1 - (1 - a) q_t changes the post-control pushback to C^- - (1 - a) C_q^-
with C_q^- = sum_t q_t [-X_t]_+, so the bound becomes a <= 1 - (C^- - eps R)/C_q^-
and can be INFEASIBLE at a = 0. A soft gate is a plausible refinement, but it
carries a different formula and is evaluated as a different condition, not
slipped in under this one.

TIMING. a_i for step s is computed from steps < s. The token gate is decided in
step s's own forward. Nothing here needs a second forward or backward: u_R and
u_D are closed-form logit gradients from tensors the loss already materialises
(see opd_task_diag.opd_pg_alignment_terms), and R, C^- are reduced once per step
by the same all-reduce the diagnostics already pay. a_i is FIXED within a step
-- every micro-batch of one update sees the same value.

WHAT IT CANNOT DO. r_i . P d_j for i != j is u_R,i . J_i P J_j^T u_D,j; no
per-token quantity carries the Jacobians. This balances each task's own reward
against its own teacher. It does not resolve cross-task interference, and must
not be described as doing so. Whether it helps is decided by each task's
evaluation accuracy, not by any number this module reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

__all__ = ["PushbackConfig", "PushbackController", "pushback_retention", "conflict_gate", "live_groups_by_task"]

# The four signal states. Kept apart because they call for different handling
# and because collapsing them was the reviewer's specific objection to "R below
# a threshold": R depends on the loss normalisation, the group count and the row
# weights, so an absolute floor on it moves whenever any of those does.
STATE_ABSENT = 0        # the task had no rows this step: do not touch the EMA
STATE_NO_PG = 1         # rows, but no live policy-gradient token: hold a = 1
STATE_FEW_GROUPS = 2    # some signal, but too few independent prompt groups: hold
STATE_OK = 3            # enough to act on
STATE_NAMES = ("absent", "no_pg", "few_groups", "ok")


@dataclass
class PushbackConfig:
    enable: bool = False
    # Allowed pushback as a fraction of the reward's own descent. A statement
    # of how much intervention is tolerated, not a value known to be optimal.
    eps: float = 0.1
    # Decay of the per-task EMA on R and C^- (applied to the SUMS, then the
    # ratio is taken -- never an EMA of ratios). 0 disables smoothing.
    ema_decay: float = 0.9
    # Statistical sufficiency, in things that do not depend on the loss scale.
    min_live_groups: int = 4      # independent prompt groups with a live advantage
    min_ctl_tokens: int = 256     # tokens in the control population
    # Floor on a. 0 permits full attenuation of conflicting tokens. If a > 0 is
    # set and the bound falls below it, the constraint is UNMET and is recorded
    # as such -- clipping to the floor is not achievement.
    a_min: float = 0.0
    # Purely numerical guard on divisions. Not a sufficiency threshold.
    tiny: float = 1e-12

    @classmethod
    def from_mapping(cls, m) -> PushbackConfig:
        if not m:
            return cls()
        m = dict(m)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        bad = sorted(set(m) - allowed)
        if bad:
            raise ValueError(f"pushback_control: unknown keys {bad}; allowed {sorted(allowed)}")
        return cls(**{k: (bool(v) if k == "enable" else (int(v) if k.startswith("min_") else float(v)))
                      for k, v in m.items()})

    def validate(self) -> None:
        if not (0.0 < self.eps):
            raise ValueError(f"pushback_control.eps must be > 0, got {self.eps}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"pushback_control.ema_decay must be in [0, 1), got {self.ema_decay}")
        if not (0.0 <= self.a_min <= 1.0):
            raise ValueError(f"pushback_control.a_min must be in [0, 1], got {self.a_min}")
        if self.min_live_groups < 1 or self.min_ctl_tokens < 1:
            raise ValueError("pushback_control.min_live_groups and min_ctl_tokens must be >= 1")


def pushback_retention(eps: float, R: float, C_neg: float, *, a_min: float = 0.0, tiny: float = 1e-12):
    """The closed form, and whether the constraint it targets is actually met.

    Returns ``(a, met)``. ``met`` is False only when a floor ``a_min > 0`` stops
    the bound from being reached -- with ``a_min = 0`` the bound is always
    attainable, since a = 0 removes all conflicting pushback.
    """
    if C_neg <= tiny:
        return 1.0, True
    bound = eps * R / C_neg
    a = min(1.0, bound)
    if a < a_min:
        return a_min, False
    return a, True


def conflict_gate(terms: dict, task_ids: torch.Tensor, a_by_task: torch.Tensor, n_task: int) -> torch.Tensor:
    """Per-token weight on the OPD term: ``a[task]`` where the teacher pushes back, else 1.

    ``terms`` is the dict from :func:`opd_task_diag.opd_pg_alignment_terms`,
    computed at the BASE coefficient. Conflicting means: in the control
    population (sampled id inside the support), the reward has a live descent
    there (``ctl_rr > 0`` -- a clipped token has u_R = 0 and is not a conflict,
    it is silence), and the teacher's push opposes it (``ctl_rd < 0``).

    Everything here is detached. The gate is a measurement of the current
    forward, not a differentiable part of the loss.
    """
    with torch.no_grad():
        rd = terms["ctl_rd"]
        conflicting = (terms["ctl_mask"] > 0) & (rd < 0) & (terms["ctl_rr"] > 0)
        flat = task_ids.reshape(-1)
        flat = flat.round().to(torch.long) if flat.is_floating_point() else flat.to(torch.long)
        # Padding rows carry a negative id; they get a = 1 and are masked out of
        # the loss anyway.
        table = torch.cat([a_by_task.to(rd.device, rd.dtype), torch.ones(1, device=rd.device, dtype=rd.dtype)])
        idx = torch.where((flat >= 0) & (flat < n_task), flat, torch.full_like(flat, n_task))
        a_row = table[idx].unsqueeze(-1)  # (bs, 1)
        return torch.where(conflicting, a_row.expand_as(rd), torch.ones_like(rd))


@dataclass
class _TaskState:
    a: float = 1.0
    ema_R: float = 0.0
    ema_C: float = 0.0
    n_obs: int = 0          # steps that contributed to the EMA
    state: int = STATE_ABSENT
    last_bound: float = 1.0
    last_met: bool = True


class PushbackController:
    """Holds a_i per task, updates it once per step, and survives a checkpoint.

    The update reads the REDUCED (all-rank) sums for R and C^- that the
    diagnostics table already produces, so it runs on every rank identically
    and needs no collective of its own.
    """

    def __init__(self, cfg: PushbackConfig, task_names):
        cfg.validate()
        self.cfg = cfg
        self.task_names = list(task_names)
        self.tasks = {n: _TaskState() for n in self.task_names}
        self.step = 0

    # ---- what the loss reads -------------------------------------------
    def a_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """a_i in task order, as applied to THIS step. Fixed until update()."""
        return torch.tensor([self.tasks[n].a for n in self.task_names], device=device, dtype=dtype)

    # ---- once per step -------------------------------------------------
    def update(self, *, R, C_neg, ctl_tokens, live_groups, present) -> dict:
        """Compute a_i for the NEXT step from this step's reduced statistics.

        Every argument is a per-task mapping (name -> number) already summed
        across ranks. ``present[i]`` says the task had rows at all;
        ``live_groups[i]`` is the number of independent prompt groups with a
        live advantage, computed on the driver where the group ids are.
        """
        cfg = self.cfg
        out = {}
        for n in self.task_names:
            st = self.tasks[n]
            if not present.get(n, False):
                # Missing is not zero. Do not feed the EMA, do not change a.
                st.state = STATE_ABSENT
            elif R.get(n, 0.0) <= cfg.tiny:
                st.state = STATE_NO_PG
                st.a = 1.0
            elif int(live_groups.get(n, 0)) < cfg.min_live_groups or int(ctl_tokens.get(n, 0)) < cfg.min_ctl_tokens:
                # Signal exists but rests on too few independent groups or too
                # few measured tokens. The EMA still learns from it; a does not
                # move on it.
                st.state = STATE_FEW_GROUPS
                self._ema(st, R[n], C_neg.get(n, 0.0))
                st.a = 1.0
            else:
                st.state = STATE_OK
                self._ema(st, R[n], C_neg.get(n, 0.0))
                a, met = pushback_retention(cfg.eps, st.ema_R, st.ema_C, a_min=cfg.a_min, tiny=cfg.tiny)
                st.a, st.last_met = a, met
                st.last_bound = (cfg.eps * st.ema_R / st.ema_C) if st.ema_C > cfg.tiny else 1.0
            out.update(self._metrics_for(n))
        self.step += 1
        return out

    def _ema(self, st: _TaskState, R: float, C: float) -> None:
        d = self.cfg.ema_decay
        if st.n_obs == 0 or d == 0.0:
            st.ema_R, st.ema_C = float(R), float(C)
        else:
            st.ema_R = d * st.ema_R + (1.0 - d) * float(R)
            st.ema_C = d * st.ema_C + (1.0 - d) * float(C)
        st.n_obs += 1

    def _metrics_for(self, n: str) -> dict:
        st = self.tasks[n]
        p = f"actor/pushback/{{}}/{n}"
        return {
            p.format("a_next"): st.a,
            p.format("ema_R"): st.ema_R,
            p.format("ema_C_neg"): st.ema_C,
            p.format("ema_ratio"): (st.ema_C / st.ema_R) if st.ema_R > self.cfg.tiny else 0.0,
            p.format("bound"): st.last_bound,
            p.format("constraint_met"): 1.0 if st.last_met else 0.0,
            p.format("state"): float(st.state),
            p.format("n_obs"): float(st.n_obs),
        }

    # ---- checkpoint ----------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "version": 1,
            "cfg": asdict(self.cfg),
            "step": self.step,
            "tasks": {n: asdict(st) for n, st in self.tasks.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        if sd.get("version") != 1:
            raise ValueError(f"pushback state version {sd.get('version')} is not 1")
        saved_cfg = sd.get("cfg", {})
        live_cfg = asdict(self.cfg)
        drift = {k: (saved_cfg.get(k), live_cfg[k]) for k in live_cfg if saved_cfg.get(k) != live_cfg[k]}
        if drift:
            # A resumed run with a different eps would apply a's computed under
            # another rule. Refuse rather than blend.
            raise ValueError(f"pushback_control config changed across resume: {drift}")
        self.step = int(sd.get("step", 0))
        for n, d in sd.get("tasks", {}).items():
            if n in self.tasks:
                self.tasks[n] = _TaskState(**d)


def live_groups_by_task(uids, task_names, advantages: torch.Tensor, response_mask: torch.Tensor) -> dict:
    """Independent prompt groups per task with at least one live-advantage token.

    Computed on the DRIVER, which is the only place that sees the whole batch
    with its group ids: the actor sees micro-batches, and "distinct groups" does
    not sum across ranks. A group whose rollouts all scored the same has
    advantage 0 everywhere and is not evidence about the reward's direction --
    the controller's sufficiency test counts groups that carry one.

    Returns ``{task_name: n_groups}`` for the tasks present. Rows with no task
    name (padding) are skipped.
    """
    with torch.no_grad():
        live_row = ((advantages.detach() != 0) & (response_mask.detach() > 0)).any(dim=-1).cpu()
    seen: dict = {}
    for uid, name, live in zip(list(uids), list(task_names), live_row.tolist()):
        if name is None or not live:
            continue
        seen.setdefault(str(name), set()).add(str(uid))
    return {name: len(groups) for name, groups in seen.items()}
