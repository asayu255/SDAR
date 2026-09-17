"""(a): rank a stuck group's rollouts by progress, at a per-task scale set by an EMA.

THE PROBLEM. A group whose eight rollouts all fail has an advantage of exactly
zero, so GRPO takes nothing from it. At step 25 of the 3-task control those
groups held 57% of ALFWorld's response tokens, 62% of Search's and 51% of
WebShop's. ALFWorld's and WebShop's carried 1.3% and 0.6% of those tasks'
reward-driven update; 34 of Search's 36 carried exactly none, and the other two
-- format-mixed -- carried 33% of Search's (mass probe, 70 groups per task).

WHAT (a) DOES. The environment counts, for every rollout, how many steps of the
task's correct sequence it actually carried out (agent_system/environments/
progress.py: k of K). Inside a stuck group whose rollouts got different
distances, the further ones are pushed up and the shorter ones down:

    score_i = (k_i - mean k) / K          in [-1, 1]
    A_i    += c_task * score_i            on every turn row of trajectory i

The mean is taken over the same samples the group's GRPO statistic uses -- turn
rows under compute_mean_std_cross_steps, trajectories otherwise -- so the added
term sums to zero over exactly what GRPO's own advantage sums to zero over. The
ordinary advantage (format penalty included) is computed first and left as it is.

A group fires only when its k are not all equal AND its furthest rollout reached
``min_top_k`` steps (ALFWorld and WebShop 2: the first step, "go to X" / a results
page showing the product, is hit by chance; Search 1, where there is one step).
A stuck group with no difference is (b)'s, and is left at zero here.

HOW c IS SET, PER TASK, EVERY STEP.

    M_t   this step's token-mean |A| over the task's real rows, BEFORE (a)
    E_t   <- (1 - alpha) E_{t-1} + alpha M_t, floored        (EMA; alpha 0.2)
    u_t   token-mean of |score| over the task's rows, fired trajectories only
    c_t   = rho * E_t / u_t, capped so that c_t * max|score| <= kappa * E_t

so the injected token-mean |A| is c_t * u_t = rho * E_t: a fixed share rho of
the task's TYPICAL reward-driven update, whatever the task and whatever the step.
Token-mean units because normalize_loss_by_task gives every row of a task the
same weight within a step, so a ratio of token-means IS a ratio of loss mass.

WHY AN EMA AND NOT M_t. At 15 groups Search's M_t swings +-52% step to step,
and in about one step in ten it is exactly 0 -- a ratio to M_t would move c by
several times per step and would switch (a) OFF on precisely the steps where it is
the only reward-driven signal. E stays positive through those steps.

WHY u_t IS NOT SMOOTHED. c_t * u_t = rho * E_t whatever u_t is, so u_t is only the
conversion from scores to the target mass. When it is tiny (one group, a
one-step difference) c_t would be large; that is what the cap is for.

M_t NEVER CONTAINS (a) ITSELF: it is read off the advantages before the addition,
otherwise the target would chase its own output.

rho = 0 leaves every advantage bit-identical to control; that is stage 1's
identity check.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

__all__ = ["PROGRESS_K_KEY", "PROGRESS_TOTAL_KEY", "DEFAULT_MIN_TOP_K", "failed_groups",
           "trajectory_progress", "score_stuck_groups", "ProgressRankController"]

PROGRESS_K_KEY = "progress_k"
PROGRESS_TOTAL_KEY = "progress_total"
DEFAULT_MIN_TOP_K = {"alfworld": 2, "webshop": 2, "search": 1}

# A group verdict, per stuck group.
FIRED = "fired"
NO_PROGRESS = "no_progress"      # some rollout has no sequence (K = 0 / missing)
NO_DIFFERENCE = "no_difference"  # all k equal -- (b)'s groups
TOP_BELOW_MIN = "top_below_min"  # differ, but nobody reached min_top_k


def _finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def failed_groups(uids, tuids, task_names, episode_rewards, rows: Iterable[int]) -> Dict[str, Dict]:
    """``{uid: {rows, task, status}}``; status "stuck" when every rollout failed.

    JUDGED ON THE ENVIRONMENT'S REWARD, NOT ON THE ROW RETURN GRPO SEES. The row
    return is the episode reward minus 0.1 on each invalid turn, and
    classify_groups takes a trajectory's return as the max over its rows -- so a
    failure whose every turn was invalid reads -0.1 against its siblings' 0, and
    a group of eight failures is called live. Those are exactly the groups with
    no success to learn from, so for (a) a group is stuck when no rollout scored:
    every task here pays 0 for a failure (ALFWorld 0/10, WebShop 0/10, Search 0/1).
    """
    out: Dict[str, Dict] = {}
    best: Dict[str, Dict[str, float]] = defaultdict(dict)
    for i in rows:
        u = str(uids[i])
        g = out.setdefault(u, {"rows": [], "task": str(task_names[i]), "status": "stuck"})
        g["rows"].append(i)
        r = _finite(episode_rewards[i])
        t = str(tuids[i])
        best[u][t] = max(best[u].get(t, float("-inf")), r if r is not None else float("-inf"))
    for u, g in out.items():
        vals = list(best[u].values())
        if len(vals) < 2 or max(vals) > 0.0 or not all(math.isfinite(v) for v in vals):
            g["status"] = "other"
    return out


def trajectory_progress(tuids, k_rows, total_rows, rows: Iterable[int]) -> Dict[str, tuple]:
    """``{traj: (k, K)}`` over the given rows; K = 0 when any row lacks a count.

    k is the maximum over the trajectory's rows: every row carries the count as
    of its own turn, and the count only grows, so the last turn holds it -- but
    the maximum does not depend on the rows arriving in turn order.
    """
    out: Dict[str, list] = {}
    for i in rows:
        t = str(tuids[i])
        k, n = _finite(k_rows[i]), _finite(total_rows[i])
        rec = out.setdefault(t, [0.0, 0.0, True])
        if k is None or n is None or n <= 0:
            rec[2] = False
            continue
        rec[0] = max(rec[0], k)
        rec[1] = max(rec[1], n)
    return {t: ((k, n) if ok and n > 0 else (0.0, 0.0)) for t, (k, n, ok) in out.items()}


def score_stuck_groups(groups: Dict[str, Dict], *, tuids, stat_rows: np.ndarray,
                       traj_prog: Dict[str, tuple], min_top_k: Dict[str, float],
                       tasks: Iterable[str], cross_steps: bool = True) -> Dict[str, Dict]:
    """Per stuck group on ``tasks`` (failed_groups' verdict): whether it fired, and the scores.

    ``stat_rows`` marks the rows that enter the group's GRPO statistic (real, not
    excluded); the mean k is weighted the way that statistic weights samples.
    """
    tasks = set(tasks)
    out: Dict[str, Dict] = {}
    for uid, g in groups.items():
        task = str(g.get("task") or "")
        if g.get("status") != "stuck" or task not in tasks:
            continue
        rows = [i for i in g["rows"] if stat_rows[i]]
        weight: Dict[str, float] = defaultdict(float)
        for i in rows:
            t = str(tuids[i])
            if cross_steps:
                weight[t] += 1.0
            else:
                weight[t] = 1.0
        trajs = sorted(weight)
        prog = [traj_prog.get(t, (0.0, 0.0)) for t in trajs]
        rec = {"task": task, "verdict": None, "scores": {},
               "k": [p[0] for p in prog], "K": max((p[1] for p in prog), default=0.0)}
        out[uid] = rec
        if not trajs or any(n <= 0 for _, n in prog):
            rec["verdict"] = NO_PROGRESS
            continue
        ks = np.asarray([p[0] for p in prog], dtype=float)
        K = float(max(p[1] for p in prog))
        if float(ks.max()) == float(ks.min()):
            rec["verdict"] = NO_DIFFERENCE
            continue
        if float(ks.max()) < float(min_top_k.get(task, 2)):
            rec["verdict"] = TOP_BELOW_MIN
            continue
        w = np.asarray([weight[t] for t in trajs], dtype=float)
        mean = float((w * ks).sum() / w.sum())
        rec["verdict"] = FIRED
        rec["scores"] = {t: (float(k) - mean) / K for t, k in zip(trajs, ks)}
    return out


class ProgressRankController:
    """Holds each task's EMA across steps and turns scores into advantage.

    The only state is the EMA per task, and it is part of the run: the trainer
    saves it beside the checkpoint so a resumed run does not restart (a) from an
    uninitialised scale.
    """

    def __init__(self, *, rho: float, ema_alpha: float = 0.2, ema_floor: float = 0.01,
                 cap_kappa: float = 1.0, min_top_k: Optional[Dict[str, float]] = None,
                 tasks: Iterable[str] = ("alfworld", "webshop", "search"),
                 cross_steps: bool = True):
        assert rho >= 0.0, f"progress_rank.rho must be >= 0, got {rho}"
        assert 0.0 < ema_alpha <= 1.0, f"progress_rank.ema_alpha must be in (0, 1], got {ema_alpha}"
        assert ema_floor > 0.0, "progress_rank.ema_floor must be > 0: it is what keeps c defined"
        assert cap_kappa > 0.0, f"progress_rank.cap_kappa must be > 0, got {cap_kappa}"
        self.rho = float(rho)
        self.alpha = float(ema_alpha)
        self.floor = float(ema_floor)
        self.kappa = float(cap_kappa)
        self.min_top_k = dict(DEFAULT_MIN_TOP_K)
        self.min_top_k.update({str(k): float(v) for k, v in (min_top_k or {}).items()})
        self.tasks = [str(t) for t in tasks]
        self.cross_steps = bool(cross_steps)
        self.ema: Dict[str, Optional[float]] = {t: None for t in self.tasks}

    # --- persistence ------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"version": 1, "ema": dict(self.ema)}

    def load_state_dict(self, state: dict) -> None:
        for t, v in (state or {}).get("ema", {}).items():
            self.ema[str(t)] = None if v is None else float(v)

    # --- one step ---------------------------------------------------------- #

    def _update_ema(self, task: str, m: float) -> Optional[float]:
        prev = self.ema.get(task)
        if prev is None:
            # Initialised by the first step that HAS a reward-driven update; until
            # then there is no scale to take a share of, and (a) stays off.
            if m > 0.0:
                self.ema[task] = max(self.floor, m)
        else:
            self.ema[task] = max(self.floor, (1.0 - self.alpha) * prev + self.alpha * m)
        return self.ema.get(task)

    def apply(self, *, advantages: torch.Tensor, mask: torch.Tensor, uids, tuids, task_names,
              episode_rewards, k_rows, total_rows, real_rows: np.ndarray,
              stat_rows: np.ndarray) -> tuple:
        """Return ``(new_advantages, metrics)``; ``advantages`` is not modified.

        ``mask``             the response mask the actor's loss uses, (rows, resp)
        ``episode_rewards``  per row, the environment's reward for its trajectory
        ``real_rows``        False for adjust_batch's padding copies (weight 0)
        ``stat_rows``        rows in the GRPO statistic: real and not excluded
        """
        n = advantages.shape[0]
        m = mask.to(torch.float64)
        tokens = m.sum(-1).cpu().numpy()
        abs_adv = (advantages.to(torch.float64).abs() * m).sum(-1).cpu().numpy()
        signed = (advantages.to(torch.float64) * m).sum(-1).cpu().numpy()
        real = np.asarray(real_rows, dtype=bool)
        stat = np.asarray(stat_rows, dtype=bool)
        names = np.asarray([str(x) for x in task_names])

        groups = failed_groups(uids, tuids, names, episode_rewards,
                               [i for i in range(n) if real[i]])
        traj_prog = trajectory_progress(tuids, k_rows, total_rows, range(n))
        verdicts = score_stuck_groups(groups, tuids=tuids, stat_rows=stat, traj_prog=traj_prog,
                                      min_top_k=self.min_top_k, tasks=self.tasks,
                                      cross_steps=self.cross_steps)
        # trajectory -> score, for the groups that fired
        traj_score: Dict[str, float] = {}
        for rec in verdicts.values():
            traj_score.update(rec["scores"])
        row_score = np.zeros(n, dtype=float)
        for i in range(n):
            s = traj_score.get(str(tuids[i]))
            # Rows kept out of the statistic by another mechanism get an advantage
            # of zero from compute_advantage; they stay at zero.
            if s is not None and (stat[i] or not real[i]):
                row_score[i] = s

        metrics: Dict[str, float] = {}
        coef = np.zeros(n, dtype=float)
        for task in self.tasks:
            rows = real & (names == task)
            p = f"progress_rank/{task}"
            task_tokens = float(tokens[rows].sum())
            if task_tokens <= 0:
                continue
            m_t = float(abs_adv[rows].sum()) / task_tokens
            ema = self._update_ema(task, m_t)
            fired_rows = rows & (row_score != 0.0)
            u = float((np.abs(row_score[fired_rows]) * tokens[fired_rows]).sum()) / task_tokens
            max_abs = float(np.abs(row_score[rows]).max()) if rows.any() else 0.0

            c, capped = 0.0, 0.0
            if self.rho > 0.0 and ema is not None and u > 0.0 and max_abs > 0.0:
                c = self.rho * ema / u
                cap = self.kappa * ema / max_abs
                if c > cap:
                    c, capped = cap, 1.0
            coef[names == task] = c

            inj = row_score * c * tokens
            up = float(inj[rows & (row_score > 0)].sum()) / task_tokens
            down = float(-inj[rows & (row_score < 0)].sum()) / task_tokens
            adv_up = float(np.clip(signed[rows], 0, None).sum()) / task_tokens
            adv_down = float(-np.clip(signed[rows], None, 0).sum()) / task_tokens

            metrics[f"{p}/c"] = c
            metrics[f"{p}/capped"] = capped
            metrics[f"{p}/ema_mean_abs_adv"] = float("nan") if ema is None else ema
            metrics[f"{p}/ema_initialized"] = float(ema is not None)
            metrics[f"{p}/mean_abs_adv"] = m_t
            metrics[f"{p}/score_mass"] = u
            metrics[f"{p}/injected_mean_abs_adv"] = c * u
            metrics[f"{p}/share_of_ema"] = (c * u / ema) if ema else 0.0
            if m_t > 0.0:
                # Against THIS step's update. Undefined on a step with none, which
                # is exactly when (a) is the whole reward-driven signal.
                metrics[f"{p}/share_of_step"] = c * u / m_t
            metrics[f"{p}/fired_token_share"] = float(tokens[fired_rows].sum()) / task_tokens
            # Push-up vs push-down, (a)'s and the ordinary advantage's, in the same
            # units. The ten-slot run failed as a 12:1 push-down; control reads 1.08.
            metrics[f"{p}/inject_up"] = up
            metrics[f"{p}/inject_down"] = down
            metrics[f"{p}/adv_up"] = adv_up
            metrics[f"{p}/adv_down"] = adv_down
            if up > 0:
                metrics[f"{p}/inject_down_over_up"] = down / up

        # group counts, per task
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for uid, g in groups.items():
            task = str(g.get("task") or "")
            if task not in self.tasks:
                continue
            counts[task]["groups"] += 1
            if g.get("status") == "stuck":
                counts[task]["groups_stuck"] += 1
                # Whether the format penalty already moves this group's advantage.
                # (a) adds on top either way; the split says how much of the
                # stuck groups' current update (a) will be diluting.
                grows = [i for i in g["rows"] if real[i]]
                amax = float(np.max(abs_adv[grows] / np.maximum(tokens[grows], 1.0))) if grows else 0.0
                counts[task]["stuck_mixed" if amax > 1e-12 else "stuck_uniform"] += 1
        for rec in verdicts.values():
            counts[rec["task"]][f"stuck_{rec['verdict']}"] += 1
        for task, cs in counts.items():
            for key, v in cs.items():
                metrics[f"progress_rank/{task}/{key}"] = float(v)

        delta = torch.as_tensor(row_score * coef, dtype=advantages.dtype, device=advantages.device)
        if not bool((delta != 0).any()):
            # Nothing to add: hand back the very same tensor, so rho = 0 (or a step
            # where nothing fired) is bit-identical to control by construction.
            return advantages, metrics
        new = advantages + delta.unsqueeze(-1) * mask.to(advantages.dtype)
        return new, metrics
