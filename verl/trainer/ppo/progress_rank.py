"""(a): rank a stuck group's rollouts by progress, at a per-task scale set by an EMA.

THE PROBLEM. A group whose eight rollouts all fail gives GRPO nothing to learn
from. At step 25 of the 3-task control those groups held 57% of ALFWorld's
response tokens, 62% of Search's and 51% of WebShop's, and none of ALFWorld's or
WebShop's carried a signal: every row in each of them scored the same (every turn
took the -0.1 format penalty), so their advantage was float32 rounding -- the 1.3%
and 0.6% of PG a mass probe attributed to them is that rounding (see FORMAT
CONTRAST). 34 of Search's 36 carried exactly none; the other two, whose rollouts
differed in format, carried 33% of Search's (mass probe, 70 groups per task).

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
``min_top_k`` steps (WebShop 2: a results page showing the product is hit by
chance; ALFWorld 1 under its milestone count, whose first milestone -- took an
object of the target type -- is not; Search 1, where there is one step).
A stuck group with no difference is (b)'s, and is left at zero here.

HOW c IS SET, PER TASK, EVERY STEP.

    M_t   this step's token-mean |A| over ALL the task's real rows, BEFORE (a)
    E_t   <- (1 - alpha) E_{t-1} + alpha M_t, floored        (EMA; alpha 0.2)
    P_t   this step's token-mean A over the rows of SUCCESSFUL trajectories in
          the task's live groups, BEFORE (a): the push a success gets
    S_t   <- (1 - alpha) S_{t-1} + alpha P_t, floored; not updated on a step
          with no live group
    u_t   sum over the rows of fired trajectories of |score| x tokens, divided by
          ALL the task's real tokens (the same denominator as M_t -- not a mean
          within the fired trajectories)
    c_t   = rho * E_t / u_t, capped so that c_t * max|score| <= kappa * S_t

so the injected token-mean |A| is c_t * u_t = rho * E_t unless capped: a fixed
share rho of the task's TYPICAL reward-driven update. Token-mean units because
normalize_loss_by_task gives every row of a task the same weight within a step,
so a ratio of token-means IS a ratio of loss mass.

THE CAP IS ANCHORED ON WHAT A SUCCESS GETS, NOT ON E. With kappa 0.5 no failed
trajectory is pushed up more than half as hard, per token, as the task's
successes typically are. It was kappa * E first, and the step-75 probe records
showed why that anchor was wrong. E is diluted by every zero row and inflated by
the format channel's few large rows, so the same kappa meant different things
per task: at training group counts the cap bound on every firing step in WebShop
and Search, and the top push came to 0.71-0.75 of the median success push in
WebShop (above its weakest successes, 0.28), 0.24-0.43 in Search and 0.14-0.28
in ALFWorld. S measures the comparison the cap exists for, and it does not move
with how many groups are stuck or whether the format penalty is lit.

WHY AN EMA AND NOT M_t. At 15 groups Search's M_t swings +-52% step to step,
and in about one step in ten it is exactly 0 -- a ratio to M_t would move c by
several times per step and would switch (a) OFF on precisely the steps where it is
the only reward-driven signal. E stays positive through those steps.

WHAT THE SCALE IMPLIES. E is a mean over every row of the task, the zero-advantage
ones included, so it sits far below the |A| of the rows that carry a signal: at
step 25, 0.123 against 0.789 for Search (86% of its rows are exactly 0), 0.315
against 0.740 for ALFWorld and 0.411 against 0.835 for WebShop. As degenerate
groups grow, M and E fall, so (a)'s injected mass rho * E falls with them: (a) adds
least, in absolute terms, to the task that is most stuck. That is the price of "a
share of the task's typical update". Per token the cap decides instead: with (a)
firing on 0-1 groups a step, u is small and the cap binds often, and then the top
push is exactly kappa * S. progress_rank/<task>/capped, c_uncapped, c_cap,
top_push_over_success and injected_mean_abs_adv record how it plays out.

WHY u_t IS NOT SMOOTHED. c_t * u_t = rho * E_t whatever u_t is, so u_t is only the
conversion from scores to the target mass. When it is tiny (one group, a
one-step difference) c_t would be large; that is what the cap is for.

(a) WAITS FOR BOTH SCALES. Until a task has seen an update (E) and a success in a
live group (S), c is 0 for it.

M_t NEVER CONTAINS (a) ITSELF: it is read off the advantages before the addition,
otherwise the target would chase its own output.

FORMAT CONTRAST, AND WHY (a) ADDS RATHER THAN REPLACES. With every environment
reward at 0, a stuck group's advantage is non-zero only when the per-turn format
penalty makes its row scores differ (some turns valid, some not). That is judged
on the SCORES (term_mass.SCORE_SPREAD_EPS), never on |A|: a group whose rows all
score -0.1 comes back with |A| = 0.0074 on every row from float32 rounding.
(a) is added after the GRPO statistic is formed, so where the format channel does
act it keeps its full size -- a lone tagged row's +19.9 stays +19.9 -- and (a)
moves any row by at most kappa * E. Replacing the advantage in stuck groups would
delete that channel from all of them, and in the 3-task run where it was traced
(floor arm B') the policy learned to write its tags through all-fail groups.
ProGPO's gate instead -- replace only where every base score is below 1e-3 --
never opens while every turn is penalised, and afterwards only in a group without
a single invalid turn, which 8 x 50 ALFWorld rows rarely are. What (a) does to
the channel is recorded: its push on format-mixed groups (inject_*_mixed) and on
the invalid-turn rows themselves (inject_*_invalid).

SHADOWS, NEVER APPLIED. On the same rollouts the step also computes what ProGPO
(2607.22724) would do: its q_fail (groups whose every score is below tau_R =
1e-3), its first-visit coverage P = (D - 1) / T (D counted by the environment
managers, T the trajectory's turn rows), and whether P varies enough to fire
(population std >= tau_P = 1e-4). shadow/progpo/* and shadow/coverage/* set that
beside k: how often each fires in the same stuck groups, how the two order the
same rollouts (Spearman on average ranks), how often each gives a winner nothing.

GROUP RECORDS. Summaries cannot answer the next question, and twice a probe had to
be re-run for want of the rows under one. So each step also leaves one record per
group in ``last_group_records`` -- per trajectory k, K, D, turns, tokens, reward,
invalid turns and (a)'s score; per group the score spread, ProGPO's gate, the
largest base |A| and the |A| mass (a) injected; per task c, c before the cap, the
cap, E, S and the token count -- which the trainer writes out as JSONL
(scripts/report_progress_groups.py reads them back).

rho = 0 leaves every advantage bit-identical to control; that is stage 1's
identity check.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from verl.trainer.ppo.term_mass import SCORE_SPREAD_EPS, score_spread

__all__ = ["PROGRESS_K_KEY", "PROGRESS_TOTAL_KEY", "COVERAGE_D_KEY", "DEFAULT_MIN_TOP_K",
           "PROGPO_TAU_R", "PROGPO_TAU_P", "failed_groups", "trajectory_progress",
           "score_stuck_groups", "average_ranks", "spearman", "coverage_progress",
           "ProgressRankController"]

PROGRESS_K_KEY = "progress_k"
PROGRESS_TOTAL_KEY = "progress_total"
COVERAGE_D_KEY = "coverage_d"
DEFAULT_MIN_TOP_K = {"alfworld": 1, "webshop": 2, "search": 1}
# ProGPO's implementation thresholds (2607.22724, Section 8 settings): a group is
# all-fail when every base score is below TAU_R, and its coverage fires when the
# population std of P reaches TAU_P.
PROGPO_TAU_R = 1e-3
PROGPO_TAU_P = 1e-4

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


def average_ranks(x) -> np.ndarray:
    """Ranks 1..n, ties sharing their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> Optional[float]:
    """Spearman's rho on average ranks; None when either side does not vary."""
    rx, ry = average_ranks(x), average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def coverage_progress(d: Optional[float], turns: int) -> Optional[float]:
    """ProGPO's P = (D - 1) / T; None without a count or a turn."""
    if d is None or turns <= 0:
        return None
    return (float(d) - 1.0) / float(turns)


class ProgressRankController:
    """Holds each task's EMAs across steps and turns scores into advantage.

    The only state is the two EMAs per task (E, the typical update; S, the typical
    success push), and it is part of the run: the trainer saves it beside the
    checkpoint so a resumed run does not restart (a) from uninitialised scales.
    """

    def __init__(self, *, rho: float, ema_alpha: float = 0.2, ema_floor: float = 0.01,
                 cap_kappa: float = 0.5, min_top_k: Optional[Dict[str, float]] = None,
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
        # S: the EMA of the push a success gets in a live group (the cap's anchor).
        self.success_ema: Dict[str, Optional[float]] = {t: None for t in self.tasks}
        # The last apply()'s groups, one dict per group; see GROUP RECORDS above.
        self.last_group_records: List[dict] = []

    # --- persistence ------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"version": 2, "ema": dict(self.ema), "success_ema": dict(self.success_ema)}

    def load_state_dict(self, state: dict) -> None:
        state = state or {}
        for t, v in state.get("ema", {}).items():
            self.ema[str(t)] = None if v is None else float(v)
        # Version 1 carried E only; S then starts uninitialised and (a) waits for a live group.
        for t, v in state.get("success_ema", {}).items():
            self.success_ema[str(t)] = None if v is None else float(v)

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

    def _update_success_ema(self, task: str, p: Optional[float]) -> Optional[float]:
        # No live group, or none whose successes were pushed up: nothing to learn
        # the anchor from this step, so it keeps its last value.
        if p is not None and p > 0.0:
            prev = self.success_ema.get(task)
            self.success_ema[task] = (max(self.floor, p) if prev is None
                                      else max(self.floor, (1.0 - self.alpha) * prev + self.alpha * p))
        return self.success_ema.get(task)

    def apply(self, *, advantages: torch.Tensor, mask: torch.Tensor, uids, tuids, task_names,
              episode_rewards, k_rows, total_rows, real_rows: np.ndarray,
              stat_rows: np.ndarray, row_scores=None, valid_rows=None,
              coverage_rows=None) -> tuple:
        """Return ``(new_advantages, metrics)``; ``advantages`` is not modified.

        ``mask``             the response mask the actor's loss uses, (rows, resp)
        ``episode_rewards``  per row, the environment's reward for its trajectory
        ``real_rows``        False for adjust_batch's padding copies (weight 0)
        ``stat_rows``        rows in the GRPO statistic: real and not excluded
        ``row_scores``       per row, the score GRPO normalised (token_level_rewards
                             summed). Without it the format split and ProGPO's
                             gate are not reported.
        ``valid_rows``       per row, is_action_valid. Without it (a)'s push on
                             invalid turns is not reported.
        ``coverage_rows``    per row, ProGPO's D as of that turn. Without it the
                             coverage shadow is not reported.

        The step's per-group records are left in ``self.last_group_records``.
        """
        n = advantages.shape[0]
        m = mask.to(torch.float64)
        tokens = m.sum(-1).cpu().numpy()
        abs_adv = (advantages.to(torch.float64).abs() * m).sum(-1).cpu().numpy()
        signed = (advantages.to(torch.float64) * m).sum(-1).cpu().numpy()
        real = np.asarray(real_rows, dtype=bool)
        stat = np.asarray(stat_rows, dtype=bool)
        names = np.asarray([str(x) for x in task_names])
        scores = None if row_scores is None else np.asarray(row_scores, dtype=float).reshape(-1)
        invalid = (None if valid_rows is None
                   else np.asarray([_finite(v) == 0.0 for v in valid_rows], dtype=bool))

        real_idx = [i for i in range(n) if real[i]]
        groups = failed_groups(uids, tuids, names, episode_rewards, real_idx)
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

        # Per group: do the row scores differ (the format channel acts), and would
        # ProGPO's gate open (every base score below tau_R)?
        spread: Dict[str, float] = {}
        gate: Dict[str, bool] = {}
        row_mixed = np.zeros(n, dtype=bool)
        if scores is not None:
            for uid, g in groups.items():
                grows = [i for i in g["rows"] if stat[i]] or list(g["rows"])
                spread[uid] = score_spread(scores[grows])
                gate[uid] = bool(np.max(np.abs(scores[grows])) < PROGPO_TAU_R)
                if spread[uid] > SCORE_SPREAD_EPS:
                    row_mixed[g["rows"]] = True

        # Per trajectory, over its real rows.
        traj: Dict[str, dict] = {}
        for i in real_idx:
            t = str(tuids[i])
            x = traj.get(t)
            if x is None:
                x = traj[t] = {"task": names[i], "turns": 0, "tokens": 0.0, "reward": None,
                               "invalid": 0, "d": None, "d_ok": coverage_rows is not None}
            x["turns"] += 1
            x["tokens"] += float(tokens[i])
            r = _finite(episode_rewards[i])
            if r is not None:
                x["reward"] = r if x["reward"] is None else max(x["reward"], r)
            if invalid is not None and invalid[i]:
                x["invalid"] += 1
            if coverage_rows is not None:
                d = _finite(coverage_rows[i])
                if d is None:
                    x["d_ok"] = False
                else:
                    x["d"] = d if x["d"] is None else max(x["d"], d)
        for x in traj.values():
            x["won"] = bool(x["reward"] is not None and x["reward"] > 0.0)
            if not x["d_ok"]:
                x["d"] = None

        # The rows the cap is anchored on: successful trajectories in live groups,
        # judged by the environment's reward like "stuck" is.
        live_success = np.zeros(n, dtype=bool)
        for g in groups.values():
            xs = {str(tuids[i]) for i in g["rows"]}
            if len(xs) < 2 or any(traj[t]["reward"] is None for t in xs):
                continue
            wins = [traj[t]["won"] for t in xs]
            if any(wins) and not all(wins):
                for i in g["rows"]:
                    if traj[str(tuids[i])]["won"]:
                        live_success[i] = True

        metrics: Dict[str, float] = {}
        coef = np.zeros(n, dtype=float)
        per_task: Dict[str, dict] = {}
        for task in self.tasks:
            rows = real & (names == task)
            p = f"progress_rank/{task}"
            task_tokens = float(tokens[rows].sum())
            if task_tokens <= 0:
                continue
            m_t = float(abs_adv[rows].sum()) / task_tokens
            ema = self._update_ema(task, m_t)
            srows = rows & live_success
            s_tokens = float(tokens[srows].sum())
            p_t = (float(signed[srows].sum()) / s_tokens) if s_tokens > 0 else None
            s_ema = self._update_success_ema(task, p_t)
            fired_rows = rows & (row_score != 0.0)
            u = float((np.abs(row_score[fired_rows]) * tokens[fired_rows]).sum()) / task_tokens
            max_abs = float(np.abs(row_score[rows]).max()) if rows.any() else 0.0

            c, capped = 0.0, 0.0
            c_uncapped, cap = None, None
            if self.rho > 0.0 and ema is not None and s_ema is not None and u > 0.0 and max_abs > 0.0:
                c_uncapped = self.rho * ema / u
                # No failed trajectory gets more, per token, than kappa times the
                # push the task's successes typically get.
                cap = self.kappa * s_ema / max_abs
                c = c_uncapped
                if c > cap:
                    c, capped = cap, 1.0
                # How far above its cap the target sat: capped alone says only that it did.
                metrics[f"{p}/c_uncapped"] = c_uncapped
                metrics[f"{p}/c_cap"] = cap
            coef[names == task] = c
            per_task[task] = {"c": c, "capped": bool(capped), "c_uncapped": c_uncapped, "c_cap": cap,
                              "ema": ema, "success_push_ema": s_ema, "task_tokens": task_tokens}

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
            if p_t is not None:
                metrics[f"{p}/success_push"] = p_t
            metrics[f"{p}/success_push_ema"] = float("nan") if s_ema is None else s_ema
            # The largest push any failed token got from (a); at most kappa * S.
            metrics[f"{p}/top_push"] = c * max_abs
            if s_ema:
                metrics[f"{p}/top_push_over_success"] = c * max_abs / s_ema
            metrics[f"{p}/score_mass"] = u
            metrics[f"{p}/injected_mean_abs_adv"] = c * u
            # c * u / E = min(rho, kappa * u / max|score|): the share actually injected.
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
            if scores is not None:
                # The same push, split by whether the format channel already moves
                # the group. The mixed share is the part of (a) outside ProGPO's
                # non-interference support (their Proposition 4.3).
                for cls, sel in (("mixed", row_mixed), ("uniform", ~row_mixed)):
                    metrics[f"{p}/inject_up_{cls}"] = float(inj[rows & sel & (row_score > 0)].sum()) / task_tokens
                    metrics[f"{p}/inject_down_{cls}"] = float(-inj[rows & sel & (row_score < 0)].sum()) / task_tokens
                if up + down > 0:
                    metrics[f"{p}/inject_share_mixed"] = (
                        (metrics[f"{p}/inject_up_mixed"] + metrics[f"{p}/inject_down_mixed"]) / (up + down))
            if invalid is not None:
                # On the invalid-turn rows themselves: a push-up here offsets the
                # format penalty's push-down on exactly the rows it is aimed at.
                metrics[f"{p}/inject_up_invalid"] = float(inj[rows & invalid & (row_score > 0)].sum()) / task_tokens
                metrics[f"{p}/inject_down_invalid"] = float(-inj[rows & invalid & (row_score < 0)].sum()) / task_tokens

            # Winners the count gives nothing: the analogue of ProGPO's Proposition
            # 4.1(ii), which puts every success above zero coverage. Trajectories
            # without a sequence (K = 0) are not counted.
            won = [traj_prog.get(t, (0.0, 0.0)) for t, x in traj.items() if x["task"] == task and x["won"]]
            won = [kk for kk, KK in won if KK > 0]
            if won:
                metrics[f"{p}/won_with_zero_k"] = sum(1 for kk in won if kk == 0) / len(won)
            won_d = [x["d"] for x in traj.values() if x["task"] == task and x["won"] and x["d"] is not None]
            if won_d:
                metrics[f"shadow/coverage/{task}/won_with_zero"] = sum(1 for d in won_d if d <= 1) / len(won_d)

        # Group counts, the shadows, and the records.
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        shadow: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        rhos: Dict[str, List[float]] = defaultdict(list)
        records: List[dict] = []
        for uid, g in groups.items():
            task = str(g.get("task") or "")
            if task not in self.tasks:
                continue
            counts[task]["groups"] += 1
            grows = [i for i in g["rows"] if real[i]]
            trajs = sorted({str(tuids[i]) for i in grows})
            xs = [traj[t] for t in trajs]
            stuck = g.get("status") == "stuck"
            if stuck:
                status = "stuck"
                counts[task]["groups_stuck"] += 1
                if uid in spread:
                    counts[task]["stuck_mixed" if spread[uid] > SCORE_SPREAD_EPS else "stuck_uniform"] += 1
            elif len(xs) >= 2 and all(x["reward"] is not None for x in xs):
                status = "saturated" if all(x["won"] for x in xs) else "live"
            else:
                status = "other"
            verdict = verdicts.get(uid, {}).get("verdict")

            # ProGPO on this group: its gate, and whether its coverage would fire.
            if gate.get(uid, False):
                shadow[task]["gate"] += 1
            P = [coverage_progress(x["d"], x["turns"]) for x in xs]
            cov_known = len(xs) >= 2 and all(v is not None for v in P)
            cov_fires = cov_known and float(np.std(np.asarray(P, dtype=float))) >= PROGPO_TAU_P
            if cov_known:
                shadow[task]["known"] += 1
            if stuck and cov_fires:
                shadow[task]["stuck_fired"] += 1
            if gate.get(uid, False) and cov_fires:
                shadow[task]["progpo_fired"] += 1
            if stuck and cov_fires and verdict == FIRED:
                shadow[task]["both_fired"] += 1
                r_s = spearman([traj_prog.get(t, (0.0, 0.0))[0] for t in trajs], P)
                if r_s is not None:
                    rhos[task].append(r_s)

            info = per_task.get(task, {})
            records.append({
                "task": task, "uid": uid, "status": status, "verdict": verdict, "trajs": trajs,
                "k": [traj_prog.get(t, (0.0, 0.0))[0] for t in trajs],
                "K": [traj_prog.get(t, (0.0, 0.0))[1] for t in trajs],
                "coverage_d": [x["d"] for x in xs],
                "turns": [x["turns"] for x in xs],
                "tokens": [x["tokens"] for x in xs],
                "reward": [x["reward"] for x in xs],
                "won": [x["won"] for x in xs],
                "invalid_turns": None if invalid is None else [x["invalid"] for x in xs],
                "score": [traj_score.get(t, 0.0) for t in trajs],
                "score_spread": spread.get(uid),
                "progpo_gate": gate.get(uid),
                "base_abs_adv_max": (float(np.max(abs_adv[grows] / np.maximum(tokens[grows], 1.0)))
                                     if grows else 0.0),
                "injected_abs_mass": float(np.sum(np.abs(row_score[grows] * coef[grows]) * tokens[grows])),
                "c": info.get("c"), "capped": info.get("capped"), "c_uncapped": info.get("c_uncapped"),
                "c_cap": info.get("c_cap"), "ema": info.get("ema"),
                "success_push_ema": info.get("success_push_ema"), "task_tokens": info.get("task_tokens"),
            })
        for rec in verdicts.values():
            counts[rec["task"]][f"stuck_{rec['verdict']}"] += 1
        for task, cs in counts.items():
            for key, v in cs.items():
                metrics[f"progress_rank/{task}/{key}"] = float(v)
            if cs.get("groups", 0):
                metrics[f"progress_rank/{task}/q_fail"] = cs.get("groups_stuck", 0) / cs["groups"]
                if scores is not None:
                    # ProGPO's own q_fail on this batch; their lambda_eff is 0.3 times it.
                    metrics[f"shadow/progpo/{task}/q_fail"] = shadow[task].get("gate", 0) / cs["groups"]
            if shadow[task].get("known", 0):
                metrics[f"shadow/coverage/{task}/stuck_fired"] = float(shadow[task].get("stuck_fired", 0))
                metrics[f"shadow/coverage/{task}/both_fired"] = float(shadow[task].get("both_fired", 0))
                if scores is not None:
                    metrics[f"shadow/progpo/{task}/fired"] = float(shadow[task].get("progpo_fired", 0))
                if rhos[task]:
                    metrics[f"shadow/coverage/{task}/spearman_vs_k"] = float(np.mean(rhos[task]))
        self.last_group_records = records

        delta = torch.as_tensor(row_score * coef, dtype=advantages.dtype, device=advantages.device)
        if not bool((delta != 0).any()):
            # Nothing to add: hand back the very same tensor, so rho = 0 (or a step
            # where nothing fired) is bit-identical to control by construction.
            return advantages, metrics
        new = advantages + delta.unsqueeze(-1) * mask.to(advantages.dtype)
        return new, metrics
