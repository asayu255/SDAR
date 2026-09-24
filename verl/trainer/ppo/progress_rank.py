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

SATURATED GROUPS (sat_rho). The same construction on the other degenerate kind:
a group whose eight rollouts ALL won gives GRPO nothing either, and late in
training those are 41-57% of ALFWorld's and WebShop's groups. Inside one, the
winners that took fewer turns are pushed up and the slower ones down:

    score_i = (mean T - T_i) / T_task      T_i = trajectory i's turns
    A_i    += c_sat * score_i              on every turn row of trajectory i

with the mean weighted like the GRPO statistic (zero-sum over its samples) and
T_task the task's turn cap (ALFWorld 50, WebShop 15). WHY TURNS: in saturated
ALFWorld groups the slowest winner took 20-23 turns against the fastest's 8, and
30-37% of its turns saw no new observation (4% for the fastest) -- the same
revisiting that makes every ALFWorld failure run into the 50-turn cap. A winner
that looped through invalid turns is longer, so it lands on the pushed-down side.
Search is left out: its turn count is the number of searches. A group fires only
when its turn counts differ by at least sat_min_spread (ALFWorld 2, WebShop 1).
c_sat = sat_rho * E / u_sat per task and step, under the same cap (no winning
token pushed up more than kappa * S); it waits for E and S like (a) does, and
sat_rho = 0 adds nothing. Prior art for the idea (step-discounted or step-decayed
returns, e.g. GiGPO and StraTA) changes the return everywhere; this adds a bounded,
zero-sum term only where the return has no spread.
"""

import math
import zlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from verl.trainer.ppo.term_mass import SCORE_SPREAD_EPS, score_spread

__all__ = ["PROGRESS_K_KEY", "PROGRESS_TOTAL_KEY", "COVERAGE_D_KEY", "DEFAULT_MIN_TOP_K",
           "PROGPO_TAU_R", "PROGPO_TAU_P", "failed_groups", "trajectory_progress",
           "score_stuck_groups", "score_saturated_groups", "average_ranks", "spearman",
           "coverage_progress", "DEFAULT_SAT_TASKS", "DEFAULT_SAT_MIN_SPREAD",
           "DEFAULT_SAT_TURN_SCALE", "ProgressRankController", "group_status",
           "trajectory_metrics", "FIRST_ORDER_WINDOW",
           "think_block_metrics", "THINK_OPEN_IDS",
           "score_mixed_groups", "sign_preserving_scale", "FEW_FAILURES"]

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
NO_DIFFERENCE = "no_difference"  # all k equal -- (b)'s groups; in a saturated group, all turn counts equal
TOP_BELOW_MIN = "top_below_min"  # differ, but nobody reached min_top_k
SPREAD_BELOW_MIN = "spread_below_min"  # saturated: turn counts differ by less than sat_min_spread
GATE_CLOSED = "gate_closed"      # saturated: the task's gate is shut (stuck share >= saturated share)
FEW_FAILURES = "few_failures"    # mixed: fewer than two failed rollouts, nothing to rank

# Saturated groups (sat_rho): which tasks, the smallest turn difference that fires,
# and the scale a turn difference is divided by (the task's turn cap). Search is
# not on the default list: its turn count is the number of searches, and fewer is
# not better. With the gate (sat_gate) it can be listed and the condition decides.
DEFAULT_SAT_TASKS = ("alfworld", "webshop")
DEFAULT_SAT_MIN_SPREAD = {"alfworld": 2.0, "webshop": 1.0, "search": 1.0}
DEFAULT_SAT_TURN_SCALE = {"alfworld": 50.0, "webshop": 15.0, "search": 4.0}
# What a saturated group's turn difference is divided by. task_constant: the task's
# turn cap (the numbers above). document: the group's own document length -- ALFWorld's
# walkthrough as a set of lines, WebShop's 3 + options, Search's 2 (search, answer) --
# so twelve turns on a four-line game weigh more than twelve on an eight-line one.
# group_mean: the group's own weighted mean turns (no document at all). A per-group
# constant in every case: the order INSIDE a group is untouched, only the weight
# between groups moves, and the total mass is still sat_rho * E per task.
SAT_SCALE_MODES = ("task_constant", "document", "group_mean")
# The null control for the saturated-group term: "shuffle" keeps every group that fires,
# its turn-weighted zero-sum and its L1 mass, and permutes which winner gets which score.
SAT_PLACEBO_MODES = ("none", "shuffle", "shuffle_dose", "sign", "bonus")


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


def score_saturated_groups(groups: Dict[str, Dict], *, tuids, stat_rows: np.ndarray,
                           traj: Dict[str, dict], tasks: Iterable[str],
                           min_spread: Dict[str, float], turn_scale: Dict[str, float],
                           cross_steps: bool = True, scale_mode: str = "task_constant",
                           placebo: str = "none", placebo_seed: int = 0) -> Dict[str, Dict]:
    """Per SATURATED group on ``tasks`` (every rollout won): whether it fired, and the scores.

    The mirror of :func:`score_stuck_groups` with the count replaced by the turn
    count, sign reversed: ``score_i = (mean T - T_i) / turn_scale``, so the
    rollouts that won in fewer turns are pushed up. ``traj`` is apply()'s
    per-trajectory table (turns = real rows, won by the environment's reward);
    the mean is weighted the way the GRPO statistic weights samples, so the term
    sums to zero over exactly what the group's own advantage sums to zero over.

    ``placebo="shuffle"`` is the null control for "the CONTENT of the ranking
    matters, not the channel": the same groups fire with the same turn-weighted
    zero-sum and the same L1 mass ``sum_i w_i |s_i|``, but which rollout gets which
    score is a permutation drawn from the group's uid (deterministic, so a resumed
    run redraws the same). The true scores are kept in ``scores_true`` for the
    records. What a permutation cannot preserve at the same time is the
    correlation with the rollouts' lengths, which is the point.

    ``placebo="shuffle_dose"`` draws the same permutation; apply() then sets its coefficient so
    that the APPLIED token-mean mass equals what the true ranking would have injected on the
    same rollouts (its own c, cap included). "shuffle" only matches the turn-weighted L1 mass
    of the scores, while the loss and the budget are token-weighted and the cap depends on
    max|s|, so its applied mass drifted to ~52% (ALFWorld) / ~63% (WebShop) of the true one
    in a replay of steps 105-124 (critical_plan_review.md, 2026-09-23). Matching the mass that way
    lets the placebo's PEAK push reach 1.5-1.8x the cap at the median and 8.6x at worst (replay of
    the satgate run), because the recentring after the permutation makes the scores spikier.

    ``placebo="sign"`` (the G1 control) matches EVERY magnitude: per group a coin keeps the true
    scores or flips their sign. That randomises the direction of the ranking AND the sign of the
    uniform bonus below, so G1 compares ranking + bonus against neither (found 2026-09-24).

    ``placebo="bonus"`` (user, 2026-09-25) keeps only the bonus. With the turn-weighted mean the
    true scores are s_i = r_i + b: a ranking r_i = (plain mean T - T_i) / scale that sums to 0 over
    the group's rollouts, plus b = mean_i s_i >= 0, the same for all of them. Every rollout gets b;
    apply() sizes the coefficient on the TRUE scores (cap included), so each group receives exactly
    the bonus component of the true arm's term, and true minus this arm is the ranking's effect.
    """
    tasks = set(tasks)
    out: Dict[str, Dict] = {}
    for uid, g in groups.items():
        task = str(g.get("task") or "")
        if task not in tasks:
            continue
        xs = [traj.get(t) for t in sorted({str(tuids[i]) for i in g["rows"]})]
        if len(xs) < 2 or any(x is None or x["reward"] is None or not x["won"] for x in xs):
            continue
        weight: Dict[str, float] = defaultdict(float)
        for i in g["rows"]:
            if stat_rows[i]:
                t = str(tuids[i])
                weight[t] = (weight[t] + 1.0) if cross_steps else 1.0
        trajs = sorted(weight)
        turns = np.asarray([traj[t]["turns"] for t in trajs], dtype=float)
        rec = {"task": task, "verdict": NO_DIFFERENCE, "scores": {}, "turns": turns.tolist(),
               "spread": float(turns.max() - turns.min()) if len(turns) else 0.0}
        out[uid] = rec
        if len(trajs) < 2 or rec["spread"] <= 0.0:
            continue
        if rec["spread"] < float(min_spread.get(task, 1.0)):
            rec["verdict"] = SPREAD_BELOW_MIN
            continue
        w = np.asarray([weight[t] for t in trajs], dtype=float)
        mean = float((w * turns).sum() / w.sum())
        scale = float(turn_scale.get(task) or turns.max())
        if scale_mode == "document":
            docs = [traj[t].get("doc_len") for t in trajs if traj[t].get("doc_len")]
            if docs:
                scale = float(max(docs))
        elif scale_mode == "group_mean":
            scale = max(mean, 1.0)
        rec["scale"] = scale
        rec["verdict"] = FIRED
        rec["scores"] = {t: (mean - float(n_turns)) / scale for t, n_turns in zip(trajs, turns)}
        if placebo in ("shuffle", "shuffle_dose"):
            true = np.asarray([rec["scores"][t] for t in trajs], dtype=float)
            rng = np.random.default_rng(zlib.crc32(str(uid).encode()) ^ int(placebo_seed))
            fake = true[rng.permutation(len(trajs))]
            fake = fake - float((w * fake).sum() / w.sum())          # turn-weighted zero-sum, as the true one
            mass_true, mass_fake = float((w * np.abs(true)).sum()), float((w * np.abs(fake)).sum())
            if mass_fake > 0.0:
                fake = fake * (mass_true / mass_fake)               # same L1 mass under the same weights
            rec["scores_true"] = dict(zip(trajs, true.tolist()))
            rec["scores"] = dict(zip(trajs, fake.tolist()))
            rec["placebo"] = placebo
        elif placebo == "sign":
            # Every magnitude of the true arm, only the DIRECTION randomised: per group a fair
            # coin (drawn from the uid, so a resume redraws it) keeps the true scores or flips
            # their sign. |score| per trajectory, the turn-weighted zero-sum, the token mass, the
            # peak and therefore the cap and the coefficient are all exactly the true arm's;
            # across groups the push favours fewer turns and more turns equally often.
            true = np.asarray([rec["scores"][t] for t in trajs], dtype=float)
            rng = np.random.default_rng(zlib.crc32(str(uid).encode()) ^ int(placebo_seed))
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            rec["scores_true"] = dict(zip(trajs, true.tolist()))
            rec["scores"] = dict(zip(trajs, (sgn * true).tolist()))
            rec["placebo"] = placebo
            rec["placebo_sign"] = sgn
        elif placebo == "bonus":
            # The bonus alone: the true scores' plain mean over the group's rollouts, for every
            # rollout (the ranking part sums to zero, so this is exactly b); no ordering is left.
            true = np.asarray([rec["scores"][t] for t in trajs], dtype=float)
            b = float(true.mean())
            rec["scores_true"] = dict(zip(trajs, true.tolist()))
            rec["scores"] = {t: b for t in trajs}
            rec["placebo"] = placebo
            rec["placebo_bonus"] = b
    return out


def score_mixed_groups(groups: Dict[str, Dict], *, tuids, stat_rows: np.ndarray,
                       traj_prog: Dict[str, tuple], traj: Dict[str, dict],
                       min_top_k: Dict[str, float], tasks: Iterable[str],
                       cross_steps: bool = True) -> Dict[str, Dict]:
    """Per MIXED group on ``tasks`` (some rollouts won, some did not): the failures ranked.

    The stuck-group ranking applied to the failed rollouts of a group whose outcome
    already separates them from its winners: ``score_i = (k_i - mean k_fail) / K`` for
    each failure, the mean taken over the failures only and weighted the way the GRPO
    statistic weights samples. The winners get nothing -- the reward has already said
    what it has to say about them -- and the term sums to zero over the failures, so
    the group's own push is redistributed among its losers, never added to.

    WHY IT IS WORTH HAVING. In the (a)+sat run's own group records, mixed groups held
    0.8x as many ALFWorld failures as stuck groups did (0.5x WebShop, 0.3x Search), and
    in half of the ALFWorld and WebShop ones the failures' k differ. Among WebShop's,
    k orders the failures the way the environment's own partial score does (mean
    Spearman +0.43 over 341 groups). Search's k almost never differs there (10%).

    ``traj`` is apply()'s per-trajectory table (won by the environment's reward).
    """
    tasks = set(tasks)
    out: Dict[str, Dict] = {}
    for uid, g in groups.items():
        task = str(g.get("task") or "")
        if task not in tasks:
            continue
        xs = {t: traj.get(t) for t in sorted({str(tuids[i]) for i in g["rows"]})}
        if len(xs) < 2 or any(x is None or x["reward"] is None for x in xs.values()):
            continue
        won = [x["won"] for x in xs.values()]
        if all(won) or not any(won):
            continue                        # saturated or stuck: another term's business
        weight: Dict[str, float] = defaultdict(float)
        for i in g["rows"]:
            t = str(tuids[i])
            if stat_rows[i] and not xs[t]["won"]:
                if cross_steps:
                    weight[t] += 1.0
                else:
                    weight[t] = 1.0
        fails = sorted(weight)
        prog = [traj_prog.get(t, (0.0, 0.0)) for t in fails]
        rec = {"task": task, "verdict": None, "scores": {},
               "k": [p[0] for p in prog], "K": max((p[1] for p in prog), default=0.0)}
        out[uid] = rec
        if len(fails) < 2:
            rec["verdict"] = FEW_FAILURES
            continue
        if any(n <= 0 for _, n in prog):
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
        w = np.asarray([weight[t] for t in fails], dtype=float)
        mean = float((w * ks).sum() / w.sum())
        rec["verdict"] = FIRED
        rec["scores"] = {t: (float(k) - mean) / K for t, k in zip(fails, ks)}
    return out


def sign_preserving_scale(adds: np.ndarray, base: np.ndarray) -> float:
    """The largest s in [0, 1] with base_i + s * adds_i <= 0 for every row.

    A mixed group's failures carry a negative advantage (their reward is below the
    group's mean); a ranking term large enough to lift one of them past zero would
    tell the policy to do MORE of a rollout that lost, reversing the outcome's own
    verdict on it. One scale for the whole group, so the term's zero sum survives.
    A failure whose advantage is already >= 0 (possible only through another
    mechanism's edit) leaves nothing to preserve and the group adds nothing.
    """
    if adds.size == 0:
        return 0.0
    if bool((base >= 0.0).any()):
        return 0.0
    up = adds > 0.0
    if not bool(up.any()):
        return 1.0
    return float(min(1.0, float(np.min(-base[up] / adds[up]))))


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


# How many steps the teacher's first-order effect is remembered over (see
# ProgressRankController.observe_update). 25 is the window the retirement rule
# was read with on the rho=0.1 run's records: fewer than 10 of the last 25 steps
# positive put WebShop's retirement at steps 145-149 and never retired the others.
FIRST_ORDER_WINDOW = 25

# The mid-response think block, as the POLICY writes it ("<th" -- the chat template
# has already closed an empty <think></think>, so the special id only appears when a
# response is re-tokenized from text) and as a re-encoding yields it.
THINK_OPEN_IDS = (13708, 151667)


def think_block_metrics(*, responses, mask, task_names, real=None,
                        ids=THINK_OPEN_IDS) -> Dict[str, float]:
    """Per task, the share of TURNS whose response opens a think block.

    THE CHANNEL, SEPARATED FROM THE ACTION FORMAT. ALFWorld and WebShop only score
    a turn whose response carries <think> </think>, and that is the channel whose
    erosion took WebShop to 0.000 at step 300 while the teacher pushed "<th" down
    (-6 at step 151, -128 at 296 in the token dumps).

    The run's own `valid_action_ratio` does move with it, but it is a late and
    shallow reading of it. It is ~0 until the policy ACQUIRES the format at all --
    the control crosses 0.5 at step 124 (ALFWorld) and 129 (WebShop), this run
    ~25 steps later -- then sits near 0.99, and through the collapse it falls only
    from 0.990 (steps 151-175) to 0.909 (276-300) while validation goes to 0.000.
    It also mixes the block with the action grammar, so a fall does not say which
    one broke. This counts the block itself, on the ids the collapse analysis
    used, and leaves the reading of it to whoever sets a threshold.

    ALFWORLD AND WEBSHOP ONLY, and a Search 0.000 is not damage. Search's rollouts
    carry no <think> block at all -- its responses open with the stray </think> the
    template invites -- and its own rule is the <search>/<answer> tag pair, which
    `valid_action_ratio/search` does report (0.983 on the control). Checked on 200
    recorded responses per task at step 150: ALFWorld 100%, WebShop 99%, Search 0%.

    Measured, never fed back: like the rest of trajectory_metrics, this is a
    reading of the rollouts and touches no advantage.
    """
    out: Dict[str, float] = {}
    if responses is None or mask is None or task_names is None:
        return out
    keep = mask.to(torch.bool)
    has = torch.zeros(responses.shape[0], dtype=torch.bool, device=responses.device)
    for tid in ids:
        has |= ((responses == int(tid)) & keep).any(dim=-1)
    has = has.detach().cpu().numpy()
    names = np.asarray(task_names, dtype=object).reshape(-1)
    rows = np.ones(len(names), dtype=bool) if real is None else np.asarray(real, dtype=bool)
    for task in dict.fromkeys(names[rows].tolist()):
        sel = rows & (names == task)
        if not sel.any():
            continue
        out[f"traj/{task}/think_block_share"] = float(has[sel].mean())
    if rows.any():
        out["traj/think_block_share"] = float(has[rows].mean())
    return out


def group_status(g: Dict, xs: List[dict]) -> str:
    """stuck (no rollout scored) / saturated (all scored) / live (both) / other."""
    if g.get("status") == "stuck":
        return "stuck"
    if len(xs) >= 2 and all(x["reward"] is not None for x in xs):
        return "saturated" if all(x["won"] for x in xs) else "live"
    return "other"


def trajectory_metrics(groups: Dict[str, Dict], traj: Dict[str, dict], *, tuids, real: np.ndarray,
                       tasks: Iterable[str], traj_prog: Dict[str, tuple],
                       turn_caps: Optional[Dict[str, float]] = None,
                       alt_prog: Optional[Dict[str, Dict[str, tuple]]] = None,
                       have_invalid: bool = False) -> Dict[str, float]:
    """What the rollouts looked like, per task: measured, never fed back.

    The step's success rate says whether a mechanism helps; these say HOW, and they
    are what the next mechanisms are judged on:

    SATURATED-GROUP RANKING (sat_rho) can only lift success by transferring
    "fewer turns" from the groups it acts on (all won) to the ones it does not (live
    groups, and the failures -- all of ALFWorld's failures stop at the turn cap). So:
      win_turns_saturated / win_turns_live / win_turns   winners' turns by group kind
      win_excess_saturated / win_excess_live   a winner's turns beyond its group's
                               fastest win (groups with >= 2 winners): the waste
                               the ranking targets, measured against the group's own
                               witness, not a constant
      win_fastest_saturated    that witness (the group's fastest win), averaged
      fail_turns, fail_at_cap  failures' turns; the share that ran into the task's
                               turn cap rather than ending on their own
      win_tokens_per_turn / fail_tokens_per_turn   fewer turns bought with longer
                               turns would show here
    THE TEACHER'S TERM (OPD, and what replaces it) eroded WebShop's format before
    its success moved (the </final> collapse): winners carrying an invalid turn went
    0% -> 37% -> 80% while success was still flat. So:
      win_with_invalid_turn    share of winners with at least one invalid turn
      win_invalid_turn_share / fail_invalid_turn_share   invalid turns / turns
    (a) on stuck groups:
      fail_progress            failures' k / K: do they get further before failing
      fail_distinct_obs_per_turn   ProGPO's P on failures (distinct observations per
                               turn): loops show as a low value; a shadow, not a signal
      fail_uncommitted         failures that never sent the terminal action (WebShop buy,
                               Search answer): the option-loop signature, per step
      win_revisits / fail_revisits   repeated actions per trajectory (a shadow ranking
                               key for saturated groups; progress.RevisitCounter)
      win_done_walkset / fail_done_walkset   ALFWorld walkthrough lines actually done
      groups_{stuck,live,saturated}_share   what the batch's groups were
    and, per alternative count (ALFWorld's milestones with / without "arrived"):
      progress_rank/<task>/alt_<name>/stuck_split_where_tied    stuck groups the
                               ranked count ties and the alternative would split
      progress_rank/<task>/alt_<name>/stuck_tied_where_split    and the reverse
    Turns are the trajectory's episode length where the batch carries it (rows can
    be dropped to make the batch divisible), else its rows here.
    """
    tasks = [str(t) for t in tasks]
    acc: Dict[str, Dict[str, list]] = {t: defaultdict(list) for t in tasks}
    alt_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_groups: Dict[str, Dict[str, int]] = {t: defaultdict(int) for t in tasks}
    caps = {str(k): float(v) for k, v in (turn_caps or {}).items()}
    turns_of = lambda x: float(x["length"]) if x.get("length") is not None else float(x["turns"])
    for uid, g in groups.items():
        task = str(g.get("task") or "")
        if task not in acc:
            continue
        trajs = sorted({str(tuids[i]) for i in g["rows"] if real[i]})
        xs = [traj[t] for t in trajs if t in traj]
        status = group_status(g, xs)
        n_groups[task][status] += 1
        a = acc[task]
        wins = [x for x in xs if x["won"]]
        if status in ("live", "saturated"):
            for x in wins:
                a[f"win_turns_{status}"].append(turns_of(x))
            if len(wins) >= 2:
                fastest = min(turns_of(x) for x in wins)
                a[f"win_excess_{status}"].extend(turns_of(x) - fastest for x in wins)
                if status == "saturated":
                    a["win_fastest_saturated"].append(fastest)
        for t, x in zip(trajs, xs):
            if x["reward"] is None:
                continue
            T = turns_of(x)
            rows_here = max(1, int(x["turns"]))
            if x["won"]:
                a["win_turns"].append(T)
                a["win_tokens_per_turn"].append(x["tokens"] / rows_here)
                if x.get("revisits") is not None:
                    a["win_revisits"].append(float(x["revisits"]))
                if x.get("done_walkset") is not None:
                    a["win_done_walkset"].append(float(x["done_walkset"]))
                if have_invalid:
                    a["win_with_invalid_turn"].append(float(x["invalid"] > 0))
                    a["_win_invalid"].append(float(x["invalid"]))
                    a["_win_rows"].append(float(x["turns"]))
            else:
                a["fail_turns"].append(T)
                a["fail_tokens_per_turn"].append(x["tokens"] / rows_here)
                if task in caps and x.get("length") is not None:
                    a["fail_at_cap"].append(float(T >= caps[task]))
                if have_invalid:
                    a["_fail_invalid"].append(float(x["invalid"]))
                    a["_fail_rows"].append(float(x["turns"]))
                k, K = traj_prog.get(t, (0.0, 0.0))
                if K > 0:
                    a["fail_progress"].append(k / K)
                cov = coverage_progress(x.get("d"), int(x["turns"]))
                if cov is not None:
                    a["fail_distinct_obs_per_turn"].append(cov)
                if x.get("task_score") is not None:
                    a["fail_task_score"].append(float(x["task_score"]))
                if x.get("revisits") is not None:
                    a["fail_revisits"].append(float(x["revisits"]))
                if x.get("done_walkset") is not None:
                    a["fail_done_walkset"].append(float(x["done_walkset"]))
                # Failures that never sent the task's terminal action (WebShop buy, Search
                # answer): the val-side signature of the option-loop collapse, per step.
                if x.get("committed") is not None:
                    a["fail_uncommitted"].append(float(not x["committed"]))
        if status == "stuck" and alt_prog:
            ranked = [traj_prog.get(t, (0.0, 0.0)) for t in trajs]
            if len(ranked) >= 2 and all(K > 0 for _, K in ranked):
                tied = len({k for k, _ in ranked}) == 1
                for name, prog in alt_prog.items():
                    alt = [prog.get(t, (0.0, 0.0)) for t in trajs]
                    if not all(K > 0 for _, K in alt):
                        continue
                    alt_tied = len({k for k, _ in alt}) == 1
                    ac = alt_counts[(task, name)]
                    ac["stuck_compared"] += 1
                    ac["stuck_split_where_tied"] += int(tied and not alt_tied)
                    ac["stuck_tied_where_split"] += int(alt_tied and not tied)
    out: Dict[str, float] = {}
    for task, a in acc.items():
        p = f"traj/{task}"
        for key, vals in a.items():
            if vals and not key.startswith("_"):
                out[f"{p}/{key}"] = float(np.mean(vals))
        for key in ("win_turns_live", "win_turns_saturated", "fail_turns"):
            if a.get(key):
                out[f"{p}/{key}_n"] = float(len(a[key]))
        if a.get("_win_rows"):
            out[f"{p}/win_invalid_turn_share"] = float(np.sum(a["_win_invalid"]) / max(1.0, np.sum(a["_win_rows"])))
        if a.get("_fail_rows"):
            out[f"{p}/fail_invalid_turn_share"] = float(np.sum(a["_fail_invalid"]) / max(1.0, np.sum(a["_fail_rows"])))
        total = sum(n_groups[task].values())
        if total:
            for status in ("stuck", "live", "saturated"):
                out[f"{p}/groups_{status}_share"] = n_groups[task].get(status, 0) / total
    for (task, name), cs in alt_counts.items():
        for key, v in cs.items():
            out[f"progress_rank/{task}/alt_{name}/{key}"] = float(v)
    return out


class ProgressRankController:
    """Holds each task's EMAs across steps and turns scores into advantage.

    The only state is the two EMAs per task (E, the typical update; S, the typical
    success push), and it is part of the run: the trainer saves it beside the
    checkpoint so a resumed run does not restart (a) from uninitialised scales.
    """

    def __init__(self, *, rho: float, ema_alpha: float = 0.2, ema_floor: float = 0.01,
                 cap_kappa: float = 0.5, min_top_k: Optional[Dict[str, float]] = None,
                 tasks: Iterable[str] = ("alfworld", "webshop", "search"),
                 cross_steps: bool = True, sat_rho: float = 0.0,
                 sat_tasks: Iterable[str] = DEFAULT_SAT_TASKS,
                 sat_min_spread: Optional[Dict[str, float]] = None,
                 sat_turn_scale: Optional[Dict[str, float]] = None,
                 mixed_rho: float = 0.0, mixed_tasks: Optional[Iterable[str]] = None,
                 sat_gate: bool = False, sat_turn_scale_mode: str = "task_constant",
                 sat_placebo: str = "none"):
        assert rho >= 0.0, f"progress_rank.rho must be >= 0, got {rho}"
        # The placebo arm: same firing, same mass, the ranking's content destroyed.
        assert sat_placebo in SAT_PLACEBO_MODES, (
            f"progress_rank.sat_placebo={sat_placebo!r}; expected one of {SAT_PLACEBO_MODES}")
        self.sat_placebo = str(sat_placebo)
        # THE GATE on the saturated-group term: per task, fire only while the EMA of
        # the task's saturated-group share exceeds the EMA of its stuck-group share --
        # while its reliably solved games outnumber its reliably failed ones, the
        # regime where the remaining failures are wandering and "fewer turns" is the
        # right push. Not "mean success > 1/2": games differ, and the shares are set
        # by the games near p = 1 and p = 0. Same alpha as E and S; no new constant.
        self.sat_gate = bool(sat_gate)
        assert sat_turn_scale_mode in SAT_SCALE_MODES, (
            f"progress_rank.sat_turn_scale_mode={sat_turn_scale_mode!r}; expected one of {SAT_SCALE_MODES}")
        self.sat_turn_scale_mode = str(sat_turn_scale_mode)
        # (a) on MIXED groups: its own share of the same E, on the failures only,
        # under the same cap and a per-group scale that keeps every failure <= 0.
        assert mixed_rho >= 0.0, f"progress_rank.mixed_rho must be >= 0, got {mixed_rho}"
        self.mixed_rho = float(mixed_rho)
        self.mixed_tasks = [str(t) for t in (mixed_tasks if mixed_tasks is not None else tasks)]
        unknown_mixed = [t for t in self.mixed_tasks if t not in [str(x) for x in tasks]]
        assert not unknown_mixed, f"progress_rank.mixed_tasks {unknown_mixed} are not in progress_rank.tasks"
        assert sat_rho >= 0.0, f"progress_rank.sat_rho must be >= 0, got {sat_rho}"
        self.sat_rho = float(sat_rho)
        self.sat_tasks = [str(t) for t in sat_tasks]
        unknown = [t for t in self.sat_tasks if t not in [str(x) for x in tasks]]
        assert not unknown, f"progress_rank.sat_tasks {unknown} are not in progress_rank.tasks"
        self.sat_min_spread = dict(DEFAULT_SAT_MIN_SPREAD)
        self.sat_min_spread.update({str(k): float(v) for k, v in (sat_min_spread or {}).items()})
        self.sat_turn_scale = dict(DEFAULT_SAT_TURN_SCALE)
        self.sat_turn_scale.update({str(k): float(v) for k, v in (sat_turn_scale or {}).items()})
        assert all(v > 0 for v in self.sat_turn_scale.values()), "progress_rank.sat_turn_scale must be > 0"
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
        # The gate's two EMAs per task (shares of the task's groups this step).
        self.gate_stuck_ema: Dict[str, Optional[float]] = {t: None for t in self.tasks}
        self.gate_sat_ema: Dict[str, Optional[float]] = {t: None for t in self.tasks}
        # The last apply()'s groups, one dict per group; see GROUP RECORDS above.
        self.last_group_records: List[dict] = []
        # The teacher term's first-order effect on the GRPO objective, per task, over
        # the last FIRST_ORDER_WINDOW updates (observe_update). Measured, not acted on.
        self.first_order: Dict[str, List[float]] = {}

    # --- persistence ------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"version": 4, "ema": dict(self.ema), "success_ema": dict(self.success_ema),
                "first_order": {t: list(v) for t, v in self.first_order.items()},
                "gate_stuck_ema": dict(self.gate_stuck_ema), "gate_sat_ema": dict(self.gate_sat_ema)}

    def load_state_dict(self, state: dict) -> None:
        state = state or {}
        for t, v in state.get("ema", {}).items():
            self.ema[str(t)] = None if v is None else float(v)
        # Version 1 carried E only; S then starts uninitialised and (a) waits for a live group.
        for t, v in state.get("success_ema", {}).items():
            self.success_ema[str(t)] = None if v is None else float(v)
        # Version 2 and earlier carried no window; it then refills from the next update.
        for t, v in (state.get("first_order", {}) or {}).items():
            self.first_order[str(t)] = [float(x) for x in v][-FIRST_ORDER_WINDOW:]
        # Version 3 and earlier carried no gate EMAs; they then start from the next step's shares.
        for key, store in (("gate_stuck_ema", self.gate_stuck_ema), ("gate_sat_ema", self.gate_sat_ema)):
            for t, v in (state.get(key, {}) or {}).items():
                store[str(t)] = None if v is None else float(v)

    def observe_update(self, metrics: dict) -> Dict[str, float]:
        """After the actor update: the teacher term's first-order effect, remembered.

        ``opd/<task>/grpo/first_order`` is <g_OPD, g_GRPO> / |g_GRPO|^2 on the logit
        gradients (cross_teacher_kl_weight.gradient_metrics): how much the teacher's
        term adds to, or takes from, the reward objective's own step. On the rho=0.1
        run it turned negative on WebShop at step ~126 in both runs, went to ~0 on
        ALFWorld and stayed positive on Search. Returned per task:
          first_order_pos_frac   positive share over the last FIRST_ORDER_WINDOW steps
                                 (the retirement rule read "fewer than 10 of 25")
          first_order_window     how many steps that share is over (it restarts empty
                                 on a checkpoint written before this existed)
        Absent when the teacher's term is off (coef 0): nothing to measure.
        """
        out: Dict[str, float] = {}
        for task in self.tasks:
            v = _finite(metrics.get(f"opd/{task}/grpo/first_order"))
            if v is None:
                continue
            w = self.first_order.setdefault(task, [])
            w.append(v)
            del w[:-FIRST_ORDER_WINDOW]
            out[f"opd/{task}/grpo/first_order_pos_frac"] = sum(1 for x in w if x > 0.0) / len(w)
            out[f"opd/{task}/grpo/first_order_window"] = float(len(w))
        return out

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
              coverage_rows=None, episode_lengths=None, turn_caps=None,
              alt_counts=None, task_score_rows=None, committed_rows=None,
              revisit_rows=None, done_walkset_rows=None, doc_len_rows=None,
              gamefile_rows=None) -> tuple:
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
        ``episode_lengths``  per row, its trajectory's length in turns (the rollout's
                             count, untouched by rows dropped from the batch)
        ``turn_caps``        ``{task: max turns}``, for the share of failures at the cap
        ``alt_counts``       ``{name: (k_rows, total_rows)}``, other progress counts to
                             compare with the ranked one on the same stuck groups
        ``task_score_rows``  per row, WebShop's continuous purchase score
        ``committed_rows``   per row, "has sent the task's terminal action" (Search's
                             <answer>, WebShop's buy; 1/0, NaN on ALFWorld). With it the
                             records carry ``committed`` per trajectory and the metrics
                             report how many of the rollouts a term pushed up, and of those
                             it pushed down, never committed.
        ``revisit_rows``     per row, actions the trajectory had already taken once, taken
                             again (a shadow ranking key for saturated groups; see
                             progress.RevisitCounter)
        ``done_walkset_rows`` per row, ALFWorld's walkthrough lines actually done, no won => K
        ``doc_len_rows``     per row, the length of its task's document for this game (the
                             scale sat_turn_scale_mode=document divides by)
        ``gamefile_rows``    per row, ALFWorld's game file; the records carry one per group
        None of these changes the advantage: they are read into metrics and records.

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
                               "invalid": 0, "d": None, "d_ok": coverage_rows is not None,
                               "length": None, "task_score": None, "committed": None,
                               "revisits": None, "done_walkset": None, "doc_len": None,
                               "gamefile": None}
            x["turns"] += 1
            if episode_lengths is not None:
                ln = _finite(episode_lengths[i])
                if ln is not None:
                    x["length"] = ln if x["length"] is None else max(x["length"], ln)
            if task_score_rows is not None:
                ts = _finite(task_score_rows[i])
                if ts is not None:
                    x["task_score"] = ts if x["task_score"] is None else max(x["task_score"], ts)
            if committed_rows is not None:
                cm = _finite(committed_rows[i])
                if cm is not None:
                    x["committed"] = bool(x["committed"]) or cm > 0.5
            if gamefile_rows is not None and x["gamefile"] is None:
                gf = gamefile_rows[i]
                if isinstance(gf, str) and gf:
                    x["gamefile"] = gf
            for col, key in ((revisit_rows, "revisits"), (done_walkset_rows, "done_walkset"),
                             (doc_len_rows, "doc_len")):
                if col is not None:
                    v = _finite(col[i])
                    if v is not None:
                        x[key] = v if x[key] is None else max(x[key], v)
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

        # Saturated groups (every rollout won by the environment's reward): the
        # turn-count ranking. Scored on every step -- the records and the sat_*
        # counts report it -- and added to the advantage only when sat_rho > 0.
        # THE GATE, decided from this step's group shares folded into the EMAs (the
        # replay that chose the rule used exactly this: EMA including the current step).
        gate_open: Dict[str, bool] = {}
        gate_q: Dict[str, Optional[float]] = {}
        shares: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for g in groups.values():
            task = str(g.get("task") or "")
            if task not in self.tasks:
                continue
            xs = [traj[t] for t in sorted({str(tuids[i]) for i in g["rows"] if real[i]}) if t in traj]
            if len(xs) < 2 or any(x["reward"] is None for x in xs):
                continue
            shares[task]["n"] += 1
            shares[task][group_status(g, xs)] += 1
        for task in self.tasks:
            n_t = shares[task]["n"]
            if n_t:
                st_share = shares[task]["stuck"] / n_t
                sa_share = shares[task]["saturated"] / n_t
                for store, val in ((self.gate_stuck_ema, st_share), (self.gate_sat_ema, sa_share)):
                    prev = store.get(task)
                    store[task] = val if prev is None else (1.0 - self.alpha) * prev + self.alpha * val
            es, ea = self.gate_stuck_ema.get(task), self.gate_sat_ema.get(task)
            condition = es is not None and ea is not None and ea > es
            gate_q[task] = (es / (es + ea)) if (es is not None and ea is not None and es + ea > 0) else None
            gate_open[task] = condition if self.sat_gate else True
        sat_tasks_now = [t for t in self.sat_tasks if gate_open.get(t, True)]
        sat_verdicts = score_saturated_groups(
            groups, tuids=tuids, stat_rows=stat, traj=traj, tasks=sat_tasks_now,
            min_spread=self.sat_min_spread, turn_scale=self.sat_turn_scale,
            cross_steps=self.cross_steps, scale_mode=self.sat_turn_scale_mode,
            placebo=self.sat_placebo)
        closed_tasks = [t for t in self.sat_tasks if t not in sat_tasks_now]
        if closed_tasks:
            # Still scored for the records, but the verdict says the gate held them.
            for uid, rec in score_saturated_groups(
                    groups, tuids=tuids, stat_rows=stat, traj=traj, tasks=closed_tasks,
                    min_spread=self.sat_min_spread, turn_scale=self.sat_turn_scale,
                    cross_steps=self.cross_steps, scale_mode=self.sat_turn_scale_mode,
                    placebo=self.sat_placebo).items():
                # What it WOULD have been: kept, so "how much firing the gate held back"
                # is a live number, not only a replay.
                rec["verdict_ungated"] = rec["verdict"]
                rec["verdict"], rec["scores"] = GATE_CLOSED, {}
                sat_verdicts[uid] = rec
        traj_sat: Dict[str, float] = {}
        for rec in sat_verdicts.values():
            traj_sat.update(rec["scores"])
        # The TRUE ranking's per-row scores (== the applied ones unless a placebo permuted them):
        # the dose-matched placebo sizes its coefficient on what the true ranking would inject.
        traj_sat_true: Dict[str, float] = {}
        for rec in sat_verdicts.values():
            if rec["scores"]:
                traj_sat_true.update(rec.get("scores_true", rec["scores"]))
        row_sat = np.zeros(n, dtype=float)
        for i in range(n):
            s = traj_sat.get(str(tuids[i]))
            if s is not None and (stat[i] or not real[i]):
                row_sat[i] = s
        row_sat_true = np.zeros(n, dtype=float)
        for i in range(n):
            s_t = traj_sat_true.get(str(tuids[i]))
            if s_t is not None and (stat[i] or not real[i]):
                row_sat_true[i] = s_t

        # Mixed groups: the failures ranked among themselves. Scored on every step
        # (the records and the mixed_* counts report it); added only when mixed_rho > 0.
        mixed_verdicts = score_mixed_groups(
            groups, tuids=tuids, stat_rows=stat, traj_prog=traj_prog, traj=traj,
            min_top_k=self.min_top_k, tasks=self.mixed_tasks, cross_steps=self.cross_steps)
        traj_mix: Dict[str, float] = {}
        traj_mix_uid: Dict[str, str] = {}
        for uid_m, rec in mixed_verdicts.items():
            traj_mix.update(rec["scores"])
            for t in rec["scores"]:
                traj_mix_uid[t] = uid_m
        row_mix = np.zeros(n, dtype=float)
        for i in range(n):
            s_m = traj_mix.get(str(tuids[i]))
            if s_m is not None and (stat[i] or not real[i]):
                row_mix[i] = s_m
        # the base advantage per token, which the sign-preserving scale is held against
        base_tok = signed / np.maximum(tokens, 1.0)
        delta_mix = np.zeros(n, dtype=float)
        mix_scale: Dict[str, float] = {}

        metrics: Dict[str, float] = {}
        coef = np.zeros(n, dtype=float)
        coef_sat = np.zeros(n, dtype=float)
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

            # The saturated-group term: its own share sat_rho of the same E, and
            # the same cap -- no winning token is pushed up more than kappa times
            # what a success typically gets in a live group.
            sat_fired = rows & (row_sat != 0.0)
            u_sat = float((np.abs(row_sat[sat_fired]) * tokens[sat_fired]).sum()) / task_tokens
            max_abs_sat = float(np.abs(row_sat[rows]).max()) if rows.any() else 0.0
            c_sat, sat_capped = 0.0, 0.0
            c_sat_uncapped, sat_cap = None, None
            if (self.sat_rho > 0.0 and task in self.sat_tasks and ema is not None and s_ema is not None
                    and u_sat > 0.0 and max_abs_sat > 0.0):
                c_sat_uncapped = self.sat_rho * ema / u_sat
                sat_cap = self.kappa * s_ema / max_abs_sat
                c_sat = c_sat_uncapped
                if c_sat > sat_cap:
                    c_sat, sat_capped = sat_cap, 1.0
                metrics[f"{p}/sat_c_uncapped"] = c_sat_uncapped
                metrics[f"{p}/sat_c_cap"] = sat_cap
                if self.sat_placebo == "shuffle_dose":
                    # DOSE-MATCHED placebo: inject exactly the token-mean mass the TRUE ranking
                    # would have injected on these same rollouts -- its own coefficient, cap
                    # included -- spread over the permuted scores. The placebo's peak per-token
                    # push is then whatever the permutation gives; it is reported, not capped.
                    tr_fired = rows & (row_sat_true != 0.0)
                    u_true = float((np.abs(row_sat_true[tr_fired]) * tokens[tr_fired]).sum()) / task_tokens
                    max_true = float(np.abs(row_sat_true[rows]).max()) if rows.any() else 0.0
                    if u_true > 0.0 and max_true > 0.0:
                        c_true = min(self.sat_rho * ema / u_true, self.kappa * s_ema / max_true)
                        target = c_true * u_true
                        c_sat = target / u_sat
                        sat_capped = float(self.sat_rho * ema / u_true > self.kappa * s_ema / max_true)
                        metrics[f"{p}/sat_placebo_true_c"] = c_true
                        metrics[f"{p}/sat_placebo_mass_ratio"] = (c_sat * u_sat) / target
                        metrics[f"{p}/sat_placebo_peak_over_cap"] = c_sat * max_abs_sat / (self.kappa * s_ema)
                    else:
                        c_sat, sat_capped = 0.0, 0.0
                elif self.sat_placebo == "bonus":
                    # BONUS-ONLY control: the TRUE arm's coefficient -- computed from the true
                    # scores, cap included -- applied to the uniform bonus alone, so each fired
                    # group gets exactly the bonus inside the true arm's term, and no more.
                    tr_fired = rows & (row_sat_true != 0.0)
                    u_true = float((np.abs(row_sat_true[tr_fired]) * tokens[tr_fired]).sum()) / task_tokens
                    max_true = float(np.abs(row_sat_true[rows]).max()) if rows.any() else 0.0
                    if u_true > 0.0 and max_true > 0.0:
                        c_true_uncapped = self.sat_rho * ema / u_true
                        c_true_cap = self.kappa * s_ema / max_true
                        c_sat = min(c_true_uncapped, c_true_cap)
                        sat_capped = float(c_true_uncapped > c_true_cap)
                        # report the coefficient that was applied, as the true arm would
                        c_sat_uncapped, sat_cap = c_true_uncapped, c_true_cap
                        metrics[f"{p}/sat_c_uncapped"] = c_sat_uncapped
                        metrics[f"{p}/sat_c_cap"] = sat_cap
                        metrics[f"{p}/sat_placebo_true_c"] = c_sat
                    else:
                        c_sat, sat_capped = 0.0, 0.0
            coef_sat[names == task] = c_sat
            # The gate, reported whether or not it is enforced: the condition, the two
            # EMAs and q = stuck / (stuck + saturated) among the task's dead groups.
            _es, _ea = self.gate_stuck_ema.get(task), self.gate_sat_ema.get(task)
            if _es is not None and _ea is not None:
                metrics[f"{p}/sat_gate_stuck_ema"] = _es
                metrics[f"{p}/sat_gate_sat_ema"] = _ea
                metrics[f"{p}/sat_gate_condition"] = float(_ea > _es)
                if gate_q.get(task) is not None:
                    metrics[f"{p}/sat_gate_q"] = gate_q[task]
            metrics[f"{p}/sat_gated_out"] = float(self.sat_gate and task in self.sat_tasks and not gate_open.get(task, True))

            # The mixed-group term: its own share mixed_rho of the same E and the same
            # cap, then shrunk per group until no failure's advantage crosses zero.
            mix_fired = rows & (row_mix != 0.0)
            u_mix = float((np.abs(row_mix[mix_fired]) * tokens[mix_fired]).sum()) / task_tokens
            max_abs_mix = float(np.abs(row_mix[rows]).max()) if rows.any() else 0.0
            c_mix, mix_capped = 0.0, 0.0
            c_mix_uncapped, mix_cap = None, None
            if (self.mixed_rho > 0.0 and task in self.mixed_tasks and ema is not None
                    and s_ema is not None and u_mix > 0.0 and max_abs_mix > 0.0):
                c_mix_uncapped = self.mixed_rho * ema / u_mix
                mix_cap = self.kappa * s_ema / max_abs_mix
                c_mix = c_mix_uncapped
                if c_mix > mix_cap:
                    c_mix, mix_capped = mix_cap, 1.0
                metrics[f"{p}/mixed_c_uncapped"] = c_mix_uncapped
                metrics[f"{p}/mixed_c_cap"] = mix_cap
            scaled = 0
            if c_mix > 0.0:
                for uid_m, rec in mixed_verdicts.items():
                    if rec["task"] != task or rec["verdict"] != FIRED:
                        continue
                    grows_m = [i for i in groups[uid_m]["rows"] if row_mix[i] != 0.0]
                    held = [i for i in grows_m if real[i]]
                    adds = c_mix * row_mix[held]
                    sc = sign_preserving_scale(adds, base_tok[held])
                    mix_scale[uid_m] = sc
                    scaled += int(sc < 1.0)
                    for i in grows_m:
                        delta_mix[i] = sc * c_mix * row_mix[i]
            if task in self.mixed_tasks:
                inj_mix = delta_mix * tokens
                metrics[f"{p}/mixed_c"] = c_mix
                metrics[f"{p}/mixed_capped"] = mix_capped
                metrics[f"{p}/mixed_score_mass"] = u_mix
                metrics[f"{p}/mixed_fired_token_share"] = float(tokens[mix_fired].sum()) / task_tokens
                metrics[f"{p}/mixed_inject_up"] = float(inj_mix[rows & (delta_mix > 0)].sum()) / task_tokens
                metrics[f"{p}/mixed_inject_down"] = float(-inj_mix[rows & (delta_mix < 0)].sum()) / task_tokens
                metrics[f"{p}/mixed_injected_mean_abs_adv"] = float(np.abs(inj_mix[rows]).sum()) / task_tokens
                metrics[f"{p}/mixed_share_of_ema"] = (
                    metrics[f"{p}/mixed_injected_mean_abs_adv"] / ema) if ema else 0.0
                task_scales = [v for u_, v in mix_scale.items() if mixed_verdicts[u_]["task"] == task]
                if task_scales:
                    # How often the sign guard had to shrink a group, and by how much.
                    metrics[f"{p}/mixed_sign_scaled_groups"] = float(scaled)
                    metrics[f"{p}/mixed_scale_mean"] = float(np.mean(task_scales))
            per_task[task] = {"c": c, "capped": bool(capped), "c_uncapped": c_uncapped, "c_cap": cap,
                              "ema": ema, "success_push_ema": s_ema, "task_tokens": task_tokens,
                              "sat_c": c_sat, "sat_capped": bool(sat_capped),
                              "sat_c_uncapped": c_sat_uncapped, "sat_c_cap": sat_cap,
                              "mixed_c": c_mix, "mixed_capped": bool(mix_capped)}
            if task in self.sat_tasks:
                inj_sat = row_sat * c_sat * tokens
                metrics[f"{p}/sat_c"] = c_sat
                metrics[f"{p}/sat_capped"] = sat_capped
                metrics[f"{p}/sat_score_mass"] = u_sat
                metrics[f"{p}/sat_injected_mean_abs_adv"] = c_sat * u_sat
                metrics[f"{p}/sat_share_of_ema"] = (c_sat * u_sat / ema) if ema else 0.0
                metrics[f"{p}/sat_top_push"] = c_sat * max_abs_sat
                if s_ema:
                    metrics[f"{p}/sat_top_push_over_success"] = c_sat * max_abs_sat / s_ema
                metrics[f"{p}/sat_fired_token_share"] = float(tokens[sat_fired].sum()) / task_tokens
                metrics[f"{p}/sat_inject_up"] = float(inj_sat[rows & (row_sat > 0)].sum()) / task_tokens
                metrics[f"{p}/sat_inject_down"] = float(-inj_sat[rows & (row_sat < 0)].sum()) / task_tokens
                if invalid is not None:
                    # A longer winner often carries invalid turns: the push-down on
                    # those rows is the part aimed at the loop behaviour itself.
                    metrics[f"{p}/sat_inject_down_invalid"] = (
                        float(-inj_sat[rows & invalid & (row_sat < 0)].sum()) / task_tokens)
                    metrics[f"{p}/sat_inject_up_invalid"] = (
                        float(inj_sat[rows & invalid & (row_sat > 0)].sum()) / task_tokens)

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

        # What the rollouts looked like (turns, format, the cap, other counts): measured only.
        alt_prog = {str(name): trajectory_progress(tuids, kr, tr, range(n))
                    for name, (kr, tr) in (alt_counts or {}).items()}
        metrics.update(trajectory_metrics(
            groups, traj, tuids=tuids, real=real,
            tasks=list(dict.fromkeys(self.tasks + self.sat_tasks)), traj_prog=traj_prog,
            turn_caps=turn_caps, alt_prog=alt_prog, have_invalid=invalid is not None))

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
            status = group_status(g, xs)
            stuck = status == "stuck"
            if stuck:
                counts[task]["groups_stuck"] += 1
                if uid in spread:
                    counts[task]["stuck_mixed" if spread[uid] > SCORE_SPREAD_EPS else "stuck_uniform"] += 1
            if status == "saturated":
                counts[task]["groups_saturated"] += 1
            if status == "live":
                counts[task]["groups_live"] += 1
            verdict = verdicts.get(uid, {}).get("verdict")
            sat_verdict = sat_verdicts.get(uid, {}).get("verdict")

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
                # The rollout's own length (rows can be dropped from the batch), WebShop's
                # continuous purchase score, and the other progress counts, per trajectory.
                "length": [x["length"] for x in xs],
                "task_score": [x["task_score"] for x in xs],
                "committed": [x["committed"] for x in xs],
                "revisits": [x["revisits"] for x in xs],
                "done_walkset": [x["done_walkset"] for x in xs],
                "doc_len": [x["doc_len"] for x in xs],
                "gamefile": next((x["gamefile"] for x in xs if x.get("gamefile")), None),
                **{f"k_{name}": [prog.get(t, (0.0, 0.0))[0] for t in trajs] for name, prog in alt_prog.items()},
                **{f"K_{name}": [prog.get(t, (0.0, 0.0))[1] for t in trajs] for name, prog in alt_prog.items()},
                "score": [traj_score.get(t, 0.0) for t in trajs],
                "score_spread": spread.get(uid),
                "progpo_gate": gate.get(uid),
                "_base_mass": float(abs_adv[grows].sum()) if grows else 0.0,
                "base_abs_adv_max": (float(np.max(abs_adv[grows] / np.maximum(tokens[grows], 1.0)))
                                     if grows else 0.0),
                "injected_abs_mass": float(np.sum(np.abs(row_score[grows] * coef[grows]) * tokens[grows])),
                "c": info.get("c"), "capped": info.get("capped"), "c_uncapped": info.get("c_uncapped"),
                "c_cap": info.get("c_cap"), "ema": info.get("ema"),
                "success_push_ema": info.get("success_push_ema"), "task_tokens": info.get("task_tokens"),
                # The saturated-group term (sat_rho): verdict, per-trajectory score and mass.
                "sat_verdict": sat_verdict,
                "sat_verdict_ungated": sat_verdicts.get(uid, {}).get("verdict_ungated", sat_verdict),
                "sat_scale": sat_verdicts.get(uid, {}).get("scale"),
                "sat_score": [traj_sat.get(t, 0.0) for t in trajs],
                # Under a sat_placebo, sat_score is the score that was applied (permuted, or
                # the true one times the group's coin under "sign") and sat_score_true the
                # turn ranking it replaced; equal otherwise.
                "sat_placebo": self.sat_placebo,
                "sat_placebo_sign": sat_verdicts.get(uid, {}).get("placebo_sign"),
                "sat_score_true": [sat_verdicts.get(uid, {}).get("scores_true", {}).get(t, traj_sat.get(t, 0.0))
                                   for t in trajs],
                "sat_injected_abs_mass": float(np.sum(np.abs(row_sat[grows] * coef_sat[grows]) * tokens[grows])),
                "sat_c": info.get("sat_c"), "sat_capped": info.get("sat_capped"),
                "sat_c_uncapped": info.get("sat_c_uncapped"), "sat_c_cap": info.get("sat_c_cap"),
                # The mixed-group term: verdict, per-trajectory score (failures only),
                # the sign guard's scale and the mass it added.
                "mixed_verdict": mixed_verdicts.get(uid, {}).get("verdict"),
                "mixed_score": [traj_mix.get(t, 0.0) for t in trajs],
                "mixed_scale": mix_scale.get(uid),
                "mixed_injected_abs_mass": float(np.sum(np.abs(delta_mix[grows]) * tokens[grows])),
                "mixed_c": info.get("mixed_c"), "mixed_capped": info.get("mixed_capped"),
            })
        # WHO GETS PUSHED, BY WHETHER THEY COMMITTED (tasks with a terminal action:
        # Search's answer, WebShop's buy). Among the rollouts of fired stuck groups, and
        # the failures of fired mixed groups: the share that never committed, on the
        # side the term pushed up and on the side it pushed down. A pushed-up side that
        # leans to "never committed" is the option-loop / search-to-the-cap failure
        # ([[mixed-term-rewards-webshop-option-loops]]) being rewarded.
        unans: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
        for rec in records:
            flags = rec.get("committed") or []
            if not any(a is not None for a in flags):
                continue
            for kind, verdict_key, score_key in (("stuck", "verdict", "score"),
                                                 ("mixed", "mixed_verdict", "mixed_score")):
                if rec.get(verdict_key) != FIRED:
                    continue
                for a, sc in zip(flags, rec.get(score_key) or []):
                    if a is None or abs(float(sc)) <= 1e-12:
                        continue
                    side = "up" if sc > 0 else "down"
                    c = unans[(rec["task"], kind, side)]
                    c[0] += 1
                    c[1] += int(not a)
        for (task, kind, side), (n_side, n_unans) in unans.items():
            metrics[f"progress_rank/{task}/{kind}_{side}_n"] = float(n_side)
            metrics[f"progress_rank/{task}/{kind}_{side}_uncommitted"] = n_unans / n_side
        # THE ALL-FAIL GROUPS' OWN GRADIENT. Their environment reward is flat, but the
        # -0.1 invalid-action penalty enters before the group statistic, and the turn-
        # level std normalisation blows one penalised turn up to |A| 8-20 per token
        # (85% of ALFWorld's all-fail groups at steps 151-300, task mean |A| 0.64). So
        # "no task signal", never "no gradient": what each stuck group already carries.
        # WHERE THE BASE GRADIENT SITS. Per task, the share of the base |A| mass (before
        # (a), summed over tokens) that lies in stuck groups and in saturated groups --
        # groups whose environment reward is flat, so whatever mass they carry comes
        # from the format penalty through the turn-level std normalisation.
        mass_by: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for rec in records:
            mass_by[rec["task"]][rec["status"]] += float(rec.get("_base_mass") or 0.0)
            mass_by[rec["task"]]["_all"] += float(rec.get("_base_mass") or 0.0)
        for task, mb in mass_by.items():
            if mb["_all"] > 0:
                metrics[f"progress_rank/{task}/stuck_base_mass_share"] = mb["stuck"] / mb["_all"]
                metrics[f"progress_rank/{task}/saturated_base_mass_share"] = mb["saturated"] / mb["_all"]
        for rec in records:
            rec.pop("_base_mass", None)
        stuck_base: Dict[str, List[float]] = defaultdict(list)
        sat_agree: Dict[str, List[float]] = defaultdict(list)
        sat_corr: Dict[str, List[float]] = defaultdict(list)
        for rec in records:
            if rec["status"] == "stuck":
                stuck_base[rec["task"]].append(float(rec.get("base_abs_adv_max") or 0.0))
            # In a FIRED saturated group: would ranking by revisits agree with turns?
            # Over pairs of winners with different turn counts, the share ordered the
            # same way by revisits (ties count as disagreement).
            if rec.get("sat_verdict") == FIRED:
                rv, tn = rec.get("revisits") or [], rec.get("turns") or []
                pairs = [(a, b) for a in range(len(tn)) for b in range(a + 1, len(tn))
                         if tn[a] != tn[b] and rv[a] is not None and rv[b] is not None]
                if pairs:
                    same = sum(1 for a, b in pairs if (tn[a] < tn[b]) == (rv[a] < rv[b]) and rv[a] != rv[b])
                    sat_agree[rec["task"]].append(same / len(pairs))
                # The applied score against the turns: -1 for the true ranking (an affine
                # function of turns), near 0 on average under the shuffle and sign placebos. The
                # check that the placebo arm is a placebo, live in every step's metrics.
                sc = np.asarray(rec.get("sat_score") or [], dtype=float)
                tt = np.asarray(tn, dtype=float)
                if len(sc) >= 3 and sc.std() > 0 and tt.std() > 0:
                    sat_corr[rec["task"]].append(float(np.corrcoef(sc, tt)[0, 1]))
        for task, vals in stuck_base.items():
            metrics[f"progress_rank/{task}/stuck_base_abs_adv_mean"] = float(np.mean(vals))
            metrics[f"progress_rank/{task}/stuck_base_abs_adv_nonzero_share"] = float(np.mean([v > 1e-6 for v in vals]))
        for task, vals in sat_agree.items():
            metrics[f"progress_rank/{task}/sat_revisit_agreement"] = float(np.mean(vals))
        for task, vals in sat_corr.items():
            metrics[f"progress_rank/{task}/sat_score_turn_corr"] = float(np.mean(vals))
        for rec in verdicts.values():
            counts[rec["task"]][f"stuck_{rec['verdict']}"] += 1
        for rec in mixed_verdicts.values():
            counts[rec["task"]][f"mixed_{rec['verdict']}"] += 1
        sat_spreads: Dict[str, List[float]] = defaultdict(list)
        sat_flips: Dict[str, List[float]] = defaultdict(list)
        for rec in sat_verdicts.values():
            counts[rec["task"]][f"sat_{rec['verdict']}"] += 1
            counts[rec["task"]]["sat_fired_ungated"] += int(rec.get("verdict_ungated", rec["verdict"]) == FIRED)
            sat_spreads[rec["task"]].append(rec["spread"])
            if rec["verdict"] == FIRED and "placebo_sign" in rec:
                sat_flips[rec["task"]].append(float(rec["placebo_sign"] < 0.0))
        for task, fl in sat_flips.items():
            # sat_placebo=sign: the share of the fired groups whose ranking was reversed (~0.5).
            metrics[f"progress_rank/{task}/sat_placebo_flip_share"] = float(np.mean(fl))
        for task, cs in list(counts.items()):
            if cs.get("sat_fired_ungated", 0):
                # Of the saturated groups that would fire, the share the gate let through.
                metrics[f"progress_rank/{task}/sat_gate_kept_share"] = cs.get("sat_fired", 0) / cs["sat_fired_ungated"]
        for task, sp in sat_spreads.items():
            # How far apart the winners of one game are, in turns: the signal's size.
            metrics[f"progress_rank/{task}/sat_turn_spread_mean"] = float(np.mean(sp))
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

        delta = torch.as_tensor(row_score * coef + row_sat * coef_sat + delta_mix,
                                dtype=advantages.dtype, device=advantages.device)
        if not bool((delta != 0).any()):
            # Nothing to add: hand back the very same tensor, so rho = sat_rho = mixed_rho = 0
            # (or a step where nothing fired) is bit-identical to control by construction.
            return advantages, metrics
        new = advantages + delta.unsqueeze(-1) * mask.to(advantages.dtype)
        return new, metrics
