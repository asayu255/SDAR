"""Does more progress mean more reward? One row per trajectory, then the curve.

WHY. (a) pushes a stuck group's further rollouts up on the premise that k -- the
steps of the task's correct sequence a rollout carried out -- tracks how close it
got to succeeding. That premise had been checked on ALFWorld with a proxy
definition (no "the environment executed it" condition) and on Search, and never
on WebShop. This module turns a probe batch into the table that checks it with
the definitions the run actually uses, and summarises it per task:

  * the success rate in bins of k/K -- whether reward rises WITH k, not only
    whether winners and losers differ
  * the success rate at each absolute k, since K differs between goals
  * within live groups, the AUC of k/K against success (ties count half): the
    same prompt, so a game's difficulty cannot masquerade as progress
  * mean k/K of winners, of losers in live groups, and of stuck-group rollouts
  * how (a) would treat the stuck groups (fired / no difference / below the
    minimum / no progress), through the same verdict code the trainer uses

A trajectory's reward is the environment's (ALFWorld 0/10, WebShop 0/10, Search
0/1), not the row return with the format penalty in it, so "won" means reward > 0.
"""
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from verl.trainer.ppo.progress_rank import (
    DEFAULT_MIN_TOP_K, FIRED, NO_DIFFERENCE, NO_PROGRESS, TOP_BELOW_MIN,
    failed_groups, score_stuck_groups, trajectory_progress)

BINS = ("k=0", "0<k/K<=1/3", "1/3<k/K<=2/3", "2/3<k/K<1", "k=K")


def k_bin(k: float, K: float) -> Optional[str]:
    if K <= 0:
        return None
    if k <= 0:
        return BINS[0]
    if k >= K:
        return BINS[4]
    r = k / K
    if r <= 1.0 / 3.0:
        return BINS[1]
    if r <= 2.0 / 3.0:
        return BINS[2]
    return BINS[3]


def trajectory_table(*, uids, tuids, task_names, episode_rewards, k_rows, total_rows,
                     real_rows, gamefiles=None, variants: Optional[Dict[str, tuple]] = None,
                     coverage_rows=None, valid_rows=None) -> List[dict]:
    """One record per trajectory over the real rows (padding copies excluded).

    ``variants`` maps a name to another ``(k_rows, total_rows)`` pair counted on the
    same rollouts -- ALFWorld's milestone count beside the walkthrough one -- and
    each lands on the record as ``k_<name>`` / ``K_<name>``.
    ``coverage_rows`` (ProGPO's D per row) adds ``coverage_d``, the largest over the
    trajectory's rows, or None when a row lacks it; ``valid_rows`` (is_action_valid)
    adds ``invalid_turns``.
    """
    rows = [i for i in range(len(tuids)) if real_rows[i]]
    prog = trajectory_progress(tuids, k_rows, total_rows, rows)
    extra = {name: trajectory_progress(tuids, kr, tr, rows) for name, (kr, tr) in (variants or {}).items()}
    recs: Dict[str, dict] = {}
    for i in rows:
        t = str(tuids[i])
        rec = recs.setdefault(t, {"traj": t, "uid": str(uids[i]), "task": str(task_names[i]),
                                  "reward": float("-inf"), "turns": 0})
        rec["turns"] += 1
        try:
            r = float(episode_rewards[i])
        except (TypeError, ValueError):
            r = float("nan")
        if np.isfinite(r):
            rec["reward"] = max(rec["reward"], r)
        if gamefiles is not None and gamefiles[i]:
            rec["gamefile"] = str(gamefiles[i])
        if coverage_rows is not None:
            try:
                d = float(coverage_rows[i])
            except (TypeError, ValueError):
                d = float("nan")
            if not rec.get("_d_missing"):
                if np.isfinite(d):
                    rec["coverage_d"] = max(rec.get("coverage_d") or 0.0, d)
                else:
                    rec["_d_missing"] = True
                    rec["coverage_d"] = None
        if valid_rows is not None:
            try:
                bad = float(valid_rows[i]) == 0.0
            except (TypeError, ValueError):
                bad = False
            rec["invalid_turns"] = rec.get("invalid_turns", 0) + int(bad)
    out = []
    for t, rec in recs.items():
        k, K = prog.get(t, (0.0, 0.0))
        rec["k"], rec["K"] = float(k), float(K)
        for name, p in extra.items():
            vk, vK = p.get(t, (0.0, 0.0))
            rec[f"k_{name}"], rec[f"K_{name}"] = float(vk), float(vK)
        rec["won"] = bool(np.isfinite(rec["reward"]) and rec["reward"] > 0.0)
        rec.pop("_d_missing", None)
        out.append(rec)
    return out


def _group_status(trajs: List[dict]) -> str:
    wins = sum(1 for x in trajs if x["won"])
    if wins == 0:
        return "stuck"
    return "saturated" if wins == len(trajs) else "live"


def summarise_progress(trajs: Iterable[dict], *, min_top_k: Optional[Dict[str, float]] = None) -> Dict:
    """Per task: the curve, the live-group AUC, and (a)'s verdicts on stuck groups.

    Groups are keyed by (batch, uid) when a record carries its batch, so two
    batches that reuse a uid never merge.
    """
    mtk = dict(DEFAULT_MIN_TOP_K)
    mtk.update(min_top_k or {})
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for x in trajs:
        by_task[x["task"]].append(x)
    out = {}
    for task, xs in sorted(by_task.items()):
        groups: Dict[tuple, List[dict]] = defaultdict(list)
        for x in xs:
            groups[(x.get("batch"), x["uid"])].append(x)
        status = {g: _group_status(v) for g, v in groups.items()}

        bins = {b: [0, 0] for b in BINS}          # [n, wins]
        by_k: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        no_seq = 0
        for x in xs:
            b = k_bin(x["k"], x["K"])
            if b is None:
                no_seq += 1
                continue
            bins[b][0] += 1
            bins[b][1] += int(x["won"])
            by_k[int(x["k"])][0] += 1
            by_k[int(x["k"])][1] += int(x["won"])

        # within live groups: P(a random winner's k/K exceeds a random loser's)
        num = den = 0.0
        win_r, live_lose_r, stuck_r = [], [], []
        for g, v in groups.items():
            ok = [x for x in v if x["K"] > 0]
            if status[g] == "live":
                W = [x["k"] / x["K"] for x in ok if x["won"]]
                L = [x["k"] / x["K"] for x in ok if not x["won"]]
                live_lose_r += L
                for a in W:
                    for c in L:
                        num += 1.0 if a > c else (0.5 if a == c else 0.0)
                        den += 1.0
            if status[g] == "stuck":
                stuck_r += [x["k"] / x["K"] for x in ok]
            win_r += [x["k"] / x["K"] for x in ok if x["won"]]

        # (a)'s verdicts, through the trainer's own code: one pseudo-row per
        # trajectory, weighted 1 (the verdict does not depend on the weights).
        flat = [x for v in groups.values() for x in v]
        keys = [f"{x.get('batch')}|{x['uid']}" for x in flat]
        tuids = [f"{x.get('batch')}|{x['traj']}" for x in flat]
        fg = failed_groups(keys, tuids, [task] * len(flat), [x["reward"] for x in flat], range(len(flat)))
        prog = {t: (x["k"], x["K"]) for t, x in zip(tuids, flat)}
        verdicts = score_stuck_groups(fg, tuids=tuids, stat_rows=np.ones(len(flat), dtype=bool),
                                      traj_prog=prog, min_top_k=mtk, tasks=[task], cross_steps=False)
        vc = defaultdict(int)
        for rec in verdicts.values():
            vc[rec["verdict"]] += 1

        out[task] = {
            "trajectories": len(xs), "groups": len(groups),
            "groups_live": sum(1 for s in status.values() if s == "live"),
            "groups_stuck": sum(1 for s in status.values() if s == "stuck"),
            "groups_saturated": sum(1 for s in status.values() if s == "saturated"),
            "no_sequence": no_seq,
            "win_rate": sum(int(x["won"]) for x in xs) / len(xs) if xs else None,
            "bins": {b: {"n": n, "wins": w, "win_rate": (w / n) if n else None}
                     for b, (n, w) in bins.items()},
            "by_k": {int(k): {"n": n, "wins": w, "win_rate": (w / n) if n else None}
                     for k, (n, w) in sorted(by_k.items())},
            "live_auc": (num / den) if den else None,
            "live_pairs": int(den),
            "mean_k_frac": {"winners": float(np.mean(win_r)) if win_r else None,
                            "live_losers": float(np.mean(live_lose_r)) if live_lose_r else None,
                            "stuck": float(np.mean(stuck_r)) if stuck_r else None},
            "stuck_verdicts": {v: int(vc.get(v, 0)) for v in (FIRED, NO_DIFFERENCE, TOP_BELOW_MIN, NO_PROGRESS)},
        }
    return out


def variant_records(trajs: Iterable[dict], name: str) -> List[dict]:
    """The same trajectories with ``k_<name>`` / ``K_<name>`` as their k and K; only
    those that carry that count (K > 0 under it)."""
    out = []
    for x in trajs:
        K = x.get(f"K_{name}", 0.0) or 0.0
        if K > 0:
            out.append(dict(x, k=x.get(f"k_{name}", 0.0), K=K))
    return out


def format_progress_report(summary: Dict, label: str = "") -> List[str]:
    """The table the probe prints; the task name is on every line."""
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}%"

    lines = []
    for task, s in summary.items():
        task = f"{task}{label}"
        lines.append(f"{task:<9} trajectories {s['trajectories']}  groups {s['groups']} "
                     f"(live {s['groups_live']}, stuck {s['groups_stuck']}, saturated {s['groups_saturated']})"
                     f"  win {pct(s['win_rate'])}  no-sequence {s['no_sequence']}")
        for b in BINS:
            v = s["bins"][b]
            lines.append(f"{task:<9}   {b:<14} n={v['n']:>4}  win {pct(v['win_rate']):>7}")
        ks = "  ".join(f"k={k}:{pct(v['win_rate'])}(n={v['n']})" for k, v in s["by_k"].items())
        lines.append(f"{task:<9}   by k: {ks}")
        auc = "-" if s["live_auc"] is None else f"{s['live_auc']:.3f}"
        m = s["mean_k_frac"]
        f3 = (lambda x: "-" if x is None else f"{x:.2f}")
        lines.append(f"{task:<9}   live-group AUC {auc} ({s['live_pairs']} pairs)   mean k/K: winners "
                     f"{f3(m['winners'])}, live losers {f3(m['live_losers'])}, stuck {f3(m['stuck'])}")
        v = s["stuck_verdicts"]
        lines.append(f"{task:<9}   stuck groups for (a): fired {v[FIRED]}, no difference {v[NO_DIFFERENCE]}, "
                     f"below min {v[TOP_BELOW_MIN]}, no progress {v[NO_PROGRESS]}")
    return lines
