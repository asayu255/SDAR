"""How much of each task's update is reward-driven and how much is teacher-driven.

THE QUESTION. A group whose eight trajectories all earn the same return has a
GRPO advantage of zero on every row -- but only if their FORMAT also agrees: the
-0.1 invalid-action penalty is a per-turn reward, so one untagged row in an
otherwise uniform group moves every advantage in it (the +19.9 fingerprint). And
the teacher KL is applied to every row whatever its group did. So a degenerate
group does not stop a task from learning; it shifts that task's update from the
reward toward the teacher. This module measures that shift, per task and per
group class, on the loss the optimizer actually sees.

THE CLASSES. live; stuck and saturated, each split by whether any advantage in
the group is non-zero (``mixed``: the format channel is carrying it) or none is
(``uniform``: the policy gradient is exactly zero there). Pooling the two would
make a stalled task's policy gradient look healthy.

THE NUMBERS, per (task, class):
  tokens         response tokens of the real rows (padding rows carry weight 0)
  pg_mass        sum over rows of  w * sum_t |pg_loss_t| * pg_loss_coef, read off
                 the actor's own per-token policy loss (at ratio 1, |A|)
  kl_mass        sum over rows of  w * kl_row_coef * sum_t KL_t * teacher_kl_coef,
                 the student-indexed top-k KL from the same forward
  pg_mass_driver the same PG sum recomputed from the batch's advantages -- the
                 self-check that the actor's rows were mapped back correctly

``w`` is the per-task row weight (task_loss_weights.py), so every mass is on the
scale the loss is summed on, and a ratio between two of them is a ratio of
gradients' loss contributions, not of token counts.
"""
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

CLASSES = ("live", "stuck_uniform", "stuck_mixed", "saturated_uniform", "saturated_mixed")
FIELDS = ("groups", "rows", "tokens", "pg_mass", "kl_mass", "pg_mass_driver")


def group_class(status: str, advantages_abs_max: float, eps: float = 1e-12) -> str:
    """The class of one group, from its verdict and its largest |advantage|."""
    if status == "live":
        return "live"
    if status not in ("stuck", "saturated"):
        raise ValueError(f"unknown group status {status!r}")
    return f"{status}_{'mixed' if advantages_abs_max > eps else 'uniform'}"


def aggregate_term_mass(groups: Dict[str, Dict], records: Iterable[dict], *,
                        row_mask: np.ndarray, advantages: np.ndarray,
                        task_weights: np.ndarray, pg_loss_coef: float) -> Dict:
    """Sum the per-row terms into (task, class) cells.

    ``groups``      classify_groups' output: uid -> {rows, status, task}
    ``records``     the actor's per-row dicts (row, tokens, pg_abs, kl, w,
                    kl_row_coef, teacher_kl_coef)
    ``row_mask``    (rows, resp) response/loss mask of the batch as the actor saw it
    ``advantages``  (rows, resp) the batch's advantages
    ``task_weights`` (rows,) the per-task row weights; 0 marks adjust_batch padding
    """
    by_row = {}
    for rec in records:
        by_row[int(rec["row"])] = rec
    mask = np.asarray(row_mask, dtype=float)
    adv = np.asarray(advantages, dtype=float)
    w = np.asarray(task_weights, dtype=float)

    cells: Dict = defaultdict(lambda: {f: 0.0 for f in FIELDS})
    missing = 0
    padding = 0
    for g in groups.values():
        rows = [i for i in g["rows"] if w[i] > 0]
        padding += len(g["rows"]) - len(rows)
        if not rows:
            continue
        amax = float(np.max(np.abs(adv[rows] * mask[rows]))) if rows else 0.0
        cls = group_class(g["status"], amax)
        cell = cells[(str(g.get("task") or ""), cls)]
        cell["groups"] += 1
        for i in rows:
            rec = by_row.get(i)
            if rec is None:
                missing += 1
                continue
            cell["rows"] += 1
            cell["tokens"] += float(rec["tokens"])
            cell["pg_mass"] += float(rec["w"]) * float(rec["pg_abs"]) * float(pg_loss_coef)
            cell["kl_mass"] += (float(rec["w"]) * float(rec.get("kl_row_coef", 1.0))
                                * float(rec["kl"]) * float(rec["teacher_kl_coef"]))
            cell["pg_mass_driver"] += (w[i] * float(np.sum(np.abs(adv[i]) * mask[i]))
                                       * float(pg_loss_coef))
    return {"cells": {f"{t}/{c}": v for (t, c), v in cells.items()},
            "rows_missing_from_actor": missing, "padding_rows_skipped": padding}


def add_batches(acc: Dict, batch_out: Dict) -> Dict:
    """Accumulate one batch's cells into a running total (in place)."""
    cells = acc.setdefault("cells", {})
    for key, v in batch_out["cells"].items():
        c = cells.setdefault(key, {f: 0.0 for f in FIELDS})
        for f in FIELDS:
            c[f] += v[f]
    for k in ("rows_missing_from_actor", "padding_rows_skipped"):
        acc[k] = acc.get(k, 0) + batch_out.get(k, 0)
    acc["batches"] = acc.get("batches", 0) + 1
    return acc


def summarise(acc: Dict) -> Dict:
    """Per task: each class's share of tokens, of PG mass and of KL mass, and PG/KL."""
    cells = acc.get("cells", {})
    tasks = sorted({k.split("/", 1)[0] for k in cells})
    out = {"batches": acc.get("batches", 0),
           "rows_missing_from_actor": acc.get("rows_missing_from_actor", 0),
           "padding_rows_skipped": acc.get("padding_rows_skipped", 0), "tasks": {}}
    pg_all = sum(c["pg_mass"] for c in cells.values())
    drv_all = sum(c["pg_mass_driver"] for c in cells.values())
    out["pg_mass_actor_vs_driver"] = (pg_all / drv_all) if drv_all else None
    for t in tasks:
        tc = {c: cells.get(f"{t}/{c}", {f: 0.0 for f in FIELDS}) for c in CLASSES}
        tot = {f: sum(v[f] for v in tc.values()) for f in FIELDS}
        task = {"totals": dict(tot),
                "pg_over_kl": (tot["pg_mass"] / tot["kl_mass"]) if tot["kl_mass"] else None,
                "classes": {}}
        for c, v in tc.items():
            task["classes"][c] = {
                "groups": v["groups"], "rows": v["rows"],
                "token_share": (v["tokens"] / tot["tokens"]) if tot["tokens"] else None,
                "pg_share": (v["pg_mass"] / tot["pg_mass"]) if tot["pg_mass"] else None,
                "kl_share": (v["kl_mass"] / tot["kl_mass"]) if tot["kl_mass"] else None,
                "pg_over_kl": (v["pg_mass"] / v["kl_mass"]) if v["kl_mass"] else None,
            }
        out["tasks"][t] = task
    return out


def format_report(summary: Dict) -> List[str]:
    """The table the probe prints: one block per task."""
    lines = [f"batches {summary['batches']}  rows missing from actor {summary['rows_missing_from_actor']}"
             f"  padding skipped {summary['padding_rows_skipped']}"
             f"  pg actor/driver {summary['pg_mass_actor_vs_driver']}"]
    for t, task in summary["tasks"].items():
        lines.append(f"\n[{t}] PG/KL overall: {task['pg_over_kl']}")
        lines.append(f"  {'class':<19}{'groups':>7}{'tokens':>9}{'PG':>9}{'KL':>9}{'PG/KL':>9}")
        for c, v in task["classes"].items():
            def pct(x):
                return "-" if x is None else f"{100 * x:.1f}%"
            ratio = "-" if v["pg_over_kl"] is None else f"{v['pg_over_kl']:.3g}"
            lines.append(f"  {c:<19}{int(v['groups']):>7}{pct(v['token_share']):>9}"
                         f"{pct(v['pg_share']):>9}{pct(v['kl_share']):>9}{ratio:>9}")
    return lines
