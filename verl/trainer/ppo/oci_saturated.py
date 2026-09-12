"""Outcome-conditioned injection into SATURATED groups.

A saturated group -- every rollout of a prompt at the same, maximal return --
produces zero advantage, and that is GRPO behaving correctly: the group baseline
equals the return, so the right update is zero. Replacing one row with a failure
does not recover a lost signal, it moves the baseline and manufactures one. With
seven successes and one failure the manufactured advantages are

    A_success = +1/sqrt(7) = +0.378,   A_failure = -sqrt(7) = -2.646

for every reward scale: the magnitudes follow from the 7:1 split alone and say
nothing about the student, the prompt, or how bad the injected failure was.
Whether that manufactured signal improves held-out accuracy is the open
question this module exists to make testable, not a property it assumes.

WHAT THIS MODULE DOES AND DOES NOT DO
  * It marks which rows are injected and which group they belong to.
  * It does NOT generate anything: the injected rows must already be in the
    batch, produced by whatever generator the arm selected.
  * It does NOT decide the gradient policy. Arm B zeroes the injected rows'
    advantage after the fact; arm A leaves it. Both are exposed because only A
    beating B shows that suppressing the failure is worth anything beyond the
    +1/sqrt(7) re-reinforcement the seven successes get in either arm.

THE JUDGEMENT IS PER STEP AND IS NOT CARRIED FORWARD. A prompt that is saturated
in this draw need not be in the next one; measured on ALFWorld, re-drawing the
same dataset row changes its label 83% of the time, because the row is a
placeholder and the environment picks the game. So no label is cached, and
nothing here reads a prompt id.
"""

from typing import Dict, Optional

import numpy as np

__all__ = ["classify_groups", "select_saturated_injections", "zero_injected_advantage"]

INJECTED_KEY = "oci_injected"


def _row_returns(batch) -> np.ndarray:
    """Per-row return. ``token_level_rewards`` is zero except where the reward was
    placed, so the row sum IS that row's return."""
    return batch.batch["token_level_rewards"].sum(-1).detach().float().cpu().numpy()


def classify_groups(batch, *, judged_rows: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """``{uid: {"rows": [...], "status": "live"|"stuck"|"saturated", "ret": float}}``.

    Grouping is over TRAJECTORIES, not rows: a GRPO advantage is the
    trajectory's return minus its group mean, so a group is degenerate when the
    trajectory returns agree, not when the rows do. 2-7% of trajectories carry
    the return on some of their rows and not others, which at n=8 makes a
    genuinely degenerate group read as live about a quarter of the time if rows
    are counted instead.

    ``judged_rows`` restricts the VERDICT to a subset (the seven plain rollouts,
    when an eighth is reserved for injection) while still reporting every row of
    the group.
    """
    uids = batch.non_tensor_batch.get("uid", None)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    if uids is None or tuids is None:
        return {}
    ret = _row_returns(batch)
    n = len(ret)
    judged = np.ones(n, dtype=bool) if judged_rows is None else np.asarray(judged_rows, dtype=bool)

    by_group: Dict[str, Dict] = {}
    for i in range(n):
        g = by_group.setdefault(str(uids[i]), {"rows": [], "traj": {}})
        g["rows"].append(i)
        if judged[i]:
            # a trajectory's return is the max over its rows; see the docstring
            k = str(tuids[i])
            g["traj"][k] = max(g["traj"].get(k, float("-inf")), float(ret[i]))

    # the reference level that splits degenerate groups into stuck and saturated
    live_vals = [v for g in by_group.values()
                 if len(set(g["traj"].values())) > 1 for v in g["traj"].values()]
    ref = float(np.mean(live_vals)) if live_vals else 0.0

    out = {}
    for uid, g in by_group.items():
        vals = list(g["traj"].values())
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        status = "live" if hi != lo else ("stuck" if lo < ref else "saturated")
        out[uid] = {"rows": g["rows"], "status": status, "ret": lo,
                    "n_judged": len(vals)}
    return out


def select_saturated_injections(batch, groups: Dict[str, Dict], *,
                                candidate_rows: np.ndarray) -> np.ndarray:
    """Boolean mask over rows: which candidate rows to keep as injections.

    A candidate is kept only when its group was judged saturated AND the
    candidate actually failed -- an injected row that also succeeds leaves the
    group uniform and changes nothing, which is the reason mixing a strong
    off-policy trace into an already-solved group is a no-op.
    """
    ret = _row_returns(batch)
    cand = np.asarray(candidate_rows, dtype=bool)
    keep = np.zeros(len(ret), dtype=bool)
    for g in groups.values():
        if g["status"] != "saturated":
            continue
        for i in g["rows"]:
            if cand[i] and ret[i] < g["ret"] - 1e-9:
                keep[i] = True
    return keep


def zero_injected_advantage(batch, injected: np.ndarray) -> int:
    """Arm B: keep the injected rows in the group mean, take their gradient away.

    The baseline has already moved by the time this runs, so the seven successes
    keep their +1/sqrt(7); what goes is only the -sqrt(7) on the injected row.
    Returns the number of rows zeroed.
    """
    adv = batch.batch.get("advantages", None)
    if adv is None:
        return 0
    import torch

    m = torch.as_tensor(np.asarray(injected, dtype=bool), device=adv.device)
    if m.ndim == 1 and adv.ndim == 2:
        m = m.unsqueeze(-1)
    adv.masked_fill_(m, 0.0)
    return int(np.asarray(injected, dtype=bool).sum())


def injection_metrics(groups: Dict[str, Dict], injected: np.ndarray, task: str = "") -> Dict:
    """What the arm actually did this step, for the log."""
    n = {"live": 0, "stuck": 0, "saturated": 0}
    for g in groups.values():
        n[g["status"]] += 1
    tot = max(sum(n.values()), 1)
    suffix = f"/{task}" if task else ""
    return {
        f"oci/groups_live{suffix}": n["live"],
        f"oci/groups_stuck{suffix}": n["stuck"],
        f"oci/groups_saturated{suffix}": n["saturated"],
        f"oci/live_frac{suffix}": n["live"] / tot,
        f"oci/injected_rows{suffix}": int(np.asarray(injected, dtype=bool).sum()),
    }
