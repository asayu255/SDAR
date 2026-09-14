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

from verl.trainer.ppo.metric_utils import get_task_names

__all__ = ["classify_groups", "classify_returns", "select_saturated_injections",
           "zero_injected_advantage"]

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
    return classify_returns(_row_returns(batch), uids, tuids, get_task_names(batch),
                            judged_rows=judged_rows)


def classify_returns(returns, uids, tuids, task_names=None, *,
                     judged_rows: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """The same verdict, from plain columns rather than from a scored batch.

    ONE STEP EARLIER IN THE PIPELINE. The ten-slot layout (see
    verl/trainer/ppo/oci_slots.py) picks each group's eighth rollout straight out
    of the rollout loop, before the reward manager has written
    ``token_level_rewards``; the only return available there is the environment's
    own. That is the number the manager goes on to copy -- both entry points
    build it with ``normalize_by_length=False`` -- so the verdict taken here and
    the verdict the advantage is built from cannot disagree.
    """
    ret = np.asarray([float(r) for r in returns], dtype=float)
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

    # THE SPLIT IS RELATIVE TO WHAT THE BATCH ACHIEVED, and the batch may not
    # contain a live group to read that from. Preference order:
    #   1. the mean over live groups -- the band where learning is happening;
    #   2. the midpoint of the degenerate returns, when there are no live groups
    #      but the degenerate ones disagree with each other;
    #   3. the batch's own minimum, when every group scored alike -- then the
    #      question is only whether that common score is the floor, and a batch
    #      of all-zero groups must read as stuck rather than saturated.
    # PER TASK, BECAUSE THE REWARD SCALES ARE NOT COMPARABLE. An earlier version
    # pooled every group in the batch into one reference level. On this mixture
    # alfworld pays about 10 for a solved episode and search pays 1, so the
    # pooled live mean is set by alfworld -- measured at 5.83 on the control
    # checkpoint -- and every search group, whose best possible return is 1,
    # read as "stuck" against it. The first probe reported 19 stuck groups where
    # the per-task split gives 10. Worse than the metric: the same verdict is
    # what select_saturated_injections reads, so a group's class was being
    # decided against another task's scale.
    group_task = {}
    for uid, g in by_group.items():
        group_task[uid] = str(task_names[g["rows"][0]]) if task_names is not None else ""

    ref_by_task: Dict[str, float] = {}
    for task in set(group_task.values()):
        vals_by_group = {
            u: list(g["traj"].values())
            for u, g in by_group.items()
            if g["traj"] and group_task[u] == task
        }
        live_vals = [v for vs in vals_by_group.values() if len(set(vs)) > 1 for v in vs]
        deg = [vs[0] for vs in vals_by_group.values() if len(set(vs)) == 1]
        if live_vals:
            ref_by_task[task] = float(np.mean(live_vals))
        elif deg and min(deg) != max(deg):
            ref_by_task[task] = (float(min(deg)) + float(max(deg))) / 2.0
        else:
            # everything alike: saturated iff that score is above the floor. The
            # floor is 0.0 -- no environment here pays for doing nothing -- so an
            # all-zero batch is stuck and an all-solved batch is saturated.
            ref_by_task[task] = 0.0 if (deg and max(deg) > 0.0) else float("inf")

    out = {}
    for uid, g in by_group.items():
        ref = ref_by_task[group_task[uid]]
        vals = list(g["traj"].values())
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        status = "live" if hi != lo else ("stuck" if lo < ref else "saturated")
        out[uid] = {"rows": g["rows"], "status": status, "ret": lo,
                    "n_judged": len(vals), "task": group_task[uid], "ref": ref}
    return out


def select_saturated_injections(batch, groups: Dict[str, Dict], *,
                                candidate_rows: np.ndarray) -> np.ndarray:
    """Boolean mask over rows: which candidate rows to keep as injections.

    A candidate is kept only when its group was judged saturated AND the
    candidate actually failed -- an injected row that also succeeds leaves the
    group uniform and changes nothing, which is the reason mixing a strong
    off-policy trace into an already-solved group is a no-op.

    PER TRAJECTORY, LIKE THE CLASSIFICATION. An earlier version compared each
    ROW's return to the group's level. ``classify_groups`` decides a trajectory's
    return by the max over its rows, so the two disagreed exactly where a
    trajectory's reward is not on every row: a candidate that succeeded could
    still have individual rows reading below the level, and those rows alone were
    kept as "injections" while the rest of the same trajectory was dropped --
    half a trajectory in the group's statistic and half out of it. A trajectory
    is kept or dropped whole.
    """
    ret = _row_returns(batch)
    cand = np.asarray(candidate_rows, dtype=bool)
    keep = np.zeros(len(ret), dtype=bool)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    for g in groups.values():
        if g["status"] != "saturated":
            continue
        rows = [i for i in g["rows"] if cand[i]]
        if not rows:
            continue
        if tuids is None:
            # No trajectory ids: every row is its own trajectory, which is the
            # single-turn case and where the row rule was already correct.
            for i in rows:
                if ret[i] < g["ret"] - 1e-9:
                    keep[i] = True
            continue
        by_traj: Dict[str, list] = {}
        for i in rows:
            by_traj.setdefault(str(tuids[i]), []).append(i)
        for idxs in by_traj.values():
            if max(ret[i] for i in idxs) < g["ret"] - 1e-9:
                for i in idxs:
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
    """What the arm actually did this step, for the log.

    SPLIT BY THE GROUP'S OWN TASK, not labelled with one. The first version
    counted every group in the batch and filed the total under ``task``, so a
    three-task batch reported "11 live alfworld groups" when alfworld has only
    fifteen groups in total and six of them were live. ``task`` is now only a
    fallback for groups whose task is unknown.
    """
    per_task: Dict[str, Dict[str, int]] = {}
    for g in groups.values():
        t = g.get("task") or task
        n = per_task.setdefault(t, {"live": 0, "stuck": 0, "saturated": 0})
        n[g["status"]] += 1
    inj = np.asarray(injected, dtype=bool)
    out = {}
    for t, n in per_task.items():
        suffix = f"/{t}" if t else ""
        tot = max(sum(n.values()), 1)
        out[f"oci/groups_live{suffix}"] = n["live"]
        out[f"oci/groups_stuck{suffix}"] = n["stuck"]
        out[f"oci/groups_saturated{suffix}"] = n["saturated"]
        out[f"oci/live_frac{suffix}"] = n["live"] / tot
    # Injected rows are counted once: the switch fires on one task and the mask
    # is over rows, so a per-task split of it would be the same number twice.
    out["oci/injected_rows"] = int(inj.sum())
    return out


def token_mass_by_class(batch, groups: Dict[str, Dict], *, multi_turn: bool = True,
                        exclude_rows=None) -> Dict:
    """Tokens, not groups, split by group class.

    THE NUMBER THAT SIZES THE MECHANISM. The gradient counts tokens; a count of
    groups does not. A saturated group finishes early and a stuck one runs to the
    turn cap, so the two counts disagree by roughly 4x and only the token count
    bounds what any injection can buy.

    NO NUMBERS QUOTED HERE. An earlier version of this docstring gave 30% of
    groups live against 75.5% of tokens on the control checkpoint. Those came
    from a payload with no ``group_unit`` label, written by the sibling report
    that counted spread over rows rather than trajectories, so the live side is
    over-stated by an unknown amount. The quantity is re-derived by running this
    function; it is not restated from memory.

    ``exclude_rows`` drops rows from the token count without moving them between
    classes -- injected candidates the group did not want are neither live nor
    stuck nor saturated tokens, and folding them into their group's class
    inflates exactly the class the mechanism is trying to size.

    Reported per class: groups, tokens, and tokens per group.
    """
    import numpy as np

    resp = batch.batch.get("responses", None)
    am = batch.batch.get("attention_mask", None)
    if resp is None or am is None:
        return {}
    n = resp.shape[1]
    key = "loss_mask" if (multi_turn and "loss_mask" in batch.batch.keys()) else "attention_mask"
    mask = batch.batch[key][:, -n:].bool()
    per_row = mask.sum(-1).detach().cpu().numpy()
    skip = (np.asarray(exclude_rows, dtype=bool) if exclude_rows is not None
            else np.zeros(len(per_row), dtype=bool))

    agg = {c: [0, 0] for c in ("live", "stuck", "saturated")}
    dropped = 0
    for g in groups.values():
        a = agg[g["status"]]
        a[0] += 1
        a[1] += int(sum(per_row[i] for i in g["rows"] if not skip[i]))
        dropped += int(sum(per_row[i] for i in g["rows"] if skip[i]))
    tot_g = max(sum(v[0] for v in agg.values()), 1)
    tot_t = max(sum(v[1] for v in agg.values()), 1)
    out = {}
    for c, (ng, nt) in agg.items():
        out[f"oci/tokmass/{c}/groups"] = ng
        out[f"oci/tokmass/{c}/tokens"] = nt
        out[f"oci/tokmass/{c}/group_share"] = ng / tot_g
        out[f"oci/tokmass/{c}/token_share"] = nt / tot_t
        out[f"oci/tokmass/{c}/tokens_per_group"] = (nt / ng) if ng else 0.0
    # what the mechanism can buy at most: the share of tokens that carry nothing
    out["oci/tokmass/dead_token_share"] = (
        agg["stuck"][1] + agg["saturated"][1]) / tot_t
    out["oci/tokmass/saturated_share_of_dead"] = (
        agg["saturated"][1] / max(agg["stuck"][1] + agg["saturated"][1], 1))
    out["oci/tokmass/excluded_tokens"] = dropped
    return out
