# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Which eight of a group's ten rollouts are trained, and which two are thrown away.

THE RULE, in full. A group's class is read off its ORDINARY rollouts (the plain
slots plus the reserve) -- at ``group_n=10`` the same eight a control group
trains, so the verdict is control's verdict and not a different one taken on
seven:

    live       -> the eight ordinary rollouts. Identical to control.
    stuck      -> seven plain + the DOCUMENT rollout, if that rollout solved the
                  game. If it did not, the reserve stays and the group is
                  control's again.
    saturated  -> seven plain + the FOREIGN-PROMPT rollout, if that rollout
                  failed. If it somehow did not, the reserve stays.

So the arm differs from control in exactly one place: a group whose eight
ordinary rollouts all agree -- which GRPO gives an advantage of exactly zero --
trades its reserve for a rollout that disagrees. Everywhere else the two runs
train the same rows.

WHY THE DROP HAPPENS HERE, BEFORE ``adjust_batch``. The two unused rollouts are
removed from the batch outright rather than masked, because everything after this
point counts rows: ``adjust_batch`` pads to a divisor, ``attach_task_loss_weights``
divides each task's share by its token count, ``_balance_batch`` splits by tokens
and ``update_policy`` cuts mini-batches by row count -- so a batch carrying ten
trajectories per group would take ~25% more optimizer steps per training step
than control, each with a quarter of its rows inert. Dropping first makes the row
count, and therefore the number of updates, control's again. It also saves the
old_log_prob and teacher forwards on rows nothing trains; the prefetched rows are
keyed by ``(traj_uid, turn_step)``, so the ones that go simply stop being asked
for.

WHAT THE KEPT SPECIAL ROW IS, DOWNSTREAM. One of the eight, with its own return
in the group's mean and std, and marked in ``oci_injected`` so the actor trains it
through the shaped off-policy term on the PLAIN prompt (LUFFY's f(rho); see
verl/trainer/ppo/oci_shaping.py) instead of the ordinary clipped ratio, which
here would be a policy gradient on a conditional distribution -- given a
walkthrough, or given another task's prompt -- that never occurs at test time.

A SPECIAL ROLLOUT THAT CANNOT BE RE-SCORED IS NOT USED. ``rho``'s numerator is
the same response tokens under the prompt with the privileged text removed, and
the removal has to be exact: the rollout loop records it as one verified
replacement and marks the row unstrippable when it is not (a left-truncated
prompt, or a replacement wider than the column). A group whose special rollout
has any such row keeps its reserve instead, and the count is reported -- stripping
approximately would make rho a comparison between two prompts that differ in more
than the conditioning, which is the one thing it must not be.
"""

from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np

from agent_system.environments.oci_layout import (
    ROLE_DOC,
    ROLE_FOREIGN,
    ROLE_NONE,
    ROLE_DOC_B,
    ROLE_PLAIN,
    ROLE_RESERVE,
    has_foreign_slot,
    role_for_slot,
    used_per_group,
)
from verl.trainer.ppo.metric_utils import get_task_names
from verl.trainer.ppo.oci_saturated import classify_returns

__all__ = ["check_config", "select_rollouts", "apply_selection",
           "ROLE_KEY", "SLOT_KEY", "INJECTED_KEY"]

ROLE_KEY = "oci_role"
SLOT_KEY = "oci_slot"
INJECTED_KEY = "oci_injected"

_EPS = 1e-9


def _col(batch, key, *, required=True):
    """A row-aligned int/float numpy view of a batch column, or None."""
    col = batch.batch.get(key, None)
    if col is None:
        assert not required, (
            f"algorithm.oci_slots.enable=True but the batch carries no {key!r} column. "
            "The layout columns are emitted by the rollout loop, which emits them "
            "when algorithm.oci_slots reaches the environment manager that builds "
            "the observations -- a missing column means the rollout was not built "
            "with the switch on, which is a launch error.")
        return None
    return col.reshape(-1).detach().cpu().numpy()


def _n_traj(tuids, rows) -> int:
    return len({str(tuids[i]) for i in rows})


def check_config(config) -> None:
    """Fail before the first rollout when the arm cannot be what it says it is."""
    cfg = config.algorithm.get("oci_slots", None)
    assert cfg is not None and bool(cfg.get("enable", False))

    for other in ("oci_sat", "oci_floor"):
        blk = config.algorithm.get(other, None)
        assert not (blk is not None and bool(blk.get("enable", False))), (
            f"algorithm.{other}.enable and algorithm.oci_slots.enable are both on. "
            "They are three ways of moving the same degenerate group's baseline -- a "
            "real injected failure, a virtual sample, and a pair of extra slots -- "
            "and running two gives one group two interventions. Pick one.")

    group_n = int(config.env.rollout.n)
    assert group_n >= 4, (
        f"env.rollout.n={group_n}: the layout needs two ordinary rollouts to read a "
        "verdict from and two more for the document and foreign slots.")

    tasks = list(cfg.get("tasks", ["alfworld"]) or ["alfworld"])
    assert tasks in (["alfworld"], ["search"]), (
        f"algorithm.oci_slots.tasks={tasks}: alfworld shows the game's own walkthrough "
        "(replays 30/30) and search shows the answer under the rule below. A WebShop "
        "document has to be built and replay-verified first.")
    _search_doc = str(cfg.get("search_doc", "answer_only") or "answer_only")
    assert not (tasks == ["search"] and _search_doc not in ("answer_rule", "progress_only")), (
        f"algorithm.oci_slots.tasks=[search] with search_doc={_search_doc}: Search's reward "
        "reads only the final <answer> string and never checks that a search happened, so "
        "that document rescues a group by being copied. Use answer_rule (the answer under "
        "the rule that a returned result must carry it first) or progress_only (no answer, "
        "only the verdict).")
    _search_doc_b = str(cfg.get("search_doc_b", "none") or "none")
    assert _search_doc_b == "none" or tasks == ["search"], (
        f"algorithm.oci_slots.search_doc_b={_search_doc_b} with tasks={tasks}: the second "
        "document row is a search-only measurement.")

    # The special rows are trained by the policy gradient alone, through the
    # shaped term, and that term is built only on the per-task-weighted branch of
    # the loss. Both facts are checked here rather than discovered as a silently
    # unshaped run.
    assert float(config.actor_rollout_ref.actor.get("pg_loss_coef", 1.0)) != 0.0, (
        "actor.pg_loss_coef=0 with algorithm.oci_slots.enable=True: the special "
        "rows would be generated, judged, kept -- and train nothing.")
    assert bool(config.actor_rollout_ref.actor.get("normalize_loss_by_task", False)), (
        "algorithm.oci_slots.enable=True needs algorithm.opd.normalize_loss_by_task=True: "
        "the shaped off-policy term is built on the per-task-weighted branch of the "
        "policy loss, and without it the special rows would train an ordinary "
        "clipped ratio taken on the privileged prompt.")
    _loss = str(cfg.get("special_loss", "shaped") or "shaped")
    assert _loss in ("shaped", "ppo", "gated"), (
        f"algorithm.oci_slots.special_loss={_loss!r}; expected 'shaped' (LUFFY's f(rho)), "
        "'ppo' (the ordinary clip on the plain prompt) or 'gated' (that clip with the "
        "NEGATIVE rows' gate reversed, so the foreign row trains only below 1-eps).")
    _filter = config.algorithm.get("filter_groups", None)
    assert not (_filter is not None and bool(_filter.get("enable", False))), (
        "algorithm.filter_groups.enable rejects and regenerates whole groups, which "
        "re-draws the layout's slots; the two are not compatible.")


def select_rollouts(batch, *, tasks: Sequence[str], group_n: int, second_doc: bool = False):
    """``(keep, injected, metrics)``: row masks over the batch as it left the rollout.

    ``keep`` is the eight trajectories per group that train; ``injected`` is the
    special one among them, if any, for the actor's shaped term.
    """
    role = _col(batch, ROLE_KEY).astype(int)
    slot = _col(batch, SLOT_KEY).astype(int)
    n = int(role.shape[0])
    # Search has no foreign slot, so its groups are one row shorter in specials
    # and one longer in ordinary rollouts. The arm covers one task at a time, so
    # the layout is uniform across the batch's on-task rows.
    foreign = all(has_foreign_slot(t) for t in tasks)
    assert foreign or not any(has_foreign_slot(t) for t in tasks), (
        f"algorithm.oci_slots.tasks={list(tasks)} mixes tasks that do and do not "
        "spend a slot on another task's prompt; their groups would train different "
        "numbers of trajectories")
    plain_n = used_per_group(group_n, foreign=foreign, second_doc=second_doc)

    uids = batch.non_tensor_batch.get("uid", None)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    assert uids is not None and tuids is not None, (
        "the layout groups by prompt and by trajectory; this batch carries no "
        "uid/traj_uid")
    task_names = get_task_names(batch)
    assert task_names is not None, (
        "the layout is per task (a group's class is read against its own task's "
        "reward scale) and this batch carries no task names")

    # THE RETURN THE ENVIRONMENT PAID, not the reward manager's score: this runs
    # before the reward manager has written token_level_rewards. The manager's
    # score IS this number (normalize_by_length=False on both entry points), so
    # the verdict here and the verdict the advantage is built from agree.
    rets = batch.non_tensor_batch.get("episode_rewards", None)
    assert rets is not None, "no episode_rewards column; the rollout writes one per row"
    ret = np.asarray([float(r) for r in rets], dtype=float)

    plan_len = _col(batch, "oci_plan_len")
    trunc = _col(batch, "oci_plan_truncated", required=False)
    strippable = plan_len > 0
    if trunc is not None:
        strippable = strippable & (trunc == 0)

    wanted = set(str(t) for t in tasks)
    on_task = np.array([str(t) in wanted for t in task_names], dtype=bool)

    # THE MARK AND THE POSITION MUST AGREE. The role is written by the env manager,
    # which knows its own slots; the slot index is the row's position in the
    # generation batch, written where that position still means something. If they
    # disagree, the batch is not the one the arm describes -- e.g. a manager built
    # with a different group size than env.rollout.n -- and every verdict below
    # would be taken on the wrong rows.
    expected = np.array([role_for_slot(int(s), group_n, foreign=foreign, second_doc=second_doc)
                         for s in slot], dtype=int)
    bad = np.nonzero(on_task & (role != expected))[0]
    assert bad.size == 0, (
        f"{bad.size} of {int(on_task.sum())} rows on {sorted(wanted)} carry a role "
        f"their slot does not predict (first: slot {int(slot[bad[0]])} marked "
        f"{int(role[bad[0]])}, expected {int(expected[bad[0]])} at group_n={group_n})")
    bad_off = np.nonzero((~on_task) & (role != ROLE_NONE))[0]
    assert bad_off.size == 0, (
        f"{bad_off.size} rows off {sorted(wanted)} carry a layout role; the env "
        "manager marks only the tasks the arm covers")

    # Off-task groups have no roles, so their ordinary rollouts are the first
    # plain_n slots -- an outcome-independent choice, made before any return is
    # read.
    judged = np.where(on_task, np.isin(role, (ROLE_PLAIN, ROLE_RESERVE)), slot < plain_n)
    groups = classify_returns(ret, uids, tuids, task_names, judged_rows=judged)

    keep = np.zeros(n, dtype=bool)
    injected = np.zeros(n, dtype=bool)
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    used_turns: Dict[str, list] = {"document": [], "foreign": []}
    returns_seen: Dict[str, list] = {"document": [], "foreign": []}
    kept_per_group = []

    for g in groups.values():
        rows = np.asarray(g["rows"], dtype=int)
        task = str(g.get("task") or "")
        counts[task][f"groups_{g['status']}"] += 1
        if not on_task[rows[0]]:
            sel = rows[slot[rows] < plain_n]
            keep[sel] = True
            kept_per_group.append(_n_traj(tuids, sel))
            continue

        by_role = {r: rows[role[rows] == r]
                   for r in (ROLE_PLAIN, ROLE_RESERVE, ROLE_DOC, ROLE_FOREIGN, ROLE_DOC_B)}
        layout = {r: _n_traj(tuids, idx) for r, idx in by_role.items()}
        assert layout == {ROLE_PLAIN: plain_n - 1, ROLE_RESERVE: 1, ROLE_DOC: 1,
                          ROLE_FOREIGN: 1 if foreign else 0,
                          ROLE_DOC_B: 1 if second_doc else 0}, (
            f"group carries {layout} trajectories by role, not "
            f"{{plain: {plain_n - 1}, reserve: 1, document: 1, "
            f"foreign: {1 if foreign else 0}, document_b: {1 if second_doc else 0}}}; "
            "the generation was not laid out as the arm describes")

        level = float(g["ret"])
        doc_ret = float(ret[by_role[ROLE_DOC]].max())
        returns_seen["document"].append(doc_ret)
        foreign_ret = float(ret[by_role[ROLE_FOREIGN]].max()) if foreign else None
        if foreign:
            returns_seen["foreign"].append(foreign_ret)

        use = ROLE_RESERVE
        if g["status"] == "stuck" and doc_ret > level + _EPS:
            counts[task]["doc_solved"] += 1
            if bool(strippable[by_role[ROLE_DOC]].all()):
                use = ROLE_DOC
            else:
                counts[task]["doc_unstrippable"] += 1
        elif foreign and g["status"] == "saturated" and foreign_ret < level - _EPS:
            counts[task]["foreign_failed"] += 1
            if bool(strippable[by_role[ROLE_FOREIGN]].all()):
                use = ROLE_FOREIGN
            else:
                counts[task]["foreign_unstrippable"] += 1

        sel = np.concatenate([by_role[ROLE_PLAIN], by_role[use]])
        keep[sel] = True
        if use != ROLE_RESERVE:
            injected[by_role[use]] = True
            name = "document" if use == ROLE_DOC else "foreign"
            counts[task][f"{name}_used"] += 1
            used_turns[name].append(int(by_role[use].size))
        kept_per_group.append(_n_traj(tuids, sel))

    assert kept_per_group and min(kept_per_group) == max(kept_per_group) == plain_n, (
        f"groups train {sorted(set(kept_per_group))} trajectories, not {plain_n}; "
        "the point of generating two extra rollouts is that every group trains the "
        "same number control does")

    metrics = {
        "oci_slots/rows_generated": int(n),
        "oci_slots/rows_trained": int(keep.sum()),
        "oci_slots/rows_dropped": int((~keep).sum()),
        "oci_slots/injected_rows": int(injected.sum()),
        "oci_slots/trained_per_group": int(plain_n),
    }
    for task, c in counts.items():
        sfx = f"/{task}" if task else ""
        for key, value in c.items():
            metrics[f"oci_slots/{key}{sfx}"] = int(value)
        if c.get("groups_stuck"):
            metrics[f"oci_slots/doc_rescue_rate{sfx}"] = (
                c.get("doc_solved", 0) / c["groups_stuck"])
        if c.get("groups_saturated"):
            metrics[f"oci_slots/foreign_fail_rate{sfx}"] = (
                c.get("foreign_failed", 0) / c["groups_saturated"])
    for name, turns in used_turns.items():
        if turns:
            metrics[f"oci_slots/{name}_turns_mean"] = float(np.mean(turns))
    for name, vals in returns_seen.items():
        if vals:
            # The sanity check on the slots themselves, over ALL groups rather than
            # the class each serves: a foreign slot that stops failing, or a
            # document slot that stops solving, is the arm quietly turning into
            # control.
            metrics[f"oci_slots/{name}_return_mean"] = float(np.mean(vals))
    return keep, injected, metrics


def apply_selection(batch, keep: np.ndarray, injected: np.ndarray):
    """Drop the untrained rollouts and mark the kept special rows."""
    import torch

    idx = np.nonzero(np.asarray(keep, dtype=bool))[0]
    out = batch.select_idxs(idx)
    out.batch[INJECTED_KEY] = torch.as_tensor(
        np.asarray(injected, dtype=bool)[idx].astype(np.int64))
    return out
