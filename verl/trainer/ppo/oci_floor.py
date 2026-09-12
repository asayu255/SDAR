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
"""Arm B', the virtual floor sample: a saturated group's statistic gains one
sample at the task's failure return, and no rollout is generated for it.

WHY THE GENERATOR IS GONE. Arm B removed the injected row's own gradient, which
left exactly one channel by which the injection could reach the weights: the
group's mean and std. A sample that only has to move a mean and a std does not
have to exist. Adding it as a term in the statistic reproduces B's update and
removes everything that was built to make a real rollout fail -- the corrupted
plan, the candidate column, the prompt-span bookkeeping, the reachability
measurement -- along with the reason the arm was stuck: five prompt designs
reached cand_fail_rate 0.222 against a pre-registered bar of 0.5, so the real
injection fired on about one saturated group in five and on none at all in 7 of
13 measured batches. The virtual sample fails by definition.

WHAT ELSE FOLLOWS FROM IT. The real injection had to consume one of the eight
rollout slots, so the arm trained the policy and the distillation term on seven
trajectories where control trained them on eight -- the confound recorded in
the design document as the one that could not be removed without a ninth slot.
B' takes no slot: all eight rollouts are real, judged and trained, and OPD sees
the same eight rows as control. The arm's only difference from control is a term
in the GRPO statistic of groups that control would have given an advantage of
exactly zero.

THIS IS THE MIRROR OF NGRPO'S DEVICE ON THE STUCK SIDE (design document 5.2),
which the document already cites, and no novelty is claimed for it.
"""
from typing import Dict, Iterable, Optional

import numpy as np

from verl.trainer.ppo.metric_utils import get_task_names


def select_floor_groups(batch, groups: Dict[str, Dict], *,
                        tasks: Optional[Iterable[str]] = None,
                        padding_mask=None) -> np.ndarray:
    """Rows of every saturated group on an ``oci_floor`` task.

    A ROW MASK FOR A PER-GROUP FACT, for the same reason the exclusion mask is a
    column: ``_balance_batch`` reorders rows before the advantage is computed, and
    a uid travels with its row. Every row of a floored group carries the mark, and
    ``compute_grpo_outcome_advantage`` reads the group's uid off any of them.

    Padding rows are marked like any other row of their group -- they are copies
    that carry their original's uid, they do not enter the statistic, and they
    receive the advantage their original's statistic produced.
    """
    n = len(batch)
    out = np.zeros(n, dtype=bool)
    if not groups:
        return out
    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        return out
    want = set(tasks or ["alfworld"])
    task_names = get_task_names(batch)
    rows_by_uid: Dict[str, list] = {}
    for i in range(n):
        rows_by_uid.setdefault(str(uids[i]), []).append(i)
    for uid, g in groups.items():
        if g.get("status") != "saturated":
            continue
        rows = rows_by_uid.get(str(uid), [])
        if not rows:
            continue
        if task_names is not None and str(task_names[rows[0]]) not in want:
            continue
        out[rows] = True
    return out


def floor_metrics(batch, groups: Dict[str, Dict], floored: np.ndarray, *,
                  tasks: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Group counts by class, per task, and how much of the batch the floor took.

    COUNTED PER TASK, NOT POOLED AND THEN LABELLED. The first version walked
    every group ``classify_groups`` returned and filed the totals under the
    floored task's name, so a three-task batch of 15 groups each reported
    "oci_floor/groups_live/alfworld: 14" -- the live groups of all 45 -- beside a
    correct ``groups/live/alfworld: 3`` from compute_group_metrics. The selection
    was right and only the label was wrong, which is the worse failure: a number
    that is read for 300 steps and is off by the other two tasks.

    The reason to keep it per task at all is the one the reference level already
    has: alfworld pays about 10 for a solved episode and search pays 1, so a
    pooled count is set by alfworld and says nothing about the others.
    """
    want = set(tasks or ["alfworld"])
    task_names = get_task_names(batch)
    by_task: Dict[str, Dict[str, int]] = {t: {"live": 0, "stuck": 0, "saturated": 0}
                                          for t in want}
    for g in groups.values():
        rows = g.get("rows") or []
        if not rows:
            continue
        t = str(task_names[rows[0]]) if task_names is not None else "alfworld"
        if t not in by_task:
            continue
        st = g.get("status")
        if st in by_task[t]:
            by_task[t][st] += 1
    out: Dict[str, float] = {
        "oci_floor/groups_floored": _floored_groups(batch, floored),
        "oci_floor/rows_floored": int(np.asarray(floored, dtype=bool).sum()),
    }
    for t, n_by in by_task.items():
        total = max(sum(n_by.values()), 1)
        out[f"oci_floor/groups_live/{t}"] = n_by["live"]
        out[f"oci_floor/groups_stuck/{t}"] = n_by["stuck"]
        out[f"oci_floor/groups_saturated/{t}"] = n_by["saturated"]
        out[f"oci_floor/live_frac/{t}"] = n_by["live"] / total
    return out


def _floored_groups(batch, floored: np.ndarray) -> int:
    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        return 0
    return len({str(uids[i]) for i in np.flatnonzero(np.asarray(floored, dtype=bool))})


def realized_advantage(batch, floored: np.ndarray) -> Dict[str, float]:
    """What the floor actually bought, read off the advantage column.

    The arm's whole claim is that these rows go from an advantage of exactly zero
    to a fixed positive number. That is checkable after the fact and nothing else
    in the run checks it, so it is a metric rather than a comment: a mean near
    zero means the floor did not reach the statistic.
    """
    floored = np.asarray(floored, dtype=bool)
    if not floored.any() or "advantages" not in batch.batch:
        return {}
    import torch

    adv = batch.batch["advantages"]
    mask = batch.batch.get("response_mask", None)
    rows = torch.as_tensor(floored, device=adv.device)
    if mask is not None:
        m = mask[rows].bool()
        vals = adv[rows][m]
    else:
        vals = adv[rows].reshape(-1)
    if vals.numel() == 0:
        return {}
    return {"oci_floor/adv_mean_on_floored": float(vals.mean()),
            "oci_floor/adv_min_on_floored": float(vals.min()),
            "oci_floor/adv_max_on_floored": float(vals.max())}
