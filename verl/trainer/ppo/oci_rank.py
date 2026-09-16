# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rank a degenerate group's own rollouts instead of injecting another one.

THE PROBLEM. A group whose eight rollouts all agree gets an advantage of exactly
zero from GRPO, and three arms that filled that silence by GENERATING a rollout
that disagrees -- a walkthrough-conditioned one for a stuck group, a
wrong-goal one for a saturated group -- all cost accuracy. The reason was the
same each time: the injected row's tokens were produced under a prompt that does
not exist at test time, so training them (up OR down) moved the policy on a
conditional distribution nobody queries, and entropy rose. See
verl/trainer/ppo/oci_slots.py for the layout those arms used.

WHAT THIS DOES INSTEAD. Nothing is generated. The eight rollouts are the
student's own, on the ordinary prompt, and the only new quantity is how likely
each of them is under a conditioning the student did not have:

    privileged  the SAME weights with this instance's document in front of the
                observation ("which of the eight would the self that knows the
                answer have written?")
    teacher     the task's teacher checkpoint on the plain prompt

Both are a length-normalised mean log-prob of the student's own sampled tokens,
so both are one forward pass with a gather -- no top-k, no backward, no strip of
anything that is trained.

WHAT IS NOT HERE YET, ON PURPOSE. The scores are not turned into advantages.
The question they exist to answer first is whether either of them separates
winners from losers in the LIVE groups, where the outcome is known and the
ranking can be checked: a scorer that cannot order a live group has no business
ordering a degenerate one. ``live_auc`` is that check and the probe reports it.
"""

from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["document_rows", "with_document", "trajectory_scores", "live_auc",
           "rank_report"]

DOC_OFF, DOC_LEN, DOC_REPL, DOC_REPL_LEN = (
    "oci_doc_off", "oci_doc_len", "oci_doc_repl", "oci_doc_repl_len")


def document_rows(batch) -> np.ndarray:
    """Rows whose document-conditioned prompt was recorded and verified.

    Length 0 means the rollout could not record the edit for that row -- no
    document for the instance, or a document render that did not fit the prompt
    window. Those rows are reported, never approximated.
    """
    col = batch.batch.get(DOC_LEN, None)
    if col is None:
        return np.zeros(len(batch), dtype=bool)
    return col.reshape(-1).detach().cpu().numpy() > 0


def with_document(batch, pad_token_id: int):
    """``(batch_with_document, spliced)``: the same rows, conditioned.

    The document makes the prompt LONGER, and ``splice_span`` refuses a row whose
    replacement no longer fits its prompt window (it passes such a row through
    untouched, which would silently score the plain prompt instead). So the
    window is widened by the largest growth in the batch FIRST -- left-padding
    costs nothing at the forward, since the attention mask is what the model
    reads -- and ``spliced`` says which rows actually took the edit.
    """
    import torch

    from verl.protocol import DataProto
    from verl.trainer.ppo.oci_reachability import splice_span

    ids, am = batch.batch["input_ids"], batch.batch["attention_mask"]
    pos = batch.batch.get("position_ids", None)
    resp_len = int(batch.batch["responses"].shape[1])
    plen = int(ids.shape[1]) - resp_len

    off = batch.batch[DOC_OFF].reshape(-1)
    take = batch.batch[DOC_LEN].reshape(-1)
    repl = batch.batch[DOC_REPL]
    rlen = batch.batch[DOC_REPL_LEN].reshape(-1)
    grow = int(torch.clamp(rlen - take, min=0).max().item()) if len(batch) else 0

    if grow:
        pad = torch.full((ids.shape[0], grow), int(pad_token_id), dtype=ids.dtype, device=ids.device)
        zero = torch.zeros((ids.shape[0], grow), dtype=am.dtype, device=am.device)
        ids = torch.cat([pad, ids[:, :plen], ids[:, plen:]], dim=1)
        am = torch.cat([zero, am[:, :plen], am[:, plen:]], dim=1)
        if pos is not None:
            pos = torch.cat([torch.zeros((pos.shape[0], grow), dtype=pos.dtype, device=pos.device),
                             pos[:, :plen], pos[:, plen:]], dim=1)
        plen += grow

    live_before = am[:, :plen].sum(-1)
    new_ids, new_am, new_pos = splice_span(ids, am, off, take, repl, rlen, int(pad_token_id),
                                           response_length=resp_len, position_ids=pos)
    want = live_before - take + rlen
    spliced = ((new_am[:, :plen].sum(-1) == want) & (take > 0)).detach().cpu().numpy()

    tensors = {k: v for k, v in batch.batch.items()}
    # The three that carry the prompt window; everything else is response-width.
    tensors.update({"input_ids": new_ids, "attention_mask": new_am})
    if new_pos is not None:
        tensors["position_ids"] = new_pos
    out = DataProto.from_dict(tensors=tensors, non_tensors=dict(batch.non_tensor_batch))
    out.meta_info = dict(batch.meta_info)
    return out, spliced


def trajectory_scores(log_probs, response_mask, traj_uid) -> Dict[str, float]:
    """Length-normalised mean log-prob per trajectory.

    Normalised because a trajectory's row count is its turn count and its rows'
    widths are its response lengths: a sum would rank by how much was written.
    """
    import torch

    lp = (log_probs * response_mask).sum(-1).detach().float().cpu().numpy()
    n = response_mask.sum(-1).detach().float().cpu().numpy()
    tot: Dict[str, float] = defaultdict(float)
    cnt: Dict[str, float] = defaultdict(float)
    for i, u in enumerate(traj_uid):
        tot[str(u)] += float(lp[i])
        cnt[str(u)] += float(n[i])
    return {u: (tot[u] / cnt[u]) for u in tot if cnt[u] > 0}


def live_auc(pairs: Sequence[tuple]) -> Optional[float]:
    """P(score of a winner > score of a loser), ties counted as half.

    ``pairs`` is ``(score, won)`` over trajectories that are comparable -- in
    practice one live group's, pooled across groups, since a score is only
    meaningful against the same prompt.
    """
    wins = [s for s, w in pairs if w]
    losses = [s for s, w in pairs if not w]
    if not wins or not losses:
        return None
    n = 0.0
    for a in wins:
        for b in losses:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / (len(wins) * len(losses))


def rank_report(batch, scores: Dict[str, Dict[str, float]], *, tasks=("alfworld",),
                group_status: Optional[Dict[str, str]] = None) -> dict:
    """Per-group records and the live-group AUC of every scorer.

    ``scores`` maps a scorer's name to ``{traj_uid: score}``.
    ``group_status`` maps a group's uid to live/stuck/saturated; when absent the
    class is derived from the returns in the batch.
    """
    from verl.trainer.ppo.metric_utils import get_task_names
    from verl.trainer.ppo.oci_saturated import classify_returns

    uids = batch.non_tensor_batch["uid"]
    tuids = batch.non_tensor_batch["traj_uid"]
    task_names = get_task_names(batch)
    rets = batch.non_tensor_batch.get("episode_rewards", None)
    if rets is None:
        rets = batch.batch["token_level_rewards"].sum(-1).detach().float().cpu().numpy()
    ret = np.asarray([float(r) for r in rets], dtype=float)

    if group_status is None:
        groups = classify_returns(ret, uids, tuids, task_names)
        group_status = {str(u): g["status"] for u, g in groups.items()}

    # one record per trajectory
    by_traj: Dict[str, dict] = {}
    for i in range(len(ret)):
        u, t = str(uids[i]), str(tuids[i])
        r = by_traj.setdefault(t, {"uid": u, "task": str(task_names[i]) if task_names is not None else "",
                                   "rows": 0, "ret": ret[i]})
        r["rows"] += 1
        r["ret"] = max(r["ret"], ret[i])
    for t, r in by_traj.items():
        r["won"] = bool(r["ret"] > 0.0)
        r["status"] = group_status.get(r["uid"], "")
        for name, sc in scores.items():
            r[name] = sc.get(t)

    wanted = set(str(x) for x in tasks)
    out = {"trajectories": len(by_traj), "scorers": sorted(scores)}
    for name in scores:
        pooled, per_group = [], []
        for u in {r["uid"] for r in by_traj.values()}:
            rows = [r for r in by_traj.values()
                    if r["uid"] == u and r["status"] == "live" and (not wanted or r["task"] in wanted)
                    and r.get(name) is not None]
            if len(rows) < 2:
                continue
            pairs = [(r[name], r["won"]) for r in rows]
            a = live_auc(pairs)
            if a is not None:
                per_group.append(a)
                pooled.extend(pairs)
        out[name] = {
            "live_groups_scored": len(per_group),
            "auc_mean_over_groups": float(np.mean(per_group)) if per_group else None,
            "auc_pooled": live_auc(pooled),
            "scored_trajectories": sum(1 for r in by_traj.values() if r.get(name) is not None),
        }
    counts: Dict[str, int] = defaultdict(int)
    for r in by_traj.values():
        counts[r["status"] or "?"] += 1
    out["trajectories_by_class"] = dict(counts)
    out["records"] = sorted(by_traj.values(), key=lambda r: (r["uid"], -float(r["ret"])))[:400]
    return out
