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

__all__ = ["document_rows", "with_document", "row_sums", "parse_actions",
           "walkthrough_progress", "build_trajectories", "auc", "group_auc_stats",
           "divergence_accuracy", "degenerate_diagnostics", "rank_report"]

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


def row_sums(log_probs, response_mask):
    """``(sum, count)`` of log-probs over each row's response tokens, as numpy.

    Sums and counts rather than means, so a trajectory's score is the TOKEN-
    weighted mean over all its rows, and a difference of two scorers on the same
    rows is exactly the mean per-token log-ratio (log q - log pi) -- the form
    most of the self-distillation literature scores with.
    """
    m = response_mask.to(log_probs.dtype)
    return ((log_probs * m).sum(-1).detach().float().cpu().numpy(),
            m.sum(-1).detach().float().cpu().numpy())


def parse_actions(tokenizer, responses, response_mask) -> list:
    """The action each row sent to the environment: the ``<action>`` span,
    lower-cased, as alfworld's projection reads it. '' when there is none."""
    import re

    out = []
    for i in range(responses.shape[0]):
        ids = responses[i][response_mask[i].bool()].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        m = re.search(r"<action>(.*?)</action>", text, flags=re.S | re.I)
        out.append(m.group(1).strip().lower() if m else "")
    return out


def walkthrough_progress(actions, walk):
    """``(cover, lcs)``: how much of the walkthrough a trajectory executed.

    cover  share of the walkthrough's actions that appear anywhere in it
    lcs    longest common subsequence with the walkthrough, over its length
    Both are NaN without a walkthrough. A proxy -- a trajectory that fetched the
    same object from another receptacle is under-counted -- and it shares its
    source with the privileged scorer, so a correlation between the two is
    partly mechanical (see the caution in the module docstring).
    """
    w = [str(a).strip().lower() for a in (walk or [])]
    if not w:
        return float("nan"), float("nan")
    a = [x for x in actions if x]
    cover = sum(1 for x in w if x in set(a)) / len(w)
    dp = [0] * (len(w) + 1)
    for x in a:
        prev = 0
        for j in range(1, len(w) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == w[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return cover, dp[len(w)] / len(w)


def build_trajectories(*, uids, tuids, turn_steps, returns, tasks, gamefiles, actions,
                       sums: Dict[str, np.ndarray], counts: np.ndarray):
    """One record per trajectory, from the scored rows only.

    ``sums[name][i]`` is row i's summed log-prob under scorer ``name``;
    ``counts[i]`` its response tokens (the same for every scorer: they all score
    the student's own tokens). Scores are token-weighted means; a ``*_gain``
    scorer is added for every scorer other than ``plain`` when ``plain`` exists.
    """
    recs: Dict[str, dict] = {}
    for i in range(len(tuids)):
        t = str(tuids[i])
        r = recs.setdefault(t, {"traj": t, "uid": str(uids[i]), "task": str(tasks[i]),
                                "gamefile": str(gamefiles[i]) if gamefiles is not None else "",
                                "ret": float(returns[i]), "tokens": 0.0,
                                "_rows": [], "_sum": defaultdict(float)})
        r["ret"] = max(r["ret"], float(returns[i]))
        r["tokens"] += float(counts[i])
        r["_rows"].append((int(turn_steps[i]), i, actions[i]))
        for name, arr in sums.items():
            r["_sum"][name] += float(arr[i])
    for r in recs.values():
        r["_rows"].sort()
        r["turns"] = len(r["_rows"])
        r["actions"] = [a for _, _, a in r["_rows"]]
        r["won"] = bool(r["ret"] > 0.0)
        n = max(r["tokens"], 1.0)
        r["score"] = {name: r["_sum"][name] / n for name in sums}
        if "plain" in sums:
            for name in sums:
                if name != "plain":
                    r["score"][f"{name}_gain"] = (r["_sum"][name] - r["_sum"]["plain"]) / n
    return recs


def classify_groups(recs) -> Dict[str, str]:
    """live / stuck / saturated per group, from the trajectories in it."""
    by = defaultdict(list)
    for r in recs.values():
        by[r["uid"]].append(r["won"])
    return {u: ("saturated" if all(w) else "stuck" if not any(w) else "live")
            for u, w in by.items()}


def auc(winners, losers) -> Optional[float]:
    """P(winner > loser), ties counted as half; None without both kinds."""
    if not winners or not losers:
        return None
    n = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in winners for b in losers)
    return n / (len(winners) * len(losers))


def _spearman(x, y) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _residualise(score, turns):
    """Score minus its within-group least-squares fit on turn count."""
    s, t = np.asarray(score, float), np.asarray(turns, float)
    if t.std() == 0:
        res = s - s.mean()
    else:
        res = s - np.polyval(np.polyfit(t, s, 1), t)
    # Rounded, so a score that turn count explains EXACTLY leaves ties and not
    # floating-point noise for the AUC to rank. Scores are log-probs of order
    # 0.1-1; 1e-9 is far below any difference that means something.
    return np.round(res, 9)


def group_auc_stats(recs, status, name, *, adjust=None, n_boot=1000, seed=0) -> dict:
    """Within-group AUC of scorer ``name`` over LIVE groups, with a bootstrap CI.

    ``adjust="turns"`` scores the residual of a within-group fit on turn count
    instead of the raw score. In alfworld a loss IS a run to the 50-turn cap, so
    a raw AUC mostly measures "the long, looping trajectory is less likely" --
    which any scorer sees; the residual asks what is left once that is removed.
    """
    per, pooled_w, pooled_l, rho_len = [], 0.0, 0.0, []
    groups = defaultdict(list)
    for r in recs.values():
        if status.get(r["uid"]) == "live" and name in r["score"]:
            groups[r["uid"]].append(r)
    num = den = 0.0
    for u, rs in groups.items():
        sc = [x["score"][name] for x in rs]
        tr = [x["turns"] for x in rs]
        if adjust == "turns":
            sc = list(_residualise(sc, tr))
        W = [s for s, x in zip(sc, rs) if x["won"]]
        L = [s for s, x in zip(sc, rs) if not x["won"]]
        a = auc(W, L)
        if a is None:
            continue
        per.append(a)
        num += a * len(W) * len(L)
        den += len(W) * len(L)
        rho = _spearman([x["score"][name] for x in rs], tr)
        if rho is not None:
            rho_len.append(rho)
    out = {"live_groups": len(per),
           "auc_mean": float(np.mean(per)) if per else None,
           "auc_pooled": (num / den) if den else None,
           "spearman_score_turns_mean": float(np.mean(rho_len)) if rho_len else None}
    if len(per) >= 2:
        rng = np.random.default_rng(seed)
        boots = [float(np.mean(rng.choice(per, size=len(per), replace=True))) for _ in range(n_boot)]
        out["auc_mean_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def divergence_accuracy(recs, status, row_sums_by_name, counts, name, *, n_boot=1000, seed=0):
    """At the first turn where a live group's trajectories take DIFFERENT
    actions, does the scorer prefer the row of a trajectory that went on to win?

    NOT CONFOUNDED BY LENGTH. Everything before that turn is identical across the
    group -- same observations, same actions, same history in the prompt -- so
    the rows compared share a prompt and differ only in the one response. Each
    winner/loser pair that took different actions there is one comparison. This
    is the check OVCSD (arXiv 2607.27937) reports at or below chance for a
    skill-conditioned self on ALFWorld and WebShop.
    """
    base, gain = name, None
    if name.endswith("_gain"):
        base, gain = name[: -len("_gain")], True
    groups = defaultdict(list)
    for r in recs.values():
        if status.get(r["uid"]) == "live":
            groups[r["uid"]].append(r)
    per, total_pairs = [], 0
    for u, rs in groups.items():
        by_t = defaultdict(dict)
        for r in rs:
            for t, i, a in r["_rows"]:
                by_t[t][r["traj"]] = (i, a)
        t_div = next((t for t in sorted(by_t) if len({a for _, a in by_t[t].values()}) > 1), None)
        if t_div is None:
            continue
        hits = n = 0.0
        won = {r["traj"]: r["won"] for r in rs}
        rows = by_t[t_div]
        for tw, (iw, aw) in rows.items():
            if not won[tw]:
                continue
            for tl, (il, al) in rows.items():
                if won[tl] or aw == al:
                    continue
                def sc(i):
                    v = row_sums_by_name[base][i]
                    if gain:
                        v = v - row_sums_by_name["plain"][i]
                    return v / max(counts[i], 1.0)
                a, b = sc(iw), sc(il)
                hits += 1.0 if a > b else 0.5 if a == b else 0.0
                n += 1
        if n:
            per.append(hits / n)
            total_pairs += int(n)
    out = {"groups": len(per), "pairs": total_pairs,
           "accuracy_mean": float(np.mean(per)) if per else None}
    if len(per) >= 2:
        rng = np.random.default_rng(seed)
        boots = [float(np.mean(rng.choice(per, size=len(per), replace=True))) for _ in range(n_boot)]
        out["accuracy_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def degenerate_diagnostics(recs, status, name, walk_of=None) -> dict:
    """What the scorer would do INSIDE the groups it is meant for.

    stuck      every trajectory failed at the cap, so length says nothing; is the
               score's order related to how much of the walkthrough each one
               executed? (cover / lcs, see walkthrough_progress)
    saturated  every trajectory won; does the score prefer the shorter ones?
    """
    out = {}
    for cls in ("stuck", "saturated"):
        groups = defaultdict(list)
        for r in recs.values():
            if status.get(r["uid"]) == cls and name in r["score"]:
                groups[r["uid"]].append(r)
        rho_turns, rho_cover, rho_lcs, spread = [], [], [], []
        for u, rs in groups.items():
            sc = [x["score"][name] for x in rs]
            spread.append(float(np.max(sc) - np.min(sc)))
            v = _spearman(sc, [x["turns"] for x in rs])
            if v is not None:
                rho_turns.append(v)
            if cls == "stuck" and walk_of is not None:
                prog = [walkthrough_progress(x["actions"], walk_of(x["gamefile"])) for x in rs]
                c = _spearman(sc, [p[0] for p in prog])
                l = _spearman(sc, [p[1] for p in prog])
                if c is not None:
                    rho_cover.append(c)
                if l is not None:
                    rho_lcs.append(l)
        m = lambda v: float(np.mean(v)) if v else None
        out[cls] = {"groups": len(groups), "score_spread_mean": m(spread),
                    "spearman_score_turns_mean": m(rho_turns)}
        if cls == "stuck":
            out[cls].update({"spearman_score_walk_cover_mean": m(rho_cover),
                             "spearman_score_walk_lcs_mean": m(rho_lcs)})
    return out


def rank_report(recs, row_sums_by_name, counts, *, walk_of=None, n_boot=1000) -> dict:
    """Every check, for every scorer, in one payload."""
    status = classify_groups(recs)
    names = sorted({n for r in recs.values() for n in r["score"]})
    out = {"trajectories": len(recs), "scorers": names,
           "groups_by_class": {c: sum(1 for v in status.values() if v == c)
                               for c in ("live", "stuck", "saturated")}}
    for name in names:
        out[name] = {
            "live_auc": group_auc_stats(recs, status, name, n_boot=n_boot),
            "live_auc_length_adjusted": group_auc_stats(recs, status, name, adjust="turns", n_boot=n_boot),
            "first_divergence": divergence_accuracy(recs, status, row_sums_by_name, counts, name, n_boot=n_boot),
            "degenerate": degenerate_diagnostics(recs, status, name, walk_of=walk_of),
        }
    keep = ("traj", "uid", "task", "gamefile", "ret", "won", "turns", "tokens", "score")
    out["records"] = [dict({k: r[k] for k in keep}, status=status[r["uid"]],
                           actions=r["actions"][:60])
                      for r in sorted(recs.values(), key=lambda r: (r["uid"], -r["ret"]))][:600]
    return out
