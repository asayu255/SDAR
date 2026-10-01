#!/usr/bin/env python
"""Read a one-sentence probe (run_search_rescue_probe_qwen3.sh with SEARCH_DOC=route_hint
SEARCH_DOC_B=route_hint, doc_sentence=explore|efficiency, doc_sentence_b=none, GROUP_N=10).

THE LAYOUT. Seven plain rows (role 1) decide the group's class (stuck = none won, saturated =
all won, live otherwise); the reserve row (role 2) is a plain rollout outside the
classification, the selection-free baseline; the second document row (role 5) wears the
document alone; the document row (role 3) wears the same document plus the sentence. The
two document rows are paired on the question, so the sentence's effect is read as
document+sentence minus document, group by group.

Per class and per row kind: success, success without writing the answer before a result
carried it, answers sent with no search, rows opening with <search>, searches per row,
repeated queries, searches sent AFTER a result already carried the answer, evidence seen,
turns, characters written per turn. Paired: McNemar on success, and the mean paired
difference (with a bootstrap 95% interval) for the counts.

Usage: analyze_search_doc_sentence.py <dump-dir> [--json out.json]
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from agent_system.environments.oci_layout import contains_answer, fold_text  # noqa: E402

PLAIN, RESERVE, DOC, DOC_B = 1, 2, 3, 5


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot_ci(diffs, n_boot=2000, seed=0):
    if not diffs:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)]


def stats(r):
    turns = r.get("turns") or []
    qs = [fold_text(t.get("query")) for t in turns if str(t.get("query") or "").strip()]
    first_ev = next((k for k, t in enumerate(turns) if t.get("info_has_answer")), None)
    after = sum(1 for k, t in enumerate(turns)
                if first_ev is not None and k > first_ev and str(t.get("query") or "").strip())
    answered = any("<answer>" in str(t.get("action") or "") and not str(t.get("query") or "").strip()
                   for t in turns)
    won = float(r.get("won") or 0.0) >= 1.0
    chars = [len(str(t.get("action") or "")) for t in turns]
    return {"won": won, "won_kept_rule": won and not bool(r.get("answer_early")),
            "answered_no_search": answered and not qs, "answer_early": bool(r.get("answer_early")),
            "opens_with_search": str((turns[0] if turns else {}).get("action") or "").lstrip().startswith("<search>"),
            "evidence_seen": bool(r.get("evidence_seen")),
            "searches": len(qs), "repeated_queries": len(qs) - len(set(qs)),
            "searches_after_evidence": after, "turns": len(turns),
            "chars_per_turn": (sum(chars) / len(chars)) if chars else 0.0}


BOOL = ("won", "won_kept_rule", "answered_no_search", "answer_early", "opens_with_search", "evidence_seen")
NUM = ("searches", "repeated_queries", "searches_after_evidence", "turns", "chars_per_turn")


def summarise(rows):
    n = len(rows)
    out = {"rows": n}
    if not n:
        return out
    for k in BOOL:
        c = sum(1 for s in rows if s[k])
        out[k] = round(c / n, 3)
        if k in ("won", "won_kept_rule"):
            out[k + "_ci95"] = wilson(c, n)
    for k in NUM:
        out[k] = round(sum(s[k] for s in rows) / n, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--json")
    a = ap.parse_args()
    recs = []
    for f in sorted(glob.glob(os.path.join(a.dump, "search_rollouts.*.jsonl"))):
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["pid"], r["reset"], r["group"])].append(r)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    pairs = collections.defaultdict(list)
    sentence = None
    dropped = 0
    for g, rs in groups.items():
        P = [r for r in rs if r["role"] == PLAIN]
        V = [r for r in rs if r["role"] == RESERVE]
        D = [r for r in rs if r["role"] == DOC]
        B = [r for r in rs if r["role"] == DOC_B]
        if len(P) != 7 or len(V) != 1 or len(D) != 1 or len(B) != 1 or len({fold_text(r["question"]) for r in rs}) != 1:
            dropped += 1
            continue
        sentence = sentence or D[0].get("sentence")
        w = sum(float(r.get("won") or 0) >= 1 for r in P)
        cls = "stuck" if w == 0 else ("saturated" if w == 7 else "live")
        by[cls]["plain"] += [stats(r) for r in P]
        by[cls]["reserve"].append(stats(V[0]))
        if D[0].get("has_document") and B[0].get("has_document"):
            sd, sb = stats(D[0]), stats(B[0])
            by[cls]["doc_plus_sentence"].append(sd)
            by[cls]["doc_only"].append(sb)
            by[cls]["reserve_same_groups"].append(stats(V[0]))
            pairs[cls].append((sd, sb))
    out = {"sentence": sentence, "groups": len(groups) - dropped, "groups_dropped": dropped,
           "classes": {c: len(v["reserve"]) for c, v in by.items()}}
    for cls, parts in by.items():
        out[cls] = {k: summarise(v) for k, v in parts.items()}
        P_ = pairs[cls]
        b = sum(1 for d, o in P_ if d["won"] and not o["won"])
        c = sum(1 for d, o in P_ if o["won"] and not d["won"])
        paired = {"pairs": len(P_), "sentence_only_won": b, "doc_only_won": c, "mcnemar_p": round(mcnemar_p(b, c), 5)}
        for k in NUM + ("answered_no_search", "opens_with_search"):
            diffs = [float(d[k]) - float(o[k]) for d, o in P_]
            paired[k + "_diff"] = round(sum(diffs) / len(diffs), 3) if diffs else None
            paired[k + "_diff_ci95"] = boot_ci(diffs)
        out[cls]["paired_sentence_minus_doc"] = paired
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
