#!/usr/bin/env python
"""Read the route_hint probe's rollout records (run_search_rescue_probe_qwen3.sh with
SEARCH_DOC=route_hint SEARCH_DOC_B=none GROUP_N=10) and answer steps 2 and 3 of the plan.

THE LAYOUT. Each group is one question: eight plain rows (role 1) decide its class
(stuck = none of the eight won, saturated = all eight, live otherwise), one reserve row
(role 2) is a ninth plain rollout that did NOT take part in the classification, and the
document row (role 3) wears the route without the answer.

  2 RESCUE   on stuck groups, the document row's success against the reserve row's.
             The reserve is the selection-free baseline: a plain row of a stuck group
             succeeds only by chance, and the eight that defined "stuck" cannot measure
             that chance (they are 0 by construction). Paired on the question (McNemar).
  3 SAFETY   does the document row stop searching or put the answer in a query?
             searches per row, answers sent with no search, rows that wrote an accepted
             answer before any result carried it (the env's answer_early), queries that
             name an answer the question does not, evidence seen, answer sent at all.
             The same numbers for the reserve and plain rows are the reference.

Usage: analyze_search_route_hint.py <dump-dir> --routes route_hints.json [--json out.json]
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from agent_system.environments.oci_layout import contains_answer, fold_text  # noqa: E402

PLAIN, RESERVE, DOC = 1, 2, 3


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def won(r):
    return float(r.get("won") or 0.0) >= 1.0


def queries(r):
    return [t.get("query") for t in (r.get("turns") or []) if str(t.get("query") or "").strip()]


def answered(r):
    return any("<answer>" in str(t.get("action") or "") and not str(t.get("query") or "").strip()
               for t in (r.get("turns") or []))


def row_stats(r, route_q):
    qs = queries(r)
    ans = r.get("answers") or []
    q_text = r.get("question") or ""
    leak_q = any(contains_answer(q, ans) and not contains_answer(q_text, ans) for q in qs)
    route_f = [fold_text(x) for x in route_q]
    copied = sum(1 for q in qs if fold_text(q) in route_f)
    return {"won": won(r), "searches": len(qs), "answered": answered(r),
            "answered_no_search": answered(r) and len(qs) == 0,
            "answer_early": bool(r.get("answer_early")), "evidence_seen": bool(r.get("evidence_seen")),
            "query_names_answer": leak_q, "won_kept_rule": won(r) and not bool(r.get("answer_early")),
            "copied_route_share": (copied / len(qs)) if qs else None,
            # The plain student opens with <search> almost always; a prompt that changes that
            # has changed the row's behaviour, whatever it then scores.
            "opens_with_search": str(((r.get("turns") or [{}])[0] or {}).get("action") or "").lstrip()
                                 .startswith("<search>"),
            "turns": int(r.get("n_turns") or len(r.get("turns") or []))}


def summarise(stats):
    n = len(stats)
    if n == 0:
        return {"rows": 0}
    out = {"rows": n}
    for key in ("won", "won_kept_rule", "answered", "answered_no_search", "answer_early",
                "evidence_seen", "query_names_answer", "opens_with_search"):
        k = sum(1 for s in stats if s[key])
        out[key] = round(k / n, 3)
        if key in ("won", "won_kept_rule"):
            out[key + "_ci95"] = wilson(k, n)
    out["searches_per_row"] = round(sum(s["searches"] for s in stats) / n, 3)
    out["turns_per_row"] = round(sum(s["turns"] for s in stats) / n, 3)
    cs = [s["copied_route_share"] for s in stats if s["copied_route_share"] is not None]
    out["copied_route_query_share"] = round(sum(cs) / len(cs), 3) if cs else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    routes = {}
    for v in json.load(open(a.routes))["flows"].values():
        routes[fold_text(v["question"])] = v
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
    bad = [g for g, rs in groups.items()
           if len(rs) != 10 or len({fold_text(r["question"]) for r in rs}) != 1]
    groups = {g: rs for g, rs in groups.items() if g not in bad}

    cls_rows = collections.defaultdict(lambda: collections.defaultdict(list))
    paired = collections.defaultdict(lambda: [0, 0, 0, 0])  # (doc, reserve): 11, 10, 01, 00
    plain_ev = collections.Counter()
    by_source = collections.defaultdict(lambda: collections.defaultdict(list))
    by_kind = collections.defaultdict(lambda: collections.defaultdict(list))
    for g, rs in groups.items():
        plain = [r for r in rs if r["role"] == PLAIN]
        res = [r for r in rs if r["role"] == RESERVE]
        doc = [r for r in rs if r["role"] == DOC]
        if len(plain) != 8 or len(res) != 1 or len(doc) != 1:
            continue
        w = sum(won(r) for r in plain)
        cls = "stuck" if w == 0 else ("saturated" if w == 8 else "live")
        route = routes.get(fold_text(rs[0]["question"]), {})
        route_q = route.get("queries", []) if route.get("hit") else []
        d, rv = doc[0], res[0]
        sd, sr = row_stats(d, route_q), row_stats(rv, route_q)
        cls_rows[cls]["plain"] += [row_stats(r, route_q) for r in plain]
        cls_rows[cls]["reserve"].append(sr)
        cls_rows[cls]["doc_all"].append(sd)
        if d.get("has_document"):
            cls_rows[cls]["doc"].append(sd)
            cls_rows[cls]["reserve_same_groups"].append(sr)
            cell = paired[cls]
            cell[(0 if sd["won"] else 2) + (0 if sr["won"] else 1)] += 1
            if cls == "stuck":
                by_source[d.get("data_source")]["doc"].append(sd)
                by_source[d.get("data_source")]["reserve"].append(sr)
                by_kind[route.get("kind")]["doc"].append(sd)
                by_kind[route.get("kind")]["reserve"].append(sr)
        if cls == "stuck":
            seen = sum(bool(r.get("evidence_seen")) for r in plain)
            plain_ev["none_saw" if seen == 0 else ("all_saw" if seen == 8 else "some_saw")] += 1
            plain_ev["route" if route.get("hit") else ("yesno" if route.get("kind") == "yesno" else "no_route")] += 1
    out = {"groups": len(groups), "groups_dropped": len(bad),
           "classes": {c: len(v["reserve"]) for c, v in cls_rows.items()}}
    for cls, parts in cls_rows.items():
        out[cls] = {k: summarise(v) for k, v in parts.items()}
        b, c = paired[cls][1], paired[cls][2]  # doc won & reserve lost, doc lost & reserve won
        out[cls]["paired_doc_vs_reserve"] = {"both": paired[cls][0], "doc_only": b, "reserve_only": c,
                                            "neither": paired[cls][3], "mcnemar_p": round(mcnemar_p(b, c), 5)}
    out["stuck_plain_evidence"] = dict(plain_ev)
    out["stuck_by_source"] = {s: {k: summarise(v) for k, v in p.items()} for s, p in by_source.items()}
    out["stuck_by_route_kind"] = {s: {k: summarise(v) for k, v in p.items()} for s, p in by_kind.items()}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
