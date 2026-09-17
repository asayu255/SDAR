#!/usr/bin/env python
"""Read run_search_rescue_probe_qwen3.sh's rollout records and answer its two questions.

  1 WHY SEARCH GROUPS STALL. For every group, the eight ordinary rows decide its
    class (stuck / live / saturated). Inside the stuck ones, how many rows had a
    result that CONTAINED the answer? Those rows searched well and failed at
    answering, and a document about how to search cannot rescue them.

  2 WHETHER THE RESCUE ROW WORKS. The ninth row is shown the answer under the
    rule that a returned result has to carry it first. Two rates: what the
    environment scored (which an answer copied out of the prompt also earns) and
    what the rule keeps.

Usage: analyze_search_rescue.py <dump-dir> [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

ROLE_DOC, ROLE_FOREIGN = 3, 4


def tag_sources(rows, train_parquet):
    """Label every record nq / hotpotqa off the training data.

    The env manager sees only the reset kwargs (question and answers), so the
    record's data_source is the task name. The split matters -- nq's questions
    already return the answer to a verbatim search far more often than hotpotqa's
    bridge questions do -- so it is recovered here by the question text.
    """
    import pyarrow.parquet as pq

    def norm(s):
        return re.sub(r"[^0-9a-z]+", " ", str(s).lower()).strip()

    table = pq.read_table(train_parquet, columns=["data_source", "env_kwargs"])
    by_q = {}
    for src, kw in zip(table.column("data_source").to_pylist(),
                       table.column("env_kwargs").to_pylist()):
        q = (kw or {}).get("question")
        if q:
            by_q[norm(q)] = src
    hit = 0
    for r in rows:
        src = by_q.get(norm(r.get("question")))
        if src:
            r["data_source"] = src
            hit += 1
    return hit, len(rows)


def load(dump_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "search_rollouts.*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def group_key(r):
    return (r.get("pid"), r.get("reset"), r.get("group"))


def rate(num, den):
    return None if not den else round(num / den, 3)


def summarise(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    out = {"rollouts": len(rows), "groups": len(groups),
           "plain": {}, "rescue": {}, "by_source": {},
           # A group is one question repeated group_n times (the batch is repeated
           # with interleave=True). If that ever stops holding, the class of a
           # group is read off rows from different questions and every number
           # below is meaningless -- so it is counted rather than assumed.
           "groups_mixed_question": 0, "groups_wrong_size": 0}
    cls_count = collections.Counter()
    # the eight ordinary rows, per class
    plain = {c: collections.Counter() for c in ("stuck", "live", "saturated")}
    # the document row, keyed by the class of its group's ordinary rows
    resc = {c: collections.Counter() for c in ("stuck", "live", "saturated")}
    by_source = collections.defaultdict(collections.Counter)

    for key, rs in groups.items():
        if len({r.get("question") for r in rs}) > 1:
            out["groups_mixed_question"] += 1
        ordinary = [r for r in rs if int(r.get("role", 0)) not in (ROLE_DOC, ROLE_FOREIGN)]
        doc = [r for r in rs if int(r.get("role", 0)) == ROLE_DOC]
        if len(ordinary) != 8 or len(doc) > 1:
            out["groups_wrong_size"] += 1
        if not ordinary:
            continue
        wins = [float(r.get("won") or 0.0) > 0 for r in ordinary]
        cls = "stuck" if not any(wins) else ("saturated" if all(wins) else "live")
        cls_count[cls] += 1
        src = (ordinary[0].get("data_source") or "unknown")

        c = plain[cls]
        c["groups"] += 1
        c["rows"] += len(ordinary)
        c["rows_won"] += sum(wins)
        # A row whose search returned the answer at least once.
        c["rows_evidence"] += sum(1 for r in ordinary if r.get("evidence_seen"))
        # A row that wrote the answer before any result carried it: it already
        # knew it (and then either scored or dropped it).
        c["rows_wrote_unseen"] += sum(1 for r in ordinary if r.get("answer_early"))
        c["groups_any_evidence"] += 1 if any(r.get("evidence_seen") for r in ordinary) else 0
        c["turns"] += sum(int(r.get("n_turns") or 0) for r in ordinary)

        if cls == "stuck":
            s = by_source[src]
            s["stuck_groups"] += 1
            s["stuck_groups_any_evidence"] += 1 if any(r.get("evidence_seen") for r in ordinary) else 0
            s["stuck_rows"] += len(ordinary)
            s["stuck_rows_evidence"] += sum(1 for r in ordinary if r.get("evidence_seen"))
        by_source[src]["groups"] += 1
        by_source[src][f"groups_{cls}"] += 1

        for r in doc:
            d = resc[cls]
            d["rows"] += 1
            # has_document is False for a yes/no question, whose slot ran plain:
            # the rule is a string test and "yes" is in almost any passage.
            if not r.get("has_document", bool(r.get("answers"))):
                d["no_document"] += 1
                continue
            won = float(r.get("won") or 0.0) > 0
            kept = won and not r.get("answer_early")
            d["scored"] += 1 if won else 0
            d["scored_and_kept_rule"] += 1 if kept else 0
            d["broke_rule"] += 1 if r.get("answer_early") else 0
            d["evidence_seen"] += 1 if r.get("evidence_seen") else 0
            d["turns"] += int(r.get("n_turns") or 0)

    out["classes"] = dict(cls_count)
    for cls in ("stuck", "live", "saturated"):
        c = plain[cls]
        out["plain"][cls] = {
            "groups": c["groups"],
            "rows": c["rows"],
            "row_success": rate(c["rows_won"], c["rows"]),
            "rows_with_evidence": rate(c["rows_evidence"], c["rows"]),
            "groups_with_any_evidence": rate(c["groups_any_evidence"], c["groups"]),
            "rows_wrote_answer_unseen": rate(c["rows_wrote_unseen"], c["rows"]),
            "turns_per_row": rate(c["turns"], c["rows"]),
        }
        d = resc[cls]
        out["rescue"][cls] = {
            "rows": d["rows"],
            "no_document": d["no_document"],
            "scored": rate(d["scored"], d["rows"] - d["no_document"]),
            "scored_and_kept_rule": rate(d["scored_and_kept_rule"], d["rows"] - d["no_document"]),
            "broke_rule": rate(d["broke_rule"], d["rows"] - d["no_document"]),
            "evidence_seen": rate(d["evidence_seen"], d["rows"] - d["no_document"]),
            "turns_per_row": rate(d["turns"], d["rows"] - d["no_document"]),
        }
    for src, c in by_source.items():
        out["by_source"][src] = {
            "groups": c["groups"],
            "stuck": rate(c["groups_stuck"], c["groups"]),
            "stuck_groups_with_any_evidence": rate(c["stuck_groups_any_evidence"], c["stuck_groups"]),
            "stuck_rows_with_evidence": rate(c["stuck_rows_evidence"], c["stuck_rows"]),
        }
    return out


def report(s):
    print(f"rollouts {s['rollouts']}  groups {s['groups']}  classes {s['classes']}")
    if s.get("groups_mixed_question") or s.get("groups_wrong_size"):
        print(f"  WARNING  groups whose rows are not one question x 9: "
              f"mixed question {s['groups_mixed_question']}, wrong size {s['groups_wrong_size']}")
    print("\n1. the eight ordinary rows")
    print(f"{'class':<10}{'groups':>7}{'success':>9}{'rows w/ evidence':>19}"
          f"{'groups w/ evidence':>20}{'wrote unseen':>14}")
    for cls in ("stuck", "live", "saturated"):
        p = s["plain"][cls]
        print(f"{cls:<10}{p['groups']:>7}{str(p['row_success']):>9}"
              f"{str(p['rows_with_evidence']):>19}{str(p['groups_with_any_evidence']):>20}"
              f"{str(p['rows_wrote_answer_unseen']):>14}")
    print("\n2. the document row, by the class of its group")
    print(f"{'class':<10}{'rows':>6}{'scored':>9}{'kept rule':>11}{'broke rule':>12}"
          f"{'evidence':>10}{'no doc':>8}")
    for cls in ("stuck", "live", "saturated"):
        d = s["rescue"][cls]
        print(f"{cls:<10}{d['rows']:>6}{str(d['scored']):>9}{str(d['scored_and_kept_rule']):>11}"
              f"{str(d['broke_rule']):>12}{str(d['evidence_seen']):>10}{d['no_document']:>8}")
    if s["by_source"]:
        print("\n3. by dataset")
        print(f"{'source':<12}{'groups':>7}{'stuck':>8}{'stuck w/ evidence':>19}"
              f"{'stuck rows w/ evidence':>24}")
        for src, c in sorted(s["by_source"].items()):
            print(f"{src:<12}{c['groups']:>7}{str(c['stuck']):>8}"
                  f"{str(c['stuck_groups_with_any_evidence']):>19}"
                  f"{str(c['stuck_rows_with_evidence']):>24}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--train-parquet", default=None,
                    help="the Search training data, to split the report into nq and hotpotqa")
    args = ap.parse_args()
    rows = load(args.dump_dir)
    if not rows:
        print(f"no records under {args.dump_dir}", file=sys.stderr)
        return 1
    if args.train_parquet:
        hit, total = tag_sources(rows, args.train_parquet)
        print(f"dataset tagged for {hit}/{total} records")
    s = summarise(rows)
    report(s)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
