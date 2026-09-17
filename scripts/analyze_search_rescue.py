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

ROLE_DOC, ROLE_FOREIGN, ROLE_DOC_B = 3, 4, 5
RESCUE_ROLES = (ROLE_DOC, ROLE_DOC_B)


def fold(s):
    return re.sub(r"[^0-9a-z]+", " ", str(s).lower()).strip()


def hit_turn(r):
    """Which search first returned the answer (1-based), or None."""
    for k, t in enumerate(r.get("turns") or [], start=1):
        if t.get("info_has_answer"):
            return k
    return None


def document_queries(rec, flows):
    """The queries this row's document printed, or [] when it had none.

    expert_flow prints a verified route; answer_rule and progress_only print the
    question verbatim. Either way the row can simply copy them, which is what the
    copy rate below measures.
    """
    if not rec.get("has_document"):
        return []
    if rec.get("variant") == "expert_flow":
        return list((flows or {}).get(fold(rec.get("question")), []))
    return [rec.get("question") or ""]


def copy_rate(rec, flows):
    """How much of what the row did was the document read back.

    Returns (share of the row's queries that are a document query, whether every
    document query was run in order). A rescue row that copies the route and then
    copies the answer satisfies the rule and scores, which bounds what the rescue
    rate can mean: it is an upper bound on the document's usefulness, not evidence
    that anything was learned.
    """
    doc = [fold(q) for q in document_queries(rec, flows) if fold(q)]
    mine = [fold(t.get("query")) for t in (rec.get("turns") or []) if fold(t.get("query"))]
    if not doc or not mine:
        return None, None
    copied = sum(1 for q in mine if q in doc) / len(mine)
    in_order = [q for q in mine if q in doc] == doc[:len([q for q in mine if q in doc])]
    return copied, bool(in_order and len([q for q in mine if q in doc]) == len(doc))


def query_overlap(rescue, ordinary):
    """How much the rescue row's queries look like its plain siblings'.

    A rescue row that has read the answer can write a query no plain row could --
    naming the thing the answer is about without naming the answer. The rule
    cannot catch that, so it is measured: the share of the rescue row's query
    words that appear in some sibling's query.
    """
    sib = set()
    for r in ordinary:
        for t in r.get("turns") or []:
            sib.update(fold(t.get("query")).split())
    mine = set()
    for t in rescue.get("turns") or []:
        mine.update(fold(t.get("query")).split())
    if not mine:
        return None
    return len(mine & sib) / len(mine)


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


def summarise(rows, flows=None):
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
    by_variant = collections.defaultdict(collections.Counter)
    # The same stuck groups, restricted to those where EVERY variant had a
    # document: the only comparison in which the two documents answer for the
    # same questions. by_variant above is the wider one, where a variant that has
    # no route for a question is simply absent from its denominator.
    paired = collections.defaultdict(collections.Counter)
    by_source = collections.defaultdict(collections.Counter)

    for key, rs in groups.items():
        if len({r.get("question") for r in rs}) > 1:
            out["groups_mixed_question"] += 1
        ordinary = [r for r in rs if int(r.get("role", 0)) not in RESCUE_ROLES + (ROLE_FOREIGN,)]
        doc = [r for r in rs if int(r.get("role", 0)) in RESCUE_ROLES]
        if len(ordinary) != 8 or len(doc) > 2:
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
            variant = r.get("variant") or ("answer_rule" if int(r.get("role", 0)) == ROLE_DOC
                                           else "progress_only")
            # The rescue rate is read per class AND per variant; the variant
            # comparison is what decides which document ships, and it only means
            # anything on the groups that need rescuing.
            targets = [resc[cls]] + ([by_variant[variant]] if cls == "stuck" else [])
            if cls == "stuck" and all(d.get("has_document") for d in doc) and len(doc) > 1:
                targets.append(paired[variant])
            # has_document is False for a yes/no question, whose slot ran plain:
            # the rule is a string test and "yes" is in almost any passage.
            if not r.get("has_document", bool(r.get("answers"))):
                for d in targets:
                    d["rows"] += 1
                    d["no_document"] += 1
                continue
            won = float(r.get("won") or 0.0) > 0
            kept = won and not r.get("answer_early")
            k = hit_turn(r)
            ov = query_overlap(r, ordinary)
            short = int(r.get("answer_words") or 0) <= 1
            copied, copied_all = copy_rate(r, flows)
            for d in targets:
                d["rows"] += 1
                d["scored"] += 1 if won else 0
                d["scored_and_kept_rule"] += 1 if kept else 0
                d["broke_rule"] += 1 if r.get("answer_early") else 0
                d["evidence_seen"] += 1 if r.get("evidence_seen") else 0
                d["turns"] += int(r.get("n_turns") or 0)
                if k is not None:
                    d["hit_turns"] += k
                    d["hit_rows"] += 1
                if ov is not None:
                    d["overlap_sum"] += ov
                    d["overlap_rows"] += 1
                d["short_rows" if short else "long_rows"] += 1
                if kept:
                    d["short_kept" if short else "long_kept"] += 1
                if r.get("answer_numeric"):
                    d["numeric_rows"] += 1
                    d["numeric_kept"] += 1 if kept else 0
                if copied is not None:
                    d["copy_sum"] += copied
                    d["copy_rows"] += 1
                    d["copied_whole_route"] += 1 if copied_all else 0

    out["classes"] = dict(cls_count)
    def _rescue_block(d):
        shown = d["rows"] - d["no_document"]
        return {
            "rows": d["rows"],
            "no_document": d["no_document"],
            "scored": rate(d["scored"], shown),
            "scored_and_kept_rule": rate(d["scored_and_kept_rule"], shown),
            "broke_rule": rate(d["broke_rule"], shown),
            "evidence_seen": rate(d["evidence_seen"], shown),
            "turns_per_row": rate(d["turns"], shown),
            "searches_to_hit": rate(d["hit_turns"], d["hit_rows"]),
            "query_overlap_with_siblings": rate(d["overlap_sum"], d["overlap_rows"]),
            "kept_rule_short_answer": rate(d["short_kept"], d["short_rows"]),
            "kept_rule_long_answer": rate(d["long_kept"], d["long_rows"]),
            "numeric_answer_rows": d["numeric_rows"],
            "kept_rule_numeric_answer": rate(d["numeric_kept"], d["numeric_rows"]),
            # TWO DENOMINATORS. "shown" is the questions this document exists for,
            # which for expert_flow are the ones a route was FOUND for and so are
            # the easier half; "all" counts a question with no document as a
            # failure, which is what a training run would actually collect.
            "kept_rule_over_all_rows": rate(d["scored_and_kept_rule"], d["rows"]),
            "copied_document_queries": rate(d["copy_sum"], d["copy_rows"]),
            "copied_whole_route": rate(d["copied_whole_route"], d["copy_rows"]),
        }

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
        out["rescue"][cls] = _rescue_block(resc[cls])
    out["rescue_by_variant_on_stuck"] = {v: _rescue_block(d) for v, d in by_variant.items()}
    out["rescue_paired_on_stuck"] = {v: _rescue_block(d) for v, d in paired.items()}
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
    print("\n2. the rescue rows, by the class of their group")
    print(f"{'class':<10}{'rows':>6}{'scored':>9}{'kept rule':>11}{'broke rule':>12}"
          f"{'evidence':>10}{'no doc':>8}")
    for cls in ("stuck", "live", "saturated"):
        d = s["rescue"][cls]
        print(f"{cls:<10}{d['rows']:>6}{str(d['scored']):>9}{str(d['scored_and_kept_rule']):>11}"
              f"{str(d['broke_rule']):>12}{str(d['evidence_seen']):>10}{d['no_document']:>8}")

    for title, key in (("2b. which document, on the stuck groups that need one",
                        "rescue_by_variant_on_stuck"),
                       ("2c. the same, restricted to groups where BOTH documents existed",
                        "rescue_paired_on_stuck")):
        if not s.get(key):
            continue
        print(f"\n{title}")
        print(f"{'variant':<15}{'rows':>6}{'kept (shown)':>14}{'kept (all rows)':>17}"
              f"{'copied queries':>16}{'copied route':>14}{'searches':>10}")
        for v, d in sorted(s[key].items()):
            print(f"{v:<15}{d['rows']:>6}{str(d['scored_and_kept_rule']):>14}"
                  f"{str(d['kept_rule_over_all_rows']):>17}"
                  f"{str(d['copied_document_queries']):>16}{str(d['copied_whole_route']):>14}"
                  f"{str(d['searches_to_hit']):>10}")
    if False:
        print("\n2b. which document, on the stuck groups that need one")
        print(f"{'variant':<15}{'rows':>6}{'kept rule':>11}{'short ans':>11}{'long ans':>10}"
              f"{'numeric':>9}{'searches':>10}{'query overlap':>15}")
        for v, d in sorted(s["rescue_by_variant_on_stuck"].items()):
            print(f"{v:<15}{d['rows']:>6}{str(d['scored_and_kept_rule']):>11}"
                  f"{str(d['kept_rule_short_answer']):>11}{str(d['kept_rule_long_answer']):>10}"
                  f"{str(d['kept_rule_numeric_answer']):>9}{str(d['searches_to_hit']):>10}"
                  f"{str(d['query_overlap_with_siblings']):>15}")
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
    ap.add_argument("--flows", default=None,
                    help="the verified routes file, to measure how much the rescue row copied")
    ap.add_argument("--train-parquet", default=None,
                    help="the Search training data, to split the report into nq and hotpotqa")
    args = ap.parse_args()
    rows = load(args.dump_dir)
    if not rows:
        print(f"no records under {args.dump_dir}", file=sys.stderr)
        return 1
    flows = None
    if args.flows and os.path.exists(args.flows):
        blob = json.load(open(args.flows))
        rowsf = blob.get("flows", blob)
        flows = {fold(v["question"]): v.get("queries", [])
                 for v in rowsf.values() if isinstance(v, dict) and v.get("hit")}
        print(f"routes loaded: {len(flows)}")
    if args.train_parquet:
        hit, total = tag_sources(rows, args.train_parquet)
        print(f"dataset tagged for {hit}/{total} records")
    s = summarise(rows, flows=flows)
    report(s)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
