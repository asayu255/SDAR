#!/usr/bin/env python3
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
"""Read the token tables an arm dumps beside its run, and say what it acted on.

    python3 scripts/sign_token_scan.py
    python3 scripts/sign_token_scan.py --step 6 --scope alfworld
    python3 scripts/sign_token_scan.py --dir ~/sign_tokens/<run> --top 40

The scalar metrics in wandb say how CONCENTRATED a weighting is; these files say
on WHAT, and only the first fits in a wandb column. ``trainer.sign_token_dump_dir``
is where the second goes -- one jsonl per table per collected step, written by
``_dump_sign_token_report``:

    sign_tokens_step000006.jsonl        scope x state
    sign_pair_tokens_step000006.jsonl   dst x src x class
    sign_events_step000006.jsonl        single positions, with decoded context
    sign_pair_events_step000006.jsonl   the same, per ordered pair

``token`` is already text: the worker runs ``convert_ids_to_tokens`` before the
rows leave it, so no tokenizer is needed here. It is deliberately NOT ``decode``
-- the leading space marker is what distinguishes two vocabulary entries that
print the same, and this tool prints tokens with ``repr`` for the same reason.

Three things this does that reading a jsonl does not.

**It puts the count ranking and the mass ranking side by side.** They are
separate ``ranked_by`` series in one file and they answer different questions:
the count ranking is what the arm TOUCHED, the mass ranking is what it MOVED.
On the runs this was written for those two disagree by a lot -- the top 64 by
nats carry ~74% of the effect while the top 64 by occurrence are ~24% of the
positions -- and a reader who sorts the file once sees only one of them.

**It shows which tokens two sources actually share at one destination.** The
run logs ``sign_weight/pair/token_overlap/{class}/{a}__and__{b}__on__{dst}`` as
a weighted Jaccard, which says how much the two off-task teachers overlap but
not where. The whole claim the cross-teacher arms rest on is that the sources
carry different information, so the intersection and the two differences are
worth naming rather than summarising: a 0.4 overlap made of function words is a
different finding from a 0.4 overlap made of tool syntax.

**It computes turnover from the files rather than waiting for the metric.**
``turnover`` inside the run compares the last two COLLECTED steps, so on a
stride of 5 it does not exist until the second collection -- step 11 for a run
whose first collection landed on a step with no weight. The dumps are already on
disk, so any two of them can be compared as soon as they exist.

Reads only what is asked for: one step's files, plus one earlier step when
turnover is available.
"""

import argparse
import glob
import json
import os

STEM = "sign_tokens_step"
# The four the arm writes. Absent files are skipped rather than reported: which
# tables a run produces depends on token_stats/event_dump, and a missing one is
# a configuration fact, not an error.
PAIR_CLASSES = ("agree", "conflict", "blindspot")
ACTED_STATES = ("agree_pos", "agree_neg", "conflict_on_pos", "conflict_on_neg")
TASKS = ("alfworld", "search", "webshop")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collected_steps(dump_dir):
    """Every step that has a token table, ascending."""
    out = []
    for p in glob.glob(os.path.join(dump_dir, f"{STEM}*.jsonl")):
        try:
            out.append(int(os.path.basename(p)[len(STEM):-len(".jsonl")]))
        except ValueError:
            continue
    return sorted(out)


def _fmt(v, width, spec):
    if v is None or v == "":
        return " " * width
    try:
        return f"{spec.format(v):>{width}}"
    except (ValueError, TypeError):
        return f"{str(v):>{width}}"


def table(title, rows, cols, top):
    if not rows:
        return
    print(f"\n  {title}")
    print("  " + "".join(f"{c[0]:>{c[1]}}" for c in cols))
    print("  " + "-" * sum(c[1] for c in cols))
    for r in rows[:top]:
        print("  " + "".join(_fmt(r.get(c[2]), c[1], c[3]) for c in cols))


def rank(rows, **eq):
    """Rows matching every field in ``eq``, in the order the run ranked them."""
    sel = [r for r in rows if all(r.get(k) == v for k, v in eq.items())]
    return sorted(sel, key=lambda r: r.get("rank", 1 << 30))


# --------------------------------------------------------------------------- #
# what the arm acted on
# --------------------------------------------------------------------------- #
def report_tokens(rows, scope, top):
    scoped = [r for r in rows if r.get("scope") == scope]
    if not scoped:
        print(f"\n[tokens] no rows for scope {scope!r}. "
              f"present: {sorted({r.get('scope') for r in rows})}")
        return

    cols = [("token", 26, "token", "{!r}"), ("id", 9, "token_id", "{}"),
            ("count", 10, "count", "{}"), ("mass", 13, "mass", "{:.5g}"),
            ("net", 14, "effect_net", "{:+.5g}"), ("gross", 13, "effect_gross", "{:.5g}")]

    print(f"\n=== tokens  scope={scope} ===")
    # The state-pooled ranking is the one the headline concentration number is
    # computed over, so it goes first.
    for ranked_by in ("abs_effect", "mass", "count"):
        table(f"all states / ranked by {ranked_by}",
              rank(scoped, state="__any__", ranked_by=ranked_by), cols, top)

    for state in ACTED_STATES:
        # Both series, adjacent: see the module docstring on why one is not
        # enough. A state with no rows simply does not print.
        for ranked_by in ("mass", "count"):
            table(f"{state} / ranked by {ranked_by}",
                  rank(scoped, state=state, ranked_by=ranked_by), cols, top)


# --------------------------------------------------------------------------- #
# whether the two sources are naming the same tokens
# --------------------------------------------------------------------------- #
def report_pairs(rows, top, ranked_by="mass"):
    if not rows:
        return
    cols = [("token", 26, "token", "{!r}"), ("id", 9, "token_id", "{}"),
            ("count", 10, "count", "{}"), ("mass", 13, "mass", "{:.5g}"),
            ("net", 14, "effect_net", "{:+.5g}")]

    print(f"\n=== pair tokens  (ranked by {ranked_by}) ===")
    tops = {}
    for cls in PAIR_CLASSES:
        for dst in TASKS:
            for src in TASKS:
                if src == dst:
                    continue
                sel = rank(rows, cls=cls, dst=dst, src=src, ranked_by=ranked_by)
                if not sel:
                    continue
                tops[(cls, dst, src)] = [r["token"] for r in sel[:top]]
                table(f"{cls}: {src} -> {dst}", sel, cols, top)

    print(f"\n=== which tokens the two sources share  (top {top}, by {ranked_by}) ===")
    print("  the visible counterpart of sign_weight/pair/token_overlap/*")
    for cls in PAIR_CLASSES:
        for dst in TASKS:
            srcs = [s for s in TASKS if s != dst]
            a, b = tops.get((cls, dst, srcs[0])), tops.get((cls, dst, srcs[1]))
            if not a or not b:
                continue
            sa, sb = set(a), set(b)
            print(f"\n  {cls} on {dst}: {srcs[0]} & {srcs[1]}  "
                  f"shared {len(sa & sb)}/{top}")
            print(f"    shared        {sorted(sa & sb)[:12]}")
            print(f"    {srcs[0]:<9s} only {sorted(sa - sb)[:8]}")
            print(f"    {srcs[1]:<9s} only {sorted(sb - sa)[:8]}")


# --------------------------------------------------------------------------- #
# the positions themselves
# --------------------------------------------------------------------------- #
def report_events(rows, limit, name):
    if not rows:
        return
    # Whichever effect column this table carries; the event tables differ
    # between the sign and cross-teacher arms and both are worth ranking by.
    key = next((k for k in ("effect_net", "extra_logit_push", "effect_gross")
                if k in rows[0]), None)
    if key is None:
        return
    print(f"\n=== {name}  ({len(rows)} rows, {min(limit, len(rows))} largest by |{key}|) ===")
    for e in sorted(rows, key=lambda r: -abs(r.get(key) or 0))[:limit]:
        label = e.get("state") or e.get("cls") or e.get("direction_class") or "?"
        print(f"\n  [{label}] token={e.get('token')!r}  {key}={e.get(key):+.5g}")
        ctx = (e.get("context") or "").replace("\n", "\\n")
        if ctx:
            print(f"    {ctx[:400]}")


# --------------------------------------------------------------------------- #
# is it the same vocabulary each time
# --------------------------------------------------------------------------- #
def report_turnover(dump_dir, steps, step, scope, top):
    older = [s for s in steps if s < step]
    if not older:
        print(f"\n=== turnover ===\n  step {step} is the first collected step; "
              f"nothing to compare against yet.")
        return
    prev = older[-1]
    cur_rows = load(os.path.join(dump_dir, f"{STEM}{step:06d}.jsonl"))
    prev_rows = load(os.path.join(dump_dir, f"{STEM}{prev:06d}.jsonl"))

    print(f"\n=== turnover  step {prev} -> {step}  scope={scope} ===")
    for ranked_by in ("abs_effect", "mass", "count"):
        cur = [r["token"] for r in rank(
            [r for r in cur_rows if r.get("scope") == scope],
            state="__any__", ranked_by=ranked_by)][:top]
        old = [r["token"] for r in rank(
            [r for r in prev_rows if r.get("scope") == scope],
            state="__any__", ranked_by=ranked_by)][:top]
        if not cur or not old:
            continue
        sa, sb = set(cur), set(old)
        # Set Jaccard, unlike the run's weighted one: here the question is
        # whether the RANKING is the same list, and a rank cut is already a set.
        print(f"\n  ranked by {ranked_by}:  jaccard {len(sa & sb) / len(sa | sb):.3f}   "
              f"kept {len(sa & sb)}/{len(sa)}")
        print(f"    entered {sorted(sa - sb)[:10]}")
        print(f"    left    {sorted(sb - sa)[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None,
                    help="trainer.sign_token_dump_dir; defaults to the only "
                         "directory under ~/sign_tokens when there is one")
    ap.add_argument("--step", type=int, default=None, help="default: newest collected")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--scope", default="__pooled__",
                    help="__pooled__ | <task> | role:<role>")
    ap.add_argument("--events", type=int, default=10, help="0 to skip")
    args = ap.parse_args()

    dump_dir = args.dir
    if dump_dir is None:
        root = os.path.expanduser("~/sign_tokens")
        cand = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
        if len(cand) != 1:
            raise SystemExit(
                f"pass --dir: {len(cand)} run directories under {root}"
                + ("\n  " + "\n  ".join(cand) if cand else ""))
        dump_dir = cand[0]
    dump_dir = os.path.expanduser(dump_dir)

    steps = collected_steps(dump_dir)
    if not steps:
        raise SystemExit(f"no {STEM}*.jsonl under {dump_dir} -- check "
                         "trainer.sign_token_dump_dir, and note the tables are "
                         "written only on token_stats.every steps")
    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        raise SystemExit(f"step {step} not collected; have {steps}")

    print(f"=== {dump_dir}")
    print(f"=== step {step} of collected {steps}")

    tag = f"step{step:06d}"
    report_tokens(load(os.path.join(dump_dir, f"{STEM}{step:06d}.jsonl")),
                  args.scope, args.top)
    report_pairs(load(os.path.join(dump_dir, f"sign_pair_tokens_{tag}.jsonl")), args.top)
    if args.events:
        report_events(load(os.path.join(dump_dir, f"sign_events_{tag}.jsonl")),
                      args.events, "events")
        report_events(load(os.path.join(dump_dir, f"sign_pair_events_{tag}.jsonl")),
                      args.events, "pair events")
    report_turnover(dump_dir, steps, step, args.scope, args.top)


if __name__ == "__main__":
    main()
