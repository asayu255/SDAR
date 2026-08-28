#!/usr/bin/env python3
"""Pull the validation metrics out of a run log, and diff two of them.

    python scripts/val_scores.py /tmp/eval_a.log
    python scripts/val_scores.py /tmp/eval_a.log /tmp/eval_b.log

WHY THIS IS NOT A GREP. ray_trainer prints

    pprint(f"Initial validation metrics: {val_metrics}")

which is pprint of an f-STRING: the whole dict is one string, and pprint breaks
it at 80 columns wherever that falls, quoting each fragment:

    ("Initial validation metrics: {'val/alfworld/test_score': 2.95, "
     "'val/success_rate': "
     "0.38742732589884377, 'val/search/test_score': 0.357}")

So a key can end one line and its value begin the next, the break moves with
the dict's contents, and there can be more than one such block in a log. Three
successive shell extractions got this wrong in three different ways -- a
per-line grep that matched a key to an empty value, a fixed -A window that cut
the dict short, and an awk range that closed on the wrong line -- and each time
the symptom was a key silently missing, which reads as "this run did not score
it" rather than as "the reader is broken". Twice I was one step from concluding
that a run had evaluated different episodes.

Reassemble the fragments, then parse. The last value wins, so a log holding
more than one block reports the final one.
"""

import re
import sys

_RAY_PREFIX = re.compile(r"^\([^)]*\) ")
# One pprint fragment: optional leading paren, a quoted run, optional trailing
# paren. Both quote styles, because pprint picks whichever the text allows.
_FRAGMENT = re.compile(r'^\(?\s*(?P<q>["\'])(?P<body>.*)(?P=q)\)?,?$')
_PAIR = re.compile(r"'(val/[^']+)':\s*([-+0-9.eE]+)")


def _joined_blocks(text):
    """Every pprint block in the log, each rebuilt into one line."""
    blocks, current = [], None
    for raw in text.splitlines():
        line = _RAY_PREFIX.sub("", raw).rstrip()
        match = _FRAGMENT.match(line.strip())
        if match is None:
            if current is not None:
                blocks.append(current)
                current = None
            continue
        body = match.group("body")
        if current is None:
            if "Initial validation metrics" not in body:
                continue
            current = body
        else:
            current += body
    if current is not None:
        blocks.append(current)
    return blocks


def scores(path):
    with open(path, errors="replace") as handle:
        text = handle.read()
    found = {}
    for block in _joined_blocks(text):
        for key, value in _PAIR.findall(block):
            found[key] = float(value)   # a later block wins
    return found


def _print_one(path, table):
    print(f"=== {path} : {len(table)} metrics ===")
    for key in sorted(table):
        print(f"  {key:56s} {table[key]}")


def _print_diff(paths, tables):
    a, b = tables
    keys = sorted(set(a) | set(b))
    print(f"{'metric':56s}{'control':>14}{'candidate':>14}{'delta':>12}")
    for key in keys:
        left, right = a.get(key), b.get(key)
        if left is None or right is None:
            # A key present in one run and not the other is worth seeing on its
            # own line: it means the two runs did not score the same things.
            print(f"{key:56s}{'-' if left is None else left:>14}"
                  f"{'-' if right is None else right:>14}{'ONLY IN ONE':>12}")
            continue
        print(f"{key:56s}{left:>14.6f}{right:>14.6f}{right - left:>+12.6f}")
    missing = [k for k in keys if (k in a) != (k in b)]
    if missing:
        print(f"\n  {len(missing)} metric(s) in only one run: {missing}")
    else:
        print(f"\n  both runs scored the same {len(keys)} metrics")


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not 1 <= len(paths) <= 2:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    tables = [scores(p) for p in paths]
    for path, table in zip(paths, tables):
        if not table:
            print(f"=== {path} : NO validation metrics found ===")
    if len(paths) == 1:
        _print_one(paths[0], tables[0])
    else:
        _print_diff(paths, tables)


if __name__ == "__main__":
    main()
