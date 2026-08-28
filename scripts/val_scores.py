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
    """Every pprint block in the log, each rebuilt into one string.

    INTERLEAVING IS NOT A TERMINATOR. Ray forwards several actors' stdout into
    one stream, so an unrelated line can land between two fragments of the same
    pprint block. A reader that treats "not a fragment" as the end of the block
    stops at the first such line and reports nothing at all -- which is what the
    first version of this did on a real log while passing on a synthetic one,
    because the synthetic one had nothing to interleave.

    So: start at "Initial validation metrics", keep every fragment after it,
    skip anything that is not one, and stop when a fragment closes the dict.
    """
    blocks, current, started = [], [], False
    for raw in text.splitlines():
        line = _RAY_PREFIX.sub("", raw).strip()
        match = _FRAGMENT.match(line)
        if match is None:
            continue
        body = match.group("body")
        if not started:
            if "Initial validation metrics" not in body:
                continue
            started, current = True, [body]
        else:
            current.append(body)
        if body.rstrip().endswith(("}", '}")', "}'")):
            blocks.append("".join(current))
            started, current = False, []
    if current:
        blocks.append("".join(current))
    return blocks


def _salvage(text, span=400):
    """Last resort: join the region after the marker and parse whatever is there.

    The block reader above understands the shape pprint produces TODAY. This
    understands only that a key and its value may be separated by a line break
    and some quoting -- so it survives a shape neither of us has seen, which is
    the failure mode that has cost this arm the most: a reader that returns
    nothing says exactly what a run that scored nothing says.

    Bounded to a few hundred lines after the marker so it cannot fabricate a
    pair by joining two unrelated parts of a long log.
    """
    lines = [_RAY_PREFIX.sub("", l).strip() for l in text.splitlines()]
    starts = [i for i, l in enumerate(lines) if "Initial validation metrics" in l]
    found = {}
    for start in starts:
        window = lines[start:start + span]
        # Drop the quoting that pprint puts at the seams, then join: a value on
        # the line after its key becomes adjacent to it.
        joined = " ".join(l.strip('()," \'') for l in window)
        for key, value in _PAIR.findall(joined):
            found[key] = float(value)
    return found


def scores(path):
    with open(path, errors="replace") as handle:
        text = handle.read()
    found = {}
    for block in _joined_blocks(text):
        for key, value in _PAIR.findall(block):
            found[key] = float(value)   # a later block wins
    return found or _salvage(text)


def explain(path, limit=6):
    """When nothing parsed, show what the file actually looks like there.

    A reader that returns an empty table says "this run did not score" and a
    broken one says exactly the same thing. This turns the second into
    something a person can see in one step instead of a round trip.
    """
    with open(path, errors="replace") as handle:
        lines = handle.read().splitlines()
    hits = [i for i, l in enumerate(lines) if "Initial validation metrics" in l]
    if not hits:
        print(f"  no line contains 'Initial validation metrics' -- the run may not have scored")
        return
    print(f"  {len(hits)} line(s) mention it; the first block, as this reader sees it:")
    for offset in range(limit):
        index = hits[0] + offset
        if index >= len(lines):
            break
        raw = lines[index]
        stripped = _RAY_PREFIX.sub("", raw).strip()
        kind = "FRAGMENT" if _FRAGMENT.match(stripped) else "not a fragment"
        print(f"    [{kind:14s}] {stripped[:110]}")


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
            explain(path)
    if len(paths) == 1:
        _print_one(paths[0], tables[0])
    else:
        _print_diff(paths, tables)


if __name__ == "__main__":
    main()
