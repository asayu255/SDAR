#!/usr/bin/env python3
"""Compare two arms' validation passes INSTANCE BY INSTANCE.

The unpaired comparison two `episode/*_success_rate` numbers give up most of the
power there is. At 126 instances a task the 95% interval on a difference of
proportions is about +-11 points, so a five-point gap -- the largest this
experiment produced -- is indistinguishable from zero and so is its opposite.
Almost all of that width is between-instance variance: some validation problems
are hard for every arm and some are easy for every arm, and an unpaired test
pays for that spread twice.

`RayPPOTrainer._dump_val_instances` writes what removes it. `val_index` is the
row's position in the validation pass, the loader is built with `shuffle=False`
over a file whose path, size and seed are pinned in the intent lock, so index i
is THE SAME PROBLEM in every arm and at every step. Differencing per index
cancels the instance difficulty and leaves the arm effect.

    python scripts/val_paired.py \\
        --a ~/val_instances/opd_grpo_multitask_..._control_qwen3_1.7b_xt1 \\
        --b ~/val_instances/opd_grpo_multitask_..._qwen3_1.7b_xt1 \\
        --step 300

WHAT THIS FILE CAN AND CANNOT ANSWER. The dump carries the episode `score` and
deliberately does not derive success from it -- that is `score == 1.0` on
alfworld and search but a separate `won` flag on webshop, and baking the guess
into the log would make the file only as good as the guess. So webshop's Acc
column cannot be paired from here; its Score column can, and that is the column
with the largest gap.
"""

import argparse
import collections
import json
import math
import os
import statistics
import sys


def _load(path, step):
    """`{val_index: row}` for one arm's pass, from a file or its directory."""
    if os.path.isdir(path):
        path = os.path.join(path, f"val_step{step}.jsonl")
    if not os.path.exists(path):
        sys.exit(f"no such validation dump: {path}")
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows[int(row["val_index"])] = row
    if not rows:
        sys.exit(f"{path} is empty")
    return path, rows


def _normal_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def _paired(diffs):
    """Mean, standard error and two-sided p of a paired difference."""
    n = len(diffs)
    if n < 2:
        return None
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        # Every pair moved identically -- including "not at all", which is the
        # common case on a task both arms solve the same way. A zero standard
        # error is not infinite confidence, it is no information about spread,
        # so say so rather than dividing by it.
        return mean, 0.0, float("nan"), (mean, mean)
    se = sd / math.sqrt(n)
    z = mean / se
    return mean, se, _normal_two_sided(z), (mean - 1.96 * se, mean + 1.96 * se)


def _unpaired(a, b):
    """The same difference scored WITHOUT the pairing, for the contrast."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    mean = statistics.fmean(b) - statistics.fmean(a)
    if se == 0.0:
        return mean, 0.0, float("nan"), (mean, mean)
    return mean, se, _normal_two_sided(mean / se), (mean - 1.96 * se, mean + 1.96 * se)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline arm: val_instance_log_dir or a val_step*.jsonl")
    ap.add_argument("--b", required=True, help="treatment arm, same form")
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--success", action="store_true",
                    help="score the SUCCESS indicator (score == 1.0) instead of the raw score. "
                         "Right for alfworld and search; wrong for webshop, whose success is a "
                         "separate flag the dump does not carry.")
    args = ap.parse_args(argv)

    path_a, rows_a = _load(args.a, args.step)
    path_b, rows_b = _load(args.b, args.step)
    print(f"A (baseline) : {path_a}\n               {rows_a[min(rows_a)]['experiment']}  n={len(rows_a)}")
    print(f"B (treatment): {path_b}\n               {rows_b[min(rows_b)]['experiment']}  n={len(rows_b)}")

    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        sys.exit("the two passes share no val_index -- nothing to pair")
    dropped = (len(rows_a) - len(shared)) + (len(rows_b) - len(shared))
    if dropped:
        print(f"[warn] {dropped} row(s) present in only one pass; paired on the {len(shared)} shared indices")

    # THE CHECK THAT MAKES THE PAIRING MEAN ANYTHING. If index i is a different
    # problem in the two passes -- a different val file, a different seed, a
    # different task order -- then differencing per index is differencing noise
    # against noise and every number below is invented. Refuse rather than
    # report it.
    mismatched = [
        i for i in shared
        if (rows_a[i].get("task"), rows_a[i].get("data_source"))
        != (rows_b[i].get("task"), rows_b[i].get("data_source"))
    ]
    if mismatched:
        i = mismatched[0]
        sys.exit(
            f"val_index {i} is a different instance in the two passes "
            f"({rows_a[i].get('task')}/{rows_a[i].get('data_source')} vs "
            f"{rows_b[i].get('task')}/{rows_b[i].get('data_source')}); "
            f"{len(mismatched)} of {len(shared)} indices disagree. The pairing key is only "
            "stable across runs that share the validation file, its size and the seed -- all "
            "three are pinned in the intent lock, so a disagreement here means the two passes "
            "are not comparable at all, paired or otherwise."
        )

    def value(row):
        return 1.0 if (args.success and row["score"] == 1.0) else (
            0.0 if args.success else float(row["score"])
        )

    by_task = collections.defaultdict(list)
    for i in shared:
        by_task[rows_a[i].get("task") or "unknown"].append(i)
    by_task["(pooled)"] = shared

    label = "success (score==1)" if args.success else "score"
    print(f"\nstep {args.step}, scoring {label}. B - A, so a NEGATIVE mean is the treatment doing worse.\n")
    head = f"{'task':<12} {'n':>4} {'mean A':>8} {'mean B':>8} {'paired Δ':>9} {'SE':>7} {'p':>7}  {'95% CI':>18}   {'unpaired SE':>11} {'gain':>5}"
    print(head)
    print("-" * len(head))
    for task, idxs in by_task.items():
        a = [value(rows_a[i]) for i in idxs]
        b = [value(rows_b[i]) for i in idxs]
        pr = _paired([y - x for x, y in zip(a, b)])
        up = _unpaired(a, b)
        if pr is None or up is None:
            continue
        mean, se, p, (lo, hi) = pr
        _, use, _, _ = up
        gain = (use / se) if se > 0 else float("nan")
        print(
            f"{task:<12} {len(idxs):>4} {statistics.fmean(a):>8.3f} {statistics.fmean(b):>8.3f} "
            f"{mean:>+9.4f} {se:>7.4f} {p:>7.3f}  [{lo:+.4f}, {hi:+.4f}]   "
            f"{use:>11.4f} {gain:>4.1f}x"
        )
    print(
        "\n'gain' is how much narrower the paired standard error is than the unpaired one. "
        "\nAt 1.0x the instances carry no shared difficulty and the pairing bought nothing; "
        "\nwell above 1.0x, the unpaired table was mostly measuring which problems came up."
    )


if __name__ == "__main__":
    main()
