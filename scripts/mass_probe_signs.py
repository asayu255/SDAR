"""Push-up vs push-down, and the advantage scale, from a finished mass probe.

WHY POST-HOC. The probe writes one record per row (tokens, sum|pg|, sum pg,
teacher KL, the row weight). The measurement takes no optimizer step, so the
importance ratio is structurally 1 -- the probe's own actor/driver self-check is
the evidence -- and at ratio 1 verl's per-token loss is exactly ``-A``. So each
record carries its row's advantage:

    |A| per token   = pg_abs / tokens
    sign of A       = MINUS the sign of pg_signed   (pg_loss = -A * ratio)

Everything below is read off those records. Nothing here needs the probe to be
re-run, and nothing here can change what the probe measured.

TWO THINGS IT ADDS to the probe's own table.

1. The PG mass split by sign. |A| alone gives the same number to an update that
   mostly pushes up and one that mostly pushes down; every problem in this line
   of work has been the second kind (the ten-slot run's 12:1). Reported as mass,
   not as a row count, because a long row carries more of the loss.

2. The |A| scale. The format channel's fingerprint is a LARGE advantage on the
   few rows that differ in format (19.946 for a lone correctly formatted row in
   an otherwise identical group). If the largest |A| in the run is small, then a
   "mixed" group's policy gradient is ordinary spread, not that channel, and the
   two must not be reported with the same words.

TASK LABELS. The records carry the row weight but not the task. normalize_loss_
by_task gives each task a weight inversely proportional to its own token count,
so within one batch the three tasks have three distinct weights -- and the label
is then pinned by matching each cluster's summed mass against the per-task
pg_mass the probe itself wrote for that batch. The match is checked, not assumed.

    python scripts/mass_probe_signs.py ~/grad_probe/mass_klwctl_step25.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_batch_rows(terms_dir, tag):
    rows = []
    for f in sorted(glob.glob(os.path.join(terms_dir, f"terms.{tag}.rank*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def label_tasks(rows, batch_summary, pg_loss_coef):
    """Weight cluster -> task name, pinned by the probe's own per-task pg_mass."""
    by_w = defaultdict(list)
    for r in rows:
        if r["w"] > 0:
            by_w[round(r["w"], 12)].append(r)
    want = {t: task["totals"]["pg_mass"] for t, task in batch_summary["tasks"].items()}
    got = {w: sum(r["w"] * r["pg_abs"] for r in rs) * pg_loss_coef for w, rs in by_w.items()}
    labels, used = {}, set()
    for w, mass in sorted(got.items(), key=lambda kv: -kv[1]):
        best, err = None, None
        for t, m in want.items():
            if t in used:
                continue
            e = abs(mass - m) / max(m, 1e-12)
            if err is None or e < err:
                best, err = t, e
        labels[w] = (best, err)
        used.add(best)
    return by_w, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--pg-loss-coef", type=float, default=1.0)
    ap.add_argument("--train-groups", type=int, default=15,
                    help="groups per task in a real training step (the probe draws 7)")
    a = ap.parse_args()
    with open(a.payload) as f:
        p = json.load(f)
    terms_dir = a.payload + ".terms"
    batches = p["mass_batches"]

    per_task = defaultdict(lambda: {"rows_up": 0, "rows_down": 0, "rows_zero": 0,
                                    "mass_up": 0.0, "mass_down": 0.0,
                                    "tok_up": 0.0, "tok_down": 0.0, "tok_zero": 0.0})
    absA = defaultdict(list)
    worst_label_err = 0.0
    n_rows = 0
    for i, bs in enumerate(batches, 1):
        rows = load_batch_rows(terms_dir, f"b{i}")
        if not rows:
            print(f"  (batch {i}: no records on disk, skipped)")
            continue
        n_rows += len(rows)
        by_w, labels = label_tasks(rows, bs, a.pg_loss_coef)
        for w, rs in by_w.items():
            t, err = labels[w]
            worst_label_err = max(worst_label_err, err)
            c = per_task[t]
            for r in rs:
                mass = r["w"] * r["pg_abs"] * a.pg_loss_coef
                # pg_loss = -A, so pg_signed < 0 is A > 0: the update pushes this
                # row's tokens UP.
                if r["pg_signed"] < 0:
                    c["rows_up"] += 1; c["mass_up"] += mass; c["tok_up"] += r["tokens"]
                elif r["pg_signed"] > 0:
                    c["rows_down"] += 1; c["mass_down"] += mass; c["tok_down"] += r["tokens"]
                else:
                    c["rows_zero"] += 1; c["tok_zero"] += r["tokens"]
                if r["tokens"] > 0 and r["pg_abs"] > 0:
                    absA[t].append(r["pg_abs"] / r["tokens"])

    print(f"records {n_rows:,} over {len(batches)} batches")
    print(f"task labels pinned by per-task PG mass; worst relative error "
          f"{worst_label_err:.2e}  (a large value means the labelling is wrong)")
    print()
    print(f"{'task':<10}{'rows up':>9}{'rows down':>11}{'rows A=0':>10}"
          f"{'mass up':>11}{'mass down':>11}{'down/up':>9}")
    for t, c in per_task.items():
        ratio = (c["mass_down"] / c["mass_up"]) if c["mass_up"] else None
        print(f"{t:<10}{c['rows_up']:>9,}{c['rows_down']:>11,}{c['rows_zero']:>10,}"
              f"{c['mass_up']:>11.4g}{c['mass_down']:>11.4g}"
              f"{'-' if ratio is None else f'{ratio:.3g}':>9}")
    tot = {k: sum(c[k] for c in per_task.values())
           for k in ("rows_up", "rows_down", "rows_zero", "mass_up", "mass_down")}
    print(f"{'ALL':<10}{tot['rows_up']:>9,}{tot['rows_down']:>11,}{tot['rows_zero']:>10,}"
          f"{tot['mass_up']:>11.4g}{tot['mass_down']:>11.4g}"
          f"{(tot['mass_down'] / tot['mass_up']):>9.3g}")
    print()
    print("|A| per token on the rows that have any (the format channel's "
          "fingerprint is 19.946 / 14.087):")
    print(f"{'task':<10}{'rows':>8}{'max':>9}{'p99':>9}{'p90':>9}{'median':>9}{'>5':>7}{'>2':>7}")
    for t, v in absA.items():
        x = np.array(v)
        print(f"{t:<10}{len(x):>8,}{x.max():>9.3f}{np.percentile(x, 99):>9.3f}"
              f"{np.percentile(x, 90):>9.3f}{np.median(x):>9.3f}"
              f"{int((x > 5).sum()):>7}{int((x > 2).sum()):>7}")

    # Zero-PG steps: the probe draws 7 groups per task, training draws 15. A rate
    # read off probe batches is a rate at 7, and must be converted before it is
    # quoted as something that happens during training.
    print()
    n_probe, n_train = None, a.train_groups
    n_draw = int(p["mass"]["batches"])
    print(f"groups carrying ANY reward-driven gradient, pooled over batches -- and "
          f"what that implies for a {n_train}-group training step.")
    print("A rate read off probe batches is a rate at 7 groups, not at "
          f"{n_train}; quoting it as a training frequency overstates it.")
    print(f"{'task':<10}{'groups':>8}{'live':>6}{'mixed':>7}{'rate':>8}"
          f"{'P(none)@7':>11}{f'@{n_train}':>9}{'1 step in':>12}{'probe seen':>12}")
    for t, task in p["mass"]["tasks"].items():
        cls = task["classes"]
        g = sum(int(cls[c]["groups"]) for c in cls)
        live = int(cls["live"]["groups"])
        # A group whose format channel left a non-zero advantage also carries
        # reward-driven gradient, so it counts here with live.
        mixed = int(cls["stuck_mixed"]["groups"]) + int(cls["saturated_mixed"]["groups"])
        rate = (live + mixed) / g if g else 0.0
        p7, pn = (1 - rate) ** 7, (1 - rate) ** n_train
        seen = sum(1 for b in batches
                   if t in b["tasks"] and not b["tasks"][t]["totals"]["pg_mass"])
        n_probe = g
        each = "never" if pn < 1e-6 else f"{1 / pn:,.0f}"
        print(f"{t:<10}{g:>8}{live:>6}{mixed:>7}{rate:>8.3f}{p7:>11.4f}{pn:>9.4f}"
              f"{each:>12}{f'{seen}/{n_draw}':>12}")
    print(f"(pooled over {n_probe} groups per task. 'probe seen' is how many probe "
          f"batches actually had zero PG mass for that task,")
    print(f" which is the 7-group rate and is the number NOT to quote; "
          f"'1 step in' is the {n_train}-group one.)")


if __name__ == "__main__":
    main()
