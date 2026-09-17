"""Read a mass-probe payload and print the per-task, per-class table.

WHAT IT ANSWERS. For one checkpoint of the 3-task run: how much of each task's
update is reward-driven (policy gradient) and how much is teacher-driven
(top-k KL), split by group class -- live, and stuck/saturated split by whether
the format channel left any non-zero advantage in the group.

WHY A SEPARATE SCRIPT. The trainer prints a running table while the probe runs;
this one adds what a running table cannot have: the spread over batches, so a
share that comes from one lucky batch is not read as a level. Both read the same
payload, so the numbers agree by construction.

    python scripts/report_mass_probe.py ~/grad_probe/mass_klwctl_step25.json
"""
import argparse
import json
import statistics
import sys

CLASSES = ("live", "stuck_uniform", "stuck_mixed", "saturated_uniform", "saturated_mixed")


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def g3(x):
    return "-" if x is None else f"{x:.3g}"


def cls_tokens(cell, task_total_tokens):
    """Absolute response tokens of a class, however the payload stored them."""
    if cell.get("tokens") is not None:
        return float(cell["tokens"])
    share = cell.get("token_share")
    return float(share) * task_total_tokens if share is not None else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--per-batch", action="store_true",
                    help="also print one line per batch per task")
    a = ap.parse_args()
    with open(a.payload) as f:
        p = json.load(f)
    if p.get("mode") != "mass":
        sys.exit(f"not a mass probe payload: mode={p.get('mode')!r}")
    s = p.get("mass")
    if not s:
        sys.exit("payload has no mass summary")

    print(f"checkpoint   {p.get('checkpoint')}")
    print(f"batches      {s['batches']}  (n={p.get('rollout_n')}, T={p.get('temperature')})")
    print(f"self-check   pg actor/driver {s.get('pg_mass_actor_vs_driver')}"
          f"   rows missing {s.get('rows_missing_from_actor')}"
          f"   padding skipped {s.get('padding_rows_skipped')}")

    batches = p.get("mass_batches") or []
    for t, task in s["tasks"].items():
        tot = task["totals"]
        print()
        print(f"=== {t}   groups {int(tot['groups'])}  rows {int(tot['rows'])}"
              f"  response tokens {int(tot['tokens']):,}"
              f"  ({tot['tokens'] / tot['rows']:.0f}/row)"
              f"   task PG/KL {g3(task['pg_over_kl'])}")
        print(f"    {'class':<19}{'groups':>7}{'rows':>6}{'tokens':>10}{'tok/row':>9}"
              f"{'tok%':>8}{'PG%':>8}{'KL%':>8}{'PG/KL':>9}{'down/up':>9}{'max|A|':>8}")
        for c in CLASSES:
            v = task["classes"].get(c)
            if not v:
                continue
            tk = cls_tokens(v, tot["tokens"])
            rows = int(v["rows"])
            # down/up and max|A| are absent from payloads written before the sign
            # split; scripts/mass_probe_signs.py recovers them from the records.
            print(f"    {c:<19}{int(v['groups']):>7}{rows:>6}{int(tk):>10,}"
                  f"{(tk / rows if rows else 0):>9.0f}"
                  f"{pct(v['token_share']):>8}{pct(v['pg_share']):>8}"
                  f"{pct(v['kl_share']):>8}{g3(v['pg_over_kl']):>9}"
                  f"{g3(v.get('pg_down_over_up')):>9}{g3(v.get('abs_a_max')):>8}")
        # The spread matters more than the level: 10 batches of 7 groups is a
        # small sample, and one batch with no live group moves a share a long way.
        if batches:
            for label, get in (
                ("stuck tok%", lambda tk: (tk["classes"]["stuck_uniform"]["token_share"] or 0)
                    + (tk["classes"]["stuck_mixed"]["token_share"] or 0)),
                ("stuck PG%", lambda tk: (tk["classes"]["stuck_uniform"]["pg_share"] or 0)
                    + (tk["classes"]["stuck_mixed"]["pg_share"] or 0)),
                ("task PG/KL", lambda tk: tk["pg_over_kl"]),
            ):
                vals = [get(b["tasks"][t]) for b in batches if t in b["tasks"]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                fmt = pct if label.endswith("%") else g3
                print(f"    per batch  {label:<11} min {fmt(min(vals))}  "
                      f"med {fmt(statistics.median(vals))}  max {fmt(max(vals))}  "
                      f"sd {fmt(sd)}   (n={len(vals)})")
        if a.per_batch:
            for i, b in enumerate(batches, 1):
                bt = b["tasks"].get(t)
                if not bt:
                    continue
                gs = {c: int(bt["classes"][c]["groups"]) for c in CLASSES}
                print(f"    batch {i:>2}  live {gs['live']}  stuck {gs['stuck_uniform']}u/"
                      f"{gs['stuck_mixed']}m  sat {gs['saturated_uniform']}u/"
                      f"{gs['saturated_mixed']}m   PG/KL {g3(bt['pg_over_kl'])}")


if __name__ == "__main__":
    main()
