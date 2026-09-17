"""What the group records say: (a), ProGPO's gate and its coverage, on the same groups.

    python scripts/report_progress_groups.py ~/grad_probe/<probe>.json.groups
    python scripts/report_progress_groups.py <default_local_dir>/progress_rank_groups

The records are the ones the trainer writes each step (verl/trainer/ppo/
progress_rank.py, GROUP RECORDS); every number below is read off them. One block
per task:

  groups       how many, the share stuck by the environment's reward (q_fail), and
               the share ProGPO's own gate opens on (every base score < 1e-3)
  invalid      the share of all turns marked invalid: before the policy writes its
               own <think> tags it is ~1, and every stuck group scores alike
  (a)          the stuck groups' verdicts; of the fired ones, how many the format
               penalty already moves, and the share of (a)'s mass that lands there
  scale        over the steps (a) had a target: how often the cap bound, and how
               far above the cap the target sat (median c_uncapped / c_cap)
  coverage     ProGPO's P = (D - 1) / T on the same stuck groups: where it would
               fire, where both fire, how the two order the same rollouts
               (Spearman, average ranks), and where ProGPO itself would fire
               (its gate open AND P varying)
  winners      the share (a)'s count puts at k = 0, and coverage at D = 1
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from verl.trainer.ppo.progress_rank import PROGPO_TAU_P, spearman  # noqa: E402
from verl.trainer.ppo.term_mass import SCORE_SPREAD_EPS  # noqa: E402


def load(paths):
    recs = []
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
        for f in files:
            with open(f) as fh:
                recs += [json.loads(line) for line in fh if line.strip()]
    return recs


def pct(num, den):
    return "-" if not den else f"{100.0 * num / den:.1f}% ({num}/{den})"


def coverage_p(rec):
    ps = []
    for d, t in zip(rec.get("coverage_d") or [], rec.get("turns") or []):
        if d is None or not t:
            return None
        ps.append((float(d) - 1.0) / float(t))
    return ps if len(ps) >= 2 else None


def report(recs):
    by_task = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        stuck = [r for r in rs if r["status"] == "stuck"]
        print(f"== {task}: {len(rs)} groups over {len({(r.get('step'), r.get('batch')) for r in rs})} step(s)")
        gated = [r for r in rs if r.get("progpo_gate") is not None]
        print(f"  groups    stuck (q_fail) {pct(len(stuck), len(rs))}"
              f"   ProGPO gate open {pct(sum(bool(r['progpo_gate']) for r in gated), len(gated))}")
        turns = sum(sum(r["turns"]) for r in rs if r.get("invalid_turns") is not None)
        bad = sum(sum(r["invalid_turns"]) for r in rs if r.get("invalid_turns") is not None)
        print(f"  invalid   {pct(bad, turns)} of turns")

        verdicts = Counter(r["verdict"] for r in stuck)
        fired = [r for r in stuck if r["verdict"] == "fired"]
        mixed = [r for r in fired if (r.get("score_spread") or 0.0) > SCORE_SPREAD_EPS]
        mass = sum(r.get("injected_abs_mass") or 0.0 for r in fired)
        mass_mixed = sum(r.get("injected_abs_mass") or 0.0 for r in mixed)
        print(f"  (a)       verdicts {dict(verdicts)}   fired and format-mixed {pct(len(mixed), len(fired))}"
              f"   injected mass on those {('-' if not mass else f'{100.0 * mass_mixed / mass:.1f}%')}")

        steps = {}
        for r in rs:
            steps[(r.get("step"), r.get("batch"))] = r
        targeted = [r for r in steps.values() if r.get("c_uncapped") is not None]
        ratios = [r["c_uncapped"] / r["c_cap"] for r in targeted if r.get("c_cap")]
        print(f"  scale     steps with a target {len(targeted)}/{len(steps)}"
              f"   capped {pct(sum(bool(r['capped']) for r in targeted), len(targeted))}"
              f"   median c_uncapped/c_cap {('-' if not ratios else f'{np.median(ratios):.3g}')}")

        known = [(r, coverage_p(r)) for r in stuck]
        known = [(r, ps) for r, ps in known if ps is not None]
        cov_fired = [(r, ps) for r, ps in known if float(np.std(ps)) >= PROGPO_TAU_P]
        both = [(r, ps) for r, ps in cov_fired if r["verdict"] == "fired"]
        rhos = [x for x in (spearman(r["k"], ps) for r, ps in both) if x is not None]
        progpo = [r for r, _ in cov_fired if r.get("progpo_gate")]
        print(f"  coverage  stuck groups with D {len(known)}   fires {pct(len(cov_fired), len(known))}"
              f"   both fire {len(both)} (a) alone {len(fired) - len(both)}"
              f"   Spearman vs k {('-' if not rhos else f'{np.mean(rhos):.2f} over {len(rhos)}')}"
              f"   ProGPO would fire {len(progpo)}")

        won_k = [(k, K) for r in rs for k, K, w in zip(r["k"], r["K"], r["won"]) if w and K > 0]
        won_d = [d for r in rs for d, w in zip(r.get("coverage_d") or [], r["won"]) if w and d is not None]
        print(f"  winners   k = 0 {pct(sum(1 for k, _ in won_k if k == 0), len(won_k))}"
              f"   D = 1 {pct(sum(1 for d in won_d if d <= 1), len(won_d))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="record directories or .jsonl files")
    report(load(ap.parse_args().paths))


if __name__ == "__main__":
    main()
