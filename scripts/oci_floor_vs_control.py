#!/usr/bin/env python3
"""Arm B' (oci_floor) against the klw control, read from wandb.

WHY A SCRIPT AND NOT A ONE-LINER. Two readings of this comparison were already
wrong for reasons a one-liner invites:

  * The run named `control` in wandb is NOT named after its arm -- its
    experiment_name is opd_grpo_multitask_cross_teacher_klw_control_qwen3_1.7b_xt1
    while its display name is just "control", and sibling runs called
    train0/train1/train2 are the TREATMENT arm. Selecting on the display name
    silently returns no control at all, which is how the first attempt reported
    "control MISSING".
  * ``Api.run.history(keys=[...])`` inner-joins the keys and returns nothing when
    any one of them is absent from the run -- and the control predates
    groups/degen_saturated_high, so asking for it emptied the whole fetch.
    scan_history does not join.

And the bin labels were then computed separately from the bin slices, so a patched
range printed "61-48". Labels and slices come from the same object here.

THE TWO ARMS DID NOT SEE THE SAME GAMES. An earlier version of this script said the
control ran on 2 GPUs and B' on 3; wandb says both ran on 3 (control on yamabuki,
B' on fuji), and the host is what matters. collect_game_files builds the game list
from `list(os.walk(...))`, and os.walk order is a property of the filesystem: the
140 valid_seen games and 3553 train games are the same set on both hosts (identical
sorted-list md5) in a different order (different walk-order md5). The seed fixes
a POSITION, not a game: validation worker i shuffles its own copy of that list with
RandomState(1001 + i) and plays element 0, so workers 0/1/2 take positions 1/24/108
on every host and get whatever game the host's order put there (126 draws with
replacement, 90 distinct games). Simulated on each host's real order this
reproduces both observed per-type mixes exactly and neither fits the other; the
training draw differs the same way. Nothing here can pair games across the runs: the per-instance dumps carry a
fresh traj_uid per run and no game id.

Usage: python scripts/oci_floor_vs_control.py [--bins 15] [--key ...]
"""
import argparse
import math
import os

DEFAULT_PROJECT = "verl_agent_opd_grpo_cross_teacher_klw_xt1"
ARMS = {
    "opd_grpo_multitask_cross_teacher_klw_control_qwen3_1.7b_xt1": "control",
    "opd_grpo_multitask_oci_floor_qwen3_1.7b_xt1": "B-prime",
}


def series(project, keys):
    """{arm: {step: {key: value}}}, taking the longest run per arm."""
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    out = {}
    for run in api.runs(f"{entity}/{project}"):
        cfg = run.config.get("trainer") or {}
        arm = ARMS.get(cfg.get("experiment_name") if isinstance(cfg, dict) else None)
        if arm is None:
            continue
        pts = {}
        for row in run.scan_history(keys=["training/global_step", *keys],
                                    page_size=2000):
            step = row.get("training/global_step")
            if step is None:
                continue
            pts[int(step)] = {k: row.get(k) for k in keys}
        if pts and (arm not in out or len(pts) > len(out[arm][1])):
            out[arm] = (run.id, pts)
    return out


def mean_sd_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd, sd / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--key", default="episode/alfworld_success_rate")
    ap.add_argument("--bins", type=int, default=15, help="steps per bin")
    args = ap.parse_args()

    dose = "oci_floor/groups_floored"
    data = series(args.project, [args.key, dose])
    for arm, (rid, pts) in sorted(data.items()):
        print(f"{arm:9s} id={rid} points={len(pts)} maxstep={max(pts)}")
    if "control" not in data or "B-prime" not in data:
        print("\nboth arms are needed; see the docstring on why a run may go missing")
        return
    ctl, arm = data["control"][1], data["B-prime"][1]
    last = max(arm)

    print(f"\n{args.key}, bins of {args.bins} steps")
    print(f"{'steps':>9}  {'control':>8}  {'B-prime':>8}  {'diff':>7}  {'floored/step':>12}")
    for lo in range(1, last + 1, args.bins):
        hi = min(lo + args.bins - 1, last)
        steps = range(lo, hi + 1)
        c = [ctl[s][args.key] for s in steps if s in ctl and ctl[s][args.key] is not None]
        b = [arm[s][args.key] for s in steps if s in arm and arm[s][args.key] is not None]
        d = [arm[s][dose] for s in steps if s in arm and arm[s][dose] is not None]
        if not c or not b:
            continue
        mc, mb = sum(c) / len(c), sum(b) / len(b)
        print(f"{lo:4d}-{hi:<4d}  {mc:8.3f}  {mb:8.3f}  {mb - mc:+7.3f}  "
              f"{(sum(d) / len(d) if d else float('nan')):12.2f}")

    steps = [s for s in range(1, last + 1) if s in ctl and s in arm
             and ctl[s][args.key] is not None and arm[s][args.key] is not None]
    mc, sc, ec = mean_sd_se([ctl[s][args.key] for s in steps])
    mb, sb, eb = mean_sd_se([arm[s][args.key] for s in steps])
    se = math.sqrt(ec ** 2 + eb ** 2)
    print(f"\nsteps 1-{last} (n={len(steps)} paired by step, NOT paired rollouts)")
    print(f"  control  mean={mc:.4f} sd={sc:.4f} se={ec:.4f}")
    print(f"  B-prime  mean={mb:.4f} sd={sb:.4f} se={eb:.4f}")
    print(f"  diff    {mb - mc:+.4f} +/- {se:.4f}  = {(mb - mc) / se:+.2f} se")
    print("  both arms ran on 3 GPUs but on DIFFERENT HOSTS (control: yamabuki,\n"
          "  B': fuji). ALFWorld lists its games in os.walk order, which differs by\n"
          "  filesystem, so the arms drew different training games and were validated\n"
          "  on different 126-game sets: a between-run comparison, not a paired one.")
    full = [v[args.key] for v in ctl.values() if v[args.key] is not None]
    tail = [ctl[s][args.key] for s in range(max(ctl) - 49, max(ctl) + 1)
            if s in ctl and ctl[s][args.key] is not None]
    print(f"\n  control reference: all steps mean={sum(full) / len(full):.4f}, "
          f"last 50 mean={sum(tail) / len(tail):.4f}")


if __name__ == "__main__":
    main()
