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
"""Split a finished run's GPU idle into checkpoint, step boundary, and ambient.

    python3 scripts/wandb_deficit_scan.py bgwezy3k
    python3 scripts/wandb_deficit_scan.py bgwezy3k ovb0yobz --project you/proj

``gpu_stall_scan.py`` does this from a 0.2 s trace, which has to be asked for
before the run starts. This does it from what wandb already recorded, so any
finished run can be re-read -- which is the only way to compare runs that were
launched weeks apart under different code.

Three things it does that reading the wandb chart does not.

**It checks what the samples are before integrating them.** The system stream
is 15-second *point* samples, not averages: in every run checked,
``utilization.gpu`` takes only integer values while ``powerWatts`` is
fractional, which an average of varying readings could not produce. A 1.65 s
event is therefore caught about 11% of the time it happens -- so "no zeros in
the chart" is not evidence that anything was fixed, and the tool says so out
loud rather than letting the chart imply it. The deficit integral
``sum((100 - sm)/100 * dt)`` is still unbiased under point sampling, because
each sample estimates the busy fraction of a randomly placed window; that is
what makes this comparable with a 0.2 s trace.

**It aligns the two streams on the wall clock.** ``_runtime`` is not shared:
a resumed run continues the previous run's ``_runtime`` in the training history
while the system stream restarts at zero. On run 2ynp20j6 the offset is 307 s,
which is a whole step -- enough to attribute a checkpoint's idle window to the
wrong step and conclude the checkpoint is innocent. Only ``_timestamp`` is
common to both.

**It cuts the samples against the step boundaries instead of averaging them.**
The interesting question is never "how much idle" but "idle attached to what",
and the three answers want completely different fixes: idle around a save is
the checkpoint, idle at a step boundary is the driver, and idle in the interior
is either a discrete event inside the step or the ambient dust between kernels.
Reported as one mean they are indistinguishable -- on run bgwezy3k the interior
looks like 1.055% of uniform loss, and 89% of it turns out to sit in 2.2% of
the windows, all of them in the step following a save.

The driver's own timers cross-check the result: a save whose write costs the
next step real time shows up as ``timing_s/step`` for step k+1 standing above
the baseline, measured by a clock that has nothing to do with NVML.

Needs ``WANDB_API_KEY`` in the environment, or a ``wandb login`` already done.
"""

import argparse
import os
import statistics
import sys

DEFAULT_PROJECT = "asayu255-/verl_agent_sft_multitask"

# A window is charged to a save if it lands within this many seconds of the save
# step's end. Generous on purpose: the write outlives the staging, and the point
# of the split is to keep the checkpoint's tail OUT of the ambient number rather
# than to date it precisely.
SAVE_PAD_S = 25.0
# The driver's per-step tail -- metrics, logging, the next batch's dispatch.
BOUNDARY_PAD_S = 8.0
# Below this a window is a discrete event, not dust. A 15 s window reading 90%
# needs 1.5 s of idle in it; nothing that gradual comes from kernel launch gaps.
DEEP_UTIL = 90.0


def _fetch(run_id, project):
    import wandb

    run = wandb.Api().run(f"{project}/runs/{run_id}")
    history = [r for r in run.scan_history(page_size=10000) if "training/global_step" in r]
    history.sort(key=lambda r: r["training/global_step"])
    system = [
        r
        for r in run.history(stream="events", pandas=False, samples=100000)
        if all(r.get(f"system.gpu.{k}.gpu") is not None for k in range(3))
    ]
    system.sort(key=lambda r: r["_timestamp"])
    return run, history, system


def _node(row, cards):
    return statistics.mean(row[f"system.gpu.{k}.gpu"] for k in range(cards))


def _sampling_note(system, cards):
    """Point samples or averages? The answer decides what a missing dip means."""
    utils = {row[f"system.gpu.{k}.gpu"] for row in system for k in range(cards)}
    integral = all(float(v).is_integer() for v in utils)
    deltas = [b["_timestamp"] - a["_timestamp"] for a, b in zip(system, system[1:])]
    period = statistics.median(deltas) if deltas else float("nan")
    return period, integral, sorted(utils)[:6]


def clock_epoch(system):
    """Wall-clock time of ``_runtime == 0`` for the system stream.

    The two streams do not share ``_runtime``: a resumed run continues the
    previous run's clock in the training history while the system stream starts
    at zero. Only ``_timestamp`` is common, so every comparison below converts
    through this.
    """
    return system[0]["_timestamp"] - system[0]["_runtime"]


def training_start(system, epoch, earliest, cards, busy=99.0):
    """When training actually started, not when the first step's timer opened.

    That timer reaches back over the dataloader spin-up -- and on a resumed run
    over the model load -- where the GPUs are idle for minutes by design.
    Counting it turns startup into "discrete events inside a step".
    """
    for row in system:
        t = row["_timestamp"] - epoch
        if t >= earliest and _node(row, cards) >= busy:
            return t
    return earliest


def classify(t, saves, bounds, save_pad=SAVE_PAD_S, boundary_pad=BOUNDARY_PAD_S):
    """Which of the three questions this window belongs to.

    Save first: a save step's end is also a step boundary, and charging its
    window to the driver would move the checkpoint's cost into the wrong column.
    """
    if any(abs(t - e) <= save_pad for e in saves):
        return "save"
    if any(abs(t - e) <= boundary_pad for e in bounds):
        return "boundary"
    return "interior"


def scan(run_id, project, cards=3):
    run, history, system = _fetch(run_id, project)
    if not history or not system:
        print(f"{run_id}: {len(history)} training rows, {len(system)} system rows -- nothing to scan")
        return

    period, integral, sample_values = _sampling_note(system, cards)
    save_freq = run.config.get("trainer", {}).get("save_freq")

    # The two streams share only the wall clock; _runtime restarts on resume.
    epoch = clock_epoch(system)
    ends = [(int(r["training/global_step"]), r["_timestamp"] - epoch, r["timing_s/step"]) for r in history]
    saves = [end for r, (_, end, _) in zip(history, ends) if r.get("timing_s/save_checkpoint") is not None]
    bounds = [end for _, end, _ in ends]
    stop = ends[-1][1]
    start = training_start(system, epoch, ends[0][1] - ends[0][2], cards)
    span = stop - start
    windows = [r for r in system if start <= r["_timestamp"] - epoch <= stop]
    if len(windows) < 10:
        print(f"{run_id}: only {len(windows)} system windows inside the training span -- skipping")
        return

    print(f"=== {run_id}  ({run.state}, save_freq={save_freq}, {len(history)} steps, {span / 3600:.2f} h)")
    print(f"    system stream: {len(windows)} windows, period {period:.1f} s, "
          f"util values {'INTEGER -> point samples' if integral else 'fractional -> averaged'} {sample_values}")
    if integral:
        for cost, what in ((run.summary.get("timing_s/save_checkpoint"), "a save"),):
            if cost:
                print(f"    -> {what} of {cost:.2f} s is caught by {100 * min(cost, period) / period:.0f}% of its "
                      f"occurrences; absence of a dip proves nothing")

    labels = {"save": "near a save (+-%ds)" % SAVE_PAD_S,
              "boundary": "step boundary (+-%ds)" % BOUNDARY_PAD_S,
              "interior": "interior"}
    groups = {name: [] for name in labels.values()}
    keys = list(groups)
    for row in windows:
        groups[labels[classify(row["_timestamp"] - epoch, saves, bounds)]].append(row)

    def report(name, rows):
        if not rows:
            return 0.0
        deficit = 100 - statistics.mean(_node(r, cards) for r in rows)
        lost = deficit / 100 * len(rows) * period
        print(f"{name:>28} {len(rows):6d} {100 * len(rows) / len(windows):6.1f}% "
              f"{deficit:8.3f}% {lost:8.0f} s {100 * lost / span:7.3f}%")
        return lost

    print(f"\n{'':>28} {'n':>6} {'share':>7} {'deficit':>9} {'lost':>10} {'of run':>8}")
    total = sum(report(name, rows) for name, rows in groups.items())
    report("TOTAL", windows)

    interior = groups[keys[2]]
    deep = [r for r in interior if _node(r, cards) < DEEP_UTIL]
    dust = [r for r in interior if _node(r, cards) >= DEEP_UTIL]
    lost_deep = sum((100 - _node(r, cards)) / 100 * period for r in deep)
    lost_dust = sum((100 - _node(r, cards)) / 100 * period for r in dust)
    print("\n  the interior splits:")
    print(f"    discrete (<{DEEP_UTIL:.0f}%){len(deep):6d} windows -> {lost_deep:7.0f} s = {100 * lost_deep / span:6.3f}% of run")
    mean_dust = f"   (mean {statistics.mean(_node(r, cards) for r in dust):.3f}%)" if dust else ""
    print(f"    AMBIENT      {len(dust):6d} windows -> {lost_dust:7.0f} s = {100 * lost_dust / span:6.3f}% of run{mean_dust}")
    if deep and saves:
        # "in the step after a save" = between a save step's end and one median
        # step later. That is the window the background write lives in.
        step_len = statistics.median([d for _, _, d in ends])
        after = sum(1 for r in deep if any(0 < (r["_timestamp"] - epoch) - e <= step_len for e in saves))
        print(f"    of the {len(deep)} discrete windows, {after} fall in the step after a save")

    # The driver's clock, which knows nothing about NVML.
    if save_freq and save_freq > 0:
        def group_of(step):
            if step % save_freq == 0:
                return "save step (k)"
            if (step - 1) % save_freq == 0:
                return "after a save (k+1)"
            if (step - 2) % save_freq == 0:
                return "k+2"
            return "baseline"

        buckets = {}
        for r in history:
            buckets.setdefault(group_of(int(r["training/global_step"])), []).append(r)
        base = buckets.get("baseline")
        if base and "after a save (k+1)" in buckets:
            print(f"\n  cross-check on the driver's timers:")
            print(f"{'':>22} {'n':>4} {'step_s':>9} {'worker_s':>10} {'mfu':>7}")
            for name in ("save step (k)", "after a save (k+1)", "k+2", "baseline"):
                rows = buckets.get(name)
                if not rows:
                    continue
                med = lambda key: statistics.median([r[key] for r in rows if isinstance(r.get(key), (int, float))])
                print(f"{name:>22} {len(rows):4d} {med('timing_s/step'):9.1f} "
                      f"{med('timing_s/update_actor_worker'):10.1f} {med('perf/mfu/actor'):7.3f}")
            baseline = statistics.median([r["timing_s/step"] for r in base])
            excess = sum(
                statistics.median([r["timing_s/step"] for r in buckets[name]]) - baseline
                for name in ("save step (k)", "after a save (k+1)", "k+2")
                if name in buckets
            )
            n_saves = len(buckets.get("save step (k)", []))
            print(f"    excess per save over k, k+1, k+2: {excess:+.1f} s"
                  f"   x{n_saves} saves = {100 * excess * n_saves / span:+.2f}% of the run")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="wandb run ids")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"entity/project (default {DEFAULT_PROJECT})")
    parser.add_argument("--cards", type=int, default=3, help="GPUs per node (default 3)")
    args = parser.parse_args(argv)

    if not os.environ.get("WANDB_API_KEY") and not os.path.exists(os.path.expanduser("~/.netrc")):
        print("no WANDB_API_KEY and no ~/.netrc -- run `wandb login` first", file=sys.stderr)
        return 1
    for run_id in args.runs:
        try:
            scan(run_id, args.project, args.cards)
        except Exception as exc:  # one bad id should not lose the other runs' output
            print(f"{run_id}: {type(exc).__name__}: {exc}\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
