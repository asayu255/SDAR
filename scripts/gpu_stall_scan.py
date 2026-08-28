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
"""Find where a GPU_PROFILER_TRACE run actually loses GPU time, per card.

    python3 scripts/gpu_stall_scan.py /tmp/trace.*.csv

Two things this does that counting samples under a threshold does not.

**It measures idle as a deficit integral, not as time-below-a-line.** NVML's
``utilization.gpu`` is the fraction of a trailing sample window (1 s to 1/6 s,
product-dependent) in which at least one kernel was resident -- a moving
average, not an instantaneous state. A real 0.8 s stall therefore never
produces 0.8 s of readings at 0%: the window slides across it and the reading
dips and recovers, so "time below 50%" reports a fraction of the truth and
shrinks further the wider the window. Integrating ``(100 - sm)/100 * dt``
instead recovers the underlying idle time, because the integral of a moving
average over a span equals the integral of what it averaged. That estimator is
window-independent, which is what makes a 0.2 s trace comparable with wandb's
15 s system metrics at all.

**It scans per card, not per node mean.** On a data-parallel run a single rank
stalling is the common failure, and it is invisible in a node mean: the other
ranks enter the collective and spin, NCCL's spin kernels count as busy, so two
cards read 100% while the third reads 0 and the mean lands at 67%. Averaging
first turns the one signal that says "this is a straggler, look at that rank"
into a shallow dip that looks like everything slowed down slightly.

The output splits the total lost time two ways -- into discrete excursions
(something specific happened; the listing says which card, when, in which
phase) and ambient deficit spread over every other sample (no event to find;
that is kernel-launch dust and only fewer/larger kernels move it). Those two
want completely different fixes, so the split is the point of the report.

One trace file is one sampler process. Pass them all: the profiler runs a
sampler in the driver and another in rank 0's worker, and each file is analysed
on its own because only within a file are consecutive rows consecutive in time.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict


def _floats(field):
    """Parse a ``a;b;c`` per-GPU cell. Missing entries come back as None."""
    out = []
    for tok in (field or "").split(";"):
        tok = tok.strip()
        try:
            out.append(float(tok))
        except ValueError:
            out.append(None)
    return out


def read_trace(path):
    """Rows as dicts, time-ordered, with malformed lines dropped.

    A run that ends in Ctrl-C or a crash leaves a torn final line, and those are
    the runs worth looking at, so a parse failure skips the row rather than the
    file.
    """
    rows, skipped = [], 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                sm = _floats(r["sm_pct_per_gpu"])
                if not sm or all(v is None for v in sm):
                    raise ValueError("no sm values")
                rows.append(
                    {
                        "ts": float(r["ts"]),
                        "clock": r.get("clock") or "",
                        "phase": r.get("phase") or "?",
                        "sm": sm,
                        "power": _floats(r.get("power_w_per_gpu")),
                        "clk": _floats(r.get("smclk_mhz_per_gpu")),
                        "pcie_rx": _floats(r.get("pcie_rx_mb_s_per_gpu")),
                        "cpu": float(r["driver_cpu_pct"]) if (r.get("driver_cpu_pct") or "").strip() else None,
                        # Absent in traces written before the gauges existed, and
                        # that has to stay readable: those runs get an empty dict
                        # and every excursion in them classifies as
                        # UNINSTRUMENTED, which is the truth about them.
                        "gauges": {
                            name: int(float(r[name]))
                            for name in GAUGE_NAMES
                            if (r.get(name) or "").strip() not in ("", "0")
                        },
                    }
                )
            except (TypeError, ValueError, AttributeError, KeyError):
                skipped += 1
    rows.sort(key=lambda r: r["ts"])
    # A trace written by two processes to one path has rows from both streams at
    # whatever offset each reached, so widths disagree; keep the majority width.
    if rows:
        width = max(len(r["sm"]) for r in rows)
        rows = [r for r in rows if len(r["sm"]) == width]
    return rows, skipped


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def sample_dts(rows, gap_factor=3.0):
    """Per-sample dt, with file gaps clipped so they are not charged as idle.

    A gap means the sampler did not run (GIL contention, a clobbered file, a
    process that died); it is not evidence about the GPU either way, so it is
    excluded from the denominator instead of counted as busy or as idle.
    """
    if len(rows) < 2:
        return [0.0] * len(rows), 0, 0.0
    raw = [rows[i]["ts"] - rows[i - 1]["ts"] for i in range(1, len(rows))]
    step = median(raw) or 0.0
    cutoff = gap_factor * step if step else float("inf")
    dts, n_gaps, gap_time = [step], 0, 0.0
    for d in raw:
        if d > cutoff:
            n_gaps += 1
            gap_time += d
            dts.append(0.0)  # this interval was not observed
        else:
            dts.append(d)
    return dts, n_gaps, gap_time


def excursions(rows, dts, gi, floor, baseline, edge_margin=1.0):
    """Contiguous windows where card ``gi`` fell away from its own baseline.

    Detection uses ``floor`` but the window is then grown outward while the card
    is still below ``baseline - edge_margin``. That matters because of the
    trailing-average window: the deep part of a dip is the middle of the
    excursion and the shoulders carry real deficit too, so a window cut at the
    threshold would drop most of the event's cost.
    """
    edge = baseline - edge_margin
    seeds = [i for i, r in enumerate(rows) if r["sm"][gi] is not None and r["sm"][gi] < floor]
    out, last_end = [], -1
    for i in seeds:
        if i <= last_end:
            continue
        a = i
        while a - 1 >= 0 and rows[a - 1]["sm"][gi] is not None and rows[a - 1]["sm"][gi] < edge:
            a -= 1
        b = i
        while b + 1 < len(rows) and rows[b + 1]["sm"][gi] is not None and rows[b + 1]["sm"][gi] < edge:
            b += 1
        last_end = b
        window = range(a, b + 1)
        lost = sum(max(0.0, baseline - rows[k]["sm"][gi]) / 100.0 * dts[k] for k in window)
        argmin = min(window, key=lambda k: rows[k]["sm"][gi])
        out.append(
            {
                "gpu": gi,
                "i0": a,
                "i1": b,
                "t0": rows[a]["ts"],
                "wall": rows[b]["ts"] - rows[a]["ts"],
                "lost_s": lost,
                "argmin": argmin,
                "min_sm": rows[argmin]["sm"][gi],
                "phases": sorted({rows[k]["phase"] for k in window}),
                # Read at the deepest sample, not averaged over the window: the
                # shoulders of an excursion are the GPU draining and refilling,
                # and the state that explains it is the one at the bottom.
                "gauges": rows[argmin].get("gauges") or {},
                "why": why(rows[argmin].get("gauges") or {}, rows[argmin].get("cpu")),
            }
        )
    return out


# Kept in step with gpu_profiler.GAUGE_NAMES. Imported when verl is importable,
# hard-coded when it is not, because this script is run against a CSV on a
# laptop as often as beside the code that wrote it.
try:  # pragma: no cover - exercised both ways in the tests
    from verl.utils.gpu_profiler import GAUGE_NAMES
except Exception:  # pragma: no cover
    GAUGE_NAMES = ("ready", "gen_inflight", "retriever_inflight", "env_inflight", "future_wait")


def why(gauges, cpu=None):
    """What the outstanding work says about an idle GPU.

    The two answers this exists to separate look identical on the device:

        ready=0,  retriever_inflight=10  ->  nothing COULD be submitted
        ready=32, retriever_inflight=10  ->  something could and was not

    and their fixes are opposite -- speed up the retriever, or stop
    lock-stepping and submit what was already waiting.

    UNINSTRUMENTED is a real answer and must stay reachable. A classifier with
    no such branch attributes every gap to whichever gauge happens to exist,
    and then argues confidently for a fix aimed at the wrong thing; this arm has
    named a cause and then measured it at under 0.05 of a slot twice.
    """
    if not gauges:
        return "UNINSTRUMENTED"
    ready = gauges.get("ready", 0)
    gen = gauges.get("gen_inflight", 0)
    retr = gauges.get("retriever_inflight", 0)
    env = gauges.get("env_inflight", 0)
    wait = gauges.get("future_wait", 0)

    if ready and not gen:
        return "SCHEDULER_STARVATION"
    if gen:
        return "GPU_SIDE"
    if retr and not ready:
        return "RETRIEVER_DEPENDENCY"
    if env and not ready:
        return "ENV_DEPENDENCY"
    if wait and (cpu is None or cpu < 20):
        return "FUTURE_RAY_WAIT"
    if cpu is not None and cpu >= 60:
        return "DRIVER_CPU"
    return "UNINSTRUMENTED"


def classify(rows, exc, floor, busy=95.0):
    """Was this one card, or the whole node?

    The distinction is the whole diagnosis. One card down while the others read
    ~100% is a straggler -- the others are spinning in a collective waiting for
    it -- and the cause is rank-local. Every card down together is a real global
    gap, and the cause is in the driver or between steps.
    """
    r = rows[exc["argmin"]]
    others = [v for j, v in enumerate(r["sm"]) if j != exc["gpu"] and v is not None]
    if not others:
        return "single-gpu"
    if min(others) >= busy:
        return "solo"
    if max(others) < floor:
        return "all"
    return "partial"


def analyse(path, floor, busy, top):
    rows, skipped = read_trace(path)
    if not rows:
        print(f"\n=== {path}: no usable rows ===")
        return
    dts, n_gaps, gap_time = sample_dts(rows)
    ngpu = len(rows[0]["sm"])
    span = rows[-1]["ts"] - rows[0]["ts"]
    observed = sum(dts)

    print(f"\n=== {os.path.basename(path)} ===")
    print(
        f"{len(rows)} samples over {span / 60:.1f} min, {ngpu} GPUs, "
        f"{median([dts[i] for i in range(1, len(dts))]) * 1000 if len(dts) > 1 else 0:.0f} ms apart"
        + (f"  ({skipped} malformed rows dropped)" if skipped else "")
    )
    if n_gaps:
        # Worth shouting about: before the per-pid trace path existed, two
        # samplers overwrote one file and the result was mostly gaps.
        print(
            f"  !! {n_gaps} sampling gaps totalling {gap_time:.0f}s ({gap_time / span * 100:.0f}% of the span) "
            f"are excluded -- the sampler was not running there"
        )

    baselines = []
    print("\nper card: mean sm, and total time with no kernel resident")
    print(f"  {'gpu':<5}{'mean sm':>9}{'baseline':>10}{'no-kernel s':>13}{'% of card':>11}")
    total_lost = 0.0
    for gi in range(ngpu):
        v = [r["sm"][gi] for r in rows if r["sm"][gi] is not None]
        base = median(v) or 100.0
        baselines.append(base)
        lost = sum((100.0 - r["sm"][gi]) / 100.0 * dts[k] for k, r in enumerate(rows) if r["sm"][gi] is not None)
        total_lost += lost
        print(f"  gpu{gi:<2}{sum(v) / len(v):>9.2f}{base:>10.0f}{lost:>13.1f}{lost / observed * 100:>10.2f}%")
    print(f"  {'all':<5}{'':>9}{'':>10}{total_lost:>13.1f}{total_lost / (observed * ngpu) * 100:>10.2f}%  <- aggregate")

    # Split that total into "something happened here" and "everywhere, always".
    all_exc = []
    for gi in range(ngpu):
        for e in excursions(rows, dts, gi, floor, baselines[gi]):
            e["kind"] = classify(rows, e, floor, busy)
            all_exc.append(e)
    all_exc.sort(key=lambda e: -e["lost_s"])
    exc_lost = sum(e["lost_s"] for e in all_exc)

    def _split(label, secs, note):
        share = (secs / total_lost * 100.0) if total_lost else 0.0
        print(f"  {label:<26}{secs:>8.1f}s{share:>7.1f}% of the loss   <- {note}")

    print(f"\nwhere that time is (excursion = a card below {floor:.0f}% of its own baseline)")
    _split(f"{len(all_exc)} discrete excursions", exc_lost, "real events; card and phase listed below")
    _split("ambient, everywhere else", total_lost - exc_lost, "launch-gap dust; needs fewer/larger kernels")
    by_kind = defaultdict(lambda: [0, 0.0])
    for e in all_exc:
        by_kind[e["kind"]][0] += 1
        by_kind[e["kind"]][1] += e["lost_s"]
    for kind, (n, s) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        what = {
            "solo": "one card stalled, the others spinning in a collective (rank-local cause)",
            "all": "every card down together (driver-side / between-step)",
            "partial": "mixed",
            "single-gpu": "only one GPU in the trace",
        }[kind]
        print(f"    {kind:<9}{n:>5} excursions {s:>8.1f}s   {what}")

    # WHY, not just where. Ranked by lost GPU-seconds rather than by count or by
    # wall: an excursion's cost is the integral of the deficit, and thirty short
    # ones can outweigh a single long one that only took a card to 60%.
    by_why = defaultdict(lambda: [0, 0.0, 0.0])
    for e in all_exc:
        row = by_why[e["why"]]
        row[0] += 1
        row[1] += e["wall"]
        row[2] += e["lost_s"]
    if by_why:
        print("\nwhy the cards were idle, by lost GPU-seconds")
        print(f"    {'reason':<22}{'events':>7}{'wall s':>9}{'lost GPU-s':>12}{'share':>8}")
        for reason, (n, wall, lost) in sorted(by_why.items(), key=lambda kv: -kv[1][2]):
            share = (lost / exc_lost * 100.0) if exc_lost else 0.0
            print(f"    {reason:<22}{n:>7}{wall:>9.1f}{lost:>12.1f}{share:>7.1f}%")
        meaning = {
            "SCHEDULER_STARVATION": "work was ready and the engine had none of it -- fix the submitting side",
            "RETRIEVER_DEPENDENCY": "nothing was submittable; the retrieval service was the dependency",
            "ENV_DEPENDENCY": "nothing was submittable; env.step was the dependency",
            "GPU_SIDE": "requests were with the engine -- its own host work, or a sample-window artefact",
            "FUTURE_RAY_WAIT": "the driver was blocked on a Future with nothing downstream recorded",
            "DRIVER_CPU": "the driver was running Python and no gauge explains what for",
            "UNINSTRUMENTED": "NO RULE MATCHED. Either the trace predates the gauges, or the gap is "
                              "somewhere nothing counts -- do not attribute it to the row above it",
        }
        for reason, _ in sorted(by_why.items(), key=lambda kv: -kv[1][2]):
            print(f"      {reason:<22}{meaning.get(reason, '')}")

    print(f"\nper phase: share of wall clock, and the loss attributed to it")
    ph_wall, ph_lost, ph_exc = defaultdict(float), defaultdict(float), defaultdict(float)
    for k, r in enumerate(rows):
        ph_wall[r["phase"]] += dts[k]
        for gi in range(ngpu):
            if r["sm"][gi] is not None:
                ph_lost[r["phase"]] += (100.0 - r["sm"][gi]) / 100.0 * dts[k]
    for e in all_exc:
        ph_exc[rows[e["argmin"]]["phase"]] += e["lost_s"]
    print(f"  {'phase':<20}{'wall%':>8}{'mean sm':>9}{'lost s':>9}{'in excursions':>15}")
    for ph, w in sorted(ph_wall.items(), key=lambda kv: -kv[1]):
        mean_sm = 100.0 - (ph_lost[ph] / (w * ngpu) * 100.0 if w else 0.0)
        print(f"  {ph:<20}{w / observed * 100:>7.1f}%{mean_sm:>9.1f}{ph_lost[ph]:>9.1f}{ph_exc[ph]:>15.1f}")

    if not all_exc:
        return
    print(f"\nthe {min(top, len(all_exc))} costliest excursions")
    print(
        f"  {'t+s':>8}{'clock':>10}{'gpu':>5}{'kind':>9}{'wall':>7}{'lost':>7}{'min sm':>8}"
        f"{'others':>14}{'cpu%':>6}  phase"
    )
    t_origin = rows[0]["ts"]
    for e in all_exc[:top]:
        r = rows[e["argmin"]]
        others = "/".join(
            "-" if v is None else f"{v:.0f}" for j, v in enumerate(r["sm"]) if j != e["gpu"]
        )
        cpu = "-" if r["cpu"] is None else f"{r['cpu']:.0f}"
        print(
            f"  {e['t0'] - t_origin:>8.1f}{r['clock']:>10}{e['gpu']:>5}{e['kind']:>9}"
            f"{e['wall']:>7.2f}{e['lost_s']:>7.2f}{e['min_sm']:>8.0f}{others:>14}{cpu:>6}  "
            f"{','.join(e['phases'])}"
        )

    # Corroboration. A card that really went idle drops its power and clock; one
    # whose reading fell while power held is more likely a counter artefact than
    # a stall, and the two want different follow-ups.
    print("\n  at each excursion's minimum, that card's power / SM clock vs its own baseline")
    for gi in range(ngpu):
        mine = [e for e in all_exc[:top] if e["gpu"] == gi]
        if not mine:
            continue
        pw = [r["power"][gi] for r in rows if gi < len(r["power"]) and r["power"][gi] is not None]
        ck = [r["clk"][gi] for r in rows if gi < len(r["clk"]) and r["clk"][gi] is not None]
        if not pw:
            print(f"    gpu{gi}: power/clock not in this trace (written before those columns existed)")
            continue
        for e in mine:
            r = rows[e["argmin"]]
            p = r["power"][gi] if gi < len(r["power"]) else None
            c = r["clk"][gi] if gi < len(r["clk"]) else None
            print(
                f"    gpu{gi} t+{e['t0'] - t_origin:7.1f}s  sm {e['min_sm']:>3.0f}   "
                f"power {'-' if p is None else f'{p:5.0f}'} W (base {median(pw):5.0f})   "
                f"clock {'-' if c is None else f'{c:5.0f}'} MHz (base {median(ck) if ck else 0:5.0f})"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="GPU_PROFILER_TRACE csv files (globs are expanded)")
    ap.add_argument("--floor", type=float, default=90.0, help="a card below this starts an excursion (default 90)")
    ap.add_argument("--busy", type=float, default=95.0, help="other cards at or above this make it 'solo' (default 95)")
    ap.add_argument("--top", type=int, default=20, help="how many excursions to list (default 20)")
    args = ap.parse_args(argv)

    paths = []
    for p in args.traces:
        hits = sorted(glob.glob(p))
        paths.extend(hits or [p])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("no such trace file: " + ", ".join(missing))

    print(
        f"scanning {len(paths)} trace file(s). Each file is one sampler process -- the profiler runs one\n"
        f"in the driver and one in rank 0's worker -- so they are analysed separately."
    )
    for p in paths:
        analyse(p, args.floor, args.busy, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
