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
                        # EVERY GAUGE COLUMN THE HEADER DECLARES, zeros kept.
                        # Absent in traces written before the gauges existed,
                        # and that has to stay readable: those runs get an empty
                        # dict and every excursion in them classifies as
                        # UNINSTRUMENTED, which is the truth about them.
                        #
                        # The zeros are what makes that distinction possible.
                        # Dropping them -- which this did, to keep the "at min:"
                        # line short -- collapses "the gauge read zero" into
                        # "this trace has no such gauge", and a classifier that
                        # asks whether a gauge EXISTS then cannot tell an old
                        # trace from a live zero. The print filters instead.
                        "gauges": {
                            name: int(float(value)) if (value or "").strip() else 0
                            for name, value in r.items()
                            if _is_gauge(name)
                        },
                        "activity": (r.get("activity") or "").strip(),
                        "stack_id": int(r["stack_id"]) if (r.get("stack_id") or "").strip() else None,
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


def read_stacks(trace_path):
    """The interned stack states, if the sampler wrote them.

    Optional on purpose. The sidecar is written beside the trace and named after
    it, and a trace copied off a box without its sidecar must still scan -- the
    excursion table simply loses the frame line, not the run.
    """
    path = trace_path + ".stacks"
    table = {}
    try:
        with open(path) as f:
            next(f, None)  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    table[int(parts[0])] = (int(parts[1]), parts[2])
    except OSError:
        return {}
    return table


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


def excursions(rows, dts, gi, floor, baseline, edge_margin=1.0, gen_full=1):
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
                # Over the window, with a pre-roll. NOT rows[argmin]: NVML
                # smooths, so the deepest sample is later than the event that
                # emptied the queue -- often after it has already refilled.
                **dict(zip(("why", "why_lead_in", "why_dwell"), why(rows, a, b, gen_full=gen_full))),
            }
        )
    return out


# THE GAUGE COLUMNS COME FROM THE FILE'S OWN HEADER, not from a list kept in
# step with the profiler's. It was a list, imported from verl when importable
# and hard-coded when not -- and the hard-coded copy went stale the day a gauge
# was added: `slots_free` was written into every row of the trace, read out of
# the header, and then dropped, so every excursion classified as though the
# gauge did not exist. A duplicated constant is the wrong shape for this;
# every column that is not one of the known fixed ones IS a gauge.
_NON_GAUGE_COLUMNS = frozenset(
    ["ts", "clock", "pid", "phase", "driver_cpu_pct", "activity", "stack_id"]
)


def _is_gauge(column):
    """Every column that is not fixed, and not one of the per-GPU series."""
    return column not in _NON_GAUGE_COLUMNS and not column.endswith("_per_gpu")


# How far back of an excursion's start to look for its cause.
#
# NVML's utilization.gpu is a moving-window average, not an instantaneous
# reading, so the sample where SM bottoms out is LATER than the event that
# emptied the queue -- by roughly the width of that window. On a 300 ms
# retrieval stall the sequence is: retriever finishes, generation restarts, and
# only then does the smoothed SM reach its minimum. At that sample
# retriever_inflight is already 0 and gen_inflight is already 1.
PRE_ROLL_S = float(os.environ.get("GPU_STALL_PRE_ROLL_S", "0.4"))


def why_one(gauges, cpu=None, gen_full=1):
    """The state at one sample, as a reason. Not a verdict on an excursion.

    ``gen_full`` is how many requests count as the engine HAVING WORK, and it is
    the difference between a useful answer and a wrong one. The first version
    tested ``if gen:`` -- any non-zero count -- and a real run answered GPU_SIDE
    for fifteen samples out of sixteen at excursions whose gauges read
    ``gen_inflight=1`` against a batch of 504. One request on three A6000s is
    not the engine having work; it is the last row of a turn draining while
    everything else waits. Calibrated from the trace's own busy-time median, so
    there is no constant here tied to a batch width.

    ``gen`` still outranks the dependency answers: a retrieval running while the
    engine is full is not why the cards are idle, however busy it looks.
    """
    if not gauges:
        return None
    ready = gauges.get("ready", 0)
    gen = gauges.get("gen_inflight", 0)
    retr = gauges.get("retriever_inflight", 0)
    env = gauges.get("env_inflight", 0)
    wait = gauges.get("future_wait", 0)

    if gen >= gen_full:
        return "GPU_SIDE"
    if gen:
        # A handful of requests: the tail of a turn. Whether that costs anything
        # depends entirely on whether a whole batch was sitting behind it.
        return "TAIL_BLOCKS_READY" if ready else "TAIL_DRAIN"
    # The engine is empty from here down.
    #
    # STARVATION NEEDS A FREE SLOT, not merely queued work. `ready` used to mean
    # "one batch that cannot be placed" and, once the pipeline grew a lookahead
    # queue, means "the queue is a queue" -- true nearly all the time. Reading
    # starvation off it alone more than doubled the number the moment the queue
    # arrived, which was a change in the instrument and not in the run. Work
    # waiting while every slot is legitimately occupied is a different finding
    # and gets a different name.
    free = gauges.get("slots_free", 0)
    if ready and free:
        # A FREE SLOT IS NOT A PLACEABLE ONE. Slots are task-typed: a free
        # alfworld slot cannot take a queued webshop batch, so "work ready, slot
        # free" covers two findings that want opposite fixes -- dispatch sooner,
        # versus make the queue's shapes match the slots that keep coming free.
        # Reported as one, the second reads as a dispatcher bug and sends the
        # next change at a dispatcher that is behaving correctly.
        #
        # `placeable_ready` counts the pending items some free slot would
        # actually accept, which is the distinction. Traces written before it
        # existed cannot draw it and keep the older, coarser answer rather than
        # having every such sample reported as a compatibility block. Zeros are
        # written, so the test is whether the COLUMN is there, not its value.
        if "placeable_ready" not in gauges:
            return "SCHEDULER_STARVATION"
        return ("SCHEDULER_STARVATION" if gauges["placeable_ready"]
                else "SLOT_COMPATIBILITY_BLOCK")
    if retr:
        return "RETRIEVER_DEPENDENCY"
    if env:
        return "ENV_DEPENDENCY"
    if ready:
        return "SLOTS_BUSY_NOT_GENERATING"
    if wait and (cpu is None or cpu < 20):
        return "FUTURE_RAY_WAIT"
    if cpu is not None and cpu >= 60:
        return "DRIVER_CPU"
    return None


def gen_full_threshold(rows, floor_frac=0.25, minimum=4, busy_sm=90.0):
    """How many in-flight requests count as "the engine has work", from the run.

    From samples where THE CARDS WERE BUSY, not merely where gen_inflight was
    non-zero. The difference is not cosmetic and the error runs the wrong way:
    a run full of turn tails has thousands of samples reading gen=1..4, those
    samples drag the median down, the threshold falls, and the tails stop being
    classified as tails. The more tail a run has, the less of it this would
    find -- a bias that hides exactly the thing it is looking for, and grows
    with the size of the finding.

    Median over the busy samples, scaled down: the threshold is "clearly less
    than a normal working load", not "a normal working load".
    """
    busy = sorted(
        r["gauges"]["gen_inflight"]
        for r in rows
        if r.get("gauges", {}).get("gen_inflight")
        and _mean_sm(r["sm"]) >= busy_sm
    )
    if not busy:
        return minimum
    return max(minimum, floor_frac * busy[len(busy) // 2])


def _mean_sm(sm):
    vals = [v for v in sm if v is not None]
    return (sum(vals) / len(vals)) if vals else 0.0


# Ranked. When an excursion's window holds more than one state -- and a short
# one usually does, because the window straddles the recovery -- the reason is
# the one that explains an EMPTY QUEUE, not the one that happens to coincide
# with the deepest sample. GPU_SIDE loses to all of them: it is the answer for
# an excursion that was never queue-starved at all, so if any sample in the
# window shows a starved queue, that is the finding.
_REASON_RANK = (
    "SCHEDULER_STARVATION",
    "SLOT_COMPATIBILITY_BLOCK",
    "TAIL_BLOCKS_READY",
    "SLOTS_BUSY_NOT_GENERATING",
    "RETRIEVER_DEPENDENCY",
    "ENV_DEPENDENCY",
    "FUTURE_RAY_WAIT",
    "DRIVER_CPU",
    "TAIL_DRAIN",
    "GPU_SIDE",
)


# The dwell line was abbreviated by first token, which read "slot:14 slots:2"
# the moment a second SLOT_ reason existed -- two different findings, one
# character apart, in the line that exists to make a close call visible.
_REASON_ABBR = {
    "SCHEDULER_STARVATION": "starved",
    "SLOT_COMPATIBILITY_BLOCK": "mismatch",
    "SLOTS_BUSY_NOT_GENERATING": "allbusy",
    "TAIL_BLOCKS_READY": "tailblock",
    "TAIL_DRAIN": "tail",
    "RETRIEVER_DEPENDENCY": "retriever",
    "ENV_DEPENDENCY": "env",
    "FUTURE_RAY_WAIT": "future",
    "DRIVER_CPU": "cpu",
    "GPU_SIDE": "gpu",
    "UNINSTRUMENTED": "unknown",
}


def _abbr(reason):
    return _REASON_ABBR.get(reason, reason.lower())


# Wide enough for every reason this scanner can print, including UNINSTRUMENTED,
# which is not in the rank tuple.
_WHY_W = max(len(r) for r in _REASON_ABBR) + 2


def why(rows, i0, i1, pre_roll=PRE_ROLL_S, gen_full=1):
    """Why an EXCURSION was idle, from the window that could have caused it.

    Reading the single deepest sample -- which this did -- attributes a stall to
    whatever was true after it had already recovered. The window runs from
    pre_roll seconds before the excursion starts to its end, and every sample in
    it votes; the highest-ranked reason present wins, with the dwell time on
    each kept so a marginal call is visible rather than silent.

    UNINSTRUMENTED stays reachable: a classifier with no such branch attributes
    every gap to whichever gauge happens to exist, and this arm has named a
    cause and then measured it under 0.05 of a slot twice.
    """
    t_from = rows[i0]["ts"] - pre_roll
    t_start = rows[i0]["ts"]
    dwell = defaultdict(int)
    lead = defaultdict(int)
    for r in rows[: i1 + 1]:
        if r["ts"] < t_from:
            continue
        reason = why_one(r.get("gauges"), r.get("cpu"), gen_full)
        if not reason:
            continue
        dwell[reason] += 1
        if r["ts"] < t_start:
            lead[reason] += 1
    if not dwell:
        return "UNINSTRUMENTED", None, {}

    # GPU_SIDE means THE QUEUE WAS FULL, so it cannot be the cause of an idle
    # caused by an empty queue -- and after a short stall most of the window is
    # recovery, where the queue is full again. So it is set aside whenever any
    # queue-emptying state appears at all, and among those, DWELL decides.
    #
    # Both halves are needed. Straight rank precedence let one sample outvote
    # fifteen ("gpu:15 scheduler:1" read as SCHEDULER_STARVATION -- right by
    # luck on that run). Straight dwell put a lagged retrieval stall back to
    # GPU_SIDE, because NVML's smoothing puts most of the window after the
    # retrieval finished. Neither rule alone survives both cases.
    starving = {k: n for k, n in dwell.items() if k != "GPU_SIDE"}
    pool = starving or dwell
    primary = max(pool, key=lambda k: (pool[k], -_REASON_RANK.index(k)
                                       if k in _REASON_RANK else -99))

    # The LEAD-IN is separate: what held in the pre-roll, before the cards fell.
    # That is the causal question, and it is worth reporting even when it lasted
    # one sample -- but as its own field, not as a verdict that overwrites the
    # thing the excursion actually consisted of.
    lead_in = None
    if lead:
        lead_in = max(lead, key=lambda k: (lead[k], -_REASON_RANK.index(k)
                                           if k in _REASON_RANK else -99))
        if lead_in == primary:
            lead_in = None
    return primary, lead_in, dict(dwell)


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
    stacks = read_stacks(path)
    gen_full = gen_full_threshold(rows)
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
        for e in excursions(rows, dts, gi, floor, baselines[gi], gen_full=gen_full):
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
        print(f"\nwhy the cards were idle, by lost GPU-seconds"
              f"   (the engine counts as having work at >= {gen_full:.0f} requests in flight,"
              f" calibrated from this run)")
        # Width from the names themselves: 22 was typed when the longest reason
        # was 21 characters, and the next one added ran into the events column.
        w = max(len(r) for r in by_why) + 2
        print(f"    {'reason':<{w}}{'events':>7}{'wall s':>9}{'lost GPU-s':>12}{'share':>8}")
        for reason, (n, wall, lost) in sorted(by_why.items(), key=lambda kv: -kv[1][2]):
            share = (lost / exc_lost * 100.0) if exc_lost else 0.0
            print(f"    {reason:<{w}}{n:>7}{wall:>9.1f}{lost:>12.1f}{share:>7.1f}%")
        meaning = {
            "SCHEDULER_STARVATION": "a slot was FREE, work was ready, and the engine had nothing -- "
                                    "the dispatcher missed it",
            "SLOT_COMPATIBILITY_BLOCK": "a slot was FREE and work was ready, but no ready batch fit "
                                        "the free slot -- the queue's task mix, not the dispatcher",
            "SLOTS_BUSY_NOT_GENERATING": "every slot was occupied and none was feeding the engine -- "
                                         "between turns, in the driver, or in the pump round trip",
            "TAIL_BLOCKS_READY": "the engine was draining a turn's last few rows while a whole batch "
                                 "waited for a slot -- more slots, or let a slot start before its tail ends",
            "TAIL_DRAIN": "a turn's last few rows were finishing and nothing else was ready to run",
            "RETRIEVER_DEPENDENCY": "nothing was submittable; the retrieval service was the dependency",
            "ENV_DEPENDENCY": "nothing was submittable; env.step was the dependency",
            "GPU_SIDE": "requests were with the engine -- its own host work, or a sample-window artefact",
            "FUTURE_RAY_WAIT": "the driver was blocked on a Future with nothing downstream recorded",
            "DRIVER_CPU": "the driver was running Python and no gauge explains what for",
            "UNINSTRUMENTED": "NO RULE MATCHED. Either the trace predates the gauges, or the gap is "
                              "somewhere nothing counts -- do not attribute it to the row above it",
        }
        for reason, _ in sorted(by_why.items(), key=lambda kv: -kv[1][2]):
            print(f"      {reason:<{w}}{meaning.get(reason, '')}")

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
        # The state at the bottom of THIS excursion, not averaged over the run.
        # A reason without it says which dependency; with it, which line.
        detail = []
        # The dwell, whenever the window held more than one state -- which a
        # short excursion usually does, because the window straddles the
        # recovery. Printing it makes a close call visible instead of leaving
        # one word standing for a vote it might have narrowly won.
        dwell = e.get("why_dwell") or {}
        if len(dwell) > 1:
            detail.append("samples " + " ".join(f"{_abbr(k)}:{v}" for k, v in
                                                sorted(dwell.items(), key=lambda kv: -kv[1])))
        # Zeros are carried in the rows (see read_trace) but not worth a line;
        # what is up at the bottom of the excursion is the readable part.
        up = {k: v for k, v in e["gauges"].items() if v}
        if up:
            detail.append("at min: " + " ".join(f"{k}={v}" for k, v in up.items()))
        if r.get("activity"):
            detail.append(f"in: {r['activity']}")
        if e.get("why_lead_in"):
            detail.insert(0, f"lead-in {e['why_lead_in']}")
        if detail:
            print(f"        {e['why']:<{_WHY_W}}{'   '.join(detail)}")
        frames = stacks.get(r.get("stack_id"))
        if frames:
            outside, text = frames
            for one in text.split(" | ")[:3]:
                print(f"        {'':<{_WHY_W}}{one}")
            if outside:
                print(f"        {'':<{_WHY_W}}(+{outside} threads outside this repo)")

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
        if not hits:
            # The profiler writes ONE FILE PER PROCESS and puts the pid in the
            # name -- GPU_PROFILER_TRACE=/tmp/t.csv produces /tmp/t.264168.csv,
            # because a driver sampler and a worker sampler opening one path "w"
            # overwrite each other. So the name the user typed is never the name
            # on disk, and this said "no such trace file" about a trace that was
            # sitting right beside it. Look for the siblings the profiler
            # actually writes before giving up.
            root, ext = os.path.splitext(p)
            hits = sorted(glob.glob(f"{root}.*{ext or '.csv'}"))
            if hits:
                print(f"  ({p} does not exist; using the per-process trace"
                      f"{'s' if len(hits) > 1 else ''} the profiler wrote: "
                      + ", ".join(os.path.basename(h) for h in hits) + ")")
        paths.extend(hits or [p])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(
            "no such trace file: " + ", ".join(missing)
            + "\n  GPU_PROFILER_TRACE=X.csv writes X.<pid>.csv, one per process."
            + "\n  Try:  ls " + os.path.splitext(missing[0])[0] + ".*"
        )

    print(
        f"scanning {len(paths)} trace file(s). Each file is one sampler process -- the profiler runs one\n"
        f"in the driver and one in rank 0's worker -- so they are analysed separately."
    )
    for p in paths:
        analyse(p, args.floor, args.busy, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
