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
"""Read the per-rank Chrome traces ``actor_capture``'s torch backend writes and
answer the two questions NVML cannot.

  1. **Which way does the straggler point?** For one micro-batch index, every
     rank's wall time, and how much of it each rank spent inside NCCL. A rank
     that is long *and* barely in NCCL is doing the work; ranks that are long
     *and* mostly in NCCL are waiting for it. NVML cannot tell those apart --
     a spinning collective is "busy" to it -- which is why this table exists.

  2. **Is the dip even a gap?** Inside each micro-batch the merged union of
     every kernel and copy is the device's real busy time; the complement is
     genuine idle, with no kernel resident on any stream. The largest hole is
     printed with the kernel that ended before it, the kernel that started
     after it, and the narrowest host-side op covering it -- which is the
     difference between "the host stopped submitting" and "the device was
     waiting on an event".

Usage
-----
    python3 scripts/actor_trace_summary.py /tmp/actor_trace/actor_rank*.json
    python3 scripts/actor_trace_summary.py /tmp/actor_trace --top 5

Times are printed in milliseconds. Chrome-trace timestamps are microseconds
from a per-process epoch, so nothing here compares two ranks' absolute clocks:
only durations of the same-numbered micro-batch, which are directly comparable.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

# Kernels, copies and memsets are all "the device is occupied". A gap is a
# stretch with none of them on any stream.
_DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
# torch names the record_function spans "user_annotation" on the CPU side; older
# builds emit them as cpu_op. Both are accepted so the file stays readable
# across torch versions.
_ANNOTATION_CATS = {"user_annotation", "cpu_op"}
_MICRO_RE = re.compile(r"^micro/(\d+)/(\d+)$")
_NCCL_RE = re.compile(r"nccl", re.IGNORECASE)


def _is_span(event) -> bool:
    """A complete event with a duration. Metadata, counters and flow arrows all
    lack one, and treating them as spans would put zero-width noise in the
    interval maths."""
    return event.get("ph") == "X" and event.get("dur") is not None


def read_trace(path):
    """One rank's file, split into what the tables need.

    Returns (micros, kernels, cpu_ops, launch_ts). ``micros`` maps the
    micro-batch counter to its (start, end); ``launch_ts`` maps a CUDA
    correlation id to the host timestamp that launched it.
    """
    with open(path) as handle:
        raw = json.load(handle)
    events = raw["traceEvents"] if isinstance(raw, dict) else raw

    micros, kernels, cpu_ops, launch_ts = {}, [], [], {}
    for event in events:
        if not _is_span(event):
            continue
        cat = event.get("cat", "")
        name = event.get("name", "")
        start, end = float(event["ts"]), float(event["ts"]) + float(event["dur"])
        correlation = (event.get("args") or {}).get("correlation")
        if cat in _ANNOTATION_CATS:
            hit = _MICRO_RE.match(name)
            if hit:
                micros[int(hit.group(1))] = (start, end)
            else:
                cpu_ops.append((start, end, name))
        elif cat in _DEVICE_CATS:
            kernels.append((start, end, name, correlation))
        elif cat == "cuda_runtime" and correlation is not None:
            launch_ts[correlation] = start

    return micros, kernels, cpu_ops, launch_ts


def merge(intervals):
    """Union of possibly-overlapping intervals.

    Concurrent streams overlap constantly, so summing durations would count the
    same wall-clock microsecond several times and report a device busier than
    100%. Only the union answers "was anything resident".
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def clip(intervals, window):
    """The parts of ``intervals`` inside ``window``, so a kernel that straddles
    a boundary is split between the two micro-batches rather than counted whole
    in both."""
    lo, hi = window
    out = []
    for start, end in intervals:
        start, end = max(start, lo), min(end, hi)
        if end > start:
            out.append((start, end))
    return out


def gaps(busy, window):
    """The complement of ``busy`` inside ``window``: the stretches with no
    kernel on any stream."""
    lo, hi = window
    out, cursor = [], lo
    for start, end in busy:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        out.append((cursor, hi))
    return out


def _innermost_over(cpu_ops, gap):
    """The narrowest host op that covers most of ``gap``.

    Two things make this less obvious than "what was running when the gap
    opened". Nesting is deep -- a Python frame contains an aten op contains a
    kernel launch -- and the narrowest one is the informative one:
    "cudaStreamSynchronize" says something, "nn.Module: Qwen3ForCausalLM" does
    not. And the blocking call does not have to start before the gap: the queue
    takes time to drain, so a host that blocks at t can leave the device busy
    for a few hundred microseconds afterwards, and the op that explains a 380 ms
    gap can begin just after it opens. Requiring it to cover the gap's start
    misses exactly that case and falls back to whatever phase encloses
    everything.
    """
    lo, hi = gap
    span = hi - lo
    covering = [(e - s, n) for s, e, n in cpu_ops if min(e, hi) - max(s, lo) >= 0.5 * span]
    if not covering:
        covering = [(e - s, n) for s, e, n in cpu_ops if s <= lo <= e]
    return min(covering)[1] if covering else "-"


def _bracket(kernels, when):
    """The device work either side of a gap, by name."""
    before = max((k for k in kernels if k[1] <= when + 1e-6), key=lambda k: k[1], default=None)
    after = min((k for k in kernels if k[0] >= when - 1e-6), key=lambda k: k[0], default=None)
    return (before[2] if before else "-", after[2] if after else "-")


def analyse(path):
    """One rank's per-micro-batch numbers."""
    micros, kernels, cpu_ops, launch_ts = read_trace(path)
    rows = []
    for counter, window in sorted(micros.items()):
        # Attribute a kernel to the micro-batch that *launched* it, not the one
        # it happened to run in: kernels lag their launch, so the boundary
        # kernels would otherwise drift into the next window.
        mine = []
        for start, end, name, correlation in kernels:
            when = launch_ts.get(correlation, start)
            if window[0] <= when < window[1]:
                mine.append((start, end, name))
        wall = window[1] - window[0]
        busy = clip(merge([(s, e) for s, e, _ in mine]), window)
        nccl = clip(merge([(s, e) for s, e, n in mine if _NCCL_RE.search(n)]), window)
        holes = sorted(gaps(busy, window), key=lambda g: g[1] - g[0], reverse=True)
        biggest = holes[0] if holes else None
        rows.append({
            "micro": counter,
            "wall_ms": wall / 1e3,
            "busy_ms": sum(e - s for s, e in busy) / 1e3,
            "nccl_ms": sum(e - s for s, e in nccl) / 1e3,
            "idle_ms": (wall - sum(e - s for s, e in busy)) / 1e3,
            "kernels": len(mine),
            "gap_ms": (biggest[1] - biggest[0]) / 1e3 if biggest else 0.0,
            "gap_at": biggest[0] if biggest else None,
            "gap_host": _innermost_over(cpu_ops, biggest) if biggest else "-",
            "gap_bracket": _bracket(mine, biggest[0]) if biggest else ("-", "-"),
        })
    return rows


def _rank_of(path) -> str:
    hit = re.search(r"rank(-?\d+)", os.path.basename(path))
    return hit.group(1) if hit else os.path.basename(path)


def _short(name, width=44):
    return name if len(name) <= width else name[: width - 1] + "…"


def report(by_rank, top):
    ranks = sorted(by_rank, key=lambda r: (len(r), r))
    counters = sorted({row["micro"] for rows in by_rank.values() for row in rows})
    if not counters:
        print("no micro/<n>/<i> annotations found -- was ACTOR_TORCH_MICRO set?")
        return

    print("\n== per micro-batch, per rank ==")
    print("  the first captured micro-batch carries the profiler's own CUPTI "
          "start-up; read from the second.\n")
    header = f"{'micro':>6}"
    for rank in ranks:
        header += f" | {'r' + rank + ' wall':>10} {'busy':>8} {'nccl':>8} {'idle':>8}"
    print(header)
    print("-" * len(header))
    for counter in counters:
        line = f"{counter:>6}"
        for rank in ranks:
            row = next((r for r in by_rank[rank] if r["micro"] == counter), None)
            if row is None:
                line += f" | {'-':>10} {'-':>8} {'-':>8} {'-':>8}"
            else:
                line += (f" | {row['wall_ms']:>10.1f} {row['busy_ms']:>8.1f} "
                         f"{row['nccl_ms']:>8.1f} {row['idle_ms']:>8.1f}")
        print(line)

    print("\n== who waits for whom ==")
    print("  The ranks leave a collective together, so the one that spent the "
          "LEAST time in NCCL is\n  the one everybody else was waiting for, and "
          "the others' NCCL time is that wait. Since\n  the micro-batch's wall "
          "is shared, the wait decomposes exactly into how much extra\n  compute "
          "that rank did and how much extra idle it sat through -- and those two "
          "want\n  completely different fixes. NVML can give neither: a spinning "
          "collective is busy to it.\n")
    print(f"{'micro':>6} | {'last in':>7} | {'others waited':>13} | "
          f"{'its extra compute':>17} | {'its extra idle':>14} | reading")
    print("-" * 92)
    for counter in counters:
        rows = {rank: next((r for r in by_rank[rank] if r["micro"] == counter), None)
                for rank in ranks}
        rows = {k: v for k, v in rows.items() if v}
        if len(rows) < 2:
            continue
        last = min(rows, key=lambda k: rows[k]["nccl_ms"])
        others = [v for k, v in rows.items() if k != last]
        waited = min(v["nccl_ms"] for v in others) - rows[last]["nccl_ms"]
        compute = {k: v["busy_ms"] - v["nccl_ms"] for k, v in rows.items()}
        extra_compute = compute[last] - max(compute[k] for k in rows if k != last)
        extra_idle = rows[last]["idle_ms"] - min(v["idle_ms"] for v in others)
        # Ranks are never matched to the microsecond, so somebody is always last
        # and somebody always waits. Naming a straggler over 1% would send the
        # next day into load balancing for nothing.
        if waited < 0.05 * rows[last]["wall_ms"]:
            reading = "balanced"
        elif extra_idle > extra_compute:
            reading = f"a stall on rank {last}"
        else:
            reading = f"rank {last} got more work"
        print(f"{counter:>6} | {'r' + last:>7} | {waited:>13.1f} | "
              f"{extra_compute:>17.1f} | {extra_idle:>14.1f} | {reading}")

    print("\n== largest idle gap in each micro-batch ==")
    print("  host = the narrowest CPU op covering most of the gap. A blocking "
          "call there (a sync, a\n  copy, an .item()) means the host stopped "
          "feeding the device; a plain aten op means it\n  was still dispatching "
          "and the wait is on the device side; '-' means the host was idle\n  too, "
          "which points outside this process entirely.\n")
    worst = sorted(
        ((rank, row) for rank, rows in by_rank.items() for row in rows),
        key=lambda pair: pair[1]["gap_ms"], reverse=True)[:top]
    for rank, row in worst:
        print(f"  rank {rank} micro {row['micro']:>4}  gap {row['gap_ms']:8.1f} ms "
              f"of {row['wall_ms']:.1f} ms wall  ({row['kernels']} kernels)")
        print(f"      host: {_short(row['gap_host'], 60)}")
        print(f"      last kernel before: {_short(row['gap_bracket'][0])}")
        print(f"      first kernel after: {_short(row['gap_bracket'][1])}")

    print("\n== totals ==")
    for rank in ranks:
        rows = by_rank[rank]
        wall = sum(r["wall_ms"] for r in rows)
        idle = sum(r["idle_ms"] for r in rows)
        nccl = sum(r["nccl_ms"] for r in rows)
        print(f"  rank {rank}: {len(rows)} micro-batches, {wall / 1e3:6.2f} s wall, "
              f"idle {idle / 1e3:5.2f} s ({100 * idle / wall:4.1f}%), "
              f"nccl {nccl / 1e3:5.2f} s ({100 * nccl / wall:4.1f}%)")


def _expand(paths):
    out = []
    for path in paths:
        if os.path.isdir(path):
            out += sorted(glob.glob(os.path.join(path, "*.json")))
        else:
            out += sorted(glob.glob(path)) or [path]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["/tmp/actor_trace"],
                        help="Chrome trace files, globs, or a directory")
    parser.add_argument("--top", type=int, default=8,
                        help="how many of the largest gaps to describe")
    args = parser.parse_args(argv)

    paths = _expand(args.paths or ["/tmp/actor_trace"])
    if not paths:
        print("no traces found", file=sys.stderr)
        return 1

    by_rank = defaultdict(list)
    for path in paths:
        print(f"reading {path} ({os.path.getsize(path) / (1 << 20):.1f} MiB)")
        by_rank[_rank_of(path)] = analyse(path)
    report(by_rank, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
