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
"""Attribute a wandb run's GPU utilisation and memory to the driver's step phases.

    WANDB_API_KEY=... python3 scripts/wandb_phase_util.py \
        asayu255-/verl_agent_opd_grpo_cross_teacher_klw_sg1/runs/n9zfny6m

wandb's system metrics (``system.gpu.N.gpu`` / ``memoryAllocated`` / ``smActive``
/ ``pipeTensorActive``) are sampled on a wall clock that knows nothing about
``timing_s/*``. But the driver's phases are SEQUENTIAL inside a step and the
step ends at the history row's ``_timestamp``, so laying the phase durations out
backwards from that timestamp gives each system sample a phase. At wandb's ~7.5 s
sampling that is coarse -- a phase shorter than a sample gets nothing -- but the
four phases that carry 99% of the step (gen, old_log_prob, sign_weight_forward,
update_actor) are 40-250 s each, and the attribution is what the GPU_PROFILER
would give without a restart.

Three tables, and the reason each exists:

* per-phase wall clock and its share of the step -- where the time goes;
* per-phase utilisation / memory / SM-active / tensor-pipe -- what the card was
  doing while it went, per phase rather than as one run-wide mean;
* the idle DEFICIT INTEGRAL per phase, ``sum (100 - util) / 100 * dt`` --
  seconds the card spent with no kernel resident, which is what an overlap can
  recover. A mean utilisation cannot be converted into seconds; this can (see
  scripts/gpu_stall_scan.py for why the integral, not time-below-a-line).

Then one representative step (median duration, no checkpoint) as a 15 s trace,
because the per-phase means hide the periodic dips inside update_actor and the
memory step between gen and old_log_prob, both of which decide what to fix.

Reads only. The API key comes from the environment and is never printed.
"""
import argparse
import bisect
import json
import os
import statistics as st
import sys

PHASES = (
    "timing_s/gen", "timing_s/reward", "timing_s/old_log_prob", "timing_s/teacher_forward",
    "timing_s/sign_weight_forward", "timing_s/adv", "timing_s/update_actor",
    "timing_s/save_checkpoint",
)
MAJOR = ("timing_s/gen", "timing_s/old_log_prob", "timing_s/sign_weight_forward", "timing_s/update_actor")


def _mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else float("nan")


def _q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def fetch(run_path, cache_dir):
    """History rows with timing/perf keys, and the system-metric stream."""
    os.makedirs(cache_dir, exist_ok=True)
    hist_path = os.path.join(cache_dir, "history.json")
    sys_path = os.path.join(cache_dir, "system.json")
    if os.path.exists(hist_path) and os.path.exists(sys_path):
        return json.load(open(hist_path)), json.load(open(sys_path))
    import wandb

    api = wandb.Api(timeout=180)
    run = api.run(run_path)
    want = ("timing_s/", "perf/", "teacher_cache/", "sign_prefetch/", "teacher_prefetch/",
            "stall/", "global_seqlen/")
    hist = []
    for r in run.scan_history():
        hist.append({k: v for k, v in r.items() if k.startswith(want) or k in ("_step", "_timestamp")})
    events = list(run.history(stream="events", pandas=False, samples=10 ** 6))
    json.dump(hist, open(hist_path, "w"))
    json.dump(events, open(sys_path, "w"))
    print(f"run {run.name} ({run.state}), commit {run.commit}, gpu {run.metadata.get('gpu')} x{run.metadata.get('gpu_count')}")
    return hist, events


def attribute(hist, events, n_gpu):
    hist = [r for r in hist if r.get("timing_s/step")]
    ev = [e for e in events if e.get("system.gpu.0.gpu") is not None]
    ev.sort(key=lambda e: e["_timestamp"])
    ts = [e["_timestamp"] for e in ev]
    dt = st.median([ts[i + 1] - ts[i] for i in range(len(ts) - 1)])
    cols = {
        "gpu": [f"system.gpu.{i}.gpu" for i in range(n_gpu)],
        "mem": [f"system.gpu.{i}.memoryAllocated" for i in range(n_gpu)],
        "sm": ["system.gpu.0.smActive"], "tensor": ["system.gpu.0.pipeTensorActive"],
    }
    acc = {k: {c: [] for c in cols} for k in PHASES}
    deficit = {k: [] for k in PHASES}
    for r in hist:
        t = r["_timestamp"]
        for k in reversed(PHASES):
            d = r.get(k) or 0.0
            if d <= 0:
                continue
            start = t - d
            lo, hi = bisect.bisect_left(ts, start), bisect.bisect_right(ts, t)
            loss = 0.0
            for e in ev[lo:hi]:
                for c, keys in cols.items():
                    vals = [e.get(key) for key in keys]
                    acc[k][c].append(_mean(vals))
                util = _mean([e.get(key) for key in cols["gpu"]])
                loss += (100 - util) / 100 * dt
            deficit[k].append(loss)
            t = start
    return hist, ev, ts, dt, acc, deficit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_path")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--cache", default=".wandb_phase_cache")
    args = ap.parse_args()
    if not os.environ.get("WANDB_API_KEY") and not os.path.isdir(args.cache):
        sys.exit("set WANDB_API_KEY (or point --cache at a previous fetch)")
    hist, events = fetch(args.run_path, os.path.join(args.cache, args.run_path.replace("/", "_")))
    hist, ev, ts, dt, acc, deficit = attribute(hist, events, args.gpus)

    step = _mean([r["timing_s/step"] for r in hist])
    print(f"\n{len(hist)} steps, step mean {step:.1f} s, system sampling {dt:.1f} s\n")
    print(f"{'phase':22s} {'mean s':>8s} {'share':>6s} {'idle s':>7s} {'util%':>6s} {'mem%':>6s} {'smAct':>6s} {'tensr':>6s}  util p10/p50/p90")
    idle_total = 0.0
    for k in PHASES:
        v = [r.get(k) for r in hist if r.get(k)]
        if not v:
            continue
        m = st.mean(v)
        a = acc[k]
        idle = _mean(deficit[k]) if deficit[k] else 0.0
        idle_total += idle
        print(f"{k[9:]:22s} {m:8.1f} {100 * m / step:5.1f}% {idle:7.1f} {_mean(a['gpu']):6.1f} {_mean(a['mem']):6.1f} "
              f"{_mean(a['sm']):6.1f} {_mean(a['tensor']):6.1f}  {_q(a['gpu'], .1):4.0f}/{_q(a['gpu'], .5):4.0f}/{_q(a['gpu'], .9):4.0f}")
    print(f"{'GPU-idle total':22s} {'':8s} {'':6s} {idle_total:7.1f}  ({100 * idle_total / step:.0f}% of the step)")

    for k in ("perf/max_memory_allocated_gb", "perf/max_alloc_retries", "stall/max_cuda_mallocs", "stall/max_gc_gen2",
              "teacher_cache/gb", "teacher_prefetch/hit_rate", "sign_prefetch/hit_rate", "global_seqlen/minmax_diff",
              "global_seqlen/microbatch_wait_frac_columns", "perf/mfu/actor", "perf/throughput"):
        v = [r.get(k) for r in hist if r.get(k) is not None]
        if v:
            print(f"  {k:46s} med {st.median(v):12.3f}  min {min(v):12.3f}  max {max(v):12.3f}")
    print("  (perf/max_memory_allocated_gb and perf/max_alloc_retries are LIFETIME high-water marks / cumulative"
          " counters, not per-step -- see verl/utils/metric/memory.py)")

    # One representative step as a trace.
    noc = [r for r in hist if not r.get("timing_s/save_checkpoint")]
    mid = sorted(noc, key=lambda r: r["timing_s/step"])[len(noc) // 2]
    end = mid["_timestamp"]
    start = end - mid["timing_s/step"]
    bounds, t = [], start
    for k in PHASES:
        d = mid.get(k) or 0.0
        bounds.append((k[9:], t, t + d))
        t += d
    lo, hi = bisect.bisect_left(ts, start), bisect.bisect_right(ts, end)
    print(f"\nrepresentative step {int(mid['_step'])} ({mid['timing_s/step']:.0f} s), one line per sample:")
    print("   t(s)  phase                 mem%   util%   smAct  tensor")
    for e in ev[lo:hi]:
        ph = next((n for n, a, b in bounds if a <= e["_timestamp"] < b), "?")
        mem = _mean([e.get(f"system.gpu.{i}.memoryAllocated") for i in range(args.gpus)])
        util = _mean([e.get(f"system.gpu.{i}.gpu") for i in range(args.gpus)])
        print(f"  {e['_timestamp'] - start:5.0f}  {ph:20s} {mem:5.1f}  {util:6.1f}  {e.get('system.gpu.0.smActive') or float('nan'):6.1f} "
              f"{e.get('system.gpu.0.pipeTensorActive') or float('nan'):6.1f}")


if __name__ == "__main__":
    main()
