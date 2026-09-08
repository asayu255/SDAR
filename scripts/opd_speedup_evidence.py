#!/usr/bin/env python3
"""Recompute every number the speedup decision record rests on, from run logs.

Each block prints the figure and the item it decides, so a later series can rerun
this instead of re-deriving. Nothing here reads wandb: the console logs carry
enough, except that they round metrics to three decimals (aggregate_logger.py),
which is why the eps-relevant ratios are recovered from pushback/bound rather
than from ctl_R and ctl_C_neg directly.

Usage:
  opd_speedup_evidence.py --run RUN.log [--turns TURNTABLE.log] [--ab OTHER.log]

  --run    a training log with perf/ and timing_s/ metric lines
  --turns  a log started with ROLLOUT_TURN_TIMING=1, for the gen decomposition
  --ab     a second run to compare step 1 against (the micro-batch A/B)
"""
import argparse, re, statistics as st, sys


def _series(path, key):
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            for m in re.finditer(rf"{re.escape(key)}:([0-9.]+)", line):
                out.append(float(m.group(1)))
    return out


def _steps(path):
    """One dict per completed step, keyed by metric name."""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "perf/time_per_step" not in line:
                continue
            rows.append({m.group(1): float(m.group(2))
                         for m in re.finditer(r"([a-zA-Z_]+/[a-zA-Z_/0-9.]+):([0-9.]+)", line)})
    return rows


def phase_breakdown(path):
    print("== 1. phase breakdown (which phase is worth attacking) ==")
    tot = _series(path, "perf/time_per_step")
    if not tot:
        print("   no perf/time_per_step lines\n")
        return
    last = slice(-30, None)
    for key, lab in (("perf/time_per_step", "step"), ("timing_s/gen", "gen"),
                     ("timing_s/update_actor", "update_actor"),
                     ("timing_s/old_log_prob", "old_log_prob")):
        v = _series(path, key)
        if v:
            print(f"   {lab:<14} all-median {st.median(v):7.1f}   last30-median {st.median(v[last]):7.1f}")
    g, u = _series(path, "timing_s/gen")[last], _series(path, "timing_s/update_actor")[last]
    s = _series(path, "perf/time_per_step")[last]
    if g and u and s:
        print(f"   last30 shares: gen {100*st.median(g)/st.median(s):.0f}%"
              f"  update {100*st.median(u)/st.median(s):.0f}%")
    print()


def paired_periodic_test(path, every=5, key="timing_s/update_actor"):
    """Item 3.5. A periodic overhead must be tested against NEIGHBOURS.

    Splitting a 60-step window into two groups and comparing medians is dominated
    by the +-20 s step-to-step swing; pairing each multiple-of-`every` step with
    the mean of its two flanking steps removes the drift.
    """
    print(f"== 2. periodic diagnostics overhead, paired ({key}, every {every}) ==")
    v = _series(path, key)
    if len(v) < every * 3:
        print("   too few steps\n")
        return
    deltas = []
    for i, x in enumerate(v, start=1):          # 1-indexed step numbers
        if i % every or i - 2 < 1 or i + 2 > len(v):
            continue
        flank = [v[i - 3], v[i - 2], v[i], v[i + 1]]   # steps i-2, i-1, i+1, i+2
        deltas.append(x - st.mean(flank))
    if not deltas:
        print("   no usable pairs\n")
        return
    pos = sum(1 for d in deltas if d > 0)
    print(f"   n={len(deltas)}  median {st.median(deltas):+.1f} s  mean {st.mean(deltas):+.1f} s"
          f"  positive {pos}/{len(deltas)}")
    print(f"   amortised over every step: {st.median(deltas)/every:+.2f} s/step")
    print("   NOTE: an unpaired group-median comparison on this data is dominated by")
    print("         the step-to-step swing and can read as tens of seconds.\n")


def gen_decomposition(path):
    """Items 3.6, 3.7, 3.8. Needs ROLLOUT_TURN_TIMING=1."""
    print("== 3. gen decomposition and the decode tail (turn table) ==")
    rows, tables, cur = [], 0, False
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "turn  active" in line:
                tables += 1
                continue
            m = re.search(r"\s(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
                          r"\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[-0-9.]", line)
            if m and "turn" not in line:
                t, a, pre, gen, tw, dec, env, tt = (float(x) for x in m.groups())
                rows.append(dict(active=a, preproc=pre, gen=gen, tchwait=tw,
                                 decode=dec, envstep=env, total=tt))
    if not rows:
        print("   no turn table in this log (start the run with ROLLOUT_TURN_TIMING=1)\n")
        return
    S = {k: sum(r[k] for r in rows) for k in ("preproc", "gen", "tchwait", "decode", "envstep", "total")}
    print(f"   {len(rows)} turn rows in {tables} tables")
    for k in ("gen", "envstep", "preproc", "tchwait", "decode"):
        print(f"   {k:<9} {S[k]:9.0f} s  {100*S[k]/S['total']:5.1f} % of the gen phase")
    print(f"   inside the engine {100*S['gen']/S['total']:.1f} %   outside {100*(S['total']-S['gen'])/S['total']:.1f} %")
    print("   -> 3.7's ceiling is envstep+preproc; 3.8's is tchwait")
    peak = max(r["active"] for r in rows)
    lo = [r for r in rows if r["active"] <= 0.4 * peak]
    hi = [r for r in rows if r["active"] > 0.7 * peak]
    if lo and hi:
        print(f"   tail (<=40% of peak {peak:.0f}): {100*sum(r['gen'] for r in lo)/S['gen']:.0f} % of engine time,"
              f" {1000*sum(r['gen'] for r in lo)/sum(r['active'] for r in lo):.0f} ms/seq/turn")
        print(f"   head (> 70% of peak):        {100*sum(r['gen'] for r in hi)/S['gen']:.0f} % of engine time,"
              f" {1000*sum(r['gen'] for r in hi)/sum(r['active'] for r in hi):.0f} ms/seq/turn")
        print("   -> the tail is what speculative decoding (3.6) addresses\n")


def allgather_volume(path, params_gib=3.78, ranks=2):
    """Item 3.1. Why halving the micro-batch count buys anything."""
    print("== 4. per-micro-batch parameter all-gather (item 3.1) ==")
    counts = [int(m.group(1)) for line in open(path, errors="ignore")
              for m in [re.search(r"\[step-gpu\] rank \d step \d+: (\d+) micro-batches", line)] if m]
    if not counts:
        print("   no [step-gpu] lines\n")
        return
    mb = st.median(counts)
    recv = params_gib * (ranks - 1) / ranks
    print(f"   micro-batches / rank / step   {mb:.0f}   (median of {len(counts)} samples)")
    print(f"   received per full-model gather {recv:.2f} GiB   (SHARD_GRAD_OP: one per micro-batch)")
    print(f"   traffic / rank / step          {recv*mb:.0f} GiB")
    print(f"   sustained bandwidth that needs, at 362.7 ms/micro-batch: "
          f"{recv*1024/0.3627:.0f} MB/s")
    print("   compare against `nvidia-smi dmon -s t` rxpci during the update phase.")
    print("   NOTE: with no NVLink (topo -m reports SYS) this crosses the host.\n")


def microbatch_token_extremes(path, sizes=(5, 10)):
    """Item 3.3. Checkpointing-off activations scale with TOKENS per micro-batch,
    so the average is the wrong statistic -- the worst micro-batch is."""
    print("== 5. micro-batch token counts: average vs worst case (item 3.3) ==")
    pm, rm = _series(path, "prompt_length/mean"), _series(path, "response_length/mean")
    pM, rM = _series(path, "prompt_length/max"), _series(path, "response_length/max")
    if not (pm and rm and pM and rM):
        print("   no prompt/response length metrics\n")
        return
    avg = st.median(pm) + st.median(rm)
    worst = max(pM) + max(rM)
    print(f"   average row {st.median(pm):.0f} + {st.median(rm):.0f} = {avg:.0f} tokens")
    print(f"   longest row {max(pM):.0f} + {max(rM):.0f} = {worst:.0f} tokens   ratio {worst/avg:.2f}x")
    for n in sizes:
        print(f"   micro-batch {n:>2}: average {n*avg:7.0f} tok   worst {n*worst:7.0f} tok")
    print("   BALANCE_MINIBATCH_COLUMNS equalises column sums, not the max micro-batch,")
    print("   so nothing bounds the worst case while use_dynamic_bsz=False. A token")
    print("   budget (item 3.2) is what bounds it; ppo_max_token_len_per_gpu is")
    print("   already configured but inert while dynamic bsz is off.\n")


def step1_ab(a, b):
    print("== 6. step-1 A/B (a knob's effect, before steady state diverges) ==")
    for key, lab in (("perf/time_per_step", "step"), ("timing_s/gen", "gen"),
                     ("timing_s/update_actor", "update"), ("timing_s/old_log_prob", "old_logp")):
        va, vb = _series(a, key), _series(b, key)
        if va and vb:
            print(f"   {lab:<9} {va[0]:8.1f} -> {vb[0]:8.1f}   {vb[0]-va[0]:+8.1f}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--turns")
    ap.add_argument("--ab", help="second run; step 1 is compared against --run's")
    a = ap.parse_args(argv)
    phase_breakdown(a.run)
    paired_periodic_test(a.run)
    gen_decomposition(a.turns or a.run)
    allgather_volume(a.run)
    microbatch_token_extremes(a.run)
    if a.ab:
        step1_ab(a.run, a.ab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
