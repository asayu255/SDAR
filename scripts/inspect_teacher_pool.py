#!/usr/bin/env python3
"""Inspect a Stage-1 teacher-trajectory pool (``<task>.pt`` / ``<task>_0000.pt``).

Read-only. Files are read one at a time and released, so this never holds more
than one of them; a task written as shards is folded back into one row of the
summary. One pass per file reports, per task:

* the rows appended by Stage-1's ``adjust_batch(mode="copy")`` padding, which are
  written into the dataset and therefore trained on more than once;
* ``r_k`` -- mean turn-rows per trajectory (sets the per-task pool size when the
  per-step draw is changed);
* ``l_k`` -- mean full sequence length per row (prompt+response; the load-balance
  cost driver, since the forward is over the whole sequence);
* mean response tokens per row and the per-task response-token share (the
  quantity a per-task loss normalisation equalises);
* the host RAM each Stage-2 arm will hold, measured from the pool's own columns
  rather than assumed -- resident, and the peak while the largest file unpickles.

Padding rows are identified structurally, not by hashing. A Stage-1 block is laid
out trajectory-major -- ``gather_rollout_data`` iterates trajectories in the outer
loop and turns in the inner one -- so every trajectory's turn-rows are contiguous.
``adjust_batch`` concatenates its duplicated rows after all of them, so a traj_uid
starting a *second* run can only be padding.

Expected padding volume: ``adjust_batch`` pads each gen step up to a multiple of
a divisor set by the generating run's micro batch sizes and world size, so it
adds 1..divisor-1 rows, ~divisor/2 on average. The pool does not record that
divisor, so it is inferred here (see ``infer_block_divisor``) and the mean
padding per step is reported against it rather than against a hardcoded guess.

The check that actually matters is the bit-identity spot check: a flagged row
must be byte-for-byte a row already present in its own trajectory. If that ever
fails, the rows being dropped are real turns, not copies.

Usage:
    python3 scripts/inspect_teacher_pool.py $HOME/data/verl-agent/sdar_multitask/teacher_traj
    python3 scripts/inspect_teacher_pool.py <dir> --sizes-only      # disk sizes, no load
"""

import argparse
import gc
import glob
import os
import re
import sys
from functools import reduce
from math import gcd

# Stage-1 shard names: <task>_0000.pt (gen.shard_every_steps).
_SHARD_RE = re.compile(r"^.+_\d{4}\.pt$")

# Running this as a plain script puts scripts/ on sys.path[0], not the repo root,
# so `verl` is not importable unless the package happens to be pip-installed into
# the active env. Put the repo root on the path ourselves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

# verl is imported lazily inside inspect_file: it drags in ray/vllm, which is slow
# and can fail for reasons unrelated to this script, and --sizes-only never needs it.

# adjust_batch pads each generation step up to a multiple of
# lcm(size_divisor_ref, size_divisor_rollout, size_divisor_actor) -- see
# agent_system/multi_turn_rollout/utils.py:168. That value depends on the micro
# batch sizes and world size the GENERATING run used, which this script cannot
# read: the pool does not record them. So it is inferred from the data instead of
# assumed. Every step's row count is a multiple of the divisor, a shard is whole
# steps, so the gcd over the shards is a multiple of it -- and the mean padding
# per step confirms the inference (a uniform remainder gives ~divisor/2).
# Hardcoding a guess here is what made an earlier version flag 63 of 90 healthy
# shards. Pass --block-divisor only to check against a value you know.

# Columns the Stage-2 loaders drop, mirroring OffPolicyOPDRayTrainer._drop_tensor_keys
# and MultiTaskSFTTrainer's extension of it. Only used to report what each arm will
# hold resident; nothing here changes what training does.
_DEAD_COLS = ("prompts", "response_mask")
_KD_ONLY_COLS = ("teacher_topk_logprobs", "teacher_topk_ids")


def _first_run_slice(uids: np.ndarray, uid) -> tuple:
    """(start, end) of uid's first contiguous run."""
    idx = np.flatnonzero(uids == uid)
    start = int(idx[0])
    end = start
    while end + 1 < len(uids) and uids[end + 1] == uid:
        end += 1
    return start, end + 1


def spot_check_bit_identity(data, is_dup: np.ndarray, n_check: int) -> tuple:
    """Verify flagged rows really are copies of a row in the same trajectory.

    Returns (checked, matched). A flagged row that matches nothing is a sign the
    structural assumption does not hold for this file.
    """
    uids = np.asarray(data.non_tensor_batch["traj_uid"])
    flagged = np.flatnonzero(is_dup)
    if len(flagged) == 0:
        return 0, 0
    rng = np.random.default_rng(0)
    sample = rng.choice(flagged, size=min(n_check, len(flagged)), replace=False)
    input_ids = data.batch["input_ids"]

    checked = matched = 0
    for row in sample.tolist():
        start, end = _first_run_slice(uids, uids[row])
        checked += 1
        for cand in range(start, end):
            if torch.equal(input_ids[row], input_ids[cand]):
                matched += 1
                break
    return checked, matched


def inspect_file(path: str, n_spot: int) -> dict:
    from verl import DataProto

    # The same function OffPolicyOPDRayTrainer._load_offpolicy_file filters with, so
    # what this reports is exactly what training will drop.
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import find_padding_duplicates

    data = DataProto.load_from_disk(path)
    n_rows = len(data)
    uids = np.asarray(data.non_tensor_batch["traj_uid"])
    n_traj = len(set(uids.tolist()))

    is_dup = find_padding_duplicates(uids)
    n_dup = int(is_dup.sum())
    keep = ~is_dup
    n_keep = int(keep.sum())

    attention_mask = data.batch["attention_mask"]
    if "response_mask" in data.batch:
        response_mask = data.batch["response_mask"]
    else:
        resp_len = data.batch["responses"].size(1)
        response_mask = attention_mask[:, -resp_len:]

    # Reduce to per-row counts BEFORE masking. Masking the (rows, seq_len) tensors
    # directly would materialise a second copy of a file that is already tens of
    # GiB; the per-row vectors are a few MiB.
    keep_t = torch.from_numpy(keep)
    full_per_row = attention_mask.sum(-1)
    resp_per_row_t = response_mask.sum(-1)
    full_tokens = int(full_per_row[keep_t].sum())
    resp_tokens = int(resp_per_row_t[keep_t].sum())
    dup_resp_tokens = int(resp_per_row_t[~keep_t].sum()) if n_dup else 0

    checked, matched = spot_check_bit_identity(data, is_dup, n_spot)

    task = data.non_tensor_batch["task_name"][0] if "task_name" in data.non_tensor_batch else "?"
    has_topk = "teacher_topk_logprobs" in data.batch

    # Bytes one row costs, per column, so the RAM the loaders will actually hold
    # can be reported rather than guessed. Both arms drop the same dead columns;
    # the SFT arm drops the teacher top-k on top, because its NLL never reads it.
    # The KD loader also narrows storage dtypes at load (lossless -- see
    # _POOL_STORE_DTYPES in opd_offpolicy_ray_trainer), so its resident bytes are
    # counted at the NARROWED width, not the width stored on disk. A pool already
    # narrow (a cache) divides by 1 and reports the same number.
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import _POOL_STORE_DTYPES

    narrowed_row = {}
    for k, t in data.batch.items():
        store = _POOL_STORE_DTYPES.get(k)
        itemsize = torch.tensor([], dtype=store).element_size() if store is not None else t.element_size()
        narrowed_row[k] = itemsize * t[0].numel()
    kd_row = sum(v for k, v in narrowed_row.items() if k not in _DEAD_COLS)
    sft_row = sum(v for k, v in narrowed_row.items() if k not in _DEAD_COLS and k not in _KD_ONLY_COLS)

    del data, attention_mask, response_mask, keep_t, full_per_row, resp_per_row_t
    gc.collect()

    return {
        "file": os.path.basename(path),
        "task": str(task),
        "has_topk": has_topk,
        "n_rows": n_rows,
        "n_keep": n_keep,
        "n_dup": n_dup,
        "n_traj": n_traj,
        "dup_over_traj": n_dup / n_traj if n_traj else float("nan"),
        "r_k": n_keep / n_traj if n_traj else float("nan"),
        "l_k": full_tokens / n_keep if n_keep else float("nan"),
        "resp_per_row": resp_tokens / n_keep if n_keep else float("nan"),
        "full_tokens": full_tokens,
        "resp_tokens": resp_tokens,
        "dup_resp_tokens": dup_resp_tokens,
        "kd_row_bytes": kd_row,
        "sft_row_bytes": sft_row,
        "disk_bytes": os.path.getsize(path),
        "spot_checked": checked,
        "spot_matched": matched,
    }


def infer_block_divisor(per_file: list, override) -> tuple:
    """adjust_batch's divisor, inferred from the shards unless one is given.

    Every generation step is padded up to a multiple of it and a shard is whole
    steps, so it divides every shard's row count -- the gcd is the tightest value
    the data can justify. Reported rather than assumed because the pool does not
    record the micro batch sizes and world size that determined it, and a
    hardcoded guess turns this check into noise the moment a run uses different
    ones.
    """
    if override:
        return int(override), "given with --block-divisor"
    counts = [r["n_rows"] for r in per_file]
    if not counts:
        return 0, "no files"
    divisor = reduce(gcd, counts)
    return divisor, f"gcd over {len(counts)} file(s)"


def merge_by_task(rows: list) -> list:
    """Fold a task's shards into one record.

    Stage 1 writes ``<task>_0000.pt`` shards when gen.shard_every_steps is set, so
    a pool is 90 files, not 3. Every quantity here is either a sum or a ratio of
    sums, and shards never split a trajectory, so folding them is exact.
    """
    merged = {}
    for r in rows:
        agg = merged.get(r["task"])
        if agg is None:
            merged[r["task"]] = agg = dict(r, files=1)
            continue
        agg["files"] += 1
        for key in ("n_rows", "n_keep", "n_dup", "n_traj", "full_tokens",
                    "resp_tokens", "dup_resp_tokens", "disk_bytes",
                    "spot_checked", "spot_matched"):
            agg[key] += r[key]
        agg["has_topk"] = agg["has_topk"] and r["has_topk"]
        agg["file"] = f"{r['task']}_* ({agg['files']} shards)"
    for agg in merged.values():
        n_traj, n_keep = agg["n_traj"], agg["n_keep"]
        agg["dup_over_traj"] = agg["n_dup"] / n_traj if n_traj else float("nan")
        agg["r_k"] = n_keep / n_traj if n_traj else float("nan")
        agg["l_k"] = agg["full_tokens"] / n_keep if n_keep else float("nan")
        agg["resp_per_row"] = agg["resp_tokens"] / n_keep if n_keep else float("nan")
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help="directories of Stage-1 <task>.pt files and/or individual .pt files. "
             "Naming files explicitly is how you leave out a task whose generation "
             "is still running -- its .pt is either stale or half-written.",
    )
    ap.add_argument("--sizes-only", action="store_true", help="print disk sizes and exit")
    ap.add_argument("--block-divisor", type=int, default=None,
                    help="adjust_batch's divisor, if you know it. Default: infer it from the data.")
    ap.add_argument("--spot-check", type=int, default=20, help="flagged rows to verify bit-identity on")
    args = ap.parse_args()

    files = []
    for path in args.paths:
        if os.path.isfile(path):
            files.append(path)
        else:
            files.extend(sorted(glob.glob(os.path.join(path, "*.pt"))))
    if not files:
        raise SystemExit(f"no *.pt files in {' '.join(args.paths)}")

    total_disk = sum(os.path.getsize(f) for f in files)
    biggest = max(files, key=os.path.getsize)
    print(f"{len(files)} file(s), {total_disk / 1024**3:.1f} GiB on disk:")
    # One line per file is unreadable for a sharded pool (30 shards/task); group.
    by_stem = {}
    for f in files:
        stem = os.path.basename(f).rsplit("_", 1)[0] if _SHARD_RE.match(os.path.basename(f)) else os.path.basename(f)
        by_stem.setdefault(stem, []).append(f)
    for stem, group in by_stem.items():
        size = sum(os.path.getsize(f) for f in group)
        label = f"{stem} ({len(group)} shards)" if len(group) > 1 else stem
        print(f"  {label:32s} {size / 1024**3:8.1f} GiB")
    print(f"  largest single file: {os.path.basename(biggest)} "
          f"({os.path.getsize(biggest) / 1024**3:.1f} GiB)")
    print("  NOTE: DataProto is a plain pickle, so a file is loaded whole into host RAM.")
    print("        Stage 2 keeps every file resident (each step samples from the whole")
    print("        pool) but never concatenates them, so its peak is roughly")
    print("        'resident + largest single file' -- which is why sharding matters.")
    if args.sizes_only:
        return

    verbose = len(files) <= 6
    rows = []
    for f in files:
        r = inspect_file(f, args.spot_check)
        rows.append(r)
        if not verbose:
            print(f"  {r['file']:28s} task={r['task']:9s} rows={r['n_rows']:>8,} "
                  f"trajs={r['n_traj']:>7,} pad={r['n_dup']:>6,} "
                  f"spot={r['spot_matched']}/{r['spot_checked']}")
            continue
        print(f"\n--- {os.path.basename(f)}")
        print(f"    task={r['task']}  topk={'yes' if r['has_topk'] else 'no'}")
        print(f"    rows={r['n_rows']:,}  trajectories={r['n_traj']:,}")
        print(f"    padding rows = {r['n_dup']:,}  ({r['n_dup'] / r['n_rows']:.1%} of rows)")
        print(f"    dup_rows / n_traj = {r['dup_over_traj']:.2f}   (expect ~1.0)")
        print(f"    bit-identity spot check: {r['spot_matched']}/{r['spot_checked']} matched "
              f"(expect all)")
        print(f"    r_k (rows/traj, padding excluded) = {r['r_k']:.2f}")
        print(f"    l_k (full tokens/row)             = {r['l_k']:.1f}")
        print(f"    response tokens/row               = {r['resp_per_row']:.1f}")

    per_file = rows
    rows = merge_by_task(rows)
    total_resp = sum(r["resp_tokens"] for r in rows)
    print("\n===== summary (per task, shards folded) =====")
    header = f"{'task':10s} {'trajs':>8s} {'rows':>10s} {'dup':>9s} {'dup%':>6s} " \
             f"{'r_k':>7s} {'l_k':>8s} {'resp/row':>9s} {'resp share':>11s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        share = r["resp_tokens"] / total_resp if total_resp else float("nan")
        print(f"{r['task']:10s} {r['n_traj']:8,d} {r['n_rows']:10,d} {r['n_dup']:9,d} "
              f"{r['n_dup'] / r['n_rows']:5.1%} {r['r_k']:7.2f} {r['l_k']:8.1f} "
              f"{r['resp_per_row']:9.1f} {share:10.1%}")

    print("\n===== host RAM Stage 2 will hold (padding rows already excluded) =====")
    kd = sum(r["n_keep"] * r["kd_row_bytes"] for r in rows) / 1024**3
    sft = sum(r["n_keep"] * r["sft_row_bytes"] for r in rows) / 1024**3
    big = os.path.getsize(biggest) / 1024**3
    print(f"  off-policy KD arm : {kd:7.2f} GiB resident, peak ~{kd + big:.2f} GiB")
    print(f"  multitask SFT arm : {sft:7.2f} GiB resident, peak ~{sft + big:.2f} GiB")
    print(f"  (SFT is lower because its NLL never reads {'/'.join(_KD_ONLY_COLS)};")
    print(f"   peak adds the largest file, {os.path.basename(biggest)}, held whole while it unpickles.")
    print(f"   Both numbers assume THIS branch's loader, which narrows storage dtypes at")
    print(f"   load [lossless]; a branch whose loader keeps the stored int64 holds ~1.9x.)")

    print("\nInterpretation")
    if len(rows) < 3:
        print(f"  NOTE: only {len(rows)} task(s) inspected, so 'resp share' is a share of")
        print("        those alone. Re-run with every task's .pt for the real shares.")
    print("  resp share  : each task's share of response tokens in the pool. With the")
    print("                current global token-mean loss this is its share of the")
    print("                gradient; a per-task normalisation drives all of them to 1/3.")
    print("  l_k         : per-task forward cost per row. max(l_k)/mean(l_k) - 1 is the")
    print("                slowdown of a layout whose ranks hold different tasks at the")
    print("                same backward.")
    print("  r_k         : rows per trajectory. If the per-step draw is changed to give")
    print("                each task equal rows, pool sizes must scale as 1/r_k to keep")
    print("                the epoch count equal across tasks.")

    print("\n===== padding sanity check =====")
    divisor, why = infer_block_divisor(per_file, args.block_divisor)
    print(f"  adjust_batch divisor: {divisor} ({why})")
    for r in rows:
        blocks = r["n_rows"] / divisor if divisor else float("nan")
        # Each generation step is padded by 1..divisor-1 rows, ~divisor/2 on
        # average, so the padding volume recovers how many steps wrote this task.
        # It should come out at the generating run's step count; well under it
        # means many steps landed on the grid and needed no padding (short-episode
        # tasks do), well over it means the padding is not what it looks like.
        implied_steps = 2 * r["n_dup"] / divisor if divisor else float("nan")
        print(f"  {r['task']:10s} rows/{divisor} = {blocks:10.2f} "
              f"{'OK' if float(blocks).is_integer() else '<-- NOT WHOLE'}   "
              f"padding {r['n_dup']:>7,} -> ~{implied_steps:.0f} generation steps")
    total_checked = sum(r["spot_checked"] for r in per_file)
    total_matched = sum(r["spot_matched"] for r in per_file)
    print(f"  bit-identity spot check: {total_matched}/{total_checked} flagged rows matched "
          f"a row in their own trajectory")

    bad = [r for r in per_file if r["spot_matched"] != r["spot_checked"]]
    off_grid = [r for r in per_file if divisor and r["n_rows"] % divisor]
    if bad:
        print("\nWARNING: flagged rows that are NOT copies, in: "
              + ", ".join(r["file"] for r in bad))
        print("  Do not filter on this result -- the structural assumption does not hold")
        print("  for these files, so find_padding_duplicates would drop real turns.")
    elif off_grid:
        print(f"\nWARNING: {len(off_grid)} file(s) are not a multiple of the divisor "
              f"{divisor}, which was given with --block-divisor. Either the generating")
        print("  run used a different one (drop the flag and let it be inferred), or the")
        print("  directory mixes shards from runs with different batch geometry.")
    else:
        print("  -> padding detection is sound; those rows are dropped at load.")


if __name__ == "__main__":
    main()
