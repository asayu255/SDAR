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

Expected padding volume: ``adjust_batch`` pads each gen step to a multiple of
lcm(log_prob_micro*W, actor_micro*W) = 240 rows, so it adds 1..239 rows, ~120 on
average. A gen step produces per_task_batch_size * env.rollout.n = 120
trajectories, so the padding row count should come out close to the trajectory
count -- ``dup_rows / n_trajectories ~= 1.0`` is the headline check.

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

# adjust_batch's divisor for these runs: lcm(log_prob_micro_bs*W, actor_micro_bs*W).
# Only used to sanity-check that padding really was applied; override if the run
# used different micro batch sizes or a different world size.
DEFAULT_BLOCK_DIVISOR = 240

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


def inspect_file(path: str, block_divisor: int, n_spot: int) -> dict:
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
    per_row = {k: t.element_size() * t[0].numel() for k, t in data.batch.items()}
    kd_row = sum(v for k, v in per_row.items() if k not in _DEAD_COLS)
    sft_row = sum(v for k, v in per_row.items() if k not in _DEAD_COLS and k not in _KD_ONLY_COLS)

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
        "block_ok": (n_rows % block_divisor) == 0,
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
        agg["block_ok"] = agg["block_ok"] and r["block_ok"]
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
    ap.add_argument("--block-divisor", type=int, default=DEFAULT_BLOCK_DIVISOR)
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
        r = inspect_file(f, args.block_divisor, args.spot_check)
        rows.append(r)
        if not verbose:
            print(f"  {r['file']:28s} task={r['task']:9s} rows={r['n_rows']:>8,} "
                  f"trajs={r['n_traj']:>7,} pad={r['n_dup']:>6,} "
                  f"block_ok={str(r['block_ok']):5s} spot={r['spot_matched']}/{r['spot_checked']}")
            continue
        print(f"\n--- {os.path.basename(f)}")
        print(f"    task={r['task']}  topk={'yes' if r['has_topk'] else 'no'}")
        print(f"    rows={r['n_rows']:,}  trajectories={r['n_traj']:,}")
        print(f"    padding rows = {r['n_dup']:,}  ({r['n_dup'] / r['n_rows']:.1%} of rows)")
        print(f"    dup_rows / n_traj = {r['dup_over_traj']:.2f}   (expect ~1.0)")
        print(f"    rows % {args.block_divisor} == 0 : {r['block_ok']}   (expect True)")
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
    print(f"   peak adds the largest file, {os.path.basename(biggest)}, held whole while it unpickles)")

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

    bad = [r for r in per_file if not r["block_ok"] or r["spot_matched"] != r["spot_checked"]]
    if bad:
        print("\nWARNING: sanity checks failed for: " + ", ".join(r["file"] for r in bad))
        print("  Do not filter on this result -- the structural assumption may not hold.")


if __name__ == "__main__":
    main()
