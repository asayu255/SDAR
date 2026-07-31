#!/usr/bin/env python3
"""Write the Stage-2 view of a teacher pool to disk, once, so runs stop rebuilding it.

Every run of the off-policy KD or multitask SFT arm reads the whole Stage-1 pool
and throws part of it away before the first optimizer step: the rows an earlier
Stage 1 appended as adjust_batch padding, and the tensor columns that arm's loss
never reads. For the SFT arm that is 339.5 GiB read to keep 139.2 GiB -- the same
2.4x of disk traffic and unpickling on every start, including every restart after
a crash.

This runs that filtering once and saves the result. Afterwards, point the arm at
the cache instead:

    python3 scripts/cache_teacher_pool.py \\
        $HOME/data/verl-agent/sdar_multitask/teacher_traj \\
        $HOME/data/verl-agent/sdar_multitask/teacher_traj_sft_cache --arm sft

    bash examples/sft_trainer/run_multitask_sft_qwen3.sh \\
        +algorithm.sft.data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_sft_cache

Nothing about the run changes. The cache holds exactly the DataProto the loader
builds in memory today, one output file per input file with the same basename and
the same row order, so sorted(glob) yields the same files in the same order and
the trajectory sampling population -- and with it every draw -- is identical.
``_load_offpolicy_file`` then finds no padding to drop and no listed column to
remove, and passes the data through unchanged.

The cache is ARM-SPECIFIC. --arm sft drops the teacher top-k columns, which the KD
loss needs; a KD run must not be pointed at an SFT cache. The written columns are
printed so this is visible, and the arm is recorded in the directory's manifest.
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time

# Running this as a plain script puts scripts/ on sys.path[0], not the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MANIFEST = "_cache_manifest.json"


def _loader_for(arm: str):
    """The trainer class whose _load_offpolicy_file defines this arm's view.

    Imported lazily: verl drags in ray/vllm, which is slow and unrelated to
    argument parsing.
    """
    if arm == "sft":
        from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer

        return MultiTaskSFTTrainer
    if arm == "kd":
        from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer

        return OffPolicyOPDRayTrainer
    raise ValueError(f"unknown arm {arm!r}")


def _gib(n):
    return n / 1024**3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory of Stage-1 <task>.pt / <task>_0000.pt files")
    ap.add_argument("dst", help="directory to write the filtered pool into (created if absent)")
    ap.add_argument(
        "--arm",
        choices=["sft", "kd"],
        required=True,
        help="which arm's view to cache. sft additionally drops the teacher top-k columns.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave already-written outputs alone, so an interrupted run can resume",
    )
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    dst = os.path.abspath(os.path.expanduser(args.dst))
    if src == dst:
        raise SystemExit("src and dst must differ -- this never overwrites the source pool")

    files = sorted(glob.glob(os.path.join(src, "*.pt")))
    if not files:
        raise SystemExit(f"no *.pt files in {src}")
    src_bytes = sum(os.path.getsize(f) for f in files)

    os.makedirs(dst, exist_ok=True)
    free = shutil.disk_usage(dst).free
    # The filtered pool is smaller than the source, so the source size is a safe
    # upper bound -- and checking before a multi-hour pass is cheaper than failing
    # in the middle of it.
    print(f"source: {len(files)} file(s), {_gib(src_bytes):.1f} GiB")
    print(f"target: {dst}  ({_gib(free):.1f} GiB free)")
    if free < src_bytes:
        print(f"  NOTE: free space is below the source size. The output is smaller "
              f"(the SFT view is ~0.41x), but if the pass runs out of disk it will "
              f"stop partway; --skip-existing lets you resume after freeing space.")

    loader = _loader_for(args.arm)
    print(f"arm: {args.arm} ({loader.__name__}), dropping columns {list(loader._drop_tensor_keys)}\n")

    written = skipped = 0
    out_bytes = 0
    rows_in = rows_out = 0
    columns = None
    t0 = time.time()
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        out_path = os.path.join(dst, name)
        if args.skip_existing and os.path.exists(out_path):
            out_bytes += os.path.getsize(out_path)
            skipped += 1
            print(f"[{i}/{len(files)}] {name}: exists, skipped")
            continue

        # The same call the trainer makes, so what lands on disk is exactly what
        # the trainer would have held in memory.
        data = loader._load_offpolicy_file(path)
        n_rows = len(data)
        rows_out += n_rows
        if columns is None:
            columns = sorted(data.batch.keys())
        elif sorted(data.batch.keys()) != columns:
            raise SystemExit(
                f"{name} has columns {sorted(data.batch.keys())}, earlier files had "
                f"{columns}; the pool is not uniform and a run would fail on it"
            )
        # Write beside the target and rename, so an interrupted write never leaves a
        # truncated file that --skip-existing would then accept.
        tmp_path = out_path + ".partial"
        data.save_to_disk(tmp_path)
        os.replace(tmp_path, out_path)
        del data
        gc.collect()

        size = os.path.getsize(out_path)
        out_bytes += size
        written += 1
        done = _gib(sum(os.path.getsize(f) for f in files[:i]))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (_gib(src_bytes) - done) / rate if rate else float("nan")
        print(f"[{i}/{len(files)}] {name}: {_gib(os.path.getsize(path)):.2f} -> "
              f"{_gib(size):.2f} GiB, {n_rows:,} rows, "
              f"{rate:.2f} GiB/s, ETA {eta / 60:.0f} min", flush=True)

    with open(os.path.join(dst, _MANIFEST), "w") as f:
        json.dump(
            {
                "source": src,
                "arm": args.arm,
                "loader": loader.__name__,
                "dropped_columns": list(loader._drop_tensor_keys),
                "columns": columns,
                "files": len(files),
                "rows": rows_out,
            },
            f,
            indent=2,
        )

    print(f"\nwrote {written} file(s) ({skipped} skipped), {rows_out:,} rows")
    print(f"  {_gib(src_bytes):7.1f} GiB in  ->  {_gib(out_bytes):7.1f} GiB out "
          f"({out_bytes / src_bytes:.2f}x)")
    print(f"  columns kept: {columns}")
    print(f"  elapsed {(time.time() - t0) / 60:.0f} min")
    print(f"\nPoint the {args.arm} arm at:\n  {dst}")
    if args.arm == "sft":
        print("  (SFT view: the teacher top-k is gone, so a KD run must NOT read this.)")


if __name__ == "__main__":
    main()
