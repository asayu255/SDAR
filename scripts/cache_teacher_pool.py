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
        $HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache --arm kd

    bash examples/opd_trainer/run_multitask_offpolicy_qwen3_nogen.sh \\
        ++algorithm.opd.teacher_data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache

Double plus: the run script already adds that key with a single '+', so another
'+' is a second append of an existing key and Hydra refuses it.

Nothing about the run changes. The cache holds exactly the DataProto the loader
builds in memory today, one output file per input file with the same basename and
the same row order, so sorted(glob) yields the same files in the same order and
the trajectory sampling population -- and with it every draw -- is identical.
``_load_offpolicy_file`` then finds no padding to drop and no listed column to
remove, and passes the data through unchanged.

The cache is ARM-SPECIFIC. --arm sft drops the teacher top-k columns, which the KD
loss needs; a KD run must not be pointed at an SFT cache. The written columns are
printed so this is visible, and the arm is recorded in the directory's manifest.
This branch carries only the KD arm, so --arm sft is available solely to build a
cache for the SFT branch to read; it cannot be used by anything here.

When one task has been regenerated, rebuild just that task:

    python3 scripts/cache_teacher_pool.py <src> <dst> --arm kd --only webshop

The other tasks' outputs and their manifest row counts are left untouched, and any
output shard the source no longer has is deleted -- the trainer globs the whole
directory, so a stale shard left behind would be trained on beside the new ones.
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


def _task_of(name):
    """Task name behind a pool filename: webshop_0007.pt and webshop.pt -> webshop.

    Stage 1 writes one file per task, or ``<task>_%04d.pt`` when it shards, so the
    numeric suffix is the only thing separating a shard from the task it belongs to.
    """
    stem = name[:-3] if name.endswith(".pt") else name
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


def _loader_for(arm: str):
    """The trainer class whose _load_offpolicy_file defines this arm's view.

    Imported lazily: verl drags in ray/vllm, which is slow and unrelated to
    argument parsing.
    """
    if arm == "sft":
        try:
            from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer
        except ImportError as e:
            # This branch is the KD arm only; the SFT trainer lives on the SFT
            # branch. Build an SFT cache there rather than guessing its view here.
            raise SystemExit(
                "--arm sft needs verl.trainer.ppo.sft_multitask_ray_trainer, which "
                f"this branch does not carry ({e}). Use --arm kd here."
            )

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
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="TASK",
        help="rebuild only these tasks (e.g. --only webshop). Other tasks' outputs and "
             "their manifest row counts are left exactly as they are.",
    )
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    dst = os.path.abspath(os.path.expanduser(args.dst))
    if src == dst:
        raise SystemExit("src and dst must differ -- this never overwrites the source pool")

    all_files = sorted(glob.glob(os.path.join(src, "*.pt")))
    if not all_files:
        raise SystemExit(f"no *.pt files in {src}")

    # --only narrows what is read and written. Everything downstream that describes
    # the *directory* -- the manifest, the stale-output sweep -- still works from the
    # full source listing, so a partial pass never claims the cache is smaller than
    # it is.
    if args.only:
        tasks = {_task_of(os.path.basename(f)) for f in all_files}
        unknown = sorted(set(args.only) - tasks)
        if unknown:
            raise SystemExit(
                f"--only names task(s) not in {src}: {unknown}. Present: {sorted(tasks)}"
            )
        selected = set(args.only)
        files = [f for f in all_files if _task_of(os.path.basename(f)) in selected]
    else:
        selected = None
        files = all_files
    src_bytes = sum(os.path.getsize(f) for f in files)

    os.makedirs(dst, exist_ok=True)
    free = shutil.disk_usage(dst).free
    # The filtered pool is smaller than the source, so the source size is a safe
    # upper bound -- and checking before a multi-hour pass is cheaper than failing
    # in the middle of it.
    scope = "" if selected is None else f" of {len(all_files)} (--only {' '.join(sorted(selected))})"
    print(f"source: {len(files)} file(s){scope}, {_gib(src_bytes):.1f} GiB")
    print(f"target: {dst}  ({_gib(free):.1f} GiB free)")
    if free < src_bytes:
        print(f"  NOTE: free space is below the source size. The output is smaller "
              f"(the SFT view is ~0.41x), but if the pass runs out of disk it will "
              f"stop partway; --skip-existing lets you resume after freeing space.")

    loader = _loader_for(args.arm)
    print(f"arm: {args.arm} ({loader.__name__}), dropping columns {list(loader._drop_tensor_keys)}\n")

    # A task regenerated into fewer shards leaves its extra outputs behind, and the
    # trainer globs the whole directory -- it would load those stale shards beside
    # the new ones and train on a mix of two generations without noticing (the
    # duplicate-traj_uid assert does not fire, because the trajectories differ).
    # Sweep them before writing anything.
    present = {os.path.basename(f) for f in all_files}
    for out in sorted(glob.glob(os.path.join(dst, "*.pt"))):
        name = os.path.basename(out)
        if name in present:
            continue
        if selected is not None and _task_of(name) not in selected:
            continue  # not this pass's business
        os.remove(out)
        print(f"removed stale output {name} (no such file in the source)")

    # Row counts are kept per file and merged with any existing manifest, so a
    # resumed or partial pass (regenerating one task, then --skip-existing) still
    # describes the whole directory rather than only what this run touched.
    manifest_path = os.path.join(dst, _MANIFEST)
    rows_by_file = {}
    previous = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                previous = json.load(f)
            if previous.get("arm") == args.arm:
                rows_by_file = dict(previous.get("rows_by_file", {}))
        except (OSError, ValueError):
            pass  # unreadable manifest is not worth failing over; it gets rewritten

    written = skipped = 0
    out_bytes = 0
    rows_out = 0
    columns = None
    t0 = time.time()
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        out_path = os.path.join(dst, name)
        if args.skip_existing and os.path.exists(out_path):
            # Skipping is for resuming an interrupted pass, not for keeping a stale
            # answer. A source file rewritten since its cache entry (a task
            # regenerated) must be redone, or the cache silently stops matching the
            # pool it claims to be a view of.
            if os.path.getmtime(path) > os.path.getmtime(out_path):
                print(f"[{i}/{len(files)}] {name}: source is newer than the cached "
                      f"copy, rebuilding", flush=True)
            else:
                out_bytes += os.path.getsize(out_path)
                skipped += 1
                note = "" if name in rows_by_file else "  (no row count recorded)"
                print(f"[{i}/{len(files)}] {name}: exists, skipped{note}")
                continue

        # The same call the trainer makes, so what lands on disk is exactly what
        # the trainer would have held in memory.
        data = loader._load_offpolicy_file(path)
        n_rows = len(data)
        rows_by_file[name] = n_rows
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

    # Only files still present in the source belong in the manifest: a task
    # regenerated into fewer shards leaves the extra ones behind here otherwise.
    # This is measured against the whole source, not this pass's selection, so
    # --only keeps describing the whole directory.
    rows_by_file = {k: v for k, v in rows_by_file.items() if k in present}
    rows_out = sum(rows_by_file.values())
    complete = len(rows_by_file) == len(all_files)
    if columns is None:  # everything was skipped
        columns = previous.get("columns")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "source": src,
                "arm": args.arm,
                "loader": loader.__name__,
                "dropped_columns": list(loader._drop_tensor_keys),
                "columns": columns,
                "files": len(all_files),
                "rows": rows_out,
                "rows_complete": complete,
                "rows_by_file": rows_by_file,
            },
            f,
            indent=2,
        )

    total = (f"{rows_out:,} rows across {len(all_files)} file(s)" if complete
             else f"{rows_out:,} rows (partial: {len(rows_by_file)}/{len(all_files)} files counted)")
    print(f"\nwrote {written} file(s) ({skipped} skipped), cache now holds {total}")
    print(f"  {_gib(src_bytes):7.1f} GiB in  ->  {_gib(out_bytes):7.1f} GiB out "
          f"({out_bytes / src_bytes:.2f}x)" + ("" if selected is None else "   [this pass only]"))
    print(f"  columns kept: {columns}")
    print(f"  elapsed {(time.time() - t0) / 60:.0f} min")
    print(f"\nPoint the {args.arm} arm at:\n  {dst}")
    if args.arm == "sft":
        print("  (SFT view: the teacher top-k is gone, so a KD run must NOT read this.)")


if __name__ == "__main__":
    main()
