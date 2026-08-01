#!/usr/bin/env python3
"""Check that a teacher pool has the structure Stage 2 assumes, and show its content.

``inspect_teacher_pool.py`` answers "how much and how many" -- sizes, trajectory
counts, padding. This answers "is each row well-formed", which is a different
question and the one that matters after a pool has been regenerated or filtered:
every invariant below is something the trainer relies on without checking, so a
violation does not raise, it trains on nonsense.

    python3 scripts/verify_teacher_pool.py <pool_dir>
    python3 scripts/verify_teacher_pool.py <pool_dir> --shards 3 --samples 2

What it checks, per shard:

  columns    the exact tensor/non-tensor key set the arm's loader leaves behind
  dtypes     int64 throughout; a float mask silently changes what sums to what
  widths     input_ids / attention_mask / position_ids share a length, and
             responses is the trailing max_response_length of it
  responses  responses == input_ids[:, -R:]. Stage 2 never re-derives the answer
             tokens; the SFT loss is an NLL on `responses` scored at the positions
             `input_ids` puts them in, so a mismatch trains on the wrong targets
             and nothing downstream can notice
  padding    attention_mask is left-padded over the prompt and right-padded over
             the response (0*1* then 1*0*). verl's left-truncation assumes it
  positions  position_ids follows the mask over the prompt and then runs straight
             on through the response's right padding, which is what vLLM writes
             and what the rotary embedding is therefore built for
  masked     every row has at least one response token. A row with none
             contributes nothing to the loss but still counts in the per-task
             token totals the SFT weights normalise by
  vocab      token ids inside the tokenizer's range
  grouping   a trajectory's turn-rows are contiguous and carry one task name

Reading the shards is the expensive part (a few GiB each), so by default it reads
one shard per task. Structure is a property of the writer, not of a particular
shard, so that is usually enough; --shards N widens it.

With --samples N it also decodes N rows per task. That is the only check a
program cannot make for you: whether the teacher was actually behaving. Read
them.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from verl import DataProto

_EXPECTED_SFT = {"attention_mask", "input_ids", "position_ids", "responses"}
_KD_EXTRA = {"prompts", "response_mask", "teacher_topk_logprobs", "teacher_topk_ids"}


def _task_of(name):
    stem = name[:-3] if name.endswith(".pt") else name
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, ok, where, message):
        self.checks += 1
        if not ok:
            self.failures.append(f"{where}: {message}")
        return ok


def _check_shard(path, rep, vocab_size):
    name = os.path.basename(path)
    data = DataProto.load_from_disk(path)
    batch, non_tensor = data.batch, data.non_tensor_batch
    keys = set(batch.keys())

    # --- columns -----------------------------------------------------------
    missing = _EXPECTED_SFT - keys
    rep.check(not missing, name, f"missing required column(s) {sorted(missing)}")
    unexpected = keys - _EXPECTED_SFT - _KD_EXTRA
    rep.check(not unexpected, name, f"unexpected column(s) {sorted(unexpected)}")
    for key in ("task_name", "traj_uid"):
        rep.check(key in non_tensor, name, f"missing non-tensor {key!r}")
    if missing or "traj_uid" not in non_tensor:
        return None  # nothing further is meaningful

    for key in sorted(_EXPECTED_SFT):
        rep.check(batch[key].dtype == torch.int64, name,
                  f"{key} has dtype {batch[key].dtype}, expected int64")

    # --- widths ------------------------------------------------------------
    n = len(data)
    seq_len = batch["input_ids"].shape[1]
    resp_len = batch["responses"].shape[1]
    for key in ("attention_mask", "position_ids"):
        rep.check(batch[key].shape == (n, seq_len), name,
                  f"{key} is {tuple(batch[key].shape)}, expected {(n, seq_len)}")
    rep.check(resp_len < seq_len, name,
              f"responses width {resp_len} is not shorter than the sequence {seq_len}")

    am = batch["attention_mask"]
    rep.check(bool(((am == 0) | (am == 1)).all()), name, "attention_mask is not 0/1")

    # --- responses are the tail of input_ids -------------------------------
    tail = batch["input_ids"][:, -resp_len:]
    same = torch.eq(tail, batch["responses"])
    bad = int((~same).any(dim=1).sum())
    rep.check(bad == 0, name,
              f"{bad}/{n} rows where responses != input_ids[:, -{resp_len}:] "
              f"(the SFT loss would score the wrong targets)")

    # --- padding layout ----------------------------------------------------
    # Prompt: left-padded, so its mask is 0*1*. Response: right-padded, 1*0*.
    prompt_mask, resp_mask = am[:, :-resp_len], am[:, -resp_len:]
    # 0*1* <=> the mask is non-decreasing along the row.
    rising = bool((prompt_mask[:, 1:] >= prompt_mask[:, :-1]).all())
    rep.check(rising, name, "prompt attention_mask is not left-padded (0*1*)")
    falling = bool((resp_mask[:, 1:] <= resp_mask[:, :-1]).all())
    rep.check(falling, name, "response attention_mask is not right-padded (1*0*)")

    resp_tokens = resp_mask.sum(dim=1)
    empty = int((resp_tokens == 0).sum())
    rep.check(empty == 0, name, f"{empty}/{n} rows have no response token")

    # --- position_ids ------------------------------------------------------
    # The prompt half follows the mask, but the response half does NOT: vLLM
    # numbers it straight on from the prompt's last position and keeps counting
    # through the right padding, which the generator spells out at
    # vllm_rollout_spmd.py:366 --
    #   attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    #   position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
    # Checking clamp(cumsum(mask)-1) over the whole row instead flags every row
    # whose response did not fill its window -- which is nearly all of them, and
    # is a bug in the check, not the pool.
    prompt_pos = torch.clamp(prompt_mask.cumsum(dim=1) - 1, min=0)
    delta = torch.arange(1, resp_len + 1, dtype=torch.long).unsqueeze(0)
    expected_pos = torch.cat([prompt_pos, prompt_pos[:, -1:] + delta], dim=1)
    pos_bad = int((expected_pos != batch["position_ids"]).any(dim=1).sum())
    rep.check(pos_bad == 0, name,
              f"{pos_bad}/{n} rows where position_ids is neither "
              f"clamp(cumsum(mask)-1) over the prompt nor a straight run over the response")

    # --- vocab range -------------------------------------------------------
    ids = batch["input_ids"]
    lo, hi = int(ids.min()), int(ids.max())
    rep.check(lo >= 0, name, f"input_ids has a negative id ({lo})")
    if vocab_size is not None:
        rep.check(hi < vocab_size, name, f"input_ids max {hi} >= vocab_size {vocab_size}")

    # --- grouping ----------------------------------------------------------
    uids = non_tensor["traj_uid"]
    tasks = non_tensor["task_name"]
    file_task = _task_of(name)
    distinct_tasks = sorted({str(t) for t in tasks})
    rep.check(len(distinct_tasks) == 1, name,
              f"holds more than one task: {distinct_tasks}")
    if len(distinct_tasks) == 1:
        rep.check(file_task in distinct_tasks[0] or distinct_tasks[0] in file_task, name,
                  f"filename says {file_task!r} but rows say {distinct_tasks[0]!r}")

    # Contiguity: a uid must occupy one run of rows, or trajectory-level sampling
    # draws a trajectory whose turns are interleaved with another's.
    change = np.flatnonzero(uids[1:] != uids[:-1]) + 1
    starts = np.concatenate([[0], change])
    rep.check(len(starts) == len(set(uids.tolist())), name,
              f"{len(starts)} contiguous runs for {len(set(uids.tolist()))} trajectories "
              "-- some trajectory's rows are not contiguous")

    turns = np.diff(np.concatenate([starts, [len(uids)]]))
    return {
        "task": distinct_tasks[0] if distinct_tasks else file_task,
        "rows": n,
        "trajs": len(starts),
        "seq_len": seq_len,
        "resp_len": resp_len,
        "turns_min": int(turns.min()), "turns_max": int(turns.max()),
        "turns_mean": float(turns.mean()),
        "resp_tok_mean": float(resp_tokens.to(torch.float64).mean()),
        "resp_tok_max": int(resp_tokens.max()),
        "clipped": float((resp_tokens == resp_len).to(torch.float64).mean()),
        "path": path,
    }


def _decode_samples(path, tokenizer, n_samples):
    data = DataProto.load_from_disk(path)
    am, ids = data.batch["attention_mask"], data.batch["input_ids"]
    resp_len = data.batch["responses"].shape[1]
    uids = data.non_tensor_batch["traj_uid"]

    # First trajectory in the shard, first n_samples turns of it: consecutive
    # turns show whether the teacher was making progress or looping.
    rows = np.flatnonzero(uids == uids[0])[:n_samples]
    out = []
    for r in rows:
        prompt = ids[r, : -resp_len][am[r, : -resp_len].bool()]
        resp = ids[r, -resp_len:][am[r, -resp_len:].bool()]
        out.append((
            tokenizer.decode(prompt.tolist(), skip_special_tokens=False),
            tokenizer.decode(resp.tolist(), skip_special_tokens=False),
        ))
    return str(uids[0]), out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", help="directory of <task>_%%04d.pt shards")
    ap.add_argument("--shards", type=int, default=1,
                    help="shards to verify per task (default 1; structure is a property "
                         "of the writer, so one is usually enough)")
    ap.add_argument("--samples", type=int, default=0,
                    help="decode this many consecutive turns of one trajectory per task. "
                         "The teacher's actual behaviour is the one thing this cannot check.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="tokenizer for --samples and the vocab-range check")
    args = ap.parse_args()

    pool = os.path.abspath(os.path.expanduser(args.pool))
    files = sorted(glob.glob(os.path.join(pool, "*.pt")))
    if not files:
        raise SystemExit(f"no *.pt files in {pool}")

    by_task = defaultdict(list)
    for f in files:
        by_task[_task_of(os.path.basename(f))].append(f)

    vocab_size, tokenizer = None, None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        vocab_size = len(tokenizer)
    except Exception as e:
        print(f"NOTE: no tokenizer ({e}); skipping the vocab-range check "
              f"and --samples\n")

    print(f"{pool}\n{len(files)} shard(s), {len(by_task)} task(s): "
          f"{', '.join(f'{t} x{len(v)}' for t, v in sorted(by_task.items()))}\n")

    rep = Report()
    summaries = []
    for task, task_files in sorted(by_task.items()):
        for path in task_files[: args.shards]:
            print(f"verifying {os.path.basename(path)} ...", flush=True)
            s = _check_shard(path, rep, vocab_size)
            if s:
                summaries.append(s)

    if summaries:
        print(f"\n{'shard':<24} {'rows':>9} {'trajs':>7} {'seq':>6} {'resp':>5} "
              f"{'turns/traj':>18} {'resp tok':>10} {'at cap':>7}")
        print("-" * 96)
        for s in summaries:
            print(f"{os.path.basename(s['path']):<24} {s['rows']:>9,} {s['trajs']:>7,} "
                  f"{s['seq_len']:>6} {s['resp_len']:>5} "
                  f"{s['turns_min']:>5}-{s['turns_max']:<4} ({s['turns_mean']:>5.2f}) "
                  f"{s['resp_tok_mean']:>10.1f} {s['clipped']:>6.1%}")
        print("\n  'at cap' is the fraction of rows whose response fills the whole "
              "max_response_length\n  window: those answers were cut off mid-token, and "
              "the student is trained to reproduce\n  the truncation.")

    if args.samples and tokenizer is not None:
        for task, task_files in sorted(by_task.items()):
            uid, samples = _decode_samples(task_files[0], tokenizer, args.samples)
            print(f"\n{'=' * 78}\n{task}  --  trajectory {uid}\n{'=' * 78}")
            for i, (prompt, resp) in enumerate(samples):
                print(f"\n--- turn {i} PROMPT (tail) " + "-" * 44)
                print(prompt[-1200:])
                print(f"\n--- turn {i} TEACHER RESPONSE " + "-" * 40)
                print(resp)

    print(f"\n{rep.checks} check(s) run")
    if rep.failures:
        print(f"\nFAILED ({len(rep.failures)}):")
        for f in rep.failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("all structural checks passed")
    if not args.samples:
        print("\nStructure is not content. Run with --samples 3 and read the decoded "
              "turns\nbefore committing 30 hours to this pool.")


if __name__ == "__main__":
    main()
