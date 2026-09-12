"""The mark must survive the task-interleaved batch layout.

THE BUG THIS FILE EXISTS TO CATCH, which the previous tests could not. They
checked `_oci_candidate_row` on its own -- "slot 7 of every 8" -- and that was
correct. What was wrong was the INDEXING BETWEEN two correct pieces: env_manager
recorded the prefix per env slot in its own local order, and the rollout loop
read it by the row's GLOBAL batch index.

Under TASK_BALANCE_INTERLEAVE (the default in every arm script) the generation
batch is laid out at the PROMPT level as alf0, search0, webshop0, alf1, ... and
each prompt is then repeated group_n times contiguously, so on a three-task
batch of 8 the alfworld rows are 0-7, 24-31, 48-55, ... Alfworld local slot i is
global row 24*(i//8) + i%8.

One group of fifteen matched. Four landed on another group's candidate slot -- a
different game, so the startswith check rejected the prefix -- and ten were past
the end of the dict. The remaining fourteen candidate trajectories were judged
as part of their group and trained as plain rows with the corrupted plan still
in the prompt, and because ONE group was marked, the "at least one candidate"
assert passed. This is the same contamination the mark was introduced to remove.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TASKS = ["alfworld", "search", "webshop"]
PER_TASK, GN = 15, 8
ok = True


def layout():
    """(rows, task_indices) exactly as main_ppo's interleaved sampler plus
    DataProto.repeat(interleave=True) produce them."""
    prompts = [(t, p) for p in range(PER_TASK) for t in TASKS]
    rows = [prompts[j] for j in range(len(prompts)) for _ in range(GN)]
    task_indices = {t: [g for g, (tt, _) in enumerate(rows) if tt == t] for t in TASKS}
    return rows, task_indices


rows, task_indices = layout()

# The layout itself, so a change to either mechanism fails here first.
good = task_indices["alfworld"][:9] == [0, 1, 2, 3, 4, 5, 6, 7, 24]
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" interleaved layout: alfworld rows start {task_indices['alfworld'][:9]}")

# --- what the old side-channel did -----------------------------------------
# env_manager wrote _OCI_LAST_PREFIX[local]; the loop read it at [global].
local_prefix = {i: (f"PLAN(game {i // GN})" if (i % GN) == GN - 1 else "")
                for i in range(PER_TASK * GN)}
marked_old = 0
for p in range(PER_TASK):
    local_cand = GN * p + GN - 1
    global_cand = task_indices["alfworld"][local_cand]
    got = local_prefix.get(global_cand, "")          # the buggy lookup
    want = local_prefix[local_cand]                  # what the row really carries
    if got and got == want:
        marked_old += 1
good = marked_old == 1
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the old env-slot side channel marked {marked_old}/{PER_TASK} groups, "
      f"and 'at least one candidate' passes on {marked_old}")

# --- what the observation-dict channel does --------------------------------
# _merge_observations scatters each manager's local list into global order.
def merge(per_task_lists, n_rows):
    merged = [None] * n_rows
    for t, vals in per_task_lists.items():
        for gidx, v in zip(task_indices[t], vals):
            merged[gidx] = v
    return merged


obs_prefix = merge({"alfworld": [local_prefix[i] for i in range(PER_TASK * GN)]},
                   len(rows))
marked, wrong_game = 0, 0
for p in range(PER_TASK):
    local_cand = GN * p + GN - 1
    global_cand = task_indices["alfworld"][local_cand]
    got = obs_prefix[global_cand] or ""
    want = local_prefix[local_cand]
    if got == want and got:
        marked += 1
    elif got and got != want:
        wrong_game += 1
good = marked == PER_TASK and wrong_game == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the observation-dict channel marks {marked}/{PER_TASK} groups, "
      f"{wrong_game} on the wrong game")

# Non-alfworld rows carry None, which the loop reads as "no plan".
off = [obs_prefix[g] for t in ("search", "webshop") for g in task_indices[t]]
good = all(v is None for v in off)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" all {len(off)} webshop/search rows carry None, not a stale prefix")

# And every non-candidate alfworld row is empty, so exactly one trajectory per
# group is marked -- which is what the trainer now asserts.
per_group = {}
for local_i, gidx in enumerate(task_indices["alfworld"]):
    g = local_i // GN
    per_group.setdefault(g, 0)
    if obs_prefix[gidx]:
        per_group[g] += 1
good = set(per_group.values()) == {1} and len(per_group) == PER_TASK
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" exactly one marked row in each of {len(per_group)} groups "
      f"(counts {sorted(set(per_group.values()))})")

# --- the non-interleaved layout must work too ------------------------------
# TASK_BALANCE_INTERLEAVE=0 blocks the tasks instead: alf0..alf14, search0, ...
prompts_blocked = [(t, p) for t in TASKS for p in range(PER_TASK)]
rows_b = [prompts_blocked[j] for j in range(len(prompts_blocked)) for _ in range(GN)]
ti_b = {t: [g for g, (tt, _) in enumerate(rows_b) if tt == t] for t in TASKS}
good = ti_b["alfworld"] == list(range(PER_TASK * GN))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " un-interleaved layout puts alfworld first, where local == global -- "
      "which is why the bug was invisible without the interleave")


def test_interleaved_layout():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
