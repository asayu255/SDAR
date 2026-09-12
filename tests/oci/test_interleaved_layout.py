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


# --- THE REAL MERGE AND THE REAL build_text_obs ----------------------------
# Everything above is the layout arithmetic. This part calls the code: the
# alfworld manager's own build_text_obs fills its prefix list, and
# MultiTaskEnvironmentManager._merge_observations scatters it. A simulation
# cannot see a length mismatch between the two, and a length mismatch is
# exactly how zip() loses rows silently.
import types

from agent_system.environments.env_manager import (
    AlfWorldEnvironmentManager, MultiTaskEnvironmentManager, OCI_PREFIX_KEY)

os.environ["PRIVILEGED_WRONG_PLAN"] = "1"
N_ALF = PER_TASK * GN


class FakeAlfEnvs:
    def __init__(self):
        self.group_n = GN
        self.is_train = True
        self.num_processes = N_ALF
        self.get_admissible_commands = [["go to cabinet 1", "help"]] * N_ALF


def fake_alf_manager():
    m = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
    m._oci_prefixes = []
    m.envs = FakeAlfEnvs()
    m.config = types.SimpleNamespace(env=types.SimpleNamespace(history_length=0))
    m.tasks = ["put a clean mug in the countertop"] * N_ALF
    # One distinct game per GROUP, the way ALFWorld seeds seed + i // group_n.
    m.gamefile = [f"/games/g{i // GN}/traj_data.json" for i in range(N_ALF)]
    return m


# The plan builder reads real game files, which are not present here, so stub
# the one function and check the PLUMBING: which slot gets a block, and whether
# the block that reaches a row is the block that row's observation carries.
import agent_system.environments.env_manager as em

_real_builder = em._wrong_plan_prefix
em._wrong_plan_prefix = lambda task, gamefile: (
    f"PLAN[{gamefile}]\n" if gamefile else "")
try:
    mgr = fake_alf_manager()
    text_obs = [f"obs {i}" for i in range(N_ALF)]
    full = mgr.build_text_obs(text_obs, mgr.envs.get_admissible_commands, init=True)

    good = len(mgr._oci_prefixes) == N_ALF == len(full)
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" build_text_obs filled {len(mgr._oci_prefixes)} prefixes for "
          f"{len(full)} observations ({N_ALF} slots)")

    # Every prefix is really at the head of its own observation, which is what
    # the rollout loop's startswith check requires.
    good = all((not pre) or obs.startswith(pre)
               for pre, obs in zip(mgr._oci_prefixes, full))
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          " every non-empty prefix is the head of its own observation")

    marked_local = [i for i, pre in enumerate(mgr._oci_prefixes) if pre]
    good = marked_local == [GN * p + GN - 1 for p in range(PER_TASK)]
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" marked local slots are the last of each group: {marked_local[:4]}...")

    # Now the real merge, with alfworld interleaved among three tasks.
    multi = MultiTaskEnvironmentManager.__new__(MultiTaskEnvironmentManager)
    multi._task_indices = task_indices
    merged = multi._merge_observations(
        {"alfworld": {"text": full, OCI_PREFIX_KEY: list(mgr._oci_prefixes)},
         "search": {"text": ["s"] * N_ALF},
         "webshop": {"text": ["w"] * N_ALF}},
        batch_size=len(rows))

    # What the rollout loop does, on the merged obs, at each global row.
    got_marks = {}
    for gidx in range(len(rows)):
        pres = merged.get(OCI_PREFIX_KEY, None)
        pre = (pres[gidx] or "") if pres is not None and pres[gidx] else ""
        content = merged["text"][gidx] or ""
        if pre and content.startswith(pre):
            got_marks[gidx] = pre

    good = len(got_marks) == PER_TASK
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" through the real merge, {len(got_marks)}/{PER_TASK} rows mark")

    # and each marks the game of ITS OWN group, not another's
    wrong = [g for g, pre in got_marks.items()
             if pre != mgr._oci_prefixes[task_indices["alfworld"].index(g)]]
    good = not wrong
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" {len(wrong)} rows carry another group's plan")

    good = sorted(got_marks) == [task_indices["alfworld"][GN * p + GN - 1]
                                 for p in range(PER_TASK)]
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" the marked global rows are {sorted(got_marks)[:4]}... "
          f"(24p+7, as the layout predicts)")

    # A validation-shaped manager marks nothing, and it has its OWN list, so it
    # cannot blank the training manager's -- which the module-level dict did.
    vmgr = fake_alf_manager()
    vmgr.envs.group_n, vmgr.envs.is_train = 1, False
    vmgr.build_text_obs([f"v {i}" for i in range(N_ALF)],
                        vmgr.envs.get_admissible_commands, init=True)
    good = (not any(vmgr._oci_prefixes)
            and [i for i, p in enumerate(mgr._oci_prefixes) if p] == marked_local)
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          " a validation manager marks nothing and leaves the training "
          "manager's prefixes intact")
finally:
    em._wrong_plan_prefix = _real_builder
    os.environ.pop("PRIVILEGED_WRONG_PLAN", None)


def test_interleaved_layout():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
