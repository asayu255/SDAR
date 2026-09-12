"""CPU test: the candidate mask is per trajectory, and the class split has a
reference level that survives a batch with no live group.

Both were wrong before. The mask counted ROWS inside a uid, so on multi-turn it
marked rows out of the middle of arbitrary trajectories; the reference level
averaged every return when nothing was live, which labels an all-failed batch
saturated and fires the wrong arm on all of it.
"""
import os, sys, types
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.metric_utils import _reference_level

GN = 8


def derive(turns_per_traj, n_groups=2, gn=GN, drop_last_traj=False):
    """The mask exactly as opd_grpo_ray_trainer derives it, on a batch whose
    trajectories are ``turns_per_traj`` rows long."""
    uids, tuids = [], []
    for g in range(n_groups):
        n_traj = gn - 1 if (drop_last_traj and g == 0) else gn
        for t in range(n_traj):
            for _ in range(turns_per_traj):
                uids.append(f"g{g}")
                tuids.append(f"g{g}:t{t}")
    order, seen = {}, {}
    for u, t in zip(uids, tuids):
        key = (str(u), str(t))
        if key not in order:
            order[key] = seen.get(str(u), 0)
            seen[str(u)] = order[key] + 1
    full = {u: n for u, n in seen.items() if n == gn}
    mask = [bool(str(u) in full and order[(str(u), str(t))] == gn - 1)
            for u, t in zip(uids, tuids)]
    return np.array(mask), np.array(tuids), seen, full


ok = True

# 1. Single-turn: the old row-counting rule and the new one agree, so the fix
#    cannot have broken the case that used to work.
mask, tuids, _, _ = derive(1)
old = np.array([(i % GN) == (GN - 1) for i in range(len(mask))])
good = bool((mask == old).all()) and int(mask.sum()) == 2
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" single-turn: new rule matches the old one, {int(mask.sum())} rows marked")

# 2. Multi-turn: exactly one trajectory per group, all of its rows.
TURNS = 3
mask, tuids, _, _ = derive(TURNS)
marked = sorted(set(tuids[mask]))
good = (marked == [f"g0:t{GN-1}", f"g1:t{GN-1}"]
        and int(mask.sum()) == 2 * TURNS)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" multi-turn: whole trajectories marked, {marked}, {int(mask.sum())} rows")

# 3. What the old rule did on the same batch: not the last trajectory at all.
old = np.array([(i % GN) == (GN - 1) for i in range(len(mask))])
old_marked = sorted(set(tuids[old]))
good = old_marked != marked and len(old_marked) > 2
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the old rule marked {len(old_marked)} different trajectories "
      f"({old_marked[:3]}...), which is the bug")

# 4. A short group has no trustworthy last slot, so nothing in it is a candidate.
mask, tuids, seen, full = derive(TURNS, drop_last_traj=True)
good = (not any(t.startswith("g0:") for t in set(tuids[mask]))
        and sorted(set(tuids[mask])) == [f"g1:t{GN-1}"]
        and len(seen) - len(full) == 1)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" a group short of n is skipped entirely: marked {sorted(set(tuids[mask]))}")

# 5. The reference level, in the four regimes that decide the class split.
cases = [
    ("some live", [0.0, 1.0], [0.0], 0.5, "live mean"),
    ("all failed", [], [0.0, 0.0, 0.0], float("inf"), "everything reads stuck"),
    ("all solved", [], [1.0, 1.0], 0.0, "everything reads saturated"),
    ("degen split", [], [0.0, 1.0], 0.5, "midpoint"),
]
for name, live, degen, want, why in cases:
    got = _reference_level(live, degen)
    good = (got == want)
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" reference level, {name}: {got} ({why})")

# 6. The regression itself: the old fallback called an all-failed batch saturated.
all_fail = [0.0, 0.0, 0.0]
old_ref = float(np.mean(all_fail))          # what the previous code used
old_stuck = sum(1 for v in all_fail if v < old_ref)
new_ref = _reference_level([], all_fail)
new_stuck = sum(1 for v in all_fail if v < new_ref)
good = old_stuck == 0 and new_stuck == len(all_fail)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" all-failed batch: old rule called {len(all_fail)-old_stuck}/{len(all_fail)} "
      f"saturated, new rule calls {new_stuck}/{len(all_fail)} stuck")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
