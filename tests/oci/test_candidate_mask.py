"""CPU test: who gets the corrupted plan, and what the class split falls back to.

TWO BUGS THIS FILE PINS.

1. The switch read ``config.env.rollout.n`` to find the slot. That is the
   TRAINING group size and the config object is shared, so the validation
   manager -- built with group_n=1, is_train=False -- also matched and one
   alfworld instance in eight was validated with a corrupted plan. The arm's
   own success-rate measurement was contaminated by the arm. It now asks the
   envs it holds.

2. ``compute_group_metrics`` fell back to the mean over all returns when no
   group was live. An all-failed batch has every return equal to that mean, so
   ``return < mean`` is false and every stuck group read as saturated -- the
   wrong arm on all of it, in exactly the regime the split exists for.

The candidate mask itself is no longer derived from the row order anywhere, so
there is nothing here to test about it: the rollout loop marks the env slot and
the mark travels as a column (see test_strip for the span, and the no-fallback
assertion in opd_grpo_ray_trainer).
"""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.metric_utils import _reference_level
from agent_system.environments.env_manager import _oci_candidate_row

GN = 8
ok = True


def envs(group_n=GN, is_train=True):
    return types.SimpleNamespace(group_n=group_n, is_train=is_train)


# --- the switch is off unless the environment variable says otherwise --------
os.environ.pop("PRIVILEGED_WRONG_PLAN", None)
good = not any(_oci_candidate_row(i, envs()) for i in range(GN))
ok &= good
print(("  OK  " if good else "  FAIL") + " switch off: no slot is a candidate")

os.environ["PRIVILEGED_WRONG_PLAN"] = "1"

# --- on a training manager, exactly the last slot of each group -------------
marks = [_oci_candidate_row(i, envs()) for i in range(3 * GN)]
good = [i for i, m in enumerate(marks) if m] == [GN - 1, 2 * GN - 1, 3 * GN - 1]
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" training manager: slots {[i for i, m in enumerate(marks) if m]} of 24")

# --- the validation manager must be untouched -------------------------------
val = [_oci_candidate_row(i, envs(group_n=1, is_train=False)) for i in range(3 * GN)]
good = not any(val)
ok &= good
print(("  OK  " if good else "  FAIL") +
      " validation manager (group_n=1, is_train=False): nothing marked")

# Both guards are load-bearing and are checked separately, because group_n=1
# alone would already have hidden the missing is_train check.
good = not any(_oci_candidate_row(i, envs(group_n=GN, is_train=False))
               for i in range(3 * GN))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " is_train=False with a train-sized group_n: still nothing marked")

good = not any(_oci_candidate_row(i, envs(group_n=1, is_train=True))
               for i in range(3 * GN))
ok &= good
print(("  OK  " if good else "  FAIL") + " group_n=1: a group of one has no odd slot")

# --- and what the old rule did on the same validation manager ---------------
old = [(i % 8) == 7 for i in range(3 * GN)]   # read config.env.rollout.n = 8
good = sum(old) == 3 and not any(val)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the old config-reading rule marked {sum(old)} validation slots, "
      f"which is the contamination")

good = not any(_oci_candidate_row(i, None) for i in range(GN))
ok &= good
print(("  OK  " if good else "  FAIL") + " no envs at all: nothing marked")
os.environ.pop("PRIVILEGED_WRONG_PLAN", None)

# --- the reference level, in the four regimes that decide the class split ----
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

# the regression itself
all_fail = [0.0, 0.0, 0.0]
old_ref = float(np.mean(all_fail))
old_stuck = sum(1 for v in all_fail if v < old_ref)
new_stuck = sum(1 for v in all_fail if v < _reference_level([], all_fail))
good = old_stuck == 0 and new_stuck == len(all_fail)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" all-failed batch: old rule called {len(all_fail)-old_stuck}/{len(all_fail)} "
      f"saturated, new rule calls {new_stuck}/{len(all_fail)} stuck")


def test_candidate_mask():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
