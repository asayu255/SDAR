"""CPU test for OCI-sat arm B', the virtual floor sample. No GPU, no rollout.

EVERY NUMBER HERE GOES THROUGH core_algos. The previous arm pre-registered
+1/sqrt(7) for the seven successes of a group that gained one injected failure,
and that value appears under NEITHER statistic mode -- it was derived on paper
from a trajectory-level unnormalised picture, while the code takes the mean and
std over TURN ROWS with ddof=1 and the real numbers were +0.375/-2.625. So the
arm's effect size is asserted against the function that computes it, not against
an algebraic claim about the function.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo import core_algos

ok = True
R = 10.0  # ALFWorld's return for a solved episode


def group(n_traj=8, turns=8, scores=None, resp_len=4):
    """A batch of one group: n_traj trajectories of `turns` rows each."""
    rows = n_traj * turns
    uid = np.array(["g0"] * rows)
    tuid = np.array([f"t{i // turns}" for i in range(rows)])
    rew = torch.zeros(rows, resp_len)
    for i in range(rows):
        s = R if scores is None else scores[i // turns]
        rew[i, 0] = s
    mask = torch.ones(rows, resp_len)
    return rew, mask, uid, tuid


def adv(floor=True, **kw):
    rew, mask, uid, tuid = group(**kw)
    fm = torch.ones(rew.shape[0]) if floor else None
    a, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
        compute_mean_std_cross_steps=True, floor_mask=fm, floor_value=0.0)
    return a[:, 0]


# --- the thing the arm exists to do ------------------------------------------
a0 = adv(floor=False)
good = bool(torch.allclose(a0, torch.zeros_like(a0)))
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" without the floor a saturated group's advantage is exactly zero "
      f"(max |A| = {float(a0.abs().max()):.2e})")

a = adv()
good = bool(a.min() > 0) and bool(torch.allclose(a, a[0].expand_as(a)))
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" with the floor every row of the group gets the same POSITIVE advantage "
      f"{float(a[0]):+.4f}")

# --- the pre-registered effect size, and its length-independence -------------
vals = {t: float(adv(turns=t)[0]) for t in (8, 20, 50)}
good = (abs(vals[8] - 0.351) < 0.002 and abs(vals[20] - 0.353) < 0.002
        and abs(vals[50] - vals[8]) < 0.005)
ok &= good
print(("  OK  " if good else "  FAIL") +
      " A_success = " + ", ".join(f"{t} turns {v:+.4f}" for t, v in vals.items()) +
      " -- the real injection gave +0.375 at 8 turns and +0.940 at 50")

# the trajectory-level statistic must see a weight of 1, not of a turn
rew, mask, uid, tuid = group(turns=8)
a_traj, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
    compute_mean_std_cross_steps=False, floor_mask=torch.ones(rew.shape[0]),
    floor_value=0.0)
# 8 samples at R plus one at 0: mean 8R/9, ddof=1 std R/3, so A = 1/3 exactly
good = abs(float(a_traj[0, 0]) - 1.0 / 3.0) < 1e-4
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" under the trajectory statistic the floor weighs one trajectory, giving "
      f"exactly 1/3 ({float(a_traj[0, 0]):.5f})")

# --- it must not touch a group it was not given -----------------------------
rew, mask, uid, tuid = group(turns=8)
uid2 = np.array(["g0"] * (len(uid) // 2) + ["g1"] * (len(uid) - len(uid) // 2))
fm = torch.zeros(rew.shape[0])
fm[: len(uid) // 2] = 1.0
a_two, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid2, traj_index=tuid,
    compute_mean_std_cross_steps=True, floor_mask=fm, floor_value=0.0)
good = bool(a_two[: len(uid) // 2, 0].min() > 0) and bool(
    torch.allclose(a_two[len(uid) // 2:, 0], torch.zeros(len(uid) - len(uid) // 2)))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " a group not marked keeps its zero advantage while the marked one moves")

# --- why only saturated groups ----------------------------------------------
# 6 successes and 2 failures: the group already has a signal. Applying the floor
# there INFLATES the successes' advantage (+0.573 -> +0.702) -- the virtual
# failure pulls the mean down by more than it widens the std, so the group is
# rewarded for a failure that did not happen on top of the two that did. That is
# double counting, not a bigger signal, and it is the reason select_floor_groups
# takes saturated groups only: there the honest advantage is exactly zero, so
# there is no true value for the term to distort.
sc = [R] * 6 + [0.0, 0.0]
rew, mask, uid, tuid = group(scores=sc)
a_live_plain, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
    compute_mean_std_cross_steps=True)
a_live_floor, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
    compute_mean_std_cross_steps=True, floor_mask=torch.ones(rew.shape[0]),
    floor_value=0.0)
good = (float(a_live_plain[0, 0]) > 0
        and float(a_live_floor[0, 0]) > float(a_live_plain[0, 0]) + 0.05)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" on a live group the floor INFLATES the successes' advantage "
      f"({float(a_live_plain[0, 0]):+.3f} -> {float(a_live_floor[0, 0]):+.3f}), "
      f"double-counting a failure -- so only saturated groups are floored")

# --- off is off --------------------------------------------------------------
rew, mask, uid, tuid = group(scores=[R] * 6 + [0.0, 0.0])
base, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
    compute_mean_std_cross_steps=True)
same, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rew, response_mask=mask, index=uid, traj_index=tuid,
    compute_mean_std_cross_steps=True, floor_mask=None, floor_value=0.0)
good = bool(torch.equal(base, same))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " floor_mask=None is bit-identical to the control path")


def test_floor():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
