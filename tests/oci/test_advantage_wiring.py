"""The advantage a 7+1 group actually produces, through core_algos.

WHY THROUGH core_algos AND NOT numpy. The design document pre-registered
A_success = +1/sqrt(7) = +0.378 and A_failure = -sqrt(7) = -2.646 as a wiring
check, and a numpy test of trajectory-level algebra confirms those numbers
happily. The training path disagrees for two independent reasons, and only
calling the real function finds either:

  1. compute_advantage's GRPO branch does not pass compute_mean_std_cross_steps,
     whose default is True, so the mean and std are taken over TURN ROWS. A
     trajectory's weight in its own baseline is its length, and
     EpisodeRewardManager writes the episode return onto every one of its rows.
  2. torch.std is ddof=1, so even the trajectory-weighted statistic gives
     +0.354 / -2.475, never +1/sqrt(7).

So the pre-registered constants were wrong under BOTH statistic modes. What this
file pins instead is the relation: the failure's LENGTH sets the magnitudes.
"""
import math, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage as G

TOK = 4


def build(n_succ=7, succ_turns=8, fail_turns=8):
    """One prompt group: n_succ successes plus one failure.

    The episode return is on EVERY turn row of a trajectory, which is what
    EpisodeRewardManager does (agent_system/reward_manager/episode.py).
    """
    rew, uid, tuid = [], [], []
    def traj(k, turns, r):
        for _ in range(turns):
            row = [0.0] * TOK
            row[-1] = float(r)
            rew.append(row); uid.append("g"); tuid.append(f"t{k}")
    for k in range(n_succ):
        traj(k, succ_turns, 1.0)
    traj(n_succ, fail_turns, 0.0)
    return (torch.tensor(rew), torch.ones(len(rew), TOK),
            np.array(uid), np.array(tuid))


def adv(fail_turns, cross=True, exclude_fail=False):
    rew, mask, uid, tuid = build(fail_turns=fail_turns)
    ex = None
    if exclude_fail:
        ex = torch.tensor([t == "t7" for t in tuid])
    a, _ = G(token_level_rewards=rew, response_mask=mask, index=uid,
             traj_index=tuid, norm_adv_by_std_in_grpo=True,
             compute_mean_std_cross_steps=cross, exclude_mask=ex)
    col = a[:, 0]
    return float(col[0]), float(col[-1])


ok = True

# 1. The wrapper's default: turn-weighted, so the magnitudes move with length.
print("  --- turn-weighted (the wrapper's default) ---")
want = {8: (0.375, -2.625), 30: (0.728, -1.358), 50: (0.940, -1.053)}
for ft, (ws, wf) in want.items():
    s, f = adv(ft)
    good = abs(s - ws) < 5e-4 and abs(f - wf) < 5e-4
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" fail runs {ft:>2} turns: A_success {s:+.3f}, A_failure {f:+.3f}")

# 2. The pre-registered constants are NOT what comes out, under either mode.
s8, f8 = adv(8)
st, ft_ = adv(8, cross=False)
good = (abs(s8 - 1 / math.sqrt(7)) > 1e-3 and abs(st - 1 / math.sqrt(7)) > 1e-3)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" +1/sqrt(7)={1/math.sqrt(7):+.4f} appears in neither mode "
      f"(turn {s8:+.4f}, trajectory {st:+.4f}; torch.std is ddof=1)")

# 3. Trajectory-weighted is length-invariant, which is the property the
#    turn-weighted statistic does not have.
vals = [adv(ft, cross=False) for ft in (8, 30, 50)]
good = all(abs(v[0] - vals[0][0]) < 1e-6 and abs(v[1] - vals[0][1]) < 1e-6 for v in vals)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" trajectory-weighted is length-invariant: A_success {vals[0][0]:+.4f}, "
      f"A_failure {vals[0][1]:+.4f} at 8, 30 and 50 turns")

# 4. THE F3 REGRESSION. Zeroing response_mask does not keep a row out of the
#    group's statistic; exclude_mask does. Without it an unwanted rollout moves
#    the yardstick of the seven the design says it does not touch.
rew, mask, uid, tuid = build(fail_turns=50)
mask_zeroed = mask.clone()
mask_zeroed[[i for i, t in enumerate(tuid) if t == "t7"]] = 0.0
a_masked, _ = G(token_level_rewards=rew, response_mask=mask_zeroed, index=uid,
                traj_index=tuid, norm_adv_by_std_in_grpo=True)
s_masked = float(a_masked[0, 0])
s_excluded, _ = adv(50, exclude_fail=True)
good = abs(s_masked - 0.940) < 5e-4 and abs(s_excluded) < 1e-9
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" response_mask=0 leaves the failure in the baseline (A_success "
      f"{s_masked:+.3f}, unchanged); exclude_mask removes it and the group goes "
      f"uniform (A_success {s_excluded:+.3f})")

# 5. An excluded row gets advantage ZERO, not the statistic it was excluded
#    from. Without this it is not merely wrong but explosive: excluding the only
#    row that differed leaves std 0, and (0 - 1)/epsilon is -1e6 sitting in the
#    advantages column for every metric downstream to read.
rew, mask, uid, tuid = build(fail_turns=8)
ex = torch.tensor([t == "t7" for t in tuid])
a_ex, _ = G(token_level_rewards=rew, response_mask=mask, index=uid,
            traj_index=tuid, norm_adv_by_std_in_grpo=True, exclude_mask=ex)
good = float(a_ex[-1, 0]) == 0.0 and abs(float(a_ex[0, 0])) < 1e-9
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" an excluded row gets advantage exactly 0 ({float(a_ex[-1,0]):+.1f}), "
      f"not (0-mean)/epsilon = -1e6")

# 6. And the exclusion does not leak into the rows that remain: a group whose
#    remaining rows are uniform is uniform, which is the correct reading.
good = abs(float(a_ex[0, 0])) < 1e-9
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the seven that remain read uniform ({float(a_ex[0,0]):+.1f}), because "
      f"after the drop the group really is")


def test_advantage_wiring():
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
