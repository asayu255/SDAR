"""(a)'s arithmetic: which groups fire, what they get, and how c follows the EMA.

WHAT IT PROTECTS.
  * rho = 0 is control bit for bit -- the identity check stage 1 starts from.
  * "stuck" is judged on the environment's reward, so a group of failures in
    which one rollout never produced a valid action is still ranked.
  * The ranking is zero-sum over the samples GRPO's own statistic uses, so it
    cannot become the net push-down that sank the ten-slot runs.
  * c is set from an EMA that stays positive on a step with no update, never
    from that step's update, and the injected mass is exactly rho * EMA.
  * The cap binds when one group fires on a small difference.
  * Padding copies are kept out of every sum and still match their originals.
No model and no GPU. The last section runs the real compute_grpo_outcome_advantage.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from verl.trainer.ppo import progress_rank as pr  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def build(groups, resp=4):
    """Rows from ``[(uid, task, [(traj, turns, env_reward, k, K, adv), ...]), ...]``.

    Every turn row of a trajectory carries the same k, K and advantage, and three
    response tokens (the last position is padding in the mask).
    """
    uids, tuids, tasks, rew, ks, Ks, adv = [], [], [], [], [], [], []
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a in trajs:
            for _ in range(turns):
                uids.append(uid)
                tuids.append(traj)
                tasks.append(task)
                rew.append(r)
                ks.append(k)
                Ks.append(K)
                adv.append(a)
    n = len(uids)
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    A = torch.tensor(adv, dtype=torch.float32).unsqueeze(-1) * mask
    return dict(advantages=A, mask=mask, uids=np.array(uids, dtype=object),
                tuids=np.array(tuids, dtype=object), task_names=np.array(tasks, dtype=object),
                episode_rewards=np.array(rew, dtype=object), k_rows=np.array(ks, dtype=object),
                total_rows=np.array(Ks, dtype=object),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool))


print("1. which groups are stuck")
b = build([
    ("g1", "alfworld", [("a", 2, 0.0, 3, 6, 0.0), ("b", 2, 0.0, 0, 6, 0.0)]),
    ("g2", "alfworld", [("c", 2, 10.0, 6, 6, 1.0), ("d", 2, 0.0, 1, 6, -1.0)]),
])
g = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(8))
check(g["g1"]["status"] == "stuck" and g["g2"]["status"] == "other",
      "a group with a success is not stuck; one with none is")
# The case classify_groups gets wrong: all failed, one rollout with every turn
# invalid. Its ROW returns read -0.1 against its sibling's 0, which makes the
# group "live" by returns. The environment says nobody succeeded.
from verl.trainer.ppo.oci_saturated import classify_returns  # noqa: E402

_u, _t = np.array(["u", "u"], dtype=object), np.array(["x", "y"], dtype=object)
_by_returns = classify_returns([0.0, -0.1], _u, _t, np.array(["alfworld"] * 2, dtype=object))
g = pr.failed_groups(_u, _t, np.array(["alfworld"] * 2, dtype=object),
                     np.array([0.0, 0.0], dtype=object), range(2))
check(_by_returns["u"]["status"] == "live" and g["u"]["status"] == "stuck",
      "two failures, one all-invalid: live by row returns, stuck by the environment's reward")
g = pr.failed_groups(np.array(["u"], dtype=object), np.array(["x"], dtype=object),
                     np.array(["search"], dtype=object), np.array([0.0], dtype=object), range(1))
check(g["u"]["status"] == "other", "a single rollout is not a group to rank")

print("2. per-trajectory progress")
tp = pr.trajectory_progress(np.array(["a", "a", "b", "b"], dtype=object),
                            np.array([1, 3, 2, None], dtype=object),
                            np.array([6, 6, 6, 6], dtype=object), range(4))
check(tp["a"] == (3.0, 6.0), "k is the maximum over the trajectory's rows")
check(tp["b"] == (0.0, 0.0), "a trajectory with a row lacking its count has no progress at all")

print("3. verdicts")
cases = {
    "fired": [("a", 1, 0.0, 3, 6, 0.0), ("b", 1, 0.0, 1, 6, 0.0)],
    "no_difference": [("a", 1, 0.0, 2, 6, 0.0), ("b", 1, 0.0, 2, 6, 0.0)],
    "top_below_min": [("a", 1, 0.0, 1, 6, 0.0), ("b", 1, 0.0, 0, 6, 0.0)],
    "no_progress": [("a", 1, 0.0, 3, 6, 0.0), ("b", 1, 0.0, 0, 0, 0.0)],
}
for want, trajs in cases.items():
    b = build([("g", "alfworld", [(f"{want}-{t}", *rest) for t, *rest in trajs])])
    grp = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(2))
    v = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"],
                              traj_prog=pr.trajectory_progress(b["tuids"], b["k_rows"], b["total_rows"], range(2)),
                              min_top_k=pr.DEFAULT_MIN_TOP_K, tasks=["alfworld"])
    check(v["g"]["verdict"] == want, f"{want}")
b = build([("g", "search", [("a", 1, 0.0, 1, 1, 0.0), ("b", 1, 0.0, 0, 1, 0.0)])])
grp = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(2))
v = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"],
                          traj_prog=pr.trajectory_progress(b["tuids"], b["k_rows"], b["total_rows"], range(2)),
                          min_top_k=pr.DEFAULT_MIN_TOP_K, tasks=["search"])
check(v["g"]["verdict"] == "fired", "Search fires on its single step (min_top_k 1)")

print("4. the score is zero-sum over what the GRPO statistic counts")
# Three rollouts of different LENGTHS: 5, 2 and 1 turns. Under the cross-steps
# statistic each turn row is a sample, so the mean k is turn-weighted.
b = build([("g", "alfworld", [("a", 5, 0.0, 4, 6, 0.0), ("b", 2, 0.0, 1, 6, 0.0),
                              ("c", 1, 0.0, 0, 6, 0.0)])])
n = len(b["uids"])
grp = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(n))
tp = pr.trajectory_progress(b["tuids"], b["k_rows"], b["total_rows"], range(n))
v = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"], traj_prog=tp,
                          min_top_k=pr.DEFAULT_MIN_TOP_K, tasks=["alfworld"], cross_steps=True)
s = v["g"]["scores"]
turn_sum = 5 * s["a"] + 2 * s["b"] + 1 * s["c"]
check(abs(turn_sum) < 1e-12, f"cross steps: the sum over turn rows is 0 ({turn_sum:.2e})")
check(abs(s["a"] - (4 - (5 * 4 + 2 * 1) / 8) / 6) < 1e-12, "and the mean is turn-weighted")
v2 = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"], traj_prog=tp,
                           min_top_k=pr.DEFAULT_MIN_TOP_K, tasks=["alfworld"], cross_steps=False)
s2 = v2["g"]["scores"]
check(abs(s2["a"] + s2["b"] + s2["c"]) < 1e-12, "per-trajectory statistic: the sum over trajectories is 0")
check(all(-1.0 <= x <= 1.0 for x in s.values()), "every score is within [-1, 1]")

print("5. rho = 0 is control")
b = build([("g", "alfworld", [("a", 3, 0.0, 4, 6, 0.1), ("b", 3, 0.0, 0, 6, -0.1)])])
ctl = pr.ProgressRankController(rho=0.0)
new, m = ctl.apply(**b)
check(new is b["advantages"], "the very same tensor comes back: bit-identical by construction")
check(m["progress_rank/alfworld/c"] == 0.0 and m["progress_rank/alfworld/stuck_fired"] == 1.0,
      "the group is still counted as firing, and c is 0")
check(ctl.ema["alfworld"] is not None, "and the EMA is tracked even at rho = 0")

print("6. the injected mass is rho times the EMA")
# Task alfworld: one live group (ordinary advantages +-1) and one stuck group
# whose rollouts differ in progress.
b = build([
    ("live", "alfworld", [("l1", 2, 10.0, 6, 6, 1.0), ("l2", 2, 0.0, 2, 6, -1.0)]),
    ("stuck", "alfworld", [("s1", 2, 0.0, 4, 6, 0.0), ("s2", 2, 0.0, 1, 6, 0.0)]),
])
rho = 0.05
ctl = pr.ProgressRankController(rho=rho, cap_kappa=100.0)
new, m = ctl.apply(**b)
p = "progress_rank/alfworld"
tokens = b["mask"].sum(-1).numpy()
task_tokens = tokens.sum()
m_t = float((b["advantages"].abs() * b["mask"]).sum()) / task_tokens
check(abs(m[f"{p}/mean_abs_adv"] - m_t) < 1e-12, "M is read off the ordinary advantages, before (a)")
check(abs(ctl.ema["alfworld"] - m_t) < 1e-12, "the first step with an update initialises the EMA to it")
added = float(((new - b["advantages"]).abs() * b["mask"]).sum()) / task_tokens
check(abs(added - rho * ctl.ema["alfworld"]) < 1e-6,
      f"added token-mean |A| = rho * EMA ({added:.6f} vs {rho * ctl.ema['alfworld']:.6f})")
check(torch.equal(new[:4], b["advantages"][:4]), "the live group's advantages are untouched")
delta = (new - b["advantages"])[4:, 0]
check(float(delta[:2].min()) > 0 and float(delta[2:].max()) < 0,
      "the further rollout is pushed up, the shorter one down")
check(abs(float(delta.sum())) < 1e-6, "and the addition sums to zero over the group's turn rows")

print("7. a step with no update keeps (a) on")
b0 = build([("stuck", "search", [("s1", 1, 0.0, 1, 1, 0.0), ("s2", 1, 0.0, 0, 1, 0.0)])])
ctl = pr.ProgressRankController(rho=0.05)
_, m = ctl.apply(**b0)
check(ctl.ema["search"] is None and m["progress_rank/search/c"] == 0.0,
      "before any update there is no scale, and (a) stays off")
b1 = build([("live", "search", [("l1", 1, 1.0, 1, 1, 0.7), ("l2", 1, 0.0, 0, 1, -0.7)])])
ctl.apply(**b1)
e1 = ctl.ema["search"]
_, m = ctl.apply(**b0)
check(m["progress_rank/search/mean_abs_adv"] == 0.0 and m["progress_rank/search/c"] > 0.0,
      "on a step whose own update is 0, c is still positive: the EMA carries the scale")
check(abs(ctl.ema["search"] - 0.8 * e1) < 1e-12, "and the EMA decays by (1 - alpha) on that step")
check("progress_rank/search/share_of_step" not in m,
      "share of this step's update is not reported when there is none")
for _ in range(200):
    ctl.apply(**b0)
check(abs(ctl.ema["search"] - 0.01) < 1e-12, "a long run of empty steps stops at the floor, not at 0")

print("8. the cap")
# Many live rows with a large update, one stuck group on a one-step difference:
# the uncapped c would put far more than the typical |A| on that group's tokens.
live = [(f"l{i}", 1, (10.0 if i % 2 else 0.0), 6, 6, (1.0 if i % 2 else -1.0)) for i in range(40)]
b = build([("live", "alfworld", live),
           ("stuck", "alfworld", [("s1", 1, 0.0, 2, 6, 0.0), ("s2", 1, 0.0, 1, 6, 0.0)])])
ctl = pr.ProgressRankController(rho=0.5, cap_kappa=1.0)
new, m = ctl.apply(**b)
per_token = float((new - b["advantages"])[-2:, 0].abs().max())
check(m["progress_rank/alfworld/capped"] == 1.0, "the cap binds")
# float32 advantages: the bound holds to their precision, not to float64's.
check(per_token <= ctl.ema["alfworld"] * (1 + 1e-6),
      f"no token gets more than kappa * EMA ({per_token:.4f} <= {ctl.ema['alfworld']:.4f})")

print("9. padding copies")
b = build([
    ("live", "webshop", [("l1", 1, 10.0, 5, 5, 1.0), ("l2", 1, 0.0, 2, 5, -1.0)]),
    ("stuck", "webshop", [("s1", 1, 0.0, 3, 5, 0.0), ("s2", 1, 0.0, 0, 5, 0.0)]),
])
ctl_a = pr.ProgressRankController(rho=0.05)
new_a, m_a = ctl_a.apply(**b)
# the same batch with row 2 (s1) duplicated as padding
pad = {k: (np.concatenate([v, v[2:3]]) if isinstance(v, np.ndarray) else torch.cat([v, v[2:3]]))
       for k, v in b.items()}
pad["real_rows"] = np.array([True] * 4 + [False])
pad["stat_rows"] = np.array([True] * 4 + [False])
ctl_b = pr.ProgressRankController(rho=0.05)
new_b, m_b = ctl_b.apply(**pad)
check(abs(m_a["progress_rank/webshop/c"] - m_b["progress_rank/webshop/c"]) < 1e-12
      and abs(ctl_a.ema["webshop"] - ctl_b.ema["webshop"]) < 1e-12,
      "a padding copy changes neither M nor u nor c")
check(torch.equal(new_b[4], new_b[2]), "and it carries the same advantage as its original")

print("10. tasks do not share a scale")
b = build([
    ("la", "alfworld", [("a1", 1, 10.0, 6, 6, 2.0), ("a2", 1, 0.0, 1, 6, -2.0)]),
    ("sa", "alfworld", [("a3", 1, 0.0, 3, 6, 0.0), ("a4", 1, 0.0, 0, 6, 0.0)]),
    ("ls", "search", [("s1", 1, 1.0, 1, 1, 0.3), ("s2", 1, 0.0, 0, 1, -0.3)]),
    ("ss", "search", [("s3", 1, 0.0, 1, 1, 0.0), ("s4", 1, 0.0, 0, 1, 0.0)]),
])
ctl = pr.ProgressRankController(rho=0.05, cap_kappa=100.0)
_, m = ctl.apply(**b)
check(abs(ctl.ema["alfworld"] / ctl.ema["search"] - 2.0 / 0.3) < 1e-5,
      "each task's EMA is its own update")
check(abs(m["progress_rank/alfworld/injected_mean_abs_adv"] - 0.05 * ctl.ema["alfworld"]) < 1e-9
      and abs(m["progress_rank/search/injected_mean_abs_adv"] - 0.05 * ctl.ema["search"]) < 1e-9,
      "and each gets rho of its own, not of the pooled batch")

print("11. state survives a checkpoint")
ctl2 = pr.ProgressRankController(rho=0.05)
ctl2.load_state_dict(ctl.state_dict())
check(ctl2.ema == ctl.ema, "EMA round-trips through state_dict")

print("12. on top of the real GRPO advantage")
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage  # noqa: E402

# One stuck ALFWorld group of four rollouts with different lengths; one turn is
# invalid, so the ordinary advantage is NOT zero (the format channel) -- (a) must
# add to it, not replace it.
rows = []   # (traj, row_reward, k, K)
for traj, turns, k in (("a", 4, 5), ("b", 3, 2), ("c", 2, 0), ("d", 3, 2)):
    for t in range(turns):
        rows.append((traj, -0.1 if (traj == "c" and t == 0) else 0.0, k, 7))
n = len(rows)
tlr = torch.zeros(n, 5)
mask = torch.zeros(n, 5)
mask[:, :4] = 1.0
for i, (_, r, _, _) in enumerate(rows):
    tlr[i, 3] = r
index = np.array(["g"] * n, dtype=object)
traj_index = np.array([r[0] for r in rows], dtype=object)
adv, _ = compute_grpo_outcome_advantage(tlr, mask, index, traj_index, compute_mean_std_cross_steps=True)
check(float(adv.abs().sum()) > 0, "the ordinary advantage is non-zero here (format channel)")
ctl = pr.ProgressRankController(rho=0.05, cap_kappa=100.0)
ctl.ema["alfworld"] = 0.3   # a scale from earlier steps
new, m = ctl.apply(advantages=adv, mask=mask, uids=index, tuids=traj_index,
                   task_names=np.array(["alfworld"] * n, dtype=object),
                   episode_rewards=np.zeros(n, dtype=object),
                   k_rows=np.array([r[2] for r in rows], dtype=object),
                   total_rows=np.array([r[3] for r in rows], dtype=object),
                   real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool))
delta = (new - adv)[:, 0]
check(abs(float(delta.sum())) < 1e-6,
      "(a)'s addition sums to zero over the rows compute_grpo_outcome_advantage centres over")
check(abs(float(adv[:, 0].sum())) < 1e-4, "as GRPO's own advantage does")
check(float(delta[0]) > 0 and float(delta[7]) < 0, "furthest pushed up, least far pushed down")
same_k = [i for i, r in enumerate(rows) if r[0] in ("b", "d")]
check(len({round(float(delta[i]), 9) for i in same_k}) == 1,
      "two rollouts that got equally far get the same addition")
check(m["progress_rank/alfworld/stuck_mixed"] == 1.0,
      "and the group is recorded as one the format channel already moves")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
