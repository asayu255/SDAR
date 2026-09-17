"""(a)'s arithmetic: which groups fire, what they get, and how c follows the EMA.

WHAT IT PROTECTS.
  * rho = 0 is control bit for bit -- the identity check stage 1 starts from.
  * "stuck" is judged on the environment's reward, so a group of failures in
    which one rollout never produced a valid action is still ranked.
  * The ranking is zero-sum over the samples GRPO's own statistic uses, so it
    cannot become the net push-down that sank the ten-slot runs.
  * c is set from an EMA that stays positive on a step with no update, never
    from that step's update, and the injected mass is exactly rho * EMA.
  * The cap binds when one group fires on a small difference, and it is anchored
    on the push successes get in live groups (S), not on the all-row EMA.
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
    "top_below_min": [("a", 1, 0.0, 1, 6, 0.0), ("b", 1, 0.0, 0, 6, 0.0)],   # judged at min 2 below
    "no_progress": [("a", 1, 0.0, 3, 6, 0.0), ("b", 1, 0.0, 0, 0, 0.0)],
}
for want, trajs in cases.items():
    b = build([("g", "alfworld", [(f"{want}-{t}", *rest) for t, *rest in trajs])])
    grp = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(2))
    v = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"],
                              traj_prog=pr.trajectory_progress(b["tuids"], b["k_rows"], b["total_rows"], range(2)),
                              min_top_k={"alfworld": 2}, tasks=["alfworld"])
    check(v["g"]["verdict"] == want, f"{want}")
b = build([("g", "alfworld", [("a", 1, 0.0, 1, 2, 0.0), ("b", 1, 0.0, 0, 2, 0.0)])])
grp = pr.failed_groups(b["uids"], b["tuids"], b["task_names"], b["episode_rewards"], range(2))
v = pr.score_stuck_groups(grp, tuids=b["tuids"], stat_rows=b["stat_rows"],
                          traj_prog=pr.trajectory_progress(b["tuids"], b["k_rows"], b["total_rows"], range(2)),
                          min_top_k=pr.DEFAULT_MIN_TOP_K, tasks=["alfworld"])
check(v["g"]["verdict"] == "fired",
      "ALFWorld fires on a one-milestone difference by default (failed look_at rollouts stop at k = 1 of 2)")
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
check(per_token <= ctl.success_ema["alfworld"] * (1 + 1e-6),
      f"no token gets more than kappa * S ({per_token:.4f} <= {ctl.success_ema['alfworld']:.4f})")
check(abs(m["progress_rank/alfworld/top_push_over_success"] - 1.0) < 1e-9
      and abs(m["progress_rank/alfworld/success_push_ema"] - 1.0) < 1e-9,
      "and when it binds the top push is exactly kappa * S (S = the successes' +1.0)")

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
check(ctl2.ema == ctl.ema and ctl2.success_ema == ctl.success_ema
      and ctl.success_ema["alfworld"] is not None, "both EMAs round-trip through state_dict")
ctl3 = pr.ProgressRankController(rho=0.05)
ctl3.load_state_dict({"version": 1, "ema": {"alfworld": 0.3}})
check(ctl3.ema["alfworld"] == 0.3 and ctl3.success_ema["alfworld"] is None,
      "a version-1 state (E only) loads, and S starts uninitialised")

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
ctl.ema["alfworld"] = 0.3   # scales from earlier steps
ctl.success_ema["alfworld"] = 1.0
new, m = ctl.apply(advantages=adv, mask=mask, uids=index, tuids=traj_index,
                   task_names=np.array(["alfworld"] * n, dtype=object),
                   episode_rewards=np.zeros(n, dtype=object),
                   k_rows=np.array([r[2] for r in rows], dtype=object),
                   total_rows=np.array([r[3] for r in rows], dtype=object),
                   real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                   row_scores=tlr.sum(-1).numpy())
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


def build2(groups, resp=4):
    """``build`` plus per-row score, validity and coverage D, from
    ``(traj, turns, env_reward, k, K, adv, score, valid, d)``; the last three may be
    lists with one value per turn."""
    b = build([(u, task, [x[:6] for x in trajs]) for u, task, trajs in groups], resp)
    sc, va, dd = [], [], []
    for _, _, trajs in groups:
        for _traj, turns, _r, _k, _K, _a, s, v, d in trajs:
            for j in range(turns):
                sc.append(s[j] if isinstance(s, list) else s)
                va.append(v[j] if isinstance(v, list) else v)
                dd.append(d[j] if isinstance(d, list) else d)
    b["row_scores"] = np.array(sc, dtype=float)
    b["valid_rows"] = np.array(va, dtype=object)
    b["coverage_rows"] = np.array(dd, dtype=object)
    return b


print("13. the format split is judged on the scores, not on |A|")
# Two stuck groups of 8 x 50 turns and the same k: in one every turn is penalised
# (the real GRPO statistic returns rounding there -- at 400 rows; some sizes happen
# to round to exactly 0), the other has one valid turn.
rows = []   # (uid, traj, score, k)
for uid, prefix in (("allbad", "a"), ("onegood", "b")):
    for j, k in enumerate((2, 1, 1, 0, 2, 1, 1, 0)):
        for t in range(50):
            rows.append((uid, f"{prefix}{j}", 0.0 if (uid == "onegood" and j == 0 and t == 0) else -0.1, k))
n = len(rows)
tlr = torch.zeros(n, 5)
mask = torch.zeros(n, 5)
mask[:, :4] = 1.0
for i, r in enumerate(rows):
    tlr[i, 3] = r[2]
index = np.array([r[0] for r in rows], dtype=object)
traj_index = np.array([r[1] for r in rows], dtype=object)
adv, _ = compute_grpo_outcome_advantage(tlr, mask, index, traj_index, compute_mean_std_cross_steps=True)
check(0.0 < float(adv[:400, 0].abs().max()) < 0.03,
      f"the all -0.1 group's |A| is rounding ({float(adv[:400, 0].abs().max()):.4f}), not zero")
kw = dict(advantages=adv, mask=mask, uids=index, tuids=traj_index,
          task_names=np.array(["alfworld"] * n, dtype=object), episode_rewards=np.zeros(n, dtype=object),
          k_rows=np.array([r[3] for r in rows], dtype=object),
          total_rows=np.array([3] * n, dtype=object),
          real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool))
ctl = pr.ProgressRankController(rho=0.05, cap_kappa=100.0)
ctl.ema["alfworld"] = 0.3
ctl.success_ema["alfworld"] = 1.0
new, m = ctl.apply(**kw, row_scores=tlr.sum(-1).numpy(),
                   valid_rows=np.array([1.0 if r[2] == 0.0 else 0.0 for r in rows], dtype=object))
p = "progress_rank/alfworld"
check(m[f"{p}/stuck_uniform"] == 1.0 and m[f"{p}/stuck_mixed"] == 1.0,
      "uniform: the group whose scores are all -0.1; mixed: the one with a valid turn")
check(m[f"{p}/q_fail"] == 1.0 and m["shadow/progpo/alfworld/q_fail"] == 0.0,
      "both are stuck by the environment, and ProGPO's gate (every |score| < 1e-3) opens on neither")
check(abs(m[f"{p}/inject_share_mixed"] - 0.5) < 1e-9
      and abs(m[f"{p}/inject_up_mixed"] + m[f"{p}/inject_up_uniform"] - m[f"{p}/inject_up"]) < 1e-12,
      "(a)'s push splits by class and the parts add up (two identical groups: half each)")
# row 400 is the one valid turn (trajectory b0, k = 2 against a mean of 1, so its
# score is +1/3): c / 3 per token, times its 4 tokens, over the task's tokens
one_valid = m[f"{p}/c"] / 3.0 * 4 / float(mask.sum())
check(abs(float(new[400, 0] - adv[400, 0]) - m[f"{p}/c"] / 3.0) < 1e-6, "(that row did get c / 3)")
check(abs(m[f"{p}/inject_down_invalid"] - m[f"{p}/inject_down"]) < 1e-12
      and abs(m[f"{p}/inject_up"] - m[f"{p}/inject_up_invalid"] - one_valid) < 1e-9,
      "on invalid-turn rows: all of the push-down, and all of the push-up but the one valid row's")
_, m_plain = pr.ProgressRankController(rho=0.05).apply(**kw)
check(f"{p}/stuck_mixed" not in m_plain and "shadow/progpo/alfworld/q_fail" not in m_plain
      and f"{p}/inject_up_invalid" not in m_plain,
      "without scores or validity nothing is guessed from |A|: those metrics are absent")

print("14. the cap, reported on both sides")
b = build([("live", "alfworld", live),
           ("stuck", "alfworld", [("s1", 1, 0.0, 2, 6, 0.0), ("s2", 1, 0.0, 1, 6, 0.0)])])
_, m = pr.ProgressRankController(rho=0.5, cap_kappa=1.0).apply(**b)
check(m[f"{p}/capped"] == 1.0 and m[f"{p}/c_uncapped"] > m[f"{p}/c_cap"]
      and m[f"{p}/c"] == m[f"{p}/c_cap"], "capped: c is the cap, and the target above it is kept")
b = build([("live", "alfworld", [("l1", 2, 10.0, 6, 6, 1.0), ("l2", 2, 0.0, 2, 6, -1.0)]),
           ("stuck", "alfworld", [("s1", 2, 0.0, 4, 6, 0.0), ("s2", 2, 0.0, 1, 6, 0.0)])])
_, m = pr.ProgressRankController(rho=0.05, cap_kappa=100.0).apply(**b)
check(m[f"{p}/capped"] == 0.0 and m[f"{p}/c"] == m[f"{p}/c_uncapped"] < m[f"{p}/c_cap"],
      "uncapped: c is the target, below the cap")
check(abs(m[f"{p}/share_of_ema"] - 0.05) < 1e-9, "and share_of_ema is the share actually injected (rho)")

print("15. winners the count gives nothing")
b = build2([("live", "search", [("w1", 1, 1.0, 0, 1, 0.5, 1.0, 1.0, 2), ("w2", 1, 1.0, 1, 1, 0.5, 1.0, 1.0, 2),
                                ("l1", 1, 0.0, 1, 1, -0.5, 0.0, 1.0, 2)]),
            ("yesno", "search", [("w3", 1, 1.0, 0, 0, 0.5, 1.0, 1.0, 1), ("l2", 1, 0.0, 0, 0, -0.5, 0.0, 1.0, 1)])])
_, m = pr.ProgressRankController(rho=0.05).apply(**b)
check(abs(m["progress_rank/search/won_with_zero_k"] - 0.5) < 1e-12,
      "k = 0 among winners that have a sequence: w1 of w1, w2 (w3's question has none)")
check(abs(m["shadow/coverage/search/won_with_zero"] - 1.0 / 3.0) < 1e-12,
      "coverage: a winner that saw one observation (D = 1) scores P = 0 -- w3 of three")

print("16. ProGPO's coverage beside k, on the same groups")
# g1: stuck, all turns valid (ProGPO's gate open), k and P both differ -> both fire
#     a: T 4 D 5 P 1.0 k 3 | b: T 4 D 3 P 0.5 k 1 | c: T 2 D 2 P 0.5 k 2 | d: T 2 D 1 P 0 k 0
# g2: stuck, one invalid turn (gate shut), k equal, P differs -> only coverage fires
# g3: stuck, all valid, k differs, P equal -> only (a) fires
# live and saturated groups for the counts and the records
b = build2([
    ("g1", "alfworld", [("a", 4, 0.0, 3, 3, 0.0, 0.0, 1.0, [2, 3, 4, 5]), ("b", 4, 0.0, 1, 3, 0.0, 0.0, 1.0, 3),
                        ("c", 2, 0.0, 2, 3, 0.0, 0.0, 1.0, 2), ("d", 2, 0.0, 0, 3, 0.0, 0.0, 1.0, 1)]),
    ("g2", "alfworld", [("e", 2, 0.0, 1, 3, 0.0, [-0.1, 0.0], [0.0, 1.0], 3), ("f", 2, 0.0, 1, 3, 0.0, 0.0, 1.0, 2)]),
    ("g3", "alfworld", [("g", 2, 0.0, 2, 3, 0.0, 0.0, 1.0, 3), ("h", 2, 0.0, 0, 3, 0.0, 0.0, 1.0, 3)]),
    ("lv", "alfworld", [("i", 1, 10.0, 3, 3, 1.0, 10.0, 1.0, 2), ("j", 1, 0.0, 1, 3, -1.0, 0.0, 1.0, 2)]),
    ("st", "alfworld", [("k", 1, 10.0, 3, 3, 0.0, 10.0, 1.0, 2), ("l", 1, 10.0, 3, 3, 0.0, 10.0, 1.0, 2)]),
])
ctl = pr.ProgressRankController(rho=0.05, cap_kappa=100.0)
new, m = ctl.apply(**b)
s = "shadow/coverage/alfworld"
check(m[f"{s}/stuck_fired"] == 2.0 and m[f"{s}/both_fired"] == 1.0
      and m["shadow/progpo/alfworld/fired"] == 1.0,
      "coverage fires in g1 and g2, both in g1 only, and ProGPO itself (gate open) in g1 only")
check(abs(m[f"{s}/spearman_vs_k"] - 4.5 / np.sqrt(22.5)) < 1e-12,
      "Spearman on average ranks in g1: k (3,1,2,0) against P (1,.5,.5,0) is 4.5/sqrt(22.5)")
check(m[f"{p}/stuck_fired"] == 2.0 and m[f"{p}/stuck_no_difference"] == 1.0,
      "(a) fires in g1 and g3 and not in g2, as before")
check(abs(m[f"{p}/q_fail"] - 3 / 5) < 1e-12 and abs(m["shadow/progpo/alfworld/q_fail"] - 2 / 5) < 1e-12,
      "q_fail: 3 of 5 stuck by the environment; ProGPO's gate opens on g1 and g3")
check(torch.equal(new[12:16], b["advantages"][12:16]) and torch.equal(new[20:], b["advantages"][20:]),
      "the shadow changes nothing: g2 (coverage fired, (a) did not) and the live and saturated rows keep theirs")

print("17. the group records")
recs = ctl.last_group_records
by = {r["uid"]: r for r in recs}
check(sorted(by) == ["g1", "g2", "g3", "lv", "st"], "one record per group")
r1 = by["g1"]
check(r1["trajs"] == ["a", "b", "c", "d"] and r1["k"] == [3.0, 1.0, 2.0, 0.0]
      and r1["coverage_d"] == [5.0, 3.0, 2.0, 1.0] and r1["turns"] == [4, 4, 2, 2],
      "per trajectory, in one order: k, D (the largest over its rows) and turns")
check(r1["status"] == "stuck" and r1["verdict"] == "fired" and r1["progpo_gate"] is True
      and r1["score_spread"] == 0.0 and r1["invalid_turns"] == [0, 0, 0, 0],
      "the group's verdict, ProGPO's gate and the format spread")
check(by["g2"]["score_spread"] > 0 and by["g2"]["invalid_turns"] == [1, 0] and by["g2"]["progpo_gate"] is False,
      "g2 carries its invalid turn and a closed gate")
check(by["lv"]["status"] == "live" and by["st"]["status"] == "saturated" and by["lv"]["verdict"] is None
      and by["st"]["won"] == [True, True], "live and saturated groups are recorded too, with no verdict")
g1_rows = [i for i, u in enumerate(b["uids"]) if u == "g1"]
gained = float(((new - b["advantages"]).abs() * b["mask"])[g1_rows].sum())
check(abs(r1["injected_abs_mass"] - gained) < 1e-5 and r1["c"] == m[f"{p}/c"] and r1["capped"] is False,
      "the injected mass is what the advantages actually gained; c and the cap flag ride along")
import json  # noqa: E402

try:
    json.dumps(recs)
    check(True, "the records are plain JSON")
except TypeError as e:
    check(False, f"the records are plain JSON ({e})")

print("18. ranks")
check(list(pr.average_ranks([3, 1, 2, 2])) == [4.0, 1.0, 2.5, 2.5], "ties share their average rank")
check(pr.spearman([1, 2, 3], [3, 2, 1]) == -1.0 and pr.spearman([1, 1], [0, 1]) is None,
      "Spearman is -1 for a reversed order and undefined when one side does not vary")
check(pr.coverage_progress(5, 4) == 1.0 and pr.coverage_progress(None, 3) is None
      and pr.coverage_progress(2, 0) is None, "P = (D - 1) / T, undefined without D or T")

print("19. the cap's anchor: what a success gets in a live group")
# Live group: 3 successes (2 turns each, A +0.8) and 5 failures (1 turn, A -0.6);
# a saturated group whose rows carry +0.3 (must not count); a stuck group firing.
b = build([
    ("lv", "webshop", [("w1", 2, 10.0, 5, 5, 0.8), ("w2", 2, 10.0, 5, 5, 0.8), ("w3", 2, 10.0, 5, 5, 0.8),
                       ("f1", 1, 0.0, 1, 5, -0.6), ("f2", 1, 0.0, 1, 5, -0.6), ("f3", 1, 0.0, 1, 5, -0.6),
                       ("f4", 1, 0.0, 1, 5, -0.6), ("f5", 1, 0.0, 1, 5, -0.6)]),
    ("sat", "webshop", [("a1", 1, 10.0, 5, 5, 0.3), ("a2", 1, 10.0, 5, 5, 0.3)]),
    ("st", "webshop", [("s1", 1, 0.0, 3, 5, 0.0), ("s2", 1, 0.0, 0, 5, 0.0)]),
])
ctl = pr.ProgressRankController(rho=0.5, cap_kappa=0.5)
new, m = ctl.apply(**b)
p = "progress_rank/webshop"
check(abs(m[f"{p}/success_push"] - 0.8) < 1e-6 and abs(ctl.success_ema["webshop"] - 0.8) < 1e-6,
      "P is the token-mean A on live groups' successful rows only (not failures, not saturated rows)")
top = float((new - b["advantages"])[-2:, 0].abs().max())
check(m[f"{p}/capped"] == 1.0 and abs(top - 0.5 * 0.8) < 1e-6
      and abs(m[f"{p}/top_push_over_success"] - 0.5) < 1e-9,
      f"capped: the top push is kappa * S = 0.4 ({top:.4f})")
# float32 advantages: S carries their precision
check(abs(m[f"{p}/c_cap"] * 0.3 - 0.4) < 1e-6, "c_cap is kappa * S / max|score| (score +-0.3 here)")
# a step with no live group leaves S where it was, and (a) still fires
b2 = build([("st", "webshop", [("s1", 1, 0.0, 3, 5, 0.0), ("s2", 1, 0.0, 0, 5, 0.0)]),
            ("sat", "webshop", [("a1", 1, 10.0, 5, 5, 0.3), ("a2", 1, 10.0, 5, 5, 0.3)])])
_, m2 = ctl.apply(**b2)
check("progress_rank/webshop/success_push" not in m2 and abs(ctl.success_ema["webshop"] - 0.8) < 1e-6
      and m2[f"{p}/c"] > 0, "no live group: S keeps its value and (a) is still on")
# before any live group has been seen, (a) waits even with E set
fresh = pr.ProgressRankController(rho=0.5, cap_kappa=0.5)
_, m3 = fresh.apply(**b2)
check(fresh.ema["webshop"] is not None and fresh.success_ema["webshop"] is None and m3[f"{p}/c"] == 0.0,
      "S uninitialised: c is 0 although E is set")
check(ctl.last_group_records[0]["success_push_ema"] == ctl.success_ema["webshop"],
      "the records carry S")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
