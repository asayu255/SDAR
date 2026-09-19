"""(a) on saturated groups (sat_rho): winners ranked by turn count.

WHAT IT PROTECTS.
  * sat_rho = 0 is the existing (a) bit for bit, with saturated groups present.
  * In a saturated group the faster winners go up and the slower down, and the
    term sums to zero over the rows GRPO's statistic uses.
  * The cap holds: no winning token is pushed up more than kappa * S.
  * A group fires only above sat_min_spread; Search is never ranked by turns.
  * The stuck-group term is untouched by turning sat on.
  * Like (a), the term waits for both scales (E and S).
No model and no GPU.
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
    """Rows from ``[(uid, task, [(traj, turns, env_reward, k, K, adv, n_invalid), ...]), ...]``.

    Every turn row of a trajectory carries the same k, K and advantage, three
    response tokens, and the first ``n_invalid`` of its turns are invalid.
    """
    uids, tuids, tasks, rew, ks, Ks, adv, valid = [], [], [], [], [], [], [], []
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, n_inv in trajs:
            for j in range(turns):
                uids.append(uid)
                tuids.append(traj)
                tasks.append(task)
                rew.append(r)
                ks.append(k)
                Ks.append(K)
                adv.append(a)
                valid.append(0 if j < n_inv else 1)
    n = len(uids)
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    A = torch.tensor(adv, dtype=torch.float32).unsqueeze(-1) * mask
    return dict(advantages=A, mask=mask, uids=np.array(uids, dtype=object),
                tuids=np.array(tuids, dtype=object), task_names=np.array(tasks, dtype=object),
                episode_rewards=np.array(rew, dtype=object), k_rows=np.array(ks, dtype=object),
                total_rows=np.array(Ks, dtype=object),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                valid_rows=np.array(valid, dtype=object))


# A live ALFWorld group (to give S a value), a saturated ALFWorld group whose
# winners took 4, 4, 8 and 12 turns (the 12-turn one with 3 invalid turns), a
# saturated WebShop group 5 vs 6 turns, a saturated ALFWorld group 5 vs 6 turns
# (below the min spread of 2), a saturated Search group 1 vs 3 turns, and a stuck
# ALFWorld group whose k differ.
LIVE = ("live", "alfworld", [("L1", 3, 10.0, 2, 2, 1.0, 0), ("L2", 3, 0.0, 1, 2, -1.0, 0)])
SAT = ("sat", "alfworld", [("S1", 4, 10.0, 2, 2, 0.0, 0), ("S2", 4, 10.0, 2, 2, 0.0, 0),
                           ("S3", 8, 10.0, 2, 2, 0.0, 0), ("S4", 12, 10.0, 2, 2, 0.0, 3)])
WS = ("ws", "webshop", [("W1", 5, 10.0, 4, 4, 0.0, 0), ("W2", 6, 10.0, 4, 4, 0.0, 0)])
NARROW = ("narrow", "alfworld", [("N1", 5, 10.0, 2, 2, 0.0, 0), ("N2", 6, 10.0, 2, 2, 0.0, 0)])
SEARCH = ("srch", "search", [("Q1", 1, 1.0, 1, 1, 0.0, 0), ("Q2", 3, 1.0, 1, 1, 0.0, 0)])
STUCK = ("stuck", "alfworld", [("F1", 5, 0.0, 1, 2, 0.0, 0), ("F2", 5, 0.0, 0, 2, 0.0, 0)])
WS_LIVE = ("wslive", "webshop", [("V1", 4, 10.0, 4, 4, 1.0, 0), ("V2", 4, 0.0, 1, 4, -1.0, 0)])


def run(ctl, groups):
    b = build(groups)
    new, metrics = ctl.apply(**b)
    return b, new, metrics


def rows_of(b, traj):
    return np.where(b["tuids"] == traj)[0]


print("1. sat_rho = 0: bit-identical, saturated groups present and still scored")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.0)
b, new, m = run(ctl, [LIVE, SAT, WS, NARROW, SEARCH, STUCK, WS_LIVE])
check(new is b["advantages"], "the very same tensor comes back")
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "fired", "the 4/4/8/12 group is scored as firing")
check(recs["sat"]["sat_score"][0] > 0 > recs["sat"]["sat_score"][3], "faster scores above slower")
check(recs["narrow"]["sat_verdict"] == "spread_below_min", "ALFWorld 5 vs 6 is below the min spread of 2")
check(recs["ws"]["sat_verdict"] == "fired", "WebShop 5 vs 6 fires (min spread 1)")
check(recs["srch"]["sat_verdict"] is None, "Search is never ranked by turns")
check(m.get("progress_rank/alfworld/groups_saturated") == 2.0, "saturated groups are counted")

print("2. waits for E and S: the first step with no live group adds nothing")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2)
b, new, m = run(ctl, [SAT, WS])
check(new is b["advantages"] and m["progress_rank/alfworld/sat_c"] == 0.0, "S unset -> c_sat = 0")

print("3. sat_rho > 0: the ranking, zero-sum, the cap")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, cap_kappa=0.5)
b, new, m = run(ctl, [LIVE, SAT, WS, NARROW, SEARCH, STUCK, WS_LIVE])
d = (new - b["advantages"])[:, 0].numpy()
per_traj = {t: float(d[rows_of(b, t)][0]) for t in ("S1", "S2", "S3", "S4")}
check(per_traj["S1"] > 0 and per_traj["S2"] > 0 and per_traj["S4"] < 0, "4-turn winners up, the 12-turn one down")
check(abs(per_traj["S1"] - per_traj["S2"]) < 1e-7, "equal turns, equal push")
sat_rows = np.concatenate([rows_of(b, t) for t in ("S1", "S2", "S3", "S4")])
check(abs(float(d[sat_rows].sum())) < 1e-5, f"zero-sum over the group's rows ({float(d[sat_rows].sum()):+.2e})")
ws_rows = np.concatenate([rows_of(b, t) for t in ("W1", "W2")])
check(float(d[rows_of(b, "W1")][0]) > 0 > float(d[rows_of(b, "W2")][0]) and abs(float(d[ws_rows].sum())) < 1e-5,
      "WebShop: 5 turns up, 6 down, zero-sum")
check(np.all(d[np.concatenate([rows_of(b, t) for t in ("N1", "N2", "Q1", "Q2", "L1", "L2")])] == 0.0),
      "below-spread, Search and live groups get nothing")
S = m["progress_rank/alfworld/success_push_ema"]
top = max(abs(v) for v in per_traj.values())
check(top <= 0.5 * S + 1e-9, f"cap: top push {top:.4f} <= kappa * S = {0.5 * S:.4f}")
check(m["progress_rank/alfworld/sat_top_push"] <= 0.5 * S + 1e-9, "the reported top push respects the cap")
check(m["progress_rank/alfworld/sat_inject_down_invalid"] > 0.0,
      "the slow winner's invalid turns take part of the push-down")
check(m["progress_rank/alfworld/sat_fired"] == 1.0 and m["progress_rank/alfworld/sat_spread_below_min"] == 1.0,
      "fired / below-spread counts")
check(abs(m["progress_rank/alfworld/sat_turn_spread_mean"] - (8.0 + 1.0) / 2) < 1e-9, "mean turn spread")

print("4. the stuck-group term is untouched by sat")
base = pr.ProgressRankController(rho=0.3, sat_rho=0.0)
both = pr.ProgressRankController(rho=0.3, sat_rho=0.2)
b1, n1, _ = run(base, [LIVE, SAT, STUCK, WS_LIVE])
b2, n2, _ = run(both, [LIVE, SAT, STUCK, WS_LIVE])
stuck_rows = np.concatenate([rows_of(b1, t) for t in ("F1", "F2")])
check(torch.equal(n1[stuck_rows], n2[stuck_rows]), "stuck rows identical with and without sat")
check(float((n1 - b1["advantages"])[stuck_rows].abs().sum()) > 0, "and the stuck term did fire")
sat_rows1 = np.concatenate([rows_of(b1, t) for t in ("S1", "S2", "S3", "S4")])
check(torch.equal(n1[sat_rows1], b1["advantages"][sat_rows1]), "sat_rho = 0 leaves the saturated rows alone")

print("5. configuration")
try:
    pr.ProgressRankController(rho=0.0, sat_rho=0.1, sat_tasks=["alfworld", "sokoban"])
    check(False, "an unknown sat task is refused")
except AssertionError:
    check(True, "an unknown sat task is refused")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_min_spread={"alfworld": 1})
b, new, m = run(ctl, [LIVE, NARROW, WS_LIVE])
check(float((new - b["advantages"])[rows_of(b, "N1")].abs().sum()) > 0, "sat_min_spread is read")

sys.exit(0 if ok else 1)
