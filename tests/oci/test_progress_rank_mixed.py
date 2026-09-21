"""(a) on MIXED groups (mixed_rho): a live group's failures ranked among themselves.

WHAT IT PROTECTS.
  * mixed_rho = 0 is the existing (a) bit for bit, with mixed groups present and
    still scored in the records.
  * In a mixed group the winners get nothing, the failures that got further get
    more than the ones that did not, and the term sums to zero over the failures'
    rows in the GRPO statistic.
  * THE SIGN GUARD: no failure's advantage is pushed past zero, however large the
    coefficient, and the guard keeps the zero sum (one scale per group).
  * The cap holds, the term waits for both scales, and the stuck-group and
    saturated-group terms are untouched by turning it on.
  * A mixed group with one failure, or with failures whose k are all equal, does
    not fire.
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
    """Rows from ``[(uid, task, [(traj, turns, env_reward, k, K, adv), ...]), ...]``."""
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


def rows_of(b, traj):
    return np.where(b["tuids"] == traj)[0]


def run(ctl, groups):
    b = build(groups)
    new, metrics = ctl.apply(**b)
    return b, new, metrics


# A mixed ALFWorld group: one winner and three failures that got to k = 3, 1, 0 of 4
# (the failures' advantage is -0.3 each: small, so a large term would flip one).
MIX = ("mix", "alfworld", [("W", 4, 10.0, 4, 4, 0.9),
                            ("F3", 5, 0.0, 3, 4, -0.3), ("F1", 5, 0.0, 1, 4, -0.3),
                            ("F0", 5, 0.0, 0, 4, -0.3)])
# a mixed group with ONE failure, and one whose failures all reached the same k
ONE = ("one", "alfworld", [("OW", 3, 10.0, 4, 4, 0.5), ("OF", 3, 0.0, 2, 4, -0.5)])
FLAT = ("flat", "alfworld", [("TW", 3, 10.0, 4, 4, 0.7), ("TF1", 3, 0.0, 2, 4, -0.35),
                              ("TF2", 3, 0.0, 2, 4, -0.35)])
# a live WebShop group to give S a value for WebShop too, a stuck and a saturated ALFWorld group
WSL = ("wsl", "webshop", [("V1", 4, 10.0, 4, 4, 1.0), ("V2", 4, 0.0, 1, 4, -0.5), ("V3", 4, 0.0, 3, 4, -0.5)])
STUCK = ("stuck", "alfworld", [("S1", 5, 0.0, 2, 4, 0.0), ("S2", 5, 0.0, 0, 4, 0.0)])
SAT = ("sat", "alfworld", [("A1", 4, 10.0, 4, 4, 0.0), ("A2", 12, 10.0, 4, 4, 0.0)])
ALL = [MIX, ONE, FLAT, WSL, STUCK, SAT]

print("1. mixed_rho = 0: bit-identical, mixed groups present and still scored")
ctl = pr.ProgressRankController(rho=0.0, mixed_rho=0.0)
b, new, m = run(ctl, ALL)
check(new is b["advantages"], "the very same tensor comes back")
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["mix"]["mixed_verdict"] == "fired", "the 3/1/0 failures are scored as firing")
check(recs["one"]["mixed_verdict"] == "few_failures", "one failure: nothing to rank")
check(recs["flat"]["mixed_verdict"] == "no_difference", "equal failures: nothing to rank")
check(recs["stuck"]["mixed_verdict"] is None and recs["sat"]["mixed_verdict"] is None,
      "stuck and saturated groups are not mixed groups")

print("2. waits for E and S: nothing on a step with no scale")
ctl = pr.ProgressRankController(rho=0.0, mixed_rho=0.3)
b, new, m = run(ctl, [STUCK])
check(new is b["advantages"], "no live group, no S -> nothing added")

print("3. mixed_rho > 0: winners untouched, the ranking, zero-sum, the sign guard")
ctl = pr.ProgressRankController(rho=0.0, mixed_rho=0.3)
b, new, m = run(ctl, ALL)
d = (new - b["advantages"])[:, 0].numpy()
A = b["advantages"][:, 0].numpy()
check(np.allclose(d[rows_of(b, "W")], 0.0), "the winner gets nothing")
f3, f1, f0 = (d[rows_of(b, t)][0] for t in ("F3", "F1", "F0"))
check(f3 > f1 > f0, f"further failures get more ({f3:+.4f} > {f1:+.4f} > {f0:+.4f})")
fail_rows = np.concatenate([rows_of(b, t) for t in ("F3", "F1", "F0")])
check(abs(float(d[fail_rows].sum())) < 1e-6, "the term sums to zero over the failures' rows")
check(bool(((A + d)[fail_rows] <= 1e-7).all()), "no failure is pushed past zero")
check(np.allclose(d[rows_of(b, "OW")], 0.0) and np.allclose(d[rows_of(b, "OF")], 0.0)
      and np.allclose(d[rows_of(b, "TF1")], 0.0), "the groups that did not fire get nothing")

print("4. the guard binds when the coefficient is large, and keeps the zero sum")
ctl = pr.ProgressRankController(rho=0.0, mixed_rho=50.0, cap_kappa=50.0)
b, new, m = run(ctl, ALL)
d = (new - b["advantages"])[:, 0].numpy()
A = b["advantages"][:, 0].numpy()
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["mix"]["mixed_scale"] is not None and recs["mix"]["mixed_scale"] < 1.0,
      f"the scale shrank the group (s = {recs['mix']['mixed_scale']:.3f})")
check(bool(((A + d)[fail_rows] <= 1e-6).all()), "every failure still <= 0 at a huge coefficient")
top = float((A + d)[rows_of(b, "F3")].max())
check(abs(top) < 1e-5, f"the top failure is brought exactly to zero, not past it ({top:+.2e})")
check(abs(float(d[fail_rows].sum())) < 1e-5, "and the zero sum survives the scale")
check(m.get("progress_rank/alfworld/mixed_sign_scaled_groups", 0.0) >= 1.0,
      "the metric reports that the guard had to act")

print("5. the other terms are untouched by turning mixed on")
base = pr.ProgressRankController(rho=0.3, sat_rho=0.2)
both = pr.ProgressRankController(rho=0.3, sat_rho=0.2, mixed_rho=0.3)
b1, n1, _ = run(base, ALL)
b2, n2, _ = run(both, ALL)
d1 = (n1 - b1["advantages"])[:, 0].numpy()
d2 = (n2 - b2["advantages"])[:, 0].numpy()
others = np.concatenate([rows_of(b1, t) for t in ("S1", "S2", "A1", "A2", "W")])
check(np.allclose(d1[others], d2[others]), "stuck, saturated and winner rows are identical")

print("6. the unit guard itself")
check(pr.sign_preserving_scale(np.array([0.2, -0.2]), np.array([-0.1, -0.1])) == 0.5,
      "a +0.2 push on a -0.1 failure is halved")
check(pr.sign_preserving_scale(np.array([0.05, -0.05]), np.array([-0.1, -0.1])) == 1.0,
      "a push that does not reach zero is left whole")
check(pr.sign_preserving_scale(np.array([0.05, -0.05]), np.array([0.0, -0.1])) == 0.0,
      "a failure already at >= 0 leaves nothing to preserve")

print("7. the config reaches the controller")
try:
    pr.ProgressRankController(rho=0.0, mixed_rho=-0.1)
    check(False, "a negative mixed_rho is refused")
except AssertionError:
    check(True, "a negative mixed_rho is refused")
try:
    pr.ProgressRankController(rho=0.0, mixed_rho=0.1, mixed_tasks=["alfworld", "sokoban"])
    check(False, "an unknown mixed task is refused")
except AssertionError:
    check(True, "an unknown mixed task is refused")
ctl = pr.ProgressRankController(rho=0.0, mixed_rho=0.1, mixed_tasks=["alfworld"])
check(ctl.mixed_tasks == ["alfworld"], "mixed_tasks narrows the term")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
