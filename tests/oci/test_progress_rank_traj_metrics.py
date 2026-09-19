"""What the rollouts looked like, per task: metrics that are measured, never fed back.

WHAT IT PROTECTS.
  * The new inputs (episode lengths, turn caps, other progress counts, WebShop's
    task score) change NOTHING in the advantage: same tensor with rho = sat_rho = 0,
    same values with both on.
  * Winners' turns by group kind (live / saturated), a winner's turns beyond its
    group's fastest win, failures at the task's turn cap, tokens per turn, the
    format of winners, failures' progress and task score -- exact on a hand batch.
  * Turns come from the rollout's episode length when the batch carries it, so a
    row dropped from the batch does not make a capped failure look uncapped.
  * The other ALFWorld count: stuck groups the ranked count ties and it splits.
  * The teacher's first-order effect is remembered over a window of steps and
    travels with the checkpoint; an old checkpoint without it still loads.
  * The per-trajectory records carry the new fields.
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
from verl.trainer.ppo.cross_teacher_kl_weight import gradient_metrics  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def build(groups, resp=4):
    """Rows from ``[(uid, task, [(traj, turns, env_reward, k, K, adv, n_invalid, length,
    k_alt, task_score), ...]), ...]``: every turn row of a trajectory carries the same
    values, three response tokens, and its first ``n_invalid`` turns are invalid."""
    cols = {c: [] for c in ("uid", "tuid", "task", "rew", "k", "K", "adv", "valid", "len", "kalt", "ts")}
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, n_inv, length, kalt, ts in trajs:
            for j in range(turns):
                for c, v in (("uid", uid), ("tuid", traj), ("task", task), ("rew", r), ("k", k), ("K", K),
                             ("adv", a), ("valid", 0 if j < n_inv else 1), ("len", length),
                             ("kalt", kalt), ("ts", ts)):
                    cols[c].append(v)
    n = len(cols["uid"])
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    A = torch.tensor(cols["adv"], dtype=torch.float32).unsqueeze(-1) * mask
    obj = lambda c: np.array(cols[c], dtype=object)
    base = dict(advantages=A, mask=mask, uids=obj("uid"), tuids=obj("tuid"), task_names=obj("task"),
                episode_rewards=obj("rew"), k_rows=obj("k"), total_rows=obj("K"),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                valid_rows=obj("valid"))
    extra = dict(episode_lengths=obj("len"), turn_caps={"alfworld": 50, "webshop": 15, "search": 4},
                 alt_counts={"arrive": (obj("kalt"), np.array([K + 1 if K else K for K in cols["K"]], dtype=object))},
                 task_score_rows=obj("ts"))
    return base, extra


NAN = float("nan")
# ALFWorld (cap 50):
#   live: winners in 10 and 14 turns, failures at the cap (one with 3 rows dropped
#         from the batch: 47 rows, episode length 50), one invalid turn on the slow winner
#   saturated: winners in 8, 8, 12 turns
#   stuck: both at k = 1 of 3 (tied), but only F1 arrived (alt k 2 vs 1): alt splits it
LIVE = ("live", "alfworld", [("L1", 10, 10.0, 3, 3, 1.0, 0, 10, 4, NAN), ("L2", 14, 10.0, 3, 3, 1.0, 1, 14, 4, NAN),
                             ("L3", 50, 0.0, 1, 3, -1.0, 0, 50, 1, NAN), ("L4", 47, 0.0, 2, 3, -1.0, 0, 50, 3, NAN)])
SAT = ("sat", "alfworld", [("S1", 8, 10.0, 3, 3, 0.0, 0, 8, 4, NAN), ("S2", 8, 10.0, 3, 3, 0.0, 0, 8, 4, NAN),
                           ("S3", 12, 10.0, 3, 3, 0.0, 0, 12, 4, NAN)])
STUCK = ("stuck", "alfworld", [("F1", 50, 0.0, 1, 3, 0.0, 0, 50, 2, NAN), ("F2", 50, 0.0, 1, 3, 0.0, 0, 50, 1, NAN)])
# WebShop (cap 15): a stuck group, one bought the wrong thing (score 0.6, 5 turns), one hit the cap
WS = ("ws", "webshop", [("W1", 5, 0.0, 2, 5, 0.0, 0, 5, NAN, 0.6), ("W2", 15, 0.0, 1, 5, 0.0, 2, 15, NAN, 0.0)])
ALL = [LIVE, SAT, STUCK, WS]


def ctl(rho=0.0, sat_rho=0.0):
    c = pr.ProgressRankController(rho=rho, sat_rho=sat_rho, tasks=["alfworld", "webshop", "search"])
    c.ema = {"alfworld": 0.5, "webshop": 0.5, "search": 0.5}
    c.success_ema = {"alfworld": 1.0, "webshop": 1.0, "search": 1.0}
    return c


print("1. the new inputs never touch the advantage")
base, extra = build(ALL)
a0, m0 = ctl().apply(**base)
a1, m1 = ctl().apply(**base, **extra)
check(a0 is base["advantages"] and a1 is base["advantages"], "rho = sat_rho = 0: the very same tensor, with or without them")
b0, _ = ctl(0.1, 0.1).apply(**base)
b1, _ = ctl(0.1, 0.1).apply(**base, **extra)
check(torch.equal(b0, b1) and not torch.equal(b0, base["advantages"]),
      "rho = sat_rho = 0.1: identical advantages with or without them (and the terms did act)")
old_keys = [k for k in m0 if not k.startswith("traj/") and "/alt_" not in k]
check(len(old_keys) > 20 and all(m1.get(k) == m0[k] for k in old_keys
                                 if not (isinstance(m0[k], float) and np.isnan(m0[k]))),
      "every metric that existed before keeps its value")
check(m0["traj/alfworld/fail_turns"] == 49.25 and m1["traj/alfworld/fail_turns"] == 50.0,
      "(the new turn metrics fall back to rows without episode lengths: L4 kept 47 of its 50)")

print("2. turns by group kind, and waste against the group's own fastest win")
m = m1
check(close(m["traj/alfworld/win_turns_live"], 12.0), "live winners: mean of 10 and 14")
check(close(m["traj/alfworld/win_excess_live"], 2.0), "live: (0 + 4) / 2 turns beyond the fastest win")
check(close(m["traj/alfworld/win_turns_saturated"], 28 / 3), "saturated winners: (8 + 8 + 12) / 3")
check(close(m["traj/alfworld/win_excess_saturated"], 4 / 3), "saturated: (0 + 0 + 4) / 3")
check(close(m["traj/alfworld/win_fastest_saturated"], 8.0), "the saturated group's witness: 8 turns")
check(close(m["traj/alfworld/win_turns"], (10 + 14 + 8 + 8 + 12) / 5), "all winners")
check(m["traj/alfworld/win_turns_live_n"] == 2 and m["traj/alfworld/win_turns_saturated_n"] == 3, "with their counts")

print("3. failures: the cap, from the rollout's own length")
check(close(m["traj/alfworld/fail_at_cap"], 1.0),
      "all four ALFWorld failures ran into the 50-turn cap -- including L4, 3 of whose rows were dropped")
check(close(m["traj/alfworld/fail_turns"], 50.0), "and their turns are the episode's, 50")
check(close(m["traj/webshop/fail_at_cap"], 0.5), "WebShop: one bought early, one hit 15")
_, m_nolen = ctl().apply(**base, **{**extra, "episode_lengths": None})
check("traj/alfworld/fail_at_cap" not in m_nolen, "without episode lengths the cap share is not guessed from rows")
_, m_nocap = ctl().apply(**base, **{**extra, "turn_caps": None})
check("traj/alfworld/fail_at_cap" not in m_nocap and "traj/alfworld/fail_turns" in m_nocap,
      "without caps: turns still reported, the cap share is not")

print("4. format, tokens, progress, WebShop's score")
check(close(m["traj/alfworld/win_with_invalid_turn"], 1 / 5), "1 of 5 ALFWorld winners carried an invalid turn")
check(close(m["traj/alfworld/win_invalid_turn_share"], 1 / 52), "1 invalid of 52 winning turns")
check(close(m["traj/webshop/fail_invalid_turn_share"], 2 / 20), "WebShop failures: 2 invalid of 20 turns")
check(close(m["traj/alfworld/win_tokens_per_turn"], 3.0), "three tokens a turn")
check(close(m["traj/alfworld/fail_progress"], (1 / 3 + 2 / 3 + 1 / 3 + 1 / 3) / 4), "failures' k / K")
check(close(m["traj/webshop/fail_task_score"], 0.3), "WebShop failures' purchase score: (0.6 + 0) / 2")
check("traj/alfworld/fail_task_score" not in m, "no score where the task has none")
check(close(m["traj/alfworld/groups_stuck_share"], 1 / 3) and close(m["traj/alfworld/groups_live_share"], 1 / 3)
      and close(m["traj/alfworld/groups_saturated_share"], 1 / 3), "group kinds: one each")
check(m["progress_rank/alfworld/groups_live"] == 1.0, "and the live count beside stuck/saturated")

print("5. the other count on the same stuck groups")
check(m["progress_rank/alfworld/alt_arrive/stuck_split_where_tied"] == 1.0,
      "the stuck group is tied at k = 1 and 'arrived' splits it")
check(m["progress_rank/alfworld/alt_arrive/stuck_tied_where_split"] == 0.0, "and never the reverse here")
check(m["progress_rank/alfworld/alt_arrive/stuck_compared"] == 1.0, "one stuck ALFWorld group compared")
check(not any(k.startswith("progress_rank/webshop/alt_") for k in m), "WebShop has no such count (K = 0): not compared")

print("6. records")
c = ctl()
c.apply(**base, **extra)
rec = {r["uid"]: r for r in c.last_group_records}
check(rec["live"]["length"] == [10, 14, 50, 50], "episode lengths per trajectory (L4: 50, not its 47 rows)")
check(rec["live"]["turns"] == [10, 14, 50, 47], "beside the rows the batch kept")
check(rec["ws"]["task_score"] == [0.6, 0.0] and rec["live"]["task_score"] == [None] * 4, "WebShop's score, None elsewhere")
check(rec["stuck"]["k_arrive"] == [2, 1] and rec["stuck"]["K_arrive"] == [4, 4], "the other count and its K")

print("7. the teacher's first-order effect, over a window of steps")
sums = {"alfworld": {"n": 10, "g_opd_sq": 4.0, "g_grpo_sq": 100.0, "g_dot": -6.0}}
gm = gradient_metrics(sums, prefix="opd")
check(close(gm["opd/alfworld/grpo/first_order"], -0.06), "<g_opd, g_grpo> / |g_grpo|^2 = -6 / 100")
check(close(gm["opd/alfworld/grpo/first_order"],
            gm["opd/alfworld/grpo/grad_cosine"] * gm["opd/alfworld/grpo/grad_norm_ratio"]), "= cosine * norm ratio")
c = ctl()
for i in range(pr.FIRST_ORDER_WINDOW + 5):
    out = c.observe_update({"opd/alfworld/grpo/first_order": 1.0 if i < 20 else -1.0,
                            "opd/webshop/grpo/first_order": float("nan")})
check(out["opd/alfworld/grpo/first_order_window"] == pr.FIRST_ORDER_WINDOW, "the window holds the last 25 steps")
check(close(out["opd/alfworld/grpo/first_order_pos_frac"], 15 / 25), "of which 15 positive (steps 5..19)")
check("opd/webshop/grpo/first_order_pos_frac" not in out, "a NaN is not a measurement")
check(c.observe_update({}) == {}, "coef 0 (no first_order in the update): nothing reported")
c2 = ctl()
c2.load_state_dict(c.state_dict())
check(c2.first_order == c.first_order and c2.ema == c.ema, "the window travels with the checkpoint")
c3 = ctl()
c3.load_state_dict({"version": 2, "ema": {"alfworld": 0.3}, "success_ema": {"alfworld": 0.9}})
check(c3.first_order == {} and c3.ema["alfworld"] == 0.3, "a version-2 checkpoint (no window) still loads")

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
