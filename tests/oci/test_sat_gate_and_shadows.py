"""The saturated-group gate (sat_gate) and the shadow columns (revisits, committed,
done_walkset).

WHAT IT PROTECTS.
  * sat_gate off: bit for bit what sat did before, Search included when listed.
  * sat_gate on: a task's saturated groups fire only while EMA(saturated share) >
    EMA(stuck share); a stuck-heavy step gives verdict "gate_closed" and adds
    nothing, a run of saturated-heavy steps opens it and the term fires; the
    condition and q are reported every step; the EMAs survive a state round-trip.
  * Search is held by the CONDITION, not by name: listed and stuck-heavy, it is
    gated; listed with the gate off, it fires.
  * Shadows: RevisitCounter counts exact repeats; AlfworldWalkSet.raw_done is the
    done count with no won => K; the Search manager writes revisits and committed;
    the loop carries the columns (NaN elsewhere); the controller records them per
    trajectory and reports win/fail revisits, fail_uncommitted and, in fired
    saturated groups, how often revisits order the winners like turns do.
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
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import progress as P  # noqa: E402
from verl.trainer.ppo import progress_rank as pr  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def build(groups, resp=4):
    """Rows from ``[(uid, task, [(traj, turns, reward, k, K, adv, revisits, committed), ...]), ...]``."""
    cols = {k: [] for k in ("uids", "tuids", "tasks", "rew", "ks", "Ks", "adv", "rv", "cm")}
    for uid, task, trajs in groups:
        for traj, turns, r, k, K, a, rv, cm in trajs:
            for _ in range(turns):
                for key, v in zip(cols, (uid, traj, task, r, k, K, a, rv, cm)):
                    cols[key].append(v)
    n = len(cols["uids"])
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    obj = lambda v: np.array(v, dtype=object)  # noqa: E731
    return dict(advantages=torch.tensor(cols["adv"], dtype=torch.float32).unsqueeze(-1) * mask, mask=mask,
                uids=obj(cols["uids"]), tuids=obj(cols["tuids"]), task_names=obj(cols["tasks"]),
                episode_rewards=obj(cols["rew"]), k_rows=obj(cols["ks"]), total_rows=obj(cols["Ks"]),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                revisit_rows=np.array([float("nan") if v is None else float(v) for v in cols["rv"]]),
                committed_rows=np.array([float("nan") if v is None else float(v) for v in cols["cm"]]))


def rows_of(b, traj):
    return np.where(b["tuids"] == traj)[0]


# ALFWorld: a live group (the scales), a saturated group 4/8/12 turns whose revisits
# 0/2/5 order like turns, stuck groups to weigh the gate down.
LIVE = ("live", "alfworld", [("L1", 3, 10.0, 2, 2, 1.0, 0, None), ("L2", 3, 0.0, 1, 2, -1.0, 4, None)])
SAT = ("sat", "alfworld", [("S1", 4, 10.0, 2, 2, 0.0, 0, None), ("S2", 8, 10.0, 2, 2, 0.0, 2, None),
                           ("S3", 12, 10.0, 2, 2, 0.0, 5, None)])
STUCK = lambda i: (f"stuck{i}", "alfworld", [(f"F{i}a", 5, 0.0, 1, 2, 0.0, 3, None), (f"F{i}b", 5, 0.0, 0, 2, 0.0, 6, None)])  # noqa: E731
SAT2 = lambda i: (f"sat{i}", "alfworld", [(f"G{i}a", 4, 10.0, 2, 2, 0.0, 0, None), (f"G{i}b", 9, 10.0, 2, 2, 0.0, 1, None)])  # noqa: E731
# Search: a live group and a saturated group 1 vs 3 turns (spread 2 >= min 1), committed flags on.
SLIVE = ("slive", "search", [("Q1", 2, 1.0, 1, 1, 1.0, 0, True), ("Q2", 3, 0.0, 0, 1, -1.0, 1, True)])
SSAT = ("ssat", "search", [("W1", 1, 1.0, 1, 1, 0.0, 0, True), ("W2", 3, 1.0, 1, 1, 0.0, 1, True)])
SSTUCK = lambda i: (f"sstuck{i}", "search", [(f"X{i}a", 4, 0.0, 1, 1, 0.0, 2, False), (f"X{i}b", 4, 0.0, 0, 1, 0.0, 1, True)])  # noqa: E731


def run(ctl, groups):
    b = build(groups)
    new, m = ctl.apply(**b)
    return b, new, m


def sat_delta(b, new, trajs):
    return float((new - b["advantages"])[np.concatenate([rows_of(b, t) for t in trajs])].abs().sum())


print("1. gate off: what sat did before, Search included when listed")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"])
b, new, m = run(ctl, [LIVE, SAT, STUCK(1), STUCK(2), STUCK(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["S1", "S2", "S3"]) > 0,
      "ALFWorld's saturated group fires on a stuck-heavy step")
check(recs["ssat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["W1", "W2"]) > 0,
      "...and so does Search's, by name")
check(m["progress_rank/alfworld/sat_gate_condition"] == 0.0 and m["progress_rank/alfworld/sat_gated_out"] == 0.0,
      f"the condition is reported (closed, q={m['progress_rank/alfworld/sat_gate_q']:.2f}) but not enforced")

print("2. gate on: a stuck-heavy step is held, saturated-heavy steps open it")
ctl = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"], sat_gate=True)
b, new, m = run(ctl, [LIVE, SAT, STUCK(1), STUCK(2), STUCK(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["sat_verdict"] == "gate_closed" and sat_delta(b, new, ["S1", "S2", "S3"]) == 0.0,
      "ALFWorld: verdict gate_closed, nothing added")
check(recs["ssat"]["sat_verdict"] == "gate_closed" and sat_delta(b, new, ["W1", "W2"]) == 0.0,
      "Search: held by the condition")
check(m["progress_rank/alfworld/sat_gated_out"] == 1.0 and m["progress_rank/alfworld/sat_gate_stuck_ema"] > m["progress_rank/alfworld/sat_gate_sat_ema"],
      "metrics: gated out, stuck EMA above saturated EMA")
check(m.get("progress_rank/alfworld/sat_gate_closed") == 1.0, "the verdict is counted")
opened_at = None
for step in range(2, 40):
    b, new, m = run(ctl, [LIVE, SAT, SAT2(step), SAT2(step + 100), SAT2(step + 200), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
    if m["progress_rank/alfworld/sat_gate_condition"] == 1.0:
        opened_at = step
        break
recs = {r["uid"]: r for r in ctl.last_group_records}
check(opened_at is not None and recs["sat"]["sat_verdict"] == "fired" and sat_delta(b, new, ["S1", "S2", "S3"]) > 0,
      f"ALFWorld opens after {opened_at} saturated-heavy steps and fires")
check(recs["ssat"]["sat_verdict"] == "gate_closed", "Search, still stuck-heavy, stays held on the same step")
check(0.0 <= m["progress_rank/alfworld/sat_gate_q"] < 0.5, f"q < 1/2 when open ({m['progress_rank/alfworld/sat_gate_q']:.2f})")

print("3. the EMAs survive a state round-trip")
state = ctl.state_dict()
ctl2 = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_tasks=["alfworld", "webshop", "search"], sat_gate=True)
ctl2.load_state_dict(state)
check(state["version"] == 4 and ctl2.gate_stuck_ema == ctl.gate_stuck_ema and ctl2.gate_sat_ema == ctl.gate_sat_ema,
      "version 4 carries both gate EMAs")
ctl3 = pr.ProgressRankController(rho=0.0, sat_rho=0.2, sat_gate=True)
ctl3.load_state_dict({"version": 3, "ema": {}, "success_ema": {}})
check(all(v is None for v in ctl3.gate_stuck_ema.values()), "a version-3 state loads with the gate EMAs unset")

print("4. the shadows in the controller")
b, new, m = run(ctl, [LIVE, SAT, SAT2(1), SAT2(2), SAT2(3), SLIVE, SSAT, SSTUCK(1), SSTUCK(2)])
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["sat"]["revisits"] == [0.0, 2.0, 5.0] and recs["sstuck1"]["committed"] == [False, True],
      "records carry revisits and committed per trajectory")
check(m["progress_rank/alfworld/sat_revisit_agreement"] == 1.0, "revisits order the 4/8/12-turn winners like turns: agreement 1")
check(abs(m["traj/search/fail_uncommitted"] - 0.4) < 1e-9, "Search failures that never answered: 2 of 5")
check("traj/alfworld/fail_uncommitted" not in m, "ALFWorld has no terminal action: nothing reported")
check(abs(m["traj/alfworld/win_revisits"] - np.mean([0, 0, 2, 5, 0, 1, 0, 1, 0, 1])) < 1e-9, "win_revisits is the winners' mean")
REV = ("rev", "alfworld", [("R1", 4, 10.0, 2, 2, 0.0, 5, None), ("R2", 12, 10.0, 2, 2, 0.0, 0, None)])
b, new, m = run(ctl, [LIVE, REV, SAT2(1), SAT2(2), SAT2(3)])
check(m["progress_rank/alfworld/sat_revisit_agreement"] < 1.0, "a group where the fast winner revisited more lowers the agreement")

print("5. the counters")
rc = P.RevisitCounter()
for a in ("go to cabinet 1", "Go To  Cabinet 1", "go to cabinet 2", "", "go to cabinet 2"):
    rc.step(a)
check(rc.revisits == 2, "exact repeats (case / spaces folded, instance numbers kept): 2")
w = P.AlfworldWalkSet(["go to desk 1", "take bowl 1 from desk 1", "go to dresser 1", "use desklamp 1"])
w.step("go to desk 1", "You arrive at desk 1.")
w.step("take bowl 1 from desk 1", "You pick up the bowl 1.", won=True)
check(w.k == w.total == 4 and w.raw_done == 2, "won => k = K, but raw_done stays at the 2 lines done")

print("6. the Search manager writes revisits and committed; the loop carries them")
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.search.projection import search_projection  # noqa: E402


class _Envs:
    group_n = 2
    is_train = True

    def reset(self, kwargs=None):
        return ["q"] * 2, [{} for _ in range(2)]

    def step(self, actions):
        return ["<information> Paris. </information>"] * 2, [0.0, 0.0], [False, False], [{"won": 0.0} for _ in actions]


cfg = OmegaConf.create({"env": {"history_length": 0, "rollout": {"n": 2}},
                        "algorithm": {"progress_rank": {"enable": True}, "oci_sat": {"enable": False},
                                      "oci_slots": {"enable": False}, "oci_rank": {"enable": False}}})
mgr = SearchEnvironmentManager(_Envs(), search_projection, cfg)
mgr.reset([{"question": "q", "ground_truth": {"target": ["Paris"]}}] * 2)
_, _, _, i1 = mgr.step(["<search> capital of france </search>", "<search> a </search>"])
_, _, _, i2 = mgr.step(["<search> capital of france </search>", "<answer> Paris </answer>"])
check([i["revisits"] for i in i2] == [1, 0] and [i["committed"] for i in i2] == [False, True],
      "row 0 repeated its query (revisits 1), row 1 answered (committed)")
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402

c = TrajectoryCollector.__new__(TrajectoryCollector)
c._progress_rank_on = True
c._queue_row_for_prefetch = lambda *a: None
batch = DataProto.from_single_dict({"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                    "traj_uid": np.array(["a", "b"], dtype=object)})
tbl, tinf = [[], []], [[], []]
c._record_turn(batch=batch, active_idx=np.array([0, 1]), active_masks=np.array([True, True]),
               infos=[dict(i2[1]), {"progress_done_walkset": 3, "revisits": 7}], traj_uid=np.array(["a", "b"], dtype=object),
               total_batch_list=tbl, total_infos=tinf, batch_size=2)
r0, r1 = tbl[0][0], tbl[1][0]
check(r0["committed"] == 1.0 and r0["revisits"] == 0.0 and np.isnan(r0["progress_done_walkset"]),
      "a Search row: committed 1, revisits 0, no walkthrough count")
check(np.isnan(r1["committed"]) and r1["revisits"] == 7.0 and r1["progress_done_walkset"] == 3.0,
      "an ALFWorld-like row: no committed flag, revisits and the walkthrough count")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
