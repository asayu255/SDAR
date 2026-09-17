"""The progress probe: per-trajectory table, the k-vs-reward curve, and its hook.

WHAT IT PROTECTS.
  * One record per trajectory, from the real rows only; won = environment reward > 0.
  * The curve bins by k/K and by absolute k, and the live-group AUC compares
    rollouts of the SAME prompt (a hard game cannot pass for low progress).
  * Groups from different batches never merge, even when a uid repeats.
  * (a)'s verdicts come from the trainer's own verdict code.
  * The trainer hook writes one JSONL per batch and refuses to run without the
    progress columns.
No model, no GPU.
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl.trainer.ppo import progress_probe as pp  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def rows_for(trajs):
    """Rows from [(uid, traj, task, reward, k, K, turns)], k growing over the turns."""
    u, t, n, r, ks, Ks, g = [], [], [], [], [], [], []
    for uid, traj, task, rew, k, K, turns in trajs:
        for j in range(turns):
            u.append(uid); t.append(traj); n.append(task); r.append(rew)
            ks.append(min(k, j + 1) if j < turns - 1 else k); Ks.append(K); g.append("")
    arr = lambda x: np.array(x, dtype=object)  # noqa: E731
    return dict(uids=arr(u), tuids=arr(t), task_names=arr(n), episode_rewards=arr(r),
                k_rows=arr(ks), total_rows=arr(Ks), real_rows=np.ones(len(u), dtype=bool),
                gamefiles=arr(g))


print("1. the table")
cols = rows_for([("g1", "a", "alfworld", 10.0, 6, 6, 4), ("g1", "b", "alfworld", 0.0, 2, 6, 5)])
tab = {x["traj"]: x for x in pp.trajectory_table(**cols)}
check(set(tab) == {"a", "b"} and tab["a"]["turns"] == 4 and tab["b"]["turns"] == 5,
      "one record per trajectory, with its turn count")
check(tab["a"]["won"] and not tab["b"]["won"] and tab["a"]["k"] == 6 and tab["b"]["k"] == 2,
      "won is reward > 0; k is the trajectory's final count")
cols["real_rows"][-1] = False
cols["k_rows"][-1] = 99
tab = {x["traj"]: x for x in pp.trajectory_table(**cols)}
check(tab["b"]["turns"] == 4 and tab["b"]["k"] == 2, "padding copies are not counted, even with a different value")

print("2. the bins")
check(pp.k_bin(0, 6) == "k=0" and pp.k_bin(2, 6) == "0<k/K<=1/3" and pp.k_bin(4, 6) == "1/3<k/K<=2/3"
      and pp.k_bin(5, 6) == "2/3<k/K<1" and pp.k_bin(6, 6) == "k=K" and pp.k_bin(1, 0) is None,
      "k/K falls in the right bin; K = 0 is no sequence")

print("3. the summary")
trajs = []
for batch in (1, 2):
    # uid "g" repeats across batches: two groups, never one
    trajs += [dict(batch=batch, uid="g", traj=f"{batch}w", task="webshop", reward=10.0, won=True, k=5, K=5),
              dict(batch=batch, uid="g", traj=f"{batch}l", task="webshop", reward=0.0, won=False, k=2, K=5)]
trajs += [dict(batch=1, uid="s", traj="s1", task="webshop", reward=0.0, won=False, k=3, K=5),
          dict(batch=1, uid="s", traj="s2", task="webshop", reward=0.0, won=False, k=1, K=5),
          dict(batch=1, uid="t", traj="t1", task="webshop", reward=0.0, won=False, k=1, K=5),
          dict(batch=1, uid="t", traj="t2", task="webshop", reward=0.0, won=False, k=1, K=5)]
s = pp.summarise_progress(trajs)["webshop"]
check(s["groups"] == 4 and s["groups_live"] == 2 and s["groups_stuck"] == 2,
      "a uid reused in another batch is another group")
check(s["bins"]["k=K"] == {"n": 2, "wins": 2, "win_rate": 1.0}
      and s["bins"]["1/3<k/K<=2/3"]["win_rate"] == 0.0, "the curve counts wins per bin")
check(s["by_k"][5]["win_rate"] == 1.0 and s["by_k"][1]["n"] == 3, "and per absolute k")
check(abs(s["live_auc"] - 1.0) < 1e-12 and s["live_pairs"] == 2,
      "live-group AUC compares a winner and a loser of the same group")
check(s["stuck_verdicts"] == {"fired": 1, "no_difference": 1, "top_below_min": 0, "no_progress": 0},
      "(a)'s verdicts: one stuck group differs and reaches 2 steps, one does not differ")
tie = [dict(batch=1, uid="h", traj="x", task="search", reward=1.0, won=True, k=1, K=1),
       dict(batch=1, uid="h", traj="y", task="search", reward=0.0, won=False, k=1, K=1)]
check(pp.summarise_progress(tie)["search"]["live_auc"] == 0.5, "a tie counts half")
check(any("stuck groups for (a)" in line for line in pp.format_progress_report({"webshop": s})),
      "the report prints the verdict line")

print("4. a second count on the same rollouts")
cols = rows_for([("g1", "a", "alfworld", 10.0, 0, 7, 3), ("g1", "b", "alfworld", 0.0, 2, 7, 3)])
km = np.array([1, 2, 3, 1, 1, 1], dtype=object)
Km = np.array([3, 3, 3, 3, 3, 3], dtype=object)
tab = {x["traj"]: x for x in pp.trajectory_table(**cols, variants={"milestone": (km, Km)})}
check(tab["a"]["k_milestone"] == 3 and tab["a"]["K_milestone"] == 3 and tab["b"]["k_milestone"] == 1,
      "each record carries k_<name> / K_<name> beside k / K")
recs = pp.variant_records(list(tab.values()), "milestone")
check(recs and all(r["k"] == r["k_milestone"] and r["K"] == 3 for r in recs)
      and pp.summarise_progress(recs)["alfworld"]["live_auc"] == 1.0
      and pp.summarise_progress(list(tab.values()))["alfworld"]["live_auc"] == 0.0,
      "and summarises on its own: here the milestones rank the winner first, the walkthrough last")

print("5. the trainer hook")
from verl import DataProto  # noqa: E402
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402

tmp = tempfile.mkdtemp()
cfg = OmegaConf.create({"algorithm": {"progress_rank": {"enable": True, "rho": 0.0, "min_top_k": {}}},
                        "env": {"env_name": "multitask"},
                        "trainer": {"grad_probe": {"out_path": os.path.join(tmp, "p.json")}}})
t = object.__new__(OPDRayTrainer)
t.config = cfg
t.global_steps = 75
c = rows_for([("g1", "a", "alfworld", 10.0, 6, 6, 2), ("g1", "b", "alfworld", 0.0, 1, 6, 2)])
nt = {"uid": c["uids"], "traj_uid": c["tuids"], "task_name": c["task_names"],
      "episode_rewards": c["episode_rewards"], "progress_k": c["k_rows"], "progress_total": c["total_rows"],
      "gamefile": c["gamefiles"]}
batch = DataProto.from_dict(tensors={"is_padding_row": torch.zeros(4, dtype=torch.bool)}, non_tensors=nt)
state = {}
t._accumulate_progress_probe(batch, state, cfg.trainer.grad_probe)
dumped = [json.loads(l) for l in open(os.path.join(tmp, "p.json.trajs", "b1.jsonl"))]
check(len(dumped) == 2 and all(d["batch"] == 1 and d["global_step"] == 75 for d in dumped),
      "one JSONL per batch, a record per trajectory, tagged with batch and step")
check(state["batches"] == 1 and state["progress"]["alfworld"]["live_auc"] == 1.0,
      "the running summary is on the probe state")
nt2 = {k: v for k, v in nt.items() if k != "progress_k"}
try:
    t._accumulate_progress_probe(DataProto.from_dict(tensors={"is_padding_row": torch.zeros(4, dtype=torch.bool)},
                                                     non_tensors=nt2), {}, cfg.trainer.grad_probe)
    check(False, "a batch without progress columns is refused")
except AssertionError as e:
    check("progress_k" in str(e), "a batch without progress columns is refused")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
