"""(a) inside the OPD+GRPO trainer: the hook, its guards, and its checkpointed EMA.

WHAT IT PROTECTS.
  * The hook reads the columns the rollout loop writes and changes only the
    advantages of a stuck group that fired.
  * A launch error fails loudly: the switch on without the progress columns, or
    (a) together with an OCI arm, which would stop it being measured against
    control with nothing else changed.
  * The EMA is saved beside the checkpoint and restored on resume, so a resumed
    run does not restart every task's scale from one step.
The trainer is built with object.__new__ and fed a DataProto directly: no Ray,
no workers, no GPU.
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

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer  # noqa: E402
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def make_config(rho=0.05, ckpt_dir="/nonexistent", **arms):
    alg = {"adv_estimator": "grpo", "compute_mean_std_cross_steps": True,
           "progress_rank": {"enable": True, "rho": rho, "ema_alpha": 0.2, "ema_floor": 0.01,
                             "cap_kappa": 100.0,
                             "min_top_k": {"alfworld": 2, "webshop": 2, "search": 1},
                             "tasks": ["alfworld", "webshop", "search"]}}
    for name in ("oci_sat", "oci_floor", "oci_slots", "oci_rank"):
        alg[name] = {"enable": bool(arms.get(name, False))}
    return OmegaConf.create({
        "algorithm": alg,
        "actor_rollout_ref": {"rollout": {"multi_turn": {"enable": False}}},
        "env": {"env_name": "multitask"},
        "trainer": {"default_local_dir": ckpt_dir, "resume_mode": "auto", "resume_from_path": None},
    })


def make_batch(with_progress=True):
    # A live alfworld group (a success) and a stuck one whose rollouts differ.
    rows = [("live", "l1", 10.0, 6, 6, 1.0), ("live", "l2", 0.0, 1, 6, -1.0),
            ("stuck", "s1", 0.0, 4, 6, 0.0), ("stuck", "s1", 0.0, 4, 6, 0.0),
            ("stuck", "s2", 0.0, 1, 6, 0.0), ("stuck", "s2", 0.0, 1, 6, 0.0)]
    n, P, R = len(rows), 3, 4
    attn = torch.ones(n, P + R, dtype=torch.long)
    attn[:, -1] = 0
    adv = torch.tensor([r[5] for r in rows], dtype=torch.float32).unsqueeze(-1) * attn[:, -R:].float()
    nt = {"uid": np.array([r[0] for r in rows], dtype=object),
          "traj_uid": np.array([r[1] for r in rows], dtype=object),
          "task_name": np.array(["alfworld"] * n, dtype=object),
          "episode_rewards": np.array([r[2] for r in rows], dtype=object)}
    if with_progress:
        nt["progress_k"] = np.array([float(r[3]) for r in rows], dtype=object)
        nt["progress_total"] = np.array([float(r[4]) for r in rows], dtype=object)
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(n, R, dtype=torch.long), "attention_mask": attn,
                 "advantages": adv, "is_padding_row": torch.zeros(n, dtype=torch.bool)},
        non_tensors=nt)


def trainer(cfg):
    t = object.__new__(OPDGRPORayTrainer)
    t.config = cfg
    t.global_steps = 0
    return t


print("1. the hook")
t = trainer(make_config())
t._progress_rank = None
b = make_batch()
t._progress_rank_controller(t.config.algorithm.progress_rank).ema["alfworld"] = 0.5
before = b.batch["advantages"].clone()
m = t._apply_progress_rank(b, t.config.algorithm.progress_rank)
after = b.batch["advantages"]
check(torch.equal(after[:2], before[:2]), "the live group is untouched")
check(float(after[2, 0]) > 0 and float(after[4, 0]) < 0, "the stuck group is ranked: further up, shorter down")
check(float(after[2, -1]) == 0.0, "masked positions stay zero")
check(m["progress_rank/alfworld/stuck_fired"] == 1.0 and m["progress_rank/alfworld/c"] > 0,
      "and the step reports what fired and at what c")

t0 = trainer(make_config(rho=0.0))
b0 = make_batch()
adv0 = b0.batch["advantages"]
t0._apply_progress_rank(b0, t0.config.algorithm.progress_rank)
check(b0.batch["advantages"] is adv0, "rho = 0: the advantage tensor is the very same object")

print("2. launch errors fail loudly")
try:
    trainer(make_config())._apply_progress_rank(make_batch(with_progress=False),
                                                make_config().algorithm.progress_rank)
    check(False, "a batch without progress columns is refused")
except AssertionError as e:
    check("progress_k" in str(e), "a batch without progress columns is refused")
for arm in ("oci_sat", "oci_floor", "oci_slots", "oci_rank"):
    cfg = make_config(**{arm: True})
    try:
        trainer(cfg)._apply_progress_rank(make_batch(), cfg.algorithm.progress_rank)
        check(False, f"(a) with {arm} is refused")
    except AssertionError as e:
        check(arm in str(e), f"(a) with {arm} is refused")

print("3. the EMA is checkpointed")
tmp = tempfile.mkdtemp()
orig_save, orig_load = OPDRayTrainer._save_checkpoint, OPDRayTrainer._load_checkpoint
try:
    OPDRayTrainer._save_checkpoint = lambda self: None

    def _fake_load(self):
        self.global_steps = 40
        return 40

    OPDRayTrainer._load_checkpoint = _fake_load
    t = trainer(make_config(ckpt_dir=tmp))
    t._progress_rank = None
    ctl = t._progress_rank_controller(t.config.algorithm.progress_rank)
    ctl.ema.update({"alfworld": 0.31, "search": 0.12, "webshop": None})
    t.global_steps = 40
    t._save_checkpoint()
    path = os.path.join(tmp, "global_step_40", OPDGRPORayTrainer.PROGRESS_RANK_STATE_FILE)
    check(os.path.exists(path) and json.load(open(path))["ema"]["alfworld"] == 0.31,
          "saved beside the step's checkpoint")
    r = trainer(make_config(ckpt_dir=tmp))
    r._progress_rank = None
    r._load_checkpoint()
    restored = r._progress_rank_controller(r.config.algorithm.progress_rank)
    check(restored.ema == {"alfworld": 0.31, "search": 0.12, "webshop": None},
          "and restored on resume before the first step uses it")
    fresh = trainer(make_config(ckpt_dir=os.path.join(tmp, "empty")))
    fresh._progress_rank = None
    OPDRayTrainer._load_checkpoint = lambda self: 0
    fresh._load_checkpoint()
    check(all(v is None for v in fresh._progress_rank_controller(
        fresh.config.algorithm.progress_rank).ema.values()), "a fresh run starts uninitialised")
finally:
    OPDRayTrainer._save_checkpoint, OPDRayTrainer._load_checkpoint = orig_save, orig_load

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
