"""OPD+GiGPO: GiGPO's advantage on the OPD+GRPO loop, and the arm's launcher.

WHAT IT PROTECTS.
  * The OPD loop now computes GiGPO's per-turn discounted returns (``step_rewards``, Eq. 5 of
    2505.10978) -- before this, only RayPPOTrainer.fit did, and ``adv_estimator=gigpo`` could not
    run here. OPDRayTrainer._attach_gigpo_step_returns gives each turn its trajectory's discounted
    return whatever the row order, does nothing for any other estimator, and fit calls it right
    after the rollout, before adjust_batch touches the rows.
  * The invalid-action penalty reaches the step returns as it reaches the token-level scores.
  * compute_advantage's GiGPO branch on the OPD side's columns: inside an ALL-SUCCESS group the
    episode term is zero and the step term alone pushes the action that reached success sooner up
    (the role sat plays); turns that share an anchor and a return get no step term.
  * The launcher composes, matches its own lock (no waiver), differs from the control only where
    the lock says, writes to its own checkpoint directory, and the lock catches a GRPO run.
No model and no GPU.
"""
import inspect
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty, compute_advantage  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


class _Cfg:
    def __init__(self, estimator, gamma=0.95):
        self.config = OmegaConf.create({"algorithm": {"adv_estimator": estimator, "gamma": gamma}})


def rows(trajs, order):
    """A batch of turn rows. ``trajs`` = {traj: [(uid, anchor, step_reward, valid), ...]} in turn
    order; ``order`` lists (traj, turn) in the order the rows sit in the batch."""
    cols = {k: [] for k in ("uid", "traj_uid", "anchor_obs", "rewards", "active_masks", "is_action_valid", "turn")}
    for t, j in order:
        uid, anchor, r, valid = trajs[t][j]
        for key, v in zip(cols, (uid, t, anchor, r, True, valid, j)):
            cols[key].append(v)
    n = len(order)
    nt = {k: np.array(v, dtype=object) for k, v in cols.items() if k != "turn"}
    nt["rewards"] = np.array(cols["rewards"], dtype=np.float32)
    nt["active_masks"] = np.array(cols["active_masks"], dtype=bool)
    nt["is_action_valid"] = np.array(cols["is_action_valid"], dtype=bool)
    return DataProto.from_dict(tensors={"input_ids": torch.zeros(n, 3, dtype=torch.long)}, non_tensors=nt), cols["turn"]


print("1. the OPD loop's step returns")
G = 0.95
TRAJ = {"A": [("g", "s0", 0.0, True), ("g", "s1", 0.0, True), ("g", "s2", 10.0, True)],
        "B": [("g", "s0", 0.0, True), ("g", "x1", 0.0, True), ("g", "x2", 0.0, True), ("g", "x3", 0.0, True),
              ("g", "s2", 10.0, True)]}
ORDER = [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("B", 2), ("A", 2), ("B", 3), ("B", 4)]
b, turns = rows(TRAJ, ORDER)
out = OPDRayTrainer._attach_gigpo_step_returns(_Cfg("gigpo"), b)
sr = out.batch["step_rewards"].tolist()
want = []
for t, j in ORDER:
    L = len(TRAJ[t])
    want.append(10.0 * G ** (L - 1 - j))
check(np.allclose(sr, want, atol=1e-5), f"each turn gets its trajectory's discounted return, rows interleaved "
      f"(A: {[round(x, 3) for x, (t, _) in zip(sr, ORDER) if t == 'A']})")
b2, _ = rows(TRAJ, ORDER)
out2 = OPDRayTrainer._attach_gigpo_step_returns(_Cfg("grpo"), b2)
check("step_rewards" not in out2.batch.keys(), "grpo: nothing is added")
src = inspect.getsource(OPDRayTrainer.fit)
i_gen, i_att = src.find("batch = gen_batch_output"), src.find("self._attach_gigpo_step_returns(batch)")
i_adj, i_slot = src.find("adjust_batch(self.config, batch)"), src.find("self._select_oci_slots(batch, metrics)")
check(0 <= i_gen < i_att < i_slot < i_adj, "fit attaches them right after the rollout, before the slot drop and adjust_batch")

print("2. the invalid-action penalty reaches the step returns")
P, R = 2, 4
b3, _ = rows({"A": [("g", "s0", 0.0, True), ("g", "s1", 10.0, False)]}, [("A", 0), ("A", 1)])
b3.batch["prompts"] = torch.zeros(2, P, dtype=torch.long)
b3.batch["attention_mask"] = torch.ones(2, P + R, dtype=torch.long)
b3.batch["token_level_scores"] = torch.tensor([[0.0, 0, 0, 10.0], [0.0, 0, 0, 10.0]])
b3 = OPDRayTrainer._attach_gigpo_step_returns(_Cfg("gigpo"), b3)
before = b3.batch["step_rewards"].clone()
b3, _m = apply_invalid_action_penalty(b3, invalid_action_penalty_coef=0.1)
d_step = (before - b3.batch["step_rewards"]).tolist()
check(np.allclose(d_step, [0.0, 0.1], atol=1e-6) and abs(float(b3.batch["token_level_scores"][1, -1]) - 9.9) < 1e-6,
      "the invalid turn loses 0.1 in its step return and in its last token's score; the valid turn nothing")

print("3. GiGPO's advantage on these columns: an all-success group")
# S reaches the goal in 2 turns, L in 4; both start at s0 and end at s2 (same anchor, same return).
TR = {"S": [("g", "s0", 0.0, True), ("g", "s2", 10.0, True)],
      "L": [("g", "s0", 0.0, True), ("g", "y1", 0.0, True), ("g", "y2", 0.0, True), ("g", "s2", 10.0, True)]}
OR = [("S", 0), ("S", 1), ("L", 0), ("L", 1), ("L", 2), ("L", 3)]
b4, _ = rows(TR, OR)
b4 = OPDRayTrainer._attach_gigpo_step_returns(_Cfg("gigpo"), b4)
n4 = len(OR)
tlr = torch.zeros(n4, R)
tlr[:, -1] = 10.0                                   # the episode return on every row, as the reward manager writes it
b4.batch["token_level_rewards"] = tlr
b4.batch["response_mask"] = torch.ones(n4, R)
b4 = compute_advantage(b4, adv_estimator="gigpo", gamma=G, step_advantage_w=1.0, gigpo_mode="mean_std_norm",
                       gigpo_enable_similarity=False)
adv = b4.batch["advantages"][:, 0].tolist()
a = {tj: v for tj, v in zip(OR, adv)}
check(all(np.isfinite(adv)), "finite advantages")
check(a[("S", 0)] > 0.5 and a[("L", 0)] < -0.5,
      f"at the shared start state the faster winner's action is pushed up, the slower one's down "
      f"({a[('S', 0)]:+.3f} vs {a[('L', 0)]:+.3f}); the episode term is 0 in an all-success group")
check(abs(a[("S", 1)]) < 1e-4 and abs(a[("L", 3)]) < 1e-4, "the shared final state with equal returns gets no step term")
check(abs(a[("L", 1)]) < 1e-4 and abs(a[("L", 2)]) < 1e-4, "states only one rollout visits form singleton groups: no step term")

print("4. the launcher and its lock")
os.environ.setdefault("RUN_TAG_SUFFIX", "_opd_gigpo")
from hydra import compose, initialize_config_dir  # noqa: E402

from tests.trainer.test_run_script_overrides_compose import _overrides  # noqa: E402
from verl.trainer.main_opd_grpo import inject_opd_grpo_config  # noqa: E402
from verl.utils.expected_config import check_expected_config  # noqa: E402

WRAP = "examples/opd_grpo_trainer/run_multitask_opd_gigpo_qwen3.sh"
CTRL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"


def composed(script, extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "verl/trainer/config")):
        return compose(config_name="ppo_trainer", overrides=list(_overrides(script)) + list(extra))


cfg = composed(WRAP)
inject_opd_grpo_config(cfg)
LOCK = os.path.join(REPO, cfg.trainer.expected_config)
check(cfg.trainer.expected_config.endswith("expected_multitask_opd_gigpo_config.yaml")
      and check_expected_config(cfg, LOCK) == [], "the wrapper composes and matches its own lock, no waiver")
check(cfg.algorithm.adv_estimator == "gigpo" and cfg.algorithm.gamma == 0.95
      and cfg.algorithm.gigpo.mode == "mean_std_norm" and not cfg.algorithm.progress_rank.enable
      and cfg.actor_rollout_ref.actor.teacher_kl_loss_coef == 0.01 and cfg.trainer.test_freq == -1,
      "gigpo (gamma .95, mean_std_norm), (a) off, the control's teacher coefficient 0.01, no in-training validation")
c12 = composed(WRAP, ["algorithm.adv_estimator=grpo"])
inject_opd_grpo_config(c12)
check([m[0] for m in check_expected_config(c12, LOCK)] == ["algorithm.adv_estimator"], "the lock catches a GRPO run")
# The control runs untagged (RUN_TAG_SUFFIX empty); the wrapper exports RUN_TAG=opd_gigpo.
_suffix = os.environ.get("RUN_TAG_SUFFIX", "")
os.environ["RUN_TAG_SUFFIX"] = ""
try:
    ctrl = composed(CTRL)
finally:
    os.environ["RUN_TAG_SUFFIX"] = _suffix
check(cfg.trainer.default_local_dir != ctrl.trainer.default_local_dir
      and cfg.trainer.default_local_dir.endswith("_opd_gigpo"),
      f"its own checkpoint directory ({os.path.basename(cfg.trainer.default_local_dir)})")


def flat(c, prefix=""):
    out = {}
    for k, v in c.items():
        key = f"{prefix}{k}"
        if OmegaConf.is_dict(v):
            out.update(flat(v, key + "."))
        else:
            out[key] = OmegaConf.to_container(v) if OmegaConf.is_config(v) else v
    return out


inject_opd_grpo_config(ctrl)
fw, fc = flat(cfg), flat(ctrl)
diff = sorted(k for k in set(fw) | set(fc) if fw.get(k) != fc.get(k))
IDENT = {"trainer.expected_config", "trainer.project_name", "trainer.experiment_name", "trainer.default_local_dir",
         "trainer.val_instance_log_dir", "trainer.sign_token_dump_dir"}
# gigpo.mode differs from the config default the control carries (mean_norm), which GRPO never reads.
EXPECTED = {"algorithm.adv_estimator", "algorithm.gamma", "algorithm.gigpo.mode", "trainer.test_freq",
            "actor_rollout_ref.model.enable_gradient_checkpointing"}
spec = {k for k in diff if ".speculative_config." in k}
rest = set(diff) - IDENT - EXPECTED - spec
check(not rest, f"against the control it differs only in the estimator and its knobs, test_freq, gradient "
      f"checkpointing, speculative decoding and the run's identity (other: {sorted(rest)})")

print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
