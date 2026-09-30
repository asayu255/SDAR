"""progress_value_gae inside the trainer: compute_advantage, the OPD+GRPO hook, the checkpoint, the launcher.

What has to hold, and why each part is tested (the estimator's arithmetic itself is
tests/trainer/test_progress_value.py's):

- compute_advantage on a batch built the way the rollout builds one -- rows collated by
  collate_fn, adjust_batch's padding copies, then reordered as _balance_batch does -- puts the
  module's per-row scalar on every response token of its row (0 off the mask), a padding copy
  gets its original's value, and 'returns' is a tensor of its own. It only READS the table: the
  batch's update is left staged and reaches the table at the commit, which is the ordinary
  decay-then-add.
- The OPD+GRPO hook scores a step against the table as the previous steps left it and commits
  right after, so the next step sees this one; its metrics reach the step's metrics.
- The combinations that would not mean what they say are refused at launch
  (check_progress_value_config), and progress_value.enable beside another estimator is
  records-only.
- The table survives a checkpoint: saved beside global_step_N by the shared loop, restored on
  resume, and a restored table predicts exactly what the saved one did; a table saved under
  another configuration is refused.
- The launcher satisfies its lock at LAM=1.0 and LAM=0.9 (one lock each), and each lock catches
  a change to the estimator, gamma, lam, the progress counters and every progress_value key.

No Ray, no workers, no GPU: the trainer is built with object.__new__, the model checkpoint and
the reward / log-prob workers are stubbed, and the launcher is only parsed and composed.
"""

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments.progress import PV_COLUMNS  # noqa: E402
from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY, adjust_batch  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.ppo import opd_grpo_ray_trainer as grpo_mod  # noqa: E402
from verl.trainer.ppo import progress_value as pv  # noqa: E402
from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer, check_progress_value_config  # noqa: E402
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, compute_advantage  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")
SCRIPT = "examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh"

P, RLEN = 4, 6
TASKS = ("alfworld", "webshop", "search")
CAPS = {"alfworld": 50, "webshop": 15, "search": 4}
WIN = {"alfworld": 10.0, "webshop": 10.0, "search": 1.0}
PENALTY = {"alfworld": 0.1, "webshop": 0.1, "search": 0.01}
GAMEFILES = ("json_2.1.1/train/pick_and_place_simple-Book-None-Desk-310/trial_T1/game.tw-pddl",
             "json_2.1.1/train/pick_two_obj_and_place-Book-None-Desk-311/trial_T2/game.tw-pddl")


# --------------------------------------------------------------------------- #
# A batch as the rollout, adjust_batch and _balance_batch leave it
# --------------------------------------------------------------------------- #

def trajectories(seed, tasks=TASKS, groups=2, size=4):
    """Per task: group 0 mixed (every other rollout wins), group 1 all failures (tied, so the format
    term acts); WebShop's group 1 is on a goal the environment cannot pay (goal_capped)."""
    rng = np.random.default_rng(seed)
    specs = []
    for task in tasks:
        for g in range(groups):
            uid = f"{task}-g{g}-s{seed}"
            for j in range(size):
                specs.append(dict(task=task, uid=uid, tuid=f"{uid}-t{j}", won=(g == 0 and j % 2 == 0),
                                  T=int(rng.integers(1, min(CAPS[task], 7) + 1)),
                                  capped=(task == "webshop" and g == 1), gamefile=GAMEFILES[g]))
    return specs


def rows_of(spec, rng):
    """One trajectory's turn rows, with the pv_* records the managers write (NaN where the task has none)."""
    task, rows, k, stag = spec["task"], [], 0, 0
    for t in range(spec["T"]):
        k_after = min(4, k + int(rng.random() < 0.4))
        last = t == spec["T"] - 1
        resp = int(rng.integers(2, RLEN + 1))
        attn = torch.zeros(P + RLEN, dtype=torch.long)
        attn[:P + resp] = 1
        ids = torch.randint(5, 100, (P + RLEN,))
        row = {"input_ids": ids, "attention_mask": attn, "prompts": ids[:P].clone(), "responses": ids[P:].clone(),
               "uid": spec["uid"], "traj_uid": spec["tuid"], "task_name": task, "turn_step": t,
               "episode_rewards": WIN[task] if spec["won"] else 0.0, "active_masks": True,
               "is_action_valid": np.bool_(rng.random() > 0.2),
               "gamefile": spec["gamefile"] if task == "alfworld" else "",
               "goal_capped": float(spec["capped"]) if task == "webshop" else float("nan")}
        for c in PV_COLUMNS:
            row[c] = float("nan")
        row.update({"pv_t": float(t), "pv_cap": float(CAPS[task]), "pv_K": 4.0,
                    "pv_k_before": float(k), "pv_k_after": float(k_after), "pv_stag_before": float(stag),
                    "pv_env_done": float(last and spec["won"]), "pv_won": float(last and spec["won"]),
                    "pv_term": float(0 if not last else (1 if spec["won"] else 3))})
        stems = {"alfworld": ("hold", "inside", "at"), "webshop": ("ongoal", "optnow"), "search": ("evid",)}[task]
        for stem in stems:
            row[f"pv_{stem}_b"], row[f"pv_{stem}_a"] = float(rng.integers(0, 2)), float(rng.integers(0, 2))
        if task == "webshop":
            # the purchase score of the open product: a continuous value in [0, 1]
            row["pv_buynow_b"], row["pv_buynow_a"] = float(rng.random()), float(rng.random())
        rows.append(row)
        stag = 0 if k_after > k else stag + 1
        k = k_after
    return rows


def rollout_batch(seed, tasks=TASKS, pad_to=7):
    """Rows collated as gather_rollout_data does, padded by adjust_batch, reordered like _balance_batch."""
    rng = np.random.default_rng(seed)
    specs = trajectories(seed, tasks)
    rows = [r for s in specs for r in rows_of(s, rng)]
    batch = DataProto.from_single_dict(collate_fn(rows))
    np.random.seed(seed)
    cfg = OmegaConf.create({"trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                            "algorithm": {"use_kl_in_reward": False},
                            "actor_rollout_ref": {"rollout": {"log_prob_micro_batch_size_per_gpu": pad_to},
                                                  "actor": {"use_kl_loss": False,
                                                            "ppo_micro_batch_size_per_gpu": pad_to}}})
    if len(batch) % pad_to == 0:
        cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = pad_to + 2
        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = pad_to + 2
    batch = adjust_batch(cfg, batch)
    assert PADDING_ROW_KEY in batch.batch.keys(), "the test batch needs padding copies"
    batch.batch["response_mask"] = batch.batch["attention_mask"][:, -RLEN:]
    batch.reorder(torch.as_tensor(rng.permutation(len(batch))))
    return batch, specs


def episode_reward_tensor(batch):
    """EpisodeRewardManager's tensor: the episode reward on the row's last response token."""
    out = torch.zeros(len(batch), RLEN)
    last = batch.batch["response_mask"].sum(-1) - 1
    for i in range(len(batch)):
        out[i, last[i]] = float(batch.non_tensor_batch["episode_rewards"][i])
    return out


def with_token_rewards(batch):
    """token_level_rewards as the pipeline leaves them: episode reward minus the format penalty."""
    tlr = episode_reward_tensor(batch)
    last = batch.batch["response_mask"].sum(-1) - 1
    for i in range(len(batch)):
        ok = bool(batch.non_tensor_batch["is_action_valid"][i])
        tlr[i, last[i]] -= 0.0 if ok else PENALTY[str(batch.non_tensor_batch["task_name"][i])]
    batch.batch["token_level_rewards"] = tlr
    return batch


def columns(batch):
    """The estimator's view of a batch, for the reference computation."""
    cols = dict(batch.non_tensor_batch)
    cols[PADDING_ROW_KEY] = batch.batch[PADDING_ROW_KEY].numpy()
    return cols


def prior_table(cfg, seed=101):
    """A table that has seen one earlier step of ALFWorld and WebShop -- Search is still empty."""
    table = pv.ProgressValueTable(cfg)
    b, _ = rollout_batch(seed, tasks=("alfworld", "webshop"))
    table.stage(pv.compute_progress_value_advantage(columns(b), table).records)
    assert table.commit()
    return table


def pv_config(lam=1.0):
    return pv.ProgressValueConfig.from_config({"enable": True}, gamma=1.0, lam=lam)


# --------------------------------------------------------------------------- #
# compute_advantage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lam", [1.0, 0.9])
def test_compute_advantage_broadcasts_the_module_and_only_stages_the_update(lam):
    cfg = pv_config(lam)
    table = prior_table(cfg)
    before = copy.deepcopy(table.state_dict())
    reference_table = copy.deepcopy(table)
    batch, specs = rollout_batch(2)
    batch = with_token_rewards(batch)
    expected = pv.compute_progress_value_advantage(columns(batch), reference_table)

    out = {}
    batch = compute_advantage(batch, adv_estimator="progress_value_gae", gamma=1.0, lam=lam, multi_turn=False,
                              progress_value_table=table, progress_value_turn_caps=CAPS, progress_value_out=out)
    mask = batch.batch["response_mask"].to(torch.float32)
    adv, ret = batch.batch["advantages"], batch.batch["returns"]

    # The module's scalar on every response token of its row, zero off the mask.
    want = torch.as_tensor(expected.advantage, dtype=torch.float32).unsqueeze(-1) * mask
    assert torch.equal(adv, want)
    assert adv.dtype == torch.float32 and adv.shape == (len(batch), RLEN)
    assert float(adv[mask == 0].abs().max()) == 0.0
    # 'returns' is its own tensor (GRPO hands back one tensor twice) and holds V + A_rl / adv_scale.
    assert ret is not adv and ret.data_ptr() != adv.data_ptr()
    assert torch.equal(ret, torch.as_tensor(expected.returns, dtype=torch.float32).unsqueeze(-1) * mask)
    assert np.array_equal(out["result"].advantage, expected.advantage)

    # A padding copy carries its original's advantage.
    pad = batch.batch[PADDING_ROW_KEY].numpy()
    key = [(str(u), int(t)) for u, t in zip(batch.non_tensor_batch["traj_uid"], batch.non_tensor_batch["pv_t"])]
    real = {key[i]: i for i in range(len(batch)) if not pad[i]}
    assert pad.sum() > 0
    for i in np.flatnonzero(pad):
        assert expected.advantage[i] == expected.advantage[real[key[i]]]

    # Search had no mass: its value is the other rollouts' mean R, the same at every turn, and at
    # lam = 1 its GAE term is 2 (R - that mean). ALFWorld and WebShop were read off the table.
    res = out["result"]
    tasks = batch.non_tensor_batch["task_name"]
    assert res.fallback[tasks == "search"].all() and not res.fallback[tasks != "search"].any()
    if lam == 1.0:
        wins = {}
        for s in specs:
            wins.setdefault(s["uid"], []).append(float(s["won"]))
        for s in (s for s in specs if s["task"] == "search"):
            others = list(wins[s["uid"]])
            others.remove(float(s["won"]))
            rows = np.flatnonzero(batch.non_tensor_batch["traj_uid"] == s["tuid"])
            assert np.allclose(res.a_rl[rows], 2.0 * (float(s["won"]) - np.mean(others)))

    # Only READ: the table is what it was, the batch's update staged on it; the commit is the
    # ordinary decay-then-add of exactly those records.
    assert table.state_dict() == before
    assert table.pending is not None and table.pending == expected.records
    assert table.commit() and table.pending is None
    reference_table.update(expected.records)
    assert table.state_dict() == reference_table.state_dict() != before


def test_compute_advantage_is_row_order_free():
    cfg = pv_config()
    # Two builds of the same step (a deepcopy of a DataProto locks the original's TensorDict).
    batch, _ = rollout_batch(3)
    other, _ = rollout_batch(3)
    other.reorder(torch.as_tensor(np.random.default_rng(9).permutation(len(other))))
    got = {}
    for b in (batch, other):
        b = compute_advantage(b, adv_estimator=AdvantageEstimator.PROGRESS_VALUE_GAE,
                              progress_value_table=prior_table(cfg), progress_value_turn_caps=CAPS)
        row_adv = b.batch["advantages"][:, 0]
        for tu, t, a in zip(b.non_tensor_batch["traj_uid"], b.non_tensor_batch["pv_t"], row_adv.tolist()):
            got.setdefault((str(tu), int(t)), set()).add(a)
    assert all(len(v) == 1 for v in got.values())


def test_compute_advantage_refusals():
    cfg = pv_config()
    with pytest.raises(AssertionError, match="progress_value_table"):
        compute_advantage(rollout_batch(4)[0], adv_estimator="progress_value_gae")
    bare = rollout_batch(4)[0]
    for c in PV_COLUMNS:
        bare.non_tensor_batch.pop(c)
    with pytest.raises(AssertionError, match="progress_value.enable"):
        compute_advantage(bare, adv_estimator="progress_value_gae", progress_value_table=pv.ProgressValueTable(cfg))
    # A cap the trajectory did not run under would key "turns remaining" on the wrong horizon.
    with pytest.raises(ValueError, match="pv_cap disagrees"):
        compute_advantage(rollout_batch(4)[0], adv_estimator="progress_value_gae",
                          progress_value_table=pv.ProgressValueTable(cfg),
                          progress_value_turn_caps={**CAPS, "webshop": 20})
    unset = rollout_batch(4)[0]
    unset.non_tensor_batch["pv_cap"] = np.array([float("nan")] * len(unset), dtype=object)
    with pytest.raises(ValueError, match="nan"):
        compute_advantage(unset, adv_estimator="progress_value_gae", progress_value_table=pv.ProgressValueTable(cfg),
                          progress_value_turn_caps=CAPS)


# --------------------------------------------------------------------------- #
# The OPD+GRPO hook
# --------------------------------------------------------------------------- #

def trainer_config(ckpt_dir, estimator="progress_value_gae", lam=1.0, pv_enable=True, n0=8.0, **alg):
    algorithm = {"adv_estimator": estimator, "gamma": 1.0, "lam": lam, "norm_adv_by_std_in_grpo": True,
                 "use_kl_in_reward": False, "use_pf_ppo": False,
                 "pf_ppo": {"reweight_method": "pow", "weight_pow": 2.0},
                 "gigpo": {"step_advantage_w": 1.0, "mode": "mean_norm", "enable_similarity": False,
                           "similarity_thresh": 0.95},
                 "compute_mean_std_cross_steps": True,
                 "progress_rank": {"enable": False, "alfworld_k": "milestone_arrive"},
                 "progress_value": {"enable": pv_enable, "n0": n0}}
    algorithm.update(alg)
    return OmegaConf.create({
        "algorithm": algorithm,
        "actor_rollout_ref": {
            "actor": {"loss_agg_mode": "token-mean", "use_invalid_action_penalty": True,
                      "invalid_action_penalty_coef": 0.1, "invalid_action_penalty_coef_by_task": dict(PENALTY)},
            "rollout": {"temperature": 1.0, "n": 1, "multi_turn": {"enable": False}}},
        "env": {"env_name": "multitask", "max_steps": 50,
                "multitask": {"tasks": list(TASKS), "max_steps": dict(CAPS)}},
        "trainer": {"default_local_dir": str(ckpt_dir), "resume_mode": "auto", "resume_from_path": None},
    })


def make_trainer(cfg):
    t = object.__new__(OPDGRPORayTrainer)
    t.config = cfg
    t.global_steps = 1
    t.reward_fn = None
    t.actor_rollout_wg = None
    t.traj_collector = SimpleNamespace(take_prefetched_log_probs=lambda: None)
    return t


@pytest.fixture
def stub_workers(monkeypatch):
    """The reward manager and the old-log-prob forward, which need workers: the episode reward on the
    last response token, and zeros."""
    monkeypatch.setattr(grpo_mod, "compute_reward", lambda batch, fn: (episode_reward_tensor(batch), {}))

    def _log_probs(wg, batch, prefetched, temperature):
        z = torch.zeros(len(batch), RLEN)
        return DataProto.from_dict(tensors={"old_log_probs": z, "entropys": z.clone()})

    monkeypatch.setattr(grpo_mod, "compute_log_prob_with_prefetch", _log_probs)


def test_the_hook_scores_against_the_previous_table_and_commits_after(tmp_path, stub_workers):
    t = make_trainer(trainer_config(tmp_path))
    for step, seed in enumerate((11, 12, 13)):
        batch, _ = rollout_batch(seed)
        # What the step must be scored against: the table as the previous steps left it.
        frozen = copy.deepcopy(getattr(t, "_progress_value", None) or pv.ProgressValueTable(pv_config()))
        expected = pv.compute_progress_value_advantage(columns(batch), frozen)
        metrics = {}
        batch, _ = t._reward_and_advantage(batch, metrics, timing_raw={})
        mask = batch.batch["response_mask"].to(torch.float32)
        assert torch.equal(batch.batch["advantages"],
                           torch.as_tensor(expected.advantage, dtype=torch.float32).unsqueeze(-1) * mask)
        # ...and the batch is in the table right after, before the step ends (and before its save).
        table = t._progress_value
        assert table.pending is None and table.updates == step + 1
        frozen.update(expected.records)
        assert table.state_dict() == frozen.state_dict()
        assert metrics["progress_value/table_updates"] == float(step + 1)
        for task in TASKS:
            assert f"progress_value/{task}/mean_abs_a_rl" in metrics
        # Step 1 has an empty table: every task on the group fallback; later steps read the table.
        assert metrics["progress_value/alfworld/fallback_share"] == (1.0 if step == 0 else 0.0)
        # The invalid-action penalty still reaches the reward tensor and its metrics (control's).
        assert "episode/valid_action_ratio/alfworld" in metrics


def test_records_only_and_the_launch_refusals(tmp_path, stub_workers):
    # progress_value.enable beside GRPO records the columns and nothing reads them.
    grpo = trainer_config(tmp_path, estimator="grpo")
    assert check_progress_value_config(grpo) is False
    t = make_trainer(grpo)
    batch, _ = rollout_batch(21)
    assert t._progress_value_kwargs(batch) == {}
    t._reward_and_advantage(batch, {}, timing_raw={})
    assert getattr(t, "_progress_value", None) is None
    assert check_progress_value_config(trainer_config(tmp_path)) is True

    refused = [
        (trainer_config(tmp_path, pv_enable=False), "progress_value.enable"),
        (trainer_config(tmp_path, progress_rank={"enable": True}), "progress_rank"),
        (trainer_config(tmp_path, use_kl_in_reward=True), "use_kl_in_reward"),
        (trainer_config(tmp_path, progress_value={"enable": True, "features": {"search": ["k", "bogus"]}}),
         "bogus"),
    ] + [(trainer_config(tmp_path, **{arm: {"enable": True}}), arm)
         for arm in ("oci_sat", "oci_floor", "oci_slots", "oci_rank")]
    for cfg, what in refused:
        with pytest.raises(AssertionError, match=what):
            check_progress_value_config(cfg)
        with pytest.raises(AssertionError, match=what):
            make_trainer(cfg)._reward_and_advantage(rollout_batch(22)[0], {}, timing_raw={})


def test_the_subclass_still_overrides_only_the_objective_hooks():
    """tests/trainer/test_opd_grpo_arm.py's rule, restated for the new helpers: none of them may
    shadow a name the shared loop defines."""
    overridden = {n for n in vars(OPDGRPORayTrainer) if not n.startswith("__") and hasattr(OPDRayTrainer, n)}
    assert overridden == {"_reward_and_advantage", "_data_metrics", "progress_desc"}
    assert OPDGRPORayTrainer._save_checkpoint is OPDRayTrainer._save_checkpoint


# --------------------------------------------------------------------------- #
# The checkpoint
# --------------------------------------------------------------------------- #

def test_the_table_survives_a_checkpoint(tmp_path, monkeypatch, stub_workers, capsys):
    # The model checkpoint under it (RayPPOTrainer's) is stubbed; the JSON is the shared loop's.
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", lambda self: None)
    t = make_trainer(trainer_config(tmp_path))
    for seed in (31, 32):
        t._reward_and_advantage(rollout_batch(seed)[0], {}, timing_raw={})
    t.global_steps = 40
    t._pre_peek_dataloader_state = None
    t._save_checkpoint()
    path = tmp_path / "global_step_40" / OPDRayTrainer.PROGRESS_VALUE_STATE_FILE
    assert path.exists()

    def _resume_at_40(self):
        self.global_steps = 40
        return 40

    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", _resume_at_40)
    r = make_trainer(trainer_config(tmp_path))
    r._load_checkpoint()
    restored = r._progress_value_table()
    assert restored.state_dict() == t._progress_value.state_dict() and restored.updates == 2
    # Identical predictions: the same batch scores bit for bit the same against both.
    probe, _ = rollout_batch(33)
    a = pv.compute_progress_value_advantage(columns(probe), t._progress_value)
    b = pv.compute_progress_value_advantage(columns(probe), restored)
    assert np.array_equal(a.advantage, b.advantage) and np.array_equal(a.value, b.value)
    assert not a.fallback.any()

    # A table saved under another configuration is refused, not read as other states -- at the
    # load, before the first rollout, not at the first step's advantage a rollout later.
    other = make_trainer(trainer_config(tmp_path, n0=4.0))
    with pytest.raises(ValueError, match="n0"):
        other._load_checkpoint()
    # ...and the same resume under another estimator never reads the table, so it is not refused.
    grpo = make_trainer(trainer_config(tmp_path, estimator="grpo", n0=4.0))
    grpo._load_checkpoint()
    assert grpo._progress_value_pending_state == json.loads(path.read_text())

    # A resume that finds no table says so and starts empty (the group fallback).
    fresh = make_trainer(trainer_config(tmp_path / "elsewhere"))
    fresh._load_checkpoint()
    assert "WARNING: no progress_value_state.json" in capsys.readouterr().out
    assert fresh._progress_value_table().updates == 0 and not fresh._progress_value.roots


def _tracker(ckpt_dir):
    p = os.path.join(str(ckpt_dir), "latest_checkpointed_iteration.txt")
    return open(p).read() if os.path.exists(p) else None


def test_the_table_is_complete_on_disk_before_the_tracker_names_the_step(tmp_path, monkeypatch, stub_workers):
    """A resume trusts latest_checkpointed_iteration.txt alone, and the base save writes it last. The
    table must already be there, whole, when it does: a kill in between would otherwise resume step N
    on an empty table (only a warning to show for it), and a half-written file would crash every
    resume attempt."""
    seen = []

    def _base_save(self):
        # What a resume would find at the moment the tracker names this step.
        path = os.path.join(str(tmp_path), f"global_step_{self.global_steps}",
                            OPDRayTrainer.PROGRESS_VALUE_STATE_FILE)
        seen.append(json.load(open(path)) if os.path.exists(path) else None)
        with open(os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", _base_save)
    t = make_trainer(trainer_config(tmp_path))
    t._reward_and_advantage(rollout_batch(41)[0], {}, timing_raw={})
    t.global_steps, t._pre_peek_dataloader_state = 40, None
    t._save_checkpoint()
    assert seen == [t._progress_value.state_dict()] and _tracker(tmp_path) == "40"
    # The ENV_RESET_PREFETCH branch (the base save under a shadowed dataloader state) too.
    t._reward_and_advantage(rollout_batch(42)[0], {}, timing_raw={})
    t.global_steps = 45
    t._pre_peek_dataloader_state = {"peeked": False}
    t.train_dataloader = SimpleNamespace()
    t._save_checkpoint()
    assert seen[-1] == t._progress_value.state_dict() and _tracker(tmp_path) == "45"

    # A write that fails part-way (a kill, a full disk) leaves no truncated file and no temporary
    # one, keeps an earlier save of the step whole, and never reaches the tracker.
    good = (tmp_path / "global_step_45" / OPDRayTrainer.PROGRESS_VALUE_STATE_FILE).read_text()
    broken = {**t._progress_value.state_dict(), "zz_unserialisable": object()}
    monkeypatch.setattr(t._progress_value, "state_dict", lambda: broken)
    t._pre_peek_dataloader_state = None
    for step in (45, 50):
        t.global_steps = step
        with pytest.raises(TypeError):
            t._save_checkpoint()
        folder = tmp_path / f"global_step_{step}"
        assert sorted(p.name for p in folder.iterdir()) == ([OPDRayTrainer.PROGRESS_VALUE_STATE_FILE]
                                                             if step == 45 else [])
    assert (tmp_path / "global_step_45" / OPDRayTrainer.PROGRESS_VALUE_STATE_FILE).read_text() == good
    assert _tracker(tmp_path) == "45" and len(seen) == 2


def test_a_grad_probe_scores_every_batch_against_the_restored_table(tmp_path, stub_workers):
    """A probe batch takes no optimizer step, and the probe holds the policy still while it reads it.
    The table is part of the advantage, so it is held still too: every probe batch is scored against
    the table the checkpoint restored, as training step N + 1 would be, and the table is not advanced."""
    cfg = trainer_config(tmp_path)
    cfg.trainer.grad_probe = {"enable": True, "mode": "tau", "n_batches": 3}
    t = make_trainer(cfg)
    t._progress_value = prior_table(pv_config())
    frozen = copy.deepcopy(t._progress_value)
    for seed in (51, 52, 53):
        batch, _ = rollout_batch(seed)
        expected = pv.compute_progress_value_advantage(columns(batch), frozen)
        metrics = {}
        batch, _ = t._reward_and_advantage(batch, metrics, timing_raw={})
        mask = batch.batch["response_mask"].to(torch.float32)
        assert torch.equal(batch.batch["advantages"],
                           torch.as_tensor(expected.advantage, dtype=torch.float32).unsqueeze(-1) * mask)
        assert t._progress_value.pending is None
        assert t._progress_value.state_dict() == frozen.state_dict()
        assert metrics["progress_value/table_updates"] == float(frozen.updates) == 1.0


# --------------------------------------------------------------------------- #
# The launcher and its locks
# --------------------------------------------------------------------------- #

def composed(monkeypatch, lam=None, extra=()):
    from tests.trainer.test_run_script_overrides_compose import _overrides

    monkeypatch.setenv("RUN_TAG_SUFFIX", "")
    if lam is None:
        monkeypatch.delenv("LAM", raising=False)
    else:
        monkeypatch.setenv("LAM", lam)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=list(_overrides(SCRIPT)) + list(extra))


def injected(monkeypatch, lam=None, extra=()):
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = composed(monkeypatch, lam, extra)
    inject_opd_grpo_config(cfg)
    return cfg


@pytest.mark.parametrize("lam", [None, "0.9"])
def test_the_launcher_matches_its_lock_at_each_lam(monkeypatch, lam):
    from verl.utils.expected_config import check_expected_config

    cfg = injected(monkeypatch, lam)
    want = "1.0" if lam is None else lam
    assert cfg.trainer.expected_config.endswith(f"expected_multitask_progress_value_gae_lam{want}_config.yaml")
    assert check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config)) == []
    alg = cfg.algorithm
    assert alg.adv_estimator == "progress_value_gae" and alg.gamma == 1.0 and alg.lam == float(want)
    assert alg.progress_value.enable and not alg.progress_rank.enable
    assert (alg.progress_rank.alfworld_k, alg.progress_rank.search_k, alg.progress_rank.webshop_k) == (
        "milestone_arrive", "evidence_answered", "session")
    assert check_progress_value_config(cfg) is True
    # The recipe underneath is v2's: the teacher never retired, validation off during training.
    assert alg.opd.retire.enable is False and cfg.trainer.test_freq == -1


def test_the_two_locks_differ_only_in_lam(monkeypatch):
    from verl.utils.expected_config import check_expected_config, load_expectations

    d = os.path.join(REPO, "examples", "opd_grpo_trainer")
    one = load_expectations(os.path.join(d, "expected_multitask_progress_value_gae_lam1.0_config.yaml"))
    nine = load_expectations(os.path.join(d, "expected_multitask_progress_value_gae_lam0.9_config.yaml"))
    assert {k for k in set(one) | set(nine) if one.get(k) != nine.get(k)} == {"algorithm.lam"}
    # ...so LAM=0.9 against the lam 1.0 lock is exactly one mismatch.
    cfg = injected(monkeypatch, "0.9")
    miss = check_expected_config(cfg, os.path.join(d, "expected_multitask_progress_value_gae_lam1.0_config.yaml"))
    assert [m[0] for m in miss] == ["algorithm.lam"]


CAUGHT = [
    "algorithm.adv_estimator=grpo",
    "algorithm.gamma=0.99",
    "algorithm.lam=0.95",
    "algorithm.progress_rank.alfworld_k=milestone",
    "algorithm.progress_rank.search_k=evidence",
    "algorithm.progress_rank.webshop_k=legacy",
    "algorithm.progress_value.prefix_discount=True",
    "algorithm.progress_value.eta=0.5",
    "algorithm.progress_value.adv_scale=1.0",
    "algorithm.progress_value.n0=4.0",
    "algorithm.progress_value.retention=0.8",
    "algorithm.progress_value.features.search=[k,rem,evid]",
    "algorithm.progress_value.rem_buckets=[0.8,0.6,0.4,0.2]",
    "algorithm.progress_value.stag_buckets=[1,3,6,10]",
    "algorithm.progress_value.format_scope=all",
    "algorithm.progress_value.format_coef=0.5",
]


@pytest.mark.parametrize("override", CAUGHT)
def test_the_lock_catches(monkeypatch, override):
    from verl.utils.expected_config import check_expected_config

    cfg = injected(monkeypatch, extra=[override])
    miss = check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config))
    assert len(miss) == 1, miss


@pytest.mark.parametrize("override,what", [
    ("algorithm.progress_rank.enable=True", "progress_rank"),
    ("algorithm.progress_value.enable=False", "progress_value.enable"),
    ("algorithm.oci_floor.enable=True", "oci_floor"),
    ("algorithm.use_kl_in_reward=True", "use_kl_in_reward"),
])
def test_the_launch_refuses(monkeypatch, override, what):
    with pytest.raises(AssertionError, match=what):
        injected(monkeypatch, extra=[override])
