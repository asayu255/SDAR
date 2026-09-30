"""The analysis records (verl/trainer/ppo/rollout_records.py) and their place in the OPD+GRPO hook.

What has to hold, and why each part is tested:

- OBSERVATION ONLY. With records on or off, the same batch and seeds give bit-identical advantages,
  batch tensors and non-tensor columns, row order, training value table (state and update count) and
  np.random / torch / python RNG states afterwards -- over several steps, in the progress_value_gae
  arm at lam 1.0 and 0.9 and in a grpo arm (records only). The records call itself changes none of
  them, and never stages or commits the training table. Files appear only when on.
- The shadow GRPO is compute_advantage's grpo branch: equal, bit for bit, on the same batch (padding
  copies, reordered rows, the invalid-action penalty already in the scores; the statistic over turns
  or trajectories, with and without the std division, with the OCI exclude / floor columns); in a grpo
  arm it is the advantage itself.
- The shadow GiGPO is OPD+GiGPO's estimator: equal, bit for bit, to the reference implementation
  (opd-gigpo's gigpo/core_gigpo.py at b2b393c, read from the repository's objects, never edited) on a
  turn-ordered fixture, and on the balanced, padded batch the trainer hands the records -- where the
  reference's step returns were computed in the rollout's turn order before the padding, as that run
  computed them. Missing anchors or per-turn rewards: not emitted, and nothing labelled GiGPO.
- The traj/<task>/* and think-block metrics equal what progress_rank reports on the same batch, and
  are left to progress_rank when it is on (one writer per key); WebShop's failure progress carries the
  count's name when progress_k holds the session count (webshop_k=session), in both writers.
- The records-only shadow value table is its own object: built by the one builder, advanced once per
  step, never self._progress_value and never in the checkpoint, restored from the records' directory on
  a resume, held still on a grad_probe batch.
- The record holds what the batch had (per group, trajectory and turn); the metrics name their units
  and read the actor's actual policy-gradient weight (the token weight, or the trajectory weight under
  pg_loss_norm=trajectory).
- A failed record costs the step's records, never the step; a combination the records cannot serve is
  refused at launch; the 2x2's launchers turn the records on and their locks pin them.

No Ray, no workers, no GPU: the trainer is built with object.__new__, the reward and log-prob workers
are stubbed, the launchers are only parsed and composed.
"""

import copy
import json
import os
import random
import subprocess
import types
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
from verl.trainer.ppo import rollout_records as rr  # noqa: E402
from verl.trainer.ppo.opd_grpo_ray_trainer import (  # noqa: E402
    OPDGRPORayTrainer,
    check_progress_value_config,
    check_rollout_records_config,
)
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer  # noqa: E402
from verl.trainer.ppo.ray_trainer import (  # noqa: E402
    GRPO_STAT_EXCLUDE_KEY,
    OCI_FLOOR_KEY,
    RayPPOTrainer,
    _get_invalid_action_penalty_coef,
    apply_invalid_action_penalty,
    compute_advantage,
)
from verl.trainer.ppo.task_loss_weights import (  # noqa: E402
    TASK_LOSS_WEIGHT_KEY,
    TASK_PG_LOSS_WEIGHT_KEY,
    attach_task_loss_weights,
)
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "verl", "trainer", "config")

P, RLEN = 4, 6
TASKS = ("alfworld", "webshop", "search")
CAPS = {"alfworld": 50, "webshop": 15, "search": 4}
WIN = {"alfworld": 10.0, "webshop": 10.0, "search": 1.0}
PENALTY = {"alfworld": 0.1, "webshop": 0.1, "search": 0.01}
GAMEFILES = ("json_2.1.1/train/pick_and_place_simple-Book-None-Desk-310/trial_T1/game.tw-pddl",
             "json_2.1.1/train/pick_two_obj_and_place-Book-None-Desk-311/trial_T2/game.tw-pddl")
THINK_ID = 151667
REF_TOKENS = {"alfworld": 5.0, "webshop": 4.0, "search": 3.0}
MINI = 20
GIGPO_REF = "b2b393c"
UNITS = ("row_mean", "row_abs_mean", "traj_sum_mean", "traj_sum_abs_mean", "token_total", "token_abs_total")


# --------------------------------------------------------------------------- #
# A batch as the rollout, adjust_batch, the task weights and _balance_batch leave it
# --------------------------------------------------------------------------- #

def trajectories(seed, tasks=TASKS, size=4):
    """Per task three groups: mixed (every other rollout wins), all failures, all successes. WebShop's
    all-failure group is on a goal the environment cannot pay (goal_capped)."""
    rng = np.random.default_rng(seed)
    specs = []
    for task in tasks:
        for g, kind in enumerate(("mixed", "all_fail", "all_success")):
            uid = f"{task}-g{g}-s{seed}"
            for j in range(size):
                specs.append(dict(task=task, uid=uid, tuid=f"{uid}-t{j}", kind=kind,
                                  won=kind == "all_success" or (kind == "mixed" and j % 2 == 0),
                                  T=int(rng.integers(1, min(CAPS[task], 7) + 1)),
                                  capped=(task == "webshop" and kind == "all_fail"),
                                  gamefile=GAMEFILES[g % 2], goal=100 + g))
    return specs


def rows_of(spec, rng):
    """One trajectory's turn rows: the rollout's columns, the progress counters and the pv_* records. Token
    ids come from the numpy generator, so two builds of one seed are identical. Anchors repeat inside a
    group (every rollout starts at "start"; later rooms are drawn from few) and the same strings recur in
    other groups, which GiGPO must not merge."""
    task, rows, k, stag = spec["task"], [], 0, 0
    for t in range(spec["T"]):
        k_after = min(4, k + int(rng.random() < 0.4))
        last = t == spec["T"] - 1
        resp = int(rng.integers(2, RLEN + 1))
        attn = torch.zeros(P + RLEN, dtype=torch.long)
        attn[:P + resp] = 1
        ids = torch.as_tensor(rng.integers(5, 100, P + RLEN), dtype=torch.long)
        if rng.random() < 0.5:
            ids[P] = THINK_ID                    # the response opens a think block
        won_now = last and spec["won"]
        row = {"input_ids": ids, "attention_mask": attn, "prompts": ids[:P].clone(), "responses": ids[P:].clone(),
               "uid": spec["uid"], "traj_uid": spec["tuid"], "task_name": task, "turn_step": t,
               "episode_rewards": WIN[task] if spec["won"] else 0.0, "episode_lengths": float(spec["T"]),
               "active_masks": True, "rewards": WIN[task] if won_now else 0.0,
               "anchor_obs": "start" if t == 0 else f"room-{int(rng.integers(0, 3))}",
               "is_action_valid": np.bool_(rng.random() > 0.25),
               "gamefile": spec["gamefile"] if task == "alfworld" else "",
               "goal_capped": float(spec["capped"]) if task == "webshop" else float("nan"),
               "goal_id": float(spec["goal"]) if task == "webshop" else float("nan"),
               "progress_k": float(k_after), "progress_total": 4.0,
               "task_score": (1.0 if won_now else float(rng.random() * 0.9) if last else 0.0)
               if task == "webshop" else float("nan"),
               "committed": float(last) if task != "alfworld" else float("nan"),
               "revisits": float(rng.integers(0, 3)),
               "progress_done_walkset": float(k_after) if task == "alfworld" else float("nan"),
               "coverage_d": float(1 + int(rng.integers(0, t + 1)))}
        for c in PV_COLUMNS:
            row[c] = float("nan")
        row.update({"pv_t": float(t), "pv_cap": float(CAPS[task]), "pv_K": 4.0,
                    "pv_k_before": float(k), "pv_k_after": float(k_after), "pv_stag_before": float(stag),
                    "pv_env_done": float(won_now), "pv_won": float(won_now),
                    "pv_term": float(0 if not last else (1 if spec["won"] else 3))})
        for stem in rr.STATE_STEMS[task]:
            row[f"pv_{stem}_b"], row[f"pv_{stem}_a"] = float(rng.integers(0, 2)), float(rng.integers(0, 2))
        if task == "webshop":
            row["pv_buynow_b"], row["pv_buynow_a"] = float(rng.random()), float(rng.random())
        rows.append(row)
        stag = 0 if k_after > k else stag + 1
        k = k_after
    return rows


def turn_ordered_batch(seed, tasks=TASKS):
    """The batch as the rollout gathers it: every trajectory's rows together, in turn order."""
    rng = np.random.default_rng(seed)
    specs = trajectories(seed, tasks)
    rows = [r for s in specs for r in rows_of(s, rng)]
    return DataProto.from_single_dict(collate_fn(rows)), specs


def balanced(batch, seed, weights=None):
    """adjust_batch's padding copies, response_mask, the per-task loss weights (optional) and a reorder like
    _balance_batch's, in the trainer's order. ``batch`` is not modified (adjust_batch concatenates)."""
    pad_to = next(d for d in (7, 8, 9, 10, 11, 13) if len(batch) % d)    # a divisor that needs copies
    cfg = OmegaConf.create({"trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                            "algorithm": {"use_kl_in_reward": False},
                            "actor_rollout_ref": {"rollout": {"log_prob_micro_batch_size_per_gpu": pad_to},
                                                  "actor": {"use_kl_loss": False,
                                                            "ppo_micro_batch_size_per_gpu": pad_to}}})
    n_real = len(batch)
    state = np.random.get_state()
    np.random.seed(seed)                        # adjust_batch draws its copies from np.random
    out = adjust_batch(cfg, batch)
    np.random.set_state(state)
    assert PADDING_ROW_KEY in out.batch.keys(), "the test batch needs padding copies"
    out.batch["response_mask"] = out.batch["attention_mask"][:, -RLEN:]
    if weights is not None:
        attach_task_loss_weights(out, n_real=n_real, mini_batch_size=MINI, metrics={}, **weights)
    out.reorder(torch.as_tensor(np.random.default_rng(seed + 1000).permutation(len(out))))
    return out


def rollout_batch(seed, weights=None, tasks=TASKS):
    return balanced(turn_ordered_batch(seed, tasks)[0], seed, weights=weights)


def episode_reward_tensor(batch):
    """EpisodeRewardManager's tensor: the episode reward on the row's last response token."""
    out = torch.zeros(len(batch), RLEN)
    last = batch.batch["response_mask"].sum(-1) - 1
    for i in range(len(batch)):
        out[i, last[i]] = float(batch.non_tensor_batch["episode_rewards"][i])
    return out


def penalised_scores(batch):
    """token_level_rewards as the trainer makes them: the episode reward, then apply_invalid_action_penalty
    in place (the scores and token_level_rewards one tensor)."""
    batch.batch["token_level_scores"] = episode_reward_tensor(batch)
    apply_invalid_action_penalty(batch, invalid_action_penalty_coef=0.1, invalid_action_penalty_coef_by_task=PENALTY)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    return batch


def penalty_coefs(batch):
    """Per row, the trainer's invalid-action coefficient (what the hook hands the GiGPO shadow)."""
    names = batch.non_tensor_batch["task_name"]
    return np.array([_get_invalid_action_penalty_coef(SimpleNamespace(non_tensor_batch={"task_name": names[i]}),
                                                      0.1, PENALTY) for i in range(len(batch))])


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #

def trainer_config(ckpt_dir, estimator="progress_value_gae", lam=1.0, records=True, pv_block=None,
                   pg_loss_norm="token", **alg):
    records_cfg = {"enable": True} if records is True else ({"enable": False} if records is False else records)
    progress_value = {"enable": True, "records": records_cfg, **(pv_block or {})}
    algorithm = {"adv_estimator": estimator, "gamma": 1.0, "lam": lam, "norm_adv_by_std_in_grpo": True,
                 "use_kl_in_reward": False, "use_pf_ppo": False,
                 "pf_ppo": {"reweight_method": "pow", "weight_pow": 2.0},
                 # the defaults of ppo_trainer.yaml -- NOT the OPD+GiGPO run's (the shadow must not read them)
                 "gigpo": {"step_advantage_w": 1.0, "mode": "mean_norm", "enable_similarity": False,
                           "similarity_thresh": 0.95},
                 "compute_mean_std_cross_steps": True,
                 "progress_rank": {"enable": False, "alfworld_k": "milestone_arrive"},
                 "progress_value": progress_value}
    algorithm.update(alg)
    return OmegaConf.create({
        "algorithm": algorithm,
        "actor_rollout_ref": {
            "actor": {"loss_agg_mode": "token-mean", "use_invalid_action_penalty": True,
                      "invalid_action_penalty_coef": 0.1, "invalid_action_penalty_coef_by_task": dict(PENALTY),
                      "pg_loss_norm": pg_loss_norm},
            "rollout": {"temperature": 1.0, "n": 1, "multi_turn": {"enable": False}}},
        "env": {"env_name": "multitask", "max_steps": 50,
                "multitask": {"tasks": list(TASKS), "max_steps": dict(CAPS)}},
        "trainer": {"default_local_dir": str(ckpt_dir), "resume_mode": "auto", "resume_from_path": None},
    })


def make_trainer(cfg, step=1):
    t = object.__new__(OPDGRPORayTrainer)
    t.config = cfg
    t.global_steps = step
    t.reward_fn = None
    t.actor_rollout_wg = None
    t.traj_collector = SimpleNamespace(take_prefetched_log_probs=lambda: None)
    return t


@pytest.fixture
def stub_workers(monkeypatch):
    """The reward manager and the old-log-prob forward, which need workers."""
    monkeypatch.setattr(grpo_mod, "compute_reward", lambda batch, fn: (episode_reward_tensor(batch), {}))

    def _log_probs(wg, batch, prefetched, temperature):
        z = torch.zeros(len(batch), RLEN)
        return DataProto.from_dict(tensors={"old_log_probs": z, "entropys": z.clone()})

    monkeypatch.setattr(grpo_mod, "compute_log_prob_with_prefetch", _log_probs)


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    random.seed(s)


def rng_states():
    np_state = np.random.get_state()
    return (np_state[0], np_state[1].copy(), np_state[2], np_state[3], np_state[4]), \
        torch.get_rng_state().clone(), random.getstate()


def assert_same_rng(a, b):
    assert a[0][0] == b[0][0] and np.array_equal(a[0][1], b[0][1]) and a[0][2:] == b[0][2:]
    assert torch.equal(a[1], b[1])
    assert a[2] == b[2]


def _same_value(x, y):
    if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
        return True
    return bool(np.all(x == y)) if not isinstance(x, (list, dict)) else x == y


def snapshot(batch):
    """Every tensor (cloned), every non-tensor column (copied) and the key sets, in row order."""
    return ({k: v.clone() for k, v in batch.batch.items()},
            {k: copy.deepcopy(v) for k, v in batch.non_tensor_batch.items()},
            copy.deepcopy(dict(batch.meta_info)))


def assert_same_batch(snap, batch):
    tensors, non_tensors, meta = snap
    assert set(tensors) == set(batch.batch.keys())
    for k, v in tensors.items():
        assert v.dtype == batch.batch[k].dtype and torch.equal(v, batch.batch[k]), k
    assert set(non_tensors) == set(batch.non_tensor_batch)
    for k, v in non_tensors.items():
        w = batch.non_tensor_batch[k]
        assert v.dtype == w.dtype and v.shape == w.shape, k
        assert all(_same_value(a, b) for a, b in zip(v.tolist(), w.tolist())), k
    assert meta == dict(batch.meta_info)


# --------------------------------------------------------------------------- #
# Observation only
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("estimator,lam", [("progress_value_gae", 1.0), ("progress_value_gae", 0.9), ("grpo", 1.0)])
def test_records_on_or_off_train_bit_identically(tmp_path, stub_workers, monkeypatch, estimator, lam):
    on = make_trainer(trainer_config(tmp_path / "on", estimator, lam, records=True))
    off = make_trainer(trainer_config(tmp_path / "off", estimator, lam, records=False))

    # ...and the records call ITSELF: batch, RNGs and the training table unchanged across it; the table
    # neither staged nor committed by it.
    real_call = OPDGRPORayTrainer._rollout_records
    calls = []

    def spy(self, batch, metrics, timing_raw, pv_kwargs):
        before = snapshot(batch), rng_states()
        table = getattr(self, "_progress_value", None)
        tstate = None if table is None else (copy.deepcopy(table.state_dict()), table.updates, table.pending)
        real_call(self, batch, metrics, timing_raw, pv_kwargs)
        assert_same_batch(before[0], batch)
        assert_same_rng(before[1], rng_states())
        if table is not None:
            assert (table.state_dict(), table.updates, table.pending) == tstate
        calls.append(bool(rr.RecordsConfig.from_config(self.config).enable))

    monkeypatch.setattr(OPDGRPORayTrainer, "_rollout_records", spy)
    for step, seed in enumerate((11, 12, 13), start=1):
        on.global_steps = off.global_steps = step
        outs = {}
        for name, t in (("on", on), ("off", off)):
            batch = rollout_batch(seed, weights={"pg_loss_norm": "token"})
            seed_all(1234 + step)
            metrics = {}
            batch, _ = t._reward_and_advantage(batch, metrics, timing_raw={})
            outs[name] = (batch, metrics, rng_states())
        (b_on, m_on, r_on), (b_off, m_off, r_off) = outs["on"], outs["off"]
        # The same batch -- advantages included, bit for bit -- the same row order, the same RNG states.
        assert torch.equal(b_on.batch["advantages"], b_off.batch["advantages"])
        assert_same_batch(snapshot(b_off), b_on)
        assert list(b_on.non_tensor_batch["traj_uid"]) == list(b_off.non_tensor_batch["traj_uid"])
        assert_same_rng(r_on, r_off)
        # The training table (value arm): the same state and update count; none in a grpo arm.
        if estimator == "progress_value_gae":
            assert on._progress_value.state_dict() == off._progress_value.state_dict()
            assert on._progress_value.updates == off._progress_value.updates == step
        else:
            assert getattr(on, "_progress_value", None) is None and getattr(off, "_progress_value", None) is None
        # Every metric of the run without records is there, unchanged; the records add only their own.
        assert all(k in m_on and _same_value(m_on[k], v) for k, v in m_off.items())
        assert all(k.startswith(("records/", "traj/")) for k in set(m_on) - set(m_off))
        assert m_on["records/failed"] == 0.0 and m_on["records/write_failed"] == 0.0
        # Files only when on.
        assert (tmp_path / "on" / "progress_value_groups" / f"step{step}.jsonl").exists()
        assert not (tmp_path / "off" / "progress_value_groups").exists()
    assert calls == [True, False] * 3


def test_the_value_arm_records_never_touch_the_training_table(tmp_path, stub_workers, monkeypatch):
    """In the progress_value_gae arm the records read the result compute_advantage computed: the table is
    staged once and committed once per step, both by the training path, and no second table is built."""
    counts = {"stage": 0, "commit": 0, "discard": 0}
    for name in counts:
        real = getattr(pv.ProgressValueTable, name)

        def wrapped(self, *a, _real=real, _name=name, **k):
            counts[_name] += 1
            return _real(self, *a, **k)

        monkeypatch.setattr(pv.ProgressValueTable, name, wrapped)
    t = make_trainer(trainer_config(tmp_path))
    for step, seed in enumerate((21, 22), start=1):
        t.global_steps = step
        metrics = {}
        t._reward_and_advantage(rollout_batch(seed), metrics, timing_raw={})
        assert metrics["records/failed"] == 0.0
        assert counts == {"stage": step, "commit": step, "discard": 0}
    assert getattr(t, "_rollout_records_value", None) is None
    assert not (tmp_path / "progress_value_groups" / "shadow_value_state").exists()


def test_a_failed_record_costs_the_records_not_the_step(tmp_path, stub_workers, monkeypatch, capsys):
    t = make_trainer(trainer_config(tmp_path))
    ref = make_trainer(trainer_config(tmp_path / "ref", records=False))

    def boom(*a, **k):
        raise RuntimeError("a bug in the records")

    monkeypatch.setattr(rr, "compute_step_records", boom)
    metrics = {}
    batch, _ = t._reward_and_advantage(rollout_batch(31), metrics, timing_raw={})
    want, _ = ref._reward_and_advantage(rollout_batch(31), {}, timing_raw={})
    assert torch.equal(batch.batch["advantages"], want.batch["advantages"])
    assert metrics["records/failed"] == 1.0
    assert t._progress_value.updates == 1                 # the step itself went through
    assert "the analysis records failed" in capsys.readouterr().out
    assert not (tmp_path / "progress_value_groups" / "step1.jsonl").exists()


# --------------------------------------------------------------------------- #
# The shadow GRPO
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cross_steps,by_std,oci", [(True, True, False), (False, True, False), (True, False, False),
                                                    (True, True, True)])
def test_the_shadow_grpo_is_compute_advantage_grpo(cross_steps, by_std, oci):
    batch = penalised_scores(rollout_batch(41))
    other = penalised_scores(rollout_batch(41))
    kwargs = {}
    if oci:
        # Two rows kept out of the statistic and one group floored, as the OCI arms mark them.
        exclude = torch.zeros(len(batch), dtype=torch.bool)
        exclude[[0, 5]] = True
        floor = torch.as_tensor(batch.non_tensor_batch["uid"] == batch.non_tensor_batch["uid"][3])
        for b in (batch, other):
            b.batch[GRPO_STAT_EXCLUDE_KEY], b.batch[OCI_FLOOR_KEY] = exclude.clone(), floor.clone()
        kwargs = {"exclude_mask": exclude, "floor_mask": floor, "floor_value": 0.0}
    before = snapshot(batch)
    got = rr.shadow_grpo(batch.batch["token_level_rewards"], batch.non_tensor_batch["uid"],
                         batch.non_tensor_batch["traj_uid"], padding_mask=batch.batch[PADDING_ROW_KEY],
                         norm_adv_by_std_in_grpo=by_std, compute_mean_std_cross_steps=cross_steps, **kwargs)
    assert_same_batch(before, batch)                      # read, never written
    out = compute_advantage(other, adv_estimator="grpo", norm_adv_by_std_in_grpo=by_std, multi_turn=False,
                            compute_mean_std_cross_steps=cross_steps,
                            oci_floor_value=0.0)
    mask = other.batch["response_mask"].to(torch.float32)
    want = out.batch["advantages"]
    assert got.dtype == np.float32
    assert torch.equal(torch.as_tensor(got).unsqueeze(-1) * mask, want)
    # Every row's first response token is on the mask here, so its column is the scalar itself.
    assert bool(mask[:, 0].all()) and torch.equal(torch.as_tensor(got), want[:, 0])


def test_in_a_grpo_arm_the_shadow_grpo_is_the_advantage(tmp_path, stub_workers):
    t = make_trainer(trainer_config(tmp_path, "grpo"))
    metrics = {}
    t._reward_and_advantage(rollout_batch(42, weights={"pg_loss_norm": "token"}), metrics, timing_raw={})
    assert metrics["records/shadow_grpo/max_abs_diff_adv"] == 0.0
    for task in TASKS:
        for kind in rr.KINDS:
            p = f"records/{task}/{kind}/"
            for unit in UNITS:
                assert metrics[p + "shadow_grpo/" + unit] == metrics[p + "adv/" + unit], (p, unit)


# --------------------------------------------------------------------------- #
# The shadow GiGPO against the OPD+GiGPO run's own implementation
# --------------------------------------------------------------------------- #

def gigpo_reference():
    """opd-gigpo's gigpo/core_gigpo.py at b2b393c, from the repository's object store (read-only)."""
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{GIGPO_REF}:gigpo/core_gigpo.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"the OPD+GiGPO reference ({GIGPO_REF}) is not in this repository: {e}")
    mod = types.ModuleType(f"core_gigpo_{GIGPO_REF}")
    exec(compile(src, f"{GIGPO_REF}:gigpo/core_gigpo.py", "exec"), mod.__dict__)
    return mod


def test_the_reference_is_the_run_configuration():
    """The constants are the OPD+GiGPO launcher's (b2b393c), and its lock pins released statistics."""
    try:
        launcher = subprocess.run(["git", "-C", REPO, "show",
                                   f"{GIGPO_REF}:examples/opd_grpo_trainer/run_multitask_opd_gigpo_qwen3.sh"],
                                  capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"the OPD+GiGPO launcher ({GIGPO_REF}) is not in this repository: {e}")
    assert f"algorithm.gamma={rr.GIGPO_GAMMA}" in launcher
    assert f"algorithm.gigpo.step_advantage_w={rr.GIGPO_STEP_ADVANTAGE_W}" in launcher
    assert f"algorithm.gigpo.mode={rr.GIGPO_MODE}" in launcher
    assert "algorithm.gigpo.enable_similarity=False" in launcher
    assert "algorithm.gigpo.exact_statistics=False" in launcher


def reference_gigpo(ref, batch, step_rewards):
    """What the OPD+GiGPO run's compute_advantage gives ``batch`` (its penalised token_level_rewards) with
    the given UNPENALISED step returns: apply_invalid_action_penalty's step_rewards rule, then
    compute_gigpo_outcome_advantage as released."""
    view = DataProto.from_dict(
        tensors={"token_level_scores": torch.zeros(len(batch), RLEN), "step_rewards": step_rewards.clone(),
                 "prompts": batch.batch["prompts"], "attention_mask": batch.batch["attention_mask"]},
        non_tensors={"is_action_valid": batch.non_tensor_batch["is_action_valid"],
                     "task_name": batch.non_tensor_batch["task_name"]})
    apply_invalid_action_penalty(view, invalid_action_penalty_coef=0.1, invalid_action_penalty_coef_by_task=PENALTY)
    adv, _ = ref.compute_gigpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"], step_rewards=view.batch["step_rewards"],
        response_mask=batch.batch["response_mask"], anchor_obs=batch.non_tensor_batch["anchor_obs"],
        index=batch.non_tensor_batch["uid"], traj_index=batch.non_tensor_batch["traj_uid"],
        step_advantage_w=1.0, mode="mean_std_norm", enable_similarity=False,
        padding_mask=batch.batch.get(PADDING_ROW_KEY, None), exact_statistics=False)
    return adv, view.batch["step_rewards"]


def shadow_on(batch):
    real = (np.ones(len(batch), dtype=bool) if PADDING_ROW_KEY not in batch.batch.keys()
            else ~batch.batch[PADDING_ROW_KEY].numpy())
    g = rr.gigpo_step_returns(batch.non_tensor_batch["rewards"], batch.non_tensor_batch["traj_uid"],
                              batch.non_tensor_batch["pv_t"], real)
    gp = rr.penalised_step_returns(g, batch.non_tensor_batch["is_action_valid"], penalty_coefs(batch))
    out = rr.shadow_gigpo(batch.batch["token_level_rewards"], gp, batch.non_tensor_batch["anchor_obs"],
                          batch.non_tensor_batch["uid"], batch.non_tensor_batch["traj_uid"])
    return out, g, gp


def test_the_shadow_gigpo_is_the_reference_on_a_turn_ordered_batch():
    ref = gigpo_reference()
    batch = turn_ordered_batch(51)[0]
    batch.batch["response_mask"] = batch.batch["attention_mask"][:, -RLEN:]
    batch = penalised_scores(batch)
    # The run's step returns: compute_step_discounted_returns on the rollout's own (turn) order.
    raw = ref.compute_step_discounted_returns(batch, gamma=0.95)
    want, want_g = reference_gigpo(ref, batch, raw)
    before = snapshot(batch)
    got, g, gp = shadow_on(batch)
    assert_same_batch(before, batch)
    assert torch.equal(g, raw) and torch.equal(gp, want_g)
    mask = batch.batch["response_mask"].to(torch.float32)
    assert torch.equal(torch.as_tensor(got["total"]).unsqueeze(-1) * mask, want)
    # The step term acts: some step group has more than one row, and its returns differ.
    assert (got["size"] > 1).any() and np.abs(got["step"]).max() > 0
    # One uid group's anchors only: "start" opens every group, and each is its own step group.
    starts = {k for k in got["key"] if k[1] == "start"}
    assert len(starts) == len(set(batch.non_tensor_batch["uid"]))


def test_the_shadow_gigpo_is_the_reference_on_the_balanced_padded_batch():
    """What the trainer hands the records: rows balanced (not in turn order) and padded. The reference's
    step returns were computed in turn order BEFORE adjust_batch and travelled with the rows, as in the
    run; the records rebuild them from (traj_uid, pv_t)."""
    ref = gigpo_reference()
    rollout, _ = turn_ordered_batch(52)
    raw = ref.compute_step_discounted_returns(rollout, gamma=0.95)
    by_key = {(str(u), int(t)): float(x) for u, t, x in
              zip(rollout.non_tensor_batch["traj_uid"], rollout.non_tensor_batch["turn_step"], raw.tolist())}
    batch = penalised_scores(balanced(rollout, 52))
    step_rewards = torch.tensor([by_key[(str(u), int(t))] for u, t in zip(batch.non_tensor_batch["traj_uid"],
                                                                          batch.non_tensor_batch["turn_step"])],
                                dtype=torch.float32)
    want, want_g = reference_gigpo(ref, batch, step_rewards)
    got, g, gp = shadow_on(batch)
    assert torch.equal(g, step_rewards) and torch.equal(gp, want_g)
    mask = batch.batch["response_mask"].to(torch.float32)
    assert torch.equal(torch.as_tensor(got["total"]).unsqueeze(-1) * mask, want)
    # Rebuilding the returns in batch order instead would be wrong here (the reference's own function
    # on the balanced rows): the turn order is what makes them right.
    wrong = ref.compute_step_discounted_returns(batch, gamma=0.95)
    assert not torch.equal(wrong, step_rewards)


def test_the_hook_records_the_reference_gigpo(tmp_path, stub_workers, monkeypatch):
    """End to end: the shadow the hook computes on its batch -- the trainer's penalty coefficients, the
    token_level_rewards it built -- is the reference's, and so is what reaches the file (4 digits)."""
    ref = gigpo_reference()
    rollout, _ = turn_ordered_batch(53)
    raw = ref.compute_step_discounted_returns(rollout, gamma=0.95)
    by_key = {(str(u), int(t)): float(x) for u, t, x in
              zip(rollout.non_tensor_batch["traj_uid"], rollout.non_tensor_batch["turn_step"], raw.tolist())}
    seen = {}
    real_shadow = rr.shadow_gigpo

    def keep(*a, **k):
        seen["out"] = real_shadow(*a, **k)
        return seen["out"]

    monkeypatch.setattr(rr, "shadow_gigpo", keep)
    t = make_trainer(trainer_config(tmp_path, "grpo"))
    metrics = {}
    batch, _ = t._reward_and_advantage(balanced(rollout, 53), metrics, timing_raw={})
    assert metrics["records/shadow_gigpo/available"] == 1.0
    step_rewards = torch.tensor([by_key[(str(u), int(s))] for u, s in zip(batch.non_tensor_batch["traj_uid"],
                                                                          batch.non_tensor_batch["turn_step"])],
                                dtype=torch.float32)
    want, _ = reference_gigpo(ref, batch, step_rewards)
    mask = batch.batch["response_mask"].to(torch.float32)
    assert torch.equal(torch.as_tensor(seen["out"]["total"]).unsqueeze(-1) * mask, want)
    recs = [json.loads(line) for line in open(tmp_path / "progress_value_groups" / "step1.jsonl")]
    row = {(str(u), int(s)): i for i, (u, s) in enumerate(zip(batch.non_tensor_batch["traj_uid"],
                                                              batch.non_tensor_batch["pv_t"]))
           if not bool(batch.batch[PADDING_ROW_KEY][i])}
    for rec in recs:
        for tr in rec["traj"]:
            for t_, g_ in zip(tr["turn"]["t"], tr["turn"]["gigpo"]):
                assert g_ == rr._num(float(want[row[(tr["traj_uid"], t_)], 0]))


def test_no_anchors_or_rewards_no_gigpo(tmp_path, stub_workers, capsys):
    for drop in ("anchor_obs", "rewards"):
        t = make_trainer(trainer_config(tmp_path / drop, "grpo"))
        batch = rollout_batch(54)
        batch.non_tensor_batch.pop(drop)
        metrics = {}
        t._reward_and_advantage(batch, metrics, timing_raw={})
        assert metrics["records/failed"] == 0.0 and metrics["records/shadow_gigpo/available"] == 0.0
        assert not any("/shadow_gigpo/" in k for k in metrics if k.startswith("records/") and
                       not k.startswith("records/shadow_gigpo/"))
        recs = [json.loads(line) for line in open(tmp_path / drop / "progress_value_groups" / "step1.jsonl")]
        assert all("gigpo" not in tr["turn"] for rec in recs for tr in rec["traj"])
        assert f"shadow_gigpo: the batch has no ['{drop}']" in capsys.readouterr().out
    # An anchor GiGPO cannot group (None) is not approximated either.
    t = make_trainer(trainer_config(tmp_path / "none", "grpo"))
    batch = rollout_batch(55)
    anchors = np.array(batch.non_tensor_batch["anchor_obs"], dtype=object)
    anchors[3] = None
    batch.non_tensor_batch["anchor_obs"] = anchors
    metrics = {}
    t._reward_and_advantage(batch, metrics, timing_raw={})
    assert metrics["records/shadow_gigpo/available"] == 0.0 and metrics["records/failed"] == 0.0
    assert "shadow_gigpo: not emitted" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# progress_rank's rollout metrics without its controller
# --------------------------------------------------------------------------- #

def test_the_traj_metrics_are_progress_ranks(tmp_path, stub_workers, monkeypatch):
    """On one batch: the traj/* and think-block metrics progress_rank reports (its controller on, rho 0,
    in a grpo arm) and the records' (progress_rank off) are the same keys and the same numbers."""
    pr_cfg = {"enable": True, "rho": 0.0, "alfworld_k": "milestone_arrive", "record_groups": False}
    with_pr = make_trainer(trainer_config(tmp_path / "pr", "grpo", records=False, progress_rank=pr_cfg))
    with_rec = make_trainer(trainer_config(tmp_path / "rec", "grpo", records=True))
    m_pr, m_rec = {}, {}
    with_pr._reward_and_advantage(rollout_batch(61), m_pr, timing_raw={})
    with_rec._reward_and_advantage(rollout_batch(61), m_rec, timing_raw={})
    traj_pr = {k: v for k, v in m_pr.items() if k.startswith("traj/")}
    traj_rec = {k: v for k, v in m_rec.items() if k.startswith("traj/")}
    assert traj_pr and traj_pr == traj_rec
    for task in TASKS:
        assert f"traj/{task}/think_block_share" in traj_rec and f"traj/{task}/groups_stuck_share" in traj_rec
    assert "traj/alfworld/fail_progress" in traj_rec and "traj/webshop/fail_task_score" in traj_rec

    # With progress_rank on, the records leave those keys to it: one writer per key.
    def refuse(*a, **k):
        raise AssertionError("the records reported traj/* beside progress_rank")

    monkeypatch.setattr(rr, "trajectory_and_think_metrics", refuse)
    both = make_trainer(trainer_config(tmp_path / "both", "grpo", records=True, progress_rank=pr_cfg))
    m_both = {}
    both._reward_and_advantage(rollout_batch(61), m_both, timing_raw={})
    assert m_both["records/failed"] == 0.0
    assert {k: v for k, v in m_both.items() if k.startswith("traj/")} == traj_pr


def test_the_webshop_failure_progress_names_its_count(tmp_path, stub_workers):
    """traj/webshop/fail_progress is k / K of WebShop's failures with k = progress_k, and progress_k's
    WebShop count is webshop_k's: the legacy count on every arm before 2026-09-30, the session count in
    the 2x2's cells. Under session the key says so -- traj/webshop/fail_progress_session -- in the
    records and on the controller's path alike, and nothing else moves: the same number from the same
    column, every other traj/* key the same."""
    key = "traj/webshop/fail_progress"
    counters = {"alfworld_k": "milestone_arrive", "webshop_k": "session"}
    legacy = make_trainer(trainer_config(tmp_path / "legacy", "grpo"))
    session = make_trainer(trainer_config(tmp_path / "session", "grpo", progress_rank={"enable": False, **counters}))
    m_leg, m_ses = {}, {}
    legacy._reward_and_advantage(rollout_batch(63), m_leg, timing_raw={})
    session._reward_and_advantage(rollout_batch(63), m_ses, timing_raw={})
    assert m_leg["records/failed"] == 0.0 and m_ses["records/failed"] == 0.0
    t_leg = {k: v for k, v in m_leg.items() if k.startswith("traj/")}
    t_ses = {k: v for k, v in m_ses.items() if k.startswith("traj/")}
    assert key in t_leg and key + "_session" not in t_leg
    assert key not in t_ses and t_ses[key + "_session"] == t_leg[key]
    assert {k: v for k, v in t_ses.items() if k != key + "_session"} == {k: v for k, v in t_leg.items() if k != key}
    assert "traj/alfworld/fail_progress" in t_ses and "traj/search/fail_progress" in t_ses

    # The controller's path (progress_rank on, rho 0) names it the same way: one name per count,
    # whichever of the two writers reports it.
    pr_cfg = {"enable": True, "rho": 0.0, "record_groups": False, **counters}
    with_pr = make_trainer(trainer_config(tmp_path / "pr", "grpo", records=False, progress_rank=pr_cfg))
    m_pr = {}
    with_pr._reward_and_advantage(rollout_batch(63), m_pr, timing_raw={})
    assert {k: v for k, v in m_pr.items() if k.startswith("traj/")} == t_ses


def test_trajectory_summaries_are_the_controllers():
    """The per-trajectory summary field for field against ProgressRankController.apply's own (read off
    the group records it leaves), on a batch with every optional column."""
    from verl.trainer.ppo.progress_rank import ProgressRankController

    batch = penalised_scores(rollout_batch(62))
    nt = batch.non_tensor_batch
    real = ~batch.batch[PADDING_ROW_KEY].numpy()
    mask = batch.batch["attention_mask"][:, -RLEN:]
    names = np.asarray(nt["task_name"], dtype=object)
    ctl = ProgressRankController(rho=0.0)
    adv = torch.zeros(len(batch), RLEN)
    ctl.apply(advantages=adv, mask=mask, uids=nt["uid"], tuids=nt["traj_uid"], task_names=names,
              episode_rewards=nt["episode_rewards"], k_rows=nt["progress_k"], total_rows=nt["progress_total"],
              real_rows=real, stat_rows=real, valid_rows=nt["is_action_valid"], coverage_rows=nt["coverage_d"],
              episode_lengths=nt["episode_lengths"], task_score_rows=nt["task_score"],
              committed_rows=nt["committed"], revisit_rows=nt["revisits"],
              done_walkset_rows=nt["progress_done_walkset"])
    mine = rr.trajectory_summaries(real=real, tuids=nt["traj_uid"], task_names=names,
                                   episode_rewards=nt["episode_rewards"],
                                   tokens=mask.to(torch.float64).sum(-1).numpy(), valid_rows=nt["is_action_valid"],
                                   coverage_rows=nt["coverage_d"], episode_lengths=nt["episode_lengths"],
                                   task_score_rows=nt["task_score"], committed_rows=nt["committed"],
                                   revisit_rows=nt["revisits"], done_walkset_rows=nt["progress_done_walkset"])
    # The controller's group records list its per-trajectory summary column by column.
    fields = {"turns": "turns", "tokens": "tokens", "reward": "reward", "won": "won", "invalid": "invalid_turns",
              "d": "coverage_d", "length": "length", "task_score": "task_score", "committed": "committed",
              "revisits": "revisits", "done_walkset": "done_walkset"}
    seen = set()
    for rec in ctl.last_group_records:
        for j, tu in enumerate(rec["trajs"]):
            seen.add(tu)
            for mine_key, their_key in fields.items():
                assert mine[tu][mine_key] == rec[their_key][j], (tu, mine_key)
    assert seen == set(mine) and len(seen) == len(trajectories(62))
    # ...and the same metrics through progress_rank's pure function.
    from verl.trainer.ppo.progress_rank import failed_groups, trajectory_metrics, trajectory_progress

    groups = failed_groups(nt["uid"], nt["traj_uid"], np.asarray([str(x) for x in names]), nt["episode_rewards"],
                           [i for i in range(len(batch)) if real[i]])
    prog = trajectory_progress(nt["traj_uid"], nt["progress_k"], nt["progress_total"], range(len(batch)))
    got = rr.trajectory_and_think_metrics(batch, task_names=names, real=real, mask=mask, turn_caps=CAPS)
    want = trajectory_metrics(groups, mine, tuids=nt["traj_uid"], real=real, tasks=list(TASKS), traj_prog=prog,
                              turn_caps=CAPS, have_invalid=True)
    assert {k: v for k, v in got.items() if "think_block" not in k} == want


# --------------------------------------------------------------------------- #
# The records-only shadow value table
# --------------------------------------------------------------------------- #

def value_config():
    return pv.ProgressValueConfig.from_config({"enable": True}, gamma=1.0, lam=1.0, horizons=CAPS,
                                              k_definitions={"alfworld": "milestone_arrive", "search": "evidence",
                                                             "webshop": "session_v1"})


def columns(batch):
    cols = dict(batch.non_tensor_batch)
    cols[PADDING_ROW_KEY] = batch.batch[PADDING_ROW_KEY].numpy()
    return cols


def test_the_shadow_value_is_its_own_table_and_follows_the_steps(tmp_path, stub_workers, monkeypatch):
    t = make_trainer(trainer_config(tmp_path, "grpo"))
    reference = pv.ProgressValueTable(value_config())
    captured = {}
    real_compute = rr.compute_step_records

    def keep(batch, **k):
        captured["value"] = k["value"]
        captured["source"] = k["value_source"]
        return real_compute(batch, **k)

    monkeypatch.setattr(rr, "compute_step_records", keep)
    for step, seed in enumerate((71, 72, 73), start=1):
        t.global_steps = step
        batch = rollout_batch(seed)
        expected = pv.compute_progress_value_advantage(columns(batch), reference)
        metrics = {}
        t._reward_and_advantage(batch, metrics, timing_raw={})
        assert captured["source"] == "shadow"
        assert np.array_equal(captured["value"].advantage, expected.advantage)
        reference.update(expected.records)
        table = t._rollout_records_value
        assert table.state_dict() == reference.state_dict() and table.updates == step and table.pending is None
        assert metrics["records/shadow_value/table_updates"] == float(step)
        assert metrics["records/shadow_value/alfworld/fallback_share"] == (1.0 if step == 1 else 0.0)
        assert not any(k.startswith("progress_value/") for k in metrics)
        saved = tmp_path / "progress_value_groups" / "shadow_value_state" / f"step{step}.json"
        assert json.loads(saved.read_text()) == reference.state_dict()
        # the shadow's advantage in the metrics, under its own name
        assert "records/alfworld/mixed/shadow_value/row_mean" in metrics
    # NEVER the arm's value table, so never in the checkpoint.
    assert getattr(t, "_progress_value", None) is None
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", lambda self: None)
    t.global_steps, t._pre_peek_dataloader_state = 3, None
    t._save_checkpoint()
    assert not (tmp_path / "global_step_3" / OPDRayTrainer.PROGRESS_VALUE_STATE_FILE).exists()

    # A run resumed at global_step_3 restores the shadow from the records' directory...
    r = make_trainer(trainer_config(tmp_path, "grpo"), step=4)
    batch = rollout_batch(74)
    expected = pv.compute_progress_value_advantage(columns(batch), copy.deepcopy(reference))
    r._reward_and_advantage(batch, {}, timing_raw={})
    assert np.array_equal(captured["value"].advantage, expected.advantage)
    assert r._rollout_records_value.updates == 4
    # ...and one that finds nothing starts empty, loudly, without refusing the step.
    lost = make_trainer(trainer_config(tmp_path / "elsewhere", "grpo"), step=9)
    metrics = {}
    lost._reward_and_advantage(rollout_batch(75), metrics, timing_raw={})
    assert lost._rollout_records_value.updates == 1 and metrics["records/failed"] == 0.0
    assert metrics["records/shadow_value/alfworld/fallback_share"] == 1.0


def test_a_grad_probe_holds_the_shadow_table_still(tmp_path, stub_workers):
    cfg = trainer_config(tmp_path, "grpo")
    cfg.trainer.grad_probe = {"enable": True, "mode": "tau", "n_batches": 2,
                              "out_path": str(tmp_path / "probe.json")}
    t = make_trainer(cfg)
    t._rollout_records_value = pv.ProgressValueTable(value_config())
    for n, seed in enumerate((76, 77), start=1):
        t._grad_probe_state = {"batches": n - 1}         # the probe's own counter, one behind
        metrics = {}
        t._reward_and_advantage(rollout_batch(seed), metrics, timing_raw={})
        assert t._rollout_records_value.updates == 0 and t._rollout_records_value.pending is None
        assert (tmp_path / "probe.json.pv_groups" / f"b{n}.jsonl").exists()
    assert not (tmp_path / "progress_value_groups").exists()


def test_shadow_value_off_records_no_value_columns(tmp_path, stub_workers):
    t = make_trainer(trainer_config(tmp_path, "grpo", records={"enable": True, "shadow_value": False}))
    metrics = {}
    t._reward_and_advantage(rollout_batch(78), metrics, timing_raw={})
    assert getattr(t, "_rollout_records_value", None) is None
    assert not any("shadow_value" in k for k in metrics)
    recs = [json.loads(line) for line in open(tmp_path / "progress_value_groups" / "step1.jsonl")]
    assert all(rec["value"] is None and "v" not in tr["turn"] for rec in recs for tr in rec["traj"])


# --------------------------------------------------------------------------- #
# What the record and the metrics hold
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pg_loss_norm", ["token", "trajectory"])
def test_the_record_and_the_metrics_hold_the_batch(tmp_path, stub_workers, monkeypatch, pg_loss_norm):
    weights = ({"pg_loss_norm": "token"} if pg_loss_norm == "token"
               else {"pg_loss_norm": "trajectory", "pg_ref_tokens": REF_TOKENS})
    t = make_trainer(trainer_config(tmp_path, pg_loss_norm=pg_loss_norm), step=5)
    captured = {}
    real_compute = rr.compute_step_records

    def keep(batch, **k):
        captured.update(k)
        return real_compute(batch, **k)

    monkeypatch.setattr(rr, "compute_step_records", keep)
    seed_batch = rollout_batch(81, weights=weights)
    assert (TASK_PG_LOSS_WEIGHT_KEY in seed_batch.batch.keys()) == (pg_loss_norm == "trajectory")
    metrics = {}
    batch, _ = t._reward_and_advantage(seed_batch, metrics, timing_raw={})
    res = captured["value"]
    nt = batch.non_tensor_batch
    real = ~batch.batch[PADDING_ROW_KEY].numpy()
    mask = batch.batch["attention_mask"][:, -RLEN:].to(torch.float64)
    tokens = mask.sum(-1).numpy()
    adv = batch.batch["advantages"].to(torch.float64)
    adv_row = ((adv * mask).sum(-1) / mask.sum(-1).clamp(min=1)).numpy()
    w_act = batch.batch[TASK_PG_LOSS_WEIGHT_KEY if pg_loss_norm == "trajectory" else TASK_LOSS_WEIGHT_KEY]
    w_act = w_act.to(torch.float64).numpy()

    # ---- the file: one line per group, every trajectory, every turn in order ----
    path = tmp_path / "progress_value_groups" / "step5.jsonl"
    assert not [p for p in path.parent.iterdir() if ".tmp." in p.name]
    recs = [json.loads(line) for line in open(path)]
    specs = {s["tuid"]: s for s in trajectories(81)}
    assert len(recs) == len({s["uid"] for s in specs.values()})
    row = {(str(u), int(s)): i for i, (u, s) in enumerate(zip(nt["traj_uid"], nt["pv_t"])) if real[i]}
    for rec in recs:
        # A fresh table: every group on the fallback, but the capped one (V = 0 by rule, not the fallback).
        assert rec["step"] == 5 and rec["value"] == "training" and rec["fallback"] is (rec["goal_capped"] != 1)
        kinds = {specs[tr["traj_uid"]]["kind"] for tr in rec["traj"]}
        assert kinds == {rec["kind"]}
        assert rec["type"] == (pv.alfworld_type(rec["gamefile"]) if rec["task"] == "alfworld" else rec["task"])
        if rec["task"] == "webshop":
            assert rec["goal_capped"] == int(rec["kind"] == "all_fail") and rec["goal_id"] in (100, 101, 102)
        else:
            assert rec["goal_capped"] is None and rec["goal_id"] is None
        for tr in rec["traj"]:
            s = specs[tr["traj_uid"]]
            rows = [row[(tr["traj_uid"], k)] for k in range(s["T"])]
            tt = tr["turn"]
            assert tr["turns"] == s["T"] and tt["t"] == list(range(s["T"])) and tr["won"] == int(s["won"])
            assert tr["tokens"] == int(tokens[rows].sum()) and tr["term"] == (1 if s["won"] else 3)
            assert tr["k_final"] == rr._num(nt["pv_k_after"][rows[-1]]) and tr["K"] == 4
            for key, arr in (("adv", adv_row), ("v", res.value), ("delta", res.delta), ("a_rl", res.a_rl),
                             ("a_fmt", res.a_fmt), ("tok", tokens),
                             ("w_tok", batch.batch[TASK_LOSS_WEIGHT_KEY].to(torch.float64).numpy())):
                assert tt[key] == [rr._num(arr[i]) for i in rows], key
            if pg_loss_norm == "trajectory":
                assert tt["w_pg"] == [rr._num(float(batch.batch[TASK_PG_LOSS_WEIGHT_KEY][i])) for i in rows]
            else:
                assert "w_pg" not in tt
            assert tt["valid"] == [int(bool(nt["is_action_valid"][i])) for i in rows]
            assert tt["rem"] == [CAPS[rec["task"]] - k for k in range(s["T"])]
            assert tt["k_b"] == [rr._num(nt["pv_k_before"][i]) for i in rows]
            for stem in rr.STATE_STEMS[rec["task"]]:
                assert tt[stem] == [rr._num(nt[f"pv_{stem}_b"][i]) for i in rows]
            assert "vadv" not in tt and len(tt["grpo"]) == len(tt["gigpo"]) == len(tt["anchor"]) == s["T"]

    # ---- the metrics: every unit named, and each number what the batch says ----
    for key in metrics:
        if key.startswith("records/") and key.count("/") == 4:
            assert key.rsplit("/", 1)[1] in UNITS, key
    for (task, kind) in [(tk, kd) for tk in TASKS for kd in rr.KINDS]:
        uids = {s["uid"] for s in specs.values() if s["task"] == task and s["kind"] == kind}
        trs = [s["tuid"] for s in specs.values() if s["uid"] in uids]
        rows = [row[(tu, k)] for tu in trs for k in range(specs[tu]["T"])]
        every = [i for i in range(len(batch)) if nt["uid"][i] in uids]
        p = f"records/{task}/{kind}/"
        assert metrics[p + "groups"] == len(uids) and metrics[p + "trajectories"] == len(trs)
        assert metrics[p + "rows"] == len(rows) and metrics[p + "tokens"] == float(tokens[rows].sum())
        for comp, x in (("adv", adv_row), ("a_rl", res.a_rl), ("a_fmt", res.a_fmt)):
            sums = [float(np.sum(x[[row[(tu, k)] for k in range(specs[tu]["T"])]])) for tu in trs]
            assert np.isclose(metrics[p + comp + "/row_mean"], np.mean(x[rows]))
            assert np.isclose(metrics[p + comp + "/row_abs_mean"], np.mean(np.abs(x[rows])))
            assert np.isclose(metrics[p + comp + "/traj_sum_mean"], np.mean(sums))
            assert np.isclose(metrics[p + comp + "/traj_sum_abs_mean"], np.mean(np.abs(sums)))
            assert np.isclose(metrics[p + comp + "/token_total"], np.sum(w_act[every] * tokens[every] * x[every]))
            assert np.isclose(metrics[p + comp + "/token_abs_total"],
                              np.sum(w_act[every] * tokens[every] * np.abs(x[every])))
    # The advantage's token totals over every task and kind are the policy-gradient loss's own sum
    # (at ratio 1): the weight the actor applies times the advantage on every loss token.
    total = sum(v for k, v in metrics.items() if k.endswith("/adv/token_total"))
    assert np.isclose(total, float((torch.as_tensor(w_act).unsqueeze(-1) * adv * mask).sum()))


def test_without_the_actual_weight_no_token_totals(tmp_path, stub_workers, capsys):
    """pg_loss_norm=trajectory on a batch that carries only the token weight (a driver that did not hand the
    actor settings to attach_task_loss_weights): the token totals would be read with a weight the actor
    does not apply, so they are left out, said once; everything else is reported."""
    t = make_trainer(trainer_config(tmp_path, pg_loss_norm="trajectory"))
    metrics = {}
    t._reward_and_advantage(rollout_batch(79, weights={"pg_loss_norm": "token"}), metrics, timing_raw={})
    assert metrics["records/failed"] == 0.0 and "records/alfworld/mixed/adv/row_mean" in metrics
    assert not any(k.endswith(("/token_total", "/token_abs_total")) for k in metrics)
    assert "token_total / token_abs_total not reported" in capsys.readouterr().out


def test_the_record_is_json_with_nulls_not_nan(tmp_path):
    rr.write_jsonl_atomic(str(tmp_path / "a" / "step1.jsonl"), [{"x": rr._num(float("nan")), "y": rr._num(1.23456)}],
                          {"batch": 2})
    assert json.loads((tmp_path / "a" / "step1.jsonl").read_text()) == {"batch": 2, "x": None, "y": 1.235}
    with pytest.raises(ValueError):
        rr.write_jsonl_atomic(str(tmp_path / "b.jsonl"), [{"x": float("nan")}])
    assert not (tmp_path / "b.jsonl").exists() and not list(tmp_path.glob("b.jsonl.tmp.*"))


# --------------------------------------------------------------------------- #
# Launch checks, the 2x2's launchers and locks
# --------------------------------------------------------------------------- #

def test_the_launch_refuses_what_the_records_cannot_serve(tmp_path):
    for est in ("progress_value_gae", "grpo"):
        cfg = trainer_config(tmp_path, est)
        cfg.algorithm.progress_value.enable = False
        with pytest.raises(AssertionError, match="records.enable needs algorithm.progress_value.enable"):
            check_progress_value_config(cfg)
    # A records-only shadow table the builder refuses (gamma < 1 with no gamma^t prefix) is refused at
    # launch, naming the switch that avoids it...
    with pytest.raises(AssertionError, match="records.shadow_value"):
        check_progress_value_config(trainer_config(tmp_path, "grpo", gamma=0.95))
    assert check_progress_value_config(trainer_config(tmp_path, "grpo", gamma=0.95,
                                                      records={"enable": True, "shadow_value": False})) is False
    # ...and off, the records check nothing.
    assert check_rollout_records_config(trainer_config(tmp_path, "grpo", records=False)) is False
    assert check_rollout_records_config(trainer_config(tmp_path, "grpo")) is True


CELLS = {
    ("grpo", "token"): "examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_qwen3.sh",
    ("grpo", "trajectory"): "examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_trajnorm_qwen3.sh",
    ("value", "token"): "examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh",
    ("value", "trajectory"): "examples/opd_grpo_trainer/run_multitask_progress_value_gae_trajnorm_qwen3.sh",
}
LOCKS = ("expected_multitask_grpo_v2recipe_config.yaml", "expected_multitask_grpo_v2recipe_trajnorm_config.yaml",
         "expected_multitask_progress_value_gae_lam1.0_config.yaml",
         "expected_multitask_progress_value_gae_lam0.9_config.yaml",
         "expected_multitask_progress_value_gae_trajnorm_lam1.0_config.yaml",
         "expected_multitask_progress_value_gae_trajnorm_lam0.9_config.yaml")
RECORDS_PINS = {"algorithm.progress_value.records.enable": True,
                "algorithm.progress_value.records.shadow_grpo": True,
                "algorithm.progress_value.records.shadow_gigpo": True,
                "algorithm.progress_value.records.shadow_value": True,
                "algorithm.progress_value.records.per_turn": True}


def _injected(monkeypatch, script, lam=None, extra=()):
    from tests.trainer.test_run_script_overrides_compose import _overrides
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    monkeypatch.setenv("RUN_TAG_SUFFIX", "")
    if lam is None:
        monkeypatch.delenv("LAM", raising=False)
    else:
        monkeypatch.setenv("LAM", lam)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="ppo_trainer", overrides=list(_overrides(script)) + list(extra))
    inject_opd_grpo_config(cfg)
    return cfg


@pytest.mark.parametrize("cell", sorted(CELLS), ids=lambda c: "-".join(c))
def test_every_cell_of_the_2x2_records(monkeypatch, cell):
    from verl.utils.expected_config import check_expected_config

    cfg = _injected(monkeypatch, CELLS[cell])
    assert check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config)) == []
    rcfg = rr.RecordsConfig.from_config(cfg)
    assert rcfg == rr.RecordsConfig(enable=True, dir=None, shadow_grpo=True, shadow_gigpo=True, shadow_value=True,
                                    per_turn=True)
    assert check_rollout_records_config(cfg) is True
    # The wrappers restate the switch (their parent spells out every records key).
    assert "algorithm.progress_value.records.enable=True" in open(os.path.join(REPO, CELLS[cell])).read()


@pytest.mark.parametrize("lock", LOCKS)
def test_every_lock_pins_the_records(lock):
    from verl.utils.expected_config import load_expectations

    want = load_expectations(os.path.join(REPO, "examples", "opd_grpo_trainer", lock))
    assert {k: want.get(k) for k in RECORDS_PINS} == RECORDS_PINS
    assert "algorithm.progress_value.records.dir" not in want        # a host path


@pytest.mark.parametrize("override", ["algorithm.progress_value.records.enable=False",
                                      "algorithm.progress_value.records.shadow_grpo=False",
                                      "algorithm.progress_value.records.shadow_gigpo=False",
                                      "algorithm.progress_value.records.shadow_value=False",
                                      "algorithm.progress_value.records.per_turn=False"])
@pytest.mark.parametrize("cell", [("value", "token"), ("grpo", "trajectory")], ids=lambda c: "-".join(c))
def test_the_locks_catch_the_records(monkeypatch, cell, override):
    from verl.utils.expected_config import check_expected_config

    cfg = _injected(monkeypatch, CELLS[cell], extra=[override])
    miss = check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config))
    assert [m[0] for m in miss] == [override.split("=")[0]], miss


def test_records_dir_is_a_host_knob(monkeypatch, tmp_path):
    from verl.utils.expected_config import check_expected_config

    cfg = _injected(monkeypatch, CELLS[("value", "token")], extra=[f"algorithm.progress_value.records.dir={tmp_path}"])
    assert check_expected_config(cfg, os.path.join(REPO, cfg.trainer.expected_config)) == []
    t = make_trainer(cfg)
    assert t._rollout_records_dir(rr.RecordsConfig.from_config(cfg)) == str(tmp_path)


def test_the_yaml_defaults_leave_every_arm_without_records():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="ppo_trainer")
    assert rr.RecordsConfig.from_config(cfg) == rr.RecordsConfig()
    assert not rr.RecordsConfig().enable
    assert check_rollout_records_config(cfg) is False


def test_the_helpers_are_new_names_on_the_subclass():
    """tests/trainer/test_opd_grpo_arm.py's rule for the records' helpers: none shadows the shared loop."""
    helpers = [n for n in vars(OPDGRPORayTrainer) if n.startswith("_rollout_records")]
    assert helpers and not any(hasattr(OPDRayTrainer, n) for n in helpers)
    overridden = {n for n in vars(OPDGRPORayTrainer) if not n.startswith("__") and hasattr(OPDRayTrainer, n)}
    assert overridden == {"_reward_and_advantage", "_data_metrics", "progress_desc"}
