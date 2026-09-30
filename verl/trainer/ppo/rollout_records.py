"""Analysis records of one training step: what every advantage did, per task and group kind, beside
what outcome GRPO and OPD+GiGPO would have done with the same rollouts. OBSERVATION ONLY: nothing here
changes training.

WHY. progress_value_gae replaced outcome GRPO, and everything progress_rank used to measure on the way
went with it -- its group records, the traj/<task>/* rollout metrics, the think-block share -- because
all of it ran inside progress_rank's controller, which runs only with (a) on. The questions the new
estimator raises need the per-turn quantities written down, on THE SAME rollouts the comparison
estimators would have seen, and with their units named: a sum over a trajectory, a mean over its
turns and a token-weighted total are three different numbers (sum A = T mean A, so "longer
trajectories get more" is a statement about sums), and only the last one is what the loss applies.

WHERE IT RUNS. OPDGRPORayTrainer._reward_and_advantage calls it LAST, once the advantages are what
the actor will read (after compute_advantage, the value table's commit and progress_rank if on),
under algorithm.progress_value.records.enable. It needs progress_value.enable (the pv_* columns and
the progress counters), and runs beside adv_estimator=progress_value_gae and beside grpo (records
only).

WHAT IT MUST NOT DO (tests/trainer/test_rollout_records.py holds it to each):
  * write to the batch -- no tensor, no non-tensor column, no row order. Everything here reads the
    batch and computes on copies; the shadow estimators get its tensors only as inputs they sum
    into new tensors (compute_grpo_outcome_advantage, GiGPO's two normalisations).
  * draw a random number, from numpy, torch or python (GiGPO's step groups get deterministic keys
    instead of its uuid4 ones, which never touched the RNGs either).
  * stage or commit the TRAINING value table. In the progress_value_gae arm the result
    compute_advantage already computed is read; the records-only shadow table (below) is an object
    of its own, kept in the records' directory and never in the checkpoint.
With records on or off the same batch gets bit-identical advantages, batch contents, row order,
table state and RNG states; only the files, the records/* metrics (with progress_rank off, the traj/*
ones too) and the step's "records" timer differ.

THE SHADOWS, per row, for this batch:
  shadow_grpo   what the control's advantage would be: compute_advantage's GRPO branch exactly --
                core_algos.compute_grpo_outcome_advantage with the trainer's own arguments (the
                statistic over turn rows under compute_mean_std_cross_steps, adjust_batch's copies
                out of it, the OCI exclude / floor columns when a run has them). Its input is
                token_level_rewards as the trainer left them, which ALREADY carry the
                invalid-action penalty (apply_invalid_action_penalty subtracts it from
                token_level_scores in place, and token_level_rewards is that tensor): applying it
                again would count it twice. In a grpo arm with nothing added on top it is the
                advantage itself (records/shadow_grpo/max_abs_diff_adv reads 0).
  shadow_gigpo  OPD+GiGPO's estimator as that run had it (.claude/worktrees/opd-gigpo at b2b393c:
                gigpo/core_gigpo.py as released, exact_statistics off; its launcher's gamma 0.95,
                step_advantage_w 1.0, mode mean_std_norm, exact anchors). The GIGPO_* constants
                below, not algorithm.gigpo: that block's defaults are another configuration
                (mean_norm), and this arm's algorithm.gamma is the value estimator's.
                  G_t      the per-turn env rewards' discounted return from turn t (Eq. 5), built per
                           trajectory in TURN order from (traj_uid, pv_t): the batch has been
                           balanced, so its row order is not the turn order the run computed them
                           in. Then the invalid-action penalty subtracted from each row's G -- on
                           this copy, row by row, as apply_invalid_action_penalty subtracts it
                           from GiGPO's step_rewards -- the per-row coefficient being the trainer's.
                  episode  GiGPO's episode term on token_level_rewards (a GRPO-shaped z over the
                           group's turn rows in float32; adjust_batch's copies COUNTED, as released)
                  step     the z of G within each step group: the rows of one uid group with an
                           identical anchor_obs, the row itself in the mean (copies counted; a
                           group of one gets 0)
                  total    episode + step_advantage_w * step
                The two normalisations are this repository's copies of the released functions
                (gigpo/core_gigpo.episode_norm_reward / step_norm_reward, the same arithmetic as
                b2b393c's with exact_statistics off), called on the rows in batch order as that
                run's compute_advantage was. NOT emitted -- and nothing is labelled GiGPO -- when
                the batch has no anchor_obs or no per-turn rewards, an anchor GiGPO cannot group,
                or a trajectory's turns cannot be put back in order.
  shadow_value  records-only arms (records.shadow_value): the value estimator's advantage on this
                batch, from an OBSERVATION-ONLY ProgressValueTable the trainer builds with the one
                builder (opd_ray_trainer.progress_value_config), staged and committed each step as
                the training table would be (dropped on a grad_probe batch). The trainer computes
                it and passes it in, like the training result.

THE RECORD, one JSON line per group in <records.dir>/step<N>.jsonl (the whole file replaced
atomically, so a step re-run after a resume leaves one copy):
  step, task, uid, kind (all_fail / all_success / mixed: the group's trajectories' R = episode
  reward > 0, as the value estimator and progress_rank judge a win), type (ALFWorld's task type,
  else the task), goal_capped, goal_id, gamefile, value ("training" / "shadow" / null: where the
  value columns come from), fallback (the value was the empty-table group fallback), and per
  trajectory (traj_uid order): traj_uid, won, reward, turns, tokens, term (pv_term of its last row),
  k_final and K (pv_k_after and pv_K of its last row); with records.per_turn a "turn" dict of
  equal-length lists in turn order:
    t, k_b, k_a, stag, rem          pv_t, pv_k_before / pv_k_after, pv_stag_before, pv_cap - pv_t
    hold inside at | ongoal optnow buynow | evid     the task's state before the action (pv_*_b)
    valid, tok                      is_action_valid; the row's loss tokens
    w_tok, w_pg                     the row's ACTUAL loss weights: the per-task token weight and,
                                    under actor.pg_loss_norm=trajectory, the policy gradient's own
    adv                             the advantage the actor reads (its mean over the loss tokens)
    v, delta, a_rl, a_fmt           the value estimator's frozen V(z_t), TD error and two terms
    vadv                            (shadow value only) the shadow value's advantage
    grpo                            shadow_grpo
    gigpo, gigpo_step, G, r, anchor, sg    shadow_gigpo's total and step term, the penalised
                                    return G_t its step statistic saw, the turn's env reward, a
                                    short sha1 of the anchor observation (equal ids = one step
                                    group within the uid group) and that step group's size as the
                                    statistic counted it
  Floats to 4 significant digits; a missing value is null.

THE METRICS, records/<task>/<kind>/<component>/<unit>..., for each component (adv; a_rl and a_fmt
in the value arm, shadow_value, shadow_value_a_rl and shadow_value_a_fmt in a records-only one;
shadow_grpo; shadow_gigpo), over the real rows (adjust_batch's copies left out):
  row_mean, row_abs_mean            per row = per turn: the signed and the absolute mean
  traj_sum_mean, traj_sum_abs_mean  per trajectory: its turns' sum, and the sum's absolute value,
                                    averaged over trajectories
  token_total, token_abs_total      sum over rows of w x tokens x A (and x |A|) with w the weight the
                                    actor puts on the policy gradient (TASK_PG_LOSS_WEIGHT_KEY under
                                    pg_loss_norm=trajectory, else TASK_LOSS_WEIGHT_KEY): the
                                    component's part of the step's policy-gradient loss at ratio 1,
                                    summed over its mini-batches (the loss is minus the signed one).
                                    Left out when the batch carries no per-task weight.
and the population each is over: records/<task>/<kind>/{groups, trajectories, rows, tokens}. With
progress_rank off, also the traj/<task>/* metrics and the think-block share progress_rank reports,
through its own pure functions (trajectory_metrics, think_block_metrics) on a faithful copy of the
per-trajectory summary its controller builds (trajectory_summaries; a test holds the two equal).
"""

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY
from verl.trainer.ppo.progress_value import alfworld_type
from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY, TASK_PG_LOSS_WEIGHT_KEY

__all__ = ["RecordsConfig", "StepRecords", "KINDS", "GIGPO_GAMMA", "GIGPO_STEP_ADVANTAGE_W", "GIGPO_MODE",
           "check_records_config", "shadow_grpo", "gigpo_step_returns", "penalised_step_returns", "shadow_gigpo",
           "trajectory_summaries", "trajectory_and_think_metrics", "compute_step_records", "write_jsonl_atomic"]

KINDS = ("all_fail", "all_success", "mixed")

# OPD+GiGPO's estimator as that run (b2b393c, run_multitask_opd_gigpo_qwen3.sh) had it. Constants, not
# algorithm.gigpo.*: the shadow is that run's estimator, whatever this run's block says.
GIGPO_GAMMA = 0.95               # algorithm.gamma=0.95: the per-turn returns' discount (Eq. 5)
GIGPO_STEP_ADVANTAGE_W = 1.0     # algorithm.gigpo.step_advantage_w=1.0 (omega, Eq. 8)
GIGPO_MODE = "mean_std_norm"     # algorithm.gigpo.mode=mean_std_norm: both terms divided by std + eps
GIGPO_EPSILON = 1e-6             # compute_gigpo_outcome_advantage's default

# The state before the action written into a turn's record, per task (the pv_<stem>_b columns).
STATE_STEMS = {"alfworld": ("hold", "inside", "at"), "webshop": ("ongoal", "optnow", "buynow"),
               "search": ("evid",)}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecordsConfig:
    """algorithm.progress_value.records (see ppo_trainer.yaml)."""

    enable: bool = False
    dir: Optional[str] = None
    shadow_grpo: bool = True
    shadow_gigpo: bool = True
    shadow_value: bool = True
    per_turn: bool = True

    @classmethod
    def from_config(cls, config) -> "RecordsConfig":
        """From the full config; a missing block or key keeps its default (a missing block is off)."""
        alg = config.get("algorithm", None) or {}
        node = (alg.get("progress_value", None) or {}).get("records", None) or {}

        def flag(key, default):
            v = node.get(key, None)
            return default if v is None else bool(v)

        d = node.get("dir", None)
        return cls(enable=flag("enable", False), dir=None if d in (None, "") else str(d),
                   shadow_grpo=flag("shadow_grpo", True), shadow_gigpo=flag("shadow_gigpo", True),
                   shadow_value=flag("shadow_value", True), per_turn=flag("per_turn", True))


def check_records_config(config) -> RecordsConfig:
    """The records' own configuration, refused where it cannot deliver what it says; returned.

    records.enable reads the pv_* columns and the progress counters, which the environment managers
    and the rollout loop write only under algorithm.progress_value.enable. (The records-only shadow
    table's own keys are checked by the trainer, through the builder that makes it.)
    """
    rcfg = RecordsConfig.from_config(config)
    if rcfg.enable:
        pv_cfg = (config.get("algorithm", None) or {}).get("progress_value", None) or {}
        assert bool(pv_cfg.get("enable", False)), (
            "algorithm.progress_value.records.enable needs algorithm.progress_value.enable=True: the "
            "records read the pv_* columns and the progress counters, which are written only with it on")
    return rcfg


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _num(x):
    """A JSON number at 4 significant digits (an int when integral), None when missing or not finite."""
    v = _finite(x)
    if v is None:
        return None
    if v.is_integer() and abs(v) < 2.0 ** 53:
        return int(v)
    return float(f"{v:.4g}")


def _nums(xs) -> list:
    return [_num(x) for x in xs]


def _anchor_id(key) -> str:
    """A short, process-independent id of an anchor observation (python's hash() is salted per process)."""
    s = key if isinstance(key, str) else repr(key)
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _column(nt: Mapping[str, Any], name: str, n: int) -> Optional[np.ndarray]:
    if name not in nt:
        return None
    a = np.asarray(nt[name]).reshape(-1)
    if len(a) != n:
        raise ValueError(f"records: column {name!r} has {len(a)} rows, the batch {n}")
    return a


def loss_mask(batch, multi_turn: bool) -> torch.Tensor:
    """The mask the actor's loss reads, (rows, response): loss_mask in multi-turn mode, else the response
    part of attention_mask (dp_actor.update_policy; the same choice _apply_progress_rank makes)."""
    resp_len = batch.batch["responses"].shape[1]
    key = "loss_mask" if (multi_turn and "loss_mask" in batch.batch.keys()) else "attention_mask"
    return batch.batch[key][:, -resp_len:]


# --------------------------------------------------------------------------- #
# The shadows
# --------------------------------------------------------------------------- #

def shadow_grpo(token_level_rewards: torch.Tensor, uid, traj_uid, *, padding_mask=None, exclude_mask=None,
                floor_mask=None, floor_value: float = 0.0, norm_adv_by_std_in_grpo: bool = True,
                compute_mean_std_cross_steps: bool = True) -> np.ndarray:
    """Per row, the float32 scalar compute_advantage's GRPO branch broadcasts over the row's tokens.

    The same function with the same arguments as that branch (padding, exclude and floor columns
    included); only the mask it multiplies by at the end is ones, which hands back the scalar itself --
    the statistic never reads the mask. It sums token_level_rewards into a new tensor and writes only
    that one.
    """
    from verl.trainer.ppo import core_algos

    ones = torch.ones(token_level_rewards.shape[0], 1, dtype=token_level_rewards.dtype,
                      device=token_level_rewards.device)
    adv, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=ones, index=uid, traj_index=traj_uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo, padding_mask=padding_mask,
        exclude_mask=exclude_mask, floor_mask=floor_mask, floor_value=float(floor_value),
        compute_mean_std_cross_steps=bool(compute_mean_std_cross_steps))
    return adv[:, 0].detach().cpu().numpy()


def _turn_order(tuids, turns, real) -> Tuple[Dict[Any, List[int]], Dict[Tuple[Any, int], int]]:
    """``({traj_uid: real rows in turn order}, {(traj_uid, turn): real row})``; raises ValueError when a
    trajectory's real rows do not carry the turns 0..T-1 exactly (they could not be put back in order)."""
    by_traj: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
    for i in range(len(tuids)):
        if not real[i]:
            continue
        t = _finite(turns[i])
        if t is None or not t.is_integer() or t < 0:
            raise ValueError(f"records: row {i} (trajectory {tuids[i]}) has turn {turns[i]!r}")
        by_traj[tuids[i]].append((int(t), i))
    order, row_of = {}, {}
    for tu, pairs in by_traj.items():
        pairs.sort()
        ts = [t for t, _ in pairs]
        if ts != list(range(len(ts))):
            raise ValueError(f"records: trajectory {tu} has turns {ts}, not 0..{len(ts) - 1}")
        order[tu] = [i for _, i in pairs]
        for t, i in pairs:
            row_of[(tu, t)] = i
    return order, row_of


def gigpo_step_returns(rewards, tuids, turns, real, *, gamma: float = GIGPO_GAMMA) -> torch.Tensor:
    """Per row, its trajectory's discounted return from the row's turn: GiGPO's step_rewards (Eq. 5).

    compute_step_discounted_returns' arithmetic line for line -- float32 rewards, a python running
    return, a float32 store -- but over each trajectory's rows in TURN order (traj_uid, turn), where
    that function takes them in batch order: it ran on the rollout's own order, and this batch has been
    balanced since. A padding copy takes its original's return, as it did there (the column was
    computed before adjust_batch and travelled with the rows).
    """
    n = len(tuids)
    r32 = np.asarray(rewards, dtype=object).reshape(-1).astype(np.float32)
    order, row_of = _turn_order(tuids, turns, real)
    all_returns = np.zeros(n, dtype=np.float32)
    for tu in order:
        idx = order[tu]
        traj_rewards = r32[idx]
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        all_returns[idx] = traj_returns
    for i in range(n):
        if not real[i]:
            src = row_of.get((tuids[i], int(float(turns[i]))))
            if src is None:
                raise ValueError(f"records: padding row {i} (trajectory {tuids[i]}) has no original")
            all_returns[i] = all_returns[src]
    return torch.tensor(all_returns, dtype=torch.float32)


def penalised_step_returns(step_returns: torch.Tensor, valid, coefs) -> torch.Tensor:
    """A COPY of ``step_returns`` with the invalid-action penalty subtracted per row, as
    apply_invalid_action_penalty does it to GiGPO's step_rewards (the same float32 operations:
    ``coef * float32(1 - valid)`` taken from the row). ``coefs`` None = the penalty is off."""
    out = step_returns.clone()
    if coefs is None:
        return out
    for i in range(out.shape[0]):
        action_valids = np.asarray(valid[i]).astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32)
        out[i] -= float(coefs[i]) * action_invalids
    return out


def shadow_gigpo(token_level_rewards: torch.Tensor, step_returns: torch.Tensor, anchor_obs, uid, traj_uid, *,
                 step_advantage_w: float = GIGPO_STEP_ADVANTAGE_W, mode: str = GIGPO_MODE,
                 epsilon: float = GIGPO_EPSILON) -> Dict[str, np.ndarray]:
    """GiGPO's joint advantage per row, and its parts; ``step_returns`` already penalised.

    The released normalisations (gigpo.core_gigpo.episode_norm_reward / step_norm_reward) on every row
    in batch order, adjust_batch's copies included as released; the step groups are the uid groups'
    identical anchors (build_step_group without similarity, keyed deterministically instead of by
    uuid4, which also prints a line per call). Raises TypeError for an anchor GiGPO cannot hash.
    Returns float32 arrays ``total``, ``episode``, ``step``, the step-group ``key`` per row (uid,
    hashable anchor) and the step group's ``size`` as the statistic counts it.
    """
    from gigpo import core_gigpo

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    n = token_level_rewards.shape[0]
    ones = torch.ones(n, 1, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    episode = core_gigpo.episode_norm_reward(token_level_rewards, ones, uid, traj_uid, epsilon, remove_std)[:, 0]
    keys = [(uid[i], core_gigpo.to_hashable(anchor_obs[i])) for i in range(n)]
    size: Dict[Any, int] = defaultdict(int)
    for k in keys:
        size[k] += 1
    step = core_gigpo.step_norm_reward(step_returns, ones, keys, epsilon, remove_std)[:, 0]
    total = episode + step_advantage_w * step
    return {"total": total.detach().cpu().numpy(), "episode": episode.detach().cpu().numpy(),
            "step": step.detach().cpu().numpy(), "key": keys,
            "size": np.array([size[k] for k in keys], dtype=np.int64)}


# --------------------------------------------------------------------------- #
# progress_rank's rollout metrics, without its controller
# --------------------------------------------------------------------------- #

def trajectory_summaries(*, real, tuids, task_names, episode_rewards, tokens, valid_rows=None,
                         coverage_rows=None, episode_lengths=None, task_score_rows=None,
                         committed_rows=None, revisit_rows=None, done_walkset_rows=None) -> Dict[str, dict]:
    """``{traj_uid: summary}`` over the real rows: the part of ProgressRankController.apply's
    per-trajectory dict that trajectory_metrics reads, built the same way field for field (turns,
    tokens, reward, invalid turns, coverage D, length, task_score, committed, revisits, done_walkset,
    won). A copy because the controller builds it inline and applying the controller would move its
    state (EMAs, Beta counts); tests/trainer/test_rollout_records.py holds the two metric sets equal.
    """
    names = np.asarray([str(x) for x in task_names])
    invalid = (None if valid_rows is None
               else np.asarray([_finite(v) == 0.0 for v in valid_rows], dtype=bool))
    traj: Dict[str, dict] = {}
    for i in range(len(names)):
        if not real[i]:
            continue
        t = str(tuids[i])
        x = traj.get(t)
        if x is None:
            x = traj[t] = {"task": names[i], "turns": 0, "tokens": 0.0, "reward": None, "invalid": 0,
                           "d": None, "d_ok": coverage_rows is not None, "length": None, "task_score": None,
                           "committed": None, "revisits": None, "done_walkset": None}
        x["turns"] += 1
        if episode_lengths is not None:
            ln = _finite(episode_lengths[i])
            if ln is not None:
                x["length"] = ln if x["length"] is None else max(x["length"], ln)
        if task_score_rows is not None:
            ts = _finite(task_score_rows[i])
            if ts is not None:
                x["task_score"] = ts if x["task_score"] is None else max(x["task_score"], ts)
        if committed_rows is not None:
            cm = _finite(committed_rows[i])
            if cm is not None:
                x["committed"] = bool(x["committed"]) or cm > 0.5
        for col, key in ((revisit_rows, "revisits"), (done_walkset_rows, "done_walkset")):
            if col is not None:
                v = _finite(col[i])
                if v is not None:
                    x[key] = v if x[key] is None else max(x[key], v)
        x["tokens"] += float(tokens[i])
        r = _finite(episode_rewards[i])
        if r is not None:
            x["reward"] = r if x["reward"] is None else max(x["reward"], r)
        if invalid is not None and invalid[i]:
            x["invalid"] += 1
        if coverage_rows is not None:
            d = _finite(coverage_rows[i])
            if d is None:
                x["d_ok"] = False
            else:
                x["d"] = d if x["d"] is None else max(x["d"], d)
    for x in traj.values():
        x["won"] = bool(x["reward"] is not None and x["reward"] > 0.0)
        if not x["d_ok"]:
            x["d"] = None
    return traj


def trajectory_and_think_metrics(batch, *, task_names, real, mask, turn_caps=None) -> Dict[str, float]:
    """traj/<task>/* and the think-block share, as ProgressRankController.apply reports them.

    Through progress_rank's pure functions (failed_groups, trajectory_progress, trajectory_metrics,
    think_block_metrics) on the inputs _apply_progress_rank hands the controller: the loss mask's
    tokens, progress_k / progress_total for the failures' k / K, and the per-trajectory summary
    (trajectory_summaries). The alternative-count comparisons (progress_rank/<task>/alt_*) are (a)'s
    and are not reported here.
    """
    from verl.trainer.ppo.progress_rank import (COVERAGE_D_KEY, PROGRESS_K_KEY, PROGRESS_TOTAL_KEY,
                                                failed_groups, think_block_metrics, trajectory_metrics,
                                                trajectory_progress)

    nt = batch.non_tensor_batch
    n = len(batch)
    names = np.asarray([str(x) for x in task_names])
    uids, tuids = nt["uid"], nt["traj_uid"]
    real = np.asarray(real, dtype=bool)
    real_idx = [i for i in range(n) if real[i]]
    groups = failed_groups(uids, tuids, names, nt["episode_rewards"], real_idx)
    tokens = mask.to(torch.float64).sum(-1).cpu().numpy()
    get = lambda key: nt[key] if key in nt else None  # noqa: E731
    traj = trajectory_summaries(real=real, tuids=tuids, task_names=names, episode_rewards=nt["episode_rewards"],
                                tokens=tokens, valid_rows=get("is_action_valid"), coverage_rows=get(COVERAGE_D_KEY),
                                episode_lengths=get("episode_lengths"), task_score_rows=get("task_score"),
                                committed_rows=get("committed"), revisit_rows=get("revisits"),
                                done_walkset_rows=get("progress_done_walkset"))
    traj_prog = (trajectory_progress(tuids, nt[PROGRESS_K_KEY], nt[PROGRESS_TOTAL_KEY], range(n))
                 if PROGRESS_K_KEY in nt and PROGRESS_TOTAL_KEY in nt else {})
    out = trajectory_metrics(groups, traj, tuids=tuids, real=real,
                             tasks=list(dict.fromkeys(names[real].tolist())), traj_prog=traj_prog,
                             turn_caps=turn_caps, alt_prog=None, have_invalid="is_action_valid" in nt)
    out.update(think_block_metrics(responses=batch.batch["responses"], mask=mask, task_names=task_names,
                                   real=real))
    return out


# --------------------------------------------------------------------------- #
# One step's records
# --------------------------------------------------------------------------- #

@dataclass
class StepRecords:
    groups: List[dict]                      # one JSON-able record per group
    metrics: Dict[str, float]
    notes: List[str] = field(default_factory=list)   # why a shadow was not emitted, for a warning


def compute_step_records(batch, *, step: int, cfg: RecordsConfig, task_names, multi_turn: bool = False,
                         pg_loss_norm: str = "token", value=None, value_source: Optional[str] = None,
                         grpo_kwargs: Optional[Mapping[str, Any]] = None, penalty_coefs=None,
                         turn_caps: Optional[Mapping[str, float]] = None,
                         traj_metrics: bool = True) -> StepRecords:
    """This step's group records and records/* metrics (see the module docstring). Reads ``batch``.

    ``task_names``     per row, the canonical task (get_task_names, or the single-task env's)
    ``pg_loss_norm``   actor.pg_loss_norm: which column is the policy gradient's actual weight
    ``value``          a progress_value.ProgressValueResult on these rows, in this order, or None
    ``value_source``   "training" (the value arm's own result) or "shadow" (the records-only table)
    ``grpo_kwargs``    compute_advantage's GRPO arguments beyond the batch's own columns:
                       norm_adv_by_std_in_grpo, compute_mean_std_cross_steps, exclude_mask,
                       floor_mask, floor_value
    ``penalty_coefs``  per row, the invalid-action penalty's coefficient for GiGPO's step returns
                       (None: the run applies no penalty)
    ``turn_caps``      {task: H}, for traj/<task>/fail_at_cap
    ``traj_metrics``   report traj/* and the think-block share (off when progress_rank reports them)
    """
    tb, nt = batch.batch, batch.non_tensor_batch
    n = len(batch)
    names = np.asarray([str(x) for x in np.asarray(task_names, dtype=object).reshape(-1)], dtype=object)
    if len(names) != n:
        raise ValueError(f"records: {len(names)} task names for {n} rows")
    uid = _column(nt, "uid", n)
    tuid = _column(nt, "traj_uid", n)
    rewards_ep = _column(nt, "episode_rewards", n)
    if uid is None or tuid is None or rewards_ep is None:
        raise KeyError("records: the batch needs uid, traj_uid and episode_rewards")
    pad = tb.get(PADDING_ROW_KEY, None)
    real = np.ones(n, dtype=bool) if pad is None else ~pad.reshape(-1).to(torch.bool).cpu().numpy()
    turns = _column(nt, "pv_t", n)
    if turns is None:
        turns = _column(nt, "turn_step", n)
    if turns is None:
        raise KeyError("records: the batch has neither pv_t nor turn_step, so no turn order")
    order, row_of = _turn_order(tuid, turns, real)
    notes: List[str] = []

    mask = loss_mask(batch, multi_turn)
    m64 = mask.to(torch.float64)
    tokens = m64.sum(-1).cpu().numpy()
    adv = tb["advantages"].to(torch.float64)
    adv_row = ((adv * m64).sum(-1) / m64.sum(-1).clamp(min=1.0)).cpu().numpy()
    w_tok = tb[TASK_LOSS_WEIGHT_KEY].to(torch.float64).reshape(-1).cpu().numpy() \
        if TASK_LOSS_WEIGHT_KEY in tb.keys() else None
    w_pg = tb[TASK_PG_LOSS_WEIGHT_KEY].to(torch.float64).reshape(-1).cpu().numpy() \
        if TASK_PG_LOSS_WEIGHT_KEY in tb.keys() else None
    # The weight the actor multiplies the policy-gradient term by: its own column under
    # pg_loss_norm=trajectory (the actor refuses to train without it), else the token weight.
    w_act = w_pg if pg_loss_norm == "trajectory" else w_tok
    if w_act is None:
        notes.append(f"token_total / token_abs_total not reported: the batch carries no "
                     f"{TASK_PG_LOSS_WEIGHT_KEY if pg_loss_norm == 'trajectory' else TASK_LOSS_WEIGHT_KEY} "
                     f"(pg_loss_norm={pg_loss_norm}), so the policy gradient's actual weight is unknown here")
    valid = _column(nt, "is_action_valid", n)

    # ---- the per-row components ----
    comps: Dict[str, np.ndarray] = {"adv": adv_row}
    if value is not None:
        if len(np.asarray(value.advantage)) != n:
            raise ValueError(f"records: the value result has {len(value.advantage)} rows, the batch {n}")
        if value_source == "training":
            comps["a_rl"], comps["a_fmt"] = np.asarray(value.a_rl), np.asarray(value.a_fmt)
        else:
            comps["shadow_value"] = np.asarray(value.advantage)
            comps["shadow_value_a_rl"], comps["shadow_value_a_fmt"] = np.asarray(value.a_rl), np.asarray(value.a_fmt)
    tlr = tb["token_level_rewards"] if "token_level_rewards" in tb.keys() else None
    grpo = None
    if cfg.shadow_grpo:
        if tlr is None:
            notes.append("shadow_grpo: the batch has no token_level_rewards")
        else:
            grpo = shadow_grpo(tlr, nt["uid"], nt["traj_uid"], padding_mask=pad, **dict(grpo_kwargs or {}))
            comps["shadow_grpo"] = grpo.astype(np.float64)
    gg, step_g = None, None
    if cfg.shadow_gigpo:
        missing = [c for c in ("anchor_obs", "rewards") if c not in nt]
        if tlr is None:
            missing.append("token_level_rewards")
        active = _column(nt, "active_masks", n)
        if missing:
            notes.append(f"shadow_gigpo: the batch has no {missing}; not emitted")
        elif active is not None and not all(bool(a) for a in active[real]):
            notes.append("shadow_gigpo: a row has active_masks False (GiGPO asserts every row active); not emitted")
        else:
            try:
                g = gigpo_step_returns(nt["rewards"], tuid, turns, real)
                step_g = penalised_step_returns(g, valid if valid is not None else np.ones(n, dtype=bool),
                                                penalty_coefs)
                gg = shadow_gigpo(tlr, step_g, nt["anchor_obs"], nt["uid"], nt["traj_uid"])
                comps["shadow_gigpo"] = gg["total"].astype(np.float64)
            except (TypeError, ValueError) as e:
                gg = None
                notes.append(f"shadow_gigpo: not emitted ({e!r})")

    # ---- groups, trajectories, kinds ----
    traj_R = {tu: 1.0 if max(_finite(rewards_ep[i]) or 0.0 for i in rows) > 0.0 else 0.0
              for tu, rows in order.items()}
    groups: Dict[Any, List[Any]] = defaultdict(list)
    for tu in sorted(order, key=str):
        groups[uid[order[tu][0]]].append(tu)
    kind_of = {}
    for u, trs in groups.items():
        rs = {traj_R[tu] for tu in trs}
        kind_of[u] = "mixed" if len(rs) > 1 else ("all_success" if rs == {1.0} else "all_fail")
    task_of_group = {u: str(names[order[trs[0]][0]]) for u, trs in groups.items()}

    metrics: Dict[str, float] = {}
    sel: Dict[Tuple[str, str], dict] = {}
    for u, trs in groups.items():
        s = sel.setdefault((task_of_group[u], kind_of[u]), {"groups": 0, "trajs": [], "all": []})
        s["groups"] += 1
        s["trajs"].extend(order[tu] for tu in trs)
    for i in range(n):
        key = (task_of_group.get(uid[i]), kind_of.get(uid[i]))
        if key in sel:
            sel[key]["all"].append(i)
    for (task, kind), s in sorted(sel.items()):
        p = f"records/{task}/{kind}/"
        rows = np.array([i for rs in s["trajs"] for i in rs], dtype=np.int64)
        metrics[p + "groups"] = float(s["groups"])
        metrics[p + "trajectories"] = float(len(s["trajs"]))
        metrics[p + "rows"] = float(len(rows))
        metrics[p + "tokens"] = float(tokens[rows].sum())
        every = np.array(s["all"], dtype=np.int64)
        for comp, x in comps.items():
            q = f"{p}{comp}/"
            metrics[q + "row_mean"] = float(np.mean(x[rows]))
            metrics[q + "row_abs_mean"] = float(np.mean(np.abs(x[rows])))
            sums = np.array([float(np.sum(x[rs])) for rs in s["trajs"]])
            metrics[q + "traj_sum_mean"] = float(np.mean(sums))
            metrics[q + "traj_sum_abs_mean"] = float(np.mean(np.abs(sums)))
            if w_act is not None:
                wt = w_act[every] * tokens[every]
                metrics[q + "token_total"] = float(np.sum(wt * x[every]))
                metrics[q + "token_abs_total"] = float(np.sum(wt * np.abs(x[every])))
    if grpo is not None:
        metrics["records/shadow_grpo/max_abs_diff_adv"] = float(
            np.max(np.abs(grpo.astype(np.float64)[real] - adv_row[real]))) if real.any() else 0.0
    if cfg.shadow_gigpo:
        metrics["records/shadow_gigpo/available"] = float(gg is not None)
    if traj_metrics:
        metrics.update(trajectory_and_think_metrics(batch, task_names=names, real=real, mask=mask,
                                                    turn_caps=turn_caps))

    # ---- the records ----
    col = {c: _column(nt, c, n) for c in ("pv_k_before", "pv_k_after", "pv_stag_before", "pv_cap", "pv_term",
                                          "pv_K", "goal_capped", "goal_id", "gamefile", "rewards")}
    stems = {task: [(stem, c) for stem in STATE_STEMS.get(task, ())
                    for c in [_column(nt, f"pv_{stem}_b", n)] if c is not None]
             for task in set(task_of_group.values())}

    records: List[dict] = []
    for u in sorted(groups, key=str):
        trs = groups[u]
        task = task_of_group[u]
        grows = [i for tu in trs for i in order[tu]]
        rec: Dict[str, Any] = {"step": int(step), "task": task, "uid": str(u), "kind": kind_of[u]}
        gamefile = None
        if col["gamefile"] is not None:
            gamefile = next((str(col["gamefile"][i]) for i in grows if str(col["gamefile"][i] or "")), None)
        rec["type"] = alfworld_type(gamefile) if task == "alfworld" else task
        capped = [_finite(col["goal_capped"][i]) for i in grows] if col["goal_capped"] is not None else []
        capped = [c for c in capped if c is not None]
        rec["goal_capped"] = (int(any(c == 1.0 for c in capped)) if capped else None)
        gid = [_finite(col["goal_id"][i]) for i in grows] if col["goal_id"] is not None else []
        gid = [g for g in gid if g is not None]
        rec["goal_id"] = int(gid[0]) if gid else None
        rec["gamefile"] = gamefile
        rec["value"] = value_source if value is not None else None
        if value is not None:
            rec["fallback"] = bool(np.asarray(value.fallback)[grows[0]])
        trecs = []
        for tu in trs:
            rows = order[tu]
            last = rows[-1]
            tr: Dict[str, Any] = {"traj_uid": str(tu), "won": int(traj_R[tu]), "reward": _num(rewards_ep[last]),
                                  "turns": len(rows), "tokens": _num(tokens[rows].sum())}
            for key, c in (("term", "pv_term"), ("k_final", "pv_k_after"), ("K", "pv_K")):
                tr[key] = _num(col[c][last]) if col[c] is not None else None
            if cfg.per_turn:
                tt: Dict[str, list] = {"t": _nums(turns[rows])}
                for key, c in (("k_b", "pv_k_before"), ("k_a", "pv_k_after"), ("stag", "pv_stag_before")):
                    if col[c] is not None:
                        tt[key] = _nums(col[c][rows])
                if col["pv_cap"] is not None:
                    tt["rem"] = [_num(_finite(col["pv_cap"][i]) - _finite(turns[i]))
                                 if _finite(col["pv_cap"][i]) is not None else None for i in rows]
                for stem, c in stems[task]:
                    tt[stem] = _nums(c[rows])
                if valid is not None:
                    tt["valid"] = [int(bool(valid[i])) for i in rows]
                tt["tok"] = _nums(tokens[rows])
                if w_tok is not None:
                    tt["w_tok"] = _nums(w_tok[rows])
                if w_pg is not None:
                    tt["w_pg"] = _nums(w_pg[rows])
                tt["adv"] = _nums(adv_row[rows])
                if value is not None:
                    tt["v"] = _nums(np.asarray(value.value)[rows])
                    tt["delta"] = _nums(np.asarray(value.delta)[rows])
                    tt["a_rl"] = _nums(np.asarray(value.a_rl)[rows])
                    tt["a_fmt"] = _nums(np.asarray(value.a_fmt)[rows])
                    if value_source != "training":
                        tt["vadv"] = _nums(np.asarray(value.advantage)[rows])
                if grpo is not None:
                    tt["grpo"] = _nums(grpo[rows])
                if gg is not None:
                    tt["gigpo"] = _nums(gg["total"][rows])
                    tt["gigpo_step"] = _nums(gg["step"][rows])
                    tt["G"] = _nums(step_g.numpy()[rows])
                    tt["r"] = _nums(col["rewards"][rows])
                    tt["anchor"] = [_anchor_id(gg["key"][i][1]) for i in rows]
                    tt["sg"] = [int(gg["size"][i]) for i in rows]
                tr["turn"] = tt
            trecs.append(tr)
        rec["traj"] = trecs
        records.append(rec)
    return StepRecords(groups=records, metrics=metrics, notes=notes)


def write_jsonl_atomic(path: str, records, extra: Optional[Mapping[str, Any]] = None) -> None:
    """One JSON line per record into ``path``, through a temporary file in the same folder and os.replace:
    a reader finds the previous file or the complete new one, never a truncated one, and a step re-run
    after a resume replaces its own file instead of appending a second copy. NaN is refused
    (allow_nan=False: it is not JSON). A failed write leaves no temporary file behind and raises."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps({**dict(extra or {}), **rec}, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
