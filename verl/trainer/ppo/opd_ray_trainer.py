"""
OPD (On-Policy Distillation) Trainer — multitask.

Unlike OPSD/SDAR (which use the *same* policy with a privileged skill prepend as
the teacher), OPD routes each sample to a *separate, single-task RL-trained*
teacher checkpoint based on its ``task_name`` (e.g. an alfworld sample is
distilled from the alfworld teacher). The student is trained purely by the KL
to its per-task teacher, evaluated on the student's own on-policy responses —
no GRPO policy-gradient, entropy, reference-KL, or reward signal enters the loss.

Teachers are created as ``role="ref"`` worker groups (one per task), each loading
a distinct ``model.path``. ``role="ref"`` forces FSDP ``CPUOffload`` at build time,
so the teachers live on CPU and only ride to GPU during log-prob computation.
"""

import copy
import json
import os
from collections import namedtuple
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import open_dict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import DataProtoConfig, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.privileged_notice import (
    TASKS as NOTICE_TASKS,
    fingerprint_adjust as notice_fingerprint_adjust,
    leak_flags as notice_leak_flags,
    normalize_task as notice_normalize_task,
    notice_prefix_ids,
    parse_notice_config,
    prepend_prefix as notice_prepend_prefix,
    verify_doc_hashes as notice_verify_doc_hashes,
)
from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_metrics_by_task,
    compute_trajectory_response_tokens,
    compute_throughout_metrics,
    compute_timing_metrics,
    get_task_names,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    _timer,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.sign_weights import SIGN_BASE_TASK
from verl.trainer.ppo.task_loss_weights import attach_task_loss_weights
from verl.utils import gpu_profiler
from verl.utils.metric import reduce_metrics

from agent_system.multi_turn_rollout import adjust_batch

# Rows per call in the sign-weight passes. The teacher's forward returns every
# row's hidden states in one tensor, so this bounds the transient that tensor and
# its concatenation make -- the standing cost is the cache, which is the same
# either way. 512 matches the ceiling the rollout prefetch already runs at.
_SIGN_WEIGHT_FORWARD_CHUNK = max(1, int(os.environ.get("SIGN_WEIGHT_FORWARD_CHUNK", "512")))

# Score the base policy and the off-task teachers inside the rollout window too,
# instead of only the on-task one. Set to 0 to go back to running all three in
# sign_weight_forward after the rollout. See _teacher_prefetch_chunk.
_ROLLOUT_PREFETCH_SIGN = os.environ.get("ROLLOUT_PREFETCH_SIGN", "1") not in ("0", "false", "False")

# What a prefetched row carries on a cross-teacher arm. Wrapped in a named pair
# rather than appended to the on-task value because that value is already three
# different shapes depending on the mode (an id, a triple, a tensor), and a
# fourth "sometimes one longer" is how a tuple gets unpacked wrong somewhere.
PrefetchedRow = namedtuple("PrefetchedRow", ("on_task", "sign_ids"))

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# The reset is pure CPU / subprocess / HTTP work and the env managers are idle
# between rollouts; the reset still runs exactly once per rollout and in the same
# order, so stateful env schedules (alfworld's game-file iterator) are unchanged.
# Opt-in; see TrajectoryCollector.prefetch_env_reset.
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


def compute_opd_data_metrics(batch: DataProto) -> dict:
    """Lightweight, advantage-free replacement for ``compute_data_metrics``.

    ``compute_data_metrics`` unconditionally reads ``advantages``/``returns``/
    ``token_level_rewards``; OPD never computes advantages, so we report only
    sequence-length stats plus (defensively) any reward/episode signals that
    happen to be present for monitoring.
    """
    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    max_response_length = batch.batch["responses"].shape[-1]
    max_prompt_length = batch.batch["attention_mask"].shape[-1] - max_response_length

    metrics = {
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Monitoring-only: env reward / success rate, never fed to the loss.
    if "token_level_scores" in batch.batch:
        seq_score = batch.batch["token_level_scores"].sum(-1)
        metrics["opd/score/mean"] = torch.mean(seq_score).detach().item()
        metrics["opd/score/max"] = torch.max(seq_score).detach().item()
        metrics["opd/score/min"] = torch.min(seq_score).detach().item()
    if "traj_uid" in batch.non_tensor_batch:
        _, unique_idx = np.unique(batch.non_tensor_batch["traj_uid"], return_index=True)
        for k, v in batch.non_tensor_batch.items():
            if "success_rate" in k:
                metrics[f"episode/{k}"] = float(v[0])
        if "episode_rewards" in batch.non_tensor_batch:
            metrics["episode/reward/mean"] = float(batch.non_tensor_batch["episode_rewards"][unique_idx].mean())
        # tokens a whole trajectory generated (response_length above is per turn)
        trajectory_response_tokens = compute_trajectory_response_tokens(batch)
        if trajectory_response_tokens is not None:
            metrics["episode/response_tokens/mean"] = float(trajectory_response_tokens.mean())
            metrics["episode/response_tokens/max"] = float(trajectory_response_tokens.max())
            metrics["episode/response_tokens/min"] = float(trajectory_response_tokens.min())

    return metrics


def compute_opd_data_metrics_by_task(batch: DataProto) -> dict:
    """Per-task breakdown of :func:`compute_opd_data_metrics`.

    Success rates are dropped from the per-task slices: they are batch-wide
    constants broadcast onto every row, and the multitask env manager already
    reports them per task as ``episode/{task}_success_rate``.
    """
    return compute_metrics_by_task(
        batch,
        lambda task_batch: {
            name: value for name, value in compute_opd_data_metrics(task_batch).items() if "success_rate" not in name
        },
    )


def check_cross_teacher_kl_weight_prerequisites(*, teacher_topk_kl, base_policy_path, n_teachers):
    """What the parameter-free arm needs before it can run.

    The same three structural requirements the sign arm has -- a shared top-k
    support, the exact base checkpoint the teachers were fine-tuned from, and at
    least one off-task teacher -- and one more: corroboration is measured among
    the OFF-TASK teachers, so two of them are needed before the channel exists
    at all. One teacher agreeing with itself is not corroboration, and with a
    single source the arm silently degenerates to the reliability channel alone.
    """
    assert teacher_topk_kl, (
        "cross_teacher_kl_weight requires algorithm.opd.kl_loss_type=topk_kl: the "
        "single-token estimator gives no support for the four models to share"
    )
    assert base_policy_path, (
        "cross_teacher_kl_weight requires algorithm.opd.cross_teacher_kl_weight.base_path "
        "(the pre-RL policy the teachers' shifts are measured against)"
    )
    assert n_teachers >= 3, (
        "cross_teacher_kl_weight needs at least two off-task teachers: the "
        "corroboration channel is their agreement with EACH OTHER, and one "
        "source cannot corroborate itself"
    )


def check_sign_weight_prerequisites(*, mode, teacher_topk_kl, base_policy_path, n_teachers):
    """What an arm needs before the weighting can run at all.

    Module-level so a test can ask "would this arm start?" by calling the same
    thing the trainer does, instead of a copy that drifts. It is also the record
    of what is NOT a prerequisite: the support may be the student's top-k or the
    teacher's own, and for a long time this refused the second one -- which is
    how a teacher-indexed weighted arm came to abort at trainer init.

    Raises AssertionError with the setting to change.
    """
    assert mode in ("position", "target"), (
        f"algorithm.opd.sign_weight.mode must be 'position' or 'target', got {mode!r}"
    )
    # The signal is the sign of log pi_m - log pi_0 on a support shared by all
    # four models. EITHER top-k is such a support. What the mechanism cannot work
    # from is the single-token estimator, which produces no support at all.
    assert teacher_topk_kl, (
        "sign weighting requires algorithm.opd.kl_loss_type=topk_kl: the "
        "single-token estimator gives no support for the four models to share"
    )
    assert base_policy_path, (
        "sign weighting requires algorithm.opd.sign_weight.base_path "
        "(the pre-RL policy the teachers' shifts are measured against)"
    )
    assert n_teachers >= 2, (
        "sign weighting needs at least one off-task teacher besides the on-task one"
    )



def _oci_adherence(tokenizer, batch, cand_mask):
    """How far down the block the candidate actually walks.

    THE QUANTITY THE AGGREGATES COULD NOT SEE. cand_fail_rate says whether the
    episode failed; rho says whether the tokens are reachable. Neither says how
    many steps of the path were taken, which is the only thing that decides
    whether a turn-budget corruption can work at all: comply for k steps and
    50-k turns remain, against the ~20 an unaided episode takes. misdirect
    established that the block IS read and that a refuted line ends the
    compliance -- this counts the steps.

    Read off the prompt, not from any bookkeeping: the block is in the
    candidate's own prompt, so the numbered lines are decoded once per prompt
    group and compared against the <action> the projection would have taken
    (same extraction: the text between the tags, stripped and lowercased).
    ``leading_k`` is consecutive compliance from the trajectory's first turn --
    the only form that spends the budget -- and ``matched`` counts agreement
    anywhere, which is higher whenever the student rejoins the path later.
    """
    import re
    import numpy as np

    from agent_system.environments.env_manager import _PLAN_FOOTER, _PLAN_HEADER

    uids = batch.non_tensor_batch.get("uid", None)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    turns = batch.non_tensor_batch.get("turn_step", None)
    if uids is None or tuids is None or turns is None:
        return {"error": "batch lacks uid/traj_uid/turn_step"}
    cand_mask = np.asarray(cand_mask, dtype=bool)
    resp, ids, am = batch.batch["responses"], batch.batch["input_ids"], batch.batch["attention_mask"]
    rlen = resp.shape[1]
    plen = ids.shape[1] - rlen

    plans, per_traj, tasks, traj_uid = {}, {}, {}, {}
    for i in np.flatnonzero(cand_mask):
        i = int(i)
        g = str(uids[i])
        if g not in plans:
            pm = am[i, :plen].bool()
            prompt = tokenizer.decode(ids[i, :plen][pm], skip_special_tokens=False)
            a, b = prompt.find(_PLAN_HEADER), prompt.find(_PLAN_FOOTER)
            plans[g] = (re.findall(r"^\s*\d+\.\s*(.+?)\s*$",
                                   prompt[a:b], flags=re.M)
                        if 0 <= a < b else [])
            _t = re.search(r"Your task is to: (.+?)(?:\n|$)", prompt)
            tasks[g] = _t.group(1).strip() if _t else ""
        plan = plans[g]
        if not plan:
            continue
        rm = am[i, plen:].bool()
        text = tokenizer.decode(ids[i, plen:][rm], skip_special_tokens=False).lower()
        s0, s1 = text.find("<action>"), text.find("</action>")
        act = text[s0 + 8:s1].strip() if 0 <= s0 < s1 else None
        per_traj.setdefault(str(tuids[i]), []).append((int(turns[i]), act, plan))
        traj_uid[str(tuids[i])] = g

    ks, rates, lens, ptrs, records = [], [], [], [], []
    for tr, rows in per_traj.items():
        rows.sort()
        base, plan = rows[0][0], rows[0][2]
        hit = [(act is not None and t - base < len(plan)
                and act == plan[t - base].lower()) for t, act, plan in rows]
        k = 0
        for h in hit:
            if not h:
                break
            k += 1
        # How far down the path it got in the end, allowing detours: the pointer
        # advances only on the step still owed, as walkthrough_stepwise does.
        ptr = 0
        for _, act, _p in rows:
            if act is not None and ptr < len(plan) and act == plan[ptr].lower():
                ptr += 1
        ks.append(k)
        ptrs.append(ptr)
        rates.append(float(np.mean(hit)) if hit else 0.0)
        lens.append(len(plan))
        records.append({"traj": tr, "uid": traj_uid.get(tr), "task": tasks.get(traj_uid.get(tr), "")[:90],
                        "plan_len": len(plan), "leading_k": k, "ptr_final": ptr})
    if not ks:
        return {"error": "no candidate row carries a readable plan block"}
    ks_a = np.asarray(ks, dtype=float)
    return {"trajectories": int(ks_a.size),
            "plan_lines_p50": float(np.percentile(lens, 50)),
            "leading_k_mean": float(ks_a.mean()),
            "leading_k_p50": float(np.percentile(ks_a, 50)),
            "leading_k_max": int(ks_a.max()),
            "leading_k_ge_40": float((ks_a >= 40).mean()),
            "leading_k_hist": {str(int(v)): int(c) for v, c in
                               zip(*np.unique(ks_a, return_counts=True))},
            "match_rate_mean": float(np.mean(rates)),
            "ptr_final_mean": float(np.mean(ptrs)),
            "full_follow_rate": float(np.mean([p >= n for p, n in zip(ptrs, lens)])),
            "records": records}


def _task_kind(task: str) -> str:
    """ALFWorld task type from its task sentence."""
    t = f" {task.lower()} "
    if " two " in t:
        return "pick_two"
    if "clean" in t:
        return "clean"
    if " hot " in t or "heat" in t:
        return "heat"
    if "cool" in t:
        return "cool"
    if "look at" in t or "examine" in t:
        return "look_at"
    return "pick_and_place"


def _aggregate_cand_records(records, status_by_uid, return_by_traj):
    """Rescue by class x plan length and class x task kind.

    ``records`` come from _oci_adherence (one per candidate trajectory); each gains
    its group's class, whether it solved, and its task kind. The whole-path probe's
    followers stopped at four lines -- the entire solution for pick_and_place and
    look_at, half of the rest -- so the rescue rate has to be read separately for
    short and long plans.
    """
    agg = {}
    for r in records:
        r["class"] = status_by_uid.get(str(r.get("uid")))
        r["solved"] = bool(return_by_traj.get(str(r.get("traj")), float("-inf")) > 0.0)
        r["kind"] = _task_kind(r.get("task", ""))
        length = "short_le4" if r["plan_len"] <= 4 else "long_ge5"
        for key in (f'{r["class"]}/{length}', f'{r["class"]}/{r["kind"]}'):
            a = agg.setdefault(key, {"trajectories": 0, "solved": 0, "full_follow": 0,
                                     "_ptr": 0, "_len": 0})
            a["trajectories"] += 1
            a["solved"] += int(r["solved"])
            a["full_follow"] += int(r["ptr_final"] >= r["plan_len"])
            a["_ptr"] += r["ptr_final"]
            a["_len"] += r["plan_len"]
    for a in agg.values():
        count = max(a["trajectories"], 1)
        a["solve_rate"] = a["solved"] / count
        a["full_follow_rate"] = a["full_follow"] / count
        a["ptr_final_mean"] = a.pop("_ptr") / count
        a["plan_len_mean"] = a.pop("_len") / count
    return agg


def _oci_sample_dump(tokenizer, batch, cand_mask, *, n_groups=3, max_chars=2600):
    """Decoded prompt tail and response for a few candidate rows and their
    plain siblings, paired by prompt group and turn.

    Paired on purpose: "the candidate wrote X" says little, "the candidate wrote
    X where its sibling on the same game and the same turn wrote Y" says whether
    the block changed the decision.
    """
    import numpy as np

    uids = batch.non_tensor_batch.get("uid", None)
    tuids = batch.non_tensor_batch.get("traj_uid", None)
    turns = batch.non_tensor_batch.get("turn_step", None)
    if uids is None or tuids is None:
        return {"error": "batch lacks uid/traj_uid"}
    resp = batch.batch["responses"]
    ids, am = batch.batch["input_ids"], batch.batch["attention_mask"]
    rlen = resp.shape[1]
    plen = ids.shape[1] - rlen

    def text(i):
        pm = am[i, :plen].bool()
        rm = am[i, plen:].bool()
        return (tokenizer.decode(ids[i, :plen][pm], skip_special_tokens=False),
                tokenizer.decode(ids[i, plen:][rm], skip_special_tokens=False))

    cand_rows = np.flatnonzero(np.asarray(cand_mask, dtype=bool))
    out, seen = [], set()
    for i in cand_rows:
        g = str(uids[i])
        t = int(turns[i]) if turns is not None else -1
        if (g, t) in seen or len(seen) >= n_groups:
            continue
        # a plain sibling: same group, same turn, not a candidate
        sib = next((j for j in range(len(batch))
                    if str(uids[j]) == g and not cand_mask[j]
                    and (turns is None or int(turns[j]) == t)), None)
        seen.add((g, t))
        cp, cr = text(int(i))
        entry = {"uid": g, "turn": t,
                 "candidate_prompt_tail": cp[-max_chars:],
                 "candidate_response": cr[:max_chars]}
        if sib is not None:
            sp, sr = text(int(sib))
            entry["sibling_prompt_tail"] = sp[-max_chars:]
            entry["sibling_response"] = sr[:max_chars]
        out.append(entry)
    return out


class OPDRayTrainer(RayPPOTrainer):
    """Multitask on-policy distillation trainer with per-task teacher routing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opd_cfg = self.config.algorithm.get("opd", {})
        teacher_paths = opd_cfg.get("teacher_paths", None)
        assert teacher_paths is not None, (
            "OPD requires algorithm.opd.teacher_paths.{alfworld,search,webshop}"
        )
        # Normalize to a plain {task_name: checkpoint_path} dict.
        self.teacher_paths = {
            self._normalize_task_name(task): path for task, path in dict(teacher_paths).items()
        }
        assert None not in self.teacher_paths, "teacher_paths contains an unknown task name"
        # Pure-distillation invariants (also enforced by main_opd config injection).
        assert not self.use_reference_policy, "OPD must not create a reference-policy worker"
        self.teacher_wg = {}
        # Distillation KL mode: "topk_kl" uses dense top-k (+tail) KL; otherwise a
        # single-sampled-token estimator (low_var_kl / kl / ...).
        actor_cfg = self.config.actor_rollout_ref.actor
        self.teacher_topk_kl = actor_cfg.get("teacher_kl_loss_type", "low_var_kl") == "topk_kl"
        self.teacher_kl_topk = int(actor_cfg.get("teacher_kl_topk", 20))
        # Support of the distillation KL: the teacher's top-k (default) or the
        # student's. See actor.student_indexed_topk in ppo_trainer.yaml.
        self.student_indexed_topk = self.teacher_topk_kl and bool(actor_cfg.get("student_indexed_topk", False))
        # Monotone across the run so a stale entry can never be mistaken for a live
        # one; the cache is cleared each step regardless.
        self._teacher_cache_counter = 0
        # Token budget for the frozen forwards that run AFTER the rollout, where
        # vLLM is asleep. The window path keeps ref.log_prob_micro_batch_size_per_gpu
        # (4 rows), which exists to survive being next to the KV pool; out here
        # that bound costs ~3.2k tokens a forward on a 1.7B model and the pass
        # runs launch-bound. 0 restores the row bound on both paths.
        self._post_rollout_token_budget = int(
            os.environ.get(
                "POST_ROLLOUT_FORWARD_TOKENS",
                self.config.actor_rollout_ref.ref.get("log_prob_max_token_len_per_gpu", 0) or 0,
            )
        )
        # The same knob for the IN-WINDOW path, and off by default because the
        # 4-row bound there was bought with an OOM (docs/speedup_mechanisms.md
        # 7.2): a chunk's activations sit next to a live vLLM KV pool.
        #
        # What changed is that the hidden-state cache no longer piles up beside
        # them -- put() copies to host memory as it goes, and teacher_cache/device_gb
        # reads 0.000 on the first run to carry it, against a teacher_cache/gb of
        # ~15. That is 7-8 GB a card the window did not have before.
        #
        # It buys card headroom and it is NOT free elsewhere: those bytes are now
        # the node's. Read perf/pinned_host_gb and the host RAM alongside this one
        # -- the first run to carry the offload was killed by Ray's node-memory
        # threshold, not by a card.
        #
        # It is worth spending because the window path is where the work IS now.
        # Prefetching the sign planes took sign_weight_forward from 143 s to 2.5 s
        # -- and gen grew by very nearly the same amount, because the forwards run
        # in there at the same ~30% MFU that made them 143 s in the first place.
        # Overlapping a launch-bound pass with a decode does not make it stop being
        # launch-bound. Opt-in, so the OOM is a decision and not a surprise.
        self._window_forward_token_budget = int(
            os.environ.get("ROLLOUT_WINDOW_FORWARD_TOKENS", "0")
        )

        # ---- cross-teacher sign agreement (optional) ----------------------- #
        # Off by default so every existing arm is untouched: with enable=false no
        # base worker is built, no extra forward runs, and the loss is the one the
        # arm had before.
        sw_cfg = dict(opd_cfg.get("sign_weight", {}) or {})
        self.sign_weight_enabled = bool(sw_cfg.get("enable", False))
        # The parameter-free arm (verl/trainer/ppo/cross_teacher_kl_weight.py).
        # It reads exactly the same four models on exactly the same support, so
        # it shares every piece of driver plumbing below -- the base worker, the
        # hidden-state cache, the sign_cache_ids columns -- and differs only in
        # what the ACTOR does with them.
        xt_cfg = dict(opd_cfg.get("cross_teacher_kl_weight", {}) or {})
        self.cross_teacher_kl_weight_enabled = bool(xt_cfg.get("enable", False))
        assert not (self.sign_weight_enabled and self.cross_teacher_kl_weight_enabled), (
            "algorithm.opd.sign_weight and algorithm.opd.cross_teacher_kl_weight are two "
            "mechanisms for one signal and both multiply the same teacher KL. Enabling "
            "both would train an arm that is neither and report both sets of metrics as "
            "if they described it; pick one."
        )
        # The TARGET arm (verl/trainer/ppo/cross_teacher_target.py). The third
        # consumer the seam below was written for: same four models, same
        # support, same cache. It differs from the two above in WHERE it reaches
        # the loss -- it rewrites the teacher's values instead of scaling the KL,
        # because a positive scalar on KL(p_s || p_on) is minimised at
        # p_s = p_on whatever the scalar is.
        xtt_cfg = dict(opd_cfg.get("cross_teacher_target", {}) or {})
        self.cross_teacher_target_enabled = bool(xtt_cfg.get("enable", False))
        # WHICH of the two target mechanisms, and -- for the staged one -- the
        # schedule, validated here rather than at first use. The curriculum is
        # a claim about the ORDER components are taught in, so a schedule that
        # does not fit inside the run is not a smaller version of the experiment:
        # a run whose stage 2 never ends never tests stage 3, and one whose
        # release finishes at step 1 is the control with extra logging.
        self.cross_teacher_target_mode = str(xtt_cfg.get("mode", "tilt"))
        self.cross_teacher_curriculum = None
        if self.cross_teacher_target_enabled and self.cross_teacher_target_mode == "curriculum":
            from verl.trainer.ppo.cross_teacher_target import curriculum_rho

            stage_steps = tuple(int(x) for x in xtt_cfg.get("stage_steps", (40, 80)))
            ramp_steps = int(xtt_cfg.get("ramp_steps", 10))
            assert len(stage_steps) == 2, (
                "cross_teacher_target.stage_steps is (last step of stage 1, last step of "
                f"stage 2); got {stage_steps}"
            )
            # Raises on an overlapping ramp, which is the one way to get a run
            # with no stage 2 at all.
            curriculum_rho(step=1, stage_steps=stage_steps, ramp_steps=ramp_steps)
            total = int(self.config.trainer.get("total_training_steps", 0) or 0)
            assert total <= 0 or stage_steps[1] + ramp_steps < total, (
                f"the curriculum finishes releasing at step {stage_steps[1] + ramp_steps} "
                f"but the run is {total} steps: there would be no fully-released stage, "
                "and the prediction the arm is judged on (the endpoint agrees with the "
                "control) is about that stage"
            )
            self.cross_teacher_curriculum = {
                "stage_steps": stage_steps, "ramp_steps": ramp_steps,
            }
        assert not (self.cross_teacher_target_enabled
                    and (self.sign_weight_enabled or self.cross_teacher_kl_weight_enabled)), (
            "algorithm.opd.cross_teacher_target moves the distillation target while "
            "sign_weight and cross_teacher_kl_weight scale it. Two of them at once "
            "trains an arm that is none of the three; pick one."
        )
        # Who needs base, the cache and the extra forwards -- as opposed to who
        # builds a weight out of them. Every gate below is this one, so adding a
        # third consumer never means finding the cache gates again.
        # The stand-alone transfer ladder (docs/privileged_multitask_notice_design.md
        # section 4): the four-model cache with NO weighting attached, so an arm
        # that leaves the loss alone can still report transfer/off_travel. A
        # fourth consumer of the same seam, and nothing more.
        ladder_cfg = dict(opd_cfg.get("transfer_ladder", {}) or {})
        self.transfer_ladder_enabled = bool(ladder_cfg.get("enable", False))
        self.cross_teacher_enabled = (
            self.sign_weight_enabled or self.cross_teacher_kl_weight_enabled
            or self.cross_teacher_target_enabled or self.transfer_ladder_enabled
        )
        # The three that WEIGHT or MOVE the target, as opposed to the ladder,
        # which only reads. On any shared setting the ladder yields to them.
        self.weighted_arm_enabled = (
            self.sign_weight_enabled or self.cross_teacher_kl_weight_enabled
            or self.cross_teacher_target_enabled
        )
        # Who needs the hidden-state cache, which is NOT the same question as who
        # picks the top-k support. student_indexed_topk needs it because the
        # on-task teacher is scored at ids that do not exist until the actor's
        # forward; sign weighting needs it because base and the off-task teachers
        # are, whichever model chose those ids. Gating the cache on the first
        # alone left a teacher-indexed weighted arm with no output projections
        # registered, no cache cleared between steps, and no witness -- while the
        # driver still ran the three extra forwards that fill it.
        self.need_hidden_cache = self.student_indexed_topk or self.cross_teacher_enabled
        self.sign_weight_mode = str(sw_cfg.get("mode", "target"))
        self.base_policy_path = (
            xtt_cfg.get("base_path", None) if self.cross_teacher_target_enabled
            else xt_cfg.get("base_path", None) if self.cross_teacher_kl_weight_enabled
            else sw_cfg.get("base_path", None) if self.weighted_arm_enabled
            else ladder_cfg.get("base_path", None)
        )
        if self.transfer_ladder_enabled and not self.weighted_arm_enabled:
            # Same three structural needs as the weighted arms -- a shared top-k
            # support, the exact base, two off-task teachers -- for the same reason.
            check_cross_teacher_kl_weight_prerequisites(
                teacher_topk_kl=self.teacher_topk_kl,
                base_policy_path=self.base_policy_path,
                n_teachers=len(self.teacher_paths),
            )
        # The privileged multitask notice. Parsed once; the text hashes are
        # checked against the lock's pins HERE, before any model loads, because
        # the text is the mechanism and a drift is a different experiment.
        self.privileged_notice = parse_notice_config(opd_cfg.get("privileged_notice", None))
        self._teacher_notice_prefix = None
        if self.privileged_notice is not None:
            notice_verify_doc_hashes(self.privileged_notice)
            if self.privileged_notice.to_teacher:
                # Teacher mode: the block's token ids per task, from the tokenizer's
                # own chat template and checked to be an exact prefix (system_block
                # asserts). Computed once, used on every on-task call.
                kw = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
                self._teacher_notice_prefix = {
                    task: notice_prefix_ids(self.tokenizer, self.privileged_notice.variant, task, kw)
                    for task in NOTICE_TASKS
                }
            print(
                f"[privileged_notice] variant={self.privileged_notice.variant} "
                f"apply_to={sorted(self.privileged_notice.apply_to)}"
                + (f" teacher prefix tokens={ {t: len(v) for t, v in self._teacher_notice_prefix.items()} }"
                   if self._teacher_notice_prefix else ""),
                flush=True,
            )
        self.base_wg = None
        if self.sign_weight_enabled:
            check_sign_weight_prerequisites(
                mode=self.sign_weight_mode,
                teacher_topk_kl=self.teacher_topk_kl,
                base_policy_path=self.base_policy_path,
                n_teachers=len(self.teacher_paths),
            )
        if self.cross_teacher_kl_weight_enabled:
            check_cross_teacher_kl_weight_prerequisites(
                teacher_topk_kl=self.teacher_topk_kl,
                base_policy_path=self.base_policy_path,
                n_teachers=len(self.teacher_paths),
            )
        if self.cross_teacher_target_enabled:
            # The same three prerequisites, and for the same reasons: a dense
            # top-k support to rewrite, a base policy to measure shifts against,
            # and two off-task teachers -- with one the sign agreement that gates
            # both channels is trivially satisfied by a single teacher agreeing
            # with itself, and the mechanism degenerates to pairwise.
            check_cross_teacher_kl_weight_prerequisites(
                teacher_topk_kl=self.teacher_topk_kl,
                base_policy_path=self.base_policy_path,
                n_teachers=len(self.teacher_paths),
            )

    # ------------------------------------------------------------------ #
    # Worker setup: actor_rollout (+ optional critic/rm) + N teachers.
    # ------------------------------------------------------------------ #
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # actor + rollout (hybrid engine)
        if not self.hybrid_engine:
            raise NotImplementedError
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # One teacher worker group per task, each with its own checkpoint.
        teacher_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        self._teacher_keys = {}
        for task, path in self.teacher_paths.items():
            teacher_cfg = copy.deepcopy(self.config.actor_rollout_ref)
            with open_dict(teacher_cfg):
                teacher_cfg.model.path = path
                # Avoid the LoRA branch in compute_ref_log_prob; teachers are full models.
                teacher_cfg.model.lora_rank = 0
            key = f"teacher_{task}"
            self._teacher_keys[task] = key
            teacher_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=teacher_cfg,
                role="ref",
            )
            self.resource_pool_to_cls[teacher_pool][key] = teacher_cls

        # The base policy the teachers were fine-tuned from. Built exactly like a
        # teacher (same role="ref") but kept out of self.teacher_paths /
        # self.teacher_wg, because those are keyed by task and drive the routing: a
        # fourth entry there would have to survive _normalize_task_name and would
        # then be looked up for rows that do not exist.
        if self.cross_teacher_enabled:
            base_cfg = copy.deepcopy(self.config.actor_rollout_ref)
            with open_dict(base_cfg):
                base_cfg.model.path = self.base_policy_path
                base_cfg.model.lora_rank = 0
            self.resource_pool_to_cls[teacher_pool]["base_policy"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=base_cfg,
                role="ref",
            )

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg = {}
        wg_kwargs = {}
        from omegaconf import OmegaConf

        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        # Initialize teachers before the rollout engine (matches actor-last ordering).
        # The base policy takes a slot in the same stacked projection as the
        # teachers, so the count has to include it before the first registration:
        # the stack is allocated once, at n_tasks * vocab, by whoever registers
        # first.
        n_teachers = len(self._teacher_keys) + (1 if self.cross_teacher_enabled else 0)
        for slot, (task, key) in enumerate(self._teacher_keys.items()):
            wg = all_wg[key]
            wg.init_model()
            self.teacher_wg[task] = wg
            if self.need_hidden_cache and not bool(self.config.trainer.get("val_only", False)):
                # The actor resolves this teacher at ids nobody has picked yet --
                # the student's top-k, or the on-task teacher's own on a
                # teacher-indexed weighted arm, where this teacher is off-task for
                # two thirds of the rows. Either way it needs its output projection
                # at update time -- by which point the ref path has resharded it. One unsharded copy per teacher, taken
                # once here, labelled with the task the cache will file it under.
                # Not in a val_only run: nothing scores a teacher there, and this
                # is 1.9 GB a rank held for the life of the process, next to a vLLM
                # engine already sized to 0.6 of the card.
                #
                # The slot is passed so the copy lands directly in its slice of the
                # stacked projection the lookup reads, instead of being cloned on
                # its own and stacked later -- that held both layouts at once, and
                # peaked here, before vLLM measures free memory.
                wg.register_teacher_lm_head(task, slot=slot, n_tasks=n_teachers)

        if self.cross_teacher_enabled:
            self.base_wg = all_wg["base_policy"]
            self.base_wg.init_model()
            if self.need_hidden_cache and not bool(self.config.trainer.get("val_only", False)):
                # Filed under a label that is deliberately not a task name: the
                # cache picks an output projection by this string, and the routing
                # picks a teacher by task name. A collision would silently score
                # rows against the wrong model.
                self.base_wg.register_teacher_lm_head(
                    SIGN_BASE_TASK, slot=n_teachers - 1, n_tasks=n_teachers
                )

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # rollout is created last so vLLM gets a better kv-cache estimate.
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        """Same as the base trainer, except when ENV_RESET_PREFETCH has peeked
        one dataloader batch ahead this step. The peeked batch has only had its
        env_kwargs used for the background env reset — it is trained on the
        *next* step — so the checkpoint must record the pre-peek dataloader
        position; saving the live (post-peek) state would make a resumed run
        skip that batch entirely.
        """
        pre_peek_state = getattr(self, "_pre_peek_dataloader_state", None)
        if pre_peek_state is None:
            return super()._save_checkpoint()
        # Shadow the bound state_dict with the pre-peek snapshot for the
        # duration of the base save (which calls train_dataloader.state_dict()).
        self.train_dataloader.state_dict = lambda: pre_peek_state
        try:
            return super()._save_checkpoint()
        finally:
            del self.train_dataloader.state_dict

    # ------------------------------------------------------------------ #
    # Per-task teacher routing.
    # ------------------------------------------------------------------ #
    def _teacher_prefetch_chunk(self, chunk):
        """Score a chunk of already-finished trajectory rows with their teachers.

        Handed to ``multi_turn_loop`` as its ``teacher_prefetch_fn`` and called on
        a background thread while the driver is doing CPU work between
        generations (see ROLLOUT_PREFETCH_TEACHER). The teachers are frozen for
        the whole run, so a row's targets do not depend on *when* it is scored --
        only the micro-batch it lands in differs from the post-rollout path, which
        moves the last bits of a packed GEMM and nothing else.

        THE BASE POLICY AND THE OFF-TASK TEACHERS RIDE ALONG. That sentence about
        frozen weights is not about the on-task teacher: it is about every model
        this arm reads, and the other three are frozen in exactly the same sense.
        Scoring only the on-task one here left :meth:`compute_sign_weight_cache`
        running three more forwards AFTER the rollout, in a phase of its own --
        measured at 169.5 s a step against 4.5 s for the prefetched on-task pass,
        which is the same work per row done in the window instead of outside it.
        The window has room: the rollout's glue was measured ~34% busy at
        hit_rate 0.99, with 21.6 s/step of tchWait spill.

        Four calls per chunk instead of one, in the same background thread and
        therefore one at a time -- the peak is one chunk's activations either way,
        which is the bound _SIGN_WEIGHT_FORWARD_CHUNK exists to hold on the serial
        path. The adaptive sizer needs no change: it measures rows/second from
        completed chunks, so four models per row simply reads as a slower rate and
        the next chunk comes out smaller.

        Args:
            chunk: list of ``((traj_uid, turn_step), row_dict)`` as queued by the
                rollout loop. Rows carry the four model-input tensors and their
                ``task_name``.

        Returns:
            ``{(traj_uid, turn_step): value}``. ``value`` is what
            ``compute_teacher_log_probs`` writes -- a ``(resp, k)`` log-prob/id
            pair under top-k, else a ``(resp,)`` log-prob row -- wrapped in a
            :data:`PrefetchedRow` beside the sign-plane keys when this arm caches
            them, which :meth:`compute_sign_weight_cache` unwraps.
        """
        by_task = {}
        for key, row in chunk:
            task = self._normalize_task_name(row.get("task_name"))
            by_task.setdefault(task, []).append((key, row))

        # student_indexed_topk resolves this teacher at ids the student has not
        # chosen yet, from hidden states the teacher keeps on whichever rank scored
        # the row. That rank is never the one that later trains the row -- the batch
        # is regrouped by task, padded and then reordered by _balance_batch -- so the
        # cache is addressed by an id assigned here and carried on the row.
        cache_id_for = {}
        if self.student_indexed_topk:
            for key, _ in chunk:
                self._teacher_cache_counter += 1
                cache_id_for[key] = self._teacher_cache_counter

        out = {}
        for task, entries in by_task.items():
            wg = self.teacher_wg.get(task)
            if wg is None:
                # Unknown task: leave it queued-but-unscored. compute_teacher_log_probs
                # raises on it with the full list of configured teachers, which is a
                # better error than one from a background thread.
                continue
            sub = DataProto.from_dict(
                tensors={
                    name: torch.stack([row[name] for _, row in entries])
                    for name in ("input_ids", "attention_mask", "position_ids", "responses")
                }
            )
            # Teacher-mode notice rides the prefetch too: the on-task teacher is
            # scored here for most rows, and a notice missing from this path would
            # make the target depend on WHEN a row was scored.
            sub = self._with_teacher_notice(sub, task)
            ids = (
                torch.tensor([cache_id_for[key] for key, _ in entries], dtype=torch.long)
                if self.student_indexed_topk
                else None
            )
            if self.teacher_topk_kl and self.student_indexed_topk:
                # Nothing comes back but the row count: the support is the
                # student's, so the values are resolved in the actor from the
                # hidden states this call just cached. What used to travel here
                # was (rows, 512, 20) log-probs plus the same in int64 ids --
                # ~860 MB a step, merged into the batch and never read.
                self._teacher_call(wg, sub, topk=True, cache_ids=ids,
                                   budget=self._window_forward_token_budget)
                for key, _ in entries:
                    out[key] = cache_id_for[key]
            elif self.teacher_topk_kl:
                scored = self._teacher_call(wg, sub, topk=True, cache_ids=ids,
                                            budget=self._window_forward_token_budget)
                tlp = scored.batch["teacher_topk_logprobs"]
                tid = scored.batch["teacher_topk_ids"]
                for j, (key, _) in enumerate(entries):
                    out[key] = (tlp[j], tid[j], cache_id_for.get(key, -1))
            else:
                scored = self._teacher_call(wg, sub, topk=False, cache_ids=ids,
                                            budget=self._window_forward_token_budget)
                lp = scored.batch["ref_log_prob"]
                for j, (key, _) in enumerate(entries):
                    out[key] = lp[j]

        sign_ids = self._prefetch_sign_planes(chunk)
        if sign_ids is not None:
            for key in list(out):
                out[key] = PrefetchedRow(on_task=out[key], sign_ids=sign_ids.get(key))
        return out

    def _prefetch_sign_planes(self, chunk):
        """Cache the base policy and each row's off-task teachers on this chunk.

        The three passes :meth:`compute_sign_weight_cache` would otherwise run
        after the rollout, run here instead. Same models, same rows, same
        per-row keys; only the window changes, and the models are frozen, so the
        values cannot.

        Returns ``{key: [base_id, off_id_0, ...]}`` in the SAME column order
        :meth:`compute_sign_weight_cache` uses -- column 0 the base policy, then
        the row's off-task teachers in sorted task order. That layout is a
        function of the row's own task and nothing else, which is what lets the
        actor read the columns positionally; deriving it twice from one rule is
        cheaper than shipping it, but only while the two rules stay one
        expression. Both call :meth:`_sign_off_tasks_for`.

        ``None`` when this arm does not cache the planes at all, which leaves the
        returned rows exactly as they were before this path existed -- and SAYS SO
        ONCE. A run measured at ``sign_prefetch/hit_rate`` 0.000 for 148 steps is
        indistinguishable, from the metric alone, between "an operator turned it
        off" and "every row was declined because its task name did not match a
        teacher". Those want opposite fixes and the metric named neither, so the
        reason is printed the first time this returns None.
        """
        if not self.cross_teacher_enabled:
            return self._decline_sign_prefetch("this arm reads no cross-teacher planes")
        if not _ROLLOUT_PREFETCH_SIGN:
            return self._decline_sign_prefetch(
                f"ROLLOUT_PREFETCH_SIGN={os.environ.get('ROLLOUT_PREFETCH_SIGN', '<unset>')!r} in the "
                "process running the rollout loop (the driver, not the workers)"
            )
        task_order = sorted(self.teacher_wg.keys())
        rows = [(key, row, self._normalize_task_name(row.get("task_name"))) for key, row in chunk]
        # A row whose task has no teacher is left for the serial path, which
        # raises on it by name rather than from a background thread.
        kept = [r for r in rows if r[2] in self.teacher_wg]
        if not kept:
            seen = sorted({r[2] for r in rows})
            return self._decline_sign_prefetch(
                f"no row in this chunk names a task with a teacher: rows carry {seen!r}, "
                f"teachers are {sorted(self.teacher_wg)!r}"
            )
        rows = kept

        out = {key: [-1] * (1 + max(0, len(task_order) - 1)) for key, _, _ in rows}

        def _cache(wg, entries, column_for):
            if not entries:
                return
            sub = DataProto.from_dict(
                tensors={
                    name: torch.stack([row[name] for _, row, _ in entries])
                    for name in ("input_ids", "attention_mask", "position_ids", "responses")
                }
            )
            ids = torch.empty(len(entries), dtype=torch.long)
            for j, (key, _, own) in enumerate(entries):
                self._teacher_cache_counter += 1
                ids[j] = self._teacher_cache_counter
                out[key][column_for(own)] = self._teacher_cache_counter
            self._teacher_call(wg, sub, topk=True, cache_ids=ids,
                               budget=self._window_forward_token_budget)

        gpu_profiler.push_phase("sign_weight_prefetch/base")
        try:
            _cache(self.base_wg, rows, lambda own: 0)
        finally:
            gpu_profiler.pop_phase("sign_weight_prefetch/base")

        for model in task_order:
            # Every row the model is NOT the on-task teacher for, which is what
            # "off-task" means and the only rows its plane is read on.
            entries = [r for r in rows if r[2] != model]
            gpu_profiler.push_phase(f"sign_weight_prefetch/{model}")
            try:
                _cache(
                    self.teacher_wg[model],
                    entries,
                    lambda own, m=model: 1 + self._sign_off_tasks_for(own, task_order).index(m),
                )
            finally:
                gpu_profiler.pop_phase(f"sign_weight_prefetch/{model}")
        return out

    def _decline_sign_prefetch(self, reason):
        """Log why the sign planes are not being prefetched, once, and return None.

        Once per process rather than per chunk: this is called from the prefetch
        thread on every turn of every step, and the interesting fact is the reason,
        which does not change. ``sign_prefetch/declined`` carries it into the
        metrics as a flag so a run that never prefetched is visible in wandb
        without reading the log.
        """
        self._sign_prefetch_declined = reason
        if not getattr(self, "_said_sign_prefetch_declined", False):
            self._said_sign_prefetch_declined = True
            print(
                f"[rollout][sign-prefetch] NOT caching the base and off-task planes in the "
                f"rollout window: {reason}. They will be scored after the rollout in "
                f"sign_weight_forward instead -- measured at 143 s a step against ~4.5 s "
                f"for the same work done in the window.",
                flush=True,
            )
        return None

    @staticmethod
    def _sign_off_tasks_for(own, task_order):
        """The off-task teachers of a row whose own task is ``own``, in column order.

        One expression, read by both the prefetch and the post-rollout pass, so
        the columns they write cannot drift apart. The actor reads them
        positionally and has no way to notice if they did.
        """
        return [t for t in task_order if t != own]

    def _teacher_call(self, wg, sub: DataProto, topk: bool, cache_ids=None, budget=0):
        """One teacher call, with the DP padding marked so it is never cached.

        ``budget`` sizes the worker's micro-batches by TOKENS rather than by rows.
        A number and not a flag, because the two callers do not want the same one:
        after the rollout vLLM is asleep and the card is nearly empty, while inside
        the window a chunk's activations sit beside a live KV pool. 0 keeps the
        worker's own row bound, which is what the window path used until the
        hidden-state cache stopped accumulating on the device.

        Not bit-identical -- a packed GEMM of a different total length rounds
        differently -- but the same rows, the same frozen weights and the same
        function, which is the accuracy class the prefetch path already has.

        ``auto_padding`` repeats rows to reach a multiple of the group's world
        size, and it repeats the whole row -- ``teacher_cache_ids`` included. Two
        ranks would then cache the same id and the exchange would see a row
        answered twice. Padding explicitly instead lets the copies carry -1: they
        are still scored (the shapes have to match) and still discarded on the way
        out, they just do not enter any cache.
        """
        pad_size = 0
        if cache_ids is not None:
            sub = sub.__class__.from_dict(
                tensors={**{k: v for k, v in sub.batch.items()}, "teacher_cache_ids": cache_ids}
            )
            sub, pad_size = pad_dataproto_to_divisor(sub, wg.world_size)
            if pad_size:
                sub.batch["teacher_cache_ids"][-pad_size:] = -1
        else:
            sub.meta_info = dict(sub.meta_info)
            sub.meta_info[DataProtoConfig.auto_padding_key] = True
        if budget and int(budget) > 0:
            sub.meta_info = dict(sub.meta_info)
            sub.meta_info["use_dynamic_bsz"] = True
            sub.meta_info["max_token_len"] = int(budget)
        if topk:
            sub.meta_info = dict(sub.meta_info)
            sub.meta_info["topk_k"] = self.teacher_kl_topk
            out = wg.compute_ref_topk_log_prob(sub)
        else:
            out = wg.compute_ref_log_prob(sub)
        if pad_size:
            out = unpad_dataproto(out, pad_size=pad_size)
        return out

    def _prefetched_teacher_rows(self, batch: DataProto):
        """Row index -> prefetch key, for the rows a prefetch could have covered.

        ``None`` when the batch carries no trajectory identity (validation, or a
        recipe that does not record turn_step), which disables the merge.
        """
        traj_uid = batch.non_tensor_batch.get("traj_uid", None)
        turn_step = batch.non_tensor_batch.get("turn_step", None)
        if traj_uid is None or turn_step is None:
            return None
        # adjust_batch's duplicates share their original's key, so two rows can map
        # to the same entry -- reading it twice is what makes them duplicates.
        return {i: (str(traj_uid[i]), int(turn_step[i])) for i in range(len(batch))}

    def _with_teacher_notice(self, sub: DataProto, task: str) -> DataProto:
        """Teacher-mode notice: prepend the task's system block to THIS teacher's input.

        Only the on-task teacher's calls come through here. The student's own rows
        are untouched -- the rollout is the control's -- so what moves is the KL
        target alone. Rows are re-left-padded to one width so the response window,
        selected from the end on the worker, is where it always was, and
        ``fingerprint_adjust`` tells the worker how much the prefix added to the
        row fingerprint so the cache entry is filed under the STUDENT's row.
        """
        prefix_by_task = getattr(self, "_teacher_notice_prefix", None)
        if prefix_by_task is None:
            return sub
        task = notice_normalize_task(task)
        if task not in prefix_by_task:
            return sub
        pre = prefix_by_task[task]
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids, mask, pos = notice_prepend_prefix(
            sub.batch["input_ids"], sub.batch["attention_mask"], [pre] * len(sub), pad_token_id=pad,
        )
        tensors = {k: v for k, v in sub.batch.items()}
        tensors.update({
            "input_ids": ids, "attention_mask": mask, "position_ids": pos,
            "fingerprint_adjust": torch.full((len(sub),), notice_fingerprint_adjust(pre), dtype=torch.long),
        })
        out = DataProto.from_dict(tensors=tensors, non_tensors=dict(sub.non_tensor_batch))
        out.meta_info = dict(sub.meta_info)
        return out

    def _notice_metrics(self, batch: DataProto) -> dict:
        """Design section 4, diagnostic 2 and the truncation floor. Per task: the
        share of responses carrying ANOTHER task's action syntax, and the share of
        rows whose prompt hit the cap with the notice at its head."""
        if getattr(self, "privileged_notice", None) is None:
            return {}
        out = {}
        task_names = batch.non_tensor_batch.get("task_name", None)
        if task_names is None:
            return out
        texts = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        by_task = {}
        for i, t in enumerate(task_names):
            nt = notice_normalize_task(t)
            if nt in NOTICE_TASKS:
                by_task.setdefault(nt, []).append(i)
        n_all = leak_all = 0
        for task, idxs in by_task.items():
            flags = notice_leak_flags([texts[i] for i in idxs], task)
            out[f"notice/leak_rate/{task}"] = sum(flags) / len(flags)
            n_all += len(flags); leak_all += sum(flags)
        if n_all:
            out["notice/leak_rate"] = leak_all / n_all
        if "notice_truncated" in batch.batch.keys():
            tr = batch.batch["notice_truncated"].float()
            nl = batch.batch["notice_len"].float()
            out["notice/truncated_frac"] = float(tr.mean())
            out["notice/prefix_tokens_mean"] = float(nl.mean())
            for task, idxs in by_task.items():
                sel = torch.tensor(idxs, dtype=torch.long)
                out[f"notice/truncated_frac/{task}"] = float(tr[sel].mean())
                out[f"notice/prefix_tokens/{task}"] = float(nl[sel].mean())
        return out

    def compute_teacher_log_probs(self, batch: DataProto, prefetched=None, metrics=None) -> None:
        """Route each sample to its task's teacher and write the distillation
        signal into ``batch.batch`` in original order.

        The student's exact (prompt, response) is fed to the teacher — no skill
        prepend. The teacher call is ``DP_COMPUTE_PROTO``-dispatched; per-task
        slices are auto-padded to the DP world size and unpadded on return.

        - default (single-token estimator): sets ``teacher_log_probs`` (bs, resp).
        - top-k mode: sets ``teacher_topk_logprobs`` and ``teacher_topk_ids``
          (bs, resp, k), the teacher's top-k log-softmax and ids per token.

        ``prefetched`` holds rows already scored during the rollout (see
        ``_teacher_prefetch_chunk``); those are filled in here and excluded from
        the per-task calls below, so each row is scored exactly once either way.
        """
        task_names = batch.non_tensor_batch.get("task_name", None)
        assert task_names is not None, "OPD requires task_name on every sample for teacher routing"
        normalized = [self._normalize_task_name(t) for t in task_names]

        bs = batch.batch["responses"].size(0)
        resp_len = batch.batch["responses"].size(1)
        seen = [False] * bs

        # Under student_indexed_topk the teacher's own top-k is not part of the
        # answer -- the support comes from the student and the values are resolved
        # in the actor -- so these two (bs, response_length, k) columns are not
        # built, not merged and not shipped. At this batch size they were ~860 MB
        # a step of pure transport.
        merge_topk = self.teacher_topk_kl and not self.student_indexed_topk
        if merge_topk:
            k = self.teacher_kl_topk
            teacher_topk_logprobs = torch.zeros((bs, resp_len, k), dtype=torch.float32)
            teacher_topk_ids = torch.zeros((bs, resp_len, k), dtype=torch.long)
        elif not self.teacher_topk_kl:
            teacher_log_probs = torch.zeros((bs, resp_len), dtype=torch.float32)

        # Every row gets a key, not just the prefetched ones. The rows the prefetch
        # missed -- the final turns, ~1% at hit_rate 0.99 -- are scored below, and
        # they need their hidden states cached exactly like the rest. Leaving them
        # at -1 makes the exchange return a zero teacher log-prob, which is not a
        # missing target but a WRONG one: exp(0)=1 at every id drives the tail mass
        # negative and the KL through the clamp.
        cache_ids = torch.full((bs,), -1, dtype=torch.long)
        keys = self._prefetched_teacher_rows(batch) if prefetched else None
        if keys is not None:
            for i, key in keys.items():
                hit = prefetched.get(key)
                if hit is None:
                    continue
                # The sign planes ride in the same entry; this path wants only
                # the on-task half. compute_sign_weight_cache reads the other.
                if isinstance(hit, PrefetchedRow):
                    hit = hit.on_task
                if self.student_indexed_topk:
                    cache_ids[i] = hit
                elif self.teacher_topk_kl:
                    if len(hit) == 3:
                        teacher_topk_logprobs[i], teacher_topk_ids[i], cache_ids[i] = hit
                    else:
                        teacher_topk_logprobs[i], teacher_topk_ids[i] = hit
                else:
                    teacher_log_probs[i] = hit
                seen[i] = True
        if metrics is not None:
            n_hit = sum(seen)
            metrics["teacher_prefetch/rows"] = n_hit
            metrics["teacher_prefetch/hit_rate"] = n_hit / bs if bs else 0.0

        for task, wg in self.teacher_wg.items():
            idxs = [i for i, t in enumerate(normalized) if t == task and not seen[i]]
            if not idxs:
                continue
            sub = batch.select_idxs(idxs)
            sub = self._with_teacher_notice(sub, task)
            miss_ids = None
            if self.student_indexed_topk:
                miss_ids = torch.empty(len(idxs), dtype=torch.long)
                for j, i in enumerate(idxs):
                    self._teacher_cache_counter += 1
                    miss_ids[j] = self._teacher_cache_counter
                    cache_ids[i] = self._teacher_cache_counter
            # The teachers run one after another in this loop, so "teacher_forward"
            # as a single phase says how long all three took together but not which
            # one dominates -- and they are not interchangeable: the tasks differ in
            # prompt length (webshop's mean prompt is ~3x alfworld's), and under
            # topk_kl each teacher ships back (rows, resp_len, k) log-probs and ids
            # instead of one value per token. Tagging per task splits both the
            # compute and that transfer out by teacher.
            gpu_profiler.push_phase(f"teacher_forward/{task}")
            try:
                if self.teacher_topk_kl and self.student_indexed_topk:
                    self._teacher_call(wg, sub, topk=True, cache_ids=miss_ids,
                                       budget=self._post_rollout_token_budget)
                    for i in idxs:
                        seen[i] = True
                elif self.teacher_topk_kl:
                    out = self._teacher_call(wg, sub, topk=True, cache_ids=miss_ids,
                                             budget=self._post_rollout_token_budget)
                    tlp = out.batch["teacher_topk_logprobs"]
                    tid = out.batch["teacher_topk_ids"]
                    for j, i in enumerate(idxs):
                        teacher_topk_logprobs[i] = tlp[j]
                        teacher_topk_ids[i] = tid[j]
                        seen[i] = True
                else:
                    out = self._teacher_call(wg, sub, topk=False, cache_ids=miss_ids,
                                             budget=self._post_rollout_token_budget)
                    lp = out.batch["ref_log_prob"]
                    for j, i in enumerate(idxs):
                        teacher_log_probs[i] = lp[j]
                        seen[i] = True
            finally:
                gpu_profiler.pop_phase(f"teacher_forward/{task}")

        if not all(seen):
            missing = sorted({normalized[i] for i in range(bs) if not seen[i]})
            raise ValueError(
                f"No teacher configured for task_name(s) {missing}; "
                f"available teachers: {sorted(self.teacher_wg.keys())}"
            )

        if merge_topk:
            batch.batch["teacher_topk_logprobs"] = teacher_topk_logprobs
            batch.batch["teacher_topk_ids"] = teacher_topk_ids
        elif self.student_indexed_topk:
            batch.batch["teacher_cache_ids"] = cache_ids
        else:
            batch.batch["teacher_log_probs"] = teacher_log_probs

    # ------------------------------------------------------------------ #
    # Cross-teacher sign agreement (optional; see sign_weights.py).
    # ------------------------------------------------------------------ #
    def compute_sign_weight_cache(self, batch: DataProto, prefetched=None, metrics=None) -> None:
        """Cache the base policy and every off-task teacher on the rows they have
        to speak for, so the actor can read them at ids the student picks.

        The weights need four models on one support, and the support is chosen
        inside the training forward. Only the final gather depends on the ids, so
        the same trick the on-task teacher already uses applies unchanged: each
        model's hidden states and full-vocabulary normaliser are cached here, and
        ``log p(v) = h . W[v] - lse`` is finished in the actor.

        Cost is three extra frozen forwards a step -- the base over every row, and
        each teacher over the 2/3 of rows that are NOT its own task. The on-task
        pass already ran in :meth:`compute_teacher_log_probs` and is reused
        through the same cache rather than repeated.

        ``prefetched`` holds rows whose three planes were already cached inside
        the rollout window (see :meth:`_prefetch_sign_planes`); those columns are
        filled in from it and excluded from the passes below, so each row is
        scored exactly once by each model either way -- the same arrangement the
        on-task teacher has had. What is left here is the misses, which at the
        hit rates the on-task path sees is a small tail rather than the batch.

        Writes two columns:

        ``sign_cache_ids``  (bs, 1 + n_off) int64. Column 0 is the base policy;
            columns 1.. are the row's off-task teachers in sorted task order. The
            actor reads them positionally, so the layout has to be a function of
            the row's own task and nothing else.
        ``sign_off_tasks``  (bs, n_off) int64, the task id behind each of those
            columns, in the same numbering as ``task_ids``. Only the diagnostics
            need it -- the weights themselves do not care which teacher is which
            -- but the pairwise agreement rates are the cheapest form of the
            transferability matrix and they cannot be built without it.
        """
        if not self.cross_teacher_enabled:
            return

        task_names = batch.non_tensor_batch.get("task_name", None)
        assert task_names is not None, "sign weighting requires task_name for on/off-task routing"
        normalized = [self._normalize_task_name(t) for t in task_names]
        bs = len(normalized)
        task_order = sorted(self.teacher_wg.keys())
        n_off = len(task_order) - 1

        # Same numbering the actor sees on task_ids, taken from the column rather
        # than rebuilt: _attach_task_ids numbers the tasks PRESENT in the batch, so
        # deriving it here from the teacher list would drift the moment a task is
        # missing from a step.
        id_names = batch.meta_info.get("task_id_names", None)
        assert id_names is not None, "sign weighting reads task_id_names; call _attach_task_ids first"
        task_id_of = {name: i for i, name in enumerate(id_names)}

        column_of = {}
        for own in task_order:
            for c, other in enumerate(self._sign_off_tasks_for(own, task_order)):
                column_of[(own, other)] = 1 + c

        sign_cache_ids = torch.full((bs, 1 + n_off), -1, dtype=torch.long)
        off_tasks = torch.full((bs, n_off), -1, dtype=torch.long)
        for i, own in enumerate(normalized):
            for c, other in enumerate(self._sign_off_tasks_for(own, task_order)):
                off_tasks[i, c] = task_id_of.get(other, -1)

        # Rows the rollout window already covered. A row is filled by all four
        # models in one chunk or by none of them, but the columns are checked
        # one at a time anyway: the passes below select on the column they are
        # about to write, so a half-filled row would still come out complete
        # rather than silently keep a -1 the actor would read as an unanswered
        # key.
        keys = self._prefetched_teacher_rows(batch) if prefetched else None
        for i, key in (keys or {}).items():
            hit = prefetched.get(key)
            ids = hit.sign_ids if isinstance(hit, PrefetchedRow) else None
            if not ids:
                continue
            for c, cid in enumerate(ids[: 1 + n_off]):
                if cid >= 0:
                    sign_cache_ids[i, c] = cid
        if metrics is not None and bs:
            n_hit = int((sign_cache_ids >= 0).all(dim=1).sum())
            metrics["sign_prefetch/rows"] = n_hit
            metrics["sign_prefetch/hit_rate"] = n_hit / bs
            # Which of the two zero-hit-rate stories this run is living. 1 means
            # the window path was asked for and answered; 0 means it declined,
            # and _decline_sign_prefetch printed why.
            metrics["sign_prefetch/enabled"] = float(
                self.cross_teacher_enabled
                and _ROLLOUT_PREFETCH_SIGN
                and getattr(self, "_sign_prefetch_declined", None) is None
            )

        # Only what the forward reads. The batch at this point also carries the
        # rollout's own columns, and every one of them would be shipped to the
        # worker and padded with it.
        lean = batch.select(
            batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[],
        )

        def _cache(wg, idxs, column_for):
            # In row chunks, NOT one call per model. compute_topk_log_prob keeps
            # every micro-batch's hidden states in a list and concatenates them
            # before the cache packs anything, so one call over the whole batch
            # builds a (rows_per_rank, response_length, hidden) tensor -- and,
            # during the concat, two of them. At this batch size that is tens of
            # GB on a card that has just finished a rollout, and it is what OOMed
            # the first run of this arm. The teacher prefetch never hit it because
            # it scores at most a few hundred rows per call; this is the same
            # bound, applied to the passes that run after the rollout.
            #
            # The chunk changes nothing a value depends on: the forward is per
            # row, each row gets its own key either way, and the chunk is a
            # multiple of the DP world size so micro-batches keep their shape.
            for start in range(0, len(idxs), _SIGN_WEIGHT_FORWARD_CHUNK):
                part = idxs[start : start + _SIGN_WEIGHT_FORWARD_CHUNK]
                ids = torch.empty(len(part), dtype=torch.long)
                for j, i in enumerate(part):
                    self._teacher_cache_counter += 1
                    ids[j] = self._teacher_cache_counter
                    sign_cache_ids[i, column_for(i)] = self._teacher_cache_counter
                self._teacher_call(wg, lean.select_idxs(part), topk=True, cache_ids=ids,
                                   budget=self._post_rollout_token_budget)

        gpu_profiler.push_phase("sign_weight_forward/base")
        try:
            _cache(self.base_wg, [i for i in range(bs) if sign_cache_ids[i, 0] < 0], lambda i: 0)
        finally:
            gpu_profiler.pop_phase("sign_weight_forward/base")

        for task in task_order:
            # Off-task rows this model has not already answered for. Selecting on
            # the column rather than on a "was this row prefetched" flag keeps the
            # two paths independent: a chunk that failed on the driver and was
            # dropped leaves its rows here, exactly as the on-task path handles
            # the same failure.
            idxs = [
                i for i, t in enumerate(normalized)
                if t != task and sign_cache_ids[i, column_of[(t, task)]] < 0
            ]
            if not idxs:
                continue
            gpu_profiler.push_phase(f"sign_weight_forward/{task}")
            try:
                _cache(self.teacher_wg[task], idxs, lambda i, m=task: column_of[(normalized[i], m)])
            finally:
                gpu_profiler.pop_phase(f"sign_weight_forward/{task}")

        assert bool((sign_cache_ids >= 0).all()), "a row was left without one of its four models"
        batch.batch["sign_cache_ids"] = sign_cache_ids
        batch.batch["sign_off_tasks"] = off_tasks

    def _dump_sign_token_report(self, actor_output) -> None:
        """Write the step's per-token sign-weight table.

        The scalar metrics say how CONCENTRATED the weighting is; this says on
        WHAT. Neither substitutes for the other, and only one of them fits in a
        wandb column, so the table goes to disk beside the run.

        One file per step rather than one appended file: a resumed run re-writes
        the steps it repeats instead of appending a second copy of them, which is
        the difference between a reader taking a groupby and a reader having to
        work out which duplicate to trust.
        """
        dump_dir = self.config.trainer.get("sign_token_dump_dir", None)
        if not dump_dir:
            return
        # One file per table. They are keyed differently -- scope/state against
        # dst/src/class -- so a merged file would give every row the other
        # table's empty columns and make a groupby depend on which is which.
        for key, stem in (
            ("sign_token_report", "sign_tokens"),
            ("sign_pair_token_report", "sign_pair_tokens"),
            ("sign_event_report", "sign_events"),
            ("sign_pair_event_report", "sign_pair_events"),
        ):
            rows = actor_output.meta_info.get(key, None)
            if not rows:
                continue
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"{stem}_step{self.global_steps:06d}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps({"step": self.global_steps, **row}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Thin training loop: rollout -> teacher_log_probs -> update_actor.
    # No old_log_prob / ref / values / advantage / reward-in-loss -- those are
    # what the two hooks below restore for the OPD+GRPO subclass.
    #
    # Hooks the OPD+GRPO arm overrides. They exist so the two arms share one
    # fit(): everything the loop does around them -- the hidden-state cache,
    # the sign-weight pass, env-reset prefetch, stop_after_steps -- is the part
    # that must stay identical between the arms, and a second copy of it is
    # exactly how "the arms differ only in the objective" stops being true.
    # ------------------------------------------------------------------ #
    progress_desc = "OPD Training"

    def _accumulate_oci_probe(self, batch, state: dict, probe_cfg) -> dict:
        """One rollout batch: the class split, the injection count, and rho.

        THE THREE THINGS THAT HAVE TO BE TRUE BEFORE OCI-sat TRAINS ANYTHING,
        and none of them is assumed anywhere else:

        1. *Where the degenerate token mass actually is.* The group count and the
           token count disagree by roughly 4x, because a saturated group finishes
           early and a stuck one runs to the turn cap. Only the token split bounds
           what the saturated arm can buy, and only ``compute_group_metrics``
           labels the classes.
        2. *That the corrupted plan actually makes the student fail.* The switch
           does not validate itself. ``cand_fail_rate`` is that check: a plan
           missing its requirement step that the student solves anyway leaves the
           group saturated and injects nothing.
        3. *That the failure is REACHABLE.* A row injected into a saturated group
           carries a large negative advantage, but what reaches the weights under
           shaping is ``A * gamma * rho / (rho + gamma)^2``, which is zero at
           rho = 0. A failure the student would only produce with the plan in
           front of it carries no gradient however large its advantage.

        No optimizer step, no generation, one extra forward pass of weights the
        actor already holds over tokens that already exist.
        """
        import numpy as np

        from verl.trainer.ppo.metric_utils import compute_group_metrics
        from verl.trainer.ppo.oci_reachability import (
            reachability_report, strippable_rows, wrong_plan_strip_fn)
        from verl.trainer.ppo.oci_saturated import (
            classify_groups, injection_metrics, select_saturated_injections,
            token_mass_by_class)

        n = state["batches"] + 1
        task_id_names = list(batch.meta_info.get("task_id_names", []) or [])
        multi_turn = bool(self.config.actor_rollout_ref.rollout.multi_turn.enable)

        rec = {"batch": n}
        rec["groups"] = compute_group_metrics(batch, with_records=True)

        # The ten-slot layout marks its rows by role, and its special rows carry
        # oci_candidate=1 as well (the rollout sets it for every row it can
        # re-score), so the role has to be read first or the single-candidate
        # reader below would take document and foreign rows for one candidate.
        _role_col = batch.batch.get("oci_role", None)
        _slots_active = _role_col is not None and bool((_role_col.reshape(-1) != 0).any())
        # RANK-ONLY: with the layout off there are no special rows and no single
        # candidate either -- every group is eight ordinary rollouts, which is
        # exactly what the rank scorers must be checked on.
        if (not _slots_active
                and bool((self.config.algorithm.get("oci_rank", None) or {}).get("enable", False))):
            import json as _json
            try:
                rec["rank"] = self._oci_rank_report(batch, probe_cfg)
            except Exception as exc:
                rec["rank"] = {"error": f"{type(exc).__name__}: {exc}"}
            _brief = {k: v for k, v in rec["rank"].items() if k != "records"}
            print(f"[grad_probe] rank batch {n}: {_json.dumps(_brief, default=float)}", flush=True)
            state.setdefault("oci", []).append(rec)
            state["batches"] = n
            return state
        if _slots_active:
            return self._accumulate_slots_probe(batch, state, probe_cfg, rec, n)

        cand = batch.batch.get("oci_candidate", None)
        if cand is None:
            # The column is emitted by the rollout loop, not derived here -- see
            # the no-fallback note in opd_grpo_ray_trainer. Say so rather than
            # reporting a reachability of nothing.
            rec["error"] = ("no oci_candidate column: the rollout must be built "
                            "with algorithm.oci_sat.enable=True")
            state.setdefault("oci", []).append(rec)
            state["batches"] = n
            return state

        cand_np = cand.reshape(-1).detach().cpu().numpy().astype(bool)
        grp = classify_groups(batch, judged_rows=~cand_np)
        injected = select_saturated_injections(batch, grp, candidate_rows=cand_np)
        drop = cand_np & ~injected
        rec["injection"] = injection_metrics(grp, injected, task="alfworld")
        rec["token_mass"] = token_mass_by_class(
            batch, grp, multi_turn=multi_turn, exclude_rows=drop)

        # Did the corrupted plan do its job? Per trajectory, not per row.
        rets = batch.batch["token_level_rewards"].sum(-1).detach().float().cpu().numpy()
        tuids = batch.non_tensor_batch.get("traj_uid", None)
        if tuids is not None:
            best = {}
            for i in np.flatnonzero(cand_np):
                k = str(tuids[i])
                best[k] = max(best.get(k, float("-inf")), float(rets[i]))
            if best:
                vals = np.array(list(best.values()), dtype=float)
                rec["cand_trajectories"] = int(vals.size)
                rec["cand_fail_rate"] = float((vals <= 0.0).mean())
                rec["cand_return_mean"] = float(vals.mean())

            # DID THE EIGHTH ROLLOUT SOLVE THE GAMES THE SEVEN COULD NOT? Split by
            # the class the seven plain rollouts put the group in. cand_fail_rate
            # pools every group, and a pooled rate cannot say whether a document
            # RESCUES a stuck group -- the one question the 7+1+1 design turns on,
            # because an injected rollout on a stuck group helps only if it
            # succeeds. Saturated and live groups are reported beside it so the
            # rescue rate can be read against how often the document helps where
            # the student already succeeds.
            _by = {c: {"groups": 0, "cand_solved": 0, "plain_success": 0.0}
                   for c in ("stuck", "live", "saturated")}
            for _uid, _g in grp.items():
                _st = _g.get("status")
                if _st not in _by:
                    continue
                _rows = _g.get("rows") or []
                _cand = [i for i in _rows if cand_np[i]]
                _plain = [i for i in _rows if not cand_np[i]]
                if not _cand:
                    continue
                _ctraj, _ptraj = {}, {}
                for i in _cand:
                    k = str(tuids[i])
                    _ctraj[k] = max(_ctraj.get(k, float("-inf")), float(rets[i]))
                for i in _plain:
                    k = str(tuids[i])
                    _ptraj[k] = max(_ptraj.get(k, float("-inf")), float(rets[i]))
                _by[_st]["groups"] += 1
                _by[_st]["cand_solved"] += int(max(_ctraj.values()) > 0.0)
                if _ptraj:
                    _by[_st]["plain_success"] += float(np.mean([v > 0.0 for v in _ptraj.values()]))
            for _c, _v in _by.items():
                n_g = max(_v["groups"], 1)
                _v["cand_solve_rate"] = _v["cand_solved"] / n_g
                _v["plain_success_rate"] = _v.pop("plain_success") / n_g
            rec["cand_by_class"] = _by

        # rho, on the candidate rows whose plan span can be removed EXACTLY. A
        # row that cannot be stripped is reported, not approximated: the whole
        # quantity is a ratio between two conditionings of the same weights, and
        # an approximate denominator makes it a ratio between two fictions.
        strip_ok = strippable_rows(batch)
        rows = np.flatnonzero(cand_np & strip_ok)
        rec["plan_span"] = {
            "candidate_rows": int(cand_np.sum()),
            "strippable_rows": int(len(rows)),
            "unstrippable_rows": int((cand_np & ~strip_ok).sum()),
        }
        _len = batch.batch.get("oci_plan_len", None)
        if _len is not None and len(rows):
            pl = _len.reshape(-1).detach().cpu().numpy()[rows]
            rec["plan_span"].update({
                "tokens_p50": float(np.percentile(pl, 50)),
                "tokens_max": int(pl.max()),
            })
        _mode = str((self.config.algorithm.get("oci_sat", {}) or {}).get("plan_corruption", ""))
        if _mode.endswith("_stepwise"):
            rec["reachability"] = {
                "skipped": "walkthrough_stepwise puts a progress line OUTSIDE the stripped "
                           "span, so the 'plain' prompt would still carry privileged text "
                           "and rho would compare privileged with privileged"}
        elif not len(rows):
            rec["reachability"] = {
                "error": "no candidate row carries a strippable plan span; is "
                         "algorithm.oci_sat.enable reaching the alfworld manager, "
                         "and did the prompt avoid truncation?"
            }
        else:
            try:
                _pad = (self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None else 0)
                strip = wrong_plan_strip_fn(self.tokenizer, _pad)
                sub = batch[rows.tolist()]
                rec["reachability"] = reachability_report(
                    self.actor_rollout_wg, sub, task_id_names, strip_fn=strip,
                    gamma=float(probe_cfg.get("gamma", 0.1)),
                )
            except Exception as exc:  # the report is the primary result; keep it
                rec["reachability"] = {"error": f"{type(exc).__name__}: {exc}"}

        # WHAT THE STUDENT ACTUALLY WROTE. Every number above is an aggregate,
        # and four probes were read without once looking at a generation. The
        # candidate's own text beside a PLAIN sibling from the SAME group -- same
        # game, same turn, one shown the block and one not -- is the only thing
        # that says whether the block is being followed, argued with, or ignored.
        # A few rows, decoded on the driver, no extra pass.
        # HOW FAR DOWN THE BLOCK IT WALKED. Everything above is an outcome; this
        # is the behaviour that produces it, and the delay mode lives or dies on
        # it -- see _oci_adherence.
        try:
            rec["adherence"] = _oci_adherence(self.tokenizer, batch, cand_np)
        except Exception as exc:
            rec["adherence"] = {"error": f"{type(exc).__name__}: {exc}"}

        # RESCUE BY PLAN LENGTH -- see _aggregate_cand_records. It is a module-level
        # function on purpose: the first version was written inline here, used `n`
        # as a loop variable, and so overwrote this method's `n` -- the batch
        # counter that `state["batches"] = n` stores below. The counter sat at the
        # last bucket's trajectory count, never reached n_batches, and the probe
        # ran on indefinitely (12 batches reported as "batch 1").
        try:
            _recs = (rec.get("adherence") or {}).get("records") or []
            if _recs and tuids is not None:
                _status = {str(u): g.get("status") for u, g in grp.items()}
                _ret = {}
                for i in np.flatnonzero(cand_np):
                    _k = str(tuids[i])
                    _ret[_k] = max(_ret.get(_k, float("-inf")), float(rets[i]))
                rec["cand_by_class_and_length"] = _aggregate_cand_records(_recs, _status, _ret)
        except Exception as exc:
            rec["cand_by_class_and_length"] = {"error": f"{type(exc).__name__}: {exc}"}

        try:
            rec["samples"] = _oci_sample_dump(
                self.tokenizer, batch, cand_np,
                n_groups=int(probe_cfg.get("dump_groups", 3)))
        except Exception as exc:
            rec["samples"] = {"error": f"{type(exc).__name__}: {exc}"}

        import json

        print(f"[grad_probe] oci batch {n}: "
              f"{json.dumps({k: v for k, v in rec.items() if k != 'groups'}, default=float)}",
              flush=True)
        state.setdefault("oci", []).append(rec)
        state["batches"] = n
        return state

    def _accumulate_slots_probe(self, batch, state: dict, probe_cfg, rec: dict, n: int) -> dict:
        """The ten-slot layout's probe: what each special slot did, and its rho.

        Runs on the batch AFTER ``_select_oci_slots`` dropped the two unused rows
        per group, so the rows here are exactly the ones a training step would
        put through the loss. Two things per role:

        1. Whether the slot did its job, from the selection's own counts
           (``document_used`` of ``groups_stuck``, ``foreign_used`` of
           ``groups_saturated``) -- the generation-time numbers the drop hides.
        2. Whether its tokens are REACHABLE from the plain prompt: rho per token,
           re-scored with the privileged edit spliced out. Split by role, because
           the training run pooled document and foreign rows into one statistic
           and the pooled median (-30) turned out to be the foreign rows alone.
        """
        import numpy as np

        from agent_system.environments.oci_layout import ROLE_DOC, ROLE_FOREIGN
        from verl.trainer.ppo.oci_reachability import (
            reachability_report, strippable_rows, wrong_plan_strip_fn)

        task_id_names = list(batch.meta_info.get("task_id_names", []) or [])
        role = batch.batch["oci_role"].reshape(-1).detach().cpu().numpy().astype(int)
        inj_col = batch.batch.get("oci_injected", None)
        inj = (inj_col.reshape(-1).detach().cpu().numpy().astype(bool)
               if inj_col is not None else np.zeros(role.shape[0], dtype=bool))
        rets = batch.batch["token_level_rewards"].sum(-1).detach().float().cpu().numpy()
        tuids = batch.non_tensor_batch.get("traj_uid", None)
        strip_ok = strippable_rows(batch)

        rec["layout"] = "oci_slots"
        rec["selection"] = dict(getattr(self, "_oci_slots_last_metrics", None) or {})
        _pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        strip = wrong_plan_strip_fn(self.tokenizer, _pad)

        rec["roles"] = {}
        for r, name in ((ROLE_DOC, "document"), (ROLE_FOREIGN, "foreign")):
            kept = np.flatnonzero((role == r) & inj)
            entry = {"rows_kept": int(kept.size),
                     "strippable_rows": int(strip_ok[kept].sum()) if kept.size else 0}
            if tuids is not None and kept.size:
                best = {}
                for i in kept:
                    k = str(tuids[i])
                    best[k] = max(best.get(k, float("-inf")), float(rets[i]))
                vals = np.array(list(best.values()), dtype=float)
                entry["trajectories_kept"] = int(vals.size)
                entry["kept_fail_rate"] = float((vals <= 0.0).mean())
                entry["kept_return_mean"] = float(vals.mean())
            rows = np.flatnonzero((role == r) & inj & strip_ok)
            if rows.size:
                try:
                    entry["reachability"] = reachability_report(
                        self.actor_rollout_wg, batch[rows.tolist()], task_id_names,
                        strip_fn=strip, gamma=float(probe_cfg.get("gamma", 0.1)))
                except Exception as exc:
                    entry["reachability"] = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                entry["reachability"] = {"skipped": "no strippable row of this role was kept"}
            try:
                entry["samples"] = _oci_sample_dump(
                    self.tokenizer, batch, (role == r) & inj,
                    n_groups=int(probe_cfg.get("dump_groups", 3)))
            except Exception as exc:
                entry["samples"] = {"error": f"{type(exc).__name__}: {exc}"}
            rec["roles"][name] = entry

        if bool((self.config.algorithm.get("oci_rank", None) or {}).get("enable", False)):
            try:
                rec["rank"] = self._oci_rank_report(batch, probe_cfg)
            except Exception as exc:
                rec["rank"] = {"error": f"{type(exc).__name__}: {exc}"}

        import json

        print(f"[grad_probe] slots batch {n}: "
              f"{json.dumps({k: v for k, v in rec.items() if k != 'groups'}, default=float)}",
              flush=True)
        state.setdefault("oci", []).append(rec)
        state["batches"] = n
        return state

    def _oci_rank_report(self, batch: DataProto, probe_cfg=None) -> dict:
        """Three scorers on the same rows, and four checks of whether any of them
        knows which trajectory is better.

        SCORERS (all score the student's OWN sampled tokens; one forward each,
        no backward, nothing generated):
          plain       the actor on the prompt it had -- the control
          privileged  the actor with the instance's document in front
          teacher     the task's teacher checkpoint on the plain prompt
        plus ``privileged_gain`` and ``teacher_gain`` (mean per-token log q -
        log pi against ``plain``), the form most self-distillation work scores.

        CHECKS (see verl/trainer/ppo/oci_rank.py): live-group AUC; the same
        after removing a within-group fit on turn count (in alfworld a loss IS a
        run to the cap, so the raw AUC mostly measures length); accuracy at the
        first turn where a live group's actions diverge (same prompt, no length
        confound); and inside stuck / saturated groups, the order against
        walkthrough progress and turn count.

        ONLY REAL ORDINARY ROWS. Rows of the ten-slot layout's document and
        foreign slots are excluded -- the first run of this probe counted a
        saturated group with a foreign row as "live" and got AUC 0.985 from rows
        whose prompt was different -- and so are adjust_batch's padding copies.
        A trajectory is scored only if every one of its real rows could be
        conditioned, so each is scored on all of its tokens by all three scorers;
        a copy never disqualifies the trajectory it duplicates (the first version
        let it, and dropped 44% of the losses). A group's class is read from ALL
        its real trajectories, scored or not.

        SELF-CHECKS (``algorithm.oci_rank.self_check_rows`` > 0): a null edit
        must reproduce the plain score, and the document prompt tokenized from
        its own text must reproduce the privileged one -- see
        ``_oci_rank_self_check``.
        """
        import numpy as np

        from agent_system.environments.oci_layout import ROLE_DOC, ROLE_FOREIGN
        from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY
        from verl.trainer.ppo.oci_rank import (build_trajectories, document_rows,
                                               group_status_from_rows, parse_actions,
                                               rank_report, row_sums, select_scored_rows,
                                               with_document)

        cfg = self.config.algorithm.get("oci_rank", None) or {}
        rec = {"rows": int(len(batch))}
        nt = batch.non_tensor_batch
        candidate = np.ones(len(batch), dtype=bool)
        role = batch.batch.get("oci_role", None)
        if role is not None:
            r = role.reshape(-1).detach().cpu().numpy()
            candidate &= ~np.isin(r, (ROLE_DOC, ROLE_FOREIGN))
        pad_col = batch.batch.get(PADDING_ROW_KEY, None)
        if pad_col is not None:
            pad = pad_col.reshape(-1).to(torch.bool).cpu().numpy()
            rec["rows_padding_copies"] = int(pad.sum())
            candidate &= ~pad
        has_doc = document_rows(batch)
        rec["rows_real_ordinary"] = int(candidate.sum())
        rec["rows_ordinary_with_document"] = int((candidate & has_doc).sum())
        if not (candidate & has_doc).any():
            rec["error"] = ("no ordinary row carries a document edit; is algorithm.oci_rank.enable "
                            "reaching the env manager, and does the task have a document?")
            return rec

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        doc_batch, spliced = with_document(batch, pad_id)
        ok = has_doc & np.asarray(spliced, dtype=bool)
        rec["rows_not_spliced"] = int((candidate & has_doc & ~ok).sum())
        tuids = np.asarray([str(t) for t in nt["traj_uid"]])
        score_mask, dropped = select_scored_rows(tuids, candidate, ok)
        idx = np.flatnonzero(score_mask)
        rec["trajectories_real"] = int(len({t for t, c in zip(tuids, candidate) if c}))
        rec["rows_scored"] = int(idx.size)
        rec["trajectories_dropped_incomplete"] = len(dropped)
        if not idx.size:
            rec["error"] = "no complete trajectory could be conditioned on its document"
            return rec
        rows = idx.tolist()
        rets = nt.get("episode_rewards", None)
        all_returns = (np.asarray([float(x) for x in rets]) if rets is not None
                       else batch.batch["token_level_rewards"].sum(-1).detach().float().cpu().numpy())
        status = group_status_from_rows([str(u) for u in nt["uid"]], tuids, all_returns, candidate)

        mask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() else None
        if mask is None:
            from verl.trainer.ppo.ray_trainer import compute_response_mask
            mask = compute_response_mask(batch)
        mask = mask[rows]

        sums, counts = {}, None
        try:
            lp = self._padded_log_prob(self.actor_rollout_wg, batch[rows])
            sums["plain"], counts = row_sums(lp, mask)
        except Exception as exc:
            rec["plain_error"] = f"{type(exc).__name__}: {exc}"
        try:
            lp = self._padded_log_prob(self.actor_rollout_wg, doc_batch[rows])
            sums["privileged"], c = row_sums(lp, mask)
            counts = c if counts is None else counts
        except Exception as exc:
            rec["privileged_error"] = f"{type(exc).__name__}: {exc}"
        try:
            task_names = [self._normalize_task_name(t) for t in nt["task_name"]]
            arr = np.full(idx.size, np.nan)
            for task, wg in self.teacher_wg.items():
                pos = [j for j, i in enumerate(rows) if task_names[i] == task]
                if not pos:
                    continue
                lp = self._padded_log_prob(wg, batch[[rows[j] for j in pos]], ref=True)
                sm, _ = row_sums(lp, mask[pos])
                arr[pos] = sm
            if not np.isnan(arr).any():
                sums["teacher"] = arr
        except Exception as exc:
            rec["teacher_error"] = f"{type(exc).__name__}: {exc}"
        if not sums:
            return rec

        returns = all_returns[idx]
        actions = parse_actions(self.tokenizer, batch.batch["responses"][rows], mask)
        gamefiles = nt.get("gamefile", None)
        recs = build_trajectories(
            uids=[str(nt["uid"][i]) for i in rows], tuids=tuids[idx],
            turn_steps=[int(nt["turn_step"][i]) for i in rows], returns=returns,
            tasks=[str(nt["task_name"][i]) for i in rows],
            gamefiles=None if gamefiles is None else [str(gamefiles[i]) for i in rows],
            actions=actions, sums=sums, counts=counts)

        walk_of = None
        if "alfworld" in tuple(cfg.get("tasks", ["alfworld"]) or ["alfworld"]):
            from agent_system.environments.env_manager import _tw_pddl

            def walk_of(gf):
                try:
                    return _tw_pddl(gf)[0] if gf else []
                except Exception:
                    return []
        rec.update(rank_report(recs, sums, counts, walk_of=walk_of,
                               n_boot=int((probe_cfg or {}).get("rank_bootstrap", 1000)),
                               status=status))
        n_check = int(cfg.get("self_check_rows", 0) or 0)
        if n_check > 0 and "plain" in sums and "privileged" in sums:
            try:
                rec["self_check"] = self._oci_rank_self_check(
                    batch, doc_batch, rows, mask, sums, counts, n_check, pad_id)
            except Exception as exc:
                rec["self_check"] = {"error": f"{type(exc).__name__}: {exc}"}
        return rec

    def _oci_rank_self_check(self, batch: DataProto, doc_batch: DataProto, rows, mask,
                             sums, counts, n_check: int, pad_id: int) -> dict:
        """Two independent re-scorings that a wrong privileged score cannot pass.

        null_edit      every document edit replaced by one that changes nothing,
                       spliced and widened exactly as far as the document batch
                       was: the prompts must come back identical and the scores
                       must equal ``plain``. Tests the splice, the widening and
                       the position ids with the document taken out of the
                       question.
        direct_render  the document prompt the rollout rendered, tokenized from
                       its TEXT and laid out the way the rollout lays out any
                       prompt: the tokens must equal the spliced prompt's and
                       the scores must equal ``privileged``. No recorded edit is
                       involved.
        Both on the same ``n_check`` scored rows, drawn with a fixed seed, and
        each compared with its reference RE-SCORED ON THOSE SAME ROWS, so a
        different micro-batch composition is not mistaken for a difference;
        ``noise_floor`` is that re-scoring against the full-batch score. The same
        weights on the same tokens agree to numerical noise; the effect being
        measured is tenths of a nat per token.
        """
        import numpy as np

        from verl.trainer.ppo.oci_rank import (compare_scores, direct_render, null_edit,
                                               prompt_tokens, row_sums, with_document)

        pick = np.sort(np.random.default_rng(0).choice(len(rows), size=min(int(n_check), len(rows)),
                                                       replace=False))
        sub = [rows[int(j)] for j in pick]
        pick_t = torch.as_tensor(pick, dtype=torch.long)
        out = {"rows": int(len(sub))}
        grow = int(doc_batch.batch["input_ids"].shape[1]) - int(batch.batch["input_ids"].shape[1])

        lp = self._padded_log_prob(self.actor_rollout_wg, batch[sub])
        s_plain, _ = row_sums(lp, mask[pick_t])
        lp = self._padded_log_prob(self.actor_rollout_wg, doc_batch[sub])
        s_priv, _ = row_sums(lp, mask[pick_t])
        out["noise_floor"] = {
            "plain": compare_scores(s_plain, sums["plain"][pick], counts[pick]),
            "privileged": compare_scores(s_priv, sums["privileged"][pick], counts[pick])}

        null_b, null_spliced = with_document(null_edit(batch), pad_id, min_grow=grow)
        same = [a == b for a, b in zip(prompt_tokens(null_b, sub), prompt_tokens(batch, sub))]
        lp = self._padded_log_prob(self.actor_rollout_wg, null_b[sub])
        s_null, _ = row_sums(lp, mask[pick_t])
        out["null_edit"] = dict(compare_scores(s_null, s_plain, counts[pick]),
                                rows_spliced=int(np.asarray(null_spliced, dtype=bool)[sub].sum()),
                                rows_prompt_identical=int(sum(same)), widened_by=grow)

        texts = batch.non_tensor_batch.get("oci_doc_prompt", None)
        if texts is None:
            out["direct_render"] = {"skipped": "no oci_doc_prompt column: the rollout stores it only "
                                               "when algorithm.oci_rank.self_check_rows > 0"}
            return out
        width = int(doc_batch.batch["input_ids"].shape[1]) - int(doc_batch.batch["responses"].shape[1])
        direct_b, too_long = direct_render(doc_batch[sub], [str(texts[i]) for i in sub],
                                           self.tokenizer, pad_id, width)
        keep = [j for j in range(len(sub)) if j not in set(too_long)]
        same = [a == b for a, b in zip(prompt_tokens(direct_b, keep),
                                       prompt_tokens(doc_batch, [sub[j] for j in keep]))]
        lp = self._padded_log_prob(self.actor_rollout_wg, direct_b)
        s_dir, _ = row_sums(lp, mask[pick_t])
        k = np.asarray(keep, dtype=int)
        out["direct_render"] = dict(
            compare_scores(s_dir[k], s_priv[k], counts[pick][k]),
            rows_too_long=len(too_long), rows_prompt_identical=int(sum(same)))
        return out

    def _padded_log_prob(self, wg, sub: DataProto, ref: bool = False):
        """``compute_log_prob`` on a worker group, padded to its world size.

        ``DataProto.chunk`` asserts the row count divides the data-parallel world
        exactly, and these sub-batches are arbitrary -- one trajectory per group,
        as many rows as it ran turns. The same padding dance the reachability
        report does, for the same reason.
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        world = int(getattr(wg, "world_size", 1) or 1)
        pad_size = 0
        if world > 1 and (len(sub) % world):
            sub, pad_size = pad_dataproto_to_divisor(sub, world)
        out = wg.compute_ref_log_prob(sub) if ref else wg.compute_log_prob(sub)
        if pad_size:
            out = unpad_dataproto(out, pad_size=pad_size)
        for key in ("ref_log_prob", "old_log_probs", "log_probs"):
            if key in out.batch.keys():
                return out.batch[key]
        raise KeyError(f"no log-prob column in {sorted(out.batch.keys())}")

    def _select_oci_slots(self, batch: DataProto, metrics: dict) -> DataProto:
        """Keep the eight rollouts each group trains and drop the other two.

        A no-op unless ``algorithm.oci_slots.enable``. The rule, and why the drop
        belongs at this point in the loop, are in verl/trainer/ppo/oci_slots.py.
        """
        cfg = self.config.algorithm.get("oci_slots", None)
        if cfg is None or not bool(cfg.get("enable", False)):
            return batch

        from verl.trainer.ppo.oci_slots import (apply_selection, check_config,
                                                select_rollouts)

        check_config(self.config)
        keep, injected, slot_metrics = select_rollouts(
            batch,
            tasks=list(cfg.get("tasks", ["alfworld"]) or ["alfworld"]),
            group_n=int(self.config.env.rollout.n),
        )
        metrics.update(slot_metrics)
        # Kept for the probe, which sees the batch only after this drop and so
        # cannot recount what was generated.
        self._oci_slots_last_metrics = dict(slot_metrics)
        # On the console as well as in the step's metrics: the metrics are logged
        # when the step completes, and the first thing a step can die of is a
        # stage AFTER this one -- which then leaves no record of what the
        # selection did with the rollout that was just paid for.
        print("[oci_slots] " + " ".join(
            f"{k.split('/', 1)[1]}={v:.3g}" if isinstance(v, float) else f"{k.split('/', 1)[1]}={v}"
            for k, v in sorted(slot_metrics.items())), flush=True)
        return apply_selection(batch, keep, injected)

    def _reward_and_advantage(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Turn the env reward into whatever this arm feeds the loss.

        Pure OPD computes it for MONITORING ONLY: it is never turned into
        advantages and never enters the loss, so a reward-manager failure must
        not take the run down with it.

        Returns ``(batch, reward_extra_infos_dict)`` -- the batch is returned
        rather than mutated because the GRPO override rebinds it (``union`` and
        ``compute_advantage`` both return new objects).
        """
        reward_extra_infos_dict = {}
        with _timer("reward", timing_raw):
            try:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            except Exception as e:  # monitoring must never break training
                print(f"[OPD] reward computation skipped: {e}")
        return batch, reward_extra_infos_dict

    def _data_metrics(self, batch: DataProto) -> dict:
        """Batch statistics for the step. Advantage-free on this arm."""
        metrics = compute_opd_data_metrics(batch=batch)
        metrics.update(compute_opd_data_metrics_by_task(batch=batch))
        return metrics

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._fast_forward_env_schedules()

        # val_only is checked on its own, not nested under val_before_train. The
        # run scripts end with trainer.val_before_train=False (the initial eval
        # costs a full validation pass and says nothing a resumed run does not
        # already know), and with the check nested a "validate this checkpoint and
        # stop" command skipped the block entirely and fell through to TRAINING
        # from the checkpoint -- which looks like a working run right up until the
        # numbers that were asked for never appear.
        val_only = bool(self.config.trainer.get("val_only", False))
        if self.val_reward_fn is not None and (val_only or self.config.trainer.get("val_before_train", True)):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
        if val_only:
            assert self.val_reward_fn is not None, "trainer.val_only=True but no validation reward fn is configured"
            return

        # Exit cleanly after this many steps, so a run can pause at a mid-point
        # checkpoint and be resumed later by the same command. This is NOT the
        # way to run a shorter experiment -- total_training_steps stays what the
        # lock pins, because it also sets the LR schedule (warmup is 10% of
        # total): a run launched with total=150 would put a different LR
        # trajectory into steps 15-30 than the 300-step control had. Stopping
        # here instead leaves schedule, data order and objective identical to a
        # straight run interrupted by a crash, which the resume path already
        # handles exactly.
        stop_after = int(self.config.trainer.get("stop_after_steps", 0) or 0)
        if stop_after and 0 < self.config.trainer.save_freq:
            assert stop_after % self.config.trainer.save_freq == 0, (
                f"trainer.stop_after_steps={stop_after} is not a checkpoint step "
                f"(save_freq={self.config.trainer.save_freq}); stopping there would "
                "discard the tail since the last save"
            )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc=self.progress_desc)
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            batch_iter = iter(self.train_dataloader)
            peeked_batch_dict = None
            while True:
                if peeked_batch_dict is not None:
                    batch_dict = peeked_batch_dict
                    peeked_batch_dict = None
                else:
                    batch_dict = next(batch_iter, None)
                    if batch_dict is None:
                        break
                # Reset the pre-peek dataloader snapshot each step; it is set
                # again below if this step peeks ahead (see _save_checkpoint).
                self._pre_peek_dataloader_state = None
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                if "task_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("task_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    if self.need_hidden_cache:
                        # Drop last step's hidden states before the teachers start
                        # filling the cache again. Ids are monotone across the run,
                        # so a leftover entry could never be mistaken for a live one
                        # -- this is about memory, not correctness. One call: the
                        # cache is per PROCESS and the three teachers are colocated,
                        # so all three share it.
                        next(iter(self.teacher_wg.values())).clear_teacher_hidden_cache()
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                            # Score finished trajectories under the rollout's own
                            # CPU glue instead of after it; a no-op unless
                            # ROLLOUT_PREFETCH_TEACHER is on.
                            teacher_prefetch_fn=self._teacher_prefetch_chunk,
                        )

                    # The train envs are idle from here until the next rollout;
                    # kick off their reset for the next step in a background
                    # thread so it overlaps the GPU training phases below.
                    if (
                        _ENV_RESET_PREFETCH
                        and not is_last_step
                        and not self.config.algorithm.filter_groups.enable
                    ):
                        # Snapshot the dataloader state before peeking: the peeked
                        # batch is trained on the NEXT step, so a checkpoint saved
                        # this step must record the pre-peek position or a resumed
                        # run would skip that batch (see _save_checkpoint).
                        if hasattr(self.train_dataloader, "state_dict"):
                            self._pre_peek_dataloader_state = self.train_dataloader.state_dict()
                        peeked_batch_dict = next(batch_iter, None)
                        if peeked_batch_dict is not None and "env_kwargs" in peeked_batch_dict:
                            # Same repeat the next multi_turn_loop applies to its
                            # gen_batch (repeat(n, interleave=True) on non-tensors
                            # is an element-wise np.repeat).
                            next_env_kwargs = np.repeat(
                                peeked_batch_dict["env_kwargs"], self.config.env.rollout.n
                            )
                            self.traj_collector.prefetch_env_reset(self.envs, next_env_kwargs)

                    del batch
                    batch = gen_batch_output

                    # TEN GENERATED, EIGHT TRAINED (algorithm.oci_slots), and the
                    # two that are not trained leave HERE -- before anything starts
                    # counting rows. adjust_batch pads to a divisor,
                    # attach_task_loss_weights divides each task's share by its
                    # token count, _balance_batch splits by tokens and
                    # update_policy cuts mini-batches by row count, so a batch
                    # still carrying them would take about a quarter more
                    # optimizer steps per training step than control, each with a
                    # quarter of its rows inert. A no-op with the arm off.
                    batch = self._select_oci_slots(batch, metrics)

                    # Rows at or past n_real are the duplicates adjust_batch appends
                    # to reach a DP/micro-divisible count; the per-task weights below
                    # need to tell them from the trajectories that were rolled out.
                    n_real = len(batch)
                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Computed from the pre-reorder row order, but _balance_batch moves
                    # the column with its rows, so the weights stay attached either way.
                    if self.config.actor_rollout_ref.actor.get("normalize_loss_by_task", False):
                        attach_task_loss_weights(
                            batch,
                            n_real=n_real,
                            # Rows in one optimizer step, globally. ppo_mini_batch_size is
                            # counted in PROMPTS: the worker multiplies it by rollout.n
                            # (then divides by the DP world size) in
                            # ActorRolloutRefWorker.__init__. The env recipes leave
                            # rollout.n at 1 and expand the group in the env manager
                            # instead, so the factor is usually 1 -- but it decides how
                            # many optimizer steps a batch becomes, which is what the
                            # weights are scaled by.
                            mini_batch_size=(
                                self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            ),
                            metrics=metrics,
                        )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # On pure OPD this scores the batch for monitoring only. The
                    # OPD+GRPO arm overrides it to also compute old_log_prob and
                    # the group-relative advantages the policy gradient needs.
                    batch, reward_extra_infos_dict = self._reward_and_advantage(
                        batch, metrics, timing_raw
                    )

                    # Taken once and read by both passes below. The collector
                    # clears it on the way out, so asking twice would hand the
                    # second caller an empty dict and quietly rescore every row
                    # it was supposed to skip.
                    prefetched = self.traj_collector.take_prefetched_teacher()

                    # ---- Per-task teacher forward pass (the distillation signal;
                    # on pure OPD the ONLY training signal, on OPD+GRPO one of two) ----
                    with _timer("teacher_forward", timing_raw):
                        # writes teacher_log_probs OR teacher_topk_{logprobs,ids} into batch
                        self.compute_teacher_log_probs(
                            batch, prefetched=prefetched, metrics=metrics
                        )

                    # tag rows with their task so the actor can split its metrics.
                    # Before the sign-weight pass rather than after: that pass files
                    # its off-task planes under these ids, and rebuilding the
                    # numbering separately would drift from this one.
                    self._attach_task_ids(batch)
                    # A dense per-row prompt-group id for the online pushback
                    # controller. Only the driver has the uids; the actor turns
                    # these into a per-task bitmap over groups that actually
                    # reached R, which is both the right filter and the only way
                    # to union group sets across ranks.
                    if "uid" in batch.non_tensor_batch:
                        from verl.trainer.ppo.opd_pushback import group_index_column

                        batch.batch["pushback_group_idx"] = group_index_column(
                            batch.non_tensor_batch["uid"], len(batch)
                        )
                    # The cross gate's prompt split (MOPD v1). A stable key per
                    # prompt -- the GRPO group's turn-0 anchor observation, hashed
                    # with the task -- decides which of the two reference halves
                    # the group feeds, and the same key is what the actor's
                    # validity window counts prompts by. Only the driver has the
                    # uids, the turn indices and the anchors, so the columns are
                    # built here and the key list rides in meta_info.
                    _cg_cfg = self.config.algorithm.get("opd", {}).get("cross_gate", None)
                    if _cg_cfg is not None and bool(_cg_cfg.get("enable", False)):
                        from verl.trainer.ppo.opd_cross_gate import cross_gate_prompt_columns

                        _nt = batch.non_tensor_batch
                        _real = np.ones(len(batch), dtype=bool)
                        _pad = batch.batch.get(PADDING_ROW_KEY, None)
                        if _pad is not None:
                            _real &= ~_pad.reshape(-1).to(torch.bool).cpu().numpy()
                        _cols = cross_gate_prompt_columns(
                            uids=_nt["uid"],
                            turn_steps=_nt.get("turn_step", np.zeros(len(batch), dtype=np.int64)),
                            anchors=_nt.get("anchor_obs", [None] * len(batch)),
                            task_names=get_task_names(batch),
                            real=_real,
                            seed=int(_cg_cfg.get("split_seed", 1)),
                        )
                        batch.batch["cross_side"] = _cols["side"]
                        batch.batch["cross_prompt_idx"] = _cols["prompt_idx"]
                        batch.meta_info["cross_prompt_keys"] = list(_cols["keys"])
                        batch.meta_info["cross_unkeyed_rows"] = int(_cols["unkeyed_rows"])
                    # The notice's own readouts (leak floor, truncation floor).
                    metrics.update(self._notice_metrics(batch))

                    # ---- Cross-teacher sign agreement (no-op unless enabled) ---- #
                    # The weights themselves are built in the actor, where the
                    # student's top-k exists; this only puts the other three models
                    # into the same cache the on-task teacher is already in.
                    if self.cross_teacher_enabled:
                        with _timer("sign_weight_forward", timing_raw):
                            self.compute_sign_weight_cache(
                                batch, prefetched=prefetched, metrics=metrics
                            )

                    if self.need_hidden_cache:
                        # After the misses are scored, not before: the cache is only
                        # complete now. Also when only the sign weights use the
                        # cache: a teacher-indexed weighted arm puts base and the
                        # off-task teachers in it without the on-task teacher going
                        # through it, and those entries need the same witness. The witness confirms every entry still
                        # reproduces the log-probs its teacher returned, i.e. that
                        # none has drifted onto another row. One call -- the cache is
                        # per process, so asking all three teachers would check the
                        # same entries three times.
                        cache_stats = next(iter(self.teacher_wg.values())).check_teacher_hidden_cache()
                        # What the cache is holding, summed over ranks. The
                        # weighted arms put four models per row into it instead of
                        # one, next to a vLLM engine already sized to 0.6 of the
                        # card, so this is the first number to read when a step
                        # dies on memory -- and the one that says whether the
                        # headroom is there before it does.
                        per_rank = cache_stats if isinstance(cache_stats, list) else [cache_stats]
                        per_rank = [r for r in per_rank if isinstance(r, dict)]
                        if per_rank:
                            metrics["teacher_cache/rows"] = sum(r["rows"] for r in per_rank)
                            metrics["teacher_cache/gb"] = sum(r["bytes"] for r in per_rank) / 1e9
                            # What of that is still on the CARD. Since put() copies
                            # to host memory as it goes, this should be ~0 between
                            # calls; it is the number that says the offload is
                            # actually happening during the rollout rather than at
                            # the first read inside the update.
                            metrics["teacher_cache/device_gb"] = (
                                sum(r.get("device_bytes", r["bytes"]) for r in per_rank) / 1e9
                            )
                            metrics["teacher_cache/witness_max_err"] = max(
                                r["witness_max_err"] for r in per_rank
                            )

                    # ---- MEASUREMENT MODES (replace the update, never ride beside it) ----
                    # A probe needs the parameters held still while it reads them,
                    # so a step in the same iteration would invalidate its own
                    # measurement. Ported from the probe lineage (grad_probe
                    # 2026-09-07); see grad_probe_driver.py.
                    #
                    # ONLY THE ZERO-BACKWARD MODES CAME ACROSS. `halves` and
                    # `terms` take per-(task, half) gradients through
                    # dp_actor/fsdp_workers hooks that this lineage rewrote
                    # underneath them (2900 changed lines in dp_actor alone), so
                    # they are refused here rather than silently no-opping.
                    probe_cfg = self.config.trainer.get("grad_probe", None)
                    if probe_cfg is not None and bool(probe_cfg.get("enable", False)):
                        from verl.trainer.ppo.grad_probe_driver import (
                            accumulate_tau_probe,
                            new_probe_state,
                            write_payload,
                        )

                        probe_mode = str(probe_cfg.get("mode", "tau") or "tau")
                        assert probe_mode in ("tau", "oci"), (
                            f"grad_probe.mode={probe_mode!r}: this branch carries only the "
                            "zero-backward probe modes ('tau', 'oci'). 'halves' and 'terms' "
                            "need the worker-side gradient hooks, which were not ported -- "
                            "run them from the probe worktree instead."
                        )
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        if getattr(self, "_grad_probe_state", None) is None:
                            self._grad_probe_state = new_probe_state()

                        if probe_mode == "tau":
                            accumulate_tau_probe(
                                self, batch, self._grad_probe_state,
                                seed=int(probe_cfg.get("seed", 0)),
                            )
                        else:
                            self._accumulate_oci_probe(batch, self._grad_probe_state, probe_cfg)

                        done = self._grad_probe_state["batches"]
                        n_batches = int(probe_cfg.get("n_batches", 1) or 1)
                        every = int(probe_cfg.get("interim_every", 1) or 0)
                        out_path = str(probe_cfg.get("out_path", "grad_probe.json"))
                        final = done >= n_batches
                        # A report at batch k is a valid result at that k, so the
                        # interim write is the same payload, not a partial one.
                        if final or (every > 0 and done % every == 0):
                            payload = {
                                "mode": probe_mode,
                                "checkpoint": str(
                                    self.config.trainer.get("resume_from_path", None)
                                    or self.config.actor_rollout_ref.model.path
                                ),
                                "n_batches": done,
                                "final": bool(final),
                                "rollout_n": int(self.config.env.rollout.get("n", 1)),
                                "temperature": float(
                                    self.config.actor_rollout_ref.rollout.temperature
                                ),
                            }
                            for k in ("advantages", "tau", "groups", "prompt_len",
                                      "oci", "reachability"):
                                if self._grad_probe_state.get(k, None):
                                    payload[k] = self._grad_probe_state[k]
                            write_payload(payload, out_path)
                        if final:
                            pprint(f"[grad_probe] mode={probe_mode}: report written after "
                                   f"{done} batch(es); no optimizer step taken.")
                            return
                        # Not done yet: skip the update and draw the next batch.
                        # The counters still advance so the log reads normally and
                        # `is_last_step` cannot stall; nothing below them runs,
                        # which is the point -- no update, no validation, no save.
                        progress_bar.update(1)
                        self.global_steps += 1
                        continue

                    with _timer("update_actor", timing_raw):
                        # update_policy scales the student logits by this temperature to
                        # match the rollout sampling distribution. The standard loop gets
                        # it from compute_log_prob; the thin OPD loop skips that step, so
                        # set it explicitly (same value compute_log_prob would set).
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        # THE CURRICULUM'S SCHEDULE, evaluated here and shipped as
                        # two numbers. Here rather than in the actor because the
                        # actor is never told which step it is on, and because
                        # global_steps is restored from the checkpoint folder --
                        # so the release is resume-correct with no state of its
                        # own, the same way the env schedules are replayed after
                        # _load_checkpoint. A broadcast scalar also cannot be
                        # changed by the mini-batch or micro-batch split, which
                        # keeps the objective invariant to both.
                        if self.cross_teacher_curriculum is not None:
                            from verl.trainer.ppo.cross_teacher_target import curriculum_rho

                            _rho = curriculum_rho(
                                step=self.global_steps, **self.cross_teacher_curriculum
                            )
                            batch.meta_info["cross_teacher_curriculum_rho"] = (
                                _rho["pair"], _rho["own"],
                            )
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                    self._dump_sign_token_report(actor_output)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir and "token_level_scores" in batch.batch:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(self._data_metrics(batch=batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if stop_after and self.global_steps > stop_after:
                    # The step just finished IS stop_after (global_steps has moved
                    # past it) and its checkpoint was saved above.
                    pprint(f"Stopping after step {stop_after} as requested (trainer.stop_after_steps); resume to continue to {self.total_training_steps}.")
                    progress_bar.close()
                    return
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
