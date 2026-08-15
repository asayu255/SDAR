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
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import open_dict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_metrics_by_task,
    compute_trajectory_response_tokens,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    _timer,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.sign_weights import (
    SIGN_WEIGHT_KEY,
    candidate_weights,
    normalize_per_task,
    position_weights,
    reweight_teacher_logprobs,
    sign_state_metrics,
)
from verl.trainer.ppo.task_loss_weights import attach_task_loss_weights
from verl.utils import gpu_profiler
from verl.utils.metric import reduce_metrics

from agent_system.multi_turn_rollout import adjust_batch


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

        # ---- cross-teacher sign agreement (optional) --------------------------
        # Off by default so every existing arm is untouched: with enable=false no
        # base worker is built, no extra forward runs, and the loss is the one
        # the arm had before.
        sw_cfg = dict(opd_cfg.get("sign_weight", {}) or {})
        self.sign_weight_enabled = bool(sw_cfg.get("enable", False))
        self.sign_weight_mode = str(sw_cfg.get("mode", "position"))
        self.sign_weight_agree = float(sw_cfg.get("agree_weight", 1.25))
        self.sign_weight_disagree = float(sw_cfg.get("disagree_weight", 0.75))
        self.sign_weight_deadzone = float(sw_cfg.get("deadzone", 0.1))
        self.base_policy_path = sw_cfg.get("base_path", None)
        self.base_wg = None
        if self.sign_weight_enabled:
            assert self.sign_weight_mode in ("position", "target"), (
                f"algorithm.opd.sign_weight.mode must be 'position' or 'target', got {self.sign_weight_mode!r}"
            )
            # The signal is the sign of log pi_m - log pi_0 on a support shared by
            # all four models, and the only shared support available is the
            # on-task teacher's top-k. The single-sampled-token estimator does not
            # produce one, so it cannot carry this mechanism.
            assert self.teacher_topk_kl, (
                "sign weighting requires algorithm.opd.kl_loss_type=topk_kl "
                "(the teacher's top-k ids are the support every model is scored on)"
            )
            assert self.base_policy_path, (
                "sign weighting requires algorithm.opd.sign_weight.base_path "
                "(the pre-RL policy the teachers' shifts are measured against)"
            )
            assert len(self.teacher_paths) >= 2, (
                "sign weighting needs at least one off-task teacher besides the on-task one"
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
        # teacher (same role="ref", so it is CPU-offloaded too) but kept out of
        # self.teacher_paths / self.teacher_wg, because those are keyed by task
        # and drive the routing: a fourth entry there would have to survive
        # _normalize_task_name and would then be looked up for rows that do not
        # exist.
        if self.sign_weight_enabled:
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
        for task, key in self._teacher_keys.items():
            wg = all_wg[key]
            wg.init_model()
            self.teacher_wg[task] = wg
        if self.sign_weight_enabled:
            self.base_wg = all_wg["base_policy"]
            self.base_wg.init_model()

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

        Args:
            chunk: list of ``((traj_uid, turn_step), row_dict)`` as queued by the
                rollout loop. Rows carry the four model-input tensors and their
                ``task_name``.

        Returns:
            ``{(traj_uid, turn_step): tensor-or-tuple}`` in the same form
            ``compute_teacher_log_probs`` writes: a ``(resp, k)`` log-prob/id pair
            under top-k, else a ``(resp,)`` log-prob row.
        """
        by_task = {}
        for key, row in chunk:
            task = self._normalize_task_name(row.get("task_name"))
            by_task.setdefault(task, []).append((key, row))

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
                },
            )
            # Same dispatch as the post-rollout path: chunks are not divisible by
            # the teacher group's world size, so let DP_COMPUTE_PROTO pad and unpad.
            sub.meta_info = {DataProtoConfig.auto_padding_key: True}
            if self.teacher_topk_kl:
                sub.meta_info["topk_k"] = self.teacher_kl_topk
                scored = wg.compute_ref_topk_log_prob(sub)
                tlp = scored.batch["teacher_topk_logprobs"]
                tid = scored.batch["teacher_topk_ids"]
                for j, (key, _) in enumerate(entries):
                    out[key] = (tlp[j], tid[j])
            else:
                scored = wg.compute_ref_log_prob(sub)
                lp = scored.batch["ref_log_prob"]
                for j, (key, _) in enumerate(entries):
                    out[key] = lp[j]
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

        if self.teacher_topk_kl:
            k = self.teacher_kl_topk
            teacher_topk_logprobs = torch.zeros((bs, resp_len, k), dtype=torch.float32)
            teacher_topk_ids = torch.zeros((bs, resp_len, k), dtype=torch.long)
        else:
            teacher_log_probs = torch.zeros((bs, resp_len), dtype=torch.float32)

        keys = self._prefetched_teacher_rows(batch) if prefetched else None
        if keys is not None:
            for i, key in keys.items():
                hit = prefetched.get(key)
                if hit is None:
                    continue
                if self.teacher_topk_kl:
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
            # Task slices are not generally divisible by the teacher group's world
            # size, so enable auto-padding: the DP dispatch pads the input to a
            # multiple of world_size and unpads the output back to len(idxs). Copy
            # meta_info first so we don't mutate the parent batch (select_idxs
            # shares the parent's meta_info dict by reference).
            sub.meta_info = dict(sub.meta_info)
            sub.meta_info[DataProtoConfig.auto_padding_key] = True
            # The teachers run one after another in this loop, so "teacher_forward"
            # as a single phase says how long all three took together but not which
            # one dominates -- and they are not interchangeable: the tasks differ in
            # prompt length (webshop's mean prompt is ~3x alfworld's), and under
            # topk_kl each teacher ships back (rows, resp_len, k) log-probs and ids
            # instead of one value per token. Tagging per task splits both the
            # compute and that transfer out by teacher.
            gpu_profiler.push_phase(f"teacher_forward/{task}")
            try:
                if self.teacher_topk_kl:
                    sub.meta_info["topk_k"] = k
                    out = wg.compute_ref_topk_log_prob(sub)
                    tlp = out.batch["teacher_topk_logprobs"]
                    tid = out.batch["teacher_topk_ids"]
                    for j, i in enumerate(idxs):
                        teacher_topk_logprobs[i] = tlp[j]
                        teacher_topk_ids[i] = tid[j]
                        seen[i] = True
                else:
                    out = wg.compute_ref_log_prob(sub)
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

        if self.teacher_topk_kl:
            batch.batch["teacher_topk_logprobs"] = teacher_topk_logprobs
            batch.batch["teacher_topk_ids"] = teacher_topk_ids
        else:
            batch.batch["teacher_log_probs"] = teacher_log_probs

    # ------------------------------------------------------------------ #
    # Cross-teacher sign agreement (optional; see sign_weights.py).
    # ------------------------------------------------------------------ #
    def _score_at_teacher_ids(self, wg, batch: DataProto, idxs: list) -> torch.Tensor:
        """Run one frozen model over ``idxs`` at the on-task teacher's top-k ids."""
        lean = batch.select(
            batch_keys=["responses", "input_ids", "attention_mask", "position_ids", "teacher_topk_ids"],
            non_tensor_batch_keys=[],
        )
        sub = lean.select_idxs(idxs)
        # Task slices are not divisible by the worker group's world size; the DP
        # dispatch pads and unpads around the call (same contract as the teacher
        # forward above).
        sub.meta_info = dict(sub.meta_info)
        sub.meta_info[DataProtoConfig.auto_padding_key] = True
        out = wg.compute_ref_topk_log_prob_at_ids(sub)
        return out.batch["topk_logprobs_at_ids"]

    def compute_sign_weights(self, batch: DataProto, metrics: Optional[dict] = None) -> None:
        """Score the base policy and the off-task teachers, and turn the sign
        agreement between them and the on-task teacher into a distillation weight.

        Writes one of two things into ``batch``, per ``sign_weight.mode``:

        - ``position``: a per-token weight under
          :data:`~verl.trainer.ppo.sign_weights.SIGN_WEIGHT_KEY`, which the actor
          multiplies into the per-token KL.
        - ``target``: a rewritten ``teacher_topk_logprobs``, so the existing loss
          distils to a reweighted target and the actor needs no change.

        Cost is three extra frozen forwards per step: the base over every row,
        and each teacher over the rows that are *not* its own task (2/3 of the
        batch each, so two batch-equivalents). The on-task teacher's pass already
        happened in :meth:`compute_teacher_log_probs` and is reused.
        """
        if not self.sign_weight_enabled:
            return

        assert "teacher_topk_ids" in batch.batch, "sign weighting runs after the teacher forward"
        on_lp = batch.batch["teacher_topk_logprobs"]  # (bs, resp, k)
        bs = on_lp.size(0)

        task_names = batch.non_tensor_batch.get("task_name", None)
        assert task_names is not None, "sign weighting requires task_name for on/off-task routing"
        normalized = [self._normalize_task_name(t) for t in task_names]

        gpu_profiler.push_phase("sign_weight_forward/base")
        try:
            base_lp = self._score_at_teacher_ids(self.base_wg, batch, list(range(bs)))
        finally:
            gpu_profiler.pop_phase("sign_weight_forward/base")

        # Every teacher scores the rows it is NOT the teacher of. Rows of its own
        # task already have their on-task values from compute_teacher_log_probs,
        # and re-scoring them here would be the same forward twice.
        task_order = sorted(self.teacher_wg.keys())
        # Seeded with the base rather than with zeros: a plane cell that is never
        # written is a teacher's own task's rows, which plane_idx below excludes.
        # If that exclusion were ever wrong, a zero would read as a shift of
        # -log pi_0 -- a large, confident, entirely fictitious opinion -- while
        # the base value reads as "this teacher did not move anything here" and
        # the weighting simply switches itself off.
        stacked = base_lp.to(torch.float32).unsqueeze(0).repeat(len(task_order), 1, 1, 1)
        for plane, task in enumerate(task_order):
            idxs = [i for i, t in enumerate(normalized) if t != task]
            if not idxs:
                continue
            gpu_profiler.push_phase(f"sign_weight_forward/{task}")
            try:
                lp = self._score_at_teacher_ids(self.teacher_wg[task], batch, idxs)
            finally:
                gpu_profiler.pop_phase(f"sign_weight_forward/{task}")
            stacked[plane, idxs] = lp.to(torch.float32)

        # For each row pick the planes of the teachers that are off-task for it.
        n_off = len(task_order) - 1
        plane_idx = torch.empty((bs, n_off), dtype=torch.long)
        for i, t in enumerate(normalized):
            others = [j for j, task in enumerate(task_order) if task != t]
            assert len(others) == n_off, f"row {i} has task {t!r}, which is not one of {task_order}"
            plane_idx[i] = torch.tensor(others, dtype=torch.long)
        off_lp = stacked[plane_idx, torch.arange(bs).unsqueeze(-1)]  # (bs, n_off, resp, k)
        off_lp = off_lp.permute(0, 2, 3, 1).contiguous()  # (bs, resp, k, n_off)

        cand_w, state = candidate_weights(
            on_lp.to(torch.float32),
            off_lp,
            base_lp.to(torch.float32),
            agree_weight=self.sign_weight_agree,
            disagree_weight=self.sign_weight_disagree,
            deadzone=self.sign_weight_deadzone,
        )

        response_mask = batch.batch["response_mask"]
        if metrics is not None:
            metrics.update(sign_state_metrics(state, response_mask))
            metrics["sign_weight/mode_is_target"] = float(self.sign_weight_mode == "target")

        if self.sign_weight_mode == "position":
            pos_w = position_weights(cand_w, on_lp.to(torch.float32))
            pos_w = normalize_per_task(pos_w, response_mask, batch.batch.get("task_ids", None))
            batch.batch[SIGN_WEIGHT_KEY] = pos_w
            if metrics is not None:
                valid = response_mask.to(torch.bool)
                sel = pos_w[valid]
                if sel.numel():
                    metrics["sign_weight/w_mean"] = float(sel.mean())
                    metrics["sign_weight/w_std"] = float(sel.std())
                    metrics["sign_weight/w_min"] = float(sel.min())
                    metrics["sign_weight/w_max"] = float(sel.max())
        else:
            batch.batch["teacher_topk_logprobs"] = reweight_teacher_logprobs(on_lp.to(torch.float32), cand_w)

    # ------------------------------------------------------------------ #
    # Thin training loop: rollout -> teacher_log_probs -> update_actor.
    # No old_log_prob / ref / values / advantage / reward-in-loss.
    # ------------------------------------------------------------------ #
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
        # run scripts set val_before_train=False (the initial eval costs a full
        # validation pass and says nothing a resumed run does not already know),
        # and with the check nested a "validate this checkpoint and stop" command
        # silently fell through to TRAINING from the checkpoint instead -- which
        # looks like it worked, right up until the numbers you wanted never
        # appear.
        val_only = bool(self.config.trainer.get("val_only", False))
        if self.val_reward_fn is not None and (val_only or self.config.trainer.get("val_before_train", True)):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
        if val_only:
            assert self.val_reward_fn is not None, "trainer.val_only=True but no validation reward fn is configured"
            return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
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

                    del batch
                    batch = gen_batch_output

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

                    # Reward is computed for monitoring only (task success); it is never
                    # turned into advantages and never enters the loss.
                    reward_extra_infos_dict = {}
                    with _timer("reward", timing_raw):
                        try:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        except Exception as e:  # monitoring must never break training
                            print(f"[OPD] reward computation skipped: {e}")

                    # ---- Per-task teacher forward pass (the only training signal) ----
                    with _timer("teacher_forward", timing_raw):
                        # writes teacher_log_probs OR teacher_topk_{logprobs,ids} into batch
                        self.compute_teacher_log_probs(
                            batch,
                            prefetched=self.traj_collector.take_prefetched_teacher(),
                            metrics=metrics,
                        )

                    # tag rows with their task so the actor can split its metrics
                    self._attach_task_ids(batch)

                    with _timer("update_actor", timing_raw):
                        # update_policy scales the student logits by this temperature to
                        # match the rollout sampling distribution. The standard loop gets
                        # it from compute_log_prob; the thin OPD loop skips that step, so
                        # set it explicitly (same value compute_log_prob would set).
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

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
                metrics.update(compute_opd_data_metrics(batch=batch))
                metrics.update(compute_opd_data_metrics_by_task(batch=batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
