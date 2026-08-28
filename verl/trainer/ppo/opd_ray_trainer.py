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
        # Who needs base, the cache and the extra forwards -- as opposed to who
        # builds a weight out of them. Every gate below is this one, so adding a
        # third consumer never means finding the cache gates again.
        self.cross_teacher_enabled = self.sign_weight_enabled or self.cross_teacher_kl_weight_enabled
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
            xt_cfg.get("base_path", None) if self.cross_teacher_kl_weight_enabled
            else sw_cfg.get("base_path", None)
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
                self._teacher_call(wg, sub, topk=True, cache_ids=ids)
                for key, _ in entries:
                    out[key] = cache_id_for[key]
            elif self.teacher_topk_kl:
                scored = self._teacher_call(wg, sub, topk=True, cache_ids=ids)
                tlp = scored.batch["teacher_topk_logprobs"]
                tid = scored.batch["teacher_topk_ids"]
                for j, (key, _) in enumerate(entries):
                    out[key] = (tlp[j], tid[j], cache_id_for.get(key, -1))
            else:
                scored = self._teacher_call(wg, sub, topk=False, cache_ids=ids)
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
        returned rows exactly as they were before this path existed.
        """
        if not (self.cross_teacher_enabled and _ROLLOUT_PREFETCH_SIGN):
            return None
        task_order = sorted(self.teacher_wg.keys())
        rows = [(key, row, self._normalize_task_name(row.get("task_name"))) for key, row in chunk]
        # A row whose task has no teacher is left for the serial path, which
        # raises on it by name rather than from a background thread.
        rows = [r for r in rows if r[2] in self.teacher_wg]
        if not rows:
            return None

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
            self._teacher_call(wg, sub, topk=True, cache_ids=ids)

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

    @staticmethod
    def _sign_off_tasks_for(own, task_order):
        """The off-task teachers of a row whose own task is ``own``, in column order.

        One expression, read by both the prefetch and the post-rollout pass, so
        the columns they write cannot drift apart. The actor reads them
        positionally and has no way to notice if they did.
        """
        return [t for t in task_order if t != own]

    def _teacher_call(self, wg, sub: DataProto, topk: bool, cache_ids=None):
        """One teacher call, with the DP padding marked so it is never cached.

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
                    self._teacher_call(wg, sub, topk=True, cache_ids=miss_ids)
                    for i in idxs:
                        seen[i] = True
                elif self.teacher_topk_kl:
                    out = self._teacher_call(wg, sub, topk=True, cache_ids=miss_ids)
                    tlp = out.batch["teacher_topk_logprobs"]
                    tid = out.batch["teacher_topk_ids"]
                    for j, i in enumerate(idxs):
                        teacher_topk_logprobs[i] = tlp[j]
                        teacher_topk_ids[i] = tid[j]
                        seen[i] = True
                else:
                    out = self._teacher_call(wg, sub, topk=False, cache_ids=miss_ids)
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
                self._teacher_call(wg, lean.select_idxs(part), topk=True, cache_ids=ids)

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
                            metrics["teacher_cache/witness_max_err"] = max(
                                r["witness_max_err"] for r in per_rank
                            )

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
