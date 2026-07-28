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
    def compute_teacher_log_probs(self, batch: DataProto) -> None:
        """Route each sample to its task's teacher and write the distillation
        signal into ``batch.batch`` in original order.

        The student's exact (prompt, response) is fed to the teacher — no skill
        prepend. The teacher call is ``DP_COMPUTE_PROTO``-dispatched; per-task
        slices are auto-padded to the DP world size and unpadded on return.

        - default (single-token estimator): sets ``teacher_log_probs`` (bs, resp).
        - top-k mode: sets ``teacher_topk_logprobs`` and ``teacher_topk_ids``
          (bs, resp, k), the teacher's top-k log-softmax and ids per token.
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

        for task, wg in self.teacher_wg.items():
            idxs = [i for i, t in enumerate(normalized) if t == task]
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

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
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
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

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
                        self.compute_teacher_log_probs(batch)

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
