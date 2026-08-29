"""
OPD + GRPO Trainer — multitask.

This is the OPD (On-Policy Distillation) multitask trainer with GRPO added back
on top. The student is trained jointly by:

  policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef

i.e. the GRPO policy-gradient (group-relative advantages from the env reward)
*plus* the per-task teacher-KL distillation that pure OPD uses. Everything else —
per-task teacher routing, the 3-task (alfworld/search/webshop) data, batch sizes,
env settings — matches the pure-OPD multitask run.

Teacher routing and worker setup are inherited unchanged from ``OPDRayTrainer``
(teachers are ``role="ref"`` worker groups, one per task). Only ``fit()`` is
overridden to restore the GRPO ``old_log_prob`` + advantage computation that the
thin OPD loop skips, while keeping the teacher forward pass as an additional
training signal.
"""

from pprint import pprint

import os

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer
from verl.trainer.ppo.ray_trainer import (
    _timer,
    agg_loss,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics

from verl.trainer.ppo.task_loss_weights import attach_task_loss_weights

from agent_system.multi_turn_rollout import adjust_batch, compute_log_prob_with_prefetch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# The reset is pure CPU / subprocess / HTTP work and the env managers are idle
# between rollouts; the reset still runs exactly once per rollout and in the same
# order, so stateful env schedules (alfworld's game-file iterator) are unchanged.
# Opt-in; see TrajectoryCollector.prefetch_env_reset.
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


class OPDGRPORayTrainer(OPDRayTrainer):
    """Multitask trainer combining GRPO policy-gradient with per-task teacher-KL distillation."""

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
    # Training loop: rollout -> old_log_prob -> reward -> advantage (GRPO)
    #                -> teacher_log_probs -> update_actor.
    # The GRPO path (old_log_prob/advantage) is restored on top of the OPD
    # teacher forward pass so both signals enter the loss.
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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD+GRPO Training")
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
                    self.clear_teacher_hidden_cache()
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                            # Score finished trajectories under the rollout's own
                            # CPU glue instead of after it; a no-op unless
                            # ROLLOUT_PREFETCH_TEACHER is on. Routing and the merge
                            # are inherited from OPDRayTrainer.
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
                    # On this arm they normalise the GRPO policy gradient as well as the
                    # teacher KL -- see DataParallelPPOActor.update_policy.
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

                    # ---- reward (env score; feeds GRPO advantages) ----
                    reward_extra_infos_dict = {}
                    with _timer("reward", timing_raw):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # ---- old_log_prob (required by the GRPO policy-gradient) ----
                    with _timer("old_log_prob", timing_raw):
                        # Reuse any per-row log probs prefetched during the rollout
                        # (ROLLOUT_PREFETCH_LOGPROB); computes everything normally
                        # when nothing was prefetched.
                        old_log_prob = compute_log_prob_with_prefetch(
                            self.actor_rollout_wg,
                            batch,
                            self.traj_collector.take_prefetched_log_probs(),
                            temperature=self.config.actor_rollout_ref.rollout.temperature,
                        )
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        metrics.update(self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode))
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- advantages (GRPO) ----
                    with _timer("adv", timing_raw):
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                    "invalid_action_penalty_coef_by_task", None
                                ),
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # ---- Per-task teacher forward pass (the distillation signal) ----
                    with _timer("teacher_forward", timing_raw):
                        # writes teacher_log_probs OR teacher_topk_{logprobs,ids} into batch
                        self.compute_teacher_log_probs(
                            batch,
                            prefetched=self.traj_collector.take_prefetched_teacher(),
                            metrics=metrics,
                        )

                    self.check_teacher_hidden_cache(metrics)

                    # tag rows with their task so the actor can split its metrics
                    self._attach_task_ids(batch)

                    with _timer("update_actor", timing_raw):
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
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
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
