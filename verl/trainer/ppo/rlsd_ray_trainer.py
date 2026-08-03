"""
RLSD (Reinforcement Learning with Self-Distillation) Trainer.

Extends the standard RayPPOTrainer with a teacher forward pass that uses
privileged information (skills) to construct token-level advantages.
"""

from contextlib import contextmanager
from pprint import pprint

import os

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider, compute_rlsd_token_advantage
from verl.utils import gpu_profiler
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_throughout_metrics,
    compute_timing_metrics,
    task_row_indices,
)
from verl.utils.model import compute_position_id_with_mask

from agent_system.multi_turn_rollout import adjust_batch, compute_log_prob_with_prefetch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# See SkillSDRayTrainer / TrajectoryCollector.prefetch_env_reset for details.
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
):
    """
    Build a teacher batch by prepending privileged skill info to each sample's prompt.

    The teacher sees (x, r) where r is the skill/privileged information.
    We prepend the skill text as a system message before the user prompt,
    then re-tokenize to get teacher input_ids/attention_mask/position_ids.
    The responses remain unchanged.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        # Decode the original prompt (student input minus response)
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        if len(valid_start) > 0:
            valid_start = valid_start[0].item()
        else:
            valid_start = 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Get privileged skill info based on gamefile, data_source, or prompt text
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        data_source = batch.non_tensor_batch.get("data_source", None)
        task_name = batch.non_tensor_batch.get("task_name", None)
        gf = gamefile[i] if gamefile is not None else None
        ds = data_source[i] if data_source is not None else None
        task = task_name[i] if task_name is not None else None
        skill_text = skill_provider.get_privileged_info_for_sample(
            task_name=task,
            gamefile=gf,
            data_source=ds,
            prompt_text=prompt_text,
        )

        # Construct teacher prompt: prepend skill as a system message
        skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
        teacher_prompt_text = skill_prefix + prompt_text

        # Tokenize the teacher prompt
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Truncate if needed (left truncation to keep the end of prompt)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_input_ids = torch.stack(teacher_input_ids_list)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )

    return teacher_batch


@contextmanager
def _profiler_phase(name: str):
    """Tag GPU samples with a sub-phase, without adding a timing_raw entry.

    ``_timer`` would do both, but its dict entries land in ``timing_s/*`` and are
    divided by the batch size for ``timing_per_token_ms/*``; these two are an
    interior view of a phase that is already reported, not two more phases.
    """
    gpu_profiler.push_phase(name)
    try:
        yield
    finally:
        gpu_profiler.pop_phase(name)


def _teacher_batch_metrics(student: DataProto, teacher: DataProto, max_prompt_length: int) -> dict:
    """What the teacher forward is actually being asked to chew through, per task.

    The GPU call is one call over the mixed batch, so its wall clock cannot be
    split by task the way a per-task teacher loop's could. What can be reported
    is the cost driver: prompt length differs several-fold between these tasks,
    and the privileged skill text prepended on top differs per task as well.

    ``skill_overhead_tokens`` is the prepend's actual cost -- teacher prompt
    tokens minus student prompt tokens -- and ``skill_truncated_ratio`` is the
    share of rows whose teacher prompt came out at exactly ``max_prompt_length``.
    That second one is worth watching: ``build_teacher_batch`` keeps the END of
    an over-long prompt, and the skill text sits at the BEGINNING, so a saturated
    row is one where the privileged information was silently cut away and the
    "teacher" is running on the student's own prompt. Length equality is a proxy
    for that, not a proof, but a ratio that is not ~0 means the teacher signal is
    not what the recipe thinks it is.
    """
    response_length = student.batch["responses"].size(1)
    student_prompt = student.batch["attention_mask"][:, :-response_length].sum(-1).float()
    teacher_prompt = teacher.batch["attention_mask"][:, :-response_length].sum(-1).float()
    overhead = teacher_prompt - student_prompt
    saturated = (teacher_prompt >= max_prompt_length).float()

    metrics = {
        "teacher_forward/prompt_tokens/mean": float(teacher_prompt.mean()),
        "teacher_forward/skill_overhead_tokens/mean": float(overhead.mean()),
        "teacher_forward/skill_truncated_ratio": float(saturated.mean()),
    }
    for task, rows in task_row_indices(student).items():
        idx = torch.from_numpy(rows)
        metrics[f"teacher_forward/prompt_tokens/mean/{task}"] = float(teacher_prompt[idx].mean())
        metrics[f"teacher_forward/skill_overhead_tokens/mean/{task}"] = float(overhead[idx].mean())
        metrics[f"teacher_forward/skill_truncated_ratio/{task}"] = float(saturated[idx].mean())
        metrics[f"teacher_forward/rows/{task}"] = int(len(rows))
    return metrics


class RLSDRayTrainer(RayPPOTrainer):
    """
    RLSD trainer that extends RayPPOTrainer with self-distillation
    using privileged skill information as teacher signal.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        # RLSD hyperparams from config
        rlsd_cfg = self.config.algorithm.get("rlsd", {})
        self.rlsd_lambda_init = rlsd_cfg.get("rlsd_lambda", 0.5)
        self.rlsd_lambda_warmdown_steps = rlsd_cfg.get("warmdown_steps", 50)
        self.rlsd_clip_eps = rlsd_cfg.get("clip_eps", 0.2)

    def _get_rlsd_lambda(self, step: int) -> float:
        """Linearly decay λ from rlsd_lambda_init to 0 over warmdown_steps."""
        if step >= self.rlsd_lambda_warmdown_steps:
            return 0.0
        return self.rlsd_lambda_init * (1.0 - step / self.rlsd_lambda_warmdown_steps)

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

    def fit(self):
        """
        The training loop of RLSD, extending the standard PPO/GRPO loop
        with teacher forward pass and token-level advantage computation.
        """
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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RLSD Training")
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
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
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
                            next_env_kwargs = np.repeat(
                                peeked_batch_dict["env_kwargs"], self.config.env.rollout.n
                            )
                            self.traj_collector.prefetch_env_reset(self.envs, next_env_kwargs)

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute student log probs (old_log_probs)
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
                        old_log_prob_metrics = self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode)
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- RLSD: Teacher forward pass ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch, metrics=metrics)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                    "invalid_action_penalty_coef_by_task", None
                                ),
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute standard GRPO sequence-level advantages
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

                        # ---- RLSD: Replace sequence-level advantage with token-level advantage ----
                        seq_advantages = batch.batch["advantages"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_log_probs = batch.batch["teacher_log_probs"]
                        response_mask = batch.batch["response_mask"]

                        current_lambda = self._get_rlsd_lambda(self.global_steps)
                        token_advantages = compute_rlsd_token_advantage(
                            seq_advantages=seq_advantages,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            rlsd_lambda=current_lambda,
                            rlsd_clip_eps=self.rlsd_clip_eps,
                        )

                        batch.batch["advantages"] = token_advantages

                        # Log RLSD-specific metrics
                        delta_t = (teacher_log_probs - student_log_probs) * response_mask
                        metrics["rlsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["rlsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["rlsd/lambda"] = current_lambda
                        metrics["rlsd/clip_eps"] = self.rlsd_clip_eps
                        for task, rows in task_row_indices(batch).items():
                            task_rows = torch.from_numpy(rows)
                            task_delta_t, task_mask = delta_t[task_rows], response_mask[task_rows]
                            metrics[f"rlsd/teacher_student_gap_mean/{task}"] = masked_mean(task_delta_t, task_mask).item()
                            metrics[f"rlsd/teacher_student_gap_std/{task}"] = masked_mean(task_delta_t ** 2, task_mask).sqrt().item()

                    # tag rows with their task so the actor can split its metrics
                    self._attach_task_ids(batch)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
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

    def _compute_teacher_log_probs(self, batch: DataProto, metrics: dict = None) -> torch.Tensor:
        """
        Compute teacher log probs by running forward pass with privileged skill info
        prepended to the prompt. Uses the same model π_θ but conditioned on (x, r).

        The driver's ``teacher_forward`` timer covers two things that behave
        nothing alike, so they are pushed as separate profiler phases:

        ``teacher_forward/build``   ``build_teacher_batch`` is a row-by-row Python
            loop that decodes each prompt, asks the skill provider for its
            privileged text and re-encodes the result -- two tokenizer calls per
            row, on the driver, with the GPU at zero. At 45 prompts x group 8
            that is 720 tokenizer calls a step charged to a phase whose name says
            "forward".
        ``teacher_forward/logprob`` the actual GPU pass.

        Both are inherited by SkillSDRayTrainer, so SDAR gets the split too.

        Args:
            batch: the student batch.
            metrics: optional dict the per-task size diagnostics are written into.
        """
        with _profiler_phase("teacher_forward/build"):
            teacher_batch = build_teacher_batch(
                batch=batch,
                skill_provider=self.skill_provider,
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation=self.config.data.get("truncation", "left"),
            )

        if metrics is not None:
            metrics.update(
                _teacher_batch_metrics(
                    student=batch,
                    teacher=teacher_batch,
                    max_prompt_length=self.config.data.max_prompt_length,
                )
            )

        # Use the same actor to compute teacher log probs
        with _profiler_phase("teacher_forward/logprob"):
            teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]

        return teacher_log_probs
