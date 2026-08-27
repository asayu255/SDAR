"""
OPD + GRPO Trainer — multitask.

This is the OPD (On-Policy Distillation) multitask trainer with GRPO added back
on top. The student is trained jointly by:

  policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * teacher_kl_coef

i.e. the GRPO policy-gradient (group-relative advantages from the env reward)
*plus* the per-task teacher-KL distillation that pure OPD uses. Everything else —
per-task teacher routing, the 3-task (alfworld/search/webshop) data, batch sizes,
env settings, the student-top-k support and the cross-teacher sign weighting —
matches the pure-OPD multitask run.

Everything except the objective is inherited from :class:`OPDRayTrainer`, and
deliberately so. The teacher routing, the hidden-state cache and its witness, the
sign-weight pass, the env-reset prefetch and ``stop_after_steps`` are the parts
that have to stay identical for an A/B against pure OPD to mean anything; a
second copy of that loop is exactly how "the arms differ only in the objective"
quietly stops being true. So this subclass overrides only the two hooks
``OPDRayTrainer.fit`` calls:

* :meth:`_reward_and_advantage` — pure OPD scores the batch for monitoring only;
  here the same reward becomes ``token_level_rewards`` and, via ``old_log_prob``
  and ``compute_advantage``, the group-relative advantages the policy gradient
  reads.
* :meth:`_data_metrics` — the advantage-bearing batch can use the full
  ``compute_data_metrics`` instead of the advantage-free OPD variant.
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
)
from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward

from agent_system.multi_turn_rollout.utils import PADDING_ROW_KEY

from agent_system.multi_turn_rollout import compute_log_prob_with_prefetch


class OPDGRPORayTrainer(OPDRayTrainer):
    """Multitask trainer combining GRPO policy-gradient with per-task teacher-KL distillation."""

    progress_desc = "OPD+GRPO Training"

    def _reward_and_advantage(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Score the batch, then turn that score into GRPO advantages.

        Unlike the pure-OPD base, the reward is NOT monitoring-only here: it is
        the policy gradient's whole signal, so a reward-manager failure is a
        failed step rather than something to print and carry on from. Hence no
        try/except around ``compute_reward``.

        Order matches the standard PPO loop: reward, then ``old_log_prob`` (the
        ratio's denominator, so it must be the policy that generated the
        responses — i.e. before ``update_actor``), then advantages.
        """
        with _timer("reward", timing_raw):
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        # ---- old_log_prob (required by the GRPO policy-gradient) ----
        with _timer("old_log_prob", timing_raw):
            # Reuse any per-row log probs prefetched during the rollout
            # (ROLLOUT_PREFETCH_LOGPROB); computes everything normally when
            # nothing was prefetched.
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

            batch = self._attach_advantage_reliability_columns(batch)

        return batch, reward_extra_infos_dict

    def _attach_advantage_reliability_columns(self, batch: DataProto) -> DataProto:
        """Per row: its advantage, and whether its prompt group carried any signal.

        The parameter-free cross-teacher arm calibrates each source teacher by
        correlating its residual support for the tokens the student emitted
        against the advantage of the trajectory it emitted them in, so it needs
        both a per-ROW advantage and a marker for which rows can inform that
        correlation. Both are driver-side facts -- the actor sees micro-batches
        and cannot see a prompt group at all -- and both are cheap here.

        ``adv_group_informative`` is false wherever the group has no spread of
        advantage. GRPO is group-relative, so a prompt whose rollouts all scored
        the same gives every one of its rows an advantage of zero; folding those
        into the correlation adds variance to the support score against none in
        the advantage and drags every pair's estimate toward zero for a reason
        that has nothing to do with the teachers. The comparison is against the
        advantages already computed -- no new threshold is introduced.

        Padding rows are excluded outright: ``adjust_batch`` appends them as
        copies carrying their original's uid, so leaving them in would count one
        trajectory twice.
        """
        adv = batch.batch["advantages"]
        mask = batch.batch["response_mask"].to(adv.dtype)
        denom = mask.sum(dim=-1).clamp(min=1)
        row_adv = (adv * mask).sum(dim=-1) / denom

        # Outcome GRPO broadcasts one score across the row, and the reliability
        # correlation is a statement about trajectories. Checked rather than
        # assumed: a step-level estimator would make the row mean an average of
        # different things and the correlation would quietly change meaning.
        spread = ((adv - row_adv.unsqueeze(-1)).abs() * mask).max()
        if float(spread) > 1e-4:
            print(
                f"[cross_teacher] advantages vary within a row (max deviation {float(spread):.3g}); "
                "the reliability correlation uses the masked row mean",
                flush=True,
            )

        uids = batch.non_tensor_batch.get("uid", batch.non_tensor_batch.get("traj_uid", None))
        real = torch.ones_like(row_adv, dtype=torch.bool)
        padding = batch.batch.get(PADDING_ROW_KEY, None)
        if padding is not None:
            real &= ~padding.reshape(-1).to(torch.bool)
        informative = torch.zeros_like(real)
        if uids is not None:
            by_uid = {}
            for i, u in enumerate(np.asarray(uids).reshape(-1).tolist()):
                if bool(real[i]):
                    by_uid.setdefault(u, []).append(i)
            for rows in by_uid.values():
                vals = [float(row_adv[i]) for i in rows]
                if len(vals) > 1 and max(vals) - min(vals) > 0:
                    for i in rows:
                        informative[i] = True

        # A DENSE group index, because the reliability statistic centres the
        # support score within the prompt group and a group's rollouts land in
        # different micro-batches and on different ranks: the accumulator pools
        # them by this id and all-reduces, which is exact where centring a local
        # fragment would not be. Dense and batch-local -- it names a prompt
        # within this step and nothing beyond it.
        group_id = torch.full_like(row_adv, -1, dtype=torch.long)
        if uids is not None:
            order = {}
            for i, u in enumerate(np.asarray(uids).reshape(-1).tolist()):
                if not bool(real[i]):
                    continue
                group_id[i] = order.setdefault(u, len(order))

        batch.batch["adv_row_value"] = row_adv
        batch.batch["adv_group_informative"] = informative
        batch.batch["adv_group_id"] = group_id
        return batch

    def _data_metrics(self, batch: DataProto) -> dict:
        """The full advantage-bearing statistics, which this arm's batch carries.

        Pure OPD reports :func:`compute_opd_data_metrics`, an advantage-free
        variant, precisely because it never computes advantages. Here they exist,
        so there is no reason to report less than the standard PPO loop does.
        """
        metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
        metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
        return metrics
