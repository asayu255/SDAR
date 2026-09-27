#!/usr/bin/env bash
# OPD+GiGPO: the control recipe with GRPO's advantage replaced by GiGPO's (2505.10978).
#
# A BASELINE for the multitask table, built on exactly the recipe the proposed method
# runs on (run_multitask_cross_teacher_klw_control_qwen3.sh: per-task teacher KL at
# 0.01, per-task loss normalisation, the invalid-action penalty), so that it differs
#   - from the beta_mirror arm only in how the advantage is formed: GiGPO's step-level
#     term over anchor-state groups instead of (a)/sat inside tied groups;
#   - from the control only in the estimator.
# GiGPO's step term also works inside all-success groups (the discounted return
# favours reaching success sooner), which is why it is the baseline sat has to beat.
#
#   algorithm.adv_estimator=gigpo     episode term + step_advantage_w x step term (Eq. 8)
#   algorithm.gamma=0.95              discount for the per-turn returns (Eq. 5); read
#                                     nowhere else on this path
#   algorithm.gigpo.step_advantage_w=1.0, mode=mean_std_norm (as GRPO's std division on
#                                     the control), enable_similarity=False (exact anchors)
# The per-turn returns are computed by OPDRayTrainer._attach_gigpo_step_returns right
# after the rollout (this loop did not compute them before, so gigpo could not run here).
# The sampling and gradient-path knobs the lock pins (speculative decoding, micro batch
# 5, gradient checkpointing off) are set here, as on the beta_mirror wrapper, so this
# script satisfies its own lock; a host launcher adds host knobs only. No in-training
# validation (test_freq=-1): checkpoints are scored afterwards like every arm.
#
# Lock: expected_multitask_opd_gigpo_config.yaml (no waiver needed).
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Its own checkpoint directory and wandb names: the control's plus _opd_gigpo.
export RUN_TAG="${RUN_TAG:-opd_gigpo}"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_opd_gigpo_config.yaml \
  algorithm.adv_estimator=gigpo \
  algorithm.gamma=0.95 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_std_norm \
  algorithm.gigpo.enable_similarity=False \
  trainer.test_freq=-1 \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=ngram" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=4" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_min=2" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_max=5" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.acceptance_method=rejection_sampler" \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
  "$@"
