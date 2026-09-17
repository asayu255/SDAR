#!/usr/bin/env bash
# (a) ON THE 3-TASK CONTROL: rank each stuck group's rollouts by progress.
#
# THE ARM. The xt1 control recipe (run_multitask_cross_teacher_klw_control_qwen3.sh)
# with exactly one thing added: algorithm.progress_rank. In a group where all
# eight rollouts failed and they got different distances along the task's correct
# sequence, the further ones are pushed up and the shorter ones down:
#
#     A_i += c_task * (k_i - mean k) / K        (zero-sum over the group's samples)
#
# k is counted by the environment managers for every rollout (ALFWorld: the task
# type's milestones; WebShop goal record; Search: a returned result held the
# answer). Nothing is generated, no row is added, the reward is untouched. (a) is
# ADDED to the ordinary advantage, so the format penalty keeps its full size where
# it acts.
#
# c IS SET PER TASK, EVERY STEP, from an EMA: the added token-mean |A| is RHO times
# the EMA of that task's ordinary token-mean |A|. RHO is the only knob, and it is
# the same for all three tasks. See verl/trainer/ppo/progress_rank.py.
#
# STAGE 1, IN ORDER.
#   1. RHO=0. c is 0 on every step, so the advantages are bit-identical to control
#      (tests/oci/test_progress_rank.py holds that). What the run checks is the
#      part a unit test cannot: that the counts reach the batch on the real stack
#      and that progress_rank/<task>/stuck_fired and fired_token_share are sane.
#      A few steps are enough:
#        RHO=0 RUN_TAG=pr_identity \
#        EXPECTED_CONFIG_WAIVE="trainer.total_training_steps algorithm.progress_rank.rho" \
#        bash examples/opd_grpo_trainer/run_multitask_progress_rank_qwen3.sh \
#          trainer.total_training_steps=5 trainer.save_freq=-1
#   2. RHO=0.05, to step 150, beside a control run from the same commit. xt1 is NOT
#      that control: it ran before the ALFWorld game-order fix (5a62ed9, 2026-09-14),
#      so its game order differs from this checkout's. Re-run the control launcher
#      at the same commit, same host type and seed, for 150 steps.
#
# WHAT TO WATCH (all logged every step).
#   progress_rank/<task>/share_of_ema          = RHO unless capped
#   progress_rank/<task>/capped, c_uncapped, c_cap
#                                              how often the cap binds, and by how much
#   progress_rank/<task>/inject_down_over_up   (a)'s push-down : push-up. Control's
#                                              ordinary update reads 1.08; the
#                                              ten-slot runs failed at 12:1
#   progress_rank/<task>/stuck_mixed           stuck groups whose row SCORES differ
#                                              in format (judged on the scores: an
#                                              all -0.1 group has rounding |A| only)
#   progress_rank/<task>/inject_share_mixed,   how much of (a) lands where the format
#     inject_up_invalid                        penalty acts, and on its invalid rows
#   shadow/progpo/<task>/q_fail, fired         what ProGPO's gate and coverage would
#   shadow/coverage/<task>/*                   do on the same groups (never applied)
#   actor entropy                              stop criterion: 2x control within
#                                              the first 30 steps
#   Every step also writes one record per group to
#   <default_local_dir>/progress_rank_groups/step<N>.jsonl; read them with
#   scripts/report_progress_groups.py. A mass probe at step 150 on this arm and on
#   control (grad_probe.mode=mass) reads (a)'s share of the loss directly.
#
# NO IN-TRAINING VALIDATION (test_freq=-1): _validate() runs before
# _save_checkpoint(), so a failed validation loses the step. Score the saved
# checkpoints afterwards with VAL_ONLY=1 VAL_CKPT=... on the control launcher.
#
# Host knobs (ROLLOUT_PUMP_TRAINING, ROLLOUT_PREFETCH_LOGPROB,
# ROLLOUT_PREFETCH_TEACHER, ...) are read by the control launcher; set them for
# the host you launch on.
set -euo pipefail
RHO="${RHO:-0.05}"
# One checkpoint directory and one wandb run per RHO unless RUN_TAG says otherwise;
# the control launcher derives every name from it.
export RUN_TAG="${RUN_TAG:-progress_rank_rho${RHO//./p}}"
echo "progress_rank: rho=$RHO run_tag=$RUN_TAG"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_multitask_cross_teacher_klw_control_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_progress_rank_config.yaml \
  algorithm.progress_rank.enable=True \
  algorithm.progress_rank.rho="$RHO" \
  trainer.test_freq=-1 \
  "$@"
