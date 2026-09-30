#!/usr/bin/env bash
# OPD + GRPO ON THE VALUE ARM'S RECIPE, TRAJECTORY-NORMALISED POLICY GRADIENT: the 2x2's
# normalisation control -- the cell the main result is read against.
#
# THE 2x2 (2026-09-30, the user: trajectory normalisation (a) is the paper's main one, the token
# normalisation (b) the control beside it). One launcher and lock set per cell:
#                         (b) token-normalised PG                     (a) trajectory-normalised PG
#   OPD + GRPO            run_multitask_grpo_v2recipe_qwen3.sh        THIS SCRIPT
#   progress_value_gae    run_multitask_progress_value_gae_qwen3.sh   run_multitask_progress_value_gae_trajnorm_qwen3.sh
# The value estimator's effect is progress_value_gae (a) against THIS cell: the same
# normalisation, so the comparison is not also a comparison of normalisations.
#
# THIS CELL. run_multitask_grpo_v2recipe_qwen3.sh (outcome GRPO on the value arm's recipe, the
# progress_value keys as records only) with the policy-gradient term normalised per trajectory,
# exactly as the value arm's (a) cell does it:
#   actor_rollout_ref.actor.pg_loss_norm=trajectory
#       each task's policy gradient divided by N_d * L_d (its real trajectories in the step times a
#       fixed reference length) instead of by the response tokens the task brought to the step;
#       turns and tokens summed inside a trajectory. Only the policy gradient: the teacher KL keeps
#       the token weights. verl/trainer/ppo/task_loss_weights.py.
#   actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}
#       L_d: beta-mirror v2's (31e7f5e) mean response tokens per trajectory over steps 1-300, the
#       same lengths as the value arm's (a) cell, fixed and never read off the batch.
#   algorithm.progress_value.allow_missing_table=False  (restated; not read by grpo)
#   algorithm.progress_value.records.enable=True  (restated: the group records, the GiGPO and value
#       shadows as in the script below, token totals read with this cell's policy-gradient weight)
#
# RUN_TAG grpo_v2recipe_trajnorm; lock expected_multitask_grpo_v2recipe_trajnorm_config.yaml:
#   bash examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_trajnorm_qwen3.sh
#
# WHAT TO WATCH: task_loss/pg_weight_ratio/<task> (T_d / (N_d L_d)), task_loss/pg_trajectories/<task>
# (N_d, 120 on this recipe), actor/pg_loss_weighted.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Kept by the scripts below (they only set RUN_TAG when unset).
export RUN_TAG="${RUN_TAG:-grpo_v2recipe_trajnorm}"
echo "grpo_v2recipe_trajnorm: run_tag=$RUN_TAG"
exec bash "$_HERE/run_multitask_grpo_v2recipe_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_grpo_v2recipe_trajnorm_config.yaml \
  actor_rollout_ref.actor.pg_loss_norm=trajectory \
  "actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}" \
  algorithm.progress_value.allow_missing_table=False \
  algorithm.progress_value.records.enable=True \
  "$@"
