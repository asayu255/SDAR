#!/usr/bin/env bash
# PROGRESS_VALUE_GAE WITH THE TRAJECTORY-NORMALISED POLICY GRADIENT: the 2x2's main cell.
#
# THE 2x2 (2026-09-30, the user: trajectory normalisation (a) is the paper's main one, the token
# normalisation (b) the control beside it). One launcher and lock set per cell:
#                         (b) token-normalised PG                     (a) trajectory-normalised PG
#   OPD + GRPO            run_multitask_grpo_v2recipe_qwen3.sh        run_multitask_grpo_v2recipe_trajnorm_qwen3.sh
#   progress_value_gae    run_multitask_progress_value_gae_qwen3.sh   THIS SCRIPT
# A row compares the normalisations under one advantage, a column the advantages under one
# normalisation. The main result is THIS cell against GRPO (a), the same normalisation; against
# GRPO (b) alone the two effects cannot be told apart.
#
# THIS CELL. run_multitask_progress_value_gae_qwen3.sh -- its recipe, its LAM, every progress_value
# key and its host knobs -- with the policy-gradient term normalised per trajectory:
#   actor_rollout_ref.actor.pg_loss_norm=trajectory
#       each task's policy gradient divided by N_d * L_d (its real trajectories in the step times a
#       fixed reference length) instead of by the response tokens the task brought to the step;
#       turns and tokens summed inside a trajectory, never divided by its own length. Only the
#       policy gradient: the teacher KL keeps the token weights. verl/trainer/ppo/task_loss_weights.py.
#       It replaces the token the script below spells out (this script's keys come after its).
#   actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}
#       L_d: beta-mirror v2's (31e7f5e) mean response tokens per trajectory over steps 1-300 (v1
#       gives 4690 / 1741 / 312), so each task's policy gradient keeps on average the size the
#       token normalisation gave it, against the teacher term too. Fixed here, never read off the
#       batch. The price: the tasks keep (b)'s average relative weights (1/L_d), not equal weight
#       per unit of success.
#   algorithm.progress_value.allow_missing_table=False
#       restated from the script below: a resume must find the value table.
#   algorithm.progress_value.records.enable=True
#       restated from the script below (which spells out every records key): the group records and
#       records/* metrics, token_total / token_abs_total read with this cell's actual policy-gradient
#       weight (task_pg_loss_weight). Observation only.
#
# ONE LOCK PER LAM, chosen by the value as the script below does: LAM=1.0 ->
# expected_multitask_progress_value_gae_trajnorm_lam1.0_config.yaml, LAM=0.9 -> ..._lam0.9_config.yaml;
# a LAM with no lock refuses to start. RUN_TAG progress_value_gae_lam1p0_trajnorm /
# progress_value_gae_lam0p9_trajnorm, so its own checkpoint directory and wandb names:
#   bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_trajnorm_qwen3.sh
#   LAM=0.9 bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_trajnorm_qwen3.sh
#
# WHAT TO WATCH, beside the script below's list: task_loss/pg_weight_ratio/<task> (T_d / (N_d L_d),
# this step's mean tokens per trajectory over L_d: 1 where the two normalisations agree, and its
# drift is how far the token normalisation would have moved the task's policy gradient),
# task_loss/pg_trajectories/<task> (N_d: 120 = 15 prompts x 8 rollouts on this recipe),
# actor/pg_loss_weighted, task_loss/optimizer_steps (the optimizer steps the batch became, rows / 60:
# the mean optimizer step's policy gradient is what (a) fixes, and this count -- the same factor in
# the (b) cells, 115 falling to 60 over v2's run -- is what it does not).
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAM="${LAM:-1.0}"
[ -f "$_HERE/expected_multitask_progress_value_gae_trajnorm_lam${LAM}_config.yaml" ] || {
  echo "progress_value_gae_trajnorm: no lock for LAM=$LAM (expected_multitask_progress_value_gae_trajnorm_lam${LAM}_config.yaml)" >&2
  exit 1
}
# The script below reads LAM from the environment; one value for both.
export LAM
# Kept by the script below (it only sets RUN_TAG when unset).
export RUN_TAG="${RUN_TAG:-progress_value_gae_lam${LAM//./p}_trajnorm}"
echo "progress_value_gae_trajnorm: lam=$LAM run_tag=$RUN_TAG"
exec bash "$_HERE/run_multitask_progress_value_gae_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_progress_value_gae_trajnorm_lam${LAM}_config.yaml \
  actor_rollout_ref.actor.pg_loss_norm=trajectory \
  "actor_rollout_ref.actor.pg_ref_tokens={alfworld:4650,webshop:2320,search:310}" \
  algorithm.progress_value.allow_missing_table=False \
  algorithm.progress_value.records.enable=True \
  "$@"
