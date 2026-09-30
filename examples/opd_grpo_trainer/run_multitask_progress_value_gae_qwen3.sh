#!/usr/bin/env bash
# PROGRESS_VALUE_GAE: a per-turn advantage from a value table of the state, by TD errors and GAE,
# in place of outcome GRPO. See verl/trainer/ppo/progress_value.py.
#
#   J_H = E[gamma^T R]      win within the task's turn cap (R = 1 iff episode_rewards > 0)
#   V(z_t)                  a table over the state BEFORE action t (ALFWorld: task type, progress
#                           k, turns since k rose, turns left; WebShop: k, stag, rem; Search: k,
#                           rem), shrunk cell -> (task, type, rem) -> task, discounted counts,
#                           frozen within a step and saved beside the checkpoint
#   delta_t = gamma V(z_{t+1}) - V(z_t), V(z_T) = R;   A_t = sum_l (gamma lam)^l delta_{t+l}
#   advantage = 2.0 * A_t + the format term (the z-score of "invalid" over the group's turn rows,
#                           in groups whose rollouts all have the same R: what control's GRPO
#                           gives there)
#
# THE ARM. The beta-mirror v2 launcher (run_multitask_progress_rank_beta_mirror_v2_qwen3.sh: its
# recipe, the sampling and gradient-path knobs its lock pins, and the host knobs of the launchers
# below it) with the advantage replaced:
#   algorithm.adv_estimator=progress_value_gae
#   algorithm.gamma=1.0, algorithm.lam=$LAM          LAM 1.0 (default): A_t = R - V(z_t), the
#                                                    table only a baseline; LAM=0.9: the TD
#                                                    residuals carry the credit
#   algorithm.progress_rank.enable=False             no (a), no sat: nothing added on top. The
#                                                    counters its keys choose are still read:
#                                                    alfworld_k=milestone_arrive and
#                                                    search_k=evidence_answered (v2's), and
#   algorithm.progress_rank.webshop_k=session        WebShop's count read off the environment's
#                                                    own session (no option over-count)
#   algorithm.progress_value.*                       every key spelled out (= the defaults), so
#                                                    this script is the arm's whole definition
# v2's other progress_rank keys (the Beta strength rule, sat) still reach the config and are not
# read with progress_rank off; the lock does not pin them. The progress_rank launcher's banner
# prints v2's rho -- ignore it.
#
# ONE LOCK PER LAM, chosen by the value: LAM=1.0 -> expected_multitask_progress_value_gae_lam1.0_
# config.yaml, LAM=0.9 -> ..._lam0.9_config.yaml; a LAM with no lock refuses to start. Each LAM
# has its own RUN_TAG (progress_value_gae_lam1p0, progress_value_gae_lam0p9), so its own
# checkpoint directory and wandb names:
#   bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh
#   LAM=0.9 bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh
#
# WHAT TO WATCH (every step, per task): progress_value/<task>/fallback_share (1 on the first
# step, then 0), calibration (mean V - mean target on the batch), a_rl_success / a_rl_failure,
# failure_pos_share, mean_abs_a_fmt, table_cells / table_mass, delta_progress / delta_stagnant.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAM="${LAM:-1.0}"
[ -f "$_HERE/expected_multitask_progress_value_gae_lam${LAM}_config.yaml" ] || {
  echo "progress_value_gae: no lock for LAM=$LAM (expected_multitask_progress_value_gae_lam${LAM}_config.yaml)" >&2
  exit 1
}
# One checkpoint directory and one wandb run per LAM unless RUN_TAG says otherwise; the control
# launcher derives every name from it (the progress_rank launcher keeps a RUN_TAG already set).
export RUN_TAG="${RUN_TAG:-progress_value_gae_lam${LAM//./p}}"
echo "progress_value_gae: lam=$LAM run_tag=$RUN_TAG"
exec bash "$_HERE/run_multitask_progress_rank_beta_mirror_v2_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_progress_value_gae_lam${LAM}_config.yaml \
  algorithm.adv_estimator=progress_value_gae \
  algorithm.gamma=1.0 \
  algorithm.lam="$LAM" \
  algorithm.progress_rank.enable=False \
  algorithm.progress_rank.alfworld_k=milestone_arrive \
  algorithm.progress_rank.search_k=evidence_answered \
  algorithm.progress_rank.webshop_k=session \
  algorithm.progress_value.enable=True \
  algorithm.progress_value.prefix_discount=False \
  algorithm.progress_value.eta=1.0 \
  algorithm.progress_value.adv_scale=2.0 \
  algorithm.progress_value.n0=8.0 \
  algorithm.progress_value.retention=0.9 \
  "algorithm.progress_value.features.alfworld=[type,k,stag,rem]" \
  "algorithm.progress_value.features.webshop=[k,stag,rem]" \
  "algorithm.progress_value.features.search=[k,rem]" \
  "algorithm.progress_value.rem_buckets=[0.8,0.6,0.4,0.2,0.1]" \
  "algorithm.progress_value.stag_buckets=[1,3,6,10,20]" \
  algorithm.progress_value.format_scope=tied \
  algorithm.progress_value.format_coef=1.0 \
  "$@"
