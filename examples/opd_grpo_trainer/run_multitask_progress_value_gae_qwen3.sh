#!/usr/bin/env bash
# PROGRESS_VALUE_GAE: a per-turn advantage from a value table of the state, by TD errors and GAE,
# in place of outcome GRPO. See verl/trainer/ppo/progress_value.py.
#
#   J_H = E[gamma^T R]      win within the task's turn cap (R = 1 iff episode_rewards > 0)
#   V(z_t)                  a table over the state BEFORE action t (ALFWorld: task type, progress
#                           k, turns since k rose, turns left; WebShop: k, stag, rem -- the
#                           buy-now score is off: it queries the evaluator; Search: k, rem),
#                           shrunk cell -> (task, type, rem) -> task, discounted counts,
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
#   actor_rollout_ref.actor.pg_loss_norm=token       the policy gradient divided by the response
#                                                    tokens each task brought to the step, like every
#                                                    other term: the default and v2's, spelled out
#                                                    because it is the 2x2's column (b) (below)
# v2's other progress_rank keys (the Beta strength rule, sat) still reach the config and are not
# read with progress_rank off; the lock does not pin them. The progress_rank launcher's banner
# prints v2's rho -- ignore it.
#
# THE 2x2 (2026-09-30, the user: trajectory normalisation (a) is the paper's main one, the token
# normalisation (b) the control beside it). One launcher and lock set per cell:
#                         (b) token-normalised PG                     (a) trajectory-normalised PG
#   OPD + GRPO            run_multitask_grpo_v2recipe_qwen3.sh        run_multitask_grpo_v2recipe_trajnorm_qwen3.sh
#   progress_value_gae    THIS SCRIPT                                 run_multitask_progress_value_gae_trajnorm_qwen3.sh
# The other three cells wrap this script (their "$@" comes after its keys, so the (a) cells'
# pg_loss_norm=trajectory replaces the token here), and every cell's lock pins its column: a
# trajectory override reaching THIS script's lock is refused at launch.
#
# ONE LOCK PER LAM, chosen by the value: LAM=1.0 -> expected_multitask_progress_value_gae_lam1.0_
# config.yaml, LAM=0.9 -> ..._lam0.9_config.yaml; a LAM with no lock refuses to start. Each LAM
# has its own RUN_TAG (progress_value_gae_lam1p0, progress_value_gae_lam0p9), so its own
# checkpoint directory and wandb names:
#   bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh
#   LAM=0.9 bash examples/opd_grpo_trainer/run_multitask_progress_value_gae_qwen3.sh
#
# RESUMING. The value table is saved beside every checkpoint (global_step_N/progress_value_state.json)
# and a resume must find it: algorithm.progress_value.allow_missing_table=False is pinned, so a
# checkpoint without the table (incomplete, or another arm's) refuses to start instead of scoring
# on an empty table. A table saved under other turn caps, k definitions, table keys or feature /
# reward schema is refused too. Starting this arm from another arm's checkpoint on purpose is a
# warm start, allow_missing_table=True, which the lock refuses: a different run, so its own lock
# (or EXPECTED_CONFIG_WAIVE=algorithm.progress_value.allow_missing_table, said out loud).
#
# ANALYSIS RECORDS (algorithm.progress_value.records, every key spelled out; observation only --
# the batch, the advantages, the table and the RNG states are the same with them off): one JSON line
# per group per step in <default_local_dir>/progress_value_groups/step<N>.jsonl (records.dir, a host
# knob, moves them), and records/<task>/<kind>/<component>/{row,traj_sum,token}_* metrics of the
# advantage, its value terms and the GRPO and GiGPO shadows on the same rollouts
# (verl/trainer/ppo/rollout_records.py). The traj/* and think-block metrics progress_rank reported
# come back with them -- WebShop's failure progress as traj/webshop/fail_progress_session: progress_k's
# WebShop count here is the session count (webshop_k=session), v2's and every earlier arm's the legacy
# one, which is not recorded here, so the key names the count and is not comparable with theirs. The
# wrappers of the 2x2 inherit them (and restate enable).
#
# WHAT TO WATCH (every step, per task): progress_value/<task>/fallback_share (1 on the first
# step, then 0), calibration (mean V - mean target on the batch), a_rl_success / a_rl_failure,
# failure_pos_share, mean_abs_a_fmt, table_cells / table_mass, delta_progress / delta_stagnant;
# progress_value/webshop/capped_won must stay 0 (a win on a goal flagged unpayable; warned);
# records/failed must stay 0 (a failed record costs the step's records only; the traceback is
# printed).
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
  actor_rollout_ref.actor.pg_loss_norm=token \
  algorithm.progress_rank.enable=False \
  algorithm.progress_rank.alfworld_k=milestone_arrive \
  algorithm.progress_rank.search_k=evidence_answered \
  algorithm.progress_rank.webshop_k=session \
  algorithm.progress_value.enable=True \
  algorithm.progress_value.allow_missing_table=False \
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
  algorithm.progress_value.records.enable=True \
  algorithm.progress_value.records.shadow_grpo=True \
  algorithm.progress_value.records.shadow_gigpo=True \
  algorithm.progress_value.records.shadow_value=True \
  algorithm.progress_value.records.per_turn=True \
  "$@"
