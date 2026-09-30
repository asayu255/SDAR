#!/usr/bin/env bash
# OPD + GRPO ON THE VALUE ARM'S RECIPE, TOKEN-NORMALISED POLICY GRADIENT: the 2x2's control cell.
#
# THE 2x2 (2026-09-30, the user: trajectory normalisation (a) is the paper's main one, the token
# normalisation (b) the control beside it). One launcher and lock set per cell:
#                         (b) token-normalised PG                     (a) trajectory-normalised PG
#   OPD + GRPO            THIS SCRIPT                                 run_multitask_grpo_v2recipe_trajnorm_qwen3.sh
#   progress_value_gae    run_multitask_progress_value_gae_qwen3.sh   run_multitask_progress_value_gae_trajnorm_qwen3.sh
# A row compares the normalisations under one advantage, a column the advantages under one
# normalisation.
#
# THIS CELL. run_multitask_progress_value_gae_qwen3.sh with the advantage put back:
#   algorithm.adv_estimator=grpo
#       outcome GRPO: compute_advantage's grpo branch, the z-score within the prompt group of the
#       rows' scores with the invalid-action penalty added, as on beta-mirror v2's parent arms.
# Everything else is that script's, and so beta-mirror v2's recipe and host knobs:
#   progress_rank off, its counter keys kept (alfworld_k milestone_arrive, search_k
#   evidence_answered, webshop_k session: the environment's own session count);
#   progress_value.enable=True, RECORDS ONLY beside grpo -- the environment managers count and the
#   rollout loop writes the pv_* columns, and nothing reaches the advantage
#   (opd_grpo_ray_trainer.check_progress_value_config). Every other progress_value key stays as that
#   script sets it, so a records-only shadow of the value estimator is this arm's own table config;
#   gamma 1.0 / lam 1.0 are not read by grpo and are pinned so the lock has one value.
#   algorithm.progress_value.allow_missing_table=False is restated here (not read by grpo: there is
#   no training table to resume).
#   algorithm.progress_value.records.enable=True, restated here (that script spells out every
#   records key): the group records and records/* metrics of this arm's GRPO advantage, beside the
#   GiGPO shadow and -- records.shadow_value -- the value estimator's advantage on these same
#   rollouts, from an observation-only table of that script's config (its own object, saved with the
#   records, never in the checkpoint). records/shadow_grpo/max_abs_diff_adv reads 0 here: the shadow
#   GRPO is this arm's own advantage.
#
# LAM IS NOT A KNOB HERE: fixed at 1.0 whatever the environment says (grpo reads no lambda, and the
# lock pins it).
#
# RUN_TAG grpo_v2recipe; lock expected_multitask_grpo_v2recipe_config.yaml:
#   bash examples/opd_grpo_trainer/run_multitask_grpo_v2recipe_qwen3.sh
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAM=1.0
# Kept by the scripts below (they only set RUN_TAG when unset).
export RUN_TAG="${RUN_TAG:-grpo_v2recipe}"
echo "grpo_v2recipe: run_tag=$RUN_TAG"
exec bash "$_HERE/run_multitask_progress_value_gae_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_grpo_v2recipe_config.yaml \
  algorithm.adv_estimator=grpo \
  algorithm.progress_value.allow_missing_table=False \
  algorithm.progress_value.records.enable=True \
  "$@"
