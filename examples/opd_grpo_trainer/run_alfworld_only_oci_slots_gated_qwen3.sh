#!/usr/bin/env bash
# THE TEN-SLOT LAYOUT ON ALFWORLD ONLY, THIRD RUN: the second run with the
# foreign row's clip reversed, and nothing else.
#
# WHAT THE SECOND RUN SAID. Its document side was an improvement -- with the
# ordinary clip the entropy ran BELOW the first run's for 32 steps on the same
# games -- and its foreign side was the cost. A negative advantage under the
# ordinary clip flattens everything below 1-eps, so the tokens that mark the
# swapped goal (rho well under 1 when re-scored under the real goal) trained
# nothing, while the vocabulary every alfworld rollout writes (rho about 1:
# `go to`, `take`, `<think>`, the receptacle names) was pushed down at full
# weight. Steps 41-50, same games: entropy 1.132 against 0.858, training
# success 0.436 against 0.533, and the entropy gap opened exactly when the
# foreign slot's usage tripled.
#
# THE CHANGE. algorithm.oci_slots.special_loss=gated: for negative rows only,
#
#     loss = -A * min(rho, 1-eps)
#
# so the gradient survives only below the band and its weight is rho itself.
# The foreign row then suppresses what distinguishes the wrong goal and leaves
# the shared vocabulary alone. Positive rows keep the ordinary clip, so this
# run differs from the second in the foreign row and in nothing else.
#
# NOT A TRUST REGION. The clip's bound is being used as a difference filter, so
# there is no monotonic-improvement argument behind it; it is an arm. Read
# oci/shaping/gated_neg_trained_frac to see whether the gate left any gradient
# at all -- at zero the foreign row has become the virtual floor.
#
# Scoring afterwards, as for the runs before it:
#
#   VAL_ONLY=1 ROLLOUT_ASYNC_GENERATE=0 \
#   VAL_CKPT=$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_gated_alfworld_only/global_step_150 \
#       bash examples/opd_grpo_trainer/run_alfworld_only_oci_slots_gated_qwen3.sh
set -euo pipefail

export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_alfworld_only_oci_slots_ppo_qwen3.sh" \
    algorithm.oci_slots.special_loss=gated \
    trainer.experiment_name="opd_grpo_alfworld_only_oci_slots_gated_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_gated_alfworld_only$RUN_TAG_SUFFIX" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_alfworld_only_oci_slots_gated_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_alfworld_only_oci_slots_gated_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_alfworld_only_oci_slots_gated_config.yaml \
    "$@"
