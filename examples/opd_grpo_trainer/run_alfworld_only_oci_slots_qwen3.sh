#!/usr/bin/env bash
# OPD + GRPO on ALFWORLD ONLY, Qwen3-1.7B, WITH THE TEN-SLOT LAYOUT: ten rollouts
# per group, eight of them trained, and the eighth chosen by what the group
# turned out to be.
#
#   slots 0-6  plain      ordinary rollouts, always trained
#   slot  7    reserve    ordinary too; trained unless a special slot replaces it
#   slot  8    document   shown this game's own walkthrough, with the per-turn
#                         line naming the step it still owes
#   slot  9    foreign    shown another task's prompt instead of its own
#
#   live      -> plain + reserve      (the same eight rollouts the baseline trains)
#   stuck     -> plain + document     when the document rollout solved the game
#   saturated -> plain + foreign      when the foreign rollout failed
#
# EIGHT TRAJECTORIES ARE TRAINED EITHER WAY. That is what the two extra rollouts
# buy: the group size, the advantage's denominator and the row count of an
# optimizer step stay the baseline's, so the arm's only difference is which
# eighth rollout a degenerate group has. The unused two are dropped before the
# batch is padded (verl/trainer/ppo/oci_slots.py).
#
# ITS BASELINE IS THE ALFWORLD-ONLY CONTROL, WHICH IT EXECS. Same objective, same
# data, same optimiser, same 300 steps, training only -- steps are scored
# afterwards as separate val-only runs, exactly as the baseline is:
#
#   VAL_ONLY=1 ROLLOUT_ASYNC_GENERATE=0 \
#   VAL_CKPT=$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_alfworld_only/global_step_300 \
#       bash examples/opd_grpo_trainer/run_alfworld_only_oci_slots_qwen3.sh
#
# THE BASELINE HAS TO BE RE-RUN ON A CHECKOUT WITH COMMIT 5a62ed9. Before it,
# ALFWorld's game list came out in os.walk order, so which games a seed draws
# depended on the host's filesystem; the existing alfworld-only baseline
# (2026-09-13, tamago) therefore played a different set of games from anything
# this arm can draw now, and comparing the two curves would be comparing game
# sets. Nothing in this script can fix that -- it is a note about which numbers
# the arm may be held against.
set -euo pipefail

# The same expression the scripts below derive, set here too because this one
# builds its names before exec'ing them.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

# Mutually exclusive with the other privileged inputs: two blocks in one
# observation are two interventions, and the strip that makes rho meaningful
# removes exactly one replacement per row.
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_alfworld_only_cross_teacher_klw_control_qwen3.sh" \
    algorithm.oci_slots.enable=True \
    'algorithm.oci_slots.tasks=[alfworld]' \
    algorithm.oci_slots.doc_mode=walkthrough_stepwise \
    algorithm.oci_slots.foreign_task=webshop \
    algorithm.oci_slots.gamma=0.1 \
    algorithm.oci_slots.opd_on_special=False \
    env.rollout.n=10 \
    algorithm.oci_sat.enable=False \
    algorithm.oci_floor.enable=False \
    algorithm.compute_mean_std_cross_steps=True \
    trainer.experiment_name="opd_grpo_alfworld_only_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_alfworld_only$RUN_TAG_SUFFIX" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_alfworld_only_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_alfworld_only_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_alfworld_only_oci_slots_config.yaml \
    "$@"
