#!/usr/bin/env bash
# THE TEN-SLOT LAYOUT ON ALFWORLD ONLY, SECOND RUN: the first run's layout with
# the two things its analysis changed, and nothing else.
#
#   1. The special rows' policy gradient is the ordinary clipped objective on
#      the plain prompt (algorithm.oci_slots.special_loss=ppo). The first run
#      shaped it with LUFFY's f(rho), and measured the document rows at rho ~ 1
#      on nine tokens in ten -- where f(rho) is a constant 0.083 on the document
#      side only. GRPO's advantages sum to zero across a group, so every rescued
#      stuck group trained a net push-down of the seven failures (-37.4 against
#      +3.1 in advantage-rows), and the arm's entropy left the baseline's at
#      step 8. The clip keeps the balance; below 1-eps it still carries only
#      -A*rho for a positive advantage and nothing for a negative one.
#   2. The foreign slot is this game's observation with another train game's
#      goal (algorithm.oci_slots.foreign_task=alfworld). The WebShop prompt's
#      rows sat at rho = e^-40, trained nothing, and acted only through the
#      group statistic. Measured before this run by
#      run_alfworld_oci_slots_probe_qwen3.sh.
#
# WHAT IS DELIBERATELY NOT CHANGED, so the comparison to the first run and to
# the baseline stays one of loss and slot text: env.rollout.n=10 and with it
# the mini-batch normalisation (ppo_mini_batch_size is scaled by the GENERATED
# n, so this arm takes ~20% fewer optimizer steps per training step than the
# baseline -- a handicap in the arm's disfavour, left in place until the loss
# question is settled).
#
# TWO OPERATIONAL CHANGES, neither of which touches the objective:
#   ppo_micro_batch_size_per_gpu 5 -> 10. The update is 43% of a 475 s step at
#     5 with an MFU of 0.15; the first run peaked at 75-80 GiB of a 95 GiB card
#     and took one allocator retry in 150 steps. Same adjust_batch divisor
#     (lcm(30,30,30) = lcm(30,30,15) = 30), and the loss sums per-row terms
#     whose weights are computed over the whole step, so regrouping rows changes
#     only the order of the arithmetic. If it OOMs anyway, resume_mode=auto
#     restarts from the last checkpoint.
#   save_freq 25 -> 10, so an OOM costs at most ten steps. 21 GB each, 30
#     checkpoints over 300 steps, against 14 TB free on fuji.
#
# It execs the first run's launcher and overrides only the keys above plus the
# names. Scoring afterwards, as for the first run:
#
#   VAL_ONLY=1 ROLLOUT_ASYNC_GENERATE=0 \
#   VAL_CKPT=$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_ppo_alfworld_only/global_step_150 \
#       bash examples/opd_grpo_trainer/run_alfworld_only_oci_slots_ppo_qwen3.sh
set -euo pipefail

export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_alfworld_only_oci_slots_qwen3.sh" \
    algorithm.oci_slots.special_loss=ppo \
    algorithm.oci_slots.foreign_task=alfworld \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    trainer.save_freq=10 \
    trainer.experiment_name="opd_grpo_alfworld_only_oci_slots_ppo_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_ppo_alfworld_only$RUN_TAG_SUFFIX" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_alfworld_only_oci_slots_ppo_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_alfworld_only_oci_slots_ppo_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_alfworld_only_oci_slots_ppo_config.yaml \
    "$@"
