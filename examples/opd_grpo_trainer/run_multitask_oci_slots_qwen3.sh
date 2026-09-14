#!/usr/bin/env bash
# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH THE TEN-SLOT LAYOUT: ten rollouts per alfworld group, eight of them
# trained, and the eighth is chosen by what the group turned out to be.
#
# ---------------------------------------------------------------------------
# WHAT THE ARM DOES. A GRPO advantage is a trajectory's return minus its group's
# mean, so a group whose rollouts all agree contributes exactly zero -- correctly,
# because the baseline equals the return. This arm gives such a group one
# rollout that disagrees, and generates it in the same pass:
#
#   slots 0-6  plain      ordinary rollouts, always trained
#   slot  7    reserve    ordinary too; trained unless a special slot replaces it
#   slot  8    document   shown this game's own walkthrough, with the per-turn
#                         line naming the step it still owes
#   slot  9    foreign    shown another task's prompt instead of its own
#
#   live      -> plain + reserve      (the same eight rollouts control trains)
#   stuck     -> plain + document     when the document rollout solved the game
#   saturated -> plain + foreign      when the foreign rollout failed
#
# EIGHT TRAJECTORIES ARE TRAINED EITHER WAY. That is what the two extra rollouts
# buy: the group size, the advantage's denominator and the row count of an
# optimizer step stay what control's are, so the arm's only difference from
# control is which eighth rollout a degenerate group has. The unused two are
# dropped before the batch is padded (verl/trainer/ppo/oci_slots.py).
#
# WHY THE FOREIGN SLOT IS ANOTHER TASK'S PROMPT. Five failure-inducing documents
# were measured on saturated groups at the control's step 300 -- the only place
# such a slot is used -- and none reached 12%: the true path 0/25, misdirection
# 2/26, a 50-turn tour in front of the true path 3/26, and the same tour with the
# per-turn progress line 3/27. A student that already solves the game ignores a
# path it does not need; in those groups the pointer advanced 1.4 lines of ~56.
# Another task's prompt makes no claim about this game at all: every action it
# produces is inadmissible here, so the episode spends its turn budget and ends
# at the failure return.
#
# WHAT TRAINS THE SPECIAL ROW. The shaped off-policy term on the PLAIN prompt
# (LUFFY's f(rho) = rho/(rho+gamma), verl/trainer/ppo/oci_shaping.py) and nothing
# else: the distillation term is off for those rows, because the teacher would be
# read on a prompt carrying a walkthrough, or on another task's prompt, and
# distilling either is a claim this arm does not make.
#
# ---------------------------------------------------------------------------
# THIS IS A WRAPPER, NOT A COPY, like the other arms here: it execs the klw
# CONTROL script -- the only run this arm can be compared against -- and
# overrides the arm's own keys. Drift is caught by the intent lock in
# expected_multitask_oci_slots_config.yaml.
#
# ITS OWN OUTPUT DIRECTORIES. The control hardcodes
# $HOME/checkpoints/...klw_control_multitask$RUN_TAG_SUFFIX with
# trainer.resume_mode=auto, so a wrapper that does not override them points a new
# arm at the control's directory and, under a RUN_TAG whose control has finished,
# resumes to the final step and exits having done nothing.
# ---------------------------------------------------------------------------
set -euo pipefail

# The same expression the control exports, set HERE too: this script builds the
# experiment_name override before exec'ing the control, so the shell expands
# $RUN_TAG_SUFFIX at that point and it has to already be set. The control
# re-exports the identical value, so this is idempotent, and the lock file
# expects exactly this suffix.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

# Mutually exclusive with the other privileged inputs: two blocks in one
# observation are two interventions, and the strip that makes rho meaningful
# removes exactly one replacement per row.
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
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
    trainer.experiment_name="opd_grpo_multitask_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_oci_slots_multitask$RUN_TAG_SUFFIX" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_multitask_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_multitask_oci_slots_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_oci_slots_config.yaml \
    "$@"
