#!/usr/bin/env bash
# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH OCI-sat ARM B': a saturated group's statistic gains one virtual sample at
# the failure return, and no rollout is generated for it.
#
# THIS REPLACES ARM B, IT DOES NOT ACCOMPANY IT. Arm B took the injected row's
# own gradient away, which left the group's mean and std as the only channel by
# which the injection could reach the weights -- and a sample that only has to
# move a mean and a std does not have to exist. So B' asks B's question with the
# generator removed, and the pre-registered comparison is unchanged: A trains on
# the injected row, B' only moves the baseline, and A beating B' is the only
# thing that shows the off-policy term is worth more than the contrast alone.
#
# WHAT THE GENERATOR COST, and why removing it is not a shortcut. Five block
# designs were measured against a pre-registered cand_fail_rate > 0.5, on 45
# candidate trajectories each at control step 300:
#   intact (the true path)                                         4.4%
#   misdirect, PDDL notation                                      15.6%
#   misdirect + detour 12 / 35                              6.7% / 20.0%
#   misdirect, environment words, coherent path, forceful framing  22.2%
#   delay, the true path behind a 50-turn non-refutable tour        8.9%
# ALFWorld's return is binary, an inadmissible action costs a turn and no
# penalty, and nothing in the action set is irreversible, so a prompt can only
# cause failure by spending the 50-turn budget -- and the student abandons any
# path the environment contradicts, while a path long enough to spend the budget
# is not read at all (44 of 45 candidates did not take the tour's first step).
# The virtual sample fails by definition.
#
# AND IT TAKES NO ROLLOUT SLOT. The injection had to consume one of the eight, so
# the arm trained the policy and the distillation term on seven trajectories
# where control trained them on eight -- the confound the design document records
# as unremovable without a ninth slot. Here all eight are real, judged and
# trained, and the arm's only difference from control is a term in the statistic
# of groups control would have given an advantage of exactly zero.
#
# ITS OWN OUTPUT DIRECTORIES, unlike the arm wrappers that came before it. The
# control hardcodes $HOME/checkpoints/...klw_control_multitask$RUN_TAG_SUFFIX and
# trainer.resume_mode is `auto`, so a wrapper that does not override it points a
# new arm at the control's directory: with an untagged launch that directory does
# not exist and the arm starts clean, but under the same RUN_TAG as a FINISHED
# control run it resumes to the final step and exits having done nothing, with no
# warning. The three $HOME-derived paths are not pinned by the lock (they are
# machine-dependent), so overriding them here is free.
#
# See run_multitask_oci_sat_qwen3.sh for why this is a wrapper rather than a copy.
set -euo pipefail

# The same expression the control exports, set HERE too: this script builds the
# experiment_name override before exec'ing the control, so the shell expands
# $RUN_TAG_SUFFIX at that point and it has to already be set. The control
# re-exports the identical value, so this is idempotent, and the lock file
# expects exactly this suffix.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
    algorithm.oci_floor.enable=True \
    'algorithm.oci_floor.tasks=[alfworld]' \
    algorithm.oci_floor.value=0.0 \
    algorithm.oci_floor.weight=trajectory \
    algorithm.oci_sat.enable=False \
    algorithm.compute_mean_std_cross_steps=True \
    trainer.experiment_name="opd_grpo_multitask_oci_floor_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_oci_floor_multitask$RUN_TAG_SUFFIX" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_multitask_oci_floor_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_multitask_oci_floor_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_oci_floor_config.yaml \
    "$@"
