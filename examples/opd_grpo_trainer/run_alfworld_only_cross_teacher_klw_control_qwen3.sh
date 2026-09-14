#!/usr/bin/env bash
# OPD + GRPO on ALFWORLD ONLY, Qwen3-1.7B: the single-task counterpart of the
# 3-task klw control. Same objective, same per-task data (15 prompts x 8
# rollouts of alfworld per step), same optimiser, same 300 steps, same teachers
# configured; only the task list, the batch size that follows from it, and the
# run's name differ. Training only -- no validation inside the run. Steps 150
# and 300 are scored afterwards as separate val-only runs of this same script:
#
#   VAL_ONLY=1 VAL_CKPT=$HOME/checkpoints/verl_agent_opd_grpo_cross_teacher_klw_control_alfworld_only/global_step_150 \
#       bash examples/opd_grpo_trainer/run_alfworld_only_cross_teacher_klw_control_qwen3.sh
#
# A WRAPPER, NOT A COPY: it execs the klw CONTROL script and overrides the keys
# below, so the control cannot drift away from this arm unnoticed. The intent
# lock pins the result (expected_alfworld_only_cross_teacher_klw_control_config.yaml);
# it differs from the control's lock in exactly the keys named in its header.
#
# The validation loader is not filtered by task, so the 3-task test.parquet
# (which also holds 126 webshop and 51,713 search rows) would send rows to
# environments this run does not build. data.val_files therefore points at the
# alfworld rows of that same file, written out once by the operator as
# test_alfworld_only.parquet (126 rows, identical to the alfworld slice of
# test.parquet, which prepare_sdar_multitask regenerates deterministically).
set -euo pipefail

RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
    data.train_batch_size=15 \
    'data.task_balance.tasks=[alfworld]' \
    'env.multitask.tasks=[alfworld]' \
    env.multitask.val_per_task_batch_size=null \
    data.val_files="$HOME/data/verl-agent/sdar_multitask/test_alfworld_only.parquet" \
    trainer.test_freq=0 \
    trainer.experiment_name="opd_grpo_alfworld_only_cross_teacher_klw_control_qwen3_1.7b${RUN_TAG_SUFFIX}" \
    trainer.default_local_dir="$HOME/checkpoints/verl_agent_opd_grpo_cross_teacher_klw_control_alfworld_only${RUN_TAG_SUFFIX}" \
    trainer.val_instance_log_dir="$HOME/val_instances/opd_grpo_alfworld_only_cross_teacher_klw_control_qwen3_1.7b${RUN_TAG_SUFFIX}" \
    trainer.sign_token_dump_dir="$HOME/sign_tokens/opd_grpo_alfworld_only_cross_teacher_klw_control_qwen3_1.7b${RUN_TAG_SUFFIX}" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_alfworld_only_cross_teacher_klw_control_config.yaml \
    "$@"
