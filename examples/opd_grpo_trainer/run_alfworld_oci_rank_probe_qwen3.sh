#!/usr/bin/env bash
# The rank scorers, checked on ORDINARY groups: the ten-slot layout is OFF, so
# every group is eight plain rollouts, exactly what control trains.
#
# WHY A SEPARATE PROBE. The first measurement ran inside the slots probe and a
# saturated group with a foreign row counted as "live": the one loser was the
# row shown another goal, so every scorer ranked it last and the AUC came out
# 0.985 for rows whose prompt was different. Here there are no special rows.
#
# WHAT IT REPORTS, per scorer (plain / privileged / teacher, and the two gains
# against plain), in the payload's oci[].rank block:
#   live_auc                   within-group AUC of score vs outcome, with CI
#   live_auc_length_adjusted   the same on the residual of a within-group fit on
#                              turn count -- in alfworld a loss IS a run to the
#                              50-turn cap, so the raw AUC mostly measures length
#   first_divergence           at the first turn a live group's actions differ,
#                              is the winner's row preferred? (same prompt, no
#                              length confound; OVCSD, arXiv 2607.27937, reports
#                              this at or below chance for a skill-conditioned self)
#   degenerate                 inside stuck groups, order vs walkthrough progress;
#                              inside saturated groups, order vs turn count
#   self_check                 SELF_CHECK_ROWS rows per batch (default 32) re-scored
#                              with a null edit (must equal plain) and from the
#                              document prompt's own text (must equal privileged)
set -euo pipefail
STEP="${STEP:-300}"
TEMP="${TEMP:-1.0}"
GROUP_N="${GROUP_N:-8}"
PER_TASK="${PER_TASK:-15}"
N_BATCHES="${N_BATCHES:-3}"
TASKS="${TASKS:-alfworld}"
N_TASKS=$(awk -F, "{print NF}" <<< "$TASKS")
GPUS="${GPUS:-3}"
TAG="${TAG:-rank}"
MODEL="${MODEL:-student}"
case "$MODEL" in
  student) CKPT="${PROBE_CKPT_DIR:-$HOME/offline_ladder/probe_hf}/klwctl_step${STEP}" ;;
  base)    CKPT="Qwen/Qwen3-1.7B" ;;
  *)       CKPT="$MODEL" ;;
esac
OUT="${OUT:-$HOME/grad_probe/oci_klwctl_step${STEP}_${TAG}.json}"
LOG="${LOG:-$HOME/logs/oci_step${STEP}_${TAG}.log}"
case "$CKPT" in /*) [ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; } ;; esac
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sdar-multitask}"
# The expectations file pins the training recipe; a probe changes the pieces
# named here on purpose and says so at startup.
export EXPECTED_CONFIG_WAIVE="algorithm.opd.kl_loss_coef actor_rollout_ref.actor.teacher_kl_loss_coef actor_rollout_ref.model.path algorithm.opd.kl_loss_type actor_rollout_ref.actor.teacher_kl_loss_type actor_rollout_ref.actor.student_indexed_topk actor_rollout_ref.rollout.temperature env.rollout.n data.train_batch_size data.task_balance.per_task_batch_size data.task_balance.tasks env.multitask.tasks"
export PYTHONUNBUFFERED=1
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
export ENV_RESET_PREFETCH=1
export ROLLOUT_KEEP_VLLM_AWAKE=1
export ROLLOUT_PREFETCH_TEACHER=0
# Throughput knobs that fail on all-alfworld batches (see the alfworld-only
# launcher); a probe has no update to speed up anyway.
export ROLLOUT_PUMP_TRAINING=0
export ROLLOUT_PREFETCH_LOGPROB=0
export TASK_BALANCE_INTERLEAVE=1
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""
export RUN_TAG=rankprobe
export RUN_TAG_SUFFIX=_rankprobe
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
exec bash examples/opd_grpo_trainer/run_multitask_qwen3.sh \
  env.search.search_url="${SEARCH_URL:-http://100.86.45.30:8000/retrieve}" \
  'trainer.logger=[console]' \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path="$CKPT" \
  trainer.save_freq=-1 \
  ++algorithm.opd.kl_loss_coef=0.01 \
  ++algorithm.opd.kl_loss_type=low_var_kl \
  ++algorithm.opd.normalize_loss_by_task=True \
  actor_rollout_ref.actor.teacher_kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.student_indexed_topk=False \
  actor_rollout_ref.rollout.temperature="$TEMP" \
  env.rollout.n="$GROUP_N" \
  ++data.task_balance.per_task_batch_size="$PER_TASK" \
  data.train_batch_size=$(( PER_TASK * N_TASKS )) \
  "data.task_balance.tasks=[${TASKS}]" \
  "env.multitask.tasks=[${TASKS}]" \
  trainer.n_gpus_per_node="$GPUS" \
  algorithm.oci_sat.enable=False \
  algorithm.oci_floor.enable=False \
  algorithm.oci_slots.enable=False \
  algorithm.oci_rank.enable=True \
  'algorithm.oci_rank.tasks=[alfworld]' \
  algorithm.oci_rank.self_check_rows="${SELF_CHECK_ROWS:-32}" \
  algorithm.compute_mean_std_cross_steps=True \
  +trainer.grad_probe.enable=True \
  +trainer.grad_probe.mode=oci \
  +trainer.grad_probe.n_batches="$N_BATCHES" \
  +trainer.grad_probe.seed=0 \
  +trainer.grad_probe.interim_every=1 \
  +trainer.grad_probe.gamma=0.1 \
  +trainer.grad_probe.out_path="$OUT" \
  "$@" \
  >"$LOG" 2>&1
