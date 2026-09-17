#!/usr/bin/env bash
# Where each task's update comes from, at one checkpoint of the 3-task control.
#
# THE QUESTION. normalize_loss_by_task already gives every task 1/3 of the loss,
# so the 69%/4% token imbalance is handled. What it cannot handle is that a
# degenerate group's rows carry weight and no REWARD signal: their GRPO advantage
# is zero unless the format channel moves it, while the teacher KL reaches every
# row regardless. So a task that stalls does not lose its share -- its update
# shifts from reward-driven to teacher-driven, and nothing in the run reports
# that. This probe measures it on the loss the optimizer sees.
#
# WHAT IT RUNS. The xt1 recipe itself (run_multitask_cross_teacher_klw_control),
# so the KL is the one that was trained: top-k, student-indexed, coefficient
# 0.01. The actor runs its own forward and its own loss with
# measure_terms_only -- no backward, no optimizer step -- and writes per-row sums
# of |policy gradient| and of the teacher KL, which the driver folds into
# (task, group class) cells: live, and stuck/saturated split by whether any
# advantage in the group is non-zero. See verl/trainer/ppo/term_mass.py.
#
# SELF-CHECK. The driver recomputes the PG mass from the batch's own advantages
# and reports actor/driver as a ratio; anything but ~1 means the rows were mapped
# back wrongly and the cells are meaningless.
#
# STEP=25 first (stuck 70% early), then 75, 150, 300. Roughly 15-20 min each.
set -euo pipefail
STEP="${STEP:-25}"
PER_TASK="${PER_TASK:-7}"
N_BATCHES="${N_BATCHES:-10}"
GPUS="${GPUS:-4}"
TEMP="${TEMP:-1.0}"
GROUP_N="${GROUP_N:-8}"
TAG="${TAG:-mass}"
CKPT="${PROBE_CKPT_DIR:-$HOME/offline_ladder/probe_hf}/klwctl_step${STEP}"
OUT="${OUT:-$HOME/grad_probe/mass_klwctl_step${STEP}.json}"
LOG="${LOG:-$HOME/logs/mass_step${STEP}.log}"
[ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sdar-multitask}"
# The recipe is xt1's; these are the pieces a probe changes on purpose.
export EXPECTED_CONFIG_WAIVE="actor_rollout_ref.model.path data.train_batch_size data.task_balance.per_task_batch_size trainer.total_training_steps"
export PYTHONUNBUFFERED=1
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
export ENV_RESET_PREFETCH=1
export ROLLOUT_KEEP_VLLM_AWAKE=1
# tamago's knobs: pump and logprob prefetch kill 3-task rollouts here, teacher
# prefetch off OOMs (see the alfworld-only launcher's notes).
export ROLLOUT_PUMP_TRAINING=0
export ROLLOUT_PREFETCH_LOGPROB=0
export ROLLOUT_PREFETCH_TEACHER=1
export TASK_BALANCE_INTERLEAVE=1
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""
export RUN_TAG=massprobe
export RUN_TAG_SUFFIX=_massprobe
echo "mass probe: step=$STEP per_task=$PER_TASK batches=$N_BATCHES gpus=$GPUS"
echo "  groups: $(( PER_TASK * 3 )) per batch, $(( PER_TASK * 3 * N_BATCHES )) total"
echo "  out: $OUT"
echo "  log: $LOG"
exec bash examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh \
  env.search.search_url="${SEARCH_URL:-http://100.86.45.30:8000/retrieve}" \
  'trainer.logger=[console]' \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path="$CKPT" \
  actor_rollout_ref.rollout.temperature="$TEMP" \
  env.rollout.n="$GROUP_N" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  ++data.task_balance.per_task_batch_size="$PER_TASK" \
  data.train_batch_size=$(( PER_TASK * 3 )) \
  trainer.n_gpus_per_node="$GPUS" \
  +trainer.grad_probe.enable=True \
  +trainer.grad_probe.mode=mass \
  +trainer.grad_probe.n_batches="$N_BATCHES" \
  +trainer.grad_probe.seed=0 \
  +trainer.grad_probe.interim_every=1 \
  +trainer.grad_probe.out_path="$OUT" \
  "$@" \
  >"$LOG" 2>&1
