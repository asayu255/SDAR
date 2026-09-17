#!/usr/bin/env bash
# Does reward rise with k? The premise (a) rests on, measured with the run's own definitions.
#
# THE QUESTION. (a) pushes a stuck group's further rollouts up because k -- the
# steps of the task's correct sequence a rollout carried out -- is supposed to track
# how close it got. That had been checked on ALFWorld with a proxy (no "the
# environment executed it" condition) and on Search, never on WebShop.
#
# WHAT IT RUNS. The xt1 recipe at klwctl_step${STEP}, with algorithm.progress_rank
# on so the environment managers count k for every rollout, and grad_probe.mode=progress:
# each batch writes one record per trajectory (task, group, k, K, environment
# reward) to <out>.trajs/b<n>.jsonl, prints the running curve per task, and takes
# no update, so the rollouts are control's whatever RHO is. See
# verl/trainer/ppo/progress_probe.py.
#
# (a)'s OWN NUMBERS AND ProGPO's, ON THE SAME ROLLOUTS. The trainer hook also runs on
# every probe batch: it writes one record per group to <out>.groups/b<n>.jsonl
# (k, K, ProGPO's coverage D, turns, invalid turns, the format spread, ProGPO's
# gate, c, the cap) and keeps its metrics per batch in the payload
# (progress_rank_metrics). RHO=0.05 makes c and the cap what the arm would use;
# at RHO=0 they are 0. Read the records with
#   python scripts/report_progress_groups.py <out>.groups
#
# STEP=75 and STEP=300 on tamago, one at a time. About 30 min each at 7 groups per
# task x 10 batches on 4 GPUs (the step-25 mass probe: 7 min to the first batch,
# then 2m46s a batch).
set -euo pipefail
STEP="${STEP:?set STEP=75 or STEP=300}"
PER_TASK="${PER_TASK:-7}"
N_BATCHES="${N_BATCHES:-10}"
GPUS="${GPUS:-4}"
TEMP="${TEMP:-1.0}"
GROUP_N="${GROUP_N:-8}"
RHO="${RHO:-0.0}"
CKPT="${PROBE_CKPT_DIR:-$HOME/offline_ladder/probe_hf}/klwctl_step${STEP}"
# A non-zero RHO gets its own payload name, so it never overwrites a rho = 0 probe.
RHO_TAG=""; [ "$RHO" = "0" ] || [ "$RHO" = "0.0" ] || RHO_TAG="_rho${RHO//./p}"
OUT="${OUT:-$HOME/grad_probe/progress_klwctl_step${STEP}${RHO_TAG}.json}"
LOG="${LOG:-$HOME/logs/progress_step${STEP}${RHO_TAG}.log}"
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
export RUN_TAG=progressprobe
export RUN_TAG_SUFFIX=_progressprobe
echo "progress probe: step=$STEP per_task=$PER_TASK batches=$N_BATCHES gpus=$GPUS rho=$RHO"
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
  +trainer.grad_probe.mode=progress \
  algorithm.progress_rank.enable=True \
  algorithm.progress_rank.rho="$RHO" \
  +trainer.grad_probe.n_batches="$N_BATCHES" \
  +trainer.grad_probe.seed=0 \
  +trainer.grad_probe.interim_every=1 \
  +trainer.grad_probe.out_path="$OUT" \
  "$@" \
  >"$LOG" 2>&1
