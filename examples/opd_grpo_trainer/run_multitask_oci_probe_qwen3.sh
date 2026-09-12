#!/usr/bin/env bash
# OCI-sat measurement: the three things the corrupted-plan switch does not
# validate about itself. No optimizer step, no extra generation.
#
# ARM-IDENTICAL EXCEPT FOR THE UPDATE. The switch, the candidate column and the
# classification all run exactly as they do in
# examples/opd_grpo_trainer/run_multitask_oci_sat_qwen3.sh; mode=oci replaces the
# optimizer step with a report. So a failure here is a failure of the arm, not of
# a probe-only code path.
#
#   1. WHERE THE DEGENERATE TOKEN MASS IS, split stuck / saturated, per task.
#      Group counts and token counts disagree by ~4x (a saturated group finishes
#      early, a stuck one runs to the turn cap) and only the token count bounds
#      what the saturated arm can buy. Reported for all three tasks: the split
#      is the unmeasured quantity the whole design turns on, and one rollout
#      answers it for every task at once.
#   2. WHETHER THE CORRUPTED PLAN ACTUALLY MAKES THE STUDENT FAIL --
#      oci/cand_fail_rate. A plan missing its requirement step that the student
#      solves anyway leaves the group saturated and injects nothing.
#   3. WHETHER THAT FAILURE IS REACHABLE -- rho and the shaping coefficient
#      A*gamma*rho/(rho+gamma)^2, which is ZERO at rho=0, not one. A failure the
#      student would only produce with the plan in front of it carries no
#      gradient however large its advantage.
#
# The candidate is the LAST ENV SLOT of each alfworld group, marked in the
# rollout loop and carried as the oci_candidate column -- NOT derived from the
# row order in the trainer, which is permuted by _balance_batch before the
# advantage is computed. ALFWorld seeds group_n consecutive workers with the same
# game, so the other seven are plain rollouts of that same game and no second
# generation pass is needed. Validation is untouched: the switch asks the envs it
# holds, and the validation manager is built with group_n=1, is_train=False.
set -euo pipefail

STEP="${STEP:-300}"
TEMP="${TEMP:-1.0}"
GROUP_N="${GROUP_N:-8}"
PER_TASK="${PER_TASK:-15}"
N_BATCHES="${N_BATCHES:-3}"
TAG="${TAG:-oci1}"
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

# The run script calls bare `python3`; without this it is the system one and
# verl's first import fails on numpy.
# The run script calls bare `python3`; without an activated env verl's first
# import fails on numpy. Override CONDA_SH/CONDA_ENV on another host.
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sdar-multitask}"

export EXPECTED_CONFIG_WAIVE="algorithm.opd.kl_loss_coef actor_rollout_ref.actor.teacher_kl_loss_coef actor_rollout_ref.model.path algorithm.opd.kl_loss_type actor_rollout_ref.actor.teacher_kl_loss_type actor_rollout_ref.actor.student_indexed_topk actor_rollout_ref.rollout.temperature env.rollout.n data.train_batch_size data.task_balance.per_task_batch_size"

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
export ENV_RESET_PREFETCH=1
export ROLLOUT_KEEP_VLLM_AWAKE=1
export ROLLOUT_PREFETCH_TEACHER=0
export TASK_BALANCE_INTERLEAVE=1
# THE SWITCH. Prepends alfworld's own expert high-level plan MINUS its
# requirement step to the last slot of each alfworld group. ~78 tokens at p50
# against an alfworld p99 turn prompt of 947 and a 4096 ceiling, so
# max_model_len is untouched and this arm stays comparable to every other one.
export PRIVILEGED_WRONG_PLAN=1
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""
export RUN_TAG=ociprobe
export RUN_TAG_SUFFIX=_ociprobe
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

# NO WANDB: the run script's key expansion is empty in a non-interactive shell
# and wandb.init aborts the job on a 13-character key. The result is the JSON.
# THE RETRIEVER IS ON WAKABA (100.86.45.30), not this host -- without the
# override the run retries a dead URL every 29 s forever at 0% GPU.
exec bash examples/opd_grpo_trainer/run_multitask_qwen3.sh \
  env.search.search_url="${SEARCH_URL:-http://100.86.45.30:8000/retrieve}" \
  'trainer.logger=[console]' \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path="$CKPT" \
  trainer.save_freq=-1 \
  ++algorithm.opd.kl_loss_coef=0.01 \
  ++algorithm.opd.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.teacher_kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.student_indexed_topk=False \
  actor_rollout_ref.rollout.temperature="$TEMP" \
  env.rollout.n="$GROUP_N" \
  ++data.task_balance.per_task_batch_size="$PER_TASK" \
  data.train_batch_size=$(( PER_TASK * 3 )) \
  algorithm.oci_sat.enable=True \
  algorithm.oci_sat.gradient_on_injected=True \
  'algorithm.oci_sat.tasks=[alfworld]' \
  +trainer.grad_probe.enable=True \
  +trainer.grad_probe.mode=oci \
  +trainer.grad_probe.n_batches="$N_BATCHES" \
  +trainer.grad_probe.seed=0 \
  +trainer.grad_probe.interim_every=1 \
  +trainer.grad_probe.gamma=0.1 \
  +trainer.grad_probe.out_path="$OUT" \
  >"$LOG" 2>&1
