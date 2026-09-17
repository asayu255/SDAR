#!/usr/bin/env bash
# Search's two open questions, measured in one pass at a fixed checkpoint, with
# no optimizer step.
#
# 1. WHY DO SEARCH GROUPS STALL? Search stuck groups are 43-51% of its groups for
#    the whole of training, but the question verbatim already returns a passage
#    holding the answer for 69% of nq questions. If the stuck groups DID retrieve
#    the answer and still failed, they failed at answering, and no document about
#    searching can rescue them. Read from the eight ordinary rows of each group:
#    per group class, how many rows received a result containing the answer.
#
# 2. DOES THE RESCUE ROW WORK UNDER THE RULE? The ninth row is the document slot
#    with search_doc=answer_rule: it is shown the answer and the rule that a
#    returned result must contain it BEFORE the row writes it anywhere, plus the
#    per-turn line saying whether one has. A row that writes it early is a copy of
#    the prompt, which is what Search's answer-only reward lets through, so the
#    rescue rate that matters is "scored AND kept the rule".
#
# 3. WHICH RESCUE DOCUMENT? Even under the rule, a row that has READ the answer can
#    write a query only someone who knows it would write ("president 1861 1865
#    assassinated" names no part of "Lincoln"), and no string test catches that. The
#    tenth row wears the variant that withholds the answer and shows only the verdict
#    (search_doc_b=expert_flow: queries written by a stronger model and verified against
#    this retriever). Both rescue rows run on the SAME group, so the two
#    rates are paired on the question, the eight siblings and the sampling.
#
# WHAT IT WRITES. One JSON line per rollout under SEARCH_PROBE_DUMP (question,
# answers, slot role and variant, every turn's action, query and result,
# evidence_seen, answer_early, won), which scripts/analyze_search_rescue.py turns
# into all three answers. The grad_probe payload at OUT carries the usual
# per-batch group classes and the injected row's own rescue rate beside them.
#
# GROUP_N=10, and search has no foreign slot: slots 0-6 plain, 7 reserve, 8 the
# second document (the verified route), 9 the document the arm would ship. A group is
# the eight ordinary rollouts control trains, plus the two rescue rows.
set -euo pipefail
STEP="${STEP:-150}"
TEMP="${TEMP:-1.0}"
GROUP_N="${GROUP_N:-10}"
PER_TASK="${PER_TASK:-20}"
N_BATCHES="${N_BATCHES:-10}"
TASKS="${TASKS:-search}"
N_TASKS=$(awk -F, "{print NF}" <<< "$TASKS")
GPUS="${GPUS:-4}"
SEARCH_DOC="${SEARCH_DOC:-answer_rule}"
SEARCH_DOC_B="${SEARCH_DOC_B:-expert_flow}"
# Routes written by a stronger model and verified against this retriever:
# 55 of the 92 stuck questions of the first run return the answer, against 35
# for the student's own eight rollouts. Built by
# data/qa_annotations/verify_claude_flows.py, which refuses any query naming
# an answer.
SEARCH_FLOW_PATH="${SEARCH_FLOW_PATH:-/opt1/ohara/data/qa_annotations/claude_flows_final.json}"
TAG="${TAG:-search_rescue}"
MODEL="${MODEL:-student}"
case "$MODEL" in
  student) CKPT="${PROBE_CKPT_DIR:-$HOME/offline_ladder/probe_hf}/klwctl_step${STEP}" ;;
  base)    CKPT="Qwen/Qwen3-1.7B" ;;
  *)       CKPT="$MODEL" ;;
esac
OUT="${OUT:-$HOME/grad_probe/oci_klwctl_step${STEP}_${TAG}.json}"
LOG="${LOG:-$HOME/logs/oci_step${STEP}_${TAG}.log}"
DUMP="${SEARCH_PROBE_DUMP:-$HOME/grad_probe/${TAG}_step${STEP}_rollouts}"
case "$CKPT" in /*) [ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; } ;; esac
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" "$DUMP"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sdar-multitask}"
# The expectations file pins the training recipe; a probe changes the pieces
# named here on purpose and says so at startup.
export EXPECTED_CONFIG_WAIVE="algorithm.opd.kl_loss_coef actor_rollout_ref.actor.teacher_kl_loss_coef actor_rollout_ref.model.path algorithm.opd.kl_loss_type actor_rollout_ref.actor.teacher_kl_loss_type actor_rollout_ref.actor.student_indexed_topk actor_rollout_ref.rollout.temperature env.rollout.n data.train_batch_size data.task_balance.per_task_batch_size data.task_balance.tasks env.multitask.tasks"
export PYTHONUNBUFFERED=1
export ENV_RESET_PREFETCH=1
export ROLLOUT_KEEP_VLLM_AWAKE=1
export ROLLOUT_PREFETCH_TEACHER=0
export ROLLOUT_PUMP_TRAINING=0
export ROLLOUT_PREFETCH_LOGPROB=0
export TASK_BALANCE_INTERLEAVE=1
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""
# The per-rollout record this probe exists for. Read by the env manager; unset,
# it writes nothing, which is what every training run does.
export SEARCH_PROBE_DUMP="$DUMP"
export RUN_TAG=searchrescue
export RUN_TAG_SUFFIX=_searchrescue
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
echo "probe: step=$STEP group_n=$GROUP_N per_task=$PER_TASK batches=$N_BATCHES search_doc=$SEARCH_DOC/$SEARCH_DOC_B"
echo "  questions: $(( PER_TASK * N_BATCHES )), rollouts: $(( PER_TASK * N_BATCHES * GROUP_N ))"
echo "  dump: $DUMP"
echo "  log:  $LOG"
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
  algorithm.oci_slots.enable=True \
  'algorithm.oci_slots.tasks=[search]' \
  algorithm.oci_slots.search_doc="$SEARCH_DOC" \
  algorithm.oci_slots.search_doc_b="$SEARCH_DOC_B" \
  algorithm.oci_slots.search_flow_path="$SEARCH_FLOW_PATH" \
  algorithm.oci_slots.gamma=0.1 \
  algorithm.oci_slots.opd_on_special=False \
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
