#!/usr/bin/env bash
# TIED-GROUP SELF-DISTILLATION (algorithm.tied_opsd): the 2026-10-01 design ("停滞・飽和群の自己蒸留").
#
# THE ARM. GRPO on the v2 recipe (run_multitask_grpo_v2recipe_qwen3.sh's cell: outcome GRPO, the
# value arm's records-only shadow, the same sampling and gradient-path knobs), with:
#   algorithm.tied_opsd.enable=True
#       in a group whose 8 rollouts all failed (stuck) or all succeeded (saturated), the student is
#       distilled toward a frozen copy of itself that reads the instance's correct document (short
#       lead; the progress sentence for stuck rows, the efficiency sentence for saturated rows; the
#       state pointer's progress line): stuck 0.5 forward + 0.5 reverse KL, saturated reverse KL, on
#       the teacher's top-20 plus a tail bucket, per distilled token weighted
#       w_f = M (1 - q_f), w_s = M min(q_s, live / q_s)  (M: the live groups' discounted mean |z|).
#       Mixed groups: GRPO only. Tags and special tokens are never distilled; on Search only the
#       queries and, after a result carried the answer, the answer. The copy is refreshed every 50
#       steps (steps 1-50: the initial model). See verl/trainer/ppo/tied_opsd.py.
#   ~algorithm.opd.teacher_paths.{alfworld,search,webshop}
#       NO external OPD teacher (the design's item 10): the three paths the control launcher adds
#       are deleted (Hydra merges "teacher_paths={}" into the dict instead of replacing it), and
#       main_opd_grpo refuses the arm if any is left; it also turns the teacher KL off.
#   algorithm.oci_slots.search_flow_path=<route_hints_v2.json>
#       Search's verified answer-free routes (4,101 of 4,500 questions); the ten-slot layout itself
#       stays off -- only the route file is read.
#   ROLLOUT_PREFETCH_TEACHER=0
#       nothing to prefetch without an external teacher (the self-teacher scores after the rollout,
#       teacher-indexed top-k: no hidden-state cache).
#
# RUN_TAG tied_opsd; lock expected_multitask_tied_opsd_config.yaml:
#   bash examples/opd_grpo_trainer/run_multitask_tied_opsd_qwen3.sh
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAM=1.0
export RUN_TAG="${RUN_TAG:-tied_opsd}"
export ROLLOUT_PREFETCH_TEACHER=0
echo "tied_opsd: run_tag=$RUN_TAG"
exec bash "$_HERE/run_multitask_progress_value_gae_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_tied_opsd_config.yaml \
  algorithm.adv_estimator=grpo \
  algorithm.progress_value.allow_missing_table=False \
  algorithm.progress_value.records.enable=True \
  algorithm.tied_opsd.enable=True \
  '~algorithm.opd.teacher_paths.alfworld' \
  '~algorithm.opd.teacher_paths.search' \
  '~algorithm.opd.teacher_paths.webshop' \
  algorithm.oci_slots.search_flow_path=/opt1/ohara/data/qa_annotations/route_hints_4500/route_hints_v2.json \
  "$@"
