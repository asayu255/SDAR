#!/usr/bin/env bash
# Score saved multitask-SFT checkpoints, one process per checkpoint.
#
#   bash examples/sft_trainer/eval_checkpoints.sh 150 300
#   bash examples/sft_trainer/eval_checkpoints.sh            # the newest checkpoint
#   bash examples/sft_trainer/eval_checkpoints.sh 300 -- env.search.search_url=...
#
# This is the other half of trainer.test_freq=-1 in run_multitask_sft_qwen3.sh.
# The training run no longer validates; this scores its checkpoints afterwards
# and logs each result to wandb AT THE CHECKPOINT'S OWN STEP, so the curve lands
# on the training run's x-axis.
#
# WHY IT IS A SEPARATE PROCESS PER CHECKPOINT, AND NOT A LOOP INSIDE ONE
#
# Because validation has to score every checkpoint on the SAME episodes, and only
# a fresh process gives that. alfworld draws each episode from TextWorld's seeded
# game-file cycle (agent_system/.../alfworld/envs.py), which is stateful: every
# reset advances it. A second validation in the same process therefore plays
# different games than the first -- so a difference between two checkpoints would
# carry a difference in what they were asked to do. Rebuilding the process
# rebuilds that cycle from env.seed at position 0.
#
# It costs a model load and an env build per checkpoint (a few minutes) against a
# validation of a few hours, and it buys back more than it costs: no state of any
# kind survives from one checkpoint's evaluation to the next.
#
# COST. A pass is all three tasks at env.multitask.val_per_task_batch_size=126
# episodes each, alfworld running to 50 turns -- 3.8 h measured on 3xA6000 (wandb
# x7g9r7bx). PASS THE STEPS YOU WANT. With no arguments this evaluates the newest
# checkpoint only, deliberately: `for every checkpoint` at save_freq=25 is twelve
# of them, i.e. about 45 GPU-hours.
#
# The training run's two validations were at step 150 and 300 (its test_freq was
# 150), so `eval_checkpoints.sh 150 300` reproduces exactly what it used to do.
#
# WHAT THIS INHERITS. It runs run_multitask_sft_qwen3.sh itself rather than
# repeating its arguments, so the model, the tasks, the episode caps, the
# per-task val sampling (val_kwargs_by_task) and the retriever are the training
# run's by construction, and expected_multitask_sft_config.yaml is enforced here
# too. It adds only keys that script does not already pass:
#
#   trainer.val_only=True            -- validate once, then return (and skip the
#                                       136.5 GiB Stage-1 pool load, which a run
#                                       that never draws a batch does not need)
#   trainer.resume_mode=resume_path
#   trainer.resume_from_path=.../global_step_<N>
#
# Nothing is passed twice; a setting this script cannot reach is one to edit in
# the run script (raising rollout.gpu_memory_utilization above 0.6 is a fair one
# to consider here -- an eval process holds no optimizer, gradients or teacher
# pool, so vLLM can have more of the card for its KV cache).
#
# The teacher pool does not have to exist for this. algorithm.sft.data_dir is
# still checked as the identity of the arm, but a val_only process never reads it.
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SFT_DIR/run_multitask_sft_qwen3.sh"
[ -f "$RUN_SCRIPT" ] || { echo "missing $RUN_SCRIPT" >&2; exit 1; }

# Read the checkpoint root out of the run script rather than repeating it, so the
# two cannot disagree about where the checkpoints are.
CKPT_DIR="${CKPT_DIR:-$(sed -n 's/^[[:space:]]*trainer\.default_local_dir=\([^ \\]*\).*/\1/p' "$RUN_SCRIPT")}"
[ -n "$CKPT_DIR" ] || { echo "could not read trainer.default_local_dir from $RUN_SCRIPT; set CKPT_DIR=" >&2; exit 1; }
[ -d "$CKPT_DIR" ] || { echo "checkpoint dir does not exist: $CKPT_DIR" >&2; exit 1; }

# Arguments are checkpoint steps; anything after a literal `--` is passed through
# to the run script as extra Hydra overrides. Keeping them apart matters because
# the steps are not overrides -- handing `150` to Hydra is a parse error.
ARG_STEPS=()
PASSTHROUGH=()
seen_sep=0
for arg in "$@"; do
    if [ "$seen_sep" = 1 ]; then
        PASSTHROUGH+=("$arg")
    elif [ "$arg" = "--" ]; then
        seen_sep=1
    else
        case "$arg" in
            ''|*[!0-9]*) echo "not a checkpoint step: '$arg' (use -- before Hydra overrides)" >&2; exit 1;;
        esac
        ARG_STEPS+=("$arg")
    fi
done

if [ "${#ARG_STEPS[@]}" -gt 0 ]; then
    # Ascending, deduplicated: wandb drops a point logged at a step below the
    # run's current one, so a sweep that walks backwards would silently record
    # only its first checkpoint.
    STEPS=$(printf '%s\n' "${ARG_STEPS[@]}" | sort -n -u)
else
    STEPS=$(ls -1d "$CKPT_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    [ -n "$STEPS" ] || { echo "no global_step_* checkpoints under $CKPT_DIR" >&2; exit 1; }
    echo "[eval] no steps given; evaluating the newest checkpoint only (step $STEPS)."
    echo "[eval] pass steps explicitly to score more, e.g.: bash $0 150 300"
fi

for STEP in $STEPS; do
    [ -d "$CKPT_DIR/global_step_$STEP/actor" ] || {
        echo "no actor checkpoint at $CKPT_DIR/global_step_$STEP/actor" >&2; exit 1; }
done

# One wandb run for the whole sweep, so the checkpoints form a curve instead of a
# scatter of one-point runs. WANDB_RESUME=allow makes the first process create it
# and the rest append; the id is per invocation, so re-running this script never
# has to log a step it has already passed (which wandb would drop).
#
# The run keeps the training run's *name* -- Tracking passes experiment_name to
# wandb.init explicitly and it is pinned by the expectations file, correctly: this
# is the same experiment. The tag and the id below are what separate the two.
# Set WANDB_RUN_ID yourself to add checkpoints to an eval run this script made
# earlier -- but only steps above the highest one already in it.
export WANDB_RUN_ID="${WANDB_RUN_ID:-sft-multitask-eval-$(date +%Y%m%d-%H%M%S)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_TAGS="${WANDB_TAGS:-checkpoint-eval}"

echo "[eval] checkpoints : $(tr '\n' ' ' <<< "$STEPS")"
echo "[eval] ckpt dir    : $CKPT_DIR"
echo "[eval] wandb run id: $WANDB_RUN_ID (resume=$WANDB_RESUME, tags=$WANDB_TAGS)"
echo "[eval] ~3.8 h per checkpoint on 3xA6000; they run one after another."

for STEP in $STEPS; do
    echo "[eval] === global_step_$STEP ==="
    bash "$RUN_SCRIPT" \
        trainer.val_only=True \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$CKPT_DIR/global_step_$STEP" \
        ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    echo "[eval] === global_step_$STEP done ==="
done

echo "[eval] all checkpoints scored into wandb run $WANDB_RUN_ID"
