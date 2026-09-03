#!/usr/bin/env bash
# Score saved off-policy checkpoints, one process per checkpoint.
#
#   bash examples/opd_trainer/eval_checkpoints.sh studenttopk 150 300
#   bash examples/opd_trainer/eval_checkpoints.sh studenttopk        # newest only
#   bash examples/opd_trainer/eval_checkpoints.sh control 300 -- env.search.search_url=...
#
# This is the other half of trainer.test_freq=-1 in the two Stage-2 run scripts.
# Training no longer validates; this scores its checkpoints afterwards and logs
# each result to wandb AT THE CHECKPOINT'S OWN STEP, so the curve lands on the
# training run's x-axis.
#
# WHY A SEPARATE PROCESS PER CHECKPOINT, AND NOT A LOOP INSIDE ONE
#
# Because validation has to score every checkpoint on the SAME episodes, and only
# a fresh process gives that. alfworld draws each episode from TextWorld's seeded
# game-file cycle (agent_system/.../alfworld/envs.py), which is stateful: every
# reset advances it. A second validation in the same process therefore plays
# different games than the first -- so a difference between two checkpoints would
# carry a difference in what they were asked to do. Rebuilding the process
# rebuilds that cycle from env.seed at position 0.
#
# It costs a model load and an env build per checkpoint against a validation of
# hours, and it buys back more than it costs: no state of any kind survives from
# one checkpoint's evaluation to the next.
#
# WHAT THIS INHERITS. It runs the arm's own run script rather than repeating its
# arguments, so the model, the tasks, the episode caps, the per-task val sampling
# (val_kwargs_by_task) and the retriever are the training run's by construction,
# and that arm's expected_config is enforced here too. It adds only keys the run
# script does not already pass:
#
#   trainer.val_only=True            -- validate once and return. The trainer also
#                                       reads this to skip the Stage-1 pool load
#                                       and, on the student-indexed arm, to not
#                                       build the three teachers: a process that
#                                       runs no update reads neither.
#   trainer.resume_mode=resume_path
#   trainer.resume_from_path=.../global_step_<N>
#
# Nothing is passed twice. A setting this script cannot reach is one to edit in
# the run script -- or, where evaluation genuinely wants a different value, one
# the run script reads from the environment. rollout.gpu_memory_utilization is
# the second kind and this script raises it: the training arm keeps FSDP
# parameters, gradients and optimizer state on the same cards and is sized at
# 0.3 for that, while an eval process holds none of them.
#
# THE CHECKPOINT DIRECTORY IS SHARED BY THE ARMS. It is read out of the arm's run
# script below, and all three off-policy scripts currently point at one tree, so
# global_step_<N> there was written by whichever arm ran last. Naming the arm
# selects the run script -- the flags, the intent lock, the wandb identity -- not
# the checkpoints. If both arms have trained into that directory, check which one
# the step you are scoring came from.
set -euo pipefail

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RAY'S TEMP DIRECTORY IS PER USER, because /tmp/ray is not.
#
# On a shared box every user's Ray defaults to the same /tmp/ray, and a raylet
# that connects to another user's runtime-env agent gets a protocol mismatch --
# "Runtime Env Agent timed out", the raylet exits, the node is marked dead and
# the actors die with it. It is not a load problem and not a stale session of
# ours; it is two Ray versions sharing one socket directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$(id -un)}"
mkdir -p "$RAY_TMPDIR" 2>/dev/null || true

# ---- which arm ------------------------------------------------------------- #
ARM="${1:-}"
case "$ARM" in
    studenttopk) RUN_SCRIPT="$OPD_DIR/run_multitask_offpolicy_studenttopk_qwen3.sh" ;;
    control|nogen) RUN_SCRIPT="$OPD_DIR/run_multitask_offpolicy_qwen3_nogen.sh" ;;
    *)
        echo "usage: bash $0 {studenttopk|control} [STEP ...] [-- hydra.override=...]" >&2
        echo "  the arm selects the run script, and with it the flags, the intent" >&2
        echo "  lock and the wandb identity the scores are logged under." >&2
        exit 1
        ;;
esac
shift
[ -f "$RUN_SCRIPT" ] || { echo "missing $RUN_SCRIPT" >&2; exit 1; }

# Read out of the run script rather than restated here, so the two cannot drift.
CKPT_DIR=$(grep -oE '^\s*trainer\.default_local_dir=\S+' "$RUN_SCRIPT" \
    | tail -1 | cut -d= -f2- | sed 's/[[:space:]]*\\*$//')
[ -n "$CKPT_DIR" ] || { echo "could not read trainer.default_local_dir from $RUN_SCRIPT" >&2; exit 1; }
CKPT_DIR=$(eval echo "$CKPT_DIR")
[ -d "$CKPT_DIR" ] || { echo "no checkpoint directory at $CKPT_DIR" >&2; exit 1; }

# ---- which checkpoints ----------------------------------------------------- #
ARG_STEPS=()
PASSTHROUGH=()
SEEN_SEP=0
for arg in "$@"; do
    if [ "$SEEN_SEP" = 1 ]; then
        PASSTHROUGH+=("$arg")
    elif [ "$arg" = "--" ]; then
        SEEN_SEP=1
    else
        case "$arg" in
            ''|*[!0-9]*)
                echo "not a checkpoint step: '$arg' (use -- before Hydra overrides)" >&2
                exit 1
                ;;
        esac
        ARG_STEPS+=("$arg")
    fi
done

if [ "${#ARG_STEPS[@]}" -gt 0 ]; then
    STEPS=$(printf '%s\n' "${ARG_STEPS[@]}" | sort -n -u)
else
    # The newest one only, deliberately: at save_freq=25 a 300-step run leaves
    # twelve checkpoints, and a validation pass is hours. Ask for the ones you
    # want to compare.
    STEPS=$(ls -1d "$CKPT_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    [ -n "$STEPS" ] || { echo "no global_step_* checkpoints under $CKPT_DIR" >&2; exit 1; }
    echo "[eval] no steps given; evaluating the newest checkpoint only (step $STEPS)."
    echo "[eval] pass steps explicitly to score more, e.g.: bash $0 $ARM 150 300"
fi

# All of them, before the first one is loaded: discovering a missing checkpoint
# after an hour of validating the previous one is the avoidable version of this.
for STEP in $STEPS; do
    [ -d "$CKPT_DIR/global_step_$STEP/actor" ] || {
        echo "no actor checkpoint at $CKPT_DIR/global_step_$STEP/actor" >&2; exit 1; }
done

# ---- what evaluation wants that training does not -------------------------- #
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"
export ROLLOUT_KEEP_VLLM_AWAKE=1

# A wandb run of its own, resumable, so a second invocation adds checkpoints to
# the same curve instead of starting a third one. The scores still land at each
# checkpoint's own step -- the trainer logs at the global_steps _load_checkpoint
# restored, not at 0.
export WANDB_RUN_ID="${WANDB_RUN_ID:-opd-offpolicy-$ARM-eval-$(date +%Y%m%d-%H%M%S)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_TAGS="${WANDB_TAGS:-checkpoint-eval}"

_CORES=$(nproc 2>/dev/null || echo '?')
echo "[eval] arm         : $ARM ($(basename "$RUN_SCRIPT"))"
echo "[eval] checkpoints : $(tr '\n' ' ' <<< "$STEPS")"
echo "[eval] ckpt dir    : $CKPT_DIR"
echo "[eval] wandb run id: $WANDB_RUN_ID (resume=$WANDB_RESUME, tags=$WANDB_TAGS)"
echo "[eval] vllm util   : $ROLLOUT_GPU_MEM_UTIL (training uses 0.3; eval holds no optimizer state)"
echo "[eval] machine     : $_CORES cores, load$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null | sed 's/^/ /')"
echo "[eval] one process per checkpoint, run one after another."

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
