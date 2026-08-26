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
# COST. A pass is all three tasks (alfworld and webshop at 126 episodes per
# batch, search at 252), alfworld running to 50 turns -- 1.26 h measured on
# 3xA6000 for 208 batches (wandb sft-multitask-eval-20260826-201115; it was
# 3.8 h before the retriever was batching). PASS THE STEPS YOU WANT. With no
# arguments this evaluates the newest checkpoint only, deliberately: `for every
# checkpoint` at save_freq=25 is twelve of them, i.e. about 15 GPU-hours.
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
# the run script -- or, where evaluation genuinely wants a different value, one
# the run script reads from the environment. rollout.gpu_memory_utilization is
# the second kind, and this script raises it (see ROLLOUT_GPU_MEM_UTIL below).
#
# The teacher pool does not have to exist for this. algorithm.sft.data_dir is
# still checked as the identity of the arm, but a val_only process never reads it.
#
# THE ROLLOUT HAS TO BE BUILT HERE. The run script defaults SKIP_ROLLOUT_BUILD=1
# because the training arm never generates, and skipping the vLLM build is what
# lets it run under expandable_segments. This process is the opposite case:
# generating IS the work. Left at the default it would load the model, build FSDP
# and the envs, and only then raise from generate_sequences -- minutes in, after
# the checkpoint is already on the GPUs. So force it off here, which also stops
# the run script exporting the allocator setting (the two travel together --
# vLLM's CuMemAllocator asserts expandable segments are off).
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

# Not ${SKIP_ROLLOUT_BUILD:-0}: an inherited 1 from the shell that last launched
# a training run would be honoured and this would fail late for a reason the
# caller has no way to see. There is no such thing as a valid eval without a
# rollout, so this one is not a knob.
if [ "${SKIP_ROLLOUT_BUILD:-0}" != "0" ]; then
    echo "[eval] SKIP_ROLLOUT_BUILD=$SKIP_ROLLOUT_BUILD in the environment; overriding to 0 -- evaluation generates."
fi
export SKIP_ROLLOUT_BUILD=0

# Same reasoning, opposite default. The training arm never generates, so
# per-turn vLLM wake/sleep costs it nothing and the flag is a plain opt-in.
# Evaluation generates on every turn, and a 2-second nvidia-smi trace of the
# 2026-08-24 eval showed the engine unmapping and remapping 21 GB every ~34 s --
# 13% of wall clock -- because the session never opened. Forced on here, and the
# run prints which state it is actually in ([rollout-session] lines) so the next
# one cannot be silently wrong again.
export ROLLOUT_KEEP_VLLM_AWAKE=1

# Per-turn preproc / gen / decode / envstep breakdown plus the GPU util measured
# during generation. It is a print at the end of each rollout, so it costs
# nothing, and it is the only instrument that says how much of an eval is spent
# waiting on the environment rather than generating. On by default here for the
# same reason the training arm leaves it off: this is the arm where the answer
# is not already known.
export ROLLOUT_TURN_TIMING="${ROLLOUT_TURN_TIMING:-1}"

# The turn table's genGPU% and perGPU% columns are filled by the NVML sampler in
# verl/utils/gpu_profiler.py, which is a no-op unless GPU_PROFILER=1. Without it
# the columns still print, as "-" -- a header with nothing under it, which is the
# same kind of silent hole as a session that never opened. They are the only
# instrument that says whether the GPU is FED during generation, as opposed to
# merely busy: a search batch's last turns run 11 and 27 trajectories in a slot
# sized for 126, and NVML calls that 100% either way. Sampling is one NVML read
# every 0.3 s on a background thread.
export GPU_PROFILER="${GPU_PROFILER:-1}"

# VAL_PIPELINE_DEPTH=2 keeps a second validation batch in flight, so one batch's
# environment, tokenising and scoring overlap the other's generation.
#
# MEASURED, same checkpoint and same retriever, on the WALL lines' s/batch:
# 17.0 s per batch at depth 1 against 14.2 at depth 2, a 16.5% shortening of the
# evaluation. Most of what it recovers is not in the turn table at all: a search
# batch's own span is 14.1 s, so 2.9 s of every 17.0 is spent between batches,
# on the main thread, decoding 126 prompts for the next batch and 126 responses
# from the last one. The GPU has nothing to do for any of it.
#
# Read s/batch and not the occupancy ratio, and never compare a span against an
# s/batch: doing that is what produced a wrong "no effect" reading of this
# change once already.
#
# Scored rows are unchanged: batches retire in submission order, the accumulation
# stays on the main thread, and the extra slots only serve tasks whose episodes
# come entirely from their own row (search). alfworld keeps a single manager --
# its games are indexed by position within the manager, so a second one would
# score a different set.
#
# The cost is a second set of search environments. Set it to 1 on a box that
# cannot spare them.
export VAL_PIPELINE_DEPTH="${VAL_PIPELINE_DEPTH:-2}"

# HOW MUCH OF THE CARD vLLM GETS.
#
# 0.6 is the training arm's number, and it is right there: that arm keeps FSDP
# parameters, gradients, optimizer state and a 136.5 GiB teacher pool on the same
# cards. An eval process holds none of them, so 0.6 leaves about a third of every
# card idle. 0.75 gives vLLM 36 GiB of the 48, which leaves 12 for FSDP, the CUDA
# context and NCCL -- measured non-KV use inside the budget is 10.9 GiB, so the
# KV cache goes from 159,600 tokens per GPU to roughly 224,000.
#
# It does NOT speed up the batch that runs today: the heaviest search turn uses
# 118,000 of the 159,600 it already had. It buys the headroom the NEXT width
# needs, below.
#
# Drop it back to 0.6 if vLLM refuses to start -- that is what an over-subscribed
# card looks like, and it fails at init rather than part-way through.
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"

# SEARCH BATCH WIDTH -- 252 is the default now, in both this repo's pinned config
# and the run script, because it was measured: 413 batches become 208 and ms/row
# does not move. It had been living on a command line behind EXPECTED_CONFIG_WAIVE.
#
# A search batch's later turns decode for a handful of trajectories in a slot
# sized for all of them: measured, the last two turns take 46% of a batch's
# generation time to carry 14% of its work, and a decode step costs the same
# whether it carries ten sequences or a hundred. So those turns are nearly free
# to widen.
#
# alfworld and webshop stay at 126: their environment managers are built at this
# size and alfworld's games are indexed by position within its manager. search's
# rows and their order do not change, only how they are grouped, so its score
# does not move.
#
# THE NEXT EXPERIMENT IS 378, and the KV budget above is what makes it possible.
# The heaviest turn at 252 is 64 active at ~1,330 prompt tokens plus 512 of
# response, about 118,000 of 159,600 -- 74%. At 378 that is about 156,500, which
# was too close to 159,600 to try, and is 70% of the 224,000 that 0.75 gives.
# Change it in BOTH places (they are checked against each other) and change
# NOTHING ELSE in the same run:
#
#   expected_multitask_sft_config.yaml, and run_multitask_sft_qwen3.sh:
#     env.multitask.val_per_task_batch_size: {alfworld: 126, search: 378, webshop: 126}
#
# Compare on ms/row from the WALL lines, NOT on s/batch or batch number --
# widening turns 208 batches into 139, so neither is the same quantity twice.

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
