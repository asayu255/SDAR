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

# RAY'S TEMP DIRECTORY IS PER USER, because /tmp/ray is not.
#
# On a shared box every user's Ray defaults to the same /tmp/ray, and a raylet
# that connects to another user's runtime-env agent gets a protocol mismatch:
#
#   runtime_env_agent_client.cc:339: Runtime Env Agent timed out in 30000ms.
#   Status: Disconnected: on_read bad version, bytes_transferred 0
#   The raylet exited immediately -> node marked dead -> ActorDiedError
#
# which cost a 504 run at 21:49 with two other users' jobs on the machine. It
# is not a load problem -- that box was 50% idle with 201 GiB free -- and not a
# stale session of ours. It is two Ray versions sharing one socket directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$(id -un)}"
mkdir -p "$RAY_TMPDIR" 2>/dev/null || true

# AND SAY WHAT ELSE WAS ON THE MACHINE, in the log, at the start.
#
# Every judgement in this arm is ms/row and wall on a box other people use. A
# run taken beside a job holding fifty cores is not comparable to one taken on
# a quiet machine, and nothing in the log said which had happened -- so a 1.5%
# difference could be the change or could be the neighbours. One line, so the
# measurement carries its own context.
# SAMPLING INTERVAL -- 0.1 s here, against the library default of 0.3 s.
#
# 0.3 s is right for "how much of the run was idle". It is not enough for "which
# 300 ms dip, and what was outstanding during it": a 300 ms stall gets zero or
# one sample inside it, so the gauges that would explain it are as likely to be
# missed as caught. At 0.1 s a 300 ms stall gets about three and a 500 ms one
# about five.
#
# Finer than 0.1 buys little: NVML's utilization.gpu is a moving-window average
# over roughly 1/6 s to 1 s depending on the card, so a 10 ms poll returns the
# same smoothed number several times rather than 10 ms of detail. It would still
# raise the chance of catching a gauge transition -- that is the only reason to
# go lower, and it costs proportionally more memory and CPU for it.
#
# Costed before being turned on: the per-sample stack state is stored as an
# interned id rather than one string per thread, which is about 1 MB over a
# 3800 s run against about 0.3 GB if the strings were kept.
export GPU_PROFILER_INTERVAL="${GPU_PROFILER_INTERVAL:-0.1}"

# Set GPU_PROFILER_TRACE to get the per-excursion table:
#
#   GPU_PROFILER_TRACE=/tmp/trace.csv bash examples/sft_trainer/eval_checkpoints.sh 300
#   python scripts/gpu_stall_scan.py /tmp/trace.csv
#
# THE FILE ON DISK IS NOT THAT NAME. The profiler writes one trace per process
# and puts the pid in it -- /tmp/trace.264168.csv -- because a driver sampler
# and a worker sampler opening one path "w" overwrite each other. The line
#
#   [gpu-profiler] per-sample trace -> ...
#
# in the log says the real name. The scanner resolves the sibling itself, so
# the command above works; `ls /tmp/trace.*` is the direct way to see them.

_CORES=$(nproc 2>/dev/null || echo '?')
echo "[eval] machine     : $_CORES cores, load$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null | sed 's/^/ /')" \
     "-- a loaded box moves ms/row; compare runs taken under similar load"

# HOW MUCH OF THE CARD vLLM GETS.
#
# 0.6 is the training arm's number, and it is right there: that arm keeps FSDP
# parameters, gradients, optimizer state and a 136.5 GiB teacher pool on the same
# cards. An eval process holds none of them, so 0.6 leaves about a third of every
# card idle. Measured non-KV use inside vLLM's own budget is 10.9 GiB, and the
# two pinned points (0.6 -> 159,600 tokens, 0.75 -> 224,000) give 8,920 KV
# tokens per GiB, so the budget is (util * 48 - 10.9) * 8920.
#
# 0.85 gives vLLM 40.8 GiB of the 48 and 266,700 KV tokens, leaving 7.2 GiB for
# FSDP, the CUDA context and NCCL. It is here because the search width is 504
# and 504 does not fit below it -- see the width check immediately below. It is
# not free headroom to spend on something else.
#
# Drop it to 0.75 (and the width back to 378) if vLLM refuses to start -- that is
# what an over-subscribed card looks like, and it fails at init rather than
# part-way through, which is the good failure.
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.85}"

# THE WIDTH AND THE BUDGET ARE ONE DECISION, so they are checked as one.
#
# Peak KV per row of batch width reads 414 tokens (what the 378 estimate
# implies) to 468 (linear in the 252 measurement), and no run so far can tell
# them apart because both fit at 378. Sized on the conservative one, 504 needs
# 235,900 tokens: 105% of what 0.75 gives and 96% of 0.80, against 88% of 0.85.
#
# 96% is where vLLM starts preempting, and a preempted sequence is recomputed
# from scratch -- which would cost more than the 1.4% the width is worth. This
# refuses at once rather than producing a slow run that looks like a result.
# Anchored to the setting. A bare "search:[0-9]*" also matches
# invalid_action_penalty_coef_by_task='{...,search:0.01,...}' further down the
# same file, and would read a width of 0 the day those lines are reordered.
_WIDTH=$(grep -o "val_per_task_batch_size='{[^}]*}'" "$RUN_SCRIPT" \
    | grep -o "search:[0-9]*" | cut -d: -f2)
# DEPTH BELONGS IN THIS SUM AND WAS NOT IN IT. The 468 tokens per row of width
# is an OBSERVED PEAK measured at VAL_PIPELINE_DEPTH=3 with the pump on, so it
# already contains three slots' worth of concurrency -- and the pump submits
# every row as its own request with ROLLOUT_PUMP_MAX_IN_FLIGHT=0, letting vLLM
# decide what fits, so slots really do stack in the engine. Scaling by width
# alone is therefore right at depth 3 and wrong at any other depth: 504 x 4
# would have passed this check while asking for a third more than 504 x 3.
#
# In row-slots, which is the quantity that matters:
#     504 x 3 = 1512      378 x 4 = 1512      504 x 4 = 2016
_DEPTH="${VAL_PIPELINE_DEPTH:-3}"
if [ -n "$_WIDTH" ]; then
    _NEEDED=$(awk -v w="$_WIDTH" -v d="$_DEPTH" 'BEGIN{print w * 468 * d / 3}')
    _BUDGET=$(awk -v u="$ROLLOUT_GPU_MEM_UTIL" 'BEGIN{print (u * 48 - 10.9) * 8920}')
    if awk -v n="$_NEEDED" -v b="$_BUDGET" 'BEGIN{exit !(n > 0.92 * b)}'; then
        echo "[eval] search width $_WIDTH at depth $_DEPTH needs up to $(printf '%.0f' "$_NEEDED") KV tokens," >&2
        echo "       and ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL gives $(printf '%.0f' "$_BUDGET")." >&2
        echo "       That is over 92% and vLLM will preempt, which costs more than the width" >&2
        echo "       is worth. Raise ROLLOUT_GPU_MEM_UTIL, or lower the width in BOTH" >&2
        echo "       run_multitask_sft_qwen3.sh and expected_multitask_sft_config.yaml." >&2
        exit 1
    fi
fi

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

# VAL_PIPELINE_DEPTH keeps more than one validation batch in flight, so one
# batch's environment, tokenising and scoring overlap another's generation.
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
# WHY 3 AND NOT 2. This is the only mechanism in the run that removes GPU-empty
# windows, and its effect is geometric, so one more slot is worth more than
# anything else on this list.
#
# A slot alternates generate (GPU) and env.step (not GPU). Call p the share of a
# slot's time outside the GPU. Every card is empty only when EVERY slot is
# outside at once, and the slots are independent, so that is p^depth:
#
#   MEASURED at depth 2 (NVML, run ...-201115, post-startup): every card empty
#   7.4% of samples -> p ~= 0.27.
#
#   depth 2 -> 7.4%      depth 3 -> 2.0%      depth 4 -> 0.5%
#
# Nothing else on this list is worth 5 points. Bounding a stalled socket
# (SEARCH_TCP_USER_TIMEOUT_S) removes the pathological tail of p; depth removes
# the ordinary body of it, which is much larger.
#
# It does NOT change generation. Batches keep their own rows, the worker group
# still runs one generate at a time, retirement is in submission order and the
# accumulation stays on the calling thread -- depth only changes which batch is
# waiting on its environment while another generates. (ROLLOUT_MERGE_GENERATES
# is the one that changes generation, measured, and is why it is not a default.)
#
# The cost is one more set of search environments and their KV cache, which is
# what ROLLOUT_GPU_MEM_UTIL above pays for. Set it to 1 on a box that cannot
# spare them; alfworld keeps a single manager either way, so only search (411 of
# 413 batches) uses the extra slots.
#
# Read [gpu-residency] to see whether it worked. Do NOT read [val-pipeline]'s
# "NOTHING running" for this -- a slot blocked in env.step is running by that
# measure while every card is idle, which is how depth looked unnecessary once
# already.
export VAL_PIPELINE_DEPTH="${VAL_PIPELINE_DEPTH:-3}"

# THE PUMP. Engines as a pool: requests go in individually and each rank's
# TokenPump keeps its engine stepping, so a rank that finishes its share of a
# collective generate takes the next slot's work instead of waiting for the
# slowest rank.
#
# ON by default because wall says so, across three completed runs of the same
# checkpoint on the same three cards:
#
#   pump ON   0.96 h   success 0.3875   search 0.3571
#   pump OFF  1.24 h   success 0.3865   search 0.3560   (V0 + multi-step)
#   pump OFF  1.26 h   success 0.3869   search 0.3523   (V1, depth 2)
#
# 24% shorter with the scores inside 0.001 of each other, and greedy search --
# the one task that cannot move by sampling -- highest on the pump run.
#
# ITS UTILISATION IS THE WORST OF THE THREE (79.90% against 83.21%), which is
# the point: this arm has chosen on utilisation three times and this is the case
# where doing so picks the setting that takes 29% longer for the same answers.
#
# REQUIRE is on with it, deliberately. Without it a pool that refuses falls back
# to the blocking path in silence, and the run completes looking ordinary while
# measuring the thing it was supposed to be compared against -- which cost one
# entire run when eos_token_id turned out to be a list. Set
# ROLLOUT_ASYNC_GENERATE=0 to turn the pump off; do not turn REQUIRE off to
# paper over a refusal.
export ROLLOUT_ASYNC_GENERATE="${ROLLOUT_ASYNC_GENERATE:-1}"
if [ "$ROLLOUT_ASYNC_GENERATE" != "0" ]; then
    export ROLLOUT_ASYNC_REQUIRE="${ROLLOUT_ASYNC_REQUIRE:-1}"
fi

# SEARCH BATCH WIDTH -- 504, in both this repo's pinned config and the run
# script. It had been living on a command line behind EXPECTED_CONFIG_WAIVE;
# every step of it since has been measured before being kept.
#
#   126 -> 252   413 batches -> 208, ms/row unchanged
#   252 -> 378   208 -> 139, ms/row 67 -> 65-66 at the same fraction of rows
#   378 -> 504   208 -> 105, expected a further ~1.4%; needs the 0.85 budget
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
# search IS NOW 504, in both places, with the budget above sized for it.
#
# 252 -> 378 was measured and kept: 208 batches became 139, ms/row read 65-66
# against 252's 67 AT THE SAME FRACTION OF ROWS, val/success_rate moved by
# 0.00002, and grep -ci preempt was 0. Predicted 2.7%, measured 1.5-3.0% --
# the first time the per-batch-cost model called a result before the fact.
#
# WHAT IT IS AIMED AT, so the result can be read: 208 batches become 105, so
# everything the driver does ONCE PER BATCH is divided and everything it does
# per row is untouched. It is not aimed at the engine duty cycle and cannot
# move it. Expected: about 1.4% over 378, which is small enough that a single
# preemption erases it -- hence the check above.
#
# Compare on ms/row from the WALL lines at the SAME FRACTION OF ROWS. Not
# s/batch, not batch number, and not [val-hash]: regrouping the same rows IS
# the change, so no two batches pair across a width change.
#
# Not aimed at, and known not to move: the ~70 s of env-manager construction,
# which is per SLOT and not per batch.
#
# THE NEXT ONE, if this pays: 580 at 0.85 is 59-66% of KV and worth about a
# further 0.5%. Below the noise of a shared box; stop here unless something
# else changes.

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
