set -x

# Multitask SFT (behaviour cloning on teacher trajectories), Qwen3-1.7B.
# The student is trained off-policy with cross-entropy on teacher tokens; the
# hard-target sibling of off-policy distillation (same data, only the loss
# differs).
#
# THIS SCRIPT HAS NO STAGE 1. It reads the *same* teacher-trajectory pool the
# off-policy KD arm uses:
#
#     $HOME/data/verl-agent/sdar_multitask/teacher_traj
#
# generated once by examples/opd_trainer/run_multitask_opd_offpolicy_qwen3.sh.
# Sharing it is the stronger comparison, not just the cheaper one: SFT and KD
# then differ in exactly one thing, the loss, with the trajectories, the
# teachers, the sampling seed and the truncation all identical. Generating a
# second pool would have re-rolled every trajectory under a different RNG
# stream, so any SFT-vs-KD gap would carry a data-sampling difference inside it.
#
# The KD pool carries teacher_topk_logprobs / teacher_topk_ids (it was written
# with collect_topk=True). The SFT loss is a plain NLL on the teacher's sampled
# tokens and never reads them, so MultiTaskSFTTrainer drops both columns as each
# file loads — see its ``_drop_tensor_keys``. Nothing about the trajectories the
# SFT arm trains on differs from the KD arm's.
#
# EVERY parameter lives as a literal argument of the python3 command below —
# there is deliberately NO variable block, NO ${VAR:-default} fallback and NO
# shared-args array, so there is exactly one place to read or edit a setting.
# One-off overrides can be appended on the command line (trailing "$@") and
# still pass the expectations check.
#
# INTENT LOCK (examples/sft_trainer/):
#   expected_multitask_sft_config.yaml — Stage-2 training knobs, validated by
#     main_sft_multitask AFTER its loss injection.
# To change a scientific knob: edit the argument below AND the expectations file
# in the same commit; a script-only edit refuses to start.
#
# The pool's own knobs are NOT under this lock — they belong to the run that
# produced it, examples/opd_trainer/run_multitask_offpolicy_qwen3.sh, which has
# no expectations file. Two of them this arm depends on, and neither is checked
# automatically, so verify them before starting:
#   * 36000 trajectories per task (that script's GEN_TRAJ_PER_TASK; its default
#     is 2400). At 15*8=120 trajectories/task/step over 300 steps, 36000 is
#     exactly one epoch. A smaller pool silently becomes replay.
#   * the same teachers, seed=1 and truncation as below.
# scripts/inspect_teacher_pool.py reports the per-task trajectory counts, and the
# host RAM this run will hold.
#
# The pool is SHARDED: the generator flushes every gen.shard_every_steps steps,
# so it is <task>_0000.pt ... <task>_0029.pt, 30 per task, not one file per task.
# Stage 2 loads shards without concatenating them, which is what keeps the peak
# at 'resident + largest shard' (~9 GiB) instead of 'resident + whole task'
# (~237 GiB for alfworld). Keep every shard of a task in this directory, and keep
# nothing else: the loader globs *.pt, and a stale or duplicated shard is caught
# only by a traj_uid collision check, which a *different* run's shards would pass.
#
# STARTUP COST. Loading reads all 339.5 GiB to keep 139.2 GiB: the rest is the
# padding rows and the columns this arm's loss never reads. Paying that on every
# start (and every restart) is avoidable -- do the filtering once and point this
# run at the result:
#
#   python3 scripts/cache_teacher_pool.py \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj_sft_cache --arm sft
#   bash examples/sft_trainer/run_multitask_sft_qwen3.sh \
#       ++algorithm.sft.data_dir=$HOME/data/.../teacher_traj_sft_cache
#
# Note the DOUBLE plus. The argument below already adds algorithm.sft.data_dir,
# so a trailing '+algorithm.sft.data_dir=' is a second append of a key that now
# exists and Hydra refuses it; '++' means append-or-override. The same applies to
# every other '+' argument here that a one-off run wants to point elsewhere.
#
# The cache is the same DataProto the loader builds today, one file per source
# file with the same name and row order, so the draws are unchanged (asserted in
# tests/trainer/test_cache_teacher_pool.py). It is ARM-SPECIFIC: the SFT cache has
# no teacher top-k, so a KD run must not read it.
#
# Throughput mechanisms (process env vars, accuracy-preserving; live in code, not
# in the expectations files — see docs/optimization_phase2.md). The first two are
# exported below so they are on without being remembered; set either to 0 to
# disable:
#   ROLLOUT_KEEP_VLLM_AWAKE=1
#   OFFPOLICY_BATCH_PREFETCH=1   — builds step k+1's batch on a background thread
#     while step k is inside update_actor (a blocking ray.get, so it holds no
#     GIL). Bit-identical to the sequential path; see _prepared_batch_iter for
#     the two RNG invariants that make that true.
#   OFFPOLICY_ACTOR_PIPELINE=1   — dispatches step k+1 to the workers before
#     waiting on step k. A Ray actor runs its calls one at a time and in order,
#     so the queued one starts the instant k returns instead of after the driver
#     has reduced k's metrics and re-serialised ~480 MB. That driver window is
#     the once-per-step all-GPUs-at-zero dip, and the longest dips in the run
#     (0.23 s trace: 98.4% mean, every dip together 0.46%). Same batches, same
#     order, same worker — the actor cannot start k+1 before k returns. It does
#     NOT run ahead across a step that saves or validates, since a checkpoint
#     taken with a call already queued would hold a later step's weights.
#   (ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY /
#    ROLLOUT_COMPACT_RECORD default to on; they speed up the validation rollouts)
#   NOTE: leave ROLLOUT_PREFETCH_LOGPROB and ENV_RESET_PREFETCH off here —
#   this stage has neither an old_log_prob phase nor a per-step train rollout.
#   TASK_BALANCE_INTERLEAVE does nothing here: it reorders the *train* sampler,
#   which this loop never iterates (it draws from the fixed pool), and the
#   validation dataloader takes no sampler at all.
#
# VALIDATION IS NOT PART OF THIS RUN (trainer.test_freq=-1). It happens after the
# fact, one process per checkpoint, via examples/sft_trainer/eval_checkpoints.sh.
#
# The reason is measured, on the 300-step run that used test_freq=150 (wandb
# x7g9r7bx, 42.5 h wall):
#
#   phase          wall     share   mean GPU util
#   training      34.1 h    80.4%      96.5%
#   validation     7.6 h    18.0%      46.0%   <- 2 passes, 3.8 h each
#   checkpoint     0.7 h     1.6%       2.1%
#   whole run     42.5 h               85.8%
#
# Two validations cost 7.6 h of the 42.5 h, and 68.6% of all the GPU time the run
# left on the floor. They are that slow because validation is not this arm's loss:
# it is a full agentic rollout (126 episodes x 3 tasks, alfworld to 50 turns),
# whose turns alternate vLLM decode with env.step -- 42% of that window has the
# GPUs at under 10%. Nothing about it depends on being inside the training loop:
# it reads a checkpoint, and save_freq=25 writes twelve of them.
#
# Moving it out leaves the training run at a measured 96.5% for its whole length
# and cuts 7.6 h off it, and the evaluation itself gets *more* correct: a fresh
# process rebuilds the val envs from env.seed, so every checkpoint is scored on
# the same episodes. Two validations in one process are not -- alfworld's
# TextWorld game-file cycle is stateful and advances on each reset, so the second
# pass draws different games than the first.
#
# CHECKPOINTS ARE WRITTEN IN THE BACKGROUND
# (actor.checkpoint.async_save=True). A save took 198 s in the run above, of which
# only the first ~20 s used the GPU -- building the sharded state dict and copying
# it to host memory. For the other ~178 s the cards sat at 0.0% SM and their 28 W
# idle floor while torch.save pickled to disk, twelve times over. The write now
# runs on a background thread over those CPU copies, and the training loop goes
# straight on to the next step.
#
# The same bytes land in the same files; what moves is when. The one visible
# consequence is that latest_checkpointed_iteration.txt is published a step late,
# by _flush_pending_checkpoint, because a tracker written before the shards are on
# disk is a tracker that can name a half-written checkpoint to a resume.
#
# DO NOT set GPU_PROFILER=1 for a real run. The profiler is entirely inert
# without it (verl/utils/gpu_profiler.py; the phase tags in dp_actor.py return
# immediately), but when on it starts an NVML sampler in the driver and in rank
# 0, prints a table every step, and with GPU_PROFILER_SYNC_PHASES=1 inserts a
# device synchronize at every phase boundary — which serializes work the run
# would otherwise overlap. It is a measurement tool; leave it off and the run is
# byte-identical to one built without it.
#
# TO MEASURE WHERE THE REMAINING ~1.5% GOES, run with
#
#   GPU_PROFILER=1 GPU_PROFILER_INTERVAL=0.2 GPU_PROFILER_TRACE=/tmp/trace.csv
#
# and read the traces with scripts/gpu_stall_scan.py /tmp/trace.*.csv. The glob
# is not optional: there are two samplers, one per process, and each writes its
# own pid-suffixed file.
#
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It is the
# obvious answer to the allocator reserving more than the model needs, and vLLM
# refuses to start under it:
#
#   AssertionError: Expandable segments are not compatible with memory pool.
#   (vllm/device_allocator/cumem.py, pytorch/pytorch#147851)
#
# vLLM v1 allocates through its own CuMemAllocator so sleep()/wake_up() can
# unmap physical pages, and that allocator asserts expandable segments are off.
# This arm never generates -- it trains from pre-made trajectories and validates
# out of band -- but the rollout is still constructed on every rank, so the
# assert fires at init_model, before the first step.
#
# ACTOR_PASS_CU_SEQLENS=1 (the default) hands the packed-sequence boundaries to
# the attention instead of letting it re-derive them. Handed position_ids, HF's
# flash-attention path works out on the DEVICE whether the sequences are packed
# and how long the longest is, then reads both on the HOST because flash-attn
# needs Python ints -- one device-to-host sync per layer per forward, doubled
# because gradient checkpointing recomputes the forward inside the backward. The
# trace measures ~80 D2H copies per micro-batch, each trailed by ~147 us with the
# device empty: 0.38% of wall, about a third of the whole idle deficit.
# unpad_input already computed both, once. Set ACTOR_PASS_CU_SEQLENS=0 to rule it
# out -- if the two ways of deriving the boundaries ever disagreed the symptom
# would be a loss curve rather than an exception. It is off automatically under
# Ulysses SP (the sequence is split after that point, so the boundaries would be
# stale) and on a transformers whose entry point does not name the kwargs.
#
# ACTOR_GC_FREEZE=1 (the default) runs one gc.collect() and then gc.freeze()
# once the model, optimizer and FSDP wrap exist, so generation-2 collections
# stop walking objects that live for the whole run and can never be freed. The
# sweep freezes the interpreter, and the forward has synchronisation points that
# drain the launch queue, so a host stop of that length lands on the device as
# an equally long stop: one card at sm 0 while the other two spin in the
# collective reading 100. That is the measured signature of the 14 solo
# excursions of 0.6-0.8 s. Startup prints "[host-gc] rank N: froze X objects",
# and X * 0.12 us (measured cost per tracked object) is the sweep cost removed
# -- which is also the number that decides whether gen-2 GC could have been the
# cause at all. ACTOR_GC_FREEZE=0 restores stock behaviour exactly, so the pair
# is the A/B; stall/gc_gen2 is already logged per rank per step.
#
# ACTOR_GC_MANUAL=1 (off) additionally disables automatic collection and runs
# one explicit sweep per step at the boundary, where before-step is 0.24-0.71 s
# of already-lost device time. It moves the residual cost rather than removing
# it, and it lets a cycle-heavy step grow the heap until the next boundary, so
# it stays off until measured.
#
# A MEASUREMENT RUN IS NOT THE BARE SCRIPT. Three of this file's defaults are
# production defaults, and each one silently changes what a measurement means.
# Copy this whole block rather than the parts you remember:
#
#   bash examples/sft_trainer/run_multitask_sft_qwen3.sh \
#     ++algorithm.sft.data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_probe \
#     ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
#     ++trainer.resume_mode=disable
#
#   data_dir            the default is the full pool: 339.5 GiB read to keep
#                       139.2 GiB, on every start. The probe pool is the same
#                       phenomenon (the dips were first seen on it) at a
#                       fraction of the startup.
#   micro_batch_size    the default 10 measures MFU 0.292; 20 measures 0.317,
#                       and 20 is the maximum (ppo_mini_batch_size 60 over 3
#                       ranks is 20 rows). Comparing a run at 10 against a
#                       remembered number at 20 is comparing nothing.
#   resume_mode         the config default is auto, so a stale global_step_N in
#                       trainer.default_local_dir is picked up silently: the run
#                       starts mid-training, replays N draws (~2 s each), and the
#                       first step's timings carry the replay.
#
# THE STALL WATCH IS ON BY DEFAULT and needs nothing set. It times every
# micro-batch of every step on every rank with two CUDA events, reads them back
# only once they have completed (query(), never synchronize, so it cannot create
# the stall it is looking for), and prints a line when one is an outlier against
# the running median:
#
#   [stall] rank 1 micro 812 at 1787038…: gpu 5900 ms (median 1010),
#           gap 12 ms (median 3), host 40 ms -> inside the micro-batch, host ran ahead
#
# THE LOG FILES ACCUMULATE ACROSS RUNS. The name is rank<N>_pid<P>.log, so each
# run writes new files beside the old ones and `grep /tmp/actor_stall/rank*.log`
# reads every run this machine has ever done. A stale line is not obviously
# stale -- it carries an epoch timestamp, not a date. Ask the run which files
# are its own instead of globbing:
#
#   grep -h "stall-watch" /tmp/ray/session_latest/logs/worker-*.out
#   # [stall-watch] rank 0 logging to /tmp/actor_stall/rank0_pid601985.log
#
# gap vs gpu says whether the device was idle BEFORE the micro-batch or slow
# inside it, and gap/<kind> says which of the three places the idle came from:
# gap/step spans two update_policy calls (the return to the driver, its logging,
# the next batch's dispatch and H2D), gap/mini holds the gradient reduce and the
# optimizer step, gap/interior is between two micro-batches of one mini-batch.
# Each is judged only against its own kind -- they differ by construction and
# happen on a fixed schedule, so one shared threshold reports the boundary every
# step and buries the events that matter. host vs gpu says whether the host was
# blocked too, which is the difference between something upstream of the device
# and a collective wait.
#
# It also prints one line per step whatever happens:
#
#   [step-gpu] rank 0 step 12: 66 micro-batches, in-micro 312.4 s, outside 4.81 s
#              (before-step 1.52, before-mini 2.10, interior 1.19; optim 4.20,
#               unaccounted 0.61 s) = 0.19% idle of 317.2 s
#
# "outside", not "idle": the gradient reduce and the optimizer step run in the
# window between two micro-batches, and on a 1.7B model sharded three ways that
# is 50-80 ms of real kernels per mini-batch -- 5.75 s of a 346 s step. Reported
# as idle it puts a 1.7% noise floor under everything. optim is measured
# separately; unaccounted is what is left, and that is the real idle.
#
# That is the half the outlier detector cannot do. An outlier needs the seconds
# to be concentrated in one micro-batch; the same seconds spread thinly over
# sixty-six of them cost exactly as much and trip nothing. The running total sees
# both, and splits the idle into the three places it can come from. Tune with
# ACTOR_STALL_FACTOR (default 3.0, x the running median) and ACTOR_STALL_MIN_MS
# (default 500); ACTOR_STALL_FACTOR=0 turns it off.
#
# Every line also goes to ACTOR_STALL_DIR/rank<N>_pid<P>.log (default
# /tmp/actor_stall), because the console shows one rank of three: Ray's log dedup
# matches the message with its numbers substituted, so the three ranks' lines are
# one pattern to it and two are collapsed into "[repeated 2x across cluster]".
# Comparing ranks is the point, so read the files:
#
#   tail -f /tmp/actor_stall/rank*.log
#   grep -h "^\[step-gpu\]" /tmp/actor_stall/rank*.log | sort -t' ' -k5 -n
#
# For a run already going without those files, Ray's own per-worker logs have the
# undeduplicated stdout:
#
#   grep -h "step-gpu" /tmp/ray/session_latest/logs/worker-*.out
#
# This is the instrument for the dips, and the capture backends below are not.
# The dips are five events in 68 minutes at unpredictable positions inside their
# steps; a capture window pinned to micro-batch 40 of step 1 has essentially no
# chance of holding one. The first Nsight capture demonstrated both halves of
# that: it caught no dip, and it WAS the largest dip in the run -- 30 s of total
# node idle inside its own window, step MFU 0.278 against 0.317 for every step
# after it. Use the watch to find which micro-batch, then aim a capture at it.
#
# WHEN NVML IS NOT ENOUGH, trace a few micro-batches of every rank. The sampler
# above cannot go below ~330 ms and one micro-batch's forward is ~1.1 s of about
# 500 kernels, so a 0.3-1.0 s stall lands inside a single actor.fwd sample and
# cannot be placed more precisely than "in that phase". Two backends, same
# window, and both are inert unless their own count is set:
#
#   ACTOR_NSYS_MICRO=20 ACTOR_NSYS_SKIP=40 bash examples/sft_trainer/run_...sh
#
# puts every rank under Ray's _nsight plugin, opens a capture range around that
# many micro-batches, and names the phases with NVTX -- which is what makes the
# collectives visible by name, and so what decides whether one rank is slow or
# the other two are waiting for it. Reports land one per process in the Ray
# session's logs/nsight directory. Nsight sees the most (driver, OS runtime),
# but collection writes a .qdstrm that a separate QdstrmImporter has to convert,
# and on this host that conversion has failed with "Wrong event order has been
# detected" -- leaving three intact 39 MB captures nothing can read. If it does,
# ACTOR_NSYS_TRACE=cuda,cudnn,cublas,nvtx drops osrt, the suspect.
#
#   ACTOR_TORCH_MICRO=6 ACTOR_TORCH_SKIP=40 bash examples/sft_trainer/run_...sh
#
# is the backend with nothing outside the process: torch.profiler writes a
# finished Chrome trace from inside each rank, so there is no conversion left to
# fail, and record_shapes puts the per-op tensor shapes in the file -- the direct
# test of whether a slow rank was handed more tokens. It sees less than Nsight
# and its own CUPTI start-up perturbs the first captured micro-batch, so ask for
# a few more than you need and read from the second. Traces go to
# ACTOR_TORCH_DIR (default /tmp/actor_trace), one per rank; read them with
#
#   python3 scripts/actor_trace_summary.py /tmp/actor_trace
#
# which prints, per micro-batch and per rank, the device's real busy time (the
# union across streams, not the sum), the NCCL share, and the idle complement --
# then names the rank everyone else waited for and splits its lateness into
# extra work versus a stall, because those want different fixes. Keep MICRO
# small: the trace is a few hundred MB for a handful of micro-batches.
#
# Measurement only: leave all of them unset and not a single NVTX push or
# profiler call happens. Do not summarise the NVML trace by counting samples under a
# threshold — utilization.gpu is the busy fraction of a trailing window, so a
# stall shorter than that window never even reads 0 and time-under-a-line
# reports a fraction of it. The scan integrates the deficit instead, which is
# what makes a 0.2 s trace and wandb's 15 s system metrics agree.

# The one variable in this file, and it is not a knob: an absolute path to this
# script's own directory. The expectations file is read inside a Ray actor, after
# Hydra has chdir'd the driver into its output directory, so a path relative to
# the launcher's cwd is not reliably resolvable by the time it is opened.
SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALFWORLD_DATA=$HOME/data/alfworld
# A placeholder here is worse than nothing. wandb validates WANDB_API_KEY before
# it consults any other source, so "your_key_here" does not merely fail to
# authenticate -- it hard-fails init and blocks the ~/.netrc that `wandb login`
# writes. And it fails LATE: the workers build, the model loads, vLLM starts,
# and the run dies in trainer.fit() at the Tracking() call, minutes in, with a
# traceback that names an authentication error rather than a missing export.
# Leave the variable unset instead and let wandb find the key itself.
if [ -z "${WANDB_API_KEY:-}" ]; then
  unset WANDB_API_KEY
  # -qs returns non-zero for a missing file as well as for a missing entry, which
  # is the condition wanted: warn unless a wandb credential is actually there.
  if ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "[wandb] no WANDB_API_KEY and no ~/.netrc entry -- run 'wandb login' once," >&2
    echo "        or export WANDB_API_KEY before this script, or pass" >&2
    echo "        trainer.logger=[console] to run without wandb." >&2
  fi
fi
# On by default, for the same reason the two FSDP knobs are literals below: a
# 300-step run gets restarted, and a mechanism that has to be exported by hand is
# one that will eventually be missing from a restart. Both are accuracy-
# preserving (see the header), so this changes throughput and nothing else.
# ROLLOUT_KEEP_VLLM_AWAKE=0 / OFFPOLICY_BATCH_PREFETCH=0 still turns either off.
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export OFFPOLICY_BATCH_PREFETCH=${OFFPOLICY_BATCH_PREFETCH:-1}
export OFFPOLICY_ACTOR_PIPELINE=${OFFPOLICY_ACTOR_PIPELINE:-1}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# Data prep — same prompts/tasks as the OPD / offline-KD runs. These literals
# are cross-checked by the expectations files (per_task_batch_size=15,
# val_per_task_size=126, total_training_steps=300, seed=1).
python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps 300 \
    --per_task_batch_size 15 \
    --env_train_per_task_size 15 \
    --val_per_task_size 126 \
    --seed 1

# ===================== Stage 2: SFT (cross-entropy on teacher tokens) =====================
python3 -m verl.trainer.main_sft_multitask \
    +trainer.expected_config=$SFT_DIR/expected_multitask_sft_config.yaml \
    data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
    data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
    data.train_batch_size=45 \
    data.val_batch_size=126 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.task_balance.enable=True \
    data.task_balance.per_task_batch_size=15 \
    +data.task_balance.num_batches=300 \
    data.task_balance.tasks=[alfworld,search,webshop] \
    +data.task_overrides.alfworld.max_prompt_length=2048 \
    +data.task_overrides.alfworld.truncation='error' \
    +data.task_overrides.search.max_prompt_length=4096 \
    +data.task_overrides.search.truncation='left' \
    +data.task_overrides.webshop.max_prompt_length=4096 \
    +data.task_overrides.webshop.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.seed=1 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.ppo_mini_batch_size=60 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.no_sync_grad_accum=True \
    actor_rollout_ref.actor.response_only_logits=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.max_model_len=4608 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.return_rollout_log_probs=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    env.env_name=multitask \
    env.seed=1 \
    env.max_steps=50 \
    env.history_length=4 \
    env.rollout.n=8 \
    env.search.search_url='http://100.86.45.30:8001/retrieve' \
    env.search.timeout=600 \
    env.search.max_retries=null \
    env.multitask.tasks=[alfworld,search,webshop] \
    env.multitask.max_steps.alfworld=50 \
    env.multitask.max_steps.search=4 \
    env.multitask.max_steps.webshop=15 \
    +env.multitask.history_length.alfworld=2 \
    +env.multitask.history_length.search=4 \
    +env.multitask.history_length.webshop=2 \
    env.multitask.val_per_task_batch_size=126 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.n_gpus_per_node=3 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    +algorithm.sft.data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +algorithm.sft.loss_coef=1.0 \
    +algorithm.sft.num_epochs=1 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.invalid_action_penalty_coef_by_task='{alfworld:0.1,search:0.01,webshop:0.1}' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_sft_multitask' \
    trainer.experiment_name=sdar_multitask_sft_multitask_qwen3_1.7b \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_sft_multitask \
    actor_rollout_ref.actor.checkpoint.async_save=True \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
# NOTE: trainer.total_training_steps is fixed at 300. With per_task_batch_size=15
# and env.rollout.n=8, each step draws 15*8=120 trajectories/task, so a
# 36000-trajectory pool is consumed exactly once over the 300 steps: one epoch, no
# replay. This is the same pool and the same horizon as the off-policy KD arm, and
# the ~1-epoch regime reported for agentic off-policy distillation, where reusing
# a smaller pool measured worse.
