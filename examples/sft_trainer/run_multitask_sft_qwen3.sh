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
# own pid-suffixed file. Do not summarise the trace by counting samples under a
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
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
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
