set -x

# Stage 1 ONLY — teacher trajectory generation for the three tasks.
#
# Each task's frozen single-task RL teacher rolls out in its own environment and
# scores its own top-20 over what it generated, writing
#   $HOME/data/verl-agent/sdar_multitask/teacher_traj/<task>_0000.pt, _0001.pt, ...
# (+gen.shard_every_steps=10 below; Stage 2 globs *.pt in that directory, so the
# shard count does not matter to it. shard_every_steps=0 restores a single
# <task>.pt, but then the whole run is buffered in RAM and concatenated at the
# end — ~198 GB for alfworld, ~400 GB at the concat peak.)
# This is the expensive stage: 36000 trajectories x 3 tasks, with a full
# multi-turn rollout per step (alfworld up to 50 turns, webshop 15, search 4
# plus a retriever round trip per turn).
#
# Split out of run_multitask_offpolicy_qwen3.sh, which runs data prep + Stage 1
# + Stage 2 back to back. This script keeps that script's data prep and Stage-1
# blocks byte-identical and drops only Stage 2, so it is self-contained: run it
# on a clean machine and it builds the prompt parquets and then the full teacher
# dataset. Stage 2 then runs from run_multitask_offpolicy_qwen3_nogen.sh on the
# .pt files produced here.
#
# Prerequisites (everything else this script makes itself):
#   * the preprocessed SearchR1 corpus at --search_dir below
#   * the three teacher checkpoints exist at the +gen.teacher_path below
#   * the retriever is up at SEARCH_URL (the search task calls it every turn)
#   * ALFWORLD_DATA points at the alfworld data
#
# Each task writes its shards as it goes and is independent of the others, so a
# failure in a later task does not lose the tasks that already finished — rerun
# this script with the finished blocks commented out. Note that rerunning a task
# DELETES its existing <task>_NNNN.pt (and any legacy <task>.pt) before it
# starts, sharded or not: Stage 2 globs the directory, so a leftover shard from a
# longer previous run would otherwise be concatenated into the new dataset — and
# traj_uid is a uuid4, so the duplicate check would not catch it either; the pool
# would simply hold two generations at once. A task that died part-way therefore
# restarts from step 1, not from its last shard.
#
# EVERY parameter lives as a literal argument of the python3 commands below —
# there is deliberately NO variable block, NO ${VAR:-default} fallback, NO
# shared-args array and NO per-task loop, so there is exactly one place to
# read or edit a setting. The only env-var passthroughs are infra endpoints
# (WANDB_API_KEY, SEARCH_URL — the retriever host moves between machines),
# which are not scientific knobs.
#
# INTENT LOCK (examples/opd_trainer/):
#   expected_multitask_offpolicy_gen_config.yaml — Stage-1 dataset knobs,
#     validated by main_opd_offpolicy_gen BEFORE its single-task restriction.
# To change a scientific knob: edit the argument below AND that expectations
# file in the same commit; a script-only edit refuses to start.
#
# NO PADDING IS WRITTEN. adjust_batch(mode="copy") rounds each generation step up
# to a DP/micro-divisible size by duplicating random rows; those copies used to be
# saved into the shards, so Stage 2 trained on those turns twice and split its
# steps into more mini-batches than the real data warrants (~+10% optimizer
# updates, worst on the short-episode tasks). Stage 1 needs no divisibility — the
# top-k worker call pads and unpads itself via auto_padding_key — so it is off.
# +gen.adjust_batch=True turns it back on for one case only: extending a pool whose
# other tasks were written when it was unconditional, so the pool stays uniform in
# format. Pass +gen.expect_pad_divisor=<n> with it and Stage 1 asserts the divisor
# this config produces matches the one that pool was written with
# (scripts/inspect_teacher_pool.py reports a pool's divisor and its padding rows).
#
# gen.num_trajectories=36000 is what makes Stage 2 a single epoch: Stage 2 draws
# per_task_batch_size 15 * env.rollout.n 8 = 120 trajectories/task/step over 300
# steps, so a 36000 pool is consumed exactly once with no replay.
#
# Generation reaches that number in exactly one pass of its own prompt pool.
# Every gen step turns data.train_batch_size 15 prompts into 15 * env.rollout.n 8
# = 120 trajectories, so 36000 takes 300 steps per task:
#   search   4500 real prompts / 15 = 300 batches = 1 epoch, no prompt reused
#   alfworld 15 placeholder rows = 1 batch/epoch x trainer.total_epochs 300 = 300
#   webshop  same as alfworld (their variety is fresh episodes, not prompts)
# Both counts land on 300 with nothing to spare, so keep --total_training_steps,
# --per_task_batch_size, env.rollout.n and trainer.total_epochs consistent if you
# change any of them.
#
# TOP-K FORWARD MEMORY — why log_prob_micro_batch_size_per_gpu is 4, not 16.
# compute_topk_log_prob's forward materializes logits_rmpad of shape
# (nnz, vocab=151936) in bf16 and then torch.logsumexp allocates a second tensor
# of the same shape, so the transient peak is ~2 * nnz * 151936 * 2 bytes.
# At micro_bs=16 that overflowed a 47.5 GiB card in the real run:
#   search   2727 tok/row -> 12.35 GiB per tensor, 24.7 GiB peak -> OOM at step 82
#   webshop  3841 tok/row -> 17.39 GiB per tensor, 34.8 GiB peak -> OOM at step 1
# Budget: vLLM takes gpu_memory_utilization 0.6 * 47.5 = 28.5 GiB and holds its
# KV THROUGH this forward (free_cache_engine=False, so it is not reclaimable),
# the FSDP fp32 shard is ~2.1 GiB, leaving ~16.9 GiB.
#
# micro_bs=4 puts the peak at 8.7 GiB for webshop's rows and 10.4 GiB even in the
# worst case where every row is the full 4608 tokens, so it fits for all three
# tasks with margin. The cost is 4x more micro-batches in the top-k phase, which
# was ~35% of a search step (~15 s of ~42 s), so expect a modest step increase.
#
# gpu_memory_utilization is deliberately NOT lowered: it cannot fix this on its
# own (even 0.4 leaves only 26.4 GiB, short of webshop's 34.8 GiB peak at
# micro_bs=16) and it would shrink the KV cache that the rollout — the larger
# part of the step — depends on.
#
# Accuracy: micro-batch grouping only, and each row's forward is independent
# under flash_attn_varlen, so this is the same distribution-preserving class as
# prefix caching, not bit-identical.
#
# Throughput mechanisms (process env vars, accuracy-preserving; they live in
# code, not in the expectations files, so they are not scientific knobs — see
# docs/optimization_phase2.md for the mechanisms and docs/optimization_gen_only.md
# for why each one is on or off *for generation specifically*). Exported below
# rather than left to the caller: generation is the hours-long stage, and every
# knob enabled here is either bit-identical or in the same
# distribution-preserving class as prefix caching.
#
# NOT enabled, with the gen-only reason:
#   TASK_BALANCE_INTERLEAVE  — no-op. It only reorders rows ACROSS tasks, and
#     Stage 1 restricts task_balance.tasks to a single task.
#   ROLLOUT_PREFETCH_LOGPROB — generation has no old_log_prob phase. (The
#     analogous win is prefetching the top-k forward; not implemented yet.)
#   ENV_RESET_PREFETCH       — not wired into TeacherTrajectoryGenerator.generate();
#     setting it here would do nothing.

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. The
# torch OOM message recommends it, but vLLM's CuMemAllocator — which this config
# uses for the sleep/wake memory pool — asserts it is unset and refuses to build
# the engine (vllm/device_allocator/cumem.py:146,
# pytorch/pytorch#147851). The top-k peak is bounded by the micro-batch size
# above instead, and at micro_bs=4 the allocations are ~2.6 GiB rather than
# 12-17 GiB, which is small enough that fragmentation is not the binding issue.

export ROLLOUT_KEEP_VLLM_AWAKE=1   # (1) one vLLM weight-sync per rollout, not per turn
# ROLLOUT_SKIP_DONE_PREPROC (2) / ROLLOUT_DECODE_ACTIVE_ONLY (E2) /
# ROLLOUT_COMPACT_RECORD (E3) default to on; all three speed up the alfworld tail.
# enable_prefix_caching (3) is passed as a Hydra arg in each block below.

# The one variable in this file, and it is not a knob: an absolute path to this
# script's own directory. The expectations files are read inside a Ray actor,
# after Hydra has chdir'd the driver into its output directory, so a path
# relative to the launcher's cwd is not reliably resolvable by the time it is
# opened. (python3 -m still requires the repo root as cwd; this only fixes the
# one path that outlives that assumption.)
OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# Data prep — same prompts/tasks as the on-policy OPD run. These literals are
# cross-checked by the expectations files (per_task_batch_size=15,
# val_per_task_size=126, total_training_steps=300, seed=1).
# Deterministic (--seed 1) and idempotent: rerunning overwrites train.parquet /
# test.parquet with byte-identical content, so it is safe to leave in place when
# rerunning generation. It is pure pandas — no GPU, Ray, env or model download —
# and finishes in seconds; its INFO logs are swallowed because importing
# verl.utils.hdfs_io sets a WARN level before logging.basicConfig runs, so an
# empty console here means success, not a no-op.
python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps 300 \
    --per_task_batch_size 15 \
    --env_train_per_task_size 15 \
    --val_per_task_size 126 \
    --seed 1

# ===================== Stage 1: teacher trajectory generation =====================
python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=alfworld \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/alfworld_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
    +gen.adjust_batch=False \
    +gen.shard_every_steps=10 \
    data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
    data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
    data.train_batch_size=15 \
    data.val_batch_size=15 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_model_len=4608 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    env.env_name=multitask \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=4 \
    env.rollout.n=8 \
    env.search.search_url=${SEARCH_URL:-http://100.86.45.31:8001/retrieve} \
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
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_alfworld \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False

# python3 -m verl.trainer.main_opd_offpolicy_gen \
#     +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_gen_config.yaml \
#     +gen.task=search \
#     +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/search_step300 \
#     +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
#     +gen.num_trajectories=36000 \
#     +gen.topk=20 \
#     +gen.adjust_batch=False \
#     +gen.shard_every_steps=10 \
#     data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
#     data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
#     data.train_batch_size=15 \
#     data.val_batch_size=15 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation='left' \
#     data.return_raw_chat=True \
#     data.task_balance.enable=True \
#     data.task_balance.per_task_batch_size=15 \
#     +data.task_balance.num_batches=300 \
#     data.task_balance.tasks=[alfworld,search,webshop] \
#     +data.task_overrides.alfworld.max_prompt_length=2048 \
#     +data.task_overrides.alfworld.truncation='error' \
#     +data.task_overrides.search.max_prompt_length=4096 \
#     +data.task_overrides.search.truncation='left' \
#     +data.task_overrides.webshop.max_prompt_length=4096 \
#     +data.task_overrides.webshop.truncation='error' \
#     +data.apply_chat_template_kwargs.enable_thinking=False \
#     +data.seed=1 \
#     actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.use_fused_kernels=False \
#     +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
#     actor_rollout_ref.actor.ppo_mini_batch_size=60 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.max_model_len=4608 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     +actor_rollout_ref.rollout.enable_prefix_caching=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.adv_estimator=grpo \
#     algorithm.use_kl_in_reward=False \
#     env.env_name=multitask \
#     env.seed=0 \
#     env.max_steps=50 \
#     env.history_length=4 \
#     env.rollout.n=8 \
#     env.search.search_url=${SEARCH_URL:-http://100.86.45.30:8001/retrieve} \
#     env.multitask.tasks=[alfworld,search,webshop] \
#     env.multitask.max_steps.alfworld=50 \
#     env.multitask.max_steps.search=4 \
#     env.multitask.max_steps.webshop=15 \
#     +env.multitask.history_length.alfworld=2 \
#     +env.multitask.history_length.search=4 \
#     +env.multitask.history_length.webshop=2 \
#     env.multitask.val_per_task_batch_size=126 \
#     env.resources_per_worker.num_cpus=0.1 \
#     trainer.n_gpus_per_node=3 \
#     trainer.ray_wait_register_center_timeout=600 \
#     trainer.nnodes=1 \
#     trainer.logger=['console'] \
#     trainer.project_name='verl_agent_opd_offpolicy_multitask' \
#     trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_search \
#     trainer.total_training_steps=300 \
#     trainer.total_epochs=300 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=-1 \
#     trainer.val_before_train=False

# python3 -m verl.trainer.main_opd_offpolicy_gen \
#     +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_gen_config.yaml \
#     +gen.task=webshop \
#     +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/webshop_step300 \
#     +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
#     +gen.num_trajectories=36000 \
#     +gen.topk=20 \
#     +gen.adjust_batch=False \
#     +gen.shard_every_steps=10 \
#     data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
#     data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
#     data.train_batch_size=15 \
#     data.val_batch_size=15 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation='left' \
#     data.return_raw_chat=True \
#     data.task_balance.enable=True \
#     data.task_balance.per_task_batch_size=15 \
#     +data.task_balance.num_batches=300 \
#     data.task_balance.tasks=[alfworld,search,webshop] \
#     +data.task_overrides.alfworld.max_prompt_length=2048 \
#     +data.task_overrides.alfworld.truncation='error' \
#     +data.task_overrides.search.max_prompt_length=4096 \
#     +data.task_overrides.search.truncation='left' \
#     +data.task_overrides.webshop.max_prompt_length=4096 \
#     +data.task_overrides.webshop.truncation='error' \
#     +data.apply_chat_template_kwargs.enable_thinking=False \
#     +data.seed=1 \
#     actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.use_fused_kernels=False \
#     +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
#     actor_rollout_ref.actor.ppo_mini_batch_size=60 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.max_model_len=4608 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     +actor_rollout_ref.rollout.enable_prefix_caching=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.adv_estimator=grpo \
#     algorithm.use_kl_in_reward=False \
#     env.env_name=multitask \
#     env.seed=0 \
#     env.max_steps=50 \
#     env.history_length=4 \
#     env.rollout.n=8 \
#     env.search.search_url=${SEARCH_URL:-http://100.86.45.30:8001/retrieve} \
#     env.multitask.tasks=[alfworld,search,webshop] \
#     env.multitask.max_steps.alfworld=50 \
#     env.multitask.max_steps.search=4 \
#     env.multitask.max_steps.webshop=15 \
#     +env.multitask.history_length.alfworld=2 \
#     +env.multitask.history_length.search=4 \
#     +env.multitask.history_length.webshop=2 \
#     env.multitask.val_per_task_batch_size=126 \
#     env.resources_per_worker.num_cpus=0.1 \
#     trainer.n_gpus_per_node=3 \
#     trainer.ray_wait_register_center_timeout=600 \
#     trainer.nnodes=1 \
#     trainer.logger=['console'] \
#     trainer.project_name='verl_agent_opd_offpolicy_multitask' \
#     trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_webshop \
#     trainer.total_training_steps=300 \
#     trainer.total_epochs=300 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=-1 \
#     trainer.val_before_train=False
