set -x

# Stage 1 ONLY — teacher trajectory generation for the three tasks.
#
# Each task's frozen single-task RL teacher rolls out in its own environment and
# scores its own top-20 over what it generated, writing
#   $HOME/data/verl-agent/sdar_multitask/teacher_traj/<task>.pt
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
# The three tasks run CONCURRENTLY, one GPU each (alfworld->GPU0, search->GPU1,
# webshop->GPU2), instead of sequentially on 3 GPUs each. They are fully
# independent — separate processes, separate frozen teachers, separate envs,
# separate output .pt — so the only thing the old sequential layout bought was
# 3-way data parallelism within a task, and that is the cheap axis here:
# tensor_model_parallel_size is 1, so each GPU already holds a complete engine,
# and 1.7B decode is weight-bandwidth bound, so going from 40 to 120
# trajectories per GPU costs far less than a third of the wall time. See
# docs/optimization_gen_only.md (G1).
#
# What to watch when tuning: per-GPU KV-cache demand triples. If the vLLM logs
# show preemption / "cache full", that is where the win is going. The companion
# knobs are gpu_memory_utilization and log_prob_micro_batch_size_per_gpu — see
# the note in docs/optimization_gen_only.md (G3) before touching either; they
# are deliberately left at their sequential-run values here so this change is
# the parallelization alone.
#
# NOTE ON THE PRODUCED DATASET: world size 3 -> 1 changes adjust_batch's
# divisor from lcm(16*3, 5*3)=240 to lcm(16, 5)=80, so a different (smaller)
# number of duplicated turn-rows is padded into each <task>.pt. The trajectory
# count, prompts, sampling temperature and episode caps are unchanged.
#
# Each task saves its own .pt as soon as that task reaches its target, and a
# failure in one task does not touch the other two — rerun this script with the
# finished blocks commented out. Note that a rerun of a task OVERWRITES its
# existing <task>.pt. Per-task stdout/stderr goes to
#   $HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_<task>.log
# because three concurrent runs would otherwise interleave on the console; the
# script waits for all three and exits non-zero if any of them failed.
#
# Ray isolation (three ray.init() on one host):
#   RAY_ADDRESS=local           force a fresh local instance instead of possibly
#                               attaching to a sibling's cluster (which would
#                               then see 0 free GPUs and hang until
#                               ray_wait_register_center_timeout)
#   RAY_TMPDIR=...              separate session/socket trees per task
#   +ray_init.include_dashboard=False   no race for the default dashboard port
#   +ray_init.object_store_memory=...   pin the split explicitly; the default
#                               ("30% of AVAILABLE memory") is racy when three
#                               instances start at once and can overflow /dev/shm
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

export ROLLOUT_KEEP_VLLM_AWAKE=1   # (1) one vLLM weight-sync per rollout, not per turn
export ROLLOUT_PREPROC_WORKERS=8   # (E1) parallel prompt tokenization, bit-identical
export SEARCH_QUERY_CACHE=1        # (D) reuse retriever hits; needs a deterministic
                                   #     (fixed-index) retriever. Only the search
                                   #     block below issues queries at all.
# ROLLOUT_SKIP_DONE_PREPROC (2) / ROLLOUT_DECODE_ACTIVE_ONLY (E2) /
# ROLLOUT_COMPACT_RECORD (E3) default to on; all three speed up the alfworld tail.
# enable_prefix_caching (3) is passed as a Hydra arg in each block below.

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
# The three blocks below launch concurrently, one GPU each, and are waited on at
# the bottom of the script.
mkdir -p $HOME/data/verl-agent/sdar_multitask/teacher_traj

CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_gen_alfworld \
python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=examples/opd_trainer/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=alfworld \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/alfworld_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
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
    trainer.n_gpus_per_node=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_alfworld \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    +ray_init.object_store_memory=8000000000 \
    > $HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_alfworld.log 2>&1 &
pid_alfworld=$!

CUDA_VISIBLE_DEVICES=1 RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_gen_search \
python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=examples/opd_trainer/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=search \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/search_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
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
    trainer.n_gpus_per_node=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_search \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    +ray_init.object_store_memory=8000000000 \
    > $HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_search.log 2>&1 &
pid_search=$!

CUDA_VISIBLE_DEVICES=2 RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_gen_webshop \
python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=examples/opd_trainer/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=webshop \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/webshop_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
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
    trainer.n_gpus_per_node=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_webshop \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    +ray_init.object_store_memory=8000000000 \
    > $HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_webshop.log 2>&1 &
pid_webshop=$!

# ===================== wait for all three tasks =====================
# Collect each exit status separately so a single failure is attributable and
# the other two tasks still run to completion (their .pt is already written).
wait $pid_alfworld
status_alfworld=$?
wait $pid_search
status_search=$?
wait $pid_webshop
status_webshop=$?

set +x
echo "================ Stage 1 summary ================"
echo "alfworld  exit=$status_alfworld  log=$HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_alfworld.log"
echo "search    exit=$status_search  log=$HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_search.log"
echo "webshop   exit=$status_webshop  log=$HOME/data/verl-agent/sdar_multitask/teacher_traj/gen_webshop.log"
echo "================================================"

if [ "$status_alfworld" -ne 0 ] || [ "$status_search" -ne 0 ] || [ "$status_webshop" -ne 0 ]; then
    echo "Stage 1 FAILED for at least one task. Fix it, then rerun this script with"
    echo "the blocks of the tasks that already produced their <task>.pt commented out."
    exit 1
fi

echo "Stage 1 complete: alfworld.pt / search.pt / webshop.pt are in"
echo "  $HOME/data/verl-agent/sdar_multitask/teacher_traj"
echo "Stage 2 next: bash examples/opd_trainer/run_multitask_offpolicy_qwen3_nogen.sh"
