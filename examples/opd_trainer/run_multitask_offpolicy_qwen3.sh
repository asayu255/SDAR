set -x
#
# SHARED CHECKPOINT DIRECTORY. trainer.default_local_dir below is also used by the
# other off-policy arms (run_multitask_offpolicy_qwen3.sh,
# run_multitask_offpolicy_qwen3_nogen.sh, run_multitask_offpolicy_studenttopk_qwen3.sh).
# Checkpoints land in default_local_dir/global_step_N and trainer.experiment_name
# is NOT in that path, so two things follow, both silent:
#   - trainer.resume_mode defaults to auto and reads
#     default_local_dir/latest_checkpointed_iteration.txt, so this run resumes
#     from whichever arm checkpointed there last. trainer.resume_mode=disable
#     starts from the base model instead.
#   - this run's saves overwrite the other arms' checkpoints at the same step.
# Check the directory before starting a different arm than the one that filled it.

# Off-policy multitask distillation (offline KD), Qwen3-1.7B.
# Stage 1: each task's frozen single-task RL teacher generates a fixed
# trajectory + top-20 dataset in its own environment. Stage 2: the student is
# distilled off-policy on that fixed teacher data with the same top-20 dense
# teacher-KL as on-policy OPD.
#
# EVERY parameter lives as a literal argument of the python3 commands below —
# there is deliberately NO variable block, NO ${VAR:-default} fallback, NO
# shared-args array and NO per-task loop, so there is exactly one place to
# read or edit a setting. The only env-var passthroughs are infra endpoints
# (WANDB_API_KEY, SEARCH_URL — the retriever host moves between machines),
# which are not scientific knobs. One-off overrides for Stage 2 can be appended on the
# command line (trailing "$@") and still pass the expectations check.
#
# INTENT LOCK (examples/opd_trainer/):
#   expected_multitask_offpolicy_gen_config.yaml — Stage-1 dataset knobs,
#     validated by main_opd_offpolicy_gen BEFORE its single-task restriction.
#   expected_multitask_offpolicy_config.yaml — Stage-2 training knobs,
#     validated by main_opd_offpolicy AFTER its loss injection.
# To change a scientific knob: edit the argument below AND the matching
# expectations file in the same commit; a script-only edit refuses to start.
#
# Throughput mechanisms (process env vars, accuracy-preserving; they live in code,
# not in the expectations files — see docs/optimization_phase2.md). The first two
# are exported below so they are on without being remembered; set either to 0 to
# disable:
#   ROLLOUT_KEEP_VLLM_AWAKE=1   — one vLLM weight-sync per rollout, not per turn
#   OFFPOLICY_BATCH_PREFETCH=1  — builds step k+1's batch on a background thread
#     while step k is inside update_actor (a blocking ray.get, so it holds no
#     GIL). Bit-identical to the sequential path; see _prepared_batch_iter for
#     the two RNG invariants that make that true.
#   (ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY /
#    ROLLOUT_COMPACT_RECORD default to on; they speed up the Stage-1 teacher
#    rollouts and the Stage-2 validation rollouts)
#   NOTE: leave ROLLOUT_PREFETCH_LOGPROB and ENV_RESET_PREFETCH off here —
#   neither stage has an old_log_prob phase or a per-step train rollout.
#

# ref.fsdp_config.* IS INERT ON THIS ARM, in both stages, and the value below is
# only there because the config schema carries it. Stage 1 loads the teacher as
# the ACTOR_ROLLOUT model (main_opd_offpolicy_gen sets
# actor_rollout_ref.model.path = gen.teacher_path) and registers only
# Role.ActorRollout / Role.Critic; Stage 2 says so in main_opd_offpolicy directly
# -- "off-policy OPD registers neither Role.RefPolicy nor teacher workers; the
# teacher top-k signal is precomputed in the Stage-1 dataset". So the ref
# placement knobs that pay off on the on-policy arms (param_offload=False,
# sharding_strategy=shard_grad_op, a smaller ref micro batch) have no forward to
# act on here. Do not port them expecting a speedup.
#
# ONE RETRIEVER, SHARED. env.search.search_url can point at the same server as
# another concurrent run. What makes that safe is not the URL but the retry policy
# beside it: env.search.max_retries=null waits for a timeout / refused connection /
# 5xx to clear instead of giving up. Giving up is not a no-op -- the client hands the
# error text back as the retrieval result, so it lands in the <information> block the
# model is trained on with nothing in the metrics to say so, and under a shared
# retriever an exhausted budget is exactly what a load spike looks like. 4xx and
# malformed JSON still fail immediately: waiting cannot turn a bad URL into a
# document. Both knobs are pinned in the expectations files, because they decide
# what enters the data.
#
# env.search.timeout=600 is generous but finite on purpose; see the expectations
# file for why null is worse here. A request that is still retrying says so in the
# log every ~60s, so an intentional wait is never mistaken for a hang.
#   TASK_BALANCE_INTERLEAVE does nothing for Stage 2: it reorders the *train*
#   sampler, which that loop never iterates (it draws from the fixed pool), and
#   the validation dataloader takes no sampler at all.
#   GPU_PROFILER must be OFF for a real run: inert when unset, but when set it
#   starts a per-step NVML sampler and (with GPU_PROFILER_SYNC_PHASES=1) a device
#   synchronize at every actor stage boundary.
#
# On the gradient path, and therefore pinned in the Stage-2 expectations file
# rather than left as env vars: fsdp_config.sharding_strategy=shard_grad_op
# (ZeRO-2) and actor.no_sync_grad_accum=True. fsdp_config.forward_prefetch=True is
# passed beside them but stays out of the expectations file — it only reorders the
# issue of all-gathers that happen either way.
#
# STAGE 1 WRITES NO PADDING. adjust_batch(mode="copy") pads each generation step
# up to a DP/micro-divisible size by duplicating random rows, and those copies used
# to be saved — so Stage 2 trained on those turns twice and split the step into
# more mini-batches than the real data warrants. Nothing in Stage 1 needs the
# divisibility (the top-k worker call pads and unpads itself), so it is off unless
# +gen.adjust_batch=True is passed; pass +gen.expect_pad_divisor=<n> with it to
# assert the divisor matches the pool being extended.
#
# STAGE 2 READS A CACHED VIEW. The loader drops any padding it does find, plus the
# columns no Stage-2 loss reads, on every start. Do it once instead:
#
#   python3 scripts/cache_teacher_pool.py \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache --arm kd
#
# then append to this script's command line (double plus — the key below already
# uses a single '+', and Hydra refuses a second append of an existing key):
#
#   ++algorithm.opd.teacher_data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache
#
# The cache is exactly what the loader would have built, file for file and row for
# row, so the run is unchanged. It is ARM-SPECIFIC — an --arm sft cache has no
# teacher top-k and this run must not read one.

# The one variable in this file, and it is not a knob: an absolute path to this
# script's own directory. The expectations files are read inside a Ray actor,
# after Hydra has chdir'd the driver into its output directory, so a path
# relative to the launcher's cwd is not reliably resolvable by the time it is
# opened. (python3 -m still requires the repo root as cwd; this only fixes the
# one path that outlives that assumption.)
OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On by default, for the same reason the three FSDP knobs are literals in the
# Stage-2 command: a 300-step run gets restarted, and a mechanism that has to be
# exported by hand is one that will eventually be missing from a restart, and the
# restarted run would then differ from the one before it for no recorded reason.
# Both are accuracy-preserving (batch prefetch is bit-identical to the sequential
# path; see _prepared_batch_iter), so this changes throughput and nothing else.
# ROLLOUT_KEEP_VLLM_AWAKE=0 / OFFPOLICY_BATCH_PREFETCH=0 still turns either off.
# OFFPOLICY_BATCH_PREFETCH applies to the Stage-2 loop only; Stage 1 ignores it.
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export OFFPOLICY_BATCH_PREFETCH=${OFFPOLICY_BATCH_PREFETCH:-1}

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# Data prep — same prompts/tasks as the on-policy OPD run. These literals are
# cross-checked by the expectations files (per_task_batch_size=15,
# val_per_task_size=126, total_training_steps=300, seed=1).
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
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_alfworld \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False

python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=search \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/search_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
    +gen.adjust_batch=False \
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
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_search \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False

python3 -m verl.trainer.main_opd_offpolicy_gen \
    +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_gen_config.yaml \
    +gen.task=webshop \
    +gen.teacher_path=/opt/home/ohara/checkpoints/teachers/webshop_step300 \
    +gen.out_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +gen.num_trajectories=36000 \
    +gen.topk=20 \
    +gen.adjust_batch=False \
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
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20_gen_webshop \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False

# ===================== Stage 2: off-policy distillation =====================
python3 -m verl.trainer.main_opd_offpolicy \
    +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_config.yaml \
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
    +algorithm.opd.teacher_data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +algorithm.opd.kl_loss_coef=1.0 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
    +algorithm.opd.sft_loss_coef=1.0 \
    +algorithm.opd.normalize_loss_by_task=True \
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
    trainer.project_name='verl_agent_opd_offpolicy_multitask' \
    trainer.experiment_name=opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20 \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_opd_offpolicy_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
