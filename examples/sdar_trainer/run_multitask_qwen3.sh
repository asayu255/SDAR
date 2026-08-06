set -x
ENGINE=${1:-vllm}
shift || true

num_cpus_per_env_worker=0.1

# SDAR hyperparameters
sdar_coef=0.01
gate_beta=5.0
skill_all=false

per_task_batch_size=15
train_data_size=45
val_per_task_size=126
# val_data_size == val_per_task_size: the test parquet is laid out
# alfworld -> webshop -> search (see prepare_sdar_multitask), so validation runs
# one single-task batch for each of alfworld and webshop and then
# ceil(n_search / val_per_task_size) batches of search — every batch holds a
# single task, and the only short one is search's own tail, which the search env
# pads with masked-out dummies. val_per_task_size applies to alfworld/webshop
# only; search is validated over its whole test parquet, as the single-task
# baseline does.
val_data_size=$val_per_task_size
group_size=8
total_training_steps=300
# Shared seed for the search data subset (prepare) and the dataloader shuffle
# (data.seed -> TaskBalancedSampler). Single-task runs use the dataloader default
# (data.seed=1); set it explicitly here so the search周回 is reproducible and the
# seed handling is consistent rather than relying on an implicit default.
seed=1
model_path="Qwen/Qwen3-1.7B"

# --- Throughput knobs (opt-in; defaults preserve the fair-comparison alignment) -
# The 3-task speedup mechanisms live in code (async rollout / TASK_BALANCE_INTERLEAVE
# / prefix caching) and are active independently of this script. These knobs expose
# the throughput-mode toggles that the old no_preprocess run used. Defaults below
# reproduce this script's accuracy-aligned behavior exactly; override via env to
# trade bit-identity for throughput. Accuracy notes:
#   PARAM_OFFLOAD=True          -> bit-identical (FSDP placement only)
#   OPTIMIZER_OFFLOAD=True      -> same algorithm, not bit-identical (Adam CPU<->GPU)
#   *_MICRO_PER_GPU larger      -> same algorithm, not bit-identical (grad-accum grouping)
#   USE_FUSED_KERNELS=True      -> not bit-identical (fused kernel path)
#   ENABLE_CHUNKED_PREFILL=True -> not bit-identical (prefill scheduling)
# max_model_len defaults to prompt(4096)+response(512)=4608: it bounds the vLLM KV
# cache to exactly what is needed and truncates no valid sequence, so it is a pure
# speedup and is on by default. prefix caching is already the code default (output
# -safe deterministic reuse) and is kept on.
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
ppo_micro_per_gpu=${PPO_MICRO_PER_GPU:-5}
log_prob_micro_per_gpu=${LOG_PROB_MICRO_PER_GPU:-16}
# The TEACHER/ref forward, not the rollout's, is what binds GPU memory here: its
# lm_head materializes ref_log_prob_micro x (skill-augmented prompt length) x
# 151936 in bf16, and 16 rows of that is the 13.10 GiB that OOMed step 3 when the
# ref was made resident. 8 halves it. Kept SEPARATE from log_prob_micro_per_gpu on
# purpose: adjust_batch's divisor is lcm(ref*W, rollout*W, ppo_micro*W), which at
# W=3 is lcm(48,48,15)=240 today and lcm(24,48,15)=240 with only the ref lowered --
# so this moves no padding and no data. Lowering BOTH (the header's earlier advice)
# would take it to 120 and change what the step trains on.
ref_log_prob_micro_per_gpu=${REF_LOG_PROB_MICRO_PER_GPU:-8}
use_fused_kernels=${USE_FUSED_KERNELS:-False}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}
enable_prefix_caching=${ENABLE_PREFIX_CACHING:-True}
max_model_len=${MAX_MODEL_LEN:-4608}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6}
# CUDA-graph decode (mechanism B). The passthrough already exists in
# vllm_rollout_spmd (enforce_eager=False is set below); capture sizes only take
# effect on a vLLM V1 engine — under the V0 engine CompilationConfig is ignored,
# which is why this stays opt-in. Example:
#   VLLM_USE_V1=1 CUDAGRAPH_CAPTURE_SIZES='[8,16,32,64,128,256,384]' bash ...
# Sampling-distribution-preserving (same class as prefix caching), not
# bit-identical. Leave both unset to reproduce the current engine behavior.
cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-}
if [ -n "${VLLM_USE_V1:-}" ]; then
    export VLLM_USE_V1
fi
extra_rollout_args=()
if [ -n "$cudagraph_capture_sizes" ]; then
    extra_rollout_args+=("+actor_rollout_ref.rollout.cudagraph_capture_sizes=$cudagraph_capture_sizes")
fi
# -----------------------------------------------------------------------------

# --- Rollout/trainer overlap mechanisms (env-var driven; live in code) ---------
# ROLLOUT_KEEP_VLLM_AWAKE / TASK_BALANCE_INTERLEAVE / ENV_RESET_PREFETCH /
# ROLLOUT_PREFETCH_LOGPROB are exported below rather than left to the operator:
# forgetting one is silent, since the config-side mechanisms keep working and only
# the rollout ones go quiet. Pass 0 to disable any of them.
# ROLLOUT_SKIP_DONE_PREPROC / DECODE_ACTIVE_ONLY / COMPACT_RECORD default to on:
#   ROLLOUT_PREFETCH_LOGPROB=1        A: prefetch old_log_prob for recorded rows
#                                        while env.step runs (rows queue per turn
#                                        as they are recorded, not at trajectory
#                                        end -- see rollout_loop.py; chunk via
#                                        ROLLOUT_PREFETCH_LOGPROB_CHUNK. §11
#                                        measured 23% reuse under the old
#                                        trajectory-end queueing)
#   ENV_RESET_PREFETCH=1              C: overlap next rollout's envs.reset with
#                                        the GPU training phases
#   ROLLOUT_DECODE_ACTIVE_ONLY=1      E: decode generated rows only (default on)
#   ROLLOUT_COMPACT_RECORD=1          E: skip recording finished rows (default on)
# -----------------------------------------------------------------------------
#
# NO speculative decoding here, and it is not a tuning choice -- it does not run
# on this stack. Setting engine_kwargs.vllm.speculative_config swaps the vLLM V0
# worker for spec_decode.SpecDecodeWorker (wrapping NGramWorker), which does not
# implement sleep(); vllm_rollout_spmd builds the engine with
# enable_sleep_mode=True and calls sleep(level=1) immediately (:210), so the run
# dies in init_workers with "Method 'sleep' is not implemented" before step 1 --
# measured on this script. The whole wake/sleep cycle ROLLOUT_KEEP_VLLM_AWAKE
# and free_cache_engine=False depend on needs that method, so this is
# structural, not a missing argument. The idea itself still fits §11's gen at
# 72.4% SM with a half-empty tail (rejection sampling preserves the sampling
# distribution exactly, and old_log_prob is recomputed by the actor), but it
# needs the V1 engine, where spec decode lives in v1/spec_decode and sleep
# exists. VLLM_USE_V1=1 changes the engine for every phase -- and §4 already
# records cudagraph_capture_sizes as V1-only for the same reason -- so that is
# its own experiment across all arms, not a knob to flip here.
#
# Reference policy placement (ref.fsdp_config below) is RESIDENT + ZeRO-2, on the
# second attempt. The first one -- param_offload=False plus
# sharding_strategy=shard_grad_op, resident shards ~1.1 GB/GPU plus a gathered
# copy ~3.2 GB/GPU that ZeRO-2 does not reshard away -- survived steps 1-2 and
# then OOMed in step 3's teacher forward: 13.10 GiB wanted for the lm_head logits
# against 8.17 GiB free on a 47.53 GiB card, short by 4.93 GiB, which is what the
# two knobs had taken. It is back together with the prerequisite the revert
# identified: ref_log_prob_micro_per_gpu=8 halves that 13.10 GiB request to ~6.6,
# which covers the 4.93 GiB the placement knobs cost with room over. Retrying the
# placement alone is what does not work; retrying it with the micro batch is what
# the revert commit said to do.
#
# Two things that measurement established, both worth keeping in mind before
# retrying:
#   - THE GPUS ARE NOT THE SAME SIZE. perf/max_memory_allocated_gb reported
#     50.7 GiB while the card that OOMed has 47.53 GiB total, and
#     max_memory_allocated can never exceed its device -- so rank 0 (the only
#     rank that reports) sits on a BIGGER GPU than the one that runs out. The
#     peak-memory metric has been describing the roomiest card, not the binding
#     one. §11's "66.7 GB reserved of 80" is that same rank, not this limit.
#   - THE BINDING ALLOCATION IS THE TEACHER LOGITS, not the ref. 13.10 GiB is
#     16 rows x ~2.9k tokens x 151936 vocab in bf16, i.e.
#     log_prob_micro_batch_size_per_gpu x (skill-augmented prompt length) x
#     vocab, materialized whole by lm_head. Teacher prompts are the long ones
#     (§11: mean 1719 tokens, webshop 2646, skill overhead ~1107). Even fully
#     reverted the headroom is ~12.8 GiB against a 13.10 GiB request, so lever
#     #2 needs the ref micro batch halved to have room -- retry them together,
#     not lever #2 alone. That is what ref_log_prob_micro_per_gpu=8 above is; it
#     is deliberately a SEPARATE knob from log_prob_micro_per_gpu, because
#     lowering both would change adjust_batch's lcm from 240 to 120 and with it
#     the padding rows the step trains on.

# --- Fair-comparison alignment with the single-task baselines -----------------
# This is a joint multitask run (one shared model/optimizer over alfworld+search+
# webshop); per-task metrics will NOT reproduce single-task values because every
# step mixes the three tasks' gradients (task transfer is inherent to multitask
# learning). The goal here is fair COMPARISON conditions, which already hold:
#   - per-task hyperparameters (max_prompt_length/truncation/max_steps/
#     history_length/kl_loss_coef/invalid_action_penalty/val_kwargs) match the
#     single-task scripts via the *_by_task / task_overrides / multitask.* keys.
#   - env-driven tasks (alfworld/webshop) sample identical game instances: the env
#     build seeds match the single-task runs (train seed=0, val seed=1000,
#     env_num=15/126, group_n=8/1); the parquet rows are placeholders (the prompt
#     is built from env observations, not the parquet).
#   - each task contributes 15 prompts x group 8 per step over 300 steps. The
#     alfworld/webshop train parquets are 15 placeholder rows that the dataloader
#     cycles (data.task_balance.num_batches) just like the single-task runs wrap
#     their 15-row parquet; the real, fresh game instances come from the env.
#   - search: train = uniform random 4500 (no repeat) from the full pool, val =
#     leading 126 rows of test.parquet (same fixed population the single-task
#     val_dataloader(shuffle=False) uses). See prepare_sdar_multitask.py.
# -----------------------------------------------------------------------------
experiment_name="sdar_multitask_qwen3_1.7b_instruct_coef${sdar_coef}_beta${gate_beta}_skillall${skill_all}"

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
# On by default, for the same reason the FSDP knobs are yaml defaults: a 300-step
# run gets restarted, and a mechanism that has to be exported by hand is one that
# will eventually be missing from a restart. All four are accuracy-preserving (see
# the mechanism block above); each still honours an explicit 0 from the caller.
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export ENV_RESET_PREFETCH=${ENV_RESET_PREFETCH:-1}
export TASK_BALANCE_INTERLEAVE=${TASK_BALANCE_INTERLEAVE:-1}
export ROLLOUT_PREFETCH_LOGPROB=${ROLLOUT_PREFETCH_LOGPROB:-1}
export HIGHLIGHT_CONFIGS'<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps "$total_training_steps" \
    --per_task_batch_size "$per_task_batch_size" \
    --env_train_per_task_size "$per_task_batch_size" \
    --val_per_task_size "$val_per_task_size" \
    --seed "$seed"

# INTENT LOCK: expected_multitask_config.yaml pins the knobs that define what
# this experiment IS (loss coefficients, seeds, batch sizes, eval protocol) plus
# the two throughput knobs that sit on the gradient path. The trainer validates
# the composed config against it after main_sdar's injection and refuses to start
# on any mismatch. To change such a knob, edit the argument below AND the
# expectations file in the same commit.
python3 -m verl.trainer.main_sdar \
    +trainer.expected_config=examples/sdar_trainer/expected_multitask_config.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
    data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    +data.seed=$seed \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.task_balance.enable=True \
    data.task_balance.per_task_batch_size=$per_task_batch_size \
    +data.task_balance.num_batches=$total_training_steps \
    data.task_balance.tasks=[alfworld,search,webshop] \
    +data.task_overrides.alfworld.max_prompt_length=2048 \
    +data.task_overrides.alfworld.truncation='error' \
    +data.task_overrides.search.max_prompt_length=4096 \
    +data.task_overrides.search.truncation='left' \
    +data.task_overrides.webshop.max_prompt_length=4096 \
    +data.task_overrides.webshop.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.ppo_mini_batch_size=60 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill \
    +actor_rollout_ref.rollout.enable_prefix_caching=$enable_prefix_caching \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.sharding_strategy=shard_grad_op \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.invalid_action_penalty_coef_by_task='{alfworld:0.1,search:0.01,webshop:0.1}' \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/alfworld \
    +algorithm.sdar.skills_dirs.alfworld=skills/alfworld \
    +algorithm.sdar.skills_dirs.search=skills/search \
    +algorithm.sdar.skills_dirs.webshop=skills/webshop \
    +algorithm.sdar.kl_loss_coef_by_task='{alfworld:0.01,search:0.001,webshop:0.01}' \
    +algorithm.sdar.skill_all=$skill_all \
    env.env_name=multitask \
    env.seed=1 \
    env.max_steps=50 \
    env.history_length=4 \
    env.rollout.n=$group_size \
    env.search.search_url='http://0.0.0.0:8000/retrieve' \
    env.multitask.tasks=[alfworld,search,webshop] \
    env.multitask.max_steps.alfworld=50 \
    env.multitask.max_steps.search=4 \
    env.multitask.max_steps.webshop=15 \
    +env.multitask.history_length.alfworld=2 \
    +env.multitask.history_length.search=4 \
    +env.multitask.history_length.webshop=2 \
    env.multitask.val_per_task_batch_size=$val_per_task_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_sdar_multitask' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=3 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "${extra_rollout_args[@]}" $@
