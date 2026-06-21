set -x
ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then
    shift
fi

export WANDB_API_KEY=${WANDB_API_KEY:-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}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export ROLLOUT_SKIP_DONE_PREPROC=${ROLLOUT_SKIP_DONE_PREPROC:-1}
export ROLLOUT_TURN_TIMING=${ROLLOUT_TURN_TIMING:-0}
export GPU_PROFILER=${GPU_PROFILER:-0}

# SDAR hyperparameters
sdar_coef=0.01
gate_beta=5.0
skill_all=false

experiment_name="${EXPERIMENT_NAME:-sdar_qwen3_1.7b_coef${sdar_coef}_beta${gate_beta}_skillall${skill_all}}"

train_data_size=15
val_data_size=126
group_size=8

# Throughput knobs shared with the multitask Qwen3 runner. Defaults preserve the
# existing single-task memory/offload behavior while exposing the same A/B levers.
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
ref_param_offload=${REF_PARAM_OFFLOAD:-True}
ppo_micro_per_gpu=${PPO_MICRO_PER_GPU:-10}
log_prob_micro_per_gpu=${LOG_PROB_MICRO_PER_GPU:-16}
seed=${SEED:-0}
test_freq=${TEST_FREQ:-150}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
save_freq=${SAVE_FREQ:-25}
search_url=${SEARCH_URL:-http://0.0.0.0:8000/retrieve}
max_model_len=${MAX_MODEL_LEN:-4608}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.5}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}
enable_prefix_caching=${ENABLE_PREFIX_CACHING:-True}
use_fused_kernels=${USE_FUSED_KERNELS:-True}
model_path=${MODEL_PATH:-Qwen/Qwen3-1.7B}

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$ref_param_offload \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/search \
    +algorithm.sdar.skill_all=$skill_all \
    env.env_name=search \
    env.seed=$seed \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url="$search_url" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=3 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=False $@
