set -x
ENGINE=${1:-vllm}
shift || true

num_cpus_per_env_worker=0.1

# OPD + GRPO hyperparameters
# Per-task single-task RL teacher checkpoints. Each sample is distilled from the
# teacher matching its task_name. Override via env vars.
teacher_alfworld=${TEACHER_ALFWORLD:-/opt/home/ohara/checkpoints/teachers/alfworld_step300}
teacher_search=${TEACHER_SEARCH:-/opt/home/ohara/checkpoints/teachers/search_step300}
teacher_webshop=${TEACHER_WEBSHOP:-/opt/home/ohara/checkpoints/teachers/webshop_step300}
opd_kl_loss_coef=${OPD_KL_LOSS_COEF:-1.0}
# KL form: single-token estimator (low_var_kl / kl / mse / abs) or dense
# top-k+tail reverse KL (topk_kl). For top-20 set OPD_KL_LOSS_TYPE=topk_kl.
opd_kl_loss_type=${OPD_KL_LOSS_TYPE:-low_var_kl}
opd_topk=${OPD_TOPK:-20}  # support size for topk_kl
# GRPO policy-gradient weight; combined loss is pg_loss*pg_loss_coef +
# teacher_kl_loss*opd_kl_loss_coef. Set 0 to recover pure OPD.
pg_loss_coef=${PG_LOSS_COEF:-1.0}

per_task_batch_size=15
train_data_size=45
val_per_task_size=126
# val_data_size == val_per_task_size: the task-sorted test parquet then yields one
# single-task batch per task, so each task is validated in its own rollout pass.
val_data_size=$val_per_task_size
group_size=8
total_training_steps=300
seed=1
model_path="Qwen/Qwen3-1.7B"

# --- Throughput knobs (opt-in; same defaults as the SDAR multitask script) ------
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
ppo_micro_per_gpu=${PPO_MICRO_PER_GPU:-10}
log_prob_micro_per_gpu=${LOG_PROB_MICRO_PER_GPU:-16}
use_fused_kernels=${USE_FUSED_KERNELS:-False}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}
enable_prefix_caching=${ENABLE_PREFIX_CACHING:-True}
max_model_len=${MAX_MODEL_LEN:-4608}
# -----------------------------------------------------------------------------

experiment_name="opd_grpo_multitask_qwen3_1.7b_pg${pg_loss_coef}_coef${opd_kl_loss_coef}_${opd_kl_loss_type}"

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps "$total_training_steps" \
    --per_task_batch_size "$per_task_batch_size" \
    --env_train_per_task_size "$per_task_batch_size" \
    --val_per_task_size "$val_per_task_size" \
    --seed "$seed"

# NOTE on the combined loss (OPD + GRPO):
#   pg_loss_coef!=0 keeps the GRPO policy gradient active (env-reward advantages),
#   while use_teacher_kl_loss=True (force-injected in main_opd_grpo.py) adds the
#   per-task teacher KL. use_kl_loss=False / entropy_coeff=0 keep the rest off.
#   adv_estimator=grpo is required so advantages are computed.
# NOTE on teachers: they are created as role="ref" worker groups, so they reuse the
#   actor_rollout_ref.ref.* settings below (log-prob micro batch, FSDP CPUOffload).
python3 -m verl.trainer.main_opd_grpo \
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.pg_loss_coef=$pg_loss_coef \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.invalid_action_penalty_coef_by_task='{alfworld:0.1,search:0.01,webshop:0.1}' \
    algorithm.use_kl_in_reward=False \
    +algorithm.opd.teacher_paths.alfworld=$teacher_alfworld \
    +algorithm.opd.teacher_paths.search=$teacher_search \
    +algorithm.opd.teacher_paths.webshop=$teacher_webshop \
    +algorithm.opd.kl_loss_coef=$opd_kl_loss_coef \
    +algorithm.opd.kl_loss_type=$opd_kl_loss_type \
    +algorithm.opd.topk=$opd_topk \
    env.env_name=multitask \
    env.seed=0 \
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
    trainer.project_name='verl_agent_opd_grpo_multitask' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=3 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_opd_grpo_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=300 \
    trainer.val_before_train=False $@