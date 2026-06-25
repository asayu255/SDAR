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
# val_data_size == val_per_task_size: the task-sorted test parquet then yields one
# single-task batch per task, so each task is validated in its own rollout pass
# (no mixed-task validation batch).
val_data_size=$val_per_task_size
group_size=8
total_training_steps=300
# Shared seed for the search data subset (prepare) and the dataloader shuffle
# (data.seed -> TaskBalancedSampler). Single-task runs use the dataloader default
# (data.seed=1); set it explicitly here so the search周回 is reproducible and the
# seed handling is consistent rather than relying on an implicit default.
seed=1
model_path="Qwen/Qwen3-1.7B"

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

python3 -m verl.trainer.main_sdar \
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
    data.task_balance.num_batches=$total_training_steps \
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
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.ppo_mini_batch_size=60 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.project_name='verl_agent_sdar_multitask' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=3 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=300 \
    trainer.val_before_train=False $@
