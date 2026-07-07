set -x
ENGINE=${1:-vllm}
shift || true

num_cpus_per_env_worker=0.1

# Off-policy OPD (offline KD) hyperparameters.
# Per-task single-task RL teacher checkpoints. In Stage 1 each task's teacher
# GENERATES its own trajectories + top-k; in Stage 2 the student is distilled
# off-policy on that fixed teacher data. Same teachers / paths as on-policy OPD.
teacher_alfworld=${TEACHER_ALFWORLD:-/opt/home/ohara/checkpoints/teachers/alfworld_step300}
teacher_search=${TEACHER_SEARCH:-/opt/home/ohara/checkpoints/teachers/search_step300}
teacher_webshop=${TEACHER_WEBSHOP:-/opt/home/ohara/checkpoints/teachers/webshop_step300}
opd_kl_loss_coef=${OPD_KL_LOSS_COEF:-1.0}
# Same top-20 dense KL as on-policy OPD (topk_kl). Off-policy is teacher-generated
# data; the distillation loss itself is identical to OPD.
opd_kl_loss_type=${OPD_KL_LOSS_TYPE:-topk_kl}
opd_topk=${OPD_TOPK:-20}  # support size for topk_kl

per_task_batch_size=15
train_data_size=45
val_per_task_size=126
val_data_size=$val_per_task_size
group_size=8
total_training_steps=300
seed=1
model_path="Qwen/Qwen3-1.7B"

# Stage-1 teacher dataset: how many whole trajectories (episodes, not turn-rows)
# to collect per task. Each training step draws per_task_batch_size*group_size
# (15*8=120) trajectories/task from this fixed pool, reshuffled/recycled across
# the $total_training_steps steps. Keep this >= 120 to avoid intra-step repeats.
gen_traj_per_task=${GEN_TRAJ_PER_TASK:-2400}
teacher_traj_dir=${TEACHER_TRAJ_DIR:-$HOME/data/verl-agent/sdar_multitask/teacher_traj}

# --- Throughput knobs (opt-in; same defaults as the OPD/SDAR multitask scripts) ---
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
ppo_micro_per_gpu=${PPO_MICRO_PER_GPU:-5}
log_prob_micro_per_gpu=${LOG_PROB_MICRO_PER_GPU:-16}
use_fused_kernels=${USE_FUSED_KERNELS:-False}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}
enable_prefix_caching=${ENABLE_PREFIX_CACHING:-True}
max_model_len=${MAX_MODEL_LEN:-4608}
search_url=${SEARCH_URL:-http://100.86.45.31:8001/retrieve}
# -----------------------------------------------------------------------------

experiment_name="opd_offpolicy_multitask_qwen3_1.7b_coef${opd_kl_loss_coef}_${opd_kl_loss_type}${opd_topk}"

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# ---- Same data preprocessing as on-policy OPD (identical prompts / tasks) ----
python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps "$total_training_steps" \
    --per_task_batch_size "$per_task_batch_size" \
    --env_train_per_task_size "$per_task_batch_size" \
    --val_per_task_size "$val_per_task_size" \
    --seed "$seed"

train_parquet=$HOME/data/verl-agent/sdar_multitask/train.parquet
val_parquet=$HOME/data/verl-agent/sdar_multitask/test.parquet

# Shared env / rollout / model knobs reused by Stage 1 (generation) and Stage 2.
common_args=(
    data.max_prompt_length=4096
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation='left'
    data.return_raw_chat=True
    data.task_balance.enable=True
    data.task_balance.per_task_batch_size=$per_task_batch_size
    +data.task_balance.num_batches=$total_training_steps
    data.task_balance.tasks=[alfworld,search,webshop]
    +data.task_overrides.alfworld.max_prompt_length=2048
    +data.task_overrides.alfworld.truncation='error'
    +data.task_overrides.search.max_prompt_length=4096
    +data.task_overrides.search.truncation='left'
    +data.task_overrides.webshop.max_prompt_length=4096
    +data.task_overrides.webshop.truncation='error'
    +data.apply_chat_template_kwargs.enable_thinking=False
    +data.seed=$seed
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels
    +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch
    actor_rollout_ref.actor.ppo_mini_batch_size=60
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_per_gpu
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu
    actor_rollout_ref.rollout.max_model_len=$max_model_len
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ENGINE
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill
    +actor_rollout_ref.rollout.enable_prefix_caching=$enable_prefix_caching
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_per_gpu
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    env.env_name=multitask
    env.seed=0
    env.max_steps=50
    env.history_length=4
    env.rollout.n=$group_size
    env.search.search_url=$search_url
    env.multitask.tasks=[alfworld,search,webshop]
    env.multitask.max_steps.alfworld=50
    env.multitask.max_steps.search=4
    env.multitask.max_steps.webshop=15
    +env.multitask.history_length.alfworld=2
    +env.multitask.history_length.search=4
    +env.multitask.history_length.webshop=2
    env.multitask.val_per_task_batch_size=$val_per_task_size
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker
    trainer.n_gpus_per_node=3
    trainer.ray_wait_register_center_timeout=600
    trainer.nnodes=1
)

# ===================== Stage 1: teacher trajectory generation =====================
# Each task's frozen teacher generates its own fixed trajectory + top-20 dataset.
declare -A teacher_for=( [alfworld]=$teacher_alfworld [search]=$teacher_search [webshop]=$teacher_webshop )
for task in alfworld search webshop; do
    python3 -m verl.trainer.main_opd_offpolicy_gen \
        +gen.task=$task \
        +gen.teacher_path=${teacher_for[$task]} \
        +gen.out_dir=$teacher_traj_dir \
        +gen.num_trajectories=$gen_traj_per_task \
        +gen.topk=$opd_topk \
        data.train_files=$train_parquet \
        data.val_files=$val_parquet \
        data.train_batch_size=$per_task_batch_size \
        data.val_batch_size=$per_task_batch_size \
        "${common_args[@]}" \
        trainer.logger=['console'] \
        trainer.project_name='verl_agent_opd_offpolicy_multitask' \
        trainer.experiment_name=${experiment_name}_gen_${task} \
        trainer.total_training_steps=$total_training_steps \
        trainer.total_epochs=300 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=False
done

# ===================== Stage 2: off-policy distillation =====================
# Student distilled on the fixed teacher data with the same top-20 teacher-KL as OPD.
python3 -m verl.trainer.main_opd_offpolicy \
    data.train_files=$train_parquet \
    data.val_files=$val_parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    "${common_args[@]}" \
    +algorithm.opd.teacher_data_dir=$teacher_traj_dir \
    +algorithm.opd.kl_loss_coef=$opd_kl_loss_coef \
    +algorithm.opd.kl_loss_type=$opd_kl_loss_type \
    +algorithm.opd.topk=$opd_topk \
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
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_opd_offpolicy_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=300 \
    trainer.val_before_train=False $@
