set -x

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
# Throughput mechanisms (opt-in process env vars, accuracy-preserving; live in
# code, not in the expectations files — see docs/optimization_phase2.md):
#   ROLLOUT_KEEP_VLLM_AWAKE=1
#   (ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY /
#    ROLLOUT_COMPACT_RECORD default to on; they speed up the Stage-1 teacher
#    rollouts and the Stage-2 validation rollouts)
#   NOTE: leave ROLLOUT_PREFETCH_LOGPROB and ENV_RESET_PREFETCH off here —
#   neither stage has an old_log_prob phase or a per-step train rollout.

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# ===================== Stage 2: off-policy distillation =====================
python3 -m verl.trainer.main_opd_offpolicy \
    +trainer.expected_config=examples/opd_trainer/expected_multitask_offpolicy_config.yaml \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.max_model_len=4608 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    env.env_name=multitask \
    env.seed=1 \
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
    +algorithm.opd.teacher_data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj \
    +algorithm.opd.kl_loss_coef=1.0 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
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
    trainer.experiment_name=sdar_multitask_offline_qwen3_1.7b \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_opd_offpolicy_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
