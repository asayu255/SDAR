set -x

# Pure OPD multitask (alfworld + search + webshop), Qwen3-1.7B.
#
# EVERY parameter lives as a literal argument of the python3 commands below —
# there is deliberately NO variable block and NO ${VAR:-default} fallback, so
# there is exactly one place to read or edit a setting. One-off overrides can
# be appended on the command line (they become trailing Hydra overrides via
# "$@") and still pass the expectations check.
#
# HOST: 2 GPUs (tamago / 100.86.45.34). The GPU count is not a scientific knob and
# is not in the intent lock, so a different host overrides it on the command line:
#   bash examples/opd_trainer/run_multitask_qwen3.sh trainer.n_gpus_per_node=3
# Teacher and checkpoint paths are written relative to $HOME so the same script
# resolves correctly on either machine.
#
# INTENT LOCK: examples/opd_trainer/expected_multitask_config.yaml pins the
# scientific knobs (loss type/coefs, seeds, batch sizes, teachers, eval
# protocol). main_opd validates the composed config against it after its own
# injection and refuses to start on any mismatch. To change such a knob, edit
# the argument below AND the expectations file in the same commit.
#
# Loss: pure per-task teacher-KL distillation on the student's own on-policy
#   responses. main_opd force-injects pg_loss_coef=0 / entropy_coeff=0 /
#   use_kl_loss=False, so nothing but the teacher KL enters the loss.
#   - algorithm.opd.kl_loss_type: low_var_kl (single-token estimator) or
#     topk_kl (dense top-k+tail reverse KL; support size algorithm.opd.topk).
#   - algorithm.opd.normalize_loss_by_task: each task contributes 1/3 of the
#     loss. Without it the token-mean hands alfworld ~69% and search ~4% of the
#     gradient, purely because of the 50/15/4-turn episode caps -- a weighting
#     nobody chose, and one the multitask SFT arm does not use. Same mechanism
#     and same weights as that arm, so the arms differ in the loss only.
# Teachers: per-task single-task RL checkpoints, created as role="ref" worker
#   groups (they reuse actor_rollout_ref.ref.* settings: log-prob micro batch,
#   FSDP CPUOffload); each sample is distilled from the teacher of its task.
#
# ONE RETRIEVER, POSSIBLY SHARED. env.search.search_url can point at the same
# server as another concurrent run. What makes that safe is not the URL but the
# retry policy beside it: env.search.max_retries=null waits for a timeout /
# refused connection / 5xx to clear instead of giving up. Giving up is not a
# no-op -- the client hands the error text back as the retrieval result, so it
# lands in the <information> block the model is trained on with nothing in the
# metrics to say so, and under a shared retriever an exhausted budget is exactly
# what a load spike looks like. 4xx and malformed JSON still fail immediately:
# waiting cannot turn a bad URL into a document. Both knobs are pinned in the
# expectations file, because they decide what enters the data.
#
# env.search.timeout=600 is generous but finite on purpose; see the expectations
# file for why null is worse here. A request that is still retrying says so in
# the log every ~60s, so an intentional wait is never mistaken for a hang.
#
# Throughput mechanisms (opt-in process env vars, accuracy-preserving; live in
# code, not in the expectations file — see docs/optimization_phase2.md):
#   ROLLOUT_KEEP_VLLM_AWAKE=1  ENV_RESET_PREFETCH=1  TASK_BALANCE_INTERLEAVE=1
#   (ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY /
#    ROLLOUT_COMPACT_RECORD default to on)
#   NOTE: leave ROLLOUT_PREFETCH_LOGPROB off here — pure OPD's thin loop has no
#   old_log_prob phase, so prefetched values would never be consumed.
#
# Actor-update mechanisms (config, not env vars). These target the PCIe
# collectives: tamago's two GPUs have no NVLink, so every FSDP all-gather and
# reduce-scatter crosses the bus, and update_actor is the phase with by far the
# highest measured PCIe traffic. Both of the first two sit on the gradient path
# and are therefore pinned in expected_multitask_config.yaml.
#   sharding_strategy=shard_grad_op — ZeRO-2. Keeps parameters gathered from
#     forward through backward, so a layer all-gathers once per micro-batch
#     instead of three times under gradient checkpointing. Arithmetic-neutral;
#     costs roughly the unsharded parameter size minus its shard in peak memory.
#   no_sync_grad_accum=True — accumulate gradients across a mini-batch's
#     micro-batches and reduce ONCE (60/5 = 12 reduces per mini-batch become 1).
#     Under ZeRO-2 this also drops the per-micro-batch re-gather. NOT
#     bit-identical: the partial sums reduce in a different order, so gradients
#     differ in their last bits (identical expectation).
#   fsdp_config.forward_prefetch=True — issue the next FSDP unit's all-gather
#     while the current one computes. Scheduling only, arithmetic untouched, so
#     it is a plain performance knob and is not pinned.

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# Data prep. These literals are shared with the training command below and are
# also cross-checked there via the expectations file (per_task_batch_size=15,
# val_per_task_size=126, total_training_steps=300, seed=1).
python3 -m examples.data_preprocess.prepare_sdar_multitask \
    --search_dir "$HOME/data/searchR1_processed_direct" \
    --local_dir "$HOME/data/verl-agent/sdar_multitask" \
    --total_training_steps 300 \
    --per_task_batch_size 15 \
    --env_train_per_task_size 15 \
    --val_per_task_size 126 \
    --seed 1

python3 -m verl.trainer.main_opd \
    +trainer.expected_config=examples/opd_trainer/expected_multitask_config.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/sdar_multitask/train.parquet \
    data.val_files=$HOME/data/verl-agent/sdar_multitask/test.parquet \
    data.train_batch_size=45 \
    data.val_batch_size=126 \
    +data.seed=1 \
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
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.ppo_mini_batch_size=60 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    +actor_rollout_ref.actor.dynamic_bsz_token_scale=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.pg_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.no_sync_grad_accum=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.rollout.max_model_len=4608 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.alfworld.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.search.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.temperature=0.4 \
    +actor_rollout_ref.rollout.val_kwargs_by_task.webshop.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.invalid_action_penalty_coef_by_task='{alfworld:0.1,search:0.01,webshop:0.1}' \
    algorithm.use_kl_in_reward=False \
    +algorithm.opd.teacher_paths.alfworld=$HOME/checkpoints/teachers/alfworld_step300 \
    +algorithm.opd.teacher_paths.search=$HOME/checkpoints/teachers/search_step300 \
    +algorithm.opd.teacher_paths.webshop=$HOME/checkpoints/teachers/webshop_step300 \
    +algorithm.opd.kl_loss_coef=1.0 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
    +algorithm.opd.normalize_loss_by_task=True \
    env.env_name=multitask \
    env.seed=1 \
    env.max_steps=50 \
    env.history_length=4 \
    env.rollout.n=8 \
    env.search.search_url='http://0.0.0.0:8000/retrieve' \
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
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_opd_multitask' \
    trainer.experiment_name=opd_multitask_qwen3_1.7b \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/verl_agent_opd_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
