set -x

# Off-policy multitask distillation (offline KD), Qwen3-1.7B --
# STUDENT-INDEXED top-k arm. Stage 2 only; it reuses the pool Stage 1 already
# wrote (run_multitask_offpolicy_qwen3.sh).
#
# WHAT DIFFERS FROM run_multitask_offpolicy_qwen3_nogen.sh, and nothing else:
#   actor.student_indexed_topk=True   the KL's support is the STUDENT's top-20
#   {actor,ref}.response_only_logits=True   lm_head on the response rows only
#   algorithm.opd.teacher_paths.*     the three teachers, loaded at Stage 2
#   trainer.expected_config           this arm's own intent lock
#   trainer.experiment_name           its own run identity
#   trainer.default_local_dir         its own checkpoint tree -- NOT cosmetic.
#     Checkpoints land in default_local_dir/global_step_N; experiment_name is not
#     in that path. Sharing the directory with the control arm would have this run
#     overwrite the control's checkpoints step for step, and -- because
#     trainer.resume_mode defaults to auto and reads
#     default_local_dir/latest_checkpointed_iteration.txt -- would have it RESUME
#     FROM THE CONTROL ARM'S WEIGHTS on its first start, silently.
#
# WHY THE TEACHERS COME BACK. The loss is a coarse-grained reverse KL: exact on
# a support, everything outside it in one tail bucket. What that drops is
#     KL_full - KL_A = tail_s * KL(p_s|A-bar || p_t|A-bar),
# weighted by the STUDENT's tail mass -- so a support taken from the teacher
# leaves precisely the region where the student has drifted outside it, which is
# the region the term exists to penalise. Taking the support from the student
# covers it. Both are lower bounds on the same full KL, so this is a tighter
# bound and not a different objective -- but the pool's top-20 was chosen before
# the student existed and cannot answer at the student's ids, so the teachers
# have to be present.
#
# WHAT THAT COSTS. One teacher forward per row, for the run: 36,000 trajectories
# per task drawn 120 a step is 300 steps, i.e. one pass over the pool -- the same
# teacher compute a single re-scoring pass would take, except evaluated where the
# student actually put its mass. On the card it adds three frozen 1.7B teachers,
# their three unsharded output projections (~1.9 GB/rank, held for the run) and
# the step's hidden-state cache. RUN ONE STEP FIRST and read
# perf/max_memory_allocated_gb and teacher_cache/gb before starting 300.
#
# THIS ARM DOES NOT COMPARE WITH THE CONTROL'S LOSS CURVE. Different support,
# different (still valid) bound. Compare validation scores, not actor/teacher_kl_loss.
#
# THE TEACHER-SIDE KNOBS BELOW ARE DELIBERATELY NOT THE ON-POLICY ARM'S. That arm
# (examples/opd_trainer/run_multitask_qwen3.sh on the student-topk branch) sets
# ref.fsdp_config.param_offload=False, ref.fsdp_config.sharding_strategy=shard_grad_op
# and ref.log_prob_micro_batch_size_per_gpu=4. None of the three transfers, because
# its teacher and this one are called differently:
#
#   param_offload. There the teacher is scored per TURN inside the rollout, so
#     paying a host-to-device copy of the weights each time is the dominant cost.
#     Here it is called three times a step, once per task: the copy is ~10 GB over
#     PCIe per step against a ~580 s step, well under 1%, and keeping offload ON
#     is what buys back the ~10 GB/rank that three resident 1.7B teachers cost --
#     on a card this arm has already added a hidden-state cache and three
#     unsharded projections to. So it stays True.
#   sharding_strategy. ZeRO-2 pays off because ZeRO-3 all-gathers a layer three
#     times per micro-batch under gradient checkpointing: forward, recompute,
#     backward. The teacher has no backward and no checkpointing, so it gathers
#     each layer once either way -- ZeRO-2 would buy nothing and hold the params
#     unsharded for the whole call. Left at the default.
#   log_prob_micro_batch_size_per_gpu. Theirs is 4 as a MEMORY bound, not a speed
#     choice. Ours is 16, inherited from a script where the ref role was never
#     instantiated at all. 16 gives the better GEMM shape and is kept -- but it is
#     the first knob to lower if the smoke step below runs out of memory, and 4 is
#     the value the on-policy arm settled on.
#
# WHAT IT GIVES BACK. The pool's two largest columns -- the teacher's recorded
# top-k, ~82 KB of the ~123 KB a row costs resident, about 105 GB across the pool
# -- are the teacher-indexed arm's training target and are read by nothing here.
# They are dropped at load, which matters because the Stage-2 profile measured
# this box at 494/503 GB of host RAM. OFFPOLICY_KEEP_TEACHER_TOPK=1 keeps them,
# for the one run that cross-checks the live Stage-2 teacher against what Stage 1
# recorded; do not leave it on for 300 steps.
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
#   OFFPOLICY_KEEP_TEACHER_TOPK=1 — keep the pool's recorded teacher top-k that
#     this arm does not read (~105 GB of host RAM). Off by default; see above.
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
# TEACHER POOL: algorithm.opd.teacher_data_dir below points at the raw Stage-1
# pool, which the loader filters on every start (padding rows written by an older
# Stage 1, plus the columns no Stage-2 loss reads). Do that once instead:
#
#   python3 scripts/cache_teacher_pool.py \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj \
#       $HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache --arm kd
#
#   bash examples/opd_trainer/run_multitask_offpolicy_qwen3_nogen.sh \
#       ++algorithm.opd.teacher_data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_kd_cache
#
# Double plus: the key is added with a single '+' below, so another '+' would be a
# second append of an existing key and Hydra refuses it. The cache holds exactly
# what the loader would have built, file for file and row for row, so the run is
# unchanged. It is ARM-SPECIFIC — an --arm sft cache has no teacher top-k and this
# run must not read one (scripts/inspect_teacher_pool.py reports what a pool holds).

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
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export OFFPOLICY_BATCH_PREFETCH=${OFFPOLICY_BATCH_PREFETCH:-1}

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

python3 -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen3-1.7B'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m); print(f'Validated {m}')"

# ===================== Stage 2: off-policy distillation =====================
python3 -m verl.trainer.main_opd_offpolicy \
    +trainer.expected_config=$OPD_DIR/expected_multitask_offpolicy_studenttopk_config.yaml \
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
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.no_sync_grad_accum=True \
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
    +algorithm.opd.student_indexed_topk=True \
    actor_rollout_ref.actor.response_only_logits=True \
    actor_rollout_ref.ref.response_only_logits=True \
    +algorithm.opd.teacher_paths.alfworld=/opt/home/ohara/checkpoints/teachers/alfworld_step300 \
    +algorithm.opd.teacher_paths.search=/opt/home/ohara/checkpoints/teachers/search_step300 \
    +algorithm.opd.teacher_paths.webshop=/opt/home/ohara/checkpoints/teachers/webshop_step300 \
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
    trainer.experiment_name=sdar_multitask_offline_studenttopk_qwen3_1.7b \
    trainer.default_local_dir=/opt/home/ohara/checkpoints/verl_agent_opd_offpolicy_multitask_studenttopk \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
