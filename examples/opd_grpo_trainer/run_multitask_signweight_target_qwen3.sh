set -x

# OPD + GRPO multitask, SIGN WEIGHTING IN *TARGET* MODE, Qwen3-1.7B.
#
# The mode=position sibling (run_multitask_signweight_qwen3.sh) reallocates how
# hard each token is learned and provably converges to the same place as the
# plain arm. This one moves the target itself, so it is the arm that can beat
# what the on-task teacher alone determines -- and the one that can be wrong.
# The two are separate claims, which is why they are separate scripts rather
# than one flag: a result belongs to whichever of them produced it.
#
# Identical to run_multitask_qwen3.sh except for algorithm.opd.sign_weight.*,
# so the two runs differ in the weighting of the teacher KL and in nothing else
# -- same teachers, same data, same batch sizes, same eval protocol. That is
# what makes the plain arm the control for this one.
#
# What the weighting does (verl/trainer/ppo/sign_weights.py): each teacher is a
# single-task RL fine-tune of one base policy, so sign(log pi_m - log pi_0) says
# whether task m's RL raised or lowered a candidate token here. Where the two
# OFF-task teachers agree with each other AND with the on-task teacher, the
# signal is treated as shared structure and weighted up (1.25); where they
# oppose it, down (0.75); where anyone is silent or the two off-task teachers
# split, it is left alone (1.0).
#
#   mode=position  one weight per token multiplies the per-token KL. The
#                  minimiser is still the on-task teacher, so this changes how
#                  fast/where the student learns, never what it converges to.
#   mode=target    the teacher's own top-k distribution is reweighted and
#                  renormalised, moving the fixed point to ~ w(v) p_teacher(v).
#                  This is the variant that can actually inject an off-task
#                  opinion -- and the one that can inject a wrong one.
#
# Cost: three extra frozen forwards per step (base over all rows, each teacher
# over the 2/3 of rows that are not its own task).
#
# enable=False reproduces the plain arm exactly: no base worker is built, no
# extra forward runs, and nothing is written into the batch.
#
# EVERY parameter lives as a literal argument of the python3 commands below —
# there is deliberately NO variable block and NO ${VAR:-default} fallback, so
# there is exactly one place to read or edit a setting. One-off overrides can
# be appended on the command line (they become trailing Hydra overrides via
# "$@") and still pass the expectations check.
#
# INTENT LOCK: examples/opd_grpo_trainer/expected_multitask_config.yaml pins
# the scientific knobs (loss type/coefs, seeds, batch sizes, teachers, eval
# protocol). main_opd_grpo validates the composed config against it after its
# own injection and refuses to start on any mismatch. To change such a knob,
# edit the argument below AND the expectations file in the same commit.
#
# Loss: policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * kl_loss_coef.
#   - actor.pg_loss_coef=1.0 keeps the GRPO policy gradient (env-reward
#     advantages; adv_estimator=grpo required); set 0 to recover pure OPD.
#   - algorithm.opd.kl_loss_coef=0.01 matches examples/grpo_opsd_trainer, which is
#     the same shape of run: a GRPO policy gradient plus an on-policy distillation
#     term. Every script there sets sdar_coef=0.01 and leaves pg_loss_coef at the
#     1.0 default, and its distillation-only sibling (examples/opsd_trainer) keeps
#     sdar_coef=0.01 while setting pg_loss_coef=0 -- i.e. the distillation
#     coefficient does not change when the policy gradient is switched on; only
#     pg_loss_coef does. This arm now follows the same rule. It was the one place
#     the two families disagreed: OPD ran its distillation term at 1.0 in both the
#     pure and the combined arm, so the combined arm was summing two
#     full-strength objectives.
#     Still worth measuring rather than trusting: topk_kl is a dense reverse KL
#     over k=20, a different quantity from the SDAR loss 0.01 was chosen for. Both
#     terms are logged already weighted and per task (actor/pg_loss_weighted,
#     actor/teacher_kl_loss_weighted), so read their ratio at step 1 and retune
#     HERE and in the expectations file together.
#   - algorithm.opd.kl_loss_type: low_var_kl (single-token estimator) or
#     topk_kl (dense top-k+tail reverse KL; support size algorithm.opd.topk).
#   - use_kl_loss=False / entropy_coeff=0 keep the other terms off;
#     use_teacher_kl_loss is force-injected in main_opd_grpo.py.
#     NOT matched to grpo_opsd, deliberately and only here: that arm also runs a
#     reference-KL term (use_kl_loss=True, kl_loss_coef=0.01) beside its
#     distillation, whereas main_opd_grpo force-injects use_kl_loss=False. Two KL
#     terms pulling the student toward two different targets is its own decision,
#     so this stays as pure OPD has it. Flagged because the coefficient above now
#     follows grpo_opsd and the KL structure does not.
#   - algorithm.opd.normalize_loss_by_task: each task contributes 1/3 of the
#     loss. Without it the token-mean hands alfworld ~69% and search ~4% of the
#     gradient, purely because of the 50/15/4-turn episode caps -- a weighting
#     nobody chose, and one the multitask SFT and pure-OPD arms do not use. It
#     weights BOTH terms of the loss above: weighting only the teacher KL would
#     leave the policy gradient on the token-count split and silently change the
#     ratio between them, which is what pg_loss_coef is supposed to control.
# Teachers: per-task single-task RL checkpoints, created as role="ref" worker
#   groups (they reuse actor_rollout_ref.ref.* settings: log-prob micro batch,
#   FSDP param_offload); each sample is distilled from the teacher of its task.
#   ref.fsdp_config.param_offload=False keeps the teacher shards resident on GPU.
#   That key used to be dead — the ref/teacher policy was pinned to FSDP
#   CPUOffload regardless of it — so with it honoured, FSDP no longer re-fetches
#   every unit's parameters from host memory on every micro-batch. teacher_forward
#   was measured pulling 7.6-8.9 GB/s over PCIe for the whole phase (gen, by
#   comparison, sits near 2). Resident costs 3 teachers * 3.4GB bf16 / 3 GPUs =
#   3.4GB per GPU. If a bigger student or a longer context leaves no room for
#   that, put it back to True — this is a placement knob, the distillation
#   targets are identical either way.
#   ref.fsdp_config.sharding_strategy=shard_grad_op goes one step further: with
#   the shards resident, FULL_SHARD still re-all-gathers each teacher's
#   parameters over PCIe for EVERY micro-batch (teacher_forward's pcieRX sat at
#   ~4.3 GB/s on the pure-OPD arm even after the offload fix). ZeRO-2 does not
#   reshard after forward, and the teachers have no backward, so each is gathered
#   once per phase and stays whole. Placement only, bit-identical logits; costs
#   up to 3 * 3.4GB gathered per GPU on top of the shards, measured at +8.2GB
#   there. Not pinned in the lock, same exemption as param_offload.
#
# ONE RETRIEVER, POSSIBLY SHARED. env.search.search_url can point at the same
# server as another concurrent run. What makes that safe is not the URL but the
# retry policy beside it: env.search.max_retries=null waits for a timeout /
# refused connection / 5xx to clear instead of giving up. Giving up is not a
# no-op — the client hands the error text back as the retrieval result, so it
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
# HOST PATHS. Teachers, checkpoints and data are all written relative to $HOME,
# and load_expectations expands $HOME before comparing, so one lock file pins the
# same teacher on either machine — what it asserts is WHICH checkpoint, not where
# $HOME is. Moving to a host with a different home needs no edit here.
#
# Throughput mechanisms (process env vars, accuracy-preserving; live in code, not
# in the expectations file — see docs/optimization_phase2.md):
#   exported below: ROLLOUT_KEEP_VLLM_AWAKE=1  ENV_RESET_PREFETCH=1
#                   TASK_BALANCE_INTERLEAVE=1
#   default on:     ROLLOUT_PREFETCH_TEACHER  ROLLOUT_SKIP_DONE_PREPROC
#                   ROLLOUT_DECODE_ACTIVE_ONLY  ROLLOUT_COMPACT_RECORD
#   NOT set:        ROLLOUT_PREFETCH_LOGPROB — see the PICK ONE note below
#   Set any of them to 0 to turn it off; the exports honour the caller's value.
#   The first three used to be left for whoever launched the run to export, which
#   is the wrong default for a 300-step run that gets restarted: forgetting them
#   is silent, since the config-side mechanisms keep working and only the rollout
#   ones go quiet.
#
# ROLLOUT_PREFETCH_TEACHER scores rows with their task's teacher during the
# rollout instead of after it. A turn's row is final the moment it is recorded
# (later turns only append), so rows are queued per turn — not, as previously, at
# trajectory end, which kept alfworld's rows out of the pool until episode end and
# capped the hit rate at 0.28-0.46. What the scoring fills is the driver's own CPU
# time — decode, envs.step, the next turn's tokenization — measured at 18% of the
# rollout with the GPU at 0. It does NOT fill the alfworld generation tail: the
# teachers share one WorkerDict per GPU with the rollout (init_workers ->
# create_colocated_worker_cls) and a Ray actor runs one call at a time, so a
# teacher call issued during generate_sequences would only queue behind it.
# Frozen teachers, so the targets do not depend on when a row is scored, but a row
# lands in a different micro-batch than the post-rollout path would put it in,
# which moves the last bits of a packed GEMM — same class as no_sync_grad_accum,
# not bit-identical.
#
#   DO NOT LOWER ROLLOUT_PREFETCH_TEACHER_CHUNK (default 128). One chunk is
#   issued per turn, so the rows that can be prefetched are capped at
#   CHUNK * turns <= CHUNK * 50 against a step of 6,880-7,200 rows: 128 tops out
#   near 0.91 hit rate, 32 near 0.23 — BELOW the 0.28-0.46 that trajectory-end
#   queueing already reached. A small chunk was right while the queue was starved
#   and is wrong now that per-turn queueing fills it from turn 1; the chunk is the
#   only thing rate-limiting this. Lower it only if tchWait measurably worsens the
#   step, and judge that on teacher_forward s/step plus wall clock, not on tchWait
#   alone — tchWait is teacher work the trainer no longer has to do, not waste.
#
#   ROLLOUT_PREFETCH_TEACHER_ADAPTIVE=1 (default) makes that 128 the FLOOR rather
#   than the size: each chunk is sized from the glue window it will actually run
#   under — (rows/second measured from the last completed chunk) x (the previous
#   turn's glue seconds) — clamped between CHUNK and
#   ROLLOUT_PREFETCH_TEACHER_CHUNK_MAX (512). One constant cannot fit both ends of
#   a rollout: the early turns hold most of the CPU glue (turn 0 alone is seconds
#   of envs.step) while the queue is barely filled, and the later turns collapse to
#   a fraction of a second of glue while the backlog is at its largest, so a fixed
#   size underfills the front and overshoots the tail. Per-turn queueing makes the
#   front-loading worse, not better — the queue is full from turn 1, so the turns
#   with the widest windows are exactly the ones a fixed 128 starves. Set it to 0
#   to go back to the fixed size. Chunking only decides how many already-final
#   rows one call scores, so like the chunk size itself it cannot change a value.
#
#   PICK ONE of ROLLOUT_PREFETCH_TEACHER / ROLLOUT_PREFETCH_LOGPROB on this
#   recipe. Unlike pure OPD (which has no old_log_prob phase at all), OPD+GRPO
#   runs both, and they want the same resource twice over: the same CPU-glue
#   window between generations, and the same colocated WorkerDict, where a Ray
#   actor serves one call at a time. Enabling both does not corrupt anything —
#   each mechanism still returns the same rows — but the second one issued waits
#   on the first instead of on an idle GPU, so the overlap they exist to buy is
#   spent queueing. TEACHER is the one on by default here, so adding
#   ROLLOUT_PREFETCH_LOGPROB=1 means running both; the rollout prints a warning
#   when it sees that. Measure them separately before assuming either is free.
#   Turning both on now costs one thing more than it used to: the adaptive chunk
#   controller below sizes the teacher chunk from the WALL-CLOCK glue window, and
#   with the log-prob prefetch also in it that window is not the teacher's to
#   fill — so it would size chunks against time already spent and spill into
#   tchWait. Set ROLLOUT_PREFETCH_TEACHER_ADAPTIVE=0 if you ever do run both.
#
# Dynamic (token-budget) micro-batching is wired below but OFF by default so the
# fixed-batch objective is unchanged. It packs micro-batches to a token budget
# (helps when the topk_kl backward forces tiny fixed micro-batches). Enable via a
# trailing override; dynamic_bsz_token_scale=True keeps token-mean exact
# (token-weighted, grouping-invariant) rather than the sample-count reweighting:
#   e.g.  bash run_multitask_qwen3.sh actor_rollout_ref.actor.use_dynamic_bsz=True
#
# Actor-update mechanisms (config, not env vars). All three are ON BY DEFAULT in
# verl/trainer/config/ppo_trainer.yaml and are still written out literally below,
# because this script's rule is that every parameter is readable in one place —
# and because two of them are pinned in the intent lock, which reads the
# EFFECTIVE config and cannot tell a default from an argument. They target the
# PCIe collectives: with no NVLink between the GPUs every FSDP all-gather and
# reduce-scatter crosses the bus, and update_actor is the phase with by far the
# highest measured PCIe traffic (~10.5 GB/s TX vs ~2.0 during gen). NCCL
# collectives count as SM-busy, so the 93-97% SM reading that made this phase
# look compute-bound could not tell computing from communicating. Both of the
# first two sit on the gradient path and are therefore pinned in
# expected_multitask_config.yaml, against the usual rule that performance knobs
# stay out of the intent lock.
#   sharding_strategy=shard_grad_op — ZeRO-2. Keeps parameters gathered from
#     forward through backward, so a layer all-gathers once per micro-batch
#     instead of three times under gradient checkpointing. Arithmetic-neutral;
#     costs roughly the unsharded parameter size minus its shard in peak memory.
#   no_sync_grad_accum=True — accumulate gradients across a mini-batch's
#     micro-batches and reduce ONCE. Under ZeRO-2 this also drops the
#     per-micro-batch re-gather. NOT bit-identical: the partial sums reduce in a
#     different order, so gradients differ in their last bits (identical
#     expectation).
#   fsdp_config.forward_prefetch=True — issue the next FSDP unit's all-gather
#     while the current one computes. Scheduling only, arithmetic untouched, so
#     it is a plain performance knob and is not pinned.
# Two mechanisms come with no knob because they change no value: the top-k
# distillation forward now runs logsumexp/topk/gather only over the rows that
# survive the response slice (prompts average ~3x responses here, and on the
# student side the discarded work was inside the autograd graph), and
# logger-only scalars stay as 0-d GPU tensors until the end of update_policy
# instead of being read back per micro-batch (~450 forced host syncs a step).
#
# Wasted-work removals. Each deletes computation whose result was already being
# thrown away, so none of them changes a value that reaches the loss.
#   actor.response_only_logits=True / ref.response_only_logits=True — run lm_head
#     on the response rows only, by handing the row selection to the model as
#     `logits_to_keep`. Everything the actor and teacher forwards return is sliced
#     to [-response_length-1:-1], and prompts are ~3x responses here, so roughly
#     three quarters of the vocab projection was built at (rows, 151936) and
#     dropped — forward and backward, and it is the step's largest activation. The
#     transformer body still runs on every token: a response position attends to
#     the prompt's KV, so that part cannot be skipped. On the teacher this also
#     shrinks the allocation that OOMed at step 136 of the pure-OPD run, because
#     ROLLOUT_PREFETCH_TEACHER (on by default here) runs it while vLLM is awake and
#     holding its KV cache. lm_head is a per-position linear map, so selecting the
#     rows before or after it is the same arithmetic — only the GEMM shape moves,
#     which is the accuracy class of a micro-batch change, so both are PINNED in
#     the intent lock and go into every arm at once.
#   The reward manager no longer decodes prompts and responses to strings on
#     training calls. EpisodeRewardManager built both for the num_examine console
#     sample, which is 0 on every training call, against a score that is the env's
#     episode reward and never looks at the text: ~14k discarded decodes a step.
#     Validation (num_examine>0) prints exactly what it did before.
#   generate_sequences no longer calls empty_cache inside a rollout session. It ran
#     once per turn (~50x per rollout) even with ROLLOUT_KEEP_VLLM_AWAKE holding
#     the engine up, forcing a device synchronize to hand back blocks the allocator
#     immediately re-acquired — and it cannot free vLLM's KV cache anyway, since
#     the engine owns that. end_rollout_session() still empties it at the boundary.
#   On THIS arm the ref half of that flag covers more than the three teachers:
#     the base policy and the off-task teacher scoring (sign_weight_forward/*)
#     are built with role="ref" from the same actor_rollout_ref config, so all
#     three of the extra frozen forwards this arm pays for shrink with it too.
# NOT taken from pure OPD: rollout.return_rollout_log_probs=False. That arm's thin
# loop has no old_log_prob phase, so nothing reads the column. THIS recipe runs the
# rollout-vs-actor drift check in RayPPOTrainer.fit (rollout_probs_diff), which is
# its only consumer — leave it on.

export ALFWORLD_DATA=$HOME/data/alfworld
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
# On by default, for the same reason the FSDP knobs are literals below: a 300-step
# run gets restarted, and a mechanism that has to be exported by hand is one that
# will eventually be missing from a restart. All three are accuracy-preserving
# (see the header); each still honours an explicit 0 from the caller.
# ROLLOUT_PREFETCH_LOGPROB is deliberately absent -- ROLLOUT_PREFETCH_TEACHER is
# already on by default and the two contend for the same glue window and the same
# colocated WorkerDict (the PICK ONE note above).
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export ENV_RESET_PREFETCH=${ENV_RESET_PREFETCH:-1}
export TASK_BALANCE_INTERLEAVE=${TASK_BALANCE_INTERLEAVE:-1}
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

python3 -m verl.trainer.main_opd_grpo \
    +trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_signweight_target_config.yaml \
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
    actor_rollout_ref.actor.pg_loss_coef=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.no_sync_grad_accum=True \
    actor_rollout_ref.actor.response_only_logits=True \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.sharding_strategy=shard_grad_op \
    actor_rollout_ref.ref.response_only_logits=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.invalid_action_penalty_coef_by_task='{alfworld:0.1,search:0.01,webshop:0.1}' \
    algorithm.use_kl_in_reward=False \
    +algorithm.opd.teacher_paths.alfworld=$HOME/checkpoints/teachers/alfworld_step300 \
    +algorithm.opd.teacher_paths.search=$HOME/checkpoints/teachers/search_step300 \
    +algorithm.opd.teacher_paths.webshop=$HOME/checkpoints/teachers/webshop_step300 \
    +algorithm.opd.kl_loss_coef=0.01 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
    +algorithm.opd.normalize_loss_by_task=True \
    +algorithm.opd.sign_weight.enable=True \
    +algorithm.opd.sign_weight.mode=target \
    +algorithm.opd.sign_weight.agree_weight=1.25 \
    +algorithm.opd.sign_weight.disagree_weight=0.75 \
    +algorithm.opd.sign_weight.deadzone=0.1 \
    +algorithm.opd.sign_weight.base_path=Qwen/Qwen3-1.7B \
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
    trainer.project_name='verl_agent_opd_grpo_signweight_target_multitask' \
    trainer.experiment_name=sdar_multitask_opd_grpo_signweight_target_qwen3_1.7b \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/verl_agent_opd_grpo_tmp_multitask \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
