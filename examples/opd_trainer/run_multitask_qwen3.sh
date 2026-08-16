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
#   FSDP param_offload); each sample is distilled from the teacher of its task.
#   ref.fsdp_config.param_offload=False keeps the teacher shards resident on GPU.
#   With them on CPU, FSDP re-fetches every unit's parameters from host memory on
#   every micro-batch, and teacher_forward was measured pulling 7.6-8.9 GB/s over
#   PCIe for the whole phase (gen, by comparison, sits near 2). Resident costs
#   3 teachers * 3.4GB bf16 / 2 GPUs = 5.1GB per GPU. If a bigger student or a
#   longer context leaves no room for that, put it back to True -- this is a
#   placement knob, the distillation targets are identical either way.
#   ref.fsdp_config.sharding_strategy=shard_grad_op goes one step further: under
#   FULL_SHARD the resident shards are still re-all-gathered over PCIe for EVERY
#   micro-batch (teacher_forward's pcieRX sat at ~4.3 GB/s even after the
#   offload fix). ZeRO-2 does not reshard after forward, and the teachers have
#   no backward, so each teacher is gathered once and then stays whole; the
#   per-micro-batch collective disappears. Placement only, bit-identical logits.
#   Costs up to 3 * 3.4GB gathered per GPU on top of the shards; measured at
#   +8.2GB (89.3 -> 97.5 max_memory_allocated) on the 2-GPU run.
#
#   ref.log_prob_micro_batch_size_per_gpu is back to 16 (it was 8 for one run),
#   and both numbers were MEMORY bounds rather than throughput choices.
#   compute_ref_topk_log_prob runs the teacher through lm_head, which
#   materialized the full vocabulary over every row of the sequence: at 16 rows
#   of webshop-length prompts that was 16 * ~2.3k tokens * 151936 * bf16 =
#   10.47 GiB in one allocation, and step 136 OOMed on exactly that with
#   9.50 GiB free. What paid for going back up is ref.response_only_logits: the
#   projection now runs on the response rows only, and prompts are ~75% of the
#   tokens here, so the same 16 rows allocate about a quarter of that. Under
#   student_indexed_topk it is smaller again -- the teacher's own top-k is built
#   for two micro-batches a step instead of every row. The ref micro batch does
#   NOT enter adjust_batch's lcm here (batch_size_divisor only includes it when
#   use_kl_in_reward or actor.use_kl_loss is set, and this arm has neither), so
#   changing it moves no padding and no data -- per-row log probs are
#   independent under rmpad. Drop it again if a longer-prompt task is added.
#
#   rollout.log_prob_micro_batch_size_per_gpu is 10, and it is NOT a throughput
#   knob here at all: this arm has no old_log_prob phase, so compute_log_prob is
#   never called and the value's only effect is on adjust_batch's divisor,
#   lcm(rollout*2, rollout*2, actor*2). At 16 that was lcm(32,32,20)=160 and step
#   1 discarded 116 rows to reach it; at 10 it is lcm(20,20,20)=20, about 10.
#   1.6% of the batch, for free. It DOES decide which rows are dropped, so it
#   changes the data and is pinned.
#
#   actor.ppo_micro_batch_size_per_gpu is 10 for the same reason (it was 5), and
#   this one is NOT free: it is a different packed GEMM, so gradients differ in
#   their last bits -- the no_sync_grad_accum class, not bit-identical. It has to
#   be the same across the arms being compared. lcm(10*2, 16*2) = 160, the same
#   divisor adjust_batch used at 5, so the padding row count does not move.
#
#   Why that allocation is tighter than it used to be: ROLLOUT_PREFETCH_TEACHER
#   moves the teacher forward INTO the rollout, where vLLM is awake and holding
#   its KV cache, instead of after it where the engine has been slept. Per-turn
#   queueing then took the share that runs in that tighter regime from about
#   half to essentially all (hit_rate 0.53 -> 0.99). Same work, less headroom --
#   so the micro batch has to be sized for the awake-engine case.
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
# Throughput mechanisms (process env vars, accuracy-preserving; they live in code,
# not in the expectations file — see docs/optimization_phase2.md). All four are
# exported below rather than left to the operator: a 300-step run gets restarted,
# and a mechanism that has to be exported by hand is one that will eventually be
# missing from a restart. Set any of them to 0 to disable:
#   ROLLOUT_KEEP_VLLM_AWAKE=1  ENV_RESET_PREFETCH=1  TASK_BALANCE_INTERLEAVE=1
#   ROLLOUT_PREFETCH_TEACHER=1
#   (ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY /
#    ROLLOUT_COMPACT_RECORD default to on)
#   NOTE: leave ROLLOUT_PREFETCH_LOGPROB off here — pure OPD's thin loop has no
#   old_log_prob phase, so prefetched values would never be consumed. It is not
#   exported below for that reason.
#
# ROLLOUT_PREFETCH_TEACHER scores rows with their task's teacher during the
# rollout instead of after it. A turn's row is final the moment it is recorded
# (later turns only append), so rows are queued per turn — not, as previously, at
# trajectory end, which kept alfworld's rows out of the pool until episode end
# and capped the hit rate at 0.28-0.46. What the scoring fills is the driver's
# own CPU time — decode, envs.step, the next turn's tokenization — measured at
# 18% of the rollout with the GPU at 0. It does NOT fill the alfworld generation
# tail: the teachers share one WorkerDict per GPU with the rollout (init_workers
# -> create_colocated_worker_cls) and a Ray actor runs one call at a time, so a
# teacher call issued during generate_sequences would only queue behind it.
# ROLLOUT_PREFETCH_TEACHER_CHUNK (default 128 rows) sets how much is attempted
# per turn; the turn table's tchWait column shows what did not fit, and the
# final turns' rows always remain for the trainer. Frozen teachers, so the
# targets do not depend on when a row is scored, but a row lands in a different
# micro-batch than the post-rollout path would put it in, which moves the last
# bits of a packed GEMM — same class as no_sync_grad_accum, not bit-identical.
#
# NO speculative decoding here, and it is not a tuning choice — it does not run
# on this stack. Setting engine_kwargs.vllm.speculative_config swaps the vLLM V0
# worker for spec_decode.SpecDecodeWorker (wrapping NGramWorker), which does not
# implement sleep(); vllm_rollout_spmd builds the engine with
# enable_sleep_mode=True and calls sleep(level=1) immediately (:210), so the run
# dies in init_workers with "Method 'sleep' is not implemented" before step 1.
# The whole wake/sleep cycle that free_cache_engine=False and
# ROLLOUT_KEEP_VLLM_AWAKE depend on needs that method, so this is structural,
# not a missing argument. The idea itself is sound for the decode tail
# (rejection sampling preserves the sampling distribution exactly, and this arm
# recomputes the teacher KL on the sampled tokens, so vLLM numerics cannot enter
# the loss) — it needs the V1 engine, where spec decode lives in v1/spec_decode
# and sleep is supported. VLLM_USE_V1=1 changes the engine for every phase, so
# that is its own experiment on every arm at once, not a knob to flip here.
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
#
# enable_gradient_checkpointing stays True, and the attempt to drop it is worth
# recording because the reasoning that said it would fit was wrong.
#
# The case for dropping it is real: update_actor is 53.5% of the step at sm
# 94.7% -- work-bound, not gap-bound -- and actor.bwd alone is 200.4s of a 563s
# step. Checkpointing makes backward recompute the forward, ~3 units where
# storing costs 2, so dropping it should take about a third off that backward.
#
# There is no room for it. The card is 94.97 GiB and step 1 peaked at
# max_memory_allocated 93.9 GB -- 99% -- so the ~13 GB of activations it would
# keep (10 rows x ~690 real tokens x 28 layers) OOMed in the first micro-batch,
# on the logsumexp. What made this look affordable was reading
# max_memory_reserved (128.5 GB) as if it were device memory: it is not. vLLM's
# CuMemAllocator maps memory that PyTorch counts in its reserved total but the
# device does not, so reserved can exceed the card. max_memory_allocated is the
# number to read.
#
# Freeing the ~13 GB elsewhere is possible in principle -- gpu_memory_utilization
# is 0.6 of 95 GiB, ~57 GB, and peak KV demand is ~40 GB -- but the margin is
# thin enough that it needs measurement, not arithmetic. Do not try it without
# first logging vLLM's KV usage (VLLM_LOGGING_LEVEL=INFO).
#
# actor.student_indexed_topk=True — NOT a speedup. It changes what the top-k KL
# is computed over, so it belongs to the science, is pinned in
# expected_multitask_config.yaml, and has to be identical across every arm being
# compared.
#   The loss is a coarse-grained reverse KL: exact on a support set A, with
#   everything outside A folded into one tail bucket. Reverse KL weights by the
#   STUDENT's mass, and the error it drops is exactly
#   tail_s * KL(p_s|Ā ‖ p_t|Ā) — a student-mass-weighted term. Taking A from the
#   teacher's top-20 therefore leaves the student's own mass uncovered exactly
#   where the student has drifted off the teacher, which is the regime the term
#   exists to penalise; taking A from the student's top-20 covers it. Both are
#   valid lower bounds on the same full KL (data processing), so this is a
#   tighter bound, not a different objective.
#   It costs no extra forward. The teacher's output splits as
#   log p_t(v) = h·W_t[v] - lse_t, and only the last gather depends on the ids —
#   ~1/42,000 of the teacher forward. So the teacher keeps running inside the
#   rollout's CPU glue where it runs today, caching h and lse; the student's
#   single training forward picks the ids; the teacher is resolved at them for
#   2·H·k. What it does introduce is rank ownership: rows are regrouped by task,
#   padded and reordered by _balance_batch between the two calls, so the rank
#   that cached a row is not the rank that trains it. verl/workers/teacher_cache.py
#   handles that with an all-gather exchange, a per-row answer count that must be
#   exactly 1 (0 = a zero target nobody noticed, 2 = two ranks claiming one key),
#   and a numerical witness that re-derives the teacher's own top-k from the
#   cached h/lse once per step.
#   Requires response_only_logits on both sides (the row map comes from there)
#   and holds one unsharded 622 MB copy of each teacher's output projection for
#   the run — 1.9 GB across the three, laid end to end as one (3*V, H) tensor so a
#   mixed micro-batch needs no grouping by task. It is affordable here and would
#   not be on a larger teacher.
#   It is now the DEFAULT in ppo_trainer.yaml, and passed here anyway so the
#   support the arm trains against is visible at the call site.
#   Three things it deletes, none of which changes a value:
#     - the teacher's own top-k. Under student indexing nothing downstream reads
#       it, so it is built for TEACHER_WITNESS_MICRO_BATCHES (2) micro-batches a
#       step as a spot check and for nothing else. It was a selection over the
#       whole vocabulary plus two scatters per row, and then ~860 MB/step of it
#       travelled to the driver to be ignored.
#     - the second logsumexp. The forward needed the normaliser for the top-k and
#       again for the cache; it is one reduction over the widest tensor in the
#       step, now computed once. topk(sorted=False) for the same reason the order
#       is never read: the KL sums over the support.
#     - the host round-trips in the lookup. It runs inside the micro-batch loop,
#       thousands of times a step, and a .tolist() or an int(tensor) there is a
#       device-to-host sync that drains the CPU run-ahead this whole effort exists
#       to protect. The ownership guard is tallied on the device and read once per
#       mini-batch instead — still before the optimizer step, so an unresolved row
#       cannot reach the weights.
#
# Wasted-work removals (config; all of them delete computation whose result was
# already being thrown away, so none of them changes a value that reaches the
# loss).
#   actor.response_only_logits=True / ref.response_only_logits=True — run lm_head
#     on the response rows only. Everything the actor and teacher forwards return
#     is sliced to [-response_length-1:-1], and prompts are ~3x responses here, so
#     roughly three quarters of the vocab projection was built at (rows, 151936)
#     and dropped — forward and backward, and it is the step's largest activation.
#     The transformer body still runs on every token: a response position attends
#     to the prompt's KV, so that part cannot be skipped. On the teacher this also
#     shrinks the allocation that OOMed at step 136 (section 9.2), because the
#     prefetch runs it while vLLM is awake and holding its KV cache. Same
#     arithmetic, different GEMM shape — the accuracy class of a micro-batch
#     change, so it goes into every arm at once.
#   rollout.return_rollout_log_probs=False — stop asking vLLM for the sampled
#     token's log-prob. Its only consumer is the rollout-vs-actor drift check in
#     RayPPOTrainer.fit, and this arm's thin loop has no old_log_prob phase to
#     compare against, so the column was built (a Python loop over every generated
#     token, every turn) and never read. Sampled tokens are unaffected. Leave it
#     True on arms that run the drift check.
#   rollout.disable_log_stats=False — MEASUREMENT, not a speedup. Turns on vLLM's
#     own statistics (prefill/decode token counts, preemptions, running batch,
#     prefix-cache hits). generate_sequences is opaque to both profilers — the
#     engine is a separate library inside a Ray worker — and it is ~52% of the
#     step, so this is the only way to see inside it. Costs log volume.
#
# ROLLOUT_PREFETCH_TEACHER_ADAPTIVE=1 (default) sizes each teacher chunk from the
# glue window it will run under instead of using one constant. The two are badly
# matched: turns 0-4 hold ~45% of the rollout's CPU glue while the queue is barely
# filled, and from turn 5 the glue collapses to ~0.4s/turn while the backlog peaks
# — which left the glue only ~34% busy at hit_rate 0.99, with 21.6s/step of
# tchWait spill. Set to 0 to go back to the fixed
# ROLLOUT_PREFETCH_TEACHER_CHUNK; that value is still the floor, and
# ROLLOUT_PREFETCH_TEACHER_CHUNK_MAX (512) the ceiling. Chunking only decides how
# many already-final rows are scored per call, so it cannot change a value.

export ALFWORLD_DATA=$HOME/data/alfworld
# NO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here, and it is not a
# tuning choice. vLLM's sleep/wake allocates through CuMemAllocator, which
# asserts on expandable segments outright ("Expandable segments are not
# compatible with memory pool", pytorch#147851) -- the engine refuses to build.
# This arm depends on that mechanism (free_cache_engine=False plus
# ROLLOUT_KEEP_VLLM_AWAKE), so the two cannot coexist. Dropping the actor's
# recomputed activations does allocate and free ~13 GB every micro-batch, which
# is the pattern expandable segments exist for, but the previous run already
# reserved 128.5 GB against 93.9 allocated -- ~35 GB of slack the caching
# allocator can reuse -- so the default is expected to absorb it.
# Traced off for this one line. `set -x` at the top of the file echoes every
# command it runs, expansions included, so with tracing on this writes the real
# key into whatever the run is tee'd to -- in plaintext, for every restart.
{ set +x; } 2>/dev/null
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}
set -x
# On by default, for the same reason the FSDP knobs are literals below: a 300-step
# run gets restarted, and a mechanism that has to be exported by hand is one that
# will eventually be missing from a restart. All four are accuracy-preserving (see
# the header); each still honours an explicit 0 from the caller.
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export ENV_RESET_PREFETCH=${ENV_RESET_PREFETCH:-1}
export TASK_BALANCE_INTERLEAVE=${TASK_BALANCE_INTERLEAVE:-1}
export ROLLOUT_PREFETCH_TEACHER=${ROLLOUT_PREFETCH_TEACHER:-1}
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
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
    actor_rollout_ref.actor.response_only_logits=True \
    actor_rollout_ref.actor.student_indexed_topk=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.return_rollout_log_probs=False \
    actor_rollout_ref.rollout.disable_log_stats=False \
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
