set -x

# Pure OPD multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH CROSS-TEACHER SIGN WEIGHTING IN TARGET MODE.

# ---------------------------------------------------------------------------
# CROSS-TEACHER SIGN WEIGHTING -- TARGET MODE.
#
# Identical to run_multitask_qwen3.sh except for algorithm.opd.sign_weight.* and
# the run's own identity, so the two differ in the weighting of the teacher KL
# and in nothing else -- same teachers, same data, same batch sizes, same eval
# protocol. That is what makes the plain arm the control for this one.
#
# What the weighting does (verl/trainer/ppo/sign_weights.py): each teacher is a
# single-task RL fine-tune of one base policy, so sign(log pi_m - log pi_0) says
# whether task m's RL raised or lowered a candidate token here. The support is
# the STUDENT's top-k -- the same one the KL already uses -- and all four models
# (on-task teacher, two off-task teachers, base) are read on it.
#
# The two modes DO NOT share a weight table, and that is the point:
#
#   position  one scalar per token multiplies the per-token KL. A KL term has no
#             direction, so agreement counts the same whether the teachers agreed
#             to raise a token or to lower it: (+,+) and (-,-) both get 1.25. The
#             minimiser is still the on-task teacher, so this changes how hard
#             each position is learned and never what the student converges to.
#   target    the teacher's own probability at that candidate is reweighted and
#             the distribution renormalised, moving the fixed point to
#             ~ w(v) p_teacher(v). Here the weight multiplies a PROBABILITY, so
#             direction is the whole content: (+,+) gets 1.25 and (-,-) gets
#             0.75. This is the variant that can inject an off-task opinion --
#             and the one that can inject a wrong one.
#
# Conflict is NOT weighted in either arm (disagree_weight=1.0). Target mode
# refuses any other value rather than pick a direction silently: deferring to the
# objecting teachers means lowering a token the on-task teacher raised and
# RAISING one it lowered, and one factor below 1.0 does the second backwards.
#
# Cost: three extra frozen forwards per step (the base over all rows, each
# teacher over the 2/3 of rows that are not its own task), and the hidden-state
# cache holds four models per row instead of one. enable=False reproduces the
# plain arm exactly: no base worker is built, no extra forward runs, and nothing
# is written into the batch.
#
# CHECKPOINT DIRECTORY. Arm-specific, and it has to be: the path is
# default_local_dir/global_step_N and trainer.resume_mode defaults to "auto", so
# two arms sharing this directory do not merely overwrite each other's
# checkpoints -- the second one silently RESUMES FROM THE FIRST.
# ---------------------------------------------------------------------------
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
# INTENT LOCK: examples/opd_trainer/expected_multitask_signweight_target_config.yaml pins the
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
#   ref.log_prob_micro_batch_size_per_gpu is 4, and it is a MEMORY bound rather
#   than a throughput choice. compute_ref_topk_log_prob runs the teacher through
#   lm_head, which materializes the full vocabulary over the rows it projects,
#   and that tensor is the widest thing either GPU allocates.
#
#   It has now OOMed twice, and the second time is the reason for 4. It was 16,
#   justified by ref.response_only_logits: the projection runs on the response
#   rows only, and at step 1 the responses were 20.5% of the tokens
#   (139.1 of 678.9), so 16 rows cost about a fifth of the full-sequence figure.
#   That justification decayed as the student trained. By step 18 the mean
#   response was 257.0 (+85%) with 22.2% of rows at the 512 cap -- alfworld 276.4
#   and 31.3% -- and a chunk of long rows put 16 * ~417 response tokens through
#   lm_head: 6.7k * 151936 * fp32 = 3.77 GiB for the logits, and logsumexp wants
#   a second buffer the same size. It asked for that 3.77 GiB with 3.49 GiB free.
#
#   4 makes the bound structural instead of empirical. data.max_response_length
#   is 512 and is pinned, so 4 rows can never project more than 2,048 response
#   tokens = 1.16 GiB, and the pair with the logsumexp temp is ~2.3 GiB. No
#   further response-length growth can break it, which matters because the
#   growth has not stopped -- clip_ratio went 0.010 -> 0.222 in 18 steps.
#
#   It costs 2.0% of throughput, and that number is measured rather than
#   estimated: over the same 18 steps of the same data, 4565 -> 4474 tokens/s.
#   The whole of it lands in one column of the turn table -- tchWait went
#   15.6 -> 26.4 s/step while preproc/gen/decode/envstep and update_actor all
#   stayed put, and teacher_prefetch/hit_rate did not move (0.978 -> 0.974). So
#   the same rows are still scored inside the rollout; scoring them just got
#   slower, and the glue window they hide in (~48 s) was already full.
#
#   Do NOT read timing_s/teacher_forward to price this knob. That column moved
#   by 0.7 s, because since the prefetch landed it only holds the rows the glue
#   could not cover -- most of the teacher's cost is inside gen now. This
#   comment used to say the knob "costs almost nothing" on exactly that basis.
#   docs/speedup_mechanisms.md section 7 has the full table, and section 7.4 the
#   way to get the 2% back (a token-based bound via ref.log_prob_use_dynamic_bsz,
#   which needs all three arms changed together).
#
#   What it does NOT cost: any change to a value. The teachers are SHARD_GRAD_OP,
#   so they do not reshard between micro-batches and smaller ones add no
#   all-gather. And the ref micro batch does NOT enter adjust_batch's lcm here
#   (size_divisor_ref falls back to the rollout value unless use_kl_in_reward or
#   actor.use_kl_loss is set, and this arm pins both False), so changing it moves
#   no padding and no data -- per-row log probs are independent under rmpad.
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
#
# WHICH TOKENS (algorithm.opd.sign_weight.token_stats). Everything else this arm
# reports about the weighting has the vocabulary summed out: frac_agree_pos says
# a fifth of candidates were reinforced and cannot say whether that is the same
# twenty tokens every step or a different thousand. Those are different
# mechanisms with identical summaries, and which one this is decides what a gain
# would mean -- "the tasks share a small stable set of moves" or "the tasks share
# a broad statistical tendency".
#
# On: the shape metrics (sign_weight/*/token/n_distinct, top64_share, and the two
# halves of Z-1 separately) join the usual scalars, and the ranked table itself is
# written per step under trainer.sign_token_dump_dir -- top tokens by how OFTEN
# the teachers agreed about them, and separately by how much they actually moved
# the target (the signed (w-1)*p, which sums over the vocabulary to Z-1). A token
# can top one list and be absent from the other, and that gap is the finding.
#
# It changes no value. The accumulation is read-only, runs beside the stats the
# arm already collects, and the weights the loss sees are untouched. What it costs
# is a dense (4 x 7 x 151,936) accumulator -- 34 MB next to a teacher output
# projection of 622 MB -- one all-reduce of that, and one device-to-host read, all
# once per update_actor rather than per micro-batch. Set enable=False to drop it.
#
# It is NOT in the intent lock, for the same reason the profiler is not: pinning
# it would make a measurement a precondition of the experiment.
#
# DID ANYTHING TRANSFER (algorithm.opd.sign_weight.{transfer_stats,pair_stats}).
# The arm exists to test whether one task's teacher can reach the student on
# another task's states, and until now nothing measured that: every number it
# reported describes the TEACHERS' agreement structure or the SIZE of the
# rewrite. Three families answer the three questions a write-up has to settle.
#
#   Is transfer happening?  transfer/off_travel/{dst}__vs__{src} is where the
#     student sits between "has not moved" (0, its position at step 0 by
#     construction -- the lock pins model.path == sign_weight.base_path) and
#     "as far toward src as its own teacher already is" (1). Both anchors are
#     exact whatever src is and however far it drifted, which matters because
#     the teachers' KL coefficients differ 10x and their measured drift 3.7x.
#     Read it against on_travel = 1 - teacher_kl_now/teacher_kl_step0: a student
#     that simply has not converged passes "closer than its teacher" for free.
#     Beside it, sign_weight/*/cf_cost is the rewrite measured at the STUDENT's
#     own distribution rather than the teacher's, exactly
#     KL(p_s||p~) - KL(p_s||p), with rewrite_progress = -1 at init and
#     rewrite_fisher carrying none of the teachers' coefficients.
#
#   Common knowledge?  sign_weight/pair/lor/{src}__on__{dst} is the Haldane-
#     corrected log odds ratio of the two teachers' signs. It is the headline
#     rather than agree_rate because it divides out each teacher's own
#     propensity to raise rather than lower, which agree_rate confounds with
#     association -- and with drift differing 3.7x that confound is not
#     hypothetical. Quote agree_mass only beside agree_pop_mass and agree_ess:
#     64% of teacher mass sits on 4.2% of candidates, so a mass-weighted rate
#     without its effective sample size cannot be told from step noise.
#
#   Whose vocabulary?  pair_stats.tokens adds the axis the two tables above
#     both sum out: SignPairCounts has the sender and not the token,
#     token_stats has the token and not the sender. Neither can say whether the
#     tokens one off-task teacher pushes into a task's states are the same ones
#     the other pushes there -- which is the difference between "the tasks share
#     a common surface vocabulary" and "each teacher contributes its own", and
#     that difference is the whole content of a transfer claim.
#     sign_weight/pair/token_overlap/{cls}/{a}__and__{b}__on__{dst} is the
#     weighted Jaccard between the two senders' token vectors: near 1 no
#     individual sender is necessary, near 0 the unanimity gate is passing on
#     the intersection of two different vocabularies. The ranked rows go to
#     sign_pair_tokens_step*.jsonl beside the per-state table.
#     Cost: T*(T-1)*3*V cells = 55 MB at T=3, i.e. LESS than the per-state table
#     already running -- the naive (dst, src, sign_on, sign_src, V) layout would
#     be 12.3M cells, and it is affordable only because the nine sign
#     combinations collapse to the three that carry an opinion from the sender
#     (agree / conflict / blindspot; a silent sender has nothing to file) and
#     the structurally empty src == dst diagonal is not allocated.
#
#   What KIND of token, and in what sentence?  event_dump writes individual
#     candidates to sign_events_step*.jsonl -- the four models' probabilities at
#     that candidate, the weight, the effect, the turn and position in the
#     episode, the row's episode score, and the decoded text around it. Every
#     other table here is an aggregate, and an aggregate cannot be read for a
#     mechanism: "the weighting acts on the same forty tokens every step" and
#     "it acts on the connectives inside <think>" produce the same top-N list.
#     Two strata per step, because either alone misleads: `top` is the largest
#     |effect| (where the mechanism is loudest, and the natural thing to quote)
#     and `spread` is a hash-ordered pseudo-random sample (what the MEDIAN event
#     looks like, which is what says whether the extremes are representative).
#     Each row carries a role -- reasoning / env_action / tool_call / env_obs /
#     tag / format -- read off the <think>, <action>, <search>, <answer> and
#     <information> spans the prompts define. A weight that fires almost
#     entirely inside <think> is a claim about reasoning style; the same number
#     concentrated inside <action> is a claim about which moves the tasks share,
#     and no scalar in this run can tell those apart.
#     RANK-0 LOCAL, unlike everything else here: a sum can be all-reduced and a
#     sample cannot, so the file holds a sample of one rank's shard. Unbiased,
#     but world_size times smaller than it looks.
#     Cost: per_step * 2 rows of about 40 numbers, one host read per
#     update_actor. context=16 tokens either side.
#
#   Specialist knowledge?  The whole specialist population is inside the
#     on-task-silent state, 64% of teacher mass, never decomposed.
#     sign_weight/blindspot/* is where src has an opinion and dst's own teacher
#     does not -- knowledge the distillation target structurally cannot carry.
#     ALWAYS read off_opinion_h1_frac beside it: at p_0 > exp(-deadzone) no
#     model CAN raise a token past the deadzone, so silence there is arithmetic
#     and not ignorance. sign_weight/gate/* splits what the unanimity rule
#     throws away into silence and dissent, which is what says whether a gate
#     redesign has a target.
#
# None of it changes a value. The accumulators read tensors the sign-weight pass
# already computed -- the student, its own teacher, the base and the off-task
# teachers are all resident on one support -- so there is no extra forward. Cost
# is a few kB of counters, two small all-reduces and one host read per
# update_actor, against a 146.7 s/step sign_weight_forward.
#
# NOT in the intent lock, for the reason the profiler is not: a diagnostic that
# can be switched on mid-run must not become part of what the experiment IS.
# The one exception is algorithm.opd.sign_weight.measure_only, which suppresses
# the rewrite and therefore DOES change the loss -- an arm using it is a
# different arm and needs its own lock, experiment_name and directories.
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
# There is no room for it. The ~13 GB of activations it would keep (10 rows x
# ~690 real tokens x 28 layers) OOMed in the first micro-batch of step 1, on the
# logsumexp, against a 94.97 GiB card. That measurement is the evidence; what
# made it look affordable beforehand was reading max_memory_reserved (128.5) as
# if it were device memory.
#
# Neither perf/max_memory_* metric is a device-memory reading here, and the
# reserved one is not the only offender: both are torch.cuda high-water marks
# divided by 1024^3 (fsdp_workers.py:719, so the _gb suffix is really GiB), and
# BOTH have printed numbers larger than the 94.97 GiB card -- reserved 144.588
# and allocated 97.659 at step 18. vLLM's CuMemAllocator is a pluggable allocator,
# so its pool lands in torch's accounting, while the physical pages behind it are
# mapped and unmapped on wake/sleep underneath. The high-water mark can therefore
# span memory that was never simultaneously resident. Read them as trends, not
# capacities: for an actual occupancy number use nvidia-smi, or the accounting in
# an OOM message ("this process has 91.38 GiB memory in use"), which is the
# allocator's own view at the moment it failed.
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
# ROLLOUT_KEEP_VLLM_AWAKE), so the two cannot coexist. Nothing about the
# gradient-checkpointing revert changes that: the assert fires in init_workers,
# before any activation is allocated.
#
# The fragmentation argument that originally justified the export is also gone.
# It rested on the actor allocating and freeing ~13 GB of activations every
# micro-batch, which only happened with checkpointing off -- it is back on, so
# that traffic does not exist. The second half of that argument was wrong on its
# own terms as well: it read "reserved 128.5 GB against 93.9 allocated" as ~35 GB
# of reusable slack. Reserved is not device memory here (the card is 94.97 GiB,
# less than the reserved figure), so there was no such slack to count.
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
    +trainer.expected_config=examples/opd_trainer/expected_multitask_signweight_target_config.yaml \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
    +algorithm.opd.sign_weight.enable=True \
    +algorithm.opd.sign_weight.mode=target \
    +algorithm.opd.sign_weight.agree_weight=1.25 \
    +algorithm.opd.sign_weight.agree_neg_weight=0.75 \
    +algorithm.opd.sign_weight.disagree_weight=1.0 \
    +algorithm.opd.sign_weight.deadzone=0.1 \
    +algorithm.opd.sign_weight.base_path=Qwen/Qwen3-1.7B \
    +algorithm.opd.sign_weight.token_stats.enable=True \
    +algorithm.opd.sign_weight.token_stats.top_n=64 \
    +algorithm.opd.sign_weight.pair_stats.enable=True \
    +algorithm.opd.sign_weight.pair_stats.tokens=True \
    +algorithm.opd.sign_weight.event_dump.enable=True \
    +algorithm.opd.sign_weight.event_dump.per_step=128 \
    +algorithm.opd.sign_weight.event_dump.context=16 \
    +algorithm.opd.sign_weight.transfer_stats.enable=True \
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
    trainer.project_name='verl_agent_opd_signweight_multitask' \
    trainer.experiment_name=opd_multitask_signweight_target_qwen3_1.7b \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/verl_agent_opd_signweight_target_multitask \
    trainer.val_instance_log_dir=$HOME/val_instances/opd_multitask_signweight_target_qwen3_1.7b \
    trainer.sign_token_dump_dir=$HOME/sign_tokens/opd_multitask_signweight_target_qwen3_1.7b \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
