set -x

# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH PARAMETER-FREE CROSS-TEACHER KL WEIGHTING.

# ---------------------------------------------------------------------------
# CROSS-TEACHER KL WEIGHTING -- PARAMETER-FREE.
#
# Identical to opd_grpo_trainer/run_multitask_qwen3.sh except for
# algorithm.opd.cross_teacher_kl_weight.*, the OPD coefficient, and the run's own
# identity. See below on the coefficient: it is 0.01 here against 1.0 on every
# earlier arm, which is why the control beside this script exists and why the
# 150-step sign-weight results are NOT a baseline for it.
#
# What it does (verl/trainer/ppo/cross_teacher_kl_weight.py). Every teacher is a
# single-task RL fine-tune of one shared base, so delta_m = log pi_m - log pi_0
# is what task m's RL wrote into the model at a candidate. The arm turns that
# into ONE positive scalar W per token position, multiplies the per-token
# teacher KL by it, and does nothing else:
#
#     L = L_GRPO + 0.01 * Agg( W * KL(student || on-task teacher) )
#
# The teacher distribution is untouched, so W cannot move what the student
# converges to -- at a position weighted by W every student logit's gradient is
# the unweighted one times W, direction included. What it reallocates is EFFORT:
# which positions inside a task get more of that task's OPD budget.
#
# THE POINT IS THAT IT HAS NO TABLE. The sign arm spends the same signal through
# agree_weight=1.25 / deadzone=0.1, so "does cross-teacher structure help" cannot
# be separated from "was 1.25 right". Here every scale is measured:
#
#   delta_hat  each teacher's shift divided by ITS OWN in-domain RMS, cumulative
#              over the run. The teachers trained at KL coefficients differing
#              10x and drifted 3.7x, so raw nats are not on a common footing.
#              The divisor is the DIAGONAL of the (destination, teacher) matrix
#              -- teacher m measured on task m's own states -- because dividing
#              by the destination-conditioned RMS would stretch the noise of a
#              teacher that barely moves out of domain up to a full unit, which
#              is what the deadzone used to suppress. The off-diagonal is still
#              recorded, as off_to_in_domain_ratio.
#   c          the minimum EVERY teacher in the run guarantees, the on-task one
#              included. A continuous min, not a sign test, so a near-zero shift
#              contributes near zero and no deadzone is needed.
#   alpha      per ordered task pair, the correlation between that source's
#              RESIDUAL support for the tokens the student actually emitted and
#              the GRPO advantage of the trajectory it emitted them in.
#              Rectified at zero: an anti-correlated source is vetoed, never
#              inverted -- nothing says a reversed policy shift points anywhere
#              useful. This is the ONE thing GRPO contributes to the mechanism.
#   e          |c| + sum_m alpha_m |delta_hat_m|, per candidate.
#   W~         1 + sum_v p_teacher(v) e(v), per position.
#   W          W~ / mu_d, the PREVIOUS step's KL-weighted per-task mean.
#
# The normaliser is what keeps this a redistribution. W~ is at least 1
# everywhere, so applying it raw is indistinguishable from a larger
# teacher_kl_loss_coef. It is KL-WEIGHTED because what a task's budget IS is
# sum(W*D), not sum(W), and those agree only when the weight and the KL are
# uncorrelated -- exactly what this arm bets is false. On the snapshot it was
# built from, sum(W*D)/sum(D) == 1 by construction. Read kl_scale, not w_mean:
# w_mean is NOT 1 here and does not need to be.
#
# WHAT IT DOES NOT DO. A position scalar cannot change the KL's gradient
# direction, so this arm does not inject an off-task teacher's tokens, does not
# push the student toward an off-task teacher where the teachers disagree, and
# never calls a teacher-only token reward-verified. The minimiser is the on-task
# teacher, always. Nor does it weight the individual vocabulary terms of the KL:
# that product is not a divergence and, on a reverse KL, raising a term's weight
# pushes the token DOWN whenever log(p_student/p_teacher) > -1.
#
# Corroboration requires EVERY teacher, the on-task one included in both the
# unanimity test and the minimum. Off-task unanimity alone was tried and is not
# what this arm runs: it credits a position where the other two tasks decisively
# OPPOSE the on-task teacher exactly as it credits one where they agree, and the
# KL that then gets strengthened points at the very opinion they contradicted,
# with no advantage-side evidence behind it -- at alpha=0 a (-1,+3,+2) candidate
# would score 2 instead of 0. The price is real and is measured rather than
# assumed: c is capped by the on-task shift and that teacher is silent at ~64%
# of teacher mass, so evidence/shared_offtask_only_* reports what the other rule
# WOULD have contributed. Nothing in the loss reads it.
#
# Cost: three extra frozen forwards per step (the base over all rows, each
# teacher over the 2/3 of rows that are not its own task), the same the sign arm
# pays -- about 146.7 s on a 618 s step by the 150-step report's section 9.
# enable=False reproduces the plain arm exactly, which is what the control run
# beside this script is.
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
#   bash examples/opd_grpo_trainer/run_multitask_qwen3.sh trainer.n_gpus_per_node=3
# Teacher and checkpoint paths are written relative to $HOME so the same script
# resolves correctly on either machine.
#
# INTENT LOCK: examples/opd_grpo_trainer/expected_multitask_cross_teacher_klw_control_config.yaml pins the
# scientific knobs (loss type/coefs, seeds, batch sizes, teachers, eval
# protocol). main_opd_grpo validates the composed config against it after its own
# injection and refuses to start on any mismatch. To change such a knob, edit
# the argument below AND the expectations file in the same commit.
#
# Loss: policy_loss = pg_loss * pg_loss_coef + teacher_kl_loss * kl_loss_coef.
#   THE OPD COEFFICIENT IS 0.01 HERE, against 1.0 on every earlier arm. That is
#   a 100x change to the teacher-KL term and it is deliberate: this arm
#   redistributes that term, and at coefficient 1.0 the redistribution competes
#   with the policy gradient rather than shaping it. The consequence is that the
#   150-step sign-weight results and every existing lock are NOT a baseline for
#   this run -- they differ in the coefficient before they differ in the
#   mechanism. That is what
#   examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh
#   is: the same code path, the same seed, the same rollouts, the same 0.01, and
#   cross_teacher_kl_weight.enable=False. It is not an ablation. Without it the
#   arm has no comparison at all.
#
#   The GRPO policy gradient and the per-task teacher KL are both in the loss;
#   this is the ONE thing that differs from the pure-OPD arm, which is the same
#   recipe with pg_loss_coef=0. main_opd_grpo shares its config injection and its
#   whole composition with main_opd (inject_distillation_config / build_and_fit),
#   so the teacher side cannot drift between the two.
#   - actor.pg_loss_coef=1.0 keeps the GRPO policy gradient (env-reward
#     group-relative advantages; algorithm.adv_estimator=grpo). Set it to 0 and
#     this recipe IS the pure-OPD one. It is not injected from algorithm.opd.*
#     and it has no default -- main_opd_grpo asserts it is set, because it is the
#     ratio between the two terms and a default here would train fine while
#     answering a different question.
#   - use_kl_loss=False / entropy_coeff=0 keep the other terms off, so the loss
#     has exactly the two terms named above; use_teacher_kl_loss is
#     force-injected. algorithm.use_kl_in_reward=False keeps the reference KL out
#     of the REWARD as well, so the only thing shaping the advantages is the env
#     score.
#   - algorithm.opd.normalize_loss_by_task weights the policy gradient and the
#     teacher KL by the SAME per-task row weights (see
#     check_task_weighting_supported). That is what keeps pg_loss_coef meaning the
#     ratio between them: weighting one term and token-meaning the other would
#     leave the coefficient describing a proportion the loss does not have.
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
#   rollout.log_prob_micro_batch_size_per_gpu is 10, and on THIS arm it is two
#   things at once. Unlike pure OPD, compute_log_prob does run here (the policy
#   gradient needs old_log_prob), so the value sizes a real forward pass. It also
#   still sets adjust_batch's divisor, lcm(rollout*2, rollout*2, actor*2): at 16
#   that was lcm(32,32,20)=160 and step 1 discarded 116 rows to reach it; at 10 it
#   is lcm(20,20,20)=20, about 10. That second effect decides which rows are
#   dropped, i.e. the data, which is why it is pinned even though the first effect
#   is pure throughput. Keep it equal to the pure-OPD arm's value, or the two arms
#   train on different rows and the comparison is not one.
#
#   THAT FORWARD IS UNBOUNDED IN THE WAY THE TEACHER'S ONCE WAS, and the number
#   is written down here rather than acted on, because acting on it costs the
#   comparison. compute_log_prob goes through _forward_micro_batch, so
#   response_only_logits applies -- but 10 rows x the pinned 512
#   data.max_response_length is 5,120 response tokens through lm_head:
#   5,120 * 151,936 * fp32 = 2.90 GiB for the logits, and logsumexp wants a
#   second buffer the same size, so ~5.8 GiB structurally. For scale, the
#   teacher's ref.log_prob_micro_batch_size_per_gpu was cut 16 -> 4 after OOMing
#   twice (the second time asking 3.77 GiB with 3.49 GiB free); 4 bounds that
#   path at 2,048 tokens = 1.16 GiB, ~2.3 GiB with the temp. This phase's bound
#   is 2.5x that, in the same regime -- ROLLOUT_KEEP_VLLM_AWAKE=1 means the
#   engine is awake and holding its KV cache here too.
#
#   Not pre-emptively lowered: this key also sets adjust_batch's divisor, so
#   lowering it changes which rows are dropped and the arm stops training on the
#   same rows as its pure-OPD control -- and being able to make that comparison
#   is the whole point of the arm. Left to measurement. IF IT DOES OOM, the fix
#   that keeps the divisor is rollout.log_prob_use_dynamic_bsz=True (it
#   interpolates from actor.use_dynamic_bsz, pinned False, so it is off today)
#   together with rollout.log_prob_max_token_len_per_gpu, which is already passed
#   below at 18432 and is inert while dynamic bsz is off. That bounds the phase by
#   TOKENS without touching the divisor, and the pure-OPD control never runs this
#   phase at all, so it cannot desynchronise the two arms.
#
#   Do not price any of this from perf/max_memory_*. Both are torch high-water
#   marks that span vLLM's allocator pool -- the teacher-indexed arm's 150-step
#   report logs 115.8 / 145.2 GB against a 94.97 GiB card
#   (docs/multitask_signweight_teachertopk_150step_report.md section 9) -- so no
#   headroom can be derived from them. Use nvidia-smi, or the accounting in an
#   OOM message.
#
#   Throughput: expect a new timing_s/old_log_prob column that the pure-OPD arm
#   does not have. By that report's section 9, three frozen-model forwards over
#   the batch cost 146.7 s, so one is ~49 s -- roughly +8% on its 618 s step.
#   That is an order of magnitude, not a prediction: those run at the ref micro
#   batch of 4, this one at 10, and this one also computes entropy.
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
# WHAT THE WEIGHTING DID (kl_weight/*). Not configurable and not in the lock:
# it is the only readout of this arm's mechanism, so it is always on.
#
#   kl_scale = sum(W*D)/sum(D) is the invariant, and the number to read first.
#     Exactly 1 on the snapshot the normaliser came from, so what it shows on a
#     live batch is the one-step lag plus the step's own drift. A persistent
#     departure means the arm bought distillation strength without saying so,
#     which is the thing the normaliser exists to prevent. w_mean is NOT 1 and
#     is a diagnostic of the W-D correlation, nothing more.
#   kl_shift_gross_frac is the fraction of the OPD term the arm moved between
#     positions -- the go/no-go number (see below). Dimensionless, so a
#     threshold on it means the same thing at any teacher_kl_loss_coef.
#   kl_shift_by_state/{state}/{net,gross_share} is THE analysis reading. The
#     weight scales the whole KL toward the on-task teacher, so the only thing
#     the mechanism can do is decide which STATES get more of it, and
#     (W-1)*D splits exactly into a per-candidate part and the normaliser's own
#     offset -- the seven states plus shift_norm_offset sum to the total with no
#     residual. Watch neutral_off_task_silent: the on-task teacher is the only
#     one with an opinion there, which is its own task-specific knowledge, and
#     the evidence is zero by construction, so that is where the arm TAKES
#     budget from. A gain and a loss both have to be read against this table or
#     neither can be attributed.
#   evidence/shared_share says which channel is carrying the mechanism. Near 1
#     the corroboration bonus IS the arm; near 0 it is decoration and what the
#     run tested is "reliable source activity", not agreement.
#   rms/{src}__on__{dst}/off_to_in_domain_ratio is the direct measurement of how
#     much less a teacher moves out of its own domain -- the signal the DIAGONAL
#     divisor exists to preserve, and which a destination-conditioned divisor
#     would force to 1 by construction.
#   adv/{src}__on__{dst}/alpha_applied, with rho, rho_lcb95 and the two partials
#     beside it. Only alpha reaches the loss. max(0, rho) carries a small
#     positive bias under the null -- roughly 0.4/sqrt(N) -- so while rho_lcb95
#     straddles zero a positive alpha is not distinguishable from that bias, and
#     that is exactly the cold-start window where the student is most plastic.
#     rho_length_controlled and rho_length_on_controlled are reported and never
#     applied: a confidence level is a knob, and so is a choice of confound.
#   adv/frac_sampled_outside_topk is the coverage of one approximation: the
#     support score's baseline runs over the top-k with a zero tail residual, so
#     an emitted token outside the top-k is not in its own baseline. The KL's
#     support is deliberately NOT widened to fix that -- moving the loss to tidy
#     a diagnostic is the wrong trade.
#   probe/alpha{000,010,100}/* run the same arithmetic at fixed alphas through
#     the same normaliser and never touch the loss. They exist because at
#     alpha=0 the corroboration channel is alone and it is structurally the
#     minority one, so a gate read there is a lower bound, not the mechanism.
#
#   token_stats / event_dump name the tokens. Without them the run can say a
#     source raised a task's KL by so many nats and cannot point at one token it
#     did it at, which is most of what a write-up needs. Three files land under
#     trainer.sign_token_dump_dir:
#       sign_tokens_step*.jsonl       per (scope, state, token): how often, how
#         much on-task teacher mass, and the nats the weighting moved there.
#       sign_pair_tokens_step*.jsonl  per (destination, source, class, token) --
#         WHICH tokens each off-task teacher's evidence acted on in each other
#         task's states, with token_overlap saying whether the two sources name
#         the same ones. This is the "what did Search bring to AlfWorld" table.
#       sign_events_step*.jsonl       individual candidates with the four models'
#         values, the weight, the turn and position, the row's score, and the
#         decoded text around them, in a top-|effect| and a hash-ordered sample.
#     The state labels and the deadzone in these are in RMS UNITS, not nats:
#     they are computed on the standardized shifts against a zero base, which is
#     what makes report_epsilon comparable across teachers. It reaches no weight.
#     What a row can be read as: "Search-derived token evidence strengthened the
#     whole KL toward AlfWorld's own teacher at this position". NOT "a Search
#     token was injected into AlfWorld" -- a position scalar cannot do that.
#
#   opd_attribution/* IS THE ARM-INDEPENDENT HALF, and it is the only family in
#     this script that also runs in the control. Everything above it needs the
#     base policy and the off-task teachers, which only
#     cross_teacher_kl_weight.enable=True makes the driver load: a control run
#     has no sign_cache_ids column at all, so its corroboration, its pair
#     evidence and its channel partition are not weight-free quantities that
#     happen to be missing -- their INPUTS do not exist, and computing them would
#     mean the control paying for the three extra frozen forwards it exists to do
#     without. What a control does have is the student's own top-20, the on-task
#     teacher's log-probs at it, and the policy-gradient coefficient, which is
#     exactly what opd_logit_push reads.
#     Under opd/ it publishes, in BOTH runs and under the same keys:
#       opd/push/base_*        the per-token push of the UNWEIGHTED OPD term,
#         Sum g0 and Sum |g0|, with the support/tail coverage share the ranking
#         is quoted against. "Which tokens does distillation push hardest, before
#         any weighting" -- and its top-N ranking lands in the same
#         sign_tokens_step*.jsonl file under ranked_by="base_logit_push".
#       opd/push/kl_mass_*     each candidate's own share of the position's D,
#         p_student(u) * (log p_student - log p_on). Nats of evidence, not of
#         push, and signed: only the position's whole D is bounded below.
#         ranked_by="kl_mass" in the same file.
#       opd/grpo/grad_{cosine,norm_ratio}   the unweighted OPD gradient against
#         the policy gradient. Beside kl_weight/grpo/grad_cosine on the weighted
#         arm this is the weighting's effect on the gradient geometry with the
#         POLICY HELD FIXED, which is a stronger comparison than the same two
#         numbers taken from two runs whose policies have diverged by the step
#         you read them.
#       opd/{task}/{kl,push_abs}_{mean,share} and the same four columns cut by
#         role -- where the unweighted budget sits, which is the denominator
#         every claim about where the WEIGHT moved budget is quoted against.
#     It runs in the weighted arm too, deliberately: there the columns are the
#     counterfactual on the SAME policy. The weighted push table then withholds
#     them (LogitPushTokens.scalar_metrics(weight_free=False)) so the series has
#     exactly one owner and one key in both runs.
#     Cost: one ~194 MB vocabulary-wide buffer on the stride step (every=5, the
#     same discipline as token_stats) and two float64 rows every step. No extra
#     forward, no extra backward -- g0 is closed-form on tensors already resident.
#
#   grpo/grad_norm_ratio and grpo/grad_cosine are analytic, in logit space, over
#     the top-k plus the tail: the descent direction for the weighted OPD term
#     is coef*W*p_s*(D - f) and for the policy gradient A*(1[v=y] - p_s), both
#     exact on tensors already resident. No second backward, so the diagnostic
#     cannot perturb the update it describes -- and both forms are checked
#     against autograd in the tests. The ratio is how much of the update the OPD
#     term is responsible for at all, which at coefficient 0.01 is the first
#     thing a reader wants and the last thing a loss curve shows. A persistently
#     negative cosine means the weighting is spending its budget against the
#     reward signal, which is a finding rather than a bug. Collected on the first
#     PPO epoch, where the ratio is 1 and the closed form is an identity.
#
#   THE ANALYSIS CUTS (added for ablations; all curated, none change the loss).
#   Every one of them is a re-scoping of columns the arm already accumulates, so
#   a role row and a task row of the same metric are the same arithmetic over
#   different positions.
#
#   kl_weight/role/{format,reasoning,env_action,tool_call,env_obs,tag}/*
#     WHERE IN THE TEXT the arm acts. "The arm moved 3% of the OPD budget" is
#     the same number whether it moved it into the tokens the environment
#     executes or into whitespace, and those are not the same finding. Roles
#     come from the tag scan in sign_weights.token_roles -- <think>, <action>,
#     <search>/<answer>, <information>, the tags themselves -- so this is a
#     property of the POSITION, not of the row. Six series each:
#       effect/kl_unweighted        where that role's OPD budget IS
#       effect/kl_shift_gross_frac  how much of it the arm moved there
#       effect/kl_shift_net         and in which direction
#       position/w_mean
#       evidence/shared_share       which channel carried it there
#       grpo/grad_cosine            does the budget moved there pull WITH the
#         reward gradient. A pooled cosine of 0.1 is consistent with strong
#         agreement inside <action> and strong disagreement in the scaffolding;
#         this is what separates them, and it is the causal reading of the arm.
#     Plus effect/kl_shift_by_state/*/gross_share per role -- the state
#     composition inside each kind of text, shares only.
#
#   kl_weight/turn/turn{0..4,5plus}/*  WHERE IN THE EPISODE. Four series: is the
#     arm front-loaded, or does it act on the late turns where the reward is
#     decided.
#
#   kl_weight/shape/*  the DISTRIBUTION of W, pooled and per task. w_cv cannot
#     tell "1.02 nearly everywhere" from "1.00 at 99% of positions and 3.0 at
#     the rest", and those are the two mechanisms the arm is being tested for --
#     the first is a larger teacher_kl_loss_coef in disguise. Quantiles
#     w_q{50,90,99} (interpolated inside a bucket; the top bucket is unbounded
#     and saturates at 5.0), then exact whole-bucket shares at 1.05 / 1.25 / 2.0:
#     frac_w_gt_* (how many positions), kl_share_w_gt_* (how much of the OPD
#     budget sits up there) and shift_share_w_gt_* (how much of the moved nats
#     the tail carries). frac_w_below_one is the only direction W~ cannot reach
#     on its own -- below 1 is the normaliser taking budget AWAY, which is what
#     separates "the arm added distillation" from "the arm moved it".
#
#   kl_weight/*/token/turnover/{top64_jaccard,effect_carryover}  the one reading
#     that needs two steps. n_distinct and top64_share are both within-step, so
#     forty tokens replaced wholesale every step reads exactly like a stable
#     forty in both. Jaccard is membership; effect_carryover is the share of
#     THIS step's nats that landed on tokens the PREVIOUS step had ranked. High
#     carryover with a low Jaccard is a stable core with a churning tail.
#
#   kl_weight/role/*/token/*  and the role: rows in sign_tokens_step*.jsonl --
#     the vocabulary cut by what was being written. "The arm reinforced go and
#     take inside <action>" and "... inside <think>" are the same row in every
#     other table here. token_stats.roles=False turns off the 22 MB table.
#
#   kl_weight/probe/alpha*/effect/kl_shift_by_state/*/gross_share  the alpha
#     series' own state composition. Until now the probes reported only how BIG
#     the counterfactual weight would be; this says whether a larger alpha moves
#     budget TOWARD the corroborated positions or just scales everything, which
#     is the question the series exists to answer. Each probe uses its own mu
#     and its own evidence, so its columns sum to that probe's (W-1)D and not to
#     the shipped arm's.
#
#   THE THREE ANALYSIS CORRECTIONS (2026-08). Each was a diagnostic reporting
#   something other than what its name said, and none touched the loss.
#
#   grpo/grad_* now uses the REAL clipped objective's per-token derivative
#     (core_algos.policy_loss_gradient_coef, pinned against autograd), not A.
#     360 rows at ppo_mini_batch_size=60 is six optimizer steps per epoch, so
#     the ratio is 1 only for the first sixth of the first epoch; after that the
#     coefficient is -A*r, and inside a bound clip branch it is exactly zero --
#     a position the objective has stopped pushing at all, which a metric using
#     A still counted at full magnitude. Both coefficients (0.01 and 1.0) and
#     the per-task row weight now multiply both terms, because the ratio is
#     between what the two contribute to ONE objective.
#
#   sign_pair_tokens_step*.jsonl now files each off-task teacher's OWN share.
#     It used to record the position's whole shift against every source that
#     spoke, so with alpha_Search = 1 and alpha_WebShop = 0 both read back
#     carrying the same nats -- and "what did Search bring to AlfWorld" included
#     WebShop's contribution plus the corroboration term neither caused. The
#     corroboration term reaches no source column on purpose: it is what ALL the
#     teachers agreed on, and it is reported whole by evidence/shared_*.
#
#   sign_events_step*.jsonl now carries real probabilities in p_base / p_on /
#     p_off_lo / p_off_hi. They were being fed the STANDARDIZED shifts, which
#     the class exponentiates -- so p_base was exp(0) = 1 on every row and p_on
#     was exp(delta_hat). The shifts ride in their own shift_on / shift_off_lo /
#     shift_off_hi columns instead. The reward column also now has the batch key
#     it comes from: this arm never selected token_level_scores, so every row
#     read nan.
#
#   Plus: a non-finite teacher KL is now zeroed and counted rather than left to
#     reach backward. W = 1 was not a guard -- 1 * NaN is NaN, and so is 0 * NaN
#     at a masked position -- and the optimizer steps before the step-end check
#     fires, so a NaN on-task teacher log-prob would destroy the weights and
#     only then be reported. kl_weight/nonfinite covers four channels now.
#
#   NON-STATIONARITY (the run is 150 steps and the student moves):
#     rms/*/current and /drift_ratio   the scale from THIS step's rows against
#       the cumulative one the weight divides by. Away from 1 by a lot means the
#       divisor is standardising against a policy that no longer exists.
#     adv/*/rho_current and /rho_cumulative, and adv/rho_sign_disagree -- the
#       fraction of ordered pairs whose step-local rho has the opposite sign to
#       the cumulative one alpha is built from.
#     adv/*/informative_group_frac_{cumulative,current}  what fraction of the
#       rows offered to a pair had a prompt group with a spread of advantages.
#       Without it, "the source does not predict reward" and "there was nothing
#       to predict" are the same alpha.
#
#   WHAT MOVED W:
#     effect/weight_kl_lift  the KL-weighted mean of W over the plain mean. At
#       exactly 1 the weight and the KL are uncorrelated, which is the arm being
#       a scalar on the whole term -- teacher_kl_loss_coef with extra steps, and
#       the null this mechanism is tested against. Above 1 it put its budget on
#       the positions that already had the most to distil.
#     effect/weight_kl_corr  the same question bounded, so it compares across
#       runs.
#
#   SEPARATING THE EVIDENCE TOKEN FROM THE AFFECTED TOKEN (2026-08). The single
#   most important distinction in the token analysis, and the one a single table
#   cannot make. W is a scalar on the POSITION, so it scales the OPD term's push
#   on every token in the support -- including tokens no teacher spoke at.
#   "Search's evidence at `retrieve` raised W here" and "the arm reinforced
#   `retrieve`" are different claims, and at a position whose OPD term was
#   pushing something else down harder, the second is false: what it reinforced
#   there is that suppression.
#
#   sign_tokens_step*.jsonl now carries BOTH tables, told apart by ranked_by:
#     ranked_by=count|mass|abs_effect   the EVIDENCE side, unchanged -- which
#       candidates a source spoke at and what their share of the moved nats was.
#     ranked_by=extra_logit_push        the AFFECTED side. Per token, the
#       unweighted descent direction g0 = coef*p_student*(D - f), what the arm
#       added ((W-1)*g0), what the objective applied (W*g0), the student's own
#       probability there and how often it was the token actually emitted.
#       Filed under direction_class, which is the product of two binary facts:
#         push_up_amplified / push_up_damped / push_down_amplified /
#         push_down_damped
#       Sign alone is not the reading -- a weight above 1 at a token the term was
#       pushing DOWN amplifies the suppression. kl_weight/push/<class>/* carries
#       the shares, and mean_p_student beside them says whether the arm was
#       amplifying tokens the student was ever going to say.
#
#   sign_pair_events_step*.jsonl (new file) -- one row per (candidate, SOURCE),
#     which the per-candidate dump cannot be: it can only report the min and max
#     over whichever off-task teachers spoke. Nineteen float columns, with the
#     probabilities and the shifts kept apart:
#       p_base / p_on / p_source / p_student        real policy probabilities
#       delta_on_raw / delta_source_raw             nats
#       delta_on_std / delta_source_std             the same in RMS units
#       alpha_source / shared_evidence / source_evidence / pre_weight /
#       applied_weight / teacher_kl /
#       source_attributed_kl_shift / weighted_logit_push / extra_logit_push /
#       advantage / reward
#     STRATIFIED per (dst, src, pair_state) rather than a global top-N, which is
#     dominated by whichever ordered pair is loudest -- and the event this arm
#     exists to find, a source acting where it DISAGREES with the on-task
#     teacher, is a minority of a minority. Three strata per cell: top by
#     |source_attributed_kl_shift|, top by |extra_logit_push| (not the same rows:
#     one is the evidence, the other the effect), and a hash spread sample.
#     GATHERED ACROSS RANKS, unlike the candidate dump -- a rank-0-local sample
#     is world_size times smaller than a reader assumes. The gather is fixed
#     shape (groups x per_group), so it is a collective on the config and cannot
#     hang on a rank whose micro-batches held nothing. ~47 KB a rank.
#
#   WHERE THE BUDGET WENT, per TRAJECTORY:
#     kl_weight[/task]/outcome/{all,adv_positive,adv_negative,success,failure}/
#       {gross_effect,net_effect,n_rows}, plus
#       adv_positive_to_negative_effect_ratio, success_to_failure_effect_ratio
#       and corr_adv_gross_effect. G_i = sum_t |W-1|D / sum_t D is the fraction
#       of a rollout's own OPD budget the arm redistributed. "The arm moved 3% of
#       the budget" and "the arm moved 3% of the budget, almost all of it on
#       rollouts that failed" are the same number and opposite findings. Success
#       is score > 0; not a causal claim, and the names avoid implying one.
#
#   WHEN A SOURCE DISAGREED:
#     kl_weight/pair_state/{src}__on__{dst}/{agree,conflict,
#       on_silent_source_active,source_silent}/{position_frac,
#       source_evidence_mean,kl_shift_net,kl_shift_gross_share}
#     kl_shift_by_state sums the sources out and evidence/{src}__on__{dst} sums
#     the states out, so the case the mechanism exists to arbitrate was in
#     neither. 144 cells at three tasks.
#
#   WHAT THE MEASUREMENT CANNOT SEE:
#     kl_weight/evidence/{src}__on__{dst}/{support_mass,tail_mass} -- how much of
#     each source teacher's probability lands inside the student's top-k at all.
#     "Search contributed little to AlfWorld" has three explanations that read
#     identically -- weak signal, low alpha, or Search's vocabulary simply not in
#     the support the whole mechanism is measured on -- and only the third is a
#     measurement artefact rather than a finding.
#
#   CHANNEL COUNTERFACTUALS (against the alpha series' MAGNITUDE ones):
#     kl_weight/channel/no_shared/*        the advantage channel alone -- is the
#       corroboration term carrying the mechanism, or decoration on top of
#       "reliable source activity"?
#     kl_weight/channel/offtask_shared/*   the off-task-only agreement rule, the
#       option this arm chose against. Each has its own lagged normaliser, so the
#       comparison is like for like, and neither touches the loss.
#
#   PER-PAIR TOKEN TURNOVER:
#     kl_weight/pair_token/{src}__on__{dst}/turnover/* -- the pooled turnover
#     cannot tell "a stable set" from "each source contributes a stable set and
#     they are different sets", and the second is this arm'"'"'s whole claim.
#
#   COST AND CADENCE. The dense vocabulary tables are ~307 MB a rank on a step
#   that collects them (TokenStateCounts 119, LogitPushTokens 111, SignPairTokens
#   55, RoleTokenCounts 22) and their content is a slow quantity, so they run on
#   a stride: token_stats.every=5 below. On the steps in between they are not
#   allocated at all. The stride is a function of the step COUNT, which every
#   rank shares, never of batch content. token turnover then compares the last
#   two COLLECTED steps, which is the comparison it was always making. Everything
#   scalar -- the weight shape, the outcome buckets, rho/RMS, pair_state -- runs
#   every step and costs a few kilobytes.
#
# GO/NO-GO BEFORE THE LONG RUN. If probe/alpha100/effect/kl_shift_gross_frac is
# below 0.05 on every destination, the arm is not redistributing the OPD budget
# in any measurable way and the three extra forwards -- about 24% of the step --
# are buying nothing. Stop and find out why rather than running to 150 steps.
# 0.05 is not a weight to tune; it is an analysis cut-off fixed before the run.
#
# COLD START is three states and kl_weight/cold_start_state names which one:
# 0 = no RMS yet, 1 = a scale but no normaliser yet, 2 = live. In 0 and 1 every
# weight is exactly 1 -- NOT the raw W~, which the older position arm applies on
# its first step, and not a within-micro-batch mean, which would make the
# objective depend on how the batch was split.
#
# NONE OF IT IS IN THE INTENT LOCK, for the reason the profiler is not: a
# diagnostic that can be switched on mid-run must not become part of what the
# experiment IS. report_epsilon is the one that looks like an exception and is
# not -- it buckets candidates into states for the report and reaches no weight,
# no evidence and no loss.
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
#   PICK ONE of ROLLOUT_PREFETCH_TEACHER / ROLLOUT_PREFETCH_LOGPROB. Unlike pure
#   OPD, this arm DOES have an old_log_prob phase, so prefetched log probs would
#   be consumed — but both prefetches fill the same driver-side glue window and
#   queue on the same colocated WorkerDict, so running both only makes them wait
#   on each other. ROLLOUT_PREFETCH_TEACHER is the one exported below: the teacher
#   forward is by far the more expensive of the two, and the log-prob phase is a
#   single call the trainer makes anyway.
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
#   rollout.return_rollout_log_probs stays True here, and is the one wasted-work
#     removal NOT taken from pure OPD. Its consumer is the rollout-vs-actor drift
#     check in RayPPOTrainer.fit (rollout_probs_diff), which needs an old_log_prob
#     to compare against: pure OPD has none and so drops the column, this arm has
#     one and reads it.
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
# Overlap the validation batches. Depth 1 is the old sequential loop; above it,
# the extra slots are restricted to search (PIPELINEABLE_VAL_TASKS) because
# alfworld's games are seeded by a row's position WITHIN its manager and would
# silently change if split across two. Accuracy-preserving: run_pipelined hands
# results back in submission order, each batch still holds the rows it always
# did, and with the pump off each generate is the same call on the same rows --
# so the [val-hash] digests match a depth-1 run, which is the check to run.
# Costs 2 extra search managers (126 envs each) at depth 3. 1 restores the old
# path exactly.
export VAL_PIPELINE_DEPTH=${VAL_PIPELINE_DEPTH:-3}
# Drive the engine as a pool instead of one blocking generate per batch, so a
# batch's decode-step seats can be filled from another batch instead of running
# the tail of every call on a mostly idle GPU. Pairs with VAL_PIPELINE_DEPTH:
# with one slot there is no second caller to fill from.
#
# NOT accuracy-preserving, unlike the four above, and it is on anyway because
# nothing in TRAINING can reach it: the pool serves only calls that pin n=1
# (do_sample=False or validate=True), and a training rollout sets neither, so it
# is refused and takes the blocking path. What it does change is VALIDATION --
# which requests share a decode step moves floating-point reduction order, so
# [val-hash] will not match a pumped run against an unpumped one. Compare
# scores, not tokens. 0 restores the blocking path.
export ROLLOUT_ASYNC_GENERATE=${ROLLOUT_ASYNC_GENERATE:-1}
# The base policy and the off-task teachers ride in the same rollout window
# as the on-task teacher above. All four are frozen, so only the window
# changes; sign_weight_forward then scores what the window missed. 0 puts
# all three back after the rollout. No-op on the control arm, which has no
# planes to cache.
export ROLLOUT_PREFETCH_SIGN=${ROLLOUT_PREFETCH_SIGN:-1}
# WHERE THIS RUN'S CHECKPOINTS GO. Empty by default, so the paths below are
# byte-for-byte what they were and an existing run resumes exactly as before.
#
# Set it to start a SEPARATE run of this same arm. resume_mode is `auto`, so a
# second run pointed at a directory that already holds global_step_* does not
# start over -- it resumes, and a directory holding a COMPLETED run resumes to
# the final step and exits with nothing done. That is not an error and prints no
# warning, which is how it is mistaken for a launch failure.
#
# The tag moves FOUR things: the two $HOME-derived directories and the two wandb
# names. The directories so a re-run does not resume the finished one; the names
# because a re-run that lands in the finished run's project and under its
# experiment name is indistinguishable from it in the charts -- the same mistake
# one layer up, two runs of one arm reported as one.
#
# The names stay PINNED even so. The lock expects the base plus this same
# suffix, which os.path.expandvars resolves out of the environment when the
# expectations file is read, so a typo in the base still fails the run in
# seconds. What the tag buys is a separate place to look, not an unchecked name.
#
#   RUN_TAG=v2 bash <this script> ...   -> $HOME/checkpoints/<arm>_v2
#                                          wandb <project>_v2 / <experiment>_v2
#
# The suffix is derived once, here, and RUN_TAG_SUFFIX is the only spelling used
# below -- including by the lock, which cannot read bash's ${VAR:+text}.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

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
    +trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_cross_teacher_klw_control_config.yaml \
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
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.no_sync_grad_accum=True \
    actor_rollout_ref.actor.response_only_logits=True \
    actor_rollout_ref.actor.student_indexed_topk=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.return_rollout_log_probs=True \
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
    +algorithm.opd.kl_loss_coef=0.01 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
    +algorithm.opd.normalize_loss_by_task=True \
    +algorithm.opd.cross_teacher_kl_weight.enable=False \
    +algorithm.opd.cross_teacher_kl_weight.base_path=Qwen/Qwen3-1.7B \
    +algorithm.opd.cross_teacher_kl_weight.report_epsilon=0.1 \
    +algorithm.opd.cross_teacher_kl_weight.max_groups=512 \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.enable=True \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.top_n=64 \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.roles=True \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.role_top_n=32 \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.every=5 \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.logit_push=True \
    +algorithm.opd.cross_teacher_kl_weight.token_stats.push_top_n=32 \
    +algorithm.opd.cross_teacher_kl_weight.event_dump.enable=True \
    +algorithm.opd.cross_teacher_kl_weight.event_dump.per_step=128 \
    +algorithm.opd.cross_teacher_kl_weight.event_dump.context=16 \
    +algorithm.opd.cross_teacher_kl_weight.event_dump.pair_strata=True \
    +algorithm.opd.cross_teacher_kl_weight.event_dump.per_group=4 \
    +algorithm.opd.opd_attribution.enable=True \
    +algorithm.opd.opd_attribution.every=5 \
    +algorithm.opd.opd_attribution.top_n=64 \
    +algorithm.opd.opd_attribution.tokens=True \
    +algorithm.opd.opd_attribution.roles=True \
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
    trainer.project_name="verl_agent_opd_grpo_cross_teacher_klw$RUN_TAG_SUFFIX" \
    trainer.experiment_name="opd_grpo_multitask_cross_teacher_klw_control_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/verl_agent_opd_grpo_cross_teacher_klw_control_multitask$RUN_TAG_SUFFIX \
    trainer.val_instance_log_dir=$HOME/val_instances/opd_grpo_multitask_cross_teacher_klw_control_qwen3_1.7b$RUN_TAG_SUFFIX \
    trainer.sign_token_dump_dir=$HOME/sign_tokens/opd_grpo_multitask_cross_teacher_klw_control_qwen3_1.7b$RUN_TAG_SUFFIX \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@"
