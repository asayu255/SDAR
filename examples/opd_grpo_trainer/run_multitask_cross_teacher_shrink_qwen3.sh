set -x

# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH THE UNIFORM-SHRINKAGE CROSS-TEACHER TARGET (lambda' = 0.6).

# ---------------------------------------------------------------------------
# WHAT THIS ARM IS. The distillation target stops being the on-task teacher and
# becomes the normalised GEOMETRIC MIXTURE of all three:
#
#   log p_tilde_d = log p_0 + sigma_d [ lambda' hat_h_d
#                                     + (1 - lambda') mean_{m != d} hat_h_m ] - log Z
#
# with lambda' = 0.6, i.e. weights (0.6, 0.2, 0.2) on (on-task, off-task, off-task).
# Designed in docs/cross_teacher_shrink_design.md; the theory it comes from is
# docs/cross_teacher_theory.md 3.3 and 5.2 (arm E3).
#
# MIXING THE KLs AND MIXING THE TARGET ARE THE SAME MECHANISM. For weights that
# sum to one,
#
#   sum_m w_m KL(pi_theta || pi'_m) = KL(pi_theta || p_tilde) - log Z
#
# and log Z does not depend on the student, so the gradients are identical
# (tests/trainer/test_cross_teacher_shrink.py::test_mixing_the_kls_is_the_mixed_target).
# It is written in the target because that is where the normalisation, the tail
# handling and the diagnostics already live.
#
# WHY THE WEIGHTS SUM TO ONE. Adding an off-task KL term ON TOP of the full
# on-task one would take the effective teacher coefficient from beta to
# beta(1 + w). Theory 4.5: the coefficient is a FIRST-order variable and the
# mixture is second-order, so an unnormalised version would confound "mixed the
# target" with "turned the teacher up" -- and the second effect is the larger one.
#
# WHY sigma_d. The teachers were trained at different KL coefficients (search at
# 0.001, the other two at 0.01), so the off-task voices are converted into the
# DESTINATION teacher's nats before mixing. Equivalent to mixing the KLs against
# gain-matched teachers pi'_m ~ pi_0 (pi_m/pi_0)^(sigma_d/sigma_m), for which
# pi'_d is pi_d itself.
#
# ---------------------------------------------------------------------------
# WHAT lambda' = 0.6 CLAIMS. lambda' = (1 + (K-1) lambda)/K with
# lambda = sigma_s^2 / (sigma_s^2 + sigma_eps^2), so 0.6 is lambda = 0.4: four
# tenths of the on-task teacher's deviation from the teachers' consensus is real
# task knowledge and six tenths is estimation error. That is a PRIOR, not a
# measurement -- the quantity is not identified from three teachers and a base
# (theory 3.2), and measuring it needs a second seed of the on-task teacher
# (theory 6.1, E4). The design document says so in its limits section, and this
# line is where the run admits it.
#
# THE RANGE IS [1/K, 1] = [1/3, 1] AND NOTHING OUTSIDE IT. lambda' = 1 is the
# control BIT-FOR-BIT (the arm contains its own control, and a test pins that).
# lambda' = 1/3 is the equal-weight geometric mean of all three teachers, i.e.
# routing removed from the target -- the pure multi-teacher OPD baseline. Below
# 1/3 the on-task teacher would be weighted under an off-task one, which no
# posterior does; the actor refuses values outside [0, 1] and the design argues
# the tighter bound.
#
# ---------------------------------------------------------------------------
# THIS ARM MOVES THE FIXED POINT, which is the difference from the curriculum
# arm beside it. mode=curriculum subtracts uncorroborated parts of the ON-TASK
# shift, is bounded by base and the on-task teacher by construction (injection
# zero), and its last stage is the control exactly -- so it is a claim about
# ORDER and only readable in intermediate steps. Here the off-task teachers can
# push PAST what the on-task teacher said, the target is different at
# convergence, and target/shrink/beyond_* counts how often it actually happens.
# That is why _EXPONENT_CLAMP stays on in this mode and is off in the other one.
#
# STOPPING AT 150 IS A STOPPING POINT, NOT THE EXPERIMENT. Because the fixed
# point moves, the natural readout is the asymptote and 150 steps only shows the
# transient. total_training_steps stays 300 so the LR schedule, the data order
# and the step index match every arm this is compared against; STOP_AFTER_STEPS
# ends the run at a checkpoint step and the run resumes to 300 if it earns it.
#
# ---------------------------------------------------------------------------
# WHAT TO READ IN THE RUN LOG / WANDB.
#   target/tv                      how far the target moved from the control's.
#   target/abs_dkl_mean            what that did to the loss, in nats.
#   target/shrink/c_to_on_absmass  THE DOSE: sum p|c| over sum p|h_d|. Set by
#                                  lambda' by construction, so this is the run
#                                  confirming the size, not discovering it.
#   target/shrink/beyond_cand_frac INJECTION, counted. Zero is impossible here
#                                  unless lambda' = 1.
#   target/shrink/role/content_share
#                                  the audit measured the shared component as
#                                  FORMAT; if the mixture lands on content
#                                  instead, that is the webshop-damage path
#                                  (audit 4.6) and it is visible here long
#                                  before the validation is.
#   target/clamped_frac_of_acted, target/clamped_mass_frac
#                                  the clamp is the only cap. READ THE MASS one.
#   target/shuffled_tv_ratio       G1 INVERTS in this mode too (measured 1.620
#                                  at step 3, against 1.45 predicted from the
#                                  audit's spreads). Shuffling breaks the +0.72
#                                  correlation between the on- and off-task
#                                  shifts, and c is proportional to their
#                                  DIFFERENCE, so |c| grows. Above 1 is the
#                                  default; near 1 would mean the two shifts are
#                                  uncorrelated.
#   target/shrink/role/*_share     WHERE THE INJECTION LANDS. The audit's
#                                  mechanism for the webshop damage is other
#                                  tasks' specific components reaching CONTENT
#                                  positions, and this is that, measured.
#   target/entropy_delta           the TARGET's entropy against the on-task
#                                  teacher's. A geometric mixture is a product of
#                                  experts and therefore SHARPER, not flatter:
#                                  -0.039 measured. (The first version of this
#                                  header predicted the opposite, from a wrong
#                                  reason -- the target is not nearer base.)
#   actor/entropy                  the STUDENT's, which the target's entropy does
#                                  not determine. The klw arm's gain was
#                                  attributed to entropy (audit 15), so a gain
#                                  here is NOT separable from exploration
#                                  whichever way this moves.
#
# PRE-REGISTERED PREDICTION (docs/cross_teacher_shrink_design.md): no task
# differs from the control by more than the validation floor at 150. The offline
# lambda' sweep over the existing dumps is flat across the whole admissible
# range, paired against the control (+0.009 [-0.022, +0.042] at lambda' = 0.8;
# +0.004 [-0.035, +0.047] at 0.6). A null here is the predicted result and is
# NOT evidence the dose was too small: at 0.6 the median tilt on emitted tokens
# is about 1.8 nats.
#
# FALSIFICATION: any task above the control by more than 2x the validation
# floor at BOTH 75 and 150, in the deterministic scoring configuration.
#
# OBSERVED ON THE FIRST LAUNCH (stopped at step 13 for the roles fix; see
# docs/cross_teacher_shrink_design.md 6.4): tv 0.068, abs_dkl_mean 0.233 nats,
# beyond_cand_frac 0.226 / beyond_mass_frac 0.309, c_to_on_absmass 0.277,
# clamped_mass_frac 0.255. The injection is real and the dose is larger than the
# predecessor target arm's 1.42% TV, so a null here will not be a null of size.
#
# TIMING, measured rather than estimated: 13 min of startup and ~12.3 min/step,
# so 150 steps is about 31 hours on two GPUs.
#
# ---------------------------------------------------------------------------
# ENVIRONMENT. SEARCH_URL must point at the retriever -- it is NOT local on
# every host (wakaba, 100.86.45.30) and the wrong value HANGS rather than
# failing. LAMBDA_PRIME and STOP_AFTER_STEPS are overridable, but the intent
# lock pins lambda_prime, so changing it means editing the lock in the same
# commit -- which is why it is NOT an environment override here.

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
# Match the mini-batch COLUMNS across ranks. _balance_batch equalises each rank's
# total over the whole batch and reports that it worked to within a token; that
# says nothing about mini-batch k, which is where the ranks actually meet. The
# measurement has been shipping for a while and prices it:
# global_seqlen/minibatch_wait_frac against _columns, and
# global_seqlen/microbatch_wait_frac against ITS _columns -- the second pair is
# the larger one on this arm, because the teacher lookup runs a collective pair
# from inside the micro-batch loop and so makes the ranks meet every
# ppo_micro_batch_size_per_gpu rows, not once a mini-batch.
#
# NOT the same thing as BALANCE_MINIBATCH, which is still off and stays off: that
# one re-deals each rank's whole partition, so rows stop sharing an optimizer step
# and the arm is no longer comparable with the ones already run. This one
# re-partitions INSIDE each column, so the sixty rows an optimizer step sees are
# the sixty it saw before and only the rank carrying each of them moves.
#
# The gradient is then invariant, not merely similar, and that rests on this arm's
# config: normalize_loss_by_task=True routes every term through
# agg_loss_by_task_weights, a weighted row SUM whose weights come from STEP-level
# token totals, and the actor multiplies back the FSDP average and the configured
# gradient_accumulation. Under a plain token-mean it would NOT hold -- that
# normalises per micro-batch -- so this belongs with normalize_loss_by_task and
# not on any arm without it.
#
# What still moves is floating-point summation order, the same class as
# ppo_micro_batch_size_per_gpu 10 -> 5 above. Both arms of the A/B therefore have
# to carry the same value, which is why it is exported here and in the control
# script and not left to a launch command. 0 restores index order.
export BALANCE_MINIBATCH_COLUMNS=${BALANCE_MINIBATCH_COLUMNS:-1}
# The base policy and the off-task teachers ride in the same rollout window
# as the on-task teacher above. All four are frozen, so only the window
# changes; sign_weight_forward then scores what the window missed. 0 puts
# all three back after the rollout. No-op on the control arm, which has no
# planes to cache.
export ROLLOUT_PREFETCH_SIGN=${ROLLOUT_PREFETCH_SIGN:-1}
# Let the TRAINING rollout through the pumped engine pool, and stop joining the
# teacher prefetch chunk before every generation. TokenPump steps the engine on a
# thread inside the worker, so between two pump_step RPCs the colocated actor's
# call slot is free -- the one thing in this tree that breaks the serialisation
# docs/gpu_profiling_report_opd.md 2.4 describes. The frozen forwards then run
# BESIDE the decode tail, which is 70.5% of gen's wall clock at 17% tensor-pipe
# activity. Needs ROLLOUT_ASYNC_GENERATE=1 and return_rollout_log_probs=False
# (below), and the same value on all three cross-teacher arms: it is not
# bit-identical (arrival timing decides which requests share a decode step).
export ROLLOUT_PUMP_TRAINING=${ROLLOUT_PUMP_TRAINING:-1}
# Score the finished rows' old_log_prob in the rollout window too. The consumer
# is real on this arm -- opd_grpo_ray_trainer's old_log_prob phase reads it
# through compute_log_prob_with_prefetch -- and the pure-OPD note that "there is
# no consumer" is about the thin loop, which has no such phase. Worth ~44 s a
# step, but only alongside ROLLOUT_PUMP_TRAINING: without it this competes with
# the teacher chunk for the same 35-40 s of glue.
export ROLLOUT_PREFETCH_LOGPROB=${ROLLOUT_PREFETCH_LOGPROB:-1}
# Size the IN-WINDOW frozen forwards by tokens instead of by 4 rows. OFF (0) by
# default and left off here, because the row bound was bought with an OOM: a
# chunk's activations sit beside a live vLLM KV pool.
#
# It is the next thing to try, and the first run to carry the mechanisms above
# says why. Prefetching the sign planes took sign_weight_forward from 143 s to
# 2.5 s a step -- and gen grew by very nearly the same amount, because the
# forwards run inside the window at the same ~30% MFU that made them 143 s
# outside it. Overlapping a launch-bound pass with a decode does not make it stop
# being launch-bound; only a bigger micro-batch does. teacher_cache/device_gb now
# reads 0.000 against a teacher_cache/gb of ~15, so the 7-8 GB a card that the
# cache used to hold during the rollout is free for exactly this.
#
# Try 8192 (2.5x the ~3.2k tokens 4 rows come to) and watch
# perf/update_peak_allocated_gb and the allocator's retry counter:
#   ROLLOUT_WINDOW_FORWARD_TOKENS=8192 bash examples/opd_grpo_trainer/...
export ROLLOUT_WINDOW_FORWARD_TOKENS=${ROLLOUT_WINDOW_FORWARD_TOKENS:-0}
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

# Scoring an existing checkpoint instead of training: VAL_ONLY=1 VAL_CKPT=<dir>.
#
# These four flags used to be typed as trailing overrides, and getting them there
# is not the safe operation it looks like. A single missing backslash in the
# chain ends the command early: the script runs with whatever survived, hydra
# never sees trainer.val_only, and the run TRAINS FROM SCRATCH into this arm's
# checkpoint directory -- for ten hours, with one "command not found" scrolled
# past somewhere above. Nothing downstream can catch it, because a training run
# is exactly what the config then asks for.
#
# So the flags are assembled here, as a unit, or not at all. VAL_CKPT is
# required rather than defaulted: a wrong-but-plausible default would resume the
# wrong step and report it under the right name.
VAL_ONLY_ARGS=()
if [ "${VAL_ONLY:-0}" = "1" ]; then
    : "${VAL_CKPT:?VAL_ONLY=1 needs VAL_CKPT=/path/to/.../global_step_N}"
    case "$VAL_CKPT" in
        *global_step_*) ;;
        *) echo "VAL_CKPT must be a global_step_N directory, got: $VAL_CKPT" >&2; exit 1 ;;
    esac
    [ -d "$VAL_CKPT/actor" ] || {
        echo "no actor/ under VAL_CKPT: $VAL_CKPT" >&2
        echo "(the shards are named model_world_size_<N>_rank_<r>.pt, where N is" >&2
        echo " n_gpus_per_node * nnodes -- a checkpoint saved on 2 GPUs cannot be" >&2
        echo " read back on 3)" >&2
        exit 1
    }
    VAL_ONLY_ARGS=(
        trainer.val_only=True
        trainer.resume_mode=resume_path
        "trainer.resume_from_path=$VAL_CKPT"
        trainer.del_local_ckpt_after_load=False
    )
    echo "[val-only] scoring $VAL_CKPT -- no training, no checkpoint written"
fi

python3 -m verl.trainer.main_opd_grpo \
    +trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_cross_teacher_shrink_config.yaml \
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
    +algorithm.opd.kl_loss_coef=0.01 \
    +algorithm.opd.kl_loss_type=topk_kl \
    +algorithm.opd.topk=20 \
    +algorithm.opd.normalize_loss_by_task=True \
    +algorithm.opd.cross_teacher_target.enable=True \
    +algorithm.opd.cross_teacher_target.base_path=Qwen/Qwen3-1.7B \
    +algorithm.opd.cross_teacher_target.mode=shrink \
    +algorithm.opd.cross_teacher_target.lambda_prime=0.6 \
    +algorithm.opd.cross_teacher_target.token_stats.enable=True \
    +algorithm.opd.cross_teacher_target.token_stats.top_n=64 \
    +algorithm.opd.cross_teacher_target.token_stats.every=5 \
    +algorithm.opd.cross_teacher_target.event_dump.enable=True \
    +algorithm.opd.cross_teacher_target.event_dump.per_step=128 \
    +algorithm.opd.cross_teacher_target.event_dump.context=16 \
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
    env.search.search_url="${SEARCH_URL:-http://0.0.0.0:8000/retrieve}" \
    env.search.timeout=600 \
    env.search.max_retries=null \
    env.multitask.tasks=[alfworld,search,webshop] \
    env.multitask.max_steps.alfworld=50 \
    env.multitask.max_steps.search=4 \
    env.multitask.max_steps.webshop=15 \
    +env.multitask.history_length.alfworld=2 \
    +env.multitask.history_length.search=4 \
    +env.multitask.history_length.webshop=2 \
    "env.multitask.val_per_task_batch_size={search: 252}" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="verl_agent_opd_grpo_cross_teacher_shrink$RUN_TAG_SUFFIX" \
    trainer.experiment_name="opd_grpo_multitask_cross_teacher_shrink_qwen3_1.7b$RUN_TAG_SUFFIX" \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/verl_agent_opd_grpo_cross_teacher_shrink_multitask$RUN_TAG_SUFFIX \
    trainer.val_instance_log_dir=$HOME/val_instances/opd_grpo_multitask_cross_teacher_shrink_qwen3_1.7b$RUN_TAG_SUFFIX \
    +trainer.val_instance_log_text=True \
    trainer.sign_token_dump_dir=$HOME/sign_tokens/opd_grpo_multitask_cross_teacher_shrink_qwen3_1.7b$RUN_TAG_SUFFIX \
    trainer.save_freq=25 \
    trainer.test_freq=150 \
    trainer.total_training_steps=300 \
    trainer.stop_after_steps=${STOP_AFTER_STEPS:-150} \
    trainer.total_epochs=300 \
    trainer.val_before_train=False "$@" "${VAL_ONLY_ARGS[@]}"
