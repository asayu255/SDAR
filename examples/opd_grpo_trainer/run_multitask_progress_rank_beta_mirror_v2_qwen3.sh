#!/usr/bin/env bash
# BETA-MIRROR V2: the beta-mirror arm (header below) with the next-run recipe of 2026-09-27:
#   algorithm.progress_rank.min_top_k.<task>=0       no firing minimum for (a) on any task
#   algorithm.progress_rank.sat_min_spread.<task>=0  sat fires on any turn gap
#   algorithm.progress_rank.search_k=evidence_answered
#                            Search k: 0 not seen / 1 seen, no answer / 2 seen and answered
#   algorithm.progress_rank.beta_exclude_capped=True
#                            WebShop groups whose goal the environment cannot pay (its option-matching
#                            bug caps the correct purchase below 1.0; flagged per row as goal_capped) are
#                            kept out of the tied-group shares the Beta is fitted to. (a) still ranks them.
#   algorithm.progress_rank.beta_share_estimator=discounted_counts
#                            the Beta's input shares from discounted group counts (MoPPS, Reinforce-Ada):
#                            the pseudo-count weighs against the data actually collected (15, 27, 37, ...
#                            groups -> 75), not 75 from the first step.
#   algorithm.progress_rank.beta_denominator=max_discounted
#                            each side divided by max(this step's mean |s|, its discounted mean): a step
#                            with smaller-than-typical gaps pushed less than m, none more.
#   algorithm.progress_rank.beta_cap=max_outcome
#                            no trajectory pushed harder per token than the strongest recent outcome push
#                            in live groups (discounted); a group over it is scaled down alone.
# and, in code, ALFWorld pick_two's "placed" counted as the objects in one receptacle at
# once (agent_system/environments/progress.py).
# Lock: expected_multitask_progress_rank_beta_mirror_v2_config.yaml (no waiver needed).
#
# BETA-MIRROR: (a) and sat sized by a per-task Beta fitted to the tied-group shares.
#
# The (a) arm (run_multitask_progress_rank_qwen3.sh) with the strength rule replaced
# and sat on all three tasks:
#   scale_mode=beta_mirror   per task and step, Beta(a, b) from the EMA shares of the
#                            all-fail / all-success groups (smoothed jointly with the
#                            mixed share, pseudo-count beta_pseudo_count per kind); (a)
#                            at the posterior E[2 sqrt(p(1-p))] of an all-fail group,
#                            sat at that of an all-success group, each divided by the
#                            mean |score| over the trajectories of the task's fired
#                            groups; no cap. See verl/trainer/ppo/progress_rank.py.
#   beta_group_size=8        the group the Beta describes; must equal env.rollout.n
#                            (main_opd_grpo refuses a launch where they differ), and a
#                            group without exactly that many real rollouts is kept out
#                            of the fit and of both updates.
#   sat_gate=False           the rule replaces the gate (beta_mirror refuses the gate).
#   sat_tasks=[alfworld,webshop,search]
#                            sat on all three tasks, ranked by turn count as before.
#   rho=0.1, sat_rho=0.1     only switch the two sides on under beta_mirror.
#   alfworld_k=milestone_arrive, mixed_rho=0.0, and the teacher never retired
#                            (algorithm.opd.retire.enable=False, pinned by the lock).
# The sampling and gradient-path knobs the lock pins (speculative decoding, micro
# batch 5, gradient checkpointing off) are here too, so this script satisfies its
# own lock without the host launcher's help; the launcher adds host knobs only.
#
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RHO="${RHO:-0.1}"
exec bash "$_HERE/run_multitask_progress_rank_qwen3.sh" \
  ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_progress_rank_beta_mirror_v2_config.yaml \
  algorithm.progress_rank.rho=0.1 \
  algorithm.progress_rank.scale_mode=beta_mirror \
  algorithm.progress_rank.beta_pseudo_count=1.0 \
  algorithm.progress_rank.beta_group_size=8 \
  algorithm.progress_rank.sat_rho=0.1 \
  algorithm.progress_rank.sat_gate=False \
  "algorithm.progress_rank.sat_tasks=[alfworld,webshop,search]" \
  algorithm.progress_rank.mixed_rho=0.0 \
  algorithm.progress_rank.alfworld_k=milestone_arrive \
  algorithm.progress_rank.min_top_k.alfworld=0 \
  algorithm.progress_rank.min_top_k.webshop=0 \
  algorithm.progress_rank.min_top_k.search=0 \
  algorithm.progress_rank.sat_min_spread.alfworld=0 \
  algorithm.progress_rank.sat_min_spread.webshop=0 \
  algorithm.progress_rank.sat_min_spread.search=0 \
  algorithm.progress_rank.search_k=evidence_answered \
  algorithm.progress_rank.beta_exclude_capped=True \
  algorithm.progress_rank.beta_share_estimator=discounted_counts \
  algorithm.progress_rank.beta_denominator=max_discounted \
  algorithm.progress_rank.beta_cap=max_outcome \
  ++algorithm.opd.retire.enable=False \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=ngram" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=4" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_min=2" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_max=5" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.acceptance_method=rejection_sampler" \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
  "$@"
