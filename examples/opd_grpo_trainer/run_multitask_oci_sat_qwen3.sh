#!/usr/bin/env bash
# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH OCI-sat ARM A: one deliberately-failing rollout injected into each
# already-solved alfworld group, and trained on.
#
# ---------------------------------------------------------------------------
# WHAT THE ARM DOES -- docs/outcome_conditioned_injection_design.md;
# mechanism verl/trainer/ppo/oci_saturated.py and oci_reachability.py.
#
# A GRPO advantage is a trajectory's return minus its prompt group's mean, so a
# group whose eight rollouts all succeeded contributes exactly zero however good
# they were. The arm replaces the last of those eight with the SAME student shown
# its own expert plan with the requirement step removed -- a privileged input
# chosen to make it FAIL -- so the remaining seven stop cancelling.
#
# The candidate slot is marked in the rollout loop, where the env slot means
# something, and travels as the `oci_candidate` column. A candidate whose group
# turned out not to be saturated, or which succeeded anyway, is dropped: its
# response_mask is zeroed AND it is excluded from the group's mean and std, which
# takes both (see F3 in the commit that added exclude_mask).
#
# ---------------------------------------------------------------------------
# THIS IS A WRAPPER, NOT A COPY. The other arm scripts in this directory are
# self-contained duplicates of the base script, which means every one of them is
# a place the control can silently drift away from. This one execs the klw
# CONTROL script -- the only run this arm can be compared against -- and overrides
# four keys. Drift is caught by the intent lock: 84 keys are pinned in
# expected_multitask_oci_sat_config.yaml and exactly five of them differ from the
# control's file (the run's name and the four below).
#
# THE SWITCH IS A CONFIG KEY (algorithm.oci_sat.enable), not an environment
# variable. It was PRIVILEGED_WRONG_PLAN, which a setsid'd process over ssh
# does not inherit -- an arm once ran as a plain student for thirty minutes and
# reported as the arm -- and which no intent lock could pin. The env manager
# reads it off its own copy of the config, so the lock fixes the arm's identity.
# ---------------------------------------------------------------------------
set -euo pipefail

# The same expression the control exports, set HERE too: this script builds
# the experiment_name override before exec'ing the control, so the shell
# expands $RUN_TAG_SUFFIX at that point and it has to already be set. The
# control re-exports the identical value, so this is idempotent, and the
# lock file expects exactly this suffix.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

# Mutually exclusive with the other privileged inputs: two blocks at the head of
# the same observation are two interventions, and the strip that measures rho
# removes only one span.
export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
    algorithm.oci_sat.enable=True \
    algorithm.oci_sat.plan_corruption=misdirect \
    algorithm.oci_sat.detour_steps=50 \
    'algorithm.oci_sat.tasks=[alfworld]' \
    algorithm.oci_sat.gradient_on_injected=True \
    algorithm.compute_mean_std_cross_steps=True \
    trainer.experiment_name="opd_grpo_multitask_oci_sat_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_oci_sat_config.yaml \
    "$@"
