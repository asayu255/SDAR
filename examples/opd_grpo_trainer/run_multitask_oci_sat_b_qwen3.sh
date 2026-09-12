#!/usr/bin/env bash
# OPD + GRPO multitask (alfworld + search + webshop), Qwen3-1.7B,
# WITH OCI-sat ARM B: the injected rollout moves the group's baseline and then
# has its own gradient removed.
#
# ARM A AND ARM B DIFFER IN ONE KEY AND IT IS THE WHOLE QUESTION. The seven
# native successes go from an advantage of exactly zero to +0.375 either way --
# that is the payload, and it is on-policy and unclipped. What B removes is the
# injected row's own gradient, which is the only part that depends on the
# injected row being REACHABLE (rho, and the shaping coefficient
# A*gamma*rho/(rho+gamma)^2, which is zero at rho=0). So B needs no claim about
# reachability at all: if B works and A does not, the mechanism is the contrast
# and the off-policy term is noise. If neither works the contrast is worth
# nothing and the design is done.
#
# The removal happens AFTER compute_advantage. Zeroing earlier would put the
# group back to uniform and there would be no arm.
#
# See run_multitask_oci_sat_qwen3.sh for what the arm does and why this is a
# wrapper rather than a copy.
set -euo pipefail

# The same expression the control exports, set HERE too: this script builds
# the experiment_name override before exec'ing the control, so the shell
# expands $RUN_TAG_SUFFIX at that point and it has to already be set. The
# control re-exports the identical value, so this is idempotent, and the
# lock file expects exactly this suffix.
export RUN_TAG=${RUN_TAG:-}
export RUN_TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

export PRIVILEGED_SKILLS=""
export PRIVILEGED_PLAN=""

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_HERE/run_multitask_cross_teacher_klw_control_qwen3.sh" \
    algorithm.oci_sat.enable=True \
    algorithm.oci_sat.plan_corruption=misdirect \
    algorithm.oci_sat.detour_steps=50 \
    'algorithm.oci_sat.tasks=[alfworld]' \
    algorithm.oci_sat.gradient_on_injected=False \
    algorithm.compute_mean_std_cross_steps=True \
    trainer.experiment_name="opd_grpo_multitask_oci_sat_b_qwen3_1.7b$RUN_TAG_SUFFIX" \
    ++trainer.expected_config=examples/opd_grpo_trainer/expected_multitask_oci_sat_b_config.yaml \
    "$@"
