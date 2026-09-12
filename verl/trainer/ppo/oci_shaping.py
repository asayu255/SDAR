# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OCI-sat arm A: the injected row's gradient, taken on the prompt that exists
at test time.

THE PROBLEM THIS SOLVES. The injected rollout is generated with a corrupted plan
in its context; the policy that will be deployed has no plan in its context. Arm
A without this file trains an ordinary clipped ratio

    pi_theta(y | x, z) / pi_old(y | x, z)

whose numerator and denominator BOTH carry the plan, so the ratio is ~1, nothing
clips, and the full advantage lands -- on a conditional distribution the model
will never be queried on. Whether that transfers to the no-plan context is an
uncontrolled hope, not a mechanism.

WHAT REPLACES IT. The numerator becomes the policy actually being optimised,
evaluated on the prompt it will actually see:

    rho_t = pi_theta(y_t | y_<t, x) / pi_old(y_t | y_<t, x, z)

-- numerator: the plain student, no plan, teacher-forced on the SAME response
tokens; denominator: the behaviour policy that generated them, which is what
``old_log_probs`` already holds. At the first inner epoch the two differ only in
the conditioning, because ``old_log_probs`` was taken with the current weights;
at later epochs the policy has also moved, which is exactly what the ratio is
for in ordinary PPO.

AND THE RATIO IS SHAPED, NOT CLIPPED. LUFFY (arXiv:2504.14945) REPLACES rho with

    f(rho) = rho / (rho + gamma)

and drops the off-policy PPO clip. The objective term is A * f(rho), so

    d/dlog pi [A * f(rho)] = A * gamma * rho / (rho + gamma)^2

because d rho / d log pi = rho. That coefficient is ZERO at rho -> 0, maximal at
rho = gamma, and falls away again as rho grows: a band-pass, ceiling 1/4 at
rho = gamma. So a failure the plain student would never produce carries no
gradient however large its advantage -- the shaping is what makes it vanish,
not what rescues it. f(rho) near 1 is NOT evidence of reachability; it means rho
is large and the coefficient is small.

NOTHING HERE IS A NEW ESTIMATOR. It is LUFFY's form with the behaviour policy
being the SAME WEIGHTS under a different conditioning rather than a different
model, which is why rho has any chance of landing in the band at all: a frozen
base's failure sits at rho -> 0 and is why that design was abandoned (see
docs/outcome_conditioned_injection_design.md section 4).
"""
from typing import Optional

import torch

# The shaping constant. LUFFY's value; the coefficient peaks at rho = gamma, so
# this also fixes WHERE the usable band sits.
DEFAULT_GAMMA = 0.1

# log rho is clamped before exp. At the top end the coefficient is already
# falling as 1/rho, so the clamp changes nothing that matters and stops an inf
# from reaching the loss; at the bottom end exp underflows to 0 harmlessly,
# which is the correct coefficient anyway.
MAX_LOG_RHO = 20.0


def shaping_coefficient(rho, gamma: float = DEFAULT_GAMMA):
    """``gamma * rho / (rho + gamma)^2`` -- what multiplies A in the gradient.

    Reported, never used to build the loss: the loss is written so autograd
    produces this, and a second hand-rolled copy of it would be a second thing
    to keep in step. ``oci_reachability`` has the same function for the probe,
    where there is no loss at all.
    """
    return gamma * rho / (rho + gamma) ** 2


def shaped_pg_losses(
    log_prob_plain: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    gamma: float = DEFAULT_GAMMA,
    max_log_rho: float = MAX_LOG_RHO,
):
    """Per-token policy loss for the injected rows. Shape ``(bs, response_len)``.

    ``log_prob_plain`` must carry gradient and must come from a forward pass on
    the PLAN-STRIPPED prompt with the same response tokens.

    The loss is ``-A * f(rho)`` and nothing else. Autograd then yields

        d loss / d log_prob_plain = -A * gamma * rho / (rho + gamma)^2

    which is the intended coefficient with the sign a minimiser wants. Writing
    the coefficient out by hand and detaching rho would give the same number
    today and drift from the objective the moment either is edited, so it is not
    done: the objective is the definition and the gradient follows from it.

    NO CLIP. The clip is what the shaping replaces. Adding one back would zero
    the gradient wherever rho left ``[1-eps, 1+eps]``, which for an off-policy
    row is almost everywhere, and the band-pass would never be reached.
    """
    log_rho = (log_prob_plain - old_log_prob).clamp(max=max_log_rho)
    rho = torch.exp(log_rho)
    return -advantages * (rho / (rho + gamma))


def shaping_diagnostics(
    log_prob_plain: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = DEFAULT_GAMMA,
    prefix: str = "oci/shaping",
) -> dict:
    """What the shaping actually did, per step, over the masked tokens.

    ``frac_in_band`` is the share of tokens whose coefficient reaches a tenth of
    its 1/4 ceiling. It is the number that says whether arm A can differ from
    arm B at all: near zero and the injected row contributes nothing, so A is B
    with extra cost.
    """
    with torch.no_grad():
        m = response_mask.to(torch.bool)
        if not bool(m.any()):
            return {f"{prefix}/tokens": 0}
        log_rho = (log_prob_plain - old_log_prob).clamp(max=MAX_LOG_RHO)[m]
        rho = torch.exp(log_rho)
        coef = shaping_coefficient(rho, gamma)
        q = torch.tensor([0.05, 0.5, 0.95], device=log_rho.device, dtype=torch.float32)
        lr_q = torch.quantile(log_rho.to(torch.float32), q)
        cf_q = torch.quantile(coef.to(torch.float32), q)
        return {
            f"{prefix}/tokens": int(m.sum()),
            f"{prefix}/log_rho_p5": float(lr_q[0]),
            f"{prefix}/log_rho_p50": float(lr_q[1]),
            f"{prefix}/log_rho_p95": float(lr_q[2]),
            f"{prefix}/rho_median": float(torch.exp(lr_q[1])),
            f"{prefix}/coef_p5": float(cf_q[0]),
            f"{prefix}/coef_p50": float(cf_q[1]),
            f"{prefix}/coef_p95": float(cf_q[2]),
            f"{prefix}/frac_in_band": float((coef > 0.025).to(torch.float32).mean()),
            f"{prefix}/gamma": gamma,
        }


def injected_rows(micro_batch, key: str = "oci_injected") -> Optional[torch.Tensor]:
    """Boolean row mask for the shaped rows, or None when the column is absent.

    Absent is the ordinary case: the column is only shipped to the actor when the
    arm runs with shaping on, and most micro-batches of an alfworld-only arm
    carry no injected row at all.
    """
    col = micro_batch.get(key, None) if hasattr(micro_batch, "get") else None
    if col is None:
        return None
    mask = col.reshape(-1).to(torch.bool)
    return mask if bool(mask.any()) else None
