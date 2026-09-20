# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the self-distillation term would do to the reward objective, measured.

THE QUESTION. Once the external teacher is retired, the choice is between adding
nothing (plain GRPO) and distilling from the SELF conditioned on this instance's
correct document. The 240-resume ablations answered the first half -- with the
SKILL-conditioned self the term lost to switching the teacher off on all three
tasks -- and the gate said why: its teacher-student gap was -0.031, so the gate
was a constant and the term reduced to pushing up whatever was sampled. The
document-conditioned self is a different teacher (the offline probe reads a gap
of about +1.6 nats per token on ALFWorld), and the honest way to find out whether
IT pulls with the reward is to build the term, measure it, and not apply it.

WHAT IS MEASURED. The same geometry the OPD term already publishes
(``opd/<task>/grpo/first_order``, see cross_teacher_kl_weight.gradient_metrics):
the term's first-order effect on the reward objective, in units of that
objective's own step,

    <g_opsd, g_grpo> / |g_grpo|^2

so the two are read off one chart and a retirement rule can be written against
either. Positive: the term adds to what GRPO does. Negative: it spends its
budget against it.

WHY THIS IS A CLOSED FORM AND NOT A SECOND BACKWARD. The SDAR term is
``g_t * (log pi_T - log pi_S)`` with the gate and the teacher detached
(sdar_utils.sdar_gated_kl), so at a position its derivative with respect to the
logits is ``-coef * g_t * (1[v = y] - p(v))`` -- the SAME direction the policy
gradient has there, ``-pg_coef * pgc_t * (1[v = y] - p(v))``. The two are
parallel at every position and can only agree or disagree through their scalars,
so the whole geometry follows from the gate and the policy-gradient coefficient.

THE ONE APPROXIMATION, and it is deliberate. Each position's shared direction
has squared length ``c_t = |1[v = y] - p|^2``, which weights both the dot product
and both norms. Carrying it exactly would mean a vocabulary-wide reduction on
the logits of every micro-batch (the OPD path can afford one only because its
top-k support is already materialised for the KL). It is dropped here -- ``c_t``
is a positive weight common to both terms, so it cannot change the sign of any
position's contribution and cancels wherever the weighting is flat. What it can
do is reweight positions in the pooled ratio, so read these keys as the geometry
under a uniform per-position weight, and never as a second decimal place on the
OPD numbers. The gate and gap columns beside them are exact.
"""

from typing import Dict, Optional

import torch

from verl.trainer.ppo.cross_teacher_kl_weight import gradient_metrics

__all__ = ["OPSD_TERMS", "gate_and_gap", "geometry_terms", "opsd_metrics"]

# The three gradient_metrics reads, under the names it expects, plus the two
# exact columns that say what the gate was doing while it did it.
OPSD_TERMS = ("g_opd_sq", "g_grpo_sq", "g_dot", "gate", "gap")


def gate_and_gap(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    gate_beta: float = 5.0,
):
    """``(gate, gap)``, both detached, exactly as the SDAR loss builds them."""
    gap = teacher_log_probs.detach() - student_log_probs.detach()
    return torch.sigmoid(gate_beta * gap), gap


def geometry_terms(
    *,
    gate: torch.Tensor,
    gap: torch.Tensor,
    pg_grad_coef: torch.Tensor,
    coef: float,
    pg_coef: float = 1.0,
    row_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Per-position columns for :class:`ScopeTermStats`, in the loss's own signs.

    Args:
        gate: (bs, resp) the detached SDAR gate.
        gap: (bs, resp) ``log pi_T - log pi_S``, detached; carried through so the
            two exact columns come from the same positions as the geometry.
        pg_grad_coef: (bs, resp) ``d(pg_losses)/d(log_prob)``
            (core_algos.policy_loss_gradient_coef) -- the policy gradient's own
            scalar, clipping included, which is why this is not ``-advantages``.
        coef: the term's coefficient (``sdar_loss_coef``), and
        pg_coef: the policy gradient's (``pg_loss_coef``). Both are carried
            because the ratio is between what the two contribute to ONE
            objective.
        row_weight: (bs,) per-task loss weight when the run normalises the loss
            per task. It multiplies both terms, so it is squared here -- the
            reported norms are those of the weighted gradient.
    """
    # BOTH IN DESCENT, which is what makes the dot product theirs and not their
    # negatives'. The SDAR term's derivative with respect to the student's
    # log-prob is -gate (the teacher and the gate are detached, so only the
    # -log pi_S half carries a gradient), and the policy gradient's is
    # pg_grad_coef; descending is the other sign of each.
    g_opsd = float(coef) * gate.detach().to(torch.float32)
    g_grpo = -float(pg_coef) * pg_grad_coef.detach().to(torch.float32)
    out = {
        "g_opd_sq": g_opsd * g_opsd,
        "g_grpo_sq": g_grpo * g_grpo,
        "g_dot": g_opsd * g_grpo,
    }
    if row_weight is not None:
        rw2 = row_weight.detach().to(torch.float32).reshape(-1, 1) ** 2
        out = {name: col * rw2 for name, col in out.items()}
    # Unweighted on purpose: "how open was the gate" is a property of the gate,
    # not of how much of the loss the row was allowed to carry.
    out["gate"] = gate.detach().to(torch.float32)
    out["gap"] = gap.detach().to(torch.float32)
    return out


def opsd_metrics(sums: dict, prefix: str = "opsd") -> dict:
    """:func:`gradient_metrics` plus the gate's own two means, per scope."""
    out = gradient_metrics(sums, prefix=prefix)
    for scope, tot in sums.items():
        n = tot.get("n", 0.0)
        if n <= 0:
            continue
        head = prefix if scope is None else f"{prefix}/{scope}"
        out[f"{head}/gate_mean"] = tot["gate"] / n
        out[f"{head}/gap_mean"] = tot["gap"] / n
    return out
