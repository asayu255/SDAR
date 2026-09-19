"""
Confidence-Gated Teacher Distillation (SDAR) utilities.

Token-level gated distillation loss where the gate is derived from
the teacher-student log-probability gap, so tokens where the teacher
is more confident receive stronger distillation signal.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def sdar_gated_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-token SDAR term before aggregation: ``g_t * (log pi_T - log pi_S)``.

    Split out of :func:`compute_sdar_loss` so that an actor which aggregates by
    per-task row weights (``agg_loss_by_task_weights``) applies exactly the same
    per-token quantity as the token-mean path does. Only the student log-probs
    carry gradients; the gate and the teacher are detached.

    Returns:
        gated_kl: (bs, response_length), carries gradients through the student.
        gate:     (bs, response_length), detached.
        delta:    (bs, response_length), ``log pi_T - log pi_S``, detached.
    """
    teacher_log_probs = teacher_log_probs.detach()
    delta = teacher_log_probs - student_log_probs.detach()
    gate = torch.sigmoid(gate_beta * delta).detach()
    return gate * (teacher_log_probs - student_log_probs), gate, delta


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-Gated Teacher Distillation loss.

    L_SDAR = agg( g_t * (log pi_teacher - log pi_student) )
    where g_t = sigmoid(beta * delta_t), delta_t = log pi_teacher - log pi_student.

    The gate g_t is detached so gradients only flow through the student log-probs.

    Args:
        student_log_probs: (bs, response_length) - log pi_theta(y_t | x, y_<t).
            Current policy forward pass; retains gradients.
        teacher_log_probs: (bs, response_length) - log pi_teacher(y_t | x, r, y_<t).
            Frozen (no grad). Teacher sees skill-augmented input.
        response_mask: (bs, response_length) - mask for valid response tokens.
        gate_beta: temperature for the sigmoid gate. Higher = sharper gating.
        loss_agg_mode: aggregation mode passed to agg_loss.

    Returns:
        sdar_loss: scalar loss.
        metrics: dict with gating statistics.
    """
    gated_kl, gate, delta_t = sdar_gated_kl(student_log_probs, teacher_log_probs, gate_beta)

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
    }

    return loss, metrics
