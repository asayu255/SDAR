# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Confidence-Gated Teacher Distillation (SDAR) utilities.

Token-level gated distillation loss where the gate is derived from
the teacher-student log-probability gap, so tokens where the teacher
is more confident receive stronger distillation signal.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.core_algos import agg_loss


# [EXPLAIN] `compute_sdar_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_log_probs = teacher_log_probs.detach()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    delta_t = teacher_log_probs - student_log_probs.detach()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gate = torch.sigmoid(gate_beta * delta_t).detach()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kl_per_token = teacher_log_probs - student_log_probs

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gated_kl = gate * kl_per_token

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_sum = response_mask.sum().clamp(min=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate_mean = (gate * response_mask).sum() / mask_sum
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gap_mean = (delta_t * response_mask).sum() / mask_sum

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
    }

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return loss, metrics
