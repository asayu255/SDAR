# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
SkillSD (Skill-based Self-Distillation) utilities.

Provides the SDL (Self Distillation Loss) computation for the SkillSD algorithm.
Skill retrieval and teacher batch construction are reused from rlsd_utils.py
and rlsd_ray_trainer.py without modification.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.core_algos import agg_loss


# [EXPLAIN] `compute_sdl_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_sdl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the Self Distillation Loss (SDL) for SkillSD.

    Args:
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
            Current policy forward pass; retains gradients.
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x ⊕ S(x), y_<t).
            Frozen (no grad). Teacher sees skill-augmented input.
        old_log_probs: (bs, response_length) — log π_old(y_t | x, y_<t).
            Frozen (no grad). Rollout-time policy.
        response_mask: (bs, response_length) — mask for valid response tokens.
        loss_agg_mode: aggregation mode passed to agg_loss.

    Returns:
        sdl_loss: scalar — the aggregated SDL loss.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_log_probs = teacher_log_probs.detach()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    old_log_probs = old_log_probs.detach()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ell = student_log_probs - teacher_log_probs

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    neg_ell_clamped = (-ell).clamp(max=20.0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k3 = torch.exp(neg_ell_clamped) - 1.0 + ell

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_rho_on = (student_log_probs - old_log_probs).clamp(max=10.0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rho_on = torch.exp(log_rho_on)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sdl_per_token = rho_on * k3

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sdl_loss = agg_loss(loss_mat=sdl_per_token, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sdl_loss
