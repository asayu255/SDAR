# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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



# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch


# [EXPLAIN] `AdaptiveKLController` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AdaptiveKLController:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, init_kl_coef, target_kl, horizon):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.value = init_kl_coef
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.target = target_kl
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.horizon = horizon

    # [EXPLAIN] `update` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update(self, current_kl, n_steps):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        target = self.target
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mult = 1 + proportional_error * n_steps / self.horizon
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.value *= mult


# [EXPLAIN] `FixedKLController` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FixedKLController:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Fixed KL controller."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, kl_coef):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.value = kl_coef

    # [EXPLAIN] `update` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update(self, current_kl, n_steps):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass


# [EXPLAIN] `get_kl_controller` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_kl_controller(kl_ctrl):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_ctrl.type == 'fixed':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif kl_ctrl.type == 'adaptive':
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `compute_onlinedpo_pref` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_onlinedpo_pref(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes preferences between pairs of sequences based on summed rewards
    and returns a mask aligned with the interleaved batch.

    Assumes inputs are interleaved: [Resp1_Prompt0, Resp2_Prompt0, Resp1_Prompt1, Resp2_Prompt1, ...]

    Args:
        token_level_rewards: Tensor of shape [batch_size * 2, seq_len]
        response_mask: Tensor of shape [batch_size * 2, seq_len]

    Returns:
        torch.Tensor: A boolean mask of shape [batch_size * 2], where True indicates
                      the corresponding entry is the chosen response for its pair.
                      Example: [True, False, False, True, ...] means for prompt 0,
                               response 1 was chosen; for prompt 1, response 2 was chosen.
    """
    # print(f"---- [DEBUG] Inside compute_onlinedpo_pref ----")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if token_level_rewards.shape[0] % 2 != 0 or response_mask.shape[0] % 2 != 0:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Input tensor batch dimension must be even for pair comparison, "
                         f"got shapes: {token_level_rewards.shape}, {response_mask.shape}")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if token_level_rewards.shape != response_mask.shape:
         # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
         raise ValueError(f"Shape mismatch between rewards {token_level_rewards.shape} and mask {response_mask.shape}")

    # 1. Calculate Sequence Scores
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    # print(f"  Calculated sequence scores shape: {scores.shape}") # [batch_size * 2]

    # 2. Reshape scores to group pairs: [batch_size, 2]
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        score_pairs = scores.view(-1, 2)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except RuntimeError as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"ERROR reshaping scores (shape {scores.shape}) into pairs: {e}")
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise e
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"  Reshaped score pairs shape: {score_pairs.shape}") # [batch_size, 2]

    # 3. Compare scores to find which index (0 or 1) is the winner within each pair
    #    winner_indices[i] = 0 if score_pairs[i, 0] >= score_pairs[i, 1] else 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    winner_indices = torch.argmax(score_pairs, dim=1) # 0 if first is max, 1 if second is max
    # Handle ties explicitly if argmax behavior isn't guaranteed (usually picks first max)
    # Alternatively: winner_mask_original = score_pairs[:, 0] >= score_pairs[:, 1]
    # print(f"  Winner indices shape: {winner_indices.shape}") # [batch_size]
    # print(f"  Number where Response 2 (index 1) is preferred: {winner_indices.sum().item()}") # Counts number of 1s

    # 4. Create the final [batch_size * 2] mask
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_pairs = score_pairs.shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_batch_size = num_pairs * 2
    # Create indices for the full batch [0, 1, 2, 3, ..., N*2-1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_indices = torch.arange(full_batch_size, device=scores.device)
    # Create indices corresponding to the winner within each pair's original index
    # E.g., if winner_indices is [0, 1, 0], pair_indices is [0, 1, 2]
    # winner_global_indices = (pair_indices * 2) + winner_indices -> [ (0*2)+0, (1*2)+1, (2*2)+0 ] -> [0, 3, 4]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pair_indices = torch.arange(num_pairs, device=scores.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    winner_global_indices = (pair_indices * 2) + winner_indices

    # Create boolean mask - True at the winner's position
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_preference_mask = torch.zeros(full_batch_size, dtype=torch.bool, device=scores.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_preference_mask[winner_global_indices] = True

    # print(f"  Output preference mask shape: {output_preference_mask.shape}") # Should be [batch_size * 2]
    # print(f"  Output mask True count (Chosen): {output_preference_mask.sum().item()}") # Should be batch_size
    # print(f"  Output mask False count (Rejected): {(~output_preference_mask).sum().item()}") # Should be batch_size
    # print(f"---- [DEBUG] Exiting compute_onlinedpo_pref ----")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_preference_mask
    


# [EXPLAIN] `compute_online_dpo_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_online_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> torch.Tensor:
    
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import torch.nn.functional as F
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if reference_free:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_logratios = torch.zeros_like(pi_logratios)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = pi_logratios - ref_logratios

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if loss_type == "sigmoid":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_type == "ipo":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        losses = (logits - 1 / (2 * beta)) ** 2
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'sigmoid', 'ipo', or 'hinge'.")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return losses.mean()

# [EXPLAIN] `get_batch_logps` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (e.g., huggingface CausalLMOutputs `logits`). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for computing the sequence log probabilities. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per sequence. Otherwise, return the sum.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given sequences.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if logits.shape[:-1] != labels.shape:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError("Logits and labels must have the same shape[:-1]")

    # Ensure labels are contiguous and on the same device as logits
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = labels.contiguous().to(logits.device)
    # Shift so that tokens < n predict n
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shift_logits = logits[..., :-1, :].contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate per token log probability    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_token_logps = per_token_logps.view(shift_logits.size(0), shift_logits.size(1)) # Reshape back to (batch_size, seq_len-1)
    
    # Create a mask for the labels that are not -100
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss_mask = (shift_labels != -100)
    
    # Apply the mask to the per token log probabilities
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    masked_logps = per_token_logps * loss_mask
    
    # Calculate the sum or average log probability per sequence
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_logps = masked_logps.sum(dim=-1)
    
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if average_log_prob:
        # Avoid division by zero for sequences with no valid tokens
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_valid_tokens = loss_mask.sum(dim=-1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sequence_logps / torch.clamp(num_valid_tokens, min=1)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sequence_logps