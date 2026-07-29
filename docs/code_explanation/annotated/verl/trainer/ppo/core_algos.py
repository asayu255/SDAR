# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F


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
    if kl_ctrl.type == "fixed":
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif kl_ctrl.type == "adaptive":
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `compute_gae_advantage_return` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lastgaelam = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages_reversed = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gen_len = token_level_rewards.shape[-1]

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for t in reversed(range(gen_len)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lastgaelam = delta + gamma * lam * lastgaelam
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            advantages_reversed.append(lastgaelam)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns = advantages + values
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = verl_F.masked_whiten(advantages, response_mask)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# [EXPLAIN] `compute_grpo_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = token_level_rewards.sum(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2score = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2mean = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2std = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seen_pairs = set()
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = scores.shape[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (index[i], traj_index[i]) in seen_pairs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2score[index[i]].append(scores[i])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not compute_mean_std_cross_steps:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_pairs.add((index[i], traj_index[i]))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in id2score:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(id2score[idx]) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.tensor(0.0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2std[idx] = torch.tensor(1.0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif len(id2score[idx]) > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"no score in prompt index: {idx}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if norm_adv_by_std_in_grpo:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores[i] = scores[i] - id2mean[index[i]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scores = scores.unsqueeze(-1) * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return scores, scores


# [EXPLAIN] `compute_grpo_passk_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    advantages = torch.zeros_like(scores)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2scores = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2indices = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seen_pairs = set()
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = scores.shape[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (index[i], traj_index[i]) in seen_pairs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx = index[i]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2scores[idx].append(scores[i])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2indices[idx].append(i)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not compute_mean_std_cross_steps:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_pairs.add((index[i], traj_index[i]))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in id2scores:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rewards = torch.stack(id2scores[idx])  # (k,)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if rewards.numel() < 2:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            topk, topk_idx = torch.topk(rewards, 2)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            r_max, r_second_max = topk[0], topk[1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            i_max = id2indices[idx][topk_idx[0].item()]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            advantage = r_max - r_second_max
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if norm_adv_by_std_in_grpo:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                std = torch.std(rewards)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                advantage = advantage / (std + epsilon)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            advantages[i_max] = advantage

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    advantages = advantages.unsqueeze(-1) * response_mask
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return advantages, advantages


# [EXPLAIN] `compute_reinforce_plus_plus_baseline_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = token_level_rewards.shape[-1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = token_level_rewards.sum(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2score = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2mean = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seen_pairs = set()
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = scores.shape[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (index[i], traj_index[i]) in seen_pairs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2score[index[i]].append(scores[i])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not compute_mean_std_cross_steps:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_pairs.add((index[i], traj_index[i]))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in id2score:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(id2score[idx]) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.tensor(0.0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif len(id2score[idx]) > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"no score in prompt index: {idx}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores[i] = scores[i] - id2mean[index[i]]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return scores, scores


# [EXPLAIN] `compute_rloo_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = token_level_rewards.sum(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2score = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2mean = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seen_pairs = set()
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = scores.shape[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (index[i], traj_index[i]) in seen_pairs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2score[index[i]].append(scores[i])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not compute_mean_std_cross_steps:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_pairs.add((index[i], traj_index[i]))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in id2score:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(id2score[idx]) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.tensor(0.0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif len(id2score[idx]) > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"no score in prompt index: {idx}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_num = len(id2score[index[i]])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response_num > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scores = scores.unsqueeze(-1) * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return scores, scores


# [EXPLAIN] `compute_reinforce_plus_plus_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns = torch.zeros_like(token_level_rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        running_return = 0

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for t in reversed(range(token_level_rewards.shape[1])):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            running_return = token_level_rewards[:, t] + gamma * running_return
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            returns[:, t] = running_return
            # Reset after EOS
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            running_return = running_return * response_mask[:, t]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = verl_F.masked_whiten(returns, response_mask)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = advantages * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return advantages, returns


# [EXPLAIN] `compute_remax_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return advantages, returns


# [EXPLAIN] `compute_rewards` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kl = old_log_prob - ref_log_prob
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return token_level_scores - kl * kl_ratio


# [EXPLAIN] `agg_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if loss_agg_mode == "token-mean":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-sum":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.mean(seq_losses)  # seq-mean
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-mean":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.mean(seq_losses)  # seq-mean
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return loss


# [EXPLAIN] `agg_loss_with_sample_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def agg_loss_with_sample_weights(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    loss_agg_mode: str,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Aggregate loss with one scalar weight per sample."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weights = sample_weights.to(device=loss_mat.device, dtype=loss_mat.dtype).view(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_weights = weights.view(-1, 1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if loss_agg_mode == "token-mean":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.sum(loss_mat * loss_mask * token_weights) / torch.sum(loss_mask)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-sum":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.mean(seq_losses * weights)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-mean":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.mean(seq_losses * weights)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss = torch.sum(seq_losses * weights) / loss_mask.shape[-1]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return loss


# [EXPLAIN] `compute_policy_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    negative_approx_kl = log_prob - old_log_prob
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ratio = torch.exp(negative_approx_kl)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses1 = -advantages * ratio
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cliprange_low is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cliprange_low = cliprange
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cliprange_high is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cliprange_high = cliprange
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses3 = -advantages * clip_ratio_c
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# [EXPLAIN] `compute_policy_loss_gspo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cliprange_low is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cliprange_low = cliprange
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cliprange_high is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cliprange_high = cliprange

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses1 = -advantages * seq_importance_ratio
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# [EXPLAIN] `compute_entropy_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return entropy_loss


# [EXPLAIN] `compute_value_loss` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vf_losses1 = (vpreds - returns) ** 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vf_losses2 = (vpredclipped - returns) ** 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return vf_loss, vf_clipfrac


# [EXPLAIN] `kl_penalty` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_penalty in ("kl", "k1"):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return logprob - ref_logprob

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_penalty == "abs":
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (logprob - ref_logprob).abs()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_penalty in ("mse", "k2"):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_penalty in ("low_var_kl", "k3"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kl = ref_logprob - logprob
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ratio = torch.exp(kl)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kld = (ratio - kl - 1).contiguous()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.clamp(kld, min=-10, max=10)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError


# [EXPLAIN] `topk_kl_per_token` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def topk_kl_per_token(
    student_topk_logprob: torch.FloatTensor,
    teacher_topk_logprob: torch.FloatTensor,
    eps: float = 1e-8,
) -> torch.FloatTensor:
    # [EXPLAIN] 両入力 shape は (batch, response_length, k)、出力は (batch, response_length) である。
    # [EXPLAIN] token ID は teacher 側で選んだ同一 support を使うため、この関数へ渡る前に student logits を
    # [EXPLAIN] `teacher_topk_ids` で gather しておく。single-sampled-token の `low_var_kl` とは別経路である。
    # [EXPLAIN] teacher は detach して gradient を遮断し、student_topk_logprob 側だけが backward graph を持つ。
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Dense per-token reverse KL over a top-k support (+ a tail bucket).

    Approximates the exact per-token reverse KL ``KL(p_student || p_teacher)`` by
    restricting the support to the teacher's top-k token ids and lumping the
    remaining vocabulary mass into a single "tail" bucket. Uses the plain (k1)
    KL definition ``sum_v p_s(v) (log p_s(v) - log p_t(v))`` rather than a
    single-sample estimator, so it is low-variance and dense.

    Both inputs are full-vocabulary log-softmax values gathered at the *teacher's*
    top-k ids, so ``exp(.).sum(-1) <= 1`` and the leftover is the tail mass.
    Only ``student_topk_logprob`` carries gradients; the teacher is detached.

    Args:
        student_topk_logprob: (bs, response_length, k) student log-probs at the
            teacher's top-k ids.
        teacher_topk_logprob: (bs, response_length, k) teacher log-probs at its
            own top-k ids (detached upstream).
    Returns:
        (bs, response_length) per-token KL (>= 0 up to the tail approximation).
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    teacher_topk_logprob = teacher_topk_logprob.detach()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    p_s = student_topk_logprob.exp()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    p_t = teacher_topk_logprob.exp()

    # Tail mass = 1 - sum(top-k mass), clamped for numerical safety.
    # [EXPLAIN] top-k 外の全 vocabulary を1個の tail bucket に集約するため、top-k 確率質量の残差を求める。
    # [EXPLAIN] `eps` clamp は log(0) を避ける数値安定化で、k 個の support と tail の合計で近似 KL を構成する。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tail_s = (1.0 - p_s.sum(dim=-1)).clamp(min=eps, max=1.0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tail_t = (1.0 - p_t.sum(dim=-1)).clamp(min=eps, max=1.0)

    # sum_k p_s (log p_s - log p_t)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    topk_term = (p_s * (student_topk_logprob - teacher_topk_logprob)).sum(dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tail_term = tail_s * (torch.log(tail_s) - torch.log(tail_t))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return topk_term + tail_term


# [EXPLAIN] `compute_pf_ppo_reweight_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad()
    # [EXPLAIN] `compute_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if reweight_method == "pow":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weights = torch.pow(torch.abs(scores), weight_pow)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif reweight_method == "max_min":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_score = torch.max(scores)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            min_score = torch.min(scores)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif reweight_method == "max_random":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_score = torch.max(scores)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weights = torch.where(scores == max_score, 0.4, 0.1)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return weights

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = data.batch["token_level_scores"].sum(dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weights = compute_weights(scores, reweight_method, weight_pow)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = scores.shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sample_indices_np = sample_indices.numpy()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_non_tensor_batch = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, array in data.non_tensor_batch.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(array, np.ndarray):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_meta_info = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, value in data.meta_info.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(value, list) and len(value) == batch_size:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resampled_meta_info[key] = value

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from copy import deepcopy

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_data = deepcopy(data)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_data.batch = type(data.batch)(resampled_batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_data.batch.batch_size = data.batch.batch_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resampled_data.meta_info = resampled_meta_info

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return resampled_data
