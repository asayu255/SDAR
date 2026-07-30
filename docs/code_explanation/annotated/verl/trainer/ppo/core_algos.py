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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

# -----------------------------------------------------------------------------
# 【このファイルの役割（日本語補足）】
# -----------------------------------------------------------------------------
# - PPO/GRPO系の学習で使う「数式部分」を、分散実行方式から独立した関数としてまとめる。
# Ray、FSDP、vLLMなどのworker管理はこのファイルでは行わない。
# - 上位の主な呼び出し元は次の2層。
# 1. ray_trainer.py:
# rewardからGAE/GRPO/RLOO/ReMax等のadvantageを作る。
# 2. dp_actor.py / dp_critic.py:
# current policy/valueのforward結果からPPO loss、value loss、KL lossを作る。
# - したがって処理境界は概ね次のようになる。
# ray_trainer.py: reward/value -> advantage
# dp_actor.py: current log_prob -> core_algos.py -> scalar loss -> loss.backward()
# 【主要なshape表記】
# - B: batch size
# - R: response token数（右paddingを含む固定長）
# - K: teacher top-k support size
# - loss_mat / advantages / log_prob: 通常 (B, R)
# - response_mask: (B, R)。有効なresponse tokenだけ1、padding/EOS後は0。
# 【勾配の境界】
# - advantage計算関数はtorch.no_grad()内で実行し、学習ターゲットを固定する。
# - policy/value/KL loss関数はcurrent model側Tensorの計算グラフを保持する。
# - old policy、reference/teacher側Tensorは呼び出し元またはこのファイル内でdetachされる。
# -----------------------------------------------------------------------------
# prompt/group IDごとにscoreを集約するadvantage推定で使用する。
from collections import defaultdict

# KL係数のclipや、DataProtoのnon-tensor index処理に使用する。
import numpy as np
# advantage・loss・samplingを実装する主要Tensorライブラリ。
import torch

# masked_mean、masked_whiten、entropyなどverl共通のmask付き演算を利用する。
import verl.utils.torch_functional as verl_F


# 観測KLがtargetより大きければ係数betaを増やし、小さければ減らす適応制御器。
# reward shapingで r - beta * KL を使う場合に、KL逸脱量を目標付近へ保つ。
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    # value: 現在のKL係数beta、target: 目標KL、horizon: 係数変化の時間スケール。
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        # current_kl > targetなら正、current_kl < targetなら負になる。
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        # 加法更新ではなく乗法更新なので、betaのスケールに比例して変化する。
        self.value *= mult


# KL係数を学習中ずっと固定する制御器。update()はインターフェース互換のno-op。
class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    # current_kl / target - 1 を相対誤差とし、1回の更新が急激にならないよう[-0.2,0.2]へclipする。
    # n_steps / horizonが大きいほど、同じ誤差でもbetaを大きく変更する。
    def update(self, current_kl, n_steps):
        pass


# Hydra/OmegaConfの設定値からfixed/adaptiveの具体的なcontrollerを生成するfactory。
def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# GAE: critic valueを利用してtokenごとのadvantageとreturnを計算する。
# delta_t = r_t + gamma * V_{t+1} - V_t
# A_t     = delta_t + gamma * lambda * A_{t+1}
# return_t = A_t + V_t
# 未来側A_{t+1}が必要なので、response末尾から先頭へ逆順に走査する。
# -----------------------------------------------------------------------------
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
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
    # advantage/returnはoptimizerで微分する対象ではないため、計算グラフを構築しない。
    with torch.no_grad():
        # 逆向き再帰で保持するA_{t+1}。系列末尾の次状態は0から開始する。
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # t=R-1, R-2, ..., 0の順にTD residualとlambda-returnを蓄積する。
        for t in reversed(range(gen_len)):
            # 最終response tokenの次状態価値は0として扱う。
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # 1-step TD error。shapeは(B,)で、batch各sampleを同時に計算する。
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # lambdaにより1-step TDと長期returnのbias/varianceを補間する。
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # criticの回帰target。whitening前のadvantageをvalueへ足して作る。
        returns = advantages + values
        # 有効response tokenだけで平均0・分散1へ正規化し、paddingは統計から除外する。
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# -----------------------------------------------------------------------------
# GRPO outcome advantage。
# 各responseのtoken rewardを1 scalar scoreへ合計し、同じprompt/group ID内で相対化する。
# criticを使わず、group内の他responseがbaselineになる。
# score_i = sum_t reward_{i,t}
# A_i = (score_i - group_mean) / (group_std + eps)   [通常GRPO]
# A_i =  score_i - group_mean                        [Dr.GRPO型]
# 得られたsample-level A_iを、そのresponseの全有効tokenへbroadcastする。
# -----------------------------------------------------------------------------
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
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
    # (B,R) -> (B,)。outcome rewardをresponse単位のscalarへ集約する。
    scores = token_level_rewards.sum(dim=-1)

    # index(prompt/group ID)ごとにscoreを集め、group統計を計算する。
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    # multi-turnで同じ(index, traj_index)が複数rowに現れる場合の重複除外に使う。
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # compute_mean_std_cross_steps=Falseでは、同一trajectoryをgroup統計へ1回だけ入れる。
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            # 同じindexを持つresponse群がGRPOの比較groupになる。
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            # group size=1では相対比較不能なので、実装上mean=0,std=1へフォールバックする。
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # group内scoreの平均。続くstdと合わせて各responseを標準化する。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # Trueは標準GRPO、Falseはstdで割らないDr.GRPO型のcentered advantage。
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        # (B,) -> (B,1)を(B,R)へbroadcastし、padding/EOS後を0にする。
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# Pass@k向けGRPO advantage。
# group内の1位responseだけに「1位score - 2位score」を与え、その他は0にする。
# top responseをさらに改善する方向へ集中した疎な学習信号になる。
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
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
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    # 初期状態では全sampleのadvantageを0にし、group winnerだけ後で書き換える。
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            # group内上位2件を取得し、勝者のmarginを計算する。
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            # 1位と2位が同点ならwinner advantageも0になる。
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            # 非zero信号は各groupのbest sample 1件だけに付与する。
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# REINFORCE++ baseline版。
# group平均scoreをbaselineとして引いた後、response全tokenへ複製し、さらにmask付きwhiteningする。
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
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
    # sample-level scoreを後でR tokenへtileするためresponse長を保持する。
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        # centered sample scoreを(B,R)へ明示的に複製し、有効tokenだけ残す。
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        # batch内の有効token全体で再標準化し、最終的なscaleを安定化する。
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


# RLOO (REINFORCE Leave-One-Out) advantage。
# sample i自身を除いた同一groupの平均をbaselineにする。
# 実装式 score_i*n/(n-1) - mean*n/(n-1) は score_i - mean_{j!=i} と等価。
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
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
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # n=group size。n=1ではleave-one-out baselineを作れないため更新しない。
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # group全体平均から自分自身を除いたbaselineとの差を、閉形式で計算する。
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# REINFORCE++のdiscounted return版。
# 各token位置からresponse末尾までの割引報酬和を作り、criticなしでそのreturnをwhitenしてadvantageにする。
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
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

    with torch.no_grad():
        # tokenごとのreturn格納先。shapeは入力rewardと同じ(B,R)。
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        # 将来報酬を累積するため末尾から逆向きに処理する。
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            # 無効領域を跨いで別episode/sequenceへreturnが伝播しないようにresetする。
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


# ReMax advantage。
# responseのreverse cumulative rewardをreturnとし、greedy decode等で得たsample-level baselineを差し引く。
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
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

    with torch.no_grad():
        # flip -> cumsum -> flipにより、各tから末尾までの未割引reward和をベクトル化して計算する。
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        # (B,) baselineを(B,R)へbroadcastし、有効response位置だけで差し引く。
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


# sampled token上のold policyとreference policyのlog-prob差を、token rewardから減算する簡易reward shaping。
# ray_trainer.apply_kl_penaltyの低レベル相当で、戻り値shapeはtoken_level_scoresと同じ。
def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    # sampled actionについてのlog ratio log(pi_old/pi_ref)。厳密なfull-vocab KLではない。
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


# -----------------------------------------------------------------------------
# (B,R)のtoken-level lossを1 scalarへ集約する共通関数。
# 集約方式により「長いresponseを重くするか」「各sequenceを等重みにするか」が変わる。
# -----------------------------------------------------------------------------
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
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
    # 全有効tokenの総和 / 全有効token数。長いsequenceはtoken数に比例して寄与が大きい。
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    # 各sequence内ではtoken-sum、その後B sequenceを平均。長いsequenceほど1sample内lossが大きくなり得る。
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    # sequenceごとにtoken平均を作ってからsequence平均。各sampleを概ね等重みに扱う。
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    # 全sequenceのtoken-sumを固定長Rで割るDr.GRPO向け形式。実batch sizeでは割らない点に注意。
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


# sampleごとのscalar weightを加えたloss集約。
# task別KL係数など、同じmicro-batch内でrowごとにloss強度を変えるときdp_actor.pyから呼ばれる。
def agg_loss_with_sample_weights(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    loss_agg_mode: str,
):
    """Aggregate loss with one scalar weight per sample."""
    # (B,)へ整形し、lossと同device/dtypeへ移す。weight自体は通常gradientを持たない。
    weights = sample_weights.to(device=loss_mat.device, dtype=loss_mat.dtype).view(-1)
    # (B,1)へしてR方向へbroadcastする。同一sample内の全tokenへ同じ係数を適用する。
    token_weights = weights.view(-1, 1)
    if loss_agg_mode == "token-mean":
        # denominatorはweight付きtoken数ではなく元の有効token数。weightは平均だけでなくloss scaleも変える。
        loss = torch.sum(loss_mat * loss_mask * token_weights) / torch.sum(loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses * weights)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses * weights)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses * weights) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


# -----------------------------------------------------------------------------
# PPO / dual-clip PPOのpolicy loss。
# r_t(theta) = exp(log pi_theta(a_t) - log pi_old(a_t))
# L_clip = min(r_t A_t, clip(r_t,1-eps_low,1+eps_high) A_t)
# 実装はgradient descent用に負号を付けたlossを作るため、目的関数のminはloss上のmaximumに対応する。
# すべての主要Tensor shapeは(B,R)。response_maskでpaddingを除外してscalarへ集約する。
# -----------------------------------------------------------------------------
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
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    # 名前はnegative_approx_klだが、値はlog(pi_current/pi_old)。ratioのlogそのもの。
    negative_approx_kl = log_prob - old_log_prob
    # importance-sampling ratio r_t。old policyで収集したtokenをcurrent policy更新に再利用する補正係数。
    ratio = torch.exp(negative_approx_kl)
    # monitoring用にmean(log pi_old - log pi_current)を計算。lossへ自動加算される項ではない。
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # unclipped surrogateの負値。optimizerはloss最小化なので、元のr*A最大化と等価。
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    # ratioを許容範囲へclipしたsurrogate。Aの符号に応じて有効になるclip側が変わる。
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    # 負の目的を最小化するためmaxを選ぶ。元のPPO式におけるmin(rA,clip(r)A)と等価。
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    # clipped branchがunclipped branchより厳しいtoken割合。更新がclip境界へ当たる頻度の診断値。
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # dual-clip用の第3候補。A<0のとき、ratioが極端に大きい場合のlossを-c*Aで制限する。
    pg_losses3 = -advantages * clip_ratio_c
    # A<0用に通常PPO lossとdual-clip下限の小さい方を選ぶ。A>=0では後のwhereで使わない。
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # negative advantage tokenだけdual-clip、positive advantage tokenは標準PPO clipを適用する。
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # token-level surrogateを設定された集約方式で1 scalarへ変換し、backward可能なlossとして返す。
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# -----------------------------------------------------------------------------
# GSPO policy loss。
# tokenごとのratioではなく、response全体の幾何平均ratioをforward値として共有する。
# 一方、detach trickにより各token log_probへgradientを流す。
# -----------------------------------------------------------------------------
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

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    # 各responseの有効token数 |y_i|。全mask=0でもdivision-by-zeroしないよう最低1にする。
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    # token log-ratioのsequence平均。expするとsequence probability ratioの幾何平均になる。
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    # forward値はsequence平均log-ratio、gradientは(log_prob-log_prob.detach())を通じて各tokenへ1で流れる。
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    # exp前の上限を10にし、非常に大きいratioによるoverflowを抑える。下限はこの実装ではclipしない。
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    # 引数loss_agg_modeではなく、実装上はGSPO推奨のseq-mean-token-meanを固定使用する。
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    # standard GSPOではdual-clip lower metricを使わないため、呼び出し側とのtuple互換用に0を返す。
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# categorical distributionのtoken entropyを計算し、mask付きscalarへ集約する互換関数。
# ここで返す値は正のentropyであり、探索促進時の符号付けは呼び出し側のloss合成で決まる。
def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    # (B,R,V)のlogitsから各token位置のH(pi)を計算し、(B,R)へ縮約する。
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


# critic/value head用のclipped regression loss。
# current valueがold valueから一度に大きく変わらないよう、PPO policy clipと類似の制約を入れる。
def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
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
    # current predictionをold value中心の[value-eps, value+eps]へclipした候補を作る。
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    # unclipped/clippedのうち誤差が大きい方をlossに採用し、clipを利用した過度な改善を許さない。
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


# sampled actionのlog-prob差から複数種類のtoken-level KL surrogateを返す。
# 入力は通常(B,R)で、full vocabulary分布ではない。呼び出し側がresponse_maskで集約する。
def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    # k1: signed log-ratio log(pi/ref)。単一sampleでは負にもなり得るが、期待値はforward KLに対応する。
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    # log-prob差の絶対値。方向を無視してpolicy/referenceの乖離を罰する。
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    # k2: 1/2*(log-ratio)^2。0近傍で対称な二次ペナルティになる。
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    # k3: exp(ref-current) - (ref-current) - 1。非負に近く、sample estimatorの分散を抑える形式。
    if kl_penalty in ("low_var_kl", "k3"):
        # ここではlog(ref/pi)を置き、ratio=ref/piを作る。
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        # 数値異常や極端なratioがlossを支配しないよう、tokenごとに[-10,10]へ制限する。
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


# -----------------------------------------------------------------------------
# Pure OPDのtop-k teacher KL。
# teacherが選んだK token ID上でstudent/teacherのfull-vocab log-softmaxを比較し、
# top-k外の確率質量は1個のtail bucketへまとめてreverse KL KL(student || teacher)を近似する。
# 入力:  student_topk_logprob, teacher_topk_logprob = (B,R,K)
# 出力:  token KL = (B,R)
# -----------------------------------------------------------------------------
def topk_kl_per_token(
    student_topk_logprob: torch.FloatTensor,
    teacher_topk_logprob: torch.FloatTensor,
    eps: float = 1e-8,
) -> torch.FloatTensor:
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
    # teacherを固定targetにし、backwardはstudent分布側だけへ流す。
    teacher_topk_logprob = teacher_topk_logprob.detach()

    # teacher top-k ID位置におけるstudent確率。K個の和は通常1未満。
    p_s = student_topk_logprob.exp()
    p_t = teacher_topk_logprob.exp()

    # Tail mass = 1 - sum(top-k mass), clamped for numerical safety.
    # top-k外の全vocabulary確率を1つのtail categoryとしてまとめる。log(0)防止でeps以上にclip。
    tail_s = (1.0 - p_s.sum(dim=-1)).clamp(min=eps, max=1.0)
    tail_t = (1.0 - p_t.sum(dim=-1)).clamp(min=eps, max=1.0)

    # sum_k p_s (log p_s - log p_t)
    # top-k support上のsum p_student*(log p_student-log p_teacher)。K次元を縮約して(B,R)。
    topk_term = (p_s * (student_topk_logprob - teacher_topk_logprob)).sum(dim=-1)
    # top-k外ではtoken別内訳を捨て、tail総質量同士のKL 1項として近似する。
    tail_term = tail_s * (torch.log(tail_s) - torch.log(tail_t))
    return topk_term + tail_term


# PF-PPO用のreward依存resampling。
# lossにweightを掛けるのではなく、scoreからsampling weightを作り、batchを復元抽出してDataProto全体を組み替える。
def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    # sampling weightとindex選択は学習グラフに含めない。
    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        # |score|^p: reward絶対値が大きいsampleを高確率で選ぶ。
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        # batch内の最大・最小scoreだけを候補にし、中間sampleのweightを0にする。
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        # 最大scoreに0.4、その他に0.1を与える固定的な優先sampling。
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    # (B,R) token scoreをsample scalar (B,)へ変換する。
    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    # 元batchと同じB件を復元抽出するため、同じsampleが複数回選ばれる可能性がある。
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    # TensorDict内の全keyを同一indexで並べ替え、input/reward/log_prob等のrow対応を維持する。
    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    # non_tensor_batchはNumPy/listなので同じindexをCPU NumPy形式へ変換する。
    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    # 元DataProtoを破壊せず、batch/non-tensor/meta_infoを整合した新しいDataProtoを返す。
    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
