# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# 【このファイルの役割（日本語補足）】
# - verl の PPO/GRPO 学習の「基底トレーナ」RayPPOTrainer を定義するファイル。
# SDAR の SkillSDRayTrainer / RLSDRayTrainer はこの RayPPOTrainer を継承している。
# - Ray の「シングルコントローラ」方式: ドライバプロセス（このクラス）が RPC で各GPUワーカ
# （actor/critic/ref/reward）を呼び出し、advantage 計算など軽い処理はドライバ側で行う。
# - 3タスク同時学習のためにこのファイルへ加わった主な追加点:
# (1) apply_invalid_action_penalty / _get_invalid_action_penalty_coef の task 別係数対応
# (2) _normalize_task_name / _validation_task_name / _validation_kwargs_for_batch による
# task 別の検証生成設定（val_kwargs_by_task）
# (3) 学習/検証バッチから task_name を pop してロールアウトへ引き渡す処理
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# 生成結果を JSONL でダンプする際に使用
import json
# パス操作・チェックポイント探索など
import os
# サンプルごとの一意ID(uid)生成に使用（GRPOグループの識別に効く）
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_throughout_metrics,
    compute_timing_metrics,
    get_task_names,
    normalize_task_name,
    process_validation_metrics,
    task_row_indices,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


# 【日本語補足】学習に登場する「役割」の列挙。どの役割をどのGPUプールに載せ、
# どのワーカクラスで動かすかを role_worker_mapping で対応づける。
class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    # 方策（学習される本体）
    Actor = 0
    # 生成エンジン（推論）
    Rollout = 1
    # Actor と Rollout を同居させたハイブリッド（本コードで使用）
    ActorRollout = 2
    # 価値関数（GAE 使用時のみ。GRPO では不使用）
    Critic = 3
    # 参照方策（KL 正則化の基準となる固定モデル）
    RefPolicy = 4
    # 報酬モデル（学習済みモデルで報酬を出す方式）
    RewardModel = 5
    # Actor+Rollout+Ref を同居させる構成
    ActorRolloutRef = 6


# 【日本語補足】advantage 推定方式の列挙。SDAR は "grpo" を使う。
# タイプミス防止のため文字列を直書きせず、この Enum 経由で参照する。
class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    # 価値関数を使う一般化アドバンテージ推定
    GAE = "gae"
    # グループ相対方策最適化（SDARで使用）
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    # ステップ単位のグループ化を加えた GRPO 拡張
    GiGPO = 'gigpo'


@dataclass
# 【日本語補足】GPU資源を「プール」として確保・割り当てる管理役。
# どのプールに何GPU×何ノード積むか(resource_pool_spec)と、
# どの役割をどのプールに載せるか(mapping)を保持する。
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    # プール名 -> 各ノードのGPU数リスト
    resource_pool_spec: dict[str, list[int]]
    # 役割 -> プール名
    mapping: dict[Role, str]
    # 実体プールのキャッシュ
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        # 全プール・全ノードの GPU 数を合計して返す（スループット計算等で使う）。
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        # 要求した GPU 数が実際にクラスタで確保可能かを検証する（不足なら早期エラー）。
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        # 仕様に従って Ray の資源プールを実際に確保する。
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    # 生成された応答トークン
    responses = data.batch["responses"]
    # 応答長
    response_length = responses.size(1)
    # 報酬（トークン単位スコア）
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 応答部分だけを 1 にするマスクを作る（マルチターンかどうかで参照するマスクが違う）。
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    # 現方策(old_log_probs)と参照方策(ref_log_prob)のトークン別 KL を計算。
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    # 応答部分だけ有効化
    kld = kld * response_mask
    # KL 係数（適応制御される）
    beta = kl_ctrl.value

    # 報酬から KL ペナルティを差し引く
    # average over sequence（系列平均）
    token_level_rewards = token_level_scores - beta * kld

    sequence_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(sequence_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # バッチ平均のスカラ
    # 適応 KL 係数を今回の KL 値で更新
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # KL 差し引き後の報酬を書き戻す
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    # Same average, restricted to the rows of one task (the coefficient is a
    # single global controller value, so it has no per-task counterpart).
    for task, rows in task_row_indices(data).items():
        metrics[f"actor/reward_kl_penalty/{task}"] = torch.mean(sequence_kl[torch.from_numpy(rows)], dim=0).item()

    return data, metrics

def _get_invalid_action_penalty_coef(data_item, invalid_action_penalty_coef, invalid_action_penalty_coef_by_task=None):
    # ★3タスク同時学習: 1サンプルに適用する「無効行動ペナルティ係数」を task 別に出し分ける。
    # 単タスク（by_task 未指定）なら常にスカラ係数を返す。
    if not invalid_action_penalty_coef_by_task:
        return float(invalid_action_penalty_coef)

    # サンプルの task_name を取得。無ければ一律係数にフォールバック。
    task_name = data_item.non_tensor_batch.get("task_name", None)
    if task_name is None:
        # by_task 無し → 従来通りの一律係数
        return float(invalid_action_penalty_coef)

    # 表記ゆれ（"alfworld/AlfredTWEnv" 等）を正規の3種名に正規化。
    task_name = str(task_name).lower()
    if "alfworld" in task_name:
        task_name = "alfworld"
    elif "webshop" in task_name:
        task_name = "webshop"
    elif "search" in task_name:
        task_name = "search"

    # task 別辞書から係数を引く（未知 task はフォールバック係数）。
    return float(invalid_action_penalty_coef_by_task.get(task_name, invalid_action_penalty_coef))


def apply_invalid_action_penalty(
    data: DataProto,
    # 一律の無効行動ペナルティ係数
    invalid_action_penalty_coef=float,
    # ★同時学習: task 別係数の辞書（任意）
    invalid_action_penalty_coef_by_task=None,
):
    # フォーマット違反など「無効な行動」を出したサンプルに、報酬でペナルティを課す。
    # トークン単位スコア（この末尾トークンを減点）
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        # GiGPO 用のステップ報酬があれば同様に減点
        step_rewards = data.batch['step_rewards']
    # サンプルごとに処理
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        # プロンプト長（応答の開始位置）
        prompt_length = prompt_ids.shape[-1]

        # 応答の有効トークン長（パディングを除く）。末尾トークン位置の特定に使う。
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        # 1=有効, 0=無効
        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        # 無効フラグ(1-有効)をテンソル化。squeeze(0) で余分な次元を除去。
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # ★このサンプルに適用する係数を task 別に取得（単タスクなら一律係数）。
        penalty_coef = _get_invalid_action_penalty_coef(
            data_item=data_item,
            invalid_action_penalty_coef=invalid_action_penalty_coef,
            invalid_action_penalty_coef_by_task=invalid_action_penalty_coef_by_task,
        )
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        # 応答の最終有効トークン位置の報酬から、無効なら penalty_coef を引く。
        reward_tensor[i, valid_response_length - 1] -= penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            # ステップ報酬側も同様に減点
            step_rewards[i] -= penalty_coef * action_invalids
    
    is_action_valid = data.non_tensor_batch['is_action_valid'].astype(np.float32)
    valid_action_ratio = np.mean(is_action_valid).item()
    # バッチ全体の「有効行動割合」をモニタリング用に集計。
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    for task, rows in task_row_indices(data).items():
        metrics[f'episode/valid_action_ratio/{task}'] = np.mean(is_action_valid[rows]).item()
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    # 応答トークン
    responses = data.batch["responses"]
    # 応答長
    response_length = responses.size(1)
    # 全体マスク（プロンプト＋応答）
    attention_mask = data.batch["attention_mask"]
    # 末尾 response_length 個 = 応答部分のマスク
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    # 応答マスクが未計算なら補う（後方互換）。
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    # adv_estimator の種類ごとに分岐。SDAR は GRPO ブランチを通る。
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE: critic の value を使う一般化アドバンテージ推定（SDAR では未使用）。
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        # ★SDAR が通る分岐。GRPO は「同一プロンプトから生成した group_size 個の応答」を
        # 1グループとして、報酬をグループ内で正規化して advantage を作る。
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    # 以下は SDAR では使わない代替 advantage 推定方式（GRPO-passk / REINFORCE++ / REMAX / RLOO / GiGPO）。
    # いずれも uid でグループ化する点は GRPO と同様。
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            # グループ標準偏差で割るか
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            # ★index=uid: グループ化キー。uid はプロンプト単位に振られ、1プロンプト=1タスクなので、
            # グループは必ず単一タスク内に閉じる → 3タスク同時学習でも task 間で advantage は混ざらない。
            index=data.non_tensor_batch["uid"],
            # 軌跡ID（マルチターンの同一軌跡を束ねる）
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        # 各トークンの advantage
        data.batch["advantages"] = advantages
        # リターン（学習ターゲット）
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    from verl.utils import gpu_profiler

    # GPU プロファイラに区間開始を通知
    gpu_profiler.push_phase(name)
    try:
        # 経過時間を計測
        with Timer(name=name, logger=None) as timer:
            # with 内のコードを実行
            yield
    finally:
        # 区間終了を通知（例外時も必ず）
        gpu_profiler.pop_phase(name)
    if name not in timing_raw:
        timing_raw[name] = 0
    # 同名区間の累積時間に加算
    timing_raw[name] += timer.last


# 【日本語補足】PPO/GRPO 学習の基底トレーナ。SDAR の SkillSDRayTrainer はこれを継承し、
# fit() をオーバーライドして教師フォワードと蒸留損失を差し込む。
# このクラス自身のドライバは軽量（RPCで各GPUワーカを呼ぶ司令塔）で、
# advantage 計算など軽い処理だけをここ（ドライバ）で行う。
class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        # 主要な依存物を属性に保持（トークナイザ・報酬関数・環境・軌跡コレクタなど）。
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # 学習用報酬関数
        self.reward_fn = reward_fn
        # 検証用報酬関数
        self.val_reward_fn = val_reward_fn
        # 学習用環境（同時学習では MultiTaskEnvironmentManager）
        self.envs = envs
        # 検証用環境
        self.val_envs = val_envs
        # 環境と対話して軌跡を集める役
        self.traj_collector = traj_collector

        # actor と rollout を同居させる方式
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        # 現状ハイブリッドのみ対応
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        # 参照方策を使うか
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # 報酬モデルを使うか
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        # 検証生成のログ出力器
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        # LoRA 使用時は、参照方策を actor から LoRA を外した状態で兼用する。
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        # 報酬に KL を混ぜる方式(use_kl_in_reward)なら、適応 KL 係数コントローラを用意。
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # advantage 推定方式に応じて critic（価値関数）が要るか決める。GAE のみ critic 必要。
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            # GRPO 系は critic 不要（グループ相対で baseline を得る）
            self.use_critic = False
        else:
            raise NotImplementedError

        # 設定の整合性チェック
        self._validate_config()
        # データローダ構築（★ここで train_sampler=TaskBalancedSampler が組み込まれる）。
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        # 各種バッチサイズ・並列サイズ・相互排他オプションなどの整合性を検証する（長いが定型処理）。
        config = self.config
        # number of GPUs total
        # 総GPU数
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        # 実効バッチ = train_batch_size × rollout.n。総GPU数で割り切れないと分配できない。
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        # 旧式(micro_batch_size)と新式(..._per_gpu)を同時指定していないか等を検証するヘルパ。
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        # 検証でサンプリングする(do_sample)なら温度>0 が必要（温度0はサンプリングと矛盾）。
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    # 【日本語補足】Dataset(1件ずつ) + Sampler(順序) + collate_fn(結合) を束ねて、
    # バッチの流れを作る「StatefulDataLoader」を構築する。Stateful=途中位置を保存/復元でき、
    # チェックポイント再開時に続きのバッチから再開できる。
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # 呼び出し側で渡されていなければ、ここで生成（通常は main_ppo/main_sdar 側で渡す）。
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # サンプラも未指定なら生成（★同時学習では TaskBalancedSampler が渡ってくる）。
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        # 学習用データローダ。
        # batch_size = gen_batch_size（無ければ train_batch_size）。同時学習では 45(=15×3)。
        # ★sampler が index を「45連続で各タスク15件ずつ」並べ、DataLoader が 45 で束ねることで
        # 各バッチのタスク比が保証される（Sampler と DataLoader の分業）。
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            # 端数バッチは捨てる（バッチサイズを一定に保つ）
            drop_last=True,
            collate_fn=collate_fn,
            # index の順序決め（TaskBalancedSampler / RandomSampler）
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            # 未指定なら検証データ全件を1バッチに
            val_batch_size = len(self.val_dataset)

        # 検証用データローダ。shuffle=False（決定的）で、先頭から順に評価する。
        # ★同時学習では test parquet がタスク順ソート済みなので、各 val バッチが単一タスクになる。
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            # 並列読み込みワーカ数
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            # 複数サンプル→1バッチ構造への結合
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        # 既定の総ステップ数 = 1エポックのバッチ数 × エポック数。
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # config で明示指定があればそちらを優先（同時学習は 300 を明示）。
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _normalize_task_name(task_name):
        return normalize_task_name(task_name)

    @staticmethod
    def _attach_task_ids(batch: DataProto):
        """Tag every row with an integer task id for worker-side per-task metrics.

        The workers only receive tensors from ``batch.batch``, so the string
        ``task_name`` cannot travel with the batch into ``update_actor``. The id
        column plus the ``task_id_names`` lookup in ``meta_info`` (both preserved
        by chunking) let the actor split its loss metrics per task. Rows with no
        task name get id ``-1`` and are skipped there.
        """
        task_names = get_task_names(batch)
        if task_names is None:
            return
        names = sorted({task_name for task_name in task_names if task_name is not None})
        name_to_id = {task_name: task_id for task_id, task_name in enumerate(names)}
        batch.batch["task_ids"] = torch.tensor(
            [name_to_id.get(task_name, -1) for task_name in task_names], dtype=torch.long
        )
        batch.meta_info["task_id_names"] = names

    @staticmethod
    def _entropy_loss_metrics(batch: DataProto, entropys, response_masks, loss_agg_mode) -> dict:
        """``actor/entropy_loss`` plus the same aggregation restricted to each task."""
        metrics = {
            "actor/entropy_loss": agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode).detach().item(),
        }
        for task, rows in task_row_indices(batch).items():
            task_rows = torch.from_numpy(rows)
            metrics[f"actor/entropy_loss/{task}"] = (
                agg_loss(loss_mat=entropys[task_rows], loss_mask=response_masks[task_rows], loss_agg_mode=loss_agg_mode).detach().item()
            )
        return metrics

    def _validation_task_name(self, batch: DataProto):
        # ★同時学習: 検証バッチが「単一タスク」であることを確認し、その task 名を返す。
        # まず non_tensor_batch から task_name を取得。
        task_names = batch.non_tensor_batch.get("task_name", None)
        if task_names is None:
            # 無ければ env_kwargs の中の task_name を拾う（フォールバック）。
            env_kwargs = batch.non_tensor_batch.get("env_kwargs", None)
            if env_kwargs is not None:
                task_names = [
                    item.get("task_name") if isinstance(item, dict) else None
                    for item in env_kwargs
                ]
        if task_names is None:
            # task 情報が全く無ければ None（task別設定は使わない）
            return None

        # バッチ内の task を集合化。
        normalized = {self._normalize_task_name(task_name) for task_name in task_names}
        normalized.discard(None)
        # task別の検証設定は「1バッチ=1タスク」前提。混在していたらエラー（test はソート済みのはず）。
        if len(normalized) > 1:
            raise ValueError(
                "Task-specific validation kwargs require single-task validation batches, "
                f"got mixed tasks: {sorted(normalized)}"
            )
        # 唯一の task 名（無ければ None）
        return next(iter(normalized), None)

    def _validation_kwargs_for_batch(self, batch: DataProto):
        # ★同時学習: この検証バッチの task に応じた生成設定（温度・サンプリング有無）を返す。
        # 既定値（全体設定）から開始。
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        kwargs = {
            "do_sample": val_kwargs.do_sample,
            "temperature": val_kwargs.temperature,
        }

        # task別設定(val_kwargs_by_task)が無ければ既定のまま返す（＝単タスクの挙動）。
        by_task = self.config.actor_rollout_ref.rollout.get("val_kwargs_by_task", None)
        if not by_task:
            return kwargs
        if OmegaConf.is_config(by_task):
            # 素の dict に変換
            by_task = OmegaConf.to_container(by_task, resolve=True)

        # このバッチの task を特定。
        task_name = self._validation_task_name(batch)
        # 生の task_name を正規の3種（alfworld/webshop/search）に正規化するユーティリティ。
        # 表記ゆれ（"alfworld/AlfredTWEnv","Webshop" 等）を吸収する。
        if task_name is None:
            return kwargs
        # task別設定で上書き（例: search は do_sample=False/temperature=0 の貪欲、他は温度0.4サンプリング）。
        task_kwargs = by_task.get(task_name, {})
        if "do_sample" in task_kwargs:
            kwargs["do_sample"] = bool(task_kwargs["do_sample"])
        if "temperature" in task_kwargs:
            kwargs["temperature"] = float(task_kwargs["temperature"])
        return kwargs

    def _validate(self):
        # 検証ループ。val_dataloader を回してタスクを解かせ、報酬・成功率などを集計する。
        # 各バッチの報酬テンソル
        reward_tensor_lst = []
        # データソース（QAの出所など）
        data_source_lst = []
        task_name_lst = []
        # ツール呼び出し回数
        tool_calling_list = []
        # 軌跡ID
        traj_uid_list = []
        # 成功率（環境が返す）
        success_rate_dict = {}

        # Lists to collect samples for the table
        # ログ表示用に入出力とスコアを集める。
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # 検証バッチを順に処理（同時学習では1バッチ=1タスク）
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            # 検証用に各プロンプトを val_kwargs.n 回複製（複数サンプルで評価する場合）。
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            # ★同時学習: task_name も生成バッチへ引き渡す（env ルーティング＆val_kwargs判定に必要）。
            if "task_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("task_name")
            # 生成に必要なキーを取り出す。
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # 生成時のメタ情報（EOS/PADトークン、検証フラグ等）。
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "validate": True,
            }
            # ★このバッチの task に応じた生成設定（温度・do_sample）を上書き適用。
            test_gen_batch.meta_info.update(self._validation_kwargs_for_batch(test_gen_batch))
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            # 検証用環境(val_envs)と複数ターン対話してタスクを解かせる（is_train=False）。
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # 報酬関数でこのバッチを採点。
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            # 系列合計スコア
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            batch_task_names = get_task_names(test_batch)
            if batch_task_names is None:
                batch_task_names = np.array([None] * reward_tensor.shape[0], dtype=object)
            task_name_lst.append(batch_task_names)
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        # 検証生成の一部をログ表示（設定時）。
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 全バッチの結果を連結して、データソース別にスコアを集計していく。
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        task_names = np.concatenate(task_name_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate test_score based on task (a task can span several data sources,
        # e.g. search validates on nq / hotpotqa / ...; for the tasks whose data
        # source is the task name this repeats the per-data-source value above)
        task_reward = {}
        for i in range(reward_tensor.shape[0]):
            task_name = task_names[i]
            if task_name is None:
                continue
            task_reward.setdefault(task_name, []).append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        task_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_task_names = task_names[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

            task_name = unique_task_names[i]
            if task_name is not None:
                task_tool_calling.setdefault(task_name, []).append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for task_name, rewards in task_reward.items():
            metric_dict[f'val/{task_name}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for task_name, tool_calls in task_tool_calling.items():
            metric_dict[f'val/{task_name}/tool_call_count/mean'] = np.mean(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict

    # 【日本語補足】GPUワーカ群を実際に起動・初期化する。役割ごとに worker class を
    # 該当プールへ割り当て、最後に vLLM が KVキャッシュ量を正しく見積もれるよう
    # actor_rollout を最後に初期化する。
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # GPUプールを確保
        self.resource_pool_manager.create_resource_pool()

        # プール -> {役割名: クラス} の対応を初期化。
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        # actor+rollout（本体）を該当プールへ登録。
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # actor_rollout は最後に初期化（vLLM の KVキャッシュ見積もりを良くするため）。
        self.actor_rollout_wg = all_wg["actor_rollout"]
        # モデル重みをロード
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        # 非同期ロールアウトモードなら、専用のサーバマネージャを立てる。
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        # チェックポイント保存: actor（必要なら critic）の重みと、dataloader の位置を保存する。
        # ※SkillSD/RLSD 側ではこのメソッドをオーバライドし、先読み時は「先読み前の位置」で保存する。
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        # dataloader の進行状態（どこまで読んだか）を保存 → 途中再開で続きから再開できる。
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        # チェックポイント読み込み: resume_mode に応じて最新or指定チェックポイントから復元する。
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                # 再開しない（最初から学習）
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _fast_forward_env_schedules(self):
        """Replay the env-side episode schedules that the restart skipped.

        Call this right after ``_load_checkpoint()`` in any ``fit()`` that resets
        the training envs once per global step. WebShop's goal RNG and ALFWorld's
        game-file cycle live in this process, not in the checkpoint, so without
        this a resumed run re-trains on the same early episodes; see
        ``agent_system/environments/resume.py``.

        Idempotent, a no-op on a fresh run, and never fatal.
        """
        if getattr(self, "_env_schedules_fast_forwarded", False):
            return
        self._env_schedules_fast_forwarded = True

        envs = getattr(self, "envs", None)
        if envs is None or getattr(self, "global_steps", 0) <= 0:
            return

        # Dynamic sampling re-runs the rollout (and its env reset) an unknown
        # number of times per step, so the number of resets already consumed
        # cannot be derived from the step count.
        if bool(OmegaConf.select(self.config, "algorithm.filter_groups.enable", default=False)):
            print(
                "[resume][env] algorithm.filter_groups.enable=True: envs are reset more than "
                f"once per global step, so the resets consumed before step {self.global_steps} "
                "cannot be derived. Skipping the env schedule fast-forward - the resumed run "
                "will sample a different episode sequence than an uninterrupted run."
            )
            return

        if str(OmegaConf.select(self.config, "env.env_name", default="") or "").lower() == "multitask" and not bool(
            OmegaConf.select(self.config, "data.task_balance.enable", default=False)
        ):
            print(
                "[resume][env] WARNING: multitask envs without data.task_balance.enable - a batch "
                "may omit a task, so its resets may not equal the step count. Replaying anyway."
            )

        try:
            from agent_system.environments.resume import fast_forward_env_schedules

            messages = fast_forward_env_schedules(envs, self.global_steps)
        except Exception as exc:  # noqa: BLE001 - never block a resume
            print(f"[resume][env] WARNING: env schedule fast-forward unavailable ({exc!r}); skipping.")
            return

        for message in messages:
            print(f"[resume][env] resuming at step {self.global_steps} -> {message}")

    # 【日本語補足】各データ並列ランクのトークン総数が均等になるようサンプルを並べ替える
    # （計算負荷の偏り＝ストラグラー回避）。注意: これはバッチ内の順序を崩すため、
    # GRPO のようなグループ単位 advantage は「並べ替え前」に計算しておく必要がある。
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        # 各サンプルの系列長（有効トークン数）を求める。
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        # 系列長が均等になる分割を計算。
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        # 分割に沿ってサンプルを並べ替え（dispatch が自動で均等分配する）。
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    # 【日本語補足】これは「基底の PPO/GRPO 学習ループ」。
    # SDAR では SkillSDRayTrainer.fit() がこれを土台に、教師フォワードと蒸留損失を
    # 追加した版でオーバーライドしている（本メソッドは非SDARの学習で使われる）。
    # 1ステップの流れは skillsd 版とほぼ同じ:
    # ロールアウト → 報酬 → old_log_prob → (ref/critic) → advantage → actor更新 → 検証/保存。
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        # 何より先にチェックポイント復元（再開時）
        self._load_checkpoint()
        self._fast_forward_env_schedules()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # 学習前検証（val_before_train）。
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                # 検証のみモード
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        # ステップは 1 始まり
        self.global_steps += 1
        last_val_metrics = None

        # エポックループ
        for epoch in range(self.config.trainer.total_epochs):
            # DataLoader から1バッチずつ
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                # DataProto に変換
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                # 生成に渡すキーを取り出す（テンソル系＋非テンソル系）。
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                # ★同時学習: task_name を生成バッチへ引き渡す（env ルーティングに必要）。
                if "task_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("task_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # 最終ステップ判定
                is_last_step = self.global_steps >= self.total_training_steps

                # ステップ全体の時間計測
                with _timer("step", timing_raw):
                    # generate a batch
                    # ロールアウト（環境との複数ターン対話）
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        # 学習用環境と複数ターン対話して軌跡を生成（同時学習では task 別サブ環境へルーティング）。
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    # REMAX のみ: 貪欲生成のベースラインを別途作る（他推定方式では不要）。
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    # 以降はロールアウト結果を batch とする
                    batch = gen_batch_output

                    # GiGPO のみ: ステップ単位の割引リターンを事前計算。
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    # バッチ整形
                    batch = adjust_batch(self.config, batch)

                    # 応答マスク付与
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # 全トークン数
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 報酬計算
                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            # 報酬モデル使用時
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            # 非同期
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            # 同期
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    # 生成方策(student)の log-prob を再計算
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        old_log_prob_metrics = self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode)
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        # 参照方策の log-prob（KL 正則化用）
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                # 別ワーカ
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                # actor内兼用
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        # 価値関数（GAE 時のみ）
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # advantage 計算
                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        # 無効行動ペナルティを適用（★同時学習では task 別係数 by_task を渡す）。
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                                                                      "invalid_action_penalty_coef_by_task", None
                                                                                  ),
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        # 報酬に KL を混ぜる方式なら適用（SDAR は False なので通常は else 側）。
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # tag rows with their task so the actor can split its metrics
                    self._attach_task_ids(batch)

                    # update critic
                    if self.use_critic:
                        # 価値関数の更新（GAE 時のみ）
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # critic ウォームアップ期間を過ぎたら actor（方策）を更新。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # 方策の更新（1バックワード＋オプティマイザstep）
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # ※SkillSD 版ではこの直前に task 別 KL 係数テンソルを batch に載せる。
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    # 検証タイミング判定（最終ステップ or test_freq 間隔）。
                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # チェックポイント保存タイミング判定（最終ステップ or save_freq 間隔）。
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                # WandB/console へ出力
                logger.log(data=metrics, step=self.global_steps)

                # 進捗を1進める
                progress_bar.update(1)
                # ステップ番号を進める
                self.global_steps += 1
                # 最終ステップに達したら学習終了
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
