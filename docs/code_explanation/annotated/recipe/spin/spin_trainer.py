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
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import traceback
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uuid
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import contextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass, field
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, Optional, Type

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from codetiming import Timer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf, open_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import Dataset, Sampler
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torchdata.stateful_dataloader import StatefulDataLoader
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from recipe.spin import core_algos
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import create_colocated_worker_cls
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics, process_validation_metrics, reduce_metrics
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.tracking import ValidationGenerationsLogger

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
WorkerType = Type[Worker]


# [EXPLAIN] `Role` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Role(Enum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    Actor = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    Rollout = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ActorRollout = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    Critic = 3
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    RefPolicy = 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    RewardModel = 5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ActorRolloutRef = 6


# [EXPLAIN] `AdvantageEstimator` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AdvantageEstimator(str, Enum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    GAE = 'gae'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    GRPO = 'grpo'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    REMAX = 'remax'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    RLOO = 'rloo'



# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `ResourcePoolManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ResourcePoolManager:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    resource_pool_spec: dict[str, list[int]]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    mapping: dict[Role, str]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    # [EXPLAIN] `create_resource_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def create_resource_pool(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.resource_pool_dict[resource_pool_name] = resource_pool

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._check_resource_available()

    # [EXPLAIN] `get_resource_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_resource_pool(self, role: Role) -> RayResourcePool:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the resource pool of the worker_cls"""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.resource_pool_dict[self.mapping[role]]

    # [EXPLAIN] `get_n_gpus` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_n_gpus(self) -> int:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the number of gpus in this cluster."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    # [EXPLAIN] `_check_resource_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _check_resource_available(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if the resource pool can be satisfied in this ray cluster."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        node_available_resources = ray.state.available_resources_per_node()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_available_gpus = sum(node_available_gpus.values())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if total_available_gpus < total_required_gpus:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for node, available_gpus in node_available_gpus.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if available_gpus >= num_gpus:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    node_available_gpus[node] -= num_gpus
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    num_nodes -= 1
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if num_nodes == 0:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        break
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if num_nodes > 0:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import masked_mean


# [EXPLAIN] `_compute_response_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Placeholder: Computes prompt and response lengths."""
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # Assuming 'prompts' and 'responses' keys exist after generation/union
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_len = batch.batch['prompts'].shape[1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        resp_len = batch.batch['responses'].shape[1]
        # This is simplified - real implementation might use attention masks
        # to get actual lengths per sample.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = batch.batch.batch_size[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_lengths_tensor = torch.full((batch_size,), prompt_len,
                                           dtype=torch.float32, device=batch.batch.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_lengths_tensor = torch.full((batch_size,), resp_len,
                                             dtype=torch.float32, device=batch.batch.device)

        # Try getting actual lengths from attention mask if possible (more accurate)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'response_mask' in batch.batch:
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             response_lengths_tensor = batch.batch['response_mask'].sum(dim=1).float()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'attention_mask' in batch.batch and 'response_mask' in batch.batch:
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             full_mask = batch.batch['attention_mask']
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             resp_mask = batch.batch['response_mask']
             # Infer prompt mask length based on where response mask starts or total length
             # This logic depends heavily on how your masks are constructed.
             # Example: prompt_lengths_tensor = full_mask.sum(dim=1).float() - response_lengths_tensor
             # Fallback to using prompt shape if mask logic is complex:
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             prompt_lengths_tensor = torch.tensor([batch.batch['prompts'].shape[1]] * batch_size,
                                                 dtype=torch.float32, device=batch.batch.device)


        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            'prompt_length': prompt_lengths_tensor,
            'response_length': response_lengths_tensor,
            'max_response_length': resp_len,
            'max_prompt_length': prompt_len # Or from config if fixed padding
        }
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except KeyError as e:
         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
         print(f"Warning: Missing key in _compute_response_info: {e}. Returning defaults.")
         # Return default/dummy values if keys are missing
         # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
         b_size = batch.batch.batch_size[0] if batch.batch.batch_size else 1
         # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
         max_resp = batch.batch.get('responses').shape[1] if batch.batch.get('responses') is not None else 0
         # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
         max_prompt = batch.batch.get('prompts').shape[1] if batch.batch.get('prompts') is not None else 0
         # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
         return {
            'prompt_length': torch.zeros(b_size), 'response_length': torch.zeros(b_size),
            'max_response_length': max_resp, 'max_prompt_length': max_prompt
         }


# --- Modified Metric Function ---
# [EXPLAIN] `compute_dpo_data_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_dpo_data_metrics(batch: DataProto) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes and returns metrics relevant for the DPO-like process.
    Assumes 'batch' contains results after generation and preference marking,
    potentially including 'dpo_logits', 'preferences', 'chosen_logps', etc.
    Removes PPO-specific advantage/return/critic metrics.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("---- [DEBUG] Computing DPO Data Metrics ----")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {}
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # --- Scores and Rewards (from reward_fn) ---
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'token_level_scores' in batch.batch and batch.batch['token_level_scores'] is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sequence_score = batch.batch['token_level_scores'].sum(-1)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics.update({
                'reward/score/mean': torch.mean(sequence_score).item(),
                'reward/score/max': torch.max(sequence_score).item(),
                'reward/score/min': torch.min(sequence_score).item(),
            })
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: print("DEBUG compute_dpo_data_metrics: 'token_level_scores' not found.")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'token_level_rewards' in batch.batch and batch.batch['token_level_rewards'] is not None:
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             sequence_reward = batch.batch['token_level_rewards'].sum(-1)
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             metrics.update({
                'reward/rewards/mean': torch.mean(sequence_reward).item(),
                'reward/rewards/max': torch.max(sequence_reward).item(),
                'reward/rewards/min': torch.min(sequence_reward).item(),
             })
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: print("DEBUG compute_dpo_data_metrics: 'token_level_rewards' not found.")

        # --- DPO Specific Metrics (if stored previously) ---
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'dpo_logits' in batch.batch and batch.batch['dpo_logits'] is not None:
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             metrics['actor/dpo_logits'] = batch.batch['dpo_logits'].mean().item()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: print("DEBUG compute_dpo_data_metrics: 'dpo_logits' not found.")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'chosen_logps' in batch.batch and batch.batch['chosen_logps'] is not None:
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             metrics['actor/chosen_logps'] = batch.batch['chosen_logps'].mean().item()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: print("DEBUG compute_dpo_data_metrics: 'chosen_logps' not found.")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'rejected_logps' in batch.batch and batch.batch['rejected_logps'] is not None:
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             metrics['actor/rejected_logps'] = batch.batch['rejected_logps'].mean().item()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else: print("DEBUG compute_dpo_data_metrics: 'rejected_logps' not found.")

        # Add metrics based on the 'preferences' mask if available
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if 'preferences' in batch.batch and batch.batch['preferences'] is not None:
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             prefs_mask = batch.batch['preferences'] # Shape [batch_size * n]
             # Calculate accuracy based on RM scores (assuming higher score -> True in mask)
             # Requires chosen/rejected scores to be available or recalculated
             # This is complex here, better calculated in the main loop or update function

        # --- Length Metrics ---
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_info = _compute_response_info(batch)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = response_info['prompt_length']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = response_info['response_length']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_response_length = response_info['max_response_length']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_prompt_length = response_info['max_prompt_length'] # Use calculated or from config

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metrics.update({
            'response_length/mean': torch.mean(response_length).item(),
            'response_length/max': torch.max(response_length).item(),
            'response_length/min': torch.min(response_length).item(),
            'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float()).item(),
            'prompt_length/mean': torch.mean(prompt_length).item(),
            'prompt_length/max': torch.max(prompt_length).item(),
            'prompt_length/min': torch.min(prompt_length).item(),
            # Prompt clip ratio might need adjustment based on how max_prompt_length is defined
            'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).item(),
        })

    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except KeyError as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"ERROR in compute_dpo_data_metrics: Missing key {e}")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"ERROR in compute_dpo_data_metrics: {e}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        traceback.print_exc()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"---- [DEBUG] Calculated DPO Data Metrics: {list(metrics.keys())} ----")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return metrics


# [EXPLAIN] `apply_kl_penalty` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    responses = data.batch['responses']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = responses.size(1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_level_scores = data.batch['token_level_scores']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = data.batch.batch_size[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = data.batch['attention_mask']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kld = kld * response_mask
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    beta = kl_ctrl.value

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_level_rewards = token_level_scores - beta * kld

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    data.batch['token_level_rewards'] = token_level_rewards

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data, metrics


# [EXPLAIN] `compute_response_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_response_mask(data: DataProto):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    responses = data.batch['responses']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = responses.size(1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = data.batch['attention_mask']
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attention_mask[:, -response_length:]


# [EXPLAIN] `compute_onlineDPO_pref` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_onlineDPO_pref(data: DataProto):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Wrapper to compute DPO preference and add it to the DataProto batch.
    Includes debugging prints.
    """
    # print(f"\n---- [DEBUG] Entering compute_onlineDPO_pref ----")
    # print(f"  Input batch keys: {list(data.batch.keys())}")

    # Check inputs
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rewards_tensor = data.batch.get('token_level_rewards')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask_tensor = data.batch.get('response_mask')

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rewards_tensor is None or mask_tensor is None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("  ERROR: Missing 'token_level_rewards' or 'response_mask' in input data!")
        # Handle error case - maybe return original data or raise?
        # Returning original data for now to potentially allow skipping
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        preferences = core_algos.compute_onlinedpo_pref(
            token_level_rewards=rewards_tensor,
            response_mask=mask_tensor
        )
        # Store the result
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch['preferences'] = preferences

    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
         print("ERROR: Function 'compute_online_dpo_preference' not found in core_algos.py!")
         # Assign dummy value or raise error
         # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
         data.batch['preferences'] = None # Indicate failure
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e_pref:
         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
         print(f"ERROR during core_algos.compute_online_dpo_preference: {e_pref}")
         # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
         import traceback
         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
         traceback.print_exc()
         # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
         data.batch['preferences'] = None # Indicate failure

    # print(f"---- [DEBUG] Exiting compute_onlineDPO_pref ----")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data



# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@contextmanager
# [EXPLAIN] `_timer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _timer(name: str, timing_raw: Dict[str, float]):
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name=name, logger=None) as timer:
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    timing_raw[name] = timer.last


# [EXPLAIN] `RaySPINTrainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RaySPINTrainer:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self,
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
        ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = processor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_fn = reward_fn
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.val_reward_fn = val_reward_fn

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.hybrid_engine:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.role_worker_mapping = role_worker_mapping
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.resource_pool_manager = resource_pool_manager
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_rm = Role.RewardModel in role_worker_mapping
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.ray_worker_group_cls = ray_worker_group_cls
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.validation_generations_logger = ValidationGenerationsLogger()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.async_rollout_mode = False

        # define in-reward KL control
        # kl loss control currently not suppoorted
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.algorithm.use_kl_in_reward:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        #     self.use_critic = True
        # elif self.config.algorithm.adv_estimator in [
        #         AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
        #         AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        # ]:
        #     self.use_critic = False
        # else:
        #     raise NotImplementedError
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_critic = False
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._validate_config()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # [EXPLAIN] `_validate_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _validate_config(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = self.config
        # number of GPUs total
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        # [EXPLAIN] `check_mutually_exclusive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if name in settings:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param = settings[name]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param_per_gpu = f"{param}_per_gpu"

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if mbs is None and mbs_per_gpu is None:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if mbs is not None and mbs_per_gpu is not None:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.critic.ppo_micro_batch_size is not None:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic and config.critic.strategy == 'fsdp':
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.data.get('val_batch_size', None) is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("[validate_config] All configuration checks passed successfully!")

    # [EXPLAIN] `_create_dataloader` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if train_dataset is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val_dataset is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if train_sampler is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if collate_fn is None:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            collate_fn = default_collate_fn

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val_batch_size is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_batch_size = len(self.val_dataset)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.trainer.total_training_steps is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_training_steps = self.config.trainer.total_training_steps

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.total_training_steps = total_training_steps
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Total training steps: {self.total_training_steps}")

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            OmegaConf.set_struct(self.config, True)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with open_dict(self.config):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if OmegaConf.select(self.config, "critic.optim"):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.config.critic.optim.total_training_steps = total_training_steps
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    # [EXPLAIN] `_maybe_log_val_generations` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _maybe_log_val_generations(self, inputs, outputs, scores):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generations_to_log = self.config.trainer.log_val_generations

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if generations_to_log == 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        samples = list(zip(inputs, outputs, scores))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng = np.random.RandomState(42)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        rng.shuffle(samples)

        # Take first N samples after shuffling
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        samples = samples[:generations_to_log]

        # Log to each configured logger
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    # [EXPLAIN] `_validate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _validate(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_source_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sample_inputs = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sample_outputs = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sample_scores = []

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for test_data in self.val_dataloader:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return {}

            # Store original inputs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            sample_inputs.extend(input_texts)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "raw_prompt" in test_batch.non_tensor_batch:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "tools_kwargs" in test_batch.non_tensor_batch:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not self.async_rollout_mode:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.async_rollout_manager.wake_up()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.async_rollout_manager.sleep()

            # unpad
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("validation generation end")

            # Store generated outputs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_ids = test_output_gen_batch.batch["responses"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            sample_outputs.extend(output_texts)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = self.val_reward_fn(test_batch, return_dict=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_tensor = result["reward_tensor"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scores = reward_tensor.sum(-1).cpu().tolist()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            sample_scores.extend(scores)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_extra_infos_dict["reward"].extend(scores)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "reward_extra_info" in result:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key, lst in result["reward_extra_info"].items():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    reward_extra_infos_dict[key].extend(lst)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val_data_dir:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key_info, lst in reward_extra_infos_dict.items():
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_sources = np.concatenate(data_source_lst, axis=0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"DEBUG: Data sources shape: {data_sources.shape}") # Added Print
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"DEBUG: reward_extra_infos_dict keys before processing: {reward_extra_infos_dict.keys()}") # Added Print

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"DEBUG: Output of process_validation_metrics (data_src2var2metric2val): {data_src2var2metric2val}") # Added Print
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metric_dict = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for data_source, var2metric2val in data_src2var2metric2val.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            core_var = "acc" if "acc" in var2metric2val else "reward"
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for var_name, metric2val in var2metric2val.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for metric_name, metric_val in metric2val.items():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        metric_sec = "val-core"
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        metric_sec = "val-aux"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    metric_dict[pfx] = metric_val

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return metric_dict
    
    # [EXPLAIN] `init_workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_workers(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Init resource pool and worker group"""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.resource_pool_manager.create_resource_pool()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.hybrid_engine:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError

        # create critic
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_reference_policy:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_rm:
            # we create a RM here
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_wg = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.wg_dicts = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.wg_dicts.append(wg_dict)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.critic_wg = all_wg['critic']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_wg.init_model()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_reference_policy:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ref_policy_wg = all_wg['ref']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.ref_policy_wg.init_model()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_rm:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rm_wg = all_wg['rm']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.actor_rollout_wg = all_wg['actor_rollout']
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_rollout_wg.init_model()

    # [EXPLAIN] `_save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'local_global_step_folder: {local_global_step_folder}')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if remove_previous_ckpt_in_save:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataloader_state_dict = self.train_dataloader.state_dict()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(local_latest_checkpointed_iteration, 'w') as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write(str(self.global_steps))

    # [EXPLAIN] `_load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _load_checkpoint(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.trainer.resume_mode == 'disable':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0

        # load from hdfs
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.trainer.default_hdfs_dir is not None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError('load from hdfs is not implemented yet')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not os.path.isabs(checkpoint_folder):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                working_dir = os.getcwd()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.trainer.resume_mode == 'auto':
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if global_step_folder is None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print('Training from scratch')
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return 0
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.trainer.resume_mode == "resume_path":
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                global_step_folder = self.config.trainer.resume_from_path
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not os.path.isabs(global_step_folder):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    working_dir = os.getcwd()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'Setting global step to {self.global_steps}')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f'Resuming from {global_step_folder}')

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_path = os.path.join(global_step_folder, 'actor')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_critic:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.exists(dataloader_local_path):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    # [EXPLAIN] `_balance_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = batch.batch['attention_mask']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = attention_mask.shape[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = self.actor_rollout_wg.world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.reorder(global_idx)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metrics.update(global_balance_stats)

        
    # [EXPLAIN] `fit_dpo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fit_dpo(self): # Renamed for clarity as standard PPO loop
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        The training loop of Online DPO using a periodically updated reference model.
        The driver process calls worker groups for computation.
        Advantage computation is replaced by DPO logic.
        """
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import traceback  # Ensure traceback is imported

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.tracking import Tracking

        # Initialize logger
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logger = None
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logger = Tracking(project_name=self.config.trainer.project_name,
                              experiment_name=self.config.trainer.experiment_name,
                              default_backend=self.config.trainer.logger,
                              config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False))
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: Failed to initialize logger: {e}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.global_steps = 0
        # Load checkpoint before doing anything
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loaded_step = self._load_checkpoint()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Starting Online DPO training from global step {self.global_steps}. Total steps: {self.total_training_steps}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Reference model update frequency: {self.config.trainer.get('ref_update_freq', 'Not Set')}")

        # Check if reference policy is configured correctly for this mode
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.use_reference_policy:
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             print("WARNING: 'use_reference_policy' is False. Periodic reference model update requires a reference policy worker. DPO updates might fail or use incorrect logic.")
             # Consider raising an error if strict adherence is required:
             # raise ValueError("Periodic reference model update requires 'use_reference_policy' to be True and a configured reference worker.")


        # Perform validation before training
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Running validation before Online DPO training...")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val_metrics = self._validate()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            pprint(f'Initial validation metrics: {val_metrics}')
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if logger and val_metrics: logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.trainer.get('val_only', False):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("Validation only mode enabled. Exiting training.")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if logger and hasattr(logger, 'finish'): logger.finish()
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return

        # Add tqdm progress bar
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Online DPO Training Progress", position=0, leave=True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        last_val_metrics = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        should_stop = False

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for epoch in range(self.config.trainer.total_epochs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if should_stop: break
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"--- Starting Online DPO Epoch {epoch} ---")
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                train_iterator = iter(self.train_dataloader)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except TypeError:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("Warning: Dataloader is not iterable.")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                train_iterator = self.train_dataloader # Fallback attempt

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for batch_idx, batch_dict in enumerate(train_iterator):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.global_steps > self.total_training_steps:
                      # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                      should_stop = True; break

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metrics = {}
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                timing_raw = {}
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                step_timer = Timer(logger=None)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ref_log_prob_computed = False # Flag to track if ref log probs were computed

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try: # Outer try-except for the whole step
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    step_timer.start()
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with _timer('step', timing_raw):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        current_batch_size = batch.batch.batch_size[0]
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"\n[Step {self.global_steps}, Batch {batch_idx}] Processing batch size: {current_batch_size}")

                        # --- Reference Model Update ---
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        ref_update_freq = self.config.trainer.get('ref_update_freq', -1)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.use_reference_policy and ref_update_freq > 0 and self.global_steps % ref_update_freq == 0:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"\n[Step {self.global_steps}] Updating Reference Model Weights from Actor...")
                            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                            try:
                                # --- This requires careful implementation with FSDP ---
                                # 1. Save actor state dict (potentially to CPU memory or disk)
                                #    This needs to be done collectively across actor worker ranks.
                                #    The checkpoint_manager might be adaptable, or use FSDP APIs directly.
                                #    Example placeholder using a conceptual save/load mechanism:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                actor_state_path = "/tmp/actor_state_mid" # Temporary path
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                self.actor_rollout_wg.save_checkpoint(actor_state_path) # Adapt save logic

                                # 2. Load the state dict onto the reference model worker group
                                #    This also needs collective loading on the ref worker ranks.
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                self.ref_policy_wg.load_checkpoint(actor_state_path,None, True) # Adapt load logic

                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(f"[Step {self.global_steps}] Reference Model Weights Updated.")
                                # Optionally remove the temporary state file
                                # os.remove(actor_state_path) # Needs rank-aware removal or shared storage

                            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                            except Exception as sync_e:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(f"ERROR during reference model sync at step {self.global_steps}: {sync_e}")
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                traceback.print_exc()

                        # Pop keys for generation
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        pop_batch_keys=['input_ids', 'attention_mask']
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if 'position_ids' in batch.batch: pop_batch_keys.append('position_ids')
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        pop_non_tensor_keys = ['raw_prompt_ids'] if 'raw_prompt_ids' in batch.non_tensor_batch else []
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            pop_non_tensor_keys.extend(['multi_modal_data', 'multi_modal_inputs'])
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        original_non_tensor_data = batch.non_tensor_batch
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        gen_batch = batch.pop(
                            batch_keys=pop_batch_keys,
                            non_tensor_batch_keys=pop_non_tensor_keys,
                        )
                        # (Add Debug prints for gen_batch if needed)

                        # Generate sequences (chosen/rejected pairs)
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer('gen', timing_raw):
                            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                            try:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                # (Add Debug prints for gen_batch_output if needed)
                            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                            except Exception as gen_e:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(f"\n!!!!!!!! ERROR DURING GENERATION (Step {self.global_steps}) !!!!!!!!")
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(gen_e); traceback.print_exc()
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                step_timer.stop(); continue

                        # Combine original prompts with generated sequences
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch.non_tensor_batch = original_non_tensor_data # Restore non-tensor data
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(current_batch_size)], dtype=object)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = batch.union(gen_batch_output)
                        # (Add Debug prints after union if needed)

                        # Compute response mask (needed for ref logprob calc and DPO prep)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        batch.batch['response_mask'] = compute_response_mask(batch)

                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.trainer.balance_batch:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            self._balance_batch(batch, metrics=metrics)

                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                        # --- Compute Log Probs for the CURRENT policy (used for KL if enabled, or ActorAsRef fallback) ---
                        # Note: For pure DPO with external ref, this 'old_log_probs' might not be strictly needed
                        #       unless used for other metrics or a fallback. Keep it for now.
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer('policy_log_prob', timing_raw):
                             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                             policy_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                             batch = batch.union(policy_log_prob_output) # Adds 'old_log_probs'
                             # (Debug prints for old_log_probs)

                        # --- Compute Log Probs using the EXTERNAL Reference Model ---
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.use_reference_policy:
                            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                            with _timer('ref_log_prob_dpo', timing_raw):
                                # print(f"---- [Step {self.global_steps}] DEBUG DPO: Calling compute_ref_log_prob ----")
                                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                                try:
                                    # 'batch' contains interleaved chosen/rejected sequences
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(batch) # Returns DataProto with 'ref_log_prob'
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    batch = batch.union(ref_log_prob_output) # Adds 'ref_log_prob' key [batch_size * n, seq_len]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    ref_log_prob_computed = True # Mark success
                                    # print(f"---- [Step {self.global_steps}] DEBUG DPO: ref_log_prob tensor shape: {batch.batch['ref_log_prob'].shape} ----")
                                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                                except Exception as ref_e:
                                     # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                     print(f"ERROR computing reference log probs at step {self.global_steps}: {ref_e}")
                                     # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                     traceback.print_exc()
                                     # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                                     batch.batch['ref_log_prob'] = None # Mark as failed
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     ref_log_prob_computed = False
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print("Warning: Skipping external reference log prob calculation as use_reference_policy is False.")
                            # DPO update will likely fail unless ActorAsRef logic is re-enabled in dp_actor


                        # --- Compute Rewards/Scores (used to determine preference) ---
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer('reward_calc', timing_raw):
                             # (Reward calculation logic using RM or reward_fn as before)
                             # ... Ensure this calculates 'token_level_rewards' or similar ...
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if self.use_rm:
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                reward_tensor_rm = self.rm_wg.compute_rm_score(batch)
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                batch = batch.union(reward_tensor_rm) # Adds 'rm_scores'

                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            reward_extra_infos_dict = {}
                            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                            try:
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.reward_fn is None:
                                    #  print(f"---- [DEBUG Step {self.global_steps}] ERROR: self.reward_fn is None! Using dummy rewards. ----")
                                     # Use rm_scores if available, otherwise zeros
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reward_tensor = batch.batch.get('rm_scores', torch.zeros_like(batch.batch['response_mask'], dtype=torch.float32))
                                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                                else:
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reward_result = self.reward_fn(batch, return_dict=True)
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reward_tensor = reward_result['reward_tensor'] # Final combined reward
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reward_extra_infos_dict = reward_result.get('reward_extra_info', {})

                            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                            except Exception:
                                # print(f'---- [DEBUG Step {self.global_steps}] Error in reward_fn call: {e}. Using dummy rewards. ----')
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                traceback.print_exc()
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                reward_tensor = torch.zeros_like(batch.batch['response_mask'], dtype=torch.float32)
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                reward_extra_infos_dict = {}

                            # Use 'token_level_rewards' as the key for preference calculation
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            batch.batch['token_level_rewards'] = reward_tensor
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if reward_extra_infos_dict: batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            

                        # --- Determine Preferences ---
                        # Uses 'token_level_rewards' to determine chosen/rejected based on score
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        batch = compute_onlineDPO_pref(batch) # Adds 'preferences' key

                        # --- Prepare DPO Batch ---
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        dpo_update_batch_proto = None # Initialize
                        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                        with _timer('prepare_dpo_batch', timing_raw):
                            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                            try:
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if 'preferences' not in batch.batch or batch.batch['preferences'] is None:
                                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                                    raise ValueError("'preferences' key missing or None after compute_onlineDPO_pref.")

                                # Check if reference log probs were computed successfully (if needed)
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.use_reference_policy and not ref_log_prob_computed:
                                     # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                                     raise ValueError("Reference log probs required but failed to compute.")

                                # Check required base keys
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                required_keys = ['input_ids', 'attention_mask', 'response_mask']
                                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                                for rk in required_keys:
                                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                    if rk not in batch.batch or batch.batch[rk] is None:
                                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                                        raise KeyError(f"Required key '{rk}' missing from batch for DPO prep.")

                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                preferences_mask = batch.batch['preferences'] # Shape [batch_size * n]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                not_preferences_mask = ~preferences_mask

                                # Gather Chosen/Rejected Base Tensors
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                chosen_input_ids = batch.batch['input_ids'][preferences_mask]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                chosen_attention_mask = batch.batch['attention_mask'][preferences_mask]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                rejected_input_ids = batch.batch['input_ids'][not_preferences_mask]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                rejected_attention_mask = batch.batch['attention_mask'][not_preferences_mask]
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                chosen_position_ids = batch.batch.get('position_ids')[preferences_mask] if 'position_ids' in batch.batch else None
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                rejected_position_ids = batch.batch.get('position_ids')[not_preferences_mask] if 'position_ids' in batch.batch else None

                                # Create Labels
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print("WARNING: Creating DPO labels using configured max_prompt_length...")
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                prompt_len = self.config.data.max_prompt_length
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                chosen_labels = chosen_input_ids.clone(); chosen_labels[:, :prompt_len] = -100
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                rejected_labels = rejected_input_ids.clone(); rejected_labels[:, :prompt_len] = -100

                                # Calculate and Gather Reference Log Probs (Sequence Level)
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if self.use_reference_policy:
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    ref_log_prob_tensor = batch.batch['ref_log_prob'] # Token level [bsz * n, seq_len]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    response_mask_full = batch.batch['response_mask'] # Response mask [bsz * n, seq_len]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    ref_sequence_logps = (ref_log_prob_tensor * response_mask_full).sum(dim=-1) # Sequence level [bsz * n]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    reference_chosen_logps = ref_sequence_logps[preferences_mask]
                                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                    reference_rejected_logps = ref_sequence_logps[not_preferences_mask]
                                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                                else:
                                     # If not using external ref, DPO needs ActorAsRef logic in dp_actor
                                     # We won't add the keys here, dp_actor will handle it (or fail if not modified)
                                     # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                     print("Info: Not adding explicit reference logps to DPO batch (use_reference_policy=False).")
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reference_chosen_logps = None # Explicitly None
                                     # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                     reference_rejected_logps = None

                                # Package Tensors
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                dpo_tensors = {
                                     'chosen_input_ids': chosen_input_ids,
                                     'chosen_attention_mask': chosen_attention_mask,
                                     'chosen_labels': chosen_labels,
                                     'rejected_input_ids': rejected_input_ids,
                                     'rejected_attention_mask': rejected_attention_mask,
                                     'rejected_labels': rejected_labels,
                                }
                                # Conditionally add reference logps if computed
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if reference_chosen_logps is not None:
                                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                                    dpo_tensors['reference_chosen_logps'] = reference_chosen_logps
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if reference_rejected_logps is not None:
                                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                                    dpo_tensors['reference_rejected_logps'] = reference_rejected_logps
                                # Add position ids if they exist
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if chosen_position_ids is not None: dpo_tensors['chosen_position_ids'] = chosen_position_ids
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if rejected_position_ids is not None: dpo_tensors['rejected_position_ids'] = rejected_position_ids

                                # Prepare Meta Info
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                dpo_meta = {
                                     'dpo_beta': OmegaConf.select(self.config.algorithm, "dpo_beta", default=0.1),
                                     'dpo_loss_type': OmegaConf.select(self.config.algorithm, "dpo_loss_type", default='sigmoid'),
                                     'dpo_label_smoothing': OmegaConf.select(self.config.algorithm, "dpo_label_smoothing", default=0.0),
                                     'use_reference_policy': self.use_reference_policy,
                                     'reference_free': not self.use_reference_policy, # False if using external ref
                                     'global_step': self.global_steps
                                }

                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                dpo_update_batch_proto = DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)
                                # print(f"---- [Step {self.global_steps}] DEBUG DPO: Prepared DPO Update Batch ----")
                                # print(f"  Keys: {list(dpo_update_batch_proto.batch.keys())}")
                                # print(f"  Meta Info: {dpo_meta}")

                            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                            except Exception as e_prep:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                print(f"ERROR preparing DPO batch at step {self.global_steps}: {e_prep}")
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                traceback.print_exc()
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                dpo_update_batch_proto = None # Skip update on error


                        # --- Actor Update Step ---
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        actor_output = None
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.config.trainer.critic_warmup <= self.global_steps and dpo_update_batch_proto:
                            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                            with _timer('update_actor', timing_raw):
                                # Pass the batch containing reference log probs (if computed)
                                # The modified update_actor_dpo expects them if reference_free=False
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                actor_output = self.actor_rollout_wg.update_actor_dpo(dpo_update_batch_proto)
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if actor_output and 'metrics' in actor_output.meta_info:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                metrics.update(reduce_metrics(actor_output.meta_info['metrics']))
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        elif dpo_update_batch_proto is None:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"Skipping actor update at step {self.global_steps} due to DPO batch preparation error.")


                        # --- Validation and Saving ---
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        test_freq = OmegaConf.select(self.config.trainer, "test_freq", default = -1)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        is_last_step = self.global_steps >= self.total_training_steps
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.val_reward_fn is not None and test_freq > 0 and (is_last_step or self.global_steps % test_freq == 0):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"\nRunning DPO validation at step {self.global_steps}...")
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            val_timing_raw = {}
                            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                            with _timer('testing', val_timing_raw):
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                val_metrics: dict = self._validate()
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if is_last_step: last_val_metrics = val_metrics
                            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                            if val_metrics:
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                metrics['time/validation_run'] = val_timing_raw.get('testing', 0)
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                metrics.update(val_metrics)
                            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                            else: print("Validation skipped or returned no metrics.")

                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default = -1)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if save_freq > 0 and ( is_last_step or self.global_steps % save_freq == 0):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            print(f"\nSaving DPO checkpoint at step {self.global_steps}...")
                            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                            with _timer('save_checkpoint', timing_raw):
                                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                                self._save_checkpoint() # Saves actor (and potentially critic if used elsewhere)
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            metrics['time/save_checkpoint'] = timing_raw.get('save_checkpoint', 0)

                    # --- End main step timer context ---

                    # --- Metrics calculation AFTER the 'step' timer block ---
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    metrics.update(compute_dpo_data_metrics(batch=batch)) # Use DPO-specific metrics
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if 'step' in timing_raw:
                         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                         metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                         # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                         print(f"Warning: 'step' key missing from timing_raw at step {self.global_steps}. Skipping throughput.")

                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    step_timer.stop()
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metrics['time/step'] = step_timer.last

                    # Log metrics
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    log_freq = OmegaConf.select(self.config.trainer, "log_freq", default = 1)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if logger and self.global_steps % log_freq == 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        log_payload = metrics.copy()
                        # Add learning rate to log payload
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if actor_output and 'actor/lr' in metrics: log_payload['actor/lr'] = metrics['actor/lr']

                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"[Step {self.global_steps} DPO] Logging Step Payload Keys: {list(log_payload.keys())}")
                        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                        try: logger.log(data=log_payload, step=self.global_steps)
                        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                        except Exception as e: print(f"Logging failed at step {self.global_steps}: {e}")

                    # Update progress bar
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    postfix_metrics = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in metrics.items() if isinstance(v, (int, float))}
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    progress_bar.set_postfix(postfix_metrics)

                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as step_e:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"\n!!!!!!!! ERROR DURING DPO Step {self.global_steps} !!!!!!!!")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Caught Exception: {step_e}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    traceback.print_exc()
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    step_timer.stop(); should_stop = True; break

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if is_last_step or should_stop:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f'Stopping DPO training at step {self.global_steps}.')
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.global_steps += 1
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                progress_bar.update(1)

            # End of epoch handling
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(self.train_dataloader, 'reset'):
                 # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                 try: self.train_dataloader.reset()
                 # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                 except Exception as e: print(f"Warning: Failed to reset train dataloader state: {e}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if should_stop: break

        # --- Final cleanup and logging ---
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        progress_bar.close()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_step = max(0, self.global_steps - 1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Online DPO Training finished at step {final_step}.")
        # Save final checkpoint
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default = -1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.config.trainer.get('val_only', False) and (save_freq <= 0 or final_step % save_freq != 0) :
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saving final DPO checkpoint at step {final_step}...")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._save_checkpoint()

        # Final validation run
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.val_reward_fn and last_val_metrics is None and not self.config.trainer.get('val_only', False):
             # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
             print("Running final validation...")
             # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
             last_val_metrics = self._validate()
             # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
             if last_val_metrics and logger:
                 # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                 last_val_metrics['final_validation'] = True
                 # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                 try: logger.log(data=last_val_metrics, step=final_step)
                 # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                 except Exception as e: print(f"[Final Val Metrics Log Error]: {e}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        pprint(f'Final validation metrics: {last_val_metrics}')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if logger and hasattr(logger, 'finish'): logger.finish()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Online DPO Training Run Complete.")
    
