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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Metrics related to the PPO trainer.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Dict, List, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.import_utils import deprecated

# Every per-task metric is the overall metric name with the task appended as the
# last path segment (e.g. "critic/score/mean" -> "critic/score/mean/alfworld"),
# so the wandb panel that holds the overall metric also holds its per-task
# breakdown.
# Canonical multitask names. A raw task_name is matched by substring so that
# dataset-specific spellings ("alfworld_eval", "webshop_train", ...) collapse to
# the same bucket the trainers already route on.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CANONICAL_TASK_NAMES = ("alfworld", "webshop", "search")

# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@deprecated("verl.utils.metric.reduce_metrics")
# [EXPLAIN] `reduce_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.metric import reduce_metrics

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reduce_metrics(metrics)


# [EXPLAIN] `normalize_task_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_task_name(task_name: Any) -> Optional[str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Map a raw ``task_name`` onto its canonical multitask bucket.

    Returns ``None`` for a missing name, and the (lower-cased) name itself when
    it matches none of the canonical tasks, so unknown tasks still get their own
    metric bucket instead of being silently merged.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_name is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_name = str(task_name).lower()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for canonical in CANONICAL_TASK_NAMES:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if canonical in task_name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return canonical
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return task_name


# [EXPLAIN] `get_task_names` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_task_names(batch: DataProto) -> Optional[np.ndarray]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Row-aligned canonical task names for a batch, or ``None`` when unavailable.

    Single-task runs carry no ``task_name`` (nor ``env_kwargs['task_name']``), in
    which case every caller falls back to logging only the overall metrics.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_names = batch.non_tensor_batch.get("task_name", None)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_names is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_kwargs = batch.non_tensor_batch.get("env_kwargs", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if env_kwargs is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_names = [item.get("task_name") if isinstance(item, dict) else None for item in env_kwargs]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_names is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    normalized = np.array([normalize_task_name(task_name) for task_name in task_names], dtype=object)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if all(task_name is None for task_name in normalized):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return normalized


# [EXPLAIN] `task_row_indices` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
# [EXPLAIN] task 名を正規化した後に global batch の row index を task ごとに保持する。
# [EXPLAIN] DataProto の select_idxs は tensor/non-tensor を同じ index で切るため、
# [EXPLAIN] per-task metric でも両者の row alignment が崩れない。
def task_row_indices(batch: DataProto) -> Dict[str, np.ndarray]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Map each task present in the batch to the row indices belonging to it.

    Rows without a usable task name are dropped (they belong to no task), and the
    mapping is empty when the batch carries no task information at all.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_names = get_task_names(batch)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_names is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    indices: Dict[str, List[int]] = defaultdict(list)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for row, task_name in enumerate(task_names):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_name is None:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        indices[task_name].append(row)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {task: np.array(rows, dtype=np.int64) for task, rows in sorted(indices.items())}


# [EXPLAIN] `iter_task_row_masks` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def iter_task_row_masks(task_ids: "torch.Tensor", task_id_names: List[str]):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Yield ``(task, row_mask)`` for every task present in a worker micro-batch.

    The workers only see the integer ``task_ids`` column attached by
    ``RayPPOTrainer._attach_task_ids``; ``task_id_names`` maps it back to the task
    name. Ids outside that mapping (``-1`` for rows whose task name was missing)
    are skipped.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_ids is None or not task_id_names:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for task_id in torch.unique(task_ids).tolist():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_id < 0 or task_id >= len(task_id_names):
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield task_id_names[task_id], task_ids == task_id


# [EXPLAIN] `with_task_suffix` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def with_task_suffix(metrics: Dict[str, Any], task: str) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Rename metrics to their per-task variant, e.g. ``a/b`` -> ``a/b/{task}``."""
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {f"{name}/{task}": value for name, value in metrics.items()}


# [EXPLAIN] `compute_metrics_by_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_metrics_by_task(batch: DataProto, metric_fn: Callable[[DataProto], Dict[str, Any]]) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Run ``metric_fn`` on each single-task slice of the batch.

    The same function that produces the overall metrics is re-run on the rows of
    one task at a time, so every metric it reports gains a per-task counterpart
    with identical semantics.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_task_metrics = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for task, rows in task_row_indices(batch).items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        per_task_metrics.update(with_task_suffix(metric_fn(batch.select_idxs(rows)), task))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return per_task_metrics


# [EXPLAIN] `_drop_batch_level_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _drop_batch_level_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Drop metrics that are batch-wide constants broadcast onto every row.

    Success rates are computed by the env manager over the whole rollout and then
    copied to every row, so slicing rows by task cannot recover a per-task value.
    The multitask env manager already reports them per task as
    ``episode/{task}_success_rate``.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {name: value for name, value in metrics.items() if "success_rate" not in name}


# [EXPLAIN] `compute_data_metrics_by_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_data_metrics_by_task(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Per-task breakdown of :func:`compute_data_metrics`."""
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return compute_metrics_by_task(
        batch,
        lambda task_batch: _drop_batch_level_metrics(compute_data_metrics(task_batch, use_critic=use_critic)),
    )


# [EXPLAIN] `_compute_response_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes information about prompts and responses from a batch.
    
    This is an internal helper function that extracts masks and lengths for prompts and responses.
    
    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        
    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = batch.batch["responses"].shape[-1]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_length = prompt_mask.sum(-1).float()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


# [EXPLAIN] `compute_trajectory_response_tokens` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
# [EXPLAIN] multi-turn では一 trajectory が複数 row になるため、row 単位の response_length と
# [EXPLAIN] trajectory 単位の総 response token を分ける。traj_uid が aggregation key である。
def compute_trajectory_response_tokens(batch: DataProto) -> Optional[np.ndarray]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Generated tokens per trajectory, i.e. per sample rather than per turn.

    A row of the batch is one env turn, so the row-level ``response_length`` is a
    per-turn length. Summing the rows that share a ``traj_uid`` gives the tokens a
    whole sample generated across its turns. Returns ``None`` when the batch
    carries no trajectory ids.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    traj_uid = batch.non_tensor_batch.get("traj_uid", None)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(traj_uid, np.ndarray) or traj_uid.shape[0] != len(batch):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = _compute_response_info(batch)["response_length"].cpu().numpy()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, trajectory_of_row = np.unique(traj_uid, return_inverse=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return np.bincount(trajectory_of_row.reshape(-1), weights=response_length)


# [EXPLAIN] `compute_data_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    advantages = batch.batch["advantages"]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    returns = batch.batch["returns"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = batch.batch["responses"].shape[-1]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = prompt_mask.size(-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_info = _compute_response_info(batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_length = response_info["prompt_length"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = response_info["response_length"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    valid_adv = torch.masked_select(advantages, response_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    valid_returns = torch.masked_select(returns, response_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unique_traj_uid, unique_idx = np.unique(batch.non_tensor_batch['traj_uid'], return_index=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trajectory_response_tokens = compute_trajectory_response_tokens(batch)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if use_critic:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        values = batch.batch["values"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_values = torch.masked_select(values, response_mask)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return_diff_var = torch.var(valid_returns - valid_values)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return_var = torch.var(valid_returns)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # episode
        "episode/reward/mean": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].mean().item(),
        "episode/reward/max": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].max().item(),
        "episode/reward/min": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].min().item(),
        "episode/length/mean": 
            batch.non_tensor_batch["episode_lengths"][unique_idx].mean().item(),
        "episode/length/max":
            batch.non_tensor_batch["episode_lengths"][unique_idx].max().item(),
        "episode/length/min": 
            batch.non_tensor_batch["episode_lengths"][unique_idx].min().item(),
        # tokens a whole trajectory generated (response_length above is per turn)
        **(
            {
                "episode/response_tokens/mean": float(trajectory_response_tokens.mean()),
                "episode/response_tokens/max": float(trajectory_response_tokens.max()),
                "episode/response_tokens/min": float(trajectory_response_tokens.min()),
            }
            if trajectory_response_tokens is not None
            else {}
        ),
        "episode/tool_call_count/mean":
            batch.non_tensor_batch["tool_callings"][unique_idx].mean().item(),
        # "episode/tool_call_count/max":
        #     batch.non_tensor_batch["tool_callings"][unique_idx].max().item(),
        # "episode/tool_call_count/min":
        #     batch.non_tensor_batch["tool_callings"][unique_idx].min().item(),
        **({f"episode/{k}": v[0].item() for k, v in batch.non_tensor_batch.items() if "success_rate" in k}),
    }
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return metrics


# [EXPLAIN] `compute_timing_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes timing metrics for different processing stages in PPO training.
    
    This function calculates both raw timing metrics (in seconds) and per-token timing metrics 
    (in milliseconds) for various processing stages like generation, reference computation, 
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_info = _compute_response_info(batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


# [EXPLAIN] `compute_throughout_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Computes throughput metrics for PPO training.
    
    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).
    
    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
            
    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics = {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }

    # Per-task token counts / throughput share. The step time itself covers all
    # tasks at once and cannot be attributed, so only the token-derived metrics
    # are split; the per-task throughputs sum to the overall one.
    # [EXPLAIN] wall-clock は混合 batch 全体の共有分母なので task 別時間とは解釈しない。
    # [EXPLAIN] task throughput は同じ分母へ各 task の token 数を割った寄与率である。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_num = np.asarray(batch.meta_info["global_token_num"])
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if token_num.shape[0] == len(batch):  # row-aligned, so it can be split by task
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task, rows in task_row_indices(batch).items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_num_tokens = int(token_num[rows].sum())
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metrics[f"perf/total_num_tokens/{task}"] = task_num_tokens
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metrics[f"perf/throughput/{task}"] = task_num_tokens / (time * n_gpus)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return metrics


# [EXPLAIN] `bootstrap_metric` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    np.random.seed(seed)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _ in range(n_bootstrap):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, reduce_fn in enumerate(reduce_fns):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


# [EXPLAIN] `calc_maj_val` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vote2vals = defaultdict(list)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for d in data:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        vote2vals[d[vote_key]].append(d[val_key])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    maj_val = vote2vals[maj_vote][0]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return maj_val


# [EXPLAIN] `process_validation_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Process validation metrics into a structured format with statistical analysis.
    
    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for sample_idx, data_source in enumerate(data_sources):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt = sample_inputs[sample_idx]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for var_name, var_vals in infos_dict.items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for prompt, var2vals in prompt2var2vals.items():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for var_name, var_vals in var2vals.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(var_vals[0], str):
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metric = {}
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                n_resps = len(var_vals)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if n_resps > 1:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    ns = []
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    n = 2
                    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
                    while n < n_resps:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        ns.append(n)
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        n *= 2
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    ns.append(n_resps)

                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for n in ns:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if var2vals.get("pred", None) is not None:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for prompt, var2metric in prompt2var2metric.items():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for var_name, metric in var2metric.items():
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for metric_name, metric_val in metric.items():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for metric_name, prompt_vals in metric2prompt_vals.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data_src2var2metric2val
