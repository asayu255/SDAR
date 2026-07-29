# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Tests for the per-task (multitask) breakdown of the logged metrics.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import unittest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_data_metrics_by_task,
    compute_metrics_by_task,
    compute_throughout_metrics,
    compute_trajectory_response_tokens,
    get_task_names,
    iter_task_row_masks,
    normalize_task_name,
    task_row_indices,
)


# [EXPLAIN] `_make_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_batch(task_names, batch_size=None, traj_uids=None, response_lengths=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A minimal rollout batch: 2 prompt tokens + 2 response tokens per row.

    ``traj_uids`` groups rows into trajectories (one row is one env turn), and
    ``response_lengths`` sets how many of the 2 response tokens each row uses.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = batch_size if batch_size is not None else len(task_names)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = torch.ones((n, 4), dtype=torch.long)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if response_lengths is not None:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for row, length in enumerate(response_lengths):
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            attention_mask[row, 2 + length :] = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensors = {
        "token_level_scores": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2),
        "token_level_rewards": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 2,
        "advantages": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 4,
        "returns": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 8,
        "values": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 16,
        "responses": torch.zeros((n, 2), dtype=torch.long),
        "attention_mask": attention_mask,
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensors = {
        "traj_uid": np.array(traj_uids if traj_uids is not None else [f"traj-{i}" for i in range(n)], dtype=object),
        "episode_rewards": np.arange(n, dtype=np.float32),
        "episode_lengths": np.arange(n, dtype=np.float32) + 1,
        "tool_callings": np.arange(n, dtype=np.float32) + 2,
        "success_rate": np.full(n, 0.25, dtype=np.float32),
    }
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_names is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        non_tensors["task_name"] = np.array(task_names, dtype=object)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


# [EXPLAIN] `TestNormalizeTaskName` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestNormalizeTaskName(unittest.TestCase):
    # [EXPLAIN] `test_canonical_tasks_match_by_substring` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_canonical_tasks_match_by_substring(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(normalize_task_name("alfworld"), "alfworld")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(normalize_task_name("AlfWorld_eval"), "alfworld")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(normalize_task_name("webshop_train"), "webshop")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(normalize_task_name("search"), "search")

    # [EXPLAIN] `test_unknown_task_keeps_its_own_bucket` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_unknown_task_keeps_its_own_bucket(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(normalize_task_name("Sokoban"), "sokoban")

    # [EXPLAIN] `test_missing_task_is_none` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_missing_task_is_none(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIsNone(normalize_task_name(None))


# [EXPLAIN] `TestTaskNameExtraction` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestTaskNameExtraction(unittest.TestCase):
    # [EXPLAIN] `test_task_names_from_task_name_field` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_names_from_task_name_field(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "webshop_train", "search"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.testing.assert_array_equal(get_task_names(batch), np.array(["alfworld", "webshop", "search"], dtype=object))

    # [EXPLAIN] `test_task_names_fall_back_to_env_kwargs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_names_fall_back_to_env_kwargs(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(None, batch_size=3)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"task_name": "alfworld"}, {"task_name": "search"}, {"task_name": "search"}], dtype=object
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.testing.assert_array_equal(get_task_names(batch), np.array(["alfworld", "search", "search"], dtype=object))

    # [EXPLAIN] `test_no_task_information_returns_none` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_no_task_information_returns_none(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIsNone(get_task_names(_make_batch(None, batch_size=3)))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIsNone(get_task_names(_make_batch([None, None])))

    # [EXPLAIN] `test_task_row_indices_groups_rows` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_row_indices_groups_rows(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["webshop", "alfworld", "webshop", None])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indices = task_row_indices(batch)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(sorted(indices), ["alfworld", "webshop"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.testing.assert_array_equal(indices["alfworld"], np.array([1]))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.testing.assert_array_equal(indices["webshop"], np.array([0, 2]))
        # the row without a task belongs to no task bucket
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(sum(len(rows) for rows in indices.values()), 3)

    # [EXPLAIN] `test_task_row_indices_empty_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_row_indices_empty_without_task_information(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(task_row_indices(_make_batch(None, batch_size=2)), {})


# [EXPLAIN] `TestComputeMetricsByTask` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeMetricsByTask(unittest.TestCase):
    # [EXPLAIN] `test_metric_fn_sees_only_the_rows_of_one_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_metric_fn_sees_only_the_rows_of_one_task(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "webshop", "alfworld"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_metrics_by_task(batch, lambda task_batch: {"rows": len(task_batch)})
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics, {"rows/alfworld": 2, "rows/webshop": 1})

    # [EXPLAIN] `test_no_per_task_metrics_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_no_per_task_metrics_without_task_information(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(compute_metrics_by_task(_make_batch(None, batch_size=2), lambda b: {"rows": len(b)}), {})


# [EXPLAIN] `TestComputeDataMetricsByTask` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeDataMetricsByTask(unittest.TestCase):
    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_names = ["alfworld", "webshop", "alfworld", "search"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = _make_batch(self.task_names)

    # [EXPLAIN] `test_every_overall_metric_has_a_per_task_counterpart` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_every_overall_metric_has_a_per_task_counterpart(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        overall = compute_data_metrics(self.batch, use_critic=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name in overall:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "success_rate" in name:  # batch-level constant, reported per task by the env manager
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for task in ("alfworld", "webshop", "search"):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.assertIn(f"{name}/{task}", per_task)

    # [EXPLAIN] `test_per_task_value_matches_the_metric_on_that_task_slice` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_per_task_value_matches_the_metric_on_that_task_slice(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        alfworld_rows = np.array([0, 2])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected = compute_data_metrics(self.batch.select_idxs(alfworld_rows), use_critic=True)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["critic/score/mean/alfworld"], expected["critic/score/mean"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["episode/reward/mean/alfworld"], expected["episode/reward/mean"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["response_length/mean/alfworld"], expected["response_length/mean"])
        # sanity: the alfworld slice is not just the overall value
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        overall = compute_data_metrics(self.batch, use_critic=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotAlmostEqual(per_task["critic/score/mean/alfworld"], overall["critic/score/mean"])

    # [EXPLAIN] `test_batch_level_success_rate_is_not_split` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_batch_level_success_rate_is_not_split(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotIn("episode/success_rate/alfworld", per_task)

    # [EXPLAIN] `test_single_task_batch_reports_the_overall_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_single_task_batch_reports_the_overall_value(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["search", "search"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_data_metrics_by_task(batch, use_critic=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        overall = compute_data_metrics(batch, use_critic=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["critic/score/mean/search"], overall["critic/score/mean"])

    # [EXPLAIN] `test_no_per_task_metrics_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_no_per_task_metrics_without_task_information(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(compute_data_metrics_by_task(_make_batch(None, batch_size=2), use_critic=False), {})


# [EXPLAIN] `TestTrainerLevelPerTaskMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestTrainerLevelPerTaskMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Per-task metrics produced on the driver, next to their overall counterpart."""

    # [EXPLAIN] `test_invalid_action_ratio_is_reported_per_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_invalid_action_ratio_is_reported_per_task(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "alfworld", "webshop", "webshop"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.batch["prompts"] = torch.zeros((4, 2), dtype=torch.long)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.non_tensor_batch["is_action_valid"] = np.array([True, False, True, True])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, metrics = apply_invalid_action_penalty(batch, invalid_action_penalty_coef=0.1)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/valid_action_ratio"], 0.75)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/valid_action_ratio/alfworld"], 0.5)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/valid_action_ratio/webshop"], 1.0)

    # [EXPLAIN] `test_reward_kl_penalty_is_reported_per_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_reward_kl_penalty_is_reported_per_task(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.core_algos import AdaptiveKLController
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "webshop"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.batch["old_log_probs"] = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.batch["ref_log_prob"] = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, metrics = apply_kl_penalty(batch, kl_ctrl=AdaptiveKLController(init_kl_coef=0.1, target_kl=0.1, horizon=1000), kl_penalty="kl")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["actor/reward_kl_penalty/alfworld"], 0.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["actor/reward_kl_penalty/webshop"], -1.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(
            metrics["actor/reward_kl_penalty"],
            (metrics["actor/reward_kl_penalty/alfworld"] + metrics["actor/reward_kl_penalty/webshop"]) / 2,
        )

    # [EXPLAIN] `test_entropy_loss_is_reported_per_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_entropy_loss_is_reported_per_task(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "webshop"])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropys = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_masks = torch.ones((2, 2))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = RayPPOTrainer._entropy_loss_metrics(batch, entropys, response_masks, "token-mean")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["actor/entropy_loss"], 2.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["actor/entropy_loss/alfworld"], 1.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["actor/entropy_loss/webshop"], 3.0)

    # [EXPLAIN] `test_task_ids_are_attached_for_the_workers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_ids_are_attached_for_the_workers(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["webshop", "alfworld", "webshop", None])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        RayPPOTrainer._attach_task_ids(batch)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(batch.meta_info["task_id_names"], ["alfworld", "webshop"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(batch.batch["task_ids"], torch.tensor([1, 0, 1, -1]))

    # [EXPLAIN] `test_task_ids_are_not_attached_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_task_ids_are_not_attached_without_task_information(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(None, batch_size=2)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        RayPPOTrainer._attach_task_ids(batch)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotIn("task_ids", batch.batch.keys())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotIn("task_id_names", batch.meta_info)


# [EXPLAIN] `TestIterTaskRowMasks` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestIterTaskRowMasks(unittest.TestCase):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """The worker-side (actor / critic) split of a micro-batch into task rows."""

    # [EXPLAIN] `test_rows_are_grouped_by_task_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rows_are_grouped_by_task_id(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_ids = torch.tensor([1, 0, 1])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        masks = dict(iter_task_row_masks(task_ids, ["alfworld", "webshop"]))

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(sorted(masks), ["alfworld", "webshop"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(masks["alfworld"], torch.tensor([False, True, False]))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(masks["webshop"], torch.tensor([True, False, True]))

    # [EXPLAIN] `test_rows_without_a_task_are_skipped` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rows_without_a_task_are_skipped(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        masks = dict(iter_task_row_masks(torch.tensor([-1, 0, 7]), ["alfworld"]))

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(list(masks), ["alfworld"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(masks["alfworld"], torch.tensor([False, True, False]))

    # [EXPLAIN] `test_nothing_is_yielded_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_nothing_is_yielded_without_task_information(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(list(iter_task_row_masks(None, ["alfworld"])), [])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(list(iter_task_row_masks(torch.tensor([0, 0]), None)), [])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(list(iter_task_row_masks(torch.tensor([0, 0]), [])), [])


# [EXPLAIN] `TestTrajectoryResponseTokens` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestTrajectoryResponseTokens(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """episode/response_tokens sums the per-turn rows of one trajectory."""

    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # two trajectories: alfworld spans 2 turns (2 + 1 tokens), webshop 1 turn (2 tokens)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = _make_batch(
            ["alfworld", "alfworld", "webshop"],
            traj_uids=["traj-a", "traj-a", "traj-b"],
            response_lengths=[2, 1, 2],
        )

    # [EXPLAIN] `test_tokens_are_summed_over_the_turns_of_a_trajectory` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tokens_are_summed_over_the_turns_of_a_trajectory(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.testing.assert_array_equal(compute_trajectory_response_tokens(self.batch), np.array([3.0, 2.0]))

    # [EXPLAIN] `test_metric_is_over_trajectories_not_turns` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_metric_is_over_trajectories_not_turns(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_data_metrics(self.batch, use_critic=False)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/response_tokens/mean"], 2.5)  # (3 + 2) / 2 trajectories
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/response_tokens/max"], 3.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["episode/response_tokens/min"], 2.0)
        # the per-turn length is a different quantity: (2 + 1 + 2) / 3 rows
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["response_length/mean"], 5 / 3)

    # [EXPLAIN] `test_metric_is_reported_per_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_metric_is_reported_per_task(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_data_metrics_by_task(self.batch, use_critic=False)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/alfworld"], 3.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/webshop"], 2.0)

    # [EXPLAIN] `test_opd_metrics_report_trajectory_tokens_per_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_opd_metrics_report_trajectory_tokens_per_task(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.trainer.ppo.opd_ray_trainer import compute_opd_data_metrics, compute_opd_data_metrics_by_task

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(compute_opd_data_metrics(self.batch)["episode/response_tokens/mean"], 2.5)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_task = compute_opd_data_metrics_by_task(self.batch)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/alfworld"], 3.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/webshop"], 2.0)


# [EXPLAIN] `TestComputeThroughputMetricsByTask` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeThroughputMetricsByTask(unittest.TestCase):
    # [EXPLAIN] `test_per_task_tokens_and_throughput_sum_to_the_overall_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_per_task_tokens_and_throughput_sum_to_the_overall_value(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(["alfworld", "webshop", "alfworld"])
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        batch.meta_info["global_token_num"] = [4, 4, 4]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_throughout_metrics(batch=batch, timing_raw={"step": 2.0}, n_gpus=2)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/total_num_tokens/alfworld"], 8)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/total_num_tokens/webshop"], 4)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(
            metrics["perf/throughput/alfworld"] + metrics["perf/throughput/webshop"],
            metrics["perf/throughput"],
        )

    # [EXPLAIN] `test_no_per_task_metrics_without_task_information` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_no_per_task_metrics_without_task_information(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = _make_batch(None, batch_size=2)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        batch.meta_info["global_token_num"] = [4, 4]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_throughout_metrics(batch=batch, timing_raw={"step": 1.0}, n_gpus=1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual([name for name in metrics if name.count("/") > 1], [])


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    unittest.main()
