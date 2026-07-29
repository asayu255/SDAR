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
"""
Tests for the per-task (multitask) breakdown of the logged metrics.
"""

import unittest

import numpy as np
import torch

from verl import DataProto
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


def _make_batch(task_names, batch_size=None, traj_uids=None, response_lengths=None):
    """A minimal rollout batch: 2 prompt tokens + 2 response tokens per row.

    ``traj_uids`` groups rows into trajectories (one row is one env turn), and
    ``response_lengths`` sets how many of the 2 response tokens each row uses.
    """
    n = batch_size if batch_size is not None else len(task_names)
    attention_mask = torch.ones((n, 4), dtype=torch.long)
    if response_lengths is not None:
        for row, length in enumerate(response_lengths):
            attention_mask[row, 2 + length :] = 0
    tensors = {
        "token_level_scores": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2),
        "token_level_rewards": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 2,
        "advantages": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 4,
        "returns": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 8,
        "values": torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 16,
        "responses": torch.zeros((n, 2), dtype=torch.long),
        "attention_mask": attention_mask,
    }
    non_tensors = {
        "traj_uid": np.array(traj_uids if traj_uids is not None else [f"traj-{i}" for i in range(n)], dtype=object),
        "episode_rewards": np.arange(n, dtype=np.float32),
        "episode_lengths": np.arange(n, dtype=np.float32) + 1,
        "tool_callings": np.arange(n, dtype=np.float32) + 2,
        "success_rate": np.full(n, 0.25, dtype=np.float32),
    }
    if task_names is not None:
        non_tensors["task_name"] = np.array(task_names, dtype=object)
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


class TestNormalizeTaskName(unittest.TestCase):
    def test_canonical_tasks_match_by_substring(self):
        self.assertEqual(normalize_task_name("alfworld"), "alfworld")
        self.assertEqual(normalize_task_name("AlfWorld_eval"), "alfworld")
        self.assertEqual(normalize_task_name("webshop_train"), "webshop")
        self.assertEqual(normalize_task_name("search"), "search")

    def test_unknown_task_keeps_its_own_bucket(self):
        self.assertEqual(normalize_task_name("Sokoban"), "sokoban")

    def test_missing_task_is_none(self):
        self.assertIsNone(normalize_task_name(None))


class TestTaskNameExtraction(unittest.TestCase):
    def test_task_names_from_task_name_field(self):
        batch = _make_batch(["alfworld", "webshop_train", "search"])
        np.testing.assert_array_equal(get_task_names(batch), np.array(["alfworld", "webshop", "search"], dtype=object))

    def test_task_names_fall_back_to_env_kwargs(self):
        batch = _make_batch(None, batch_size=3)
        batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"task_name": "alfworld"}, {"task_name": "search"}, {"task_name": "search"}], dtype=object
        )
        np.testing.assert_array_equal(get_task_names(batch), np.array(["alfworld", "search", "search"], dtype=object))

    def test_no_task_information_returns_none(self):
        self.assertIsNone(get_task_names(_make_batch(None, batch_size=3)))
        self.assertIsNone(get_task_names(_make_batch([None, None])))

    def test_task_row_indices_groups_rows(self):
        batch = _make_batch(["webshop", "alfworld", "webshop", None])
        indices = task_row_indices(batch)
        self.assertEqual(sorted(indices), ["alfworld", "webshop"])
        np.testing.assert_array_equal(indices["alfworld"], np.array([1]))
        np.testing.assert_array_equal(indices["webshop"], np.array([0, 2]))
        # the row without a task belongs to no task bucket
        self.assertEqual(sum(len(rows) for rows in indices.values()), 3)

    def test_task_row_indices_empty_without_task_information(self):
        self.assertEqual(task_row_indices(_make_batch(None, batch_size=2)), {})


class TestComputeMetricsByTask(unittest.TestCase):
    def test_metric_fn_sees_only_the_rows_of_one_task(self):
        batch = _make_batch(["alfworld", "webshop", "alfworld"])
        metrics = compute_metrics_by_task(batch, lambda task_batch: {"rows": len(task_batch)})
        self.assertEqual(metrics, {"rows/alfworld": 2, "rows/webshop": 1})

    def test_no_per_task_metrics_without_task_information(self):
        self.assertEqual(compute_metrics_by_task(_make_batch(None, batch_size=2), lambda b: {"rows": len(b)}), {})


class TestComputeDataMetricsByTask(unittest.TestCase):
    def setUp(self):
        self.task_names = ["alfworld", "webshop", "alfworld", "search"]
        self.batch = _make_batch(self.task_names)

    def test_every_overall_metric_has_a_per_task_counterpart(self):
        overall = compute_data_metrics(self.batch, use_critic=True)
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)

        for name in overall:
            if "success_rate" in name:  # batch-level constant, reported per task by the env manager
                continue
            for task in ("alfworld", "webshop", "search"):
                self.assertIn(f"{name}/{task}", per_task)

    def test_per_task_value_matches_the_metric_on_that_task_slice(self):
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)
        alfworld_rows = np.array([0, 2])
        expected = compute_data_metrics(self.batch.select_idxs(alfworld_rows), use_critic=True)

        self.assertAlmostEqual(per_task["critic/score/mean/alfworld"], expected["critic/score/mean"])
        self.assertAlmostEqual(per_task["episode/reward/mean/alfworld"], expected["episode/reward/mean"])
        self.assertAlmostEqual(per_task["response_length/mean/alfworld"], expected["response_length/mean"])
        # sanity: the alfworld slice is not just the overall value
        overall = compute_data_metrics(self.batch, use_critic=True)
        self.assertNotAlmostEqual(per_task["critic/score/mean/alfworld"], overall["critic/score/mean"])

    def test_batch_level_success_rate_is_not_split(self):
        per_task = compute_data_metrics_by_task(self.batch, use_critic=True)
        self.assertNotIn("episode/success_rate/alfworld", per_task)

    def test_single_task_batch_reports_the_overall_value(self):
        batch = _make_batch(["search", "search"])
        per_task = compute_data_metrics_by_task(batch, use_critic=False)
        overall = compute_data_metrics(batch, use_critic=False)
        self.assertAlmostEqual(per_task["critic/score/mean/search"], overall["critic/score/mean"])

    def test_no_per_task_metrics_without_task_information(self):
        self.assertEqual(compute_data_metrics_by_task(_make_batch(None, batch_size=2), use_critic=False), {})


class TestTrainerLevelPerTaskMetrics(unittest.TestCase):
    """Per-task metrics produced on the driver, next to their overall counterpart."""

    def test_invalid_action_ratio_is_reported_per_task(self):
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        batch = _make_batch(["alfworld", "alfworld", "webshop", "webshop"])
        batch.batch["prompts"] = torch.zeros((4, 2), dtype=torch.long)
        batch.non_tensor_batch["is_action_valid"] = np.array([True, False, True, True])

        _, metrics = apply_invalid_action_penalty(batch, invalid_action_penalty_coef=0.1)

        self.assertAlmostEqual(metrics["episode/valid_action_ratio"], 0.75)
        self.assertAlmostEqual(metrics["episode/valid_action_ratio/alfworld"], 0.5)
        self.assertAlmostEqual(metrics["episode/valid_action_ratio/webshop"], 1.0)

    def test_reward_kl_penalty_is_reported_per_task(self):
        from verl.trainer.ppo.core_algos import AdaptiveKLController
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty

        batch = _make_batch(["alfworld", "webshop"])
        batch.batch["old_log_probs"] = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        batch.batch["ref_log_prob"] = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])

        _, metrics = apply_kl_penalty(batch, kl_ctrl=AdaptiveKLController(init_kl_coef=0.1, target_kl=0.1, horizon=1000), kl_penalty="kl")

        self.assertAlmostEqual(metrics["actor/reward_kl_penalty/alfworld"], 0.0)
        self.assertAlmostEqual(metrics["actor/reward_kl_penalty/webshop"], -1.0)
        self.assertAlmostEqual(
            metrics["actor/reward_kl_penalty"],
            (metrics["actor/reward_kl_penalty/alfworld"] + metrics["actor/reward_kl_penalty/webshop"]) / 2,
        )

    def test_entropy_loss_is_reported_per_task(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = _make_batch(["alfworld", "webshop"])
        entropys = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        response_masks = torch.ones((2, 2))

        metrics = RayPPOTrainer._entropy_loss_metrics(batch, entropys, response_masks, "token-mean")

        self.assertAlmostEqual(metrics["actor/entropy_loss"], 2.0)
        self.assertAlmostEqual(metrics["actor/entropy_loss/alfworld"], 1.0)
        self.assertAlmostEqual(metrics["actor/entropy_loss/webshop"], 3.0)

    def test_task_ids_are_attached_for_the_workers(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = _make_batch(["webshop", "alfworld", "webshop", None])
        RayPPOTrainer._attach_task_ids(batch)

        self.assertEqual(batch.meta_info["task_id_names"], ["alfworld", "webshop"])
        torch.testing.assert_close(batch.batch["task_ids"], torch.tensor([1, 0, 1, -1]))

    def test_task_ids_are_not_attached_without_task_information(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = _make_batch(None, batch_size=2)
        RayPPOTrainer._attach_task_ids(batch)

        self.assertNotIn("task_ids", batch.batch.keys())
        self.assertNotIn("task_id_names", batch.meta_info)


class TestIterTaskRowMasks(unittest.TestCase):
    """The worker-side (actor / critic) split of a micro-batch into task rows."""

    def test_rows_are_grouped_by_task_id(self):
        task_ids = torch.tensor([1, 0, 1])
        masks = dict(iter_task_row_masks(task_ids, ["alfworld", "webshop"]))

        self.assertEqual(sorted(masks), ["alfworld", "webshop"])
        torch.testing.assert_close(masks["alfworld"], torch.tensor([False, True, False]))
        torch.testing.assert_close(masks["webshop"], torch.tensor([True, False, True]))

    def test_rows_without_a_task_are_skipped(self):
        masks = dict(iter_task_row_masks(torch.tensor([-1, 0, 7]), ["alfworld"]))

        self.assertEqual(list(masks), ["alfworld"])
        torch.testing.assert_close(masks["alfworld"], torch.tensor([False, True, False]))

    def test_nothing_is_yielded_without_task_information(self):
        self.assertEqual(list(iter_task_row_masks(None, ["alfworld"])), [])
        self.assertEqual(list(iter_task_row_masks(torch.tensor([0, 0]), None)), [])
        self.assertEqual(list(iter_task_row_masks(torch.tensor([0, 0]), [])), [])


class TestTrajectoryResponseTokens(unittest.TestCase):
    """episode/response_tokens sums the per-turn rows of one trajectory."""

    def setUp(self):
        # two trajectories: alfworld spans 2 turns (2 + 1 tokens), webshop 1 turn (2 tokens)
        self.batch = _make_batch(
            ["alfworld", "alfworld", "webshop"],
            traj_uids=["traj-a", "traj-a", "traj-b"],
            response_lengths=[2, 1, 2],
        )

    def test_tokens_are_summed_over_the_turns_of_a_trajectory(self):
        np.testing.assert_array_equal(compute_trajectory_response_tokens(self.batch), np.array([3.0, 2.0]))

    def test_metric_is_over_trajectories_not_turns(self):
        metrics = compute_data_metrics(self.batch, use_critic=False)

        self.assertAlmostEqual(metrics["episode/response_tokens/mean"], 2.5)  # (3 + 2) / 2 trajectories
        self.assertAlmostEqual(metrics["episode/response_tokens/max"], 3.0)
        self.assertAlmostEqual(metrics["episode/response_tokens/min"], 2.0)
        # the per-turn length is a different quantity: (2 + 1 + 2) / 3 rows
        self.assertAlmostEqual(metrics["response_length/mean"], 5 / 3)

    def test_metric_is_reported_per_task(self):
        per_task = compute_data_metrics_by_task(self.batch, use_critic=False)

        self.assertAlmostEqual(per_task["episode/response_tokens/mean/alfworld"], 3.0)
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/webshop"], 2.0)

    def test_opd_metrics_report_trajectory_tokens_per_task(self):
        from verl.trainer.ppo.opd_ray_trainer import compute_opd_data_metrics, compute_opd_data_metrics_by_task

        self.assertAlmostEqual(compute_opd_data_metrics(self.batch)["episode/response_tokens/mean"], 2.5)
        per_task = compute_opd_data_metrics_by_task(self.batch)
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/alfworld"], 3.0)
        self.assertAlmostEqual(per_task["episode/response_tokens/mean/webshop"], 2.0)


class TestComputeThroughputMetricsByTask(unittest.TestCase):
    def test_per_task_tokens_and_throughput_sum_to_the_overall_value(self):
        batch = _make_batch(["alfworld", "webshop", "alfworld"])
        batch.meta_info["global_token_num"] = [4, 4, 4]
        metrics = compute_throughout_metrics(batch=batch, timing_raw={"step": 2.0}, n_gpus=2)

        self.assertEqual(metrics["perf/total_num_tokens/alfworld"], 8)
        self.assertEqual(metrics["perf/total_num_tokens/webshop"], 4)
        self.assertAlmostEqual(
            metrics["perf/throughput/alfworld"] + metrics["perf/throughput/webshop"],
            metrics["perf/throughput"],
        )

    def test_no_per_task_metrics_without_task_information(self):
        batch = _make_batch(None, batch_size=2)
        batch.meta_info["global_token_num"] = [4, 4]
        metrics = compute_throughout_metrics(batch=batch, timing_raw={"step": 1.0}, n_gpus=1)
        self.assertEqual([name for name in metrics if name.count("/") > 1], [])


if __name__ == "__main__":
    unittest.main()
