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
Tests for the metric utilities in verl.trainer.ppo.metric_utils.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import unittest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import MagicMock, patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.ppo.metric_utils import (
    bootstrap_metric,
    calc_maj_val,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.metric import (
    reduce_metrics,
)


# [EXPLAIN] `TestReduceMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestReduceMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the reduce_metrics function."""

    # [EXPLAIN] `test_reduce_metrics_basic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_reduce_metrics_basic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test that reduce_metrics correctly computes means."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "accuracy": [0.0, 0.5, 1.0],
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = reduce_metrics(metrics)
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(result["loss"], 2.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(result["accuracy"], 0.5)
    
    # [EXPLAIN] `test_reduce_metrics_empty` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_reduce_metrics_empty(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test that reduce_metrics handles empty lists."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {
            "empty": [],
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = reduce_metrics(metrics)
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertTrue(np.isnan(result["empty"]))
    
    # [EXPLAIN] `test_reduce_metrics_single_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_reduce_metrics_single_value(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test that reduce_metrics works with single values."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = {
            "single": [5.0],
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = reduce_metrics(metrics)
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(result["single"], 5.0)


# [EXPLAIN] `TestComputeDataMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeDataMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the compute_data_metrics function."""
    
    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Set up common test data."""
        # Create a mock DataProto object
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch.batch = {
            "token_level_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "token_level_rewards": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "returns": torch.tensor([[1.1, 1.2], [1.3, 1.4]]),
            "responses": torch.zeros((2, 2)),  # 2 samples, 2 tokens each
            "attention_mask": torch.tensor([
                [1, 1, 1, 1],  # 2 prompt tokens, 2 response tokens
                [1, 1, 1, 1],
            ]),
            "values": torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
        }
    
    # [EXPLAIN] `test_compute_data_metrics_with_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_compute_data_metrics_with_critic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test compute_data_metrics with critic enabled."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_data_metrics(self.batch, use_critic=True)
        
        # Check that all expected metrics are present
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/score/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/rewards/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/advantages/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/returns/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/values/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/vf_explained_var", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("response_length/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("prompt_length/mean", metrics)
        
        # Check some specific values
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["critic/score/mean"], 5.0)  # Sum of token_level_scores
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["critic/rewards/mean"], 2.5)  # Sum of token_level_rewards
    
    # [EXPLAIN] `test_compute_data_metrics_without_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_compute_data_metrics_without_critic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test compute_data_metrics with critic disabled."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_data_metrics(self.batch, use_critic=False)
        
        # Check that critic-specific metrics are not present
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotIn("critic/values/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertNotIn("critic/vf_explained_var", metrics)
        
        # Check that other metrics are still present
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/score/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("critic/rewards/mean", metrics)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("response_length/mean", metrics)


# [EXPLAIN] `TestComputeTimingMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeTimingMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the compute_timing_metrics function."""
    
    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Set up common test data."""
        # Create a mock DataProto object
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch.batch = {
            "responses": torch.zeros((2, 3)),  # 2 samples, 3 response tokens each
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1],  # 3 prompt tokens, 3 response tokens
                [1, 1, 1, 1, 1, 1],
            ]),
        }
        
        # Mock the _compute_response_info function to return known values
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_info = {
            "prompt_length": torch.tensor([3.0, 3.0]),
            "response_length": torch.tensor([3.0, 3.0]),
            "response_mask": torch.ones((2, 3)),
        }
    
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch("verl.trainer.ppo.metric_utils._compute_response_info")
    # [EXPLAIN] `test_compute_timing_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_compute_timing_metrics(self, mock_compute_response_info):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test compute_timing_metrics with various timing data."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_compute_response_info.return_value = self.response_info
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        timing_raw = {
            "gen": 0.5,  # 500ms
            "ref": 0.3,  # 300ms
            "values": 0.2,  # 200ms
        }
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_timing_metrics(self.batch, timing_raw)
        
        # Check raw timing metrics
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["timing_s/gen"], 0.5)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["timing_s/ref"], 0.3)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["timing_s/values"], 0.2)
        
        # Check per-token timing metrics
        # gen uses only response tokens (6 tokens)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["timing_per_token_ms/gen"], 0.5 * 1000 / 6, places=5)
        
        # ref and values use all tokens (12 tokens)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["timing_per_token_ms/ref"], 0.3 * 1000 / 12, places=5)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(metrics["timing_per_token_ms/values"], 0.2 * 1000 / 12, places=5)


# [EXPLAIN] `TestComputeThroughputMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestComputeThroughputMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the compute_throughout_metrics function."""
    
    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Set up common test data."""
        # Create a mock DataProto object
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch.meta_info = {
            "global_token_num": [100, 200, 300],  # 600 tokens total
        }
    
    # [EXPLAIN] `test_compute_throughout_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_compute_throughout_metrics(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test compute_throughout_metrics with various timing data."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        timing_raw = {
            "step": 2.0,  # 2 seconds per step
        }
        
        # Test with 1 GPU
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=1)
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/throughput"], 600 / 2.0)  # 300 tokens/sec
        
        # Test with 2 GPUs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=2)
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(metrics["perf/throughput"], 600 / (2.0 * 2))  # 150 tokens/sec/GPU


# [EXPLAIN] `TestBootstrapMetric` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestBootstrapMetric(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the bootstrap_metric function."""
    
    # [EXPLAIN] `test_bootstrap_metric_basic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_bootstrap_metric_basic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test bootstrap_metric with simple data and functions."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = [1, 2, 3, 4, 5]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reduce_fns = [np.mean, np.max]
        
        # Use a fixed seed for reproducibility
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = bootstrap_metric(data, subset_size=3, reduce_fns=reduce_fns, n_bootstrap=100, seed=42)
        
        # Check that we get two results (one for each reduce_fn)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(len(result), 2)
        
        # Each result should be a tuple of (mean, std)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mean_result, max_result = result
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(len(mean_result), 2)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(len(max_result), 2)
        
        # The mean of means should be close to the true mean (3.0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(mean_result[0], 3.0, delta=0.3)
        
        # The mean of maxes should be close to the expected value for samples of size 3
        # For samples of size 3 from [1,2,3,4,5], the expected max is around 4.0-4.5
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertGreater(max_result[0], 3.5)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertLess(max_result[0], 5.0)
    
    # [EXPLAIN] `test_bootstrap_metric_empty` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_bootstrap_metric_empty(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test bootstrap_metric with empty data."""
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.assertRaises(ValueError):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            bootstrap_metric([], subset_size=1, reduce_fns=[np.mean])


# [EXPLAIN] `TestCalcMajVal` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestCalcMajVal(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the calc_maj_val function."""
    
    # [EXPLAIN] `test_calc_maj_val_basic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_calc_maj_val_basic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test calc_maj_val with simple data."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "A", "val": 0.7},
        ]
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = calc_maj_val(data, vote_key="pred", val_key="val")
        
        # "A" is the majority vote, so we should get the first "val" for "A"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertEqual(result, 0.9)
    
    # [EXPLAIN] `test_calc_maj_val_tie` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_calc_maj_val_tie(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test calc_maj_val with tied votes."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "B", "val": 0.7},
            {"pred": "A", "val": 0.6},
        ]
        
        # In case of a tie, the first key in sorted order wins
        # This depends on Python's dict implementation, but for this test
        # we just verify that one of the valid values is returned
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = calc_maj_val(data, vote_key="pred", val_key="val")
        
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertTrue(result in [0.9, 0.8])


# [EXPLAIN] `TestProcessValidationMetrics` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestProcessValidationMetrics(unittest.TestCase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests for the process_validation_metrics function."""
    
    # [EXPLAIN] `test_process_validation_metrics_basic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_process_validation_metrics_basic(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test process_validation_metrics with simple data."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_sources = ["source1", "source1", "source2"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sample_inputs = ["prompt1", "prompt1", "prompt2"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
        }
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = process_validation_metrics(
            data_sources, sample_inputs, infos_dict, seed=42
        )
        
        # Check the structure of the result
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("source1", result)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("source2", result)
        
        # Check that source1 has metrics for score
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("score", result["source1"])
        
        # Check that mean@2 is present for source1/score
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("mean@2", result["source1"]["score"])
        
        # Check the value of mean@2 for source1/score
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertAlmostEqual(result["source1"]["score"]["mean@2"], 0.85)
    
    # [EXPLAIN] `test_process_validation_metrics_with_pred` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_process_validation_metrics_with_pred(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Test process_validation_metrics with prediction data."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_sources = ["source1", "source1", "source1"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sample_inputs = ["prompt1", "prompt1", "prompt1"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
            "pred": ["A", "B", "A"],
        }
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = process_validation_metrics(
            data_sources, sample_inputs, infos_dict, seed=42
        )
        
        # Check that majority voting metrics are present
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.assertIn("maj@2/mean", result["source1"]["score"])
        
        # For bootstrap with n=2, the majority vote could be either A or B
        # depending on the random sampling, so we don't check the exact value


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    unittest.main()
