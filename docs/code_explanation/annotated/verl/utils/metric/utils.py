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
Metrics utils.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np


# [EXPLAIN] `reduce_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in metrics.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "max" in key:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics[key] = np.max(val)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "min" in key:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics[key] = np.min(val)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics[key] = np.mean(val)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return metrics
