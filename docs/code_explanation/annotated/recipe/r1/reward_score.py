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


# [EXPLAIN] `reward_func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from recipe.r1.tasks import math

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return math.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source == "Idavidrein/gpqa":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from recipe.r1.tasks import gpqa

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return gpqa.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from recipe.r1.tasks import livecodebench

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return livecodebench.compute_score(solution_str, ground_truth)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError
