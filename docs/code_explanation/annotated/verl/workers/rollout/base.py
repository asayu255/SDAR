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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["BaseRollout"]


# [EXPLAIN] `BaseRollout` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseRollout(ABC):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """

        Args:
            dataloader: an Iterable of TensorDict that consistently generates prompts. Note that the dataloader
            should handle when the training stops.
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Generate sequences"""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
