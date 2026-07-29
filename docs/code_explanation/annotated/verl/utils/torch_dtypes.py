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
Adapted from Cruise.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


# [EXPLAIN] `PrecisionType` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class PrecisionType:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    HALF = "16"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FLOAT = "32"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FULL = "64"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    BFLOAT = "bf16"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    MIXED = "mixed"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `supported_type` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def supported_type(precision: Union[str, int]) -> bool:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return any(x == precision for x in PrecisionType)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `supported_types` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def supported_types() -> list[str]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [x.value for x in PrecisionType]

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `is_fp16` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def is_fp16(precision):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return precision in HALF_LIST

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `is_fp32` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def is_fp32(precision):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return precision in FLOAT_LIST

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `is_bf16` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def is_bf16(precision):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return precision in BFLOAT_LIST

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `to_dtype` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def to_dtype(precision):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if precision in HALF_LIST:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.float16
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif precision in FLOAT_LIST:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.float32
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif precision in BFLOAT_LIST:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.bfloat16
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError(f"unexpected precision: {precision}")

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `to_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def to_str(precision):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if precision == torch.float16:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "fp16"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif precision == torch.float32:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "fp32"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif precision == torch.bfloat16:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "bf16"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError(f"unexpected precision: {precision}")
