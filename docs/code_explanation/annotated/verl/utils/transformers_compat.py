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
Compatibility utilities for different versions of transformers library.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib.metadata
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import lru_cache
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging import version

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    # [EXPLAIN] `flash_attn_supports_top_left_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def flash_attn_supports_top_left_mask():
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@lru_cache
# [EXPLAIN] `is_transformers_version_in_range` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # Get the installed version of the transformers library
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformers_version_str = importlib.metadata.version("transformers")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except importlib.metadata.PackageNotFoundError as e:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    transformers_version = version.parse(transformers_version_str)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lower_bound_check = True
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if min_version is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lower_bound_check = version.parse(min_version) <= transformers_version

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    upper_bound_check = True
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if max_version is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        upper_bound_check = transformers_version <= version.parse(max_version)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return lower_bound_check and upper_bound_check
