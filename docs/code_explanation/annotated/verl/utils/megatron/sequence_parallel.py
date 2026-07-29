# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu


# [EXPLAIN] `mark_parameter_as_sequence_parallel` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def mark_parameter_as_sequence_parallel(parameter):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parameter.sequence_parallel = True


# [EXPLAIN] `is_sequence_parallel_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_sequence_parallel_param(param):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return hasattr(param, "sequence_parallel") and param.sequence_parallel


# [EXPLAIN] `pad_to_sequence_parallel` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pad_to_sequence_parallel(unpad_tokens: torch.Tensor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """pad the tokens such that the total length is a multiple of sp world size

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding

    Returns:
        the padded tokens: (total_nnz + pad_size,...)

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_nnz = unpad_tokens.shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_world_size = mpu.get_tensor_model_parallel_world_size()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_size = 0 if total_nnz % sp_world_size == 0 else sp_world_size - total_nnz % sp_world_size

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pad_size > 0:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if unpad_tokens.ndim == 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif unpad_tokens.ndim == 2:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return unpad_tokens
