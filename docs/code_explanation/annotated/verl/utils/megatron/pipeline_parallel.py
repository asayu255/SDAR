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
from megatron.core import parallel_state as mpu

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .sequence_parallel import pad_to_sequence_parallel


# [EXPLAIN] `compute_transformers_input_shapes` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_transformers_input_shapes(batches, meta_info):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron

    # pre-compute input shapes for each micro-batch at each pp stage
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_shapes = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model_inputs in batches:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = model_inputs["input_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = model_inputs["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if meta_info["sequence_parallel"]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # compute shapes for model_inputs
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            input_shapes.append(
                torch.Size(
                    [
                        input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(),
                        1,
                        meta_info["hidden_size"],
                    ]
                )
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # compute shapes for model_inputs
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info["hidden_size"]]))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_shapes


# [EXPLAIN] `make_batch_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def make_batch_generator(batches, vpp_size):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if vpp_size > 1:
        # has vpp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_generator = [iter(b) for b in batch_generator]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # no vpp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_generator = iter(batches)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return batch_generator
