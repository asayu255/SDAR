# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from verl.utils.megatron_utils import unwrap_model

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding


# [EXPLAIN] `gptmodel_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Default forward pass for GPT models with optional sequence packing."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pre_process = unwrap_model(model).pre_process
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    post_process = unwrap_model(model).post_process
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pack_seqs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size, seq_len = attention_mask.shape[:2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.contiguous()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if post_process and logits_processor is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_dict = logits_processor(output_orig, **args)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = {k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process) for k, v in output_dict.items()}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size, sequence_length = attention_mask.shape
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if value_model and post_process:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output[..., 0]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `gptmodel_forward_qwen2_5_vl` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gptmodel_forward_qwen2_5_vl(*args, **kwargs):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Forward pass for Qwen2.5 VL model (not implemented)."""
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("VLM is not supported yet")
