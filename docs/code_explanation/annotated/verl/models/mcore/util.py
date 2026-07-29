# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from megatron.core.packed_seq_params import PackedSeqParams


# [EXPLAIN] `preprocess_packed_seqs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def preprocess_packed_seqs(input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True) -> tuple[torch.Tensor, PackedSeqParams]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1 gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = input_ids.shape[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_size = mpu.get_context_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_rank = mpu.get_context_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlens_in_batch_padded = seqlens_in_batch + pad_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_seqlen_in_batch = seqlens_in_batch_padded.max().item()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape = list(input_ids.shape[1:])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape[0] = seqlens_in_batch_padded.sum().item() // cp_size
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pre_process:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(batch_size):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cp_size <= 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                seqlen = seqlens_in_batch[i]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids_rmpad[cu_seqlens_padded[i] : cu_seqlens_padded[i] + seqlen] = input_ids[i, attention_mask[i]]
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            seqlen = seqlens_in_batch_padded[i] // cp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            half_seqlen = seqlen // 2
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            start_idx = cu_seqlens_padded[i] // cp_size
            # split to 2 chunks
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            d = input_ids[i, attention_mask[i]]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            remain_start = seqlens_in_batch_padded[i] - half_seqlen * (cp_rank + 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            remain_end = seqlens_in_batch_padded[i] - half_seqlen * cp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            remain_end = min(remain_end, d.shape[0])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            remain_len = remain_end - remain_start
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if remain_len > 0:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[remain_start:remain_end]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pre_process:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return input_ids, packed_seq_params


# [EXPLAIN] `postprocess_packed_seqs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Postprocess packed sequences
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not post_process:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output_list[mpu.get_context_parallel_rank()] = output
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_list = [output]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cp_size <= 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            s = attention_mask[i].sum().item()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_new[i, attention_mask[i]] = output[0][packed_seq_params.cu_seqlens_q_padded[i] : packed_seq_params.cu_seqlens_q_padded[i] + s]
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        s_len_padded_chunk = (packed_seq_params.cu_seqlens_q_padded[i + 1] - packed_seq_params.cu_seqlens_q_padded[i]) // cp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        half_seqlen = s_len_padded_chunk // 2
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        s_len = attention_mask[i].sum().item()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        s_len_padded = s_len_padded_chunk * cp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for j in range(cp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            o = output_list[j][0]
            # split to 2 chunks
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            packed_start_idx = packed_seq_params.cu_seqlens_q_padded[i] // cp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_new[i, attention_mask[i]] = tmp[:s_len]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_new


# [EXPLAIN] `remove_left_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert attention_mask.ndim == 2
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert position_ids.ndim == 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_size = mpu.get_context_parallel_world_size()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = input_ids.shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_lens = attention_mask.sum(dim=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len = seq_lens.max().item()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sequence_parallel:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq_len = seq_len + pad_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape[1] = seq_len
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pre_process:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_attention_mask = torch.zeros(dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pre_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pre_process:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return new_input_ids, new_attention_mask, new_position_ids
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return input_ids, new_attention_mask, new_position_ids


# [EXPLAIN] `recover_left_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def recover_left_padding(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Recover left padding from result
    return result
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not post_process:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape = list(result.shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape[1] = origin_seqlen
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return new_result
