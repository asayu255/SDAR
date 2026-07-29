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
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLAttention,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import is_npu_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.transformers_compat import is_transformers_version_in_range
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_flash_attn_2_available():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_npu_available:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _flash_use_top_left_mask = flash_attn_supports_top_left_mask()

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_flash_deterministic_enabled = os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


# [EXPLAIN] `get_rope_index` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1405
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    spatial_merge_size = processor.image_processor.merge_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokens_per_second = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attention_mask is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.ones_like(input_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_index, video_index = 0, 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = input_ids[attention_mask == 1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_nums, video_nums = 0, 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_tokens = input_ids[vision_start_indices + 1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_nums = (vision_tokens == image_token_id).sum()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_nums = (vision_tokens == video_token_id).sum()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_tokens = input_ids.tolist()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        llm_pos_ids_list: list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        st = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remain_images, remain_videos = image_nums, video_nums
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _ in range(image_nums + video_nums):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if image_token_id in input_tokens and remain_images > 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed_image = input_tokens.index(image_token_id, st)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed_image = len(input_tokens) + 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if video_token_id in input_tokens and remain_videos > 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed_video = input_tokens.index(video_token_id, st)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed_video = len(input_tokens) + 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if ed_image < ed_video:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                second_per_grid_t = 0
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                image_index += 1
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                remain_images -= 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed = ed_image
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                video_index += 1
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                remain_videos -= 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ed = ed_video

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_len = ed - st

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if st < len(input_tokens):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_len = len(input_tokens) - st
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attention_mask is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = attention_mask.long().cumsum(-1) - 1
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            position_ids.masked_fill_(attention_mask == 0, 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return position_ids


# [EXPLAIN] `prepare_fa2_from_position_ids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert position_ids.ndim == 2  # (batch_size, seq_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids = position_ids.view(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens = torch.cat(
        (
            (position_ids == 0).nonzero().view(-1).to(torch.int32),
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))


# [EXPLAIN] `_custom_flash_attention_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _flash_supports_deterministic:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if kwargs.get("softcap") is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert position_ids.ndim == 2  # (batch_size, seq_length / sp_size)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_size = get_ulysses_sequence_parallel_world_size()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sp_size > 1:
        # qkv: (batch_size, seq_length / sp_size, num_head, head_size)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        validate_ulysses_config(query_states.size(2), sp_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (batch_size, seq_length)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = query_states.size(0)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        q, k, v, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=is_causal,
            **flash_kwargs,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )  # do not pass position_ids to old flash_attention_forward

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attn_output


# [EXPLAIN] `qwen2_vl_attn_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen2_vl_attn_forward(
    self: "Qwen2VLAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> tuple[torch.Tensor, None, None]:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = self.k_proj(hidden_states)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = self.v_proj(hidden_states)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos, sin = position_embeddings
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sliding_window = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sliding_window = self.config.sliding_window

    # This is before the transpose
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_len = query_states.shape[2]

    # FA2 uses non-transposed inputs
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = query_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = key_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = value_states.transpose(1, 2)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids.ndim == 3:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = position_ids[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        sliding_window=sliding_window,
        use_top_left_mask=_flash_use_top_left_mask,
        position_ids=position_ids,  # important: pass position ids
    )  # (batch_size, seq_length / sp_size, num_head, head_size)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = self.o_proj(attn_output)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_transformers_version_in_range(min_version="4.54.0"):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return attn_output, None
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return attn_output, None, None


# [EXPLAIN] `_get_input_embeds` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_input_embeds(
    model: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_embeds = model.get_input_embeddings()(input_ids)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pixel_values is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pixel_values = pixel_values.type(model.visual.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_image_features = image_embeds.shape[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if n_image_tokens != n_image_features:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = input_ids == model.config.image_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_unsqueezed = mask.unsqueeze(-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_mask = mask_expanded.to(inputs_embeds.device)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pixel_values_videos is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_video_features = video_embeds.shape[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if n_video_tokens != n_video_features:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = input_ids == model.config.video_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_unsqueezed = mask.unsqueeze(-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_mask = mask_expanded.to(inputs_embeds.device)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pixel_values is None and pixel_values_videos is None:  # handle mixed text-image data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = model.config.vision_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        inputs_embeds += 0.0 * image_embeds.mean()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if attention_mask is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = attention_mask.to(inputs_embeds.device)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return inputs_embeds, attention_mask


# [EXPLAIN] `process_position_ids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def process_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids.ndim != 3 or position_ids.size(0) != 4:
        # we concat the text position ids with the 3D vision position ids by default
        # see https://github.com/huggingface/transformers/pull/39447
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"position_ids should be a 3D tensor of shape (4, batch_size, seq_length), got {position_ids.shape}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_transformers_version_in_range(max_version="4.53.3"):
        # transformers < 4.54.0 only accepts vision position ids, so we discard the text position ids here
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = position_ids[1:]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return position_ids


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `Qwen2VLCausalLMOutputForPPO` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen2VLCausalLMOutputForPPO(Qwen2VLCausalLMOutputWithPast):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs: Optional[torch.FloatTensor] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy: Optional[torch.FloatTensor] = None


# [EXPLAIN] `qwen2_vl_base_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen2_vl_base_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return self.language_model(input_ids=None, **kwargs)


# [EXPLAIN] `qwen2_vl_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen2_vl_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_transformers_version_in_range(min_version="4.52.0"):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=process_position_ids(position_ids),
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds, attention_mask = _get_input_embeds(
            self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=process_position_ids(position_ids),
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# [EXPLAIN] `forward_with_normal_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_normal_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen2VLCausalLMOutputWithPast":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = self.lm_head(hidden_states)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Qwen2VLCausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


# [EXPLAIN] `forward_with_torch_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_torch_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]

    # Loss calculations
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if labels is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif input_ids is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fused_linear_for_ppo = FusedLinearForPPO()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


# [EXPLAIN] `forward_with_triton_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_triton_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]

    # Loss calculations
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if labels is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif input_ids is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
