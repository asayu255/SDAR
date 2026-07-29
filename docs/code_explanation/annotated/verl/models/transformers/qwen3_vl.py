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
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `get_rope_index` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    spatial_merge_size = processor.image_processor.merge_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_token_id = processor.image_token_id
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    video_token_id = processor.video_token_id
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to seperate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if video_grid_thw is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_grid_thw[:, 0] = 1

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attention_mask is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.ones_like(input_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_index, video_index = 0, 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = attention_mask.to(input_ids.device)
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

            # t_index is always 0 because llm_grid_t is always 1
            # (we use timestamps to encode the temporal information for videos)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
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
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return position_ids


# [EXPLAIN] `_get_input_embeds` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_embeds = model.get_input_embeddings()(input_ids)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_mask, video_mask = None, None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pixel_values is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pixel_values = pixel_values.type(model.visual.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
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
        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
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

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    visual_pos_masks = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    deepstack_visual_embeds = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_mask = image_mask[..., 0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_mask = video_mask[..., 0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        visual_pos_masks = image_mask | video_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        deepstack_visual_embeds = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_mask_joint = image_mask[visual_pos_masks]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_mask_joint = video_mask[visual_pos_masks]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            embed_joint[image_mask_joint, :] = img_embed
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            embed_joint[video_mask_joint, :] = vid_embed
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            deepstack_visual_embeds.append(embed_joint)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif image_mask is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_mask = image_mask[..., 0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        visual_pos_masks = image_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        deepstack_visual_embeds = deepstack_image_embeds
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif video_mask is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        video_mask = video_mask[..., 0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        visual_pos_masks = video_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        deepstack_visual_embeds = deepstack_video_embeds

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pixel_values is None and pixel_values_videos is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = model.config.vision_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_embeds, dummy_deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        inputs_embeds += 0.0 * image_embeds.mean()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for emb in dummy_deepstack_image_embeds or []:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            inputs_embeds += 0.0 * emb.mean()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if attention_mask is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = attention_mask.to(inputs_embeds.device)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `Qwen3VLCausalLMOutputForPPO` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs: Optional[torch.FloatTensor] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy: Optional[torch.FloatTensor] = None


# [EXPLAIN] `qwen3_vl_base_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    kwargs.update(input_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


# [EXPLAIN] `forward_with_normal_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(input_ids, **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = self.lm_head(hidden_states)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


# [EXPLAIN] `forward_with_torch_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(input_ids, **kwargs)
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
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


# [EXPLAIN] `forward_with_triton_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(input_ids, **kwargs)
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
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
