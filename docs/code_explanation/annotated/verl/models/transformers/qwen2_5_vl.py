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
from dataclasses import dataclass
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Optional, Tuple, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `Qwen2_5_VLCausalLMOutputForPPO` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen2_5_VLCausalLMOutputForPPO(Qwen2_5_VLCausalLMOutputWithPast):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs: Optional[torch.FloatTensor] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy: Optional[torch.FloatTensor] = None


# [EXPLAIN] `forward_for_ppo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_for_ppo(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputForPPO]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    r"""
    Copy paste Qwen2_5_VL's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/qwen2_5_vl.py
    ```"""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if inputs_embeds is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = self.model.embed_tokens(input_ids)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pixel_values is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pixel_values = pixel_values.type(self.visual.dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            n_image_features = image_embeds.shape[0]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if n_image_tokens != n_image_features:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mask = input_ids == self.config.image_token_id
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
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            n_video_features = video_embeds.shape[0]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if n_video_tokens != n_video_features:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mask = input_ids == self.config.video_token_id
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
        if attention_mask is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_size, seq_length, _ = inputs_embeds.shape
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = position_ids.add(delta)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not return_dict:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("forward_for_ppo has to return_dict")

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
        raise RuntimeError("To use forward_for_ppo, either labels or input_ids must be provided.")

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
    return Qwen2_5_VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
