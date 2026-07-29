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
from typing import Callable, Optional, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.cache_utils import Cache
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_flash_attention_utils import _flash_attention_forward
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.utils import logging

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.get_logger(__name__)


# [EXPLAIN] `qwen2_flash_attn_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen2_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Adapted from transformers 4.47.1 to support Ulysses sequence parallelism.

    NOTE: This function is only tested on transformers versions between 4.45.0 and 4.47.1.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, q_len, _ = hidden_states.size()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = self.q_proj(hidden_states)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = self.k_proj(hidden_states)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = self.v_proj(hidden_states)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        validate_ulysses_config(self.num_heads, ulysses_sp_size)

        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_q_len = query_states.size(2)  # full seq length

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_embeddings is None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cos, sin = self.rotary_emb(value_states, position_ids)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cos, sin = position_embeddings
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if past_key_value is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_dtype = query_states.dtype
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if input_dtype == torch.float32:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.is_autocast_enabled():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif hasattr(self.config, "_pre_quantization_dtype"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            target_dtype = self.config._pre_quantization_dtype
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            target_dtype = self.q_proj.weight.dtype

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning_once(f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = query_states.to(target_dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = key_states.to(target_dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = query_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = key_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = value_states.transpose(1, 2)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None and self.layer_idx >= self.config.max_window_layers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sliding_window = self.config.sliding_window
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sliding_window = None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    # use full_q_len to reshape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = self.o_proj(attn_output)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not output_attentions:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_weights = None

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attn_output, attn_weights, past_key_value


# [EXPLAIN] `qwen2_attn_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Adapted from transformers 4.49.0 to support Ulysses sequence parallelism for transformers >= 4.48.0.

    NOTE: This function has been tested only on transformers versions between 4.48.0 and 4.50.0.
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, q_len, _ = hidden_states.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_shape = (bsz, q_len, -1, self.head_dim)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        validate_ulysses_config(self.config.num_attention_heads, ulysses_sp_size)

        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_q_len = query_states.size(2)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos, sin = position_embeddings
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sliding_window = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None and self.layer_idx >= self.config.max_window_layers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sliding_window = self.config.sliding_window

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_interface: Callable = eager_attention_forward
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.config._attn_implementation != "eager":
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.warning_once('`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    ########## AlltoAll for Ulysses ##########
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = self.o_proj(attn_output)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attn_output, attn_weights
