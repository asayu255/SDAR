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
from typing import Optional, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.cache_utils import Cache
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_flash_attention_utils import _flash_attention_forward

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import gather_heads_scatter_seq, gather_outpus_and_unpad, gather_seq_scatter_heads, get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_rank, get_ulysses_sequence_parallel_world_size, validate_ulysses_config


# [EXPLAIN] `_merge_with_image_features` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _merge_with_image_features(
    self,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Args:
        inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, input_embed_dim)`):
            The input embeddings.
        input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            The input ids.
        image_features (:obj:`torch.Tensor` of shape :obj:`(image_token_nums, image_feature_dim)`):
            The image features to merge with the input embeddings.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_token_index: int = self.config.media_placeholder_token_id

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_feature_nums, image_feature_dim = image_features.shape

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert image_feature_dim == input_embed_dim

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_token_nums = (input_ids == image_token_index).sum()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_image_token_nums = torch.tensor([image_token_nums], dtype=image_token_nums.dtype, device=input_ids.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_image_token_nums = gather_outpus_and_unpad(total_image_token_nums, gather_dim=0)  # [sp_size]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert image_feature_nums == total_image_token_nums.sum()

    # (batch_size, sequence_length / sp, input_embed_dim) -> (batch_size * sequence_length / sp, input_embed_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_embeds = inputs_embeds.reshape(-1, input_embed_dim)

    # (batch_size, sequence_length / sp) -> (batch_size * sequence_length / sp)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = input_ids.flatten()

    # split image features and fill in the image token positions if there are image tokens
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_image_features = image_features.split(total_image_token_nums.tolist(), dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_rank = get_ulysses_sequence_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_features = sp_image_features[sp_rank]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_embeds[input_ids == image_token_index] = image_features

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs_embeds = inputs_embeds.reshape((batch_size, sequence_length, input_embed_dim))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return inputs_embeds


# Copied from transformers.models.llama.modeling_llama.rotate_half
# [EXPLAIN] `rotate_half` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def rotate_half(x):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Rotates half the hidden dims of the input."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x1 = x[..., : x.shape[-1] // 2]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x2 = x[..., x.shape[-1] // 2 :]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# [EXPLAIN] `apply_rotary_pos_emb` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b, h, s, d = q.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b, h, s, d = k.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
# [EXPLAIN] `repeat_kv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if n_rep == 1:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden_states
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# [EXPLAIN] `_ulysses_flash_attn_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, q_len, _ = hidden_states.size()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.q_lora_rank is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        q = self.q_proj(hidden_states)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kv_seq_len = value_states.shape[-2]

    # patch to get all emb
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    kv_seq_len *= ulysses_sp_size

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.q_head_dim != self.v_head_dim:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

    # patch
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        validate_ulysses_config(self.num_heads, ulysses_sp_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = repeat_kv(key_states, num_key_value_groups)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = repeat_kv(value_states, num_key_value_groups)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        # (batch_size, num_head / sp_size, seq_length, head_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_q_len = query_states.size(2)  # full_q_len = seq_length

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.concat(position_ids_list, dim=-1)

    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_q_len = q_len

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    query_states = query_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key_states = key_states.transpose(1, 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    value_states = value_states.transpose(1, 2)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dropout_rate = self.attention_dropout if self.training else 0.0

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        dropout=dropout_rate,
        sliding_window=None,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        position_ids=position_ids,  # important: pass position ids
        softmax_scale=self.softmax_scale,
    )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sp_size > 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if self.q_head_dim != self.v_head_dim:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attn_output = self.o_proj(attn_output)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return attn_output, None, None
