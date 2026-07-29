# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2 model."""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, Tuple, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.utils.checkpoint
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import ModelParallelConfig, mpu, parallel_state, tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_outputs import BaseModelOutputWithPast
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.models.qwen2.modeling_qwen2 import CausalLMOutputWithPast

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron import sequence_parallel as sp_utils
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron import tensor_parallel as tp_utils
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import TransformerConfig, convert_config

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .layers import ParallelQwen2DecoderLayer, ParallelQwen2DecoderLayerRmPad, ParallelQwen2RMSNorm

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from Qwen2 pretrained checkpoint
"""


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# [EXPLAIN] `_make_causal_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Make causal mask used for bi-directional self-attention.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, tgt_len = input_ids_shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask_cond = torch.arange(mask.size(-1), device=device)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = mask.to(dtype)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
# [EXPLAIN] `_expand_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, src_len = mask.size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tgt_len = tgt_len if tgt_len is not None else src_len

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inverted_mask = 1.0 - expanded_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# [EXPLAIN] `ParallelQwen2Model` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2Model(nn.Module):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.padding_idx = config.pad_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layers = nn.ModuleList([ParallelQwen2DecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # [EXPLAIN] `_prepare_decoder_attention_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        combined_attention_mask = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if input_shape[-1] > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return combined_attention_mask

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """

        Args:
            input_ids: input ids. shape (batch_size, seq_length)
            attention_mask: attention_mask. shape (batch_size, seq_length)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size, seq_length = input_ids.shape
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = inputs_embeds

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, decoder_layer in enumerate(self.layers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = layer_outputs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = self.norm(hidden_states)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden_states


# [EXPLAIN] `ParallelQwen2ForCausalLM` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ForCausalLM(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = ParallelQwen2Model(config, megatron_config=megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = outputs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self.lm_head(hidden_states)[0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.float()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# [EXPLAIN] `ParallelQwen2ModelRmPad` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ModelRmPad(nn.Module):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.padding_idx = config.pad_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layers = nn.ModuleList([ParallelQwen2DecoderLayerRmPad(config, megatron_config) for _ in range(config.num_hidden_layers)])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs_embeds = inputs_embeds.transpose(0, 1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = inputs_embeds
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, decoder_layer in enumerate(self.layers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = layer_outputs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = self.norm(hidden_states)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden_states


# [EXPLAIN] `ParallelQwen2ForCausalLMRmPad` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ForCausalLMRmPad(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = ParallelQwen2ModelRmPad(config, megatron_config=megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._init_head(config)

    # [EXPLAIN] `_init_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_head(self, config: Qwen2Config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    # [EXPLAIN] `_forward_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self.lm_head(hidden_states)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return logits

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size, sequence_length = input_ids.shape

        # remove padding here
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states = outputs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self._forward_head(hidden_states)

        # remove padding from sequence parallel
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            totol_nnz = cu_seqlens[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = logits[:totol_nnz]  # (total_nnz_padded)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
        # add removed padding back
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = pad_input(logits, indices, batch_size, seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# [EXPLAIN] `ParallelQwen2ForValueRmPad` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ForValueRmPad(ParallelQwen2ForCausalLMRmPad):
    # [EXPLAIN] `_init_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_head(self, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    # [EXPLAIN] `_forward_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_head(self, hidden_states):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.float()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return logits

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = super().forward(input_ids, attention_mask, position_ids)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output.logits = torch.squeeze(output.logits, dim=-1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output


# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Support pipeline parallelism
"""


# [EXPLAIN] `ParallelQwen2ModelRmPadPP` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ModelRmPadPP(nn.Module):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]
    This model definition supports pipeline parallelism. To support pp and vpp,
    - This model only contains layer in this pp stage and vpp chunk
    - When calling get_model in Megatron, this rank will instantiate all the vpp chunks in this pp.
    Args:
        config: Qwen2Config
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig, pre_process, post_process):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.padding_idx = config.pad_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_process = pre_process
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.post_process = post_process
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pre_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.embed_tokens = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_size = megatron_config.pipeline_model_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vpp_size is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_layer_this_model = self.num_layer_vpp_chunk
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            offset = vpp_rank * (config.num_hidden_layers // vpp_size) + (pp_rank * self.num_layer_vpp_chunk)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_layer_this_model = self.num_layer_per_pp
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            offset = pp_rank * self.num_layer_per_pp

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layers = nn.ModuleList()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(self.num_layer_this_model):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer = ParallelQwen2DecoderLayerRmPad(config, megatron_config, layer_idx=i + offset)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.layers.add_module(f"{i}", layer)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if post_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.norm = ParallelQwen2RMSNorm(config, megatron_config)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.norm = None

    # [EXPLAIN] `set_input_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_input_tensor(self, input_tensor):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.input_tensor = input_tensor

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.pre_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

            # vocab parallel embedding will not do sequence parallel reduce-scatter in open source megatron
            # so need to deal with it by handle here:
            # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inputs_embeds = inputs_embeds.transpose(0, 1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.megatron_config.sequence_parallel:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = inputs_embeds
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # self.hidden_states should be passed by Megatron
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = self.input_tensor

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, decoder_layer in enumerate(self.layers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = layer_outputs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.post_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = self.norm(hidden_states)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden_states


# [EXPLAIN] `ParallelQwen2ForCausalLMRmPadPP` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ForCausalLMRmPadPP(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        config: Qwen2Config,
        megatron_config: ModelParallelConfig,
        pre_process,
        post_process,
        share_embeddings_and_output_weights,
    ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config: TransformerConfig = convert_config(config, megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = ParallelQwen2ModelRmPadPP(config, megatron_config=megatron_config, pre_process=pre_process, post_process=post_process)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vocab_size = config.vocab_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pre_process = pre_process
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.post_process = post_process
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if post_process:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._init_head(config)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pre_process or post_process:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.setup_embeddings_and_output_layer()

    # [EXPLAIN] `set_input_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_input_tensor(self, input_tensor):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(input_tensor) == 1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.model.set_input_tensor(input_tensor[0])

    # [EXPLAIN] `_init_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_head(self, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            **column_kwargs,
        )

    # [EXPLAIN] `setup_embeddings_and_output_layer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setup_embeddings_and_output_layer(self) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """
        # Set `is_embedding_or_output_parameter` attribute.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.pre_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.model.embed_tokens.weight.is_embedding_or_output_parameter = True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.post_process and self.lm_head.weight is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.lm_head.weight.is_embedding_or_output_parameter = True

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.share_embeddings_and_output_weights:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.shared_embedding_or_output_weight().shared_embedding = True

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.post_process and not self.pre_process:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.lm_head.weight.data.fill_(0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.lm_head.weight.shared = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.lm_head.weight.shared_embedding = True

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.is_initialized() and parallel_state.is_rank_in_embedding_group():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight = self.shared_embedding_or_output_weight()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight.data = weight.data.cuda()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.all_reduce(weight.data, group=parallel_state.get_embedding_group())

    # [EXPLAIN] `shared_embedding_or_output_weight` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.pre_process:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.model.embed_tokens.weight
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.post_process:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.lm_head.weight
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] `_forward_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # print(f'logits shape before forward_head: {hidden_states.shape}, vocab_size = {self.config.vocab_size}') # [4, 32, 4096]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_weight = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.share_embeddings_and_output_weights:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_weight = self.shared_embedding_or_output_weight()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self.lm_head(hidden_states, weight=output_weight)[0]
        # print(f'logits shape after forward_head: {logits.shape}') # [8, 32, 8]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return logits

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        # original input
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # Note that input_ids, attention_mask and position_ids should be passed to every pp layer.
        # In the first pp, input_ids will be used, in other pp layers hidden_states will be used inside self.model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size, sequence_length = input_ids.shape
        # remove padding here
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz+pad)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = self.model(
            input_ids=input_ids_rmpad,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.post_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = outputs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = self._forward_head(hidden_states)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension # torch.Size([8, 32, 16])

            # remove padding from sequence parallel
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.megatron_config.sequence_parallel:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                totol_nnz = cu_seqlens[-1]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits = logits[:totol_nnz]  # (total_nnz_padded)
            # add removed padding back. If input is already rmpad, we let the caller pad_input
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = pad_input(logits, indices, batch_size, seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return outputs


# [EXPLAIN] `ParallelQwen2ForValueRmPadPP` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelQwen2ForValueRmPadPP(ParallelQwen2ForCausalLMRmPadPP):
    # [EXPLAIN] `_init_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_head(self, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    # [EXPLAIN] `_forward_head` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _forward_head(self, hidden_states):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.float()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return logits

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.post_process:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output.logits = torch.squeeze(output.logits, dim=-1)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return output
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return output
