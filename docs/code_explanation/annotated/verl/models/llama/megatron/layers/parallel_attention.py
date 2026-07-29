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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from einops import rearrange
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flash_attn.layers.rotary import apply_rotary_emb
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import ModelParallelConfig, tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import LlamaConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.utils import is_flash_attn_2_available

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.llama.megatron.layers.parallel_linear import QKVParallelLinear
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron import tensor_parallel as tp_utils


# [EXPLAIN] `LlamaRotaryEmbedding` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LlamaRotaryEmbedding(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dim = dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_position_embeddings = max_position_embeddings
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base = base
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    # [EXPLAIN] `_set_cos_sin_cache` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_seq_len_cached = seq_len
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        emb = torch.cat((freqs, freqs), dim=-1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if seq_len > self.max_seq_len_cached:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# [EXPLAIN] `LlamaLinearScalingRotaryEmbedding` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.scaling_factor = scaling_factor
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(dim, max_position_embeddings, base, device)

    # [EXPLAIN] `_set_cos_sin_cache` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_seq_len_cached = seq_len
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        t = t / self.scaling_factor

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        emb = torch.cat((freqs, freqs), dim=-1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# [EXPLAIN] `LlamaDynamicNTKScalingRotaryEmbedding` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.scaling_factor = scaling_factor
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(dim, max_position_embeddings, base, device)

    # [EXPLAIN] `_set_cos_sin_cache` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_seq_len_cached = seq_len

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if seq_len > self.max_position_embeddings:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        emb = torch.cat((freqs, freqs), dim=-1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# [EXPLAIN] `LlamaLlama3ScalingRotaryEmbedding` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LlamaLlama3ScalingRotaryEmbedding(LlamaRotaryEmbedding):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dim, config, max_position_embeddings=2048, base=10000, device=None):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(dim, max_position_embeddings, base, device)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.factor = config.rope_scaling["factor"]  # `8` in the original implementation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.high_freq_factor = config.rope_scaling["high_freq_factor"]  # `1` in the original implementation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.low_freq_factor = config.rope_scaling["low_freq_factor"]  # `4` in the original implementation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        high_freq_wavelen = self.old_context_len / self.high_freq_factor

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wavelen = 2 * math.pi / self.inv_freq
        # wavelen < high_freq_wavelen: do nothing; wavelen > low_freq_wavelen: divide by factor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / self.factor, self.inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (self.high_freq_factor - self.low_freq_factor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / self.factor + smooth_factor * inv_freq_llama
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())


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


# [EXPLAIN] `apply_rotary_pos_emb` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return q_embed, k_embed


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


# [EXPLAIN] `ParallelLlamaAttention` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelLlamaAttention(nn.Module):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hidden_size = config.hidden_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_heads = config.num_attention_heads
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.head_dim = self.hidden_size // self.num_heads
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_key_value_heads = config.num_key_value_heads
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_position_embeddings = config.max_position_embeddings
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rope_theta = config.rope_theta

        # assign values after tp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.num_heads % tp_size == 0, f"num_head must be divisible by tp_size. Got num_head={self.num_heads}, tp_size={tp_size}"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.num_key_value_heads % tp_size == 0, f"num_key_value_heads must be divisible by tp_size. Got num_key_value_heads={self.num_key_value_heads}, tp_size={tp_size}"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_heads_per_tp = self.num_heads // tp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hidden_size_per_tp = self.hidden_size // tp_size

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if (self.head_dim * self.num_heads) != self.hidden_size:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert row_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # [self.q_size, self.k_size, self.v_size]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.q_size = self.num_heads_per_tp * self.head_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._init_rope()

    # [EXPLAIN] `_init_rope` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_rope(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.rope_scaling is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rope_type_key = "type" if "type" in self.config.rope_scaling else "rope_type"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scaling_type = self.config.rope_scaling[rope_type_key]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scaling_factor = self.config.rope_scaling["factor"]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if scaling_type == "linear":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif scaling_type == "dynamic":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif scaling_type == "llama3":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.rotary_emb = LlamaLlama3ScalingRotaryEmbedding(
                    self.head_dim,
                    self.config,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # [EXPLAIN] `_shape` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz, q_len, _ = hidden_states.size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        qkv = self.qkv_proj(hidden_states)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = query_states.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kv_seq_len = key_states.shape[-2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attn_weights.size() != (bsz, self.num_heads_per_tp, q_len, kv_seq_len):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads_per_tp, q_len, kv_seq_len)}, but is {attn_weights.size()}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attention_mask is not None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = torch.matmul(attn_weights, value_states)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if attn_output.size() != (bsz, self.num_heads_per_tp, q_len, self.head_dim):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads_per_tp, q_len, self.head_dim)}, but is {attn_output.size()}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_tp)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output = self.o_proj(attn_output)[0]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return attn_output


# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Remove padding Attention
- Using Flash-attn 2
- Compatible with sequence parallel
"""


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_flash_attn_2_available():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn import flash_attn_varlen_func
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# [EXPLAIN] `apply_rotary_pos_emb_rmpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_rotary_pos_emb_rmpad(q, k, cos, sin, position_ids, indices, sequence_length):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = position_ids.shape[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q = pad_input(q, indices, batch_size, sequence_length)  # (batch_size, seqlen, num_head, head_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k = pad_input(k, indices, batch_size, sequence_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_embed = index_first_axis(rearrange(q_embed, "b s ... -> (b s) ..."), indices)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_embed = index_first_axis(rearrange(k_embed, "b s ... -> (b s) ..."), indices)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return q_embed, k_embed


# use flash-attn rotary embeddings with rmpad
# cos/sin shoudl be: (seq_length, rotary_dim / 2)
# [EXPLAIN] `apply_rotary_pos_emb_rmpad_flash` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    q_embed = apply_rotary_emb(q, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_embed = apply_rotary_emb(k, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return q_embed, k_embed


# [EXPLAIN] `ParallelLlamaAttentionRmPad` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelLlamaAttentionRmPad(ParallelLlamaAttention):
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen_in_batch: int = None,
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_nnz, _, _ = hidden_states.size()  # This is the total_nnz padded after sequence parallel

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        qkv = self.qkv_proj(hidden_states)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)  # (total_nnz, 1, hidden_size)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_nnz = cu_seqlens[-1]  # total_nnz before sp padding
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query_states = query_states[:total_nnz]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            key_states = key_states[:total_nnz]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            value_states = value_states[:total_nnz]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states = query_states.view(total_nnz, self.num_heads_per_tp, self.head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key_states = key_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        value_states = value_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cos, sin = cos[:, : cos.shape[1] // 2], sin[:, : sin.shape[1] // 2]  # flash attn only needs half
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_states, key_states = apply_rotary_pos_emb_rmpad_flash(query_states, key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch)
        # query_states, key_states = apply_rotary_pos_emb_rmpad(query_states, key_states, cos, sin, position_ids, indices,

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_dtype = query_states.dtype
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if input_dtype == torch.float32:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            query_states = query_states.to(torch.float16)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            key_states = key_states.to(torch.float16)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            value_states = value_states.to(torch.float16)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=True,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output_unpad = attn_output_unpad.to(input_dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, 1, self.hidden_size_per_tp).contiguous()

        # sequence parallel reduce_scatter is performed inside RowColumnParallel if enabled
        # Here we need to repad
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.megatron_config.sequence_parallel:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attn_output_unpad = self.o_proj(attn_output_unpad)[0]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return attn_output_unpad
