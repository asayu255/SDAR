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

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
"""Fused-kernel forward passes for dense (non-vision) causal LMs.

This mirrors the per-model fused forwards (e.g. ``qwen2_vl.py``) for plain
``*ForCausalLM`` models such as Qwen2/Qwen3/Llama. Instead of materializing the
full ``(num_tokens, vocab_size)`` logits tensor and then gathering log-probs,
it runs the transformer backbone (``self.model``) to obtain hidden states and
computes per-token ``log_probs``/``entropy`` directly with a fused
linear-cross-entropy kernel. The returned object exposes ``.log_probs`` and
``.entropy`` (``.logits`` stays ``None``), which is what ``DataParallelPPOActor``
consumes when ``use_fused_kernels=True``.

Assumes Ulysses sequence parallelism is disabled (the default); under SP>1 the
hidden states would need to be gathered before the fused projection.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.modeling_outputs import CausalLMOutputWithPast


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `CausalLMOutputForPPO` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs: Optional[torch.FloatTensor] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy: Optional[torch.FloatTensor] = None


# [EXPLAIN] `_rolled_labels` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _rolled_labels(labels: Optional[torch.LongTensor], input_ids: Optional[torch.LongTensor]) -> torch.LongTensor:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if labels is not None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.roll(labels, shifts=-1, dims=-1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif input_ids is not None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.roll(input_ids, shifts=-1, dims=-1)
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise RuntimeError("To use the fused forward, either labels or input_ids must be provided.")


# [EXPLAIN] `forward_with_torch_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position=None,
    logits_to_keep: int = 0,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rolled_labels = _rolled_labels(labels, input_ids)

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
    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


# [EXPLAIN] `forward_with_triton_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position=None,
    logits_to_keep: int = 0,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_states = outputs[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rolled_labels = _rolled_labels(labels, input_ids)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
