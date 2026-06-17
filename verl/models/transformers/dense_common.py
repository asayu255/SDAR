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

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def _rolled_labels(labels: Optional[torch.LongTensor], input_ids: Optional[torch.LongTensor]) -> torch.LongTensor:
    if labels is not None:
        return torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        return torch.roll(input_ids, shifts=-1, dims=-1)
    raise RuntimeError("To use the fused forward, either labels or input_ids must be provided.")


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
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

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
    hidden_states = outputs[0]
    rolled_labels = _rolled_labels(labels, input_ids)

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


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
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

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
    hidden_states = outputs[0]
    rolled_labels = _rolled_labels(labels, input_ids)

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
