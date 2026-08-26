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
"""Turn generated token ids into the DataProto the rollout loop consumes.

Everything the rollout does after the engine hands back token ids is a pure
function of the prompts and those ids: pad the responses to the configured
length, extend position_ids past the prompt, mask the response at its first eos,
concatenate. No engine state is read.

It lives here rather than inline in ``vLLMRollout.generate_sequences`` because a
second caller now needs it. The blocking path collects a whole batch from one
``LLM.generate`` call; the pumped path collects the same rows one request at a
time and assembles them on the driver. Two copies of this arithmetic would be
two chances to drift, and a drift here is silent -- a position_id off by one or
a mask that ends a token late does not raise, it just trains on the wrong thing.
So there is one copy and both paths call it.

There is no vllm import in this module on purpose: the driver assembles too, and
the tests run on CPU.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


def repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


def sampling_kwargs_for(meta_info: Dict[str, Any], val_kwargs: Any) -> Dict[str, Any]:
    """The sampling overrides ``generate_sequences`` derives from meta_info.

    Greedy wins over validation: ``do_sample=False`` means greedy no matter what
    the validation config asks for, which is how the blocking path reads it.
    An empty dict means "leave the configured sampling params alone".
    """
    if not meta_info.get("do_sample", True):
        return {
            "best_of": 1,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "temperature": 0,
            "n": 1,
        }
    if meta_info.get("validate", False):
        return {
            "top_k": val_kwargs.top_k,
            "top_p": val_kwargs.top_p,
            "temperature": meta_info.get("temperature", val_kwargs.temperature),
            "n": 1,
        }
    return {}


def seed_for_prompt(prompt_token_ids: Any, row: int = 0) -> int:
    """A sampling seed that depends on the prompt and the row, and nothing else.

    Under the pump, which requests share a decode step is decided by arrival
    timing, so a generator shared across whatever happens to be resident draws
    differently for the same prompt from one run to the next. Keying the seed to
    the prompt removes that source of drift: the same prompt draws the same way
    whenever it arrives.

    THE ROW IS NOT OPTIONAL DECORATION. Validation with ``val_kwargs.n > 1``
    repeats each row n times in the DataProto itself
    (``RayPPOTrainer._validate``: ``test_batch.repeat(...)``), not through
    SamplingParams.n -- so the n copies of one row reach the engine as n
    requests with BYTE-IDENTICAL prompts. Seeded on the prompt alone they would
    all draw the same tokens, take the same action, see the same observation and
    build the same next prompt: n samples collapsed into n copies of one, with
    nothing raised and the sample variance moving in the direction that looks
    like a win. Mixing the row in keeps the copies distinct while keeping every
    one of them stable across runs.

    This does not make the pumped path reproducible -- the logits still move
    with batch composition, which is the same reason merging changed
    generation. It removes a second source of drift stacked on top of that one.

    Lives here rather than beside the engine so the driver and the CPU tests can
    reach it without vllm, and so there is one definition of it rather than two.
    """
    import zlib

    ids = np.asarray(list(prompt_token_ids), dtype=np.int64)
    return zlib.crc32(ids.tobytes() + int(row).to_bytes(8, "little", signed=True)) & 0x7FFFFFFF


def assemble_generation_output(
    *,
    idx: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_token_ids: List[List[int]],
    non_tensor_batch: Dict[str, Any],
    eos_token_id: Any,
    pad_token_id: int,
    response_length: int,
    n: int = 1,
    rollout_log_probs: Optional[List[List[float]]] = None,
) -> DataProto:
    """Build the generation DataProto from prompts and response token ids.

    ``idx``/``attention_mask``/``position_ids`` are the left-padded prompt
    tensors, one row per prompt. ``response_token_ids`` has ``n`` rows per
    prompt, in prompt order -- the layout ``LLM.generate`` returns and the one
    ``repeat_interleave`` below assumes when it widens the prompts to match.
    """
    if n > 1:
        idx = repeat_interleave(idx, n)
        attention_mask = repeat_interleave(attention_mask, n)
        position_ids = repeat_interleave(position_ids, n)
        # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
        if "tools_kwargs" in non_tensor_batch:
            non_tensor_batch["tools_kwargs"] = repeat_interleave(non_tensor_batch["tools_kwargs"], n)

    batch_size = idx.size(0)
    if len(response_token_ids) != batch_size:
        raise ValueError(f"expected {batch_size} responses for {batch_size} prompt rows, got {len(response_token_ids)}")

    response = pad_2d_list_to_length(response_token_ids, pad_token_id, max_length=response_length).to(idx.device)
    if rollout_log_probs is not None:
        rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=response_length).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)

    seq = torch.cat([idx, response], dim=-1)

    delta_position_id = torch.arange(1, response.size(1) + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

    # prompt: left pad + response: right pad
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
    response_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    tensors = {
        "prompts": idx,
        "responses": response,
        "input_ids": seq,  # here input_ids become the whole sentences
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if rollout_log_probs is not None:
        # we will recompute old log prob with actor
        tensors["rollout_log_probs"] = rollout_log_probs

    return DataProto(batch=TensorDict(tensors, batch_size=batch_size), non_tensor_batch=non_tensor_batch)
