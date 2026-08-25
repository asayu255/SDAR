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
"""The assembly two rollout paths share has to mean one thing.

`vLLMRollout.generate_sequences` used to do this arithmetic inline. The pumped
path assembles the same rows on the driver, so the arithmetic moved into a
function both call. These tests pin what that function produces -- a mask that
ends one token late, or a position_id that restarts instead of continuing, is
silent everywhere downstream.
"""

import numpy as np
import pytest
import torch

from verl.workers.rollout.generation_output import (
    assemble_generation_output,
    repeat_interleave,
    sampling_kwargs_for,
)

PAD = 0
EOS = 2


def _prompts(rows):
    """Left-padded prompts, the layout the rollout hands the engine."""
    idx = torch.tensor(rows)
    attention_mask = (idx != PAD).to(torch.int64)
    # position_ids restart at 0 on the first real token, as verl builds them.
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    return idx, attention_mask, position_ids


def test_response_is_padded_to_the_configured_length():
    idx, mask, pos = _prompts([[PAD, 7, 8]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11, 12, EOS]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=6,
    )
    assert out.batch["responses"].tolist() == [[11, 12, EOS, PAD, PAD, PAD]]
    assert out.batch["prompts"].tolist() == idx.tolist()
    assert out.batch["input_ids"].tolist() == [[PAD, 7, 8, 11, 12, EOS, PAD, PAD, PAD]]


def test_attention_mask_covers_the_response_through_its_eos_and_no_further():
    idx, mask, pos = _prompts([[PAD, 7, 8]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11, EOS]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=4,
    )
    # prompt half unchanged, response half on through eos then off
    assert out.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 0, 0]]


def test_position_ids_continue_from_the_last_prompt_position():
    idx, mask, pos = _prompts([[PAD, PAD, 7]])
    assert pos.tolist() == [[0, 0, 0]]
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11, 12]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=3,
    )
    # the response continues from the prompt's LAST position id, not from its length
    assert out.batch["position_ids"].tolist() == [[0, 0, 0, 1, 2, 3]]


def test_rows_stay_paired_with_their_own_prompt():
    idx, mask, pos = _prompts([[PAD, 7, 8], [5, 6, 9]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11], [22]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=2,
    )
    assert out.batch["input_ids"].tolist() == [[PAD, 7, 8, 11, PAD], [5, 6, 9, 22, PAD]]


def test_n_greater_than_one_widens_prompts_to_match_the_samples():
    idx, mask, pos = _prompts([[PAD, 7, 8], [5, 6, 9]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11], [12], [21], [22]],
        non_tensor_batch={"tools_kwargs": np.array(["a", "b"], dtype=object)},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=1,
        n=2,
    )
    # prompt rows repeat_interleaved, so sample j of prompt i sits next to prompt i
    assert out.batch["prompts"].tolist() == [[PAD, 7, 8], [PAD, 7, 8], [5, 6, 9], [5, 6, 9]]
    assert out.batch["responses"].tolist() == [[11], [12], [21], [22]]
    assert list(out.non_tensor_batch["tools_kwargs"]) == ["a", "a", "b", "b"]


def test_a_missing_row_is_an_error_not_a_silent_shift():
    idx, mask, pos = _prompts([[PAD, 7, 8], [5, 6, 9]])
    with pytest.raises(ValueError, match="expected 2 responses"):
        assemble_generation_output(
            idx=idx,
            attention_mask=mask,
            position_ids=pos,
            response_token_ids=[[11]],
            non_tensor_batch={},
            eos_token_id=EOS,
            pad_token_id=PAD,
            response_length=2,
        )


def test_rollout_log_probs_are_padded_with_minus_one_and_float32():
    idx, mask, pos = _prompts([[PAD, 7, 8]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11, 12]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=4,
        rollout_log_probs=[[-0.5, -0.25]],
    )
    lp = out.batch["rollout_log_probs"]
    assert lp.dtype == torch.float32
    assert lp.tolist() == [[-0.5, -0.25, -1.0, -1.0]]


def test_no_rollout_log_probs_key_when_none_were_asked_for():
    idx, mask, pos = _prompts([[PAD, 7, 8]])
    out = assemble_generation_output(
        idx=idx,
        attention_mask=mask,
        position_ids=pos,
        response_token_ids=[[11]],
        non_tensor_batch={},
        eos_token_id=EOS,
        pad_token_id=PAD,
        response_length=1,
    )
    assert "rollout_log_probs" not in out.batch.keys()


def test_repeat_interleave_handles_both_tensors_and_object_arrays():
    assert repeat_interleave(torch.tensor([[1], [2]]), 2).tolist() == [[1], [1], [2], [2]]
    assert list(repeat_interleave(np.array(["a", "b"], dtype=object), 2)) == ["a", "a", "b", "b"]


class _ValKwargs:
    top_k = 5
    top_p = 0.9
    temperature = 0.7


def test_greedy_beats_validate():
    # do_sample=False is greedy no matter what the validation config says --
    # reading these two in the other order would sample during evaluation.
    kwargs = sampling_kwargs_for({"do_sample": False, "validate": True}, _ValKwargs)
    assert kwargs["temperature"] == 0
    assert kwargs["n"] == 1
    assert kwargs["top_k"] == -1


def test_validate_takes_the_val_kwargs_and_lets_meta_info_override_temperature():
    assert sampling_kwargs_for({"validate": True}, _ValKwargs) == {
        "top_k": 5,
        "top_p": 0.9,
        "temperature": 0.7,
        "n": 1,
    }
    hot = sampling_kwargs_for({"validate": True, "temperature": 1.3}, _ValKwargs)
    assert hot["temperature"] == 1.3


def test_plain_training_call_overrides_nothing():
    # Empty, not a dict of defaults: the caller's own **kwargs have to survive.
    assert sampling_kwargs_for({}, _ValKwargs) == {}
