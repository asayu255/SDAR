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
    seed_for_prompt,
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


# --------------------------------------------------------------------------- #
# seed_for_prompt
# --------------------------------------------------------------------------- #
def test_the_same_prompt_and_row_always_seed_the_same():
    """That stability is the whole point: the draw stops depending on arrival order."""
    assert seed_for_prompt([5, 6, 7], 3) == seed_for_prompt([5, 6, 7], 3)


def test_identical_prompts_on_different_rows_seed_differently():
    """Validation with val_kwargs.n > 1 repeats a row in the DataProto itself.

    The n copies arrive as n requests with byte-identical prompts. Seeded on the
    prompt alone they would draw identically, take the same action, see the same
    observation -- n samples collapsed into n copies of one, and the sample
    variance would shrink, which reads as a stability win rather than a bug.
    """
    seeds = {seed_for_prompt([5, 6, 7], row) for row in range(4)}
    assert len(seeds) == 4


def test_different_prompts_on_the_same_row_seed_differently():
    assert seed_for_prompt([5, 6, 7], 1) != seed_for_prompt([5, 6, 8], 1)


def test_a_prompt_is_not_confused_with_a_row_number():
    """The row is appended as fixed-width bytes, not concatenated into the ids."""
    assert seed_for_prompt([1, 2], 3) != seed_for_prompt([1, 2, 3], 0)


def test_the_seed_is_in_range_for_vllm():
    for ids, row in (([0], 0), ([2 ** 20, 7], 251), ([1] * 4096, 9)):
        seed = seed_for_prompt(ids, row)
        assert isinstance(seed, int) and 0 <= seed <= 0x7FFFFFFF


def test_numpy_and_list_prompts_seed_the_same():
    assert seed_for_prompt(np.array([4, 5, 6]), 2) == seed_for_prompt([4, 5, 6], 2)


# --------------------------------------------------------------------------- #
# what the driver has to forward
# --------------------------------------------------------------------------- #
class _RecordingMeta(dict):
    """A meta_info that remembers which keys were asked for."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


class _SeedValKwargs:
    top_k = 20
    top_p = 0.8
    temperature = 0.7


def test_the_driver_forwards_every_meta_key_the_sampling_choice_reads():
    """Guard against a fourth key being added here and silently not crossing.

    _PUMP_META_KEYS is what the driver puts on the wire. If sampling_kwargs_for
    grows a key that is not in it, the rank would derive different sampling
    params from the blocking path -- and nothing would say so.
    """
    from agent_system.multi_turn_rollout.rollout_loop import _PUMP_META_KEYS

    read = set()
    for meta in (
        {"do_sample": False},
        {"do_sample": True, "validate": True, "temperature": 0.5},
        {"do_sample": True, "validate": False},
    ):
        recorder = _RecordingMeta(meta)
        sampling_kwargs_for(recorder, _SeedValKwargs())
        read |= recorder.read

    assert read <= set(_PUMP_META_KEYS), (
        f"sampling_kwargs_for reads {sorted(read - set(_PUMP_META_KEYS))}, "
        f"which the driver does not forward"
    )


# --------------------------------------------------------------------------- #
# the two shapes the two paths hand over
# --------------------------------------------------------------------------- #
def test_the_array_path_and_the_list_path_agree():
    """The pump sends int32 arrays; the blocking path sends lists.

    They must pad to the same tensor, because the only reason the pump sends
    arrays is transport cost -- if the padded result differed, the two paths
    would generate differently for a reason that has nothing to do with
    generation.
    """
    import numpy as np

    from verl.workers.rollout.generation_output import _pad_rows

    rows = [[1, 2, 3], [4], []]
    arrays = [np.asarray(row, dtype=np.int32) for row in rows]
    assert _pad_rows(arrays, 0, 5).tolist() == _pad_rows(rows, 0, 5).tolist()
    assert _pad_rows(arrays, 0, 5).dtype == _pad_rows(rows, 0, 5).dtype


def test_a_row_longer_than_response_length_is_not_truncated():
    """max_length is a floor on the width, not a cap on a row."""
    import numpy as np

    from verl.workers.rollout.generation_output import _pad_rows

    long_row = [np.arange(7, dtype=np.int32)]
    assert _pad_rows(long_row, 0, 3).shape == (1, 7)


def test_a_mixed_batch_falls_back_rather_than_guessing():
    """One list among the arrays means the fast path does not apply."""
    import numpy as np

    from verl.workers.rollout.generation_output import _pad_rows

    mixed = [np.asarray([1, 2], dtype=np.int32), [3]]
    assert _pad_rows(mixed, 0, 3).tolist() == [[1, 2, 0], [3, 0, 0]]


def test_the_seed_does_not_change_shape_by_shape():
    """seed_for_prompt now takes the array directly instead of list()-ing it.

    An array and the list of the same ids have to seed identically, or the
    transport change would silently move what every prompt samples.
    """
    import numpy as np

    from verl.workers.rollout.generation_output import seed_for_prompt

    ids = [11, 22, 33]
    assert seed_for_prompt(np.asarray(ids, dtype=np.int32), 4) == seed_for_prompt(ids, 4)
    assert seed_for_prompt(np.asarray(ids, dtype=np.int64), 4) == seed_for_prompt(ids, 4)
