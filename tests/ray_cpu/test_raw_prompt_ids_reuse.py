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
"""Reusing the prompt tokenisation instead of encoding the same string twice.

preprocess_single_sample tokenised the identical string once padded (for
input_ids) and once bare (for raw_prompt_ids) -- 252 redundant encodes per turn
on the calling thread. The non-pad tokens of the first are the second, but
raw_prompt_ids is what vLLM generates from, so the equality is verified on the
first calls of each process rather than assumed, and one mismatch disables the
reuse for good.
"""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

import agent_system.multi_turn_rollout.rollout_loop as rollout_loop  # noqa: E402


class _Tokenizer:
    """encode() returns a fixed id list and counts how often it was asked."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.encodes = 0

    def encode(self, text, add_special_tokens=False):
        self.encodes += 1
        return list(self.ids)


def _collector(truncation="left", max_prompt_length=8):
    self = types.SimpleNamespace()
    self.config = types.SimpleNamespace(
        data=types.SimpleNamespace(truncation=truncation, max_prompt_length=max_prompt_length)
    )
    self._raw_prompt_ids = rollout_loop.TrajectoryCollector._raw_prompt_ids.__get__(self)
    self._encode_raw_prompt = rollout_loop.TrajectoryCollector._encode_raw_prompt.__get__(self)
    return self


def _row(ids, width=8):
    """A left-padded input_ids/attention_mask pair holding ``ids``."""
    pad = width - len(ids)
    input_ids = torch.tensor([0] * pad + list(ids))
    mask = torch.tensor([0] * pad + [1] * len(ids))
    return input_ids, mask


@pytest.fixture(autouse=True)
def _fresh_state():
    rollout_loop._RAW_IDS_STATE.update(enabled=True, verified=0)
    yield
    rollout_loop._RAW_IDS_STATE.update(enabled=True, verified=0)


def test_the_first_calls_verify_then_the_encode_stops():
    tokenizer = _Tokenizer([5, 6, 7])
    collector = _collector()
    input_ids, mask = _row([5, 6, 7])

    for _ in range(rollout_loop._RAW_IDS_VERIFY):
        assert collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False) == [5, 6, 7]
    assert tokenizer.encodes == rollout_loop._RAW_IDS_VERIFY

    for _ in range(10):
        assert collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False) == [5, 6, 7]
    assert tokenizer.encodes == rollout_loop._RAW_IDS_VERIFY  # not one more


def test_a_mismatch_disables_the_reuse_for_good(capsys):
    """raw_prompt_ids is what vLLM generates from; a silent divergence would
    change every trajectory. The wrong answer must never be returned, and the
    fast path must not be retried."""
    tokenizer = _Tokenizer([5, 6, 7, 8])  # encode disagrees with the tensors
    collector = _collector()
    input_ids, mask = _row([5, 6, 7])

    assert collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False) == [5, 6, 7, 8]
    assert "[raw-ids]" in capsys.readouterr().out
    assert rollout_loop._RAW_IDS_STATE["enabled"] is False

    assert collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False) == [5, 6, 7, 8]
    assert tokenizer.encodes == 2  # every later call is the old path


def test_multimodal_rows_always_encode():
    """Their raw_prompt is a different string from the tokenised one."""
    tokenizer = _Tokenizer([1, 2])
    collector = _collector()
    input_ids, mask = _row([5, 6, 7])

    assert collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, True) == [1, 2]
    assert tokenizer.encodes == 1


def test_middle_truncation_always_encodes():
    """postprocess_data does not implement 'middle'; the two paths would differ."""
    tokenizer = _Tokenizer(list(range(20)))
    collector = _collector(truncation="middle", max_prompt_length=8)
    input_ids, mask = _row([5, 6, 7])

    got = collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False)
    assert got == [0, 1, 2, 3] + [16, 17, 18, 19]  # middle-truncated encode
    assert tokenizer.encodes == 1


def test_the_kill_switch_forces_the_old_path():
    rollout_loop._RAW_IDS_STATE["enabled"] = False
    tokenizer = _Tokenizer([5, 6, 7])
    collector = _collector()
    input_ids, mask = _row([5, 6, 7])

    collector._raw_prompt_ids(tokenizer, "p", input_ids, mask, False)
    assert tokenizer.encodes == 1


def test_the_encode_fallback_still_truncates_left():
    tokenizer = _Tokenizer(list(range(20)))
    collector = _collector(truncation="left", max_prompt_length=8)
    assert collector._encode_raw_prompt(tokenizer, "p") == list(range(12, 20))
