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
"""Detokenising only the sample that gets printed.

The reward manager prints one example per data source per call, at ten percent
odds -- roughly one row of a 252-row validation batch. It decoded every row's
prompt and response first: 504 detokenisations a batch, 503 of them discarded,
on the thread the whole pipeline waits between batches for.

The scoring itself never needed the text. It reads episode_rewards and
episode_lengths off the batch, and the strings exist for the print alone.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from verl import DataProto  # noqa: E402
from agent_system.reward_manager.episode import EpisodeRewardManager  # noqa: E402


class _CountingTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.decodes = 0

    def decode(self, ids, **kwargs):
        self.decodes += 1
        return "text"


def _batch(rows, prompt_len=8, response_len=4):
    total = prompt_len + response_len
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(rows, prompt_len, dtype=torch.long),
            "responses": torch.ones(rows, response_len, dtype=torch.long),
            "attention_mask": torch.ones(rows, total, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["popqa"] * rows, dtype=object),
            "episode_rewards": np.array([1.0] * rows, dtype=object),
            "episode_lengths": np.array([1.0] * rows, dtype=object),
        },
    )


def test_scoring_does_not_decode_every_row(monkeypatch):
    """252 rows scored used to cost 504 detokenisations; the score needs none."""
    monkeypatch.setattr(np.random, "random", lambda: 1.0)  # never prints
    tokenizer = _CountingTokenizer()
    manager = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1)

    reward = manager(_batch(64))

    assert tokenizer.decodes == 0
    assert reward.shape[0] == 64


def test_the_printed_sample_is_still_decoded(monkeypatch, capsys):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # always prints
    tokenizer = _CountingTokenizer()
    manager = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1)

    manager(_batch(64))

    out = capsys.readouterr().out
    assert "[popqa][prompt] text" in out
    assert "[popqa][response] text" in out
    # one example, so one prompt and one response -- not 128
    assert tokenizer.decodes == 2


def test_the_scores_are_unchanged_by_not_decoding(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 1.0)
    manager = EpisodeRewardManager(tokenizer=_CountingTokenizer(), num_examine=1)

    reward = manager(_batch(4))

    # the score lands on the last valid response token of each row
    assert reward[:, 3].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert reward.sum().item() == 4.0
