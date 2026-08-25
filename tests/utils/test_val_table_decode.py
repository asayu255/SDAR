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
"""The validation sample table's decodes, and the batch response digest.

Every validation row's prompt and response were decoded to feed a table capped
at trainer.log_val_generations samples -- which this arm sets to 0, so 104,000
detokenisations per evaluation fed nothing, on the thread the pipeline waits
between batches for.

The digest exists for the generate-merge adoption gate: merging regroups rows
inside a call and greedy decoding is only reduction-order deterministic, so
merged and unmerged runs must be shown to produce the same tokens, not only the
same scores. Batches retire in submission order, so equal digests at equal
batch indices settle it without storing 52,000 generations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402


class _CountingTokenizer:
    def __init__(self):
        self.decodes = 0

    def decode(self, ids, **kwargs):
        self.decodes += 1
        return "text"


def test_a_disabled_table_decodes_nothing():
    tokenizer = _CountingTokenizer()
    rows = torch.ones(252, 8, dtype=torch.long)

    assert RayPPOTrainer._decode_for_val_table(tokenizer, rows, 0) == []
    assert tokenizer.decodes == 0


def test_an_enabled_table_decodes_every_row():
    """The table samples AFTER collection, so the rows all have to exist."""
    tokenizer = _CountingTokenizer()
    rows = torch.ones(16, 8, dtype=torch.long)

    texts = RayPPOTrainer._decode_for_val_table(tokenizer, rows, 10)
    assert len(texts) == 16
    assert tokenizer.decodes == 16


def test_the_digest_is_stable_and_sensitive():
    a = torch.arange(64, dtype=torch.long).reshape(4, 16)
    b = a.clone()
    c = a.clone()
    c[2, 7] += 1  # one token differs

    assert RayPPOTrainer._response_digest(a) == RayPPOTrainer._response_digest(b)
    assert RayPPOTrainer._response_digest(a) != RayPPOTrainer._response_digest(c)
    assert len(RayPPOTrainer._response_digest(a)) == 12
