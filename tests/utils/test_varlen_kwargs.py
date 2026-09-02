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
"""Passing cu_seqlens instead of letting the attention re-derive it per layer.

HF's flash-attention path, handed position_ids, decides on the device whether
the sequences are packed and how long the longest is, then reads both on the
host because flash-attn needs Python ints. That is one device-to-host sync per
layer per forward -- 28 layers, doubled by gradient checkpointing recomputing
the forward inside the backward -- and the trace measures ~80 D2H copies per
micro-batch, each trailed by ~147 us of empty device.

unpad_input already computed both once. What has to hold is that they are only
handed over when handing them over is correct, because every failure here is
silent: wrong boundaries do not raise, they produce attention over the wrong
spans.
"""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor import dp_actor as mod
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _actor(**attrs):
    """An actor with only the fields _varlen_kwargs reads."""
    actor = object.__new__(mod.DataParallelPPOActor)
    actor.use_ulysses_sp = attrs.get("use_ulysses_sp", False)
    return actor


def test_the_boundaries_are_handed_over_when_that_is_safe():
    cu = torch.tensor([0, 10, 25], dtype=torch.int32)
    with mock.patch.object(mod, "_VARLEN_KWARGS", True), \
         mock.patch.object(mod, "_flash_attention_takes_varlen_kwargs", lambda: True):
        out = _actor()._varlen_kwargs(cu, 15)

    assert out["max_length_q"] == 15 and out["max_length_k"] == 15
    # q and k are the same sequence here: self-attention over one packed batch
    assert out["cu_seq_lens_q"] is cu and out["cu_seq_lens_k"] is cu


def test_ulysses_sequence_parallel_keeps_the_slow_path():
    """The sequence is split across ranks after this point, so boundaries taken
    on the unsplit batch describe a different tensor than the attention sees.
    verl's monkey_patch all-gathers position_ids for exactly that reason --
    handing over stale cu_seqlens would be wrong, not merely unhelpful, and
    wrong silently."""
    with mock.patch.object(mod, "_VARLEN_KWARGS", True), \
         mock.patch.object(mod, "_flash_attention_takes_varlen_kwargs", lambda: True):
        assert _actor(use_ulysses_sp=True)._varlen_kwargs(torch.tensor([0, 10]), 10) == {}


def test_a_transformers_that_does_not_name_the_kwargs_keeps_the_slow_path():
    """An older entry point takes unknown keywords into **kwargs and forwards
    them to flash-attn, which does not know them either. The check is by
    signature rather than by version string for that reason."""
    with mock.patch.object(mod, "_VARLEN_KWARGS", True), \
         mock.patch.object(mod, "_flash_attention_takes_varlen_kwargs", lambda: False):
        assert _actor()._varlen_kwargs(torch.tensor([0, 10]), 10) == {}


def test_a_backend_whose_unpad_returns_fewer_values_keeps_the_slow_path():
    """flash-attn 2.7 added a fifth return and the NPU shim has its own arity.
    The call site indexes rather than unpacks, and passes None when the values
    are not there -- an optimisation driven by a profile must not turn a working
    backend into a ValueError."""
    with mock.patch.object(mod, "_VARLEN_KWARGS", True), \
         mock.patch.object(mod, "_flash_attention_takes_varlen_kwargs", lambda: True):
        assert _actor()._varlen_kwargs(None, None) == {}


def test_it_can_be_turned_off_without_a_code_change():
    """It changes which code path computes the attention boundaries. If the two
    ever disagree the symptom is a loss curve, not an exception, so there has to
    be a way to rule it out from the command line."""
    with mock.patch.object(mod, "_VARLEN_KWARGS", False), \
         mock.patch.object(mod, "_flash_attention_takes_varlen_kwargs", lambda: True):
        assert _actor()._varlen_kwargs(torch.tensor([0, 10]), 10) == {}


def test_the_signature_probe_is_cached():
    """It runs once per micro-batch otherwise -- 77 times a step, each an
    inspect.signature over a transformers function."""
    mod._flash_attention_takes_varlen_kwargs.cache_clear()
    first = mod._flash_attention_takes_varlen_kwargs()
    assert mod._flash_attention_takes_varlen_kwargs() is first
    assert mod._flash_attention_takes_varlen_kwargs.cache_info().hits >= 1
