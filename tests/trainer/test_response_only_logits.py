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
"""lm_head runs on the rows whose logits are actually read.

``_forward_micro_batch`` slices everything it returns to
``[:, -response_length - 1 : -1]``, so with a 4096-token prompt cap and a
512-token response cap roughly three quarters of the (rows, 151936) projection
was built and dropped -- forward and backward, and it is the step's largest
activation. ``response_only_logits`` moves the row selection in front of the
projection instead.

The mechanism is a row map, so that is what these test: that
``response_row_selection`` picks exactly the rows the old slice kept, that its
three return values agree with each other, and that reading a per-response-token
input through ``sel_slot`` is the same as scattering it across the full sequence
and gathering it back -- the step ``_topk_from_response_logits`` takes the short
way round.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor.dp_actor import _supports_logits_to_keep, response_row_selection
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _unpad_indices(attention_mask):
    """What flash-attn's unpad_input returns for `indices`: the flattened
    (batch * seqlen) positions of the non-padding tokens, ascending."""
    return torch.nonzero(attention_mask.flatten(), as_tuple=True)[0]


def _left_padded_mask(prompt_lens, response_lens, seqlen, response_length):
    """The layout this stack produces: prompts left-padded, responses right-padded."""
    mask = torch.zeros((len(prompt_lens), seqlen), dtype=torch.long)
    prompt_end = seqlen - response_length
    for row, (p_len, r_len) in enumerate(zip(prompt_lens, response_lens)):
        mask[row, prompt_end - p_len : prompt_end] = 1
        mask[row, prompt_end : prompt_end + r_len] = 1
    return mask


def test_the_selected_rows_are_the_ones_the_old_path_kept():
    """The full-logits path built every packed row, re-padded, then took the
    response window. Selecting first has to leave the same rows."""
    seqlen, response_length = 16, 5
    mask = _left_padded_mask([6, 3], [5, 4], seqlen, response_length)
    indices = _unpad_indices(mask)

    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)

    lo = seqlen - response_length - 1
    seq_pos = indices % seqlen
    expected = torch.nonzero((seq_pos >= lo) & (seq_pos < seqlen - 1), as_tuple=True)[0]
    assert torch.equal(sel, expected)
    assert torch.equal(sel_indices, indices[sel])
    assert torch.equal(sel_slot, seq_pos[sel] - lo)


def test_the_window_is_offset_by_one_because_a_logit_predicts_the_next_token():
    """Predicting the response_length response tokens needs positions
    seqlen - response_length - 1 .. seqlen - 2, i.e. the last prompt position is
    in and the last response position is out."""
    seqlen, response_length = 16, 5
    mask = torch.ones((1, seqlen), dtype=torch.long)  # nothing padded: positions == columns

    sel, _, _ = response_row_selection(_unpad_indices(mask), seqlen, response_length)

    assert sel.tolist() == [10, 11, 12, 13, 14]
    assert len(sel) == response_length
    assert seqlen - 1 not in sel.tolist()  # the final position predicts nothing kept


def test_padding_shrinks_the_selection_rather_than_shifting_it():
    """Response padding means fewer packed rows in the window, not different ones.
    pad_input zero-fills the missing slots, which is what the full path put there."""
    seqlen, response_length = 16, 5
    full = _left_padded_mask([4], [5], seqlen, response_length)
    short = _left_padded_mask([4], [2], seqlen, response_length)

    sel_full, _, slot_full = response_row_selection(_unpad_indices(full), seqlen, response_length)
    sel_short, _, slot_short = response_row_selection(_unpad_indices(short), seqlen, response_length)

    assert len(sel_short) < len(sel_full)
    # Same slots, truncated: the rows that remain sit where they sat.
    assert slot_short.tolist() == slot_full.tolist()[: len(slot_short)]


def test_a_row_with_no_response_tokens_selects_nothing_from_it():
    seqlen, response_length = 16, 5
    mask = _left_padded_mask([4, 4], [5, 0], seqlen, response_length)
    indices = _unpad_indices(mask)

    _, sel_indices, _ = response_row_selection(indices, seqlen, response_length)

    # Row 1 contributes only its last prompt position (which predicts the first
    # response token); it has no response tokens of its own.
    assert (sel_indices // seqlen == 1).sum() == 1


def test_reading_a_per_response_input_through_sel_slot_matches_the_long_way():
    """``_topk_from_response_logits`` indexes topk_ids at (sample, slot) instead of
    scattering it into (bs, seqlen, k) and gathering that back through `indices`.
    The two have to be the same map."""
    seqlen, response_length, k = 16, 5, 3
    bs = 2
    mask = _left_padded_mask([6, 3], [5, 4], seqlen, response_length)
    indices = _unpad_indices(mask)
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
    topk_ids = torch.randint(0, 1000, (bs, response_length, k))

    short_way = topk_ids[sel_indices // seqlen, sel_slot]

    full = torch.zeros((bs, seqlen, k), dtype=topk_ids.dtype)
    full[:, -response_length - 1 : -1, :] = topk_ids
    long_way = full.reshape(bs * seqlen, k)[indices][sel]

    assert torch.equal(short_way, long_way)


def test_logits_to_keep_support_is_detected_through_the_wrappers():
    """The model arrives wrapped (FSDP, possibly torch.compile), so the check has
    to unwrap before reading the signature -- it runs at worker init so that the
    flag fails there rather than in the first micro-batch."""

    class _Inner(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep=None):
            return input_ids

    class _NoSupport(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._fsdp_wrapped_module = inner

        def forward(self, *a, **k):
            return self._fsdp_wrapped_module(*a, **k)

    assert _supports_logits_to_keep(_Inner())
    assert _supports_logits_to_keep(_Wrapper(_Inner()))
    assert _supports_logits_to_keep(_Wrapper(_Wrapper(_Inner())))
    assert not _supports_logits_to_keep(_Wrapper(_NoSupport()))


def test_the_model_returns_logits_for_exactly_the_rows_it_is_given():
    """The contract the whole mechanism rests on.

    HF resolves the argument as ``slice(-logits_to_keep, None) if isinstance(
    logits_to_keep, int) else logits_to_keep`` -- a tensor is used directly as an
    index into the sequence axis, which is what lets the packed path hand it a row
    map. If a transformers version ever changed that to an int-only meaning, this
    would start selecting the LAST n rows and the loss would train on the wrong
    positions while still running. ``_supports_logits_to_keep`` catches the
    argument going away; only this catches it changing meaning.
    """
    transformers = pytest.importorskip("transformers")
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM
    except Exception as e:  # pragma: no cover - older transformers
        pytest.skip(f"Qwen3 unavailable in transformers {transformers.__version__}: {e}")

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        )
    ).eval()
    input_ids = torch.randint(0, 64, (1, 12))
    sel = torch.tensor([3, 4, 7, 10])

    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits
        kept = model(input_ids=input_ids, use_cache=False, logits_to_keep=sel).logits

    assert kept.shape == (1, len(sel), 64)
    # Same arithmetic, different GEMM shape: exact equality is not promised, the
    # values are.
    assert torch.allclose(kept, full[:, sel], atol=1e-5), (
        "logits_to_keep did not select the given rows -- the row map is being "
        "read as something other than an index"
    )


def test_a_self_referential_wrapper_chain_does_not_hang():
    """A broken wrapper must not hang worker startup."""

    class _Loop(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids

    loop = _Loop()
    loop.module = loop  # points at itself

    assert _supports_logits_to_keep(loop) is False
