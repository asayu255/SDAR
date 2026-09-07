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

"""Scattering straight into the response window instead of the whole sequence.

The change is an optimisation, so the tests are equality tests: bit-identical
values, bit-identical gradients, and the same zero-fill where the packed batch
had nothing -- on the awkward cases as well as the tidy one.
"""

import ast
import inspect

import pytest
import torch
from flash_attn.bert_padding import pad_input, unpad_input

from verl.workers.actor.dp_actor import response_row_selection, response_scatter_indices


def _packed(batch, seqlen, response_length, lengths, seed=0):
    """A packed batch with per-row sequence lengths, and its response selection.

    ``lengths`` are total (prompt + response) token counts, left-padded exactly
    as the actor's attention mask is, so a short row's response window is only
    partly present -- the case that separates a correct index map from one that
    happens to work on full rows.
    """
    torch.manual_seed(seed)
    mask = torch.zeros(batch, seqlen, dtype=torch.int32)
    for i, n in enumerate(lengths):
        mask[i, seqlen - n:] = 1
    _, indices, *_ = unpad_input(torch.zeros(batch, seqlen, 1), mask)
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
    return mask, indices, sel, sel_indices, sel_slot


CASES = [
    # tidy: every row full
    (4, 64, 16, [64, 64, 64, 64]),
    # short rows: the response window is only partly present
    (4, 64, 16, [64, 40, 20, 10]),
    # one row shorter than the window itself
    (3, 32, 16, [32, 8, 5]),
    # a row with no response positions at all
    (3, 32, 8, [32, 32, 2]),
]


@pytest.mark.parametrize("batch,seqlen,rl,lengths", CASES)
@pytest.mark.parametrize("width", [1, 5])
def test_the_window_scatter_is_bit_identical_to_scatter_then_slice(batch, seqlen, rl, lengths, width):
    _, _, sel, sel_indices, sel_slot = _packed(batch, seqlen, rl, lengths)
    x = torch.randn(len(sel), width, dtype=torch.float32)

    old = pad_input(x, indices=sel_indices, batch=batch, seqlen=seqlen)[:, -rl - 1 : -1, :]
    new = pad_input(
        x,
        indices=response_scatter_indices(sel_indices, sel_slot, seqlen, rl),
        batch=batch,
        seqlen=rl,
    )
    assert new.shape == old.shape
    assert torch.equal(new, old), "an optimisation may not change a single value"


@pytest.mark.parametrize("batch,seqlen,rl,lengths", CASES)
def test_the_gradient_is_bit_identical_too(batch, seqlen, rl, lengths):
    _, _, sel, sel_indices, sel_slot = _packed(batch, seqlen, rl, lengths, seed=1)
    base = torch.randn(len(sel), 3, dtype=torch.float64)
    upstream = torch.randn(batch, rl, 3, dtype=torch.float64)

    a = base.clone().requires_grad_(True)
    (pad_input(a, indices=sel_indices, batch=batch, seqlen=seqlen)[:, -rl - 1 : -1, :]
     * upstream).sum().backward()

    b = base.clone().requires_grad_(True)
    (pad_input(b, indices=response_scatter_indices(sel_indices, sel_slot, seqlen, rl),
               batch=batch, seqlen=rl) * upstream).sum().backward()

    assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("batch,seqlen,rl,lengths", CASES)
def test_the_positions_the_packed_batch_lacked_are_still_zero(batch, seqlen, rl, lengths):
    _, _, sel, sel_indices, sel_slot = _packed(batch, seqlen, rl, lengths, seed=2)
    x = torch.full((len(sel), 2), 7.0)
    new = pad_input(x, indices=response_scatter_indices(sel_indices, sel_slot, seqlen, rl),
                    batch=batch, seqlen=rl)
    present = torch.zeros(batch, rl, dtype=torch.bool)
    present[sel_indices // seqlen, sel_slot] = True
    assert torch.equal(new[present], torch.full((int(present.sum()), 2), 7.0))
    assert torch.equal(new[~present], torch.zeros(int((~present).sum()), 2))


def test_the_index_map_is_a_permutation_of_the_window_positions():
    """Two selected rows must never land on one window slot: that would silently
    overwrite one of them, and pad_input's assignment would not complain."""
    for batch, seqlen, rl, lengths in CASES:
        _, _, sel, sel_indices, sel_slot = _packed(batch, seqlen, rl, lengths, seed=3)
        idx = response_scatter_indices(sel_indices, sel_slot, seqlen, rl)
        assert idx.numel() == torch.unique(idx).numel()
        assert int(idx.min()) >= 0
        assert int(idx.max()) < batch * rl


def test_the_storage_it_holds_is_the_window_not_the_sequence():
    """The point of the change. The old result was a VIEW of a full-sequence
    tensor, so keeping it kept the sequence; the new one owns just the window."""
    batch, seqlen, rl, hidden = 4, 4608, 512, 2048
    _, _, sel, sel_indices, sel_slot = _packed(batch, seqlen, rl, [seqlen] * batch, seed=4)
    x = torch.randn(len(sel), hidden, dtype=torch.bfloat16)

    old = pad_input(x, indices=sel_indices, batch=batch, seqlen=seqlen)[:, -rl - 1 : -1, :]
    new = pad_input(x, indices=response_scatter_indices(sel_indices, sel_slot, seqlen, rl),
                    batch=batch, seqlen=rl)

    old_bytes = old.untyped_storage().nbytes()
    new_bytes = new.untyped_storage().nbytes()
    assert old_bytes == batch * seqlen * hidden * 2
    assert new_bytes == batch * rl * hidden * 2
    assert old_bytes / new_bytes == pytest.approx(seqlen / rl, rel=1e-6)
    # the numbers quoted in the review, for this exact shape
    assert old_bytes / 2**20 == pytest.approx(72.0)
    assert new_bytes / 2**20 == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# the call sites: none of them may go back to scattering the whole sequence


def test_the_response_only_path_never_scatters_the_full_sequence():
    """Checked on the syntax tree, not the text: the docstrings still describe
    the window the old code sliced, and should.

    The rule is about WHICH map a call uses. A pad_input fed from
    ``sel_indices`` -- the response-row selection -- has nothing outside the
    window to place, so building a full-sequence grid is pure waste. The legacy
    full-logits path (``response_only_logits=False``, and the ulysses branch
    with it) is fed from ``indices``, the whole packed batch, and there the full
    grid is what it computed; that one is deliberately left alone.
    """
    import verl.workers.actor.dp_actor as m

    tree = ast.parse(inspect.getsource(m))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "pad_input"]
    assert calls, "the forward path stopped calling pad_input at all"

    window_maps = {"resp_idx", "ri_"}
    converted = 0
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        idx, seqlen = kw.get("indices", ""), kw.get("seqlen", "")
        if idx == "indices":
            assert seqlen == "seqlen", "the legacy path is expected to keep the full grid"
            continue
        assert "response_scatter_indices" in idx or idx in window_maps, (
            f"pad_input(indices={idx!r}) is neither the legacy map nor a window map"
        )
        assert "response_length" in seqlen or seqlen == "rl_", (
            f"pad_input(seqlen={seqlen!r}) builds a full-sequence grid from a window map"
        )
        converted += 1
    assert converted >= 5, f"expected the five converted sites, saw {converted}"

    # and none of the converted ones slices a sequence axis back off
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        inner = node.value
        if not (isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "pad_input"):
            continue
        kw = {k.arg: ast.unparse(k.value) for k in inner.keywords}
        if kw.get("indices") == "indices":
            continue
        sl = ast.unparse(node.slice)
        assert "-1" not in sl, f"pad_input(...)[{sl}] is a scatter-then-slice again"
