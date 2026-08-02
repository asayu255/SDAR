"""Unit tests for ``response_row_selection``.

The top-k distillation forward used to run logsumexp / topk / gather over every
unpadded row -- prompt included -- and then keep only the response slice.
``response_row_selection`` picks those rows up front instead. The saving is real
only if the selection is exactly the set of rows the old path's slice kept, and
the per-row ``sel_slot`` addresses the same ``topk_ids`` entry the old path's
scatter-then-gather produced, so both are checked against a literal
reimplementation of that path.

CPU-only: this exercises index arithmetic, not a model, so ``unpad_input`` and
``pad_input`` are reimplemented here from their definitions (nonzero of the
flattened mask, and scatter into a zero-filled grid) rather than importing
flash_attn.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor.dp_actor import response_row_selection
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _ragged_mask(batch_size, prompt_len, response_len, rng):
    """Left-padded prompt + right-padded response, the layout the rollout writes."""
    seqlen = prompt_len + response_len
    mask = torch.zeros((batch_size, seqlen), dtype=torch.long)
    for b in range(batch_size):
        n_prompt = int(torch.randint(1, prompt_len + 1, (1,), generator=rng))
        n_resp = int(torch.randint(1, response_len + 1, (1,), generator=rng))
        mask[b, prompt_len - n_prompt : prompt_len + n_resp] = 1
    return mask


def _unpad_indices(mask):
    """What ``unpad_input`` returns: flattened positions of the attended tokens."""
    return torch.nonzero(mask.flatten(), as_tuple=True)[0]


def _old_path_slice(indices, values, topk_ids, batch_size, seqlen, response_length):
    """The pre-change mapping, written out literally.

    Scatter ``topk_ids`` across the full sequence, gather it back at every
    unpadded row, scatter the per-row results into a (bs, seqlen, k) grid, and
    slice. Returns (per-response-slot value, per-response-slot id).
    """
    k = topk_ids.size(-1)
    lo = seqlen - response_length - 1
    full_ids = torch.zeros((batch_size, seqlen, k), dtype=torch.long)
    full_ids[:, lo : seqlen - 1, :] = topk_ids
    ids_rmpad = full_ids.reshape(batch_size * seqlen, k)[indices]  # (total_nnz, k)

    out_val = torch.zeros((batch_size * seqlen,), dtype=values.dtype)
    out_ids = torch.zeros((batch_size * seqlen, k), dtype=torch.long)
    out_val[indices] = values
    out_ids[indices] = ids_rmpad
    return (
        out_val.view(batch_size, seqlen)[:, lo : seqlen - 1],
        out_ids.view(batch_size, seqlen, k)[:, lo : seqlen - 1, :],
    )


def _new_path_slice(indices, values, topk_ids, batch_size, seqlen, response_length):
    """The same result via response_row_selection, as the actor now computes it."""
    k = topk_ids.size(-1)
    lo = seqlen - response_length - 1
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_length)
    ids_resp = topk_ids[sel_indices // seqlen, sel_slot]  # (n_resp, k)

    out_val = torch.zeros((batch_size * seqlen,), dtype=values.dtype)
    out_ids = torch.zeros((batch_size * seqlen, k), dtype=torch.long)
    out_val[sel_indices] = values[sel]
    out_ids[sel_indices] = ids_resp
    return (
        out_val.view(batch_size, seqlen)[:, lo : seqlen - 1],
        out_ids.view(batch_size, seqlen, k)[:, lo : seqlen - 1, :],
    )


@pytest.mark.parametrize("trial", range(60))
def test_matches_the_scatter_then_gather_path(trial):
    """Same slice out, over randomised ragged prompt/response splits."""
    rng = torch.Generator().manual_seed(trial)
    batch_size = int(torch.randint(1, 5, (1,), generator=rng))
    prompt_len = int(torch.randint(1, 12, (1,), generator=rng))
    response_len = int(torch.randint(1, 8, (1,), generator=rng))
    seqlen = prompt_len + response_len
    k = 3

    mask = _ragged_mask(batch_size, prompt_len, response_len, rng)
    indices = _unpad_indices(mask)
    # A distinct value per unpadded row, so a misrouted row cannot coincide.
    values = torch.arange(1, len(indices) + 1, dtype=torch.float32)
    topk_ids = torch.arange(batch_size * response_len * k, dtype=torch.long).view(
        batch_size, response_len, k
    )

    old_val, old_ids = _old_path_slice(indices, values, topk_ids, batch_size, seqlen, response_len)
    new_val, new_ids = _new_path_slice(indices, values, topk_ids, batch_size, seqlen, response_len)
    assert torch.equal(old_val, new_val)
    assert torch.equal(old_ids, new_ids)


def test_selection_is_the_predict_shifted_window():
    """A logit at position t predicts token t+1, so the window is offset by one."""
    batch_size, seqlen, response_len = 1, 10, 4
    indices = torch.arange(seqlen, dtype=torch.long)  # nothing masked
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_len)

    # positions 5..8 predict the tokens at 6..9, which are the 4 response tokens
    assert sel.tolist() == [5, 6, 7, 8]
    assert sel_indices.tolist() == [5, 6, 7, 8]
    assert sel_slot.tolist() == [0, 1, 2, 3]


def test_response_padding_rows_are_absent_not_selected():
    """Rows the mask dropped cannot be selected; their slots stay zero-filled."""
    batch_size, prompt_len, response_len = 2, 4, 4
    seqlen = prompt_len + response_len
    mask = torch.zeros((batch_size, seqlen), dtype=torch.long)
    mask[0, 2:6] = 1  # 2 prompt tokens, then 2 response tokens (early EOS)
    mask[1, 0:8] = 1  # full row
    indices = _unpad_indices(mask)

    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_len)
    # row 0 contributes only positions 3,4,5 (>= lo=3, < 7); row 1 contributes 3..6
    assert (sel_indices // seqlen).tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert sel_slot.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert sel_slot.max() < response_len


def test_empty_selection_is_handled():
    """A micro-batch with no attended response position yields empty tensors."""
    seqlen, response_len = 8, 4
    indices = torch.arange(3, dtype=torch.long)  # only positions 0,1,2 < lo=3
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_len)
    assert sel.numel() == 0 and sel_indices.numel() == 0 and sel_slot.numel() == 0
