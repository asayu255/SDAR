"""The student's own top-k, taken out of the training forward's logits.

Under ``student_indexed_topk`` the actor's forward has to hand back two things it
did not before: the ids the student ranks first, and the student's log-probs at
them -- from ONE logits tensor, because a second student forward would cost the
thing this arm is trying to afford. The teacher is then resolved at those ids
from cached hidden states (verl/workers/teacher_cache.py, tested there).

What is pinned here is the forward half:

* the values are a real full-vocabulary log-softmax at the model's own top-k, not
  a softmax over the k selected logits;
* passing the normaliser in rather than recomputing it changes nothing -- the
  forward now computes one logsumexp over (n_resp, vocab) and shares it, where it
  used to run a second full reduction over the widest tensor in the step;
* ``sorted=False`` changes nothing either, because the KL sums over the support
  and never reads the order. It is asserted as a SET so a future change to the
  ordering cannot quietly break the values.

``pad_input`` comes from flash_attn, which is CUDA-only, so it is substituted
with the scatter it performs. The rest of the function under test is the real
one.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor import dp_actor
    from verl.workers.actor.dp_actor import DataParallelPPOActor, response_row_selection
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


VOCAB, K = 41, 6


def _pad_input(hidden_states, indices, batch, seqlen):
    """``pad_input``: scatter packed rows back into a zero-filled (batch*seqlen) grid."""
    out = torch.zeros((batch * seqlen,) + hidden_states.shape[1:], dtype=hidden_states.dtype)
    out[indices] = hidden_states
    return out.view((batch, seqlen) + hidden_states.shape[1:])


@pytest.fixture(autouse=True)
def _flash_attn_stub(monkeypatch):
    monkeypatch.setattr(dp_actor, "pad_input", _pad_input, raising=False)


def _layout(batch_size=3, prompt_len=5, response_len=4, seed=0):
    """A ragged left-padded prompt / right-padded response batch, packed the way
    ``unpad_input`` packs it, plus the row map the response-only path derives."""
    rng = torch.Generator().manual_seed(seed)
    seqlen = prompt_len + response_len
    mask = torch.zeros((batch_size, seqlen), dtype=torch.long)
    for b in range(batch_size):
        n_prompt = int(torch.randint(1, prompt_len + 1, (1,), generator=rng))
        n_resp = int(torch.randint(1, response_len + 1, (1,), generator=rng))
        mask[b, prompt_len - n_prompt : prompt_len + n_resp] = 1
    indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
    sel, sel_indices, sel_slot = response_row_selection(indices, seqlen, response_len)
    logits = torch.randn((sel.numel(), VOCAB), generator=rng) * 2.0
    return logits, sel_indices, sel_slot, batch_size, seqlen, response_len


def _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len, lse=None):
    return actor._topk_from_response_logits(
        logits_resp=logits,
        sel_indices=sel_indices,
        sel_slot=sel_slot,
        batch_size=bs,
        seqlen=seqlen,
        response_length=resp_len,
        topk_k=K,
        topk_ids=None,
        lse=lse,
    )


def test_the_values_are_a_full_vocabulary_log_softmax_at_the_models_own_ids():
    """The claim the KL rests on: both sides of it are full-vocabulary log-probs
    at the same ids, whichever model chose them. A softmax over the k selected
    logits would be a different quantity, and the tail bucket (1 - sum) would then
    be identically zero rather than the mass the support misses."""
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    logits, sel_indices, sel_slot, bs, seqlen, resp_len = _layout(seed=1)

    lp, ids = _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len)

    reference = torch.log_softmax(logits, dim=-1)
    # Read the scattered result back at the rows it came from.
    for row, flat in enumerate(sel_indices.tolist()):
        b, slot = flat // seqlen, sel_slot[row].item()
        got_ids = ids[b, slot]
        torch.testing.assert_close(
            lp[b, slot], reference[row][got_ids], rtol=0, atol=1e-6
        )
        # ...and those ids really are this row's k largest logits.
        assert set(got_ids.tolist()) == set(torch.topk(logits[row], K).indices.tolist())


def test_sharing_the_normaliser_changes_no_value():
    """The forward needs the logsumexp for the top-k and again for the cached
    entry. It is one reduction over the widest tensor in the step, so it is now
    computed once and passed in -- which must be arithmetically the same thing,
    not merely close."""
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    logits, sel_indices, sel_slot, bs, seqlen, resp_len = _layout(seed=2)

    lp_self, ids_self = _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len)
    shared = torch.logsumexp(logits, dim=-1, keepdim=True)
    lp_shared, ids_shared = _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len, lse=shared)

    assert torch.equal(lp_self, lp_shared)
    assert torch.equal(ids_self, ids_shared)


def test_padding_positions_stay_zero():
    """A response slot no packed row landed in is left at zero by the scatter, as
    it was before. The loss masks them, but a non-zero there would mean the row
    map and the window disagree."""
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    logits, sel_indices, sel_slot, bs, seqlen, resp_len = _layout(seed=3)

    lp, ids = _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len)

    written = {(flat // seqlen, sel_slot[r].item()) for r, flat in enumerate(sel_indices.tolist())}
    for b in range(bs):
        for slot in range(resp_len):
            if (b, slot) in written:
                continue
            assert torch.all(lp[b, slot] == 0)
            assert torch.all(ids[b, slot] == 0)


def test_the_support_is_a_set_not_an_order():
    """``sorted=False`` is only safe because nothing reads the order: the KL sums
    over the support. Pinned as a set so a change to either side stays visible."""
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    logits, sel_indices, sel_slot, bs, seqlen, resp_len = _layout(seed=4)

    _, ids = _call(actor, logits, sel_indices, sel_slot, bs, seqlen, resp_len)

    for row, flat in enumerate(sel_indices.tolist()):
        b, slot = flat // seqlen, sel_slot[row].item()
        got = ids[b, slot].tolist()
        assert len(set(got)) == K, "the k ids must be distinct"
        assert set(got) == set(torch.topk(logits[row], K, sorted=True).indices.tolist())
