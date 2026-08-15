"""``response_only_logits``: restricting lm_head to the response rows.

The prompt rows' logits were computed at ``(rows, vocab)`` and dropped -- every
consumer of the actor/teacher forward (sampled-token log-prob, entropy, top-k KL)
is sliced to ``[:, -response_length-1:-1]`` before it leaves. This path hands the
row selection to the model as ``logits_to_keep`` so the projection never runs on
them.

lm_head is a per-position linear map, so selecting rows before or after it is the
same arithmetic. These tests pin that: a stand-in model whose body is arbitrary
but position-wise-consistent is run both ways, and the three outputs must match.

The stand-in exists because the claim is about *composition* (selection commutes
with the projection, and the downstream padding/slicing is unchanged), not about
Qwen3. Whether the installed transformers actually accepts a tensor
``logits_to_keep`` is a separate, environment-dependent question, checked at
construction by ``_supports_logits_to_keep`` and covered below.

CPU-only.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor.dp_actor import _supports_logits_to_keep, response_row_selection
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


VOCAB = 37


def _ragged_mask(batch_size, prompt_len, response_len, rng):
    """Left-padded prompt + right-padded response, the layout the rollout writes."""
    seqlen = prompt_len + response_len
    mask = torch.zeros((batch_size, seqlen), dtype=torch.long)
    for b in range(batch_size):
        n_prompt = int(torch.randint(1, prompt_len + 1, (1,), generator=rng))
        n_resp = int(torch.randint(1, response_len + 1, (1,), generator=rng))
        mask[b, prompt_len - n_prompt : prompt_len + n_resp] = 1
    return mask


def _pad_input(values, indices, batch, seqlen):
    """``pad_input``: scatter packed rows back into a zero-filled (batch*seqlen) grid."""
    out = torch.zeros((batch * seqlen,) + values.shape[1:], dtype=values.dtype)
    out[indices] = values
    return out.view((batch, seqlen) + values.shape[1:])


class _Model:
    """Position-wise body + linear head, with the ``logits_to_keep`` contract.

    ``logits_to_keep`` as a tensor indexes dim 1 of the hidden states *before* the
    head runs -- the same thing HF's causal-LM forwards do. The body is nonlinear
    across the vocab dimension but computed per position, which is what makes
    row selection and projection commute.
    """

    def __init__(self, hidden=8, vocab=VOCAB, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn((64, hidden), generator=g)
        self.head = torch.randn((hidden, vocab), generator=g)
        self.calls = []

    def __call__(self, input_ids, logits_to_keep=None, **kw):
        h = self.emb[input_ids % self.emb.shape[0]]  # (1, n, hidden)
        h = torch.tanh(h)
        if logits_to_keep is not None:
            h = h[:, logits_to_keep, :]
        self.calls.append(h.shape[1])
        return type("Out", (), {"logits": h @ self.head})()


def _reference(model, input_ids_rmpad, indices, batch_size, seqlen, response_length, labels, topk_k):
    """Full-logits path: project everything, then slice."""
    logits = model(input_ids_rmpad).logits.squeeze(0)  # (total_nnz, vocab)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    log_probs = _pad_input(lp.unsqueeze(-1), indices, batch_size, seqlen).squeeze(-1)
    log_probs = log_probs[:, -response_length - 1 : -1]

    sel, sel_indices, _ = response_row_selection(indices, seqlen, response_length)
    lse = torch.logsumexp(logits[sel], dim=-1, keepdim=True)
    tvals, tids = torch.topk(logits[sel], k=topk_k, dim=-1)
    t_lp = _pad_input((tvals - lse).float(), sel_indices, batch_size, seqlen)
    t_id = _pad_input(tids.float(), sel_indices, batch_size, seqlen)
    return log_probs, t_lp[:, -response_length - 1 : -1, :], t_id[:, -response_length - 1 : -1, :].round().long()


def _response_only(model, input_ids_rmpad, indices, batch_size, seqlen, response_length, labels, topk_k):
    """response_only_logits path: select rows, then project."""
    sel, sel_indices, _ = response_row_selection(indices, seqlen, response_length)
    logits_resp = model(input_ids_rmpad, logits_to_keep=sel).logits.squeeze(0)  # (n_resp, vocab)

    lp = torch.log_softmax(logits_resp.float(), dim=-1).gather(-1, labels[sel].unsqueeze(-1)).squeeze(-1)
    log_probs = _pad_input(lp.unsqueeze(-1), sel_indices, batch_size, seqlen).squeeze(-1)
    log_probs = log_probs[:, -response_length - 1 : -1]

    lse = torch.logsumexp(logits_resp, dim=-1, keepdim=True)
    tvals, tids = torch.topk(logits_resp, k=topk_k, dim=-1)
    t_lp = _pad_input((tvals - lse).float(), sel_indices, batch_size, seqlen)
    t_id = _pad_input(tids.float(), sel_indices, batch_size, seqlen)
    return log_probs, t_lp[:, -response_length - 1 : -1, :], t_id[:, -response_length - 1 : -1, :].round().long()


@pytest.mark.parametrize("trial", range(12))
def test_matches_the_full_logits_path(trial):
    """Same log-probs, same top-k values, same top-k ids -- over randomised ragged
    prompt/response splits, which is where an off-by-one in the row map would show."""
    rng = torch.Generator().manual_seed(trial)
    batch_size, prompt_len, response_len = 4, 11, 5
    seqlen = prompt_len + response_len

    mask = _ragged_mask(batch_size, prompt_len, response_len, rng)
    indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
    input_ids_rmpad = torch.randint(0, VOCAB, (1, indices.numel()), generator=rng)
    labels = torch.randint(0, VOCAB, (indices.numel(),), generator=rng)
    model = _Model(seed=trial)

    ref = _reference(model, input_ids_rmpad, indices, batch_size, seqlen, response_len, labels, topk_k=3)
    got = _response_only(model, input_ids_rmpad, indices, batch_size, seqlen, response_len, labels, topk_k=3)

    torch.testing.assert_close(got[0], ref[0])
    torch.testing.assert_close(got[1], ref[1])
    assert torch.equal(got[2], ref[2])


def test_the_head_actually_runs_on_fewer_rows():
    """The point of the change. On this mixture prompts dominate, so the selected
    row count must be well under the packed total -- otherwise the flag is a no-op
    wearing a cost."""
    rng = torch.Generator().manual_seed(0)
    batch_size, prompt_len, response_len = 4, 24, 6
    seqlen = prompt_len + response_len
    mask = _ragged_mask(batch_size, prompt_len, response_len, rng)
    indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
    input_ids_rmpad = torch.randint(0, VOCAB, (1, indices.numel()), generator=rng)
    model = _Model()

    model(input_ids_rmpad)
    sel, _, _ = response_row_selection(indices, seqlen, response_len)
    model(input_ids_rmpad, logits_to_keep=sel)

    full_rows, resp_rows = model.calls
    assert resp_rows < full_rows
    assert resp_rows == sel.numel()


def test_gradient_reaches_only_the_selected_rows():
    """The backward saving is the other half of the effect: with the projection
    behind the selection, prompt positions carry no head gradient at all."""
    rng = torch.Generator().manual_seed(3)
    batch_size, prompt_len, response_len = 3, 9, 4
    seqlen = prompt_len + response_len
    mask = _ragged_mask(batch_size, prompt_len, response_len, rng)
    indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]

    hidden = torch.randn((1, indices.numel(), 6), requires_grad=True)
    head = torch.randn((6, VOCAB))
    sel, _, _ = response_row_selection(indices, seqlen, response_len)

    (hidden[:, sel, :] @ head).sum().backward()

    touched = (hidden.grad.squeeze(0).abs().sum(-1) > 0).nonzero(as_tuple=True)[0]
    assert torch.equal(touched, sel)


def test_capability_probe_unwraps_and_reports():
    class _Plain:
        def forward(self, input_ids, logits_to_keep=None):
            return None

    class _NoSupport:
        def forward(self, input_ids):
            return None

    class _Wrapper:
        def __init__(self, inner):
            self._fsdp_wrapped_module = inner

        def forward(self, *a, **kw):  # the wrapper's own signature must not decide
            return None

    assert _supports_logits_to_keep(_Plain())
    assert not _supports_logits_to_keep(_NoSupport())
    assert _supports_logits_to_keep(_Wrapper(_Plain()))
    assert not _supports_logits_to_keep(_Wrapper(_NoSupport()))


def test_capability_probe_terminates_on_a_self_referential_wrapper():
    """A broken wrapper chain must not hang worker startup."""

    class _Loop:
        def forward(self, input_ids):
            return None

    a = _Loop()
    b = _Loop()
    a.module = b
    b.module = a
    assert _supports_logits_to_keep(a) is False
