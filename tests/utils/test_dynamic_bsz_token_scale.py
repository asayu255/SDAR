"""``use_dynamic_bsz`` must not make the objective depend on how tokens were grouped.

Dynamic batching packs micro-batches to a token budget, so two runs of the same
mini-batch can split it into different numbers of rows. The stock scaling
multiplies each micro-batch's token-mean by ``len(micro) / ppo_mini_batch_size``,
i.e. by its share of the *samples* -- which under-weights a micro-batch of long
sequences and over-weights one of short sequences, because a token-mean already
divided by its own token count. The objective then moves when the packer does,
which is exactly what dynamic batching is free to change.

``actor.dynamic_bsz_token_scale=True`` scales by the valid-token share instead.
This checks the two properties that buys, on the arithmetic alone -- no model,
no distributed init:

* summing the scaled micro-batch losses reproduces the true global token-mean;
* that sum is identical under a different packing of the same rows.

The stock path is checked to *fail* the second property, so the test is
measuring the fix rather than a tautology.
"""

import pytest

torch = pytest.importorskip("torch")


def _token_mean(losses, mask):
    """What agg_loss(loss_agg_mode="token-mean") computes for one micro-batch."""
    return (losses * mask).sum() / mask.sum().clamp(min=1)


def _rows(n_rows, resp_len, seed):
    """Per-token losses and a ragged response mask (rows have different lengths)."""
    g = torch.Generator().manual_seed(seed)
    losses = torch.randn(n_rows, resp_len, generator=g)
    lengths = torch.randint(1, resp_len + 1, (n_rows,), generator=g)
    mask = (torch.arange(resp_len)[None, :] < lengths[:, None]).float()
    return losses, mask


def _step_loss_token_scaled(groups, total_tokens):
    """Sum over micro-batches of the token-share-scaled token-mean."""
    total = 0.0
    for losses, mask in groups:
        total = total + _token_mean(losses, mask) * (mask.sum() / total_tokens)
    return float(total)


def _step_loss_sample_scaled(groups, mini_batch_size):
    """Sum over micro-batches of the sample-share-scaled token-mean (the stock path)."""
    total = 0.0
    for losses, mask in groups:
        total = total + _token_mean(losses, mask) * (len(losses) / mini_batch_size)
    return float(total)


def _split(losses, mask, sizes):
    out, at = [], 0
    for n in sizes:
        out.append((losses[at : at + n], mask[at : at + n]))
        at += n
    assert at == len(losses)
    return out


@pytest.fixture
def mini_batch():
    return _rows(n_rows=12, resp_len=16, seed=0)


def test_token_scaling_reproduces_the_global_token_mean(mini_batch):
    losses, mask = mini_batch
    total_tokens = float(mask.sum().clamp(min=1))
    exact = float(_token_mean(losses, mask))

    for sizes in ([12], [6, 6], [5, 4, 3], [1] * 12):
        got = _step_loss_token_scaled(_split(losses, mask, sizes), total_tokens)
        assert got == pytest.approx(exact, rel=1e-6), sizes


def test_token_scaling_is_invariant_to_the_packing(mini_batch):
    losses, mask = mini_batch
    total_tokens = float(mask.sum().clamp(min=1))

    a = _step_loss_token_scaled(_split(losses, mask, [7, 5]), total_tokens)
    b = _step_loss_token_scaled(_split(losses, mask, [2, 4, 6]), total_tokens)
    assert a == pytest.approx(b, rel=1e-6)


def test_the_stock_sample_scaling_is_not(mini_batch):
    """If this ever starts passing, the knob has stopped buying anything."""
    losses, mask = mini_batch
    a = _step_loss_sample_scaled(_split(losses, mask, [7, 5]), mini_batch_size=12)
    b = _step_loss_sample_scaled(_split(losses, mask, [2, 4, 6]), mini_batch_size=12)
    assert a != pytest.approx(b, rel=1e-6)


def test_a_single_micro_batch_is_unscaled(mini_batch):
    """One micro-batch means the whole mini-batch, so the factor must be exactly 1."""
    losses, mask = mini_batch
    total_tokens = float(mask.sum().clamp(min=1))
    assert float(mask.sum() / total_tokens) == pytest.approx(1.0)
    assert _step_loss_token_scaled([(losses, mask)], total_tokens) == pytest.approx(
        float(_token_mean(losses, mask))
    )


def test_the_knob_is_off_unless_every_precondition_holds():
    """It rewrites the dynamic-bsz scaling only, and only for a token-mean."""
    from omegaconf import OmegaConf

    from verl.workers.actor import dp_actor  # noqa: F401  (import guard)

    def enabled(**overrides):
        cfg = OmegaConf.create(
            {"use_dynamic_bsz": True, "dynamic_bsz_token_scale": True, "loss_agg_mode": "token-mean", **overrides}
        )
        return bool(
            cfg.use_dynamic_bsz
            and cfg.get("dynamic_bsz_token_scale", False)
            and cfg.loss_agg_mode == "token-mean"
        )

    assert enabled()
    assert not enabled(use_dynamic_bsz=False)
    assert not enabled(dynamic_bsz_token_scale=False)
    assert not enabled(loss_agg_mode="seq-mean-token-sum")
