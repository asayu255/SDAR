"""Exactness test for the dynamic-bsz token-mean scaling (speedup #3).

When ``use_dynamic_bsz=True`` the actor packs micro-batches by token budget, so
micro-batch composition (and each micro-batch's valid-token count) varies. Under
``loss_agg_mode="token-mean"`` the stock scaling ``len(data)/ppo_mini_batch_size``
reweights by *sample* count, which does NOT reproduce the global token-mean when
micro-batches differ in length. The added ``dynamic_bsz_token_scale`` path scales
each micro-batch's token-mean by ``micro_valid_tokens / minibatch_valid_tokens``
(a token-sum divided by the mini-batch's total tokens), which is exactly the
global token-mean and is invariant to how tokens are grouped into micro-batches.

This test replicates that arithmetic (no model needed) and asserts:
* token-scale reproduces the global token-mean for ANY grouping (grouping-invariant);
* the stock sample-count scaling generally does NOT (documents the difference).
"""

import pytest

torch = pytest.importorskip("torch")


def _masked_mean(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _accumulate(groups, mini_valid_tokens, mini_n, mode):
    """Sum per-micro-batch contributions the way dp_actor.update_policy does."""
    total = torch.zeros(())
    for loss_m, mask_m in groups:
        policy_loss = _masked_mean(loss_m, mask_m)  # token-mean within the micro-batch
        if mode == "token":
            total = total + policy_loss * (mask_m.sum() / mini_valid_tokens)
        elif mode == "sample":
            total = total + policy_loss * (loss_m.shape[0] / mini_n)
        else:
            raise ValueError(mode)
    return total


def _global_token_mean(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _split(loss, mask, sizes):
    groups, i = [], 0
    for s in sizes:
        groups.append((loss[i : i + s], mask[i : i + s]))
        i += s
    return groups


def test_token_scale_equals_global_token_mean_and_is_grouping_invariant():
    torch.manual_seed(0)
    N, T = 8, 6
    loss = torch.randn(N, T)
    # Variable-length sequences (variable valid-token counts per sample).
    lengths = torch.tensor([6, 1, 4, 2, 5, 3, 6, 1])
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()

    mini_valid = mask.sum().clamp(min=1)
    target = _global_token_mean(loss, mask)

    # Two very different micro-batch groupings of the SAME mini-batch.
    grouping_a = _split(loss, mask, [2, 2, 2, 2])         # equal-size (fixed-bsz-like)
    order = torch.argsort(lengths)                          # length-sorted (dynamic-bsz-like)
    grouping_b = _split(loss[order], mask[order], [3, 3, 2])

    for groups in (grouping_a, grouping_b):
        got = _accumulate(groups, mini_valid, N, mode="token")
        assert torch.allclose(got, target, atol=1e-6), "token-scale must equal global token-mean for any grouping"


def test_sample_count_scaling_differs_from_global_token_mean():
    torch.manual_seed(1)
    N, T = 6, 5
    loss = torch.randn(N, T)
    lengths = torch.tensor([5, 1, 4, 1, 5, 2])
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()

    target = _global_token_mean(loss, mask)
    # length-sorted grouping (as rearrange_micro_batches would produce)
    order = torch.argsort(lengths)
    groups = _split(loss[order], mask[order], [3, 3])

    sample_scaled = _accumulate(groups, mask.sum(), N, mode="sample")
    token_scaled = _accumulate(groups, mask.sum().clamp(min=1), N, mode="token")

    assert torch.allclose(token_scaled, target, atol=1e-6)
    # With unequal lengths across micro-batches, sample-count scaling drifts from
    # the true token-mean (this is the objective shift the token path removes).
    assert not torch.allclose(sample_scaled, target, atol=1e-4)
