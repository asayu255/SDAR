"""The direct-KL (``position``) arm: what it does to the loss, and what says so.

``position`` mode is the weighting applied to the DIVERGENCE rather than to a
probability -- one positive scalar per token multiplying the whole per-token KL.
Until now the arm reported a single number for the entire mechanism
(``w_mean_pre_norm``, the weight before per-task normalisation), which cannot
separate the two things a reader has to tell apart:

* the arm redistributed effort between positions -- the mechanism, and
* the arm distilled harder everywhere -- a change to ``teacher_kl_loss_coef``
  wearing the mechanism's name.

:func:`position_ratio_metrics` separates them, and the tests below are mostly
about the exact identities that let each number be read as a quantity:
``sum_groups w_from_* == w_pre - 1`` with no residual, ``kl_scale == 1`` exactly
when the arm only redistributed, ``w_kl_lift == 1`` exactly when the weighting is
uncorrelated with where the student's error is.

Plus the two claims the module docstring makes about gradients, pinned here
rather than left as prose: weighting a position by ``W`` scales every logit's
gradient by ``W`` and moves nothing else, and the term-wise product the module
declines to implement moves an agreed-up token DOWN.
"""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.sign_weights import (
        POSITION_BANDS,
        POSITION_STATE_GROUPS,
        POSITION_TERMS,
        STATE_AGREE_POS,
        STATE_CONFLICT_ON_POS,
        STATE_NEUTRAL_ON,
        ScopeTermStats,
        TokenStateCounts,
        position_decomposition_terms,
        position_ratio_metrics,
        position_weights,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ["alfworld", "search", "webshop"]
RANGE = (1.0, 1.25)


def _logprob(bs, resp, k, scale=1.0, seed=None):
    """A full-vocab log-softmax gathered at k ids, so the top-k mass is < 1."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.log_softmax(scale * torch.randn(bs, resp, k + 6), dim=-1)[..., :k]


def _fold(terms, mask, task_ids=None, n_tasks=len(TASKS)):
    """One micro-batch through ScopeTermStats, rendered as ratio metrics."""
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=n_tasks, device="cpu")
    stats.update(terms, response_mask=mask, task_ids=task_ids)
    return position_ratio_metrics(stats.sums(task_names=TASKS)), stats


# --------------------------------------------------------------------------- #
# The exact identities
# --------------------------------------------------------------------------- #
def test_the_state_groups_partition_the_whole_departure_from_neutrality():
    """``sum_groups w_from_* == w_pre - 1``, with no residual anywhere.

    The tail carries weight 1 by construction, so it contributes nothing to the
    excess -- which is what makes the four groups a partition rather than four
    of five terms. A residual here would mean a state exists that no column
    accounts for, and the shares below would silently not add to 1.
    """
    bs, resp, k = 3, 5, 7
    on = _logprob(bs, resp, k, seed=0)
    w = torch.empty(bs, resp, k).uniform_(0.75, 1.25)
    state = torch.randint(0, 7, (bs, resp, k))
    pre = position_weights(w, on)
    terms = position_decomposition_terms(
        position_weight=pre,
        applied_weight=pre,
        candidate_weight=w,
        state=state,
        on_task_logprob=on,
        teacher_kl=torch.rand(bs, resp),
        weight_range=(0.75, 1.25),
    )
    total = sum(terms[f"w_from_{name}"] for name, _ in POSITION_STATE_GROUPS)
    assert torch.allclose(total, pre - 1.0, atol=1e-6)


def test_every_declared_term_is_produced_and_nothing_else_is():
    """``ScopeTermStats`` indexes by name, so a mismatch is a KeyError at step 1."""
    bs, resp, k = 2, 3, 4
    on = _logprob(bs, resp, k, seed=1)
    terms = position_decomposition_terms(
        position_weight=torch.ones(bs, resp),
        applied_weight=torch.ones(bs, resp),
        candidate_weight=torch.ones(bs, resp, k),
        state=torch.zeros(bs, resp, k, dtype=torch.long),
        on_task_logprob=on,
        teacher_kl=torch.rand(bs, resp),
        weight_range=RANGE,
    )
    assert set(terms) == set(POSITION_TERMS)
    for name, t in terms.items():
        assert t.shape == (bs, resp), name


def test_a_neutral_table_moves_nothing_at_all():
    """Every weight 1 => the arm is the plain one, and every reading says so."""
    bs, resp, k = 2, 4, 5
    on = _logprob(bs, resp, k, seed=2)
    kl = torch.rand(bs, resp) + 0.1
    ones = torch.ones(bs, resp)
    terms = position_decomposition_terms(
        position_weight=ones,
        applied_weight=ones,
        candidate_weight=torch.ones(bs, resp, k),
        state=torch.full((bs, resp, k), STATE_NEUTRAL_ON),
        on_task_logprob=on,
        teacher_kl=kl,
        weight_range=(1.0, 1.0),
    )
    m, _ = _fold(terms, torch.ones(bs, resp))
    assert m["sign_weight/pos/kl_scale"] == pytest.approx(1.0)
    assert m["sign_weight/pos/kl_shift_net"] == pytest.approx(0.0, abs=1e-9)
    assert m["sign_weight/pos/kl_shift_gross"] == pytest.approx(0.0, abs=1e-9)
    assert m["sign_weight/pos/w_norm_drift"] == pytest.approx(0.0, abs=1e-9)
    assert m["sign_weight/pos/w_kl_lift"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# kl_scale: the mechanism against a coefficient change
# --------------------------------------------------------------------------- #
def _hand_terms(w, kl, *, pre=None, weight_range=RANGE):
    """Terms for a hand-written (w, kl) pair, with no candidate structure."""
    bs, resp = w.shape
    pre = w if pre is None else pre
    on = torch.log(torch.full((bs, resp, 2), 0.25))
    return position_decomposition_terms(
        position_weight=pre,
        applied_weight=w,
        candidate_weight=torch.ones(bs, resp, 2),
        state=torch.full((bs, resp, 2), STATE_NEUTRAL_ON),
        on_task_logprob=on,
        teacher_kl=kl,
        weight_range=weight_range,
    )


def test_kl_scale_is_the_factor_applied_to_the_total_not_the_mean_weight():
    """The two coincide only when weight and KL are uncorrelated.

    That difference is the whole reason ``kl_scale`` exists as its own number:
    an arm whose weights average exactly 1 can still have multiplied the total
    teacher KL by 1.4 by putting its heavy weights where the KL is.
    """
    w = torch.tensor([[0.5, 1.5]])
    kl = torch.tensor([[1.0, 3.0]])
    m, _ = _fold(_hand_terms(w, kl), torch.ones(1, 2), n_tasks=0)
    assert m["sign_weight/pos/w_mean"] == pytest.approx(1.0)
    # sum w*kl = 0.5 + 4.5 = 5.0 against sum kl = 4.0
    assert m["sign_weight/pos/kl_scale"] == pytest.approx(1.25)
    assert m["sign_weight/pos/w_kl_lift"] == pytest.approx(1.25)


def test_net_over_gross_separates_a_redistribution_from_a_rescale():
    """1 = the arm scaled everything the same way; 0 = it only moved effort."""
    kl = torch.tensor([[1.0, 1.0]])
    up, _ = _fold(_hand_terms(torch.tensor([[1.2, 1.2]]), kl), torch.ones(1, 2), n_tasks=0)
    assert up["sign_weight/pos/kl_shift_net_over_gross"] == pytest.approx(1.0)
    mixed, _ = _fold(_hand_terms(torch.tensor([[1.2, 0.8]]), kl), torch.ones(1, 2), n_tasks=0)
    # 1e-6 rather than exact: the terms are float32, so "0.8" is 0.8 to about
    # seven digits and a ratio of two small differences inherits that.
    assert mixed["sign_weight/pos/kl_shift_net_over_gross"] == pytest.approx(0.0, abs=1e-6)
    assert mixed["sign_weight/pos/kl_shift_gross"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Where the weighting spent its budget
# --------------------------------------------------------------------------- #
def test_the_lift_says_whether_the_weight_landed_where_the_student_was_wrong():
    """Above 1: on positions already far from the teacher. Below 1: near it."""
    kl = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Linear in kl, so the correlation is exactly +-1 and the assertions below
    # are calibration points rather than "it went the right way".
    aligned = torch.tensor([[0.7, 0.9, 1.1, 1.3]])
    opposed = torch.tensor([[1.3, 1.1, 0.9, 0.7]])
    hi, _ = _fold(_hand_terms(aligned, kl), torch.ones(1, 4), n_tasks=0)
    lo, _ = _fold(_hand_terms(opposed, kl), torch.ones(1, 4), n_tasks=0)
    assert hi["sign_weight/pos/w_kl_lift"] > 1.0
    assert lo["sign_weight/pos/w_kl_lift"] < 1.0
    assert hi["sign_weight/pos/w_kl_corr"] == pytest.approx(1.0, abs=1e-6)
    assert lo["sign_weight/pos/w_kl_corr"] == pytest.approx(-1.0, abs=1e-6)


def test_the_correlation_matches_the_textbook_one():
    torch.manual_seed(3)
    w = 1.0 + 0.2 * torch.randn(4, 9)
    kl = (torch.rand(4, 9) + 0.05) * (1.0 + w)
    m, _ = _fold(_hand_terms(w, kl), torch.ones(4, 9), n_tasks=0)
    expect = float(np.corrcoef(w.reshape(-1).numpy(), kl.reshape(-1).numpy())[0, 1])
    assert m["sign_weight/pos/w_kl_corr"] == pytest.approx(expect, abs=1e-6)


def test_the_bands_cut_the_table_not_the_observed_weights():
    """A position at the table's ceiling is in every band, at its floor in none.

    Cutting the observed distribution instead would report the same share at
    every step by construction, which is the one thing these are for.
    """
    kl = torch.tensor([[1.0, 1.0]])
    pre = torch.tensor([[1.25, 1.0]])
    m, _ = _fold(_hand_terms(pre, kl, pre=pre), torch.ones(1, 2), n_tasks=0)
    for q in POSITION_BANDS:
        tag = f"band{int(q * 100):02d}"
        assert m[f"sign_weight/pos/{tag}_n_share"] == pytest.approx(0.5)
        assert m[f"sign_weight/pos/{tag}_kl_share"] == pytest.approx(0.5)


def test_a_band_that_holds_the_kl_reports_a_lift_above_one():
    kl = torch.tensor([[9.0, 1.0]])
    pre = torch.tensor([[1.25, 1.0]])
    m, _ = _fold(_hand_terms(pre, kl, pre=pre), torch.ones(1, 2), n_tasks=0)
    assert m["sign_weight/pos/band75_kl_share"] == pytest.approx(0.9)
    assert m["sign_weight/pos/band75_kl_lift"] == pytest.approx(1.8)


def test_the_state_shares_add_to_one():
    bs, resp, k = 2, 6, 5
    on = _logprob(bs, resp, k, seed=4)
    state = torch.full((bs, resp, k), STATE_NEUTRAL_ON)
    state[..., 0] = STATE_AGREE_POS
    state[..., 1] = STATE_CONFLICT_ON_POS
    w = torch.ones(bs, resp, k)
    w[..., 0] = 1.25
    w[..., 1] = 0.75
    pre = position_weights(w, on)
    terms = position_decomposition_terms(
        position_weight=pre, applied_weight=pre, candidate_weight=w, state=state,
        on_task_logprob=on, teacher_kl=torch.rand(bs, resp) + 0.1, weight_range=(0.75, 1.25),
    )
    m, _ = _fold(terms, torch.ones(bs, resp), n_tasks=0)
    shares = [m[f"sign_weight/pos/w_share/{name}"] for name, _ in POSITION_STATE_GROUPS]
    # The denominator is w_pre - 1 ~ 0.05, so the float32 residual on the
    # numerator is amplified about twentyfold on its way into the share.
    assert sum(shares) == pytest.approx(1.0, abs=1e-5)


def test_the_masked_positions_are_excluded_from_every_reading():
    w = torch.tensor([[1.5, 1.5, 1.0]])
    kl = torch.tensor([[10.0, 10.0, 1.0]])
    mask = torch.tensor([[0.0, 0.0, 1.0]])
    m, _ = _fold(_hand_terms(w, kl), mask, n_tasks=0)
    assert m["sign_weight/pos/w_mean"] == pytest.approx(1.0)
    assert m["sign_weight/pos/kl_scale"] == pytest.approx(1.0)


def test_the_per_task_scopes_are_the_rows_of_that_task_only():
    w = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
    kl = torch.ones(2, 2)
    task_ids = torch.tensor([0, 1])
    m, _ = _fold(_hand_terms(w, kl), torch.ones(2, 2), task_ids=task_ids)
    assert m["sign_weight/alfworld/pos/w_mean"] == pytest.approx(2.0)
    assert m["sign_weight/search/pos/w_mean"] == pytest.approx(1.0)
    assert m["sign_weight/pos/w_mean"] == pytest.approx(1.5)


def test_no_metric_name_collides_with_the_reducers_max_min_dispatch():
    """``reduce_metrics`` dispatches on the substrings "max" and "min".

    A term called e.g. ``w_min`` would be reduced by max/min across ranks
    instead of averaged, which is a different number and looks like nothing.
    """
    w = torch.tensor([[1.1, 0.9]])
    m, _ = _fold(_hand_terms(w, torch.ones(1, 2)), torch.ones(1, 2), n_tasks=0)
    assert m, "no metrics rendered"
    for name in m:
        assert "max" not in name and "min" not in name, name


# --------------------------------------------------------------------------- #
# TokenStateCounts under the two modes
# --------------------------------------------------------------------------- #
def _tsc_inputs(bs=2, resp=3, k=4, vocab=32, seed=5):
    torch.manual_seed(seed)
    on = _logprob(bs, resp, k)
    ids = torch.randint(0, vocab, (bs, resp, k))
    state = torch.full((bs, resp, k), STATE_NEUTRAL_ON)
    state[..., 0] = STATE_AGREE_POS
    w = torch.ones(bs, resp, k)
    w[..., 0] = 1.25
    return on, ids, state, w


def test_position_mode_refuses_the_target_modes_normaliser():
    """Silently substituting Z is the mislabelling this branch removes."""
    on, ids, state, w = _tsc_inputs()
    tsc = TokenStateCounts(vocab_size=32, n_tasks=0, device="cpu", mode="position")
    with pytest.raises(AssertionError, match="position_scale"):
        tsc.update(
            support_ids=ids, state=state, weight=w, on_task_logprob=on,
            response_mask=torch.ones(2, 3),
        )


def test_the_position_mode_effect_is_the_nats_the_weighting_added():
    """Summed over tokens and positions, up to the tail nobody can be charged for.

    ``w_applied - 1 = sum_v p(v) (w(v)/m - 1) + tail (1/m - 1)``. The first term
    is what the table credits to tokens; the second belongs to the vocabulary
    outside the support and has no id to be filed under. Multiplying through by
    the position's KL turns the identity into nats, which is what makes the
    ranking comparable across positions of very different cost.
    """
    bs, resp, k, vocab = 2, 3, 5, 40
    on, ids, state, w = _tsc_inputs(bs, resp, k, vocab, seed=6)
    # Distinct ids per position, so no token is charged twice and the totals are
    # a clean sum rather than a scatter with collisions.
    ids = torch.arange(bs * resp * k).reshape(bs, resp, k) % vocab
    kl = torch.rand(bs, resp) + 0.2
    m = 1.05
    pre = position_weights(w, on)
    applied = pre / m
    tsc = TokenStateCounts(vocab_size=vocab, n_tasks=0, device="cpu", mode="position")
    tsc.update(
        support_ids=ids, state=state, weight=w, on_task_logprob=on,
        response_mask=torch.ones(bs, resp),
        position_scale=torch.full((bs, resp), m), teacher_kl=kl,
    )
    got = float((tsc.eff_pos + tsc.eff_neg).sum())

    p = on.exp()
    tail = 1.0 - p.sum(-1)
    charged = (applied - 1.0) - tail * (1.0 / m - 1.0)
    assert got == pytest.approx(float((charged * kl).sum()), abs=1e-6)


def test_the_effect_column_is_named_by_the_mode():
    """nats and probabilities must never land in the same series."""
    on, ids, state, w = _tsc_inputs()
    mask = torch.ones(2, 3)
    tgt = TokenStateCounts(vocab_size=32, n_tasks=0, device="cpu", mode="target")
    tgt.update(support_ids=ids, state=state, weight=w, on_task_logprob=on, response_mask=mask)
    pos = TokenStateCounts(vocab_size=32, n_tasks=0, device="cpu", mode="position")
    pos.update(
        support_ids=ids, state=state, weight=w, on_task_logprob=on, response_mask=mask,
        position_scale=torch.ones(2, 3), teacher_kl=torch.ones(2, 3),
    )
    assert any("/token/dq_pos_sum" in k for k in tgt.scalar_metrics())
    assert any("/token/dkl_nats_pos_sum" in k for k in pos.scalar_metrics())
    assert not any("dkl" in k for k in tgt.scalar_metrics())
    assert not any("/token/dq_" in k for k in pos.scalar_metrics())
    for rows, kind in ((tgt.top_tokens(), "dq"), (pos.top_tokens(), "dkl_nats")):
        assert rows
        assert {r["effect_kind"] for r in rows} == {kind}
        assert all("effect_net" in r and "effect_gross" in r for r in rows)


def test_target_mode_is_untouched_by_the_position_branch():
    """The rewrite's dq is still ``p (w/Z - 1)`` with Z rebuilt per position."""
    bs, resp, k, vocab = 2, 3, 5, 40
    on, _ids, state, w = _tsc_inputs(bs, resp, k, vocab, seed=7)
    ids = torch.arange(bs * resp * k).reshape(bs, resp, k) % vocab
    tsc = TokenStateCounts(vocab_size=vocab, n_tasks=0, device="cpu", mode="target")
    tsc.update(
        support_ids=ids, state=state, weight=w, on_task_logprob=on,
        response_mask=torch.ones(bs, resp),
    )
    p = on.exp()
    z = (p * w).sum(-1) + (1.0 - p.sum(-1))
    expect = (p * (w / z.unsqueeze(-1) - 1.0)).sum()
    assert float((tsc.eff_pos + tsc.eff_neg).sum()) == pytest.approx(float(expect), abs=1e-6)


# --------------------------------------------------------------------------- #
# The gradient claims the module docstring makes
# --------------------------------------------------------------------------- #
def _kl_grad(student_logits, teacher_lp, *, position_weight=None, term_weight=None):
    z = student_logits.clone().requires_grad_(True)
    student_lp = torch.log_softmax(z, dim=-1)[..., : teacher_lp.size(-1)]
    if term_weight is None:
        kl = topk_kl_per_token(student_lp, teacher_lp)
        if position_weight is not None:
            kl = kl * position_weight
    else:
        p_s = student_lp.exp()
        kl = (term_weight * p_s * (student_lp - teacher_lp)).sum(-1)
    kl.sum().backward()
    return -z.grad.clone()


def test_a_position_weight_scales_every_logit_gradient_and_moves_nothing_else():
    """The claim that makes ``position`` a schedule and not a target change."""
    torch.manual_seed(8)
    logits = torch.randn(1, 1, 9)
    teacher = torch.log_softmax(torch.randn(1, 1, 9), dim=-1)[..., :5]
    plain = _kl_grad(logits, teacher)
    for w in (0.5, 1.7, 3.0):
        scaled = _kl_grad(logits, teacher, position_weight=torch.full((1, 1), w))
        assert torch.allclose(scaled, plain * w, atol=1e-6)


def test_weighting_the_kl_terms_themselves_pushes_an_agreed_token_down():
    """Why neither mode multiplies the per-candidate terms.

    The reverse-KL term at ``v`` is a cost the student pays for its OWN mass
    there, so scaling it up is an instruction to hold less of it. The sign only
    flips once the teacher holds more than ``e`` times the student's mass, which
    would make the weight's meaning depend on the student's current position
    rather than on what the teachers said.
    """
    teacher = torch.log_softmax(torch.tensor([[[1.4, 0.6, 0.1, -0.4, -1.0]]]), dim=-1)
    logits = torch.cat([teacher, torch.full((1, 1, 4), -6.0)], dim=-1)  # near-matched
    ones = torch.ones(1, 1, 5)
    plain = _kl_grad(logits, teacher, term_weight=ones)
    heavier = ones.clone()
    heavier[..., 0] = 1.5
    assert _kl_grad(logits, teacher, term_weight=heavier)[0, 0, 0] < plain[0, 0, 0]

    # ... and the far-gap regime where it flips, so the condition is pinned too.
    far = logits.clone()
    far[..., 0] -= 1.5
    assert _kl_grad(far, teacher, term_weight=heavier)[0, 0, 0] > _kl_grad(
        far, teacher, term_weight=ones
    )[0, 0, 0]
