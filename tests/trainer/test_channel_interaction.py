"""Which of the two channels the reward signal disagrees with, and where each aims.

``e(v) = |c(v)| + sum_m alpha_{d,m} |dhat_m(v)|`` -- both terms non-negative,
both added. So the channels never push a position in opposite directions, and
"are the two signals in conflict" cannot be answered at a position at all. What
is fixed is the per-task budget: ``W = W~/mu_d`` preserves the task mean, so
raising one position lowers every other one's share. The competition is over
WHICH POSITIONS each channel picks, and whether what it picks pulls with the
reward.

Two readings, both off the exact partition ``build_position_weight`` already
makes -- ``W - 1 = S + sum_m R_m + (1/mu - 1)``:

``channel/allocation_{cosine,corr}``  do S and R aim at the same positions.
``grpo/{shared,source}_grad_cosine``  does each channel's OWN addition to the
                                      logit push, ``S g0`` and ``R g0``, pull
                                      with the policy gradient. The existing
                                      ``grpo/grad_cosine`` is taken on the
                                      applied ``W``, which carries the base OPD
                                      direction and both channels at once.
"""

import math

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        GRAD_TERMS,
        POSITION_TERMS,
        ROLE_CUT_SUFFIXES,
        gradient_metrics,
        logit_gradient_terms,
        position_terms,
        position_weight_metrics,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _sums(cols, mask=None):
    """What ScopeTermStats would hand the renderer, for one pooled scope."""
    mask = torch.ones_like(next(iter(cols.values()))) if mask is None else mask
    tot = {name: float((col * mask).sum()) for name, col in cols.items()}
    tot["n"] = float(mask.sum())
    return {None: tot}


def _built(*, shared, source, weight=None, mu=1.0):
    """The fields position_terms reads, with the two channels set directly."""
    sh = torch.tensor(shared, dtype=torch.float32)
    sr = torch.tensor(source, dtype=torch.float32)
    w = (1.0 + sh + sr) if weight is None else torch.tensor(weight, dtype=torch.float32)
    return {
        "weight": w,
        "pre_weight": w * mu,
        "mu": torch.full_like(w, mu),
        "available": torch.ones(w.size(0), dtype=torch.bool),
        # Pre-mu, so a column that read the evidence where it should read the
        # partition comes out mu times too large instead of identical.
        "evidence_shared": sh * mu,
        "evidence_shared_offtask_only": sh * mu,
        "push_shared": sh,
        "push_by_source": sr.unsqueeze(-1),
    }


# --------------------------------------------------------------------------- #
# where each channel aims
# --------------------------------------------------------------------------- #
def test_the_channel_columns_are_the_exact_partition_and_not_the_evidence():
    """S and R have to be the shares of W - 1, not of W~ - 1: the evidence
    columns beside them are pre-mu, and mixing the two would make the two
    readings below disagree with every effect metric in the run."""
    built = _built(shared=[[0.2, 0.4]], source=[[0.1, 0.3]], mu=2.0)
    cols = position_terms(built, torch.ones(1, 2))
    assert torch.allclose(cols["push_shared"], torch.tensor([[0.2, 0.4]]))
    assert torch.allclose(cols["push_source"], torch.tensor([[0.1, 0.3]]))
    # And they add to W - 1 with the normaliser offset, which is what makes a
    # share of them a share of the applied budget.
    total = cols["push_shared"] + cols["push_source"] + (1.0 / 2.0 - 1.0)
    assert torch.allclose(total, built["weight"] - 1.0 + (1.0 / 2.0 - 1.0), atol=1e-6)


def test_two_channels_aiming_at_the_same_positions_read_as_redundant():
    m = position_weight_metrics(_sums(position_terms(
        _built(shared=[[1.0, 0.0, 1.0, 0.0]], source=[[2.0, 0.0, 2.0, 0.0]]),
        torch.ones(1, 4),
    )))
    assert m["kl_weight/channel/allocation_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert m["kl_weight/channel/allocation_corr"] == pytest.approx(1.0, abs=1e-6)


def test_channels_that_split_the_text_read_as_negatively_correlated():
    """The reading the uncentered cosine cannot give. Both channels only ever
    ADD, so their raw overlap is bounded below by 0 and 'they never oppose' is
    true by construction; against a task budget mu holds fixed, picking
    disjoint positions IS taking budget from each other."""
    cols = position_terms(
        _built(shared=[[1.0, 0.0, 1.0, 0.0]], source=[[0.0, 1.0, 0.0, 1.0]]),
        torch.ones(1, 4),
    )
    m = position_weight_metrics(_sums(cols))
    assert m["kl_weight/channel/allocation_cosine"] == pytest.approx(0.0, abs=1e-6)
    assert m["kl_weight/channel/allocation_corr"] == pytest.approx(-1.0, abs=1e-6)


def test_the_uncentered_cosine_cannot_go_negative_whatever_the_allocation():
    """Pinned so the pair is never read as two versions of one number: a
    non-negative overlap is a property of two non-negative channels, and only
    the centered one carries the competition."""
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        cols = position_terms(
            _built(shared=torch.rand((2, 8), generator=g).tolist(),
                   source=torch.rand((2, 8), generator=g).tolist()),
            torch.ones(2, 8),
        )
        m = position_weight_metrics(_sums(cols))
        assert m["kl_weight/channel/allocation_cosine"] >= -1e-9


def test_a_masked_position_is_outside_both_channel_moments():
    """Second moments over padding would drag both variances toward the mean of
    a region the arm never weighted."""
    cols = position_terms(
        _built(shared=[[1.0, 9.0, 1.0, 9.0]], source=[[1.0, 0.0, 1.0, 0.0]]),
        torch.ones(1, 4),
    )
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    m = position_weight_metrics(_sums(cols, mask))
    assert m["kl_weight/channel/shared_push_mean"] == pytest.approx(1.0)
    assert m["kl_weight/channel/source_push_mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# which channel the reward disagrees with
# --------------------------------------------------------------------------- #
def _grad(*, shared, source, teacher_shift, pg_coef, k=3, bs=1, resp=2):
    """One call of the real gradient path, with the student and teacher set so
    ``g0`` is non-degenerate and the policy coefficient set per position."""
    g = torch.Generator().manual_seed(2)
    student = torch.log_softmax(torch.randn((bs, resp, k), generator=g), dim=-1)
    teacher = torch.log_softmax(student + torch.tensor(teacher_shift), dim=-1)
    kl = (student.exp() * (student - teacher)).sum(dim=-1)
    onehot = torch.zeros((bs, resp, k))
    onehot[..., 0] = 1.0
    return logit_gradient_terms(
        student_logprob=student, teacher_logprob=teacher,
        weight=torch.ones((bs, resp)), teacher_kl=kl,
        pg_grad_coef=torch.tensor(pg_coef, dtype=torch.float32),
        sampled_onehot=onehot, coef=1.0,
        push_shared=torch.tensor(shared, dtype=torch.float32),
        push_source=torch.tensor(source, dtype=torch.float32),
    )


SHIFT = [[[0.4, -0.2, 0.1], [-0.3, 0.5, -0.1]]]


def test_the_two_channels_can_disagree_with_the_reward_in_opposite_directions():
    """The arm choice rests on exactly this: one channel positive and the other
    negative means the composite is spending part of its budget against the
    reward signal, and the pooled cosine -- taken on W, which carries the base
    OPD direction and both channels -- cannot show it."""
    # Each channel is given one position, so its cosine carries that position's
    # g0 . g_grpo alone. Flipping the policy coefficient flips that dot AT that
    # position, but the two positions have independent g0 directions -- so the
    # flip that makes them disagree is read off the data rather than assumed.
    probe = _grad(shared=[[1.0, 0.0]], source=[[0.0, 1.0]],
                  teacher_shift=SHIFT, pg_coef=[[1.0, 1.0]])
    d0 = float(probe["g_shared_dot"].sum())
    d1 = float(probe["g_source_dot"].sum())
    assert d0 != 0.0 and d1 != 0.0
    cols = _grad(shared=[[1.0, 0.0]], source=[[0.0, 1.0]], teacher_shift=SHIFT,
                 pg_coef=[[1.0, -1.0 if d0 * d1 > 0 else 1.0]])
    m = gradient_metrics(_sums(cols))
    sh = m["kl_weight/grpo/shared_grad_cosine"]
    src = m["kl_weight/grpo/source_grad_cosine"]
    assert sh * src < 0, (sh, src)
    # Both substantial, so the split is not one channel reading as ~0 noise.
    assert min(abs(sh), abs(src)) > 0.1
    # And the pooled cosine on W cannot report the split: it is one number.
    assert "kl_weight/grpo/grad_cosine" in m


def test_a_channel_that_allocated_nothing_reports_no_cosine():
    """Zero over zero is not a cosine of 0, and a flat 0 series reads as
    'the channel is orthogonal to the reward' rather than 'absent'."""
    cols = _grad(shared=[[1.0, 1.0]], source=[[0.0, 0.0]],
                 teacher_shift=SHIFT, pg_coef=[[1.0, 1.0]])
    m = gradient_metrics(_sums(cols))
    assert "kl_weight/grpo/shared_grad_cosine" in m
    assert "kl_weight/grpo/source_grad_cosine" not in m
    assert m["kl_weight/channel/source_grad_norm"] == pytest.approx(0.0)


def test_the_channels_cannot_oppose_each_other_at_a_logit():
    """Both scale the SAME g0, so their mutual cosine is >= 0 for any
    allocation. Pinned because a positive value here is redundancy, not
    agreement, and the allocation correlation is the one that can go negative."""
    g = torch.Generator().manual_seed(9)
    for _ in range(20):
        sh = torch.rand((1, 2), generator=g)
        sr = torch.rand((1, 2), generator=g)
        pg = (torch.rand((1, 2), generator=g) - 0.5) * 4
        cols = _grad(shared=sh.tolist(), source=sr.tolist(),
                     teacher_shift=SHIFT, pg_coef=pg.tolist())
        m = gradient_metrics(_sums(cols))
        assert m["kl_weight/channel/shared_source_grad_cosine"] >= -1e-6


def test_the_channel_gradient_is_the_channel_s_share_of_the_added_push():
    """``(W - 1) g0`` is the mechanism's addition and S + R + offset partitions
    ``W - 1``, so ``S g0`` is the shared channel's part of it -- exactly, with
    no counterfactual normaliser. A channel scaled by t scales its norm by t."""
    one = _grad(shared=[[1.0, 1.0]], source=[[0.0, 0.0]],
                teacher_shift=SHIFT, pg_coef=[[1.0, 1.0]])
    three = _grad(shared=[[3.0, 3.0]], source=[[0.0, 0.0]],
                  teacher_shift=SHIFT, pg_coef=[[1.0, 1.0]])
    n1 = math.sqrt(float(one["g_shared_sq"].sum()))
    n3 = math.sqrt(float(three["g_shared_sq"].sum()))
    assert n3 == pytest.approx(3.0 * n1, rel=1e-5)
    # And the cosine, being scale free, does not move.
    m1 = gradient_metrics(_sums(one))["kl_weight/grpo/shared_grad_cosine"]
    m3 = gradient_metrics(_sums(three))["kl_weight/grpo/shared_grad_cosine"]
    assert m1 == pytest.approx(m3, abs=1e-6)


def test_the_channel_columns_default_to_zero_so_the_term_tuple_is_fixed():
    """The accumulator is built from GRAD_TERMS once; a caller that omits the
    partition must still produce every column or the index_add_ writes a short
    row into the wrong cells."""
    g = torch.Generator().manual_seed(4)
    student = torch.log_softmax(torch.randn((1, 2, 3), generator=g), dim=-1)
    # A teacher that actually differs, so g0 is non-zero and the columns are
    # zero because the CHANNEL is absent rather than because nothing is pushing.
    teacher = torch.log_softmax(student + torch.tensor(SHIFT), dim=-1)
    kl = (student.exp() * (student - teacher)).sum(dim=-1)
    onehot = torch.zeros((1, 2, 3))
    onehot[..., 0] = 1.0
    common = dict(
        student_logprob=student, teacher_logprob=teacher,
        weight=torch.ones((1, 2)), teacher_kl=kl,
        pg_grad_coef=torch.ones((1, 2)), sampled_onehot=onehot, coef=1.0,
    )
    cols = logit_gradient_terms(**common)
    assert set(cols) == set(GRAD_TERMS)
    # The pooled columns are live, so the fixture is not silently degenerate.
    assert float(cols["g_opd_sq"].sum()) > 0.0
    assert float(cols["g_grpo_sq"].sum()) > 0.0
    for name in ("g_shared_sq", "g_shared_dot", "g_source_sq", "g_source_dot", "g_cross_dot"):
        assert float(cols[name].abs().sum()) == 0.0, name
    # And a channel supplied alone leaves only the other one at zero.
    only_shared = logit_gradient_terms(**common, push_shared=torch.ones((1, 2)))
    assert float(only_shared["g_shared_sq"].sum()) > 0.0
    assert float(only_shared["g_source_sq"].abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------------- #
def test_the_new_columns_are_declared_in_the_term_tuples():
    """ScopeTermStats sizes its buffer from these; a column produced but not
    named is silently dropped, and a column named but not produced is a KeyError
    inside the micro-batch loop."""
    for name in ("push_shared", "push_source", "push_shared_sq",
                 "push_source_sq", "push_cross"):
        assert name in POSITION_TERMS
    for name in ("g_shared_sq", "g_shared_dot", "g_source_sq",
                 "g_source_dot", "g_cross_dot"):
        assert name in GRAD_TERMS


def test_the_role_cut_publishes_the_per_channel_cosines():
    """The format-concentration question IS the role cut: shared corroboration
    is meant to find structure common to every teacher, so it landing on format
    is by design -- what is not settled is whether that budget pulls with the
    reward."""
    assert "/grpo/shared_grad_cosine" in ROLE_CUT_SUFFIXES
    assert "/grpo/source_grad_cosine" in ROLE_CUT_SUFFIXES
