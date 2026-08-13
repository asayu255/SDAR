"""Unit tests for cross-teacher sign-agreement weighting (multitask OPD+GRPO).

Covered:
* the sign table itself — agreement (both raise, or both lower) weights a
  candidate up, opposition weights it down, and anything else is left alone;
* the two "left alone" cases that decide whether the mechanism does anything at
  all: a shift inside the deadzone, and off-task teachers that split;
* the collapse to one weight per position (teacher-mass average, neutral tail);
* per-task mean-1 normalisation, without which the arm is indistinguishable from
  a larger ``teacher_kl_loss_coef``;
* what each mode does to the FIXED POINT of the distillation loss, found by
  actually minimising ``topk_kl_per_token`` rather than by assertion:
  ``position`` must leave it at the on-task teacher, ``target`` must move it to
  the reweighted teacher;
* the reason ``target`` reweights the distribution instead of the KL's terms —
  the loss is a reverse KL, so scaling a candidate's term pushes the student
  AWAY from that candidate;
* both degeneracies that make this arm's control valid: all-neutral weights, and
  ``agree_weight == disagree_weight == 1``.

CPU-only, no Ray or workers. If the verl stack is not installed the module is
skipped.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.sign_weights import (
        STATE_AGREE_NEG,
        STATE_AGREE_POS,
        STATE_CONFLICT_ON_NEG,
        STATE_CONFLICT_ON_POS,
        STATE_NEUTRAL_OFF,
        STATE_NEUTRAL_ON,
        candidate_weights,
        normalize_per_task,
        position_weights,
        reweight_teacher_logprobs,
        sign_state_metrics,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


AGREE, DISAGREE, DEADZONE = 1.25, 0.75, 0.1


def _weights(on_delta, off_deltas, *, deadzone=DEADZONE, agree=AGREE, disagree=DISAGREE):
    """Build (1, 1, k) inputs from plain lists of shifts against a zero base."""
    base = torch.zeros(1, 1, len(on_delta))
    on = torch.tensor(on_delta, dtype=torch.float32).view(1, 1, -1)
    off = torch.stack([torch.tensor(d, dtype=torch.float32).view(1, 1, -1) for d in off_deltas], dim=-1)
    return candidate_weights(on, off, base, agree_weight=agree, disagree_weight=disagree, deadzone=deadzone)


# --------------------------------------------------------------------------- #
# The sign table
# --------------------------------------------------------------------------- #


def test_sign_table_matches_the_four_cases():
    # candidates:      (+,+,+)  (-,-,-)   (+,-,-)   (-,+,+)
    w, state = _weights(
        on_delta=[1.0, -1.0, 1.0, -1.0],
        off_deltas=[[1.0, -1.0, -1.0, 1.0], [1.0, -1.0, -1.0, 1.0]],
    )
    assert w.view(-1).tolist() == [AGREE, AGREE, DISAGREE, DISAGREE]
    assert state.view(-1).tolist() == [
        STATE_AGREE_POS,
        STATE_AGREE_NEG,
        STATE_CONFLICT_ON_POS,
        STATE_CONFLICT_ON_NEG,
    ]


def test_mutual_suppression_is_agreement_not_conflict():
    """Both teachers lowering a token is shared structure, not a disagreement.

    Easy to get backwards: the *values* are both negative, but the judgements
    coincide, so the candidate must be weighted up like the (+,+) case.
    """
    w, state = _weights(on_delta=[-2.0], off_deltas=[[-1.5], [-0.9]])
    assert w.item() == AGREE
    assert state.item() == STATE_AGREE_NEG


# --------------------------------------------------------------------------- #
# The two neutral paths
# --------------------------------------------------------------------------- #


def test_deadzone_silences_a_teacher_that_barely_moved():
    # on-task inside the deadzone -> neutral regardless of what the others say
    w, state = _weights(on_delta=[0.05], off_deltas=[[1.0], [1.0]])
    assert w.item() == 1.0
    assert state.item() == STATE_NEUTRAL_ON

    # one off-task teacher inside the deadzone -> no consensus
    w, state = _weights(on_delta=[1.0], off_deltas=[[1.0], [0.05]])
    assert w.item() == 1.0
    assert state.item() == STATE_NEUTRAL_OFF

    # the same shift with a smaller deadzone does count
    w, _ = _weights(on_delta=[1.0], off_deltas=[[1.0], [0.05]], deadzone=0.01)
    assert w.item() == AGREE


def test_off_task_split_is_neutral():
    """One off-task teacher raising while the other lowers is not evidence."""
    w, state = _weights(on_delta=[1.0], off_deltas=[[1.0], [-1.0]])
    assert w.item() == 1.0
    assert state.item() == STATE_NEUTRAL_OFF


def test_deadzone_is_exclusive_at_the_boundary():
    """|delta| == deadzone counts as silence, so the threshold has one meaning."""
    w, _ = _weights(on_delta=[DEADZONE], off_deltas=[[1.0], [1.0]])
    assert w.item() == 1.0


# --------------------------------------------------------------------------- #
# Position mode
# --------------------------------------------------------------------------- #


def test_position_weight_averages_by_teacher_mass_with_a_neutral_tail():
    # teacher puts 0.5 on an agreeing candidate, 0.25 on a conflicting one,
    # leaving 0.25 of tail that carries no sign and must enter at weight 1.
    on_lp = torch.log(torch.tensor([0.5, 0.25])).view(1, 1, 2)
    cand_w = torch.tensor([AGREE, DISAGREE]).view(1, 1, 2)
    got = position_weights(cand_w, on_lp).item()
    assert got == pytest.approx(0.5 * AGREE + 0.25 * DISAGREE + 0.25 * 1.0)


def test_position_weight_is_bounded_by_the_candidate_weights():
    on_lp = torch.log(torch.tensor([0.6, 0.3, 0.05])).view(1, 1, 3)
    for cand in ([AGREE] * 3, [DISAGREE] * 3, [AGREE, DISAGREE, 1.0]):
        got = position_weights(torch.tensor(cand).view(1, 1, 3), on_lp).item()
        assert DISAGREE - 1e-6 <= got <= AGREE + 1e-6


def test_position_weight_is_exactly_one_when_nothing_is_flagged():
    on_lp = torch.log(torch.tensor([0.5, 0.3])).view(1, 1, 2)
    got = position_weights(torch.ones(1, 1, 2), on_lp).item()
    assert got == pytest.approx(1.0)


def test_normalize_per_task_gives_each_task_unit_mean():
    # two rows of task 0 (weights 1.5) and two of task 1 (weights 0.5): a single
    # batch-wide mean would leave both groups off 1.0 in opposite directions.
    weight = torch.tensor([[1.5, 1.5], [1.5, 1.5], [0.5, 0.5], [0.5, 0.5]])
    mask = torch.ones(4, 2)
    task_ids = torch.tensor([0, 0, 1, 1])
    out = normalize_per_task(weight, mask, task_ids)
    for t in (0, 1):
        rows = task_ids == t
        assert out[rows].mean().item() == pytest.approx(1.0)


def test_normalize_per_task_ignores_masked_tokens():
    weight = torch.tensor([[2.0, 99.0]])
    mask = torch.tensor([[1.0, 0.0]])
    out = normalize_per_task(weight, mask, torch.tensor([0]))
    assert out[0, 0].item() == pytest.approx(1.0)


def test_normalize_per_task_keeps_relative_order():
    weight = torch.tensor([[AGREE, 1.0, DISAGREE]])
    out = normalize_per_task(weight, torch.ones(1, 3), torch.tensor([0]))
    assert out[0, 0] > out[0, 1] > out[0, 2]


# --------------------------------------------------------------------------- #
# Target mode
# --------------------------------------------------------------------------- #


def test_target_reweighting_stays_a_valid_distribution():
    on_lp = torch.log(torch.tensor([0.5, 0.2, 0.1])).view(1, 1, 3)
    cand_w = torch.tensor([AGREE, DISAGREE, 1.0]).view(1, 1, 3)
    out = reweight_teacher_logprobs(on_lp, cand_w)
    mass = out.exp().sum(-1).item()
    assert 0.0 < mass < 1.0  # room left for the tail, as the KL's tail term needs


def test_target_reweighting_moves_mass_toward_agreement():
    """The point of ``target`` mode: agreed tokens end up above where the on-task
    teacher put them, opposed tokens below."""
    on_lp = torch.log(torch.tensor([0.4, 0.4, 0.1])).view(1, 1, 3)
    cand_w = torch.tensor([AGREE, DISAGREE, 1.0]).view(1, 1, 3)
    new_p = reweight_teacher_logprobs(on_lp, cand_w).exp().view(-1)
    old_p = on_lp.exp().view(-1)
    assert new_p[0] > old_p[0]  # agreed: up
    assert new_p[1] < old_p[1]  # opposed: down
    assert new_p[0] > new_p[1]


def test_target_reweighting_is_identity_when_all_neutral():
    on_lp = torch.log(torch.tensor([0.5, 0.2])).view(1, 1, 2)
    out = reweight_teacher_logprobs(on_lp, torch.ones(1, 1, 2))
    assert torch.allclose(out, on_lp, atol=1e-6)


def test_target_scale_is_anchored_by_the_tail():
    """Neutral == 1.0 == the tail's weight, so a fully neutral position cannot
    drift; and scaling every candidate up does move mass out of the tail, which
    is why the neutral value is not free to be chosen."""
    on_lp = torch.log(torch.tensor([0.4, 0.3])).view(1, 1, 2)
    scaled = reweight_teacher_logprobs(on_lp, torch.full((1, 1, 2), 2.0))
    assert scaled.exp().sum(-1).item() > on_lp.exp().sum(-1).item()


# --------------------------------------------------------------------------- #
# What the loss actually converges to
# --------------------------------------------------------------------------- #


def _minimise_student(teacher_lp, *, position_weight=1.0, steps=4000, lr=0.05):
    """Gradient-descend a 4-slot student (3 candidates + tail) on the real loss.

    Returns the student's converged probabilities over the 3 top-k candidates.
    """
    logits = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        student_lp = torch.log_softmax(logits, dim=-1)[:3].view(1, 1, 3)
        loss = position_weight * topk_kl_per_token(student_lp, teacher_lp).sum()
        loss.backward()
        opt.step()
    return torch.log_softmax(logits.detach(), dim=-1)[:3].exp()


def test_position_mode_leaves_the_fixed_point_at_the_on_task_teacher():
    """The safety claim, checked by minimising rather than by assertion."""
    teacher_lp = torch.log(torch.tensor([0.5, 0.3, 0.1])).view(1, 1, 3)
    for w in (DISAGREE, 1.0, AGREE):
        got = _minimise_student(teacher_lp, position_weight=w)
        assert torch.allclose(got, teacher_lp.exp().view(-1), atol=2e-3), f"weight {w} moved the target"


def test_target_mode_moves_the_fixed_point_to_the_reweighted_teacher():
    teacher_lp = torch.log(torch.tensor([0.4, 0.4, 0.1])).view(1, 1, 3)
    cand_w = torch.tensor([AGREE, DISAGREE, 1.0]).view(1, 1, 3)
    target_lp = reweight_teacher_logprobs(teacher_lp, cand_w)
    got = _minimise_student(target_lp)
    assert torch.allclose(got, target_lp.exp().view(-1), atol=2e-3)
    # and it is genuinely a different place than the plain teacher
    assert got[0] > teacher_lp.exp().view(-1)[0] + 1e-3


def test_multiplying_the_kl_terms_would_move_the_student_the_wrong_way():
    """Why ``target`` reweights the distribution and not the KL's terms.

    ``topk_kl_per_token`` is a reverse KL: the per-candidate term is a cost paid
    on the STUDENT's own mass. Scaling the term of a token both teachers endorse
    therefore teaches the student to avoid it. This test pins that trap so the
    "obvious" implementation is not reintroduced.
    """
    p = torch.tensor([0.5, 0.5])
    w = torch.tensor([AGREE, DISAGREE])

    def term_weighted_loss(q1):
        q = torch.stack([q1, 1 - q1])
        return (w * q * (q.log() - p.log())).sum()

    grid = torch.linspace(0.01, 0.99, 99)
    best = grid[torch.stack([term_weighted_loss(q) for q in grid]).argmin()]
    assert best < 0.5 - 0.05, "term-wise product should push mass AWAY from the up-weighted token"


# --------------------------------------------------------------------------- #
# Degeneracies that make the plain OPD+GRPO arm a valid control
# --------------------------------------------------------------------------- #


def test_equal_weights_are_a_no_op_in_both_modes():
    on_lp = torch.log(torch.tensor([0.5, 0.2, 0.1])).view(1, 1, 3)
    w, _ = _weights(on_delta=[1.0, -1.0, 1.0], off_deltas=[[1.0, -1.0, -1.0], [1.0, -1.0, -1.0]], agree=1.0, disagree=1.0)
    assert torch.allclose(w, torch.ones_like(w))
    assert position_weights(w, on_lp).item() == pytest.approx(1.0)
    assert torch.allclose(reweight_teacher_logprobs(on_lp, w), on_lp, atol=1e-6)


def test_all_neutral_batch_is_a_no_op():
    """No teacher moved anything: weights are 1 everywhere and the arm reduces to
    plain OPD+GRPO, which is what makes a null result interpretable."""
    on_lp = torch.log(torch.tensor([0.5, 0.2])).view(1, 1, 2)
    w, state = _weights(on_delta=[0.0, 0.0], off_deltas=[[0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(w, torch.ones_like(w))
    assert (state == STATE_NEUTRAL_ON).all()
    assert position_weights(w, on_lp).item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_state_metrics_are_shares_over_valid_tokens_only():
    _, state = _weights(
        on_delta=[1.0, 1.0, 0.0, 0.0],
        off_deltas=[[1.0, -1.0, 1.0, 1.0], [1.0, -1.0, 1.0, 1.0]],
    )
    state = state.expand(1, 2, 4).clone()
    mask = torch.tensor([[1.0, 0.0]])  # second position is padding
    out = sign_state_metrics(state, mask)
    assert out["sign_weight/frac_agree_pos"] == pytest.approx(0.25)
    assert out["sign_weight/frac_conflict_on_pos"] == pytest.approx(0.25)
    assert out["sign_weight/frac_neutral_on_task_silent"] == pytest.approx(0.5)
    assert sum(out.values()) == pytest.approx(1.0)


def test_state_metrics_are_empty_for_an_all_padding_batch():
    _, state = _weights(on_delta=[1.0], off_deltas=[[1.0], [1.0]])
    assert sign_state_metrics(state, torch.zeros(1, 1)) == {}
