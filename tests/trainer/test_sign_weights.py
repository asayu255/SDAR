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
"""Cross-teacher sign weighting: the weight tables, the rewrite, the diagnostics.

The two modes do NOT share a table, and most of what is checked here is that
difference. ``position`` scales a KL term, which has no direction, so agreement
counts the same either way. ``target`` scales a PROBABILITY, so agreeing to lower
a token has to lower it -- weighting both agreements up would raise the target
probability of tokens every teacher agreed to suppress.
"""

import math

import pytest
import torch

from verl.trainer.ppo.sign_weights import (
    STATE_AGREE_NEG,
    STATE_AGREE_POS,
    STATE_CONFLICT_ON_NEG,
    STATE_CONFLICT_ON_POS,
    STATE_NEUTRAL_OFF_SILENT,
    STATE_NEUTRAL_OFF_SPLIT,
    STATE_NEUTRAL_ON,
    SignWeightStats,
    candidate_weights,
    normalize_per_task,
    position_weights,
    reweight_teacher_logprobs,
)

AGREE = 1.25
AGREE_NEG = 0.75
DEADZONE = 0.1


def _weights(on_delta, off_deltas, *, mode, disagree=1.0, base=-2.0):
    """One position of k candidates, built from the shifts each teacher applied.

    ``base`` is the base policy's log-prob; the teachers sit at ``base + delta``,
    which is what the mechanism reads.
    """
    k = len(on_delta)
    base_lp = torch.full((1, 1, k), float(base))
    on_lp = base_lp + torch.tensor(on_delta).view(1, 1, k)
    off_lp = torch.stack(
        [base_lp + torch.tensor(d).view(1, 1, k) for d in off_deltas], dim=-1
    )
    return candidate_weights(
        on_lp,
        off_lp,
        base_lp,
        mode=mode,
        agree_weight=AGREE,
        agree_neg_weight=AGREE_NEG,
        disagree_weight=disagree,
        deadzone=DEADZONE,
    )


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "on,off,state",
    [
        (1.0, (1.0, 1.0), STATE_AGREE_POS),
        (-1.0, (-1.0, -1.0), STATE_AGREE_NEG),
        (1.0, (-1.0, -1.0), STATE_CONFLICT_ON_POS),
        (-1.0, (1.0, 1.0), STATE_CONFLICT_ON_NEG),
    ],
)
def test_states_are_read_from_the_four_sign_combinations(on, off, state):
    for mode in ("position", "target"):
        _, s = _weights([on], [[off[0]], [off[1]]], mode=mode)
        assert s.item() == state


def test_position_weights_both_agreements_up():
    """No direction: the weight buys learning effort, and a KL term has no sign."""
    w_pos, _ = _weights([1.0], [[1.0], [1.0]], mode="position")
    w_neg, _ = _weights([-1.0], [[-1.0], [-1.0]], mode="position")
    assert w_pos.item() == pytest.approx(AGREE)
    assert w_neg.item() == pytest.approx(AGREE)


def test_target_lowers_a_token_every_teacher_lowered():
    """The whole point of the split table.

    In target mode the weight multiplies the teacher's probability, so weighting
    mutual suppression UP -- which is right for position -- would hand more mass
    to a token all three teachers agreed to demote, undoing the edit the
    agreement is evidence for.
    """
    w_pos, _ = _weights([1.0], [[1.0], [1.0]], mode="target")
    w_neg, _ = _weights([-1.0], [[-1.0], [-1.0]], mode="target")
    assert w_pos.item() == pytest.approx(AGREE)
    assert w_neg.item() == pytest.approx(AGREE_NEG)
    assert w_neg.item() < 1.0 < w_pos.item()


def test_conflict_is_neutral_at_the_configured_default():
    for mode in ("position", "target"):
        w, _ = _weights([1.0], [[-1.0], [-1.0]], mode=mode)
        assert w.item() == pytest.approx(1.0)


def test_position_mode_may_attenuate_conflict():
    w, s = _weights([1.0], [[-1.0], [-1.0]], mode="position", disagree=0.75)
    assert w.item() == pytest.approx(0.75)
    assert s.item() == STATE_CONFLICT_ON_POS


def test_target_mode_refuses_a_directionless_conflict_weight():
    """A single conflict factor cannot say which way to pull a probability.

    Deferring to the objecting teachers means lowering a token the on-task
    teacher raised and RAISING one it lowered; one number below 1.0 does the
    second one backwards. Refuse rather than pick silently.
    """
    with pytest.raises(AssertionError, match="direction-aware conflict"):
        _weights([1.0], [[-1.0], [-1.0]], mode="target", disagree=0.75)


# --------------------------------------------------------------------------- #
# The deadzone and the two neutral paths
# --------------------------------------------------------------------------- #


def test_deadzone_silences_a_teacher_that_barely_moved():
    w, s = _weights([1.0], [[1.0], [0.05]], mode="target")
    assert w.item() == pytest.approx(1.0)
    assert s.item() == STATE_NEUTRAL_OFF_SILENT


def test_a_split_and_a_silence_are_different_states():
    """Both leave the weight at 1.0, and they are different diagnoses.

    A split says the other tasks really do pull apart at this candidate; a
    silence says the deadzone swallowed the evidence. Lumping them -- as the
    single neutral_off state used to -- makes the most common state in the run
    unreadable.
    """
    _, split = _weights([1.0], [[1.0], [-1.0]], mode="target")
    _, silent = _weights([1.0], [[1.0], [0.0]], mode="target")
    assert split.item() == STATE_NEUTRAL_OFF_SPLIT
    assert silent.item() == STATE_NEUTRAL_OFF_SILENT


def test_on_task_silence_wins_over_everything():
    _, s = _weights([0.05], [[1.0], [1.0]], mode="target")
    assert s.item() == STATE_NEUTRAL_ON


def test_deadzone_is_exclusive_at_the_boundary():
    """|delta| == deadzone is inside it, so the threshold has one meaning.

    base=0 so the shift the mechanism reads is the literal delta: at base=-2 the
    subtraction leaves 0.1 as 0.10000000000000009 and the boundary is untestable.
    """
    _, at = _weights([DEADZONE], [[1.0], [1.0]], mode="target", base=0.0)
    _, just_over = _weights([DEADZONE + 1e-3], [[1.0], [1.0]], mode="target", base=0.0)
    assert at.item() == STATE_NEUTRAL_ON
    assert just_over.item() == STATE_AGREE_POS


# --------------------------------------------------------------------------- #
# position: the collapse to one weight per token
# --------------------------------------------------------------------------- #


def test_position_weight_averages_by_teacher_mass_with_a_neutral_tail():
    on_lp = torch.log(torch.tensor([[[0.5, 0.2]]]))
    w = torch.tensor([[[1.25, 1.0]]])
    # 0.5*1.25 + 0.2*1.0 + tail 0.3*1.0
    assert position_weights(w, on_lp).item() == pytest.approx(0.625 + 0.2 + 0.3)


def test_position_weight_is_exactly_one_when_nothing_is_flagged():
    on_lp = torch.log(torch.tensor([[[0.5, 0.2]]]))
    w = torch.ones(1, 1, 2)
    assert position_weights(w, on_lp).item() == pytest.approx(1.0)


def test_position_weight_cannot_be_swung_by_tokens_the_teacher_ruled_out():
    """One confident token outweighs nineteen the teacher has abandoned."""
    on_lp = torch.log(torch.tensor([[[0.9] + [0.001] * 19]]))
    w = torch.tensor([[[1.0] + [1.25] * 19]])
    assert position_weights(w, on_lp).item() < 1.01


# --------------------------------------------------------------------------- #
# position: the per-task normalisation
# --------------------------------------------------------------------------- #


def test_normalize_per_task_gives_each_task_unit_mean():
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]])
    mask = torch.ones_like(w)
    tasks = torch.tensor([0, 0, 1])
    out = normalize_per_task(w, mask, tasks)
    assert out[:2].mean().item() == pytest.approx(1.0)
    assert out[2].mean().item() == pytest.approx(1.0)


def test_normalize_per_task_uses_supplied_means_when_given():
    """The weights only exist inside the forward, which sees one micro-batch.

    Normalising by a micro-batch's own mean would make the objective depend on
    how the batch happened to be split, so the caller passes the previous step's
    per-task means instead.
    """
    w = torch.full((2, 2), 1.25)
    mask = torch.ones_like(w)
    tasks = torch.tensor([0, 1])
    out = normalize_per_task(w, mask, tasks, means={0: 1.25, 1: 1.0})
    assert out[0].mean().item() == pytest.approx(1.0)
    assert out[1].mean().item() == pytest.approx(1.25)


def test_normalize_per_task_ignores_masked_tokens():
    w = torch.tensor([[1.0, 99.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert normalize_per_task(w, mask)[0, 0].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# target: the rewrite
# --------------------------------------------------------------------------- #


def test_target_rewrite_matches_the_worked_example():
    """p=(0.4, 0.2), tail 0.4, w=(1.25, 0.75) -> Z=1.05."""
    on_lp = torch.log(torch.tensor([[[0.4, 0.2]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    out = reweight_teacher_logprobs(on_lp, w).exp()
    assert out[0, 0, 0].item() == pytest.approx(0.50 / 1.05, abs=1e-6)
    assert out[0, 0, 1].item() == pytest.approx(0.15 / 1.05, abs=1e-6)
    # The tail is not returned; the KL recovers it as 1 - sum, and that is
    # exactly tail / Z -- weight 1.0 fixes its numerator, not its share.
    assert (1.0 - out.sum(-1)).item() == pytest.approx(0.40 / 1.05, abs=1e-6)


def test_target_rewrite_is_exactly_identity_when_nothing_is_flagged():
    """Z = sum(p) + tail = 1, so the rewrite is not an intervention of its own.

    This is what rules out "the renormalisation is what helped": with uniform
    weights it does nothing at all.
    """
    on_lp = torch.log(torch.rand(3, 5, 7) * 0.1)
    out = reweight_teacher_logprobs(on_lp, torch.ones(3, 5, 7))
    assert (out - on_lp).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_target_rewrite_stays_a_valid_distribution():
    on_lp = torch.log(torch.rand(4, 6, 5) * 0.15)
    w = torch.where(torch.rand(4, 6, 5) > 0.5, 1.25, 0.75)
    out = reweight_teacher_logprobs(on_lp, w).exp()
    assert (out >= 0).all()
    assert (out.sum(-1) <= 1.0 + 1e-5).all()


def test_target_scale_is_anchored_by_the_tail():
    """Scaling every weight by a constant is NOT a no-op, and that is the point.

    The tail keeps its weight of 1.0, so only the top-k side of Z moves. Without
    that anchor the mechanism would have no absolute scale and a uniform 2x would
    silently mean "move all the mass out of the tail".
    """
    on_lp = torch.log(torch.tensor([[[0.4, 0.2]]]))
    once = reweight_teacher_logprobs(on_lp, torch.full((1, 1, 2), 1.0)).exp()
    twice = reweight_teacher_logprobs(on_lp, torch.full((1, 1, 2), 2.0)).exp()
    assert twice.sum().item() > once.sum().item()


# --------------------------------------------------------------------------- #
# What each mode does to the thing the loss is minimised by
# --------------------------------------------------------------------------- #


def _reverse_kl(student_lp, teacher_lp):
    """The k+1 category reverse KL the actor computes, in one place for the tests."""
    p_s, p_t = student_lp.exp(), teacher_lp.exp()
    tail_s = (1.0 - p_s.sum(-1)).clamp(min=1e-8)
    tail_t = (1.0 - p_t.sum(-1)).clamp(min=1e-8)
    return (p_s * (student_lp - teacher_lp)).sum(-1) + tail_s * (tail_s.log() - tail_t.log())


def _descend_to_the_minimiser(teacher_lp, weight=None, steps=4000, lr=0.1):
    """Where the student ends up under this loss, found by actually descending."""
    logits = torch.zeros(1, 1, teacher_lp.size(-1) + 1, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lp = torch.log_softmax(logits, dim=-1)
        loss = _reverse_kl(lp[..., :-1], teacher_lp)
        if weight is not None:
            loss = loss * weight
        loss.sum().backward()
        opt.step()
    return torch.log_softmax(logits, dim=-1).detach().exp()[..., :-1]


def test_position_mode_leaves_the_minimiser_at_the_on_task_teacher():
    """A positive per-token scalar cannot move where the loss is minimised."""
    teacher = torch.log(torch.tensor([[[0.5, 0.3]]]))
    got = _descend_to_the_minimiser(teacher, weight=torch.tensor([[1.25]]))
    assert got[0, 0, 0].item() == pytest.approx(0.5, abs=2e-3)
    assert got[0, 0, 1].item() == pytest.approx(0.3, abs=2e-3)


def test_target_mode_moves_the_minimiser_to_the_reweighted_teacher():
    teacher = torch.log(torch.tensor([[[0.5, 0.3]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    rewritten = reweight_teacher_logprobs(teacher, w)
    got = _descend_to_the_minimiser(rewritten)
    assert got[0, 0, 0].item() == pytest.approx(rewritten.exp()[0, 0, 0].item(), abs=2e-3)
    assert got[0, 0, 0].item() > 0.5  # both teachers raised it, so it gains mass


def test_multiplying_the_kl_terms_would_move_the_student_the_wrong_way():
    """Kept as an executable warning, not as a feature.

    The per-candidate term of a reverse KL is a cost the student pays for its OWN
    mass at that candidate. Scaling the term of a token both teachers endorsed
    makes the student flee it -- the opposite of what the weight is for. This is
    the operation ``target`` mode is defined to avoid.
    """
    teacher = torch.log(torch.tensor([[[0.5, 0.5]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    logits = torch.zeros(1, 1, 3, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(4000):
        opt.zero_grad()
        lp = torch.log_softmax(logits, dim=-1)
        p_s = lp[..., :-1].exp()
        tail_s = (1.0 - p_s.sum(-1)).clamp(min=1e-8)
        tail_t = 1e-8
        per_term = w * p_s * (lp[..., :-1] - teacher)
        (per_term.sum(-1) + tail_s * (tail_s.log() - math.log(tail_t))).sum().backward()
        opt.step()
    got = torch.log_softmax(logits, dim=-1).exp()[0, 0]
    assert got[0].item() < got[1].item()


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def _stats_inputs(k=3):
    base = torch.zeros(2, 1, k)
    on = base + torch.tensor([[[1.0, -1.0, 0.0]]]).expand(2, 1, k).clone()
    off = torch.stack([base + 1.0, base + 1.0], dim=-1)
    return on, off, base


def test_state_fractions_are_pooled_over_micro_batches_not_averaged():
    """Two micro-batches with different valid-token counts must pool, not average."""
    on, off, base = _stats_inputs()
    _, state = candidate_weights(
        on, off, base, mode="target", agree_weight=AGREE, agree_neg_weight=AGREE_NEG,
        disagree_weight=1.0, deadzone=DEADZONE,
    )
    full = torch.ones(2, 1)
    half = torch.tensor([[1.0], [0.0]])

    pooled = SignWeightStats()
    for mask in (full, half):
        pooled.update_candidates(
            state=state, on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
            response_mask=mask, deadzone=DEADZONE,
        )
    m = pooled.metrics()
    # 3 rows' worth of candidates were valid (2 + 1), 3 candidates each.
    assert m["sign_weight/n_candidates"] == pytest.approx(9.0)
    assert m["sign_weight/n_tokens"] == pytest.approx(3.0)
    # Every valid candidate set is the same, so the shares are exact thirds.
    assert m["sign_weight/frac_agree_pos"] == pytest.approx(1 / 3)
    assert m["sign_weight/frac_conflict_on_neg"] == pytest.approx(1 / 3)
    assert m["sign_weight/frac_neutral_on_task_silent"] == pytest.approx(1 / 3)


def test_mass_fractions_weight_by_the_teacher_probability():
    """Count share and mass share are different numbers, and mass is the one that
    says how much of the target the mechanism can actually move."""
    on = torch.log(torch.tensor([[[0.6, 0.02]]]))
    base = torch.log(torch.tensor([[[0.2, 0.5]]]))
    off = torch.stack([base + 1.0, base + 1.0], dim=-1)
    _, state = candidate_weights(
        on, off, base, mode="target", agree_weight=AGREE, agree_neg_weight=AGREE_NEG,
        disagree_weight=1.0, deadzone=DEADZONE,
    )
    st = SignWeightStats()
    st.update_candidates(
        state=state, on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
        response_mask=torch.ones(1, 1), deadzone=DEADZONE,
    )
    m = st.metrics()
    # One of two candidates agreed up, so the count share is a half...
    assert m["sign_weight/frac_agree_pos"] == pytest.approx(0.5)
    # ...but it holds 0.6 of the 0.62 covered mass.
    assert m["sign_weight/mass_frac_agree_pos"] == pytest.approx(0.6 / 0.62, abs=1e-5)


def test_pairwise_agreement_rate_is_reported_per_ordered_teacher_pair():
    base = torch.zeros(2, 1, 2)
    on = base + torch.tensor([[[1.0, 1.0]]])
    # plane 0 agrees on both candidates; plane 1 agrees on one and opposes on one.
    off = torch.stack([base + 1.0, base + torch.tensor([[[1.0, -1.0]]])], dim=-1)
    _, state = candidate_weights(
        on, off, base, mode="target", agree_weight=AGREE, agree_neg_weight=AGREE_NEG,
        disagree_weight=1.0, deadzone=DEADZONE,
    )
    st = SignWeightStats(task_names=["alfworld", "search", "webshop"])
    st.update_candidates(
        state=state, on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
        response_mask=torch.ones(2, 1), deadzone=DEADZONE,
        task_ids=torch.tensor([0, 0]),
        off_plane_tasks=torch.tensor([[1, 2], [1, 2]]),
    )
    m = st.metrics()
    assert m["sign_weight/agree_rate/search__on__alfworld"] == pytest.approx(1.0)
    assert m["sign_weight/agree_rate/webshop__on__alfworld"] == pytest.approx(0.5)


def test_per_teacher_shift_and_deadzone_occupancy():
    base = torch.zeros(1, 1, 2)
    on = base + torch.tensor([[[2.0, 0.0]]])  # alfworld: one big move, one silence
    off = torch.stack([base + 0.5, base + 0.5], dim=-1)
    _, state = candidate_weights(
        on, off, base, mode="target", agree_weight=AGREE, agree_neg_weight=AGREE_NEG,
        disagree_weight=1.0, deadzone=DEADZONE,
    )
    st = SignWeightStats(task_names=["alfworld", "search", "webshop"])
    st.update_candidates(
        state=state, on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
        response_mask=torch.ones(1, 1), deadzone=DEADZONE,
        task_ids=torch.tensor([0]),
        off_plane_tasks=torch.tensor([[1, 2]]),
    )
    m = st.metrics()
    assert m["sign_weight/abs_delta_mean/alfworld"] == pytest.approx(1.0)
    assert m["sign_weight/deadzone_frac/alfworld"] == pytest.approx(0.5)
    assert m["sign_weight/abs_delta_mean/search"] == pytest.approx(0.5)
    assert m["sign_weight/deadzone_frac/search"] == pytest.approx(0.0)


def test_target_diagnostics_are_zero_when_nothing_is_flagged():
    on = torch.log(torch.tensor([[[0.4, 0.2]]]))
    st = SignWeightStats()
    st.update_target(
        on_task_logprob=on, candidate_weight=torch.ones(1, 1, 2), response_mask=torch.ones(1, 1)
    )
    m = st.metrics()
    assert m["sign_weight/target_kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["sign_weight/target_entropy_delta"] == pytest.approx(0.0, abs=1e-6)
    assert m["sign_weight/target_tv"] == pytest.approx(0.0, abs=1e-6)
    assert m["sign_weight/inv_z"] == pytest.approx(1.0, abs=1e-6)


def test_target_kl_closed_form_matches_the_explicit_divergence():
    """``log Z - sum p log w`` is the same number as rebuilding p~ and measuring."""
    on = torch.log(torch.tensor([[[0.4, 0.2]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    st = SignWeightStats()
    st.update_target(on_task_logprob=on, candidate_weight=w, response_mask=torch.ones(1, 1))
    got = st.metrics()["sign_weight/target_kl"]

    q = reweight_teacher_logprobs(on, w)
    p, qq = on.exp(), q.exp()
    p_tail, q_tail = 1.0 - p.sum(-1), 1.0 - qq.sum(-1)
    want = (p * (on - q)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())
    assert got == pytest.approx(want.item(), abs=1e-6)


def test_inv_z_shows_the_tail_shrinking_when_a_likely_token_is_raised():
    """Agreement to raise sits on high-probability candidates and agreement to
    lower on low-probability ones, so Z > 1 and the tail loses share. This is the
    systematic sharpening a sign-shuffle control cannot rule out."""
    on = torch.log(torch.tensor([[[0.5, 0.02]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    st = SignWeightStats()
    st.update_target(on_task_logprob=on, candidate_weight=w, response_mask=torch.ones(1, 1))
    m = st.metrics()
    assert m["sign_weight/inv_z"] < 1.0
    assert m["sign_weight/target_entropy_delta"] < 0.0


def test_target_kl_ratio_is_reported_against_the_loss_it_sits_inside():
    on = torch.log(torch.tensor([[[0.4, 0.2]]]))
    w = torch.tensor([[[1.25, 0.75]]])
    st = SignWeightStats()
    st.update_target(
        on_task_logprob=on, candidate_weight=w, response_mask=torch.ones(1, 1),
        teacher_kl=torch.tensor([[0.5]]),
    )
    m = st.metrics()
    assert m["sign_weight/target_kl_ratio"] == pytest.approx(m["sign_weight/target_kl"] / 0.5)


def test_position_pre_normalisation_mean_is_reported_per_task():
    """The number that says whether the arm is just a bigger teacher_kl_loss_coef."""
    st = SignWeightStats(task_names=["alfworld", "search"])
    st.update_position(
        position_weight=torch.tensor([[1.2, 1.2], [1.0, 1.0]]),
        response_mask=torch.ones(2, 2),
        task_ids=torch.tensor([0, 1]),
    )
    m = st.metrics()
    assert m["sign_weight/alfworld/w_mean_pre_norm"] == pytest.approx(1.2)
    assert m["sign_weight/search/w_mean_pre_norm"] == pytest.approx(1.0)


def test_teacher_coverage_is_the_mean_covered_mass():
    """Sum of the teacher's probability over the support, averaged per token.

    The ceiling on target mode's leverage: the mass_frac_* shares are fractions
    OF this, so a run where the support covers little of the teacher would look
    identical to one where it covers all of it without this number beside them.
    """
    on = torch.log(torch.tensor([[[0.5, 0.3]], [[0.15, 0.05]]]))  # covers 0.8 / 0.2
    base = torch.full((2, 1, 2), -2.0)
    off = torch.stack([base + 1.0, base + 1.0], dim=-1)
    _, state = candidate_weights(
        on, off, base, mode="target", agree_weight=AGREE, agree_neg_weight=AGREE_NEG,
        disagree_weight=1.0, deadzone=DEADZONE,
    )
    st = SignWeightStats()
    st.update_candidates(
        state=state, on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
        response_mask=torch.ones(2, 1), deadzone=DEADZONE,
    )
    assert st.metrics()["sign_weight/teacher_coverage"] == pytest.approx(0.5, abs=1e-5)
