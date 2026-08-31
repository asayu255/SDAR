"""The separation the source channel was rebuilt for, as numbers.

The arm's own measurement is what this file exists to keep true. On the shipped
run, with the source gated by a per-pair correlation against the row's
advantage, **53.3% of the source channel's mass sat in the ``agree`` state** --
candidates where the on-task teacher and that source moved the same way, which
is the corroboration channel's territory. Only 13.3% was in
``on_silent_source_active``, the state the channel was designed for. The channel
was, by mass, a second copy of the one beside it.

Two changes fix that and both are tested here rather than argued:

``relu(|hat_m| - |hat_on|)``  the source only ever gets what the on-task teacher
                             does not already say, so a source at the on-task
                             teacher's own volume adds exactly zero however
                             loudly it spoke;
``teacher_similarity``       the reliability is computed at the candidate from
                             the teachers themselves, so it carries no state,
                             needs no warm-up, and cannot be zero for a whole
                             destination the way a correlation against a
                             degenerate advantage was.

The metrics that make both readable in a live run are asserted too. They are the
only way the 53.3% gets re-measured on the next run instead of assumed away.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        CHANNEL_PROBES,
        GATE_BINS,
        POSITION_TERMS,
        build_position_weight,
        compute_raw_policy_shifts,
        decompose_common_residual,
        decorrelated_off_shifts,
        position_terms,
        position_weight_metrics,
        source_exclusive_shift,
        teacher_similarity,
    )
    from verl.trainer.ppo.sign_weights import ScopeTermStats
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ("alfworld", "search", "webshop")


def _lp(*shape, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape), dim=-1)


def _student_like(teacher_logprob):
    """A student plane for :func:`build_position_weight`, distinct from the teacher.

    The measure the weight aggregates against is the STUDENT's mass, not the
    teacher's, so a fixture that passed the teacher twice would make the two
    indistinguishable and every test here would pass under the bug the measure
    was changed to fix. Derived deterministically from the teacher and NOT equal
    to it: a fixed reversal of the support, which keeps it a valid log-softmax
    over the same candidates while putting its mass somewhere else.
    """
    return torch.log_softmax(teacher_logprob.detach().flip(-1) * 1.3, dim=-1)


def _build(on, off, base, *, bs):
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    return build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on),
        response_mask=torch.ones(on.shape[0], on.shape[1]),
        task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
    ), task_ids


def _random(bs=6, resp=5, k=7, seed=11):
    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(2)], dim=-1)
    return _build(on, off, base, bs=bs)


def _metrics(got, task_ids, n_tasks=3):
    kl = torch.rand(*got["weight"].shape) + 0.1
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=n_tasks, device="cpu")
    stats.update(
        position_terms(got, kl),
        response_mask=torch.ones_like(kl),
        task_ids=task_ids,
    )
    return position_weight_metrics(stats.sums(task_names=TASKS))


# --------------------------------------------------------------------------- #
# the redundancy the split removes
# --------------------------------------------------------------------------- #
def test_a_source_inside_the_on_task_teachers_shadow_earns_nothing():
    """The ``agree`` state, in its purest form. Both sources move the candidate
    the on-task teacher's way and by less than it does; under the old rule this
    was the state holding a majority of the source channel's mass, and here it
    is exactly zero for the source and paid once by the corroboration."""
    hat_on = torch.tensor([[[2.0]]])
    hat_off = torch.tensor([[[[1.5, 1.0]]]])
    assert float(source_exclusive_shift(hat_on=hat_on, hat_off=hat_off).sum()) == 0.0
    dec = decompose_common_residual(hat_on=hat_on, hat_off=hat_off)
    # 1.25, not 2.0: each source corroborates with what it actually said, and
    # they said 1.5 and 1.0. The ceiling is |hat_on| and a teacher that only got
    # most of the way there does not reach it.
    assert float(dec["common_soft"]) == pytest.approx(1.25), "and the agreement IS paid"
    assert float(
        decompose_common_residual(hat_on=hat_on, hat_off=torch.tensor([[[[2.5, 3.0]]]]))["common_soft"]
    ) == pytest.approx(2.0), "the ceiling is reached only when both sources clear it"


def test_the_source_is_paid_in_full_where_the_on_task_teacher_is_silent():
    """``on_silent_source_active``: the ceiling is zero, so the whole shift is
    excess. The state the channel was designed for is the state it now keeps."""
    hat_on = torch.tensor([[[0.0]]])
    hat_off = torch.tensor([[[[3.0, 2.0]]]])
    excl = source_exclusive_shift(hat_on=hat_on, hat_off=hat_off)
    assert excl.reshape(-1).tolist() == pytest.approx([3.0, 2.0])
    assert float(decompose_common_residual(hat_on=hat_on, hat_off=hat_off)["common_soft"]) == 0.0


def test_the_source_mass_moves_out_of_the_agree_state():
    """The claim as a comparison, on one batch: what the ungated full shift
    would have charged inside the on-task teacher's shadow, against what the
    exclusive shift charges there. The first is the 53.3% reading."""
    torch.manual_seed(3)
    hat_on = torch.randn(4, 6, 8).abs()
    hat_off = hat_on.unsqueeze(-1) * torch.tensor([0.6, 0.9]) + 0.05
    full = hat_off.abs().sum()
    excess = source_exclusive_shift(hat_on=hat_on, hat_off=hat_off).sum()
    assert float(full) > 0.0
    assert float(excess) / float(full) < 0.05, "almost all of it was the on-task teacher's"


# --------------------------------------------------------------------------- #
# the gate carries no state
# --------------------------------------------------------------------------- #
def test_the_gate_is_a_function_of_this_candidate_and_nothing_else():
    """What the advantage correlation could not be. It was accumulated over the
    run, was still within noise of zero after 130 steps at every pair, and was
    identically zero for two of the six across the whole control window; a run
    could not test a channel that spent its only measured window warming up."""
    off = torch.randn(3, 4, 5, 2)
    first = teacher_similarity(off)
    for _ in range(3):
        assert torch.equal(teacher_similarity(off), first), "no accumulator, no drift"
    half = teacher_similarity(off[:1])
    assert torch.allclose(half, first[:1]), "and no dependence on the batch it arrived in"


def test_the_gate_is_live_on_the_first_step_at_every_destination():
    got, _ = _random()
    assert float(got["q_sim"].max()) > 0.0
    assert float((got["evidence"] - got["common_soft"].abs()).abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# the partition still holds exactly
# --------------------------------------------------------------------------- #
def test_the_two_channels_add_to_the_pre_weight_exactly():
    got, _ = _random()
    total = got["evidence_shared"] + got["evidence_by_source"].sum(dim=(-1, -2))
    assert torch.allclose(total, got["pre_weight"] - 1.0, atol=1e-5)


def test_w_minus_one_still_splits_three_ways():
    got, _ = _random()
    total = got["push_shared"] + got["push_by_source"].sum(dim=-1) + got["push_normalizer"]
    assert torch.allclose(total, got["weight"] - 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# the metrics that re-measure all of the above on a live run
# --------------------------------------------------------------------------- #
def test_the_two_gates_are_reported_as_the_share_each_let_through():
    """A chain, not one number: a low ``exclusive_pass_rate`` means the sources
    are redundant with the on-task teacher, a low ``gate_pass_rate`` means they
    disagree with each other, and the fix for those is not the same fix."""
    got, task_ids = _random()
    m = _metrics(got, task_ids)
    for key in ("exclusive_pass_rate", "gate_pass_rate", "gate_mean"):
        full = f"kl_weight/evidence/{key}"
        assert full in m, full
        # float64 sums of float32 terms: a share can land a hair outside.
        assert -1e-6 <= m[full] <= 1.0 + 1e-6, (full, m[full])
    for task in TASKS:
        assert f"kl_weight/{task}/evidence/exclusive_pass_rate" in m


def test_the_pass_rates_read_the_way_the_construction_says():
    """Sources inside the shadow: nothing survives the ceiling. Sources that
    oppose each other: the excess survives the ceiling and the gate takes it."""
    bs, resp, k = 4, 3, 5
    torch.manual_seed(7)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)

    shadowed, ids = _build(on, torch.stack([on.clone(), on.clone()], dim=-1), base, bs=bs)
    m = _metrics(shadowed, ids)
    assert m["kl_weight/evidence/exclusive_pass_rate"] == pytest.approx(0.0, abs=1e-6)
    assert m["kl_weight/evidence/gate_mean"] > 0.0, "they agreed; there was just nothing to add"

    d = on - base
    opposed, ids = _build(on, torch.stack([base + 4.0 * d, base - 4.0 * d], dim=-1), base, bs=bs)
    m = _metrics(opposed, ids)
    assert m["kl_weight/evidence/exclusive_pass_rate"] > 0.3, "there was excess"
    assert m["kl_weight/evidence/gate_pass_rate"] == pytest.approx(0.0, abs=1e-6)
    assert m["kl_weight/evidence/gate_mean"] == pytest.approx(0.0, abs=1e-6)


def test_a_counterfactual_scope_publishes_no_gate_reading_rather_than_a_zero():
    """A channel scope is handed a pre-weight and an evidence tensor and no gate
    masses at all. A 0.0 there would read as "both gates closed on this
    channel", which is a finding, and it would be a fabricated one."""
    got, task_ids = _random()
    kl = torch.rand(*got["weight"].shape) + 0.1
    chan = {
        "weight": got["channel_pre_weight"]["source_only"],
        "pre_weight": got["channel_pre_weight"]["source_only"],
        "evidence": got["channel_evidence"]["source_only"],
        "state": got["state"], "teacher_prob": got["teacher_prob"],
        "available": got["available"],
        "evidence_shared": torch.zeros_like(got["evidence_shared"]),
        "evidence_shared_offtask_only": torch.zeros_like(got["evidence_shared"]),
    }
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=3, device="cpu")
    stats.update(
        position_terms(chan, kl),
        response_mask=torch.ones_like(kl), task_ids=task_ids,
    )
    m = position_weight_metrics(stats.sums(task_names=TASKS), prefix="kl_weight/channel/source_only")
    assert "kl_weight/channel/source_only/position/w_mean" in m, "the scope did render"
    for key in ("exclusive_pass_rate", "gate_pass_rate", "gate_mean"):
        assert f"kl_weight/channel/source_only/evidence/{key}" not in m


# --------------------------------------------------------------------------- #
# the placebo
# --------------------------------------------------------------------------- #
def test_the_shuffled_gate_is_built_and_is_never_the_shipped_one():
    """The independence the gate assumes is not free: every teacher here is the
    same base model after RL on a different task with the same recipe, so two of
    them agreeing can be one shared generation grammar answering twice. The
    shuffled channel is the null for that, and it has to exist in the run before
    a structural-token mask can be derived from it rather than written by hand."""
    got, _ = _random()
    assert "shuffled_gate" in CHANNEL_PROBES
    assert "shuffled_gate" in got["channel_pre_weight"]
    assert "ungated_source" in got["channel_pre_weight"]
    live = got["pre_weight"]
    assert not torch.allclose(got["channel_pre_weight"]["shuffled_gate"], live, atol=1e-6)
    assert torch.all(got["channel_pre_weight"]["ungated_source"] >= live - 1e-6)


def test_the_shuffle_cannot_reach_the_weight():
    """A diagnostic, and rolled across padded positions, so the one thing that
    must stay true is that the shipped evidence never sees it."""
    torch.manual_seed(5)
    off = torch.randn(2, 6, 4, 2)
    rolled = decorrelated_off_shifts(off)
    assert not torch.equal(rolled, off)
    got, _ = _random()
    from verl.trainer.ppo.cross_teacher_kl_weight import candidate_kl_evidence

    rebuilt = candidate_kl_evidence(
        common=got["common_soft"],
        source_gate=got["q_sim"],
        exclusive=got["source_exclusive"],
    )
    assert torch.allclose(rebuilt, got["evidence"], atol=1e-6), (
        "the shipped evidence is the live gate, reproducible from the returned parts"
    )


# --------------------------------------------------------------------------- #
# the corrections a review caught before the run
# --------------------------------------------------------------------------- #
def test_the_corroboration_is_continuous_in_the_off_task_volume():
    """The defect this rule was rewritten to remove, pinned as a counter-example.

    The first draft graded the on-task shift by a RATIO, ``f = sum_m
    relu(sign(hat_on) hat_m) / sum_m |hat_m|``. A ratio is scale-free in its own
    arguments, so two off-task teachers that between them moved 0.02 licensed
    the whole of an on-task shift of 10 -- and every claim in the module about
    near-zero shifts self-attenuating and about needing no deadzone was false
    under it. Scaling the sources must scale the corroboration.
    """
    on = torch.tensor([[[10.0]]])
    def c(scale):
        return float(decompose_common_residual(
            hat_on=on, hat_off=torch.tensor([[[[1.0, 1.0]]]]) * scale
        )["common_soft"])

    assert c(1.0) == pytest.approx(1.0)
    for scale in (1e-1, 1e-2, 1e-3):
        assert c(scale) == pytest.approx(scale, rel=1e-4), "no floor, no deadzone"
    ratio = 2 * 0.01 / (2 * 0.01)
    assert ratio == pytest.approx(1.0), "which is what the ratio form would have scored"
    assert c(0.0) == 0.0


def test_the_corroboration_still_reaches_the_ceiling_and_never_passes_it():
    on = torch.rand(3, 4, 5) * 4.0 - 2.0
    for off in (on.unsqueeze(-1) * torch.tensor([9.0, 9.0]), torch.randn(3, 4, 5, 2)):
        c = decompose_common_residual(hat_on=on, hat_off=off)["common_soft"]
        assert torch.all(c.abs() <= on.abs() + 1e-6)
        assert torch.all(torch.sign(c) * torch.sign(on) >= 0), "and keeps the on-task sign"


def test_one_loud_source_cannot_hide_anothers_silence():
    """A mean over teachers, not a max and not a sum. The loud one gets its vote
    and the silent one gets its zero, and the candidate is credited with the
    average of the two rather than with the louder of them."""
    loud_and_silent = float(decompose_common_residual(
        hat_on=torch.tensor([[[2.0]]]), hat_off=torch.tensor([[[[9.0, 0.0]]]])
    )["common_soft"])
    both_loud = float(decompose_common_residual(
        hat_on=torch.tensor([[[2.0]]]), hat_off=torch.tensor([[[[9.0, 9.0]]]])
    )["common_soft"])
    assert loud_and_silent == pytest.approx(1.0)
    assert both_loud == pytest.approx(2.0)


def test_the_per_source_votes_add_to_the_applied_corroboration():
    """Unlike a minimum over a unanimity, the applied rule HAS a per-source
    decomposition -- which is what the attribution table is allowed to report."""
    on = torch.randn(2, 3, 4)
    off = torch.randn(2, 3, 4, 2)
    dec = decompose_common_residual(hat_on=on, hat_off=off)
    assert torch.allclose(
        dec["common_soft_by_source"].sum(dim=-1), dec["common_soft"], atol=1e-6
    )


def test_the_weight_aggregates_against_the_student_and_not_the_teacher():
    """The measure, as a difference rather than a docstring. Two builds that
    differ only in which distribution the candidate expectation is taken against
    must not produce the same weight -- if they did, the reverse KL's own
    measure would be unobservable and the correction unverifiable."""
    torch.manual_seed(13)
    bs, resp, k = 3, 4, 6
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(2)], dim=-1)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)

    def build(student):
        return build_position_weight(
            shifts=compute_raw_policy_shifts(
                on_task_logprob=on, off_task_logprobs=off, base_logprob=base
            ),
            on_task_logprob=on, student_logprob=student,
            response_mask=torch.ones(bs, resp),
            task_ids=task_ids, off_plane_tasks=planes,
            diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
            diag_valid=torch.ones(3, dtype=torch.bool),
            normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
        )

    student = _lp(bs, resp, k)
    assert not torch.allclose(build(student)["pre_weight"], build(on)["pre_weight"], atol=1e-4)
    assert torch.allclose(build(student)["mass"], student.exp(), atol=1e-6)
    assert not build(student)["mass"].requires_grad, "a coefficient, never a gradient path"


def test_the_shuffle_permutes_inside_each_rows_own_response():
    """Rows have different lengths. A roll over the padded axis moves padding
    into live positions, and the placebo's gap from the live gate would then
    carry response length as well as the teacher correspondence it is for."""
    off = torch.arange(24, dtype=torch.float32).reshape(2, 6, 1, 2)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0] * 6])
    rolled = decorrelated_off_shifts(off, response_mask=mask)
    assert torch.equal(rolled[0, 4:], off[0, 4:]), "padding is left where it is"
    for row, length in ((0, 4), (1, 6)):
        for m in range(2):
            assert sorted(rolled[row, :length, 0, m].tolist()) == sorted(
                off[row, :length, 0, m].tolist()
            ), "and the live prefix is a permutation of itself"
    assert not torch.equal(rolled[0, :4], off[0, :4])


def test_the_channel_is_omitted_when_it_cannot_be_built_honestly():
    """No mask, no placebo. A shuffled gate that also moved padding would
    manufacture the finding it exists to test for."""
    torch.manual_seed(2)
    bs, resp, k = 2, 4, 5
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(2)], dim=-1)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    kw = dict(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on),
        task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
        diag_valid=torch.ones(3, dtype=torch.bool),
    )
    assert "shuffled_gate" not in build_position_weight(**kw)["channel_pre_weight"]
    with_mask = build_position_weight(response_mask=torch.ones(bs, resp), **kw)
    assert "shuffled_gate" in with_mask["channel_pre_weight"]


def test_the_gates_two_teacher_reading_does_not_survive_a_third():
    """Documented rather than fixed, and pinned so it cannot be discovered as a
    surprise. At n_off == 2 -- what the multitask arms run -- ``q`` is what the
    module says it is; past that, three of its claims weaken."""
    assert float(teacher_similarity(torch.tensor([[[[3.0, 0.0]]]]))) == 0.0
    assert float(teacher_similarity(torch.tensor([[[[3.0, 3.0, 0.0]]]]))) > 0.0, (
        "one silent teacher no longer shuts the gate"
    )
    assert float(teacher_similarity(torch.tensor([[[[3.0, 3.0, -1.0]]]]))) > 0.0, (
        "and a dissenter rides through on the majority: q is one number for all"
    )
    twice = torch.tensor([[[[3.0, 3.0]]]])
    once = torch.tensor([[[[3.0]]]])
    on = torch.tensor([[[1.0]]])
    q2 = teacher_similarity(twice)
    dup = float(q2 * source_exclusive_shift(hat_on=on, hat_off=twice).sum(dim=-1))
    solo = float(source_exclusive_shift(hat_on=on, hat_off=once).sum(dim=-1))
    assert dup == pytest.approx(2 * solo), "duplicating a teacher doubles the source"


# --------------------------------------------------------------------------- #
# what each channel did to the LOSS, which no weight share answers
# --------------------------------------------------------------------------- #
def _built_with_kl(bs=4, resp=5, k=6, seed=31):
    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(2)], dim=-1)
    got, ids = _build(on, off, base, bs=bs)
    kl = torch.rand(bs, resp) + 0.1
    base_kl = torch.rand(bs, resp) + 0.1
    return got, ids, kl, base_kl


def _fold(got, ids, kl, *, base_kl=None, row_weight=None, coef=1.0):
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=3, device="cpu")
    stats.update(
        position_terms(got, kl, row_weight=row_weight, coef=coef, base_kl=base_kl),
        response_mask=torch.ones_like(kl), task_ids=ids,
    )
    return position_weight_metrics(stats.sums(task_names=TASKS))


def test_the_three_channels_add_to_what_the_arm_did_to_the_kl():
    """The reading an arm choice rests on and the one a weight share cannot
    give: of the nats this scope's distillation moved, how many were the
    corroboration, how many the source, and how many the normaliser taking
    budget off the positions with no evidence."""
    got, ids, kl, base_kl = _built_with_kl()
    m = _fold(got, ids, kl, base_kl=base_kl)
    for c in ("shared", "source", "normalizer"):
        assert f"kl_weight/channel/{c}/kl_added_raw" in m
        assert f"kl_weight/channel/{c}/objective_contribution" in m
        assert 0.0 <= m[f"kl_weight/channel/{c}/added_abs_share"] <= 1.0
    shares = sum(m[f"kl_weight/channel/{c}/added_abs_share"] for c in
                 ("shared", "source", "normalizer"))
    assert shares == pytest.approx(1.0, abs=1e-6)
    # float32 columns folded into float64 sums: the residual is float noise on
    # the gross, not a missing term. Anything above this is a channel being
    # dropped or double counted.
    assert abs(m["kl_weight/channel/decomposition_residual"]) < 1e-6, (
        "(W - 1) D, formed independently, is the three parts"
    )


def test_only_the_normalizer_can_take_budget_away():
    got, ids, kl, _ = _built_with_kl()
    m = _fold(got, ids, kl)
    assert m["kl_weight/channel/shared/kl_added_raw"] >= 0.0
    assert m["kl_weight/channel/source/kl_added_raw"] >= 0.0
    assert m["kl_weight/channel/normalizer/kl_added_raw"] < 0.0, (
        "mu > 1 on this fixture, so the normaliser is where the budget comes from"
    )


def test_the_objective_columns_carry_the_task_share_and_the_coefficient():
    """Raw nats are not comparable across tasks that enter the loss with
    different weights, and are two orders of magnitude away from what the
    distillation term actually contributes."""
    got, ids, kl, _ = _built_with_kl()
    plain = _fold(got, ids, kl)
    scaled = _fold(got, ids, kl, row_weight=torch.full((4,), 2.0), coef=0.01)
    for c in ("shared", "source", "normalizer"):
        assert scaled[f"kl_weight/channel/{c}/kl_added_raw"] == pytest.approx(
            plain[f"kl_weight/channel/{c}/kl_added_raw"]
        ), "the nats do not move"
        assert scaled[f"kl_weight/channel/{c}/objective_contribution"] == pytest.approx(
            0.02 * plain[f"kl_weight/channel/{c}/kl_added_raw"], rel=1e-5
        )


# --------------------------------------------------------------------------- #
# does the on-task teacher say anything the base does not, where the arm spends
# --------------------------------------------------------------------------- #
def test_the_novelty_columns_answer_the_base_pull_question():
    """``W`` multiplies KL(student || on-task teacher), and ``hat_on`` is that
    teacher measured against the BASE, so the source channel's own region is by
    construction where the two are the same distribution. Raising W there is a
    pull toward the base whatever the evidence said -- and whether that is what
    happens is a number, not an argument."""
    got, ids, kl, base_kl = _built_with_kl()
    m = _fold(got, ids, kl, base_kl=base_kl)
    for key in ("teacher_novelty", "source_weighted_novelty", "shared_weighted_novelty"):
        assert f"kl_weight/base/{key}" in m
    assert m["kl_weight/base/kl_base_mean"] > 0.0

    # A teacher that IS the base: novelty is exactly zero, everywhere and under
    # both channels' weighting. That is the reading the run is looking for.
    same = _fold(got, ids, kl, base_kl=kl)
    assert same["kl_weight/base/teacher_novelty"] == pytest.approx(0.0, abs=1e-9)
    assert same["kl_weight/base/source_weighted_novelty"] == pytest.approx(0.0, abs=1e-9)
    assert same["kl_weight/base/shared_weighted_novelty"] == pytest.approx(0.0, abs=1e-9)


def test_the_novelty_columns_are_absent_rather_than_zero_without_a_base():
    got, ids, kl, _ = _built_with_kl()
    m = _fold(got, ids, kl)
    assert not any("/base/" in k for k in m), "not measured is not novelty zero"


# --------------------------------------------------------------------------- #
# the gate's shape, and the placebo as a ratio
# --------------------------------------------------------------------------- #
def test_the_gate_bins_separate_a_selector_from_an_attenuator():
    """``gate_mean = 0.1`` is two mechanisms -- every candidate gated to a
    tenth, or a tenth of them gated fully -- and only the second is what a gate
    is for."""
    got, ids, kl, _ = _built_with_kl()
    m = _fold(got, ids, kl)
    fracs = [m[f"kl_weight/gate/q_mass_frac_gt_{int(round(t * 100)):03d}"] for t in GATE_BINS]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert fracs == sorted(fracs, reverse=True), "the bins nest"


def test_the_placebo_is_published_as_a_ratio_against_the_live_gate():
    got, ids, kl, _ = _built_with_kl()
    m = _fold(got, ids, kl)
    for key in ("shuffled_to_live_gate_ratio", "shuffled_to_live_source_evidence_ratio"):
        assert f"kl_weight/gate/{key}" in m
        assert m[f"kl_weight/gate/{key}"] >= 0.0


def test_without_a_mask_the_placebo_ratio_reads_one_and_not_a_finding():
    """No mask, no honest shuffle. The ratio is then exactly 1 -- "not measured"
    -- rather than a number that would read as "the shuffle changed nothing"."""
    torch.manual_seed(17)
    bs, resp, k = 3, 4, 5
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(2)], dim=-1)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    got = build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on),
        task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
    )
    m = _fold(got, task_ids, torch.rand(bs, resp) + 0.1)
    assert m["kl_weight/gate/shuffled_to_live_gate_ratio"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the counterfactuals are a symmetric set
# --------------------------------------------------------------------------- #
def test_shared_only_and_source_only_are_the_same_kind_of_object():
    """They used to live in two namespaces -- shared-only was the bottom of the
    alpha probe series and source-only was a channel -- so the two arms of one
    mechanism were compared through two different sets of published columns."""
    got, _ = _random()
    for name in ("shared_only", "source_only", "unweighted"):
        assert name in got["channel_pre_weight"], name
    shared = got["channel_evidence"]["shared_only"]
    source = got["channel_evidence"]["source_only"]
    assert torch.allclose(shared + source, got["evidence"], atol=1e-6), (
        "and together they are the whole evidence, with nothing double counted"
    )
    assert torch.count_nonzero(got["channel_evidence"]["unweighted"]) == 0
    assert torch.allclose(
        got["channel_pre_weight"]["unweighted"], torch.ones_like(got["pre_weight"])
    )
