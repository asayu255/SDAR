"""Parameter-free cross-teacher weighting of the on-task OPD KL.

The arm produces one number per token position, ``W``, multiplies the per-token
teacher KL by it, and changes nothing else. Everything below is about the two
places where that number could quietly mean something other than what the design
says, plus the identities that let it be read as a quantity at all.

* **Scale.** ``delta`` is in nats and the three teachers were trained at KL
  coefficients differing 10x, so raw magnitudes are not on a common footing. The
  divisor is each teacher's OWN in-domain RMS -- the DIAGONAL of the ordered-pair
  matrix -- and the tests pin that it is the diagonal, because dividing by the
  destination-conditioned RMS instead would stretch a teacher's out-of-domain
  noise up to a full unit and silently undo the reason the deadzone was dropped.

* **Monotonicity.** Corroboration must never score lower than conflict. The
  obvious formula, ``|c| + sum alpha|delta_hat - c|``, does exactly that once
  ``alpha`` passes ``1/n_off``; the shipped one does not, for every ``alpha`` in
  [0, 1], and that is asserted rather than argued.

Plus the invariant the normaliser exists for: on the snapshot it was built from,
``sum W*D / sum D`` is 1 -- not ``mean(W)``, which is a different number whenever
the weight and the KL are correlated, i.e. exactly when the arm is doing
something.
"""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        ADV_MOMENTS,
        AdvantageReliabilityStats,
        CumulativePolicyShiftRMS,
        PreviousStepTaskKLWeightedMean,
        candidate_kl_evidence,
        candidate_mass,
        compute_raw_policy_shifts,
        decompose_common_residual,
        decorrelated_off_shifts,
        group_center,
        position_pre_weight,
        source_exclusive_shift,
        standardize_policy_shifts,
        tail_logprob,
        teacher_similarity,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ["alfworld", "search", "webshop"]


def _lp(bs, resp, k, scale=1.0, seed=None):
    """A full-vocab log-softmax gathered at k ids, so the support's mass is < 1."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.log_softmax(scale * torch.randn(bs, resp, k + 6), dim=-1)[..., :k]


# --------------------------------------------------------------------------- #
# raw shifts
# --------------------------------------------------------------------------- #
def test_the_shift_is_the_log_ratio_against_the_shared_base():
    on, base = _lp(2, 3, 4, seed=0), _lp(2, 3, 4, seed=1)
    off = torch.stack([_lp(2, 3, 4, seed=2), _lp(2, 3, 4, seed=3)], dim=-1)
    s = compute_raw_policy_shifts(on_task_logprob=on, off_task_logprobs=off, base_logprob=base)
    assert torch.allclose(s["on"], on - base)
    assert torch.allclose(s["off"][..., 0], off[..., 0] - base)


def test_the_tail_is_the_supports_complement_and_eps_is_only_a_floor():
    """The leftover is real probability mass, not a rounding error: it is what
    makes every expectation here run over something that sums to 1."""
    lp = torch.log(torch.tensor([[[0.5, 0.25]]]))
    assert float(tail_logprob(lp).exp()) == pytest.approx(0.25, abs=1e-6)
    # A support that covers everything: the clamp binds and the result is finite
    # rather than -inf. Numerical safety only, never a transfer-strength knob.
    full = torch.log(torch.tensor([[[0.5, 0.5]]]))
    assert torch.isfinite(tail_logprob(full)).all()
    assert float(tail_logprob(full).exp()) < 1e-6


def test_the_tail_shift_compares_the_same_supports_complement():
    """Not a per-candidate value: the lumped mass outside the support, on both
    sides of the ratio, so it can sit in the same expectation as the rest."""
    on, base = _lp(1, 1, 4, seed=4), _lp(1, 1, 4, seed=5)
    off = _lp(1, 1, 4, seed=6).unsqueeze(-1)
    s = compute_raw_policy_shifts(on_task_logprob=on, off_task_logprobs=off, base_logprob=base)
    expect = math.log(1 - float(on.exp().sum())) - math.log(1 - float(base.exp().sum()))
    assert float(s["tail_on"]) == pytest.approx(expect, abs=1e-5)


def test_a_teacher_identical_to_the_base_shifts_nothing():
    base = _lp(2, 2, 3, seed=7)
    s = compute_raw_policy_shifts(
        on_task_logprob=base, off_task_logprobs=base.unsqueeze(-1), base_logprob=base
    )
    for key in ("on", "off", "tail_on", "tail_off"):
        assert float(s[key].abs().max()) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the RMS matrix
# --------------------------------------------------------------------------- #
def _rms_batch(*, bs, resp, k, n_off, seed=0):
    torch.manual_seed(seed)
    return {
        "student": _lp(bs, resp, k),
        "on": _lp(bs, resp, k),
        "off": torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1),
        "base": _lp(bs, resp, k),
    }


def _fold_rms(stats, b, task_ids, off_plane_tasks, mask=None):
    bs, resp = b["on"].shape[:2]
    stats.update(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=b["on"], off_task_logprobs=b["off"], base_logprob=b["base"]
        ),
        student_logprob=b["student"],
        response_mask=torch.ones(bs, resp) if mask is None else mask,
        task_ids=task_ids,
        off_plane_tasks=off_plane_tasks,
    )
    # The step boundary. update() writes a rank-local delta; all_reduce() is what
    # folds it into the cumulative total, exactly as update_policy does.
    stats.all_reduce()
    return stats


def test_the_diagonal_is_the_teacher_measured_where_it_operates():
    """THE test for the scale choice.

    Teacher 1 is loud on its own task's states and quiet on task 0's. The
    diagonal must report the loud number and the off-diagonal the quiet one; a
    destination-conditioned divisor would use the quiet one and stretch that
    teacher's out-of-domain noise back up to a full unit.
    """
    torch.manual_seed(11)
    bs, resp, k = 4, 5, 3
    base = _lp(bs, resp, k, seed=12)
    student = _lp(bs, resp, k, seed=13)
    stats = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")

    def rows(dst, on_gain, off_gain, plane_task):
        shifts = {
            "on": torch.full((bs, resp, k), on_gain),
            "off": torch.full((bs, resp, k, 1), off_gain),
            "tail_on": torch.full((bs, resp), on_gain),
            "tail_off": torch.full((bs, resp, 1), off_gain),
        }
        stats.update(
            shifts=shifts, student_logprob=student, response_mask=torch.ones(bs, resp),
            task_ids=torch.full((bs,), dst), off_plane_tasks=torch.full((bs, 1), plane_task),
        )

    # task 1's own rows: teacher 1 is the on-task one and moves by 2.0
    rows(dst=1, on_gain=2.0, off_gain=0.0, plane_task=2)
    # task 0's rows: teacher 1 is off-task here and moves by 0.5
    rows(dst=0, on_gain=1.0, off_gain=0.5, plane_task=1)
    stats.all_reduce()

    diag, valid = stats.diagonal()
    snap = stats.snapshot()
    assert bool(valid[1]) and float(diag[1]) == pytest.approx(2.0, abs=1e-6)
    assert float(snap["sigma"][0, 1]) == pytest.approx(0.5, abs=1e-6)
    # The ratio IS the domain-applicability reading, and it survives.
    assert float(snap["sigma"][0, 1] / diag[1]) == pytest.approx(0.25, abs=1e-6)
    del base


def test_the_rms_is_a_student_weighted_expectation_not_a_slot_average():
    """Twenty candidates the student has ruled out must not outvote the one it
    is about to emit."""
    bs, resp, k = 1, 1, 3
    student = torch.log(torch.tensor([[[0.90, 0.005, 0.005]]]))
    shifts = {
        "on": torch.tensor([[[1.0, 10.0, 10.0]]]),
        "off": torch.zeros(bs, resp, k, 1),
        "tail_on": torch.zeros(bs, resp),
        "tail_off": torch.zeros(bs, resp, 1),
    }
    stats = CumulativePolicyShiftRMS(n_tasks=1, device="cpu")
    stats.update(
        shifts=shifts, student_logprob=student, response_mask=torch.ones(bs, resp),
        task_ids=torch.zeros(bs, dtype=torch.long), off_plane_tasks=torch.zeros(bs, 1, dtype=torch.long),
    )
    stats.all_reduce()
    diag, _ = stats.diagonal()
    # slot mean would be sqrt((1+100+100)/3) = 8.19; the student's measure gives
    # 0.9*1 + 0.005*100 + 0.005*100 = 1.9 (the 0.09 tail carries shift 0).
    assert float(diag[0]) == pytest.approx(math.sqrt(1.9), abs=1e-4)


def test_the_rms_does_not_depend_on_the_micro_batch_split():
    b = _rms_batch(bs=4, resp=3, k=3, n_off=2, seed=14)
    ids = torch.tensor([0, 1, 2, 0])
    planes = torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])
    whole = _fold_rms(CumulativePolicyShiftRMS(n_tasks=3, device="cpu"), b, ids, planes)
    split = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    for r in range(4):
        _fold_rms(
            split,
            {key: v[r : r + 1] for key, v in b.items()},
            ids[r : r + 1],
            planes[r : r + 1],
        )
    assert torch.allclose(whole.snapshot()["sigma"], split.snapshot()["sigma"], atol=1e-9)


def test_padding_and_masked_positions_reach_no_cell():
    b = _rms_batch(bs=2, resp=3, k=3, n_off=1, seed=15)
    stats = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _fold_rms(stats, b, torch.tensor([0, -1]), torch.tensor([[1], [1]]), mask=mask)
    assert float(stats.snapshot()["n"][0]) == 1.0
    assert float(stats.snapshot()["n"][1]) == 0.0


def test_an_unobserved_or_zero_cell_is_unavailable_and_not_epsilon_patched():
    stats = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    _diag, valid = stats.diagonal()
    assert not bool(valid.any()), "nothing observed yet"
    # A teacher identical to the base: every shift is exactly zero, so the
    # second moment is zero and the cell has no scale to offer.
    stats.update(
        shifts={
            "on": torch.zeros(1, 1, 2), "off": torch.zeros(1, 1, 2, 1),
            "tail_on": torch.zeros(1, 1), "tail_off": torch.zeros(1, 1, 1),
        },
        student_logprob=_lp(1, 1, 2, seed=17), response_mask=torch.ones(1, 1),
        task_ids=torch.zeros(1, dtype=torch.long), off_plane_tasks=torch.zeros(1, 1, dtype=torch.long),
    )
    stats.all_reduce()
    _diag, valid = stats.diagonal()
    assert not bool(valid[0]), "a zero sigma is unavailable, not a tiny divisor"


def test_the_rms_state_survives_a_round_trip():
    b = _rms_batch(bs=2, resp=2, k=3, n_off=2, seed=18)
    a = _fold_rms(CumulativePolicyShiftRMS(n_tasks=3, device="cpu"), b,
                  torch.tensor([0, 1]), torch.tensor([[1, 2], [0, 2]]))
    c = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    c.load_state_dict(a.state_dict())
    assert torch.allclose(a.snapshot()["sigma"], c.snapshot()["sigma"])
    with pytest.raises(AssertionError):
        CumulativePolicyShiftRMS(n_tasks=2, device="cpu").load_state_dict(a.state_dict())


# --------------------------------------------------------------------------- #
# standardisation
# --------------------------------------------------------------------------- #
def _std(delta_on, delta_off, diag, valid=None, dst=0, planes=(1, 2)):
    bs = delta_on.size(0)
    d = torch.as_tensor(diag, dtype=torch.float32)
    return standardize_policy_shifts(
        shifts={"on": delta_on, "off": delta_off},
        diag=d,
        diag_valid=torch.ones_like(d, dtype=torch.bool) if valid is None else torch.as_tensor(valid),
        task_ids=torch.full((bs,), dst),
        off_plane_tasks=torch.tensor(planes).view(1, -1).expand(bs, -1).contiguous(),
    )


def test_scaling_a_teachers_whole_shift_leaves_the_standardized_one_alone():
    """The KL coefficients differed 10x; that must not reach the weight."""
    on = torch.tensor([[[1.0, -2.0, 0.5]]])
    off = torch.tensor([[[[1.0, 0.5], [-2.0, 1.0], [0.5, -0.25]]]])
    a = _std(on, off, diag=[1.0, 1.0, 1.0])
    b = _std(on * 4, off * 4, diag=[4.0, 4.0, 4.0])
    assert torch.allclose(a["on"], b["on"], atol=1e-6)
    assert torch.allclose(a["off"], b["off"], atol=1e-6)


def test_the_ratios_between_a_teachers_own_tokens_are_preserved():
    on = torch.tensor([[[1.0, 3.0, -2.0]]])
    off = torch.zeros(1, 1, 3, 2)
    got = _std(on, off, diag=[2.0, 1.0, 1.0])["on"]
    assert torch.allclose(got, on / 2.0, atol=1e-6)


def test_a_quiet_out_of_domain_source_keeps_a_small_standardized_shift():
    """The property the diagonal buys, stated as an assertion.

    Halving a source's shift on THIS destination, with its own in-domain RMS
    fixed, halves what the evidence sees. Under a destination-conditioned
    divisor both would come out at one unit and the difference would vanish.
    """
    on = torch.ones(1, 1, 3)
    loud = torch.full((1, 1, 3, 1), 1.0)
    quiet = torch.full((1, 1, 3, 1), 0.5)
    diag = [1.0, 1.0]
    a = _std(on, loud, diag=diag, planes=(1,))["off"]
    b = _std(on, quiet, diag=diag, planes=(1,))["off"]
    assert torch.allclose(b, a * 0.5, atol=1e-6)


def test_a_row_whose_scale_is_missing_is_marked_unavailable():
    on = torch.ones(2, 1, 2)
    off = torch.ones(2, 1, 2, 2)
    got = _std(on, off, diag=[1.0, 1.0, 1.0], valid=[True, True, False])
    assert got["row_available"].tolist() == [False, False], "source 2 has no scale"
    got = _std(on, off, diag=[1.0, 1.0, 1.0], valid=[False, True, True])
    assert got["row_available"].tolist() == [False, False], "the destination has no scale"
    got = _std(on, off, diag=[1.0, 1.0, 1.0], valid=[True, True, True])
    assert got["row_available"].tolist() == [True, True]


def test_an_untagged_row_is_unavailable_and_indexes_nothing():
    on, off = torch.ones(1, 1, 2), torch.ones(1, 1, 2, 1)
    got = standardize_policy_shifts(
        shifts={"on": on, "off": off},
        diag=torch.tensor([1.0, 1.0]), diag_valid=torch.tensor([True, True]),
        task_ids=torch.tensor([-1]), off_plane_tasks=torch.tensor([[1]]),
    )
    assert got["row_available"].tolist() == [False]
    assert torch.isfinite(got["on"]).all()


# --------------------------------------------------------------------------- #
# common / common_ev / residual
# --------------------------------------------------------------------------- #
def _one(on_v, off_v):
    """One candidate: on-task shift ``on_v`` and a list of off-task shifts."""
    hat_on = torch.tensor([[[float(on_v)]]])
    hat_off = torch.tensor([[[[float(x) for x in off_v]]]])
    return decompose_common_residual(hat_on=hat_on, hat_off=hat_off)


def test_common_is_the_minimum_every_teacher_including_the_on_task_one_guarantees():
    got = _one(1.0, [3.0, 2.0])
    assert float(got["common"]) == pytest.approx(1.0)
    got = _one(-1.0, [-3.0, -2.0])
    assert float(got["common"]) == pytest.approx(-1.0)


def test_one_dissenting_teacher_zeroes_the_on_task_inclusive_common():
    assert float(_one(1.0, [-3.0, 2.0])["common"]) == 0.0
    assert float(_one(0.0, [3.0, 2.0])["common"]) == 0.0, "a silent on-task teacher breaks it"


def test_common_never_exceeds_the_on_task_shift():
    for on_v in (0.2, 1.0, 5.0):
        got = _one(on_v, [9.0, 9.0])
        assert abs(float(got["common"])) <= on_v + 1e-6


def test_the_residual_completes_the_source_shift():
    got = _one(1.0, [3.0, 2.0])
    c = float(got["common"])
    assert float(got["residual"][..., 0]) == pytest.approx(3.0 - c)
    assert float(got["residual"][..., 1]) == pytest.approx(2.0 - c)


def test_common_ev_is_the_off_task_only_counterfactual_and_reaches_no_weight():
    """Computed, reported, never applied.

    It answers the one question the all-teacher rule cannot: how much
    corroboration the on-task teacher's silence costs, at the ~64% of teacher
    mass where it says nothing. That the two differ exactly there is the point.
    """
    silent = _one(0.0, [3.0, 2.0])
    assert float(silent["common"]) == 0.0, "the applied bonus needs the on-task teacher"
    assert float(silent["common_ev"]) == pytest.approx(2.0), "the counterfactual sees it"
    assert float(_one(1.0, [3.0, -2.0])["common_ev"]) == 0.0, "the sources split"


def test_the_counterfactual_is_reported_beside_the_applied_share():
    """Both in the logs, and NOT ordered any more -- which is the change.

    While the applied rule was ``1[unanimous] min_j |hat_j|`` the off-task-only
    variant was the same rule with one teacher removed, so it could only be
    larger and a run could read the gap as "what the on-task teacher's silence
    costs". ``common_soft`` is a different rule -- a ceiling at ``|hat_on|``
    and a graded vote, neither of which is a minimum over a unanimity -- so the
    two are no longer nested and the ratio is a comparison, not a bound. What
    replaces the bound is the invariant below: the applied corroboration cannot
    exceed the on-task teacher's own shift.
    """
    got, ctx = _built(bs=4, resp=6, seed=45)
    kl = torch.rand(*got["weight"].shape) + 0.1
    m = _fold_position(got, kl, ctx["task_ids"])
    assert "kl_weight/evidence/shared_mean" in m
    assert "kl_weight/evidence/legacy_hard_offtask_only_mean" in m
    assert m["kl_weight/evidence/shared_mean"] >= 0.0
    assert torch.all(got["common_soft"].abs() <= got["hat_on"].abs() + 1e-6), (
        "the corroboration is capped by the on-task teacher at every candidate"
    )


def test_a_single_source_cannot_corroborate_itself():
    assert float(_one(1.0, [3.0])["common_ev"]) == 0.0


def test_a_near_zero_shift_attenuates_itself_without_a_deadzone():
    """No fixed gate: the minimum drags the corroboration down continuously, so
    drift noise costs a small number instead of tripping a threshold."""
    for eps in (1e-1, 1e-3, 1e-6):
        # rel, not abs: the shifts are float32 and 0.1 is 0.10000000149 there.
        assert float(_one(1.0, [3.0, eps])["common_ev"]) == pytest.approx(eps, rel=1e-5)


# --------------------------------------------------------------------------- #
# candidate evidence -- the two factors, and the separation they buy
# --------------------------------------------------------------------------- #
def _gate(off_v):
    return teacher_similarity(torch.tensor([[[[float(x) for x in off_v]]]]))


def _excl(on_v, off_v):
    return source_exclusive_shift(
        hat_on=torch.tensor([[[float(on_v)]]]),
        hat_off=torch.tensor([[[[float(x) for x in off_v]]]]),
    )


def _evidence(on_v, off_v, source_scale=1.0):
    return float(
        candidate_kl_evidence(
            common=_one(on_v, off_v)["common_soft"],
            source_gate=_gate(off_v),
            exclusive=_excl(on_v, off_v),
            source_scale=source_scale,
        )
    )


# --- the graded corroboration -------------------------------------------------
def test_the_applied_corroboration_is_the_on_task_shift_under_unanimity():
    """Not ``min_j |hat_j|``: the ceiling is the on-task teacher's own shift, and
    every source that cleared it votes the full ceiling rather than the quietest
    one setting the size for all of them."""
    assert float(_one(1.0, [3.0, 2.0])["common_soft"]) == pytest.approx(1.0)
    assert float(_one(-1.0, [-3.0, -3.0])["common_soft"]) == pytest.approx(-1.0)


def test_one_dissenting_teacher_grades_the_corroboration_instead_of_zeroing_it():
    """The veto measured at ``suppression_ratio`` 1.5-4.7x on the live run stops
    being a counterfactual: the dissenter costs its share of the off-task mass."""
    assert float(_one(1.0, [3.0, 2.0])["common"]) == pytest.approx(1.0)
    assert float(_one(1.0, [3.0, -2.0])["common"]) == 0.0, "the old rule vetoed"
    assert float(_one(1.0, [3.0, -2.0])["common_soft"]) == pytest.approx(0.5), (
        "one vote of min(3, 1) = 1 and one of 0, averaged"
    )
    assert float(_one(1.0, [-3.0, -2.0])["common_soft"]) == 0.0, "all of them dissent"


def test_the_corroboration_is_still_capped_by_the_on_task_teacher():
    for on_v in (0.2, 1.0, 5.0):
        assert abs(float(_one(on_v, [9.0, 9.0])["common_soft"]) ) <= on_v + 1e-6


def test_a_silent_on_task_teacher_zeroes_the_corroboration_twice_over():
    """``sign(hat_on) = 0`` drives the vote to zero AND ``hat_on`` multiplies
    through, so the vote's numerical noise at a silent teacher multiplies
    something that is already zero."""
    assert float(_one(0.0, [3.0, 2.0])["common_soft"]) == 0.0
    assert float(_one(1e-12, [3.0, 2.0])["common_soft"]) == pytest.approx(0.0, abs=1e-9)


def test_the_corroboration_keeps_the_on_task_teachers_sign():
    assert float(_one(-2.0, [-3.0, -4.0])["common_soft"]) == pytest.approx(-2.0)


# --- the similarity gate ------------------------------------------------------
def test_identical_teachers_score_one_and_opposite_ones_score_zero():
    assert float(_gate([2.0, 2.0])) == pytest.approx(1.0)
    assert float(_gate([-3.0, -3.0])) == pytest.approx(1.0), "agreement, not positivity"
    assert float(_gate([2.0, -2.0])) == 0.0, "clamped, not negative"


def test_a_silent_teacher_scores_zero_and_not_full_agreement():
    """The failure that rules out the sign-agreement ratio. ``|sum| / sum |.|``
    is ``|h|/|h| = 1`` with one teacher silent -- one teacher agreeing with
    itself, credited as corroboration. A product over PAIRS cannot do that."""
    assert float(_gate([3.0, 0.0])) == 0.0
    assert float(_gate([0.0, 0.0])) == 0.0
    ratio = abs(3.0 + 0.0) / (abs(3.0) + abs(0.0))
    assert ratio == pytest.approx(1.0), "which is exactly what the ratio would have said"


def test_the_gate_reads_magnitude_agreement_and_not_only_sign():
    """``2r/(1+r^2)``. The lopsided off-task pair is the measured case: two of
    the three destinations run at ``bottleneck_share`` near 0.7/0.1."""
    for r, want in ((1.0, 1.0), (2.0, 0.8), (3.0, 0.6), (10.0, 2 * 10 / 101)):
        assert float(_gate([r, 1.0])) == pytest.approx(want, abs=1e-5)
        assert abs(1.0 + r) / (1.0 + r) == pytest.approx(1.0), "the ratio calls them all 1"


def test_the_gate_is_scale_free_and_bounded_by_one():
    assert float(_gate([6.0, 4.0])) == pytest.approx(float(_gate([3.0, 2.0])))
    g = torch.rand(4, 5, 7, 3) * 8.0 - 4.0
    q = teacher_similarity(g)
    assert float(q.max()) <= 1.0 + 1e-6 and float(q.min()) >= 0.0


def test_one_teacher_leaves_the_gate_shut_without_a_branch():
    """The ``n_off < 2`` special case ``common_ev`` has to write out falls out of
    the algebra here: with no pairs there is no cross term."""
    assert float(teacher_similarity(torch.tensor([[[[5.0]]]]))) == 0.0


def test_the_gate_generalises_past_two_teachers():
    assert float(_gate([2.0, 2.0, 2.0])) == pytest.approx(1.0)
    assert float(_gate([2.0, 2.0, 0.0])) == pytest.approx(0.5), "8 / (2 * 8)"


# --- the exclusive shift ------------------------------------------------------
def test_the_source_only_gets_what_the_on_task_teacher_does_not_already_say():
    assert _excl(1.0, [3.0, 2.0]).reshape(-1).tolist() == pytest.approx([2.0, 1.0])
    assert _excl(5.0, [3.0, 2.0]).reshape(-1).tolist() == pytest.approx([0.0, 0.0])


def test_the_two_channels_partition_the_source_shift_exactly():
    """``|h_m| = min(|h_m|, |h_on|) + relu(|h_m| - |h_on|)``: the ceiling the
    corroboration is capped at, plus the excess the source takes. Nothing is
    counted twice and nothing is dropped."""
    on = torch.rand(3, 4, 5) * 6.0 - 3.0
    off = torch.rand(3, 4, 5, 2) * 6.0 - 3.0
    capped = torch.minimum(off.abs(), on.abs().unsqueeze(-1))
    assert torch.allclose(capped + source_exclusive_shift(hat_on=on, hat_off=off), off.abs())


def test_the_excess_survives_a_conflict_because_the_split_is_by_magnitude():
    """The specification is "the on-task shift is small and the off-task one is
    large", which does not become "the signs agree" without a threshold."""
    assert _excl(1.0, [-3.0]).reshape(-1).tolist() == pytest.approx([2.0])


# --- the evidence -------------------------------------------------------------
def test_corroboration_still_scores_above_conflict():
    """The defect the formula exists to avoid, restated for the graded rule: a
    split still cannot outscore a unanimity, at any source strength."""
    for scale in (0.0, 0.2, 0.5, 0.75, 1.0):
        assert _evidence(1.0, [3.0, 2.0], scale) > _evidence(1.0, [3.0, -2.0], scale)


def test_flipping_the_on_task_sign_costs_exactly_the_corroboration():
    """Neither source factor reads the on-task SIGN -- the gate is off-task only
    and the excess is magnitudes -- so the whole difference is ``|c|``."""
    on_agrees = _evidence(1.0, [3.0, 2.0])
    on_opposes = _evidence(-1.0, [3.0, 2.0])
    assert on_agrees - on_opposes == pytest.approx(1.0, abs=1e-5)
    assert float(_one(-1.0, [3.0, 2.0])["common_soft"]) == 0.0
    assert float(_one(-1.0, [3.0, 2.0])["common_ev"]) == pytest.approx(2.0), (
        "still measured, never applied"
    )


def test_the_broken_alternative_would_have_failed_that():
    """Pinned as a counter-example so the fix cannot be undone as a cleanup."""
    def broken(on_v, off_v, alpha):
        d = _one(on_v, off_v)
        r = d["residual"].reshape(-1)
        return abs(float(d["common"])) + alpha * float(r.abs().sum())

    assert broken(1.0, [3.0, 2.0], 1.0) < broken(1.0, [3.0, -2.0], 1.0)


def test_evidence_is_non_negative_and_zero_only_with_no_signal():
    assert _evidence(0.0, [0.0, 0.0]) == pytest.approx(0.0)
    for on_v, off_v, sc in ((1.0, [3.0, -2.0], 0.3), (-1.0, [-1.0, -1.0], 1.0), (0.0, [2.0, 2.0], 0.0)):
        assert _evidence(on_v, off_v, sc) >= 0.0


def test_the_evidence_is_the_two_channels_added():
    """1 for the corroboration, and 12/13 of the 2 + 1 the sources add on top."""
    assert _evidence(1.0, [3.0, 2.0]) == pytest.approx(1.0 + (12 / 13) * 3.0, abs=1e-5)


def test_the_source_scale_is_a_probe_knob_and_zero_leaves_the_corroboration():
    assert _evidence(1.0, [3.0, 2.0], 0.0) == pytest.approx(1.0)
    half = _evidence(1.0, [3.0, 2.0], 0.5) - 1.0
    assert half == pytest.approx(0.5 * (_evidence(1.0, [3.0, 2.0]) - 1.0), abs=1e-5)


def test_sources_are_summed_not_averaged():
    """Both sources clear the ceiling by 3, and the gate is 1 at parity."""
    assert _evidence(3.0, [6.0, 6.0]) == pytest.approx(3.0 + 3.0 + 3.0)


def test_a_second_teacher_that_says_nothing_new_adds_nothing():
    """The separation, as one number. With the off-task teachers at the on-task
    teacher's own volume the source channel is empty -- which is the 53.3% of
    source mass the advantage rule was spending inside the ``agree`` state."""
    assert _evidence(3.0, [3.0, 3.0]) == pytest.approx(3.0)
    assert _evidence(3.0, [3.0, 3.0], 0.0) == pytest.approx(3.0)


def test_the_evidence_signature_takes_no_advantage_and_no_residual():
    """Structural. The reliability is the gate, computed at the candidate; the
    residual and the row's reward belong to the diagnostics only."""
    import inspect

    params = set(inspect.signature(candidate_kl_evidence).parameters)
    assert params == {"common", "source_gate", "exclusive", "source_scale"}


# --- the placebo --------------------------------------------------------------
def test_the_shuffle_keeps_every_teachers_own_shifts_and_breaks_the_pairing():
    off = torch.rand(2, 9, 4, 2) * 6.0 - 3.0
    rolled = decorrelated_off_shifts(off)
    for m in range(2):
        assert torch.allclose(
            rolled[..., m].reshape(2, -1).sort(dim=-1).values,
            off[..., m].reshape(2, -1).sort(dim=-1).values,
        ), "a roll is a permutation of the same row's own shifts"
    assert not torch.allclose(rolled, off), "and it moves them past each other"


def test_the_shuffle_leaves_a_gate_that_needed_the_pairing_with_nothing():
    """Teachers that agree only where they are looking at the same candidate
    lose the agreement; a constant offset -- grammar, not content -- keeps it.
    The ratio between the two is what a structural mask would be derived from."""
    resp = 8
    alternating = torch.tensor([2.0, -2.0]).repeat(resp // 2)
    paired = alternating.reshape(1, resp, 1, 1).expand(1, resp, 1, 2).contiguous()
    assert float(teacher_similarity(paired).mean()) == pytest.approx(1.0)
    assert float(teacher_similarity(decorrelated_off_shifts(paired)).mean()) == pytest.approx(0.0)

    grammar = torch.full((1, resp, 1, 2), 2.0)
    assert float(teacher_similarity(grammar).mean()) == pytest.approx(1.0)
    assert float(teacher_similarity(decorrelated_off_shifts(grammar)).mean()) == pytest.approx(1.0)


def test_the_shuffle_is_deterministic_and_reaches_no_weight():
    off = torch.rand(2, 8, 3, 2)
    assert torch.equal(decorrelated_off_shifts(off), decorrelated_off_shifts(off))
    assert torch.equal(decorrelated_off_shifts(off[:, :1]), off[:, :1]), "nothing to roll"


# --------------------------------------------------------------------------- #
# position weight
# --------------------------------------------------------------------------- #
def test_the_collapsed_form_matches_the_explicit_sum_over_the_support_and_tail():
    on = _lp(2, 3, 5, seed=20)
    e = torch.rand(2, 3, 5)
    p = on.exp()
    explicit = (p * (1.0 + e)).sum(-1) + (1.0 - p.sum(-1))
    assert torch.allclose(position_pre_weight(evidence=e, mass=candidate_mass(on)), explicit, atol=1e-6)


def test_no_evidence_leaves_the_position_untouched():
    on = _lp(2, 3, 4, seed=21)
    got = position_pre_weight(evidence=torch.zeros(2, 3, 4), mass=candidate_mass(on))
    assert torch.allclose(got, torch.ones(2, 3), atol=1e-6)


def test_the_students_own_mass_decides_how_much_a_candidate_counts():
    """The STUDENT's, and that is the whole of the correction. The loss is a
    reverse KL, so a candidate's share of it is the student's probability there;
    weighting the evidence by the teacher's instead moved the weight most where
    the KL was smallest."""
    student = torch.log(torch.tensor([[[0.80, 0.01]]]))
    heavy = position_pre_weight(
        evidence=torch.tensor([[[1.0, 0.0]]]), mass=candidate_mass(student)
    )
    light = position_pre_weight(
        evidence=torch.tensor([[[0.0, 1.0]]]), mass=candidate_mass(student)
    )
    assert float(heavy) == pytest.approx(1.80, abs=1e-5)
    assert float(light) == pytest.approx(1.01, abs=1e-5)


def test_the_tail_is_neutral_so_a_thin_support_is_modulated_thinly():
    thin = torch.log(torch.tensor([[[0.05, 0.05]]]))   # 90% of the mass is tail
    got = position_pre_weight(evidence=torch.full((1, 1, 2), 2.0), mass=candidate_mass(thin))
    assert float(got) == pytest.approx(1.0 + 0.1 * 2.0, abs=1e-5)


def test_the_weight_grows_linearly_in_the_evidence():
    on = _lp(1, 1, 3, seed=22)
    e = torch.rand(1, 1, 3)
    a = position_pre_weight(evidence=e, mass=candidate_mass(on)) - 1.0
    b = position_pre_weight(evidence=e * 3.0, mass=candidate_mass(on)) - 1.0
    assert float(b) == pytest.approx(float(a) * 3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the normaliser -- and the invariant it exists for
# --------------------------------------------------------------------------- #
def _mean_stats(pre, kl, task_ids, mask=None, row_weights=None, n_tasks=3):
    stats = PreviousStepTaskKLWeightedMean(n_tasks=n_tasks, device="cpu")
    stats.update(
        pre_weight=pre, teacher_kl=kl,
        response_mask=torch.ones_like(pre) if mask is None else mask,
        task_ids=task_ids, row_weights=row_weights,
    )
    return stats


def test_the_snapshot_makes_kl_scale_exactly_one_and_not_the_mean_weight():
    """THE invariant. ``mean(W)`` and ``sum(WD)/sum(D)`` are different numbers
    whenever the weight and the KL are correlated -- which is exactly when the
    arm is doing something -- and the one the coefficient's meaning depends on
    is the second."""
    torch.manual_seed(30)
    n = 2000
    pre = (1.0 + torch.rand(n)).view(1, n)
    kl = (0.5 + pre * torch.rand(1, n))           # deliberately correlated
    ids = torch.zeros(1, dtype=torch.long)
    mu = float(_mean_stats(pre, kl, ids).snapshot()["mean"][0])
    w = pre / mu
    # In float64, as the accumulator holds it, the identity is exact.
    kl64 = kl.to(torch.float64)
    w64 = pre.to(torch.float64) / mu
    assert float((w64 * kl64).sum() / kl64.sum()) == pytest.approx(1.0, abs=1e-12)
    # The weight the loss actually applies is float32, so what a run sees is the
    # identity to about 1e-7. Stated rather than hidden: kl_scale is read off a
    # live batch and this is its floor.
    assert float((w * kl).sum() / kl.sum()) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(w.mean()) - 1.0) > 1e-3, "the plain mean is NOT 1 here, and need not be"

    plain = float(pre.mean())
    w_plain = pre / plain
    assert float(w_plain.mean()) == pytest.approx(1.0)
    assert abs(float((w_plain * kl).sum() / kl.sum()) - 1.0) > 1e-3, (
        "a plain-mean normaliser leaves the effective OPD strength off 1"
    )


def test_each_task_gets_its_own_normaliser():
    pre = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
    kl = torch.ones(2, 2)
    snap = _mean_stats(pre, kl, torch.tensor([0, 1])).snapshot()
    assert float(snap["mean"][0]) == pytest.approx(2.0)
    assert float(snap["mean"][1]) == pytest.approx(1.0)
    assert not bool(snap["valid"][2])


def test_the_normaliser_weights_positions_by_the_kl_they_carry():
    """A position with no KL cannot vote on how the KL budget is split."""
    pre = torch.tensor([[3.0, 1.0]])
    kl = torch.tensor([[0.0, 1.0]])
    snap = _mean_stats(pre, kl, torch.zeros(1, dtype=torch.long)).snapshot()
    assert float(snap["mean"][0]) == pytest.approx(1.0)


def test_the_row_weights_of_the_loss_aggregation_are_the_ones_used():
    """The invariant has to hold for the quantity that enters the objective, not
    for an unweighted stand-in of it."""
    pre = torch.tensor([[2.0], [1.0]])
    kl = torch.ones(2, 1)
    ids = torch.zeros(2, dtype=torch.long)
    flat = float(_mean_stats(pre, kl, ids).snapshot()["mean"][0])
    tilted = float(_mean_stats(pre, kl, ids, row_weights=torch.tensor([3.0, 1.0])).snapshot()["mean"][0])
    assert flat == pytest.approx(1.5)
    assert tilted == pytest.approx((3 * 2 + 1 * 1) / 4)


def test_masked_and_untagged_rows_are_excluded():
    pre = torch.tensor([[5.0, 1.0], [9.0, 9.0]])
    kl = torch.ones(2, 2)
    snap = _mean_stats(pre, kl, torch.tensor([0, -1]), mask=torch.tensor([[0.0, 1.0], [1.0, 1.0]])).snapshot()
    assert float(snap["mean"][0]) == pytest.approx(1.0)


def test_the_normaliser_does_not_depend_on_the_micro_batch_split():
    torch.manual_seed(31)
    pre, kl = 1.0 + torch.rand(6, 4), torch.rand(6, 4) + 0.1
    ids = torch.tensor([0, 1, 2, 0, 1, 2])
    whole = _mean_stats(pre, kl, ids).snapshot()["mean"]
    split = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    for r in range(6):
        split.update(
            pre_weight=pre[r : r + 1], teacher_kl=kl[r : r + 1],
            response_mask=torch.ones(1, 4), task_ids=ids[r : r + 1], row_weights=None,
        )
    assert torch.allclose(whole, split.snapshot()["mean"], atol=1e-12)


def test_feeding_the_weighted_kl_back_in_would_drift_and_the_unweighted_one_does_not():
    """The failure mode is silent: step 1 runs at W = 1 so it is right exactly
    once, and the compounding starts at step 2 where no metric is looking."""
    torch.manual_seed(32)
    pre = (1.0 + torch.rand(1, 400))
    kl = 0.5 + pre * torch.rand(1, 400)
    ids = torch.zeros(1, dtype=torch.long)

    mu1 = float(_mean_stats(pre, kl, ids).snapshot()["mean"][0])
    # correct: the same unweighted KL again on an unchanged distribution
    mu2 = float(_mean_stats(pre, kl, ids).snapshot()["mean"][0])
    assert mu2 == pytest.approx(mu1, abs=1e-12)
    # wrong: the KL after the weight multiplied it
    mu2_bad = float(_mean_stats(pre, kl * (pre / mu1), ids).snapshot()["mean"][0])
    assert abs(mu2_bad - mu1) > 1e-3


def test_the_normaliser_state_survives_a_round_trip():
    pre, kl = 1.0 + torch.rand(2, 3), torch.rand(2, 3) + 0.1
    a = _mean_stats(pre, kl, torch.tensor([0, 1]))
    b = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.snapshot()["mean"], b.snapshot()["mean"])
    a.reset()
    assert not bool(a.snapshot()["valid"].any())


# --------------------------------------------------------------------------- #
# group centring
# --------------------------------------------------------------------------- #
def test_group_centring_removes_each_prompts_own_offset():
    vals = torch.tensor([10.0, 12.0, 100.0, 102.0])
    got = group_center(vals, torch.tensor([0, 0, 1, 1]), torch.ones(4))
    assert got.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_group_centring_ignores_invalid_rows_in_the_mean():
    vals = torch.tensor([10.0, 12.0, 1000.0])
    got = group_center(vals, torch.tensor([0, 0, 0]), torch.tensor([1.0, 1.0, 0.0]))
    assert got[:2].tolist() == pytest.approx([-1.0, 1.0])


def test_group_centring_works_per_column_for_multi_source_scores():
    vals = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    got = group_center(vals, torch.tensor([0, 0]), torch.ones(2))
    assert torch.allclose(got, torch.tensor([[-1.0, -10.0], [1.0, 10.0]]))


# --------------------------------------------------------------------------- #
# advantage reliability
# --------------------------------------------------------------------------- #
def _adv(advantage, support, *, dst=0, planes=(1,), informative=None, length=None, on_score=None):
    a = torch.as_tensor(advantage, dtype=torch.float32)
    s = torch.as_tensor(support, dtype=torch.float32).reshape(a.numel(), -1)
    n = a.numel()
    stats = AdvantageReliabilityStats(n_tasks=3, device="cpu")
    stats.update(
        advantage=a,
        support_score=s,
        on_support_score=torch.zeros(n) if on_score is None else torch.as_tensor(on_score, dtype=torch.float32),
        length=torch.ones(n) if length is None else torch.as_tensor(length, dtype=torch.float32),
        informative=torch.ones(n, dtype=torch.bool) if informative is None else torch.as_tensor(informative),
        task_ids=torch.full((n,), dst),
        off_plane_tasks=torch.tensor(planes).view(1, -1).expand(n, -1).contiguous(),
    )
    stats.all_reduce()
    return stats


def test_a_source_that_ranks_the_rollouts_the_way_the_reward_does_gets_alpha_one():
    a = [1.0, 2.0, 3.0, 4.0]
    got = _adv(a, a).alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho"] == pytest.approx(1.0, abs=1e-6)
    assert got["alpha"] == pytest.approx(1.0, abs=1e-6)


def test_an_anti_correlated_source_is_vetoed_and_not_inverted():
    got = _adv([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]).alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho"] == pytest.approx(-1.0, abs=1e-6)
    assert got["alpha"] == 0.0, "vetoed; the shift is not re-used with the sign flipped"


def test_an_uninformative_source_lands_near_zero():
    torch.manual_seed(33)
    a = torch.randn(400).tolist()
    s = torch.randn(400).tolist()
    got = _adv(a, s).alpha(task_names=TASKS)[("alfworld", "search")]
    assert abs(got["rho"]) < 0.2 and got["alpha"] < 0.2


def test_the_correlation_matches_the_textbook_one():
    torch.manual_seed(34)
    a, s = torch.randn(200), None
    s = 0.7 * a + 0.7 * torch.randn(200)
    got = _adv(a.tolist(), s.tolist()).alpha(task_names=TASKS)[("alfworld", "search")]
    expect = float(np.corrcoef(a.numpy(), s.numpy())[0, 1])
    assert got["rho"] == pytest.approx(expect, abs=1e-5)


def test_groups_with_no_reward_spread_and_padding_copies_do_not_enter():
    """GRPO is group-relative, so a prompt whose rollouts all scored the same
    gives every row an advantage of zero: folding it in adds variance to S
    against none in A and drags every correlation toward zero."""
    a = [1.0, 2.0, 3.0, 4.0]
    ref = _adv(a, a).alpha(task_names=TASKS)[("alfworld", "search")]
    padded = _adv(
        a + [0.0, 0.0, 0.0], a + [50.0, -50.0, 7.0],
        informative=[True] * 4 + [False] * 3,
    ).alpha(task_names=TASKS)[("alfworld", "search")]
    assert padded["n"] == ref["n"] == 4
    assert padded["rho"] == pytest.approx(ref["rho"], abs=1e-9)


def test_each_ordered_pair_is_its_own_cell():
    a = [1.0, 2.0, 3.0, 4.0]
    stats = _adv(a, [[x, -x] for x in a], planes=(1, 2))
    got = stats.alpha(task_names=TASKS)
    assert got[("alfworld", "search")]["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert got[("alfworld", "webshop")]["alpha"] == 0.0
    assert ("search", "alfworld") not in got, "no rows of task search were folded in"


def test_the_table_is_zero_on_the_diagonal_and_where_undefined():
    table = _adv([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]).alpha_table()
    assert table.shape == (3, 3)
    assert float(table[0, 0]) == 0.0
    assert float(table[0, 1]) == pytest.approx(1.0, abs=1e-6)
    assert float(table[1, 0]) == 0.0, "never observed"


def test_reliability_does_not_depend_on_the_batch_split():
    torch.manual_seed(35)
    a, s = torch.randn(12), torch.randn(12)
    whole = _adv(a.tolist(), s.tolist()).alpha_table()
    split = AdvantageReliabilityStats(n_tasks=3, device="cpu")
    for r in range(0, 12, 4):
        split.update(
            advantage=a[r : r + 4], support_score=s[r : r + 4].reshape(4, 1),
            on_support_score=torch.zeros(4), length=torch.ones(4),
            informative=torch.ones(4, dtype=torch.bool),
            task_ids=torch.zeros(4, dtype=torch.long), off_plane_tasks=torch.full((4, 1), 1),
        )
    split.all_reduce()
    assert torch.allclose(whole, split.alpha_table(), atol=1e-6)


def test_too_few_rows_leave_alpha_undefined_rather_than_confident():
    got = _adv([1.0, 2.0], [1.0, 2.0]).alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho"] is None and got["alpha"] == 0.0


def test_the_lower_confidence_bound_is_reported_and_never_applied():
    """A confidence level is a knob and this arm has none, so the bound is a
    diagnostic. It is what says when a positive alpha is indistinguishable from
    the rectifier's own small-sample bias."""
    torch.manual_seed(36)
    a = torch.randn(20)
    s = 0.4 * a + torch.randn(20)
    got = _adv(a.tolist(), s.tolist()).alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho_lcb95"] is not None
    assert got["rho_lcb95"] < got["rho"]
    assert got["alpha"] == pytest.approx(max(0.0, got["rho"]))


def test_the_length_and_on_task_controls_are_reported_and_never_applied():
    torch.manual_seed(37)
    n = 300
    length = torch.rand(n) * 10 + 1
    a = length + 0.2 * torch.randn(n)          # reward confounded with length
    s = length + 0.2 * torch.randn(n)          # so is the support score
    got = _adv(a.tolist(), s.tolist(), length=length).alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho"] > 0.9, "the raw correlation is almost all length"
    assert abs(got["rho_length_controlled"]) < 0.3, "and almost none of it survives the control"
    assert got["alpha"] == pytest.approx(max(0.0, got["rho"]))


def test_the_reliability_state_survives_a_round_trip():
    stats = _adv([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    other = AdvantageReliabilityStats(n_tasks=3, device="cpu")
    other.load_state_dict(stats.state_dict())
    assert torch.allclose(stats.alpha_table(), other.alpha_table())
    bad = dict(stats.state_dict())
    bad["moments"] = ("n", "a")
    with pytest.raises(AssertionError, match="moment layout"):
        AdvantageReliabilityStats(n_tasks=3, device="cpu").load_state_dict(bad)


def test_the_moment_layout_is_complete_enough_for_the_partials():
    assert ADV_MOMENTS[0] == "n"
    for pair in ("as", "al", "ao", "sl", "so", "lo"):
        assert pair in ADV_MOMENTS, pair


# --------------------------------------------------------------------------- #
# the orchestrator, and the effect metrics
# --------------------------------------------------------------------------- #
from verl.trainer.ppo.cross_teacher_kl_weight import (  # noqa: E402
    CHANNEL_PROBES,
    POSITION_TERMS,
    PROBE_ALPHAS,
    STATE_TERMS,
    PairEvidenceStats,
    build_position_weight,
    position_terms,
    position_weight_metrics,
    probe_name,
    state_shift_metrics,
    state_shift_terms,
)
from verl.trainer.ppo.sign_weights import ScopeTermStats  # noqa: E402


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


def _built(bs=3, resp=4, k=5, n_off=2, seed=40, normalizer="auto", diag_valid=None, alpha=0.5):
    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1)
    shifts = compute_raw_policy_shifts(on_task_logprob=on, off_task_logprobs=off, base_logprob=base)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)[:, :n_off]
    alpha_table = torch.full((3, 3), float(alpha))
    alpha_table.fill_diagonal_(0.0)
    norm = None
    if normalizer == "auto":
        norm = {"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)}
    elif normalizer is not None:
        norm = normalizer
    got = build_position_weight(
        shifts=shifts, on_task_logprob=on, student_logprob=_student_like(on),
        response_mask=torch.ones(bs, resp), task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=alpha_table, normalizer=norm,
        diag_valid=torch.ones(3, dtype=torch.bool) if diag_valid is None else torch.as_tensor(diag_valid),
    )
    return got, {"on": on, "task_ids": task_ids, "planes": planes, "shifts": shifts}


def test_the_state_columns_partition_the_whole_kl_shift():
    """Seven states plus the normaliser's own offset, with no residual. A column
    that did not add up would make every share below a correlated summary rather
    than a decomposition."""
    got, ctx = _built()
    kl = torch.rand(*got["weight"].shape) + 0.1
    terms = state_shift_terms(got, kl)
    total = sum(terms[t] for t in STATE_TERMS)
    assert torch.allclose(total, (got["weight"] - 1.0) * kl, atol=1e-5)
    assert set(terms) == set(STATE_TERMS)


def test_the_state_labels_are_the_shipped_seven_and_report_epsilon_reaches_no_weight():
    from verl.trainer.ppo.sign_weights import STATE_NAMES

    got, ctx = _built()
    assert set(int(x) for x in got["state"].unique().tolist()) <= set(STATE_NAMES)
    loose = build_position_weight(
        shifts=ctx["shifts"], on_task_logprob=ctx["on"], student_logprob=_student_like(ctx["on"]), response_mask=None, task_ids=ctx["task_ids"],
        off_plane_tasks=ctx["planes"], diag=torch.ones(3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        alpha_table=torch.full((3, 3), 0.5).fill_diagonal_(0.0),
        normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
        report_epsilon=5.0,
    )
    assert torch.equal(loose["weight"], got["weight"]), "labels only"
    assert not torch.equal(loose["state"], got["state"]), "but it does move the labels"


def test_cold_start_is_exactly_one_and_not_the_unnormalised_weight():
    """The existing position arm applies the raw W~ on its first step. Inheriting
    that here would be an unannounced increase in distillation strength for as
    long as the normaliser takes to exist."""
    got, _ = _built(normalizer=None)
    assert torch.allclose(got["weight"], torch.ones_like(got["weight"]))
    assert float(got["pre_weight"].max()) > 1.0, "the raw weight is still measured"


def test_a_task_without_a_scale_stays_at_one_and_reports_nothing_it_cannot_measure():
    got, _ = _built(diag_valid=[True, False, True])
    avail = got["available"]
    assert avail.tolist() == [True, False, True] or not bool(avail[1])
    idx = (~avail).nonzero().flatten()
    assert idx.numel()
    for i in idx.tolist():
        assert torch.allclose(got["weight"][i], torch.ones_like(got["weight"][i]))
        assert float(got["hat_on"][i].abs().max()) == 0.0
        assert float(got["hat_off"][i].abs().max()) == 0.0


def test_the_probe_series_scales_the_source_and_its_top_is_the_shipped_weight():
    """The names are unchanged and one of the two readings is: ``alpha000`` is
    still the corroboration channel alone, which is what a control arm
    reproduces. The other has moved. The number is a plain multiplier on the
    gated source now, not a reliability, so the top of the series is the arm
    itself rather than an upper bracket -- what the bracket used to say is the
    ``ungated_source`` channel's job."""
    got, _ = _built()
    assert set(got["probe_pre_weight"]) == {probe_name(a) for a in PROBE_ALPHAS}
    lo = got["probe_pre_weight"][probe_name(0.0)]
    hi = got["probe_pre_weight"][probe_name(1.0)]
    assert torch.all(hi >= lo - 1e-6), "more source cannot mean less evidence"
    assert float((hi - lo).max()) > 0.0
    assert torch.allclose(got["pre_weight"], hi, atol=1e-6), "scale 1.0 IS the arm"


def test_the_advantage_table_no_longer_reaches_the_weight():
    """The demotion, as a test rather than as a comment. Two runs that differ
    only in the reliability table have to produce the same weight; the table is
    still carried, for the diagnostics that ask whether the reward-free gate
    landed where the reward would have."""
    off_gate, _ = _built(alpha=0.0)
    full_gate, _ = _built(alpha=1.0)
    assert torch.allclose(off_gate["weight"], full_gate["weight"], atol=1e-7)
    assert torch.allclose(off_gate["evidence"], full_gate["evidence"], atol=1e-7)
    assert float(off_gate["row_alpha"].max()) == 0.0
    assert float(full_gate["row_alpha"].max()) == 1.0


def test_the_gate_channel_counterfactuals_bracket_the_shipped_gate():
    """``ungated_source`` is q = 1 and can only be larger; ``shuffled_gate`` is
    the same arithmetic on teachers that were never looking at the same
    candidate, and is the null the shared base can produce on its own."""
    got, _ = _built()
    assert set(got["channel_pre_weight"]) == set(CHANNEL_PROBES)
    ungated = got["channel_pre_weight"]["ungated_source"]
    assert torch.all(ungated >= got["pre_weight"] - 1e-6)
    assert float((ungated - got["pre_weight"]).max()) > 0.0, "the gate is closing something"
    assert got["channel_pre_weight"]["shuffled_gate"].shape == got["pre_weight"].shape


def test_the_evidence_by_source_columns_add_up_with_the_shared_one():
    got, _ = _built()
    total = got["evidence_shared"] + got["evidence_by_source"].sum(dim=(-1, -2))
    assert torch.allclose(total, got["pre_weight"] - 1.0, atol=1e-5)


def _fold_position(got, kl, task_ids, n_tasks=3):
    stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=n_tasks, device="cpu")
    stats.update(position_terms(got, kl), response_mask=torch.ones_like(kl), task_ids=task_ids)
    return position_weight_metrics(stats.sums(task_names=TASKS))


def test_kl_scale_is_one_on_the_snapshot_the_normaliser_came_from():
    """The end-to-end version of the normaliser's invariant: build W~ and D,
    take the KL-weighted mean, feed it back, and the effective OPD strength is
    unchanged."""
    got, ctx = _built(bs=6, resp=8, seed=41, normalizer=None)
    kl = torch.rand(*got["weight"].shape) + 0.1
    mean = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    mean.update(
        pre_weight=got["pre_weight"], teacher_kl=kl,
        response_mask=torch.ones_like(kl), task_ids=ctx["task_ids"],
    )
    again = build_position_weight(
        shifts=ctx["shifts"], on_task_logprob=ctx["on"], student_logprob=_student_like(ctx["on"]), response_mask=None, task_ids=ctx["task_ids"],
        off_plane_tasks=ctx["planes"], diag=torch.ones(3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        alpha_table=torch.full((3, 3), 0.5).fill_diagonal_(0.0),
        normalizer=mean.snapshot(),
    )
    m = _fold_position(again, kl, ctx["task_ids"])
    for task in TASKS:
        assert m[f"kl_weight/{task}/effect/kl_scale"] == pytest.approx(1.0, abs=1e-5)
    assert m["kl_weight/effect/kl_scale"] == pytest.approx(1.0, abs=1e-5)


def test_both_sides_of_one_are_reached_once_the_normaliser_exists():
    got, ctx = _built(bs=6, resp=8, seed=42)
    w = got["weight"]
    assert float(w.min()) < 1.0 < float(w.max())


def test_the_effect_metrics_are_the_ratios_a_reader_needs():
    got, ctx = _built(bs=4, resp=6, seed=43)
    kl = torch.rand(*got["weight"].shape) + 0.1
    m = _fold_position(got, kl, ctx["task_ids"])
    for key in (
        "position/w_mean", "position/w_cv", "position/available_frac",
        "evidence/shared_share", "effect/kl_scale", "effect/kl_shift_gross_frac",
        "effect/redistribution_ratio",
    ):
        assert f"kl_weight/{key}" in m, key
    # the gate's number is a fraction of the OPD term, so it is unit-free
    assert 0.0 <= m["kl_weight/effect/kl_shift_gross_frac"] < 10.0


def test_no_metric_name_collides_with_the_reducers_max_min_dispatch():
    got, ctx = _built(bs=2, resp=3, seed=44)
    kl = torch.rand(*got["weight"].shape) + 0.1
    m = _fold_position(got, kl, ctx["task_ids"])
    stats = ScopeTermStats(names=STATE_TERMS, n_tasks=3, device="cpu")
    stats.update(state_shift_terms(got, kl), response_mask=torch.ones_like(kl), task_ids=ctx["task_ids"])
    m.update(state_shift_metrics(stats.sums(task_names=TASKS)))
    pair = PairEvidenceStats(n_tasks=3, device="cpu")
    pair.update(
        evidence=got["evidence_by_source"].sum(dim=2),
        shift=got["evidence_by_source"].sum(dim=2),
        response_mask=torch.ones_like(kl), task_ids=ctx["task_ids"], off_plane_tasks=ctx["planes"],
    )
    m.update(pair.metrics(task_names=TASKS))
    assert m
    for name in m:
        assert "max" not in name and "min" not in name, name


def test_the_specialist_state_has_a_column_and_is_where_budget_is_taken_from():
    """``neutral_off_task_silent`` -- the on-task teacher is the only one with an
    opinion. The evidence is zero there by construction, so it is what the arm
    takes budget away from, and the earlier four-value label set had no name for
    it at all."""
    assert "shift_neutral_off_task_silent" in STATE_TERMS
    bs, resp, k = 1, 1, 2
    on = torch.log(torch.tensor([[[0.6, 0.3]]]))
    # on-task teacher moved; both sources sit exactly on the base
    shifts = {
        "on": torch.tensor([[[1.0, 1.0]]]),
        "off": torch.zeros(bs, resp, k, 2),
        "tail_on": torch.zeros(bs, resp),
        "tail_off": torch.zeros(bs, resp, 2),
    }
    got = build_position_weight(
        shifts=shifts, on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=torch.zeros(1, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2]]), diag=torch.ones(3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        alpha_table=torch.ones(3, 3).fill_diagonal_(0.0),
        normalizer={"mean": torch.full((3,), 1.4), "valid": torch.ones(3, dtype=torch.bool)},
    )
    assert float(got["pre_weight"]) == pytest.approx(1.0), "no off-task evidence at all"
    assert float(got["weight"]) < 1.0, "so it loses budget to the task's other positions"
    assert int(got["state"][0, 0, 0]) == 6, "neutral_off_task_silent"


# --------------------------------------------------------------------------- #
# group centring inside the reliability statistic
# --------------------------------------------------------------------------- #
def _adv_grouped(advantage, support, groups, **kw):
    a = torch.as_tensor(advantage, dtype=torch.float32)
    n = a.numel()
    stats = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=kw.pop("max_groups", 8))
    stats.update(
        advantage=a,
        support_score=torch.as_tensor(support, dtype=torch.float32).reshape(n, 1),
        on_support_score=torch.zeros(n), length=torch.ones(n),
        informative=torch.ones(n, dtype=torch.bool),
        task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=torch.full((n, 1), 1),
        group_ids=torch.as_tensor(groups, dtype=torch.long),
    )
    stats.all_reduce()
    return stats.alpha(task_names=TASKS)[("alfworld", "search")]


def test_a_per_prompt_offset_in_the_support_score_is_divided_out():
    """A prompt every rollout finds hard shifts the whole group's score. The
    advantage it is correlated against has already had exactly that removed, so
    leaving it in only deflates a real correlation -- which on a statistic this
    noisy is the difference between a signal and a shrug."""
    # Two groups, identical within-group structure, wildly different offsets.
    a = [-1.0, 1.0, -1.0, 1.0]
    s = [-1.0, 1.0, 99.0, 101.0]
    groups = [0, 0, 1, 1]
    assert _adv_grouped(a, s, groups)["rho"] == pytest.approx(1.0, abs=1e-6)
    # Without the grouping the offset dominates and the correlation collapses.
    flat = _adv(a, s).alpha(task_names=TASKS)[("alfworld", "search")]
    assert abs(flat["rho"]) < 0.1


def test_one_group_holding_every_row_reproduces_the_ordinary_variance():
    """The refinement has to degenerate cleanly, or it is a different statistic."""
    torch.manual_seed(50)
    a, s = torch.randn(60), torch.randn(60)
    grouped = _adv_grouped(a.tolist(), s.tolist(), [0] * 60)["rho"]
    plain = _adv(a.tolist(), s.tolist()).alpha(task_names=TASKS)[("alfworld", "search")]["rho"]
    assert grouped == pytest.approx(plain, abs=1e-9)


def test_the_group_pooling_survives_being_split_across_micro_batches():
    """A group's rollouts land in different micro-batches and on different
    ranks, so centring locally would centre a fragment."""
    a = [-1.0, 1.0, -1.0, 1.0]
    s = [-1.0, 1.0, 99.0, 101.0]
    groups = [0, 1, 0, 1]
    whole = _adv_grouped(a, s, groups)["rho"]
    split = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    for r in range(4):
        split.update(
            advantage=torch.tensor([a[r]]), support_score=torch.tensor([[s[r]]]),
            on_support_score=torch.zeros(1), length=torch.ones(1),
            informative=torch.ones(1, dtype=torch.bool),
            task_ids=torch.zeros(1, dtype=torch.long), off_plane_tasks=torch.full((1, 1), 1),
            group_ids=torch.tensor([groups[r]]),
        )
    split.all_reduce()
    assert split.alpha(task_names=TASKS)[("alfworld", "search")]["rho"] == pytest.approx(whole, abs=1e-9)


def test_a_group_id_past_the_buffer_loses_power_rather_than_mixing_prompts():
    a = [-1.0, 1.0, -1.0, 1.0]
    s = [-1.0, 1.0, 99.0, 101.0]
    got = _adv_grouped(a, s, [0, 0, 99, 99], max_groups=8)
    assert got["n"] == 4, "the moments still see every row"
    assert got["rho"] is not None


def test_the_group_buffer_is_part_of_the_resumable_state():
    stats = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    stats.update(
        advantage=torch.tensor([-1.0, 1.0]), support_score=torch.tensor([[0.0], [2.0]]),
        on_support_score=torch.zeros(2), length=torch.ones(2),
        informative=torch.ones(2, dtype=torch.bool),
        task_ids=torch.zeros(2, dtype=torch.long), off_plane_tasks=torch.full((2, 1), 1),
        group_ids=torch.tensor([0, 0]),
    )
    stats.all_reduce()
    other = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    other.load_state_dict(stats.state_dict())
    assert torch.allclose(stats.group, other.group)
    with pytest.raises(AssertionError, match="max_groups"):
        AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4).load_state_dict(stats.state_dict())


# --------------------------------------------------------------------------- #
# the residual support score
# --------------------------------------------------------------------------- #
def test_the_support_score_measures_the_choice_and_not_how_opinionated_the_source_is():
    """z = r(y) - E_student[r]. A source that likes every candidate equally
    scores zero however loudly it likes them."""
    from verl.trainer.ppo.cross_teacher_kl_weight import residual_support_score

    student = torch.log(torch.tensor([[[0.5, 0.5]]]))
    flat = residual_support_score(
        residual_at_sampled=torch.tensor([[[7.0]]]),
        residual=torch.full((1, 1, 2, 1), 7.0),
        student_logprob=student, response_mask=torch.ones(1, 1),
    )
    assert float(flat) == pytest.approx(0.0, abs=1e-5)


def test_backing_the_emitted_token_above_the_students_average_scores_positive():
    from verl.trainer.ppo.cross_teacher_kl_weight import residual_support_score

    student = torch.log(torch.tensor([[[0.5, 0.5]]]))
    got = residual_support_score(
        residual_at_sampled=torch.tensor([[[3.0]]]),
        residual=torch.tensor([[[[3.0], [1.0]]]]),
        student_logprob=student, response_mask=torch.ones(1, 1),
    )
    assert float(got) == pytest.approx(3.0 - 2.0, abs=1e-5)


def test_the_trajectory_score_is_a_sum_over_valid_positions():
    """A sum, because the policy gradient adds each valid token to the loss.
    The length-normalised reading is carried as a diagnostic moment instead of
    replacing it."""
    from verl.trainer.ppo.cross_teacher_kl_weight import residual_support_score

    student = torch.log(torch.full((1, 3, 2), 0.5))
    got = residual_support_score(
        residual_at_sampled=torch.tensor([[[2.0], [2.0], [50.0]]]),
        residual=torch.zeros(1, 3, 2, 1),
        student_logprob=student, response_mask=torch.tensor([[1.0, 1.0, 0.0]]),
    )
    assert float(got) == pytest.approx(4.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# The wiring: what a CPU suite can still check about a GPU-only path
# --------------------------------------------------------------------------- #
from tests.trainer.test_transfer_metrics import _update_policy_source  # noqa: E402


def _actor_source() -> str:
    import inspect

    from verl.workers.actor import dp_actor

    return open(inspect.getsourcefile(dp_actor), encoding="utf-8").read()


def test_every_reader_of_the_kl_runs_before_the_weight_multiplies_it():
    """The normaliser composes with itself if it is fed the weighted KL, and
    since the first step runs at W = 1 it would be right exactly once and drift
    from the second. kl_scale would also come out 1 by construction rather than
    by measurement."""
    src = _update_policy_source()
    multiply = src.index('teacher_kld = teacher_kld * xt_built["weight"]')
    for reader in (
        "self._xt_mean.update(",
        "xt_position_stats.update(",
        "xt_state_stats.update(",
        "xt_pair_stats.update(",
    ):
        assert src.index(reader) < multiply, reader


def test_the_weight_is_the_only_thing_that_touches_the_loss():
    """One line, and it multiplies the teacher KL. No target rewrite, no
    per-vocabulary-term weighting, and nothing on the policy gradient."""
    src = _update_policy_source()
    assert src.count('teacher_kld = teacher_kld * xt_built["weight"]') == 1
    # The arm's own block in the micro-batch loop, not the config gate above it.
    block = src[src.index("xt_built = None\n"):]
    block = block[: block.index("loss_mode = ")]
    for forbidden in ("reweight_teacher_logprobs", "pg_loss", "advantages"):
        assert forbidden not in block, forbidden


def test_the_accumulators_are_gated_on_the_config_and_not_on_the_batch():
    """A rank whose micro-batch carries no sign columns must still build them,
    or the all-reduces below it deadlock against its neighbours'."""
    src = _update_policy_source()
    for line in src.splitlines():
        if any(
            ctor in line
            for ctor in (
                "CumulativePolicyShiftRMS(", "PreviousStepTaskKLWeightedMean(",
                "AdvantageReliabilityStats(", "PairEvidenceStats(",
            )
        ):
            assert "xt_enabled" not in line, line
    gate = src[src.index("if xt_cfg_on:"):src.index("xt_on = ")]
    assert "xt_enabled" not in gate


def test_the_collectives_run_unconditionally_once_the_arm_is_on():
    src = _update_policy_source()
    tail = src[src.rindex("        if xt_on:"):]
    for call in (
        "self._xt_rms.all_reduce()", "self._xt_mean.all_reduce()",
        "self._xt_adv.all_reduce()", "xt_position_stats.all_reduce()",
        "xt_state_stats.all_reduce()", "xt_pair_stats.all_reduce()",
    ):
        assert call in tail, call
    # ... and no line between the gate and the last of them is itself a branch,
    # so none can be skipped on what a rank's micro-batches happened to hold.
    head = tail[: tail.index("xt_pair_stats.all_reduce()")].splitlines()[1:]
    branches = [
        ln.strip() for ln in head
        if ln.strip().startswith(("if ", "elif ", "while ")) and "is_initialized" not in ln
    ]
    assert not branches, branches


def test_the_normaliser_is_snapshotted_and_then_cleared_in_that_order():
    """One step's lag. Snapshotting after the reset would hand every step an
    empty divisor; not resetting at all would make it a run-long average, which
    is a different mechanism."""
    src = _update_policy_source()
    block = src[src.index("xt_mean_snapshot = None"):src.index("# Individual candidates")]
    assert block.index("self._xt_mean.snapshot()") < block.index("self._xt_mean.reset()")


def test_the_micro_batch_is_never_treated_as_a_dataproto():
    src = _update_policy_source()
    body = src[src.index('responses = data["responses"]'):]
    offenders = [ln.strip() for ln in body.splitlines() if "data.batch" in ln.split("#")[0]]
    assert not offenders, offenders


def test_the_two_mechanisms_refuse_to_run_together():
    """They are two ways to spend one signal and both multiply the same KL.
    Together they train an arm that is neither and report both sets of metrics
    as if they described it."""
    for src in (_actor_source(), open("verl/trainer/ppo/opd_ray_trainer.py").read()):
        assert "sign_weight and cross_teacher_kl_weight" in src or (
            "algorithm.opd.sign_weight and algorithm.opd.cross_teacher_kl_weight" in src
        )


def test_the_driver_shares_one_gate_for_the_cache_and_the_base_worker():
    """The four models, the hidden-state cache and the sign_cache_ids columns
    are the same for both arms; only what the ACTOR does with them differs. A
    per-mechanism gate is how the second arm came to run three extra forwards
    and then read nothing."""
    src = open("verl/trainer/ppo/opd_ray_trainer.py").read()
    for marker in (
        "self.cross_teacher_enabled = (",
        "or self.cross_teacher_kl_weight_enabled",
        "or self.cross_teacher_target_enabled",
        "self.need_hidden_cache = self.student_indexed_topk or self.cross_teacher_enabled",
        "if not self.cross_teacher_enabled:",
    ):
        assert marker in src, marker
    # and no cache gate is left on the sign flag alone
    for line in src.splitlines():
        if "sign_weight_enabled" in line and "cross_teacher" not in line:
            assert "self.sign_weight_enabled = " in line or "check_sign_weight" in line or (
                "if self.sign_weight_enabled:" in line
            ), line


def test_the_config_injection_carries_the_block_to_the_actor():
    """The weight is built where the student's top-k exists, which is inside the
    actor's forward, so the settings have to reach the actor config while
    staying authored under algorithm.opd with the other scientific knobs."""
    src = open("verl/trainer/main_opd.py").read()
    assert "config.actor_rollout_ref.actor.cross_teacher_kl_weight = xt_cfg" in src
    assert 'xt_cfg = opd_cfg.get("cross_teacher_kl_weight", None)' in src
    # and it takes the ref-side prerequisite with it
    forced = src[src.index('xt_cfg = opd_cfg.get("cross_teacher_kl_weight", None)'):]
    assert "config.actor_rollout_ref.ref.student_indexed_topk = True" in forced


def test_the_grpo_trainer_attaches_what_the_reliability_needs():
    src = open("verl/trainer/ppo/opd_grpo_ray_trainer.py").read()
    for col in ("adv_row_value", "adv_group_informative", "adv_group_id"):
        assert f'batch.batch["{col}"]' in src, col
    # padding copies carry their original's uid, so leaving them in counts one
    # trajectory twice
    assert "PADDING_ROW_KEY" in src


# --------------------------------------------------------------------------- #
# the resumable state
# --------------------------------------------------------------------------- #
from verl.trainer.ppo.cross_teacher_kl_weight import (  # noqa: E402
    SIDECAR_NAME,
    load_sidecar_state,
    sidecar_state,
)


def _accumulated(seed=60):
    rms = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    b = _rms_batch(bs=2, resp=3, k=3, n_off=2, seed=seed)
    _fold_rms(rms, b, torch.tensor([0, 1]), torch.tensor([[1, 2], [0, 2]]))
    mean = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    mean.update(
        pre_weight=1.0 + torch.rand(2, 3), teacher_kl=torch.rand(2, 3) + 0.1,
        response_mask=torch.ones(2, 3), task_ids=torch.tensor([0, 1]),
    )
    adv = _adv([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    return rms, mean, adv


IDENTITY = {
    "base_path": "Qwen/Qwen3-1.7B",
    "temperature": 1.0,
    "task_order": TASKS,
    "teacher_paths": {"alfworld": "/t/a", "search": "/t/s", "webshop": "/t/w"},
}


def test_the_accumulated_state_survives_a_resume():
    """It is training state, not a diagnostic. Restoring the parameters and
    starting these from zero puts the run back at cold start -- weight 1 for two
    steps, then a scale rebuilt from a handful of positions -- with a step number
    in the hundreds on the logs and nothing in the metrics to say so."""
    rms, mean, adv = _accumulated()
    blob = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=adv.alpha_table(), identity=IDENTITY)

    r2 = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    m2 = PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu")
    a2 = AdvantageReliabilityStats(n_tasks=3, device="cpu")
    alpha = load_sidecar_state(blob, rms=r2, mean=m2, adv=a2, identity=IDENTITY)

    assert torch.allclose(rms.snapshot()["sigma"], r2.snapshot()["sigma"])
    assert torch.allclose(mean.snapshot()["mean"], m2.snapshot()["mean"])
    assert torch.allclose(adv.alpha_table(), a2.alpha_table())
    assert torch.allclose(alpha, adv.alpha_table())


@pytest.mark.parametrize("key,value", [
    ("base_path", "Qwen/Qwen3-4B"),
    ("temperature", 0.7),
    ("task_order", ["search", "alfworld", "webshop"]),
    # One teacher swapped. Every delta is log pi_teacher - log pi_base, so this
    # changes the scale, the corroboration and the reliability for every pair
    # that teacher appears in -- and leaves base_path matching.
    ("teacher_paths", {"alfworld": "/t/a", "search": "/t/OTHER", "webshop": "/t/w"}),
])
def test_a_resume_that_changes_what_the_numbers_mean_is_refused(key, value):
    """The shifts are relative to ONE base checkpoint, the log-probs were
    normalised at one temperature, and every matrix here is indexed by task
    order. Blending across a change would keep the arithmetic finite and the
    meaning gone."""
    rms, mean, adv = _accumulated()
    blob = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=None, identity=IDENTITY)
    with pytest.raises(AssertionError, match=key):
        load_sidecar_state(
            blob,
            rms=CumulativePolicyShiftRMS(n_tasks=3, device="cpu"),
            mean=PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu"),
            adv=AdvantageReliabilityStats(n_tasks=3, device="cpu"),
            identity={**IDENTITY, key: value},
        )


def test_a_checkpoint_that_does_not_record_an_identity_key_is_refused():
    """An absent key used to pass, which made the check strictly WEAKER the older
    the checkpoint was -- exactly backwards, since an old checkpoint is the one
    most likely to have been written under a different base or teacher set."""
    rms, mean, adv = _accumulated()
    partial = {k: v for k, v in IDENTITY.items() if k != "teacher_paths"}
    blob = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=None, identity=partial)
    with pytest.raises(AssertionError, match="does not record"):
        load_sidecar_state(
            blob,
            rms=CumulativePolicyShiftRMS(n_tasks=3, device="cpu"),
            mean=PreviousStepTaskKLWeightedMean(n_tasks=3, device="cpu"),
            adv=AdvantageReliabilityStats(n_tasks=3, device="cpu"),
            identity=IDENTITY,
        )


def test_one_reader_then_a_broadcast_rather_than_every_rank_opening_the_file():
    """Rank 0 writes it, so rank 0 reads it. Every rank opening the same path
    works on a shared filesystem and silently does not elsewhere: a rank that
    read a stale copy would continue from a different accumulated scale than its
    neighbours, and the weight is built rank-locally so nothing compares them."""
    actor = _actor_source()
    assert "def _read_sidecar_on_rank_zero" in actor
    block = actor[actor.index("def _read_sidecar_on_rank_zero"):]
    block = block[: block.index("def _xt_accumulate_reliability")]
    assert "get_rank() if dist_on else 0" in block
    assert "broadcast_object_list" in block
    assert "torch.load(" in block
    # the presence decision travels with the payload, so ranks cannot disagree
    # about whether to cold-start
    assert block.index("payload[0] = torch.load(") < block.index("broadcast_object_list")


def test_the_identity_written_by_the_worker_names_the_teachers():
    src = open("verl/workers/fsdp_workers.py").read()
    block = src[src.index("def _cross_teacher_identity"):]
    block = block[: block.index("def load_checkpoint")]
    for key in ("base_path", "temperature", "task_order", "teacher_paths"):
        assert f'"{key}"' in block, key
    inject = open("verl/trainer/main_opd.py").read()
    assert "cross_teacher_kl_weight.teacher_paths" in inject, (
        "the actor config is where the identity is written from, so the teachers "
        "have to reach it"
    )


def test_an_unknown_sidecar_version_is_refused_rather_than_guessed():
    rms, mean, adv = _accumulated()
    blob = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=None, identity=IDENTITY)
    blob["version"] = 99
    with pytest.raises(AssertionError, match="version"):
        load_sidecar_state(
            blob, rms=rms, mean=mean, adv=adv, identity=IDENTITY,
        )
    # Version 1 is not "old and probably fine": it is the advantage-gated
    # mechanism, whose corroboration, source term and normaliser are all
    # different quantities. Resuming would divide this rule's W~ by the other
    # rule's lagged mean and no metric would show it.
    blob["version"] = 1
    with pytest.raises(AssertionError, match="version"):
        load_sidecar_state(blob, rms=rms, mean=mean, adv=adv, identity=IDENTITY)
    fresh = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=None, identity=IDENTITY)
    fresh["mechanism"] = "something_else"
    with pytest.raises(AssertionError, match="mechanism"):
        load_sidecar_state(fresh, rms=rms, mean=mean, adv=adv, identity=IDENTITY)


def test_the_worker_writes_it_beside_the_checkpoint_from_rank_zero_only():
    src = open("verl/workers/fsdp_workers.py").read()
    assert "self._save_cross_teacher_sidecar(local_path)" in src
    block = src[src.index("def _save_cross_teacher_sidecar"):src.index("def _cross_teacher_identity")]
    assert "dist.get_rank() != 0" in block, "one writer"
    # The filename comes from the module rather than being spelled twice: a
    # writer and a reader that disagree about it fail as a silent cold start.
    assert "SIDECAR_NAME" in src and SIDECAR_NAME not in src
    # ... and the restore is deferred to where the accumulators exist
    assert "self.actor.cross_teacher_sidecar_path = " in src
    actor = _actor_source()
    assert "load_sidecar_state(" in actor
    assert "self.cross_teacher_sidecar_path = None" in actor, "consumed once"


# --------------------------------------------------------------------------- #
# the two runs
# --------------------------------------------------------------------------- #
import os as _os  # noqa: E402

hydra = pytest.importorskip("hydra")
yaml = pytest.importorskip("yaml")

from hydra import compose, initialize_config_dir  # noqa: E402

from tests.trainer.test_run_script_overrides_compose import _overrides  # noqa: E402

REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = _os.path.join(REPO, "verl", "trainer", "config")
TREATMENT = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_qwen3.sh"
CONTROL = "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh"
IDENTITY_KEYS = {
    "trainer.expected_config", "trainer.project_name", "trainer.experiment_name",
    "trainer.default_local_dir", "trainer.val_instance_log_dir", "trainer.sign_token_dump_dir",
}


def _flat(cfg, prefix=""):
    from omegaconf import DictConfig

    out = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, DictConfig):
            out.update(_flat(value, prefix=f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _effective(script, home="/opt/home/tester"):
    _os.environ["HOME"] = home
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return _flat(compose(config_name="ppo_trainer", overrides=list(_overrides(script))))


def test_the_control_differs_from_the_treatment_in_the_weight_and_nothing_else():
    """It is not an ablation -- it is the only run this arm can be compared
    against. The coefficient moved 1.0 -> 0.01, so every earlier result differs
    from this one before the mechanism does."""
    a, b = _effective(TREATMENT), _effective(CONTROL)
    differing = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert differing - IDENTITY_KEYS == {
        "algorithm.opd.cross_teacher_kl_weight.enable"
    }, sorted(differing - IDENTITY_KEYS)


def test_the_two_runs_do_not_share_a_checkpoint_directory():
    """resume_mode defaults to auto, so a shared default_local_dir means the
    second run RESUMES FROM the first and reports it under its own name."""
    a, b = _effective(TREATMENT), _effective(CONTROL)
    for key in ("trainer.default_local_dir", "trainer.val_instance_log_dir"):
        assert a[key] != b[key], key


def _composed(script, home="/opt/home/tester"):
    _os.environ["HOME"] = home
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=list(_overrides(script)))


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_each_run_satisfies_its_own_intent_lock(script):
    """The lock exists to fail in seconds instead of hours, and only does that
    if the script it guards passes it. Run through the same injection and the
    same checker the entry point uses -- several pinned keys, the actor-side
    mirrors among them, do not exist until that injection has run."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config
    from verl.utils.expected_config import enforce_expected_config

    cfg = _composed(script)
    inject_opd_grpo_config(cfg)
    lock = cfg.trainer.expected_config
    assert lock.startswith("examples/opd_grpo_trainer/"), lock
    assert enforce_expected_config(cfg, _os.path.join(REPO, lock), tag="test:xt") > 0


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_the_injection_carries_the_block_onto_the_actor(script):
    """The weight is built inside the actor's forward, so a block that stayed
    under algorithm.opd would reach nothing and the arm would train plain."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = _composed(script)
    inject_opd_grpo_config(cfg)
    xt = cfg.actor_rollout_ref.actor.cross_teacher_kl_weight
    assert bool(xt.enable) == bool(cfg.algorithm.opd.cross_teacher_kl_weight.enable)
    assert xt.base_path == cfg.algorithm.opd.cross_teacher_kl_weight.base_path
    if bool(xt.enable):
        # The four models are read at ids nobody knows until that forward picks
        # a support, so the ref side has to keep hidden states.
        assert cfg.actor_rollout_ref.ref.student_indexed_topk is True


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_the_arm_independent_attribution_is_on_in_both_runs(script):
    """The one family that runs in the control. Off in either script and there
    is no per-token comparison to make: the control publishes no kl_weight/*
    series at all, so opd/* is the entire overlap between the two runs' token
    tables."""
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    cfg = _composed(script)
    inject_opd_grpo_config(cfg)
    assert bool(cfg.algorithm.opd.opd_attribution.enable) is True
    # It is built where the student's top-k exists, which is the actor's
    # forward, so a block that stayed under algorithm.opd would reach nothing.
    assert bool(cfg.actor_rollout_ref.actor.opd_attribution.enable) is True
    # Its own prerequisites, which are the whole of what it needs -- no base
    # policy and no off-task teachers, which is why the control can run it.
    assert bool(cfg.actor_rollout_ref.actor.student_indexed_topk) is True
    assert cfg.actor_rollout_ref.actor.teacher_kl_loss_type == "topk_kl"
    assert bool(cfg.actor_rollout_ref.actor.use_teacher_kl_loss) is True
    # And somewhere for the ranked rows to land.
    assert cfg.trainer.sign_token_dump_dir


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_the_coefficient_is_pinned_at_the_value_that_rules_out_the_old_baselines(script):
    cfg = _effective(script)
    # Authored under algorithm.opd with the other scientific knobs; main_opd
    # copies it onto the actor at startup, which is why the lock -- validated
    # AFTER that injection -- is where the actor-side value is checked.
    assert cfg["algorithm.opd.kl_loss_coef"] == 0.01
    assert cfg["actor_rollout_ref.actor.pg_loss_coef"] == 1.0
    lock = yaml.safe_load(open(_os.path.join(REPO, cfg["trainer.expected_config"])))
    assert lock["actor_rollout_ref.actor.teacher_kl_loss_coef"] == 0.01, (
        "the coefficient is the reason no earlier run is a baseline; it belongs in the lock"
    )


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_the_old_mechanism_is_off_and_its_knobs_are_absent(script):
    """Two mechanisms for one signal. The trainer refuses both at once, and a
    recipe that carried a stale agree_weight would look like it still meant
    something."""
    cfg = _effective(script)
    assert not cfg.get("algorithm.opd.sign_weight.enable", False)
    # The ARGUMENTS, not the prose: the header explains what the old table was
    # and why this arm has none, and a test that forbade the words would forbid
    # saying so.
    args = [
        ln for ln in open(_os.path.join(REPO, script)).read().splitlines()
        if not ln.lstrip().startswith("#") and "=" in ln
    ]
    for knob in ("agree_weight", "agree_neg_weight", "disagree_weight", "deadzone"):
        assert not [ln for ln in args if knob in ln], knob


@pytest.mark.parametrize("script", [TREATMENT, CONTROL])
def test_the_run_takes_the_support_the_mechanism_is_defined_on(script):
    """Every model is measured on the STUDENT's top-k, and the ref side has to
    keep hidden states because those ids do not exist until the actor's forward."""
    cfg = _effective(script)
    assert cfg["actor_rollout_ref.actor.student_indexed_topk"] is True
    assert cfg["algorithm.opd.kl_loss_type"] == "topk_kl"


def test_the_treatment_pins_the_base_checkpoint_the_whole_scale_is_measured_against():
    cfg = _effective(TREATMENT)
    lock = yaml.safe_load(open(_os.path.join(REPO, cfg["trainer.expected_config"])))
    assert lock["algorithm.opd.cross_teacher_kl_weight.base_path"] == cfg["actor_rollout_ref.model.path"], (
        "the shifts are relative to the checkpoint the teachers were fine-tuned FROM; "
        "the student starts there too, which is what makes step 0 the zero of the ladder"
    )


# --------------------------------------------------------------------------- #
# the actor's reliability pass, driven end to end
# --------------------------------------------------------------------------- #
def test_the_reliability_pass_runs_on_realistic_shapes_and_files_the_right_cells():
    """The one new path a CPU suite can otherwise only read.

    It looks the EMITTED token up on its own -- one extra id per model, out of
    the same cached hidden states -- so its shapes differ from everything else
    here: a width-1 support against the width-k one the weight is built on. A
    broadcast error between them is invisible until a GPU run reaches
    update_actor, which is ten minutes in.
    """
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    bs, resp, k, n_off, vocab = 4, 5, 6, 2, 40
    torch.manual_seed(70)

    def lp(width):
        return torch.log_softmax(torch.randn(bs, resp, width + 4), dim=-1)[..., :width]

    actor = object.__new__(DataParallelPPOActor)
    actor._xt_adv = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    # The support-width lookups the real actor makes against its teacher cache.
    actor._teacher_logprobs_at = lambda **kw: lp(kw["ids"].size(-1))
    actor._cross_teacher_planes = lambda data, ids: (
        lp(ids.size(-1)),
        torch.stack([lp(ids.size(-1)) for _ in range(n_off)], dim=-1),
    )

    on, base = lp(k), lp(k)
    off = torch.stack([lp(k) for _ in range(n_off)], dim=-1)
    task_ids = torch.tensor([0, 0, 1, 2])
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    built = build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
        alpha_table=torch.zeros(3, 3),
        normalizer={"mean": torch.full((3,), 1.1), "valid": torch.ones(3, dtype=torch.bool)},
    )
    support_ids = torch.randint(0, vocab, (bs, resp, k))
    data = {
        "responses": torch.randint(0, vocab, (bs, resp)),
        "teacher_cache_ids": torch.arange(bs),
        "input_ids": torch.randint(0, vocab, (bs, resp + 3)),
        "attention_mask": torch.ones(bs, resp + 3, dtype=torch.long),
        "sign_off_tasks": planes,
        "adv_row_value": torch.tensor([1.0, -1.0, 0.5, -0.5]),
        "adv_group_informative": torch.ones(bs, dtype=torch.bool),
        "adv_group_id": torch.tensor([0, 0, 1, 1]),
    }
    counter = torch.zeros(2, dtype=torch.float64)
    DataParallelPPOActor._xt_accumulate_reliability(
        actor, data=data, built=built, student_topk_logprob=lp(k),
        support_ids=support_ids, response_mask=torch.ones(bs, resp),
        task_ids=task_ids, diag=(torch.ones(3), torch.ones(3, dtype=torch.bool)),
        outside_counter=counter,
    )
    actor._xt_adv.all_reduce()

    got = actor._xt_adv.alpha(task_names=TASKS)
    # Rows of task 0 name search and webshop as their sources; a pair nobody's
    # rows named must have no cell at all.
    assert ("alfworld", "search") in got and ("alfworld", "webshop") in got
    assert got[("alfworld", "search")]["n"] == 2.0
    for row in got.values():
        assert row["alpha"] >= 0.0
        assert row["rho"] is None or math.isfinite(row["rho"])
    assert float(counter[1]) == bs * resp
    assert 0.0 <= float(counter[0]) <= float(counter[1])


def test_an_arm_with_no_advantages_leaves_the_reliability_untouched():
    """Pure OPD is a configuration, not a failure: alpha stays 0 and the
    corroboration channel runs alone."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor = object.__new__(DataParallelPPOActor)
    actor._xt_adv = AdvantageReliabilityStats(n_tasks=3, device="cpu")
    DataParallelPPOActor._xt_accumulate_reliability(
        actor, data={"responses": torch.zeros(1, 1, dtype=torch.long)}, built=None,
        student_topk_logprob=None, support_ids=None, response_mask=None,
        task_ids=torch.zeros(1, dtype=torch.long), diag=None,
        outside_counter=torch.zeros(2, dtype=torch.float64),
    )
    actor._xt_adv.all_reduce()
    assert float(actor._xt_adv.alpha_table().abs().sum()) == 0.0


def test_the_statistics_are_collected_on_the_first_ppo_epoch_only():
    """The weight is needed on every epoch; the statistics are not. Later epochs
    re-visit the same rows against a student that has already moved, so folding
    them in counts each trajectory once per epoch and mixes two policies into
    one cumulative scale."""
    import ast

    src = _update_policy_source()
    assert "xt_collect = epoch == 0" in src
    tree = ast.parse(src)

    def guarded_calls(node, guards=()):
        """Every call in the tree, paired with the `if` tests enclosing it."""
        if isinstance(node, ast.If):
            test = ast.dump(node.test)
            for child in node.body:
                yield from guarded_calls(child, guards + (test,))
            for child in node.orelse:
                yield from guarded_calls(child, guards)
            return
        if isinstance(node, ast.Call):
            yield node, guards
        for child in ast.iter_child_nodes(node):
            yield from guarded_calls(child, guards)

    def name_of(call):
        f = call.func
        return ast.unparse(f) if hasattr(ast, "unparse") else getattr(f, "attr", "")

    collected = {}
    for call, guards in guarded_calls(tree):
        collected.setdefault(name_of(call), []).append(guards)

    for accumulate in (
        "self._xt_rms.update",
        "self._xt_accumulate_reliability",
        "self._xt_mean.update",
        "xt_position_stats.update",
        "xt_state_stats.update",
        "xt_pair_stats.update",
    ):
        sites = collected.get(accumulate, [])
        assert sites, accumulate
        for guards in sites:
            assert any("xt_collect" in g for g in guards), accumulate
    # ... and the weight itself is NOT gated on it, or later epochs would train
    # on a different objective from the first.
    for guards in collected["build_position_weight"]:
        assert not any("xt_collect" in g for g in guards)


# --------------------------------------------------------------------------- #
# what R > 1 does to a cumulative buffer
# --------------------------------------------------------------------------- #
class _FakeCollective:
    """Emulate ``all_reduce(SUM)`` over R ranks that ran identical data.

    Every rank holds the same buffer here, so a SUM across R of them is a
    multiply by R. That is exactly the operation the real collective performs,
    and it is enough to catch the bug it hides: reducing a CUMULATIVE buffer
    gives ``Q_n = R*Q_{n-1} + sum_r delta_n`` instead of ``Q_{n-1} + sum_r
    delta_n``. The failure looks like success -- Q and N inflate together so
    sigma does not blow up, it freezes -- so nothing short of this catches it.
    """

    def __init__(self, ranks):
        self.ranks = ranks
        self._saved = None

    def __enter__(self):
        import verl.trainer.ppo.cross_teacher_kl_weight as mod

        self._saved = mod.torch.distributed
        ranks = self.ranks

        class _Dist:
            ReduceOp = type("ReduceOp", (), {"SUM": "sum"})

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def all_reduce(tensor, op=None):
                tensor.mul_(ranks)

        mod.torch.distributed = _Dist
        return self

    def __exit__(self, *exc):
        import verl.trainer.ppo.cross_teacher_kl_weight as mod

        mod.torch.distributed = self._saved
        return False


def _rms_step(stats, gain, n_pos=1):
    lp = torch.log(torch.full((1, n_pos, 2), 0.4))
    stats.update(
        shifts={
            "on": torch.full((1, n_pos, 2), float(gain)),
            "off": torch.zeros(1, n_pos, 2, 1),
            "tail_on": torch.zeros(1, n_pos),
            "tail_off": torch.zeros(1, n_pos, 1),
        },
        student_logprob=lp, response_mask=torch.ones(1, n_pos),
        task_ids=torch.zeros(1, dtype=torch.long),
        off_plane_tasks=torch.zeros(1, 1, dtype=torch.long),
    )


def test_the_cumulative_scale_is_not_reduced_a_second_time_each_step():
    """THE regression for the two-rank bug.

    A single-rank suite cannot see it: all_reduce is the identity at R=1, and
    every existing test runs there. At R=2 the shipped-before-this version
    double-counted the running total every step, so step 1 came to carry half
    the cumulative by step 8 and the scale stopped tracking the run.
    """
    gains = [1.0, 3.0, 5.0, 7.0]
    single = CumulativePolicyShiftRMS(n_tasks=1, device="cpu")
    for g in gains:
        _rms_step(single, g)
        _rms_step(single, g)          # two ranks' worth of rows, one process
        single.all_reduce()

    with _FakeCollective(ranks=2):
        dual = CumulativePolicyShiftRMS(n_tasks=1, device="cpu")
        for g in gains:
            _rms_step(dual, g)        # each rank contributes its own half
            dual.all_reduce()

    assert float(dual.n[0]) == float(single.n[0])
    assert float(dual.q[0]) == pytest.approx(float(single.q[0]), rel=1e-12)
    assert float(dual.diagonal()[0][0]) == pytest.approx(float(single.diagonal()[0][0]), rel=1e-12)


def test_every_step_keeps_its_own_weight_in_the_cumulative_scale():
    """The symptom the double reduce produced, stated directly: with it, the
    first step's contribution dominates exponentially and sigma freezes."""
    with _FakeCollective(ranks=2):
        stats = CumulativePolicyShiftRMS(n_tasks=1, device="cpu")
        _rms_step(stats, 1.0)
        stats.all_reduce()
        early = float(stats.diagonal()[0][0])
        for _ in range(6):
            _rms_step(stats, 9.0)
            stats.all_reduce()
        late = float(stats.diagonal()[0][0])
    # Six steps of a 9x louder teacher must move the scale most of the way there.
    assert late > 0.8 * 9.0, f"the scale froze near its first step ({early} -> {late})"


def test_the_reliability_moments_are_not_reduced_a_second_time_each_step():
    def fold(stats, a, s, groups):
        n = len(a)
        stats.update(
            advantage=torch.tensor(a), support_score=torch.tensor(s).reshape(n, 1),
            on_support_score=torch.zeros(n), length=torch.ones(n),
            informative=torch.ones(n, dtype=torch.bool),
            task_ids=torch.zeros(n, dtype=torch.long), off_plane_tasks=torch.full((n, 1), 1),
            group_ids=torch.tensor(groups),
        )

    steps = [([-1.0, 1.0], [-2.0, 2.0], [0, 0]), ([-1.0, 1.0], [1.0, -1.0], [0, 0])]
    single = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    for a, s, g in steps:
        fold(single, a, s, g)
        fold(single, a, s, g)
        single.all_reduce()

    with _FakeCollective(ranks=2):
        dual = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
        for a, s, g in steps:
            fold(dual, a, s, g)
            dual.all_reduce()

    assert torch.allclose(dual.buf, single.buf, rtol=1e-12)
    assert torch.allclose(dual.between, single.between, rtol=1e-12)


def test_group_ids_from_different_steps_are_never_pooled():
    """``adv_group_id`` is dense and re-issued from zero every step, so a buffer
    that survived the step would add step 1's group 0 to step 2's unrelated
    group 0 and then square the sum. Merging groups understates the
    between-group term, which overstates the within-group variance and pulls
    every correlation toward zero -- undoing the centring it exists for."""
    stats = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)

    def fold(s_values):
        n = len(s_values)
        stats.update(
            advantage=torch.tensor([-1.0, 1.0]), support_score=torch.tensor(s_values).reshape(n, 1),
            on_support_score=torch.zeros(n), length=torch.ones(n),
            informative=torch.ones(n, dtype=torch.bool),
            task_ids=torch.zeros(n, dtype=torch.long), off_plane_tasks=torch.full((n, 1), 1),
            group_ids=torch.tensor([0, 0]),
        )

    # Two steps whose "group 0" are different prompts with wildly different
    # offsets. Pooled, their summed score nearly cancels and the between-group
    # term collapses; kept apart, each offset is removed on its own.
    fold([-1.0, 1.0])
    stats.all_reduce()
    fold([999.0, 1001.0])
    stats.all_reduce()

    got = stats.alpha(task_names=TASKS)[("alfworld", "search")]
    assert got["rho"] == pytest.approx(1.0, abs=1e-6), (
        "each step's own group offset must be divided out, not merged with the other's"
    )


def test_the_between_group_correction_survives_a_resume():
    stats = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    stats.update(
        advantage=torch.tensor([-1.0, 1.0]), support_score=torch.tensor([[9.0], [11.0]]),
        on_support_score=torch.zeros(2), length=torch.ones(2),
        informative=torch.ones(2, dtype=torch.bool),
        task_ids=torch.zeros(2, dtype=torch.long), off_plane_tasks=torch.full((2, 1), 1),
        group_ids=torch.tensor([0, 0]),
    )
    stats.all_reduce()
    other = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    other.load_state_dict(stats.state_dict())
    assert torch.allclose(stats.between, other.between)
    assert torch.allclose(stats.grouped, other.grouped)
    assert other.alpha(task_names=TASKS)[("alfworld", "search")]["rho"] == pytest.approx(
        stats.alpha(task_names=TASKS)[("alfworld", "search")]["rho"], abs=1e-12
    )


def test_rendering_before_the_step_boundary_is_refused_rather_than_stale():
    """A total that has not absorbed this step yet is a silently wrong number,
    and silently wrong statistics are this module's whole failure mode."""
    stats = CumulativePolicyShiftRMS(n_tasks=1, device="cpu")
    _rms_step(stats, 1.0)
    with pytest.raises(AssertionError, match="unreduced step delta"):
        stats.diagonal()
    stats.all_reduce()
    assert float(stats.diagonal()[0][0]) > 0


# --------------------------------------------------------------------------- #
# non-finite: protect the loss, then fail loudly
# --------------------------------------------------------------------------- #
def test_a_non_finite_teacher_neutralises_the_position_and_is_counted():
    """It must not reach the loss, and it must not pass for silence either.

    A non-finite weight multiplied into the KL is a non-finite gradient and a
    poisoned optimizer state. Forcing W = 1 stops that; the COUNT is what stops
    a corrupted teacher from looking like a quiet mechanism for the next
    thousand steps.
    """
    on = torch.log(torch.full((2, 1, 2), 0.4))
    bad = on.clone()
    bad[1, 0, 0] = float("nan")
    shifts = compute_raw_policy_shifts(
        on_task_logprob=on, off_task_logprobs=bad.unsqueeze(-1).expand(2, 1, 2, 2).contiguous(),
        base_logprob=on,
    )
    got = build_position_weight(
        shifts=shifts, on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=torch.zeros(2, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2], [1, 2]]), diag=torch.ones(3),
        diag_valid=torch.ones(3, dtype=torch.bool), alpha_table=torch.ones(3, 3),
        normalizer={"mean": torch.ones(3), "valid": torch.ones(3, dtype=torch.bool)},
    )
    assert torch.isfinite(got["weight"]).all(), "the loss never sees it"
    assert torch.isfinite(got["pre_weight"]).all()
    assert int(got["nonfinite"]) > 0, "and it is not mistaken for silence"


def test_a_non_finite_normaliser_is_invalid_rather_than_a_nan_in_the_loss():
    """``den > 0`` alone lets a NaN numerator through: clamp propagates NaN and
    torch.where selects the valid branch, so it rides mu into the weight."""
    m = PreviousStepTaskKLWeightedMean(n_tasks=1, device="cpu")
    m.num[0] = float("nan")
    m.den[0] = 5.0
    snap = m.snapshot()
    assert not bool(snap["valid"][0])
    assert snap["nonfinite"] == 1

    lp = torch.log(torch.full((1, 1, 2), 0.4))
    got = build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=lp, off_task_logprobs=lp.unsqueeze(-1).expand(1, 1, 2, 2).contiguous(),
            base_logprob=lp,
        ),
        on_task_logprob=lp, student_logprob=_student_like(lp), response_mask=None, task_ids=torch.zeros(1, dtype=torch.long),
        off_plane_tasks=torch.tensor([[0, 0]]), diag=torch.ones(1),
        diag_valid=torch.ones(1, dtype=torch.bool), alpha_table=torch.zeros(1, 1),
        normalizer=snap,
    )
    assert torch.isfinite(got["weight"]).all()
    assert float(got["weight"]) == pytest.approx(1.0)


def test_the_step_fails_on_any_non_finite_and_says_which_stage():
    from verl.trainer.ppo.cross_teacher_kl_weight import assert_all_finite

    assert_all_finite({"weight": 0, "cumulative_scale": 0, "normaliser": 0})
    with pytest.raises(AssertionError, match="cumulative_scale=3"):
        assert_all_finite({"weight": 0, "cumulative_scale": 3, "normaliser": 0})


def test_the_failure_is_raised_after_the_collectives_not_inside_them():
    """One rank leaving a collective its neighbours are still waiting on hangs
    the job instead of ending it."""
    src = _update_policy_source()
    tail = src[src.rindex("        if xt_on:"):]
    reduce_at = tail.rindex("torch.distributed.all_reduce(_t")
    assert tail.index("assert_all_finite(") > reduce_at


# --------------------------------------------------------------------------- #
# token attribution and the gradient-interference metrics
# --------------------------------------------------------------------------- #
from verl.trainer.ppo.cross_teacher_kl_weight import (  # noqa: E402
    GRAD_TERMS,
    gradient_metrics,
    logit_gradient_terms,
    per_candidate_shift,
)


def test_the_per_candidate_shift_is_the_summand_of_the_state_table():
    """One definition, three tables. The per-state table, the per-token table
    and the source attribution all decompose the same nats, and computing it
    three times is how three views of one number drift apart."""
    got, _ = _built(bs=3, resp=4, seed=46)
    kl = torch.rand(*got["weight"].shape) + 0.1
    per_cand = per_candidate_shift(got, kl)
    state = state_shift_terms(got, kl)
    attributed = sum(state[t] for t in STATE_TERMS if t != "shift_norm_offset")
    assert torch.allclose(per_cand.sum(dim=-1), attributed, atol=1e-5)


def test_the_analytic_opd_gradient_matches_autograd():
    """The claim that lets the interference metric cost no second backward -- and
    a second backward would not merely be expensive, it would make a diagnostic
    able to change the run it describes."""
    from verl.trainer.ppo.core_algos import topk_kl_per_token

    torch.manual_seed(80)
    k, V, W, coef = 4, 9, 1.7, 0.01
    logits = torch.randn(1, 1, V, requires_grad=True)
    teacher = torch.log_softmax(torch.randn(1, 1, V), -1)[..., :k]

    lp = torch.log_softmax(logits, -1)
    kl = topk_kl_per_token(lp[..., :k], teacher)
    (coef * W * kl).sum().backward()
    auto = -logits.grad[0, 0].clone()

    p_s = lp[..., :k].detach().exp()
    f = lp[..., :k].detach() - teacher
    analytic = coef * W * p_s * (kl.detach().unsqueeze(-1) - f)
    assert torch.allclose(analytic[0, 0], auto[:k], atol=1e-6)

    got = logit_gradient_terms(
        student_logprob=lp[..., :k].detach(), teacher_logprob=teacher,
        weight=torch.full((1, 1), W), teacher_kl=kl.detach(),
        pg_grad_coef=torch.zeros(1, 1), sampled_onehot=torch.zeros(1, 1, k), coef=coef,
    )
    # the tail bucket carries the rest of the vocabulary, so the norm covers it
    assert float(got["g_opd_sq"]) == pytest.approx(
        float((analytic ** 2).sum() + auto[k:].sum() ** 2), rel=1e-4
    )


def test_the_analytic_policy_gradient_matches_autograd_at_ratio_one():
    """The logit-space half of the closed form: given the loss's derivative with
    respect to log pi(y), the descent direction on logit v is that coefficient
    times (1[v=y] - p). At ratio 1 and outside the clip the coefficient is -A,
    which is the case checked here; policy_loss_gradient_coef supplies it
    everywhere else, and its own autograd test covers the branches."""
    torch.manual_seed(81)
    k, V, A, slot = 4, 9, 2.5, 2
    logits = torch.randn(1, 1, V, requires_grad=True)
    lp = torch.log_softmax(logits, -1)
    (-A * lp[0, 0, slot]).backward()
    auto = -logits.grad[0, 0]

    onehot = torch.zeros(1, 1, k)
    onehot[0, 0, slot] = 1.0
    analytic = A * (onehot[0, 0] - lp[..., :k].detach().exp()[0, 0])
    assert torch.allclose(analytic, auto[:k], atol=1e-5)


def test_the_policy_coefficient_is_per_position_not_per_row():
    """It used to be the row's advantage, broadcast. The clipped objective's
    derivative is not constant along a row -- one token can be past the bound
    and the next inside it -- so a per-row quantity cannot represent it, and a
    position clipped off has to read as zero rather than as the row's A."""
    bs, resp, k = 3, 5, 4
    lp = torch.log_softmax(torch.randn(bs, resp, k + 3), -1)[..., :k]
    coefs = torch.randn(bs, resp)
    coefs[:, 2] = 0.0  # this column is clipped off in every row
    got = logit_gradient_terms(
        student_logprob=lp, teacher_logprob=lp,
        weight=torch.ones(bs, resp), teacher_kl=torch.rand(bs, resp),
        pg_grad_coef=coefs, sampled_onehot=torch.zeros(bs, resp, k), coef=0.01,
    )
    for name in GRAD_TERMS:
        assert got[name].shape == (bs, resp), name
    assert torch.allclose(got["g_grpo_sq"][:, 2], torch.zeros(bs)), "clipped off"
    assert float(got["g_grpo_sq"][:, [0, 1, 3, 4]].sum()) > 0


def test_the_interference_metrics_are_a_ratio_and_a_cosine():
    """The ratio says how much of the update the OPD term is responsible for at
    all -- at coefficient 0.01 that is the first thing a reader wants and the
    last thing a loss curve shows."""
    bs, resp, k = 2, 3, 4
    torch.manual_seed(82)
    lp = torch.log_softmax(torch.randn(bs, resp, k + 3), -1)[..., :k]
    stats = ScopeTermStats(names=GRAD_TERMS, n_tasks=3, device="cpu")
    stats.update(
        logit_gradient_terms(
            student_logprob=lp, teacher_logprob=torch.log_softmax(torch.randn(bs, resp, k), -1),
            weight=torch.ones(bs, resp), teacher_kl=torch.rand(bs, resp) + 0.1,
            pg_grad_coef=torch.tensor([[1.0, -1.0, 0.0], [-2.0, 0.5, 1.5]]),
            sampled_onehot=torch.nn.functional.one_hot(
                torch.randint(0, k, (bs, resp)), k
            ).float(),
            coef=0.01,
        ),
        response_mask=torch.ones(bs, resp), task_ids=torch.tensor([0, 1]),
    )
    m = gradient_metrics(stats.sums(task_names=TASKS))
    assert m["kl_weight/grpo/grad_norm_ratio"] > 0
    assert -1.0 - 1e-6 <= m["kl_weight/grpo/grad_cosine"] <= 1.0 + 1e-6
    for name in m:
        assert "max" not in name and "min" not in name, name


def test_a_perfectly_aligned_pair_reads_cosine_one():
    stats = ScopeTermStats(names=GRAD_TERMS, n_tasks=0, device="cpu")
    stats.update(
        # The channel columns are zero here: this pins the POOLED cosine, and a
        # channel that allocated nothing must render no cosine of its own.
        {"g_opd_sq": torch.tensor([[4.0]]), "g_grpo_sq": torch.tensor([[9.0]]),
         "g_dot": torch.tensor([[6.0]]),
         **{name: torch.zeros(1, 1) for name in GRAD_TERMS
            if name not in ("g_opd_sq", "g_grpo_sq", "g_dot")}},
        response_mask=torch.ones(1, 1), task_ids=None,
    )
    m = gradient_metrics(stats.sums())
    assert m["kl_weight/grpo/grad_cosine"] == pytest.approx(1.0)
    assert m["kl_weight/grpo/grad_norm_ratio"] == pytest.approx(2.0 / 3.0)
    assert "kl_weight/grpo/shared_grad_cosine" not in m
    assert "kl_weight/grpo/source_grad_cosine" not in m


def test_the_token_tables_are_driven_from_the_new_arms_own_switches():
    """The existing dumps are gated on the sign arm's config, so without this
    the run can say a source raised a task's KL by so many nats and cannot name
    one token it did it at."""
    src = _update_policy_source()
    assert 'xt_token_cfg = (xt_cfg.get("token_stats", None) or {})' in src
    assert 'xt_event_cfg = (xt_cfg.get("event_dump", None) or {})' in src
    for ctor in ("TokenStateCounts(", "SignPairTokens(", "SignEventSamples("):
        assert src.count(ctor) >= 2, f"{ctor} is not built for the new arm"
    # the tables read the standardized shifts, so their labels and deadzone are
    # in RMS units and comparable across teachers
    actor = _actor_source()
    helper = actor[actor.index("def _xt_token_tables"):]
    helper = helper[: helper.index("def _read_sidecar_on_rank_zero")]
    assert 'base_logprob=zero_base' in helper
    assert 'mass=built["mass"]' in helper, (
        "a standardized shift does not exponentiate to a probability"
    )
    assert "per_candidate_shift(" in helper


def test_the_gradient_metric_is_collected_where_the_ratio_is_one():
    src = _update_policy_source()
    # Anchored on the call, not on the accumulator it feeds: the same columns go
    # to the task cut and the role cut, so the terms are built once and the
    # coefficient has to be right at the one place that builds them.
    call = src[src.index("logit_gradient_terms("):]
    call = call[: call.index("\n                                )")]
    assert "coef=float(self.config.get(" in call, "the real OPD coefficient, not 1"
    # Epoch 0 only: a later epoch's clipping makes the closed form for the
    # policy gradient an approximation rather than an identity.
    guard = src[: src.index("logit_gradient_terms(")]
    assert "if xt_built is not None and xt_collect:" in guard
    # Both cuts read the SAME columns. A second call would be a second closed
    # form that could drift from the one the task cut reports.
    assert src.count("logit_gradient_terms(") == 1
    assert "xt_role_grad_stats.update(" in src


# --------------------------------------------------------------------------- #
# the analysis cuts: role, turn, weight shape, token identity over time
# --------------------------------------------------------------------------- #
def test_the_position_scope_accumulator_partitions_by_position_not_by_row():
    """The whole reason it is a second class. ScopeTermStats files a row's total
    once, under the row's task; a role changes several times inside one row, and
    filing the row under its first token's role would report the arm acting on
    reasoning wherever a response happened to open with <think>."""
    from verl.trainer.ppo.cross_teacher_kl_weight import PositionScopeTermStats

    bs, resp = 3, 6
    acc = PositionScopeTermStats(names=("a", "b"), n_scopes=4, device="cpu")
    a = torch.arange(bs * resp, dtype=torch.float32).reshape(bs, resp)
    b = torch.ones(bs, resp)
    mask = torch.ones(bs, resp)
    mask[1, 4:] = 0.0
    scope = torch.arange(bs * resp).reshape(bs, resp) % 4
    acc.update({"a": a, "b": b}, response_mask=mask, scope_ids=scope)

    got = acc.sums(scope_names=("s0", "s1", "s2", "s3"), include_pooled=True)
    # Pooled equals the masked total, and the four scopes partition it exactly.
    assert got[None]["a"] == pytest.approx(float((a * mask).sum()))
    assert got[None]["n"] == pytest.approx(float(mask.sum()))
    parts = sum(v["a"] for k, v in got.items() if k is not None)
    assert parts == pytest.approx(got[None]["a"])
    # And a per-scope sum is what a brute-force selection gives.
    for s in range(4):
        want = float((a * mask * (scope == s)).sum())
        assert got[f"s{s}"]["a"] == pytest.approx(want)


def test_the_position_scope_accumulator_files_an_out_of_range_scope_pooled_only():
    """An untagged position is a real position; inventing a scope for it would
    put it in a bucket a reader would then attribute to."""
    from verl.trainer.ppo.cross_teacher_kl_weight import PositionScopeTermStats

    acc = PositionScopeTermStats(names=("a",), n_scopes=2, device="cpu")
    a = torch.ones(1, 4)
    scope = torch.tensor([[0, 1, -1, 9]])
    acc.update({"a": a}, response_mask=torch.ones(1, 4), scope_ids=scope)
    got = acc.sums(scope_names=("s0", "s1"), include_pooled=True)
    assert got[None]["a"] == pytest.approx(4.0)
    assert got["s0"]["a"] == pytest.approx(1.0)
    assert got["s1"]["a"] == pytest.approx(1.0)
    assert sum(v["a"] for k, v in got.items() if k is not None) == pytest.approx(2.0)


def test_the_role_cut_renders_through_the_same_functions_as_the_task_cut():
    """A role row and a task row of the same metric have to be the same
    arithmetic over different positions. Sharing the renderer is how that is
    guaranteed rather than reviewed."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        PositionScopeTermStats,
        ROLE_SCOPE_NAMES,
    )

    got, _ = _built()
    kl = torch.rand(*got["weight"].shape) + 0.1
    acc = PositionScopeTermStats(
        names=POSITION_TERMS, n_scopes=len(ROLE_SCOPE_NAMES), device="cpu"
    )
    roles = torch.arange(got["weight"].numel()).reshape(got["weight"].shape) % len(ROLE_SCOPE_NAMES)
    acc.update(position_terms(got, kl), response_mask=torch.ones_like(kl), scope_ids=roles)
    out = position_weight_metrics(
        acc.sums(scope_names=ROLE_SCOPE_NAMES), prefix="kl_weight/role"
    )
    assert "kl_weight/role/reasoning/effect/kl_scale" in out
    assert "kl_weight/role/env_action/position/w_mean" in out
    # The pooled scope is not published: it would be a second key for the number
    # the task-scoped accumulator already reports at the top level.
    assert not any(k.startswith("kl_weight/role/effect") for k in out)


def test_the_weight_histogram_thresholds_are_counts_not_interpolations():
    """Every reported threshold is a bucket edge, so "above t" is a sum of whole
    buckets and matches a brute-force count exactly. An interpolated share would
    move with the bucket layout, which is not a property of the run."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        WEIGHT_THRESHOLDS,
        WeightShiftHistogram,
    )

    torch.manual_seed(3)
    bs, resp = 4, 32
    w = torch.rand(bs, resp) * 3.0
    kl = torch.rand(bs, resp) + 0.05
    mask = torch.ones(bs, resp)
    mask[0, 20:] = 0.0
    hist = WeightShiftHistogram(n_tasks=3, device="cpu")
    hist.update(weight=w, teacher_kl=kl, response_mask=mask, task_ids=torch.tensor([0, 1, 2, 0]))
    out = hist.metrics(task_names=TASKS)

    # In float64, as the accumulator is: a float32 brute force would differ at
    # 1e-7 and the tolerance would then be hiding an ordering difference rather
    # than pinning the arithmetic.
    w64, kl64, m64 = w.double(), kl.double(), mask.double()
    n = float(m64.sum())
    shift = (w64 - 1.0).abs() * kl64 * m64
    for t in WEIGHT_THRESHOLDS:
        above = (w64 > t).double() * m64
        tag = f"{int(round(t * 100)):03d}"
        assert out[f"kl_weight/shape/frac_w_gt_{tag}"] == pytest.approx(float(above.sum()) / n)
        assert out[f"kl_weight/shape/shift_share_w_gt_{tag}"] == pytest.approx(
            float((shift * above).sum()) / float(shift.sum()), rel=1e-12
        )
    assert out["kl_weight/shape/frac_w_below_one"] == pytest.approx(
        float(((w64 <= 0.99).double() * m64).sum()) / n
    )


def test_the_weight_histogram_ignores_masked_positions_whatever_their_weight():
    """A padded position carries an arbitrary W. If it reached a bucket it would
    show up as a tail the run does not have."""
    from verl.trainer.ppo.cross_teacher_kl_weight import WeightShiftHistogram

    hist = WeightShiftHistogram(n_tasks=1, device="cpu")
    w = torch.tensor([[1.0, 1.0, 99.0]])
    kl = torch.tensor([[1.0, 1.0, 1.0]])
    hist.update(
        weight=w, teacher_kl=kl,
        response_mask=torch.tensor([[1.0, 1.0, 0.0]]), task_ids=torch.tensor([0]),
    )
    out = hist.metrics(task_names=["a"])
    assert out["kl_weight/shape/frac_w_gt_200"] == pytest.approx(0.0)
    assert out["kl_weight/shape/w_q50"] == pytest.approx(1.0, abs=0.02)


def test_the_weight_quantile_lands_inside_the_bucket_the_mass_is_in():
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        WEIGHT_BUCKET_EDGES,
        WeightShiftHistogram,
    )

    hist = WeightShiftHistogram(n_tasks=1, device="cpu")
    # Everything between 1.1 and 1.25, so every quantile must land there.
    w = torch.full((1, 100), 1.2)
    hist.update(
        weight=w, teacher_kl=torch.ones(1, 100),
        response_mask=torch.ones(1, 100), task_ids=torch.tensor([0]),
    )
    out = hist.metrics(task_names=["a"])
    for q in ("w_q50", "w_q90", "w_q99"):
        assert 1.1 <= out[f"kl_weight/shape/{q}"] <= 1.25
    assert 1.25 in WEIGHT_BUCKET_EDGES and 1.1 in WEIGHT_BUCKET_EDGES


def test_token_turnover_separates_a_stable_set_from_one_that_is_replaced():
    """n_distinct and top_share are both within-step: forty tokens replaced
    wholesale every step reads exactly like a stable forty in both. This is the
    reading that tells them apart, so it is asserted on both cases."""
    from verl.trainer.ppo.sign_weights import STATE_AGREE_POS, TokenStateCounts

    def _table(ids):
        t = TokenStateCounts(vocab_size=32, n_tasks=1, device="cpu", top_n=4, mode="position")
        k = len(ids)
        t.update(
            support_ids=torch.tensor(ids).reshape(1, 1, k),
            state=torch.full((1, 1, k), STATE_AGREE_POS),
            weight=torch.full((1, 1, k), 2.0),
            on_task_logprob=torch.full((1, 1, k), -1.0),
            response_mask=torch.ones(1, 1),
            task_ids=torch.tensor([0]),
            effect=torch.full((1, 1, k), 0.5),
        )
        return t

    first = _table([1, 2, 3, 4])
    _m, state = first.turnover(previous=None, task_names=["a"], prefix="xt")
    assert _m == {}, "nothing to compare against on the first step"

    same, _ = _table([1, 2, 3, 4]).turnover(previous=state, task_names=["a"], prefix="xt")
    assert same["xt/token/turnover/top4_jaccard"] == pytest.approx(1.0)
    assert same["xt/token/turnover/effect_carryover"] == pytest.approx(1.0)

    fresh, _ = _table([9, 10, 11, 12]).turnover(previous=state, task_names=["a"], prefix="xt")
    assert fresh["xt/token/turnover/top4_jaccard"] == pytest.approx(0.0)
    assert fresh["xt/token/turnover/effect_carryover"] == pytest.approx(0.0)

    # Half kept: membership halves, and so does the mass, since every token here
    # carries the same effect.
    half, _ = _table([1, 2, 30, 31]).turnover(previous=state, task_names=["a"], prefix="xt")
    assert 0.0 < half["xt/token/turnover/top4_jaccard"] < 1.0
    assert half["xt/token/turnover/effect_carryover"] == pytest.approx(0.5)


def test_the_role_token_table_decomposes_the_same_nats_the_state_table_does():
    """Three tables off one quantity. If the role table summed to something else
    the run would have two different answers to "how many nats did the arm
    move" and no way to tell which was the mechanism."""
    from verl.trainer.ppo.sign_weights import RoleTokenCounts, ROLE_NAMES

    got, _ = _built()
    kl = torch.rand(*got["weight"].shape) + 0.1
    shift = per_candidate_shift(got, kl)
    bs, resp, k = shift.shape
    support = torch.arange(bs * resp * k).reshape(bs, resp, k) % 50
    roles = torch.arange(bs * resp).reshape(bs, resp) % len(ROLE_NAMES)
    mask = torch.ones(bs, resp)
    mask[0, -1] = 0.0

    tbl = RoleTokenCounts(vocab_size=64, device="cpu", top_n=4)
    tbl.update(support_ids=support, roles=roles, effect=shift, response_mask=mask)
    n, pos, neg = tbl._cpu()
    assert float((pos + neg).sum()) == pytest.approx(
        float((shift * mask.unsqueeze(-1)).sum()), abs=1e-6
    )
    assert int(n.sum()) == int(mask.sum()) * k
    # And the shares over roles are a partition of the gross.
    out = tbl.scalar_metrics(prefix="xt")
    shares = [v for key, v in out.items() if key.endswith("/token/shift_gross_share")]
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)


def test_the_role_token_rows_carry_the_schema_the_worker_decodes():
    """The rows are appended to the same report the other tables go to, so they
    have to carry token_id and say which quantity effect_* is in."""
    from verl.trainer.ppo.sign_weights import RoleTokenCounts

    tbl = RoleTokenCounts(vocab_size=16, device="cpu", top_n=3)
    tbl.update(
        support_ids=torch.tensor([[[1, 2]]]),
        roles=torch.tensor([[1]]),
        effect=torch.tensor([[[0.5, -0.25]]]),
        response_mask=torch.ones(1, 1),
    )
    rows = tbl.top_tokens()
    assert rows and all("token_id" in r for r in rows)
    assert {r["scope"] for r in rows} == {"role:reasoning"}
    assert {r["effect_kind"] for r in rows} == {"dkl_nats"}
    assert {r["ranked_by"] for r in rows} == {"count", "abs_effect"}


def test_a_probe_partitions_its_own_shift_not_the_live_arms():
    """The alpha series only reads as an ablation if each probe's state columns
    add up to that probe's own ``(W - 1) D``. Built from the live mu and the
    live evidence they would add up to the shipped arm's shift at every alpha,
    and the series would be three copies of one number."""
    from verl.workers.actor.dp_actor import _xt_apply_normalizer, _xt_normalizer_mu

    got, ctx = _built()
    kl = torch.rand(*got["weight"].shape) + 0.1
    snap = {"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)}
    seen = []
    for name, pre in got["probe_pre_weight"].items():
        mu = _xt_normalizer_mu(pre, snap, ctx["task_ids"])
        probe = {
            "weight": pre / mu, "pre_weight": pre, "mu": mu,
            "evidence": got["probe_evidence"][name],
            "state": got["state"], "teacher_prob": got["teacher_prob"],
            "mass": got["mass"],
        }
        terms = state_shift_terms(probe, kl)
        total = sum(terms[t] for t in STATE_TERMS)
        assert torch.allclose(total, (probe["weight"] - 1.0) * kl, atol=1e-5), name
        # The helper the training path uses agrees with pre/mu everywhere.
        assert torch.allclose(_xt_apply_normalizer(pre, snap, ctx["task_ids"]), probe["weight"])
        seen.append(float(total.abs().sum()))
    # alpha = 0 is the corroboration channel alone; alpha = 1 adds every
    # source's own magnitude, so the probes are ordered and genuinely differ.
    assert seen[0] < seen[-1], seen


def test_the_probe_normaliser_is_one_where_the_snapshot_has_no_mean():
    """Cold start runs at W = 1, so the divisor that reproduces the applied
    weight is ``pre`` itself. Returning 1 instead would make the probe's state
    columns sum to ``(pre - 1) D``, which no arm ever paid."""
    from verl.workers.actor.dp_actor import _xt_apply_normalizer, _xt_normalizer_mu

    got, ctx = _built()
    pre = got["probe_pre_weight"][probe_name(1.0)]
    for snap in (None, {"mean": torch.full((3,), 1.2), "valid": torch.zeros(3, dtype=torch.bool)}):
        mu = _xt_normalizer_mu(pre, snap, ctx["task_ids"])
        assert torch.allclose(pre / mu, torch.ones_like(pre))
        assert torch.allclose(_xt_apply_normalizer(pre, snap, ctx["task_ids"]), torch.ones_like(pre))


def test_probe_evidence_is_zero_where_the_position_is_unavailable():
    """It rides beside probe_pre_weight, which is forced to 1 there. Evidence
    that survived would charge a candidate for a position the arm did not act
    on."""
    got, _ = _built(diag_valid=[True, False, True])
    avail = got["available"]
    idx = (~avail).nonzero().flatten()
    assert idx.numel()
    for name, ev in got["probe_evidence"].items():
        for i in idx.tolist():
            assert float(ev[i].abs().max()) == 0.0, name


def test_every_new_accumulator_is_reduced_before_it_is_rendered():
    """A per-rank ratio is not the pooled ratio, and a top-N list has no
    meaningful per-rank average. Each of these renders from a reduced buffer or
    reports one rank's shard under a global name."""
    src = _update_policy_source()
    for name in (
        "xt_weight_hist",
        "xt_turn_stats",
        "xt_role_position_stats",
        "xt_role_state_stats",
        "xt_role_grad_stats",
        "xt_role_token_stats",
    ):
        assert f"{name} = " in src or f"{name} = None" in src, name
    # The three role accumulators are reduced by the loop that renders them.
    loop = src[src.index("for _acc, _render in ("):]
    loop = loop[: loop.index("# The rows are stashed")]
    for name in ("xt_role_position_stats", "xt_role_state_stats", "xt_role_grad_stats"):
        assert name in loop, name
    assert "_acc.all_reduce()" in loop
    for name in ("xt_weight_hist", "xt_turn_stats", "xt_role_token_stats", "xt_probe_state_stats"):
        assert f"{name}.all_reduce()" in src or f"{name}[_name]" in src or "_st.all_reduce()" in src


def test_the_new_cuts_are_gated_on_config_and_never_on_batch_content():
    """A collective whose participation depends on what a rank's rows contained
    hangs the job. Roles come from the worker's tag ids, which is a startup
    fact; the accumulators are built and reduced on that alone."""
    src = _update_policy_source()
    # Built under the arm's own switch.
    build = src[: src.index("xt_token_cfg = ")]
    assert "PositionScopeTermStats(names=XT_POSITION_TERMS" in build
    assert "if xt_on else None" in build
    # Reduced under `if xt_on:`, not under anything derived from the batch.
    reduce_block = src[src.index("xt_position_stats.all_reduce()"):]
    reduce_block = reduce_block[: reduce_block.index("if xt_token_stats is not None:")]
    for forbidden in ("task_ids is not None", "xt_roles_mb is not None", ".numel()", ".any()"):
        assert forbidden not in reduce_block, forbidden


def test_the_role_columns_are_the_same_ones_the_task_cut_gets():
    """Computed once and handed to four accumulators. Two calls would be two
    definitions of one number, and the role rows would stop summing to the task
    rows the first time either changed."""
    src = _update_policy_source()
    # Exactly one call each, and it is the assignment: any second one would be a
    # second definition of the same columns.
    assert src.count("xt_pos_cols = xt_position_terms(") == 1, "one call"
    assert src.count("xt_state_shift_terms(xt_built, teacher_kld)") == 1, "one call"
    assert src.count("xt_state_cols = xt_state_shift_terms(xt_built, teacher_kld)") == 1
    for user in (
        "xt_position_stats.update(\n                                xt_pos_cols",
        "xt_role_position_stats.update(\n                                    xt_pos_cols",
        "xt_turn_stats.update(\n                                xt_pos_cols",
    ):
        assert user in src, user


def test_the_turn_scope_is_clamped_so_a_long_episode_cannot_index_past_the_buffer():
    src = _update_policy_source()
    call = src[src.index("xt_turn_stats.update("):]
    call = call[: call.index(")\n")]
    assert "turn_index(response_mask)" in call
    assert "clamp(max=XT_TURN_BUCKETS - 1)" in call


def test_the_published_cuts_are_curated_and_stay_curated():
    """The accumulators hold every column position_weight_metrics can render.
    Publishing all of them for six roles and six turns is three hundred series a
    step, which is not a richer analysis than the arm already had -- it is one
    where the five readings that matter are five of three hundred. This pins the
    selection so re-widening it is a decision rather than a drift."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        PositionScopeTermStats,
        ROLE_CUT_SUFFIXES,
        ROLE_SCOPE_NAMES,
        TURN_CUT_SUFFIXES,
        TURN_SCOPE_NAMES,
        select_metrics,
    )

    got, _ = _built()
    kl = torch.rand(*got["weight"].shape) + 0.1
    shape = got["weight"].shape
    scope = torch.arange(got["weight"].numel()).reshape(shape) % len(ROLE_SCOPE_NAMES)

    acc = PositionScopeTermStats(names=POSITION_TERMS, n_scopes=len(ROLE_SCOPE_NAMES), device="cpu")
    acc.update(position_terms(got, kl), response_mask=torch.ones_like(kl), scope_ids=scope)
    full = position_weight_metrics(acc.sums(scope_names=ROLE_SCOPE_NAMES), prefix="kl_weight/role")
    kept = select_metrics(full, ROLE_CUT_SUFFIXES)
    assert len(kept) < len(full), "the point is that it is a subset"
    # Five per role from the position family; the rest of ROLE_CUT_SUFFIXES is
    # served by the state and gradient accumulators.
    assert len(kept) == 5 * len(ROLE_SCOPE_NAMES), sorted(kept)
    assert all(k.endswith(ROLE_CUT_SUFFIXES) for k in kept)

    state = PositionScopeTermStats(names=STATE_TERMS, n_scopes=len(ROLE_SCOPE_NAMES), device="cpu")
    state.update(state_shift_terms(got, kl), response_mask=torch.ones_like(kl), scope_ids=scope)
    st = select_metrics(
        state_shift_metrics(state.sums(scope_names=ROLE_SCOPE_NAMES), prefix="kl_weight/role"),
        ROLE_CUT_SUFFIXES,
    )
    # Shares, never nats: a per-state net inside one role is a number no reader
    # has a denominator for.
    assert all(k.endswith("/gross_share") for k in st)
    assert len(st) == len(STATE_TERMS) * len(ROLE_SCOPE_NAMES)

    turn = PositionScopeTermStats(names=POSITION_TERMS, n_scopes=len(TURN_SCOPE_NAMES), device="cpu")
    turn.update(position_terms(got, kl), response_mask=torch.ones_like(kl),
                scope_ids=torch.zeros(shape, dtype=torch.long))
    tk = select_metrics(
        position_weight_metrics(turn.sums(scope_names=TURN_SCOPE_NAMES), prefix="kl_weight/turn"),
        TURN_CUT_SUFFIXES,
    )
    assert len(tk) == len(TURN_CUT_SUFFIXES), "one populated turn scope"


def test_the_turn_scope_names_cover_every_bucket_the_clamp_can_produce():
    from verl.trainer.ppo.cross_teacher_kl_weight import TURN_BUCKETS, TURN_SCOPE_NAMES

    assert len(TURN_SCOPE_NAMES) == TURN_BUCKETS
    assert TURN_SCOPE_NAMES[-1].endswith("plus"), "the last one is open-ended"


def test_the_role_tag_ids_reach_the_actor_on_any_arm():
    """The role cut is useless without them, and they are tokenised in the
    worker because the actor has no tokenizer. Gated on the sign arm they would
    be silently absent here and every role row would read `format`."""
    import ast
    import inspect

    from verl.workers import fsdp_workers

    tree = ast.parse(inspect.getsource(fsdp_workers))

    def _blocks(node):
        """Every statement list in the tree, so 'same block' is checkable."""
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                yield body
        for child in ast.iter_child_nodes(node):
            yield from _blocks(child)

    def _is(stmt, text):
        return isinstance(stmt, ast.Assign) and text in ast.unparse(stmt.targets[0])

    together = [
        body for body in _blocks(tree)
        if any(_is(s, "sign_role_tag_ids") for s in body)
    ]
    assert together, "the assignment moved"
    # Siblings of the actor's own construction: no branch of any kind stands
    # between them, so an arm that never sets sign_weight still gets the ids.
    for body in together:
        assert any(_is(s, "self.actor") and not _is(s, "sign_role_tag_ids") for s in body), (
            "the tag ids are gated on something the actor's construction is not"
        )


# --------------------------------------------------------------------------- #
# the policy gradient the metric compares against is the one the run takes
# --------------------------------------------------------------------------- #
def test_the_closed_form_pg_coefficient_is_the_loss_differentiated():
    """It IS compute_policy_loss_per_token differentiated. Autograd is the only
    honest check: a change to the objective that is not mirrored in the closed
    form has to fail here rather than quietly report the old gradient."""
    from verl.trainer.ppo.core_algos import (
        compute_policy_loss_per_token,
        policy_loss_gradient_coef,
    )

    torch.manual_seed(5)
    bs, resp = 6, 9
    old = torch.randn(bs, resp).double()
    # Wide enough that every branch is exercised: inside the clip, above it,
    # below it, and past the dual-clip bound on negative advantages.
    lp = (old + torch.randn(bs, resp).double() * 0.9).requires_grad_(True)
    adv = torch.randn(bs, resp).double() * 2.0
    mask = torch.ones(bs, resp).double()
    kw = dict(cliprange=0.2, cliprange_low=0.2, cliprange_high=0.28, clip_ratio_c=3.0)

    losses, *_ = compute_policy_loss_per_token(
        old_log_prob=old, log_prob=lp, advantages=adv, response_mask=mask, **kw
    )
    losses.sum().backward()
    want = lp.grad
    got = policy_loss_gradient_coef(
        old_log_prob=old, log_prob=lp.detach(), advantages=adv, **kw
    )
    assert torch.allclose(got, want, atol=1e-10), (got - want).abs().max()

    # And the branches really were hit, or the agreement above is vacuous.
    r = (lp.detach() - old).exp()
    assert bool(((r > 1.28) & (adv > 0)).any()), "clip above, positive advantage"
    assert bool(((r < 0.8) & (adv < 0)).any()), "clip below, negative advantage"
    assert float((got == 0).double().mean()) > 0.05, "some positions are clipped off"


def test_the_pg_coefficient_is_not_the_advantage_once_the_ratio_moves():
    """The first mini-batch of the first epoch runs at ratio 1; the other five
    do not, because each takes an optimizer step. A metric using -A would be
    reporting a gradient the run stopped taking after 60 of its 360 rows."""
    from verl.trainer.ppo.core_algos import policy_loss_gradient_coef

    old = torch.zeros(1, 4)
    adv = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    kw = dict(cliprange=0.2, clip_ratio_c=3.0)

    at_one = policy_loss_gradient_coef(
        old_log_prob=old, log_prob=old.clone(), advantages=adv, **kw
    )
    assert torch.allclose(at_one, -adv), "ratio 1, nothing clipped"

    moved = policy_loss_gradient_coef(
        old_log_prob=old, log_prob=torch.full((1, 4), 0.3), advantages=adv, **kw
    )
    r = math.exp(0.3)
    assert r > 1.2, "the point is to be past the upper bound"
    # Positive advantage above the upper bound: the clamped branch wins and the
    # gradient is exactly zero. Negative advantage: pg1 stays larger, so the
    # coefficient survives as -A*r -- which is what the dual clip bounds, not
    # the ratio clip.
    assert float(moved[0, 0]) == pytest.approx(0.0)
    assert float(moved[0, 2]) == pytest.approx(r, rel=1e-6)
    assert not torch.allclose(moved, -adv)


def test_the_gradient_metric_reads_the_real_coefficient_and_both_coefficients():
    """A ratio between what the two terms contribute to ONE objective. At 0.01
    against 1.0 the coefficients are most of the answer, and the per-task row
    weight decides how much each row contributes to the pooled norms."""
    got, _ = _built()
    bs, resp, k = got["hat_on"].shape
    kl = torch.rand(bs, resp) + 0.1
    onehot = torch.zeros(bs, resp, k)
    onehot[..., 0] = 1.0
    student = _lp(bs, resp, k, seed=7)

    def terms(**over):
        kw = dict(
            student_logprob=student, teacher_logprob=got["hat_on"],
            weight=got["weight"], teacher_kl=kl,
            pg_grad_coef=torch.full((bs, resp), -1.0),
            sampled_onehot=onehot, coef=0.01, pg_coef=1.0,
        )
        kw.update(over)
        return logit_gradient_terms(**kw)

    base = terms()
    # A coefficient of zero -- every position clipped off -- leaves no policy
    # gradient at all, which -A could never report.
    dead = terms(pg_grad_coef=torch.zeros(bs, resp))
    assert float(dead["g_grpo_sq"].sum()) == pytest.approx(0.0)
    assert float(dead["g_dot"].sum()) == pytest.approx(0.0)
    assert float(dead["g_opd_sq"].sum()) == pytest.approx(float(base["g_opd_sq"].sum()))

    # pg_loss_coef scales the policy side quadratically and the cross term once.
    half = terms(pg_coef=0.5)
    assert float(half["g_grpo_sq"].sum()) == pytest.approx(
        0.25 * float(base["g_grpo_sq"].sum()), rel=1e-5
    )
    assert float(half["g_dot"].sum()) == pytest.approx(
        0.5 * float(base["g_dot"].sum()), rel=1e-5
    )

    # The row weight multiplies BOTH, so it cannot move a per-row cosine and
    # must move the pooled one whenever the rows differ.
    rw = torch.linspace(0.5, 2.0, bs)
    w = terms(row_weight=rw)
    cos = lambda t: float(t["g_dot"].sum()) / math.sqrt(
        float(t["g_opd_sq"].sum()) * float(t["g_grpo_sq"].sum())
    )
    assert cos(w) != pytest.approx(cos(base), rel=1e-6)
    flat = terms(row_weight=torch.full((bs,), 1.0))
    assert cos(flat) == pytest.approx(cos(base), rel=1e-9)


# --------------------------------------------------------------------------- #
# who actually caused the nats a source is charged with
# --------------------------------------------------------------------------- #
def test_a_source_with_nothing_to_add_is_charged_no_effect():
    """One teacher exceeds the on-task ceiling and one sits exactly on it, so
    only the first raised the weight and only the first may appear in the table.
    Filing the position's whole shift against every source that SPOKE reports
    the opposite -- and it is the table 'what did Search bring to AlfWorld' is
    read from. Under the exclusive split, speaking and adding are different
    things, which is the whole point of the split."""
    from verl.trainer.ppo.sign_weights import SignPairTokens

    torch.manual_seed(90)
    bs, resp, k = 4, 3, 5
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    # Column 0 moves twice as far as the on-task teacher; column 1 moves exactly
    # as far, so its excess is identically zero while its shift is not.
    off = torch.stack([base + 2.0 * (on - base), on.clone()], dim=-1)
    shifts = compute_raw_policy_shifts(
        on_task_logprob=on, off_task_logprobs=off, base_logprob=base
    )
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    alpha = torch.zeros(3, 3)
    got = build_position_weight(
        shifts=shifts, on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=alpha,
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.1), "valid": torch.ones(3, dtype=torch.bool)},
    )
    assert float(got["source_exclusive"][..., 1].abs().max()) == pytest.approx(0.0)
    assert float(got["q_sim"].max()) > 0.0, "the two teachers still agree"
    kl = torch.rand(bs, resp) + 0.1
    inv_mu = (1.0 / got["mu"].clamp(min=1e-12)).unsqueeze(-1)
    source_shift = got["evidence_by_source"] * (inv_mu * kl.unsqueeze(-1)).unsqueeze(-1)
    assert float(source_shift[..., 1].abs().sum()) == pytest.approx(0.0), "no excess"
    assert float(source_shift[..., 0].abs().sum()) > 0

    tbl = SignPairTokens(n_tasks=3, vocab_size=64, device="cpu", top_n=4)
    tbl.update(
        support_ids=torch.randint(0, 64, (bs, resp, k)),
        on_task_logprob=got["hat_on"], off_task_logprobs=got["hat_off"],
        base_logprob=torch.zeros_like(got["hat_on"]),
        response_mask=torch.ones(bs, resp), task_ids=task_ids,
        off_plane_tasks=planes, deadzone=0.1,
        effect=source_shift, mass=got["teacher_prob"],
    )
    _n, _mass, eff = tbl._cpu()
    # Only the trusted source carries nats. Read back through the pair index the
    # table itself builds, so this checks the routing and not just the input.
    charged = {}
    for row in range(bs):
        d, s0, s1 = int(task_ids[row]), int(planes[row, 0]), int(planes[row, 1])
        charged.setdefault((d, s0), 0.0)
        charged.setdefault((d, s1), 0.0)
    for (d, s) in charged:
        pair = d * 2 + s - (1 if s > d else 0)
        charged[(d, s)] = float(eff[pair].abs().sum())
    exceeding, matching = [], []
    for row in range(bs):
        d = int(task_ids[row])
        exceeding.append(charged[(d, int(planes[row, 0]))])
        matching.append(charged[(d, int(planes[row, 1]))])
    assert sum(exceeding) > 0
    assert sum(matching) == pytest.approx(0.0), "a source at the ceiling moved nothing"


def test_the_pair_table_still_takes_one_effect_column_for_the_sign_arm():
    """The sign arm's weight is a function of the sign PATTERN and cannot be
    decomposed over the teachers that produced it, so filing the total against
    each source is the right answer there. Both shapes have to keep working."""
    from verl.trainer.ppo.sign_weights import SignPairTokens

    bs, resp, k, n_off = 2, 2, 3, 2
    kw = dict(
        support_ids=torch.randint(0, 32, (bs, resp, k)),
        on_task_logprob=_lp(bs, resp, k, seed=91),
        off_task_logprobs=torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1),
        base_logprob=_lp(bs, resp, k),
        response_mask=torch.ones(bs, resp),
        task_ids=torch.tensor([0, 1]),
        off_plane_tasks=torch.tensor([[1, 2], [0, 2]]),
        deadzone=0.05,
    )
    flat = SignPairTokens(n_tasks=3, vocab_size=32, device="cpu")
    flat.update(effect=torch.full((bs, resp, k), 0.25), **kw)
    per_src = SignPairTokens(n_tasks=3, vocab_size=32, device="cpu")
    per_src.update(effect=torch.full((bs, resp, k, n_off), 0.25), **kw)
    # Same value in every column -> the two agree, which is what makes the
    # three-dimensional form a special case rather than a different table.
    assert torch.allclose(flat._cpu()[2], per_src._cpu()[2])

    with pytest.raises(AssertionError, match="columns for"):
        SignPairTokens(n_tasks=3, vocab_size=32, device="cpu").update(
            effect=torch.zeros(bs, resp, k, n_off + 1), **kw
        )


# --------------------------------------------------------------------------- #
# what the event dump says the four models said
# --------------------------------------------------------------------------- #
def test_the_event_dump_columns_named_for_probabilities_hold_probabilities():
    """SignEventSamples exponentiates what it is given. Handed the standardized
    shifts it wrote exp(delta_hat) into p_on and exp(0) = 1 into p_base -- and
    nothing downstream could tell, because both are finite and in range."""
    from verl.trainer.ppo.sign_weights import EVENT_FLOATS, SignEventSamples

    bs, resp, k, n_off = 2, 3, 4, 2
    on = _lp(bs, resp, k, seed=95)
    base = _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1)
    shift_on = torch.randn(bs, resp, k)
    ev = SignEventSamples(capacity=8, context=2, device="cpu")
    ev.update(
        support_ids=torch.randint(0, 50, (bs, resp, k)),
        state=torch.zeros(bs, resp, k, dtype=torch.long),
        weight=torch.ones(bs, resp, k), effect=torch.randn(bs, resp, k),
        on_task_logprob=on, off_task_logprobs=off, base_logprob=base,
        student_logprob=_lp(bs, resp, k), response_mask=torch.ones(bs, resp),
        responses=torch.randint(0, 50, (bs, resp)),
        norm=torch.ones(bs, resp), teacher_kl=torch.rand(bs, resp),
        task_ids=torch.tensor([0, 1]),
        shift_on=shift_on, shift_off=torch.randn(bs, resp, k, n_off),
    )
    rows = ev.rows()
    assert rows
    for r in rows:
        for col in ("p_base", "p_on", "p_student", "p_off_lo", "p_off_hi"):
            assert 0.0 < r[col] <= 1.0, (col, r[col])
        # p_base = 1 exactly is the signature of the bug: exp of a zero base.
        assert r["p_base"] != pytest.approx(1.0)
    # The shifts are their own columns and are NOT constrained to (0, 1].
    assert {"shift_on", "shift_off_lo", "shift_off_hi"} <= set(EVENT_FLOATS)
    assert any(r["shift_on"] < 0 for r in rows), "a shift can be negative"
    assert all(r["shift_off_lo"] <= r["shift_off_hi"] for r in rows)


def test_an_arm_without_standardized_shifts_reports_nan_rather_than_zero():
    """Zero is a value a shift can legitimately take, so an arm that never
    measured one has to say so."""
    from verl.trainer.ppo.sign_weights import SignEventSamples

    bs, resp, k = 1, 2, 3
    ev = SignEventSamples(capacity=4, context=1, device="cpu")
    ev.update(
        support_ids=torch.randint(0, 20, (bs, resp, k)),
        state=torch.zeros(bs, resp, k, dtype=torch.long),
        weight=torch.ones(bs, resp, k), effect=torch.ones(bs, resp, k),
        on_task_logprob=_lp(bs, resp, k, seed=96),
        off_task_logprobs=torch.stack([_lp(bs, resp, k)], dim=-1),
        base_logprob=_lp(bs, resp, k), student_logprob=_lp(bs, resp, k),
        response_mask=torch.ones(bs, resp), responses=torch.randint(0, 20, (bs, resp)),
        norm=torch.ones(bs, resp), teacher_kl=torch.rand(bs, resp),
    )
    for r in ev.rows():
        for col in ("shift_on", "shift_off_lo", "shift_off_hi", "reward"):
            assert r[col] != r[col], f"{col} should be nan, got {r[col]}"


def test_the_cross_teacher_dump_selects_the_row_score_it_reports():
    """The reward column exists on both arms, and only the sign arm was adding
    the batch key it comes from -- so on this arm every row read nan."""
    src = _update_policy_source()
    xt = src[src.index('select_keys += ["sign_cache_ids", "sign_off_tasks"]'):]
    xt = xt[: xt.index("if sign_enabled:")]
    assert '"token_level_scores" in data.batch.keys()' in xt
    assert 'select_keys.append("token_level_scores")' in xt


def test_the_event_dump_is_handed_the_real_planes_and_the_shifts_separately():
    import inspect

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    src = inspect.getsource(DataParallelPPOActor._xt_token_tables)
    call = src[src.index("event_stats.update("):]
    assert "on_task_logprob=on_task_logprob" in call
    assert "off_task_logprobs=off_planes" in call
    assert "base_logprob=base_plane" in call
    assert 'shift_on=built["hat_on"]' in call
    assert 'shift_off=built["hat_off"]' in call
    # And the aggregate tables keep the standardized form, whose deadzone is in
    # RMS units and comparable across teachers.
    agg = src[: src.index("event_stats.update(")]
    assert 'on_task_logprob=built["hat_on"]' in agg
    assert "base_logprob=zero_base" in agg


# --------------------------------------------------------------------------- #
# the KL the weight multiplies
# --------------------------------------------------------------------------- #
def test_a_nonfinite_teacher_kl_is_neutralised_and_counted_not_multiplied_by_one():
    """W = 1 is not a guard: 1 * NaN is NaN, and 0 * NaN at a masked position is
    NaN too, so agg_loss carries it into backward and the optimizer steps before
    the step-end check ever runs."""
    src = _update_policy_source()
    block = src[src.index("if xt_nonfinite is not None:"):]
    block = block[: block.index("# Read BEFORE the position weight")]
    assert "torch.isfinite(teacher_kld)" in block
    assert "xt_nonfinite[3]" in block
    assert "torch.where(" in block and "torch.zeros_like(teacher_kld)" in block
    # Counted inside the mask only: a padded position is not a teacher failure.
    assert "response_mask.to(torch.bool)" in block
    # Before the weight multiplies it, and before the cold-start branch, so the
    # first step is covered too -- there xt_built is None and the KL still
    # reaches the loss.
    assert src.index("if xt_nonfinite is not None:") < src.index(
        'teacher_kld = teacher_kld * xt_built["weight"]'
    )
    assert "teacher_kl" in src[src.index("assert_all_finite({"):][:400]


def test_the_nonfinite_tally_has_a_slot_for_every_channel_it_reports():
    src = _update_policy_source()
    alloc = src[src.index("xt_nonfinite = torch.zeros("):]
    alloc = alloc[: alloc.index("\n")]
    reported = src[src.index("assert_all_finite({"):]
    reported = reported[: reported.index("})")]
    assert alloc.count("torch.zeros(4") == 1, alloc
    assert reported.count("xt_nonfinite[") == 4, reported


# --------------------------------------------------------------------------- #
# is the cumulative estimate still describing the run
# --------------------------------------------------------------------------- #
def test_the_rms_current_scope_is_this_steps_rows_and_the_cumulative_is_not():
    """The weight divides by the cumulative sigma. Over 150 steps the student
    moves, so the teachers' shifts on its states are not stationary and a
    divisor that averages step 1 into step 150 is a scale nobody chose. Only a
    step-local reading can say so."""
    acc = CumulativePolicyShiftRMS(n_tasks=3, device="cpu")
    ids = torch.tensor([0, 0])
    planes = torch.tensor([[1, 2], [1, 2]])
    mask = torch.ones(2, 4)

    def feed(scale):
        # The student's whole mass on the support, so the tail contributes
        # nothing and sigma is exactly |scale|.
        acc.update(
            shifts={
                "on": torch.full((2, 4, 3), scale),
                "off": torch.full((2, 4, 3, 2), scale),
                "tail_on": torch.zeros(2, 4),
                "tail_off": torch.zeros(2, 4, 2),
            },
            student_logprob=torch.full((2, 4, 3), math.log(1 / 3)),
            response_mask=mask, task_ids=ids, off_plane_tasks=planes,
        )
        acc.all_reduce()

    feed(1.0)
    first = acc.snapshot()
    assert acc.snapshot(scope="current")["sigma"][0, 0] == pytest.approx(
        float(first["sigma"][0, 0]), rel=1e-9
    ), "one step in, the two coincide"

    feed(3.0)
    cum, now = acc.snapshot(), acc.snapshot(scope="current")
    assert float(now["sigma"][0, 0]) == pytest.approx(3.0, rel=1e-6), "this step alone"
    assert float(cum["sigma"][0, 0]) == pytest.approx(math.sqrt(5.0), rel=1e-6), "both steps"
    assert float(now["sigma"][0, 0]) > float(cum["sigma"][0, 0]), "the drift is visible"
    # And the cumulative path is untouched: the diagonal the weight divides by
    # still comes from the cumulative snapshot.
    diag, _valid = acc.diagonal()
    assert float(diag[0]) == pytest.approx(float(cum["sigma"][0, 0]), rel=1e-9)


def test_the_current_scope_is_empty_before_the_first_reduce():
    """Not zero, not the pending delta: no step has been folded, so there is
    nothing to report and every cell is invalid."""
    acc = CumulativePolicyShiftRMS(n_tasks=2, device="cpu")
    now = acc.snapshot(scope="current")
    assert not bool(now["valid"].any())
    assert float(now["n"].sum()) == 0.0


def test_the_reliability_current_scope_sees_only_this_steps_rollouts():
    """rho_cumulative is what alpha is built from. rho_current is the only thing
    that can say it has gone stale -- and a pair whose two disagree in sign is
    one where the applied alpha and this step's own evidence point opposite
    ways."""
    acc = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    n = 40
    planes = torch.zeros(n, 2, dtype=torch.long)
    planes[:, 0] = 1
    planes[:, 1] = 2

    def feed(sign):
        s = torch.linspace(-1.0, 1.0, n)
        acc.update(
            advantage=sign * s,
            support_score=torch.stack([s, torch.randn(n) * 0.01], dim=-1),
            on_support_score=torch.zeros(n), length=torch.full((n,), 100.0),
            informative=torch.ones(n, dtype=torch.bool),
            task_ids=torch.zeros(n, dtype=torch.long),
            off_plane_tasks=planes, group_ids=torch.arange(n) % 8,
        )
        acc.all_reduce()

    feed(+1.0)
    up = acc.alpha(task_names=TASKS)[("alfworld", "search")]
    assert up["rho"] > 0.9
    now = acc.alpha(task_names=TASKS, scope="current")[("alfworld", "search")]
    assert now["rho"] == pytest.approx(up["rho"], rel=1e-6), "one step in, the same rows"

    # Two more steps the same way, then one that reverses: the cumulative
    # estimate still leans positive while this step alone says the opposite,
    # which is the situation the sign-disagreement flag exists to name. (An
    # exactly cancelling pair would put the cumulative rho at 0, where there is
    # no sign to disagree with.)
    feed(+1.0)
    feed(-1.0)
    cum = acc.alpha(task_names=TASKS)[("alfworld", "search")]
    step = acc.alpha(task_names=TASKS, scope="current")[("alfworld", "search")]
    assert step["rho"] < -0.9, "this step alone reversed"
    assert step["n"] == pytest.approx(float(n)), "and it is this step's rows only"
    assert cum["n"] == pytest.approx(3.0 * n)
    assert abs(cum["rho"]) < abs(step["rho"]), "the cumulative one is diluted"
    # The disagreement this reports is exactly a sign flip between the two.
    assert step["rho"] * cum["rho"] < 0


def test_the_informative_fraction_separates_no_signal_from_no_spread():
    """Every rollout of a prompt scoring the same gives every row zero advantage.
    Without this column, 'the source does not predict reward' and 'there was
    nothing to predict' are the same alpha."""
    acc = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=8)
    n = 20
    planes = torch.zeros(n, 2, dtype=torch.long)
    planes[:, 0], planes[:, 1] = 1, 2
    informative = torch.zeros(n, dtype=torch.bool)
    informative[: n // 4] = True
    acc.update(
        advantage=torch.randn(n), support_score=torch.randn(n, 2),
        on_support_score=torch.zeros(n), length=torch.full((n,), 50.0),
        informative=informative, task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=planes, group_ids=torch.arange(n) % 8,
    )
    acc.all_reduce()
    row = acc.alpha(task_names=TASKS)[("alfworld", "search")]
    assert row["informative_group_frac"] == pytest.approx(0.25)
    assert row["n"] == pytest.approx(float(n // 4)), "only the informative rows"


def test_a_pair_with_no_informative_row_still_reports_that_it_was_offered():
    """Otherwise the pair goes missing from the log entirely, which reads as
    'not measured' when what happened is 'measured, and there was nothing in
    it'."""
    acc = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    n = 8
    planes = torch.zeros(n, 2, dtype=torch.long)
    planes[:, 0], planes[:, 1] = 1, 2
    acc.update(
        advantage=torch.zeros(n), support_score=torch.randn(n, 2),
        on_support_score=torch.zeros(n), length=torch.full((n,), 10.0),
        informative=torch.zeros(n, dtype=torch.bool),
        task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=planes, group_ids=torch.arange(n) % 4,
    )
    acc.all_reduce()
    row = acc.alpha(task_names=TASKS)[("alfworld", "search")]
    assert row["n"] == 0.0
    assert row["alpha"] == 0.0
    assert row["rho"] is None
    assert row["informative_group_frac"] == pytest.approx(0.0)


def test_weight_kl_lift_says_whether_the_weight_landed_where_the_kl_was():
    """At exactly 1 the weight and the KL are uncorrelated, which is the arm
    being a scalar on the whole term -- teacher_kl_loss_coef with extra steps.
    That is the null this whole mechanism is tested against."""
    def render(w, kl):
        stats = ScopeTermStats(names=POSITION_TERMS, n_tasks=0, device="cpu")
        built = {
            "weight": w, "pre_weight": w, "available": torch.ones(w.size(0), dtype=torch.bool),
            "evidence_shared": torch.zeros_like(w),
            "evidence_shared_offtask_only": torch.zeros_like(w),
            "push_shared": torch.zeros_like(w),
            "push_by_source": torch.zeros_like(w).unsqueeze(-1),
        }
        stats.update(position_terms(built, kl), response_mask=torch.ones_like(kl), task_ids=None)
        return position_weight_metrics(stats.sums())

    w = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    aligned = render(w, torch.tensor([[0.1, 1.0, 0.1, 1.0]]))
    against = render(w, torch.tensor([[1.0, 0.1, 1.0, 0.1]]))
    flat = render(w, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))

    assert aligned["kl_weight/effect/weight_kl_lift"] > 1.0
    assert against["kl_weight/effect/weight_kl_lift"] < 1.0
    assert flat["kl_weight/effect/weight_kl_lift"] == pytest.approx(1.0)
    assert aligned["kl_weight/effect/weight_kl_corr"] == pytest.approx(1.0, abs=1e-6)
    assert against["kl_weight/effect/weight_kl_corr"] == pytest.approx(-1.0, abs=1e-6)
    # No spread in the KL -> no correlation to report, rather than a zero that
    # reads as "measured and found nothing".
    assert "kl_weight/effect/weight_kl_corr" not in flat


def test_the_offered_count_survives_a_resume_and_an_old_sidecar_reads_as_absent():
    """The numerator of informative_group_frac was already in the sidecar. A
    denominator that restarts at zero would make the fraction read above 1, or
    undefined, for as long as the run took to re-accumulate it."""
    planes = torch.zeros(6, 2, dtype=torch.long)
    planes[:, 0], planes[:, 1] = 1, 2
    inf = torch.tensor([True, True, False, False, False, False])

    def fed():
        acc = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
        acc.update(
            advantage=torch.randn(6), support_score=torch.randn(6, 2),
            on_support_score=torch.zeros(6), length=torch.full((6,), 10.0),
            informative=inf, task_ids=torch.zeros(6, dtype=torch.long),
            off_plane_tasks=planes, group_ids=torch.arange(6) % 4,
        )
        acc.all_reduce()
        return acc

    saved = fed().state_dict()
    assert "offered" in saved
    restored = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    restored.load_state_dict(saved)
    assert restored.alpha(task_names=TASKS)[("alfworld", "search")][
        "informative_group_frac"
    ] == pytest.approx(2.0 / 6.0)

    # A sidecar written before the key existed: absent, not a ratio over a
    # denominator that restarted.
    old = dict(saved)
    old.pop("offered")
    old.pop("doffered")
    stale = AdvantageReliabilityStats(n_tasks=3, device="cpu", max_groups=4)
    stale.load_state_dict(old)
    row = stale.alpha(task_names=TASKS)[("alfworld", "search")]
    assert row["informative_group_frac"] is None
    assert row["n"] == pytest.approx(2.0), "the moments still came back"


def test_the_pair_token_table_is_wired_to_the_per_source_shift():
    """The routing is right and the input has to be too. Handed the position's
    TOTAL shift the table is correct arithmetic on the wrong quantity, and every
    source reads back carrying every other source's nats plus the corroboration
    term none of them caused."""
    import inspect

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    src = inspect.getsource(DataParallelPPOActor._xt_token_tables)
    assert 'source_shift = built["evidence_by_source"]' in src
    call = src[src.index("pair_token_stats.update("):]
    call = call[: call.index("\n            )")]
    assert "effect=source_shift" in call, call
    assert "effect=shift" not in call, "the pooled total is the bug"
    # And the per-state / per-token tables keep the total, which is what they
    # decompose -- the two must not be swapped.
    tok = src[src.index("token_stats.update("):]
    tok = tok[: tok.index("\n            )")]
    assert "effect=shift," in tok


# --------------------------------------------------------------------------- #
# the evidence token and the token whose logit moved are not the same token
# --------------------------------------------------------------------------- #
def test_the_unweighted_push_is_the_gradient_of_the_unweighted_opd_term():
    """One definition, read by the norm metric and by the push table. A second
    copy is how 'the arm amplified this token' and 'the arm's gradient norm'
    come to describe different quantities under one name."""
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.cross_teacher_kl_weight import opd_logit_push

    torch.manual_seed(100)
    k, V, coef = 4, 9, 0.01
    logits = torch.randn(1, 1, V, requires_grad=True)
    teacher = torch.log_softmax(torch.randn(1, 1, V), -1)[..., :k]
    lp = torch.log_softmax(logits, -1)
    kl = topk_kl_per_token(lp[..., :k], teacher)
    (coef * kl).sum().backward()
    auto = -logits.grad[0, 0]

    got = opd_logit_push(
        student_logprob=lp[..., :k].detach(), teacher_logprob=teacher,
        teacher_kl=kl.detach(), coef=coef,
    )
    assert torch.allclose(got["g0"][0, 0], auto[:k], atol=1e-7)
    assert float(got["g0_tail"][0, 0]) == pytest.approx(float(auto[k:].sum()), rel=1e-4)


def test_the_four_direction_classes_are_the_product_of_two_facts():
    """Sign alone is not the reading: a weight above 1 at a token the term was
    pushing DOWN amplifies the suppression, and calling that 'reinforced' would
    invert the claim."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        PUSH_CLASSES, push_direction_class,
    )

    g0 = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    w = torch.tensor([[0.5, 2.0]])
    cls = push_direction_class(g0, w)
    named = [[PUSH_CLASSES[int(c)] for c in row] for row in cls[0]]
    assert named[0] == ["push_down_damped", "push_up_damped"]
    assert named[1] == ["push_down_amplified", "push_up_amplified"]


def test_the_push_table_names_the_support_not_the_tokens_that_supplied_evidence():
    """The claim this separation protects: W is a scalar on the POSITION, so
    every token in the support has its logit moved, including ones no teacher
    spoke at. A table conflating the two would say 'Search reinforced retrieve'
    when what it reinforced there is the suppression of something else."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        LogitPushTokens, PUSH_CLASSES, opd_logit_push,
    )

    bs, resp, k = 2, 3, 4
    student = _lp(bs, resp, k, seed=101)
    teacher = _lp(bs, resp, k, seed=102)
    kl = torch.rand(bs, resp) + 0.1
    push = opd_logit_push(
        student_logprob=student, teacher_logprob=teacher, teacher_kl=kl, coef=0.01
    )
    w = torch.tensor([[0.6, 1.4, 1.0], [2.0, 0.9, 1.1]])
    support = torch.arange(bs * resp * k).reshape(bs, resp, k) % 40
    mask = torch.ones(bs, resp)
    mask[1, 2] = 0.0

    tbl = LogitPushTokens(vocab_size=64, n_tasks=3, device="cpu", top_n=4)
    tbl.update(
        support_ids=support, g0=push["g0"], weight=w, coef_applied_weight=w,
        response_mask=mask, task_ids=torch.tensor([0, 1]),
        sampled_onehot=torch.nn.functional.one_hot(
            torch.zeros(bs, resp, dtype=torch.long), k
        ).to(torch.float32),
        p_student=push["p_student"],
    )
    buf = tbl._cpu()
    i = {t: j for j, t in enumerate(tbl.TERMS)}
    # The pooled scope holds EVERY masked candidate, not just the ones a teacher
    # spoke at: k per live position.
    assert float(buf[0, :, :, i["n"]].sum()) == pytest.approx(float(mask.sum()) * k)
    # And the added push is exactly (W - 1) * g0 over those.
    want = ((w.unsqueeze(-1) - 1.0) * push["g0"] * mask.unsqueeze(-1)).sum()
    assert float(buf[0, :, :, i["extra"]].sum()) == pytest.approx(float(want), abs=1e-9)
    # A weight of exactly 1 adds nothing, whatever g0 was doing there.
    assert float(buf[0, :, :, i["extra_abs"]].sum()) > 0
    unity = ((w == 1.0).unsqueeze(-1) * push["g0"].abs() * mask.unsqueeze(-1)).sum()
    assert float(unity) > 0, "there is such a position"
    # Classes partition the count, and the shares sum to one.
    m = tbl.scalar_metrics(task_names=TASKS)
    shares = [m[f"kl_weight/push/{c}/gross_share"] for c in PUSH_CLASSES
              if f"kl_weight/push/{c}/gross_share" in m]
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)


def test_the_push_table_refuses_two_different_weights():
    """The class and the magnitude have to come from the SAME W, or a token is
    filed under one and measured under the other."""
    from verl.trainer.ppo.cross_teacher_kl_weight import LogitPushTokens

    tbl = LogitPushTokens(vocab_size=8, n_tasks=1, device="cpu")
    kw = dict(
        support_ids=torch.zeros(1, 1, 2, dtype=torch.long),
        g0=torch.ones(1, 1, 2), response_mask=torch.ones(1, 1),
    )
    with pytest.raises(AssertionError, match="APPLIED weight"):
        tbl.update(weight=torch.ones(1, 1), coef_applied_weight=torch.full((1, 1), 2.0), **kw)


def test_the_push_rows_say_which_direction_and_carry_the_students_own_mass():
    """'The arm amplified this token' and 'the arm amplified a token the student
    was never going to say' are different findings with the same nats."""
    from verl.trainer.ppo.cross_teacher_kl_weight import LogitPushTokens

    tbl = LogitPushTokens(vocab_size=16, n_tasks=1, device="cpu", top_n=3)
    tbl.update(
        support_ids=torch.tensor([[[1, 2]]]),
        g0=torch.tensor([[[1.0, -1.0]]]),
        weight=torch.tensor([[2.0]]), coef_applied_weight=torch.tensor([[2.0]]),
        response_mask=torch.ones(1, 1), task_ids=torch.tensor([0]),
        sampled_onehot=torch.tensor([[[1.0, 0.0]]]),
        p_student=torch.tensor([[[0.7, 0.2]]]),
    )
    rows = {(r["scope"], r["token_id"]): r for r in tbl.top_tokens(task_names=["a"])}
    up = rows[("__pooled__", 1)]
    down = rows[("__pooled__", 2)]
    assert up["direction_class"] == "push_up_amplified"
    assert down["direction_class"] == "push_down_amplified"
    assert up["extra_logit_push"] == pytest.approx(1.0)
    assert down["extra_logit_push"] == pytest.approx(-1.0)
    assert up["p_student_mean"] == pytest.approx(0.7)
    assert up["sampled_count"] == 1 and down["sampled_count"] == 0


# --------------------------------------------------------------------------- #
# per-trajectory: which rollouts the budget went to
# --------------------------------------------------------------------------- #
def test_the_outcome_buckets_separate_where_the_budget_went():
    """'The arm moved 3% of the budget' and 'the arm moved 3% of the budget,
    almost all of it on rollouts that failed' are the same number and opposite
    findings."""
    from verl.trainer.ppo.cross_teacher_kl_weight import OutcomeEffectStats

    acc = OutcomeEffectStats(n_tasks=3, device="cpu")
    # Two rows: the first has a positive advantage and a scored episode and a
    # weight far from 1; the second is the opposite on both counts.
    acc.update(
        weight=torch.tensor([[2.0, 2.0], [1.0, 1.0]]),
        teacher_kl=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        response_mask=torch.ones(2, 2),
        advantage=torch.tensor([1.5, -1.5]),
        task_ids=torch.tensor([0, 0]),
        reward=torch.tensor([1.0, 0.0]),
    )
    m = acc.metrics(task_names=TASKS)
    assert m["kl_weight/outcome/adv_positive/gross_effect"] == pytest.approx(1.0)
    assert m["kl_weight/outcome/adv_negative/gross_effect"] == pytest.approx(0.0)
    assert m["kl_weight/outcome/reward_positive/gross_effect"] == pytest.approx(1.0)
    assert m["kl_weight/outcome/reward_nonpositive/gross_effect"] == pytest.approx(0.0)
    # Both sides have to exist for the ratio, and the zero side is not an
    # infinity -- a step with no scoring rows is not a step with an infinite
    # ratio.
    assert "kl_weight/outcome/reward_positive_to_nonpositive_effect_ratio" not in m
    assert m["kl_weight/outcome/all/n_rows"] == pytest.approx(2.0)
    assert m["kl_weight/alfworld/outcome/adv_positive/gross_effect"] == pytest.approx(1.0)


def test_the_outcome_correlation_is_over_trajectories_not_positions():
    """The trajectory-level companion to weight_kl_corr. A per-position reading
    cannot say whether the ROLLOUTS the arm spent on were the ones that scored."""
    from verl.trainer.ppo.cross_teacher_kl_weight import OutcomeEffectStats

    n = 30
    a = torch.linspace(-1.0, 1.0, n)
    # G_i = |w - 1| exactly, since every KL is 1 -- so making w track a makes
    # the correlation +1.
    acc = OutcomeEffectStats(n_tasks=1, device="cpu")
    acc.update(
        weight=(1.0 + a.abs() * a.sign() * 0 + (1.0 + a)).reshape(n, 1).expand(n, 4).contiguous(),
        teacher_kl=torch.ones(n, 4), response_mask=torch.ones(n, 4),
        advantage=a, task_ids=torch.zeros(n, dtype=torch.long),
    )
    m = acc.metrics(task_names=["a"])
    assert m["kl_weight/outcome/corr_adv_gross_effect"] > 0.9
    # The reward buckets stay empty without a reward: the advantage is
    # group-relative and says nothing about what the episode scored.
    assert "kl_weight/outcome/reward_positive/gross_effect" not in m


def test_a_row_with_no_kl_contributes_no_ratio():
    """G_i divides by the row's own KL. A padded row would otherwise enter the
    correlation as a zero it did not earn."""
    from verl.trainer.ppo.cross_teacher_kl_weight import OutcomeEffectStats

    acc = OutcomeEffectStats(n_tasks=1, device="cpu")
    acc.update(
        weight=torch.tensor([[2.0], [2.0]]), teacher_kl=torch.tensor([[1.0], [0.0]]),
        response_mask=torch.ones(2, 1), advantage=torch.tensor([1.0, 1.0]),
        task_ids=torch.zeros(2, dtype=torch.long),
    )
    assert acc.metrics()["kl_weight/outcome/all/n_rows"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# source x disagreement state
# --------------------------------------------------------------------------- #
def test_the_pair_state_table_keeps_both_axes_the_others_collapse():
    """kl_shift_by_state sums the sources out; evidence/{src}__on__{dst} sums the
    states out. 'When Search disagreed with AlfWorld's own teacher, what did the
    arm do' is in neither, and arbitrating exactly that is the mechanism."""
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        PAIR_STATES, PairStateEvidenceStats, pair_state_index,
    )

    hat_on = torch.tensor([[[[1.0], [0.0], [1.0], [0.0]]]]).squeeze(-1).unsqueeze(0)
    hat_on = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])           # (1, 1, 4)
    hat_off = torch.tensor([[[[1.0], [1.0], [-1.0], [0.0]]]])  # (1, 1, 4, 1)
    st = pair_state_index(hat_on=hat_on, hat_off=hat_off, deadzone=0.1)
    assert [PAIR_STATES[int(x)] for x in st[0, 0, :, 0]] == [
        "agree", "on_silent_source_active", "conflict", "source_silent",
    ]

    acc = PairStateEvidenceStats(n_tasks=3, device="cpu")
    acc.update(
        state=st,
        evidence=torch.ones(1, 1, 4, 1),
        shift=torch.tensor([[[[2.0], [3.0], [-5.0], [7.0]]]]),
        response_mask=torch.ones(1, 1),
        task_ids=torch.tensor([0]), off_plane_tasks=torch.tensor([[1]]),
    )
    m = acc.metrics(task_names=TASKS)
    head = "kl_weight/pair_state/search__on__alfworld"
    # CANDIDATES, not positions: one position contributed all four states, so
    # each is a quarter of the candidate slots AND present at 100% of positions.
    # Reporting the first under the second's name is what this pair of
    # assertions pins.
    for s in PAIR_STATES:
        assert m[f"{head}/{s}/candidate_frac"] == pytest.approx(0.25)
        assert m[f"{head}/{s}/position_any_frac"] == pytest.approx(1.0)
    assert m[f"{head}/candidate_count"] == pytest.approx(4.0)
    assert m[f"{head}/n_positions"] == pytest.approx(1.0)
    assert not any(k.endswith("/position_frac") for k in m), "the misnamed key is gone"
    assert m[f"{head}/conflict/kl_shift_net"] == pytest.approx(-5.0)
    shares = [m[f"{head}/{s}/kl_shift_gross_share"] for s in PAIR_STATES]
    assert sum(shares) == pytest.approx(1.0)
    assert m[f"{head}/source_silent/kl_shift_gross_share"] == pytest.approx(7.0 / 17.0)


# --------------------------------------------------------------------------- #
# the channel counterfactuals
# --------------------------------------------------------------------------- #
def test_the_channel_probes_remove_a_channel_rather_than_scaling_one():
    """The alpha series asks 'what if the advantage channel were scaled
    differently'. These ask 'what if a channel were not there', which is the
    ablation the design decisions were made on."""
    from verl.trainer.ppo.cross_teacher_kl_weight import CHANNEL_PROBES

    got, _ = _built(alpha=0.5)
    assert set(got["channel_pre_weight"]) == set(CHANNEL_PROBES)
    live = got["pre_weight"]
    no_shared = got["channel_pre_weight"]["source_only"]
    off_shared = got["channel_pre_weight"]["legacy_hard_offtask_shared"]
    # Dropping the corroboration term can only lower the pre-weight: every term
    # in W~ - 1 is non-negative.
    assert bool((no_shared <= live + 1e-6).all())
    assert float((live - no_shared).abs().sum()) > 0, "the channel carries something"
    # And the off-task-only rule is the counterfactual the arm chose against, so
    # it is a different number from both.
    assert float((off_shared - live).abs().sum()) > 0
    assert got["channel_evidence"]["source_only"].shape == got["evidence"].shape


def test_a_channel_probe_is_zero_where_the_position_is_unavailable():
    got, _ = _built(diag_valid=[True, False, True])
    idx = (~got["available"]).nonzero().flatten()
    assert idx.numel()
    for name, pre in got["channel_pre_weight"].items():
        for i in idx.tolist():
            assert torch.allclose(pre[i], torch.ones_like(pre[i])), name
            assert float(got["channel_evidence"][name][i].abs().max()) == 0.0, name


# --------------------------------------------------------------------------- #
# support coverage
# --------------------------------------------------------------------------- #
def test_support_mass_separates_a_weak_source_from_an_invisible_one():
    """'Search contributed little to AlfWorld' has three explanations that read
    identically -- weak signal, low alpha, or Search's vocabulary simply not in
    the support the whole mechanism is measured on."""
    from verl.trainer.ppo.cross_teacher_kl_weight import PairEvidenceStats

    acc = PairEvidenceStats(n_tasks=3, device="cpu")
    acc.update(
        evidence=torch.ones(2, 3, 2), shift=torch.ones(2, 3, 2),
        response_mask=torch.ones(2, 3), task_ids=torch.tensor([0, 0]),
        off_plane_tasks=torch.tensor([[1, 2], [1, 2]]),
        # Search covers most of the student's top-k here; WebShop almost none.
        support_mass=torch.stack(
            [torch.full((2, 3), 0.8), torch.full((2, 3), 0.02)], dim=-1
        ),
    )
    m = acc.metrics(task_names=TASKS)
    assert m["kl_weight/evidence/search__on__alfworld/support_mass"] == pytest.approx(0.8)
    assert m["kl_weight/evidence/webshop__on__alfworld/tail_mass"] == pytest.approx(0.98)
    # Absent rather than zero when the caller did not measure it.
    bare = PairEvidenceStats(n_tasks=3, device="cpu")
    bare.update(
        evidence=torch.ones(1, 2, 1), shift=torch.ones(1, 2, 1),
        response_mask=torch.ones(1, 2), task_ids=torch.tensor([0]),
        off_plane_tasks=torch.tensor([[1]]),
    )
    assert "kl_weight/evidence/search__on__alfworld/support_mass" not in bare.metrics(
        task_names=TASKS
    )


# --------------------------------------------------------------------------- #
# the pair-stratified dump, end to end through the real assembly
# --------------------------------------------------------------------------- #
def _pair_event_rows(bs=6, resp=4, k=5, n_off=2, per_group=2, seed=120):
    """Drive dp_actor's own column assembly, not a reimplementation of it."""
    from verl.trainer.ppo.cross_teacher_kl_weight import opd_logit_push
    from verl.trainer.ppo.sign_weights import PairEventSamples
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([_lp(bs, resp, k) for _ in range(n_off)], dim=-1)
    shifts = compute_raw_policy_shifts(
        on_task_logprob=on, off_task_logprobs=off, base_logprob=base
    )
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    alpha = torch.full((3, 3), 0.6)
    alpha.fill_diagonal_(0.0)
    built = build_position_weight(
        shifts=shifts, on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=task_ids, off_plane_tasks=planes,
        # Not all ones: with a unit divisor the raw and standardized columns are
        # the same number and the test that they are separate would pass on a
        # table that had merged them.
        diag=torch.tensor([0.5, 2.0, 1.5]), alpha_table=alpha,
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.1), "valid": torch.ones(3, dtype=torch.bool)},
    )
    kl = torch.rand(bs, resp) + 0.1
    mask = torch.ones(bs, resp)
    mask[0, -1] = 0.0
    support = torch.randint(0, 500, (bs, resp, k))
    data = {
        "responses": torch.randint(0, 500, (bs, resp)),
        "sign_off_tasks": planes,
    }
    push = opd_logit_push(
        student_logprob=_lp(bs, resp, k), teacher_logprob=on, teacher_kl=kl, coef=0.01
    )
    stats = PairEventSamples(n_tasks=3, per_group=per_group, context=2, device="cpu")
    DataParallelPPOActor._xt_pair_events(
        None, stats=stats, built=built, teacher_kl=kl, data=data,
        support_ids=support, student_topk_logprob=_lp(bs, resp, k),
        on_task_logprob=on, response_mask=mask, task_ids=task_ids,
        report_epsilon=0.1, roles=None, planes=(base, off), raw_shifts=shifts,
        alpha_table=alpha, row_reward=torch.rand(bs), row_advantage=torch.randn(bs),
        push=push,
    )
    return stats.rows(task_names=TASKS), built, stats


def test_the_pair_event_rows_carry_the_source_specific_columns():
    """A per-candidate row can only report the min and max over whichever
    off-task teachers spoke. 'What did Search bring to AlfWorld' needs Search's
    own probability, Search's own shift and Search's own alpha ON THE ROW."""
    from verl.trainer.ppo.sign_weights import (
        PAIR_EVENT_FLOATS, PAIR_EVENT_INTS, PAIR_STATES,
    )

    rows, _built, stats = _pair_event_rows()
    assert rows, "the sampler produced nothing"
    for r in rows:
        assert set(PAIR_EVENT_INTS) - {"dst", "src", "pair_state", "state", "role"} <= set(r)
        assert set(PAIR_EVENT_FLOATS) <= set(r)
        assert r["dst"] != r["src"], "the diagonal is structurally absent"
        assert r["pair_state"] in PAIR_STATES
        assert 0.0 < r["p_base"] <= 1.0 and 0.0 < r["p_source"] <= 1.0
        assert 0.0 < r["p_on"] <= 1.0 and 0.0 < r["p_student"] <= 1.0
        assert r["alpha_diagnostic_source"] == pytest.approx(0.6), "reported, not applied"
        assert "context_ids" in r and len(r["context_ids"]) == 5
    # Every stratum is present and they are not the same rows.
    assert {r["stratum"] for r in rows} == set(stats.STRATA)


def test_the_pair_event_sampling_is_per_cell_so_a_rare_pair_survives():
    """A global top-N is dominated by whichever ordered pair is loudest, and the
    event this arm exists to find is a minority of a minority. Every cell that
    occurred at all has to be represented."""
    rows, _built, stats = _pair_event_rows(per_group=2)
    seen = {}
    for r in rows:
        if r["stratum"] != "top_shift":
            continue
        seen.setdefault((r["dst"], r["src"], r["pair_state"]), 0)
        seen[(r["dst"], r["src"], r["pair_state"])] += 1
    assert len(seen) >= 4, sorted(seen)
    assert max(seen.values()) <= 2, "per_group is a per-cell cap, not a global one"


def test_the_two_top_strata_rank_on_different_quantities():
    """One is about the evidence, the other about the effect on the student.
    Collapsing them is exactly the conflation this whole split exists to stop."""
    rows, _built, _stats = _pair_event_rows(per_group=3)
    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)
    shift_top = max(abs(r["source_attributed_kl_shift"]) for r in by["top_shift"])
    push_top = max(abs(r["extra_logit_push"]) for r in by["top_push"])
    assert shift_top >= max(
        abs(r["source_attributed_kl_shift"]) for r in by["spread"]
    ), "top_shift really is the top of its key"
    assert push_top >= max(abs(r["extra_logit_push"]) for r in by["spread"])
    # And the two rankings do not coincide -- if they did, one of them is
    # measuring the other and the split buys nothing.
    ids = lambda name: {(r["dst"], r["src"], r["token_id"], r["position"]) for r in by[name]}
    assert ids("top_shift") != ids("top_push")


def test_the_pair_event_row_reports_the_raw_shift_and_the_standardized_one():
    """delta_*_raw is in nats and delta_*_std is in the teacher's own RMS units.
    A single column would make report_epsilon incomparable across teachers or
    the nats incomparable with the model's own log-probs."""
    rows, built, _stats = _pair_event_rows()
    assert any(
        r["delta_source_raw"] != pytest.approx(r["delta_source_std"], rel=1e-6)
        for r in rows
    ), "the two columns are not the same number"
    for r in rows:
        # p_source is a probability; delta_source_raw is its log-ratio to base,
        # so exp of the second times p_base recovers the first.
        assert r["p_source"] == pytest.approx(
            r["p_base"] * math.exp(r["delta_source_raw"]), rel=1e-4
        )


def test_a_masked_position_never_reaches_the_pair_dump():
    rows, _built, _stats = _pair_event_rows()
    for r in rows:
        assert r["position"] < r["row_len"] or r["row_len"] == 0


def test_an_empty_cell_does_not_hand_its_slots_to_the_loudest_one():
    """The per-cell guarantee is the whole point. topk over an all-masked cell
    still returns indices -- of rows belonging to OTHER cells -- so retaining
    their unmasked scores lets a cell that never fired donate its slots to the
    cell that fired most, and the stratification silently becomes a global
    top-N wearing per-cell labels."""
    import collections

    rows, _built, stats = _pair_event_rows(per_group=2)
    per_cell = collections.Counter(
        (r["stratum"], r["dst"], r["src"], r["pair_state"]) for r in rows
    )
    assert per_cell, "the sampler produced nothing"
    over = {k: v for k, v in per_cell.items() if v > stats.per_group}
    assert not over, over
    # And the total is at most the cap, not exactly it -- a run where some cells
    # never fired must come out SHORT rather than padded from elsewhere.
    for stratum in stats.STRATA:
        n = sum(v for k, v in per_cell.items() if k[0] == stratum)
        assert n <= stats.n_groups * stats.per_group
    assert sum(per_cell.values()) < len(stats.STRATA) * stats.n_groups * stats.per_group


class _FakeGather:
    """Emulate ``all_gather`` over R ranks that ran identical data."""

    def __init__(self, ranks):
        self.ranks = ranks
        self._saved = None

    def __enter__(self):
        import verl.trainer.ppo.sign_weights as mod

        self._saved = mod.torch.distributed
        ranks = self.ranks

        class _Dist:
            ReduceOp = type("ReduceOp", (), {"SUM": "sum"})

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return ranks

            @staticmethod
            def get_rank():
                return 0

            @staticmethod
            def all_gather(out, src):
                for t in out:
                    t.copy_(src)

            @staticmethod
            def all_reduce(t, op=None):
                t.mul_(ranks)

        mod.torch.distributed = _Dist
        return self

    def __exit__(self, *a):
        import verl.trainer.ppo.sign_weights as mod

        mod.torch.distributed = self._saved
        return False


def test_the_pair_dump_gathers_every_rank_and_still_caps_each_cell():
    """A rank-0-local sample is world_size times smaller than a reader assumes,
    and for a cell that fires a handful of times a step that is the difference
    between a few examples and none. The gather is FIXED SHAPE -- groups x
    per_group whatever the batch held -- so it is a collective on the config and
    cannot hang on a rank whose micro-batches held nothing."""
    import collections

    solo, _b, stats = _pair_event_rows(per_group=2)
    with _FakeGather(3):
        gathered, _b2, stats2 = _pair_event_rows(per_group=2)
    # Same data on every rank, so the re-selection keeps the same per-cell cap.
    per_cell = collections.Counter(
        (r["stratum"], r["dst"], r["src"], r["pair_state"]) for r in gathered
    )
    assert not {k: v for k, v in per_cell.items() if v > stats2.per_group}
    # The rows survive the pack/unpack round trip intact: ids stay exact through
    # the float64 transport and the labels come back as labels.
    assert {r["token_id"] for r in gathered} <= {r["token_id"] for r in solo} | {
        r["token_id"] for r in gathered
    }
    for r in gathered:
        assert isinstance(r["dst"], str) and isinstance(r["src"], str)
        assert r["token_id"] == int(r["token_id"])
        assert 0.0 < r["p_source"] <= 1.0


# --------------------------------------------------------------------------- #
# the source-wise decomposition of the added push
# --------------------------------------------------------------------------- #
def test_the_logit_push_decomposition_is_exact():
    """W - 1 = B_shared/mu + sum_m B_m/mu + (1/mu - 1), to float32.

    The whole per-source attribution rests on this being an identity rather than
    an approximation: if the parts only nearly add up, "Search supplied 30% of
    the push at this token" is a correlated summary and not a share.
    """
    for kwargs in ({}, {"alpha": 0.0}, {"alpha": 1.0}, {"seed": 7}, {"n_off": 1}):
        got, _ = _built(**kwargs)
        parts = (
            got["push_shared"] + got["push_by_source"].sum(dim=-1) + got["push_normalizer"]
        )
        assert torch.allclose(parts, got["weight"] - 1.0, atol=1e-6), kwargs


def test_the_decomposition_is_zero_wherever_the_weight_was_neutralised():
    """A cold-start row and a row whose task has no RMS both sit at W = 1, and
    every share has to be zero there -- otherwise the attribution reports a push
    at positions the mechanism never touched."""
    got, _ = _built(normalizer=None)
    assert torch.equal(got["weight"], torch.ones_like(got["weight"]))
    for key in ("push_shared", "push_normalizer"):
        assert torch.count_nonzero(got[key]) == 0, key
    assert torch.count_nonzero(got["push_by_source"]) == 0

    partial, _ = _built(diag_valid=[True, False, True])
    parts = (
        partial["push_shared"] + partial["push_by_source"].sum(dim=-1)
        + partial["push_normalizer"]
    )
    assert torch.allclose(parts, partial["weight"] - 1.0, atol=1e-6)


def _no_excess(seed=90, bs=4, resp=3, k=5):
    """Every off-task teacher exactly on the on-task teacher's ceiling: the
    sources speak, agree, and add nothing. The source channel is empty for a
    reason that is a property of the evidence rather than of a gate."""
    torch.manual_seed(seed)
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    off = torch.stack([on.clone(), on.clone()], dim=-1)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    return build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
    )


def test_the_normalizer_offset_is_nobody_s_and_is_not_negligible():
    """With the sources adding nothing beyond the on-task teacher they
    contribute nothing, yet W is not 1 -- mu is a whole-task divisor. Folding
    that offset into the sources would credit a teacher for an effect it had no
    part in."""
    got = _no_excess()
    assert torch.count_nonzero(got["push_by_source"]) == 0
    moved = (got["weight"] - 1.0).abs().max()
    assert moved > 0.01, "the offset alone should move the weight"
    assert torch.allclose(
        got["push_normalizer"] + got["push_shared"], got["weight"] - 1.0, atol=1e-6
    )


def test_activity_separates_a_source_with_nothing_to_add_from_a_gated_one():
    """``evidence_by_source`` is zero for two opposite reasons and the
    pre-gate activity is what tells them apart:

      the sources SPOKE and the on-task teacher already covered it -- the
      channel is redundant here, and the corroboration is carrying the position;
      the sources spoke and DISAGREED with each other -- the gate closed, and
      the position has excess evidence nobody is willing to vouch for.

    Reporting only the applied column makes those the same finding."""
    covered = _no_excess()
    assert torch.count_nonzero(covered["evidence_by_source"]) == 0
    assert covered["activity_by_source"].abs().sum() > 0, "the sources did speak"
    assert float(covered["q_sim"].max()) == pytest.approx(1.0), "and agreed perfectly"
    assert float(covered["evidence_shared"].abs().sum()) > 0, "corroboration took it"

    torch.manual_seed(91)
    bs, resp, k = 4, 3, 5
    on, base = _lp(bs, resp, k), _lp(bs, resp, k)
    # Loud and exactly opposed: excess on both columns, gate shut on all of it.
    off = torch.stack([base + 4.0 * (on - base), base - 4.0 * (on - base)], dim=-1)
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    split = build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, student_logprob=_student_like(on), response_mask=None, task_ids=task_ids, off_plane_tasks=planes,
        diag=torch.ones(3), alpha_table=torch.zeros(3, 3),
        diag_valid=torch.ones(3, dtype=torch.bool),
        normalizer={"mean": torch.full((3,), 1.2), "valid": torch.ones(3, dtype=torch.bool)},
    )
    assert float(split["q_sim"].max()) == pytest.approx(0.0), "the teachers oppose"
    assert float(split["source_exclusive"].max()) > 0.0, "there WAS excess to take"
    assert torch.count_nonzero(split["evidence_by_source"]) == 0
    assert split["activity_by_source"].abs().sum() > 0


def test_the_source_outcome_table_splits_one_source_s_effect_by_outcome():
    from verl.trainer.ppo.cross_teacher_kl_weight import SourceOutcomeStats

    acc = SourceOutcomeStats(n_tasks=3, device="cpu")
    # Two rows, same destination and sources. Source 0 acts only on the row with
    # positive advantage, source 1 only on the other.
    push = torch.tensor([
        [[0.5, 0.0], [0.5, 0.0]],
        [[0.0, 0.25], [0.0, 0.25]],
    ])                                                        # (2, 2, 2)
    acc.update(
        push_by_source=push,
        teacher_kl=torch.ones(2, 2), response_mask=torch.ones(2, 2),
        advantage=torch.tensor([1.0, -1.0]),
        task_ids=torch.tensor([0, 0]),
        off_plane_tasks=torch.tensor([[1, 2], [1, 2]]),
        reward=torch.tensor([1.0, 0.0]),
    )
    m = acc.metrics(task_names=TASKS)
    a = "kl_weight/source_outcome/search__on__alfworld"
    b = "kl_weight/source_outcome/webshop__on__alfworld"
    # Pooled over both rows, so each source's effect is halved by the row it did
    # nothing on -- which is the point: a source that fires on half the rollouts
    # has half the budget share of one that fires on all of them.
    assert m[f"{a}/all/effect"] == pytest.approx(0.25)
    assert m[f"{a}/adv_positive/effect"] == pytest.approx(0.5)
    assert m[f"{a}/adv_negative/effect"] == pytest.approx(0.0)
    assert m[f"{b}/adv_negative/effect"] == pytest.approx(0.25)
    assert m[f"{b}/adv_positive/effect"] == pytest.approx(0.0)
    assert m[f"{a}/reward_positive/effect"] == pytest.approx(0.5)
    assert m[f"{a}/all/n_rows"] == pytest.approx(2.0)


def test_the_source_outcome_correlation_is_per_source():
    """The pooled outcome table sums the sources out, so a source whose spending
    tracks the advantage and one whose spending opposes it are invisible in it
    as long as they cancel."""
    from verl.trainer.ppo.cross_teacher_kl_weight import SourceOutcomeStats

    n = 40
    a = torch.linspace(-1.0, 1.0, n)
    with_adv = (a - a.min() + 0.05)
    against = (a.max() - a + 0.05)
    push = torch.stack([with_adv, against], dim=-1).reshape(n, 1, 2).expand(n, 3, 2)
    acc = SourceOutcomeStats(n_tasks=3, device="cpu")
    acc.update(
        push_by_source=push.contiguous(), teacher_kl=torch.ones(n, 3),
        response_mask=torch.ones(n, 3), advantage=a,
        task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2]]).expand(n, 2),
    )
    m = acc.metrics(task_names=TASKS)
    assert m["kl_weight/source_outcome/search__on__alfworld/corr_adv_source_effect"] > 0.99
    assert m["kl_weight/source_outcome/webshop__on__alfworld/corr_adv_source_effect"] < -0.99


def test_the_push_table_reports_how_much_of_the_push_it_can_name():
    """The rows name student top-k tokens; the OPD term also acts on the tail
    bucket, which has none. Quoting the ranking without that share states a
    coverage the table does not have."""
    from verl.trainer.ppo.cross_teacher_kl_weight import LogitPushTokens

    acc = LogitPushTokens(vocab_size=16, n_tasks=2, device="cpu", top_n=4)
    acc.update(
        support_ids=torch.tensor([[[1, 2]]]),
        g0=torch.tensor([[[3.0, 1.0]]]),
        g0_tail=torch.tensor([[4.0]]),
        weight=torch.tensor([[2.0]]),
        coef_applied_weight=torch.tensor([[2.0]]),
        response_mask=torch.ones(1, 1),
        task_ids=torch.tensor([0]),
    )
    m = acc.scalar_metrics(task_names=["a", "b"])
    # |W-1| = 1: support carries 3 + 1 = 4, the tail carries 4.
    assert m["kl_weight/push/support_extra_abs_share"] == pytest.approx(0.5)
    assert m["kl_weight/push/tail_extra_abs_share"] == pytest.approx(0.5)
    assert m["kl_weight/push/tail_weighted_abs_share"] == pytest.approx(0.5)
    assert m["kl_weight/a/push/support_extra_abs_share"] == pytest.approx(0.5)


def test_the_coverage_shares_are_absent_rather_than_wrong_without_the_tail():
    """Defaulting the tail to zero would report 100% coverage for a table that
    simply was not told what it was missing."""
    from verl.trainer.ppo.cross_teacher_kl_weight import LogitPushTokens

    acc = LogitPushTokens(vocab_size=16, n_tasks=2, device="cpu", top_n=4)
    acc.update(
        support_ids=torch.tensor([[[1, 2]]]), g0=torch.tensor([[[3.0, 1.0]]]),
        weight=torch.tensor([[2.0]]), coef_applied_weight=torch.tensor([[2.0]]),
        response_mask=torch.ones(1, 1), task_ids=torch.tensor([0]),
    )
    m = acc.scalar_metrics(task_names=["a", "b"])
    assert "kl_weight/push/support_extra_abs_share" not in m
    assert m["kl_weight/push/extra_abs_total"] == pytest.approx(4.0)


def test_the_pre_alpha_column_is_not_defaulted_to_the_post_alpha_one():
    """PairEvidenceStats and PairStateEvidenceStats both take activity as an
    optional argument. Defaulting it to evidence would make the vetoed case
    unreadable in exactly the table built to read it."""
    from verl.trainer.ppo.cross_teacher_kl_weight import PairEvidenceStats

    acc = PairEvidenceStats(n_tasks=3, device="cpu")
    acc.update(
        evidence=torch.zeros(1, 2, 1), shift=torch.zeros(1, 2, 1),
        response_mask=torch.ones(1, 2), task_ids=torch.tensor([0]),
        off_plane_tasks=torch.tensor([[1]]),
        activity=torch.full((1, 2, 1), 0.75),
    )
    m = acc.metrics(task_names=TASKS)
    head = "kl_weight/evidence/search__on__alfworld"
    assert m[f"{head}/source_shift_mean"] == pytest.approx(0.0)
    assert m[f"{head}/source_raw_shift_mean"] == pytest.approx(0.75)

    silent = PairEvidenceStats(n_tasks=3, device="cpu")
    silent.update(
        evidence=torch.zeros(1, 2, 1), shift=torch.zeros(1, 2, 1),
        response_mask=torch.ones(1, 2), task_ids=torch.tensor([0]),
        off_plane_tasks=torch.tensor([[1]]),
    )
    ms = silent.metrics(task_names=TASKS)
    assert ms[f"{head}/source_raw_shift_mean"] == pytest.approx(0.0)


def test_the_event_push_columns_add_to_the_total_on_every_row():
    """extra_logit_push is the whole mechanism's effect and is IDENTICAL across
    the source rows of one candidate, which is what made the dump unable to say
    'Search raised this AlfWorld token'. The split has to be exact, or the
    per-source column is an apportionment dressed as an attribution."""
    rows, _built, _stats = _pair_event_rows(per_group=3)
    assert rows
    for r in rows:
        parts = (
            r["extra_push_sources_all"] + r["extra_push_shared"] + r["extra_push_normalizer"]
        )
        assert parts == pytest.approx(r["extra_logit_push"], abs=1e-9, rel=1e-5), r
        # The per-source column is a share of the all-source one, and with two
        # off-task planes it is strictly smaller wherever the other one acted.
        assert abs(r["extra_push_source"]) <= abs(r["extra_push_sources_all"]) + 1e-12


def test_the_per_source_push_column_differs_across_the_sources_of_one_candidate():
    """The failure this column fixes: every source row of a candidate carrying
    the same total. If they still agree, the attribution is decorative."""
    rows, _built, _stats = _pair_event_rows(per_group=4)
    by_candidate = {}
    for r in rows:
        by_candidate.setdefault(
            (r["dst"], r["token_id"], r["position"], r["stratum"]), {}
        )[r["src"]] = r
    shared = [v for v in by_candidate.values() if len(v) > 1]
    assert shared, "no candidate appeared under two sources; widen the fixture"
    assert any(
        len({round(r["extra_push_source"], 12) for r in v.values()}) > 1 for v in shared
    ), "the per-source column is constant across sources -- it is not attributing"
    for v in shared:
        totals = {round(r["extra_logit_push"], 12) for r in v.values()}
        assert len(totals) == 1, "the TOTAL, by contrast, is a property of the candidate"


def _indented_block(src: str, header: str) -> str:
    """The lines under ``header``, by indentation. Source-level because the real
    path is GPU-only, and this is the boundary a CPU suite can still police."""
    lines = src.split("\n")
    i = next(k for k, l in enumerate(lines) if header in l)
    indent = len(lines[i]) - len(lines[i].lstrip())
    j = next(
        (k for k in range(i + 1, len(lines))
         if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= indent),
        len(lines),
    )
    return "\n".join(lines[i:j])


def test_the_cross_teacher_block_never_reads_the_sign_arms_stash():
    """sign_cand_inputs is built inside `if sign_enabled:` and is None whenever
    sign_weight.enable is off.

    This arm is a DIFFERENT mechanism and runs with the sign arm off, so every
    read of that dict here is a crash -- and one that hides until step 2: at step
    1 no cumulative RMS exists, xt_built stays None, and the block that reads it
    never opens. The cross-teacher arm therefore died at step 2 on the first real
    launch, after 13 minutes of rollout, having passed every test in this file.

    The two names it actually needs are the ones the arm already asserts are
    present at the top of `if xt_enabled:`.
    """
    src = _update_policy_source()
    block = _indented_block(src, "if xt_built is not None and xt_collect:")
    assert "sign_push" not in block  # sanity: the slice is not empty
    assert "_xt_token_tables(" in block, "the slice missed the token tables"
    assert "xt_push_token_stats.update(" in block, "the slice missed the push table"
    assert "sign_cand_inputs" not in block, (
        "the cross-teacher block reads the sign arm's stash; it is None when "
        "sign_weight.enable is off, which is how this arm runs"
    )
    for name in ("sign_support_ids", "sign_on_task_logprobs"):
        assert name in block, name


def test_the_names_the_cross_teacher_block_uses_are_asserted_before_it_reads_them():
    """The assert at the top of the arm's own block is what makes those two names
    safe to dereference a hundred lines below. Order, not just presence: an
    assert that ran after the read would document an invariant the code has
    already violated."""
    src = _update_policy_source()
    guard = src.index(
        "assert sign_support_ids is not None and sign_on_task_logprobs is not None"
    )
    reader = src.index("if xt_built is not None and xt_collect:")
    assert guard < reader


def test_the_teacher_forward_does_not_compute_a_log_prob_it_discards():
    """compute_topk_log_prob unpacks `_, _, topk_out` -- the entropy and the
    sampled-token log-prob are thrown away on every micro-batch.

    need_log_prob=True costs a log-softmax and a gather over (n_resp, vocab),
    the widest tensor in the step. The actor path derives the flag for itself;
    this one never passed it, so it defaulted to True and paid on every
    micro-batch of every frozen model -- four models a row on this arm.
    """
    import inspect

    from verl.workers.actor import dp_actor

    src = inspect.getsource(dp_actor.DataParallelPPOActor.compute_topk_log_prob)
    start = src.index("self._forward_micro_batch(")
    depth, end = 0, None
    for i in range(start, len(src)):          # to the call's own closing paren
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None
    call = src[start : end + 1]
    assert "need_log_prob=False" in call, call
    # The result really is discarded -- if a later change starts reading it, the
    # flag above has to be revisited rather than left as a silent None.
    assert "_, _, topk_out = self._forward_micro_batch(" in src
