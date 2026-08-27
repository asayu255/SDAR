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
        compute_raw_policy_shifts,
        decompose_common_residual,
        group_center,
        position_pre_weight,
        standardize_policy_shifts,
        tail_logprob,
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


def test_common_ev_is_the_off_task_teachers_own_unanimity():
    """Off-task only, because the on-task teacher is silent at 64% of teacher
    mass and an on-task-inclusive minimum would kill the channel there."""
    assert float(_one(0.0, [3.0, 2.0])["common_ev"]) == pytest.approx(2.0)
    assert float(_one(0.0, [3.0, 2.0])["common"]) == 0.0
    assert float(_one(1.0, [3.0, -2.0])["common_ev"]) == 0.0, "the sources split"


def test_common_ev_ignores_whether_the_on_task_teacher_agrees():
    """Deliberate, and the reason kl_shift_by_state is a required metric.

    A KL term has no direction to disagree with: "both other tasks moved
    decisively here" is a statement about the position, not about who was right.
    """
    agree = float(_one(1.0, [3.0, 2.0])["common_ev"])
    oppose = float(_one(-1.0, [3.0, 2.0])["common_ev"])
    assert agree == pytest.approx(oppose) == pytest.approx(2.0)


def test_a_single_source_cannot_corroborate_itself():
    assert float(_one(1.0, [3.0])["common_ev"]) == 0.0


def test_a_near_zero_shift_attenuates_itself_without_a_deadzone():
    """No fixed gate: the minimum drags the corroboration down continuously, so
    drift noise costs a small number instead of tripping a threshold."""
    for eps in (1e-1, 1e-3, 1e-6):
        # rel, not abs: the shifts are float32 and 0.1 is 0.10000000149 there.
        assert float(_one(1.0, [3.0, eps])["common_ev"]) == pytest.approx(eps, rel=1e-5)


# --------------------------------------------------------------------------- #
# candidate evidence -- the monotonicity that the design turns on
# --------------------------------------------------------------------------- #
def _evidence(on_v, off_v, alpha):
    d = _one(on_v, off_v)
    a = torch.full((1, len(off_v)), float(alpha))
    return float(candidate_kl_evidence(common_ev=d["common_ev"], hat_off=torch.tensor(
        [[[[float(x) for x in off_v]]]]), source_alpha=a))


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5, 0.75, 1.0])
def test_corroboration_never_scores_below_conflict_at_any_alpha(alpha):
    """The defect this formula exists to avoid.

    ``|c| + sum alpha|delta_hat - c|`` subtracts the shared part from every
    source, crediting agreement once and debiting it n_off times, so past
    ``alpha = 1/n_off`` a split scores HIGHER. Here the gap is ``|c_ev|``,
    independent of alpha.
    """
    unanimous = _evidence(1.0, [3.0, 2.0], alpha)
    split = _evidence(1.0, [3.0, -2.0], alpha)
    assert unanimous - split == pytest.approx(2.0, abs=1e-5)


def test_the_broken_alternative_would_have_failed_that():
    """Pinned as a counter-example so the fix cannot be undone as a cleanup."""
    def broken(on_v, off_v, alpha):
        d = _one(on_v, off_v)
        r = d["residual"].reshape(-1)
        return abs(float(d["common"])) + alpha * float(r.abs().sum())

    assert broken(1.0, [3.0, 2.0], 1.0) < broken(1.0, [3.0, -2.0], 1.0)


def test_evidence_is_non_negative_and_zero_only_with_no_signal():
    assert _evidence(0.0, [0.0, 0.0], 1.0) == pytest.approx(0.0)
    for on_v, off_v, a in ((1.0, [3.0, -2.0], 0.3), (-1.0, [-1.0, -1.0], 1.0), (0.0, [2.0, 2.0], 0.0)):
        assert _evidence(on_v, off_v, a) >= 0.0


def test_alpha_zero_leaves_only_the_corroboration_bonus():
    assert _evidence(1.0, [3.0, 2.0], 0.0) == pytest.approx(2.0)


def test_alpha_one_admits_the_full_standardized_source_shift():
    """The full shift, not the residual: alpha is estimated on the residual and
    applied to the whole thing, and mixing those is the reversal above."""
    assert _evidence(1.0, [3.0, 2.0], 1.0) == pytest.approx(2.0 + 3.0 + 2.0)


def test_sources_are_summed_not_averaged():
    one = _evidence(1.0, [3.0, 3.0], 1.0)
    assert one == pytest.approx(3.0 + 3.0 + 3.0), "min is 3, both sources add 3"


def test_alpha_is_per_source():
    d = _one(1.0, [3.0, 2.0])
    e = candidate_kl_evidence(
        common_ev=d["common_ev"],
        hat_off=torch.tensor([[[[3.0, 2.0]]]]),
        source_alpha=torch.tensor([[1.0, 0.0]]),
    )
    assert float(e) == pytest.approx(2.0 + 3.0)


def test_the_evidence_signature_cannot_be_handed_a_residual():
    """Structural: alpha gates the full shift, and the residual belongs to the
    reliability and attribution paths only."""
    import inspect

    params = set(inspect.signature(candidate_kl_evidence).parameters)
    assert params == {"common_ev", "hat_off", "source_alpha"}


# --------------------------------------------------------------------------- #
# position weight
# --------------------------------------------------------------------------- #
def test_the_collapsed_form_matches_the_explicit_sum_over_the_support_and_tail():
    on = _lp(2, 3, 5, seed=20)
    e = torch.rand(2, 3, 5)
    p = on.exp()
    explicit = (p * (1.0 + e)).sum(-1) + (1.0 - p.sum(-1))
    assert torch.allclose(position_pre_weight(evidence=e, on_task_logprob=on), explicit, atol=1e-6)


def test_no_evidence_leaves_the_position_untouched():
    on = _lp(2, 3, 4, seed=21)
    got = position_pre_weight(evidence=torch.zeros(2, 3, 4), on_task_logprob=on)
    assert torch.allclose(got, torch.ones(2, 3), atol=1e-6)


def test_the_teachers_own_mass_decides_how_much_a_candidate_counts():
    on = torch.log(torch.tensor([[[0.80, 0.01]]]))
    heavy = position_pre_weight(evidence=torch.tensor([[[1.0, 0.0]]]), on_task_logprob=on)
    light = position_pre_weight(evidence=torch.tensor([[[0.0, 1.0]]]), on_task_logprob=on)
    assert float(heavy) == pytest.approx(1.80, abs=1e-5)
    assert float(light) == pytest.approx(1.01, abs=1e-5)


def test_the_tail_is_neutral_so_a_thin_support_is_modulated_thinly():
    thin = torch.log(torch.tensor([[[0.05, 0.05]]]))   # 90% of the mass is tail
    got = position_pre_weight(evidence=torch.full((1, 1, 2), 2.0), on_task_logprob=thin)
    assert float(got) == pytest.approx(1.0 + 0.1 * 2.0, abs=1e-5)


def test_the_weight_grows_linearly_in_the_evidence():
    on = _lp(1, 1, 3, seed=22)
    e = torch.rand(1, 1, 3)
    a = position_pre_weight(evidence=e, on_task_logprob=on) - 1.0
    b = position_pre_weight(evidence=e * 3.0, on_task_logprob=on) - 1.0
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
        shifts=shifts, on_task_logprob=on, task_ids=task_ids, off_plane_tasks=planes,
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
        shifts=ctx["shifts"], on_task_logprob=ctx["on"], task_ids=ctx["task_ids"],
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


def test_the_probe_bracket_covers_alpha_zero_and_one_and_never_touches_the_weight():
    """The Phase-2 go/no-go is judged on the top of this bracket. At alpha=0 the
    corroboration channel is alone, and it is structurally the minority one --
    the on-task teacher is silent at 64% of teacher mass -- so a gate read there
    would be measuring a lower bound and calling it the mechanism."""
    got, _ = _built(alpha=0.0)
    assert set(got["probe_pre_weight"]) == {probe_name(a) for a in PROBE_ALPHAS}
    lo = got["probe_pre_weight"][probe_name(0.0)]
    hi = got["probe_pre_weight"][probe_name(1.0)]
    assert torch.all(hi >= lo - 1e-6), "more alpha cannot mean less evidence"
    assert float((hi - lo).max()) > 0.0
    # alpha=0 in the TABLE means the training weight equals the alpha=0 probe.
    assert torch.allclose(got["pre_weight"], lo, atol=1e-6)


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
        shifts=ctx["shifts"], on_task_logprob=ctx["on"], task_ids=ctx["task_ids"],
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
        shifts=shifts, on_task_logprob=on, task_ids=torch.zeros(1, dtype=torch.long),
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
        "self.cross_teacher_enabled = self.sign_weight_enabled or self.cross_teacher_kl_weight_enabled",
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


IDENTITY = {"base_path": "Qwen/Qwen3-1.7B", "temperature": 1.0, "task_order": TASKS}


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


def test_an_unknown_sidecar_version_is_refused_rather_than_guessed():
    rms, mean, adv = _accumulated()
    blob = sidecar_state(rms=rms, mean=mean, adv=adv, alpha=None, identity=IDENTITY)
    blob["version"] = 2
    with pytest.raises(AssertionError, match="version"):
        load_sidecar_state(
            blob, rms=rms, mean=mean, adv=adv, identity=IDENTITY,
        )


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
        on_task_logprob=on, task_ids=task_ids, off_plane_tasks=planes,
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
