"""Which teacher decided the corroboration, and which one held it down.

``shared_share`` on the live run is 0.98: the corroboration bonus IS this arm.
But ``c = 1[all agree] * sign * min_j |hat_j|`` is decided by ONE teacher at a
time -- the smallest magnitude, or whoever broke the unanimity -- and the logs
attributed it to nobody. "WebShop and Search corroborated AlfWorld" can be true
while Search alone capped every one of those bonuses.

Two readings, both off tensors ``build_position_weight`` already has:

``bottleneck``  argmin over ``{on} u off``. The module docstring asserts ``c`` is
                capped by ``|hat_on|`` and that the on-task teacher is silent at
                ~64% of teacher mass; this is the measurement behind that.
``without``     ``|c_{-j}|``, leave-one-out on the evidence. It is >= ``|c|`` by
                construction, so one number covers both ways a teacher holds the
                bonus down -- being the minimum, and breaking the unanimity.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        corroboration_attribution,
        decompose_common_residual,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _attr(on, off):
    """``on`` a list of scalars, ``off`` a list of lists. Shapes (1,1,k[,n_off])."""
    hat_on = torch.tensor(on, dtype=torch.float32).view(1, 1, -1)
    hat_off = torch.tensor(off, dtype=torch.float32).view(1, 1, len(on), -1)
    return (
        corroboration_attribution(hat_on=hat_on, hat_off=hat_off),
        decompose_common_residual(hat_on=hat_on, hat_off=hat_off),
    )


# --------------------------------------------------------------------------- #
# who decides the magnitude
# --------------------------------------------------------------------------- #
def test_the_bottleneck_is_the_teacher_holding_the_minimum():
    """Column 0 is the on-task teacher, then the off-task ones in plane order."""
    a, _ = _attr([3.0, 3.0, 2.0], [[5.0, 4.0], [1.0, 2.0], [9.0, 0.5]])
    assert a["bottleneck"].flatten().tolist() == [0, 1, 2]


def test_the_on_task_teacher_is_a_candidate_bottleneck_like_any_other():
    """It is inside the min, not just the sign reference -- which is exactly the
    claim the docstring makes about c being capped by |hat_on|."""
    a, d = _attr([0.2], [[5.0, 6.0]])
    assert a["bottleneck"].flatten().tolist() == [0]
    assert d["common"].flatten().item() == pytest.approx(0.2)


def test_the_margin_is_the_gap_to_the_runner_up():
    """A near-zero margin means the argmin is a coin flip and the attribution
    above should not be read as a decision."""
    a, _ = _attr([1.0, 1.0], [[1.05, 4.0], [3.0, 4.0]])
    m = a["margin"].flatten().tolist()
    assert m[0] == pytest.approx(0.05, abs=1e-6)
    assert m[1] == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# leave-one-out on the evidence
# --------------------------------------------------------------------------- #
def test_dropping_a_teacher_never_lowers_the_corroboration():
    """The property the single number rests on: a dropped teacher can only raise
    the minimum or restore a unanimity it was breaking. If this can go negative,
    "suppression" is not a suppression."""
    g = torch.Generator().manual_seed(3)
    hat_on = torch.randn((4, 6, 5), generator=g)
    hat_off = torch.randn((4, 6, 5, 2), generator=g)
    a = corroboration_attribution(hat_on=hat_on, hat_off=hat_off)
    d = decompose_common_residual(hat_on=hat_on, hat_off=hat_off)
    assert bool((a["without"] >= d["common"].abs().unsqueeze(-1) - 1e-6).all())


def test_dropping_the_on_task_teacher_is_the_off_task_only_counterfactual():
    """Column 0 has to equal common_ev exactly -- the same quantity the module
    already reports -- or the columns are not the leave-one-out they claim."""
    g = torch.Generator().manual_seed(11)
    hat_on = torch.randn((3, 4, 5), generator=g)
    hat_off = torch.randn((3, 4, 5, 2), generator=g)
    a = corroboration_attribution(hat_on=hat_on, hat_off=hat_off)
    d = decompose_common_residual(hat_on=hat_on, hat_off=hat_off)
    assert torch.allclose(a["without"][..., 0], d["common_ev"], atol=0)


def test_a_teacher_that_broke_the_unanimity_shows_up_as_suppression():
    """The case a bottleneck alone cannot see: c is 0, so there is no argmin to
    attribute, and yet one teacher is the entire reason the bonus is missing."""
    # on +1, off_0 +2 (agrees), off_1 -3 (breaks it).
    a, d = _attr([1.0], [[2.0, -3.0]])
    assert d["common"].flatten().item() == 0.0
    without = a["without"].flatten().tolist()
    assert without[2] == pytest.approx(1.0), "dropping the dissenter restores min(1,2)"
    assert without[1] == pytest.approx(0.0), "dropping the agreeing one leaves the dissent"


def test_a_teacher_that_is_neither_reads_as_exactly_zero():
    """Redundancy has to be distinguishable from suppression, or every teacher
    looks necessary."""
    # off_1 is the largest and agrees: it neither caps the min nor blocks it.
    a, d = _attr([1.0], [[2.0, 9.0]])
    c = d["common"].abs().flatten().item()
    without = a["without"].flatten().tolist()
    assert c == pytest.approx(1.0)
    assert without[2] - c == pytest.approx(0.0), "a redundant teacher suppresses nothing"
    assert without[0] - c == pytest.approx(1.0), "dropping on-task lifts the cap to min(2,9)"


def test_a_silent_on_task_teacher_blocks_the_bonus_and_the_column_says_so():
    """The state the module calls neutral_on_task_silent, from the attribution
    side: sign(hat_on) == 0 kills the unanimity outright."""
    a, d = _attr([0.0], [[2.0, 3.0]])
    assert d["common"].flatten().item() == 0.0
    assert a["without"].flatten().tolist()[0] == pytest.approx(2.0)


def test_the_columns_line_up_with_the_off_task_planes():
    """The caller maps column 1+m to off-plane m; a transposed pair table would
    otherwise credit the wrong teacher and read as a finding."""
    a, _ = _attr([5.0], [[0.25, 7.0]])
    assert a["bottleneck"].flatten().tolist() == [1]
    assert a["without"].shape[-1] == 3


# --------------------------------------------------------------------------- #
# the accumulator
# --------------------------------------------------------------------------- #
TASKS = ["alfworld", "search", "webshop"]


def _acc(*, on, off, task_ids, off_planes, prob=None, tie_epsilon=0.05):
    """One position, ``len(on)`` candidates per row. Shapes (bs, 1, k[, n_off])."""
    from verl.trainer.ppo.cross_teacher_kl_weight import CorroborationAttributionStats

    hat_on = torch.tensor(on, dtype=torch.float32).unsqueeze(1)          # (bs, 1, k)
    hat_off = torch.tensor(off, dtype=torch.float32).unsqueeze(1)        # (bs, 1, k, n_off)
    p = torch.ones_like(hat_on) if prob is None else torch.tensor(prob).unsqueeze(1)
    acc = CorroborationAttributionStats(n_tasks=3, device="cpu", tie_epsilon=tie_epsilon)
    acc.update(
        attribution=corroboration_attribution(hat_on=hat_on, hat_off=hat_off),
        common=decompose_common_residual(hat_on=hat_on, hat_off=hat_off)["common"],
        teacher_prob=p,
        response_mask=torch.ones(hat_on.size(0), 1),
        task_ids=torch.tensor(task_ids),
        off_plane_tasks=torch.tensor(off_planes),
    )
    return acc.metrics(task_names=TASKS)


def test_the_on_task_teacher_gets_a_slot_and_not_a_self_pair_label():
    """``{src}__on__{dst}`` cannot name the on-task teacher without reading as a
    task paired with itself, which is the one cell the module's own claim about
    ``|hat_on|`` capping ``c`` lives in."""
    m = _acc(on=[[1.0]], off=[[[4.0, 5.0]]], task_ids=[0], off_planes=[[1, 2]])
    assert "kl_weight/legacy_hard_common/alfworld/on_task/bottleneck_share" in m
    assert not any("alfworld__on__alfworld" in k for k in m)
    assert m["kl_weight/legacy_hard_common/alfworld/on_task/bottleneck_share"] == pytest.approx(1.0)


def test_the_bottleneck_shares_are_a_partition_of_the_applied_corroboration():
    """Weighted by p|c| -- the per-candidate term of evidence_shared -- so the
    shares divide the corroboration the objective APPLIED. Weighted by candidate
    count instead, a million near-zero bonuses would outvote the ones that moved
    the loss."""
    m = _acc(
        # candidate 0: on-task caps at 1. candidate 1: search (plane 0) caps at 2.
        on=[[1.0, 9.0]],
        off=[[[4.0, 5.0], [2.0, 8.0]]],
        task_ids=[0], off_planes=[[1, 2]],
    )
    head = "kl_weight/legacy_hard_common/alfworld"
    shares = {
        s: m[f"{head}/{s}/bottleneck_share"]
        for s in ("on_task", "search", "webshop") if f"{head}/{s}/bottleneck_share" in m
    }
    assert sum(shares.values()) == pytest.approx(1.0)
    # 1 of the 3 nats of applied corroboration is the on-task cap, 2 are search's.
    assert shares["on_task"] == pytest.approx(1.0 / 3.0)
    assert shares["search"] == pytest.approx(2.0 / 3.0)
    # A teacher that never bound the minimum reports 0, it does not vanish:
    # an absent key reads as "this pair never occurred", which is the
    # opposite finding about a teacher that was consulted every time.
    assert shares["webshop"] == pytest.approx(0.0)


def test_a_teacher_can_be_zero_on_bottleneck_and_dominant_on_suppression():
    """The reading no existing metric can express, and the reason both columns
    are here: a teacher that never sets the bonus and cancels it outright."""
    # on +1, search +2 (agrees), webshop -3 (vetoes every unanimity) -> c = 0.
    m = _acc(on=[[1.0]], off=[[[2.0, -3.0]]], task_ids=[0], off_planes=[[1, 2]])
    head = "kl_weight/legacy_hard_common/alfworld"
    assert m[f"{head}/shared_mass_mean"] == pytest.approx(0.0)
    # No applied mass to take a share of, so the share keys are absent by
    # design; suppression is still counted, which is the whole point.
    assert not any(k.endswith("/bottleneck_share") for k in m)


def test_suppression_is_reported_against_the_corroboration_that_survived():
    """A ratio above 1 means the teacher cancels more than the arm applies."""
    m = _acc(
        # candidate 0: unanimous, c = 1 (on-task caps). candidate 1: webshop vetoes.
        on=[[1.0, 1.0]],
        off=[[[4.0, 5.0], [2.0, -3.0]]],
        task_ids=[0], off_planes=[[1, 2]],
    )
    head = "kl_weight/legacy_hard_common/alfworld"
    assert m[f"{head}/shared_mass_mean"] == pytest.approx(0.5), "1 nat over 2 candidates"
    # webshop: candidate 0 it is redundant (|c_-w| = min(1,4) = 1, no lift);
    # candidate 1 dropping it restores min(1,2) = 1. Total lift 1 over applied 1.
    assert m[f"{head}/webshop/suppression_ratio"] == pytest.approx(1.0)
    # search: redundant on both -- candidate 1 stays vetoed without it.
    assert m[f"{head}/search/suppression_ratio"] == pytest.approx(0.0)
    # on-task: dropping it lifts candidate 0's cap 1 -> min(4,5) = 4.
    assert m[f"{head}/on_task/suppression_ratio"] == pytest.approx(3.0)


def test_a_near_tie_is_flagged_rather_than_counted_as_a_decision():
    """argmin names a teacher even between two that are within noise of each
    other, and a share built out of coin flips reads exactly like one built out
    of decisions."""
    m = _acc(on=[[1.0]], off=[[[1.01, 5.0]]], task_ids=[0], off_planes=[[1, 2]],
             tie_epsilon=0.05)
    head = "kl_weight/legacy_hard_common/alfworld"
    assert m[f"{head}/on_task/near_tie_share"] == pytest.approx(1.0)

    clear = _acc(on=[[1.0]], off=[[[5.0, 6.0]]], task_ids=[0], off_planes=[[1, 2]],
                 tie_epsilon=0.05)
    assert clear[f"{head}/on_task/near_tie_share"] == pytest.approx(0.0)


def test_the_columns_are_filed_under_this_row_s_own_off_task_planes():
    """``sign_off_tasks`` differs per row -- plane 0 is search on an AlfWorld row
    and alfworld on a Search row. Filing by column index would merge two
    different teachers into one cell and the number would still look plausible."""
    m = _acc(
        on=[[9.0], [9.0]],
        # row 0 (alfworld): planes [search, webshop], search caps at 1.
        # row 1 (search):   planes [alfworld, webshop], alfworld caps at 1.
        off=[[[1.0, 5.0]], [[1.0, 5.0]]],
        task_ids=[0, 1], off_planes=[[1, 2], [0, 2]],
    )
    assert m["kl_weight/legacy_hard_common/alfworld/search/bottleneck_share"] == pytest.approx(1.0)
    assert m["kl_weight/legacy_hard_common/search/alfworld/bottleneck_share"] == pytest.approx(1.0)
    # The other side of the gate: webshop was never a DESTINATION here, so it
    # gets no head at all -- distinct from the zeros a consulted-but-redundant
    # teacher reports above.
    assert not any(k.startswith("kl_weight/legacy_hard_common/webshop/") for k in m)


def test_masked_positions_and_unavailable_rows_contribute_nothing():
    """A row with no RMS yet comes out of build_position_weight with zeroed
    shifts; a padded position is not a candidate at all. Either one counted
    would inflate the denominator the shares are taken out of."""
    from verl.trainer.ppo.cross_teacher_kl_weight import CorroborationAttributionStats

    # The two positions differ in BOTH the mass and the bottleneck, so a mask
    # that is accepted and ignored cannot land on the same numbers.
    hat_on = torch.tensor([[[1.0], [9.0]]])            # (1, 2, 1)
    hat_off = torch.tensor([[[[4.0, 5.0]], [[3.0, 6.0]]]])
    acc = CorroborationAttributionStats(n_tasks=3, device="cpu")
    acc.update(
        attribution=corroboration_attribution(hat_on=hat_on, hat_off=hat_off),
        common=decompose_common_residual(hat_on=hat_on, hat_off=hat_off)["common"],
        teacher_prob=torch.ones_like(hat_on),
        response_mask=torch.tensor([[1.0, 0.0]]),
        task_ids=torch.tensor([0]), off_plane_tasks=torch.tensor([[1, 2]]),
    )
    head = "kl_weight/legacy_hard_common/alfworld"
    m = acc.metrics(task_names=TASKS)
    # Unmasked it would be (1 + 3) / 2 = 2.0, and search would hold 3/4 of the
    # bottleneck mass instead of none of it.
    assert m[f"{head}/shared_mass_mean"] == pytest.approx(1.0)
    assert m[f"{head}/on_task/bottleneck_share"] == pytest.approx(1.0)
    assert m[f"{head}/search/bottleneck_share"] == pytest.approx(0.0)
    assert m[f"{head}/on_task/bottleneck_candidate_frac"] == pytest.approx(1.0)


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


def test_the_attribution_rides_out_of_build_position_weight():
    """The accumulator reads it from the built mapping; a name change there is a
    silent KeyError inside a `if xt_on` branch that no CPU test path enters."""
    from verl.trainer.ppo.cross_teacher_kl_weight import build_position_weight

    bs, resp, k, n_off, n_task = 2, 3, 4, 2, 3
    g = torch.Generator().manual_seed(7)
    built = build_position_weight(
        shifts={
            "on": torch.randn((bs, resp, k), generator=g),
            "off": torch.randn((bs, resp, k, n_off), generator=g),
        },
        on_task_logprob=torch.log_softmax(torch.randn((bs, resp, k), generator=g), dim=-1), student_logprob=_student_like(torch.log_softmax(torch.randn((bs, resp, k), generator=g), dim=-1)), response_mask=None,
        task_ids=torch.tensor([0, 1]),
        off_plane_tasks=torch.tensor([[1, 2], [0, 2]]),
        diag=torch.ones(n_task),
        diag_valid=torch.ones(n_task, dtype=torch.bool),
        alpha_table=torch.full((n_task, n_task), 0.5),
    )
    a = built["attribution"]
    assert a["bottleneck"].shape == (bs, resp, k)
    assert a["without"].shape == (bs, resp, k, 1 + n_off)
    assert torch.allclose(a["without"][..., 0], built["common_ev"], atol=0)
