"""Knowledge-transfer instrumentation for the multitask sign-weighting arms.

Three questions the arm exists to answer and could not:

* **Is transfer happening?** Everything shipped describes the TEACHERS' agreement
  or the SIZE of the rewrite. Nothing said where the student ended up. That is
  :class:`OffTaskLadderStats` (how far the student travelled toward the teachers
  it is not trained on) and :func:`rewrite_decomposition_terms` (what the rewrite
  cost the student, as an exact decomposition).
* **Is what transfers common knowledge?** The unanimity gate makes "agreement"
  mean "all three teachers moved together", i.e. common by construction. The
  pair readings off :class:`SignPairCounts` measure association per ordered pair
  instead, with the marginal sign-propensity divided out.
* **Is it one teacher's specialist knowledge?** That population is hidden inside
  the on-task-silent state, which carries 64% of teacher mass and is never
  decomposed. The blind-spot readings decompose it, with a headroom axis so the
  log-space ceiling can be told from ignorance.

The load-bearing properties are the exact identities and the calibration points
-- they are what let a number be read as a quantity rather than a trend.
"""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.sign_weights import (
        REWRITE_TERMS,
        OffTaskLadderStats,
        ScopeTermStats,
        SignPairCounts,
        _dist_parts,
        _topk_kl_from_parts,
        candidate_weights,
        reweight_teacher_logprobs,
        rewrite_decomposition_terms,
        rewrite_ratio_metrics,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


TASKS = ["alfworld", "search", "webshop"]


def _logprob(bs, resp, k, scale=1.0, seed=None):
    """A full-vocab log-softmax gathered at k ids, so the top-k mass is < 1."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.log_softmax(scale * torch.randn(bs, resp, k + 6), dim=-1)[..., :k]


# --------------------------------------------------------------------------- #
# ScopeTermStats
# --------------------------------------------------------------------------- #
def test_the_scope_accumulator_means_over_exactly_the_valid_positions():
    s = ScopeTermStats(names=["a", "b"], n_tasks=2, device="cpu")
    s.update(
        {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([[10.0, 20.0], [30.0, 40.0]])},
        response_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        task_ids=torch.tensor([0, 1]),
    )
    m = s.metrics(task_names=["x", "y"])
    assert m["sign_weight/a"] == pytest.approx((1 + 2 + 3) / 3)
    assert m["sign_weight/x/a"] == pytest.approx((1 + 2) / 2)
    assert m["sign_weight/y/a"] == pytest.approx(3.0)  # the masked position is gone
    assert m["sign_weight/y/b"] == pytest.approx(30.0)


def test_a_row_with_no_task_reaches_the_pooled_scope_only():
    """adjust_batch's padding and any untagged row are real positions, but filing
    them under a task would invent one."""
    s = ScopeTermStats(names=["a"], n_tasks=2, device="cpu")
    s.update(
        {"a": torch.tensor([[2.0], [6.0]])},
        response_mask=torch.ones(2, 1),
        task_ids=torch.tensor([0, -1]),
    )
    sums = s.sums(task_names=["x", "y"])
    assert sums[None]["n"] == 2 and sums[None]["a"] == pytest.approx(8.0)
    assert sums["x"]["n"] == 1 and sums["x"]["a"] == pytest.approx(2.0)
    assert "y" not in sums


def test_the_scope_accumulator_is_additive_and_drops_a_stale_rendering():
    s = ScopeTermStats(names=["a"], n_tasks=1, device="cpu")
    args = ({"a": torch.ones(1, 1)}, torch.ones(1, 1), torch.tensor([0]))
    s.update(*args)
    assert s.sums()[None]["a"] == pytest.approx(1.0)
    assert s._cpu_cache is not None
    s.update(*args)
    assert s._cpu_cache is None, "folding more in must drop a rendering taken before it"
    assert s.sums()[None]["a"] == pytest.approx(2.0)
    s.all_reduce()
    assert s._cpu_cache is None


# --------------------------------------------------------------------------- #
# The transfer ladder — Q1
# --------------------------------------------------------------------------- #
def test_the_ladder_kl_is_the_same_kl_the_loss_uses():
    """The rungs are hand-written so the five distributions can share their
    exponentials. That is only safe while they agree with the shipped KL."""
    a, b = _logprob(3, 4, 7, seed=0), _logprob(3, 4, 7, seed=1)
    p_a, tail_a = _dist_parts(a)
    _p_b, tail_b = _dist_parts(b)
    torch.testing.assert_close(
        _topk_kl_from_parts(a, p_a, tail_a, b, tail_b), topk_kl_per_token(a, b)
    )


def _ladder(student, on, base, off, tasks, planes, n_tasks=3):
    L = OffTaskLadderStats(n_tasks=n_tasks, device="cpu")
    L.update(
        student_logprob=student, on_task_logprob=on, base_logprob=base,
        off_task_logprobs=off, response_mask=torch.ones(student.size(0), student.size(1)),
        task_ids=tasks, off_plane_tasks=planes,
    )
    return L.metrics(task_names=TASKS)


def test_off_travel_is_zero_at_the_base_and_one_at_the_teacher():
    """The two calibration points that make the ratio readable as a quantity.

    0 at initialisation is not a coincidence: the lock pins
    ``model.path == sign_weight.base_path``, so the student IS the base at step 0.
    1 means the student has moved exactly as far toward this off-task teacher as
    its own teacher sits -- what plain on-task distillation buys, with no
    transfer needed to explain it.
    """
    base, on = _logprob(4, 3, 6, seed=2), _logprob(4, 3, 6, seed=3)
    off = torch.stack([_logprob(4, 3, 6, seed=4), _logprob(4, 3, 6, seed=5)], dim=-1)
    tasks = torch.tensor([0, 1, 2, 0])
    planes = torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])

    at_base = _ladder(base, on, base, off, tasks, planes)
    at_teacher = _ladder(on, on, base, off, tasks, planes)
    travel = [k for k in at_base if "/off_travel/" in k]
    assert len(travel) == 6, sorted(travel)  # every ordered pair, dst != src
    for k in travel:
        assert at_base[k] == pytest.approx(0.0, abs=1e-9)
        assert at_teacher[k] == pytest.approx(1.0, abs=1e-9)


def test_off_travel_is_keyed_by_the_ordered_pair():
    """"What alfworld picked up from search" and the reverse are different
    quantities; the shipped agreement matrix is already asymmetric."""
    m = _ladder(
        _logprob(3, 2, 5, seed=6), _logprob(3, 2, 5, seed=7), _logprob(3, 2, 5, seed=8),
        torch.stack([_logprob(3, 2, 5, seed=9), _logprob(3, 2, 5, seed=10)], dim=-1),
        torch.tensor([0, 1, 2]), torch.tensor([[1, 2], [0, 2], [0, 1]]),
    )
    assert "transfer/off_travel/alfworld__vs__search" in m
    assert "transfer/off_travel/search__vs__alfworld" in m
    # a task is never its own off-task plane
    assert not any("alfworld__vs__alfworld" in k for k in m)


def test_the_anchors_hold_however_far_the_off_task_teacher_drifted():
    """What the ratio actually buys, stated exactly.

    It is NOT scale invariance -- KL is not linear, so rescaling the off-task
    teacher does move an intermediate value, and this test would fail if it
    claimed otherwise. What is exact is the two ANCHORS: 0 means the student has
    not moved and 1 means it has moved as far toward this teacher as its own
    teacher sits, and both hold whatever the off-task teacher is and however far
    it drifted. That is what makes the number readable across pairs whose
    teachers differ 3.7x in measured drift -- as a position between two fixed
    ends, never as a magnitude.
    """
    base, on = _logprob(4, 3, 6, seed=11), _logprob(4, 3, 6, seed=12)
    tasks, planes = torch.tensor([0, 1, 2, 0]), torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])
    pair = "alfworld__vs__search"

    raw = []
    for scale in (0.3, 3.0):
        off = torch.stack(
            [_logprob(4, 3, 6, scale, seed=13), _logprob(4, 3, 6, scale, seed=14)], dim=-1
        )
        at_base = _ladder(base, on, base, off, tasks, planes)
        at_teacher = _ladder(on, on, base, off, tasks, planes)
        assert at_base[f"transfer/off_travel/{pair}"] == pytest.approx(0.0, abs=1e-9)
        assert at_teacher[f"transfer/off_travel/{pair}"] == pytest.approx(1.0, abs=1e-9)
        raw.append(at_teacher[f"transfer/kl_to_off/stu/{pair}"])

    # ...while the raw distance the anchors are built from moves by an order of
    # magnitude with the off-task teacher's own drift. That gap is the reason the
    # ratio exists.
    assert raw[1] > 3 * raw[0]


def test_the_ladder_ignores_padding_and_unnamed_planes():
    base, on = _logprob(2, 2, 5, seed=15), _logprob(2, 2, 5, seed=16)
    off = torch.stack([_logprob(2, 2, 5, seed=17)], dim=-1)
    L = OffTaskLadderStats(n_tasks=3, device="cpu")
    L.update(
        student_logprob=on, on_task_logprob=on, base_logprob=base, off_task_logprobs=off,
        response_mask=torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
        task_ids=torch.tensor([0, -1]),          # row 1 has no task
        off_plane_tasks=torch.tensor([[1], [-1]]),  # ...and no named plane
    )
    m = L.metrics(task_names=TASKS)
    assert m["transfer/kl_to_off/n_positions/alfworld__vs__search"] == pytest.approx(2.0)
    assert len([k for k in m if "/n_positions/" in k]) == 1


# --------------------------------------------------------------------------- #
# The rewrite decomposition — Q1
# --------------------------------------------------------------------------- #
def _decomp(student, on, base, w, teacher_kl=None):
    if teacher_kl is None:
        teacher_kl = topk_kl_per_token(student, reweight_teacher_logprobs(on, w))
    return rewrite_decomposition_terms(
        student_logprob=student, on_task_logprob=on, base_logprob=base,
        candidate_weight=w, teacher_kl=teacher_kl,
    )


def _weights(on, off, base):
    return candidate_weights(
        on, off, base, mode="target", agree_weight=1.5, agree_neg_weight=0.5,
        disagree_weight=1.0, deadzone=0.1,
    )[0]


def test_cf_cost_is_exactly_the_extra_kl_the_rewrite_costs_the_student():
    """``cf_cost == KL(p_s||p~) - KL(p_s||p)``, an identity rather than an
    approximation: substituting ``log p~ = log p + log w - log Z`` leaves
    ``KL(p_s||p) - sum p_s log w + log Z`` because p_s sums to one over the k+1
    categories. Everything downstream reads cf_cost as that difference."""
    student, on, base = _logprob(3, 4, 8, seed=20), _logprob(3, 4, 8, seed=21), _logprob(3, 4, 8, seed=22)
    off = torch.stack([_logprob(3, 4, 8, seed=23), _logprob(3, 4, 8, seed=24)], dim=-1)
    w = _weights(on, off, base)
    t = _decomp(student, on, base, w)
    direct = topk_kl_per_token(student, reweight_teacher_logprobs(on, w)) - topk_kl_per_token(student, on)
    torch.testing.assert_close(t["cf_cost"], direct, atol=2e-6, rtol=0)


def test_the_clamp_residual_is_the_identity_measured_not_assumed():
    """``teacher_kl == control_teacher_kl + cf_cost`` holds analytically; the
    residual is what the 1e-8 tail clamp does to it. It is reported so that a
    task whose teacher_coverage is 1.000 -- where the clamp binds -- can be
    recognised instead of trusted."""
    student, on, base = _logprob(3, 4, 8, seed=25), _logprob(3, 4, 8, seed=26), _logprob(3, 4, 8, seed=27)
    off = torch.stack([_logprob(3, 4, 8, seed=28), _logprob(3, 4, 8, seed=29)], dim=-1)
    t = _decomp(student, on, base, _weights(on, off, base))
    assert float(t["cf_clamp_resid"].abs().max()) < 2e-6


def test_an_inert_weight_makes_every_rewrite_term_exactly_zero():
    """w == 1 everywhere IS the no-rewrite arm. If any term drifted from zero
    there, its value on a live arm would be partly an artefact of the machinery
    rather than of the rewrite."""
    student, on, base = _logprob(3, 4, 8, seed=30), _logprob(3, 4, 8, seed=31), _logprob(3, 4, 8, seed=32)
    t = _decomp(student, on, base, torch.ones(3, 4, 8), teacher_kl=topk_kl_per_token(student, on))
    for name in ("cf_cost", "rewrite_align", "rewrite_span", "rewrite_fisher", "log_z", "cf_clamp_resid"):
        assert float(t[name].abs().max()) < 1e-6, name
    # ...and the control KL is then just the loss's own KL
    torch.testing.assert_close(t["control_teacher_kl"], topk_kl_per_token(student, on))


def test_rewrite_progress_is_minus_one_where_the_student_has_not_moved():
    """The calibration point. -1 is where a run starts by construction, so it is
    the number that says the metric is wired up, not a finding."""
    on, base = _logprob(3, 4, 8, seed=33), _logprob(3, 4, 8, seed=34)
    off = torch.stack([_logprob(3, 4, 8, seed=35), _logprob(3, 4, 8, seed=36)], dim=-1)
    w = _weights(on, off, base)
    s = ScopeTermStats(names=REWRITE_TERMS, n_tasks=1, device="cpu")
    s.update(_decomp(base, on, base, w), response_mask=torch.ones(3, 4), task_ids=torch.zeros(3, dtype=torch.long))
    assert rewrite_ratio_metrics(s.sums())["sign_weight/rewrite_progress"] == pytest.approx(-1.0, abs=1e-5)


def test_rewrite_fisher_is_a_variance_and_survives_a_rescaled_teacher():
    """It is Var_{p_s}(log w), so non-negative -- and unlike every distance here
    it does not carry the teachers' KL coefficients, which differ 3.7x."""
    student, on, base = _logprob(3, 4, 8, seed=37), _logprob(3, 4, 8, seed=38), _logprob(3, 4, 8, seed=39)
    off = torch.stack([_logprob(3, 4, 8, seed=40), _logprob(3, 4, 8, seed=41)], dim=-1)
    t = _decomp(student, on, base, _weights(on, off, base))
    assert float(t["rewrite_fisher"].min()) >= -1e-6
    # depends on w and p_s only: the base cancels out of it entirely
    t2 = _decomp(student, on, _logprob(3, 4, 8, seed=99), _weights(on, off, base))
    torch.testing.assert_close(t["rewrite_fisher"], t2["rewrite_fisher"])


def test_the_log_z_moments_are_a_different_functional_from_inv_z():
    """The arm has only ever reported E[1/Z]. E[log Z] and its variance are what
    say how far the two are allowed to be from each other."""
    on, base = _logprob(4, 5, 8, seed=42), _logprob(4, 5, 8, seed=43)
    off = torch.stack([_logprob(4, 5, 8, seed=44), _logprob(4, 5, 8, seed=45)], dim=-1)
    t = _decomp(_logprob(4, 5, 8, seed=46), on, base, _weights(on, off, base))
    s = ScopeTermStats(names=REWRITE_TERMS, n_tasks=1, device="cpu")
    s.update(t, response_mask=torch.ones(4, 5), task_ids=torch.zeros(4, dtype=torch.long))
    out = rewrite_ratio_metrics(s.sums())
    assert out["sign_weight/log_z_var"] >= 0
    assert out["sign_weight/log_z_mean"] == pytest.approx(float(t["log_z"].mean()), rel=1e-6)


# --------------------------------------------------------------------------- #
# The pair table — Q2 and Q3
# --------------------------------------------------------------------------- #
def _pair_from_cells(cells, deadzone=0.5):
    """A SignPairCounts filled from an explicit {(s_on, s_off): count} table.

    delta is built directly (base is all zeros), so the signs are exactly the
    ones named -- the point is to check the arithmetic on top, not the signs.
    """
    rows = [k for k, n in cells.items() for _ in range(n)]
    n = len(rows)
    pc = SignPairCounts(n_tasks=2, device="cpu")
    pc.update(
        on_task_logprob=torch.tensor([[[r[0] * 1.0]] for r in rows]),
        off_task_logprobs=torch.tensor([[[[r[1] * 1.0]]] for r in rows]),
        base_logprob=torch.zeros(n, 1, 1),
        student_logprob=torch.zeros(n, 1, 1),
        response_mask=torch.ones(n, 1),
        task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=torch.ones(n, 1, dtype=torch.long),
        deadzone=deadzone,
    )
    return pc.metrics(task_names=["dst", "src"], min_count=1)


K = "sign_weight/pair/{}/src__on__dst"


def test_the_pair_table_counts_every_candidate_once_per_plane():
    bs, resp, k = 6, 5, 8
    pc = SignPairCounts(n_tasks=3, device="cpu")
    pc.update(
        on_task_logprob=_logprob(bs, resp, k, seed=50),
        off_task_logprobs=torch.stack([_logprob(bs, resp, k, seed=51), _logprob(bs, resp, k, seed=52)], dim=-1),
        base_logprob=_logprob(bs, resp, k, seed=53),
        student_logprob=_logprob(bs, resp, k, seed=54),
        response_mask=torch.ones(bs, resp),
        task_ids=torch.tensor([0, 1, 2, 0, 1, 2]),
        off_plane_tasks=torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1]]),
        deadzone=0.1,
    )
    count, _mass, _sq, totals = pc._cpu()
    assert float(count.sum()) == bs * resp * k * 2  # once per off-task plane
    # ...but the population totals exactly once, NOT once per plane. Getting this
    # wrong halves every "share of the population" number.
    assert float(totals[:, 0].sum()) == bs * resp * k


def test_the_odds_ratio_is_unmoved_by_a_marginal_skew_that_shifts_agreement():
    """THE property that makes this the headline of the pair family.

    An agreement rate confounds association with each teacher's own propensity
    to raise rather than lower -- and the teachers differ 3.7x in measured drift,
    so that confound is not hypothetical. Both tables below have odds ratio 36;
    only the marginal differs. agree_count moves, the log odds ratio does not.

    The residual gap at small counts is the Haldane +0.5 correction, which exists
    to keep the estimator finite when a cell is empty; it vanishes with count,
    and the confound does not.
    """
    for scale, tol in ((1, 0.05), (100, 5e-4)):
        base = _pair_from_cells({(1, 1): 80 * scale, (1, -1): 20 * scale,
                                 (-1, 1): 10 * scale, (-1, -1): 90 * scale})
        skew = _pair_from_cells({(1, 1): 320 * scale, (1, -1): 20 * scale,
                                 (-1, 1): 40 * scale, (-1, -1): 90 * scale})
        assert abs(skew[K.format("agree_count")] - base[K.format("agree_count")]) > 0.02
        assert abs(skew[K.format("lor")] - base[K.format("lor")]) < tol
    assert base[K.format("lor")] == pytest.approx(math.log(36), abs=1e-3)


def test_independent_signs_give_no_association_and_identical_ones_give_a_lot():
    indep = _pair_from_cells({(a, b): 50 for a in (1, -1) for b in (1, -1)})
    assert indep[K.format("agree_count")] == pytest.approx(0.5)
    assert indep[K.format("lor")] == pytest.approx(0.0, abs=0.05)

    same = _pair_from_cells({(1, 1): 100, (-1, -1): 100})
    assert same[K.format("agree_count")] == pytest.approx(1.0)
    assert same[K.format("lor")] > 5


def test_the_gate_split_partitions_the_population_it_is_a_share_of():
    m = _pair_from_cells({(1, 1): 80, (1, -1): 20, (1, 0): 15, (-1, -1): 90, (-1, 0): 5})
    parts = sum(m[f"sign_weight/gate/{x}_mass/src__on__dst"]
                for x in ("concur", "veto_silent", "veto_dissent"))
    assert parts == pytest.approx(m["sign_weight/gate/pop_mass/src__on__dst"], rel=1e-9)
    fracs = sum(m[f"sign_weight/gate/{x}_frac/src__on__dst"]
                for x in ("concur", "veto_silent", "veto_dissent"))
    assert fracs == pytest.approx(1.0, rel=1e-9)


def test_the_gate_split_tells_a_silent_veto_from_a_dissenting_one():
    """Which one dominates decides what a gate redesign would have to fix: a
    units problem, or tasks that genuinely pull apart."""
    silent = _pair_from_cells({(1, 0): 100, (1, 1): 10})
    dissent = _pair_from_cells({(1, -1): 100, (1, 1): 10})
    assert silent["sign_weight/gate/veto_silent_frac/src__on__dst"] > 0.85
    assert silent["sign_weight/gate/veto_dissent_frac/src__on__dst"] < 0.05
    assert dissent["sign_weight/gate/veto_dissent_frac/src__on__dst"] > 0.85
    assert dissent["sign_weight/gate/veto_silent_frac/src__on__dst"] < 0.05


def test_the_blind_spot_is_where_the_on_task_teacher_said_nothing():
    """Where specialist knowledge would have to live: the on-task teacher did not
    move, so the distillation target cannot carry what the off-task one knows."""
    m = _pair_from_cells({(0, 1): 60, (0, -1): 40, (1, 1): 25})
    assert m["sign_weight/blindspot/off_opinion_mass_pos/src__on__dst"] > 0
    assert m["sign_weight/blindspot/off_opinion_mass_neg/src__on__dst"] > 0
    # the candidates where the on-task teacher DID speak are not in it
    quiet = _pair_from_cells({(1, 1): 100})
    assert quiet["sign_weight/blindspot/off_opinion_mass_pos/src__on__dst"] == pytest.approx(0.0)


def test_the_headroom_axis_separates_arithmetic_silence_from_ignorance():
    """At ``p_0 > exp(-deadzone)`` no model CAN raise a token past the deadzone,
    so its silence there is arithmetic. Without this split a large blind-spot
    mass cannot be told from the log-space ceiling."""
    n = 40
    # base log-prob 0.0 => p_0 = 1 => -base = 0, never above the deadzone
    pc_ceiling = SignPairCounts(n_tasks=2, device="cpu")
    pc_ceiling.update(
        on_task_logprob=torch.zeros(n, 1, 1),
        off_task_logprobs=torch.ones(n, 1, 1, 1),
        base_logprob=torch.zeros(n, 1, 1),
        student_logprob=torch.zeros(n, 1, 1),
        response_mask=torch.ones(n, 1), task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=torch.ones(n, 1, dtype=torch.long), deadzone=0.5,
    )
    m = pc_ceiling.metrics(task_names=["dst", "src"], min_count=1)
    assert m["sign_weight/blindspot/off_opinion_h1_frac_pos/src__on__dst"] == pytest.approx(0.0)

    # base log-prob -5 => plenty of headroom
    pc_room = SignPairCounts(n_tasks=2, device="cpu")
    pc_room.update(
        on_task_logprob=torch.full((n, 1, 1), -5.0),
        off_task_logprobs=torch.full((n, 1, 1, 1), -4.0),
        base_logprob=torch.full((n, 1, 1), -5.0),
        student_logprob=torch.zeros(n, 1, 1),
        response_mask=torch.ones(n, 1), task_ids=torch.zeros(n, dtype=torch.long),
        off_plane_tasks=torch.ones(n, 1, dtype=torch.long), deadzone=0.5,
    )
    m2 = pc_room.metrics(task_names=["dst", "src"], min_count=1)
    assert m2["sign_weight/blindspot/off_opinion_h1_frac_pos/src__on__dst"] == pytest.approx(1.0)


def test_the_effective_sample_size_falls_when_the_mass_concentrates():
    """The mass weight is heavy-tailed -- 64% of teacher mass on 4.2% of
    candidates on the shipped run -- so a mass-weighted rate without an ESS
    cannot be told from step noise."""
    n = 60
    def _ess(logp):
        pc = SignPairCounts(n_tasks=2, device="cpu")
        pc.update(
            on_task_logprob=logp, off_task_logprobs=logp.unsqueeze(-1),
            base_logprob=torch.full_like(logp, -20.0), student_logprob=torch.zeros_like(logp),
            response_mask=torch.ones(n, 1), task_ids=torch.zeros(n, dtype=torch.long),
            off_plane_tasks=torch.ones(n, 1, dtype=torch.long), deadzone=0.1,
        )
        return pc.metrics(task_names=["dst", "src"], min_count=1)["sign_weight/pair/agree_ess/src__on__dst"]

    flat = torch.full((n, 1, 1), math.log(0.5))
    spiky = torch.full((n, 1, 1), math.log(1e-4))
    spiky[0, 0, 0] = math.log(0.9)
    assert _ess(flat) == pytest.approx(float(n), rel=1e-6)   # equal weights -> n
    assert _ess(spiky) < 2.0                                  # one candidate carries it


def test_the_pair_table_ignores_padding_and_unnamed_planes():
    n = 4
    pc = SignPairCounts(n_tasks=3, device="cpu")
    pc.update(
        on_task_logprob=torch.zeros(n, 1, 2), off_task_logprobs=torch.zeros(n, 1, 2, 1),
        base_logprob=torch.zeros(n, 1, 2), student_logprob=torch.zeros(n, 1, 2),
        response_mask=torch.tensor([[1.0], [1.0], [0.0], [1.0]]),
        task_ids=torch.tensor([0, -1, 0, 0]),
        off_plane_tasks=torch.tensor([[1], [1], [1], [-1]]),
        deadzone=0.1,
    )
    count, _m, _s, totals = pc._cpu()
    # only row 0 has a valid (task, plane, mask) triple: 2 candidates
    assert float(count.sum()) == 2.0
    # the population total counts rows with a task and a mask, plane or no plane
    assert float(totals[:, 0].sum()) == 4.0


def test_a_low_count_cell_is_omitted_rather_than_reported_noisy():
    """A rate over a handful of candidates goes through reduce_metrics looking
    exactly like a rate over millions."""
    assert _pair_from_cells({(1, 1): 3}, deadzone=0.5) == {} or all(
        "pair/lor" not in k
        for k in SignPairCounts(n_tasks=2, device="cpu").metrics(task_names=["dst", "src"])
    )
    thin = SignPairCounts(n_tasks=2, device="cpu")
    thin.update(
        on_task_logprob=torch.ones(3, 1, 1), off_task_logprobs=torch.ones(3, 1, 1, 1),
        base_logprob=torch.zeros(3, 1, 1), student_logprob=torch.zeros(3, 1, 1),
        response_mask=torch.ones(3, 1), task_ids=torch.zeros(3, dtype=torch.long),
        off_plane_tasks=torch.ones(3, 1, dtype=torch.long), deadzone=0.5,
    )
    assert thin.metrics(task_names=["dst", "src"]) == {}  # default min_count is 1000


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def _update_policy_source() -> str:
    """The body of ``DataParallelPPOActor.update_policy``, read from the file.

    ``inspect.getsource`` hands back the ``GPUMemoryLogger`` wrapper instead, so
    the two wiring tests below would pass against a two-line closure no matter
    what the method actually does.
    """
    import ast
    import inspect
    import textwrap

    from verl.workers.actor import dp_actor

    text = open(inspect.getsourcefile(dp_actor), encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_policy":
            return textwrap.dedent(ast.get_source_segment(text, node))
    raise AssertionError("update_policy not found in dp_actor.py")


def test_the_measurement_gates_never_depend_on_batch_content():
    """These accumulators run COLLECTIVES. A rank whose batch happened to lack
    the cached planes must still reach the all_reduce, or it hangs every other
    rank. So construction and reduction gate on the CONFIG, and only the update
    gates on the data."""
    src = _update_policy_source()
    assert "sign_cfg_on = bool(sign_cfg and sign_cfg.get" in src
    assert 'sign_enabled = sign_cfg_on and "sign_cache_ids" in data.batch.keys()' in src
    for gate in ("transfer_on = target_mode and bool(", "pair_on = sign_cfg_on and bool("):
        assert gate in src, gate
    # the constructors must read the config-only flags, never sign_enabled
    for line in src.splitlines():
        if "ScopeTermStats(" in line or "OffTaskLadderStats(" in line or "SignPairCounts(" in line:
            assert "sign_enabled" not in line, line


def test_measure_only_suppresses_both_writes_and_nothing_else():
    """The observer arm. It must not be faked by setting the weights to 1.0:
    reweight_teacher_logprobs still subtracts a log z that differs from 0 by
    float error, and its tail clamp can bind, so over 150 steps that would be a
    different trajectory and would not be a control."""
    src = _update_policy_source()
    assert "if not sign_measure_only:" in src
    body = src.split("if not sign_measure_only:")
    assert len(body) == 3, "both the target write and the position write must be gated"
    assert "reweight_teacher_logprobs(" in body[1]
    assert "normalize_per_task(" in body[2]
    # the measurement itself is NOT gated on it
    head = src.split("if not sign_measure_only:")[0]
    assert "candidate_weights(" in head
