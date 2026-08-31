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


def _build(on, off, base, *, bs):
    task_ids = torch.arange(bs) % 3
    planes = torch.stack([(task_ids + 1) % 3, (task_ids + 2) % 3], dim=-1)
    return build_position_weight(
        shifts=compute_raw_policy_shifts(
            on_task_logprob=on, off_task_logprobs=off, base_logprob=base
        ),
        on_task_logprob=on, task_ids=task_ids, off_plane_tasks=planes,
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
    assert float(dec["common_soft"]) == pytest.approx(2.0), "and the agreement IS paid"


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
        "weight": got["channel_pre_weight"]["no_shared"],
        "pre_weight": got["channel_pre_weight"]["no_shared"],
        "evidence": got["channel_evidence"]["no_shared"],
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
    m = position_weight_metrics(stats.sums(task_names=TASKS), prefix="kl_weight/channel/no_shared")
    assert "kl_weight/channel/no_shared/position/w_mean" in m, "the scope did render"
    for key in ("exclusive_pass_rate", "gate_pass_rate", "gate_mean"):
        assert f"kl_weight/channel/no_shared/evidence/{key}" not in m


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
