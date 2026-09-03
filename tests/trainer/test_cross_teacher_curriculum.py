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
"""What the CURRICULUM mode claims by construction, checked as arithmetic.

Same rule as ``test_cross_teacher_target.py``: every assertion here is a line in
``docs/cross_teacher_curriculum_design.md`` that says "by construction", and it
is a test rather than a comment because the previous arms' equivalent claims
were comments and two of them were false.

Four of these carry the design's whole argument and would each invalidate it:
``test_the_layers_nest``, ``test_the_target_never_leaves_base_to_on_task``
(off-task injection is zero), ``test_full_release_is_the_control_bit_for_bit``
(the fixed point is unchanged) and ``test_the_schedule_is_a_pure_function_of_the_step``
(resume correctness).
"""
import math

import pytest
import torch

from verl.trainer.ppo.cross_teacher_target import (
    LAYER_BRANCHES,
    LAYERS,
    build_target,
    curriculum_exponent,
    curriculum_rho,
    nested_layers,
)

STAGES = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0))


def _chain(k=6, bs=2, resp=3, n_off=2, seed=11):
    """The same fixture the tilt tests use, so the two modes are compared on one input."""
    gen = torch.Generator().manual_seed(seed)
    on = torch.log_softmax(torch.randn(bs, resp, k, generator=gen), dim=-1) + math.log(0.99)
    base = torch.log_softmax(torch.randn(bs, resp, k, generator=gen), dim=-1) + math.log(0.99)
    off = torch.log_softmax(torch.randn(bs, resp, k, n_off, generator=gen), dim=-2) + math.log(0.99)
    return dict(
        on_logprob=on, off_logprob=off, base_logprob=base,
        diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
        task_ids=torch.zeros(bs, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2]] * bs),
    )


def _cur(rho_pair=0.0, rho_own=0.0, **over):
    kw = _chain()
    kw.update(over)
    return build_target(mode="curriculum", rho={"pair": rho_pair, "own": rho_own}, **kw), kw


def _layers(on, off, sigma=1.0):
    """nested_layers for plain lists: (k,) on-task nats and (k, n_off) off-task hats."""
    sh = torch.tensor(on, dtype=torch.float32).reshape(1, 1, -1)
    ho = torch.tensor(off, dtype=torch.float32).reshape(1, 1, sh.size(-1), -1)
    got = nested_layers(shift_on=sh, hat_off=ho,
                        sigma_on=torch.full((1, 1, 1), float(sigma)))
    return {n: got[n].reshape(-1) for n in LAYERS}, got["branch"].reshape(-1)


# ------------------------------------------------------------------ the layers
def test_the_layers_nest():
    """|shared| <= |pair| <= |own|, and every non-zero layer carries the on-task sign.

    The release order means nothing without this: a "coarser" layer that could
    exceed a finer one would make the curriculum teach MORE early, not less.
    """
    gen = torch.Generator().manual_seed(3)
    on = torch.randn(4, 7, 5, generator=gen) * 2.0
    off = torch.randn(4, 7, 5, 3, generator=gen) * 2.0
    got = nested_layers(shift_on=on, hat_off=off, sigma_on=torch.full((4, 1, 1), 1.3))
    assert torch.all(got["shared"].abs() <= got["pair"].abs() + 1e-6)
    assert torch.all(got["pair"].abs() <= got["own"].abs() + 1e-6)
    for name in ("shared", "pair"):
        nz = got[name] != 0
        assert torch.equal(got[name][nz].sign(), got["own"][nz].sign())


def test_the_three_layers_sum_to_the_on_task_shift():
    gen = torch.Generator().manual_seed(5)
    on = torch.randn(3, 6, 4, generator=gen)
    off = torch.randn(3, 6, 4, 2, generator=gen)
    got = nested_layers(shift_on=on, hat_off=off, sigma_on=torch.ones(3, 1, 1))
    total = got["shared"] + (got["pair"] - got["shared"]) + (got["own"] - got["pair"])
    assert torch.allclose(total, on, atol=1e-12)


def test_the_shared_layer_is_the_min_over_all_three_teachers():
    lay, _ = _layers([2.0], [[3.0, 5.0]])
    assert lay["shared"].item() == pytest.approx(2.0)   # capped by the on-task voice
    lay, _ = _layers([4.0], [[3.0, 5.0]])
    assert lay["shared"].item() == pytest.approx(3.0)   # capped by the quietest


def test_the_pair_layer_is_a_union_not_an_average():
    """One agreeing off-task teacher is enough, and it is not halved by the other."""
    lay, _ = _layers([4.0], [[3.0, -9.0]])
    assert lay["shared"].item() == 0.0                  # not all three agree
    assert lay["pair"].item() == pytest.approx(3.0)      # NOT 1.5, and not 0
    # the LOUDEST agreeing teacher sets it, capped by the on-task voice
    lay, _ = _layers([4.0], [[1.0, 3.0]])
    assert lay["pair"].item() == pytest.approx(3.0)
    assert lay["shared"].item() == pytest.approx(1.0)


def test_an_opposed_off_task_teacher_corroborates_nothing():
    lay, _ = _layers([2.0], [[-3.0, -5.0]])
    assert lay["shared"].item() == 0.0
    assert lay["pair"].item() == 0.0
    assert lay["own"].item() == pytest.approx(2.0)


def test_a_position_where_the_teacher_equals_base_is_untouched():
    """24.5% of positions on this mixture. There is no shift there to be backed."""
    lay, branch = _layers([0.0], [[3.0, 5.0]])
    assert all(lay[n].item() == 0.0 for n in LAYERS)
    assert branch.item() == LAYER_BRANCHES.index("none")


def test_the_layers_are_insensitive_to_duplicating_a_teacher():
    dup, _ = _layers([4.0], [[2.0, 2.0]])
    lone, _ = _layers([4.0], [[2.0, 9.0]])
    assert dup["shared"].item() == pytest.approx(2.0)
    assert dup["pair"].item() == pytest.approx(2.0)
    # a second, louder teacher raises the pair layer but not the shared one
    assert lone["shared"].item() == pytest.approx(2.0)
    assert lone["pair"].item() == pytest.approx(4.0)


def test_the_off_task_volume_is_read_in_the_on_task_teachers_nats():
    """sigma_on converts the standardised off-task shift; the cap is a raw shift."""
    small, _ = _layers([4.0], [[1.0, 1.0]], sigma=1.0)
    large, _ = _layers([4.0], [[1.0, 1.0]], sigma=3.0)
    assert small["shared"].item() == pytest.approx(1.0)
    assert large["shared"].item() == pytest.approx(3.0)


def test_the_layer_branch_names_how_many_teachers_back_the_direction():
    _, three = _layers([2.0], [[3.0, 5.0]])
    _, pair = _layers([2.0], [[3.0, -5.0]])
    _, own = _layers([2.0], [[-3.0, -5.0]])
    _, none = _layers([0.0], [[3.0, 5.0]])
    assert three.item() == LAYER_BRANCHES.index("three")
    assert pair.item() == LAYER_BRANCHES.index("pair")
    assert own.item() == LAYER_BRANCHES.index("own")
    assert none.item() == LAYER_BRANCHES.index("none")


# ---------------------------------------------------------------- the exponent
def test_the_exponent_subtracts_and_never_amplifies():
    """c is opposite in sign to the on-task shift and bounded by it in magnitude."""
    gen = torch.Generator().manual_seed(7)
    on = torch.randn(3, 5, 4, generator=gen) * 2.0
    off = torch.randn(3, 5, 4, 2, generator=gen) * 2.0
    lay = nested_layers(shift_on=on, hat_off=off, sigma_on=torch.ones(3, 1, 1))
    for rp, ro in STAGES:
        c = curriculum_exponent(layers=lay, rho_pair=rp, rho_own=ro)
        assert torch.all(c.abs() <= on.abs() + 1e-6)
        nz = c != 0
        assert torch.all(c[nz].sign() != on[nz].sign())


def test_full_release_makes_the_exponent_exactly_zero():
    """Not approximately: the loss must be handed the on-task teacher's own bits."""
    gen = torch.Generator().manual_seed(9)
    lay = nested_layers(shift_on=torch.randn(2, 4, 6, generator=gen),
                        hat_off=torch.randn(2, 4, 6, 2, generator=gen),
                        sigma_on=torch.ones(2, 1, 1))
    c = curriculum_exponent(layers=lay, rho_pair=1.0, rho_own=1.0)
    assert torch.all(c == 0.0)


def test_the_exponent_is_continuous_in_rho():
    gen = torch.Generator().manual_seed(11)
    lay = nested_layers(shift_on=torch.randn(2, 3, 4, generator=gen),
                        hat_off=torch.randn(2, 3, 4, 2, generator=gen),
                        sigma_on=torch.ones(2, 1, 1))
    prev = curriculum_exponent(layers=lay, rho_pair=0.0, rho_own=0.0)
    for i in range(1, 21):
        cur = curriculum_exponent(layers=lay, rho_pair=i / 20.0, rho_own=0.0)
        assert torch.all((cur - prev).abs() < 0.4)
        prev = cur


def test_the_exponent_is_continuous_across_an_off_task_sign_flip():
    """No deadzone: as an off-task teacher's shift crosses zero, its corroboration
    goes to zero with it, so the layer it gates does not jump."""
    for eps in (1e-4, 1e-6):
        pos, _ = _layers([2.0], [[eps, 5.0]])
        neg, _ = _layers([2.0], [[-eps, 5.0]])
        assert abs(pos["shared"].item() - neg["shared"].item()) < 1e-3
        assert abs(pos["pair"].item() - neg["pair"].item()) < 1e-3


def test_the_exponent_is_continuous_across_on_task_silence():
    for eps in (1e-4, 1e-6):
        pos, _ = _layers([eps], [[3.0, 5.0]])
        neg, _ = _layers([-eps], [[3.0, 5.0]])
        for name in LAYERS:
            assert abs(pos[name].item() - neg[name].item()) < 1e-3


# ------------------------------------------------------------------- the target
def test_the_target_never_leaves_base_to_on_task():
    """OFF-TASK INJECTION IS ZERO -- the property the fallback channel gave up.

    p_on e^c = p_0 e^a with a between 0 and the on-task shift, so every candidate
    lands in the closed interval between the two models' probabilities. Checked
    BEFORE the normaliser, which is where the claim lives: 1/Z then rescales the
    whole position, head included, and that is the accepted cost the tilt path
    already documents.
    """
    for rp, ro in STAGES:
        got, kw = _cur(rp, ro)
        p0 = kw["base_logprob"].exp()
        p_on = kw["on_logprob"].exp()
        tilted = p_on * got["c"].exp().float()
        lo, hi = torch.minimum(p0, p_on), torch.maximum(p0, p_on)
        assert torch.all(tilted >= lo - 1e-6)
        assert torch.all(tilted <= hi + 1e-6)


def test_the_unbacked_part_of_a_shift_is_returned_all_the_way_to_base():
    """With no off-task teacher agreeing anywhere, stage 1's target IS base.

    This is what the exponent clamp would break, which is why curriculum mode
    passes clamp=None: on a candidate the teacher suppressed by more than the
    clamp, a clamped stage 1 would stop short of base and aim at a distribution
    the design does not name.
    """
    kw = _chain()
    # every off-task teacher flat at base -> no corroboration anywhere
    kw["off_logprob"] = kw["base_logprob"].unsqueeze(-1).expand_as(kw["off_logprob"]).clone()
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0}, **kw)
    reached = (kw["on_logprob"] + got["c"].float())
    assert torch.allclose(reached, kw["base_logprob"], atol=1e-5)


def test_a_large_suppression_is_restored_without_a_clamp():
    """The clamp is 5.0 nats on the tilt path. Here a 12-nat gap must close fully."""
    k = 3
    base = torch.log_softmax(torch.zeros(1, 1, k), dim=-1)
    on = base.clone()
    on[..., 0] -= 12.0            # the teacher crushed this candidate
    on = on - on.exp().sum(-1, keepdim=True).log()
    kw = dict(on_logprob=on, base_logprob=base,
              off_logprob=base.unsqueeze(-1).expand(1, 1, k, 2).clone(),
              diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
              task_ids=torch.zeros(1, dtype=torch.long),
              off_plane_tasks=torch.tensor([[1, 2]]))
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0}, **kw)
    assert got["c"][..., 0].item() > 5.0                  # past the tilt clamp
    assert not bool(got["clamped"].any())
    assert (on[..., 0] + got["c"][..., 0].float()).item() == pytest.approx(
        base[..., 0].item(), abs=1e-4)


def test_full_release_is_the_control_bit_for_bit():
    got, kw = _cur(1.0, 1.0)
    assert torch.all(got["c"] == 0.0)
    assert not bool(got["live"].any())
    assert torch.equal(got["target_logprob"], kw["on_logprob"])


def test_a_restricted_stage_really_does_move_the_target():
    """Or every identity above is vacuous."""
    got, kw = _cur(0.0, 0.0)
    assert got["moved"].sum() > 0
    assert not torch.equal(got["target_logprob"], kw["on_logprob"])


def test_the_target_is_a_distribution_at_every_stage():
    for rp, ro in STAGES:
        got, kw = _cur(rp, ro)
        tail = 1.0 - kw["on_logprob"].exp().sum(dim=-1)
        support = got["target_logprob"].exp().sum(dim=-1)
        assert torch.allclose(support + tail / got["log_z"].exp().float(),
                              torch.ones_like(support), atol=1e-6)


def test_the_stages_are_ordered_by_how_much_they_hold_back():
    got_s, _ = _cur(0.0, 0.0)
    got_p, _ = _cur(1.0, 0.0)
    got_o, _ = _cur(1.0, 1.0)
    assert got_s["moved"].sum() > got_p["moved"].sum() > got_o["moved"].sum()


def test_cold_start_rows_are_a_no_op():
    got, kw = _cur(0.0, 0.0, diag_valid=torch.tensor([False, True, True]))
    assert torch.equal(got["target_logprob"], kw["on_logprob"])


def test_curriculum_mode_refuses_to_guess_the_schedule():
    kw = _chain()
    with pytest.raises(AssertionError, match="curriculum_rho"):
        build_target(mode="curriculum", **kw)


def test_per_task_sigma_is_read_independently():
    a, _ = _cur(0.0, 0.0)
    b, _ = _cur(0.0, 0.0, diag=torch.tensor([1.0, 10.0, 1.0]))
    assert not torch.equal(a["c"], b["c"])


# ----------------------------------------------------------------- the schedule
def test_the_schedule_is_a_pure_function_of_the_step():
    """Which is what makes resume correct: global_steps comes back from the
    checkpoint folder, so a run restarted at 150 does not re-run stage 1."""
    sched = dict(stage_steps=(40, 80), ramp_steps=10)
    for step in (1, 40, 45, 50, 80, 85, 90, 150, 300):
        assert curriculum_rho(step=step, **sched) == curriculum_rho(step=step, **sched)
    assert curriculum_rho(step=150, **sched) == {"pair": 1.0, "own": 1.0}


def test_the_schedule_boundaries_are_the_preregistered_ones():
    sched = dict(stage_steps=(40, 80), ramp_steps=10)
    r = lambda s: curriculum_rho(step=s, **sched)
    assert r(1) == {"pair": 0.0, "own": 0.0}          # stage 1
    assert r(40) == {"pair": 0.0, "own": 0.0}         # last step of stage 1
    assert r(45) == {"pair": 0.5, "own": 0.0}         # mid ramp
    assert r(50) == {"pair": 1.0, "own": 0.0}         # stage 2
    assert r(80) == {"pair": 1.0, "own": 0.0}         # last step of stage 2
    assert r(85) == {"pair": 1.0, "own": 0.5}
    assert r(90) == {"pair": 1.0, "own": 1.0}         # the control, from here on
    assert r(300) == {"pair": 1.0, "own": 1.0}


def test_the_schedule_is_monotone_and_bounded():
    prev = {"pair": -1.0, "own": -1.0}
    for step in range(1, 121):
        cur = curriculum_rho(step=step, stage_steps=(40, 80), ramp_steps=10)
        assert 0.0 <= cur["pair"] <= 1.0 and 0.0 <= cur["own"] <= 1.0
        assert cur["pair"] >= prev["pair"] and cur["own"] >= prev["own"]
        # the own layer is never released before the pair layer is fully out
        assert cur["own"] == 0.0 or cur["pair"] == 1.0
        prev = cur


def test_a_hard_switch_is_ramp_one():
    r = lambda s: curriculum_rho(step=s, stage_steps=(40, 80), ramp_steps=1)
    assert r(40) == {"pair": 0.0, "own": 0.0}
    assert r(41) == {"pair": 1.0, "own": 0.0}


def test_overlapping_ramps_are_refused():
    with pytest.raises(AssertionError, match="stage 2 would start"):
        curriculum_rho(step=1, stage_steps=(40, 45), ramp_steps=10)


# -------------------------------------------------------------------- the table
def test_step_stats_render_the_curriculum_quantities_and_not_the_tilt_ones():
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    kw = _chain(bs=3, resp=4, k=8)
    bs, resp = kw["on_logprob"].shape[:2]
    mask = torch.ones(bs, resp)
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0},
                       curriculum_counterfactuals=True, response_mask=mask, **kw)
    stats = TargetStepStats(n_tasks=3, device="cpu", mode="curriculum")
    stats.update(
        built=got, p_on=got["p_on"],
        support_ids=torch.randint(0, 50, kw["on_logprob"].shape),
        response_mask=mask, task_ids=kw["task_ids"],
        d_on=torch.rand(bs, resp), d_base=torch.rand(bs, resp) + 1.0,
        student_logprob=torch.log_softmax(torch.randn(*kw["on_logprob"].shape), dim=-1)
        + math.log(0.99),
        on_logprob=kw["on_logprob"],
        roles=torch.randint(0, 3, (bs, resp)),
    )
    m = stats.metrics(task_names=["alfworld", "search", "webshop"])
    for key in ("target/tv", "target/layer/shared/backed_frac",
                "target/layer/pair/backed_frac", "target/layer/own/backed_frac",
                "target/layer/shared/role/structural_share",
                "target/layer/pair/role/content_share",
                "target/retained_shuffled_ratio",
                "target/stage_kl/shared", "target/stage_kl/pair", "target/stage_kl/own",
                "target/stage_tv/shared", "target/stage_abs_dkl/pair",
                "target/layer_branch/three/cand_frac",
                "target/layer_branch/own/mass_frac",
                "target/kl_to_base", "target/withheld_smass_mean",
                "target/tail/cand_frac", "target/tail/student_mass_frac",
                "target/branch/agree/mass_frac", "target/alfworld/tv"):
        assert key in m, key
    # The tilt path's columns must be ABSENT, not zero: a structurally-zero
    # column reads as a measurement.
    for key in ("target/channel/a_share", "target/channel/b_share",
                "target/shuffled_tv_ratio", "target/clamped_per_step",
                "target/channel/a_only_tv"):
        assert key not in m, key
    assert m["target/mass_error_max"] < 1e-12
    # backed_frac nests like the layers do, and own is the whole shift.
    assert 0.0 <= m["target/layer/shared/backed_frac"] <= m["target/layer/pair/backed_frac"] <= 1.0
    assert m["target/layer/own/backed_frac"] == pytest.approx(1.0, abs=1e-12)
    fracs = sum(m[f"target/layer_branch/{b}/cand_frac"] for b in LAYER_BRANCHES)
    assert fracs == pytest.approx(1.0, abs=1e-9)


def test_stage_kl_is_what_the_loss_would_have_been_at_that_stage():
    """Checked against the loss's OWN reverse-KL function, not against the
    formula the metric is computed with.

    The point of the column is that "what would stage 1 cost right now" is
    answerable at every step of the run, including after full release when there
    is nothing live to measure. So it must be the real per-position KL against
    that stage's target, on the same denominator as the plain on-task KL --
    which is why d_on enters the table a second time, unrestricted by `live`.
    """
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    kw = _chain(bs=2, resp=3, k=6)
    bs, resp = kw["on_logprob"].shape[:2]
    mask = torch.ones(bs, resp)
    student = torch.log_softmax(torch.randn(*kw["on_logprob"].shape), dim=-1) + math.log(0.99)
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0},
                       curriculum_counterfactuals=True, response_mask=mask, **kw)
    d_on = topk_kl_per_token(student_topk_logprob=student,
                             teacher_topk_logprob=kw["on_logprob"])
    stats = TargetStepStats(n_tasks=1, device="cpu", mode="curriculum")
    stats.update(built=got, p_on=got["p_on"],
                 support_ids=torch.zeros(bs, resp, 6, dtype=torch.long),
                 response_mask=mask, task_ids=None, d_on=d_on, d_base=d_on + 1.0,
                 student_logprob=student, on_logprob=kw["on_logprob"])
    m = stats.metrics(task_names=["t"])
    assert m["target/stage_kl/own"] == pytest.approx(d_on.mean().item(), rel=1e-6)
    for name in ("shared", "pair"):
        target = kw["on_logprob"] + got[f"stage_log_w_{name}"].to(kw["on_logprob"].dtype)
        want = topk_kl_per_token(student_topk_logprob=student,
                                 teacher_topk_logprob=target).mean().item()
        assert m[f"target/stage_kl/{name}"] == pytest.approx(want, rel=1e-4), name
    # No ordering is asserted between the stages. Holding a component back moves
    # the target toward base, and whether that is nearer to or further from the
    # student is a fact about the student, not about the construction -- the
    # design's prediction that stage 1's teacher KL comes out BELOW the
    # control's is about a student that starts at base, and is a run result.


def test_the_restricted_stages_are_still_measured_after_full_release():
    """The whole reason the two stage columns are computed every step."""
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    kw = _chain(bs=2, resp=3, k=6)
    bs, resp = kw["on_logprob"].shape[:2]
    mask = torch.ones(bs, resp)
    got = build_target(mode="curriculum", rho={"pair": 1.0, "own": 1.0},
                       curriculum_counterfactuals=True, response_mask=mask, **kw)
    assert torch.all(got["c"] == 0.0) and not bool(got["live"].any())
    stats = TargetStepStats(n_tasks=1, device="cpu", mode="curriculum")
    stats.update(built=got, p_on=got["p_on"],
                 support_ids=torch.zeros(bs, resp, 6, dtype=torch.long),
                 response_mask=mask, task_ids=None,
                 d_on=torch.rand(bs, resp) + 0.5, d_base=torch.rand(bs, resp) + 1.5,
                 student_logprob=torch.log_softmax(
                     torch.randn(*kw["on_logprob"].shape), dim=-1) + math.log(0.99),
                 on_logprob=kw["on_logprob"])
    m = stats.metrics(task_names=["t"])
    assert m["target/tv"] == 0.0 and m["target/live_frac"] == 0.0
    assert m["target/stage_tv/shared"] > 0.0
    assert m["target/stage_kl/shared"] != m["target/stage_kl/own"]


def test_the_retained_shuffled_ratio_is_the_shared_layers_own_ratio():
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    kw = _chain(bs=4, resp=6, k=8)
    bs, resp = kw["on_logprob"].shape[:2]
    mask = torch.ones(bs, resp)
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0},
                       curriculum_counterfactuals=True, response_mask=mask, **kw)
    assert "shared_shuffled" in got
    stats = TargetStepStats(n_tasks=1, device="cpu", mode="curriculum")
    stats.update(built=got, p_on=got["p_on"],
                 support_ids=torch.zeros(bs, resp, 8, dtype=torch.long),
                 response_mask=mask, task_ids=None)
    m = stats.metrics(task_names=["t"])
    p = got["p_on"].to(torch.float64)
    want = ((got["shared_shuffled"].to(torch.float64).abs() * p).sum()
            / (got["layer_shared"].to(torch.float64).abs() * p).sum()).item()
    assert m["target/retained_shuffled_ratio"] == pytest.approx(want, rel=1e-9)


# ------------------------------------------------- the metrics added for analysis
def _stats_for(got, kw, *, student=None, d_on=None, d_base=None, off_plane_tasks=None,
               task_names=("alfworld", "search", "webshop")):
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    bs, resp, k = kw["on_logprob"].shape
    stats = TargetStepStats(n_tasks=len(task_names), device="cpu", mode="curriculum")
    stats.update(built=got, p_on=got["p_on"],
                 support_ids=torch.zeros(bs, resp, k, dtype=torch.long),
                 response_mask=torch.ones(bs, resp), task_ids=kw["task_ids"],
                 d_on=d_on, d_base=d_base, student_logprob=student,
                 on_logprob=kw["on_logprob"], off_plane_tasks=off_plane_tasks)
    return stats.metrics(task_names=list(task_names))


def test_kl_to_base_is_the_mean_over_all_positions_not_the_live_ones():
    """Unlike acted_novelty's d_base, which is live-restricted: at full release
    nothing is live, and that is exactly when "where does the student sit between
    base and the teacher" is still worth reading."""
    got, kw = _cur(1.0, 1.0)
    bs, resp = kw["on_logprob"].shape[:2]
    d_base = torch.rand(bs, resp) + 0.2
    m = _stats_for(got, kw, d_base=d_base, d_on=d_base * 0.5)
    assert not bool(got["live"].any())
    assert m["target/kl_to_base"] == pytest.approx(d_base.mean().item(), rel=1e-6)


def test_the_pair_source_names_the_teacher_that_set_the_pair_layer():
    """Only the second off-task plane (task 2) agrees with the on-task teacher, so
    the whole pair layer is attributed to it -- by task NAME, per destination."""
    k, bs = 4, 1
    base = torch.log_softmax(torch.zeros(bs, 1, k), dim=-1)
    on = base.clone(); on[..., 0] += 2.0
    on = on - on.exp().sum(-1, keepdim=True).log()
    off_agree = base.clone(); off_agree[..., 0] += 1.5
    off_agree = off_agree - off_agree.exp().sum(-1, keepdim=True).log()
    off_oppose = base.clone(); off_oppose[..., 0] -= 1.5
    off_oppose = off_oppose - off_oppose.exp().sum(-1, keepdim=True).log()
    off = torch.stack([off_oppose, off_agree], dim=-1)          # plane 0: task 1, plane 1: task 2
    kw = dict(on_logprob=on, off_logprob=off, base_logprob=base,
              diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
              task_ids=torch.zeros(bs, dtype=torch.long),
              off_plane_tasks=torch.tensor([[1, 2]]))
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0}, **kw)
    assert got["layer_pair_source"][..., 0].item() == 1                 # the second plane
    m = _stats_for(got, kw, off_plane_tasks=kw["off_plane_tasks"])
    assert m["target/alfworld/pair_source/webshop/share"] == pytest.approx(1.0)
    assert "target/alfworld/pair_source/search/share" in m
    assert m["target/alfworld/pair_source/search/share"] == pytest.approx(0.0)
    # The pair layer at candidate 0 is the agreeing teacher's SHIFT (not its
    # logit: log-softmax renormalises), capped by the on-task shift.
    want = min((on - base)[..., 0].item(), (off_agree - base)[..., 0].item())
    assert got["layer_pair"][..., 0].item() == pytest.approx(want, abs=1e-5)
    assert got["layer_shared"][..., 0].item() == 0.0
    # Candidates 1-3 were lowered by both the on-task teacher and the agreeing
    # one (renormalisation), so the pair layer is non-zero there too and comes
    # from the same plane; the opposing plane raised them and sets nothing.
    assert torch.all(got["layer_pair_source"][..., 1:] == 1)


def test_the_tail_region_is_measured_and_stage_one_keeps_only_what_all_three_suppress():
    """Design 3.2-5: the student is confident, the on-task teacher gives ~0.
    Case A: all three teachers suppressed the candidate -> it is in the shared
    layer, stage 1 distils it, three_frac = 1. Case B: only the on-task teacher
    did -> stage 1 returns it to base, three_frac = 0, and the withheld amount
    the student feels sits entirely in the tail."""
    k = 3
    base = torch.log_softmax(torch.tensor([[[0.0, 0.0, 0.0]]]), dim=-1)          # 1/3 each
    on = torch.log_softmax(torch.tensor([[[-12.0, 0.0, 0.0]]]), dim=-1)          # crushed cand 0
    student = torch.log_softmax(torch.tensor([[[3.0, 0.0, 0.0]]]), dim=-1)       # confident on 0
    assert student[..., 0].exp().item() > 0.5 and on[..., 0].exp().item() < 1e-3
    common = dict(base_logprob=base, diag=torch.ones(3),
                  diag_valid=torch.ones(3, dtype=torch.bool),
                  task_ids=torch.zeros(1, dtype=torch.long),
                  off_plane_tasks=torch.tensor([[1, 2]]))
    # A: the off-task teachers crushed it too
    off_a = on.unsqueeze(-1).expand(1, 1, k, 2).clone()
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0},
                       on_logprob=on, off_logprob=off_a, **common)
    m = _stats_for(got, dict(on_logprob=on, task_ids=common["task_ids"]), student=student)
    assert m["target/tail/cand_frac"] == pytest.approx(1.0 / k)
    assert m["target/tail/three_frac"] == pytest.approx(1.0)
    # B: only the on-task teacher did
    off_b = base.unsqueeze(-1).expand(1, 1, k, 2).clone()
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0},
                       on_logprob=on, off_logprob=off_b, **common)
    m = _stats_for(got, dict(on_logprob=on, task_ids=common["task_ids"]), student=student)
    assert m["target/tail/three_frac"] == pytest.approx(0.0)
    # Not exactly 1: crushing candidate 0 renormalised candidates 1 and 2 UP by
    # log(1/2) - log(1/3), nobody backs that either, and the student's 5% there
    # feels a little of it. The tail still carries essentially all of it.
    assert m["target/tail/withheld_share"] > 0.99
    assert m["target/tail/student_mass_frac"] > 0.5


def test_gradient_geometry_at_full_release_is_the_control_exactly():
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.cross_teacher_target import (
        CURRICULUM_GRAD_TERMS, curriculum_gradient_metrics, curriculum_gradient_terms,
    )

    got, kw = _cur(1.0, 1.0)
    bs, resp, k = kw["on_logprob"].shape
    student = torch.log_softmax(torch.randn(bs, resp, k), dim=-1) + math.log(0.97)
    on_kl = topk_kl_per_token(student_topk_logprob=student, teacher_topk_logprob=kw["on_logprob"])
    cols = curriculum_gradient_terms(
        student_logprob=student, target_logprob=got["target_logprob"],
        on_logprob=kw["on_logprob"], live_kl=on_kl, on_kl=on_kl,
        pg_grad_coef=torch.randn(bs, resp), sampled_onehot=torch.zeros(bs, resp, k),
        coef=0.01, pg_coef=1.0,
    )
    assert set(cols) == set(CURRICULUM_GRAD_TERMS)
    assert torch.allclose(cols["g_live_sq"], cols["g_ctl_sq"])
    assert torch.allclose(cols["g_live_ctl"], cols["g_ctl_sq"])
    assert torch.allclose(cols["g_live_grpo"], cols["g_ctl_grpo"])
    sums = {None: {n: float(v.sum()) for n, v in cols.items()}}
    sums[None]["n"] = float(bs * resp)
    m = curriculum_gradient_metrics(sums, prefix="target")
    assert m["target/control/grad_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert m["target/control/grad_norm_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["target/grpo/grad_cosine"] == pytest.approx(m["target/control/grpo_grad_cosine"], abs=1e-6)


def test_stage_one_pulls_against_the_control_where_the_reward_moved_the_student_past_base():
    """The reference-KL-to-base reading of design 3.2-4, as a sign.

    The on-task teacher raised candidate 0 and nobody backs it, so stage 1's
    target is base there. A student that has already moved toward the teacher
    on candidate 0 is pulled BACK by stage 1 and FORWARD by the control: the two
    OPD directions point opposite ways at that logit and their cosine is negative.
    """
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.cross_teacher_target import curriculum_gradient_terms

    k = 3
    base = torch.log_softmax(torch.tensor([[[0.0, 0.0, 0.0]]]), dim=-1)
    on = torch.log_softmax(torch.tensor([[[3.0, 0.0, 0.0]]]), dim=-1)          # teacher raised 0
    student = torch.log_softmax(torch.tensor([[[1.5, 0.0, 0.0]]]), dim=-1)     # halfway there
    off = base.unsqueeze(-1).expand(1, 1, k, 2).clone()                         # nobody backs it
    kw = dict(on_logprob=on, off_logprob=off, base_logprob=base, diag=torch.ones(3),
              diag_valid=torch.ones(3, dtype=torch.bool),
              task_ids=torch.zeros(1, dtype=torch.long), off_plane_tasks=torch.tensor([[1, 2]]))
    got = build_target(mode="curriculum", rho={"pair": 0.0, "own": 0.0}, **kw)
    assert torch.allclose(got["target_logprob"], base, atol=1e-5)               # stage 1 = base
    live_kl = topk_kl_per_token(student_topk_logprob=student, teacher_topk_logprob=got["target_logprob"])
    on_kl = topk_kl_per_token(student_topk_logprob=student, teacher_topk_logprob=on)
    cols = curriculum_gradient_terms(
        student_logprob=student, target_logprob=got["target_logprob"], on_logprob=on,
        live_kl=live_kl, on_kl=on_kl, pg_grad_coef=torch.zeros(1, 1),
        sampled_onehot=torch.zeros(1, 1, k), coef=0.01,
    )
    assert cols["g_live_ctl"].item() < 0.0
    assert cols["g_live_sq"].item() > 0.0 and cols["g_ctl_sq"].item() > 0.0
    assert cols["g_grpo_sq"].item() == 0.0                                        # no reward term here


def test_the_role_cut_suffixes_are_keys_the_renderer_produces():
    from verl.trainer.ppo.cross_teacher_target import (
        CURRICULUM_GRAD_ROLE_CUT_SUFFIXES, CURRICULUM_GRAD_TERMS, curriculum_gradient_metrics,
    )

    sums = {"env_action": {n: 1.0 for n in CURRICULUM_GRAD_TERMS}}
    sums["env_action"]["n"] = 10.0
    m = curriculum_gradient_metrics(sums, prefix="target/role")
    for suf in CURRICULUM_GRAD_ROLE_CUT_SUFFIXES:
        assert f"target/role/env_action{suf}" in m, suf


def test_the_token_table_can_be_filed_by_layer():
    """TokenStateCounts with the curriculum's state vocabulary: every rendered
    key and every dumped row carries the LAYER names, none of the sign arm's."""
    from verl.trainer.ppo.sign_weights import STATE_NAMES, TokenStateCounts

    names = {i: n for i, n in enumerate(LAYER_BRANCHES)}
    acted = tuple(i for i, n in enumerate(LAYER_BRANCHES) if n != "none")
    tab = TokenStateCounts(vocab_size=50, n_tasks=1, device="cpu", top_n=8, mode="target",
                           state_names=names, acted_states=acted)
    bs, resp, k = 2, 3, 5
    ids = torch.randint(0, 50, (bs, resp, k))
    state = torch.randint(0, len(LAYER_BRANCHES), (bs, resp, k))
    w = torch.rand(bs, resp, k) + 0.5
    lp = torch.log_softmax(torch.randn(bs, resp, k), dim=-1)
    tab.update(support_ids=ids, state=state, weight=w, on_task_logprob=lp,
               response_mask=torch.ones(bs, resp), task_ids=torch.zeros(bs, dtype=torch.long),
               effect=(w - 1.0) * lp.exp())
    m = tab.scalar_metrics(task_names=["t"], prefix="target")
    assert any(key.endswith("/three") for key in m), sorted(m)[:5]
    assert not any(any(key.endswith("/" + sn) for sn in STATE_NAMES.values()) for key in m)
    rows = tab.top_tokens(task_names=["t"])
    states = {r["state"] for r in rows}
    assert states <= set(LAYER_BRANCHES) | {"__any__"}, states
    # and the default is unchanged
    dflt = TokenStateCounts(vocab_size=50, n_tasks=1, device="cpu", top_n=8, mode="target")
    assert dflt.state_names == STATE_NAMES
