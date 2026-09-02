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
"""What ``cross_teacher_target`` claims by construction, checked as arithmetic.

Every assertion here corresponds to a line in ``docs/cross_teacher_target_design.md``
that says "by construction". They are tests and not comments because the previous
arm's equivalent claims were comments, and two of them were false: its
renormalisation was not the identity at unit weight, and its deadzone silenced
two thirds of the teacher's probability mass rather than the noise it was for.
"""
import math

import pytest
import torch

from verl.trainer.ppo.cross_teacher_target import (
    BRANCHES,
    build_target,
    normalized_weight,
    off_task_consensus,
    target_exponent,
)


def _c(hat_on, hat_off, scale=1.0):
    """c for plain lists, so the boundary cases below read as arithmetic."""
    on = torch.tensor(hat_on, dtype=torch.float32).reshape(1, 1, -1)
    off = torch.tensor(hat_off, dtype=torch.float32).reshape(1, 1, on.size(-1), -1)
    return target_exponent(hat_on=on, consensus=off_task_consensus(off),
                           exponent_scale=scale)["c"].reshape(-1)


# --------------------------------------------------------------- the consensus
def test_consensus_sign_requires_all_off_task_to_share_it():
    off = torch.tensor([[[[1.0, 2.0], [-1.0, -2.0], [1.0, -2.0], [0.0, 2.0]]]])
    got = off_task_consensus(off)
    assert torch.equal(got["s"].reshape(-1), torch.tensor([1.0, -1.0, 0.0, 0.0]))


def test_consensus_volume_is_the_geometric_mean_not_the_minimum():
    off = torch.tensor([[[[1.0, 4.0]]]])
    assert off_task_consensus(off)["L"].item() == pytest.approx(2.0)  # not 1.0


def test_consensus_volume_is_exactly_zero_when_any_shift_is_exactly_zero():
    # Load-bearing: at 65% of head candidates one of the shifts is numerically
    # zero, and a clamped log would report those as a small positive volume.
    off = torch.tensor([[[[0.0, 5.0]]]])
    assert off_task_consensus(off)["L"].item() == 0.0


def test_consensus_volume_is_insensitive_to_duplicating_a_teacher():
    dup = off_task_consensus(torch.tensor([[[[1.3, 1.3]]]]))["L"].item()
    lone = off_task_consensus(torch.tensor([[[[1.3, 4.0]]]]))["L"].item()
    assert dup == pytest.approx(1.3)
    assert lone == pytest.approx(math.sqrt(1.3 * 4.0))


# ------------------------------------------------------------- the two channels
def test_channels_telescope_to_one_expression():
    torch.manual_seed(0)
    on = torch.randn(4, 7, 20)
    off = torch.randn(4, 7, 20, 2)
    got = target_exponent(hat_on=on, consensus=off_task_consensus(off))
    assert torch.allclose(got["a"] + got["b"], got["c"], atol=1e-6)


def test_agreeing_branch_is_the_full_off_task_voice():
    # O < L and O > L both give s*L when the on-task teacher signs along.
    assert _c([0.5], [[2.0, 2.0]])[0].item() == pytest.approx(2.0)
    assert _c([9.0], [[2.0, 2.0]])[0].item() == pytest.approx(2.0)


def test_partition_is_exact_so_no_evidence_is_counted_twice():
    on = torch.tensor([[[1.0]]])
    off = torch.tensor([[[[3.0, 3.0]]]])
    got = target_exponent(hat_on=on, consensus=off_task_consensus(off))
    # a takes the part inside the on-task envelope, b the excess.
    assert got["a"].item() == pytest.approx(1.0)
    assert got["b"].item() == pytest.approx(2.0)


def test_a_confident_opposed_on_task_teacher_is_not_overridden():
    # O >= L on the opposed side closes the relu exactly.
    assert _c([-2.0], [[2.0, 2.0]])[0].item() == 0.0
    assert _c([-5.0], [[2.0, 2.0]])[0].item() == 0.0


def test_split_off_task_teachers_contribute_nothing():
    assert _c([1.0], [[3.0, -3.0]])[0].item() == 0.0


# ------------------------------------------------------- continuity, i.e. no eps
def test_c_is_continuous_across_on_task_silence():
    """The boundary the predecessor needed report_epsilon to straddle."""
    off = [[2.0, 2.0]]
    for eps in (1e-1, 1e-3, 1e-6):
        hi = _c([+eps], off)[0].item()
        lo = _c([-eps], off)[0].item()
        assert abs(hi - lo) < 2 * eps + 1e-6
    assert _c([+1e-9], off)[0].item() == pytest.approx(_c([-1e-9], off)[0].item(), abs=1e-6)


def test_c_is_continuous_across_the_off_task_sign_flip():
    """L is a geometric mean, so the jump as s flips is itself of size zero."""
    for eps in (1e-2, 1e-4, 1e-6):
        near = _c([1.0], [[eps, 2.0]])[0].item()
        past = _c([1.0], [[-eps, 2.0]])[0].item()
        assert past == 0.0
        assert near < 3 * math.sqrt(eps * 2.0)


def test_c_is_continuous_where_b_closes():
    off = [[2.0, 2.0]]
    for eps in (1e-2, 1e-4):
        assert _c([-2.0 + eps], off)[0].item() == pytest.approx(eps, abs=1e-5)
        assert _c([-2.0 - eps], off)[0].item() == 0.0


def test_branch_labels_name_the_case_the_value_came_from():
    on = torch.tensor([[[1.0, -1.0, 0.0, 1.0]]])
    off = torch.tensor([[[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, -2.0]]]])
    branch = target_exponent(hat_on=on, consensus=off_task_consensus(off))["branch"]
    assert [BRANCHES[i] for i in branch.reshape(-1).tolist()] == [
        "agree", "conflict", "on_silent", "split",
    ]


# ------------------------------------------------- the tilt-and-renormalise step
def _random_position(gen, k=20):
    p = torch.distributions.Dirichlet(torch.full((k,), 0.3)).sample()
    tail = torch.rand(1, generator=gen).item() * 0.02
    return (p * (1 - tail)).reshape(1, 1, k), tail


def test_the_target_is_a_distribution():
    """``sum_S w p + p_tail / Z == 1``. This is the invariant the mechanism owns
    since the revision; the exchange it replaced owned a stronger one (``Z == 1``,
    i.e. the support conserved its own mass and the tail never moved) and that
    one is deliberately gone."""
    gen = torch.Generator().manual_seed(3)
    worst = 0.0
    for _ in range(400):
        p, _ = _random_position(gen)
        p64 = p.to(torch.float64)
        c = torch.randn(1, 1, p.size(-1), generator=gen) * 1.5
        got = normalized_weight(c=c, p_on=p)
        total = (got["w"] * p64).sum() + got["tail"] * got["inv_z"]
        worst = max(worst, abs((total - 1.0).item()))
    assert worst < 1e-14


def test_the_head_is_taxed_and_that_is_the_accepted_cost():
    """``c = 0`` no longer implies ``w = 1``; it implies ``w = 1/Z``. Asserted
    rather than lamented: the capacity exchange bought head-invariance at a
    throttle of ``T/A_U = 0.026`` and the revision spent it back."""
    gen = torch.Generator().manual_seed(4)
    taxed = 0
    for _ in range(200):
        p, _ = _random_position(gen)
        c = torch.randn(1, 1, p.size(-1), generator=gen) * 1.5
        c[..., ::3] = 0.0
        got = normalized_weight(c=c, p_on=p)
        untilted = got["w"][..., ::3]
        # every untilted candidate carries exactly the same factor, 1/Z
        assert torch.allclose(untilted, got["inv_z"].unsqueeze(-1).expand_as(untilted))
        taxed += int(not torch.allclose(untilted, torch.ones_like(untilted)))
    assert taxed > 190, taxed


def test_sign_faithfulness_is_relative_to_log_z():
    """The exchange guaranteed ``w > 1`` iff ``c > 0``. Normalisation weakens that
    to ``c > log Z``, and the test says so instead of asserting the old rule --
    a candidate the teachers agreed to raise CAN lose probability."""
    gen = torch.Generator().manual_seed(5)
    for _ in range(400):
        p, _ = _random_position(gen)
        c = torch.randn(1, 1, p.size(-1), generator=gen) * 1.5
        got = normalized_weight(c=c, p_on=p)
        log_z = got["log_z"].unsqueeze(-1)
        c64 = c.to(torch.float64)
        assert torch.all(got["w"][c64 > log_z] > 1.0 - 1e-12)
        assert torch.all(got["w"][c64 < log_z] < 1.0 + 1e-12)


def test_the_exponent_clamp_bounds_the_whole_intervention():
    """With the capacity exchange gone the clamp is the only cap, so it has to
    hold: ``|log w| <= 2 * clamp`` on the support and ``|log Z| <= clamp``."""
    from verl.trainer.ppo.cross_teacher_target import _EXPONENT_CLAMP

    gen = torch.Generator().manual_seed(6)
    for _ in range(400):
        p, _ = _random_position(gen)
        c = torch.randn(1, 1, p.size(-1), generator=gen) * 40.0     # far past it
        got = normalized_weight(c=c, p_on=p)
        assert torch.all(got["log_w"].abs() <= 2 * _EXPONENT_CLAMP + 1e-9)
        assert torch.all(got["log_z"].abs() <= _EXPONENT_CLAMP + 1e-9)
        assert torch.all(got["clamped"] == (c.to(torch.float64).abs() > _EXPONENT_CLAMP))


def test_an_untilted_position_is_a_bit_identical_no_op():
    p = torch.full((1, 1, 4), 0.2)
    got = normalized_weight(c=torch.zeros(1, 1, 4), p_on=p)
    assert torch.all(got["w"] == 1.0) and torch.all(got["log_w"] == 0.0)
    assert got["log_z"].item() == 0.0 and got["inv_z"].item() == 1.0
    assert got["moved"].item() == 0.0


def test_unavailable_rows_are_a_no_op():
    p = torch.full((2, 1, 4), 0.25)
    c = torch.tensor([[[1.0, -1.0, 2.0, -2.0]]]).expand(2, 1, 4).contiguous()
    got = normalized_weight(c=c, p_on=p,
                            row_available=torch.tensor([True, False]))
    assert torch.all(got["w"][1] == 1.0)
    assert not torch.all(got["w"][0] == 1.0)


def test_moved_is_the_total_variation_including_the_tail():
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):
        p, _ = _random_position(gen)
        p64 = p.to(torch.float64)
        c = torch.randn(1, 1, p.size(-1), generator=gen) * 1.5
        got = normalized_weight(c=c, p_on=p)
        tv = 0.5 * (
            (got["w"] * p64 - p64).abs().sum()
            + (got["tail"] * got["inv_z"] - got["tail"]).abs()
        )
        assert tv.item() == pytest.approx(got["moved"].item(), abs=1e-12)


def test_the_up_side_is_no_longer_throttled_by_the_down_side():
    """The measurement that forced the revision, as a regression test. The
    capacity exchange scaled the up side by ``T / A_U``; with a thin down side
    that was 0.026 in production. Normalisation gives the up candidate its full
    ``e^c`` up to the shared divisor."""
    p = torch.tensor([[[0.9, 0.05, 0.05]]])
    c = torch.tensor([[[0.0, 2.0, -0.01]]])       # nothing to supply, plenty to ask
    got = normalized_weight(c=c, p_on=p)
    ratio = (got["w"][..., 1] / got["inv_z"]).item()
    assert ratio == pytest.approx(math.exp(2.0), rel=1e-9)


def test_dkl_is_log_z_minus_the_students_average_tilt():
    """``dKL = log Z - <c>_{p_s}`` is what TargetStepStats reports and what the
    scale decision reads, so the identity behind it is checked here."""
    gen = torch.Generator().manual_seed(8)
    p, _ = _random_position(gen)
    c = torch.randn(1, 1, p.size(-1), generator=gen) * 1.2
    got = normalized_weight(c=c, p_on=p)
    ps = torch.distributions.Dirichlet(torch.full((p.size(-1),), 0.5)).sample()
    ps = (ps * 0.97).reshape(1, 1, -1).to(torch.float64)
    tail_s = 1.0 - ps.sum(dim=-1)
    from_logs = -(ps * got["log_w"]).sum(dim=-1) + tail_s * got["log_z"]
    from_c = got["log_z"] - (ps * c.to(torch.float64)).sum(dim=-1)
    assert torch.allclose(from_logs, from_c, atol=1e-12)


# ------------------------------------------------------------------ end to end
def _chain(k=6, bs=2, resp=3, n_off=2, seed=11):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs, resp, k, generator=gen)
    on = torch.log_softmax(logits, dim=-1) + math.log(0.99)
    base = torch.log_softmax(torch.randn(bs, resp, k, generator=gen), dim=-1) + math.log(0.99)
    off = torch.log_softmax(torch.randn(bs, resp, k, n_off, generator=gen), dim=-2) + math.log(0.99)
    return dict(
        on_logprob=on, off_logprob=off, base_logprob=base,
        diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
        task_ids=torch.zeros(bs, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2]] * bs),
    )


def test_end_to_end_the_target_sums_to_one_with_its_tail():
    """The support's mass is NOT preserved any more -- that was the exchange. What
    holds is that the support plus its rescaled tail is a distribution, which is
    also what topk_kl_per_token reconstructs as 1 - sum_S exp(target_logprob)."""
    kw = _chain()
    got = build_target(**kw)
    tail_before = 1.0 - kw["on_logprob"].exp().sum(dim=-1)
    on_support = got["target_logprob"].exp().sum(dim=-1)
    assert torch.allclose(on_support + tail_before / got["log_z"].exp().float(),
                          torch.ones_like(on_support), atol=1e-6)
    # and the support's own mass really did move, or the test above is vacuous
    assert not torch.allclose(on_support, kw["on_logprob"].exp().sum(dim=-1), atol=1e-4)


def test_identity_when_nothing_agrees():
    """c == 0 everywhere must leave the target bit-identical. The predecessor's
    target arm could not say this: its renormalisation moved the distribution
    even at unit weight."""
    kw = _chain()
    # One off-task teacher silent everywhere -> L == 0 -> c == 0.
    kw["off_logprob"] = kw["base_logprob"].unsqueeze(-1).expand_as(kw["off_logprob"]).clone()
    got = build_target(**kw)
    assert torch.all(got["c"] == 0.0)
    assert torch.equal(got["target_logprob"], kw["on_logprob"])


def test_cold_start_rows_are_untouched():
    kw = _chain()
    kw["diag_valid"] = torch.tensor([False, True, True])
    got = build_target(**kw)
    assert torch.equal(got["target_logprob"], kw["on_logprob"])


def test_exponent_scale_is_the_only_knob_and_it_moves_the_intervention():
    kw = _chain()
    small = build_target(**kw, exponent_scale=1.0)
    large = build_target(**kw, exponent_scale=2.148)     # what the arm now pins
    assert large["moved"].sum() > small["moved"].sum()


def test_shuffled_counterfactual_is_built_on_request():
    kw = _chain()
    bs, resp = kw["on_logprob"].shape[:2]
    got = build_target(**kw, shuffle_counterfactual=True,
                       response_mask=torch.ones(bs, resp))
    assert "shuffled_moved" in got and got["shuffled_moved"].shape == got["moved"].shape


def test_channel_counterfactuals_partition_sensibly():
    kw = _chain()
    got = build_target(**kw, channel_counterfactuals=True)
    # Nothing exact holds across the two -- a total variation is not additive in
    # the exponent -- so what is asserted is that each is a TV of its own channel:
    # non-negative, and zero exactly when that channel is silent.
    for name in ("a", "b"):
        assert torch.all(got[f"{name}_only_moved"] >= 0)
        silent = (got[name] == 0).all(dim=-1)
        assert torch.all(got[f"{name}_only_moved"][silent] == 0)


def test_step_stats_render_the_preregistered_quantities():
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats, sign_state_labels

    kw = _chain(bs=3, resp=4, k=8)
    bs, resp = kw["on_logprob"].shape[:2]
    mask = torch.ones(bs, resp)
    got = build_target(**kw, shuffle_counterfactual=True,
                       channel_counterfactuals=True, response_mask=mask)
    stats = TargetStepStats(n_tasks=3, device="cpu")
    ids = torch.randint(0, 50, kw["on_logprob"].shape)
    stats.update(built=got, p_on=got["p_on"], support_ids=ids,
                 response_mask=mask, task_ids=kw["task_ids"],
                 d_on=torch.rand(bs, resp), d_base=torch.rand(bs, resp) + 1.0,
                 student_logprob=torch.log_softmax(
                     torch.randn(*kw["on_logprob"].shape), dim=-1) + math.log(0.99))
    m = stats.metrics(task_names=["alfworld", "search", "webshop"])
    for key in ("target/tv", "target/live_frac", "target/acted_mass_frac",
                "target/branch/agree/mass_frac", "target/max_abs_log_w",
                "target/max_abs_log_z", "target/log_z_mean", "target/dkl_mean",
                "target/abs_dkl_mean", "target/abs_dkl_live_mean",
                "target/mass_error_max", "target/shuffled_tv_ratio",
                "target/alfworld/tv"):
        assert key in m, key
    # The throttle metrics belonged to the exchange and must not linger.
    assert not any(k.endswith("throttle_up") or k.endswith("throttle_down") for k in m)
    # "p_tilde is a distribution", as the metric reports it.
    assert m["target/mass_error_max"] < 1e-12
    # branch fracs sum to one over candidates
    tot = sum(m[f"target/branch/{b}/cand_frac"] for b in ("agree", "conflict", "on_silent", "split"))
    assert abs(tot - 1.0) < 1e-9
    labels = sign_state_labels(got["branch"], got["consensus_sign"])
    assert labels.min() >= 0 and labels.max() <= 6


def test_per_task_sigma_is_read_independently():
    """A row's own teacher and its sources are each divided by their own diagonal,
    so inflating one task's sigma must change only what reads it."""
    kw = _chain()
    a = build_target(**kw)["c"]
    kw2 = dict(kw)
    kw2["diag"] = torch.tensor([1.0, 10.0, 1.0])
    b = build_target(**kw2)["c"]
    assert not torch.equal(a, b)
