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
"""What the SHRINK mode claims by construction, checked as arithmetic.

Same rule as the tilt and curriculum tests: every assertion here is a line in
``docs/cross_teacher_shrink_design.md`` that says "by construction".

Three of these carry the design's whole argument:
``test_lambda_one_is_the_control_bit_for_bit`` (the arm contains its own
control), ``test_the_target_is_the_normalised_geometric_mixture`` and
``test_mixing_the_kls_is_the_mixed_target`` (writing the mixture in the loss and
writing it in the target are the same mechanism -- the claim the design is
built on), and ``test_the_target_leaves_the_base_to_on_task_bracket`` (this arm
INJECTS, which is exactly what the curriculum arm does not do).
"""
import math

import pytest
import torch

from verl.trainer.ppo.cross_teacher_target import (
    _EXPONENT_CLAMP,
    build_target,
    shrink_exponent,
)

K_OFF = 2


def _chain(k=6, bs=2, resp=3, n_off=K_OFF, seed=11, headroom=0.99):
    """The tilt/curriculum fixture, so all three modes are compared on one input."""
    gen = torch.Generator().manual_seed(seed)
    lg = math.log(headroom)
    on = torch.log_softmax(torch.randn(bs, resp, k, generator=gen), dim=-1) + lg
    base = torch.log_softmax(torch.randn(bs, resp, k, generator=gen), dim=-1) + lg
    off = torch.log_softmax(torch.randn(bs, resp, k, n_off, generator=gen), dim=-2) + lg
    return dict(
        on_logprob=on, off_logprob=off, base_logprob=base,
        diag=torch.ones(3), diag_valid=torch.ones(3, dtype=torch.bool),
        task_ids=torch.zeros(bs, dtype=torch.long),
        off_plane_tasks=torch.tensor([[1, 2]] * bs),
    )


def _shrink(lam, **over):
    kw = _chain()
    kw.update(over)
    return build_target(mode="shrink", lambda_prime=lam, **kw), kw


# --------------------------------------------------------------- the endpoints
def test_lambda_one_is_the_control_bit_for_bit():
    """lambda' = 1 has to reproduce the on-task teacher's own BITS, not approximate it.

    The arm contains its own control, so every comparison against the control is
    a comparison against this line. `(1 - lambda')` is exactly 0.0 in floating
    point, which is why the tilt is written as a subtraction off the RAW shift.
    """
    got, kw = _shrink(1.0)
    assert torch.equal(got["target_logprob"], kw["on_logprob"])
    assert torch.all(got["c"] == 0.0)
    assert not bool(got["live"].any())
    assert float(got["moved"].max()) == 0.0
    assert not bool(got["beyond"].any())


def test_lambda_one_third_is_the_equal_weight_mean_of_all_three_teachers():
    """1/K removes routing from the TARGET: every teacher gets the same weight."""
    got, kw = _shrink(1.0 / 3.0)
    h_on = kw["on_logprob"] - kw["base_logprob"]
    h_off = kw["off_logprob"] - kw["base_logprob"].unsqueeze(-1)
    want = (h_on + h_off.sum(dim=-1)) / 3.0
    assert torch.allclose(got["a"], want, atol=1e-6)


def test_lambda_prime_is_required_and_bounded():
    kw = _chain()
    with pytest.raises(AssertionError, match="needs lambda_prime"):
        build_target(mode="shrink", **kw)
    with pytest.raises(AssertionError, match="not a weight"):
        build_target(mode="shrink", lambda_prime=1.5, **kw)


# ------------------------------------------------------- the mixture, two ways
def test_the_target_is_the_normalised_geometric_mixture():
    """log p_tilde = sum_m w_m log pi_m - log Z, with w = (0.6, 0.2, 0.2)."""
    kw = _chain(headroom=1.0)                       # no tail, so Z closes on the support
    got = build_target(mode="shrink", lambda_prime=0.6, **kw)
    mixed = (
        0.6 * kw["on_logprob"]
        + 0.2 * kw["off_logprob"][..., 0]
        + 0.2 * kw["off_logprob"][..., 1]
    )
    assert torch.allclose(got["target_logprob"], torch.log_softmax(mixed, dim=-1), atol=1e-5)
    assert torch.allclose(got["target_logprob"].exp().sum(dim=-1),
                          torch.ones(kw["on_logprob"].shape[:2]), atol=1e-5)


def test_mixing_the_kls_is_the_mixed_target():
    """sum_m w_m KL(p_s || pi_m) - KL(p_s || p_tilde) is a constant in p_s.

    This is the design's central identity: putting the off-task teachers in the
    LOSS as extra KL terms and putting them in the TARGET as a geometric mixture
    differ by ``log Z``, which does not depend on the student and so has no
    gradient. Checked on two unrelated students -- a constant is only a constant
    if it does not move.
    """
    kw = _chain(headroom=1.0)
    got = build_target(mode="shrink", lambda_prime=0.6, **kw)
    w = (0.6, 0.2, 0.2)
    teachers = [kw["on_logprob"], kw["off_logprob"][..., 0], kw["off_logprob"][..., 1]]
    gaps = []
    for seed in (1, 2):
        gen = torch.Generator().manual_seed(seed)
        ls = torch.log_softmax(torch.randn(kw["on_logprob"].shape, generator=gen), dim=-1)
        ps = ls.exp()
        kl = lambda t: (ps * (ls - t)).sum(dim=-1)
        gaps.append(sum(wi * kl(t) for wi, t in zip(w, teachers)) - kl(got["target_logprob"]))
    assert torch.allclose(gaps[0], gaps[1], atol=1e-5)
    # ...and that constant IS -log Z, measured the way the metrics report it.
    assert torch.allclose(gaps[0].to(torch.float64), -got["log_z"], atol=1e-5)


def test_the_off_task_voices_are_converted_into_the_destination_unit():
    """sigma_d * mean_m (h_m / sigma_m), not the raw mean: the teachers have
    different gains (search was trained at a 10x smaller KL coefficient)."""
    kw = _chain()
    kw["diag"] = torch.tensor([1.0, 2.0, 4.0])
    got = build_target(mode="shrink", lambda_prime=0.5, **kw)
    h_off = kw["off_logprob"] - kw["base_logprob"].unsqueeze(-1)
    sigma_d = 1.0                                   # diag[task_ids == 0]
    want = sigma_d * (h_off[..., 0] / 2.0 + h_off[..., 1] / 4.0) / 2.0
    assert torch.allclose(got["off_nats"], want, atol=1e-6)
    # the raw mean is a DIFFERENT number, which is the point of the conversion
    assert not torch.allclose(got["off_nats"], h_off.mean(dim=-1), atol=1e-3)


# ------------------------------------------------------------------- injection
def test_the_target_leaves_the_base_to_on_task_bracket():
    """THE DIFFERENCE FROM mode="curriculum", which cannot leave it at all.

    `beyond` is true exactly where the tilt on base opposes the on-task shift or
    exceeds it, i.e. where the off-task teachers pushed past what the on-task
    teacher said. Hand-built so the two ways out of the bracket are both hit.
    """
    sig = torch.ones(1, 1, 1)
    # candidate 0: off agrees but is louder -> past the on-task teacher
    # candidate 1: off opposes            -> the other side of base
    # candidate 2: off agrees and quieter -> inside, shrinkage only
    on = torch.tensor([[[2.0, 2.0, 2.0]]])
    off = torch.tensor([[[[6.0, 6.0], [-6.0, -6.0], [1.0, 1.0]]]])
    got = shrink_exponent(shift_on=on, hat_off=off, sigma_on=sig, lambda_prime=0.6)
    assert got["beyond"].reshape(-1).tolist() == [True, True, False]
    a = got["a"].reshape(-1)
    assert a[0].item() > 2.0                 # past the on-task shift
    assert a[1].item() < 0.0                 # across base
    assert 0.0 < a[2].item() < 2.0           # strictly inside


def test_injection_grows_as_lambda_falls_and_is_empty_at_one():
    kw = _chain()
    fracs = []
    for lam in (1.0, 0.8, 0.6, 1.0 / 3.0):
        got = build_target(mode="shrink", lambda_prime=lam, **kw)
        fracs.append(float(got["beyond"].to(torch.float64).mean()))
    assert fracs[0] == 0.0
    assert fracs == sorted(fracs), fracs


# ----------------------------------------------------------------- the ratelimit
def test_the_clamp_binds_and_is_reported():
    """The clamp STAYS in this mode: nothing else bounds one candidate's move."""
    kw = _chain()
    kw["off_logprob"] = kw["off_logprob"] - 40.0 * torch.nn.functional.one_hot(
        torch.zeros(kw["off_logprob"].shape[:2], dtype=torch.long), kw["off_logprob"].size(2)
    ).unsqueeze(-1).to(kw["off_logprob"].dtype)
    got = build_target(mode="shrink", lambda_prime=0.6, **kw)
    assert bool(got["clamped"].any())
    assert float(got["c_eff"].abs().max()) <= _EXPONENT_CLAMP + 1e-6
    # and the distribution is still a distribution after the clamp
    total = got["target_logprob"].exp().sum(dim=-1) + got["tail"] * got["inv_z"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_the_tail_is_rescaled_and_never_tilted():
    kw = _chain()
    got = build_target(mode="shrink", lambda_prime=0.6, **kw)
    tail_t = got["tail"] * got["inv_z"]
    total = got["target_logprob"].exp().sum(dim=-1) + tail_t
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert torch.all(got["tail"] > 0)


# -------------------------------------------------------------------- the table
def _stats_for(lam, *, bs=3, resp=4, k=8):
    from verl.trainer.ppo.cross_teacher_target import TargetStepStats

    kw = _chain(bs=bs, resp=resp, k=k)
    mask = torch.ones(bs, resp)
    got = build_target(mode="shrink", lambda_prime=lam,
                       shuffle_counterfactual=True, response_mask=mask, **kw)
    stats = TargetStepStats(n_tasks=3, device="cpu", mode="shrink")
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
    return stats.metrics(task_names=["alfworld", "search", "webshop"])


def test_step_stats_render_the_shrink_quantities_and_not_the_other_modes():
    m = _stats_for(0.6)
    for key in ("target/tv", "target/abs_dkl_mean", "target/entropy_delta",
                "target/shrink/off_to_on_absmass", "target/shrink/c_to_on_absmass",
                "target/shrink/agree_cand_frac", "target/shrink/agree_mass_frac",
                "target/shrink/beyond_cand_frac", "target/shrink/beyond_mass_frac",
                "target/shrink/role/structural_share",
                "target/shrink/role/content_share",
                "target/kl_to_base", "target/clamped_per_step",
                "target/shuffled_tv_ratio",          # G1, in its original sense
                "target/branch/agree/mass_frac", "target/alfworld/tv"):
        assert key in m, key
    # a column a structurally-absent channel would fill with zeros reads as a
    # measurement, so the other modes' keys must not be rendered at all
    for key in ("target/channel/a_share", "target/channel/b_share",
                "target/layer/shared/backed_frac", "target/stage_kl/shared",
                "target/retained_shuffled_ratio"):
        assert key not in m, key


def test_the_table_says_nothing_happened_at_lambda_one():
    m = _stats_for(1.0)
    assert m["target/tv"] == 0.0
    assert m["target/live_frac"] == 0.0
    assert m["target/shrink/beyond_cand_frac"] == 0.0
    assert m["target/shrink/c_to_on_absmass"] == 0.0
    assert m["target/abs_dkl_mean"] == 0.0
    assert m["target/mass_error_max"] == 0.0


def test_the_dose_the_table_reports_is_the_one_lambda_sets():
    """|c| = (1 - lambda') |off_nats - h_d|, so halving (1 - lambda') halves it."""
    a = _stats_for(0.6)["target/shrink/c_to_on_absmass"]
    b = _stats_for(0.8)["target/shrink/c_to_on_absmass"]
    assert a == pytest.approx(2.0 * b, rel=1e-6)
