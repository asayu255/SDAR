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

"""Tests for the per-id precision weighting of the RL and OPD signals.

The load-bearing one is section 1: the closed-form logit gradient is compared
against autograd of the loss the trainer actually uses, so a change to
``topk_kl_per_token`` that is not mirrored in ``topk_kl_logit_grad`` fails here
rather than silently aggregating the gradient of the old objective.
"""

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import topk_kl_per_token
from verl.trainer.ppo.opd_logit_precision import (
    LogitPrecisionConfig,
    LmHeadState,
    StepAccumulator,
    StepMoments,
    TaskFit,
    fit_weights,
    lm_head_metrics,
    select_lm_ids,
    topk_kl_logit_grad,
)

V, K = 37, 5


def _mk(bs=3, resp=4, vocab=V, k=K, seed=0, tail=True):
    """Logits plus a student top-k support and a teacher scored at the same ids.

    ``tail=False`` makes the support hold essentially all the mass, which is the
    case where the loss's tail clamp binds and the gradient loses that term.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs, resp, vocab, generator=g, dtype=torch.float64)
    if not tail:
        logits[..., :k] += 40.0
    topk_ids = logits.topk(k, dim=-1).indices
    teacher = torch.randn(bs, resp, vocab, generator=g, dtype=torch.float64)
    t_lp = torch.log_softmax(teacher, dim=-1).gather(-1, topk_ids)
    return logits, topk_ids, t_lp


def _student_topk_logprob(logits, topk_ids):
    return torch.log_softmax(logits, dim=-1).gather(-1, topk_ids)


# ---------------------------------------------------------------------------
# 1. the closed form is the loss's own gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tail", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_closed_form_reproduces_autograd_of_the_loss(tail, seed):
    logits, topk_ids, t_lp = _mk(seed=seed, tail=tail)
    z = logits.clone().requires_grad_(True)
    loss = topk_kl_per_token(_student_topk_logprob(z, topk_ids), t_lp).sum()
    (want,) = torch.autograd.grad(loss, z)
    want = -want                                     # the module reports -dL/dz

    support, kappa = topk_kl_logit_grad(_student_topk_logprob(logits, topk_ids), t_lp)
    pi = torch.softmax(logits, dim=-1)
    got = pi * kappa.unsqueeze(-1)
    got.scatter_add_(-1, topk_ids, support)

    assert torch.allclose(got, want, atol=1e-9, rtol=1e-7), (got - want).abs().max()


def test_the_opd_logit_gradient_sums_to_zero_over_the_vocabulary():
    """Softmax is shift invariant, so every logit gradient must sum to zero."""
    logits, topk_ids, t_lp = _mk(seed=3)
    support, kappa = topk_kl_logit_grad(_student_topk_logprob(logits, topk_ids), t_lp)
    pi = torch.softmax(logits, dim=-1)
    g = pi * kappa.unsqueeze(-1)
    g.scatter_add_(-1, topk_ids, support)
    assert g.sum(-1).abs().max() < 1e-12


def test_a_teacher_equal_to_the_student_pushes_nothing():
    logits, topk_ids, _ = _mk(seed=4)
    s_lp = _student_topk_logprob(logits, topk_ids)
    support, kappa = topk_kl_logit_grad(s_lp, s_lp.clone())
    assert support.abs().max() < 1e-12
    assert kappa.abs().max() < 1e-12


# ---------------------------------------------------------------------------
# 2. the accumulator reproduces a brute-force sum
# ---------------------------------------------------------------------------


def _brute(logits, responses, mask, pg_coef, rho, topk_ids, s_lp, t_lp):
    pi = torch.softmax(logits.double(), dim=-1)
    bs, resp, vocab = logits.shape
    a = torch.zeros(bs, vocab, dtype=torch.float64)
    d = torch.zeros(bs, vocab, dtype=torch.float64)
    support, kappa = topk_kl_logit_grad(s_lp.double(), t_lp.double())
    for b in range(bs):
        for t in range(resp):
            if mask[b, t] == 0:
                continue
            w = float(rho[b])
            onehot = torch.zeros(vocab, dtype=torch.float64)
            onehot[responses[b, t]] = 1.0
            a[b] += w * float(pg_coef[b, t]) * (onehot - pi[b, t])
            g = pi[b, t] * float(kappa[b, t])
            g = g.index_add(0, topk_ids[b, t], support[b, t].double())
            d[b] += w * g
    return a, d


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_the_accumulator_matches_a_brute_force_sum_and_ignores_chunking(chunk):
    torch.manual_seed(7)
    bs, resp = 4, 6
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=11)
    s_lp = _student_topk_logprob(logits, topk_ids)
    responses = torch.randint(0, V, (bs, resp))
    mask = torch.ones(bs, resp)
    mask[0, -2:] = 0
    pg_coef = torch.randn(bs, resp, dtype=torch.float64)
    ratio = torch.rand(bs, resp, dtype=torch.float64) + 0.5
    rho = torch.rand(bs, dtype=torch.float64) + 0.1
    task_ids = torch.tensor([0, 0, 1, 1])
    adv = torch.randn(bs, dtype=torch.float64)

    acc = StepAccumulator(2, V, dtype=torch.float64)
    acc.add_micro_batch(
        logits=logits, responses=responses, response_mask=mask, task_ids=task_ids,
        row_weight=rho, pg_coef=pg_coef, ratio=ratio, row_advantage=adv,
        topk_ids=topk_ids, student_topk_logprob=s_lp, teacher_topk_logprob=t_lp,
        chunk_tokens=chunk,
    )
    a_rows, d_rows = _brute(logits, responses, mask, pg_coef, rho, topk_ids, s_lp, t_lp)
    for task in (0, 1):
        sel = (task_ids == task).nonzero(as_tuple=True)[0]
        assert torch.allclose(acc.a[task], a_rows[sel].sum(0), atol=1e-10), task
        assert torch.allclose(acc.d[task], d_rows[sel].sum(0), atol=1e-10), task


def test_both_pushes_sum_to_zero_over_the_vocabulary():
    """The property the fit relies on to skip centring the second moments."""
    torch.manual_seed(8)
    bs, resp = 3, 5
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=12)
    acc = StepAccumulator(1, V, dtype=torch.float64)
    acc.add_micro_batch(
        logits=logits, responses=torch.randint(0, V, (bs, resp)),
        response_mask=torch.ones(bs, resp), task_ids=torch.zeros(bs, dtype=torch.long),
        row_weight=torch.ones(bs, dtype=torch.float64),
        pg_coef=torch.randn(bs, resp, dtype=torch.float64),
        ratio=torch.ones(bs, resp, dtype=torch.float64),
        row_advantage=torch.randn(bs, dtype=torch.float64),
        topk_ids=topk_ids, student_topk_logprob=_student_topk_logprob(logits, topk_ids),
        teacher_topk_logprob=t_lp,
    )
    assert abs(float(acc.a[0].sum())) < 1e-10
    assert abs(float(acc.d[0].sum())) < 1e-10


def test_rows_with_no_task_are_skipped():
    torch.manual_seed(9)
    bs, resp = 4, 3
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=13)
    kw = dict(
        logits=logits, responses=torch.randint(0, V, (bs, resp)),
        response_mask=torch.ones(bs, resp),
        row_weight=torch.ones(bs, dtype=torch.float64),
        pg_coef=torch.ones(bs, resp, dtype=torch.float64),
        ratio=torch.ones(bs, resp, dtype=torch.float64),
        row_advantage=torch.ones(bs, dtype=torch.float64),
        topk_ids=topk_ids, student_topk_logprob=_student_topk_logprob(logits, topk_ids),
        teacher_topk_logprob=t_lp,
    )
    full = StepAccumulator(1, V, dtype=torch.float64).add_micro_batch(
        task_ids=torch.tensor([0, 0, -1, -1]), **kw)
    half = StepAccumulator(1, V, dtype=torch.float64).add_micro_batch(
        task_ids=torch.tensor([0, 0, -1, -1]), **{**kw, "logits": logits})
    assert float(full.n_rows[0]) == 2.0
    assert torch.allclose(full.a[0], half.a[0])


# ---------------------------------------------------------------------------
# 3. the permutation null
# ---------------------------------------------------------------------------


def test_the_closed_form_permutation_variance_matches_monte_carlo():
    """sigma^2 is the variance of the RL push under permuting the advantages."""
    torch.manual_seed(5)
    n_rows, vocab = 12, 9
    m = torch.randn(n_rows, vocab, dtype=torch.float64)          # row profiles
    adv = torch.randn(n_rows, dtype=torch.float64)
    adv = adv - adv.mean()                                        # GRPO centres them

    sm, sm2, sa2 = m.sum(0), (m * m).sum(0), float((adv * adv).sum())
    spread = sm2 - sm * sm / n_rows
    closed = spread * (sa2 / (n_rows - 1))

    g = torch.Generator().manual_seed(1)
    draws = torch.stack([adv[torch.randperm(n_rows, generator=g)] @ m for _ in range(40000)])
    assert draws.mean(0).abs().max() < 0.05
    mc = draws.var(0, unbiased=True)
    assert torch.allclose(mc, closed, rtol=0.06), (mc / closed)


# ---------------------------------------------------------------------------
# 4. the fit
# ---------------------------------------------------------------------------


def _synthetic(lam, noise_rl, noise_teacher, vocab=4000, n_rows=64, seed=0):
    """Moments that follow the model the fit assumes.

    Built through StepAccumulator.moments() rather than by hand, so the tests
    exercise the same permutation-variance arithmetic the trainer runs.
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(vocab, generator=g, dtype=torch.float64)
    acc = StepAccumulator(1, vocab, dtype=torch.float64)
    acc.a[0] = theta + noise_rl * torch.randn(vocab, generator=g, dtype=torch.float64)
    acc.d[0] = lam * theta + noise_teacher * torch.randn(vocab, generator=g, dtype=torch.float64)
    acc.act[0] = torch.rand(vocab, generator=g, dtype=torch.float64) + 0.1
    # Row profiles whose permutation variance comes out at noise_rl^2.
    acc.n_rows[0] = n_rows
    acc.sa2[0] = n_rows
    acc.sm[0] = torch.zeros(vocab, dtype=torch.float64)
    acc.sm2[0] = torch.full((vocab,), noise_rl ** 2 * (n_rows - 1) / n_rows, dtype=torch.float64)
    acc.n_tokens[0] = 10_000
    return acc.moments(LogitPrecisionConfig())


def _stream(lam, noise_rl, noise_teacher, *, steps=40, decay=0.8, vocab=4000,
            n_rows=64, seed=0, live_frac=1.0):
    """A smoothed stream of noisy steps -- what the trainer actually fits on.

    ``live_frac`` is the share of rows carrying a nonzero advantage. At small
    values the permutation null collapses (sum_r A_r^2 goes to zero with the
    push), which is the case the across-step variance exists to survive.
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(vocab, generator=g, dtype=torch.float64)
    cfg = LogitPrecisionConfig()
    ema = StepMoments(1, vocab, dtype=torch.float64)
    n_live = max(2, int(n_rows * live_frac))
    for _ in range(steps):
        acc = StepAccumulator(1, vocab, dtype=torch.float64)
        acc.a[0] = theta + noise_rl * torch.randn(vocab, generator=g, dtype=torch.float64)
        acc.d[0] = lam * theta + noise_teacher * torch.randn(vocab, generator=g, dtype=torch.float64)
        acc.act[0] = torch.rand(vocab, generator=g, dtype=torch.float64) + 0.1
        acc.n_rows[0] = n_rows
        # Only the live rows carry advantage, so the null shrinks with them.
        acc.sa2[0] = float(n_live)
        acc.sm[0] = torch.zeros(vocab, dtype=torch.float64)
        acc.sm2[0] = torch.full((vocab,), noise_rl ** 2 * (n_rows - 1) / n_rows,
                                dtype=torch.float64)
        acc.n_tokens[0] = 10_000
        ema.ema_(acc.moments(cfg), decay)
    return ema


@pytest.mark.parametrize("lam", [0.25, 1.0, 3.0])
def test_the_fit_recovers_a_known_teacher_scale(lam):
    acc = _stream(lam, noise_rl=0.5, noise_teacher=0.3, seed=100 + int(lam * 4))
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    assert fit.valid and fit.reason in ("ok", "residual_floored"), fit.reason
    assert fit.sigma2_is_across_step
    assert fit.lam == pytest.approx(lam, rel=0.08), fit.lam
    assert fit.varsigma2 == pytest.approx(0.09, rel=0.25), fit.varsigma2
    assert fit.tau2 == pytest.approx(1.0, rel=0.10), fit.tau2


def test_a_teacher_that_is_pure_noise_is_switched_off():
    # Seeded apart from _synthetic's own generator: reusing seed 2 would redraw
    # the same normals it used for theta and hand the fit a perfect teacher.
    g = torch.Generator().manual_seed(999)
    acc = _synthetic(0.0, noise_rl=0.5, noise_teacher=1.0, seed=2)
    acc.d[0] = torch.randn(acc.vocab, generator=g, dtype=torch.float64)
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    assert fit.valid
    assert abs(fit.lam_raw) < 0.1, fit.lam_raw
    if fit.reason == "teacher_uninformative":
        assert float(fit.implied_beta.abs().max()) == 0.0
    else:
        assert float(fit.implied_beta[fit.keep].median()) < 0.2


def test_a_teacher_pulling_against_its_own_reward_is_reported_and_clamped():
    acc = _synthetic(-1.0, noise_rl=0.4, noise_teacher=0.3, seed=4)
    off = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    assert off.lam_raw < 0, off.lam_raw
    assert off.lam == 0.0
    assert off.reason == "teacher_uninformative"
    on = fit_weights(acc, ["t"], LogitPrecisionConfig(allow_negative_lambda=True))["t"]
    assert on.lam == pytest.approx(-1.0, rel=0.10)


def test_the_implied_beta_is_the_ratio_the_loss_would_use():
    """implied_beta must equal lambda sigma^2 / varsigma^2, in units of kl_loss_coef."""
    acc = _synthetic(2.0, noise_rl=0.5, noise_teacher=0.3, seed=6)
    cfg = LogitPrecisionConfig()
    fit = fit_weights(acc, ["t"], cfg)["t"]
    want = fit.lam * fit.sigma2 / fit.varsigma2
    assert torch.allclose(fit.implied_beta[fit.keep], want[fit.keep], rtol=1e-9)


def test_a_noisier_reward_at_an_id_gives_the_teacher_more_weight():
    """The whole point: beta is not one constant, it tracks the reward's noise.

    The extra noise goes into BOTH the observed push and the null that measures
    it. Raising only the null would tell the fit that ids are noisy while the
    data says otherwise, which underestimates the prior and is a different test.
    """
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, seed=8)
    g = torch.Generator().manual_seed(81)
    extra = 3.5                                     # sd of the added reward noise
    acc.a[0][:100] += extra * torch.randn(100, generator=g, dtype=torch.float64)
    acc.a2[0][:100] = acc.a[0][:100] ** 2
    acc.ad[0][:100] = acc.a[0][:100] * acc.d[0][:100]
    acc.sig2[0][:100] += extra ** 2
    # base_beta matches the synthetic scale (theta ~ N(0,1)), so the relative
    # cap sits far above both groups and the comparison is of the fit, not of
    # the clamp.
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    noisy = fit.implied_beta[:100].median()
    rest = fit.implied_beta[100:][fit.keep[100:]].median()
    assert float(noisy) > 10.0 * float(rest), (noisy, rest)


def test_a_task_with_too_little_data_is_refused_rather_than_guessed():
    acc = StepAccumulator(1, 500, dtype=torch.float64).moments(LogitPrecisionConfig())
    acc.n_tokens[0] = 3
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    assert not fit.valid and fit.reason == "too_few_tokens"
    assert fit.implied_beta is None


def test_dropped_ids_fall_back_to_the_base_coefficient():
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, seed=10)
    acc.act[0][:50] = 0.0
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=0.01)["t"]
    assert not bool(fit.keep[:50].any())
    assert torch.allclose(fit.implied_beta[:50], torch.full((50,), 0.01, dtype=torch.float64))


# ---------------------------------------------------------------------------
# 5. plumbing
# ---------------------------------------------------------------------------


def test_the_ema_is_a_convex_combination_field_by_field():
    a = _synthetic(1.0, 0.5, 0.3, vocab=64, seed=20)
    b = _synthetic(2.0, 0.5, 0.3, vocab=64, seed=21)
    want = 0.8 * a.a2[0] + 0.2 * b.a2[0]
    a.ema_(b, 0.8)
    assert torch.allclose(a.a2[0], want)


def test_the_state_round_trips_and_refuses_a_reshaped_run():
    a = _synthetic(1.0, 0.5, 0.3, vocab=64, seed=22)
    b = StepMoments(1, 64, dtype=torch.float64)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.d2[0], b.d2[0])
    with pytest.raises(ValueError, match="vocab"):
        StepMoments(1, 65, dtype=torch.float64).load_state_dict(a.state_dict())


@pytest.mark.parametrize("bad", [{"ema_decay": 1.0}, {"n_strata": 0}, {"chunk_tokens": 0},
                                 {"sigma2_floor_frac": float("nan")}])
def test_the_config_refuses_values_that_would_silently_do_nothing(bad):
    with pytest.raises(ValueError):
        LogitPrecisionConfig(**bad).validate()


def test_metrics_name_every_task_and_carry_the_headline_ratio():
    acc = _synthetic(1.5, 0.5, 0.3, seed=24)
    m = fit_weights(acc, ["alfworld"], LogitPrecisionConfig())["alfworld"].metrics()
    assert m["logit_prec/alfworld/valid"] == 1.0
    assert "logit_prec/alfworld/beta_median" in m
    assert "logit_prec/alfworld/lam_raw" in m


def test_the_beta_cap_binds_visibly_rather_than_silently():
    """A cap that starts deciding the answer has to show up in the metrics.

    lambda_max became the content of the cross-gate strength arm without that
    being visible until afterwards; ``beta_at_cap_frac`` is the guard against
    repeating it.
    """
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, seed=30)
    loose = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    assert loose.at_cap_frac == 0.0
    tight = fit_weights(acc, ["t"], LogitPrecisionConfig(max_beta_ratio=1.0),
                        base_beta=0.01)["t"]
    assert tight.at_cap_frac == pytest.approx(1.0)
    assert float(tight.implied_beta[tight.keep].abs().max()) <= 0.01 + 1e-12
    assert tight.metrics()["logit_prec/t/beta_at_cap_frac"] == pytest.approx(1.0)


def test_the_reason_is_reported_as_a_code_for_the_logger():
    acc = StepAccumulator(1, 500, dtype=torch.float64).moments(LogitPrecisionConfig())
    acc.n_tokens[0] = 3
    m = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"].metrics()
    assert m["logit_prec/t/reason"] == 3.0      # too_few_tokens
    assert m["logit_prec/t/valid"] == 0.0


# ---------------------------------------------------------------------------
# 6. the packed layout, which is the one the trainer uses
# ---------------------------------------------------------------------------


def _pack(dense_logits, mask):
    """(bs, resp, V) + mask -> the (T, V) layout remove-padding produces."""
    sel = mask.reshape(-1).bool()
    bs, resp, _ = dense_logits.shape
    rows = torch.arange(bs).unsqueeze(-1).expand(bs, resp).reshape(-1)
    return sel, rows[sel]


def test_the_packed_path_agrees_with_the_padded_one_on_ragged_rows():
    """Production packs away the padding, and the rows here are of three lengths."""
    torch.manual_seed(21)
    bs, resp = 5, 8
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=31)
    s_lp = _student_topk_logprob(logits, topk_ids)
    responses = torch.randint(0, V, (bs, resp))
    lengths = torch.tensor([8, 3, 6, 1, 5])
    mask = (torch.arange(resp).unsqueeze(0) < lengths.unsqueeze(-1)).to(torch.float64)
    pg_coef = torch.randn(bs, resp, dtype=torch.float64)
    ratio = torch.rand(bs, resp, dtype=torch.float64) + 0.5
    rho = torch.rand(bs, dtype=torch.float64) + 0.1
    task_ids = torch.tensor([0, 1, 0, 1, -1])
    adv = torch.randn(bs, dtype=torch.float64)

    padded = StepAccumulator(2, V, dtype=torch.float64).add_micro_batch(
        logits=logits, responses=responses, response_mask=mask, task_ids=task_ids,
        row_weight=rho, pg_coef=pg_coef, ratio=ratio, row_advantage=adv,
        topk_ids=topk_ids, student_topk_logprob=s_lp, teacher_topk_logprob=t_lp,
    )

    sel, token_row = _pack(logits, mask)
    support, kappa = topk_kl_logit_grad(s_lp, t_lp)
    packed = StepAccumulator(2, V, dtype=torch.float64).add_packed(
        logits=logits.reshape(-1, V)[sel],
        token_row=token_row,
        token_task=task_ids.clamp(min=0).unsqueeze(-1).expand(bs, resp).reshape(-1)[sel],
        sampled_ids=responses.reshape(-1)[sel],
        w_rl=(pg_coef * rho.unsqueeze(-1)).reshape(-1)[sel],
        w_opd=(kappa * rho.unsqueeze(-1)).reshape(-1)[sel],
        w_act=rho.unsqueeze(-1).expand(bs, resp).reshape(-1)[sel],
        w_prof=(ratio * rho.unsqueeze(-1)).reshape(-1)[sel],
        row_task=task_ids, row_adv=adv,
        topk_ids=topk_ids.reshape(bs * resp, -1)[sel],
        support_term=support.reshape(bs * resp, -1)[sel],
    )
    for f in ("a", "d", "act", "sm", "sm2", "sa2", "n_rows", "n_tokens"):
        got, want = getattr(packed, f), getattr(padded, f)
        assert torch.allclose(got, want, atol=1e-11), (f, (got - want).abs().max())


def test_the_packed_path_drops_rows_with_no_task():
    """A row tagged -1 is adjust_batch padding and must not reach any task."""
    torch.manual_seed(22)
    bs, resp = 3, 4
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=32)
    s_lp = _student_topk_logprob(logits, topk_ids)
    mask = torch.ones(bs, resp, dtype=torch.float64)
    kw = dict(
        logits=logits, responses=torch.randint(0, V, (bs, resp)), response_mask=mask,
        row_weight=torch.ones(bs, dtype=torch.float64),
        pg_coef=torch.randn(bs, resp, dtype=torch.float64),
        ratio=torch.ones(bs, resp, dtype=torch.float64),
        row_advantage=torch.ones(bs, dtype=torch.float64),
        topk_ids=topk_ids, student_topk_logprob=s_lp, teacher_topk_logprob=t_lp,
    )
    with_pad = StepAccumulator(1, V, dtype=torch.float64).add_micro_batch(
        task_ids=torch.tensor([0, 0, -1]), **kw)
    only_real = StepAccumulator(1, V, dtype=torch.float64).add_micro_batch(
        task_ids=torch.tensor([0, 0, -1]),
        **{**kw, "response_mask": mask.clone()})
    assert float(with_pad.n_rows[0]) == 2.0
    assert float(with_pad.n_tokens[0]) == 8.0
    assert torch.allclose(with_pad.d[0], only_real.d[0])


# ---------------------------------------------------------------------------
# 7. the wiring: config injection and the guards on the actor
# ---------------------------------------------------------------------------


def test_the_config_reaches_the_actor_under_its_own_key():
    import os
    import pathlib

    from hydra import compose, initialize_config_dir

    from verl.trainer.main_opd import inject_distillation_config

    root = pathlib.Path(__file__).resolve().parents[2]
    cfgdir = str(root / "verl" / "trainer" / "config")
    base = [
        "+algorithm.opd.task_diag=True",
        "+algorithm.opd.kl_loss_coef=0.01",
        "+algorithm.opd.kl_loss_type=topk_kl",
        "+algorithm.opd.logit_precision.enable=True",
        "+algorithm.opd.logit_precision.observe_only=True",
        "+algorithm.opd.logit_precision.ema_decay=0.8",
    ]
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=base)
    inject_distillation_config(cfg)
    got = cfg.actor_rollout_ref.actor.logit_precision
    assert got is not None and got["enable"] is True and got["observe_only"] is True
    assert float(got["ema_decay"]) == 0.8

    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        off = compose(config_name="ppo_trainer", overrides=base[:3])
    inject_distillation_config(off)
    assert off.actor_rollout_ref.actor.logit_precision is None


def test_the_actor_refuses_the_settings_the_measurement_cannot_be_taken_under():
    """Three preconditions, each stated where it is needed rather than assumed."""
    import inspect

    from verl.workers.actor import dp_actor

    src = inspect.getsource(dp_actor)
    assert "logit_precision reads the packed response logits" in src
    assert "logit_precision measures the teacher's push on the STUDENT's top-k" in src
    # The OPD term is assembled once so the reweighting cannot miss a branch.
    # init, three assembly branches, one reweighting swap.
    assert src.count("_opd_term = ") == 5
    assert src.count("policy_loss = policy_loss + _opd_term") == 1
    # The applied weights are last step's, and the code has to say why.
    assert "One step of staleness" in src


def test_an_unknown_knob_is_refused_rather_than_dropped():
    known = set(LogitPrecisionConfig.__dataclass_fields__)
    assert "enable" in known and "observe_only" in known
    assert "emadecay" not in known           # the typo the guard exists to catch


def test_observe_only_is_the_default():
    """The pilot must be what you get by turning it on, not an extra flag."""
    assert LogitPrecisionConfig().observe_only is True
    assert LogitPrecisionConfig().enable is False


# ---------------------------------------------------------------------------
# 8. the failure mode a review found: do not switch the teacher off where the
#    reward is silent, which is where the teacher is the only source
# ---------------------------------------------------------------------------


def test_where_the_reward_says_nothing_the_teacher_still_gets_weight():
    """62 percent of webshop's tokens have zero advantage; the OPD term is the
    only gradient there. A prior estimated from the RL push alone collapses at
    exactly those coordinates, so it must not multiply the teacher's weight."""
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, seed=40)
    # A block of ids whose RL push is pure noise: the observed push IS the
    # noise, and the null says so.
    g = torch.Generator().manual_seed(41)
    n_dead = 400
    big = 6.0
    acc.a[0][:n_dead] = big * torch.randn(n_dead, generator=g, dtype=torch.float64)
    # The null is deliberately conservative (it permutes across the task's rows
    # rather than within a prompt group, destroying group identity as well), so
    # it comes out at or above the observed spread. 1.3x puts the stratum
    # unambiguously on the collapsed side instead of within sampling noise of
    # the boundary, which is what made the first version of this test flaky.
    acc.a2[0][:n_dead] = acc.a[0][:n_dead] ** 2
    acc.ad[0][:n_dead] = acc.a[0][:n_dead] * acc.d[0][:n_dead]
    acc.sig2[0][:n_dead] = 1.3 * big ** 2
    # Put them in their own activity stratum so the collapse is not diluted.
    acc.act[0][:n_dead] = 10.0
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]

    assert fit.valid
    assert fit.no_rl_signal_frac > 0.0, "the collapse has to be visible in the metrics"
    dead = fit.implied_beta[:n_dead]
    assert float(dead.abs().min()) > 0.0, "the teacher was switched off where it is the only source"
    live = fit.implied_beta[n_dead:][fit.keep[n_dead:]]
    assert float(dead.median()) > float(live.median()), (dead.median(), live.median())


def test_the_shrinkage_is_reported_but_never_multiplies_the_applied_weight():
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, seed=42)
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    want = fit.lam * fit.sigma2 / fit.varsigma2
    assert torch.allclose(fit.implied_beta[fit.keep], want[fit.keep], rtol=1e-9)
    assert fit.shrink is not None and float(fit.shrink[fit.keep].mean()) < 1.0


def test_lambda_carries_its_own_uncertainty_and_effective_sample_size():
    """A regression settled by ten format tokens must say so."""
    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, vocab=2000, seed=43)
    spread = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    # Now concentrate almost all of the product into a handful of ids.
    acc2 = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, vocab=2000, seed=43)
    for f in ("a", "a2", "ad", "d", "d2"):
        getattr(acc2, f)[0][:5] *= 300.0
    conc = fit_weights(acc2, ["t"], LogitPrecisionConfig())["t"]
    assert spread.n_eff > 100.0, spread.n_eff
    assert conc.n_eff < 20.0, conc.n_eff
    assert math.isfinite(spread.lam_se) and spread.lam_se > 0.0
    assert "logit_prec/t/n_eff_ids" in conc.metrics()
    assert "logit_prec/t/lam_t" in conc.metrics()


# ---------------------------------------------------------------------------
# 9. the apply path
# ---------------------------------------------------------------------------


def _kl_scalar(logits, topk_ids, t_lp, coef):
    s = torch.log_softmax(logits, dim=-1).gather(-1, topk_ids)
    return s, coef * topk_kl_per_token(s, t_lp).sum()


def test_a_unit_weight_reproduces_the_original_gradient_exactly():
    """The apply path has to be a no-op at omega = 1, or the arm is confounded."""
    from verl.trainer.ppo.opd_logit_precision import reweighted_opd_surrogate

    logits, topk_ids, t_lp = _mk(seed=50)
    z = logits.clone().requires_grad_(True)
    s, scalar = _kl_scalar(z, topk_ids, t_lp, 0.01)
    (want,) = torch.autograd.grad(scalar, z, retain_graph=True)

    sur = reweighted_opd_surrogate(scalar, s, torch.ones_like(s))
    (got,) = torch.autograd.grad(sur, z)
    assert torch.allclose(got, want, atol=1e-12), (got - want).abs().max()


def test_the_weight_lands_exactly_on_the_support_and_averages_the_background():
    from verl.trainer.ppo.opd_logit_precision import reweighted_opd_surrogate

    logits, topk_ids, t_lp = _mk(seed=51)
    z = logits.clone().requires_grad_(True)
    s, scalar = _kl_scalar(z, topk_ids, t_lp, 0.01)
    (gs,) = torch.autograd.grad(scalar, s, retain_graph=True)
    w = 0.3 + 2.0 * torch.rand(s.shape, generator=torch.Generator().manual_seed(3), dtype=s.dtype)

    sur = reweighted_opd_surrogate(scalar, s, w)
    (got,) = torch.autograd.grad(sur, z)

    pi = torch.softmax(logits, dim=-1)
    want = -pi * (w * gs).sum(-1, keepdim=True)
    want = want.scatter_add(-1, topk_ids, w * gs)
    assert torch.allclose(got, want, atol=1e-12), (got - want).abs().max()


def test_the_reweighted_cotangent_still_sums_to_zero():
    """Softmax ignores a uniform logit shift; the parameters do not."""
    from verl.trainer.ppo.opd_logit_precision import reweighted_opd_surrogate

    logits, topk_ids, t_lp = _mk(seed=52)
    z = logits.clone().requires_grad_(True)
    s, scalar = _kl_scalar(z, topk_ids, t_lp, 0.01)
    w = torch.rand(s.shape, generator=torch.Generator().manual_seed(4), dtype=s.dtype) * 5.0
    (got,) = torch.autograd.grad(reweighted_opd_surrogate(scalar, s, w), z)
    assert got.sum(-1).abs().max() < 1e-12


def test_the_gathered_ratio_is_relative_to_the_run_coefficient():
    from verl.trainer.ppo.opd_logit_precision import beta_ratio_at

    acc = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, vocab=200, seed=53)
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    ids = torch.tensor([[[0, 5, 9]]])
    got = beta_ratio_at(fit, ids, 1.0)
    assert torch.allclose(got.reshape(-1).double(), fit.implied_beta[[0, 5, 9]], rtol=1e-6)
    # A refused fit must leave the run at its own coefficient, never at zero.
    bad = TaskFit("t", False, "too_few_tokens")
    assert torch.allclose(beta_ratio_at(bad, ids, 0.01), torch.ones_like(ids, dtype=torch.float32))


# ---------------------------------------------------------------------------
# 10. the smoother must not invent a dead reward during warm-up
#     (the pilot's first step reported exactly that, from the first version)
# ---------------------------------------------------------------------------


def test_the_first_step_fit_is_not_scaled_down_by_the_smoother():
    """A bias-corrected EMA of one step must equal that step.

    The first implementation smoothed the RAW row sums. An EMA starting at zero
    scales them all by 1 - decay^t, and tau^2 = mean(a^2) - mean(sigma^2) is
    quadratic in that factor through the first term and linear through the
    second, so at step 1 (factor 0.2) it needed a reliability above 0.8 to come
    out positive. The live pilot duly reported tau2 = 0,
    no_rl_signal_frac = 1.0 and "rl_push_is_all_noise" for all three tasks.
    """
    cfg = LogitPrecisionConfig(min_eff_steps=1e9)   # pin the permutation path
    one = _synthetic(1.5, noise_rl=0.5, noise_teacher=0.3, seed=60)
    direct = fit_weights(one, ["t"], cfg, base_beta=1.0)["t"]

    ema = StepMoments(1, one.vocab, dtype=torch.float64)
    ema.ema_(one, 0.8)                       # exactly one step of smoothing
    smoothed = fit_weights(ema, ["t"], cfg, base_beta=1.0)["t"]

    assert smoothed.valid and smoothed.reason == direct.reason
    assert smoothed.tau2 == pytest.approx(direct.tau2, rel=1e-9)
    assert smoothed.lam == pytest.approx(direct.lam, rel=1e-9)
    assert smoothed.no_rl_signal_frac == pytest.approx(direct.no_rl_signal_frac)
    assert torch.allclose(smoothed.implied_beta[smoothed.keep],
                          direct.implied_beta[direct.keep], rtol=1e-9)


@pytest.mark.parametrize("steps", [1, 2, 5, 20])
def test_a_constant_stream_of_steps_gives_the_same_fit_at_every_step(steps):
    """Smoothing a repeated step must be a no-op, at step 1 and at step 20.

    Pinned to the permutation path (min_eff_steps above any reachable count),
    because that is the path whose degree mismatch made the warm-up scale the
    fit. The across-step variance of a repeated step is zero by construction,
    which is a different question.
    """
    cfg = LogitPrecisionConfig(min_eff_steps=1e9)
    one = _synthetic(1.0, noise_rl=0.5, noise_teacher=0.3, vocab=1500, seed=61)
    direct = fit_weights(one, ["t"], cfg, base_beta=1.0)["t"]
    ema = StepMoments(1, one.vocab, dtype=torch.float64)
    for _ in range(steps):
        ema.ema_(one, 0.8)
    got = fit_weights(ema, ["t"], cfg, base_beta=1.0)["t"]
    assert got.tau2 == pytest.approx(direct.tau2, rel=1e-9), steps
    assert got.lam == pytest.approx(direct.lam, rel=1e-9), steps


def test_the_permutation_variance_survives_the_trip_through_moments():
    """moments() must form sigma^2 from ONE step's sums, not from smoothed ones."""
    n_rows, vocab = 12, 9
    g = torch.Generator().manual_seed(62)
    m = torch.randn(n_rows, vocab, generator=g, dtype=torch.float64)
    adv = torch.randn(n_rows, generator=g, dtype=torch.float64)
    adv = adv - adv.mean()

    acc = StepAccumulator(1, vocab, dtype=torch.float64)
    acc.sm[0] = m.sum(0)
    acc.sm2[0] = (m * m).sum(0)
    acc.sa2[0] = (adv * adv).sum()
    acc.n_rows[0] = n_rows
    want = (acc.sm2[0] - acc.sm[0] ** 2 / n_rows) * (float(acc.sa2[0]) / (n_rows - 1))
    assert torch.allclose(acc.moments(LogitPrecisionConfig()).sig2[0], want, atol=1e-12)


# ---------------------------------------------------------------------------
# 11. the review's finding 2: sigma^2 must not collapse with the advantages
# ---------------------------------------------------------------------------


def test_the_permutation_null_collapses_with_the_advantages_and_the_new_one_does_not():
    """The defect, and the fix, side by side.

    sigma^2_perm[v] = spread[v] * sum_r A_r^2 / (N - 1): only spread depends on
    the id, and the scalar factor goes to zero with the advantages. So on a task
    where most rows carry no advantage -- webshop's 62 percent -- the measured
    coefficient shrinks uniformly, and the BLUE hands a push that is exactly
    zero an infinite precision. The across-step variance sees the advantage
    draw's own variability and does not.
    """
    perm = LogitPrecisionConfig(min_eff_steps=1e9)
    across = LogitPrecisionConfig()
    rows = []
    for live in (1.0, 0.5, 0.1, 0.03):
        mom = _stream(2.0, noise_rl=0.5, noise_teacher=0.3, seed=200, live_frac=live)
        f_perm = fit_weights(mom, ["t"], perm, base_beta=1.0)["t"]
        f_acr = fit_weights(mom, ["t"], across, base_beta=1.0)["t"]
        b_perm = float(f_perm.implied_beta[f_perm.keep].median()) if f_perm.implied_beta is not None else float("nan")
        b_acr = float(f_acr.implied_beta[f_acr.keep].median()) if f_acr.implied_beta is not None else float("nan")
        rows.append((live, b_perm, b_acr))

    full_perm, full_acr = rows[0][1], rows[0][2]
    # The permutation path tracks sum_r A_r^2 downward; the across-step one holds.
    assert rows[-1][1] < 0.2 * full_perm, rows
    assert rows[-1][2] > 0.5 * full_acr, rows


def test_the_measured_coefficient_is_no_longer_a_pure_function_of_the_spread():
    """Under the permutation null, implied_beta was EXACTLY proportional to it."""
    mom = _stream(2.0, noise_rl=0.5, noise_teacher=0.3, seed=201)
    perm = fit_weights(mom, ["t"], LogitPrecisionConfig(min_eff_steps=1e9), base_beta=1.0)["t"]
    r = perm.implied_beta[perm.keep] / mom.corrected().sig2[0][perm.keep]
    assert float(r.std() / r.mean().abs()) < 1e-12, "the old path should be exactly proportional"

    across = fit_weights(mom, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    r2 = across.implied_beta[across.keep] / mom.corrected().sig2[0][across.keep]
    assert float(r2.std() / r2.mean().abs()) > 1e-3, "the new path must not be"


def test_a_warming_up_fit_is_reported_but_never_applied():
    from verl.trainer.ppo.opd_logit_precision import beta_ratio_matrix

    one = _synthetic(1.5, noise_rl=0.5, noise_teacher=0.3, vocab=300, seed=202)
    fit = fit_weights(one, ["t"], LogitPrecisionConfig())["t"]
    assert fit.reason == "warming_up" and fit.valid
    assert fit.implied_beta is not None, "the pilot still needs to see it"
    assert not fit.sigma2_is_across_step
    assert fit.metrics()["logit_prec/t/reason"] == 6.0
    # but the apply path leaves the run at its own coefficient
    ratio = beta_ratio_matrix({"t": fit}, ["t"], 300, 0.01)
    assert torch.allclose(ratio, torch.ones_like(ratio))


def test_the_effective_step_count_reaches_the_ema_limit():
    """n_eff = w^2 / w2 must be 1 after one step and (1+d)/(1-d) in the limit."""
    one = _synthetic(1.0, 0.5, 0.3, vocab=64, seed=203)
    for decay, limit in ((0.8, 9.0), (0.9, 19.0)):
        ema = StepMoments(1, 64, dtype=torch.float64)
        ema.ema_(one, decay)
        assert ema.n_eff_steps() == pytest.approx(1.0, rel=1e-9), decay
        for _ in range(400):
            ema.ema_(one, decay)
        assert ema.n_eff_steps() == pytest.approx(limit, rel=1e-6), decay
        assert ema.corrected().n_eff_steps() == pytest.approx(limit, rel=1e-6), decay


# ---------------------------------------------------------------------------
# 10. the lm-head comparison: is the weight-space gradient more than H times a
#     common hidden direction?
# ---------------------------------------------------------------------------

HID = 6


def _lm_batch(bs=3, resp=4, seed=0, hidden=HID, n_tasks=2):
    g = torch.Generator().manual_seed(seed)
    logits, topk_ids, t_lp = _mk(bs=bs, resp=resp, seed=seed)
    s_lp = _student_topk_logprob(logits, topk_ids)
    mask = torch.ones(bs, resp, dtype=torch.float64)
    mask[-1, resp // 2:] = 0.0
    return dict(
        logits=logits, topk_ids=topk_ids, student_topk_logprob=s_lp, teacher_topk_logprob=t_lp,
        responses=torch.randint(0, V, (bs, resp), generator=g),
        response_mask=mask,
        task_ids=torch.tensor([i % n_tasks for i in range(bs)]),
        row_weight=torch.rand(bs, generator=g, dtype=torch.float64) + 0.2,
        pg_coef=torch.randn(bs, resp, generator=g, dtype=torch.float64),
        ratio=torch.rand(bs, resp, generator=g, dtype=torch.float64) + 0.5,
        row_advantage=torch.randn(bs, generator=g, dtype=torch.float64),
        hidden=torch.randn(bs, resp, hidden, generator=g, dtype=torch.float64),
    )


def _ids_and_slot(ids):
    slot = torch.full((V,), -1, dtype=torch.int64)
    slot[ids] = torch.arange(ids.numel(), dtype=torch.int64)
    return slot


def _run_lm(batch, ids, beta=0.01, n_tasks=2, chunk=2):
    acc = StepAccumulator(n_tasks, V, dtype=torch.float64)
    lm = LmHeadState(n_tasks, int(ids.numel()), batch["hidden"].shape[-1], dtype=torch.float64)
    acc.add_micro_batch(chunk_tokens=chunk, lm_state=lm, lm_ids=ids,
                        lm_slot_of_id=_ids_and_slot(ids), base_beta=beta, **batch)
    return acc, lm


def test_G_is_the_lm_head_gradient_summed_over_positions():
    """The claim the whole comparison rests on, against a position-by-position loop."""
    b = _lm_batch(seed=3)
    ids = torch.arange(0, V, 3, dtype=torch.int64)[:8]
    acc, lm = _run_lm(b, ids)

    sup, kap = topk_kl_logit_grad(b["student_topk_logprob"], b["teacher_topk_logprob"])
    pi = torch.softmax(b["logits"], dim=-1)
    bs, resp = b["response_mask"].shape
    G_rl = torch.zeros(2, ids.numel(), HID, dtype=torch.float64)
    G_opd = torch.zeros_like(G_rl)
    for r in range(bs):
        i = int(b["task_ids"][r])
        for t in range(resp):
            if b["response_mask"][r, t] == 0:
                continue
            rho = b["row_weight"][r]
            a_vec = (b["pg_coef"][r, t] * rho) * (
                torch.nn.functional.one_hot(b["responses"][r, t], V).double() - pi[r, t])
            d_vec = (kap[r, t] * rho) * pi[r, t]
            d_vec = d_vec.index_add(0, b["topk_ids"][r, t], sup[r, t] * rho)
            G_rl[i] += a_vec[ids][:, None] * b["hidden"][r, t][None, :]
            G_opd[i] += d_vec[ids][:, None] * b["hidden"][r, t][None, :]
    assert torch.allclose(lm.G_rl, G_rl, atol=1e-12), (lm.G_rl - G_rl).abs().max()
    assert torch.allclose(lm.G_opd, G_opd, atol=1e-12), (lm.G_opd - G_opd).abs().max()
    # and the id-space halves are the same sums with x replaced by 1
    assert torch.allclose(lm.G_rl.sum(-1) * 0 + lm.G_rl[..., 0] * 0, torch.zeros_like(lm.G_rl[..., 0]))


@pytest.mark.parametrize("chunk", [1, 2, 5, 64])
def test_the_lm_head_accumulation_does_not_depend_on_the_chunking(chunk):
    b = _lm_batch(seed=4)
    ids = torch.arange(0, V, 2, dtype=torch.int64)[:9]
    ref = _run_lm(b, ids, chunk=1)[1]
    got = _run_lm(b, ids, chunk=chunk)[1]
    assert torch.allclose(ref.G_rl, got.G_rl, atol=1e-12)
    assert torch.allclose(ref.G_opd, got.G_opd, atol=1e-12)


def test_a_constant_hidden_state_makes_G_exactly_rank_one():
    """The identity the decision rule is read against.

    With x_t the same at every position, G[v, :] = H[v] xbar exactly, so the
    fitted rank-one share must be 1. If this drifts, the metric is measuring
    the fit's conditioning rather than the model's hidden states.
    """
    b = _lm_batch(seed=5)
    b["hidden"] = torch.randn(HID, dtype=torch.float64).expand_as(b["hidden"]).contiguous()
    ids = torch.arange(0, V, 3, dtype=torch.int64)[:8]
    acc, lm = _run_lm(b, ids)
    m = lm_head_metrics(lm, ids, acc.a, acc.d, 0.01, ["t0", "t1"])
    for t in ("t0", "t1"):
        assert m[f"lm_head/rank1_share/{t}"] == pytest.approx(1.0, abs=1e-9)
        assert m[f"lm_head/rank1_share_xbar/{t}"] == pytest.approx(1.0, abs=1e-9)


def test_a_varying_hidden_state_is_detected_as_not_rank_one():
    """The other side: the metric must be able to come back BELOW 1, or it
    cannot answer the question it exists for."""
    b = _lm_batch(seed=6)
    ids = torch.arange(0, V, 3, dtype=torch.int64)[:8]
    acc, lm = _run_lm(b, ids)
    m = lm_head_metrics(lm, ids, acc.a, acc.d, 0.01, ["t0", "t1"])
    shares = [m[f"lm_head/rank1_share/{t}"] for t in ("t0", "t1")]
    assert all(0.0 <= x <= 1.0 + 1e-9 for x in shares), shares
    assert min(shares) < 0.95, shares
    # the xbar version fixes u instead of fitting it, so it can only be smaller
    for t in ("t0", "t1"):
        assert m[f"lm_head/rank1_share_xbar/{t}"] <= m[f"lm_head/rank1_share/{t}"] + 1e-9


def test_under_rank_one_the_pair_identity_holds_exactly():
    """cos(G_i, G_j) = cos(xbar_i, xbar_j) cos(H_i, H_j) when G really is rank one.

    This is the statement that decides the question. Built with a DIFFERENT
    constant hidden direction per task, so the xbar factor is not 1 and the
    identity is not trivially satisfied -- and checked in BOTH sign regimes,
    because a rank-one G still flips the pair's sign when the two tasks' mean
    hidden directions are opposed. "Weight space can only rescale the answer"
    is true of the magnitude and false of the sign.
    """
    b = _lm_batch(seed=7)
    g = torch.Generator().manual_seed(71)
    dirs = torch.randn(2, HID, generator=g, dtype=torch.float64)
    h = torch.zeros_like(b["hidden"])
    for r in range(h.shape[0]):
        h[r] = dirs[int(b["task_ids"][r])]
    b["hidden"] = h
    ids = torch.arange(0, V, 3, dtype=torch.int64)[:8]
    acc, lm = _run_lm(b, ids)
    m = lm_head_metrics(lm, ids, acc.a, acc.d, 0.01, ["t0", "t1"])
    tag = "t0__t1"
    assert m[f"lm_head/cos_G_over_pred/{tag}"] == pytest.approx(1.0, abs=1e-9)
    pred = m[f"lm_head/cos_xbar/{tag}"] * m[f"lm_head/cos_H/{tag}"]
    assert m[f"lm_head/cos_G/{tag}"] == pytest.approx(pred, abs=1e-9)
    # the sign follows cos_xbar, and both regimes are reachable
    assert m[f"lm_head/sign_agree_G_H/{tag}"] == float(m[f"lm_head/cos_xbar/{tag}"] > 0)
    b2 = _lm_batch(seed=7)
    h2 = torch.zeros_like(b2["hidden"])
    flip = torch.stack([dirs[0], -dirs[0] * 1.0]) if m[f"lm_head/cos_xbar/{tag}"] > 0 else torch.stack([dirs[0], dirs[0] * 1.0])
    for r in range(h2.shape[0]):
        h2[r] = flip[int(b2["task_ids"][r])]
    b2["hidden"] = h2
    acc2, lm2 = _run_lm(b2, ids)
    m2 = lm_head_metrics(lm2, ids, acc2.a, acc2.d, 0.01, ["t0", "t1"])
    assert m2[f"lm_head/cos_G_over_pred/{tag}"] == pytest.approx(1.0, abs=1e-9)
    assert m2[f"lm_head/sign_agree_G_H/{tag}"] != m[f"lm_head/sign_agree_G_H/{tag}"], (
        "the two hidden-direction regimes must disagree on the pair's sign")


def test_ids_outside_the_kept_set_contribute_nothing():
    b = _lm_batch(seed=8)
    ids = torch.arange(0, V, 3, dtype=torch.int64)[:8]
    _, ref = _run_lm(b, ids)
    # a second run whose id set is the same prefix plus one extra id must agree
    ids2 = torch.cat([ids, torch.tensor([int(x) for x in range(V) if x not in set(ids.tolist())][:1])]).sort().values
    _, got = _run_lm(b, ids2)
    keep = torch.tensor([int((ids2 == v).nonzero()[0]) for v in ids])
    assert torch.allclose(ref.G_rl, got.G_rl[:, keep], atol=1e-12)
    assert torch.allclose(ref.G_opd, got.G_opd[:, keep], atol=1e-12)


def test_select_lm_ids_takes_the_pooled_energy_and_refuses_an_empty_state():
    a = torch.zeros(2, V, dtype=torch.float64)
    d = torch.zeros_like(a)
    assert select_lm_ids(a, d, 0.01, 4).numel() == 0, "a zero state must not rank anything"
    a[0, 5] = 3.0
    a[1, 9] = 2.0
    d[0, 11] = 400.0            # beta = 0.01 -> contributes 4.0
    got = set(select_lm_ids(a, d, 0.01, 3).tolist())
    assert got == {5, 9, 11}, got
    assert select_lm_ids(a, d, 10 * V, 10 * V).numel() == V, "m above the vocabulary must clamp"


def test_the_lm_head_state_round_trips_and_refuses_a_reshaped_run():
    lm = LmHeadState(2, 5, HID, dtype=torch.float64)
    lm.G_rl.normal_(); lm.G_opd.normal_(); lm.xsum.normal_(); lm.n_tok.fill_(3.0); lm.w.fill_(1.0)
    other = LmHeadState(2, 5, HID, dtype=torch.float64).load_state_dict(lm.state_dict())
    assert torch.equal(other.G_rl, lm.G_rl) and torch.equal(other.xsum, lm.xsum)
    with pytest.raises(ValueError, match="n_ids"):
        LmHeadState(2, 6, HID, dtype=torch.float64).load_state_dict(lm.state_dict())


def test_the_measurement_is_off_by_default_and_refuses_a_negative_width():
    assert LogitPrecisionConfig().lm_head_topm == 0
    LogitPrecisionConfig(lm_head_topm=512).validate()
    with pytest.raises(ValueError, match="lm_head_topm"):
        LogitPrecisionConfig(lm_head_topm=-1).validate()


def test_no_ids_means_no_metrics_rather_than_a_guess():
    b = _lm_batch(seed=9)
    ids = torch.zeros(0, dtype=torch.int64)
    acc = StepAccumulator(2, V, dtype=torch.float64)
    lm = LmHeadState(2, 0, HID, dtype=torch.float64)
    acc.add_micro_batch(chunk_tokens=2, lm_state=lm, lm_ids=ids,
                        lm_slot_of_id=_ids_and_slot(torch.zeros(0, dtype=torch.int64)),
                        base_beta=0.01, **b)
    m = lm_head_metrics(lm, ids, acc.a, acc.d, 0.01, ["t0", "t1"])
    assert m == {"lm_head/n_ids": 0.0}


def test_the_lm_head_hook_finds_the_module_by_leaf_name_and_reads_its_input():
    """FSDP wraps the root, so the attribute path is unreliable while the leaf
    name survives. The hook must also fire only while the step asks for it."""
    import torch.nn as nn

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HID, V, bias=False)

        def forward(self, x):
            return self.lm_head(x)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self._fsdp_wrapped_module = Inner()

        def forward(self, x):
            return self._fsdp_wrapped_module(x)

    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = Wrapped()
    assert actor._lp_install_lm_head_hook() is True
    assert actor._lp_install_lm_head_hook() is True, "installing twice must be a no-op"

    x = torch.randn(1, 4, HID)
    actor._lp_want_logits = False
    actor.actor_module(x)
    assert actor._lp_take_hidden() is None, "the hook must not stash outside the step"

    actor._lp_want_logits = True
    actor.actor_module(x)
    got = actor._lp_take_hidden()
    assert got is not None and got.shape == (4, HID)
    assert torch.equal(got, x.squeeze(0))
    assert actor._lp_take_hidden() is None, "the stash is consumed, not left behind"


def test_a_module_tree_without_an_lm_head_skips_rather_than_raises():
    import torch.nn as nn

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = nn.Sequential(nn.Linear(HID, HID))
    assert actor._lp_install_lm_head_hook() is False
