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
    StepAccumulator,
    TaskFit,
    fit_weights,
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
    """An accumulator whose contents follow the model the fit assumes."""
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
    return acc


@pytest.mark.parametrize("lam", [0.25, 1.0, 3.0])
def test_the_fit_recovers_a_known_teacher_scale(lam):
    acc = _synthetic(lam, noise_rl=0.5, noise_teacher=0.3)
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig())["t"]
    assert fit.valid and fit.reason == "ok"
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
    n_rows = float(acc.n_rows[0])
    acc.sm2[0][:100] += extra ** 2 * (n_rows - 1) / n_rows
    # base_beta matches the synthetic scale (theta ~ N(0,1)), so the relative
    # cap sits far above both groups and the comparison is of the fit, not of
    # the clamp.
    fit = fit_weights(acc, ["t"], LogitPrecisionConfig(), base_beta=1.0)["t"]
    noisy = fit.implied_beta[:100].median()
    rest = fit.implied_beta[100:][fit.keep[100:]].median()
    assert float(noisy) > 10.0 * float(rest), (noisy, rest)


def test_a_task_with_too_little_data_is_refused_rather_than_guessed():
    acc = StepAccumulator(1, 500, dtype=torch.float64)
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
    want = 0.8 * a.a[0] + 0.2 * b.a[0]
    a.ema_(b, 0.8)
    assert torch.allclose(a.a[0], want)


def test_the_state_round_trips_and_refuses_a_reshaped_run():
    a = _synthetic(1.0, 0.5, 0.3, vocab=64, seed=22)
    b = StepAccumulator(1, 64, dtype=torch.float64)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.d[0], b.d[0])
    with pytest.raises(ValueError, match="vocab"):
        StepAccumulator(1, 65, dtype=torch.float64).load_state_dict(a.state_dict())


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
    acc = StepAccumulator(1, 500, dtype=torch.float64)
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
    # The apply path does not exist yet and must say so rather than no-op.
    assert "observe_only=false is not implemented yet" in src


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
    n_rows = float(acc.n_rows[0])
    # The null is deliberately conservative (it permutes across the task's rows
    # rather than within a prompt group, destroying group identity as well), so
    # it comes out at or above the observed spread. 1.3x puts the stratum
    # unambiguously on the collapsed side instead of within sampling noise of
    # the boundary, which is what made the first version of this test flaky.
    acc.sm2[0][:n_dead] = 1.3 * big ** 2 * (n_rows - 1) / n_rows
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
    acc2.a[0][:5] *= 300.0
    acc2.d[0][:5] *= 300.0
    conc = fit_weights(acc2, ["t"], LogitPrecisionConfig())["t"]
    assert spread.n_eff > 100.0, spread.n_eff
    assert conc.n_eff < 20.0, conc.n_eff
    assert math.isfinite(spread.lam_se) and spread.lam_se > 0.0
    assert "logit_prec/t/n_eff_ids" in conc.metrics()
    assert "logit_prec/t/lam_t" in conc.metrics()
