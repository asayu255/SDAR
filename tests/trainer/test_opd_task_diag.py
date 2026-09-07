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

"""The OPD arm's readout, checked against autograd rather than against itself.

The numbers this module produces are the ones the arm will be read by, so the
central test is not "the code runs" but "the inner product it reports is the
inner product of the two gradients the loss actually takes".
"""

import ast
import inspect
import math

import pytest
import torch

from verl.trainer.ppo.core_algos import policy_loss_gradient_coef, topk_kl_per_token
from verl.trainer.ppo.opd_task_diag import OpdTaskDiagStats, opd_pg_alignment_terms


def _case(vocab=13, k=5, seed=0, peaked=False):
    """One token, one sampled id, a teacher top-k support, and full logits."""
    g = torch.Generator().manual_seed(seed)
    scale = 6.0 if peaked else 1.0
    logits = torch.randn(vocab, generator=g, dtype=torch.float64) * scale
    t_logits = torch.randn(vocab, generator=g, dtype=torch.float64) * scale
    t_lp_full = torch.log_softmax(t_logits, dim=-1)
    topk_ids = torch.topk(t_lp_full, k).indices
    # The sampled token is inside the support: that is the population the
    # alignment metric is defined on, and align_cover reports the rest.
    sampled = int(topk_ids[k // 2])
    return logits, t_lp_full, topk_ids, sampled


def _autograd_pushes(logits, t_lp_full, topk_ids, sampled, adv, old_lp):
    """The two descent directions on the FULL logit vector, from autograd."""
    z = logits.clone().requires_grad_(True)
    lp_full = torch.log_softmax(z, dim=-1)
    s_topk = lp_full[topk_ids].reshape(1, 1, -1)
    t_topk = t_lp_full[topk_ids].reshape(1, 1, -1)

    kl = topk_kl_per_token(student_topk_logprob=s_topk, teacher_topk_logprob=t_topk).reshape(())
    (g_kl,) = torch.autograd.grad(kl, z, retain_graph=True)

    ratio = (lp_full[sampled] - old_lp).exp()
    pg = -adv * ratio
    (g_pg,) = torch.autograd.grad(pg, z)
    # Descent convention: positive means the objective pushes this logit up.
    return -g_kl, -g_pg, kl.detach()


def _pg_coef(logits, sampled, adv, old_lp, cliprange=None):
    """dL_pg/dlog p, from the same closed form dp_actor hands to the metric."""
    lp = torch.log_softmax(logits, dim=-1)[sampled].reshape(1, 1)
    return policy_loss_gradient_coef(
        old_log_prob=torch.tensor([[old_lp]], dtype=torch.float64),
        log_prob=lp,
        advantages=torch.tensor([[adv]], dtype=torch.float64),
        cliprange=cliprange if cliprange is not None else 1e9,
        clip_ratio_c=1e9 if cliprange is None else 3.0,
    )


def _terms_for(logits, t_lp_full, topk_ids, sampled, adv, old_lp, cliprange=None):
    lp_full = torch.log_softmax(logits, dim=-1)
    s_topk = lp_full[topk_ids].reshape(1, 1, -1)
    t_topk = t_lp_full[topk_ids].reshape(1, 1, -1)
    kl = topk_kl_per_token(student_topk_logprob=s_topk, teacher_topk_logprob=t_topk)
    return opd_pg_alignment_terms(
        student_topk_logprob=s_topk,
        teacher_topk_logprob=t_topk,
        teacher_kl=kl,
        topk_ids=topk_ids.reshape(1, 1, -1),
        response_ids=torch.tensor([[sampled]]),
        log_prob=lp_full[sampled].reshape(1, 1),
        pg_grad_coef=_pg_coef(logits, sampled, adv, old_lp, cliprange),
    )


# ---------------------------------------------------------------------------
# 1. the formula, against autograd


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_opd_push_matches_autograd_on_the_support(seed):
    """g_opd(v) = p(v) (D - f(v)) is exact for the top-k+tail KL, not a fit.

    The tail bucket's dependence on the support logits (tail = 1 - sum) cancels
    exactly; if it did not, this would be off by p(v) * tail_s.
    """
    logits, t_lp, ids, sampled = _case(seed=seed)
    g_kl, _, kl = _autograd_pushes(logits, t_lp, ids, sampled, adv=1.0, old_lp=-2.0)

    lp_full = torch.log_softmax(logits, dim=-1)
    p = lp_full.exp()
    f = lp_full[ids] - t_lp[ids]
    analytic_on_support = p[ids] * (kl - f)
    assert torch.allclose(analytic_on_support, g_kl[ids], atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_reported_dot_is_the_true_inner_product_minus_the_tail_logits(seed):
    """The one identity the metric's honesty rests on.

    <g_pg, g_opd> over the FULL vocabulary is A*rho*[g_opd(a) - sum_v p(v) g_opd(v)].
    The metric runs that sum over the top-k support only, so it differs from the
    truth by exactly A*rho*sum_{v not in S} p(v)^2 (D - f_tail) -- computed here
    and asserted, rather than waved at as "small".
    """
    logits, t_lp, ids, sampled = _case(seed=seed)
    adv, old_lp = 0.7, -2.0
    g_kl, g_pg, kl = _autograd_pushes(logits, t_lp, ids, sampled, adv, old_lp)
    terms = _terms_for(logits, t_lp, ids, sampled, adv, old_lp)

    lp_full = torch.log_softmax(logits, dim=-1)
    p = lp_full.exp()
    off = torch.ones(len(p), dtype=torch.bool)
    off[ids] = False
    f_tail = (1 - p[ids].sum()).log() - (1 - t_lp[ids].exp().sum()).log()
    ratio = float((lp_full[sampled] - old_lp).exp())
    dropped = adv * ratio * float((p[off] ** 2 * (kl - f_tail)).sum())

    truth_full = float((g_kl * g_pg).sum())
    got = float(terms["dot"].reshape(()))
    assert got == pytest.approx(truth_full + dropped, abs=1e-10)


def test_the_tail_approximation_vanishes_as_the_off_support_mass_does():
    """The dropped term carries p(v)^2 off the support, so it is not "small"
    by assertion -- it shrinks with the off-support mass, and this pins the rate.

    A synthetic 200-token vocabulary with a flat-ish policy leaves 87% of the
    mass outside a top-20 support, which no trained policy does; the sequence
    below walks from that to a realistic one and reports what the metric costs
    at each point.
    """
    errs = []
    for scale in (6.0, 10.0, 16.0):
        g = torch.Generator().manual_seed(7)
        vocab, k = 200, 20
        logits = torch.randn(vocab, generator=g, dtype=torch.float64) * scale
        t_logits = torch.randn(vocab, generator=g, dtype=torch.float64) * scale
        t_lp = torch.log_softmax(t_logits, dim=-1)
        ids = torch.topk(t_lp, k).indices
        sampled = int(ids[k // 2])
        lp_full = torch.log_softmax(logits, dim=-1)
        old_lp = float(lp_full[sampled]) - 0.05

        g_kl, g_pg, _ = _autograd_pushes(logits, t_lp, ids, sampled, 1.0, old_lp)
        terms = _terms_for(logits, t_lp, ids, sampled, 1.0, old_lp)
        truth = float((g_kl * g_pg).sum())
        got = float(terms["dot"].reshape(()))
        errs.append(abs(got - truth) / max(abs(truth), 1e-30))

    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-5, f"off-support share should be negligible on a peaked policy, got {errs}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_reported_norms_are_the_support_truncation_of_the_true_ones(seed):
    logits, t_lp, ids, sampled = _case(seed=seed)
    adv, old_lp = -0.4, -1.5
    g_kl, g_pg, _ = _autograd_pushes(logits, t_lp, ids, sampled, adv, old_lp)
    terms = _terms_for(logits, t_lp, ids, sampled, adv, old_lp)

    # OPD: exactly the support part of the true squared norm.
    assert float(terms["opd_sq"].reshape(())) == pytest.approx(float((g_kl[ids] ** 2).sum()), rel=1e-12)

    # PG: ||g_pg||^2 = (A rho)^2 (1 - 2 p_a + sum_v p^2). The metric truncates
    # only that sum of squares, so the gap to autograd is the tail's share of it.
    lp_full = torch.log_softmax(logits, dim=-1)
    p = lp_full.exp()
    off = torch.ones(len(p), dtype=torch.bool)
    off[ids] = False
    scale2 = float((adv * (lp_full[sampled] - old_lp).exp()) ** 2)
    truth_full = float((g_pg ** 2).sum())
    assert float(terms["pg_sq"].reshape(())) == pytest.approx(
        truth_full - scale2 * float((p[off] ** 2).sum()), rel=1e-12
    )


# ---------------------------------------------------------------------------
# 2. what the sign means


def test_a_teacher_pulling_against_a_positive_advantage_reports_a_conflict():
    """Reward wants the sampled token up; teacher wants it down -> dot < 0.

    Full support (k = vocab), so the tail is empty and the reading is exact: a
    uniform student makes sum_v p(v) g_opd(v) vanish identically, and the sign
    of the dot is the sign of the teacher's push on the sampled logit alone.
    """
    vocab = 6
    logits = torch.zeros(vocab, dtype=torch.float64)  # uniform student
    sampled = 0
    # Same as the student except that the sampled token's mass is moved onto
    # another one: the teacher is pulling exactly the token the reward wants.
    p_t = torch.tensor([0.12, 0.30, 0.145, 0.145, 0.145, 0.145], dtype=torch.float64)
    assert float(p_t.sum()) == pytest.approx(1.0)
    t_lp = p_t.log()
    ids = torch.topk(t_lp, vocab).indices
    assert sampled in ids.tolist()
    old_lp = float(torch.log_softmax(logits, -1)[sampled])

    conflict = _terms_for(logits, t_lp, ids, sampled, adv=1.0, old_lp=old_lp)
    assert float(conflict["dot"].reshape(())) < 0
    assert float(conflict["cos"].reshape(())) < 0
    assert -1.0 <= float(conflict["cos"].reshape(())) <= 1.0

    # Same teacher, same state, NEGATIVE advantage: the reward now wants the
    # sampled token down too, and the two agree.
    agree = _terms_for(logits, t_lp, ids, sampled, adv=-1.0, old_lp=old_lp)
    assert float(agree["dot"].reshape(())) > 0


def test_a_zero_advantage_token_carries_no_alignment():
    logits, t_lp, ids, sampled = _case(seed=5)
    terms = _terms_for(logits, t_lp, ids, sampled, adv=0.0, old_lp=-2.0)
    assert float(terms["pg_live"].reshape(())) == 0.0
    assert float(terms["align_mask"].reshape(())) == 0.0
    assert float(terms["dot"].reshape(())) == 0.0
    # but the budget column is still defined there -- that is the whole point of
    # the adv-zero split.
    assert float(terms["opd_sq"].reshape(())) > 0


def test_a_sampled_token_outside_the_support_is_excluded_not_guessed():
    vocab, k = 8, 3
    logits = torch.zeros(vocab, dtype=torch.float64)
    t_lp = torch.log_softmax(torch.arange(vocab, dtype=torch.float64), dim=-1)
    ids = torch.topk(t_lp, k).indices
    outside = int((set(range(vocab)) - set(ids.tolist())).pop())
    terms = _terms_for(logits, t_lp, ids, outside, adv=1.0, old_lp=-2.0)
    assert float(terms["align_mask"].reshape(())) == 0.0
    assert float(terms["dot"].reshape(())) == 0.0


def test_missing_advantages_do_not_fabricate_an_orthogonal_reading():
    logits, t_lp, ids, sampled = _case(seed=6)
    lp_full = torch.log_softmax(logits, dim=-1)
    terms = opd_pg_alignment_terms(
        student_topk_logprob=lp_full[ids].reshape(1, 1, -1),
        teacher_topk_logprob=t_lp[ids].reshape(1, 1, -1),
        teacher_kl=torch.tensor([[0.3]], dtype=torch.float64),
        topk_ids=ids.reshape(1, 1, -1),
        response_ids=torch.tensor([[sampled]]),
        log_prob=lp_full[sampled].reshape(1, 1),
        pg_grad_coef=None,
    )
    assert float(terms["align_mask"].reshape(())) == 0.0
    assert float(terms["opd_sq"].reshape(())) > 0


# ---------------------------------------------------------------------------
# 3. the table


def _stats_with(rows, names=("alfworld", "search", "webshop"), terms=None):
    """rows: list of (task_id, kl_per_token, adv_per_token, basis, coef)."""
    n = len(rows)
    t = max(2, max((len(r[1]) for r in rows), default=2))
    task_ids = torch.tensor([r[0] for r in rows])
    kl = torch.zeros(n, t, dtype=torch.float32)
    adv = torch.zeros(n, t, dtype=torch.float32)
    mask = torch.zeros(n, t, dtype=torch.float32)
    basis = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    coef = torch.tensor([r[4] for r in rows], dtype=torch.float32)
    for i, r in enumerate(rows):
        kl[i, : len(r[1])] = torch.tensor(r[1], dtype=torch.float32)
        adv[i, : len(r[2])] = torch.tensor(r[2], dtype=torch.float32)
        mask[i, : len(r[1])] = 1.0
    if terms == "flat":
        terms = {
            "opd_sq": torch.ones(n, t),
            "dot": torch.zeros(n, t),
            "cos": torch.zeros(n, t),
            "pg_sq": torch.zeros(n, t),
            "align_mask": torch.zeros(n, t),
            "pg_live": torch.zeros(n, t),
        }
    st = OpdTaskDiagStats(n_tasks=len(names), device=torch.device("cpu"))
    st.update(task_ids=task_ids, response_mask=mask, teacher_kl=kl, advantages=adv,
              terms=terms, row_basis=basis, row_coef=coef)
    return st.rows(list(names))


def test_the_shares_are_formed_from_pooled_sums_not_from_per_row_means():
    out = _stats_with([
        (0, [1.0, 1.0], [1.0, 1.0], 1.0, 2.0),
        (1, [1.0, 1.0], [1.0, 1.0], 1.0, 1.0),
    ])
    assert out["actor/opd_diag/kl_share_base/alfworld"] == pytest.approx(0.5)
    assert out["actor/opd_diag/kl_share_eff/alfworld"] == pytest.approx(2 / 3)
    assert out["actor/opd_diag/kl_share_eff/search"] == pytest.approx(1 / 3)


def test_the_share_is_not_the_coefficient_and_the_ratio_is():
    """The check that b landed has to be eff/base per task, not the share.

    A uniform amplification moves every share by nothing at all, so a reader who
    took kl_share_eff for b would see 1/3 on an arm that multiplied everything
    by 1.111 and conclude the coefficient never arrived.
    """
    uniform = _stats_with([
        (0, [1.0, 1.0], [1.0, 1.0], 1.0, 1.111),
        (1, [1.0, 1.0], [1.0, 1.0], 1.0, 1.111),
        (2, [1.0, 1.0], [1.0, 1.0], 1.0, 1.111),
    ])
    for task in ("alfworld", "search", "webshop"):
        assert uniform[f"actor/opd_diag/kl_share_eff/{task}"] == pytest.approx(1 / 3)
        assert uniform[f"actor/opd_diag/kl_share_base/{task}"] == pytest.approx(1 / 3)
        assert uniform[f"actor/opd_diag/kl_ratio_eff_base/{task}"] == pytest.approx(1.111)


def test_the_ratios_equal_b_even_when_the_row_weights_differ():
    """eff/base is b_task whatever the per-task loss weight is: both sums carry
    the same basis, so it divides out. That is what makes it a wiring check."""
    out = _stats_with([
        (0, [2.0, 4.0], [1.0, 1.0], 0.37, 1.076431),
        (1, [1.0, 1.0], [1.0, 1.0], 5.10, 1.191101),
        (2, [3.0, 3.0], [1.0, 1.0], 0.02, 0.5),
    ], terms="flat")
    for task, b in (("alfworld", 1.076431), ("search", 1.191101), ("webshop", 0.5)):
        assert out[f"actor/opd_diag/kl_ratio_eff_base/{task}"] == pytest.approx(b, rel=1e-6)
        assert out[f"actor/opd_diag/push_l2_ratio_eff_base/{task}"] == pytest.approx(b, rel=1e-6)


def test_the_gradient_norm_carries_the_loss_row_weight():
    """The bug this pins: the KL sums had the row weight and the norms did not.

    Same token gradients, same b, only the per-task loss weight changed. If the
    norm ignores it, the budget ratio comes out of a term the loss never took.
    """
    flat = _stats_with([
        (0, [1.0], [1.0], 1.0, 1.0),
        (1, [1.0], [1.0], 1.0, 1.0),
        (2, [1.0], [1.0], 1.0, 0.5),
    ], terms="flat")
    skewed = _stats_with([
        (0, [1.0], [1.0], 0.1, 1.0),
        (1, [1.0], [1.0], 1.0, 1.0),
        (2, [1.0], [1.0], 4.0, 0.5),
    ], terms="flat")
    assert flat["actor/opd_diag/budget_ratio_logit"] != pytest.approx(
        skewed["actor/opd_diag/budget_ratio_logit"]
    ), "the row weight must reach the norm, or the ratio cannot tell these apart"
    # and each task's own norm scales with its weight
    assert skewed["actor/opd_diag/push_l2_logit_base/webshop"] == pytest.approx(
        40 * skewed["actor/opd_diag/push_l2_logit_base/alfworld"], rel=1e-5
    )


def test_the_advantage_split_and_the_live_pg_share_are_different_numbers():
    out = _stats_with([
        (2, [1.0, 3.0, 5.0, 7.0], [0.0, 0.0, 0.5, -0.5], 1.0, 0.5),
    ], terms="flat")
    assert out["actor/opd_diag/adv_zero_frac/webshop"] == pytest.approx(0.5)
    assert out["actor/opd_diag/kl_mean_adv_zero/webshop"] == pytest.approx(2.0)
    assert out["actor/opd_diag/kl_mean_adv_live/webshop"] == pytest.approx(6.0)
    # the flat terms carry pg_live = 0 everywhere, so a live advantage does NOT
    # imply a live policy gradient -- which is the whole point of reporting both
    assert out["actor/opd_diag/pg_live_frac/webshop"] == pytest.approx(0.0)


def test_padding_rows_land_nowhere():
    out = _stats_with([
        (0, [2.0], [1.0], 1.0, 1.0),
        (-1, [100.0], [1.0], 1.0, 1.0),
    ])
    assert out["actor/opd_diag/tokens/alfworld"] == pytest.approx(1.0)
    assert out["actor/opd_diag/kl_mean/alfworld"] == pytest.approx(2.0)
    assert "actor/opd_diag/tokens/search" not in out


def test_the_budget_ratio_is_b_under_a_uniform_arm_and_a_mix_otherwise():
    """It is sum_j b_j L_j / sum_j L_j in logit space -- NOT the parameter-space
    budget the offline rule set to 1.000, which is why nothing here asserts 1."""
    uni = _stats_with([
        (0, [1.0], [1.0], 1.0, 1.110833),
        (1, [1.0], [1.0], 1.0, 1.110833),
        (2, [1.0], [1.0], 1.0, 1.110833),
    ], terms="flat")
    assert uni["actor/opd_diag/budget_ratio_logit"] == pytest.approx(1.110833, rel=1e-6)

    red = _stats_with([
        (0, [1.0], [1.0], 1.0, 1.076431),
        (1, [1.0], [1.0], 1.0, 1.191101),
        (2, [1.0], [1.0], 1.0, 0.5),
    ], terms="flat")
    # equal norms here, so the ratio is the plain mean of b -- not 1
    assert red["actor/opd_diag/budget_ratio_logit"] == pytest.approx(
        (1.076431 + 1.191101 + 0.5) / 3, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 4. the wiring, and the fact that the flag off changes nothing


def test_the_readout_is_off_by_default_and_gated_on_the_config_alone():
    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    src = ast.unparse(fn)
    assert "self.config.get('teacher_kl_task_diag', False)" in src
    # config-only, because rows() runs a collective
    assert "opd_diag_stats = OpdTaskDiagStats(n_tasks=n_task, device=sign_dev) if" in src
    assert "data.batch" not in src.split("opd_diag_stats = ")[1].split("else None")[0]
    # and it is read once, after the backward, not inside the loss
    assert src.index("opd_diag_stats.update(") > src.index("loss.backward()")
    assert "opd_diag_stats.rows(" in src


def test_the_readout_never_enters_the_loss():
    src = inspect.getsource(opd_pg_alignment_terms)
    assert ".detach()" in src
    tree = ast.parse(inspect.getsource(OpdTaskDiagStats))
    assert any(
        isinstance(n, ast.withitem) and "no_grad" in ast.unparse(n.context_expr)
        for n in ast.walk(tree)
    ), "the accumulator must run under no_grad"


def test_the_table_runs_on_batch_shaped_terms_not_only_on_none():
    """The terms branch of update() -- the one the run actually takes.

    _stats_with above passes terms=None, so without this the (rows, T, k)
    shapes never meet the accumulator until a GPU is holding them.
    """
    torch.manual_seed(0)
    rows, t, k, vocab = 5, 7, 6, 40
    student_topk = torch.log_softmax(torch.randn(rows, t, k), dim=-1)
    teacher_topk = torch.log_softmax(torch.randn(rows, t, k), dim=-1)
    topk_ids = torch.stack([
        torch.stack([torch.randperm(vocab)[:k] for _ in range(t)]) for _ in range(rows)
    ])
    # half the sampled ids inside the support, half outside
    response_ids = topk_ids[..., 0].clone()
    response_ids[:, ::2] = vocab + 1
    kl = torch.rand(rows, t) * 0.1
    log_prob = torch.log_softmax(torch.randn(rows, vocab + 2), dim=-1)[:, :1].expand(rows, t).contiguous()
    terms = opd_pg_alignment_terms(
        student_topk_logprob=student_topk,
        teacher_topk_logprob=teacher_topk,
        teacher_kl=kl,
        topk_ids=topk_ids,
        response_ids=response_ids,
        log_prob=log_prob,
        pg_grad_coef=policy_loss_gradient_coef(
            old_log_prob=log_prob - 0.05,
            log_prob=log_prob,
            advantages=torch.where(torch.rand(rows, t) < 0.4, 0.0, 1.0),
            cliprange=0.2,
        ),
    )
    for key in ("opd_sq", "dot", "cos", "pg_sq", "align_mask"):
        assert terms[key].shape == (rows, t), key

    mask = torch.ones(rows, t)
    mask[:, -2:] = 0.0  # padding at the end of every row
    s = OpdTaskDiagStats(n_tasks=3, device=torch.device("cpu"))
    s.update(
        task_ids=torch.tensor([0, 1, 2, 0, -1]),
        response_mask=mask,
        teacher_kl=kl,
        advantages=torch.where(torch.rand(rows, t) < 0.4, 0.0, 1.0),
        terms=terms,
        row_basis=torch.ones(rows),
        row_coef=torch.tensor([1.076431, 1.191101, 0.5, 1.076431, 1.0]),
    )
    out = s.rows(["alfworld", "search", "webshop"])
    assert out["actor/opd_diag/tokens/alfworld"] == pytest.approx(2 * (t - 2))
    for key, value in out.items():
        assert value == value, key                       # no NaN reaches the logger
        assert abs(value) != float("inf"), key
    for task in ("alfworld", "search", "webshop"):
        assert 0.0 <= out[f"actor/opd_diag/align_cover/{task}"] <= 1.0
        assert -1.0 <= out[f"actor/opd_diag/pg_cos_mean/{task}"] <= 1.0
        assert out[f"actor/opd_diag/push_l2_logit_base/{task}"] > 0
        assert out[f"actor/opd_diag/push_l2_logit_eff/{task}"] > 0
    assert out["actor/opd_diag/budget_ratio_logit"] > 0


# ---------------------------------------------------------------------------
# 5. the PPO clip. -A*rho reports a position the objective has stopped pushing
#    as a full-magnitude conflict; the closed-form derivative does not.


def _conflict_state():
    """Uniform student, full support, teacher pulling the sampled token down."""
    vocab = 6
    logits = torch.zeros(vocab, dtype=torch.float64)
    p_t = torch.tensor([0.12, 0.30, 0.145, 0.145, 0.145, 0.145], dtype=torch.float64)
    t_lp = p_t.log()
    ids = torch.topk(t_lp, vocab).indices
    return logits, t_lp, ids, 0


def test_a_token_whose_clip_binds_reports_no_conflict_at_all():
    logits, t_lp, ids, sampled = _conflict_state()
    lp = float(torch.log_softmax(logits, -1)[sampled])
    # ratio 1.5 with a positive advantage: the clamped branch is selected and the
    # true derivative is exactly zero.
    old_lp = lp - math.log(1.5)
    coef = _pg_coef(logits, sampled, adv=1.0, old_lp=old_lp, cliprange=0.2)
    assert float(coef.reshape(())) == 0.0, "fixture must actually bind the clip"

    clipped = _terms_for(logits, t_lp, ids, sampled, adv=1.0, old_lp=old_lp, cliprange=0.2)
    assert float(clipped["pg_live"].reshape(())) == 0.0
    assert float(clipped["align_mask"].reshape(())) == 0.0
    assert float(clipped["dot"].reshape(())) == 0.0
    assert float(clipped["cos"].reshape(())) == 0.0

    # What the old -A*rho reading would have said about the SAME position: a
    # conflict at full magnitude, and a cosine at the floor.
    p = torch.log_softmax(logits, -1).exp()
    kl = float(topk_kl_per_token(
        student_topk_logprob=torch.log_softmax(logits, -1)[ids].reshape(1, 1, -1),
        teacher_topk_logprob=t_lp[ids].reshape(1, 1, -1)).reshape(()))
    g_opd = p * (kl - (torch.log_softmax(logits, -1) - t_lp))
    naive_dot = 1.0 * 1.5 * float(g_opd[sampled] - (p * g_opd).sum())
    # Sign and non-triviality are the point: the truth is exactly 0 and the old
    # reading is a definite conflict, whatever the state's magnitude happens to be.
    assert naive_dot < -1e-3, "the old reading really did call this a conflict"

    # and with the clip out of the way the metric DOES see it, so the difference
    # is the clip and not the state.
    unclipped = _terms_for(logits, t_lp, ids, sampled, adv=1.0, old_lp=old_lp)
    assert float(unclipped["align_mask"].reshape(())) == 1.0
    assert float(unclipped["dot"].reshape(())) == pytest.approx(naive_dot, rel=1e-9)
    assert float(unclipped["cos"].reshape(())) < 0


def test_a_zero_norm_is_excluded_from_the_cosine_rather_than_averaged_as_zero():
    """A position with no OPD push has no direction to compare; counting it as
    cosine 0 would pull the mean toward 'orthogonal' with positions that carry
    nothing. The student and teacher agree exactly here, so g_opd vanishes."""
    vocab = 5
    logits = torch.zeros(vocab, dtype=torch.float64)
    t_lp = torch.log_softmax(logits, dim=-1)          # teacher == student
    ids = torch.topk(t_lp, vocab).indices
    terms = _terms_for(logits, t_lp, ids, 0, adv=1.0, old_lp=float(t_lp[0]))
    assert float(terms["opd_sq"].reshape(())) == pytest.approx(0.0, abs=1e-24)
    assert float(terms["pg_live"].reshape(())) == 1.0, "the PG side is alive"
    assert float(terms["align_mask"].reshape(())) == 0.0, "but there is no cosine to take"


def test_the_per_row_coefficient_reaches_the_attribution_push_with_the_right_powers():
    """Point of the plumbing: b enters through the OPD coefficient, so an inner
    product picks up b and a squared norm picks up b^2 -- and the policy
    gradient, which shares row_weight, is untouched."""
    from verl.trainer.ppo.cross_teacher_kl_weight import opd_logit_push

    rows, t, k = 3, 4, 5
    torch.manual_seed(0)
    student = torch.log_softmax(torch.randn(rows, t, k), dim=-1)
    teacher = torch.log_softmax(torch.randn(rows, t, k), dim=-1)
    kl = torch.rand(rows, t) * 0.1
    b = torch.tensor([1.076431, 1.191101, 0.5])

    scalar = opd_logit_push(student_logprob=student, teacher_logprob=teacher,
                            teacher_kl=kl, coef=0.01)
    per_row = opd_logit_push(student_logprob=student, teacher_logprob=teacher,
                             teacher_kl=kl, coef=0.01 * b)
    for i in range(rows):
        assert torch.allclose(per_row["g0"][i], scalar["g0"][i] * b[i], atol=1e-7)
        assert torch.allclose(per_row["g0_tail"][i], scalar["g0_tail"][i] * b[i], atol=1e-7)
        # squared norm therefore moves by b^2
        n_scalar = float((scalar["g0"][i] ** 2).sum())
        n_row = float((per_row["g0"][i] ** 2).sum())
        assert n_row == pytest.approx(n_scalar * float(b[i]) ** 2, rel=1e-6)
