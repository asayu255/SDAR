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

"""MOPD v3, checked against autograd rather than against itself.

The claim the whole arm rests on is which logit direction a tilted target
injects. That is checked here by differentiating the actual loss the actor
takes -- beta * KL(p_theta || q*) through topk_kl_per_token -- and comparing to
d + beta F_p c. Everything else (the clip carrying through, the RLSD factor
matching the shipped one, alpha never removing the RL signal on fallback) is
checked the same way: against the thing it claims to equal, not a restatement.
"""

import math

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import policy_loss_gradient_coef, topk_kl_per_token
from verl.trainer.ppo.opd_target_distill import (
    FALLBACK_NO_RECEIVER,
    FALLBACK_NONE,
    N_ROLES,
    UNUSABLE_FEW_TOKENS,
    UNUSABLE_NEVER_SEEN,
    UNUSABLE_STALE,
    TargetDistillConfig,
    TargetDistillController,
    TargetDistillRefs,
    TargetDistillStats,
    build_target,
    fisher_apply,
    opd_strength_ratio,
    rl_support_direction,
    rlsd_support_factor,
    solve_alpha,
)
from verl.trainer.ppo.sign_weights import ROLE_ENV_ACTION, ROLE_FORMAT, ROLE_TAG

TASKS = ["alfworld", "search", "webshop"]
V, K = 40, 5
BETA = 0.01


def _cfg(**kw):
    base = dict(enable=True, eta=1.0, epsilon_w=0.2, clamp=2.0, integrate=True,
                ema_decay=0.0, window_steps=3, min_tokens=1, max_staleness=2)
    base.update(kw)
    return TargetDistillConfig(**base)


def _batch(bs=6, T=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    lp_s = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    lp_t = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    ids = torch.topk(lp_s, K, dim=-1).indices
    resp = ids[..., 0].clone()
    resp[0, 0] = int((ids[0, 0].max() + 1) % V)
    while resp[0, 0] in ids[0, 0]:
        resp[0, 0] = int((resp[0, 0] + 1) % V)
    coef = torch.randn(bs, T, generator=g, dtype=torch.float64)
    adv = torch.randn(bs, T, generator=g, dtype=torch.float64)
    roles = torch.full((bs, T), ROLE_FORMAT, dtype=torch.long)
    roles[:, 2:4] = ROLE_ENV_ACTION
    roles[:, 6] = ROLE_TAG
    return dict(lp_full_s=lp_s, lp_full_t=lp_t, topk_ids=ids, response_ids=resp,
                pg_grad_coef=coef, advantages=adv, roles=roles,
                task_ids=torch.tensor([i % 3 for i in range(bs)]),
                lp_s_full=lp_s, lp_t_full=lp_t)


def _refs(alpha=None, sbar=None, ctrl=None, nT=3):
    return TargetDistillRefs(
        alpha=torch.ones(nT, N_ROLES) if alpha is None else alpha,
        sbar=torch.full((nT, N_ROLES), 1e-2) if sbar is None else sbar,
        control_roles=_cfg().control_role_mask() if ctrl is None else ctrl)


def _build(b, refs=None, cfg=None):
    cfg = cfg or _cfg()
    s = torch.gather(b["lp_full_s"], -1, b["topk_ids"])
    t = torch.gather(b["lp_full_t"], -1, b["topk_ids"])
    return build_target(
        student_topk_logprob=s, teacher_topk_logprob=t, topk_ids=b["topk_ids"],
        response_ids=b["response_ids"], pg_grad_coef=b["pg_grad_coef"],
        advantages=b["advantages"],
        teacher_kl_base=topk_kl_per_token(student_topk_logprob=s, teacher_topk_logprob=t),
        task_ids=b["task_ids"], roles=b["roles"], refs=refs or _refs(), cfg=cfg), s, t


# ---------------------------------------------------------------------------
# 1. THE identity: what a tilted target injects


def test_the_injected_direction_is_d_plus_beta_F_c_against_autograd():
    """-grad_z[beta KL(p||q*)] = d + beta F_p c, differentiating the LOSS the actor takes."""
    g = torch.Generator().manual_seed(3)
    z = torch.randn(V, generator=g, dtype=torch.float64)
    q_full = torch.log_softmax(torch.randn(V, generator=g, dtype=torch.float64) * 2, -1)
    ids = torch.topk(torch.log_softmax(z, -1), K).indices
    c = torch.randn(K, generator=g, dtype=torch.float64) * 0.3
    c = c - c.mean()                                  # re-centred, as build_target does

    def loss(zv, tgt_lp):
        s = torch.log_softmax(zv, -1)[ids].reshape(1, 1, -1)
        return BETA * topk_kl_per_token(student_topk_logprob=s,
                                        teacher_topk_logprob=tgt_lp.reshape(1, 1, -1)).reshape(())
    zt = z.clone().requires_grad_(True)
    d, = torch.autograd.grad(loss(zt, q_full[ids]), zt)
    d = -d[ids]                                        # the plain OPD direction on the support
    # the tilted target, exactly as build_target forms it
    p_on = q_full[ids].exp()
    tail = (1.0 - p_on.sum()).clamp(min=0.0)
    zz = (p_on * c.exp()).sum() + tail
    tgt = q_full[ids] + c - zz.log()
    zt = z.clone().requires_grad_(True)
    got, = torch.autograd.grad(loss(zt, tgt), zt)
    got = -got[ids]
    # p is the FULL-vocabulary softmax gathered at the support -- NOT renormalised
    # over it. topk_kl_per_token keeps the tail as its own bucket, and the tilt's
    # contribution to the gradient is p_k(c_k - sum_{v in S} p_v c_v) with that
    # unnormalised sum. Renormalising is a 9e-4 error on this fixture, which is
    # what this test caught the first time it was written.
    p = torch.log_softmax(z, -1)[ids].exp()
    want = d + BETA * fisher_apply(p.reshape(1, 1, -1), c.reshape(1, 1, -1)).reshape(-1)
    assert torch.allclose(got, want, atol=1e-12), f"max diff {(got - want).abs().max():.3e}"
    # and the renormalised form really is wrong, so the test is not vacuous
    pr = p / p.sum()
    bad = d + BETA * fisher_apply(pr.reshape(1, 1, -1), c.reshape(1, 1, -1)).reshape(-1)
    assert (got - bad).abs().max() > 1e-6


def test_dropping_the_mean_subtraction_is_a_real_error_not_a_detail():
    g = torch.Generator().manual_seed(4)
    p = torch.rand(1, 1, K, generator=g, dtype=torch.float64); p = p / p.sum(-1, keepdim=True)
    c = torch.randn(1, 1, K, generator=g, dtype=torch.float64) * 0.5
    exact = fisher_apply(p, c)
    naive = p * c
    rel = (exact - naive).norm() / exact.norm()
    assert rel > 0.05, "the mean-subtraction should matter; if it does not the fixture is degenerate"
    assert torch.allclose(fisher_apply(p, c + 7.0), exact, atol=1e-14), "F_p kills a constant"


# ---------------------------------------------------------------------------
# 2. the three per-token pieces


def test_r_is_the_clipped_pg_direction_and_zero_off_support():
    b = _batch()
    built, s, t = _build(b)
    p = s.exp()
    hit = (b["topk_ids"] == b["response_ids"].unsqueeze(-1)).double()
    want = (-b["pg_grad_coef"]).unsqueeze(-1) * (hit - p)
    want = want * built["in_support"].unsqueeze(-1)
    assert torch.allclose(built["r"], want, atol=1e-12)
    assert built["in_support"][0, 0] == 0 and float(built["r"][0, 0].abs().sum()) == 0.0


def test_a_clipped_token_leaves_the_target_at_the_teacher():
    """c_t = 0 inside a clip branch => r = 0 => c = 0 => q* = q, exactly."""
    b = _batch()
    b["pg_grad_coef"] = torch.zeros_like(b["pg_grad_coef"])
    built, s, t = _build(b)
    assert torch.allclose(built["target_logprob"], t, atol=0.0), "a fully clipped batch must not move the target"
    assert float(built["inject"].abs().max()) == 0.0
    assert float(built["tv"].abs().max()) == 0.0


def test_f_is_one_at_the_running_mean_and_bounded_by_two():
    s = torch.tensor([0.0, 1.0, 2.0, 100.0], dtype=torch.float64)
    f = opd_strength_ratio(d_norm=s, sbar=torch.ones(4, dtype=torch.float64), delta=1e-30)
    assert f[0] == 0.0
    assert math.isclose(float(f[1]), 1.0)
    assert float(f[2]) > 1.0 and float(f[3]) < 2.0
    assert bool((f <= 2.0).all())


def test_e_matches_the_shipped_rlsd_weight_and_softens_the_failed_trajectory():
    from verl.trainer.ppo.rlsd_utils import compute_rlsd_token_advantage
    g = torch.Generator().manual_seed(5)
    lq = torch.randn(4, 3, generator=g, dtype=torch.float64)
    lp = torch.randn(4, 3, generator=g, dtype=torch.float64)
    A = torch.tensor([[-1.0] * 3, [1.0] * 3, [-2.0] * 3, [0.5] * 3], dtype=torch.float64)
    e = rlsd_support_factor(teacher_logprob_a=lq, student_logprob_a=lp,
                            advantages=A, epsilon_w=0.2)
    # the shipped routine, with lambda = 1 so token_adv = A * w_t exactly
    ta = compute_rlsd_token_advantage(seq_advantages=A, student_log_probs=lp,
                                      teacher_log_probs=lq, response_mask=torch.ones(4, 3),
                                      rlsd_lambda=1.0, rlsd_clip_eps=0.2)
    assert torch.allclose(e, ta / A, atol=1e-12), "e must be the shipped w_t"
    # the property the design invokes: a backed token on a failed trajectory is softened
    backed_and_failed = (A < 0) & (lq > lp)
    assert bool((e[backed_and_failed] < 1.0).all())
    assert bool(((e >= 0.8) & (e <= 1.2)).all())


# ---------------------------------------------------------------------------
# 3. the target as a whole


def test_off_control_roles_and_padding_rows_keep_the_teacher():
    b = _batch()
    built, s, t = _build(b)
    tag = b["roles"] == ROLE_TAG
    assert torch.allclose(built["target_logprob"][tag], t[tag], atol=0.0)
    assert float(built["alpha_t"][tag].abs().max()) == 0.0
    b2 = _batch(seed=9)
    b2["task_ids"] = torch.full_like(b2["task_ids"], -1)      # all padding
    built2, _, t2 = _build(b2)
    assert torch.allclose(built2["target_logprob"], t2, atol=0.0)


def test_eta_zero_and_alpha_zero_are_both_exactly_pure_opd():
    b = _batch()
    for kw, refs in ((dict(eta=0.0), None), ({}, _refs(alpha=torch.zeros(3, N_ROLES)))):
        built, s, t = _build(b, refs=refs, cfg=_cfg(**kw))
        assert torch.allclose(built["target_logprob"], t, atol=0.0)
        assert float(built["inject"].abs().max()) == 0.0


def test_c_is_recentred_and_the_clamp_is_reported():
    b = _batch()
    built, s, t = _build(b, cfg=_cfg(clamp=1e-3))
    live = built["live"] > 0
    assert float(built["c"][live].sum(-1).abs().max()) < 1e-12, "c must sum to zero on the support"
    assert float(built["clamped"].max()) > 0.0, "a tight clamp must be reported"
    assert float(built["c_eff"].abs().max()) <= 1e-3 + 1e-12


def test_the_target_is_a_distribution_and_tv_is_bounded():
    b = _batch()
    built, s, t = _build(b)
    tot = built["target_logprob"].exp().sum(-1)
    assert bool((tot <= 1.0 + 1e-9).all()), "support mass must not exceed one"
    assert bool(((built["tv"] >= 0.0) & (built["tv"] <= 1.0 + 1e-9)).all())


# ---------------------------------------------------------------------------
# 4. the integration


def test_no_conflict_leaves_alpha_at_one():
    G = np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
    sol = solve_alpha(np.ones(3), G, np.ones(3, bool))
    assert np.allclose(sol["alpha"], 1.0) and sol["fallback"] == FALLBACK_NONE


def test_a_conflicting_sender_is_reduced_and_the_constraint_is_restored():
    # receiver 0 is hurt by sender 2 badly enough that alpha = 1 violates
    G = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    valid = np.array([True, False, False])
    sol = solve_alpha(np.ones(3), G, valid)
    assert sol["slack_at_one"][0] < 0.0, "fixture: the constraint must bind"
    assert sol["alpha"][2] < 1.0, "the offending sender must be reduced"
    assert sol["slack"][0] >= -1e-9, "the constraint must be restored"
    assert sol["fallback"] == FALLBACK_NONE


@pytest.mark.parametrize("seed", range(5))
def test_the_solver_is_feasible_and_no_worse_than_a_grid(seed):
    g = np.random.default_rng(seed)
    K = g.uniform(0.1, 2.0, 3)
    G = g.normal(0, 1, (3, 3)); G[np.diag_indices(3)] = np.abs(G[np.diag_indices(3)]) + 0.5
    valid = np.array([True, True, g.uniform() > 0.5])
    sol = solve_alpha(K, G, valid)
    if sol["fallback"] != FALLBACK_NONE:
        assert np.allclose(sol["alpha"], 1.0), "a fallback must be alpha = 1, never 0"
        return
    assert ((G @ sol["alpha"])[valid] >= -1e-8).all()
    obj = lambda a: 0.5 * float((K * (1 - a) ** 2).sum())
    best = None
    for a in np.array(np.meshgrid(*[np.linspace(0, 1, 26)] * 3, indexing="ij")).reshape(3, -1).T:
        if ((G @ a)[valid] >= 0).all() and (best is None or obj(a) < best):
            best = obj(a)
    if best is not None:
        assert obj(sol["alpha"]) <= best + 1e-6, (
            f"the exact solve is worse than a 26^3 grid: {obj(sol['alpha']):.4f} > {best:.4f}")


def test_every_fallback_path_returns_one_not_zero():
    """With distillation-only, alpha = 0 deletes the RL signal. A fallback must
    never do that -- it returns the task's own weighted direction."""
    G = np.array([[1.0, -5.0], [-5.0, 1.0]])
    for K, Gm, valid in ((np.array([np.nan, 1.0]), G, np.ones(2, bool)),
                         (np.ones(2), np.full((2, 2), np.inf), np.ones(2, bool)),
                         (np.ones(2), G, np.zeros(2, bool))):
        sol = solve_alpha(K, Gm, valid)
        assert np.allclose(sol["alpha"], 1.0), f"fallback {sol['fallback']} returned {sol['alpha']}"
    assert solve_alpha(np.ones(2), G, np.zeros(2, bool))["fallback"] == FALLBACK_NO_RECEIVER


# ---------------------------------------------------------------------------
# 5. the accumulator and the controller


def _step(ctl, names, seed, bs=6, T=7):
    b = _batch(seed=seed, bs=bs, T=T)
    refs = ctl.refs_to_device(names, torch.device("cpu"))
    built, s, t = _build(b, refs=refs, cfg=ctl.cfg)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"],
              response_mask=torch.ones(bs, T), topk_ids=b["topk_ids"],
              advantages=b["advantages"], row_basis=None, p_s=s.exp())
    return ctl.update(names, st.reduced()), built


def test_stats_match_a_dense_recomputation():
    b = _batch(seed=11)
    built, s, t = _build(b)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    basis = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])       # row 5 is padding
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"],
              response_mask=torch.ones(6, 7), topk_ids=b["topk_ids"],
              advantages=b["advantages"], row_basis=basis, p_s=s.exp())
    red = st.reduced()
    want_fv = torch.zeros(3, N_ROLES, V, dtype=torch.float64)
    n_live = torch.zeros(3, N_ROLES, dtype=torch.float64)
    fv = fisher_apply(s.exp(), built["r"])
    for i in range(6):
        if basis[i] == 0:
            continue
        for tt in range(7):
            if built["live"][i, tt] <= 0:
                continue
            ti, c = int(b["task_ids"][i]), int(b["roles"][i, tt])
            want_fv[ti, c].index_add_(0, b["topk_ids"][i, tt], fv[i, tt].double())
            n_live[ti, c] += 1
    assert torch.allclose(red["fv_sum"].double(), want_fv, atol=1e-5)
    assert torch.allclose(red["tok"][..., 1], n_live)


def test_a_task_absent_from_a_batch_holds_its_state_and_ages():
    ctl = TargetDistillController(_cfg(max_staleness=1), V, TASKS)
    _step(ctl, TASKS, 1)
    before = ctl._ref("webshop", ROLE_FORMAT).sbar
    b = _batch(seed=2); b["task_ids"] = torch.tensor([0, 1, 0, 1, 0, 1])
    refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
    built, s, t = _build(b, refs=refs, cfg=ctl.cfg)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              topk_ids=b["topk_ids"], advantages=b["advantages"], row_basis=None, p_s=s.exp())
    m = ctl.update(TASKS, st.reduced())
    assert ctl._ref("webshop", ROLE_FORMAT).sbar == before, "missing is not zero"
    m = ctl.update(TASKS, st.reduced())
    assert m["actor/target/unusable_reason/webshop/format"] == float(UNUSABLE_STALE)


def test_integration_off_pins_alpha_at_one():
    ctl = TargetDistillController(_cfg(integrate=False), V, TASKS)
    m, _ = _step(ctl, TASKS, 1)
    for t in TASKS:
        for rn in ("format", "env_action"):
            assert m[f"actor/target/alpha/{t}/{rn}"] == 1.0


def test_metrics_report_what_was_injected():
    ctl = TargetDistillController(_cfg(), V, TASKS)
    m, built = _step(ctl, TASKS, 1)
    for k in ("actor/target/inject_norm/alfworld/format",
              "actor/target/inject_over_r/alfworld/format",
              "actor/target/cos_inject_r/alfworld/format",
              "actor/target/tv_qstar_q/alfworld/format",
              "actor/target/f_mean/alfworld/format",
              "actor/target/e_mean/alfworld/format",
              "actor/target/clamped_frac/alfworld/format",
              "actor/target/recenter_residual/alfworld/format",
              "actor/target/alpha/alfworld/format"):
        assert k in m, k
    assert 0.0 <= m["actor/target/tv_qstar_q/alfworld/format"] <= 1.0


def test_state_round_trips_and_a_changed_config_is_refused():
    ctl = TargetDistillController(_cfg(ema_decay=0.5), V, TASKS)
    for s in (1, 2):
        _step(ctl, TASKS, s)
    sd = ctl.state_dict()
    d = TargetDistillController(_cfg(ema_decay=0.5), V, TASKS)
    d.load_state_dict(sd)
    assert d.step == ctl.step and d.alpha == ctl.alpha and d.usable == ctl.usable
    seen = 0
    for key, st in ctl.refs.items():
        st2 = d.refs[key]
        assert st.sbar == st2.sbar and st.k == st2.k and st.n_obs == st2.n_obs
        assert (st.fv is None) == (st2.fv is None)
        if st.fv is not None:
            assert torch.equal(st.fv, st2.fv) and torch.equal(st.fg, st2.fg)
            seen += 1
    assert seen > 0, "fixture: at least one reference must have been populated"
    r1 = ctl.refs_to_device(TASKS, torch.device("cpu"))
    r2 = d.refs_to_device(TASKS, torch.device("cpu"))
    assert torch.equal(r1.alpha, r2.alpha) and torch.equal(r1.sbar, r2.sbar)
    e = TargetDistillController(_cfg(ema_decay=0.5, eta=2.0), V, TASKS)
    with pytest.raises(ValueError, match="changed across resume"):
        e.load_state_dict(sd)


def test_config_parsing_and_validation():
    c = TargetDistillConfig.from_mapping({"enable": True, "roles": "format,env_action", "eta": "1.5"})
    assert c.roles == ("format", "env_action") and c.eta == 1.5
    with pytest.raises(ValueError):
        TargetDistillConfig.from_mapping({"enable": True, "etaa": 1.0})
    for bad in (dict(eta=-1.0), dict(epsilon_w=1.0), dict(clamp=0.0), dict(ema_decay=1.0),
                dict(roles=("format", "nope"))):
        with pytest.raises(ValueError):
            _cfg(**bad).validate()
    m = _cfg().control_role_mask()
    assert bool(m[ROLE_FORMAT]) and bool(m[ROLE_ENV_ACTION]) and not bool(m[ROLE_TAG])
