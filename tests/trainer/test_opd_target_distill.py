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
    FALLBACK_NO_COMMON_DIR,
    FALLBACK_NONFINITE,
    FALLBACK_NO_RECEIVER,
    FALLBACK_NO_SIGNAL,
    FALLBACK_NONE,
    N_ROLES,
    N_SIDES,
    UNUSABLE_FEW_TOKENS,
    UNUSABLE_FEW_PROMPTS,
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
from verl.trainer.ppo.sign_weights import (
    ROLE_ENV_ACTION, ROLE_ENV_OBS, ROLE_FORMAT, ROLE_TAG, ROLE_TOOL_CALL,
)

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


def _refs(alpha=None, sbar=None, ctrl=None, nT=3, ready=True, tgt=None):
    return TargetDistillRefs(
        alpha=torch.ones(nT, N_ROLES) if alpha is None else alpha,
        sbar=torch.full((nT, N_ROLES), 1e-2) if sbar is None else sbar,
        sbar_ready=torch.full((nT, N_ROLES), bool(ready)),
        control_roles=_cfg().control_role_mask() if ctrl is None else ctrl,
        target_roles=_cfg().target_role_mask() if tgt is None else tgt)


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
    # NOT "F_p kills a constant". On the loss's support p sums to m < 1 and
    # F_p 1 = p (1 - m), so adding a constant CHANGES the injected direction.
    # The earlier version of this test asserted the opposite and only passed
    # because the fixture had normalised p -- the one case that never occurs.
    q = torch.rand(1, 1, K, generator=g, dtype=torch.float64) * 0.4       # sums to m < 1
    shifted = fisher_apply(q, c + 7.0)
    base = fisher_apply(q, c)
    want = base + 7.0 * q * (1.0 - q.sum(-1, keepdim=True))
    assert torch.allclose(shifted, want, atol=1e-14), "F_p 1 must be p (1 - m)"
    assert (shifted - base).norm() > base.norm(), (
        "on a partial support a constant shift is a change of mechanism, not a re-parameterisation")
    # and on a full support it IS killed -- which is why the claim was plausible
    assert torch.allclose(fisher_apply(p, c + 7.0), exact, atol=1e-14)


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


def test_off_target_roles_and_padding_rows_keep_the_teacher():
    b = _batch()
    b["roles"] = b["roles"].clone()
    b["roles"][:, 6] = ROLE_ENV_OBS                # not generated by the student
    built, s, t = _build(b)
    off = b["roles"] == ROLE_ENV_OBS
    assert torch.allclose(built["target_logprob"][off], t[off], atol=0.0)
    assert float(built["alpha_t"][off].abs().max()) == 0.0
    b2 = _batch(seed=9)
    b2["task_ids"] = torch.full_like(b2["task_ids"], -1)      # all padding
    built2, _, t2 = _build(b2)
    assert torch.allclose(built2["target_logprob"], t2, atol=0.0)


def test_a_rewritten_but_unintegrated_role_keeps_its_own_tasks_rl():
    """tool_call is search's alone, so it can never have a cross-task partner.
    alpha = 0 there would not weaken the integration -- with pg_loss_coef = 0 it
    would delete search's reward on <search>/<answer>, which is its whole task."""
    b = _batch()
    b["roles"] = torch.full_like(b["roles"], ROLE_TOOL_CALL)
    cfg = _cfg()
    assert not bool(cfg.control_role_mask()[ROLE_TOOL_CALL])
    assert bool(cfg.target_role_mask()[ROLE_TOOL_CALL])
    # alpha for EVERY task is 0 in the table; the role is not integrated, so the
    # table must be ignored there and the target must still move.
    built, s, t = _build(b, refs=_refs(alpha=torch.zeros(3, N_ROLES)), cfg=cfg)
    live = built["live"] > 0
    assert bool(live.any()), "fixture: some tool_call token must be live"
    assert float(built["alpha_t"][live].min()) == 1.0
    assert float(built["inject"][live].abs().max()) > 0.0
    # while an INTEGRATED role does read the table, and is switched off by it
    b2 = _batch()
    b2["roles"] = torch.full_like(b2["roles"], ROLE_FORMAT)
    built2, _, t2 = _build(b2, refs=_refs(alpha=torch.zeros(3, N_ROLES)), cfg=cfg)
    assert torch.allclose(built2["target_logprob"], t2, atol=0.0)


def test_eta_zero_is_exactly_pure_opd_everywhere():
    b = _batch()
    built, s, t = _build(b, cfg=_cfg(eta=0.0))
    assert torch.allclose(built["target_logprob"], t, atol=0.0)
    assert float(built["inject"].abs().max()) == 0.0


def test_alpha_zero_is_pure_opd_ON_THE_INTEGRATED_ROLES_ONLY():
    """alpha = 0 is a statement about the cross-task solve, so it reaches only the
    roles the solve covers. Elsewhere alpha is pinned at 1 by construction and the
    task keeps its own RL -- that is the point of separating the two role sets."""
    b = _batch()
    built, s, t = _build(b, refs=_refs(alpha=torch.zeros(3, N_ROLES)))
    ctrl = _cfg().control_role_mask()[b["roles"]]
    assert torch.allclose(built["target_logprob"][ctrl], t[ctrl], atol=0.0)
    assert float(built["inject"][ctrl].abs().max()) == 0.0
    other = (~ctrl) & (built["live"] > 0)
    assert bool(other.any()), "fixture: some live token must sit off the integrated roles"
    assert float(built["inject"][other].abs().max()) > 0.0


def test_c_is_recentred_before_the_clamp_and_the_clamp_is_reported():
    """The centring happens on the RAW tilt, so it is the pre-clamp quantity that
    sums to zero. After a clamp that bites it does not, and claiming otherwise
    would be claiming the clamp is a no-op."""
    b = _batch()
    cfg = _cfg(clamp=1e-3)
    built, s, t = _build(b, cfg=cfg)
    live = built["live"] > 0
    raw = cfg.eta * (built["f"] * built["e"]).unsqueeze(-1) * built["r"]
    centred = raw - raw.mean(-1, keepdim=True)
    assert float(centred[live].sum(-1).abs().max()) < 1e-12
    assert torch.allclose(built["c_base"][live],
                          centred[live].clamp(-cfg.clamp, cfg.clamp), atol=1e-12)
    assert float(built["clamped"].max()) > 0.0, "a tight clamp must be reported"
    assert float(built["c_eff"].abs().max()) <= 1e-3 * (1 + 1e-9)


def test_alpha_multiplies_after_the_clamp_so_the_injection_is_linear_in_it():
    """The statistics are accumulated on F c_base and scaled by alpha; that is
    only the direction the loss took if alpha is applied AFTER the clamp. Clamping
    alpha * c instead makes the two disagree exactly where the clamp bites."""
    b = _batch()
    cfg = _cfg(clamp=1e-3)                       # tight enough to saturate
    a = torch.full((3, N_ROLES), 0.4)
    built = _build(b, refs=_refs(alpha=a), cfg=cfg)[0]
    assert float(built["clamped"].mean()) > 0.1, "fixture: the clamp must bite"
    assert torch.allclose(built["c"], built["alpha_t"].unsqueeze(-1) * built["c_base"], atol=1e-14)
    assert torch.allclose(built["inject"], built["alpha_t"].unsqueeze(-1) * built["inject_base"],
                          atol=1e-12)
    # the alternative the fix removed: clamping after alpha would have bounded c
    # at the clamp itself rather than at alpha * clamp. Read on an INTEGRATED
    # role, since elsewhere alpha is pinned at 1 and the two agree.
    ctrl = cfg.control_role_mask()[b["roles"]]
    assert float(built["c"][ctrl].abs().max()) <= 0.4 * cfg.clamp * (1 + 1e-6)
    assert float(built["c_base"][ctrl].abs().max()) > 0.4 * cfg.clamp, "fixture: the clamp must bind"


def test_f_is_neutral_until_sbar_has_been_seeded():
    """sbar = 0 makes the ratio 2 for every token -- the cap. The first step must
    not inject at double strength while the scale it is relative to is unknown."""
    b = _batch()
    cold = _build(b, refs=_refs(sbar=torch.zeros(3, N_ROLES), ready=False))[0]
    assert torch.allclose(cold["f"], torch.ones_like(cold["f"]))
    hot = _build(b, refs=_refs(sbar=torch.zeros(3, N_ROLES), ready=True))[0]
    assert float(hot["f"].max()) > 1.99, "without the guard sbar = 0 pins f at the cap"


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


def _step(ctl, names, seed, bs=6, T=7, side=None, prompt_idx=None):
    b = _batch(seed=seed, bs=bs, T=T)
    refs = ctl.refs_to_device(names, torch.device("cpu"))
    built, s, t = _build(b, refs=refs, cfg=ctl.cfg)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"],
              response_mask=torch.ones(bs, T), topk_ids=b["topk_ids"],
              advantages=b["advantages"], row_basis=None, p_s=s.exp(),
              side=torch.arange(bs) % 2 if side is None else side,
              prompt_idx=torch.arange(bs) if prompt_idx is None else prompt_idx)
    return ctl.update(names, st.reduced()), built


def test_stats_match_a_dense_recomputation():
    b = _batch(seed=11)
    built, s, t = _build(b)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    basis = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])       # row 5 is padding
    side = torch.arange(6) % 2
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"],
              response_mask=torch.ones(6, 7), topk_ids=b["topk_ids"],
              advantages=b["advantages"], row_basis=basis, p_s=s.exp(),
              side=side, prompt_idx=torch.arange(6))
    red = st.reduced()
    want_fv = torch.zeros(3, N_ROLES, N_SIDES, V, dtype=torch.float64)
    want_fg = torch.zeros(3, N_ROLES, N_SIDES, V, dtype=torch.float64)
    n_live = torch.zeros(3, N_ROLES, dtype=torch.float64)
    fv = fisher_apply(s.exp(), built["r"])
    for i in range(6):
        if basis[i] == 0:
            continue
        for tt in range(7):
            if built["live"][i, tt] <= 0:
                continue
            ti, c, sd = int(b["task_ids"][i]), int(b["roles"][i, tt]), int(side[i])
            want_fv[ti, c, sd].index_add_(0, b["topk_ids"][i, tt], fv[i, tt].double())
            # the SENDER reference is F_p c_base -- centred, clamped, after F --
            # not F_p rtilde, so it is the direction the loss took at alpha = 1
            want_fg[ti, c, sd].index_add_(0, b["topk_ids"][i, tt],
                                          built["inject_base"][i, tt].double())
            n_live[ti, c] += 1
    assert torch.allclose(red["fv_sum"].double(), want_fv, atol=1e-5)
    assert torch.allclose(red["fg_sum"].double(), want_fg, atol=1e-5)
    assert torch.allclose(red["tok"][..., 1], n_live)
    # prompts are counted once per prompt, not once per token
    assert int(red["prompts"][int(b["task_ids"][0]), ROLE_FORMAT].sum()) >= 1
    assert bool(red["prompts"].dtype == torch.bool)


def test_a_task_absent_from_a_batch_holds_its_state_and_ages():
    ctl = TargetDistillController(_cfg(max_staleness=1), V, TASKS)
    _step(ctl, TASKS, 1)
    before = ctl.sbar[("webshop", ROLE_FORMAT)]
    b = _batch(seed=2); b["task_ids"] = torch.tensor([0, 1, 0, 1, 0, 1])
    refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
    built, s, t = _build(b, refs=refs, cfg=ctl.cfg)
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    st.update(built=built, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              topk_ids=b["topk_ids"], advantages=b["advantages"], row_basis=None, p_s=s.exp(),
              side=torch.arange(6) % 2, prompt_idx=torch.arange(6))
    m = ctl.update(TASKS, st.reduced())
    assert ctl.sbar[("webshop", ROLE_FORMAT)] == before, "missing is not zero"
    m = ctl.update(TASKS, st.reduced())
    assert m["actor/target/invalid_reason/webshop/format"] == float(UNUSABLE_STALE)


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
    assert d.sbar == ctl.sbar and d.sbar_n == ctl.sbar_n
    for key, st in ctl.refs.items():
        st2 = d.refs[key]
        assert st.k == st2.k and st.n_obs == st2.n_obs
        assert list(st.prompts) == list(st2.prompts)
        assert list(st.pg_prompts) == list(st2.pg_prompts)
        assert (st.fv is None) == (st2.fv is None)
        if st.fv is not None:
            assert torch.equal(st.fv, st2.fv) and torch.equal(st.fg, st2.fg)
            seen += 1
    assert seen > 0, "fixture: at least one reference must have been populated"
    r1 = ctl.refs_to_device(TASKS, torch.device("cpu"))
    r2 = d.refs_to_device(TASKS, torch.device("cpu"))
    assert torch.equal(r1.alpha, r2.alpha) and torch.equal(r1.sbar, r2.sbar)
    assert torch.equal(r1.sbar_ready, r2.sbar_ready) and bool(r1.sbar_ready.any())
    with pytest.raises(ValueError, match="is not 2"):
        TargetDistillController(_cfg(ema_decay=0.5), V, TASKS).load_state_dict(dict(sd, version=1))
    e = TargetDistillController(_cfg(ema_decay=0.5, eta=2.0), V, TASKS)
    with pytest.raises(ValueError, match="changed across resume"):
        e.load_state_dict(sd)


# ---------------------------------------------------------------------------
# 6. the defects the second review found, one test each


@pytest.mark.parametrize("scale", [1e3, 1.0, 1e-6, 1e-12, 1e-20, 1e-30])
def test_the_solver_answers_the_same_question_at_every_scale(scale):
    """The Gram is measured at 1e-24 and below. The objective was compared against
    an ABSOLUTE 1e-15, so below that scale the enumeration kept whichever feasible
    point it reached first -- returning alpha = 0 where the optimum was 0.5, which
    under distillation-only is the difference between half the RL signal and none."""
    G = np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * scale
    K = np.array([1.0, 1.0, 1.0]) * scale * scale
    sol = solve_alpha(K, G, np.ones(3, bool))
    assert sol["fallback"] == FALLBACK_NONE
    assert np.allclose(sol["alpha"], [1.0, 0.5, 1.0], atol=1e-6)


def test_a_zero_row_receiver_constrains_nothing_and_is_not_divided_by_zero():
    G = np.array([[0.0, 0.0], [-1.0, 1.0]])
    sol = solve_alpha(np.ones(2), G, np.ones(2, bool))
    assert np.isfinite(sol["alpha"]).all() and (sol["slack"][1] >= -1e-9)


def test_unsolved_and_no_common_direction_are_different_findings():
    """Both act by keeping alpha = 1, but they say opposite things about the run:
    one is 'the arithmetic failed', the other is 'the arithmetic succeeded and its
    answer was: switch every sender off'. A single code would hide the second."""
    conflict = np.array([[1.0, -3.0], [-3.0, 1.0]])
    sol = solve_alpha(np.ones(2), conflict, np.ones(2, bool))
    assert sol["fallback"] == FALLBACK_NO_COMMON_DIR and sol["converged"]
    assert np.allclose(sol["alpha"], 1.0), "alpha = 0 would delete the RL signal entirely"
    assert (np.asarray(sol["slack_at_one"]) < 0).any(), "the metric must still show the conflict"
    nan = solve_alpha(np.array([np.nan, 1.0]), conflict, np.ones(2, bool))
    assert nan["fallback"] == FALLBACK_NONFINITE and not nan["converged"]
    dead = solve_alpha(np.zeros(2), conflict, np.ones(2, bool))
    assert dead["fallback"] == FALLBACK_NO_SIGNAL
    assert len({sol["fallback"], nan["fallback"], dead["fallback"]}) == 3


def test_sbar_is_seeded_from_the_step_aggregate_not_a_micro_batch():
    """Two micro-batches with very different ||d|| must leave the same sbar
    whichever order they arrive in -- otherwise the scale f is relative to depends
    on rank placement, and no two runs are comparable."""
    def run(order):
        ctl = TargetDistillController(_cfg(ema_decay=0.0), V, TASKS)
        st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
        refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
        for seed in order:
            b = _batch(seed=seed)
            built, sl, t = _build(b, refs=refs, cfg=ctl.cfg)
            st.update(built=built, task_ids=b["task_ids"], roles=b["roles"],
                      response_mask=torch.ones(6, 7), topk_ids=b["topk_ids"],
                      advantages=b["advantages"], row_basis=None, p_s=sl.exp(),
                      side=torch.arange(6) % 2, prompt_idx=torch.arange(6))
        ctl.update(TASKS, st.reduced())
        return dict(ctl.sbar)
    a, b = run([3, 7]), run([7, 3])
    assert a.keys() == b.keys()
    for k in a:
        assert math.isclose(a[k], b[k], rel_tol=1e-12), k


def test_the_first_step_is_not_injected_at_the_cap():
    """Before any sbar exists f must be 1, not the ratio's cap of 2."""
    ctl = TargetDistillController(_cfg(), V, TASKS)
    r0 = ctl.refs_to_device(TASKS, torch.device("cpu"))
    assert not bool(r0.sbar_ready.any())
    b = _batch()
    assert torch.allclose(_build(b, refs=r0, cfg=ctl.cfg)[0]["f"], torch.ones(6, 7, dtype=torch.float64))
    _step(ctl, TASKS, 1)
    assert bool(ctl.refs_to_device(TASKS, torch.device("cpu")).sbar_ready.any())


def test_the_unread_vocabulary_buffers_are_gone():
    """v_sum and g_sum were 10 MiB each on the device, all-reduced every step, and
    nothing ever read them. The constraint is written after F_p."""
    st = TargetDistillStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"))
    assert not hasattr(st, "v_sum") and not hasattr(st, "g_sum")
    assert set(st.reduced()) == {"tok", "fv_sum", "fg_sum", "side_tok", "prompts",
                                 "pg_prompts", "unkeyed", "fhist"}


def test_the_magnitude_metrics_carry_beta_and_name_their_population():
    """beta F_p c is the intervention; ||F_p c|| alone is off by 100x at beta =
    0.01. And a ratio whose numerator is a live-token mean and whose denominator
    is an all-token mean is not a ratio of anything."""
    ctl = TargetDistillController(_cfg(beta=0.01), V, TASKS)
    m, built = _step(ctl, TASKS, 1)
    k = "alfworld/format"
    nb = m[f"actor/target/inject_norm_nobeta/{k}"]
    assert math.isclose(m[f"actor/target/inject_norm/{k}"], 0.01 * nb, rel_tol=1e-9)
    # both ratios exist, on populations that are stated, and the live one is the
    # larger because its numerator excludes the tokens that contribute nothing
    all_basis = m[f"actor/target/inject_over_d/{k}"]
    live_basis = m[f"actor/target/inject_over_d_live/{k}"]
    assert all_basis > 0 and live_basis > 0
    live_frac = m[f"actor/target/live_frac/{k}"]
    assert 0.0 < live_frac <= 1.0
    ctl2 = TargetDistillController(_cfg(beta=1.0), V, TASKS)
    m2, _ = _step(ctl2, TASKS, 1)
    assert math.isclose(m2[f"actor/target/inject_over_r/{k}"],
                        100.0 * m[f"actor/target/inject_over_r/{k}"], rel_tol=1e-6), \
        "inject_over_r must scale with beta"


def test_the_reference_health_metrics_the_design_names_are_all_emitted():
    ctl = TargetDistillController(_cfg(min_prompts=1, min_pg_prompts=1), V, TASKS)
    for s in (1, 2, 3):
        m, _ = _step(ctl, TASKS, s)
    k = "alfworld/format"
    for name in ("ref_cos_sides_fv", "ref_cos_sides_fg", "prompts_side1", "prompts_side2",
                 "pg_prompts_side1", "tokens_window_side1", "staleness_side1",
                 "n_eff_side1", "n_distinct_side1", "ref_norm_fv_side1", "ref_norm_fg_side1",
                 "removed_by_integration", "removed_frac", "f_p10", "f_p50", "f_p90",
                 "e_clip_frac", "sbar_ready", "valid", "invalid_reason", "K_side1"):
        assert f"actor/target/{name}/{k}" in m, name
    assert "actor/target/unkeyed_rows" in m
    assert m[f"actor/target/f_p10/{k}"] <= m[f"actor/target/f_p90/{k}"]
    assert -1.0 - 1e-9 <= m[f"actor/target/ref_cos_sides_fv/{k}"] <= 1.0 + 1e-9
    for i in TASKS:
        for j in TASKS:
            assert f"actor/target/cos_vg/{i}/{j}/format" in m


def test_a_reference_with_too_few_prompts_is_not_usable():
    """The design fixes 4 prompts / 2 with a PG direction / 64 tokens. A window
    that has only ever seen one prompt cannot say anything about a direction."""
    ctl = TargetDistillController(_cfg(min_prompts=4, min_pg_prompts=2, min_tokens=1), V, TASKS)
    for s in (1, 2, 3):
        m, _ = _step(ctl, TASKS, s, prompt_idx=torch.zeros(6, dtype=torch.long))
    assert m["actor/target/valid/alfworld/format"] == 0.0
    assert m["actor/target/invalid_reason/alfworld/format"] == float(UNUSABLE_FEW_PROMPTS)
    assert m["actor/target/alpha/alfworld/format"] == 1.0, "an unusable reference must not zero alpha"
    # a pair with one side starved: the conditions are checked on BOTH halves, so
    # a reference that exists only on the prompts that happened to hash to side 1
    # is not half-usable, it is unusable.
    ctl2 = TargetDistillController(_cfg(min_prompts=1, min_pg_prompts=1, min_tokens=1), V, TASKS)
    for s in (1, 2, 3):
        m2, _ = _step(ctl2, TASKS, s, side=torch.zeros(6, dtype=torch.long),
                      prompt_idx=torch.arange(6))
    assert m2["actor/target/prompts_side1/alfworld/format"] > 0
    assert m2["actor/target/prompts_side2/alfworld/format"] == 0
    assert m2["actor/target/invalid_reason/alfworld/format"] == float(UNUSABLE_NEVER_SEEN), \
        "one side starved must make the pair unusable, not silently half-usable"


def test_the_gram_pairs_disjoint_prompt_halves():
    """G[i, i] is the term that could confirm itself: the same tokens on both
    sides of the inner product. The two halves never share a prompt, so pairing
    side 1 with side 2 is what makes it evidence."""
    ctl = TargetDistillController(_cfg(ema_decay=0.0, min_prompts=1, min_pg_prompts=1), V, TASKS)
    for s in (1, 2):
        m, _ = _step(ctl, TASKS, s)
    c = ROLE_FORMAT
    for n in TASKS:
        fv = [ctl._ref(n, c, sd).fv for sd in range(N_SIDES)]
        fg = [ctl._ref(n, c, sd).fg for sd in range(N_SIDES)]
        if fv[0] is None or fv[1] is None or fg[0] is None or fg[1] is None:
            continue
        want = 0.5 * (float((fv[0].double() * fg[1].double()).sum())
                      + float((fv[1].double() * fg[0].double()).sum()))
        beta2 = ctl.cfg.beta ** 2
        assert math.isclose(m[f"actor/target/gram_vg/{n}/{n}/format"], beta2 * want, rel_tol=1e-9)
        same = 0.5 * (float((fv[0].double() * fg[0].double()).sum())
                      + float((fv[1].double() * fg[1].double()).sum()))
        assert not math.isclose(want, same, rel_tol=1e-6), \
            "fixture: the same-side product must differ, or the test proves nothing"


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
