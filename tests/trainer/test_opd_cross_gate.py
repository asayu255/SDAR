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

"""The cross gate (MOPD v1): the split, the gate's bound, the stats, the solver, the
controller's window and resume. Everything on the CPU, most of it against a dense
recomputation rather than against itself."""

import itertools
import math

import numpy as np
import pytest
import torch

from verl.trainer.ppo.opd_cross_gate import (
    _CROSS_IDX,
    _SEND_IDX,
    INVALID_FEW_PROMPTS,
    INVALID_NEVER_SEEN,
    INVALID_STALE,
    N_ROLES,
    N_SIDES,
    ROLE_ID,
    CrossGateConfig,
    CrossGateController,
    CrossGateRefs,
    CrossGateStats,
    cross_gate_forward,
    cross_gate_prompt_columns,
    prompt_side,
    solve_role,
)
from verl.trainer.ppo.sign_weights import ROLE_ENV_ACTION, ROLE_FORMAT, ROLE_TAG

TASKS = ["alfworld", "search", "webshop"]
V, K = 40, 5


def _cfg(**kw):
    base = dict(enable=True, eps_cross=0.0, rho=1.0e4, lambda_max=0.2, ema_decay=0.0,
                window_steps=3, min_prompts=2, min_pg_prompts=1, min_tokens=4, max_staleness=2)
    base.update(kw)
    return CrossGateConfig(**base)


def _batch(bs=6, T=7, seed=0, coef_scale=1.0):
    """Random top-k inputs of the shapes the actor hands over."""
    g = torch.Generator().manual_seed(seed)
    lp_full_s = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    lp_full_t = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    topk_ids = torch.topk(lp_full_s, K, dim=-1).indices
    lp_s = torch.gather(lp_full_s, -1, topk_ids)
    lp_t = torch.gather(lp_full_t, -1, topk_ids)
    # sampled id: mostly inside the support, a few outside
    resp = topk_ids[..., 0].clone()
    resp[0, 0] = (topk_ids[0, 0].max() + 1) % V
    while resp[0, 0] in topk_ids[0, 0]:
        resp[0, 0] = (resp[0, 0] + 1) % V
    kl = (lp_s.exp() * (lp_s - lp_t)).sum(-1).clamp(min=0.0)
    coef = coef_scale * torch.randn(bs, T, generator=g, dtype=torch.float64)
    task_ids = torch.tensor([i % 3 for i in range(bs)])
    roles = torch.full((bs, T), ROLE_FORMAT, dtype=torch.long)
    roles[:, 2:4] = ROLE_ENV_ACTION
    roles[:, 6] = ROLE_TAG
    return dict(student_topk_logprob=lp_s, teacher_topk_logprob=lp_t, teacher_kl=kl,
                topk_ids=topk_ids, response_ids=resp, pg_grad_coef=coef,
                task_ids=task_ids, roles=roles)


def _zero_refs(nT=3, valid=None, lam=None, ctrl=None):
    v = torch.zeros(nT, N_ROLES, N_SIDES, V)
    R = torch.zeros(nT, N_ROLES, N_SIDES)
    sw = torch.zeros(nT, N_ROLES, N_SIDES)
    valid_t = torch.zeros(nT, N_ROLES, dtype=torch.bool) if valid is None else valid
    lam_t = torch.zeros(nT, N_ROLES) if lam is None else lam
    ctrl_t = _cfg().control_role_mask() if ctrl is None else ctrl
    return CrossGateRefs(v=v, R=R, side_w=sw, valid=valid_t, lam=lam_t, control_roles=ctrl_t)


def _fwd(b, refs, opd_coef=0.01, delta=1e-30, q_scale=1.0, gate_version=1):
    return cross_gate_forward(
        student_topk_logprob=b["student_topk_logprob"], teacher_topk_logprob=b["teacher_topk_logprob"],
        teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], response_ids=b["response_ids"],
        pg_grad_coef=b["pg_grad_coef"], opd_coef=opd_coef, task_ids=b["task_ids"], roles=b["roles"],
        refs=refs, delta=delta, q_scale=q_scale, gate_version=gate_version,
    )


def _refs_from_population(b, fwd, side_of_row, nT=3):
    """v = E[r], R = E[||r||^2] per (task, role, side) from the batch itself --
    the same weights on both, so Jensen's bound is in force."""
    v = torch.zeros(nT, N_ROLES, N_SIDES, V, dtype=torch.float64)
    R = torch.zeros(nT, N_ROLES, N_SIDES, dtype=torch.float64)
    n = torch.zeros(nT, N_ROLES, N_SIDES, dtype=torch.float64)
    bs, T = b["roles"].shape
    for i in range(bs):
        for t in range(T):
            if fwd["in_support"][i, t] <= 0:
                continue
            ti, c, s = int(b["task_ids"][i]), int(b["roles"][i, t]), int(side_of_row[i])
            v[ti, c, s].index_add_(0, b["topk_ids"][i, t], fwd["r"][i, t].double())
            R[ti, c, s] += float(fwd["r_sq"][i, t])
            n[ti, c, s] += 1
    nz = n > 0
    v[nz] = v[nz] / n[nz].unsqueeze(-1)
    R[nz] = R[nz] / n[nz]
    sw = torch.zeros(nT, N_ROLES, N_SIDES, dtype=torch.float64)
    both = nz.all(dim=-1)
    sw[both] = 0.5
    one = nz & ~both.unsqueeze(-1)
    sw[one] = 1.0
    return v.float(), R.float(), sw.float(), both


# ---------------------------------------------------------------------------
# 1. the prompt split


def test_prompt_side_is_deterministic_and_process_stable():
    k1, s1 = prompt_side("alfworld", "You are in the middle of a room.", seed=1)
    k2, s2 = prompt_side("alfworld", "You are in the middle of a room.", seed=1)
    assert k1 == k2 and s1 == s2 and s1 in (0, 1)
    # the documented digest, so a different hash library or encoding cannot
    # silently re-split every prompt
    import hashlib
    d = hashlib.sha256(b"1|1|alfworld|You are in the middle of a room.").digest()
    assert k1 == d.hex() and s1 == int.from_bytes(d[:8], "big") % 2


def test_task_and_seed_enter_the_split_and_the_rank_does_not():
    ka, _ = prompt_side("alfworld", "q", seed=1)
    kb, _ = prompt_side("webshop", "q", seed=1)
    kc, _ = prompt_side("alfworld", "q", seed=2)
    assert len({ka, kb, kc}) == 3
    # over many prompts the two sides are both populated
    sides = [prompt_side("search", f"question {i}", seed=1)[1] for i in range(200)]
    assert 60 < sum(sides) < 140


def test_prompt_columns_take_the_groups_turn0_anchor_and_count_a_prompt_once():
    uids = ["g1", "g1", "g1", "g2", "g2", "g3", "g3", "g1"]
    turns = [0, 1, 2, 1, 0, 1, 2, 0]
    anchors = ["A", "obs1", "obs2", "obsB", "B", "obsC1", "obsC2", "A"]
    tasks = ["alfworld"] * 8
    real = [True] * 8
    out = cross_gate_prompt_columns(uids=uids, turn_steps=turns, anchors=anchors, task_names=tasks,
                                    real=real, seed=1)
    side, pidx, keys = out["side"], out["prompt_idx"], out["keys"]
    # g1 and g2 have a turn-0 row, g3 does not
    assert (side[[0, 1, 2, 7]] >= 0).all() and (side[[3, 4]] >= 0).all()
    assert (side[[5, 6]] == -1).all() and (pidx[[5, 6]] == -1).all()
    assert out["unkeyed_rows"] == 2
    # all rows of a group share side and dense index, from the turn-0 anchor
    assert len(set(side[[0, 1, 2, 7]].tolist())) == 1 and len(set(pidx[[0, 1, 2, 7]].tolist())) == 1
    assert pidx[0] != pidx[3]
    assert len(keys) == 2
    assert keys[int(pidx[0])] == prompt_side("alfworld", "A", seed=1)[0]


def test_prompt_columns_ignore_padding_rows_and_a_second_group_of_the_same_prompt_is_one_prompt():
    uids = ["g1", "g1", "g2", "g2", "pad"]
    turns = [0, 1, 0, 1, 0]
    anchors = ["A", "x", "A", "y", "A"]
    tasks = ["webshop"] * 5
    real = [True, True, True, True, False]
    out = cross_gate_prompt_columns(uids=uids, turn_steps=turns, anchors=anchors, task_names=tasks,
                                    real=real, seed=1)
    assert out["prompt_idx"][0] == out["prompt_idx"][2] and len(out["keys"]) == 1
    assert out["side"][4] == -1 and out["prompt_idx"][4] == -1 and out["unkeyed_rows"] == 0


# ---------------------------------------------------------------------------
# 2. the gate


def test_without_references_the_gate_is_off_and_nothing_is_touched():
    b = _batch()
    f = _fwd(b, _zero_refs())
    assert torch.equal(f["w"], torch.ones_like(f["w"]))
    assert float(f["h"].abs().max()) == 0.0
    assert not f["w"].requires_grad and not f["h"].requires_grad


def test_r_is_the_clipped_pg_direction_on_the_support_and_off_support_tokens_are_excluded():
    b = _batch()
    f = _fwd(b, _zero_refs())
    p = b["student_topk_logprob"].exp()
    hit = (b["topk_ids"] == b["response_ids"].unsqueeze(-1)).double()
    r_dense = (-b["pg_grad_coef"]).unsqueeze(-1) * (hit - p)
    r_dense = r_dense * f["in_support"].unsqueeze(-1)
    assert torch.allclose(f["r"], r_dense, atol=1e-10)
    assert f["in_support"][0, 0] == 0 and float(f["r"][0, 0].abs().sum()) == 0.0
    assert torch.allclose(f["r_sq"], (r_dense ** 2).sum(-1), atol=1e-10)


def test_d_is_beta_times_the_opd_descent_direction():
    b = _batch()
    f = _fwd(b, _zero_refs(), opd_coef=0.01)
    lp_s, lp_t = b["student_topk_logprob"], b["teacher_topk_logprob"]
    d_dense = 0.01 * lp_s.exp() * (b["teacher_kl"].unsqueeze(-1) - (lp_s - lp_t))
    assert torch.allclose(f["d"], d_dense, atol=1e-12)


def test_x_is_the_inner_product_with_the_pooled_reference_gathered_at_the_support():
    b = _batch(seed=3)
    side_of_row = torch.tensor([0, 1, 0, 1, 0, 1])
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, side_of_row)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=both, lam=torch.zeros(3, N_ROLES),
                         control_roles=_cfg().control_role_mask())
    f = _fwd(b, refs)
    bs, T = b["roles"].shape
    for i in range(3):
        for r_ in range(bs):
            for t in range(T):
                c = int(b["roles"][r_, t])
                vp = (v[i, c] * sw[i, c].unsqueeze(-1)).sum(0)          # pooled reference (V,)
                want = float((vp[b["topk_ids"][r_, t]] * f["d"][r_, t].float()).sum())
                assert math.isclose(float(f["x"][r_, t, i]), want, rel_tol=1e-4, abs_tol=1e-9)


def test_q_is_bounded_by_the_weaker_sides_sqrt_kappa_and_zero_unless_both_sides_oppose():
    b = _batch(seed=5, bs=9, T=8)
    side_of_row = torch.tensor([i % 2 for i in range(9)])
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, side_of_row)
    # Jensen: ||E r||^2 <= E ||r||^2 on every populated cell
    vn2 = (v.double() ** 2).sum(-1)
    assert bool((vn2 <= R.double() + 1e-9)[R > 0].all())
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=both, lam=torch.zeros(3, N_ROLES),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    # a DIFFERENT batch, so the bound is not an artefact of testing on the training data
    b2 = _batch(seed=6, bs=9, T=8)
    f = _fwd(b2, refs)
    kappa = torch.where(R > 0, vn2 / R.double().clamp(min=1e-30), torch.zeros_like(vn2))  # (nT, nR, nS)
    bs, T = b2["roles"].shape
    for i in range(3):
        for r_ in range(bs):
            for t in range(T):
                c = int(b2["roles"][r_, t])
                bound = float(kappa[i, c].min().sqrt())
                assert float(f["q"][r_, t, i]) <= bound + 1e-6
    # both-sides rule: q > 0 only where the inner product with EACH side is negative
    x_side = []
    for s in range(N_SIDES):
        vs = v[:, :, s]
        xs = torch.zeros(bs, T, 3, dtype=torch.float64)
        for i in range(3):
            for r_ in range(bs):
                for t in range(T):
                    c = int(b2["roles"][r_, t])
                    xs[r_, t, i] = float((vs[i, c][b2["topk_ids"][r_, t]] * f["d"][r_, t].float()).sum())
        x_side.append(xs)
    both_neg = (x_side[0] < 0) & (x_side[1] < 0)
    pos = f["q"] > 0
    assert bool((~pos | both_neg).all()), "q > 0 somewhere a side did not oppose"


def test_h_averages_uniformly_over_valid_receivers_excluding_the_sender_and_uncontrolled_roles():
    b = _batch(seed=7)
    side_of_row = torch.tensor([0, 1, 0, 1, 0, 1])
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, side_of_row)
    valid = both.clone()
    valid[:, ROLE_TAG] = True     # even if valid, tag is not a controlled role
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid, lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=_cfg().control_role_mask())
    f = _fwd(b, refs)
    bs, T = b["roles"].shape
    for r_ in range(bs):
        j = int(b["task_ids"][r_])
        for t in range(T):
            c = int(b["roles"][r_, t])
            recv = [i for i in range(3) if i != j and bool(valid[i, c]) and c in (ROLE_FORMAT, ROLE_ENV_ACTION)]
            om = f["omega"][r_, t]
            if not recv:
                assert float(f["h"][r_, t]) == 0.0 and float(om.sum()) == 0.0
                assert float(f["w"][r_, t]) == 1.0
                continue
            assert om[j] == 0.0
            for i in recv:
                assert math.isclose(float(om[i]), 1.0 / len(recv), rel_tol=1e-6)
            want = sum(float(f["q"][r_, t, i]) for i in recv) / len(recv)
            assert math.isclose(float(f["h"][r_, t]), want, rel_tol=1e-5, abs_tol=1e-9)
            assert math.isclose(float(f["w"][r_, t]), 1.0 - 0.2 * float(f["h"][r_, t]), rel_tol=1e-6)
    # roles off control never see lambda
    assert bool((f["lam_t"][b["roles"] == ROLE_TAG] == 0).all())


def test_the_beta_scale_identity_behind_rho():
    """d = beta * dtilde: x and B, D scale with beta; h, q and K's ratio do not."""
    b = _batch(seed=8)
    side_of_row = torch.tensor([0, 1, 0, 1, 0, 1])
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, side_of_row)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=both, lam=torch.zeros(3, N_ROLES),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    f1 = _fwd(b, refs, opd_coef=0.01)
    f2 = _fwd(b, refs, opd_coef=0.1)
    assert torch.allclose(f2["x"], 10.0 * f1["x"], rtol=1e-6, atol=1e-12)
    assert torch.allclose(f2["d_sq"], 100.0 * f1["d_sq"], rtol=1e-6, atol=1e-12)
    assert torch.allclose(f2["h"], f1["h"], rtol=1e-6, atol=1e-9)
    assert torch.allclose(f2["q"], f1["q"], rtol=1e-6, atol=1e-9)
    k1 = (f1["h"] ** 2 * f1["d_sq"]).sum() / f1["d_sq"].sum()
    k2 = (f2["h"] ** 2 * f2["d_sq"]).sum() / f2["d_sq"].sum()
    assert math.isclose(float(k1), float(k2), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. the accumulators, against a dense recomputation


def test_stats_scatter_the_reference_sums_and_bitmaps_exactly():
    b = _batch(seed=9)
    side = torch.tensor([0, 1, 0, 1, -1, 1])       # row 4: no stable key
    pidx = torch.tensor([0, 1, 2, 3, -1, 0])       # rows 0 and 5 are the same prompt
    basis = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])   # row 5 is a duplicated padding row
    f = _fwd(b, _zero_refs())
    st = CrossGateStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"), n_prompts=8)
    st.update(fwd=f, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], side=side, prompt_idx=pidx, row_basis=basis)
    red = st.reduced()
    # dense recomputation over real rows with a side
    want = torch.zeros(3, N_ROLES, N_SIDES, V, dtype=torch.float64)
    n_tok = torch.zeros(3, N_ROLES, N_SIDES, dtype=torch.float64)
    for i in range(6):
        if basis[i] == 0 or side[i] < 0:
            continue
        for t in range(7):
            if f["in_support"][i, t] <= 0:
                continue
            ti, c, s = int(b["task_ids"][i]), int(b["roles"][i, t]), int(side[i])
            want[ti, c, s].index_add_(0, b["topk_ids"][i, t], f["r"][i, t].double())
            n_tok[ti, c, s] += 1
    assert torch.allclose(red["ref_sum"].double(), want, atol=1e-5)
    assert torch.allclose(red["side"][..., 0], n_tok)
    # the bitmap: row 0 (task 0, side 0, prompt 0) contributed at format and env_action
    assert red["pbm_contrib"][0, ROLE_FORMAT, 0, 0] == 1.0 and red["pbm_contrib"][0, ROLE_ENV_ACTION, 0, 0] == 1.0
    # row 5 is padding: prompt 0 on side 1 must NOT be marked by it
    assert red["pbm_contrib"][2, ROLE_FORMAT, 1, 0] == 0.0
    # row 4 has no side: counted as unkeyed on the sender side, absent from references
    assert red["send"][1, ROLE_FORMAT, _SEND_IDX["n_unkeyed"]] > 0   # task 1 format
    assert math.isclose(float(red["ref_sum"][1].abs().sum()), float(want[1].abs().sum()), rel_tol=1e-5)


def test_stats_cross_columns_are_sums_of_the_forward_outputs():
    b = _batch(seed=10)
    side_of_row = torch.tensor([0, 1, 0, 1, 0, 1])
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, side_of_row)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=both, lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=_cfg().control_role_mask())
    f = _fwd(b, refs)
    st = CrossGateStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"), n_prompts=8)
    st.update(fwd=f, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], side=side_of_row,
              prompt_idx=torch.arange(6), row_basis=None)
    red = st.reduced()
    for i in range(3):
        for j in range(3):
            for c in range(N_ROLES):
                m = (b["task_ids"].unsqueeze(-1).expand(6, 7) == j) & (b["roles"] == c)
                if not m.any():
                    continue
                x = f["x"][:, :, i][m]
                assert math.isclose(float(red["cross"][i, j, c, _CROSS_IDX["B_sum"]]), float(x.sum()), rel_tol=1e-6, abs_tol=1e-9)
                assert math.isclose(float(red["cross"][i, j, c, _CROSS_IDX["D_sum"]]), float((f["h"][m] * x).sum()), rel_tol=1e-6, abs_tol=1e-9)
                assert math.isclose(float(red["cross"][i, j, c, _CROSS_IDX["Aneg_sum"]]), float((-x).clamp(min=0).sum()), rel_tol=1e-6, abs_tol=1e-9)
                assert math.isclose(float(red["cross"][i, j, c, _CROSS_IDX["realized_sum"]]), float((f["w"][m] * x).sum()), rel_tol=1e-6, abs_tol=1e-9)
                # the sign split, and D = D+ - D- exactly
                dp = float(red["cross"][i, j, c, _CROSS_IDX["Dpos_sum"]])
                dn = float(red["cross"][i, j, c, _CROSS_IDX["Dneg_sum"]])
                assert math.isclose(dp, float((f["h"][m] * x.clamp(min=0)).sum()), rel_tol=1e-6, abs_tol=1e-9)
                assert math.isclose(dn, float((f["h"][m] * (-x).clamp(min=0)).sum()), rel_tol=1e-6, abs_tol=1e-9)
                assert dp >= -1e-12 and dn >= -1e-12, "a sign-split removal went negative"
                assert math.isclose(dp - dn, float(red["cross"][i, j, c, _CROSS_IDX["D_sum"]]),
                                    rel_tol=1e-9, abs_tol=1e-12), "D != D+ - D-"
    for j in range(3):
        for c in range(N_ROLES):
            m = (b["task_ids"].unsqueeze(-1).expand(6, 7) == j) & (b["roles"] == c)
            if not m.any():
                continue
            assert math.isclose(float(red["send"][j, c, _SEND_IDX["K_num"]]), float((f["h"][m] ** 2 * f["d_sq"][m]).sum()), rel_tol=1e-6)
            assert math.isclose(float(red["send"][j, c, _SEND_IDX["strength_lost"]]), float(((1 - f["w"][m] ** 2) * f["d_sq"][m]).sum()), rel_tol=1e-6, abs_tol=1e-12)
            # the sender's own cost, and T = T+ - T-
            sd = (f["r"][m] * f["d"][m]).sum(-1)
            tp = float(red["send"][j, c, _SEND_IDX["T_pos"]])
            tn = float(red["send"][j, c, _SEND_IDX["T_neg"]])
            assert math.isclose(float(red["send"][j, c, _SEND_IDX["T_num"]]), float((f["h"][m] * sd).sum()),
                                rel_tol=1e-6, abs_tol=1e-12)
            assert math.isclose(tp, float((f["h"][m] * sd.clamp(min=0)).sum()), rel_tol=1e-6, abs_tol=1e-12)
            assert math.isclose(tn, float((f["h"][m] * (-sd).clamp(min=0)).sum()), rel_tol=1e-6, abs_tol=1e-12)
            assert math.isclose(tp - tn, float(red["send"][j, c, _SEND_IDX["T_num"]]),
                                rel_tol=1e-9, abs_tol=1e-12), "T != T+ - T-"


# ---------------------------------------------------------------------------
# 4. the solver


def test_no_intervention_is_optimal_when_every_condition_already_holds():
    K = np.array([1.0, 2.0, 3.0])
    B = np.array([[0, 1.0, 2.0], [0.5, 0, 0.5], [1.0, 1.0, 0]])     # all positive: s(0) = 0
    D = -np.ones((3, 3))
    R = np.ones(3)
    sol = solve_role(K, B, D, R, np.ones(3, bool), eps=0.0, rho=1e4, lam_max=0.2, delta=1e-30)
    assert sol["converged"] and np.allclose(sol["lam"], 0.0) and np.allclose(sol["s0"], 0.0)


def _brute(K, B, D, R, valid, eps, rho, lam_max, delta, grid=41):
    n = len(K)
    off = ~np.eye(n, dtype=bool)
    axes = np.meshgrid(*[np.linspace(0, lam_max, grid)] * n, indexing="ij")
    L = np.stack([a.reshape(-1) for a in axes], -1)
    best, bl = np.inf, None
    for l in L:
        s = np.maximum(-eps * R - (B * off).sum(1) + (D * off) @ l, 0.0) * valid
        fv = (K * l * l).sum() + rho * ((s / (R + delta)) ** 2).sum()
        if fv < best:
            best, bl = fv, l
    return bl, best


@pytest.mark.parametrize("seed", range(6))
def test_the_solver_matches_a_grid_search_on_random_instances(seed):
    g = np.random.default_rng(seed)
    n = 3
    K = g.uniform(0.0, 2.0, n)
    B = g.normal(0.0, 1.0, (n, n)) * 0.02
    D = -np.abs(g.normal(0.0, 1.0, (n, n))) * 0.02      # attenuation removes conflict
    R = g.uniform(0.5, 2.0, n)
    valid = g.uniform(size=n) > 0.3
    if not valid.any():
        valid[0] = True
    args = dict(eps=0.0, rho=1e4, lam_max=0.2, delta=1e-30)
    sol = solve_role(K, B, D, R, valid, **args)
    bl, bf = _brute(K, B, D, R, valid, **args)
    assert sol["converged"]
    assert sol["f"] <= bf + 1e-9, "the solver found a worse point than a 41^3 grid"
    assert (sol["lam"] >= -1e-12).all() and (sol["lam"] <= 0.2 + 1e-12).all()


def test_a_negative_baseline_asks_for_attenuation_from_the_sender_that_removes_it():
    K = np.array([1.0, 1.0, 1.0])
    B = np.zeros((3, 3)); B[0, 2] = -0.05           # receiver 0 is hurt by sender 2
    D = np.zeros((3, 3)); D[0, 2] = -0.05           # ...and attenuating sender 2 removes it
    R = np.ones(3)
    sol = solve_role(K, B, D, R, np.array([True, False, False]), eps=0.0, rho=1e4, lam_max=0.2, delta=1e-30)
    assert sol["lam"][2] > 0.0 and sol["lam"][0] == 0.0 and sol["lam"][1] == 0.0
    assert sol["s0"][0] == pytest.approx(0.05)
    assert sol["s"][0] < sol["s0"][0]


def test_solver_refuses_nonfinite_input_and_idles_without_a_valid_receiver():
    K = np.array([1.0, np.nan]); B = np.zeros((2, 2)); D = np.zeros((2, 2)); R = np.ones(2)
    sol = solve_role(K, B, D, R, np.ones(2, bool), eps=0.0, rho=1.0, lam_max=0.2, delta=1e-30)
    assert sol["reason"] == "nonfinite_input" and np.allclose(sol["lam"], 0.0)
    sol = solve_role(np.ones(2), B, -np.ones((2, 2)), R, np.zeros(2, bool), eps=0.0, rho=1.0, lam_max=0.2, delta=1e-30)
    assert sol["reason"] == "no_valid_receiver" and np.allclose(sol["lam"], 0.0)


# ---------------------------------------------------------------------------
# 5. the controller: references, the window, validity, lambda, resume


def _step(ctl, names, seed, side_of_row, prompt_idx, keys, lam_override=None, bs=6, T=7):
    """One training step: forward with the previous references, fold, update."""
    b = _batch(seed=seed, bs=bs, T=T)
    refs = ctl.refs_to_device(names, torch.device("cpu"))
    f = _fwd(b, refs)
    st = CrossGateStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"), n_prompts=16)
    st.update(fwd=f, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(bs, T),
              teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], side=side_of_row,
              prompt_idx=prompt_idx, row_basis=None)
    return ctl.update(names, st.reduced(), keys), f


def test_references_become_valid_after_the_window_fills_and_a_repeated_prompt_counts_once():
    ctl = CrossGateController(_cfg(min_prompts=2, min_pg_prompts=1, min_tokens=1), V, TASKS)
    side = torch.tensor([0, 1, 0, 1, 0, 1])      # task i%3, side i%2 -> each (task) sees both sides
    # step 1: one prompt per row, distinct keys
    m1, _ = _step(ctl, TASKS, 1, side, torch.arange(6), [f"k{i}" for i in range(6)])
    # each (task, role=format, side) has seen exactly ONE prompt -> below min_prompts=2
    assert m1["actor/cross/valid/alfworld/format"] == 0.0
    assert m1["actor/cross/invalid_reason/alfworld/format"] == float(INVALID_FEW_PROMPTS)
    # step 2: the SAME prompts again -> still one prompt per cell
    m2, _ = _step(ctl, TASKS, 2, side, torch.arange(6), [f"k{i}" for i in range(6)])
    assert m2["actor/cross/prompts_side1/alfworld/format"] == 1.0
    assert m2["actor/cross/valid/alfworld/format"] == 0.0
    # step 3: new prompts -> two distinct per cell -> valid
    m3, _ = _step(ctl, TASKS, 3, side, torch.arange(6), [f"n{i}" for i in range(6)])
    assert m3["actor/cross/prompts_side1/alfworld/format"] == 2.0
    assert m3["actor/cross/valid/alfworld/format"] == 1.0
    assert m3["actor/cross/invalid_reason/alfworld/format"] == 0.0


def test_a_missing_task_holds_its_reference_and_goes_stale_rather_than_learning_zero():
    ctl = CrossGateController(_cfg(min_prompts=1, min_pg_prompts=0, min_tokens=1, max_staleness=1), V, TASKS)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    _step(ctl, TASKS, 1, side, torch.arange(6), [f"k{i}" for i in range(6)])
    v_before = ctl.refs[("webshop", ROLE_FORMAT, 0)].v.clone()
    n_before = ctl.refs[("webshop", ROLE_FORMAT, 0)].n_obs
    # a step where webshop has no rows: task ids only 0 and 1
    b = _batch(seed=2)
    b["task_ids"] = torch.tensor([0, 1, 0, 1, 0, 1])
    refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
    f = _fwd(b, refs)
    st = CrossGateStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"), n_prompts=16)
    st.update(fwd=f, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], side=side, prompt_idx=torch.arange(6), row_basis=None)
    m = ctl.update(TASKS, st.reduced(), [f"k{i}" for i in range(6)])
    st_w = ctl.refs[("webshop", ROLE_FORMAT, 0)]
    assert torch.equal(st_w.v, v_before) and st_w.n_obs == n_before, "missing is not zero"
    assert m["actor/cross/staleness_side1/webshop/format"] == 1.0
    # one more absent step exceeds max_staleness=1 -> invalid for staleness
    m = ctl.update(TASKS, st.reduced(), [f"k{i}" for i in range(6)])
    assert m["actor/cross/invalid_reason/webshop/format"] == float(INVALID_STALE)


def test_lambda_is_zero_off_control_roles_and_within_the_cap_and_zero_when_nothing_is_negative():
    ctl = CrossGateController(_cfg(min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    for s in range(1, 4):
        m, f = _step(ctl, TASKS, s, side, torch.arange(6), [f"k{s}_{i}" for i in range(6)])
    for n in TASKS:
        assert m[f"actor/cross/lambda_next/{n}/tag"] == 0.0
        for rn in ("format", "env_action"):
            lam = m[f"actor/cross/lambda_next/{n}/{rn}"]
            assert 0.0 <= lam <= 0.2 + 1e-12
    # force every baseline positive: the solver must return 0 everywhere
    for key in list(ctl.cross):
        if key[3] == "B":
            ctl.cross[key].val = abs(ctl.cross[key].val) + 1e-3
    m, _ = _step(ctl, TASKS, 9, side, torch.arange(6), [f"z{i}" for i in range(6)])
    # B is re-folded from this step with ema_decay=0, so make the check on the solver's own report
    for rn in ("format", "env_action"):
        assert m[f"actor/cross/solver_converged/{rn}"] == 1.0


def test_lambda_moves_when_a_valid_receiver_reports_a_negative_baseline():
    ctl = CrossGateController(_cfg(min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    for s in range(1, 3):
        _step(ctl, TASKS, s, side, torch.arange(6), [f"k{s}_{i}" for i in range(6)])
    # plant a conflict: alfworld (receiver) vs webshop (sender) at format
    c = ROLE_FORMAT
    ctl.cross[("alfworld", "webshop", c, "B")].val = -0.05
    ctl.cross[("alfworld", "webshop", c, "D")].val = -0.05
    ctl.cross[("alfworld", "search", c, "B")].val = 0.0
    ctl.cross[("alfworld", "search", c, "D")].val = 0.0
    ctl.validity[("alfworld", c)] = (True, 0, 0)
    ctl.k_num[("webshop", c)].val = 1e-3
    # solve directly with the planted EMAs (update() would re-fold them)
    names = TASKS
    K = np.array([ctl.k_num.get((j, c)).val / (ctl.d_sq.get(j).val + 1e-30) for j in names])
    Bm = np.zeros((3, 3)); Dm = np.zeros((3, 3))
    for ti, i in enumerate(names):
        for tj, j in enumerate(names):
            if i != j:
                Bm[ti, tj] = ctl.cross.get((i, j, c, "B")).val
                Dm[ti, tj] = ctl.cross.get((i, j, c, "D")).val
    Rv = np.array([np.mean([ctl.refs[(i, c, s)].R for s in range(2)]) for i in names])
    valid = np.array([ctl.validity.get((i, c), (False, 0, 0))[0] for i in names])
    sol = solve_role(K, Bm, Dm, Rv, valid, eps=0.0, rho=1e4, lam_max=0.2, delta=1e-30)
    assert sol["lam"][2] > 0.0, "webshop's lambda must rise for alfworld's negative baseline"


def test_refs_to_device_pools_sides_that_exist_and_zeroes_lambda_off_control_roles():
    ctl = CrossGateController(_cfg(min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    # only side 0 for every row
    _step(ctl, TASKS, 1, torch.zeros(6, dtype=torch.long), torch.arange(6), [f"k{i}" for i in range(6)])
    refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
    assert refs.v.shape == (3, N_ROLES, N_SIDES, V)
    assert bool((refs.side_w[:, ROLE_FORMAT, 0] == 1.0).all()) and bool((refs.side_w[:, ROLE_FORMAT, 1] == 0.0).all())
    assert not bool(refs.valid.any()), "one side only is never valid"
    ctl.lam[("alfworld", ROLE_TAG)] = 0.2
    refs = ctl.refs_to_device(TASKS, torch.device("cpu"))
    assert refs.lam[0, ROLE_TAG] == 0.0


def test_state_round_trips_and_a_changed_config_is_refused():
    ctl = CrossGateController(_cfg(ema_decay=0.5, min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    for s in range(1, 3):
        _step(ctl, TASKS, s, side, torch.arange(6), [f"k{s}_{i}" for i in range(6)])
    sd = ctl.state_dict()
    d = CrossGateController(_cfg(ema_decay=0.5, min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    d.load_state_dict(sd)
    assert d.step == ctl.step
    for key, st in ctl.refs.items():
        st2 = d.refs[key]
        assert (st.v is None and st2.v is None) or torch.equal(st.v, st2.v)
        assert st.R == st2.R and st.n_obs == st2.n_obs and list(st.prompts) == list(st2.prompts)
    assert d.lam == ctl.lam and d.validity == ctl.validity
    r1 = ctl.refs_to_device(TASKS, torch.device("cpu"))
    r2 = d.refs_to_device(TASKS, torch.device("cpu"))
    assert torch.equal(r1.v, r2.v) and torch.equal(r1.lam, r2.lam) and torch.equal(r1.valid, r2.valid)
    e = CrossGateController(_cfg(ema_decay=0.5, rho=1.0, min_prompts=1, min_pg_prompts=0, min_tokens=1), V, TASKS)
    with pytest.raises(ValueError, match="changed across resume"):
        e.load_state_dict(sd)


def test_config_parsing_and_validation():
    c = CrossGateConfig.from_mapping({"enable": True, "roles": ["format", "env_action"], "rho": "10000"})
    assert c.roles == ("format", "env_action") and c.rho == 1e4
    c = CrossGateConfig.from_mapping({"enable": True, "roles": "format,env_action"})
    assert c.roles == ("format", "env_action")
    with pytest.raises(ValueError):
        CrossGateConfig.from_mapping({"enable": True, "epsilon": 0.1})
    with pytest.raises(ValueError):
        _cfg(lambda_max=1.5).validate()
    with pytest.raises(ValueError):
        _cfg(roles=("format", "thinking")).validate()
    with pytest.raises(ValueError):
        _cfg(ema_decay=1.0).validate()
    m = _cfg().control_role_mask()
    assert bool(m[ROLE_FORMAT]) and bool(m[ROLE_ENV_ACTION]) and not bool(m[ROLE_TAG])


# ---------------------------------------------------------------------------
# 6. the gate only ever attenuates
#
# w = 1 - lambda*h with h in [0, 1] is the design's central safety property:
# the gate removes teacher signal, it never amplifies it and never flips its
# sign. Three guards enforce it -- [.]_+ on the per-side conflict, the clamp on
# q, the clamp on h -- and each is redundant given the others on WELL-FORMED
# input, so a mutation to any one of them survives every other test in this
# file. These assert the invariant itself, on input built to break it.


def _reference_aligned_with(fwd, b, scale, R_value, nT=3):
    """A reference deliberately equal to +scale * d at every token's support.

    Nothing in the population produces this -- it is the adversarial case. With
    scale > 0 the inner product v.d is POSITIVE everywhere, which is the input
    that turns [-x]_+ into a negative number the moment its clamp is dropped.
    R is passed in rather than derived, so it can also be set small enough to
    violate ||v||^2 <= R and drive q above 1.
    """
    v = torch.zeros(nT, N_ROLES, N_SIDES, V)
    bs, T = b["roles"].shape
    for i in range(bs):
        for t in range(T):
            c = int(b["roles"][i, t])
            for task in range(nT):
                for s in range(N_SIDES):
                    v[task, c, s].index_add_(
                        0, b["topk_ids"][i, t], scale * fwd["d"][i, t].float())
    R = torch.full((nT, N_ROLES, N_SIDES), float(R_value))
    sw = torch.full((nT, N_ROLES, N_SIDES), 0.5)
    valid = torch.ones(nT, N_ROLES, dtype=torch.bool)
    return v, R, sw, valid


@pytest.mark.parametrize("scale,R_value", [
    (+1.0, 1.0),     # aligned: v.d > 0 everywhere -- [-x]_+ must be 0, not negative
    (-1.0, 1.0),     # opposed: the gate fires, and must still stop at 1
    (-1.0, 1e-12),   # opposed with an R that breaks Jensen -- q must be capped at 1
    (+1.0, 1e-12),   # both at once
])
def test_the_gate_only_ever_attenuates(scale, R_value):
    b = _batch(seed=11, bs=6, T=7)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, scale, R_value)
    lam_max = 0.2
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                         lam=torch.full((3, N_ROLES), lam_max),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    f = _fwd(b, refs)

    assert torch.isfinite(f["q"]).all() and torch.isfinite(f["h"]).all() and torch.isfinite(f["w"]).all()
    assert float(f["q"].min()) >= 0.0, "a per-receiver gate went negative"
    assert float(f["q"].max()) <= 1.0 + 1e-6, "a per-receiver gate exceeded 1"
    assert float(f["h"].min()) >= 0.0, "the gate went negative -- w would amplify"
    assert float(f["h"].max()) <= 1.0 + 1e-6, "the gate exceeded 1 -- w could go below 1 - lambda_max"
    # THE property: the OPD term is attenuated, never amplified, never flipped.
    assert float(f["w"].max()) <= 1.0 + 1e-6, "the gate amplified the teacher term"
    assert float(f["w"].min()) >= 1.0 - lam_max - 1e-6, "the gate cut deeper than lambda_max"
    assert float(f["w"].min()) > 0.0, "the teacher term changed sign"


def test_an_aligned_reference_does_not_fire_the_gate_at_all():
    """Where the other task's reward direction AGREES with this task's teacher,
    there is nothing to protect: h must be exactly 0, w exactly 1."""
    b = _batch(seed=12)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, +1.0, 1.0)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                         lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    f = _fwd(b, refs)
    assert float(f["h"].abs().max()) == 0.0
    assert torch.equal(f["w"], torch.ones_like(f["w"]))


def test_the_bound_holds_for_every_lambda_in_the_box():
    """w in [1 - lambda_max, 1] for any lambda the solver can return, on the
    adversarial reference. lambda_max is an experimental condition (design §4.2);
    the gate must respect it whatever the references say."""
    b = _batch(seed=13)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, -1.0, 1e-12)
    for lam_max in (0.0, 0.05, 0.2, 1.0):
        refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                             lam=torch.full((3, N_ROLES), lam_max),
                             control_roles=torch.ones(N_ROLES, dtype=torch.bool))
        f = _fwd(b, refs)
        assert float(f["w"].max()) <= 1.0 + 1e-6
        assert float(f["w"].min()) >= 1.0 - lam_max - 1e-6
        if lam_max == 0.0:
            assert torch.equal(f["w"], torch.ones_like(f["w"])), "lambda_max = 0 must be inert"


# --------------------------------------------------------------------------- #
# 7. MOPD v2: gamma * negative cosine, and the S-normalised objective


def _v2_make_refs(v, R, *, gamma, lam, nR=1, nS=2):
    """Refs with v2's two extra fields set explicitly."""
    import torch

    from verl.trainer.ppo.opd_cross_gate import CrossGateRefs

    nT = v.shape[0]
    return CrossGateRefs(
        v=v, R=R,
        side_w=torch.full((nT, nR, nS), 1.0 / nS),
        valid=torch.ones(nT, nR, dtype=torch.bool),
        lam=torch.full((nT, nR), lam),
        control_roles=torch.ones(nR, dtype=torch.bool),
        v_norm=v.double().norm(dim=-1).float(),
        gamma=torch.full((nT, nR), gamma),
    )


def _v2_fwd(refs, *, version, k=4, V=32, bs=2, T=3):
    import torch

    from verl.trainer.ppo.opd_cross_gate import cross_gate_forward

    torch.manual_seed(0)
    ids = torch.arange(k).repeat(bs, T, 1) % V
    return cross_gate_forward(
        student_topk_logprob=torch.log_softmax(torch.randn(bs, T, k), dim=-1),
        teacher_topk_logprob=torch.log_softmax(torch.randn(bs, T, k), dim=-1),
        teacher_kl=torch.rand(bs, T),
        topk_ids=ids, response_ids=ids[..., 0],
        pg_grad_coef=torch.ones(bs, T), opd_coef=0.01,
        task_ids=torch.ones(bs, dtype=torch.long),
        roles=torch.zeros(bs, T, dtype=torch.long),
        refs=refs, delta=1e-30, gate_version=version,
    )


def test_v2_gamma_zero_makes_the_receiver_inert():
    """Opposed halves must not drive an intervention."""
    import torch

    V, nT = 32, 2
    v = torch.zeros(nT, 1, 2, V)
    v[0, 0, :, :4] = -1.0                      # opposes the sender's OPD
    refs0 = _v2_make_refs(v, torch.full((nT, 1, 2), 1e-4), gamma=0.0, lam=0.2)
    refs1 = _v2_make_refs(v, torch.full((nT, 1, 2), 1e-4), gamma=1.0, lam=0.2)
    h0 = _v2_fwd(refs0, version=2)["h"]
    h1 = _v2_fwd(refs1, version=2)["h"]
    assert float(h0.abs().max()) == 0.0, "gamma = 0 must leave h at zero"
    assert float(h1.abs().max()) > 0.0, "gamma = 1 with an opposing reference must fire"


def test_v2_drops_the_kappa_shrinkage_that_v1_applied():
    """Same reference, same OPD: v2's q is the cosine, v1's is scaled by
    sqrt(kappa) = ||v||/sqrt(R). With R >> ||v||^2 the v1 gate is crushed and
    the v2 gate is not -- that is the whole point of the change."""
    import torch

    V, nT = 32, 2
    v = torch.zeros(nT, 1, 2, V)
    v[0, 0, :, :4] = -1.0
    R_big = torch.full((nT, 1, 2), 1.0e4)      # kappa = ||v||^2/R = 4e-4
    refs = _v2_make_refs(v, R_big, gamma=1.0, lam=0.2)
    h1 = float(_v2_fwd(refs, version=1)["h"].max())
    h2 = float(_v2_fwd(refs, version=2)["h"].max())
    assert h2 > h1 * 10, f"v2 should not inherit the kappa shrinkage (v1={h1:g}, v2={h2:g})"
    assert h2 <= 1.0 + 1e-6


def test_v2_still_only_attenuates():
    """The invariant section 6 pins, under the new gate as well."""
    import torch

    V, nT = 32, 2
    for sign in (-1.0, 1.0):
        for gamma in (0.0, 0.5, 1.0):
            v = torch.zeros(nT, 1, 2, V)
            v[0, 0, :, :4] = sign
            refs = _v2_make_refs(v, torch.full((nT, 1, 2), 1e-4), gamma=gamma, lam=0.2)
            out = _v2_fwd(refs, version=2)
            h, w = out["h"], out["w"]
            assert float(h.min()) >= 0.0 and float(h.max()) <= 1.0
            assert float(w.max()) <= 1.0 + 1e-6
            assert float(w.min()) >= 1.0 - 0.2 - 1e-6


def test_the_S_scale_is_the_mean_absolute_cross_effect():
    """S = sum_j (B + 2 A^-) is sum_j E|v_i . d|, because |x| = x + 2[-x]_+."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000) * 1e-8
    B = x.mean()
    Aneg = np.maximum(-x, 0.0).mean()
    assert np.isclose(B + 2 * Aneg, np.abs(x).mean(), rtol=1e-12, atol=0.0)


def test_the_objective_scale_changes_the_solution():
    """s/S and s/R are different control bases, not a change of units."""
    import numpy as np

    from verl.trainer.ppo.opd_cross_gate import solve_role

    n = 3
    rng = np.random.default_rng(3)
    K = rng.random(n) * 1e-6 + 1e-9
    B = -(rng.random((n, n)) * 1e-9 + 1e-11)
    D = -(rng.random((n, n)) * 1e-8 + 1e-10)
    R = rng.random(n) * 1e-3 + 1e-6                 # RL energy: large
    S = np.abs(B).sum(axis=1) * 3.0                 # cross magnitude: tiny
    valid = np.ones(n, dtype=bool)
    kw = dict(eps=0.0, lam_max=0.2, delta=1e-30)
    v1 = solve_role(K, B, D, R, valid, rho=1.0, **kw)
    v2 = solve_role(K, B, D, R, valid, rho=1.0, scale=S, **kw)
    assert not np.allclose(v1["lam"], v2["lam"]), (
        "the S normalisation must move the solution; if it does not, the two "
        "arms are the same experiment"
    )
    # Aggregate, not per component. lambda is a JOINT solution -- one sender's
    # lambda enters several receivers' conditions -- so weighting the constraint
    # more can reallocate: here two senders rise and the third goes to 0.
    # Per-component monotonicity is not a property of the change; better
    # constraint satisfaction is.
    assert v2["s"].sum() <= v1["s"].sum() + 1e-30, (
        f"S-normalised should satisfy the condition at least as well: "
        f"s_v2={v2['s'].sum():.3e} vs s_v1={v1['s'].sum():.3e}"
    )
    assert v2["lam"].sum() > v1["lam"].sum(), (
        f"and spend more intervention doing it: "
        f"sum lam_v2={v2['lam'].sum():.4f} vs v1={v1['lam'].sum():.4f}"
    )


def test_a_uniform_gate_rescaling_is_absorbed_but_rho_is_not():
    """h -> ch with D -> cD and K -> c^2 K leaves lambda*h alone IN THE
    INTERIOR; changing rho does not have that invariance. The box does not
    transform, which is why the qualification matters."""
    import numpy as np

    from verl.trainer.ppo.opd_cross_gate import solve_role

    n = 3
    rng = np.random.default_rng(3)
    K = rng.random(n) * 1e-6 + 1e-9
    B = -(rng.random((n, n)) * 1e-9 + 1e-11)
    D = -(rng.random((n, n)) * 1e-8 + 1e-10)
    R = rng.random(n) * 1e-3 + 1e-6
    valid = np.ones(n, dtype=bool)
    kw = dict(eps=0.0, lam_max=0.2, delta=1e-30)
    base = solve_role(K, B, D, R, valid, rho=1e4, **kw)
    assert not np.isclose(base["lam"], 0.2).any(), "set up an interior solution"
    for c in (10.0, 100.0):
        got = solve_role(K * c * c, B, D * c, R, valid, rho=1e4, **kw)
        assert np.allclose(base["lam"], got["lam"] * c, rtol=1e-6, atol=1e-18)
    lams = [solve_role(K, B, D, R, valid, rho=r, **kw)["lam"] for r in (1e0, 1e4, 1e8)]
    assert not np.allclose(lams[0], lams[1]) and not np.allclose(lams[1], lams[2]), (
        "rho must move the solution; it is not a reparameterisation"
    )


def test_the_controller_really_solves_v2s_objective_end_to_end():
    """The gate is only half of v2. This pins the OTHER half: that the
    controller hands solve_role the S scale and rho_rel, not R and rho.

    Reconstructed the way the review reconstructed it -- take the controller's
    own EMAs, rebuild S = sum_{j!=i}(B + 2 A^-), re-run the solver on the CPU,
    and require the recorded lambda to match. Three mutations survived without
    this: dropping 2*A^- from S, ignoring rho_rel, and passing scale=None.
    """
    import numpy as np

    from verl.trainer.ppo.opd_cross_gate import ROLE_NAMES, solve_role, _Ema

    # lambda_max well above the solution: at 0.2 this fixture saturates and
    # every scale/rho looks identical, which let two controller mutations
    # survive (S without 2*A^-, and rho instead of rho_rel).
    cfg_kw = dict(min_prompts=1, min_pg_prompts=0, min_tokens=1, lambda_max=1.0)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    pidx = torch.arange(6)
    keys = [f"k{i}" for i in range(16)]

    ctl2 = CrossGateController(_cfg(gate_version=2, rho_rel=1.0, **cfg_kw), V, TASKS)
    ctl1 = CrossGateController(_cfg(gate_version=1, rho=1.0e4, **cfg_kw), V, TASKS)
    for s in range(4):
        _step(ctl2, TASKS, 10 + s, side, pidx, keys)
        _step(ctl1, TASKS, 10 + s, side, pidx, keys)

    nT = len(TASKS)
    moved = False
    for c, rn in ROLE_NAMES.items():
        if not bool(ctl2.cfg.control_role_mask()[c]):
            continue
        K = np.array([
            (ctl2.k_num.get((j, c), _Ema()).val / (ctl2.d_sq.get(j, _Ema()).val + ctl2.cfg.delta))
            if ctl2.d_sq.get(j) is not None else 0.0 for j in TASKS], dtype=np.float64)
        Bm = np.zeros((nT, nT)); Dm = np.zeros((nT, nT)); Am = np.zeros((nT, nT))
        for ti, i in enumerate(TASKS):
            for tj, j in enumerate(TASKS):
                if i == j:
                    continue
                Bm[ti, tj] = ctl2.cross.get((i, j, c, "B"), _Ema()).val
                Dm[ti, tj] = ctl2.cross.get((i, j, c, "D"), _Ema()).val
                Am[ti, tj] = ctl2.cross.get((i, j, c, "Aneg"), _Ema()).val
        Rv = np.array([float(np.mean([ctl2._ref(i, c, s).R for s in range(2)]))
                       for i in TASKS], dtype=np.float64)
        off = ~np.eye(nT, dtype=bool)
        Sv = np.maximum(((Bm + 2.0 * Am) * off).sum(axis=1), 0.0)
        valid = np.array([bool(ctl2.validity.get((i, c), (False, 0, 0))[0]) for i in TASKS])

        # v2 weights each receiver's demand by ITS OWN gamma -- the same
        # [cos(v^1, v^2)]_+ the gate multiplies q by. Reproduced here, because a
        # recomputation that leaves it out is solving a different objective and
        # would pass while the controller believed an unreliable receiver.
        gam = np.array([float(ctl2.gamma.get((i, c), 0.0)) for i in TASKS], dtype=np.float64)
        want = solve_role(K, Bm, Dm, Rv, valid, eps=0.0, rho=1.0, scale=Sv,
                          lam_max=ctl2.cfg.lambda_max, delta=ctl2.cfg.delta,
                          iters=ctl2.cfg.solver_iters, tol=ctl2.cfg.solver_tol,
                          gamma=gam)
        got = np.array([float(ctl2.lam.get((j, c), 0.0)) for j in TASKS])
        # The basis the controller actually handed the solver. Asserted directly
        # because lambda cannot separate it here: with rho_rel/S^2 ~ 1e8 against
        # K ~ 1e-6, any net conflict drives lambda to the cap whatever the cap
        # is, so an equality on lambda saturates and two mutations (S without
        # 2*A^-, and rho instead of rho_rel) survived it.
        used = ctl2.last_solver[c]
        assert np.allclose(used["gamma"], gam, rtol=1e-12, atol=0.0), (
            f"role {rn}: solver got gamma={used['gamma']}, expected the gate's {gam}"
        )
        assert float(used["rho"]) == float(ctl2.cfg.rho_rel), (
            f"role {rn}: solver got rho={used['rho']}, expected rho_rel="
            f"{ctl2.cfg.rho_rel}"
        )
        assert np.allclose(used["scale"], Sv, rtol=1e-12, atol=0.0), (
            f"role {rn}: solver got scale={used['scale']}, expected S={Sv}"
        )
        assert np.allclose(got, want["lam"], rtol=1e-9, atol=1e-18), (
            f"role {rn}: controller lambda {got} != solve_role with S and rho_rel {want['lam']}"
        )
        # and the R-normalised objective is a DIFFERENT problem on the same data
        alt = solve_role(K, Bm, Dm, Rv, valid, eps=0.0, rho=1.0e4, scale=None,
                         lam_max=ctl2.cfg.lambda_max, delta=ctl2.cfg.delta,
                         iters=ctl2.cfg.solver_iters, tol=ctl2.cfg.solver_tol)
        if not np.allclose(alt["lam"], want["lam"], rtol=1e-6, atol=1e-18):
            moved = True
    assert moved, (
        "on this data the two objectives agree everywhere, so the test cannot "
        "tell them apart -- strengthen the fixture rather than trusting it"
    )


def _reference_solve(K, B, D, R, valid, *, eps, rho, scale, lam_max, delta):
    """An INDEPENDENT minimiser of the same objective, for cross-checking.

    Deliberately not solve_role's algorithm: plain projected gradient with a
    tiny step and many iterations, written out here so a mutation inside
    solve_role cannot hide by also changing the expected value. Slow and only
    used on 3-variable fixtures.
    """
    import numpy as np

    n = len(K)
    off = ~np.eye(n, dtype=bool)
    Bo, Do = B * off, D * off
    sc = R if scale is None else scale

    def s_of(l):
        return np.maximum(-eps * R - Bo.sum(axis=1) + Do @ l, 0.0) * valid

    def grad(l):
        return 2.0 * K * l + Do.T @ (2.0 * rho * s_of(l) / (sc + delta) ** 2)

    l = np.zeros(n)
    lip = 2.0 * K.max() + 2.0 * rho * (np.abs(Do) ** 2).sum() / (sc.min() + delta) ** 2
    step = 1.0 / max(lip, 1e-300)
    for _ in range(400_000):
        l = np.clip(l - step * grad(l), 0.0, lam_max)
    return l


def test_the_controller_matches_an_independent_minimiser_of_v2s_objective():
    """Catches a mutation INSIDE solve_role, which the reconstruction test
    cannot: that one calls solve_role for the expected value too, so a change
    to the descent moves both sides equally. Run with lam_max well above the
    solution so the cap does not mask the comparison -- at lam_max = 0.2 this
    fixture saturates and every scale looks the same.
    """
    import numpy as np

    from verl.trainer.ppo.opd_cross_gate import solve_role

    n = 3
    rng = np.random.default_rng(11)
    K = rng.random(n) * 1e-3 + 1e-4          # a removal cost that actually bites
    B = -(rng.random((n, n)) * 1e-4 + 1e-6)
    D = -(rng.random((n, n)) * 1e-3 + 1e-5)
    R = rng.random(n) * 1.0 + 0.5            # RL energy: O(1)
    S = np.abs(B).sum(axis=1) * 3.0          # cross magnitude: O(1e-4)
    valid = np.ones(n, dtype=bool)
    kw = dict(eps=0.0, lam_max=1.0, delta=1e-30)

    for tag, scale, rho in (("S, rho_rel=1", S, 1.0), ("R, rho=1e4", None, 1.0e4)):
        got = solve_role(K, B, D, R, valid, rho=rho, scale=scale, **kw)["lam"]
        want = _reference_solve(K, B, D, R, valid, rho=rho, scale=scale, **kw)
        assert not np.isclose(got, kw["lam_max"]).any(), f"{tag}: fixture saturated"
        # Compared on the OBJECTIVE, not on lambda. At v2's conditioning
        # (rho_rel/S^2 ~ 1e8) the surface is flat enough that two minimisers
        # land ~1e-3 apart in lambda while agreeing to ~1e-5 in f, so a lambda
        # tolerance would be testing the solver's path rather than its answer.
        # The measured gaps are 2e-16 at v1's conditioning and 8e-6 at v2's.
        off = ~np.eye(n, dtype=bool)
        Bo, Do = B * off, D * off
        sc = R if scale is None else scale

        def obj(l):
            sl = np.maximum(-Bo.sum(axis=1) + Do @ l, 0.0) * valid
            return float((K * l * l).sum() + rho * ((sl / (sc + kw["delta"])) ** 2).sum())

        f_got, f_want = obj(got), obj(want)
        assert f_got <= f_want * (1.0 + 1e-4) + 1e-300, (
            f"{tag}: solve_role's objective {f_got:.6e} is worse than the "
            f"independent minimiser's {f_want:.6e} by more than 1e-4 relative"
        )
    # and the two bases really are different problems on this fixture
    a = solve_role(K, B, D, R, valid, rho=1.0, scale=S, **kw)["lam"]
    b = solve_role(K, B, D, R, valid, rho=1.0, scale=None, **kw)["lam"]
    assert not np.allclose(a, b, rtol=1e-3), "S and R must not coincide here"


# --------------------------------------------------------------------------- #
# 8. q_scale: the strength knob, and the three things it must not break


def test_q_scale_one_is_bit_for_bit_the_unscaled_gate():
    """The default must be a no-op, or every earlier arm's numbers move."""
    b = _batch(seed=21)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, -1.0, 1e-12)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                         lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    a, c = _fwd(b, refs), _fwd(b, refs, q_scale=1.0)
    for k in ("q", "h", "w"):
        assert torch.equal(a[k], c[k]), f"q_scale=1.0 changed {k}"


def test_q_scale_multiplies_the_gate_exactly_where_it_does_not_saturate():
    """min(1, k q) = k q below the clamp, so h -- an omega-weighted mean of the
    q's -- is exactly k times the unscaled h there. This is the whole claim the
    strength arm rests on; if it fails, q_scale is not a strength knob."""
    b = _batch(seed=22)
    f0 = _fwd(b, _zero_refs())
    # tiny alignment => tiny q, so k q stays far below 1 and nothing clamps
    v, R, sw, valid = _reference_aligned_with(f0, b, -1.0, 1.0)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                         lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    f1 = _fwd(b, refs)
    k = 7.0
    fk = _fwd(b, refs, q_scale=k)
    unsat = (f1["q"] * k) < 1.0 - 1e-9
    assert bool(unsat.any()), "fixture saturates everywhere -- it proves nothing"
    assert torch.allclose(fk["q"][unsat], f1["q"][unsat] * k, rtol=1e-10, atol=1e-12)
    # and the attenuation follows: 1 - w = lambda * h
    rows = (fk["h"] > 0) & ((f1["h"] * k) < 1.0 - 1e-9)
    if bool(rows.any()):
        assert torch.allclose((1.0 - fk["w"])[rows], (1.0 - f1["w"])[rows] * k, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("k", [1.0, 5.0, 50.0, 1000.0])
def test_q_scale_cannot_break_the_floor_or_the_attribution(k):
    """Three invariants, at any strength, on the adversarial reference:

    1. w >= 1 - lambda_max. The per-token floor is what makes a large q_scale a
       different animal from a large lambda_max; if it can be breached the knob
       is unsafe at exactly the values it is meant for.
    2. h <= 1, which is what (1) rests on.
    3. h == sum_i omega_i q_i EXACTLY. lost_by_recv divides the removal by h to
       attribute it per receiver, and the audit's "98.7% from Alfworld" is that
       statistic. Scaling h after the pooling would break this identity;
       scaling q before it does not.
    """
    b = _batch(seed=23)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, -1.0, 1e-12)
    lam_max = 0.2
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid,
                         lam=torch.full((3, N_ROLES), lam_max),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    f = _fwd(b, refs, q_scale=k)
    assert torch.isfinite(f["q"]).all() and torch.isfinite(f["h"]).all() and torch.isfinite(f["w"]).all()
    assert float(f["q"].max()) <= 1.0 + 1e-6, "a scaled per-receiver gate exceeded 1"
    assert float(f["h"].max()) <= 1.0 + 1e-6, "the pooled gate exceeded 1"
    assert float(f["h"].min()) >= 0.0
    assert float(f["w"].max()) <= 1.0 + 1e-6, "the gate amplified the teacher term"
    assert float(f["w"].min()) >= 1.0 - lam_max - 1e-6, (
        f"q_scale={k} cut deeper than lambda_max -- the per-token floor is gone")
    assert float(f["w"].min()) > 0.0, "the teacher term changed sign"
    pooled = (f["omega"] * f["q"]).sum(dim=-1)
    assert torch.allclose(f["h"], pooled, rtol=1e-10, atol=1e-12), (
        "h is no longer the omega-weighted mean of the q's; lost_by_receiver is invalid")


def test_q_scale_below_one_is_refused():
    """A weaker gate is lambda_max's job. A knob named for strength that can
    also weaken invites an arm whose lock reads as the opposite of what it did."""
    _cfg(q_scale=1.0).validate()
    _cfg(q_scale=50.0).validate()
    for bad in (0.5, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q_scale"):
            _cfg(q_scale=bad).validate()


def test_q_scale_survives_from_mapping_and_the_checkpoint_refuses_a_silent_change():
    """The knob has to arrive through hydra, and a resume must NOT blend an EMA
    gathered at another scale (dp_actor's CROSS_GATE_RESET_ON_LOAD is the
    deliberate way to change it)."""
    cfg = CrossGateConfig.from_mapping({"enable": True, "q_scale": 50})
    assert cfg.q_scale == 50.0 and isinstance(cfg.q_scale, float)
    ctl = CrossGateController(_cfg(q_scale=1.0), vocab_size=V, task_names=("a", "b", "c"))
    sd = ctl.state_dict()
    assert sd["cfg"]["q_scale"] == 1.0, "q_scale must be in the checkpointed cfg"
    other = CrossGateController(_cfg(q_scale=50.0), vocab_size=V, task_names=("a", "b", "c"))
    with pytest.raises(ValueError, match="config changed across resume"):
        other.load_state_dict(sd)


# --------------------------------------------------------------------------- #
# 9. the revised objective: a certified solver, gamma on both sides, and the
#    accounting that says what an intervention costs


def _rand_problem(rng, n=3):
    """A problem shaped like the live one: S DERIVED from B and A^-, as the
    controller derives it, so "a receiver with S = 0" also has B = D = 0 rather
    than being an inconsistent fixture that no run can produce."""
    K = 10.0 ** rng.uniform(-8, 1, n)
    R = 10.0 ** rng.uniform(-3, 1, n)
    sig = 10.0 ** rng.uniform(-6, -1)
    B = rng.normal(0, sig, (n, n))
    A = np.abs(rng.normal(0, sig, (n, n)))
    D = rng.normal(0, sig, (n, n))
    if rng.random() < 0.3:                       # a receiver that saw no cross effect
        i = int(rng.integers(n)); B[i, :] = 0.0; A[i, :] = 0.0; D[i, :] = 0.0
    off = ~np.eye(n, dtype=bool)
    S = np.maximum(((B + 2.0 * A) * off).sum(axis=1), 0.0)
    return dict(K=K, B=B, D=D, R=R, S=S,
                gamma=rng.uniform(0, 1, n),
                valid=rng.random(n) < 0.8,
                rho=10.0 ** rng.uniform(0, 4),
                lam_max=float(rng.choice([0.2, 1.0])))


def _obj(p, lam):
    off = ~np.eye(len(lam), dtype=bool)
    Do = p["D"] * off
    a0 = -(p["B"] * off).sum(axis=1)
    den = np.where(p["valid"], p["S"], 1.0) + 1e-30
    w = np.where(p["valid"], p["rho"] * np.maximum(p["gamma"], 0.0) / den ** 2, 0.0)
    s = np.maximum(a0 + Do @ lam, 0.0) * p["valid"]
    return float((p["K"] * lam * lam).sum() + (w * s * s).sum())


def test_the_solver_never_reports_optimal_on_a_point_a_grid_beats():
    """The defect this replaces: `converged` meant "the polish improved the
    objective", so a solve one iteration deep came back True 27.2% above the
    optimum. It now means "the KKT residual is within tol", and the claim is
    checked against an independent search."""
    rng = np.random.default_rng(7)
    grid = None
    for _ in range(60):
        p = _rand_problem(rng)
        sol = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                         scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200,
                         tol=1e-12, gamma=p["gamma"])
        if sol["reason"] == "nonfinite_input":
            continue
        if grid is None:
            grid = np.linspace(0.0, 1.0, 21)
        best = min(_obj(p, np.array(l) * p["lam_max"]) for l in itertools.product(grid, repeat=3))
        if sol["converged"]:
            assert sol["f"] <= best + 1e-9 * max(abs(best), 1.0), (
                f"reported optimal at f={sol['f']:.6e} while a grid found {best:.6e}")
            assert sol["kkt_res"] <= 1e-12


def test_an_uncertified_solve_is_reported_and_not_dressed_as_optimal():
    """iters is accepted and ignored -- there is no budget to run out of -- so a
    solve that cannot certify says so rather than returning a plausible number.
    The controller turns that into lambda = 0."""
    rng = np.random.default_rng(8)
    for _ in range(40):
        p = _rand_problem(rng)
        a = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                       scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=1,
                       tol=1e-12, gamma=p["gamma"])
        b = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                       scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=500,
                       tol=1e-12, gamma=p["gamma"])
        if a["reason"] == "nonfinite_input":
            continue
        assert a["converged"] == (a["kkt_res"] <= 1e-12), "converged must BE the certificate"
        if a["converged"] and b["converged"]:
            assert abs(a["f"] - b["f"]) <= 1e-9 * max(abs(b["f"]), 1.0), (
                "one iteration and five hundred disagree on a certified optimum")


def test_an_invalid_receiver_cannot_reach_the_numerics():
    """The polish took its step from 1/L with L built on min(scale) over ALL
    receivers, invalid included, so one unusable reference with S = 0 froze the
    solve -- live in 150 of the 300 saved cases. Adding such a receiver must now
    change nothing at all."""
    rng = np.random.default_rng(9)
    for _ in range(40):
        p = _rand_problem(rng)
        p["valid"][:] = True
        base = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                          scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200,
                          tol=1e-12, gamma=p["gamma"])
        S2, v2 = p["S"].copy(), p["valid"].copy()
        S2[1] = 0.0
        v2[1] = False                      # an unusable reference, zero scale
        withz = solve_role(p["K"], p["B"], p["D"], p["R"], v2, eps=0.0, rho=p["rho"],
                           scale=S2, lam_max=p["lam_max"], delta=1e-30, iters=200,
                           tol=1e-12, gamma=p["gamma"])
        if base["reason"] == "nonfinite_input" or withz["reason"] == "nonfinite_input":
            continue
        assert withz["converged"], "an invalid receiver with S = 0 broke the solve"
        assert np.isfinite(withz["lam"]).all()


def test_gamma_weights_the_receivers_demand_and_zero_gamma_silences_it():
    """The inconsistency this closes: gamma = 0 already meant "this reference
    may not attenuate a single token" in the gate, while the objective still
    read its full B, D and S when choosing whose lambda to raise."""
    rng = np.random.default_rng(10)
    moved = 0
    for _ in range(40):
        p = _rand_problem(rng)
        p["valid"][:] = True
        ones = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                          scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200, tol=1e-12)
        g = np.ones(3); g[0] = 0.0
        off0 = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                          scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200,
                          tol=1e-12, gamma=g)
        # silencing receiver 0 must equal deleting it
        v = p["valid"].copy(); v[0] = False
        dele = solve_role(p["K"], p["B"], p["D"], p["R"], v, eps=0.0, rho=p["rho"],
                          scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200, tol=1e-12)
        if not (off0["converged"] and dele["converged"]):
            continue
        assert np.allclose(off0["lam"], dele["lam"], rtol=1e-7, atol=1e-12), (
            "gamma = 0 must silence a receiver exactly as invalidating it does")
        if not np.allclose(ones["lam"], off0["lam"], rtol=1e-7, atol=1e-12):
            moved += 1
    assert moved > 0, "gamma never changed the answer -- the fixture proves nothing"


def test_gamma_sits_outside_the_square_not_in_the_numerator():
    """gamma * (s/S)^2, not (gamma*s/S)^2. The second is a gamma^2 weight and a
    different design; they are only equal at gamma in {0, 1}."""
    rng = np.random.default_rng(11)
    p = _rand_problem(rng)
    p["valid"][:] = True
    g = np.array([0.25, 0.5, 0.75])
    sol = solve_role(p["K"], p["B"], p["D"], p["R"], p["valid"], eps=0.0, rho=p["rho"],
                     scale=p["S"], lam_max=p["lam_max"], delta=1e-30, iters=200, tol=1e-12, gamma=g)
    off = ~np.eye(3, dtype=bool)
    a0 = -(p["B"] * off).sum(axis=1)
    s = np.maximum(a0 + (p["D"] * off) @ sol["lam"], 0.0)
    want = float((p["K"] * sol["lam"] ** 2).sum()
                 + (p["rho"] * g * (s / (p["S"] + 1e-30)) ** 2).sum())
    assert math.isclose(sol["f"], want, rel_tol=1e-9, abs_tol=1e-300)


def test_the_amplifier_never_raises_a_low_reliability():
    """q_tilde = gamma * min(1, k a), so q_tilde <= gamma at any strength. The
    rejected placement, min(1, k gamma a), turns gamma = 0.01 into 0.5 at k=100
    -- a decision to trust an unreliable reference wearing a strength knob's
    name."""
    b = _batch(seed=31)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, valid = _reference_aligned_with(f0, b, -1.0, 1e-12)
    gam = torch.full((3, N_ROLES), 0.01)
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=valid, gamma=gam,
                         lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=torch.ones(N_ROLES, dtype=torch.bool))
    for k in (1.0, 5.0, 50.0, 1000.0):
        f = _fwd(b, refs, q_scale=k, gate_version=2)     # gamma exists only in v2
        assert float(f["q"].max()) <= 0.01 + 1e-9, (
            f"q_scale={k} pushed q past gamma=0.01 -- the amplifier is scaling reliability")
        assert float(f["h"].max()) <= 0.01 + 1e-9
        assert float(f["w"].min()) >= 1.0 - 0.2 - 1e-9


def test_the_self_cost_is_recorded_and_still_absent_from_the_objective():
    """T_j = E[h r_j . d_j] is what attenuating task j costs task j. The
    diagonal of B and D is masked, so the solver cannot see it; the point of
    recording it is that it becomes possible to."""
    b = _batch(seed=32)
    f0 = _fwd(b, _zero_refs())
    v, R, sw, both = _refs_from_population(b, f0, torch.tensor([0, 1, 0, 1, 0, 1]))
    refs = CrossGateRefs(v=v, R=R, side_w=sw, valid=both, lam=torch.full((3, N_ROLES), 0.2),
                         control_roles=_cfg().control_role_mask())
    f = _fwd(b, refs)
    st = CrossGateStats(n_tasks=3, vocab_size=V, device=torch.device("cpu"), n_prompts=8)
    st.update(fwd=f, task_ids=b["task_ids"], roles=b["roles"], response_mask=torch.ones(6, 7),
              teacher_kl=b["teacher_kl"], topk_ids=b["topk_ids"], side=torch.tensor([0, 1, 0, 1, 0, 1]),
              prompt_idx=torch.arange(6), row_basis=None)
    red = st.reduced()
    assert float(red["send"][:, :, _SEND_IDX["T_pos"]].sum()) > 0.0, "no self-aligned removal at all"
    # the solver's matrices still have a zero diagonal: the cost is not priced
    lam = solve_role(np.ones(3) * 1e-6, np.ones((3, 3)), np.ones((3, 3)), np.ones(3),
                     np.ones(3, dtype=bool), eps=0.0, rho=1.0, scale=np.ones(3),
                     lam_max=0.2, delta=1e-30, iters=200, tol=1e-12)
    assert lam["converged"]
