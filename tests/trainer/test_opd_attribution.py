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
"""The arm-independent half of the cross-teacher instrumentation.

The property every test here is protecting is one sentence: these columns must
come out IDENTICAL whether the arm that produced them applied a weight or not.
That is what makes "which tokens does the distillation term push hardest" a
comparison between the weighted arm and its control rather than two tables that
happen to share column names.

The control cannot be measured any other way. It runs with
cross_teacher_kl_weight.enable=False, which stops the driver loading the base
policy and caching the off-task teachers -- so its corroboration, its pair
evidence and its channel partition are not weight-free quantities that happen to
be missing, their INPUTS do not exist. What it does have is the student's own
top-k and the on-task teacher's log-probs at it, which is exactly what
opd_logit_push reads.
"""

import re

import pytest
import torch

from verl.trainer.ppo.cross_teacher_kl_weight import (
    OPD_ROLE_CUT_SUFFIXES,
    OPD_TERMS,
    LogitPushTokens,
    PositionScopeTermStats,
    gradient_metrics,
    logit_gradient_terms,
    opd_attribution_metrics,
    opd_attribution_terms,
    opd_logit_push,
    select_metrics,
)
from verl.trainer.ppo.sign_weights import ScopeTermStats

TASKS = ["alfworld", "search", "webshop"]


def _batch(seed: int = 0, bs: int = 4, resp: int = 6, k: int = 5, vocab: int = 17):
    """Student / teacher / KL on a support, plus the ids and masks around them."""
    g = torch.Generator().manual_seed(seed)
    # -1.0 so the top-k does not carry the whole distribution: a non-empty tail
    # is what makes g0_tail and the coverage shares mean anything.
    student = torch.log_softmax(torch.randn(bs, resp, k, generator=g), dim=-1) - 1.0
    teacher = torch.log_softmax(torch.randn(bs, resp, k, generator=g), dim=-1) - 1.0
    kl = torch.rand(bs, resp, generator=g) * 0.4
    ids = torch.randint(0, vocab, (bs, resp, k), generator=g)
    mask = torch.ones(bs, resp)
    mask[-1, -2:] = 0.0                       # a masked tail, so masking is exercised
    tasks = torch.tensor([0, 1, 2, 0])
    return dict(
        student=student, teacher=teacher, kl=kl, ids=ids, mask=mask, tasks=tasks,
        vocab=vocab, k=k, bs=bs, resp=resp,
    )


def _fold(b, weight, *, top_n=6, gap=True):
    push = opd_logit_push(
        student_logprob=b["student"], teacher_logprob=b["teacher"],
        teacher_kl=b["kl"], coef=0.01,
    )
    table = LogitPushTokens(
        vocab_size=b["vocab"], n_tasks=len(TASKS), device="cpu", top_n=top_n,
    )
    table.update(
        support_ids=b["ids"], g0=push["g0"], weight=weight, coef_applied_weight=weight,
        response_mask=b["mask"], task_ids=b["tasks"],
        sampled_onehot=torch.zeros_like(push["g0"]),
        p_student=push["p_student"], g0_tail=push["g0_tail"],
        gap=push["gap"] if gap else None,
    )
    return table, push


def _weight(b, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return 0.6 + 1.4 * torch.rand(b["bs"], b["resp"], generator=g)


WEIGHT_FREE = ("base", "base_abs", "kl_mass", "kl_mass_abs")


# --------------------------------------------------------------------------- #
# The invariant the whole family rests on
# --------------------------------------------------------------------------- #
def _class_summed(table):
    """(scope, token, term). The class axis is where W enters the LAYOUT: a
    token's occurrences are filed by push_direction_class, which reads W, so the
    same nats sit in different rows in the two arms. Summing it out is not a
    convenience -- it is the operation that makes these columns comparable, and
    it is what _base_rows and the weight-free scalars both do."""
    K = len(LogitPushTokens.TERMS)
    return table.buf.view(table.n_scopes, table.n_classes, table.vocab_size, K).sum(dim=1)


def test_weight_free_columns_do_not_move_when_the_weight_does():
    """Equal to float64 noise, not bit-equal, and the gap is not a defect: W
    decides which of the four class rows each occurrence is added into, so the
    two arms reach the same total by summing the same terms in a different
    order. index_add_ promises no order either. A leak of W into these columns
    would be a relative difference of percent, not of 1e-15."""
    b = _batch()
    ones = torch.ones(b["bs"], b["resp"])
    w = _weight(b)
    t1, _ = _fold(b, ones)
    t2, _ = _fold(b, w)
    idx = {t: i for i, t in enumerate(LogitPushTokens.TERMS)}
    a, c = _class_summed(t1), _class_summed(t2)
    for name in WEIGHT_FREE:
        x, y = a[..., idx[name]], c[..., idx[name]]
        scale = float(x.abs().max())
        assert float((x - y).abs().max()) <= 8e-16 * max(scale, 1e-30), f"{name} moved when W did"
    # and the weighted ones did move by a real amount, or the above proves nothing
    ex, ey = a[..., idx["extra_abs"]], c[..., idx["extra_abs"]]
    assert float((ex - ey).abs().max()) > 1e-6 * float(ey.abs().max())


def test_the_class_axis_is_the_one_thing_a_weight_reshuffles():
    """W does not change a token's unweighted push; it changes which cell that
    push is filed in. A ranking taken per class would compare an arm's four
    partial views against a control's two."""
    b = _batch()
    t1, _ = _fold(b, torch.ones(b["bs"], b["resp"]))
    t2, _ = _fold(b, _weight(b))
    idx = {t: i for i, t in enumerate(LogitPushTokens.TERMS)}
    K = len(LogitPushTokens.TERMS)
    raw1 = t1.buf.view(t1.n_scopes, t1.n_classes, t1.vocab_size, K)[..., idx["base_abs"]]
    raw2 = t2.buf.view(t2.n_scopes, t2.n_classes, t2.vocab_size, K)[..., idx["base_abs"]]
    assert not torch.equal(raw1, raw2), "the fixture never crosses a class boundary"
    assert torch.equal(raw1.sum(dim=1), raw2.sum(dim=1))
    # a control fills only the two "damped" cells, because 1 > 1 is false
    live = {i for i in range(t1.n_classes) if float(raw1[0, i].sum()) > 0}
    assert live == {0, 2}


def test_weight_free_scalars_and_rankings_match_across_arms():
    b = _batch(seed=3)
    ones = torch.ones(b["bs"], b["resp"])
    m1 = _fold(b, ones)[0].scalar_metrics(task_names=TASKS, prefix="opd")
    m2 = _fold(b, _weight(b))[0].scalar_metrics(task_names=TASKS, prefix="opd")
    shared = [k for k in m1 if any(t in k for t in ("base", "kl_mass"))]
    assert shared, "no weight-free scalars were published at all"
    for key in shared:
        assert m2[key] == pytest.approx(m1[key], abs=0.0), key
    free = ("scope", "ranked_by", "direction_class", "rank", "token_id", "count",
            "base_logit_push", "base_logit_push_abs", "kl_mass", "kl_mass_abs",
            "p_student_mean", "sampled_count")

    def _rows(weight):
        return [
            {k: r[k] for k in free}
            for r in _fold(b, weight)[0].top_tokens(TASKS)
            if r["ranked_by"] != "extra_logit_push"
        ]

    r1, r2 = _rows(ones), _rows(_weight(b))
    assert r1 and r1 == r2


# --------------------------------------------------------------------------- #
# What the columns actually hold
# --------------------------------------------------------------------------- #
def test_base_and_kl_mass_are_the_quantities_they_are_named_for():
    b = _batch(seed=1)
    table, push = _fold(b, _weight(b))
    m = table.scalar_metrics(task_names=TASKS, prefix="opd")
    mask = b["mask"].unsqueeze(-1).double()
    g0 = push["g0"].double()
    assert m["opd/push/base_net_total"] == pytest.approx(float((g0 * mask).sum()), rel=1e-12)
    assert m["opd/push/base_abs_total"] == pytest.approx(float((g0.abs() * mask).sum()), rel=1e-12)
    klm = push["p_student"].double() * push["gap"].double() * mask
    assert m["opd/push/kl_mass_total"] == pytest.approx(float(klm.sum()), rel=1e-12)
    assert m["opd/push/kl_mass_abs_total"] == pytest.approx(float(klm.abs().sum()), rel=1e-12)


def test_kl_mass_is_signed_and_its_share_is_quoted_on_the_gross():
    """A candidate's share of D is genuinely negative where the teacher is the
    confident one. Only the position's whole D is bounded below, so a top-N
    share taken on the signed sum would be a share of a cancelled total."""
    b = _batch(seed=5)
    table, push = _fold(b, torch.ones(b["bs"], b["resp"]))
    klm = push["p_student"].double() * push["gap"].double() * b["mask"].unsqueeze(-1).double()
    assert float(klm.min()) < 0.0, "fixture has no negative candidate; the test is vacuous"
    m = table.scalar_metrics(task_names=TASKS, prefix="opd")
    assert m["opd/push/kl_mass_abs_total"] > abs(m["opd/push/kl_mass_total"])
    key = [k for k in m if k.startswith("opd/push/kl_mass_abs_top")][0]
    assert 0.0 < m[key] <= 1.0


def test_gap_absent_leaves_kl_mass_at_zero_rather_than_guessing():
    b = _batch()
    m = _fold(b, torch.ones(b["bs"], b["resp"]), gap=False)[0].scalar_metrics(
        task_names=TASKS, prefix="opd"
    )
    assert not [k for k in m if "kl_mass" in k]
    assert "opd/push/base_abs_total" in m       # the rest of the table still ran


def test_tail_coverage_has_its_own_weight_free_denominator():
    """The weighted coverage share is a share of W*|g|; at W != 1 that is a
    different number from the unweighted one, and the base ranking has to be
    quotable against its own."""
    b = _batch(seed=2)
    w = _weight(b)
    mw = _fold(b, w)[0].scalar_metrics(task_names=TASKS, prefix="opd")
    m1 = _fold(b, torch.ones(b["bs"], b["resp"]))[0].scalar_metrics(task_names=TASKS, prefix="opd")
    assert mw["opd/push/tail_base_abs_share"] == pytest.approx(m1["opd/push/tail_base_abs_share"])
    assert mw["opd/push/tail_weighted_abs_share"] != pytest.approx(
        mw["opd/push/tail_base_abs_share"]
    )


# --------------------------------------------------------------------------- #
# The early return that used to swallow them
# --------------------------------------------------------------------------- #
def test_a_run_with_no_weight_still_publishes_the_weight_free_half():
    """scalar_metrics returns early when the ADDED push is zero, which is
    exactly the state a control arm is in on every step. The weight-free keys
    are emitted before that return or the one run they exist for is the one run
    that never gets them."""
    b = _batch()
    m = _fold(b, torch.ones(b["bs"], b["resp"]))[0].scalar_metrics(task_names=TASKS, prefix="opd")
    assert not [k for k in m if "extra" in k], "W == 1 published an added-push series"
    for key in ("base_abs_total", "base_net_total", "base_n_distinct", "kl_mass_total"):
        assert f"opd/push/{key}" in m
    for task in TASKS:
        assert f"opd/{task}/push/base_abs_total" in m


def test_ranked_rows_exist_for_a_control_and_carry_no_push_class():
    """At W == 1 every candidate lands in a *_damped cell, because 1 > 1 is
    false. Splitting an unweighted ranking by those four would read as a
    finding and is an artefact, so the base rows are taken class-summed."""
    b = _batch()
    rows = _fold(b, torch.ones(b["bs"], b["resp"]))[0].top_tokens(TASKS)
    assert {r["ranked_by"] for r in rows} == {"base_logit_push", "kl_mass"}
    assert {r["direction_class"] for r in rows} <= {"base_up", "base_down"}
    assert all(r["extra_logit_push"] == 0.0 for r in rows)
    assert {r["scope"] for r in rows} == {"__pooled__", *TASKS}


def test_base_ranking_sums_the_class_axis():
    """A token pushed both ways lands in two different W classes. Ranked per
    class it would compete with itself; the ranking is on the total."""
    b = _batch(seed=11)
    w = _weight(b)
    table, push = _fold(b, w, top_n=b["vocab"])
    idx = {t: i for i, t in enumerate(LogitPushTokens.TERMS)}
    buf = table.buf.view(table.n_scopes, table.n_classes, b["vocab"], len(LogitPushTokens.TERMS))
    spread = int(((buf[0, :, :, idx["base_abs"]] > 0).sum(dim=0) > 1).sum())
    assert spread > 0, "fixture puts no token in two classes; the test is vacuous"
    rows = [r for r in table.top_tokens(TASKS)
            if r["ranked_by"] == "base_logit_push" and r["scope"] == "__pooled__"]
    want = buf[0, :, :, idx["base_abs"]].sum(dim=0)
    assert [r["token_id"] for r in rows] == torch.topk(want, len(rows)).indices.tolist()


def test_weight_free_render_has_exactly_one_owner():
    """The weighted arm runs a second instance at W = 1 to own these columns.
    Publishing them from both would be one series under two keys in the arm and
    one key in the control -- the opposite of comparable."""
    b = _batch()
    table = _fold(b, _weight(b))[0]
    full = table.scalar_metrics(task_names=TASKS, prefix="kl_weight")
    held = table.scalar_metrics(task_names=TASKS, prefix="kl_weight", weight_free=False)
    dropped = set(full) - set(held)
    assert dropped and all(("base" in k or "kl_mass" in k) for k in dropped)
    assert set(held) - set(full) == set()
    assert not [r for r in table.top_tokens(TASKS, weight_free=False)
                if r["ranked_by"] != "extra_logit_push"]


def test_weighted_rows_still_carry_the_unweighted_push_as_a_column():
    """'The arm amplified this token' is unreadable without how hard the plain
    term was pushing it already."""
    b = _batch()
    rows = [r for r in _fold(b, _weight(b))[0].top_tokens(TASKS, weight_free=False)]
    assert rows
    assert all("base_logit_push" in r and "kl_mass" in r for r in rows)


# --------------------------------------------------------------------------- #
# The gradient half
# --------------------------------------------------------------------------- #
def _grad_inputs(b, seed: int = 4):
    g = torch.Generator().manual_seed(seed)
    onehot = torch.zeros_like(b["student"])
    onehot[..., 0] = 1.0
    return dict(
        pg_grad_coef=torch.randn(b["bs"], b["resp"], generator=g) * 0.2,
        sampled_onehot=onehot,
    )


def test_opd_terms_are_the_weighted_terms_at_unit_weight():
    b = _batch()
    gi = _grad_inputs(b)
    got = opd_attribution_terms(
        student_logprob=b["student"], teacher_logprob=b["teacher"], teacher_kl=b["kl"],
        coef=0.01, pg_coef=1.0, **gi,
    )
    want = logit_gradient_terms(
        student_logprob=b["student"], teacher_logprob=b["teacher"],
        weight=torch.ones(b["bs"], b["resp"]), teacher_kl=b["kl"],
        coef=0.01, pg_coef=1.0, **gi,
    )
    assert set(got) == set(OPD_TERMS)
    for name in want:
        assert torch.allclose(got[name], want[name]), name
    for name in ("g_shared_sq", "g_shared_dot", "g_source_sq", "g_source_dot", "g_cross_dot"):
        assert float(got[name].abs().sum()) == 0.0


def test_push_abs_is_the_gross_descent_direction_including_the_tail():
    b = _batch()
    cols = opd_attribution_terms(
        student_logprob=b["student"], teacher_logprob=b["teacher"], teacher_kl=b["kl"],
        coef=0.01, **_grad_inputs(b),
    )
    push = opd_logit_push(
        student_logprob=b["student"], teacher_logprob=b["teacher"],
        teacher_kl=b["kl"], coef=0.01,
    )
    want = push["g0"].abs().sum(dim=-1) + push["g0_tail"].abs()
    assert torch.allclose(cols["push_abs"], want)
    assert torch.allclose(cols["d_kl"], b["kl"])


def test_the_row_weight_is_linear_on_the_sizes_and_quadratic_on_the_norms():
    """The gradient columns are norms of a weighted gradient; d_kl and push_abs
    are the quantities themselves. One dict, two scalings, and a reader summing
    them would otherwise be right to expect one."""
    b = _batch()
    gi = _grad_inputs(b)
    kw = dict(student_logprob=b["student"], teacher_logprob=b["teacher"],
              teacher_kl=b["kl"], coef=0.01, **gi)
    plain = opd_attribution_terms(**kw)
    rw = torch.full((b["bs"],), 3.0)
    scaled = opd_attribution_terms(row_weight=rw, **kw)
    assert torch.allclose(scaled["d_kl"], plain["d_kl"] * 3.0)
    assert torch.allclose(scaled["push_abs"], plain["push_abs"] * 3.0)
    assert torch.allclose(scaled["g_opd_sq"], plain["g_opd_sq"] * 9.0)


def test_no_partition_publishes_no_channel_series():
    """Two keys pinned at 0.0 forever would read as 'the channels did nothing',
    which an unweighted caller is in no position to say."""
    b = _batch()
    st = ScopeTermStats(names=OPD_TERMS, n_tasks=len(TASKS), device="cpu")
    st.update(
        opd_attribution_terms(
            student_logprob=b["student"], teacher_logprob=b["teacher"], teacher_kl=b["kl"],
            coef=0.01, **_grad_inputs(b),
        ),
        response_mask=b["mask"], task_ids=b["tasks"],
    )
    out = opd_attribution_metrics(st.sums(task_names=TASKS))
    assert not [k for k in out if "/channel/" in k]
    assert "opd/grpo/grad_cosine" in out and "opd/grpo/grad_norm_ratio" in out


def test_a_real_partition_still_publishes_them():
    """The guard above must be on the partition being absent, not on the code
    path being new: the weighted arm's channel cosines still have to render."""
    b = _batch()
    gi = _grad_inputs(b)
    cols = logit_gradient_terms(
        student_logprob=b["student"], teacher_logprob=b["teacher"],
        weight=_weight(b), teacher_kl=b["kl"], coef=0.01,
        push_shared=torch.full((b["bs"], b["resp"]), 0.3),
        push_source=torch.full((b["bs"], b["resp"]), 0.1),
        **gi,
    )
    st = ScopeTermStats(names=list(cols), n_tasks=len(TASKS), device="cpu")
    st.update(cols, response_mask=b["mask"], task_ids=b["tasks"])
    out = gradient_metrics(st.sums(task_names=TASKS))
    assert "kl_weight/channel/shared_grad_norm" in out
    assert "kl_weight/grpo/shared_grad_cosine" in out


def test_shares_are_a_partition_of_the_pooled_total():
    b = _batch()
    st = ScopeTermStats(names=OPD_TERMS, n_tasks=len(TASKS), device="cpu")
    st.update(
        opd_attribution_terms(
            student_logprob=b["student"], teacher_logprob=b["teacher"], teacher_kl=b["kl"],
            coef=0.01, **_grad_inputs(b),
        ),
        response_mask=b["mask"], task_ids=b["tasks"],
    )
    out = opd_attribution_metrics(st.sums(task_names=TASKS))
    for label in ("kl", "push_abs"):
        shares = [out[f"opd/{t}/{label}_share"] for t in TASKS]
        assert sum(shares) == pytest.approx(1.0)
    # the pooled scope has no share of itself
    assert "opd/kl_share" not in out and "opd/kl_mean" in out


def test_the_role_cut_publishes_every_suffix_it_curates():
    """A curated cut is a list of strings, and a typo in one of them empties a
    series silently rather than failing."""
    b = _batch()
    roles = torch.randint(0, 6, (b["bs"], b["resp"]))
    st = PositionScopeTermStats(names=OPD_TERMS, n_scopes=6, device="cpu")
    st.update(
        opd_attribution_terms(
            student_logprob=b["student"], teacher_logprob=b["teacher"], teacher_kl=b["kl"],
            coef=0.01, **_grad_inputs(b),
        ),
        response_mask=b["mask"], scope_ids=roles,
    )
    names = ["format", "reasoning", "env_action", "tool_call", "env_obs", "tag"]
    rendered = opd_attribution_metrics(st.sums(scope_names=names), prefix="opd/role")
    kept = select_metrics(rendered, OPD_ROLE_CUT_SUFFIXES)
    assert kept
    for suffix in OPD_ROLE_CUT_SUFFIXES:
        assert [k for k in kept if k.endswith(suffix)], suffix


# --------------------------------------------------------------------------- #
# Where the actor calls it from
# --------------------------------------------------------------------------- #
from tests.trainer.test_transfer_metrics import (  # noqa: E402
    _update_policy_source as _actor_source,
)


def test_the_attribution_is_taken_before_the_weight_is_applied():
    """teacher_kld is multiplied in place. One line later and every column here
    would be the WEIGHTED term's, reported under a name that says it is not."""
    src = _actor_source()
    fold = src.index("if opd_attr_on and epoch == 0")
    apply_ = src.index('teacher_kld = teacher_kld * xt_built["weight"]')
    assert fold < apply_


def test_the_attribution_is_not_nested_inside_an_arm_gate():
    """A control run reaches the fold with xt_built None on every step of the
    run, and that is the run these columns exist for."""
    src = _actor_source().splitlines()
    line = next(i for i, ln in enumerate(src) if "if opd_attr_on and epoch == 0" in ln)
    indent = len(src[line]) - len(src[line].lstrip())
    # Walk out to the enclosing block headers and check none of them is an arm gate.
    depth = indent
    for i in range(line - 1, -1, -1):
        ln = src[i]
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        here = len(ln) - len(ln.lstrip())
        if here < depth and ln.strip().endswith(":"):
            assert not re.search(r"\b(xt_built|xt_on|xt_enabled|xt_cfg_on|sign_enabled)\b", ln), ln
            depth = here
            if here == 0:
                break


def test_the_policy_gradient_coefficient_is_shared_with_the_weighted_geometry():
    """Two coefficients would make opd/grpo/grad_cosine and
    kl_weight/grpo/grad_cosine comparisons against different policy gradients."""
    src = _actor_source()
    assert "if xt_grad_stats is not None or opd_grad_stats is not None:" in src
    assert src.count("xt_pg_grad_coef = policy_loss_gradient_coef(") == 1
