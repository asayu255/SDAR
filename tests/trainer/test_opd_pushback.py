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

"""The pushback controller: the closed form, the gate, the state machine, resume."""

import pytest
import torch

from verl.trainer.ppo.opd_pushback import (
    STATE_ABSENT,
    STATE_FEW_GROUPS,
    STATE_NO_PG,
    STATE_OK,
    PushbackConfig,
    PushbackController,
    conflict_gate,
    pushback_retention,
)

TASKS = ["alfworld", "search", "webshop"]


def _cfg(**kw):
    base = dict(enable=True, eps=0.1, ema_decay=0.0, min_live_groups=2, min_ctl_tokens=4)
    base.update(kw)
    return PushbackConfig(**base)


# ---------------------------------------------------------------------------
# 1. the closed form


@pytest.mark.parametrize("ratio,expected", [(0.02, 1.0), (0.10, 1.0), (0.50, 0.2), (1.0, 0.1)])
def test_the_closed_form_matches_the_worked_table(ratio, expected):
    """eps = 0.1: pushback ratios of 2%, 10%, 50%, 100% -> a of 1, 1, 0.2, 0.1."""
    R = 10.0
    a, met = pushback_retention(0.1, R, ratio * R)
    assert a == pytest.approx(expected)
    assert met


def test_no_pushback_means_no_attenuation():
    a, met = pushback_retention(0.1, R=5.0, C_neg=0.0)
    assert a == 1.0 and met


def test_a_floor_that_blocks_the_bound_is_reported_as_unmet():
    """Clipping to a_min is not achievement: the pushback the rule set out to
    cap is still above eps R, and the caller has to know."""
    a, met = pushback_retention(0.1, R=1.0, C_neg=10.0, a_min=0.3)   # bound 0.01 < floor
    assert a == 0.3 and not met
    a, met = pushback_retention(0.1, R=1.0, C_neg=0.2, a_min=0.3)    # bound 0.5 >= floor
    assert a == 0.5 and met


def test_the_closed_form_is_the_minimiser_of_the_stated_problem():
    """min 1/2 (a-1)^2 s.t. a C <= eps R, 0 <= a <= 1 -- checked on a grid."""
    eps, R, C = 0.1, 3.0, 12.0
    grid = torch.linspace(0, 1, 100001)
    feasible = grid[grid * C <= eps * R + 1e-12]
    best = feasible[torch.argmin((feasible - 1) ** 2)]
    a, _ = pushback_retention(eps, R, C)
    assert a == pytest.approx(float(best), abs=1e-4)


# ---------------------------------------------------------------------------
# 2. the gate


def _terms(mask, rr, rd):
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return {"ctl_mask": t(mask), "ctl_rr": t(rr), "ctl_rd": t(rd)}


def test_the_gate_attenuates_only_conflicting_live_in_population_tokens():
    terms = _terms(
        mask=[[1, 1, 1, 0, 1]],
        rr=[[1.0, 1.0, 0.0, 1.0, 1.0]],
        rd=[[-1.0, +1.0, -1.0, -1.0, 0.0]],
    )
    #        conflict aligned clipped off-support orthogonal
    w = conflict_gate(terms, torch.tensor([2]), torch.tensor([1.0, 1.0, 0.25]), 3)
    assert w.tolist() == [[0.25, 1.0, 1.0, 1.0, 1.0]]


def test_the_gate_reads_the_row_task_and_leaves_padding_at_one():
    terms = _terms(mask=[[1], [1], [1]], rr=[[1.0], [1.0], [1.0]], rd=[[-1.0], [-1.0], [-1.0]])
    w = conflict_gate(terms, torch.tensor([0, 2, -1]), torch.tensor([0.5, 1.0, 0.1]), 3)
    assert w.squeeze(-1).tolist() == pytest.approx([0.5, 0.1, 1.0])


def test_the_gate_is_detached():
    terms = {k: v.requires_grad_(True) for k, v in _terms([[1]], [[1.0]], [[-1.0]]).items()}
    w = conflict_gate(terms, torch.tensor([0]), torch.tensor([0.5]), 1)
    assert not w.requires_grad


# ---------------------------------------------------------------------------
# 3. the state machine


def _ctl(cfg=None):
    return PushbackController(cfg or _cfg(), TASKS)


def _upd(c, R, C, tokens=1000, groups=10, present=None):
    present = present or {n: True for n in TASKS}
    return c.update(R=R, C_neg=C, ctl_tokens={n: tokens for n in TASKS},
                    live_groups={n: groups for n in TASKS}, present=present)


def test_a_starts_at_one_and_is_fixed_until_update():
    c = _ctl()
    assert c.a_tensor(TASKS).tolist() == [1.0, 1.0, 1.0]
    a0 = c.a_tensor(TASKS).clone()
    _upd(c, R={n: 1.0 for n in TASKS}, C={n: 5.0 for n in TASKS})
    assert not torch.equal(a0, c.a_tensor(TASKS)), "update() is the only thing that moves a"


def test_strong_pushback_attenuates_and_weak_does_not():
    c = _ctl()
    _upd(c, R={"alfworld": 10.0, "search": 10.0, "webshop": 10.0},
            C={"alfworld": 0.2, "search": 1.0, "webshop": 5.0})   # ratios 2%, 10%, 50%
    assert c.a_tensor(TASKS).tolist() == pytest.approx([1.0, 1.0, 0.2])


def test_missing_task_does_not_feed_the_ema_or_move_a():
    c = _ctl(_cfg(ema_decay=0.5))
    _upd(c, R={"alfworld": 10.0}, C={"alfworld": 5.0}, present={"alfworld": True, "search": False, "webshop": False})
    st = c.tasks["search"]
    assert st.state == STATE_ABSENT and st.n_obs == 0 and st.a == 1.0
    assert c.tasks["alfworld"].n_obs == 1


def test_no_pg_signal_holds_a_at_one():
    c = _ctl()
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 5.0 for n in TASKS})
    assert c.tasks["webshop"].a == pytest.approx(0.2)
    _upd(c, R={n: 0.0 for n in TASKS}, C={n: 5.0 for n in TASKS})
    assert c.tasks["webshop"].state == STATE_NO_PG and c.tasks["webshop"].a == 1.0


def test_too_few_groups_learns_but_does_not_act():
    c = _ctl(_cfg(min_live_groups=8))
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 5.0 for n in TASKS}, groups=3)
    st = c.tasks["webshop"]
    assert st.state == STATE_FEW_GROUPS and st.a == 1.0 and st.n_obs == 1
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 5.0 for n in TASKS}, groups=8)
    assert c.tasks["webshop"].state == STATE_OK and c.tasks["webshop"].a == pytest.approx(0.2)


def test_the_ema_is_on_the_sums_not_on_the_ratio():
    """Ratio of EMAs, never EMA of ratios."""
    c = _ctl(_cfg(ema_decay=0.5))
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 1.0 for n in TASKS})   # ratio .1
    _upd(c, R={n: 1.0 for n in TASKS}, C={n: 10.0 for n in TASKS})   # ratio 10
    st = c.tasks["alfworld"]
    assert st.ema_R == pytest.approx(5.5) and st.ema_C == pytest.approx(5.5)
    assert st.a == pytest.approx(0.1 * 5.5 / 5.5)                    # from the sums: 0.1
    # an EMA of the RATIOS would have said (0.1 + 10) / 2 = 5.05 -> a = 0.1/5.05


def test_metrics_carry_state_bound_and_constraint():
    c = _ctl(_cfg(a_min=0.5))
    m = _upd(c, R={n: 1.0 for n in TASKS}, C={n: 10.0 for n in TASKS})   # bound 0.01 < floor
    assert m["actor/pushback/a_next/webshop"] == 0.5
    assert m["actor/pushback/constraint_met/webshop"] == 0.0
    assert m["actor/pushback/state/webshop"] == float(STATE_OK)
    assert m["actor/pushback/bound/webshop"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 4. resume


def test_state_round_trips_and_a_changed_eps_is_refused():
    c = _ctl(_cfg(ema_decay=0.5))
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 5.0 for n in TASKS})
    sd = c.state_dict()
    d = _ctl(_cfg(ema_decay=0.5))
    d.load_state_dict(sd)
    assert d.a_tensor(TASKS).tolist() == c.a_tensor(TASKS).tolist()
    assert d.tasks["webshop"].ema_C == c.tasks["webshop"].ema_C and d.step == 1
    e = _ctl(_cfg(ema_decay=0.5, eps=0.2))
    with pytest.raises(ValueError, match="changed across resume"):
        e.load_state_dict(sd)


def test_config_validation_refuses_the_obvious():
    with pytest.raises(ValueError):
        _cfg(eps=0.0).validate()
    with pytest.raises(ValueError):
        _cfg(ema_decay=1.0).validate()
    with pytest.raises(ValueError):
        PushbackConfig.from_mapping({"enable": True, "epsilon": 0.1})


# ---------------------------------------------------------------------------
# 5. group coverage: a dense index on the driver, a bitmap in the actor


def test_the_group_index_is_dense_and_stable_and_marks_padding():
    from verl.trainer.ppo.opd_pushback import group_index_column

    idx = group_index_column(["g1", "g1", "g2", None, "g1"], 5)
    assert idx.tolist() == [0, 0, 1, -1, 0], "dense, stable, -1 for a missing uid"
    assert group_index_column(["a", "b"], 4).tolist() == [0, 1, -1, -1], "padded rows"


def test_the_bitmap_counts_only_groups_that_reached_R_and_unions_them():
    """The two things a driver-side count of 'groups with a non-zero advantage'
    got wrong: it never asked whether the group survived the loss mask, the clip
    and the top-k support, and per-rank counts of distinct groups cannot be
    summed."""
    from verl.trainer.ppo.opd_task_diag import OpdTaskDiagStats

    n, t = 4, 2
    # rows 0,1 -> group 0 ; rows 2,3 -> group 1. Only row 0 has a live reward.
    rr = torch.zeros(n, t); rr[0, 0] = 1.0
    terms = {
        "opd_sq": torch.ones(n, t), "dot": torch.zeros(n, t), "cos": torch.zeros(n, t),
        "pg_sq": torch.zeros(n, t), "align_mask": torch.zeros(n, t), "pg_live": torch.zeros(n, t),
        "tail_mass": torch.zeros(n, t), "ctl_mask": torch.ones(n, t),
        "ctl_rr": rr, "ctl_rd": torch.zeros(n, t), "ctl_dd": torch.ones(n, t),
        "ctl_c_neg": torch.zeros(n, t), "ctl_conflict": torch.zeros(n, t),
    }
    st = OpdTaskDiagStats(n_tasks=1, device=torch.device("cpu"), n_groups=8)
    st.update(task_ids=torch.zeros(n, dtype=torch.long), response_mask=torch.ones(n, t),
              teacher_kl=torch.ones(n, t), advantages=torch.ones(n, t), terms=terms,
              row_basis=torch.ones(n), row_coef=torch.ones(n),
              group_idx=torch.tensor([0, 0, 1, 1]))
    assert st.reduced_groups(["alfworld"]) == {"alfworld": 1}, "group 1 never reached R"

    # the same group seen twice must count once -- what a per-rank sum broke
    st2 = OpdTaskDiagStats(n_tasks=1, device=torch.device("cpu"), n_groups=8)
    rr2 = torch.ones(n, t)
    st2.update(task_ids=torch.zeros(n, dtype=torch.long), response_mask=torch.ones(n, t),
               teacher_kl=torch.ones(n, t), advantages=torch.ones(n, t),
               terms=dict(terms, ctl_rr=rr2), row_basis=torch.ones(n), row_coef=torch.ones(n),
               group_idx=torch.tensor([0, 0, 0, 0]))
    assert st2.reduced_groups(["alfworld"]) == {"alfworld": 1}


def test_a_zero_weight_row_is_not_evidence():
    """A duplicated row carries basis 0: it cannot move the loss, so it must not
    count toward the token or group coverage the sufficiency test reads."""
    from verl.trainer.ppo.opd_task_diag import OpdTaskDiagStats

    n, t = 2, 3
    terms = {
        "opd_sq": torch.ones(n, t), "dot": torch.zeros(n, t), "cos": torch.zeros(n, t),
        "pg_sq": torch.zeros(n, t), "align_mask": torch.zeros(n, t), "pg_live": torch.zeros(n, t),
        "tail_mass": torch.zeros(n, t), "ctl_mask": torch.ones(n, t),
        "ctl_rr": torch.ones(n, t), "ctl_rd": torch.zeros(n, t), "ctl_dd": torch.ones(n, t),
        "ctl_c_neg": torch.zeros(n, t), "ctl_conflict": torch.zeros(n, t),
    }
    st = OpdTaskDiagStats(n_tasks=1, device=torch.device("cpu"), n_groups=8)
    st.update(task_ids=torch.zeros(n, dtype=torch.long), response_mask=torch.ones(n, t),
              teacher_kl=torch.ones(n, t), advantages=torch.ones(n, t), terms=terms,
              row_basis=torch.tensor([1.0, 0.0]),      # second row duplicated
              row_coef=torch.ones(n), group_idx=torch.tensor([0, 1]))
    out = st.rows(["alfworld"])
    assert out["actor/opd_diag/ctl_tokens/alfworld"] == 6.0, "all tokens, for reference"
    assert out["actor/opd_diag/ctl_tokens_weighted/alfworld"] == 3.0, "only the weighted row"
    assert st.reduced_groups(["alfworld"]) == {"alfworld": 1}, "the zero-weight group is not evidence"


# 6. the actor wiring, on the syntax tree


def _update_policy_src():
    import ast
    import inspect

    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    return ast.unparse(fn)


def test_the_gate_is_decided_before_the_loss_and_applied_in_every_branch():
    src = _update_policy_src()
    i_terms = src.index("_pb_terms = opd_pg_alignment_terms(")
    i_gate = src.index("_pb_w = conflict_gate(")
    i_loss = src.index("_kld_for_loss = teacher_kld if _pb_w is None else teacher_kld * _pb_w")
    i_bwd = src.index("loss.backward()")
    assert i_terms < i_gate < i_loss < i_bwd, "terms -> gate -> gated KL -> backward"
    # every place the OPD term reaches the objective takes the gated tensor
    # ast.unparse drops the redundant parentheses around the conditional
    assert "loss_mat=_kld_for_loss if _kl_row_coef is None else _kld_for_loss * _kl_row_coef.reshape(-1, 1)" in src
    assert "row_kl = (_kld_for_loss * response_mask).sum(-1)" in src
    # ...and the ungated one still feeds the unweighted metric
    assert "teacher_kl_loss = agg_loss(loss_mat=teacher_kld, loss_mask=response_mask" in src


def test_off_is_the_original_expression_bit_for_bit():
    """With no gate, _kld_for_loss IS teacher_kld and the b-less branch takes the
    pre-existing line, so a control run is the same code path, not a w=1 one."""
    src = _update_policy_src()
    assert "if _kl_row_coef is None and _pb_w is None:" in src
    assert "policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef" in src


def test_the_control_inputs_are_measured_at_beta_and_the_gate_is_recorded():
    src = _update_policy_src()
    assert "opd_coef=_teacher_kl_coef_scalar" in src, "beta, not beta*b and not the gate"
    assert "gate_w=_pend['gate_w']" in src
    assert "'terms': _pb_terms" in src, "the terms are computed once and reused after backward"
    assert src.count("opd_pg_alignment_terms(") == 1, "not recomputed after the backward"


def test_a_is_read_once_per_step_and_updated_once_from_the_reduced_table():
    src = _update_policy_src()
    assert src.count("pushback.a_tensor(") == 1
    assert "_table = opd_diag_stats.reduced()" in src
    assert "pushback.update(" in src
    assert src.index("_table = opd_diag_stats.reduced()") < src.index("pushback.update(")
    # a_applied is what THIS step used; a_next is what update() decided
    assert "actor/pushback/a_applied/" in src
    # requires the readout, and says so
    assert "teacher_kl_pushback needs teacher_kl_task_diag=True" in src


def test_group_coverage_never_reads_the_rebound_data_name():
    """THE P1 REGRESSION. `data` is rebound inside update_policy three times --
    a micro-batch dict, a TensorDict, and a metrics dict -- so anything at the
    end of the function that reaches for `data.meta_info` raises AttributeError
    on the first update. The group counts come off the reduced bitmap instead.
    """
    import ast
    import inspect

    import verl.workers.actor.dp_actor as m

    src = inspect.getsource(m)
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    body = ast.unparse(fn)
    assert "data.meta_info.get('pushback" not in body
    assert "reduced_groups(" in body

    # And, structurally: every name rebound inside the micro-batch loop is
    # unsafe at the end of the function. Assert the controller-update block
    # touches none of them.
    rebound = set()
    for n in ast.walk(fn):
        tgts = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, (ast.For, ast.AsyncFor)) else [])
        for t in tgts:
            if isinstance(t, ast.Name):
                rebound.add(t.id)
    assert "data" in rebound, "fixture: data really is rebound in this function"
    tail = body[body.index("pushback.update("):]
    for name in ("data",):
        assert f"{name}.meta_info" not in tail, f"{name} is rebound in the loop; unsafe here"


def test_the_group_index_reaches_the_micro_batch_and_the_accumulator():
    import ast
    import inspect

    import verl.workers.actor.dp_actor as m
    import verl.trainer.ppo.opd_ray_trainer as t

    src = ast.unparse(next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
                           if isinstance(n, ast.FunctionDef) and n.name == "update_policy"))
    assert "select_keys.append('pushback_group_idx')" in src
    assert "group_idx=data.get('pushback_group_idx', None)" in src
    assert 'batch.batch["pushback_group_idx"] = group_index_column(' in inspect.getsource(t)


def test_the_checkpoint_carries_the_controller_state():
    import inspect

    import verl.workers.fsdp_workers as w
    import verl.utils.checkpoint.fsdp_checkpoint_manager as cm

    assert "extra_state_provider=self.actor.actor_extra_state_dict" in inspect.getsource(w)
    assert "extra_state_consumer=self.actor.load_actor_extra_state_dict" in inspect.getsource(w)
    csrc = inspect.getsource(cm)
    assert 'extra_state_dict["actor_extra"] = self._extra_state_provider()' in csrc
    assert 'self._extra_state_consumer(extra_state_dict.get("actor_extra", None))' in csrc


def test_exclusivity_is_enforced_at_injection():
    from omegaconf import OmegaConf

    from verl.trainer.main_opd import inject_distillation_config

    def build(opd_extra):
        c = OmegaConf.create({
            "algorithm": {"opd": {"kl_loss_coef": 0.01, "task_diag": True, **opd_extra}},
            "actor_rollout_ref": {"actor": {}, "model": {}, "rollout": {}, "ref": {}},
            "data": {}, "trainer": {},
        })
        inject_distillation_config(c)
        return c.actor_rollout_ref.actor

    pb = {"enable": True, "eps": 0.1, "ema_decay": 0.9, "min_live_groups": 4, "min_ctl_tokens": 256}
    a = build({"pushback_control": pb})
    assert dict(a.teacher_kl_pushback)["enable"] is True
    # all-ones static coefficient is tolerated; anything else is not
    build({"pushback_control": pb, "kl_loss_coef_by_task": {"webshop": 1.0}})
    with pytest.raises(ValueError, match="does not stack"):
        build({"pushback_control": pb, "kl_loss_coef_by_task": {"webshop": 0.5}})
    with pytest.raises(ValueError, match="One at a time"):
        build({"pushback_control": pb, "cross_teacher_kl_weight": {"enable": True}})
    with pytest.raises(ValueError, match="task_diag=True"):
        c = OmegaConf.create({
            "algorithm": {"opd": {"kl_loss_coef": 0.01, "task_diag": False, "pushback_control": pb}},
            "actor_rollout_ref": {"actor": {}, "model": {}, "rollout": {}, "ref": {}},
            "data": {}, "trainer": {},
        })
        inject_distillation_config(c)
    # off by default, and off means the actor key is None
    assert build({}).get("teacher_kl_pushback", "MISSING") is None


# ---------------------------------------------------------------------------
# 7. tasks that come and go (the P3 regression)


def test_a_task_missing_from_a_batch_is_absent_not_an_error():
    """The driver builds task_id_names from the tasks PRESENT in the batch, so
    the list legitimately shortens and renumbers. Asserting on it turned the
    STATE_ABSENT case into a crash; keying state by position would have applied
    webshop's retention to search."""
    c = _ctl()
    _upd(c, R={n: 10.0 for n in TASKS}, C={n: 5.0 for n in TASKS})
    assert c.tasks["webshop"].a == pytest.approx(0.2)

    # next batch has no search rows: the driver hands over two names, reordered
    short = ["alfworld", "webshop"]
    a = c.a_tensor(short)
    assert a.tolist() == pytest.approx([c.tasks["alfworld"].a, c.tasks["webshop"].a]), (
        "index i must be the retention of THIS batch's task id i"
    )
    out = c.update(R={"alfworld": 10.0, "webshop": 10.0}, C_neg={"alfworld": 0.0, "webshop": 5.0},
                   ctl_tokens={n: 1000 for n in short}, live_groups={n: 10 for n in short},
                   present={"alfworld": True, "webshop": True})
    assert out["actor/pushback/state/search"] == float(STATE_ABSENT)
    assert c.tasks["search"].a == pytest.approx(0.2), (
        "untouched: it keeps the retention it had, is not reset to 1, and does not crash"
    )
    assert c.tasks["search"].n_obs == 1, "and the EMA was not fed a zero"
    assert c.tasks["webshop"].a == pytest.approx(0.2)


def test_a_task_appearing_later_enters_at_one():
    c = PushbackController(_cfg(), ["alfworld"])
    assert c.task_names == ["alfworld"]
    a = c.a_tensor(["alfworld", "search"])
    assert a.tolist() == [1.0, 1.0]
    assert "search" in c.task_names


def test_retention_is_reported_as_the_loss_sees_it():
    """kl_retained must carry the row weight on BOTH sides. A duplicated row
    (basis 0) contributes KL to the unweighted ratio and nothing to the loss."""
    from verl.trainer.ppo.opd_task_diag import OpdTaskDiagStats

    n, t = 2, 2
    kl = torch.tensor([[1.0, 1.0], [4.0, 4.0]])          # row 1 has most of the KL
    w = torch.tensor([[0.5, 1.0], [0.0, 0.0]])           # row 1 gated to nothing
    terms = {
        "opd_sq": torch.ones(n, t), "dot": torch.zeros(n, t), "cos": torch.zeros(n, t),
        "pg_sq": torch.zeros(n, t), "align_mask": torch.zeros(n, t), "pg_live": torch.zeros(n, t),
        "tail_mass": torch.zeros(n, t), "ctl_mask": torch.ones(n, t),
        "ctl_rr": torch.ones(n, t), "ctl_rd": torch.zeros(n, t), "ctl_dd": torch.ones(n, t),
        "ctl_c_neg": torch.zeros(n, t), "ctl_conflict": torch.zeros(n, t),
    }
    st = OpdTaskDiagStats(n_tasks=1, device=torch.device("cpu"))
    st.update(task_ids=torch.zeros(n, dtype=torch.long), response_mask=torch.ones(n, t),
              teacher_kl=kl, advantages=torch.ones(n, t), terms=terms,
              row_basis=torch.tensor([1.0, 0.0]), row_coef=torch.ones(n), gate_w=w)
    out = st.rows(["alfworld"])
    # the loss only sees row 0: (0.5*1 + 1*1) / (1 + 1) = 0.75
    assert out["actor/pushback/kl_retained/alfworld"] == pytest.approx(0.75)
    # unweighted drags row 1 in, which the loss never sees: 1.5 / 10
    assert out["actor/pushback/kl_retained_unweighted/alfworld"] == pytest.approx(0.15)
    assert out["actor/pushback/kl_retained/alfworld"] != pytest.approx(
        out["actor/pushback/kl_retained_unweighted/alfworld"])


def test_conflict_and_gated_are_separate_numbers():
    """conflict_frac must be a fact about the tokens; gated_frac is what the
    controller did. At a = 1 the second is 0 however much conflict there is."""
    from verl.trainer.ppo.opd_task_diag import OpdTaskDiagStats

    n, t = 1, 4
    conflict = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    terms = {
        "opd_sq": torch.ones(n, t), "dot": torch.zeros(n, t), "cos": torch.zeros(n, t),
        "pg_sq": torch.zeros(n, t), "align_mask": torch.zeros(n, t), "pg_live": torch.zeros(n, t),
        "tail_mass": torch.zeros(n, t), "ctl_mask": torch.ones(n, t),
        "ctl_rr": torch.ones(n, t), "ctl_rd": torch.zeros(n, t), "ctl_dd": torch.ones(n, t),
        "ctl_c_neg": torch.zeros(n, t), "ctl_conflict": conflict,
    }
    st = OpdTaskDiagStats(n_tasks=1, device=torch.device("cpu"))
    st.update(task_ids=torch.zeros(n, dtype=torch.long), response_mask=torch.ones(n, t),
              teacher_kl=torch.ones(n, t), advantages=torch.ones(n, t), terms=terms,
              row_basis=torch.ones(n), row_coef=torch.ones(n),
              gate_w=torch.ones(n, t))      # a = 1: nothing gated
    out = st.rows(["alfworld"])
    assert out["actor/pushback/conflict_frac/alfworld"] == pytest.approx(0.5)
    assert out["actor/pushback/gated_frac/alfworld"] == 0.0
