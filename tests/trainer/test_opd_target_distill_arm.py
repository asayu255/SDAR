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

"""MOPD v3 as WIRED: where the target replaces the teacher, and what is refused.

The arithmetic is in test_opd_target_distill.py. What can still go wrong is the
plumbing, and the one that would be silent is the GRPO term being left on -- the
run would train, the metrics would look sane, and the reward would be counted
twice. That is checked at the injection, not by reading the run script.
"""

import ast
import inspect

import pytest


def _update_policy_src():
    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    return ast.unparse(fn), fn


# ---------------------------------------------------------------------------
# 1. the swap happens where the loss reads the teacher


def test_the_target_replaces_the_teacher_before_the_kl_is_taken():
    src, _ = _update_policy_src()
    i_build = src.index("_td = build_target(")
    i_swap = src.index("teacher_topk_lp = _td['target_logprob']")
    i_kl = src.index("teacher_kld = topk_kl_per_token(")
    assert i_build < i_swap < i_kl, "the target must be in place before the KL is taken"


def test_the_untilted_kl_is_computed_first_and_fed_to_build_target():
    """build_target derives ||d|| from the base KL. Taking it from opd_task_diag
    would be circular: that module computes it from the KL being rewritten."""
    src, _ = _update_policy_src()
    i_base = src.index("_td_base_kl = topk_kl_per_token(")
    i_build = src.index("_td = build_target(")
    assert i_base < i_build
    call = src[i_build:src.index("_td_pending = {")]
    assert "teacher_kl_base=_td_base_kl" in call
    assert "pg_grad_coef=xt_pg_grad_coef" in call, "the clip branches must reach the target"
    assert "advantages=data.get('advantages', None)" in call


def test_the_stats_fold_after_the_backward_and_the_solve_is_once_per_step():
    src, _ = _update_policy_src()
    assert src.index("_td = build_target(") < src.index("loss.backward()") < src.index("td_stats.update(")
    assert src.count("target_distill.refs_to_device(") == 1
    assert src.count("target_distill.update(") == 1
    assert src.count("td_stats.reduced()") == 1
    assert "actor/target/alpha_read/" in src


def test_it_is_exclusive_with_the_other_teacher_kl_mechanisms_in_the_actor_too():
    src, _ = _update_policy_src()
    assert "teacher_kl_target_distill is enabled together with the cross gate" in src


def test_every_name_the_inserted_blocks_read_is_bound_earlier():
    """The class of bug that killed the pushback arm's first update: a name read
    at a point where the loop has rebound or not yet bound it."""
    import builtins

    import verl.workers.actor.dp_actor as m

    tree = ast.parse(inspect.getsource(m))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    module_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    module_names |= {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
                     if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    module_names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    module_names |= set(dir(builtins))

    def bound_before(lineno):
        bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if getattr(n, "lineno", 10 ** 9) >= lineno:
                continue
            tgts = []
            if isinstance(n, ast.Assign):
                tgts = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
                tgts = [n.target]
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                tgts = [it.optional_vars for it in n.items if it.optional_vars is not None]
            for t in tgts:
                for leaf in ast.walk(t):
                    if isinstance(leaf, ast.Name):
                        bound.add(leaf.id)
        return bound

    markers = ("_td = build_target(", "td_stats.update(", "target_distill.update(",
               "target_distill.refs_to_device(")
    checked = 0
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.Expr, ast.If)):
            continue
        stmts = [node] if not isinstance(node, ast.If) else [
            s for s in node.body if any(mk in ast.unparse(s) for mk in markers)]
        for s in stmts:
            if not any(mk in ast.unparse(s) for mk in markers):
                continue
            reads = {n.id for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            missing = sorted(reads - bound_before(s.lineno) - module_names)
            assert not missing, f"unbound at line {s.lineno}: {missing}"
            checked += 1
    assert checked >= 4, f"expected the four v3 statements, found {checked}"


# ---------------------------------------------------------------------------
# 2. the checkpoint


def test_the_checkpoint_carries_the_controller():
    import verl.workers.actor.dp_actor as m

    assert '"target_distill"' in inspect.getsource(m.DataParallelPPOActor.actor_extra_state_dict)
    src = inspect.getsource(m.DataParallelPPOActor.load_actor_extra_state_dict)
    assert 'sd.get("target_distill", None)' in src and "_target_distill_pending_state" in src
    src = inspect.getsource(m.DataParallelPPOActor.target_distill_controller)
    assert "model_vocab_size(self.actor_module)" in src and "_target_distill_pending_state" in src


# ---------------------------------------------------------------------------
# 3. what the injection refuses


def _opd(**kw):
    base = {"task_diag": True}
    base.update(kw)
    return base


def test_the_grpo_term_must_be_off():
    """THE check. With pg_loss_coef left at 1 the update is r + d + beta F c and
    the reward is counted twice -- silently, because the run would train fine."""
    from verl.trainer.main_opd import validate_target_distill_exclusivity

    td = {"enable": True, "eta": 1.0}
    validate_target_distill_exclusivity(td, None, 0.0, _opd(target_distill=td))
    for bad in (1.0, 0.5, 1e-6):
        with pytest.raises(ValueError, match="pg_loss_coef"):
            validate_target_distill_exclusivity(td, None, bad, _opd(target_distill=td))


def test_it_refuses_every_other_mechanism_on_the_same_term():
    from verl.trainer.main_opd import (
        validate_cross_gate_exclusivity,
        validate_pushback_exclusivity,
        validate_target_distill_exclusivity,
    )

    td = {"enable": True}
    for other in ("sign_weight", "cross_teacher_kl_weight", "cross_teacher_target",
                  "pushback_control", "cross_gate"):
        with pytest.raises(ValueError, match=f"{other}.enable=True"):
            validate_target_distill_exclusivity(td, None, 0.0,
                                                _opd(target_distill=td, **{other: {"enable": True}}))
    with pytest.raises(ValueError, match="does not stack"):
        validate_target_distill_exclusivity(td, {"alfworld": 0.9}, 0.0, _opd(target_distill=td))
    # and from the other side
    with pytest.raises(ValueError, match="target_distill.enable=True"):
        validate_cross_gate_exclusivity({"enable": True}, None, True,
                                        _opd(cross_gate={"enable": True}, target_distill=td))
    with pytest.raises(ValueError, match="target_distill.enable=True"):
        validate_pushback_exclusivity({"enable": True, "eps": 0.003}, None, True,
                                      _opd(pushback_control={"enable": True}, target_distill=td))
    # off is off
    validate_target_distill_exclusivity({"enable": False}, {"alfworld": 0.5}, 1.0, {})
    validate_target_distill_exclusivity(None, None, 1.0, {})


def test_a_bad_knob_is_refused_at_startup():
    from verl.trainer.main_opd import validate_target_distill_exclusivity

    for bad in ({"enable": True, "eta": -1.0}, {"enable": True, "clamp": 0.0},
                {"enable": True, "epsilon_w": 1.0}, {"enable": True, "roles": ["format", "nope"]}):
        with pytest.raises(ValueError):
            validate_target_distill_exclusivity(bad, None, 0.0, {})


def test_the_injection_reaches_the_actor(tmp_path):
    import os
    import pathlib

    from hydra import compose, initialize_config_dir

    from verl.trainer.main_opd import inject_distillation_config

    root = pathlib.Path(__file__).resolve().parents[2]
    base = [
        "+algorithm.opd.task_diag=True",
        "+algorithm.opd.kl_loss_coef=0.01",
        "+algorithm.opd.kl_loss_type=topk_kl",
        "+algorithm.opd.target_distill.enable=True",
        "+algorithm.opd.target_distill.eta=1.0",
        "+algorithm.opd.target_distill.integrate=True",
        "+algorithm.opd.target_distill.roles=[format,env_action]",
        "actor_rollout_ref.actor.pg_loss_coef=0.0",
    ]
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    with initialize_config_dir(config_dir=str(root / "verl" / "trainer" / "config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=base)
    inject_distillation_config(cfg)
    got = cfg.actor_rollout_ref.actor.teacher_kl_target_distill
    assert got is not None and got["enable"] is True and bool(got["integrate"]) is True
    assert float(cfg.actor_rollout_ref.actor.pg_loss_coef) == 0.0
    assert cfg.actor_rollout_ref.actor.teacher_kl_cross_gate is None
    assert cfg.actor_rollout_ref.actor.teacher_kl_pushback is None
    # and with the GRPO term left on it is refused
    with initialize_config_dir(config_dir=str(root / "verl" / "trainer" / "config"), version_base=None):
        cfg2 = compose(config_name="ppo_trainer",
                       overrides=[o for o in base if "pg_loss_coef" not in o])
    with pytest.raises(ValueError, match="pg_loss_coef"):
        inject_distillation_config(cfg2)


# ---------------------------------------------------------------------------
# 4. THE regression: pg_loss_coef == 0 must not switch the mechanism off
#
# The arm REQUIRES pg_loss_coef = 0, and the actor has three fast paths that
# read that as "no policy-gradient signal at all": select_keys drops advantages
# and old_log_probs, need_log_prob skips the full-vocabulary log_prob, and the
# clipped coefficient is computed only inside the pg branch. With all three
# taken, r = 0 and e = 1, so c = 0 and q* = q -- the arm runs as pure OPD while
# emitting every metric, which is silent.
#
# The AST tests above did NOT catch this: they assert that
# "pg_grad_coef=xt_pg_grad_coef" appears in the call, which stayed true while the
# value was None. So these evaluate the real conditions instead of reading them.


def _assign_expr(fn_name, target):
    """The source expression of `target = ...` inside `fn_name`, as an ast.Expression."""
    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target:
            return ast.Expression(body=node.value)
    raise AssertionError(f"no assignment to {target} in {fn_name}")


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def test_need_log_prob_is_true_for_this_arm():
    """log_prob is what r is built from, and the guard on build_target requires
    it. If the fast path takes it, the mechanism never runs."""
    expr = _assign_expr("update_policy", "need_log_prob")
    ns = dict(pg_loss_coef=0, teacher_topk_kl=True, use_teacher_kl_loss=True,
              self=type("S", (), {"config": _Cfg(use_kl_loss=False, use_sdl_loss=False,
                                                 use_sdar_loss=False)})())
    ns["self"].config.use_kl_loss = False
    assert eval(compile(ast.fix_missing_locations(expr), "<t>", "eval"),
                {}, dict(ns, _td_needs_pg=True)) is True, \
        "with the arm on, the full-vocabulary log_prob must still be computed"
    # and without it the fast path is still taken, so the exemption is narrow
    assert eval(compile(ast.fix_missing_locations(expr), "<t>", "eval"),
                {}, dict(ns, _td_needs_pg=False)) is False


def test_the_reward_inputs_are_selected_for_this_arm():
    """advantages feed e and old_log_probs feed the clip ratio. Dropped, e = 1
    and r = 0 -- and `data.get("advantages", None)` returns None silently rather
    than raising, so nothing downstream complains."""
    src, _ = _update_policy_src()
    assert "_td_needs_pg = needs_policy_gradient_inputs(" in src
    # The guard on the select must admit this arm. Evaluated, not read: the
    # condition is what decides, and a reverted `if pg_loss_coef != 0:` looks
    # perfectly reasonable in a diff.
    expr = _assign_guard_of("select_keys += ['old_log_probs', 'advantages']")
    assert eval(expr, {}, dict(pg_loss_coef=0, _td_needs_pg=True)) is True, \
        "at pg_loss_coef = 0 with the arm on, the reward's inputs must still be selected"
    assert eval(expr, {}, dict(pg_loss_coef=0, _td_needs_pg=False)) is False, \
        "and the exemption must stay narrow"
    assert eval(expr, {}, dict(pg_loss_coef=1.0, _td_needs_pg=False)) is True
    # bound for both branches -- the coefficient below needs them at pg_loss_coef = 0
    i_bind = src.index("old_log_prob = data['old_log_probs']")
    i_coef = src.index("xt_pg_grad_coef = policy_loss_gradient_coef(")
    assert i_bind < i_coef


def _assign_guard_of(stmt_src):
    """The `if` test that guards the statement whose unparse is `stmt_src`."""
    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and any(
                ast.unparse(b).strip() == stmt_src for b in node.body):
            return compile(ast.Expression(body=node.test), "<t>", "eval")
    raise AssertionError(f"no if-statement guarding {stmt_src!r}")


def test_the_clipped_coefficient_is_computed_when_the_pg_branch_is_skipped():
    """policy_loss_gradient_coef does not depend on pg_loss_coef -- that scales
    the loss, not the derivative -- so ONE computation sits above the branch and
    v3 gets it at pg_loss_coef = 0. Two computations would break the invariant
    test_opd_attribution guards: the diagnostics and the weighted geometry must
    read the same policy gradient."""
    src, _ = _update_policy_src()
    assert src.count("xt_pg_grad_coef = policy_loss_gradient_coef(") == 1
    # the guard admits this arm, and still requires the inputs to exist
    expr = _assign_guard_of(
        "xt_pg_grad_coef = policy_loss_gradient_coef(old_log_prob=old_log_prob, "
        "log_prob=log_prob, advantages=advantages, cliprange=clip_ratio, "
        "cliprange_low=clip_ratio_low, cliprange_high=clip_ratio_high, "
        "clip_ratio_c=clip_ratio_c).detach()")
    off = dict(xt_grad_stats=None, opd_grad_stats=None, opd_diag_stats=None,
               pushback=None, cross_gate=None, target_distill=None,
               log_prob=object(), old_log_prob=object())
    assert eval(expr, {}, dict(off, target_distill=object())) is True, \
        "the arm must get the coefficient even when the pg branch is skipped"
    assert eval(expr, {}, off) is False, "and nothing else asks for it"
    assert eval(expr, {}, dict(off, target_distill=object(), log_prob=None)) is False
    assert eval(expr, {}, dict(off, target_distill=object(), old_log_prob=None)) is False
    # it is computed BEFORE either branch, so both reach it
    i_coef = src.index("xt_pg_grad_coef = policy_loss_gradient_coef(")
    assert i_coef < src.index("if pg_loss_coef != 0:")


def test_the_predicate_is_true_for_the_arms_own_lock():
    from verl.trainer.ppo.opd_target_distill import needs_policy_gradient_inputs
    from verl.utils.expected_config import load_expectations
    import os

    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    for arm in ("tdist", "tdist_int"):
        flat = load_expectations(f"examples/opd_grpo_trainer/expected_multitask_opd_coef_{arm}_config.yaml")
        assert flat["actor_rollout_ref.actor.pg_loss_coef"] == 0.0
        assert needs_policy_gradient_inputs(
            {"enable": flat["actor_rollout_ref.actor.teacher_kl_target_distill.enable"]}) is True
    assert needs_policy_gradient_inputs(None) is False
    assert needs_policy_gradient_inputs({"enable": False}) is False


def test_the_failure_mode_itself_produces_a_dead_target():
    """What the three fast paths would have delivered: no advantages, no
    coefficient. c = 0 and q* = q -- named here so the regression is legible."""
    import torch

    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.opd_target_distill import (
        N_ROLES, TargetDistillConfig, TargetDistillRefs, build_target,
    )

    g = torch.Generator().manual_seed(0)
    V, K, bs, T = 40, 5, 4, 5
    lp_s = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    lp_t = torch.log_softmax(torch.randn(bs, T, V, generator=g, dtype=torch.float64), -1)
    ids = torch.topk(lp_s, K, dim=-1).indices
    s, t = torch.gather(lp_s, -1, ids), torch.gather(lp_t, -1, ids)
    cfg = TargetDistillConfig(enable=True, eta=1.0, integrate=False)
    refs = TargetDistillRefs(alpha=torch.ones(3, N_ROLES), sbar=torch.full((3, N_ROLES), 1e-2),
                             control_roles=cfg.control_role_mask())
    common = dict(student_topk_logprob=s, teacher_topk_logprob=t, topk_ids=ids,
                  response_ids=ids[..., 0],
                  teacher_kl_base=topk_kl_per_token(student_topk_logprob=s, teacher_topk_logprob=t),
                  task_ids=torch.tensor([0, 1, 2, 0]),
                  roles=torch.zeros(bs, T, dtype=torch.long), refs=refs, cfg=cfg)
    dead = build_target(pg_grad_coef=None, advantages=None, **common)
    assert float(dead["c"].abs().max()) == 0.0
    assert torch.equal(dead["target_logprob"], t), "no coefficient => the target IS the teacher"
    assert float(dead["e"].abs().max()) == 1.0, "no advantages => the RLSD factor is inert"
    live = build_target(pg_grad_coef=torch.randn(bs, T, generator=g, dtype=torch.float64),
                        advantages=torch.randn(bs, T, generator=g, dtype=torch.float64), **common)
    assert float(live["c"].abs().max()) > 0.0
    assert not torch.equal(live["target_logprob"], t)
