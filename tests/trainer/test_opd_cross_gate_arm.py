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

"""The cross gate as WIRED: where the actor reads and applies it, what the
driver attaches, what the checkpoint carries, what the injection refuses.

The module's arithmetic is tested in test_opd_cross_gate.py. What can still go
wrong is the plumbing, and each assertion here is a way it went wrong for the
pushback arm or could for this one: the gate decided after the loss, the
statistics folded before the backward, a name read at the tail of update_policy
after the loop rebound it, a column the driver attaches but the actor never
selects, a controller the checkpoint forgets.
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
# 1. order of operations inside update_policy


def test_the_gate_is_decided_before_the_loss_and_the_stats_folded_after_the_backward():
    src, _ = _update_policy_src()
    i_fwd = src.index("cross_gate_forward(")
    i_loss = src.index("_kld_for_loss = teacher_kld if _pb_w is None else teacher_kld * _pb_w")
    i_bwd = src.index("loss.backward()")
    i_stats = src.index("cross_stats.update(")
    assert i_fwd < i_loss < i_bwd < i_stats


def test_the_gate_multiplies_the_kl_through_the_same_line_the_pushback_gate_used():
    """One line applies whichever per-token weight is in force. The cross gate
    hands its w to that line rather than adding a second multiplication."""
    src, _ = _update_policy_src()
    assert "_pb_w = _cg['w'].to(teacher_kld.dtype)" in src
    assert src.count("_kld_for_loss = teacher_kld if _pb_w is None else teacher_kld * _pb_w") == 1


def test_the_forward_reads_the_clipped_pg_coefficient_and_beta_without_lambda():
    src, _ = _update_policy_src()
    call = src[src.index("cross_gate_forward("):]
    call = call[:call.index("_pb_w = _cg")]
    assert "pg_grad_coef=xt_pg_grad_coef" in call, "the reference is built from dL/dlogp, never from A"
    assert "opd_coef=_teacher_kl_coef_scalar" in call, "beta in, lambda out"
    assert "roles=_cg_roles" in call and "token_roles(responses, sign_role_tags)" in src


def test_the_per_micro_batch_names_are_reset_together():
    src, _ = _update_policy_src()
    i_pb = src.index("_pb_w = None")
    i_cg = src.index("_cg_pending = None")
    assert 0 < i_cg - i_pb < 200, "_cg_pending is reset where _pb_w is, once per micro-batch"


def test_references_and_lambda_are_read_once_per_step_and_updated_once_from_the_reduced_sums():
    src, _ = _update_policy_src()
    assert src.count("cross_gate.refs_to_device(") == 1
    assert src.count("cross_gate.update(") == 1
    assert src.count("cross_stats.reduced()") == 1
    # lambda as applied is reported from the refs read at the top, not recomputed
    assert "actor/cross/lambda_applied/" in src
    # and the controller is refused alongside the pushback controller
    assert "teacher_kl_cross_gate and teacher_kl_pushback are both enabled" in src


def test_the_prompt_keys_are_read_before_the_loop_and_the_tail_never_touches_the_rebound_name():
    """The P1 regression, for this arm: `data` is rebound in the loop, so the
    key list is captured at the top and the tail reads only that local."""
    src, fn = _update_policy_src()
    i_keys = src.index("_cg_prompt_keys = list(data.meta_info.get('cross_prompt_keys'")
    i_loop = src.index("_pb_w = None")
    assert i_keys < i_loop
    rebound = set()
    for n in ast.walk(fn):
        tgts = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, (ast.For, ast.AsyncFor)) else [])
        for t in tgts:
            if isinstance(t, ast.Name):
                rebound.add(t.id)
    assert "data" in rebound
    tail = src[src.index("cross_gate.update("):]
    assert "data.meta_info" not in tail and "data.get(" not in tail.split("for name, entries")[0]


# ---------------------------------------------------------------------------
# 2. the columns: driver -> micro-batch -> accumulator


def test_the_split_columns_reach_the_micro_batch_and_the_driver_attaches_them():
    import verl.trainer.ppo.opd_ray_trainer as t

    src, _ = _update_policy_src()
    assert "for _k in ('cross_side', 'cross_prompt_idx'):" in src
    assert "select_keys.append(_k)" in src
    assert "side=data.get('cross_side', None)" in src
    assert "prompt_idx=data.get('cross_prompt_idx', None)" in src
    tsrc = inspect.getsource(t)
    assert 'batch.batch["cross_side"] = _cols["side"]' in tsrc
    assert 'batch.batch["cross_prompt_idx"] = _cols["prompt_idx"]' in tsrc
    assert 'batch.meta_info["cross_prompt_keys"] = list(_cols["keys"])' in tsrc
    assert 'batch.meta_info["cross_unkeyed_rows"]' in tsrc
    # the split is keyed on the turn-0 anchor of the GRPO group, by task
    assert 'turn_steps=_nt.get("turn_step"' in tsrc and 'anchors=_nt.get("anchor_obs"' in tsrc
    assert "task_names=get_task_names(batch)" in tsrc
    # and padding rows are excluded from it
    assert "PADDING_ROW_KEY" in tsrc.split("cross_gate_prompt_columns(")[1].split(")")[0] or \
        "_pad = batch.batch.get(PADDING_ROW_KEY" in tsrc


# ---------------------------------------------------------------------------
# 3. the checkpoint


def test_the_checkpoint_carries_the_controller_state():
    import verl.workers.actor.dp_actor as m

    src = inspect.getsource(m.DataParallelPPOActor.actor_extra_state_dict)
    assert '"cross_gate"' in src
    src = inspect.getsource(m.DataParallelPPOActor.load_actor_extra_state_dict)
    assert 'sd.get("cross_gate", None)' in src and "_cross_gate_pending_state" in src
    src = inspect.getsource(m.DataParallelPPOActor.cross_gate_controller)
    assert "_cross_gate_pending_state" in src and "model_vocab_size(self.actor_module)" in src


# ---------------------------------------------------------------------------
# 4. the injection refuses anything stacked on the gate


def _opd(**kw):
    base = {"task_diag": True}
    base.update(kw)
    return base


def test_exclusivity_is_enforced_at_injection():
    from verl.trainer.main_opd import validate_cross_gate_exclusivity, validate_pushback_exclusivity

    cg = {"enable": True, "eps_cross": 0.0, "rho": 1e4, "lambda_max": 0.2}
    # clean
    validate_cross_gate_exclusivity(cg, None, True, _opd(cross_gate=cg))
    # a static coefficient other than 1
    with pytest.raises(ValueError, match="does not stack"):
        validate_cross_gate_exclusivity(cg, {"alfworld": 0.9}, True, _opd(cross_gate=cg))
    validate_cross_gate_exclusivity(cg, {"alfworld": 1.0}, True, _opd(cross_gate=cg))
    # the self gate
    with pytest.raises(ValueError, match="pushback_control.enable=True"):
        validate_cross_gate_exclusivity(cg, None, True, _opd(cross_gate=cg, pushback_control={"enable": True}))
    # and from the other side
    with pytest.raises(ValueError, match="cross_gate.enable=True"):
        validate_pushback_exclusivity({"enable": True, "eps": 0.003}, None, True,
                                      _opd(pushback_control={"enable": True}, cross_gate=cg))
    # the cross-teacher weightings
    for other in ("sign_weight", "cross_teacher_kl_weight", "cross_teacher_target"):
        with pytest.raises(ValueError, match=f"{other}.enable=True"):
            validate_cross_gate_exclusivity(cg, None, True, _opd(cross_gate=cg, **{other: {"enable": True}}))
    # the readout it needs
    with pytest.raises(ValueError, match="task_diag=True"):
        validate_cross_gate_exclusivity(cg, None, False, _opd(cross_gate=cg))
    # a bad knob is refused here, at startup
    with pytest.raises(ValueError):
        validate_cross_gate_exclusivity({"enable": True, "lambda_max": 1.5}, None, True, {})
    # off is off
    validate_cross_gate_exclusivity({"enable": False, "lambda_max": 1.5}, {"alfworld": 0.5}, False, {})
    validate_cross_gate_exclusivity(None, None, False, {})


def test_the_injection_copies_the_knobs_onto_the_actor_and_refuses_stacking(tmp_path):
    """Through the real injection, on a config with the cross gate on."""
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
        "+algorithm.opd.cross_gate.enable=True",
        "+algorithm.opd.cross_gate.eps_cross=0.0",
        "+algorithm.opd.cross_gate.rho=10000.0",
        "+algorithm.opd.cross_gate.lambda_max=0.2",
        "+algorithm.opd.cross_gate.roles=[format,env_action]",
    ]
    os.environ.setdefault("RUN_TAG_SUFFIX", "")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=base)
    inject_distillation_config(cfg)
    got = cfg.actor_rollout_ref.actor.teacher_kl_cross_gate
    assert got is not None and got["enable"] is True and float(got["rho"]) == 1e4
    assert list(got["roles"]) == ["format", "env_action"]
    assert cfg.actor_rollout_ref.actor.teacher_kl_pushback is None
    # stacked on the self gate: refused
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg2 = compose(config_name="ppo_trainer", overrides=base + [
            "+algorithm.opd.pushback_control.enable=True", "+algorithm.opd.pushback_control.eps=0.003"])
    with pytest.raises(ValueError):
        inject_distillation_config(cfg2)


# ---------------------------------------------------------------------------
# 5. every name the inserted blocks READ is bound earlier in update_policy
#
# The pushback arm's first update died on a name that a string test could not
# see was rebound. This is the cheapest check that survives: for each name the
# cross-gate blocks read, there is an assignment, a for-target, a with-target or
# a parameter EARLIER in the function's source.


def _bound_names_before(fn, lineno):
    bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if getattr(n, "lineno", 10**9) >= lineno:
            continue
        tgts = []
        if isinstance(n, (ast.Assign,)):
            tgts = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor)):
            tgts = [n.target]
        elif isinstance(n, ast.NamedExpr):
            tgts = [n.target]
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            tgts = [it.optional_vars for it in n.items if it.optional_vars is not None]
        for t in tgts:
            for leaf in ast.walk(t):
                if isinstance(leaf, ast.Name):
                    bound.add(leaf.id)
    return bound


def _names_read(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def test_every_name_the_cross_gate_blocks_read_is_bound_earlier():
    import builtins

    import verl.workers.actor.dp_actor as m

    src = inspect.getsource(m)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    module_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    module_names |= {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
                     if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    module_names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    module_names |= set(dir(builtins))

    # the blocks: found by a string each one alone contains
    markers = (
        "cross_gate_forward(",          # the gate, before the loss
        "cross_stats.update(",          # the fold, after the backward
        "cross_gate.update(",           # the controller, at the tail
        "cross_gate.refs_to_device(",   # the once-per-step read
    )
    checked = 0
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.Expr, ast.If)):
            continue
        text = ast.unparse(node)
        if not any(mk in text for mk in markers):
            continue
        # restrict to the statement that contains the marker, not an enclosing if
        stmts = [node] if not isinstance(node, ast.If) else [
            s for s in node.body if any(mk in ast.unparse(s) for mk in markers)]
        for s in stmts:
            bound = _bound_names_before(fn, s.lineno) | module_names
            missing = sorted(n for n in _names_read(s) if n not in bound)
            assert not missing, f"unbound at line {s.lineno}: {missing}\n{ast.unparse(s)[:200]}"
            checked += 1
    assert checked >= 4, f"expected to find the four cross-gate statements, found {checked}"
