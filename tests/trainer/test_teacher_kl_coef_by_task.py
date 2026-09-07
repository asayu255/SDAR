"""Per-task multiplier on the teacher-KL coefficient.

``teacher_kl_loss_coef`` is one scalar for all tasks, so distilling one task
less than another was not expressible. The cross-effect measurement (theory doc
section 4.14) is per teacher -- how much task j's teacher moves the OTHER tasks'
reward -- so acting on it needs a per-task coefficient.

These tests pin the three properties the wiring has to have: b = 1 reproduces
the old behaviour exactly, a coefficient that cannot be applied stops the run
instead of training uniformly, and the multiplier reaches BOTH aggregation paths
(the per-task-weighted one and the plain token-mean).
"""

import pytest
import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor

TASKS = ["alfworld", "search", "webshop"]


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


class _Stub:
    teacher_kl_row_coef = DataParallelPPOActor.teacher_kl_row_coef

    def __init__(self, by_task=None):
        self.config = _Cfg(teacher_kl_loss_coef_by_task=by_task)


def _coef(by_task, task_ids, names=TASKS, n=None):
    s = _Stub(by_task)
    t = torch.tensor(task_ids)
    return s.teacher_kl_row_coef(t, names, n or len(task_ids),
                                device=torch.device("cpu"), dtype=torch.float32)


def test_unset_is_none_so_the_old_path_is_taken_unchanged():
    assert _coef(None, [0, 1, 2]) is None
    assert _coef({}, [0, 1, 2]) is None


def test_the_multiplier_lands_on_the_right_rows():
    c = _coef({"webshop": 0.5}, [0, 1, 2, 2, 0])
    assert c.tolist() == [1.0, 1.0, 0.5, 0.5, 1.0]


def test_all_three_can_be_set_including_above_one():
    c = _coef({"alfworld": 1.17, "search": 1.32, "webshop": 0.51}, [0, 1, 2])
    assert c.tolist() == pytest.approx([1.17, 1.32, 0.51], abs=1e-6)


def test_a_task_absent_from_the_dict_keeps_one():
    c = _coef({"webshop": 0.0}, [0, 1, 2])
    assert c.tolist() == [1.0, 1.0, 0.0]


def test_it_refuses_rather_than_silently_training_uniform():
    """Configured-but-inapplicable is the failure mode this project keeps hitting."""
    with pytest.raises(AssertionError, match="no task_ids"):
        _Stub({"webshop": 0.5}).teacher_kl_row_coef(
            None, TASKS, 3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(AssertionError, match="no task_id_names"):
        _coef({"webshop": 0.5}, [0, 1, 2], names=[])


def test_a_typo_in_a_task_name_is_an_error_not_a_no_op():
    with pytest.raises(AssertionError, match="webshopp"):
        _coef({"webshopp": 0.5}, [0, 1, 2])


def test_the_weighted_path_multiplies_the_row_weight():
    """b_task * task_loss_weight is what reaches the weighted sum."""
    kld = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    mask = torch.ones(3, 2)
    w = torch.tensor([0.5, 0.5, 0.5])
    row_kl = (kld * mask).sum(-1)                      # [2, 4, 8]

    base = float((row_kl * w).sum())                   # 1 + 2 + 4 = 7
    c = _coef({"webshop": 0.25}, [0, 1, 2])
    got = float((row_kl * (w * c)).sum())              # 1 + 2 + 1 = 4
    assert base == pytest.approx(7.0)
    assert got == pytest.approx(4.0)


def test_the_token_mean_path_scales_before_the_mean_and_keeps_the_denominator():
    """The unweighted path scales the matrix, not the mask."""
    from verl.trainer.ppo.core_algos import agg_loss

    kld = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    mask = torch.ones(3, 2)
    plain = float(agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean"))
    c = _coef({"webshop": 0.0}, [0, 1, 2])
    scaled = float(agg_loss(loss_mat=kld * c.reshape(-1, 1), loss_mask=mask,
                            loss_agg_mode="token-mean"))
    assert plain == pytest.approx(1.0)
    # two of three rows survive, and the denominator is still all six tokens
    assert scaled == pytest.approx(4.0 / 6.0)


def test_both_aggregation_paths_are_wired_in_update_policy():
    """Neither branch may be left on the uniform coefficient."""
    import ast
    import inspect

    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    src = ast.unparse(fn)
    assert "_kl_row_coef = self.teacher_kl_row_coef(" in src
    # the token-mean branch
    assert "loss_mat=teacher_kld * _kl_row_coef.reshape(-1, 1)" in src
    # the per-task-weighted branch
    assert "task_loss_weight * _kl_row_coef" in src


def test_the_config_is_plumbed_from_algorithm_opd():
    import inspect

    from verl.trainer.main_opd import inject_distillation_config

    src = inspect.getsource(inject_distillation_config)
    assert "teacher_kl_loss_coef_by_task" in src
    assert 'opd_cfg.get(\n            "kl_loss_coef_by_task", None\n        )' in src


def test_hydra_passes_the_dict_form_this_reads():
    """The launch writes {alfworld:1.0,...}; confirm that is what arrives."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    v = OverridesParser.create().parse_overrides(
        ["algorithm.opd.kl_loss_coef_by_task={alfworld:1.0,search:1.32,webshop:0.51}"]
    )[0].value()
    got = {str(k): float(x) for k, x in dict(v).items()}
    assert got == {"alfworld": 1.0, "search": 1.32, "webshop": 0.51}


def test_injection_reaches_the_actor_and_the_default_stays_uniform():
    from omegaconf import OmegaConf

    from verl.trainer.main_opd import inject_distillation_config

    def build(opd_extra):
        c = OmegaConf.create({
            "algorithm": {"opd": {"kl_loss_coef": 0.01, **opd_extra}},
            "actor_rollout_ref": {"actor": {}, "model": {}, "rollout": {}, "ref": {}},
            "data": {}, "trainer": {},
        })
        inject_distillation_config(c)
        return c.actor_rollout_ref.actor

    a = build({})
    assert a.get("teacher_kl_loss_coef_by_task", "MISSING") is None

    a = build({"kl_loss_coef_by_task": {"webshop": 0.5}})
    assert dict(a.teacher_kl_loss_coef_by_task) == {"webshop": 0.5}
    # the base coefficient is untouched: the effective one is the product
    assert float(a.teacher_kl_loss_coef) == 0.01


def test_the_expectations_files_pin_it_to_null():
    """The control has to declare that it did NOT use per-task coefficients."""
    from verl.utils.expected_config import load_expectations

    for f in ("examples/opd_grpo_trainer/expected_multitask_config.yaml",
              "examples/opd_trainer/expected_multitask_config.yaml"):
        exp = load_expectations(f)
        assert "actor_rollout_ref.actor.teacher_kl_loss_coef_by_task" in exp, f
        assert exp["actor_rollout_ref.actor.teacher_kl_loss_coef_by_task"] is None, f


def test_unset_takes_the_original_expressions_in_both_branches():
    """b unset must reproduce the previous loss exactly, not approximately."""
    import ast
    import inspect

    import verl.workers.actor.dp_actor as m

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(m)))
              if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    src = ast.unparse(fn)
    assert ("if _kl_row_coef is None:\n"
            "                            policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef"
            ) in src.replace("\n                        ", "\n                            ") or (
        "policy_loss = policy_loss + teacher_kl_loss * teacher_kl_coef" in src)
    assert ("_row_w = task_loss_weight if _kl_row_coef is None "
            "else task_loss_weight * _kl_row_coef") in src
    # and the metric is only emitted when the coefficient is actually set
    assert "if _kl_row_coef is not None:" in src
    assert src.index("if _kl_row_coef is not None:") < src.index("if task_loss_weight is None:")
