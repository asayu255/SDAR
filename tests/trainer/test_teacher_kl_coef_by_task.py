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
    validate_teacher_kl_task_ids = DataParallelPPOActor.validate_teacher_kl_task_ids

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
    # The token-mean path scales BEFORE the mean. Since the online gate landed the
    # tensor it scales is _kld_for_loss, which IS teacher_kld whenever no gate is
    # in force -- the same expression, one indirection earlier.
    assert "_kld_for_loss * _kl_row_coef.reshape(-1, 1)" in src
    assert "_kld_for_loss = teacher_kld if _pb_w is None else teacher_kld * _pb_w" in src
    # the per-task-weighted branch
    assert "task_loss_weight * _kl_row_coef" in src


def test_the_config_is_plumbed_from_algorithm_opd():
    import inspect

    from verl.trainer.main_opd import inject_distillation_config

    src = inspect.getsource(inject_distillation_config)
    assert "teacher_kl_loss_coef_by_task" in src
    assert 'validate_kl_coef_by_task(\n            opd_cfg.get("kl_loss_coef_by_task", None)\n        )' in src, (
        "the value must pass through the startup validator on its way to the actor"
    )


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

    # the readout rides the same injection, and is off unless asked for
    assert build({}).teacher_kl_task_diag is False
    assert build({"task_diag": True}).teacher_kl_task_diag is True


def test_a_coefficient_outside_the_box_is_refused_at_startup_not_at_step_150():
    """The box is the design's clip. A value outside it is an intervention
    strength nobody chose, and it must not survive config injection."""
    import pytest
    from omegaconf import OmegaConf

    from verl.trainer.main_opd import inject_distillation_config

    def build(by_task):
        c = OmegaConf.create({
            "algorithm": {"opd": {"kl_loss_coef": 0.01, "kl_loss_coef_by_task": by_task}},
            "actor_rollout_ref": {"actor": {}, "model": {}, "rollout": {}, "ref": {}},
            "data": {}, "trainer": {},
        })
        inject_distillation_config(c)
        return c

    build({"alfworld": 1.076431, "search": 1.191101, "webshop": 0.5})  # the arm
    for bad in ({"webshop": 1.6}, {"webshop": 0.49}, {"webshop": float("nan")}, {}):
        with pytest.raises(ValueError):
            build(bad)


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
    # and the effective-coefficient metric is only emitted when b is set. It is
    # built from the CONFIG at the end of the call rather than read off the row
    # tensor per micro-batch: b_task is a constant, and the read was three host
    # syncs a micro-batch for a number nothing measured.
    assert "if _by_task:" in src
    assert "_coef = self.config.get('teacher_kl_loss_coef', 1.0)" in src
    assert "actor/teacher_kl_coef_effective/" in src
    assert "_kl_row_coef[" not in src, "the effective coefficient must not be read back off the device"


# ---------------------------------------------------------------------------
# review round 3: the ids themselves, the wiring on ONE forward, real gradients


def test_an_out_of_range_task_id_is_an_error_not_a_silent_one():
    """id 3 in a three-task run used to fall through and train at b = 1.

    Checked ONCE per update on the arranged batch, not per micro-batch: the ids
    do not depend on the student's forward, and every one of these checks reads
    a device value back to the host.
    """
    s = _Stub({"webshop": 0.5})
    with pytest.raises(AssertionError, match="outside the 3 tasks"):
        s.validate_teacher_kl_task_ids(torch.tensor([0, 1, 3]), TASKS)


def test_a_non_integer_task_id_is_an_error_not_truncated():
    """2.9 used to become 2 and be treated as webshop."""
    s = _Stub({"webshop": 0.5})
    with pytest.raises(AssertionError, match="non-integer"):
        s.validate_teacher_kl_task_ids(torch.tensor([0.0, 1.0, 2.9]), TASKS)


def test_the_validator_accepts_what_the_arm_actually_carries():
    s = _Stub({"alfworld": 1.076431, "search": 1.191101, "webshop": 0.5})
    s.validate_teacher_kl_task_ids(torch.tensor([0, 1, 2, 2, -1, -1]), TASKS)
    s.validate_teacher_kl_task_ids(torch.tensor([0.0, 1.0, 2.0]), TASKS)
    # unset -> nothing to validate, and no complaint about missing ids
    _Stub(None).validate_teacher_kl_task_ids(None, None)


def test_the_validator_still_refuses_a_typo_and_missing_ids():
    with pytest.raises(AssertionError, match="not tasks in"):
        _Stub({"webshopp": 0.5}).validate_teacher_kl_task_ids(torch.tensor([0, 1, 2]), TASKS)
    with pytest.raises(AssertionError, match="no task_ids"):
        _Stub({"webshop": 0.5}).validate_teacher_kl_task_ids(None, TASKS)


def test_the_hot_path_lookup_reads_nothing_back_to_the_host():
    """The whole point of hoisting the checks. A device->host read here is one
    sync per micro-batch per check; at ~500 micro-batches a step that was the
    largest remaining source of them in this block."""
    import ast
    import inspect

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    src = inspect.getsource(DataParallelPPOActor.teacher_kl_row_coef)
    tree = ast.parse(src.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            assert name not in ("item", "tolist", "nonzero"), f"{name}() syncs the host"
    assert "bool(" not in src, "a Python bool() of a device tensor is a sync"
    # and it is a single gather, not a per-task masked assignment
    assert "table[torch.where(" in src


def test_an_out_of_range_id_still_gets_one_from_the_lookup_itself():
    """Belt and braces: the validator is what refuses it, but if one ever
    reached the lookup it must land on the fallback slot rather than index out
    of bounds or silently take another task's coefficient."""
    c = _coef({"alfworld": 1.5, "search": 1.4, "webshop": 0.5}, [0, 1, 2, 7, -1])
    assert c.tolist() == pytest.approx([1.5, 1.4, 0.5, 1.0, 1.0], rel=1e-6)


def test_padding_rows_with_a_negative_id_are_exempt():
    """They are masked out of the loss by task_loss_weight = 0 already."""
    c = _coef({"webshop": 0.5}, [0, 1, 2, -1, -1])
    assert c.tolist() == [1.0, 1.0, 0.5, 1.0, 1.0]


def test_a_length_mismatch_between_ids_and_rows_is_an_error():
    with pytest.raises(AssertionError, match="entries for 3 rows"):
        _coef({"webshop": 0.5}, [0, 1], n=3)


def test_wiring_check_compares_the_same_forward_before_and_after_the_coefficient():
    """The per-task weighted KL with and without b, from ONE kld tensor.

    This is the check the launch runs in its first steps. It must be a
    same-forward comparison: two arms' KL losses are not b-fold apart once their
    parameters have diverged.
    """
    kld = torch.tensor([[1.0, 3.0], [2.0, 2.0], [4.0, 0.0], [1.0, 1.0]])
    mask = torch.ones(4, 2)
    w = torch.tensor([0.25, 0.25, 0.25, 0.25])
    ids = [0, 1, 2, 2]
    b = _coef({"alfworld": 1.076, "search": 1.191, "webshop": 0.5}, ids)

    row_kl = (kld * mask).sum(-1)
    before = row_kl * w
    after = row_kl * (w * b)
    for tid, name, expect in ((0, "alfworld", 1.076), (1, "search", 1.191), (2, "webshop", 0.5)):
        sel = torch.tensor([t == tid for t in ids])
        ratio = float(after[sel].sum() / before[sel].sum())
        assert ratio == pytest.approx(expect, abs=1e-6), name


def test_a_small_model_gets_the_intended_per_task_gradient_scaling():
    """Not a stub: a real parameter, a real KL-shaped loss, real autograd.

    Each task's rows feed a separate parameter so the per-task gradient can be
    read off directly; the coefficient must scale exactly that task's gradient
    and leave the others untouched.
    """
    torch.manual_seed(0)
    theta = torch.zeros(3, requires_grad=True)             # one scalar per task
    ids = [0, 0, 1, 2, 2, 2]
    target = torch.tensor([0.3, -0.2, 0.5, -0.4, 0.1, 0.6])
    w = torch.full((6,), 1.0 / 6)

    def loss_with(b_vec):
        # "row_kl" = squared distance of the task's parameter to a target: a
        # convex stand-in with a nonzero, task-separable gradient
        row_kl = torch.stack([(theta[t] - target[i]) ** 2 for i, t in enumerate(ids)])
        return (row_kl * (w * b_vec)).sum()

    ones = torch.ones(6)
    g_control = torch.autograd.grad(loss_with(ones), theta)[0]
    b = _coef({"alfworld": 1.076, "search": 1.191, "webshop": 0.5}, ids)
    g_treat = torch.autograd.grad(loss_with(b), theta)[0]

    ratio = (g_treat / g_control).tolist()
    assert ratio == pytest.approx([1.076, 1.191, 0.5], abs=1e-6)
