"""Per-task teacher retirement on the measured first-order effect.

What has to hold:

* nothing retires before the warmup or before the window is full, however negative
  the term looks -- the control's own steps are negative 20-25% of the time on the
  two tasks that must never retire;
* a task whose window mean turns negative retires, and only that task;
* retirement is permanent and survives a checkpoint round trip;
* the multipliers reach the actor's per-task coefficient, composing with a
  configured one rather than replacing it;
* a run without algorithm.opd.retire builds no controller and passes nothing.
"""

import random

import pytest
from omegaconf import OmegaConf


def _ctl(tasks=("alfworld", "webshop", "search"), **kw):
    from verl.trainer.ppo.teacher_retirement import TeacherRetirement

    return TeacherRetirement(tasks, **kw)


def _feed(ctl, values, start=1, tasks=("alfworld", "webshop", "search")):
    """Drive one task's series; the others stay positive."""
    out = []
    for i, v in enumerate(values):
        m = {f"opd/{t}/grpo/first_order": (v if t == tasks[1] else 1e-4) for t in tasks}
        out.append(ctl.observe(m, start + i))
    return out


def test_nothing_retires_before_the_warmup_or_a_full_window():
    ctl = _ctl(window=25, warmup=50)
    # 40 steps of a clearly negative term: the window fills, the warmup does not.
    _feed(ctl, [-1e-3] * 40)
    assert ctl.retired == {}
    assert ctl.coef_by_task() == {"alfworld": 1.0, "webshop": 1.0, "search": 1.0}
    # A controller started at the warmup still needs a full window.
    late = _ctl(window=25, warmup=50)
    _feed(late, [-1e-3] * 10, start=60)
    assert late.retired == {}


def test_the_task_whose_window_turns_negative_retires_and_only_that_one():
    ctl = _ctl(window=25, warmup=50)
    _feed(ctl, [-1e-3] * 60)
    assert set(ctl.retired) == {"webshop"}
    # window full at step 25, warmup satisfied at 50 -> it fires exactly at 50
    assert ctl.retired["webshop"] == 50
    assert ctl.coef_by_task() == {"alfworld": 1.0, "webshop": 0.0, "search": 1.0}


def test_a_healthy_task_with_noisy_steps_does_not_retire():
    # The control reads a positive first_order on 75-80% of steps for the two tasks
    # that never collapsed; a quarter of the steps being negative must not fire.
    rng = random.Random(0)
    ctl = _ctl(window=25, warmup=50)
    for step in range(1, 301):
        v = 3e-4 if rng.random() < 0.78 else -2e-4
        ctl.observe({f"opd/{t}/grpo/first_order": v for t in ctl.tasks}, step)
    assert ctl.retired == {}


def test_retirement_is_permanent_and_survives_a_checkpoint():
    ctl = _ctl(window=5, warmup=0)
    _feed(ctl, [-1e-3] * 6)
    assert "webshop" in ctl.retired
    state = ctl.state_dict()

    back = _ctl(window=5, warmup=0)
    back.load_state_dict(state)
    assert back.retired == ctl.retired
    assert back.coef_by_task()["webshop"] == 0.0
    # A retired task stops producing the metric; it must not come back.
    for step in range(100, 140):
        back.observe({"opd/alfworld/grpo/first_order": 1e-4}, step)
    assert back.retired == ctl.retired
    assert back.coef_by_task()["webshop"] == 0.0


def test_the_metrics_name_the_step_it_fired():
    ctl = _ctl(window=5, warmup=0)
    outs = _feed(ctl, [-1e-3] * 6)
    fired = [o for o in outs if o.get("opd/webshop/retired") == 1.0]
    assert fired and fired[0]["opd/webshop/retire_step"] == float(ctl.retired["webshop"])
    assert outs[-1]["opd/retired_tasks"] == 1.0
    assert outs[0]["opd/webshop/first_order_window"] == 1.0


def test_the_multiplier_composes_with_a_configured_per_task_coefficient():
    import torch

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class _Stub:
        config = OmegaConf.create({"teacher_kl_loss_coef_by_task": {"webshop": 0.5}})

    names = ["alfworld", "webshop", "search"]
    ids = torch.tensor([0, 1, 2, 1])
    got = DataParallelPPOActor.teacher_kl_row_coef(
        _Stub(), ids, names, 4, device=torch.device("cpu"), dtype=torch.float32,
        runtime_by_task={"alfworld": 1.0, "webshop": 0.0, "search": 1.0})
    assert got.tolist() == [1.0, 0.0, 1.0, 0.0]

    # ...and with no configured one, the multipliers stand alone.
    class _Plain:
        config = OmegaConf.create({})

    got2 = DataParallelPPOActor.teacher_kl_row_coef(
        _Plain(), ids, names, 4, device=torch.device("cpu"), dtype=torch.float32,
        runtime_by_task={"alfworld": 1.0, "webshop": 0.0, "search": 1.0})
    assert got2.tolist() == [1.0, 0.0, 1.0, 0.0]
    # Neither set: None, which is what every run that does not retire passes.
    assert DataParallelPPOActor.teacher_kl_row_coef(
        _Plain(), ids, names, 4, device=torch.device("cpu"), dtype=torch.float32) is None


def test_a_run_without_the_switch_builds_no_controller():
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    class _Self:
        pass

    me = _Self()
    me.config = OmegaConf.create({"algorithm": {"opd": {"kl_loss_coef": 0.01}},
                                  "env": {"multitask": {"tasks": ["alfworld", "webshop", "search"]}}})
    assert OPDRayTrainer._teacher_retirement(me) is None

    me.config.algorithm.opd.retire = {"enable": True, "window": 25, "warmup": 50}
    ctl = OPDRayTrainer._teacher_retirement(me)
    assert ctl is not None and ctl.window == 25 and ctl.warmup == 50
    assert ctl.coef_by_task() == {"alfworld": 1.0, "webshop": 1.0, "search": 1.0}
    # built once and kept, so its window is not reset every step
    assert OPDRayTrainer._teacher_retirement(me) is ctl
