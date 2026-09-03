"""Training and validation as two processes, not one loop.

The Stage-2 run scripts train with ``trainer.test_freq=-1`` and do not validate;
``examples/opd_trainer/eval_checkpoints.sh`` scores the saved checkpoints
afterwards, one process per checkpoint.

The separation is not tidiness. alfworld draws each episode from TextWorld's
seeded game-file cycle, which is stateful: every reset advances it, so a second
validation inside one process plays different games than the first. Two
checkpoints scored in one process are therefore not scored on the same episodes,
and a difference between them carries a difference in what they were asked to do.
A fresh process rebuilds the cycle from ``env.seed`` at position 0.

What that costs is a model load and an env build per checkpoint. What it needs
from the trainer is three things, each of which was missing or wrong here:

1. ``trainer.val_only`` has to be reachable. It was nested inside the
   ``val_before_train`` branch, and every Stage-2 script passes
   ``val_before_train=False`` -- so a process asked to validate would have
   trained instead, silently, for as long as the training run takes.
2. A validation process must not load the Stage-1 pool. It never draws a batch,
   and the pool is ~149 GiB resident on a box measured at 98% of host RAM.
3. On the student-indexed arm it must not build the three teachers either. They
   exist to be scored during the update, and there is no update.
"""

import pathlib
import re

import pytest

_OPD = pathlib.Path(__file__).resolve().parents[2] / "examples/opd_trainer"
_EVAL = _OPD / "eval_checkpoints.sh"
_STAGE2 = [
    _OPD / "run_multitask_offpolicy_qwen3_nogen.sh",
    _OPD / "run_multitask_offpolicy_studenttopk_qwen3.sh",
]


# --------------------------------------------------------------------------- #
# the scripts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", _STAGE2, ids=lambda p: p.name)
def test_a_stage2_script_does_not_validate_inline(path):
    """test_freq=-1 is the half of the separation that lives in the run script."""
    assert re.search(r"^\s*trainer\.test_freq=-1\s*\\?$", path.read_text(), re.MULTILINE), (
        f"{path.name} still validates during training; the eval harness assumes it does not"
    )


def test_the_eval_harness_exists_and_is_runnable():
    assert _EVAL.exists(), "the other half of test_freq=-1 is missing"
    text = _EVAL.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "val_only=True" in text
    assert "resume_mode=resume_path" in text
    assert "resume_from_path" in text


def test_the_harness_adds_only_what_the_run_script_does_not_pass():
    """It runs the arm's own script rather than repeating its arguments.

    That is what makes the model, the tasks, the episode caps, the per-task val
    sampling and the intent lock the training run's by construction. A harness
    that restated any of them would drift from the arm it is supposed to be
    scoring, and nothing would say so.
    """
    text = _EVAL.read_text()
    passed = set(re.findall(r"^\s+(trainer\.\S+?)=", text, re.MULTILINE))
    assert passed == {"trainer.val_only", "trainer.resume_mode", "trainer.resume_from_path"}, (
        f"the harness passes trainer keys beyond the three evaluation needs: {passed}"
    )
    for restated in ("actor_rollout_ref.model.path", "data.val_files", "env.multitask.tasks",
                     "trainer.experiment_name", "trainer.default_local_dir"):
        assert f"{restated}=" not in text, (
            f"{restated} is restated in the harness; it must come from the run script"
        )


def test_the_harness_reads_the_checkpoint_directory_out_of_the_run_script():
    """Restating the path is how the two drift. It is parsed instead."""
    text = _EVAL.read_text()
    assert "trainer.default_local_dir" in text and "grep" in text
    # and the parse has to survive the trailing backslash the run script has
    assert "sed" in text


def test_evaluation_raises_the_vllm_budget_through_the_run_script_s_hook():
    """The one setting evaluation genuinely wants different, and the run script
    reads it from the environment rather than the harness passing it twice."""
    assert "ROLLOUT_GPU_MEM_UTIL" in _EVAL.read_text()
    for path in _STAGE2:
        assert "gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.3}" in path.read_text(), (
            f"{path.name} hardcodes the vLLM budget; the harness cannot raise it for eval"
        )


# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #


def test_val_only_is_read_on_its_own_not_nested_under_val_before_train():
    """The bug this pins would have made every eval process a training run.

    Source-level because the alternative is standing up Ray, and because the
    defect is structural: the flag was correct, it was the nesting that made it
    unreachable for exactly the scripts that use it.
    """
    import inspect

    from verl.trainer.ppo import opd_offpolicy_ray_trainer as mod

    src = inspect.getsource(mod.OffPolicyOPDRayTrainer.fit)
    guard = re.search(r"if self\.val_reward_fn is not None and \((.*?)\):", src, re.DOTALL)
    assert guard, "could not find the validation guard in fit()"
    condition = guard.group(1)
    assert "val_only" in condition, (
        "val_only is not in the guard, so a val_only process falls through to training "
        "whenever val_before_train is False -- which every Stage-2 script sets"
    )


@pytest.mark.parametrize("val_only", [True, False])
def test_the_pool_is_loaded_only_when_a_batch_will_be_drawn(val_only):
    from unittest.mock import patch

    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    from tests.trainer.test_offpolicy_student_topk import _Cfg

    trainer = OffPolicyOPDRayTrainer.__new__(OffPolicyOPDRayTrainer)
    trainer.use_reference_policy = False
    trainer.config = _Cfg(
        algorithm=_Cfg(opd=_Cfg(teacher_data_dir="/pool", student_indexed_topk=False)),
        actor_rollout_ref=_Cfg(actor=_Cfg(teacher_kl_loss_type="topk_kl", teacher_kl_topk=20,
                                          student_indexed_topk=False)),
        data=_Cfg(task_balance=_Cfg(per_task_batch_size=15)),
        env=_Cfg(rollout=_Cfg(n=8)),
        trainer=_Cfg(val_only=val_only),
    )
    loaded = []
    trainer._load_offpolicy_data = lambda: loaded.append(True)
    with patch.object(RayPPOTrainer, "__init__", lambda self, *a, **k: None):
        OffPolicyOPDRayTrainer.__init__(trainer)

    assert trainer.val_only is val_only
    assert loaded == ([] if val_only else [True]), (
        "a val_only process loaded the pool it never reads"
        if val_only else "a training process skipped the pool it trains on"
    )
    if val_only:
        # and the attributes the rest of the class expects still exist, empty
        assert trainer._task_shards == {} and trainer._task_to_trajs == {}


def test_init_workers_skips_the_teachers_in_a_val_only_run():
    """Three 1.7B models loaded onto the cards the rollout is about to size its
    KV cache against, and queried by nothing."""
    import inspect

    from verl.trainer.ppo import opd_offpolicy_ray_trainer as mod

    src = inspect.getsource(mod.OffPolicyOPDRayTrainer.init_workers)
    head = src[: src.index("return super().init_workers()")]
    assert "self.val_only" in head, (
        "init_workers does not consider val_only, so an eval process builds the teachers"
    )
    assert "not self.student_indexed_topk or self.val_only" in head
