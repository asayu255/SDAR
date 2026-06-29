"""Unit tests for off-policy multitask distillation (offline KD).

Covered (CPU-only; Ray / workers are bypassed):
* the fixed teacher dataset round-trips through ``DataProto.save_to_disk`` and
  ``OffPolicyOPDRayTrainer._load_offpolicy_data`` with correct per-task indexing;
* ``OffPolicyOPDRayTrainer._offpolicy_batch_iter`` yields the right number of
  task-balanced steps, each with ``per_task_traj_per_step`` rows per task drawn
  from the matching task;
* the top-k teacher-KL math (``topk_kl_per_token``) is the *same* one OPD uses:
  zero when the student matches the teacher, positive when it does not.

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.trainer.ppo.core_algos import topk_kl_per_token
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _make_task_proto(task, n, resp_len=4, k=3):
    """A minimal teacher-trajectory DataProto for one task."""
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros((n, resp_len), dtype=torch.long),
            "teacher_topk_logprobs": torch.full((n, resp_len, k), -1.0),
            "teacher_topk_ids": torch.zeros((n, resp_len, k), dtype=torch.long),
        },
        non_tensors={"task_name": np.array([task] * n, dtype=object)},
    )


def _bare_trainer():
    """An OffPolicyOPDRayTrainer with __init__ bypassed (only the fixed-data
    helpers are exercised)."""
    return object.__new__(OffPolicyOPDRayTrainer)


def test_load_offpolicy_data_concats_and_indexes_per_task(tmp_path):
    _make_task_proto("alfworld", 5).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 3).save_to_disk(str(tmp_path / "search.pt"))
    _make_task_proto("webshop", 4).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()

    assert len(trainer.offpolicy_data) == 12
    sizes = {t: len(idx) for t, idx in trainer._task_to_indices.items()}
    assert sizes == {"alfworld": 5, "search": 3, "webshop": 4}
    # Every index for a task must actually point at that task's rows.
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    for task, idx in trainer._task_to_indices.items():
        assert all(task_names[i] == task for i in idx)


def test_offpolicy_batch_iter_is_task_balanced(tmp_path):
    trainer = _bare_trainer()
    trainer.offpolicy_data = DataProto.concat(
        [_make_task_proto("alfworld", 10), _make_task_proto("search", 10), _make_task_proto("webshop", 10)]
    )
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    trainer._task_to_indices = {
        t: np.where(task_names == t)[0] for t in ("alfworld", "search", "webshop")
    }
    trainer.per_task_traj_per_step = 4
    trainer.total_training_steps = 6
    trainer.config = OmegaConf.create({"data": {"seed": 1}})

    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 6
    for batch in steps:
        assert len(batch) == 4 * 3  # per_task_traj_per_step * num_tasks
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        assert counts == {"alfworld": 4, "search": 4, "webshop": 4}


def test_pool_recycles_when_drawing_more_than_available():
    """Drawing more per step than a task's pool size must still succeed
    (pool reshuffled + recycled) and stay balanced."""
    trainer = _bare_trainer()
    trainer.offpolicy_data = DataProto.concat(
        [_make_task_proto("alfworld", 3), _make_task_proto("search", 3)]
    )
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    trainer._task_to_indices = {t: np.where(task_names == t)[0] for t in ("alfworld", "search")}
    trainer.per_task_traj_per_step = 5  # > 3 available per task
    trainer.total_training_steps = 2
    trainer.config = OmegaConf.create({"data": {"seed": 7}})

    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 2
    for batch in steps:
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        assert counts == {"alfworld": 5, "search": 5}


def test_topk_teacher_kl_matches_opd_loss():
    """Off-policy distillation reuses OPD's top-k KL: ~0 when student==teacher,
    strictly positive when the student diverges."""
    bs, resp_len, k = 2, 3, 4
    teacher = torch.log_softmax(torch.randn(bs, resp_len, k), dim=-1)
    # Normalize so the (top-k + tail) buckets are a valid distribution sub-mass.
    teacher = teacher - 1.0  # leave tail mass; values stay valid log-probs

    matched = topk_kl_per_token(student_topk_logprob=teacher.clone(), teacher_topk_logprob=teacher)
    assert torch.allclose(matched, torch.zeros_like(matched), atol=1e-5)

    shifted_student = teacher + torch.tensor([0.5, -0.3, 0.1, -0.2])
    diverged = topk_kl_per_token(student_topk_logprob=shifted_student, teacher_topk_logprob=teacher)
    assert (diverged.abs() > 1e-4).any()
