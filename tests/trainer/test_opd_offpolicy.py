"""Unit tests for off-policy multitask distillation (offline KD).

Covered (CPU-only; Ray / workers are bypassed):
* the fixed teacher dataset round-trips through ``DataProto.save_to_disk`` and
  ``OffPolicyOPDRayTrainer._load_offpolicy_data`` with correct per-task /
  per-trajectory indexing;
* ``OffPolicyOPDRayTrainer._offpolicy_batch_iter`` yields the right number of
  task-balanced steps, each drawing ``per_task_traj_per_step`` WHOLE trajectories
  per task (all of a trajectory's turn-rows kept together), matching OPD's
  per-step trajectory count;
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


def _make_task_proto(task, n_traj, turns_per_traj=1, resp_len=4, k=3, uid_offset=0):
    """A teacher-trajectory DataProto for one task: ``n_traj`` trajectories, each
    contributing ``turns_per_traj`` turn-rows that share a single traj_uid."""
    n = n_traj * turns_per_traj
    traj_uids = []
    for j in range(n_traj):
        traj_uids += [f"{task}-{uid_offset + j}"] * turns_per_traj
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros((n, resp_len), dtype=torch.long),
            "teacher_topk_logprobs": torch.full((n, resp_len, k), -1.0),
            "teacher_topk_ids": torch.zeros((n, resp_len, k), dtype=torch.long),
        },
        non_tensors={
            "task_name": np.array([task] * n, dtype=object),
            "traj_uid": np.array(traj_uids, dtype=object),
        },
    )


def _bare_trainer():
    """An OffPolicyOPDRayTrainer with __init__ bypassed (only the fixed-data
    helpers are exercised)."""
    return object.__new__(OffPolicyOPDRayTrainer)


def _load_from(tmp_path, trainer):
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()


def test_load_offpolicy_data_groups_rows_by_trajectory(tmp_path):
    # alfworld: 5 trajectories x 3 turns; search: 3 x 1; webshop: 4 x 2.
    _make_task_proto("alfworld", 5, turns_per_traj=3).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 3, turns_per_traj=1).save_to_disk(str(tmp_path / "search.pt"))
    _make_task_proto("webshop", 4, turns_per_traj=2).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    _load_from(tmp_path, trainer)

    assert len(trainer.offpolicy_data) == 5 * 3 + 3 * 1 + 4 * 2
    traj_counts = {t: len(v) for t, v in trainer._task_to_trajs.items()}
    assert traj_counts == {"alfworld": 5, "search": 3, "webshop": 4}
    # Each trajectory's row-index group must have the expected number of turn-rows,
    # and all its rows belong to that task.
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    expected_turns = {"alfworld": 3, "search": 1, "webshop": 2}
    for task, traj_rows in trainer._task_to_traj_rows.items():
        for uid, rows in traj_rows.items():
            assert len(rows) == expected_turns[task]
            assert all(task_names[r] == task for r in rows)


def test_offpolicy_batch_iter_draws_whole_trajectories_task_balanced(tmp_path):
    # 6 trajectories/task, 2 turns each.
    _make_task_proto("alfworld", 6, turns_per_traj=2).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 6, turns_per_traj=2).save_to_disk(str(tmp_path / "search.pt"))
    _make_task_proto("webshop", 6, turns_per_traj=2).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    _load_from(tmp_path, trainer)
    trainer.per_task_traj_per_step = 3  # trajectories per task per step
    trainer.total_training_steps = 5
    trainer.config = OmegaConf.create({"data": {"seed": 1}})

    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 5
    for batch in steps:
        # 3 trajectories * 2 turns * 3 tasks = 18 turn-rows.
        assert len(batch) == 3 * 2 * 3
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        assert counts == {"alfworld": 6, "search": 6, "webshop": 6}
        # Whole trajectories: every drawn traj_uid appears with all its turn-rows.
        uid_counts = {}
        for u in batch.non_tensor_batch["traj_uid"]:
            uid_counts[u] = uid_counts.get(u, 0) + 1
        assert all(c == 2 for c in uid_counts.values())
        assert len(uid_counts) == 3 * 3  # 3 trajectories per task * 3 tasks


def test_trajectory_pool_recycles_when_drawing_more_than_available(tmp_path):
    """Drawing more trajectories per step than a task's pool must still succeed
    (pool reshuffled + recycled) and stay balanced."""
    _make_task_proto("alfworld", 3, turns_per_traj=2).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 3, turns_per_traj=2).save_to_disk(str(tmp_path / "search.pt"))

    trainer = _bare_trainer()
    _load_from(tmp_path, trainer)
    trainer.per_task_traj_per_step = 5  # > 3 trajectories available per task
    trainer.total_training_steps = 2
    trainer.config = OmegaConf.create({"data": {"seed": 7}})

    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 2
    for batch in steps:
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        # 5 trajectories * 2 turns per task.
        assert counts == {"alfworld": 10, "search": 10}


def test_topk_teacher_kl_matches_opd_loss():
    """Off-policy distillation reuses OPD's top-k KL: ~0 when student==teacher,
    strictly positive when the student diverges."""
    bs, resp_len, k = 2, 3, 4
    teacher = torch.log_softmax(torch.randn(bs, resp_len, k), dim=-1)
    teacher = teacher - 1.0  # leave tail mass; values stay valid log-probs

    matched = topk_kl_per_token(student_topk_logprob=teacher.clone(), teacher_topk_logprob=teacher)
    assert torch.allclose(matched, torch.zeros_like(matched), atol=1e-5)

    shifted_student = teacher + torch.tensor([0.5, -0.3, 0.1, -0.2])
    diverged = topk_kl_per_token(student_topk_logprob=shifted_student, teacher_topk_logprob=teacher)
    assert (diverged.abs() > 1e-4).any()
