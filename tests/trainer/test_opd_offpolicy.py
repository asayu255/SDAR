"""Unit tests for off-policy multitask distillation (offline KD).

Covered (CPU-only; Ray / workers are bypassed):
* the fixed teacher dataset round-trips through ``DataProto.save_to_disk`` and
  ``OffPolicyOPDRayTrainer._load_offpolicy_data`` with correct per-task,
  per-trajectory indexing;
* ``OffPolicyOPDRayTrainer._offpolicy_batch_iter`` yields the right number of
  task-balanced steps, each drawing ``per_task_traj_per_step`` whole trajectories
  per task from the matching task;
* ``find_padding_duplicates`` recovers exactly the rows an earlier Stage 1
  appended as adjust_batch padding, and ``_load_offpolicy_file`` drops them;
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
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import (
        OffPolicyOPDRayTrainer,
        find_padding_duplicates,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _make_task_proto(task, n_traj, turns_per_traj=1, resp_len=4, k=3):
    """A minimal teacher-trajectory DataProto for one task.

    ``n_traj`` trajectories, each contributing ``turns_per_traj`` contiguous
    turn-rows that share a traj_uid -- the layout Stage 1 writes.
    """
    n = n_traj * turns_per_traj
    traj_uids = []
    for j in range(n_traj):
        traj_uids += [f"{task}-{j}"] * turns_per_traj
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


def _index_by_task(trainer, tasks):
    """The trajectory grouping _load_offpolicy_data builds, for a hand-made pool."""
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    traj_uids = trainer.offpolicy_data.non_tensor_batch["traj_uid"]
    trainer._task_to_traj_rows = {}
    trainer._task_to_trajs = {}
    for task in tasks:
        traj_to_rows = {}
        for ridx in np.where(task_names == task)[0]:
            traj_to_rows.setdefault(traj_uids[ridx], []).append(int(ridx))
        trainer._task_to_traj_rows[task] = {
            u: np.array(r, dtype=np.int64) for u, r in traj_to_rows.items()
        }
        trainer._task_to_trajs[task] = np.array(list(traj_to_rows.keys()), dtype=object)


def test_load_offpolicy_data_concats_and_indexes_per_task(tmp_path):
    _make_task_proto("alfworld", 5).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 3).save_to_disk(str(tmp_path / "search.pt"))
    _make_task_proto("webshop", 4).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()

    assert len(trainer.offpolicy_data) == 12
    sizes = {t: len(trajs) for t, trajs in trainer._task_to_trajs.items()}
    assert sizes == {"alfworld": 5, "search": 3, "webshop": 4}
    # Every row a task's trajectories point at must actually be that task's.
    task_names = trainer.offpolicy_data.non_tensor_batch["task_name"]
    for task, traj_rows in trainer._task_to_traj_rows.items():
        for rows in traj_rows.values():
            assert all(task_names[i] == task for i in rows.tolist())


def test_offpolicy_batch_iter_is_task_balanced(tmp_path):
    trainer = _bare_trainer()
    trainer.offpolicy_data = DataProto.concat(
        [_make_task_proto("alfworld", 10), _make_task_proto("search", 10), _make_task_proto("webshop", 10)]
    )
    _index_by_task(trainer, ("alfworld", "search", "webshop"))
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
    _index_by_task(trainer, ("alfworld", "search"))
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


def test_find_padding_duplicates_marks_only_repeat_runs():
    # One Stage-1 block: three trajectories laid out trajectory-major, then the
    # adjust_batch padding (copies of a row from B and one from A) appended.
    uids = np.array(["A", "A", "A", "B", "B", "C", "C"] + ["B", "A"], dtype=object)
    expected = np.array([False] * 7 + [True] * 2)
    assert (find_padding_duplicates(uids) == expected).all()

    # Consecutive copies of the *same* uid form one run and are all padding.
    uids = np.array(["A", "A", "B"] + ["A", "A"], dtype=object)
    assert (find_padding_duplicates(uids) == np.array([False, False, False, True, True])).all()

    # A following block's fresh trajectories are not padding, even though the
    # previous block ended with padding rows.
    uids = np.array(["A", "A", "B"] + ["A"] + ["C", "C", "D"], dtype=object)
    expected = np.array([False, False, False, True, False, False, False])
    assert (find_padding_duplicates(uids) == expected).all()

    # A pool written after Stage 1 stopped padding has nothing to drop.
    uids = np.array(["A", "A", "B", "C", "C"], dtype=object)
    assert not find_padding_duplicates(uids).any()
    assert len(find_padding_duplicates(np.array([], dtype=object))) == 0


def test_load_offpolicy_file_drops_padding_rows(tmp_path):
    """Padding rows are dropped, and the rows they were copied from survive."""
    proto = _make_task_proto("search", 3, turns_per_traj=2)
    # Emulate Stage-1 padding: append copies of rows 0 (search-0) and 3 (search-1).
    padded = DataProto.concat([proto, proto.select_idxs([0, 3])])
    path = str(tmp_path / "search.pt")
    padded.save_to_disk(path)

    loaded = OffPolicyOPDRayTrainer._load_offpolicy_file(path)

    assert len(padded) == 8 and len(loaded) == 6
    uids = loaded.non_tensor_batch["traj_uid"].tolist()
    # Every trajectory keeps exactly its real turns, in the original order.
    assert uids == ["search-0"] * 2 + ["search-1"] * 2 + ["search-2"] * 2


def test_load_offpolicy_data_filters_before_indexing(tmp_path):
    """The trajectory grouping is built from filtered rows, so no trajectory
    carries a duplicated turn."""
    proto = _make_task_proto("webshop", 4, turns_per_traj=3)
    DataProto.concat([proto, proto.select_idxs([1, 2, 5])]).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()

    assert len(trainer.offpolicy_data) == 12  # 4 trajectories * 3 turns, padding gone
    assert len(trainer._task_to_trajs["webshop"]) == 4
    assert all(len(rows) == 3 for rows in trainer._task_to_traj_rows["webshop"].values())


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
