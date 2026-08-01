"""Unit tests for off-policy multitask distillation (offline KD).

Covered (CPU-only; Ray / workers are bypassed):
* the fixed teacher dataset round-trips through ``DataProto.save_to_disk`` and
  ``OffPolicyOPDRayTrainer._load_offpolicy_data`` into one DataProto per file
  with correct per-trajectory, task-local indexing -- and a pool split into
  shards yields the same batches, step for step, as the same rows in one file;
* ``OffPolicyOPDRayTrainer._offpolicy_batch_iter`` yields the right number of
  task-balanced steps, each drawing ``per_task_traj_per_step`` WHOLE trajectories
  per task (all of a trajectory's turn-rows kept together), matching OPD's
  per-step trajectory count;
* ``find_padding_duplicates`` recovers exactly the rows an earlier Stage 1
  appended as adjust_batch padding, and ``_load_offpolicy_file`` drops them
  along with the columns no Stage-2 loss reads -- and does nothing at all to a
  pool ``scripts/cache_teacher_pool.py`` has already filtered;
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

    total = sum(len(s) for shards in trainer._task_shards.values() for s in shards)
    assert total == 5 * 3 + 3 * 1 + 4 * 2
    traj_counts = {t: len(v) for t, v in trainer._task_to_trajs.items()}
    assert traj_counts == {"alfworld": 5, "search": 3, "webshop": 4}
    # Each trajectory's row-index group must have the expected number of turn-rows,
    # and address rows inside a shard that holds that task.
    expected_turns = {"alfworld": 3, "search": 1, "webshop": 2}
    for task, traj_rows in trainer._task_to_traj_rows.items():
        for uid, (shard_idx, rows) in traj_rows.items():
            assert len(rows) == expected_turns[task]
            shard = trainer._task_shards[task][shard_idx]
            task_names = shard.non_tensor_batch["task_name"]
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
    _load_from(tmp_path, trainer)

    assert sum(len(s) for s in trainer._task_shards["webshop"]) == 12  # 4 trajs * 3 turns, padding gone
    assert len(trainer._task_to_trajs["webshop"]) == 4
    assert all(len(rows) == 3 for _, rows in trainer._task_to_traj_rows["webshop"].values())


def test_load_offpolicy_file_drops_dead_columns(tmp_path):
    """prompts / response_mask are a quarter of every row and nothing reads them."""
    proto = _make_task_proto("search", 3, turns_per_traj=2)
    n, resp_len = len(proto), proto.batch["responses"].shape[-1]
    proto.batch["prompts"] = torch.zeros((n, 8), dtype=torch.long)
    proto.batch["response_mask"] = torch.ones((n, resp_len), dtype=torch.long)
    path = str(tmp_path / "search.pt")
    proto.save_to_disk(path)

    loaded = OffPolicyOPDRayTrainer._load_offpolicy_file(path)

    assert "prompts" not in loaded.batch
    assert "response_mask" not in loaded.batch
    # Everything the stage does read survives, rows and all.
    assert set(loaded.batch.keys()) == {"responses", "teacher_topk_logprobs", "teacher_topk_ids"}
    assert len(loaded) == n
    assert loaded.non_tensor_batch["traj_uid"].tolist() == proto.non_tensor_batch["traj_uid"].tolist()


def test_load_offpolicy_file_drops_columns_and_padding_together(tmp_path):
    """The single pass has to survive doing both at once."""
    proto = _make_task_proto("webshop", 4, turns_per_traj=2)
    proto.batch["prompts"] = torch.zeros((len(proto), 8), dtype=torch.long)
    padded = DataProto.concat([proto, proto.select_idxs([0, 5])])
    path = str(tmp_path / "webshop.pt")
    padded.save_to_disk(path)

    loaded = OffPolicyOPDRayTrainer._load_offpolicy_file(path)

    assert "prompts" not in loaded.batch
    assert len(loaded) == 8
    assert loaded.non_tensor_batch["traj_uid"].tolist() == [f"webshop-{j}" for j in range(4) for _ in range(2)]


def test_already_cached_pool_passes_through_unchanged(tmp_path):
    """What scripts/cache_teacher_pool.py writes is what the loader would have
    built, so loading the cache is a no-op: nothing left to drop, same rows."""
    proto = _make_task_proto("alfworld", 3, turns_per_traj=2)
    proto.batch["prompts"] = torch.zeros((len(proto), 8), dtype=torch.long)
    raw = str(tmp_path / "alfworld.pt")
    DataProto.concat([proto, proto.select_idxs([0, 2])]).save_to_disk(raw)

    cached_dir = tmp_path / "cache"
    cached_dir.mkdir()
    once = OffPolicyOPDRayTrainer._load_offpolicy_file(raw)
    once.save_to_disk(str(cached_dir / "alfworld.pt"))
    twice = OffPolicyOPDRayTrainer._load_offpolicy_file(str(cached_dir / "alfworld.pt"))

    assert set(twice.batch.keys()) == set(once.batch.keys())
    assert len(twice) == len(once) == 6
    assert torch.equal(twice.batch["responses"], once.batch["responses"])
    assert twice.non_tensor_batch["traj_uid"].tolist() == once.non_tensor_batch["traj_uid"].tolist()


def test_duplicate_trajectory_across_files_is_rejected(tmp_path):
    """traj_uid is a uuid4, so the same one in two files means the directory
    holds overlapping shards -- two runs' output, or a shard written twice.
    Merging them would build a trajectory whose turns came from different
    rollouts, so it has to fail instead."""
    proto = _make_task_proto("search", 4, turns_per_traj=2)
    proto.save_to_disk(str(tmp_path / "search_0000.pt"))
    proto.save_to_disk(str(tmp_path / "search_0001.pt"))  # same shard written twice

    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(tmp_path)
    with pytest.raises(AssertionError, match="more than one file"):
        trainer._load_offpolicy_data()


def _tagged(proto):
    """Give every row a unique, checkable value so two pools can be compared."""
    n, resp_len = len(proto), proto.batch["responses"].shape[-1]
    proto.batch["responses"] = torch.arange(n * resp_len, dtype=torch.long).reshape(n, resp_len)
    return proto


def _run_iter(data_dir, steps=5, per_task=3, seed=11):
    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(data_dir)
    trainer._load_offpolicy_data()
    trainer.per_task_traj_per_step = per_task
    trainer.total_training_steps = steps
    trainer.config = OmegaConf.create({"data": {"seed": seed}})
    return trainer, list(trainer._offpolicy_batch_iter())


def test_sharded_pool_is_indistinguishable_from_one_file(tmp_path):
    """Stage 1 writes <task>_0000.pt shards when gen.shard_every_steps is set;
    at 36k trajectories that is what makes the pool loadable at all. Which
    shard a row landed in must not reach the training batches: same rows, same
    order, step for step."""
    whole, sharded = tmp_path / "whole", tmp_path / "sharded"
    whole.mkdir()
    sharded.mkdir()
    for task, n_traj in (("alfworld", 8), ("search", 8)):
        proto = _tagged(_make_task_proto(task, n_traj, turns_per_traj=2))
        proto.save_to_disk(str(whole / f"{task}.pt"))
        # Same rows, same order, cut into shards on trajectory boundaries the way
        # a flush every N generation steps cuts them.
        for i, start in enumerate(range(0, len(proto), 4)):
            part = proto.select_idxs(list(range(start, min(start + 4, len(proto)))))
            part.save_to_disk(str(sharded / f"{task}_{i:04d}.pt"))

    one, steps_one = _run_iter(whole)
    many, steps_many = _run_iter(sharded)

    assert {t: len(s) for t, s in one._task_shards.items()} == {"alfworld": 1, "search": 1}
    assert {t: len(s) for t, s in many._task_shards.items()} == {"alfworld": 4, "search": 4}
    # The sampling population is ordered the same, so the draws match.
    for task in one._task_to_trajs:
        assert one._task_to_trajs[task].tolist() == many._task_to_trajs[task].tolist()

    assert len(steps_one) == len(steps_many) == 5
    for a, b in zip(steps_one, steps_many):
        assert torch.equal(a.batch["responses"], b.batch["responses"])
        assert a.non_tensor_batch["traj_uid"].tolist() == b.non_tensor_batch["traj_uid"].tolist()
        assert a.non_tensor_batch["task_name"].tolist() == b.non_tensor_batch["task_name"].tolist()


def test_shards_of_one_task_are_read_in_write_order(tmp_path):
    """Shard files are numbered, and sorted() has to put them back in the order
    Stage 1 flushed them -- otherwise the trajectory population is permuted and
    a resumed/regenerated pool draws a different sequence."""
    proto = _tagged(_make_task_proto("webshop", 12, turns_per_traj=1))
    for i in range(12):
        proto.select_idxs([i]).save_to_disk(str(tmp_path / f"webshop_{i:04d}.pt"))

    trainer = _bare_trainer()
    _load_from(tmp_path, trainer)

    assert trainer._task_to_trajs["webshop"].tolist() == [f"webshop-{j}" for j in range(12)]
    assert [s for s, _ in trainer._task_to_traj_rows["webshop"].values()] == list(range(12))


def test_nothing_is_concatenated_across_files(tmp_path):
    """The point of the shard layout: a task's files stay the objects they were
    loaded as, so the load peak is 'resident + one shard' rather than twice the
    pool. A shard the loader kept whole must be the same length as its file."""
    for i in range(3):
        _make_task_proto("search", 4, turns_per_traj=2, uid_offset=i * 4).save_to_disk(
            str(tmp_path / f"search_{i:04d}.pt")
        )

    trainer = _bare_trainer()
    _load_from(tmp_path, trainer)

    shards = trainer._task_shards["search"]
    assert len(shards) == 3 and [len(s) for s in shards] == [8, 8, 8]
    # Rows are addressed within their own shard, never in a pool-wide index space.
    for _, (shard_idx, rows) in trainer._task_to_traj_rows["search"].items():
        assert rows.max() < len(shards[shard_idx])


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
