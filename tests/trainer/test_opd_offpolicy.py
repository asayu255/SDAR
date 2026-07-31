"""Unit tests for off-policy multitask distillation (offline KD).

Covered (CPU-only; Ray / workers are bypassed):
* the fixed teacher dataset round-trips through ``DataProto.save_to_disk`` and
  ``OffPolicyOPDRayTrainer._load_offpolicy_data`` into one DataProto per task with
  correct per-trajectory, task-local indexing;
* ``OffPolicyOPDRayTrainer._offpolicy_batch_iter`` yields the right number of
  task-balanced steps, each drawing ``per_task_traj_per_step`` whole trajectories
  per task from the matching task;
* ``find_padding_duplicates`` recovers exactly the rows an earlier Stage 1
  appended as adjust_batch padding, and ``_load_offpolicy_file`` drops them;
* the SFT arm, which reads this same pool, additionally drops the teacher top-k
  columns its NLL never reads, without changing any row the loss does read;
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
    from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer
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


def _index_by_task(trainer, pool_by_task):
    """The per-task grouping _load_offpolicy_data builds, for hand-made pools.

    Accepts either one DataProto per task or a list of shards. Indices are
    (shard, rows-in-shard), so each task's rows are addressed inside the file
    they were loaded from -- there is no concatenated pool to index into.
    """
    trainer._task_shards = {
        task: list(v) if isinstance(v, list) else [v] for task, v in pool_by_task.items()
    }
    trainer._task_to_traj_rows = {}
    trainer._task_to_trajs = {}
    for task, shards in trainer._task_shards.items():
        traj_to_rows = {}
        for shard_idx, shard in enumerate(shards):
            traj_uids = shard.non_tensor_batch["traj_uid"]
            for ridx in range(len(shard)):
                traj_to_rows.setdefault(traj_uids[ridx], (shard_idx, []))[1].append(int(ridx))
        trainer._task_to_traj_rows[task] = {
            u: (s, np.array(r, dtype=np.int64)) for u, (s, r) in traj_to_rows.items()
        }
        trainer._task_to_trajs[task] = np.array(list(traj_to_rows.keys()), dtype=object)


def test_load_offpolicy_data_indexes_per_task(tmp_path):
    _make_task_proto("alfworld", 5).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_task_proto("search", 3).save_to_disk(str(tmp_path / "search.pt"))
    _make_task_proto("webshop", 4).save_to_disk(str(tmp_path / "webshop.pt"))

    trainer = _bare_trainer()
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()

    # One DataProto per file, never concatenated.
    assert set(trainer._task_shards) == {"alfworld", "search", "webshop"}
    assert {t: len(s) for t, s in trainer._task_shards.items()} == {
        "alfworld": 1, "search": 1, "webshop": 1
    }
    sizes = {t: len(trajs) for t, trajs in trainer._task_to_trajs.items()}
    assert sizes == {"alfworld": 5, "search": 3, "webshop": 4}
    # Indices address rows inside the shard they name.
    for task, traj_rows in trainer._task_to_traj_rows.items():
        for shard_idx, rows in traj_rows.values():
            task_names = trainer._task_shards[task][shard_idx].non_tensor_batch["task_name"]
            assert all(task_names[i] == task for i in rows.tolist())


def test_offpolicy_batch_iter_is_task_balanced(tmp_path):
    trainer = _bare_trainer()
    _index_by_task(trainer, {
        "alfworld": _make_task_proto("alfworld", 10),
        "search": _make_task_proto("search", 10),
        "webshop": _make_task_proto("webshop", 10),
    })
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
    _index_by_task(trainer, {
        "alfworld": _make_task_proto("alfworld", 3),
        "search": _make_task_proto("search", 3),
    })
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

    assert sum(len(s) for s in trainer._task_shards["webshop"]) == 12  # 4 trajs * 3 turns, padding gone
    assert len(trainer._task_to_trajs["webshop"]) == 4
    assert all(len(rows) == 3 for _, rows in trainer._task_to_traj_rows["webshop"].values())


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
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()

    assert trainer._task_to_trajs["webshop"].tolist() == [f"webshop-{j}" for j in range(12)]
    assert [s for s, _ in trainer._task_to_traj_rows["webshop"].values()] == list(range(12))


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


def test_sft_arm_also_drops_the_teacher_topk_columns(tmp_path):
    """The SFT arm reads the KD arm's pool but not the KD arm's target.

    Same file, two loaders: KD keeps the teacher's top-k distribution because its
    loss is a KL against it; SFT's loss is an NLL on the sampled tokens, so those
    columns are dead weight (the largest thing in a row) and go at load. The rows
    themselves, and everything the NLL reads, must be identical either way --
    that is what makes the two arms differ only in the loss.
    """
    proto = _make_task_proto("search", 3, turns_per_traj=2)
    n = len(proto)
    proto.batch["input_ids"] = torch.arange(n * 8, dtype=torch.long).reshape(n, 8)
    proto.batch["prompts"] = torch.zeros((n, 8), dtype=torch.long)
    path = str(tmp_path / "search.pt")
    proto.save_to_disk(path)

    kd = OffPolicyOPDRayTrainer._load_offpolicy_file(path)
    sft = MultiTaskSFTTrainer._load_offpolicy_file(path)

    assert "teacher_topk_logprobs" in kd.batch and "teacher_topk_ids" in kd.batch
    assert "teacher_topk_logprobs" not in sft.batch and "teacher_topk_ids" not in sft.batch
    # The parent's drops still apply to the subclass; it extends, not replaces.
    assert "prompts" not in sft.batch
    # Identical data underneath.
    assert len(sft) == len(kd) == n
    assert torch.equal(sft.batch["input_ids"], kd.batch["input_ids"])
    assert torch.equal(sft.batch["responses"], kd.batch["responses"])
    assert sft.non_tensor_batch["traj_uid"].tolist() == kd.non_tensor_batch["traj_uid"].tolist()


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
