"""The cached pool must be indistinguishable from the pool it was built from.

``scripts/cache_teacher_pool.py`` moves the filtering the trainer does at load
(padding rows, dead columns) to a one-off offline pass, so runs stop paying for
it. That is only worth doing if it is invisible: same rows, same order, same
trajectory sampling population, and therefore the same batch at every step.

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import contextlib
import io
import os
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.trainer.ppo.opd_offpolicy_ray_trainer import OffPolicyOPDRayTrainer
    from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "cache_teacher_pool.py")


def _shard(task, n_traj, turns, first_uid, resp_len=4, k=3, pad=2):
    """One Stage-1 shard: whole trajectories, then adjust_batch's padding copies."""
    uids = []
    for j in range(first_uid, first_uid + n_traj):
        uids += [f"{task}-{j}"] * turns
    n = len(uids)
    tensors = {
        "responses": torch.arange(n * resp_len, dtype=torch.long).reshape(n, resp_len) + first_uid,
        "input_ids": torch.arange(n * 8, dtype=torch.long).reshape(n, 8) + first_uid,
        "attention_mask": torch.ones((n, 8), dtype=torch.long),
        "position_ids": torch.arange(8).repeat(n, 1),
        "prompts": torch.zeros((n, 4), dtype=torch.long),
        "response_mask": torch.ones((n, resp_len), dtype=torch.long),
        "teacher_topk_logprobs": torch.full((n, resp_len, k), -1.0),
        "teacher_topk_ids": torch.zeros((n, resp_len, k), dtype=torch.long),
    }
    proto = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "task_name": np.array([task] * n, dtype=object),
            "traj_uid": np.array(uids, dtype=object),
        },
    )
    return DataProto.concat([proto, proto.select_idxs(list(range(pad)))])


def _make_pool(root):
    os.makedirs(root, exist_ok=True)
    for task, turns in (("alfworld", 3), ("search", 1), ("webshop", 2)):
        for s in range(3):
            _shard(task, 4, turns, first_uid=s * 4).save_to_disk(
                os.path.join(root, f"{task}_{s:04d}.pt")
            )
    return root


def _iterate(trainer_cls, data_dir, steps=5, per_task=3, seed=5):
    trainer = object.__new__(trainer_cls)
    trainer.teacher_data_dir = str(data_dir)
    with contextlib.redirect_stdout(io.StringIO()):  # the loader is chatty
        trainer._load_offpolicy_data()
    trainer.per_task_traj_per_step = per_task
    trainer.total_training_steps = steps
    trainer.config = OmegaConf.create({"data": {"seed": seed}})
    return trainer, list(trainer._offpolicy_batch_iter())


def _build_cache(src, dst, arm):
    result = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", arm],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "arm, trainer_cls", [("sft", MultiTaskSFTTrainer), ("kd", OffPolicyOPDRayTrainer)]
)
def test_cached_pool_yields_identical_batches(tmp_path, arm, trainer_cls):
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / f"cache_{arm}"
    _build_cache(src, dst, arm)

    original, from_source = _iterate(trainer_cls, src)
    cached, from_cache = _iterate(trainer_cls, dst)

    # The sampling population is ordered by first appearance across sorted(glob),
    # so identical ordering is what makes every subsequent draw identical.
    assert list(original._task_to_trajs) == list(cached._task_to_trajs)
    for task, trajs in original._task_to_trajs.items():
        assert trajs.tolist() == cached._task_to_trajs[task].tolist()

    assert len(from_source) == len(from_cache) == 5
    for step, (a, b) in enumerate(zip(from_source, from_cache)):
        assert sorted(a.batch.keys()) == sorted(b.batch.keys()), step
        for key in a.batch.keys():
            assert torch.equal(a.batch[key], b.batch[key]), (step, key)
        for key in a.non_tensor_batch:
            assert a.non_tensor_batch[key].tolist() == b.non_tensor_batch[key].tolist(), (step, key)


def test_cache_is_already_filtered(tmp_path):
    """Nothing is left for the trainer to do at load -- that is the whole point."""
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst, "sft")

    for name in sorted(os.listdir(dst)):
        if not name.endswith(".pt"):
            continue
        raw = DataProto.load_from_disk(os.path.join(dst, name))
        assert sorted(raw.batch.keys()) == [
            "attention_mask", "input_ids", "position_ids", "responses"
        ], name
        # Loading it again is a no-op: no padding found, no listed column present.
        with contextlib.redirect_stdout(io.StringIO()):
            reloaded = MultiTaskSFTTrainer._load_offpolicy_file(os.path.join(dst, name))
        assert len(reloaded) == len(raw), name


def test_sft_cache_keeps_less_than_the_kd_cache(tmp_path):
    """The arms are not interchangeable, and the manifest has to say which is which."""
    import json

    src = _make_pool(tmp_path / "pool")
    _build_cache(src, tmp_path / "sft", "sft")
    _build_cache(src, tmp_path / "kd", "kd")

    sft = json.load(open(tmp_path / "sft" / "_cache_manifest.json"))
    kd = json.load(open(tmp_path / "kd" / "_cache_manifest.json"))
    assert sft["arm"] == "sft" and kd["arm"] == "kd"
    assert set(sft["columns"]) < set(kd["columns"])
    assert "teacher_topk_logprobs" in kd["columns"]
    assert "teacher_topk_logprobs" not in sft["columns"]
    # Same rows either way; only the columns differ.
    assert sft["rows"] == kd["rows"]


def test_refuses_to_write_over_the_source(tmp_path):
    src = _make_pool(tmp_path / "pool")
    result = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(src), "--arm", "sft"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must differ" in result.stdout + result.stderr


def test_skip_existing_resumes(tmp_path):
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst, "sft")
    # Drop one output, as an interrupted pass would have.
    victim = os.path.join(dst, "search_0001.pt")
    os.remove(victim)

    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "sft", "--skip-existing"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert os.path.exists(victim)
    assert "wrote 1 file(s) (8 skipped)" in out.stdout


def test_skip_existing_rebuilds_a_regenerated_task(tmp_path):
    """Regenerating one task must not leave its stale view in the cache.

    --skip-existing exists to resume an interrupted pass, not to preserve an
    answer the source has moved past. A cache that silently disagrees with the
    pool it claims to be a view of is worse than no cache.
    """
    import time

    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst, "sft")
    before = DataProto.load_from_disk(os.path.join(dst, "webshop_0000.pt"))

    # Regenerate webshop with different trajectories, as a fresh Stage 1 would.
    time.sleep(0.01)  # mtime granularity
    for s in range(3):
        _shard("webshop", 4, 2, first_uid=100 + s * 4).save_to_disk(
            os.path.join(src, f"webshop_{s:04d}.pt")
        )

    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "sft", "--skip-existing"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "source is newer" in out.stdout
    assert "wrote 3 file(s) (6 skipped)" in out.stdout

    after = DataProto.load_from_disk(os.path.join(dst, "webshop_0000.pt"))
    assert after.non_tensor_batch["traj_uid"][0] == "webshop-100"
    assert before.non_tensor_batch["traj_uid"][0] == "webshop-0"
    # The untouched tasks are still whole, so the cache describes the whole pool.
    trainer, _ = _iterate(MultiTaskSFTTrainer, dst)
    assert {t: len(v) for t, v in trainer._task_to_trajs.items()} == {
        "alfworld": 12, "search": 12, "webshop": 12
    }
