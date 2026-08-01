"""The cached pool must be indistinguishable from the pool it was built from.

``scripts/cache_teacher_pool.py`` moves the filtering the trainer does at load
(padding rows, dead columns) to a one-off offline pass, so runs stop paying for
it. That is only worth doing if it is invisible: same rows, same order, same
trajectory sampling population, and therefore the same batch at every step.

This branch carries only the KD arm, so every case here runs ``--arm kd``; the
SFT view is exercised on the SFT branch, which owns the trainer that defines it.

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import contextlib
import io
import json
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
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "cache_teacher_pool.py")

# What a KD cache holds: everything the top-k KL reads, and nothing else.
_KD_COLUMNS = [
    "attention_mask", "input_ids", "position_ids", "responses",
    "teacher_topk_ids", "teacher_topk_logprobs",
]


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


def _iterate(data_dir, steps=5, per_task=3, seed=5):
    trainer = object.__new__(OffPolicyOPDRayTrainer)
    trainer.teacher_data_dir = str(data_dir)
    with contextlib.redirect_stdout(io.StringIO()):  # the loader is chatty
        trainer._load_offpolicy_data()
    trainer.per_task_traj_per_step = per_task
    trainer.total_training_steps = steps
    trainer.config = OmegaConf.create({"data": {"seed": seed}})
    return trainer, list(trainer._offpolicy_batch_iter())


def _build_cache(src, dst, arm="kd"):
    result = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", arm],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_cached_pool_yields_identical_batches(tmp_path):
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache_kd"
    _build_cache(src, dst)

    original, from_source = _iterate(src)
    cached, from_cache = _iterate(dst)

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
    _build_cache(src, dst)

    for name in sorted(os.listdir(dst)):
        if not name.endswith(".pt"):
            continue
        raw = DataProto.load_from_disk(os.path.join(dst, name))
        assert sorted(raw.batch.keys()) == _KD_COLUMNS, name
        # Loading it again is a no-op: no padding found, no listed column present.
        with contextlib.redirect_stdout(io.StringIO()):
            reloaded = OffPolicyOPDRayTrainer._load_offpolicy_file(os.path.join(dst, name))
        assert len(reloaded) == len(raw), name


def test_kd_cache_keeps_the_teacher_topk_and_records_the_arm(tmp_path):
    """The arms are not interchangeable, and the manifest has to say which is which."""
    src = _make_pool(tmp_path / "pool")
    _build_cache(src, tmp_path / "kd")

    kd = json.load(open(tmp_path / "kd" / "_cache_manifest.json"))
    assert kd["arm"] == "kd"
    assert kd["columns"] == _KD_COLUMNS
    # The KD loss is a KL against the teacher's distribution, so those columns stay.
    assert "teacher_topk_logprobs" in kd["columns"]
    assert set(kd["dropped_columns"]) == {"prompts", "response_mask"}


def test_refuses_to_write_over_the_source(tmp_path):
    src = _make_pool(tmp_path / "pool")
    result = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(src), "--arm", "kd"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must differ" in result.stdout + result.stderr


def test_skip_existing_resumes(tmp_path):
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst)
    # Drop one output, as an interrupted pass would have.
    victim = os.path.join(dst, "search_0001.pt")
    os.remove(victim)

    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "kd", "--skip-existing"],
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
    _build_cache(src, dst)
    before = DataProto.load_from_disk(os.path.join(dst, "webshop_0000.pt"))

    # Regenerate webshop with different trajectories, as a fresh Stage 1 would.
    time.sleep(0.01)  # mtime granularity
    for s in range(3):
        _shard("webshop", 4, 2, first_uid=100 + s * 4).save_to_disk(
            os.path.join(src, f"webshop_{s:04d}.pt")
        )

    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "kd", "--skip-existing"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "source is newer" in out.stdout
    assert "wrote 3 file(s) (6 skipped)" in out.stdout

    after = DataProto.load_from_disk(os.path.join(dst, "webshop_0000.pt"))
    assert after.non_tensor_batch["traj_uid"][0] == "webshop-100"
    assert before.non_tensor_batch["traj_uid"][0] == "webshop-0"
    # The untouched tasks are still whole, so the cache describes the whole pool.
    trainer, _ = _iterate(dst)
    assert {t: len(v) for t, v in trainer._task_to_trajs.items()} == {
        "alfworld": 12, "search": 12, "webshop": 12
    }


def test_only_rebuilds_one_task_and_leaves_the_rest(tmp_path):
    """--only webshop must touch webshop and nothing else, manifest included."""
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst)
    untouched = {
        name: os.path.getmtime(os.path.join(dst, name))
        for name in os.listdir(dst)
        if name.endswith(".pt") and not name.startswith("webshop")
    }

    for s in range(3):  # a fresh Stage 1 for webshop only
        _shard("webshop", 4, 2, first_uid=100 + s * 4).save_to_disk(
            os.path.join(src, f"webshop_{s:04d}.pt")
        )
    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "kd", "--only", "webshop"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "wrote 3 file(s)" in out.stdout

    for name, mtime in untouched.items():
        assert os.path.getmtime(os.path.join(dst, name)) == mtime, name
    assert DataProto.load_from_disk(
        os.path.join(dst, "webshop_0000.pt")
    ).non_tensor_batch["traj_uid"][0] == "webshop-100"

    # The manifest still describes all nine files, not just the three rebuilt.
    manifest = json.load(open(dst / "_cache_manifest.json"))
    assert manifest["files"] == 9
    assert manifest["rows_complete"] is True
    assert len(manifest["rows_by_file"]) == 9
    assert manifest["columns"] == _KD_COLUMNS

    trainer, _ = _iterate(dst)
    assert {t: len(v) for t, v in trainer._task_to_trajs.items()} == {
        "alfworld": 12, "search": 12, "webshop": 12
    }


def test_only_deletes_shards_the_source_no_longer_has(tmp_path):
    """Regenerating into fewer shards must not leave the extra ones readable.

    The trainer globs the cache directory, so a leftover shard is loaded beside the
    new ones -- a silent mix of two generations that no assert catches, because the
    stale trajectories are simply different, not duplicated.
    """
    src = _make_pool(tmp_path / "pool")
    dst = tmp_path / "cache"
    _build_cache(src, dst)

    os.remove(os.path.join(src, "webshop_0002.pt"))  # regenerated into 2 shards
    for s in range(2):
        _shard("webshop", 6, 2, first_uid=100 + s * 6).save_to_disk(
            os.path.join(src, f"webshop_{s:04d}.pt")
        )
    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(dst), "--arm", "kd", "--only", "webshop"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "removed stale output webshop_0002.pt" in out.stdout
    assert not os.path.exists(os.path.join(dst, "webshop_0002.pt"))

    trainer, _ = _iterate(dst)
    assert {t: len(v) for t, v in trainer._task_to_trajs.items()} == {
        "alfworld": 12, "search": 12, "webshop": 12
    }


def test_only_rejects_a_task_that_is_not_there(tmp_path):
    """A typo must not silently write nothing and report success."""
    src = _make_pool(tmp_path / "pool")
    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(tmp_path / "cache"),
         "--arm", "kd", "--only", "webshopp"],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    combined = out.stdout + out.stderr
    assert "webshopp" in combined and "alfworld" in combined


def test_sft_arm_is_not_available_on_this_branch(tmp_path):
    """--arm sft needs the SFT trainer, which lives on the SFT branch. It has to
    say so rather than raising a bare ImportError."""
    src = _make_pool(tmp_path / "pool")
    out = subprocess.run(
        [sys.executable, _SCRIPT, str(src), str(tmp_path / "cache"), "--arm", "sft"],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "--arm kd" in out.stdout + out.stderr
