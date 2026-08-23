# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint shards are written off the training thread.

Measured on this arm (3xA6000, Qwen3-1.7B, wandb x7g9r7bx): a save took 198 s, of
which the first ~20 s built the sharded state dict and copied it to host memory
and the remaining ~178 s had the GPUs at 0.0% SM and their 28 W idle floor --
torch.save pickling to disk with the training loop waiting on it, twelve times
over a run.

The risk the mechanism introduces is not a slow checkpoint, it is a checkpoint
that looks complete and is not, so these tests are mostly about the ordering that
prevents that: nothing may treat a checkpoint as saved until the write has landed,
and a write that fails must reach the main thread.
"""

import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf

    from verl.utils.checkpoint import fsdp_checkpoint_manager as ckpt_mod
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _use_the_writer_thread(monkeypatch):
    """These tests cover the fallback writer, which is a thread.

    The default writer is a forked child (it does not hold the GIL, so it does
    not stall the next step's kernel launches). Its own contracts differ enough
    to need their own file -- a failed child is rewritten by the parent rather
    than raised, and a threading.Event cannot observe another process -- so they
    live in tests/utils/test_fork_checkpoint_writer.py. What is checked here is
    that CKPT_FORK_WRITER=0 still behaves exactly as it always did, which is the
    A/B those measurements depend on.
    """
    monkeypatch.setattr(ckpt_mod, "_FORK_WRITER", False)


class _Manager:
    """A checkpoint manager with the real async machinery and no FSDP.

    Only the writer half is exercised: building the state dict is a collective and
    a device copy, which is precisely the part that does NOT move to the thread.
    """

    def __init__(self):
        self.rank = 0
        self.async_save = True
        self._pending_save = None
        self._pending_error = None
        self._fork_broken = False

    start = FSDPCheckpointManager._start_async_write
    wait = FSDPCheckpointManager.wait_for_pending_save


@pytest.fixture
def blocking_write(monkeypatch):
    """Hold the write open so a test can observe the caller running past it."""
    started, release = threading.Event(), threading.Event()

    def _blocked(writes):
        started.set()
        assert release.wait(10), "the test never released the write"

    monkeypatch.setattr(ckpt_mod, "_write_shards", _blocked)
    return started, release


def test_the_write_runs_off_the_calling_thread(blocking_write):
    started, release = blocking_write
    manager = _Manager()

    manager.start([(None, "shard.pt")], "/ckpt/global_step_25")

    assert started.wait(10), "the write never began"
    # The caller got here while the write is still going: that is the whole point.
    assert manager._pending_save is not None

    release.set()
    assert manager.wait() == "/ckpt/global_step_25"
    assert manager._pending_save is None


def test_the_bytes_are_the_same_ones(tmp_path):
    """Async is a change of when, not of what. A shard written on the thread has
    to load back identical to one written inline."""
    state = {"w": torch.arange(64, dtype=torch.float32).reshape(8, 8), "step": 25}
    inline, background = tmp_path / "a.pt", tmp_path / "b.pt"

    torch.save(state, inline)
    manager = _Manager()
    manager.start([(state, str(background))], str(tmp_path))
    manager.wait()

    got = torch.load(background, weights_only=False)
    want = torch.load(inline, weights_only=False)
    assert got["step"] == want["step"]
    assert torch.equal(got["w"], want["w"])


def test_a_failed_write_is_raised_on_the_main_thread(tmp_path):
    """The failure mode worth fearing: a run that ends reporting success with no
    usable checkpoint, because the only thread that knew was the writer's."""
    manager = _Manager()
    manager.start([({"w": torch.zeros(2)}, str(tmp_path / "nope" / "x.pt"))], str(tmp_path))

    with pytest.raises(RuntimeError, match="background checkpoint write failed"):
        manager.wait()


def test_the_error_is_only_raised_once(tmp_path):
    manager = _Manager()
    manager.start([({"w": torch.zeros(2)}, str(tmp_path / "nope" / "x.pt"))], str(tmp_path))
    with pytest.raises(RuntimeError):
        manager.wait()

    assert manager.wait() is None  # cleared, so the next save is not blamed for it


def test_waiting_with_nothing_in_flight_is_fine():
    assert _Manager().wait() is None


def test_two_writes_cannot_be_in_flight_at_once(blocking_write):
    """Each in-flight save holds its staged copy of the shards in host memory
    (~5-6 GiB a rank here), and the rotation that deletes old checkpoints runs
    right after this guard."""
    started, release = blocking_write
    manager = _Manager()
    manager.start([(None, "first")], "/ckpt/a")
    assert started.wait(10)

    with pytest.raises(AssertionError, match="already in flight"):
        manager.start([(None, "second")], "/ckpt/b")

    release.set()
    manager.wait()


class _Trainer:
    """The parts of RayPPOTrainer the tracker-publishing path touches."""

    def __init__(self, tmp_path, async_save, use_critic=False):
        self.config = OmegaConf.create({
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": async_save}}},
            "critic": {"checkpoint": {"async_save": False}},
        })
        self.use_critic = use_critic
        self.global_steps = 0
        self.waits = 0
        self._pending_checkpoint_step = None

        outer = self

        class _WG:
            def wait_for_checkpoint(self):
                outer.waits += 1

        self.actor_rollout_wg = _WG()
        self.critic_wg = _WG()

    _async_checkpoint_save = RayPPOTrainer._async_checkpoint_save
    _flush_pending_checkpoint = RayPPOTrainer._flush_pending_checkpoint

    def tracker(self):
        path = Path(self.config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
        return path.read_text() if path.exists() else None


def test_the_tracker_waits_for_the_write(tmp_path):
    """A resume reads this file. Publishing step N before N's shards are on disk
    is what would let a crash leave it naming a half-written directory."""
    trainer = _Trainer(tmp_path, async_save=True)
    trainer._pending_checkpoint_step = 25

    assert trainer.tracker() is None  # save_checkpoint returned; nothing published

    trainer._flush_pending_checkpoint()

    assert trainer.waits == 1
    assert trainer.tracker() == "25"


def test_the_tracker_names_the_checkpoint_that_was_saved_not_the_current_step(tmp_path):
    """The flush happens a step later, so it must publish the pending step rather
    than wherever training has got to."""
    trainer = _Trainer(tmp_path, async_save=True)
    trainer._pending_checkpoint_step = 25
    trainer.global_steps = 26

    trainer._flush_pending_checkpoint()

    assert trainer.tracker() == "25"


def test_flushing_twice_publishes_once(tmp_path):
    trainer = _Trainer(tmp_path, async_save=True)
    trainer._pending_checkpoint_step = 25
    trainer._flush_pending_checkpoint()
    trainer._flush_pending_checkpoint()

    assert trainer.waits == 1  # the second call has nothing pending to wait for


def test_the_synchronous_path_does_not_call_the_workers(tmp_path):
    """With async_save off the behaviour has to be what it was, down to the round
    trips -- a worker group without wait_for_checkpoint (megatron) still has to
    work."""
    trainer = _Trainer(tmp_path, async_save=False)
    trainer._pending_checkpoint_step = 25

    trainer._flush_pending_checkpoint()

    assert trainer.waits == 0
    assert trainer.tracker() == "25"


def test_the_critic_is_waited_for_when_it_is_the_one_saving_async(tmp_path):
    trainer = _Trainer(tmp_path, async_save=False, use_critic=True)
    trainer.config.critic.checkpoint.async_save = True
    trainer._pending_checkpoint_step = 25

    trainer._flush_pending_checkpoint()

    assert trainer.waits == 2  # actor and critic
