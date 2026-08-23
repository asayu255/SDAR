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
"""The forked checkpoint writer: it must write, and it must never lose a shard.

The reason to fork at all is that a writer *thread* holds the GIL while pickling
and stalls the next step's kernel launches. The reason to be careful is that
forking a process carrying CUDA, NCCL-watchdog and vLLM threads can produce a
child that deadlocks on a lock no surviving thread will release.

So two properties carry the weight, and they pull in opposite directions. The
child really has to be a separate process (a version that quietly ran the write
inline would pass any test that only checked the bytes on disk). And every way
the child can fail -- non-zero exit, signal, never finishing -- has to end with
the shards on disk anyway, written by the parent, with the fork path disabled so
the failure is paid once rather than once per save.

The manager is built with ``object.__new__`` because ``__init__`` wants an FSDP
model, an optimizer and a live process group; the writer path touches none of
them.
"""

import os
import signal
import time

import pytest
import torch

from verl.utils.checkpoint import fsdp_checkpoint_manager as mod
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _manager(fork_broken=False):
    m = object.__new__(FSDPCheckpointManager)
    m.rank = 0
    m._pending_save = None
    m._pending_error = None
    m._fork_broken = fork_broken
    return m


def _writes(tmp_path, n=2):
    return [({"w": torch.arange(16, dtype=torch.float32) + i}, str(tmp_path / f"shard{i}.pt"))
            for i in range(n)]


def test_forked_write_lands_on_disk(tmp_path):
    m = _manager()
    writes = _writes(tmp_path)

    m._start_async_write(writes, str(tmp_path))
    assert m._pending_save[0] == "fork", "should have forked, not fallen back to the thread"
    path = m.wait_for_pending_save()

    assert path == str(tmp_path)
    for state, p in writes:
        assert torch.equal(torch.load(p, weights_only=False)["w"], state["w"])
    assert m._pending_save is None
    assert m._fork_broken is False


def test_the_writer_really_is_another_process(tmp_path):
    """A version that wrote inline would pass the bytes-on-disk test above.

    The child records its own pid; a different pid is the only direct evidence
    that the GIL the write holds is not this interpreter's.
    """
    m = _manager()
    target = str(tmp_path / "pid.pt")

    class _RecordsItsPid(dict):
        def __reduce__(self):
            return (dict, ({"pid": os.getpid()},))

    m._start_async_write([(_RecordsItsPid(), target)], str(tmp_path))
    m.wait_for_pending_save()

    assert torch.load(target, weights_only=False)["pid"] != os.getpid()


def test_child_failure_is_rewritten_by_the_parent(tmp_path, monkeypatch):
    """A child that dies must not cost a checkpoint: the parent writes it."""
    m = _manager()
    writes = _writes(tmp_path)
    real = mod._write_shards
    calls = []

    def _explode_in_the_child(w):
        calls.append(os.getpid())
        if os.getpid() != m_pid:
            raise RuntimeError("boom in the child")
        return real(w)

    m_pid = os.getpid()
    monkeypatch.setattr(mod, "_write_shards", _explode_in_the_child)

    m._start_async_write(writes, str(tmp_path))
    with pytest.warns(UserWarning, match="forked checkpoint writer failed"):
        m.wait_for_pending_save()

    for state, p in writes:
        assert torch.equal(torch.load(p, weights_only=False)["w"], state["w"])
    assert m._fork_broken is True, "one failure must disable the fork path"


def test_a_stuck_child_is_killed_and_the_shards_written(tmp_path, monkeypatch):
    """The deadlocked-child case, which is the whole reason for the timeout.

    A child that never returns is indistinguishable from one deadlocked on a
    lock it inherited locked, so it gets the same treatment: killed, shards
    written here, fork path disabled.
    """
    m = _manager()
    writes = _writes(tmp_path)
    monkeypatch.setattr(mod, "_FORK_TIMEOUT_S", 0.5)
    real = mod._write_shards
    parent = os.getpid()

    def _hang_in_the_child(w):
        if os.getpid() != parent:
            time.sleep(600)
        return real(w)

    monkeypatch.setattr(mod, "_write_shards", _hang_in_the_child)

    m._start_async_write(writes, str(tmp_path))
    pid = m._pending_save[1]
    started = time.monotonic()
    with pytest.warns(UserWarning, match="did not finish within"):
        m.wait_for_pending_save()

    assert time.monotonic() - started < 60, "should give up at the timeout, not wait it out"
    for state, p in writes:
        assert torch.equal(torch.load(p, weights_only=False)["w"], state["w"])
    assert m._fork_broken is True
    with pytest.raises(OSError):
        os.kill(pid, 0)          # reaped, not left as a zombie or still running


def test_fork_broken_falls_back_to_the_thread(tmp_path):
    m = _manager(fork_broken=True)
    writes = _writes(tmp_path)

    m._start_async_write(writes, str(tmp_path))

    assert m._pending_save[0] == "thread"
    m.wait_for_pending_save()
    for state, p in writes:
        assert torch.equal(torch.load(p, weights_only=False)["w"], state["w"])


def test_env_flag_off_uses_the_thread(tmp_path, monkeypatch):
    """CKPT_FORK_WRITER=0 is the A/B for whether forking helped, so "off" has to
    mean the old path exactly."""
    monkeypatch.setattr(mod, "_FORK_WRITER", False)
    m = _manager()
    writes = _writes(tmp_path)

    m._start_async_write(writes, str(tmp_path))

    assert m._pending_save[0] == "thread"
    m.wait_for_pending_save()
    for state, p in writes:
        assert torch.equal(torch.load(p, weights_only=False)["w"], state["w"])


def test_thread_failure_still_raises(tmp_path, monkeypatch):
    """The fallback path keeps its own contract: a thread that failed and left no
    shards must raise rather than let the run report success."""
    monkeypatch.setattr(mod, "_FORK_WRITER", False)
    monkeypatch.setattr(mod, "_write_shards", lambda w: (_ for _ in ()).throw(RuntimeError("disk full")))
    m = _manager()

    m._start_async_write(_writes(tmp_path), str(tmp_path))
    with pytest.raises(RuntimeError, match="background checkpoint write failed"):
        m.wait_for_pending_save()


def test_wait_is_a_noop_with_nothing_in_flight():
    m = _manager()
    assert m.wait_for_pending_save() is None


def test_a_second_write_cannot_start_while_one_is_in_flight(tmp_path):
    m = _manager()
    m._start_async_write(_writes(tmp_path), str(tmp_path))
    try:
        with pytest.raises(AssertionError, match="already in flight"):
            m._start_async_write(_writes(tmp_path), str(tmp_path))
    finally:
        m.wait_for_pending_save()


def test_a_rewrite_that_also_fails_is_raised(tmp_path, monkeypatch):
    """The fork path swallows the child's failure only because it repairs it.

    Rewriting in the parent is what makes a dead child harmless, so when the
    rewrite fails too there is nothing left to fall back on and the run must hear
    about it -- otherwise this path would be strictly worse than the thread it
    replaced, which always raised.
    """
    m = _manager()
    monkeypatch.setattr(mod, "_write_shards",
                        lambda w: (_ for _ in ()).throw(RuntimeError("disk full")))

    m._start_async_write(_writes(tmp_path), str(tmp_path))
    with pytest.warns(UserWarning, match="forked checkpoint writer failed"):
        with pytest.raises(RuntimeError, match="disk full"):
            m.wait_for_pending_save()

    assert m._fork_broken is True


def test_the_child_leaves_without_running_atexit_handlers(tmp_path):
    """``os._exit`` rather than ``sys.exit``.

    The child shares the parent's CUDA context, NCCL comms and stdio buffers. An
    atexit handler running in the child would tear down or flush something this
    process still owns, which is a far worse failure than a lost checkpoint
    because it would corrupt the *parent*.
    """
    import inspect

    source = inspect.getsource(mod.FSDPCheckpointManager._start_forked_write)
    assert "os._exit(" in source
    assert "sys.exit(" not in source
