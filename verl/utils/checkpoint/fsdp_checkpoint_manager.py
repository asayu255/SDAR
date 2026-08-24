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

import os
import signal
import subprocess
import threading
import time
import traceback
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager

try:
    from torch.distributed._shard.sharded_tensor import ShardedTensor as _ShardedTensor
except Exception:  # pragma: no cover - torch without the sharded-tensor API
    _ShardedTensor = None

try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:  # pragma: no cover - torch without the DTensor API
    _DTensor = None


# A background *thread* writing the checkpoint shares the GIL with the training
# loop, and torch.save holds it for as long as it is pickling. Measured on run
# bgwezy3k, that inflates the step after each save from a 297 s median to
# 420-519 s -- 186 s of excess per save, 2.25% of the run. Writing from a forked
# child removes the contention by construction: a separate interpreter has a
# separate GIL, and fork's copy-on-write means the staged shards are not copied
# or transferred, only read.
#
# Off with CKPT_FORK_WRITER=0, which restores the thread and is the A/B for
# whether this helped. Also skipped where os.fork does not exist.
_FORK_WRITER = os.environ.get("CKPT_FORK_WRITER", "1").lower() not in ("0", "false", "no", "")
# How long wait_for_pending_save() will wait before it gives up on the child and
# writes the shards itself. Generous on purpose: it is only reached a whole step
# after the fork (~300 s), by which point a healthy write (~178 s solo) is long
# done, so this fires only for a child that is stuck rather than slow.
_FORK_TIMEOUT_S = float(os.environ.get("CKPT_FORK_TIMEOUT_S", "900"))
# Overlap the save's device-to-host copy with the next step instead of paying
# it on the training thread. CKPT_SNAPSHOT=0 keeps the blocking copy, which is
# the A/B and the fallback.
_SNAPSHOT_OFFLOAD = os.environ.get("CKPT_SNAPSHOT", "1").lower() not in ("0", "false", "no", "")


def fork_writer_enabled() -> bool:
    return _FORK_WRITER and hasattr(os, "fork")


def _write_shards(writes):
    """Pickle each ``(state_dict, path)`` to disk. No device or collective work."""
    for state_dict, path in writes:
        torch.save(state_dict, path)


def _finish_offload_to_cpu(obj, moved, name, offload=None):
    """Return ``obj`` with every tensor on CPU, recording what had to move.

    ``offload_to_cpu`` on the sharded state-dict configs moves the sharded
    parameters, but not everything the state dict carries: buffers and other
    plain tensors can come back still on the device. The writer thread hid that
    -- ``torch.save`` calls ``storage.cpu()`` itself, a quiet D2H inside the
    "async" write, device work the design said the writer must never do. The
    forked child could not hide it: a CUDA context does not survive fork, so its
    first CUDA call was its last ("CUDA error: initialization error", measured
    on run nbq51imk, all three ranks, first save). Staging finishes the offload
    here, on the thread where CUDA works, so the writer -- thread or child --
    receives tensors it can pickle without touching the device.

    ``moved`` collects ``(name, device, bytes)`` per straggler, so the log can
    say exactly which tensors offload_to_cpu missed rather than that some did.
    ShardedTensor is checked before Tensor because it subclasses it, and its
    local shards are moved in place -- the state dict owns them, nothing else
    holds these copies.
    """
    if _ShardedTensor is not None and isinstance(obj, _ShardedTensor):
        for index, shard in enumerate(obj.local_shards()):
            if shard.tensor.device.type != "cpu":
                moved.append((f"{name}<shard{index}>", str(shard.tensor.device),
                              shard.tensor.numel() * shard.tensor.element_size()))
                shard.tensor = _route(shard.tensor, f"{name}<shard{index}>", offload)
        return obj
    if _DTensor is not None and isinstance(obj, _DTensor):
        local = obj.to_local()
        if local.device.type != "cpu":
            moved.append((f"{name}<local>", str(local.device),
                          local.numel() * local.element_size()))
            # In place, and NOT via DTensor.from_local: from_local normalizes
            # the local tensor to the mesh's device type, so it hands back a
            # CUDA shard that round-trips fine here and kills the forked writer,
            # which has no CUDA context (measured, scripts/check_dtensor_offload.py).
            # Assigning the shard keeps it on CPU and survives save/load. Safe
            # because the walk owns this state-dict entry -- FSDP built it for
            # this save and nothing else holds a reference.
            obj._local_tensor = _route(local, f"{name}<local>", offload)
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu":
            return obj
        moved.append((name, str(obj.device), obj.numel() * obj.element_size()))
        return _route(obj, name, offload)
    if isinstance(obj, dict):
        return {key: _finish_offload_to_cpu(value, moved, f"{name}.{key}", offload) for key, value in obj.items()}
    if isinstance(obj, torch.Size):
        return obj
    if isinstance(obj, tuple):
        items = [_finish_offload_to_cpu(value, moved, f"{name}[{i}]", offload) for i, value in enumerate(obj)]
        return type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
    if isinstance(obj, list):
        return [_finish_offload_to_cpu(value, moved, f"{name}[{i}]", offload) for i, value in enumerate(obj)]
    return obj


class _Offload:
    """Where a save's device tensors go, and by which route.

    Two routes. The blocking one copies device -> pinned host on the training
    thread: correct, simple, and worth ~0.3-0.6 s of dead device time per save
    because a D2H moves no SM work and NVML reads the card as idle for its
    duration.

    The overlapped one splits that in half. A device -> device copy into a
    persistent snapshot buffer runs on the main stream (7 GB of traffic at
    ~1.5 TB/s, tens of milliseconds, and it IS SM work so the card stays busy),
    and the snapshot -> pinned host copy is then issued on a side stream, where
    it overlaps the next step's compute and costs the training thread nothing.
    The snapshot exists precisely because the side stream cannot read the live
    tensors: the next step overwrites them, and stream ordering does not
    protect a reader on another stream.
    """

    def __init__(self, mirror, snapshot=None):
        self.mirror = mirror        # name -> pinned host tensor, reused across saves
        self.snapshot = snapshot    # name -> device tensor, or None to copy inline
        self.pairs = []             # (device snapshot, host mirror) for the side stream

    def stage(self, tensor, name):
        if type(tensor) is not torch.Tensor:
            # Tensor subclasses carry placement and dispatch rules a plain
            # buffer cannot receive: DTensor refuses plain.copy_(dtensor)
            # outright. Let the subclass move itself rather than crash a save
            # -- slower, and correct for anything FSDP hands back.
            return tensor.detach().cpu()
        host = _slot(self.mirror, name, tensor, pin=True)
        if self.snapshot is None:
            host.copy_(tensor)
            return host
        device_slot = _slot(self.snapshot, name, tensor, device=tensor.device)
        device_slot.copy_(tensor)   # main stream, ordered against the next step
        self.pairs.append((device_slot, host))
        return host


def _slot(store, name, like, pin=False, device="cpu"):
    """A reusable buffer shaped like ``like``, kept across saves.

    A fresh allocation is paid for in page faults as the copy touches each new
    page, and a fresh *pinned* allocation additionally page-locks memory every
    time. A slot whose shape or dtype changed is replaced rather than truncated
    into: optimizer state is fixed once allocated, but a silently short copy
    would corrupt the checkpoint, which is worse than an allocation. Pinning is
    a finite resource, so failing to pin degrades to pageable rather than
    taking the save down.
    """
    slot = store.get(name)
    if slot is not None and slot.shape == like.shape and slot.dtype == like.dtype:
        return slot
    try:
        slot = torch.empty(like.shape, dtype=like.dtype, device=device, pin_memory=pin)
    except (RuntimeError, MemoryError):
        slot = torch.empty(like.shape, dtype=like.dtype, device=device)
    store[name] = slot
    return slot


def _route(tensor, name, offload):
    """Send one straggler to CPU by whichever route this save is using."""
    if offload is None:
        return tensor.detach().cpu()
    return offload.stage(tensor, name)


def _into_mirror(tensor, name, mirror):
    """Copy ``tensor`` to CPU, into the same PINNED memory as the last save.

    Two costs are being removed. A fresh allocation is paid for in page faults
    as the copy touches each new page; the mirror pays that once. And a copy
    into ordinary pageable memory cannot use DMA -- the driver stages it
    through a bounce buffer at roughly 4 GB/s, which is what made the offload
    walk 1.7-1.9 s for ~7 GB (measured, run 46o9xef3). Pinned pages let the
    same copy go over PCIe at 12-25 GB/s.

    Pinning is a finite resource and the allocation can fail; pageable is then
    still correct, just slower, so a failure degrades rather than raises. A slot
    whose shape or dtype changed is replaced rather than truncated into --
    optimizer state is fixed once allocated, but a silently short copy would
    corrupt the checkpoint, which is worse than an allocation.
    """
    if mirror is None:
        return tensor.detach().cpu()
    return _Offload(mirror).stage(tensor, name)


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
        async_save (bool):
            Write the staged shards on a background thread instead of blocking
            the training loop on them. See :py:meth:`save_checkpoint`.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        async_save: bool = False,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )
        # Only meaningful where the state dicts are staged to CPU, which is the
        # same condition the configs below use: with offload_to_cpu=False the
        # shards handed to torch.save are the live device tensors, and writing
        # them while the next step updates them would save a mix of two steps.
        self.async_save = bool(async_save) and is_cuda_available
        if async_save and not self.async_save:
            warnings.warn(
                "async checkpoint saving needs the CPU-staged state dict "
                "(offload_to_cpu), which is only used on CUDA; saving synchronously",
                stacklevel=2,
            )
        # ("thread", thread, path) or ("fork", pid, err_fd, writes, path)
        self._pending_save = None
        self._pending_error = None  # exception it died with, re-raised on the main thread
        # One fork failure disables the fork path for the rest of the run. A
        # process carrying CUDA, NCCL-watchdog and vLLM threads can fork a child
        # that deadlocks on a lock no surviving thread will release, and paying
        # the timeout once per save for the rest of a 300-step run would be far
        # worse than the contention this is trying to remove.
        self._fork_broken = False
        self.last_write_seconds = None   # the writer's own time on the write
        self.last_writer_kind = None     # "fork" or "thread", for the same line
        self._thread_write_seconds = None
        # Persistent CPU destinations for the offload walk, keyed by tensor
        # name. The stragglers are ~4.6 GB of Adam state, and allocating fresh
        # pageable memory for them on every save costs page faults on every
        # copied byte; reusing the same pages does not. Safe to reuse because
        # save_checkpoint joins the pending write before staging again -- the
        # writer (thread or forked child) is finished with these pages before
        # anything overwrites them.
        self._offload_mirror = {}
        # Persistent device buffers for the overlapped route, plus the stream
        # and the copy still in flight on it. Allocated on the first save that
        # needs them; an allocation failure pins _snapshot_ok False and every
        # later save takes the blocking route, which is correct and only slower.
        self._snapshot_buffers = {}
        self._snapshot_ok = _SNAPSHOT_OFFLOAD
        self._copy_stream = None
        self._pending_copy = None   # (event, writes, path, started) awaiting its D2H
        # GenerationConfig.from_pretrained goes to the HF hub, and the old code
        # paid that network round trip on EVERY save. The config cannot change
        # mid-run; fetch it once.
        self._generation_config = None

    def _start_async_write(self, writes, local_path: str):
        """Hand the staged shards to a forked child (or a thread) and return.

        ``writes`` holds CPU copies produced by the offload_to_cpu state dict, so
        the writer reads memory the training loop no longer touches. It runs no
        collective and issues no device work, which is what makes it safe to run
        beside the next step -- a barrier from here would be a second thread
        entering NCCL and is exactly what this must not do.

        A forked child is preferred because a thread does not actually get the
        write off the critical path: it holds the GIL while pickling, so the next
        step's kernel launches -- which are Python calls -- stall behind it. The
        child has its own interpreter and cannot do that. The thread remains as
        the fallback, and is what CKPT_FORK_WRITER=0 selects.
        """
        assert self._pending_save is None, "a checkpoint write is already in flight"

        if fork_writer_enabled() and not self._fork_broken and self._start_forked_write(writes, local_path):
            return

        self._thread_write_seconds = None

        def _run():
            write_started = time.monotonic()
            try:
                _write_shards(writes)
                self._thread_write_seconds = time.monotonic() - write_started
            except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
                self._pending_error = e

        thread = threading.Thread(
            target=_run, name=f"ckpt-write-rank{self.rank}", daemon=True
        )
        self._pending_save = ("thread", thread, local_path, time.monotonic())
        thread.start()

    def _start_forked_write(self, writes, local_path: str) -> bool:
        """Fork a child that writes the shards and exits. True if it started.

        The child must touch nothing that belongs to the threads it did not
        inherit. Only the forking thread survives a fork, so any lock another
        thread held at that instant stays locked forever in the child -- which is
        why the child does no logging, no CUDA, no NCCL, no Ray, and leaves via
        ``os._exit`` so that no atexit handler runs and tries to tear down a CUDA
        context or flush a stdio buffer this process still owns.

        ``writes`` stays referenced by the parent as well. Copy-on-write means the
        child reads the same physical pages rather than copying ~20 GB, and the
        parent needs them anyway to rewrite the shards itself if the child fails.
        """
        read_fd, write_fd = os.pipe()
        try:
            pid = os.fork()
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            self._fork_broken = True
            message = (
                f"[ckpt-write] rank {self.rank}: could not fork a checkpoint writer "
                f"({exc!r}); falling back to the writer thread"
            )
            print(message, flush=True)
            warnings.warn(message, stacklevel=2)
            return False

        if pid == 0:                                    # child
            status = 0
            try:
                os.close(read_fd)
                write_started = time.monotonic()
                _write_shards(writes)
                # The receipt. A clean exit status alone is not proof the shards
                # were written -- waitpid can be robbed of the real status by a
                # SIGCHLD-ignoring host process -- so the parent requires this
                # line, and gets the child's own write time with it.
                os.write(write_fd, f"ok {time.monotonic() - write_started:.1f}".encode())
            except BaseException:                       # noqa: BLE001 - reported over the pipe
                status = 1
                try:
                    os.write(write_fd, traceback.format_exc()[-4096:].encode())
                except BaseException:                   # noqa: BLE001 - nothing left to report with
                    pass
            finally:
                try:
                    os.close(write_fd)
                except BaseException:                   # noqa: BLE001
                    pass
            os._exit(status)

        os.close(write_fd)                              # parent
        print(f"[ckpt-write] rank {self.rank}: forked writer pid {pid} started", flush=True)
        self._pending_save = ("fork", pid, read_fd, writes, local_path, time.monotonic())
        return True

    def _reap_forked_write(self, pid, err_fd, writes, path):
        """Wait for the child; return its self-reported write seconds, or rewrite.

        Success is the ``ok <seconds>`` line on the pipe, not the exit status.
        The status can be fabricated -- a host process that ignores SIGCHLD robs
        waitpid of the real one -- and an exit status says nothing about whether
        the shards actually landed; the receipt is written by the child after
        ``_write_shards`` returns, so it can only exist if they did.

        Any failure -- no receipt, a non-zero exit, a signal, or a child that
        never finishes -- ends the same way: the fork path is disabled for the
        rest of the run and this thread writes the shards synchronously (None is
        returned; the caller's duration then reflects that rewrite). The
        checkpoint is what matters, and degrading to a synchronous write is
        exactly the behaviour async_save started from. Only a failure of that
        rewrite is worth raising.
        """
        deadline = time.monotonic() + _FORK_TIMEOUT_S
        interval = 0.01
        status = None
        timed_out = False
        while True:
            try:
                done, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:                   # already reaped; treat as success
                done, status = pid, 0
            if done:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(interval)
            interval = min(interval * 2, 0.2)

        detail = ""
        if timed_out:
            for sig in (signal.SIGKILL, None):
                if sig is None:
                    break
                try:
                    os.kill(pid, sig)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
            detail = f"writer did not finish within {_FORK_TIMEOUT_S:.0f}s"
        else:
            try:
                detail = os.read(err_fd, 65536).decode(errors="replace").strip()
            except OSError:
                detail = ""
        try:
            os.close(err_fd)
        except OSError:
            pass

        receipt = detail.startswith("ok ")
        dirty_exit = status is not None and not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0)
        if receipt and not timed_out and not dirty_exit:
            try:
                return float(detail.split()[1])
            except (IndexError, ValueError):
                return None

        self._fork_broken = True
        message = (
            f"[ckpt-write] rank {self.rank}: forked checkpoint writer failed "
            f"({detail or f'status {status}'}); writing {path} on this thread "
            "and using the writer thread from now on"
        )
        print(message, flush=True)
        warnings.warn(message, stacklevel=2)
        _write_shards(writes)
        return None

    def _issue_overlapped_d2h(self, pairs):
        """Copy snapshot -> pinned host on a side stream. Returns the event, or None.

        The side stream waits for the main stream so it cannot start before the
        device-to-device snapshot it reads has landed; the main stream does NOT
        wait for it, which is the entire point -- the next step's kernels are
        issued immediately and the copy rides alongside them.
        """
        try:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream()
            self._copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._copy_stream):
                for device_slot, host in pairs:
                    host.copy_(device_slot, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
            return event
        except Exception as exc:      # noqa: BLE001 - degrade, never fail a save
            warnings.warn(
                f"[rank-{self.rank}]: could not overlap the checkpoint copy ({exc!r}); "
                "falling back to the blocking one",
                stacklevel=2,
            )
            self._snapshot_ok = False
            return None

    def start_write_if_copy_done(self) -> bool:
        """Fork the writer if the overlapped copy has landed. Never blocks.

        Called from the training loop once the step is under way -- by then the
        copy issued at the previous save has had a whole forward to finish in,
        so this normally forks without waiting and the write gets a full step to
        run in, exactly as it did when the copy was synchronous. Returns whether
        it started one, so a caller can log it.
        """
        pending = self._pending_copy
        if pending is None:
            return False
        event, writes, path, started = pending
        if not event.query():
            return False
        self._pending_copy = None
        print(
            f"[ckpt-write] rank {self.rank}: overlapped copy landed "
            f"{time.monotonic() - started:.1f} s after staging; starting the write",
            flush=True,
        )
        self._start_async_write(writes, path)
        return True

    def _drain_pending_copy(self):
        """Block until the overlapped copy lands, then start its write.

        The backstop for start_write_if_copy_done: anything that needs the
        shards -- the next save, the flush that publishes the tracker -- must
        find a write in flight rather than a copy nobody picked up.
        """
        if self._pending_copy is None:
            return
        event, writes, path, _ = self._pending_copy
        self._pending_copy = None
        event.synchronize()
        self._start_async_write(writes, path)

    def _remove_paths_in_background(self, paths):
        """Fire-and-forget ``rm -rf`` for rotated-out checkpoint directories.

        Correctness does not depend on when the delete lands (see the call
        site); what it must never do is block the save or take the run down. A
        Popen that fails to spawn degrades to the synchronous remove rather
        than leaking the disk.
        """
        for path in paths:
            try:
                subprocess.Popen(
                    ["rm", "-rf", "--", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.remove_previous_save_local_path([path])

    def wait_for_pending_save(self):
        """Block until the background write finishes; re-raise what it caught.

        Returns the path that was being written, or ``None`` if nothing was in
        flight. Safe to call at any time, including when ``async_save`` is off.

        Callers must reach this before treating a checkpoint as complete. A
        failure that is only recorded on the writer would otherwise let a run
        finish reporting success with no usable checkpoint on disk, which is the
        failure mode this whole mechanism could plausibly introduce. With the
        forked writer that guarantee is stronger rather than weaker: a child that
        fails or hangs is rewritten here before this returns, so the shards exist
        by the time anyone can act on them.
        """
        self._drain_pending_copy()
        pending, self._pending_save = self._pending_save, None
        path = None
        if pending is not None:
            started = pending[-1]
            write_seconds = None
            if pending[0] == "fork":
                _, pid, err_fd, writes, path, _ = pending
                write_seconds = self._reap_forked_write(pid, err_fd, writes, path)
            else:
                _, thread, path, _ = pending
                thread.join()
                write_seconds = self._thread_write_seconds
            # Two different numbers, and the first probe confused them: the time
            # the writer spent writing (the child's or the thread's own clock),
            # and the time from start to this flush -- which is roughly a whole
            # step regardless of the write, because the flush only runs here.
            # The 500 s the first probe printed was the second number wearing
            # the first one's label.
            flush_after = time.monotonic() - started
            self.last_write_seconds = write_seconds if write_seconds is not None else flush_after
            # Names what actually wrote the shards. A fork whose child died was
            # rewritten here, and calling that "fork" would hide the one event
            # worth noticing in a log.
            self.last_writer_kind = pending[0]
            if pending[0] == "fork" and self._fork_broken:
                self.last_writer_kind = "fork-failed-rewritten-by-parent"
            print(
                f"[ckpt-write] rank {self.rank}: {self.last_writer_kind} writer finished "
                f"{os.path.basename(str(path))}: write {self.last_write_seconds:.1f} s, "
                f"start-to-flush {flush_after:.1f} s",
                flush=True,
            )
        error, self._pending_error = self._pending_error, None
        if error is not None:
            raise RuntimeError(
                f"[rank-{self.rank}]: background checkpoint write failed"
                + (f" ({path})" if path else "")
            ) from error
        return path

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        With ``async_save`` the three ``torch.save`` calls run on a background
        thread and this returns once the shards are staged. Measured on this arm
        (3xA6000, Qwen3-1.7B, wandb x7g9r7bx) a save took 198 s, of which only the
        first ~20 s touched the GPU at all -- building the sharded state dict and
        copying it to host memory. The remaining ~178 s had SM at 0.0%, the memory
        controller at 0.0% and the cards at their 28 W idle floor: pure pickling
        and disk I/O, with the training loop blocked behind it for no reason.

        What stays on this thread is what cannot leave it: ``state_dict()`` is a
        collective and a device copy, and the barriers are NCCL collectives, which
        must be issued from one thread in the same order on every rank. What moves
        is only the write of tensors that are already CPU copies, so the next
        step's updates cannot reach them.

        Two obligations come with it, both on the caller:

        * ``wait_for_pending_save()`` must run before anything treats the
          checkpoint as complete -- above all before the
          ``latest_checkpointed_iteration.txt`` that a resume reads, which would
          otherwise be able to name a half-written directory. This method calls it
          on entry, so the write is also drained before the rotation below deletes
          anything and before a second save stages another copy.
        * the process must not exit with a write in flight.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # Before the rotation below removes a directory and before this stages a
        # second copy of the shards: at most one save is ever in flight.
        self.wait_for_pending_save()

        # record the previous global step
        self.previous_global_step = global_step

        # Rotation deletes ~17-20 GB of an old checkpoint, and a synchronous
        # unlink walk of that much sits on the save's critical path with the
        # device idle behind it. Hand it to rm -rf in a fire-and-forget child:
        # the directory being removed is at least two saves old, fully written
        # (wait_for_pending_save above), and the tracker can never name it, so
        # nothing observes whether the delete happens now or seconds later.
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self._remove_paths_in_background(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard.
        #
        # Under async_save the model's offload_to_cpu is OFF so its D2H can be
        # overlapped like the optimizer's: FSDP does that copy synchronously on
        # this thread, and it was the whole remaining save cost (model build
        # 3.2-3.8 s of a 3.58 s save, run ovb0yobz). The walk now unwraps a
        # DTensor to its local shard, snapshots that, and swaps the shard in
        # place -- checked on the device rather than assumed, after an earlier
        # blind attempt crashed every rank (ke23u52t) and a from_local version
        # silently moved the shard back to CUDA.
        offload_model = is_cuda_available and not (self.async_save and _DTensor is not None)
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=offload_model)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                stage_started = time.perf_counter()
                model_state_dict = self.model.state_dict()
                model_built = time.perf_counter()
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                optim_built = time.perf_counter()
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                writes = [
                    (model_state_dict, model_path),
                    (optimizer_state_dict, optim_path),  # TODO: address optimizer is None
                    (extra_state_dict, extra_path),
                ]
                if self.async_save:
                    moved = []
                    walk_started = time.perf_counter()
                    offload = _Offload(
                        self._offload_mirror,
                        self._snapshot_buffers if (self._snapshot_ok and is_cuda_available) else None,
                    )
                    try:
                        writes = [
                            (_finish_offload_to_cpu(state, moved, kind, offload), target)
                            for (state, target), kind in zip(writes, ("model", "optim", "extra"))
                        ]
                    except torch.cuda.OutOfMemoryError:
                        # The snapshot is ~7 GB of device memory. If it does not
                        # fit, the blocking route still saves the checkpoint --
                        # give up the overlap for the rest of the run rather than
                        # the run itself, and drop what was allocated so far.
                        warnings.warn(
                            f"[rank-{self.rank}]: no room for the checkpoint snapshot; "
                            "copying on the training thread from now on",
                            stacklevel=2,
                        )
                        self._snapshot_ok = False
                        self._snapshot_buffers.clear()
                        moved = []
                        offload = _Offload(self._offload_mirror)
                        writes = [
                            (_finish_offload_to_cpu(state, moved, kind, offload), target)
                            for (state, target), kind in zip(writes, ("model", "optim", "extra"))
                        ]
                    walk_done = time.perf_counter()
                    if moved:
                        sample = ", ".join(entry[0] for entry in moved[:4])
                        if len(moved) > 4:
                            sample += ", ..."
                        print(
                            f"[ckpt-write] rank {self.rank}: moved {len(moved)} tensors "
                            f"({sum(entry[2] for entry in moved) / 1e6:.0f} MB) that "
                            f"offload_to_cpu left on the device ({sample})",
                            flush=True,
                        )
                    event = self._issue_overlapped_d2h(offload.pairs) if offload.pairs else None
                    if event is None:
                        if offload.pairs:
                            # The issue failed after the snapshot was taken; the
                            # mirrors hold nothing yet, so copy them here.
                            for device_slot, host in offload.pairs:
                                host.copy_(device_slot)
                        self._start_async_write(writes, local_path)
                    else:
                        self._pending_copy = (event, writes, local_path, time.monotonic())
                else:
                    _write_shards(writes)

        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                if self._generation_config is None:
                    self._generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config = self._generation_config
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local -- under async_save, to have *staged*
        # it: the shards land later, and wait_for_pending_save() is what says they
        # are there. Nothing below reads them, and the hf_model branch works off a
        # fresh full state dict rather than the files.
        torch.distributed.barrier()

        if self.async_save:
            # The whole device-idle window this save cost, phase by phase. The
            # builds are FSDP assembling the state dicts (collectives, pinned
            # to this thread); the walk is the snapshot copy into the reused
            # mirror; extras is the rank-0 config/tokenizer writes plus the
            # barrier absorbing the slowest rank. What dominates here decides
            # what to shrink next.
            print(
                f"[ckpt-write] rank {self.rank}: save cost "
                f"{time.perf_counter() - stage_started:.1f} s on this thread "
                f"(model build {model_built - stage_started:.1f}, "
                f"optim build {optim_built - model_built:.1f}, "
                f"offload {walk_done - walk_started:.2f} "
                f"[{'snapshot, overlapped' if self._pending_copy else 'blocking copy'}], "
                f"extras+barrier {time.perf_counter() - walk_done:.1f})",
                flush=True,
            )

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)
