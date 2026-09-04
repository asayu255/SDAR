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
"""GPU memory as every rank sees it, not just as rank 0 sees it.

``DataProto.concat`` keeps ``meta_info`` from the FIRST worker only, so anything
a worker returns in ``meta_info["metrics"]`` reaches the logger from rank 0
alone -- the other ranks' copies are dropped on the floor. ``reduce_metrics``
then runs over what are already scalars, which makes a key like
``perf/max_memory_reserved_gb`` read as a cross-rank reduction when it is one
card's number under a misleading name.

That matters here because the ranks are not interchangeable in practice even
though they should be on paper. FSDP ``shard_grad_op`` replicates parameters and
shards gradients and optimizer state evenly, and ``_balance_batch`` hands every
rank the same row count and (Karmarkar-Karp) near-identical token totals, so the
steady-state footprint is expected to match. What does not match is the PEAK:
KK equalises the SUM per rank, and the way it gets there is to put the few
longest sequences on one rank and compensate with short ones. Micro-batches are
then cut by row count, so the rank holding the long rows draws a micro-batch
with several times the tokens -- and the logits it has to materialise scale with
exactly that. The caching allocator gives the peak back only where
``empty_cache`` is called, which on these arms is the rollout boundary and not
the training path, so ``nvidia-smi`` shows a ratchet across a step: whatever a
rank once reached during the update, it keeps until the next generation.

One all-gather of five floats per step is what makes any of that visible.

The last of the five is not about the card at all. ``pinned_host_bytes`` reports
the page-locked HOST pool, which every other counter here is blind to and which
no CUDA metric can see -- and which is what actually killed run e8x57zyu, at
Ray's node-memory threshold rather than on a GPU. It is summed across ranks
rather than maxed, because that memory is charged to the box the ranks share.

Two more of the five are the allocator's own counters. ``num_alloc_retries``
increments every time a malloc could not be served from the cache and the
allocator had to release cached blocks back to the driver and try again -- and
``cudaFree`` synchronizes the device, so each retry is a full queue drain in the
middle of a micro-batch. That is the difference between "this phase is slow" and
"this phase keeps stopping the GPU to go shopping for memory", and no amount of
NVML sampling can tell them apart from outside. ``num_ooms`` is the same counter
one step further along: an allocation that failed even after the retry.
"""

import torch

__all__ = [
    "per_rank_memory_metrics", "device_footprint_gb", "reset_phase_peak", "phase_peak_metrics",
    "pinned_host_bytes", "node_memory_breakdown",
]

_GB = 1024.0**3

# The lifetime high-water marks, carried across the resets reset_phase_peak does.
# torch's own max_memory_* ARE the lifetime marks only while nothing resets them;
# once a phase window opens one, the reading after it covers that phase alone.
# Both readings are wanted -- see reset_phase_peak -- so the older one is kept
# here rather than lost.
_LIFETIME = [0.0, 0.0]


def _allocator_counters(device) -> tuple:
    """(num_alloc_retries, num_ooms) for this rank, 0 when unavailable.

    Never lets a diagnostic take the run down: any backend without
    ``memory_stats`` just reports zeros.
    """
    try:
        stats = device.memory_stats()
    except Exception:
        return 0.0, 0.0
    return float(stats.get("num_alloc_retries", 0)), float(stats.get("num_ooms", 0))


def pinned_host_bytes() -> float:
    """Page-locked HOST memory torch's pinned pool owns -- handed out and cached.

    The counter this file was missing, and the one that would have named a crash.
    Pinned memory is not the process's to give back: ``CachingHostAllocator``
    rounds every request up to a power of two, keys its free list by that bucket,
    and does not return blocks to the OS (on torch 2.8 at all; on 2.14 only above
    ``pinned_max_cached_size``, SIZE_MAX by default). So the pool is a ratchet over
    the distinct bucket sizes a run has ever asked for, it is invisible to every
    CUDA memory metric above, and RSS is where it shows up -- next to Ray's own
    accounting, which kills the node at ``RAY_memory_usage_threshold``.

    That is how run e8x57zyu died at step 10: the teacher cache had started pinning
    its per-put chunks, host RAM climbed 5.06 GB a step from 207 GB, and nothing in
    ``perf/*`` moved, because everything in ``perf/*`` was about the card.

    WHICH KEY, and why it is not the larger of the two. On torch 2.8 -- what
    setup.py pins, so what the runs are on -- ``getStats`` fills ``reserved_bytes``
    from the slow-path counter (+size on every ``cudaHostAlloc``, -size only on
    ``cudaFreeHost``, i.e. the pool) and ``allocated_bytes`` from the per-bucket
    handed-out counters. Those per-bucket counters are BROKEN on 2.8: a block that
    is freed with a stream event pending is returned to the free list by
    ``process_events_for_specific_size(size)``, which decrements the bucket by the
    ``size`` argument -- and the generic ``process_events()`` passes -1. So every
    block the micro-batch loop has read from (which records an event on it) leaves
    its full size in ``allocated_bytes.current`` when it is recycled. Run p7qdwsyl
    logged exactly that: +16.04 GiB a step, one 8 GiB store block a rank, to a
    "pool" of 1027 GB on a 251 GB node. The first version of this function took
    ``max`` of the two and reported the broken one.

    On 2.14 the counters were reworked: ``allocated_bytes`` IS the pool (active +
    cached, at the rounded size), handed-out moved to ``active_bytes``, and there
    is no ``reserved_bytes``. So: ``reserved_bytes.current`` when the key exists,
    ``allocated_bytes.current`` only when it does not. Never the max.

    0.0 on any backend without the counters, so a diagnostic never takes a run down.
    """
    try:
        stats = torch.cuda.host_memory_stats()
    except Exception:  # noqa: BLE001 - a measurement must not break what it measures
        return 0.0
    if "reserved_bytes.current" in stats:
        return float(stats["reserved_bytes.current"] or 0.0)
    return float(stats.get("allocated_bytes.current", 0.0) or 0.0)


def node_memory_breakdown(prefix: str = "node_mem", top: int = 12, process_iter=None, shm_usage=None) -> dict:
    """Host memory by PROCESS CLASS, node-wide, plus the shared-memory object store.

    ``perf/cpu_memory_used_gb`` is one number for the whole box, and three runs
    have now died against Ray's threshold with only that number and a kill-time
    "top 10" to reason from. This is the breakdown that turns the next one into a
    lookup: how much do the two training workers hold, how much the driver, how
    much the few hundred environment actors together, and how much sits in
    ``/dev/shm``, where Ray's plasma store lives and fills to its cap with
    dead-but-cached objects as a matter of course.

    Ray sets each worker's process title to ``ray::<Class>.<method>``, so grouping
    on the class name collapses ~200 ``WebshopWorker`` processes into one row.
    Sizes are PSS (proportional set size) where the kernel offers it, so pages
    shared between processes -- shared libraries, and the plasma store mapped into
    every worker that reads from it -- are split rather than counted once per
    process; RSS is the fallback. One sweep of /proc per step, ~0.1 s at 300
    processes, from rank 0 only.

    ``process_iter`` and ``shm_usage`` are injection points for tests. Never
    raises: a process that exits mid-sweep is skipped, and a box without
    ``/dev/shm`` just omits that key.
    """
    import os

    import psutil

    if process_iter is None:
        process_iter = lambda: psutil.process_iter(["cmdline", "name"])  # noqa: E731
    if shm_usage is None:
        shm_usage = lambda: psutil.disk_usage("/dev/shm").used  # noqa: E731

    by_class: dict = {}
    for proc in process_iter():
        try:
            cmd = proc.info.get("cmdline") if isinstance(getattr(proc, "info", None), dict) else None
            head = (cmd[0] if cmd else None) or (proc.info.get("name") if isinstance(getattr(proc, "info", None), dict) else None) or "?"
            try:
                size = proc.memory_full_info().pss
            except Exception:  # noqa: BLE001 - no smaps_rollup, or no permission
                size = proc.memory_info().rss
        except Exception:  # noqa: BLE001 - exited or inaccessible mid-sweep
            continue
        if head.startswith("ray::"):
            cls = head[5:].split(".")[0].split("(")[0] or "ray"
        else:
            cls = os.path.basename(head)[:24] or "?"
        by_class[cls] = by_class.get(cls, 0.0) + float(size)

    ranked = sorted(by_class.items(), key=lambda kv: -kv[1])[:top]
    out = {f"{prefix}/{cls}_gb": size / _GB for cls, size in ranked}
    out[f"{prefix}/other_gb"] = sum(size for cls, size in by_class.items() if cls not in dict(ranked)) / _GB
    try:
        out[f"{prefix}/dev_shm_gb"] = float(shm_usage()) / _GB
    except Exception:  # noqa: BLE001
        pass
    return out


def reset_phase_peak(device) -> None:
    """Start a fresh peak window on this rank. Pairs with :func:`phase_peak_metrics`.

    ``max_memory_allocated`` is a high-water mark since the PROCESS started, and
    nothing in these arms ever resets it. That is the right choice for
    ``per_rank_memory_metrics`` -- a ratchet is what you want when the question is
    "did this run ever come close to the card" -- but it makes the number useless
    for the question that actually decides a knob: how much does ONE phase need?

    On this stack the two differ by more than a little. The rollout holds a vLLM
    pool sized to 0.6 of the card plus a hidden-state cache, and the actor update
    runs after that pool is asleep. Read without a reset, the update's "peak" is
    the rollout's, and every headroom argument built on it is about a phase that
    was not running. The intent lock's note that gradient checkpointing could not
    be turned off because step 1 reached 93.9 GiB is exactly that reading, and it
    cannot be told apart from a real update peak without this.

    THE LIFETIME MARKS ARE PRESERVED ACROSS THIS, and that is not incidental.
    ``per_rank_memory_metrics`` reads the same counters, and it means the ratchet:
    "did this run ever come close to the card". Resetting in front of it silently
    turned that key into the update phase's number under the run-long name -- with
    the ratchet gone, the series could even go DOWN between steps, which is the
    one thing a high-water mark cannot do. So the pre-reset values are folded into
    a process-level running max here and taken back into account there.

    A no-op on a backend that cannot reset, so a diagnostic never takes a run down.
    """
    try:
        _LIFETIME[0] = max(_LIFETIME[0], device.max_memory_allocated())
        _LIFETIME[1] = max(_LIFETIME[1], device.max_memory_reserved())
        device.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - a measurement must not break what it measures
        pass


def phase_peak_metrics(device, prefix: str, group=None) -> dict:
    """Cross-rank max of the peak since the last :func:`reset_phase_peak`.

    One key, not the per-rank spread ``per_rank_memory_metrics`` reports: the
    spread question is about the whole step and is already answered there, while
    this one is read against the card's capacity, where the max is the only
    number that matters.
    """
    allocated = device.max_memory_allocated() / _GB
    reserved = device.max_memory_reserved() / _GB
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {f"{prefix}_allocated_gb": allocated, f"{prefix}_reserved_gb": reserved}
    local = torch.tensor([allocated, reserved], dtype=torch.float64, device=device.current_device())
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX, group=group)
    rows = local.tolist()
    return {f"{prefix}_allocated_gb": rows[0], f"{prefix}_reserved_gb": rows[1]}


def per_rank_memory_metrics(device, prefix: str = "perf") -> dict:
    """Peak allocated/reserved and allocator retries for every rank.

    ``device`` is the module returned by ``get_torch_device()``. The memory
    figures are high-water marks since the process started, not the current
    step's peak, so they only ever climb -- the spread between ranks is the
    informative part. The retry counter is cumulative for the same reason: what
    matters is whether it moves between steps.

    Falls back to this rank's own numbers when torch.distributed is not up, so
    single-process runs and tests keep working.
    """
    # max(counter, what was seen before the last reset). Without the second term
    # this reports whatever window reset_phase_peak last opened, under a name that
    # has meant the run's high-water mark since it was added.
    allocated = max(device.max_memory_allocated(), _LIFETIME[0]) / _GB
    reserved = max(device.max_memory_reserved(), _LIFETIME[1]) / _GB
    retries, ooms = _allocator_counters(device)
    pinned = pinned_host_bytes() / _GB

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {
            f"{prefix}/max_memory_allocated_gb": allocated,
            f"{prefix}/max_memory_reserved_gb": reserved,
            f"{prefix}/max_alloc_retries": retries,
            f"{prefix}/max_alloc_ooms": ooms,
            f"{prefix}/pinned_host_gb": pinned,
        }

    world_size = torch.distributed.get_world_size()
    local = torch.tensor([allocated, reserved, retries, ooms, pinned], dtype=torch.float64,
                         device=device.current_device())
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    rows = [g.tolist() for g in gathered]

    metrics = {}
    for rank, (alloc, res, retry, oom, pin) in enumerate(rows):
        metrics[f"{prefix}/memory_allocated_gb/rank{rank}"] = alloc
        metrics[f"{prefix}/memory_reserved_gb/rank{rank}"] = res
        metrics[f"{prefix}/alloc_retries/rank{rank}"] = retry
        metrics[f"{prefix}/alloc_ooms/rank{rank}"] = oom
        metrics[f"{prefix}/pinned_host_gb/rank{rank}"] = pin
    allocs = [r[0] for r in rows]
    reserveds = [r[1] for r in rows]
    metrics[f"{prefix}/max_alloc_retries"] = max(r[2] for r in rows)
    metrics[f"{prefix}/max_alloc_ooms"] = max(r[3] for r in rows)
    # SUMMED, not maxed, and that is the point: page-locked memory is charged to
    # the NODE, and the ranks are separate processes on one box. The max would say
    # "one rank holds 16 GB" where what kills the run is that two of them do.
    metrics[f"{prefix}/pinned_host_gb"] = sum(r[4] for r in rows)
    # Named so reduce_metrics' key convention ("max" -> np.max, "min" -> np.min)
    # is right rather than merely harmless: these already ARE the cross-rank
    # extremes, and reducing a scalar leaves them alone.
    metrics[f"{prefix}/max_memory_allocated_gb"] = max(allocs)
    metrics[f"{prefix}/min_memory_allocated_gb"] = min(allocs)
    metrics[f"{prefix}/max_memory_reserved_gb"] = max(reserveds)
    metrics[f"{prefix}/min_memory_reserved_gb"] = min(reserveds)
    metrics[f"{prefix}/memory_allocated_spread_gb"] = max(allocs) - min(allocs)
    return metrics


def device_footprint_gb(device) -> float:
    """Device memory in use right now, the way nvidia-smi counts it.

    ``max_memory_allocated`` is what the model asked for and
    ``max_memory_reserved`` is a counter that can outlive the pages behind it
    (vLLM's CuMemAllocator unmaps out of band, which is how a 48 GiB card
    reports 66 GiB reserved). Neither answers "how much of the card is gone".
    ``mem_get_info`` does: it is the driver's own free/total, so it counts the
    CUDA context, every allocator's pool, and any other process on the device.

    Differencing it across a build is how to price a component that this arm
    pays for and never uses.
    """
    free, total = device.mem_get_info()
    return (total - free) / _GB
