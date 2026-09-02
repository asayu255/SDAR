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
"""Memory reported per rank, and reduced without lying about it.

nvidia-smi on a three-card run showed 46.9 / 30.1 / 35.0 GiB while wandb showed
one flat "perf/max_memory_reserved_gb". Both were right: DataProto.concat keeps
meta_info from the first worker, so the logged number was rank 0's, and the name
made it look like a cross-rank max. These pin the shape of the replacement.
"""

import pytest
import torch

from verl.utils.metric import reduce_metrics
from verl.utils.metric.memory import (
    device_footprint_gb,
    per_rank_memory_metrics,
    phase_peak_metrics,
    reset_phase_peak,
)

_GB = 1024.0**3


class _Device:
    """The get_torch_device() surface this helper uses."""

    def __init__(self, allocated_gb, reserved_gb, retries=0, ooms=0, stats=True, resettable=True):
        self._a, self._r = allocated_gb * _GB, reserved_gb * _GB
        self._stats = {"num_alloc_retries": retries, "num_ooms": ooms} if stats else None
        self._resettable = resettable
        self.resets = 0

    def reset_peak_memory_stats(self):
        if not self._resettable:
            raise RuntimeError("this backend cannot reset peak stats")
        self.resets += 1

    def max_memory_allocated(self):
        return self._a

    def max_memory_reserved(self):
        return self._r

    def memory_stats(self):
        if self._stats is None:
            raise RuntimeError("no memory_stats on this backend")
        return self._stats

    def current_device(self):
        return "cpu"


def test_without_distributed_it_reports_this_ranks_own_numbers():
    """Single-process runs and tests still get something sensible."""
    m = per_rank_memory_metrics(_Device(39.877, 45.5))

    assert m["perf/max_memory_allocated_gb"] == pytest.approx(39.877)
    assert m["perf/max_memory_reserved_gb"] == pytest.approx(45.5)
    assert not [k for k in m if "rank" in k]     # no per-rank keys to mislead


def _fake_all_gather(per_rank):
    """Stand in for torch.distributed with a fixed set of per-rank readings.

    Each entry is (allocated, reserved) or (allocated, reserved, retries, ooms).
    """

    def all_gather(out_list, local, *a, **kw):
        for tensor, row in zip(out_list, per_rank):
            row = tuple(row) + (0.0, 0.0)
            tensor.copy_(torch.tensor(row[:4], dtype=tensor.dtype))

    return all_gather


@pytest.fixture
def dist3(monkeypatch):
    def _apply(per_rank):
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(per_rank))
        monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather(per_rank))

    return _apply


def test_each_rank_gets_its_own_key(dist3):
    """The three cards in the report that started this: 45.8 / 29.4 / 34.2."""
    dist3([(39.9, 45.8), (24.1, 29.4), (28.6, 34.2)])
    m = per_rank_memory_metrics(_Device(39.9, 45.8))

    assert m["perf/memory_reserved_gb/rank0"] == pytest.approx(45.8)
    assert m["perf/memory_reserved_gb/rank1"] == pytest.approx(29.4)
    assert m["perf/memory_reserved_gb/rank2"] == pytest.approx(34.2)
    assert m["perf/memory_allocated_gb/rank1"] == pytest.approx(24.1)


def test_the_max_is_actually_a_max_across_ranks_now(dist3):
    """The old code put THIS rank's number under the name "max". If rank 0 is
    not the hungriest rank, that under-reports the run's real ceiling."""
    dist3([(24.1, 29.4), (39.9, 45.8), (28.6, 34.2)])
    m = per_rank_memory_metrics(_Device(24.1, 29.4))   # rank 0 is the SMALLEST

    assert m["perf/max_memory_allocated_gb"] == pytest.approx(39.9)
    assert m["perf/max_memory_reserved_gb"] == pytest.approx(45.8)
    assert m["perf/min_memory_allocated_gb"] == pytest.approx(24.1)


def test_the_spread_is_reported_because_the_spread_is_the_finding(dist3):
    """Parameters are replicated and the batch is balanced by rows AND tokens,
    so the ranks are supposed to agree. A spread means the activation peak
    diverged, which is the thing worth an alert."""
    dist3([(39.9, 45.8), (24.1, 29.4), (28.6, 34.2)])
    m = per_rank_memory_metrics(_Device(39.9, 45.8))

    assert m["perf/memory_allocated_spread_gb"] == pytest.approx(39.9 - 24.1)


def test_balanced_ranks_report_no_spread(dist3):
    dist3([(30.0, 36.0)] * 3)
    m = per_rank_memory_metrics(_Device(30.0, 36.0))

    assert m["perf/memory_allocated_spread_gb"] == pytest.approx(0.0)
    assert m["perf/max_memory_allocated_gb"] == m["perf/min_memory_allocated_gb"]


def test_reduce_metrics_leaves_the_already_reduced_values_alone(dist3):
    """These reach the logger through reduce_metrics, whose key convention picks
    np.max for "max" and np.min for "min". They are already the cross-rank
    extremes, so the reduction has to be a no-op rather than a second opinion."""
    dist3([(39.9, 45.8), (24.1, 29.4), (28.6, 34.2)])
    m = per_rank_memory_metrics(_Device(39.9, 45.8))

    reduced = reduce_metrics(dict(m))

    assert reduced["perf/max_memory_allocated_gb"] == pytest.approx(39.9)
    assert reduced["perf/min_memory_allocated_gb"] == pytest.approx(24.1)
    assert reduced["perf/memory_reserved_gb/rank2"] == pytest.approx(34.2)
    assert reduced["perf/memory_allocated_spread_gb"] == pytest.approx(15.8)


def test_a_single_gpu_run_still_produces_the_rank0_key(dist3):
    dist3([(30.0, 36.0)])
    m = per_rank_memory_metrics(_Device(30.0, 36.0))

    assert m["perf/memory_allocated_gb/rank0"] == pytest.approx(30.0)
    assert m["perf/memory_allocated_spread_gb"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The allocator's own counters
# --------------------------------------------------------------------------- #
def test_alloc_retries_are_reported_per_rank(dist3):
    """num_alloc_retries counts the times a malloc missed the cache and the
    allocator had to cudaFree and retry. cudaFree synchronizes the device, so
    each retry is a queue drain in the middle of a micro-batch -- exactly the
    thing an NVML sampler sees as "sm 0 at full power" and cannot name."""
    dist3([(39.9, 45.8, 12, 0), (24.1, 29.4, 0, 0), (28.6, 34.2, 3, 0)])
    m = per_rank_memory_metrics(_Device(39.9, 45.8, retries=12))

    assert m["perf/alloc_retries/rank0"] == pytest.approx(12)
    assert m["perf/alloc_retries/rank1"] == pytest.approx(0)
    assert m["perf/alloc_retries/rank2"] == pytest.approx(3)
    assert m["perf/max_alloc_retries"] == pytest.approx(12)
    assert m["perf/max_alloc_ooms"] == pytest.approx(0)


def test_a_clean_run_reports_zero_retries(dist3):
    """Zero is the finding that kills the hypothesis, so it has to be logged
    rather than absent."""
    dist3([(30.0, 36.0, 0, 0)] * 3)
    m = per_rank_memory_metrics(_Device(30.0, 36.0))

    assert m["perf/max_alloc_retries"] == 0
    assert all(m[f"perf/alloc_retries/rank{i}"] == 0 for i in range(3))


def test_a_backend_without_memory_stats_does_not_break_the_step():
    """A diagnostic must never be able to fail a training step."""
    m = per_rank_memory_metrics(_Device(30.0, 36.0, stats=False))

    assert m["perf/max_alloc_retries"] == 0
    assert m["perf/max_memory_allocated_gb"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# What the card actually lost
# --------------------------------------------------------------------------- #
def test_device_footprint_is_used_not_free():
    """mem_get_info returns (free, total); the useful number is the difference.

    Getting this backwards would price a component as its own negation, which is
    exactly the kind of thing that reads plausible in a log line.
    """
    class _Dev:
        def mem_get_info(self):
            return (8 * _GB, 48 * _GB)      # 8 free of 48

    assert device_footprint_gb(_Dev()) == pytest.approx(40.0)


def test_device_footprint_counts_every_process_not_just_this_one():
    """It is the driver's own accounting, so it includes the CUDA context and
    any allocator's pool -- which is the whole point when pricing vLLM, whose
    memory never appears in this process's max_memory_allocated."""
    class _Dev:
        def __init__(self, used):
            self.used = used

        def mem_get_info(self):
            return ((48 - self.used) * _GB, 48 * _GB)

    before, after = device_footprint_gb(_Dev(6.0)), device_footprint_gb(_Dev(14.5))

    assert after - before == pytest.approx(8.5)     # what building it cost


# --------------------------------------------------------------------------- #
# Both of these all_gather, so both are COLLECTIVES: every rank has to reach
# them, in the same order, or the run hangs rather than fails. In update_actor
# they sit after the loss and before the metrics go back, on a path with no
# rank-conditional branch and no early return. A later `if self.rank == 0:`
# around either one would deadlock three GPUs with no traceback, so the shape
# of that call site is pinned here.
# --------------------------------------------------------------------------- #


def _update_actor_source():
    import inspect
    import textwrap

    import verl.workers.fsdp_workers as fsdp_workers

    return textwrap.dedent(inspect.getsource(fsdp_workers.ActorRolloutRefWorker.update_actor))


def test_both_gathers_are_reached_unconditionally():
    src = _update_actor_source()
    body = src.split("\n")
    calls = [
        i for i, line in enumerate(body)
        if "per_rank_memory_metrics(" in line or "per_rank_stall_counter_metrics(" in line
    ]
    assert len(calls) == 2, f"expected both gathers in update_actor, found {len(calls)}"

    for i in calls:
        indent = len(body[i]) - len(body[i].lstrip())
        # Walk out to column 0, checking nothing on the way is a rank test or a
        # conditional the other ranks could take differently.
        for j in range(i - 1, -1, -1):
            line = body[j]
            if not line.strip():
                continue
            here = len(line) - len(line.lstrip())
            if here >= indent:
                continue
            indent = here
            opener = line.strip()
            assert not opener.startswith(("if ", "elif ", "try:", "except", "for ", "while ")), (
                f"{body[i].strip()} is inside `{opener}` -- a collective under a "
                "conditional hangs the ranks that do not take it"
            )
            if opener.startswith("def "):
                break


def test_neither_gather_sits_after_a_return():
    """An early return past a collective is the same deadlock, arrived at from
    the other side."""
    src = _update_actor_source()
    body = src.split("\n")
    first_gather = min(
        i for i, line in enumerate(body)
        if "per_rank_memory_metrics(" in line or "per_rank_stall_counter_metrics(" in line
    )
    before = [ln.strip() for ln in body[:first_gather]]
    assert not any(ln.startswith("return") for ln in before), (
        "update_actor returns before the collectives; whichever rank takes that "
        "path leaves the others waiting in all_gather"
    )


# ------------------------------------------------------- the per-phase window
def test_the_phase_window_reports_the_peak_since_its_reset():
    """perf/max_memory_* is a process-lifetime ratchet, which on this stack means
    "the rollout's vLLM pool" whichever phase is asking. The update's own peak is
    a different number and it is the one the checkpointing and micro-batch
    decisions turn on."""
    device = _Device(62.5, 71.0)
    reset_phase_peak(device)
    assert device.resets == 1

    m = phase_peak_metrics(device, "perf/update_peak")
    assert m["perf/update_peak_allocated_gb"] == pytest.approx(62.5)
    assert m["perf/update_peak_reserved_gb"] == pytest.approx(71.0)


def test_a_backend_that_cannot_reset_is_not_a_failed_run():
    """A measurement must never take down what it measures."""
    reset_phase_peak(_Device(1.0, 2.0, resettable=False))


def test_the_phase_window_does_not_emit_per_rank_keys():
    """The spread question is the whole step's and per_rank_memory_metrics
    already answers it; this one is read against the card's capacity, where the
    cross-rank max is the only number that matters."""
    m = phase_peak_metrics(_Device(10.0, 12.0), "perf/update_peak")
    assert set(m) == {"perf/update_peak_allocated_gb", "perf/update_peak_reserved_gb"}
