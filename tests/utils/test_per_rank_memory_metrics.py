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
from verl.utils.metric.memory import per_rank_memory_metrics

_GB = 1024.0**3


class _Device:
    """The get_torch_device() surface this helper uses."""

    def __init__(self, allocated_gb, reserved_gb):
        self._a, self._r = allocated_gb * _GB, reserved_gb * _GB

    def max_memory_allocated(self):
        return self._a

    def max_memory_reserved(self):
        return self._r

    def current_device(self):
        return "cpu"


def test_without_distributed_it_reports_this_ranks_own_numbers():
    """Single-process runs and tests still get something sensible."""
    m = per_rank_memory_metrics(_Device(39.877, 45.5))

    assert m["perf/max_memory_allocated_gb"] == pytest.approx(39.877)
    assert m["perf/max_memory_reserved_gb"] == pytest.approx(45.5)
    assert not [k for k in m if "rank" in k]     # no per-rank keys to mislead


def _fake_all_gather(per_rank):
    """Stand in for torch.distributed with a fixed set of per-rank readings."""

    def all_gather(out_list, local, *a, **kw):
        for tensor, (alloc, res) in zip(out_list, per_rank):
            tensor.copy_(torch.tensor([alloc, res], dtype=tensor.dtype))

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
