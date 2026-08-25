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
"""Per-step deltas of the three counters that separate the deep-dip hypotheses.

The rare deep GPU dips -- one card (sometimes all three) near zero for under a
second, roughly once per half hour -- sit inside micro-batches, below the stall
watch's outlier threshold and too rare for a fixed profiler window to catch.
But each of the three candidate mechanisms increments a counter the process
already keeps, so the cheap instrument is to log the per-step delta of each and
read which one moved on the step that dipped:

  ``cuda_mallocs``   the caching allocator called cudaMalloc for a new segment.
                     cudaMalloc synchronizes the device, so a multi-hundred-MB
                     segment (a record-sized micro-batch growing the activation
                     pool) is a mid-micro-batch stall on exactly the rank whose
                     pool grew -- solo when one rank drew the record, several
                     ranks at once when the balanced batch grew all their peaks
                     together. From ``memory_stats()["num_device_alloc"]``.
  ``gc_gen2``        a Python generation-2 collection ran. It freezes the host,
                     the launch queue drains, and the device sits empty until
                     the sweep ends -- per-rank independent timing, so solo.
  ``dynamo_graphs``  torch.compile compiled a graph this step (a guard miss on
                     a shape it had not seen). Host-bound for the duration;
                     frequency should decay over the run as the cache fills,
                     which is itself a fingerprint the other two lack.

None of these says how LONG the stall was -- the 0.2 s NVML sampler still owns
duration and phase -- but a dip on a step where exactly one counter moved is an
attribution, and a dip on a step where none moved rules all three out at once.

Deltas are against the previous call on the same rank, so the numbers are
"events during this step". The first call after import reports zeros. Gathered
across ranks the same way ``per_rank_memory_metrics`` is, and for the same
reason: ``DataProto.concat`` keeps only rank 0's ``meta_info``, so anything not
all-gathered here never reaches the logger from the other ranks.
"""

import gc

import torch

__all__ = ["per_rank_stall_counter_metrics"]

_NAMES = ("cuda_mallocs", "gc_gen2", "dynamo_graphs")
_prev = None


def _read_counters(device) -> list:
    """Current absolute values, best effort. A diagnostic must never take the
    run down, and none of these counters is guaranteed to exist on every torch
    build -- a missing one just stays zero and its deltas read as 'never moved'.
    """
    mallocs = 0.0
    try:
        mallocs = float(device.memory_stats().get("num_device_alloc", 0))
    except Exception:
        pass
    gen2 = 0.0
    try:
        gen2 = float(gc.get_stats()[-1].get("collections", 0))
    except Exception:
        pass
    graphs = 0.0
    try:
        from torch._dynamo.utils import counters as dynamo_counters

        graphs = float(dynamo_counters["stats"].get("unique_graphs", 0))
    except Exception:
        pass
    return [mallocs, gen2, graphs]


def deltas(current, previous) -> list:
    """Events since the previous reading.

    Clamped at zero because two of the sources can go backwards without meaning
    anything happened: ``torch._dynamo.reset()`` clears its counters, and a
    negative "events this step" would read as data corruption in a chart.
    """
    if previous is None:
        return [0.0] * len(current)
    return [max(0.0, c - p) for c, p in zip(current, previous)]


def per_rank_stall_counter_metrics(device, prefix: str = "stall") -> dict:
    """One row per rank of {cudaMalloc, gen-2 GC, dynamo graph} events this step.

    ``device`` is the module returned by ``get_torch_device()``. Falls back to
    this rank's own numbers when torch.distributed is not up, so single-process
    runs and tests keep working.
    """
    global _prev
    current = _read_counters(device)
    local = deltas(current, _prev)
    _prev = current

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {f"{prefix}/{name}": value for name, value in zip(_NAMES, local)}

    world_size = torch.distributed.get_world_size()
    tensor = torch.tensor(local, dtype=torch.float64, device=device.current_device())
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    rows = [g.tolist() for g in gathered]

    metrics = {}
    for rank, row in enumerate(rows):
        for name, value in zip(_NAMES, row):
            metrics[f"{prefix}/{name}/rank{rank}"] = value
    for i, name in enumerate(_NAMES):
        # "max" so reduce_metrics' key convention reduces it to itself.
        metrics[f"{prefix}/max_{name}"] = max(row[i] for row in rows)
    return metrics
