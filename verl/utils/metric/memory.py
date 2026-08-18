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
exactly that. The caching allocator never gives the peak back either
(``empty_cache`` is only called on the generation path, which the off-policy
arms never take), so ``nvidia-smi`` shows a ratchet: whatever a rank once
reached, it keeps.

One all-gather of two floats per step is what makes any of that visible.
"""

import torch

__all__ = ["per_rank_memory_metrics"]

_GB = 1024.0**3


def per_rank_memory_metrics(device, prefix: str = "perf") -> dict:
    """Peak allocated/reserved for every rank, plus the spread across them.

    ``device`` is the module returned by ``get_torch_device()``. Both figures are
    high-water marks since the process started, not the current step's peak, so
    they only ever climb -- the spread between ranks is the informative part.

    Falls back to this rank's own numbers when torch.distributed is not up, so
    single-process runs and tests keep working.
    """
    allocated = device.max_memory_allocated() / _GB
    reserved = device.max_memory_reserved() / _GB

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {
            f"{prefix}/max_memory_allocated_gb": allocated,
            f"{prefix}/max_memory_reserved_gb": reserved,
        }

    world_size = torch.distributed.get_world_size()
    local = torch.tensor([allocated, reserved], dtype=torch.float64, device=device.current_device())
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    pairs = [g.tolist() for g in gathered]

    metrics = {}
    for rank, (alloc, res) in enumerate(pairs):
        metrics[f"{prefix}/memory_allocated_gb/rank{rank}"] = alloc
        metrics[f"{prefix}/memory_reserved_gb/rank{rank}"] = res
    allocs = [a for a, _ in pairs]
    reserveds = [r for _, r in pairs]
    # Named so reduce_metrics' key convention ("max" -> np.max, "min" -> np.min)
    # is right rather than merely harmless: these already ARE the cross-rank
    # extremes, and reducing a scalar leaves them alone.
    metrics[f"{prefix}/max_memory_allocated_gb"] = max(allocs)
    metrics[f"{prefix}/min_memory_allocated_gb"] = min(allocs)
    metrics[f"{prefix}/max_memory_reserved_gb"] = max(reserveds)
    metrics[f"{prefix}/min_memory_reserved_gb"] = min(reserveds)
    metrics[f"{prefix}/memory_allocated_spread_gb"] = max(allocs) - min(allocs)
    return metrics
