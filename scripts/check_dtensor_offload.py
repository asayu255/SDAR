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
"""Can a DTensor's device-to-host copy be overlapped like the optimizer's?

The optimizer state is plain tensors, so the checkpoint save already snapshots
it on the main stream and copies it to pinned host memory on a side stream --
0.00 s on the training thread. The model state dict is DTensors, and those
still go through FSDP's own offload_to_cpu: 3.2-3.8 s per save, now the entire
remaining cost (run ovb0yobz).

Routing them the same way needs three things to hold, and one earlier attempt
to guess at them cost a crashed run, so this asks the GPU instead of guessing:

  1. torch.empty_like(dt) gives a DTensor sharing the placement -- the snapshot.
  2. snapshot.copy_(live) works DTensor-to-DTensor on the main stream.
  3. a DTensor with a CPU local shard can be rebuilt on a CUDA mesh, and
     torch.save/torch.load round-trips it identically to what FSDP produces.

(3) is the doubtful one, and it is the one that decides the design: FSDP's own
offload_to_cpu produces exactly such an object, so it is representable -- what
is unknown is whether DTensor.from_local will build it without a collective or
a device check.

Run on the GPU box, one process, no training:

    torchrun --nproc_per_node=1 scripts/check_dtensor_offload.py

Prints PASS/FAIL per step; the first FAIL is the answer.
"""

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device; run this on the training box")
        return 1

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    live = distribute_tensor(torch.arange(64, dtype=torch.float32), mesh, [Shard(0)])
    print(f"[rank {rank}] live: {type(live).__name__} on {live.device}, "
          f"local {tuple(live.to_local().shape)} on {live.to_local().device}")

    # 1. snapshot buffer
    try:
        snapshot = torch.empty_like(live)
        ok = isinstance(snapshot, DTensor) and snapshot.to_local().device.type == "cuda"
        print(f"[rank {rank}] 1 empty_like keeps DTensor on device: {'PASS' if ok else 'FAIL'} "
              f"({type(snapshot).__name__})")
        if not ok:
            return 1
    except Exception as exc:
        print(f"[rank {rank}] 1 empty_like: FAIL {exc!r}")
        return 1

    # 2. D2D copy on the main stream
    try:
        snapshot.copy_(live)
        same = torch.equal(snapshot.to_local(), live.to_local())
        print(f"[rank {rank}] 2 snapshot.copy_(live) D2D: {'PASS' if same else 'FAIL (values differ)'}")
        if not same:
            return 1
    except Exception as exc:
        print(f"[rank {rank}] 2 snapshot.copy_(live): FAIL {exc!r}")
        return 1

    # 3. rebuild a DTensor around a pinned CPU local shard, and round-trip it
    try:
        local = snapshot.to_local()
        host = torch.empty(local.shape, dtype=local.dtype, device="cpu", pin_memory=True)
        host.copy_(local)                       # this is the copy that would move to the side stream
        rebuilt = DTensor.from_local(host, mesh, snapshot.placements, run_check=False)
        print(f"[rank {rank}] 3a from_local(cpu shard) on a cuda mesh: PASS "
              f"(local on {rebuilt.to_local().device})")

        path = f"/tmp/dtensor_probe_rank{rank}.pt"
        torch.save({"w": rebuilt}, path)
        loaded = torch.load(path, weights_only=False)["w"]
        matches = torch.equal(loaded.to_local().cpu(), live.to_local().cpu())
        print(f"[rank {rank}] 3b torch.save/load round-trip: {'PASS' if matches else 'FAIL (values differ)'} "
              f"({type(loaded).__name__})")
        os.remove(path)
        if not matches:
            return 1
    except Exception as exc:
        print(f"[rank {rank}] 3 from_local / round-trip: FAIL {exc!r}")
        print(f"[rank {rank}]    trying the in-place fallback instead...")
        return _try_in_place(rank, live, snapshot)

    print(f"[rank {rank}] ALL PASS -- the model's offload can be overlapped like the optimizer's")
    return 0


def _try_in_place(rank, live, snapshot):
    """Fallback: swap the DTensor's local shard for a pinned host one, in place.

    The real walk's ShardedTensor branch already mutates ``shard.tensor`` this
    way, so if from_local refuses to build the object, assigning
    ``_local_tensor`` may still produce it -- and that is enough here, because
    the walk owns these state-dict entries and nothing else holds a reference.
    A private attribute, hence the probe: it either works on this torch build,
    or the model's offload stays with FSDP and the ~3.2 s stays with it.
    """
    try:
        local = snapshot.to_local()
        host = torch.empty(local.shape, dtype=local.dtype, device="cpu", pin_memory=True)
        host.copy_(local)
        snapshot._local_tensor = host
        print(f"[rank {rank}] 3c in-place _local_tensor swap: PASS "
              f"(local now on {snapshot.to_local().device})")

        path = f"/tmp/dtensor_probe_inplace_rank{rank}.pt"
        torch.save({"w": snapshot}, path)
        loaded = torch.load(path, weights_only=False)["w"]
        matches = torch.equal(loaded.to_local().cpu(), live.to_local().cpu())
        os.remove(path)
        print(f"[rank {rank}] 3d round-trip after the swap: "
              f"{'PASS' if matches else 'FAIL (values differ)'}")
        if matches:
            print(f"[rank {rank}] FALLBACK PASS -- overlap the model via the in-place swap")
            return 0
    except Exception as exc:
        print(f"[rank {rank}] 3c in-place swap: FAIL {exc!r}")

    print(f"[rank {rank}]    -> the model's D2H cannot be overlapped; FSDP's "
          "offload_to_cpu stays, and the ~3.2 s per save with it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
