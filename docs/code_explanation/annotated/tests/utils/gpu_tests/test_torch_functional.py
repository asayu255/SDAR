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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.multiprocessing as mp

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import distributed_masked_mean, distributed_mean_max_min_std


# [EXPLAIN] `_worker_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.set_device(rank)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # each rank holds tensor [rank+1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean, gmax, gmin, gstd = distributed_mean_max_min_std(local, True, True, True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    values = [float(i + 1) for i in range(world_size)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exp_mean = sum(values) / len(values)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exp_max = max(values)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exp_min = min(values)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    var = sum((x - exp_mean) ** 2 for x in values) / (len(values) - 1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exp_std = var**0.5

    # all ranks should see the same result
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(mean.cpu(), torch.tensor(exp_mean)), f"mean@{rank}"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(gmax.cpu(), torch.tensor(exp_max)), f"max@{rank}"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(gmin.cpu(), torch.tensor(exp_min)), f"min@{rank}"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(gstd.cpu(), torch.tensor(exp_std)), f"std@{rank}"

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.destroy_process_group()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize("world_size", [2, 4])
# [EXPLAIN] `test_distributed_mean_max_min_std` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_distributed_mean_max_min_std(world_size, tmp_path):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rendezvous_file = str(tmp_path / "rdzv_mean")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mp.spawn(
        fn=_worker_mean,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


# [EXPLAIN] `_worker_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _worker_mask(rank: int, world_size: int, rendezvous_file: str):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.set_device(rank)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"cuda:{rank}")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rank == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = torch.tensor([1, 0], device=f"cuda:{rank}", dtype=torch.float32)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = torch.tensor([0, 1], device=f"cuda:{rank}", dtype=torch.float32)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gmean = distributed_masked_mean(local_tensor, mask)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    valid_values = [1.0] + [2 * i + 2.0 for i in range(1, world_size)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_mean = sum(valid_values) / len(valid_values)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.allclose(gmean.cpu(), torch.tensor(expected_mean)), f"masked_mean@{rank}"

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.destroy_process_group()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize("world_size", [2, 4])
# [EXPLAIN] `test_distributed_masked_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_distributed_masked_mean(world_size, tmp_path):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rendezvous_file = str(tmp_path / "rdzv_mask")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mp.spawn(
        fn=_worker_mask,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )
