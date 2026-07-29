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
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto, DataProtoFuture
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.worker import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# Pytest fixture for Ray setup/teardown
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.fixture
# [EXPLAIN] `ray_init_shutdown` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def ray_init_shutdown():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(num_cpus=100)
    # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
    yield
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# Define a simple worker for testing
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `DecoratorTestWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DecoratorTestWorker(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, initial_value=0):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.value = initial_value
        # Simulate some setup if needed
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(0.1)  # Ensure worker init completes

    # Test method for synchronous DP compute (default behavior)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] `dp_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dp_compute(self, data: DataProto) -> DataProto:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(0.1)  # Simulate work
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["output"] = data.batch["input"] + self.value + rank_value
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # Test async def method with DP compute (default behavior)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    # [EXPLAIN] `async_dp_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def async_dp_compute(self, data: DataProto) -> DataProto:
        # Simulate async work
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await asyncio.sleep(0.1)  # Simulate async work
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["output_async"] = data.batch["input"] * 2 + self.value + rank_value
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data


# Test function for synchronous DP compute
# [EXPLAIN] `test_decorator_dp_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_decorator_dp_compute(ray_init_shutdown):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tests the default behavior of a synchronous decorated method with DP_COMPUTE_PROTO.
    Verifies the result correctness.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_workers = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)  # Use CPU for simplicity
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = RayWorkerGroup(resource_pool, cls_with_args, name_prefix=f"decorator_test_sync_dp_{int(time.time())}")

    # Prepare input data (size 4, for 2 workers)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_tensor = torch.arange(4, dtype=torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    # Call the decorated method
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = worker_group.dp_compute(data)

    # Assert the result correctness
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(output, DataProto), "Expected DataProto result"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "output" in output.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == len(data), "Output length should match input length"

    # Expected output calculation for DP_COMPUTE_PROTO with 2 workers
    # Worker 0 gets data[0:2], Worker 1 gets data[2:4]
    # Worker 0 adds initial_value(10) + rank(0) = 10
    # Worker 1 adds initial_value(10) + rank(1) = 11
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output_part1 = torch.tensor([0, 1], dtype=torch.float32) + 10 + 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output_part2 = torch.tensor([2, 3], dtype=torch.float32) + 10 + 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(output.batch["output"], expected_output, msg="Sync DP compute output data mismatch")


# Test function for async def method with DP compute
# [EXPLAIN] `test_decorator_async_function` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_decorator_async_function(ray_init_shutdown):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tests the decorator with an `async def` method using DP_COMPUTE_PROTO.
    Verifies that the call returns a future and the result is correct after .get().
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_workers = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=5)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = RayWorkerGroup(resource_pool, cls_with_args, name_prefix=f"decorator_test_async_dp_{int(time.time())}")

    # Prepare input data (size 4, for 2 workers)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_tensor = torch.arange(4, dtype=torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    # Call the async decorated method - this should return a future
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    future_output: DataProtoFuture = worker_group.async_dp_compute(data)

    # Assert that the call returned a future
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(future_output, DataProtoFuture), "Expected DataProtoFuture for async def call"

    # Get the result (this should block)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_data = future_output.get()

    # Assert the result correctness
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(result_data, DataProto)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "output_async" in result_data.batch.keys()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(result_data) == len(data), "Output length should match input length"

    # Expected output calculation for DP_COMPUTE_PROTO with 2 workers
    # Worker 0 gets data[0:2], Worker 1 gets data[2:4]
    # Worker 0 calculates: input * 2 + initial_value(5) + rank(0)
    # Worker 1 calculates: input * 2 + initial_value(5) + rank(1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output_part1 = (torch.tensor([0, 1], dtype=torch.float32) * 2) + 5 + 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output_part2 = (torch.tensor([2, 3], dtype=torch.float32) * 2) + 5 + 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(result_data.batch["output_async"], expected_output, msg="Async DP compute output data mismatch")
