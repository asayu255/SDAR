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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
e2e test verl.single_controller.ray
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, Execute, collect_all_to_all, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.worker import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# [EXPLAIN] `two_to_all_dispatch_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(arg) == 2
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size - 2):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            arg.append(arg[i % 2])
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(v) == 2
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size - 2):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            v.append(v[i % 2])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args, kwargs


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `TestActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, x) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._x = x

    # [EXPLAIN] `foo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def foo(self, y):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._x + y

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    # [EXPLAIN] `foo_rank_zero` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def foo_rank_zero(self, x, y):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._x + y + x

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(Dispatch.ONE_TO_ALL, blocking=False)
    # [EXPLAIN] `foo_one_to_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def foo_one_to_all(self, x, y):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._x + y + x

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(Dispatch.ALL_TO_ALL, blocking=False)
    # [EXPLAIN] `foo_all_to_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def foo_all_to_all(self, x, y):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._x + y + x

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
    # [EXPLAIN] `foo_custom` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def foo_custom(self, x, y):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._x + y + x


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_gpus=0.1)
# [EXPLAIN] `remote_call_wg` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remote_call_wg(worker_names):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = RayWorkerGroup.from_detached(worker_names=worker_names, ray_cls_with_init=class_with_args, name_prefix=None)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(worker_group.worker_names)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert output_ref == [8, 10, 8, 10]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_ref = worker_group.foo_rank_zero(x=1, y=2)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert output_ref == 5

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return worker_group.worker_names


# [EXPLAIN] `add_one` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def add_one(data):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = data.to("cuda")
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    data += 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = data.to("cpu")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data


# [EXPLAIN] `test_basics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_basics():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([4], use_gpu=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(worker_group.worker_names)

    # this will wait for all the results
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = worker_group.execute_all_sync("foo", y=3)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert output == [5, 5, 5, 5]

    # this is a list of object reference. It won't block.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_ref = worker_group.execute_all_async("foo", y=4)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(output_ref)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ray.get(output_ref) == [6, 6, 6, 6]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_ref = worker_group.foo_one_to_all(x=1, y=2)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ray.get(output_ref) == [5, 5, 5, 5]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_ref = worker_group.foo_all_to_all(x=[1, 2, 3, 4], y=[5, 6, 7, 8])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert ray.get(output_ref) == [8, 10, 12, 14]

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(ray.get(remote_call_wg.remote(worker_group.worker_names)))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = worker_group.execute_func_rank_zero(add_one, torch.ones(2, 2))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(output, torch.ones(2, 2) + 1)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_basics()
