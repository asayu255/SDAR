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
In this test, we instantiate a data parallel worker with 8 GPUs
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tensordict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from codetiming import Timer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import distributed as dist

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base import Worker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ray_utils import parallel_put


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `DummyWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DummyWorker(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.init_process_group()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    # [EXPLAIN] `do_nothing` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def do_nothing(self, data):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in data.batch.keys():
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            data.batch[key] += 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensordict.__version__ >= "0.5.0":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data.batch = data.batch.consolidate()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data


# [EXPLAIN] `test_data_transfer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_data_transfer():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init()
    # construct resource pool
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool([8])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_with_init = RayClassWithInitArgs(cls=DummyWorker)
    # construct worker group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = RayWorkerGroup(resource_pool, cls_with_init)

    # this is real dataset size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 4096
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 32768

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_dict = {}

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(2):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data_dict[str(i)] = torch.randint(0, 10000, (batch_size, seqlen))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors=data_dict)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(data)

    # we manually split data here and send to each worker
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_list = data.chunk(wg.world_size)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(wg.world_size):
        # consolidate is necessary
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensordict.__version__ >= "0.5.0":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data_list[i].batch = data_list[i].batch.consolidate()

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name="ray.pickle", initial_text=True):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(wg.world_size):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.cloudpickle.pickle.dumps(data_list[i])

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name="raw.pickle", initial_text=True):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import pickle

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(wg.world_size):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            pickle.dumps(data_list[i])

    # we put in advance
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name="put", initial_text=True):
        # takes around 40 seconds
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_list_ref = parallel_put(data_list)
        # for i in range(wg.world_size):
        #     data_list[i] = ray.put(data_list[i])

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name="launch", initial_text=True):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_ref = wg.do_nothing(data_list_ref)

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with Timer(name="get", initial_text=True):
        # takes around 40 seconds
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_lst = ray.get(output_ref)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for input_data, output_data in zip(data_list, output_lst):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in input_data.batch.keys():
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert torch.all(torch.eq(input_data.batch[key] + 1, output_data.batch[key])), (
                input_data.batch[key],
                output_data.batch[key],
                key,
            )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()
