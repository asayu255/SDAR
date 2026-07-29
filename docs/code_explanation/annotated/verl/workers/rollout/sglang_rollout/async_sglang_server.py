# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.requests import Request
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.responses import JSONResponse

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.async_server import AsyncServerBase

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `AsyncSglangServer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncSglangServer(AsyncServerBase):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: DictConfig, dp_size: int, dp_rank: int, wg_prefix: str):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout_config = config.get("rollout", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._tp_size = rollout_config.get("tensor_model_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._dp_size = dp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._dp_rank = dp_rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.wg_prefix = wg_prefix
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers = []

    # [EXPLAIN] `init_engine` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def init_engine(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched_actors = [actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")]

        # TODO support multi node
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for matched_actor in matched_actors:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            current_rank = int(matched_actor["name"].split(":")[-1])

            # send to all works in this tp group, because sglang is SPMD
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if current_rank >= self._dp_rank * self._tp_size and current_rank < (self._dp_rank + 1) * self._tp_size:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.workers.append(ray.get_actor(**matched_actor))

    # [EXPLAIN] `chat_completion` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def chat_completion(self, raw_request: Request):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request = await raw_request.json()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_dp_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_future = worker.execute_method.remote("chat_completion", request)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_dp_lst.append(output_future)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = await asyncio.gather(*output_dp_lst)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for output in outputs:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if output is not None:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return JSONResponse(output)
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError("AsyncSglangServer No output from workers self._dp_rank: {self._dp_rank}, self._tp_size: {self._tp_size}, self.workers: {self.workers}")

    # [EXPLAIN] `wake_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def wake_up(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            worker.resume.remote()

    # [EXPLAIN] `sleep` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def sleep(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            worker.offload.remote()
