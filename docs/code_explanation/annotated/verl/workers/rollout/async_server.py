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
import heapq
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import socket
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import asynccontextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Dict, List, Tuple, Type
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from uuid import uuid4

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import aiohttp
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import fastapi
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uvicorn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from cachetools import LRUCache
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from openai import AsyncOpenAI
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from openai.types.chat.chat_completion import ChatCompletion
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.requests import Request

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.base import RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_tokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)


# [EXPLAIN] `_get_free_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_free_port():
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with socket.socket() as sock:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sock.bind(("", 0))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sock.getsockname()[1]


# [EXPLAIN] `AsyncServerBase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncServerBase(ABC):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Base class for AsyncServer."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.address = ray._private.services.get_node_ip_address()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.port = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.server_ready = asyncio.Event()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        asyncio.create_task(self._start_fastapi_server())

    # [EXPLAIN] `_start_fastapi_server` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def _start_fastapi_server(self):
        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @asynccontextmanager
        # [EXPLAIN] `lifespan` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        async def lifespan(app: fastapi.FastAPI):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("FastAPI startup")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.server_ready.set()
            # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os._exit(-1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        app = fastapi.FastAPI(lifespan=lifespan)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.port = _get_free_port()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        server = uvicorn.Server(config)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await server.serve()

    # [EXPLAIN] `get_server_address` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def get_server_address(self) -> Tuple[str, int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get FastAPI server address."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await self.server_ready.wait()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"{self.address}:{self.port}"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `chat_completion` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def chat_completion(self, raw_request: Request):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """OpenAI chat completion API.

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `init_engine` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def init_engine(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Init async LLM engine."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `wake_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def wake_up(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Wake up engine to load model weights and build kv cache."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `sleep` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def sleep(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Sleep engine to offload model weights and discard kv cache."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `ChatCompletionScheduler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ChatCompletionScheduler:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_name = "/".join(model_path.split("/")[-2:])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = copy_to_local(model_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # Least requests load balancing
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weighted_addresses = [[0, address] for address in server_addresses]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

    # [EXPLAIN] `submit_chat_completions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Submit a chat completion request to the server with the least number of requests.

        Args:
            callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None], async callback function
                to handle the response. The callback function should have the following signature:

                ```python
                async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
                    ...
                ```
                - completions: chat completion response from server.
                - info: user provided `callback_additional_info`.
                - exception: exception raise from OpenAI client if request failed, otherwise None.

                **CAUTION**: the callback function must be async and non-blocking, if you have any blocking operation,
                please move to seperate thread or process pool to avoid blocking the event loop.

            callback_additional_info: Dict[str, Any], additional info to pass to the callback function.

            **chat_complete_request: dict, request parameters same as OpenAI AsyncCompletions.create.
                OpenAI API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "extra_headers" not in chat_complete_request:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            chat_complete_request["extra_headers"] = {}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_headers = chat_complete_request["extra_headers"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request_id = extra_headers.get("x-request-id", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if request_id:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if request_id.startswith("chatcmpl-"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                request_id = request_id[len("chatcmpl-") :]
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                extra_headers["x-request-id"] = request_id

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            address = self.request_id_to_address.pop(request_id)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            address = self.weighted_addresses[0][1]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.weighted_addresses[0][0] += 1
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request_id = uuid4().hex
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.request_id_to_address[request_id] = address
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        chat_complete_request["extra_headers"]["x-request-id"] = request_id

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        completions, exception = None, None
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            completions = await self._chat_completions_aiohttp(address, **chat_complete_request)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # Let user handle the exception
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            exception = e

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await callback(completions, callback_additional_info, exception)

    # [EXPLAIN] `_chat_completions_openai` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return await client.chat.completions.create(**chat_complete_request)

    # [EXPLAIN] `_chat_completions_aiohttp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            extra_headers = chat_complete_request.pop("extra_headers")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            timeout = aiohttp.ClientTimeout(total=None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            session = aiohttp.ClientSession(timeout=timeout)
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data = await resp.json()
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return ChatCompletion(**data)
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await session.close()

    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `AsyncLLMServerManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncLLMServerManager:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, *, scheduler_kwargs: Dict[str, Any] = None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize AsyncLLMServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
            scheduler_kwargs: Dict[str, Any], kwargs for chat scheduler.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.worker_group = worker_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        workers_info = ray.get(register_center.get_worker_info.remote())
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(workers_info) == self.worker_group.world_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.async_llm_servers = [None] * self.rollout_dp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.server_addresses = [None] * self.rollout_dp_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )

        # Start all server instances, restart if address already in use.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        unready_dp_ranks = set(range(self.rollout_dp_size))
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while len(unready_dp_ranks) > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for rollout_dp_rank, server in servers.items():
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    address = ray.get(server.get_server_address.remote())
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.server_addresses[rollout_dp_rank] = address
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.async_llm_servers[rollout_dp_rank] = server
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    unready_dp_ranks.remove(rollout_dp_rank)
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    ray.kill(server)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        # Init user provided chat scheduler in sperate thread.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler: ChatCompletionScheduler = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler_loop = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler_ready = threading.Event()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler_thread = threading.Thread(target=self._init_chat_scheduler, daemon=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.chat_scheduler_thread.start()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.chat_scheduler_ready.wait()

    # [EXPLAIN] `_init_chat_scheduler` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_chat_scheduler(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler_loop = asyncio.new_event_loop()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        asyncio.set_event_loop(self.chat_scheduler_loop)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module_path, class_name = self.config.rollout.chat_scheduler.rsplit(".", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = importlib.import_module(module_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        scheduler_cls = getattr(module, class_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_scheduler = scheduler_cls(
            config=self.config.rollout,
            model_path=self.config.model.path,
            server_addresses=self.server_addresses,
            **self.scheduler_kwargs,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.chat_scheduler_ready.set()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.chat_scheduler_loop.run_forever()

    # [EXPLAIN] `wake_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def wake_up(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Wake up all vllm instances."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    # [EXPLAIN] `sleep` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sleep(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Sleep all vllm instances."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    # [EXPLAIN] `submit_chat_completions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Submit a chat completion request to chat scheduler and wait until it is done.
        To submit multiple requests in parallel, please use `generate_sequences` instead.

        Args: same as ChatCompletionScheduler.submit_chat_completions.
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info=callback_additional_info,
                **chat_complete_request,
            ),
            self.chat_scheduler_loop,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        future.result()

    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Generate multiple sequences in parallel via chat scheduler."""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future = asyncio.run_coroutine_threadsafe(self.chat_scheduler.generate_sequences(prompts, **sampling_params), self.chat_scheduler_loop)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return future.result()


# [EXPLAIN] `async_server_class` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend, should be "vllm" or "sglang".

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rollout_backend == "vllm":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return AsyncvLLMServer
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif rollout_backend == "sglang":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return AsyncSglangServer
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError
