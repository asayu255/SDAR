# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import ExitStack
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Optional, Tuple, TypeVar
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from uuid import uuid4

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray.actor

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base_tool import BaseTool
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .schemas import OpenAIFunctionToolSchema

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
# [EXPLAIN] `PoolMode` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class PoolMode(Enum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Execution pool mode enumeration."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ThreadMode = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ProcessMode = 2


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
# [EXPLAIN] `TokenBucketWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TokenBucketWorker:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Ray actor for rate limiting using token bucket algorithm."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, rate_limit: int):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit = rate_limit
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_count = 0  # For observability
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._semaphore = threading.Semaphore(rate_limit)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @ray.method(concurrency_group="acquire")
    # [EXPLAIN] `acquire` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def acquire(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Acquire a token from the bucket."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._semaphore.acquire()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_count += 1

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @ray.method(concurrency_group="release")
    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def release(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Release a token back to the bucket."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._semaphore.release()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_count -= 1

    # [EXPLAIN] `get_current_count` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_current_count(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get current number of acquired tokens."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.current_count


# [EXPLAIN] `SearchExecutionWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SearchExecutionWorker:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Worker for executing search operations with optional rate limiting."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    # [EXPLAIN] `_init_rate_limit` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_rate_limit(self, rate_limit):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize singleton rate limiter."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    # [EXPLAIN] `ping` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def ping(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Health check method."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute function with optional rate limiting."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rate_limit_worker:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with ExitStack() as stack:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                stack.callback(self.rate_limit_worker.release.remote)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                ray.get(self.rate_limit_worker.acquire.remote())
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return fn(*fn_args, **fn_kwargs)
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logger.warning(f"Error when executing search: {e}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return fn(*fn_args, **fn_kwargs)


# [EXPLAIN] `init_search_execution_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_search_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initialize search execution pool."""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mode == PoolMode.ThreadMode:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.remote(SearchExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Process mode is not implemented yet")


# [EXPLAIN] `SearchTool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SearchTool(BaseTool):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config, tool_schema)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._instance_dict = {}

        # Worker and rate limiting configuration
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_workers = config.get("num_workers", 120)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit = config.get("rate_limit", 120)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.timeout = config.get("timeout", 30)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.execution_pool = init_search_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)

        # Retrieval service configuration
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.retrieval_service_url = config.get("retrieval_service_url")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.topk = config.get("topk", 3)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.retrieval_service_url == "":
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("retrieval_service_url is not set")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.info(f"Initialized SearchTool with config: {config}")

    # [EXPLAIN] `get_openai_tool_schema` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Return the OpenAI tool schema."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.tool_schema

    # [EXPLAIN] `create` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if instance_id is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            instance_id = str(uuid4())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return instance_id

    # [EXPLAIN] `execute_search` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result_text, metadata

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        timeout = self.timeout
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query_list_from_params = parameters.get("query_list")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return json.dumps({"result": error_msg}), 0.0, {}

        # Execute search using Ray execution pool
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result_text, metadata = await self.execution_pool.execute.remote(self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout)

            # Store results in instance dictionary
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics = {"query_count": metadata.get("query_count", 0), "status": metadata.get("status", "unknown"), "total_results": metadata.get("total_results", 0), "api_request_error": metadata.get("api_request_error")}

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return result_text, 0.0, metrics

        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error(f"[SearchTool] Execution failed: {e}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return error_result, 0.0, {"error": str(e)}

    # [EXPLAIN] `calc_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._instance_dict[instance_id]["reward"]

    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def release(self, instance_id: str, **kwargs) -> None:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if instance_id in self._instance_dict:
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del self._instance_dict[instance_id]
