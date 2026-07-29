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
import ray.util.multiprocessing

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.base_tool import BaseTool
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .schemas import OpenAIFunctionToolSchema

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
T = TypeVar("T")


# [EXPLAIN] `PoolMode` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class PoolMode(Enum):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ThreadMode = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ProcessMode = 2


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
# [EXPLAIN] `TokenBucketWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TokenBucketWorker:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, rate_limit: int):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit = rate_limit
        # this only used for observalability
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_count = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._semaphore = threading.Semaphore(rate_limit)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @ray.method(concurrency_group="acquire")
    # [EXPLAIN] `acquire` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def acquire(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._semaphore.acquire()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_count += 1

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @ray.method(concurrency_group="release")
    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def release(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._semaphore.release()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_count -= 1

    # [EXPLAIN] `get_current_count` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_current_count(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.current_count


# [EXPLAIN] `ExecutionWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ExecutionWorker:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    # [EXPLAIN] `_init_rate_limit` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    # [EXPLAIN] `ping` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def ping(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
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
                logger.warning(f"Error when executing code: {e}")


# [EXPLAIN] `init_execution_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mode == PoolMode.ThreadMode:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)


# [EXPLAIN] `SandboxFusionTool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SandboxFusionTool(BaseTool):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config, tool_schema)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._instance_dict = {}
        # TODO: better documentation for the config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_workers = config.get("num_workers", 10)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rate_limit = config.get("rate_limit", 10)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.default_timeout = config.get("default_timeout", 30)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.default_language = config.get("default_language", "python")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.execution_pool = init_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.sandbox_fusion_url == "":
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("sandbox_fusion_url is not set")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_msg = f"Init SandboxFusionTool with config: {config}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.info(log_msg)

    # [EXPLAIN] `get_openai_tool_schema` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.tool_schema

    # [EXPLAIN] `create` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if instance_id is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            instance_id = str(uuid4())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return instance_id

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        code = parameters.get("code", "")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        timeout = parameters.get("timeout", self.default_timeout)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        language = parameters.get("language", self.default_language)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(code, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            code = str(code)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result, result, result.strip()

    # [EXPLAIN] `execute_code` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def execute_code(self, instance_id, code, timeout=30, language="python"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result_status, metadata = _process_single_case(0, None, None, self.sandbox_fusion_url, code, timeout, language)
        # we should always expect this since we don't have correct answer
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if metadata["run_status"] == "Finished":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return actual_output
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "no stdout here"

    # [EXPLAIN] `calc_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._instance_dict[instance_id]["reward"]

    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def release(self, instance_id: str, **kwargs) -> None:
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del self._instance_dict[instance_id]
