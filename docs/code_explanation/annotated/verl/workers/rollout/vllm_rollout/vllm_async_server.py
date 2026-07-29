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
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections.abc import AsyncGenerator
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import cloudpickle
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.requests import Request
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.responses import JSONResponse, StreamingResponse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm import SamplingParams
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.engine.arg_utils import AsyncEngineArgs
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.entrypoints.logger import RequestLogger
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.v1.engine.async_llm import AsyncLLM
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.v1.executor.abstract import Executor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.worker.worker_base import WorkerWrapperBase

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.async_server import AsyncServerBase

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)


# [EXPLAIN] `ExternalRayDistributedExecutor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ExternalRayDistributedExecutor(Executor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """An executor that engines are launched by external ray actors."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    uses_ray: bool = False

    # [EXPLAIN] `_init_executor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_executor(self) -> None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fields = self.vllm_config.instance_id.split(":")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init(namespace=namespace)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."

        # [EXPLAIN] `get_pg_index_and_local_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fields = actor_name.split(":")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"instance_id: {self.vllm_config.instance_id} intializes with external actors: {actor_names}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.collective_rpc("init_worker", args=([kwargs],))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.collective_rpc("init_device")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.collective_rpc("load_model")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    # [EXPLAIN] `collective_rpc` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(method, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sent_method = method
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sent_method = cloudpickle.dumps(method)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = ray.get([worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return outputs

    # [EXPLAIN] `check_health` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def check_health(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `AsyncvLLMServer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncvLLMServer(AsyncServerBase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vllm_dp_size = vllm_dp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.vllm_dp_rank = vllm_dp_rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.wg_prefix = wg_prefix
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.engine: AsyncLLM = None

    # [EXPLAIN] `init_engine` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def init_engine(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Init vLLM AsyncLLM engine."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = self.config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_path = config.model.path
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_name = "/".join(model_path.split("/")[-2:])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = copy_to_local(model_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trust_remote_code = config.model.get("trust_remote_code", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = config.rollout

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k in config.keys():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(SamplingParams(), str(k)):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                kwargs[k] = config.get(k)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"override_generation_config: {kwargs}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )

        # init async llm engine
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_config = engine_args.create_engine_config()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        namespace = ray.get_runtime_context().namespace
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_config = self.engine.model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
        )

    # [EXPLAIN] `chat_completion` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def chat_completion(self, raw_request: Request):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request_json = await raw_request.json()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request = ChatCompletionRequest(**request_json)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(generator, ErrorResponse):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if request.stream:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return StreamingResponse(content=generator, media_type="text/event-stream")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(generator, ChatCompletionResponse)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return JSONResponse(content=generator.model_dump())

    # [EXPLAIN] `chat_completion_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def chat_completion_generator(self, request: ChatCompletionRequest) -> AsyncGenerator[Tuple[int, str]]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Direct chat completion without FastAPI.

        Args:
            request: ChatCompletionRequest, request object.

        Returns:
            AsyncGenerator[Tuple[int, str]]: async generator of (status_code, data) pairs.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generator = await self.openai_serving_chat.create_chat_completion(request)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(generator, ErrorResponse):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = generator.model_dump_json(exclude_unset=True)
            # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
            yield generator.code, f"data: {data}\n\n"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if request.stream:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            async for chunk in generator:
                # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                yield 200, chunk
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(generator, ChatCompletionResponse)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = generator.model_dump_json(exclude_unset=True)
            # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
            yield 200, f"data: {data}\n\n"

    # [EXPLAIN] `wake_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def wake_up(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await self.engine.wake_up()

    # [EXPLAIN] `sleep` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await self.engine.reset_prefix_cache()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await self.engine.sleep()
