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
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import socket
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tempfile
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import asynccontextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import aiohttp
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import fastapi
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uvicorn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from datasets import load_dataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from openai.types.chat.chat_completion import ChatCompletion
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.requests import Request
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from starlette.responses import JSONResponse

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from examples.ppo_trainer.naive_chat_scheduler import NaiveChatCompletionScheduler
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto


# [EXPLAIN] `_get_free_port` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_free_port():
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with socket.socket() as sock:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sock.bind(("", 0))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sock.getsockname()[1]


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `Sandbox` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Sandbox:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Sandbox to execute python code.

    WARNING: This class is for testing purpose only, do not use it in production.
    Please use a sandbox with strong isolation and security restrictions instead.
    """

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

    # [EXPLAIN] `code_execution` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def code_execution(self, request: Request):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        request_json = await request.json()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        code = request_json["code"]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"execute code:\n{code}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(temp_file, "w") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write(code)

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            process = await asyncio.create_subprocess_exec(sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stdout, stderr = await process.communicate()

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return JSONResponse(content={"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": process.returncode})
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.unlink(temp_file)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except:  # noqa: E722
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass

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

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os._exit(-1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        app = fastapi.FastAPI(lifespan=lifespan)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        app.router.add_api_route("/code/execution", self.code_execution, methods=["POST"])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.port = _get_free_port()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        server = uvicorn.Server(config)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await server.serve()

    # [EXPLAIN] `get_server_address` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def get_server_address(self) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get FastAPI server address."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await self.server_ready.wait()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"{self.address}:{self.port}"


# [EXPLAIN] `ToolChatCompletionScheduler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ToolChatCompletionScheduler(NaiveChatCompletionScheduler):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """This is a demo chat completion scheduler that supports sandbox code execution
    described in ReTool paper: https://arxiv.org/pdf/2504.11536
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, model_path, server_addresses, sandbox_address, system_prompt, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config, model_path, server_addresses, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sandbox_address = sandbox_address
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.system_prompt = system_prompt

    # [EXPLAIN] `sandbox_code_execution` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute python code in sandbox."""
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            session = aiohttp.ClientSession()
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            async with session.post(
                url=f"http://{self.sandbox_address}/code/execution",
                json={"code": code},
            ) as resp:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return await resp.json()
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await session.close()

    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body={
                "include_stop_str_in_output": True,
                "stop": ["</answer>", "</code>"],
            },
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        do_sample = batch.meta_info.get("do_sample", True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_validate = batch.meta_info.get("validate", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not do_sample or is_validate:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs["n"] = 1
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs["temperature"] = 0

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        kwargs.update(sampling_params)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_turns = 3

        # [EXPLAIN] `callback` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            batch_conversations[batch_index].append({"role": role, "content": content})

            # STEP 0: check if we reach max turns
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if turn == max_turns:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[id={completions.id},turn={turn}] Reach max turns {max_turns}, done!")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return

            # STEP 1: check if we got answer
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if matches:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[id={completions.id},turn={turn}] Got answer: {matches[0]}, done!")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return

            # STEP 2: check if we got code block
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            matches = re.findall(r"<code>\s*```python(.*?)```\s*</code>", content, re.DOTALL)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not matches:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"[id={completions.id},turn={turn}] No code block found, done!")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return

            # STEP 3: execute code block in sandbox
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            code = matches[0].strip()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = await self.sandbox_code_execution(code)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            stdout, stderr = result["stdout"], result["stderr"]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            batch_conversations[batch_index].append({"role": "tool", "content": f"{stdout}{stderr}"})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[id={completions.id},turn={turn}] Code block executed, continue...")

            # STEP 4: resubmit chat completions with code block output
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            extra_headers = {"x-request-id": completions.id}
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **kwargs,
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks, batch_conversations = [], [None] * len(batch)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_conversations[batch_index] = [{"role": "system", "content": self.system_prompt}] + list(conversation)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 1,
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await asyncio.gather(*tasks)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        # _postprocess assumes n>=1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_conversations = [[conversation] for conversation in batch_conversations]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._postprocess(batch, batch_conversations, kwargs["n"])


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
system_prompt = """
You are a helpful assistant. Let's solve math problem in following steps:
1. Write a python code first and return the code to user, the code must be in following format:

<code>
```python
import os

print(...)
```
</code>

The code must explictly print necessary output to stdout. Remember stop generation at </code> immediately and return the code.
2. User will send the python code to a external sandbox to execute and get output from stdout.
3. User will send the output in format <interpreter>output</interpreter> to you, and you should use the output to answer the question.
The answer format must be: <answer>\\boxed{'The final answer goes here.'}</answer>
"""


# [EXPLAIN] `test_vllm_tool_calling` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_vllm_tool_calling():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # Load config
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.model.path = "Qwen/Qwen2-7B-Instruct"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.mode = "async"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.chat_scheduler = "tests.workers.rollout.test_vllm_tool_calling.ToolChatCompletionScheduler"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.prompt_length = 8192
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.response_length = 8192

    # Init sandbox and async rollout manager
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sandbox = Sandbox.options(num_cpus=1).remote()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"sandbox_address": sandbox_address, "system_prompt": system_prompt})

    # Build dataset
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])})

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result = async_rollout_manager.generate_sequences(prompts=prompts)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(result) == len(dataset)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_vllm_tool_calling()
