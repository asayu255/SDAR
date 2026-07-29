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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig, OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from openai.types.chat.chat_completion import ChatCompletion
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse, ErrorResponse

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto


# [EXPLAIN] `init_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_config() -> DictConfig:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_path = "Qwen/Qwen2-7B-Instruct"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.model.path = model_path
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.mode = "async"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.prompt_length = 4096
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.rollout.response_length = 4096

    # test sleep/wake_up with fsdp offload
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return config


# [EXPLAIN] `test_vllm_multi_turn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_vllm_multi_turn(config):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_rollout_manager = init_async_rollout_manager(config)

    # test sleep and wake_up
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    async_rollout_manager.sleep()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    async_rollout_manager.wake_up()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_chat_scheduler = async_rollout_manager.chat_scheduler

    # =========================== 2. Multi turn rollout  ===========================
    # [EXPLAIN] `callback` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert exception is None, f"exception: {exception}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages, round = info["messages"], info["round"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        message = completions.choices[0].message
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        messages.append({"role": message.role, "content": message.content})
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_headers = {"x-request-id": completions.id}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if round == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            messages.append({"role": "user", "content": "What is your name?"})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 1},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif round == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            messages.append({"role": "user", "content": "What is your favorite color?"})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 2},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Done!")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    messages = [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    async_rollout_manager.submit_chat_completions(
        callback=callback,
        callback_additional_info={"messages": messages, "round": 0},
        model=model_name,
        messages=messages,
    )
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(messages) == 6
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for round, message in enumerate(messages):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if round % 2 == 0:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert message["role"] == "user"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert message["role"] == "assistant"

    # =========================== 3. Generate sequences  ===========================
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
        },
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result = async_rollout_manager.generate_sequences(prompts=batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(result) == 2
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result.batch["input_ids"].size(1) == seq_len
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result.batch["attention_mask"].size(1) == seq_len
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result.batch["position_ids"].size(1) == seq_len

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# [EXPLAIN] `test_vllm_streaming_response` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
async def test_vllm_streaming_response(config):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_rollout_manager = init_async_rollout_manager(config)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    async_llm_server = async_rollout_manager.async_llm_servers[0]

    # non-streaming request
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "What is your name?"}],
        stream=False,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generator = async_llm_server.chat_completion_generator.remote(request)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    async for ref in generator:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        status_code, data = await ref
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f">>>> status_code: {status_code}, {data}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data[len("data: ") :].rstrip()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if status_code != 200:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = ErrorResponse(**json.loads(data))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = ChatCompletionResponse(**json.loads(data))
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response.choices[0].message.role == "assistant"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response.choices[0].message.content is not None

    # streaming request
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "How are you?"}],
        stream=True,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generator = async_llm_server.chat_completion_generator.remote(request)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    async for ref in generator:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        status_code, data = await ref
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f">>>> status_code: {status_code}, {data}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data[len("data: ") :].rstrip()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if status_code != 200:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = ErrorResponse(**json.loads(data))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif data == "[DONE]":
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = ChatCompletionStreamResponse(**json.loads(data))
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response.choices[0].delta.role is None or response.choices[0].delta.role == "assistant"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response.choices[0].delta.content is not None

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.shutdown()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = init_config()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_vllm_multi_turn(config)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    asyncio.run(test_vllm_streaming_response(config))
