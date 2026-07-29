# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Adapted from tests/workers/rollout/test_sglang_async_rollout_sf_tools.py


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import AsyncMock, MagicMock, patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoConfig, AutoTokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from utils_sglang import (
    get_rollout_config,
    prepare_inputs,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.search_tool import SearchTool
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + "How's the weather lately?"


# [EXPLAIN] `get_search_messages` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_search_messages():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_0_msg = {
        "role": "assistant",
        "content": "Let me search the web.",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "today's weather"}}}],
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_1_msg = {
        "role": "assistant",
        "content": "Let me search again.",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "tomorrow's weather"}}}],
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_2_msg = {
        "role": "assistant",
        "content": "<answer>Today is sunny and tomorrow will be cloudy in Beijing.</answer>",
    }

    # Mock search tool responses
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_0_msg = {"role": "tool", "content": "Today's weather in Beijing is sunny."}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_1_msg = {"role": "tool", "content": "Tomorrow's weather in Beijing is cloudy."}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    user_prompts = [user_prompt]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return user_prompts, expect_turn_array, tool_return_array


# [EXPLAIN] `TestRolloutWithSearchTools` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestRolloutWithSearchTools:
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `qwen_tokenizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def qwen_tokenizer(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = "Qwen/Qwen2.5-0.5B"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer.pad_token = tokenizer.eos_token
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tokenizer

    # we only need this for tokenizer
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `qwen_model_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def qwen_model_config(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = "Qwen/Qwen2.5-0.5B"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = AutoConfig.from_pretrained(local_model_path)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `search_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search_data(self, qwen_tokenizer):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        user_prompt, expect_turn_array, tool_return_array = get_search_messages()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = [[message] for message in user_prompt]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return prompts, preencode_turn_array, preencode_tool_return_array

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `search_rollout_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search_rollout_config(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_prompt_length = 4096
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_response_length = 3000
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dtype = "bfloat16"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_parallel_size = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tool_path = "./resource/tool_configs/search_tool_config"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rollout_config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `search_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search_data_proto(self, search_data, qwen_tokenizer):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        preencode_prompts, _, _ = search_data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = [qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages = np.asarray(preencode_prompts)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tools_kwargs = np.array(
            [
                {
                    "search": {
                        "create_kwargs": {"ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing.", "data_source": "searchR1_nq"},
                    },
                }
            ],
            dtype=object,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        index = np.array([0], dtype=object)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return prompts

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_tools_registration` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(rollout._tool_schemas) == 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "search" in rollout._tool_map.keys()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.tools.search_tool import SearchTool

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(rollout._tool_map["search"], SearchTool)
        # depend on the tokenizer
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert rollout._tool_call_parser_type == "qwen25"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_rollout_req_creation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rollout_req_creation(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(req_list) == 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(req_list[0].tools) == 1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(type(req_list[0].tools[0]))
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert req_list[0].tools[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description="Searches the web for relevant information based on the given query.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="A list of fully-formed semantic queries. The tool will return search results for each query.",
                            items={"type": "string"},
                        )
                    },
                    required=["query_list"],
                ),
                strict=False,
            ),
        )

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_over_size_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_over_size_case(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_rollout_config.multi_turn.max_turns = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req.finalize = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = [req]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, _ = search_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future = asyncio.Future()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 3000}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call.return_value = future
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._tp_rank = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(output_req_list) == 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_req = output_req_list[0]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.reward_scores == {"search": []}, f"output_req.reward_scores: {output_req.reward_scores}"
        # we should only have two message, one for prompt, second for response.
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(output_req.messages) == 2
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SearchTool, "execute", new_callable=AsyncMock)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_tool_call_basic_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tool_call_basic_case(self, mock_sampling, mock_engine, mock_env, mock_execute, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, tool_return_array = search_data

        # Mock search tool execution to return predefined responses
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_rollout_config.multi_turn.max_turns = 10
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req.finalize = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = [req]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = [asyncio.Future() for i in expect_turn_array]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if idx < len(expect_turn_array) - 1:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert rollout._function_call_parser.has_tool_call(turn)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert rollout._function_call_parser.parse_non_stream(turn)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call.side_effect = futures
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._tp_rank = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_req_list = loop.run_until_complete(asyncio.gather(*[rollout._async_rollout_a_request(req, True, False) for req in req_list]))

        # Verify conversation completed successfully with proper tool usage
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_req = output_req_list[0]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "search" in output_req.metrics
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.metrics["search"][0]["status"] == "success"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_execute.await_count == 2
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
        # Verify tool response messages contain expected content
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_counter = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for msg in output_req.messages:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if msg.role == "tool":
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert msg.content == tool_return_array[search_counter]
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                search_counter += 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert search_counter == 2

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SearchTool, "execute", new_callable=AsyncMock)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_tool_call_batch_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tool_call_batch_case(self, mock_sampling, mock_engine, mock_env, mock_execute, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, tool_return_array = search_data

        # Mock tool execution for large batch (100 requests * 2 calls each)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        search_rollout_config.multi_turn.max_turns = 10
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(
            actor_module="",
            config=search_rollout_config,
            tokenizer=qwen_tokenizer,
            model_hf_config=qwen_model_config,
        )
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        base_req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_nums = 100
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_turns_map = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_turns_counter = {}

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(req_nums):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_req = deepcopy(base_req)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_req.batch_data_id = i
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tmp_req.request_id = i
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            req_list.append(MagicMock(wraps=tmp_req, spec=AsyncRolloutRequest))

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            futures = [asyncio.Future() for _ in expect_turn_array]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, (fut, turn) in enumerate(zip(futures, expect_turn_array)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                fut.set_result(
                    {
                        "text": turn,
                        "meta_info": {
                            "id": "dummy",
                            "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                            "prompt_tokens": len(turn),
                            "completion_tokens": 100,
                        },
                    }
                )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            req_turns_map[i] = futures
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            req_turns_counter[i] = 0

        # [EXPLAIN] `hacked_handle_engine_call` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest, *_args, **_kwargs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fut = req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            req_turns_counter[_req.batch_data_id] += 1
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return await fut

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with patch.object(SGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout._tp_rank = 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loop = asyncio.get_event_loop()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_req_list = loop.run_until_complete(asyncio.gather(*[rollout._async_rollout_a_request(r, True, False) for r in req_list]))

        # Verify all requests completed successfully
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(output_req_list) == req_nums
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for out_req in output_req_list:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert out_req.state == AsyncRolloutRequestStateEnum.COMPLETED
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert "search" in out_req.metrics
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for metric in out_req.metrics["search"]:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert metric["status"] == "success"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(out_req.messages) == 6  # user + 3 assistant + 2 tool
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert sum(1 for m in out_req.messages if m.role == "tool") == 2

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mock_execute.await_count == 2 * req_nums
