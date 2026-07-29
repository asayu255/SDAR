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
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from copy import deepcopy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import wraps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from unittest.mock import MagicMock, patch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.testing._internal.common_distributed import MultiProcessTestCase
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
from verl.tools.sandbox_fusion_tools import TokenBucketWorker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
sandbox_url = ""


# [EXPLAIN] `get_sandbox_fusion_messages` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_sandbox_fusion_messages():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    user_prompt = {
        "role": "user",
        "content": """
            Solve the following problem step by step. You now have the ability to selectively 
            write executable Python code to enhance your reasoning process. \n\n**user question:**\nThere 
            are 152 students at Dala High School. Assume the following: \n- 100 students take a Math class \n- 94 
            students take a Science class \n- 57 students take an English class \n- 73 students take a Math class 
            and a Science class \n- 24 students take a Math class and an English class \n- 27 students take a Science 
            class and an English class \n- 22 students take a Math class and a Science class and an English class\n \nHow 
            many students take neither a Math class nor a Science class nor an Eglish class?\n\nRemember to place the final 
            answer in the last part using the format: \n<answer>\n\boxed{'The final answer goes here.'}\n</answer>
        """,
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_0_msg = {
        "role": "assistant",
        "content": """
            Okay, so I need to find out how many students at Dala High School are not taking any of the three classes: Math, 
            Science, or English. The total number of students is 152. Let me see... I remember this is a problem about sets 
            and maybe using the principle of inclusion-exclusion. Let me recall how that works.\n\nFirst, the inclusion-exclusion 
            principle for three sets says that the total number of students taking at least one of the classes is equal to the 
            sum of the numbers in each individual class, minus the sum of the numbers in each pair of classes, plus the number in 
            all three classes. Then, subtract that total from the overall number of students to get those not taking any of the 
            three. \n\nLet me write that down step by step. Let M be the set of students taking Math, S for Science, and E for English. 
            Then:\n\nTotal in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - |M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|\n\nGiven the 
            numbers:\n\n|M| = 100\n\n|S| = 94\n\n|E| = 57\n\n|M ∩ S| = 73\n\n|M ∩ E| = 24\n\n|S ∩ E| = 27\n\n|M ∩ S ∩ E| = 22\n\nSo plugging 
            these into the formula:\n\nTotal = 100 + 94 + 57 - 73 - 24 - 27 + 22\n\nLet me compute that step by step using code to ensure 
            accuracy.\n
        """,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {
                        "code": """M = 100\\nS = 94\\nE = 57\\nM_S = 73\\nM_E = 24\\nS_E = 27\\nM_S_E = 
                        22\\n\\ntotal_in_any = M + S + E - M_S - M_E - S_E + M_S_E\\nstudents_neither = 152 - total_in_any\\nprint(students_neither)""",
                    },
                },
            }
        ],
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_0_msg = {
        "role": "tool",
        "content": """3""",
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_1_msg = {
        "role": "assistant",
        "content": """
            The code returns 3, indicating that 3 students take none of the classes. To verify, I'll break down each
            category to ensure the total matches:\n\n- Only Math and Science: 73 - 22 = 51\n- Only Math and English:
            24 - 22 = 2\n- Only Science and English: 27 - 22 = 5\n- Only Math: 100 - 51 - 2 - 22 = 25\n- Only Science:
            94 - 51 - 5 - 22 = 16\n- Only English: 57 - 2 - 5 - 22 = 28\n\nSumming all categories:\n
        """,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {
                        "code": """only_M_S = 73 - 22\\nonly_M_E = 24 - 22\\nonly_S_E = 27 - 22\\n\\nonly_M = 100 - only_M_S -
                        only_M_E - 22\\nonly_S = 94 - only_M_S - only_S_E - 22\\nonly_E = 57 - only_M_E - only_S_E - 22\\n\\ntotal_verify
                        = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22\\nprint(total_verify)""",
                    },
                },
            }
        ],
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_1_msg = {
        "role": "tool",
        "content": """149""",
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_2_msg = {
        "role": "assistant",
        "content": """
            The verification total is 149, so students not taking any classes are 152 - 149 = 3, confirming the initial
            result.\n\n<answer>\n\\boxed{3}\n</answer>
        """,
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    user_prompts = [user_prompt]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return user_prompts, expect_turn_array, tool_return_array


# [EXPLAIN] `skip_if_valid_sandbox` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def skip_if_valid_sandbox(url):
    # [EXPLAIN] `decorator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decorator(func):
        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @wraps(func)
        # [EXPLAIN] `wrapper` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def wrapper(*args, **kwargs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if url == "" or url is None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                pytest.skip("No valid sandbox url provided")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return wrapper

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return decorator


# [EXPLAIN] `TestRolloutWithTools` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestRolloutWithTools:
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
    # [EXPLAIN] `sandbox_fusion_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sandbox_fusion_data(self, qwen_tokenizer):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        user_prompt, expect_turn_array, tool_return_array = get_sandbox_fusion_messages()
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
    # [EXPLAIN] `sandbox_fusion_rollout_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sandbox_fusion_rollout_config(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_prompt_length = 1024
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_response_length = 1024
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dtype = "bfloat16"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_parallel_size = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tool_path = "./resource/tool_configs/sandbox_fusion_tool_config"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rollout_config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @pytest.fixture
    # [EXPLAIN] `sandbox_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sandbox_data_proto(self, sandbox_fusion_data, qwen_tokenizer):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        preencode_prompts, _, _ = sandbox_fusion_data
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
                    "code_interpreter": {
                        "create_kwargs": {"ground_truth": "test-solution-str"},
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
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(rollout._tool_schemas) == 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "code_interpreter" in rollout._tool_map.keys()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.tools.sandbox_fusion_tools import SandboxFusionTool

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(rollout._tool_map["code_interpreter"], SandboxFusionTool)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert rollout._tool_call_parser_type == "qwen25"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_rollout_req_creation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rollout_req_creation(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)
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
                name="code_interpreter",
                description="A tool for executing code.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "code": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The code to execute.",
                            enum=None,
                        )
                    },
                    required=["code"],
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
    def test_over_size_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sandbox_fusion_rollout_config.multi_turn.max_turns = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req.finalize = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = [req]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future = asyncio.Future()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 1024}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
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
        assert output_req.reward_scores == {"code_interpreter": []}
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
    @skip_if_valid_sandbox(sandbox_url)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_tool_call_basic_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tool_call_basic_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sandbox_fusion_rollout_config.multi_turn.max_turns = 10
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self._tool_map["code_interpreter"].sandbox_fusion_url = sandbox_url
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req.finalize = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = [req]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout._handle_engine_call = MagicMock()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = [asyncio.Future() for i in expect_turn_array]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
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
        # here we verify whether the code sandbox is executed correctly
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert output_req.metrics == {"code_interpreter": ["3", "149"]}
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert rollout._handle_engine_call.call_count == 3
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        code_counter = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for msg in output_req.messages:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if msg.role == "tool":
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                code_counter += 1
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert msg.content == tool_return_array[code_counter]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert code_counter == 2

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @skip_if_valid_sandbox(sandbox_url)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    # [EXPLAIN] `test_tool_call_batch_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_tool_call_batch_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sandbox_fusion_rollout_config.multi_turn.max_turns = 10
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self._tool_map["code_interpreter"].sandbox_fusion_url = sandbox_url
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_nums = 100
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_turns_counter = {}
        # this map should a Map[id:List[Futures]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        req_turns_map = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(req_nums):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _temp_req = deepcopy(req)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _temp_req.batch_data_id = i
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _temp_req.request_id = i
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            req_list.append(MagicMock(wraps=_temp_req, spec=AsyncRolloutRequest))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            futures = [asyncio.Future() for i in expect_turn_array]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if idx < len(expect_turn_array) - 1:
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert rollout._function_call_parser.has_tool_call(turn)
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert rollout._function_call_parser.parse_non_stream(turn)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            req_turns_map[_temp_req.batch_data_id] = futures
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            req_turns_counter[_temp_req.batch_data_id] = 0

        # [EXPLAIN] `hacked_handle_engine_call` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            req_turns_counter[_req.batch_data_id] += 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            re = await result
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return re

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with patch.object(SGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
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
            assert len(output_req_list) == req_nums
            # FIGUER out how to count this
            # assert rollout._handle_engine_call.call_count == 3 * req_nums
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for output_req in output_req_list:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
                # here we verify whether the code sandbox is executed correctly
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert output_req.metrics == {"code_interpreter": ["3", "149"]}
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                code_counter = 0
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for msg in output_req.messages:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if msg.role == "tool":
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        code_counter += 1
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert code_counter == 2


# [EXPLAIN] `RayMultiProcessTestCase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RayMultiProcessTestCase(MultiProcessTestCase):
    # [EXPLAIN] `setUp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def setUp(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().setUp()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init(ignore_reinit_error=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("init_single cluster")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._spawn_processes()

    # [EXPLAIN] `tearDown` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tearDown(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("tearDown_single cluster")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.shutdown()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `TestActor` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestActor:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, rank, world_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._world_size = world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._rank = rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.time_list = []

    # [EXPLAIN] `record_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def record_rank(self, rank):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.rank_list.append(rank)

    # [EXPLAIN] `get_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rank(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._rank

    # [EXPLAIN] `ping` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def ping(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] `record_execution_time` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def record_execution_time(self, time):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.time_list.append(time)

    # [EXPLAIN] `get_time` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_time(self, timeout):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import time

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        now = time.time()
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while time.time() - now < timeout:
            # for start and end time
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.time_list) == self._world_size * 2:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.time_list.sort()
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return self.time_list[-1] - self.time_list[0]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                time.sleep(1)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] `verify_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def verify_rank(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import time

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        now = time.time()
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while time.time() - now < 10:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.rank_list) == self._world_size:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(self.rank_list)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.rank_list.sort()
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i in range(self._world_size):
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.rank_list[i] != i:
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return False
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return True
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                time.sleep(1)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] `TestRayGlobalActorCase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestRayGlobalActorCase(RayMultiProcessTestCase):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self) -> int:
        # for DP = 8
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 2

    # [EXPLAIN] `test_basic_multi_process_init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_basic_multi_process_init(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        handle = TestActor.remote(self.rank, self.world_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        re = ray.get(handle.get_rank.remote())
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert re == self.rank, f"rank not match: {re} != {self.rank}"

    # def test_global_actor(self):
    #     ray.init("auto",namespace="test",ignore_reinit_error=True)
    #     handle = TestActor.options(get_if_exists=True,name="test-actor").remote(self.rank,self.world_size)
    #     handle.record_rank.remote(self.rank)
    #     # since test actor's concurrency is 1, we need to wait for all processes to finish
    #     time.sleep(5)
    #     assert ray.get(handle.ping.remote()) == True # make sure actor handle is valid
    #     if self.rank == 0:
    #         assert ray.get(handle.verify_rank.remote()) == True
    #     else:
    #         # get_actor use weak_ref, so we need to make sure the actor is not garbage collected
    #         time.sleep(10)


# [EXPLAIN] `TestSingleNodeRateLimiterCase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestSingleNodeRateLimiterCase(RayMultiProcessTestCase):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self) -> int:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 1

    # [EXPLAIN] `test_rate_limiter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rate_limiter(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=3)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=3, mode=PoolMode.ThreadMode)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        center = TestActor.options(get_if_exists=True, name="test-actor").remote(self.rank, self.world_size)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get(exec_worker.ping.remote())

        # [EXPLAIN] `fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def fn(i):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import time

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(3)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return i

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = [exec_worker.execute.remote(fn, i) for i in range(6)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = loop.run_until_complete(asyncio.gather(*tasks))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        duration = end - start
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        center.record_execution_time.remote(start)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        center.record_execution_time.remote(end)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Total time: {duration:.2f} seconds for rank: {self.rank}")

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert results == list(range(6))
        # we have 6 task with rate limit of 3, therefore we need at least 2 round: 3*2=6 seconds
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert duration > 6
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert duration < 10

    # [EXPLAIN] `test_rotten_execution` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rotten_execution(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get(exec_worker.ping.remote())

        # [EXPLAIN] `fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def fn(i):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i == 10:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise Exception("test")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return i

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = [exec_worker.execute.remote(fn, i) for i in range(20)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = loop.run_until_complete(asyncio.gather(*tasks))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expect_result = [None] + list(range(10)) + list(range(11, 20))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sorted_data = sorted(results, key=lambda x: (x is not None, x))
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert sorted_data == expect_result, f"results: {results}, expect_result: {expect_result}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rate_limiter = TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rate = ray.get(rate_limiter.get_current_count.remote())
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert rate == 0, f"rate: {rate}"


# [EXPLAIN] `TestMultiNodeRateLimiterCase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestMultiNodeRateLimiterCase(RayMultiProcessTestCase):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self) -> int:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 2

    # [EXPLAIN] `test_rate_limiter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def test_rate_limiter(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        center = TestActor.options(get_if_exists=True, name="test-actor").remote(self.rank, self.world_size)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get(exec_worker.ping.remote())

        # [EXPLAIN] `fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def fn(i):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import time

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(2)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return i

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tasks = [exec_worker.execute.remote(fn, i) for i in range(6)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = loop.run_until_complete(asyncio.gather(*tasks))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end = time.time()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        duration = end - start
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        center.record_execution_time.remote(start)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        center.record_execution_time.remote(end)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Total time: {duration:.2f} seconds for rank: {self.rank}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert results == list(range(6))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(5)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_cost = ray.get(center.get_time.remote(10))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"for total cost: {total_cost}")
            # # we have 6 task each node * 2node = 12 task, each task take 2 second.
            # with rate limit of 6,
            # therefore we need at least 2 round: 12/6*2=4 seconds
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert total_cost > 4, total_cost
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(10)
