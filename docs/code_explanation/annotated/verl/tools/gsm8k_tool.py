# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from typing import Any, Optional, Tuple
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from uuid import uuid4

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score import gsm8k

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base_tool import BaseTool
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .schemas import OpenAIFunctionToolSchema

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `Gsm8kTool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Gsm8kTool(BaseTool):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A demo tool for calculating the reward of gsm8k.

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
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config, tool_schema)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._instance_dict = {}

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
            "reward": 0.0,
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return instance_id

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        answer = parameters.get("answer", "")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(answer, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            answer = str(answer)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if answer.startswith("#### "):
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self._instance_dict[instance_id]["response"] = answer
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self._instance_dict[instance_id]["response"] = "#### " + answer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self._instance_dict[instance_id]["reward"] = reward

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"Current parsed {answer=} {reward=}", tool_reward, {}

    # [EXPLAIN] `calc_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def release(self, instance_id: str, **kwargs) -> None:
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del self._instance_dict[instance_id]
