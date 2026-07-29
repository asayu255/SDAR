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
from typing import Any, Optional, Tuple
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from uuid import uuid4

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .schemas import OpenAIFunctionToolSchema


# [EXPLAIN] `BaseTool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseTool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Base class for tools.

    A tool should support the following methods:

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name = tool_schema.function.name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tool_schema = tool_schema

    # [EXPLAIN] `get_openai_tool_schema` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
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
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return str(uuid4())
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return instance_id

    # [EXPLAIN] `execute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "Updated the tool state.", 0.0, {}

    # [EXPLAIN] `calc_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0.0

    # [EXPLAIN] `release` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def release(self, instance_id: str, **kwargs) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
