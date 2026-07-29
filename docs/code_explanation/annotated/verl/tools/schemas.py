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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Literal

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pydantic import BaseModel


# [EXPLAIN] `OpenAIFunctionPropertySchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionPropertySchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The schema of a parameter in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    type: str
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    description: str | None = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    enum: list[str] | None = None


# [EXPLAIN] `OpenAIFunctionParametersSchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionParametersSchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The schema of parameters in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    type: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    properties: dict[str, OpenAIFunctionPropertySchema]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    required: list[str]


# [EXPLAIN] `OpenAIFunctionSchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionSchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The schema of a function in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    name: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    description: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    parameters: OpenAIFunctionParametersSchema
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    strict: bool = False


# [EXPLAIN] `OpenAIFunctionToolSchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionToolSchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The schema of a tool in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    type: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    function: OpenAIFunctionSchema


# [EXPLAIN] `OpenAIFunctionParsedSchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionParsedSchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The parsed schema of a tool in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    name: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    arguments: str  # JSON string


# [EXPLAIN] `OpenAIFunctionCallSchema` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionCallSchema(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The parsed schema of a tool in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    name: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    arguments: dict[str, Any]

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `from_openai_function_parsed_schema` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_openai_function_parsed_schema(parsed_schema: OpenAIFunctionParsedSchema) -> tuple["OpenAIFunctionCallSchema", bool]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        has_decode_error = False
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arguments = json.loads(parsed_schema.arguments)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except json.JSONDecodeError:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arguments = {}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(arguments, dict):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arguments = {}
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            has_decode_error = True

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


# [EXPLAIN] `OpenAIFunctionToolCall` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OpenAIFunctionToolCall(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The tool call in OpenAI format."""

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    id: str
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    type: Literal["function"] = "function"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    function: OpenAIFunctionCallSchema
