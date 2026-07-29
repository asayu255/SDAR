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
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List, Literal, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pydantic import BaseModel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask


# [EXPLAIN] `FinishReasonTypeEnum` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FinishReasonTypeEnum(str, Enum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The enum for finish reason type."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    LENGTH = "length"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    STOP = "stop"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    TOOL_CALL = "tool_calls"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if value == "stop":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls.STOP
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif value == "length":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls.LENGTH
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif value == "tool_calls":
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls.TOOL_CALL
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported finish reason type: {value}")


# [EXPLAIN] `Message` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Message(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    role: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    content: str
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


# [EXPLAIN] `AsyncRolloutRequestStateEnum` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncRolloutRequestStateEnum(str, Enum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The enum for async rollout request state."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    PENDING = "pending"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    RUNNING = "running"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    COMPLETED = "completed"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FAILED = "failed"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    TOOL_CALLING = "tool_calling"


# [EXPLAIN] `AsyncRolloutRequest` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncRolloutRequest(BaseModel):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The data model for async rollout."""

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_data_id: int = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rollout_offset: int = 0
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    request_id: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    state: AsyncRolloutRequestStateEnum
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    messages: List[Message]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tools_kwargs: Dict[str, Any] = {}
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    input_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    prompt_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    response_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    attention_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    prompt_attention_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    response_attention_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    position_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    prompt_position_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    response_position_ids: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    loss_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    prompt_loss_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    response_loss_mask: List[int]
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    reward_scores: Dict[str, float]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_len: int = 8192
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_model_len: int = 32768
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metrics: Dict[str, List[Any]] = {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_config: dict = {
        "chatml": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "tool_prefix_msg": "\n<|im_start|>tool\n",
            "tool_suffix_msg": "<|im_end|>",
        },
        "qwen": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "merge_tool_response": True,
            "tool_prefix_msg": "\n<|im_start|>user",
            "tool_suffix_msg": "<|im_end|>",
            "tool_response_prefix_msg": "\n<tool_response>\n",
            "tool_response_suffix_msg": "\n</tool_response>",
        },
    }

    # [EXPLAIN] `get_generation_prompt` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tokenizer.apply_chat_template(  # type: ignore
            conversation=[msg.model_dump() for msg in self.messages],
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None,
            add_generation_prompt=True,
            tokenize=True,
        )

    # [EXPLAIN] `add_assistant_message` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        format: Literal["chatml", "qwen"] = "chatml",
        already_over_long: bool = False,
    ) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Currently, we only support chatml format."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.messages.append(msg)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tool_calls is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            content_with_tool_calls: str = tokenizer.apply_chat_template(  # type: ignore
                conversation=[msg.model_dump()], add_generation_prompt=False, tokenize=False
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            content_with_tool_calls = content
        # TODO: support other formats
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if format in self.format_config:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_msg = self.format_config[format]["assistant_prefix_msg"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_msg = self.format_config[format]["assistant_suffix_msg"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tool_calls is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                content = content_with_tool_calls.split(f"{prefix_msg}")[-1].split(f"{suffix_msg}")[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                append_token_ids = content_token_ids
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _loss_mask = [1] * len(content_token_ids)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                append_token_ids = prefix_token_ids + content_token_ids
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                _loss_mask = [0] * len(prefix_token_ids) + [1] * len(content_token_ids)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                max_len = max(len(prefix_token_ids), len(suffix_token_ids))
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(
                    f"""Unsupported end of message format: 
                    {tokenizer.decode(self.input_ids[-max_len:])},
                    {tokenizer.decode(self.input_ids)=}, {self.messages=}"""
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not already_over_long:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                append_token_ids += suffix_token_ids
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _loss_mask += [1] * len(suffix_token_ids)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.input_ids += append_token_ids
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _attention_mask = [1] * len(append_token_ids)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.attention_mask += _attention_mask
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _delta_position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_position_id = self.position_ids[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.loss_mask += _loss_mask
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.position_ids += _position_ids
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported format: {format}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    # [EXPLAIN] `add_tool_response_message` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def add_tool_response_message(self, tokenizer: PreTrainedTokenizer, content: str, last_tool: bool, format: Literal["chatml", "qwen"] = "chatml") -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Currently, we only support chatml format."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        msg = Message(role="tool", content=content)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.messages.append(msg)
        # TODO: support other formats
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if format in self.format_config:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            merge_tool_responses = self.format_config[format].get("merge_tool_response", False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_msg = self.format_config[format]["tool_prefix_msg"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_msg = self.format_config[format]["tool_suffix_msg"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_resp = self.format_config[format].get("tool_response_prefix_msg", "")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_resp_token_ids = tokenizer.encode(prefix_resp, add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_resp = self.format_config[format].get("tool_response_suffix_msg", "")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            suffix_resp_token_ids = tokenizer.encode(suffix_resp, add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_suffix_token_ids = suffix_resp_token_ids + (suffix_token_ids if last_tool or not merge_tool_responses else [])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids or self.input_ids[-len(suffix_resp_token_ids) :] == suffix_resp_token_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                append_token_ids = prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.input_ids[-len(prefix_resp_token_ids) :] == prefix_resp_token_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                append_token_ids = content_token_ids + full_suffix_token_ids
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                append_token_ids = prefix_token_ids + prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-len(prefix_token_ids) :])}")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.input_ids += append_token_ids
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _attention_mask = [1] * len(append_token_ids)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.attention_mask += _attention_mask
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _delta_position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_position_id = self.position_ids[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.loss_mask += [0] * len(append_token_ids)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.position_ids += _position_ids
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported format: {format}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    # [EXPLAIN] `update_metrics` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        metrics: should be a dict of tools_name -> Any
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.metrics.get(tool_id) is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.metrics[tool_id] = []
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.metrics[tool_id].append(metrics)

    # [EXPLAIN] `finalize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_scores = reward_scores
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            pass
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.truncate_output_ids(tokenizer)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    # [EXPLAIN] `truncate_output_ids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.input_ids = self.input_ids[: self.max_model_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.attention_mask = self.attention_mask[: self.max_model_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.position_ids = self.position_ids[: self.max_model_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.loss_mask = self.loss_mask[: self.max_model_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
