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
import importlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Optional, Type

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn

# Supported models in Megatron-LM
# Architecture -> (module, class).
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_MODELS = {
    "LlamaForCausalLM": (
        "llama",
        ("ParallelLlamaForCausalLMRmPadPP", "ParallelLlamaForValueRmPadPP", "ParallelLlamaForCausalLMRmPad"),
    ),
    "Qwen2ForCausalLM": (
        "qwen2",
        ("ParallelQwen2ForCausalLMRmPadPP", "ParallelQwen2ForValueRmPadPP", "ParallelQwen2ForCausalLMRmPad"),
    ),
    "MistralForCausalLM": (
        "mistral",
        ("ParallelMistralForCausalLMRmPadPP", "ParallelMistralForValueRmPadPP", "ParallelMistralForCausalLMRmPad"),
    ),
}


# return model class
# [EXPLAIN] `ModelRegistry` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ModelRegistry:
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `load_model_cls` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_model_cls(model_arch: str, value=False) -> Optional[Type[nn.Module]]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if model_arch not in _MODELS:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        megatron = "megatron"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module_name, model_cls_name = _MODELS[model_arch]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not value:  # actor/ref
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_cls_name = model_cls_name[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif value:  # critic/rm
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_cls_name = model_cls_name[1]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return getattr(module, model_cls_name, None)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `get_supported_archs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_supported_archs() -> List[str]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return list(_MODELS.keys())
