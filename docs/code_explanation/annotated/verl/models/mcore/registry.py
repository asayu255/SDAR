# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Registry module for model architecture components.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from enum import Enum
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Callable

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .config_converter import (
    PretrainedConfig,
    TransformerConfig,
    hf_to_mcore_config_dense,
    hf_to_mcore_config_dpskv3,
    hf_to_mcore_config_llama4,
    hf_to_mcore_config_mixtral,
    hf_to_mcore_config_qwen2_5_vl,
    hf_to_mcore_config_qwen2moe,
    hf_to_mcore_config_qwen3moe,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .model_forward import gptmodel_forward_no_padding, model_forward_gen
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .model_forward_fused import fused_forward_model_gen
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .model_initializer import (
    BaseModelInitializer,
    DeepseekV3Model,
    DenseModel,
    MixtralModel,
    Qwen2MoEModel,
    Qwen3MoEModel,
    Qwen25VLModel,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .weight_converter import (
    McoreToHFWeightConverterDense,
    McoreToHFWeightConverterDpskv3,
    McoreToHFWeightConverterMixtral,
    McoreToHFWeightConverterQwen2_5_VL,
    McoreToHFWeightConverterQwen2Moe,
    McoreToHFWeightConverterQwen3Moe,
)


# [EXPLAIN] `SupportedModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SupportedModel(Enum):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    LLAMA = "LlamaForCausalLM"  # tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN2 = "Qwen2ForCausalLM"  # tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN2_MOE = "Qwen2MoeForCausalLM"  # pending
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DEEPSEEK_V3 = "DeepseekV3ForCausalLM"  # not tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    MIXTRAL = "MixtralForCausalLM"  # tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"  # not supported
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    LLAMA4 = "Llama4ForConditionalGeneration"  # not tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN3 = "Qwen3ForCausalLM"  # tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN3_MOE = "Qwen3MoeForCausalLM"  # tested
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    GLM4_MOE = "Glm4MoeForCausalLM"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN3_TOKEN_CLASSIFICATION = "Qwen3ForTokenClassification"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN3_MOE_VL = "Qwen3VLMoeForConditionalGeneration"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    QWEN3_VL = "Qwen3VLForConditionalGeneration"


# Registry for model configuration converters
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_CONFIG_CONVERTER_REGISTRY: dict[SupportedModel, Callable[[PretrainedConfig, torch.dtype], TransformerConfig]] = {
    SupportedModel.LLAMA: hf_to_mcore_config_dense,
    SupportedModel.QWEN2: hf_to_mcore_config_dense,
    SupportedModel.QWEN2_MOE: hf_to_mcore_config_qwen2moe,
    SupportedModel.DEEPSEEK_V3: hf_to_mcore_config_dpskv3,
    SupportedModel.MIXTRAL: hf_to_mcore_config_mixtral,
    SupportedModel.QWEN2_5_VL: hf_to_mcore_config_qwen2_5_vl,
    SupportedModel.LLAMA4: hf_to_mcore_config_llama4,
    SupportedModel.QWEN3: hf_to_mcore_config_dense,
    SupportedModel.QWEN3_MOE: hf_to_mcore_config_qwen3moe,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: hf_to_mcore_config_dense,
}

# Registry for model initializers
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_INITIALIZER_REGISTRY: dict[SupportedModel, type[BaseModelInitializer]] = {
    SupportedModel.LLAMA: DenseModel,
    SupportedModel.QWEN2: DenseModel,
    SupportedModel.QWEN2_MOE: Qwen2MoEModel,
    SupportedModel.MIXTRAL: MixtralModel,
    SupportedModel.DEEPSEEK_V3: DeepseekV3Model,
    SupportedModel.QWEN2_5_VL: Qwen25VLModel,
    SupportedModel.LLAMA4: DenseModel,
    SupportedModel.QWEN3: DenseModel,
    SupportedModel.QWEN3_MOE: Qwen3MoEModel,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: DenseModel,
}

# Registry for model forward functions
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_FORWARD_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: model_forward_gen(),
    SupportedModel.QWEN2: model_forward_gen(),
    SupportedModel.QWEN2_MOE: model_forward_gen(),
    SupportedModel.MIXTRAL: model_forward_gen(),
    SupportedModel.DEEPSEEK_V3: model_forward_gen(),
    SupportedModel.LLAMA4: model_forward_gen(),
    SupportedModel.QWEN3: model_forward_gen(),
    SupportedModel.QWEN3_MOE: model_forward_gen(),
    SupportedModel.QWEN2_5_VL: model_forward_gen(True),
    SupportedModel.QWEN3_MOE_VL: model_forward_gen(True),
    SupportedModel.QWEN3_VL: model_forward_gen(True),
    SupportedModel.DEEPSEEK_V3: model_forward_gen(),
    SupportedModel.GLM4_MOE: model_forward_gen(),
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: model_forward_gen(),
}

# Registry for model forward functions
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_FORWARD_NOPAD_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: gptmodel_forward_no_padding,
    SupportedModel.QWEN2: gptmodel_forward_no_padding,
    SupportedModel.QWEN2_MOE: gptmodel_forward_no_padding,
    SupportedModel.MIXTRAL: gptmodel_forward_no_padding,
    SupportedModel.DEEPSEEK_V3: gptmodel_forward_no_padding,
    SupportedModel.QWEN2_5_VL: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_MOE_VL: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_VL: gptmodel_forward_no_padding,
    SupportedModel.LLAMA4: gptmodel_forward_no_padding,
    SupportedModel.QWEN3: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_MOE: gptmodel_forward_no_padding,
    SupportedModel.DEEPSEEK_V3: gptmodel_forward_no_padding,
    SupportedModel.GLM4_MOE: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: gptmodel_forward_no_padding,
}

# Registry for model forward functions
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_FORWARD_FUSED_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: fused_forward_model_gen(),
    SupportedModel.QWEN2: fused_forward_model_gen(),
    SupportedModel.QWEN2_MOE: fused_forward_model_gen(),
    SupportedModel.MIXTRAL: fused_forward_model_gen(),
    SupportedModel.DEEPSEEK_V3: fused_forward_model_gen(),
    SupportedModel.QWEN2_5_VL: fused_forward_model_gen(True),
    SupportedModel.QWEN3_MOE_VL: fused_forward_model_gen(True),
    SupportedModel.QWEN3_VL: fused_forward_model_gen(True),
    SupportedModel.LLAMA4: fused_forward_model_gen(),
    SupportedModel.QWEN3: fused_forward_model_gen(),
    SupportedModel.QWEN3_MOE: fused_forward_model_gen(),
    SupportedModel.DEEPSEEK_V3: fused_forward_model_gen(),
    SupportedModel.GLM4_MOE: fused_forward_model_gen(),
}

# Registry for model weight converters
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MODEL_WEIGHT_CONVERTER_REGISTRY: dict[SupportedModel, type] = {
    SupportedModel.LLAMA: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2_MOE: McoreToHFWeightConverterQwen2Moe,
    SupportedModel.MIXTRAL: McoreToHFWeightConverterMixtral,
    SupportedModel.DEEPSEEK_V3: McoreToHFWeightConverterDpskv3,
    SupportedModel.QWEN3: McoreToHFWeightConverterDense,
    SupportedModel.QWEN3_MOE: McoreToHFWeightConverterQwen3Moe,
    SupportedModel.QWEN2_5_VL: McoreToHFWeightConverterQwen2_5_VL,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: McoreToHFWeightConverterDense,
}


# [EXPLAIN] `get_supported_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_supported_model(model_type: str) -> SupportedModel:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return SupportedModel(model_type)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ValueError as err:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        supported_models = [e.value for e in SupportedModel]
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


# [EXPLAIN] `hf_to_mcore_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Convert huggingface PretrainedConfig to mcore TransformerConfig.

    Args:
        hf_config: The huggingface PretrainedConfig.
        dtype: The dtype of the model.
        **override_transformer_config_kwargs: The kwargs to override the transformer config.

    Returns:
        The mcore TransformerConfig.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MODEL_CONFIG_CONVERTER_REGISTRY[model](hf_config, dtype, **override_transformer_config_kwargs)


# [EXPLAIN] `init_mcore_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: bool = True,
    post_process: bool = None,
    *,
    share_embeddings_and_output_weights: bool = False,
    value: bool = False,
    **extra_kwargs,  # may be used for vlm and moe
) -> nn.Module:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Initialize a Mcore model.

    Args:
        tfconfig: The transformer config.
        hf_config: The HuggingFace config.
        pre_process: Optional pre-processing function.
        post_process: Optional post-processing function.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        value: Whether to use value.
        **extra_kwargs: Additional keyword arguments.

    Returns:
        The initialized model.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    initializer_cls = MODEL_INITIALIZER_REGISTRY[model]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    initializer = initializer_cls(tfconfig, hf_config)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return initializer.initialize(
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        value=value,
        **extra_kwargs,
    )


# [EXPLAIN] `get_mcore_forward_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_mcore_forward_fn(hf_config: PretrainedConfig) -> Callable:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get the forward function for given model architecture.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MODEL_FORWARD_REGISTRY[model]


# [EXPLAIN] `get_mcore_forward_no_padding_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_mcore_forward_no_padding_fn(hf_config: PretrainedConfig) -> Callable:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get the forward function for given model architecture.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MODEL_FORWARD_NOPAD_REGISTRY[model]


# [EXPLAIN] `get_mcore_forward_fused_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_mcore_forward_fused_fn(hf_config: PretrainedConfig) -> Callable:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get the forward function for given model architecture.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MODEL_FORWARD_FUSED_REGISTRY[model]


# [EXPLAIN] `get_mcore_weight_converter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_mcore_weight_converter(hf_config: PretrainedConfig, dtype: torch.dtype) -> Callable:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get the weight converter for given model architecture.
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_supported_model(hf_config.architectures[0])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tfconfig = hf_to_mcore_config(hf_config, dtype)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return MODEL_WEIGHT_CONVERTER_REGISTRY[model](hf_config, tfconfig)
