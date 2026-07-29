# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# use mcore transformer config to initialize the model
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.models.gpt.gpt_model import GPTModel

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .config_converter import PretrainedConfig, TransformerConfig


# [EXPLAIN] `BaseModelInitializer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseModelInitializer(ABC):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Base class for model initializers."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tfconfig = tfconfig
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.has_vp_stage = inspect.signature(get_gpt_decoder_block_spec).parameters.get("vp_stage", None) is not None

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] `get_rope_scaling_args` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rope_scaling_args(self) -> dict:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get rope scaling args."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rope_scaling_args = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "rope_scaling" in self.hf_config:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rope_scaling_args

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vp_stage = extra_kwargs.get("vp_stage", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rope_scaling_args = self.get_rope_scaling_args()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
            **({} if not self.has_vp_stage else {"vp_stage": vp_stage}),
        )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if post_process and value:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig
            )

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model


# [EXPLAIN] `DenseModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DenseModel(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for dense models like Llama and Qwen2."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)


# [EXPLAIN] `Qwen2MoEModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen2MoEModel(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for Qwen2 MoE models."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)

        # Patch layer spec for shared experts
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(transformer_layer_spec.layer_specs)):
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return transformer_layer_spec

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = super().initialize(**kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if freeze_moe_router:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for layer in model.decoder.layers:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                layer.mlp.router.weight.requires_grad = False
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model


# [EXPLAIN] `MixtralModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MixtralModel(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for Mixtral models."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return transformer_layer_spec

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(self, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = super().initialize(**kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freeze_moe_router = kwargs.get("freeze_moe_router", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if freeze_moe_router:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for layer in model.decoder.layers:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                layer.mlp.router.weight.requires_grad = False
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model


# [EXPLAIN] `Qwen3MoEModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen3MoEModel(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for Qwen3 MoE models."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return transformer_layer_spec

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = super().initialize(**kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if freeze_moe_router:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for layer in model.decoder.layers:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                layer.mlp.router.weight.requires_grad = False
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model


# [EXPLAIN] `DeepseekV3Model` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DeepseekV3Model(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for DeepseekV3 models."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return transformer_layer_spec

    # [EXPLAIN] `get_rope_scaling_args` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rope_scaling_args(self) -> dict:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get rope scaling args."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rope_scaling_args = {}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rope_scaling_args

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(
        self,
        **kwargs,
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vp_stage = kwargs.get("vp_stage", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if freeze_moe_router:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.tfconfig.moe_router_load_balancing_type = "none"
        # MTP
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.tfconfig.mtp_num_layers is not None and self.tfconfig.mtp_num_layers > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mtp_block_spec = get_gpt_mtp_block_spec(
                self.tfconfig, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage
            )
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs["mtp_block_spec"] = mtp_block_spec

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = super().initialize(**kwargs)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if freeze_moe_router:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for layer in model.decoder.layers:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hasattr(layer.mlp, "router"):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    layer.mlp.router.weight.requires_grad = False
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model


# [EXPLAIN] `Qwen25VLModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Qwen25VLModel(BaseModelInitializer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Initializer for Qwen2.5 VL models."""

    # [EXPLAIN] `get_transformer_layer_spec` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformer_layer_spec(self, vp_stage=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return transformer_layer_spec

    # [EXPLAIN] `initialize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize(
        self,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs,
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tfconfig = self.tfconfig
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hf_config = self.hf_config
        # Qwen2_5_VLForConditionalGeneration
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from copy import deepcopy

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_layer_spec = self.get_transformer_layer_spec()

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from .qwen2_5_vl import Qwen2_5VLModel, get_vision_model_config, get_vision_projection_config

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_transformer_config = get_vision_model_config(deepcopy(tfconfig))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_transformer_config.pipeline_model_parallel_size = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_transformer_config.first_pipeline_num_layers = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_projection_config = get_vision_projection_config(
            deepcopy(tfconfig),
            vision_transformer_config.hidden_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        qwen25_vl_model = Qwen2_5VLModel(
            language_transformer_config=tfconfig,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=hf_config.vocab_size,
            language_max_sequence_length=hf_config.max_position_embeddings,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type="mlp",
            language_rotary_base=hf_config.rope_theta,
            pre_process=pre_process,
            post_process=post_process,
            add_decoder=True,
            add_encoder=True,
            parallel_output=True,
            language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if post_process and value:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            qwen25_vl_model.language_model.output_layer = LinearForLastLayer(
                input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
            )

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return qwen25_vl_model
