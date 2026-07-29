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

# convert huggingface config to mcore transformer config


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import TypeVar

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PretrainedConfig

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
T = TypeVar("T", bound=TransformerConfig)


# [EXPLAIN] `_get_base_transformer_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_base_transformer_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> dict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create a base TransformerConfig with common parameters across different model architectures.
    TODO: (ycl) use dataclass or converter config?

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

    Returns:
        TransformerConfig with common parameters
    """

    # Common parallel state parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_p2p_comm = False

    # Base configuration with common parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "add_bias_linear": True,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_context_parallel_world_size(),
        "overlap_p2p_comm": overlap_p2p_comm,
        "batch_p2p_comm": batch_p2p_comm,
        "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_config.update(override_transformer_config_kwargs)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return base_config


# [EXPLAIN] `_get_mla_transformer_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_mla_transformer_config(
    hf_config: PretrainedConfig, mla_rope_config: dict, dtype: torch.dtype, **override_transformer_config_kwargs
) -> dict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create a MLATransformerConfig with common parameters across different model architectures.
    This is specifically for MLA models like DeepseekV3.

    Args:
        hf_config: HuggingFace model configuration
        mla_rope_config: MLA specific RoPE configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

    Returns:
        MLATransformerConfig with common parameters
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    base_config = _get_base_transformer_config(hf_config=hf_config, dtype=dtype, **override_transformer_config_kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mla_config = {
        # MLA specific parameters
        "q_lora_rank": hf_config.q_lora_rank,
        "kv_lora_rank": hf_config.kv_lora_rank,
        "qk_head_dim": hf_config.qk_nope_head_dim,
        "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
        "v_head_dim": hf_config.v_head_dim,
        "rotary_base": hf_config.rope_theta,
        "rotary_scaling_factor": mla_rope_config["factor"],
        "rope_type": mla_rope_config["type"],
        "max_position_embeddings": mla_rope_config["original_max_position_embeddings"],
        "beta_fast": mla_rope_config["beta_fast"],
        "beta_slow": mla_rope_config["beta_slow"],
        "mscale": mla_rope_config["mscale"],
        "mscale_all_dim": mla_rope_config["mscale_all_dim"],
    }

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_config.update(mla_config)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return base_config


# [EXPLAIN] `check_and_construct_configs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def check_and_construct_configs(original_config: dict, cls: type[T]) -> T:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Check and disable incompatible configurations for older Megatron version.

    Args:
        original_config (dict): The original model configuration.

    Returns:
        dict: The updated model configuration with incompatible settings disabled.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    removed_keys = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in original_config.keys():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not hasattr(cls, key):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            removed_keys.append(key)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if removed_keys:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.warn(
            f"The following keys are not supported in the current Megatron version and will be removed: {removed_keys}",
            stacklevel=2,
        )
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in removed_keys:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            original_config.pop(key)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    original_config = mapping_string_to_attn_backend(original_config)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Overridden {cls.__name__} init config: {original_config}")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return cls(**original_config)


# [EXPLAIN] `hf_to_mcore_config_dense` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_dense(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # for LlamaForCausalLM or Qwen2ForCausalLM
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    qkv_bias = True if "Qwen2" in hf_config.architectures[0] else getattr(hf_config, "attention_bias", False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    qk_layernorm = True if "Qwen3" in hf_config.architectures[0] else False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=qkv_bias,
        qk_layernorm=qk_layernorm,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return check_and_construct_configs(args, TransformerConfig)


# [EXPLAIN] `hf_to_mcore_config_qwen2moe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_qwen2moe(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_shared_expert_overlap=True,
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=True,
        add_qkv_bias=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return check_and_construct_configs(args, TransformerConfig)


# [EXPLAIN] `hf_to_mcore_config_mixtral` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_mixtral(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        num_moe_experts=hf_config.num_local_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_router_score_function="softmax",
        moe_shared_expert_intermediate_size=None,  # mixtral has no shared expert
        moe_shared_expert_overlap=False,  # mixtral has no shared expert
        moe_ffn_hidden_size=hf_config.intermediate_size,
        moe_router_bias_update_rate=0.001,
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        # Other optimizations
        persist_layer_norm=True,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return check_and_construct_configs(args, TransformerConfig)


# [EXPLAIN] `hf_to_mcore_config_qwen3moe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_qwen3moe(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=False,
        qk_layernorm=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return check_and_construct_configs(args, TransformerConfig)


# [EXPLAIN] `hf_to_mcore_config_dpskv3` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_dpskv3(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> MLATransformerConfig:
    # DeepseekV3ForCausalLM
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.transformer.enums import AttnBackend

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from .patch_v012 import apply_patch

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    apply_patch()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mla_rope_config = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "rope",
    }
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        mla_rope_config.update(hf_config.rope_scaling)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    moe_layer_freq = [1] * hf_config.num_hidden_layers
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(min(hf_config.first_k_dense_replace, hf_config.num_hidden_layers)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        moe_layer_freq[i] = 0

    # disable MTP and quantization for now
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "num_nextn_predict_layers" in hf_config:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert hf_config.num_nextn_predict_layers == 0, (
            "MTP is not supported for now, please modify the config.json to set num_nextn_predict_layers to 0"
        )
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "quantization_config" not in hf_config or not hf_config.quantization_config, (
        "quantization is not supported for now, please modify the config.json to remove quantization_config"
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args: dict = _get_mla_transformer_config(
        hf_config=hf_config,
        mla_rope_config=mla_rope_config,
        dtype=dtype,
        # Additional parameters
        use_cpu_initialization=False,
        add_bias_linear=False,
        attention_backend=AttnBackend.fused,
        qk_layernorm=True,
        # Standard MoE parameters
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_bias_update_rate=0.001,
        moe_router_enable_expert_bias=True,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.n_routed_experts,
        moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        moe_aux_loss_coeff=getattr(hf_config, "aux_loss_alpha", 0.001),
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_overlap=True,
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_layer_freq=moe_layer_freq,
        # mcore 0.12 moe
        moe_router_dtype="fp64",
        disable_bf16_reduced_precision_matmul=True,
        # Other optimizations
        # deallocate_pipeline_outputs=True,
        # gradient_accumulation_fusion=True,
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    transformer_config = check_and_construct_configs(args, MLATransformerConfig)
    # MTP
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "num_nextn_predict_layers" in hf_config:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_config.mtp_num_layers = hf_config.num_nextn_predict_layers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_config.mtp_loss_scaling_factor = 0.1

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return transformer_config


# [EXPLAIN] `hf_to_mcore_config_qwen2_5_vl` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_qwen2_5_vl(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        add_bias_linear=False,
        # qwen specific
        add_qkv_bias=True,
        mrope_section=hf_config.rope_scaling["mrope_section"],
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    args.update(override_transformer_config_kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = mapping_string_to_attn_backend(args)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return TransformerConfig(**args)


# [EXPLAIN] `hf_to_mcore_config_llama4` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hf_to_mcore_config_llama4(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")


# [EXPLAIN] `mapping_string_to_attn_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def mapping_string_to_attn_backend(args: dict) -> dict:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "attention_backend" in args and isinstance(args["attention_backend"], str):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.transformer.enums import AttnBackend

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        args["attention_backend"] = AttnBackend[args["attention_backend"]]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args
