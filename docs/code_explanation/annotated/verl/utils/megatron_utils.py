# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""Pretrain utilities."""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gc
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.distributed import DistributedDataParallel as DDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.distributed import DistributedDataParallelConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.enums import ModelType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer import TransformerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer.module import Float16Module
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.utils import get_attr_wrapped_model
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PretrainedConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.megatron.tensor_parallel as tp_utils
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import normalize_model_name
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_dtypes import PrecisionType


# [EXPLAIN] `get_model_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_model_config(model):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return get_attr_wrapped_model(model, "config", allow_none=False)


# [EXPLAIN] `get_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Build the model."""
    # Build model.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mpu.get_pipeline_model_parallel_world_size() > 1 and mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert model_type != ModelType.encoder_and_decoder, "Interleaved schedule not supported for model with both encoder and decoder"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pre_process = mpu.is_pipeline_first_stage()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            post_process = mpu.is_pipeline_last_stage()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            this_model.model_type = model_type
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model.append(this_model)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pre_process = mpu.is_pipeline_first_stage()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        post_process = mpu.is_pipeline_last_stage()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        add_encoder = True
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        add_decoder = True
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if model_type == ModelType.encoder_and_decoder:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, "Split rank needs to be specified for model with both encoder and decoder"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rank = mpu.get_pipeline_model_parallel_rank()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                world_size = mpu.get_pipeline_model_parallel_world_size()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pre_process = rank == 0 or rank == split_rank
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                add_encoder = mpu.is_pipeline_stage_before_split()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                add_decoder = mpu.is_pipeline_stage_after_split()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = model_provider_func(pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model.model_type = model_type

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(model, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model_module in model:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param in model_module.parameters():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mpu.get_data_parallel_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model_module in model:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config: TransformerConfig = get_model_config(model[0])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config.fp8 = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tfconfig: TransformerConfig = model[0].config
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = [Float16Module(config, model_module) for model_module in model]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if wrap_with_ddp:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ddp_models = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model_chunk_idx, model_chunk in enumerate(model):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=DistributedDataParallelConfig(
                    overlap_grad_reduce=False,
                    use_distributed_optimizer=use_distributed_optimizer,
                    grad_reduce_in_fp32=True,  # [old] accumulate_allreduce_grads_in_fp32=True,
                ),
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ddp_models.append(ddp_model)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model_module in model:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model_module.broadcast_params()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return model


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


# [EXPLAIN] `unwrap_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return_list = True
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(model, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = [model]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return_list = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unwrapped_model = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model_module in model:
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while isinstance(model_module, module_instances):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_module = model_module.module
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        unwrapped_model.append(model_module)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not return_list:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return unwrapped_model[0]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return unwrapped_model


# [EXPLAIN] `convert_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_config(hf_config: PretrainedConfig, megatron_config) -> TransformerConfig:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"megatron config {megatron_config}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dt = PrecisionType.to_dtype(megatron_config.params_dtype)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"pipeline_dtype=megatron_config {dt}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    overlap_p2p_comm = mpu.get_virtual_pipeline_model_parallel_world_size() is not None and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_p2p_comm = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    transformer_config = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        #    max_position_embeddings=hf_config.max_position_embeddings,
        activation_func=F.silu,
        normalization="RMSNorm",
        #    rotary_percent=False, # default,
        gated_linear_unit=True,  # for llama
        use_cpu_initialization=True,
        apply_residual_connection_post_layernorm=False,  # check what's this mean
        add_bias_linear=False,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        pipeline_dtype=dt,
        params_dtype=dt,
        sequence_parallel=mpu.get_tensor_model_parallel_world_size() > 1,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        moe_token_dispatcher_type="alltoall",
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),
        add_qkv_bias=qkv_bias,
        bf16=dt is torch.bfloat16,
    )

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return transformer_config


# [EXPLAIN] `init_megatron_optim_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_megatron_optim_config(optim_config: Dict) -> OptimizerConfig:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = OptimizerConfig(
        optimizer="adam",
        lr=optim_config.get("lr"),
        clip_grad=optim_config.get("clip_grad"),
        weight_decay=optim_config.get("weight_decay"),
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return config


# [EXPLAIN] `mcore_model_parallel_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def mcore_model_parallel_config(
    sequence_parallel: bool,
    params_dtype: torch.dtype,
) -> ModelParallelConfig:
    # WARNING: Code should not reach this point. This function is deprecated and will be removed.
    # Please use hf_to_mcore_config_dense() from verl.models.mcore.config_converter instead.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    warnings.warn(
        "Code should not reach this point. This function is deprecated and will be removed. Please use hf_to_mcore_config_dense() from verl.models.mcore.config_converter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ModelParallelConfig(
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=True,
        fp16=False,
        timers=None,
    )


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_megatron_model_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_megatron_model_to_cpu(models):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model_chunk in models:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(model_chunk, DDP):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for buffers in model_chunk_all_buffers:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for buffer in buffers:
                    # offload parameters
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if buffer.param_data.storage().size() > 0:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        buffer.param_data_size = buffer.param_data.storage().size()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.param_data.storage().resize_(0)

                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.grad_data.storage().resize_(0)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # we need this for ref module
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _, param in model_chunk.named_parameters():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param.data = param.data.to("cpu", non_blocking=True)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if param.grad is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param.grad = param.grad.to("cpu", non_blocking=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    gc.collect()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_megatron_model_to_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_megatron_model_to_gpu(models, load_grad=True):
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model_chunk in models:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(model_chunk, DDP):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for buffers in model_chunk_all_buffers:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if load_grad:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.grad_data.zero_()

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if buffer.param_data.storage().size() == 0:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # we need this for ref module
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            device_id = torch.cuda.current_device()
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _, param in model_chunk.named_parameters():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param.data = param.data.to(device_id, non_blocking=True)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if param.grad is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param.grad = param.grad.to(device_id, non_blocking=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    gc.collect()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_megatron_copy_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_megatron_copy_params(optimizers):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    # [EXPLAIN] `_iter_opts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _iter_opts(opt):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(opt, ChainedOptimizer):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return opt.chained_optimizers
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [opt]

    # [EXPLAIN] `offload_tensor_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload_tensor_to_cpu(tensor):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor.data = tensor.data.to("cpu", non_blocking=True)

    # [EXPLAIN] `offload_group_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload_group_to_cpu(group):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if group is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(group, list):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for param_group in group:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(param_group, list):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for param in param_group:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        offload_tensor_to_cpu(param)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    offload_tensor_to_cpu(param_group)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _opt in _iter_opts(optimizers):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_megatron_copy_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_megatron_copy_params(optimizers):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    # [EXPLAIN] `_iter_opts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _iter_opts(opt):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(opt, ChainedOptimizer):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return opt.chained_optimizers
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [opt]

    # [EXPLAIN] `load_tensor_to_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_tensor_to_gpu(tensor):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device_id = torch.cuda.current_device()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    # [EXPLAIN] `load_group_to_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_group_to_gpu(group):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if group is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(group, list):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for param_group in group:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(param_group, list):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for param in param_group:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        load_tensor_to_gpu(param)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    load_tensor_to_gpu(param_group)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _opt in _iter_opts(optimizers):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_megatron_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_megatron_optimizer(optimizers):
    # [EXPLAIN] `_iter_opts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _iter_opts(opt):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(opt, ChainedOptimizer):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return opt.chained_optimizers
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [opt]

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _opt in _iter_opts(optimizers):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        offload_megatron_copy_params(_opt)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        opt_state_dict_values = _opt.optimizer.state.values()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for v in opt_state_dict_values:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "exp_avg" in v:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "exp_avg_sq" in v:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        gc.collect()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_megatron_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_megatron_optimizer(optimizers):
    # [EXPLAIN] `_iter_opts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _iter_opts(opt):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(opt, ChainedOptimizer):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return opt.chained_optimizers
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [opt]

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _opt in _iter_opts(optimizers):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        load_megatron_copy_params(_opt)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        opt_state_dict_values = _opt.optimizer.state.values()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for v in opt_state_dict_values:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "exp_avg" in v:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                v["exp_avg"] = v["exp_avg"].to(torch.cuda.current_device(), non_blocking=True)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "exp_avg_sq" in v:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                v["exp_avg_sq"] = v["exp_avg_sq"].to(torch.cuda.current_device(), non_blocking=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        gc.collect()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()


# [EXPLAIN] `print_rank_0` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def print_rank_0(message):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """If distributed is initialized, print only on rank 0."""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.is_initialized():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(message, flush=True)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(message, flush=True)


# [EXPLAIN] `get_model_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_model_checkpoint_path(checkpoint_path):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(checkpoint_path, exist_ok=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(checkpoint_path, "model")


# [EXPLAIN] `get_hf_model_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_hf_model_checkpoint_path(checkpoint_path):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(checkpoint_path, exist_ok=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(checkpoint_path, "huggingface")


# [EXPLAIN] `get_hf_config_and_tokenizer_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_hf_config_and_tokenizer_checkpoint_path(checkpoint_path):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(checkpoint_path, exist_ok=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(checkpoint_path, "hf_config_and_tokenizer")


# [EXPLAIN] `get_optimizer_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_optimizer_checkpoint_path(checkpoint_path, use_distributed_optimizer=True):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(os.path.join(checkpoint_path, "optim"), exist_ok=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not use_distributed_optimizer:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return os.path.join(checkpoint_path, "optim", "optim.pt")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_rank = mpu.get_tensor_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_rank = mpu.get_context_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_rank = mpu.get_data_parallel_rank()
    # TODO: support ep
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(checkpoint_path, "optim", f"distrib_optim_pp{pp_rank}_tp{tp_rank}_cp{cp_rank}_dp{dp_rank}.pt")


# [EXPLAIN] `get_rng_states_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_rng_states_checkpoint_path(checkpoint_path, only_rank0_save=True):
    # save rng states cause interrupts
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(os.path.join(checkpoint_path, "rng_states"), exist_ok=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if only_rank0_save:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return os.path.join(checkpoint_path, "rng_states", "rng_states.pt")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_rank = mpu.get_data_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_rank = mpu.get_tensor_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_rank = mpu.get_context_parallel_rank()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(checkpoint_path, "rng_states", f"rng_states_pp{pp_rank}_tp{tp_rank}_cp{cp_rank}_dp{dp_rank}.pt")


# [EXPLAIN] `convert_megatron_model_to_transformers_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_megatron_model_to_transformers_model(
    name,
    param,
    config: PretrainedConfig,
    tp_size: int,
    num_query_groups: int,
    convert_qkv_gate_up_by_trunk_concat=False,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Convert megatron model to transformers model."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_params = {}

    # [EXPLAIN] `convert_qkv_shard` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_qkv_shard(full_tensor, q_name, k_name, v_name):
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal config
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal tp_size
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal num_query_groups

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        q_shard_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        k_shard_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        v_shard_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_size_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.num_key_value_heads >= tp_size:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_size = q_size_tp + 2 * kv_size_tp
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(tp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_query_groups_per_partition = num_query_groups // tp_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    q_part = qkv_part_chunk[:q_size_chunk]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    q_shard_list.append(q_part)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    k_shard_list.append(k_part)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    v_shard_list.append(v_part)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_size_tp = hidden_size_per_head
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_size = q_size_tp + 2 * kv_size_tp
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(tp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_query_groups_per_partition = num_query_groups // tp_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    q_part = qkv_part_chunk[:q_size_chunk]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    q_shard_list.append(q_part)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if i * config.num_key_value_heads % tp_size == 0:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        k_shard_list.append(k_part)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        v_shard_list.append(v_part)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_params[q_name] = torch.cat(q_shard_list, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_params[k_name] = torch.cat(k_shard_list, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_params[v_name] = torch.cat(v_shard_list, dim=0)

    # [EXPLAIN] `convert_gate_up_shard` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_gate_up_shard(full_tensor, gate_name, up_name):
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal config
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal tp_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        intermediate_size_tp = config.intermediate_size // tp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate_weight_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        up_weight_list = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(tp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            gate_weight_list.append(gate_weight_tp)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            up_weight_list.append(up_weight_tp)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_params[gate_name] = torch.cat(gate_weight_list, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_params[up_name] = torch.cat(up_weight_list, dim=0)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if name == "embedding.word_embeddings.weight":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        new_params["model.embed_tokens.weight"] = param
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "self_attention" in name:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splitted_name = name.split(".")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = splitted_name[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        component = splitted_name[4]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param_type = splitted_name[5]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if component == "linear_proj":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.self_attn.o_proj.weight"] = param
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif component == "linear_qkv" and not isinstance(param, list):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if param_type == "layer_norm_weight":
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                new_params[f"model.layers.{layer_number}.input_layernorm.weight"] = param
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if convert_qkv_gate_up_by_trunk_concat:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    convert_qkv_shard(
                        param,
                        f"model.layers.{layer_number}.self_attn.q_proj.{param_type}",
                        f"model.layers.{layer_number}.self_attn.k_proj.{param_type}",
                        f"model.layers.{layer_number}.self_attn.v_proj.{param_type}",
                    )
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_params[f"model.layers.{layer_number}.self_attn.qkv_proj.{param_type}"] = param
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif component == "q_layernorm" or component == "k_layernorm":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hf_component = component.replace("layer", "")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.self_attn.{hf_component}.weight"] = param
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(param, list) and len(param) == 3
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert param_type == "weight" or param_type == "bias"
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.self_attn.q_proj.{param_type}"] = param[0]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.self_attn.k_proj.{param_type}"] = param[1]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.self_attn.v_proj.{param_type}"] = param[2]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "mlp" in name:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splitted_name = name.split(".")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = splitted_name[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        component = splitted_name[4]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param_type = splitted_name[5]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if component == "linear_fc1" and not isinstance(param, list):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if param_type == "layer_norm_weight":
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                new_params[f"model.layers.{layer_number}.post_attention_layernorm.weight"] = param
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif param_type == "weight":
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if convert_qkv_gate_up_by_trunk_concat:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    convert_gate_up_shard(
                        param,
                        f"model.layers.{layer_number}.mlp.gate_proj.weight",
                        f"model.layers.{layer_number}.mlp.up_proj.weight",
                    )
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    new_params[f"model.layers.{layer_number}.mlp.gate_up_proj.weight"] = param
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif component == "linear_fc1" and isinstance(param, list):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(param) == 2
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert param_type == "weight" or param_type == "bias"
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.mlp.gate_proj.weight"] = param[0]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.mlp.up_proj.weight"] = param[1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif component == "linear_fc2":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            new_params[f"model.layers.{layer_number}.mlp.down_proj.weight"] = param
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif name == "decoder.final_layernorm.weight":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        new_params["model.norm.weight"] = param
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif name == "output_layer.weight":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        new_params["lm_head.weight"] = param
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Unknown param name: {name}")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return new_params.keys(), new_params.values()


# [EXPLAIN] `broadcast_from_megatron_pp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def broadcast_from_megatron_pp(tensor: torch.Tensor):
    # tensor is not None only in one of the pp ranks
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tensor is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        shape = tensor.shape
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dtype = tensor.dtype
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        partition_dim = getattr(tensor, "partition_dim", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_spec = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_gather_object(object_list=tensor_spec_output, obj=tensor_spec, group=mpu.get_pipeline_model_parallel_group())
    # find the src rank
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target_tensor_spec = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_rank = None
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for rank, tensor_spec in enumerate(tensor_spec_output):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor_spec is not None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if target_tensor_spec is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                target_tensor_spec = tensor_spec
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("A tensor exists on two pp ranks")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            src_rank = rank
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert target_tensor_spec is not None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tensor is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor = torch.empty(size=target_tensor_spec[0], dtype=target_tensor_spec[1], device=torch.cuda.current_device())
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if target_tensor_spec[2] is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor.tensor_model_parallel = target_tensor_spec[2]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if target_tensor_spec[3] is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor.partition_dim = target_tensor_spec[3]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_rank = torch.distributed.get_global_rank(group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.broadcast(tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group())
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor


# [EXPLAIN] `broadcast_str_from_megatron_pp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def broadcast_str_from_megatron_pp(obj: Any):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obj_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_gather_object(object_list=obj_output, obj=obj, group=mpu.get_pipeline_model_parallel_group())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_rank = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target_obj = None
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for rank, item in enumerate(obj_output):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if item is not None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if target_obj is not None:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("An object exists on two pp ranks")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            target_obj = item
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            src_rank = rank

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert target_obj is not None, "No valid object found to broadcast."

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_rank = torch.distributed.get_global_rank(group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obj_output = [None] * torch.distributed.get_world_size(group=mpu.get_pipeline_model_parallel_group())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obj_output[0] = target_obj
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.broadcast_object_list(object_list=obj_output, src=global_rank, group=mpu.get_pipeline_model_parallel_group())

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return obj_output[0]


# [EXPLAIN] `default_tp_concat_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def default_tp_concat_fn(layer_name_mapping, name, train_params, infer_params, model_config, convert_qkv_gate_up_by_simple_split=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    name: name of the parameter
    train_params: training parameters
    infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered from micro_dp_group
    model_config: huggingface model_config
    TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
    definition so that it is model-agnostic. If the model doesn't implement this function,
    we can throw an error to force user disable TP HybridEngine.
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import mpu

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
        # if the tensor is qkv, for each param on tp, split into q, k, v
        # concat q, k, v separately.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        q_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        k_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        v_lst = []
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for infer_param in infer_params:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_query_groups_per_partition = model_config.num_key_value_heads // mpu.get_tensor_model_parallel_world_size()
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for chunk in infer_param.chunk(num_query_groups_per_partition):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                split_size = [kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition, kv_size_per_tp // num_query_groups_per_partition, kv_size_per_tp // num_query_groups_per_partition]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q, k, v = chunk.split(split_size)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                q_lst.append(q)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                k_lst.append(k)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                v_lst.append(v)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        q = torch.cat(q_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        k = torch.cat(k_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        v = torch.cat(v_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif layer_name_mapping.get("gate_proj_layer_name") in name:
        # if the tensor is gate and proj
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate_lst = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        up_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for infer_param in infer_params:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate, up = infer_param.chunk(2)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            gate_lst.append(gate)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            up_lst.append(up)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate = torch.cat(gate_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        up = torch.cat(up_lst, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "mlp.experts.linear_fc2.weight" in name:  # moe
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infer_params = torch.cat(infer_params, dim=1)

    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # concat tensor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(train_params))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return infer_params


# [EXPLAIN] `per_tensor_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def per_tensor_generator(actor_module, model_config, weight_converter, transformer_config, layer_name_mapping, convert_qkv_gate_up_by_simple_split=True):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import parallel_state as mpu

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_size = mpu.get_expert_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    etp_size = mpu.get_expert_tensor_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_group = mpu.get_expert_model_parallel_group()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    etp_group = mpu.get_expert_tensor_parallel_group()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vpp_size = len(actor_module)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_gather_group = mpu.get_tensor_model_parallel_group()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

    # [EXPLAIN] `tensor_generator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_generator():
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for scan_vpp_idx in range(vpp_size):
            # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
            yield from actor_module[scan_vpp_idx].named_parameters()

    # we need first make all rank get full model information
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    meta_info = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for scan_vpp_idx in range(vpp_size):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx, (name, _) in enumerate(actor_module[scan_vpp_idx].named_parameters()):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_gather_object(object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gen_func = tensor_generator()

    # lazy load tensor for full model
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if model_config.tie_word_embeddings and ("output_layers" in name):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import warnings

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            warnings.warn("Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2)
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cur_pp_rank == pp_rank:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cur_name, cur_tensor = next(gen_func)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except StopIteration:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cur_name, cur_tensor = None, None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_tensor, cur_name = None, None

        # pp broadcast model tensor and name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_name = broadcast_str_from_megatron_pp(cur_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

        # (xya): this is a hack to fix the name of the parameters
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while cur_name.startswith("module."):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_name = cur_name[len("module.") :]

        # EP
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ".mlp.experts.linear_fc" in cur_name and ep_size > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_experts = weight_converter.mcore_config.num_moe_experts
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_experts_per_rank = num_experts // ep_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.all_gather(infer_params, broad_pp_tensor, group=ep_group)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            name_prefix, local_expert_id = cur_name.split(".weight")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_expert_id = int(local_expert_id)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(ep_size)]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for name, param in zip(global_expert_names, infer_params):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if etp_size > 1:
                    # gather etp
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    etp_params = [torch.empty_like(param) for _ in range(etp_size)]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    torch.distributed.all_gather(etp_params, param, group=etp_group)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    params = etp_params
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    params = [param]

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                merge_params = default_tp_concat_fn(layer_name_mapping, name, broad_pp_tensor, params, model_config, convert_qkv_gate_up_by_simple_split)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not isinstance(merge_params, list):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    merge_params = [merge_params]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                converted_names, converted_params = weight_converter.convert_param(name, merge_params)

                # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                yield from zip(converted_names, converted_params)
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # tp all gather
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
            # allocate a new tensor with proper size
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if all_gather_group_size <= 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                infer_params = [broad_pp_tensor]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_params = default_tp_concat_fn(layer_name_mapping, cur_name, broad_pp_tensor, infer_params, model_config, convert_qkv_gate_up_by_simple_split)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_params = broad_pp_tensor

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(infer_params, list):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_params = [infer_params]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)

        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield from zip(converted_names, converted_params)


# [EXPLAIN] `get_transformer_layer_offset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_transformer_layer_offset(pipeline_rank, vp_rank, config: TransformerConfig):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    '''
    Get the index offset of any pipeline stage, given the level of pipelining.

    Make pp_rank and vpp_rank as two arguments to make it more flexible,
    which is able to fetch layer offset for any pipeline stage.
    The original function only returns the layer offset for current pipeline stage.

    Extension to https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_layer.py::get_transformer_layer_offset"""
    '''
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.pipeline_model_parallel_size > 1:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            middle_pipeline_stages = config.pipeline_model_parallel_size
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layers_in_first_pipeline_stage = 0 if config.num_layers_in_first_pipeline_stage is None else config.num_layers_in_first_pipeline_stage
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layers_in_last_pipeline_stage = 0 if config.num_layers_in_last_pipeline_stage is None else config.num_layers_in_last_pipeline_stage

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            middle_num_layers = config.num_layers - num_layers_in_first_pipeline_stage - num_layers_in_last_pipeline_stage

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = 0 if config.num_layers_in_first_pipeline_stage is None else config.num_layers_in_first_pipeline_stage // vp_size

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = 0 if config.num_layers_in_last_pipeline_stage is None else config.num_layers_in_last_pipeline_stage // vp_size

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = middle_num_layers // vp_size

                # First stage + middle stage + last stage
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                total_virtual_chunks = num_layers_per_virtual_model_chunk_in_first_pipeline_stage + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage + num_layers_per_virtual_model_chunk_in_last_pipeline_stage

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if pipeline_rank == 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    offset = vp_rank * total_virtual_chunks
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    offset = vp_rank * total_virtual_chunks + num_layers_per_virtual_model_chunk_in_first_pipeline_stage + (pipeline_rank - 1) * (num_layers_per_vritual_model_chunk_in_middle_pipeline_stage // middle_pipeline_stages)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if middle_pipeline_stages > 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    num_layers_per_pipeline_rank = 0

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                middle_pipeline_rank = pipeline_rank if config.num_layers_in_first_pipeline_stage is None else pipeline_rank - 1

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if pipeline_rank == 0:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    offset = 0
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    offset = (middle_pipeline_rank * num_layers_per_pipeline_rank) + num_layers_in_first_pipeline_stage
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.account_for_embedding_in_pipeline_split:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                num_layers += 1

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if config.account_for_loss_in_pipeline_split:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                num_layers += 1

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                total_virtual_chunks = num_layers // vp_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if config.account_for_embedding_in_pipeline_split and not mpu.is_pipeline_first_stage():
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    offset -= 1
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if config.account_for_embedding_in_pipeline_split and not mpu.is_pipeline_first_stage():
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    offset -= 1
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        offset = 0
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return offset
