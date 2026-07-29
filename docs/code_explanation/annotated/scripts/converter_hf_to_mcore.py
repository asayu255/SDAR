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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import dist_checkpointing
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.dist_checkpointing.serialization import StrictHandling
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.models.gpt.gpt_model import ModelType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoConfig, AutoModelForCausalLM

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.mcore import hf_to_mcore_config
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import get_model


# [EXPLAIN] `_init_args` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _init_args():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--output_path", type=str, required=True, help="The path for the output mcore model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--use_cpu_initialization", action="store_true", help="Whether to use cpu initialization")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--test", action="store_true", help="Whether to test the conversion")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args


# [EXPLAIN] `MegatronConfig` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronConfig:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.params_dtype = torch.bfloat16


# [EXPLAIN] `ModelConfig` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ModelConfig:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.path = None


# [EXPLAIN] `Config` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Config:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = ModelConfig()


# [EXPLAIN] `test_conversion` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_conversion(megatron_model_provider, tfconfig, output_path, model):
    ########### test ###########
    # load model
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_test = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
        transformer_config=tfconfig,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ref_state_dict = model_test[0].module.sharded_state_dict()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist_checkpointing.load(ref_state_dict, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dut_state_dict = model[0].module.state_dict()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name in dut_state_dict.keys():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dut_state_dict[name] is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Warning] {name} is none in dut_state_dict")
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dut_data = dut_state_dict[name].data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in ref_state_dict:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ref_data = ref_state_dict[name].data
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert dut_data.shape == ref_state_dict.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"{name} is equal")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Warning] {name} is not in ref_state_dict")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name in ref_state_dict.keys():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ref_state_dict[name] is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Warning] {name} is none in ref_state_dict")
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_data = ref_state_dict[name].data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in dut_state_dict:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dut_data = dut_state_dict[name].data
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert dut_data.shape == ref_data.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"{name} is equal")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Warning] {name} is not in dut_state_dict")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Conversion test passed!")


# [EXPLAIN] `convert_checkpoint_from_transformers_to_megatron` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_checkpoint_from_transformers_to_megatron(hf_model, model, hf_config):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_attention_heads = hf_config.num_attention_heads
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_key_value_heads = hf_config.num_key_value_heads
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden_dim = hf_config.hidden_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    head_dim = getattr(hf_config, "head_dim", hidden_dim // num_attention_heads)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_attention_heads != num_key_value_heads:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("[WARNING] Converting GQA model")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    has_qkv_bias = getattr(hf_config, "qkv_bias", False) or getattr(hf_config, "attention_bias", False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    has_share_expert = getattr(hf_config, "shared_expert_intermediate_size", None)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for layer, hf_layer in zip(model.decoder.layers, hf_model.model.layers):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.input_layernorm.weight)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q = hf_layer.self_attn.q_proj.weight.view([num_key_value_heads, head_dim * num_attention_heads // num_key_value_heads, -1])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k = hf_layer.self_attn.k_proj.weight.view([num_key_value_heads, head_dim, -1])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            v = hf_layer.self_attn.v_proj.weight.view([num_key_value_heads, head_dim, -1])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_qkv.weight.copy_(qkv)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if has_qkv_bias:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_bias = hf_layer.self_attn.q_proj.bias.view([num_key_value_heads, -1])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_bias = hf_layer.self_attn.k_proj.bias.view([num_key_value_heads, -1])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v_bias = hf_layer.self_attn.v_proj.bias.view([num_key_value_heads, -1])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(hf_layer.self_attn, "q_norm"):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.self_attention.q_layernorm.weight.copy_(hf_layer.self_attn.q_norm.weight.data)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.self_attention.k_layernorm.weight.copy_(hf_layer.self_attn.k_norm.weight.data)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, hf_expert in enumerate(hf_layer.mlp.experts):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.mlp.experts.linear_fc1._parameters[f"weight{idx}"].copy_(fc1_weight)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.mlp.experts.linear_fc2._parameters[f"weight{idx}"].copy_(hf_expert.down_proj.weight)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if has_share_expert:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.mlp.shared_experts.gate_weight.copy_(hf_layer.mlp.shared_expert_gate.weight)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shared_fc1_weight = torch.cat([hf_layer.mlp.shared_expert.gate_proj.weight, hf_layer.mlp.shared_expert.up_proj.weight])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_expert.down_proj.weight)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.output_layer.weight.copy_(hf_model.lm_head.weight)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `convert_checkpoint_from_transformers_to_megatron_dpskv3` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model, hf_config, tfconfig):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    warnings.warn("MTP model is not supported yet", stacklevel=2)

    # [EXPLAIN] `safe_copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def safe_copy(
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
        skip_dtype_assert: bool = False,
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not skip_dtype_assert:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if src_tensor.dtype != dst_tensor.dtype:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Get source dtype {src_tensor.dtype}, but target dtype {dst_tensor.dtype}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert src_tensor.shape == dst_tensor.shape
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dst_tensor.data.copy_(src_tensor.data)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return src_tensor.numel()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for layer_idx, (layer, hf_layer) in enumerate(zip(model.decoder.layers, hf_model.model.layers)):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(layer_idx)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hf_config.q_lora_rank is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_q_down_proj.weight.copy_(hf_layer.self_attn.q_a_proj.weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_q_up_proj.weight.copy_(hf_layer.self_attn.q_b_proj.weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(hf_layer.self_attn.q_a_layernorm.weight)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer.self_attention.linear_kv_down_proj.weight.copy_(hf_layer.self_attn.kv_a_proj_with_mqa.weight)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer.self_attention.linear_kv_up_proj.weight.copy_(hf_layer.self_attn.kv_b_proj.weight)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(hf_layer.self_attn.kv_a_layernorm.weight)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not hasattr(layer.mlp, "router"):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.post_attention_layernorm.weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.linear_fc1.weight.copy_(torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight]))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.linear_fc2.weight.copy_(hf_layer.mlp.down_proj.weight)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)
            # NOTE: the e_score_correction_bias in mcore model will be initialized with bfloat16 and \
            # recover to fp32 in the first forward. There is always a diff in the bias between two models (~0.3%)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            safe_copy(hf_layer.mlp.gate.e_score_correction_bias, layer.mlp.router.expert_bias, skip_dtype_assert=True)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tfconfig.moe_grouped_gemm:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    linear_fc1_weighti = getattr(layer.mlp.experts.linear_fc1, "weight" + str(i))
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    linear_fc1_weighti.copy_(fc1_weight)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    linear_fc2_weighti = getattr(layer.mlp.experts.linear_fc2, "weight" + str(i))
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    linear_fc2_weighti.copy_(hf_expert.down_proj.weight)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    expert = layer.mlp.experts.local_experts[i]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            shared_fc1_weight = torch.cat([hf_layer.mlp.shared_experts.gate_proj.weight, hf_layer.mlp.shared_experts.up_proj.weight])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_experts.down_proj.weight)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not hf_config.tie_word_embeddings:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model.output_layer.weight.copy_(hf_model.lm_head.weight)


# [EXPLAIN] `convert_hf_to_mcore` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_hf_to_mcore(hf_model_path, output_path, use_cpu_initialization=False, test=False, trust_remote_code=False):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(output_path, exist_ok=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(os.listdir(output_path)) > 0 and not test:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Output path {output_path} is not empty, skipping conversion")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # init torch distributed and mpu
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["RANK"] = "0"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["WORLD_SIZE"] = "1"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["MASTER_ADDR"] = "localhost"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["MASTER_PORT"] = "12355"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.init_process_group("nccl")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model_parallel_cuda_manual_seed(0)

    # init hf config
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(hf_config, flush=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cfg = Config()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cfg.model.path = hf_model_path
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tfconfig.use_cpu_initialization = use_cpu_initialization
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # init megatron model
    # [EXPLAIN] `megatron_model_provider` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def megatron_model_provider(pre_process, post_process):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.mcore import init_mcore_model

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parallel_model = init_mcore_model(
            tfconfig,
            hf_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=tie_word_embeddings,
            value=False,
        )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return parallel_model

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        transformer_config=tfconfig,
    )

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with warnings.catch_warnings():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.simplefilter("ignore")

    # init hf model
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_state_dict = hf_model.state_dict()

    # load hf state dict to megatron model
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "Qwen2MoeForCausalLM" in hf_config.architectures:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "DeepseekV3ForCausalLM" in hf_config.architectures:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model[0].module, hf_config, tfconfig=tfconfig)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif "Qwen3MoeForCausalLM" in hf_config.architectures:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert not use_cpu_initialization, "use_cpu_initialization is only supported for MoE model"
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        load_state_dict_to_megatron_gptmodel(
            state_dict=hf_state_dict,
            wrapped_models=model,
            config=hf_config,
            params_dtype=torch.bfloat16,
            is_value_model=False,
        )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    megatron_state_dict = model[0].module.sharded_state_dict()
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del hf_state_dict, hf_model

    # save megatron model
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(os.listdir(output_path)) == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist_checkpointing.save(megatron_state_dict, output_path, sharded_strategy=None, async_sharded_save=False)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if test:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        test_conversion(megatron_model_provider, tfconfig, output_path, model)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = _init_args()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    convert_hf_to_mcore(args.hf_model_path, args.output_path, args.use_cpu_initialization, args.test, args.trust_remote_code)
