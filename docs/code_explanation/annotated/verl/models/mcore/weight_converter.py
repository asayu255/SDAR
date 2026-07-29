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

# online convert mcore weight to pure huggingface weight, no any fusion
# including format conversion and name mapping
# not including resharding
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer import TransformerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PretrainedConfig


# [EXPLAIN] `McoreToHFWeightConverterBase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterBase:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, hf_config: PretrainedConfig, mcore_config: TransformerConfig):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.mcore_config = mcore_config

    # [EXPLAIN] `convert_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> torch.Tensor:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `McoreToHFWeightConverterDense` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterDense(McoreToHFWeightConverterBase):
    # [EXPLAIN] `_convert_attention_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.self_attention.linear_proj.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight'
        # 'decoder.layers.0.self_attention.linear_qkv.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.bias'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "self_attention.linear_qkv.bias" in name or "self_attention.linear_qkv.weight" in name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_type = name.split(".")[-1]
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert param_type == "bias" or param_type == "weight"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_proj.{param_type}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_proj.{param_type}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 3
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "self_attention.linear_proj.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "self_attention.q_layernorm.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_norm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "self_attention.k_layernorm.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_norm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params

    # [EXPLAIN] `_convert_mlp_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight'
        # 'decoder.layers.0.mlp.linear_fc1.weight'
        # 'decoder.layers.0.mlp.linear_fc2.weight'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "mlp.linear_fc1.weight" in name:
            # split gate_proj and up_proj
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.gate_proj.weight")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.up_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 2
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.linear_fc1.layer_norm_weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.linear_fc2.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.down_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params

    # [EXPLAIN] `convert_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in direct_name_mapping:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [direct_name_mapping[name]], [params_one_group[0]]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "self_attention" in name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._convert_attention_param(name, params_one_group)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp" in name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._convert_mlp_param(name, params_one_group)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")


# [EXPLAIN] `McoreToHFWeightConverterQwen2Moe` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterQwen2Moe(McoreToHFWeightConverterDense):
    # [EXPLAIN] `_convert_mlp_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.pre_mlp_layernorm.weight',
        # 'decoder.layers.0.mlp.router.weight',
        # 'decoder.layers.0.mlp.shared_experts.gate_weight',
        # 'decoder.layers.0.mlp.shared_experts.linear_fc1.weight',
        # 'decoder.layers.0.mlp.shared_experts.linear_fc2.weight'
        # moe1
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight1',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight2',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight3',
        # moe2
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight1',
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "pre_mlp_layernorm" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.router.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "shared_experts.gate_weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert_gate.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "shared_experts.linear_fc1.weight" in name:  # split gate_proj and up_proj
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.up_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 2
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "shared_experts.linear_fc2.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.down_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc1" in name:  # split gate_proj and up_proj
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 2
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc2" in name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params


# [EXPLAIN] `McoreToHFWeightConverterDpskv3` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterDpskv3(McoreToHFWeightConverterBase):
    # [EXPLAIN] `_convert_attention_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # mcore
        # 'decoder.layers.0.input_layernorm.weight'
        # 'decoder.layers.0.self_attention.linear_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_proj.weight'
        # 'decoder.layers.0.self_attention.linear_kv_down_proj.weight'
        # 'decoder.layers.0.self_attention.linear_kv_up_proj.layer_norm_weight'
        # 'decoder.layers.0.self_attention.linear_kv_up_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_down_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_up_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_up_proj.layer_norm_weight'
        # hf
        # 'model.layers.0.input_layernorm.weight'
        # 'model.layers.0.self_attn.o_proj.weight'
        # 'model.layers.0.self_attn.q_proj.weight'
        # 'model.layers.0.self_attn.kv_a_proj_with_mqa.weight'
        # 'model.layers.0.self_attn.kv_a_layernorm.weight'
        # 'model.layers.0.self_attn.kv_b_proj.weight'
        # 'model.layers.0.self_attn.q_a_proj.weight'
        # 'model.layers.0.self_attn.q_b_proj.weight'
        # 'model.layers.0.self_attn.q_a_layernorm.weight'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name_map_after_layer = {
            "input_layernorm.weight": "input_layernorm.weight",
            "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
            "self_attention.linear_q_proj.weight": "self_attn.q_proj.weight",
            "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
            "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
            "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
            "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
            "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
        }
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(params) == 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name_after_layer = name.split(f".{layer_number}.")[1]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        convert_names.append(f"model.layers.{layer_number}.{name_map_after_layer[name_after_layer]}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params

    # [EXPLAIN] `_convert_mlp_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # mcore dense
        # 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight'
        # 'decoder.layers.0.mlp.linear_fc2.weight'
        # 'decoder.layers.0.mlp.linear_fc1.weight'
        #       ---
        # 'decoder.layers.1.mlp.shared_experts.linear_fc1.weight'
        #       ---
        # 'decoder.layers.1.mlp.shared_experts.linear_fc2.weight'
        # hf dense
        # 'model.layers.0.post_attention_layernorm.weight'
        # 'model.layers.0.mlp.down_proj.weight'
        # 'model.layers.0.mlp.gate_proj.weight'
        # 'model.layers.0.mlp.up_proj.weight'
        # 'model.layers.1.mlp.shared_experts.gate_proj.weight'
        # 'model.layers.1.mlp.shared_experts.up_proj.weight'
        # 'model.layers.1.mlp.shared_experts.down_proj.weight'

        # mcore moe
        # 'decoder.layers.1.pre_mlp_layernorm.weight'
        # 'decoder.layers.1.mlp.router.weight'
        # 'decoder.layers.1.mlp.router.expert_bias'
        # 'decoder.layers.1.mlp.experts.linear_fc1.weight0'
        #       ---
        # 'decoder.layers.1.mlp.experts.linear_fc2.weight0'
        # hf moe
        # 'model.layers.1.post_attention_layernorm.weight'
        # 'model.layers.1.mlp.gate.weight'
        # 'model.layers.1.mlp.gate.e_score_correction_bias'
        # 'model.layers.1.mlp.experts.0.gate_proj.weight'
        # 'model.layers.1.mlp.experts.0.up_proj.weight'
        # 'model.layers.1.mlp.experts.0.down_proj.weight'

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name_map_after_layer = {
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            "mlp.linear_fc2.weight": "mlp.down_proj.weight",
            "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
            "mlp.linear_fc1.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "mlp.shared_experts.linear_fc1.weight": ["mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"],
            "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        }
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name_after_layer = name.split(f".{layer_number}.")[1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name_after_layer in name_map_after_layer:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mapped_name = name_map_after_layer[name_after_layer]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(mapped_name, list):
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(params) == len(mapped_name)
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for one in mapped_name:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    convert_names.append(f"model.layers.{layer_number}.{one}")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(params) == 1
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                convert_names.append(f"model.layers.{layer_number}.{mapped_name}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "mlp.experts.linear_fc1.weight" in name:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                expert_id = name.split("weight")[-1]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(params) == 2
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif "mlp.experts.linear_fc2.weight" in name:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                expert_id = name.split("weight")[-1]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(params) == 1
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError(f"Unsupported parameter name: {name}")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params

    # [EXPLAIN] `convert_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in direct_name_mapping:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [direct_name_mapping[name]], [params_one_group[0]]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "self_attention" in name or "input_layernorm.weight" in name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._convert_attention_param(name, params_one_group)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp" in name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._convert_mlp_param(name, params_one_group)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")


# [EXPLAIN] `McoreToHFWeightConverterMixtral` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterMixtral(McoreToHFWeightConverterDense):
    # [EXPLAIN] `_convert_mlp_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # decoder.layers.0.mlp.router.weight
        # decoder.layers.0.mlp.experts.linear_fc1.weight0 - weight7
        # decoder.layers.0.mlp.experts.linear_fc2.weight0 - weight7

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "pre_mlp_layernorm" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.router.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.gate.weight")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc1.weight" in name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc2.weight" in name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params


# [EXPLAIN] `McoreToHFWeightConverterQwen3Moe` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class McoreToHFWeightConverterQwen3Moe(McoreToHFWeightConverterDense):
    # [EXPLAIN] `_convert_mlp_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # qwen3 moe no share expert

        # 'decoder.layers.0.pre_mlp_layernorm.weight',
        # 'decoder.layers.0.mlp.router.weight',
        # moe1
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight1',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight2',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight3',
        # moe2
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight1',
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_number = name.split(".")[2]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        convert_names = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "pre_mlp_layernorm" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.router.weight" in name:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc1" in name:  # split gate_proj and up_proj
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 2
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "mlp.experts.linear_fc2" in name:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expert_id = name.split("weight")[-1]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(params) == 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return convert_names, params
