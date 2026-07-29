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


# [EXPLAIN] `get_weight_loader` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_weight_loader(arch: str):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        "LlamaForCausalLM": load_state_dict_to_megatron_gptmodel,
        "Qwen2ForCausalLM": load_state_dict_to_megatron_gptmodel,
    }

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError(f"Model architectures {arch} loader are not supported for now. Supported architectures: {_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}")


# [EXPLAIN] `get_weight_saver` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_weight_saver(arch: str):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.models.mcore.saver import merge_megatron_ckpt_gptmodel, merge_megatron_ckpt_gptmodel_dpskv3, merge_megatron_ckpt_gptmodel_mixtral, merge_megatron_ckpt_gptmodel_qwen_moe

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY = {
        "LlamaForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen2ForCausalLM": merge_megatron_ckpt_gptmodel,
        "MixtralForCausalLM": merge_megatron_ckpt_gptmodel_mixtral,
        "Qwen2MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "DeepseekV3ForCausalLM": merge_megatron_ckpt_gptmodel_dpskv3,
        "Qwen3ForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen3MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
    }
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if arch in _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY[arch]
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError(f"Model architectures {arch} saver are not supported for now. Supported architectures: {_MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY.keys()}")
