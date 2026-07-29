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

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering unsupported issues.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SUPPORTED_MOE_MODELS = []

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from msgspec import field
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging import version as vs
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.lora.models import LoRAModel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.lora.request import LoRARequest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.lora.utils import get_adapter_absolute_path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import get_version


# [EXPLAIN] `patch_vllm_moe_model_weight_loader` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not SUPPORTED_MOE_MODELS:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    original_model_type = type(model)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = model.runnable
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    MLP_ATTR_MAPPING = {}
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ImportError:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if inner_model is None:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if type(inner_model).__name__ == "Qwen3MoeLLMForCausalLM":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for layer_idx, layer in enumerate(inner_model.layers):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mlp = getattr(layer, mlp_attr, None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not mlp:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        experts = getattr(mlp, "experts", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not experts or not hasattr(experts, "weight_loader"):
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # Patch the weight loaders
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name, param in mlp.named_parameters():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "w13_weight" in name or "w2_weight" in name:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param.weight_loader = experts.weight_loader


# [EXPLAIN] `TensorLoRARequest` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TensorLoRARequest(LoRARequest):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    peft_config:dict = field(default=None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lora_tensors:dict = field(default=None)


# [EXPLAIN] `VLLMHijack` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class VLLMHijack():
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `hijack` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def hijack():
        # [EXPLAIN] `hijack__load_adapter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to load memory-based LoRA tensors.
            """
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                supported_lora_modules = (
                    self._adapter_manager.supported_lora_modules)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                packed_modules_mapping = (
                    self._adapter_manager.packed_modules_mapping)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                expected_lora_modules: List[str] = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for module in supported_lora_modules:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if module in packed_modules_mapping:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        expected_lora_modules.extend(
                            packed_modules_mapping[module])
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        expected_lora_modules.append(module)

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                expected_lora_modules = list(set(expected_lora_modules))

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lora_tensors = None
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from vllm.lora.peft_helper import PEFTHelper
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(lora_request, TensorLoRARequest):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    peft_config = lora_request.peft_config
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora_tensors = lora_request.lora_tensors
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    peft_helper = PEFTHelper.from_dict(peft_config)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    peft_helper = PEFTHelper.from_local_dir(
                        lora_path, self.max_position_embeddings)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                model = self._adapter_manager.model
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                hf_to_vllm_mapper = None
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if (hasattr(model, "hf_to_vllm_mapper")
                        and model.hf_to_vllm_mapper is not None):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(lora_request, TensorLoRARequest):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        embeddings=None,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper
                    )
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        target_embedding_padding=self.vocab_size +
                        self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise e

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"LoRA added vocab size {lora.extra_vocab_size} "
                                f"is greater than lora_extra_vocab_size "
                                f"{self.lora_config.lora_extra_vocab_size}.")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return lora

        # [EXPLAIN] `do_hijack` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def do_hijack(target_cls, target_method_name, hooking_method):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(target_cls, target_method_name, hooking_method)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)


# [EXPLAIN] `is_version_ge` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_version_ge(pkg:str='vllm', minver:str="0.7.3"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """ check if the package version is greater than or equal to the minimum version """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return vs.parse(get_version(pkg)) >= vs.parse(minver)