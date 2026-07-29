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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Utilities to create common models from huggingface
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, Optional, Type

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    MistralForSequenceClassification,
    PretrainedConfig,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.registry import ModelRegistry


# [EXPLAIN] `LambdaLayer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LambdaLayer(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, fn):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.fn = fn

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, *args, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.fn(*args, **kwargs)


# [EXPLAIN] `squeeze` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def squeeze(x):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.squeeze(x, dim=-1)


# [EXPLAIN] `update_model_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def update_model_config(module_config, override_config_kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in override_config_kwargs.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(val, dict):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            update_model_config(getattr(module_config, key), val)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(module_config, key, val)


# [EXPLAIN] `get_huggingface_actor_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if override_config_kwargs is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config_kwargs = {}
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(override_config_kwargs, Dict), f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    update_model_config(module_config, override_config_kwargs)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return module_config


# [EXPLAIN] `get_generation_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return GenerationConfig.from_pretrained(model)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except OSError:  # Not found
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            config = get_huggingface_actor_config(
                model,
                trust_remote_code=trust_remote_code,
            )
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return GenerationConfig.from_model_config(config)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except OSError:  # Not found
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None


# [EXPLAIN] `create_huggingface_actor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if override_config_kwargs is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config_kwargs = {}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if automodel_kwargs is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        automodel_kwargs = {}
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(override_config_kwargs, Dict), f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module_config = get_huggingface_actor_config(model_name, override_config_kwargs, trust_remote_code=automodel_kwargs.get("trust_remote_code", False))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return module


# [EXPLAIN] `create_huggingface_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    critic_module: nn.Module = create_huggingface_actor(model_name, override_config_kwargs=override_config_kwargs, automodel_kwargs=automodel_kwargs)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if automodel_kwargs is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        automodel_kwargs = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    torch_dtype = automodel_kwargs.get("torch_dtype", torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    critic_module.lm_head = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype), LambdaLayer(fn=squeeze))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return critic_module


# [EXPLAIN] `get_model_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_model_size(model: nn.Module, scale="auto"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_params = sum(p.numel() for p in model.parameters())

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if scale == "auto":
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if n_params > 1e9:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scale = "B"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif n_params > 1e6:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scale = "M"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif n_params > 1e3:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scale = "K"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            scale = ""

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if scale == "B":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_params = n_params / 1e9
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif scale == "M":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_params = n_params / 1e6
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif scale == "K":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_params = n_params / 1e3
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif scale == "":
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Unknown scale {scale}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return n_params, scale


# [EXPLAIN] `print_model_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def print_model_size(model: nn.Module, name: str = None):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n_params, scale = get_model_size(model, scale="auto")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if name is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name = model.__class__.__name__
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"{name} contains {n_params:.2f}{scale} parameters")


# [EXPLAIN] `create_random_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_random_mask(
    input_ids: torch.Tensor,
    max_ratio_of_valid_token: float,
    max_ratio_of_left_padding: float,
    min_ratio_of_valid_token: float = 0,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Create a random mask given input_ids. Support left padding and right padding.
    Process:
    - Sample valid token length
    - Sample left_padding length
    - Generate padding

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:

    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.0
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.0
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size, sequence_length = input_ids.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: we can make this faster
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for index in range(num_left_padding):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            masks[i, index] = 0

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for index in range(num_left_padding + num_valid, sequence_length):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            masks[i, index] = 0
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return masks


# [EXPLAIN] `compute_position_id_with_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_position_id_with_mask(mask):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


# [EXPLAIN] `normalize_model_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_model_name(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Transform the model name in each model_chunk in each pp stage into the name in inference engine
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.megatron_utils import get_transformer_layer_offset

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layer_offset = get_transformer_layer_offset(pp_rank, vpp_rank, transformer_config)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if layer_name in name:  # belong to an intermediate layer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        split_name = name.split(".")
        # find the num next to split_name
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, name in enumerate(split_name):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if name == layer_name:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_num_idx = i + 1
        # check the name
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(split_name) >= layer_num_idx + 1, f"split_name = {split_name}"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert split_name[layer_num_idx].isdigit(), f"split_name = {split_name}"
        # increment layer_num_idx by layer_offset
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name = ".".join(split_name)  # weight name in inference_tp_model
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return name


# [EXPLAIN] `normalize_pp_vpp_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def normalize_pp_vpp_params(params, num_hidden_layers, layer_name="layers"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Normalize the pp vpp params into a complete named parameters.
    This is useful when gather parameters from pp ranks and passed to a model without pp

    params: Iterable[List[Dict[str, param]]]
        params contains a list of pp, with a list of vpp named_parameters in each vpp chunk.
    output: Dict[str, param]

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = len(params)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for pp_rank in range(len(params)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vpp_size = len(params[pp_rank])
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for vpp_rank in range(vpp_size):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for name, param in params[pp_rank][vpp_rank].items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                normalized_name = normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers, layer_name=layer_name)
                # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                yield normalized_name, param


# [EXPLAIN] `get_parallel_model_from_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_parallel_model_from_config(config, megatron_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import ModelParallelConfig

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(megatron_config, ModelParallelConfig)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_class = _get_parallel_model_architecture_from_config(config, value)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = model_class(
        config,
        megatron_config,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return model


# [EXPLAIN] `_get_parallel_model_architecture_from_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> Type[nn.Module]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    architectures = getattr(config, "architectures", [])
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arch in architectures:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_cls = ModelRegistry.load_model_cls(arch, value)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("after load model cls")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if model_cls is not None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return model_cls
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError(f"Model architectures {architectures} are not supported for now. Supported architectures: {ModelRegistry.get_supported_archs()}")


# [EXPLAIN] `_load_hf_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _load_hf_model(config, model_config, is_value_model, local_cache_path):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Helper function containing the loading hf model logic"""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from accelerate import init_empty_weights
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import parallel_state as mpu

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.models.mcore.saver import _megatron_calc_global_rank

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    architectures = getattr(model_config, "architectures", [])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = os.path.expanduser(local_cache_path)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.model.path.startswith("hdfs:"):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.fs import copy_to_local

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"start download from {config.model.path}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = copy_to_local(src=config.model.path, cache_dir=local_cache_path, use_shm=config.model.get('use_shm', False))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("finish download")
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_model_path = config.model.path
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"load from local dir {local_model_path}")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=mpu.get_context_parallel_rank())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cpu_init_weights = lambda: torch.device("cpu")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_context = init_empty_weights if torch.distributed.get_rank() != src_rank else cpu_init_weights
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with init_context(), warnings.catch_warnings():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        warnings.simplefilter("ignore")
        # TODO: to find a better way to load mistral7b-rm lm_head
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "mistral7b-rm" in config.model.path:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = MistralForSequenceClassification.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto",  # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )  # use score head instead of lm_head
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = model.state_dict()
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            state_dict["lm_head.weight"] = state_dict["score.weight"]
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"][:32000]  # workaround, 32001 -> 32000
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_value_model = True
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto", # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = model.state_dict()

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return architectures, model, state_dict, is_value_model


# [EXPLAIN] `load_megatron_model_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_megatron_model_weights(config, model_config, parallel_model, params_dtype, is_value_model=False, local_cache_path="~/.cache/verl/rlhf"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load weights for verl customized model."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    architectures, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model, local_cache_path)

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.models.weight_loader_registry import get_weight_loader

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"before weight loader: architectures = {architectures}...")
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arch in architectures:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"call weight loader arch = {arch}, model config = {model.config}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight_loader = get_weight_loader(arch)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        weight_loader(
            state_dict=state_dict,
            wrapped_models=parallel_model,
            config=model.config,
            params_dtype=params_dtype,
            is_value_model=is_value_model,
            tie_word_embeddings=model_config.tie_word_embeddings,
        )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return model.config


# [EXPLAIN] `load_megatron_gptmodel_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_megatron_gptmodel_weights(config, model_config, parallel_model, params_dtype, is_value_model=False, local_cache_path="~/.cache/verl/rlhf"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load weights for mcore GPT model."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model, local_cache_path)

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=parallel_model,
        config=model.config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    del state_dict, model


# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp
# [EXPLAIN] `pad_packed_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """pad the tokens such that the total length is a multiple of size.
    This function is useful when applying sequence parallel and context parallel

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding
        cu_seqlens: (total_nnz + 1,)
        max_seqlen_in_batch: int

    Returns:

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    F = nn.functional

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_nnz = unpad_tokens.shape[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_size = 0 if total_nnz % size == 0 else size - total_nnz % size

    # we assume adding a new data in the batch with seqlen pad_size
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pad_size > 0:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if unpad_tokens.ndim == 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif unpad_tokens.ndim == 2:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return unpad_tokens, cu_seqlens, max_seqlen_in_batch


# [EXPLAIN] `load_mcore_dist_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_mcore_dist_weights(parallel_model, dist_weight_path, is_value_model=False):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import dist_checkpointing
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.dist_checkpointing.serialization import StrictHandling
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.models.gpt.gpt_model import GPTModel

    # strict = StrictHandling.IGNORE_ALL if is_value_model else StrictHandling.ASSUME_OK_UNEXPECTED
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    strict = StrictHandling.ASSUME_OK_UNEXPECTED
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for model in parallel_model:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(model.module, GPTModel):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ssd = model.module.sharded_state_dict()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ssd = model.module.module.sharded_state_dict()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_value_model:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k in list(ssd.keys()):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "output_layer" in k:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    ssd.pop(k)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist_checkpointing.load(ssd, dist_weight_path, strict=strict)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return


# [EXPLAIN] `get_parallel_gptmodel_from_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_parallel_gptmodel_from_config(tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.models.gpt.gpt_model import GPTModel

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    use_te = True
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    transformer_layer_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rope_scaling_args = {}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if hf_config.rope_scaling is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling["factor"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parallel_model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
        **rope_scaling_args,
    )
    # # for layer in parallel_model.decoder.layers:
    # layer.self_attention.core_attention.flash_attention.softmax_scale = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if post_process and value:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parallel_model.output_layer = LinearForLastLayer(input_size=tfconfig.hidden_size, output_size=1, config=tfconfig)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return parallel_model
