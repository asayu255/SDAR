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
import functools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import contextmanager, nullcontext
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import OrderedDict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging import version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed import DeviceMesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp._runtime_utils import _lazy_init
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.trainer_pt_utils import get_module_class_from_name
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_torch_device, get_device_name

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if version.parse(torch.__version__) >= version.parse("2.6"):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
elif version.parse(torch.__version__) >= version.parse("2.4"):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
# [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
else:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = None, None, None, None


# [EXPLAIN] `init_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_fn(x: torch.nn.Module):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.get_rank() != 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = x.to_empty(device=get_torch_device().current_device(), recurse=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x


# [EXPLAIN] `get_init_weight_context_manager` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from accelerate import init_empty_weights

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cpu_init_weights = lambda: torch.device("cpu")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if use_meta_tensor:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mesh is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        init_context = cpu_init_weights
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
# [EXPLAIN] `get_fsdp_wrap_policy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this once we have make all config in verl from OmegaConf to data class.
    # [EXPLAIN] `_get_attr` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_attr(attr_name, default_value=None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hasattr(config, "get"):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return config.get(attr_name, default_value)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return config.__getattribute__(attr_name)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _get_attr("disable", False):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_transformer_layer_cls_to_wrap = _get_attr("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_num_params = _get_attr("min_num_params", 0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    auto_wrap_policy = None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    policies = []

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_lora:

        # [EXPLAIN] `lambda_policy_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def lambda_policy_fn(module):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return bool(len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and module.weight.requires_grad)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        policies.append(lambda_policy)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if min_num_params > 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        policies.append(size_policy)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_cls_to_wrap = set()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            transformer_cls = get_module_class_from_name(module, layer_class)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if transformer_cls is None:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                transformer_cls_to_wrap.add(transformer_cls)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        policies.append(transformer_policy)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(policies) > 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return auto_wrap_policy


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_fsdp_model_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fsdp_version(model) == 2:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        offload_fsdp2_model_to_cpu(model, empty_cache)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _lazy_init(model, model)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert model._is_root, "Only support root model offloading to CPU"
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for handle in model._all_handles:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if handle._offload_params:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_param = handle.flat_param
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr() and id(flat_param.data) != id(flat_param._local_shard) and flat_param.data.size() == flat_param._local_shard.size()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_param._local_shard = flat_param.data
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert id(flat_param._local_shard) != id(flat_param.data)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if empty_cache:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_fsdp2_model_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for param in model.parameters():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param.data = param.data.to(torch.device("cpu"), non_blocking=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if empty_cache:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_fsdp_model_to_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_fsdp_model_to_gpu(model: FSDP):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fsdp_version(model) == 2:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        load_fsdp2_model_to_gpu(model)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _lazy_init(model, model)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert model._is_root, "Only support root model loading to GPU"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_id = get_torch_device().current_device()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for handle in model._all_handles:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if handle._offload_params:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_param = handle.flat_param
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_param._local_shard = flat_param.data


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_fsdp2_model_to_gpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_fsdp2_model_to_gpu(model):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device = torch.cuda.current_device()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for param in model.parameters():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param.data = param.data.to(device, non_blocking=True)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `offload_fsdp_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def offload_fsdp_optimizer(optimizer):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not optimizer.state:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for param_group in optimizer.param_groups:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param in param_group["params"]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state = optimizer.state[param]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, value in state.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(value, torch.Tensor):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    state[key] = value.to("cpu", non_blocking=True)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@torch.no_grad()
# [EXPLAIN] `load_fsdp_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_fsdp_optimizer(optimizer, device_id):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not optimizer.state:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for param_group in optimizer.param_groups:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param in param_group["params"]:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state = optimizer.state[param]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, value in state.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(value, torch.Tensor):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    state[key] = value.to(device_id, non_blocking=True)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@contextmanager
# [EXPLAIN] `meta_device_init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def meta_device_init():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device = torch.device("meta")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    old_register_parameter = nn.Module.register_parameter
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    registered = set()

    # [EXPLAIN] `register_empty_parameter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def register_empty_parameter(module, name, param):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if param is not None and param not in registered:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param_cls = type(module._parameters[name])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs = module._parameters[name].__dict__
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs["requires_grad"] = param.requires_grad
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            registered.add(module._parameters[name])

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nn.Module.register_parameter = register_empty_parameter
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield
    # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
    finally:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        registered.clear()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nn.Module.register_parameter = old_register_parameter


# [EXPLAIN] `parallel_load_safetensors` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parallel_load_safetensors(filepath):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Parallel load safetensors from huggingface checkpoint

    Huggingface checkpoint contains:

    - config.json: a json file for model configuration
    - model.safetensor.index.json: a json file for safetensors (parameters & buffers) index
    - model-000x-of-ooxx.safetensors: a binary file for safetensors (parameters & buffers) chunks

    Or (when model is small),

    - model.safetensors: a binary file for all parameters and buffers

    Each rank will own a part of model chunks and load them directly into GPU memory.
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from safetensors.torch import load_file

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    safetensors2param = {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    index_file = os.path.join(filepath, "model.safetensors.index.json")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if os.path.exists(index_file):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        index = json.load(open(index_file, "rb"))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param_name, filename in index["weight_map"].items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            safetensors2param.setdefault(filename, []).append(param_name)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # in this case, the model is small and we can load it all at once
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param_file = os.path.join(filepath, "model.safetensors")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        states = load_file(param_file)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param_name in states:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del states

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_files = len(safetensors2param)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ckpt_chunks = sorted(safetensors2param.keys())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size = dist.get_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    size = int(math.ceil(total_files / world_size))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ckpt_chunks = [ckpt_chunks[rank * size : rank * size + size] for rank in range(world_size)]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shard_states = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device = get_torch_device().current_device()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for rank, files in enumerate(ckpt_chunks):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rank == dist.get_rank():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for file in files:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                file = os.path.join(filepath, file)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                shard_states.update(states)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for file in files:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for param_name in safetensors2param[file]:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    shard_states[param_name] = rank
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return shard_states


# [EXPLAIN] `parallel_init_module_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def parallel_init_module_fn(module: torch.nn.Module, shard_states: Dict[str, torch.nn.Parameter]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Generate a function to initialize sub-modules in the `module` with `shard_states`
    from huggingface checkpoint.

    Args:
        module (torch.nn.Module): the global module to be initialized
        shard_states (Dict[str, torch.nn.Parameter]): the shard states from huggingface checkpoint

    Returns:
        init_fn (Callable): a function to initialize sub-modules in the `module` with `shard_states`
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state2fqn = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, state in itertools.chain(module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    materialized_states = {}

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad()
    # [EXPLAIN] `create_and_sync_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def create_and_sync_state(param_name, state, is_param):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert param_name in shard_states, f"{param_name} not loaded"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = get_torch_device().current_device()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_param:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:  # buffer
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param = torch.empty_like(state.data, device=device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loaded = shard_states[param_name]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
            # NOTE: loaded.dtype can be different with param.dtype
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            param.data.copy_(loaded.data)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(param.data, src=dist.get_rank())
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(loaded, int)  # the rank that holds the state
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(param.data, src=loaded)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        shard_states.pop(param_name)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del loaded
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return param

    # [EXPLAIN] `init_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name, state in param_and_buffers:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not state.is_meta:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_param = name in sub_mod._parameters
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (not is_param) and fqn not in shard_states:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if state.is_meta:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise RuntimeError(f"find a non-persistent buffer ({fqn}) initiated with device meta. Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device.")
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # for shared parameter, we get it from the first time it is created
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if state in shared:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if state not in materialized_states:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if fqn in shard_states:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        shard_states.pop(fqn)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                materialize_state = create_and_sync_state(fqn, state, is_param)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_param:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                sub_mod._parameters[name] = materialize_state
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                sub_mod._buffers[name] = materialize_state
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if recurse:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for module in sub_mod.children():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sub_mod

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return init_fn


# [EXPLAIN] `fsdp_version` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fsdp_version(model):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(model, FSDP):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 1
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(model, FSDPModule):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 2
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 0


# [EXPLAIN] `get_fsdp_state_ctx` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fsdp_version(model) == 1:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return nullcontext()


# [EXPLAIN] `fsdp2_load_full_state_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dist.get_rank() == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = model.to_empty(device=torch.cuda.current_device())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cpu_offload = cpu_offload is not None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, buf in model.named_buffers():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast(buf, src=0)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if cpu_offload:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.to("cpu", non_blocking=True)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for buf in model.buffers():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            buf.data = buf.data.to(torch.cuda.current_device())


# [EXPLAIN] `apply_fsdp2` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def apply_fsdp2(model, fsdp_kwargs, config):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """model: AutoModelForCausalLM"""
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    modules = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, module in model.named_modules():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            modules.append(module)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for idx, module in enumerate(modules):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        fully_shard(module, **fsdp_kwargs)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


# [EXPLAIN] `fsdp2_clip_grad_norm_` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(parameters, torch.Tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parameters = [parameters]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # prevent generators from being exhausted
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parameters = list(parameters)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    grads = [p.grad for p in parameters if p.grad is not None]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return total_norm

# [EXPLAIN] `layered_summon_lora_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def layered_summon_lora_params(fsdp_module)->OrderedDict:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from peft.utils.save_and_load import get_peft_model_state_dict

    # [EXPLAIN] `__prefix_submodules` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __prefix_submodules(module, prefix):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name, submodule in module.named_modules():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if name.startswith(prefix) and "." not in name[len(prefix):]:
                # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                yield name, submodule

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lora_params = OrderedDict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefix_list = [
        '_fsdp_wrapped_module.base_model.model.',
        '_fsdp_wrapped_module.base_model.model.model.',
        '_fsdp_wrapped_module.base_model.model.model.layers.'
    ]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for prefix in prefix_list:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.","base_model.model.")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if name.endswith('.model') or name.endswith('.layers'):
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fsdp_version(submodule) > 0:
                # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                with FSDP.summon_full_params(submodule, writeback=False):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    sub_lora_params = get_peft_model_state_dict(fsdp_module._fsdp_wrapped_module, state_dict=submodule.state_dict())
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    sub_lora_params = {f"{prefix}.{name}": param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                        for name, param in sub_lora_params.items()}
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    lora_params.update(sub_lora_params)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    submodule._is_root = False
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.cuda.empty_cache()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return lora_params