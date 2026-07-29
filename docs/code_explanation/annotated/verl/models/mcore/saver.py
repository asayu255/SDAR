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
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.distributed import DistributedDataParallel as LocalDDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer.module import Float16Module
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.nn.parallel import DistributedDataParallel as torchDDP

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import print_rank_0, unwrap_model


# [EXPLAIN] `_megatron_calc_global_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _megatron_calc_global_rank(tp_rank: int = 0, dp_rank: int = 0, pp_rank: int = 0, cp_rank: int = 0, ep_rank: int = 0):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Calculate global rank with support for CP/EP parallelism"""

    # Get parallel sizes for each dimension
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_size = mpu.get_data_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_size = mpu.get_context_parallel_world_size()
    # ep_size = mpu.get_expert_model_parallel_world_size()

    # Verify total GPU count matches (must be consistent with parallel_state.py)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_size = tp_size * dp_size * pp_size * cp_size
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert total_size == torch.distributed.get_world_size(), f"{tp_size}x{dp_size}x{pp_size}x{cp_size} != {torch.distributed.get_world_size()}"

    # Core calculation logic (corresponds to RankGenerator order parameter)
    # Assumes default order is "tp-cp-ep-dp-pp"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ((pp_rank * dp_size + dp_rank) * cp_size + cp_rank) * tp_size + tp_rank


# [EXPLAIN] `_megatron_calc_layer_map` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _megatron_calc_layer_map(config):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Calculate the mapping of global layer_idx to local layer_idx
    Returns:
        layer_map (Dict: int -> tuple(int, int, int)):
            mapping from the global layer index to
            a tuple of (pp_rank, virtual_pp_rank, layer_idx inside model)
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import mpu

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layer_map = dict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for pp_rank_idx in range(pp_size):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for virtual_pp_rank_idx in range(virtual_pp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer_offset = virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) + pp_rank_idx * num_layers_per_model
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for layer_idx in range(num_layers_per_model):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,
                    virtual_pp_rank_idx,
                    layer_idx,
                )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return layer_map


# [EXPLAIN] `merge_megatron_ckpt_gptmodel` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def merge_megatron_ckpt_gptmodel(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Merge sharded parameters of a Megatron module into a merged checkpoint.

    Args:
        wrapped_models (list of megatron.core.distributed.DistributedDataParallel):
            The local DDP wrapped megatron modules.
        config (str or None):
            HF config for model
        dtype: model params type
        is_value_model: if model is value model
        tie_word_embeddings: tie_word_embeddings
    Returns:
        state_dict (dict):
            The merged state_dict in rank 0, and an empty dictionary in other ranks.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_time = time.time()

    # [EXPLAIN] `_get_gpt_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_gpt_model(model):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_rank = mpu.get_data_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_rank = mpu.get_context_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mp_group = mpu.get_model_parallel_group()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dist.get_rank() == 0:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(wrapped_models, (list, tuple)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wrapped_models = list(wrapped_models)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(wrapped_models) == virtual_pp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    models = [None] * len(wrapped_models)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, wrapped_model in enumerate(wrapped_models):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(models[i].decoder.layers) == num_layers_per_model, "len model layers {} not equal to num_layers_per_model {}".format(len(models[i].decoder.layers), num_layers_per_model)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state_dict = dict()

    # [EXPLAIN] `_get_cpu_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_cpu_tensor(tensor: torch.Tensor):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor.device == torch.device("cpu"):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return tensor.detach().clone()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor.detach().cpu()

    # [EXPLAIN] `_broadcast_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _broadcast_tensor(tensor, name, src_pp_rank) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """broadcast tensor across mp_group"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == src_rank:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tensor is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                weight = None
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor_shape = None
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                weight = tensor
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor_shape = weight.shape
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight = None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_shape = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obj_list = [tensor_shape]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_shape = obj_list[0]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0(f"tensor:[{name}] not exist, skip collect")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if weight is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight = torch.empty(
                tensor_shape,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast(weight, src=src_rank, group=mp_group)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[name] = _get_cpu_tensor(weight)

    # [EXPLAIN] `_broadcast_tp_shard_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _broadcast_tp_shard_tensor(tensor, name, src_pp_rank, concat_dim=0, mutate_func=None) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """broadcast tensor in tp shards across mp_group"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obj_list = [chunk_shape]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = obj_list[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0(f"tp_shard tensor:[{name}] not exist, skip collecting")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_tensors = [None] * tp_size

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(tp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.distributed.get_rank() == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_tensor = torch.concat(chunk_tensors, dim=concat_dim)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mutate_func is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_tensor = mutate_func(full_tensor)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[name] = full_tensor

    # [EXPLAIN] `_broadcast_tp_shard_tensor_gate_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, src_pp_rank) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """broadcast tensor in tp shards across mp_group"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obj_list = [chunk_shape]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = obj_list[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_tensors = [None] * tp_size

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(tp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.distributed.get_rank() == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_tensor = torch.concat(chunk_tensors, dim=0)
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
            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    # [EXPLAIN] `_broadcast_tp_shard_tensor_qkv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, src_pp_rank):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """broadcast tensor in tp shards across mp_group"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obj_list = [chunk_shape]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_shape = obj_list[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0(f"tp_shard tensor:[{q_name}] not exist, skip collecting")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_tensors = [None] * tp_size

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(tp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.distributed.get_rank() == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_tensor = torch.concat(chunk_tensors, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_weight_list = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k_weight_list = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            v_weight_list = []
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
                    num_query_groups_per_partition = wrapped_models[0].config.num_query_groups // tp_size
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
                        q_weight_list.append(q_part)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        k_weight_list.append(k_part)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        v_weight_list.append(v_part)
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
                    num_query_groups_per_partition = wrapped_models[0].config.num_query_groups // tp_size
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
                        q_weight_list.append(q_part)
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if i * config.num_key_value_heads % tp_size == 0:
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            k_weight_list.append(k_part)
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            v_weight_list.append(v_part)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[q_name] = torch.cat(q_weight_list, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[k_name] = torch.cat(k_weight_list, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[v_name] = torch.cat(v_weight_list, dim=0)

    # empty cache before collecting weights
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()
    # Embeddings
    # -------------------
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dp_rank == 0 and cp_rank == 0:  # models are identical across cp ranks
        # Embeddings
        # -------------------
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print_rank_0("collecting embeddings...")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gpt_model_module = _get_gpt_model(models[0])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _broadcast_tp_shard_tensor(
            gpt_model_module.embedding.word_embeddings.weight if pp_rank == 0 else None,
            "model.embed_tokens.weight",
            src_pp_rank=0,
        )

        # Transformer layers
        # -------------------
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_map = _megatron_calc_layer_map(config)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for layer in range(config.num_hidden_layers):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0(f"collecting layer #{layer}...")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            layer_name = f"model.layers.{layer}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            src_pp_rank, src_virtual_pp_rank, src_layer_idx = layer_map[layer]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gpt_model_module = _get_gpt_model(models[src_virtual_pp_rank])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sync_layer = gpt_model_module.decoder.layers[src_layer_idx]

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tensor(
                sync_layer.self_attention.linear_qkv.layer_norm_weight,
                f"{layer_name}.input_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if gpt_model_module.config.qk_layernorm:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _broadcast_tensor(
                    sync_layer.self_attention.q_layernorm.weight,
                    f"{layer_name}.self_attn.q_norm.weight",
                    src_pp_rank=src_pp_rank,
                )
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _broadcast_tensor(
                    sync_layer.self_attention.k_layernorm.weight,
                    f"{layer_name}.self_attn.k_norm.weight",
                    src_pp_rank=src_pp_rank,
                )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tp_shard_tensor_qkv(
                sync_layer.self_attention.linear_qkv.weight,
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if gpt_model_module.config.add_qkv_bias:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _broadcast_tp_shard_tensor_qkv(
                    sync_layer.self_attention.linear_qkv.bias,
                    f"{layer_name}.self_attn.q_proj.bias",
                    f"{layer_name}.self_attn.k_proj.bias",
                    f"{layer_name}.self_attn.v_proj.bias",
                    src_pp_rank=src_pp_rank,
                )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tp_shard_tensor(
                sync_layer.self_attention.linear_proj.weight,
                f"{layer_name}.self_attn.o_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tensor(
                sync_layer.mlp.linear_fc1.layer_norm_weight,
                f"{layer_name}.post_attention_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tp_shard_tensor_gate_up(
                sync_layer.mlp.linear_fc1.weight,
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            _broadcast_tp_shard_tensor(
                sync_layer.mlp.linear_fc2.weight,
                f"{layer_name}.mlp.down_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

        # Final Layernorm
        # -------------------
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print_rank_0("collecting final layernorm...")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gpt_model_module = _get_gpt_model(models[-1])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _broadcast_tensor(
            getattr(gpt_model_module.decoder.final_layernorm, "weight", None),
            "model.norm.weight",
            src_pp_rank=pp_size - 1,
        )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tie_word_embeddings:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0("tie word embedding skip load lm_head...")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_rank_0("collecting lm_head...")

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_value_model:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lm_head_weight = None
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if pp_rank == pp_size - 1:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lm_head_weight = getattr(gpt_model_module.output_layer, "weight", None)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _broadcast_tensor(lm_head_weight, "lm_head.weight", src_pp_rank=pp_size - 1)

            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _broadcast_tp_shard_tensor(
                    getattr(gpt_model_module.output_layer, "weight", None) if pp_rank == pp_size - 1 else None,
                    "lm_head.weight",
                    src_pp_rank=pp_size - 1,
                )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.barrier()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.get_rank() == 0:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in state_dict.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dtype != v.dtype:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state_dict[k] = v.to(dtype)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print_rank_0(f"merge megatron ckpt done, time elapsed {time.time() - start_time}s")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return state_dict


# [EXPLAIN] `merge_megatron_ckpt_gptmodel_qwen_moe` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def merge_megatron_ckpt_gptmodel_qwen_moe(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_qwen_moe is not implemented")


# [EXPLAIN] `merge_megatron_ckpt_gptmodel_dpskv3` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def merge_megatron_ckpt_gptmodel_dpskv3(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_dpskv3 is not implemented")


# [EXPLAIN] `merge_megatron_ckpt_gptmodel_mixtral` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def merge_megatron_ckpt_gptmodel_mixtral(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_mixtral is not implemented")
