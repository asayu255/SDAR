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
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist


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


# [EXPLAIN] `load_state_dict_to_megatron_qwen2` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_state_dict_to_megatron_qwen2(state_dict, wrapped_models, config, params_dtype, is_value_model=False, tie_word_embeddings=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load merged state_dict to sharded Megatron module in training."""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import DistributedDataParallel as LocalDDP
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import mpu
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core.transformer.module import Float16Module
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.megatron_utils import print_rank_0, unwrap_model

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_time = time.time()

    # [EXPLAIN] `_get_gpt_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_gpt_model(model):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model

    # [EXPLAIN] `fetch_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fetch_params(module):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for param in module.parameters():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.fetch(param.data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_rank = mpu.get_data_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mp_group = mpu.get_model_parallel_group()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.get_rank() == 0:
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
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers, f"num_layers_per_model: {num_layers_per_model} * pp_size: {pp_size} * virtual_pp_size: {virtual_pp_size} != config.num_hidden_layers: {config.num_hidden_layers}"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    models = [None] * len(wrapped_models)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, wrapped_model in enumerate(wrapped_models):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gpt_model_module = _get_gpt_model(models[i])
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(gpt_model_module.model.layers) == num_layers_per_model

    # [EXPLAIN] `_fetch_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _fetch_tensor(tensor, name) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """fetch tensor"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor = tensor.data.copy_(state_dict[name], non_blocking=True)

    # [EXPLAIN] `_fetch_tp_shard_tensor_vocab` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _fetch_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """fetch tensor in tp shards"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in state_dict:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_weight = state_dict[name]

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mutate_func is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_weight = mutate_func(full_weight)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tensor is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor = tensor.data.copy_(tensor_chunk[tp_rank], non_blocking=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"tp_shard tensor:[{name}] not in state_dict, skip loading")

    # [EXPLAIN] `_fetch_tp_shard_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _fetch_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """fetch tensor in tp shards"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if name in state_dict:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            full_weight = state_dict[name]

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mutate_func is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                full_weight = mutate_func(full_weight)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tensor is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor = tensor.data.copy_(tensor_chunk[tp_rank], non_blocking=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"tp_shard tensor:[{name}] not in state_dict, skip loading")

    # [EXPLAIN] `_fetch_tp_shard_tensor_gate_up` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _fetch_tp_shard_tensor_gate_up(tensor, gate_name, up_name) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """fetch gate_up tensor in tp shards"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if gate_name in state_dict and up_name in state_dict:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate_weight = state_dict[gate_name]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            up_weight = state_dict[up_name]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            new_gate_up_weight = torch.empty(config.intermediate_size * 2, config.hidden_size, dtype=params_dtype, device=torch.cuda.current_device())
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(tp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                intermediate_size_tp = config.intermediate_size // tp_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gate_weight_tp = gate_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                up_weight_tp = up_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                new_gate_up_weight[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)].copy_(torch.cat([gate_weight_tp, up_weight_tp], dim=0))

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if tensor is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor = tensor.data.copy_(tensor_chunk[tp_rank], non_blocking=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"tp_shard tensor:[{gate_name}, {up_name}] not in state_dict, skip loading")

    # [EXPLAIN] `_fetch_tp_shard_tensor_qkv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _fetch_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, bias=False) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """fetch tensor in tp shards across mp_group"""
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal mp_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert q_name in state_dict and k_name in state_dict and v_name in state_dict
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_weight_q = state_dict[q_name]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_weight_k = state_dict[k_name]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_weight_v = state_dict[v_name]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_size_per_head = config.hidden_size // config.num_attention_heads

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.num_key_value_heads >= tp_size:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_size_tp = config.hidden_size // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_size = q_size_tp + 2 * kv_size_tp
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not bias:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_weight_qkv = torch.empty(total_size * tp_size, config.hidden_size, dtype=params_dtype, device=torch.cuda.current_device())
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_weight_qkv = torch.empty(total_size * tp_size, dtype=params_dtype, device=torch.cuda.current_device())
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(tp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_part = full_weight_k[i * kv_size_tp : (i + 1) * kv_size_tp]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v_part = full_weight_v[i * kv_size_tp : (i + 1) * kv_size_tp]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part], dim=0))

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_size_tp = config.hidden_size // tp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_size_tp = hidden_size_per_head
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_size = q_size_tp + 2 * kv_size_tp
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not bias:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_weight_qkv = torch.empty(total_size * tp_size, config.hidden_size, dtype=params_dtype, device=torch.cuda.current_device())
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_weight_qkv = torch.empty(total_size * tp_size, dtype=params_dtype, device=torch.cuda.current_device())
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(tp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                start_idx = i * config.num_key_value_heads // tp_size * hidden_size_per_head
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                end_idx = (i * config.num_key_value_heads // tp_size + 1) * hidden_size_per_head
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_part = full_weight_k[start_idx:end_idx]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v_part = full_weight_v[start_idx:end_idx]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                new_weight_qkv[i * total_size : (i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part], dim=0))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_chunk = torch.chunk(new_weight_qkv, tp_size, dim=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensor is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor = tensor.data.copy_(tensor_chunk[tp_rank], non_blocking=True)

    # Embeddings
    # -------------------
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print_rank_0("loading embeddings...")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gpt_model_module = _get_gpt_model(models[0])
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pp_rank == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embed_tokens_weight = gpt_model_module.model.embed_tokens.weight
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

    # Transformer layers
    # -------------------
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layer_map = _megatron_calc_layer_map(config)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_layer_per_pp = config.num_hidden_layers // pp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layer_list = []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if vpp_size is not None:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for vpp_rank in range(vpp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layer_vpp_chunk = num_layer_per_pp // vpp_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_layer_this_model = num_layer_vpp_chunk
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            offset = vpp_rank * (config.num_hidden_layers // mpu.get_virtual_pipeline_model_parallel_world_size()) + (mpu.get_pipeline_model_parallel_rank() * num_layer_vpp_chunk)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            layer_list.extend(list(range(offset, offset + num_layer_this_model)))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_layer_this_model = num_layer_per_pp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        offset = pp_rank * num_layer_per_pp
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        layer_list.extend(list(range(offset, offset + num_layer_this_model)))

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for layer in layer_list:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"{torch.distributed.get_rank()} loading layer #{layer}...")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_name = f"model.layers.{layer}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"{torch.distributed.get_rank()} offset: {offset}, num_layer_this_model: {num_layer_this_model}, layer_name: {layer_name}, layer_map[layer]: {layer_map[layer]}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sync_layer = gpt_model_module.model.layers[dst_layer_idx]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tensor(
            sync_layer.input_layernorm.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.input_layernorm.weight",
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor_qkv(
            sync_layer.self_attn.qkv_proj.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.self_attn.q_proj.weight",
            f"{layer_name}.self_attn.k_proj.weight",
            f"{layer_name}.self_attn.v_proj.weight",
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor_qkv(
            sync_layer.self_attn.qkv_proj.bias if dst_pp_rank == pp_rank else None,
            f"{layer_name}.self_attn.q_proj.bias",
            f"{layer_name}.self_attn.k_proj.bias",
            f"{layer_name}.self_attn.v_proj.bias",
            bias=True,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor(
            sync_layer.self_attn.o_proj.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.self_attn.o_proj.weight",
            chunk_dim=1,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tensor(
            sync_layer.post_attention_layernorm.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.post_attention_layernorm.weight",
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor_gate_up(
            sync_layer.mlp.gate_up_proj.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.mlp.gate_proj.weight",
            f"{layer_name}.mlp.up_proj.weight",
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _fetch_tp_shard_tensor(
            sync_layer.mlp.down_proj.weight if dst_pp_rank == pp_rank else None,
            f"{layer_name}.mlp.down_proj.weight",
            chunk_dim=1,
        )
    # Final Layernorm
    # -------------------
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print_rank_0("loading final layernorm...")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gpt_model_module = _get_gpt_model(models[-1])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _fetch_tensor(
        getattr(gpt_model_module.model.norm, "weight", None),
        "model.norm.weight",
    )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tie_word_embeddings:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print_rank_0("tie_word_embeddings skip load lm_head")
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print_rank_0("loading lm_head...")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pp_rank + 1 == pp_size:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lm_head_weight = gpt_model_module.lm_head.weight

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_value_model:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "lm_head.weight" in state_dict and state_dict["lm_head.weight"].shape[0] == 1:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    _fetch_tensor(lm_head_weight, "lm_head.weight")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print_rank_0("load lm_head from value_head weight")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif "reward_head.weight" in state_dict and state_dict["reward_head.weight"].shape[0] == 1:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    _fetch_tensor(lm_head_weight, "reward_head.weight")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print_rank_0("load lm_head from value_head weight")
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    _fetch_tensor(None, "lm_head.weight")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print_rank_0("fail to match lm_head in value_model")

            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                _fetch_tp_shard_tensor(lm_head_weight, "lm_head.weight")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.barrier()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")
