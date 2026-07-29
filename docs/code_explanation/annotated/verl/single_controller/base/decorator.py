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
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import wraps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from types import FunctionType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProtoFuture, _padding_size_key
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import DynamicEnum

# here we add a magic number of avoid user-defined function already have this attribute
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAGIC_ATTR = "attrs_3141562937"


# [EXPLAIN] `Dispatch` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Dispatch(DynamicEnum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _registry = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _next_value = 0


# [EXPLAIN] `init_predefined_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_predefined_dispatch_mode():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("RANK_ZERO")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("ONE_TO_ALL")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("ALL_TO_ALL")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("MEGATRON_COMPUTE")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("MEGATRON_PP_AS_DP")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("MEGATRON_PP_ONLY")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("MEGATRON_COMPUTE_PROTO")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("MEGATRON_PP_AS_DP_PROTO")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("DP_COMPUTE")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("DP_COMPUTE_PROTO")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


# [EXPLAIN] `Execute` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Execute(DynamicEnum):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _registry = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _next_value = 0


# [EXPLAIN] `init_predefined_execute_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def init_predefined_execute_mode():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Execute.register("ALL")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
init_predefined_dispatch_mode()
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
init_predefined_execute_mode()


# [EXPLAIN] `_split_args_kwargs_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto, DataProtoFuture

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(arg, (DataProto, DataProtoFuture))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        splitted_args.append(arg.chunk(chunks=chunks))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_kwargs = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(val, (DataProto, DataProtoFuture))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return splitted_args, splitted_kwargs


# [EXPLAIN] `_split_args_kwargs_data_proto_with_auto_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto, DataProtoFuture

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_kwargs = {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data_proto_len = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padding_size = None
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(arg, (DataProto, DataProtoFuture))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(arg, DataProto) and arg.is_padding_enabled():
            # for padding, we only support DataProto with same length
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if data_proto_len is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_proto_len = len(arg)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                splitted_kwargs[_padding_size_key] = padding_size
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert data_proto_len == len(arg), f"expecting all arg share same length of {data_proto_len}, but got {len(arg)}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_proto_len = len(arg)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            arg.padding(padding_size=padding_size)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        splitted_args.append(arg.chunk(chunks=chunks))

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(val, (DataProto, DataProtoFuture))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(val, DataProto) and val.is_padding_enabled():
            # for padding, we only support DataProto with same length
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if data_proto_len is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_proto_len = len(val)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                padding_size = chunks - (data_proto_len % chunks)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                splitted_kwargs[_padding_size_key] = padding_size
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert data_proto_len == len(val), f"expecting all arg share same length of {data_proto_len}, but got {len(val)}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_proto_len = len(val)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return splitted_args, splitted_kwargs


# [EXPLAIN] `dispatch_one_to_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_one_to_all(worker_group, *args, **kwargs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = tuple([arg] * worker_group.world_size for arg in args)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args, kwargs


# [EXPLAIN] `dummy_direct_rollout_call` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise NotImplementedError("Direct rollout call is forbidden.")


# [EXPLAIN] `dispatch_all_to_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_all_to_all(worker_group, *args, **kwargs):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args, kwargs


# [EXPLAIN] `collect_all_to_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_all_to_all(worker_group, output):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `dispatch_megatron_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_megatron_compute(worker_group, *args, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup), f"worker_group must be MegatronWorkerGroup, Got {type(worker_group)}"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_args = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.dp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformed_args = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            transformed_args.append(arg[local_dp_rank])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_args.append(transformed_args)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_args = tuple(all_args)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_kwargs = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.dp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformed_v = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            transformed_v.append(v[local_dp_rank])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_kwargs[k] = transformed_v
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return all_args, all_kwargs


# [EXPLAIN] `collect_megatron_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_megatron_compute(worker_group, output):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Only collect the data from the tp=0 and pp=last and every dp ranks
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_in_dp = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = worker_group.get_megatron_global_info().pp_size
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for global_rank in range(worker_group.world_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1 and local_rank_info.cp_rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_in_dp.append(output[global_rank])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_in_dp


# [EXPLAIN] `dispatch_megatron_compute_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_megatron_compute_data_proto(worker_group, *args, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


# [EXPLAIN] `_concat_data_proto_or_future` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _concat_data_proto_or_future(output: List):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import ray

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto, DataProtoFuture

    # make sure all the elements in output has the same type
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for o in output:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert type(o) is type(output[0])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    o = output[0]

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(o, DataProto):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.concat(output)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(o, ray.ObjectRef):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProtoFuture.concat(output)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `collect_megatron_compute_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_megatron_compute_data_proto(worker_group, output):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Each output must be a DataProto. We concat the dim=0 of output
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import ray

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = collect_megatron_compute(worker_group, output)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for o in output:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _concat_data_proto_or_future(output)


# [EXPLAIN] `dispatch_megatron_pp_as_dp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_megatron_pp_as_dp(worker_group, *args, **kwargs):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    treat pp as dp.
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_size = worker_group.pp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_size = worker_group.dp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cp_size = worker_group.cp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_dp_cp_size = pp_size * dp_size * cp_size

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_args = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(arg, (List, Tuple)) and len(arg) == pp_dp_cp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformed_args = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_cp_rank = worker_group.get_megatron_rank_info(rank=i).cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arg_rank = dp_cp_rank * pp_size + local_pp_rank

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            transformed_args.append(arg[arg_rank])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_args.append(transformed_args)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_args = tuple(all_args)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_kwargs = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(v, (List, Tuple)) and len(v) == pp_dp_cp_size, f"expect len(v)=={pp_dp_cp_size}, got {len(v)}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        transformed_v = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(worker_group.world_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_cp_rank = worker_group.get_megatron_rank_info(rank=i).cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arg_rank = dp_cp_rank * pp_size + local_pp_rank
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            transformed_v.append(v[arg_rank])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_kwargs[k] = transformed_v
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return all_args, all_kwargs


# [EXPLAIN] `collect_megatron_pp_as_dp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_megatron_pp_as_dp(worker_group, output):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    treat pp as dp. Only collect data on tp=0
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_in_dp = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for global_rank in range(worker_group.world_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_rank_info.tp_rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_in_dp.append(output[global_rank])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_in_dp


# [EXPLAIN] `collect_megatron_pp_only` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_megatron_pp_only(worker_group, output):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_in_pp = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for global_rank in range(worker_group.world_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_in_pp.append(output[global_rank])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_in_pp


# [EXPLAIN] `dispatch_megatron_pp_as_dp_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_megatron_pp_as_dp_data_proto(worker_group, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pp_dp_cp_size = worker_group.dp_size * worker_group.pp_size * worker_group.cp_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(pp_dp_cp_size, *args, **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ret = dispatch_megatron_pp_as_dp(worker_group, *splitted_args, **splitted_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ret


# [EXPLAIN] `collect_megatron_pp_as_dp_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_megatron_pp_as_dp_data_proto(worker_group, output):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = collect_megatron_pp_as_dp(worker_group, output)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _concat_data_proto_or_future(output)


# [EXPLAIN] `dispatch_dp_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_dp_compute(worker_group, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.worker_group import WorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, WorkerGroup)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.world_size
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in kwargs.items():
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.world_size
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return args, kwargs


# [EXPLAIN] `collect_dp_compute` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_dp_compute(worker_group, output):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.worker_group import WorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, WorkerGroup)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(output) == worker_group.world_size
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `dispatch_dp_compute_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.worker_group import WorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return splitted_args, splitted_kwargs


# [EXPLAIN] `dispatch_dp_compute_data_proto_with_func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.single_controller.base.worker_group import WorkerGroup

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(worker_group, WorkerGroup)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return splitted_args_with_func, splitted_kwargs


# [EXPLAIN] `collect_dp_compute_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collect_dp_compute_data_proto(worker_group, output):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import ray

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.protocol import DataProto

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for o in output:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = collect_dp_compute(worker_group, output)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _concat_data_proto_or_future(output)


# Global registry for dispatch mode.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.MEGATRON_COMPUTE: {
        "dispatch_fn": dispatch_megatron_compute,
        "collect_fn": collect_megatron_compute,
    },
    Dispatch.MEGATRON_PP_AS_DP: {
        "dispatch_fn": dispatch_megatron_pp_as_dp,
        "collect_fn": collect_megatron_pp_as_dp,
    },
    Dispatch.MEGATRON_PP_ONLY: {"dispatch_fn": dispatch_one_to_all, "collect_fn": collect_megatron_pp_only},
    Dispatch.MEGATRON_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_megatron_compute_data_proto,
        "collect_fn": collect_megatron_compute_data_proto,
    },
    Dispatch.MEGATRON_PP_AS_DP_PROTO: {
        "dispatch_fn": dispatch_megatron_pp_as_dp_data_proto,
        "collect_fn": collect_megatron_pp_as_dp_data_proto,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


# [EXPLAIN] `get_predefined_dispatch_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_predefined_dispatch_fn(dispatch_mode):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


# [EXPLAIN] `register_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Register a new dispatch mode.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _check_dispatch_mode(dispatch_mode)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


# [EXPLAIN] `update_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Update the dispatch mode.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _check_dispatch_mode(dispatch_mode)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


# [EXPLAIN] `get_predefined_execute_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_predefined_execute_fn(execute_mode):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return predefined_execute_mode_fn[execute_mode]


# [EXPLAIN] `_check_dispatch_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _check_dispatch_mode(dispatch_mode):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(dispatch_mode, (Dispatch, Dict)), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(dispatch_mode, Dict):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        necessary_keys = ["dispatch_fn", "collect_fn"]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in necessary_keys:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


# [EXPLAIN] `_check_execute_mode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _check_execute_mode(execute_mode):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


# [EXPLAIN] `_materialize_futures` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _materialize_futures(*args, **kwargs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_args = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for arg in args:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(arg, DataProtoFuture):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arg = arg.get()
        # add more type to materialize
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        new_args.append(arg)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in kwargs.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(v, DataProtoFuture):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs[k] = v.get()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_args = tuple(new_args)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return new_args, kwargs


# [EXPLAIN] `register` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _check_execute_mode(execute_mode=execute_mode)

    # [EXPLAIN] `decorator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decorator(func):
        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @wraps(func)
        # [EXPLAIN] `inner` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def inner(*args, **kwargs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if materialize_futures:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                args, kwargs = _materialize_futures(*args, **kwargs)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return func(*args, **kwargs)

        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @wraps(func)
        # [EXPLAIN] `async_inner` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        async def async_inner(*args, **kwargs):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if materialize_futures:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                args, kwargs = _materialize_futures(*args, **kwargs)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return await func(*args, **kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        setattr(wrapper, MAGIC_ATTR, attrs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return wrapper

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return decorator
