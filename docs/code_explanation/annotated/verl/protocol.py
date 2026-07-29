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
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import contextlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pickle
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass, field
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Callable, Dict, List, Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tensordict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging import version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import DataLoader

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_torch_device
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import union_two_dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import allgather_dict_tensors

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["DataProto", "union_tensor_dict"]

# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with contextlib.suppress(Exception):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    tensordict.set_lazy_legacy(False).set()


# [EXPLAIN] `_DataProtoConfigMeta` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _DataProtoConfigMeta(type):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _config = {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    auto_padding_key = "_verl_auto_padding"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `auto_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def auto_padding(cls):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @auto_padding.setter
    # [EXPLAIN] `auto_padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def auto_padding(cls, enabled: bool):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls._config[cls.auto_padding_key] = enabled


# [EXPLAIN] `DataProtoConfig` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
    pass


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_padding_size_key = "_padding_size_key_x123d"


# [EXPLAIN] `pad_dataproto_to_divisor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(data, DataProto), "data must be a DataProto"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(data) % size_divisor != 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_size = size_divisor - len(data) % size_divisor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_protos = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        remaining_pad = pad_size
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while remaining_pad > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            take_size = min(remaining_pad, len(data))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            padding_protos.append(data[:take_size])
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            remaining_pad -= take_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_padded = DataProto.concat([data] + padding_protos)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(data) == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logging.warning("padding a DataProto with no item, no changed made")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_size = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_padded = data
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data_padded, pad_size


# [EXPLAIN] `unpad_dataproto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def unpad_dataproto(data: "DataProto", pad_size):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pad_size != 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data[:-pad_size]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return data


# [EXPLAIN] `union_tensor_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Union two tensordicts."""
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in tensor_dict2.keys():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key not in tensor_dict1.keys():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_dict1[key] = tensor_dict2[key]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert tensor_dict1[key].equal(tensor_dict2[key]), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor_dict1


# [EXPLAIN] `union_numpy_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in tensor_dict2.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key in tensor_dict1:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(tensor_dict2[key], np.ndarray)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert pd.DataFrame(tensor_dict2[key]).equals(pd.DataFrame(tensor_dict1[key])), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_dict1[key] = val

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor_dict1


# [EXPLAIN] `list_of_dict_to_dict_of_list` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(list_of_dict) == 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    keys = list_of_dict[0].keys()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = {key: [] for key in keys}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in list_of_dict:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, item in data.items():
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert key in output
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output[key].append(item)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `fold_batch_dim` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fold_batch_dim(data: "DataProto", new_batch_size):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = data.batch.batch_size[0]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert batch_size % new_batch_size == 0

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor: TensorDict = data.batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor = data.non_tensor_batch

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor = tensor.view(new_batch_size, -1)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    tensor.auto_batch_size_(batch_dims=1)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in non_tensor.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


# [EXPLAIN] `unfold_batch_dim` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def unfold_batch_dim(data: "DataProto", batch_dims=2):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Unfold the first n dims as new batch dim
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor: TensorDict = data.batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor = data.non_tensor_batch
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    tensor.auto_batch_size_(batch_dims=batch_dims)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor = tensor.view(-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = tensor.batch_size[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor_new = {}

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in non_tensor.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


# [EXPLAIN] `collate_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collate_fn(x: list["DataProtoItem"]):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor_batch = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in x:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        batch.append(data.batch)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        non_tensor_batch.append(data.non_tensor_batch)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = torch.stack(batch).contiguous()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in non_tensor_batch.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_batch[key] = np.array(val, dtype=object)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `DataProtoItem` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DataProtoItem:
    # TODO(zhangchi.usc1992) add consistency check
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch: TensorDict = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor_batch: Dict = field(default_factory=dict)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    meta_info: Dict = field(default_factory=dict)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `DataProto` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DataProto:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch: TensorDict = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensor_batch: Dict = field(default_factory=dict)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    meta_info: Dict = field(default_factory=dict)

    # [EXPLAIN] `__post_init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __post_init__(self):
        # perform necessary checking
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.check_consistency()

    # [EXPLAIN] `__len__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __len__(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.batch.batch_size[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            random_key = list(self.non_tensor_batch.keys())[0]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.non_tensor_batch[random_key].shape[0]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 0

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(self, item):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(item, slice):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(item, (int, np.integer)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_data = self.batch[item] if self.batch is not None else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise TypeError(f"Indexing with {type(item)} is not supported")

    # [EXPLAIN] `__getstate__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getstate__(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import io

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer = io.BytesIO()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch = self.batch.contiguous()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch = self.batch.consolidate()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.save(self.batch, buffer)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_bytes = buffer.getvalue()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    # [EXPLAIN] `__setstate__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __setstate__(self, data):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import io

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = torch.load(batch_deserialized, weights_only=False, map_location="cpu" if not get_torch_device().is_available() else None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.non_tensor_batch = non_tensor_batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.meta_info = meta_info

    # [EXPLAIN] `save_to_disk` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_to_disk(self, filepath):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(filepath, "wb") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            pickle.dump(self, f)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `load_from_disk` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_from_disk(filepath) -> "DataProto":
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(filepath, "rb") as f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = pickle.load(f)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data

    # [EXPLAIN] `print_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def print_size(self, prefix=""):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_of_tensordict = 0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is None:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, tensor in self.batch.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                size_of_tensordict += tensor.element_size() * tensor.numel()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size_of_numpy_array = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, numpy_array in self.non_tensor_batch.items():
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            size_of_numpy_array += numpy_array.nbytes

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        size_of_numpy_array /= 1024**3
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        size_of_tensordict /= 1024**3

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if prefix:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            message = f"{prefix}, " + message
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(message)

    # [EXPLAIN] `check_consistency` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def check_consistency(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.non_tensor_batch is not None:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, val in self.non_tensor_batch.items():
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert isinstance(val, np.ndarray)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_size = self.batch.batch_size[0]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, val in self.non_tensor_batch.items():
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert isinstance(val, np.ndarray), f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for {key=}, got {type(val)=}"
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert val.shape[0] == batch_size, f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_single_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None, auto_padding=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a DataProto from a dict of tensors and non_tensors"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensors = {}

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in data.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(val, torch.Tensor):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensors[key] = val
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif isinstance(val, np.ndarray):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensors[key] = val
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unsupported type in data {type(val)}")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_dict(cls, tensors: Optional[Dict[str, torch.Tensor]] = None, non_tensors=None, meta_info=None, num_batch_dims=1, auto_padding=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if non_tensors is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tensors is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensors = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if meta_info is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            meta_info = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if non_tensors is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensors = {}

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(non_tensors, dict)

        # get and check batch size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pivot_key = None
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, tensor in tensors.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if batch_size is None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                batch_size = tensor.shape[:num_batch_dims]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pivot_key = key
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                current_batch = tensor.shape[:num_batch_dims]
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert batch_size == current_batch, f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in non_tensors.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not isinstance(val, np.ndarray):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensors[key] = np.array(val, dtype=object)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if auto_padding:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            meta_info[DataProtoConfig.auto_padding_key] = True
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    # [EXPLAIN] `to` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def to(self, device) -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch = self.batch.to(device)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self

    # [EXPLAIN] `select` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if batch_keys is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_keys = tuple(batch_keys)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub_batch = self.batch.select(*batch_keys)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub_batch = self.batch

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if non_tensor_batch_keys is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch = self.non_tensor_batch

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if deepcopy:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if meta_info_keys is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub_meta_info = self.meta_info

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if deepcopy:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sub_meta_info = copy.deepcopy(sub_meta_info)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    # [EXPLAIN] `select_idxs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def select_idxs(self, idxs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(idxs, list):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs = torch.tensor(idxs)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if idxs.dtype != torch.bool:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idxs = idxs.type(torch.int32)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(idxs, np.ndarray):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs_np = idxs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs_torch = torch.from_numpy(idxs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:  # torch.Tensor
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs_torch = idxs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idxs_np = idxs.detach().cpu().numpy()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            selected_batch = TensorDict(source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,), device=self.batch.device)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            selected_batch = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        selected_non_tensor = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            selected_non_tensor[key] = val[idxs_np]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    # [EXPLAIN] `slice` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def slice(self, start=None, end=None, step=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        slice_obj = slice(start, end, step)

        # Handle the batch data
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sliced_batch = self.batch[slice_obj]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sliced_batch = None

        # Handle the non-tensor batch data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sliced_non_tensor = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    # [EXPLAIN] `pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if batch_keys is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_keys = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if meta_info_keys is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            meta_info_keys = []
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if non_tensor_batch_keys is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch_keys = []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors = {}
        # tensor batch
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in batch_keys:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert key in self.batch.keys()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensors[key] = self.batch.pop(key)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensors = {}
        # non tensor batch
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in non_tensor_batch_keys:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert key in self.non_tensor_batch.keys()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensors[key] = self.non_tensor_batch.pop(key)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        meta_info = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in meta_info_keys:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert key in self.meta_info.keys()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            meta_info[key] = self.meta_info.pop(key)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    # [EXPLAIN] `rename` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Note that this function only rename the key in the batch
        """

        # [EXPLAIN] `validate_input` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def validate_input(keys):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if keys is not None:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(keys, str):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    keys = [keys]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                elif isinstance(keys, list):
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    pass
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return keys

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        old_keys = validate_input(old_keys)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_keys = validate_input(new_keys)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(new_keys) != len(old_keys):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self

    # [EXPLAIN] `union` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def union(self, other: "DataProto") -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = union_tensor_dict(self.batch, other.batch)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self

    # [EXPLAIN] `make_iterator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        r"""Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.


        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that ``batch.batch_size[0] % mini_batch_size == 0``.
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs (Any): internally, it returns a DataLoader over the batch. The dataloader_kwargs is the kwargs passed to the DataLoader.

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is ``self.batch.batch_size * epochs // mini_batch_size``
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dataloader_kwargs is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataloader_kwargs = {}

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if seed is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            generator = torch.Generator()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            generator.manual_seed(seed)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            generator = None

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(dataloader_kwargs, Dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        train_dataloader = DataLoader(dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs)

        # [EXPLAIN] `get_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def get_data():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _ in range(epochs):
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for d in train_dataloader:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    d.meta_info = self.meta_info
                    # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                    yield d

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return iter(get_data())

    # [EXPLAIN] `is_padding_enabled` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def is_padding_enabled(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Check if padding is enabled for the DataProto.
        Returns:
            bool: True if padding is enabled, False otherwise.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    # [EXPLAIN] `padding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def padding(self, padding_size, padding_candidate=""):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Pad the DataProto by concating with padding_candidate.repeat(padding_size)

        Args:
            padding_size (int): the number of repeated padding_candidate
            padding_candidate: the item to be repeated and appended to the DataProto, only supporting ["first", "last"]
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if padding_size == 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_part = padding_candidate.repeat(padding_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padded_dp = DataProto.concat([self, padding_part])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = padded_dp.batch
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.non_tensor_batch = padded_dp.non_tensor_batch

    # [EXPLAIN] `chunk` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def chunk(self, chunks: int) -> List["DataProto"]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.is_padding_enabled():
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(self) % chunks == 0, f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz_in_batch = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_lst = [None for _ in range(chunks)]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(val, np.ndarray)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if bsz_in_batch is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensor_lst = np.array_split(val, chunks)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(non_tensor_lst) == chunks
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(chunks):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(chunks):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output.append(type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `concat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def concat(data: List["DataProto"]) -> "DataProto":
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for batch in data:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            batch_lst.append(batch.batch)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in non_tensor_batch.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls = type(data[0]) if len(data) > 0 else DataProto
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    # [EXPLAIN] `reorder` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reorder(self, indices):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Note that this operation is in-place
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indices_np = indices.detach().numpy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.batch = self.batch[indices]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    # [EXPLAIN] `repeat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def repeat(self, repeat_times=2, interleave=True):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if interleave:
                # Interleave the data
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Stack the data
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_tensors = {key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:]) for key, tensor in self.batch.items()}

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_batch = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        repeated_non_tensor_batch = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if interleave:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    # [EXPLAIN] `unfold_column_chunks` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def unfold_column_chunks(self, n_split: int, split_keys: Optional[List[str]] = None):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Split along the second dim into `n_split`, unfold it to the first dim (batch dim)
        Useful in passing grouped tensors that doesn't want to be shuffled in dataset.
        keys not in split_keys are repeated to match the shape
        Note that if the `split_keys` is not provided, it will repeat all the keys in the second dim.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unfolded_batch = {}
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key in self.batch.keys():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if key in split_keys if split_keys is not None else False:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    shape = list(self.batch[key].shape)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    shape[0] = self.batch[key].shape[0] * n_split
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    shape[1] = self.batch[key].shape[1] // n_split
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # locate the `unfolded_batch` as a TensorDict on the same device as the original batch
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unfolded_batch = TensorDict(source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unfolded_batch = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        repeated_non_tensor_batch = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key in split_keys:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shape = list(val.shape)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shape[0] = val.shape[0] * n_split
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                shape[1] = val.shape[1] // n_split
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    # [EXPLAIN] `sample_level_repeat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample_level_repeat(self, repeat_times):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Repeat each row of the batch data a specified number of times.

        Args:
            repeat_times (torch.tensor, list, tuple, ndarray):  Number of times to repeat the data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(repeat_times, tuple):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeat_times = list(repeat_times)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(repeat_times, torch.Tensor):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(repeat_times.shape) == 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeat_times = repeat_times.tolist()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif isinstance(repeat_times, np.ndarray):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(repeat_times.shape) == 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeat_times = repeat_times.tolist()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(repeat_times, list), f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        repeat_times = torch.tensor(repeat_times)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.batch is not None:
            # Interleave the data
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_batch = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        repeated_non_tensor_batch = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in self.non_tensor_batch.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `DataProtoFuture` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DataProtoFuture:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible.
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    collect_fn: Callable
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    futures: List[ray.ObjectRef]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dispatch_fn: Callable = None

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `concat` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def concat(data: List[ray.ObjectRef]) -> "DataProtoFuture":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] `chunk` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def chunk(self, chunks: int) -> List["DataProtoFuture"]:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from functools import partial

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        arg_future_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            # [EXPLAIN] `dispatch_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def dispatch_fn(x, i, chunks):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return x.chunk(chunks=chunks)[i]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            arg_future = DataProtoFuture(collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            arg_future_lst.append(arg_future)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return arg_future_lst

    # [EXPLAIN] `get` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = ray.get(self.futures)  # dp_size.
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for o in output:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert isinstance(o, DataProto)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = self.collect_fn(output)  # select dp, concat
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.dispatch_fn is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output


# [EXPLAIN] `all_gather_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def all_gather_data_proto(data: DataProto, process_group):
    # Note that this is an inplace operator just like torch.distributed.all_gather
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group_size = torch.distributed.get_world_size(group=process_group)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(data, DataProto)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prev_device = data.batch.device
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data.batch = data.batch.to(get_torch_device().current_device())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data.batch = data.batch.to(prev_device)
    # all gather non_tensor_batch
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_non_tensor_batch = [None for _ in range(group_size)]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
