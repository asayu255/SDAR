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
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

# 【このファイルの役割（日本語補足）】
# - verl 内の関数・Ray worker・分散rank間で学習バッチを受け渡す共通データ形式を定義する。
# - DataProto は次の3領域を1つにまとめる。
# batch: TensorDict。input_ids、log_probs、advantagesなどGPU計算対象のTensor。
# non_tensor_batch: NumPy配列。task_name、raw_prompt、env_kwargsなどPython objectを含むrow情報。
# meta_info: temperatureやmicro-batch設定など、row方向に分割しないバッチ全体の付帯情報。
# - select/pop/chunk/concat/repeat/reorderは、Tensorとnon-tensorのrow対応を崩さないことが最重要。
# - DataProtoFutureはRay ObjectRefをdriverで即時ray.getせず、worker間処理を遅延接続するための薄いfuture表現。
# - all_gather_data_protoはUlysses SP groupなどで各rankのDataProtoをdim=0方向へ集約する。
# 標準ライブラリ。contextlibはTensorDict互換設定の失敗を安全に無視するために使う。
import contextlib
import copy
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

# NumPy配列は、Tensor化できない文字列・辞書・可変長objectをrow単位で保持する。
import numpy as np
import pandas as pd
# DataProtoFutureが保持するObjectRefとray.getによる遅延回収に使用する。
import ray
# TensorDictは複数Tensorを共通batch_size付きの一つのバッチとして操作する。
import tensordict
import torch
import torch.distributed
from packaging import version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.utils.device import get_torch_device
from verl.utils.py_functional import union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]

# TensorDict 0.5以降の新しい非lazy挙動を有効化する。
# 古いバージョン等でAPIが存在しなくても、プロトコル自体は利用できるよう例外を抑制する。
with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()


# -----------------------------------------------------------------------------
# DataProto全体のpadding設定
# -----------------------------------------------------------------------------
# metaclass propertyにすることで、DataProtoConfig.auto_paddingをインスタンス化せず参照・更新できる。
class _DataProtoConfigMeta(type):
    _config = {}

    auto_padding_key = "_verl_auto_padding"

    # 環境変数はプロセス全体、_configはPython実行中の明示設定として扱い、どちらかがTrueなら有効。
    @property
    def auto_padding(cls):
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    pass


_padding_size_key = "_padding_size_key_x123d"


# -----------------------------------------------------------------------------
# 分散dispatch前のbatch-size padding
# -----------------------------------------------------------------------------
# world_size等で割り切れないbatchを先頭rowの複製で埋め、後でpad_sizeだけ除去できる形にする。
def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    # DataProtoのTensor/non-tensorを一緒に複製する必要があるため、素のTensorは受け付けない。
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        # 元batchより多くpaddingが必要なケースにも対応し、最大len(data)行ずつ複製する。
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        # concatを通すことでbatchとnon_tensor_batchを同じ順序で連結する。meta_infoは先頭を共有する。
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


# dispatch結果を再結合した後、末尾に追加した複製rowだけをsliceで除去する。
def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


# -----------------------------------------------------------------------------
# 同じrow集合へ新しい列を横方向に結合する処理
# -----------------------------------------------------------------------------
# 例: rollout batchへold_log_probsやteacher_log_probsを追加する。dim=0の行数は同一でなければならない。
def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    # 競合keyが存在する場合、暗黙上書きせずTensor内容が完全一致することを要求する。
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    return tensor_dict1


# non_tensor_batch版のunion。object dtypeやNaNを含むため単純なarray_equalではなくDataFrame.equalsを使う。
def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert pd.DataFrame(tensor_dict2[key]).equals(pd.DataFrame(tensor_dict1[key])), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


# [{a:x1,b:y1}, {a:x2,b:y2}]を{a:[x1,x2], b:[y1,y2]}へ転置する補助関数。
def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


# -----------------------------------------------------------------------------
# batch次元のfold / unfold
# -----------------------------------------------------------------------------
# [B,...]を[new_B, B/new_B,...]へ再解釈し、group化されたsampleを同じrowの第2次元へまとめる。
def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    # TensorDict全keyを同じbatch viewへ変換する。Tensor実データの並び順は変えない。
    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


# foldされた先頭batch_dims個の次元を再び単一batch次元へ平坦化する。
def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    Unfold the first n dims as new batch dim
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    # non-tensorも同じreshapeを行い、Tensor側とのrow/group対応を保持する。
    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


# DataLoaderがDataProtoItemのlistを受け取ったとき、Tensorはstackし、Python objectはobject配列へまとめる。
def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


# 単一integer indexの戻り値。batch dimensionを持たない1row分の軽量コンテナ。
# -----------------------------------------------------------------------------
# DataProto本体
# -----------------------------------------------------------------------------
# dim=0を共通のrow軸とし、batchとnon_tensor_batchの全keyが同じ行数・同じ順序を持つ。
# meta_infoはrow軸を持たず、分割された各DataProtoへ原則として同じdict参照が渡される。
@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    # GPU/CPU間をto()で移動するTensor群。TensorDict.batch_size[0]がDataProtoの長さになる。
    batch: TensorDict = None
    # 文字列や辞書を含められるNumPy配列群。各value.shape[0]はbatch sizeと一致する必要がある。
    non_tensor_batch: Dict = field(default_factory=dict)
    # バッチ全体の設定・統計。通常、slice/chunkしてもrowごとには切らず同じ内容を引き継ぐ。
    meta_info: Dict = field(default_factory=dict)

    # dataclass生成直後に構造上の不整合を早期検出する。
    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()

    # TensorがあればTensorDictのbatch_sizeを正とし、無ければnon-tensorの先頭keyから行数を得る。
    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    # index型によって戻り値が異なる点が重要。intだけDataProtoItem、slice/list/maskはDataProtoを返す。
    def __getitem__(self, item):
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
        # data[a:b]はTensorとnon-tensorを同じsliceで切る。
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        # 任意index列やbool maskではselect_idxsへ委譲し、row対応を一括維持する。
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        # DataLoaderのdatasetアクセスとの互換性のため、1rowだけはDataProtoItemとして返す。
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    # Ray/pickle転送時のserialize hook。TensorDictをtorch.save可能な連続表現へ固めてbytes化する。
    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        # Tensorストレージはtorch serialization、NumPy/metaは通常のpickle経路で運ばれる。
        torch.save(self.batch, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    # deserialize hook。acceleratorが無いdriverではCPUへmapし、GPU workerでは保存時deviceを利用可能。
    def __setstate__(self, data):
        import io

        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        batch = torch.load(batch_deserialized, weights_only=False, map_location="cpu" if not get_torch_device().is_available() else None)
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    # 上記__getstate__を利用してDataProto全体をpickleファイルへ保存する。
    def save_to_disk(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    # Tensor領域とNumPy領域の概算メモリをGiB単位で表示する診断用関数。
    def print_size(self, prefix=""):
        size_of_tensordict = 0
        if self.batch is None:
            for key, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for key, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)

    # -------------------------------------------------------------------------
    # 構造不変条件
    # -------------------------------------------------------------------------
    # 通常DataProtoはbatch dimensionを1つだけ持ち、すべてのnon-tensor列のdim=0が同じBである。
    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        # listのまま保持するとTensor側のindex操作と挙動が揃わないため、NumPy配列を必須にする。
        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            # batch[B,...]とnon_tensor_batch[key][B,...]が常にrow-alignedであることを確認する。
            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for {key=}, got {type(val)=}"
                assert val.shape[0] == batch_size, f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    @classmethod
    # TensorとNumPyが混在する1つのdictを型で自動分類し、DataProtoへ変換する入口。
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None, auto_padding=False):
        """Create a DataProto from a dict of tensors and non_tensors"""
        tensors = {}
        non_tensors = {}

        # torch.Tensorはbatchへ、np.ndarrayはnon_tensor_batchへ入る。その他の型は明示的に拒否する。
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    # 明示的に3領域を渡すfactory。全Tensorの先頭num_batch_dims形状が一致することを検証する。
    def from_dict(cls, tensors: Optional[Dict[str, torch.Tensor]] = None, non_tensors=None, meta_info=None, num_batch_dims=1, auto_padding=False):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        # 最初のTensorをpivotとしてbatch shapeを決め、以降の全keyが同じrow軸を持つか確認する。
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        # Python list等はobject dtypeのNumPy配列へ正規化し、後続のslice/concatを統一する。
        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        # TensorDictへbatch_sizeを明示することで、各keyを一つのdatasetの列として操作できる。
        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    # TensorDictだけを指定deviceへ移すin-place操作。NumPy/metaはCPU上に残る。
    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    # -------------------------------------------------------------------------
    # 列方向の抽出
    # -------------------------------------------------------------------------
    # row数は変えず、必要なkey集合だけを持つDataProtoを作る。既定では内部objectを共有する。
    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        # deepcopy=Trueのときのみ、可変なdict/listを含むnon-tensorを独立コピーする。
        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    # -------------------------------------------------------------------------
    # 行方向の任意選択
    # -------------------------------------------------------------------------
    # list / NumPy / Torch indexをTensor用とNumPy用の両形式へ揃え、同じrowを選択する。
    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        # bool maskならTrue数、integer indexならindex個数が新しいTensorDict.batch_sizeになる。
        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            # 全Tensor keyへ同一idxs_torchを適用し、列間のrow順序を完全に同期させる。
            selected_batch = TensorDict(source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,), device=self.batch.device)
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    # 連続またはstep付きsliceをTensor/NumPyの双方へ同じように適用する。meta_infoは共有する。
    def slice(self, start=None, end=None, step=None):
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
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    # 指定keyを元DataProtoから破壊的に取り除き、その列だけを持つ新しいDataProtoとして返す。
    # trainerでprompt列をgen_batchへ移す処理などに使われる。
    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        # TensorDict.popにより元self.batchから列が消えるため、selectとは異なりin-place変更を伴う。
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    # Tensor列名だけをin-place変更する。non_tensor_batch/meta_infoのkeyは対象外。
    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}")

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    # 同一row集合の別DataProtoを列方向にin-place結合する。競合keyは値の一致が必須。
    def union(self, other: "DataProto") -> "DataProto":
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
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    # DataProto自身をPyTorch Datasetとして扱い、int index→DataProtoItem→collate_fnの経路でmini-batch化する。
    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        r"""Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.


        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that ``batch.batch_size[0] % mini_batch_size == 0``.
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs (Any): internally, it returns a DataLoader over the batch. The dataloader_kwargs is the kwargs passed to the DataLoader.

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        # shuffle等はdataloader_kwargsで制御される。seed指定時は専用Generatorで再現性を持たせる。
        train_dataloader = DataLoader(dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs)

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    # 個別DataProtoのmeta設定とプロセス全体設定のORで、不均等chunkを許可するか決める。
    def is_padding_enabled(self):
        """
        Check if padding is enabled for the DataProto.
        Returns:
            bool: True if padding is enabled, False otherwise.
        """
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    # firstまたはlast rowを指定回数複製して末尾へ追加するin-place padding。
    def padding(self, padding_size, padding_candidate=""):
        """Pad the DataProto by concating with padding_candidate.repeat(padding_size)

        Args:
            padding_size (int): the number of repeated padding_candidate
            padding_candidate: the item to be repeated and appended to the DataProto, only supporting ["first", "last"]
        """
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    # -------------------------------------------------------------------------
    # Data-parallel dispatchの中心処理
    # -------------------------------------------------------------------------
    # dim=0をchunks個へ分割し、TensorDictが実際に作った境界をnon-tensorにも同じように適用する。
    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        # padding無効時は全rankへ等しいrow数を送る設計のため、Bがchunksで割り切れることを要求する。
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."

        bsz_in_batch = None
        if self.batch is not None:
            # TensorDict.chunkの結果はpadding有効時などに不均等サイズになり得るため、各chunkの実サイズを記録する。
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            # Tensor側と同じ境界位置をNumPyのarray_splitへ渡し、row対応を維持する。
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        return output

    @staticmethod
    # chunkや各worker出力を逆にdim=0方向へ連結する。全要素のkey集合・並びが同じことを前提とする。
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        # TensorDictはtorch.cat対応のため、各Tensor列が一括して同じ順序で連結される。
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        cls = type(data[0]) if len(data) > 0 else DataProto
        # meta_infoはrow方向に連結せず、先頭DataProtoのdictを採用する。
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    # 任意permutationをTensorとNumPyの両方へ適用するin-place操作。負荷分散用のrow並べ替えで使われる。
    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    # rollout.nなどに合わせて全rowを等回数複製する。
    # interleave=True: [a,a,b,b]、False: [a,b,a,b]の順序になる。
    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                # 同一promptから複数trajectoryを作る場合、各promptを隣接配置するinterleaveがgroup処理に便利。
                repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}
            else:
                # Stack the data
                repeated_tensors = {key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:]) for key, tensor in self.batch.items()}

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    # 第2次元に格納されたgroup/chunkをbatch次元へ展開する。split対象外の列は各展開rowへ複製する。
    def unfold_column_chunks(self, n_split: int, split_keys: Optional[List[str]] = None):
        """Split along the second dim into `n_split`, unfold it to the first dim (batch dim)
        Useful in passing grouped tensors that doesn't want to be shuffled in dataset.
        keys not in split_keys are repeated to match the shape
        Note that if the `split_keys` is not provided, it will repeat all the keys in the second dim.
        """
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    # split_keysは[B,G,...]を[B*n_split,G/n_split,...]へreshapeする。
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    # prompt ID等の非分割列は、展開された各group要素へ同じ値を複製する。
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # locate the `unfolded_batch` as a TensorDict on the same device as the original batch
            unfolded_batch = TensorDict(source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device)
        else:
            unfolded_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    # rowごとに異なる複製回数を指定する可変repeat。filter後のsample補正などに利用できる。
    def sample_level_repeat(self, repeat_times):
        """
        Repeat each row of the batch data a specified number of times.

        Args:
            repeat_times (torch.tensor, list, tuple, ndarray):  Number of times to repeat the data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
        # TorchとNumPyの両repeat APIで同じ回数列を使える共通表現へ正規化する。
        repeat_times = torch.tensor(repeat_times)

        if self.batch is not None:
            # Interleave the data
            repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )


# -----------------------------------------------------------------------------
# Ray futureを保持したままDataProtoのcollect/dispatch計画だけを組み立てる型
# -----------------------------------------------------------------------------
# driverで中間DataProtoを即座に実体化しないため、worker処理同士を非同期に接続できる。
@dataclass
class DataProtoFuture:
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

    # 複数workerのDataProtoを1つへまとめる関数。通常はDataProto.concat。
    collect_fn: Callable
    # 各workerが返した未解決ObjectRef。get()が呼ばれるまでdriverは内容を取得しない。
    futures: List[ray.ObjectRef]
    # collect後のDataProtoから宛先rank用chunkだけを選ぶ任意後処理。
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    # future自体は解決せず、解決後にi番目chunkを選ぶdispatch_fn付きfutureをchunks個作る。
    def chunk(self, chunks: int) -> List["DataProtoFuture"]:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures)
            arg_future_lst.append(arg_future)
        return arg_future_lst

    # ここが実際の同期点。全ObjectRefをray.getし、collect_fn、必要ならdispatch_fnの順に適用する。
    def get(self):
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)  # select dp, concat
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output


# -----------------------------------------------------------------------------
# process group内のDataProto all-gather
# -----------------------------------------------------------------------------
# Tensorはdevice collective、Python objectを含むnon-tensorはall_gather_objectで集めるin-place操作。
# FSDPUlyssesShardingManagerでは、同じSP groupのrankが同じsample集合を持つために使用される。
def all_gather_data_proto(data: DataProto, process_group):
    # Note that this is an inplace operator just like torch.distributed.all_gather
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    # collective実行のため一時的に現在のacceleratorへ移し、完了後は元deviceへ戻す。
    prev_device = data.batch.device
    data.batch = data.batch.to(get_torch_device().current_device())
    # 各Tensor keyをdim=0でrank順に連結する。結果のbatch sizeはlocal_B * group_size。
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data.batch = data.batch.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    # object配列や辞書はTensor collectiveに載せられないため、pickleベースのobject collectiveを使う。
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
