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
This file contains utilities to manipulate torch memory buffers
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn


# [EXPLAIN] `MemoryBuffer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MemoryBuffer:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A memory buffer is a contiguous torch tensor that may combine multiple tensors sharing with the underlying
    memory. It must have a unique type to support this behavior.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, numel: int, numel_padded: int, dtype: torch.dtype, source: Optional[torch.Tensor] = None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.numel = numel
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.numel_padded = numel_padded
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dtype = dtype
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if source is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.data = source
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.data = torch.zeros(self.numel_padded, dtype=self.dtype, device="cuda", requires_grad=False)

    # [EXPLAIN] `zero` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def zero(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reset the buffer to zero."""
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.data.zero_()

    # [EXPLAIN] `get` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get(self, shape, start_index):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end_index = start_index + shape.numel()
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_tensor = self.data[start_index:end_index]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer_tensor = buffer_tensor.view(shape)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return buffer_tensor


# [EXPLAIN] `calc_padded_numel` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """for cuda memory alignment, make sure alignment by 128-bits"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    align_numel = 128 // torch.finfo(dtype).bits
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    numel = shape.numel()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (numel + align_numel - 1) // align_numel * align_numel


# [EXPLAIN] `get_weight_buffer_meta_from_module` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_weight_buffer_meta_from_module(module: nn.Module) -> Dict[str, Dict]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Return a dictionary containing name to a shape and dtype.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weight_buffer_meta = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, param in sorted(module.named_parameters()):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight_buffer_meta[name] = {"shape": param.shape, "dtype": param.dtype}
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return weight_buffer_meta


# [EXPLAIN] `build_memory_buffer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_memory_buffer(weight_buffer_meta: Dict[str, Dict]) -> Dict[torch.dtype, MemoryBuffer]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Build the memory buffer given weight_buffer_meta

    Args:
        weight_buffer_meta: contains mapping from name to a dictionary containing shape and dtype of the tensors

    Returns: a large memory buffer for each dtype that can hold all the tensors

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    memory_buffers = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_numel_map = {}  # map from dtype to the total numel
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, meta_info in sorted(weight_buffer_meta.items()):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        shape = meta_info["shape"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dtype = meta_info["dtype"]

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(shape, torch.Size)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(dtype, torch.dtype)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dtype not in total_numel_map:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_numel_map[dtype] = 0

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        total_numel_map[dtype] += calc_padded_numel(shape, dtype)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dtype, total_numel in total_numel_map.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        memory_buffers[dtype] = MemoryBuffer(total_numel, total_numel, dtype)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return memory_buffers


# [EXPLAIN] `build_memory_reference_from_module` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_memory_reference_from_module(module: torch.nn.Module, memory_buffers: Dict[torch.dtype, MemoryBuffer], maintain_weight=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_index = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dtype in memory_buffers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start_index[dtype] = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, param in sorted(module.named_parameters()):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        memory_buffer = memory_buffers[param.dtype]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer = memory_buffer.get(shape=param.shape, start_index=start_index[param.dtype])
        # need to increment start_index
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        start_index[param.dtype] += calc_padded_numel(param.shape, param.dtype)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if maintain_weight:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            buffer.copy_(param.data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param.data = buffer


# [EXPLAIN] `build_memory_reference` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_memory_reference(weight_buffer_meta: Dict[str, Dict], memory_buffers: Dict[torch.dtype, MemoryBuffer]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Build the memory references. The memory buffers are built using the build_memory_buffer API.
    This API will allocate a weight buffer pointer to the memory buffer according to the weight_buffer_meta.

    Args:
        weight_buffer_meta:
        memory_buffers:

    Returns:

    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    start_idx = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weight_buffers = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for dtype in memory_buffers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start_idx[dtype] = 0

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for name, meta_info in sorted(weight_buffer_meta.items()):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        shape = meta_info["shape"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dtype = meta_info["dtype"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        buffer = memory_buffers[dtype].get(shape, start_index=start_idx[dtype])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        start_idx[dtype] += calc_padded_numel(shape, dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight_buffers[name] = buffer

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return weight_buffers


# [EXPLAIN] `MemoryBufferModuleWrapper` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MemoryBufferModuleWrapper:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that we do not design MemoryBufferModuleWrapper as an nn.Module due to
    - It will change the checkpoint name
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, module: nn.Module):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.module = module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weight_buffer_meta = get_weight_buffer_meta_from_module(self.module)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        build_memory_reference_from_module(self.module, self.memory_buffers)

    # [EXPLAIN] `get_memory_buffers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_memory_buffers(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.memory_buffers

    # [EXPLAIN] `get_weight_buffer_meta` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_weight_buffer_meta(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.weight_buffer_meta


# [EXPLAIN] `MegatronMemoryBufferForRollout` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronMemoryBufferForRollout:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    We assume that
    - inference engine has tp + dp
    - actor has tp + pp + dp
    - the tp between inference engine and actor should be the same
    - memory_buffers: contains a list of memory_buffers, each is a dict from dtype to MemoryBuffer
    - weight_buffers: contains a list of weight_buffers, each is a dict from name to param
    - named_parameters: a dict from name to parameter that normalizes the names from pp and vpp. Note that
        the named_parameters may not be directly compatible with inference engine. User has to take care of
        this part such as the layout mismatches. (e.g. qkv transpose)
    - Note that weight_buffer, named_parameters and memory_buffers share the same underlying GPU memory.
    - When doing weight sync, the data is transfer via memory buffers
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, transform_memory_param_fn):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._memory_buffers = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._weight_buffers = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._named_parameters = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.transform_memory_param_fn = transform_memory_param_fn

    # [EXPLAIN] `initialize_weight_buffer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def initialize_weight_buffer(self, weight_buffer_meta_pp: List[Dict[str, Dict]]):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Initialize the weight buffer. The weight buffer is obtained according to the actor. We will construct
        a large buffer for each dtype in the weight_buffer.

        Args:
            weight_buffer_meta: contains pp models, each pp models contains a dictionary of mapping from

        Returns: None

        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weight_buffer_meta_pp = weight_buffer_meta_pp

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for weight_buffer_meta in self.weight_buffer_meta_pp:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            memory_buffer = build_memory_buffer(weight_buffer_meta)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._memory_buffers.append(memory_buffer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._weight_buffers.append(None)

    # [EXPLAIN] `build_memory_reference` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def build_memory_reference(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, weight_buffer_meta in enumerate(self.weight_buffer_meta_pp):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._weight_buffers[i] = build_memory_reference(weight_buffer_meta, self._memory_buffers[i])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._named_parameters = self.transform_memory_param_fn(self._weight_buffers)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `named_parameters` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def named_parameters(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._named_parameters

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `weight_buffers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def weight_buffers(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._weight_buffers

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `memory_buffers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def memory_buffers(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._memory_buffers
