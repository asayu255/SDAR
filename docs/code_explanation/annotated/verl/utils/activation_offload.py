# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Functionality for CPU offloading of tensors saved for backward pass."""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from __future__ import annotations

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import functools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import FSDPModule as FSDP2

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `_get_unique_tensor_key` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_unique_tensor_key(tensor):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return key


# [EXPLAIN] `FSDPParameterFilter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPParameterFilter:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_parameters_storage = set()

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, tensor):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    # [EXPLAIN] `update_model_parameters` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_model_parameters(self, model):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        new_storage = set()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for p in model.parameters():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            new_storage.add(p.data.untyped_storage().data_ptr())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_parameters_storage = new_storage


# [EXPLAIN] `CpuOffloadHookWithOffloadHandler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CpuOffloadHookWithOffloadHandler:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if handler_extra_kwargs is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            handler_extra_kwargs = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.offload_handler: OffloadHandler = offload_handler
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.handler_extra_kwargs: dict[str, Any] = handler_extra_kwargs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inside_context = False

    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inside_context = True
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch._C._autograd._push_saved_tensors_default_hooks(self.on_save_for_backward, self.on_get_saved_tensor)

    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, *args: Any):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inside_context = False
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch._C._autograd._pop_saved_tensors_default_hooks()

    # [EXPLAIN] `on_save_for_backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return retrieve_identifier

    # [EXPLAIN] `on_get_saved_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor


# [EXPLAIN] `OffloadHandler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class OffloadHandler:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A base class for CPU offload-handler."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self) -> None:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] `tensor_push` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Tensor push."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("`tensor_push is not implented in OffloadHandler class. Inherit this class and implement your custom tensor_push.")

    # [EXPLAIN] `tensor_pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_pop(self, tensor_tag: Any, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Tensor pop."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("`tensor_pop is not implented in OffloadHandler class. Inherit this class and implement your custom tensor_pop.")


# [EXPLAIN] `GroupCommitFunction` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class GroupCommitFunction(torch.autograd.Function):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(ctx, tensor, cpu_offload_handler):
        # pylint: disable=missing-function-docstring
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cpu_offload_handler.on_group_commit_forward()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cpu_offload_handler = ctx.cpu_offload_handler
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cpu_offload_handler.on_group_commit_backward()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return grad_output, None


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
group_prefetch_offload_commit = GroupCommitFunction.apply


# [EXPLAIN] `SynchronizedGroupOffloadHandler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SynchronizedGroupOffloadHandler(OffloadHandler):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, num_offload_group, tensor_need_offloading_checker=(lambda _: True)) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_offload_group = num_offload_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tensor_need_offloading_checker = tensor_need_offloading_checker

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.groupid_reset()

    # [EXPLAIN] `groupid_reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def groupid_reset(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_group, self.tensor_count_current_group = (0, 0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.torch_tensor_count = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tensor_tag_to_state = {}

    # [EXPLAIN] `on_group_commit_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_group_commit_forward(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_group += 1  # increment
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tensor_count_current_group = 0  # reset

    # [EXPLAIN] `on_group_commit_backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_group_commit_backward(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """On group commit backward."""
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_group -= 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.current_group >= 0

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `offload` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload(src_tensor, pin_memory=True):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Offload."""

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cpu_backup.copy_(src_tensor, non_blocking=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state = (src_tensor.device, cpu_backup)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return state

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `reload` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reload(state, non_blocking=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reload."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dev, cpu_backup = state
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if non_blocking is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_blocking = cpu_backup.is_pinned()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cpu_backup.to(dev, non_blocking=non_blocking)

    # [EXPLAIN] `tensor_push` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Tensor push."""
        # obtain a unique tensor tag
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.tensor_count_current_group += 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert tensor_tag not in self.tensor_tag_to_state
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.tensor_tag_to_state[tensor_tag] = state
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # will be offloaded together after group commit
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.tensor_tag_to_state[tensor_tag] = tensor

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor_tag

    # [EXPLAIN] `tensor_pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_pop(self, tensor_tag, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Tensor pop."""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert tensor_tag in self.tensor_tag_to_state
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state = self.tensor_tag_to_state.pop(tensor_tag)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(state, tuple):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor = state
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor


# [EXPLAIN] `AsyncDoubleBufferGroupOffloadHandler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
        )
        # Number of layers in the model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_layers = num_model_group
        # Data Structure to maintain reference to activation tensors
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tensor_tag_to_buf = {}
        # Tracking the number of layers offloaded
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.offloaded_group_count = 0
        # Core data structure that decides the window for offloading
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layer_window_map = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.group_offload_mapping = {}

        # Logic to make offloading load balance across computation
        # for optimal CPU/GPU interconnect usage
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        constant = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(self.num_offload_group):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i < (self.num_layers % self.num_offload_group):
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.layer_window_map[i] += i + 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                constant = i + 1
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.layer_window_map[i] += constant

        # allocate streams and events for synchronization
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.d2h_stream = torch.cuda.Stream()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.h2d_stream = torch.cuda.Stream()

    # [EXPLAIN] `tensor_push` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        need_offload = not torch_stray_tensor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        need_offload = need_offload and self.tensor_need_offloading_checker(tensor)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if need_offload:
            # obtain a unique tensor tag
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.tensor_count_current_group += 1

            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert tensor_tag not in self.tensor_tag_to_state
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.tensor_tag_to_state[tensor_tag] = tensor

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.current_group < self.num_offload_group:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.tensor_tag_to_buf[tensor_tag] = tensor
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_tag = tensor
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor_tag

    # [EXPLAIN] `tensor_pop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def tensor_pop(self, tensor_tag, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Tensor pop."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(tensor_tag, torch.Tensor):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return tensor_tag
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert tensor_tag in self.tensor_tag_to_state
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert not isinstance(tensor, tuple)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor

    # [EXPLAIN] `bulk_offload_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def bulk_offload_group(self, group_to_offload):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Bulk offload group."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        offload_mapping = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        offload_size = 0
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.cuda.stream(self.d2h_stream):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for tensor_tag, state in self.tensor_tag_to_state.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                group_id, _ = tensor_tag
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if group_id == group_to_offload:
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert not isinstance(state, tuple)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    key = _get_unique_tensor_key(state)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if key not in offload_mapping:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        offload_mapping[key] = state
                    # if offload, return the reference to cpu copy
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, tensor in offload_mapping.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state = SynchronizedGroupOffloadHandler.offload(tensor)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_size += tensor.numel() * tensor.element_size()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                offload_mapping[key] = state

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.group_offload_mapping[group_to_offload] = offload_mapping

    # [EXPLAIN] `synchronize_on_group_commit_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def synchronize_on_group_commit_forward(self, current_group):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Synchronize on group commit forward."""

        # For the first group, kickstart the offload after we have
        # the first compute completion
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_group == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.bulk_offload_group(current_group)

        # Window map data structure helps us synchronize based on number
        # of layers offloaded
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.layer_window_map[self.offloaded_group_count] == current_group:
            # Stream synchronization both ways
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

            # Time to free the activation memory after usage
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if tensor_tag[0] == self.offloaded_group_count:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.tensor_tag_to_buf[tensor_tag] = None

            # Time to offload the next group
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.offloaded_group_count < (self.num_offload_group - 1):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.bulk_offload_group(self.offloaded_group_count + 1)

            # Increment the offload group count to keep track
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.offloaded_group_count += 1

    # [EXPLAIN] `on_group_commit_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_group_commit_forward(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """This function will cause host device synchronization"""
        # handle synchronization events
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.synchronize_on_group_commit_forward(self.current_group)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().on_group_commit_forward()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad
    # [EXPLAIN] `bulk_reload_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def bulk_reload_group(self, group_to_reload):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Bulk reload group."""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert group_to_reload < self.num_offload_group

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            offload_mapping = self.group_offload_mapping.pop(group_to_reload)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert offload_mapping is not None
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, state in offload_mapping.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                offload_mapping[key] = SynchronizedGroupOffloadHandler.reload(state)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for tensor_label, state in self.tensor_tag_to_state.items():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                group_id, _ = tensor_label
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if group_id == group_to_reload and not isinstance(state, torch.Tensor):
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert isinstance(state, tuple), f"{group_id} {state}"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    key, shape = state
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    recovered_tensor = offload_mapping[key].view(shape)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    # [EXPLAIN] `on_group_commit_backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_group -= 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.current_group >= 0

        # Layer window data structure helps us to reload at right times
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            # Stream synchronization both ways
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.current_stream().wait_stream(self.h2d_stream)

            # Time to reload the next group
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrease the offloading group counter
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # Last group computation needs to wait till all the reloads complete
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.current_group == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.offloaded_group_count = 0


# [EXPLAIN] `get_activation_offload_context` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_activation_offload_context(num_layers: int = 1, model_layers: int = 1, tensor_need_offloading_checker=(lambda t: True)):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    # [EXPLAIN] `group_prefetch_offload_commit_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def group_prefetch_offload_commit_async(tensor):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )


# [EXPLAIN] `ActivationHandler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ActivationHandler:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._offload_ctx = offload_ctx
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._sync_func = sync_func
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._enable_ckpt = enable_ckpt
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._tensor_filter = tensor_filter
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if enable_ckpt:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.checkpoint_fn = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=True,
            )

    # [EXPLAIN] `pre_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pre_forward(self, module):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if module.training:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._offload_ctx.__enter__()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._tensor_filter.update_model_parameters(module)

    # [EXPLAIN] `post_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def post_forward(self, module):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if module.training:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._offload_ctx.__exit__(None, None, None)

    # [EXPLAIN] `_pack_kwargs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _pack_kwargs(self, *args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwarg_keys = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_args = list(args)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in kwargs.items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            kwarg_keys.append(k)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            flat_args.append(v)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tuple(flat_args), tuple(kwarg_keys)

    # [EXPLAIN] `_unpack_kwargs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _unpack_kwargs(self, flat_args, kwarg_keys):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(kwarg_keys) == 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return flat_args, {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        args = flat_args[: -len(kwarg_keys)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return args, kwargs

    # [EXPLAIN] `_ckpt_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _ckpt_forward(self, forward_method, *args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        flat_args, kwarg_keys = self._pack_kwargs(*args, **kwargs)

        # [EXPLAIN] `my_function` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def my_function(*inputs):
            # unpack back into args and kwargs
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            nonlocal forward_method, kwarg_keys
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            unpacked_args, unpacked_kwargs = self._unpack_kwargs(inputs, kwarg_keys)
            # run original module
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return forward_method(*unpacked_args, **unpacked_kwargs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.checkpoint_fn(
            my_function,
            *flat_args,
        )

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, module, forward_method, *args, **kwargs):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not module.training:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return forward_method(*args, **kwargs)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self._enable_ckpt:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ret = forward_method(*args, **kwargs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ret = self._ckpt_forward(forward_method, *args, **kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        binded_tensor = ret
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(ret, tuple):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            binded_tensor = ret[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        binded_tensor = self._sync_func(binded_tensor)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_ret = binded_tensor
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(ret, tuple):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_ret = (final_ret,) + ret[1:]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return final_ret

    # [EXPLAIN] `wrap_module_forward_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def wrap_module_forward_method(self, module):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        orig_method = module.forward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        handler = self

        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @functools.wraps(orig_method)
        # [EXPLAIN] `wrapped_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def wrapped_method(model_self, *args, **kwargs):
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            nonlocal handler
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            handler.pre_forward(model_self)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            handler.post_forward(model_self)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return out

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module.forward = wrapped_method.__get__(module, type(module))


# [EXPLAIN] `enable_activation_offloading` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def enable_activation_offloading(model, strategy, enable_ckpt=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Enable activation offloading for the model. It groups activations by TransformerLayer and offloads activation
    groups asynchronously. This means that the offloading of the i-th activation group and the computation of the i+1-th
    activation group happen at the same time, and there are at most two activation groups in GPU memory.

    Args:
        model: the model to enable activation offloading
        strategy: the training strategy of the model, such as "fsdp"
        enable_ckpt: whether activation checkpointing(also called gradient checkpointing) has been enabled for the model

    Note:
        For best efficiency, activation offloading is usually combined with activation checkpointing. However, this
        implementation of activation offloading is conflicted with the implementation of activation checkpointing in
        some training strategies. This function resolves this conflict, and therefore requires the "strategy" and
        "enable_ckpt" arguments.

    Returns:

    """

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert strategy == "fsdp" or strategy == "fsdp2", "activation offloading only supports fsdp strategy"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    layers = []

    # [EXPLAIN] `get_layers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_layers(module):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for name, child in module.named_children():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not isinstance(child, (FSDP, FSDP2)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                get_layers(child)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                wrapped_module = child
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(child, FSDP):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    wrapped_module = child._fsdp_wrapped_module
                # In some cases, torch.nn.Embedding is wrapped with FSDP alone. However, the activation
                # size of torch.nn.Embedding is small, so it's not necessary to offload it.
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    layers.append(child)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    get_layers(model)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(layers) < 3:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning(f"Find only {len(layers)} fsdp layers, not neccessary to enable async activation offloading")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_filter = FSDPParameterFilter()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    context, sync_func = get_activation_offload_context(len(layers) - 1, len(layers), tensor_filter)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if enable_ckpt:
        # The implementation of activation checkpointing in transformers library is incompatible with activation offloading,
        # so it will be disabled, but this implementation supports another version of activation checkpointing, so that
        # these two features can be enabled at the same time.
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for module in model.modules():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(module, "gradient_checkpointing_disable"):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                module.gradient_checkpointing_disable()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for layer in layers:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = layer
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(layer, FSDP):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            module = module._fsdp_wrapped_module
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        handler.wrap_module_forward_method(module)
