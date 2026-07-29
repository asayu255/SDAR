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
Utilities for DeepSpeed Ulysses Sequence Parallelism.
DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Optional, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import Tensor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed import ProcessGroup

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


# [EXPLAIN] `set_ulysses_sequence_parallel_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Set ulysses sequence parallel process group.
    """
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


# [EXPLAIN] `get_ulysses_sequence_parallel_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get ulysses sequence parallel process group.
    """
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


# [EXPLAIN] `get_ulysses_sequence_parallel_world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get ulysses sequence parallel world size.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dist.get_world_size(group) if group else 1


# [EXPLAIN] `get_ulysses_sequence_parallel_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get ulysses sequence parallel rank.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dist.get_rank(group) if group else 0


# [EXPLAIN] `gather_seq_scatter_heads` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not group:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_size = x.size(seq_dim) - unpadded_dim_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = _unpad_tensor(x, seq_dim, padding_size)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x


# [EXPLAIN] `gather_heads_scatter_seq` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not group:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dim_size = x.size(seq_dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dim_size % sp_world != 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_size = sp_world - (dim_size % sp_world)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = _pad_tensor(x, seq_dim, padding_size)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


# [EXPLAIN] `_pad_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape = list(x.shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shape[dim] = padding_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.cat([x, pad], dim=dim)


# [EXPLAIN] `_unpad_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    slc = [slice(None)] * len(x.shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    slc[dim] = slice(0, -padding_size)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x[slc]


# [EXPLAIN] `slice_input_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_world_size = dist.get_world_size(group)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_rank = get_ulysses_sequence_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dim_size = x.size(dim)
    # pad before slice
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if padding and dim_size % sp_world_size:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        padding_size = sp_world_size - (dim_size % sp_world_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parts = x.size(dim) // sp_world_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    slc = [slice(None)] * len(x.shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x[slc].contiguous()


# [EXPLAIN] `all_to_all_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_world_size = dist.get_world_size(group)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if async_op:

        # [EXPLAIN] `wait` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def wait():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            comm.wait()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.cat(output_list, dim=gather_dim).contiguous()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return wait
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.cat(output_list, dim=gather_dim).contiguous()


# [EXPLAIN] `all_gather_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sp_world_size = dist.get_world_size(group=group)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_shape = list(local_tensor.shape)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_shape[0] = output_shape[0] * sp_world_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `SeqAllToAll` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SeqAllToAll(torch.autograd.Function):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.group = group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.scatter_dim = scatter_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.gather_dim = gather_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.async_op = async_op
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


# [EXPLAIN] `Gather` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Gather(torch.autograd.Function):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.group = group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.gather_dim = gather_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.grad_scaler = grad_scaler
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.async_op = async_op

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sp_world_size = dist.get_world_size(group=group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.sp_world_size = sp_world_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sp_rank = dist.get_rank(group=group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.sp_rank = sp_rank

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_shape = list(local_tensor.size())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        split_size = local_shape[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        part_size = local_shape[gather_dim]  # store original size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.part_size = part_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = all_gather_tensor(local_tensor, group, async_op)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ctx.grad_scaler:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            grad_output = grad_output * ctx.sp_world_size
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


# [EXPLAIN] `gather_outpus_and_unpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gather_outpus_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if group is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if unpad_dim is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if padding_size == 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return x
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = _unpad_tensor(x, unpad_dim, padding_size)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return x

# [EXPLAIN] `ulysses_pad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def ulysses_pad(
    input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1
):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids_rmpad is not None:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert position_ids_rmpad.size(0) == 1
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert input_ids_rmpad.size(1) == position_ids_rmpad.size(1)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sp_size <= 1:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return input_ids_rmpad, position_ids_rmpad, 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, total_seq_len = input_ids_rmpad.shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pad_size > 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if position_ids_rmpad is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if position_ids_rmpad.dim() == 3:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_ids_rmpad, position_ids_rmpad, pad_size

# [EXPLAIN] `ulysses_pad_and_slice_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def ulysses_pad_and_slice_inputs(input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
        input_ids_rmpad, position_ids_rmpad, sp_size
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if position_ids_rmpad is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_ids_rmpad, position_ids_rmpad, pad_size


# [EXPLAIN] `validate_ulysses_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def validate_ulysses_config(num_heads, ulysses_sequence_size):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ulysses_sequence_size > 1:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert num_heads % ulysses_sequence_size == 0, f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"
