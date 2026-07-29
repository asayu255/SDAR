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
Contain small torch utilities
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import math
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import contextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.optim import Optimizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.optim.lr_scheduler import LambdaLR
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


# [EXPLAIN] `gather_from_labels` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def gather_from_labels(data, label):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `logprobs_from_logits` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logprobs_from_logits(logits, labels, inplace_backward=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_dim = logits.shape[:-1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        last_dim = logits.shape[-1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.reshape(-1, last_dim)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        labels = labels.reshape(-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.view(*batch_dim)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = logprobs_from_logits_v2(logits, labels)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `logprobs_from_logits_flash_attn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(output, tuple), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return -output[0]


# [EXPLAIN] `logprobs_from_logits_naive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logprobs_from_logits_naive(logits, labels):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logp = F.log_softmax(logits, dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logpy = gather_from_labels(logp, labels)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logpy


# [EXPLAIN] `logprobs_from_logits_v2` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A memory efficient implementation of logprobs_from_logits
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if logits.dtype in [torch.float32, torch.float64]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logprobs_labels = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logprobs_labels.append(row_logprobs_labels)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logprobs_labels = torch.stack(logprobs_labels)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logprobs_labels


# [EXPLAIN] `clip_by_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def clip_by_value(x, tensor_min, tensor_max):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return clipped


# [EXPLAIN] `entropy_from_logits` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def entropy_from_logits(logits: torch.Tensor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Calculate entropy from logits."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pd = torch.nn.functional.softmax(logits, dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return entropy


# [EXPLAIN] `masked_sum` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def masked_sum(values, mask, axis=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute mean of tensor with a masked values."""
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (values * mask).sum(axis=axis)


# [EXPLAIN] `masked_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def masked_mean(values, mask, axis=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


# [EXPLAIN] `masked_var` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def masked_var(values, mask, unbiased=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute variance of tensor with masked values."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean = masked_mean(values, mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    centered_values = values - mean
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    variance = masked_mean(centered_values**2, mask)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if unbiased:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_sum = mask.sum()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mask_sum == 0:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mask_sum == 1:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bessel_correction = mask_sum / (mask_sum - 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        variance = variance * bessel_correction
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return variance


# [EXPLAIN] `masked_whiten` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def masked_whiten(values, mask, shift_mean=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not shift_mean:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        whitened += mean
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return whitened


# [EXPLAIN] `get_response_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_response_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g.
    response_id = torch.tensor([[20, 10, 34, 1, 0, 0, 0],
                                [78, 0, 76, 2, 1, 0, 0],
                                [23, 98, 1, 0, 0, 0, 0],
                                [33, 3, 98, 45, 1, 0, 0]])
    #eos_token=1
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    #eos_token=[1,2]
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device)).int()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)


# [EXPLAIN] `compute_grad_norm` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_grad_norm(model: nn.Module):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_grad_square = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for param in model.parameters():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if param.grad is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return total_grad_square


# [EXPLAIN] `broadcast_dict_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def broadcast_dict_tensor(tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    TODO: optimize this. Technically, we only need one broadcast
    """

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in tensors.sorted_keys:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


# [EXPLAIN] `allgather_dict_tensors` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(tensors, TensorDict):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_tensor_dict = True
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors_as_dict = tensors.to_dict()
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors_as_dict = tensors
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        is_tensor_dict = False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sorted_keys = sorted(tensors_as_dict.keys())
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in sorted_keys:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        val = tensors_as_dict[key]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output[key] = [torch.empty_like(val) for _ in range(size)]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output[key] = torch.cat(output[key], dim=dim)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_tensor_dict:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `split_dict_tensor_into_batches` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert tensors.batch_size[0] % batch_size == 0, f"input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensors.split(batch_size)


# [EXPLAIN] `pad_2d_list_to_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = max(len(sub_list) for sub_list in response)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor = torch.tensor(padded_response)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor


# [EXPLAIN] `pad_sequence_to_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tensors.shape[-1] >= max_seq_len:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensors
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


# [EXPLAIN] `postprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        input_ids: Token indices [batch_size, seq_len]
        attention_mask: Mask [batch_size, seq_len]
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: "left", "right" or "error"

    Returns:
        (input_ids, attention_mask) padded/truncated to max_length
    """
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert truncation in ["left", "right", "middle", "error"]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert input_ids.ndim == 2

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_length = input_ids.shape[-1]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sequence_length < max_length:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = pad_sequence_to_length(input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif sequence_length > max_length:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if truncation == "left":
            # actually, left truncation may not be reasonable
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = input_ids[:, -max_length:]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = attention_mask[:, -max_length:]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif truncation == "right":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = input_ids[:, :max_length]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = attention_mask[:, :max_length]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif truncation == "middle":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            left_half = max_length // 2
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            right_half = max_length - left_half
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif truncation == "error":
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_ids, attention_mask


# [EXPLAIN] `tokenize_and_postprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def tokenize_and_postprocess_data(prompt: str, tokenizer: PreTrainedTokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tokenize text and process outputs to consistent tensor shapes.

    Args:
        prompt: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: Truncation strategy ("left"/"right"/"error")

    Returns:
        Tuple of (input_ids, attention_mask) from postprocess_data
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = input_data["input_ids"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = input_data["attention_mask"]

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return postprocess_data(input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation)


# [EXPLAIN] `remove_pad_token` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Remove the pad token.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    no_padding_batch = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for ids, mask in zip(input_ids, attention_mask):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        no_padding_batch.append((ids[len(ids) - mask.sum() :]).cpu().numpy().tolist())
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return no_padding_batch


# [EXPLAIN] `log_probs_from_logits_response` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_probs_from_logits_response(input_ids, logits, response_length):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)

    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]

    Returns:
        response_log_prob:
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_logits = logits[:, -response_length - 1 : -1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = input_ids[:, -response_length:]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return response_log_prob


# [EXPLAIN] `log_probs_from_logits_response_rmpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import pad_input, unpad_input

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size, seqlen = input_ids.shape
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `log_probs_from_logits_all_rmpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import pad_input

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output


# [EXPLAIN] `post_process_logits` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if temperature != 1.0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logits


# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Optimizer related
"""


# [EXPLAIN] `get_cosine_schedule_with_warmup` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    coef = (1 - min_lr_ratio) * 0.5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    intercept = (1 + min_lr_ratio) * 0.5

    # [EXPLAIN] `lr_lambda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def lr_lambda(current_step):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_step < num_warmup_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return min_lr_ratio + (1.0 - min_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return max(min_lr_ratio, x * coef + intercept)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# [EXPLAIN] `get_constant_schedule_with_warmup` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create a constant LR schedule with a linear warmup phase.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of steps to ramp up the LR from 0 to initial value.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.

    Returns:
        LambdaLR: Scheduler that increases LR linearly during warmup, then holds it constant.
    """

    # [EXPLAIN] `lr_lambda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def lr_lambda(current_step):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_step < num_warmup_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return float(current_step) / float(max(1.0, num_warmup_steps))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return 1.0

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# [EXPLAIN] `prepare_decoder_attention_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    combined_attention_mask = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if input_shape[-1] > 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# [EXPLAIN] `_make_causal_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Make causal mask used for bi-directional self-attention.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, tgt_len = input_ids_shape
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask_cond = torch.arange(mask.size(-1), device=device)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = mask.to(dtype)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
# [EXPLAIN] `_expand_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bsz, src_len = mask.size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tgt_len = tgt_len if tgt_len is not None else src_len

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inverted_mask = 1.0 - expanded_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# [EXPLAIN] `get_unpad_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_unpad_data(attention_mask):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# [EXPLAIN] `get_wsd_schedule_with_warmup` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Create a Warmup-Stable-Decay learning rate scheduler.

    The schedule follows three phases:
    1. Warmup: Learning rate increases linearly from 0 to the initial LR
    2. Stable: Learning rate remains constant at the initial LR
    3. Decay: Learning rate decreases following a cosine curve to min_lr_ratio * initial LR

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum learning rate ratio w.r.t the initial learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule during decay phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        stable_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio of non-warmup steps that should maintain a constant learning rate.
            Set to 0.0 to behave exactly like cosine schedule.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_stable_steps = int(remaining_steps * stable_ratio)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_decay_steps = remaining_steps - num_stable_steps

    # [EXPLAIN] `lr_lambda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def lr_lambda(current_step):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_step < num_warmup_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return float(current_step) / float(max(1, num_warmup_steps))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_step < num_warmup_steps + num_stable_steps:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 1.0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_step < num_training_steps:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return min_lr_ratio

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@contextmanager
# [EXPLAIN] `check_cuda_is_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def check_cuda_is_available():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Some modules must be imported after CUDA is initialized. Such as sglang's sharding manager.

    This context manager checks if CUDA is available and raises an error if it is not.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not torch.cuda.is_available():
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError("CUDA must be initialized before importing this module.")

    # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
    yield


# [EXPLAIN] `distributed_mean_max_min_std` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def distributed_mean_max_min_std(local_tensor, compute_max=True, compute_min=True, compute_std=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute distributed statistics across all processes.

    Args:
        local_tensor: Tensor containing local values
        compute_max: Include maximum value calculation
        compute_min: Include minimum value calculation
        compute_std: Include standard deviation calculation

    Returns:
        Tuple containing (mean, max, min, std) in this order. None for disabled metrics.
    """
    # Sum the local tensor across all processes
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_sum = torch.sum(local_tensor)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_num = torch.tensor(torch.numel(local_tensor), device="cuda")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_mean = local_sum / local_num

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if compute_max:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_max = torch.max(local_tensor)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_max = None

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if compute_min:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_min = torch.min(local_tensor)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_reduce(local_min, op=torch.distributed.ReduceOp.MIN)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_min = None

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if compute_std:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        square_diff = torch.sum(torch.pow(local_tensor - global_mean, 2))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.all_reduce(square_diff, op=torch.distributed.ReduceOp.SUM)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_std = torch.sqrt(square_diff / (local_num - 1))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_std = None

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return global_mean, local_max, local_min, global_std


# [EXPLAIN] `distributed_masked_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def distributed_masked_mean(local_tensor, local_mask):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute global mean of non-masked elements across distributed processes.

    Args:
        local_tensor (torch.Tensor): Input tensor with local values
        local_mask (torch.Tensor): Binary mask (1=valid, 0=ignore) matching local_tensor shape

    Returns:
        torch.Tensor: Global mean of all valid elements across processes
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_tensor = local_tensor * local_mask

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_sum = torch.sum(local_tensor)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_num = torch.sum(local_mask)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_mean = local_sum / local_num
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return global_mean
