# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)


# [EXPLAIN] `is_torch_npu_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_torch_npu_available() -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Check the availability of NPU"""
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import torch_npu  # noqa: F401

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.npu.is_available()
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ImportError:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
is_cuda_available = torch.cuda.is_available()
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
is_npu_available = is_torch_npu_available()


# [EXPLAIN] `get_visible_devices_keyword` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_visible_devices_keyword() -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


# [EXPLAIN] `get_device_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_device_name() -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_cuda_available:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = "cuda"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif is_npu_available:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = "npu"
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = "cpu"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return device


# [EXPLAIN] `get_torch_device` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_torch_device() -> any:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_name = get_device_name()
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return getattr(torch, device_name)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except AttributeError:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.cuda


# [EXPLAIN] `get_device_id` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_device_id() -> int:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Return current device id based on the device type.
    Returns:
        device index
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return get_torch_device().current_device()


# [EXPLAIN] `get_nccl_backend` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_nccl_backend() -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_cuda_available:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "nccl"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif is_npu_available:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "hccl"
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")


# [EXPLAIN] `set_expandable_segments` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_expandable_segments(enable: bool) -> None:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_cuda_available:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
