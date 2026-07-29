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
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import datetime
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.logger.aggregate_logger import DecoratorLoggerBase
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_torch_device


# [EXPLAIN] `_get_current_mem_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> Tuple[str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Get current memory usage."""
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert unit in ["GB", "MB", "KB"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_allocated = get_torch_device().memory_allocated()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_free, mem_total = get_torch_device().mem_get_info()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_used = mem_total - mem_free
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_used = f"{mem_used / divisor:.{precision}f}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mem_total = f"{mem_total / divisor:.{precision}f}"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mem_allocated, mem_reserved, mem_used, mem_total


# [EXPLAIN] `log_gpu_memory_usage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        message = f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if logger is None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(message)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.log(msg=message, level=level)


# [EXPLAIN] `GPUMemoryLogger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class GPUMemoryLogger(DecoratorLoggerBase):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.debug.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if dist.is_initialized() and dist.get_world_size() > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = dist.get_rank()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = 0
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(role, logger, level, rank, log_only_rank_0)

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, decorated_function: callable):
        # [EXPLAIN] `f` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def f(*args, **kwargs):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.log(decorated_function, *args, **kwargs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, func, *args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        name = func.__name__
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        message = f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.logging_function(message)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = func(*args, **kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        message = f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.logging_function(message)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

# [EXPLAIN] `log_print` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_print(ctn: Any):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    frame = inspect.currentframe().f_back
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    function_name = frame.f_code.co_name
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    line_number = frame.f_lineno
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    file_name = frame.f_code.co_filename.split('/')[-1]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"[{file_name}:{line_number}:{function_name}]: {ctn}")