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
Contain small python utility functions
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import multiprocessing
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import queue  # Import the queue module for exception type hint
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import signal
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import wraps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from types import SimpleNamespace
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Dict, Iterator, Optional, Tuple


# --- Top-level helper for multiprocessing timeout ---
# This function MUST be defined at the top level to be pickleable
# [EXPLAIN] `_mp_target_wrapper` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: Tuple, kwargs: Dict[str, Any]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Internal wrapper function executed in the child process.
    Calls the original target function and puts the result or exception into the queue.
    """
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = target_func(*args, **kwargs)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        mp_queue.put((True, result))  # Indicate success and put result
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # Ensure the exception is pickleable for the queue
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import pickle

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            pickle.dumps(e)  # Test if the exception is pickleable
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mp_queue.put((False, e))  # Indicate failure and put exception
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except (pickle.PicklingError, TypeError):
            # Fallback if the original exception cannot be pickled
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mp_queue.put((False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}")))


# Renamed the function from timeout to timeout_limit
# [EXPLAIN] `timeout_limit` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def timeout_limit(seconds: float, use_signals: bool = False):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).
    """

    # [EXPLAIN] `decorator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decorator(func):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_signals:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if os.name != "posix":
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            # Issue deprecation warning if use_signals is explicitly True
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
            @wraps(func)
            # [EXPLAIN] `wrapper_signal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def wrapper_signal(*args, **kwargs):
                # [EXPLAIN] `handler` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
                def handler(signum, frame):
                    # Update function name in error message if needed (optional but good practice)
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                old_handler = signal.getsignal(signal.SIGALRM)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.signal(signal.SIGALRM, handler)
                # Use setitimer for float seconds support, alarm only supports integers
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.setitimer(signal.ITIMER_REAL, seconds)

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result = func(*args, **kwargs)
                # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
                finally:
                    # Reset timer and handler
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    signal.signal(signal.SIGALRM, old_handler)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return result

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return wrapper_signal
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # --- Multiprocessing based timeout (existing logic) ---
            # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
            @wraps(func)
            # [EXPLAIN] `wrapper_mp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def wrapper_mp(*args, **kwargs):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                q = multiprocessing.Queue(maxsize=1)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                process.start()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                process.join(timeout=seconds)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if process.is_alive():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    process.terminate()
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    process.join(timeout=0.5)  # Give it a moment to terminate
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if process.is_alive():
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Warning: Process {process.pid} did not terminate gracefully after timeout.")
                    # Update function name in error message if needed (optional but good practice)
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!")

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    success, result_or_exc = q.get(timeout=0.1)  # Small timeout for queue read
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if success:
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return result_or_exc
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise result_or_exc  # Reraise exception from child
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except queue.Empty as err:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    exitcode = process.exitcode
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if exitcode is not None and exitcode != 0:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise RuntimeError(f"Child process exited with error (exitcode: {exitcode}) before returning result.") from err
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # Should have timed out if queue is empty after join unless process died unexpectedly
                        # Update function name in error message if needed (optional but good practice)
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise TimeoutError(f"Operation timed out or process finished unexpectedly without result (exitcode: {exitcode}).") from err
                # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
                finally:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    q.close()
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    q.join_thread()

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return wrapper_mp

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return decorator


# [EXPLAIN] `union_two_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def union_two_dict(dict1: Dict, dict2: Dict):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in dict2.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key in dict1:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dict1[key] = val

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dict1


# [EXPLAIN] `append_to_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def append_to_dict(data: Dict, new_data: Dict):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in new_data.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key not in data:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data[key] = []
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        data[key].append(val)


# [EXPLAIN] `NestedNamespace` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class NestedNamespace(SimpleNamespace):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A nested version of SimpleNamespace that recursively converts dictionaries to namespaces.

    This class allows for dot notation access to nested dictionary structures by recursively
    converting dictionaries to NestedNamespace objects.

    Example:
        config_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = NestedNamespace(config_dict)
        # Access with: config.a, config.b.c, config.b.d

    Args:
        dictionary: The dictionary to convert to a nested namespace.
        **kwargs: Additional attributes to set on the namespace.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dictionary, **kwargs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(**kwargs)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, value in dictionary.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(value, dict):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.__setattr__(key, NestedNamespace(value))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.__setattr__(key, value)


# [EXPLAIN] `DynamicEnumMeta` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DynamicEnumMeta(type):
    # [EXPLAIN] `__iter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __iter__(cls) -> Iterator[Any]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return iter(cls._registry.values())

    # [EXPLAIN] `__contains__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __contains__(cls, item: Any) -> bool:
        # allow `name in EnumClass` or `member in EnumClass`
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(item, str):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return item in cls._registry
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return item in cls._registry.values()

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(cls, name: str) -> Any:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls._registry[name]

    # [EXPLAIN] `__reduce_ex__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __reduce_ex__(cls, protocol):
        # Always load the existing module and grab the class
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)

    # [EXPLAIN] `names` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def names(cls):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return list(cls._registry.keys())

    # [EXPLAIN] `values` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def values(cls):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return list(cls._registry.values())


# [EXPLAIN] `DynamicEnum` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DynamicEnum(metaclass=DynamicEnumMeta):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _registry: Dict[str, "DynamicEnum"] = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _next_value: int = 0

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, name: str, value: int):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name = name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.value = value

    # [EXPLAIN] `__repr__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __repr__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    # [EXPLAIN] `__reduce_ex__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __reduce_ex__(self, protocol):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Unpickle via: getattr(import_module(module).Dispatch, 'ONE_TO_ALL')
        so the existing class is reused instead of re-executed.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = importlib.import_module(self.__class__.__module__)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        enum_cls = getattr(module, self.__class__.__name__)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return getattr, (enum_cls, self.name)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `register` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def register(cls, name: str) -> "DynamicEnum":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key = name.upper()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key in cls._registry:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"{key} already registered")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        member = cls(key, cls._next_value)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cls._registry[key] = member
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        setattr(cls, key, member)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        cls._next_value += 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return member

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `remove` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def remove(cls, name: str):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key = name.upper()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        member = cls._registry.pop(key)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        delattr(cls, key)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return member

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `from_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return cls._registry.get(name.upper())

# [EXPLAIN] `convert_to_regular_types` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def convert_to_regular_types(obj):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Convert Hydra configs and other special types to regular Python types."""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from omegaconf import ListConfig, DictConfig
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(obj, (ListConfig, DictConfig)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(obj, (list, tuple)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [convert_to_regular_types(x) for x in obj]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(obj, dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return obj