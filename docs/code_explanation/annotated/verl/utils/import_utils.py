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
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib.util
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import cache, wraps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@cache
# [EXPLAIN] `is_megatron_core_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_megatron_core_available():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mcore_spec = importlib.util.find_spec("megatron.core")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ModuleNotFoundError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mcore_spec = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mcore_spec is not None


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@cache
# [EXPLAIN] `is_vllm_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_vllm_available():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_spec = importlib.util.find_spec("vllm")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ModuleNotFoundError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_spec = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return vllm_spec is not None


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@cache
# [EXPLAIN] `is_sglang_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_sglang_available():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sglang_spec = importlib.util.find_spec("sglang")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ModuleNotFoundError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sglang_spec = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return sglang_spec is not None


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@cache
# [EXPLAIN] `is_nvtx_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_nvtx_available():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nvtx_spec = importlib.util.find_spec("nvtx")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ModuleNotFoundError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nvtx_spec = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return nvtx_spec is not None


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@cache
# [EXPLAIN] `is_trl_available` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_trl_available():
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trl_spec = importlib.util.find_spec("trl")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except ModuleNotFoundError:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        trl_spec = None
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return trl_spec is not None


# [EXPLAIN] `import_external_libs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def import_external_libs(external_libs=None):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if external_libs is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not isinstance(external_libs, list):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        external_libs = [external_libs]
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import importlib

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for external_lib in external_libs:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        importlib.import_module(external_lib)


# [EXPLAIN] `load_extern_type` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_extern_type(file_path: Optional[str], type_name: Optional[str]) -> type:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load a external data type based on the file path and type name"""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not file_path:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if file_path.startswith("pkg://"):
        # pkg://verl.utils.dataset.rl_dataset
        # pkg://verl/utils/dataset/rl_dataset
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module_name = file_path[6:].replace("/", ".")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = importlib.import_module(module_name)

    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # file://verl/utils/dataset/rl_dataset
        # file:///path/to/verl/utils/dataset/rl_dataset.py
        # or without file:// prefix
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if file_path.startswith("file://"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            file_path = file_path[7:]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not os.path.exists(file_path):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise FileNotFoundError(f"Custom type file '{file_path}' not found.")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        module = importlib.util.module_from_spec(spec)
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            spec.loader.exec_module(module)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError(f"Error loading module from '{file_path}'") from e

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not hasattr(module, type_name):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AttributeError(f"Custom type '{type_name}' not found in '{file_path}'.")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return getattr(module, type_name)


# [EXPLAIN] `_get_qualified_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _get_qualified_name(func):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Get full qualified name including module and class (if any)."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module = func.__module__
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    qualname = func.__qualname__
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return f"{module}.{qualname}"


# [EXPLAIN] `deprecated` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def deprecated(replacement: str = ""):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Decorator to mark functions or classes as deprecated."""

    # [EXPLAIN] `decorator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decorator(obj):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        qualified_name = _get_qualified_name(obj)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(obj, type):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            original_init = obj.__init__

            # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
            @wraps(original_init)
            # [EXPLAIN] `wrapped_init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def wrapped_init(self, *args, **kwargs):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                msg = f"Warning: Class '{qualified_name}' is deprecated."
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if replacement:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    msg += f" Please use '{replacement}' instead."
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return original_init(self, *args, **kwargs)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obj.__init__ = wrapped_init
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return obj

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:

            # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
            @wraps(obj)
            # [EXPLAIN] `wrapped` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
            def wrapped(*args, **kwargs):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                msg = f"Warning: Function '{qualified_name}' is deprecated."
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if replacement:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    msg += f" Please use '{replacement}' instead."
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return obj(*args, **kwargs)

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return wrapped

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return decorator
