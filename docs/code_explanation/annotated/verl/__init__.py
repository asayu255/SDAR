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
import importlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from importlib.metadata import PackageNotFoundError
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from importlib.metadata import version as get_version

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging.version import parse as parse_version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .protocol import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .utils.logging_utils import set_basic_config
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .utils.device import is_npu_available

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
with open(os.path.join(version_folder, "version/version")) as f:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    __version__ = f.read().strip()


# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
set_basic_config(level=logging.WARNING)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["DataProto", "__version__"]

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import importlib

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if importlib.util.find_spec("modelscope") is None:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from modelscope.utils.hf_util import patch_hub

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    patch_hub()

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if is_npu_available:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    package_name = 'transformers'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    required_version_spec = '4.51.0'
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        installed_version = get_version(package_name)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        installed = parse_version(installed_version)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        required = parse_version(required_version_spec)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not installed >= required:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except PackageNotFoundError as e:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}"
        ) from e
