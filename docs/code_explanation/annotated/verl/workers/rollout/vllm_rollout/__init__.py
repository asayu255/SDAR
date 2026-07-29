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
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from importlib.metadata import PackageNotFoundError, version

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from packaging.version import Version


# [EXPLAIN] `get_version` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_version(pkg):
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return version(pkg)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except PackageNotFoundError:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
vllm_package_name = "vllm"
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
vllm_package_version = get_version(vllm_package_name)
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if vllm_package_version is None:
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise PackageNotFoundError("To use vllm rollout, please ensure the 'vllm' package is properly installed. See https://verl.readthedocs.io/en/latest/start/install.html for more details")

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if "ROCM_PATH" in os.environ:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import re

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if match:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vllm_package_version = match.group(1)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")
###

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if Version(vllm_package_version) <= Version("0.6.3"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vllm_mode = "customized"
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from .vllm_rollout import vLLMRollout  # noqa: F401
# [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
else:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vllm_mode = "spmd"
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
