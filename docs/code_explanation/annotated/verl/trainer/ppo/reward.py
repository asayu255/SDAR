# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
import multiprocessing
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import partial

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score import default_compute_score


# [EXPLAIN] `get_custom_reward_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_custom_reward_fn(config):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import importlib.util
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import sys

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_fn_config = config.get("custom_reward_function") or {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    file_path = reward_fn_config.get("path")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not file_path:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(file_path):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module = importlib.util.module_from_spec(spec)
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        sys.modules["custom_module"] = module
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        spec.loader.exec_module(module)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    function_name = reward_fn_config.get("name")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not hasattr(module, function_name):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"using customized reward function '{function_name}' from '{file_path}'")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    raw_fn = getattr(module, function_name)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    # [EXPLAIN] `wrapped_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def wrapped_fn(*args, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return raw_fn(*args, **kwargs, **reward_kwargs)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return wrapped_fn


# [EXPLAIN] `load_reward_manager` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if reward_manager_name == "naive":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.reward_manager import NaiveRewardManager

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_manager_cls = NaiveRewardManager
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif reward_manager_name == "prime":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.reward_manager import PrimeRewardManager

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_manager_cls = PrimeRewardManager
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif reward_manager_name == "batch":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.reward_manager import BatchRewardManager

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_manager_cls = BatchRewardManager
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif reward_manager_name == "dapo":
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.workers.reward_manager import DAPORewardManager

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_manager_cls = DAPORewardManager
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    compute_score = get_custom_reward_fn(config)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_compute_score = compute_score

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if compute_score is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sandbox_config = config.reward_model.get("sandbox_fusion")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sandbox_url:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sandbox_manager = multiprocessing.Manager()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            final_compute_score = default_compute_score

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


# [EXPLAIN] `compute_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reward(data: DataProto, reward_fn):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_result = reward_fn(data, return_dict=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_tensor = reward_result["reward_tensor"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Error in reward_fn: {e}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_tensor = reward_fn(data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_extra_infos_dict = {}

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reward_tensor, reward_extra_infos_dict


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `compute_reward_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reward_async(data: DataProto, config, tokenizer):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return compute_reward(data, reward_fn)
