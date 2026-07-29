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
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


# [EXPLAIN] `get_ppo_ray_runtime_env` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_ppo_ray_runtime_env():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in list(runtime_env["env_vars"].keys()):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.environ.get(key) is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            runtime_env["env_vars"].pop(key, None)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return runtime_env
