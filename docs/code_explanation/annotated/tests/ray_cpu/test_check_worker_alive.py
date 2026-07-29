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
import subprocess
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time


# [EXPLAIN] `test` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wait_time = 10

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    my_env = os.environ.copy()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    my_env["WAIT_TIME"] = str(wait_time)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    p = subprocess.Popen(["python3", "-u", "./check_worker_alive/main.py"], env=my_env, stdout=subprocess.PIPE)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    count = 0
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while b"foo started" not in p.stdout.read():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(1)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        count += 1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if count > 40:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("timeout for start foo in check_worker_alive/main.py")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(
        time.time(),
        f"wait 1.5 wait time {wait_time * 1.5} to let signal returned to process but still not exceed process wait time",
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(wait_time * 1.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(time.time(), "start checking")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert p.poll() is not None, f"process {p} still alive, expecting signal raised abort"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert p.returncode != 0, f"process {p} exit with code 0, expecting not-zero exit code"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("test passed")


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test()
