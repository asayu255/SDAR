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
Trajectory tracker can be inserted into code to save the intermediate results.
The results will be dump to hdfs for offline comparison.
Each process will have a client that first move all the tensors to CPU
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import io
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tempfile
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import deque

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.hdfs_io import copy, makedirs

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
remote_copy = ray.remote(copy)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `save_to_hdfs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def save_to_hdfs(data: io.BytesIO, name, hdfs_dir, verbose):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    filename = name + ".pth"
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with tempfile.TemporaryDirectory() as tmpdirname:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_filepath = os.path.join(tmpdirname, filename)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(local_filepath, "wb") as f:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            f.write(data.getbuffer())
        # upload to hdfs

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if verbose:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saving {local_filepath} to {hdfs_dir}")
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            copy(local_filepath, hdfs_dir)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(e)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `TrajectoryTracker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TrajectoryTracker:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, hdfs_dir, verbose) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hdfs_dir = hdfs_dir
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        makedirs(hdfs_dir)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.verbose = verbose

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.handle = deque()

    # [EXPLAIN] `dump` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dump(self, data: io.BytesIO, name):
        # get a temp file and write to it
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.handle.append(save_to_hdfs.remote(data, name, self.hdfs_dir, self.verbose))

    # [EXPLAIN] `wait_for_hdfs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def wait_for_hdfs(self):
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while len(self.handle) != 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = self.handle.popleft()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.get(future)


# [EXPLAIN] `dump_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dump_data(data, name):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    enable = os.getenv("VERL_ENABLE_TRACKER", "0") == "1"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not enable:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    buffer = io.BytesIO()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.save(data, buffer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tracker = get_trajectory_tracker()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(tracker.dump.remote(buffer, name))


# [EXPLAIN] `get_trajectory_tracker` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_trajectory_tracker():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_dir = os.getenv("VERL_TRACKER_HDFS_DIR", default=None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    verbose = os.getenv("VERL_TRACKER_VERBOSE", default="0") == "1"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert hdfs_dir is not None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(hdfs_dir, verbose)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tracker


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # testing
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["VERL_ENABLE_TRACKER"] = "1"
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["VERL_TRACKER_HDFS_DIR"] = "~/debug/test"

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @ray.remote
    # [EXPLAIN] `process` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def process(iter):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = {"obs": torch.randn(10, 20)}
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dump_data(data, f"process_{iter}_obs")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_lst = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(10):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output_lst.append(process.remote(i))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = ray.get(output_lst)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tracker = get_trajectory_tracker()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(tracker.wait_for_hdfs.remote())
