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
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import shutil
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tempfile
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from filelock import FileLock
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer, ProcessorMixin

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import is_cuda_available, is_npu_available


# [EXPLAIN] `BaseCheckpointManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseCheckpointManager:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if checkpoint_contents is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            checkpoint_contents = ["model", "optimizer", "extra"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.previous_global_step = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.previous_saved_paths = []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer = optimizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.lr_scheduler = lr_scheduler
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processing_class = processing_class
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.checkpoint_contents = checkpoint_contents

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank = torch.distributed.get_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.world_size = torch.distributed.get_world_size()

    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `checkpath` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def checkpath(local_path: str, hdfs_path: str):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return local_path is not None, local_path if local_path is not None else hdfs_path

    # [EXPLAIN] `remove_previous_save_local_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def remove_previous_save_local_path(self, path):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(path, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            path = [path]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for p in path:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            abs_path = os.path.abspath(p)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not os.path.exists(abs_path):
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            shutil.rmtree(abs_path, ignore_errors=True)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `local_mkdir` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def local_mkdir(path):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not os.path.isabs(path):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            working_dir = os.getcwd()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with FileLock(lock_path, timeout=60):  # Add timeout
                # make a new dir
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.makedirs(path, exist_ok=True)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            # Even if the lock is not acquired, try to create the directory
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.makedirs(path, exist_ok=True)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return path

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `get_rng_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_rng_state():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_cuda_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rng_state["cuda"] = torch.cuda.get_rng_state()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif is_npu_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rng_state["npu"] = torch.npu.get_rng_state()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rng_state

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `load_rng_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_rng_state(rng_state):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.set_rng_state(rng_state["cpu"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.set_state(rng_state["numpy"])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        random.setstate(rng_state["random"])

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_cuda_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_rng_state(rng_state["cuda"])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif is_npu_available:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.npu.set_rng_state(rng_state["npu"])


# [EXPLAIN] `find_latest_ckpt_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if path is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tracker_file = get_checkpoint_tracker_filename(path)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(tracker_file):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(tracker_file, "rb") as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        iteration = int(f.read().decode())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(ckpt_path):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Checkpoint does not exist: %s", ckpt_path)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Found checkpoint: %s", ckpt_path)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ckpt_path


# [EXPLAIN] `get_checkpoint_tracker_filename` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_checkpoint_tracker_filename(root_path: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")
