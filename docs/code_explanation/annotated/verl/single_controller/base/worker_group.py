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
the class of WorkerGroup
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import signal
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Callable, Dict, List

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn


# [EXPLAIN] `ResourcePool` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ResourcePool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Manages a pool of resources across multiple nodes, tracking process counts and GPU allocations.
    The class provides methods to calculate world size, local world sizes, and local ranks
    across all nodes in the pool.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, process_on_nodes=None, max_colocate_count: int = 10, n_gpus_per_node=8) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize the ResourcePool with node processes and GPU configuration.

        Args:
            process_on_nodes (List[int], optional): List of process counts per node. Defaults to empty list.
            max_colocate_count (int, optional): Maximum number of processes that can be colocated. Defaults to 10.
            n_gpus_per_node (int, optional): Number of GPUs available per node. Defaults to 8.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if process_on_nodes is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            process_on_nodes = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._store = process_on_nodes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_colocate_count = max_colocate_count
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.n_gpus_per_node = n_gpus_per_node  # this is left for future huawei GPU that contains 16 GPUs per node

    # [EXPLAIN] `add_node` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def add_node(self, process_count):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._store.append(process_count)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Total number of processes across all nodes in the pool."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sum(self._store)

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self) -> Any:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._store

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `store` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def store(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._store

    # [EXPLAIN] `local_world_size_list` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def local_world_size_list(self) -> List[int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Returns a flat list where each process has its local world size."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nested_local_world_size_list = [[local_world_size for _ in range(local_world_size)] for local_world_size in self._store]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [item for row in nested_local_world_size_list for item in row]

    # [EXPLAIN] `local_rank_list` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def local_rank_list(self) -> List[int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Returns a flat list of local ranks for all processes across all nodes."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [item for row in nested_local_rank_list for item in row]


# [EXPLAIN] `ClassWithInitArgs` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ClassWithInitArgs:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Wrapper class that stores constructor arguments for deferred instantiation.
    This class is particularly useful for remote class instantiation where
    the actual construction needs to happen at a different time or location.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, cls, *args, **kwargs) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize the ClassWithInitArgs instance.

        Args:
            cls: The class to be instantiated later
            *args: Positional arguments for the class constructor
            **kwargs: Keyword arguments for the class constructor
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cls = cls
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.args = args
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.kwargs = kwargs

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.fused_worker_used = False

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self) -> Any:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Instantiate the stored class with the stored arguments."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.cls(*self.args, **self.kwargs)


# [EXPLAIN] `check_workers_alive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def check_workers_alive(workers: List, is_alive: Callable, gap_time: float = 1) -> None:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Continuously monitors worker processes and raises SIGABRT if any worker dies.

    Args:
        workers (List):
            List of worker objects to monitor
        is_alive (Callable):
            Function to check if a worker is alive
        gap_time (float):
            Time interval between checks
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import time

    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while True:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in workers:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not is_alive(worker):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logging.warning(f"worker {worker} is not alive sending signal to main thread")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                signal.raise_signal(signal.SIGABRT)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(gap_time)


# [EXPLAIN] `WorkerGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WorkerGroup:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Base class for managing a group of workers in a distributed system.
    The class provides methods for worker management, aliveness checking, and method binding.
    """

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fused_worker_execute_fn_name = "_fuw_execute"

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_init_with_detached_workers = resource_pool is None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.fused_worker_used = False

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if resource_pool is not None:
            # handle the case when WorkGroup is attached to an existing one
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._procecss_dispatch_config = resource_pool()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._procecss_dispatch_config = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._workers = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._worker_names = []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._master_addr = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._master_port = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._checker_thread: threading.Thread = None

    # [EXPLAIN] `_is_worker_alive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _is_worker_alive(self, worker):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Check if a worker is alive. Must be implemented by derived classes."""
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    # [EXPLAIN] `_block_until_all_workers_alive` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _block_until_all_workers_alive(self) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Blocks until all workers in the group are alive."""
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while True:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if False in all_state:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                time.sleep(1)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

    # [EXPLAIN] `start_worker_aliveness_check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Starts a background thread to monitor worker aliveness.

        Args:
            every_n_seconds (int): Interval between aliveness checks
        """
        # before starting checking worker aliveness, make sure all workers are already alive
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._block_until_all_workers_alive()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._checker_thread = threading.Thread(target=check_workers_alive, args=(self._workers, self._is_worker_alive, every_n_seconds))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._checker_thread.start()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def world_size(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Number of workers in the group."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self._workers)

    # [EXPLAIN] `_bind_worker_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _bind_worker_method(self, user_defined_cls, func_generator):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Binds worker methods to the WorkerGroup based on registered attributes.

        Args:
            user_defined_cls (type): The class containing methods to bind
            func_generator (Callable): Function that generates the bound method

        Returns:
            List[str]: List of method names that were successfully bound
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        method_names = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for method_name in dir(user_defined_cls):
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                method = getattr(user_defined_cls, method_name)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # if it is a property, it will fail because Class doesn't have instance property
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(method, MAGIC_ATTR):
                # this method is decorated by register
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attribute = getattr(method, MAGIC_ATTR)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert isinstance(attribute, Dict), f"attribute must be a dictionary. Got {type(attribute)}"
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dispatch_mode = attribute["dispatch_mode"]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                execute_mode = attribute["execute_mode"]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                blocking = attribute["blocking"]

                # get dispatch fn
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    dispatch_fn = fn["dispatch_fn"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    collect_fn = fn["collect_fn"]
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert isinstance(dispatch_mode, dict)
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert "dispatch_fn" in dispatch_mode
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert "collect_fn" in dispatch_mode
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    collect_fn = dispatch_mode["collect_fn"]

                # get execute_fn_name
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # get execute_fn from string
                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    execute_fn = getattr(self, wg_execute_fn_name)
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert callable(execute_fn), "execute_fn must be callable"
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise

                # bind a new method to the RayWorkerGroup
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )

                # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
                try:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(self, method_name, func)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    method_names.append(method_name)
                # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
                except Exception as e:
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return method_names
