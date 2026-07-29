# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from appworld import AppWorld, load_task_ids

# [EXPLAIN] `load_available_ports` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_available_ports(port_file="appworld_ports.ports"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Load available port list from file
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(port_file):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise FileNotFoundError(f"Port file {port_file} does not exist. Please run the service startup script first.")
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ports = []
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(port_file, 'r') as f:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for line in f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            line = line.strip()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if line and line.isdigit():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                ports.append(int(line))
    
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not ports:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError(f"No valid ports found in port file {port_file}.")
    
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ports

# [EXPLAIN] `AppWorldWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AppWorldWorker:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.
    """
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, worker_id, max_interactions, port):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_step_count = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_interactions = max_interactions
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.worker_id = worker_id
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.url = f"http://0.0.0.0:{port}"

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, task_id):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reset the environment with a new task."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.env is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.env.close()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(2)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.current_step_count = 0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env = AppWorld(
            task_id=task_id,
            experiment_name=f'default_{self.worker_id}',
            remote_environment_url=self.url,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self.env.task.instruction
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
        }
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, info

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute one step in the environment."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.env is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.current_step_count += 1

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = self.env.execute(action)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if done:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_success = self.env.evaluate().success

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 10.0 if is_success else 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {"won": is_success, "step_count": self.current_step_count}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = 0.0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = {"won": False, "step_count": self.current_step_count}

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, reward, done, info

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Close the environment."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.env is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.env.close()


# [EXPLAIN] `AppWorldEnvs` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AppWorldEnvs:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, 
                 dataset_name,
                 max_interactions,
                 seed,
                 env_num,
                 group_n,
                 start_server_id,
                 resources_per_worker,
                 port_file="appworld_ports.ports"
                 ):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dataset_name = dataset_name
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_interactions = max_interactions
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env_num = env_num
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.group_n = group_n
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_processes = env_num * group_n
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_ids = load_task_ids(dataset_name)
   
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.env_num > len(self.task_ids):
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_name}' ({len(self.task_ids)}). Please reducing env_num to {len(self.task_ids)}.")
            
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_ports = load_available_ports(port_file)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.available_ports = all_ports[start_server_id:start_server_id + self.num_processes]

        # Check if we have enough ports
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(self.available_ports) < self.num_processes:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(
                f"Need {self.num_processes} ports, but only {len(self.available_ports)} available ports. "
                f"Please ensure enough service instances are started."
            )

        # Initialize Ray if not already initialized
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not ray.is_initialized():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.init()

        # Create Ray actors (workers)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_worker = ray.remote(**resources_per_worker)(AppWorldWorker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(self.num_processes):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            port = self.available_ports[i]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            worker = env_worker.remote(
                worker_id=start_server_id + i,
                max_interactions=self.max_interactions,
                port=port
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.workers.append(worker)

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, actions):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        actions: Must be a list with length equal to self.num_processes, 
        each sent to the corresponding worker.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        # Send step commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, worker in enumerate(self.workers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.step.remote(actions[i])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # Collect results
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        done_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info_list = []

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs, reward, done, info in results:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            obs_list.append(obs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_list.append(reward)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            done_list.append(done)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info_list.append(info)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs_list, reward_list, done_list, info_list

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Reset all worker environments simultaneously, 
        returning each environment's initial observation and info.
        """
        # randomly select self.env_num task_id from self.task_ids
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_id = np.repeat(task_id, self.group_n).tolist()

        # Send reset commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, worker in enumerate(self.workers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.reset.remote(task_id[i])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # Collect results
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info_list = []

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs, info in results:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            obs_list.append(obs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info_list.append(info)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs_list, info_list

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Close all workers."""
        # Send close commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.close.remote()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)
        
        # Wait for all workers to close
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.get(futures)
        
        # Shutdown Ray actors
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.kill(worker)

    # [EXPLAIN] `render` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def render(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Implement this if visualization is needed."""
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

# [EXPLAIN] `build_appworld_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_appworld_envs(dataset_name="train",
                        max_interactions=50,
                        seed=0,
                        env_num=1, 
                        group_n=1,
                        start_server_id=0,
                        resources_per_worker={"num_cpus": 0.1},
                        ):

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        resources_per_worker=resources_per_worker
    )