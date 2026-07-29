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
import gymnasium as gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gym_cards.envs import Point24Env, EZPointEnv, BlackjackEnv, NumberLineEnv


# [EXPLAIN] `GymCardsWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class GymCardsWorker:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Ray remote actor that replaces the worker function.
    Each actor holds an independent instance of the specified gym environment.
    """
    
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, env_id):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Initialize the gym environment in this worker"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if env_id == 'gym_cards/Points24-v0':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.env = Point24Env()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif env_id == 'gym_cards/EZPoints-v0':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.env = EZPointEnv()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif env_id == 'gym_cards/Blackjack-v0':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.env = BlackjackEnv()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif env_id == 'gym_cards/NumberLine-v0':
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.env = NumberLineEnv()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError(f"Unknown env_id: {env_id}")
    
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a step in the environment"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, reward, done, _, info = self.env.step(action)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, reward, done, info
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, seed_for_reset=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reset the environment with optional seed"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if seed_for_reset is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs, info = self.env.reset(seed=seed_for_reset)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs, info = self.env.reset()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, info


# [EXPLAIN] `GymMultiProcessEnv` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class GymMultiProcessEnv(gym.Env):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Ray-based parallel environment wrapper for gym cards environments.
    - env_id: Gym environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self,
                 env_id,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # Initialize Ray if not already initialized
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not ray.is_initialized():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.init()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env_id = env_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.is_train = is_train
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.group_n = group_n
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env_num = env_num
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_processes = env_num * group_n

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        np.random.seed(seed)
    
        # Create Ray remote actors instead of processes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_worker = ray.remote(**resources_per_worker)(GymCardsWorker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _ in range(self.num_processes):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            worker = env_worker.remote(self.env_id)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.workers.append(worker)

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, actions):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Perform step in parallel.
        :param actions: list or numpy array, length must equal self.num_processes.
        :return: obs_list, reward_list, done_list, info_list
        """
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker, action in zip(self.workers, actions):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.step.remote(action)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # Collect results
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_list, reward_list, done_list, info_list = [], [], [], []
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
        
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(obs_list[0], np.ndarray):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs_list = np.array(obs_list)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs_list, reward_list, done_list, info_list

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.is_train:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Repeat seed for environments in the same group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seeds = np.repeat(seeds, self.group_n)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seeds = seeds.tolist()

        # Send reset commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, worker in enumerate(self.workers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.reset.remote(seeds[i])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # Collect results
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_list, info_list = [], []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for obs, info in results:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            obs_list.append(obs)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info_list.append(info)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(obs_list[0], np.ndarray):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs_list = np.array(obs_list)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs_list, info_list

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Close all Ray actors.
        """
        # Kill all Ray actors
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.kill(worker)

    # [EXPLAIN] `__del__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __del__(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.close()


# [EXPLAIN] `build_gymcards_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_gymcards_envs(env_name,
                        seed,
                        env_num,
                        group_n,
                        resources_per_worker,
                        is_train=True):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Externally exposed constructor function to create parallel Gym environments.
    - env_name: [gym_cards/Blackjack-v0, gym_cards/NumberLine-v0, gym_cards/EZPoints-v0, gym_cards/Points24-v0]
    - seed: For reproducible randomness
    - env_num: Number of distinct environments
    - group_n: Number of environment replicas under the same seed
    - is_train: Determines the seed range used (train/test)
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return GymMultiProcessEnv(
        env_id=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
    )