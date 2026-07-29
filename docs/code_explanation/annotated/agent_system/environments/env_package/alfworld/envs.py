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
import yaml
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gymnasium as gym
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from gymnasium import spaces
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torchvision.transforms as T
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

# [EXPLAIN] `load_config_file` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_config_file(path):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert os.path.exists(path), "Invalid config file"
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(path) as reader:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = yaml.safe_load(reader)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return config

# [EXPLAIN] `get_obs_image` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_obs_image(env):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    transform = T.Compose([T.ToTensor()])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    current_frames = env.get_frames()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_tensors = [transform(i).cuda() for i in current_frames]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(len(image_tensors)):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        image_tensors[i]*= 255
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_tensors[i] = image_tensors[i].int()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    image_tensors = torch.stack(image_tensors, dim=0)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return image_tensors

# [EXPLAIN] `compute_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reward(info, multi_modal=False):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if multi_modal:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward = 10.0 * float(info['won'])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reward

# [EXPLAIN] `AlfworldWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AlfworldWorker:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, seed, base_env):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.env.seed(seed)
    
    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, action):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Execute a step in the environment"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions = [action] 
        
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, scores, dones, infos = self.env.step(actions)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        infos['observation_text'] = obs
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, scores, dones, infos
    
    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Reset the environment"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs, infos = self.env.reset()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        infos['observation_text'] = obs
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return obs, infos
    
    # [EXPLAIN] `getobs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def getobs(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get current observation image"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = get_obs_image(self.env)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image = image.cpu()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return image

    # TextWorld attribute names for the game-file cycle, most likely first.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _GAMEFILE_ITER_ATTRS = ('_gamefiles_iterator', 'gamefiles_iterator', '_gamefile_iterator')

    # [EXPLAIN] `_find_game_iterator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _find_game_iterator(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Locate the object holding TextWorld's game-file cycle.

        ``textworld.gym.make()`` returns the batch env directly today, but it may
        be wrapped, so probe the env, its ``.unwrapped`` and the ``.env`` chain.
        Returns ``(holder, attr_name)`` or ``(None, None)``.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        candidates = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current = self.env
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _ in range(8):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if current is None or any(current is seen for seen in candidates):
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            candidates.append(current)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            current = getattr(current, 'env', None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        unwrapped = getattr(self.env, 'unwrapped', None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if unwrapped is not None and not any(unwrapped is seen for seen in candidates):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            candidates.append(unwrapped)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for holder in candidates:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for attr in self._GAMEFILE_ITER_ATTRS:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if getattr(holder, attr, None) is not None:
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return holder, attr
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None, None

    # [EXPLAIN] `skip_games` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def skip_games(self, num_resets: int) -> dict:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Advance the game-file cycle by ``num_resets`` episodes, loading nothing.

        Checkpoint resume support. ``TextworldBatchGymEnv.reset()`` pulls
        ``batch_size`` game files off a seeded, shuffled cycle and only then loads
        them; pulling from the generator alone is pure Python, while loading would
        cost minutes across all workers.

        Called with ``num_resets == 0`` this is a pure probe. Never raises: on any
        problem it reports ``supported: False`` and leaves the iterator untouched.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        holder, attr = self._find_game_iterator()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_type = type(getattr(self.env, 'unwrapped', self.env)).__name__
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if holder is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {'supported': False, 'attr': None, 'skipped': 0,
                    'batch_size': None, 'env_type': env_type}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = int(getattr(holder, 'batch_size', 1) or 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        iterator = getattr(holder, attr)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        skipped = 0
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _ in range(max(0, int(num_resets)) * batch_size):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                next(iterator)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                skipped += 1
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except (StopIteration, TypeError) as exc:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {'supported': False, 'attr': attr, 'skipped': skipped,
                    'batch_size': batch_size, 'env_type': env_type, 'error': repr(exc)}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {'supported': True, 'attr': attr, 'skipped': skipped,
                'batch_size': batch_size, 'env_type': env_type}

# [EXPLAIN] `AlfworldEnvs` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AlfworldEnvs(gym.Env):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        
        # Initialize Ray if not already initialized
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not ray.is_initialized():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.init()
            
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = load_config_file(alf_config_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_type = config['env']['type']
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.multi_modal = (env_type == 'AlfredThorEnv')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_processes = env_num * group_n
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.workers = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(self.num_processes):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.workers.append(worker)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self, actions):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

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
        text_obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dones_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info_list = []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, (obs, scores, dones, info) in enumerate(results):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k in info.keys():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                info[k] = info[k][0]

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            text_obs_list.append(obs[0])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dones_list.append(dones[0])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info_list.append(info)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.prev_admissible_commands[i] = info['admissible_commands']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            rewards_list.append(compute_reward(info, self.multi_modal))

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.multi_modal:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_obs_list = self.getobs()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_obs_list = None

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        image_obs_list = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info_list = []

        # Send reset commands to all workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.reset.remote()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # Collect results
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get(futures)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, (obs, info) in enumerate(results):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k in info.keys():
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                info[k] = info[k][0] 
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            text_obs_list.append(obs[0])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.prev_admissible_commands[i] = info['admissible_commands']
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            info_list.append(info)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.multi_modal:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_obs_list = self.getobs()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            image_obs_list = None

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return text_obs_list, image_obs_list, info_list

    # [EXPLAIN] `probe_game_iterator` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def probe_game_iterator(self) -> dict:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Zero-cost check on one worker that :py:meth:`fast_forward` will work here."""
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ray.get(self.workers[0].skip_games.remote(0))

    # [EXPLAIN] `fast_forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fast_forward(self, num_resets: int) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Advance every worker's game-file cycle by ``num_resets`` episodes.

        Checkpoint resume support: each worker holds a batch_size-1 TextWorld env
        and :py:meth:`reset` resets all of them, so one manager reset consumes
        exactly one game per worker. Replaying the resets the restart skipped
        makes the next ``reset()`` hand out the games an uninterrupted run would.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if num_resets <= 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 'skipped 0 games per worker'

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        results = ray.get([worker.skip_games.remote(num_resets) for worker in self.workers])

        # [EXPLAIN] `_complete` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def _complete(result):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            expected = int(num_resets) * int(result.get('batch_size') or 1)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return bool(result.get('supported')) and result.get('skipped') == expected

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bad = [r for r in results if not _complete(r)]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if bad:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (
                f'WARNING fast-forward INCOMPLETE - only {len(results) - len(bad)}/{len(results)} '
                f'workers advanced (sample={bad[0]}). TextWorld\'s game-file iterator was not '
                f'found or was exhausted, so the resumed run replays games from the start of the '
                f'cycle instead of continuing the uninterrupted sequence. Training itself is '
                f'unaffected; only which games are seen differs.'
            )
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            f'skipped {int(num_resets) * int(results[0].get("batch_size") or 1)} games on each of '
            f'{len(results)} workers (attr={sorted({r["attr"] for r in results})})'
        )

    # [EXPLAIN] `getobs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def getobs(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        futures = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            future = worker.getobs.remote()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            futures.append(future)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        images = ray.get(futures)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return images

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `get_admissible_commands` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_admissible_commands(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.prev_admissible_commands

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Close all workers
        """
        # Kill all Ray actors
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for worker in self.workers:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ray.kill(worker)

# [EXPLAIN] `build_alfworld_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)