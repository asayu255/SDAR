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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()
        return image

    # TextWorld attribute names for the game-file cycle, most likely first.
    _GAMEFILE_ITER_ATTRS = ('_gamefiles_iterator', 'gamefiles_iterator', '_gamefile_iterator')

    # resume時はTextWorld wrapper chainからseeded game-file iteratorを探索し、実際のgame loadをせずgeneratorだけを進める。1 manager resetが各workerで1 gameを消費するという不変条件に依存する。
    # 内部attributeがversion差で見つからない、またはiteratorが尽きた場合は部分進行を報告し、復旧を停止せず再現性低下を上位へ通知する。
    def _find_game_iterator(self):
        """Locate the object holding TextWorld's game-file cycle.

        ``textworld.gym.make()`` returns the batch env directly today, but it may
        be wrapped, so probe the env, its ``.unwrapped`` and the ``.env`` chain.
        Returns ``(holder, attr_name)`` or ``(None, None)``.
        """
        candidates = []
        current = self.env
        for _ in range(8):
            if current is None or any(current is seen for seen in candidates):
                break
            candidates.append(current)
            current = getattr(current, 'env', None)
        unwrapped = getattr(self.env, 'unwrapped', None)
        if unwrapped is not None and not any(unwrapped is seen for seen in candidates):
            candidates.append(unwrapped)

        for holder in candidates:
            for attr in self._GAMEFILE_ITER_ATTRS:
                if getattr(holder, attr, None) is not None:
                    return holder, attr
        return None, None

    def skip_games(self, num_resets: int) -> dict:
        """Advance the game-file cycle by ``num_resets`` episodes, loading nothing.

        Checkpoint resume support. ``TextworldBatchGymEnv.reset()`` pulls
        ``batch_size`` game files off a seeded, shuffled cycle and only then loads
        them; pulling from the generator alone is pure Python, while loading would
        cost minutes across all workers.

        Called with ``num_resets == 0`` this is a pure probe. Never raises: on any
        problem it reports ``supported: False`` and leaves the iterator untouched.
        """
        holder, attr = self._find_game_iterator()
        env_type = type(getattr(self.env, 'unwrapped', self.env)).__name__
        if holder is None:
            return {'supported': False, 'attr': None, 'skipped': 0,
                    'batch_size': None, 'env_type': env_type}

        batch_size = int(getattr(holder, 'batch_size', 1) or 1)
        iterator = getattr(holder, attr)
        skipped = 0
        try:
            for _ in range(max(0, int(num_resets)) * batch_size):
                next(iterator)
                skipped += 1
        except (StopIteration, TypeError) as exc:
            return {'supported': False, 'attr': attr, 'skipped': skipped,
                    'batch_size': batch_size, 'env_type': env_type, 'error': repr(exc)}
        return {'supported': True, 'attr': attr, 'skipped': skipped,
                'batch_size': batch_size, 'env_type': env_type}

# 各ALFWorld instanceをRay actorに隔離し、step/resetを全actorへ`.remote()`投入した後、単一`ray.get(futures)`でdriverへ集約する。これはenvironment IPCの待機でGPU collectiveではない。
# worker seedは`seed + group index`なので同じGRPO group内の複数trajectoryは同じ初期task scheduleを共有し、action差による結果を比較できる。admissible commandsは次turnの合法action制約として保持する。
class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def probe_game_iterator(self) -> dict:
        """Zero-cost check on one worker that :py:meth:`fast_forward` will work here."""
        return ray.get(self.workers[0].skip_games.remote(0))

    def fast_forward(self, num_resets: int) -> str:
        """Advance every worker's game-file cycle by ``num_resets`` episodes.

        Checkpoint resume support: each worker holds a batch_size-1 TextWorld env
        and :py:meth:`reset` resets all of them, so one manager reset consumes
        exactly one game per worker. Replaying the resets the restart skipped
        makes the next ``reset()`` hand out the games an uninterrupted run would.
        """
        if num_resets <= 0:
            return 'skipped 0 games per worker'

        results = ray.get([worker.skip_games.remote(num_resets) for worker in self.workers])

        def _complete(result):
            expected = int(num_resets) * int(result.get('batch_size') or 1)
            return bool(result.get('supported')) and result.get('skipped') == expected

        bad = [r for r in results if not _complete(r)]
        if bad:
            return (
                f'WARNING fast-forward INCOMPLETE - only {len(results) - len(bad)}/{len(results)} '
                f'workers advanced (sample={bad[0]}). TextWorld\'s game-file iterator was not '
                f'found or was exhausted, so the resumed run replays games from the start of the '
                f'cycle instead of continuing the uninterrupted sequence. Training itself is '
                f'unaffected; only which games are seen differs.'
            )
        return (
            f'skipped {int(num_resets) * int(results[0].get("batch_size") or 1)} games on each of '
            f'{len(results)} workers (attr={sorted({r["attr"] for r in results})})'
        )

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)