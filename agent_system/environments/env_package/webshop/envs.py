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

import ray
import gym
import numpy as np

# Heap cap handed to each worker's embedded JVM (see ``WebshopWorker.__init__``).
# 512MB is ~50x WebShop's steady-state Lucene working set; raise it via
# ``SDAR_WEBSHOP_JVM_OPTIONS`` if a future index needs more.
WEBSHOP_JVM_OPTIONS = os.environ.get('SDAR_WEBSHOP_JVM_OPTIONS', '-Xmx512m -Xms64m -XX:+UseSerialGC')

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os

        # WebShop's search engine is pyserini's ``LuceneSearcher``, i.e. an
        # embedded JVM per worker. Left alone, HotSpot sizes its max heap at 1/4
        # of *physical* RAM -- 62GB a worker here -- so 120 workers grow their
        # heaps for tens of GB of anonymous memory before any of them bothers to
        # collect. WebShop's working set is a handful of MB, so cap it. Must be
        # set before ``web_agent_site.engine`` imports pyserini, which is what
        # starts the JVM. See docs/webshop_worker_memory.md.
        os.environ.setdefault('_JAVA_OPTIONS', WEBSHOP_JVM_OPTIONS)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)

    def mem_stats(self):
        """Anonymous vs file-backed RSS of this worker, in MiB.

        ``psutil.virtual_memory().used`` -- what ``perf/cpu_memory_used_gb`` and
        Ray's OOM killer watch -- excludes page cache, so only the *anonymous*
        half of a worker's RSS counts against the run. Splitting the two is what
        tells a real leak apart from the Lucene index being paged in.
        """
        import os

        stats = {}
        try:
            with open('/proc/self/smaps_rollup') as f:
                for line in f:
                    key, _, value = line.partition(':')
                    if key in ('Rss', 'Anonymous'):
                        stats[key.lower()] = int(value.split()[0]) / 1024
        except OSError:
            return {}
        stats['file_mib'] = stats.get('rss', 0.0) - stats.get('anonymous', 0.0)
        stats['rss_mib'] = stats.pop('rss', 0.0)
        stats['anon_mib'] = stats.pop('anonymous', 0.0)
        stats['pid'] = os.getpid()
        try:
            stats['sessions'] = len(self.env.server.user_sessions)
        except AttributeError:
            pass
        return stats

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))

        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def fast_forward(self, num_resets: int) -> str:
        """Replay ``num_resets`` goal draws without touching the workers.

        Checkpoint resume support: ``_rng`` is rebuilt from the seed on every
        start, so a resumed run would redraw the goals of step 1 onwards. One
        ``reset()`` consumes exactly one ``_rng.choice`` (see :py:meth:`reset` --
        the call below must stay identical to it), so replaying the draws the
        restart skipped makes the next ``reset()`` select the goals an
        uninterrupted run would have selected. The workers are untouched: only
        the RNG advances, which costs well under a millisecond per draw.
        """
        if num_resets <= 0:
            return 'replayed 0 goal draws'
        for _ in range(int(num_resets)):
            self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        return (
            f'replayed {int(num_resets)} goal draws '
            f'(env_num={self.env_num}, goal pool={len(self.goal_idxs)})'
        )

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )