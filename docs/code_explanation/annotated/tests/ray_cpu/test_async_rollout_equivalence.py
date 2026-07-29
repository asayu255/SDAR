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
"""CPU-only accuracy gate for the async rollout core.

Proves that the async per-sequence schedule
(`async_rollout_core.collect_async`) collects *exactly* the same trajectories as
the synchronous turn-based loop semantics of
`TrajectoryCollector.vanilla_multi_turn_loop`, for a deterministic policy+env.

Since SDAR training recomputes all log-probs on the sampled tokens, "same
trajectories given the same policy/env" is precisely the condition for "no
accuracy impact". This test needs no torch/numpy/GPU, so it can gate every
change to the async scheduler. Run:  python tests/ray_cpu/test_async_rollout_equivalence.py
"""

import asyncio
import importlib.util
import os
import sys

# Load async_rollout_core directly by path so this test stays torch-free (the
# package __init__ imports rollout_loop which needs torch; the core does not).
_CORE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "agent_system",
        "multi_turn_rollout",
        "async_rollout_core.py",
    )
)
_spec = importlib.util.spec_from_file_location("async_rollout_core", _CORE_PATH)
_core = importlib.util.module_from_spec(_spec)
sys.modules["async_rollout_core"] = _core  # needed for dataclass annotation resolution
_spec.loader.exec_module(_core)
collect_async = _core.collect_async


# --------------------------------------------------------------------------- #
# Deterministic mock policy + env (no cross-trajectory state).
# --------------------------------------------------------------------------- #
def mock_policy(traj_id, step, obs):
    # Action depends only on this trajectory's (id, step, obs) -> deterministic.
    return f"act[{traj_id}|{step}|{obs}]"


def make_mock_env(done_at):
    """done_at[i] = number of steps trajectory i runs before env returns done."""

    def env_step_sync(traj_id, step, action):
        done = (step + 1) >= done_at[traj_id]
        reward = round(0.1 * traj_id + step, 3)  # deterministic per (i, step)
        next_obs = f"o[{traj_id}|{step + 1}]"
        info = {"valid": (step % 2 == 0)}
        return next_obs, reward, done, info

    return env_step_sync


# --------------------------------------------------------------------------- #
# Reference implementation of the SYNC loop's data semantics
# (mirrors TrajectoryCollector.vanilla_multi_turn_loop).
# --------------------------------------------------------------------------- #
def sync_reference(n_traj, max_steps, initial_obs, policy, env_step_sync, rollout_n):
    done = [False] * n_traj
    obs = list(initial_obs)
    steps = [[] for _ in range(n_traj)]
    ep_reward = [0.0] * n_traj
    ep_length = [0] * n_traj

    for turn in range(max_steps):
        active = [i for i in range(n_traj) if not done[i]]
        if not active:
            break
        actions = {i: policy(i, turn, obs[i]) for i in active}  # generate active only
        for i in active:
            next_obs, reward, d, info = env_step_sync(i, turn, actions[i])
            # turn_step (gather) == position in the per-traj list == turn index,
            # because an active trajectory is appended every turn from 0 until done.
            steps[i].append((turn, actions[i], reward, bool(d), info))
            ep_reward[i] += float(reward)
            ep_length[i] += 1
            obs[i] = next_obs
            done[i] = done[i] or bool(d)
        if all(done):
            break

    # GRPO grouping partition (which trajectories share a uid).
    groups = {}
    if rollout_n and rollout_n > 0:
        for i in range(n_traj):
            groups.setdefault(i // rollout_n, []).append(i)
    else:
        groups[0] = list(range(n_traj))
    partition = sorted(sorted(v) for v in groups.values())
    return steps, ep_reward, ep_length, partition


def async_to_comparable(result):
    n = len(result.trajectories)
    steps = [[] for _ in range(n)]
    ep_reward = [0.0] * n
    ep_length = [0] * n
    for tr in result.trajectories:
        for s in tr.steps:
            steps[tr.traj_id].append((s.step, s.action, s.reward, s.done, s.info))
        ep_reward[tr.traj_id] = tr.episode_reward
        ep_length[tr.traj_id] = tr.episode_length
    # uid grouping partition
    uid_to_ids = {}
    for tr in result.trajectories:
        uid_to_ids.setdefault(tr.uid, []).append(tr.traj_id)
    partition = sorted(sorted(v) for v in uid_to_ids.values())
    traj_uids = [tr.traj_uid for tr in result.trajectories]
    return steps, ep_reward, ep_length, partition, traj_uids


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _run_case(n_traj, max_steps, rollout_n, done_at, slow_set=None, max_in_flight=None):
    initial_obs = [f"o[{i}|0]" for i in range(n_traj)]
    env_step_sync = make_mock_env(done_at)

    # deterministic uid/traj_uid factories (independent counters; we compare
    # grouping *structure*, not raw ids, so call order doesn't matter)
    uc = {"n": 0}
    tc = {"n": 0}

    def uid_factory():
        uc["n"] += 1
        return f"uid{uc['n']}"

    def traj_uid_factory():
        tc["n"] += 1
        return f"traj{tc['n']}"

    async def gen_fn(i, step, obs):
        await asyncio.sleep(0)  # real AsyncLLM generation yields to the loop
        return mock_policy(i, step, obs)

    async def env_fn(i, step, action):
        if slow_set and i in slow_set:
            await asyncio.sleep(0.01)  # emulate slow env (e.g. search HTTP)
        return env_step_sync(i, step, action)

    result = asyncio.run(
        collect_async(
            n_traj=n_traj,
            max_steps=max_steps,
            initial_obs=initial_obs,
            generate_action=gen_fn,
            env_step=env_fn,
            rollout_n=rollout_n,
            uid_factory=uid_factory,
            traj_uid_factory=traj_uid_factory,
            max_in_flight=max_in_flight,
        )
    )

    a_steps, a_r, a_l, a_part, a_traj_uids = async_to_comparable(result)
    s_steps, s_r, s_l, s_part = sync_reference(
        n_traj, max_steps, initial_obs, mock_policy, env_step_sync, rollout_n
    )

    assert a_steps == s_steps, "per-step records differ between async and sync!"
    assert a_r == s_r, "episode rewards differ!"
    assert a_l == s_l, "episode lengths differ!"
    assert a_part == s_part, f"GRPO uid grouping differs! async={a_part} sync={s_part}"
    assert len(set(a_traj_uids)) == n_traj, "traj_uids must be unique"
    return result, s_l


def test_equivalence_mixed_horizons():
    # Emulate the real task mix: search finishes ~4, webshop ~15, alfworld ~50.
    n = 24
    rollout_n = 8
    done_at = []
    for i in range(n):
        task = (i // rollout_n) % 3
        done_at.append({0: 4, 1: 15, 2: 50}[task])
    _run_case(n_traj=n, max_steps=50, rollout_n=rollout_n, done_at=done_at)
    print("PASS: mixed-horizon trajectories identical (sync == async)")


def test_equivalence_capped_at_max_steps():
    # Trajectories that never finish must be capped at max_steps in both.
    n = 6
    done_at = [100] * n  # never finishes within max_steps
    _, lengths = _run_case(n_traj=n, max_steps=12, rollout_n=2, done_at=done_at)
    assert all(l == 12 for l in lengths), lengths
    print("PASS: never-done trajectories capped at max_steps identically")


def test_equivalence_all_finish_immediately():
    n = 5
    done_at = [1] * n  # done after a single step
    _, lengths = _run_case(n_traj=n, max_steps=50, rollout_n=1, done_at=done_at)
    assert all(l == 1 for l in lengths), lengths
    print("PASS: single-step trajectories identical")


def test_no_grouping():
    # rollout_n=0 -> all share one uid; trajectories still identical.
    n = 7
    done_at = [3, 5, 2, 8, 1, 4, 6]
    _run_case(n_traj=n, max_steps=10, rollout_n=0, done_at=done_at)
    print("PASS: ungrouped rollout identical")


def test_continuous_batching_overlap():
    # With slow env steps on some trajectories, others must keep generating
    # concurrently (continuous batching) -> peak concurrency > 1, and results
    # still identical to sync.
    n = 12
    done_at = [5] * n
    slow = {0, 1, 2}  # these have slow env (await sleep)
    result, _ = _run_case(
        n_traj=n, max_steps=5, rollout_n=4, done_at=done_at, slow_set=slow
    )
    assert result.max_concurrent_generate > 1, (
        f"expected overlap, peak={result.max_concurrent_generate}"
    )
    print(f"PASS: continuous batching overlaps (peak concurrent generate={result.max_concurrent_generate})")


def test_max_in_flight_cap_preserves_data():
    # Capping in-flight generations must not change the collected data.
    n = 10
    done_at = [i % 7 + 1 for i in range(n)]
    _run_case(n_traj=n, max_steps=20, rollout_n=2, done_at=done_at, max_in_flight=3)
    print("PASS: max_in_flight cap preserves trajectories")


if __name__ == "__main__":
    test_equivalence_mixed_horizons()
    test_equivalence_capped_at_max_steps()
    test_equivalence_all_finish_immediately()
    test_no_grouping()
    test_continuous_batching_overlap()
    test_max_in_flight_cap_preserves_data()
    print("\nALL ASYNC-ROLLOUT EQUIVALENCE TESTS PASSED")
