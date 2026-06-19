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
"""Engine/-env-agnostic async rollout orchestration core.

This module contains ONLY the control flow that decides, per trajectory, the
sequence of (generate action -> step env) interactions and how the resulting
per-step records are laid out. It has no torch / numpy / vLLM dependencies so it
can be unit-tested on CPU to *prove* that the async schedule collects exactly the
same trajectories as the synchronous turn-based loop
(`TrajectoryCollector.vanilla_multi_turn_loop`).

Why this is the accuracy-critical piece
---------------------------------------
In SDAR/verl-agent the rollout engine only chooses *which actions are sampled*;
all training log-probs (old/teacher/ref) are recomputed downstream on the
sampled tokens. So "async must not change accuracy" reduces to "async must
collect the same trajectories given the same policy and envs". Each trajectory's
interaction depends only on its own (obs, action) history, never on how other
trajectories are batched/timed — therefore a per-sequence async schedule yields
identical trajectories to the lock-step loop for any deterministic policy+env.
This module makes that property explicit and testable.

The real integration (`async_rollout_loop.py`, added next) plugs in:
  * generate_action  -> vLLM AsyncLLM per-request generation (continuous batching)
  * env_step         -> thread-pooled per-trajectory env step
and reuses the records produced here to build the exact same DataProto that
`gather_rollout_data` already consumes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Type aliases for the two pluggable async operations.
# generate_action(traj_id, step, obs) -> action
GenerateFn = Callable[[int, int, Any], Awaitable[Any]]
# env_step(traj_id, step, action) -> (next_obs, reward, done, info)
EnvStepFn = Callable[[int, int, Any], Awaitable[tuple]]


@dataclass
class StepRecord:
    """One (action, env-response) interaction of a single trajectory.

    Mirrors exactly the per-step information the synchronous loop appends to
    total_batch_list[i] / total_infos[i] for an *active* step. The async loop
    never produces records for finished trajectories (the sync loop produces
    them only to immediately discard them via active_masks=False), so the
    downstream effective batch is identical.
    """

    traj_id: int
    step: int
    obs: Any
    action: Any
    reward: float
    done: bool
    info: Any = None


@dataclass
class TrajResult:
    traj_id: int
    uid: Any
    traj_uid: Any
    steps: List[StepRecord] = field(default_factory=list)
    episode_reward: float = 0.0
    episode_length: int = 0


@dataclass
class RolloutResult:
    trajectories: List[TrajResult]
    # Diagnostics that let tests assert the schedule actually overlaps work.
    max_concurrent_generate: int = 0


def assign_uids(n_traj: int, rollout_n: int, uid_factory: Callable[[], Any]) -> List[Any]:
    """Group trajectories into GRPO groups exactly like vanilla_multi_turn_loop:
    consecutive blocks of ``rollout_n`` share a uid. rollout_n <= 0 means a
    single group. This is order-independent for downstream advantage grouping,
    but we keep the identical layout to avoid any divergence.
    """
    uids: List[Any] = []
    if rollout_n and rollout_n > 0:
        uid = None
        for i in range(n_traj):
            if i % rollout_n == 0:
                uid = uid_factory()
            uids.append(uid)
    else:
        uid = uid_factory()
        uids = [uid for _ in range(n_traj)]
    return uids


async def collect_async(
    n_traj: int,
    max_steps: int,
    initial_obs: List[Any],
    generate_action: GenerateFn,
    env_step: EnvStepFn,
    rollout_n: int = 0,
    uid_factory: Optional[Callable[[], Any]] = None,
    traj_uid_factory: Optional[Callable[[], Any]] = None,
    max_in_flight: Optional[int] = None,
) -> RolloutResult:
    """Drive ``n_traj`` trajectories concurrently to completion.

    Each trajectory independently repeats ``generate_action`` -> ``env_step``
    until the env reports ``done`` or it has taken ``max_steps`` steps — the same
    termination rule as the synchronous loop (which breaks a trajectory once
    ``is_done`` or the shared ``for _step in range(max_steps)`` is exhausted).

    Args mirror the sync loop's state. ``max_in_flight`` optionally caps the
    number of concurrently *generating* trajectories (does not change collected
    data, only scheduling pressure on the engine).
    """
    if uid_factory is None:
        import uuid

        uid_factory = lambda: str(uuid.uuid4())  # noqa: E731
    if traj_uid_factory is None:
        import uuid

        traj_uid_factory = lambda: str(uuid.uuid4())  # noqa: E731

    uids = assign_uids(n_traj, rollout_n, uid_factory)
    results = [
        TrajResult(traj_id=i, uid=uids[i], traj_uid=traj_uid_factory())
        for i in range(n_traj)
    ]

    # Concurrency instrumentation + optional generation gate.
    state = {"in_flight": 0, "peak": 0}
    gate = asyncio.Semaphore(max_in_flight) if max_in_flight else None

    async def _generate(traj_id: int, step: int, obs: Any) -> Any:
        if gate is not None:
            await gate.acquire()
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            return await generate_action(traj_id, step, obs)
        finally:
            state["in_flight"] -= 1
            if gate is not None:
                gate.release()

    async def _run_trajectory(i: int) -> None:
        res = results[i]
        obs = initial_obs[i]
        for step in range(max_steps):
            action = await _generate(i, step, obs)
            next_obs, reward, done, info = await env_step(i, step, action)
            res.steps.append(
                StepRecord(
                    traj_id=i,
                    step=step,
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=bool(done),
                    info=info,
                )
            )
            res.episode_reward += float(reward)
            res.episode_length += 1
            obs = next_obs
            if done:
                break

    await asyncio.gather(*[_run_trajectory(i) for i in range(n_traj)])
    return RolloutResult(trajectories=results, max_concurrent_generate=state["peak"])
