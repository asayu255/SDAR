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
# [EXPLAIN] この実験的 core は engine/env の実装を callback に隔離し、同一 trajectory 内の
# [EXPLAIN] generate→step の因果順だけを固定する。trajectory 間の完了順は変わっても、
# [EXPLAIN] results[i] へ格納するため最終 row 順は入力 traj_id 順に安定する。
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from __future__ import annotations

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass, field
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Type aliases for the two pluggable async operations.
# generate_action(traj_id, step, obs) -> action
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
GenerateFn = Callable[[int, int, Any], Awaitable[Any]]
# env_step(traj_id, step, action) -> (next_obs, reward, done, info)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
EnvStepFn = Callable[[int, int, Any], Awaitable[tuple]]


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `StepRecord` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class StepRecord:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """One (action, env-response) interaction of a single trajectory.

    Mirrors exactly the per-step information the synchronous loop appends to
    total_batch_list[i] / total_infos[i] for an *active* step. The async loop
    never produces records for finished trajectories (the sync loop produces
    them only to immediately discard them via active_masks=False), so the
    downstream effective batch is identical.
    """

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    traj_id: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    step: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    obs: Any
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    action: Any
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    reward: float
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    done: bool
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    info: Any = None


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `TrajResult` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TrajResult:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    traj_id: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    uid: Any
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    traj_uid: Any
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    steps: List[StepRecord] = field(default_factory=list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    episode_reward: float = 0.0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    episode_length: int = 0


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `RolloutResult` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RolloutResult:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    trajectories: List[TrajResult]
    # Diagnostics that let tests assert the schedule actually overlaps work.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_concurrent_generate: int = 0


# [EXPLAIN] `assign_uids` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def assign_uids(n_traj: int, rollout_n: int, uid_factory: Callable[[], Any]) -> List[Any]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Group trajectories into GRPO groups exactly like vanilla_multi_turn_loop:
    consecutive blocks of ``rollout_n`` share a uid. rollout_n <= 0 means a
    single group. This is order-independent for downstream advantage grouping,
    but we keep the identical layout to avoid any divergence.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    uids: List[Any] = []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rollout_n and rollout_n > 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        uid = None
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(n_traj):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i % rollout_n == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                uid = uid_factory()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            uids.append(uid)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        uid = uid_factory()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        uids = [uid for _ in range(n_traj)]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return uids


# [EXPLAIN] `collect_async` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
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
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Drive ``n_traj`` trajectories concurrently to completion.

    Each trajectory independently repeats ``generate_action`` -> ``env_step``
    until the env reports ``done`` or it has taken ``max_steps`` steps — the same
    termination rule as the synchronous loop (which breaks a trajectory once
    ``is_done`` or the shared ``for _step in range(max_steps)`` is exhausted).

    Args mirror the sync loop's state. ``max_in_flight`` optionally caps the
    number of concurrently *generating* trajectories (does not change collected
    data, only scheduling pressure on the engine).
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if uid_factory is None:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import uuid

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        uid_factory = lambda: str(uuid.uuid4())  # noqa: E731
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if traj_uid_factory is None:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import uuid

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_uid_factory = lambda: str(uuid.uuid4())  # noqa: E731

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    uids = assign_uids(n_traj, rollout_n, uid_factory)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = [
        TrajResult(traj_id=i, uid=uids[i], traj_uid=traj_uid_factory())
        for i in range(n_traj)
    ]

    # Concurrency instrumentation + optional generation gate.
    # [EXPLAIN] semaphore が制限するのは生成中 request 数だけで、env_step の並行性や
    # [EXPLAIN] trajectory の意味論は変えない。0/None は無制限として扱われる。
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state = {"in_flight": 0, "peak": 0}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    gate = asyncio.Semaphore(max_in_flight) if max_in_flight else None

    # [EXPLAIN] `_generate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def _generate(traj_id: int, step: int, obs: Any) -> Any:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if gate is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await gate.acquire()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        state["in_flight"] += 1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        state["peak"] = max(state["peak"], state["in_flight"])
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return await generate_action(traj_id, step, obs)
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            state["in_flight"] -= 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if gate is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                gate.release()

    # [EXPLAIN] trajectory ごとに coroutine を一つ持たせることで、遅い環境が別 trajectory の
    # [EXPLAIN] 次 token 生成を止めない。一方、同一 trajectory は await により逐次実行される。
    async def _run_trajectory(i: int) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        res = results[i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs = initial_obs[i]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for step in range(max_steps):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action = await _generate(i, step, obs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_obs, reward, done, info = await env_step(i, step, action)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
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
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            res.episode_reward += float(reward)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            res.episode_length += 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs = next_obs
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if done:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    await asyncio.gather(*[_run_trajectory(i) for i in range(n_traj)])
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return RolloutResult(trajectories=results, max_concurrent_generate=state["peak"])
