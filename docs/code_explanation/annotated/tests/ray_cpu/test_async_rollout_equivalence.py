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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import asyncio
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import importlib.util
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys

# Load async_rollout_core directly by path so this test stays torch-free (the
# package __init__ imports rollout_loop which needs torch; the core does not).
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
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
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_spec = importlib.util.spec_from_file_location("async_rollout_core", _CORE_PATH)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_core = importlib.util.module_from_spec(_spec)
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
sys.modules["async_rollout_core"] = _core  # needed for dataclass annotation resolution
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
_spec.loader.exec_module(_core)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
collect_async = _core.collect_async


# --------------------------------------------------------------------------- #
# Deterministic mock policy + env (no cross-trajectory state).
# --------------------------------------------------------------------------- #
# [EXPLAIN] `mock_policy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def mock_policy(traj_id, step, obs):
    # Action depends only on this trajectory's (id, step, obs) -> deterministic.
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return f"act[{traj_id}|{step}|{obs}]"


# [EXPLAIN] `make_mock_env` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def make_mock_env(done_at):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """done_at[i] = number of steps trajectory i runs before env returns done."""

    # [EXPLAIN] `env_step_sync` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def env_step_sync(traj_id, step, action):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        done = (step + 1) >= done_at[traj_id]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward = round(0.1 * traj_id + step, 3)  # deterministic per (i, step)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        next_obs = f"o[{traj_id}|{step + 1}]"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = {"valid": (step % 2 == 0)}
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return next_obs, reward, done, info

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return env_step_sync


# --------------------------------------------------------------------------- #
# Reference implementation of the SYNC loop's data semantics
# (mirrors TrajectoryCollector.vanilla_multi_turn_loop).
# --------------------------------------------------------------------------- #
# [EXPLAIN] `sync_reference` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def sync_reference(n_traj, max_steps, initial_obs, policy, env_step_sync, rollout_n):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done = [False] * n_traj
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = list(initial_obs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    steps = [[] for _ in range(n_traj)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_reward = [0.0] * n_traj
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_length = [0] * n_traj

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for turn in range(max_steps):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        active = [i for i in range(n_traj) if not done[i]]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not active:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions = {i: policy(i, turn, obs[i]) for i in active}  # generate active only
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in active:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_obs, reward, d, info = env_step_sync(i, turn, actions[i])
            # turn_step (gather) == position in the per-traj list == turn index,
            # because an active trajectory is appended every turn from 0 until done.
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            steps[i].append((turn, actions[i], reward, bool(d), info))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ep_reward[i] += float(reward)
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            ep_length[i] += 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs[i] = next_obs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            done[i] = done[i] or bool(d)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if all(done):
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break

    # GRPO grouping partition (which trajectories share a uid).
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    groups = {}
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rollout_n and rollout_n > 0:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(n_traj):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            groups.setdefault(i // rollout_n, []).append(i)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        groups[0] = list(range(n_traj))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partition = sorted(sorted(v) for v in groups.values())
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return steps, ep_reward, ep_length, partition


# [EXPLAIN] `async_to_comparable` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def async_to_comparable(result):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = len(result.trajectories)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    steps = [[] for _ in range(n)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_reward = [0.0] * n
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ep_length = [0] * n
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for tr in result.trajectories:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for s in tr.steps:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            steps[tr.traj_id].append((s.step, s.action, s.reward, s.done, s.info))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ep_reward[tr.traj_id] = tr.episode_reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ep_length[tr.traj_id] = tr.episode_length
    # uid grouping partition
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    uid_to_ids = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for tr in result.trajectories:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        uid_to_ids.setdefault(tr.uid, []).append(tr.traj_id)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partition = sorted(sorted(v) for v in uid_to_ids.values())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    traj_uids = [tr.traj_uid for tr in result.trajectories]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return steps, ep_reward, ep_length, partition, traj_uids


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
# [EXPLAIN] `_run_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _run_case(n_traj, max_steps, rollout_n, done_at, slow_set=None, max_in_flight=None):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    initial_obs = [f"o[{i}|0]" for i in range(n_traj)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    env_step_sync = make_mock_env(done_at)

    # deterministic uid/traj_uid factories (independent counters; we compare
    # grouping *structure*, not raw ids, so call order doesn't matter)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    uc = {"n": 0}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tc = {"n": 0}

    # [EXPLAIN] `uid_factory` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def uid_factory():
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        uc["n"] += 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"uid{uc['n']}"

    # [EXPLAIN] `traj_uid_factory` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def traj_uid_factory():
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        tc["n"] += 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"traj{tc['n']}"

    # [EXPLAIN] `gen_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def gen_fn(i, step, obs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        await asyncio.sleep(0)  # real AsyncLLM generation yields to the loop
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return mock_policy(i, step, obs)

    # [EXPLAIN] `env_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    async def env_fn(i, step, action):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if slow_set and i in slow_set:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            await asyncio.sleep(0.01)  # emulate slow env (e.g. search HTTP)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return env_step_sync(i, step, action)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
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

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a_steps, a_r, a_l, a_part, a_traj_uids = async_to_comparable(result)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    s_steps, s_r, s_l, s_part = sync_reference(
        n_traj, max_steps, initial_obs, mock_policy, env_step_sync, rollout_n
    )

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert a_steps == s_steps, "per-step records differ between async and sync!"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert a_r == s_r, "episode rewards differ!"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert a_l == s_l, "episode lengths differ!"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert a_part == s_part, f"GRPO uid grouping differs! async={a_part} sync={s_part}"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(set(a_traj_uids)) == n_traj, "traj_uids must be unique"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return result, s_l


# [EXPLAIN] `test_equivalence_mixed_horizons` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_equivalence_mixed_horizons():
    # Emulate the real task mix: search finishes ~4, webshop ~15, alfworld ~50.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 24
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rollout_n = 8
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(n):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task = (i // rollout_n) % 3
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        done_at.append({0: 4, 1: 15, 2: 50}[task])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _run_case(n_traj=n, max_steps=50, rollout_n=rollout_n, done_at=done_at)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: mixed-horizon trajectories identical (sync == async)")


# [EXPLAIN] `test_equivalence_capped_at_max_steps` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_equivalence_capped_at_max_steps():
    # Trajectories that never finish must be capped at max_steps in both.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 6
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = [100] * n  # never finishes within max_steps
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, lengths = _run_case(n_traj=n, max_steps=12, rollout_n=2, done_at=done_at)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert all(l == 12 for l in lengths), lengths
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: never-done trajectories capped at max_steps identically")


# [EXPLAIN] `test_equivalence_all_finish_immediately` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_equivalence_all_finish_immediately():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = [1] * n  # done after a single step
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _, lengths = _run_case(n_traj=n, max_steps=50, rollout_n=1, done_at=done_at)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert all(l == 1 for l in lengths), lengths
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: single-step trajectories identical")


# [EXPLAIN] `test_no_grouping` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_no_grouping():
    # rollout_n=0 -> all share one uid; trajectories still identical.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 7
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = [3, 5, 2, 8, 1, 4, 6]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _run_case(n_traj=n, max_steps=10, rollout_n=0, done_at=done_at)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: ungrouped rollout identical")


# [EXPLAIN] `test_continuous_batching_overlap` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_continuous_batching_overlap():
    # With slow env steps on some trajectories, others must keep generating
    # concurrently (continuous batching) -> peak concurrency > 1, and results
    # still identical to sync.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 12
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = [5] * n
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    slow = {0, 1, 2}  # these have slow env (await sleep)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result, _ = _run_case(
        n_traj=n, max_steps=5, rollout_n=4, done_at=done_at, slow_set=slow
    )
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result.max_concurrent_generate > 1, (
        f"expected overlap, peak={result.max_concurrent_generate}"
    )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"PASS: continuous batching overlaps (peak concurrent generate={result.max_concurrent_generate})")


# [EXPLAIN] `test_max_in_flight_cap_preserves_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_max_in_flight_cap_preserves_data():
    # Capping in-flight generations must not change the collected data.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    n = 10
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    done_at = [i % 7 + 1 for i in range(n)]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _run_case(n_traj=n, max_steps=20, rollout_n=2, done_at=done_at, max_in_flight=3)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("PASS: max_in_flight cap preserves trajectories")


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_equivalence_mixed_horizons()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_equivalence_capped_at_max_steps()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_equivalence_all_finish_immediately()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_no_grouping()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_continuous_batching_overlap()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_max_in_flight_cap_preserves_data()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\nALL ASYNC-ROLLOUT EQUIVALENCE TESTS PASSED")
