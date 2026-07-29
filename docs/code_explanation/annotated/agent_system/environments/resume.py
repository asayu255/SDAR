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

# [EXPLAIN] checkpoint は dataloader 位置を復元しても Ray actor 内の環境 RNG/cycle
# [EXPLAIN] までは保存しないため、stateful な環境だけ reset 回数を再生して連続実行時の
# [EXPLAIN] episode 順へ追いつかせる。これは model state の復元とは独立した決定性対策である。
"""Replay env-side episode schedules after a checkpoint resume.

Some environments pick their episode from an internal, seeded schedule instead of
from the training batch:

* WebShop draws goals from ``self._rng`` (``webshop/envs.py``).
* ALFWorld pulls the next game from TextWorld's seeded game-file cycle
  (``alfworld/envs.py``).

Those schedules live in the training process (and its Ray actors) and are rebuilt
from the seed on every start, so a resumed run restarts them at position 0 while
an uninterrupted run would be ``global_steps`` resets in — the resumed run then
re-trains on the same early episodes. Replaying the skipped resets restores the
sequence.

Search is unaffected: its episodes arrive through ``env_kwargs`` from the
dataloader, whose position is restored from the checkpoint.

The invariant this relies on is that the training loop resets the envs exactly
once per completed global step. That holds for ``vanilla_multi_turn_loop`` and is
preserved by ``ENV_RESET_PREFETCH`` (which moves the single reset earlier without
adding one). It does *not* hold when ``algorithm.filter_groups.enable`` is set --
callers must check that before passing a step count in.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Iterator, List, Tuple


# [EXPLAIN] `_leaf_envs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _leaf_envs(env_manager) -> Iterator[Tuple[str, Any]]:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Yield ``(label, vector_env)`` for every task behind an env manager.

    Handles both the single-task managers (``EnvironmentManagerBase.envs``) and
    ``MultiTaskEnvironmentManager``, which keeps one sub-manager per task in
    ``.managers`` and has no ``.envs`` of its own.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    managers = getattr(env_manager, "managers", None)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(managers, dict) and managers:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task in sorted(managers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            leaf = getattr(managers[task], "envs", None)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if leaf is not None:
                # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
                yield task, leaf
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    leaf = getattr(env_manager, "envs", None)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if leaf is not None:
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield type(leaf).__name__, leaf


# [EXPLAIN] `fast_forward_env_schedules` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def fast_forward_env_schedules(envs, num_resets: int) -> List[str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Advance every stateful schedule reachable from ``envs`` by ``num_resets``.

    Vector envs opt in by exposing a ``fast_forward(num_resets)`` method; the ones
    that do not (search, sokoban, ...) select their episodes statelessly and are
    skipped. Returns one human-readable message per env that was advanced.

    Never raises: failing to replay costs reproducibility of the episode order,
    which is not worth killing a resumed run over, so failures are reported and
    training continues.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    messages: List[str] = []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if envs is None or num_resets is None or num_resets <= 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return messages

    # [EXPLAIN] multitask manager では task 名を sort して走査するが、各 leaf は独立した
    # [EXPLAIN] schedule を持つため task 間の呼び出し順は episode 選択結果を変えない。
    for label, leaf in _leaf_envs(envs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fast_forward = getattr(leaf, "fast_forward", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not callable(fast_forward):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            messages.append(f"{label}: nothing to replay (episode selection is stateless)")
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            messages.append(f"{label}: {fast_forward(num_resets)}")
        # [EXPLAIN] ここは学習継続性を優先する best-effort 境界である。失敗は握り潰さず
        # [EXPLAIN] message に残るが、再開後の episode 順が完全一致しない可能性は残る。
        except Exception as exc:  # noqa: BLE001 - never block a resume
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            messages.append(
                f"{label}: WARNING fast_forward({num_resets}) failed ({exc!r}); "
                f"the resumed run will replay episodes from the start of the schedule"
            )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return messages
