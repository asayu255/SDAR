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

from typing import Any, Iterator, List, Tuple


def _leaf_envs(env_manager) -> Iterator[Tuple[str, Any]]:
    """Yield ``(label, vector_env)`` for every task behind an env manager.

    Handles both the single-task managers (``EnvironmentManagerBase.envs``) and
    ``MultiTaskEnvironmentManager``, which keeps one sub-manager per task in
    ``.managers`` and has no ``.envs`` of its own.
    """
    managers = getattr(env_manager, "managers", None)
    if isinstance(managers, dict) and managers:
        for task in sorted(managers):
            leaf = getattr(managers[task], "envs", None)
            if leaf is not None:
                yield task, leaf
        return

    leaf = getattr(env_manager, "envs", None)
    if leaf is not None:
        yield type(leaf).__name__, leaf


def fast_forward_env_schedules(envs, num_resets: int) -> List[str]:
    """Advance every stateful schedule reachable from ``envs`` by ``num_resets``.

    Vector envs opt in by exposing a ``fast_forward(num_resets)`` method; the ones
    that do not (search, sokoban, ...) select their episodes statelessly and are
    skipped. Returns one human-readable message per env that was advanced.

    Never raises: failing to replay costs reproducibility of the episode order,
    which is not worth killing a resumed run over, so failures are reported and
    training continues.
    """
    messages: List[str] = []
    if envs is None or num_resets is None or num_resets <= 0:
        return messages

    for label, leaf in _leaf_envs(envs):
        fast_forward = getattr(leaf, "fast_forward", None)
        if not callable(fast_forward):
            messages.append(f"{label}: nothing to replay (episode selection is stateless)")
            continue
        try:
            messages.append(f"{label}: {fast_forward(num_resets)}")
        except Exception as exc:  # noqa: BLE001 - never block a resume
            messages.append(
                f"{label}: WARNING fast_forward({num_resets}) failed ({exc!r}); "
                f"the resumed run will replay episodes from the start of the schedule"
            )
    return messages
