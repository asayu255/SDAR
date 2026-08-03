# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""The speedup mechanisms ship ON, and only where they are supposed to.

Every knob here defaults to off in the code that reads it -- ``fsdp_config.get(
"forward_prefetch", False)`` and friends -- because that is the safe reading for
a config that predates the key. What turns them on is the shipped
``ppo_trainer.yaml``, which nothing else asserts: flipping a line there is a
one-character change that no other test would notice, and the whole point of
these being defaults is that no run script has to remember them.

The second half is the blast radius. ``sharding_strategy`` and
``forward_prefetch`` are read by whichever ``fsdp_config`` block the worker was
built from, so putting them under the actor is what keeps them out of the ref,
critic and reward-model paths -- the ref especially, where ZeRO-2 would fight
the CPU offload the teachers rely on.
"""

import os

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

_CFG = os.path.join(
    os.path.dirname(__file__), "..", "..", "verl", "trainer", "config", "ppo_trainer.yaml"
)


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(os.path.abspath(_CFG))


@pytest.mark.parametrize(
    "key, expected",
    [
        # ZeRO-2: one all-gather per micro-batch instead of three.
        ("actor_rollout_ref.actor.fsdp_config.sharding_strategy", "shard_grad_op"),
        # Overlap the next unit's all-gather with the current unit's compute.
        ("actor_rollout_ref.actor.fsdp_config.forward_prefetch", True),
        # One gradient reduce per mini-batch instead of one per micro-batch.
        ("actor_rollout_ref.actor.no_sync_grad_accum", True),
        # Teachers stay resident instead of being re-streamed from host memory
        # on every micro-batch. False here means "do not offload".
        ("actor_rollout_ref.ref.fsdp_config.param_offload", False),
    ],
)
def test_the_shipped_config_turns_the_mechanism_on(cfg, key, expected):
    got = OmegaConf.select(cfg, key)
    assert got == expected and type(got) is type(expected), (
        f"{key} ships as {got!r}; the mechanism is off by default in the reader, "
        f"so this file is the only thing enabling it"
    )


@pytest.mark.parametrize(
    "block",
    [
        "actor_rollout_ref.ref.fsdp_config",
        "critic.model.fsdp_config",
        "reward_model.model.fsdp_config",
    ],
)
@pytest.mark.parametrize("key", ["sharding_strategy", "forward_prefetch"])
def test_the_actor_only_knobs_stay_out_of_the_other_workers(cfg, block, key):
    """These three build their own FSDP from their own block.

    ``get_sharding_strategy`` reads the block it is handed, so a copy of the key
    landing here would silently move that worker to ZeRO-2 as well. For the ref
    that is actively wrong: its parameters are offloaded to CPU by design, and
    ZeRO-2 exists to keep parameters resident, so the two cancel -- which is why
    ``get_sharding_strategy`` warns about exactly that combination.
    """
    node = OmegaConf.select(cfg, block)
    assert node is not None, f"{block} is missing from the config"
    assert key not in node, (
        f"{block}.{key} would extend an actor-only default to {block.split('.')[0]}"
    )


def test_teacher_prefetch_defaults_on():
    """The env-var half of the same story.

    Inert outside OPD -- ``multi_turn_loop`` ignores it unless the trainer hands
    over a ``teacher_prefetch_fn`` -- so defaulting it on costs nothing to the
    recipes that have no teacher.
    """
    try:
        from agent_system.multi_turn_rollout import rollout_loop
    except Exception as e:  # pragma: no cover - environment without full deps
        pytest.skip(f"rollout_loop import unavailable: {e}")

    assert os.environ.get("ROLLOUT_PREFETCH_TEACHER") is None, (
        "the ambient environment sets ROLLOUT_PREFETCH_TEACHER, so this test "
        "cannot see the shipped default"
    )
    assert rollout_loop._ROLLOUT_PREFETCH_TEACHER is True
