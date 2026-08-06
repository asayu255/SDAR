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
built from, so putting them under the actor is what keeps them out of the
critic and reward-model paths.

The ref is the exception, and only for ``sharding_strategy``. It carries its own
copy of that key, defaulting to null so the shipped behaviour is unchanged, and
the run script opts in -- because the teachers here are RESIDENT
(``ref.fsdp_config.param_offload=False``), not offloaded. With the shards on the
GPU, FULL_SHARD still re-all-gathers each teacher for every micro-batch; ZeRO-2
does not reshard after forward and the teachers have no backward, so each is
gathered once per phase. What ``get_sharding_strategy`` warns about is ZeRO-2
combined with param_offload=True, which is not this configuration.
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
        # The reference policy stays resident instead of being re-streamed from
        # host memory on every micro-batch. False here means "do not offload".
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
    "block, key",
    [
        # The ref keeps forward_prefetch out, but carries sharding_strategy --
        # see the module docstring and the null-default test below.
        ("actor_rollout_ref.ref.fsdp_config", "forward_prefetch"),
        ("critic.model.fsdp_config", "sharding_strategy"),
        ("critic.model.fsdp_config", "forward_prefetch"),
        ("reward_model.model.fsdp_config", "sharding_strategy"),
        ("reward_model.model.fsdp_config", "forward_prefetch"),
    ],
)
def test_the_actor_only_knobs_stay_out_of_the_other_workers(cfg, block, key):
    """These build their own FSDP from their own block.

    ``get_sharding_strategy`` reads the block it is handed, so a copy of the key
    landing here would silently move that worker to ZeRO-2 as well.
    """
    node = OmegaConf.select(cfg, block)
    assert node is not None, f"{block} is missing from the config"
    assert key not in node, (
        f"{block}.{key} would extend an actor-only default to {block.split('.')[0]}"
    )


def test_the_ref_carries_sharding_strategy_but_does_not_ship_it_on(cfg):
    """Declared so a run script can ask for it without a `+` override, null so
    that asking is what turns it on.

    The teachers are resident here, so ZeRO-2 removes the per-micro-batch
    re-all-gather that FULL_SHARD does over PCIe -- measured at pcieRX 768 -> 208
    on the pure-OPD arm, for +8.2 GB of gathered parameters. Worth opting into,
    not worth defaulting to: the memory cost scales with the number of teachers.
    """
    node = OmegaConf.select(cfg, "actor_rollout_ref.ref.fsdp_config")
    assert "sharding_strategy" in node, (
        "ref.fsdp_config.sharding_strategy is missing; the run script would need "
        "a `+` override to ask for ZeRO-2 on the teachers"
    )
    assert node.sharding_strategy is None, (
        f"ref.fsdp_config.sharding_strategy ships as {node.sharding_strategy!r}; "
        "it must default to null so the shipped behaviour stays FULL_SHARD"
    )
