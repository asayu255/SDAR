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
"""``trainer.expected_config`` has to actually be read, or the lock is decorative.

``tests/trainer/test_expected_config.py`` covers the validator. This covers the
one line that calls it: ``_validate_config`` is the choke point every trainer in
this repo reaches, so pinning the intent there is what makes
``+trainer.expected_config=<yaml>`` mean something on a real run.

Two things have to hold. A run whose effective config drifted from the lock must
abort *here*, in the seconds before any worker is built -- the whole value of the
mechanism is that it fails before a multi-hour run, not after. And a run that
does not pass the key must behave exactly as it did before the key existed,
since every recipe in this repo is such a run.
"""

import os

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "verl",
    "trainer",
    "config",
    "ppo_trainer.yaml",
)


class _Stub:
    """Only the attributes ``_validate_config`` reads off ``self``."""

    def __init__(self, config):
        self.config = config
        self.use_critic = False
        self.use_reference_policy = True


def _config(expect_file=None):
    config = OmegaConf.load(CONFIG_PATH)
    config.data.train_batch_size = 45
    config.trainer.n_gpus_per_node = 2
    config.trainer.nnodes = 1
    config.actor_rollout_ref.actor.ppo_micro_batch_size = None
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 5
    config.actor_rollout_ref.ref.log_prob_micro_batch_size = None
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 16
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size = None
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 16
    if expect_file is not None:
        OmegaConf.update(config, "trainer.expected_config", str(expect_file), force_add=True)
    return config


def test_no_expected_config_key_leaves_validation_unchanged():
    """Every existing recipe is this case."""
    RayPPOTrainer._validate_config(_Stub(_config()))


def test_a_matching_lock_passes(tmp_path):
    lock = tmp_path / "expect.yaml"
    lock.write_text(
        '"actor_rollout_ref.actor.no_sync_grad_accum": true\n'
        '"actor_rollout_ref.actor.fsdp_config.sharding_strategy": shard_grad_op\n'
        '"data.train_batch_size": 45\n'
    )
    RayPPOTrainer._validate_config(_Stub(_config(lock)))


def test_a_drifted_knob_aborts_before_any_worker_is_built(tmp_path):
    """The scenario the lock exists for: the script says one thing, the effective
    config says another, and without this the run reports the wrong arm."""
    lock = tmp_path / "expect.yaml"
    lock.write_text('"actor_rollout_ref.actor.fsdp_config.sharding_strategy": full_shard\n')

    with pytest.raises(AssertionError) as excinfo:
        RayPPOTrainer._validate_config(_Stub(_config(lock)))

    message = str(excinfo.value)
    assert "actor_rollout_ref.actor.fsdp_config.sharding_strategy" in message
    assert "shard_grad_op" in message and "full_shard" in message


def test_the_lock_sees_the_shipped_defaults_not_just_explicit_overrides(tmp_path):
    """A default that flips underneath a comparison is exactly what this catches,
    so the check has to read the composed config rather than the CLI arguments."""
    lock = tmp_path / "expect.yaml"
    # Never passed on any command line -- it comes from ppo_trainer.yaml.
    lock.write_text('"actor_rollout_ref.actor.fsdp_config.forward_prefetch": true\n')
    RayPPOTrainer._validate_config(_Stub(_config(lock)))


def test_an_empty_value_is_treated_as_no_lock(tmp_path):
    """`trainer.expected_config=` (or null) must not try to open "" and crash."""
    config = _config()
    OmegaConf.update(config, "trainer.expected_config", None, force_add=True)
    RayPPOTrainer._validate_config(_Stub(config))


def test_a_missing_lock_file_says_where_it_looked(tmp_path):
    """Relative paths resolve against the launch directory, like skills_dir.

    A bare FileNotFoundError from inside the validator would send someone
    looking at the yaml rather than at their working directory.
    """
    config = _config(tmp_path / "does_not_exist.yaml")
    with pytest.raises(FileNotFoundError) as excinfo:
        RayPPOTrainer._validate_config(_Stub(config))
    message = str(excinfo.value)
    assert "does_not_exist.yaml" in message
    assert "launch directory" in message
