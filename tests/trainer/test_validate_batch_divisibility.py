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
"""The prompt-batch divisibility check is relaxed for the env-driven recipes.

``data.train_batch_size`` counts *prompts*, and in the env recipes those rows are
placeholders the environments turn into rollouts -- nothing dispatches them to the
workers unpadded. Requiring the prompt count to divide the GPU count would mean a
run moved from 3 GPUs to 2 has to change its per-task batch size, i.e. change the
experiment. The plain (non-env) path keeps the hard assert.
"""

import os

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "verl", "trainer", "config", "ppo_trainer.yaml")


class _Stub:
    """Only the attributes ``_validate_config`` reads off ``self``."""

    def __init__(self, config):
        self.config = config
        self.use_critic = False
        self.use_reference_policy = True


def _config(train_batch_size, n_gpus, with_env=True):
    config = OmegaConf.load(CONFIG_PATH)
    config.data.train_batch_size = train_batch_size
    config.trainer.n_gpus_per_node = n_gpus
    config.trainer.nnodes = 1
    # the multitask scripts' values; only the *_per_gpu variants may be set
    config.actor_rollout_ref.actor.ppo_micro_batch_size = None
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 5
    config.actor_rollout_ref.ref.log_prob_micro_batch_size = None
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 16
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size = None
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 16
    if not with_env:
        del config["env"]
    return config


def test_env_recipe_allows_prompt_count_indivisible_by_gpus():
    # the multitask scripts: 15 prompts x 3 tasks on 2 GPUs
    RayPPOTrainer._validate_config(_Stub(_config(45, 2)))


def test_env_recipe_still_accepts_the_divisible_case():
    RayPPOTrainer._validate_config(_Stub(_config(45, 3)))


def test_plain_recipe_keeps_the_hard_check():
    with pytest.raises(AssertionError, match="divisible by total n_gpus"):
        RayPPOTrainer._validate_config(_Stub(_config(45, 2, with_env=False)))
