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
"""Envs are built on first use, not at startup.

They are Ray actors that idle until a validation pass runs -- 252 of the 492 in
multitask -- and with ``trainer.test_freq <= 0`` they are never used at all.
"""

import pytest

pytest.importorskip("torch")

try:
    from agent_system.environments.env_manager import LazyEnvManager
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"env_manager import unavailable: {e}", allow_module_level=True)


class _Envs:
    def __init__(self):
        self.tag = "envs"

    def reset(self):
        return "reset called"


def test_builder_does_not_run_until_something_is_used():
    calls = []
    lazy = LazyEnvManager(lambda: (calls.append(1), _Envs())[1])
    assert calls == []  # constructing the wrapper builds nothing

    assert lazy.tag == "envs"
    assert calls == [1]


def test_builder_runs_only_once():
    calls = []
    lazy = LazyEnvManager(lambda: (calls.append(1), _Envs())[1])

    lazy.reset()
    lazy.reset()
    assert lazy.tag == "envs"
    assert calls == [1]


def test_method_calls_reach_the_real_manager():
    lazy = LazyEnvManager(_Envs)
    assert lazy.reset() == "reset called"


def test_materialize_returns_the_same_instance():
    lazy = LazyEnvManager(_Envs)
    assert lazy.materialize() is lazy.materialize()


class _ValOnlyGuard:
    """The part of RayPPOTrainer the fast-forward guard needs, and nothing else.

    Binding the real method onto a stub keeps this a unit test of the guard
    rather than of Ray, FSDP and vLLM.
    """

    def __init__(self, envs, val_only):
        from omegaconf import OmegaConf

        self.envs = envs
        self.global_steps = 150
        self.config = OmegaConf.create(
            {
                "trainer": {"val_only": val_only},
                "algorithm": {"filter_groups": {"enable": False}},
                "env": {"env_name": "multitask"},
                "data": {"task_balance": {"enable": True}},
            }
        )

    def run(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        RayPPOTrainer._fast_forward_env_schedules(self)


def test_a_val_only_run_does_not_materialize_the_training_envs():
    """Validating a saved checkpoint in its own process must not build the 360
    training env actors -- that is the host RAM the split run exists to not hold,
    and holding both sides at once is what ended the run at step 150.

    OPDRayTrainer.fit calls _fast_forward_env_schedules right after
    _load_checkpoint, i.e. before the val_only return, and resume_path sets
    global_steps > 0 so the existing early-outs do not fire. _leaf_envs then does
    getattr(env_manager, "managers", None), which LazyEnvManager forwards to
    materialize() -- so reaching the replay at all is enough to build them and the
    guard has to come first.
    """
    calls = []
    lazy = LazyEnvManager(lambda: (calls.append(1), _Envs())[1])

    _ValOnlyGuard(lazy, val_only=True).run()

    assert calls == []


def test_a_training_resume_still_replays_the_schedules():
    """The guard must not disarm the fast-forward for the case it exists for:
    a resumed TRAINING run, which would otherwise re-train on the early
    episodes."""
    calls = []
    lazy = LazyEnvManager(lambda: (calls.append(1), _Envs())[1])

    _ValOnlyGuard(lazy, val_only=False).run()

    assert calls == [1]
