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
