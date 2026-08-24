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
"""The launcher's environment wins over the pinned Ray runtime env.

``PPO_RAY_RUNTIME_ENV`` pins ``CUDA_DEVICE_MAX_CONNECTIONS=1`` for every worker.
That is Megatron's requirement -- its tensor-parallel overlap needs the
communication kernels on the same hardware queue as the compute, in program
order. FSDP wants the opposite: ``forward_prefetch`` issues the next layer's
all-gather on a separate stream precisely so it overlaps, and one connection
serialises the pair that was supposed to overlap.

The FSDP arms opt out by exporting the variable before launching
(``examples/sft_trainer/run_multitask_sft_qwen3.sh``). That only works because
``get_ppo_ray_runtime_env`` drops keys the launcher already set, leaving Ray to
pass the process environment through untouched. This file pins that behaviour:
the export is silent when it stops working -- the workers would simply go back
to one connection and the run would get slower with nothing in the log to say
so.
"""

import pytest

from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV, get_ppo_ray_runtime_env


def test_the_pin_is_still_there():
    """If the pin is ever dropped upstream, the export below becomes dead code
    rather than an override, and the comment explaining it becomes a lie."""
    assert PPO_RAY_RUNTIME_ENV["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_unset_leaves_the_pin_in_place(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    assert get_ppo_ray_runtime_env()["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


@pytest.mark.parametrize("value", ["8", "1", "32"])
def test_an_exported_value_removes_the_pin(monkeypatch, value):
    """Removed, not overwritten: the key is dropped from the runtime env so the
    worker inherits the launcher's process environment for it."""
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", value)
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in get_ppo_ray_runtime_env()["env_vars"]


def test_the_filter_does_not_mutate_the_pinned_table(monkeypatch):
    """get_ppo_ray_runtime_env copies before filtering. Without the copy the
    first call would strip the key from the module-level dict and every later
    call in the same process -- including other arms' -- would silently lose the
    pin."""
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    get_ppo_ray_runtime_env()
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    assert get_ppo_ray_runtime_env()["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_the_other_pins_are_untouched_by_one_override(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    for key in ("NCCL_DEBUG", "NCCL_CUMEM_ENABLE", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(key, raising=False)
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["NCCL_CUMEM_ENABLE"] == "0"
    assert env["TOKENIZERS_PARALLELISM"] == "true"


def test_the_run_script_exports_it():
    """The script is the only place the override lives, so a rename or a moved
    block should fail here rather than in a run whose numbers quietly regress."""
    with open("examples/sft_trainer/run_multitask_sft_qwen3.sh") as handle:
        script = handle.read()
    assert "export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}" in script
