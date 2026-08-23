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
"""SKIP_ROLLOUT_BUILD: the flag that lets the SFT arm run under expandable_segments.

The off-policy arms never generate (test_freq=-1, validation is a separate
process), yet the vLLM engine is built on every rank anyway -- and merely
building it forbids PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, because
vLLM's CuMemAllocator asserts expandable segments are off. Skipping the build is
what unlocks the allocator mode that removes cudaMalloc segment-growth stalls.

What has to hold: the flag defaults off (every other arm keeps its rollout), a
skipped rollout leaves the session hooks harmless, and a run that skipped the
build but then tries to generate dies with words that name the flag -- an
AttributeError three frames deep would send whoever hits it straight past the
actual cause.
"""

import pytest

import verl.workers.fsdp_workers as fsdp_workers


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SKIP_ROLLOUT_BUILD", raising=False)
    assert fsdp_workers.skip_rollout_build() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("0", False), ("no", False), ("", False)])
def test_flag_parses_like_the_other_switches(monkeypatch, value, expected):
    monkeypatch.setenv("SKIP_ROLLOUT_BUILD", value)
    assert fsdp_workers.skip_rollout_build() is expected


def test_generate_sequences_names_the_flag_when_skipped():
    """The failure must say SKIP_ROLLOUT_BUILD, not unravel as an AttributeError.

    Only the guard is under test, so the worker is assembled with object.__new__
    and the method body is entered just far enough to reach it.
    """
    worker = object.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_rollout = True
    worker.rollout = None
    worker.rollout_sharding_manager = None

    class _StaysWhereItIs:
        meta_info = {}

        def to(self, _device):
            return self

    inner = fsdp_workers.ActorRolloutRefWorker.generate_sequences
    # unwrap the @register dispatch decorator if it wrapped the function
    inner = getattr(inner, "__wrapped__", inner)
    with pytest.raises(RuntimeError, match="SKIP_ROLLOUT_BUILD"):
        inner(worker, _StaysWhereItIs())


def test_session_hooks_are_noops_without_a_rollout():
    """ROLLOUT_KEEP_VLLM_AWAKE calls these from the rollout loop; a skipped
    build must leave them callable rather than a second place to crash."""
    worker = object.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_rollout = True
    worker.rollout = None
    worker.rollout_sharding_manager = None

    begin = getattr(fsdp_workers.ActorRolloutRefWorker.begin_rollout_session, "__wrapped__",
                    fsdp_workers.ActorRolloutRefWorker.begin_rollout_session)
    end = getattr(fsdp_workers.ActorRolloutRefWorker.end_rollout_session, "__wrapped__",
                  fsdp_workers.ActorRolloutRefWorker.end_rollout_session)
    begin(worker)
    end(worker)
