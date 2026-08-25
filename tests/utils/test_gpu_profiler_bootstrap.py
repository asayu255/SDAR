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
"""Who starts the NVML sampler, and what happens to a reader when nobody does.

The turn table's genGPU% column printed "-" on every evaluation ever run,
GPU_PROFILER=1 included. The sampler is created by push_phase, which lives in
the trainer's fit loop; a validation-only run never calls it, so
mean_util_between found no sampler and answered None for every window. A column
of dashes reads as "the GPU was not measured" when it means "nothing asked".

So a reader can start it now. These tests hold that, and that the flag still
decides -- an unset GPU_PROFILER must not have a sampler thread appear because
somebody printed a table.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from verl.utils import gpu_profiler  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(gpu_profiler, "_ENABLED", None)
    monkeypatch.setattr(gpu_profiler, "_sampler", None)
    monkeypatch.setattr(gpu_profiler, "_sampler_failed", False)
    yield


class _Backend:
    n_gpus = 3

    def sample(self):
        return [{"sm_util": 0.0, "mem_util": 0.0} for _ in range(self.n_gpus)]


def _with_gpus(monkeypatch):
    monkeypatch.setattr(gpu_profiler, "_make_backend", lambda: _Backend())
    monkeypatch.setattr(gpu_profiler, "_make_host_sampler", lambda: None)


def test_disabled_starts_nothing(monkeypatch):
    monkeypatch.delenv("GPU_PROFILER", raising=False)
    _with_gpus(monkeypatch)

    assert gpu_profiler.ensure_started() is False
    assert gpu_profiler._sampler is None


def test_enabled_starts_a_sampler(monkeypatch):
    monkeypatch.setenv("GPU_PROFILER", "1")
    _with_gpus(monkeypatch)

    assert gpu_profiler.ensure_started() is True
    assert gpu_profiler._sampler is not None


def test_a_second_call_reuses_the_same_sampler(monkeypatch):
    """It is called once per rollout, 413 times in an evaluation."""
    monkeypatch.setenv("GPU_PROFILER", "1")
    _with_gpus(monkeypatch)

    gpu_profiler.ensure_started()
    first = gpu_profiler._sampler
    for _ in range(5):
        assert gpu_profiler.ensure_started() is True
    assert gpu_profiler._sampler is first


def test_a_box_with_no_gpus_says_so_and_does_not_retry(monkeypatch):
    monkeypatch.setenv("GPU_PROFILER", "1")
    calls = []

    def _no_backend():
        calls.append(1)
        return None

    monkeypatch.setattr(gpu_profiler, "_make_backend", _no_backend)

    assert gpu_profiler.ensure_started() is False
    assert gpu_profiler.ensure_started() is False
    assert len(calls) == 1  # the failure is remembered


def test_a_reader_without_a_sampler_answers_none(monkeypatch):
    """The state that produced a column of dashes: enabled, never started."""
    monkeypatch.setenv("GPU_PROFILER", "1")

    assert gpu_profiler.mean_util_between(0.0, 1.0) is None
    assert gpu_profiler.per_gpu_util_between(0.0, 1.0) is None
