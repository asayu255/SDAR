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
"""The deep-dip discriminators: deltas, not absolutes, and never fatal.

The value of these numbers is entirely in the correlation "this step dipped AND
this counter moved", so what has to hold is that a quiet step reads zero, an
event reads as the event count, and a counter source that is missing or resets
reads as silence rather than as noise or a crash.
"""

import importlib
import sys

import pytest


def _fresh():
    sys.modules.pop("verl.utils.metric.stall_counters", None)
    return importlib.import_module("verl.utils.metric.stall_counters")


class _Device:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


def test_the_first_reading_is_zero_not_the_process_lifetime():
    """cudaMalloc has run hundreds of times by the first step (model build,
    vLLM). Reporting the absolute would make step 1 look like a malloc storm
    and bury the one real event at step 40."""
    mod = _fresh()
    out = mod.per_rank_stall_counter_metrics(_Device({"num_device_alloc": 412}))
    assert out == {"stall/cuda_mallocs": 0.0, "stall/gc_gen2": 0.0,
                   "stall/dynamo_graphs": 0.0}


def test_a_malloc_between_steps_reads_as_exactly_that_many_events():
    mod = _fresh()
    device = _Device({"num_device_alloc": 412})
    mod.per_rank_stall_counter_metrics(device)
    device._stats = {"num_device_alloc": 415}

    out = mod.per_rank_stall_counter_metrics(device)

    assert out["stall/cuda_mallocs"] == 3.0


def test_a_quiet_step_reads_zero_everywhere():
    mod = _fresh()
    device = _Device({"num_device_alloc": 400})
    mod.per_rank_stall_counter_metrics(device)

    out = mod.per_rank_stall_counter_metrics(device)

    assert all(v == 0.0 for v in out.values())


def test_a_counter_that_goes_backwards_is_silence_not_negative_events():
    """torch._dynamo.reset() clears its counters mid-run. 'Minus forty events'
    in a chart reads as corruption; the clamp makes a reset read as a quiet
    step, which costs one step of attribution and nothing else."""
    mod = _fresh()
    assert mod.deltas([5.0, 5.0, 40.0], [3.0, 5.0, 90.0]) == [2.0, 0.0, 0.0]


def test_a_backend_without_the_counter_never_raises():
    class _Broken:
        def memory_stats(self):
            raise RuntimeError("no allocator stats on this backend")

    mod = _fresh()
    out = mod.per_rank_stall_counter_metrics(_Broken())
    assert out["stall/cuda_mallocs"] == 0.0     # absent reads as 'never moved'
