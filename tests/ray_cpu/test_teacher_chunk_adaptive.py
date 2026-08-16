# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
"""Adaptive teacher chunk sizing, and what happens when a chunk fails.

The chunk decides how many already-final rows one prefetch call scores. It can
therefore never change a value -- the teacher is frozen and the rows are fixed --
so these tests are about scheduling and failure handling, not arithmetic:

  * the size tracks the glue window the chunk will run under, bounded by the
    fixed size (floor) and _..._CHUNK_MAX (ceiling);
  * a chunk that raises ON THE DRIVER is dropped, not propagated:
    compute_teacher_log_probs rescores whatever is missing and its completeness
    guard raises on anything left over, so failing loudly here would kill a step
    for no gain;
  * a chunk that raises INSIDE THE WORKER is fatal. That one is about collectives,
    not values: the teacher forward is FSDP, so a rank that raises has walked out
    of an all-gather the others are still in, and they will block there until the
    NCCL watchdog aborts them. Dropping it does not save the run, it only delays
    the same death by the watchdog timeout and loses the cause.

Run:  python tests/ray_cpu/test_teacher_chunk_adaptive.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_system.multi_turn_rollout import rollout_loop as RL
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def _collector(rate=None, glue=None):
    c = TrajectoryCollector.__new__(TrajectoryCollector)
    c._logprob_prefetch_enabled = False
    c._logprob_pending = []
    c._teacher_prefetch_fn = lambda chunk: {k: "scored" for k, _ in chunk}
    c._teacher_pending = []
    c._prefetched_teacher = {}
    c._teacher_future = None
    c._teacher_executor = None
    c._teacher_rate = rate
    c._teacher_inflight_rows = 0
    c._teacher_inflight_start = None
    c._teacher_glue_s = glue
    c._teacher_glue_mark = None
    return c


def test_size_falls_back_to_the_fixed_chunk_until_both_inputs_exist():
    """Rate needs a completed chunk, glue needs a completed turn. Before either,
    there is nothing to adapt from -- turn 0 in particular has no queue at all."""
    assert _collector(rate=None, glue=None)._teacher_chunk_size() == RL._ROLLOUT_PREFETCH_TEACHER_CHUNK
    assert _collector(rate=50.0, glue=None)._teacher_chunk_size() == RL._ROLLOUT_PREFETCH_TEACHER_CHUNK
    assert _collector(rate=None, glue=5.0)._teacher_chunk_size() == RL._ROLLOUT_PREFETCH_TEACHER_CHUNK


def test_a_wide_glue_window_gets_a_bigger_chunk():
    """The front of the rollout is glue-rich (turn 0 alone is ~8s) and was being
    served 128 rows at a time. 200 rows/s under a 5s window wants 1000, capped."""
    c = _collector(rate=200.0, glue=5.0)
    assert c._teacher_chunk_size() == RL._ROLLOUT_PREFETCH_TEACHER_CHUNK_MAX


def test_a_narrow_glue_window_never_goes_below_the_floor():
    """The tail's glue is ~0.4s. Shrinking the chunk to fit it exactly would stop
    the queue draining at all, so the configured size stays the floor."""
    c = _collector(rate=200.0, glue=0.05)  # wants 10
    assert c._teacher_chunk_size() == RL._ROLLOUT_PREFETCH_TEACHER_CHUNK


def test_size_is_proportional_between_the_bounds():
    c = _collector(rate=100.0, glue=2.0)  # wants 200
    want = 200
    lo, hi = RL._ROLLOUT_PREFETCH_TEACHER_CHUNK, RL._ROLLOUT_PREFETCH_TEACHER_CHUNK_MAX
    assert lo <= want <= hi, "test assumes 200 sits between the bounds"
    assert c._teacher_chunk_size() == want


def test_the_glue_window_only_records_positive_spans():
    c = _collector()
    c.note_teacher_glue_window(0.0)
    assert c._teacher_glue_s is None
    c.note_teacher_glue_window(1.5)
    assert c._teacher_glue_s == 1.5


def test_a_driver_side_failure_is_dropped_rather_than_propagated():
    """A chunk that raises ON THE DRIVER must not take the step down with it. Those
    rows were already removed from the pending pool, so they simply miss the prefetch
    and the trainer scores them in teacher_forward -- identical values, one slower
    step. Silent corruption is not possible: an unscored row trips the completeness
    guard in compute_teacher_log_probs."""
    c = _collector()

    class _Boom:
        def result(self):
            raise RuntimeError("simulated bookkeeping error")

    c._teacher_future = _Boom()
    c._teacher_inflight_rows = 128
    c._teacher_inflight_start = 0.0

    waited = c._join_teacher_prefetch()  # must not raise

    assert waited >= 0.0
    assert c._prefetched_teacher == {}
    assert c._teacher_future is None
    assert c._teacher_rate is None, "a failed chunk's timing must not feed the controller"
    assert c._teacher_inflight_rows == 0


def test_a_worker_side_failure_is_fatal_rather_than_dropped():
    """A chunk that raises INSIDE THE WORKER must propagate.

    This is the step-18 crash. The teacher forward is FSDP, so a rank that raises
    (OOM in lm_head) abandons an all-gather the other rank is still sitting in.
    Dropping the chunk let the rollout march on while rank 1 blocked in
    _ALLGATHER_BASE for the full 1,800,000 ms NCCL watchdog timeout and then aborted
    -- the run died anyway, half an hour later, with a traceback that named neither
    the OOM nor the chunk. There is nothing to recover: rescoring on the serial path
    needs the same process group.
    """
    ray_exc = pytest.importorskip("ray.exceptions")
    c = _collector()

    class _WorkerBoom:
        def result(self):
            raise ray_exc.RayTaskError(
                "teacher_alfworld_compute_ref_topk_log_prob", "traceback", torch.OutOfMemoryError("CUDA OOM")
            )

    c._teacher_future = _WorkerBoom()
    c._teacher_inflight_rows = 128
    c._teacher_inflight_start = 0.0

    with pytest.raises(RuntimeError, match="left an FSDP collective"):
        c._join_teacher_prefetch()

    assert c._teacher_future is None, "the future must be cleared even on the fatal path"
    assert c._teacher_inflight_rows == 0


def test_a_dead_worker_is_fatal_too():
    """The same chunk loop saw ActorDiedError right after the OOM, once rank 1's
    watchdog had aborted the process. Continuing to prefetch against a dead actor
    just burns turns until something else notices."""
    ray_exc = pytest.importorskip("ray.exceptions")
    if not hasattr(ray_exc, "ActorDiedError"):
        pytest.skip("this ray version has no ActorDiedError")
    c = _collector()

    class _Dead:
        def result(self):
            raise ray_exc.ActorDiedError()

    c._teacher_future = _Dead()
    c._teacher_inflight_rows = 128
    c._teacher_inflight_start = 0.0

    with pytest.raises(RuntimeError, match="no longer usable"):
        c._join_teacher_prefetch()


def test_the_worker_side_types_resolve_and_exclude_plain_errors():
    """The discriminator is the whole fix, so pin what it does and does not match."""
    types = RL._worker_side_errors()
    assert types, "ray is installed here, so at least RayTaskError must resolve"
    assert all(issubclass(t, BaseException) for t in types)
    assert not isinstance(RuntimeError("driver-side"), types)
    assert not isinstance(torch.OutOfMemoryError("bare, not from a worker"), types)


def test_a_successful_chunk_updates_the_rate_and_merges_results():
    c = _collector()

    class _Done:
        def result(self):
            return {("t-0", 0): "scored"}

    c._teacher_future = _Done()
    c._teacher_inflight_rows = 100
    c._teacher_inflight_start = RL._now() - 0.5  # ~200 rows/s

    c._join_teacher_prefetch()

    assert c._prefetched_teacher == {("t-0", 0): "scored"}
    assert c._teacher_rate is not None and c._teacher_rate > 0
    assert c._teacher_inflight_rows == 0


def test_the_rate_is_smoothed_across_chunks():
    """One stalled turn (a retriever hiccup inside the glue) must not swing the
    next chunk to an extreme."""
    c = _collector()
    c._teacher_rate = 200.0

    class _Done:
        def result(self):
            return {}

    c._teacher_future = _Done()
    c._teacher_inflight_rows = 100
    c._teacher_inflight_start = RL._now() - 10.0  # 10 rows/s, a big outlier
    c._join_teacher_prefetch()

    assert 10.0 < c._teacher_rate < 200.0, "a single outlier must be damped, not adopted"


def test_join_is_a_noop_with_nothing_in_flight():
    assert _collector()._join_teacher_prefetch() == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all adaptive-chunk tests passed")
