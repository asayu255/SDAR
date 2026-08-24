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
"""Whether the log can say, by itself, that two batches overlapped.

The per-batch turn table cannot answer that. Pipelining does not change what a
batch costs -- ``SHARE gen(GPU-busy)`` reads the same at depth 1 and depth 2 --
it changes only when the second batch runs. So the evidence has to be a pair of
numbers that only concurrency can separate: the sum of the batches' own spans,
and the wall clock from the first batch's start to this one's end. Serial
running holds them equal; every overlapped second pushes the sum above the wall.

These tests pin the arithmetic and the reset, because the ratio is the number
the pipeline's worth is judged on and a silently accumulating counter would read
as a speedup that never happened.
"""

import os
import re
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rollout_loop  # noqa: E402


@pytest.fixture(autouse=True)
def fresh():
    rollout_loop.reset_batch_wall()
    yield
    rollout_loop.reset_batch_wall()


def ratio(line):
    return float(re.search(r"serial/wall=([0-9.]+)x", line).group(1))


def field(line, name):
    return float(re.search(rf"{name}=([0-9.]+)s", line).group(1))


def test_one_batch_is_exactly_serial():
    line = rollout_loop._record_batch_wall(100.0, 110.0, "primary")
    assert "batch#0" in line
    assert field(line, "span") == 10.0
    assert field(line, "wall-since-first-batch") == 10.0
    assert ratio(line) == 1.0


def test_back_to_back_batches_stay_at_one():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(10.0, 20.0, "primary")
    assert "batch#1" in line
    assert field(line, "sum-of-spans") == 20.0
    assert field(line, "wall-since-first-batch") == 20.0
    assert ratio(line) == 1.0


def test_a_gap_between_batches_drops_below_one():
    # the shape of an un-pipelined run whose batches are separated by
    # preparation on the calling thread: wall grows, the spans do not.
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(15.0, 25.0, "primary")
    assert ratio(line) == pytest.approx(0.8)


def test_fully_overlapped_batches_read_two():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(0.0, 10.0, "extra-1")
    assert field(line, "sum-of-spans") == 20.0
    assert field(line, "wall-since-first-batch") == 10.0
    assert ratio(line) == 2.0


def test_half_overlapped_batches_read_between():
    # what depth 2 should actually look like on search: the second batch starts
    # while the first is still stepping its environment.
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(5.0, 15.0, "extra-1")
    assert ratio(line) == pytest.approx(20.0 / 15.0, abs=0.005)


def test_wall_is_measured_from_the_first_batch_not_this_one():
    rollout_loop._record_batch_wall(100.0, 110.0, "primary")
    line = rollout_loop._record_batch_wall(110.0, 130.0, "primary")
    assert field(line, "wall-since-first-batch") == 30.0


def test_slot_name_is_carried_through():
    line = rollout_loop._record_batch_wall(0.0, 1.0, "extra-1")
    assert "slot=extra-1" in line


def test_reset_starts_a_new_accounting_period():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    rollout_loop.reset_batch_wall()
    line = rollout_loop._record_batch_wall(1000.0, 1010.0, "primary")
    assert "batch#0" in line
    assert field(line, "sum-of-spans") == 10.0
    assert field(line, "wall-since-first-batch") == 10.0


def test_batches_are_numbered_in_completion_order():
    lines = [rollout_loop._record_batch_wall(float(i), float(i) + 1, "primary") for i in range(5)]
    assert [re.search(r"batch#(\d+)", line).group(1) for line in lines] == ["0", "1", "2", "3", "4"]


def test_zero_length_period_does_not_divide_by_zero():
    line = rollout_loop._record_batch_wall(5.0, 5.0, "primary")
    assert "serial/wall=nan" in line


def test_concurrent_records_do_not_lose_a_batch():
    # the counter is touched from the pipeline's worker threads.
    barrier = threading.Barrier(8)
    lines = []
    lock = threading.Lock()

    def one(i):
        barrier.wait()
        line = rollout_loop._record_batch_wall(0.0, 1.0, f"slot-{i}")
        with lock:
            lines.append(line)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(int(re.search(r"batch#(\d+)", line).group(1)) for line in lines) == list(range(8))
    assert max(field(line, "sum-of-spans") for line in lines) == 8.0


def _records(n):
    return [
        {
            "turn": i,
            "active": 4,
            "preproc": 0.1,
            "gen": 1.0,
            "decode": 0.1,
            "envstep": 0.2,
            "gen_util": 99.0,
            "gen_util_per_gpu": [99.0, 99.0],
        }
        for i in range(n)
    ]


def test_the_table_gains_a_wall_line_when_a_span_is_given(capsys):
    rollout_loop._print_turn_timing(_records(3), span=(0.0, 12.0), slot="extra-1")
    out = capsys.readouterr().out
    assert "WALL   slot=extra-1  batch#0" in out
    assert "span=12.0s" in out


def test_the_table_is_unchanged_without_a_span(capsys):
    rollout_loop._print_turn_timing(_records(3))
    out = capsys.readouterr().out
    assert "SHARE" in out
    assert "WALL" not in out


def test_an_empty_batch_records_nothing(capsys):
    rollout_loop._print_turn_timing([], span=(0.0, 12.0), slot="primary")
    assert capsys.readouterr().out == ""
    # and the period is still empty, so the next real batch is #0
    assert "batch#0" in rollout_loop._record_batch_wall(0.0, 1.0, "primary")


def test_the_slot_label_is_per_thread():
    """Two pipeline slots run on two threads; neither may read the other's name."""
    seen = {}
    started = threading.Barrier(2)

    def one(name):
        with rollout_loop.slot_label(name):
            started.wait()
            seen[name] = rollout_loop._current_slot()

    threads = [threading.Thread(target=one, args=(n,)) for n in ("primary", "extra-1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"primary": "primary", "extra-1": "extra-1"}


def test_the_slot_label_is_restored_on_the_way_out():
    assert rollout_loop._current_slot() == "-"
    with rollout_loop.slot_label("primary"):
        assert rollout_loop._current_slot() == "primary"
        with rollout_loop.slot_label("extra-1"):
            assert rollout_loop._current_slot() == "extra-1"
        assert rollout_loop._current_slot() == "primary"
    assert rollout_loop._current_slot() == "-"


def test_the_slot_label_is_restored_after_a_failed_rollout():
    with pytest.raises(RuntimeError):
        with rollout_loop.slot_label("extra-1"):
            raise RuntimeError("environment died")
    assert rollout_loop._current_slot() == "-"


def test_an_unlabelled_rollout_still_prints(capsys):
    """Training rollouts do not go through the pipeline and have no slot."""
    rollout_loop._print_turn_timing(_records(2), span=(0.0, 5.0), slot=rollout_loop._current_slot())
    assert "WALL   slot=-  batch#0" in capsys.readouterr().out
