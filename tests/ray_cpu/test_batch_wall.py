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
"""Seconds of wall per batch, which is what a pipeline is judged on.

The per-batch turn table cannot answer it: pipelining does not change what a
batch costs -- ``SHARE gen(GPU-busy)`` reads the same at depth 1 and depth 2 --
so the evidence has to be wall clock across batches.

The occupancy ratio reported alongside it is NOT a speedup, and these tests pin
that distinction because it was read as one. Under pipelining a batch's own span
inflates: the generate call it sits in is queued behind another batch's. Two
slots each reporting a doubled span read 2.00x with nothing gained -- measured
1.82x on a run whose s/batch moved by 1.5%.
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


def busy(line):
    return float(re.search(r"slots-busy=([0-9.]+)x", line).group(1))


def field(line, name):
    return float(re.search(rf"{name}=([0-9.]+)s", line).group(1))


def per_batch(line, scope="all"):
    return float(re.search(rf"{scope}=([0-9.]+)s", line).group(1))


def test_one_batch_is_exactly_serial():
    line = rollout_loop._record_batch_wall(100.0, 110.0, "primary")
    assert "batch#0" in line
    assert field(line, "span") == 10.0
    assert field(line, "wall") == 10.0
    assert per_batch(line) == 10.0
    assert busy(line) == 1.0


def test_back_to_back_batches_stay_at_one():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(10.0, 20.0, "primary")
    assert "batch#1" in line
    assert field(line, "wall") == 20.0
    assert per_batch(line) == 10.0
    assert busy(line) == 1.0


def test_a_gap_between_batches_drops_below_one():
    # the shape of an un-pipelined run whose batches are separated by
    # preparation on the calling thread: wall grows, the spans do not.
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(15.0, 25.0, "primary")
    assert busy(line) == pytest.approx(0.8)
    assert per_batch(line) == pytest.approx(12.5)


def test_fully_overlapped_batches_read_two():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(0.0, 10.0, "extra-1")
    assert field(line, "wall") == 10.0
    assert busy(line) == 2.0
    # and the figure that actually matters halved, which the ratio does not say
    assert per_batch(line) == 5.0


def test_half_overlapped_batches_read_between():
    # what depth 2 should actually look like on search: the second batch starts
    # while the first is still stepping its environment.
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    line = rollout_loop._record_batch_wall(5.0, 15.0, "extra-1")
    assert busy(line) == pytest.approx(20.0 / 15.0, abs=0.005)
    assert per_batch(line) == pytest.approx(7.5)


def test_wall_is_measured_from_the_first_batch_not_this_one():
    rollout_loop._record_batch_wall(100.0, 110.0, "primary")
    line = rollout_loop._record_batch_wall(110.0, 130.0, "primary")
    assert field(line, "wall") == 30.0


def test_slot_name_is_carried_through():
    line = rollout_loop._record_batch_wall(0.0, 1.0, "extra-1")
    assert "slot=extra-1" in line


def test_reset_starts_a_new_accounting_period():
    rollout_loop._record_batch_wall(0.0, 10.0, "primary")
    rollout_loop.reset_batch_wall()
    line = rollout_loop._record_batch_wall(1000.0, 1010.0, "primary")
    assert "batch#0" in line
    assert field(line, "wall") == 10.0
    assert per_batch(line) == 10.0


def test_batches_are_numbered_in_completion_order():
    lines = [rollout_loop._record_batch_wall(float(i), float(i) + 1, "primary") for i in range(5)]
    assert [re.search(r"batch#(\d+)", line).group(1) for line in lines] == ["0", "1", "2", "3", "4"]


def test_zero_length_period_does_not_divide_by_zero():
    line = rollout_loop._record_batch_wall(5.0, 5.0, "primary")
    assert "slots-busy=nan" in line


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


def test_the_legend_is_printed_once_at_the_start():
    first = rollout_loop._record_batch_wall(0.0, 1.0, "primary")
    second = rollout_loop._record_batch_wall(1.0, 2.0, "primary")
    assert "legend" in first and "OCCUPANCY, not speedup" in first
    assert "legend" not in second


def test_the_window_has_no_rate_until_there_are_two_batches():
    assert "last20=nans" in rollout_loop._record_batch_wall(0.0, 10.0, "primary")


def test_the_trailing_window_drops_the_expensive_prefix():
    """alfworld and webshop are the first two batches and cost multiples of a
    search batch, so a cumulative figure over 413 would carry them forever."""
    rollout_loop._record_batch_wall(0.0, 231.0, "primary")  # alfworld
    line = None
    t = 231.0
    for _ in range(25):  # search batches, 14s each
        line = rollout_loop._record_batch_wall(t, t + 14.0, "primary")
        t += 14.0

    assert per_batch(line, "last20") == pytest.approx(14.0)
    assert per_batch(line) > 20.0  # the cumulative figure still carries alfworld


def test_the_window_tracks_a_change_in_rate():
    t = 0.0
    for _ in range(20):
        rollout_loop._record_batch_wall(t, t + 24.0, "primary")
        t += 24.0
    slow = rollout_loop._record_batch_wall(t, t + 24.0, "primary")
    t += 24.0
    for _ in range(20):  # the retriever is fixed; batches get faster
        fast = rollout_loop._record_batch_wall(t, t + 14.0, "primary")
        t += 14.0

    assert per_batch(slow, "last20") == pytest.approx(24.0)
    assert per_batch(fast, "last20") == pytest.approx(14.0)


def test_occupancy_of_two_slots_says_nothing_about_speedup():
    """The reading that has to be impossible: 2.00x with no gain at all.

    Two slots whose spans doubled because each is queued behind the other run
    exactly as fast as one slot at the original span, and the occupancy ratio
    reads 2.00x for both.
    """
    solo = [rollout_loop._record_batch_wall(float(i) * 14, float(i) * 14 + 14, "primary") for i in range(6)][-1]
    solo_rate = per_batch(solo, "last20")

    rollout_loop.reset_batch_wall()
    t = 0.0
    for _ in range(10):  # two slots, each batch taking twice as long
        rollout_loop._record_batch_wall(t, t + 28.0, "primary")
        pipelined = rollout_loop._record_batch_wall(t, t + 28.0, "extra-1")
        t += 28.0

    assert busy(pipelined) == pytest.approx(2.0, abs=0.01)
    assert per_batch(pipelined, "last20") == pytest.approx(solo_rate, abs=1.0)


class _Tensor:
    def __init__(self, total):
        self._total = total

    def sum(self):
        return self._total


class _Batch:
    def __init__(self, total):
        self.batch = {"attention_mask": _Tensor(total)}


def test_token_counts_split_prompt_from_generated():
    prompt, generated = rollout_loop._token_counts(_Batch(48_300), _Batch(49_120))
    assert (prompt, generated) == (48_300, 820)


def test_token_counts_never_go_negative():
    """The output mask should cover the prompt, but a truncating rollout could
    hand back less; a negative token count would be worse than a zero."""
    assert rollout_loop._token_counts(_Batch(100), _Batch(80)) == (100, 0)


def test_token_counts_are_absent_rather_than_wrong():
    class _NoMask:
        batch = {}

    assert rollout_loop._token_counts(_NoMask(), _NoMask()) == (None, None)
    assert rollout_loop._token_counts(None, None) == (None, None)


def test_the_table_carries_the_token_columns(capsys):
    records = [
        {
            "turn": 0,
            "active": 126,
            "preproc": 0.2,
            "gen": 1.0,
            "decode": 0.0,
            "envstep": 0.4,
            "gen_util": 72.0,
            "gen_util_per_gpu": [72.0, 72.0, 71.0],
            "prompt_tok": 48_300,
            "gen_tok": 820,
        },
        {
            "turn": 1,
            "active": 126,
            "preproc": 0.2,
            "gen": 5.4,
            "decode": 0.0,
            "envstep": 0.3,
            "gen_util": 79.0,
            "gen_util_per_gpu": [79.0, 79.0, 79.0],
            "prompt_tok": 61_000,
            "gen_tok": 43_000,
        },
    ]
    rollout_loop._print_turn_timing(records)
    out = capsys.readouterr().out

    assert "promptTok" in out and "genTok" in out
    assert "48,300" in out and "43,000" in out
    assert "TOKENS prompt=109,300" in out
    assert "generated=43,820" in out
    assert "UPPER bound" in out  # prefix-cache hits are not recomputed


def test_a_table_without_token_counts_is_unchanged(capsys):
    rollout_loop._print_turn_timing(_records(2))
    out = capsys.readouterr().out
    assert "TOKENS" not in out
    assert "SHARE" in out
