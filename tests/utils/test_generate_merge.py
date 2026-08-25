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
"""Merging generate calls that are already queued, and never waiting for one.

The seats a search batch's last turns leave empty -- twelve trajectories over
three ranks, per-GPU occupancy 61/89/56 -- can only be filled from another
batch. The pipeline has one, but runs it deliberately out of phase, so waiting
for it would trade away the 16.5% the staggering bought. Hence: merge only what
is queued at this instant, and if nothing is, go alone immediately.

What must hold above all is that a merge is invisible downstream. Rows come
back to their own caller, in their own order, and a caller whose key differs
never shares a call with one whose key does not.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from verl.utils.generate_merge import GenerateMerger  # noqa: E402


def _merger():
    def concat(batches):
        return [row for batch in batches for row in batch]

    def split(output, sizes):
        parts, start = [], 0
        for size in sizes:
            parts.append(output[start : start + size])
            start += size
        return parts

    return GenerateMerger(concat=concat, split=split)


def _issue(batch):
    """The worker group: answers each row, and cannot run twice at once."""
    return [f"out({row})" for row in batch]


def test_a_lone_caller_goes_straight_through():
    merger = _merger()
    assert merger.call("k", ["a", "b"], 2, _issue) == ["out(a)", "out(b)"]
    assert merger.merges == 0
    assert merger.calls == 1


def test_a_lone_caller_does_not_wait():
    """Waiting for a second batch would trade away what staggering bought."""
    merger = _merger()
    started = time.perf_counter()
    merger.call("k", ["a"], 1, _issue)
    assert time.perf_counter() - started < 0.05


def test_a_queued_caller_is_merged_and_gets_its_own_rows_back():
    merger = _merger()
    entered = threading.Event()
    release = threading.Event()

    def slow_issue(batch):
        entered.set()
        release.wait(5)
        return _issue(batch)

    results = {}

    def first():
        results["first"] = merger.call("k", ["a", "b"], 2, slow_issue)

    def second():
        results["second"] = merger.call("k", ["c"], 1, _issue)

    t1 = threading.Thread(target=first)
    t1.start()
    entered.wait(5)  # the first call is in flight
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.05)  # the second is queued behind it
    release.set()
    t1.join(5)
    t2.join(5)

    assert results["first"] == ["out(a)", "out(b)"]
    assert results["second"] == ["out(c)"]


def test_different_keys_never_share_a_call():
    """Two tasks' sampling parameters differ; merging them would generate one
    task's rows under the other's temperature."""
    merger = _merger()
    seen = []

    def recording_issue(batch):
        seen.append(list(batch))
        return _issue(batch)

    entered = threading.Event()
    release = threading.Event()

    def slow_issue(batch):
        entered.set()
        release.wait(5)
        return recording_issue(batch)

    t1 = threading.Thread(target=lambda: merger.call("alfworld", ["a"], 1, slow_issue))
    t1.start()
    entered.wait(5)
    out = {}
    t2 = threading.Thread(target=lambda: out.update(r=merger.call("search", ["s"], 1, recording_issue)))
    t2.start()
    release.set()
    t1.join(5)
    t2.join(5)

    assert out["r"] == ["out(s)"]
    assert ["a"] in seen and ["s"] in seen
    for call in seen:
        assert not ({"a"} & set(call) and {"s"} & set(call))


def test_every_row_comes_back_once_under_contention():
    merger = _merger()
    barrier = threading.Barrier(8)
    results = {}
    lock = threading.Lock()

    def one(i):
        barrier.wait()
        got = merger.call("k", [f"r{i}"], 1, _issue)
        with lock:
            results[i] = got

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert results == {i: [f"out(r{i})"] for i in range(8)}
    assert merger.calls <= 8  # some of them shared


def test_a_failing_call_raises_for_its_caller():
    merger = _merger()

    def exploding(batch):
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError, match="engine died"):
        merger.call("k", ["a"], 1, exploding)


def test_a_failure_does_not_poison_the_key():
    """The next batch through must not inherit the last one's exception, nor
    find a key still marked as issuing and wait on a leader that has gone."""
    merger = _merger()

    def exploding(batch):
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError):
        merger.call("k", ["a"], 1, exploding)

    assert merger.call("k", ["b"], 1, _issue) == ["out(b)"]


def test_a_caller_queued_behind_a_failure_runs_on_its_own():
    """It was not part of the failed call, so it must not inherit the failure."""
    merger = _merger()
    entered = threading.Event()
    release = threading.Event()
    outcome = {}

    def exploding(batch):
        entered.set()
        release.wait(5)
        raise RuntimeError("engine died")

    def first():
        try:
            merger.call("k", ["a"], 1, exploding)
        except RuntimeError as exc:
            outcome["first"] = str(exc)

    def second():
        outcome["second"] = merger.call("k", ["b"], 1, _issue)

    t1 = threading.Thread(target=first)
    t1.start()
    entered.wait(5)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.05)
    release.set()
    t1.join(5)
    t2.join(5)

    assert outcome["first"] == "engine died"
    assert outcome["second"] == ["out(b)"]


def test_the_counters_start_at_nothing_merged():
    merger = _merger()
    merger.call("k", ["a", "b"], 2, _issue)
    assert (merger.calls, merger.merges, merger.rows_merged) == (1, 0, 0)


def _proto(rows, meta=None, width=8):
    """A DataProto-shaped stand-in for the merge key: tensors and meta_info."""
    import torch

    from verl import DataProto

    return DataProto.from_dict(
        tensors={"input_ids": torch.zeros(rows, width, dtype=torch.long)},
        meta_info=dict(meta or {}),
    )


def test_the_key_separates_different_sampling_parameters():
    """search validates greedily and alfworld at temperature 0.4; one call
    cannot serve both."""
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    greedy = rollout_loop._merge_key(_proto(4, {"do_sample": False, "temperature": 0}))
    sampled = rollout_loop._merge_key(_proto(4, {"do_sample": True, "temperature": 0.4}))
    assert greedy != sampled


def test_the_key_separates_different_tensor_widths():
    """alfworld's prompts are padded to 2048 and search's to 4096; concatenating
    across that would need a re-pad, so they must not meet."""
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    assert rollout_loop._merge_key(_proto(4, width=8)) != rollout_loop._merge_key(_proto(4, width=16))


def test_the_key_ignores_how_many_rows_there_are():
    """Row count is the one thing merging is allowed to differ on."""
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    meta = {"do_sample": False}
    assert rollout_loop._merge_key(_proto(4, meta)) == rollout_loop._merge_key(_proto(12, meta))


def test_splitting_returns_each_caller_its_own_rows_in_order():
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    merged = _proto(6)
    merged.batch["input_ids"][:, 0] = __import__("torch").arange(6)
    parts = rollout_loop._split_by_rows(merged, [2, 3, 1])

    assert [len(p) for p in parts] == [2, 3, 1]
    assert parts[0].batch["input_ids"][:, 0].tolist() == [0, 1]
    assert parts[1].batch["input_ids"][:, 0].tolist() == [2, 3, 4]
    assert parts[2].batch["input_ids"][:, 0].tolist() == [5]


def test_merging_is_off_unless_asked_for():
    """It regroups rows inside a generate call, and this is the scoring path."""
    import agent_system.multi_turn_rollout.rollout_loop as rollout_loop

    assert rollout_loop._ROLLOUT_MERGE_GENERATES is False
