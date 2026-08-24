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
"""Round-robining the validation batch by task, and the one thing it must not do.

The generation batch reaches the workers as contiguous per-rank chunks, and the
evaluation set is stored grouped by task -- so rank 0 gets alfworld, rank 1 gets
search, rank 2 gets webshop. A 2-second nvidia-smi sample during generation on
the 2026-08-24 evaluation reads ``87 / 0 / 0`` and ``100 / 31 / 53``: two cards
idle while the third works, which is the batch layout rather than a stall. It
degrades as the rollout runs, because search ends in a few turns and alfworld
runs to fifty, so the surviving rows sit on one rank and the other two get
nothing at all.

The invariant that makes the fix safe, and that these tests exist for: **rows
keep their order within a task**. alfworld draws its episodes from TextWorld's
seeded game-file cycle by position, so row k of alfworld must stay row k of
alfworld or the checkpoint gets scored on different games -- the exact failure
the per-checkpoint-process design was built to prevent. Round-robin preserves
it; anything that shuffles within a task does not, and would fail silently.
"""

import numpy as np
import pytest

from verl.utils.task_interleave import interleave_enabled, interleaved_order


def _tasks(**counts):
    """[('a', n), ...] laid out the way the evaluation set is: task by task."""
    rows = []
    for task, n in counts.items():
        rows.extend([task] * n)
    return rows


def test_three_equal_tasks_round_robin():
    assert interleaved_order(_tasks(a=3, b=3, c=3)) == [0, 3, 6, 1, 4, 7, 2, 5, 8]


def test_every_row_appears_exactly_once():
    """It is a permutation or it is data loss."""
    tasks = _tasks(alfworld=42, search=42, webshop=42)
    order = interleaved_order(tasks)

    assert sorted(order) == list(range(len(tasks)))


def test_within_task_order_is_preserved():
    """The seeding invariant. alfworld's row k plays game k; a layout that moves
    row k relative to its own task scores a different episode set."""
    tasks = _tasks(alfworld=17, search=9, webshop=23)
    order = interleaved_order(tasks)

    for task in ("alfworld", "search", "webshop"):
        original = [i for i, t in enumerate(tasks) if t == task]
        after = [i for i in order if tasks[i] == task]
        assert after == original, f"{task} rows were reordered among themselves"


def test_unequal_counts_leave_the_remainder_in_order():
    """A task that runs out drops out of the rotation; what is left is the tail,
    still in its own order."""
    order = interleaved_order(_tasks(a=4, b=2))

    assert order == [0, 4, 1, 5, 2, 3]


def test_a_single_task_is_a_no_op():
    """Nothing to mix -- and returning a permutation anyway would reorder rows
    for no benefit, which for alfworld means different games."""
    assert interleaved_order(["alfworld"] * 6) == list(range(6))


def test_empty_batch():
    assert interleaved_order([]) == []


def test_numpy_object_names_group_together():
    """The column arrives as numpy object scalars; three spellings of one task
    would each take their own slot in the rotation and balance nothing."""
    tasks = np.array(["a", "a", "b", "b"], dtype=object)

    assert interleaved_order(tasks) == [0, 2, 1, 3]


def test_bytes_and_str_names_are_the_same_task():
    assert interleaved_order([b"a", "a", "b", "b"]) == [0, 2, 1, 3]


@pytest.mark.parametrize("ranks", [2, 3, 6])
def test_the_dp_chunks_actually_end_up_balanced(ranks):
    """The point of the whole exercise: contiguous chunks of the interleaved
    order -- which is how the batch is split across ranks -- must each hold a
    near-equal task mix. Before the fix, each chunk held one task."""
    tasks = _tasks(alfworld=42, search=42, webshop=42)
    order = interleaved_order(tasks)
    per_rank = len(order) // ranks

    for rank in range(ranks):
        chunk = order[rank * per_rank : (rank + 1) * per_rank]
        counts = {t: sum(1 for i in chunk if tasks[i] == t) for t in ("alfworld", "search", "webshop")}
        assert max(counts.values()) - min(counts.values()) <= 1, f"rank {rank} got {counts}"


def test_the_unbalanced_layout_is_what_we_are_fixing():
    """The control: the stored layout really does hand each rank one task, so
    the test above is measuring a change and not a tautology."""
    tasks = _tasks(alfworld=42, search=42, webshop=42)
    stored = list(range(len(tasks)))
    per_rank = len(stored) // 3

    for rank in range(3):
        chunk = stored[rank * per_rank : (rank + 1) * per_rank]
        assert len({tasks[i] for i in chunk}) == 1


def test_the_flag_is_off_by_default(monkeypatch):
    """A scoring path must not change what it reports because someone pulled."""
    monkeypatch.delenv("VAL_TASK_INTERLEAVE", raising=False)
    assert interleave_enabled() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_the_flag_parses_like_the_other_switches(monkeypatch, value, expected):
    monkeypatch.setenv("VAL_TASK_INTERLEAVE", value)
    assert interleave_enabled() is expected
