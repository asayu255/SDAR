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
"""One task per validation batch, at a size chosen per task.

A mixed validation batch is not a slow path -- get_task_names refuses it -- and
today nothing enforces single-task batches: the loader slices every
val_batch_size rows and alfworld and webshop happen to hold exactly that many.
Raise one task's size and the first batch becomes alfworld plus webshop.

The size is worth raising for search alone: its episodes end at different turns,
so a batch's last turns decode for a handful of trajectories in a slot sized for
all of them (measured: 46% of a batch's generation time carrying 14% of its
work), and a decode step costs the same whether it carries ten sequences or a
hundred. alfworld must not move -- its games are indexed by position within its
environment manager, so resizing it scores a different set.

The tests that matter here are the two safety ones: a batch never holds two
tasks, and uniform sizes reproduce the old batches exactly.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from verl.utils.val_batching import TaskBatchSampler, task_batches, task_names_of  # noqa: E402

ARM = ["alfworld"] * 126 + ["webshop"] * 126 + ["search"] * 500


def test_a_batch_never_holds_two_tasks():
    for batch in task_batches(ARM, {"search": 252}, 126):
        assert len({ARM[i] for i in batch}) == 1


def test_uniform_sizes_reproduce_the_plain_loader():
    """The old path slices every 126 rows; with all tasks at 126 the alignment
    happens to give the same answer, and it must keep giving it."""
    grouped = task_batches(ARM, {}, 126)
    plain = [list(range(start, min(start + 126, len(ARM)))) for start in range(0, len(ARM), 126)]
    assert grouped == plain


def test_the_named_task_is_the_only_one_resized():
    batches = task_batches(ARM, {"search": 252}, 126)
    assert [len(b) for b in batches] == [126, 126, 252, 248]


def test_every_row_appears_once_and_in_file_order():
    flat = [i for batch in task_batches(ARM, {"search": 252}, 126) for i in batch]
    assert flat == list(range(len(ARM)))


def test_a_short_tail_is_its_own_batch():
    names = ["search"] * 260
    assert [len(b) for b in task_batches(names, {"search": 252}, 126)] == [252, 8]


def test_a_task_appearing_twice_is_not_merged_across_the_gap():
    """Merging separated runs would reorder rows, and scoring joins by position."""
    names = ["search"] * 3 + ["webshop"] * 2 + ["search"] * 3
    assert task_batches(names, {"search": 252}, 126) == [[0, 1, 2], [3, 4], [5, 6, 7]]


def test_an_empty_dataset_yields_no_batches():
    assert task_batches([], {"search": 252}, 126) == []


@pytest.mark.parametrize("sizes,default", [({}, 0), ({}, -1), ({"search": 0}, 126), ({"search": -5}, 126)])
def test_a_non_positive_size_is_refused(sizes, default):
    with pytest.raises(ValueError):
        task_batches(ARM, sizes, default)


def test_the_sampler_is_the_loader_contract():
    sampler = TaskBatchSampler(ARM, {"search": 252}, 126)
    assert len(sampler) == 4
    assert [len(b) for b in sampler] == [126, 126, 252, 248]
    # iterable more than once: the loader iterates it every epoch
    assert [len(b) for b in sampler] == [126, 126, 252, 248]


class _Frame:
    column_names = ["task_name", "prompt"]

    def __getitem__(self, key):
        return {"task_name": ARM, "prompt": ["x"] * len(ARM)}[key]


class _Dataset:
    dataframe = _Frame()


def test_task_names_come_off_the_table_not_through_getitem():
    """__getitem__ tokenises the row; 52,000 of those to read one column is
    minutes of work for an answer the column already holds."""
    assert task_names_of(_Dataset()) == ARM


def test_task_names_are_absent_rather_than_wrong():
    class _NoColumn:
        class dataframe:
            column_names = ["prompt"]

    assert task_names_of(_NoColumn()) is None
    assert task_names_of(object()) is None
