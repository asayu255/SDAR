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
"""Validation batches that hold one task, at a size chosen per task.

Today's single-task validation batches are an alignment, not a rule: the loader
slices the task-sorted parquet every ``val_batch_size`` rows, and alfworld and
webshop happen to hold exactly that many. Raise the size and the first batch
becomes alfworld's 126 rows plus webshop's -- a mixed batch, which
``_validation_task_name`` refuses.

The size wants to differ by task. Search's episodes end at different turns, so a
batch's later turns generate for a handful of trajectories in a slot sized for
all of them: measured, the last two turns of a search batch take 46% of its
generation time to carry 14% of its work. A decode step costs what it costs
whether it carries ten sequences or a hundred, so those turns are nearly free to
widen -- while alfworld must stay at 126, its games being indexed by position
within its environment manager.

So: group by task, chunk each group by that task's own size. Order is the file's
throughout, and with every task at the same size the batches are exactly the
ones the plain loader produced.
"""

from typing import Dict, Iterator, List, Optional, Sequence


def task_batches(task_names: Sequence, sizes: Dict[str, int], default: int) -> List[List[int]]:
    """Index lists, one per batch: runs of one task, chunked by that task's size.

    A task appearing in two separate runs of the file gets separate batches --
    the runs are not merged, because merging would reorder rows.
    """
    if default <= 0:
        raise ValueError(f"default validation batch size must be positive, got {default}")
    for task, size in sizes.items():
        if size <= 0:
            raise ValueError(f"validation batch size for {task!r} must be positive, got {size}")

    batches: List[List[int]] = []
    run_task = _MISSING = object()
    run: List[int] = []

    def flush():
        if not run:
            return
        size = sizes.get(run_task, default)
        for start in range(0, len(run), size):
            batches.append(run[start : start + size])

    for index, name in enumerate(task_names):
        if name != run_task:
            flush()
            run_task, run = name, []
        run.append(index)
    flush()
    return batches


class TaskBatchSampler:
    """A ``batch_sampler`` for the validation loader. Deterministic, stateless."""

    def __init__(self, task_names: Sequence, sizes: Dict[str, int], default: int):
        self._batches = task_batches(task_names, sizes, default)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def task_names_of(dataset, key: str = "task_name") -> Optional[List]:
    """The task of every row, in file order, or None if the rows do not carry one.

    Read off the underlying table rather than through ``__getitem__``: that path
    tokenises and builds tensors for the row, which for 52,000 validation rows
    is minutes of work to answer a question the column already answers.
    """
    frame = getattr(dataset, "dataframe", None)
    if frame is None:
        return None
    columns = getattr(frame, "column_names", None)
    if columns is None or key not in columns:
        return None
    return list(frame[key])
