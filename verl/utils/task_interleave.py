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
"""Round-robin a multitask batch by task, so the data-parallel split is mixed.

The generation batch is handed to the workers as contiguous chunks, one per
rank. A batch laid out as [alfworld x42 | search x42 | webshop x42] therefore
gives rank 0 nothing but alfworld, rank 1 nothing but search, rank 2 nothing but
webshop -- and the three tasks neither generate for the same number of tokens
nor finish on the same turn. Measured on the 2026-08-24 evaluation, with a
2-second nvidia-smi sampler running during generation:

    0, 87   1,  0   2,  0
    0, 100  1, 31   2, 53
    0, 87   1, 25   2, 88

Two cards idle while the third works is not a stall to hunt for: it is the batch
layout. And it gets worse as the rollout proceeds, because search episodes end
within a few turns while alfworld runs to fifty -- so the active rows collapse
onto whichever rank holds the surviving task, and the other two ranks are handed
nothing at all.

``TaskBalancedSampler`` already solves this for training (TASK_BALANCE_INTERLEAVE).
It cannot help here: it is a *sampler*, and the validation dataloader takes none
-- validation reads the fixed evaluation set in dataset order, which is grouped
by task.

What this must preserve
-----------------------
**Within-task order.** alfworld draws each episode from TextWorld's seeded
game-file cycle by position (``seed + i // group_n``, i counted within the
alfworld manager). Round-robin keeps every task's rows in their original
relative order, so row k of alfworld is still row k of alfworld and still plays
the same game. A layout that shuffled within a task would silently score the
checkpoint on different episodes -- the one thing the per-checkpoint process
design exists to prevent.

What it does change: which rank generates a given row, hence -- at temperature
above zero -- the specific tokens sampled for it. That is the same accuracy
class as any other batch-composition change (and the active set already changes
composition every turn as trajectories finish). Scores stay comparable across
checkpoints scored the same way, and are not bit-comparable with checkpoints
scored before it was turned on.
"""

import os
from collections import OrderedDict
from typing import List, Sequence


def interleave_enabled() -> bool:
    """Off by default: it moves evaluation numbers by a hair, and a scoring path
    should not change what it reports because someone upgraded."""
    return os.environ.get("VAL_TASK_INTERLEAVE", "0").strip().lower() in ("1", "true", "yes", "on")


def interleaved_order(task_names: Sequence) -> List[int]:
    """Row indices in round-robin task order: alf0, search0, web0, alf1, ...

    Tasks are cycled in first-appearance order, and a task that runs out simply
    drops out of the rotation -- with unequal task counts the tail is whatever
    remains, still in its original order. Returns a permutation of
    ``range(len(task_names))``; the identity when there is nothing to mix.
    """
    buckets = OrderedDict()
    for index, task in enumerate(task_names):
        buckets.setdefault(_key(task), []).append(index)
    if len(buckets) < 2:
        return list(range(len(task_names)))

    order = []
    queues = list(buckets.values())
    position = 0
    while len(order) < len(task_names):
        for queue in queues:
            if position < len(queue):
                order.append(queue[position])
        position += 1
    return order


def _key(task):
    """Group by the task's name as a plain string.

    The column arrives as numpy object scalars, bytes or str depending on how
    the parquet was written, and three spellings of the same task would each get
    their own slot in the rotation -- which is not an error, just a rotation
    that does not balance anything.
    """
    if isinstance(task, bytes):
        return task.decode()
    return str(task)
