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
"""The per-task metric loop stops asking the device which tasks it holds.

``iter_task_row_masks`` used ``torch.unique(task_ids).tolist()``, a device read
and therefore a host sync, once per micro-batch -- 284 of them a step on this
arm. The comment beside the loop said "no host sync happens here"; the sync was
at the top of it.

Walking the task names instead costs nothing to the host but yields tasks with no
rows, whose masked mean is 0/0. The deferred metrics carry a presence weight so
the value they report is the one the old path reported: the mean over the
micro-batches the task actually appeared in.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.metric_utils import iter_task_row_masks
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

NAMES = ["alfworld", "search", "webshop"]


class _NoReadTensor(torch.Tensor):
    """A tensor that fails if anything moves its data to the host.

    ``tolist`` / ``item`` / ``numpy`` are what a sync looks like from Python;
    every one of them would be a ``cudaStreamSynchronize`` on a real device.
    """

    @staticmethod
    def make(data):
        return torch.tensor(data).as_subclass(_NoReadTensor)

    def tolist(self, *a, **k):
        raise AssertionError("tolist() on task_ids: that is the per-micro-batch host sync")

    def item(self, *a, **k):
        raise AssertionError("item() on task_ids: that is the per-micro-batch host sync")

    def numpy(self, *a, **k):
        raise AssertionError("numpy() on task_ids: that is the per-micro-batch host sync")


def test_walking_the_names_reads_nothing_back_from_the_device():
    task_ids = _NoReadTensor.make([0, 0, 2, 1])

    got = list(iter_task_row_masks(task_ids, NAMES, include_absent=True))

    assert [name for name, _ in got] == NAMES


def test_the_default_still_reads_and_still_skips_absent_tasks():
    """dp_critic and any arm whose loop body calls .item() keep the old path."""
    task_ids = torch.tensor([0, 0, 2, 2])

    got = list(iter_task_row_masks(task_ids, NAMES))

    assert [name for name, _ in got] == ["alfworld", "webshop"]


def test_absent_tasks_come_out_with_an_empty_mask():
    task_ids = torch.tensor([0, 0, 2, 2])

    masks = dict(iter_task_row_masks(task_ids, NAMES, include_absent=True))

    assert masks["search"].sum() == 0
    assert masks["alfworld"].tolist() == [True, True, False, False]
    assert masks["webshop"].tolist() == [False, False, True, True]


def test_ids_outside_the_name_table_are_in_no_task():
    """-1 marks a row whose task name was missing; it must not join a task."""
    task_ids = torch.tensor([0, -1, 2])

    masks = dict(iter_task_row_masks(task_ids, NAMES, include_absent=True))

    assert sum(int(m.sum()) for m in masks.values()) == 2


def _old_value(per_micro):
    """What the previous code reported: mean over the micro-batches holding the task."""
    present = [v for v in per_micro if v is not None]
    return float(torch.stack(present).mean())


def _new_value(per_micro):
    """What the deferred pair reports: sum of values over count of presences,
    with an absent micro-batch contributing (0, 0)."""
    values, weights = [], []
    for v in per_micro:
        values.append(torch.tensor(0.0) if v is None else v)
        weights.append(torch.tensor(0.0 if v is None else 1.0))
    return float(torch.stack(values).sum() / torch.stack(weights).sum().clamp(min=1))


@pytest.mark.parametrize("per_micro", [
    [torch.tensor(2.0), torch.tensor(4.0)],                    # present throughout
    [torch.tensor(2.0), None, torch.tensor(4.0)],              # absent in the middle
    [None, torch.tensor(3.0)],                                 # absent first
    [torch.tensor(1.5), None, None, torch.tensor(2.5), None],  # mostly absent
])
def test_the_presence_weight_reproduces_the_old_average(per_micro):
    assert _new_value(per_micro) == pytest.approx(_old_value(per_micro))


def test_a_task_absent_from_every_micro_batch_reports_zero_not_nan():
    """The old path never emitted the key at all. Zero is what the new one gives;
    what must not happen is a NaN poisoning the logged series."""
    value = _new_value([None, None])

    assert value == 0.0
    assert not np.isnan(value)


def test_the_masked_mean_matches_agg_loss_where_the_task_is_present():
    """_defer_task writes the token-mean out by hand so an empty mask gives 0
    rather than 0/0; where the mask is not empty it has to be the same number."""
    from verl.trainer.ppo.core_algos import agg_loss

    loss_mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    den = mask.sum()
    by_hand = (loss_mat * mask).sum() / den.clamp(min=1)

    assert by_hand == pytest.approx(float(agg_loss(loss_mat, mask, "token-mean")))
