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
"""Reporting the per-task metric without asking the device which tasks are here.

``iter_task_row_masks`` calls ``torch.unique(task_ids).tolist()`` to yield only
the tasks a micro-batch actually holds. That ``.tolist()`` is a device read, so
it is a host sync, once per micro-batch -- and on the pure-OPD arms the loop it
guards computes exactly one metric, which is deferred and syncs nothing.

Walking all three task names instead costs two extra masked means and no sync,
but it means the loop body now runs on masks that select nothing. The number
that comes out the other end must not change, and that is what these hold: an
absent task has to contribute nothing rather than a NaN, and the average has to
stay the average over the micro-batches where the task WAS present.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.trainer.ppo.metric_utils import iter_task_row_masks

TASKS = ["alfworld", "search", "webshop"]


def _token_mean(loss_mat, mask):
    """verl_F.masked_mean, which is what agg_loss(token-mean) calls."""
    return (loss_mat * mask).sum() / (mask.sum() + 1e-8)


def _defer_task_value(loss_mat, mask):
    """dp_actor._defer_task, reproduced: the value and its presence weight."""
    den = mask.sum()
    value = (loss_mat * mask).sum() / den.clamp(min=1)
    return value, (den > 0).to(value.dtype)


def test_including_absent_tasks_yields_every_name():
    ids = torch.tensor([0, 0, 2])          # search absent
    present = [t for t, _ in iter_task_row_masks(ids, TASKS)]
    everything = [t for t, _ in iter_task_row_masks(ids, TASKS, include_absent=True)]
    assert present == ["alfworld", "webshop"]
    assert everything == TASKS


def test_the_masks_agree_wherever_both_paths_yield_a_task():
    ids = torch.tensor([0, 0, 2, 1])
    a = dict(iter_task_row_masks(ids, TASKS))
    b = dict(iter_task_row_masks(ids, TASKS, include_absent=True))
    for task, mask in a.items():
        assert torch.equal(mask, b[task]), task


def test_an_absent_task_gets_an_all_false_mask_not_a_missing_entry():
    ids = torch.tensor([0, 0])
    masks = dict(iter_task_row_masks(ids, TASKS, include_absent=True))
    assert masks["search"].sum() == 0
    assert masks["search"].shape == ids.shape


def test_rows_with_no_task_are_still_excluded():
    """-1 is the id for a row whose task name was missing. It must not become a
    fourth task, and it must not be swept into task 0."""
    ids = torch.tensor([-1, 0, -1])
    masks = dict(iter_task_row_masks(ids, TASKS, include_absent=True))
    assert list(masks) == TASKS
    assert masks["alfworld"].tolist() == [False, True, False]


def test_the_reported_number_is_unchanged():
    """The whole point. Four micro-batches, one of which has no search rows;
    both paths must land on the same mean."""
    torch.manual_seed(0)
    rows, seq = 4, 6
    old_vals, new_vals, new_weights = [], [], []

    for micro in range(4):
        loss = torch.randn(rows, seq).abs()
        mask = torch.ones(rows, seq)
        mask[:, -2:] = 0                         # right padding
        ids = torch.tensor([0, 0, 1, 2]) if micro != 2 else torch.tensor([0, 0, 0, 2])

        old = dict(iter_task_row_masks(ids, TASKS))
        if "search" in old:
            old_vals.append(_token_mean(loss[old["search"]], mask[old["search"]]))

        new = dict(iter_task_row_masks(ids, TASKS, include_absent=True))
        v, w = _defer_task_value(loss[new["search"]], mask[new["search"]])
        new_vals.append(v)
        new_weights.append(w)

    assert len(old_vals) == 3, "search should be absent from exactly one micro-batch"
    old_mean = torch.stack(old_vals).mean()
    new_mean = torch.stack(new_vals).sum() / torch.stack(new_weights).sum().clamp(min=1)
    # Not bit-identical: masked_mean divides by (n + 1e-8) and _defer_task by n.
    assert torch.allclose(old_mean, new_mean, rtol=1e-6, atol=0), (old_mean, new_mean)


def test_an_absent_task_contributes_zero_and_not_nan():
    """0/0 is the failure this replaces. It would reach wandb as NaN and poison
    the whole series, since a NaN in the stack makes the mean NaN too."""
    loss = torch.randn(0, 6)
    mask = torch.zeros(0, 6)
    value, weight = _defer_task_value(loss, mask)
    assert torch.isfinite(value) and value == 0.0
    assert weight == 0.0

    # and the naive form really does produce the NaN
    naive = (loss * mask).sum() / mask.sum()
    assert torch.isnan(naive)


def test_a_task_absent_from_every_micro_batch_reports_zero_not_a_crash():
    """A task that never appears -- an arm run on two tasks with a three-name
    table -- must not divide by zero at the flush."""
    values = torch.zeros(4)
    weights = torch.zeros(4)
    assert torch.isfinite(values.sum() / weights.sum().clamp(min=1))


# --------------------------------------------------------------------------- #
# The gate. _defer_task hard-codes token-mean, so anything that changes what the
# loop computes -- another loss term switched on, or a different aggregation --
# has to switch the fast path back off.
# --------------------------------------------------------------------------- #


def _gate_source():
    import ast
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "verl", "workers", "actor", "dp_actor.py"
    )
    src = open(path).read()
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "sync_free_task_metrics" for t in n.targets)
    )
    return ast.get_source_segment(src, node)


@pytest.mark.parametrize(
    "term",
    ["pg_loss_coef", "entropy_coeff", "use_kl_loss", "use_sdl_loss", "use_sdar_loss"],
)
def test_every_syncing_loss_term_is_in_the_gate(term):
    """Each of these adds a branch that calls .item() INSIDE the loop. On an
    empty mask that is a NaN, not a slow path."""
    assert term in _gate_source()


def test_the_aggregation_mode_is_in_the_gate():
    """The one condition the upstream version does not have. loss_agg_mode is
    not a term that can be switched off, so it cannot be inferred from the
    others -- under seq-mean-* _defer_task would report a different quantity
    under the same name, and nothing would say so."""
    assert 'loss_agg_mode == "token-mean"' in _gate_source()


# --------------------------------------------------------------------------- #
# The hard-label CE term, which this branch has and the branch this came from
# does not.
# --------------------------------------------------------------------------- #


def test_the_ce_per_task_metric_is_weight_aware_too():
    """The CE is the one metric in the loop that sync_free_task_metrics leaves on.

    include_absent=True runs the loop for tasks with no rows in this micro-batch,
    so every metric it computes has to survive an empty mask. Everything else is
    switched off by the condition itself; the CE is not, because it is on in this
    branch's shipped arm and excluding it would retire the mechanism from the only
    run that uses it. A plain agg_loss there is 0/0.
    """
    import inspect

    from verl.workers.actor import dp_actor

    src = inspect.getsource(dp_actor)
    # The per-task block, not the loss body: both mention use_sft_loss, and only
    # the one keyed by task name runs under include_absent.
    anchor = src.index('f"actor/sft_loss/{task}"')
    ce = src[src.rindex('if self.config.get("use_sft_loss", False):', 0, anchor) : ]
    ce = ce[: ce.index("append_to_dict(metrics, task_metrics)")]
    assert "if sync_free_task_metrics:" in ce
    assert "_defer_task(" in ce
    assert ce.index("_defer_task(") < ce.index("agg_loss(")


def test_an_absent_task_contributes_zero_weight_not_nan():
    """The arithmetic behind that, on the helper's own formula."""
    loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    absent = torch.zeros((2, 2))
    den = absent.sum()
    value = (loss_mat * absent).sum() / den.clamp(min=1)
    weight = (den > 0).to(value.dtype)
    assert torch.isfinite(value) and float(weight) == 0.0
    # and the plain token-mean it replaces is what would have been logged
    assert torch.isnan((loss_mat * absent).sum() / absent.sum())
