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
"""Per-task normalisation of a token-mean distillation loss.

A multitask step mixes trajectories from tasks whose episodes are nowhere near
the same length, and a plain token-mean therefore weights a task by its share of
the batch's response tokens rather than by any deliberate choice. On the
alfworld / webshop / search mixture the episode caps of 50/15/4 turns put
alfworld around 69% of the response tokens and search around 4%, so "multitask
distillation" is in practice mostly alfworld distillation.

This module computes the per-row weights that turn that into an equal split. The
driver holds the whole step's batch, so the weights can be attached there once
and read by the actor, which sees only micro-batches.

The mechanism is shared by every arm that trains on a per-token distillation
signal (on-policy teacher KL, off-policy top-k KL, hard-label CE), so that the
arms differ in their loss and not in how the tasks are weighted inside it.
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import get_task_names

__all__ = ["TASK_LOSS_WEIGHT_KEY", "attach_task_loss_weights"]

# Column the driver writes and ``DataParallelPPOActor.update_policy`` reads.
# Its presence is what switches the actor from the plain token-mean to the
# weighted sum; absent, nothing changes.
TASK_LOSS_WEIGHT_KEY = "task_loss_weight"


def attach_task_loss_weights(
    batch: DataProto,
    *,
    n_real: int,
    mini_batch_size: int,
    metrics: dict,
    metric_prefix: str = "task_loss",
) -> None:
    """Give every row the weight that makes each task's share of the loss ``1/T``.

    Weighting a row by ``1 / (num_tasks * T_task)`` makes the *sum* over a task's
    rows of ``sum_over_tokens(loss)`` equal to that task's own token-mean divided
    by ``num_tasks``, so the tasks contribute equally regardless of how many rows
    or tokens each brought.

    ``T_task`` is the task's response-token total over the *whole* step rather
    than per mini-batch. Per mini-batch, search contributes only a handful of
    rows, and dividing by a token count drawn from those few rows would amplify
    their noise -- and would divide by zero whenever a mini-batch happened to
    contain none of a task at all. The step-level totals come from thousands of
    rows and need no all_reduce, since the driver has the whole batch here.

    Scaling by the mini-batch count keeps a single optimizer step's loss at the
    same O(1) magnitude as the unweighted token-mean it replaces: without it each
    step would carry ``1/num_mini_batches`` of the step's loss and the effective
    learning rate would collapse. Summed over the step's mini-batches the loss is
    then ``(1/num_tasks) * sum_task token_mean(task)``, i.e. the equal-share
    token-mean, per optimizer step.

    Rows at or past ``n_real`` are ``adjust_batch``'s padding -- on the on-policy
    path those are duplicated trajectories, on the off-policy path duplicated pool
    rows. They get weight 0 and are excluded from ``T_task``, so they neither
    contribute nor dilute. Their originals are still in the batch with a full
    weight, so no turn loses its signal. This is a real difference from the
    unweighted token-mean, which counts a duplicated row twice.

    Args:
        batch: the step's batch, after ``adjust_batch`` and after
            ``response_mask`` has been computed. Modified in place: the weights
            are written to ``batch.batch[TASK_LOSS_WEIGHT_KEY]``.
        n_real: number of rows before ``adjust_batch`` padded the batch.
        mini_batch_size: rows in one optimizer step, counted globally --
            ``ppo_mini_batch_size * rollout.n``, since the workers convert the
            prompt-counted config value themselves (and divide by the DP world
            size, which the actor undoes along with FSDP's gradient averaging).
        metrics: dict the per-task diagnostics are written into.
        metric_prefix: namespace for those diagnostics.
    """
    response_mask = batch.batch["response_mask"]
    row_tokens = response_mask.sum(-1).to(torch.float64)
    real = torch.zeros(len(batch), dtype=torch.bool)
    real[:n_real] = True

    task_names = get_task_names(batch)
    assert task_names is not None, (
        "per-task loss normalisation requires per-row task names on the batch"
    )
    tasks = sorted({t for t in task_names[:n_real] if t is not None})
    assert tasks, "no task names on the real rows of the batch"

    num_mini_batches = len(batch) / int(mini_batch_size)
    assert float(num_mini_batches).is_integer(), (
        f"batch of {len(batch)} rows is not divisible by ppo_mini_batch_size "
        f"{mini_batch_size}; the per-step loss scale would drift between steps"
    )

    weights = torch.zeros(len(batch), dtype=torch.float32)
    real_tokens = float(row_tokens[real].sum())
    for task in tasks:
        rows = torch.from_numpy(np.asarray(task_names == task)) & real
        task_tokens = float(row_tokens[rows].sum())
        assert task_tokens > 0, f"task {task} contributed no response tokens"
        weights[rows] = float(num_mini_batches / (len(tasks) * task_tokens))
        # The share the plain token-mean *would* have given this task. Logged so
        # the imbalance this is correcting stays visible in the run.
        metrics[f"{metric_prefix}/token_share/{task}"] = task_tokens / real_tokens
        metrics[f"{metric_prefix}/rows/{task}"] = int(rows.sum())
    batch.batch[TASK_LOSS_WEIGHT_KEY] = weights
    metrics[f"{metric_prefix}/padding_rows"] = len(batch) - n_real
