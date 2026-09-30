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

THE POLICY GRADIENT HAS A SECOND OPTION (actor.pg_loss_norm). The weights above
divide each task's share by T_d, the response tokens the task brought to THIS
step, and the actor applies them to every term, the policy gradient included:
"token", the default and what every existing arm runs. For the teacher KL that is
the point -- it is a per-token loss. For the policy gradient it puts the lengths
the policy just sampled into the denominator, which makes the update a ratio
estimator: its expectation is in general not the gradient of the fixed objective
it stands for, even with a baseline that does not depend on the trajectory. (The
review of bec1dad built the counterexample from the production functions:
one-turn episodes that always win, a short or a long reply at 1/2 each, a frozen
baseline of 1/2. The success objective's gradient is 0; the token-weighted
update's expectation is -0.0145.)

"trajectory" gives the policy-gradient term a row weight of its own,

    num_mini_batches / (D * N_d * L_d)

where N_d is the number of the task's real trajectories in the step (distinct
traj_uid over the rows before adjust_batch's padding) and L_d is a FIXED reference
response length per task, tokens per trajectory (actor.pg_ref_tokens), never read
off the batch. num_mini_batches is the token weight's own factor, the number of
optimizer steps update_policy cuts the batch into. Summed as the actor sums every
term -- over a trajectory's turn rows and over each turn's generated tokens, never
divided by the trajectory's own length, with FSDP's average over the DP ranks and
the division by gradient_accumulation undone exactly, so neither adds a factor --
the step's policy-gradient loss at the rollout policy has the gradient
num_mini_batches * g, where

    g = (1/D) sum_d 1/(N_d L_d) sum_{i in d} sum_t A_it sum_{u in turn t} grad log pi(y_itu)

and the batch is taken in num_mini_batches optimizer steps, each with its own
rows' part: the MEAN OPTIMIZER STEP carries exactly g. N_d is fixed by the batch
design (prompts per task x the group size) and L_d by the config, so nothing in
g's weights depends on what was sampled: (a) is the token weight with T_d replaced
by N_d * L_d, the same num_mini_batches in both, and that replacement is the whole
difference. What it buys is limited to what it says. With an advantage whose
expectation is the objective's (the Monte Carlo return minus a baseline that does
not depend on the trajectory), g -- the mean optimizer step's ON-POLICY, PRE-CLIP
policy gradient -- is an unbiased estimate of the gradient of the fixed objective
(1/D) sum_d J_d / L_d. Under (b) the same mean optimizer step is the ratio
estimator above.

WHAT IT DOES NOT FIX: HOW MANY OPTIMIZER STEPS A BATCH BECOMES. num_mini_batches
= ceil(rows / mini_batch_size), and a row is a turn, so it follows the episode
lengths the policy sampled. Measured on beta-mirror v2's log: 115 (steps 1-10)
down to 60 (steps 291-300), 36 to 123 in all, about 13% step to step around that
trend, and -0.98 against ALFWorld's training success (-0.92 around the trends).
The step's summed update, num_mini_batches * g, is therefore NOT a fixed multiple
of g, and its expectation is not the objective's gradient up to a constant. The
one-task toy in tests/trainer/test_pg_loss_norm.py (a 1-turn or a 3-turn episode
at 1/2 each, every one a win, so the objective's gradient is 0) shows both halves:
E[g] = 0, while the step's summed policy gradient at one row per mini-batch has
expectation +1/8 along the log-odds of the 3-turn reply -- towards the episodes
that bring more rows. That factor is the same in (b) and in every arm here: it is
how many PPO iterations a batch is taken in, not how the batch is weighted, and it
sits with the clip, the later mini-batches' moved ratio, an approximate value
under lambda < 1 and the teacher term as matters this module does not settle.
Fixing it would take a fixed number of optimizer steps per training step (a
mini-batch that grows and shrinks with the batch); the (a) column would then run
another optimizer schedule than the (b) column, and comparing them would compare
schedules -- in effect learning rates -- as well. Not done; task_loss/optimizer_steps
logs the count under trajectory. A gamma^t prefix is the advantage's business
(progress_value.prefix_discount), not this module's.

WHY A PER-TASK L_d AND NOT ONE CONSTANT. Moving the denominator from tokens to
trajectories changes the policy gradient's size, and with it its size against the
teacher KL, which keeps the token weights; left alone, (a) against (b) would
compare learning rates and teacher strengths as much as normalisations. With L_d
at a run's own mean response tokens per trajectory each task's policy gradient
keeps, on average, the size (b) gave it -- fixed before the run, never corrected
from the current batch. The price is said here so no one reads more into it: the
tasks then keep the relative weights (b) gave them on average, 1/L_d each, which
is NOT equal weight per unit of success (one common L would be, and would move
every task's policy gradient against its teacher term). What drifts is logged:
task_loss/pg_weight_ratio/<task> = T_d / (N_d * L_d), the (a) weight over the (b)
weight.

Only the policy-gradient term takes the trajectory weight. The teacher KL, the
reference KL, the entropy bonus and SDAR keep TASK_LOSS_WEIGHT_KEY.
"""

import math
from typing import Mapping, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import get_task_names

__all__ = [
    "PG_LOSS_NORMS",
    "TASK_LOSS_WEIGHT_KEY",
    "TASK_PG_LOSS_WEIGHT_KEY",
    "attach_task_loss_weights",
    "check_pg_loss_norm_config",
    "pg_loss_norm_kwargs",
]

# Column the driver writes and ``DataParallelPPOActor.update_policy`` reads.
# Its presence is what switches the actor from the plain token-mean to the
# weighted sum; absent, nothing changes.
TASK_LOSS_WEIGHT_KEY = "task_loss_weight"

# The policy-gradient term's own row weight, written only under
# actor.pg_loss_norm=trajectory. The actor reads it only when its own config asks
# for that mode, so a batch that happens to carry it changes nothing in token mode.
TASK_PG_LOSS_WEIGHT_KEY = "task_pg_loss_weight"

# actor.pg_loss_norm's values: "token" is (b), the task's realised response
# tokens; "trajectory" is (a), its real trajectory count times a fixed length.
PG_LOSS_NORMS = ("token", "trajectory")


def attach_task_loss_weights(
    batch: DataProto,
    *,
    n_real: int,
    mini_batch_size: int,
    metrics: dict,
    metric_prefix: str = "task_loss",
    pg_loss_norm: str = "token",
    pg_ref_tokens: Optional[Mapping[str, float]] = None,
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
    token-mean, per optimizer step. A short final mini-batch is fine and is the
    common case -- see the count below for why.

    Rows at or past ``n_real`` are ``adjust_batch``'s padding -- on the on-policy
    path those are duplicated trajectories, on the off-policy path duplicated pool
    rows. They get weight 0 and are excluded from ``T_task``, so they neither
    contribute nor dilute. Their originals are still in the batch with a full
    weight, so no turn loses its signal. This is a real difference from the
    unweighted token-mean, which counts a duplicated row twice.

    Under ``pg_loss_norm="trajectory"`` the policy-gradient term gets a second
    column, ``TASK_PG_LOSS_WEIGHT_KEY`` = ``num_mini_batches / (num_tasks * N_task
    * L_task)`` on the task's real rows and 0 on the padding (see the module
    docstring for what that is the gradient of: the mean optimizer step's, while
    num_mini_batches itself moves with the batch's rows). ``N_task`` counts TRAJECTORIES,
    the distinct ``traj_uid`` among the task's real rows, not rows: a row is one
    turn, so a row count would divide by the sampled episode lengths again, which
    is the dependence this mode exists to remove. A padding copy carries its
    original's ``traj_uid`` and is not counted a second time. The same
    mini-batch count and the same undo of FSDP's averaging and the gradient
    accumulation apply as for the token weight, so a short final mini-batch is
    handled the same way.

    Args:
        batch: the step's batch, after ``adjust_batch`` and after
            ``response_mask`` has been computed. Modified in place: the weights
            are written to ``batch.batch[TASK_LOSS_WEIGHT_KEY]`` (and, under
            ``pg_loss_norm="trajectory"``, ``batch.batch[TASK_PG_LOSS_WEIGHT_KEY]``).
        n_real: number of rows before ``adjust_batch`` padded the batch.
        mini_batch_size: rows in one optimizer step, counted globally --
            ``ppo_mini_batch_size * rollout.n``, since the workers convert the
            prompt-counted config value themselves (and divide by the DP world
            size, which the actor undoes along with FSDP's gradient averaging).
        metrics: dict the per-task diagnostics are written into.
        metric_prefix: namespace for those diagnostics.
        pg_loss_norm: ``actor.pg_loss_norm``; "token" (the default) writes
            nothing beyond the token weight, bit for bit what this function has
            always written.
        pg_ref_tokens: ``actor.pg_ref_tokens``, ``{task: L_task}``; read only
            under "trajectory", where every task on the real rows needs a
            positive entry.
    """
    assert pg_loss_norm in PG_LOSS_NORMS, (
        f"actor.pg_loss_norm={pg_loss_norm!r}; expected one of {PG_LOSS_NORMS}"
    )
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

    # Checked before anything is written, so a misconfigured run stops with the
    # batch untouched rather than half-weighted.
    traj_pg = pg_loss_norm == "trajectory"
    if traj_pg:
        ref_tokens = _positive_ref_tokens(pg_ref_tokens, tasks)
        traj_uid = batch.non_tensor_batch.get("traj_uid", None)
        assert traj_uid is not None, (
            "actor.pg_loss_norm=trajectory counts each task's trajectories by traj_uid, "
            "and this batch carries none"
        )
        traj_uid = np.asarray(traj_uid, dtype=object)
        real_np = real.numpy()

    # The batch is generally NOT a multiple of the mini-batch size, and does not
    # need to be. adjust_batch rounds to lcm(log_prob_micro * W, ppo_micro * W)
    # -- 160 against a mini-batch of 60 on the multitask run -- so the last
    # mini-batch update_policy splits off is usually short.
    #
    # That short mini-batch needs no special handling *here*: update_policy
    # divides every mini-batch by the CONFIGURED gradient_accumulation and the
    # weights are multiplied by the same constant, so the two cancel and a
    # mini-batch contributes exactly the sum of its rows' weighted losses however
    # many rows it has. num_mini_batches only sets the overall scale, so what it
    # has to equal is the number of optimizer steps the batch becomes -- which is
    # what batch.split() produces, i.e. the ceiling. It is a count of THIS batch's
    # rows, which are turns: it keeps each optimizer step at the step objective's
    # size (the mean optimizer step carries it), and the step as a whole carries it
    # num_mini_batches times -- a number that moves with the sampled episode
    # lengths, under either normalisation (module docstring, WHAT IT DOES NOT FIX).
    num_mini_batches = math.ceil(len(batch) / int(mini_batch_size))

    weights = torch.zeros(len(batch), dtype=torch.float32)
    # float32 like the token weight: the actor casts both to the loss's dtype on
    # the multiply, and a wider column would promote the loss instead.
    pg_weights = torch.zeros(len(batch), dtype=torch.float32) if traj_pg else None
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
        if traj_pg:
            n_traj = len(set(traj_uid[np.asarray(task_names == task) & real_np].tolist()))
            pg_weights[rows] = float(num_mini_batches / (len(tasks) * n_traj * ref_tokens[task]))
            metrics[f"{metric_prefix}/pg_trajectories/{task}"] = n_traj
            # The (a) weight over the (b) weight: this step's mean response tokens
            # per trajectory over the fixed reference. 1 where the two
            # normalisations agree; its drift over a run is how far the token
            # normalisation would have moved this task's policy gradient.
            metrics[f"{metric_prefix}/pg_weight_ratio/{task}"] = task_tokens / (n_traj * ref_tokens[task])
    batch.batch[TASK_LOSS_WEIGHT_KEY] = weights
    if traj_pg:
        batch.batch[TASK_PG_LOSS_WEIGHT_KEY] = pg_weights
        # How many optimizer steps this batch becomes: the factor between g (the mean optimizer
        # step's policy gradient) and the step's summed update, which this normalisation does
        # not fix (module docstring). Logged under trajectory only; token writes what it always did.
        metrics[f"{metric_prefix}/optimizer_steps"] = num_mini_batches
    metrics[f"{metric_prefix}/padding_rows"] = len(batch) - n_real


def _positive_ref_tokens(pg_ref_tokens: Optional[Mapping[str, float]], tasks) -> dict:
    """``{task: L_task}`` for every task in ``tasks``, each a positive number, or a refusal.

    A missing entry is refused rather than defaulted: L_task is the task's whole
    policy-gradient scale under the trajectory normalisation, and a default would
    be a scale nobody chose.
    """
    assert pg_ref_tokens is not None, (
        "actor.pg_loss_norm=trajectory divides each task's policy gradient by its trajectory "
        "count times actor.pg_ref_tokens[task], a fixed reference length; pg_ref_tokens is unset"
    )
    ref = {str(k): float(v) for k, v in dict(pg_ref_tokens).items()}
    missing = sorted(set(tasks) - set(ref))
    assert not missing, (
        f"actor.pg_ref_tokens has no entry for {missing} (it names {sorted(ref)}); every task "
        "trained under pg_loss_norm=trajectory needs its fixed reference length"
    )
    bad = {t: ref[t] for t in tasks if not (math.isfinite(ref[t]) and ref[t] > 0)}
    assert not bad, f"actor.pg_ref_tokens must be positive and finite; got {bad}"
    return ref


def pg_loss_norm_kwargs(actor_config) -> dict:
    """``attach_task_loss_weights``'s two policy-gradient arguments, read from the actor config.

    One reader for the driver's call and for :func:`check_pg_loss_norm_config`, so
    the key names and the default ("token", i.e. what every existing arm runs)
    are spelled in one place.
    """
    norm = str(actor_config.get("pg_loss_norm", "token") or "token")
    ref = actor_config.get("pg_ref_tokens", None)
    return {
        "pg_loss_norm": norm,
        "pg_ref_tokens": None if ref is None else {str(k): float(v) for k, v in dict(ref).items()},
    }


def check_pg_loss_norm_config(actor_config, tasks=None) -> str:
    """Refuse a policy-gradient normalisation this module cannot deliver; return the mode.

    For a launch-time check (the step-1 driver call and the actor would refuse the
    same things, but only after the first rollout). "trajectory" needs the
    per-task weights on at all -- the PG column is written by
    ``attach_task_loss_weights``, which runs only under normalize_loss_by_task,
    and the actor reads it only on its weighted path -- and a positive reference
    length for every task the run trains (``tasks``, when given).

    Takes the ACTOR config after main_opd's injection: normalize_loss_by_task is
    set there from algorithm.opd.normalize_loss_by_task.
    """
    kw = pg_loss_norm_kwargs(actor_config)
    norm = kw["pg_loss_norm"]
    assert norm in PG_LOSS_NORMS, (
        f"actor.pg_loss_norm={norm!r}; expected one of {PG_LOSS_NORMS}"
    )
    if norm == "trajectory":
        assert bool(actor_config.get("normalize_loss_by_task", False)), (
            "actor.pg_loss_norm=trajectory is a second per-task weight beside the token one, "
            "written by attach_task_loss_weights; it needs algorithm.opd.normalize_loss_by_task=True"
        )
        # Without a task list, every entry that is there must still be usable.
        names = tasks if tasks is not None else (kw["pg_ref_tokens"] or {})
        _positive_ref_tokens(kw["pg_ref_tokens"], sorted(str(t) for t in names))
    return norm
