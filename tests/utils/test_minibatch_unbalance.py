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
"""Balanced ranks are not balanced mini-batches.

``_balance_batch`` equalises each rank's total over the whole batch, and the
existing metric reports that it worked -- ``global_seqlen/balanced_min`` and
``balanced_max`` differ by a single token in a real step. That number has been
read as "the ranks are balanced", and it does not mean that.

The ranks do not meet once per batch. They meet at the gradient reduce and the
optimizer step that end every mini-batch, sixty-odd times a step, and each
mini-batch is a contiguous slice of the rank's rows. Nothing constrains slice k
to hold the same tokens on every rank -- ``_check_and_sort_partitions`` sorts
each partition by original index, throwing away the length ordering the
partitioner worked in. Every rank then waits for the slowest at each meeting,
and NVML cannot see it: a spinning collective counts as busy.

These cover that the new metric measures that wait rather than the whole-batch
spread, and that its counterfactual is a real one.
"""

import random

from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_minibatch_unbalance,
    log_seqlen_unbalance,
)


def test_ranks_balanced_to_a_token_can_still_wait_on_every_mini_batch():
    """The headline claim, in the smallest form that shows it.

    Two ranks, four rows each, identical totals -- the whole-batch metric calls
    this perfect. But rank 0 carries its long rows first and rank 1 carries them
    last, so with two rows per mini-batch rank 1 waits through the first and
    rank 0 waits through the second.
    """
    seqlens = [100, 100, 1, 1,      # rank 0's rows, in order
               1, 1, 100, 100]      # rank 1's rows, in order
    partitions = [[0, 1, 2, 3], [4, 5, 6, 7]]

    whole = log_seqlen_unbalance(seqlens, partitions, prefix="p")
    assert whole["p/balanced_max"] == whole["p/balanced_min"] == 202   # "perfect"

    mini = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=2, prefix="p")
    # each of the two mini-batches: max 200, mean 101 -> 99 waited, of 101
    assert mini["p/minibatch_wait_frac"] == 99 / 101
    assert mini["p/minibatch_spread_max"] == 198
    # and ordering both ranks by length removes all of it, which is the headroom
    assert mini["p/minibatch_wait_frac_sorted"] == 0.0


def test_already_matched_mini_batches_report_no_wait():
    """The metric has to be zero when there is nothing to fix, or every run
    looks like it has a problem."""
    seqlens = [10, 20, 30, 40] * 3
    partitions = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    mini = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=2, prefix="p")

    assert mini["p/minibatch_wait_frac"] == 0.0
    assert mini["p/minibatch_spread_mean"] == 0.0
    assert mini["p/minibatch_spread_max"] == 0.0


def test_the_wait_is_against_the_mean_not_the_minimum():
    """Three ranks, one slow. Two of them wait, and they wait by different
    amounts; charging the whole max-min spread would roughly double the number
    and make a real 5% look like 10%."""
    seqlens = [100, 60, 20]
    partitions = [[0], [1], [2]]

    mini = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=1, prefix="p")

    assert mini["p/minibatch_wait_frac"] == (100 - 60) / 60      # mean is 60
    assert mini["p/minibatch_spread_max"] == 80                  # not what is charged


def test_a_partial_last_mini_batch_does_not_skew_the_number():
    """batch.split leaves a short tail when the rows do not divide evenly. Its
    columns are still comparable across ranks, but a rank with one more chunk
    than another must not have that chunk compared against nothing."""
    seqlens = [10, 10, 10,      # rank 0: 3 rows -> chunks of 2 and 1
               10, 10, 10]      # rank 1: same
    partitions = [[0, 1, 2], [3, 4, 5]]

    mini = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=2, prefix="p")

    assert mini["p/minibatch_wait_frac"] == 0.0


def test_it_says_nothing_rather_than_guessing_when_the_size_is_unknown():
    """An arm with no actor config to read must lose the metric, not emit a
    fabricated one."""
    assert log_minibatch_unbalance([1, 2, 3, 4], [[0, 1], [2, 3]], 0, prefix="p") == {}


def test_against_the_real_partitioner_on_a_realistic_batch():
    """End to end through the code the trainer actually calls.

    The claim under test is that get_seqlen_balanced_partitions equalising the
    rank totals leaves the mini-batches unequal -- so this asserts both halves:
    the totals match to within a rounding error, and the per-mini-batch wait
    does not vanish.
    """
    random.seed(0)
    # 3 ranks, 60 mini-batches of 20 rows each, lengths spread the way a
    # multi-task batch's are: a long-tailed mix rather than one distribution.
    seqlens = [int(random.lognormvariate(0, 0.5) * mean)
               for mean in (700, 700, 1400) for _ in range(1200)]
    random.shuffle(seqlens)
    partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)

    whole = log_seqlen_unbalance(seqlens, partitions, prefix="p")
    mini = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=20, prefix="p")

    spread = whole["p/balanced_max"] - whole["p/balanced_min"]
    assert spread <= 3, spread                       # the ranks are balanced...
    assert mini["p/minibatch_wait_frac"] > 0.02      # ...and the mini-batches are not
    # ordering each rank's rows by length is a real improvement, not a wash
    assert mini["p/minibatch_wait_frac_sorted"] < mini["p/minibatch_wait_frac"] / 2
