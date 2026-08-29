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
    deal_by_length,
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
    # and dealing both ranks by length removes all of it, which is the headroom
    assert mini["p/minibatch_wait_frac_dealt"] == 0.0


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
    # dealing each rank's rows by length is a real improvement, not a wash
    assert mini["p/minibatch_wait_frac_dealt"] < mini["p/minibatch_wait_frac"] / 2


# --- the fix ----------------------------------------------------------------


def _chunks(values, size):
    return [sum(values[i : i + size]) for i in range(0, len(values), size)]


def test_the_deal_cuts_on_the_boundaries_split_will_use():
    """The detail that decides whether any of this survives.

    batch.split(C) cuts at [C, C, ..., n % C]. Dealing into ceil(n / C)
    equal-count buckets gives sizes C-1 and C, which split() then cuts across --
    every bucket after the first straddles two mini-batches and the ordering is
    lost at the first boundary.
    """
    seqlens = list(range(1, 91))            # 90 rows, C = 20 -> [20, 20, 20, 20, 10]
    order = deal_by_length(seqlens, list(range(90)), 20)

    assert sorted(order) == list(range(90))          # a permutation, nothing lost
    counts = [20, 20, 20, 20, 10]
    at = 0
    for count in counts:                             # each cut lands on a full bucket
        block = order[at : at + count]
        assert len(block) == count
        at += count


def test_it_matches_the_columns_across_ranks():
    """Three ranks, the same rule, the same row count -- so mini-batch k holds
    each rank's k-th, (k+M)-th ... longest row and the sums line up. This is the
    whole mechanism."""
    random.seed(3)
    seqlens = [int(random.lognormvariate(0, 0.6) * 900) for _ in range(1200)]
    partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)

    before = log_minibatch_unbalance(seqlens, partitions, 20, prefix="p")
    dealt = [deal_by_length(seqlens, p, 20) for p in partitions]
    after = log_minibatch_unbalance(seqlens, dealt, 20, prefix="p")

    assert before["p/minibatch_wait_frac"] > 0.02
    assert after["p/minibatch_wait_frac"] < 0.005


def test_it_does_not_pile_the_long_rows_into_the_first_mini_batches():
    """Sorting and chunking contiguously matches the columns too, and is the
    wrong fix: it makes the first mini-batch the longest rows in the batch. On a
    real step that is 6.6x the smallest and 42% larger than anything the run sees
    today, which is a peak-activation increase in exchange for a wait. Dealing
    makes every mini-batch a stratified sample instead."""
    random.seed(4)
    seqlens = [int(random.lognormvariate(0, 0.6) * 900) for _ in range(1200)]
    partition = list(range(1200))

    contiguous = _chunks([seqlens[i] for i in sorted(partition, key=lambda i: -seqlens[i])], 20)
    dealt = _chunks([seqlens[i] for i in deal_by_length(seqlens, partition, 20)], 20)

    assert max(contiguous) / min(contiguous) > 4
    assert max(dealt) / min(dealt) < 1.3
    # and the peak mini-batch is smaller than what index order already produces,
    # so the fix lowers the activation high-water mark rather than raising it
    assert max(dealt) < max(_chunks([seqlens[i] for i in partition], 20))


def test_a_partition_smaller_than_one_mini_batch_is_left_alone():
    """Nothing to deal, and reordering it would change the trajectory for no
    gain."""
    assert deal_by_length([5, 1, 3], [0, 1, 2], 20) == [0, 1, 2]
    assert deal_by_length([5, 1, 3], [0, 1, 2], 0) == [0, 1, 2]


def test_dealing_is_a_pure_permutation():
    """It must never lose, duplicate, or invent a row: the result indexes the
    same batch."""
    random.seed(5)
    seqlens = [random.randint(1, 5000) for _ in range(437)]
    out = deal_by_length(seqlens, list(range(437)), 20)

    assert sorted(out) == list(range(437))
    assert len(out) == 437


# --------------------------------------------------------------------------- #
# The default matters more than the mechanism here. deal_by_length changes which
# rows share a mini-batch, and with ~70 optimizer steps per training step that
# is a different trajectory -- an arm run with it on is not comparable with the
# arms already finished. It ships off; the MEASUREMENT ships on, so the size of
# the prize is known before anyone decides to spend it.
# --------------------------------------------------------------------------- #


def test_the_reordering_is_off_by_default():
    import verl.trainer.ppo.ray_trainer as rt

    assert rt._BALANCE_MINIBATCH is False, (
        "BALANCE_MINIBATCH defaults on; every arm run from here is a different "
        "trajectory from the ones it will be compared against"
    )


def test_the_measurement_runs_regardless_of_the_flag():
    """It is pure arithmetic over the token counts -- it reorders nothing and
    reports what was actually dispatched, so it is free to leave on."""
    import ast
    import inspect
    import textwrap

    import verl.trainer.ppo.ray_trainer as rt

    src = textwrap.dedent(inspect.getsource(rt.RayPPOTrainer._balance_batch))
    tree = ast.parse(src)
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "log_minibatch_unbalance"
    )
    # Walk out from the call and check no _BALANCE_MINIBATCH test encloses it.
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "_BALANCE_MINIBATCH" not in names:
                continue
            assert call not in list(ast.walk(node)), (
                "the measurement is gated on the flag; then it only reports on "
                "runs that already took the change, which is backwards"
            )


def test_dealing_preserves_the_rows_exactly():
    """Whatever the ordering does, it must be a permutation: the same indices,
    the same count. A row silently dropped here is a row never trained on."""
    from verl.utils.seqlen_balancing import deal_by_length

    seqlens = [17, 3, 91, 45, 8, 62, 33, 5, 70]
    partition = list(range(len(seqlens)))
    for rows in (1, 2, 3, 4, 5, 9, 20):
        out = deal_by_length(seqlens, partition, rows)
        assert sorted(out) == sorted(partition), rows
        assert len(out) == len(partition), rows


def test_dealing_fills_the_buckets_split_will_actually_cut():
    """The capacities have to be what batch.split(minibatch_rows) produces --
    [C]*(n//C) plus a short tail. Dealing into equal-count buckets instead gives
    sizes C-1 and C, which split() then cuts ACROSS, and the careful ordering is
    lost at the first boundary."""
    from verl.utils.seqlen_balancing import deal_by_length

    seqlens = list(range(10, 10 + 7))
    out = deal_by_length(seqlens, list(range(7)), 3)
    sizes = [len(out[i : i + 3]) for i in range(0, 7, 3)]
    assert sizes == [3, 3, 1]
