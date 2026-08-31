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

import copy
import heapq
from typing import List, Tuple

import torch
from torch import distributed as dist


def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    # add some metrics of seqlen sum on dp ranks
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def deal_by_length(seqlen_list: List[int], partition: List[int], minibatch_rows: int) -> List[int]:
    """Reorder one rank's rows so its mini-batches match the other ranks'.

    ``get_seqlen_balanced_partitions`` equalises the rank totals and then
    ``_check_and_sort_partitions`` sorts each partition by original index,
    discarding the length ordering it worked in. What survives is a rank total;
    what does not is any relationship between rank A's k-th mini-batch and rank
    B's. Since the ranks meet at the gradient reduce that ends every mini-batch,
    that difference is a wait -- measured at 12.7% of a rank's tokens in a real
    step, and invisible to utilization.gpu because a spinning collective is busy
    to it.

    Longest-first, dealt round-robin. Every rank runs the same rule over the same
    number of rows, so mini-batch k holds each rank's k-th, (k+M)-th, (k+2M)-th
    ... longest row and the columns match by construction.

    The capacities are exactly what ``batch.split(minibatch_rows)`` will later
    cut -- ``[C] * (n // C)`` plus a short tail. Dealing into ``ceil(n / C)``
    equal-count buckets instead gives sizes C-1 and C, which split() then cuts
    across, and the careful ordering is lost at the first boundary.

    Sorting the rows and chunking them contiguously would also match the columns,
    and is the wrong fix: it puts every long row in the first mini-batches, which
    on a real step makes the largest mini-batch 6.6x the smallest and 42% larger
    than anything the run sees today. Dealing makes each mini-batch a stratified
    sample instead -- on that same step the largest is 1.17x the smallest, and
    47% BELOW today's largest, so the peak activation footprint falls rather than
    rises.

    Returns the partition's indices in the new order. Same indices, same count.
    """
    n = len(partition)
    if minibatch_rows <= 0 or n <= minibatch_rows:
        return list(partition)
    caps = [minibatch_rows] * (n // minibatch_rows)
    if n % minibatch_rows:
        caps.append(n % minibatch_rows)
    buckets = [[] for _ in caps]
    at = 0
    for idx in sorted(partition, key=lambda i: -seqlen_list[i]):
        while len(buckets[at]) >= caps[at]:
            at = (at + 1) % len(caps)
        buckets[at].append(idx)
        at = (at + 1) % len(caps)
    return [i for bucket in buckets for i in bucket]


def rebalance_minibatch_columns(
    seqlen_list: List[int],
    partitions: List[List[int]],
    minibatch_rows: int,
    micro_batch_rows: int = 0,
) -> List[List[int]]:
    """Re-partition INSIDE each mini-batch column, so the columns match without
    moving a row to a different optimizer step.

    ``deal_by_length`` matches the columns by re-dealing each rank's whole
    partition, which changes WHICH ROWS SHARE A MINI-BATCH. With ~70 optimizer
    steps in a training step that is a different trajectory, and it is why that
    ordering ships off by default.

    This one is the same repair under a constraint that removes that objection.
    Mini-batch k is, today,

        {ordered[r][k*M : (k+1)*M]  for every rank r}

    -- W*M rows that meet at one optimizer step. Take exactly those rows back,
    partition THEM into W equal-size groups by token count, and write the groups
    back into the same column. The column's membership is unchanged, so the
    optimizer step sees the same rows; only which rank carries which of them
    moves.

    WHAT THAT BUYS, AND WHY IT IS NOT MERELY BIT-SHUFFLING. On the arms this
    exists for the loss is a weighted row SUM -- ``normalize_loss_by_task=True``
    routes every term through ``agg_loss_by_task_weights``, whose row weights
    come from STEP-level token totals (``verl/trainer/ppo/task_loss_weights.py``)
    and so do not depend on where a row is placed. FSDP's gradient average and
    the per-mini-batch division by the CONFIGURED gradient_accumulation are both
    undone by the ``task_dp_world_size * gradient_accumulation`` factor the actor
    multiplies in. So the mini-batch's gradient is

        sum over the column's W*M rows of w_i * grad(sum_t loss_it)

    which is invariant under this reassignment in exact arithmetic. The step's
    two cumulative statistics are float64 ``index_add_`` totals reduced with
    ``ReduceOp.SUM`` once per step, so they are invariant too. What moves is
    floating-point summation order -- the same class as ``ppo_micro_batch_size_
    per_gpu`` 10 -> 5, which this arm already took deliberately -- and NOT which
    rows the optimizer sees together.

    IT DOES NOT HOLD FOR AN UNWEIGHTED TOKEN-MEAN. ``agg_loss(..., "token-mean")``
    divides by the MICRO-BATCH's own token count, so moving a row across ranks
    reweights it. Callers on that path get a different objective, not a different
    rounding, which is why the switch in ``_balance_batch`` is off by default and
    the run scripts that turn it on are the ones that pin
    ``normalize_loss_by_task``.

    ``micro_batch_rows`` then deals each rank's share of the column longest-first,
    the way ``deal_by_length`` deals a whole partition. That is not cosmetic here:
    the ranks do NOT only meet at the mini-batch boundary on these arms. The
    teacher lookup runs a pair of collectives INSIDE the micro-batch loop
    (``exchange_teacher_logprobs_multi``, called from ``_teacher_logprobs_at``),
    so they meet every ``ppo_micro_batch_size_per_gpu`` rows. Balancing the
    column and then handing it over in index order would leave that finer
    meeting as unmatched as it is today. Passing 0 skips it.

    Returns the partitions in the new order: the same indices, the same count per
    rank, and the same multiset of indices in every column.
    """
    k_partitions = len(partitions)
    if k_partitions < 2 or minibatch_rows <= 0:
        return [list(p) for p in partitions]
    n = len(partitions[0])
    if any(len(p) != n for p in partitions):
        raise ValueError(
            "rebalance_minibatch_columns needs equal-size partitions -- the columns are "
            f"defined by position, and these are {[len(p) for p in partitions]}"
        )
    out = [[] for _ in partitions]
    for a in range(0, n, minibatch_rows):
        b = min(a + minibatch_rows, n)
        column = [i for p in partitions for i in p[a:b]]
        groups = get_seqlen_balanced_partitions(
            [seqlen_list[i] for i in column], k_partitions=k_partitions, equal_size=True
        )
        for r, group in enumerate(groups):
            rows = [column[j] for j in group]
            if micro_batch_rows:
                rows = deal_by_length(seqlen_list, rows, micro_batch_rows)
            out[r].extend(rows)
    return out


def log_minibatch_unbalance(seqlen_list: List[int], partitions: List[List[int]], minibatch_rows: int, prefix,
                           micro_batch_rows: int = 0):
    """How balanced the MINI-BATCHES are across ranks, which is a different
    question from how balanced the ranks are.

    ``get_seqlen_balanced_partitions`` equalises each rank's total over the whole
    batch, and ``log_seqlen_unbalance`` reports that it succeeded -- typically to
    within a single token. But the ranks do not meet once per batch. They meet at
    the gradient reduce and optimizer step that end every mini-batch, and each
    mini-batch is a contiguous slice of the rank's rows: ``dataloader =
    batch.split(ppo_mini_batch_size)``. Whether slice k holds the same number of
    tokens on every rank is not something the whole-batch balance constrains, and
    ``_check_and_sort_partitions`` sorts each partition by original index, which
    discards the length ordering the partitioner worked in.

    Every rank waits for the slowest one at each of those meetings, so the loss is
    the sum over mini-batches of (slowest - mean), not (whole-batch max - min).
    That is what ``wait_frac`` reports, as a fraction of one rank's tokens. It
    assumes the time a mini-batch takes is proportional to its tokens, which is
    close enough for the packed forward this arm runs, and it is an upper bound
    on what any per-mini-batch rebalancing could recover -- NOT a measured stall.

    ``wait_frac_dealt`` is the same number under ``deal_by_length``, and
    ``wait_frac_columns`` under :func:`rebalance_minibatch_columns` -- the two
    orderings a fix would actually use. The gap between either and ``wait_frac``
    is the headroom, measured rather than assumed.

    ``micro_batch_rows`` ADDS THE MEETING THAT ACTUALLY HAPPENS MORE OFTEN. The
    mini-batch boundary is where FSDP reduces, and on an arm that reads a teacher
    out of another rank's cache it is not the only place the ranks meet:
    ``exchange_teacher_logprobs_multi`` runs an all_gather and an all_reduce pair
    from inside the micro-batch loop, so they meet every
    ``ppo_micro_batch_size_per_gpu`` rows. ``microbatch_wait_frac`` is the same
    arithmetic chunked at that width, and it is the larger of the two numbers --
    a column can hold matching totals and still be split into micro-batches that
    do not. Reported alongside rather than instead: which one binds depends on
    whether the arm runs that lookup, and the pair says so.
    """
    if minibatch_rows <= 0:
        return {}

    def _tokens(index_lists):
        return [[seqlen_list[i] for i in rows] for rows in index_lists]

    def _wait(ordered, width):
        chunks = [[sum(rows[i : i + width]) for i in range(0, len(rows), width)] for rows in ordered]
        n = min(len(c) for c in chunks)
        waited = total = 0.0
        spreads = []
        for k in range(n):
            column = [c[k] for c in chunks]
            mean = sum(column) / len(column)
            waited += max(column) - mean
            total += mean
            spreads.append(max(column) - min(column))
        return waited, total, spreads

    per_rank = _tokens(partitions)
    waited, total, spreads = _wait(per_rank, minibatch_rows)
    if not total:
        return {}

    dealt = _tokens([deal_by_length(seqlen_list, p, minibatch_rows) for p in partitions])
    dealt_waited, dealt_total, _ = _wait(dealt, minibatch_rows)
    # The counterfactual this metric exists to price, run on the same rows that
    # were actually dispatched. Cheap enough to leave on: Karmarkar-Karp over
    # world_size * minibatch_rows items, once per mini-batch column -- the whole
    # of log_minibatch_unbalance measures 42 ms on a 7,080-row step at
    # minibatch_rows=30, on the DRIVER, against a step of ~530 s.
    columns = _tokens(rebalance_minibatch_columns(
        seqlen_list, partitions, minibatch_rows, micro_batch_rows=micro_batch_rows,
    ))
    col_waited, col_total, _ = _wait(columns, minibatch_rows)

    out = {
        f"{prefix}/minibatch_spread_mean": sum(spreads) / len(spreads),
        f"{prefix}/minibatch_spread_max": max(spreads),
        f"{prefix}/minibatch_wait_frac": waited / total,
        f"{prefix}/minibatch_wait_frac_dealt": (dealt_waited / dealt_total) if dealt_total else 0.0,
        f"{prefix}/minibatch_wait_frac_columns": (col_waited / col_total) if col_total else 0.0,
    }
    if micro_batch_rows > 0:
        micro_waited, micro_total, _ = _wait(per_rank, micro_batch_rows)
        micro_col_waited, micro_col_total, _ = _wait(columns, micro_batch_rows)
        out[f"{prefix}/microbatch_wait_frac"] = (micro_waited / micro_total) if micro_total else 0.0
        out[f"{prefix}/microbatch_wait_frac_columns"] = (
            (micro_col_waited / micro_col_total) if micro_col_total else 0.0
        )
    return out


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


def rearrange_micro_batches(batch, max_token_len, dp_group=None, num_batches_divided_by=None, same_micro_num_in_dp=True, min_num_micro_batch=None):
    """
    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    # this is per local micro_bsz
    max_seq_len = batch["attention_mask"].shape[-1]
    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    seq_len_effective = seq_len_effective.tolist()
    assert num_micro_batches <= len(seq_len_effective)

    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = []
        for idx in partition:
            curr_micro_batch.append(batch[idx : idx + 1])
        curr_micro_batch = torch.cat(curr_micro_batch)

        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
