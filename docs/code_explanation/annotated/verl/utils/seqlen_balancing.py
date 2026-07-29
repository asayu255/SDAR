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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import heapq
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import distributed as dist


# [EXPLAIN] `karmarkar_karp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    # [EXPLAIN] `Set` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class Set:
        # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __init__(self) -> None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sum = 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.items = []

        # [EXPLAIN] `add` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def add(self, idx: int, val: int):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.items.append((idx, val))
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.sum += val

        # [EXPLAIN] `merge` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def merge(self, other):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, val in other.items:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.items.append((idx, val))
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                self.sum += val

        # [EXPLAIN] `__lt__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __lt__(self, other):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.sum != other.sum:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return self.sum < other.sum
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(self.items) != len(other.items):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return len(self.items) < len(other.items)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.items < other.items

    # [EXPLAIN] `State` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
    class State:
        # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.k = k
            # sets should always be decreasing order
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sets = [Set() for _ in range(k)]
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i, (idx, seqlen) in enumerate(items):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.sets[i].add(idx=idx, val=seqlen)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sets = sorted(self.sets, reverse=True)

        # [EXPLAIN] `get_partitions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def get_partitions(self):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            partitions = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(len(self.sets)):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cur_partition = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for idx, _ in self.sets[i].items:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    cur_partition.append(idx)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                partitions.append(cur_partition)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return partitions

        # [EXPLAIN] `merge` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def merge(self, other):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(self.k):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.sets[i].merge(other.sets[self.k - 1 - i])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sets = sorted(self.sets, reverse=True)

        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @property
        # [EXPLAIN] `spread` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def spread(self) -> int:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.sets[0].sum - self.sets[-1].sum

        # [EXPLAIN] `__lt__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.spread != other.spread:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return self.spread > other.spread
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.sets[0] > other.sets[0]

        # [EXPLAIN] `__repr__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __repr__(self) -> str:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            repr_str = "["
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(self.k):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if i > 0:
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    repr_str += ","
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                repr_str += "{"
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if j > 0:
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        repr_str += ","
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    repr_str += str(seqlen)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                repr_str += "}"
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            repr_str += "]"
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return repr_str

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    states_pq = []
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if equal_size:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            items = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(k_partitions):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                seqlen, idx = sorted_seqlen_list[offset + i]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                items.append((idx, seqlen))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for seqlen, idx in sorted_seqlen_list:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while len(states_pq) > 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state0 = heapq.heappop(states_pq)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state1 = heapq.heappop(states_pq)
        # merge states
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        state0.merge(state1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        heapq.heappush(states_pq, state0)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    final_state = states_pq[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partitions = final_state.get_partitions()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if equal_size:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, partition in enumerate(partitions):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return partitions


# [EXPLAIN] `greedy_partition` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    bias = sum(seqlen_list) + 1 if equal_size else 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partitions = [[] for _ in range(k_partitions)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partition_sums = [0 for _ in range(k_partitions)]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for seqlen, i in sorted_seqlen:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        min_idx = None
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for j in range(k_partitions):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                min_idx = j
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        partitions[min_idx].append(i)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        partition_sums[min_idx] += seqlen
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if equal_size:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, partition in enumerate(partitions):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return partitions


# [EXPLAIN] `get_seqlen_balanced_partitions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    # [EXPLAIN] `_check_and_sort_partitions` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _check_and_sort_partitions(partitions):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seen_idx = set()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sorted_partitions = [None] * k_partitions
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, partition in enumerate(partitions):
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert len(partition) > 0, f"the {i}-th partition is empty"
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx in partition:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_idx.add(idx)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sorted_partitions[i] = sorted(partition)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert seen_idx == set(range(len(seqlen_list)))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sorted_partitions

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _check_and_sort_partitions(partitions)


# [EXPLAIN] `log_seqlen_unbalance` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    # add some metrics of seqlen sum on dp ranks
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = len(seqlen_list) // k_partition
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_sum_seqlen = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_sum_seqlen = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_sum_seqlen = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for offset in range(0, len(seqlen_list), batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            min_sum_seqlen = cur_sum_seqlen
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_sum_seqlen = cur_sum_seqlen
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        total_sum_seqlen += cur_sum_seqlen

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    balanced_sum_seqlen_list = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for partition in partitions:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


# [EXPLAIN] `ceildiv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def ceildiv(a, b):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return -(a // -b)


# [EXPLAIN] `roundup_divisible` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def roundup_divisible(a, b):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ((a + b - 1) // b) * b


# [EXPLAIN] `rearrange_micro_batches` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def rearrange_micro_batches(batch, max_token_len, dp_group=None, num_batches_divided_by=None, same_micro_num_in_dp=True, min_num_micro_batch=None):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_seq_len = batch["attention_mask"].shape[-1]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if min_num_micro_batch is not None:
        # used to support pp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dist.is_initialized() and same_micro_num_in_dp:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_micro_batches = num_micro_batches.cpu().item()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_batches_divided_by is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len_effective = seq_len_effective.tolist()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert num_micro_batches <= len(seq_len_effective)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    micro_batches = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for partition in micro_bsz_idx:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        curr_micro_batch = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in partition:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            curr_micro_batch.append(batch[idx : idx + 1])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        curr_micro_batch = torch.cat(curr_micro_batch)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        micro_batches.append(curr_micro_batch)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return micro_batches, micro_bsz_idx


# [EXPLAIN] `get_reverse_idx` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_reverse_idx(idx_map):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reverse_idx_map = copy.deepcopy(idx_map)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, idx in enumerate(idx_map):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reverse_idx_map[idx] = i

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reverse_idx_map
