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

# =============================================================================
# 【このファイルの役割】
# =============================================================================
# - 可変長系列を含むbatchを、各partition / micro-batchの「有効token総数」がなるべく
# 均等になるように分割するためのutilityを定義する。
# - dp_actor.pyのdynamic batch経路では、sample数固定ではなくtoken budget固定で
# micro-batchを作るため、rearrange_micro_batches()が直接呼ばれる。
# - ray_trainer.py側では、global batchをDP rankへ配る際のsequence length balancingにも
# get_seqlen_balanced_partitions()が利用される。
# - このファイルで行うall_reduceはparameter gradientの同期ではない。
# 各DP rankが同じ回数だけforward/backward collectiveへ参加できるよう、
# 「micro-batch数」をrank間でMAXへ揃えるための制御通信である。
# 【重要な用語】
# - B: 現在のlocal batchに含まれるsample数
# - S: padding後の固定sequence length
# - effective sequence length: attention_mask.sum(dim=1)で得るsampleごとの有効token数
# - partition: 元batchのrow indexをまとめたlist
# - token budget: 1 micro-batchが処理する有効token総数の目安max_token_len
# 【dynamic micro-batchの大まかな流れ】
# attention_mask(B,S)
# ↓ rowごとの有効token数を数える
# seq_len_effective(B,)
# ↓ 全token数/max_token_lenからmicro-batch数を決定
# num_micro_batches
# ↓ Karmarkar-Karpでtoken総数が均等になるようrow indexを分配
# micro_bsz_idx = [[row...], [row...], ...]
# ↓ TensorDictをindexで切り出してconcat
# micro_batches
# micro-batch化によりrow順が変わるため、forward結果を元のbatch順に戻すときは
# get_reverse_idx()を使用する。
# =============================================================================
# inverse index mapを作る際、入力listを破壊せず同じ長さの作業領域を作るために使用する。
import copy
# Karmarkar-KarpのStateをpriority queueで管理するための標準library。
import heapq
from typing import List, Tuple

import torch
# DP rank間でmicro-batch数を合わせるall_reduceに使用する。
from torch import distributed as dist


def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    # -------------------------------------------------------------------------
    # Karmarkar-Karp / Largest Differencing Methodをk-way partitionへ拡張した実装。
    # 入力は値そのものではなく、
    # seqlen_list[i] = 元batchのrow iが持つ有効token数
    # と解釈する。
    # 戻り値はtoken長ではなく元row indexのpartitionである。
    # 目的:
    # 各partitionに属するseqlenの合計値をなるべく近づける。
    # equal_size=True:
    # 各partitionのsample数も同数に制約する。
    # equal_size=False:
    # sample数は異なってよく、token合計の均等化を優先する。
    # dynamic micro-batchではこちらを使う。
    # -------------------------------------------------------------------------
    class Set:
        # 1つの最終partition候補を表す内部構造。
        # itemsには[(元row index, sequence length), ...]を保持し、
        # sumにはそのpartitionが担当するtoken総数をキャッシュする。
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            # Karmarkar-KarpのState同士を統合する際、対応するSetの内容を結合する。
            for idx, val in other.items:
                # rowをこのpartitionへ追加し、token総数も同時に更新する。
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            # sorted() / heap比較で安定した順序を作る。
            # まずtoken総数、次にsample数、最後にitems自体を比較する。
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        # Stateはk個のSetをまとめた「部分的なk-way分割案」。
        # mergeを繰り返すことで、最終的に全sampleを含む1つのStateへ集約される。
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            # sets[0]が最大token総数、sets[-1]が最小token総数になるよう降順に保つ。
            self.sets = [Set() for _ in range(k)]
            # equal_size=Falseでは1 itemからStateを作り、
            # equal_size=Trueでは最初からk itemを1つずつ各Setへ配置する。
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            # 内部の(idx, seqlen)からidxだけを抜き出し、外部API用のpartitionへ変換する。
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
            # merge後に再度token総数の降順へ並べる。
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            # このState内で最も重いpartitionと最も軽いpartitionのtoken差。
            # 0に近いほどbalancedである。
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            # heapqは最小値を先にpopするため、比較を意図的に反転している。
            # spreadが大きいStateを「小さい」と判定し、優先的にmerge対象へする。
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            # debug表示用。例: [{10,7},{9,8}]のようにpartition内のtoken長を表示する。
            repr_str = "["
            # 最大のSetと最小のSetを組み合わせる「opposite pairing」を行う。
            # selfは降順、otherも降順なので、other[k-1-i]を選ぶと
            # selfの大きいpartition + otherの小さいpartition
            # selfの小さいpartition + otherの大きいpartition
            # となり、合計値の差を縮めやすい。
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

    # (sequence length, original row index)へ変換してlength昇順に並べる。
    # 元row indexを保持するため、最終partitionから元batchのrowを切り出せる。
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    # Stateを管理するpriority queue。
    states_pq = []
    if equal_size:
        # 各partitionのsample数を完全に同じにするにはBがkで割り切れる必要がある。
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        # sorted listをk個ずつ取り出し、各Stateのk個のSetへ1件ずつ入れる。
        # この初期化により、最終merge後も各partitionのitem数が一致する。
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        # sample数制約がない場合、各sampleを単独Stateとしてheapへ入れる。
        # State内部では1つのSetだけが値を持ち、残りは空Setである。
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # Stateが1個になるまで、spreadが大きい2状態を取り出してopposite pairingでmergeする。
    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    # 最終Stateは全rowをちょうど1回ずつ含むk partitionを持つ。
    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # Karmarkar-Karpとは別の単純greedy partition実装。
    # 現在のget_seqlen_balanced_partitions()はこの関数ではなくkarmarkar_karp()を使用するが、
    # 比較実験やfallback用途として残されている。
    # equal_size=Trueでは各itemへ全長合計より大きいbiasを加える。
    # 1 sample追加するコストがsequence length差より必ず大きくなるため、
    # partition_sums最小を選ぶgreedy処理がsample数も均等化しやすくなる。
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        # 現時点で合計コストが最小のpartitionを線形探索する。
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        # 各partitionのsample数がB/kであることを最終確認する。
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
    # 各partitionに最低1 sample必要なので、partition数はsample数以下でなければならない。
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        # balancing algorithmの出力を検証し、各partition内のrow indexを昇順へ戻す。
        # 昇順化は元batchに近い順序でTensorを切り出し、結果の再現性を保つために有用。
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            # 空partitionがあると、そのmicro-batchでforwardできない。
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        # すべての元row indexが少なくとも1回含まれていることを確認する。
        # algorithm上は重複も生じないため、この集合一致で全row coverageを検証できる。
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    # 実際の分割はKarmarkar-Karpへ委譲する。
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    # add some metrics of seqlen sum on dp ranks
    # balancing前後のtoken負荷差をmetricとして返す。
    # この関数はpartitionを変更せず、性能診断用の数値だけを計算する。
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    # balancing前は元の連続rowを等sample数で各rankへ配ったと仮定する。
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0
    for offset in range(0, len(seqlen_list), batch_size):
        # 元順序でbatch_size件ずつ区切ったときのtoken総数。
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    # balancing後はpartitionで指定されたrowのtoken総数を直接数える。
    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    # minmax_diffが小さいほど、最も重いrankと軽いrankの待ち時間差が小さい。
    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def ceildiv(a, b):
    # ceil(a / b)を整数演算のみで求める。
    # 例: ceildiv(101, 100)=2。dynamic micro-batch数の下限計算に使用する。
    return -(a // -b)


def roundup_divisible(a, b):
    # a以上で最小のbの倍数を返す。
    # pipeline parallelではmicro-batch数をvirtual pipeline sizeの倍数へ揃える必要がある。
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
    # -------------------------------------------------------------------------
    # dynamic micro-batchの中心関数。
    # 固定micro-batch sizeでは「何sample入るか」を固定するが、ここでは
    # 「attention_mask=1のtokenを合計して、おおよそmax_token_len以下になるように」
    # batchを分割する。
    # 注意:
    # num_micro_batchesはtoken budgetから計算した個数であり、Karmarkar-Karpは
    # その個数へtoken負荷を均等化する。各partitionがmax_token_lenを厳密に超えないことを
    # 保証するbin packingではない。総token数に基づく平均的なbudgetである。
    # -------------------------------------------------------------------------
    # attention_maskは(B,S)。Sより小さいmax_token_lenでは、最長1 sampleさえ格納できないため禁止する。
    max_seq_len = batch["attention_mask"].shape[-1]
    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    # 各rowのpaddingを除いた有効token数を求める。
    # shape: (B,S) --sum(dim=1)--> (B,)
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    # local batch全体の有効token総数。
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    # ceil(total/max)個あれば、平均token数はmax_token_len以下になる。
    # ただし1 micro-batchには最低1 sample必要なのでBを上限にする。
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        # pipeline parallel等が最低micro-batch数を要求する場合に下限を上げる。
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        # ---------------------------------------------------------------------
        # 【重要】これはgradient all-reduceではない。
        # rankごとにlocal batchのsequence length分布が異なると、
        # rank 0は3 micro-batch、rank 1は4 micro-batchとなる可能性がある。
        # FSDP/DDPのbackward collectiveは各rankが同じ順番・同じ回数参加する必要があるため、
        # micro-batch回数が違うと片方だけ次のcollectiveを呼び、hangする可能性がある。
        # そこで各rankのnum_micro_batchesをMAX all-reduceし、最も多いrankへ全rankを合わせる。
        # dp_group=Noneならdefault process groupが使われる。
        # ---------------------------------------------------------------------
        num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        # 後段はPythonのrange/partition数として使うためCPU scalarへ戻す。
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        # virtual pipeline parallelのschedule条件を満たすよう、micro-batch数を指定値の倍数へ切り上げる。
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    # partition algorithmはPython listを受け取るためTensorから変換する。
    seq_len_effective = seq_len_effective.tolist()
    # 空micro-batchを作らない設計なので、micro-batch数はsample数以下である必要がある。
    assert num_micro_batches <= len(seq_len_effective)

    # sample数を揃える必要はないためequal_size=False。
    # 例: length=[100,90,10,10], k=2なら、[100,10]と[90,10]のようにtoken合計を揃える。
    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    micro_batches = []

    # partitionは元batchのrow index list。
    # 各rowをbatch dimensionを保ったまま[idx:idx+1]で取り出し、TensorDictとしてconcatする。
    for partition in micro_bsz_idx:
        curr_micro_batch = []
        for idx in partition:
            curr_micro_batch.append(batch[idx : idx + 1])
        curr_micro_batch = torch.cat(curr_micro_batch)

        micro_batches.append(curr_micro_batch)

    # micro_batchesはbalanced orderであり元row順ではない。
    # micro_bsz_idxを同時に返し、dp_actor.compute_log_prob等が結果を元順へ復元できるようにする。
    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    # -------------------------------------------------------------------------
    # dynamic batchで並べ替えた出力を元row順へ戻すinverse permutationを作る。
    # 例:
    # micro-batchをflattenした元row index順 = idx_map = [2, 0, 3, 1]
    # output_balanced[0]は元row 2の結果
    # output_balanced[1]は元row 0の結果
    # inverseは[1, 3, 0, 2]となり、
    # output_original = output_balanced[inverse]
    # とすると元row 0,1,2,3の順に戻る。
    # -------------------------------------------------------------------------
    # 同じ長さのlistを確保する目的でdeepcopyする。入力idx_map自体は変更しない。
    reverse_idx_map = copy.deepcopy(idx_map)

    # idx_map[i]=元row番号なので、inverse[元row番号]=balanced出力上の位置iを記録する。
    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
