# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.multiprocessing as mp

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import create_random_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import ceildiv, get_reverse_idx, rearrange_micro_batches


# [EXPLAIN] `test_seqlen_balancing` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_seqlen_balancing():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = create_random_mask(input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataproto = DataProto.from_single_dict(data)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    micro_batches, micro_bsz_idx_lst = rearrange_micro_batches(dataproto.batch, max_token_len=300)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = torch.cat(micro_batches)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    micro_bsz_idx = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for idx in micro_bsz_idx_lst:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        micro_bsz_idx.extend(idx)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reverse_idx_map = get_reverse_idx(micro_bsz_idx)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reverse_idx_map = torch.tensor(reverse_idx_map)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    new_batch = batch[reverse_idx_map]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(new_batch, dataproto.batch)


# [EXPLAIN] `_worker` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _worker(rank, world_size, init_method, max_token_len, use_same_dp, min_mb):
    # 1) init process group & CUDA
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.set_device(rank)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    # 2) build a small random batch (each rank different length to force mismatch)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(42 + rank)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(0, 10, (20 + rank * 5, 100), device=f"cuda:{rank}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0.1,
        max_ratio_of_valid_token=0.9,
        min_ratio_of_valid_token=0.5,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp = {"input_ids": input_ids, "attention_mask": attention_mask}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    proto = DataProto.from_single_dict(dp)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = proto.batch

    # 3) call rearrange_micro_batches with one of the two params under test
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    micros, idx_lst = rearrange_micro_batches(
        batch,
        max_token_len=max_token_len,
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=use_same_dp,
        min_num_micro_batch=min_mb,
    )

    # 4) check the enforced counts
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_seqlen = seq_len_effective.sum().item()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if min_mb is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected = max(local, min_mb)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(micros) == expected
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if use_same_dp:
        # gather all local_counts
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        counts = [torch.zeros(1, device=f"cuda:{rank}") for _ in range(world_size)]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        counts[rank].fill_(local)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.all_gather(counts, counts[rank])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected = max(int(c.item()) for c in counts)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(micros) == expected
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # if neither, we get the local natural count
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(micros) == local

    # 5) reconstruction sanity: concat→reverse_idx→orig
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    flat = torch.cat(micros, dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for sub in idx_lst:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        idx.extend(sub)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inv = get_reverse_idx(idx)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inv = torch.tensor(inv, device=flat.device)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reconstructed = flat[inv]
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(reconstructed, batch)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dist.destroy_process_group()


# [EXPLAIN] `test_seqlen_balancing_distributed_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_seqlen_balancing_distributed_params(tmp_path):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_file = tmp_path / "dist_init"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    init_file.write_text("")  # empty file
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    init_method = f"file://{init_file}"

    # test min_num_micro_batch only
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, False, 4),
        nprocs=world_size,
        join=True,
    )

    # test same_micro_num_in_dp only
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, True, None),
        nprocs=world_size,
        join=True,
    )
