# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""CPU-only equivalence gates for the rollout speedup mechanisms.

Covers the pure-logic pieces of the A/C/D/E optimizations:

  * compute_log_prob_with_prefetch (A): merging prefetched per-row log probs
    with the normally computed remainder must equal computing every row.
  * prefetch_env_reset / _reset_envs (C): the prefetched reset is consumed
    exactly once, matched by env identity and kwargs, and a kwargs mismatch
    fails loudly instead of silently double-resetting.
  * SearchToolGroup query cache (D): enabled cache returns the identical
    result text without a second API call; disabled cache always calls.
  * compact record (E): dropping active_masks=False rows from
    total_batch_list produces the identical effective batch from
    gather_rollout_data-style filtering (prefix property).

Run:  python tests/ray_cpu/test_rollout_speedup_mechanisms.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
os.environ.setdefault("ROLLOUT_PREFETCH_LOGPROB", "1")

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto  # noqa: E402
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout.utils import compute_log_prob_with_prefetch  # noqa: E402
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.multi_turn_rollout.rollout_loop import (  # noqa: E402
    TrajectoryCollector,
    _env_kwargs_equal,
)


# --------------------------------------------------------------------------- #
# A: compute_log_prob_with_prefetch merge equivalence
# --------------------------------------------------------------------------- #
# [EXPLAIN] `FakeWorkerGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FakeWorkerGroup:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Deterministic stand-in for actor_rollout_wg.compute_log_prob.

    Produces per-row values that depend only on the row's input_ids content, so
    a row computed inside any sub-batch (prefetch chunk or trainer remainder)
    yields the same result as inside the full batch — exactly the per-row
    independence property the real rmpad compute_log_prob has.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, world_size, response_length):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.world_size = world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_length = response_length
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.calls = []

    # [EXPLAIN] `compute_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_log_prob(self, data: DataProto) -> DataProto:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(data) % self.world_size == 0, "DP dispatch needs a divisible batch"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.calls.append(len(data))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seeds = data.batch["input_ids"].float().sum(dim=-1, keepdim=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        offsets = torch.arange(self.response_length, dtype=torch.float32)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto.from_dict(
            tensors={
                "old_log_probs": -(seeds + offsets) / 100.0,
                "entropys": (seeds - offsets) / 200.0,
            },
            meta_info={"temperature": 1.0},
        )


# [EXPLAIN] `_make_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _make_batch(num_rows, seq_len, response_length, seed=0):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generator = torch.Generator().manual_seed(seed)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(1, 1000, (num_rows, seq_len), generator=generator)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(seq_len).expand(num_rows, seq_len).contiguous(),
            "responses": input_ids[:, -response_length:].contiguous(),
        },
        non_tensors={
            "traj_uid": np.array([f"traj-{i % 5}" for i in range(num_rows)], dtype=object),
            "turn_step": np.array([i // 5 for i in range(num_rows)], dtype=object),
        },
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return batch


# [EXPLAIN] `test_prefetch_merge_equivalence` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prefetch_merge_equivalence():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size, response_length = 2, 6
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(num_rows=10, seq_len=12, response_length=response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reference_wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected = reference_wg.compute_log_prob(batch)

    # Prefetch rows 0..3 and 7 the same way the rollout loop does: per-row calls
    # through the same compute function.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetch_wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetched = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetch_idx = [0, 1, 2, 3, 7]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sub = batch.select_idxs(prefetch_idx)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sub_padded, pad_size = pad_dataproto_to_divisor(sub, world_size)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = unpad_dataproto(prefetch_wg.compute_log_prob(sub_padded), pad_size=pad_size)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for j, i in enumerate(prefetch_idx):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prefetched[key] = (out.batch["old_log_probs"][j], out.batch["entropys"][j])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merge_wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merged = compute_log_prob_with_prefetch(merge_wg, batch, prefetched, temperature=1.0)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(merged.batch["old_log_probs"], expected.batch["old_log_probs"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(merged.batch["entropys"], expected.batch["entropys"])
    # Only the 5 missing rows (padded to a multiple of world_size) were computed.
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert merge_wg.calls == [6], merge_wg.calls
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("A: prefetch merge equivalence OK")


# [EXPLAIN] `test_prefetch_merge_all_and_none` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prefetch_merge_all_and_none():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size, response_length = 2, 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(num_rows=4, seq_len=8, response_length=response_length, seed=1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reference = FakeWorkerGroup(world_size, response_length).compute_log_prob(batch)

    # No prefetch -> plain call.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = compute_log_prob_with_prefetch(wg, batch, {}, temperature=1.0)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(out.batch["old_log_probs"], reference.batch["old_log_probs"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert wg.calls == [4]

    # Everything prefetched -> no worker call at all.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetched = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(4):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prefetched[key] = (
            reference.batch["old_log_probs"][i],
            reference.batch["entropys"][i],
        )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    out = compute_log_prob_with_prefetch(wg, batch, prefetched, temperature=1.0)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(out.batch["old_log_probs"], reference.batch["old_log_probs"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(out.batch["entropys"], reference.batch["entropys"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert wg.calls == []
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("A: all/none prefetched edge cases OK")


# [EXPLAIN] `test_prefetch_merge_duplicated_rows` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_prefetch_merge_duplicated_rows():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """adjust_batch(mode='copy') duplicates rows; duplicates share a key and must
    both be filled from the same prefetched value."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size, response_length = 2, 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch = _make_batch(num_rows=4, seq_len=8, response_length=response_length, seed=2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dup = batch.select_idxs([0, 1, 2, 3, 1, 2])  # rows 1 and 2 duplicated
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reference = FakeWorkerGroup(world_size, response_length).compute_log_prob(dup)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prefetched = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in (1, 2):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prefetched[key] = (
            reference.batch["old_log_probs"][i],
            reference.batch["entropys"][i],
        )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merged = compute_log_prob_with_prefetch(wg, dup, prefetched, temperature=1.0)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(merged.batch["old_log_probs"], reference.batch["old_log_probs"])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(merged.batch["entropys"], reference.batch["entropys"])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("A: duplicated-row (adjust_batch copy) merge OK")


# --------------------------------------------------------------------------- #
# A: rollout-side pending pool -> prefetch chunking
# --------------------------------------------------------------------------- #
# [EXPLAIN] `test_rollout_prefetch_pending_pool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_rollout_prefetch_pending_pool():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector._logprob_pending = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector._prefetched_log_probs = {}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size, response_length, seq_len = 2, 4, 8
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = FakeWorkerGroup(world_size, response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rows = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(5):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = torch.full((seq_len,), i + 1, dtype=torch.long)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "position_ids": torch.arange(seq_len),
                "responses": input_ids[-response_length:].clone(),
            }
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        collector._logprob_pending.append(((f"t-{i}", 0), rows[-1]))

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector._prefetch_pending_log_probs(wg)
    # Default chunk (64) covers all 5 rows; padded to a multiple of world_size.
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert wg.calls == [6], wg.calls
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not collector._logprob_pending
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert set(collector._prefetched_log_probs) == {(f"t-{i}", 0) for i in range(5)}

    # The stored per-row values must equal a batched computation of those rows.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full = DataProto.from_dict(
        tensors={
            name: torch.stack([row[name] for row in rows])
            for name in ("input_ids", "attention_mask", "position_ids", "responses")
        }
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_padded, pad_size = pad_dataproto_to_divisor(full, world_size)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected = unpad_dataproto(
        FakeWorkerGroup(world_size, response_length).compute_log_prob(full_padded), pad_size=pad_size
    )
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(5):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs, entropys = collector._prefetched_log_probs[(f"t-{i}", 0)]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.equal(log_probs, expected.batch["old_log_probs"][i])
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.equal(entropys, expected.batch["entropys"][i])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    taken = collector.take_prefetched_log_probs()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(taken) == 5 and not collector._prefetched_log_probs
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("A: rollout pending pool + take OK")


# --------------------------------------------------------------------------- #
# C: env reset prefetch consume / mismatch semantics
# --------------------------------------------------------------------------- #
# [EXPLAIN] `FakeEnvs` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FakeEnvs:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reset_count = 0

    # [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset(self, kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.reset_count += 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ({"text": [f"obs-{self.reset_count}"]}, [{"seq": self.reset_count}])


# [EXPLAIN] `test_env_reset_prefetch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_env_reset_prefetch():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector._env_reset_prefetch = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    collector._env_reset_executor = None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    envs = FakeEnvs()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kwargs = np.array([{"task_name": "search", "question": "q1"}], dtype=object)

    # Prefetched reset is consumed by the next matching _reset_envs call and the
    # env is reset exactly once.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector.prefetch_env_reset(envs, kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs, infos = collector._reset_envs(envs, kwargs)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert envs.reset_count == 1 and infos[0]["seq"] == 1
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert collector._env_reset_prefetch is None

    # Without a pending prefetch the call resets synchronously.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs, infos = collector._reset_envs(envs, kwargs)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert envs.reset_count == 2

    # A different env manager (validation) must not consume the train prefetch.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    other_envs = FakeEnvs()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector.prefetch_env_reset(envs, kwargs)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector._env_reset_prefetch["future"].result()  # let the background reset land
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector._reset_envs(other_envs, kwargs)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert other_envs.reset_count == 1 and envs.reset_count == 3  # prefetch issued
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert collector._env_reset_prefetch is not None
    # ...and the train rollout still consumes it afterwards.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector._reset_envs(envs, kwargs)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert envs.reset_count == 3  # consumed, no extra reset

    # Mismatched kwargs must fail loudly (a silent re-reset would advance
    # stateful env schedules a second time).
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    collector.prefetch_env_reset(envs, kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mismatch = np.array([{"task_name": "search", "question": "DIFFERENT"}], dtype=object)
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        collector._reset_envs(envs, mismatch)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except RuntimeError:
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise AssertionError("mismatched env_kwargs must raise")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("C: env reset prefetch semantics OK")


# [EXPLAIN] `test_env_kwargs_equal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_env_kwargs_equal():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = [{"q": "x", "arr": np.array([1, 2])}]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    b = [{"q": "x", "arr": np.array([1, 2])}]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert _env_kwargs_equal(a, b)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert _env_kwargs_equal(None, None)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not _env_kwargs_equal(a, None)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not _env_kwargs_equal(a, [{"q": "y", "arr": np.array([1, 2])}])
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not _env_kwargs_equal(a, [{"q": "x", "arr": np.array([1, 3])}])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    repeated = np.repeat(np.array([{"q": "x"}], dtype=object), 3)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert _env_kwargs_equal(repeated, list(repeated))
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("C: _env_kwargs_equal OK")


# --------------------------------------------------------------------------- #
# E: compact record produces the identical effective batch
# --------------------------------------------------------------------------- #
# [EXPLAIN] `test_compact_record_equivalence` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_compact_record_equivalence():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Simulate per-turn recording with and without inactive rows and check the
    gather-side filtering (active rows only, enumerate-based turn_step) agrees."""

    # [EXPLAIN] `simulate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def simulate(compact):
        # traj 0 finishes at turn 1, traj 1 runs all 4 turns.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        done_at = {0: 1, 1: 99}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total = [[], []]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for turn in range(4):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(2):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                active = turn <= done_at[i]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if compact and not active:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                total[i].append({"traj": i, "turn": turn, "active_masks": active})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        effective = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(2):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for step_idx, row in enumerate(total[i]):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if row["active_masks"]:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    effective.append((row["traj"], row["turn"], step_idx))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return effective

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert simulate(compact=True) == simulate(compact=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("E: compact record equivalence OK")


# --------------------------------------------------------------------------- #
# D: search query cache
# --------------------------------------------------------------------------- #
# [EXPLAIN] `test_search_query_cache` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_search_query_cache():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    os.environ["SEARCH_QUERY_CACHE"] = "1"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    search_mod_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "agent_system",
            "environments",
            "env_package",
            "search",
            "third_party",
            "skyrl_gym",
            "tools",
            "search.py",
        )
    )
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import importlib.util
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import types

    # Importing the search package pulls `gym` (unneeded for the cache logic);
    # stub it when absent so this test stays CPU/deps-light.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "gym" not in sys.modules:
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import gym  # noqa: F401
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except ImportError:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gym_stub = types.ModuleType("gym")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gym_stub.Env = object
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            sys.modules["gym"] = gym_stub

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    spec = importlib.util.spec_from_file_location("search_tool_cache_test", search_mod_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module = importlib.util.module_from_spec(spec)
    # The module imports its ToolGroup base via the package path; make sure the
    # repo root import used inside the module resolves.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    spec.loader.exec_module(module)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    calls = {"n": 0}

    # [EXPLAIN] `fake_call_search_api` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fake_call_search_api(**kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        calls["n"] += 1
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {"result": [[{"document": {"contents": f"doc for {kwargs['query']}"}}]]}, None

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    module.call_search_api = fake_call_search_api
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    group = module.SearchToolGroup(search_url="http://fake:8000/retrieve", log_requests=False)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first = group.search("who wrote hamlet")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    second = group.search("who wrote hamlet")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert first == second
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert calls["n"] == 1, f"expected 1 API call with cache on, got {calls['n']}"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    third = group.search("different query")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert calls["n"] == 2 and third != first
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("D: search query cache OK")


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_prefetch_merge_equivalence()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_prefetch_merge_all_and_none()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_prefetch_merge_duplicated_rows()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_rollout_prefetch_pending_pool()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_env_reset_prefetch()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_env_kwargs_equal()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_compact_record_equivalence()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_search_query_cache()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("\nALL ROLLOUT SPEEDUP MECHANISM TESTS PASSED")
