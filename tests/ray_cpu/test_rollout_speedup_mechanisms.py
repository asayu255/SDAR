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
"""CPU-only equivalence gates for the rollout speedup mechanisms.

Covers the pure-logic pieces of the A/C/E optimizations:

  * compute_log_prob_with_prefetch (A): merging prefetched per-row log probs
    with the normally computed remainder must equal computing every row.
  * prefetch_env_reset / _reset_envs (C): the prefetched reset is consumed
    exactly once, matched by env identity and kwargs, and a kwargs mismatch
    fails loudly instead of silently double-resetting.
  * compact record (E): dropping active_masks=False rows from
    total_batch_list produces the identical effective batch from
    gather_rollout_data-style filtering (prefix property).

Run:  python tests/ray_cpu/test_rollout_speedup_mechanisms.py
"""

import os
import sys

os.environ.setdefault("ROLLOUT_PREFETCH_LOGPROB", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto  # noqa: E402
from agent_system.multi_turn_rollout.utils import compute_log_prob_with_prefetch  # noqa: E402
from agent_system.multi_turn_rollout.rollout_loop import (  # noqa: E402
    TrajectoryCollector,
    _env_kwargs_equal,
)


# --------------------------------------------------------------------------- #
# A: compute_log_prob_with_prefetch merge equivalence
# --------------------------------------------------------------------------- #
class FakeWorkerGroup:
    """Deterministic stand-in for actor_rollout_wg.compute_log_prob.

    Produces per-row values that depend only on the row's input_ids content, so
    a row computed inside any sub-batch (prefetch chunk or trainer remainder)
    yields the same result as inside the full batch — exactly the per-row
    independence property the real rmpad compute_log_prob has.
    """

    def __init__(self, world_size, response_length):
        self.world_size = world_size
        self.response_length = response_length
        self.calls = []

    def compute_log_prob(self, data: DataProto) -> DataProto:
        assert len(data) % self.world_size == 0, "DP dispatch needs a divisible batch"
        self.calls.append(len(data))
        seeds = data.batch["input_ids"].float().sum(dim=-1, keepdim=True)
        offsets = torch.arange(self.response_length, dtype=torch.float32)
        return DataProto.from_dict(
            tensors={
                "old_log_probs": -(seeds + offsets) / 100.0,
                "entropys": (seeds - offsets) / 200.0,
            },
            meta_info={"temperature": 1.0},
        )


def _make_batch(num_rows, seq_len, response_length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, 1000, (num_rows, seq_len), generator=generator)
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
    return batch


def test_prefetch_merge_equivalence():
    world_size, response_length = 2, 6
    batch = _make_batch(num_rows=10, seq_len=12, response_length=response_length)
    reference_wg = FakeWorkerGroup(world_size, response_length)
    expected = reference_wg.compute_log_prob(batch)

    # Prefetch rows 0..3 and 7 the same way the rollout loop does: per-row calls
    # through the same compute function.
    prefetch_wg = FakeWorkerGroup(world_size, response_length)
    prefetched = {}
    prefetch_idx = [0, 1, 2, 3, 7]
    sub = batch.select_idxs(prefetch_idx)
    sub_padded, pad_size = pad_dataproto_to_divisor(sub, world_size)
    out = unpad_dataproto(prefetch_wg.compute_log_prob(sub_padded), pad_size=pad_size)
    for j, i in enumerate(prefetch_idx):
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        prefetched[key] = (out.batch["old_log_probs"][j], out.batch["entropys"][j])

    merge_wg = FakeWorkerGroup(world_size, response_length)
    merged = compute_log_prob_with_prefetch(merge_wg, batch, prefetched, temperature=1.0)

    assert torch.equal(merged.batch["old_log_probs"], expected.batch["old_log_probs"])
    assert torch.equal(merged.batch["entropys"], expected.batch["entropys"])
    # Only the 5 missing rows (padded to a multiple of world_size) were computed.
    assert merge_wg.calls == [6], merge_wg.calls
    print("A: prefetch merge equivalence OK")


def test_prefetch_merge_all_and_none():
    world_size, response_length = 2, 4
    batch = _make_batch(num_rows=4, seq_len=8, response_length=response_length, seed=1)
    reference = FakeWorkerGroup(world_size, response_length).compute_log_prob(batch)

    # No prefetch -> plain call.
    wg = FakeWorkerGroup(world_size, response_length)
    out = compute_log_prob_with_prefetch(wg, batch, {}, temperature=1.0)
    assert torch.equal(out.batch["old_log_probs"], reference.batch["old_log_probs"])
    assert wg.calls == [4]

    # Everything prefetched -> no worker call at all.
    prefetched = {}
    for i in range(4):
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        prefetched[key] = (
            reference.batch["old_log_probs"][i],
            reference.batch["entropys"][i],
        )
    wg = FakeWorkerGroup(world_size, response_length)
    out = compute_log_prob_with_prefetch(wg, batch, prefetched, temperature=1.0)
    assert torch.equal(out.batch["old_log_probs"], reference.batch["old_log_probs"])
    assert torch.equal(out.batch["entropys"], reference.batch["entropys"])
    assert wg.calls == []
    print("A: all/none prefetched edge cases OK")


def test_prefetch_merge_duplicated_rows():
    """adjust_batch(mode='copy') duplicates rows; duplicates share a key and must
    both be filled from the same prefetched value."""
    world_size, response_length = 2, 4
    batch = _make_batch(num_rows=4, seq_len=8, response_length=response_length, seed=2)
    dup = batch.select_idxs([0, 1, 2, 3, 1, 2])  # rows 1 and 2 duplicated
    reference = FakeWorkerGroup(world_size, response_length).compute_log_prob(dup)

    prefetched = {}
    for i in (1, 2):
        key = (batch.non_tensor_batch["traj_uid"][i], int(batch.non_tensor_batch["turn_step"][i]))
        prefetched[key] = (
            reference.batch["old_log_probs"][i],
            reference.batch["entropys"][i],
        )
    wg = FakeWorkerGroup(world_size, response_length)
    merged = compute_log_prob_with_prefetch(wg, dup, prefetched, temperature=1.0)
    assert torch.equal(merged.batch["old_log_probs"], reference.batch["old_log_probs"])
    assert torch.equal(merged.batch["entropys"], reference.batch["entropys"])
    print("A: duplicated-row (adjust_batch copy) merge OK")


# --------------------------------------------------------------------------- #
# A: rollout-side pending pool -> prefetch chunking
# --------------------------------------------------------------------------- #
def test_rollout_prefetch_pending_pool():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._logprob_pending = []
    collector._prefetched_log_probs = {}

    world_size, response_length, seq_len = 2, 4, 8
    wg = FakeWorkerGroup(world_size, response_length)
    rows = []
    for i in range(5):
        input_ids = torch.full((seq_len,), i + 1, dtype=torch.long)
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "position_ids": torch.arange(seq_len),
                "responses": input_ids[-response_length:].clone(),
            }
        )
        collector._logprob_pending.append(((f"t-{i}", 0), rows[-1]))

    collector._prefetch_pending_log_probs(wg)
    # Default chunk (64) covers all 5 rows; padded to a multiple of world_size.
    assert wg.calls == [6], wg.calls
    assert not collector._logprob_pending
    assert set(collector._prefetched_log_probs) == {(f"t-{i}", 0) for i in range(5)}

    # The stored per-row values must equal a batched computation of those rows.
    full = DataProto.from_dict(
        tensors={
            name: torch.stack([row[name] for row in rows])
            for name in ("input_ids", "attention_mask", "position_ids", "responses")
        }
    )
    full_padded, pad_size = pad_dataproto_to_divisor(full, world_size)
    expected = unpad_dataproto(
        FakeWorkerGroup(world_size, response_length).compute_log_prob(full_padded), pad_size=pad_size
    )
    for i in range(5):
        log_probs, entropys = collector._prefetched_log_probs[(f"t-{i}", 0)]
        assert torch.equal(log_probs, expected.batch["old_log_probs"][i])
        assert torch.equal(entropys, expected.batch["entropys"][i])

    taken = collector.take_prefetched_log_probs()
    assert len(taken) == 5 and not collector._prefetched_log_probs
    print("A: rollout pending pool + take OK")


def test_queue_row_for_prefetch_routes_by_mechanism():
    """Rows enter exactly the pending pools whose mechanism is on.

    The loop calls this once per recorded active row (record time, not
    trajectory end), so the helper is also the single place that fixes the
    queue-entry key format the trainer matches on: (str(traj_uid), int(step)).
    """
    row = {"input_ids": torch.zeros(4, dtype=torch.long)}
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._logprob_prefetch_enabled = False
    collector._teacher_prefetch_fn = None
    collector._logprob_pending = []
    collector._teacher_pending = []

    # Both mechanisms off: recording a row queues nothing.
    collector._queue_row_for_prefetch("t-0", 0, row)
    assert not collector._logprob_pending and not collector._teacher_pending

    # Teacher prefetch only (the pure-OPD configuration).
    collector._teacher_prefetch_fn = lambda chunk: {}
    collector._queue_row_for_prefetch(np.str_("t-0"), np.int64(3), row)
    assert not collector._logprob_pending
    assert collector._teacher_pending == [(("t-0", 3), row)]
    key = collector._teacher_pending[0][0]
    assert type(key[0]) is str and type(key[1]) is int  # trainer-side dict keys

    # Both on: one call feeds both pools with the same entry.
    collector._logprob_prefetch_enabled = True
    collector._queue_row_for_prefetch("t-1", 0, row)
    assert collector._logprob_pending == [(("t-1", 0), row)]
    assert collector._teacher_pending[-1] == (("t-1", 0), row)
    print("A: per-turn queue routing OK")


# --------------------------------------------------------------------------- #
# C: env reset prefetch consume / mismatch semantics
# --------------------------------------------------------------------------- #
class FakeEnvs:
    def __init__(self):
        self.reset_count = 0

    def reset(self, kwargs):
        self.reset_count += 1
        return ({"text": [f"obs-{self.reset_count}"]}, [{"seq": self.reset_count}])


def test_env_reset_prefetch():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._env_reset_prefetch = None
    collector._env_reset_executor = None

    envs = FakeEnvs()
    kwargs = np.array([{"task_name": "search", "question": "q1"}], dtype=object)

    # Prefetched reset is consumed by the next matching _reset_envs call and the
    # env is reset exactly once.
    collector.prefetch_env_reset(envs, kwargs)
    obs, infos = collector._reset_envs(envs, kwargs)
    assert envs.reset_count == 1 and infos[0]["seq"] == 1
    assert collector._env_reset_prefetch is None

    # Without a pending prefetch the call resets synchronously.
    obs, infos = collector._reset_envs(envs, kwargs)
    assert envs.reset_count == 2

    # A different env manager (validation) must not consume the train prefetch.
    other_envs = FakeEnvs()
    collector.prefetch_env_reset(envs, kwargs)
    collector._env_reset_prefetch["future"].result()  # let the background reset land
    collector._reset_envs(other_envs, kwargs)
    assert other_envs.reset_count == 1 and envs.reset_count == 3  # prefetch issued
    assert collector._env_reset_prefetch is not None
    # ...and the train rollout still consumes it afterwards.
    collector._reset_envs(envs, kwargs)
    assert envs.reset_count == 3  # consumed, no extra reset

    # Mismatched kwargs must fail loudly (a silent re-reset would advance
    # stateful env schedules a second time).
    collector.prefetch_env_reset(envs, kwargs)
    mismatch = np.array([{"task_name": "search", "question": "DIFFERENT"}], dtype=object)
    try:
        collector._reset_envs(envs, mismatch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched env_kwargs must raise")
    print("C: env reset prefetch semantics OK")


def test_env_kwargs_equal():
    a = [{"q": "x", "arr": np.array([1, 2])}]
    b = [{"q": "x", "arr": np.array([1, 2])}]
    assert _env_kwargs_equal(a, b)
    assert _env_kwargs_equal(None, None)
    assert not _env_kwargs_equal(a, None)
    assert not _env_kwargs_equal(a, [{"q": "y", "arr": np.array([1, 2])}])
    assert not _env_kwargs_equal(a, [{"q": "x", "arr": np.array([1, 3])}])
    repeated = np.repeat(np.array([{"q": "x"}], dtype=object), 3)
    assert _env_kwargs_equal(repeated, list(repeated))
    print("C: _env_kwargs_equal OK")


# --------------------------------------------------------------------------- #
# E: compact record produces the identical effective batch
# --------------------------------------------------------------------------- #
def test_compact_record_equivalence():
    """Simulate per-turn recording with and without inactive rows and check the
    gather-side filtering (active rows only, enumerate-based turn_step) agrees."""

    def simulate(compact):
        # traj 0 finishes at turn 1, traj 1 runs all 4 turns.
        done_at = {0: 1, 1: 99}
        total = [[], []]
        for turn in range(4):
            for i in range(2):
                active = turn <= done_at[i]
                if compact and not active:
                    continue
                total[i].append({"traj": i, "turn": turn, "active_masks": active})
        effective = []
        for i in range(2):
            for step_idx, row in enumerate(total[i]):
                if row["active_masks"]:
                    effective.append((row["traj"], row["turn"], step_idx))
        return effective

    assert simulate(compact=True) == simulate(compact=False)
    print("E: compact record equivalence OK")


if __name__ == "__main__":
    test_prefetch_merge_equivalence()
    test_prefetch_merge_all_and_none()
    test_prefetch_merge_duplicated_rows()
    test_rollout_prefetch_pending_pool()
    test_queue_row_for_prefetch_routes_by_mechanism()
    test_env_reset_prefetch()
    test_env_kwargs_equal()
    test_compact_record_equivalence()
    print("\nALL ROLLOUT SPEEDUP MECHANISM TESTS PASSED")
