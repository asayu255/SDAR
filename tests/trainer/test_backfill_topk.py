"""Unit tests for the SFT->off-policy top-k backfill (CPU-only; no Ray/workers).

Covered:
* ``_chunk_bounds`` splits a row range into in-order, gap-free, non-overlapping
  blocks of at most chunk_size;
* the chunked assembly appends ``teacher_topk_logprobs`` / ``teacher_topk_ids``
  of shape (N, resp, k) while preserving row order and the existing fields
  (``responses`` / ``traj_uid``).

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.main_opd_offpolicy_backfill import _chunk_bounds
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def test_chunk_bounds_covers_range_in_order():
    assert _chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert _chunk_bounds(8, 4) == [(0, 4), (4, 8)]
    assert _chunk_bounds(3, 10) == [(0, 3)]
    assert _chunk_bounds(0, 4) == []
    # Union covers exactly range(n) with no gaps / overlaps.
    covered = []
    for s, e in _chunk_bounds(25, 7):
        covered += list(range(s, e))
    assert covered == list(range(25))


def _sft_proto(n, resp_len=4, extra_prompt=2):
    """SFT teacher data: token sequences only, NO teacher top-k fields.
    responses[i] == i so row order can be verified after chunked assembly."""
    return DataProto.from_dict(
        tensors={
            "responses": torch.arange(n).view(n, 1).repeat(1, resp_len),
            "input_ids": torch.zeros((n, resp_len + extra_prompt), dtype=torch.long),
            "attention_mask": torch.ones((n, resp_len + extra_prompt), dtype=torch.long),
            "position_ids": torch.zeros((n, resp_len + extra_prompt), dtype=torch.long),
        },
        non_tensors={
            "task_name": np.array(["alfworld"] * n, dtype=object),
            "traj_uid": np.array([f"t{i}" for i in range(n)], dtype=object),
        },
    )


def test_backfill_appends_topk_preserving_rows_and_order():
    n, resp_len, k = 25, 4, 3
    data = _sft_proto(n, resp_len)
    assert "teacher_topk_logprobs" not in data.batch

    # Mimic the runner's chunked scoring: each chunk's fake top-k encodes the
    # global row index, so a correct cat preserves order.
    tlp_list, tid_list = [], []
    for s, e in _chunk_bounds(n, 7):
        idx = torch.arange(s, e).view(e - s, 1, 1).float().repeat(1, resp_len, k)
        tlp_list.append(idx)
        tid_list.append(idx.long())
    data.batch["teacher_topk_logprobs"] = torch.cat(tlp_list, dim=0)
    data.batch["teacher_topk_ids"] = torch.cat(tid_list, dim=0)

    assert data.batch["teacher_topk_logprobs"].shape == (n, resp_len, k)
    assert data.batch["teacher_topk_ids"].shape == (n, resp_len, k)
    assert len(data) == n
    for i in range(n):
        assert torch.all(data.batch["teacher_topk_logprobs"][i] == i)  # order preserved
        assert torch.all(data.batch["responses"][i] == i)              # responses unchanged
        assert data.non_tensor_batch["traj_uid"][i] == f"t{i}"         # traj_uid unchanged
