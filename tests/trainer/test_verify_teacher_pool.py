"""The verifier has to fail on the corruptions it claims to catch.

A checker that passes everything is worse than no checker: it is a reason not to
look. Each test breaks exactly one invariant in an otherwise-valid shard and
asserts the script exits non-zero naming that invariant.
"""

import os
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "verify_teacher_pool.py")

PROMPT_LEN, RESP_LEN = 12, 5


def _valid(task="webshop", n_traj=3, turns=2, resp_used=3, prompt_used=7):
    """A shard laid out exactly as Stage 1 writes one."""
    n = n_traj * turns
    seq = PROMPT_LEN + RESP_LEN

    attention_mask = torch.zeros((n, seq), dtype=torch.long)
    attention_mask[:, PROMPT_LEN - prompt_used : PROMPT_LEN] = 1   # left-padded prompt
    attention_mask[:, PROMPT_LEN : PROMPT_LEN + resp_used] = 1     # right-padded response

    input_ids = torch.randint(1, 900, (n, seq), dtype=torch.long)
    responses = input_ids[:, -RESP_LEN:].clone()
    # Built the way vllm_rollout_spmd.py builds it: the prompt half follows the
    # mask, the response half counts straight on through its right padding.
    prompt_pos = torch.clamp(attention_mask[:, :PROMPT_LEN].cumsum(dim=1) - 1, min=0)
    position_ids = torch.cat(
        [prompt_pos, prompt_pos[:, -1:] + torch.arange(1, RESP_LEN + 1).unsqueeze(0)],
        dim=1,
    )

    uids = []
    for j in range(n_traj):
        uids += [f"{task}-{j}"] * turns
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        },
        non_tensors={
            "task_name": np.array([task] * n, dtype=object),
            "traj_uid": np.array(uids, dtype=object),
        },
    )


def _run(pool):
    return subprocess.run(
        [sys.executable, _SCRIPT, str(pool), "--shards", "1"],
        capture_output=True, text=True,
    )


def _pool(tmp_path, proto, name="webshop_0000.pt", sub="pool"):
    root = tmp_path / sub
    root.mkdir(exist_ok=True)
    proto.save_to_disk(os.path.join(root, name))
    return root


def test_a_well_formed_shard_passes(tmp_path):
    out = _run(_pool(tmp_path, _valid()))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "all structural checks passed" in out.stdout


def test_responses_that_are_not_the_tail_of_input_ids_fail(tmp_path):
    """The one that would silently train on the wrong targets."""
    proto = _valid()
    proto.batch["responses"][2, 1] += 1
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "responses != input_ids" in out.stdout


def test_a_right_padded_prompt_fails(tmp_path):
    proto = _valid()
    proto.batch["attention_mask"][1, :PROMPT_LEN] = torch.tensor([1] * 7 + [0] * 5)
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "left-padded" in out.stdout


def test_a_row_with_no_response_token_fails(tmp_path):
    proto = _valid()
    proto.batch["attention_mask"][0, PROMPT_LEN:] = 0
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "no response token" in out.stdout


def test_position_ids_that_do_not_follow_the_mask_fail(tmp_path):
    proto = _valid()
    proto.batch["position_ids"][0] += 1
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "position_ids" in out.stdout


def test_the_response_positions_run_through_the_padding(tmp_path):
    """The layout vllm_rollout_spmd.py:366 documents, pinned as its own case.

    The naive reading -- clamp(cumsum(attention_mask) - 1) across the whole row --
    holds the position flat over the response's right padding. Every row whose
    response did not fill its window disagrees with that, so a checker written
    the naive way rejects almost every real pool. This asserts the flat version
    is the one that fails.
    """
    proto = _valid(resp_used=3)              # 3 of RESP_LEN=5 response tokens used
    real = proto.batch["position_ids"][0].tolist()
    assert real[-RESP_LEN:] == [real[PROMPT_LEN - 1] + i for i in range(1, RESP_LEN + 1)]
    assert _run(_pool(tmp_path, proto)).returncode == 0

    flat = _valid(resp_used=3)
    naive = torch.clamp(flat.batch["attention_mask"].cumsum(dim=1) - 1, min=0)
    rebuilt = DataProto.from_dict(
        tensors={**dict(flat.batch), "position_ids": naive},
        non_tensors=dict(flat.non_tensor_batch),
    )
    # A separate directory: --shards 1 reads only the first shard of each task.
    out = _run(_pool(tmp_path, rebuilt, sub="naive"))
    assert out.returncode == 1
    assert "position_ids" in out.stdout


def test_non_contiguous_trajectory_rows_fail(tmp_path):
    """Trajectory-level sampling draws runs of rows, not sets of them."""
    proto = _valid()
    uids = proto.non_tensor_batch["traj_uid"]
    uids[1], uids[2] = uids[2], uids[1]
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "not contiguous" in out.stdout


def test_a_shard_holding_two_tasks_fails(tmp_path):
    proto = _valid()
    proto.non_tensor_batch["task_name"][3] = "alfworld"
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "more than one task" in out.stdout


def test_a_missing_column_fails(tmp_path):
    proto = _valid()
    del proto.batch["position_ids"]
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "missing required column" in out.stdout


def test_a_float_mask_fails(tmp_path):
    proto = _valid()
    proto.batch["attention_mask"] = proto.batch["attention_mask"].float()
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 1
    assert "dtype" in out.stdout


def _with_extra_column(key, shape):
    """A valid shard plus one more tensor column. Rebuilt rather than mutated:
    save_to_disk locks the underlying TensorDict."""
    proto = _valid()
    n = len(proto)
    tensors = dict(proto.batch)
    tensors[key] = torch.zeros((n, *shape), dtype=torch.long)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={k: v for k, v in proto.non_tensor_batch.items()},
    )


def test_a_kd_pool_is_also_a_valid_pool(tmp_path):
    proto = _with_extra_column("teacher_topk_ids", (RESP_LEN, 3))
    out = _run(_pool(tmp_path, proto))
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_stray_column_fails(tmp_path):
    """The KD columns are named; anything else means the writer changed."""
    out = _run(_pool(tmp_path, _with_extra_column("something_else", (2,))))
    assert out.returncode == 1
    assert "unexpected column" in out.stdout
