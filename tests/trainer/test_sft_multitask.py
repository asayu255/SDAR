"""Unit tests for multitask SFT (behaviour cloning on teacher trajectories).

Covered (CPU-only; Ray / workers are bypassed):
* the SFT cross-entropy loss direction: ``agg_loss(-log_prob, mask)`` decreases
  as the student's log-prob of the (teacher) tokens increases;
* ``MultiTaskSFTTrainer._resolve_data_dir`` reads ``algorithm.sft.data_dir``;
* the inherited fixed-data loading + task-balanced iteration work on SFT data
  that carries NO teacher top-k fields (only token sequences).

If the verl stack (numpy/torch/...) is not installed, the module is skipped.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.trainer.ppo.sft_multitask_ray_trainer import MultiTaskSFTTrainer
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _sft_loss(log_prob, response_mask):
    """The exact expression used in dp_actor.update_policy's use_sft_loss branch."""
    return agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode="token-mean")


def test_sft_loss_decreases_as_student_logprob_increases():
    bs, resp_len = 2, 5
    response_mask = torch.ones(bs, resp_len)
    low = torch.full((bs, resp_len), -3.0)   # student assigns low prob to teacher tokens
    high = torch.full((bs, resp_len), -0.5)  # student assigns high prob to teacher tokens
    assert _sft_loss(high, response_mask) < _sft_loss(low, response_mask)
    # Cross-entropy == negative mean log-prob over the masked tokens.
    assert torch.allclose(_sft_loss(low, response_mask), torch.tensor(3.0), atol=1e-5)


def test_sft_loss_respects_response_mask():
    log_prob = torch.tensor([[-1.0, -2.0, -3.0, -100.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # last token masked out
    # mean of [1,2,3] == 2.0; the masked -100 must not contribute.
    assert torch.allclose(_sft_loss(log_prob, mask), torch.tensor(2.0), atol=1e-5)


def _make_sft_task_proto(task, n, resp_len=4):
    """SFT teacher data: token sequences only, NO teacher top-k fields."""
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros((n, resp_len), dtype=torch.long),
            "input_ids": torch.zeros((n, resp_len + 2), dtype=torch.long),
            "attention_mask": torch.ones((n, resp_len + 2), dtype=torch.long),
        },
        non_tensors={"task_name": np.array([task] * n, dtype=object)},
    )


def test_resolve_data_dir_reads_sft_namespace():
    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.config = OmegaConf.create({"algorithm": {"sft": {"data_dir": "/some/dir"}}})
    assert trainer._resolve_data_dir() == "/some/dir"

    trainer.config = OmegaConf.create({"algorithm": {"sft": {}}})
    with pytest.raises(AssertionError):
        trainer._resolve_data_dir()


def test_load_and_balanced_iter_on_topk_free_data(tmp_path):
    _make_sft_task_proto("alfworld", 6).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_sft_task_proto("search", 6).save_to_disk(str(tmp_path / "search.pt"))

    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()
    assert len(trainer.offpolicy_data) == 12
    assert "teacher_topk_logprobs" not in trainer.offpolicy_data.batch
    assert set(trainer._task_to_indices) == {"alfworld", "search"}

    trainer.per_task_traj_per_step = 3
    trainer.total_training_steps = 4
    trainer.config = OmegaConf.create({"data": {"seed": 1}})
    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 4
    for batch in steps:
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        assert counts == {"alfworld": 3, "search": 3}
