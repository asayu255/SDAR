"""Unit tests for multitask SFT (behaviour cloning on teacher trajectories).

Covered (CPU-only; Ray / workers are bypassed):
* the SFT cross-entropy loss direction: ``agg_loss(-log_prob, mask)`` decreases
  as the student's log-prob of the (teacher) tokens increases;
* ``MultiTaskSFTTrainer._resolve_data_dir`` reads ``algorithm.sft.data_dir``, and
  refuses a loss configuration that would need the columns this arm drops;
* the inherited fixed-data loading + task-balanced iteration work on SFT data
  that carries NO teacher top-k fields (only token sequences);
* ``_attach_sft_loss_weights`` hands every task an equal share of the loss and
  zeroes the adjust_batch padding rows.

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


def _make_sft_task_proto(task, n_traj, turns_per_traj=1, resp_len=4):
    """SFT teacher data: token sequences only, NO teacher top-k fields.
    ``n_traj`` trajectories, each contributing ``turns_per_traj`` turn-rows that
    share a single traj_uid."""
    n = n_traj * turns_per_traj
    traj_uids = []
    for j in range(n_traj):
        traj_uids += [f"{task}-{j}"] * turns_per_traj
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros((n, resp_len), dtype=torch.long),
            "input_ids": torch.zeros((n, resp_len + 2), dtype=torch.long),
            "attention_mask": torch.ones((n, resp_len + 2), dtype=torch.long),
        },
        non_tensors={
            "task_name": np.array([task] * n, dtype=object),
            "traj_uid": np.array(traj_uids, dtype=object),
        },
    )


def _sft_config(data_dir="/some/dir", **actor_overrides):
    """The composed config as main_sft_multitask leaves it after its injection."""
    actor = {"use_sft_loss": True, "use_teacher_kl_loss": False}
    actor.update(actor_overrides)
    sft = {"data_dir": data_dir} if data_dir is not None else {}
    return OmegaConf.create({"algorithm": {"sft": sft}, "actor_rollout_ref": {"actor": actor}})


def test_resolve_data_dir_reads_sft_namespace():
    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.config = _sft_config()
    assert trainer._resolve_data_dir() == "/some/dir"

    trainer.config = _sft_config(data_dir=None)
    with pytest.raises(AssertionError):
        trainer._resolve_data_dir()


def test_resolve_data_dir_rejects_a_loss_that_needs_the_dropped_columns():
    """The arm throws the teacher top-k away at load, so a teacher-KL term would
    read a key that is not there. It has to fail before the pool is loaded, not
    on the first optimizer step."""
    trainer = object.__new__(MultiTaskSFTTrainer)

    trainer.config = _sft_config(use_teacher_kl_loss=True)
    with pytest.raises(AssertionError, match="use_teacher_kl_loss"):
        trainer._resolve_data_dir()

    # ...and the arm is only itself when the SFT loss is actually on.
    trainer.config = _sft_config(use_sft_loss=False)
    with pytest.raises(AssertionError, match="use_sft_loss"):
        trainer._resolve_data_dir()


def test_load_and_balanced_iter_on_topk_free_data(tmp_path):
    # 6 trajectories/task, 2 turns each; NO teacher top-k fields (SFT data).
    _make_sft_task_proto("alfworld", 6, turns_per_traj=2).save_to_disk(str(tmp_path / "alfworld.pt"))
    _make_sft_task_proto("search", 6, turns_per_traj=2).save_to_disk(str(tmp_path / "search.pt"))

    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.teacher_data_dir = str(tmp_path)
    trainer._load_offpolicy_data()
    # One DataProto per task; the pool is never concatenated.
    assert {t: sum(len(s) for s in v) for t, v in trainer._task_shards.items()} == {
        "alfworld": 12, "search": 12
    }
    assert "teacher_topk_logprobs" not in trainer._task_shards["alfworld"][0].batch
    assert set(trainer._task_to_trajs) == {"alfworld", "search"}
    assert {t: len(v) for t, v in trainer._task_to_trajs.items()} == {"alfworld": 6, "search": 6}

    trainer.per_task_traj_per_step = 3  # trajectories per task per step
    trainer.total_training_steps = 4
    trainer.config = OmegaConf.create({"data": {"seed": 1}})
    steps = list(trainer._offpolicy_batch_iter())
    assert len(steps) == 4
    for batch in steps:
        counts = {}
        for t in batch.non_tensor_batch["task_name"]:
            counts[t] = counts.get(t, 0) + 1
        # 3 trajectories * 2 turns per task.
        assert counts == {"alfworld": 6, "search": 6}
        # Whole trajectories drawn: each traj_uid appears with all its turn-rows.
        uid_counts = {}
        for u in batch.non_tensor_batch["traj_uid"]:
            uid_counts[u] = uid_counts.get(u, 0) + 1
        assert all(c == 2 for c in uid_counts.values())


def _weight_batch(rows_per_task, tokens_per_row, n_padding=0):
    """A batch with a known per-task response-token budget.

    ``rows_per_task`` maps task -> row count; every one of that task's rows carries
    ``tokens_per_row[task]`` response tokens. ``n_padding`` rows are appended to
    stand in for adjust_batch's copies (their task name is reused from the first
    task, exactly as a real copy would carry its source row's task).
    """
    resp_len = max(tokens_per_row.values())
    masks, names = [], []
    for task, n_rows in rows_per_task.items():
        for _ in range(n_rows):
            row = torch.zeros(resp_len)
            row[: tokens_per_row[task]] = 1.0
            masks.append(row)
            names.append(task)
    first_task = next(iter(rows_per_task))
    for _ in range(n_padding):
        row = torch.zeros(resp_len)
        row[: tokens_per_row[first_task]] = 1.0
        masks.append(row)
        names.append(first_task)
    return DataProto.from_dict(
        tensors={"response_mask": torch.stack(masks)},
        non_tensors={"task_name": np.array(names, dtype=object)},
    )


def _attach_weights(batch, n_real, mini_batch_size):
    trainer = object.__new__(MultiTaskSFTTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"ppo_mini_batch_size": mini_batch_size}}}
    )
    metrics = {}
    trainer._attach_sft_loss_weights(batch, n_real=n_real, metrics=metrics)
    return batch.batch["sft_loss_weight"], metrics


def test_sft_weights_give_every_task_an_equal_share():
    # Deliberately lopsided: alfworld brings 24x the response tokens of search.
    rows = {"alfworld": 12, "webshop": 6, "search": 2}
    tokens = {"alfworld": 8, "webshop": 4, "search": 2}
    batch = _weight_batch(rows, tokens)
    weights, metrics = _attach_weights(batch, n_real=len(batch), mini_batch_size=5)

    num_mini_batches = len(batch) / 5
    row_tokens = batch.batch["response_mask"].sum(-1)
    task_names = batch.non_tensor_batch["task_name"]

    # Each task's total weight-times-tokens is the same 1/num_tasks share.
    budgets = {}
    for task in rows:
        sel = torch.from_numpy(np.asarray(task_names == task))
        budgets[task] = float((weights[sel] * row_tokens[sel]).sum())
    for task, budget in budgets.items():
        assert budget == pytest.approx(num_mini_batches / len(rows), rel=1e-6), task

    # The unweighted token share it corrects for is the lopsided one.
    assert metrics["sft/token_share/alfworld"] == pytest.approx(96 / 124)
    assert metrics["sft/token_share/search"] == pytest.approx(4 / 124)
    assert sum(metrics[f"sft/token_share/{t}"] for t in rows) == pytest.approx(1.0)


def test_sft_weighted_loss_recovers_the_mean_of_per_task_means():
    rows = {"alfworld": 12, "webshop": 6, "search": 2}
    tokens = {"alfworld": 8, "webshop": 4, "search": 2}
    batch = _weight_batch(rows, tokens)
    weights, _ = _attach_weights(batch, n_real=len(batch), mini_batch_size=5)
    response_mask = batch.batch["response_mask"]
    num_mini_batches = len(batch) / 5

    # Give each task a different constant per-token NLL. Summed over the step the
    # weighted loss must be num_mini_batches * mean(per-task token-means), i.e. the
    # per-task means enter with equal weight even though their token counts do not.
    nll_by_task = {"alfworld": 3.0, "webshop": 1.0, "search": 0.5}
    task_names = batch.non_tensor_batch["task_name"]
    per_token = torch.zeros_like(response_mask)
    for task, value in nll_by_task.items():
        sel = torch.from_numpy(np.asarray(task_names == task))
        per_token[sel] = value

    row_nll = (per_token * response_mask).sum(-1)
    total = float((row_nll * weights).sum())
    expected = num_mini_batches * sum(nll_by_task.values()) / len(nll_by_task)
    assert total == pytest.approx(expected, rel=1e-6)


def test_sft_weights_zero_the_padding_rows():
    rows = {"alfworld": 8, "search": 2}
    tokens = {"alfworld": 4, "search": 2}
    batch = _weight_batch(rows, tokens, n_padding=5)
    n_real = 10
    weights, metrics = _attach_weights(batch, n_real=n_real, mini_batch_size=3)

    assert (weights[n_real:] == 0).all()
    assert (weights[:n_real] > 0).all()
    assert metrics["sft/padding_rows"] == 5
    # The padding rows are alfworld copies; excluding them keeps alfworld's share
    # at what its real rows alone contribute.
    assert metrics["sft/token_share/alfworld"] == pytest.approx(32 / 36)
    # And the per-task budgets still come out equal.
    row_tokens = batch.batch["response_mask"].sum(-1)
    task_names = batch.non_tensor_batch["task_name"]
    num_mini_batches = len(batch) / 3
    for task in rows:
        sel = torch.from_numpy(np.asarray(task_names == task))
        sel[n_real:] = False
        assert float((weights[sel] * row_tokens[sel]).sum()) == pytest.approx(
            num_mini_batches / len(rows), rel=1e-6
        )
