"""Unit tests for per-task normalisation of the distillation loss.

``attach_task_loss_weights`` writes one weight per row; the actor multiplies each
row's token-summed loss by it and un-does FSDP's gradient averaging and the
gradient-accumulation division. What has to hold end to end is that a step's loss
becomes ``(1/num_tasks) * sum_task token_mean(task)`` -- the equal-share
token-mean -- at the same magnitude as the unweighted token-mean it replaces.

These are CPU-only: the driver-side weights are computed exactly as in
production, and the worker-side arithmetic is replayed by ``_replay_actor`` (the
same formula as ``DataParallelPPOActor.update_policy``) rather than by standing
up FSDP.
"""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY, attach_task_loss_weights
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_batch(task_names, row_tokens, resp_len=None):
    """A batch whose row ``i`` has ``row_tokens[i]`` unmasked response tokens."""
    bs = len(task_names)
    resp_len = resp_len or max(row_tokens)
    response_mask = torch.zeros((bs, resp_len), dtype=torch.long)
    for i, n in enumerate(row_tokens):
        response_mask[i, :n] = 1
    return DataProto.from_dict(
        tensors={"response_mask": response_mask},
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )


def _replay_actor(batch, per_token_loss, *, mini_batch_size, dp_world_size, micro_batch_size):
    """The loss a step accumulates, following update_policy's arithmetic exactly.

    Each rank takes a contiguous slice of the batch (DP sharding), splits it into
    mini-batches, and each mini-batch into micro-batches. A micro-batch's
    contribution is ``sum(row_loss * weight) * dp_world_size *
    gradient_accumulation``, then divided by ``gradient_accumulation`` when the
    loss is formed; FSDP averages across ranks. Returns the mean over the step's
    optimizer steps, i.e. what one step of training sees.
    """
    weights = batch.batch[TASK_LOSS_WEIGHT_KEY]
    mask = batch.batch["response_mask"]
    row_loss = (per_token_loss * mask).sum(-1)

    bs = len(batch)
    per_rank = bs // dp_world_size
    mini_per_rank = mini_batch_size // dp_world_size
    # The CONFIGURED value, which is what update_policy divides by even when the
    # final mini-batch is short and yields fewer micro-batches than this.
    gradient_accumulation = mini_per_rank // micro_batch_size
    num_mini_batches = math.ceil(bs / mini_batch_size)

    step_losses = []
    for mb in range(num_mini_batches):
        rank_losses = []
        for rank in range(dp_world_size):
            base = rank * per_rank + mb * mini_per_rank
            stop = min(base + mini_per_rank, (rank + 1) * per_rank)  # short tail
            total = 0.0
            lo = base
            while lo < stop:
                hi = min(lo + micro_batch_size, stop)
                weighted = float((row_loss[lo:hi] * weights[lo:hi]).sum())
                weighted = weighted * dp_world_size * gradient_accumulation
                total += weighted / gradient_accumulation  # loss = policy_loss / grad_accum
                lo = hi
            rank_losses.append(total)
        step_losses.append(sum(rank_losses) / dp_world_size)  # FSDP averages the ranks
    return sum(step_losses) / len(step_losses)


# ``attach_task_loss_weights`` stores the weights as a float32 column (one scalar
# per row, travelling into the actor alongside the batch), so a weight like
# 2 / (3 * 256) carries about 3e-8 of relative quantisation error and no replay of
# the actor can be exact to more than that. The structural mistakes these tests
# exist to catch -- a missing dp_world_size, the wrong mini-batch count, dividing
# by the per-mini-batch token total instead of the step's -- are all integer-factor
# sized, so 1e-6 catches every one of them with 30x the headroom float32 needs.
_REL = 1e-6


def _token_mean(per_token_loss, mask, rows):
    sel = torch.zeros(len(mask), dtype=torch.bool)
    sel[rows] = True
    return float((per_token_loss * mask)[sel].sum() / mask[sel].sum())


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_equal_share_despite_lopsided_token_counts():
    """The 69/27/4 token split becomes a 1/3 split of the loss."""
    # 4 rows/task, but alfworld rows are 16x longer than search rows.
    tasks = ["alfworld"] * 4 + ["webshop"] * 4 + ["search"] * 4
    tokens = [32] * 4 + [12] * 4 + [2] * 4
    batch = _make_batch(tasks, tokens)
    metrics = {}
    attach_task_loss_weights(batch, n_real=12, mini_batch_size=6, metrics=metrics)

    weights = batch.batch[TASK_LOSS_WEIGHT_KEY]
    mask = batch.batch["response_mask"]
    row_tokens = mask.sum(-1).double()

    # sum over a task's rows of (weight * row_tokens) is the same for every task:
    # that product is exactly the task's share of a uniform per-token loss.
    shares = [float((weights[i : i + 4].double() * row_tokens[i : i + 4]).sum()) for i in (0, 4, 8)]
    assert shares[0] == pytest.approx(shares[1]) == pytest.approx(shares[2])

    # ... and the unweighted token shares are the lopsided ones being corrected.
    assert metrics["task_loss/token_share/alfworld"] == pytest.approx(128 / 184)
    assert metrics["task_loss/token_share/search"] == pytest.approx(8 / 184)
    assert sum(v for k, v in metrics.items() if k.startswith("task_loss/token_share/")) == pytest.approx(1.0)
    assert metrics["task_loss/rows/alfworld"] == 4
    assert metrics["task_loss/padding_rows"] == 0


def test_padding_rows_get_zero_weight_and_do_not_dilute():
    """adjust_batch's duplicates are excluded from the task's token total."""
    tasks = ["alfworld"] * 2 + ["search"] * 2 + ["alfworld"] * 2  # last 2 are padding
    tokens = [10, 10, 4, 4, 10, 10]
    batch = _make_batch(tasks, tokens)
    metrics = {}
    attach_task_loss_weights(batch, n_real=4, mini_batch_size=3, metrics=metrics)

    weights = batch.batch[TASK_LOSS_WEIGHT_KEY]
    assert float(weights[4]) == 0.0 and float(weights[5]) == 0.0
    assert metrics["task_loss/padding_rows"] == 2
    assert metrics["task_loss/rows/alfworld"] == 2
    # T_alfworld counted the 2 real rows only: 2 mini-batches / (2 tasks * 20 tokens)
    assert float(weights[0]) == pytest.approx(2 / (2 * 20))
    assert float(weights[2]) == pytest.approx(2 / (2 * 8))


def test_indivisible_batch_counts_the_short_final_mini_batch():
    """adjust_batch rounds to lcm(log_prob_micro*W, ppo_micro*W), not to the
    mini-batch size, so most batches end with a short mini-batch. The scale must
    follow the number of optimizer steps that produces, i.e. the ceiling."""
    batch = _make_batch(["alfworld"] * 5, [4] * 5)
    attach_task_loss_weights(batch, n_real=5, mini_batch_size=2, metrics={})

    # 5 rows / 2 = 3 optimizer steps (2 + 2 + 1), one task, 20 tokens total
    assert float(batch.batch[TASK_LOSS_WEIGHT_KEY][0]) == pytest.approx(3 / (1 * 20))


def test_missing_task_names_are_rejected():
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones((4, 3), dtype=torch.long)})
    with pytest.raises(AssertionError, match="requires per-row task names"):
        attach_task_loss_weights(batch, n_real=4, mini_batch_size=2, metrics={})


# --------------------------------------------------------------------------- #
# End-to-end: what the optimizer actually sees
# --------------------------------------------------------------------------- #
def test_step_loss_is_the_equal_share_token_mean():
    """One optimizer step's loss == (1/3) * sum_task token_mean(task)."""
    torch.manual_seed(0)
    # 24 rows: 8 per task, laid out so every mini-batch is a mix of tasks.
    tasks = ["alfworld", "webshop", "search"] * 8
    tokens = [32, 12, 2] * 8
    batch = _make_batch(tasks, tokens)
    attach_task_loss_weights(batch, n_real=24, mini_batch_size=12, metrics={})

    mask = batch.batch["response_mask"]
    per_token_loss = torch.rand(mask.shape, dtype=torch.float64) * mask

    got = _replay_actor(
        batch, per_token_loss, mini_batch_size=12, dp_world_size=2, micro_batch_size=3
    )
    expected = sum(
        _token_mean(per_token_loss, mask, [i for i, t in enumerate(tasks) if t == task]) / 3
        for task in ("alfworld", "webshop", "search")
    )
    assert got == pytest.approx(expected, rel=_REL)


def test_step_loss_magnitude_matches_the_unweighted_token_mean():
    """A uniform per-token loss gives the same number either way.

    The weighting redistributes across tasks; it does not rescale the loss, so
    the effective learning rate is unchanged. With a constant per-token loss the
    equal-share mean and the plain token-mean coincide, which pins the scale.
    """
    tasks = ["alfworld", "webshop", "search"] * 8
    tokens = [32, 12, 2] * 8
    batch = _make_batch(tasks, tokens)
    attach_task_loss_weights(batch, n_real=24, mini_batch_size=12, metrics={})

    mask = batch.batch["response_mask"]
    per_token_loss = mask.double() * 0.5

    got = _replay_actor(
        batch, per_token_loss, mini_batch_size=12, dp_world_size=2, micro_batch_size=3
    )
    assert got == pytest.approx(0.5, rel=_REL)


def test_step_loss_is_invariant_to_micro_batch_grouping():
    """Grouping rows into micro-batches must not change the objective."""
    tasks = ["alfworld", "webshop", "search"] * 8
    tokens = [32, 12, 2] * 8
    batch = _make_batch(tasks, tokens)
    attach_task_loss_weights(batch, n_real=24, mini_batch_size=12, metrics={})

    mask = batch.batch["response_mask"]
    torch.manual_seed(1)
    per_token_loss = torch.rand(mask.shape, dtype=torch.float64) * mask

    a = _replay_actor(batch, per_token_loss, mini_batch_size=12, dp_world_size=2, micro_batch_size=3)
    b = _replay_actor(batch, per_token_loss, mini_batch_size=12, dp_world_size=2, micro_batch_size=6)
    assert a == pytest.approx(b, rel=_REL)


def test_short_final_mini_batch_keeps_the_equal_share():
    """The real failure: 6880 rows against ppo_mini_batch_size 60.

    adjust_batch rounds to lcm(log_prob_micro*W, ppo_micro*W) = 160, so the row
    count is a multiple of 160 and only every third step is also a multiple of
    60. The short mini-batch that leaves needs no special handling: update_policy
    divides by the CONFIGURED gradient_accumulation and the weights multiply by
    the same constant, so they cancel whatever the mini-batch's length.
    """
    # 30 rows, mini-batch 12 -> 12 + 12 + 6, i.e. a short final mini-batch.
    tasks = ["alfworld", "webshop", "search"] * 10
    tokens = [32, 12, 2] * 10
    batch = _make_batch(tasks, tokens)
    attach_task_loss_weights(batch, n_real=30, mini_batch_size=12, metrics={})

    mask = batch.batch["response_mask"]
    torch.manual_seed(3)
    per_token_loss = torch.rand(mask.shape, dtype=torch.float64) * mask

    got = _replay_actor(
        batch, per_token_loss, mini_batch_size=12, dp_world_size=2, micro_batch_size=3
    )
    expected = sum(
        _token_mean(per_token_loss, mask, [i for i, t in enumerate(tasks) if t == task]) / 3
        for task in ("alfworld", "webshop", "search")
    )
    assert got == pytest.approx(expected, rel=_REL)


def test_short_final_mini_batch_keeps_the_magnitude():
    """A uniform per-token loss still comes out at its own value, not scaled."""
    tasks = ["alfworld", "webshop", "search"] * 10
    tokens = [32, 12, 2] * 10
    batch = _make_batch(tasks, tokens)
    attach_task_loss_weights(batch, n_real=30, mini_batch_size=12, metrics={})

    mask = batch.batch["response_mask"]
    got = _replay_actor(
        batch, mask.double() * 0.5, mini_batch_size=12, dp_world_size=2, micro_batch_size=3
    )
    assert got == pytest.approx(0.5, rel=_REL)
