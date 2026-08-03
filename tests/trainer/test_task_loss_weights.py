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

Tolerances are ``rel=1e-6``, not tighter: ``attach_task_loss_weights`` stores
float32 weights, so a few thousand of them summed in an order that depends on the
torch build lands about 1e-8 relative away from the exact answer. A tolerance
below float32's own resolution does not test the arithmetic, it tests the
summation order.
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
    assert got == pytest.approx(expected, rel=1e-6)


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
    assert got == pytest.approx(0.5, rel=1e-6)


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
    assert a == pytest.approx(b, rel=1e-6)


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
    assert got == pytest.approx(expected, rel=1e-6)


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
    assert got == pytest.approx(0.5, rel=1e-6)


# --------------------------------------------------------------------------- #
# OPD+GRPO: the same weights applied to the policy-gradient term
# --------------------------------------------------------------------------- #
# Pure OPD has one live loss term, so weighting it is the whole story. OPD+GRPO
# adds the GRPO policy gradient, and the two are combined as
# ``pg_loss * pg_loss_coef + teacher_kl * teacher_kl_coef``. Weighting only the
# KL would leave the policy gradient on the token-count split, which does not
# just leave that term unbalanced -- it changes the RATIO between the terms,
# task by task, and that ratio is what pg_loss_coef exists to set.
from verl.trainer.ppo.core_algos import (  # noqa: E402
    agg_loss,
    agg_loss_by_task_weights,
    compute_policy_loss,
    compute_policy_loss_per_token,
)


def _policy_inputs(bs=8, resp=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "old_log_prob": torch.randn(bs, resp, generator=g),
        "log_prob": torch.randn(bs, resp, generator=g),
        "advantages": torch.randn(bs, resp, generator=g),
        "response_mask": torch.ones(bs, resp),
    }


@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"])
def test_splitting_compute_policy_loss_changed_no_value(loss_agg_mode):
    """The refactor that exposed the per-token matrix must be a no-op.

    ``compute_policy_loss`` now delegates the clipping to
    ``compute_policy_loss_per_token`` and aggregates the result. Every existing
    caller -- every non-OPD recipe in the repo -- goes through that path, so a
    drift here would silently change unrelated runs.
    """
    inp = _policy_inputs()
    pg_loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss(
        **inp, cliprange=0.2, clip_ratio_c=3.0, loss_agg_mode=loss_agg_mode
    )
    mat, clipfrac2, ppo_kl2, clipfrac_lower2 = compute_policy_loss_per_token(
        **inp, cliprange=0.2, clip_ratio_c=3.0
    )
    recomposed = agg_loss(loss_mat=mat, loss_mask=inp["response_mask"], loss_agg_mode=loss_agg_mode)

    assert torch.equal(pg_loss, recomposed)
    assert torch.equal(clipfrac, clipfrac2)
    assert torch.equal(ppo_kl, ppo_kl2)
    assert torch.equal(clipfrac_lower, clipfrac_lower2)


def test_per_token_policy_loss_keeps_its_gradient():
    """The weighted path backprops through the matrix, not through a detached copy."""
    inp = _policy_inputs()
    inp["log_prob"] = inp["log_prob"].clone().requires_grad_(True)
    mat, _, _, _ = compute_policy_loss_per_token(**inp, cliprange=0.2, clip_ratio_c=3.0)
    weights = torch.rand(mat.size(0))
    agg_loss_by_task_weights(loss_mat=mat, loss_mask=inp["response_mask"], row_weights=weights).backward()
    assert inp["log_prob"].grad is not None and torch.isfinite(inp["log_prob"].grad).all()


def test_task_weighted_aggregation_is_the_weighted_token_sum():
    """``agg_loss_by_task_weights`` is sum_i w_i * sum_t (loss * mask), nothing else.

    Stated as a test because it is deliberately NOT one of ``agg_loss``'s modes:
    token-mean divides by the batch's token count, which is the very quantity the
    weights exist to stop deciding the answer.
    """
    loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    w = torch.tensor([0.5, 2.0])
    got = agg_loss_by_task_weights(loss_mat=loss, loss_mask=mask, row_weights=w)
    assert float(got) == pytest.approx(0.5 * (1.0 + 2.0) + 2.0 * (4.0 + 5.0 + 6.0))


def test_both_loss_terms_get_the_same_equal_share_treatment():
    """pg and teacher-KL, weighted by one weight vector, each land at 1/T per task.

    The point is not that either term is individually correct -- the tests above
    already pin that -- but that they are correct under the SAME weights, which is
    what leaves ``pg_loss_coef`` meaning the ratio between them rather than a
    ratio that varies by task.
    """
    tasks = ["alfworld"] * 6 + ["search"] * 2
    row_tokens = [10] * 6 + [1] * 2  # 60 vs 2 tokens: the imbalance being corrected
    batch = _make_batch(tasks, row_tokens)
    metrics = {}
    attach_task_loss_weights(batch, n_real=len(tasks), mini_batch_size=8, metrics=metrics)

    mask = batch.batch["response_mask"]
    g = torch.Generator().manual_seed(7)
    pg = torch.rand(mask.shape, generator=g)
    kl = torch.rand(mask.shape, generator=g)

    alf = list(range(6))
    sea = [6, 7]
    for name, per_token in (("pg", pg), ("kl", kl)):
        got = _replay_actor(batch, per_token, mini_batch_size=8, dp_world_size=2, micro_batch_size=2)
        want = 0.5 * (_token_mean(per_token, mask, alf) + _token_mean(per_token, mask, sea))
        assert got == pytest.approx(want, rel=1e-6), name

    # And the combination: a coefficient applied to a term scales that term's
    # contribution and nothing else.
    combined = _replay_actor(batch, 1.0 * pg + 0.25 * kl, mini_batch_size=8, dp_world_size=2, micro_batch_size=2)
    separate = (
        1.0 * _replay_actor(batch, pg, mini_batch_size=8, dp_world_size=2, micro_batch_size=2)
        + 0.25 * _replay_actor(batch, kl, mini_batch_size=8, dp_world_size=2, micro_batch_size=2)
    )
    assert combined == pytest.approx(separate, rel=1e-6)


def test_weighting_only_one_term_would_distort_their_ratio():
    """The negative control for the choice above -- why both terms are weighted.

    With alfworld holding 30x search's tokens, leaving the policy gradient on the
    plain token-mean makes the pg:KL ratio differ per task by that same factor,
    even though pg_loss_coef is a single number. If this ever stops failing, the
    weighting has been silently applied to both sides of the comparison and the
    test has lost its meaning.
    """
    tasks = ["alfworld"] * 6 + ["search"] * 2
    batch = _make_batch(tasks, [10] * 6 + [1] * 2)
    attach_task_loss_weights(batch, n_real=len(tasks), mini_batch_size=8, metrics={})

    mask = batch.batch["response_mask"]
    per_token = torch.ones(mask.shape)
    alf, sea = list(range(6)), [6, 7]

    # Weighted: each task contributes 1/2 of its own token-mean.
    weighted = {t: 0.5 for t in ("alfworld", "search")}
    # Unweighted token-mean: a task contributes its share of the batch's tokens.
    total = float((per_token * mask).sum())
    unweighted = {
        "alfworld": float((per_token * mask)[torch.tensor(alf)].sum()) / total,
        "search": float((per_token * mask)[torch.tensor(sea)].sum()) / total,
    }
    ratio = {t: unweighted[t] / weighted[t] for t in weighted}
    assert ratio["alfworld"] / ratio["search"] == pytest.approx(30.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# What the weighted path refuses to run under
# --------------------------------------------------------------------------- #
from verl.workers.actor.dp_actor import check_task_weighting_supported  # noqa: E402


def _actor_config(**over):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "use_sdl_loss": False,
            "use_sdar_loss": False,
        }
    )
    for k, v in over.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


def test_opd_grpo_config_is_accepted_with_a_live_policy_gradient():
    """The point of the GRPO extension: pg_loss_coef != 0 is no longer refused.

    Pure OPD asserted pg_loss_coef == 0 because it only weighted the teacher KL.
    Both terms are weighted now, so the OPD+GRPO arm's own config has to pass.
    """
    check_task_weighting_supported(
        _actor_config(pg_loss_coef=1.0, entropy_coeff=0.0),
        use_teacher_kl_loss=True,
        ulysses_sequence_parallel_size=1,
    )


@pytest.mark.parametrize(
    "over, why",
    [
        ({"use_dynamic_bsz": True}, "rescales the mini-batch loss"),
        ({"ppo_epochs": 2}, "reuses each mini-batch"),
        ({"loss_agg_mode": "seq-mean-token-mean"}, "aggregates differently"),
        ({"policy_loss.loss_mode": "gspo"}, "sequence-level ratio"),
        ({"use_kl_loss": True}, "unweighted extra term"),
        ({"use_sdl_loss": True}, "unweighted extra term"),
        ({"use_sdar_loss": True}, "unweighted extra term"),
    ],
)
def test_configs_that_would_silently_undo_the_weighting_are_refused(over, why):
    """Each of these produces a plausible number with the weighting quietly gone."""
    with pytest.raises(AssertionError):
        check_task_weighting_supported(
            _actor_config(**over), use_teacher_kl_loss=True, ulysses_sequence_parallel_size=1
        )


def test_sequence_parallel_and_missing_teacher_kl_are_refused():
    with pytest.raises(AssertionError):
        check_task_weighting_supported(
            _actor_config(), use_teacher_kl_loss=True, ulysses_sequence_parallel_size=2
        )
    with pytest.raises(AssertionError):
        check_task_weighting_supported(
            _actor_config(), use_teacher_kl_loss=False, ulysses_sequence_parallel_size=1
        )
