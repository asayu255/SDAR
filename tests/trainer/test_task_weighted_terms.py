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
"""Every loss term is weighted by the same numbers, so the objective is preserved.

``test_task_loss_weights.py`` checks the weights themselves. This checks the
property that made plan A the right shape: with ONE weight column applied to
every term, the multitask loss becomes the mean of the per-task losses a set of
single-task runs would produce. Each coefficient therefore keeps the meaning the
single-task recipes give it, and only the split across tasks moves.

If a term were left on the plain token-mean, that identity would break for the
whole objective -- the unweighted term alone would keep the 69/27/4 token split
-- which is why the actor's guard is a whitelist rather than a blacklist.

The reference-KL term gets its own case: it carries a SECOND per-row vector
(``kl_loss_coef_by_task``), and the two have to multiply. Before this, the
per-task coefficient was scaled by the task's token share on top of itself, so
the 10:10:1 the single-task recipes ask for was arriving as roughly 172:67:1.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss, agg_loss_with_sample_weights
    from verl.trainer.ppo.task_loss_weights import TASK_LOSS_WEIGHT_KEY, attach_task_loss_weights
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

# alfworld / webshop / search, with the lopsided per-turn lengths the episode caps
# produce. 6 rows per task.
TASKS = ["alfworld"] * 6 + ["webshop"] * 6 + ["search"] * 6
TOKENS = [32] * 6 + [12] * 6 + [2] * 6


def _batch():
    n, width = len(TASKS), max(TOKENS)
    mask = torch.zeros(n, width, dtype=torch.long)
    for row, valid in enumerate(TOKENS):
        mask[row, :valid] = 1
    batch = DataProto.from_dict(tensors={"response_mask": mask})
    batch.non_tensor_batch["task_name"] = np.array(TASKS, dtype=object)
    return batch


def _rows(task):
    return [i for i, t in enumerate(TASKS) if t == task]


def _token_mean(loss_mat, mask, rows=None):
    """What a single-task run's agg_loss(token-mean) computes for those rows."""
    if rows is not None:
        loss_mat, mask = loss_mat[rows], mask[rows]
    return float((loss_mat * mask).sum() / mask.sum())


def _weighted(loss_mat, mask, weights, row_scale=None):
    """The actor's _task_weighted, minus the dp/grad-accum factors it cancels.

    Those two are undone exactly by FSDP's gradient averaging and the
    ``loss / gradient_accumulation`` below them, and are covered end-to-end in
    test_task_loss_weights.py; here they would only obscure the identity.
    """
    row_loss = (loss_mat * mask).sum(-1)
    w = weights if row_scale is None else weights * row_scale
    return float((row_loss * w).sum())


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(0)
    batch = _batch()
    attach_task_loss_weights(batch, n_real=len(TASKS), mini_batch_size=len(TASKS), metrics={})
    mask = batch.batch["response_mask"]
    # weights carry a num_mini_batches factor; one mini-batch here, so it is 1
    weights = batch.batch[TASK_LOSS_WEIGHT_KEY].double()
    return batch, mask, weights


# --------------------------------------------------------------------------- #
# The identity
# --------------------------------------------------------------------------- #


def test_a_weighted_term_is_the_mean_of_the_per_task_token_means(fixture):
    _, mask, weights = fixture
    per_token = torch.rand(mask.shape, dtype=torch.float64) * mask

    got = _weighted(per_token, mask, weights)
    expected = sum(_token_mean(per_token, mask, _rows(t)) for t in ("alfworld", "webshop", "search")) / 3

    assert got == pytest.approx(expected, rel=1e-6)


def test_the_whole_four_term_objective_is_the_mean_of_the_single_task_objectives(fixture):
    """pg, entropy, reference KL and SDAR, each with its own coefficient."""
    _, mask, weights = fixture
    terms = {
        "pg": (torch.rand(mask.shape, dtype=torch.float64) * mask, 1.0),
        "entropy": (torch.rand(mask.shape, dtype=torch.float64) * mask, -0.001),
        "kl": (torch.rand(mask.shape, dtype=torch.float64) * mask, 0.001),
        "sdar": (torch.rand(mask.shape, dtype=torch.float64) * mask, 0.01),
    }

    got = sum(coef * _weighted(mat, mask, weights) for mat, coef in terms.values())

    # what three single-task runs would each compute, averaged
    per_task = [
        sum(coef * _token_mean(mat, mask, _rows(task)) for mat, coef in terms.values())
        for task in ("alfworld", "webshop", "search")
    ]
    assert got == pytest.approx(sum(per_task) / 3, rel=1e-6)


def test_leaving_one_term_unweighted_breaks_the_identity(fixture):
    """The negative control for the whitelist guard in the actor."""
    _, mask, weights = fixture
    pg = torch.rand(mask.shape, dtype=torch.float64) * mask
    entropy = torch.rand(mask.shape, dtype=torch.float64) * mask

    mixed = _weighted(pg, mask, weights) - 0.001 * _token_mean(entropy, mask)
    expected = sum(
        _token_mean(pg, mask, _rows(t)) - 0.001 * _token_mean(entropy, mask, _rows(t))
        for t in ("alfworld", "webshop", "search")
    ) / 3

    assert mixed != pytest.approx(expected, rel=1e-6)


def test_coefficient_ratios_are_untouched(fixture):
    """Only the split across tasks moves; a term's weight relative to another's
    is still exactly the ratio of their coefficients."""
    _, mask, weights = fixture
    mat = torch.rand(mask.shape, dtype=torch.float64) * mask

    assert _weighted(mat, mask, weights) * 0.01 / (_weighted(mat, mask, weights) * 0.001) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# The reference-KL term's second per-row vector
# --------------------------------------------------------------------------- #

# what the single-task recipes use: run_alfworld_qwen3.sh / run_webshop_qwen3.sh
# = 0.01, run_search_qwen3.sh = 0.001
COEF_BY_TASK = {"alfworld": 0.01, "webshop": 0.01, "search": 0.001}


def _coef_column(mask):
    return torch.tensor([COEF_BY_TASK[t] for t in TASKS], dtype=torch.float64)


def test_the_per_task_kl_coefficient_arrives_at_its_single_task_value(fixture):
    _, mask, weights = fixture
    kld = torch.rand(mask.shape, dtype=torch.float64) * mask
    coef = _coef_column(mask)

    got = _weighted(kld, mask, weights, row_scale=coef)
    expected = sum(
        COEF_BY_TASK[t] * _token_mean(kld, mask, _rows(t)) for t in ("alfworld", "webshop", "search")
    ) / 3

    assert got == pytest.approx(expected, rel=1e-6)


def test_the_old_path_scaled_each_coefficient_by_its_token_share(fixture):
    """The distortion being removed, measured rather than asserted in prose.

    ``agg_loss_with_sample_weights`` divides by the WHOLE batch's token count, so
    a per-row coefficient there comes out multiplied by the task's token share.
    """
    _, mask, weights = fixture
    kld = torch.ones(mask.shape, dtype=torch.float64) * mask  # unit loss isolates the weighting
    coef = _coef_column(mask)

    old = float(agg_loss_with_sample_weights(kld, mask, coef, "token-mean"))
    shares = {t: float(mask[_rows(t)].sum() / mask.sum()) for t in ("alfworld", "webshop", "search")}
    assert old == pytest.approx(sum(COEF_BY_TASK[t] * shares[t] for t in shares))

    # the intended ratio, and the one the token share actually delivered
    intended = COEF_BY_TASK["alfworld"] / COEF_BY_TASK["search"]
    delivered = (COEF_BY_TASK["alfworld"] * shares["alfworld"]) / (COEF_BY_TASK["search"] * shares["search"])
    assert intended == pytest.approx(10.0)
    assert delivered > 100  # ~170 on this split: the leash is far looser than asked

    # ... and the new path restores it
    new = _weighted(kld, mask, weights, row_scale=coef)
    restored = (COEF_BY_TASK["alfworld"] / 3) / (COEF_BY_TASK["search"] / 3)
    assert restored == pytest.approx(intended)
    assert new == pytest.approx(sum(COEF_BY_TASK[t] / 3 for t in COEF_BY_TASK), rel=1e-6)


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


def test_the_unweighted_path_is_the_plain_token_mean(fixture):
    """With no weight column the actor calls agg_loss, i.e. nothing changes."""
    _, mask, _ = fixture
    mat = torch.rand(mask.shape, dtype=torch.float64) * mask
    assert float(agg_loss(mat, mask, "token-mean")) == pytest.approx(_token_mean(mat, mask))


def test_equal_token_counts_make_the_two_paths_agree(fixture):
    """The weighting is a renormalisation and nothing else: give the tasks equal
    token totals and the weighted loss falls back onto the token-mean."""
    n = 9
    mask = torch.ones(n, 4, dtype=torch.long)
    batch = DataProto.from_dict(tensors={"response_mask": mask})
    batch.non_tensor_batch["task_name"] = np.array(["a", "b", "c"] * 3, dtype=object)
    attach_task_loss_weights(batch, n_real=n, mini_batch_size=n, metrics={})
    weights = batch.batch[TASK_LOSS_WEIGHT_KEY].double()

    mat = torch.rand(mask.shape, dtype=torch.float64)
    assert _weighted(mat, mask, weights) == pytest.approx(_token_mean(mat, mask), rel=1e-6)


# --------------------------------------------------------------------------- #
# The wiring in the actor
# --------------------------------------------------------------------------- #


def _update_policy_loop_source():
    """The body of update_policy's micro-batch loop, as text."""
    from verl.workers.actor import dp_actor as mod

    source = open(mod.__file__.replace(".pyc", ".py")).read()
    start = source.index("for micro_idx, data in enumerate(micro_batches):")
    end = source.index('with _actor_phase("actor.optim"):', start)
    return source[start:end]


def test_every_term_the_actor_can_build_is_routed_through_the_weighting():
    """A term added without weighting would be summed into a differently
    normalised number, and the identity above would quietly stop holding.

    The runtime whitelist in update_policy is the real guard; this catches the
    likelier accident, which is a refactor dropping one of the wirings.
    """
    body = _update_policy_loop_source()
    for term, marker in [
        ("policy gradient", 'agg_fn=partial(_task_weighted, name="actor/pg_loss")'),
        ("entropy", "_agg = _task_weighted if task_weighted else agg_loss"),
        ("reference KL", "_task_weighted(kld, row_scale=kl_loss_coef)"),
        ("SDL", 'agg_fn=partial(_task_weighted, name="actor/sdl_loss")'),
        ("SDAR", 'agg_fn=partial(_task_weighted, name="sdar/loss")'),
    ]:
        assert marker in body, f"the {term} term is no longer routed through _task_weighted"


def test_the_per_task_diagnostics_stay_unweighted():
    """actor/pg_loss/<task> and friends are token-means so they can be compared
    against a single-task run; weighting them would make them incomparable."""
    body = _update_policy_loop_source()
    task_block = body[body.index("for task, rows in iter_task_row_masks("):]
    assert "_task_weighted" not in task_block, (
        "the per-task diagnostics must stay on the plain token-mean"
    )
