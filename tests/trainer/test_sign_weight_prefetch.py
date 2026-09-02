"""The three frozen planes, scored inside the rollout window instead of after it.

``_teacher_prefetch_chunk`` already scores the ON-TASK teacher on rows as they
finalize, on a background thread, in the CPU glue between generations. The base
policy and the off-task teachers are frozen in exactly the same sense, so they
can ride in the same window -- which is what
:meth:`OPDRayTrainer._prefetch_sign_planes` does, leaving
``compute_sign_weight_cache`` to score only what the window missed.

Moving a forward changes nothing about its value. What it can change, and what
these tests are about, is the BOOKKEEPING around it:

1. **The column layout.** The actor reads ``sign_cache_ids`` positionally -- it
   has no way to notice that column 2 holds search's key on one row and
   webshop's on the next. Two paths now write those columns, so they have to
   agree about which model owns which column, per row, for every task.
2. **The double-scoring.** A row the window covered must not be forwarded again
   after the rollout; that is the whole saving, and a re-forward would also file
   a second key for a row that already has one.
3. **The fallback.** A prefetch chunk that fails on the driver is dropped by
   design (the mechanism degrades to the serial path). Rows it did not cover
   must still come out fully keyed, or the actor reads a -1 as an unanswered key
   and the exchange returns a zero log-prob -- not a missing target but a wrong
   one.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.opd_ray_trainer import PrefetchedRow
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

from tests.trainer.test_sign_weight_routing import (
    _FakeRefWG,
    _make_batch,
    _make_trainer,
)

TASKS = ["webshop", "alfworld", "search", "alfworld", "search", "webshop"]


def _fixture(task_names=TASKS):
    teachers = {t: _FakeRefWG(t) for t in ("alfworld", "search", "webshop")}
    base = _FakeRefWG("base")
    trainer = _make_trainer(teachers, base)
    return trainer, teachers, base


def _batch_with_identity(task_names):
    """``_make_batch`` plus the trajectory identity the prefetch merge keys on."""
    batch = _make_batch(task_names)
    bs = len(task_names)
    batch.non_tensor_batch["traj_uid"] = np.array([f"t{i}" for i in range(bs)], dtype=object)
    batch.non_tensor_batch["turn_step"] = np.array([0] * bs, dtype=object)
    return batch


def _chunk_from(batch, rows=None):
    """The rollout loop's queue form: ``((traj_uid, turn_step), row_dict)``."""
    rows = range(len(batch)) if rows is None else rows
    out = []
    for i in rows:
        row = {name: batch.batch[name][i] for name in
               ("input_ids", "attention_mask", "position_ids", "responses")}
        row["task_name"] = batch.non_tensor_batch["task_name"][i]
        key = (str(batch.non_tensor_batch["traj_uid"][i]), int(batch.non_tensor_batch["turn_step"][i]))
        out.append((key, row))
    return out


def _owner_of_each_column(batch, teachers, base):
    """``{(row, column): model name}`` read back from who was handed each key."""
    ids = batch.batch["sign_cache_ids"]
    owner = {}
    for name, wg in [("base", base)] + sorted(teachers.items()):
        for key in wg.cache_ids:
            for i in range(ids.size(0)):
                for c in range(ids.size(1)):
                    if int(ids[i, c]) == key:
                        owner[(i, c)] = name
    return owner


# --------------------------------------------------------------------------- #
# 1. the layout the actor reads positionally
# --------------------------------------------------------------------------- #
def test_the_prefetched_columns_hold_the_same_models_the_serial_pass_would_have():
    """The one invariant that cannot be checked at runtime.

    ``sign_cache_ids`` carries no labels: the actor takes column 0 as the base
    policy and columns 1.. as this row's off-task teachers in sorted order. If
    the two paths disagree about that order for even one task, the arm reads a
    real teacher log-prob from the wrong teacher and nothing downstream notices.
    """
    serial_trainer, serial_teachers, serial_base = _fixture()
    serial_batch = _batch_with_identity(TASKS)
    serial_trainer.compute_sign_weight_cache(serial_batch)
    serial_owner = _owner_of_each_column(serial_batch, serial_teachers, serial_base)

    pre_trainer, pre_teachers, pre_base = _fixture()
    pre_batch = _batch_with_identity(TASKS)
    prefetched = pre_trainer._teacher_prefetch_chunk(_chunk_from(pre_batch))
    pre_trainer.compute_sign_weight_cache(pre_batch, prefetched=prefetched)
    pre_owner = _owner_of_each_column(pre_batch, pre_teachers, pre_base)

    assert serial_owner == pre_owner
    # And it is the layout the docstring promises, not just a matching pair of
    # wrong ones.
    order = sorted(serial_teachers)
    for i, own in enumerate(TASKS):
        assert serial_owner[(i, 0)] == "base"
        for c, other in enumerate([t for t in order if t != own]):
            assert serial_owner[(i, 1 + c)] == other


def test_the_two_paths_derive_the_off_task_order_from_one_expression():
    """They agree above because they call the same helper. Pin that, so the next
    edit to either path cannot quietly fork the rule."""
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    order = ["alfworld", "search", "webshop"]
    assert OPDRayTrainer._sign_off_tasks_for("alfworld", order) == ["search", "webshop"]
    assert OPDRayTrainer._sign_off_tasks_for("search", order) == ["alfworld", "webshop"]
    assert OPDRayTrainer._sign_off_tasks_for("webshop", order) == ["alfworld", "search"]


def test_every_row_is_fully_keyed_through_the_prefetch_path():
    """A -1 left in the column reaches the actor as an unanswered key, and the
    exchange answers those with a zero log-prob -- a wrong target, not a missing
    one."""
    trainer, _, _ = _fixture()
    batch = _batch_with_identity(TASKS)
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch))
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched)
    assert bool((batch.batch["sign_cache_ids"] >= 0).all())


def test_keys_stay_unique_across_models_and_rows():
    """Two rows sharing a key makes the exchange answer one from the other's
    cache. The counter is shared between the two paths, so this is the check
    that they are not each numbering from their own base."""
    trainer, teachers, base = _fixture()
    batch = _batch_with_identity(TASKS)
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch))
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched)
    keys = [k for wg in [base, *teachers.values()] for k in wg.cache_ids]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# 2. the saving itself
# --------------------------------------------------------------------------- #
def test_a_prefetched_row_is_not_forwarded_again_after_the_rollout():
    """The whole point. Scoring it twice would cost the forward this change
    exists to move, and file a second key for a row that already has one."""
    trainer, teachers, base = _fixture()
    batch = _batch_with_identity(TASKS)
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch))

    seen_before = {"base": list(base.rows), **{t: list(w.rows) for t, w in teachers.items()}}
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched)

    assert base.rows == seen_before["base"], "the base policy ran again after the rollout"
    for task, wg in teachers.items():
        assert wg.rows == seen_before[task], f"{task} ran again after the rollout"


def test_the_prefetch_scores_each_model_on_exactly_the_rows_it_speaks_for():
    """Same routing rule as the serial pass: the base over every row, each
    teacher over the rows that are NOT its own task."""
    trainer, teachers, base = _fixture()
    batch = _batch_with_identity(TASKS)
    trainer._teacher_prefetch_chunk(_chunk_from(batch))

    assert sorted(base.rows) == list(range(len(TASKS)))
    for task, wg in teachers.items():
        # The on-task rows appear once, from the on-task pass the prefetch
        # already ran; the off-task rows are the ones the planes add.
        off = sorted(i for i, t in enumerate(TASKS) if t != task)
        on = sorted(i for i, t in enumerate(TASKS) if t == task)
        assert sorted(wg.rows) == sorted(off + on), task


# --------------------------------------------------------------------------- #
# 3. the fallback
# --------------------------------------------------------------------------- #
def test_rows_the_window_missed_are_scored_after_the_rollout():
    """A chunk that fails on the driver is dropped by design. Those rows have to
    come out of the serial pass exactly as they did before this path existed."""
    trainer, teachers, base = _fixture()
    batch = _batch_with_identity(TASKS)
    covered = [0, 1, 2]
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch, covered))

    base_after_prefetch = list(base.rows)
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched)

    missed = [i for i in range(len(TASKS)) if i not in covered]
    assert sorted(base.rows[len(base_after_prefetch):]) == missed
    assert bool((batch.batch["sign_cache_ids"] >= 0).all())


def test_no_prefetch_at_all_is_the_path_that_was_there_before():
    """The env switch and a run with the prefetch off both land here."""
    trainer, teachers, base = _fixture()
    batch = _batch_with_identity(TASKS)
    trainer.compute_sign_weight_cache(batch, prefetched=None)

    assert sorted(base.rows) == list(range(len(TASKS)))
    for task, wg in teachers.items():
        assert sorted(wg.rows) == sorted(i for i, t in enumerate(TASKS) if t != task)
    assert bool((batch.batch["sign_cache_ids"] >= 0).all())


def test_the_planes_are_skipped_when_the_arm_is_off():
    """A run with neither cross-teacher arm has no base worker and no columns to
    fill; the prefetch must not reach for either."""
    trainer, _, _ = _fixture()
    trainer.cross_teacher_enabled = False
    trainer.base_wg = None
    batch = _batch_with_identity(TASKS)
    out = trainer._teacher_prefetch_chunk(_chunk_from(batch))
    assert out and not any(isinstance(v, PrefetchedRow) for v in out.values())


# --------------------------------------------------------------------------- #
# 4. the wrapper the on-task path has to see through
# --------------------------------------------------------------------------- #
def test_the_on_task_value_survives_being_wrapped():
    """``compute_teacher_log_probs`` reads the same entries. Under
    student_indexed_topk its value is a bare cache id, and a pair arriving where
    an int was expected would be written into a LongTensor column as... nothing
    good."""
    trainer, _, _ = _fixture()
    batch = _batch_with_identity(TASKS)
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch))

    assert all(isinstance(v, PrefetchedRow) for v in prefetched.values())
    assert all(isinstance(v.on_task, int) for v in prefetched.values())

    trainer.compute_teacher_log_probs(batch, prefetched=prefetched)
    cache_ids = batch.batch["teacher_cache_ids"]
    keys = trainer._prefetched_teacher_rows(batch)
    for i, key in keys.items():
        assert int(cache_ids[i]) == prefetched[key].on_task


def test_the_hit_rate_is_reported():
    """The number this change exists to move. Without it, a prefetch that
    silently stopped covering rows reads as "sign_weight_forward got slower"."""
    trainer, _, _ = _fixture()
    batch = _batch_with_identity(TASKS)
    prefetched = trainer._teacher_prefetch_chunk(_chunk_from(batch, [0, 1, 2]))
    metrics = {}
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched, metrics=metrics)
    assert metrics["sign_prefetch/rows"] == 3
    assert metrics["sign_prefetch/hit_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 5. why the window path declined, when it does
# --------------------------------------------------------------------------- #
def test_declining_says_why_and_flags_it_in_the_metrics():
    """A run measured at sign_prefetch/hit_rate 0.000 for 148 straight steps is,
    from the metric alone, indistinguishable between "an operator turned it off"
    and "every row was declined". Those want opposite fixes."""
    import verl.trainer.ppo.opd_ray_trainer as mod

    trainer, teachers, base = _fixture()
    saved = mod._ROLLOUT_PREFETCH_SIGN
    mod._ROLLOUT_PREFETCH_SIGN = False
    try:
        batch = _batch_with_identity(TASKS)
        assert trainer._prefetch_sign_planes(_chunk_from(batch)) is None
        assert "ROLLOUT_PREFETCH_SIGN" in trainer._sign_prefetch_declined

        metrics = {}
        trainer.compute_sign_weight_cache(batch, prefetched=None, metrics=metrics)
        assert metrics["sign_prefetch/enabled"] == 0.0
        assert metrics["sign_prefetch/hit_rate"] == 0.0
    finally:
        mod._ROLLOUT_PREFETCH_SIGN = saved


def test_a_chunk_whose_tasks_have_no_teacher_says_which_names_it_saw():
    """The other zero-hit story. The reason names both sides, because the useful
    fact is the mismatch, not that there was one."""
    trainer, _, _ = _fixture()
    batch = _batch_with_identity(["webshop", "alfworld"])
    chunk = _chunk_from(batch)
    for _, row in chunk:
        row["task_name"] = "not_a_task"
    assert trainer._prefetch_sign_planes(chunk) is None
    reason = trainer._sign_prefetch_declined
    assert "not_a_task" in reason and "alfworld" in reason


def test_a_working_prefetch_reports_enabled():
    trainer, _, _ = _fixture()
    batch = _batch_with_identity(TASKS)
    sign_ids = trainer._prefetch_sign_planes(_chunk_from(batch))
    assert sign_ids is not None
    prefetched = {key: PrefetchedRow(on_task=1, sign_ids=sign_ids[key]) for key, _ in _chunk_from(batch)}
    metrics = {}
    trainer.compute_sign_weight_cache(batch, prefetched=prefetched, metrics=metrics)
    assert metrics["sign_prefetch/enabled"] == 1.0
    assert metrics["sign_prefetch/hit_rate"] == 1.0


# --------------------------------------------------------------------------- #
# 6. the post-rollout token budget
# --------------------------------------------------------------------------- #
def test_the_post_rollout_pass_asks_for_a_token_budget():
    """ref.log_prob_micro_batch_size_per_gpu is 4 rows because the WINDOW path
    scores next to a live vLLM KV pool. Out here the pool is asleep, and 4 rows
    is ~3.2k tokens a forward on a 1.7B model -- launch-bound, measured at ~30%
    MFU over 143 s a step."""
    trainer, teachers, base = _fixture()
    trainer._post_rollout_token_budget = 18432
    trainer.compute_sign_weight_cache(_batch_with_identity(TASKS))
    assert base.meta, "the base plane was never scored"
    for wg in [base] + list(teachers.values()):
        for meta in wg.meta:
            assert meta["use_dynamic_bsz"] is True
            assert meta["max_token_len"] == 18432


def test_the_window_path_keeps_the_row_bound():
    """The bound exists for the window and has to stay there: a chunk's
    activations sit beside the KV pool, which is what OOMed this arm once."""
    trainer, teachers, base = _fixture()
    trainer._post_rollout_token_budget = 18432
    trainer._prefetch_sign_planes(_chunk_from(_batch_with_identity(TASKS)))
    for wg in [base] + list(teachers.values()):
        for meta in wg.meta:
            assert "use_dynamic_bsz" not in meta
            assert "max_token_len" not in meta


def test_a_zero_budget_is_the_behaviour_that_was_there_before():
    trainer, teachers, base = _fixture()
    trainer._post_rollout_token_budget = 0
    trainer.compute_sign_weight_cache(_batch_with_identity(TASKS))
    for wg in [base] + list(teachers.values()):
        for meta in wg.meta:
            assert "use_dynamic_bsz" not in meta
