"""Which model is scored on which rows, and which column its key lands in.

The weights read four models on one support. Three of them are cached by the
driver here, and the actor reads them back POSITIONALLY -- column 0 is the base,
columns 1.. are this row's off-task teachers -- so a column that does not mean
what the actor thinks it means is a silent swap of one teacher for another, with
every shape still correct. That is what this file pins.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


class _FakeRefWG:
    """Records the rows it was asked for and the cache keys they were filed under."""

    world_size = 1

    def __init__(self, name):
        self.name = name
        self.rows = []  # original indices, in call order
        self.cache_ids = []

    def compute_ref_topk_log_prob(self, sub):
        first_tok = sub.batch["responses"][:, 0]
        self.rows.extend(first_tok.tolist())
        self.cache_ids.extend(sub.batch["teacher_cache_ids"].tolist())
        return DataProto.from_dict(
            tensors={"teacher_scored": torch.ones(len(first_tok), 1, dtype=torch.bool)}
        )


def _make_batch(task_names, resp_len=4, prompt_len=3):
    bs = len(task_names)
    # responses[i, 0] == i -> the fake worker can report which rows it saw.
    responses = torch.zeros((bs, resp_len), dtype=torch.long)
    responses[:, 0] = torch.arange(bs)
    seq = prompt_len + resp_len
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "input_ids": torch.ones((bs, seq), dtype=torch.long),
            "attention_mask": torch.ones((bs, seq), dtype=torch.long),
            "position_ids": torch.arange(seq).unsqueeze(0).repeat(bs, 1),
        },
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )
    batch.meta_info["task_id_names"] = sorted({t for t in task_names})
    return batch


def _make_trainer(teachers, base, enabled=True):
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    # Only compute_sign_weight_cache is exercised, so the heavy __init__ is
    # bypassed and the handful of attributes it reads are set by hand.
    trainer = object.__new__(OPDRayTrainer)
    trainer.teacher_wg = teachers
    trainer.base_wg = base
    trainer.sign_weight_enabled = enabled
    trainer.teacher_topk_kl = True
    trainer.teacher_kl_topk = 20
    trainer.student_indexed_topk = True
    trainer._teacher_cache_counter = 0
    return trainer


def _fixture(task_names):
    teachers = {t: _FakeRefWG(t) for t in ("alfworld", "search", "webshop")}
    base = _FakeRefWG("base")
    trainer = _make_trainer(teachers, base)
    batch = _make_batch(task_names)
    trainer.compute_sign_weight_cache(batch)
    return teachers, base, batch


TASKS = ["webshop", "alfworld", "search", "alfworld", "search", "webshop"]


def test_the_base_policy_is_scored_on_every_row():
    """It is the reference all three shifts are measured against, so a row it
    missed has no delta for any teacher at all."""
    _, base, _ = _fixture(TASKS)
    assert sorted(base.rows) == list(range(len(TASKS)))


def test_each_teacher_is_scored_only_where_it_is_off_task():
    """Its own task's rows were already scored by compute_teacher_log_probs and
    are read back from the same cache; re-running them is the same forward twice."""
    teachers, _, _ = _fixture(TASKS)
    for task, wg in teachers.items():
        want = sorted(i for i, t in enumerate(TASKS) if t != task)
        assert sorted(wg.rows) == want, task
        assert all(TASKS[i] != task for i in wg.rows), task


def test_every_column_holds_the_key_the_model_that_filled_it_was_given():
    """The actor reads the columns positionally, so this is the whole contract."""
    teachers, base, batch = _fixture(TASKS)
    ids = batch.batch["sign_cache_ids"]
    order = sorted(teachers)

    base_keys = set(base.cache_ids)
    for i, own in enumerate(TASKS):
        assert int(ids[i, 0]) in base_keys, f"row {i} column 0 is not a base key"
        for c, other in enumerate([t for t in order if t != own]):
            assert int(ids[i, 1 + c]) in set(teachers[other].cache_ids), (
                f"row {i} column {1 + c} should hold {other}'s key"
            )


def test_the_off_task_labels_match_the_columns_they_describe():
    """Only the diagnostics read these, but the pairwise agreement rates are the
    cheapest form of the transferability matrix and they are meaningless if the
    label and the column disagree."""
    _, _, batch = _fixture(TASKS)
    names = batch.meta_info["task_id_names"]
    off = batch.batch["sign_off_tasks"]
    order = sorted(names)
    for i, own in enumerate(TASKS):
        want = [names.index(t) for t in order if t != own]
        assert off[i].tolist() == want, f"row {i} ({own})"


def test_no_row_is_left_without_one_of_its_models():
    _, _, batch = _fixture(TASKS)
    assert bool((batch.batch["sign_cache_ids"] >= 0).all())


def test_keys_are_unique_across_models_and_rows():
    """One key, one entry, one owner. Two models sharing a key would make the
    exchange answer twice and the ownership guard fire -- or, worse, return one
    model's log-probs where the other's were asked for."""
    teachers, base, batch = _fixture(TASKS)
    ids = batch.batch["sign_cache_ids"].flatten().tolist()
    assert len(set(ids)) == len(ids)
    issued = base.cache_ids + [k for wg in teachers.values() for k in wg.cache_ids]
    assert len(set(issued)) == len(issued)


def test_disabled_writes_nothing_and_runs_nothing():
    """enable=false has to be indistinguishable from the plain arm, not merely
    harmless: no extra forward, and no column for the actor to react to."""
    teachers = {t: _FakeRefWG(t) for t in ("alfworld", "search", "webshop")}
    base = _FakeRefWG("base")
    trainer = _make_trainer(teachers, base, enabled=False)
    batch = _make_batch(TASKS)
    trainer.compute_sign_weight_cache(batch)
    assert "sign_cache_ids" not in batch.batch.keys()
    assert base.rows == [] and all(wg.rows == [] for wg in teachers.values())


def test_alias_task_names_are_normalised_before_routing():
    """The batch carries raw names like 'search/nq'; the teachers are keyed by the
    canonical task, and an unnormalised name would make a teacher off-task for
    its own rows."""
    tasks = ["alfworld_easy", "search/nq", "webshop"]
    teachers = {t: _FakeRefWG(t) for t in ("alfworld", "search", "webshop")}
    base = _FakeRefWG("base")
    trainer = _make_trainer(teachers, base)
    batch = _make_batch(tasks)
    batch.meta_info["task_id_names"] = ["alfworld", "search", "webshop"]
    trainer.compute_sign_weight_cache(batch)
    assert teachers["alfworld"].rows == [1, 2]
    assert teachers["search"].rows == [0, 2]
    assert teachers["webshop"].rows == [0, 1]
