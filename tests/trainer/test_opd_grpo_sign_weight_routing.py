"""Driver-side tests for cross-teacher sign weighting in multitask OPD+GRPO.

``test_sign_weights.py`` covers the arithmetic. This file covers the part that
can be wrong without the arithmetic being wrong: which model is asked about
which rows, and whether the answers end up attached to the row they belong to.

Specifically:
* each teacher is queried on exactly the rows it is *not* the teacher of, so the
  on-task pass done by ``compute_teacher_log_probs`` is never repeated;
* the base policy is queried on every row;
* every model is scored at the ON-TASK teacher's top-k ids -- the whole signal
  is a comparison, so a model answering about its own ids would make the signs
  meaningless while still producing plausible numbers;
* each row's two off-task planes are the two teachers that are not its own, in
  the right order for the unanimity test;
* the two modes write what the actor expects, and only that;
* the mechanism is inert when disabled, which is what makes the plain OPD+GRPO
  arm this arm's control.

CPU-only; worker groups are mocked. Skipped if the verl stack is unavailable.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    from verl import DataProto
    from verl.trainer.ppo.sign_weights import SIGN_WEIGHT_KEY
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


K = 2
RESP = 3
TASKS = ("alfworld", "search", "webshop")
# Shift each fake model reports, in nats, relative to the base's 0.0. Chosen so
# every sign combination the table has appears in one batch.
SHIFT = {"alfworld": 1.0, "search": 1.0, "webshop": -1.0}


# log-prob the base reports everywhere. log(0.15) leaves room for a +1 nat shift
# to stay a legal log-probability across K candidates.
_BASE_LOGPROB = float(torch.log(torch.tensor(0.15)))

# Only response position 0 carries a shift; the rest is untouched by every
# model, so it lands in the deadzone. Without this variation *inside* a task the
# per-task mean-1 normalisation would rescale a uniformly-flagged task straight
# back to 1.0 and the position-mode tests would pass vacuously.
_SHIFTED_POS = 0


def _shift_field(shift, bs):
    field = torch.zeros(bs, RESP, K, dtype=torch.float32)
    field[:, _SHIFTED_POS, :] = shift
    return field


class _FakeModelWG:
    """Worker group that reports a fixed shift and records what it was asked.

    ``compute_ref_topk_log_prob_at_ids`` returns log-probs whose shift against
    the base is exactly ``shift`` at the flagged position and 0 elsewhere, so the
    tests can reason about signs without arithmetic. The ids it was handed are
    recorded so a test can prove the caller passed the on-task teacher's support
    rather than letting each model answer about its own.
    """

    def __init__(self, shift):
        self.shift = shift
        self.rows_seen = []
        self.ids_seen = []

    def compute_ref_topk_log_prob_at_ids(self, sub):
        rows = sub.batch["responses"][:, 0].tolist()  # responses[i, 0] == original index
        self.rows_seen.append(rows)
        self.ids_seen.append(sub.batch["teacher_topk_ids"].clone())
        bs = sub.batch["responses"].size(0)
        out = torch.full((bs, RESP, K), _BASE_LOGPROB) + _shift_field(self.shift, bs)
        return DataProto.from_dict(tensors={"topk_logprobs_at_ids": out})


def _make_batch(task_names):
    bs = len(task_names)
    responses = torch.zeros((bs, RESP), dtype=torch.long)
    responses[:, 0] = torch.arange(bs)
    # The on-task teacher's ids: unique per row so a mis-routed call is visible.
    ids = torch.arange(bs).view(bs, 1, 1).expand(bs, RESP, K) * 100 + torch.arange(K)
    on_task = torch.full((bs, RESP, K), _BASE_LOGPROB)
    for i, task in enumerate(task_names):
        on_task[i] += _shift_field(SHIFT[task], 1)[0]
    return DataProto.from_dict(
        tensors={
            "responses": responses,
            "input_ids": responses,
            "attention_mask": torch.ones(bs, RESP, dtype=torch.long),
            "position_ids": torch.arange(RESP).expand(bs, RESP).clone(),
            "response_mask": torch.ones(bs, RESP),
            "teacher_topk_ids": ids.contiguous(),
            "teacher_topk_logprobs": on_task,
            "task_ids": torch.tensor([TASKS.index(t) for t in task_names]),
        },
        non_tensors={"task_name": np.array(task_names, dtype=object)},
    )


def _make_trainer(mode="position", *, enabled=True, deadzone=0.1):
    from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

    # Bypass the heavy __init__: only compute_sign_weights is exercised, so set
    # exactly the attributes it reads.
    trainer = object.__new__(OPDRayTrainer)
    trainer.sign_weight_enabled = enabled
    trainer.sign_weight_mode = mode
    trainer.sign_weight_agree = 1.25
    trainer.sign_weight_disagree = 0.75
    trainer.sign_weight_deadzone = deadzone
    trainer.teacher_wg = {t: _FakeModelWG(SHIFT[t]) for t in TASKS}
    trainer.base_wg = _FakeModelWG(0.0)
    return trainer


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_each_teacher_is_asked_only_about_rows_of_other_tasks():
    trainer = _make_trainer()
    task_names = ["alfworld", "search", "webshop", "alfworld"]
    trainer.compute_sign_weights(_make_batch(task_names))

    expected_off_rows = {
        "alfworld": [1, 2],  # every row that is not alfworld
        "search": [0, 2, 3],
        "webshop": [0, 1, 3],
    }
    for task, wg in trainer.teacher_wg.items():
        assert wg.rows_seen == [expected_off_rows[task]], f"{task} teacher saw the wrong rows"


def test_base_policy_is_asked_about_every_row():
    trainer = _make_trainer()
    trainer.compute_sign_weights(_make_batch(["alfworld", "search", "webshop"]))
    assert trainer.base_wg.rows_seen == [[0, 1, 2]]


def test_every_model_is_scored_at_the_on_task_teachers_ids():
    """The signal is a difference between models, so they must share a support."""
    trainer = _make_trainer()
    batch = _make_batch(["alfworld", "search", "webshop"])
    ids = batch.batch["teacher_topk_ids"]
    trainer.compute_sign_weights(batch)

    assert torch.equal(trainer.base_wg.ids_seen[0], ids)
    for task, wg in trainer.teacher_wg.items():
        rows = wg.rows_seen[0]
        assert torch.equal(wg.ids_seen[0], ids[rows]), f"{task} teacher got the wrong ids"


def test_row_gets_the_two_teachers_that_are_not_its_own():
    """An alfworld row must be judged by search+webshop, and vice versa.

    With these fixed shifts (alfworld +1, search +1, webshop -1) the off-task
    pair is unanimous only for the webshop row, whose judges are alfworld and
    search: it is the one row that can be flagged at all. Any mix-up in the plane
    assembly changes which row that is, so this pins the assembly rather than the
    arithmetic. Read in ``target`` mode because ``position`` mode's per-task
    normalisation deliberately erases a difference that is uniform within a task
    (see :func:`test_position_mode_only_sees_within_task_variation`).
    """
    trainer = _make_trainer("target")
    batch = _make_batch(["alfworld", "search", "webshop"])
    before = batch.batch["teacher_topk_logprobs"].clone()
    trainer.compute_sign_weights(batch)
    after = batch.batch["teacher_topk_logprobs"]

    # alfworld row: judges are search (+1) and webshop (-1) -> split -> neutral
    # search row:   judges are alfworld (+1) and webshop (-1) -> split -> neutral
    assert torch.allclose(after[0], before[0], atol=1e-6)
    assert torch.allclose(after[1], before[1], atol=1e-6)
    # webshop row: judges are alfworld (+1) and search (+1) -> unanimous (+),
    # on-task webshop is (-1) -> conflict -> its mass is pushed down.
    assert after[2, _SHIFTED_POS].exp().sum() < before[2, _SHIFTED_POS].exp().sum()
    # ...and only at the position anyone actually moved.
    assert torch.allclose(after[2, 1], before[2, 1], atol=1e-6)


def test_position_mode_only_sees_within_task_variation():
    """Per-task mean-1 normalisation removes whatever is uniform inside a task.

    That is the point of it -- a task whose every token is flagged the same way
    would otherwise just be distilled harder, which is a change to
    ``teacher_kl_loss_coef`` wearing this mechanism's name. The consequence worth
    knowing is the flip side: this mode can only act on *differences between
    tokens of the same task*, never on a task-wide tilt.
    """
    trainer = _make_trainer("position")
    batch = _make_batch(["webshop"])
    trainer.compute_sign_weights(batch)
    w = batch.batch[SIGN_WEIGHT_KEY]

    # the flagged position is pulled below the untouched ones ...
    assert w[0, _SHIFTED_POS] < w[0, 1]
    # ... but the row as a whole still averages to 1.
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# What each mode writes
# --------------------------------------------------------------------------- #


def test_position_mode_writes_a_token_weight_and_leaves_the_teacher_alone():
    trainer = _make_trainer("position")
    batch = _make_batch(["alfworld", "search", "webshop"])
    before = batch.batch["teacher_topk_logprobs"].clone()
    trainer.compute_sign_weights(batch)

    assert batch.batch[SIGN_WEIGHT_KEY].shape == (3, RESP)
    assert torch.equal(batch.batch["teacher_topk_logprobs"], before)


def test_target_mode_rewrites_the_teacher_and_writes_no_token_weight():
    trainer = _make_trainer("target")
    batch = _make_batch(["alfworld", "search", "webshop"])
    before = batch.batch["teacher_topk_logprobs"].clone()
    trainer.compute_sign_weights(batch)

    assert SIGN_WEIGHT_KEY not in batch.batch
    # only the webshop row is flagged at all (see the routing test above)
    assert torch.equal(batch.batch["teacher_topk_logprobs"][:2], before[:2])
    assert not torch.equal(batch.batch["teacher_topk_logprobs"][2], before[2])


def test_position_weights_are_mean_one_within_each_task():
    trainer = _make_trainer("position")
    task_names = ["alfworld", "alfworld", "webshop", "webshop"]
    batch = _make_batch(task_names)
    trainer.compute_sign_weights(batch)
    w = batch.batch[SIGN_WEIGHT_KEY]
    for t in (0, 2):
        assert w[t : t + 2].mean().item() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Inertness
# --------------------------------------------------------------------------- #


def test_disabled_trainer_touches_nothing():
    trainer = _make_trainer(enabled=False)
    batch = _make_batch(["alfworld", "search", "webshop"])
    before = batch.batch["teacher_topk_logprobs"].clone()
    trainer.compute_sign_weights(batch)

    assert SIGN_WEIGHT_KEY not in batch.batch
    assert torch.equal(batch.batch["teacher_topk_logprobs"], before)
    assert trainer.base_wg.rows_seen == []
    assert all(wg.rows_seen == [] for wg in trainer.teacher_wg.values())


def test_a_deadzone_that_swallows_every_shift_is_a_no_op():
    """The diagnostic case: too large a deadzone silences everything, and the arm
    silently becomes its own control."""
    trainer = _make_trainer("position", deadzone=10.0)
    batch = _make_batch(["alfworld", "search", "webshop"])
    trainer.compute_sign_weights(batch)
    assert torch.allclose(batch.batch[SIGN_WEIGHT_KEY], torch.ones(3, RESP))


def test_metrics_report_the_state_mix():
    trainer = _make_trainer("position")
    batch = _make_batch(["alfworld", "search", "webshop"])
    metrics = {}
    trainer.compute_sign_weights(batch, metrics=metrics)

    # 3 rows x 3 positions x K candidates, of which only position 0 was moved:
    #   webshop row      -> conflict (off-task unanimous +, on-task -)
    #   alfworld, search -> off-task split
    #   every other position -> nobody moved, so the on-task teacher is silent
    slots = 3 * RESP * K
    assert metrics["sign_weight/frac_conflict_on_pos"] == pytest.approx(0.0)
    assert metrics["sign_weight/frac_conflict_on_neg"] == pytest.approx(K / slots)
    assert metrics["sign_weight/frac_neutral_off_task_split"] == pytest.approx(2 * K / slots)
    assert metrics["sign_weight/frac_neutral_on_task_silent"] == pytest.approx(3 * (RESP - 1) * K / slots)
    assert sum(v for k, v in metrics.items() if k.startswith("sign_weight/frac_")) == pytest.approx(1.0)
    assert metrics["sign_weight/w_max"] <= trainer.sign_weight_agree + 1e-6
