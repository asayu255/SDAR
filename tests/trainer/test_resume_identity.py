"""The resume that could never succeed.

The sidecar's identity pins what its accumulated numbers mean: the base every
shift is relative to, the teachers, the temperature, and the TASK ORDER every
matrix is indexed by. Three of those four are config and are known the moment
the process starts. The fourth is not -- it is whatever the first batch names.

The worker snapshots the identity in ``load_checkpoint``, which runs before any
batch has arrived, so it recorded ``task_order = []``. The checkpoint it is
compared against was written by a run that had completed a step and therefore
had a real order. Those can never be equal, so EVERY resume of this arm died on
``task_order`` -- after the models were built, the checkpoint located and the
rollout run, which is what made it look like a training failure rather than a
bookkeeping one.

The order is known where the accumulators are built, and that is already where
the restore happens (the accumulators are indexed by task and do not exist until
the first batch names them). These tests pin that it is read from there, and
that filling it in did not turn the check into a rubber stamp.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.trainer.ppo.cross_teacher_kl_weight import (
        AdvantageReliabilityStats,
        CumulativePolicyShiftRMS,
        PreviousStepTaskKLWeightedMean,
        load_sidecar_state,
        resume_identity,
        sidecar_state,
    )
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

TASKS = ["alfworld", "search", "webshop"]
# What the worker can know at load_checkpoint: all config, no batch yet.
SNAPSHOT = {
    "base_path": "Qwen/Qwen3-1.7B",
    "temperature": 1.0,
    "task_order": [],
    "teacher_paths": {t: f"/ckpt/{t}" for t in TASKS},
}


def _accumulated(tasks=TASKS):
    n = len(tasks)
    rms = CumulativePolicyShiftRMS(n_tasks=n, device="cpu")
    mean = PreviousStepTaskKLWeightedMean(n_tasks=n, device="cpu")
    adv = AdvantageReliabilityStats(n_tasks=n, device="cpu", max_groups=8)
    return rms, mean, adv


def _written(tasks=TASKS):
    """A sidecar as the SAVE side writes it -- with a real task order, because
    by then a step has run."""
    rms, mean, adv = _accumulated(tasks)
    return sidecar_state(
        rms=rms, mean=mean, adv=adv, alpha=None,
        identity={**SNAPSHOT, "task_order": list(tasks)},
    )


def test_the_snapshot_alone_fails_the_check_it_was_meant_to_pass():
    """The bug, reproduced. Without the fill-in the two sides compare an empty
    list against the order the checkpoint recorded, which cannot match however
    correct the resume is."""
    rms, mean, adv = _accumulated()
    with pytest.raises(AssertionError, match="task_order"):
        load_sidecar_state(_written(), rms=rms, mean=mean, adv=adv, identity=SNAPSHOT)


def test_the_run_s_own_order_completes_the_snapshot():
    rms, mean, adv = _accumulated()
    load_sidecar_state(
        _written(), rms=rms, mean=mean, adv=adv,
        identity=resume_identity(SNAPSHOT, TASKS),
    )


def test_the_other_three_keys_are_carried_through_untouched():
    """Only the order is unknown at load time. Rebuilding the whole identity
    here instead of completing the snapshot would drop the base, the teachers
    and the temperature -- and load_sidecar_state only checks keys the caller
    names, so the check would go quiet rather than fail."""
    out = resume_identity(SNAPSHOT, TASKS)
    for key in ("base_path", "temperature", "teacher_paths"):
        assert out[key] == SNAPSHOT[key]
    assert set(out) == set(SNAPSHOT)


def test_the_snapshot_is_not_mutated():
    """It is the worker's, held across the resume; a caller that edited it in
    place would leave the next reader with this run's order as if it had been
    checkpointed."""
    before = dict(SNAPSHOT)
    resume_identity(SNAPSHOT, TASKS)
    assert SNAPSHOT == before


# --------------------------------------------------------------------------- #
# it is a check, not a rubber stamp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "order",
    [
        pytest.param(["search", "alfworld", "webshop"], id="permuted"),
        pytest.param(["alfworld", "webshop"], id="a task dropped"),
        pytest.param(["alfworld", "search", "webshop", "sokoban"], id="a task added"),
    ],
)
def test_a_different_task_axis_is_still_refused(order):
    """Every matrix in the sidecar is indexed by this order, so a resume that
    disagrees about it would read another task's RMS and reliability under this
    task's name -- silently, and with a step number in the hundreds."""
    rms, mean, adv = _accumulated(order)
    with pytest.raises(AssertionError, match="task_order"):
        load_sidecar_state(
            _written(), rms=rms, mean=mean, adv=adv,
            identity=resume_identity(SNAPSHOT, order),
        )


def test_a_different_base_is_still_refused():
    """The fill-in touches one key. A resume onto another base checkpoint has to
    stay refused, because every delta is measured against it."""
    rms, mean, adv = _accumulated()
    with pytest.raises(AssertionError, match="base_path"):
        load_sidecar_state(
            _written(), rms=rms, mean=mean, adv=adv,
            identity=resume_identity({**SNAPSHOT, "base_path": "Qwen/Qwen3-4B"}, TASKS),
        )


def test_a_swapped_teacher_is_still_refused():
    rms, mean, adv = _accumulated()
    swapped = {**SNAPSHOT["teacher_paths"], "search": "/ckpt/search_step150"}
    with pytest.raises(AssertionError, match="teacher_paths"):
        load_sidecar_state(
            _written(), rms=rms, mean=mean, adv=adv,
            identity=resume_identity({**SNAPSHOT, "teacher_paths": swapped}, TASKS),
        )


# --------------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------------- #
def test_the_actor_completes_the_identity_from_the_batch_s_own_task_names():
    """The seam the bug lived in. The actor must pass the order it sized the
    accumulators by -- not the worker's snapshot, which is the thing that is
    empty."""
    src = open("verl/workers/actor/dp_actor.py").read()
    block = src[src.index("restored = load_sidecar_state("):]
    block = block[: block.index(")\n") + 1]
    assert "resume_identity(" in block
    assert "task_id_names" in block, "the run's own order, from the batch"
    assert 'getattr(self, "cross_teacher_identity"' in block, "the worker's config half"
