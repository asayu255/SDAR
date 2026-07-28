"""Unit tests for the off-policy / SFT batch-prefetch path.

``OFFPOLICY_BATCH_PREFETCH=1`` moves the driver-side batch preparation of step
k+1 onto a background thread so it overlaps step k's update_actor. It claims to
be bit-identical, and the claim rests on two invariants that are easy to break
and silent when broken:

* ``adjust_batch`` draws its duplicate-row indices from the **global** numpy
  RNG. The prefetch thread must be that stream's only consumer and must consume
  it in step order, or a different set of rows gets duplicated -- with no error.
* ``_offpolicy_batch_iter``'s own seeded generator must be advanced by one
  thread only, or the trajectory permutations change.

Also covered: the prepared batches and their balance statistics arrive in step
order, at most one batch is prepared ahead of the consumer, and a producer
exception reaches the consumer instead of deadlocking it.

CPU-only; no Ray, no GPU, no workers.
"""

import threading
import time

import pytest

np = pytest.importorskip("numpy")

try:
    from verl.trainer.ppo import opd_offpolicy_ray_trainer as mod
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)

OffPolicyOPDRayTrainer = mod.OffPolicyOPDRayTrainer


class _FakeTrainer(OffPolicyOPDRayTrainer):
    """Drives _prepared_batch_iter without Ray, workers or real DataProtos.

    _prepare_batch is replaced by a stand-in that consumes the global numpy RNG
    exactly the way adjust_batch does, so the ordering invariant is what is
    under test rather than DataProto plumbing.
    """

    def __init__(self, n_steps=6, fail_at=None, draw_size=3):
        self.total_training_steps = n_steps
        self._fail_at = fail_at
        self._draw_size = draw_size
        self.prepared = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def _offpolicy_batch_iter(self):
        for step in range(self.total_training_steps):
            yield {"step": step}

    def _prepare_batch(self, batch):
        step = batch["step"]
        if self._fail_at is not None and step == self._fail_at:
            raise RuntimeError(f"boom at step {step}")
        # Stand-in for adjust_batch's np.random.choice over the global stream.
        drawn = np.random.choice(100, self._draw_size, replace=False).tolist()
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        self.prepared.append(step)
        return {"step": step, "drawn": drawn}, {"global_seqlen/mean": float(step)}

    def _consume(self, batch):
        with self._lock:
            self._in_flight -= 1


def _run(prefetch, monkeypatch, **kwargs):
    monkeypatch.setattr(mod, "_BATCH_PREFETCH", prefetch)
    trainer = _FakeTrainer(**kwargs)
    np.random.seed(1234)
    out = []
    for batch, prep_metrics in trainer._prepared_batch_iter():
        out.append((batch, prep_metrics))
        trainer._consume(batch)
    return trainer, out


def test_prefetch_draws_the_same_global_rng_values_as_the_sequential_path(monkeypatch):
    """The invariant that makes the mechanism bit-identical.

    adjust_batch's duplicate-row indices come from the global numpy RNG. If the
    prefetch thread consumed that stream out of order -- or shared it with the
    main thread -- these lists would differ and every step would train on a
    different set of padded rows, silently.
    """
    _, sequential = _run(False, monkeypatch)
    _, prefetched = _run(True, monkeypatch)

    assert [b["drawn"] for b, _ in sequential] == [b["drawn"] for b, _ in prefetched]


def test_batches_and_balance_stats_arrive_in_step_order(monkeypatch):
    _, out = _run(True, monkeypatch, n_steps=8)

    assert [b["step"] for b, _ in out] == list(range(8))
    # The balance statistics must stay paired with their own batch: under
    # prefetch they are computed a step early and carried, not recomputed.
    assert [m["global_seqlen/mean"] for _, m in out] == [float(i) for i in range(8)]


def test_yields_the_same_sequence_with_and_without_prefetch(monkeypatch):
    _, sequential = _run(False, monkeypatch, n_steps=8)
    _, prefetched = _run(True, monkeypatch, n_steps=8)

    assert [b["step"] for b, _ in sequential] == [b["step"] for b, _ in prefetched]
    assert [m for _, m in sequential] == [m for _, m in prefetched]


def test_at_most_one_batch_is_prepared_ahead_of_the_consumer(monkeypatch):
    # A maxsize=1 queue alone would allow three prepared batches to coexist
    # (consumed + queued + in-progress); the semaphore is what caps it at two.
    trainer, _ = _run(True, monkeypatch, n_steps=10)
    assert trainer.max_in_flight <= 2


def test_prefetch_actually_runs_ahead_of_the_consumer(monkeypatch):
    """The point of the mechanism: step k+1 is prepared while step k is in use.

    Asserted by waiting for the producer to get ahead rather than by sleeping a
    fixed amount, so a loaded machine slows the test down instead of failing it.
    """
    monkeypatch.setattr(mod, "_BATCH_PREFETCH", True)
    trainer = _FakeTrainer(n_steps=10)
    np.random.seed(1234)

    it = trainer._prepared_batch_iter()
    first, _ = next(it)
    assert first["step"] == 0

    deadline = time.monotonic() + 10.0
    while len(trainer.prepared) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    # While the consumer still holds step 0, step 1 has already been prepared.
    assert trainer.prepared[:2] == [0, 1]
    assert trainer.max_in_flight == 2

    it.close()


def test_every_step_is_delivered(monkeypatch):
    trainer, out = _run(True, monkeypatch, n_steps=10)
    assert trainer.prepared == list(range(10))
    assert len(out) == 10


def test_producer_exception_reaches_the_consumer(monkeypatch):
    monkeypatch.setattr(mod, "_BATCH_PREFETCH", True)
    trainer = _FakeTrainer(n_steps=6, fail_at=3)
    np.random.seed(1234)

    seen = []
    with pytest.raises(RuntimeError, match="boom at step 3"):
        for batch, _ in trainer._prepared_batch_iter():
            seen.append(batch["step"])
            trainer._consume(batch)

    # Steps before the failure are still delivered; the consumer does not hang.
    assert seen == [0, 1, 2]


def test_disabled_by_default_env(monkeypatch):
    monkeypatch.delenv("OFFPOLICY_BATCH_PREFETCH", raising=False)
    # The module-level flag is read at import; assert the default is off so an
    # unset environment reproduces the pre-existing behaviour exactly.
    assert mod._BATCH_PREFETCH is False or mod._BATCH_PREFETCH == 0
