"""update_policy's stages must be attributable, and free when not asked for.

The driver's profiler sees one opaque ``update_actor`` bucket because
update_policy runs in a worker process. dp_actor._actor_phase gives that bucket
an interior. This covers the parts that can be checked without a GPU: the gating
(off by default, rank 0 only), that the phase names reach the profiler in order,
and that the optional synchronize is wired.
"""

import importlib
import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor import dp_actor as mod
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def test_phase_is_a_no_op_when_stage_profiling_is_off():
    """The default path must not touch the profiler at all."""
    with mock.patch.object(mod, "_PROFILE_STAGES", False), \
         mock.patch.object(mod.gpu_profiler, "push_phase") as push, \
         mock.patch.object(mod.gpu_profiler, "pop_phase") as pop:
        with mod._actor_phase("actor.fwd"):
            pass
        push.assert_not_called()
        pop.assert_not_called()


def test_phases_are_pushed_and_popped_in_order():
    calls = []
    with mock.patch.object(mod, "_PROFILE_STAGES", True), \
         mock.patch.object(mod, "_SYNC_PHASES", False), \
         mock.patch.object(mod.gpu_profiler, "push_phase", lambda n: calls.append(("push", n))), \
         mock.patch.object(mod.gpu_profiler, "pop_phase", lambda n: calls.append(("pop", n))):
        with mod._actor_phase("actor.fwd"):
            pass
        with mod._actor_phase("actor.bwd"):
            pass
    assert calls == [
        ("push", "actor.fwd"), ("pop", "actor.fwd"),
        ("push", "actor.bwd"), ("pop", "actor.bwd"),
    ]


def test_a_raising_body_still_pops():
    """An unbalanced stack would mis-tag every later sample in the run."""
    calls = []
    with mock.patch.object(mod, "_PROFILE_STAGES", True), \
         mock.patch.object(mod, "_SYNC_PHASES", False), \
         mock.patch.object(mod.gpu_profiler, "push_phase", lambda n: calls.append(("push", n))), \
         mock.patch.object(mod.gpu_profiler, "pop_phase", lambda n: calls.append(("pop", n))):
        with pytest.raises(RuntimeError):
            with mod._actor_phase("actor.bwd"):
                raise RuntimeError("boom")
    assert calls == [("push", "actor.bwd"), ("pop", "actor.bwd")]


def test_sync_phases_synchronizes_on_both_edges():
    """Opt-in only: it serializes what the run would otherwise overlap."""
    device = mock.MagicMock()
    with mock.patch.object(mod, "_PROFILE_STAGES", True), \
         mock.patch.object(mod, "_SYNC_PHASES", True), \
         mock.patch.object(mod, "get_torch_device", lambda: device), \
         mock.patch.object(mod.gpu_profiler, "push_phase"), \
         mock.patch.object(mod.gpu_profiler, "pop_phase"):
        with mod._actor_phase("actor.optim"):
            pass
    assert device.synchronize.call_count == 2


def _phases_used():
    """Every actor.* phase actually opened anywhere in the worker.

    Read from the files rather than inspect.getsource: update_policy is wrapped
    by GPUMemoryLogger, which does not use functools.wraps, so getsource returns
    the decorator's inner function and finds nothing.

    Both files, because _actor_phase is defined in dp_actor but not only used
    there: actor.h2d wraps the batch's move onto the device, which happens in
    fsdp_workers before update_policy is called at all.
    """
    import os

    paths = [mod.__file__.replace(".pyc", ".py")]
    paths.append(os.path.join(os.path.dirname(paths[0]), "..", "..", "workers", "fsdp_workers.py"))
    used = set()
    for path in paths:
        for line in open(path).read().splitlines():
            if '_actor_phase("' in line and not line.strip().startswith("def "):
                used.add(line.split('_actor_phase("')[1].split('"')[0])
    return used


def test_the_instrumented_stages_are_the_ones_the_report_orders():
    """A stage the report does not know about lands in the alphabetical tail,
    which silently breaks the top-to-bottom reading of a step's interior."""
    from verl.utils import gpu_profiler

    used = _phases_used()
    assert used, "update_policy is not instrumented"
    assert used <= set(gpu_profiler._PHASE_ORDER), used - set(gpu_profiler._PHASE_ORDER)
    # Every stage the report orders should actually be produced, or the ordering
    # list is documenting something that no longer exists.
    ordered_stages = {p for p in gpu_profiler._PHASE_ORDER if p.startswith("actor.")}
    assert used == ordered_stages, ordered_stages ^ used
