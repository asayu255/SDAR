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
"""The gen-2 layers: they must actually act, once, and be reversible.

Three things carry the weight. The freezes have to move the live set out of the
collector's reach -- that is the whole mechanism, and a version returning a
happy report without calling ``gc.freeze()`` would look identical in the log.
The re-freeze has to fire exactly once, at the configured boundary, or it is
either a leak amplifier (every step's transients frozen forever) or a no-op.
And ``ACTOR_GC_FREEZE=0`` has to restore stock behaviour exactly -- all layers
off together -- because that flag is the A/B that decides whether the solo
excursions were this at all; if "off" still swept or froze, the experiment
could only ever confirm.

Every test unfreezes and re-enables in a finally, because these are process-wide
interpreter settings and leaking them would silently change how every later test
in the session collects.
"""

import gc
import importlib
import sys
import weakref

import pytest


def _fresh():
    sys.modules.pop("verl.utils.host_gc", None)
    return importlib.import_module("verl.utils.host_gc")


@pytest.fixture(autouse=True)
def _restore_gc():
    yield
    gc.unfreeze()
    gc.enable()


def test_freeze_moves_the_live_set_out_of_reach(monkeypatch):
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    monkeypatch.delenv("ACTOR_GC_MANUAL", raising=False)
    mod = _fresh()

    gc.unfreeze()
    assert gc.get_freeze_count() == 0

    report = mod.freeze_permanent_heap()

    assert report["enabled"] is True
    # The interpreter always has thousands of tracked containers alive; the
    # assertion that matters is that the frozen set is non-empty, i.e. freeze()
    # was really called and really took the live objects.
    assert report["frozen"] > 0
    assert gc.get_freeze_count() == report["frozen"]
    assert report["seconds"] >= 0.0


def test_freeze_defaults_on_and_leaves_the_collector_enabled(monkeypatch):
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    monkeypatch.delenv("ACTOR_GC_MANUAL", raising=False)
    mod = _fresh()

    assert mod.freeze_enabled() is True
    assert mod.manual_enabled() is False

    mod.freeze_permanent_heap()

    # Freeze alone must not disable automatic collection: objects allocated after
    # this point are ordinary garbage and still have to be swept.
    assert gc.isenabled()


@pytest.mark.parametrize("value", ["0", "false", "False", "no", ""])
def test_freeze_off_is_stock_behaviour(monkeypatch, value):
    monkeypatch.setenv("ACTOR_GC_FREEZE", value)
    mod = _fresh()

    gc.unfreeze()
    report = mod.freeze_permanent_heap()

    assert report["enabled"] is False
    assert report["frozen"] == 0
    # Nothing frozen, collector untouched -- otherwise the A/B is not an A/B.
    assert gc.get_freeze_count() == 0
    assert gc.isenabled()


def test_manual_disables_the_automatic_collector(monkeypatch):
    monkeypatch.setenv("ACTOR_GC_MANUAL", "1")
    mod = _fresh()

    report = mod.freeze_permanent_heap()

    assert report["manual"] is True
    assert not gc.isenabled()


def test_freeze_off_disables_every_layer(monkeypatch):
    """``ACTOR_GC_FREEZE=0`` must switch the whole package off, whatever the
    other knobs say. Off means off: a combination that silently swept, froze,
    or disabled the collector while reporting the feature disabled would make
    the A/B able only to confirm.
    """
    monkeypatch.setenv("ACTOR_GC_FREEZE", "0")
    monkeypatch.setenv("ACTOR_GC_MANUAL", "1")
    monkeypatch.setenv("ACTOR_GC_BOUNDARY_COLLECT", "1")
    monkeypatch.setenv("ACTOR_GC_REFREEZE_STEP", "1")
    mod = _fresh()

    gc.unfreeze()
    mod.freeze_permanent_heap()

    assert gc.isenabled()
    assert mod.collect_at_step_boundary() == 0.0
    assert mod.refreeze_if_due() is None
    assert mod.refreeze_if_due() is None
    assert gc.get_freeze_count() == 0


def test_boundary_collect_runs_by_default(monkeypatch):
    """The per-step sweep is part of the default package now: post-freeze it is
    cheap, it runs in device-idle time, and draining the survivors is what keeps
    the automatic full collection from firing mid-forward."""
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    monkeypatch.delenv("ACTOR_GC_MANUAL", raising=False)
    monkeypatch.delenv("ACTOR_GC_BOUNDARY_COLLECT", raising=False)
    mod = _fresh()
    mod.freeze_permanent_heap()

    before = gc.get_stats()[-1]["collections"]
    elapsed = mod.collect_at_step_boundary()

    assert elapsed >= 0.0
    assert gc.get_stats()[-1]["collections"] > before
    assert gc.isenabled()


def test_boundary_collect_has_an_off_switch(monkeypatch):
    monkeypatch.setenv("ACTOR_GC_BOUNDARY_COLLECT", "0")
    monkeypatch.delenv("ACTOR_GC_MANUAL", raising=False)
    mod = _fresh()
    mod.freeze_permanent_heap()

    assert mod.collect_at_step_boundary() == 0.0


def test_boundary_collect_ignores_the_off_switch_under_manual(monkeypatch):
    """Manual mode turned the automatic collector off; the boundary sweep is its
    replacement schedule and must run even when the default sweep is opted out,
    or the heap never gets collected at all."""
    monkeypatch.setenv("ACTOR_GC_MANUAL", "1")
    monkeypatch.setenv("ACTOR_GC_BOUNDARY_COLLECT", "0")
    mod = _fresh()
    mod.freeze_permanent_heap()

    before = gc.get_stats()[-1]["collections"]
    elapsed = mod.collect_at_step_boundary()

    assert elapsed >= 0.0
    assert gc.get_stats()[-1]["collections"] > before


def test_refreeze_fires_once_after_the_warmup_step(monkeypatch):
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    monkeypatch.delenv("ACTOR_GC_REFREEZE_STEP", raising=False)
    mod = _fresh()
    mod.freeze_permanent_heap()

    # Lists, deliberately: the collector untracks dicts whose contents are all
    # atomic, so 300 small dicts would vanish from the tracked set at the
    # collect() inside the refreeze and the delta would read 1. Lists are never
    # untracked, so these 300 are guaranteed to be what the second freeze takes.
    born_in_step_zero = [[index] for index in range(300)]

    assert mod.refreeze_if_due() is None            # boundary before step 0

    report = mod.refreeze_if_due()                  # boundary before step 1
    assert report is not None
    assert report["frozen_delta"] >= 300            # the warm-up objects went in
    assert report["frozen_total"] == gc.get_freeze_count()

    assert mod.refreeze_if_due() is None            # once, and never again
    assert mod.refreeze_if_due() is None
    del born_in_step_zero


def test_refreeze_step_is_configurable(monkeypatch):
    monkeypatch.setenv("ACTOR_GC_REFREEZE_STEP", "3")
    mod = _fresh()
    mod.freeze_permanent_heap()

    assert mod.refreeze_if_due() is None            # 0 steps completed
    assert mod.refreeze_if_due() is None            # 1
    assert mod.refreeze_if_due() is None            # 2
    assert mod.refreeze_if_due() is not None        # 3 completed -> fire
    assert mod.refreeze_if_due() is None


def test_refreeze_zero_disables_the_second_freeze(monkeypatch):
    monkeypatch.setenv("ACTOR_GC_REFREEZE_STEP", "0")
    mod = _fresh()
    mod.freeze_permanent_heap()

    for _ in range(5):
        assert mod.refreeze_if_due() is None


def test_frozen_cycles_survive_a_collection(monkeypatch):
    """The point of the freeze, stated as a measurement rather than a claim.

    A reference cycle is the one kind of garbage only the collector can reclaim.
    Freeze it and a later full collection must leave it alone -- that is exactly
    what "the sweep no longer visits these objects" means, and it is why the
    call belongs after construction (where the live set is the permanent set)
    and nowhere else.
    """
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    mod = _fresh()

    class _Node:
        pass

    node = _Node()
    node.self_reference = node          # unreachable by refcounting alone
    witness = weakref.ref(node)

    mod.freeze_permanent_heap()
    del node
    gc.collect()

    assert witness() is not None, "a frozen cycle was collected: freeze() did not take"


def test_unfrozen_cycles_are_still_collected(monkeypatch):
    """The control for the test above: without the freeze the same cycle goes.

    Without this, the assertion above would also pass if ``_Node`` happened to be
    kept alive by something else in the test session, and the freeze would be
    credited for it.
    """
    monkeypatch.setenv("ACTOR_GC_FREEZE", "0")
    mod = _fresh()

    class _Node:
        pass

    node = _Node()
    node.self_reference = node
    witness = weakref.ref(node)

    mod.freeze_permanent_heap()
    del node
    gc.collect()

    assert witness() is None


# --------------------------------------------------------------------------- #
# The interaction this repo has and the branch it came from did not: the second
# freeze fires at the top of update_actor, and by then the teacher hidden cache
# is FULL. The trainer clears it at the top of the NEXT step, so the freeze
# necessarily catches one step's worth of bf16 hidden states and puts them in
# the permanent generation.
#
# That is safe, but only for a reason worth pinning: gc.freeze() exempts objects
# from CYCLE collection, not from refcounting. clear() empties the dicts in
# place, the refcounts fall to zero, the memory goes back. If a future change
# gave a cache entry a reference cycle -- an autograd graph, a back-pointer to
# the cache -- the freeze would turn a transient into a permanent leak of one
# step's cache, and it would not show up as anything except memory.
# --------------------------------------------------------------------------- #


def test_a_frozen_teacher_cache_still_releases_its_entries_on_clear():
    import gc as _gc
    import weakref

    torch = pytest.importorskip("torch")
    try:
        from verl.workers.teacher_cache import TeacherHiddenCache
    except Exception as e:  # pragma: no cover - environment without full deps
        pytest.skip(f"teacher cache unavailable: {e}")

    cache = TeacherHiddenCache()
    h = torch.zeros(2, 3, 4)
    lse = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cache.put(
        cache_ids=torch.tensor([0, 1]),
        task="alfworld",
        h=h,
        lse=lse,
        live_mask=lse != 0,
    )
    assert len(cache) == 2

    # Everything the cache is holding right now goes into the permanent
    # generation -- this is exactly what refreeze_if_due does at the top of
    # step 1, with a full cache.
    _gc.collect()
    _gc.freeze()
    try:
        held = [weakref.ref(t) for t in cache._h.values()]
        assert len(held) == 2, "nothing to watch -- an empty all() passes vacuously"
        assert all(r() is not None for r in held)

        cache.clear()
        del h, lse
        _gc.collect()

        assert len(cache) == 0
        assert all(r() is None for r in held), (
            "a frozen cache entry outlived clear(); gc.freeze() exempts cycles "
            "from collection, so a cache entry that gained a reference cycle "
            "would now leak one step of hidden states permanently"
        )
    finally:
        _gc.unfreeze()


def test_the_freeze_does_not_hide_a_later_cycle():
    """The bound on what a freeze can cost.

    Only what is alive AT the freeze is exempt. Everything a later step
    allocates -- including cycles -- is collected normally, which is why the
    trade is 'one init plus one warm-up step' rather than 'the whole run'.
    """
    import gc as _gc
    import weakref

    _gc.collect()
    _gc.freeze()
    try:
        class Node:
            pass

        a, b = Node(), Node()
        a.peer, b.peer = b, a          # a cycle, created after the freeze
        ref = weakref.ref(a)
        del a, b
        _gc.collect()
        assert ref() is None, "a post-freeze cycle must still be collectable"
    finally:
        _gc.unfreeze()
