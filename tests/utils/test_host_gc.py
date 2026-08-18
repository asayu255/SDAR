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
"""The gen-2 freeze: it must actually freeze, and it must be reversible.

Two things carry the weight. The freeze has to move the live set out of the
collector's reach -- that is the whole mechanism, and a version that returns a
happy report without calling ``gc.freeze()`` would look identical in the log.
And ``ACTOR_GC_FREEZE=0`` has to restore stock behaviour exactly, because that
flag is the A/B that decides whether the solo excursions were this at all; if
"off" still froze, the experiment could only ever confirm.

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


def test_manual_needs_the_freeze_to_be_on(monkeypatch):
    """``ACTOR_GC_FREEZE=0 ACTOR_GC_MANUAL=1`` must not disable the collector.

    Off means off. A combination that silently turned automatic collection off
    while reporting the feature disabled would be the worst of both: no freeze,
    no sweeps, and a heap that grows until the step boundary.
    """
    monkeypatch.setenv("ACTOR_GC_FREEZE", "0")
    monkeypatch.setenv("ACTOR_GC_MANUAL", "1")
    mod = _fresh()

    mod.freeze_permanent_heap()

    assert gc.isenabled()
    assert mod.collect_at_step_boundary() == 0.0


def test_boundary_collect_is_a_noop_under_automatic_collection(monkeypatch):
    """With Python still collecting on its own schedule, a forced full sweep here
    would be extra work, not relocated work."""
    monkeypatch.delenv("ACTOR_GC_FREEZE", raising=False)
    monkeypatch.setenv("ACTOR_GC_MANUAL", "0")
    mod = _fresh()

    assert mod.collect_at_step_boundary() == 0.0


def test_boundary_collect_runs_when_manual(monkeypatch):
    monkeypatch.setenv("ACTOR_GC_MANUAL", "1")
    mod = _fresh()
    mod.freeze_permanent_heap()

    before = gc.get_stats()[-1]["collections"]
    elapsed = mod.collect_at_step_boundary()

    assert elapsed >= 0.0
    assert gc.get_stats()[-1]["collections"] > before


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
