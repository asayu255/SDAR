"""The vLLM session is counted by depth, not by a bool.

``multi_turn_loop`` opens a session per rollout. ``_validate`` opens one around
the WHOLE validation, so on every batch after the first the two are nested. With
a bool, the inner scope's ``end_rollout_session`` releases the outer one, vLLM
sleeps, and the next batch wakes it again -- the hoist silently does nothing
while the log still says a session was opened. That failure costs the 10.4% of
the evaluation wall the hoist exists to save and shows up nowhere, which is why
the counter is pinned here rather than left to the run.

The worker itself needs a live FSDP/vLLM stack, so what is exercised is the two
methods against a stand-in sharding manager that records its enter/exit calls.
That is exactly the surface the depth counter lives on.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


class _Manager:
    """Stands in for FSDPVLLMShardingManager; counts wake/sleep."""

    def __init__(self):
        self.enters = 0
        self.exits = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        self.exits += 1
        return False


@pytest.fixture
def vllm_manager_is(monkeypatch):
    """Make ``begin_rollout_session``'s isinstance check answer for a stand-in.

    The real ``verl.workers.sharding_manager.fsdp_vllm`` cannot be imported
    without vLLM installed, and the method imports it by name at call time, so
    the module is substituted in ``sys.modules`` rather than patched in place.
    """
    import sys
    import types

    def _install(cls):
        mod = types.ModuleType("verl.workers.sharding_manager.fsdp_vllm")
        mod.FSDPVLLMShardingManager = cls
        monkeypatch.setitem(sys.modules, "verl.workers.sharding_manager.fsdp_vllm", mod)

    return _install


def _worker(manager=None):
    """A worker carrying only what the two session methods touch."""
    w = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    w._is_rollout = True
    w._rank = 0
    w.rollout_sharding_manager = manager
    return w


def _unregistered(method):
    """The method itself, past whatever @register wrapped it in."""
    return getattr(method, "__wrapped__", method)


def _begin(w):
    _unregistered(ActorRolloutRefWorker.begin_rollout_session)(w)


def _end(w):
    _unregistered(ActorRolloutRefWorker.end_rollout_session)(w)


def test_a_nested_scope_does_not_wake_the_engine_twice(vllm_manager_is):
    vllm_manager_is(_Manager)
    m = _Manager()
    w = _worker(m)

    _begin(w)          # _validate's hoist
    _begin(w)          # multi_turn_loop's own, batch 1
    _end(w)
    assert (m.enters, m.exits) == (1, 0), "the inner close must not sleep vLLM"

    _begin(w)          # batch 2
    _end(w)
    assert (m.enters, m.exits) == (1, 0), "the outer session must still hold"

    _end(w)            # the hoist closes
    assert (m.enters, m.exits) == (1, 1)


def test_the_session_still_closes_when_nobody_nests(vllm_manager_is):
    """The pre-hoist shape: one scope per rollout, opened and closed."""
    vllm_manager_is(_Manager)
    m = _Manager()
    w = _worker(m)
    _begin(w)
    _end(w)
    assert (m.enters, m.exits) == (1, 1)
    assert not w._rollout_session_active


def test_an_unpaired_close_is_a_no_op(vllm_manager_is):
    """``rollout_session`` closes in a finally, so a begin that DECLINED (no
    manager, not vLLM's) is still followed by an end. It must not drive the depth
    negative and then swallow the next real close."""
    vllm_manager_is(_Manager)
    m = _Manager()
    w = _worker(m)

    _end(w)
    assert getattr(w, "_rollout_session_depth", 0) == 0
    assert (m.enters, m.exits) == (0, 0)

    _begin(w)
    _end(w)
    assert (m.enters, m.exits) == (1, 1)


def test_a_non_vllm_manager_is_declined_without_entering(vllm_manager_is):
    class _Other:
        pass

    vllm_manager_is(_Manager)
    w = _worker(_Other())

    _begin(w)
    _end(w)
    assert getattr(w, "_rollout_session_active", False) is False
    assert getattr(w, "_rollout_session_depth", 0) == 0


def test_a_missing_manager_is_declined_before_the_vllm_import():
    """SKIP_ROLLOUT_BUILD leaves no manager at all. Deciding that before the
    import is what keeps this hook from being the one place a skipped build still
    pays for vLLM -- and it is why this test needs no stand-in module at all,
    which on a box without vLLM is the difference between passing and erroring."""
    w = _worker(None)

    _begin(w)
    _end(w)
    assert getattr(w, "_rollout_session_active", False) is False


def test_the_generate_count_survives_the_nesting(vllm_manager_is):
    """``end_rollout_session`` reports how many generate calls one wake served,
    and a session that served 1 is the symptom of a hoist that did not take. The
    counter is reset by the OUTERMOST open only, or every nested batch would
    reset it and the number would always read 1."""
    vllm_manager_is(_Manager)
    m = _Manager()
    w = _worker(m)

    _begin(w)
    for _ in range(3):
        _begin(w)
        # what generate_sequences does on the session path
        w._rollout_session_generates = getattr(w, "_rollout_session_generates", 0) + 1
        _end(w)
    assert w._rollout_session_generates == 3
    _end(w)
    assert (m.enters, m.exits) == (1, 1)
