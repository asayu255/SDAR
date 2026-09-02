"""Where the two retriever lineages meet, which neither branch ran together.

The search client on this branch is assembled from two lines of work that
diverged at c504b27 and never merged:

* cross-teacher (``afde2c1`` then ``96213d2``): coalesce a window of queries into
  one request, and when the retriever answers 5xx to a multi-query request four
  times running, halve it and remember the size that was refused;
* gpu-utilization (``9d70df1`` … ``23b4eea``): bound the connect separately from
  the read, bound a wedged socket with TCP_USER_TIMEOUT, and widen the fan-out
  window from 10 ms to 100 ms so a turn arrives as one request.

They are complementary by design and adversarial by arithmetic: a wider window
collects more queries, and more queries is exactly what makes a request too large
to serve. Each side has thorough tests for its own half against its own defaults
-- ``test_search_batching.py`` drives the split end to end, ``test_search_retry.py``
drives the bounds -- and between them they leave two questions that only arise
once the halves are in one file, which is what this file is for.

CPU-only: no server is started.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

_SRC = (
    Path(__file__).resolve().parents[2]
    / "agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py"
)


def _load():
    """Load the client without dragging in the gym-based environments.

    Same shape as tests/ray_cpu/test_search_batching.py's loader, including the
    removal of the stub afterwards: left in sys.modules it answers other files'
    ``pytest.importorskip("gym")`` for them, and they then run against a
    simulator that is not installed.
    """
    installed = []
    if "gym" not in sys.modules:
        gym = types.ModuleType("gym")
        gym.Env = object
        gym.spaces = types.ModuleType("gym.spaces")
        sys.modules["gym"] = gym
        sys.modules["gym.spaces"] = gym.spaces
        installed = ["gym", "gym.spaces"]
    try:
        spec = importlib.util.spec_from_file_location("_search_under_test", _SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_search_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in installed:
            sys.modules.pop(name, None)


@pytest.fixture
def mod(monkeypatch):
    m = _load()
    monkeypatch.setattr(m.time, "sleep", lambda _d: None)
    return m


class _RecordingSession:
    """Answers every request, recording the timeout and the query count."""

    def __init__(self, status_code=200):
        self.status_code = status_code
        self.timeouts = []
        self.sizes = []

    def post(self, url, headers=None, json=None, timeout=None):
        # The client sends {"query": ...}, a string or a list of them.
        query = json["query"]
        rows = query if isinstance(query, list) else [query]
        self.timeouts.append(timeout)
        self.sizes.append(len(rows))
        outer = self

        class _R:
            status_code = outer.status_code
            text = ""

            def json(self):
                return {"result": [[{"document": {"contents": "d"}}] for _ in rows]}

            def raise_for_status(self):
                pass

        return _R()

    def close(self):
        pass


def test_both_lineages_defaults_survived_the_merge():
    """A merge that dropped one side's knob leaves its mechanism inert, and inert
    is indistinguishable from working until the failure it prevents arrives."""
    m = _load()
    # cross-teacher: coalescing, and the split that bounds its request size
    assert m._BATCH_ENABLED is True
    assert m._SPLIT_AFTER_5XX == 4
    assert m._BATCH_GROW_S == 300
    assert m._BATCH_RETRY_S == 300
    # gpu-utilization: the socket bounds, and the wider fan-out window
    assert m._CONNECT_TIMEOUT_S == 5
    assert m._USER_TIMEOUT_S == 10
    assert m._BATCH_WINDOW_S == pytest.approx(0.100)


def test_the_wider_window_did_not_disarm_the_split_budget():
    """The 100 ms window is the reason a request grows large enough to be
    refused, so the budget that hands it back for splitting is what keeps the
    widening safe. One without the other is the 21-minute stall 96213d2 fixed."""
    m = _load()
    assert m._SPLIT_AFTER_5XX > 0, "splitting is off; a wide window would just stall on 5xx"
    assert m._BATCH_WINDOW_S > 0 and m._BATCH_ENABLED, "batching is off; nothing to split"


def test_the_connect_bound_reaches_the_request_the_splitter_sends(mod):
    """The cross-lineage question neither branch could ask.

    Splitting turns one request into several, and each opens its own socket. The
    bound is applied inside ``_search_api_request``, so it covers the halves for
    free -- but that is a property of where the call sits, and a refactor that
    sent a half by any other route would multiply the exposure the bound exists
    to remove. Pin it on the function every path goes through.
    """
    session = _RecordingSession()
    payload, error = mod._search_api_request(
        "http://retriever", ["a", "b", "c"], topk=1, timeout=600, session=session,
    )
    assert error is None and payload is not None
    assert session.timeouts, "no request was made"
    for t in session.timeouts:
        assert isinstance(t, tuple), "the connect/read split did not reach this call"
        assert t[0] == mod._CONNECT_TIMEOUT_S
        assert t[1] == 600, "the read budget must survive the split"


def test_a_single_query_gets_the_same_bound_as_a_batch(mod):
    """A slice of one is sent as a bare string and carries no 5xx budget, so the
    unbounded wait is the only thing left on it. That makes the connect bound the
    only thing standing between a dead route and an unbounded wait, on exactly
    the path the splitter falls back to."""
    session = _RecordingSession()
    mod._search_api_request(
        "http://retriever", "just one", topk=1, timeout=None, session=session,
    )
    assert session.timeouts == [(mod._CONNECT_TIMEOUT_S, None)]
