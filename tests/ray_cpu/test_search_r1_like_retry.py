"""The retry policy of the OTHER search client, ``verl/tools/utils/search_r1_like_utils``.

There are two search clients in this tree and they are reached by different
paths. ``agent_system/.../skyrl_gym/tools/search.py`` is the one the multitask
arms run, through the search environment; this one backs ``verl/tools/search_tool.py``,
the verl-native tool-calling path used when ``actor_rollout_ref.rollout.multi_turn.tool_config_path``
names a tool config -- which the off-policy arms do not set. So nothing here is on
the offline KD arm's critical path today, and the reason to keep the two clients
agreeing is that the difference between them is invisible at the call site: the
caller of either substitutes ``error_msg`` for the retrieved document, so a
budget that runs out writes "Search error: ..." into the ``<information>`` block
the model is trained on, with nothing in the metrics to say so.

What is pinned:

* a connection or timeout failure is retried until the service answers, past the
  bounded MAX_RETRIES that applies to everything else;
* the backoff still ramps from INITIAL_RETRY_DELAY and saturates at
  MAX_RETRY_DELAY, so an unbounded retry does not drift into hour-long sleeps;
* ``SEARCH_WAIT_FOR_SERVICE=0`` restores the old bounded budget, because
  "block forever on a retriever that is never coming back" has to remain a choice.

CPU-only: no server is started, ``requests.post`` is a stub.
"""

import importlib.util
import time
import types

import pytest

requests = pytest.importorskip("requests")

_PATH = "verl/tools/utils/search_r1_like_utils.py"


def _load():
    """Load the module by path.

    Imported as a file rather than through ``verl.tools`` so the test does not
    drag in the rollout stack; the client itself needs nothing but ``requests``.
    """
    spec = importlib.util.spec_from_file_location("_srlu_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    m = _load()
    m.time = types.SimpleNamespace(sleep=lambda d: m._slept.append(d), monotonic=time.monotonic)
    m._slept = []
    return m


def _responder(mod, fail_first, exc=None):
    """A post() that fails ``fail_first`` times and then succeeds."""
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise (exc or requests.exceptions.ConnectionError("[Errno 113] No route to host"))

        class _R:
            status_code = 200

            def json(self):
                return {"result": [[]]}

            def raise_for_status(self):
                pass

        return _R()

    mod.requests = types.SimpleNamespace(post=post, exceptions=requests.exceptions)
    return calls


def test_an_unreachable_retriever_is_waited_out_past_the_bounded_budget(mod):
    """The failure this replaces is the quiet one: giving up after MAX_RETRIES
    and training on the error string as though it were a retrieved document."""
    calls = _responder(mod, fail_first=6 * mod.MAX_RETRIES)
    payload, error = mod.call_search_api("http://retriever", ["q"], topk=3)
    assert error is None and payload == {"result": [[]]}
    assert calls["n"] == 6 * mod.MAX_RETRIES + 1


@pytest.mark.parametrize(
    "exc",
    [requests.exceptions.ConnectionError("refused"), requests.exceptions.Timeout("t")],
)
def test_both_unreachable_shapes_are_waited_out(mod, exc):
    """Connection refused and timeout are the same condition seen from two sides;
    the old code handled them in two near-identical branches and it was the
    duplication that let them drift."""
    calls = _responder(mod, fail_first=mod.MAX_RETRIES + 5, exc=exc)
    _, error = mod.call_search_api("http://retriever", ["q"], topk=3)
    assert error is None and calls["n"] == mod.MAX_RETRIES + 6


def test_the_backoff_ramps_and_then_saturates(mod):
    """Linear growth would reach hour-long sleeps under an unbounded retry, and a
    flat cap from the first attempt would hammer a service that is merely slow."""
    _responder(mod, fail_first=3 * mod.MAX_RETRY_DELAY)
    mod.call_search_api("http://retriever", ["q"], topk=3)
    assert mod._slept, "expected the retry loop to sleep"
    assert mod._slept[0] == mod.INITIAL_RETRY_DELAY
    assert max(mod._slept) <= mod.MAX_RETRY_DELAY
    assert mod._slept[-1] == mod.MAX_RETRY_DELAY
    assert mod._slept == sorted(mod._slept), "the backoff must be monotone"


def test_opting_out_restores_the_bounded_budget(mod):
    """Blocking forever on a retriever that is never coming back has to stay a
    choice, which is what SEARCH_WAIT_FOR_SERVICE=0 is for."""
    mod.WAIT_FOR_SERVICE = False
    calls = {"n": 0}

    def always_fail(*args, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("down")

    mod.requests = types.SimpleNamespace(post=always_fail, exceptions=requests.exceptions)
    payload, error = mod.call_search_api("http://retriever", ["q"], topk=3)
    assert payload is None and error is not None
    assert calls["n"] == mod.MAX_RETRIES


def test_a_non_retryable_failure_still_gives_up_at_once(mod):
    """Waiting cannot turn a 4xx or malformed JSON into a document, and retrying
    one under an unbounded policy would block the rollout forever."""
    calls = {"n": 0}

    def bad_request(*args, **kwargs):
        calls["n"] += 1

        class _R:
            status_code = 404
            text = "nope"

            def json(self):
                return {}

            def raise_for_status(self):
                raise requests.exceptions.HTTPError("404")

        return _R()

    mod.requests = types.SimpleNamespace(post=bad_request, exceptions=requests.exceptions)
    payload, error = mod.call_search_api("http://retriever", ["q"], topk=3)
    assert payload is None and error is not None
    assert calls["n"] == 1, "a 4xx must not be retried at all"


def test_server_errors_keep_the_bounded_budget_even_while_waiting_for_service(mod):
    """5xx means the service ANSWERED, so it is not the unreachable case the
    indefinite wait exists for -- it keeps the old budget."""
    calls = {"n": 0}

    def server_error(*args, **kwargs):
        calls["n"] += 1

        class _R:
            status_code = 503
            text = ""

            def json(self):
                return {}

            def raise_for_status(self):
                pass

        return _R()

    mod.requests = types.SimpleNamespace(post=server_error, exceptions=requests.exceptions)
    payload, error = mod.call_search_api("http://retriever", ["q"], topk=3)
    assert payload is None and error is not None
    assert calls["n"] == mod.MAX_RETRIES
