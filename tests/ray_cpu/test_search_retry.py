"""The search client's retry policy, which decides what enters the trajectory.

``call_search_api``'s caller substitutes ``error_msg`` for the retrieved
document, so a retry budget that runs out does not fail the run -- it writes the
string "Search error: ..." into the ``<information>`` block the model is trained
on, with nothing in the metrics to say so. Sharing one retriever between
concurrent runs makes that the expected outcome of a load spike rather than a
remote one, so the rules are pinned here:

* retryable failures (timeout, refused connection, 5xx) are retried, and with
  ``max_retries=None`` they are retried until the retriever answers;
* non-retryable ones (4xx, malformed JSON) give up at once -- waiting cannot
  turn them into a document;
* backoff is capped, so an unbounded retry does not drift into hour-long sleeps.

CPU-only: no server is started, ``session.post`` is a stub.
"""

import json

import pytest

requests = pytest.importorskip("requests")

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import search as mod  # noqa: E402


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        # payload=None with a body is how a non-JSON response is spelled here,
        # so the "was a payload given" question is kept separate from the
        # default one substituted for the successful case.
        self._given_payload = payload
        self._payload = payload if payload is not None else {"result": [[]]}
        self.text = text

    def json(self):
        if self.text and self._given_payload is None:
            raise json.JSONDecodeError("bad", self.text, 0)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


class _Session:
    """Replays a scripted sequence of outcomes, counting attempts."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.timeouts = []

    def post(self, *args, **kwargs):
        self.calls += 1
        self.timeouts.append(kwargs.get("timeout"))
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Backoff is asserted on separately; every other test runs at full speed."""
    slept = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    return slept


def _call(session, **kwargs):
    return mod.call_search_api(
        retrieval_service_url="http://retriever:8001/retrieve",
        query="q",
        session=session,
        log_requests=False,
        **kwargs,
    )


def test_a_successful_call_does_not_retry():
    session = _Session([_Response(payload={"result": ["doc"]})])
    response, error = _call(session)
    assert error is None and response == {"result": ["doc"]}
    assert session.calls == 1


@pytest.mark.parametrize(
    "outcome",
    [
        _Response(status_code=503),
        requests.exceptions.Timeout("read timed out"),
        requests.exceptions.ConnectionError("connection refused"),
    ],
    ids=["http_503", "timeout", "connection_refused"],
)
def test_retryable_failures_are_retried_up_to_the_budget(outcome):
    session = _Session([outcome])
    response, error = _call(session, max_retries=4)
    assert response is None and error is not None
    assert session.calls == 4


@pytest.mark.parametrize(
    "outcome",
    [
        _Response(status_code=503),
        requests.exceptions.Timeout("read timed out"),
        requests.exceptions.ConnectionError("connection refused"),
    ],
    ids=["http_503", "timeout", "connection_refused"],
)
def test_unbounded_retry_waits_for_the_retriever(outcome):
    """The requested behaviour: a saturated shared retriever must not end up
    substituting an error string for the document."""
    # Fails 25 times -- well past the default budget of 10 -- then answers.
    session = _Session([outcome] * 25 + [_Response(payload={"result": ["doc"]})])
    response, error = _call(session, max_retries=None)
    assert error is None and response == {"result": ["doc"]}
    assert session.calls == 26


@pytest.mark.parametrize("budget", [None, 0, -1])
def test_none_and_non_positive_budgets_all_mean_unlimited(budget):
    session = _Session([requests.exceptions.Timeout("t")] * 15 + [_Response()])
    response, error = _call(session, max_retries=budget)
    assert error is None and response is not None
    assert session.calls == 16


def test_a_4xx_gives_up_immediately():
    """No amount of waiting turns a 404 into a document, and retrying forever
    would hang the run instead of surfacing a bad URL."""
    session = _Session([_Response(status_code=404)])
    response, error = _call(session, max_retries=None)
    assert response is None and "404" in error
    assert session.calls == 1


def test_malformed_json_gives_up_immediately():
    session = _Session([_Response(payload=None, text="<html>not json</html>")])
    response, error = _call(session, max_retries=None)
    assert response is None and "JSON" in error
    assert session.calls == 1


def test_backoff_is_capped(_no_sleeping):
    """Linear growth would reach hour-long sleeps under an unbounded retry."""
    session = _Session([requests.exceptions.Timeout("t")] * 200 + [_Response()])
    _call(session, max_retries=None)
    # The coalescer holds a query for its batching window before sending, and
    # that sleep is not backoff -- it lands first and would otherwise read as a
    # 10 ms initial delay.
    backoff = [s for s in _no_sleeping if s != mod._BATCH_WINDOW_S]
    assert backoff, "expected the retry loop to sleep"
    assert max(backoff) <= mod.MAX_RETRY_DELAY
    assert backoff[0] == mod.INITIAL_RETRY_DELAY  # still backs off from 1s
    assert backoff[-1] == mod.MAX_RETRY_DELAY     # and saturates


def test_the_read_budget_is_passed_through_including_none():
    """timeout=None is what makes requests wait indefinitely; it must reach post().

    It reaches it as the READ half of a (connect, read) pair now -- connect is
    bounded separately, because a dead route stalls there and waiting it out
    buys nothing (see test_connect_is_bounded_even_when_read_is_generous). The
    property this test was written for is unchanged: whatever the caller asked
    to wait for a retriever's ANSWER is what requests is told.
    """
    seen = []

    class _Recording(_Session):
        def post(self, *args, **kwargs):
            seen.append(kwargs.get("timeout", "absent"))
            return super().post(*args, **kwargs)

    _call(_Recording([_Response()]), timeout=None)
    _call(_Recording([_Response()]), timeout=600)
    assert [read for _connect, read in seen] == [None, 600]


def test_default_budget_is_unchanged():
    """Configs written before max_retries existed must behave exactly as before."""
    session = _Session([requests.exceptions.Timeout("t")])
    response, error = _call(session)
    assert response is None and error is not None
    assert session.calls == mod.MAX_RETRIES


# --------------------------------------------------------------------------- #
# connect is bounded separately from read
# --------------------------------------------------------------------------- #
def test_connect_is_bounded_even_when_read_is_generous():
    """A scalar timeout applies to BOTH phases, so connect inherits the read's.

    With timeout=600 a connect that never completes waits 600s, and the whole
    coalescing window waits with it: the followers wait on the leader unbounded
    by design, and three pipeline slots share one retriever, so the node stops.
    Bounding connect separately is worth doing on its own account.

    It is NOT, however, where the 40s stalls went -- see
    test_a_wedged_socket_is_bounded_by_the_user_timeout for the log line that
    rules connect out.
    """
    session = _Session([_Response(payload={"result": ["doc"]})])
    _call(session, timeout=600)

    connect, read = session.timeouts[0]
    assert connect == mod._CONNECT_TIMEOUT_S
    assert read == 600, "the read budget must survive; a 250-query batch takes seconds"


def test_a_read_budget_shorter_than_the_connect_bound_wins():
    session = _Session([_Response(payload={"result": ["doc"]})])
    _call(session, timeout=1)
    assert session.timeouts[0] == (1, 1)


def test_no_read_timeout_still_bounds_the_connect():
    """timeout=None means "wait for the retriever", not "wait for a dead route"."""
    session = _Session([_Response(payload={"result": ["doc"]})])
    _call(session, timeout=None)
    connect, read = session.timeouts[0]
    assert connect == mod._CONNECT_TIMEOUT_S and read is None


def test_the_connect_bound_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(mod, "_CONNECT_TIMEOUT_S", 0)
    session = _Session([_Response(payload={"result": ["doc"]})])
    _call(session, timeout=600)
    assert session.timeouts[0] == 600, "0 restores the single scalar requests used to get"


# --------------------------------------------------------------------------- #
# a socket with a request already outstanding
# --------------------------------------------------------------------------- #
def _tcp_user_timeout(options):
    """The TCP_USER_TIMEOUT entry from a socket_options list, or None."""
    import socket

    opt = getattr(socket, "TCP_USER_TIMEOUT", None)
    if opt is None:
        return None
    for level, name, value in options:
        if level == socket.IPPROTO_TCP and name == opt:
            return value
    return None


needs_user_timeout = pytest.mark.skipif(
    not hasattr(__import__("socket"), "TCP_USER_TIMEOUT"),
    reason="TCP_USER_TIMEOUT is Linux-only",
)


@needs_user_timeout
def test_a_wedged_socket_is_bounded_by_the_user_timeout():
    """The 40s stalls were past connect, on a socket with data outstanding.

    MEASURED, run sft-multitask-eval-20260826-201115: 19 samples of 15s with all
    three GPUs at 0%, GPU power 282W -> 99W, and host CPU, disk, network and
    thread count identical to a busy sample. The search log dates them:
    "search recovered after 2 attempts (41s)".

    TWO attempts is the discriminator. EHOSTUNREACH on a withdrawn route returns
    at once, so covering 40s of a genuinely dead route would take about nine
    attempts at this backoff. Two means attempt 1 spent the whole 40s inside one
    socket and attempt 2, on a fresh connection, succeeded immediately -- which
    is a wedged connection, not an outage, and lands past connect() where
    neither the connect bound nor keepalive can reach it. Linux falls back to
    tcp_retries2 there, about 15 minutes.
    """
    value = _tcp_user_timeout(mod._socket_health_options())
    assert value == int(mod._USER_TIMEOUT_S * 1000), "the option is set in milliseconds"
    assert 0 < value <= 60_000, "a bound above a minute is not a bound for this workload"


@needs_user_timeout
def test_the_user_timeout_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(mod, "_USER_TIMEOUT_S", 0)
    assert _tcp_user_timeout(mod._socket_health_options()) is None, "0 keeps the kernel default"


def test_keepalive_survives_the_user_timeout():
    """The two cover different sockets; adding one must not drop the other.

    Keepalive is for an IDLE socket whose peer went away -- nothing outstanding,
    so TCP_USER_TIMEOUT's clock never starts and only probes find the corpse.
    """
    import socket

    options = mod._socket_health_options(idle_s=30, interval_s=10, probes=3)
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    for name, expected in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
        opt = getattr(socket, name, None)
        if opt is not None:
            assert (socket.IPPROTO_TCP, opt, expected) in options


def test_the_options_reach_the_adapter():
    """A list nothing installs is a comment. urllib3's kwarg is not public API."""

    class _PoolManager:
        def __init__(self):
            self.connection_pool_kw = {}

    class _Adapter:
        def __init__(self):
            self.poolmanager = _PoolManager()

    adapter = _Adapter()
    mod._enable_tcp_keepalive(adapter)
    installed = adapter.poolmanager.connection_pool_kw["socket_options"]
    assert installed == mod._socket_health_options()

    # Asserted against the installed list rather than the helper's return value,
    # so this fails on a build where the option was never added at all.
    import socket

    if hasattr(socket, "TCP_USER_TIMEOUT"):
        assert _tcp_user_timeout(installed), "the bound has to reach a real socket to bound anything"


def test_an_adapter_without_the_kwarg_does_not_fail_the_run():
    """A urllib3 that moved its internals costs the tuning, not the evaluation."""

    class _Adapter:
        poolmanager = None  # attribute access raises inside the helper

    mod._enable_tcp_keepalive(_Adapter())  # must not raise


def test_urllib3s_own_defaults_survive():
    """socket_options REPLACES urllib3's defaults; it does not extend them.

    urllib3 ships TCP_NODELAY as its default, so building this list from scratch
    turned Nagle back on for the one session that sends nothing but small JSON
    POSTs. Against a peer that delays its ACKs that is tens of milliseconds on
    every retrieval -- on the hottest path in the evaluation, and invisible.
    """
    import socket

    options = mod._socket_health_options()
    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in options, (
        "TCP_NODELAY is urllib3's default and has to survive being added to"
    )
    for entry in mod._urllib3_default_socket_options():
        assert entry in options, f"dropped one of urllib3's own defaults: {entry}"


def test_an_option_is_not_installed_twice():
    """The health options and urllib3's defaults may name the same option."""
    seen = [(level, name) for level, name, _value in mod._socket_health_options()]
    assert len(seen) == len(set(seen)), seen


def test_urllib3_defaults_fall_back_to_tcp_nodelay(monkeypatch):
    """A urllib3 that moved the attribute must not cost TCP_NODELAY silently."""
    import socket
    import urllib3.connection

    monkeypatch.delattr(urllib3.connection.HTTPConnection, "default_socket_options", raising=False)
    assert mod._urllib3_default_socket_options() == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
