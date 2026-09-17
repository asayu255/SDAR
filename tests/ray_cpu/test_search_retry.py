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

    def post(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _un_batched(monkeypatch):
    """One request at a time, which is what every test in this file is about.

    call_search_api now coalesces concurrent queries, and a leader holds its
    window open with a time.sleep before the request goes out. That sleep is not
    a retry backoff, but it lands in _no_sleeping ahead of one -- which is how
    test_backoff_is_capped came to read the 10ms window as the first attempt's
    1s. Batching has its own file (tests/ray_cpu/test_search_batching.py); here
    the wrapper degenerates to a direct send and the retry loop is what is left.
    """
    monkeypatch.setattr(mod, "_BATCH_ENABLED", False)


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
    assert _no_sleeping, "expected the retry loop to sleep"
    assert max(_no_sleeping) <= mod.MAX_RETRY_DELAY
    assert _no_sleeping[0] == mod.INITIAL_RETRY_DELAY  # still backs off from 1s
    assert _no_sleeping[-1] == mod.MAX_RETRY_DELAY     # and saturates


def test_timeout_is_passed_through_including_none():
    """timeout=None is what makes requests wait indefinitely; it must reach post()."""
    seen = []

    class _Recording(_Session):
        def post(self, *args, **kwargs):
            seen.append(kwargs.get("timeout", "absent"))
            return super().post(*args, **kwargs)

    _call(_Recording([_Response()]), timeout=None)
    _call(_Recording([_Response()]), timeout=600)
    assert seen == [None, 600]


def test_default_budget_is_unchanged():
    """Configs written before max_retries existed must behave exactly as before."""
    session = _Session([requests.exceptions.Timeout("t")])
    response, error = _call(session)
    assert response is None and error is not None
    assert session.calls == mod.MAX_RETRIES
