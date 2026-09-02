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
"""Coalescing the retriever's requests, and what must survive it.

The index is Flat, so one search reads the whole 32 GB of embeddings however many
queries it is handed. Measured against the retriever in use: one unloaded query is
80 ms, 126 concurrent ones take 7.5 s. That 93x is the index being re-read 126
times, and it is why nothing on the server side fixes it -- both server processes
already share the same two GPUs and the same bandwidth, and there is no room on
them for a third copy of the index.

So the fix is to stop sending 126 requests. The environments must not be able to
tell: each caller gets the same documents in the same single-query response shape
it always got, and the queries keep their order. These tests hold that, and the
failure modes that would make a coalescer worse than no coalescer -- a leader that
dies with followers waiting on it, and a server that does not accept a list at all.
"""

import importlib.util
import os
import pathlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py"
)


def _load_search_tool():
    """Load the client without dragging in the gym-based environments.

    Importing the module normally walks the search package's __init__, which
    imports envs.py, which imports gym. The HTTP client under test needs none of
    it, and a CPU test box has no reason to install a simulator to check how
    requests are batched.
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
        spec = importlib.util.spec_from_file_location("search_tool", _PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["search_tool"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Only needed while the import runs. Left in sys.modules it would answer
        # other files' pytest.importorskip("gym") for them, and they would then
        # run against a simulator that is not there --
        # tests/trainer/test_webshop_worker_memory.py stopped skipping and
        # started failing, from a file it has nothing to do with.
        for name in installed:
            sys.modules.pop(name, None)


search_tool = _load_search_tool()


@pytest.fixture(autouse=True)
def _fresh_coalescer(monkeypatch):
    monkeypatch.setattr(search_tool, "_COALESCER", search_tool._Coalescer())
    monkeypatch.setattr(search_tool, "_BATCH_ENABLED", True)
    monkeypatch.setattr(search_tool, "_BATCH_WINDOW_S", 0.05)


class _Server:
    """Records what it was asked for, and answers in the retriever's shape."""

    def __init__(self, accepts_lists=True, fail_all=False):
        self.accepts_lists = accepts_lists
        self.fail_all = fail_all
        self.requests = []
        self._lock = threading.Lock()

    def __call__(self, query):
        with self._lock:
            self.requests.append(query)
        if self.fail_all:
            return None, "retriever is down"
        if isinstance(query, list):
            if not self.accepts_lists:
                return None, "API Request Error: 422 Unprocessable Entity"
            return {"result": [[{"document": {"contents": f"doc for {q}"}}] for q in query]}, None
        return {"result": [[{"document": {"contents": f"doc for {query}"}}]]}, None


def _concurrent(server, queries, **kw):
    """Call through the coalescer from one thread per query, as the rollout does."""
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [
            pool.submit(search_tool._COALESCER.call, "http://r/retrieve", q, server, **kw)
            for q in queries
        ]
        return [f.result() for f in futures]


def test_concurrent_queries_collapse_into_a_handful_of_requests():
    """126 requests were the problem. Exactly one is not the requirement --
    a thread that starts after the window closes correctly forms the next batch,
    and on a loaded box some will. What must hold is that the count collapses and
    that no query is lost or sent twice."""
    server = _Server()
    queries = [f"q{i}" for i in range(126)]
    _concurrent(server, queries)

    assert len(server.requests) <= 4, server.requests
    sent = [q for request in server.requests for q in (request if isinstance(request, list) else [request])]
    assert sorted(sent) == sorted(queries)


def test_each_caller_gets_its_own_documents():
    """The whole point: coalescing must not shuffle which answer goes where."""
    server = _Server()
    queries = [f"q{i}" for i in range(64)]
    out = _concurrent(server, queries)

    for query, (response, error) in zip(queries, out):
        assert error is None
        assert response["result"][0][0]["document"]["contents"] == f"doc for {query}"


def test_the_response_keeps_the_single_query_shape():
    """Callers index result[0]. If a coalesced answer handed back the whole
    batch, every environment would read another trajectory's document."""
    server = _Server()
    (response, _), = _concurrent(server, ["only"])

    assert list(response.keys()) == ["result"]
    assert len(response["result"]) == 1


def test_a_lone_query_is_sent_as_a_bare_string():
    """Un-batched behaviour byte for byte, so a single environment never probes
    a server for list support it might not have."""
    server = _Server()
    _concurrent(server, ["alone"])

    assert server.requests == ["alone"]


def test_a_server_that_refuses_lists_still_answers_every_caller():
    server = _Server(accepts_lists=False)
    queries = [f"q{i}" for i in range(8)]
    out = _concurrent(server, queries)

    for query, (response, error) in zip(queries, out):
        assert error is None, "a rejected batch must not become the retrieved document"
        assert response["result"][0][0]["document"]["contents"] == f"doc for {query}"


def test_a_server_that_refuses_lists_is_not_asked_twice():
    server = _Server(accepts_lists=False)
    _concurrent(server, ["a", "b", "c"])
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False

    before = len(server.requests)
    _concurrent(server, ["d", "e", "f"])
    assert all(isinstance(r, str) for r in server.requests[before:])


def test_a_real_outage_still_reaches_the_caller():
    """Falling back to single requests must not swallow a genuine failure --
    the caller turns error_msg into the observation the model reads."""
    server = _Server(fail_all=True)
    out = _concurrent(server, ["a", "b"])

    for response, error in out:
        assert response is None
        assert "down" in error


def test_a_leader_that_raises_does_not_hang_its_followers():
    """Followers wait without a timeout, so the leader filling every slot in a
    finally is the only thing between an exception and a wedged rollout."""

    def exploding(query):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _concurrent(exploding, ["a", "b", "c", "d"])


def test_queries_that_miss_the_window_form_the_next_batch():
    server = _Server()
    first = _concurrent(server, ["a", "b"])
    second = _concurrent(server, ["c", "d"])

    assert len(server.requests) == 2
    assert server.requests == [["a", "b"], ["c", "d"]]
    assert second[0][0]["result"][0][0]["document"]["contents"] == "doc for c"


def test_different_topk_do_not_share_a_batch():
    """A window may only hold requests that were identical apart from the query."""
    server = _Server()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(search_tool._COALESCER.call, "http://r/retrieve", "a", server, topk=3),
            pool.submit(search_tool._COALESCER.call, "http://r/retrieve", "b", server, topk=10),
        ]
        [f.result() for f in futures]

    assert len(server.requests) == 2


def test_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setattr(search_tool, "_BATCH_ENABLED", False)
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False


def test_a_result_count_that_does_not_match_is_an_error_not_a_mix_up():
    """Silently zipping a short result list onto the batch would give some
    environments another trajectory's document and the rest nothing."""

    def short(query):
        return {"result": [[{"document": {"contents": "only one"}}]]}, None

    out = _concurrent(short, ["a", "b", "c"])

    for response, error in out:
        assert response is None
        assert "3 queries" in error


def test_a_url_that_rejects_lists_stops_being_batched():
    server = _Server(accepts_lists=False)
    _concurrent(server, ["a", "b", "c"])
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False

    # and the next round goes singly, without probing the server again
    server.requests.clear()
    _concurrent(server, ["d", "e"])
    assert all(isinstance(request, str) for request in server.requests)


def test_the_disable_lifts_after_the_cooldown(monkeypatch):
    """The usual reason a URL starts accepting lists is that the retriever was
    restarted. Nothing else would ever tell us, and an evaluation runs for hours
    -- so the flag has to expire rather than be permanent."""
    clock = [1000.0]
    monkeypatch.setattr(search_tool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_tool, "_BATCH_RETRY_S", 300.0)

    server = _Server(accepts_lists=False)
    _concurrent(server, ["a", "b", "c"])
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False

    clock[0] += 299.0
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False
    clock[0] += 2.0
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is True


def test_a_restarted_retriever_is_picked_up_without_restarting_the_run(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(search_tool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_tool, "_BATCH_RETRY_S", 300.0)

    server = _Server(accepts_lists=False)
    _concurrent(server, ["a", "b", "c"])

    server.accepts_lists = True  # the retriever was restarted with the new code
    clock[0] += 301.0
    server.requests.clear()
    results = _concurrent(server, ["d", "e", "f"])

    assert any(isinstance(request, list) for request in server.requests), server.requests
    assert [response["result"][0][0]["document"]["contents"] for response, _ in results] == [
        "doc for d",
        "doc for e",
        "doc for f",
    ]


def test_the_cooldown_can_be_turned_off(monkeypatch):
    """A retriever that will never be restarted should not be probed forever."""
    clock = [1000.0]
    monkeypatch.setattr(search_tool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_tool, "_BATCH_RETRY_S", 0.0)

    _concurrent(_Server(accepts_lists=False), ["a", "b", "c"])
    clock[0] += 100000.0
    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is False


# ---------------------------------------------------------------------------
# A window has no size of its own, and the retriever has a ceiling. Where the
# two meet, every attempt answers 5xx and an unbounded retry waits on it for as
# long as the run lasts -- which is how one validation pass sat on a single
# request for 21 minutes with the GPUs idle. Measured on the retriever in use,
# 383 queries in a request were served and 384 were not, so the ceiling is not a
# constant either side can be configured with: it is free memory over the cost
# of a query, and it moves. These hold the way out -- halve the request, learn
# the size, and never mistake an outage for it.
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.text = "Internal Server Error" if status_code >= 500 else "ok"
        self._payload = payload if payload is not None else {"result": [[]]}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise search_tool.requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class _HttpSession:
    """Answers the first ``failures`` posts with ``status``, then serves."""

    def __init__(self, failures, status=500, raises=None):
        self.failures = failures
        self.status = status
        self.raises = raises
        self.posts = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts += 1
        if self.posts <= self.failures:
            if self.raises is not None:
                raise self.raises
            return _Response(self.status)
        return _Response(200)

    def close(self):
        pass


def _request(session, query, **kw):
    return search_tool._search_api_request(
        retrieval_service_url="http://r/retrieve",
        query=query,
        session=session,
        log_requests=False,
        max_retries=None,
        **kw,
    )


def test_a_batch_that_keeps_drawing_5xx_is_handed_back_for_splitting(monkeypatch):
    """Raised, not returned: an error that is returned becomes the
    <information> block the student reads, and nothing here belongs there."""
    monkeypatch.setattr(search_tool.time, "sleep", lambda _s: None)
    session = _HttpSession(failures=10**6)

    with pytest.raises(search_tool._RetryableServerError) as caught:
        _request(session, ["a", "b"], server_error_budget=3)

    assert caught.value.status_code == 500
    assert caught.value.attempts == 3
    assert session.posts == 3


def test_a_single_query_still_waits_out_a_5xx(monkeypatch):
    """The unbounded wait is the whole point of max_retries=None, and a lone
    query has nothing to give up -- it cannot be made smaller."""
    monkeypatch.setattr(search_tool.time, "sleep", lambda _s: None)
    session = _HttpSession(failures=8)

    response, error = _request(session, "a")

    assert error is None and response is not None
    assert session.posts == 9


def test_an_outage_does_not_count_towards_the_split_budget(monkeypatch):
    """A refused connection says nothing about how large the request was. The
    retriever outage this client already survives (44 s of No route to host,
    every query recovered) must not be turned into a size hunt."""
    monkeypatch.setattr(search_tool.time, "sleep", lambda _s: None)
    session = _HttpSession(
        failures=8, raises=search_tool.requests.exceptions.ConnectionError("No route to host")
    )

    response, error = _request(session, ["a", "b"], server_error_budget=3)

    assert error is None and response is not None
    assert session.posts == 9


def test_the_budget_is_off_by_default(monkeypatch):
    """Existing callers must keep the behaviour they were written against."""
    monkeypatch.setattr(search_tool.time, "sleep", lambda _s: None)
    session = _HttpSession(failures=8)

    response, error = _request(session, ["a", "b"])

    assert error is None and session.posts == 9


class _SizeLimitedServer:
    """Serves at most ``limit`` queries per request, and refuses larger ones the
    way the retry loop reports a retriever that keeps answering 5xx."""

    def __init__(self, limit):
        self.limit = limit
        self.requests = []
        self._lock = threading.Lock()

    @property
    def served(self):
        return [n for n in self.requests if n <= self.limit]

    def __call__(self, query):
        queries = query if isinstance(query, list) else [query]
        with self._lock:
            self.requests.append(len(queries))
        if len(queries) > self.limit:
            raise search_tool._RetryableServerError("Server Error (500)", 500, 4)
        if isinstance(query, list):
            return {"result": [[{"document": {"contents": f"doc for {q}"}}] for q in query]}, None
        return {"result": [[{"document": {"contents": f"doc for {query}"}}]]}, None


def test_a_request_too_large_is_split_until_it_is_served():
    server = _SizeLimitedServer(limit=3)
    queries = [f"q{i}" for i in range(16)]

    out = _concurrent(server, queries)

    assert all(error is None for _, error in out)
    assert [response["result"][0][0]["document"]["contents"] for response, _ in out] == [
        f"doc for {q}" for q in queries
    ]
    # Every query served exactly once, and by fewer requests than sending them
    # one at a time would have taken -- the fallback this replaces sent 16.
    assert sum(server.served) == 16
    assert len(server.requests) < 16, server.requests


def test_the_size_that_was_refused_is_not_sent_again():
    """The cost of finding the ceiling is paid once. A window that re-probed it
    every turn would spend the run bisecting instead of retrieving."""
    server = _SizeLimitedServer(limit=3)
    _concurrent(server, [f"q{i}" for i in range(16)])
    probes = len(server.requests)

    server.requests.clear()
    out = _concurrent(server, [f"p{i}" for i in range(16)])

    assert all(error is None for _, error in out)
    assert server.requests == server.served, server.requests  # nothing refused
    assert len(server.requests) < probes


def test_splitting_does_not_disable_batching_for_the_url():
    """A retriever that cannot serve 400 queries at once still takes lists --
    dropping to one request per query is the outcome to avoid, not the cure."""
    server = _SizeLimitedServer(limit=3)
    _concurrent(server, [f"q{i}" for i in range(16)])

    assert search_tool._COALESCER.enabled_for("http://r/retrieve") is True
    assert max(server.served) > 1, server.requests


def test_the_cap_is_relaxed_so_a_box_that_frees_memory_gets_big_batches_again(monkeypatch):
    """The ceiling moves both ways. A cap learned while the trainer held its KV
    cache would otherwise outlive it for the rest of the run."""
    clock = [1000.0]
    monkeypatch.setattr(search_tool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(search_tool, "_BATCH_GROW_S", 300.0)

    coalescer = search_tool._COALESCER
    coalescer._learn_cap("http://r/retrieve", 16)
    assert coalescer.batch_cap("http://r/retrieve") == 8

    clock[0] += 299.0
    assert coalescer.batch_cap("http://r/retrieve") == 8
    clock[0] += 2.0
    assert coalescer.batch_cap("http://r/retrieve") == 16


def test_a_learned_cap_only_ever_shrinks_while_it_holds():
    coalescer = search_tool._COALESCER
    assert coalescer._learn_cap("http://r/retrieve", 400) == 200
    assert coalescer._learn_cap("http://r/retrieve", 200) == 100
    # A later refusal of a larger size cannot undo what a smaller one taught.
    assert coalescer._learn_cap("http://r/retrieve", 400) == 100
