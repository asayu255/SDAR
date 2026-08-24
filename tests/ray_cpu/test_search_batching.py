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
