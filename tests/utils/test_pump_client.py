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
"""The driver end of the pumped path, against a worker group that is a dict.

The failure this guards is a hang: a trajectory awaiting a future nobody will
ever resolve stalls its slot, the slot stalls the pipeline, and the run sits
there until someone kills it. So every way a round can go wrong -- a rank
raising, a request coming back failed, the client closing underneath a waiter --
has to end with the future *done*, not abandoned.
"""

import concurrent.futures
import threading
import time

import pytest

from verl.workers.rollout.pump_client import PumpClient, PumpFailed, PumpUnavailable

WORLD = 3
# Long enough that a request only leaves the fake when the test says so.
NEVER = 10**9


class FakeWorkerGroup:
    """Three ranks that finish a request `latency_rounds` rounds after it arrives."""

    def __init__(self, world_size=WORLD, latency_rounds=1, handshake=None):
        self.world_size = world_size
        self.latency_rounds = latency_rounds
        self.rounds = 0
        self.payload_log = []
        self.resident = [dict() for _ in range(world_size)]  # rank -> {rid: rounds_left}
        self.raise_on_round = None
        self.stopped = False
        self._handshake = handshake or [
            {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8,
             "eos_token_id": 2, "n": 1}
            for _ in range(world_size)
        ]
        self.lock = threading.Lock()

    def rollout_pump_step(self, payloads):
        assert len(payloads) == self.world_size, "payload must be per-rank"
        if payloads[0].get("handshake"):
            return self._handshake
        if payloads[0].get("stop"):
            self.stopped = True
            return [{"finished": [], "failed": [], "in_flight": 0}] * self.world_size

        with self.lock:
            self.rounds += 1
            self.payload_log.append([list(p["submit"]) for p in payloads])
            if self.raise_on_round == self.rounds:
                raise RuntimeError("rank exploded")

            replies = []
            for rank, payload in enumerate(payloads):
                for request_id, prompt_token_ids, _meta in payload["submit"]:
                    self.resident[rank][request_id] = (self.latency_rounds, list(prompt_token_ids))
                finished, failed, still = [], [], {}
                for request_id, (left, prompt) in self.resident[rank].items():
                    if left <= 1:
                        if request_id.endswith("-boom"):
                            failed.append((request_id, "PumpClosed: no output"))
                        else:
                            # An answer that identifies its prompt and its rank.
                            finished.append((request_id, [rank * 100 + prompt[0]]))
                    else:
                        still[request_id] = (left - 1, prompt)
                self.resident[rank] = still
                replies.append({"finished": finished, "failed": failed, "in_flight": len(still)})
            return replies


def _client(wg, **kw):
    return PumpClient(wg, round_s=0.001, printer=lambda *a, **k: None, **kw)


def test_the_handshake_carries_the_rank_s_configured_n():
    """It decides which calls the driver may offer: a training call leaves the
    sampling params alone, so what it asks for is this number, and the pool
    returns one sequence per request."""
    client = _client(FakeWorkerGroup())
    assert client.handshake()["n"] == 1


def test_a_worker_without_the_n_key_reports_it_as_unknown():
    """0, not 1. Guessing 1 for a rank that did not say is the scoring change the
    whole refusal exists to prevent -- _pump_pins_one_sample declines on it."""
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2}
        for _ in range(WORLD)
    ]
    client = _client(FakeWorkerGroup(handshake=handshake))
    assert client.handshake()["n"] == 0


def test_ranks_that_disagree_about_n_are_refused():
    """A rank that would produce a different number of sequences would make the
    answer depend on which rank a row landed on."""
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8,
         "eos_token_id": 2, "n": 1 if rank == 0 else 4}
        for rank in range(WORLD)
    ]
    client = _client(FakeWorkerGroup(handshake=handshake))
    with pytest.raises(PumpUnavailable, match="disagree on n"):
        client.handshake()


def test_handshake_reports_what_the_driver_needs_to_assemble():
    client = _client(FakeWorkerGroup())
    assert client.handshake() == {"pad_token_id": 0, "response_length": 8, "eos_token_id": 2, "n": 1}


def test_a_rank_that_refuses_stops_the_path_with_its_own_reason():
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2},
        {"refused": "tensor_model_parallel_size > 1", "in_session": True},
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2},
    ]
    client = _client(FakeWorkerGroup(handshake=handshake))
    with pytest.raises(PumpUnavailable, match="rank 1 refused.*tensor_model_parallel_size"):
        client.handshake()


def test_no_open_session_is_refused():
    handshake = [
        {"refused": None, "in_session": False, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2}
    ] * WORLD
    client = _client(FakeWorkerGroup(handshake=handshake))
    with pytest.raises(PumpUnavailable, match="no open rollout session"):
        client.handshake()


def test_ranks_disagreeing_on_the_pad_token_is_refused_not_averaged():
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2},
        {"refused": None, "in_session": True, "pad_token_id": 7, "response_length": 8, "eos_token_id": 2},
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": 2},
    ]
    client = _client(FakeWorkerGroup(handshake=handshake))
    with pytest.raises(PumpUnavailable, match="disagree on pad_token_id"):
        client.handshake()


def test_a_model_with_several_end_tokens_handshakes():
    """Qwen3 reports generation_config.eos_token_id as [151645, 151643].

    get_response_mask takes an int or a list, so a list is a perfectly good answer
    here; it only has to survive the agreement check, which must not hash it.
    """
    eos = [151645, 151643]
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": eos}
    ] * WORLD
    client = _client(FakeWorkerGroup(handshake=handshake))
    assert client.handshake()["eos_token_id"] == eos


def test_ranks_disagreeing_on_a_list_eos_is_still_caught():
    handshake = [
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": [1, 2]},
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": [1, 2]},
        {"refused": None, "in_session": True, "pad_token_id": 0, "response_length": 8, "eos_token_id": [1]},
    ]
    client = _client(FakeWorkerGroup(handshake=handshake))
    with pytest.raises(PumpUnavailable, match="disagree on eos_token_id.*rank 2"):
        client.handshake()


def test_each_future_gets_its_own_answer():
    wg = FakeWorkerGroup()
    client = _client(wg).start()
    try:
        futures = [client.submit([i], {"temperature": 0}) for i in range(1, 7)]
        answers = [f.result(timeout=10) for f in futures]
    finally:
        client.close()
    # The fake answers with rank*100 + the prompt's first token, so the prompt
    # each answer came back for is readable off the value.
    assert sorted(a[0] % 100 for a in answers) == [1, 2, 3, 4, 5, 6]
    assert client.finished == 6


def _until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.005)
    return predicate()


def test_a_burst_is_spread_over_the_ranks():
    # Rounds are ~1ms here, so "slow" has to mean slow in rounds, not in seconds:
    # a latency of 50 rounds retires inside a tenth of a second.
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    client = _client(wg).start()
    try:
        for i in range(9):
            client.submit([i], {})
        assert _until(lambda: sum(len(r) for r in wg.resident) == 9)
        assert sorted(len(r) for r in wg.resident) == [3, 3, 3]
    finally:
        client.close()


def test_placement_follows_the_in_flight_counts_the_ranks_reported():
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    client = _client(wg).start()
    try:
        # One request per rank, reported back, then retire rank 1's by hand so
        # that rank is the emptiest -- and see where the next request goes.
        for i in range(WORLD):
            client.submit([i], {})
        assert _until(lambda: sum(len(r) for r in wg.resident) == WORLD)
        with wg.lock:
            wg.resident[1].clear()
        assert _until(lambda: client._in_flight[1] == 0)
        client.submit([99], {})
        assert _until(lambda: len(wg.resident[1]) == 1), (
            f"the next request went somewhere other than the emptiest rank: "
            f"{[len(r) for r in wg.resident]}"
        )
    finally:
        client.close()


def test_a_failed_request_raises_on_its_own_future_and_leaves_the_rest_alone():
    wg = FakeWorkerGroup()
    client = _client(wg).start()
    try:
        good = client.submit([1], {})
        bad = client.submit([2], {})
        # Rename the second request so the fake fails exactly that one.
        with client._lock:
            rid = next(r for r in client._pending if client._pending[r] is bad)
            client._pending[rid + "-boom"] = client._pending.pop(rid)
            client._inbox = [(r + "-boom" if r == rid else r, p, o) for r, p, o in client._inbox]
        with pytest.raises(PumpFailed, match="no output"):
            bad.result(timeout=10)
        assert good.result(timeout=10)
    finally:
        client.close()


def test_a_round_that_raises_fails_every_waiter_rather_than_hanging_it():
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    wg.raise_on_round = 1
    client = _client(wg).start()
    try:
        futures = [client.submit([i], {}) for i in range(4)]
        for future in futures:
            with pytest.raises(PumpFailed, match="rank exploded"):
                future.result(timeout=10)
        with pytest.raises(PumpFailed, match="dead"):
            client.submit([9], {})
    finally:
        client.close()


def test_closing_fails_a_waiter_instead_of_abandoning_it():
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    client = _client(wg).start()
    future = client.submit([1], {})
    client.close()
    with pytest.raises(PumpFailed):
        future.result(timeout=10)
    assert wg.stopped, "the ranks were never told to drop what they were holding"


def test_a_cap_holds_requests_on_the_driver_rather_than_in_the_engine():
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    client = PumpClient(wg, round_s=0.001, max_in_flight=4, printer=lambda *a, **k: None).start()
    try:
        for i in range(10):
            client.submit([i], {})
        assert _until(lambda: sum(len(r) for r in wg.resident) == 4)
        time.sleep(0.1)  # many rounds; nothing retires, so nothing more may go
        assert sum(len(r) for r in wg.resident) == 4
        # Room appears -> the held requests go, in the order they were submitted.
        with wg.lock:
            wg.resident[0].clear()
            wg.resident[1].clear()
            wg.resident[2].clear()
        assert _until(lambda: sum(len(r) for r in wg.resident) == 4)
        # Order is per round, not per rank: within a round the requests are
        # spread over the ranks, but a later round must never carry a request
        # that was submitted before one an earlier round carried.
        per_round = [
            [int(rid.rsplit("-", 1)[1]) for rank in round_ for (rid, _p, _m) in rank]
            for round_ in wg.payload_log
        ]
        sent_rounds = [ids for ids in per_round if ids]
        for earlier, later in zip(sent_rounds, sent_rounds[1:]):
            assert max(earlier) < min(later), f"a held request lost its place: {earlier} then {later}"
        assert sorted(i for ids in sent_rounds for i in ids) == list(range(8))
    finally:
        client.close()


def test_an_idle_client_does_not_spend_round_trips():
    wg = FakeWorkerGroup()
    client = _client(wg).start()
    try:
        time.sleep(0.2)  # many round_s intervals
        assert wg.rounds == 0
    finally:
        client.close()


def test_meta_info_is_carried_through_to_the_rank_that_serves_the_request():
    wg = FakeWorkerGroup()
    client = _client(wg).start()
    try:
        client.submit([1], {"validate": True, "temperature": 0.7}).result(timeout=10)
    finally:
        client.close()
    submitted = [s for round_ in wg.payload_log for rank in round_ for s in rank]
    assert submitted and submitted[0][2] == {"validate": True, "temperature": 0.7}


# --------------------------------------------------------------------------- #
# Not hanging
# --------------------------------------------------------------------------- #
class BlackHoleWG(FakeWorkerGroup):
    """Ranks that accept a request and then never mention it again.

    vLLM aborting a request, or a reply lost on the way back, looks exactly like
    this from the driver -- and _pending staying non-empty means the round loop
    never goes idle, so the client spins happily while the caller blocks.
    """

    def rollout_pump_step(self, payloads):
        if payloads[0].get("handshake") or payloads[0].get("stop"):
            return super().rollout_pump_step(payloads)
        with self.lock:
            self.rounds += 1
            for rank, payload in enumerate(payloads):
                for request_id, _p, _m in payload["submit"]:
                    self.resident[rank][request_id] = None
            return [{"finished": [], "failed": [], "in_flight": len(self.resident[r])}
                    for r in range(self.world_size)]


def test_a_request_no_rank_ever_answers_fails_instead_of_hanging():
    client = _client(BlackHoleWG(), request_timeout_s=0.2)
    client.handshake()
    client.start()
    try:
        future = client.submit([1, 2, 3], {})
        with pytest.raises(PumpFailed, match="neither finished nor failed"):
            future.result(timeout=5)
        assert client.timed_out == 1
    finally:
        client.close(timeout=2)


def test_the_watchdog_can_be_turned_off():
    client = _client(BlackHoleWG(), request_timeout_s=0)
    client.handshake()
    client.start()
    try:
        future = client.submit([1, 2, 3], {})
        with pytest.raises(concurrent.futures.TimeoutError):
            future.result(timeout=0.4)
    finally:
        client.close(timeout=2)


def test_a_slow_request_is_not_killed_by_the_watchdog():
    """It is a stuck-detector, not a deadline; a long generation must survive it."""
    client = _client(FakeWorkerGroup(latency_rounds=40), request_timeout_s=30)
    client.handshake()
    client.start()
    try:
        assert client.submit([7], {}).result(timeout=10) is not None
        assert client.timed_out == 0
    finally:
        client.close(timeout=2)


# --------------------------------------------------------------------------- #
# Request ids
# --------------------------------------------------------------------------- #
def test_two_clients_never_hand_out_the_same_request_id():
    """A rank's pump and its done-queue outlive one client.

    Numbering from 0 again would reissue ids the previous client already used,
    and a leftover completion would then resolve a different request with the
    wrong tokens -- with nothing raised anywhere.
    """
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    first, second = _client(wg), _client(wg)
    first.submit([1], {})
    second.submit([1], {})
    ids = [rid for c in (first, second) for rid in c._pending]
    assert len(set(ids)) == 2, f"two clients issued the same id: {ids}"


def test_one_client_still_numbers_its_own_requests_in_order():
    wg = FakeWorkerGroup(latency_rounds=NEVER)
    client = _client(wg)
    for _ in range(3):
        client.submit([1], {})
    tails = [int(rid.rsplit("-", 1)[1]) for rid in client._pending]
    assert sorted(tails) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# what crosses Ray per request
# --------------------------------------------------------------------------- #
def test_the_payload_that_actually_crosses_ray_is_an_array():
    """Tested on the queued payload, not on the helper that builds it.

    _as_id_list was made to return an int32 array, and PumpClient.submit then
    called list() on it one function later -- rebuilding every token as an
    np.int32 scalar, which is HEAVIER than the Python ints the array replaced.
    The helper's own test passed throughout. Only the payload shows it.

    MEASURED here, 252 requests of 1,300 tokens under pickle protocol 5:

        list[ndarray[int32]]    3.3 ms pickle,   2.0 ms unpickle,  1.32 MB
        list[list[np.int32]]  922.1 ms pickle, 116.0 ms unpickle,  4.92 MB

    Roughly a second per turn, on the driver thread, inside the window the
    residency census tags `gen` and the cards read empty.
    """
    import numpy as np

    client = _client(FakeWorkerGroup())
    client.submit(np.asarray([7, 8, 9], dtype=np.int64), {"validate": True})

    (_request_id, ids, _carried), = client._inbox
    assert isinstance(ids, np.ndarray), f"a {type(ids).__name__} here undoes the transport fix"
    assert ids.dtype == np.int32
    assert ids.tolist() == [7, 8, 9]


def test_a_plain_list_submitted_also_reaches_the_wire_as_an_array():
    """One shape on the wire, whatever the caller had."""
    import numpy as np

    client = _client(FakeWorkerGroup())
    client.submit([7, 8, 9], {"validate": True})

    (_request_id, ids, _carried), = client._inbox
    assert isinstance(ids, np.ndarray) and ids.tolist() == [7, 8, 9]


def test_the_driver_helper_hands_over_an_array_too():
    """The other end of the same path -- kept, but it is not the one that broke."""
    import numpy as np

    from agent_system.multi_turn_rollout.rollout_loop import _as_id_list

    out = _as_id_list(np.asarray([5, 6, 7], dtype=np.int64))
    assert isinstance(out, np.ndarray) and out.dtype == np.int32
    assert _as_id_list([5, 6, 7]).tolist() == [5, 6, 7]


def test_the_worker_converts_back_at_the_vllm_boundary():
    """vLLM wants a list; the wire does not. The conversion belongs on the worker."""
    import numpy as np

    from verl.workers.rollout.token_pump import _as_plain_ids

    assert _as_plain_ids(np.asarray([1, 2], dtype=np.int32)) == [1, 2]
    assert _as_plain_ids([1, 2]) == [1, 2]
    assert all(type(v) is int for v in _as_plain_ids(np.asarray([1, 2], dtype=np.int32)))
