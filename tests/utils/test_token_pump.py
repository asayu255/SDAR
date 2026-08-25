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
"""Driving one engine as a pool: submit a request, await that request.

The blocking LLM.generate(prompts) forces a rollout into lockstep -- every
trajectory waits for the slowest to finish generating, then waits again at the
turn boundary. Measured, that is 10.6% of the evaluation's wall with no slot on
the GPU plus a fixed ~0.6 s per call. Driving step() directly lets a trajectory
leave as soon as its own answer is done.

The engine's state is shared, so exactly one thread may call step(); everything
here exists to keep that true while letting many trajectories submit. The tests
that matter are the ones about what happens when it goes wrong: a request left
waiting on a stopped or dead pump hangs a trajectory, and a hung trajectory
hangs the whole rollout.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from verl.workers.rollout.token_pump import PumpClosed, TokenPump  # noqa: E402


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _Output:
    def __init__(self, request_id, token_ids, finished=True):
        self.request_id = request_id
        self.finished = finished
        self.outputs = [_Completion(token_ids)]


class _Engine:
    """Emits one token per step per request; finishes after `length` steps.

    Records the thread every step() came from, because exactly one may.
    """

    def __init__(self, length=3, fail_after=None, step_seconds=0.0):
        self.running = {}
        self.added = []
        self.step_threads = set()
        self.length = length
        self.fail_after = fail_after
        self.step_seconds = step_seconds
        self.steps = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def add_request(self, request_id, prompt, params, **kw):
        with self._lock:
            self.added.append((request_id, tuple(prompt["prompt_token_ids"]), params))
            self.running[request_id] = []
            self.max_concurrent = max(self.max_concurrent, len(self.running))

    def has_unfinished_requests(self):
        with self._lock:
            return bool(self.running)

    def step(self):
        self.step_threads.add(threading.current_thread().name)
        self.steps += 1
        if self.step_seconds:
            time.sleep(self.step_seconds)
        if self.fail_after is not None and self.steps > self.fail_after:
            raise RuntimeError("engine died")
        done = []
        with self._lock:
            for request_id, tokens in list(self.running.items()):
                tokens.append(100 + len(tokens))
                if len(tokens) >= self.length:
                    done.append(_Output(request_id, list(tokens)))
                    del self.running[request_id]
        return done


def test_a_request_comes_back_to_its_own_caller():
    engine = _Engine(length=2)
    with TokenPump(engine, idle_wait_s=0.001) as pump:
        a = pump.submit([1, 2, 3], "greedy")
        b = pump.submit([9], "greedy")
        assert a.result(5) == [100, 101]
        assert b.result(5) == [100, 101]

    assert {req for req, _, _ in engine.added} == {"token-pump-0", "token-pump-1"}
    assert dict((req, ids) for req, ids, _ in engine.added)["token-pump-0"] == (1, 2, 3)


def test_the_prompt_goes_in_as_token_ids():
    """TITO: no re-tokenisation between the loop and the engine."""
    engine = _Engine(length=1)
    with TokenPump(engine) as pump:
        pump.submit([7, 8, 9], "params").result(5)
    request_id, ids, params = engine.added[0]
    assert ids == (7, 8, 9)
    assert params == "params"


def test_requests_submitted_late_join_the_ones_already_running():
    """The whole point: a trajectory back from its environment rejoins a pool
    that never stopped, instead of waiting for a turn boundary."""
    engine = _Engine(length=20, step_seconds=0.01)
    with TokenPump(engine) as pump:
        first = pump.submit([1], "p")
        time.sleep(0.05)  # first is mid-flight
        second = pump.submit([2], "p")
        first.result(5)
        second.result(5)
    assert engine.max_concurrent >= 2


def test_only_one_thread_ever_steps_the_engine():
    """step() advances shared engine state; two callers would corrupt it."""
    engine = _Engine(length=2)
    with TokenPump(engine) as pump:
        futures = [pump.submit([i], "p") for i in range(16)]
        for f in futures:
            f.result(5)
    assert len(engine.step_threads) == 1


def test_an_idle_pump_does_not_spin_on_step():
    engine = _Engine(length=1)
    with TokenPump(engine, idle_wait_s=0.01) as pump:
        time.sleep(0.1)
        idle_steps = engine.steps
        pump.submit([1], "p").result(5)
    assert idle_steps == 0  # nothing resident, nothing stepped


def test_a_dead_engine_fails_every_waiter():
    """A future nobody resolves hangs a trajectory, and that hangs the rollout."""
    engine = _Engine(length=100, fail_after=2)
    pump = TokenPump(engine).start()
    futures = [pump.submit([i], "p") for i in range(4)]
    for f in futures:
        with pytest.raises(RuntimeError, match="engine died"):
            f.result(5)
    pump.stop()


def test_submitting_to_a_dead_pump_raises_rather_than_hangs():
    engine = _Engine(length=100, fail_after=1)
    pump = TokenPump(engine).start()
    pump.submit([1], "p")
    time.sleep(0.1)
    with pytest.raises(PumpClosed):
        pump.submit([2], "p")
    pump.stop()


def test_stopping_fails_what_is_still_outstanding():
    engine = _Engine(length=10_000, step_seconds=0.002)  # cannot finish in the window
    pump = TokenPump(engine).start()
    pending = pump.submit([1], "p")
    time.sleep(0.05)
    pump.stop(timeout=5)
    with pytest.raises(PumpClosed):
        pending.result(5)


def test_submitting_after_stop_raises():
    pump = TokenPump(_Engine()).start()
    pump.stop()
    with pytest.raises(PumpClosed):
        pump.submit([1], "p")


def test_the_counters_describe_the_run():
    engine = _Engine(length=2)
    with TokenPump(engine) as pump:
        for i in range(5):
            pump.submit([i], "p").result(5)
    assert pump.submitted == 5
    assert pump.finished == 5
    assert pump.steps >= 5
    assert "5 requests, 5 finished" in pump.line()


def test_an_output_with_no_completion_fails_its_future():
    """Rather than resolving with nothing and letting the caller index [0]."""

    class _Empty(_Engine):
        def step(self):
            with self._lock:
                out = []
                for request_id in list(self.running):
                    empty = _Output(request_id, [])
                    empty.outputs = []
                    out.append(empty)
                    del self.running[request_id]
            return out

    with TokenPump(_Empty()) as pump:
        with pytest.raises(PumpClosed, match="no output"):
            pump.submit([1], "p").result(5)


def test_stopping_aborts_what_the_engine_still_holds():
    """Requests nobody will collect would hold KV blocks into the next rollout."""
    engine = _Engine(length=10_000, step_seconds=0.002)
    aborted = []
    engine.abort_request = aborted.append

    pump = TokenPump(engine).start()
    pump.submit([1], "p")
    time.sleep(0.05)
    pump.stop(timeout=5)

    assert aborted and len(aborted[0]) == 1
