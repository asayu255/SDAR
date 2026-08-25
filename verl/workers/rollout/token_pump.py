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
"""Drive one vLLM engine as a pool: submit a request, await that request.

``LLM.generate(prompts)`` adds every prompt, steps until all of them finish, and
returns them together. The batching underneath is already continuous -- what is
synchronous is that call. A rollout that goes through it has to move in lockstep:
every trajectory waits for the slowest one to finish generating before any of
them can step its environment, and every one of them waits again at the turn
boundary. Measured on the evaluation arm, that lockstep is 10.6% of wall with no
slot on the GPU at all, plus a fixed ~0.6 s per call paid once a turn.

This drives the same engine one step at a time instead. A trajectory submits its
own prompt, awaits its own answer, steps its own environment and comes back --
so a trajectory waiting on the retriever is simply not in the pool, and the ones
that are keep the GPU fed.

ONE THREAD OWNS THE ENGINE. ``step()`` advances shared engine state, so it is
called from the pump thread and nowhere else; submissions from other threads go
through a queue that the pump drains between steps. Callers get an ordinary
concurrent.futures.Future, which asyncio can await via wrap_future.

Nothing here changes what is generated: same weights, same sampling parameters,
same prompt token ids. What it changes is which requests are resident together
in a decode step, and that moves floating-point reduction order -- so results
are not bit-identical to the blocking path, and not reproducible run to run
either, because the composition now depends on when each trajectory arrives.
That is a real cost and it is the reason this is gated off by default.
"""

import queue
import threading
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence


class PumpClosed(RuntimeError):
    """Raised for a submission after the pump has stopped, or on a dead pump."""


class TokenPump:
    """A pool over one engine: token ids in, token ids out, per request."""

    def __init__(self, engine: Any, idle_wait_s: float = 0.001, name: str = "token-pump"):
        self._engine = engine
        self._idle_wait_s = idle_wait_s
        self._name = name
        self._submissions: "queue.Queue" = queue.Queue()
        self._pending: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[BaseException] = None
        self._next_id = 0
        self.steps = 0
        self.submitted = 0
        self.finished = 0
        self.peak_in_flight = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "TokenPump":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the pump and fail anything still outstanding.

        A request left waiting on a stopped pump would hang its trajectory, and
        a hung trajectory hangs the whole rollout -- so they are failed, loudly,
        rather than abandoned.
        """
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self._fail_all(self._failure or PumpClosed("token pump stopped"))

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- submission --------------------------------------------------------

    def submit(self, prompt_token_ids: Sequence[int], sampling_params: Any, request_id: Optional[str] = None) -> Future:
        """Queue one generation. The future carries the response token ids."""
        if self._stopping.is_set():
            raise PumpClosed("token pump is stopping; no new requests")
        if self._failure is not None:
            raise PumpClosed(f"token pump died: {self._failure}")
        future: Future = Future()
        with self._lock:
            if request_id is None:
                request_id = f"{self._name}-{self._next_id}"
                self._next_id += 1
            self._pending[request_id] = future
            self.submitted += 1
        self._submissions.put((request_id, list(prompt_token_ids), sampling_params))
        self._wake.set()
        return future

    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- the pump ----------------------------------------------------------

    def _run(self) -> None:
        try:
            # Checked at the TOP, not only when the engine drains: a request that
            # never finishes would otherwise keep the pump stepping forever and
            # stop() could not stop it. An aborted rollout has to be able to put
            # the engine down.
            while not self._stopping.is_set():
                added = self._drain_submissions()
                if not self._engine.has_unfinished_requests():
                    if not added:
                        # Nothing resident and nothing queued: wait to be poked
                        # rather than spinning on step() against an empty engine.
                        self._wake.wait(self._idle_wait_s)
                        self._wake.clear()
                    continue
                self.steps += 1
                for output in self._engine.step():
                    if getattr(output, "finished", False):
                        self._resolve(output)
        except BaseException as exc:  # noqa: BLE001 - the pump owns every waiter
            self._failure = exc
            self._fail_all(exc)
        finally:
            self._abort_outstanding()

    def _abort_outstanding(self) -> None:
        """Tell the engine to drop what nobody will collect.

        Left resident, those requests would keep consuming KV cache blocks for
        the rest of the process -- which on this arm is the next rollout.
        """
        with self._lock:
            request_ids = list(self._pending)
        if not request_ids:
            return
        try:
            self._engine.abort_request(request_ids)
        except Exception:  # noqa: BLE001 - shutting down; the engine may be gone
            pass

    def _drain_submissions(self) -> bool:
        added = False
        while True:
            try:
                request_id, prompt_token_ids, sampling_params = self._submissions.get_nowait()
            except queue.Empty:
                return added
            self._engine.add_request(
                request_id=request_id,
                prompt={"prompt_token_ids": prompt_token_ids},
                params=sampling_params,
            )
            added = True
            with self._lock:
                self.peak_in_flight = max(self.peak_in_flight, len(self._pending))

    def _resolve(self, output: Any) -> None:
        with self._lock:
            future = self._pending.pop(output.request_id, None)
            if future is not None:
                self.finished += 1
        if future is None or future.cancelled():
            return
        completions = list(getattr(output, "outputs", ()))
        if not completions:
            future.set_exception(PumpClosed(f"request {output.request_id} finished with no output"))
            return
        future.set_result(list(completions[0].token_ids))

    def _fail_all(self, exc: BaseException) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(exc)

    def line(self) -> str:
        return (
            f"[token-pump] {self.submitted} requests, {self.finished} finished, "
            f"{self.steps} engine steps, peak {self.peak_in_flight} in flight"
        )
