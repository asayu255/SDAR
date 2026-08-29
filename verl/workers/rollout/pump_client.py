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
"""The driver's end of the pumped rollout path.

``generate_sequences`` sends a batch and blocks: the rows of one call are the
only rows the engine has, so the tail of every call decodes a handful of
stragglers on a mostly idle GPU. The validation pipeline already runs several
batches at once to cover that, and ``GenerateMerger`` already coalesces the ones
that happen to be queued together -- measured, that caught 21.5% of calls.

This is the same idea without the "happen to". Every pipeline slot submits its
rows here, one client per driver process, and they are resident in the engine
together whether or not their calls lined up. A slot still waits for its own
rows, so nothing about a slot's trajectories changes; what changes is who else is
decoding alongside them.

One thread does all the talking. A Ray actor runs its methods one at a time, so
a round has to carry submissions and completions together or the collect would
queue behind the submit it is waiting for. That thread is also the only place
rank assignment happens, which is why it is decided at dispatch time from the
in-flight counts the last round returned rather than when a caller submits.

Requests go to whichever rank is least loaded. That is only sound because every
rank holds a whole model (tensor_model_parallel_size=1); the worker refuses the
whole path otherwise.
"""

import os
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PUMP_ROUND_S = float(os.environ.get("ROLLOUT_PUMP_ROUND_S", "0.02"))
_PUMP_REPORT_EVERY = int(os.environ.get("ROLLOUT_PUMP_REPORT_EVERY", "0"))
# 0 means "hand everything over and let the engine schedule it", which is the
# right default: vllm admits what its KV cache fits and queues the rest. A cap
# is here for the case where it does not -- if the engine starts preempting and
# recomputing, holding requests on the driver is cheaper than holding them in
# half-finished KV blocks.
_PUMP_MAX_IN_FLIGHT = int(os.environ.get("ROLLOUT_PUMP_MAX_IN_FLIGHT", "0"))
# A request that a rank neither finishes nor fails would otherwise be waited on
# forever: _pending keeps it, so no round is ever idle, and the driver's
# future.result() has nothing to wake it. Long enough that a genuinely slow
# generation is never killed by it -- this is a stuck-detector, not a deadline.
_PUMP_REQUEST_TIMEOUT_S = float(os.environ.get("ROLLOUT_PUMP_REQUEST_TIMEOUT_S", "900"))


def _now() -> float:
    return time.monotonic()


class PumpUnavailable(RuntimeError):
    """The worker group cannot serve the pumped path; use the blocking one."""


class PumpFailed(RuntimeError):
    """A round raised. Every outstanding request is failed with this."""


class PumpClient:
    """Submit prompts, await responses, one round trip per round for all ranks."""

    def __init__(
        self,
        worker_group: Any,
        round_s: float = _PUMP_ROUND_S,
        max_in_flight: int = _PUMP_MAX_IN_FLIGHT,
        name: str = "pump",
        printer=print,
        request_timeout_s: float = _PUMP_REQUEST_TIMEOUT_S,
    ):
        self._wg = worker_group
        self._world_size = int(worker_group.world_size)
        self._round_s = round_s
        self._max_in_flight = max_in_flight
        self._name = name
        self._print = printer
        self._request_timeout_s = request_timeout_s

        self._lock = threading.Lock()
        self._inbox: List[Tuple[str, List[int], Dict[str, Any]]] = []
        self._pending: Dict[str, Future] = {}
        self._deadlines: Dict[str, float] = {}
        self._in_flight = [0] * self._world_size
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[BaseException] = None
        self._next_id = 0
        # Request ids have to be unique for the LIFE OF THE WORKER, not of this
        # client. A rank's TokenPump and its done-queue outlive one client, and a
        # second client numbering from 0 again would hand out ids the previous
        # one already used -- letting a leftover completion resolve a different
        # request with the wrong tokens, silently.
        self._epoch = uuid.uuid4().hex[:8]

        self.rounds = 0
        self.submitted = 0
        self.finished = 0
        self.peak_in_flight = 0
        self.timed_out = 0
        self.handshake_info: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def handshake(self) -> Dict[str, Any]:
        """Ask every rank whether it can be pumped, and for what the driver needs.

        Raises PumpUnavailable with the rank's own reason rather than returning a
        flag, so the caller either has a usable client or an explanation to log.
        """
        replies = self._wg.rollout_pump_step([{"handshake": True}] * self._world_size)
        for rank, reply in enumerate(replies):
            refused = reply.get("refused")
            if refused:
                raise PumpUnavailable(f"rank {rank} refused the pumped path: {refused}")
            if not reply.get("in_session"):
                raise PumpUnavailable(
                    f"rank {rank} has no open rollout session; the pump steps the engine "
                    "between rounds and would be stepping a sleeping one"
                )
        first = dict(replies[0])
        for key in ("pad_token_id", "response_length", "eos_token_id"):
            # Compared by equality rather than through a set, because a model with
            # several end tokens reports eos_token_id as a list (Qwen3 gives
            # [151645, 151643]) and a list cannot go into a set.
            expected = first.get(key)
            for rank, reply in enumerate(replies[1:], start=1):
                if reply.get(key) != expected:
                    raise PumpUnavailable(
                        f"ranks disagree on {key}: rank 0 says {expected!r}, "
                        f"rank {rank} says {reply.get(key)!r}"
                    )
        self.handshake_info = {
            "pad_token_id": first["pad_token_id"],
            "response_length": first["response_length"],
            "eos_token_id": first["eos_token_id"],
        }
        return self.handshake_info

    def start(self) -> "PumpClient":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"{self._name}-client", daemon=True)
            self._thread.start()
        return self

    def close(self, timeout: float = 30.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self._fail_all(self._failure or PumpFailed("pump client closed"))
        try:
            self._wg.rollout_pump_step([{"stop": True}] * self._world_size)
        except Exception as exc:  # noqa: BLE001 - shutting down
            self._print(f"[{self._name}] stop round failed: {type(exc).__name__}: {exc}", flush=True)

    # -- submission --------------------------------------------------------

    def submit(self, prompt_token_ids: Any, meta_info: Dict[str, Any]) -> Future:
        """Queue one prompt. The future carries its response token ids.

        ``meta_info`` travels rather than sampling parameters: the rank derives
        those from it exactly as the blocking path does, so there is no second
        reading of the config to disagree with the first.
        """
        future: Future = Future()
        carried = {str(k): _plain(v) for k, v in meta_info.items()}
        with self._lock:
            # Checked inside the lock, with the insertion. Outside it they decide
            # nothing: the round thread can die between the check and the insert,
            # and _fail_all would then have already emptied _pending. This future
            # would land in the dictionary nobody reads again, and its waiter --
            # whose whole job is to wait -- would wait forever.
            if self._failure is not None:
                raise PumpFailed(f"pump client is dead: {self._failure}")
            if self._stopping.is_set():
                raise PumpFailed("pump client is closing; no new requests")
            request_id = f"{self._name}-{self._epoch}-{self._next_id}"
            self._next_id += 1
            self._pending[request_id] = future
            self._deadlines[request_id] = _now() + self._request_timeout_s
            # NOT list(): the caller hands over an int32 array precisely so the
            # tokens cross Ray as one buffer, and list() on an array rebuilds
            # every token as an np.int32 scalar -- heavier than the Python ints
            # the array was replacing. Measured on 252 x 1,300 tokens under
            # pickle protocol 5: 3.3 ms and 1.32 MB as arrays against 922.1 ms
            # and 4.92 MB as a list of scalars -- the sizes are exact, the
            # timings move by about a third from run to run on a busy box and
            # the ratio does not. The worker calls _as_plain_ids at the vLLM
            # boundary, which is the only place a list is required.
            self._inbox.append((request_id, _as_wire_ids(prompt_token_ids), carried))
            self.submitted += 1
        self._wake.set()
        return future

    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- the round loop ----------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                with self._lock:
                    outgoing, self._inbox = self._inbox, []
                    idle = not outgoing and not self._pending
                if idle:
                    # Nothing submitted and nothing outstanding: no reason to
                    # spend a round trip asking three ranks about nothing.
                    self._wake.wait(self._round_s)
                    self._wake.clear()
                    continue
                self._round(outgoing)
        except BaseException as exc:  # noqa: BLE001 - this thread owns every waiter
            with self._lock:
                self._failure = exc
            self._fail_all(PumpFailed(f"pump round failed: {type(exc).__name__}: {exc}"))

    def _round(self, outgoing: List[Tuple[str, List[int], Dict[str, Any]]]) -> None:
        if self._max_in_flight:
            room = max(0, self._max_in_flight - sum(self._in_flight))
            if len(outgoing) > room:
                outgoing, held = outgoing[:room], outgoing[room:]
                with self._lock:
                    # Back on the front: submission order is the caller's order,
                    # and a request that keeps losing its place is a slot that
                    # never finishes its turn.
                    self._inbox[:0] = held

        payloads = [{"submit": [], "timeout_s": self._round_s} for _ in range(self._world_size)]
        # Least-in-flight, updated as we place, so a burst spreads instead of
        # landing on whichever rank happened to be emptiest last round.
        placed = list(self._in_flight)
        for submission in outgoing:
            rank = min(range(self._world_size), key=lambda r: placed[r])
            payloads[rank]["submit"].append(submission)
            placed[rank] += 1

        replies = self._wg.rollout_pump_step(payloads)
        self.rounds += 1

        resolved: List[Tuple[Future, Any, Optional[str]]] = []
        with self._lock:
            for rank, reply in enumerate(replies):
                self._in_flight[rank] = int(reply.get("in_flight", 0))
                for request_id, token_ids in reply.get("finished", ()):
                    future = self._pending.pop(request_id, None)
                    self._deadlines.pop(request_id, None)
                    if future is not None:
                        self.finished += 1
                        resolved.append((future, token_ids, None))
                for request_id, message in reply.get("failed", ()):
                    future = self._pending.pop(request_id, None)
                    self._deadlines.pop(request_id, None)
                    if future is not None:
                        resolved.append((future, None, message))
            # Anything a rank has neither finished nor failed for longer than the
            # timeout is stuck, and a stuck request is a trajectory that never
            # ends. Fail it here, where the reason can be named, rather than
            # leaving the driver blocked in future.result() with nothing to read.
            if self._request_timeout_s > 0:
                cutoff = _now()
                for request_id, deadline in list(self._deadlines.items()):
                    if deadline > cutoff:
                        continue
                    self._deadlines.pop(request_id, None)
                    future = self._pending.pop(request_id, None)
                    if future is not None:
                        self.timed_out += 1
                        resolved.append((future, None,
                                         f"request {request_id} was neither finished nor failed by any "
                                         f"rank within {self._request_timeout_s:.0f}s"))
            self.peak_in_flight = max(self.peak_in_flight, len(self._pending))

        for future, token_ids, message in resolved:
            if future.cancelled():
                continue
            if message is None:
                future.set_result(token_ids)
            else:
                future.set_exception(PumpFailed(message))

        if _PUMP_REPORT_EVERY and self.rounds % _PUMP_REPORT_EVERY == 0:
            self._print(self.line(), flush=True)

    def _fail_all(self, exc: BaseException) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
            self._deadlines = {}
            self._inbox = []
        for future in pending.values():
            if not future.done():
                future.set_exception(exc)

    def line(self) -> str:
        return (
            f"[{self._name}] {self.submitted} requests, {self.finished} finished, "
            f"{self.timed_out} timed out, "
            f"{self.rounds} rounds, peak {self.peak_in_flight} in flight, "
            f"per-rank in flight {self._in_flight}"
            + (f", capped at {self._max_in_flight}" if self._max_in_flight else "")
        )


def _as_wire_ids(prompt_token_ids):
    """Token ids in the shape that crosses Ray as a buffer rather than objects."""
    import numpy as np

    if isinstance(prompt_token_ids, np.ndarray):
        return prompt_token_ids.astype(np.int32, copy=False)
    return np.fromiter(prompt_token_ids, dtype=np.int32)


def _plain(value: Any) -> Any:
    """OmegaConf scalars come through as their own types; the worker keys a
    sampling-params cache on what it derives from these, so they have to be
    ordinary hashable values by the time they cross."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value
