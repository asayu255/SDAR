import json
import logging
import os
import requests
import uuid
import time
import threading
from typing import Tuple, Optional, Any, Dict, List, Union
from urllib.parse import urlparse

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool, ToolGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
# The linear backoff is capped so an unbounded retry does not drift into
# hour-long sleeps: once the retriever is merely saturated rather than down,
# waiting longer between attempts only adds latency.
MAX_RETRY_DELAY = 30
# How often a still-retrying request says so. Without this an unbounded wait is
# indistinguishable from a hang: the rollout's turn barrier blocks on the slowest
# env, so one stuck query stalls every other trajectory in the step with nothing
# in the log to say why.
RETRY_PROGRESS_EVERY = 60.0

# Coalescing. The index this talks to is Flat, so one search reads the whole 32 GB
# of embeddings no matter how many queries it is given: 126 separate requests read
# it 126 times. Measured against the retriever in use, one unloaded query is 80 ms
# and 126 concurrent ones take 7.5 s -- a 93x inflation that is entirely the index
# being re-read, and that no number of server processes or replicas can remove
# because they share the same GPUs and the same bandwidth.
#
# The rollout hands every environment its action at once (a ThreadPoolExecutor over
# all of them), so the calls arrive within a few milliseconds of each other. Holding
# the first one for a short window and sending whatever accumulated as a single
# request turns those 126 reads into one, without the environments knowing.
_BATCH_ENABLED = os.environ.get("SEARCH_BATCH_REQUESTS", "1").strip().lower() not in ("0", "false", "no", "")
# Long enough to catch a fan-out, short enough to be noise against an 80 ms query.
_BATCH_WINDOW_S = float(os.environ.get("SEARCH_BATCH_WINDOW_MS", "10")) / 1000.0
# How long a URL stays un-batched after it rejects a list. A retriever restarted
# with a server that does take lists is the usual reason the answer changes, and
# nothing else would ever tell us: the flag is set once and an evaluation runs
# for hours. Re-probing costs one rejected request per period -- the queries are
# sent singly either way. 0 makes the disable permanent.
_BATCH_RETRY_S = float(os.environ.get("SEARCH_BATCH_RETRY_S", "300"))
# How many consecutive 5xx a *multi-query* request absorbs before it is handed
# back for splitting. The retriever encodes and searches a request in chunks of
# its own retrieval_batch_size, so a request larger than what its GPU can hold
# answers 5xx to every attempt, forever: measured on the retriever in use, 383
# queries were served and 384 were not, with the server configured to chunk at
# 512. That ceiling is free memory divided by the cost of a query, so it moves
# with whatever else is on the box -- which is why it cannot be a fixed cap in
# either the client or the server, and why the answer is to find it by halving.
#
# The budget must clear a genuine outage, because those also answer 5xx while a
# restarting server is up but not ready. Observed recoveries took 2-6 attempts;
# 4 attempts is 6 s of backoff before the first split, and splitting a request
# the server is merely too busy to serve is harmless -- the halves are retried
# under the same policy. 0 restores the unbounded wait for batches too.
_SPLIT_AFTER_5XX = max(0, int(os.environ.get("SEARCH_SPLIT_AFTER_5XX", "4")))
# How long a learned size cap holds before it is doubled. The ceiling rises as
# the box frees memory, and nothing would ever tell us: a validation pass that
# learned a cap while the trainer held its KV cache would keep it for the rest
# of the run. Re-probing costs one split request per period; 0 makes a cap
# permanent, the way 0 makes the un-batched flag permanent above.
_BATCH_GROW_S = float(os.environ.get("SEARCH_BATCH_GROW_S", "300"))


class _RetryableServerError(Exception):
    """A multi-query request the retriever answered 5xx to, repeatedly.

    Raised instead of returned so it cannot be mistaken for the errors that end
    up in an ``<information>`` block: nothing here reaches a trajectory. Only a
    caller that passed ``server_error_budget`` can see it, and the only caller
    that does is the coalescer, which can act on it by sending less at once.
    """

    def __init__(self, message: str, status_code: int, attempts: int):
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


def _search_api_request(
    retrieval_service_url: str,
    query: Union[str, List[str]],
    topk: int = 3,
    return_scores: bool = True,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
    max_retries: Optional[int] = MAX_RETRIES,
    server_error_budget: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a single query.

    Retries are split by whether waiting can help. A saturated or restarting
    retriever answers with a timeout, a refused connection or a 5xx, and the
    same request a minute later succeeds -- those are retried. A 4xx, malformed
    JSON or a programming error will answer the same way forever, so those give
    up immediately rather than hanging the run.

    That distinction is what ``max_retries=None`` rests on. Giving up on a
    retryable failure is not free: the caller turns ``error_msg`` into the
    ``<information>`` block the student sees, so an exhausted retry budget puts
    the string "Search error: ..." into the trajectory *as if it were the
    retrieved document*, and the run trains on it without a trace in the
    metrics. When one retriever is shared between runs that is not a remote
    possibility -- it is what a load spike looks like.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: Per-request timeout in seconds. ``None`` waits indefinitely --
            see the note in ``SearchToolGroup.__init__`` on why a generous
            finite timeout with unbounded retries is the better way to spell
            "never give up".
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.
        max_retries: Attempts before giving up on a retryable failure. ``None``
            (or <= 0) retries until the retriever answers.
        server_error_budget: Consecutive 5xx after which to raise
            ``_RetryableServerError`` rather than keep waiting. Set only for a
            request carrying several queries, where the caller can respond by
            sending fewer; a single query has nothing to give up and keeps the
            unbounded wait, which is the whole point of ``max_retries=None``.

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    unlimited = max_retries is None or max_retries <= 0
    budget = "inf" if unlimited else str(max_retries)
    started = time.monotonic()
    last_progress = started

    def _backoff(attempt: int) -> float:
        return min(INITIAL_RETRY_DELAY * attempt, MAX_RETRY_DELAY)

    def _wait(attempt: int, reason: str) -> None:
        """Sleep before the next attempt, reporting an unbounded wait periodically."""
        nonlocal last_progress
        delay = _backoff(attempt)
        elapsed = time.monotonic() - started
        if unlimited and elapsed - (last_progress - started) >= RETRY_PROGRESS_EVERY:
            last_progress = time.monotonic()
            logger.warning(
                f"{log_prefix}still waiting on {retrieval_service_url} after {elapsed:.0f}s "
                f"and {attempt} attempt(s); last failure: {reason}. The rollout is "
                f"blocked on this query by design (max_retries=None)."
            )
        else:
            logger.info(f"{log_prefix}Retrying after {delay} seconds...")
        time.sleep(delay)

    last_error = None
    attempt = 0
    server_errors = 0
    while True:
        attempt += 1
        can_retry = unlimited or attempt < max_retries
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt}/{budget}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                server_errors += 1
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt}/{budget}"
                logger.warning(last_error)
                # A request the retriever cannot serve at this size answers the
                # same way however long we wait. Hand it back while the caller
                # still has something to try -- fewer queries in one request.
                if server_error_budget and server_errors >= server_error_budget:
                    if should_close_session:
                        session.close()
                    raise _RetryableServerError(last_error, response.status_code, server_errors)
                if can_retry:
                    _wait(attempt, f"HTTP {response.status_code}")
                    continue
                break

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt}")
            elif attempt > 1:
                # Recovered after retrying. Worth one line even with request
                # logging off: it is the only evidence the retriever wobbled.
                logger.info(
                    f"{log_prefix}search recovered after {attempt} attempts "
                    f"({time.monotonic() - started:.0f}s)"
                )

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except _RetryableServerError:
            # Not an outcome of the request loop: the caller asked to be told.
            raise
        except requests.exceptions.ConnectionError as e:
            # A refused connection says nothing about how large the request was,
            # so it must not count towards the budget that decides to split one.
            server_errors = 0
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if can_retry:
                _wait(attempt, "connection error")
                continue
            break
        except requests.exceptions.Timeout as e:
            server_errors = 0
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if can_retry:
                _wait(attempt, "timeout")
                continue
            break
        except requests.exceptions.RequestException as e:
            # 4xx and friends: the same request will fail the same way, so
            # waiting cannot turn this into a document.
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed. The caller substitutes this message
    # for the retrieved document, so say plainly that the trajectory is now
    # carrying an error string.
    logger.error(
        f"{log_prefix}API Request Failed after {attempt} attempt(s) / "
        f"{time.monotonic() - started:.0f}s: {last_error}. This turn's "
        f"<information> block will contain the error text, not a document."
    )

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error



class _QuerySlot:
    """One caller's place in a coalesced request."""

    __slots__ = ("query", "done", "response", "error")

    def __init__(self, query: str):
        self.query = query
        self.done = threading.Event()
        self.response = None
        self.error = None


class _Coalescer:
    """Turn concurrent single-query calls into one request per short window.

    The first caller of a window becomes its leader: it waits out the window, takes
    everything that accumulated, sends it as one request and hands each caller its
    own slice. Every other caller just waits for its slot. A caller that arrives
    after the leader has taken the batch starts the next window.

    Each caller still receives exactly the ``{"result": [documents]}`` shape a
    single-query call returns, so nothing downstream can tell the difference --
    same queries, same order, same server, same documents.

    Batching is per (url, topk, return_scores, timeout, retries): a window only
    ever holds requests that would have been identical apart from the query.

    A window has no size of its own -- it is however many environments happened
    to act at once -- and the retriever has a ceiling on how many queries it can
    serve in one request that moves with the memory left on its GPU. Where the
    two meet, the request answers 5xx to every attempt and unbounded retries
    wait on it forever. So a batch that keeps drawing 5xx is halved until it is
    served, and the size that worked is remembered for the next window.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._windows: Dict[Any, list] = {}
        # Set false for a URL whose server rejects a list. Learned by trying,
        # rather than by parsing an error string: a batch that fails and then
        # succeeds one query at a time is a server that does not take lists.
        self._batchable: Dict[str, bool] = {}
        self._retry_at: Dict[str, float] = {}
        # The largest request this URL was seen to serve, learned the same way:
        # a size that answered 5xx to every attempt is a size not to send again.
        # Unset means unlimited, which is where every URL starts -- capping by
        # default would cost an index re-read per extra request on a retriever
        # that was never going to refuse.
        self._max_batch: Dict[str, int] = {}
        self._grow_at: Dict[str, float] = {}

    def enabled_for(self, url: str) -> bool:
        if not _BATCH_ENABLED:
            return False
        with self._lock:
            if self._batchable.get(url, True):
                return True
            if _BATCH_RETRY_S <= 0 or time.monotonic() < self._retry_at.get(url, 0.0):
                return False
            self._batchable[url] = True
        logger.warning(f"{url}: re-probing batched search after {_BATCH_RETRY_S:.0f}s un-batched")
        return True

    def batch_cap(self, url: str) -> Optional[int]:
        """The most queries to put in one request to this URL, or None.

        Doubled once per grow period so a cap learned under a transient squeeze
        does not outlive it. It rises past any window this rollout can produce
        and stops mattering, which is the same thing as being forgotten.
        """
        grown = None
        with self._lock:
            cap = self._max_batch.get(url)
            if cap is not None and _BATCH_GROW_S > 0 and time.monotonic() >= self._grow_at.get(url, 0.0):
                cap = grown = cap * 2
                self._max_batch[url] = cap
                self._grow_at[url] = time.monotonic() + _BATCH_GROW_S
        if grown is not None:
            logger.info(f"{url}: retrying batches of up to {grown} queries after {_BATCH_GROW_S:.0f}s capped")
        return cap

    def _learn_cap(self, url: str, refused: int) -> int:
        """Record that ``refused`` queries in one request was too many."""
        cap = max(1, refused // 2)
        with self._lock:
            cap = min(cap, self._max_batch.get(url, cap))
            self._max_batch[url] = cap
            self._grow_at[url] = time.monotonic() + _BATCH_GROW_S
        return cap

    def call(self, url: str, query: str, send, **key_parts):
        # Checked here as well as in call_search_api: once a URL has told us it
        # does not take lists, no caller should be able to make it say so again.
        if not self.enabled_for(url):
            return send(query)
        slot = _QuerySlot(query)
        key = (url,) + tuple(sorted(key_parts.items()))
        with self._lock:
            window = self._windows.setdefault(key, [])
            window.append(slot)
            leader = len(window) == 1
        if not leader:
            # No timeout: the leader fills every slot in a finally, including
            # when its request raises. A bound here would race a slow retriever
            # and hand the environment an error it did not have.
            slot.done.wait()
            return slot.response, slot.error

        time.sleep(_BATCH_WINDOW_S)
        with self._lock:
            batch = self._windows.pop(key, [])
        self._flush(url, batch, send)
        return slot.response, slot.error

    def _flush(self, url, batch, send):
        try:
            self._send(url, batch, send)
        except BaseException as exc:  # noqa: BLE001 - a leader that dies must not hang its followers
            for item in batch:
                if item.error is None and item.response is None:
                    item.error = f"coalesced search failed: {exc}"
            raise
        finally:
            for item in batch:
                item.done.set()

    def _send(self, url, batch, send):
        """Send one slice, halving it if the retriever refuses it for its size.

        The halves go back through the same path, so the search for a size the
        retriever will serve costs log2(n) requests once and nothing after: the
        cap it lands on is remembered, and the next window is sliced to it.

        A slice of one is sent as a bare string, which sets no budget: there the
        unbounded wait is the only thing left, and it is the behaviour
        ``max_retries=None`` was asked for. Splitting never turns a retriever
        outage into an error string in a trajectory -- it only stops a request
        that is too large from waiting on a server that will never serve it.
        """
        cap = self.batch_cap(url)
        if cap and len(batch) > cap:
            for start in range(0, len(batch), cap):
                self._send(url, batch[start : start + cap], send)
            return

        queries = [item.query for item in batch]
        try:
            # One query goes as a bare string: identical to the un-batched call,
            # so a lone environment never probes a server for list support it may
            # not have.
            response, error = send(queries[0] if len(queries) == 1 else queries)
        except _RetryableServerError as exc:
            if len(batch) == 1:
                raise  # a budget is never set for a bare string; nothing to split
            cap = self._learn_cap(url, len(batch))
            logger.warning(
                f"{url}: {len(batch)} queries in one request drew HTTP {exc.status_code} on "
                f"{exc.attempts} consecutive attempts; batches capped at {cap} for "
                f"{_BATCH_GROW_S:.0f}s and this one re-sent in slices. The retriever is up -- "
                "this request was too large for it to serve, which waiting cannot change."
            )
            # Re-send the same queries, not two halves: the cap just learned is
            # what slices them, so the rest of this window does not have to
            # rediscover a size that has already been refused.
            self._send(url, batch, send)
            return
        if len(queries) > 1 and error:
            # A returned error is not the size refusal handled above -- that one
            # is raised. So this is a server that does not accept a list, or one
            # that failed in a way retrying cannot fix. Re-send one at a time: if
            # that works, it was the former and this URL stops batching; if it
            # fails too, the caller gets the error it would have got anyway.
            logger.warning(f"batched search of {len(queries)} queries failed ({error}); retrying singly")
            singles = [send(item.query) for item in batch]
            if all(single_error is None for _, single_error in singles):
                with self._lock:
                    self._batchable[url] = False
                    self._retry_at[url] = time.monotonic() + _BATCH_RETRY_S
                logger.warning(
                    f"{url} rejected a batched request but served the queries individually; "
                    "batching disabled for it. Restart the retriever with a server that "
                    "accepts a list of queries -- this URL is re-probed every "
                    f"{_BATCH_RETRY_S:.0f}s, so a restart is picked up without restarting the run."
                )
            for item, (single_response, single_error) in zip(batch, singles):
                item.response, item.error = single_response, single_error
            return
        self._distribute(batch, response, error)

    @staticmethod
    def _distribute(batch, response, error):
        if error is not None or not response:
            for item in batch:
                item.error = error or "search returned no response"
            return
        results = response.get("result")
        if not isinstance(results, list) or len(results) != len(batch):
            got = len(results) if isinstance(results, list) else "no"
            for item in batch:
                item.error = f"retriever returned {got} results for {len(batch)} queries"
            return
        for item, documents in zip(batch, results):
            # The single-query shape, so callers cannot tell they were batched.
            item.response = {"result": [documents]}


_COALESCER = _Coalescer()


def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
    max_retries: Optional[int] = MAX_RETRIES,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieve for one query, coalescing it with whatever else is in flight.

    Same arguments, same return, same documents as issuing the request alone --
    see _Coalescer for why the requests are worth merging and what keeps the
    result indistinguishable. SEARCH_BATCH_REQUESTS=0 sends them one at a time.
    """

    def send(payload_query):
        return _search_api_request(
            retrieval_service_url=retrieval_service_url,
            query=payload_query,
            topk=topk,
            return_scores=return_scores,
            timeout=timeout,
            log_requests=log_requests,
            session=session,
            max_retries=max_retries,
            # A list can be made smaller; a bare string cannot. So only a batch
            # is allowed to stop waiting on a 5xx, and it stops in order to send
            # less -- never to give up on the query.
            server_error_budget=_SPLIT_AFTER_5XX if isinstance(payload_query, list) else None,
        )

    if not _COALESCER.enabled_for(retrieval_service_url):
        return send(query)
    return _COALESCER.call(
        retrieval_service_url,
        query,
        send,
        topk=topk,
        return_scores=return_scores,
        timeout=timeout,
        max_retries=max_retries,
    )


def _enable_tcp_keepalive(adapter, idle_s: int = 30, interval_s: int = 10, probes: int = 3) -> None:
    """Ask urllib3's pools to set SO_KEEPALIVE (and the Linux tuning) on new sockets.

    Best-effort: the socket options are platform-specific and urllib3's kwargs
    are not part of its public API, so a version that does not accept them
    leaves the default behaviour rather than failing the run. The retry loop is
    still correct without keepalive -- it just takes the OS default to notice a
    dead peer.
    """
    import socket

    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", idle_s), ("TCP_KEEPINTVL", interval_s), ("TCP_KEEPCNT", probes)):
        opt = getattr(socket, name, None)
        if opt is not None:
            options.append((socket.IPPROTO_TCP, opt, value))
    try:
        adapter.poolmanager.connection_pool_kw["socket_options"] = options
    except Exception as e:  # pragma: no cover - urllib3 internals moved
        logger.warning(f"could not enable TCP keepalive on the search session: {e}")


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        format_reference += f"Doc {idx+1}: {content}\n"
    return format_reference


class SearchToolGroup(ToolGroup):
    # Class-level session pool shared across all instances
    _session_pool = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for the given base URL"""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=512,  # Number of connection pools
                    pool_maxsize=512,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                # TCP keepalive. A request waiting on a retriever that has gone
                # away sees nothing at the socket layer -- no FIN, no RST -- and
                # with a long or absent read timeout it waits for the OS default,
                # which is hours. Keepalive probes turn that into a connection
                # error within ~a minute, which the retry loop then handles.
                _enable_tcp_keepalive(adapter)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url="http://127.0.0.1:8000/retrieve",
        topk=3,
        timeout=DEFAULT_TIMEOUT,
        log_requests=True,
        max_retries=MAX_RETRIES,
    ):
        """
        ``max_retries=None`` never gives up on a retryable failure, which is what
        one retriever shared between concurrent runs needs: the alternative is
        that a load spike exhausts the budget and the error string is trained on
        as though it were the retrieved document (see ``call_search_api``).

        Prefer ``timeout=<generous finite>`` with ``max_retries=None`` over
        ``timeout=None``. Both mean "wait for the retriever", but they differ
        when a connection dies without being closed -- a server restart, a
        dropped route. With no timeout the socket blocks forever with nothing to
        detect it, and because the rollout's turn barrier waits on the slowest
        env, one dead socket stalls every trajectory in the step. A finite
        timeout turns the same event into one more retry, which then succeeds
        against the restarted server. ``timeout=None`` is still accepted for the
        case where the retriever is known to be merely slow and never absent.
        """
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.max_retries = max_retries
        self.log_requests = log_requests

        # Extract base URL for session sharing
        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        self.session = self._get_shared_session(self.base_url)
        if self.log_requests:
            logger.info(f"SearchToolGroup initialized using shared session pool for {self.base_url}")

        super().__init__(name="SearchToolGroup")

    @tool
    def search(self, query: str) -> str:
        # NOTE(shu): add warning messages here?
        if query is None:
            return ""

        query = query.strip()

        try:
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,  # Pass our shared session for connection reuse
                max_retries=self.max_retries,
            )
        except Exception as e:
            error_msg = f"API Request Exception during batch search: {e}"
            logger.error(f"Batch search: {error_msg}")

        metadata = {
            "query": query,
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "total_results": 0,
            "formatted_result": None,
        }

        result_text = json.dumps({"result": "Search request failed or timed out after retries."})

        if error_msg:
            metadata["status"] = "api_error"
            result_text = json.dumps({"result": f"Search error: {error_msg}"})
            logger.error(f"Batch search: API error occurred: {error_msg}")
        elif api_response:
            logger.debug(f"Batch search: API Response: {api_response}")
            metadata["api_response"] = api_response

            try:
                raw_results = api_response.get("result", [])
                if raw_results:
                    pretty_results = []
                    total_results = 0
                    for retrieval in raw_results:
                        formatted = _passages2string(retrieval)
                        pretty_results.append(formatted)
                        total_results += len(retrieval) if isinstance(retrieval, list) else 1

                    final_result = "\n---\n".join(pretty_results)
                    result_text = json.dumps({"result": final_result})
                    metadata["status"] = "success"
                    metadata["total_results"] = total_results
                    metadata["formatted_result"] = final_result
                    if self.log_requests:
                        logger.info(f"Batch search: Successful, got {total_results} total results")
                else:
                    result_text = json.dumps({"result": "No search results found."})
                    metadata["status"] = "no_results"
                    metadata["total_results"] = 0
                    if self.log_requests:
                        logger.info("Batch search: No results found")
            except Exception as e:
                error_msg = f"Error processing search results: {e}"
                result_text = json.dumps({"result": error_msg})
                metadata["status"] = "processing_error"
                logger.error(f"Batch search: {error_msg}")
        else:
            metadata["status"] = "unknown_api_state"
            result_text = json.dumps({"result": "Unknown API state (no response and no error message)."})
            logger.error("Batch search: Unknown API state.")

        return result_text
