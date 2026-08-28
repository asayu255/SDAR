import contextlib
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
# 100 ms, and NOT the 10 ms this used to be. "Within a few milliseconds of each
# other" is what the fan-out intends, not what it achieves: 252 threads are
# started at once (search/envs.py: max_workers = min(batch_size, 256)) but reach
# this call under the GIL, one at a time, over roughly 300 ms. A 10 ms window
# therefore opened and closed about thirty times per turn and sent about eight
# queries each, and those thirty requests then fought over the retriever's single
# GPU encoder -- the same 3-query request measured 42 ms and 432 ms depending on
# what else was in flight.
#
# Measured, one turn of a 252-row search batch:
#
#     window 10 ms  -> envstep 28.32 s,  gen share 37-46%
#     window 100 ms -> envstep  0.34 s,  gen share 90-94%
#
# The window is paid once per turn per pipeline slot and only when there is a
# query to hold, so 100 ms buys an 80x reduction for 0.4 s across a batch. Going
# further has little left to win: what remains is 0.34 s.
_BATCH_WINDOW_S = float(os.environ.get("SEARCH_BATCH_WINDOW_MS", "100")) / 1000.0
# How long a URL stays un-batched after it rejects a list. A retriever restarted
# with a server that does take lists is the usual reason the answer changes, and
# nothing else would ever tell us: the flag is set once and an evaluation runs
# for hours. Re-probing costs one rejected request per period -- the queries are
# sent singly either way. 0 makes the disable permanent.
_BATCH_RETRY_S = float(os.environ.get("SEARCH_BATCH_RETRY_S", "300"))
# CONNECT is bounded separately from READ, because they fail for different
# reasons and only one of them is worth waiting out.
#
# `timeout=600` is a scalar, and requests applies a scalar to both phases. So a
# route that has gone away -- [Errno 113] No route to host, which is what a
# WireGuard/Tailscale hiccup looks like from here -- hangs in connect() while
# the kernel retries ARP, and the caller sits there. Measured in one 45-minute
# evaluation: "search recovered after 2 attempts (41s)", "(33s)", "(40s)" --
# one attempt, one backoff, and about forty seconds inside a single connect.
#
# Those forty seconds are not one row's. The coalescing window's followers wait
# on the leader with no bound (deliberately -- see _Coalescer.call), so one
# stalled connect holds every row in the window, and with three pipeline slots
# on one retriever it holds all three. In the system stream it looks like the
# whole node died: GPU 0%, its memory controller 0%, host CPU flat, no disk, no
# extra network. That accounted for most of what was left of the GPU deficit.
#
# A healthy retriever accepts in well under a second on a LAN or a tailnet, so
# 5 s is generous for connect and turns a dropped route into one fast retry
# instead of a forty-second stall. READ stays as configured: a batch of 250
# queries legitimately takes seconds, and giving up on it would put an error
# string into the trajectory where a document belongs.
_CONNECT_TIMEOUT_S = float(os.environ.get("SEARCH_CONNECT_TIMEOUT_S", "5"))

# How long a socket may hold UNACKNOWLEDGED data before the kernel errors it.
# This is the bound the connect timeout above cannot give: a request that got a
# connection and then lost its peer is past connect, and Linux would otherwise
# follow tcp_retries2 (~15 min) or, as measured here, whatever the tunnel outage
# happened to be (~40 s). See _socket_health_options for the evidence.
# 0 leaves the kernel default.
_USER_TIMEOUT_S = float(os.environ.get("SEARCH_TCP_USER_TIMEOUT_S", "10"))


def _split_timeout(timeout):
    """(connect, read) for requests, bounding connect however read is set."""
    if _CONNECT_TIMEOUT_S <= 0:
        return timeout
    if timeout is None:
        return (_CONNECT_TIMEOUT_S, None)
    return (min(_CONNECT_TIMEOUT_S, timeout), timeout)


def _search_api_request(
    retrieval_service_url: str,
    query: Union[str, List[str]],
    topk: int = 3,
    return_scores: bool = True,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
    max_retries: Optional[int] = MAX_RETRIES,
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
                timeout=_split_timeout(timeout),
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt}/{budget}"
                logger.warning(last_error)
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

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if can_retry:
                _wait(attempt, "connection error")
                continue
            break
        except requests.exceptions.Timeout as e:
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



@contextlib.contextmanager
def _profiler_inflight(name, n=1):
    """Hold n on a profiler gauge, if the profiler is importable.

    Guarded because this module runs inside the environment workers, which are
    started in contexts where verl may not be on the path at all; a retrieval
    tool that cannot retrieve because a profiler is missing would be an absurd
    way to lose a run.
    """
    try:
        from verl.utils import gpu_profiler
    except Exception:  # pragma: no cover - profiler is optional
        yield
        return
    with gpu_profiler.inflight(name, n):
        yield


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
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._windows: Dict[Any, list] = {}
        # Set false for a URL whose server rejects a list. Learned by trying,
        # rather than by parsing an error string: a batch that fails and then
        # succeeds one query at a time is a server that does not take lists.
        self._batchable: Dict[str, bool] = {}
        self._retry_at: Dict[str, float] = {}

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
        # ONE query, one count, whether this caller does the HTTP or waits for
        # the caller that does. The stack sampler reported "10.78 threads in
        # search.py <- threading wait" and that was read as ten requests in
        # flight; it is ten waiters behind one. The gauge counts the queries
        # outstanding, which is the number the classifier needs -- and the
        # window between them is why a follower's wait is not idle time the
        # retriever is responsible for.
        with _profiler_inflight("retriever_inflight"):
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
            queries = [item.query for item in batch]
            # One query goes as a bare string: identical to the un-batched call,
            # so a lone environment never probes a server for list support it may
            # not have.
            response, error = send(queries[0] if len(queries) == 1 else queries)
            if len(queries) > 1 and error:
                # Either the server does not accept a list or the retriever is
                # unwell. Re-send one at a time: if that works, it was the former
                # and this URL stops batching; if it fails too, the caller gets
                # the error it would have got anyway.
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
        except BaseException as exc:  # noqa: BLE001 - a leader that dies must not hang its followers
            for item in batch:
                if item.error is None and item.response is None:
                    item.error = f"coalesced search failed: {exc}"
            raise
        finally:
            for item in batch:
                item.done.set()

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


def _urllib3_default_socket_options():
    """What urllib3 would have put on the socket if we passed nothing.

    Read from urllib3 rather than hardcoded so a version that adds an option
    keeps it. The fallback is the value urllib3 has shipped for years, which is
    the one that matters: TCP_NODELAY off would be a regression, not a default.
    """
    try:
        from urllib3.connection import HTTPConnection

        defaults = HTTPConnection.default_socket_options
        if defaults:
            return list(defaults)
    except Exception as e:  # pragma: no cover - urllib3 internals moved
        logger.warning(f"could not read urllib3's default socket options ({e}); assuming TCP_NODELAY")
    import socket

    return [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def _socket_health_options(idle_s: int = 30, interval_s: int = 10, probes: int = 3):
    """The socket options that decide how fast a dead retriever peer is noticed.

    Two different failures need two different knobs, and only one of them was
    set here before.

    KEEPALIVE covers an *idle* socket whose peer went away: no data outstanding,
    nothing to retransmit, so without probes the kernel never learns and the
    next request rides a corpse. 30 s idle + 3 probes 10 s apart.

    TCP_USER_TIMEOUT covers a socket with data *outstanding* -- the request went
    out and no acknowledgement came back. Keepalive does not apply (the socket
    is not idle) and Linux instead follows tcp_retries2, roughly 15 minutes.
    That is the failure this evaluation actually hits.

    MEASURED, run sft-multitask-eval-20260826-201115: 19 samples of 15 s in
    which all three GPUs sat at 0% while host CPU, disk, network and thread
    count were indistinguishable from a busy sample and GPU power fell from
    282 W to 99 W -- the whole node waiting on an answer that was not coming.
    The search log dates them: "recovered after 2 attempts (41s)".

    TWO attempts is what identifies the failure. A route that is down for 40 s
    fails instantly (EHOSTUNREACH is not a wait), so covering 40 s would take
    about nine attempts with this backoff. Two attempts means the FIRST one
    spent the whole 40 s inside one wedged socket and the second, on a fresh
    connection, succeeded at once. Bounding that socket is therefore worth
    roughly three quarters of the 6.3 points of idle it costs.

    Best-effort: these are platform-specific and urllib3's socket_options kwarg
    is not public API, so a platform or version that lacks them keeps the OS
    default rather than failing the run. The retry loop stays correct either
    way -- it just takes longer to get its turn.
    """
    import socket

    # urllib3's socket_options kwarg REPLACES its defaults, it does not extend
    # them, and its default is TCP_NODELAY. Building this list from scratch
    # therefore turns Nagle back on for the one session that sends nothing but
    # small JSON POSTs -- against a server that delays its ACKs, that is tens of
    # milliseconds added to every retrieval, on the hottest path in the run.
    # Start from whatever urllib3 would have used and add to it.
    options = list(_urllib3_default_socket_options())

    def add(level, opt, value):
        if opt is None:
            return
        if any(lvl == level and name == opt for lvl, name, _ in options):
            return  # already carried over from urllib3's defaults
        options.append((level, opt, value))

    add(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", idle_s), ("TCP_KEEPINTVL", interval_s), ("TCP_KEEPCNT", probes)):
        add(socket.IPPROTO_TCP, getattr(socket, name, None), value)
    # Milliseconds, and Linux-only. 0 leaves the kernel default (tcp_retries2).
    if _USER_TIMEOUT_S > 0:
        add(socket.IPPROTO_TCP, getattr(socket, "TCP_USER_TIMEOUT", None), int(_USER_TIMEOUT_S * 1000))
    return options


def _enable_tcp_keepalive(adapter, idle_s: int = 30, interval_s: int = 10, probes: int = 3) -> None:
    """Install :func:`_socket_health_options` on an adapter's future sockets."""
    options = _socket_health_options(idle_s, interval_s, probes)
    try:
        adapter.poolmanager.connection_pool_kw["socket_options"] = options
    except Exception as e:  # pragma: no cover - urllib3 internals moved
        logger.warning(f"could not set socket health options on the search session: {e}")
        return
    # Said out loud because silence used to mean either "installed" or "the
    # logger was never configured", and the difference is 40 s of GPU idle.
    import socket

    bounds = []
    if any(o == getattr(socket, "TCP_USER_TIMEOUT", None) for _l, o, _v in options):
        bounds.append(f"wedged socket {_USER_TIMEOUT_S:.0f}s")
    else:
        bounds.append("wedged socket UNBOUNDED (no TCP_USER_TIMEOUT on this platform)")
    if _CONNECT_TIMEOUT_S > 0:
        bounds.append(f"connect {_CONNECT_TIMEOUT_S:.0f}s")
    logger.info(f"search socket bounds: {', '.join(bounds)} ({len(options)} socket options installed)")


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
                # Socket health. A request waiting on a retriever that has
                # gone away sees nothing at the socket layer -- no FIN, no RST.
                # Keepalive reaps an idle corpse before the next request rides
                # it; TCP_USER_TIMEOUT bounds one that already has a request
                # outstanding, which is the case that was costing this run 6.3
                # points of GPU idle. Both turn a silent wait into a connection
                # error, which the retry loop then handles on a fresh socket.
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
