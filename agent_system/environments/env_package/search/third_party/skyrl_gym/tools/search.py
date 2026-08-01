import json
import logging
import requests
import uuid
import time
import threading
from typing import Tuple, Optional, Any, Dict
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
                timeout=timeout,
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
