import json
import logging
import os
import requests
import uuid
import time
import threading
from collections import OrderedDict
from typing import Tuple, Optional, Any, Dict
from urllib.parse import urlparse

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool, ToolGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
# Upper bound (seconds) for the backoff delay, so indefinite retries do not grow
# the wait without limit. Override with SEARCH_MAX_RETRY_DELAY.
MAX_RETRY_DELAY = int(os.environ.get("SEARCH_MAX_RETRY_DELAY", "30"))

# When enabled (default), connection/timeout errors -- the retriever is
# unreachable or not started yet (e.g. "[Errno 113] No route to host") -- are
# retried indefinitely until the service becomes reachable again, instead of
# giving up after MAX_RETRIES. Other errors (4xx, JSON decode, 5xx) still stop
# after MAX_RETRIES. Set SEARCH_WAIT_FOR_SERVICE=0 to restore the old bounded
# behavior. NOTE: if the retriever never comes back, callers block forever.
WAIT_FOR_SERVICE = os.environ.get("SEARCH_WAIT_FOR_SERVICE", "1").strip().lower() in ("1", "true", "yes", "on")

# Cache retrieval results per (url, topk, query). RL rollouts re-issue the same
# search query many times (identical questions across GRPO group members and
# across epochs of the same subset), and the retriever is deterministic for a
# fixed index, so a hit returns the byte-identical observation while skipping
# the HTTP round trip. Only successful lookups are cached (errors are retried
# normally). Opt-in: SEARCH_QUERY_CACHE=1; size cap via SEARCH_QUERY_CACHE_SIZE.
_QUERY_CACHE_ENABLED = os.environ.get("SEARCH_QUERY_CACHE", "0").strip().lower() in ("1", "true", "yes", "on")
_QUERY_CACHE_MAX_SIZE = int(os.environ.get("SEARCH_QUERY_CACHE_SIZE", "100000"))


# retrieval HTTPはtimeoutとserver 5xx/connection failureを区別し、bounded retryを行う。`WAIT_FOR_SERVICE`有効時のconnection/timeoutだけはservice起動待ちとして無期限retryになり得る。
# 4xx、JSON decode、その他例外は即座にerror resultへ変換する。sleepはこのhost threadだけを止め、他envのthreadやGPU実行を直接同期しない。
def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a single query.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.

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

    last_error = None
    attempt = 0
    while True:
        attempt += 1
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES:
                    delay = min(INITIAL_RETRY_DELAY * attempt, MAX_RETRY_DELAY)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                    continue
                break

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt}")

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Retriever is unreachable / not started yet. Retry indefinitely while
            # WAIT_FOR_SERVICE is enabled, otherwise fall back to MAX_RETRIES.
            if isinstance(e, requests.exceptions.Timeout):
                last_error = f"{log_prefix}Timeout Error: {e}"
            else:
                last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if WAIT_FOR_SERVICE or attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * attempt, MAX_RETRY_DELAY)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
                continue
            break
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    logger.error(f"{log_prefix}API Request Failed after {attempt} attempts: {last_error}")

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        format_reference += f"Doc {idx+1}: {content}\n"
    return format_reference


# 全Search envでbase URL別`requests.Session`を共有してconnection poolを再利用し、`(URL, topk, query)`をkeyにbounded LRUで同一retrieval結果を共有する。
# session poolとquery cacheは別lockで保護される。cache hitはHTTPを完全に省くが、retriever indexがrun中に変わらないことを前提とするため実験中のindex更新時は無効化が必要である。
class SearchToolGroup(ToolGroup):
    # Class-level session pool shared across all instances
    _session_pool = {}
    _session_lock = threading.Lock()

    # Class-level LRU query cache shared across all env instances (all envs of a
    # run query the same retriever index).
    _query_cache: "OrderedDict[tuple, str]" = OrderedDict()
    _query_cache_lock = threading.Lock()

    @classmethod
    def _query_cache_get(cls, key):
        with cls._query_cache_lock:
            result = cls._query_cache.get(key)
            if result is not None:
                cls._query_cache.move_to_end(key)
            return result

    @classmethod
    def _query_cache_put(cls, key, value):
        with cls._query_cache_lock:
            cls._query_cache[key] = value
            cls._query_cache.move_to_end(key)
            while len(cls._query_cache) > _QUERY_CACHE_MAX_SIZE:
                cls._query_cache.popitem(last=False)

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
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    def __init__(self, search_url="http://127.0.0.1:8000/retrieve", topk=3, timeout=DEFAULT_TIMEOUT, log_requests=True):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
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

        cache_key = (self.search_url, self.topk, query) if _QUERY_CACHE_ENABLED else None
        if cache_key is not None:
            cached = self._query_cache_get(cache_key)
            if cached is not None:
                return cached

        try:
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,  # Pass our shared session for connection reuse
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
                if cache_key is not None:
                    # Cache only successful lookups (incl. deterministic empty
                    # results); errors keep hitting the API.
                    self._query_cache_put(cache_key, result_text)
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
