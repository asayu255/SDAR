# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import requests
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uuid
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import OrderedDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Tuple, Optional, Any, Dict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from urllib.parse import urlparse

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool, ToolGroup

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(logging.INFO)

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DEFAULT_TIMEOUT = 30
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_RETRIES = 10
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INITIAL_RETRY_DELAY = 1
# Upper bound (seconds) for the backoff delay, so indefinite retries do not grow
# the wait without limit. Override with SEARCH_MAX_RETRY_DELAY.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_RETRY_DELAY = int(os.environ.get("SEARCH_MAX_RETRY_DELAY", "30"))

# When enabled (default), connection/timeout errors -- the retriever is
# unreachable or not started yet (e.g. "[Errno 113] No route to host") -- are
# retried indefinitely until the service becomes reachable again, instead of
# giving up after MAX_RETRIES. Other errors (4xx, JSON decode, 5xx) still stop
# after MAX_RETRIES. Set SEARCH_WAIT_FOR_SERVICE=0 to restore the old bounded
# behavior. NOTE: if the retriever never comes back, callers block forever.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
WAIT_FOR_SERVICE = os.environ.get("SEARCH_WAIT_FOR_SERVICE", "1").strip().lower() in ("1", "true", "yes", "on")

# Cache retrieval results per (url, topk, query). RL rollouts re-issue the same
# search query many times (identical questions across GRPO group members and
# across epochs of the same subset), and the retriever is deterministic for a
# fixed index, so a hit returns the byte-identical observation while skipping
# the HTTP round trip. Only successful lookups are cached (errors are retried
# normally). Opt-in: SEARCH_QUERY_CACHE=1; size cap via SEARCH_QUERY_CACHE_SIZE.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_QUERY_CACHE_ENABLED = os.environ.get("SEARCH_QUERY_CACHE", "0").strip().lower() in ("1", "true", "yes", "on")
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_QUERY_CACHE_MAX_SIZE = int(os.environ.get("SEARCH_QUERY_CACHE_SIZE", "100000"))


# [EXPLAIN] `call_search_api` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
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
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request_id = str(uuid.uuid4())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_prefix = f"[Search Request ID: {request_id}] "

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    payload = {"query": query, "topk": topk, "return_scores": return_scores}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Use provided session or create a new one for this request
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if session is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        session = requests.Session()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        should_close_session = True
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        should_close_session = False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    last_error = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attempt = 0
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while True:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        attempt += 1
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if log_requests:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info(
                    f"{log_prefix}Attempt {attempt}: Calling search API at {retrieval_service_url}"
                )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response.status_code in [500, 502, 503, 504]:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt}"
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.warning(last_error)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if attempt < MAX_RETRIES:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    delay = min(INITIAL_RETRY_DELAY * attempt, MAX_RETRY_DELAY)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    time.sleep(delay)
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break

            # Check for other HTTP errors (e.g., 4xx)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            response.raise_for_status()

            # If successful (status code 2xx)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if log_requests:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt}")

            # Close session if we created it
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if should_close_session:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                session.close()

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return response.json(), None

        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Retriever is unreachable / not started yet. Retry indefinitely while
            # WAIT_FOR_SERVICE is enabled, otherwise fall back to MAX_RETRIES.
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(e, requests.exceptions.Timeout):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                last_error = f"{log_prefix}Timeout Error: {e}"
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                last_error = f"{log_prefix}Connection Error: {e}"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.warning(last_error)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if WAIT_FOR_SERVICE or attempt < MAX_RETRIES:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                delay = min(INITIAL_RETRY_DELAY * attempt, MAX_RETRY_DELAY)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                time.sleep(delay)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except requests.exceptions.RequestException as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}API Request Error: {e}"
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on other request errors
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except json.JSONDecodeError as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_response_text = response.text if "response" in locals() else "N/A"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on JSON decode errors
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}Unexpected Error: {e}"
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.error(f"{log_prefix}API Request Failed after {attempt} attempts: {last_error}")

    # Close session if we created it
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if should_close_session:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        session.close()

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return None, last_error


# [EXPLAIN] `_passages2string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _passages2string(retrieval_result):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_reference = ""
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for idx, doc_item in enumerate(retrieval_result):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        content = doc_item["document"]["contents"].strip()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        format_reference += f"Doc {idx+1}: {content}\n"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return format_reference


# [EXPLAIN] `SearchToolGroup` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SearchToolGroup(ToolGroup):
    # Class-level session pool shared across all instances
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _session_pool = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _session_lock = threading.Lock()

    # Class-level LRU query cache shared across all env instances (all envs of a
    # run query the same retriever index).
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _query_cache: "OrderedDict[tuple, str]" = OrderedDict()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    _query_cache_lock = threading.Lock()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `_query_cache_get` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _query_cache_get(cls, key):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with cls._query_cache_lock:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = cls._query_cache.get(key)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if result is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                cls._query_cache.move_to_end(key)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return result

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `_query_cache_put` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _query_cache_put(cls, key, value):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with cls._query_cache_lock:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cls._query_cache[key] = value
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            cls._query_cache.move_to_end(key)
            # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
            while len(cls._query_cache) > _QUERY_CACHE_MAX_SIZE:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                cls._query_cache.popitem(last=False)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @classmethod
    # [EXPLAIN] `_get_shared_session` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get or create a shared session for the given base URL"""
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with cls._session_lock:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if base_url not in cls._session_pool:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                session = requests.Session()
                # Configure connection pooling
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=512,  # Number of connection pools
                    pool_maxsize=512,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                session.mount("http://", adapter)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                session.mount("https://", adapter)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                cls._session_pool[base_url] = session
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info(f"Created shared session pool for {base_url}")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return cls._session_pool[base_url]

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, search_url="http://127.0.0.1:8000/retrieve", topk=3, timeout=DEFAULT_TIMEOUT, log_requests=True):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.search_url = search_url
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.topk = topk
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.timeout = timeout
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.log_requests = log_requests

        # Extract base URL for session sharing
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parsed_url = urlparse(self.search_url)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.session = self._get_shared_session(self.base_url)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.log_requests:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(f"SearchToolGroup initialized using shared session pool for {self.base_url}")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(name="SearchToolGroup")

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @tool
    # [EXPLAIN] `search` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def search(self, query: str) -> str:
        # NOTE(shu): add warning messages here?
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if query is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ""

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        query = query.strip()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cache_key = (self.search_url, self.topk, query) if _QUERY_CACHE_ENABLED else None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cache_key is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cached = self._query_cache_get(cache_key)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cached is not None:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return cached

        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,  # Pass our shared session for connection reuse
            )
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            error_msg = f"API Request Exception during batch search: {e}"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error(f"Batch search: {error_msg}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        metadata = {
            "query": query,
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "total_results": 0,
            "formatted_result": None,
        }

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result_text = json.dumps({"result": "Search request failed or timed out after retries."})

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if error_msg:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metadata["status"] = "api_error"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result_text = json.dumps({"result": f"Search error: {error_msg}"})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error(f"Batch search: API error occurred: {error_msg}")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif api_response:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.debug(f"Batch search: API Response: {api_response}")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metadata["api_response"] = api_response

            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_results = api_response.get("result", [])
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if raw_results:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    pretty_results = []
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    total_results = 0
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for retrieval in raw_results:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        formatted = _passages2string(retrieval)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        pretty_results.append(formatted)
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        total_results += len(retrieval) if isinstance(retrieval, list) else 1

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    final_result = "\n---\n".join(pretty_results)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result_text = json.dumps({"result": final_result})
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "success"
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["total_results"] = total_results
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["formatted_result"] = final_result
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.log_requests:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        logger.info(f"Batch search: Successful, got {total_results} total results")
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result_text = json.dumps({"result": "No search results found."})
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "no_results"
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["total_results"] = 0
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.log_requests:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        logger.info("Batch search: No results found")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if cache_key is not None:
                    # Cache only successful lookups (incl. deterministic empty
                    # results); errors keep hitting the API.
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self._query_cache_put(cache_key, result_text)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                error_msg = f"Error processing search results: {e}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result_text = json.dumps({"result": error_msg})
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metadata["status"] = "processing_error"
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.error(f"Batch search: {error_msg}")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metadata["status"] = "unknown_api_state"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result_text = json.dumps({"result": "Unknown API state (no response and no error message)."})
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.error("Batch search: Unknown API state.")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return result_text
