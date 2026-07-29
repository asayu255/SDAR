# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import traceback
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import uuid
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, Dict, List, Optional, Tuple

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import requests

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DEFAULT_TIMEOUT = 30  # Default search request timeout
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_RETRIES = 10
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INITIAL_RETRY_DELAY = 1
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
API_TIMEOUT = 10
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

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)


# [EXPLAIN] `call_search_api` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def call_search_api(retrieval_service_url: str, query_list: List[str], topk: int = 3, return_scores: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request_id = str(uuid.uuid4())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_prefix = f"[Search Request ID: {request_id}] "

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

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
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(f"{log_prefix}Attempt {attempt}: Calling search API at {retrieval_service_url}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = requests.post(
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
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt}")
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

    # If loop finishes without returning success, return the last recorded error
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


# [EXPLAIN] `_passages2string` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _passages2string(retrieval_result):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Convert retrieval results to formatted string."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_reference = ""
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for idx, doc_item in enumerate(retrieval_result):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        content = doc_item["document"]["contents"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        title = content.split("\n")[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text = "\n".join(content.split("\n")[1:])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return format_reference.strip()


# [EXPLAIN] `perform_single_search_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def perform_single_search_batch(retrieval_service_url: str, query_list: List[str], topk: int = 3, concurrent_semaphore: Optional[threading.Semaphore] = None, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, Dict[str, Any]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    api_response = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    error_msg = None

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if concurrent_semaphore:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with concurrent_semaphore:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                api_response, error_msg = call_search_api(retrieval_service_url=retrieval_service_url, query_list=query_list, topk=topk, return_scores=True, timeout=timeout)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            api_response, error_msg = call_search_api(retrieval_service_url=retrieval_service_url, query_list=query_list, topk=topk, return_scores=True, timeout=timeout)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        error_msg = f"API Request Exception during batch search: {e}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Batch search: {error_msg}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        traceback.print_exc()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
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
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info("Batch search: No results found")
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
    return result_text, metadata
