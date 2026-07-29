# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import concurrent.futures  # <-- Import concurrent.futures
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
DEFAULT_TIMEOUT = 10  # Default compile and run timeout
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_RETRIES = 3
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INITIAL_RETRY_DELAY = 1
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
API_TIMEOUT = 10

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
SUPPORTED_LANGUAGES = ["python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp", "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl", "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script", "jest", "verilog", "python_gpu", "lean", "swift", "racket"]


# [EXPLAIN] `call_sandbox_api` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def call_sandbox_api(sandbox_fusion_url: str, code: str, stdin: str, compile_timeout: int, run_timeout: int, language: str = "python") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if language not in SUPPORTED_LANGUAGES:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        error_msg = f"{log_prefix}Unsupported language: {language}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(error_msg)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None, error_msg

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    last_error = None  # Store the last error encountered

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for attempt in range(MAX_RETRIES):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}")  # <-- Use internal log_prefix
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response.status_code == 504:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                last_error = f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt {attempt + 1}/{MAX_RETRIES}"  # <-- Use internal log_prefix
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.warning(last_error)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    time.sleep(delay)
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            response.raise_for_status()

            # If successful (status code 2xx)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")  # <-- Use internal log_prefix
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return response.json(), None

        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except requests.exceptions.RequestException as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on non-504 request errors
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except json.JSONDecodeError as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_response_text = response.text if "response" in locals() else "N/A"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on JSON decode errors
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


# [EXPLAIN] `_process_single_case` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _process_single_case(case_index: int, stdin_data: Any, expected_output: Any, sandbox_fusion_url: str, generation: str, timeout: int, language: str, concurrent_semaphore: Optional[threading.Semaphore] = None, fn_name: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Helper function to process a single test case."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    api_response = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    error_msg = None
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.info(f"Processing test case {case_index + 1}.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    current_generation_code = generation

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if fn_name and language == "python":
        # Wrapper assumes stdin_data is a JSON string for function arguments.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: {{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class() 
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)
        
        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1) 
"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_generation_code = wrapper_code

    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if concurrent_semaphore:
            # logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with concurrent_semaphore:
                # logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, stdin=str(stdin_data), compile_timeout=timeout, run_timeout=timeout, language=language)
            # logger.debug(f"Case {case_index + 1}: Semaphore released.")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, stdin=str(stdin_data), compile_timeout=timeout, run_timeout=timeout, language=language)
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Case {case_index + 1}: {error_msg}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        traceback.print_exc()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metadata = {
        "case_index": case_index,
        "input": str(stdin_data),
        "expected_output": str(expected_output),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_status = -1  # Default error: API request error or unknown sandbox error

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if error_msg:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        metadata["status"] = "api_error"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result_status = -1  # API request itself failed (includes timeout after retries)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        # Log code and input only on error for brevity
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Case {case_index}: input: {str(stdin_data)}")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif api_response:
        # --- Add debug logging ---
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        metadata["api_response"] = api_response
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        metadata["api_status"] = api_response.get("status")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        compile_result = api_response.get("compile_result")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        run_result = api_response.get("run_result")

        # Extract compile information
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if compile_result:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["compile_status"] = compile_result.get("status")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["compile_duration"] = compile_result.get("execution_time")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["compile_stderr"] = compile_result.get("stderr")

        # Extract run information
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if run_result:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["run_status"] = run_result.get("status")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["stdout"] = run_result.get("stdout")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["exit_code"] = run_result.get("return_code")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            metadata["duration"] = run_result.get("execution_time")

        # --- Determine status based on API response ---
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        api_status = metadata["api_status"]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if api_status == "SandboxError":
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metadata["status"] = "sandbox_error"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result_status = -1  # Internal sandbox error
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif api_status == "Failed":
            # --- Add debug logging ---
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.debug(f"API returned Failed status. Response: {api_response}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.debug(f"Compile Result: {compile_result}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.debug(f"Run Result: {run_result}")
            # --- Check the logic here ---
            # Compile failed or timed out
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            is_compile_error = compile_result and (metadata["compile_status"] in ["Error", "TimeLimitExceeded"] or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if is_compile_error:
                # Differentiate between compile_error and compile_timeout based on specific status
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if metadata["compile_status"] == "TimeLimitExceeded":
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "compile_timeout"
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:  # Includes Error and Finished but return_code != 0 cases
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "compile_error"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result_status = -4
            # Run failed or timed out
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif run_result:
                # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                is_runtime_error = metadata["run_status"] == "TimeLimitExceeded" or metadata["run_status"] == "Error" or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if is_runtime_error:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if metadata["run_status"] == "TimeLimitExceeded":
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metadata["status"] = "timeout"  # Runtime timeout
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        result_status = -3
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:  # Includes Error and Finished with non-zero return_code
                        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                        metadata["status"] = "runtime_error"
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        result_status = -2
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # Other Failed status with run_result, classify as unknown failure
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "unknown_failure"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result_status = -1  # Default to -1
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Status is Failed but neither a clear compile error nor run_result exists
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metadata["status"] = "unknown_failure_state"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result_status = -1  # Default to -1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif api_status == "Success":
            # Run completed successfully, now check the answer
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if run_result and metadata["run_status"] == "Finished":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                # Note: Output might contain trailing newlines, need normalization
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result_status = True
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "success"
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    result_status = False
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata["status"] = "wrong_answer"
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # Status is Success but run_result status is not Finished, this is unexpected
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                metadata["status"] = "unexpected_success_state"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result_status = -1  # Classify as unknown error
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # API returned an unknown top-level status
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.warning(f"Unknown API status received: {api_status}")
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            metadata["status"] = f"unknown_api_status_{api_status}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result_status = -1  # Default to -1
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        metadata["status"] = "unknown_api_state"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result_status = -1
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return result_status, metadata


# [EXPLAIN] `check_correctness` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def check_correctness(sandbox_fusion_url: str, in_outs: Optional[dict], generation: str, timeout: int = DEFAULT_TIMEOUT, language: str = "python", concurrent_semaphore: Optional[threading.Semaphore] = None) -> Tuple[List[Any], List[Dict[str, Any]]]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.info("Starting correctness check for generation.")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning("Invalid in_outs format provided.")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [-1], [{"error": "Invalid input/output data"}]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inputs = in_outs["inputs"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_outputs = in_outs["outputs"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fn_name = in_outs.get("fn_name")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_cases = len(inputs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = [None] * num_cases  # Initialize with placeholders
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    metadata_list = [None] * num_cases  # Initialize with placeholders

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_cases == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning("Empty inputs provided.")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [], []

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(inputs) != len(expected_outputs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning(f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)}).")
        # Return error based on the number of inputs provided
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    first_compile_error_index = -1

    # max_workers is limited by sandbox_fusion_max_concurrent from concurrent_semaphore
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        # Submit all tasks, passing the concurrent_semaphore to _process_single_case
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        future_to_index = {executor.submit(_process_single_case, i, stdin_data, expected_outputs[i], sandbox_fusion_url, generation, timeout, language, concurrent_semaphore, fn_name): i for i, stdin_data in enumerate(inputs)}

        # Process results as they complete
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for future in concurrent.futures.as_completed(future_to_index):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            index = future_to_index[future]
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                result_status, metadata = future.result()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                results[index] = result_status
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metadata_list[index] = metadata

                # Check for compile error (-4)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if result_status == -4:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        first_compile_error_index = index
                    # Optimization: could potentially cancel futures for index > first_compile_error_index
                    # However, cancellation is not guaranteed. Post-processing is safer.

            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as exc:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.error(f"Test case {index} generated an exception: {exc}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                traceback.print_exc()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                results[index] = -1  # Mark as API/internal error
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]),
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Post-processing for compile errors
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if first_compile_error_index != -1:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.warning(f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors.")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(first_compile_error_index + 1, num_cases):
            # Only update if not already processed (though it should be None or have a result)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                results[i] = -4
                # Update or create metadata for skipped cases due to compile error
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if metadata_list[i] is None:  # If future failed before returning metadata
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]),
                        "api_request_error": None,
                        "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                    }
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:  # If future completed but result is overridden
                    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                    metadata_list[i]["status"] = "compile_error_skipped"

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logger.info(f"Correctness check finished. Results: {results}")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return results, metadata_list
