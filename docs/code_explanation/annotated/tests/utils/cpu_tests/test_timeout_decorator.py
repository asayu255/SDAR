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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import multiprocessing
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import threading
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest  # Import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.py_functional import timeout_limit as timeout

# --- Test Task Functions ---
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TEST_TIMEOUT_SECONDS = 1.5  # Timeout duration for tests
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
LONG_TASK_DURATION = TEST_TIMEOUT_SECONDS + 0.5  # Duration slightly longer than timeout


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@timeout(seconds=TEST_TIMEOUT_SECONDS)  # Keep global decorator for mp tests
# [EXPLAIN] `quick_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def quick_task(x):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A task that completes quickly."""
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(0.1)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "quick_ok"


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@timeout(seconds=TEST_TIMEOUT_SECONDS)  # Keep global decorator for mp tests
# [EXPLAIN] `slow_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def slow_task(x):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A task that takes longer than the timeout."""
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(LONG_TASK_DURATION)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "slow_finished"  # This return value indicates it didn't time out


# REMOVE global decorator here
# [EXPLAIN] `task_raises_value_error` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def task_raises_value_error():  # Now truly not globally decorated
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A task that intentionally raises a ValueError."""
    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
    raise ValueError("Specific value error from task")


# --- Top-level function for signal test in subprocess ---
# Keep this decorated globally for the specific subprocess test case
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
# [EXPLAIN] `top_level_decorated_quick_task_signal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def top_level_decorated_quick_task_signal():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A pickleable top-level function decorated with signal timeout."""
    # Assuming this calls the logic of quick_task directly for the test purpose
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(0.1)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "quick_ok_signal_subprocess"  # Different return for clarity if needed


# --- Top-level function for signal test in subprocess ---
# Keep this decorated globally for the specific subprocess test case
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
# [EXPLAIN] `top_level_decorated_slow_task_signal` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def top_level_decorated_slow_task_signal():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """A pickleable top-level function decorated with signal timeout."""
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    time.sleep(LONG_TASK_DURATION)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return "slow_finished"


# --- NEW: Top-level helper function to run target in process ---
# [EXPLAIN] `run_target_and_put_in_queue` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_target_and_put_in_queue(target_func, q):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Top-level helper function to run a target function and put its result or exception into a queue.
    This function is pickleable and can be used as the target for multiprocessing.Process.
    """
    # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
    try:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        result = target_func()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        q.put(("success", result))
    # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
    except Exception as e:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        q.put(("error", e))


# Use a module-level fixture to set the start method on macOS
# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.fixture(scope="module", autouse=True)  # Changed scope to module
# [EXPLAIN] `set_macos_start_method` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_macos_start_method():
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sys.platform == "darwin":
        # Force fork method on macOS to avoid pickling issues with globally decorated functions
        # when running tests via pytest discovery.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        current_method = multiprocessing.get_start_method(allow_none=True)
        # Only set if not already set or if set to something else (less likely in test run)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if current_method is None or current_method != "fork":
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                multiprocessing.set_start_method("fork", force=True)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except RuntimeError:
                # Might fail if context is already started, ignore in that case.
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                pass


# [EXPLAIN] `test_quick_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_quick_task():  # Renamed from test_multiprocessing_quick_task
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests timeout handles a quick task correctly."""
    # Call the globally decorated function directly
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result = quick_task(1)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert result == "quick_ok"  # Use pytest assert


# [EXPLAIN] `test_slow_task_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_slow_task_timeout():  # Renamed from test_multiprocessing_slow_task_timeout
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests timeout correctly raises TimeoutError for a slow task."""
    # Call the globally decorated function directly within pytest.raises
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(TimeoutError) as excinfo:  # Use pytest.raises
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        slow_task(1)
    # Check the error message from the multiprocessing implementation
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)  # Use pytest assert


# [EXPLAIN] `test_internal_exception` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_internal_exception():  # Renamed from test_multiprocessing_internal_exception
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests timeout correctly propagates internal exceptions."""
    # Apply the default timeout decorator dynamically to the undecorated function
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS)(task_raises_value_error)  # Apply decorator dynamically
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(ValueError) as excinfo:  # Use pytest.raises
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        decorated_task()  # Call the dynamically decorated function
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert str(excinfo.value) == "Specific value error from task"  # Use pytest assert


# --- Test the signal implementation (use_signals=True) ---
# Note: As per py_functional.py, use_signals=True currently falls back to
# multiprocessing on POSIX. These tests verify that behavior.


# [EXPLAIN] `test_signal_quick_task_main_process` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_signal_quick_task_main_process():  # Removed self
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests signal timeout handles a quick task correctly in the main process."""

    # Apply the signal decorator dynamically
    # [EXPLAIN] `plain_quick_task_logic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def plain_quick_task_logic():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(0.1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "quick_ok_signal"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_quick_task_logic)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert decorated_task() == "quick_ok_signal"  # Use pytest assert


# [EXPLAIN] `test_signal_slow_task_main_process_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_signal_slow_task_main_process_timeout():  # Removed self
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Tests signal timeout correctly raises TimeoutError for a slow task in the main process."""

    # Apply the signal decorator dynamically
    # [EXPLAIN] `plain_slow_task_logic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def plain_slow_task_logic():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        time.sleep(LONG_TASK_DURATION)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "slow_finished_signal"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_slow_task_logic)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with pytest.raises(TimeoutError) as excinfo:  # Use pytest.raises
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        decorated_task()
    # Check the error message (falls back to multiprocessing message on POSIX)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)  # Use pytest assert


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.skip(reason="this test won't pass. Just to show why use_signals should not be used")
# [EXPLAIN] `test_signal_in_thread_does_not_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_signal_in_thread_does_not_timeout():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Tests that signal-based timeout does NOT work reliably in a child thread.
    The TimeoutError from the signal handler is not expected to be raised.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_container = []  # Use a list to store result from thread
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exception_container = []  # Use a list to store exception from thread

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
    # [EXPLAIN] `slow_task_in_thread` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def slow_task_in_thread():
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Thread: Starting slow task...")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(LONG_TASK_DURATION)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Thread: Slow task finished.")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "slow_finished_in_thread"
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # Catch any exception within the thread's target function
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Thread: Caught exception: {e}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            exception_container.append(e)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None  # Indicate failure

    # [EXPLAIN] `thread_target` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def thread_target():
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # Run the decorated function inside the thread
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res = slow_task_in_thread()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if res is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                result_container.append(res)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # This might catch exceptions happening *outside* the decorated function
            # but still within the thread target, though less likely here.
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Thread Target: Caught exception: {e}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            exception_container.append(e)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    thread = threading.Thread(target=thread_target)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Main: Starting thread...")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    thread.start()
    # Wait longer than the timeout + task duration to ensure the thread finishes
    # regardless of whether timeout worked or not.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    thread.join(timeout=LONG_TASK_DURATION + 1)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(exception_container) == 1
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(exception_container[0], TimeoutError)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not result_container


# [EXPLAIN] `test_in_thread_timeout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_in_thread_timeout():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    result_container = []  # Use a list to store result from thread
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    exception_container = []  # Use a list to store exception from thread

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=False)
    # [EXPLAIN] `slow_task_in_thread` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def slow_task_in_thread():
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Thread: Starting slow task...")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            time.sleep(LONG_TASK_DURATION)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Thread: Slow task finished.")
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "slow_finished_in_thread"
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # Catch any exception within the thread's target function
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Thread: Caught exception: {e}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            exception_container.append(e)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None  # Indicate failure

    # [EXPLAIN] `thread_target` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def thread_target():
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # Run the decorated function inside the thread
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            res = slow_task_in_thread()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if res is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                result_container.append(res)
        # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
        except Exception as e:
            # This might catch exceptions happening *outside* the decorated function
            # but still within the thread target, though less likely here.
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Thread Target: Caught exception: {e}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            exception_container.append(e)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    thread = threading.Thread(target=thread_target)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Main: Starting thread...")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    thread.start()
    # Wait longer than the timeout + task duration to ensure the thread finishes
    # regardless of whether timeout worked or not.
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    thread.join(timeout=LONG_TASK_DURATION + 1)

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(exception_container) == 1
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(exception_container[0], TimeoutError)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert not result_container
