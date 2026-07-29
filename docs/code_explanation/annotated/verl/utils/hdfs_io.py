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
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import shutil

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_HDFS_PREFIX = "hdfs://"

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_HDFS_BIN_PATH = shutil.which("hdfs")


# [EXPLAIN] `exists` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def exists(path: str, **kwargs) -> bool:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _is_non_local(path):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _exists(path, **kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.exists(path)


# [EXPLAIN] `_exists` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _exists(file_path: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """hdfs capable to check whether a file_path is exists"""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if file_path.startswith("hdfs"):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.path.exists(file_path)


# [EXPLAIN] `makedirs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _mkdir(name, **kwargs)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


# [EXPLAIN] `_mkdir` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _mkdir(file_path: str) -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """hdfs mkdir"""
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if file_path.startswith("hdfs"):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(file_path, exist_ok=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return True


# [EXPLAIN] `copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def copy(src: str, dst: str, **kwargs) -> bool:
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return _copy(src, dst)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.isdir(src):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return shutil.copytree(src, dst, **kwargs)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return shutil.copy(src, dst, **kwargs)


# [EXPLAIN] `_copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if to_path.startswith("hdfs"):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if from_path.startswith("hdfs"):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if from_path.startswith("hdfs"):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                shutil.copy(from_path, to_path)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                returncode = 0
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except shutil.SameFileError:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                returncode = 0
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception as e:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                returncode = -1
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return returncode == 0


# [EXPLAIN] `_run_cmd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _run_cmd(cmd: str, timeout=None):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return os.system(cmd)


# [EXPLAIN] `_hdfs_cmd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _hdfs_cmd(cmd: str) -> str:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


# [EXPLAIN] `_is_non_local` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _is_non_local(path: str):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return path.startswith(_HDFS_PREFIX)
