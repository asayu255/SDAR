#!/usr/bin/env python
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

# -*- coding: utf-8 -*-
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""File-system agnostic IO APIs"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hashlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import shutil
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tempfile
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hashlib

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from hdfs_io import copy, exists, makedirs  # for internal use only
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from .hdfs_io import copy, exists, makedirs

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["copy", "exists", "makedirs"]

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_HDFS_PREFIX = "hdfs://"


# [EXPLAIN] `is_non_local` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_non_local(path):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """Check if a path is a non-local (HDFS) path.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is an HDFS path, False otherwise.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return path.startswith(_HDFS_PREFIX)


# [EXPLAIN] `md5_encode` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def md5_encode(path: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Generate an MD5 hash of a path string.

    This function is used to create unique identifiers for paths, typically
    for creating cache directories or lock files.

    Args:
        path (str): The path to encode.

    Returns:
        str: The hexadecimal MD5 hash of the path.
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return hashlib.md5(path.encode()).hexdigest()


# [EXPLAIN] `get_local_temp_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Generate a unique local cache path for an HDFS resource.
    Creates a MD5-hashed subdirectory in cache_dir to avoid name conflicts,
    then returns path combining this subdirectory with the HDFS basename.

    Args:
        hdfs_path (str): Source HDFS path to be cached
        cache_dir (str): Local directory for storing cached files

    Returns:
        str: Absolute local filesystem path in format:
            {cache_dir}/{md5(hdfs_path)}/{basename(hdfs_path)}
    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    encoded_hdfs_path = md5_encode(hdfs_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(temp_dir, exist_ok=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dst

# [EXPLAIN] `verify_copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def verify_copy(src: str, dest: str) -> bool:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    verify the copy of src to dest by comparing their sizes and file structures.

    return:
        bool: True if the copy is verified, False otherwise.
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(src):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(dest):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if os.path.isfile(src) != os.path.isfile(dest):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if os.path.isfile(src):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_size = os.path.getsize(src)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dest_size = os.path.getsize(dest)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if src_size != dest_size:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return True

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_files = set()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dest_files = set()

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for root, dirs, files in os.walk(src):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rel_path = os.path.relpath(root, src)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dest_root = os.path.join(dest, rel_path) if rel_path != '.' else dest

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not os.path.exists(dest_root):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for entry in os.listdir(root):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            src_entry = os.path.join(root, entry)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            src_files.add(os.path.relpath(src_entry, src))

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for entry in os.listdir(dest_root):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dest_entry = os.path.join(dest_root, entry)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dest_files.add(os.path.relpath(dest_entry, dest))

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if src_files != dest_files:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for rel_path in src_files:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        src_entry = os.path.join(src, rel_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dest_entry = os.path.join(dest, rel_path)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.isdir(src_entry) != os.path.isdir(dest_entry):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return False

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.isfile(src_entry):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            src_size = os.path.getsize(src_entry)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dest_size = os.path.getsize(dest_entry)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if src_size != dest_size:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return False

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return True


# [EXPLAIN] `copy_to_shm` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def copy_to_shm(src:str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
        Load the model into   /dev/shm   to make the process of loading the model multiple times more efficient.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    shm_model_root = '/dev/shm/verl-cache/'
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    src_abs = os.path.abspath(os.path.normpath(src))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dest = os.path.join(shm_model_root, hashlib.md5(src_abs.encode('utf-8')).hexdigest())
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(dest, exist_ok=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dest = os.path.join(dest, os.path.basename(src_abs))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if os.path.exists(dest) and verify_copy(src, dest):
        # inform user and depends on him
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and restart the task.")
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if os.path.isdir(src):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            shutil.copy2(src, dest)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dest

# [EXPLAIN] `_record_directory_structure` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _record_directory_structure(folder_path):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    record_file = os.path.join(folder_path, ".directory_record.txt")
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(record_file, "w") as f:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for root, dirs, files in os.walk(folder_path):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for dir_name in dirs:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                f.write(f"dir:{relative_dir}\n")
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for file_name in files:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if file_name != ".directory_record.txt":
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    f.write(f"file:{relative_file}\n")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return record_file


# [EXPLAIN] `_check_directory_structure` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _check_directory_structure(folder_path, record_file):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not os.path.exists(record_file):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    existing_entries = set()
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for root, dirs, files in os.walk(folder_path):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for dir_name in dirs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            existing_entries.add(f"dir:{relative_dir}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for file_name in files:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if file_name != ".directory_record.txt":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                existing_entries.add(f"file:{relative_file}")
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(record_file) as f:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        recorded_entries = set(f.read().splitlines())
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return existing_entries == recorded_entries


# [EXPLAIN] `copy_to_local` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def copy_to_local(src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False, use_shm:bool=False) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Copy files/directories from HDFS to local cache with validation.

    Args:
        src (str): Source path - HDFS path (hdfs://...) or local filesystem path
        cache_dir (str, optional): Local directory for cached files. Uses system tempdir if None
        filelock (str): Base name for file lock. Defaults to ".file.lock"
        verbose (bool): Enable copy operation logging. Defaults to False
        always_recopy (bool): Force fresh copy ignoring cache. Defaults to False
        use_shm (bool): Enable shared memory copy. Defaults to False

    Returns:
        str: Local filesystem path to copied resource
    """
    # Save to a local path for persistence.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = copy_local_path_from_hdfs(src, cache_dir, filelock, verbose, always_recopy)
    # Load into shm to improve efficiency.
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if use_shm:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return copy_to_shm(local_path)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return local_path

# [EXPLAIN] `copy_local_path_from_hdfs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False) -> str:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Deprecated. Please use copy_to_local instead."""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from filelock import FileLock

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if is_non_local(src):
        # download from hdfs to local
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cache_dir is None:
            # get a temp folder
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cache_dir = tempfile.gettempdir()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(cache_dir, exist_ok=True)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert os.path.exists(cache_dir)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        filelock = md5_encode(src) + ".lock"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lock_file = os.path.join(cache_dir, filelock)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with FileLock(lock_file=lock_file):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if always_recopy and os.path.exists(local_path):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if os.path.isdir(local_path):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    shutil.rmtree(local_path, ignore_errors=True)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    os.remove(local_path)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not os.path.exists(local_path):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if verbose:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Copy from {src} to {local_path}")
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                copy(src, local_path)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if os.path.isdir(local_path):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    _record_directory_structure(local_path)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif os.path.isdir(local_path):
                # always_recopy=False, local path exists, and it is a folder: check whether there is anything missed
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                record_file = os.path.join(local_path, ".directory_record.txt")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if not _check_directory_structure(local_path, record_file):
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if verbose:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"Recopy from {src} to {local_path} due to missing files or directories.")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    shutil.rmtree(local_path, ignore_errors=True)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    copy(src, local_path)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    _record_directory_structure(local_path)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return local_path
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return src
