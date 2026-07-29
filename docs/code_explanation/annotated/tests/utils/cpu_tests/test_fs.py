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
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.fs as fs


# [EXPLAIN] `test_record_and_check_directory_structure` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_record_and_check_directory_structure(tmp_path):
    # Create test directory structure
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_dir = tmp_path / "test_dir"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_dir.mkdir()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    (test_dir / "file1.txt").write_text("test")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    (test_dir / "subdir").mkdir()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    (test_dir / "subdir" / "file2.txt").write_text("test")

    # Create structure record
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    record_file = fs._record_directory_structure(test_dir)

    # Verify record file exists
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert os.path.exists(record_file)

    # Initial check should pass
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert fs._check_directory_structure(test_dir, record_file) is True

    # Modify structure and verify check fails
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    (test_dir / "new_file.txt").write_text("test")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert fs._check_directory_structure(test_dir, record_file) is False


# [EXPLAIN] `test_copy_from_hdfs_with_mocks` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_copy_from_hdfs_with_mocks(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    # side_effect will simulate the copy by creating parent dirs + empty file
    # [EXPLAIN] `fake_copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fake_copy(src: str, dst: str, *args, **kwargs):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dst_path = Path(dst)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dst_path.write_bytes(b"")  # touch an empty file

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    # Test parameters
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_cache = tmp_path / "cache"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_path = "hdfs://test/path/file.txt"

    # Test initial copy
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_path = os.path.join(test_cache, fs.md5_encode(hdfs_path), os.path.basename(hdfs_path))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert local_path == expected_path
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert os.path.exists(local_path)


# [EXPLAIN] `test_always_recopy_flag` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_always_recopy_flag(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    copy_call_count = 0

    # [EXPLAIN] `fake_copy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def fake_copy(src: str, dst: str, *args, **kwargs):
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        nonlocal copy_call_count
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        copy_call_count += 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dst_path = Path(dst)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dst_path.write_bytes(b"")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_cache = tmp_path / "cache"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_path = "hdfs://test/path/file.txt"

    # Initial copy (always_recopy=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert copy_call_count == 1

    # Force recopy (always_recopy=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    fs.copy_to_local(hdfs_path, cache_dir=test_cache, always_recopy=True)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert copy_call_count == 2

    # Subsequent normal call (always_recopy=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert copy_call_count == 2  # Should not increment
