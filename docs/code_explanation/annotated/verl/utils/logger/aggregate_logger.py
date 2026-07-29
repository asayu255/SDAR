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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
A Ray logger will receive logging info from different processes.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numbers
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict


# [EXPLAIN] `concat_dict_to_str` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def concat_dict_to_str(dict: Dict, step):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = [f"step:{step}"]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k, v in dict.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(v, numbers.Number):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output.append(f"{k}:{v:.3f}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_str = " - ".join(output)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output_str


# [EXPLAIN] `LocalLogger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class LocalLogger:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.print_to_console = print_to_console
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if print_to_console:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Using LocalLogger is deprecated. The constructor API will change ")

    # [EXPLAIN] `flush` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def flush(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, data, step):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.print_to_console:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(concat_dict_to_str(data, step=step), flush=True)


# [EXPLAIN] `DecoratorLoggerBase` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DecoratorLoggerBase:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.role = role
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.logger = logger
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.level = level
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank = rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.log_only_rank_0 = log_only_rank_0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.logging_function = self.log_by_logging
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if logger is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.logging_function = self.log_by_print

    # [EXPLAIN] `log_by_print` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_by_print(self, log_str):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.log_only_rank_0 or self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"{self.role} {log_str}", flush=True)

    # [EXPLAIN] `log_by_logging` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log_by_logging(self, log_str):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.logger is None:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError("Logger is not initialized")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.log_only_rank_0 or self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.logger.log(self.level, f"{self.role} {log_str}")
