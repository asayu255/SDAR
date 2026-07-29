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
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed


# [EXPLAIN] `Profiler` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Profiler:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skip_prof = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.saved = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prof = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank = torch.distributed.get_rank()
        # we need to validate the config before using the profiler
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._validate()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.use_profile and self.rank in self.config.profile_ranks:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Profiler] Profiler init for rank {self.rank}")

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=max(self.config.step_start - 1, 0),
                    warmup=1 if self.config.step_start > 0 else 0,
                    active=self.config.step_end - self.config.step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    # [EXPLAIN] `_validate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _validate(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.use_profile:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.profile_ranks is None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[WARNING] Profile ranks is not set, default to rank 0")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.profile_ranks = [0]
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.step_start >= 0, "[ERROR] Profile step start must be greater than 0"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.step_end >= 0, "[ERROR] Profile step end must be greater than 0"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.config.step_start < self.config.step_end, "[ERROR] Profile step start must be less than step end"

    # [EXPLAIN] `check` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def check(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.prof is not None and not self.skip_prof

    # [EXPLAIN] `start` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def start(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.check():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Profiler] started for rank {self.rank}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.prof.start()

    # [EXPLAIN] `step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def step(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.check():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.prof.step()

    # [EXPLAIN] `stop` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def stop(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.check():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Profiler] stopped for rank {self.rank}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.prof.stop()

    # [EXPLAIN] `save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.prof is not None and not self.saved:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not os.path.exists(self.config.save_path):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                os.makedirs(self.config.save_path)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            save_file_name = f"/prof_start_{self.config.step_start}_end_{self.config.step_end}_rank_{self.rank}.json"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Profiler] Saving trace to {self.config.save_path + save_file_name}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.prof.export_chrome_trace(self.config.save_path + save_file_name)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.skip_prof = True
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.saved = True

    # [EXPLAIN] `stop_and_save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def stop_and_save(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.check():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.stop()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.save()

    # [EXPLAIN] `stop_trace` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def stop_trace(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.check():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[Profiler] Trace stopped for rank {self.rank}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.skip_prof = True
