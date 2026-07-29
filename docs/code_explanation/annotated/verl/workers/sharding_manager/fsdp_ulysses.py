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
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import DeviceMesh

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import all_gather_data_proto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base import BaseShardingManager


# [EXPLAIN] `FSDPUlyssesShardingManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPUlyssesShardingManager(BaseShardingManager):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, device_mesh: DeviceMesh):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.seed_offset = 12345

    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            # TODO: check how to set seed for each model

    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # revert to previous sp group
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # TODO: check how to set seed for each model

    # [EXPLAIN] `preprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            group = self.device_mesh["sp"].get_group()

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            all_gather_data_proto(data=data, process_group=group)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # [EXPLAIN] `postprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Split the data to follow FSDP partition
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sp_size = self.device_mesh["sp"].size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sp_rank = self.device_mesh["sp"].get_local_rank()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = data.chunk(chunks=sp_size)[sp_rank]
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data
