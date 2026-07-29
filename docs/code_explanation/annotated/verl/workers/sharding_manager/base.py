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
"""
Sharding manager to implement HybridEngine
"""

from verl import DataProto


# sharding managerのcontext境界はtraining表現からinference表現への一時的な切替を表す。`__enter__`でweight/RNG/memoryをrollout側へ移し、`__exit__`でtraining状態へ戻す。
# `preprocess_data`/`postprocess_data`は同じglobal batchをinference TP/SP配置へgatherし、終了後に元rankのsliceへ戻す契約である。
class BaseShardingManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
