# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py


# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import tensor_parallel


# [EXPLAIN] `QKVParallelLinear` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class QKVParallelLinear(tensor_parallel.ColumnParallelLinear):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        input_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.input_size = input_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.q_output_size = num_heads * head_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.kv_output_size = num_key_value_heads * head_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.head_dim = head_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gather_output = gather_output
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skip_bias_add = skip_bias_add

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_size = self.input_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_size = (num_heads + 2 * num_key_value_heads) * self.head_dim

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )


# [EXPLAIN] `MergedColumnParallelLinear` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MergedColumnParallelLinear(tensor_parallel.ColumnParallelLinear):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        input_size,
        gate_ouput_size,
        up_output_size,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.input_size = input_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.output_size = gate_ouput_size + up_output_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gather_output = gather_output
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skip_bias_add = skip_bias_add

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(
            input_size=self.input_size,
            output_size=self.output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )
