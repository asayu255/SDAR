# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from megatron.core import ModelParallelConfig, tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers.activations import ACT2FN

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.llama.megatron.layers.parallel_linear import MergedColumnParallelLinear
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron import tensor_parallel as tp_utils


# [EXPLAIN] `ParallelLlamaMLP` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ParallelLlamaMLP(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config, megatron_config: ModelParallelConfig = None) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hidden_size = config.hidden_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.intermediate_size = config.intermediate_size
        # The weight is only [hidden_size, intermediate_size // model_parallel_world_size]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if megatron_config is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert row_kwargs.get("config", False), "must have ModelParallelConfig"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            gate_ouput_size=self.intermediate_size,
            up_output_size=self.intermediate_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.gate_size = self.intermediate_size // tp_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.down_proj = tensor_parallel.RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.act_fn = ACT2FN[config.hidden_act]

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, x):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate_up = self.gate_up_proj(x)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        gate, up = gate_up.split(self.gate_size, dim=-1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.down_proj(self.act_fn(gate) * up)[0]
