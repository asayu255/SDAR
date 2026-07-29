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
Test memory buffers
- We start with two models with the same weights
- We use Memory buffer to make one of the models and then compare the parameters
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import gc

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import LlamaConfig, LlamaModel


# [EXPLAIN] `test_memory_buffers` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_memory_buffers():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    llama_config = LlamaConfig(
        vocab_size=256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = LlamaModel(config=llama_config).cuda()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_copy = LlamaModel(config=llama_config).cuda()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model_copy.load_state_dict(model.state_dict())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    norm_factor = 1024**3

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    t_before = torch.cuda.get_device_properties(0).total_memory / norm_factor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r_before = torch.cuda.memory_reserved(0) / norm_factor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a_before = torch.cuda.memory_allocated(0) / norm_factor

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Before Total memory: {t_before} GB, reserved: {r_before} GB, allocated: {a_before} GB")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    t = torch.cuda.get_device_properties(0).total_memory / norm_factor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    r = torch.cuda.memory_reserved(0) / norm_factor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    a = torch.cuda.memory_allocated(0) / norm_factor

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    gc.collect()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"After Total memory: {t} GB, reserved: {r} GB, allocated: {a} GB")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    change_ratio = (a - a_before) / a_before
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert change_ratio < 0.01, f"make sure the allocated change is less than 1%, Got {change_ratio}"

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_copy.named_parameters()):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert name1 == name2
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.eq(param1.data, param2.data).all(), f"{param1.data}, {param2.data}, {name1}"


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_memory_buffers()
