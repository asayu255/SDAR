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
import math

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.flops_counter import FlopsCounter

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3"}


# [EXPLAIN] `Config` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Config:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config_dict):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, value in config_dict.items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(self, key, value)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CONFIG = {
    "llama": {
        "config": {  # llama2-7B
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum + 12*sum(seqlen^2)*layer*head*head_dim
        # 6*(32000*4096*2+32*(4096*4096*4+4096*11008*3))*(512+1024+2048) + 12*(512*512+1024*1024+2048*2048)*32*4096
        # 6*(32000*4096*2+32*(4096*4096*4+4096*11008*3))*(4096+4096+4096) + 12*(4096*4096+4096*4096+4096*4096)*32*4096
        "expected_flops_tuple": (153555818250240 / 1e12, 575955114393600 / 1e12),
    },
    "qwen2": {
        "config": {  # Qwen/Qwen2.5-7B-Instruct
            "model_type": "qwen2",
            "vocab_size": 152064,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum + 12*sum(seqlen^2)*layer*head*head_dim
        # 6*(152064*3584*2+28*(3584*(3584+512+512+3584)+3584*18944*3))*(512+1024+2048) + 12*(512*512+1024*1024+2048*2048)*28*3584
        # 6*(152064*3584*2+28*(3584*(3584+512+512+3584)+3584*18944*3))*(4096+4096+4096) + 12*(4096*4096+4096*4096+4096*4096)*28*3584
        "expected_flops_tuple": (170388331954176 / 1e12, 622070178250752 / 1e12),
    },
    "qwen3": {
        "config": {  # Qwen/Qwen3-8B
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+ hidden*inter*3))*token_sum + 12*sum(seqlen^2)*layer*head*head_dim
        # 6*(151936*4096*2+36*(4096*(128*32+128*8*2+128*32)+4096*12288*3))*(512+1024+2048) + 12*(512*512+1024*1024+2048*2048)*36*128*32
        # 6*(151936*4096*2+36*(4096*(128*32+128*8*2+128*32)+4096*12288*3))*(4096+4096+4096) + 12*(4096*4096+4096*4096+4096*4096)*36*128*32
        "expected_flops_tuple": (185867930959872 / 1e12, 692924253732864 / 1e12),
    },
    "qwen3_moe": {
        "config": {  # Qwen/Qwen3-30B-A3B-Base
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # 6*(vocab*hidden*2+layer*(hidden*(q+k+v+head*head_dim)+hidden*inter*top_k_exp*3 + hidden*num_experts))*token_sum + 12*sum(seqlen^2)*layer*head*head_dim
        # 6*(151936*2048*2+48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128))*(512+1024+2048) + 12*(512*512+1024*1024+2048*2048)*48*128*32
        # 6*(151936*2048*2+48*(2048*(128*32+128*4*2+128*32)+2048*768*8*3+2048*128))*(4096+4096+4096) + 12*(4096*4096+4096*4096+4096*4096)*48*128*32
        "expected_flops_tuple": (85087060230144 / 1e12, 365944098521088 / 1e12),
    },
    "deepseek_v3": {
        "config": {  # deepseek-ai/DeepSeek-Prover-V2-671B
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_tuple": ([512, 1024, 2048], [4096, 4096, 4096]),
        # (1536*7168+128*192*1536+7168*(512+64)+128*(128+128)*512+128*128*7168) = 187105280
        # 6*(129280*7168*2+ 3*(7168*18432*3+187105280)+ 58*(187105280+7168*256+7168*2048*9*3))*(512+1024+2048) + 12*(512*512+1024*1024+2048*2048)*61*192*128
        # 6*(129280*7168*2+ 3*(7168*18432*3+187105280)+ 58*(187105280+7168*256+7168*2048*9*3))*(4096+4096+4096) + 12*(4096*4096+4096*4096+4096*4096)*61*192*128
        "expected_flops_tuple": (906535995703296 / 1e12, 3674028304760832 / 1e12),
    },
}


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize(
    "config_type",
    ["llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3"],
)
# [EXPLAIN] `test_flops_counter` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_flops_counter(config_type: str):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_config = CONFIG[config_type]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = Config(test_config["config"])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    flops_counter = FlopsCounter(config)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for batch_seqlens, expected_flops in zip(test_config["batch_seqlens_tuple"], test_config["expected_flops_tuple"]):
        # set delta time to 1 to get the flops
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        counted_flops, _ = flops_counter.estimate_flops(batch_seqlens, 1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}")
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert math.isclose(counted_flops, expected_flops), f"Expect flops for {test_config['config']} is {expected_flops}, but get {counted_flops}"
