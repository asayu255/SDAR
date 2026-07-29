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
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask, create_random_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import log_probs_from_logits_all_rmpad, masked_mean

# TODO(sgm): add more models for test
# we only need one scale for each model
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
test_configs = [
    LlamaConfig(num_hidden_layers=1),
    MistralConfig(num_hidden_layers=1),
    GemmaConfig(num_hidden_layers=1),
    Qwen2Config(num_hidden_layers=1),
]


# [EXPLAIN] `test_hf_casual_models` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_hf_casual_models():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 128
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = 127

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.device("cuda"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = model.to(device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = compute_position_id_with_mask(attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_rmpad = model(input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False).logits  # (1, total_nnz, vocab_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        origin_logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        origin_logits_rmpad, origin_logits_indices, *_ = unpad_input(origin_logits, attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_rmpad = logits_rmpad.squeeze(0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=logits_rmpad,
            indices=indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        origin_log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=origin_logits_rmpad,
            indices=origin_logits_indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(
            masked_mean(log_probs, attention_mask[:, -response_length - 1 : -1]),
            masked_mean(origin_log_probs, attention_mask[:, -response_length - 1 : -1]),
            atol=1e-2,
            rtol=1e-5,
        )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Check pass")


# [EXPLAIN] `test_hf_value_models` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_hf_value_models():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 4
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 128

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config.num_labels = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config.classifier_dropout = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config.hidden_dropout = 0
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.device("cuda"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = AutoModelForTokenClassification.from_config(config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = model.to(device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = compute_position_id_with_mask(attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        origin_logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rmpad_logits = model(input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False).logits  # (1, total_nnz, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rmpad_logits = rmpad_logits.squeeze(0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_logits = pad_input(rmpad_logits, indices, batch_size, seqlen=seqlen)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(
            masked_mean(pad_logits, attention_mask[:, :, None]),
            masked_mean(origin_logits, attention_mask[:, :, None]),
            atol=1e-2,
            rtol=1e-5,
        )
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Value model check pass")


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_hf_casual_models()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_hf_value_models()
