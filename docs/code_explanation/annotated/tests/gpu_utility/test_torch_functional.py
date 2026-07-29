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
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flash_attn.bert_padding import unpad_input

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import create_random_mask


# [EXPLAIN] `test_log_probs_from_logits_response_rmpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_log_probs_from_logits_response_rmpad():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.torch_functional import log_probs_from_logits_response, log_probs_from_logits_response_rmpad

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_size = 32000
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt_length = 256
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = 256

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_length + response_length), device="cuda")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = create_random_mask(input_ids=input_ids, max_ratio_of_left_padding=0.2, max_ratio_of_valid_token=0.8, min_ratio_of_valid_token=0.6)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_mask = attention_mask[:, -response_length:]

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(response_mask[:, 0] == 1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.randn(batch_size, prompt_length + response_length, vocab_size, device="cuda")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits_rmpad = unpad_input(logits, attention_mask)[0]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output = log_probs_from_logits_response(input_ids=input_ids, logits=logits, response_length=response_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actual_output = log_probs_from_logits_response_rmpad(input_ids=input_ids, attention_mask=attention_mask, logits_rmpad=logits_rmpad, response_length=response_length)

    # This should bitwise align as only this operation only contains gather operators
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(torch.eq(actual_output * response_mask, expected_output * response_mask))


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
# [EXPLAIN] `test_logprobs_from_logits_v2` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_logprobs_from_logits_v2(dtype):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.torch_functional import logprobs_from_logits_naive, logprobs_from_logits_v2

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_size = 32000
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq_len = 512

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device="cuda")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda", dtype=dtype)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    expected_output = logprobs_from_logits_naive(labels=labels, logits=logits)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actual_output = logprobs_from_logits_v2(labels=labels, logits=logits)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dtype in [torch.float16, torch.bfloat16]:  # float16 falls back to an exactly equivalent method
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.equal(actual_output, expected_output)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:  # small numerical difference when using gather / logsumexp approach
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)


# [EXPLAIN] `test_lr_scheduler` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_lr_scheduler():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch import nn

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = nn.Linear(10, 10)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    constant_lr = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=2)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lr_lst = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _ in range(5):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lr_lst.append(constant_lr.get_last_lr()[0])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        constant_lr.step()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(lr_lst, [0.0, 0.0005, 0.001, 0.001, 0.001])

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.torch_functional import get_cosine_schedule_with_warmup

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cosine_lr = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=2, num_training_steps=5, min_lr_ratio=0.1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lr_lst = []

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _ in range(5):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lr_lst.append(cosine_lr.get_last_lr()[0])
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        cosine_lr.step()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(lr_lst, [0.0, 0.0005, 0.001, 0.0007750000000000002, 0.0003250000000000002])
