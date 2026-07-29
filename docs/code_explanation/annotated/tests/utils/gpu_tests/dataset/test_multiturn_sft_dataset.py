# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Test the MultiTurnSFTDataset implementation
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


# [EXPLAIN] `test_multiturn_sft_dataset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_multiturn_sft_dataset():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Starting test...")
    # Create a temporary parquet file with test data
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_data = {
        "messages": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And what is 4+4?"},
                {"role": "assistant", "content": "4+4 equals 8."},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"},
            ],
        ]
    }

    # Create test directory if it doesn't exist
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs("test_data", exist_ok=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_file = "test_data/test.parquet"

    # Save test data to parquet
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    df = pd.DataFrame(test_data)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    df.to_parquet(test_file)

    # Initialize tokenizer and dataset
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = {"max_length": 512, "truncation": "error", "multiturn": {"messages_key": "messages"}}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, config=config)

    # Test 1: Dataset Length
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    # Get items for testing
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    item0 = dataset[0]  # Math conversation
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    item1 = dataset[1]  # Joke conversation

    # Test 2: Required Keys and Types
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    required_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key in required_keys:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert key in item0, f"Missing key {key} in dataset item"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key}"
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert item0[key].dtype == torch.long, f"Expected torch.long for {key}, got {item0[key].dtype}"

    # Test 3: Shape Consistency
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert item0["loss_mask"].shape == item0["input_ids"].shape, "Loss mask shape doesn't match input_ids shape"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert item0["attention_mask"].shape == item0["input_ids"].shape, "Attention mask shape doesn't match input_ids shape"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert item0["position_ids"].shape == item0["input_ids"].shape, "Position IDs shape doesn't match input_ids shape"

    # Test 4: Loss Mask Pattern - Math Conversation
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss_mask0 = item0["loss_mask"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids0 = item0["input_ids"]

    # Find assistant response positions
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    assistant_positions0 = torch.where(loss_mask0 == 1)[0]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(assistant_positions0) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    assistant_text0 = tokenizer.decode(input_ids0[loss_mask0 == 1])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Math conversation assistant text: {assistant_text0}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "2+2 equals 4" in assistant_text0, "First assistant response not found"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "4+4 equals 8" in assistant_text0, "Second assistant response not found"

    # Test 5: Loss Mask Pattern - Joke Conversation
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    loss_mask1 = item1["loss_mask"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids1 = item1["input_ids"]

    # Find assistant response positions
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    assistant_positions1 = torch.where(loss_mask1 == 1)[0]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert len(assistant_positions1) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    assistant_text1 = tokenizer.decode(input_ids1[loss_mask1 == 1])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Joke conversation assistant text: {assistant_text1}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "chicken cross the road" in assistant_text1, "First assistant response not found"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert "other side" in assistant_text1, "Second assistant response not found"

    # Test 6: Attention Mask Pattern
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask0 = item0["attention_mask"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sequence_length = torch.sum(attention_mask0)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert sequence_length > 0, "No tokens marked as attended in attention mask"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(attention_mask0[:sequence_length] == 1), "Incorrect attention mask pattern"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sequence_length < len(attention_mask0):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.all(attention_mask0[sequence_length:] == 0), "Padding not properly masked"

    # Test 7: Position IDs Pattern
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids0 = item0["position_ids"]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.equal(position_ids0[:sequence_length], torch.arange(sequence_length)), "Position IDs not sequential for non-padded tokens"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if sequence_length < len(position_ids0):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert torch.all(position_ids0[sequence_length:] == 0), "Padding position IDs not zero"

    # Test 8: Verify loss mask for assistant responses
    # Get the full conversation text
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_text = tokenizer.decode(input_ids0)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"\nFull conversation text:\n{full_text}")

    # Get the assistant responses
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 1])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"\nAssistant responses (from loss mask):\n{assistant_text}")

    # Verify that loss mask is set for all assistant responses
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for msg in test_data["messages"][0]:  # First conversation
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if msg["role"] == "assistant":
            # The content should appear in the masked text
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert msg["content"] in assistant_text, f"Assistant message '{msg['content']}' not found in masked text"

            # The content should NOT appear in the non-masked text
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert msg["content"] not in non_assistant_text, f"Assistant message '{msg['content']}' found in non-assistant text"

    # Test 9: Verify non-assistant parts have loss_mask=0
    # Get non-assistant text
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"\nNon-assistant text (from loss mask):\n{non_assistant_text}")

    # Verify that system and user messages are in the non-assistant text
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for msg in test_data["messages"][0]:  # First conversation
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if msg["role"] in ["system", "user"]:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert msg["content"] in non_assistant_text, f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"

            # And verify they're NOT in the assistant text
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert msg["content"] not in assistant_text, f"{msg['role'].title()} message '{msg['content']}' found in assistant text"

    # Test 10: Verify padding behavior
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padding_config = {"max_length": 1024, "truncation": "error", "multiturn": {"messages_key": "messages"}}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    small_dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, config=padding_config)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    padded_item = small_dataset[0]

    # Get actual sequence length (before padding)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actual_length = torch.sum(padded_item["attention_mask"])

    # Verify padding tokens
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(padded_item["input_ids"][actual_length:] == tokenizer.pad_token_id), "Padding tokens not set correctly"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(padded_item["attention_mask"][actual_length:] == 0), "Attention mask not set correctly for padding"
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.all(padded_item["loss_mask"][actual_length:] == 0), "Loss mask not set correctly for padding"

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("All tests passed!")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Starting test...")
