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
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import Dataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_tokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_local_path_from_hdfs


# [EXPLAIN] `MultiTurnSFTDataset` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MultiTurnSFTDataset(Dataset):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = config or {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.truncation = config.get("truncation", "error")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        multiturn_config = config.get("multiturn", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.messages_key = multiturn_config.get("messages_key", "messages")

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.truncation in ["error", "left", "right"]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(parquet_files, List):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parquet_files = [parquet_files]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.parquet_files = parquet_files
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(tokenizer, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tokenizer = hf_tokenizer(tokenizer)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer: PreTrainedTokenizer = tokenizer

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._download()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._read_files_and_process()

    # [EXPLAIN] `_download` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _download(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, parquet_file in enumerate(self.parquet_files):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    # [EXPLAIN] `_read_files_and_process` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _read_files_and_process(self):
        # [EXPLAIN] `series_to_item` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def series_to_item(ls):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import numpy
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import pandas

            # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                ls = ls[0]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ls

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataframes = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for parquet_file in self.parquet_files:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataframe = pd.read_parquet(parquet_file)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dataframes.append(dataframe)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    # [EXPLAIN] `__len__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __len__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self.messages)

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(self, item):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = self.tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages = self.messages[item]

        # First, get the full conversation tokens
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = full_tokens[0]  # The output is already a tensor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask by identifying assistant responses
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant responses
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_messages = messages[: i + 1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

            # Get tokens for messages up to previous point
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

            # Calculate start and end positions
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            end_pos = prefix_tokens[0].shape[0]

            # If this is an assistant message, set loss mask
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if msg["role"] == "assistant":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sequence_length = input_ids.shape[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sequence_length < self.max_length:
            # Pad sequences
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = torch.cat((input_ids, padded_input_ids))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif sequence_length > self.max_length:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.truncation == "left":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids = input_ids[-self.max_length :]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attention_mask = attention_mask[-self.max_length :]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_mask = loss_mask[-self.max_length :]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.truncation == "right":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids = input_ids[: self.max_length]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attention_mask = attention_mask[: self.max_length]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_mask = loss_mask[: self.max_length]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.truncation == "error":
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = position_ids * attention_mask

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
