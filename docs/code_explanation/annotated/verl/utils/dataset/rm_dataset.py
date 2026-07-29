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
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import Dataset

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_tokenizer


# [EXPLAIN] `download_files_distributed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def download_files_distributed(download_fn):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import torch.distributed

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.is_initialized():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # download files
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            download_fn()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # download anyway
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        download_fn()


# [EXPLAIN] `RMDataset` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RMDataset(Dataset):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        max_length=1024,
        add_eos=True,
        cache_dir="~/.cache/verl/rm",
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(parquet_files, List):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parquet_files = [parquet_files]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.parquet_files = parquet_files
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cache_dir = os.path.expanduser(cache_dir)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(tokenizer, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tokenizer = hf_tokenizer(tokenizer)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompt_key = prompt_key
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chosen_key = chosen_key
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rejected_key = rejected_key

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.add_eos = add_eos
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_length = max_length

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._download()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._read_files_and_tokenize()

    # [EXPLAIN] `_download` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _download(self):
        # [EXPLAIN] `_download_files` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def _download_files():
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.fs import copy, is_non_local

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            os.makedirs(self.cache_dir, exist_ok=True)
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert os.path.exists(self.cache_dir)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i, parquet_file in enumerate(self.parquet_files):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if is_non_local(parquet_file):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not os.path.exists(dst):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        copy(src=parquet_file, dst=dst)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    self.parquet_files[i] = dst

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        download_files_distributed(_download_files)

    # [EXPLAIN] `_read_files_and_tokenize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _read_files_and_tokenize(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataframes = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataframe = pd.read_parquet(parquet_file)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dataframes.append(dataframe)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dataframe = pd.concat(dataframes)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompts = self.dataframe[self.prompt_key].tolist()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    # [EXPLAIN] `__len__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __len__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self.prompts)

    # [EXPLAIN] `_pad_to_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _pad_to_length(self, input_ids, attention_mask):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        curr_length = input_ids.shape[-1]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if curr_length < self.max_length:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = torch.cat((input_ids, torch.zeros(size=(self.max_length - curr_length,), dtype=input_ids.dtype)), dim=-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.cat((attention_mask, torch.zeros(size=(self.max_length - curr_length,), dtype=attention_mask.dtype)), dim=-1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif curr_length > self.max_length:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = input_ids[: self.max_length]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = attention_mask[: self.max_length]

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return input_ids, attention_mask

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(self, item):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt = self.prompts[item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_response = self.chosen_responses[item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rejected_response = self.rejected_responses[item]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_response_ids = self.tokenizer(chosen_response, return_tensors="pt")["input_ids"][0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rejected_response_ids = self.tokenizer(rejected_response, return_tensors="pt")["input_ids"][0]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.add_eos:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chosen_response_ids = torch.cat((chosen_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rejected_response_ids = torch.cat((rejected_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_input_ids = torch.cat((prompt_ids, chosen_response_ids), dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_attention_mask = torch.ones_like(chosen_input_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rejected_input_ids = torch.cat((prompt_ids, rejected_response_ids), dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_input_ids, chosen_attention_mask = self._pad_to_length(chosen_input_ids, chosen_attention_mask)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rejected_input_ids, rejected_attention_mask = self._pad_to_length(rejected_input_ids, rejected_attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = torch.stack((chosen_input_ids, rejected_input_ids), dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
