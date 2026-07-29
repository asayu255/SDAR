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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
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
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask


# [EXPLAIN] `SFTDataset` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SFTDataset(Dataset):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_key = config.get("prompt_key", "prompt")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_key = config.get("response_key", "response")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_dict_keys = config.get("response_dict_keys", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_length = config.get("max_length", 1024)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        truncation = config.get("truncation", "error")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_shm = config.get('use_shm', False)

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert truncation in ["error", "left", "right"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.truncation = truncation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_shm = use_shm

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

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_length = max_length

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._download()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._read_files_and_tokenize()

    # [EXPLAIN] `_download` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _download(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, parquet_file in enumerate(self.parquet_files):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    # [EXPLAIN] `_read_files_and_tokenize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _read_files_and_tokenize(self):
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
            # read parquet files and cache
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataframe = pd.read_parquet(parquet_file)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dataframes.append(dataframe)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dataframe = pd.concat(dataframes)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompts = self.dataframe[self.prompt_key]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"self.prompts={self.prompts}")
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(self.prompts, pd.DataFrame):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.prompts = self.prompts.squeeze()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompts = self.prompts.tolist()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.responses = self.dataframe[self.response_key]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in self.response_dict_keys:
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except Exception:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"self.responses={self.responses}")
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(self.responses, pd.DataFrame):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.responses = self.responses.squeeze()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.responses = self.responses.tolist()

    # [EXPLAIN] `__len__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __len__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self.prompts)

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(self, item):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = self.tokenizer

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt = self.prompts[item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response = self.responses[item]

        # apply chat template
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_ids = prompt_ids_output["input_ids"][0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_ids = response_ids_output["input_ids"][0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_attention_mask = response_ids_output["attention_mask"][0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = prompt_ids.shape[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = response_ids.shape[0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sequence_length = input_ids.shape[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if sequence_length < self.max_length:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = torch.cat((input_ids, padded_input_ids))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif sequence_length > self.max_length:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids = input_ids[-self.max_length :]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attention_mask = attention_mask[-self.max_length :]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.truncation == "right":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_ids = input_ids[: self.max_length]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                attention_mask = attention_mask[: self.max_length]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif self.truncation == "error":
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = compute_position_id_with_mask(attention_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loss_mask = attention_mask.clone()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if prompt_length > 1:
            # mask out prompt for SFT.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
