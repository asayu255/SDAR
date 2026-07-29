# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List, Optional, Union

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import datasets
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig, ListConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.utils.data import Dataset
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import PreTrainedTokenizer, ProcessorMixin

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__name__)


# [EXPLAIN] `_left_pad_tensor` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _left_pad_tensor(tensor: torch.Tensor, target_len: int, pad_value) -> torch.Tensor:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_len = target_len - tensor.shape[-1]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if pad_len <= 0:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_shape = (*tensor.shape[:-1], pad_len)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.cat((pad, tensor), dim=-1)


# [EXPLAIN] `collate_fn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def collate_fn(data_list: list[dict]) -> dict:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensors = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_tensors = defaultdict(list)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for data in data_list:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, val in data.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(val, torch.Tensor):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                tensors[key].append(val)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                non_tensors[key].append(val)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_token_ids = [data.get("pad_token_id", 0) if data.get("pad_token_id", 0) is not None else 0 for data in data_list]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in tensors.items():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if key in {"input_ids", "attention_mask", "position_ids"} and len({tensor.shape[-1] for tensor in val}) > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_len = max(tensor.shape[-1] for tensor in val)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            padded = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for tensor, pad_token_id in zip(val, pad_token_ids):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if key == "input_ids":
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    pad_value = int(pad_token_id)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    pad_value = 0
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                padded.append(_left_pad_tensor(tensor, max_len, pad_value))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            val = padded
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensors[key] = torch.stack(val, dim=0)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for key, val in non_tensors.items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        non_tensors[key] = np.array(val, dtype=object)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return {**tensors, **non_tensors}


# [EXPLAIN] `RLHFDataset` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RLHFDataset(Dataset):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(data_files, (List, ListConfig)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data_files = [data_files]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.data_files = copy.deepcopy(data_files)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.processor = processor
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prompt_key = config.get("prompt_key", "prompt")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.image_key = config.get("image_key", "images")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.video_key = config.get("video_key", "videos")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_overrides = self._normalize_task_overrides(config.get("task_overrides", {}))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.return_raw_chat = config.get("return_raw_chat", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.return_full_prompt = config.get("return_full_prompt", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.truncation = config.get("truncation", "error")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_workers = min(self.num_workers, os.cpu_count())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_shm = config.get('use_shm', False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chat_template_func = config.get("chat_template_func", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.filter_prompts = config.get("filter_prompts", True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.serialize_dataset = False
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._download()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._read_files_and_tokenize()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_normalize_task_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _normalize_task_name(task_name):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_name is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_name = str(task_name).lower()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "alfworld" in task_name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "alfworld"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "webshop" in task_name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "webshop"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "search" in task_name:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return "search"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return task_name

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `_normalize_task_overrides` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _normalize_task_overrides(task_overrides):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_overrides is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if OmegaConf.is_config(task_overrides):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_overrides = OmegaConf.to_container(task_overrides, resolve=True)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return dict(task_overrides)

    # [EXPLAIN] `_task_name_from_example` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _task_name_from_example(self, example: dict):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_name = example.get("task_name")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_name is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            env_kwargs = example.get("env_kwargs")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(env_kwargs, dict):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_name = env_kwargs.get("task_name")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._normalize_task_name(task_name)

    # [EXPLAIN] `_task_config_value` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _task_config_value(self, example: dict, key: str, default):
        # [EXPLAIN] row の task_name を正規化し、task-specific override があれば global data config より優先する。
        # [EXPLAIN] これにより AlfWorld/Search/WebShop で max_prompt_length と truncation 方針を変えられる。
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_name = self._task_name_from_example(example)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_name is None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return default
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_cfg = self.task_overrides.get(task_name, {})
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not isinstance(task_cfg, dict):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return default
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return task_cfg.get(key, default)

    # [EXPLAIN] `_max_prompt_length_for_example` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _max_prompt_length_for_example(self, example: dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return int(self._task_config_value(example, "max_prompt_length", self.max_prompt_length))

    # [EXPLAIN] `_truncation_for_example` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _truncation_for_example(self, example: dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._task_config_value(example, "truncation", self.truncation)

    # [EXPLAIN] `_download` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _download(self, use_origin_parquet=False):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.fs import copy_to_local

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i, parquet_file in enumerate(data_files):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    # [EXPLAIN] `_read_files_and_tokenize` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _read_files_and_tokenize(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataframes = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for parquet_file in self.data_files:
            # read parquet files and cache
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dataframes.append(dataframe)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.filter_overlong_prompts:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tokenizer = self.tokenizer
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_key = self.prompt_key
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self._max_prompt_length_for_example(doc),
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than configured max_prompt_length tokens",
            )

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"filter dataset len: {len(self.dataframe)}")

    # [EXPLAIN] `resume_dataset_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def resume_dataset_state(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.serialize_dataset:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._read_files_and_tokenize()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    # [EXPLAIN] `__len__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __len__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return len(self.dataframe)

    # [EXPLAIN] `_build_messages` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_messages(self, example: dict):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages: list = example.pop(self.prompt_key)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.image_key in example or self.video_key in example:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for message in messages:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                content = message["content"]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                content_list = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for segment in re.split("(<image>|<video>)", content):
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if segment == "<image>":
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        content_list.append({"type": "image"})
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif segment == "<video>":
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        content_list.append({"type": "video"})
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        content_list.append({"type": "text", "text": segment})

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                message["content"] = content_list

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return messages

    # [EXPLAIN] `__getitem__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getitem__(self, item):
        # [EXPLAIN] parquet row から message/token Tensor と non-tensor metadata を作り、
        # [EXPLAIN] `task_name` と Search の `env_kwargs` を collate 後の DataProto まで保持する。
        # [EXPLAIN] prompt 長超過時は task-specific 方針に従い left/right truncate または error にする。
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        row_dict: dict = self.dataframe[item]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_prompt_length = self._max_prompt_length_for_example(row_dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        truncation = self._truncation_for_example(row_dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages = self._build_messages(row_dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_inputs = {}

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.processor is not None:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.dataset.vision_utils import process_image, process_video

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            multi_modal_data = {}

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            images = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.image_key in row_dict:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                multi_modal_data["image"] = images

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            videos = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.video_key in row_dict:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                multi_modal_data["video"] = [video.numpy() for video in videos]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = model_inputs.pop("input_ids")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = model_inputs.pop("attention_mask")

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "second_per_grid_ts" in model_inputs:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            row_dict["multi_modal_data"] = multi_modal_data
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = model_inputs.pop("input_ids")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = model_inputs.pop("attention_mask")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.models.transformers.qwen3_vl import get_rope_index
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
                from verl.models.transformers.qwen2_vl import get_rope_index

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_mask = attention_mask[0].bool()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = compute_position_id_with_mask(attention_mask)

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["input_ids"] = input_ids[0]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["attention_mask"] = attention_mask[0]
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["position_ids"] = position_ids[0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(raw_prompt_ids) > max_prompt_length:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if truncation == "left":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif truncation == "right":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[: max_prompt_length]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif truncation == "middle":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                left_half = max_prompt_length // 2
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                right_half = max_prompt_length - left_half
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif truncation == "error":
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")

        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["pad_token_id"] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # encode prompts without chat template
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.return_raw_chat:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.return_full_prompt:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        index = row_dict.get("extra_info", {}).get("index", 0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if need_tools_kwargs and not tools_kwargs:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["index"] = index
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        row_dict["tools_kwargs"] = tools_kwargs
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return row_dict

    # [EXPLAIN] `__getstate__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __getstate__(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.serialize_dataset:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state = self.__dict__.copy()

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "dataframe" in state:
                # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
                del state["dataframe"]
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return state

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.__dict__.copy()
