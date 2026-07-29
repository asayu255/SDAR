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
This script is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python scripts/model_merger.py merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import argparse
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import re
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from abc import ABC, abstractmethod
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from concurrent.futures import ThreadPoolExecutor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass, field
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pathlib import Path
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from accelerate import init_empty_weights
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from safetensors.torch import load_file
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed._tensor import Placement, Shard
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PretrainedConfig,
)

# [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
try:
    # for torch 2.5+
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.tensor import DTensor
# [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
except ImportError:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed._tensor import DTensor

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tqdm import tqdm

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_processor, hf_tokenizer


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `ModelMergerConfig` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ModelMergerConfig:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    operation: str  # 'merge' or 'test'
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    backend: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    local_dir: str
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    hf_model_config_path: str
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target_dir: Optional[str] = "tmp"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_upload_path: Optional[str] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    private: bool = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_hf_dir: Optional[str] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tie_word_embedding: bool = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    is_value_model: bool = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_model_path: Optional[str] = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_upload: bool = field(init=False)

    # [EXPLAIN] `__post_init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __post_init__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.operation == "test":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.target_dir = None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hf_upload_path = None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.private = False


# [EXPLAIN] `BaseModelMerger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BaseModelMerger(ABC):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: ModelMergerConfig):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_model_config_path = config.hf_model_config_path

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if config.hf_model_path:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("Warning: --hf_model_path is deprecated and will be removed in a future version. Currently verl will save huggingface model configuration files into checkpoint directories. Therefore, there is no need to provide --hf_model_path. ")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hf_model_config_path = config.hf_model_path

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = AutoConfig.from_pretrained(self.hf_model_config_path)

    # [EXPLAIN] `get_transformers_auto_model_class` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_transformers_auto_model_class(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "ForTokenClassification" in self.model_config.architectures[0]:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return AutoModelForTokenClassification
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "ForCausalLM" in self.model_config.architectures[0]:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return AutoModelForCausalLM
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "ForConditionalGeneration" in self.model_config.architectures[0]:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return AutoModelForVision2Seq

        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    # [EXPLAIN] `patch_model_generation_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def patch_model_generation_config(self, model):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if model.can_generate():
            # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
            try:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            # [EXPLAIN] 発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。
            except OSError:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"Warning: Generation config file not found in {self.hf_model_config_path}, using a generation config created from the model config.")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model

    # [EXPLAIN] `save_hf_model_and_tokenizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_model_class = self.get_transformers_auto_model_class()
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with init_empty_weights():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = auto_model_class.from_config(self.model_config, torch_dtype=torch.bfloat16)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.to_empty(device="cpu")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = self.patch_model_generation_config(model)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Saving model to {self.config.target_dir}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del state_dict
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del model

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        processor = hf_processor(self.hf_model_config_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer = hf_tokenizer(self.hf_model_config_path)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if processor is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saving processor to {self.config.target_dir}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            processor.save_pretrained(self.config.target_dir)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if tokenizer is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Saving tokenizer to {self.config.target_dir}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tokenizer.save_pretrained(self.config.target_dir)

    # [EXPLAIN] `upload_to_huggingface` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def upload_to_huggingface(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from huggingface_hub import HfApi

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        api = HfApi()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @abstractmethod
    # [EXPLAIN] `merge_and_save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def merge_and_save(self):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Subclasses should implement this method")


# [EXPLAIN] `FSDPModelMerger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPModelMerger(BaseModelMerger):
    # [EXPLAIN] `_get_world_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_world_size(self) -> int:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Extracts the FSDP world_size from checkpoint filenames (e.g., 'model_world_size_8_rank_0.pt')."""
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for filename in os.listdir(self.config.local_dir):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if match:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return int(match.group(1))
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise FileNotFoundError(f"Could not determine world size. No file matching 'model_world_size_(\d+)_rank_0.pt' found in {self.config.local_dir}")

    # [EXPLAIN] `_load_rank_zero_state_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.load(Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt", map_location="cpu", weights_only=False)

    # [EXPLAIN] `_extract_device_mesh_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pivot_key = sorted(list(state_dict.keys()))[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight = state_dict[pivot_key]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(weight, DTensor):
            # get sharding info
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            device_mesh = weight.device_mesh
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh = device_mesh.mesh
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh_dim_names = device_mesh.mesh_dim_names
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # for non-DTensor
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh = np.array([world_size], dtype=np.int64)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh_dim_names = ("fsdp",)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return mesh, mesh_dim_names

    # [EXPLAIN] `_calculate_shard_configuration` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _calculate_shard_configuration(self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Calculates the total number of shards and the shape of the device mesh."""
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_shards = mesh.shape[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mesh_shape = (mesh.shape[-1],)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return total_shards, mesh_shape

    # [EXPLAIN] `_merge_by_placement` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Merges a list of tensors based on their DTensor placement"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if placement.is_replicate():
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return tensors[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif placement.is_partial():
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError("Partial placement is not supported yet")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif placement.is_shard():
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.cat(tensors, dim=placement.dim).contiguous()

        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Unsupported placement: {placement}")

    # [EXPLAIN] `_load_and_merge_state_dicts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _load_and_merge_state_dicts(self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_state_dict_lst = [None] * total_shards

        # [EXPLAIN] `process_one_shard` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def process_one_shard(rank: int, model_state_dict_lst: list):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_state_dict_lst[rank] = state_dict
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return state_dict

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                future.result()

        # Merge state dicts from all shards
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_dict = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        param_placements: dict[str, list] = {}

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in set(model_state_dict_lst[0].keys()):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict[key] = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tensor = model_state_shard.pop(key)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(tensor, DTensor):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        placements = placements[1:]

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if key not in param_placements:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        param_placements[key] = placements
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                        assert param_placements[key] == placements
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    state_dict[key].append(tensor.bfloat16())

        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del model_state_dict_lst

        # Merge tensors
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in sorted(state_dict):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not isinstance(state_dict[key], list):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"No need to merge key {key}")
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if key in param_placements:
                # merge shards
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                placements: tuple[Shard] = param_placements[key]
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert len(placements) == 1
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    shards = state_dict[key]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # 2-D list, FSDP + TP
                    # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                    raise NotImplementedError("FSDP + TP is not supported yet")
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return state_dict

    # [EXPLAIN] `merge_and_save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def merge_and_save(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = self._get_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.operation == "test":
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not self.config.test_hf_dir:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("test_hf_dir must be provided for test operation")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._test_state_dict(merged_state_dict)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.operation == "merge":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.save_hf_model_and_tokenizer(merged_state_dict)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.hf_upload:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.upload_to_huggingface()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unknown operation: {self.config.operation}")

    # [EXPLAIN] `_test_state_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        auto_model_class = self.get_transformers_auto_model_class()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hf_state_dict = hf_model.state_dict()
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del hf_model

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hf_model_keys = set(hf_state_dict.keys())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        collected_keys = set(state_dict.keys())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        missing_keys = hf_model_keys - collected_keys
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_keys = collected_keys - hf_model_keys
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key in hf_model_keys:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hf_shape = hf_state_dict[key].shape
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            collected_shape = state_dict[key].shape
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert hf_shape == collected_shape, f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hf_dtype = hf_state_dict[key].dtype
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            collected_dtype = state_dict[key].dtype
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert hf_dtype == collected_dtype, f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")


# [EXPLAIN] `MegatronModelMerger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronModelMerger(BaseModelMerger):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: ModelMergerConfig):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron_utils import get_hf_config_and_tokenizer_checkpoint_path

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config.hf_model_config_path = get_hf_config_and_tokenizer_checkpoint_path(config.local_dir)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(config)

    # [EXPLAIN] `_get_tp_pp_rank_from_sharded_dir` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_tp_pp_rank_from_sharded_dir(self, sharded_dir: str) -> tuple[int, int]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        match = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert match, f"Invalid sharded dir {sharded_dir}"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = int(match.group(1))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_rank = int(match.group(2))
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tp_rank, pp_rank

    # [EXPLAIN] `_check_megatron_checkpoint_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _check_megatron_checkpoint_path(self, model_path: str) -> tuple[list[str], int, int]:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Validates the Megatron checkpoint structure (presence of 'model.pt' in sharded directories).
        Determines TP and PP sizes from directory names.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_size = 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharded_dirs = sorted(os.listdir(model_path))
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for sharded_dir in sharded_dirs:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert "model.pt" in os.listdir(Path(model_path) / sharded_dir), f"model.pt not found in {sharded_dir}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(sharded_dir)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tp_size = max(tp_size, tp_rank + 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            pp_size = max(pp_size, pp_rank + 1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return sharded_dirs, tp_size, pp_size

    # [EXPLAIN] `_merge_across_tp` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_across_tp(self, key: str, tp_data: list[torch.Tensor], config: PretrainedConfig, tp_size: int, is_value_model: bool = False) -> torch.Tensor | list[torch.Tensor]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "linear_fc1.weight" in key:
            # if the tensor is gate and proj
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate_lst = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            up_lst = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for infer_param in tp_data:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gate, up = infer_param.chunk(2)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                gate_lst.append(gate)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                up_lst.append(up)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gate = torch.cat(gate_lst, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            up = torch.cat(up_lst, dim=0)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [gate, up]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q_lst = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k_lst = []
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            v_lst = []
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert config.num_attention_heads % config.num_key_value_heads == 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert tp_data[0].shape[0] % (num_q_per_kv + 2) == 0
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kv_size_per_tp = tp_data[0].shape[0] // (num_q_per_kv + 2)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for infer_param in tp_data:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                num_query_groups_per_partition = config.num_key_value_heads // tp_size
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                    ]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    q, k, v = chunk.split(split_size)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    q_lst.append(q)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    k_lst.append(k)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    v_lst.append(v)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            q = torch.cat(q_lst, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            k = torch.cat(k_lst, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            v = torch.cat(v_lst, dim=0)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return [q, k, v]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif "layer_norm" in key or "layernorm" in key or "output_layer" in key and is_value_model:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return tp_data[0]
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dim = 0
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "linear_fc2.weight" in key or "self_attention.linear_proj" in key:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dim = 1
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.cat(tp_data, dim=dim)

    # [EXPLAIN] `_load_state_dicts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _load_state_dicts(self, model_ckpt_path: str, sharded_dirs: list[str], tp_size: int, pp_size: int) -> list[list[dict]]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_state_dict_lst = [[None for _ in range(tp_size)] for _ in range(pp_size)]

        # [EXPLAIN] `_process_one_megatron_shard` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def _process_one_megatron_shard(sharded_dir: str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_file_path = Path(model_ckpt_path) / sharded_dir / "model.pt"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_dict = torch.load(model_file_path, map_location="cpu", weights_only=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(sharded_dir)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model_state_dict_lst[pp_rank][tp_rank] = state_dict

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            futures = [executor.submit(_process_one_megatron_shard, sharded_dir) for sharded_dir in sharded_dirs]
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for future in tqdm(futures, desc=f"Loading {len(sharded_dirs)} Megatron shards", total=len(sharded_dirs)):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                future.result()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return model_state_dict_lst

    # [EXPLAIN] `_merge_state_dicts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _merge_state_dicts(self, model_state_dict_lst: list[list[dict]], tp_size: int, pp_size: int) -> dict[str, torch.Tensor]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_dict = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vpp_size = len(model_state_dict_lst[0][0])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layers_cum = 0

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for vpp_rank in range(vpp_size):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for pp_rank in range(pp_size):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                layers_handled = 0
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                keys = model_state_dict_lst[pp_rank][0][vpp_rank].keys()
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key in keys:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if "extra_state" in key:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        continue
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if self.config.tie_word_embedding and ("output_layer" in key):
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print("skip lm_head and reward_head loading because of tie_word_embeddings")
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        continue

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    new_key = key
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if "decoder.layers." in key:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        local_layer_no = int(key.split(".")[2])
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        layers_handled = max(local_layer_no, layers_handled)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        global_layer_no = local_layer_no + layers_cum
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_key_list = key.split(".")
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_key_list[2] = str(global_layer_no)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_key = ".".join(new_key_list)

                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    tp_data = [model_state_dict_lst[pp_rank][tp_rank][vpp_rank][key] for tp_rank in range(tp_size)]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    merged = self._merge_across_tp(new_key, tp_data, self.model_config, tp_size, self.config.is_value_model)

                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not isinstance(merged, list):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        state_dict[new_key] = merged
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif len(merged) == 3:
                        # split qkv
                        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                        for n, d in zip(["q", "k", "v"], merged):
                            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                            state_dict[new_key.replace("linear_qkv", f"linear_{n}")] = d
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif len(merged) == 2:
                        # split gate up
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        state_dict[new_key.replace("linear_fc1", "gate_proj")] = merged[0]
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        state_dict[new_key.replace("linear_fc1", "up_proj")] = merged[1]

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                layers_cum += layers_handled + 1  # zero based

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return state_dict

    # [EXPLAIN] `merge_and_save` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def merge_and_save(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron_utils import get_model_checkpoint_path

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_ckpt_path = get_model_checkpoint_path(self.config.local_dir)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sharded_dirs, tp_size, pp_size = self._check_megatron_checkpoint_path(model_ckpt_path)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"sharded_dirs: {sharded_dirs}, tp_size: {tp_size}, pp_size: {pp_size}, mp_size: {len(sharded_dirs)}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_state_dict_lst = self._load_state_dicts(model_ckpt_path, sharded_dirs, tp_size, pp_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merged_state_dict = self._merge_state_dicts(model_state_dict_lst, tp_size, pp_size)
        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del model_state_dict_lst

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.operation == "test":
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not self.config.test_hf_dir:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError("test_hf_dir must be provided for test operation")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._test_state_dict(merged_state_dict)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.operation == "merge":
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.save_hf_model_and_tokenizer(merged_state_dict)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.hf_upload:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.upload_to_huggingface()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Unknown operation: {self.config.operation}")

    # [EXPLAIN] `_test_state_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Compares the merged Megatron state_dict against a reference safetensors model.
        Applies necessary name mappings from Megatron to Hugging Face conventions using _replace_name.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ref_state_dict = load_file(Path(self.config.test_hf_dir) / "model.safetensors")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        params_mapping = [
            # (megatron core gpt model name, vllm model name)
            ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
            ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
            ("embedding.word_embeddings", "model.embed_tokens"),
            ("self_attention.linear_qkv", "self_attn.qkv_proj"),
            ("self_attention.linear_proj", "self_attn.o_proj"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
            ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
            ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
            ("mlp.linear_fc1", "mlp.gate_up_proj"),
            ("mlp.linear_fc2", "mlp.down_proj"),
            ("decoder.final_layernorm", "model.norm"),
            ("output_layer", "lm_head"),
            ("self_attention.linear_q", "self_attn.q_proj"),
            ("self_attention.linear_k", "self_attn.k_proj"),
            ("self_attention.linear_v", "self_attn.v_proj"),
        ]

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for original_name, loaded_weight in state_dict.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            name = self._replace_name(original_name, params_mapping)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not name or name.endswith(".bias") and name not in ref_state_dict:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "rotary_emb.inv_freq" in name:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.tie_word_embedding and "lm_head.weight" in name:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if name not in ref_state_dict:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise RuntimeError(f"key: {name} not exist in state_dict")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            param = ref_state_dict[name]
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert loaded_weight.dtype == param.dtype
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(loaded_weight, param, atol=1e-2, rtol=5e-2)

    # [EXPLAIN] `_replace_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _replace_name(self, megatron_name: str, name_mapping: list[tuple[str, str]]) -> str:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for m_name, v_name in name_mapping:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if m_name not in megatron_name:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "layers" in megatron_name:  # deal with decoder layers
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                megatron_name = megatron_name.replace("decoder", "model")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                megatron_name_list = megatron_name.split(".")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param_name_list = megatron_name_list[:3]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    param_name_list.append(v_name)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param_name = ".".join(param_name_list)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param_name_list = megatron_name_list[:3]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    weight_or_bias = megatron_name_list[-1]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    param_name_list.append(v_name)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    param_name_list.append(weight_or_bias)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param_name = ".".join(param_name_list)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return param_name
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                param_name = megatron_name.replace(m_name, v_name)
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return param_name
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None  # Return None if no mapping found


# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    parser = argparse.ArgumentParser(description="verl model merger")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    base_op_parser = argparse.ArgumentParser(add_help=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_op_parser.add_argument("--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_op_parser.add_argument("--local_dir", type=str, required=True, help="Path to the saved model checkpoints")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_op_parser.add_argument("--hf_model_path", type=str, default=None, help="(Deprecated) Path to the original Hugging Face model for config.")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_op_parser.add_argument("--tie-word-embedding", action="store_true", help="Whether to tie word embedding weights (currently only Megatron supported)")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    base_op_parser.add_argument("--is-value-model", action="store_true", help="Whether the model is a value model (currently only Megatron supported)")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    merge_parser.add_argument("--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    merge_parser.add_argument("--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    merge_parser.add_argument("--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    test_parser = subparsers.add_parser("test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_parser.add_argument("--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    args = parser.parse_args()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_path": args.hf_model_path,
        "hf_model_config_path": args.local_dir,
    }

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if args.operation == "merge":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            private=args.private,
            test_hf_dir=None,
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(config.target_dir, exist_ok=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif args.operation == "test":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,
            # the following args are not used by test operation
            target_dir=None,
            hf_upload_path=None,
            private=False,
        )
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Unknown operation: {args.operation}")

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.backend == "fsdp":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merger = FSDPModelMerger(config)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif config.backend == "megatron":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        merger = MegatronModelMerger(config)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    merger.merge_and_save()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
