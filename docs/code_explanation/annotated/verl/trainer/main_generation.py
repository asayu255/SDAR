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
Generate responses given a dataset of prompts
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import hydra
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["NCCL_DEBUG"] = "WARN"
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from pprint import pprint

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pandas as pd
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_tokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.hdfs_io import makedirs
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.fsdp_workers import ActorRolloutRefWorker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import is_cuda_available


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@hydra.main(config_path="config", config_name="generation", version_base=None)
# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main(config):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    run_generation(config)


# [EXPLAIN] `run_generation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_generation(config) -> None:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not ray.is_initialized():
        # this is for local ray cluster
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.get(main_task.remote(config))


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote(num_cpus=1)
# [EXPLAIN] `main_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main_task(config):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    OmegaConf.resolve(config)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_path = copy_to_local(config.model.path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trust_remote_code = config.data.get("trust_remote_code", False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if config.rollout.temperature == 0.0:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dataset = pd.read_parquet(config.data.path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    chat_lst = dataset[config.data.prompt_key].tolist()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    chat_lst = [chat.tolist() for chat in chat_lst]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer.padding_side = "left"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tokenizer.pad_token is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tokenizer.pad_token = tokenizer.eos_token

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    wg.init_model()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_samples = len(dataset)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config_batch_size = config.data.batch_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_batch = -(-total_samples // config_batch_size)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_lst = [[] for _ in range(config.data.n_samples)]

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for batch_idx in range(num_batch):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = inputs["input_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = inputs["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = compute_position_id_with_mask(attention_mask)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = DataProto.from_dict(batch_dict)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for n_sample in range(config.data.n_samples):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_padded = wg.generate_sequences(data_padded)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output_texts = []
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(len(output)):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                data_item = output[i]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prompt_length = data_item.batch["prompts"].shape[-1]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                output_texts.append(response_str)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            output_lst[n_sample].extend(output_texts)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_lst = np.array(output_lst, dtype=object)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    dataset["responses"] = output_lst

    # write to a new parquet
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_dir = os.path.dirname(config.data.output_path)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    makedirs(output_dir, exist_ok=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dataset.to_parquet(config.data.output_path)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    main()
