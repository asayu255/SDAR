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
from verl.single_controller.base.worker import DistGlobalInfo, DistRankInfo, Worker


# [EXPLAIN] `MegatronWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronWorker(Worker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, cuda_visible_devices=None) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(cuda_visible_devices)

    # [EXPLAIN] `get_megatron_global_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_megatron_global_info(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core import parallel_state as mpu

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp_size = mpu.get_data_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cp_size = mpu.get_context_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return info

    # [EXPLAIN] `get_megatron_rank_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_megatron_rank_info(self):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core import parallel_state as mpu

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp_rank = mpu.get_data_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cp_rank = mpu.get_context_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return info

    # [EXPLAIN] `_init_hf_config_and_tf_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _init_hf_config_and_tf_config(
        self,
        model_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
    ):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from transformers import AutoConfig

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.models.mcore import hf_to_mcore_config
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils import hf_tokenizer
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.fs import copy_to_local
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.local_path = copy_to_local(model_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)

        # Step 2: get the hf
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.architectures = getattr(hf_config, "architectures", None)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"Model config after override: {hf_config}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        # [EXPLAIN] `add_optimization_config_to_tf_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def add_optimization_config_to_tf_config(tf_config):
            # add optimization config to tf_config, e.g. checkpointing
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.model.get("enable_gradient_checkpointing", False):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gradient_checkpointing_cfg = dict(self.config.model.get("gradient_checkpointing_kwargs", dict()))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tf_config.recompute_method = gradient_checkpointing_cfg.get("activations_checkpoint_method", "full")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tf_config.recompute_granularity = gradient_checkpointing_cfg.get("activations_checkpoint_granularity", "full")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                tf_config.recompute_num_layers = gradient_checkpointing_cfg.get("activations_checkpoint_num_layers", -1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if megatron_config := self.config.get("megatron", {}):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if extra := megatron_config.get("extra", {}):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for k, v in extra.items():
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        setattr(tf_config, k, v)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        add_optimization_config_to_tf_config(tf_config)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"TF config: {tf_config}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tf_config = tf_config
