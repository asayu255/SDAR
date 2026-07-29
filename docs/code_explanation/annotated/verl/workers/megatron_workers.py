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
The main entry point to run the PPO algorithm
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import warnings

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from codetiming import Timer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.megatron.worker import MegatronWorker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils import hf_tokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.flops_counter import FlopsCounter
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import load_mcore_dist_weights, load_megatron_gptmodel_weights
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.actor.megatron_actor import MegatronPPOActor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.critic.megatron_critic import MegatronPPOCritic
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `set_random_seed` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_random_seed(seed):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import random

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import numpy as np
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import torch

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.manual_seed(seed)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    np.random.seed(seed)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    random.seed(seed)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.cuda.device_count() > 0:
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core import tensor_parallel

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# [EXPLAIN] `ActorRolloutRefWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ActorRolloutRefWorker(MegatronWorker):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config: DictConfig, role: str):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ["LOCAL_RANK"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_device(rank)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.actor.megatron.sequence_parallel:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        set_random_seed(seed=self.config.actor.megatron.seed)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.role = role
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_param = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_grad = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_optimizer = False

        # normalize config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor and self._is_rollout:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.actor.get("ppo_micro_batch_size", None):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self._is_ref:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.ref.get("log_prob_micro_batch_size", None):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and `log_prob_micro_batch_size` should not be None at the same time."
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    # [EXPLAIN] `_build_model_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.models.gpt.gpt_model import ModelType

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.model import get_generation_config, print_model_size

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.generation_config = get_generation_config(self.local_path)

        # [EXPLAIN] `megatron_actor_model_provider` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def megatron_actor_model_provider(pre_process, post_process):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import init_mcore_model

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, value=False, freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            parallel_model.cuda()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return parallel_model

        # Step 3: initialize the megatron model
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor and self._is_rollout:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_module = get_model(
                megatron_actor_model_provider,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"actor_module: {len(actor_module)}")
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.actor.load_weight:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.config.actor.megatron.use_dist_checkpointing:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    load_mcore_dist_weights(actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.rank == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print_model_size(actor_module[0])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self._is_ref:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"self.config.ref.load_weight: {self.config.ref.load_weight}")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ref_module = get_model(
                model_provider_func=megatron_actor_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            # ref_module = nn.ModuleList(ref_module)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.ref.load_weight:  # should align with the actor:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert self.config.actor.load_weight == self.config.ref.load_weight
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("load ref weight start")
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.config.ref.megatron.use_dist_checkpointing:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    load_mcore_dist_weights(ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After ref module init", logger=logger)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            optim_config = init_megatron_optim_config(optim_config)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            optim_config = None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            actor_optimizer = None

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return actor_module, actor_optimizer, self.hf_config, optim_config

    # [EXPLAIN] `_build_rollout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_rollout(self, trust_remote_code=False):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from torch.distributed.device_mesh import init_device_mesh

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.weight",
        }
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.rollout.name == "vllm":
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from torch.distributed.device_mesh import init_device_mesh

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_tp = self.config.rollout.tensor_model_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp = self.world_size // infer_tp
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if vllm_mode == "customized":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rollout = vLLMRollout(
                    actor_module=self.actor_module,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif vllm_mode == "spmd":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rollout = vLLMRollout(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            # perform weight resharding between actor and rollout
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import get_mcore_weight_converter

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.config.rollout.name in ["sglang", "sglang_async"]:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.rollout.name == "sglang_async":
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
            # However, due to verl's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.workers.sharding_manager.megatron_sglang import MegatronSGLangShardingManager

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            infer_tp = self.config.rollout.tensor_model_parallel_size
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp = self.world_size // infer_tp
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            local_path = copy_to_local(self.config.model.path)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)

            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import get_mcore_weight_converter

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            sharding_manager = MegatronSGLangShardingManager(
                actor_module=self.actor.actor_module,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError("Only vllmRollout is supported with Megatron now")

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return rollout, sharding_manager

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import importlib

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            importlib.import_module(self.config.model.external_lib)

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_dtypes import PrecisionType

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            override_transformer_config = OmegaConf.to_container(self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self._is_ref:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            override_transformer_config = OmegaConf.to_container(self.config.ref.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            override_transformer_config = None
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.param_dtype = torch.bfloat16
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            optim_config = self.config.actor.optim if self._is_actor else None
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.actor_module, self.actor_optimizer, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_param:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_megatron_model_to_cpu(self.actor_module)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_optimizer:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_megatron_optimizer(self.actor_optimizer)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_rollout:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.rollout, self.sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After rollout init", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_ref:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After ref model init", logger=logger)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._ref_is_offload_param:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_megatron_model_to_cpu(self.ref_module)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_actor:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.flops_counter = FlopsCounter(self.actor_model_config)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                model_config=self.actor_model_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                tokenizer=self.tokenizer,
                optimizer=self.actor_optimizer,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After init_model finish", logger=logger)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="update_actor", logger=logger)
    # [EXPLAIN] student actor の micro-batch forward、teacher 分布との KL、gradient accumulation、optimizer step を実行する。
    def update_actor(self, data: DataProto):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_actor
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.actor_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_optimizer(self.actor_optimizer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data.batch = data.batch.cuda()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = micro_batch_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataloader = self.actor.make_minibatch_iterator(data=data)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with Timer(name="update_policy", logger=None) as timer:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics = self.actor.update_policy(dataloader=dataloader)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        delta_time = timer.last
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_num_tokens = data.meta_info["global_token_num"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

        # TODO: here, we should return all metrics
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProto(meta_info={"metrics": metrics})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.actor_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_optimizer(self.actor_optimizer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_rollout
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.actor_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After load actor params during generate_sequences", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts.batch = prompts.batch.cuda()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        prompts.meta_info.update(meta_info)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with self.sharding_manager:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_param:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_megatron_model_to_cpu(self.actor_module)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._is_offload_optimizer:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_megatron_optimizer(self.actor_optimizer)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After entering sharding manager", logger=logger)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompts = self.sharding_manager.preprocess_data(prompts)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.rollout.generate_sequences(prompts=prompts)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.sharding_manager.postprocess_data(output)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # clear kv cache
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    # [EXPLAIN] `compute_ref_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_ref_log_prob(self, data: DataProto):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_ref
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._ref_is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = micro_batch_size
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["temperature"] = self.config.rollout.temperature
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(torch.cuda.current_device())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._ref_is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.ref_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    # [EXPLAIN] `compute_log_prob` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_log_prob(self, data: DataProto):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self._is_actor
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        # we should always recompute old_log_probs when it is HybridEngine
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["temperature"] = self.config.rollout.temperature
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(torch.cuda.current_device())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProto.from_dict(tensors={"old_log_probs": output, "entropys": entropys}, meta_info={"temperature": self.config.rollout.temperature})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # clear kv cache
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.actor_module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.actor_module)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.actor_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_optimizer(self.actor_optimizer)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `load_pretrained_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.actor_module)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.barrier()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.actor_module)


# [EXPLAIN] `CriticWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CriticWorker(MegatronWorker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ["LOCAL_RANK"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_device(rank)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.megatron.sequence_parallel:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_param = self.config.megatron.param_offload
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.get("ppo_micro_batch_size", None):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    # [EXPLAIN] `_build_critic_model_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_critic_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.models.gpt.gpt_model import ModelType

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.model import print_model_size

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))

        # [EXPLAIN] `megatron_critic_model_provider` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def megatron_critic_model_provider(pre_process, post_process):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import init_mcore_model

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, share_embeddings_and_output_weights=False, value=True, freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            parallel_model.cuda()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return parallel_model

        # Step 3: initialize the megatron model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_module = get_model(
            model_provider_func=megatron_critic_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.load_weight:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t0 = time.time()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.megatron.use_dist_checkpointing:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                load_mcore_dist_weights(critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                load_megatron_gptmodel_weights(self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            t1 = time.time()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if torch.distributed.get_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"critic load_weight time: {t1 - t0}")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.rank == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optim_config = init_megatron_optim_config(optim_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return critic_module, critic_optimizer, self.hf_config, optim_config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # create critic
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_dtypes import PrecisionType

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import importlib

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            importlib.import_module(self.config.model.external_lib)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_transformer_config = OmegaConf.to_container(self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.param_dtype = torch.bfloat16
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic_module, self.critic_optimizer, self.critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_optimizer(self.critic_optimizer)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.flops_counter = FlopsCounter(self.critic_model_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.critic_model_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            tokenizer=self.tokenizer,
            optimizer=self.critic_optimizer,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] `compute_values` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_values(self, data: DataProto):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = micro_batch_size
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(torch.cuda.current_device())
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.critic_module)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        values = self.critic.compute_values(data=data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProto.from_dict(tensors={"values": values})
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.critic_module)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] `update_critic` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_critic(self, data: DataProto):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(torch.cuda.current_device())

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_optimizer(self.critic_optimizer)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dataloader = self.critic.make_minibatch_iterator(data)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with Timer(name="update_critic", logger=None) as timer:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            metrics = self.critic.update_critic(dataloader=dataloader)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        delta_time = timer.last
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        global_num_tokens = data.meta_info["global_token_num"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = DataProto(batch=None, meta_info={"metrics": metrics})

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_optimizer(self.critic_optimizer)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `load_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.critic_module)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.critic_module)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_optimizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_optimizer(self.critic_optimizer)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `save_checkpoint` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_megatron_model_to_gpu(self.critic_module)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self._is_offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_megatron_model_to_cpu(self.critic_module)


# [EXPLAIN] `RewardModelWorker` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RewardModelWorker(MegatronWorker):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, config):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ["LOCAL_RANK"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_device(rank)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.megatron.sequence_parallel:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.micro_batch_size is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    # [EXPLAIN] `_build_rm_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_rm_model(self, model_path, override_model_config, override_transformer_config):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core.models.gpt.gpt_model import ModelType

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.megatron_utils import get_model

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))

        # [EXPLAIN] `megatron_rm_model_provider` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def megatron_rm_model_provider(pre_process, post_process):
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import init_mcore_model

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parallel_model = init_mcore_model(
                self.tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=False,
                value=True,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            parallel_model.cuda()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return parallel_model

        # Step 3: initialize the megatron model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_model = get_model(
            model_provider_func=megatron_rm_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # reward_model = nn.ModuleList(reward_model)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.load_weight:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.megatron.use_dist_checkpointing:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                load_mcore_dist_weights(reward_model, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                load_megatron_gptmodel_weights(self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True)

        # TODO: add more optimizer args into config
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return reward_model, self.hf_config

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # create critic
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from omegaconf import OmegaConf

        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from verl.utils.torch_dtypes import PrecisionType

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            import importlib

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            importlib.import_module(self.config.model.external_lib)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        override_transformer_config = OmegaConf.to_container(self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_shm = self.config.model.get('use_shm', False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rm_tokenizer = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rm_tokenizer_path is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.param_dtype = torch.bfloat16
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] `compute_rm_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_rm_score(self, data: DataProto):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = data.to(torch.cuda.current_device())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = self.rm.compute_reward(data)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = output.to("cpu")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output
