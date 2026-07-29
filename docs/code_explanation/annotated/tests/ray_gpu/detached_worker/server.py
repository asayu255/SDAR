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
Server starts a Trainer. Client sends data to the server to train.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["MEGATRON_USE_CUDA_TIMER"] = "0"
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["MEGATRON_START_PROCESS_TIMER"] = "False"
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["NCCL_DEBUG"] = "WARN"

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import ray
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import tensor_parallel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.models.gpt.gpt_model import ModelType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import LlamaConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.llama.megatron import ParallelLlamaForCausalLMRmPadPP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.decorator import Dispatch, register
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.base.megatron.worker import MegatronWorker
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron.optimizer import get_megatron_optimizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import get_model, init_megatron_optim_config, mcore_model_parallel_config


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@ray.remote
# [EXPLAIN] `Trainer` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Trainer(MegatronWorker):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not torch.distributed.is_initialized():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ["LOCAL_RANK"])
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.init_process_group(backend="nccl")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_device(rank)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            tensor_parallel.model_parallel_cuda_manual_seed(10)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [EXPLAIN] `init_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_model(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_model_config = LlamaConfig(
            vocab_size=256,
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        megatron_config = mcore_model_parallel_config(sequence_parallel=True, params_dtype=torch.bfloat16)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.megatron_config = megatron_config

        # [EXPLAIN] `megatron_actor_model_provider` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def megatron_actor_model_provider(pre_process, post_process):
            # vpp is not supported yet because it will hang for some reason. Need debugging
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            parallel_model = ParallelLlamaForCausalLMRmPadPP(
                config=actor_model_config,
                megatron_config=megatron_config,
                pre_process=pre_process,
                post_process=post_process,
            )
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            parallel_model.cuda()
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return parallel_model

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_module = get_model(
            model_provider_func=megatron_actor_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_module = nn.ModuleList(actor_module)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optim_config = OmegaConf.create({"lr": 1e-6, "clip_grad": 1.0})

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        optim_config = init_megatron_optim_config(optim_config)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer_config = optim_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model = actor_module[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.optimizer = actor_optimizer

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # [EXPLAIN] `train_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def train_model(self, data: DataProto) -> DataProto:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = data.batch["input_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = data.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = data.batch["position_ids"]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.optimizer.zero_grad()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.model.zero_grad_buffer(zero_buffer=(not self.optimizer_config.use_distributed_optimizer))  # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        # update for 1 iteration
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        output.mean().backward()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(self.megatron_config, self.megatron_config.timers)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto(batch=TensorDict({"loss": output.detach()}, batch_size=output.shape[0]))


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ray.init(address="auto", namespace="verl")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    resource_pool = RayResourcePool(process_on_nodes=[2], detached=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cls_with_init_args = RayClassWithInitArgs(cls=Trainer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_group = NVMegatronRayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=cls_with_init_args,
        name_prefix="trainer",
        detached=True,
    )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    worker_group.init_model()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    worker_names = worker_group.worker_names
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(worker_names)
