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
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import DistributedDataParallel as LocalDDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.transformer.module import Float16Module
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import all_gather_data_proto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import LLM, vllm_version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import parallel_state as vllm_ps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron_utils import (
    get_model,
    per_tensor_generator,
    unwrap_model,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.memory_buffer import (
    build_memory_buffer,
    build_memory_reference_from_module,
    get_weight_buffer_meta_from_module,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import check_cuda_is_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base import BaseShardingManager

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `AllGatherPPModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class AllGatherPPModel:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, model_provider, use_distributed_optimizer=True) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(
            "[WARNING] This class is deprecated and will no longer be supported. \
Consider using the `MegatronPPOActor` class directly as a replacement."
        )
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._pp_group = mpu.get_pipeline_model_parallel_group()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._model_chunk_size = self._vpp_size or 1

        # each one holds a list of model_chunks in this pp stage
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._pp_models = [None] * self.pp_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rank_list = list(range(self.pp_size))
        # make current rank the last one to initialize
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        rank_list[self.pp_rank], rank_list[-1] = rank_list[-1], rank_list[self.pp_rank]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._this_rank_models = None

        # store the parameter of each pp stage
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory_buffers = [None] * self.pp_size
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for cur_pp_rank in rank_list:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(
                "create pp model",
                f"torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, reserved {torch.cuda.memory_reserved() / 1e9:.4f} GB",
            )
            # since the last initialized rank is the current pp rank, after init, the pp rank is still correct
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cur_pp_rank != self.pp_rank:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                models = get_model(model_provider, wrap_with_ddp=False, use_distributed_optimizer=False)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                models = nn.ModuleList(models)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.pp_models[cur_pp_rank] = models
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # for regular model, we wrapped it with DDP
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                models = get_model(model_provider, wrap_with_ddp=True, use_distributed_optimizer=use_distributed_optimizer)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self._this_rank_models = nn.ModuleList(models)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                self.pp_models[cur_pp_rank] = nn.ModuleList(unwrap_model(models, (torchDDP, LocalDDP)))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._build_param_buffer(cur_pp_rank)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._build_param_references(cur_pp_rank, maintain_weight=cur_pp_rank == self.pp_rank)

            # TODO: after binding to the memory buffer, we can load the checkpoint here
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cur_pp_rank != self.pp_rank:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for model in self.pp_models[cur_pp_rank]:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    model.eval()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._offload_params_to_cpu(cur_pp_rank)

    # [EXPLAIN] `_build_param_buffer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_param_buffer(self, pp_rank):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Build the parameter buffer in each pp rank"""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pp_rank == self._pp_rank:
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.utils.memory_buffer import MemoryBuffer

            # The code here is very hard-coded, based on the following assumptions:
            # 1. `len(_this_rank_models) == 1`
            # 2. `_this_rank_models[0]` is a instance of `DistributedDataParallel` and `use_distributed_optimizer=True`
            # 3. Only bfloat16 data type is used in parameters
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            source = self._this_rank_models[0].buffers[0].param_data
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.memory_buffers[pp_rank] = {torch.bfloat16: MemoryBuffer(source.numel(), source.numel(), torch.bfloat16, source)}
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = self.pp_models[pp_rank]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            weight_buffer_meta = get_weight_buffer_meta_from_module(model)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.memory_buffers[pp_rank] = build_memory_buffer(weight_buffer_meta)

    # [EXPLAIN] `_build_param_references` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_param_references(self, pp_rank, maintain_weight=False):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if pp_rank == self._pp_rank:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = self.pp_models[pp_rank]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        build_memory_reference_from_module(model, self.memory_buffers[pp_rank], maintain_weight=maintain_weight)

    # [EXPLAIN] `_load_params_to_cuda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _load_params_to_cuda(self, pp_rank, to_empty=False):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert pp_rank != self.pp_rank, f"unexpected to load current pp rank [{pp_rank}] back to cuda"
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for buffer in self.memory_buffers[pp_rank].values():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not to_empty:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                buffer.data = buffer.data.to(torch.cuda.current_device(), non_blocking=True)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                buffer.data = torch.empty_like(buffer.data, device="cuda")
        # rebuild reference after loading to CUDA
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._build_param_references(pp_rank)

    # [EXPLAIN] `_offload_params_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _offload_params_to_cpu(self, pp_rank, to_empty=False):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert pp_rank != self.pp_rank, f"unexpected to offload current pp rank [{pp_rank}] to cpu"
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for buffer in self.memory_buffers[pp_rank].values():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not to_empty:
                # offload the whole memory buffer to CPU
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                buffer.data = buffer.data.to("cpu", non_blocking=True)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                buffer.data = torch.empty_like(buffer.data, device="cpu")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self._build_param_references(pp_rank)

    # [EXPLAIN] `load_params_to_cuda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_params_to_cuda(self, to_empty=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """load all model params to cuda"""
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for cur_pp_rank in range(self.pp_size):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cur_pp_rank != self.pp_rank:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._load_params_to_cuda(cur_pp_rank, to_empty=to_empty)

    # [EXPLAIN] `allgather_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def allgather_params(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """allgather params of all pp ranks. Return a list of handles"""
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for cur_pp_rank in range(self.pp_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            global_src = dist.get_global_rank(group=self.pp_group, group_rank=cur_pp_rank)

            # NOTE(sgm): the async op may cause memory leakage of the memory_buffer/pp_models

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for _, param in sorted(self.pp_models[cur_pp_rank].named_parameters()):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                dist.broadcast(tensor=param.data, src=global_src, group=self.pp_group, async_op=False)

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, *inputs, **kwargs):
        # [EXPLAIN] 失敗し得る処理を開始し、後続の例外処理へ制御を接続する。
        try:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prev_output = None
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for cur_chunk_rank in range(self._model_chunk_size):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self._vpp_size:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    mpu.set_virtual_pipeline_model_parallel_rank(cur_chunk_rank)

                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for cur_pp_rank in range(self.pp_size):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(prev_output)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    ret = self.pp_models[cur_pp_rank][cur_chunk_rank](*inputs, **kwargs)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(None)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    prev_output = ret
        # [EXPLAIN] 成功・失敗にかかわらず resource 解放または状態復元を実行する。
        finally:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self._vpp_size:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                mpu.set_virtual_pipeline_model_parallel_rank(0)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            mpu.set_pipeline_model_parallel_rank(self.pp_rank)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ret

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, *inputs, **kwargs):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.forward(*inputs, **kwargs)

    # [EXPLAIN] `eval` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def eval(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model in self.pp_models[self.pp_rank]:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model.eval()

    # [EXPLAIN] `train` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def train(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model in self.pp_models[self.pp_rank]:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model.train()

    # [EXPLAIN] `offload_params_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload_params_to_cpu(self, to_empty=False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """offload params of models that are not of current pp rank to cpu"""
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for cur_pp_rank in range(self.pp_size):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cur_pp_rank != self.pp_rank:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self._offload_params_to_cpu(cur_pp_rank, to_empty=to_empty)

    # [EXPLAIN] `get_all_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_all_params(self):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get all the parameters of the models in all pp ranks

        Returns:
            params: List[List[Dict[str, Tensor]]]: a list of parameters in all pp, where each is a list of dict
                tensors of each model chunk

        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        params = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for pp_rank in range(self.pp_size):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            params.append([])
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for model_chunk_idx in range(len(self.pp_models[pp_rank])):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                params[pp_rank].append({})
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pp_model = self.pp_models[pp_rank][model_chunk_idx]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                pp_model = unwrap_model(pp_model, ((torchDDP, LocalDDP, Float16Module)))  # not use Float16Module
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for name, param in pp_model.named_parameters():
                    # NOTE(gh) workaround: should not get lora params for inference
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if "lora" in name:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        continue
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    params[pp_rank][model_chunk_idx][name] = param

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return params

    # [EXPLAIN] `update_this_rank_models` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_this_rank_models(self, new_models):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._this_rank_models = new_models
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._pp_models[self.pp_rank] = unwrap_model(new_models, (torchDDP, LocalDDP))

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `this_rank_models` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def this_rank_models(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._this_rank_models

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `pp_size` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pp_size(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._pp_size

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `pp_rank` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pp_rank(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._pp_rank

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `pp_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pp_group(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._pp_group

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `pp_models` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def pp_models(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._pp_models


# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank 
   to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
# NOTICE: in new version of vLLM, We need to all-gather all tp rank's model weights
# For code reuse, we directly assign Megatron's TENSOR_MODEL_PARALLEL_GROUP to this
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
_MICRO_DATA_PARALLEL_GROUP = None


# [EXPLAIN] `MegatronVLLMShardingManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronVLLMShardingManager(BaseShardingManager):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @check_cuda_is_available()
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        module: AllGatherPPModel = None,
    ):
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from megatron.core import parallel_state as mpu

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.actor_module = actor_module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inference_engine = inference_engine
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.transformer_config = transformer_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layer_name_mapping = layer_name_mapping
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.weight_converter = weight_converter
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.module = module
        # initialize groups for vllm inference
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rank = torch.distributed.get_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.world_size = torch.distributed.get_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.infer_tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.infer_tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.infer_tp_group = vllm_ps.get_tensor_model_parallel_group()
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vllm_version not in ("0.5.4", "0.6.3"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.infer_tp_group = self.infer_tp_group.device_group
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_ep_size = mpu.get_expert_model_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_ep_rank = mpu.get_expert_model_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_ep_group = mpu.get_expert_model_parallel_group()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_etp_size = mpu.get_expert_tensor_parallel_world_size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_etp_rank = mpu.get_expert_tensor_parallel_rank()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_etp_group = mpu.get_expert_tensor_parallel_group()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.need_tp_reshard = self.train_tp_size != self.infer_tp_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.train_tp_larger = self.train_tp_size > self.infer_tp_size

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            per_tensor_param = per_tensor_generator(self.actor_module, self.model_config, self.weight_converter, self.transformer_config, self.layer_name_mapping, convert_qkv_gate_up_by_simple_split=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.sync_model_weights(per_tensor_param, load_format="megatron")
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # > 0.7.2
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up(tags=["weights"])
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            per_tensor_param = per_tensor_generator(
                self.actor_module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            patch_vllm_moe_model_weight_loader(model)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            loaded_params = model.load_weights(per_tensor_param)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logger.info(info)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up(tags=["kv_cache"])

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, exc_type, exc_value, traceback):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.offload_model_weights()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.sleep(level=1)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for model in self.actor_module:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            model.train()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    # [EXPLAIN] `preprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.infer_tp_size == 1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_gather_data_proto(data, self.infer_tp_group)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    # [EXPLAIN] `postprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.infer_tp_size == 1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data.chunk(chunks=self.infer_tp_size)[self.infer_tp_rank]
