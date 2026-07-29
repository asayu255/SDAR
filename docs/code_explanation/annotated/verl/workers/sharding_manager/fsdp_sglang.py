# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sglang.srt.entrypoints.engine import Engine
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sglang.srt.utils import MultiprocessingSerializer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import DeviceMesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.tensor import DTensor

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import all_gather_data_proto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import check_cuda_is_available

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# [EXPLAIN] `_preprocess_tensor_for_update_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(tensor, DTensor):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tensor.full_tensor()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tensor


# [EXPLAIN] `FSDPSGLangShardingManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPSGLangShardingManager(BaseShardingManager):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @check_cuda_is_available()
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.module = module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inference_engine = inference_engine
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.offload_param = offload_param

        # Full params
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.full_params = full_params
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if full_params and fsdp_version(self.module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif fsdp_version(self.module) == 1:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tp_size = self.device_mesh["infer_tp"].size()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = torch.cuda.get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_rng_state(self.torch_random_states)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = None

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.module)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        params = self.module.state_dict()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        # Copy, not share memory
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.update_weights(params)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
        del params
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            offload_fsdp_model_to_cpu(self.module)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.torch_random_states = torch.cuda.get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_rng_state(self.gen_random_states)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, exc_type, exc_value, traceback):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.release_memory()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.module.train()

        # add empty cache after each compute
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()

        # restore random states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = torch.cuda.get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.set_rng_state(self.torch_random_states)

    # [EXPLAIN] `update_weights` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_weights(self, params):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        named_tensors = [(k, v) for k, v in params.items()]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        load_format = None
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                gathered_serialized_tensors = None
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )

    # [EXPLAIN] `release_memory` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def release_memory(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.release_memory_occupation()

    # [EXPLAIN] `preprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """All gather across tp group to make each rank has identical input."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.tp_size == 1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group = self.device_mesh["infer_tp"].get_group()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_gather_data_proto(data=data, process_group=group)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # [EXPLAIN] `postprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.tp_size == 1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data.chunk(chunks=self.tp_size)[self.tp_rank]
