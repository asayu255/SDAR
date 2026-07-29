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
import inspect
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import logging
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import OrderedDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from peft import PeftModel
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import DeviceMesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

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
from dataclasses import asdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import all_gather_data_proto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import LLM, vllm_version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import parallel_state as vllm_ps
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.device import get_torch_device
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import check_cuda_is_available
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .base import BaseShardingManager

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
logger = logging.getLogger(__file__)
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



# [EXPLAIN] `FSDPVLLMShardingManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FSDPVLLMShardingManager(BaseShardingManager):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @check_cuda_is_available()
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # vLLM > v0.6.3
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device_mesh = device_mesh
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.offload_param = offload_param
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.load_format = load_format
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layered_summon = layered_summon

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
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = get_torch_device().get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().set_rng_state(self.torch_random_states)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base_sync_done: bool = 'dummy' not in load_format
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            VLLMHijack.hijack()

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] `__collect_lora_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def __collect_lora_params()->OrderedDict:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from peft.utils.save_and_load import get_peft_model_state_dict

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lora_params = OrderedDict()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if fsdp_version(self.module) > 0:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.layered_summon:
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if not self.base_sync_done:
                        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora_params = layered_summon_lora_params(self.module)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
                    with FSDP.summon_full_params(self.module, writeback=False):
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if self.base_sync_done:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                        for name, param in lora_params.items()}
                        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                        else:
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            model = self.module._fsdp_wrapped_module.base_model.model
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            model = model.to('cpu')
                            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                            for name, param in model.state_dict().items():
                                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                                if any(x in name for x in ['_flat_param', 'lora_']):
                                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                                    continue
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                            model = model.to(orig_dev)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    torch.cuda.empty_cache()
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.base_sync_done:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    model = self.module._fsdp_wrapped_module.base_model.model
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    model = model.to('cpu')
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for name, param in model.state_dict().items():
                        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                            continue
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        lora_params[name] = param.detach().cpu()
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    model = model.to(orig_dev)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.offload_param:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            load_fsdp_model_to_gpu(self.module)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        peft_config = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            params = __collect_lora_params()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            params = self.module.state_dict()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        load_format = "hf" if self.full_params else "dtensor"

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del params
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up(tags=["weights"])
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up()

            # update model params
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.update_params(params, peft_config=peft_config)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del params
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.offload_param:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                offload_fsdp_model_to_cpu(self.module)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().empty_cache()

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.wake_up(tags=["kv_cache"])

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.torch_random_states = get_torch_device().get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().set_rng_state(self.gen_random_states)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
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

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.module.train()

        # add empty cache after each compute
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        get_torch_device().empty_cache()

        # restore random states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device_mesh is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.gen_random_states = get_torch_device().get_rng_state()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            get_torch_device().set_rng_state(self.torch_random_states)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    # [EXPLAIN] `preprocess_data` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """All gather across tp group to make each rank has identical input."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.tp_size == 1:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            group = vllm_ps.get_tensor_model_parallel_group()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        all_gather_data_proto(data=data, process_group=group)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
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

    # [EXPLAIN] `update_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_params(self, updated_params, peft_config=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = self.model_runner.model
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if peft_config:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.base_sync_done:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lora_int_id=int(time.time_ns() % 0x7FFFFFFF)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] `replace_lora_wrapper` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
                def replace_lora_wrapper(k):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return k.replace(".weight", ".base_layer.weight")
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                        return k.replace(".bias", ".base_layer.bias")
                    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                    return k
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        patch_vllm_moe_model_weight_loader(model)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.base_sync_done = True
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
