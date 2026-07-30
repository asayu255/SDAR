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
"""
The main entry point to run the PPO algorithm
"""

# 【このファイルの役割（日本語補足）】
# - Ray上の1 GPU workerプロセスで、FSDP/FSDP2モデルの生成・保持・offload・実行を管理する。
# - ray_trainer.pyからRPCで呼ばれ、dp_actor.py / dp_critic.pyへ実際のforward・backwardを委譲する
# 「分散実行とモデルライフサイクルの境界層」である。
# - ActorRolloutRefWorkerはroleに応じて、次の機能を同じworkerクラスへ組み合わせる。
# actor   : DataParallelPPOActor.update_policy()を呼び、学習更新を行う。
# rollout : HF/vLLM/SGLangで応答を生成し、FSDP重みを推論engineへ同期する。
# ref     : 更新しないreference/teacher policyとしてlog-probまたはtop-k分布を計算する。
# - CriticWorkerはGAE用value model、RewardModelWorkerはmodel-based reward、
# AsyncActorRolloutRefWorkerはasync vLLM RPCを担当する。
# - dp_actor.py内のloss.backward()で起動するFSDP勾配通信は、このファイルでmodelを
# FSDP/FSDP2 wrapしたことによってautograd hookへ登録される。
# - 重要な通信は4種類あり、混同しないこと。
# (1) FSDP parameter all-gather / gradient reduce-scatter
# (2) Ulysses sequence-parallelのall-to-all / all-gather
# (3) FSDP actor重みからvLLM/SGLangへの推論weight同期
# (4) Ray driverとworker間のDataProto RPC転送
# 【import群の読み方】
# - torch.distributed / FSDP: rank間process group、parameter shard、gradient通信。
# - Dispatch / register: RayWorkerGroupからメソッドをどう各rankへ配送するかを指定する。
# - FSDPUlyssesShardingManager: DP配置のDataProtoをSP group向けにall-gatherし、終了後に再分割する。
# - fsdp_utils: FSDP1/FSDP2のwrap、state_dict load、CPU offloadを共通化する。
# - DataProto: tensor batch・non-tensor batch・meta_infoを一体でworker間へ運ぶ。
import logging
import os
import warnings
from typing import Union

# process memory使用量をmetricsへ記録する。
import psutil
import torch
import torch.distributed
# worker内のactor/critic更新区間を計時する。
from codetiming import Timer
# Hydra/OmegaConf設定を読み、struct mode内の限定的な追記を行う。
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl.utils.py_functional import convert_to_regular_types
from verl import DataProto
# model forwardをremove-padding/Ulysses/fused kernel対応へ差し替える。
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
# Ray RPCの配送方式をdecoratorで宣言する。
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
# FSDP shardを含むmodel/optimizer/scheduler checkpointを統合管理する。
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
# FSDP1/FSDP2差分とmanual/native offloadを吸収するhelper群。
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    layered_summon_lora_params,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
# FSDP data-parallel配置とUlysses sequence-parallel配置を接続するmanager。
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available


from peft import LoraConfig, TaskType, get_peft_model
from codetiming import Timer

# LoRA adapter保存時のrank/barrier処理に使うdistributed alias。
import torch.distributed as dist
# FSDP1型判定、constructor、LoRA parameter集約に使用する。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
from dataclasses import asdict
import json


# loggerは各Ray workerプロセスごとに生成される。
# VERL_LOGGING_LEVELを環境変数で上書きでき、GPU memory計測などの出力レベルを統一する。
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# device_nameは"cuda" / "npu"等を抽象化した名称。
# autocastやdevice meshのbackend選択で同じ値を再利用する。
device_name = get_device_name()


# world_size個のrankを、FSDP parameter shard用device meshへ配置する。
# - fsdp_size < 0 または fsdp_size >= world_size:
# 1次元mesh (world_size,) を作り、全rankで1つのFULL_SHARD groupを構成する。
# - 0 < fsdp_size < world_size:
# 2次元mesh (world_size/fsdp_size, fsdp_size) を作る。
# 内側"fsdp"次元でparameterをshardし、外側"ddp"次元で同じshardをreplicateする
# HYBRID_SHARD構成になる。
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


# device meshの次元数をFSDP1のShardingStrategyへ対応付ける。
# FULL_SHARDはparameter・gradient・optimizer stateを全rankへ分割する。
# HYBRID_SHARDはnode内等でFULL_SHARDし、group間ではreplicateする構成である。
def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


# ===============================================================================
# ActorRolloutRefWorker
# ===============================================================================
# main_ppo.pyがrole_worker_mappingへ登録し、RayPPOTrainer.init_workers()が生成する主要worker。
# 呼び出し関係:
# RayPPOTrainer / SkillSDRayTrainer
# -> RayWorkerGroup.actor_rollout_* / ref_*
# -> ActorRolloutRefWorker
# -> DataParallelPPOActor (dp_actor.py)
# -> rollout engine (HF / vLLM / SGLang)
# 同じclassをactor・rollout・refへ再利用することで、同一GPU上へ機能をcolocateできる。
class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    # 初期化ではまだHugging Face model本体を生成しない。
    # ここではprocess group、FSDP/SP mesh、role flag、offload設定、per-rank batch sizeを確定する。
    # 実モデル構築はRayPPOTrainer.init_workers()から呼ばれるinit_model()で行う。
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        # Rayが各GPU workerへRANK/WORLD_SIZEを環境変数として渡す。
        # 未初期化時だけprocess groupを作り、CPU collectiveにはGloo、GPU collectiveにはNCCL
        # （NPUではHCCL）を利用する複合backendを指定する。
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", rank=rank, world_size=world_size)

        # build device mesh for FSDP
        # FSDP meshはparameter/gradientの分散範囲を定める。
        # このmeshをFSDP constructorまたはFSDP2 fully_shardへ渡すことで、
        # dp_actor.pyのbackward時にどのrank間でreduce-scatterするかが決まる。
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        # Ulysses用meshはFSDP meshとは用途が異なる。
        # shape=(dp, sp)の"sp"次元で1 sampleのsequenceを分割し、"dp"次元には別sampleを配置する。
        # sp_size=1ならmanagerは実質no-opになる。
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # LoRA有効時はactor modelへPEFT adapterを装着する。
        # reference log-probは後段でadapterを一時無効化し、base modelをrefとして再利用できる。
        self._lora_rank = self.config.model.get('lora_rank', 0)
        self._is_lora = self._lora_rank > 0

        # role文字列を3つのbooleanへ展開する。
        # actor_rolloutは学習modelと推論engineを同じworkerへ同居させ、
        # actor_rollout_refはさらにreference機能も同居させる。
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # manual offload flag。
        # param_offloadはmodel shard、optimizer_offloadはoptimizer stateを更新区間外にCPUへ移す。
        # FSDP2 native offload_policyを使う場合は後でこのmanual flagをFalseへ切り替える。
        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        # configに指定されたbatch sizeはglobal値を含み得るため、このworkerが担当するDP sample数へ正規化する。
        # sequence-parallel rankは同じsampleの異なるsequence断片を処理するので、DP分母からSP sizeを除く。
        if self._is_actor:
            # rollout.n回分へ複製されたsampleをmini-batchへ含めた後、
            # 実効data-parallel degree = device_mesh.size() / ulysses_sequence_parallel_size で割る。
            # この値がdp_actor.pyから見えるper-rank ppo_mini_batch_sizeになる。
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        # rollout/ref forward用micro-batchも同じdata-parallel degreeでper-GPU値へ変換する。
        # ここで得た*_per_gpuがcompute_log_prob()からDataProto.meta_infoへ注入される。
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    # actorまたはreference modelを次の順序で構築する中心関数。
    # 1. tokenizer / processor / HF configを読む。
    # 2. AutoModelをmeta-device対応context内で生成する。
    # 3. remove-padding、Ulysses、fused kernel用monkey patchを適用する。
    # 4. 必要ならgradient checkpointing / LoRAを追加する。
    # 5. FSDP1またはFSDP2でparameterをshardする。
    # 6. actorの場合だけoptimizerとLR schedulerを作る。
    # referenceではoptim_config=Noneのため、更新経路を持たない。
    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        # 大規模model生成前のGPU memoryを記録する。
        # Ray workerごとに実行されるため、rank別OOM箇所の特定に使える。
        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        # tokenizerはtext token ID、processorはVLMの画像等を含む入力前処理を担当する。
        # processorがNoneならcheckpoint managerにはtokenizerを渡す。
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        # model_dtypeはmodel parameterをロードするdtype。
        # actorの既定値をfp32にするのは、optimizerのmaster stateまでbf16化される事故を避けるため。
        # referenceはoptimizerを持たないためbf16を既定にしてmemoryを節約する。
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        # HF configにはFlashAttention 2を指定し、tokenizer側のspecial token IDとCLI overrideを反映する。
        # generation_configはrollout時のeos/pad ID取得にも利用される。
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
                
        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        # meta tensor初期化を使うと、全rankで完全なmodel parameterを一度保持せずにFSDP shardへ移行できる。
        # tie_word_embeddings有効modelではhang回避のためmeta初期化を無効化する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        # model architectureがVision2Seq mappingに含まれる場合はVLM class、
        # それ以外はcausal LM classからcheckpointをロードする。
        # この時点ではまだFSDP wrap前の通常のnn.Moduleである。
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            # Ligerおよびverl monkey patchはforward kernelを差し替える。
            # remove-paddingではattention_mask=1のtokenだけをvarlen FlashAttentionへ渡し、
            # Ulyssesではattention内部にsequence all-to-allを組み込む。
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = "torch"
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            # 一部submoduleがcheckpoint指定dtypeと異なる場合を揃える。
            # このcast後のparameterをFSDPがflatten/shardする。
            actor_module.to(torch_dtype)

            # gradient checkpointingはforward activationを保存せずbackward時に再計算する。
            # activation memoryを減らす代わりに追加forward計算が発生する。
            # use_reentrant=Falseは現行PyTorch/FSDPとの互換性を優先した設定。
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # LoRAではbase parameterの大半を固定し、低rank adapterだけを学習する。
            # enable_input_require_grads()は凍結base modelでもadapterまでgradient graphを接続するために必要。
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        # 全rankがmodelロードとpatch適用を完了してからFSDP wrapへ進むbarrier。
        # rankごとにconstructor進行が大きくずれることによるcollective mismatchを防ぐ。
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        # mixed precisionは用途別にdtypeを分ける。
        # - param_dtype : forward/backward時のparameter
        # - reduce_dtype: gradient collectiveの通信・集約dtype
        # - buffer_dtype: running stats等のbuffer
        # optimizer stateのdtypeとは別概念である。
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # auto_wrap_policyはTransformer block等の境界ごとにFSDP unitを作る。
        # unitごとにforward前parameter all-gather、backward後reduce-scatter/reshardが行われる。
        # HF rolloutで特定modelがhangする場合はwrapを無効化しroot単位にする。
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get('lora_rank', 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        # ここで作ったmeshとstrategyが、dp_actor.pyのloss.backward()に登録される
        # FSDP autograd hookの通信groupを決定する。
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        # referenceは更新しないためFSDP1 CPUOffloadでparameter shardをCPUへ逃がす。
        # actor/criticではgradient accumulation中の正しさを保つためconstructorのCPUOffloadを無効化する。
        # manual offloadは更新区間の前後で明示的に行う。
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        # FSDP1経路。
        # sync_module_states=Trueによりrank 0相当のparameter/bufferを初期化時にbroadcastし、
        # 全rankで同一重みから開始する。
        # use_orig_params=Falseではflattened parameterをoptimizerへ渡す。
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        # FSDP2経路。
        # fully_shardをmodule階層へ適用した後、wrap前に退避したfull_stateをmeshへ分散ロードする。
        # FSDP2 native offload_policy使用時はmanual load/offloadとの二重適用を避ける。
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # activation offload有効時はcheckpointed activationをCPUへ退避するhookを追加する。
        # parameter/optimizer offloadとは対象が異なる。
        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        # optimizerはactor roleにだけ生成する。
        # reference/teacherは同じDataParallelPPOActor classを使うがoptimizer=Noneでforward専用となる。
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            # schedulerはtrainer全体のtotal training stepsを基準にする。
            # 実際のscheduler.step()はupdate_actor()で1 actor updateごとに呼ばれる。
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    # rollout backendと、FSDP training modelから推論engineへ重みを受け渡すsharding managerを構築する。
    # rollout mesh:
    # shape=(rollout DP, inference TP)
    # 同じTP groupで1 modelのtensor-parallel shardを持ち、DP group間でprompt batchを分担する。
    # FSDP meshとrollout meshはparameterの分割軸が異なるため、vLLM/SGLang managerが
    # FSDP state_dictを集約・再分割して推論engineへ同期する。
    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        # world_sizeはtensor_model_parallel_sizeで割り切れる必要がある。
        # 割り切れない場合、同じTP groupを全rankへ構成できず初期化時に停止する。
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        # HF rolloutはFSDP model自身でgenerateするためweight変換が不要。
        # BaseShardingManagerはデータ・重み状態を変更しない。
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        # vLLM rolloutでは学習modelと独立した推論engineを持つ。
        # FSDPVLLMShardingManager.__enter__()がFSDP shardからfull weightを復元し、
        # vLLMが期待するTP layoutへ同期する。
        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            lora_kwargs = {'lora_kwargs': {"enable_lora":True, "max_loras":1, "max_lora_rank":self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
            # customized modeはFSDP moduleを直接渡す統合実装。
            # spmd modeはmodel pathとdevice meshから各rankがvLLM executorを構築する。
            # sync/asyncに応じてrollout classを切り替える。
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            # sharding managerはrollout前後の状態遷移を管理する。
            # enter: model weight同期、engine wake、必要ならFSDP parameter unshard。
            # exit : engine sleep、parameter reshard/offload、RNG/train mode復元。
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get('layered_summon', False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        # SGLang backendも独立推論engineを持つため、専用managerでFSDP weightを同期する。
        # import自体がGPU capabilityを検査するため、GPUを持つworker内で遅延importする。
        elif rollout_name in ["sglang", "sglang_async"]:
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. "
                    "Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76

            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    # Dispatch.ONE_TO_ALL:
    # Ray driverからの1回の呼び出しをworker group内の全rankへ同じ引数で配送する。
    # model初期化やcheckpoint操作のように、全rankが同じcollective順序で参加すべき処理に使う。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # role flagに従いactor model、rollout engine、reference modelを組み立てる公開初期化RPC。
    # actor_rollout roleでは同じactor_module_fsdpを学習側とrollout weight sourceで共有する。
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get('use_shm', False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        # actorまたはrollout機能があればactor系HF modelを1つ生成する。
        # rollout-onlyではoptimizerを作らないが、FSDP wrapはweight同期・memory分散のため行う。
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            # FSDP1 wrapper内側の元moduleを保持する。
            # LoRA adapterのdisable/saveや一部state操作はunwrapped moduleへアクセスする必要がある。
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        # rollout roleでは同じFSDP actorをweight sourceとして推論engineとsharding managerを作る。
        # actor update後、次のrollout session開始時に更新済みweightが再同期される。
        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        # standalone referenceは同じcheckpointから別のFSDP modelを作る。
        # optimizer=Noneなので更新されず、compute_ref_log_prob/topkだけを提供するteacherとなる。
        # LoRA時は別modelを作らずbase actorをreferenceとして再利用する経路もある。
        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        # actor roleではFSDP modelとoptimizerをdp_actor.pyのDataParallelPPOActorへ渡す。
        # 以後、このworkerはRPC・offload・shardingを担当し、loss/backward本体はself.actorへ委譲する。
        # FlopsCounterは処理token数と実時間からMFUを推定する。
        # CheckpointManagerはFSDP shard、optimizer、scheduler、tokenizer/processorを整合して保存する。
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # actor更新RPC。ray_trainer.pyのactor_rollout_wg.update_actor(batch)から呼ばれる。
    # 処理境界:
    # worker: GPU load/offload、SP data配置、timer/MFU、scheduler
    # dp_actor.py: micro-batch forward、loss、loss.backward、gradient clipping、optimizer.step
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())

        assert self._is_actor
        # manual offload設定時、初期化直後からparameter/optimizerをCPUへ移し、
        # rollout・forward・updateの各RPC入口で必要なものだけGPUへ戻す。
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_torch_device().current_device())

        # Ulysses managerのpreprocess_data()はSP group内でDataProtoをall-gatherする。
        # 同じsampleをSP rank全員が持った後、dp_actor._forward_micro_batch()がsequenceをrank別にsliceする。
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            # self.actor.update_policy()内部でmicro-batchごとにloss.backward()が呼ばれる。
            # modelがこのファイルでFSDP wrap済みなので、backward中にFSDP hookがparameter gradientを
            # reduce-scatterする。明示的all_reduceはここには書かれていない。
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            # global_num_tokensは元のglobal batch全体のtoken数。
            # 各rankの経過時間とworld_sizeを組み合わせ、理論FLOPsに対する利用率MFUを記録する。
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            # optimizer stepはdp_actor内で既に完了している。
            # ここでは対応するLR schedulerをactor update単位で1 step進める。
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            # metricsだけをmeta_infoへ入れたDataProtoを返す。
            # Tensor batchを返さないため、大きなGPU tensorをRay driverへ転送しない。
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        # manual offload時は更新直前にparameter shardとoptimizer stateをGPUへ復帰させる。
        # すべてのrankが同じ順序でloadし、後続FSDP collectiveへ参加する。
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    # Dispatch.DP_COMPUTE_PROTO:
    # 入力DataProtoをdata-parallel rankへ分割して配送し、戻り値をrank順に再結合するRPC。
    # 各rankは自分のlocal rowsだけをGPUへ載せる。Ulysses利用時はその後SP group内で再all-gatherする。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())

        assert self._is_rollout

        # rollout backendへeos/pad IDをmeta_infoで渡す。
        # DataProto.meta_infoはrow tensorではなく、local batch全体に共通する制御情報である。
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # ROLLOUT_KEEP_VLLM_AWAKE session mode: when begin_rollout_session() has
        # entered the sharding manager for the whole multi-turn rollout, skip the
        # per-turn __enter__/__exit__ (which would re-sync the *unchanged* actor
        # weights to vLLM and sleep/wake the engine on every turn). The actor
        # weights are frozen during a rollout, so a single sync at session start
        # produces the identical weights for every turn -> generation is unchanged.
        # Only the per-turn data sharding (preprocess/postprocess) still runs.
        if getattr(self, "_rollout_session_active", False):
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)
        else:
            with self.rollout_sharding_manager:
                log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                output = self.rollout.generate_sequences(prompts=prompts)

                log_gpu_memory_usage("After rollout generation", logger=logger)

                output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # clear kv cache
        # 生成結果はCPUへ戻した後、不要になったKV cache/allocator cacheを解放する。
        # actor update用batchはdriver側でresponses等を元batchへunionする。
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_rollout_session(self):
        """Open the rollout sharding manager once for an entire multi-turn rollout.

        Without this, generate_sequences() re-enters the sharding manager every
        turn, which re-gathers the FSDP state_dict, re-syncs the full model weights
        to vLLM, and wakes/sleeps the engine — ~50x per rollout, all redundant
        because the actor weights are frozen during a rollout. Hoisting the
        enter/exit to the rollout boundary keeps vLLM awake and the weights synced
        once. Accuracy-safe: identical (frozen) weights are used on every turn, so
        the generated tokens are unchanged. No-op unless the vLLM sharding manager
        is in use. Paired with end_rollout_session() in a finally block.
        """
        assert self._is_rollout
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        if not isinstance(self.rollout_sharding_manager, FSDPVLLMShardingManager):
            return
        # 通常はwith rollout_sharding_managerのenter/exitごとにFSDP->vLLM weight同期とwake/sleepが走る。
        # session modeではmulti-turnの最初に1回だけenterし、各turnではprompt dataのpre/postprocessだけ行う。
        if getattr(self, "_rollout_session_active", False):
            return
        self.rollout_sharding_manager.__enter__()
        self._rollout_session_active = True

    # ONE_TO_ALLで全rankのrollout sharding managerを同時に開く。
    # TP/FSDP collectiveを含むため、一部rankだけがenterするとdeadlockする。
    # begin_rollout_sessionと対になる終了RPC。
    # finallyから必ず呼び、vLLM sleep・FSDP reshard・RNG/train mode復元を保証する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_rollout_session(self):
        """Close the sharding manager opened by begin_rollout_session(), restoring
        exactly the post-generation state of the non-session path (sleep/offload
        vLLM, restore RNG + train mode, empty cache)."""
        if not getattr(self, "_rollout_session_active", False):
            return
        self._rollout_session_active = False
        self.rollout_sharding_manager.__exit__(None, None, None)



    # 生成RPC。prompt rowsはDP rankへ分割され、rollout engineのTP group内で生成される。
    # 生成中はactor parameterを更新しないため、multi-turn session全体で同じweightを再利用できる。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        # 更新後、次のrolloutや他model処理へGPU memoryを譲るためmanual offloadを戻す。
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext
        # LoRA actorをreferenceとして使う場合、disable_adapter contextでbase modelだけをforwardする。
        # 通常actor log-probではnullcontextとなりadapterを有効のまま計算する。
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        # dp_actorへforward分割条件をmeta_infoで渡す。
        # fixed micro-batchまたはmax_token_len基準dynamic micro-batchのどちらを使うかがここで決まる。
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # FSDP1 root moduleはforward後にunshard状態が残る場合がある。
        # 明示的reshard(True)でparameterを再分割し、次処理前にmemoryを回収する。
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    # rolloutで生成したresponseについて、現在actorのsampled-token log-probを再計算するRPC。
    # 結果はold_log_probsとしてPPO ratio、SDLのon-policy補正等に使われる。
    # forwardはdp_actor.compute_log_prob()内でno_grad実行され、parameter gradientは作らない。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info['is_lora'] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={'ref_log_prob': data.batch['old_log_probs']})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_torch_device().current_device())

        # reference専用configのmicro-batch/token budgetを使用する。
        # actor forwardと独立にmemory上限を設定できる。
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        # SP group内でdataを揃え、dp_actorがresponse log-prob/entropyを計算する。
        # postprocessでSP複製分を取り除き、元のDP rank担当rowsだけへ戻す。
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    # reference log-prob計算RPC。
    # - LoRA: actorのadapterを無効化したbase modelをreferenceとして再利用。
    # - 非LoRA: standalone self.ref_policyをforward。
    # teacher/ref側はno_gradで、返却tensorはCPUへ移される。
    # Pure OPD top-k teacher forward。
    # 各response tokenでteacher上位K token IDと、そのID位置のfull-vocab log-softmaxを返す。
    # 出力shape:
    # teacher_topk_ids      : (local_B, response_length, K), int64
    # teacher_topk_logprobs : (local_B, response_length, K), float
    # 後でstudent actorへ同じIDを渡し、同一support上のKLを計算する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_topk_log_prob(self, data: DataProto):
        """Teacher-side top-k log-probs for OPD distillation: per response token,
        the teacher's top-k token ids and the teacher's full-vocab log-softmax at
        those ids. ``topk_k`` is taken from data.meta_info (default 20)."""
        assert self._is_ref
        data = data.to(get_torch_device().current_device())

        # topk_kはDataProto.meta_infoから読み、未指定時20。
        # top-k経路はdp_actor側の制約によりfused kernel/Ulysses SPと併用できない実装がある。
        topk_k = int(data.meta_info.get("topk_k", 20))
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            topk_logprob, topk_ids = self.ref_policy.compute_topk_log_prob(data=data, topk_k=topk_k)
            output = DataProto.from_dict(
                tensors={"teacher_topk_logprobs": topk_logprob, "teacher_topk_ids": topk_ids},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # CheckpointManager内部でmodel/optimizer/scheduler/RNG等を保存した後、barrierで全rank完了を揃える。
        # 一部rankだけ先に次処理へ進みcheckpoint directoryを触る競合を防ぐ。
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        dist.barrier()

        # LoRA adapter保存ではFSDP shardからadapter parameterを一時集約し、rank 0だけがファイルへ書く。
        # base model checkpointとは別ディレクトリにPEFT configと重みを保存する。
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(self.actor_module.peft_config.get('default', {}))
                peft_config['task_type'] = peft_config['task_type'].value
                peft_config['peft_type'] = peft_config['peft_type'].value
                peft_config['target_modules'] = list(peft_config['target_modules'])
            try:
                if isinstance(self.actor_module_fsdp, FSDP):
                    self.actor_module_fsdp = self.actor_module_fsdp.cuda()
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding='utf-8') as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[rank-{self.rank}]: Save LoRA Adapter Error ({e})")

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    # actor checkpoint保存RPC。
    # 全rankがFSDP shard保存へ参加し、LoRA時はrank 0がadapterだけをsafetensorsとして追加保存する。
    # checkpoint loadもONE_TO_ALLで全rank同時実行する。
    # manual offload時はload前にmodelをGPUへ戻し、完了後に再offloadする。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)


# ===============================================================================
# CriticWorker
# ===============================================================================
# GAE等でvalue functionを使う場合のworker。
# GRPO/Pure OPDでcriticを無効にしている構成では生成されない。
# DataParallelPPOCriticへvalue forward・critic loss/backwardを委譲し、
# この層はactorと同様にFSDP/SP/offload/Ray RPCを管理する。
class CriticWorker(Worker):
    # critic用process group・FSDP/SP mesh・batch sizeを初期化する。
    # actorとmodel architecture/tokenizerが異なる可能性があるため独立workerとして構築する。
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        # critic parameter/optimizerのmanual offload設定。
        # criticもgradient accumulationを行うためFSDP1 constructor CPUOffloadは後段で無効にし、
        # 更新RPC境界で明示的load/offloadする。
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        # criticのglobal mini/micro/forward batch sizeをper-DP-rankへ正規化する。
        # value forward用forward_micro_batch_sizeと更新用ppo_micro_batch_sizeは別設定。
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        self._is_lora = self.config.model.get('lora_rank', 0) > 0

    # critic modelをTokenClassification(num_labels=1)として生成し、
    # FSDP wrap、optimizer、schedulerまで組み立てる。
    # 各token位置のscalar valueを出力するarchitectureとして利用する。
    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        # critic checkpointとactor tokenizer pathを別々に取得する。
        # actorが生成したinput_idsを正しく解釈するため、special token情報はactor tokenizer側へ合わせる。
        use_shm = config.model.get('use_shm', False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        # criticはoptimizerを持つため既定fp32でロードし、FSDP mixed precision時にbf16 forwardへcastする。
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # AutoModelForTokenClassificationへnum_labels=1を設定し、
        # sequence各位置からvalue scalarを出すcritic headとして使う。
        from transformers import AutoConfig, AutoModelForTokenClassification

        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation="flash_attention_2", trust_remote_code=config.model.get("trust_remote_code", False))
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        # actorと同様にmeta initialization contextでmodelを構築し、
        # remove-padding/Ulysses monkey patchを適用する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # critic LoRA経路。設定上のtask_typeはCAUSAL_LMを使用しているが、
        # 実際の出力headはTokenClassification model側に保持される。
        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                'task_type': TaskType.CAUSAL_LM,
                'r': self.config.model.lora_rank,
                'lora_alpha': self.config.model.lora_alpha,
                'target_modules': convert_to_regular_types(self.config.model.target_modules),
                'bias': "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        # critic用mixed precisionとauto-wrap policy。
        # FSDP unit境界がvalue backward時のparameter all-gather/reduce-scatter単位になる。
        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.get('lora_rank', 0) > 0)

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        # critic FSDP1ではCPUOffloadを無効化する。
        # micro-batch間に未更新gradientを蓄積するため、parameterを毎forward後CPUへ退避すると
        # 不正確な結果になるケースを避けている。
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=False,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        # critic FSDP2ではnative offload_policyを選択できる。
        # 有効時はmanual offload flagを下げ、同じstateを二重に移動しない。
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        # critic optimizer/scheduler。
        # optimizer.step本体はdp_critic.pyで行い、このworkerはupdate_critic終了後にschedulerを進める。
        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    # critic model初期化RPC。
    # DataParallelPPOCriticとCheckpointManagerへ同じFSDP model/optimizerを渡す。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        # value計算用micro-batch size/token budgetをmeta_infoへ注入する。
        # dynamic batching時はsample数ではなく有効token総数を上限に分割する。
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    # value forward RPC。no_gradのmicro-batch forwardをdp_criticへ委譲し、
    # 応答token位置のvaluesをDataProtoとしてCPUへ返す。
    # critic更新RPC。
    # load/offloadとSP data配置を行い、value loss/backward/optimizer stepはdp_criticへ委譲する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_torch_device().current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # self.critic.update_critic()中にmicro-batch backwardとFSDP gradient通信が発生する。
            # worker側では所要時間からMFUを計算しschedulerを更新する。
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    # critic checkpoint保存。全rankでshardを書いた後barrierし、manual offload状態へ戻す。
    # critic checkpoint復元。model/optimizer stateを全rankで読み、barrier後にoffload設定を復元する。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
# ===============================================================================
# RewardModelWorker
# ===============================================================================
# model-based rewardを使う場合のforward専用worker。
# AutoModelForTokenClassificationの最終有効token scoreを、response末尾のtoken-level scoreへ変換する。
# optimizerを持たず、FSDP parameterは常にoffload可能な推論modelとして扱う。
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    # reward model用process group、FSDP/SP mesh、per-GPU micro-batch sizeを初期化する。
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # remove-padding有効時はpadding tokenを除いてreward modelをforwardし、最後に(B,S)へ復元する。
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    # reward model checkpoint/tokenizerを読み、FSDP1またはFSDP2でforward専用modelを構築する。
    # input_tokenizerが異なる場合は、actorの会話をreward model用chat templateへ再tokenizeする。
    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get('use_shm', False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # actorとreward modelのtokenizer/chat templateが同じなら入力Tensorをそのまま使用する。
        # 異なる場合はraw_promptと生成responseを文字列へ戻してtarget tokenizerで再構築する。
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # reward modelはoptimizerを持たないためbf16でロードし、memoryを節約する。
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        # reward modelもTransformer block単位でFSDP wrapする。
        # FSDP1ではCPUOffloadを常時有効にし、forward時に必要なparameterを自動転送する。
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=False,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    # reward model初期化RPC。外部model class登録後、forward専用FSDP moduleを作る。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    # 1 reward micro-batchのforward。
    # remove-padding経路:
    # input_ids(B,S) -> valid token列(1,nnz)
    # -> 必要ならUlyssesでsequence分割
    # -> logits(nnz,1)をgather
    # -> pad_inputで(B,S)へ復元
    # 最後に各sampleの最終有効token位置だけをscalar scoreとして取り出す。
    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        # reward modelは学習しないためtorch.no_grad()。
        # bf16 autocastでforwardし、gradient graph/optimizer stateを作らない。
        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            # attention_maskでpaddingを除外し、indicesを保存する。
            # indicesはreward logitを元のbatch/sequence位置へscatterする際にも使う。
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                # SP size>1ではvalid token列をrankごとにsliceし、
                # forward後にgather_outpus_and_unpad()でglobal valid-token順へ戻す。
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    # sample scalar scoreをresponse token-level tensorへ配置する。
    # 最終有効tokenだけ非ゼロにし、ray_trainer側の報酬加算形式(B,R)へ合わせる。
    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        # position_ids*attention_maskが最大となる位置を各sampleの最終有効tokenとして選ぶ。
        # sequence classification rewardを1 scalar/sampleへ縮約する。
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    # actor入力とreward model入力のchat template/tokenizerが異なる場合の変換。
    # raw promptへassistant responseを追加し、target tokenizerでright-paddingして再tokenizeする。
    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    # reward score計算RPC。
    # 必要ならchat templateを切り替え、fixed/dynamic micro-batchでforwardし、
    # dynamic batchingの並べ替えを元row順へ戻してrm_scores(B,R)を返す。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_torch_device().current_device())

        # perform forward computation
        # SP groupへreward入力と元dataを揃える。
        # 元dataはresponse_lengthや最終有効位置へscoreを展開するため保持する。
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # dynamic batchでは有効token総数を上限としてsequence長を均等化する。
            # 出力concat後、保存したindicesの逆写像で元のsample順へ復元する。
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # FSDP1 forward後にroot parameterがunshard状態なら明示的にreshardし、GPU memoryを回収する。
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
# ===============================================================================
# AsyncActorRolloutRefWorker
# ===============================================================================
# async vLLM serverからDIRECT_ROLLOUT_METHOD RPCで操作されるworker。
# 通常のgenerate_sequences RPCは使わず、外部distributed executorが各TP rankへmethodを直接配送する。
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    # 基底classと同じrollout/sharding managerを作り、global rankをvLLMのDP rank/TP rankへ分解する。
    # TP rank 0がcontrol methodの代表ログを出す。
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    # async構成では生成要求をAsyncLLMServer経由で処理するため、
    # 通常のDP_COMPUTE_PROTO generate_sequencesは誤用防止のため無効化する。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    # Dispatch.DIRECT_ROLLOUT_METHOD:
    # DataProto分割を介さず、method名と引数をvLLM executorの各rankへ直接転送する。
    # collective_rpcの呼び出し順序はTP rank間で一致させる必要がある。
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    # resume/offloadはasync rollout engineのsleep/wake状態を外部serverから制御する薄いRPC。
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def resume(self):
        return self.rollout.resume()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def offload(self):
        return self.rollout.offload()
