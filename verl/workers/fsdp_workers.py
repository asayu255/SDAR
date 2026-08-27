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

import logging
import os
import time
import warnings
from typing import Union

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl.utils.py_functional import convert_to_regular_types
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.metric.memory import device_footprint_gb, per_rank_memory_metrics
from verl.utils.metric.stall_counters import per_rank_stall_counter_metrics
from verl.utils.host_gc import collect_at_step_boundary, freeze_permanent_heap, refreeze_if_due
from verl.utils.phase_timing import PhaseTimer, mark as _mark
from verl.workers.actor.dp_actor import _actor_phase
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
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available


from peft import LoraConfig, TaskType, get_peft_model
from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
from dataclasses import asdict
import json


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh, fsdp_config=None):
    """FSDP sharding strategy for a mesh, optionally overridden by config.

    Defaults are unchanged: FULL_SHARD (ZeRO-3) on a 1-D mesh, HYBRID_SHARD on 2-D.

    ``fsdp_config.sharding_strategy: shard_grad_op`` selects ZeRO-2 instead, which
    keeps parameters gathered from the forward through the backward rather than
    resharding after each. That removes two of the three all-gathers a layer does
    per micro-batch under gradient checkpointing (forward, recompute, backward),
    leaving only the gradient reduce-scatter as collective traffic.

    The arithmetic is untouched: an all-gather only moves bytes, the forward and
    backward kernels see identical inputs, and gradients are reduced by the same
    reduce-scatter in the same reduce_dtype. What changes is memory -- parameters
    stay unsharded for the length of a micro-batch's backward, so peak grows by
    roughly the unsharded parameter size minus its shard.
    """
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        default, alternatives = ShardingStrategy.FULL_SHARD, {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }
    elif device_mesh.ndim == 2:
        default, alternatives = ShardingStrategy.HYBRID_SHARD, {
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
            "shard_grad_op": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")

    requested = None
    if fsdp_config is not None:
        requested = fsdp_config.get("sharding_strategy", None)
    if requested is None:
        return default
    requested = str(requested).strip().lower()
    if requested == "shard_grad_op" and fsdp_config.get("param_offload", False):
        warnings.warn(
            "fsdp_config.sharding_strategy=shard_grad_op with param_offload=True: "
            "the parameters ZeRO-2 keeps resident are the ones offloading sends to "
            "CPU, so the all-gathers this is meant to remove come back as transfers.",
            stacklevel=2,
        )
    if requested not in alternatives:
        raise ValueError(
            f"fsdp_config.sharding_strategy={requested!r} is not supported for a "
            f"{device_mesh.ndim}-D mesh; choose one of {sorted(alternatives)}"
        )
    return alternatives[requested]


def _fsdp_param_dtype(module, default):
    """The dtype FSDP casts parameters to for the forward.

    ``summon_full_params`` returns the float32 masters, but the projection that
    produced the log-probs ran at ``MixedPrecision.param_dtype``. Anything
    recomputing that projection has to use the same one or it is doing different
    arithmetic. Walks the wrapper chain because only the FSDP module carries it.
    """
    seen = module
    for _ in range(8):
        mp = getattr(seen, "mixed_precision", None)
        dtype = getattr(mp, "param_dtype", None)
        if dtype is not None:
            return dtype
        nxt = getattr(seen, "_fsdp_wrapped_module", None) or getattr(seen, "_orig_mod", None)
        if nxt is None or nxt is seen:
            break
        seen = nxt
    return default


# Per-call breakdown of generate_sequences, printed from rank 0 every N calls.
#
# The driver measures this call as one number and cannot see inside it: the Ray
# round trip, the sharding manager's data reshaping, the engine call and the
# detokenisation are one opaque span from out there. A search batch pays that
# span once per turn, tens of thousands of times in an evaluation.
#
# Summed here and printed as a mean, so the cost is attributed rather than
# guessed at. What the driver measures MINUS what this prints is the Ray round
# trip, which is the one leg neither side can time alone.
#
# Off by default: it is a diagnostic, and a run that is not being profiled
# should not carry even the dict.
_GEN_PHASE_TIMING = os.environ.get("ROLLOUT_TURN_TIMING", "0").strip().lower() in ("1", "true", "yes", "on")
_GEN_PHASES = ("to_device", "preprocess", "generate", "postprocess", "to_cpu")


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", rank=rank, world_size=world_size)

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get('lora_rank', 0)
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
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
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

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

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
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
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

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
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
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

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get('lora_rank', 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_config)

        # TODO: add transformer policy
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation.
        #
        # For the reference / teacher policy, fsdp_config.param_offload decides. It
        # used to be forced on, which made that key dead: the ref path has no manual
        # offload (compute_ref_log_prob / compute_ref_topk_log_prob never call
        # load_fsdp_model_to_gpu), so _is_offload_param was set from it and then read
        # by nobody, and FSDP did the offloading unconditionally instead.
        #
        # The cost of forcing it is paid per all-gather, not per step: FSDP re-fetches
        # each unit's parameters from CPU on every micro-batch. For the OPD teachers
        # that is the whole model over PCIe once per micro-batch (measured at 7.6-8.9
        # GB/s sustained pcieRX through teacher_forward, against ~2 GB/s during gen),
        # and with no NVLink on the box it competes with the collectives. Keeping the
        # shards resident costs param_bytes / world_size per teacher and removes it.
        #
        # Correctness is not at stake either way here -- the ref runs under no_grad, so
        # the grad-accumulation problem that forces this off for the actor cannot arise.
        cpu_offload = None if role == "actor" else (CPUOffload(offload_params=True) if fsdp_config.get("param_offload", False) else None)
        fsdp_strategy = self.config.actor.strategy
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
                # Off by default (upstream behaviour). fsdp_config.forward_prefetch=True
                # issues the NEXT FSDP unit's all-gather while the current one computes,
                # overlapping communication it would otherwise serialize -- scheduling
                # only, the arithmetic is untouched. Worth turning on when collectives
                # run over PCIe (no NVLink) and the profile is communication-bound.
                forward_prefetch=bool(fsdp_config.get("forward_prefetch", False)),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                # Same rule as the FSDP1 branch above: the ref/teacher honours
                # fsdp_config.param_offload instead of being forced onto CPU.
                cpu_offload = None if role == "actor" else (CPUOffloadPolicy(pin_memory=True) if fsdp_config.get("param_offload", False) else None)

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

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

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

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            lora_kwargs = {'lora_kwargs': {"enable_lora":True, "max_loras":1, "max_lora_rank":self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get('use_shm', False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

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
            # The actor has no tokenizer, and the event dump needs to know which
            # token ids open and close <think> / <action> / <search> / <answer>
            # to say what KIND of position an event sat at. Tokenised once here,
            # where a tokenizer exists, and handed down as ids. add_special_tokens
            # is off: the tag is being matched as a substring of a response, not
            # encoded as a standalone sequence, so a BOS in front of it would
            # match nothing.
            from verl.trainer.ppo.sign_weights import TAG_ROLES

            self.actor.sign_role_tag_ids = {
                tag: self.tokenizer.encode(tag, add_special_tokens=False) for tag in TAG_ROLES
            }

        if self._is_rollout:
            # Priced, because nothing else here reports it. The engine is built
            # on every rank, sleeps itself at construction, and still holds a
            # CUDA context, whatever survives sleep(level=1), and the allocator
            # high-water mark its profiling run left behind. That is device
            # memory the actor could be spending on a larger micro batch, and
            # neither counter next door sees it: allocated only counts this
            # process's live tensors and reserved is a counter that outlives its
            # pages (vLLM's CuMemAllocator unmaps out of band). mem_get_info is
            # the driver's own free/total, which is what nvidia-smi shows.
            before = device_footprint_gb(get_torch_device())
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            after = device_footprint_gb(get_torch_device())
            print(
                f"[rollout-footprint] rank {self.rank}: building the "
                f"{self.config.rollout.name} rollout took the device from "
                f"{before:.2f} to {after:.2f} GiB in use ({after - before:+.2f} GiB, "
                f"gpu_memory_utilization={self.config.rollout.gpu_memory_utilization}, "
                f"enforce_eager={self.config.rollout.enforce_eager})",
                flush=True,
            )

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
            # Set by register_teacher_lm_head when student_indexed_topk is on; the
            # cache needs to know which teacher an entry belongs to.
            self._teacher_lm_head_task = None
            # Micro-batches per step that also build the teacher's own top-k,
            # purely as a witness. Reset in clear_teacher_hidden_cache.
            self._teacher_witness_budget = int(os.environ.get("TEACHER_WITNESS_MICRO_BATCHES", "2"))

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

        # Everything alive at this point -- the module tree, the sharded
        # parameters and optimizer state, the tokenizer, Ray's plumbing -- lives
        # for the whole run, and gen-2 collections walk all of it without ever
        # being able to free any of it. Freezing it here takes those objects out
        # of the sweep's reach. See verl/utils/host_gc.py for why a host-side
        # sweep shows up as a GPU dip.
        report = freeze_permanent_heap()
        if report["enabled"]:
            print(
                f"[host-gc] rank {self.rank}: froze {report['frozen']} objects "
                f"({report['collected']} collected, manual={report['manual']}) "
                f"in {report['seconds']:.2f} s",
                flush=True,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Step boundary: the device is idle here by construction, so host-side
        # GC work is free. Freeze once more to capture what the warm-up step
        # created and keeps -- Dynamo caches, Adam's lazily allocated state,
        # FSDP's deferred structures, none of it visible to the init-time freeze
        # -- then the per-step sweep, which after the freezes costs milliseconds
        # and drains the survivor count that would otherwise trip a full
        # collection mid-forward, where it lands on the device.
        refrozen = refreeze_if_due()
        if refrozen is not None:
            print(
                f"[host-gc] rank {self.rank}: re-froze +{refrozen['frozen_delta']} warm-up objects "
                f"(total {refrozen['frozen_total']}, {refrozen['collected']} collected) "
                f"in {refrozen['seconds']:.2f} s",
                flush=True,
            )
        collect_at_step_boundary()

        # Support all hardwares.
        #
        # Tagged because this is the near side of the step boundary that survived
        # pipelining. A Ray actor runs its calls one at a time, so the worker
        # cannot start moving step k+1's batch onto the device until k has
        # returned -- whatever that costs is unhideable from the driver side, and
        # until it is measured it is indistinguishable from the Ray
        # deserialisation that precedes it (which lands outside this phase, as
        # (idle/other), because it happens before the method body).
        with _actor_phase("actor.h2d"):
            data = data.to(get_torch_device().current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_torch_device().current_device())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            # The driver's timing_s/update_actor is a blocking ray.get around this
            # call, so it also covers serializing the batch into the object store,
            # the workers pulling their shards, and the metrics coming back. At a
            # few thousand rows that batch is hundreds of MB, and the GPUs sit idle
            # for all of it. Reporting the compute time separately makes the
            # difference readable: timing_s/update_actor minus this is transport,
            # and a GPU-idle window inside update_actor is one or the other.
            metrics = dict(metrics)
            metrics["timing_s/update_actor_worker"] = delta_time
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            # Every rank's, not just this one's. The two lines this replaces read
            # max_memory_* on the local card and shipped it in meta_info, and
            # DataProto.concat keeps meta_info from the FIRST worker only -- so
            # perf/max_memory_reserved_gb reached wandb as rank 0's number under
            # a name that reads like a reduction. The ranks are meant to match
            # (shard_grad_op replicates parameters, _balance_batch equalises rows
            # and tokens), which makes a spread a finding, and it was invisible.
            # Same two key names, so the history stays continuous; they are now
            # actually the cross-rank maxima they claimed to be.
            metrics.update(per_rank_memory_metrics(get_torch_device()))
            # Which of {cudaMalloc segment growth, gen-2 GC, dynamo recompile}
            # fired on this rank this step. Each is a candidate mechanism for a
            # sub-second in-micro-batch stall and each increments a counter the
            # process already keeps, so a dip on a step where exactly one moved
            # is an attribution. Note the interaction with host_gc: freezing does
            # not stop collections, it empties what they walk, so stall/gc_gen2
            # rising is not evidence the freeze failed.
            metrics.update(per_rank_stall_counter_metrics(get_torch_device()))
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})
            # The per-token sign-weight diagnostic, decoded. It rides beside
            # "metrics" rather than inside it because it is a table, not
            # scalars, and reduce_metrics would average it into nonsense.
            # DataProto.concat keeps rank 0's meta_info, which is the right one
            # here: the actor all-reduced the counts before ranking them, so
            # every rank carries the same global table.
            #
            # This is where the ids become text: the actor process has no
            # tokenizer. convert_ids_to_tokens, not decode -- the raw piece
            # (with its space marker) is what identifies the vocabulary entry,
            # while decode would strip exactly that and merge distinct tokens.
            for attr, key in (
                ("last_token_report", "sign_token_report"),
                ("last_pair_token_report", "sign_pair_token_report"),
                ("last_event_report", "sign_event_report"),
                ("last_pair_event_report", "sign_pair_event_report"),
            ):
                report = getattr(self.actor, attr, None)
                if not report:
                    continue
                pieces = self.tokenizer.convert_ids_to_tokens([r["token_id"] for r in report])
                rows = [{**row, "token": piece} for row, piece in zip(report, pieces)]
                # Event rows also carry the window around the position. Decoded
                # here rather than dumped as ids for the same reason the token
                # column is: the trainer would have to hold a second tokenizer,
                # and a reader would have to hold a third. decode, not
                # convert_ids_to_tokens, because this one is meant to be READ --
                # the point is the sentence the event sat in.
                for row in rows:
                    ctx = row.pop("context_ids", None)
                    if ctx is None:
                        continue
                    row["context"] = self.tokenizer.decode(ctx, skip_special_tokens=False)
                output.meta_info[key] = rows

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    def _record_gen_phases(self, marks):
        """Accumulate one call's phase times; print the mean every N calls.

        Per call is unprintable -- a generate happens once per turn, tens of
        thousands of times in an evaluation -- and a mean over a period says the
        same thing without burying the log.
        """
        timer = getattr(self, "_gen_phase_timer", None)
        if timer is None:
            timer = self._gen_phase_timer = PhaseTimer(
                "gen-phases",
                _GEN_PHASES,
                every=int(os.environ.get("ROLLOUT_GEN_PHASE_EVERY", "50")),
                note="(driver's gen column minus this = the Ray round trip)",
                rank=lambda: getattr(self, "_rank", None),
            )
        timer.record(marks)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        marks = {} if _GEN_PHASE_TIMING else None
        _t = time.perf_counter()
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())
        if marks is not None:
            marks["to_device"] = time.perf_counter() - _t

        assert self._is_rollout

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
            # Counted so end_rollout_session can say how many generates one wake
            # served. A session that served 1 bought nothing, and that is the
            # symptom of a hoist that silently did not take.
            self._rollout_session_generates = getattr(self, "_rollout_session_generates", 0) + 1
            _t = time.perf_counter()
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            _t = _mark(marks, "preprocess", _t)
            output = self.rollout.generate_sequences(prompts=prompts)
            _t = _mark(marks, "generate", _t)
            output = self.rollout_sharding_manager.postprocess_data(output)
            _mark(marks, "postprocess", _t)
        else:
            # Said once, because it is the difference between one wake per rollout
            # and one per turn -- 21 GB of vLLM state unmapped and remapped every
            # time. _say_session covers a session that opened and closed; this
            # covers the case it cannot see, where begin_rollout_session was never
            # called at all, and the two states are otherwise indistinguishable
            # from outside (the sharding manager's own enter/exit log is DEBUG on
            # a logger pinned to WARN).
            if getattr(self, "_rank", None) == 0 and not getattr(self, "_rollout_no_session_warned", False):
                self._rollout_no_session_warned = True
                print(
                    "[rollout-session] rank 0: generating WITHOUT a session -- every turn "
                    "will wake and sleep vLLM and re-sync the frozen weights. Set "
                    "ROLLOUT_KEEP_VLLM_AWAKE=1 in the process that runs the rollout loop "
                    "(the driver, not just the workers).",
                    flush=True,
                )
            with self.rollout_sharding_manager:
                log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

                _t = time.perf_counter()
                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                _t = _mark(marks, "preprocess", _t)
                output = self.rollout.generate_sequences(prompts=prompts)
                _t = _mark(marks, "generate", _t)

                log_gpu_memory_usage("After rollout generation", logger=logger)

                output = self.rollout_sharding_manager.postprocess_data(output)
                _mark(marks, "postprocess", _t)

        _t = time.perf_counter()
        output = output.to("cpu")
        if marks is not None:
            marks["to_cpu"] = time.perf_counter() - _t
            self._record_gen_phases(marks)

        # clear kv cache -- but NOT inside a rollout session. This call releases
        # cached blocks back to the driver and forces a device synchronize, and it
        # ran once per turn (~50x per rollout) even with the session holding vLLM
        # awake, so every turn paid a sync plus the allocator re-acquiring the same
        # blocks. It cannot free vLLM's KV cache either: the engine owns that, and
        # empty_cache only returns *unused* cached blocks to the driver. The
        # sharding manager's own note says to keep this to the wake/sleep
        # boundaries; end_rollout_session() exits it, which does exactly that.
        if not getattr(self, "_rollout_session_active", False):
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
        # Before the import, deliberately: with SKIP_ROLLOUT_BUILD there is no
        # manager at all, and pulling in the vLLM machinery just to conclude
        # "not a vLLM manager" would make this hook the one place a skipped
        # build still pays for -- or crashes on -- vLLM.
        if self.rollout_sharding_manager is None:
            self._say_session("no rollout sharding manager (SKIP_ROLLOUT_BUILD)")
            return
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        if not isinstance(self.rollout_sharding_manager, FSDPVLLMShardingManager):
            self._say_session(f"manager is {type(self.rollout_sharding_manager).__name__}, not vLLM's")
            return
        # Nesting, by depth rather than by a bool. The rollout loop opens a
        # session per multi_turn_loop; _validate opens one around the whole
        # validation, which is 413 of those on this arm. With a bool the inner
        # scope's end_rollout_session would close the outer one on the first
        # batch and every batch after it would wake and sleep vLLM again -- the
        # hoist would silently do nothing. Only the outermost scope enters and
        # exits; the inner ones just count.
        self._rollout_session_depth = getattr(self, "_rollout_session_depth", 0) + 1
        if self._rollout_session_depth > 1:
            return
        self.rollout_sharding_manager.__enter__()
        self._rollout_session_active = True
        self._rollout_session_generates = 0
        self._say_session("opened -- vLLM stays awake until the outermost scope closes")

    def _say_session(self, message: str):
        """One line per distinct session outcome, from rank 0.

        Every early return above is a silent downgrade to per-turn wake/sleep, and
        each has a different cause and a different fix. Printing the reason is what
        makes the two states tellable apart at all -- the manager's own enter/exit
        logging is DEBUG under a WARN logger, so it never reaches a log file.
        Deduplicated because these run once per rollout, not once per run.
        """
        # Defensively, because this is a print: `rank` is a property over
        # `self._rank`, which is set by the worker's distributed init. A hook
        # that used to be a silent no-op must not start raising AttributeError
        # on a worker that has not reached that point -- observing something is
        # not a licence to break it.
        if getattr(self, "_rank", None) != 0:
            return
        seen = getattr(self, "_rollout_session_said", None)
        if seen is None:
            seen = self._rollout_session_said = set()
        if message in seen:
            return
        seen.add(message)
        print(f"[rollout-session] rank 0: {message}", flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_rollout_session(self):
        """Close the session opened by begin_rollout_session(), restoring exactly
        the post-generation state of the non-session path (sleep/offload vLLM,
        restore RNG + train mode, empty cache).

        Reference-counted: only the call that balances the OUTERMOST
        begin_rollout_session actually exits the manager."""
        depth = getattr(self, "_rollout_session_depth", 0)
        if depth <= 0:
            # begin_rollout_session declined (no manager, or not vLLM's), or this
            # is an unpaired call. Either way there is nothing open to close.
            return
        self._rollout_session_depth = depth - 1
        if self._rollout_session_depth > 0:
            # An outer scope still holds it. Sleeping vLLM here is exactly the
            # bug the depth counter exists to prevent.
            return
        if not getattr(self, "_rollout_session_active", False):
            return
        self._rollout_session_active = False
        self.rollout_sharding_manager.__exit__(None, None, None)
        # The count is the whole point: one session covering N generate calls is
        # the working state, and N wake/sleep cycles is the broken one. A session
        # that served 1 generate is a session that bought nothing.
        served = getattr(self, "_rollout_session_generates", 0)
        if getattr(self, "_rank", None) == 0:
            print(
                f"[rollout-session] rank 0: closed after {served} generate call"
                f"{'' if served == 1 else 's'} on one wake",
                flush=True,
            )



    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
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
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

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

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
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

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_topk_log_prob(self, data: DataProto):
        """Teacher-side top-k log-probs for OPD distillation: per response token,
        the teacher's top-k token ids and the teacher's full-vocab log-softmax at
        those ids. ``topk_k`` is taken from data.meta_info (default 20)."""
        assert self._is_ref
        data = data.to(get_torch_device().current_device())

        topk_k = int(data.meta_info.get("topk_k", 20))
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        # student_indexed_topk resolves this teacher at ids the student picks during
        # its own training forward, which has not happened yet. Only the final gather
        # depends on those ids, so keep what does not: the body's hidden states and
        # the full-vocabulary normaliser.
        #
        # In that mode the teacher's OWN top-k is not part of the answer -- nothing
        # downstream reads it -- so it is built for a couple of micro-batches per
        # step as a witness and for nothing else. It was a selection over the whole
        # vocabulary plus two scatters back to (bs, response_length, k) for every
        # row, and then ~860 MB/step of it travelled to the driver to be ignored.
        cache_ids = data.batch.get("teacher_cache_ids", None) if "teacher_cache_ids" in data.batch.keys() else None
        want_hidden = cache_ids is not None and bool(self.config.ref.get("student_indexed_topk", False))

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if want_hidden:
                hidden, lse, w_rows, w_ids, w_lp = self.ref_policy.compute_topk_log_prob(
                    data=data, topk_k=topk_k, return_hidden=True,
                    witness_micro_batches=self._take_witness_budget(),
                )
                self._cache_teacher_hidden(
                    cache_ids, hidden, lse, w_rows, w_ids, w_lp,
                    attention_mask=data.batch["attention_mask"],
                    input_ids=data.batch["input_ids"],
                )
                # The driver only needs the row count back (it unpads by it); the
                # values it used to merge here are now resolved in the actor.
                output = DataProto.from_dict(
                    tensors={"teacher_scored": torch.ones(len(data), 1, dtype=torch.bool)},
                )
            else:
                topk_logprob, topk_ids = self.ref_policy.compute_topk_log_prob(data=data, topk_k=topk_k)
                output = DataProto.from_dict(
                    tensors={"teacher_topk_logprobs": topk_logprob, "teacher_topk_ids": topk_ids},
                )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    def _take_witness_budget(self) -> int:
        """How many of this call's micro-batches also build the teacher's own top-k.

        A budget rather than a rate: the witness catches an entry filed under the
        wrong row, which is a systematic failure of the routing, so any handful of
        rows shows it. Spending it on the step's first calls keeps the check
        cheap and keeps it running every step. Reset by
        ``clear_teacher_hidden_cache``.
        """
        take = min(self._teacher_witness_budget, 1)
        self._teacher_witness_budget -= take
        return take

    def _cache_teacher_hidden(self, cache_ids, hidden, lse, witness_rows, witness_ids, witness_lp,
                              attention_mask=None, input_ids=None):
        """Keep this call's hidden states so the actor can score arbitrary ids later.

        One entry per ROW, holding every response position that carries signal,
        because that is the granularity the student picks its top-k at: a row is
        scored once but every position gets its own support set.

        The teacher's own top-k goes in as the witness on the sampled rows:
        recomputing it from ``hidden`` and ``lse`` must reproduce ``witness_lp``,
        and does not if the entry is ever paired with the wrong row.
        """
        from verl.workers.teacher_cache import get_teacher_cache, row_fingerprint

        cache = get_teacher_cache()
        if self._teacher_lm_head_task is None:
            raise RuntimeError("teacher lm_head was never registered; see _register_teacher_lm_head")
        task = self._teacher_lm_head_task

        # Keying per position instead -- repeating the row's id across its
        # positions -- would leave each row with whichever position was written
        # last, and silently: the witness stored under the same key collapses with
        # it and stays self-consistent.
        #
        # Only the response positions that are actually trained are kept.
        # response_length is the cap, not the length: the mask over the same
        # window the forward scored, [-response_length-1:-1], says which slots are
        # real, and the rest is padding the loss never reads. At 512 cap against
        # ~127 generated tokens, storing it padded costs about 4x.
        #
        # The temperature travels with the entry: ``lse`` normalises logits the
        # forward already divided, while ``hidden`` is raw, so the read side has to
        # redo the division. Same value this call passed in meta_info.
        live_mask = fingerprints = None
        if attention_mask is not None:
            resp_len = lse.shape[1]
            live_mask = attention_mask[:, -resp_len - 1 : -1].bool()
            # What this row IS, so the actor can check that the key it holds names
            # the row it is training. The key alone is taken on trust otherwise.
            fingerprints = row_fingerprint(input_ids, attention_mask).to("cpu")
        cache.put(
            cache_ids.to("cpu"),
            task,
            hidden.detach(),
            lse.detach(),
            witness_rows=witness_rows,
            witness_ids=None if witness_ids is None else witness_ids.detach(),
            witness_lp=None if witness_lp is None else witness_lp.detach(),
            temperature=self.config.rollout.temperature,
            live_mask=live_mask,
            fingerprints=fingerprints,
        )

    def _register_teacher_lm_head(self, task: str, slot=None, n_tasks=None):
        """Hand the process cache an unsharded copy of this teacher's projection.

        The ref path reshards after every call, so by the time the actor update runs
        the parameter cannot be indexed at arbitrary ids. One 622 MB copy per teacher
        for a 1.7B model, held for the run -- the alternative is an all-gather of the
        whole teacher inside every micro-batch.

        ``slot``/``n_tasks`` put the copy straight into its slice of the stacked
        projection the lookup reads. Cloning first and stacking later would hold
        both layouts at once -- another ~1.9 GB, peaking before vLLM has sized its
        KV cache, which is exactly when free memory is being measured.
        """
        from verl.workers.teacher_cache import get_teacher_cache

        module = self.ref_policy.actor_module
        with FSDP.summon_full_params(module, writeback=False, offload_to_cpu=False):
            inner = module
            for _ in range(8):
                nxt = getattr(inner, "_fsdp_wrapped_module", None) or getattr(inner, "_orig_mod", None)
                if nxt is None or nxt is inner:
                    break
                inner = nxt
            head = getattr(inner, "lm_head", None)
            if head is None:
                raise RuntimeError("teacher has no lm_head; student_indexed_topk cannot resolve its log-probs")
            # Kept at the dtype the FORWARD projected in, not the dtype
            # summon_full_params hands back. FSDP keeps float32 masters and casts
            # to param_dtype for the forward, so the summoned weight is float32
            # while lm_head actually ran in bfloat16. Recomputing from the float32
            # copy is a different projection: the difference is ~eps_bf16 * |logit|,
            # about 0.25 nats at the logit magnitudes this model reaches, which is
            # what the witness caught on the first real run. Casting also halves
            # what is held -- 1.9 GB across three teachers rather than 3.7.
            weight = head.weight.detach().to(_fsdp_param_dtype(module, head.weight.dtype))
            if n_tasks is None:
                get_teacher_cache().register_lm_head(task, weight.clone())
            else:
                get_teacher_cache().register_lm_head(task, weight, slot=slot, n_tasks=n_tasks)
        self._teacher_lm_head_task = task

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def register_teacher_lm_head(self, task: str, slot=None, n_tasks=None):
        assert self._is_ref
        self._register_teacher_lm_head(task, slot=slot, n_tasks=n_tasks)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_teacher_hidden_cache(self):
        """Drop the previous step's entries. Called once per step by the trainer."""
        from verl.workers.teacher_cache import get_teacher_cache

        get_teacher_cache().clear()
        self._teacher_witness_budget = int(os.environ.get("TEACHER_WITNESS_MICRO_BATCHES", "2"))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_teacher_hidden_cache(self, atol: float = 1e-3):
        """Run the witness over everything cached this step. Raises on failure.

        Returns the witness's worst error alongside what the cache is holding, so
        the size is on the same call rather than a second round trip. The sign-
        weighting arms cache four models per row instead of one, and that number
        is the first thing to look at when a step dies on memory.
        """
        from verl.workers.teacher_cache import get_teacher_cache

        cache = get_teacher_cache()
        worst = cache.check_witness(atol=atol)
        return {"witness_max_err": worst, "rows": len(cache), "bytes": cache.nbytes()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        self._save_cross_teacher_sidecar(local_path)
        dist.barrier()

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _save_cross_teacher_sidecar(self, local_path):
        """The parameter-free arm's accumulated state, beside the actor's.

        Written by rank 0 only, into the same directory and at the same step as
        the checkpoint it belongs to. It is training state rather than a
        diagnostic: resuming the parameters while starting the RMS, the
        normaliser and the reliability moments from zero puts the run back at
        cold start -- every weight 1 for two steps, then a scale rebuilt from a
        handful of positions -- with a step number in the hundreds on the logs
        and nothing in the metrics to say so.
        """
        from verl.trainer.ppo.cross_teacher_kl_weight import SIDECAR_NAME, sidecar_state

        state = getattr(self.actor, "cross_teacher_state", None)
        if state is None or dist.get_rank() != 0:
            return
        os.makedirs(local_path, exist_ok=True)
        torch.save(
            sidecar_state(**state, identity=self._cross_teacher_identity()),
            os.path.join(local_path, SIDECAR_NAME),
        )

    def _cross_teacher_identity(self) -> dict:
        """What the accumulated numbers are measured against.

        A resume that disagrees on any of it is not this run continued: the
        shifts are relative to one base checkpoint, the RMS is per teacher, the
        log-probs were normalised at one temperature, and every matrix here is
        indexed by task order.
        """
        xt = (self.config.actor.get("cross_teacher_kl_weight", None) or {})
        teachers = xt.get("teacher_paths", None) or {}
        return {
            "base_path": xt.get("base_path", None),
            "temperature": float(self.config.rollout.temperature),
            "task_order": list(getattr(self.actor, "cross_teacher_task_order", []) or []),
            # The teachers themselves, not just the base. Every delta is
            # log pi_teacher - log pi_base, so swapping ONE teacher checkpoint
            # changes the scale, the corroboration and the reliability for every
            # pair that teacher appears in -- and leaves the base path matching.
            "teacher_paths": {str(k): str(v) for k, v in sorted(dict(teachers).items())},
        }

    # ONE_TO_ALL, matching save_checkpoint above and the critic's own pair: the
    # same path goes to every rank and each loads its own shard. Without the
    # decorator RayWorkerGroup never binds the method at all -- _bind_worker_method
    # only copies over what carries MAGIC_ATTR -- so EVERY resume died with
    # "'RayWorkerGroup' object has no attribute 'load_checkpoint'", after the
    # models were built and the checkpoint located. save_checkpoint was
    # decorated, so runs looked healthy right up to the first restart.
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        # Handed to the actor rather than loaded here: the accumulators are
        # indexed by task and do not exist until the first batch names them, so
        # the restore happens where they are built.
        if getattr(self, "actor", None) is not None:
            from verl.trainer.ppo.cross_teacher_kl_weight import SIDECAR_NAME

            self.actor.cross_teacher_sidecar_path = os.path.join(local_path, SIDECAR_NAME)
            self.actor.cross_teacher_identity = self._cross_teacher_identity()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)


class CriticWorker(Worker):
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
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
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

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

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

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation="flash_attention_2", trust_remote_code=config.model.get("trust_remote_code", False))
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

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
        sharding_strategy = get_sharding_strategy(fsdp_mesh, self.config.model.fsdp_config)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
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
                forward_prefetch=bool(fsdp_config.get("forward_prefetch", False)),
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
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
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

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

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get('use_shm', False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

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

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh, self.config.model.fsdp_config)

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
                forward_prefetch=bool(self.config.model.fsdp_config.get("forward_prefetch", False)),
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

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

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

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
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

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
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
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

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def resume(self):
        return self.rollout.resume()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def offload(self):
        return self.rollout.offload()
