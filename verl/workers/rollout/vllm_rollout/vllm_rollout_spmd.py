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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import queue
import time
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, ListConfig
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.utils.phase_timing import PhaseTimer
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.generation_output import assemble_generation_output, sampling_kwargs_for, seed_for_prompt
from verl.workers.rollout.token_pump import TokenPump

# Seeding each pumped request from its own prompt. Default on: it can only make
# the pumped path LESS run-to-run variable, and the pumped path is opt-in
# already. ROLLOUT_PUMP_SEED=0 restores vllm's shared generator.
_PUMP_SEED = os.environ.get("ROLLOUT_PUMP_SEED", "1").strip().lower() not in ("0", "false", "no", "off")

from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _ask_for_the_final_output_only(sampling_params) -> None:
    """One RequestOutput per request when it finishes, not one per step.

    Cumulative output -- the default when SamplingParams is built directly -- has
    step() return a fresh RequestOutput carrying the whole token list so far, for
    every resident request, on every step. The answer is the same either way; the
    allocation is not, and it lands on the thread that owns the engine.

    Guarded, because the enum has moved between vllm versions and a pump that
    allocates more is better than a rollout that cannot import.
    """
    try:
        from vllm.sampling_params import RequestOutputKind
    except ImportError:
        return
    sampling_params.output_kind = RequestOutputKind.FINAL_ONLY


# Where a generate_sequences call's seconds go, split at the engine boundary.
#
# Measured from outside, the call leaves the GPU idle for a fixed ~0.6 s
# whatever it carries -- and the worker's own legs (device transfer, the
# sharding manager's reshaping, detokenisation, the trip back) all time at
# 0.00. That put the cost inside this function, but "inside vllm.generate" and
# "inside the Python around it" are different problems with different fixes,
# and from the worker they are one number.
#
# So: the input list build, the engine call, and the output assembly are timed
# separately. The assembly in particular is a Python loop over every response
# followed by a pad-and-concatenate, which for 252 rows of up to 512 tokens is
# not obviously cheap.
_ROLLOUT_PHASE_TIMING = os.environ.get("ROLLOUT_TURN_TIMING", "0").strip().lower() in ("1", "true", "yes", "on")
_ROLLOUT_PHASES = PhaseTimer(
    "rollout-phases",
    ("build_inputs", "engine", "assemble"),
    every=int(os.environ.get("ROLLOUT_GEN_PHASE_EVERY", "50")),
    note="(engine = vllm; the other two are the Python around it)",
)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.get("enable_prefix_caching", True),
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        from verl.utils.engine_overlap import report_engine_overlap

        report_engine_overlap(
            engine_kwargs,
            explicit={
                "enable_chunked_prefill": config.enable_chunked_prefill,
                "max_num_seqs": config.max_num_seqs,
                "enforce_eager": config.enforce_eager,
            },
            engine=self.inference_engine,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        # Whether to ask the engine for the sampled token's log-prob. See
        # rollout.return_rollout_log_probs in ppo_trainer.yaml: the column is only
        # read by the rollout-vs-actor drift check, so recipes without an
        # old_log_prob phase pay for it and never look at it.
        self.return_rollout_log_probs = bool(config.get("return_rollout_log_probs", True))

        kwargs = dict(
            n=1,
            # 0 => just the sampled token's own log-prob (the actor recomputes the
            # rest). None => the engine returns no log-probs at all.
            logprobs=0 if self.return_rollout_log_probs else None,
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        # Pumped path (see pump_step); nothing is created until it is used.
        self._pump = None
        self._pump_done: "queue.Queue" = queue.Queue()
        self._pump_sampling_cache: Dict[Any, SamplingParams] = {}
        self._pump_name = str(torch.distributed.get_rank()) if torch.distributed.is_initialized() else "0"

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        _phase_t = time.perf_counter() if _ROLLOUT_PHASE_TIMING else None
        _phase_marks = {} if _ROLLOUT_PHASE_TIMING else None

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        sampling_override = sampling_kwargs_for(prompts.meta_info, self.config.val_kwargs)
        if sampling_override:
            # Empty means "leave the configured sampling params alone", which is
            # not the same as "override them with nothing" -- the caller's own
            # kwargs have to survive that case.
            kwargs = sampling_override

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id=lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}",lora_int_id=lora_int_id,lora_path="/simon-stub-path")] * batch_size

        if _phase_marks is not None:
            _phase_marks["build_inputs"] = time.perf_counter() - _phase_t
            _phase_t = time.perf_counter()

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            if _phase_marks is not None:
                _phase_marks["engine"] = time.perf_counter() - _phase_t
                _phase_t = time.perf_counter()

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = [] if self.return_rollout_log_probs else None
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if rollout_log_probs is not None:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            n = self.sampling_params.n if do_sample else 1

        # Outside the sampling-params context on purpose: everything left is
        # arithmetic on the ids the engine already returned, and the pumped
        # path runs the identical arithmetic on the driver.
        output = assemble_generation_output(
            idx=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_token_ids=response,
            non_tensor_batch=non_tensor_batch,
            eos_token_id=eos_token_id,
            pad_token_id=self.pad_token_id,
            response_length=self.config.response_length,
            n=n,
            rollout_log_probs=rollout_log_probs,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        if _phase_marks is not None:
            _phase_marks["assemble"] = time.perf_counter() - _phase_t
            _ROLLOUT_PHASES.record(_phase_marks)

        return output

    # -- pumped path -------------------------------------------------------
    #
    # generate_sequences above is one blocking call per batch: every row is
    # added, the engine is stepped until the last of them finishes, and only
    # then does the caller get anything back. The rows of a *different* batch
    # cannot join in the middle, so on the evaluation arm the engine spends the
    # tail of every call decoding a handful of stragglers on a mostly idle GPU.
    #
    # Below, the same engine is driven a step at a time by a thread that owns
    # it, and rows are handed in one at a time from whichever caller has them
    # ready. What is generated does not change -- same weights, same sampling
    # params, same prompt token ids -- but which requests share a decode step
    # does, and that moves reduction order. See token_pump.py.

    def _pump_sampling_params(self, override: Dict[str, Any]) -> SamplingParams:
        """The params generate_sequences would have used, for the same meta_info.

        Built through update_sampling_params so the pumped path cannot drift from
        the blocking one: same context manager, same configured defaults
        underneath, just captured instead of used in place.
        """
        key = tuple(sorted(override.items()))
        cached = self._pump_sampling_cache.get(key)
        if cached is None:
            with self.update_sampling_params(**override):
                cached = deepcopy(self.sampling_params)
            _ask_for_the_final_output_only(cached)
            self._pump_sampling_cache[key] = cached
        return cached

    def _pump_seeded(self, params: SamplingParams, prompt_token_ids: Sequence[int], row: int = 0) -> SamplingParams:
        """Pin this request's sampling to its prompt, not to its arrival order.

        Without a seed, vllm draws from one generator shared by whatever is
        resident, so the same prompt sampled in a different decode step draws
        differently -- and under the pump, which requests share a step is decided
        by arrival timing. Seeding from the prompt's own token ids and the row
        makes that one source of run-to-run drift go away. The row is required,
        not decorative -- see seed_for_prompt, where n-sample validation sends
        byte-identical prompts that a prompt-only seed would collapse.

        It does NOT make the pumped path reproducible. The logits still move with
        batch composition (the same reason merging changed generation), and this
        cannot touch that. It removes a second, avoidable source on top of it.

        Greedy needs no generator, so it is left alone -- as is n > 1, which the
        pump refuses anyway and which a shared seed would collapse into n copies
        of one sample.
        """
        if not _PUMP_SEED or params.n != 1:
            return params
        if not getattr(params, "temperature", 0):
            return params
        seeded = copy(params)
        seeded.seed = seed_for_prompt(prompt_token_ids, row)
        return seeded

    def _pump_refuse(self) -> Optional[str]:
        """Why this rollout cannot be pumped, or None if it can.

        Each reason is a case where a request is not independent of the others
        in its call, so handing them to the engine separately would not be the
        same computation:
          * tp > 1 -- the ranks of a tensor-parallel group must add the same
            requests in the same order or they deadlock on the next collective;
          * LoRA -- generate_sequences picks the adapter per call;
          * rollout log-probs -- the pump returns token ids, not logprobs.
        Multimodal is refused per submission, not here: the same rollout serves
        text-only rows fine.
        """
        if self.config.get("tensor_model_parallel_size", 1) != 1:
            return "tensor_model_parallel_size > 1"
        if self.lora_kwargs:
            return "LoRA adapters are configured"
        if self.return_rollout_log_probs:
            return "rollout log-probs are requested"
        return None

    def pump_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Take new requests, return whatever has finished. One RPC per round.

        Submitting and collecting share a call because a Ray actor runs its
        methods one at a time: two separate calls would take turns, and the
        collect would sit in front of the submit it was waiting for.
        """
        refusal = self._pump_refuse()
        if payload.get("handshake"):
            return {
                "refused": refusal,
                "pad_token_id": self.pad_token_id,
                "response_length": self.config.response_length,
                "finished": [],
                "failed": [],
                "in_flight": 0,
            }
        if refusal is not None:
            raise RuntimeError(f"pump_step called on a rollout that cannot be pumped: {refusal}")

        if payload.get("stop"):
            pump, self._pump = self._pump, None
            if pump is not None:
                pump.stop()
            # Drain what stopping just produced. Left in the queue, a completion
            # from this session would be handed to whoever asks next -- and the
            # next session's driver starts a fresh client, so nothing about the
            # id it is keyed by makes that impossible on its own.
            dropped = 0
            while True:
                try:
                    self._pump_done.get_nowait()
                except queue.Empty:
                    break
                dropped += 1
            return {"finished": [], "failed": [], "in_flight": 0, "stopped": True, "dropped": dropped}

        if self._pump is None:
            self._pump = TokenPump(self.inference_engine.llm_engine, name=f"token-pump-{self._pump_name}").start()

        finished: List[Any] = []
        failed: List[Any] = []

        for request_id, prompt_token_ids, meta_info in payload.get("submit", ()):
            # The worker derives the sampling params, not the driver: they come
            # from the same meta_info and the same config the blocking path reads,
            # through the same context manager, so the two paths cannot drift.
            params = self._pump_sampling_params(sampling_kwargs_for(dict(meta_info), self.config.val_kwargs))
            if params.n != 1:
                # The driver only offers calls whose meta_info pins n=1. Reaching
                # here means it stopped doing that, and quietly keeping sample 0
                # of n would be a scoring change nobody asked for.
                failed.append((request_id, f"pumped path cannot serve n={params.n}; it returns one sequence per request"))
                continue
            row = int(dict(meta_info).get("row", 0))
            future = self._pump.submit(prompt_token_ids,
                                       self._pump_seeded(params, prompt_token_ids, row),
                                       request_id=request_id)
            future.add_done_callback(lambda f, rid=request_id: self._pump_done.put((rid, f)))

        timeout_s = float(payload.get("timeout_s", 0.02))
        try:
            # Block for the first completion so an idle round costs one wait
            # rather than a spin, then take everything else already sitting there.
            item = self._pump_done.get(timeout=timeout_s)
        except queue.Empty:
            item = None
        while item is not None:
            request_id, future = item
            error = future.exception()
            if error is None:
                finished.append((request_id, list(future.result())))
            else:
                failed.append((request_id, f"{type(error).__name__}: {error}"))
            try:
                item = self._pump_done.get_nowait()
            except queue.Empty:
                item = None

        return {
            "finished": finished,
            "failed": failed,
            "in_flight": self._pump.in_flight(),
            "steps": self._pump.steps,
        }


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
