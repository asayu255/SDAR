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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from contextlib import contextmanager
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import List

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import DictConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm import SamplingParams

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
# [EXPLAIN] `_pre_process_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return token_ids


# [EXPLAIN] `FIREvLLMRollout` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FIREvLLMRollout(vLLMRollout):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_fire_sampling = config.get("use_fire_sampling", False)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_fire_sampling:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs_0 = kwargs.copy()
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs_0["temperature"] = 30
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            kwargs_0["max_tokens"] = 1
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "top_k" not in kwargs_0 or kwargs_0["top_k"] <= 0:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                kwargs_0["top_k"] = 16
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sampling_params.max_tokens = config.response_length - 1
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for k in config.keys():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hasattr(SamplingParams(), str(k)):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    kwargs_0[k] = config.get(k)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.sampling_params_0 = SamplingParams(**kwargs_0)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @contextmanager
    # [EXPLAIN] `update_sampling_params` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update_sampling_params(self, **kwargs):
        # update sampling params
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        old_sampling_params_args = {}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if kwargs:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, value in kwargs.items():
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if hasattr(self.sampling_params, key):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    old_value = getattr(self.sampling_params, key)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    old_sampling_params_args[key] = old_value
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    setattr(self.sampling_params, key, value)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_fire_sampling:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            old_sampling_params_args_0 = {}
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if kwargs:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key, value in kwargs.items():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if hasattr(self.sampling_params_0, key):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_value = getattr(self.sampling_params_0, key)
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        old_sampling_params_args_0[key] = old_value
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        setattr(self.sampling_params_0, key, value)
        # [EXPLAIN] 現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for key, value in old_sampling_params_args.items():
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            setattr(self.sampling_params, key, value)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_fire_sampling:
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for key, value in old_sampling_params_args_0.items():
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                setattr(self.sampling_params_0, key, value)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad()
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.free_cache_engine:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.init_cache_engine()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = prompts.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_token_id = prompts.meta_info["eos_token_id"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = idx.size(0)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(batch_size):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        do_sample = prompts.meta_info.get("do_sample", True)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not do_sample:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.use_fire_sampling:
            # users can customize different sampling_params at different run
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with self.update_sampling_params(**kwargs):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False,
                )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = output[0].to(idx.device)  # (bs, response_length)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with self.update_sampling_params(**kwargs):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output_0 = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params_0,
                    prompt_token_ids=idx_list,
                    use_tqdm=False,
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_idx_list = []
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for i in range(batch_size):
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    new_idx_list.append(idx_list[i] + output_0[0][i].tolist())
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=new_idx_list,
                    use_tqdm=False,
                )

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = torch.cat([output_0[0], output[0]], dim=1).to(idx.device)  # (bs, response_length)
            # log_probs = torch.cat([output_0[1], output[1]], dim=1).to(idx.device)  # (bs, response_length)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if response.shape[1] < self.config.response_length:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.n > 1 and do_sample:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx = idx.repeat_interleave(self.config.n, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch_size = batch_size * self.config.n
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq = torch.cat([idx, response], dim=-1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = response.size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_position_ids = position_ids[:, -1:] + delta_position_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.free_cache_engine:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.inference_engine.free_cache_engine()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto(batch=batch)
