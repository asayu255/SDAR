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
Megatron Reward Model.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.pipeline_parallel import get_forward_backward_func
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.megatron.pipeline_parallel import make_batch_generator
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import broadcast_dict_tensor, pad_sequence_to_length
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.reward_model.base import BasePPORewardModel


# [EXPLAIN] `MegatronRewardModel` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class MegatronRewardModel(BasePPORewardModel):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        config,
        model_config,
        reward_model_module: torch.nn.ModuleList,
        hf_config,
        tf_config,
        sft_tokenizer=None,
        rm_tokenizer=None,
    ):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_model_module = reward_model_module
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hf_config = hf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tf_config = tf_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.model_config = model_config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.device = "cuda"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sft_tokenizer = sft_tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rm_tokenizer = rm_tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.use_different_tokenizer = rm_tokenizer is not None

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"MegatronRewardModel.config: {self.config}")

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.megatron.param_offload:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.offload_params_to_cpu()

    # [EXPLAIN] `re_encode_by_rm_tokenizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def re_encode_by_rm_tokenizer(self, data: DataProto) -> DataProto:
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.use_different_tokenizer, "re-encode need rm tokenizer not be None!"
        # need to use rm tokenizer to re-generate input_ids, attention_mask and position_ids
        # 1. remove pad for each sequence
        # 2. decode by sft_tokenizer, remove sft system prompts
        # 3. encode by rm_tokenizer with rm system prompts, get rm_input_ids
        # 4. generate attention_mask and position_ids
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = data.batch["input_ids"]  # (bs, seq_len)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = data.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = data.batch["position_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ori_values = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        _, ori_seqlen = input_ids.size(0), input_ids.size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_for_rm = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask_for_rm = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_for_rm = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        print_decode = True
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ori_seqlen = ori_seqlen + 128
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for id, mask in zip(input_ids, attention_mask):
            # 1. remove pad for each sequence
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            non_zero_indices = torch.nonzero(mask).view(-1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            begin_pos, end_pos = non_zero_indices[0].item(), non_zero_indices[-1].item()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_id = id[begin_pos : end_pos + 1]
            # 2. decode by sft_tokenizer, remove sft system prompts
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            decode_result = self.sft_tokenizer.decode(valid_id)
            # workaround
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            decode_with_rm_chat = decode_result.replace("<|user|>\n", "[INST] ").replace("</s>\n<|assistant|>\n", " [/INST]").replace("</s> \n<|assistant|>\n", " [/INST]") + "</s>"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if print_decode and torch.distributed.get_rank() == 0:
                # only print first decode result
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(
                    f"device {torch.cuda.current_device()}: sft decode result:\n{decode_result}\n \
                        \ndevice {torch.cuda.current_device()}: sft decode result with \
                        rm chat template:\n{decode_with_rm_chat}\n\n"
                )
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                print_decode = False
            # 3. encode by rm_tokenizer
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_input_ids = self.rm_tokenizer(decode_with_rm_chat, return_tensors="pt")["input_ids"][0].to(input_ids.device)
            # 4. generate attention_mask and position_ids
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_attention_mask = torch.ones_like(rm_input_ids, device=input_ids.device)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_seqlen = rm_input_ids.shape[-1]
            # NOTE(gh): the later reward compute will process the shape (bs, seqlen_pad_128)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if cur_seqlen > ori_seqlen:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print(f"warninig: rm encode seqlen {cur_seqlen} > sft encode seqlen {ori_seqlen}")
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_input_ids = rm_input_ids[:ori_seqlen]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_attention_mask = rm_attention_mask[:ori_seqlen]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # right padding
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_input_ids = pad_sequence_to_length(rm_input_ids, ori_seqlen, self.rm_tokenizer.pad_token_id)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rm_attention_mask = pad_sequence_to_length(rm_attention_mask, ori_seqlen, 0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rm_position_ids = torch.arange(0, ori_seqlen, device=input_ids.device)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            input_ids_for_rm.append(torch.unsqueeze(rm_input_ids, dim=0))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            attention_mask_for_rm.append(torch.unsqueeze(rm_attention_mask, dim=0))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            position_ids_for_rm.append(torch.unsqueeze(rm_position_ids, dim=0))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_for_rm = torch.cat(input_ids_for_rm, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask_for_rm = torch.cat(attention_mask_for_rm, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_for_rm = torch.cat(position_ids_for_rm, dim=0)

        # (bs, seqlen) will not change, but input_ids, attention_mask and position_ids will change
        # NOTE(gh): need to replace into origin values after compute reward!
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["input_ids"] = input_ids_for_rm
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["attention_mask"] = attention_mask_for_rm
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        data.batch["position_ids"] = position_ids_for_rm

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data, ori_values

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad()
    # [EXPLAIN] `compute_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def compute_reward(self, data: DataProto) -> DataProto:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.megatron.param_offload:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.load_params_to_cuda()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_different_tokenizer:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = data.batch["input_ids"]  # (bs, seq_len')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = data.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = data.batch["position_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_token_len = data.meta_info.get("max_token_len", None)
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert micro_batch_size is not None, "micro batch size is needed for forward compute"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert max_token_len is not None, "use_dynamic_bsz is True, but max_token_len is None!"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            max_token_len = max_token_len * self.config.megatron.context_parallel_size

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        responses = data.batch["responses"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = responses.size(0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = responses.size(1)

        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.forward_batch(data, use_dynamic_bsz=use_dynamic_bsz, micro_batch_size=micro_batch_size, max_token_len=max_token_len)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits = torch.cat(output["output"], dim=0)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if use_dynamic_bsz:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    indices = output["indices"]
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    indices = list(itertools.chain.from_iterable(indices))
                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert len(indices) == logits.size(0), f"{len(indices)} vs. {logits.size()}"
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    logits = logits[revert_indices]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1]),
                    device=input_ids.device,
                )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = logits.to(torch.float32)

            # broadcast across pp ranks
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.distributed.broadcast(
                tensor=logits,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )

        # (bs, seqlen', hidden_size) -> (bs, seqlen', 1) -> (bs, seqlen')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_rewards = logits
        # find the last token reward
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # (bs, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        rewards = torch.gather(token_level_rewards, dim=1, index=ends)  # (bs, 1)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.use_different_tokenizer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            data.batch.update(ori_values)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = ori_values["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = ori_values["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = ori_values["position_ids"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_rewards = rewards.expand(attention_mask.shape[0], attention_mask.shape[1])  # (bs, ori_seqlen)

        # assign last valid token reward to ori position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bs,)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_mask = torch.zeros_like(attention_mask)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.0

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_rewards = token_level_rewards * eos_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_level_rewards = token_level_rewards[:, -response_length:]

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.config.megatron.param_offload:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.offload_params_to_cpu()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # add empty cache after each compute
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.empty_cache()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = TensorDict({"rm_scores": token_level_rewards}, batch_size=input_ids.shape[0])

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto(batch=batch)

    # [EXPLAIN] `forward_batch` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward_batch(self, data: DataProto, use_dynamic_bsz=False, micro_batch_size=None, max_token_len=None):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mini_batch = data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mini_batch.batch = mini_batch.batch.contiguous()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        broadcast_dict_tensor(mini_batch.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        indices = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if vpp_size is not None and vpp_size > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, num_batches_divided_by=microbatch_group_size_per_vp_stage, max_token_len=max_token_len)
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_seqlen = max_token_len
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert micro_batch_size is not None, "micro_batch_size is needed to be passed in when not using dynamic batch size"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = mini_batch.batch.split(micro_batch_size)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            seq_len = micro_batches[0]["input_ids"].shape[1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            total_seqlen = micro_batch_size * seq_len
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_micro_batch = len(micro_batches)

        # compute input shapes for pp stages
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        forward_backward_func = get_forward_backward_func()

        # [EXPLAIN] `loss_func` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def loss_func(output):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return 1.0, output

        # [EXPLAIN] `forward_step` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def forward_step(batch_iter, model):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            batch = next(batch_iter)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = batch["input_ids"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = batch["attention_mask"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = batch["position_ids"]
            # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
            from verl.models.mcore import get_mcore_forward_fn

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            forward_fn = get_mcore_forward_fn(self.hf_config)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return output, loss_func

        # batch should be a list of batches inside micro-batches
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.reward_model_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=True,
            )
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=True,
            )
        # loss_reduces contains the stats returned from loss_func
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        losses_reduced = {"output": losses_reduced}
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if use_dynamic_bsz:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            losses_reduced["indices"] = indices
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return losses_reduced

    # [EXPLAIN] `offload_params_to_cpu` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def offload_params_to_cpu(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device == "cuda":
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for reward_model_module in self.reward_model_module:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for name, param in reward_model_module.named_parameters():
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param.data = param.data.to("cpu", non_blocking=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.device = "cpu"
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.empty_cache()

    # [EXPLAIN] `load_params_to_cuda` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def load_params_to_cuda(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.device == "cpu":
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for reward_model_module in self.reward_model_module:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for name, param in reward_model_module.named_parameters():
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    param.data = param.data.to(torch.cuda.current_device(), non_blocking=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.device = "cuda"
