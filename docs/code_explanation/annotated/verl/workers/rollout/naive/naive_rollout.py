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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch import nn

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import logprobs_from_logits

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from ..base import BaseRollout

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
__all__ = ["NaiveRollout"]


# [EXPLAIN] `NaiveRollout` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class NaiveRollout(BaseRollout):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, module: nn.Module, config):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits field.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.config = config
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.module = module

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @torch.no_grad()
    # [EXPLAIN] `generate_sequences` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Generate sequences"""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_token_id = prompts.meta_info["eos_token_id"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_size = idx.size(0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = idx.size(1)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.module.eval()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_lst = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _ in range(self.config.response_length):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            # we use huggingface APIs here
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            logits = logits[:, -1, :] / self.config.temperature  # (bs, vocab_size)
            # optionally crop the logits to only the top k options
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.top_k is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.config.do_sample:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idx_next = torch.multinomial(probs, num_samples=1)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for token_id in eos_token_id:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            prev_attention_mask.to(attention_mask.dtype)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # append sampled index to the running sequence and continue
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            idx = torch.cat((idx, idx_next), dim=1)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            logits_lst.append(logits)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = idx[:, :prompt_length]  # (bs, prompt_length)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response = idx[:, prompt_length:]  # (bs, response_length)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch = TensorDict(
            {
                "input_ids": prompts,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.module.train()

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return DataProto(batch=batch)
