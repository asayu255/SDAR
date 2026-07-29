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
from collections import defaultdict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.reward_score import default_compute_score


# [EXPLAIN] `DAPORewardManager` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DAPORewardManager:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """The reward manager."""

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tokenizer = tokenizer
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.compute_score = compute_score or default_compute_score
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.reward_fn_key = reward_fn_key
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.overlong_buffer_cfg = overlong_buffer_cfg
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_resp_len = max_resp_len

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.overlong_buffer_cfg is not None:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    # [EXPLAIN] `__call__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __call__(self, data: DataProto, return_dict: bool = False):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "rm_scores" in data.batch.keys():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if return_dict:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return {"reward_tensor": data.batch["rm_scores"]}
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return data.batch["rm_scores"]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_extra_info = defaultdict(list)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        already_print_data_sources = {}

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(len(data)):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data_item = data[i]  # DataProtoItem

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_ids = data_item.batch["prompts"]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_length = prompt_ids.shape[-1]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_ids = data_item.batch["responses"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            eos_token = self.tokenizer.eos_token
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if response_str.endswith(eos_token):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                response_str = response_str[: -len(eos_token)]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            score: float
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(result, dict):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score = result["score"]
                # Store the information including original reward
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for key, value in result.items():
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    reward_extra_info[key].append(value)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                score = result

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward = score

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.overlong_buffer_cfg.enable:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                overlong_buffer_len = self.overlong_buffer_cfg.len
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                expected_len = self.max_resp_len - overlong_buffer_len
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                exceed_len = valid_response_length - expected_len
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                reward += overlong_reward
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if self.overlong_buffer_cfg.log:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            reward_tensor[i, valid_response_length - 1] = reward

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if data_source not in already_print_data_sources:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                already_print_data_sources[data_source] = 0

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if already_print_data_sources[data_source] < self.num_examine:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                already_print_data_sources[data_source] += 1
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[prompt]", prompt_str)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[response]", response_str)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                print("[ground_truth]", ground_truth)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if isinstance(result, dict):
                    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                    for key, value in result.items():
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        print(f"[{key}]", value)
                # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                else:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("[score]", score)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if return_dict:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return reward_tensor
