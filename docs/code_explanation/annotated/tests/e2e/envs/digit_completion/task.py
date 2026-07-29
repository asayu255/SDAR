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
"""Task and environment definition for digit completion."""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np


# [EXPLAIN] `DigitCompletion` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class DigitCompletion:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    The implementation of a simple digit completion task.
    The prompt is a sequence of numbers with fixed difference. The task is to complete the next N numbers.
    If the max number is reached, the next number should be modulo with max number.

    For example,
    - prompt = [1, 2, 3]
    - N = 5
    - max_number = 6

    the response should be [4, 5, 6, 7%6, 8%6] = [4, 5, 6, 0, 1]

    Note that the tokenizer is char-level to increase the difficulty.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, max_number: int, max_diff: int, max_num_in_response: int, seed=0):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """

        Args:
            max_number: the maximum number allowed in the arithmetic sequence
            max_diff: the maximum diff. The actual common diff will be sampled from [0, max_diff]
            max_num_in_response: the maximum number in the response
        """
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_number = max_number
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_diff = max_diff
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_num_in_response = max_num_in_response
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.max_num_in_response < 10
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.max_number > 0
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert self.max_diff > 0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.max_number_length = len(str(max_number))
        # {num1},{num2}:{max_num_in_response},{max_number}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self._prompt_length = self.max_number_length * 2 + 4 + self.max_number_length  # no negative is allowed

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.np_rng = np.random.default_rng(seed=seed)

    # [EXPLAIN] `__str__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __str__(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return f"Prompt length: {self.prompt_length}. Response length: {self.response_length}, Max number: {self.max_number}. Max diff: {self.max_diff}, Max number in response: {self.max_num_in_response}"

    # [EXPLAIN] `get_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_state(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return {"rng": self.np_rng}

    # [EXPLAIN] `set_state` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_state(self, state):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert "rng" in state, "rng must be inside state"
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.np_rng = state["rng"]

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `prompt_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def prompt_length(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._prompt_length

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @property
    # [EXPLAIN] `response_length` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def response_length(self):
        # number length + comma length + [EOS]
        # The actual number times 1.5 to allow 'U'
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (self.max_num_in_response * self.max_number_length + (self.max_num_in_response - 1) + 1) * 2

    # [EXPLAIN] `add` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def add(self, a, b):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (a + b) % self.max_number

    # [EXPLAIN] `get_all_prompts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_all_prompts(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_prompts = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for first_num in range(self.max_number + 1):
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for diff in range(0, self.max_diff + 1):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                second_num = self.add(first_num, diff)
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for num_to_complete in range(self.max_num_in_response + 1):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    prompt = str(first_num) + "," + str(second_num) + f":{self.max_number},{num_to_complete}"
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    all_prompts.append(prompt)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return all_prompts

    # [EXPLAIN] `sample_str_prompts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample_str_prompts(self):
        # step 1: sample initial numbers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        first_num = self.np_rng.integers(self.max_number + 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        diff = self.np_rng.integers(self.max_diff + 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        second_num = self.add(first_num, diff)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_to_complete = self.np_rng.integers(self.max_num_in_response + 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt = str(first_num) + "," + str(second_num) + f":{self.max_number},{num_to_complete}"
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return prompt

    # [EXPLAIN] `sample_batch_str_prompts` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def sample_batch_str_prompts(self, batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        str_prompts = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for _ in range(batch_size):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            str_prompts.append(self.sample_str_prompts())
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return str_prompts


# [EXPLAIN] `compute_attention_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_attention_mask(prompts, pad_token_id):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = np.ones_like(prompts)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask[prompts == pad_token_id] = 0
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return mask


# [EXPLAIN] `compute_position_id_with_mask` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_position_id_with_mask(mask):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return np.clip(np.cumsum(mask, axis=-1) - 1, a_min=0, a_max=None)


# [EXPLAIN] `generate_ground_truth_response` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_ground_truth_response(prompt: str):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Generate ground truth response given a prompt."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num, info = prompt.split(":")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num1, num2 = num.split(",")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_number, num_to_gen = info.split(",")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num1 = int(num1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num2 = int(num2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_number = int(max_number)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_to_gen = int(num_to_gen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    diff = (num2 - num1) % max_number
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    results = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    last_num = num2
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for _ in range(num_to_gen):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        curr = (last_num + diff) % max_number
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        results.append(str(curr))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        last_num = curr
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = ",".join(results)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return response


# [EXPLAIN] `compute_reward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_reward(prompt: str, response: str, sequence_reward=1.0):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """We compute dense reward here so that we can directly train RL without SFT"""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = len(response)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_response = generate_ground_truth_response(prompt)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    per_token_reward = sequence_reward / (len(ground_truth_response) + 1)  # including [EOS]

    # pad
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward = np.zeros(response_length, dtype=np.float32)  # this assumes that each char is a token
    # assign reward until mismatches
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ground_truth_idx = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(response_length):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ground_truth_idx == len(ground_truth_response):
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ground_truth_response_token = ground_truth_response[ground_truth_idx]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_token = response[i]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if ground_truth_response_token == response_token:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward[i] = per_token_reward
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            ground_truth_idx += 1
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # no matches
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return reward, {"ground_truth_response": ground_truth_response}


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task = DigitCompletion(max_number=20, max_diff=3, max_num_in_response=5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(task.sample_str_prompts())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt = "7,8:20,0"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = ""
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(compute_reward(prompt, response))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt = "7,8:20,0"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = "E000"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(compute_reward(prompt, response))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompt = "9,10:20,2"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = "11,12,13"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(compute_reward(prompt, response))
