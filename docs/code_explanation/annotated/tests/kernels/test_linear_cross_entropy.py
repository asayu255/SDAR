#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
import typing

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.experimental.torch_functional import FusedLinearForPPO
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import logprobs_from_logits

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
fused_linear_for_ppo = FusedLinearForPPO()
# [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
fused_linear_for_ppo.compile(dynamic=True)


# [EXPLAIN] `run_torch_entropy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_torch_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction="none") -> typing.List[torch.Tensor]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden = hidden.squeeze(0).to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weight = weight.transpose(0, 1).to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.matmul(hidden, weight)  # [num_tokens, vocab_size]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy = entropy_a - entropy_b
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logprobs = torch.nn.functional.cross_entropy(logits, labels.squeeze(0), reduction=reduction)  # [num_tokens]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logprobs = torch.neg(logprobs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logprobs, entropy


# [EXPLAIN] `run_verl_original_entropy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_verl_original_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> typing.List[torch.Tensor]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden = hidden.squeeze(0).to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weight = weight.transpose(0, 1).to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.matmul(hidden, weight)  # [num_tokens, vocab_size]
    # compute entropy
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy = compute_entropy_from_logits(logits)  # ((total_nnz / sp) + pad)
    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logprobs = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=False)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logprobs, entropy


# To be tested
# [EXPLAIN] `run_verl_torch_fused_entropy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def run_verl_torch_fused_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hidden = hidden.to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    weight = weight.to(torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logprobs, entropy = fused_linear_for_ppo(
        hidden,
        weight,
        labels,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return logprobs.squeeze(0), entropy.squeeze(0)


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
MAX_TEST_CASES = 5


# [EXPLAIN] `TestLinearCrossEntropy` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TestLinearCrossEntropy:
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, test_case_idx: int) -> None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.test_case_idx = test_case_idx

    # [EXPLAIN] `cleanup` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def cleanup(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.empty_cache()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.reset_peak_memory_stats()
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import gc

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        gc.collect()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.synchronize()

    # [EXPLAIN] `generate_hyper` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_hyper(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dtype = torch.bfloat16
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.test_case_idx == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch_size = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_tokens = 1937
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hidden_size = 3584
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.vocab_size = 152064
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.test_case_idx == 1:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch_size = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_tokens = 2169
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hidden_size = 896
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.vocab_size = 151936
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.test_case_idx == 2:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch_size = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_tokens = 1530
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hidden_size = 2048
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.vocab_size = 32256
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.test_case_idx == 3:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch_size = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_tokens = 1388
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hidden_size = 4096
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.vocab_size = 102400
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif self.test_case_idx == 4:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.batch_size = 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.num_tokens = 8192
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.hidden_size = 4096
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.vocab_size = 102400
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise ValueError(f"Invalid test case index: {test_case_idx}")

    # [EXPLAIN] `generate_forward_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_forward_inputs(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden = torch.empty((self.batch_size, self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5).requires_grad_()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight = torch.empty((self.vocab_size, self.hidden_size), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5).requires_grad_()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.num_tokens), device="cuda")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return hidden, weight, labels

    # [EXPLAIN] `generate_backward_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def generate_backward_inputs(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        g_entropy = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        g_logprobs = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return g_entropy, g_logprobs

    # [EXPLAIN] `verify_correctness` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def verify_correctness(self, iterations=5):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.cleanup()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.generate_hyper()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_forward_latency = list()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_backward_latency = list()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_forward_latency = list()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_backward_latency = list()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_fused_forward_latency = list()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_fused_backward_latency = list()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        start_event = torch.cuda.Event(enable_timing=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        end_event = torch.cuda.Event(enable_timing=True)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(iterations):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"[INFO]: Iteration {i + 1} / {iterations}...", end="\r")
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden, weight, labels = self.generate_forward_inputs()

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (verl_logprobs, verl_entropy) = run_verl_original_entropy(hidden, weight, labels)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verl_forward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (verl_fused_logprobs, verl_fused_entropy) = run_verl_torch_fused_entropy(hidden, weight, labels)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verl_fused_forward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(torch_logprobs, verl_logprobs, atol=1e-4, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(torch_entropy, verl_entropy, atol=1e-4, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(torch_logprobs, verl_fused_logprobs, atol=1e-4, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(torch_entropy, verl_fused_entropy, atol=1e-4, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(verl_logprobs, verl_fused_logprobs, atol=1e-4, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(verl_entropy, verl_fused_entropy, atol=1e-4, rtol=1e-4)

            # backward
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            g_entropy, g_logprobs = self.generate_backward_inputs()

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch_backward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (d_verl_hidden, d_verl_weight) = torch.autograd.grad((verl_entropy, verl_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verl_backward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            start_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            (d_verl_fused_hidden, d_verl_fused_weight) = torch.autograd.grad((verl_fused_entropy, verl_fused_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            end_event.record()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.cuda.synchronize()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            verl_fused_backward_latency.append(start_event.elapsed_time(end_event))

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_torch_hidden, d_verl_hidden, atol=1e-2, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_torch_weight, d_verl_weight, atol=1e-2, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_torch_hidden, d_verl_fused_hidden, atol=1e-2, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_torch_weight, d_verl_fused_weight, atol=1e-2, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_verl_hidden, d_verl_fused_hidden, atol=1e-2, rtol=1e-4)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            torch.testing.assert_close(d_verl_weight, d_verl_fused_weight, atol=1e-2, rtol=1e-4)

        # remove first latency
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_forward_latency = torch_forward_latency[1:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_backward_latency = torch_backward_latency[1:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_forward_latency = verl_forward_latency[1:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_backward_latency = verl_backward_latency[1:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_fused_forward_latency = verl_fused_forward_latency[1:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        verl_fused_backward_latency = verl_fused_backward_latency[1:]

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("\n[INFO]: Verified forward & backward correctness.")

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Forward pass: Torch implementation average time: {sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Backward pass: torch implementation average time: {sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Forward pass: VeRL implementation average time: {sum(verl_forward_latency) / len(verl_forward_latency):.2f} ms")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Backward pass: VeRL implementation average time: {sum(verl_backward_latency) / len(verl_backward_latency):.2f} ms")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Forward pass: VeRL Fused Entropy implementation average time: {sum(verl_fused_forward_latency) / len(verl_fused_forward_latency):.2f} ms")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: Backward pass: VeRL Fused Entropy implementation average time: {sum(verl_fused_backward_latency) / len(verl_fused_backward_latency):.2f} ms")

    # [EXPLAIN] `check_storage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def check_storage(self, method_name, run_forward):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.cleanup()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.generate_hyper()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden, weight, labels = self.generate_forward_inputs()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.reset_peak_memory_stats()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        (logprobs, entropy) = run_forward(hidden, weight, labels)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.synchronize()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: {method_name} Forward pass peak memory: {torch_max_memory:.2f} MB")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        g_entropy, g_logprobs = self.generate_backward_inputs()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.reset_peak_memory_stats()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        (d_torch_hidden, d_torch_weight) = torch.autograd.grad((entropy, logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.cuda.synchronize()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        torch_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO]: {method_name} Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

    # [EXPLAIN] `check_storage_all` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def check_storage_all(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.check_storage("Torch", run_torch_entropy)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.check_storage("VeRL", run_verl_original_entropy)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.check_storage("VeRL Torch Fused", run_verl_torch_fused_entropy)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history()

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for test_case_idx in range(MAX_TEST_CASES):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"[INFO] Running test case {test_case_idx}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        test = TestLinearCrossEntropy(test_case_idx)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        test.verify_correctness()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        test.check_storage_all()

    # torch.cuda.memory._dump_snapshot("test_linear_cross_entropy.pkl")
