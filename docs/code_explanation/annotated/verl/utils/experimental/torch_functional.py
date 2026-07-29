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
from typing import Optional, Tuple
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch


# [EXPLAIN] `_fused_linear_for_ppo_fwd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = (hidden_states @ vocab_weights.t()) / temperature
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    orig_dtype = logits.dtype
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = logits.to(torch.float32)

    # Slower but more numerically stable to do log_softmax than probs.log()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    probs = logits.softmax(dim=-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_probs = logits.log_softmax(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return token_log_probs.to(orig_dtype), entropy.to(orig_dtype)


# [EXPLAIN] `_fused_linear_for_ppo_bwd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = (hidden_states @ vocab_weights.t()) / temperature
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    orig_dtype = logits.dtype
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = logits.to(torch.float32)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    probs = logits.softmax(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dlogits = 0

    # Gradient from log_probs
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dlog_probs is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

    # Gradient from entropy
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dentropy is not None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = logits.log_softmax(dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dlogits = dlogits.to(orig_dtype) / temperature

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dhidden_states = dlogits @ vocab_weights
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dvocab_weights = (dlogits.t() @ hidden_states)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dhidden_states, dvocab_weights


# [EXPLAIN] `FusedLinearForPPOFunction` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FusedLinearForPPOFunction(torch.autograd.Function):

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        orig_ndim = hidden_states.ndim
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        orig_batch_size = -1
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if orig_ndim == 3:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            orig_batch_size = hidden_states.shape[0]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_states = hidden_states.flatten(0, 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            input_ids = input_ids.flatten(0, 1)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        T = hidden_states.shape[0]

        # Allocate memory for outputs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_probs = hidden_states.new_zeros(T, requires_grad=output_requires_grad)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)

        # Perform forward one chunk at a time
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for chunk_start in range(0, T, chunk_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_end = min(chunk_start + chunk_size, T)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_probs[chunk_start:chunk_end] = chunk_log_probs
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            entropy[chunk_start:chunk_end] = chunk_entropy

        # Cast the output back to the original input dimension
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if orig_ndim == 3:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            log_probs = log_probs.view(orig_batch_size, -1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            entropy = entropy.view(orig_batch_size, -1)

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.orig_batch_size = orig_batch_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.orig_ndim = orig_ndim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.temperature = temperature
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ctx.chunk_size = chunk_size

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return log_probs, entropy

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert dlog_probs is not None or dentropy is not None

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        orig_batch_size = ctx.orig_batch_size
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        orig_ndim = ctx.orig_ndim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        temperature = ctx.temperature
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chunk_size = ctx.chunk_size

        # Here orig_ndim refers to the orig_ndim of hidden_states
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if orig_ndim == 3:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dlog_probs is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dlog_probs = dlog_probs.flatten()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dentropy is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dentropy = dentropy.flatten()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        T = hidden_states.shape[0]

        # Allocate memory for outputs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dhidden_states = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if hidden_states.requires_grad:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dhidden_states = torch.zeros_like(hidden_states)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dvocab_weights = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if vocab_weights.requires_grad:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dvocab_weights = torch.zeros_like(vocab_weights)

        # Perform backward one chunk at a time
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for chunk_start in range(0, T, chunk_size):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_end = min(chunk_start + chunk_size, T)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_dlog_probs = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dlog_probs is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            chunk_dentropy = None
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if dentropy is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hidden_states.requires_grad:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                dhidden_states[chunk_start:chunk_end] += h
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if vocab_weights.requires_grad:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                dvocab_weights += v

        # Cast the output back to the original input dimension
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if orig_ndim == 3 and hidden_states.requires_grad:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden_size = hidden_states.shape[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
        )


# [EXPLAIN] `FusedLinearForPPO` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class FusedLinearForPPO(torch.nn.Module):

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, chunk_size: int = 512):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.chunk_size = chunk_size

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = input_ids.to(torch.int64)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return FusedLinearForPPOFunction.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            self.chunk_size,
        )
