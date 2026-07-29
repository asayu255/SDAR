# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
Utilities for using tensor_parallel in megatron
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import TYPE_CHECKING, Dict

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core import parallel_state as mpu
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.nn import init

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if TYPE_CHECKING:
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import ModelParallelConfig


# [EXPLAIN] `update_kwargs_with_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def update_kwargs_with_config(dictionary: Dict, config: "ModelParallelConfig"):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    dictionary["config"] = config
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dictionary


# [EXPLAIN] `get_default_kwargs_for_model_parallel_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_default_kwargs_for_model_parallel_config():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return model_parallel_config_kwargs


# [EXPLAIN] `get_default_model_parallel_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_default_model_parallel_config():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import ModelParallelConfig

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


# [EXPLAIN] `get_common_default_kwargs_for_parallel_linear` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_common_default_kwargs_for_parallel_linear():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    default_model_parallel_config = get_default_model_parallel_config()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return common_default_kwargs


# [EXPLAIN] `get_default_kwargs_for_column_parallel_linear` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_default_kwargs_for_column_parallel_linear():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import ModelParallelConfig

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    common_default_kwargs.update(column_default_kwargs)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return common_default_kwargs


# [EXPLAIN] `get_default_kwargs_for_row_parallel_linear` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_default_kwargs_for_row_parallel_linear():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return common_default_kwargs


# [EXPLAIN] `get_default_kwargs_for_parallel_embedding` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_default_kwargs_for_parallel_embedding():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import ModelParallelConfig

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return embedding_default_kwargs


# [EXPLAIN] `is_tensor_parallel_param` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def is_tensor_parallel_param(param):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


# [EXPLAIN] `get_tensor_parallel_partition_dim` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_tensor_parallel_partition_dim(param):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert is_tensor_parallel_param(param)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return param.partition_dim


# [EXPLAIN] `get_tensor_parallel_partition_stride` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_tensor_parallel_partition_stride(param):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert is_tensor_parallel_param(param)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return param.partition_stride


# [EXPLAIN] `_VocabParallelEntropy` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class _VocabParallelEntropy(torch.autograd.Function):
    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
        @torch.compile(dynamic=True)
        # [EXPLAIN] `mul_reduce` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def mul_reduce(a, b):
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return (a * b).sum(dim=-1, keepdim=True)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return entropy.squeeze(dim=-1)

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @staticmethod
    # [EXPLAIN] `backward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        softmax_logits.mul_(vocab_parallel_logits)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        softmax_logits.mul_(-1)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return softmax_logits


# [EXPLAIN] `vocab_parallel_entropy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


# [EXPLAIN] `vocab_parallel_log_probs_from_logits` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def vocab_parallel_log_probs_from_logits(logits, labels):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import tensor_parallel

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


# [EXPLAIN] `vocab_parallel_log_probs_from_logits_response_rmpad` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from flash_attn.bert_padding import pad_input, unpad_input

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size, seqlen = input_ids.shape
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output
