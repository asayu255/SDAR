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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.optimizer import OptimizerConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native


# [EXPLAIN] `get_megatron_optimizer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer.
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )


# TODO: add get_optimizer_param_scheduler(optimizer) to implement lr scheuler.
