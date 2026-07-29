# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import pickle
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Any, List, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed as dist


# [EXPLAIN] `broadcast_pyobj` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rank == src:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(data) == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(tensor_size, src=src, group=dist_group)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            serialized_data = pickle.dumps(data)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            size = len(serialized_data)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(tensor_size, src=src, group=dist_group)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            dist.broadcast(tensor_data, src=src, group=dist_group)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast(tensor_size, src=src, group=dist_group)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        size = tensor_size.item()

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if size == 0:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dist.broadcast(tensor_data, src=src, group=dist_group)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        serialized_data = bytes(tensor_data.cpu().numpy())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        data = pickle.loads(serialized_data)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return data
