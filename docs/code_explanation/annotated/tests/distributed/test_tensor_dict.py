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
import os

# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
os.environ["NCCL_DEBUG"] = "WARN"

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto, all_gather_data_proto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group


# [EXPLAIN] `test_all_gather_data_proto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_all_gather_data_proto():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_mesh = torch.distributed.device_mesh.init_device_mesh("cuda", mesh_shape=[2, 2], mesh_dim_names=["dp", "tp"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    global_rank = torch.distributed.get_rank()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    obs = torch.tensor([[1 * global_rank, 2 * global_rank + 1], [3 * global_rank, 4 * global_rank + 1]])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = ["a", "b"] if global_rank % 2 == 0 else ["b", "a"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    labels = np.array(labels, dtype=object)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    all_gather_data_proto(data=data, process_group=device_mesh.get_group("dp"))

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if global_rank == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_labels = ["a", "b", "a", "b"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif global_rank == 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_labels = ["b", "a", "b", "a"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif global_rank == 2:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_labels = ["a", "b", "a", "b"]
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif global_rank == 3:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device="cuda")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        expected_labels = ["b", "a", "b", "a"]

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(data.batch["obs"], expected_obs, atol=0, rtol=0)
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert (data.non_tensor_batch["labels"] == expected_labels).all()
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert data.meta_info == {"info": "test_info"}


# [EXPLAIN] `test_vocab_parallel_entropy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_vocab_parallel_entropy():
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from megatron.core import parallel_state as mpu

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.debug import log_gpu_memory_usage
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.torch_functional import entropy_from_logits

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mpu.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 128
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_size = 155136

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits = torch.randn(batch_size * seqlen, vocab_size, device="cuda", requires_grad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target = torch.randint(low=0, high=vocab_size, size=(batch_size * seqlen,), device="cuda", dtype=torch.int64)

    # broadcast across tp
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.broadcast(logits, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.broadcast(target, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_rank = mpu.get_tensor_model_parallel_rank()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_size_per_tp = vocab_size // mpu.get_tensor_model_parallel_world_size()

    # get the local logits of each tp
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_parallel_logits = logits.clone().detach()[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp].requires_grad_()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits.grad = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vocab_parallel_logits.grad = None

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log_gpu_memory_usage("begin")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log_gpu_memory_usage("after forward")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    grad_output = torch.randn_like(output_entropy)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    output_entropy.backward(grad_output)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log_gpu_memory_usage("after backward")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    target_entropy = entropy_from_logits(logits)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(output_entropy, target_entropy)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    target_entropy.backward(grad_output)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(logits.grad[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits.grad)
    # make sure logits is not altered
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(logits[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if mpu.get_tensor_model_parallel_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("test_vocab_parallel_entropy passes")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mpu.destroy_model_parallel()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank, rank, world_size = initialize_global_process_group()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_all_gather_data_proto()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_vocab_parallel_entropy()
