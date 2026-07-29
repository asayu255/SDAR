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
import contextlib
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from dataclasses import dataclass

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import pytest
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed import init_device_mesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoModelForCausalLM, LlamaConfig, PretrainedConfig, Qwen2Config

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.models.transformers.monkey_patch import apply_monkey_patch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.protocol import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask, create_random_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
    ulysses_pad_and_slice_inputs,
)
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

# TODO(sgm): add more models for test
# we only need one scale for each model


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@dataclass
# [EXPLAIN] `SequenceParallelConfig` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SequenceParallelConfig:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    config: PretrainedConfig
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    sp_size: int
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    is_valid: bool


# [EXPLAIN] `test_configs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_configs():
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return [
        SequenceParallelConfig(LlamaConfig(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=32), sp_size=8, is_valid=True),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584),
            sp_size=4,
            is_valid=True,
        ),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584),
            sp_size=8,
            is_valid=False,
        ),
        SequenceParallelConfig(Qwen2Config(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4), sp_size=4, is_valid=True),
        SequenceParallelConfig(Qwen2Config(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4), sp_size=8, is_valid=True),
    ]


# [EXPLAIN] `sync_model_parameters_global` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def sync_model_parameters_global(layer):
    # synchronize weights
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for p in layer.parameters():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.broadcast(tensor=p.data, src=0)


# [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
@pytest.mark.parametrize("test_config", test_configs())
# [EXPLAIN] `test_hf_casual_fwd_bwd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_hf_casual_fwd_bwd(test_config):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not torch.distributed.is_initialized():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        initialize_global_process_group()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    context = contextlib.nullcontext() if test_config.is_valid else pytest.raises(AssertionError)
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with context:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        world_size = torch.distributed.get_world_size()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        _hf_casual_fwd_bwd(test_config.config, test_config.sp_size, world_size // test_config.sp_size)

    # TODO: seems not work, will cause `socketStartConnect: Connect to xxx failed : Software caused connection abort`
    # torch.distributed.destroy_process_group()


# [EXPLAIN] `_hf_casual_fwd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _hf_casual_fwd(config, sp_size, dp_size):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 128
    # response_length = 127

    # patch before load
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.device("cuda"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        apply_monkey_patch(model, sp_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = model.to(device="cuda")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sync_model_parameters_global(model)

    # different rank will generate different input_ids following fsdp
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = create_random_mask(input_ids=input_ids, max_ratio_of_left_padding=0, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.8)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids = compute_position_id_with_mask(attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_inputs = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.int().cuda(),
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_inputs = DataProto.from_dict(model_inputs)

    # 1. perform ulysses forward
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with sharding_manager:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_inputs = sharding_manager.preprocess_data(model_inputs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = model_inputs.batch["input_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = model_inputs.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = model_inputs.batch["position_ids"]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        # unpad the position_ids to align the rotary
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

        # slice input tensor for ulysses
        # input_ids are padded and sliced
        # postition_ids are only padded but not sliced
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded, use_cache=False).logits  # (1, total_nnz/n, vocab_size)

        # all_gather output
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

    # 2. perform normal forward
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    set_ulysses_sequence_parallel_group(None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits_rmpad_local = model(input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False).logits  # (1, total_nnz, vocab_size)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean_local = logits_rmpad_local.mean()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean_full = logits_full.mean()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=1e-5)


# [EXPLAIN] `_hf_casual_fwd_bwd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _hf_casual_fwd_bwd(config, sp_size, dp_size):
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seqlen = 128
    # response_length = 127

    # patch before load
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.device("cuda"):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        apply_monkey_patch(model, sp_size)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model = model.to(device="cuda")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        sync_model_parameters_global(model)

    # different rank will generate different input_ids following fsdp
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = create_random_mask(input_ids=input_ids, max_ratio_of_left_padding=0, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.8)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids = compute_position_id_with_mask(attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_inputs = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.int().cuda(),
    }

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_inputs = DataProto.from_dict(model_inputs)

    # 1. perform ulysses forward
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with sharding_manager:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        model_inputs = sharding_manager.preprocess_data(model_inputs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = model_inputs.batch["input_ids"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = model_inputs.batch["attention_mask"]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = model_inputs.batch["position_ids"]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        # unpad the position_ids to align the rotary
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

        # slice input tensor for ulysses
        # input_ids are padded and sliced
        # postition_ids are only padded but not sliced
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded, use_cache=False).logits  # (1, total_nnz/n, vocab_size)

        # all_gather output
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

    # 2. perform normal forward
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    set_ulysses_sequence_parallel_group(None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids_full = copy.deepcopy(input_ids_rmpad)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids_full = copy.deepcopy(position_ids_rmpad)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model_no_sp = copy.deepcopy(model)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    logits_rmpad_local = model_no_sp(input_ids_full, position_ids=position_ids_full, use_cache=False).logits  # (1, total_nnz, vocab_size)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean_local = logits_rmpad_local.mean()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mean_full = logits_full.mean()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mean_full.backward()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    mean_local.backward()

    # 3. check the gradients
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    grad = model.model.layers[0].self_attn.q_proj.weight.grad
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    grad_full = model_no_sp.model.layers[0].self_attn.q_proj.weight.grad
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=1e-5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.testing.assert_close(grad, grad_full, atol=1e-2, rtol=1e-5)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    pytest.main([__file__, "-svv"])
