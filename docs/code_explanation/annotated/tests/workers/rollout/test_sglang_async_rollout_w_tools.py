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
# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=2 $(which pytest) \
    -s test_sglang_async_rollout_w_tools.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import init_device_mesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager


# [EXPLAIN] `test_async_sglang_rollout_w_tool` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_async_sglang_rollout_w_tool():
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    initialize_global_process_group()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    clean_torchelastic_env()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = 32
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = 16
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dtype = "bfloat16"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_parallel_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = "Qwen/Qwen2.5-0.5B"

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    preencode_prompts = [
        [{"role": "user", "content": prompt, "tool_calls": None}]
        for prompt in [
            "Who won the Champions League in 2019?",
            "The founder of Apple is",
            "What's the best way to learn python?",
        ]
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_device_mesh = init_device_mesh("cuda", mesh_shape=(tensor_parallel_size,), mesh_dim_names=("fsdp",))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inference_device_mesh_cpu = init_device_mesh("cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=("dp", "infer_tp", "pp"))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, None)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rollout = SGLangRollout(actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rollout_sharding_manager = FSDPSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout._engine,
        model_config=actor_model.config,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with rollout_sharding_manager:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"preprocessed {input_ids.shape=}")

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        messages = np.asarray(preencode_prompts)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages})

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        prompts.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompts = rollout_sharding_manager.preprocess_data(prompts)
        # log_gpu_memory_usage("Before generating sequences", logger=None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = rollout.generate_sequences(prompts=prompts)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"generated {output.batch['responses'].shape=}")
        # log_gpu_memory_usage("After generating sequences", logger=None)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = rollout_sharding_manager.postprocess_data(output)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"postprocessed {output.batch['responses'].shape=}")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sglang_output = output.to("cpu")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"hf response: {hf_response_tokens}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"sglang response: {sglang_response_tokens}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("SGLang w tool Test Passed!")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.barrier()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.destroy_process_group()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_async_sglang_rollout_w_tool()
