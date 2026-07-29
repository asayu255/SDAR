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
    -s test_sglang_async_spmd.py
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import asyncio

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sglang.srt.entrypoints.engine import Engine
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from sglang.srt.utils import broadcast_pyobj
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import init_device_mesh
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)


# [EXPLAIN] `_pre_process_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return token_ids


# [EXPLAIN] `test_sglang_spmd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_sglang_spmd():
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    initialize_global_process_group(spmd=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    clean_torchelastic_env()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = 16
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = 16

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = "Qwen/Qwen2.5-0.5B"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    preencode_prompts = ["Who won the Champions League in 2019?", "The founder of Apple is", "What's your name?"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids, attention_mask, _ = prepare_inputs(tokenizer, preencode_prompts, max_prompt_length)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_parallel_size = 2
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    inference_device_mesh_cpu = init_device_mesh("cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=["dp", "tp", "pp"])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tp_rank = inference_device_mesh_cpu["tp"].get_local_rank()

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if tp_rank == 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        llm = Engine(
            model_path=local_model_path,
            dtype="bfloat16",
            mem_fraction_static=0.5,
            enable_memory_saver=True,
            tp_size=inference_device_mesh_cpu["tp"].size(),
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_ids = input_ids.cuda()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx_list = []

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(input_ids.shape[0]):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sampling_params = dict(
            n=1,
            temperature=0,
            top_p=1,
            top_k=-1,
            max_new_tokens=max_response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            ignore_eos=False,
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        loop = asyncio.get_event_loop()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = loop.run_until_complete(llm.async_generate(input_ids=idx_list, sampling_params=sampling_params))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = None

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    [outputs] = broadcast_pyobj(
        [outputs],
        rank=inference_device_mesh_cpu["tp"].get_local_rank(),
        src=inference_device_mesh_cpu["tp"].mesh[0].item(),
        dist_group=inference_device_mesh_cpu["tp"].get_group(),
        force_cpu_device=False,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sglang_response_tokens = [output["text"] for output in outputs]

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"sglang response: {sglang_response_tokens}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens), "Strings differ more than 10%:\n"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("SPMD Test Passed!")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.barrier()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.destroy_process_group()
