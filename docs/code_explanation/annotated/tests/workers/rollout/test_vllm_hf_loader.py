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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm import SamplingParams

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.third_party.vllm import LLM, vllm_version
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import pad_sequence_to_length
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs


# [EXPLAIN] `levenshtein` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def levenshtein(s1, s2):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(m + 1):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp[i][0] = i  # Deletion from s1 to empty string
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for j in range(n + 1):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(1, m + 1):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for j in range(1, n + 1):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dp[m][n]


# [EXPLAIN] `are_lists_similar` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def are_lists_similar(a, b):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if len(a) != len(b):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("The lists are of different lengths.")
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return False

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_length = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    total_diff = 0

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for s1, s2 in zip(a, b):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_len = max(len(s1), len(s2))
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        total_length += max_len
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        diff = levenshtein(s1, s2)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        total_diff += diff
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Comparing strings:\n{s1}\n{s2}\nDifference: {diff} characters\n")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    percentage_difference = (total_diff / total_length) * 100
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Total difference: {percentage_difference:.2f}%")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return percentage_difference <= 10


# [EXPLAIN] `test_vllm_with_hf` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_vllm_with_hf():
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."

    # fill rollout config
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = 16
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = 16

    # Initialize model and token
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = "~/.cache/verl/rlhf"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = os.path.expanduser(local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_path = "deepseek-ai/deepseek-llm-7b-chat"
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.fs import copy_to_local

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer.pad_token = tokenizer.eos_token
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = prompts["input_ids"]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = prompts["attention_mask"]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    actor_model.to(torch.bfloat16)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_model_config = AutoConfig.from_pretrained(local_model_path)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    temperature = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_p = 1

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if vllm_version in (
        "0.5.4",
        "0.6.3",
    ):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        kwargs["detokenize"] = False
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sampling_params = SamplingParams(**kwargs)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_parallel_size = 4

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    llm = LLM(
        model=actor_model,
        tokenizer=tokenizer,
        model_hf_config=actor_model_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.1,
        load_format="hf",
    )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("start generation")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = input_ids.cuda()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = attention_mask.cuda()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    batch_size = input_ids.size(0)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    idx_list = []
    # parse idx from torch.Tensor to List[List[str]]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(batch_size):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        idx_list.append(_pre_process_inputs(tokenizer.pad_token_id, input_ids[i]))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = llm.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vllm_output = outputs[0].cuda()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    llm.free_cache_engine()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    llm = None
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import gc

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.empty_cache()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    gc.collect()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generation_config = GenerationConfig(do_sample=False)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    actor_model.cuda()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_response_length,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False,
    )  # may OOM when use_cache = True
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq = output.sequences
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = seq[:, max_prompt_length:]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_response_tokens = tokenizer.batch_decode(response)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vllm_response_tokens = tokenizer.batch_decode(vllm_output)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"hf response: {hf_response_tokens}")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"vllm response: {vllm_response_tokens}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert are_lists_similar(hf_response_tokens, vllm_response_tokens), "Strings differ more than 10%:\n"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Check Pass")


# if __name__ == "__main__":
#     test_vllm_with_hf()
