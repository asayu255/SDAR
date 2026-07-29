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
from torch.distributed.fsdp import CPUOffload, MixedPrecision
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoModelForCausalLM, AutoTokenizer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from vllm import LLM, SamplingParams

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import pad_sequence_to_length


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
    return percentage_difference <= 15


# [EXPLAIN] `test_vllm_spmd` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_vllm_spmd():
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and token
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = "~/.cache/verl/rlhf"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = os.path.expanduser(local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.fs import copy_to_local

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    actor_model.to(torch.bfloat16)

    # fill rollout config
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_prompt_length = 16
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    max_response_length = 32
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name?",
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

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("start generation")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = input_ids.cuda()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = attention_mask.cuda()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    temperature = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    top_p = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tensor_parallel_size = 4

    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.device_mesh import init_device_mesh

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    FSDP.set_state_dict_type(fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig())

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state_dict = fsdp_model.state_dict()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    sampling_params = SamplingParams(**kwargs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=1,
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    vllm_response_tokens = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for output in outputs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generated_text = output.outputs[0].text
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        vllm_response_tokens.append(generated_text)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size = torch.distributed.get_world_size()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    model.load_weights(((name, param.full_tensor() if world_size != 1 else param) for name, param in state_dict.items()))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    verl_vllm_response_tokens = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for output in outputs:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        generated_text = output.outputs[0].text
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        verl_vllm_response_tokens.append(generated_text)

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if torch.distributed.get_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"vllm response: {vllm_response_tokens}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"verl-vllm response: {verl_vllm_response_tokens}")
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert are_lists_similar(vllm_response_tokens, verl_vllm_response_tokens), "Strings differ more than 10%:\n"
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("Check Pass")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.distributed.destroy_process_group()


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_vllm_spmd()
