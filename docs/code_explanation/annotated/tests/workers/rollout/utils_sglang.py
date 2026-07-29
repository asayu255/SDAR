# Copyright 2023-2024 SGLang Team
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
from datetime import timedelta

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.torch_functional import pad_sequence_to_length


# ====================== utils ======================
# [EXPLAIN] `levenshtein` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def levenshtein(s1, s2):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    m, n = len(s1), len(s2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(m + 1):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp[i][0] = i
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for j in range(n + 1):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dp[0][j] = j
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(1, m + 1):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for j in range(1, n + 1):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dp[m][n]


# [EXPLAIN] `are_lists_similar` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def are_lists_similar(a, b, threshold=10):
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
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        total_diff += levenshtein(s1, s2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    percentage_difference = (total_diff / total_length) * 100
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Total difference: {percentage_difference:.2f}%")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return percentage_difference <= threshold


# [EXPLAIN] `initialize_global_process_group` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def initialize_global_process_group(timeout_second=36000, spmd=False):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import torch.distributed

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not torch.distributed.is_initialized():  # Check if already initialized
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Initializing process group...")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("Process group already initialized.")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank = int(os.environ["LOCAL_RANK"])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rank = int(os.environ["RANK"])
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    world_size = int(os.environ["WORLD_SIZE"])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    torch.cuda.set_device(local_rank)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not CUDA_VISIBLE_DEVICES:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if spmd:
            # CUDA_VISIBLE_DEVICES = ','.join(str(i) for i in range(tensor_parallel_size))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            CUDA_VISIBLE_DEVICES = ",".join(str(i) for i in range(world_size))
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            CUDA_VISIBLE_DEVICES = str(local_rank)
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"CUDA_VISIBLE_DEVICES is not set, set to {CUDA_VISIBLE_DEVICES}")

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return local_rank, rank, world_size


# [EXPLAIN] `clean_torchelastic_env` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def clean_torchelastic_env():
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if k in os.environ:
            # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
            del os.environ[k]


# [EXPLAIN] `load_tokenizer_and_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_tokenizer_and_model(local_model_path, dtype="bfloat16"):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer.pad_token = tokenizer.eos_token
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=getattr(torch, dtype), device_map="cuda")
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tokenizer, model


# [EXPLAIN] `prepare_inputs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def prepare_inputs(tokenizer, prompts, max_prompt_length):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_ids = pad_sequence_to_length(tokenized["input_ids"], max_prompt_length, pad_token_id, left_pad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    attention_mask = pad_sequence_to_length(tokenized["attention_mask"], max_prompt_length, pad_token_id=0, left_pad=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids = compute_position_id_with_mask(attention_mask)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position_ids = pad_sequence_to_length(position_ids, max_prompt_length, pad_token_id=0, left_pad=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_ids, attention_mask, position_ids


# [EXPLAIN] `generate_hf_output` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_hf_output(model, input_ids, attention_mask, tokenizer, max_response_length):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generation_config = GenerationConfig(do_sample=False)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        max_new_tokens=max_response_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        output_scores=False,
        return_dict_in_generate=True,
        use_cache=False,
    )
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seq = output.sequences
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response = seq[:, input_ids.shape[1] :]
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return tokenizer.batch_decode(response)


# [EXPLAIN] `get_rollout_config` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_config_path):
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
    rollout_config = OmegaConf.create(
        {
            "name": "sglang",
            "load_format": "dummy_dtensor",
            "enforce_eager": False,
            "free_cache_engine": False,
            "dtype": dtype,
            "gpu_memory_utilization": 0.5,
            "ignore_eos": False,
            "max_num_batched_tokens": 8192,
            "prompt_length": max_prompt_length,
            "response_length": max_response_length,
            "tensor_model_parallel_size": tensor_parallel_size,
            "multi_turn": {
                "max_turns": 4,
                "enable": True,
                "tool_config_path": tool_config_path,
                "format": "chatml",
            },
            "max_model_len": None,
            **sampling_params,
        }
    )

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return rollout_config
