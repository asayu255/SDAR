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
from omegaconf import OmegaConf
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import CPUOffload, MixedPrecision
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from transformers import AutoModelForCausalLM, AutoTokenizer

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.fs import copy_to_local
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.model import compute_position_id_with_mask
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.workers.rollout.hf_rollout import HFRollout

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
BASE_HF_ROLLOUT_CONFIG = {
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1,
    "prompt_length": 64,
    "response_length": 64,
    "do_sample": True,
    "n": 1,
    "val_kwargs": {
        "top_k": -1,
        "top_p": 1.0,
        "temperature": 0,
        "n": 1,
        "do_sample": False,
    },
}


# [EXPLAIN] `prepare_input_dataproto` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def prepare_input_dataproto(tokenizer, config, validate):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    preencode_prompts = [
        [{"role": "user", "content": "Who won the Champions League in 2019?"}],
        [{"role": "user", "content": "The founder of Apple is"}],
        [{"role": "user", "content": "What's your name"}],
    ]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    formatted_prompts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in preencode_prompts]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    prompts = tokenizer(formatted_prompts, return_tensors="pt", padding="max_length", max_length=config.prompt_length)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input_dataproto = DataProto.from_dict(
        {
            "input_ids": prompts["input_ids"],
            "attention_mask": prompts["attention_mask"],
            "position_ids": compute_position_id_with_mask(prompts["attention_mask"]),
        },
        meta_info={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "validate": validate,
        },
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return input_dataproto


# [EXPLAIN] `prepare_fsdp_model` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def prepare_fsdp_model(model, world_size):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from torch.distributed.device_mesh import init_device_mesh

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_model = FSDP(
        model,
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
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return fsdp_model


# [EXPLAIN] `test_hf_rollout` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_hf_rollout(n: int = 1, do_sample: bool = True, validate: bool = False):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    config = OmegaConf.create(BASE_HF_ROLLOUT_CONFIG)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    config.update({"n": n, "do_sample": do_sample})

    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and tokenizer
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = "~/.cache/verl/rlhf"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_cache_path = os.path.expanduser(local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize FSDP model
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    actor_model.to(torch.bfloat16)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fsdp_model = prepare_fsdp_model(actor_model, world_size)

    # Initialize HFRollout and start generate
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    hf_rollout = HFRollout(fsdp_model, OmegaConf.create(config))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    input = prepare_input_dataproto(tokenizer, config, validate).to(torch.cuda.current_device())
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    outputs = hf_rollout.generate_sequences(input)

    # check generated batch size is expected
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    generated_batch_size = outputs.batch.batch_size[0]
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert generated_batch_size == input.batch.batch_size[0] * config.n

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(generated_batch_size):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_tokens = outputs.batch["prompts"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_mask = prompt_tokens != tokenizer.pad_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_tokens = prompt_tokens[prompt_mask]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_tokens = outputs.batch["responses"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_mask = response_tokens != tokenizer.pad_token_id
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_tokens = response_tokens[response_mask]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        attention_mask = outputs.batch["attention_mask"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position_ids = outputs.batch["position_ids"][i]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_length = outputs.batch["prompts"].size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_length = outputs.batch["responses"].size(1)

        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert attention_mask.size(0) == prompt_length + response_length
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert position_ids.size(0) == prompt_length + response_length

        # check response attention mask is expected
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_attention = attention_mask[prompt_length:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        eos_positions = (outputs.batch["responses"][i] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(eos_positions) > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            first_eos_pos = eos_positions[0].item()
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response_attention[: first_eos_pos + 1].all(), "Response attention mask should be 1 until EOS"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if first_eos_pos + 1 < response_length:
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert not response_attention[first_eos_pos + 1 :].any(), "Response attention mask should be 0 after EOS"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response_attention.all(), "Response attention mask should be all 1 if no EOS token"

        # check response position ids is expected
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prompt_positions = position_ids[:prompt_length]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        response_positions = position_ids[prompt_length:]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_response_length = min(len(response_tokens), response_length)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if valid_response_length > 0:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert response_positions[0] == prompt_positions[-1] + 1
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for j in range(1, valid_response_length):
                # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                assert response_positions[j] == response_positions[j - 1] + 1

        # print generated text for inspection
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if torch.distributed.get_rank() == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"prompt: {decoded_prompt}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print(f"response: {decoded_response}")
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print("=" * 30)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_hf_rollout(n=2, do_sample=True, validate=False)
    # test_hf_rollout(n=1, do_sample=False, validate=True)
    # test_hf_rollout(n=1, do_sample=True, validate=False)
