#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=${NUM_GPUS:-8}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENGINE=${ENGINE:-vllm}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_FSDP_PARAM_OFFLOAD=${ACTOR_FSDP_PARAM_OFFLOAD:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_FSDP_OPTIMIZER_OFFLOAD=${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_FSDP_PARAM_OFFLOAD=${REF_FSDP_PARAM_OFFLOAD:-True}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_PAD=${RM_PAD:-True}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
FUSED_KERNELS=${FUSED_KERNELS:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
USE_KL=${USE_KL:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CUSTOM_REWARD_FN=${CUSTOM_REWARD_FN:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True} # For vLLM VLM placeholder issue: https://github.com/vllm-project/vllm/issues/15185
# LoRA config
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LORA_RANK=${LORA_RANK:-0}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
USE_SHM=${USE_SHM:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LOAD_FORMAT=${LOAD_FORMAT:-dummy_dtensor}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LAYERED_SUMMON=${LAYERED_SUMMON:-False}
# Validation
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RESUME_MODE=${RESUME_MODE:-disable}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SAVE_FREQ=${SAVE_FREQ:--1}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

# whether to save hf_model
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SAVE_HF_MODEL=${SAVE_HF_MODEL:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
FSDP_SIZE=${FSDP_SIZE:--1}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SP_SIZE=${SP_SIZE:-1}

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "${SAVE_HF_MODEL}" = "True" ]; then
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    CHECKPOINT_CONTENTS="['model','hf_model','optimizer','extra']"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
else
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    CHECKPOINT_CONTENTS="['model','optimizer','extra']"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_traj_micro_bsz_per_gpu=2 # b
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
n_resp_per_prompt=4 # g

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
reward_fn_name=null
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
reward_fn_file_path=null
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
output_file="$(pwd)/output.txt"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    reward_fn_name="my_reward_function"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    reward_fn_file_path="$(pwd)/my_reward_function.py"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    rm -rf "${reward_fn_file_path}"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    cat <<EOF > "$reward_fn_file_path"
def ${reward_fn_name}(data_source, solution_str, ground_truth, extra_info=None):
    print(f"Congratulations!!! You have called ${reward_fn_name} successfully!!!")
    return 0.1
EOF

    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    rm -rf "${output_file}"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="${VERL_EXP_NAME:-$(basename "${MODEL_ID,,}")-function-reward-minimal}"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_shm=${USE_SHM} \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.model.use_fused_kernels=${FUSED_KERNELS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_FSDP_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.checkpoint.contents=${CHECKPOINT_CONTENTS} \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.load_format=${LOAD_FORMAT} \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON} \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_FSDP_PARAM_OFFLOAD}" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    custom_reward_function.path="${reward_fn_file_path}"\
    custom_reward_function.name="${reward_fn_name}"\
    algorithm.use_kl_in_reward="${USE_KL}" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@ \
    | tee "${output_file}"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    python3 tests/e2e/check_custom_rwd_fn.py --output_file="${output_file}"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    check_exit_code=$?
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    rm -rf "${reward_fn_file_path}"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    rm -rf "${output_file}"
    # Return the exit code of check_custom_rwd_fn.py if it fails
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if [ $check_exit_code -ne 0 ]; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        exit $check_exit_code
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi
