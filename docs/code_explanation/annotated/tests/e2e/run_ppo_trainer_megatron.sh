#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=${NUM_GPUS:-8}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
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

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ppo_max_token_len_per_gpu=2400
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
forward_max_token_len_per_gpu=4800
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
COMMON_PP=${COMMON_PP:-2}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_VPP=${COMMON_VPP:-2}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_CP=${COMMON_CP:-2}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_TP=${COMMON_TP:-2}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
INFER_TP=${INFER_TP:-$COMMON_TP}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ROLLOUT_TP=${ROLLOUT_TP:-$INFER_TP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_PP=${REF_PP:-$COMMON_PP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_VPP=${REF_VPP:-$COMMON_VPP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_CP=${REF_CP:-$COMMON_CP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_TP=${REF_TP:-$TRAIN_TP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_TP=${CRITIC_TP:-$TRAIN_TP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_PP=${RM_PP:-$COMMON_PP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_VPP=${RM_VPP:-$COMMON_VPP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_CP=${RM_CP:-$COMMON_CP}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_TP=${RM_TP:-$TRAIN_TP}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ALL_OFFLOAD=${ALL_OFFLOAD:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CHECKPOINT_CONTENTS=['model','hf_model','optimizer','extra']
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SKIP_SAVE_HF_MODEL=${SKIP_SAVE_HF_MODEL:-0}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ $SKIP_SAVE_HF_MODEL -eq 1 ]; then
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    CHECKPOINT_CONTENTS=['model','optimizer','extra']
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENGINES=("vllm" "sglang")

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="$(basename "${MODEL_ID,,}")-megatron-gsm8k-minimal"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
for ENGINE in "${ENGINES[@]}"; do
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml'\
        algorithm.adv_estimator="${ADV_ESTIMATOR}" \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VAL_FILES}" \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=512 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$ACTOR_PP \
        actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$ACTOR_VPP \
        actor_rollout_ref.actor.megatron.context_parallel_size=$ACTOR_CP \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$ACTOR_TP \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.checkpoint.contents=$CHECKPOINT_CONTENTS \
        actor_rollout_ref.rollout.name="${ENGINE}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$REF_PP \
        actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=$REF_VPP \
        actor_rollout_ref.ref.megatron.context_parallel_size=$REF_CP \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$REF_TP \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        critic.optim.lr=2e-5 \
        critic.model.path="${MODEL_PATH}" \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        critic.ppo_max_token_len_per_gpu=${forward_max_token_len_per_gpu} \
        critic.megatron.pipeline_model_parallel_size=$CRITIC_PP \
        critic.megatron.virtual_pipeline_model_parallel_size=$CRITIC_VPP \
        critic.megatron.context_parallel_size=$CRITIC_CP \
        critic.megatron.tensor_model_parallel_size=$CRITIC_TP \
        critic.checkpoint.contents=$CHECKPOINT_CONTENTS \
        reward_model.enable=True \
        reward_model.model.path="${MODEL_PATH}" \
        reward_model.micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        reward_model.megatron.pipeline_model_parallel_size=$RM_PP \
        reward_model.megatron.virtual_pipeline_model_parallel_size=$RM_VPP \
        reward_model.megatron.context_parallel_size=$RM_CP \
        reward_model.megatron.tensor_model_parallel_size=$RM_TP \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_penalty=kl \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl-test' \
        trainer.experiment_name="${exp_name}" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.resume_mode="${RESUME_MODE}" \
        trainer.total_epochs=2 \
        trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done
