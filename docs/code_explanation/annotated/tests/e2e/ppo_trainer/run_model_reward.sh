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
RM_PAD=${RM_PAD:-True}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SP_SIZE=${SP_SIZE:-1}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SEQ_BALANCE=${SEQ_BALANCE:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LIGER=${LIGER:-False}
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
train_max_token_num_per_gpu=32768
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
infer_max_token_num_per_gpu=32768

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="$(basename "${MODEL_ID,,}")-model-reward-minimal"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_liger="${LIGER}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz="${SEQ_BALANCE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_max_token_num_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_max_token_num_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_max_token_num_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.optim.lr=1e-5 \
    critic.ulysses_sequence_parallel_size="${SP_SIZE}" \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.use_dynamic_bsz="${SEQ_BALANCE}" \
    critic.ppo_max_token_len_per_gpu=${train_max_token_num_per_gpu} \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.ulysses_sequence_parallel_size="${SP_SIZE}" \
    reward_model.model.path="${MODEL_PATH}" \
    reward_model.model.use_remove_padding="${RM_PAD}" \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.use_dynamic_bsz="${SEQ_BALANCE}" \
    reward_model.forward_max_token_len_per_gpu=${infer_max_token_num_per_gpu} \
    reward_model.micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${VAL_BEFORE_TRAIN}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@
