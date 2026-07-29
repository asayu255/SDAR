#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=${NUM_GPUS:-8}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
adv_estimator=grpo

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
kl_coef=0.0
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
use_kl_in_reward=False
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
use_kl_loss=False
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
kl_loss_coef=0.0

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
clip_ratio_low=0.2
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
clip_ratio_high=0.28

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
max_prompt_length=1024
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
max_response_length=2048
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
enable_overlong_buffer=True
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
overlong_buffer_len=128
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
overlong_penalty_factor=1.0

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
loss_agg_mode="token-mean"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
enable_filter_groups=True
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
filter_groups_metric=seq_reward
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
max_num_gen_batches=10

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
gen_prompt_bsz=$((train_prompt_bsz * 4))

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="$(basename "${MODEL_ID,,}")-dapo-minimal"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m recipe.dapo.main_dapo \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    reward_model.reward_manager=dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=['console'] \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.total_training_steps=1 $@
