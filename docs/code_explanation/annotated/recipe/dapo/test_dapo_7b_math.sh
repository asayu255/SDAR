#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
project_name='DAPO'
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name='DAPO-Qwen2.5-7b-MATH-0527a1'

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
adv_estimator=grpo

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
use_kl_in_reward=False
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
kl_coef=0.0
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
use_kl_loss=False
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
kl_loss_coef=0.0

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
clip_ratio_low=0.2
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
clip_ratio_high=0.28

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
max_prompt_length=$((1024 * 2))
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
max_response_length=$((1024 * 8))
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
enable_overlong_buffer=True
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
overlong_buffer_len=$((1024 * 4))
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
overlong_penalty_factor=1.0

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
loss_agg_mode="token-mean"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_prompt_bsz=512
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
n_resp_per_prompt=16
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_prompt_mini_bsz=32

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NNODES=${NNODES:-8}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Paths
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
temperature=1.0
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
top_p=1.0
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
val_top_p=0.7

# Performance Related Parameter
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
sp_size=4
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
use_dynamic_bsz=True
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
offload=True
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
gen_tp=4
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
fsdp_size=32

# remember to set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for this model

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
