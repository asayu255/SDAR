#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# in e2e_sppo.yml, we set NUM_GPUS=8 L20

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=${NUM_GPUS:-8}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
gsm8k_train_path=./data/math/train.parquet
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
gsm8k_test_path=./data/math/test.parquet
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_files="['$gsm8k_train_path']"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
test_files="['$gsm8k_test_path']"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="Qwen2.5-0.5B-Instruct-sppo-minimal"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m recipe.sppo.main_sppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="./models/Qwen2.5-0.5B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang  \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.total_epochs=2 $@
