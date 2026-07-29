#!/usr/bin/env bash

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -e -x

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
OUTPUT_FILE="/tmp/output_ray_trainer.txt"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export PATH=$PATH:~/.local/bin

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
rm -rf $OUTPUT_FILE
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 tests/e2e/arithmetic_sequence/rl/main_trainer.py \
    algorithm.adv_estimator=gae \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    data.train_batch_size=800 \
    data.max_prompt_length=16 \
    data.max_response_length=32 \
    data.return_raw_input_ids=True \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    actor_rollout_ref.model.external_lib=tests.e2e.envs.digit_completion \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=200 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.ppo_micro_batch_size_per_gpu=128 \
    critic.model.path=tests/e2e/arithmetic_sequence/model \
    critic.optim.lr=1e-3 \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=200 \
    trainer.experiment_name=arithmetic_sequences \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=1 \
    trainer.test_freq=1 \
    trainer.save_freq=110 | tee $OUTPUT_FILE;

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
python3 tests/e2e/check_results.py --output_file=$OUTPUT_FILE
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
rm -rf $OUTPUT_FILE
