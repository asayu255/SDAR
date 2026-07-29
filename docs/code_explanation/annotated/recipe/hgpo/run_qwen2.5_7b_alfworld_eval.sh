# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -x
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENGINE=${1:-vllm}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
ulimit -u 65536
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export VLLM_ATTENTION_BACKEND=XFORMERS
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export HF_HOME=${HF_HOME} # hugging face home directory
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export WANDB_API_KEY=${WANDB_API_KEY} # wandb api key
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export WANDB_DIR=${WANDB_DIR} # wandb directory
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} # cuda visible devices

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
project_name="qwen2.5_7b_alfworld_eval"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
val_out=True # True for evaluation on in-domain data, False for evaluation on out-of-domain data

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
eval_experiment_names=(
    # example: "k2_hgpo_length_alpha1.0_baseGroup_False"
    # example: "k4_hgpo_length_alpha1.0_baseGroup_False"
)
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CHECKPOINTS_DIR=${CHECKPOINTS_DIR} # checkpoints directory
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
seeds=(123 456 789)  # three random seeds for evaluation

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_data_size=16
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
val_data_size=128
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
group_size=8
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

# We only use data preparation to indicate the modality and the data size.
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 4)) # evaluate 4 × val_data_size tasks during each iteration

# loop: first by experiment name, then by seed
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
for eval_experiment_name in "${eval_experiment_names[@]}"; do
    # parse number after k in experiment name as history_length (e.g. k6->6, k4->4), default 2 if not found
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    history_length=$(echo "$eval_experiment_name" | sed -n 's/.*k\([0-9]\+\).*/\1/p')
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    history_length=${history_length:-4}
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    echo "=========================================="
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Experiment: $eval_experiment_name, history_length: $history_length"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    echo "=========================================="

    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    eval_dir="${CHECKPOINTS_DIR}/qwen2.5_7b_alfworld_train/${eval_experiment_name}"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if [ ! -d "$eval_dir" ]; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Error: checkpoint directory does not exist: $eval_dir, skip."
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        continue
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi

    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    log_dir="logs/${eval_dir}"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    mkdir -p "$log_dir"

    # loop: first by seed, then by experiment name
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    for seed in "${seeds[@]}"; do
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "------------------------------------------"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Running experiment: $eval_experiment_name, seed: $seed"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "------------------------------------------"

        # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
        temp_log="${log_dir}/output_seed${seed}.log"

        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=hgpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.hgpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.seed=$seed \
    env.history_length=$history_length \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name="${eval_experiment_name}_seed${seed}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=160 \
    trainer.default_local_dir=${eval_dir} \
    trainer.val_only=True \
    trainer.val_out=${val_out} \
    trainer.val_before_train=True $@ 2>&1 | tee "$temp_log"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Completed run with seed: $seed"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo ""
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    done
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Completed all seeds for experiment: $eval_experiment_name"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo ""
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done
