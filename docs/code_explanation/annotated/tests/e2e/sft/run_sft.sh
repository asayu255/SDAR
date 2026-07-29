#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=${NUM_GPUS:-8}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SP_SIZE=${SP_SIZE:-1}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LIGER=${LIGER:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
MULTITURN=${MULTITURN:-False}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
LORA_RANK=${LORA_RANK:-0}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RM_PAD=${RM_PAD:-True}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
micro_bsz=2
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
NUM_GPUS=8

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
project_name="verl-test"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
exp_name="$(basename "${MODEL_ID,,}")-sft-minimal"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ckpts_home=${ckpts_home:-$HOME/${project_name}/${exp_name}}

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p "${ckpts_home}"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.multiturn.enable="${MULTITURN}" \
    data.multiturn.messages_key=messages \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=${micro_bsz} \
    model.partial_pretrain="${MODEL_PATH}" \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.use_liger="${LIGER}" \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding="${RM_PAD}" \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_training_steps=1 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
rm -rf "${ckpts_home:?}/*"