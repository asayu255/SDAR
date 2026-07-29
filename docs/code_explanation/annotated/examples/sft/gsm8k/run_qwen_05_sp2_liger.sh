# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -x

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "$#" -lt 2 ]; then
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    exit 1
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
nproc_per_node=$1
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
save_path=$2

# Shift the arguments so $@ refers to the rest
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
shift 2

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct-sp2-liger \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
