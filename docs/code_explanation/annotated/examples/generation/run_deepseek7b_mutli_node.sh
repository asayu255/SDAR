# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -x

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
data_path=$HOME/data/rlhf/gsm8k/test.parquet
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
save_path=$HOME/data/rlhf/math/deepseek_v2_lite_gen_test.parquet
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
model_path=deepseek-ai/deepseek-llm-7b-chat

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path\
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=16 \
    rollout.gpu_memory_utilization=0.8
