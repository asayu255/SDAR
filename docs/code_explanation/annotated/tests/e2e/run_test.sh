#!/bin/bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -xeuo pipefail

# Get the configuration name and engine name from arguments
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
CONFIG_NAME="$1"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENGINE="${2:-vllm}"

# Download model if needed
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir "$HOME/models/Qwen/Qwen2.5-0.5B"

# Run the training with the specified configuration
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" "$@" 