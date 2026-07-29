#!/usr/bin/env bash
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -uxo pipefail

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export OVERWRITE=${OVERWRITE:-0}

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p "${VERL_HOME}/data"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi
