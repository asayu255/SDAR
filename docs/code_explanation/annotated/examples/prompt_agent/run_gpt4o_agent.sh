#!/usr/bin/env bash

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ENV_NAME="alfoworld"

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [[ "$ENV_NAME" == "alfoworld" ]]; then
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo "Launching AlfWorld agent..."
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  python3 -m examples.prompt_agent.gpt4o_alfworld
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
else
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  exit 1
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi
