#!/usr/bin/env bash
# Serial training: grpo search on 3B -> 7B -> 1.7B
# Only the first (3B) run does val_before_train
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
set -x

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
source $(conda info --base)/etc/profile.d/conda.sh

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export http_proxy=http://10.217.142.137:8080
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export https_proxy=http://10.217.142.137:8080
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export no_proxy=localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,33.0.0.0/8
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export grpc_proxy=""

# Start retrieval server once
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
conda activate retriever
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
bash examples/search/retriever/retrieval_launch.sh > retrieval_server_grpo_serial.log 2>&1 &
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
RETRIEVER_PID=$!

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Waiting for retrieval server to start..."
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
for i in $(seq 1 400); do
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if curl -s -o /dev/null -w "%{http_code}" -X POST http://0.0.0.0:8000/retrieve -H "Content-Type: application/json" -d '{"query": "test", "topk": 1}' --max-time 5 | grep -q "200"; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Retrieval server is ready!"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        break
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if [ $i -eq 400 ]; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "ERROR: Retrieval server failed to start after 400 attempts"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        kill $RETRIEVER_PID 2>/dev/null
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        exit 1
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    sleep 5
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done

# Run 3B (with val_before_train)
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
echo "========== Starting GRPO 3B =========="
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
bash "$SCRIPT_DIR/run_search.sh"

# Run 7B
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
echo "========== Starting GRPO 7B =========="
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
bash "$SCRIPT_DIR/run_search_7b.sh"

# Run 1.7B
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
echo "========== Starting GRPO Qwen3-1.7B =========="
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
bash "$SCRIPT_DIR/run_search_qwen3_1b.sh"

# Cleanup
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
kill $RETRIEVER_PID 2>/dev/null
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "All GRPO search training done."
