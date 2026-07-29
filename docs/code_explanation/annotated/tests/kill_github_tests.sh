#!/bin/bash

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "$#" -ne 1 ]; then
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Usage: $0 YOUR_GITHUB_TOKEN"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Please provide exactly one input argument for your github token."
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    exit 1
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# Set your GitHub repository details
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
OWNER="volcengine"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
REPO="verl"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
TOKEN=$1

# API URL for workflow runs
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
API_URL="https://api.github.com/repos/$OWNER/$REPO/actions/runs?status=queued"

# Check required commands
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting."; exit 1; }

# Get queued workflow runs
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
response=$(curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github.v3+json" "$API_URL")

# Run this for debugging
# echo $response

# Extract run IDs
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
queued_run_ids=$(echo "$response" | jq -r '.workflow_runs[] | .id')

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ -z "$queued_run_ids" ]; then
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "No queued workflow runs found."
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    exit 0
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# Cancel each queued run
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
for run_id in $queued_run_ids; do
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Cancelling run $run_id"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    cancel_url="https://api.github.com/repos/$OWNER/$REPO/actions/runs/$run_id/cancel"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github.v3+json" "$cancel_url"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Cancelled all queued workflow runs."
