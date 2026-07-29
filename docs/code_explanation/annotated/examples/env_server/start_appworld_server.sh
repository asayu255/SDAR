#!/bin/bash

# Configuration parameters
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
train_batch_size=16
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
val_batch_size=168
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
group_size=8

# Start services
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
current_port=7000

# Activate conda environment
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
source ~/miniconda3/etc/profile.d/conda.sh
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
conda activate appworld

# Stop existing appworld services
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
ps aux | grep "appworld serve" | grep -v grep | awk '{print $2}' | xargs -r kill -9
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
sleep 5

# Calculate total number of instances needed
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
total_instances=$((train_batch_size * group_size + val_batch_size))

# Port file (.ports extension for easy gitignore)
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
port_file="appworld_ports.ports"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
> "$port_file"  # Clear port file

# Function to check if port is in use
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
check_port() {
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    local port=$1
    # Use netstat or ss to check if port is in use
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if command -v ss >/dev/null 2>&1; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        ss -tuln | grep -q ":$port "
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    elif command -v netstat >/dev/null 2>&1; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        netstat -tuln | grep -q ":$port "
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    else
        # If neither is available, try connecting to port
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
}

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Starting $total_instances AppWorld service instances..."
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Port usage will be saved to: $port_file"

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
started_count=0

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
while [ $started_count -lt $total_instances ]; do
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if check_port $current_port; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Port $current_port is already in use, skipping..."
        # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
        current_port=$((current_port + 1))
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        continue
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
    
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Starting service on port $current_port..."
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if appworld serve environment --port $current_port & then
        # Write successfully started port to file
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "$current_port" >> "$port_file"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Service successfully started on port: $current_port"
        # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
        started_count=$((started_count + 1))
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    else
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Failed to start on port $current_port, trying next port..."
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
    
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    current_port=$((current_port + 1))
    
    # Prevent infinite loop by setting maximum port range
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    if [ $current_port -gt 9000 ]; then
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        echo "Error: Reached maximum port range (9000), cannot start more services"
        # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
        break
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    fi
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done