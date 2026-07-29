#!/bin/bash

# Displays information on how to use script
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
helpFunction()
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
{
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo "Usage: $0 [-d small|all]"
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  exit 1 # Exit script after printing help
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
}

# Get values of command line flags
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
while getopts d: flag
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
do
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  case "${flag}" in
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    d) data=${OPTARG};;
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  esac
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
done

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ -z "$data" ]; then
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo "[ERROR]: Missing -d flag"
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  helpFunction
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# Install Python Dependencies
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
conda install spacy
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
pip install -r requirements_arm.txt;

# Install Environment Dependencies via `conda`
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
conda install -c pytorch faiss-cpu;
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p data;
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd data;
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ "$data" == "small" ]; then
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
elif [ "$data" == "all" ]; then
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
else
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  echo "[ERROR]: argument for `-d` flag not recognized"
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  helpFunction
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd ..

# Download spaCy large NLP model
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python -m spacy download en_core_web_sm

# Build search engine index
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd search_engine
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p resources resources_100 resources_1k resources_100k
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
python convert_product_file_format.py # convert items.json => required doc format
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p indexes
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
./run_indexing.sh
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd ..

# Create logging folder + samples of log data
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
get_human_trajs () {
  # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
  PYCMD=$(cat <<EOF
import gdown
url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
gdown.download_folder(url, quiet=True, remaining_ok=True)
EOF
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  )
  # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
  python -c "$PYCMD"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
}
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
mkdir -p user_session_logs/
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd user_session_logs/
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Downloading 50 example human trajectories..."
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
get_human_trajs
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Downloading example trajectories complete"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
cd ..