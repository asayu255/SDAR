# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
save_path=$HOME/data/searchR1

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
index_file=$save_path/e5_Flat.index
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
corpus_file=$save_path/wiki-18.jsonl
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
retriever_name=e5
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
retriever_path=intfloat/e5-base-v2

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000 \