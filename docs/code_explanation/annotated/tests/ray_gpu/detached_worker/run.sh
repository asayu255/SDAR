#!/bin/bash
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
ray start --head --port=6379
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 server.py
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python3 client.py
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
ray stop --force