<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Detached Worker
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## How to run (Only on a single node)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Start a local ray cluster: 
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
ray start --head --port=6379
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Run the server
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python3 server.py
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- On another terminal, Run the client
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python3 client.py
```
