# Multitask Environment Routing

## Global-to-local routing

`MultiTaskEnvironmentManager` は mixed global batch の `task_name` を正規化し、
task ごとの global index list を作る。action/kwargs を task-local batch へ
slice し、各 manager の reset/step を `ThreadPoolExecutor` で並列実行する。

戻り値は task-local 順から global index へ scatter される。
observation、reward、done、info、success evaluator のすべてで同じ復元規則を
使う。早期終了済み task は last observation を再利用して short-circuit し、
他 task の step を継続する。

## Task 固有動作

### AlfWorld

- TextWorld/subprocess worker と gamefile schedule を使う
- admissible commands を observation とともに返す
- Pure OPD 設定の最大 step は 50
- resume 時は reset 回数に対応する game iterator 位置も進める

### Search

- parquet の question/ground truth/data source を `env_kwargs` で受け取る
- HTTP retriever を呼び、query history と passage text を更新する
- 最大 step は 4
- `SEARCH_QUERY_CACHE` は同じ query の response text を再利用する性能機構

### WebShop

- Ray actor または task-specific worker が environment state を保持する
- task score を reward/success metric として返す
- 最大 step は 15
- resume 時は RNG/reset schedule を checkpoint step に対応させる

task-specific max_steps は global `env.max_steps` と別に manager へ渡される。
global batch の一部 task が終了しても、残り task の row ordering と metric mask
は維持される。
