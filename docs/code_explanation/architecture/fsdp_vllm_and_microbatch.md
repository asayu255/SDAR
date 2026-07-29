# FSDP・rollout engine・micro-batch

## worker と重み

Ray resource pool は actor/rollout、task-specific teacher などの worker group を配置する。
teacher は `ref` role の forward-only workerとして構築され、CPU offload を有効にする。
同一 pool 上に置くことと、全 teacher parameter が常時 GPU resident であることは同義でない。

FSDP は parameter shard、all-gather、offload の境界を作る。teacher forward は task slice
ごとに呼ばれ、返却される top-k signal は student update 前に global batch へ再配置される。
teacher tensor は学習対象ではなく、student loss 側では gradient を切って扱う。

## rollout backend

- HF rollout は model forward/generate を micro-batch に分割する単純な基準実装。
- vLLM rollout は推論 engine の KV cache と batching を使い、学習 workerとの重み同期境界を持つ。
- SGLang rollout は async request と tool call state machine を持つ。

backend が生成順や throughput を変えても、下流で old/teacher/ref log-prob を再計算する経路では、
学習データとして保持された token、mask、row ordering が意味論上の契約になる。

## batch単位を混同しない

`train_batch_size` は prompt、`rollout.n` は prompt当たりtrajectory、multi-turn収集後の
DataProto row は turn、`micro_batch_size` は forward/backward の分割単位である。
dynamic batching を使う場合は row 数ではなく token budget で grouping が変わる。
Pure OPD の実験設定では dynamic batch を無効にし、固定 micro-batch の比較可能性を優先する。
