# Multitask Batching

## TaskBalancedSampler

dataset 初期化時に `task_to_indices` を作り、task ごとの row index を保持する。
1 batch では各 task から `per_task_batch_size` 件を取り、必要数に満たない task
は shuffle 済み index list を循環する。乱数は `seed + epoch` で再現される。

default layout は task ごとの contiguous block である。
`TASK_BALANCE_INTERLEAVE=1` は同じ sample 集合を round-robin 順へ並べ替える。
sample の選択内容を変える機構ではない。

DP dispatch は global batch を rank-local chunk へ分けるため、interleave は
各 chunk に複数 task が入りやすくし、task 別 teacher forward の偏りを減らす。
ただし backend の乱数消費順・kernel scheduling・floating-point ordering まで
bit-identical かは実装コメントだけでは断定せず、equivalence test の対象とする。

## Batch size の区別

```text
prompt batch:                    45
per-task prompt batch:           15
trajectory group size:            8
actor global mini-batch:         60
actor micro-batch / GPU:          2
teacher micro-batch / GPU:        4
rollout log-prob micro-batch:     16
trainer GPUs:                      3
```

global mini-batch は worker constructor 内で world size と sequence-parallel size
を考慮した rank-local 値へ正規化される。micro-batch はさらに gradient
accumulation の単位であり、prompt batch や trajectory count と同義ではない。

dynamic token budget は token 数を基準に micro-batch grouping を変える別経路で、
現在の Pure OPD run は dynamic batch size を無効にしている。
