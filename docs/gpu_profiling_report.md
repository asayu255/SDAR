# GPU プロファイリングと高速化の記録（multitask SFT arm）

自分用の作業記録。3×RTX A6000 の 1 ノードで Qwen3-1.7B の multitask SFT を回すにあたり、
`update_actor` の内訳を計測して 487 → 415 s/step まで縮めた過程と、その途中で分かったこと。
数値はすべて `ppo_micro_batch_size_per_gpu=10` 時点のもの（この後 OOM 対策で 5 に変更した
ので、現構成の実測値ではない）。

計測に使った機構は `verl/utils/gpu_profiler.py` と `verl/workers/actor/dp_actor.py` の
phase タグ。`GPU_PROFILER=1` が無ければ完全に no-op なので、本実験ではすべて切ってある。

---

## 0. 計測環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA RTX A6000 48 GB × 3（driver 550.144.03, CUDA 12.4） |
| ホスト RAM | 515 GB |
| モデル | Qwen/Qwen3-1.7B |
| FSDP | ZeRO-2（`sharding_strategy=shard_grad_op`）、`forward_prefetch=True` |
| バッチ | `ppo_mini_batch_size=60`（GPU あたり 20）、`ppo_micro_batch_size_per_gpu=10` → `gradient_accumulation=2` |
| 系列長 | prompt 最大 4096 / response 最大 512（合計 4608） |
| その他 | gradient checkpointing 有効、`use_fused_kernels=False`、`use_remove_padding=True` |
| 1 step の規模 | 約 4,300〜4,600 行、380〜400 万トークン |

---

## 1. なぜ既存の計測では足りなかったか

もともと出ていたのは `timing_s/update_actor` ひとつだけで、これはドライバ側の blocking
`ray.get` を含む。つまり

```
timing_s/update_actor = バッチの直列化 + object store 経由の転送 + worker の計算 + メトリクス返送
```

が 1 バケットに潰れている。GPU が遊んでいる区間も「計算時間」として数えられるので、
この値だけを見ていると何を削れば良いか分からない。

そこで 2 つ足した。

1. **`timing_s/update_actor_worker`**（`verl/workers/fsdp_workers.py:671`）
   worker 側で `update_policy` を実際に回していた時間。
   `update_actor − update_actor_worker` が転送時間になる。

2. **worker 内の phase 分解**（`dp_actor.py` の `_actor_phase`）
   rank 0 に NVML サンプラを常駐させ、`actor.fwd` / `actor.bwd` / `actor.task_metrics` /
   `actor.optim` のタグで `update_policy` の内側を割る。NVML の読みはデバイス単位なので、
   worker プロセスに置いたサンプラでも同じ GPU を見る。タグが帰属を可能にしている。

   注意点として、カーネル起動は非同期なので phase の wall clock は「その仕事が
   *発行された* 時刻」であって GPU が終えた時刻ではない。境界は launch queue 1 本分ぼやける。
   `GPU_PROFILER_SYNC_PHASES=1` で境界ごとに同期すれば正確になるが、本来重なる処理を
   直列化するので「帰属」としては読めても「速度」としては読めない。

3. **毎サンプルの CSV 出力**（`GPU_PROFILER_TRACE`）
   集計表では平均に埋もれる一瞬の落ち込みを追うため。0.1 s 間隔で全サンプルを行として吐く。

---

## 2. 分かったこと

### 2.1 `sm_util` 98.5% は「よく計算している」ではない

NVML の `sm_util` は **「1 つ以上のカーネルが乗っていた時間の割合」** であって FLOP 利用率
ではない。カーネルが何をしていようが、たとえ NCCL の通信カーネルが回っているだけでも
100% に近づく。

計測では全 phase で 97〜99.5% だった。ここで「もう限界」と判断しかけたが、これは誤読で、
**SM 98.5% と低い MFU は矛盾なく両立する**。両立するのは通信律速・帯域律速のとき。

同時に取っていた `memBW%` が 53.7% だったことが手がかりになった。演算律速なら帯域はもっと
低いか、あるいは帯域律速なら 80% 以上に張り付く。53% は「どちらでもない = 待っている」を示す。

### 2.2 GPU util の急落は step 境界であって学習処理ではない

wandb のチャートで 45% まで落ちる点があり、その正体を追った。

trace CSV の低いサンプルには、いずれも `actor.*` のどの phase タグも付いていなかった。
つまり **`update_actor` の外側**にいる。落ち込みは step 境界にしか現れない。

隣接する 2 つの区間からなる:

```
[step N-1 の計算  GPU 98%]
     ↓ ray.get が返る
[882-891 行 メトリクス計算・wandb 送信 / next() バッチ準備]  ← (idle/other)
     ↓ 858 行に入る
[batch を object store へ直列化、worker が pull]   ← update_actor だが GPU は空
     ↓
[step N の forward  GPU 98%]
```

| 区間 | 時間 | 出どころ |
|---|---:|---|
| `(idle/other)` | 1 秒未満（0.5〜0.6 s） | ドライバ側 phase 表 |
| Ray の dispatch | 0.94 s | `update_actor − update_actor_worker` |

合計で wall clock の **約 0.4%**。チャート上の目立ち方と実コストが 2 桁乖離していた。

**GPU は止まっていない。** `(idle/other)` の `sm%` は 47.5（step 1）/ 61.8（step 2）で、
前 step の末尾のカーネルが非同期に流れている。チャートの底が 0% でなく 45% なのは
これが理由で、`(idle/other)` という名前は「GPU が idle」ではなく
「**phase スタックが空**」の意味である。

**測定の限界を 2 つ明記しておく。**

1. **`(idle/other)` の内訳は分離していない。** このバケットに入るのは 882-891 行
   （`compute_opd_data_metrics` ×2 / timing / throughput / `logger.log`）と
   `next()`（バッチ準備）の 2 つで、どちらが支配的かは測っていない。
   `_timer("batch_wait")` を `next()` に巻けば確定する。
   なお 859-880 行は `step` とラベルされるはずのところ、`step` phase は 3 step とも
   **0 サンプル**だったので、0.3 s 未満として候補から外れる。
2. **`(idle/other)` の値そのものの誤差が ±0.3 s ある。** 同じ表の `cpu%` が
   `update_actor` の 4 に対して **100**、つまりドライバの Python が 1 コアを
   占有していてサンプラースレッドがスケジュールされない。interval 0.1 s に対し
   実際の `dt` は約 0.3 s で、飛んだ区間は「採取できた時点の phase」に丸ごと
   計上される。**測ろうとしている量と同じ桁の帰属誤差**である。

それでもこの節の結論は変わらない。誤差を最も悲観的に取って 0.9 s としても
転送と合わせて 1.84 s = 0.44%。**精度が足りない測定でも、桁が 2 つ違えば
判断には十分**というのがここでの理屈である。

この過程で 2 回誤診している（後述）。教訓は **「グラフで目立つ = 効く」ではない**、
および **落ち込みの幅ではなく面積を見る**。

### 2.3 step 予算の内訳

no_sync とメトリクス遅延集約を入れた後の 1 step（415 s）:

| phase | 時間 | 比率 | sm% | memBW% |
|---|---:|---:|---:|---:|
| `actor.bwd` | 296.6 s | 71.5% | 98.8 | 56.1 |
| `actor.fwd` | 93.6 s | 22.6% | 97.7 | 48.0 |
| `actor.task_metrics` | 19.5 s | 4.7% | 99.0 | 48.6 |
| `actor.optim` | 0.5 s | 0.1% | 99.5 | 45.5 |
| step 境界 | 4.6 s | 1.1% | 99.3 | 38.2 |
| **合計** | **414.7 s** | 100% | 98.6 | 53.6 |

optimizer の更新そのものは 0.1% しかない。**削るべき場所は backward の中の通信**、
というのがこの表の答え。

### 2.4 律速はノード内トポロジ

`nvidia-smi topo -m` を取ったところ:

```
GPU0  0000:3B:00.0  NUMA node 0
GPU1  0000:AF:00.0  NUMA node 1
GPU2  0000:D8:00.0  NUMA node 1

GPU0 <-> GPU1 : SYS     (UPI 越え)
GPU0 <-> GPU2 : SYS
GPU1 <-> GPU2 : NODE
```

GPU0 だけが別 NUMA ノードにあり、他 2 枚との経路が `SYS`（CPU の UPI を跨ぐ）。
NVLink は無い。NCCL はこの経路でホストメモリ経由のステージングにフォールバックし、
実効帯域は **約 4.4 GB/s**（PCIe 4.0 x16 の 20〜25 GB/s に対して）。

3 GPU の all-reduce は最も遅いリンクに律速されるので、**この構成の下限はここで決まる**。
コードでは解決できない。改善できるのは「通信の回数を減らす」ことだけで、
「1 回を速くする」ことはできない。

### 2.5 メトリクスそのものの欠陥を 2 件

**`perf/mfu/actor` が全 step 0.000 だった。**
原因は推定側ではなく分母。`verl/utils/flops_counter.py:73` の `get_device_flops` が
未知の GPU に対して `float("inf")` を返す設計になっていて、MFU は
`estimated / promised` なので `x / inf = 0.0` になる。
qwen3 は FLOPs 推定の対応表に入っていたが、**RTX A6000 が GPU 対応表に無かった**。
A40 と同じ GA102 なので 154.8 TFLOPS（BF16 dense tensor）を追加した（`3d21864`）。

「未対応だから欠測」ではなく「未対応だから 0」になる設計なので、
**壊れた指標に見えて実は未対応、という紛らわしい失敗の仕方**をしていた。

**`perf/max_memory_allocated_gb` / `_reserved_gb` が物理容量を超える値を報告する。**
48 GiB のカードに対して 57.289 / 66.898 という値が出ていた。
`fsdp_workers.py:675` は素の `torch.cuda.max_memory_allocated()` を rank ごとに取り、
`reduce_metrics` はキーに `max` を含むので rank 間 np.max。つまり単一 GPU の値のはずが
物理容量を超えている。**空き容量の判断材料として使えない。**
`nvidia-smi` を正とすべき（実測 46.8 / 48.0 / 48.7 GiB / 49140 MiB）。

これに気づかず「メモリが減ったので no_sync は効いていない」と誤った推論をした（後述）。

---

## 3. 実装した高速化

| # | 機構 | 効果 | 精度への影響 | commit |
|---|---|---|---|---|
| 1 | FSDP ZeRO-2（`shard_grad_op`） | 層あたり 3 回の all-gather を 1 回に | 中立 | `cb60d7b` |
| 2 | `forward_prefetch` | all-gather 発行の前倒し | 中立 | `960a16a` `ebd67d2` |
| 3 | `no_sync` 勾配蓄積 | **−45 s/step** | 丸め順序のみ | `154e8dd` |
| 4 | メトリクス遅延集約 | **−13.5 s/step** | 中立（かつバグ修正） | `65797f7` |
| 5 | pool の shard 分割ロード | ピーク 360 GiB → 常駐+最大 shard | 完全同一 | `360317a` |
| 6 | pool のオフラインキャッシュ | 339.5 → 136.5 GiB、起動から 79 分削除 | ビット同一 | `47fb1e6` |
| 7 | バッチ prefetch | dispatch と計算の重畳 | ビット同一 | 既存機構 |

1〜3 は FSDP の通信、4 は CPU-GPU 同期、5〜7 はデータ供給。**互いに独立ではなく、
特に 1 と 3 は片方だけでは意味が変わる**（3.3 節）。

以下、機構ごとに「何を変えたか / なぜ効くか / 実装 / 効果 / 精度 / 注意点」で記す。

### 3.1 FSDP ZeRO-2（`sharding_strategy=shard_grad_op`）

**何を変えたか。** FSDP の既定は 1 次元メッシュで `FULL_SHARD`（ZeRO-3）。これを
`SHARD_GRAD_OP`（ZeRO-2）に変えた。`verl/workers/fsdp_workers.py:88` の
`get_sharding_strategy` に config からの上書きを足してある（既定は変えていない）。

**なぜ効くか。** ZeRO-3 はパラメータを常時シャードで持ち、使う直前に all-gather して
使い終わったら即座に捨てる。gradient checkpointing が有効だと、1 つの層は
1 micro-batch あたり **3 回**触られる:

1. forward
2. backward 前の再計算（checkpoint の復元）
3. backward 本体

ZeRO-3 では 3 回とも all-gather が要る。ZeRO-2 は forward の後に reshard しないので
**1 回で済む**。残る集団通信は勾配の reduce-scatter だけになる。

**効果。** 単独では測っていない（`cb60d7b` は計測基盤を入れる前）。ただし 3.3 節の
`no_sync` が forward まで速くした理由がこの設定に依存しているので、実質的には
セットで −45 s に寄与している。

**精度。** 中立。all-gather はバイトを動かすだけで、forward / backward のカーネルが
見る入力は同一。勾配も同じ `reduce_dtype` の同じ reduce-scatter で縮約される。

**代償。** メモリ。パラメータが micro-batch の backward の間ずっと unsharded で
居座るので、ピークが「unsharded パラメータ − そのシャード」ぶん増える。
1.7B・bf16 なら 3.4 GB × (1 − 1/3) ≈ **+2.3 GB**。

**注意点。** 勾配経路に乗る設定なので、性能ノブでありながら
`expected_multitask_sft_config.yaml` に記載した。run の同一性が
このファイルから見えない場所に依存すべきではないため。

### 3.2 `forward_prefetch`

**何を変えたか。** FSDP のコンストラクタ引数 1 つ。`fsdp_workers.py:376`（actor）と
`:1130`（critic）で config から読むようにした。既定は upstream どおり `False`。

```python
forward_prefetch=bool(fsdp_config.get("forward_prefetch", False)),
```

**なぜ効くか。** 既定では、FSDP ユニット N の計算が終わってからユニット N+1 の
all-gather を発行する。つまり通信と計算が直列になる。`forward_prefetch=True` は
**N の計算中に N+1 の all-gather を発行する**ので、通信が計算の裏に隠れる。

NVLink があれば通信が短いので効果は小さいが、**この構成は PCIe（しかも一部は
UPI 越え、2.4 節）なので隠す価値のある長さがある**。

**効果。** 単独では測っていない。487 s の測定時点で既に有効だったため、
それ以降の比較には現れない。

**精度。** 中立。all-gather の発行タイミングが変わるだけで、演算も通信内容も同一。

**注意点。** レポート 5 節 ③ に書いたとおり、これを「未検証の改善案」として提示した
のは誤りだった。**測定済み構成の内容を把握していなかった**ことによる。

### 3.3 `no_sync` 勾配蓄積（最大の効果）

**何を変えたか。** `gradient_accumulation` 個の micro-batch のうち、**最後以外を
FSDP の `no_sync()` の中で回す**ようにした。`actor_rollout_ref.actor.no_sync_grad_accum=True`
で有効、既定は `False`。

**なぜ効くか（表層）。** 既定の FSDP は micro-batch ごとに勾配を reduce-scatter する。
`gradient_accumulation=2` なら mini-batch あたり 2 回。`no_sync` を掛ければ **1 回**で済む。

**なぜ効くか（実際）。** 上の説明だけなら backward しか速くならないはずだが、実測では
**forward も 115 → 93.6 s に縮んだ**。理由を torch 2.13 のソースで確認した。

```python
# torch/distributed/fsdp/_runtime_utils.py
def _should_free_in_backward(state, handle) -> bool:
    if not handle.uses_sharded_strategy:
        return False
    return (state._sync_gradients
            or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES)

# RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = {FULL_SHARD, HYBRID_SHARD}
# SHARD_GRAD_OP は含まれない
```

`SHARD_GRAD_OP` はこの集合に入っていないので、`_sync_gradients=False` の間は
**パラメータが unsharded のまま残る**。したがって次の micro-batch は all-gather を
やり直さない。**reduce-scatter だけでなく all-gather も半減する。**

**これは 3.1 節の ZeRO-2 に依存している。** ZeRO-3 のままだと
`_should_free_in_backward` の第 2 項が真になり、`no_sync` 中でも reshard されるので
all-gather は減らない。**1 と 3 は組み合わせて初めてこの効果になる。**

さらに `_post_backward_hook` を読むと:

```python
_post_backward_reshard(state, handle)
if not state._sync_gradients:
    if handle._use_orig_params:
        handle._use_unsharded_grad_views()
    return          # reduce-scatter せずに戻る
```

`flat_param.grad` が **unsharded のまま累積される**。1.7B・bf16 なら
sharded 1.13 GB → unsharded 3.4 GB、差し引き **約 +2.3 GB**。

**実装上の制約。** FSDP1 の `no_sync()` は `TrainingState.IDLE` を要求する。forward が
モジュールを `FORWARD_BACKWARD` に移してしまうので、**backward だけを包むことは
できない**。micro-batch の**境界**で enter / exit する必要がある。

```python
if self.no_sync_grad_accum:
    if micro_idx == 0 and n_micro > 1:
        accum_ctx = _grad_sync_context(self.actor_module, True)
        if accum_ctx is not None:
            accum_ctx.__enter__()
    elif micro_idx == n_micro - 1 and accum_ctx is not None:
        accum_ctx.__exit__(None, None, None)
        accum_ctx = None
```

`_grad_sync_context` は FSDP1 なら `module.no_sync()`、FSDP2 なら
`_fsdp2_no_sync(module)`、どちらでもなければ `None` を返す。`None` のときに
enter を丸ごと飛ばせるよう、yield ではなく return にしてある。
`update_policy` の入口で `_sync_gradients = True` に戻す防御も入れた（前 step が
例外で抜けた場合に備えて）。

**効果。** fwd −21.4 s、bwd −23.4 s、合わせて **−45 s/step**。
ここから all-gather 総量 ≈ 43 s（fwd の 37%）、reduce-scatter 総量 ≈ 47 s
（bwd の 15%）と逆算できる。

**精度。** **中立ではない。** unsharded の micro-batch 勾配を先に足してから 1 回
reduce-scatter するので、シャードごとに reduce-scatter して足す場合と浮動小数点の
結合順序が変わる。差は丸め誤差レベルで期待値は同一だが、勾配の作り方が変わる以上
`expected_multitask_sft_config.yaml` に記載した。

**注意点。** +2.3 GB が後の OOM の一因になり、micro batch を 5 に下げる判断に
つながった（8 節）。**OOM が出たら最初に戻すべきノブ**である。

### 3.4 メトリクス遅延集約

**何を変えたか。** `actor/sft_loss` などの per-task メトリクスを micro-batch ループの
中で `.item()` していたのを、`torch.Tensor` のまま溜めて最後に 1 回だけ読むように変えた。

```python
deferred_metrics = {}

def _defer(name, value):
    deferred_metrics.setdefault(name, []).append(value.detach())

# ... ループの外で
{name: torch.stack(values).mean().item() for name, values in deferred_metrics.items()}
```

**なぜ効くか。** `.item()` は GPU → CPU の同期点である。呼ぶたびに CPU が
「そこまでのカーネルが全部終わる」のを待つので、**CPU が先回りしてカーネルを
積んでおけなくなる**。micro-batch ごとに数個ずつ呼んでいたので、step あたり
数百回の同期が入っていた。

**効果。** `actor.task_metrics` が 33 → 19.5 s（**−13.5 s**）。加えて `actor.fwd` の
`idle%` が 0.8、`maxGap` が 0.6 s まで下がった（同期が消えて CPU が先行できるように
なったぶん）。fwd/bwd の短縮のうちどれだけがこれによるものかは分離していない。

**副産物 — 実害のあるバグが直った。** `actor/sft_loss_weighted` はループ内で
**代入**していたので最後の micro-batch の値しか残らない。`_balance_batch` の
並べ替え後、最後の micro-batch は adjust_batch の padding 行（重み 0）だけに
なることが多く、**0.000 が記録され続けていた**。padding が 3 行しかなかった step 5 で
だけ 0.492 が出て、それで気づいた。

「メトリクスがゼロ」を書式の問題だと思い込みかけたが、実際には集計の欠陥だった。

**精度。** 中立。学習には一切影響しない（ログの値が正しくなっただけ）。

### 3.5 pool の shard 分割ロード

**何を変えたか。** タスクごとに全シャードを 1 つの DataProto に `concat` していたのを
やめ、**シャードのリストのまま保持して索引で引く**ようにした。

```python
self._task_shards[task]              # [DataProto, ...] 30個
self._task_to_traj_rows[task][uid]   # (shard_idx, np.array([行番号...]))
```

**なぜ必要だったか。** `concat` の最中は「元の 30 個」と「結合後の 1 個」が同時に
存在する。36,000 軌跡では **ピーク約 360 GiB**、ホスト RAM 515 GB に対して危険域だった。

**実装。** `_gather_trajs`（`opd_offpolicy_ray_trainer.py:557`）が、必要なシャードから
だけ行を抜いて抽選順に並べ直す。

```python
order = np.argsort(shard_of, kind="stable")      # シャードごとにまとめる
parts = [shards[int(s)].select_idxs(row_of[order[ordered_shard == s]].tolist())
         for s in np.unique(ordered_shard)]
merged = DataProto.concat(parts)
return merged.select_idxs(np.argsort(order, kind="stable").tolist())   # 抽選順に戻す
```

**最後の 1 行が本質。** シャード別に集めると行がシャード順に並び、`adjust_batch` が
複製する行も `_balance_batch` の分割位置も変わってしまう。順序を戻すことで
**「同等」ではなく「同一」**になり、プールがディスク上でどう分割されていようと
下流から区別できない。代償は step バッチ（数千行）1 個ぶんの追加コピー。

**効果。** ピーク 360 GiB → **常駐 136.5 + 最大シャード 3.8 = 140.4 GiB**。

**精度。** 完全同一。

### 3.6 pool のオフラインキャッシュ

**何を変えたか。** 起動のたびにやっていたフィルタを 1 回だけ実行してディスクに置く
`scripts/cache_teacher_pool.py` を追加した。

**なぜ必要だったか。** トレーナは読み込み直後に 2 種類のものを捨てていた。

- **padding 行** — 生成時の `adjust_batch` が既存行を複製して足したもの。残すと
  同じ turn を 2 回学習する
- **使わない列** — `prompts`（4096 列、on-policy のダンプ専用）、`response_mask`
  （`attention_mask` から再計算できる）、`teacher_topk_logprobs` / `teacher_topk_ids`
  （KD の損失専用、SFT の NLL は読まない）

339.5 GiB 読んで 136.5 GiB しか使わない。**捨てる 203 GiB を毎起動・毎再起動で
読んで unpickle していた**（79 分）。

**実装。** トレーナがメモリ上に作るのと同じ DataProto を、そのまま
`save_to_disk` する。トレーナ側は変更なし — キャッシュを読むと
「padding 行 0 件、落とす列なし」と判定してそのまま通す。

**なぜ結果が変わらないか。** 入力 1 ファイルにつき出力 1 ファイル、**同じ basename・
同じ行順**。したがって `sorted(glob)` が同じ順序を返し、軌跡の抽選母集団
（＝ファイルを順に見て初めて出会った順）が変わらない。
`tests/trainer/test_cache_teacher_pool.py` が両 arm・5 step 分をテンソル比較して確認。

**効果。** 起動から **79 分が消えた**。ディスク 339.5 → 136.5 GiB（0.40x）。

**注意点。** キャッシュは **arm 固有**。SFT 版には top-k が無いので KD run が読むと
壊れる。`_cache_manifest.json` に arm を記録してある。
また、トレーナは `data_dir` の `*.pt` を無条件に glob するので、**別 run のシャードが
紛れ込むと静かに混ざる**（`traj_uid` の重複検査は別 run の軌跡を素通りさせる）。
`--only <task>` は再構築時に source に無い出力を削除する。

### 3.7 バッチ prefetch

**何を変えたか。** step k+1 の抽選と `_prepare_batch` を、step k の `update_actor` の
裏でバックグラウンドスレッドに実行させる。`OFFPOLICY_BATCH_PREFETCH=1`。

**なぜ効くか。** `update_actor` は blocking な `ray.get` であり、**GIL を持たない**。
その間 Python スレッドは自由に動ける。

**実装**（`_prepared_batch_iter`）。深さ 1 の `queue.Queue` と `Semaphore(1)`。

```python
room.release()      # バッチを使う前に解放 → 次のバッチが update_actor 中に作られる
```

`room` を「バッチを yield する前」に解放しているのが要点で、後にすると次の準備が
`update_actor` の後ろにずれる。

**ビット同一性の根拠。** 準備スレッドは 1 本だけ・深さ 1 なので、抽選も準備も
**step 順に、単一スレッドで**実行される。したがって乱数列（`default_rng(data.seed)` と
global numpy RNG）の消費順が逐次実行と同じになる。

**効果。** **未測定。** `update_actor` の外側の総時間が 1 秒未満、という上限しか
分かっていない（2.2 節）。`_timer("batch_wait")` を `next()` に巻けば確定する。

**注意点。** 準備済みバッチがもう 1 つメモリに載る。数千行なので数百 MB。
また `d1750e5` の resume 対応は、**prefetch スレッドを作る前に**replay を走らせる
必要がある（replay が両方の RNG 列の唯一の消費者である間に、step 順で終わらせる
ため）。

---

## 4. 採用しなかった手法

| 手法 | 判定 | 理由 |
|---|---|---|
| gradient checkpointing 無効化 | **不可** | 約 +10 GB 必要。実測の空きは 0.5〜2.2 GiB |
| fused kernels | 保留 | 下記 |
| response-only lm_head | 未実装 | −20〜30 s 見込み、3 arm 共通で入れられる。実装コスト中 |

### fused kernels について（当初の見積もりが誤りだった）

`use_fused_kernels=True` は「prompt 位置の logits を計算しなくなるので速い」と
考えたが、実装を読んで誤りと分かった。

`verl/utils/experimental/torch_functional.py` の `FusedLinearForPPO` が分割するのは
**token 次元**（`chunk_size=512`）であって vocab 次元ではない。prompt 位置の logits も
従来どおり全部計算する。変わるのは:

| | 非 fused | fused |
|---|---|---|
| logits の実体化 | `(total_nnz, 151936)` ≈ 2.7 GB、backward 用に保持 | 512 token チャンクのみ、保持せず |
| lm_head matmul | 3 回（fwd 1 + bwd 2） | **4 回**（backward で logits を再計算） |

`torch.autograd.Function` の forward が logits を保存せず backward で再計算するので、
**FLOPs は約 33% 増える**。1 GPU あたり lm_head 1 パスが約 796 TFLOP、実効 100 TFLOPS
として 1 パス約 8 秒なので、**速度は +8 s、メモリは −2.5〜5 GB**。

つまり fused kernels は速度の手段ではなく **メモリの手段**。
OOM が出たときの第一手として温存する。なお KD arm では top-k KL が logits を必要と
するため使えない（`dp_actor.py:200` で `NotImplementedError`）。

「prompt 位置を計算しない」のは別の案（response-only lm_head）で、そちらは
response が全トークンの 24.3%（prompt 平均 662.8 / response 平均 212.6）なので
lm_head の 75.7% が消える。3 arm 共通で入れられるが未実装。

---

## 5. 誤った判断の記録

自分用なので、間違えた過程も残す。どれも「測る前に決めた」ことが原因。

**① GPU util の落ち込みをチェックポイント保存と診断した。**
実際には当該 run は `save_freq=-1` で `_save_checkpoint` が一度も走っていなかった。
自分が書いた smoke コマンドの中身を確認せずに答えた。

**② 「転送は原因ではない」と切り捨てた。**
`update_actor − update_actor_worker` の 0.7〜1.1 s を、step 全体の 415 s に対する
比率が小さいという理由で無視した。しかし落ち込みの正体はまさにそこだった。
**「全体に対して小さい」と「その瞬間の原因ではない」は別の主張**。

**③ `forward_prefetch` を未検証の改善案として提示した。**
指摘されて確認したところ、比較対象にしていた 487 s の run に既に入っていた。
測定済みの構成を把握していなかった。

**④ 「3〜5 倍の余地がある」と述べた。**
根拠が無かった。撤回し、現実的な上限は 1.6〜1.7 倍とした。

**⑤ メモリが減ったことを根拠に「no_sync が効いていない」と推論した。**
`perf/max_memory_allocated_gb` が 59.4 → 57.3 と下がったので、
unsharded 勾配 +2.3 GB と矛盾すると考えた。しかしこの指標は 48 GiB を超える値を
返しており（2.5 節）、そもそも比較に使える精度が無かった。
ログを grep すれば 1 コマンドで確定する話を、信頼できない指標から推論しようとした。

**⑥ MFU を「約 10%」と述べた。**
測定していない。当時 `perf/mfu/actor` は分母のバグで 0.000 しか返していなかった。
手計算では `6ND` 概算に gradient checkpointing の再計算を足して 1 GPU あたり
約 42 TFLOPS、154.8 に対して **概ね 25〜30%** になる。いずれにせよ推定値であり、
確定値は分母を直した `3d21864` 以降の run で取る。

**⑦ `position_ids` の検査式を間違え、正常な pool を異常と報告した。**
`clamp(cumsum(attention_mask)-1, 0)` を行全体に当てたが、vLLM は response 側の
position を右パディングを突き抜けて振り続ける（`vllm_rollout_spmd.py:366` に
コメントで明記されている）。誤検出率が `at cap` の割合と行単位で一致したことで
「データではなく検査が悪い」と判定できた。**通ったケースの分布が診断の決め手になった。**

---

## 6. 結果

| | 変更前 | 変更後 |
|---|---:|---:|
| `timing_s/update_actor_worker` | 487 s | 414.8 s |
| `actor.fwd` | 115.0 s | 93.6 s |
| `actor.bwd` | 320.0 s | 296.6 s |
| `actor.task_metrics` | 33.0 s | 19.5 s |
| throughput | 3,075 tok/s | 3,406 tok/s |
| 起動時の pool ロード | 79 分 | キャッシュ済み |
| 300 step の想定所要 | 40.6 時間 | 34.6 時間 |

**−15%。** うち no_sync が −45 s、メトリクス遅延集約が −13.5 s。

**この分解は推論であって測定ではない。** 2 つは同じ run で同時に入ったので、
「task_metrics の −13.5 s は遅延集約、fwd/bwd の −45 s は no_sync」という帰属は
機構から説明が付くというだけである。分離するには
`++actor_rollout_ref.actor.no_sync_grad_accum=False` で数 step 走らせて比べればよい
（20 分程度）。

**前後で `sm%` は比較していない。** 改良前の run の phase 表は残していないが、
取っていたとしても使えない。`sm_util` は NCCL のカーネルも busy に数えるので、
集団通信を減らす改良は **busy 率ではなく busy な時間そのもの**を削る（2.1 節）。
前後ともほぼ 98% になることが期待され、差は出ない。本来これを示すべき指標は MFU だが、
両方の run で分母のバグにより 0.000 だった（2.5 節）。`3d21864` 以降は使える。

残っている余地は response-only lm_head の −20〜30 s 程度で、
それを入れても 1.2 倍には届かない。**通信律速の下限（2.4 節）がすぐそこにある。**
これ以上を狙うなら GPU の物理配置か GPU 数を変えるしかない。

---

## 7. 再現手順

```bash
export GPU_PROFILER=1
export GPU_PROFILER_INTERVAL=0.1        # 既定 0.3。一瞬の落ち込みを追うとき
export GPU_PROFILER_IDLE_THRESH=60      # 既定 30 では 45% の落ち込みを idle と数えない
export GPU_PROFILER_ROLLUP_EVERY=1      # 既定 25。数 step しか回さないとき
export GPU_PROFILER_TRACE=/tmp/gpu_trace.csv
```

trace の読み方:

```bash
# 低稼働のサンプルとその前後
awk -F, 'NR==1 || $4+0 < 60' /tmp/gpu_trace.csv | head -40
```

**本実験では絶対に立てないこと。** サンプラがドライバと rank 0 に常駐し、毎 step 表を
出力する。特に `GPU_PROFILER_SYNC_PHASES=1` は phase 境界ごとに
`device.synchronize()` を入れるので、本来重なる処理を直列化して実際に遅くなる。
未設定なら phase タグは即 `yield` して返るだけで、プロファイラが無い場合と同一の実行になる。

---

## 8. この後の変更

計測後、OOM 対策で `ppo_micro_batch_size_per_gpu` を 10 → 5 に下げた（`2c9d398`）。
このとき `gradient_accumulation` は 2 → 4 になる。

**no_sync との相互作用に注意。** no_sync 下では通信は mini-batch あたり 1 回で固定なので、
micro を半分にしても通信は増えない。no_sync が無ければ micro 半減は通信 2 倍を意味していた。
**micro=5 が成立するのは no_sync が入っているからで、順序が逆だと成立しなかった。**

代償として micro-batch 数が step あたり 142 → 284 になり、カーネル起動と Python の
オーバーヘッドが倍増する。本レポートの 415 s/step は micro=10 の値であり、
現構成の実測値ではない。
