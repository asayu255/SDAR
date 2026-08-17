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

---

## 9. 完走した run の全期間内訳（wandb x7g9r7bx）

7 節までは `update_actor` の内側の話で、計測範囲が 1 step の中に閉じていた。
300 step を完走した run（micro=5、`test_freq=150`、42.5 時間）の wandb を
フェーズに割り当て直したところ、**残っていた無駄は step の内側ではなく外側にあった**。

### 9.1 内訳

`system.gpu.*.gpu`（15 s サンプル、3 GPU 平均）を、`timing_s/step`・`update_actor`・
`save_checkpoint`・`testing` から復元した窓に割り当てた:

| フェーズ | 壁時間 | 全体比 | 平均 util | 「100% との差」の面積 |
|---|---:|---:|---:|---:|
| 学習（`update_actor`） | 34.14 h | 80.4% | 96.5% | 20.0% |
| **検証（`testing`）** | **7.64 h** | **18.0%** | **46.0%** | **68.6%** |
| checkpoint 保存 | 0.67 h | 1.6% | 2.1% | 10.8% |
| その他 | 0.03 h | 0.1% | — | 0.6% |
| **合計** | **42.48 h** | 100% | **85.8%** | 100% |

同 run の `perf/mfu/actor` は **0.254**、`perf/throughput` は 3,043 tok/s。
6 節の手計算（25〜30%）と一致し、`3d21864` の分母修正が効いていることも確認できた。

**検証 1 回が 13,500〜14,000 秒（3.8 時間）。** これが 2 回で 7.6 時間、
run の 18%、そして**取りこぼした GPU 時間の 68.6%** を占めていた。
7 節までの最適化が削ったのが 1 step あたり 72 秒（300 step で 6 時間）なので、
**同じ桁の無駄が、測っていなかった外側に残っていた**ことになる。

### 9.2 なぜ検証だけ 46% なのか

検証はこの arm の損失ではない。SFT の学習は教師トークンへの teacher forcing で、
GPU から見れば fwd/bwd しかない（だから 96.5%）。一方 `_validate` は
**エージェント rollout** で、1 ターンが

```
preprocess(CPU) → vLLM 生成(GPU, デコード律速) → decode(CPU) → env.step(GPU 完全 idle)
```

のロックステップになる。検証窓のサンプル分布は二峰性で、
**42% が util 10% 未満**（env.step・リセット・リトリーバ待ち）、
44% が 80〜100%（生成中）。平均 46% は「常に半分使っている」ではなく
「**半分近くは完全に止まっている**」の意味である。2.1 節と同じ読み違いをしやすい。

タスク構成も効いている。3 タスク × 126 エピソードを `val_dataloader` が
タスク単位で順に回し、alfworld だけ 50 ターンまで走る。search（4 ターン）と
webshop（15 ターン）が終わっても alfworld の尻尾が残り、そこは小さくなった
バッチのデコードだけになる。

### 9.3 計測の落とし穴（1 回間違えた）

wandb の `events` ストリームは **GPU 系と非 GPU 系のサンプルが交互**に入っていて、
`system.gpu.*` は 1 行おきに NaN になる（20,389 行中 10,194 行）。
これを落とさずに平均を取ると学習フェーズが 94.0%、落とすと 96.5% になる。
最初の集計は前者で出していた。

サンプル間隔も GPU 系は 15 秒で、2 節の 0.1 秒トレースとは 2 桁違う。
**step 境界の 1 秒未満の落ち込みは wandb のチャートにはそもそも写らない**ので、
9.1 の学習フェーズ 96.5% と 2.3 節の 98.6% は矛盾していない（粒度が違う）。
チャートで谷に見えるものと、プロファイラで谷に見えるものは別物である。

### 9.4 やったこと — 検証を学習ループから外した

`trainer.test_freq=-1` にして、checkpoint を後から別プロセスで採点する
（`examples/sft_trainer/eval_checkpoints.sh`）。実装は 4 点:

1. **`trainer.test_freq=-1`**（run script）と、それを期待値ファイルに固定。
   性能ノブではなく「この run が何を測るか」の決定なので lock に載せた。
   併せて `save_freq=25` も固定した — checkpoint は評価の入力になったので、
   それを書く頻度が評価側の前提になる。
2. **`val_only` が pool を読まないようにした**（`_maybe_load_offpolicy_data`）。
   検証しかしないプロセスにとって 136.5 GiB の Stage-1 pool は完全な死荷重で、
   これを毎 checkpoint 払うなら分離する意味がなくなる。`algorithm.sft.data_dir`
   の存在チェックは arm の同一性の確認なので残し、glob だけを飛ばす。
3. **`val_only` だけで検証が走るようにした**（`_should_validate_before_training`）。
   run script は `val_before_train=False` を渡すので、以前は
   `val_only=True` を足しても**何もせず正常終了する**組み合わせだった。
   これを直したことで、評価コマンドが足すキーが
   run script の渡さないキー（`val_only` / `resume_*`）だけで済み、
   **同じキーを Hydra に 2 回渡さない**構成になった。
4. **評価ドライバ**は run script 自体を呼ぶ。引数を書き写さないので、
   タスク・ターン上限・`val_kwargs_by_task`・リトリーバが定義上一致し、
   期待値ファイルも評価側で再度効く。

**checkpoint ごとに別プロセスなのは意図的。** alfworld のエピソードは
TextWorld の seed 付き game-file cycle から引かれ、`reset()` のたびに進む
（`skip_games` が触っているのがその iterator）。同一プロセスで 2 回検証すると
**2 回目は別のゲームを引く**ので、checkpoint 間の差にタスクの差が混ざる。
プロセスを作り直せば cycle が `env.seed` から作り直され、どの checkpoint も
同じエピソードで採点される。モデルロードと env 構築を毎回払うが、
数時間の検証に対して数分であり、買えるものが「前の評価の状態が一切残らない」なので釣り合う。

なお **`test_freq=-1` は学習プロセスから val env 252 個も消す**。`make_envs` は
`LazyEnvManager` を返すので（`c4d520f`）、`_validate` が呼ばれなければ実体化されない。
学習プロセスは pool を持ち val env を持たない、評価プロセスは val env を持ち
pool を持たない、という分かれ方になる。`28ef08e` が「両方を同時に持つのが
step 150 で run を終わらせた原因（GPU ではなくホスト RAM）」と書いているのは
この話で、その分離がようやく既定になった。

### 9.5 これで何が変わって、何が変わらないか

**変わる。** 学習 run は 42.5 → 34.8 時間（`save_checkpoint` は残るので 34.1 ではない）、
チャートは全期間 96.5% でほぼ平坦になる。評価は独立に走らせ直せるので、
リトリーバの不調が学習 run を道連れにすることもなくなる。
checkpoint 間の比較可能性は 9.4 の理由で**上がる**。

**変わらない。** 検証そのものは速くならない。3.8 時間は 3.8 時間のままで、
学習 + 評価の GPU 時間の合計はほぼ同じである。移動しただけで、消してはいない。
`eval_checkpoints.sh` が引数なしで最新の 1 個しか評価しないのはそのためで、
12 個回せば 45 時間かかる。

**期待してはいけない。** 「vLLM が学習中に握っているメモリが返るので micro を
戻せる」と一度考えたが、誤り。`FSDPVLLMShardingManager.__exit__` は
`sleep(level=1)` を呼んでおり（`fsdp_vllm.py:224`）、rollout の外では
weights は CPU に落ち KV cache は捨てられている。学習中の
`memoryAllocated` 78% は FSDP 側の実体で、検証を外しても空かない。
micro=10 / gradient checkpointing 無効化を狙うなら、
**学習プロセスで rollout engine をそもそも作らない**（worker の役割を actor だけにする）
という別の変更が要る。9.1 の表はその判断材料にはなっていない。

### 9.6 checkpoint 保存の非同期化

分離後に残る谷はこれだけになるので、続けて実装した
（`actor.checkpoint.async_save=True`）。

**198 秒の内訳を先に測った。** 12 回分 160 サンプルを窓内の位置で割ると、
きれいに 2 つに分かれる:

| 窓内の位置 | SM util | メモリコントローラ | 電力 |
|---|---:|---:|---:|
| 0〜20 s | 18.5% | 2.5% | 83 W |
| **20〜198 s** | **0.0%** | **0.0%** | **28 W** |

前半 20 秒が sharded state dict の構築とホストへのコピー、
**後半約 178 秒は GPU が完全に停止している**。電力がアイドル下限の 28 W、
メモリコントローラも 0.0% なので DMA すら走っていない。`torch.save` の
pickle とディスク書き込みだけで、その間トレーニングループが待っている。

**動かしたもの / 動かさなかったもの。** バックグラウンドスレッドに出したのは
`torch.save` 3 本だけ。`state_dict()` は collective かつデバイスコピーなので残し、
`torch.distributed.barrier()` も残した — **NCCL collective を別スレッドから
撃つと、メインスレッドの学習 collective と rank 間で順序が食い違う**。
書き込みが触るのは `offload_to_cpu=True` が作った CPU コピーなので、
次の step が GPU 上のパラメータを更新しても汚れない。これが安全性の根拠で、
`is_cuda_available` が偽（= offload されない）なら async は自動的に無効になる。

**tracker の順序が本質。** `latest_checkpointed_iteration.txt` は resume が読む。
書き込み完了前に step N を publish すると、**クラッシュ時に書きかけの
ディレクトリを指した tracker が残る**。そこで publish を
`_flush_pending_checkpoint`（`ray_trainer.py`）に移し、全 rank の
`wait_for_checkpoint` が返ってから書くようにした。

呼ぶ位置も選んでいる。`update_actor` の**後**に置く — 前に置くと join が
そのまま待ち時間になり、非同期にした意味が消える。178 秒の書き込みに対して
step が 415 秒なので、次の step が終わる頃には確実に終わっており join は 0 秒。
結果、tracker は 1 step ぶん遅れる（約 7 分）だけで済む。ループ脱出後にもう 1 回
呼ぶのは最終 checkpoint のためで、これだけは隠す相手がいないので実際に待つ。

`finally` には**入れていない**。ここで死んだ run は tracker が 1 つ前の
checkpoint を指したまま残るが、それが望ましい状態である（現在の checkpoint は
書きかけかもしれない）。また unwind 中に例外を上げると、run を止めた本来の
失敗を覆い隠す。

**失敗の扱い。** 書き込みスレッドが捕まえた例外は保持し、次の
`wait_for_pending_save` でメインスレッドに再送出する。これが無いと
「run は正常終了、checkpoint は 1 つも無い」という、この機構が持ち込みうる
最悪の失敗になる。

**見込み。** 窓 198 → 約 20 秒、12 回で **約 36 分**。評価分離後の 34.8 h・
94.7% に対して **34.2 h・約 96.3%**（= 学習フェーズ自身の 96.5% とほぼ同じ）。
壁時計としては 1.7% だが、これで**構造的な谷が無くなる**。

**未確定。** 17 GB を 178 秒 = 約 95 MB/s は遅い。書き込み先が
ネットワーク FS なのか、`torch.save` の pickle が律速なのかは**切り分けていない**。
pickle 律速ならスレッドが GIL を握る時間が長く、`OFFPOLICY_BATCH_PREFETCH` の
準備スレッドと競合しうる（ただし準備は step あたり数秒なので、
415 秒の step に対しては吸収されるはず）。`torch.save` の前後に計時を入れれば確定する。
また、この 36 分は `save_freq` を 25 → 50 にしても半減する。そちらは無改造だが
resume 粒度と評価点が減る。

### 9.7 pure OPD arm から移した 3 機構

`claude/pure-opd-multitask` の `docs/speedup_mechanisms.md` と照合したところ、
**あちらで稼働しているのにこちらに無い機構が 3 つ**あった。teacher 系や
`old_log_prob` 依存のものは SFT には該当しないが、この 3 つは該当する。

#### `actor.response_only_logits`（最大。学習ステップに効く）

`_forward_micro_batch` が返すものは**すべて** `[:, -response_length-1:-1]` に
切られる。`use_remove_padding=True` なので transformer 本体のパディングは既に
除かれているが、**`lm_head` は packed された全トークンに対して
`(rows, 151936)` を作り、その 75.7% を捨てていた**（prompt 平均 662.8 /
response 平均 212.6、4 節）。forward と backward の両方で、しかも step 中で
最大の活性テンソルである。

`logits_to_keep` に**行番号のテンソル**を渡すと HF が
`slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep`
で解決するので、lm_head だけを応答行に絞れる。transformer 本体は全トークンを
見たまま（因果 attention なので応答位置は prompt の KV を読む）。

**精度は「同じ演算・違う GEMM 形状」。** lm_head は位置ごとの線形写像なので、
行を選ぶのが射影の前でも後でも値は同じ。ただし GEMM の形が変わるので
最下位ビットは micro-batch を変えたときと同じように動く。
したがって `expected_multitask_sft_config.yaml` に pin した ——
`no_sync_grad_accum` を pin したのと同じ理由である。

**これは性能ノブである以上に arm 間の一貫性の問題だった。** pure OPD arm は
これを本番構成で有効にし、intent lock で `true` に固定している。SFT arm に
無い状態では、**「SFT と KD は損失以外すべて同一」という run script の主張が
成り立っていなかった**。あちらの文書も「so it goes into every arm at once」と
書いている。

#### `rollout.return_rollout_log_probs=False`（評価プロセスに効く）

vLLM に sampled token の log-prob を要求し、生成トークンごとの Python ループで
列を組み立てていた。**唯一の消費者は `RayPPOTrainer.fit` の
rollout-vs-actor drift 検査**で、この arm の薄いループには比較対象の
`old_log_prob` フェーズが無い。作って捨てていた。
生成トークン自体は変わらない。drift 検査を回す arm では `True` のままにすること。

#### rollout session 中の `empty_cache` 抑止（評価プロセスに効く）

`generate_sequences` の末尾の `empty_cache()` が session 判定の外にあり、
vLLM を起こしたままにする ① を入れてもなお**毎ターン走ってデバイス同期を
強制していた**（1 rollout に約 50 回）。しかも vLLM の KV は解放できない
（エンジンが所有しており、`empty_cache` が返すのは*未使用*ブロックだけ）。
session 機構はこちらにもあったのに、この抑止だけが抜けていた。

#### 併せて `flops_counter` に RTX PRO 6000 Blackwell を追加

A6000 は追加済みだった（`3d21864`）が Blackwell が無く、そのホストで回すと
2.5 節と同じ形で `perf/mfu/actor` がまた `0.000` になる。

#### 未計測であることの明示

**3 つとも、この arm では効果を測っていない。** 移植元の pure OPD arm では
response-only 化（teacher 側の lse/topk/gather）が
**ピークメモリ 121.2 → 93.9 GB** を出しているが、それは別の機構
（同じ「応答行だけ」の考え方を top-k KL に適用したもの）の数字であり、
`lm_head` 版の単独計測ではない。こちらでの効果は
**`actor.fwd` / `actor.bwd` の時間と `max_memory_allocated`（`nvidia-smi` を正とする、
2.5 節）を前後で比べて確定させること。** 活性が減るぶん micro=5 → 10 に
戻せる可能性があり、そちらの方が効果として大きいかもしれない。

### 9.8 adjust_batch まわり —— 直したものと、判断待ちのもの

同じ照合で `adjust_batch` に関する指摘が 2 件出た。片方は潜在的なクラッシュ、
もう片方は**実測 2.77% の無駄**だが run の同一性に触る。

#### 直した: ミニバッチ数を厳密除算から切り上げに

`_attach_sft_loss_weights` は

```python
num_mini_batches = len(batch) / mini_batch_size
assert float(num_mini_batches).is_integer()
```

としていた。`adjust_batch` が丸める先は
`lcm(log_prob_micro × W, ppo_micro × W)` であって `ppo_mini_batch_size` ではない。
**現構成（3 GPU）では divisor が 240 で 60 がこれを割り切るため、たまたま常に成立する。**
しかし 2 GPU にすると divisor は 160 になり、60 との lcm は 480 ——
**3 step に 1 回しか成立しない**。pure OPD arm はこれで step 2 で落ちている。

短いミニバッチで終わること自体に問題はない。`update_policy` は
`batch.split(ppo_mini_batch_size)` で切るので短い末尾は通常の状態であり、
**損失は*設定値*の `gradient_accumulation` で割られ、重みは同じ定数で掛けられる**ので
両者は相殺し、ミニバッチは行数に関わらずその行の重み付き損失の和をちょうど寄与する。
`num_mini_batches` が決めるのは全体のスケールだけなので、
**そのバッチが何回の optimizer step になるか＝切り上げ**でなければならない。

`math.ceil` に変えてアサートを外した。**現構成では値が変わらない**（常に厳密除算だった）。
GPU 数やマイクロバッチを変えた瞬間にランダムな step で落ちる地雷を外しただけである。

#### 直した(2): step バッチの padding を、このループが必要とする分だけに

**この arm は `compute_log_prob` を一度も呼ばない。** トレーナのループは
`update_actor` だけで `old_log_prob` フェーズが無く、検証も
`recompute_log_prob: False` である。したがって
`rollout.log_prob_micro_batch_size_per_gpu=16` の**唯一の効果は
`adjust_batch` の divisor を決めること**で、16 × 3 = 48 と
`ppo_micro × W` = 15 の lcm で **240** になる。

`adjust_batch(mode="copy")` はその倍数まで**行を複製して埋める**。実測（x7g9r7bx）:

| | 値 |
|---|---:|
| 実行 | 4,260.8 行/step |
| padding | **122.4 行/step**（最大 239） |
| 割合 | **2.77%**（最大 5.76%） |

平均 122.4 ≈ 240/2 で、余りが一様に散る形と一致する。
SFT では padding 行は重み 0 なので損失には効かないが、
**forward と backward は満額払っている**。2.77% が完全な空回りである。

**当初は「`log_prob_micro_batch_size_per_gpu` を下げる」を考えたが、それは筋が悪い。**
継承元のノブを触って副作用を避ける迂回であり、しかも off-policy KD arm も
同じ値を使うので両アームを縛る。正しくは **Stage 2 が自分の制約から divisor を出す**ことである。

**実際の制約はどこまでか。** `_prepare_batch` の旧コメントは 2 つ挙げていた:

1. `_balance_batch` の partitioner —— `get_seqlen_balanced_partitions(..., equal_size=True)`
   は「行数が `k_partitions` で割り切れること」を要求する。**k_partitions は world_size = 3。**
2. 「短い末尾ミニバッチは誤スケールする」—— **9.8 の切り上げ修正で否定された。**
   重み付き SFT では割る定数と掛ける定数が同じなので相殺する。

つまり残っているのは **3 で割り切れること**だけだった。
`_step_batch_divisor` を足し、SFT アームのときだけ divisor を
`ppo_micro × world_size = 15` にする（`adjust_batch` に `size_divisor` 引数を追加）。

実測 300 step に当てはめた結果:

| | divisor 240 | divisor 15 |
|---|---:|---:|
| padding | 122.4 行/step | **7.2 行/step** |
| バッチ比 | 2.77% | 0.17% |
| ミニバッチ数/step | 73.1 | 71.5 |

**戻るのは 2.60%**（300 step で 34,575 行）。

**なぜ 3 ではなく 15 か。** 3 まで落とせば padding は約 1 行になり、さらに
micro batch size が完全にデータから外れる（利点）。しかし `ppo_micro × W` なら
**すべての micro-batch が満杯のまま**で、新たに通る形は「短いミニバッチ」だけ ——
これは pure OPD arm が本番で通している形である。3 にすると「短い micro-batch」という
**どちらのアームでも一度も走ったことのない経路**が加わる。差は 0.14% なので、
実績のある経路に留めた。必要になれば 3 に落とせる。

**SFT アーム限定である。** 重みの無い経路（off-policy KD）には相殺が無く、
短いミニバッチは実際に誤スケールする —— そこは旧コメントの後半が今も正しい。
`use_sft_loss` が立っていないときは `None` を返して従来どおりの divisor に戻る。
`use_dynamic_bsz`（トークン数で micro を切る）と、per-GPU の micro が
未設定のときも同様。

**代償。** どの行が複製されるかと、1 step が何回の optimizer step になるかが変わる。
`ppo_micro_batch_size_per_gpu` は**このアームでは純粋な性能ノブではなくなった** ——
ただし divisor はもともと micro サイズの lcm だったので、完全に自由だったわけでもない。
expectations ファイルの「性能ノブは pin しない」規則の脇に、この例外を明記した。

### 9.9 次に残っているもの

* **学習フェーズの 96.5%**: 残り 3.5% は step 境界と micro=5 の起動オーバーヘッド。
  2.2 節のとおり面積は小さい。
* **MFU 25.4%**: 2.4 節の通信律速。util を 100% にしてもここは動かない。
  指標としては util より tok/s と MFU を見るべきである。
