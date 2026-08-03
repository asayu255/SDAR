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

trace CSV を突き合わせた結果、落ち込みは step 境界にのみ現れ、内訳は

- step 後のメトリクス計算（`(idle/other)` phase、約 0.5 s）
- Ray のバッチ dispatch（`update_actor − update_actor_worker` = 0.7〜1.1 s）

合計で wall clock の **約 0.4%**。チャート上の目立ち方と実コストが 2 桁乖離していた。

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
| 1 | FSDP ZeRO-2（`shard_grad_op`） | forward→backward 間の再 all-gather 除去 | 中立 | `cb60d7b` |
| 2 | `no_sync` 勾配蓄積 | **−45 s/step** | 丸め順序のみ | `154e8dd` |
| 3 | `forward_prefetch` | all-gather 発行の前倒し | 中立 | `960a16a` `ebd67d2` |
| 4 | メトリクス遅延集約 | **−13.5 s/step** | 中立（かつバグ修正） | `65797f7` |
| 5 | pool の shard 分割ロード | ピーク 360 GiB → 常駐+最大 shard | 完全同一 | `360317a` |
| 6 | pool のオフラインキャッシュ | 339.5 → 136.5 GiB、起動から 79 分削除 | ビット同一 | `47fb1e6` |
| 7 | バッチ prefetch | dispatch と計算の重畳 | ビット同一 | 既存機構 |

### 3.1 `no_sync` 勾配蓄積（最大の効果）

`gradient_accumulation=2` のとき、既定の FSDP は micro-batch ごとに勾配を
reduce-scatter する。つまり mini-batch あたり 2 回。
FSDP の `no_sync()` を最後の micro-batch 以外に掛ければ 1 回で済む。

これは通信回数を半減させるだけだと思っていたが、実際には **forward も速くなった**
（115 → 93.6 s）。理由を torch 2.13 のソースで確認した。

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

さらに `_post_backward_hook` を読むと:

```python
_post_backward_reshard(state, handle)
if not state._sync_gradients:
    if handle._use_orig_params:
        handle._use_unsharded_grad_views()
    return          # reduce-scatter せずに戻る
```

`flat_param.grad` が **unsharded のまま累積される**。1.7B・bf16 なら
sharded 1.13 GB → unsharded 3.4 GB、差し引き **約 +2.3 GB** のメモリと引き換え。
これが後の OOM の一因になり、micro batch を 5 に下げる判断につながった。

実装は `dp_actor.py` の `_grad_sync_context`。FSDP1 の `no_sync()` は
`TrainingState.IDLE` を要求するので backward だけを包むことができず、
micro-batch の**境界**で enter / exit する必要がある。

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

**演算的には中立ではない。** unsharded の micro-batch 勾配を先に足してから 1 回
reduce-scatter するので、shard ごとに reduce-scatter して足す場合と浮動小数点の
結合順序が変わる。差は丸め誤差レベルだが、勾配の作り方が変わる以上
`expected_multitask_sft_config.yaml` に載せた。

### 3.2 メトリクス遅延集約（ついでにバグ修正だった）

`actor/sft_loss` 等の per-task メトリクスを micro-batch ループの中で `.item()` して
いたため、micro-batch ごとに CPU-GPU 同期が入っていた。`torch.Tensor` のまま溜めて
最後に 1 回 `torch.stack(...).mean().item()` する形に変えた。−13.5 s。

**同時に実害のあるバグが直った。** `actor/sft_loss_weighted` はループ内で
**代入**していたので最後の micro-batch の値しか残らない。
`_balance_batch` の並べ替え後、最後の micro-batch は adjust_batch の padding 行
（重み 0）だけになることが多く、**0.000 が記録され続けていた**。
padding が 3 行しかなかった step 5 でだけ 0.492 が出て、それで気づいた。

「メトリクスがゼロ」を書式の問題だと思い込みかけたが、実際には集計の欠陥だった。

### 3.3 pool の扱い（起動時間とホスト RAM）

Stage 1 の出力は 90 shard・339.5 GiB。これを毎回読んで、adjust_batch の padding 行と
この arm の損失が読まない列（`prompts` / `response_mask` / `teacher_topk_*`）を捨てて
136.5 GiB にしていた。**毎起動・毎再起動で 2.4 倍のディスク I/O と unpickle を払っていた。**

- **shard を concat しない**（`360317a`）: 軌跡 → (shard 番号, 行番号) の索引を持ち、
  抽選時に必要な shard からだけ `select_idxs` する。ピークが
  「常駐 + 全体のコピー」から「常駐 + 最大 shard（3.8 GiB）」になった。
- **フィルタを 1 回だけ行う**（`47fb1e6`, `scripts/cache_teacher_pool.py`）:
  トレーナがメモリ上に作るのと同じ DataProto をそのままディスクに書く。
  入力 1 ファイルにつき出力 1 ファイル、同じ basename・同じ行順なので
  `sorted(glob)` の順序が変わらず、**軌跡の抽選列がビット単位で同一**
  （`tests/trainer/test_cache_teacher_pool.py` で 5 step 分をテンソル比較して確認）。
  79 分が起動から消えた。

キャッシュは arm 固有（SFT 版には top-k が無い）なので `_cache_manifest.json` に
arm を記録している。

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
