# スパイク調査 —— 全経過

学習中に GPU 使用率が落ちる現象（以下「スパイク」）を追った記録。
何を測り、何が分かり、何を間違えたか。時系列。

目的は一貫して **SFT の全期間で GPU 使用率 100%** であり、スパイクの除去は
そのための必須項目である。この文書はスパイクだけを扱う。調査中に見つかった
別の損失（NCCL の待ち、16.5%）は 9.13 節に分離してある。

---

## 1. スパイクの定義

wandb のシステムメトリクス（15 秒間隔）で、1 枚または複数の GPU の
`utilization.gpu` が 100% から大きく落ちるサンプル。

観測例:

```
x74nckl6   step 3  sm= 33.0/ 39.0/  0.0    3 枚
           step 6  sm= 87.0/ 95.0/ 99.0    1 枚
           step 7  sm=  9.0/  8.0/ 59.0    3 枚
           step 8  sm= 89.0/ 88.0/100.0    2 枚
           step 11 sm= 29.0/ 72.0/ 89.0    3 枚
0mjkp6c2   step 55 sm= 16.0/100.0/ 96.0    1 枚
07hnhxjn   step 8  sm=  3.0/  0.0/ 20.0    3 枚
```

直接観測は 3 run 合計で 7 件（別に profiler 由来が 3 件、resume 由来が 7 件
あるが、それらはスパイクではない）。

---

## 2. 計測器と、それぞれが答えたこと

### 2.1 wandb システムメトリクス（15 秒間隔）

最初の観測源。ここから **deficit 積分** `Σ (100 − sm)/100 · dt` で総損失を出す。

```
07hnhxjn steps 2-7（41.8 分、profiler の step 1 は除外）
  gpu0 1.18%   gpu1 1.12%   gpu2 1.09%   node 1.13%
```

**これがスパイク問題の全体サイズであり、天井である。** 内訳がどうであれ、
ここを完全に消して戻るのは 1.1% である。

**重大な注意**: `utilization.gpu` は **約 1 秒の移動窓**の busy 率であって、
15 秒はサンプル間隔にすぎない。この 2 つを混同すると、イベントが
**15 倍まれで 15 倍大きく**見える。本調査ではこの読み違いを 2 回犯した（4 節）。

### 2.2 NVML サンプラー 0.2 秒（`gpu_profiler`）

`GPU_PROFILER=1 GPU_PROFILER_INTERVAL=0.2 GPU_PROFILER_TRACE=...`。
`scripts/gpu_stall_scan.py` で読む。

出た答え: **~12 件/step、1 件 0.3〜1.0 秒、1 枚だけ、`actor.fwd` の中**。

**イベントを実際に解像した唯一の計測器である。** ただし「forward のどこか」
までで、何が起きているかは言えない。

### 2.3 Nsight Systems（`ACTOR_NSYS_MICRO=20`）

step 1 の micro-batch 40〜59 を 3 rank でキャプチャ。結果は三重の失敗:

1. **窓にイベントが入らなかった**
2. **`.qdstrm` → `.nsys-rep` の変換が 3 本とも失敗**
   （`QdstrmImporter`: "Wrong event order has been detected"）
3. **キャプチャ自身が run 最大のスパイクになった** ——
   自分の窓の中で 30 秒のノード全停止、その step の MFU 0.278 対 0.317

`ACTOR_NSYS_TRACE` で osrt を外せば変換が通る可能性はあるが、未検証。

### 2.4 stall watch（`actor_capture.py`、既定オン）

全 rank・全 step・全 micro-batch に CUDA event を 2 個ずつ。読み戻しは
`query()` が完了を返した時だけで `synchronize()` は呼ばない
（呼べば計測器が探しているストールを自分で作る）。

出力は 2 種類。異常値の `[stall]` 行と、毎 step の `[step-gpu]` 行。
gap は 3 種類に分けて判定する —— `gap/step`（update_policy 間）、
`gap/mini`（gradient reduce + optimizer step）、`gap/interior`（micro-batch 間）。
分けないと構造的に大きい境界の gap が毎回発火して、300 行の中に本命が埋まる。

**7 step 連続の結果:**

```
[step-gpu] rank 2 step 2: 77 micro, in-micro 343.2 s, outside 6.52 s
           (before-step 0.29, before-mini 6.23, interior 0.00;
            optim 6.31, unaccounted 0.21 s) = 0.06% idle
... step 3: 0.06%   step 4: 0.20%   step 5: 0.06%   step 6: 0.05%   step 7: 0.06%
```

**これが本調査で最も価値のある否定的結果である。micro-batch の外は
0.05〜0.20% でクリーンであり、7 step 連続で再現する。** すなわち
1.1% の deficit は **ほぼ全部 micro-batch の中**にある。

`[stall]` が発火したのは 1 回だけで、それは torch profiler 自身の
597 MB 書き出し（21.9 秒、3 rank 同時、中央値の 274 倍）だった。

**watch の死角**: `in-micro` は micro-batch の start event から end event までの
**長さ**であって**占有時間**ではない。micro-batch の中で 0.5 秒デバイスが
空いても、それは `in-micro` に含まれて見えない。スパイクが watch に映らないのは
故障ではなく、測っている量が違うためである。

### 2.5 torch.profiler（`ACTOR_TORCH_MICRO=20`）

step 1 の micro-batch 40〜59、3 rank、各 597 MB。
`scripts/actor_trace_summary.py` で読む。Nsight と違い変換工程が無い。

```
device idle: rank0 2.22%  rank1 2.03%  rank2 2.18%
             micro-batch あたり 98〜158 ms でほぼ一定、バースト無し
最大ギャップ: 65.8 ms、host op は aten::nonzero、micro-batch の先頭、3 rank 同一値
```

**この窓には深い dip が入っていない。** `sm=16` を出すには 1 秒窓のうち
約 840 ms 空いている必要があるが、最大 65.8 ms で 13 倍足りない。
深い dip は約 30 分に 1 件なので、111 秒のキャプチャでの期待値は 0.06 件 ——
**窓が狭かったのではなく 16 倍短い**（step 全体に広げても 3 倍にしかならない）。

---

## 3. 今、確定していること

### 3.1 スパイクではないもの（すべて数字付きで排除済み）

| 候補 | 実測 | 判定 |
| --- | --- | --- |
| step 境界（driver・Ray・H2D・ログ） | `before-step` 0.24〜0.71 s/step | × |
| gradient reduce + optimizer step | `before-mini` ≒ `optim`、実カーネル | × |
| micro-batch 間 | `interior` 0.00 s | × |
| rank 間のワーク不均衡（straggler） | 深い dip は 3 枚同時。待つ側は NCCL で busy に見えるはず | × |
| ディスク I/O | dip 時 0.02/s（平常 0.04） | × |
| ネットワーク | dip 時 0.4〜2.2 MB/s（平常中央値 1.0、p95 3.6） | × |
| ホストメモリ | 通して 12.8%、空き 439 GiB | × |
| allocator retry | 7 step で 2 回（予測 ~84） | × |
| vLLM | rollout 構築でデバイスは **−3.5 GiB**（きれいになる） | × |
| teacher pool | driver のホスト RAM。GPU を 1 バイトも使わない | × |

### 3.2 スパイクであるもの

**micro-batch の中（forward / backward）。** 総量 1.1%。

現時点の最有力候補は **`aten::nonzero` による device→host 同期**:

* micro-batch の先頭、3 rank 共通、**65.8 ms で値まで同一**、繰り返す
* `nonzero` は出力サイズがデータ依存なので、ホストへのコピーを強制する
* 候補は 2 箇所 —— `unpad_input` 内（`verl/utils/torch_functional.py:579`、
  および flash-attn 側）と、`response_row_selection`
  （`verl/workers/actor/dp_actor.py:140`、`response_only_logits=True` の経路）
* 1 step 66〜77 micro-batch × 65.8 ms = 4.3〜5.1 秒 ≒ **350 秒 step の 1.2〜1.5%**

**この 1 つで 1.1% の deficit がほぼ説明できる。** ただし断定はしない ——
65.8 ms がぴったり同じ値で繰り返すのは単なる同期のコストとしては大きすぎ、
機構が完全には分かっていない。

未解決なのは、まれに出る深いサンプル（`sm=3/0/20` など）が
この均一な 65.8 ms 群と同一のものなのか、別物なのか。トレースの窓には
入らなかったので、まだ答えが無い。

---

## 4. 間違えたこと

### 4.1 NVML の窓を 15 秒と読んだ（2 回）

`utilization.gpu` は約 1 秒の移動窓の busy 率であり、15 秒はサンプル間隔である。
この読み違いにより:

* イベントが実際の **1/15 の頻度**に見えた（5 件/12 step ← 実際は ~75 件相当）
* 1 件あたり **15 倍大きく**見えた（「12.6 秒の穴」← 実際は 0.84 秒）

**この 1 つの誤りが、その後の判断をほぼ全部狂わせた。** 具体的には:

* 「固定窓には当たらない、確率ほぼゼロ」として torch profiler を却下した。
  実レートで計算すると 84〜98% で当たる。**却下したものが正解だった。**
* 代わりに全 run 検出器（stall watch）を作った。これ自体は
  「外側はクリーン」という決定的な否定的結果を出したので無駄ではないが、
  スパイクは原理的に映らない。
* 「rank 0 に 12.6 秒の穴があるはず」と予測した。実際は 0.09 秒だった。
* 「1 イベントで deficit の 49%」と書いた。これも同じ計算ミス。

### 4.2 optimizer step を「idle」と呼んだ

`before-mini` の 5.75 秒 / 346 秒を 6 step 分 idle と報告した。実際は
gradient reduce と Adam 更新の**実カーネル**（570M パラメータで 50〜80 ms、
実測 81 ms/mini-batch と一致）。**探しているストールの下に 1.7% の
偽のノイズ床を敷いていた。** `span("optim")` で名前付き計測して差し引く形に修正。

### 4.3 測れていない値から結論を出した

run 最初の micro-batch には比較対象の前イベントが無いので `before-step` は
構造的に 0.0 になる。それを「step 境界は GPU 上ゼロ」と読んだ。`n/a` を出す形に修正。

### 4.4 ユーザの override を 3 回落とした

`data_dir=teacher_traj_probe`、`ppo_micro_batch_size_per_gpu=20`、
`resume_mode=disable` の 3 つを、「これで回してください」という指示から
落とした。結果、フルプール（起動で 339.5 GiB 読む）、MFU 0.292（対 0.317）、
古い checkpoint からの再開（replay 102 秒）が起きた。
実行スクリプトのヘッダに 1 ブロックとして記録済み。

### 4.5 見つけたものと探していたものを混同した

NCCL の待ち 16.5% を見つけた時、これを「スパイクの正体」と報告した。
違う。**回っている collective は NVML から busy に見えるので、
スパイク（＝使用率が落ちる現象）ではない。** 別の損失である。

### 4.6 計測器が 3 回スパイクを作った

* Nsight のキャプチャ: step 1 で 30 秒のノード全停止
* torch profiler の書き出し: micro 60 の前で 21.9 秒、3 rank 同時
* `GPU_PROFILER_SYNC_PHASES`: phase 境界ごとに device synchronize

**observer effect はこの arm では仮説ではなく、実測で最大のイベントである。**

---

## 5. スパイク除去のために、次にやること

### 5.1 `aten::nonzero` の同期を消せるか調べる（最有力）

65.8 ms × 66〜77 micro-batch で 1.2〜1.5%。**deficit 1.1% とほぼ一致する。**

* `response_row_selection`（`dp_actor.py:140`）は `indices` が昇順で
  `cu_seqlens` も手元にあるので、`nonzero` を使わずに選択を構成できる可能性がある
* `unpad_input` 側の `nonzero` は flash-attn の API 境界にあるが、
  verl は `torch_functional.py:579` に自前の実装も持っている

まず**どちらの `nonzero` なのかを特定する**。トレースの host op は
`aten::nonzero` としか出ないので、片方に NVTX / `record_function` を
入れて 1 回取り直せば決まる。

### 5.2 深いサンプルが同一現象か確かめる

`sm=3/0/20` のような深いサンプルが、均一な 65.8 ms 群の一部なのか別物なのかは
未解決。0.2 秒サンプラーを併走させれば 0.9 秒のイベントに 4〜5 サンプル乗るので、
phase までは特定できる:

```bash
GPU_PROFILER=1 GPU_PROFILER_INTERVAL=0.2 GPU_PROFILER_TRACE=/tmp/trace.csv bash ...
python3 scripts/gpu_stall_scan.py /tmp/trace.*.csv
```

stall watch と併走させても干渉しない（watch は synchronize しない）。

### 5.3 それでも足りなければ、窓を広げたトレース

`ACTOR_TORCH_MICRO` を step 全体（66〜77）にすると 2 GB 級のトレースになるが、
深い dip が入る確率は 3 倍になる。書き出しコスト（現状 21.9 秒）も 3 倍に
なるので、その step は捨てる前提で。

---

## 6. 実行方法（計測用の完全なコマンド）

スクリプトの既定は本番用であり、そのまま回すと計測にならない。

```bash
bash examples/sft_trainer/run_multitask_sft_qwen3.sh \
  ++algorithm.sft.data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_probe \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
  ++trainer.resume_mode=disable
```

stall watch は既定でオンで、何も設定は要らない。出力は
`/tmp/actor_stall/rank<N>_pid<P>.log`（Ray のコンソール重複除去で
3 rank のうち 1 本しか見えないため）。既に走っている run については
Ray 自身の per-worker ログに重複除去前の stdout がある:

```bash
grep -h "step-gpu" /tmp/ray/session_latest/logs/worker-*.out
```
