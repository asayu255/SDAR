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

一度この節は「先頭の `aten::nonzero` 65.8 ms が正体」と書いたが、**それは
summary ツールの帰属バグによる誤読だった**（4.7 節）。訂正後の分解は
以下の通りで、実体は 2 つの母集団に分かれる。

#### (A) 一様な ambient —— 毎 micro-batch に 35〜90 ms、合計 ~1%

トレースの per-micro idle 98〜158 ms から、誤帰属分（前 iteration の
optimizer step の尻尾 ~66 ms、デバイスは busy）を引いた残り。NVML の
deficit 1.13% ≒ 350 秒 step で 4 秒 ≒ 66〜77 micro で 50〜60 ms/micro と
一致する。**バーストではなく、全 micro-batch に薄く塗られている。**

機構の候補（複合と考えられる）:

1. **eager 起動オーバーヘッドの累積。** 1 micro-batch に ~7,840 カーネル。
   残余 idle 35〜90 ms ÷ 7,840 = **カーネル間 4〜11 µs** で、Python の
   dispatch + autograd + gradient checkpointing の再計算境界のコストと
   ちょうど同じ桁。個々のギャップは NVML にも watch にも見えないサイズ。
2. **同期点直後のパイプライン再充填。** `unpad_input` の `nonzero`
   （forward 先頭）と `response_row_selection` の `nonzero`（lm_head 前）は
   出力サイズがデータ依存なので device→host 同期を強制する。同期自体は
   前段の仕事と重なって無害だが、**同期が明けた瞬間はキューが空**で、
   ホストが再充填するまでの数 ms が毎回落ちる。1 micro に 2 箇所。
3. **`CUDA_DEVICE_MAX_CONNECTIONS=1`**（verl の ppo runtime env 既定、
   Megatron の TP 順序制御用）。ハードウェアキューを 1 本に制限するので、
   FSDP の `forward_prefetch`（次層 all-gather と当該層 compute の重なり）を
   阻害しうる。ただしこれが削るのは主に**重なり = NCCL 露出時間**（busy に
   見える側）で、idle への寄与は小さい。要検証。

#### (B) まれな深いイベント —— 0.3〜1 秒、~30 分に 1 回、micro-batch の中

トレースの 111 秒窓には期待値 0.06 件で入らなかった（2.5 節）。機構候補は
3 つあり、**それぞれがプロセス既存のカウンタを動かす**ので、per-step の
カウンタ差分で判別できる（5 節の計器を実装済み）:

1. **allocator のセグメント成長（`cudaMalloc`）。** `cudaMalloc` は
   device-synchronizing で、数百 MB〜GB 級なら 0.1〜1 秒止まる。実測と
   噛み合う点が多い: `memory_reserved` は step を跨いで実際に成長し続けて
   いる（50.9→55.6→60.7→66.1 GiB）。成長を引き起こすのは**記録更新級の
   micro-batch**（実測で 36k トークン、中央値の 2.4 倍）で、それが 1 rank
   に落ちれば **solo の dip**、balance 後の 3 rank が同時にピーク更新すれば
   **3 枚同時の dip** になる —— 観測された両方の署名を 1 機構で説明できる
   唯一の候補。なお `num_alloc_retries` は動いていない（retry は
   empty_cache 後の再試行だけを数える別カウンタで、通常のセグメント成長は
   `num_device_alloc` に出る —— 以前この 2 つを混同して仮説を棄却しかけた）。
2. **torch.compile の再コンパイル。** `use_torch_compile=True`
   （log-prob/entropy 経路）。未見の shape で guard が落ちると再コンパイルが
   走り、その間ホストは Python/Inductor に張り付いてデバイスは空になる。
   rank ごとに shape 履歴が独立なので **solo になる**。判別特徴: 頻度が
   run 後半に向かって**減衰する**はず（キャッシュが埋まるので）。
3. **Python gen-2 GC。** 世代 2 の回収はホストを凍らせ、キューが枯れた
   時点からデバイスが空く。rank 独立のタイミングなので solo。頻度は
   アロケーション量依存で、~分オーダーに 1 回はありうる。

共通の増幅器として、**(2) の同期点**がある: ホスト側の停止（GC・再コンパイル・
スケジューラのヒッチ）は、キューに仕事が積んであれば無害だが、`nonzero` の
同期直後はキューが空なので、**ホストの停止がそのままデバイスの停止になる**。

なお深いイベントと (A) は独立ではない可能性がある: セグメント成長を起こす
記録更新級 micro-batch は、mini-batch 単位のトークン偏り（9.13 節の 16.5% の
原因）が作っている。`BALANCE_MINIBATCH=1` は per-micro の最大トークンを
36k→18k に下げるので、**副作用として (B)-1 の発生源も減らすはず**である。

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

### 4.6 「65.8 ms の nonzero が正体」と書いた（帰属バグ）

`actor_trace_summary.py` は busy/idle を「**その窓の中で launch された**
カーネル」だけから計算していた。micro-batch の窓はホスト時刻で開くので、
**前 iteration の optimizer step（窓の外で launch、固定サイズ ~80 ms）の
カーネルが窓の先頭で実行されている時間**が「idle」に化け、その間ホストが
最初の同期点（`nonzero`）で待っているため「host: aten::nonzero」と表示された。

誤読のサインは全部出ていた: watch（stream event 基準）は同じ時間帯を
`gap/mini 81 ms ≈ optim 82 ms、unaccounted ≈ 0` と正しく分類していたし、
65.8 ms が rank 間・micro 間で**値まで同一**なのは optimizer が固定サイズ
だから。2 つの計器が食い違ったら、先に食い違いの理由を潰すべきだった。

ツールは修正済み（occupancy は全カーネルの union から計算し、carry 列で
optimizer の尻尾を明示する）。**手元の既存トレース 3 本に再実行すれば
GPU 時間ゼロで検証できる。**

### 4.7 計測器が 3 回スパイクを作った

* Nsight のキャプチャ: step 1 で 30 秒のノード全停止
* torch profiler の書き出し: micro 60 の前で 21.9 秒、3 rank 同時
* `GPU_PROFILER_SYNC_PHASES`: phase 境界ごとに device synchronize

**observer effect はこの arm では仮説ではなく、実測で最大のイベントである。**

---

## 5. スパイク除去のために、次にやること

### 5.1 修正した summary を既存トレースに再実行（GPU 不要、まずこれ）

```bash
git pull
python3 scripts/actor_trace_summary.py /tmp/actor_trace
```

carry 列（optimizer の尻尾）が ~66 ms/micro で出て、idle 列が実質値
（~35〜90 ms/micro）に下がるはず。**gap 表から 65.8 ms の nonzero 行が消え、
代わりに本物の最大ギャップ**（機構 (A)-1/2 のどれか）が名前付きで出る。

### 5.2 深いイベントはカウンタ差分で判別（実装済み、常時オン）

`per_rank_stall_counter_metrics`（`verl/utils/metric/stall_counters.py`）が
毎 step、rank ごとに 3 つの差分を wandb に出す:

| metric | 動いたら |
| --- | --- |
| `stall/cuda_mallocs/rank{N}` | その step でセグメント成長（cudaMalloc）があった |
| `stall/gc_gen2/rank{N}` | gen-2 GC が走った |
| `stall/dynamo_graphs/rank{N}` | torch.compile が新しいグラフを作った |

**dip の出た step でどれか 1 つだけが動いていれば、それが答え。どれも
動いていなければ 3 仮説とも棄却**で、候補はホストヒッチ×同期点に絞られる。
持続時間と phase は 0.2 秒 NVML サンプラー併走で取る:

```bash
GPU_PROFILER=1 GPU_PROFILER_INTERVAL=0.2 GPU_PROFILER_TRACE=/tmp/trace.csv bash ...
python3 scripts/gpu_stall_scan.py /tmp/trace.*.csv
```

### 5.3 ambient 側の打ち手（判明後）

* (A)-2 の `nonzero` 同期は除去可能性がある: `response_row_selection` は
  `cu_seqlens` が手元にあるので `nonzero` 無しで構成できる余地があり、
  `unpad_input` も verl 自前実装（`torch_functional.py:579`）側なら差し替え
  可能。ただし**効果は同期直後の再充填バブル数 ms × 2/micro** であって
  65.8 ms ではない（4.6 節）。
* (A)-1 は `use_fused_kernels`（現在 False）でカーネル数自体を減らすのが
  正攻法。torch.compile のモデル本体適用と CUDA Graphs は varlen と
  可変 shape のため現実的でない。
* `CUDA_DEVICE_MAX_CONNECTIONS=1` は FSDP 構成では外す実験の価値がある
  （Megatron 用の設定が runtime env 既定で全 arm に付いている）。

## 6. 実行方法（計測用の完全なコマンド）

スクリプトの既定は本番用であり、そのまま回すと計測にならない。
GPU 側で `git pull` してから:

```bash
cd ~/SDAR
git fetch origin claude/gpu-utilization-optimization-j1piv8
git checkout claude/gpu-utilization-optimization-j1piv8
git pull origin claude/gpu-utilization-optimization-j1piv8

BALANCE_MINIBATCH=1 \
bash examples/sft_trainer/run_multitask_sft_qwen3.sh \
  ++algorithm.sft.data_dir=$HOME/data/verl-agent/sdar_multitask/teacher_traj_probe \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
  ++trainer.resume_mode=disable
```

`BALANCE_MINIBATCH=1` だけが明示指定を要る（既定オフ）。これが MFU
0.315 → 0.339 を出した変更で、これを落とすと 7 節の数字と比較できない。
`ACTOR_PASS_CU_SEQLENS=1` と `ACTOR_GC_FREEZE=1` は既定オンなので何も要らない。

**起動直後、学習が始まる前に決め手の数字が出る**（8.4 節）:

```bash
grep "host-gc" /tmp/ray/session_latest/logs/worker-*.out
# [host-gc] rank 0: froze XXXXXXX objects (N collected, manual=False) in T s
```

`froze` が 500 万オーダーなら gen-2 GC が原因で確定（かつこの run で既に修正済み）、
10 万オーダーなら gen-2 GC は原因ではあり得ないので allocator へ移る。

A/B を取るなら 2 本目を `ACTOR_GC_FREEZE=0` で回して `stall/gc_gen2` と
solo excursion の増減を比べる。

stall watch は既定でオンで、何も設定は要らない。出力は
`/tmp/actor_stall/rank<N>_pid<P>.log`（Ray のコンソール重複除去で
3 rank のうち 1 本しか見えないため）。既に走っている run については
Ray 自身の per-worker ログに重複除去前の stdout がある:

```bash
grep -h "step-gpu" /tmp/ray/session_latest/logs/worker-*.out
```

**`/tmp/actor_stall/rank*.log` を glob してはいけない。** ファイル名は
`rank<N>_pid<P>.log` なので run ごとに別ファイルが増え続け、glob は
このマシンで走った全 run を読む。しかも古い行は epoch 秒しか持たないので
一見して古いと分からない（実際、torch profiler がトレースを書いた
21.9 秒の計測器スパイクを 4 時間後に「今の run の観測」と読み違えた）。
今の run に自分のファイルを言わせる:

```bash
grep -h "stall-watch" /tmp/ray/session_latest/logs/worker-*.out
# [stall-watch] rank 0 logging to /tmp/actor_stall/rank0_pid601985.log
```


---

## 7. 結果（run 5asi02yu）

`BALANCE_MINIBATCH=1` と cu_seqlens の受け渡しを入れた 3 step:

```
perf/mfu/actor                    0.339  0.338  0.338     (対 0.314-0.316)
global_seqlen/minibatch_wait_frac 0.001  0.000  0.000     (対 0.125)
minibatch_spread_mean             17.2   13.3   11.1 tok  (対 ~4,200)
gpu-profiler update_actor         sm 98.8% → 99.2%, idle 0.0%, maxGap 0.0
```

**MFU +7.6%。** ただし予測は +16.5% だった —— NCCL の待ち時間は全部が
そのまま wall-clock の節約に変換されるわけではない。理由は測っていないので
書かない。**2 つの変更が同じ run に入っているので、内訳も分離できていない**
（分けるなら `BALANCE_MINIBATCH=0` か `ACTOR_PASS_CU_SEQLENS=0` の片方だけ
落とした run が要る）。

学習側の指標も見ておく: `sft_loss` 1.086/1.123/1.086（対 1.050/1.085/1.066）、
`grad_norm` 44.2/44.6/37.4（対 50.7/51.6/43.4）。**grad_norm が下がっている
のは層化の効果として筋が通る** —— 各 mini-batch が長さの偏った塊ではなく
全長域のサンプルになるので、勾配の分散が下がる。損失が僅かに高いのは
どの行が同じ mini-batch に入るかが変わったためで、同じ軌跡ではない
（ビット同一でないことは最初から分かっている）。

### カウンタが仕事をした

step 2 で `stall/cuda_mallocs/rank1: 62`、同じ step で
`memory_reserved_gb/rank1` が 53.4 → 66.3 GiB。**rank 1 だけがセグメントを
62 回増やしている。** `cudaMalloc` は device-synchronizing なので、これは
まさに 3.2 節 (B)-1 が予測した「1 rank だけが記録更新級の micro-batch を
引いてプールを伸ばす」形である。深い dip がこの step に出ているかは
wandb の GPU サンプルと突き合わせて確認する。

step 3 では `rank0: 4`、他は 0。`gc_gen2` は全 rank で 1-2、
`dynamo_graphs` は全 step 0 —— **torch.compile の再コンパイルは
起きていない**ので、3.2 節 (B)-2 はこの run では棄却される。

---

## 8. スパイクの原因 —— 現時点の答え

質問は「結局スパイクの原因は何なのか」である。**まだ確定していない。**
ただし残った候補は 2 つで、**どちらかを決める数字は 1 つだけ**、しかも
その数字は次の run の起動時に印字される。以下、確定していることと
していないことを分けて書く。

### 8.1 スパイクの形（run 5asi02yu の実測、確定）

| 項目 | 値 |
| --- | --- |
| 件数 | 14 件（solo）/ 25.8 分 |
| 1 件の長さ | 0.6〜0.8 秒（幅が狭い） |
| 合計損失 | 6.0 秒 = ノード時間の **0.13%** |
| 落ちるカード | 1 枚だけ sm 0〜2、他の 2 枚は 100/100 |
| 位置 | `actor.fwd` の中（構造的な境界ではない） |
| 電力/クロック | 210〜238 W（基準 270）／1905 MHz（基準 1605） |

最後の行は異常ではない。**電力が下がってクロックが上がるのは、
仕事のないカードがブーストしているだけ**である（負荷がないので
電力枠が余る）。原因ではなく結果を見ている。

他の 2 枚が 100 なのは「計算している」からではない。**NCCL の spin kernel は
NVML から busy に見える**ので、collective で待っているカードは 100 と読める。
つまり **solo excursion の当事者は 0 に落ちている 1 枚**であり、
残り 2 枚はその 1 枚を待っている。

ここから機構に対する制約が 4 つ出る:

1. **1 rank ずつ**起きる → プロセス横断の事象（driver・dataloader・
   ネットワーク・ホスト全体）ではない。rank ごとに独立なタイミングを持つ
   何かである。
2. **sm が 0 になる** → デバイスが遅いのではなく、**ホストが投入を止めている**。
   デバイス律速なら sm は下がらない。
3. **forward の途中**に落ちる → step 境界でも optimizer でもなく、
   kernel launch と kernel launch の間である。
4. **0.6〜0.8 秒に集中**していて裾が長くない → コストがほぼ固定の
   「特定の 1 操作」の署名。競合（contention）なら分布はもっと広がる。

### 8.2 原因ではないもの（すべて数字付きで排除済み）

3.1 節の 10 項目（step 境界 0.24〜0.71 s、optimizer と reduce は実 kernel、
micro-batch 間 `interior` 0.00 s、ディスク、ネットワーク、ホストメモリ
12.8% 使用・439 GiB 空き、`alloc_retries` 0、vLLM は **−3.5 GiB** で
むしろデバイスが綺麗になる、teacher pool はドライバのホスト RAM、
gap 時間の 94〜99% でホストは CUDA runtime 呼び出しの中にいない）に加えて、

* **torch.compile の再コンパイル —— 棄却。** `stall/dynamo_graphs` は
  全 rank・全 step で 0。3.2 節 (B)-2 は死んだ。

### 8.3 残った 2 つと、発生率による選別

`stall/*` カウンタは 3 step × 3 rank = **9 rank-step** 分ある:

| カウンタ | 動いた rank-step | 内訳 |
| --- | --- | --- |
| `gc_gen2` | **9 / 9** | 全 rank 全 step で 1〜2 回 |
| `cuda_mallocs` | **2 / 9** | rank1 step2 で 62、rank0 step3 で 4、他 0 |
| `dynamo_graphs` | 0 / 9 | —— |

スパイクは 25.8 分の窓に **14 件**、つまり **run のあいだ一定の率で
繰り返し起きている**（約 110 秒に 1 件）。カウンタの窓（step 1〜3）と
トレースの窓は完全には重なっていないので件数どうしを直接割るのは避けるが、
**「繰り返すかどうか」だけで選別がつく**:

* **gen-2 GC は率が合う。** 全 rank・全 step で必ず 1〜2 回。
  つまり step あたり rank あたり 1〜2 回のペースで**恒常的に**発火し続ける。
  しかも rank ごとに独立なアロケーション量で発火するので **solo になる**。
  制約 1〜3 を全部満たす。
* **allocator のセグメント成長は率が合わない。** 9 rank-step のうち動いたのは
  2 回だけ。しかも大きい方（rank1 の 62 回、`memory_reserved` 53.4 → 66.3 GiB）は
  **プールが定常サイズまで育つ step 2 の一回性のイベント**であり、
  定常状態に入った後も 110 秒ごとに繰り返す性質のものではない。
  一発の実在イベントとしては本物だが、**繰り返す 14 件の説明にはならない**。

**したがって主犯候補は入れ替わる。** 7 節では `cuda_mallocs: 62` を
「カウンタが仕事をした」と書いたが、あれは *一発の実在イベント*を捉えた
だけで、*繰り返す 14 件*の説明にはならない。率で見ると gen-2 GC が前に出る。

なお **solo であることは選別に使えない**（両方 solo になる）し、
**長さも選別に使えない**（62 × 約 10 ms ≈ 620 ms で cudaMalloc も 0.6〜0.8 s の
帯に入る）。効くのは率だけである。

### 8.4 決め手になる 1 つの数字

gen-2 の掃引コストは追跡オブジェクト数にほぼ比例する。この箱で実測:

```
      held    tracked  gen-2 sweep  us/object
         0       4811        0.3 ms      0.064
    100000     204811       26.4 ms      0.129
   1000000    2004811      239.1 ms      0.119
   2000000    4004811      468.3 ms      0.117
```

**約 0.12 µs / object、きれいに線形。** ——ただし**この合成ヒープの数字は
実機に当てはめてはいけなかった**。下の 8.7 で実測が出て 3.2 倍ずれていた。
合成ヒープは同型の小さい dict / list ばかりで、実機は torch の module /
tensor / weakref が並び、`tp_traverse` の重さもメモリ局所性も違う。
**必要なのは実ヒープでの実測**であり、それは `freeze_permanent_heap()` が
起動時に印字する。学習が始まる前に出るので GPU を 1 step も回さずに決着する。

### 8.5 打ち手（診断と修正が同じ 1 回の run になる）

`verl/utils/host_gc.py`:

* `ACTOR_GC_FREEZE=1`（**既定オン**）—— モデル・optimizer・FSDP ラップが
  でき上がった直後に `gc.collect()` → `gc.freeze()`。この時点で生きている
  オブジェクトは *run 全体を通じて生き続けるもの* なので、恒久世代に
  移しても回収可能なものを隠すことにはならない。以降の掃引はそれらを
  一切走査しない。以後のアロケーションは通常どおり回収される。
* `ACTOR_GC_MANUAL=1`（既定オフ）—— 加えて `gc.disable()` し、
  step 境界（`before-step` = 実測 0.24〜0.71 秒、どのみちデバイスが空いている）で
  明示的に 1 回 `gc.collect()` する。残りのコストを *消す* のではなく
  *すでに失われている時間へ移す*。自動回収を切るとサイクルの多い step で
  ヒープが境界まで伸びるので、既定オフにして測定を待つ。

`ACTOR_GC_FREEZE=0` で完全に素の挙動に戻るので、**この 2 つがそのまま A/B**
である。`stall/gc_gen2` は既に per-rank per-step で記録されているので:

| 観測 | 結論 |
| --- | --- |
| solo excursion が消える | **gen-2 GC で確定、かつ修正済み** |
| excursion は残る | gen-2 GC を棄却。allocator へ（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`） |

`stall/gc_gen2` は freeze 後も 0 にはならない —— 掃引は依然として走り、
走査対象が恒久世代の分だけ減るだけである。**見るべきは回数ではなく
excursion の有無。**

### 8.6 現時点の正直な要約

* スパイクの**形**は確定している（8.1）。
* **原因ではないもの**は 11 項目、すべて数字付きで確定している（8.2）。
* **原因**は未確定。候補は 2 つに絞れており、率では gen-2 GC が前に出る（8.3）。
* 決め手は追跡オブジェクト数ひとつで、**次の run の起動ログに出る**（8.4）。
* その run は同時に修正の A/B にもなっている（8.5）。

そして規模の話を忘れないために書いておく: このスパイクは
**ノード時間の 0.13%** である。残っている損失の内訳は
ambient 0.58% / discrete 0.32%（うち solo 0.13%）で、
**大きいのは ambient の方**であり、それは `actor.bwd` に住んでいる
（`actor.bwd` の損失 20.9 秒のうち excursion 由来は 0.4 秒だけ）。
スパイクを完全に消しても 100% には届かない。

### 8.7 実測（1 回目の起動、run 開始前）

```
[host-gc] rank 0: froze 1090474 objects (10 collected, manual=False) in 0.42 s
[host-gc] rank 1: froze 1090403 objects (10 collected, manual=False) in 0.43 s
[host-gc] rank 2: froze 1090446 objects (10 collected, manual=False) in 0.43 s
```

**約 109 万オブジェクト、3 rank で一致**（同じシャードを持つので当然）。
そしてこの行にはもう一つ、より重要な数字が入っている: **`in 0.42 s`**。
これは `gc.collect()`（引数なし = 世代 2 の full collection）を
**実ヒープに対して**走らせた実測値である。

| | オブジェクト数 | 1 掃引 | µs/object |
| --- | --- | --- | --- |
| 合成（8.4 の表） | 1.09 M 相当 | 0.13 s | 0.123 |
| **実機（この行）** | **1.09 M** | **0.42 s** | **0.388** |

**合成の見積もりは 3.2 倍甘かった。** 0.123 µs/object のまま当てはめれば
109 万 → 0.13 秒で「0.6〜0.8 秒には足りない、棄却」と読んでいた。
実測は 0.42 秒で、**帯のすぐ下**である。

しかもこの 0.42 秒は**起動直後**、activation も autograd グラフも
バッチの Python オブジェクトもまだ無い状態のヒープに対する値である。
学習中の追跡集合はこれより**大きい**。

結論: **gen-2 GC は棄却されない。0.6〜0.8 秒を出すだけの規模がある。**
8.3 の発生率（9/9 rank-step）と合わせて、候補 2 つのうち gen-2 GC が
規模でも率でも条件を満たし、allocator は率で落ちている。

確定ではない —— 「規模が足りる」は「それが原因である」ではない。
決着は同じ run が付ける: freeze は 109 万個を以後すべての掃引から外すので、
**solo excursion が消えれば確定、残れば棄却**である。

---

## 9. 除去策 —— どの機構をどの層が塞ぐか

8 節の 4 制約（rank ごと独立 / ホストが投入を止めている / forward の途中 /
コストほぼ固定）に合うのは「ホストを止める固定コストの操作」であり、
この調査で挙がったそのクラスの機構は gen-2 GC・allocator 成長・
再コンパイル（棄却済み）の 3 つ。除去は 1 つに賭けず、層で塞ぐ。
**すべて既定オンで、`BALANCE_MINIBATCH=1` だけ明示が要る。**

| 層 | 塞ぐもの | 状態 |
| --- | --- | --- |
| init 直後の freeze（`ACTOR_GC_FREEZE=1`） | 109 万個・実測 0.42 s の掃引本体。恒久オブジェクトを全掃引の対象から外す | 済み |
| step 0 完了後の re-freeze（`ACTOR_GC_REFREEZE_STEP=1`） | warm-up 中に生まれる恒久物 —— Dynamo の guard/キャッシュ、Adam が初回 step で遅延確保する状態、FSDP の遅延構築物。init freeze は実行前なので見えない。カウンタの初回読みは仕様上 0 を返すため `dynamo_graphs=0` は step 0 のコンパイルを否定しない —— だからここも凍らせる | 今回 |
| 毎 step 境界の collect（`ACTOR_GC_BOUNDARY_COLLECT=1`、自動 GC は残す） | 生存オブジェクトの pending を毎 step 排水し、CPython の自動 full collection（pending > 最古世代の 25%）が **forward の最中に**発火する条件を潰す。freeze 後の 1 掃引は ms 級で、before-step の実測 idle 0.24〜0.71 s の中で走るので無料 | 今回 |
| cu_seqlens 手渡し（`ACTOR_PASS_CU_SEQLENS=1`） | forward の同期点 ~80 箇所/micro。同期点が減れば launch queue が深く保たれ、残る短いホスト停止はデバイスに映らない | 済み |
| `BALANCE_MINIBATCH=1`（要明示） | 記録更新級 micro-batch（per-micro 最大 36k→18k トークン）。allocator 成長イベント（rank1 の 62 回 cudaMalloc の型）の発生源そのもの | 済み |
| `ACTOR_GC_MANUAL=1`（既定オフ） | 上記でも残る場合の最終手段: 自動回収を切り境界掃引だけにする。ヒープ成長リスクがあるので測ってから | 実装済み・待機 |

`ACTOR_GC_FREEZE=0` で GC の 3 層がまとめて素の挙動に戻る（A/B は 1 フラグ）。

### 判定と、その先

判定は **solo excursion の有無**で行う。`stall/gc_gen2` は 0 にならず、
むしろ**増え得る**: freeze で最古世代の母数が減るぶん 25% 則が早く発火する。
回数が増えて 1 回が 0.4 秒 → 数 ms になるのが成功であり、カウンタだけ見ると
失敗に見える。

この全層を通してなお 0.6〜0.8 秒の solo excursion が残るなら、gen-2 GC は
棄却で、候補は allocator に戻る。その本命
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` は vLLM の CuMemAllocator
が assert で拒否する（スクリプトヘッダ参照、pytorch/pytorch#147851）ので、
道は 2 つ: 起動時にプールを峰まで温める warm-up パスを足すか、この arm では
そもそも使われない rollout の構築をやめて expandable_segments を解禁するか
（`val_before_train=False`・`test_freq=-1` なので run 中 vLLM は一度も推論
しない —— 検証は別プロセス）。

なおこのパッケージが対象にしたのはスパイク（0.13%）である。より大きい
**ambient 0.58% は `actor.bwd` に住んでいて、手つかず**のまま残っている。
