# GPU プロファイリングと高速化の記録（pure OPD multitask arm）

自分用の作業記録。`docs/gpu_profiling_report.md`（multitask SFT arm）の姉妹編。
tamago の 2 GPU で Qwen3-1.7B の pure on-policy distillation を回すにあたり、
step 予算の内訳を計測して 3 つの機構を入れた過程と、その途中で分かったこと。

**SFT arm とはホストが違う。** SFT 側は RTX A6000 48 GB × 3、こちらは tamago の
2 GPU で、`perf/max_memory_allocated_gb` は 93.9 GB ある。**s/step も util も
arm 間で横並び比較してはいけない。** 比較すべきは task success rate だけで、
そちらはホストの影響を受けない。

計測に使った機構は SFT arm と共通（`verl/utils/gpu_profiler.py` の phase タグ、
`GPU_PROFILER=1` が無ければ完全に no-op）＋ OPD 固有の turn table。

---

## 0. 計測環境

| 項目 | 値 |
|---|---|
| GPU | tamago の 2 GPU（`trainer.n_gpus_per_node=2`）。**NVLink 無し、PCIe のみ** |
| モデル | Qwen/Qwen3-1.7B（student）＋ 単一タスク RL の teacher 3 体 |
| FSDP | ZeRO-2（`sharding_strategy=shard_grad_op`）、`no_sync_grad_accum=True`、`forward_prefetch=True` |
| バッチ | `ppo_mini_batch_size=60`、`ppo_micro_batch_size_per_gpu=5`、`log_prob_micro_batch_size_per_gpu=16` |
| 系列長 | prompt 最大 4096 / response 最大 512（合計 4608） |
| 損失 | per-task teacher-KL のみ（`topk_kl`, k=20）。`pg_loss_coef=0` / `entropy_coeff=0` / `use_kl_loss=False` を `main_opd` が強制注入 |
| 1 step の規模 | 6,880〜7,200 行（`adjust_batch` 後） |
| 実測 | **548.3 s/step**（100 step 累積平均）、`TOTAL (run) sm 74.4` |

---

## 1. SFT arm の計測と何が違ったか

SFT arm では `update_actor` **単独**を `actor.fwd` / `actor.bwd` / `actor.task_metrics` /
`actor.optim` に割り、全 phase で `sm_util` 97〜99.5% を得た。

OPD で同じ道具を使うと、全く違う絵が出る。理由は単純で、**OPD の step の半分は
`update_actor` ではなく `gen`（rollout）だから**である。SFT は固定データセットを
読むだけなので生成が無い。したがって:

- SFT arm のレポートの結論（「削るべきは backward の中の通信」）は、**OPD の
  `update_actor` フェーズについてはそのまま当てはまる**。実際 ZeRO-2 / no_sync /
  forward_prefetch はそのまま移植した（`fb86b0d`）。
- しかし **step 全体の律速は別の場所にある**。それが本レポートの主題。

計測に足したもの:

1. **trainer レベルの phase タグ**
   `gen` / `teacher_forward.<task>` / `update_actor` / `reward` を `_timer` 経由で自動タグ。
   `teacher_forward` はタスク別に分けてある（`90f8de9`）ので、どのタスクの teacher が
   高いのかが share% で直接読める。

2. **rollout の turn table**（`ROLLOUT_TURN_TIMING=1`、`rollout_loop.py:167`）
   1 ターンを `preproc / gen / tchWait / decode / envstep` に割り、per-GPU の `gen_util` を
   付ける。末尾に `cpu-glue(preproc+decode+envstep, GPU-idle)=X%` と
   `teacher-spill(GPU-busy)=X%` の 2 行（`:203`〜`:209`）。
   **`gen` の中身と `gen` の外側を分離できるのはこの表だけ。**

3. **`pcieRX` / `nvlink` 列**
   ホストに NVLink が無いので FSDP の集団通信も PCIe に出る。パラメータの
   CPU→GPU 再ストリーミングと集団通信が同じ列に混ざるが、**phase で分ければ判別できる**。

---

## 2. 分かったこと

### 2.1 `sm%` も `memBW%` も時間ベースであって、効率ではない

これは SFT arm のレポート 2.1 節の続きだが、**あちらの記述には補正が要る**。

NVML の `utilization.gpu`（= `sm_util`）は「1 つ以上のカーネルが乗っていた時間の割合」。
これは SFT レポートの通り。問題は `utilization.memory`（= `mem_bw_util`）の方で、
`gpu_profiler.py:20` は "HBM bandwidth busy" と書いているが、これは
**達成帯域 ÷ ピーク帯域ではない**。`utilization.gpu` と全く同じ定義で、
**サンプリング窓の中で HBM の read/write が in-flight だった時間の割合**である。

したがって「`memBW%` 53.7% は帯域を半分しか使えていない」とは読めない。
毎サイクル 1 バイトだけ読んでいても 100% になる。

正しい使い方は **単独で見ず、`sm%` との差を見る**こと:

| 関係 | 意味 | 打ち手 |
|---|---|---|
| `memBW% ≈ sm%` | メモリ律速のカーネルが支配的 | バッチを増やす / 融合 / 量子化 |
| `memBW% ≪ sm%` | 演算律速のカーネルが支配的 | 精度・カーネル選択・並列化 |

`update_actor` が後者になるのは**健全**で、無駄ではない。`5 × 4608 ≒ 23k tokens` の
micro-batch は立派な GEMM で、Tensor Core とキャッシュの中で仕事が完結し HBM を
あまり叩かない。ここが低いのを見て「帯域が余っているから余地がある」と読むのは誤り。

**`memBW%` は効率指標ではなく、そのフェーズがどちらの律速かを判別するラベル。**
本当のピーク比を知りたければ DCGM（`DCGM_FI_PROF_DRAM_ACTIVE`、`PIPE_TENSOR_ACTIVE`）か
Nsight Compute が要る。NVML では原理的に測れない。

### 2.2 失点の 2/3 は `gen`

100 step 累積:

| phase | 時間 | 比率 | sm% | 失点 | 寄与 |
|---|---:|---:|---:|---:|---:|
| `gen` | 25,151.4 s | 45.9% | 65.3 | **15.9 pt** | **65%** |
| `update_actor` | 25,001.1 s | 45.6% | 83.1 | 7.7 pt | 32% |
| `teacher_forward/alfworld` | 3,866.9 s | 7.1% | 89.4 | 0.8 pt | 3% |
| `teacher_forward/webshop` | 139.5 s | 0.3% | — | — | — |
| **合計** | | | | 24.4 pt | → 75.6 |

失点 = 時間比 ×（100 − sm%）。合計 24.4 pt を引くと 75.6 で、実測の
`TOTAL (run) sm 74.4` とほぼ一致する（差 1.2 pt は `reward` と step 境界）。

**`update_actor` を SFT arm 並みに詰めても、取れるのは全体の 1/3 弱。**
SFT arm の結論をそのまま持ってきて backward の通信だけを追うと、
最大の塊を見逃す。

### 2.3 CPU glue は潰し切った。残っているのは `generate_sequences` の内側

rollout 中に GPU が 0 になる区間は 2 種類ある。

**(a) ドライバの CPU 仕事**（decode、`envs.step`、次ターンの tokenize）
run script が測った通り **rollout の 18%** を占めていた。これは潰せた:

- `envs.step` ∥ old_log_prob prefetch（`rollout_loop.py:877`）
- teacher chunk ∥ decode + envstep（`:844` で生成直後に発行）

**(b) `generate_sequences` の内側**
turn table の `gen` 列そのものが `gen_util 65.3` にしかならない。これは
**ドライバからは触れない**。中身は主に **decode のテール**である。

`train_batch_size × env.rollout.n ÷ 2 GPU` 本の系列がターン開始時に走り、完了順に
減っていく。終盤に残り数本になると decode は純粋な帯域律速になり、以前
`disable_log_stats` 有効時に観測した **6〜7 ms/decode-step** がこれ。
ターンごとにこのテールが繰り返される。

**(a) と (b) を混同しないこと。** `cpu-glue` が下がっても `gen_util` は上がらない。
別の穴である。

### 2.4 Ray の colocated worker が GPU 呼び出し同士の重畳を構造的に禁じる

これが本レポートで最も重要な発見で、後述の誤診の原因でもある。

`init_workers` → `create_colocated_worker_cls` により、**rollout・actor・teacher は
GPU あたり 1 個の `WorkerDict` に同居する**。そして `max_concurrency` は設定されていない
＝ Ray actor は**同時に 1 呼び出ししか実行しない**。

帰結:

- `generate_sequences` の実行中に teacher forward を発行しても、**キューに並ぶだけ**で
  重ならない。生成が終わってから走る。
- したがって teacher overlap で埋められるのは **(a) の CPU glue の窓だけ**であって、
  alfworld の長い生成テールではない。
- 同じ理由で `update_actor` と `gen` も原理的に重ならない。

**「GPU が空いている」と「そこに仕事を入れられる」は別。** 前者は NVML が教えてくれるが、
後者はワーカーの配置が決める。

### 2.5 util を下げながら速くなる変更がある

teacher の CPUOffload を解除した結果:

| | 解除前 | 解除後 |
|---|---:|---:|
| `teacher_forward/alfworld` の `pcieRX` | 7,755〜8,204 | **4,307** |

PCIe 転送中も NVML は busy と数えるので、**この改善は `sm%` を下げる方向に働く**。

逆に、util を上げて遅くする変更も簡単に作れる:

| 変更 | `sm%` | step 時間 |
|---|---|---|
| `enforce_eager=True`（CUDA graph 無効） | 上がる | 遅くなる |
| `enable_prefix_caching=False` | 上がる | 大幅に遅くなる |
| teacher の CPUOffload を戻す | 上がる | 遅くなる |

**NVML util は目的関数として壊れている。** 見るべきは `548.3 s/it` の方。
util は「どこを見るか」を決める中間指標であって、最適化対象ではない。

### 2.6 100% は構造的に到達不能で、しかも狙ってはいけない

2.4 の帰結として、colocated 設計を維持する限り GPU 呼び出しは直列である。
残る手（chunked prefill / KV 予算引き上げ / micro-batch 拡大）を全部入れた場合の
見積り:

| | `sm%` | `s/it` |
|---|---:|---:|
| 現在 | 74.4 | 548 |
| ＋ chunked prefill、KV 予算 0.7 | 〜80 | 〜505 |
| ＋ micro-batch 5→10、DP imbalance 詰め | **82–85** | 〜490 |
| async / disaggregated | 90+ | 〜380 |

**最終行は選べない。** 生成と訓練を重ねるとは、訓練中のポリシーで生成していない
＝ off-policy 化するということで、**`pure-opd` arm の存在意義そのものを壊す**。
（実装面でも `AsyncActorRolloutRefWorker.generate_sequences` は
`NotImplementedError` を投げるので、現状使えない。）

**この arm に残っている idle の大部分は、実験の定義が要求している idle である。**
最適化で消せる類のものではない。

補足として、1.7B というサイズ自体も効いている。decode 1 ステップは
「3.4 GB を HBM から流して行列ベクトル積を 1 回」であり、バッチが小さいと
演算器は原理的に遊ぶ。7B や 70B なら同じコードで util は上がるが、
それは効率が良いのではなく 1 トークンあたりの仕事が重いだけ。
**74% は 1.7B の multi-turn agent RL としては悪くない。**

### 2.7 プロファイラ自身の欠陥 1 件（未修正）

`GPU_PROFILER_TRACE` を指定すると、**ドライバ（`OPDTaskRunner`）と各 `WorkerDict` が
同じパスを mode `"w"` で開く**（`gpu_profiler.py:389`）。互いに切り詰め合うので
CSV は解析不能になる。SFT arm は単一プロセスで測っていたので踏まなかった。

今回は per-step / cumulative の集計表で代替した。**分散構成で trace を使うなら
パスに rank を混ぜる必要がある。**

---

## 3. 実装した高速化

| # | 機構 | 効果（実測） | 精度への影響 | commit |
|---|---|---|---|---|
| 1 | teacher の CPUOffload 解除 | `teacher_forward` 67.4 → 40.1 s/step、`pcieRX` 8,204 → 4,307 | 中立（配置のみ） | `07b669d` |
| 2 | response-only lse / topk / gather | 対象行が約 1/9、**ピークメモリ 121.2 → 93.9 GB** | ビット同一 | `07b669d` |
| 3 | chunked teacher overlap | `gen` の sm 56–59 → 65.3、`hit_rate 0.284–0.463` | 完全同一 | `e949d42` |
| — | （関連）per-task loss 正規化 | 高速化ではない | **意図的に変更** | `9503dc8` `f74e1f6` |

数値の出どころと、比べてはいけないものについては 3.5 節。

### 3.0 変更前の実測（steps 92–96）

3 機構を入れる前の同一ホスト・同一 config の run。

| step | `gen` | `teacher` | `update_actor` | `step` | tokens | tok/s | memGB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 92 | 223.2 | 67.1 | 215.3 | 508.8 | 3,273,296 | 3,216.6 | 121.2 |
| 93 | 241.0 | 79.4 | 270.2 | 593.4 | 4,143,464 | 3,491.1 | 121.2 |
| 94 | 223.0 | 67.3 | 228.1 | 521.6 | 3,543,577 | 3,397.0 | 121.2 |
| 95 | 220.1 | 64.0 | 217.7 | 505.4 | 3,329,938 | 3,294.1 | 121.2 |
| 96 | 202.4 | 59.4 | 192.4 | 457.0 | 2,930,964 | 3,207.0 | 121.2 |
| **平均** | **222.1** | **67.4** | **224.7** | **517.2** | **3,444,248** | **3,321.2** | |

phase 別の sm:

| phase | sm% | `pcieRX` |
|---|---:|---:|
| `gen` | 56.0 / 56.6 / 59.3 / 57.2 | 96–123 |
| `update_actor` | 82.4 / 84.8 / 84.9 | 1,773–1,902 |
| `teacher_forward/alfworld` | 89.5 / 89.8 / 91.0 | **7,591 / 7,755 / 8,204** |
| `teacher_forward/search` | 76.9 / 77.7 / 86.3 | **7,643 / 8,558 / 8,943** |
| `teacher_forward/webshop` | 92.7 / 94.7 / 95.3 | 4,503 / 4,899 / 4,926 |
| `TOTAL/step` | 71.3 / 73.7 | |

**`teacher_forward` の `pcieRX` だけが桁違いに高い**（`gen` の 60〜90 倍）ことが、
3.1 節の変更の直接の根拠になった。webshop が alfworld / search の 6 割程度なのは
エピソード上限が 4 ターンと短く、スコアする行数が少ないから。

### 3.1 teacher の CPUOffload 解除

`fsdp_workers.py:372` は元々こうなっていた:

```python
cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
```

つまり **role が "actor" でなければ設定に関係なく強制的に CPU オフロード**。
teacher は role="ref" の worker group として作られるので、これに該当する。

FSDP は CPUOffload 下で**マイクロバッチごとに全ユニットのパラメータをホストから
取り直す**。teacher は 3 体 × 3.4 GB で、`teacher_forward` の間ずっと
7.6〜8.9 GB/s を PCIe に流していた（比較として `gen` は 2 前後）。

修正:

```python
cpu_offload = None if role == "actor" else (
    CPUOffload(offload_params=True) if fsdp_config.get("param_offload", False) else None)
```

**副次的な発見: `ref.fsdp_config.param_offload` は dead key だった。** 上の強制により
値が読まれておらず、ref パスには手動オフロードの呼び出しも無い。つまりこの
キーはこれまで何も制御していなかった。run script 側を `False` に変更し、
`ppo_trainer.yaml` のコメントを実態に合わせて書き直した。

常駐コストは 3 teachers × 3.4 GB bf16 ÷ 2 GPU = **5.1 GB / GPU**。
`max_memory_allocated_gb 93.902` に対して十分収まる。

### 3.2 response-only lse / topk / gather

`use_remove_padding=True` の forward は `unpad_input` で全トークンを 1 本の
`(total_nnz, hidden)` に詰める。top-k KL の実装は、この**全行**に対して
`logsumexp` と `topk` を掛けていた。

しかし損失に使うのは応答部分だけである。この構成は prompt 最大 4096 /
response 最大 512 なので、**約 8/9 が捨てられる計算**をしていたことになる。

`dp_actor.py:114` に選択を切り出した:

```python
def response_row_selection(indices, seqlen, response_length):
    seq_pos = indices % seqlen
    lo = seqlen - response_length - 1
    sel = torch.nonzero((seq_pos >= lo) & (seq_pos < seqlen - 1), as_tuple=True)[0]
    return sel, indices[sel], seq_pos[sel] - lo
```

`unpad_input` が返す `indices` は詰めた行の元位置（`batch*seqlen` の平坦添字）なので、
`indices % seqlen` がそのまま系列内位置になる。応答行だけを選んで
`logsumexp` / `topk` / `gather` を掛け、`pad_input` で書き戻す。

`-response_length-1 : -1` のずらしは次トークン予測のためで、選択条件の
`lo = seqlen - response_length - 1` と `< seqlen - 1` がそれに対応している。
**ビット同一**（選ばれる行と値が変わらない）。`tests/trainer/test_response_row_selection.py` で
`indices` の全パターンに対して照合している。

### 3.3 chunked teacher overlap

2.4 の制約下で teacher forward を隠す唯一の窓は、**ドライバが CPU を焼いている
区間**（decode / `envs.step` / 次ターンの tokenize、rollout の 18%）である。

やったこと: **ターンの途中で終了した trajectory を、その場でチャンクにまとめて
teacher に投げる。**

- `rollout_loop.py:664` `_launch_teacher_prefetch()` — 生成直後（`:844`）に
  1 ワーカーの `ThreadPoolExecutor` へ投げる
- `:682` `_join_teacher_prefetch() -> float` — 次の `generate_sequences` の直前に回収し、
  **待たされた秒数を返す**
- `:698` `take_prefetched_teacher()` — trainer が結果を引き取る
- `opd_ray_trainer.py:254` `_teacher_prefetch_chunk()` — タスク別にまとめ、
  `compute_ref_topk_log_prob` を呼ぶ
- `:324` `compute_teacher_log_probs(batch, prefetched, metrics)` — ヒットした行を埋め、
  **残りだけ**をタスク別呼び出しに回す

teacher は凍結されているので**スコアは投げた時点に依存しない**。ただし行が
post-rollout パスとは違う micro-batch に載るため、詰めた GEMM の末尾が動く。
**`no_sync_grad_accum` と同じクラスで、ビット同一ではない**（期待値は同一）。

**チューニング。** チャンクが大きすぎると、次の生成の直前に回収しきれず待つ。
`_join_teacher_prefetch` の戻り値を turn table の `tchWait` 列と
`teacher-spill(GPU-busy)=X%` 行に出したので、これを見て決められる:

| `ROLLOUT_PREFETCH_TEACHER_CHUNK` | `tchWait` 合計 |
|---|---|
| 128（既定） | 11.6 s |
| **32** | **2.0〜2.5 s** |

`tchWait` は**無駄ではない**（trainer がやらずに済んだ teacher forward）ので、
0 にする必要はない。生成を待たせない範囲で最大化するのが正しい。

`gen` 列は `tchwait` を引いてある（`rollout_loop.py:892`）ので、**チャンクの
出しすぎが「生成が遅い」に化けない**。

### 3.4 （関連）per-task loss 正規化

高速化ではないが同じ一連の作業なので記録する。素の token-mean は、
エピソード上限（50/15/4 ターン）が生むトークン数の差をそのまま損失の重みにする。
実測で **alfworld 0.568〜0.782 / search 0.030〜0.063**。誰も選んでいない重み付けで、
かつ SFT arm はこれを使っていない。

`verl/trainer/ppo/task_loss_weights.py` を両 arm 共通の実装として置き、
重みを `num_mini_batches / (num_tasks * T_task)` とした。
OPD と offline-KD で**同じモジュールを共有**しているので、arm 間の差が
損失の定義だけに閉じる。

### 3.5 前後比較と、その限界

**`s/step` を直接比べてはいけない。** 変更後の 100 step 累積は 548.3 s/step で、
3.0 節の baseline 平均 517.2 より**遅い**。しかしこれは悪化ではなく、
**トークン量が違う**（baseline 3.44 M tokens/step に対し変更後は 4.89〜5.20 M、**+43%**）。
同じ step でも仕事量が別物なので、`s/step` は比較の単位にならない。

フレームワーク自身の `perf/throughput`（定義は両 run で同一）で見る:

| | tok/s |
|---|---|
| baseline（step 92–96） | 3,207〜3,491（平均 **3,321**） |
| 変更後 step 1 | 3,535 / 3,555 |
| 変更後 step 2 | **3,999** |

**平均比 +20%、baseline 最良値比 +14.6%。ただし変更後は step 1–2 の 2 点しかなく、
どちらも warmup の影響下にある。** この数字を確定値として扱ってはいけない。

**同一指標・同一定義で確実に言えるのは以下だけ:**

| 指標 | before | after | 差 |
|---|---:|---:|---|
| `gen` の sm | 56.0〜59.3 | **65.3** | **+6〜9 pt** |
| `teacher_forward` | 67.4 s/step | **40.1 s/step**（100 step 累積） | **−27.3 s (−40%)** |
| `teacher/alfworld` の `pcieRX` | 7,591〜8,204 | **4,307** | 約半減 |
| `perf/max_memory_allocated_gb` | **121.2** | **90.9〜93.9** | **−27 GB（トークンは +43%）** |
| `update_actor` の sm | 82.4〜84.9 | 83.1 | 変化なし（手を入れていない） |
| `TOTAL/step` の sm | 71.3〜73.7 | 74.4 | +1〜3 pt |

**ピークメモリ −27 GB は当初の見積りに入っていなかった効果。** teacher の常駐化は
逆に +5.1 GB/GPU 増やすので、差し引き 32 GB 分を稼いだのは 3.2 節の
response-only 化である。`topk_kl` は `(total_nnz, 151936)` に対して `logsumexp` と
`topk` を掛けており、これを応答行だけに絞った分が活性メモリにそのまま効く。
**「対象行が 1/9」は計算量の話だが、効果としてはメモリの方が大きかった。**

**まだ足りていない計測: 変更後の step 90–100 における `perf/throughput`。**
同じ step 帯・同じデータ位相での比較はそこでしか取れない。現状の throughput 比較は
warmup 2 点に依存している。wandb から拾えば確定する。

---

## 4. 採用しなかった手法

| 手法 | 判定 | 理由 |
|---|---|---|
| `reward` の overlap | 見送り | 2.3〜3.1 s / step、`cpu% ≈ 103` で GPU は idle。300 step で約 12 分。走行中断の価値なし |
| chunked prefill 有効化 | **次 run** | `gen` −5〜10% 見込み。サンプリング分布は不変 |
| KV 予算 `gpu_memory_utilization` 0.6 → 0.7 | **次 run** | `gen` −3〜8%。OOM 判定は `max_memory_reserved 145.3 GB` から逆算 |
| `ppo_micro_batch_size_per_gpu` 5 → 10 | **次 run** | `update_actor` −2〜3%。`adjust_batch` の lcm が 160 → 320 になり padding が増える |
| async / disaggregated RL | **不可** | on-policy 定義に反する（2.6 節） |

**「次 run」の 3 つは 3 arm 全部に同時に入れること。** いずれも損失の計算式を
変えないが、片方の arm だけ有効にすると「実験条件が揃っていない」と言われる余地を作る。

---

## 5. 誤った判断の記録

**① teacher forward を alfworld の生成テールに隠せると診断した。**
最大の誤り。`create_colocated_worker_cls` が teacher と rollout を同じ Ray actor に
置き、`max_concurrency` 未設定で 1 呼び出しずつしか走らないことを見落としていた
（2.4 節）。**見積りを 65 s から 20〜24 s に修正した。**
教訓は **GPU の空きだけを見て「そこに仕事を入れられる」と結論しないこと** ―
入れられるかはワーカーの配置が決める。

**② `VLLM_USE_V1=1` の未指定を「本物の交絡要因」と述べた。**
`environment.yml` は `vllm==0.11.0` を pin しており、このバージョンでは V0 が
既に削除されている。つまり export は no-op で、交絡は起きない。
リスクが実在するのは 0.8.x のみ。**依存の pin を見ずに一般論で答えた。**

**③ `memBW%` を「達成帯域 ÷ ピーク」と読み、「走っている間も帯域は 1/3」と述べた。**
実際は時間ベースの指標で、ピーク比は NVML では測れない（2.1 節）。
SFT arm レポートの 2.1 節も同じ前提で書かれているので、両方読み直す必要がある。

**④ 不要な assert を入れて step 2 でクラッシュさせた。**
「バッチ行数は `ppo_mini_batch_size` で割り切れる」と仮定したが、`adjust_batch` は
`lcm(log_prob_micro × W, ppo_micro × W) = lcm(32, 10) = 160` に丸める。
mini-batch は 60 なので、`lcm(160, 60) = 480` ―― **割り切れるのは 3 step に 1 回程度**。
step 1（7,200 行）は偶然通り、step 2（6,880 行）で落ちた。

さらに悪いことに、**この assert はそもそも不要だった**。`self.gradient_accumulation` は
設定値の定数で、重みを掛ける側（`dp_actor.py:879`）と割る側
（`loss = policy_loss / self.gradient_accumulation`）の両方に同じ値が現れて相殺する。
短い最終 mini-batch はその行の重み付き損失の総和をそのまま寄与するだけで、
特別扱いは要らない。`num_mini_batches = math.ceil(...)` に直し、
`rows = 160·k` / mini-batch 60 のランダム 300 通りで再生検証した。

**「不変条件を測らずに assert した」のが原因。** 検証のつもりの assert が
クラッシュの原因になった。

**⑤ 3 機構の効果を代理指標だけで報告し、baseline を表に載せなかった。**
`pcieRX` / `hit_rate` / `tchWait` は機構が動いていることの証拠にはなるが、
**「どれだけ速くなったか」ではない**。指摘されて steps 92–96 の実測を掘り起こしたのが
3.0 節で、その時点で 2 つのことが分かった ―― ①トークン量が +43% 違うので
`s/step` の直接比較は誤り（3.5 節）、②ピークメモリ −27 GB という、
見積りに入っていなかった最大の効果を見落としていた。
**代理指標は「効いている」を示すが「効果量」は示さない。両方要る。**

**⑥ wandb チャートの凡例にある GPU 2 を見て、3 枚目が遊んでいる可能性を指摘した。**
`run_multitask_qwen3.sh:11` に `HOST: 2 GPUs (tamago / 100.86.45.34)` と明記されており、
`trainer.n_gpus_per_node=2`。凡例の GPU 2 は同じ project に投げている
**SFT arm（A6000 × 3）の系列が残っていただけ**。
**チャートの凡例より run script を先に読むべきだった。**

---

## 6. 本番での検証（step 1〜106）

3 つの機構は本番で動作を確認済み。

| 指標 | 実測 | 判定 |
|---|---|---|
| `teacher_prefetch/hit_rate` | step 1 の 0.20 → **0.284〜0.463** | 定常的にヒット |
| `tchWait` | ターンあたり 0.0〜0.6 s、step 合計 3.4〜11.5 s、spill 0.9〜5.3% | チャンク 32 が適正 |
| `teacher_forward/alfworld` `pcieRX` | **4,307**（解除前 7,755〜8,204） | 3.1 節の通り |
| `actor/teacher_kl_loss` | 0.006〜0.008、`_weighted` 0.009〜0.015 | 同オーダー＝正規化でスケールが壊れていない |
| `task_loss/token_share` | alfworld 0.568〜0.782 / search 0.030〜0.063 | 補正対象の不均衡は実在し、100 step 経っても持続 |
| `actor/grad_norm` | step 1 の約 41 → **2.1〜3.6** | warmup 由来の一過性。以降 clip は効いていない |
| `perf/max_memory_allocated_gb` | 93.902 で横ばい（reserved 145.266、host 187〜190 GB） | リーク無し |
| DP-IMBALANCE | 3.7〜9.6 pp（`TASK_BALANCE_INTERLEAVE` 導入前は 9〜18 pp） | 半減 |

### 6.1 retriever 断を retry ポリシーが完全に吸収した

13:00:08〜13:00:52 の約 44 秒、`100.86.45.30:8000` への接続が
`[Errno 113] No route to host` で落ち、数百リクエストがバックオフ再試行に入った。
**全件が復旧した**（`search recovered after 3-6 attempts (49s / 51s / 57s / 58s)`）。

これは `env.search.max_retries=null` ＋ `SEARCH_WAIT_FOR_SERVICE` の設計通りの挙動で、
重要なのは **エラー文字列が `<information>` ブロックに 1 度も入らなかった**こと。
諦めていれば、リトリーバのエラーテキストが検索結果として学習データに混入し、
メトリクスには何も出なかった。

代償は step 100 が 647.2 s（`turn 0 envstep = 69.62 s`、`cpu-glue 34.6%`）になった
1 step 分だけで、直後から 480〜580 s に戻った。**対処不要。**

---

## 7. 再現手順

```bash
export GPU_PROFILER=1
export ROLLOUT_TURN_TIMING=1            # turn table。gen の中と外を分離するのはこれだけ
export GPU_PROFILER_ROLLUP_EVERY=1      # 既定 25。数 step しか回さないとき
# export GPU_PROFILER_TRACE=...         # 分散では使えない（2.7 節）
```

turn table の読み方:

- `cpu-glue(...)` — ドライバが CPU を焼いていて GPU が 0 の割合。**prefetch で埋められる穴**
- `teacher-spill(GPU-busy)` — チャンクを回収しきれず待った割合。**無駄ではない**が
  生成を遅らせるので、`ROLLOUT_PREFETCH_TEACHER_CHUNK` を下げて詰める
- `gen_util` — `generate_sequences` の内側。**ドライバからは触れない**（2.3 節）

高速化機構の on/off（プロセス env var、config ではない）:

```bash
export ROLLOUT_PREFETCH_TEACHER=1           # 3.3 節。既定 off
export ROLLOUT_PREFETCH_TEACHER_CHUNK=32    # 既定 128。tchWait を見て決める
export ROLLOUT_KEEP_VLLM_AWAKE=1
export ENV_RESET_PREFETCH=1
export TASK_BALANCE_INTERLEAVE=1
# ROLLOUT_PREFETCH_LOGPROB は pure OPD では立てない。
# 薄いループに old_log_prob phase が無く、prefetch した値が消費されない。
```

**本実験で `GPU_PROFILER_SYNC_PHASES=1` は立てないこと。** phase 境界ごとに
`device.synchronize()` を入れるので、本来重なる処理を直列化して実際に遅くなる。
帰属としては読めても速度としては読めない。

---

## 8. 残課題

- **変更後 step 90–100 の `perf/throughput` を wandb から取る**（3.5 節）。
  現状の +14.6〜20% は warmup 2 点に依存しており、確定値ではない。
  baseline は 3,207〜3,491 tok/s。**これが最優先。**
- **`GPU_PROFILER_TRACE` の多重 open**（2.7 節）。パスに rank を混ぜれば直る。未修正。
- **decode テールの深掘り**。`disable_log_stats=False` で 6〜7 ms/decode-step の内訳を取り、
  `cudagraph_capture_sizes`（V1 のみ）が効くか見る。走行中は測れないので run 後。
- **offline-KD arm の `experiment_name` 不一致**。
  `run_multitask_offpolicy_qwen3.sh:464` は
  `opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20` を渡すが、
  `expected_multitask_offpolicy_config.yaml` は `sdar_multitask_offline_qwen3_1.7b` を pin
  している（`nogen` スクリプトの方は一致）。**intent lock 側の意図を確認して揃える。**
- **次 run の 3 項目**（4 節）を 3 arm 揃えて入れる。到達点は 82〜85%、〜490 s/it。
