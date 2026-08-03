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

### 1.1 計測は 3 層あり、層ごとに別の表が出る

これを把握していないと「phase が粗い」と誤解する。**タグは 3 つの別プロセス/別機構に
散っていて、それぞれ独立した表を印字する。**

| 層 | 実装 | 出力先 | phase |
|---|---|---|---|
| **driver** | `opd_ray_trainer.py` の `_timer` | `(OPDTaskRunner pid=…)` | `step`(:491) / `gen`(:492) / `reward`(:567) / `teacher_forward`(:577)＋タスク別 / `update_actor`(:588) / `dump_rollout_generations`(:601) / `testing`(:615) / `save_checkpoint`(:622) |
| **worker** | `dp_actor.py` の `_actor_phase`（**rank 0 のみ**） | `(WorkerDict pid=…) update_policy stages` | `actor.fwd`(:756) / `actor.bwd`(:907) / `actor.task_metrics`(:917) / `actor.optim`(:998) |
| **rollout** | turn table（`ROLLOUT_TURN_TIMING=1`） | `[rollout-turn-timing]` | ターンごとに `preproc / gen / tchWait / decode / envstep` |

worker 層が別プロセスなのは必然で、`_timer` はドライバで動くため
`update_actor` はドライバ側からは**1 個の不透明なバケット**にしかならない。
`_actor_phase` は worker プロセスに**第 2 のサンプラーを立てて**その内側を割る。
NVML の読みはデバイス単位なので、どのプロセスから読んでも同じ GPU が見える ――
**phase タグだけが帰属を可能にしている**（`dp_actor.py:78`〜 のドキュメント文字列）。

`_PROFILE_STAGES = gpu_profiler.enabled() and rank == 0`（`:187`）。rank 0 限定なのは、
3 つ同時にサンプラーが回ると NVML のポーリングが 3 倍になり、
**同一の読みに対して 3 つの表が交錯して印字される**から。

### 1.2 タグの被覆範囲 —— 何が割れていて、何が割れていないか

| 処理 | タグ | 備考 |
|---|---|---|
| rollout（生成） | ✓ driver `gen` ＋ turn table | **内側は割れていない**（下記） |
| reward | ✓ driver `reward` | 純 CPU |
| teacher forward | ✓ **タスク別** `teacher_forward/<task>` | `90f8de9` で分割 |
| student forward | △ `actor.fwd` に同居 | lse/topk/gather も同じタグの中 |
| KL の計算 | **✗ 未タグ** | `_actor_phase` の外（`:756` は `_forward_micro_batch` 1 行だけを包む） |
| backward | ✓ `actor.bwd` | |
| optimizer ＋ scheduler | ✓ `actor.optim` | LR スケジュールは optim の中 |
| metrics | ✓ `actor.task_metrics` | |
| batch 組み立て（`adjust_batch` / `attach_task_loss_weights` / `_balance_batch`） | ✗ driver `(idle/other)` | 実測 0.6 s/step で無視可能（1.3 節） |
| checkpoint | ✓ `save_checkpoint` | |
| validation | ✓ `testing` | |
| `env.reset` | 対象外 | `_ENV_RESET_PREFETCH` で意図的にバックグラウンド |

**割れていない実質は 2 箇所だけで、うち 1 つは引き算で上界が取れる。**

**① KL の計算（未タグ、ただし ≲1 s）。** `_actor_phase("actor.fwd")` は
`self._forward_micro_batch(...)` **1 行だけ**を包んでいて、`topk_kl_per_token` と
重み付き集約（`:845`〜）はブロックの外にある。worker 表では `(idle/other)` に落ちて
gen 中のサンプルと区別できない。しかし
`update_actor` − `actor.* 合計` = **3.9 s** であり、そのうち Ray の dispatch が
約 3.0 s（`update_actor` − `timing_s/update_actor_worker`）なので、
**KL の実計算は 1 s 未満**。タグを足す価値がないことが確定した。

**② `gen` の内側は、どちらのサンプラーからも原理的に届かない。これが本レポートの
最大の弱点。** `gen` は step の 46%、`sm 65.2` で失点の 2/3 を占めるのに、
- vLLM の wake/sleep と weight sync
- prefill
- decode
- KV 逼迫による preemption

が一切分離されていない。`_timer` はドライバに、`_actor_phase` は actor モジュールにしか
置けず、**vLLM エンジンの内部は Ray worker の中の別ライブラリ**だからである。
2.3 節で「decode のテール」と書いた根拠は turn table の間接証拠
（`gen_util 65`、`full_batch 48–68% vs shrunk 71–74%`）だけで、直接の内訳ではない。

### 1.3 1 step の完全な内訳（step 91、変更前）

driver 表と worker 表が揃っている step。両サンプラーは同じ壁時計を見ているので
入れ子で読める。**TOTAL 509.3 s、sm 71.2。**

| phase | 層 | 時間 | share | sm% | memBW% | idle% | cpu% | memGB | pcieRX |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`gen`** | driver | **223.2 s** | 43.8% | **56.0** | 33.3 | 15.9 | 14 | 68.0 | 123 |
| `reward` | driver | 2.6 s | 0.5% | **0.0** | 0.0 | 100 | 100 | 16.8 | 0 |
| `teacher_forward`（glue） | driver | 0.3 s | 0.1% | 0.0 | 0.0 | 100 | 102 | 16.8 | 0 |
| ├ `/alfworld` | driver | 39.3 s | 7.7% | 89.8 | 32.6 | 6.0 | 9 | 55.3 | **7,755** |
| ├ `/search` | driver | 3.7 s | 0.7% | 77.7 | 24.9 | 7.7 | 11 | 70.9 | **7,643** |
| └ `/webshop` | driver | 23.5 s | 4.6% | 92.7 | 38.0 | 0.0 | 5 | 68.6 | 4,503 |
| **`update_actor`** | driver | **215.5 s** | 42.3% | **82.4** | 31.2 | 1.5 | 4 | 79.3 | 1,902 |
| ├ `actor.fwd`（student forward） | worker | **58.4 s** | 11.5% | 81.2 | 31.0 | 3.5 | 104 | 79.7 | 2,218 |
| ├ `actor.bwd` | worker | **149.7 s** | 29.4% | 83.6 | 31.6 | 0.2 | 104 | 79.1 | 1,697 |
| ├ `actor.task_metrics` | worker | 0.6 s | 0.1% | 94.0 | 40.0 | 0.0 | 103 | 80.0 | 32 |
| ├ `actor.optim`（＋scheduler） | worker | 2.9 s | 0.6% | 92.7 | 27.4 | 0.0 | 105 | 79.0 | **9,585** |
| └ KL ＋ loss 組立 ＋ dispatch | *未タグ* | **3.9 s** | 0.8% | — | — | — | — | — | — |
| `step`（境界） | driver | 0.6 s | 0.1% | 14.0 | 6.5 | 100 | 121 | 16.8 | 0 |
| **合計** | | **509.3 s** | 100% | **71.2** | | | | | |

driver タグの合計は 508.7 s で、**未計上は 0.6 s** ―― 1.2 節で「batch 組み立ては
無視可能」と書いた根拠がこれ。

この表からしか読めないこと:

- **`actor.bwd` 単独で step の 29.4%。** タグ済み phase では `gen` に次ぐ塊で、
  `bwd/fwd = 2.56` は gradient checkpointing の再計算込みとして妥当。
  SFT arm の「削るべきは backward」という結論は、**この phase については正しい**。
- **`actor.optim` の PCIe が突出。** 2.9 s（0.6%）しかないのに
  `pcieTX 14,924 / pcieRX 9,585` で、他のどの phase の 5〜7 倍。
  optimizer state が CPU 側にあり更新のたびに全量を往復させている。
  **時間比で見ると完全に見逃す**部分で、ここは今回手を入れていない。
- **`reward` は `sm 0.0 / memBW 0.0 / cpu 100`。** GPU が完全に遊ぶ純 CPU 区間で、
  4 節で overlap 候補としながら見送った根拠。
- **`gen` の `memBW/sm = 0.59` に対し `update_actor` は `0.38`。**
  2.1 節の判別ラベルどおり、gen の方がメモリ律速寄り＝ decode の性質が出ている。

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

### 2.8 律速はホスト RAM で、validation がその崖だった（run が落ちた）

**step 150 の最初の validation で OOM し、run が死んだ。** GPU ではなくホスト RAM。
本レポートで追いかけていた GPU 利用率とは別の軸に、より低い天井があった。

```
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file.
  (1) The process is killed by SIGKILL by OOM killer due to high memory usage.
  [repeated 169x / 217x / 57x / 33x across cluster]
→ ActorUnavailableError: keepalive watchdog timeout
```

数百の環境ワーカーが一斉に SIGKILL され、巻き添えで GPU の `WorkerDict` が落ち、
driver がタイムアウトした。タイミングは `test_gen_batch` →
`Initializing AlfredTWEnv...` の直後で、疑う余地がない。

**原因は validation が環境の母数を倍増させること。**

| | 環境数 | 生成時期 |
|---|---:|---|
| train envs | `15 × 3 tasks × env.rollout.n=8` = **360** | step 1、以後常駐 |
| val envs | `val_per_task_batch_size=126 × 3 tasks` = **378** | **最初の validation** |
| 合計 | **738** | |

val envs は `LazyEnvManager`（`env_manager.py:1052`）なので**最初の validation まで
作られない**。`test_freq=150` は、その生成を step 150 に置いていた。

数字が合う。tamago は **256 GB**（`docs/webshop_worker_memory.md`）。step 149 の
`cpu_memory_used_gb` は **195.1 GB（76%）**で、Ray の閾値 0.95（243 GB）まで残り 48 GB。
そこへ 378 ワーカーを新規に立てれば、1 ワーカー 130 MB でも足りない。

**定常成長（+150 MB/step）は壁ではなかった。** それだけなら step 300 で約 218 GB に
収まる。**壁は step 150 の段差の方**で、成長率だけを見ていると見えない。

`val_batch_size` と `val_per_task_batch_size` は**両方 intent lock に入っている**
（`expected_multitask_config.yaml:75,105`）＝評価プロトコルなので下げられない。
一方 **`test_freq` はロック外**なので、run 中の validation を切って
チェックポイントを後からオフライン評価するのが、科学的に中立な回避策になる:

```bash
bash examples/opd_trainer/run_multitask_qwen3.sh \
  env.search.search_url=http://100.86.45.30:8000/retrieve \
  trainer.test_freq=-1
```

評価する checkpoint も `val_batch_size=126` も `val_kwargs_by_task` も変わらない。
学習中に測るか後で測るかの違いだけで、ピークは消える。
**offpolicy arm も同じ崖に当たるので、同じ扱いが要る。**

根本対策は validation の前に train envs を `close()`（`env_manager.py:836` に実装あり）
して終了後に再構築することだが、episode schedule が resume 用にステートフル
（`3bce704`）なので、閉じて作り直すと進行位置がずれる危険がある。**走行中に入れる
変更ではない。**

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

`perf/throughput`（per-GPU、定義は両 run で同一）で見る:

| | steps | tok/s 平均 | 範囲 |
|---|---:|---:|---|
| baseline | 92–96（5 点） | **3,321** | 3,207 〜 3,491 |
| 変更後 | 100–148（**49 点**） | **3,594** | 3,332 〜 3,942 |

**+8.2%。** 分布で見るとより明確で、**49 step 中 38 step（78%）が baseline の最大値
3,491 を上回る**。下回った 11 step のうち最低の 3,332 は 13:00 のリトリーバ断の step。

wandb のログ行は `perf/throughput` を含まないことがあるが、`global_seqlen/mean` と
profiler の `TOTAL/step` から復元できる。baseline で検算すると
`3,543,577 ÷ (521.581 × 2) = 3,397.0` で `perf/throughput:3396.958` に一致するので、
**`global_seqlen/mean ÷ wall_s` がそのまま per-GPU throughput** になる。

**当初この数字を「+14.6〜20%」と述べたのは誤り**で、それは step 1–2 の warmup 2 点だけを
見た値だった。サンプルを 2 → 49 に増やして +8.2% に落ち着いた。**2 点の測定を
「速くなった量」として報告してはいけない**（5 節 ⑤ と同じ誤り）。

**同一指標・同一定義で確実に言えるのは以下だけ:**

| 指標 | before | after | 差 |
|---|---:|---:|---|
| `gen` の sm | 56.0〜59.3 | **65.3** | **+6〜9 pt** |
| `teacher_forward` | 67.4 s/step | **40.1 s/step**（100 step 累積） | **−27.3 s (−40%)** |
| `teacher/alfworld` の `pcieRX` | 7,591〜8,204 | **4,307** | 約半減 |
| `perf/max_memory_allocated_gb` | **121.2** | **90.9〜93.9** | **−27 GB（トークンは +43%）** |
| `update_actor` の sm | 82.4〜84.9 | 83.2 | 変化なし（手を入れていない） |
| `TOTAL/step` の sm | 71.3〜73.7 | 74.4 | +1〜3 pt |
| **throughput** | **3,321 tok/s** | **3,594 tok/s** | **+8.2%（49 step）** |

**ピークメモリ −27 GB は当初の見積りに入っていなかった効果。** teacher の常駐化は
逆に +5.1 GB/GPU 増やすので、差し引き 32 GB 分を稼いだのは 3.2 節の
response-only 化である。`topk_kl` は `(total_nnz, 151936)` に対して `logsumexp` と
`topk` を掛けており、これを応答行だけに絞った分が活性メモリにそのまま効く。
**「対象行が 1/9」は計算量の話だが、効果としてはメモリの方が大きかった。**

**`s/step` そのものは 544.5（125 step 累積）で baseline の 517.2 より大きいままだが、
それはトークン量が増えたからで、単位仕事あたりでは 8.2% 速い。** 速度の議論は
必ず throughput で行うこと。

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

**⑥ ホスト RAM の増加を「対処不要」と判断し、その直後に run が OOM で死んだ。**
最も高くついた誤り。`cpu_memory_used_gb` が 188 → 195 GB（+150 MB/step）と
増えているのを見て、こう書いた ――「baseline run は step 94 の時点で既に 217.8 GB に
達しており、旧 run が問題なく通過した水準」。3 点間違えている:

1. **ホストの総容量を確認せずに「通過した水準」と言った。** tamago は 256 GB なので
   217.8 GB は **85%**、Ray の閾値のすぐ手前である。「通過した」のではなく
   「ぎりぎりだった」が正しい。**比率を出さずに絶対値を比べた。**
2. **validation が第二の環境群を遅延生成することを勘定に入れなかった。**
   定常成長だけを見ていたが、実際の壁は成長ではなく **step 150 の段差**だった
   （2.8 節）。**傾きを見て段差を見なかった。**
3. **`docs/webshop_worker_memory.md` がこのホストのこの OOM を正面から記録しているのに、
   読まずに判断した。** リポジトリ内に答えがあった。

教訓は **「余裕がある」と言う前に分母を出すこと**、そして
**単調な指標を見たら傾きだけでなく将来の段差を探すこと**。

**⑦ wandb チャートの凡例にある GPU 2 を見て、3 枚目が遊んでいる可能性を指摘した。**
`run_multitask_qwen3.sh:11` に `HOST: 2 GPUs (tamago / 100.86.45.34)` と明記されており、
`trainer.n_gpus_per_node=2`。凡例の GPU 2 は同じ project に投げている
**SFT arm（A6000 × 3）の系列が残っていただけ**。
**チャートの凡例より run script を先に読むべきだった。**

**⑧ spec decode を「config だけで入る」と判断し、エンジンを確認しなかった。**
2.3(b) の decode テールに対する手として `engine_kwargs.vllm.speculative_config` を
入れたが、**step 1 に届かず `init_workers` で落ちた**（sdar arm で実測）:

```
WARNING ... Methods determine_num_available_blocks,device_config not implemented
            in <vllm.spec_decode.ngram_worker.NGramWorker object ...>
NotImplementedError: Method 'sleep' is not implemented.
  vllm_rollout_spmd.py:210  self.inference_engine.sleep(level=1)
```

`speculative_config` を渡すと vLLM は V0 の `SpecDecodeWorker` に差し替わり、
これは `sleep()` を実装していない。一方 `vllm_rollout_spmd` はエンジンを
`enable_sleep_mode=True` で作って直後に `sleep(level=1)` を呼ぶので、必ず落ちる。

**判断の誤りは 2 段ある。**
1. 5 節②で「`vllm==0.11.0` pin なので V0 は無い」と書いたのは *`VLLM_USE_V1` の
   export が no-op である*ことの根拠であって、**実行中のエンジンが V1 である証明では
   なかった**。トレースバックの `vllm/engine/llm_engine.py` /
   `vllm/executor/uniproc_executor.py` / `vllm.spec_decode.*` はすべて V0 の経路で、
   この環境は実際には V0 で動いている。前の誤りを根拠に次の結論を積んだ。
2. **verl 側が要求するエンジン機能を確認しなかった。** この rollout は wake/sleep を
   前提に組まれていて（`free_cache_engine=False`、`ROLLOUT_KEEP_VLLM_AWAKE`）、
   エンジンの worker を差し替える設定はすべてその前提と衝突しうる。
   「サンプリング分布を保存するか」だけを検証して、
   **「そのエンジン構成が verl の呼ぶメソッドを持つか」を検証しなかった。**

教訓は 2.4 節と同じ形をしている ―― **「理論上入れられる」と「この配置で動く」は別**。
2.4 では Ray の worker 配置、ここでは vLLM の worker 差し替えだった。
V1 エンジンなら spec decode は `v1/spec_decode` にあり sleep も実装されているので、
**`VLLM_USE_V1=1` を全 arm で検証する別実験**として残す。

---

## 6. 本番での検証（step 1〜149）

3 つの機構は本番で動作を確認済み。

| 指標 | 実測 | 判定 |
|---|---|---|
| `teacher_prefetch/hit_rate` | step 1 の 0.20 → **0.284〜0.463** | 定常的にヒット |
| `tchWait` | ターンあたり 0.0〜0.6 s、step 合計 3.4〜11.5 s、spill 0.9〜5.3% | チャンク 32 が適正 |
| `teacher_forward/alfworld` `pcieRX` | **4,307**（解除前 7,755〜8,204） | 3.1 節の通り |
| `actor/teacher_kl_loss` | 0.006〜0.008、`_weighted` 0.009〜0.015 | 同オーダー＝正規化でスケールが壊れていない |
| `task_loss/token_share` | alfworld 0.568〜0.782 / search 0.030〜0.063 | 補正対象の不均衡は実在し、100 step 経っても持続 |
| `actor/grad_norm` | step 1 の約 41 → **2.1〜3.6** | warmup 由来の一過性。以降 clip は効いていない |
| `perf/max_memory_allocated_gb` | 93.902 で完全固定（reserved 145.266） | GPU 側にリーク無し |
| `perf/cpu_memory_used_gb` | step 101 の 188.0 → step 149 の 195.1（**+150 MB/step**） | 下記参照 |

**ホスト RAM は横ばいではなく緩やかに増えている。** 当初 187〜190 の観測窓で「横ばい」と
書いたが、窓が短すぎた。このペースなら step 300 で約 217 GB。ただし **baseline run は
step 94 の時点で既に 217.8 GB に達しており**、そこは問題なく通過している水準なので
対処は不要と判断した。GPU 側は 49 step 完全固定。
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

15:11 にも短い断（`RemoteDisconnected` × 8）があり、全件 2 attempts / 9 s で復旧している。

### 6.2 run の終わり方

**step 150 の最初の validation でホスト RAM の OOM により停止**（2.8 節）。
学習そのものは step 149 まで完全に健全で、上表の指標はすべて安定していた。
最後の checkpoint は `global_step_125`（`save_freq=25`）。`resume_mode: auto` なので
同じコマンドで 125 から再開するが、**`trainer.test_freq=-1` を付けないと
step 150 で同じ崖に落ちる。**

この run から得られた性能上の結論（3.5 節の +8.2%、1.3 節の phase 内訳、
2.8 節のホスト RAM 制約）はすべて有効で、再取得の必要はない。

---

## 7. 再現手順

```bash
export GPU_PROFILER=1
export ROLLOUT_TURN_TIMING=1            # turn table。gen の中と外を分離するのはこれだけ
export GPU_PROFILER_ROLLUP_EVERY=1      # 既定 25。数 step しか回さないとき
# export GPU_PROFILER_TRACE=...         # 分散では使えない（2.7 節）
```

**worker 側の表（`update_policy stages`）が出ているかは別に確認すること。**
`_PROFILE_STAGES = gpu_profiler.enabled() and rank == 0`（`dp_actor.py:187`）で、
`gpu_profiler.enabled()` は **worker プロセスの** `GPU_PROFILER` を読む。
`constants_ppo.py` の Ray runtime env には `GPU_PROFILER` が入っていないので、
worker がドライバの環境を継承しない構成では**黙って出力されない**:

```bash
grep -c "update_policy stages" ~/logs/*.log    # 0 なら worker 層が動いていない
```

これが 0 だと `update_actor` は 1 バケットのままで、1.3 節の表は作れない。

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

**計測機構に足りないもの**（1.2 節、優先度順）:

| # | 追加 | 手段 | 効果 |
|---|---|---|---|
| 1 | **`gen` の内訳** | `disable_log_stats=False` で vLLM 自身の統計（prefill/decode トークン数、preemption 回数、running batch 推移）を出す | 失点の 65% を占める場所の内訳が推定から実測になる。**最優先** |
| 2 | `actor.fwd` を model / lse+topk+gather に分割 | `_actor_phase` を `_forward_micro_batch` の中に 2 個 | 3.2 節の効果を直接測れる（今はメモリでしか語れない） |
| 3 | `actor.optim` の PCIe を追う | optimizer state の配置を確認 | 1.3 節の突出（他 phase の 5〜7 倍）の正体 |
| — | KL のタグ | — | **不要と確定**（≲1 s、1.2 節） |

いずれも再起動が必要。

**その他:**

- **validation の env teardown**（2.8 節）。`test_freq=-1` は回避策であって修正ではない。
  `close()` は実装済みだが episode schedule のステートと衝突する。
- **`GPU_PROFILER_TRACE` の多重 open**（2.7 節）。パスに rank を混ぜれば直る。未修正。
- **`cudagraph_capture_sizes`**（V1 のみ）が decode テールに効くか。上記 1 の後。
- **offline-KD arm の `experiment_name` 不一致**。
  `run_multitask_offpolicy_qwen3.sh:464` は
  `opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20` を渡すが、
  `expected_multitask_offpolicy_config.yaml` は `sdar_multitask_offline_qwen3_1.7b` を pin
  している（`nogen` スクリプトの方は一致）。**intent lock 側の意図を確認して揃える。**
- **次 run の 3 項目**（4 節）を 3 arm 揃えて入れる。到達点は 82〜85%、〜490 s/it。

---

## 9. 実装済み・未計測の 2 機構（＋起動しなかった 1 つ）

3.5 節・5 節⑤の教訓により、**効果量はここに書かない**（まだ測っていない）。
書くのは「何を変えたか」「精度クラス」「次 run で何を見るか」だけ。

| 機構 | 変更箇所 | 精度クラス |
|---|---|---|
| teacher prefetch のターン単位化 | `rollout_loop.py`（記録時に `_queue_row_for_prefetch`） | 既存 prefetch と同一（期待値同一・micro-batch 構成のみ） |
| teacher の ZeRO-2 化 | `ref.fsdp_config.sharding_strategy=shard_grad_op`（run script; ノブは actor と共通の `get_sharding_strategy`） | **ビット同一**（配置と通信タイミングのみ） |
| ~~ngram speculative decoding~~ | **撤回。このスタックでは起動しない**（下記 3） | — |

根拠と設計:

1. **ターン単位化。** 行（=1 ターン）は `total_batch_list[i].append` された瞬間に不変で、
   teacher は run 全体で凍結。従来は軌跡完了時にまとめて queue していたため、バッチの
   大半を占める alfworld の行がエピソード終端（〜50 ターン）まで pool に入らず、
   hit_rate が 0.284〜0.463 で頭打ちだった（6 節）。記録時 queue なら glue 窓はターン 1
   から捌ける。**最終ターン近傍の行は原理的に残る**ので hit_rate 1.0 にはならない。
2. **ZeRO-2 teacher。** CPUOffload 解除後も `teacher_forward` の `pcieRX` は 4,307 で、
   これは FULL_SHARD が micro-batch ごとに繰り返す all-gather（3.1 節の表の残り半分）。
   SHARD_GRAD_OP は forward 後に reshard せず、teacher に backward は無いので、
   フェーズ先頭の 1 回で gather したまま走る。gather 済み実体の分だけメモリは増える
   （最大 3 × 3.4 GB/GPU、`max_memory_allocated 93.902` に対し許容）。
   **増えるのは GPU 側であって、2.8 節で run を殺したホスト RAM ではない。**
   2 つの天井は別軸なので混同しないこと ―― この機構は step 150 の崖には効かないし、
   悪化もさせない。見るのは `perf/max_memory_allocated_gb` の方。
3. **spec decode は撤回した。設定ミスではなく、この verl + vLLM では動かない。**
   `engine_kwargs.vllm.speculative_config` を渡すと vLLM は V0 の
   `spec_decode.SpecDecodeWorker`（内側に `NGramWorker`）に差し替わる。この
   ラッパーは `sleep()` を実装していない一方、`vllm_rollout_spmd` はエンジンを
   `enable_sleep_mode=True` で作り、直後に `sleep(level=1)` を呼ぶ（`:210`）ので、
   **step 1 に入る前に `init_workers` で
   `NotImplementedError: Method 'sleep' is not implemented` で落ちる**（sdar arm で実測）。
   `free_cache_engine=False` と `ROLLOUT_KEEP_VLLM_AWAKE` が依存する wake/sleep
   サイクル全体がこのメソッドを要求するので、引数の不足ではなく構造的な非互換である。
   狙い自体（2.3(b) の帯域律速な decode テールを、1 回の重みストリームあたりの
   トークン数を増やして償却する）は有効で、棄却サンプリングは分布を厳密に保存し、
   この arm は teacher KL を学習側 forward で計算するため vLLM の数値が損失に入る
   経路も無い。**必要なのは V1 エンジン**（spec decode が `v1/spec_decode` にあり、
   sleep も実装されている）。ただし `VLLM_USE_V1=1` は全フェーズのエンジンを
   変えるので、**全 arm 同時の別実験**であって、ここで倒すノブではない。
   → 5 節⑧に記録。

次 run で見るもの:

- `teacher_prefetch/hit_rate` — 0.28〜0.46 からどこまで上がるか。**ただしチャンク
  サイズを上げないと上がらない。** 発行は 1 ターン 1 チャンクのままなので、
  prefetch できる行数は `CHUNK × ターン数 ≤ CHUNK × 50` で頭打ちになる。
  1 step は 6,880〜7,200 行（0 節）だから:

  | `CHUNK` | 上限行数 | hit_rate の上限 |
  |---:|---:|---:|
  | 32 | 1,600 | **約 0.23** |
  | 128（既定） | 6,400 | 約 0.91 |
  | 160 | 8,000 | 1.0（行数側が先に尽きる） |

  **32 は既に達成している 0.28〜0.46 を下回る**ので、ターン単位化と組み合わせるなら
  戻してはいけない。ターン単位化以前は queue が枯れていた（alfworld の行が
  エピソード終端まで入らない）ため小さいチャンクで足りていたが、いまは
  **チャンクが唯一の律速**である。128 から始めて、下げるのは `tchWait` が
  step 時間を実際に悪化させたときだけ。3.3 節の通り `tchWait` 自体は無駄では
  ない（trainer がやらずに済んだ分）ので、判断は `tchWait` 単独ではなく
  `teacher_forward` の s/step ＋ step 全体の壁時計で行う（3.3 節の表の取り直し）。
- `teacher_forward/*` の `pcieRX` — 4,307 → 集団通信消滅でどこまで落ちるか。
  `perf/max_memory_allocated_gb` — 93.902 からの増分が見積り内か。
- 効果の主張は `perf/throughput`（tok/s、warmup 除外 30 step 以上）でのみ行う（3.5 節）。
  `gen` は今回どちらの機構の対象でもないので、動かないのが正常。
- **3 arm 同時適用**（4 節の原則）。ターン単位化と ZeRO-2 teacher は opd-grpo にも
  同じ変更を入れる。
