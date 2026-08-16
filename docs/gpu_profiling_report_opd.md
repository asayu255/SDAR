# GPU プロファイリングと高速化の記録（pure OPD multitask arm）

自分用の作業記録。`docs/gpu_profiling_report.md`（multitask SFT arm）の姉妹編。
tamago の 2 GPU で Qwen3-1.7B の pure on-policy distillation を回すにあたり、
step 予算の内訳を計測して 5 つの機構を入れた過程と、その途中で分かったこと。

**結論から:** 単位仕事あたりの速度は **+13.9%**（`perf/throughput` 3,321 → 3,594 → 3,782 tok/s）。
**`s/step` は 517.2 → 548.3 → 496.3 と単調でないので、これを結果として引用してはいけない**
―― run ごとに step あたりのトークン量が最大 +43% 違う（3.5 節）。

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
| 実測 | 3 機構時点: **548.3 s/step**（100 step 累積）、`TOTAL sm 74.4` → 5 機構: **496.3 s/step**（150 step 累積）、`TOTAL sm 73.9`。**s/step はトークン量が違うので比較不可**（3.5 節） |

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
  **時間比で見ると完全に見逃す**部分で、ここは今回手を入れていない。
  当初これを「optimizer state が CPU 側にあり毎回往復している」と書いたが、
  **その説明は成立しない** —— run script は `optimizer_offload=False` で Adam は
  GPU 常駐である。より当たりそうなのは backward の reduce-scatter の尾が
  この phase の窓で drain していることで（`_actor_phase` の注記どおり、
  kernel launch は非同期なので phase 境界は 1 launch queue 分ずれる）、
  だとすればこれは `actor.optim` のコストではない。8 節の追跡項目としては残すが、
  2.9 s しかないので優先度は最低。
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

| 構成 | steps | n | tok/s 平均 | 対 baseline |
|---|---|---:|---:|---:|
| baseline（3.0 節） | 92–96 | 5 | **3,321** | — |
| ＋ 3.1〜3.3 の 3 機構 | 100–148 | **49** | **3,594** | **+8.2%** |
| ＋ 9 節の 2 機構（ターン単位化 / ZeRO-2 teacher） | 260–299 | **39** | **3,782** | **+13.9%** |

**累積 +13.9%。** 中間の +8.2% は分布で見るとより明確で、**49 step 中 38 step（78%）が
baseline の最大値 3,491 を上回る**。下回った 11 step のうち最低の 3,332 は
13:00 のリトリーバ断の step。最終段の +5.2%（3,594 → 3,782）は
9.1 節の n=1 制御 A/B が出した +5.2% と一致した（9 節の「限界」を参照）。

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
| **throughput（9 節の 2 機構まで含む）** | **3,321 tok/s** | **3,782 tok/s** | **+13.9%（39 step）** |

**ピークメモリ −27 GB は当初の見積りに入っていなかった効果。** teacher の常駐化は
逆に +5.1 GB/GPU 増やすので、差し引き 32 GB 分を稼いだのは 3.2 節の
response-only 化である。`topk_kl` は `(total_nnz, 151936)` に対して `logsumexp` と
`topk` を掛けており、これを応答行だけに絞った分が活性メモリにそのまま効く。
**「対象行が 1/9」は計算量の話だが、効果としてはメモリの方が大きかった。**

**`s/step` そのものは 544.5（125 step 累積）で baseline の 517.2 より大きいままだが、
それはトークン量が増えたからで、単位仕事あたりでは 8.2% 速い。** 速度の議論は
必ず throughput で行うこと。

**最終構成の `s/step` は 496.3（150 step 累積）で baseline の 517.2 を下回るが、
これも「速くなった量」として引用してはいけない。** 3 つの run は
step あたりのトークン量が 3.44 M / 4.89〜5.20 M / 4.7 M 級とばらついており、
496.3 < 517.2 は改善と減量の合成である。**3 つの数字 517.2 → 548.3 → 496.3 が
単調でないこと自体が、`s/step` が指標になっていない証拠**として読むこと。
実際の短縮は throughput の 3,321 → 3,594 → 3,782（**+13.9%**）の方である。

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

## 6. 本番での検証（step 1〜149、および再開後 125〜300）

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

### 6.2 run の終わり方と、再開後の完走

**1 回目の run は step 150 の最初の validation でホスト RAM の OOM により停止**（2.8 節）。
学習そのものは step 149 まで完全に健全で、上表の指標はすべて安定していた。
最後の checkpoint は `global_step_125`（`save_freq=25`）。

**対処は `trainer.test_freq=-1` を足して 125 から再開する**（`resume_mode: auto`）。
このノブは intent lock に入っていないので、科学的な条件は動かない
（`val_batch_size` も `val_per_task_batch_size` もロック側にあり、そのまま）。
代償は **in-run validation が 1 度も走らないこと**で、評価は保存済み
checkpoint に対するオフライン実行に回す（8 節）。

**再開した run は `global_step_300` まで完走した。** 150 step 累積:

| 指標 | 値 |
|---|---|
| `TOTAL/step` | **496.3 s**（sm 73.9） |
| `gen` | 38,833.6 s = 52.2%（sm 66.7 / memBW 43.1） |
| `update_actor` | 34,746.1 s = 46.7%（sm 83.6） |
| `teacher_forward/alfworld` | **74.0 s = 0.1%**（ターン単位化で `gen` 側へ移動） |
| `teacher_prefetch/hit_rate` | **0.989〜0.994** で 40 step 安定 |
| `perf/max_memory_allocated_gb` | **103.060**（+9.2 GB、ZeRO-2 teacher の見積り通り） |
| teacher spill | 7.1〜9.9% |

`gen` の `memBW%` が約 35 → **43〜46** に上がっているのは、メモリ律速な teacher の
仕事がそのフェーズの内側へ入ったからで、劣化ではない。**フェーズ境界が動いたときは
フェーズ単体の数字を前後比較してはいけない**（9.2 節・5 節⑤と同じ形）。

この run から得られた性能上の結論（3.5 節の +13.9%、1.3 節の phase 内訳、
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
| 3 | `actor.optim` の PCIe を追う | `GPU_PROFILER_SYNC_PHASES=1` で境界を締めて再測 | 1.3 節の突出（他 phase の 5〜7 倍）の正体。`optimizer_offload=False` なので CPU 往復説は棄却済み、残る仮説は backward の collective 尾の染み出し |
| — | KL のタグ | — | **不要と確定**（≲1 s、1.2 節） |

いずれも再起動が必要。

**その他:**

- **validation の env teardown**（2.8 節）。`test_freq=-1` は回避策であって修正ではない。
  `close()` は実装済みだが episode schedule のステートと衝突する。
- **保存済み checkpoint のオフライン評価。** `test_freq=-1` で完走させた結果、
  この run には in-run validation の点が 1 つも無い。`global_step_{150,…,300}` に対して
  別プロセスで評価を回すこと。**3 arm の比較はこの評価が揃うまで成立しない。**
- **`GPU_PROFILER_TRACE` の多重 open**（2.7 節）。パスに rank を混ぜれば直る。未修正。
- **`cudagraph_capture_sizes`**（V1 のみ）が decode テールに効くか。上記 1 の後。
- **offline-KD arm の `experiment_name` 不一致**。
  `run_multitask_offpolicy_qwen3.sh:464` は
  `opd_offpolicy_multitask_qwen3_1.7b_coef1.0_topk_kl20` を渡すが、
  `expected_multitask_offpolicy_config.yaml` は `sdar_multitask_offline_qwen3_1.7b` を pin
  している（`nogen` スクリプトの方は一致）。**intent lock 側の意図を確認して揃える。**
- **次 run の 3 項目**（4 節）を 3 arm 揃えて入れる。到達点は 82〜85%、〜490 s/it。

---

## 9. 2 機構の実測（＋起動しなかった 1 つ）

実装時は未計測だったが、**step 126 を新旧コードで 1 回ずつ回して測れた**。
数値は 9.4 節。

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

### 9.1 実測（step 126、新旧コードで 1 回ずつ）

**3.5 節が `s/step` 比較を禁じた理由がこの比較には無い。** 同じ step 126 を同じ
checkpoint 125 から再開しているので、行数は完全一致（alfworld 3989 / search 290 /
webshop 953 ＋ padding 48 = 5,280）、トークン数の差は **+0.05%**、
`episode/success_rate` も 0.569 で同一。**同じ仕事を測っている**ので直接引き算できる。

| 指標 | 旧コード | 新コード | 差 |
|---|---:|---:|---:|
| `perf/total_num_tokens` | 4,672,843 | 4,675,061 | +0.05%（交絡なし） |
| **`timing_s/step`** | **610.7 s** | **580.9 s** | **−29.8 s（−4.9%）** |
| **`perf/throughput`** | **3,826 tok/s** | **4,024 tok/s** | **+5.2%** |
| `teacher_prefetch/hit_rate` | 0.531 | **0.991** | +0.46 |
| `teacher_prefetch/rows` | 2,806 | **5,230** / 5,280 | 残り 50 |
| **`timing_s/teacher_forward`** | **33.68 s** | **1.30 s** | **−32.4 s** |
| `tchWait` 合計（rollout 内） | 19.3 s | 28.0 s | +8.7 s |
| rollout 壁時計 | 277.7 s | 280.9 s | +3.2 s |
| `timing_s/gen` | 290.6 s | 291.7 s | +1.1 s（横ばい） |
| `timing_s/update_actor` | 282.1 s | 284.3 s | +2.2 s（横ばい） |
| `gen` の `pcieRX` / `pcieTX` | 768 / 853 | **208 / 194** | 約 1/4 |
| `gen` の `memGB` | 75.8 | 87.1 | +11.3（常駐 teacher） |
| `perf/max_memory_allocated_gb` | 89.3 | **97.5** | **+8.2 GB** |

**sm% は step 全体で +0.8 pt しか動いていない。** driver の phase 表を壁時計で
重み付けし直すと（2.2 節と同じ「失点 = 時間比 ×（100 − sm%）」）:

| phase | 旧 時間 / sm / 失点 | 新 時間 / sm / 失点 |
|---|---|---|
| `gen` | 290.3 s / 66.1 / **16.11 pt** | 291.7 s / 69.0 / **15.57 pt** |
| `update_actor` | 282.3 s / 84.3 / 7.26 pt | 284.1 s / 84.6 / 7.53 pt |
| **`teacher_forward` 合計** | **33.7 s / 87.4 / 0.70 pt** | **1.5 s / 9.6 / 0.23 pt** |
| ├ `/alfworld` | 32.8 s / 89.8 | 0.9 s / 16.0 |
| ├ `/search` | 0 s（残りなし） | 0 s（残りなし） |
| ├ `/webshop` | 0 s（残りなし） | 0 s（残りなし） |
| └ glue | 0.9 s / 0.0 | 0.6 s / 0.0 |
| `reward` ＋ `step` | 4.4 s / 0.0 | 3.5 s / 0.0 |
| **合計** | 610.7 s / **75.2** | 580.8 s / **76.1** |

`teacher_forward` は 1.3 節と同じくタスク別に割ってあるので、**3 タスク＋glue の
合計**で読む（`timing_s/teacher_forward` と一致: 33.7 / 1.5 に対し 33.676 / 1.300）。

worker 側も同様に横ばい: `TOTAL/step` 84.6 → 85.0、`actor.fwd` 83.9 → 83.2、
`actor.bwd` 84.9 → 85.7 ―― 手を入れていないので当然である。

**この表は「利得はすべて alfworld から来た」とも言っている。** 両 step とも
`/search` と `/webshop` は行が印字されていない ＝ **post-rollout に残った仕事が
ゼロ**で、旧コードの時点で既に全量 prefetch されていた。エピソード上限が 4 / 15
ターンなので軌跡が早く終わり、**軌跡完了時キューでも間に合っていた**からである。
行数で確かめられる:

| | search＋webshop | alfworld | 合計 |
|---|---:|---:|---:|
| 旧 prefetch | 1,243 / 1,243（100%） | 1,563 / 3,989（39.2%） | 2,806 |
| 新 prefetch | 1,243 / 1,243（100%） | **3,987 / 3,989（99.9%）** | 5,230 |

移ったのは **alfworld の 2,424 行だけ**。残っていた 2,426 行を 32.8 s で処理していた
から 13.5 ms/行 で、2,424 行 ≒ 32.7 s ―― `teacher_forward` の −32.4 s とぴたり合う。
**ターン単位化が解いていたのは「50 ターン走る軌跡の行が終盤までキューに入らない」
という alfworld 固有の問題**であって、他の 2 タスクには最初から効いていなかった。

**`gen` の 66.1 → 69.0 を「生成が効率化した」と読んではいけない。** この phase は
いま **teacher forward をほぼ全部内包している**（rollout 中に走るので driver の
`_timer` では `gen` に落ちる）。teacher forward は元々 sm 89.8 の phase なので、
高い sm の仕事が `gen` に流入した分だけ平均が上がる ―― **phase の中身が変わった
比較**であって、生成そのものは何も変わっていない。rollout 内訳で見るのが正しい:

| | 旧 | 新 |
|---|---:|---:|
| rollout 壁時計 | 277.7 s | 280.9 s |
| ├ 生成（GPU busy） | 209.2 s | 205.9 s |
| ├ teacher（GPU busy） | 19.4 s | 28.1 s |
| └ glue（**GPU idle**） | 49.2 s | 46.9 s |
| **rollout の GPU-busy 率** | **82.3%** | **83.3%** |

**+1.0 pt。** これが「元々遊んでいた窓をどれだけ埋めたか」の実額で、
9.1 の −29.8 s に対して驚くほど小さい。理由は単純で、**この機構は idle を埋めて
速くしたのではなく、post-rollout にあった 32.4 s の仕事を、既に GPU が動いている
rollout の中へ畳み込んだ**から。step が 29.8 s 縮んでも、その分 phase 表の分母も
縮むので sm% はほとんど動かない。

**2.5 節の逆向きの実例になっている。** あちらは「util が下がって速くなる」変更の話
だった。ここは「**util がほぼ動かないまま 4.9% 速くなる**」変更である。どちらも
同じ結論を指す ―― **`sm%` を目的関数にしていたら、この機構は「効果なし」と
判定されていた。** 見るべきは `perf/throughput` の方（3.5 節）。

**帰属は引き算で閉じる。** `teacher_forward` −32.4 s ＋ rollout +3.2 s = −29.2 s に対し
step は −29.8 s（残差 0.6 s はノイズ）。**32.4 s 分の teacher 仕事を rollout 内へ移し、
その 90% が元々 GPU が遊んでいた glue に吸われた**という 3.3 節の設計どおりの結果
（内訳は上の rollout 表）。

**hit_rate 0.991 の 0.009 は設計上の残余。** 5,280 行のうち残ったのは 50 行で、
これは**最終ターン（turn 49、active 49）の行**そのもの。上の 1 で「最終ターン近傍の
行は原理的に残る」と書いた分がそのまま出た。チャンク 128 で足りており、
`ROLLOUT_PREFETCH_TEACHER_CHUNK` の追加調整は不要。

**ZeRO-2 の署名も今回は見えた。** `max_memory_allocated` +8.2 GB は gather 済み
teacher の分（上の 2 の見積り 3.4 GB/体 と整合）。`gen` フェーズの `pcieRX` が 768 → 208
に落ちているのは、その `gen` が**いま teacher forward をほぼ全部含んでいる**にも
かかわらず、なので、micro-batch ごとの all-gather が消えたことを示している。

**動かなかったもの:** `gen` と `update_actor` は横ばい。どちらも今回の 2 機構の対象では
ないので正常（`gen` を狙う spec decode は 3 で撤回した）。

**当初の限界と、その後の確認:** 上の比較は各 1 step で、仕事量を揃えたので
`s/step` を引き算できるが n=1 であることは変わらない ―― と書いて
「30 step 以上で `perf/throughput` を取り直すこと」を残していた。**取り直した。**

steps 260–299（**n=39**）の `perf/throughput` 平均は **3,782 tok/s** で、
3 機構のみの構成（steps 100–148、n=49）の 3,594 に対し **+5.2%**。
**n=1 の制御 A/B が出した +5.2% と一致する。** 39 step 分の
`teacher_prefetch/hit_rate` も 0.989〜0.994 の帯に収まり、
`max_memory_allocated_gb 103.060` も ZeRO-2 teacher の +9.2 GB 見積りと整合した。
この項目は解消済み ―― baseline からの累積は **+13.9%**（3.5 節）。

**なお n=1 A/B が当たったのは、仕事量を突き合わせたからであって
「1 step で十分だった」からではない。** 5 節⑤で誤ったのは、揃えていない 2 点の
`s/step` を差として報告した方である。両者を混同しないこと。

### 9.2 step 136 で OOM した。機構が teacher forward の「実行環境」を変えていた

9.1 の直後、**step 136 の teacher prefetch チャンクで GPU OOM**:

```
teacher_webshop_compute_ref_topk_log_prob -> lm_head
torch.OutOfMemoryError: Tried to allocate 10.47 GiB.
GPU 0 has a total capacity of 94.97 GiB of which 9.50 GiB is free.
（PyTorch reserved 84.26 GiB / allocated 74.67 GiB）
```

**10.47 GiB の正体は teacher のロジット**である。`compute_ref_topk_log_prob` は
`lm_head` を通すので語彙全体が実体化し、
`ref.log_prob_micro_batch_size_per_gpu=16` × webshop 級のプロンプト長 ≈ 2.3k トークン
× 151,936 × bf16 = 10.47 GiB がひとつのアロケーションになる。webshop の teacher で
落ちているのは、**チャンクがタスク別にまとまる**（`_teacher_prefetch_chunk`）ため、
最長プロンプトのタスクだけで埋まったチャンクが最悪ケースを作るから。
step 135 は webshop の行シェアが 0.271（step 126 は 0.194）と高かった。

**なぜ以前は落ちなかったのか ―― ここが今回の設計上の見落とし。**

`ROLLOUT_PREFETCH_TEACHER` は teacher forward を **rollout の中**へ移す機構である。
rollout 中は **vLLM が起きていて KV キャッシュを掴んでいる**（`gpu_memory_utilization=0.6`、
`ROLLOUT_KEEP_VLLM_AWAKE=1`）。post-rollout の `teacher_forward` フェーズは、
セッションが閉じてエンジンが sleep した後に走るので、**空きメモリの前提がまるで違う**。

- ターン単位化の前: hit_rate 0.28〜0.53 ―― teacher 仕事の**半分近くは post-rollout**、
  つまり広い方の環境で走っていた。
- ターン単位化の後: hit_rate 0.991 ―― **ほぼ全部が狭い方**で走る。

**仕事の総量は変えていない。走る場所を変えた。** 9.1 で測った −29.8 s はその移動の
利得だが、同じ移動が「vLLM が起きている間に 10 GiB のアロケーションを要求する」
という代償を持っていた。**「どこに仕事を入れられるか」は 2.4 節で配置の問題として
扱ったが、「入れた先にメモリがあるか」は見ていなかった。** 5 節①と同じ形の誤りである。

**対処:** `ref.log_prob_micro_batch_size_per_gpu` を 16 → **8**（run script）。
ピークが約 5.2 GiB に半減し、空き 9.50 GiB に対して余裕ができる。
このアームは `use_kl_in_reward` も `actor.use_kl_loss` も false なので、
`batch_size_divisor` は ref の micro を lcm に入れない（`utils.py:168`）＝
**adjust_batch のパディングもデータも動かない**。rmpad 下の log-prob は行ごとに
独立なので値も実質同一。より長いプロンプトのタスクを足すなら 4 まで下げる。

**ZeRO-2 teacher は残した。** 9.1 で pcieRX 768 → 208 の効果が出ており、
外せば +8.2 GiB 戻せるので、micro 8 でまだ落ちるならこちらが次の緩衝材になる。

**メモリ指標は当てにならない。** `perf/max_memory_allocated_gb` は 99.549 を報告して
いるが、落ちた GPU の全容量は 94.97 GiB で、`max_memory_allocated()` が自分の
デバイス容量を超えることはない。**この指標は単一デバイスの値として読んではいけない**
（sdar arm でも同じ矛盾が出ており、あちらは GPU のサイズ自体が揃っていない）。
メモリの判断は **OOM メッセージの数字**か、`nvidia-smi` で直接見ること。

### 9.3 まだ残っている枠

- **GPU-idle な glue が 46.9 s 残っている**が、hit_rate 0.991 なので teacher 仕事は
  もう無い。埋めるには別種の仕事が要る ―― pure OPD の薄いループには old_log_prob
  フェーズが無いので、この arm では埋められない。
- **`gen` が失点の 15.6 pt / 23.9 pt、依然として 2/3 を占める。** 表の 69.0 は
  teacher forward の流入で嵩上げされた値で、生成そのものは 2.3(b) の decode テール
  のまま手つかず。spec decode は 5 節⑧の理由で入らなかったので、
  **V1 エンジンの検証が次の一手。**


---

## 10. 捨てていた計算を消す（未計測、実装のみ）

9.3 の「glue はもう埋まっている」は**誤り**だった。turn table を 40 rollout ぶん
集計すると、glue の実効 sm は **34%** しかない。分解すると `gen` phase の 67.2% は

```
0.744 × 71.7(生成窓)  +  0.085 × ~95(tchWait)  +  0.172 × ~34(glue)  =  67.2
```

で、**glue を 100% 埋めても phase の上限は 78.5%**。失点の主因は最初から glue では
なく、生成窓そのものが 71.7% しか出ていないこと（＝ decode テール）だった。

同時に、**GPU が「忙しい」区間の中に、結果が捨てられている計算が 3 か所あった**。
sm% では原理的に見えない類のもので、実装のみ済み・**効果は未計測**である。

### 10.1 入れたもの

| # | 機構 | 何を消したか | 精度 |
|---|---|---|---|
| 0 | `rollout.disable_log_stats=False` | （計測）vLLM 内部統計を出す | — |
| 1 | teacher chunk の glue 追従 | 固定 128 → 前ターンの glue × 実測 rows/s（128〜512） | 値に触れない |
| 2 | `rollout.return_rollout_log_probs=False` | 毎ターン全生成トークンを Python 走査して作る `rollout_log_probs`。消費者は drift 検査だけで、この arm には比較対象の `old_log_prob` phase が無い | 生成トークン不変 |
| 3 | reward の decode 省略 | `num_examine=0` のとき捨てられる約 14,000 回/step の `tokenizer.decode` | reward tensor 不変 |
| 5 | **response-only `lm_head`** | prompt 行の語彙射影。`(rows, 151936)` を作って捨てていた。forward/backward 両方、かつ step 最大の活性 | **ビット非同一**（GEMM 形状） |
| 5b | 死んだ sampled-token log-prob | `pg_loss_coef=0` ＋ topk_kl では `log_prob` の唯一の用途が `.device` / `.dtype` の取得だった | 値不変 |
| 6 | session 中の `empty_cache` 抑止 | ① で vLLM を起こしたままにしてもなお毎ターン走っていた同期 | ビット同一 |

**5 が本命。** transformer 本体は削れない —— response の位置は prompt の KV を読むので、
因果 attention の下では prompt トークンも全層通す必要がある。**削れるのは射影だけ**で、
`logits_to_keep` に行選択を渡して `lm_head` をその手前に移す。lm_head は位置ごとの
線形写像なので、選んでから掛けても掛けてから選んでも同じ値になる。
prompt が全トークンの約 75% なので射影の約 3/4 が消える。

teacher 側にも同じ knob を入れた。9.2 の 10.47 GiB のスパイクは
`compute_ref_topk_log_prob` が `lm_head` を通すことによるもので、**その大半が
prompt 行**だった。ここが縮めば `ref.log_prob_micro_batch_size_per_gpu` を 8 から
戻す余地と、`gpu_memory_utilization` を 0.6 から上げる余地が同時に生まれる
（4 節の「次 run」候補は、**この順序でしか安全に入らない**）。

### 10.2 精度について

5 だけがビット非同一で、理由は GEMM の形状が変わることだけである
（`no_sync_grad_accum` と同じクラス）。student の勾配と teacher の targets の
両方に乗るので **intent lock に固定**し、3 arm 同時に入れる。他は値が動かない。

`_supports_logits_to_keep` が worker 起動時に模型の forward を検査し、
非対応なら**即座に落とす**。黙って従来経路に落ちると「速いはずが速くない」run が
静かに走ることになり、5 節⑤で記録した誤りと同じ形になるため。

### 10.3 残っている枠（優先順）

1. **`gen` の decode テール** —— 50 ターン中 `active ≤ 100` の 39 本が `gen` 壁時計の
   **70.5%** を食い、1 系列あたりのコストが head の **2.6 倍**。手つかず。
   spec decode は V0 では起動せず、**V1 エンジン検証が前提**。#0 の統計が
   その設計材料になる
2. gradient checkpointing の A/B（`bwd/fwd = 2.56`）。5 でメモリが空いた**後**
3. `ppo_micro_batch_size_per_gpu` 5 → 8。`adjust_batch` の lcm が 160 → 32 になり
   padding 行が約 3.6% → 0.5% に落ちる副次効果つき（10 は lcm 160 のままなので
   8 の方が筋が良い）
4. preprocess の batch tokenizer 化。ただし preproc を縮めると glue が縮んで
   `tchWait` が増えるので、5 の後
5. chunked prefill / KV 予算。**3 arm 同時**

### 10.4 検証方法

`s/step` で語らないこと（3.5 節）。同一 checkpoint・同一 step の n=1 制御 A/B で
`perf/throughput` と phase 内訳を取り、そのうえで 30 step 以上の throughput で
確認する ―― 9.1 → 9 節「限界」でやった手順そのものである。
**`sm%` は成果指標ではない**：5 は sm% が横ばいか下がったまま実行時間だけ縮む
可能性が高い（2.5 節）。見るのは `actor.fwd` / `actor.bwd` の壁時計と
`perf/max_memory_allocated_gb`、それに #0 で新しく見えるようになる
vLLM の prefill/decode トークン数と preemption 回数。

---

## 11. top-k の支持集合を teacher から student へ（`student_indexed_topk`）

**これは高速化ではない。** 損失そのものが変わる。3 arm 同時に入れ、
intent lock（`expected_multitask_config.yaml`）に固定した。

### 11.1 なぜ student 側が正しいか

いま使っている損失は「粗視化した reverse KL」である。支持集合 A の上では厳密、
A の外はすべて 1 個の tail bucket に潰す。落ちる誤差は恒等式で

```
KL_full − KL_A  =  tail_s · KL( p_s|Ā ‖ p_t|Ā )
```

—— **student の質量 `tail_s` で重み付けされている**。したがって A を teacher の
top-20 から取ると、student が teacher から外れて質量を置いた場所がちょうど A の外に
残る。それはこの項が罰するために存在する領域そのものである。A を student の top-20
から取ればそこが覆われる。

data processing により **どちらも full KL の下界で、常に ≥ 0**。つまりこれは
「別の目的関数」ではなく **同じ KL のより締まった下界**である。

### 11.2 なぜ追加 forward が 0 なのか

teacher の出力は

```
log p_t(v) = h · W_t[v] − lse_t
```

と分解でき、**ids に依存するのは最後の gather だけ**である。その gather は
`2·H·k`、teacher forward 全体の約 **1/42,000**。だから teacher は今と同じ場所 ——
rollout の CPU glue の中 —— で走り続け、`h` と `lse` だけを置いていく。student は
訓練 forward **1 回**で ids を選び、その ids で teacher を解決する。student forward は
1 回のままで、FLOPs はどの案でも同一である（`lse` は全語彙和なので `lm_head` は
どのみち避けられない）。変わるのは**スケジューリングとデータ移動だけ**。

### 11.3 これが持ち込む唯一の新しい危険：rank 所有権

teacher が走った rank と、その行を後で訓練する rank は**一致しない**。2 つの呼び出しの
間で行は task ごとに束ね直され、`adjust_batch` で padding され、そのうえ
`_balance_batch` が token 数を揃えるために並べ替える。**恒等写像になることは無い。**

`verl/workers/teacher_cache.py` はこれを 2 つの独立な番人で押さえる。どちらも
他方を包含しない —— 正しく自己整合な entry が誤った id の下に置かれれば witness は
通り、正しい id の下で壊れた entry は count を通る。

| 番人 | 問うこと | 破れ方 |
|---|---|---|
| 所有数（all-gather → SUM） | 「これは私が頼んだ行か」 | 0 = 誰も持っていない（**zero target**）、2 = 2 rank が同じ key を主張 |
| witness（teacher 自身の top-k で再計算） | 「保存した h/lse は teacher が出した値をまだ再現するか」 | 別の行と組めば数 nat ずれる |

### 11.4 実装した後に見つかった 3 つの誤り（すべて P0、修正済み）

| # | 誤り | なぜ静かだったか | 何が起きていたか |
|---|---|---|---|
| 1 | **cache key の上書き** | witness も同じ key で上書きされるので**自己整合のまま通る** | 行の key を全 response 位置にわたって繰り返していたため、行全体が「最後のトークン」の h/lse に潰れていた。**速度問題ではなく、学習対象そのものが変わる** |
| 2 | **prefetch miss 行が zero target** | 例外にならない | `sub = batch.select_idxs(idxs)` には `teacher_cache_ids` 列が無く（関数末尾で書かれる）、miss 行は cache されず −1 のまま。exchange が 0 を返し、`exp(0)=1` が全 k で立って tail 質量が負になり clamp に落ちる |
| 3 | **DP padding が id を複製** | 所有数 2 として顕在化はする（が run が落ちる） | `auto_padding` は行を丸ごと複製するので `teacher_cache_ids` も複製され、2 rank が同じ key を cache する |

修正：`put` は `(n, response_length, hidden)` / `(n, response_length)` の**行単位**しか
受け付けず、平坦化された入力と重複 key を書き込み時に例外にする。witness は
**全位置**を見る（padding は `lse == 0` で判定して飛ばす）。miss 行にも per-task loop で
新しい id を振る。`_teacher_call` が `pad_dataproto_to_divisor` で明示的に padding し、
padding 行の id を −1 にする。witness は `compute_teacher_log_probs` の**後**に
移し（cache が完成するのはそこ）、**1 回**に減らした（cache はプロセス単位なので
3 teacher に聞くと同じ entry を 3 回見ることになる）。

あわせて `temperature` を entry に持たせた。`lse` は forward が割った**後**の logits の
正規化定数なのに `h` は生のままなので、読み出し側で同じ除算をやり直す必要がある。
`temperature=1.0` に固定しているので現状は無害だが、T≠1 で静かに壊れる形だった。

### 11.5 費用 —— cache は「実際に学習される response 部分」だけを持つ

`response_length=512` は**上限であって長さではない**。この run の実測は
`global_seqlen/mean = 3,543,577` / 約 7,000 行 = **506 token/行**で、そのうち prompt が
約 75%（10.1 節）なので **response は平均 127 token**。padding 込みで持つと
**4 分の 3 は損失が一度も読まない領域**である。

そこで entry は行の**実位置だけ**に詰める。書き込み時に 1 回の gather で packed 化し、
各 key はその view を持つ（＝ padded 入力は即座に解放できる）。読み出しでは padding は
ゼロで再構成する —— もともとその値だったので、値は変わらない。詰めたぶん
`W[ids]` の gather と narrow GEMM も padding 位置を触らなくなる。

| 項目 | padded | **packed（現状）** |
|---|---|---|
| `h`（bf16） | 7.3 GB | **1.8 GB** |
| witness ids（int64）＋ lp（fp32） | 0.43 GB | **0.11 GB** |
| `lse`（fp32） | 7 MB | 2 MB |
| teacher lm_head の非 shard コピー × 3 | 1.9 GB | 1.9 GB（変わらず） |
| **合計 / rank** | **9.6 GB** | **3.8 GB** |

（rank あたり約 3,500 行、H=2048。行数と 75% は実測、127 token はそこからの導出。）

実位置は `attention_mask[:, -response_length-1:-1]` から取る。`lse == 0` でも同じ集合に
なる（`pad_input` がゼロ埋めするため）が、そちらは推論なので**両者が一致することを
テストで固定**し、本番では mask を使う。詰め戻しは prefix を仮定する —— response は
右詰めで、窓は最後の prompt トークンから開くので成り立つ —— が、**仮定せず検査する**：
穴があれば書き込み時に例外になる。

残る 1.9 GB は lm_head の非 shard コピー ×3 で、これは行数に依らない固定費である。

step 頭から `update_actor` の終わりまで GPU に載る。**9.2 で一度 OOM している**ので、
300 step を回す前に 1 step の smoke test で `perf/max_memory_allocated_gb` を見ること。

**二重保持を 2 か所で潰した**（どちらも精度には無関係、OOM とスループットの問題）：

1. **finalize 時の hidden 二重保持。** 読み出し側は連続 buffer が要るが、`torch.cat` で
   作ると元の put 単位 packed tensor を entry が参照し続け、**actor update の間ずっと
   cache 本体をほぼ 2 セット**持つ。行ごとに copy して entry をその slice に張り替え、
   ある chunk の最後の行が出た時点でその chunk を解放する。id は put 順に振られるので
   key 昇順の copy は chunk を 1 つずつ枯らしていき、**ピークは「store ＋ chunk 1 個」
   （約 1.8 GB ＋ 0.27 GB）**。`cat` なら「store ＋ 全 chunk」で 3.6 GB だった
2. **head stack 時の 1.9 GB ピーク。** 各 teacher を個別に clone してから `cat` すると
   両レイアウトが同時に存在する。しかもこれが起きるのは worker init 中 —— **vLLM が
   free memory を測って KV cache を決める直前**である。`register_teacher_lm_head` に
   slot を渡し、`summon_full_params` の中から**直接 stack 済み buffer の自分の slice へ
   copy** するようにした（clone も `cat` も無くなる）

### 11.6 student-topK に付随して消した無駄

**規定 ON**（`ppo_trainer.yaml`）。`teacher_kl_loss_type=topk_kl` のときだけ効き、
それを使う run script はこの arm だけなので他 arm への波及は無い。

| # | 消したもの | なぜ無駄だったか | 精度 |
|---|---|---|---|
| 1 | **teacher 自身の topK** | student indexing では誰も読まない。全語彙の選択 ＋ 行ごとに 2 回の scatter を全行に対して行い、さらに **約 860 MB/step** を driver に送って捨てていた | 値不変（witness としてのみ、既定 2 micro-batch/step だけ構築） |
| 2 | **LSE の二重計算** | topK 用と cache 用に同じ logsumexp を 2 回。step で最も横に広いテンソルに対する完全リダクション | ビット同一 |
| 3 | `topk(sorted=False)` | KL は支持集合上の和なので k 内の順序は読まれない | 値不変（テストで置換不変性を固定） |
| 4 | **lookup の host round-trip** | `logprobs_at` は micro-batch ループの中で **step あたり数千回**走る。そこでの `.tolist()` / `int(tensor)` は device→host 同期で、この一連の作業が守ろうとしている CPU の先行実行をそのつど止める | 値不変 |
| 5 | **done 行の collate** | 終了済み軌跡の行は `gather_rollout_data` で捨てられる。50 ターンの後半ではバッチの大半がそれで、列ごとに slice して dict を作ってから捨てていた | 値不変 |

4 の中身：teacher の 3 つの lm_head を `(3V, H)` として端から端まで並べ、ids に
task のオフセットを足す。こうすると **task ごとのグルーピングが不要**になる ——
micro-batch は `_balance_batch` によってタスク混在なので、3 つの重みを使い分けるには
mask → `nonzero` → 同期が要る。メモリは既に払っている 1.9 GB のままで、配置が変わるだけ。
所有権ガード（0 = 誰も持っていない、2 = 二重）は device 上で加算し、**mini-batch ごとに
1 回**読む。optimizer step の直前なので、未解決の行が重みに届くことはない。

また `w_ids` を float32 に広げるのをやめた（bf16 のまま、累積は float32）。ここで
最大のテンソルで、広げても精度は戻らない —— 積はどちらでも厳密で、累積はどちらも
float32 である。

### 11.7 まだ未検証

forward hook（`_capture_last_hidden`）と `FSDP.summon_full_params` 経路は
**実機で一度も走っていない**。CPU 上の 51 tests（うち 3 つは本物の 2 プロセス gloo）で
値・所有権・番人は押さえてあるが、FSDP 実体の上での動作は別物である。
1 step の smoke test を 300 step の前に必ず挟むこと。
