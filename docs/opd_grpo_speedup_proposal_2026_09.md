# OPD+GRPO multitask（2 GPU）の高速化提案 — 2026-09-08 の実測から

状態: **提案のみ。実装なし。走行中の `ARM=pushback` run には触れていない**（物理メモリの実測は `nvidia-smi` の読み取りだけ）。
対象コード: `claude/opd-per-task-coef`（`f7333cb`）、起動スクリプト `examples/opd_grpo_trainer/run_multitask_opd_coef_qwen3.sh`。
先行文書: [speedup_mechanisms.md](speedup_mechanisms.md)（稼働中の機構一覧）、
[cross_teacher_target_speedup_proposal.md](cross_teacher_target_speedup_proposal.md)（前回の提案。#1〜7 実装済み、#8 未実施）、
[gpu_profiling_report_opd.md](gpu_profiling_report_opd.md)（pure OPD の profile）。

実測の出所:
* redistribute run（`76w6x8o4`、step 1〜89）の wandb ローカル `.wandb` から復元した 372 指標 × 89 step（`output.log` は 1 行 4096 文字で切れているため）
* pushback run の step 1〜2 を `nvidia-smi` で 5 秒おきにサンプルした物理メモリ（`docs/opd_grpo_speedup_gpu_samples_2026_09_08.tsv`、step 1〜2 の 5 秒サンプル）

---

## 0. 結論

1. **step の内訳（直近 30 step の中央値）: 482 s = gen 268 s（56%）+ update_actor 191 s（40%）+ old_log_prob 12 s + その他 3 s。**
   teacher forward は rollout 窓に隠れて 0.4 s。前回提案の #1〜7（pump、prefetch、host sync 除去）は効いている。
2. **「GPU メモリの上界に達している」は update_actor には当てはまらない。** 物理使用量は update 中 **rank 0 で 46.5 GB、rank 1 で 60.9 GB**（97.9 GB 中）。
   `perf/update_peak_allocated_gb` の 87 GB は、sleep 中の vLLM プール（CuMemAllocator = torch の pluggable allocator）が
   物理を解放した後も torch の会計に残る人工物である（起動スクリプト自身のコメントもそう書いている）。
   **update 側には 37〜50 GB の物理的な余裕がある。** 天井に張り付いているのは rollout 窓だけ（86.9〜91.4 GB、余裕 6〜11 GB）。
3. **gen は「デコード性能」ではなく「ターンの往復」に律速されている。** GPU あたり 1 step の生成トークンは約 40 万で、gen 268 s なら
   1.5k token/s。デコード 1 step が 5〜10 ms なら**実効的な同時デコード本数は 7〜15 本**、初期の 180 本に対して 1 桁小さい。
   alfworld は平均 36 ターン（上限 50）、1 軌跡 4,734 トークンを逐次生成するので、rollout の大半は少数本の tail である。
4. **提案の骨子は 2 つ。** (A) update は「余裕がある物理メモリを micro-batch の拡大に使う」（メモリは増えるが上限内、fixed-shape なら設定だけ）。
   (B) gen は「tail の 1 トークンあたりの待ちを削る」（投機的デコード、往復の短縮。メモリ中立）。
   (C) その前に 3 step の計測 run で、gen の内訳（turn table）と update の rank 別物理ピークを取る。
5. **見込み**: (A) で update −15〜35%、(B) で gen −15〜30%（要測定）。合わせて **482 → 350〜400 s**。
   これを超えるのは、次 step の rollout と update を重ねる 1-step off-policy 化（step ≈ max(gen, update) ≈ 280 s）だけで、
   これは実験の定義を変える（§3.11）。
6. **走行中の比較実験との整合**: 数値を変える手（micro-batch、dynamic bsz、checkpointing）は「加算順序」クラスで、
   比較する全アームに同じ値を入れる必要がある。pushback run は step 2 なので、いま knob を確定して再起動するのが最も安い（§4）。

---

## 1. 現状の内訳（実測）

### 1.1 phase 別

| phase | 89 step 中央値 | 直近 30 step | 備考 |
|---|---:|---:|---|
| step | 505.6 | **482.3** | |
| gen | 266.8 | **267.9** | 内部は未分割（`ROLLOUT_TURN_TIMING=off`） |
| update_actor | 218.1 | **191.1** | worker 側 185.5、Ray dispatch 5〜6 s |
| old_log_prob | 16.7 | 11.6 | 窓内 prefetch が効いた残り |
| teacher_forward | 0.5 | 0.4 | 窓内に隠れている |
| reward + adv | 1.5 | 1.4 | |

`opd_attribution` / `token_stats`（`every=5`）が走る step は update が **215 s 対 195 s（+20 s）**。平均で +4 s/step（1%）。

### 1.2 update の規模

| 量 | 値 |
|---|---:|
| トークン / step（2 GPU 合計） | 4.2 M（alfworld 2.7 M、webshop 1.3 M、search 0.2 M） |
| 行（ターン）/ step | 4.4〜6.9 k（step により変動） |
| `ppo_micro_batch_size_per_gpu` | 5 → rank あたり 440〜690 micro-batch / step |
| micro-batch あたりトークン | 平均 4〜5 k、最大 5 × (2552 + 512) ≈ 15 k |
| `ppo_mini_batch_size` | 60 行 → 73〜115 optimizer step / step（実験の定義、変えない） |
| `perf/mfu/actor` | 0.29 |
| DP 待ち | `microbatch_wait_frac` 0.011、`minibatch_wait_frac` 0.000（`BALANCE_MINIBATCH_COLUMNS` で解決済み） |

### 1.3 rollout の規模

| タスク | 軌跡 / step（2 GPU） | 平均ターン | 最大 | 応答トークン / 軌跡 | prompt 平均 |
|---|---:|---:|---:|---:|---:|
| alfworld | 120 | 36.2 | 50 | 4,734 | 435 |
| webshop | 120 | 8.0 | 15 | 1,615 | 1,126 |
| search | 120 | 2.4 | 4 | 305 | 476 |

GPU あたり応答トークン ≈ (4734 + 1615 + 305) × 60 ≈ **40 万 / step**。

---

## 2. メモリの実態（物理）

pushback run の step 1〜2、`nvidia-smi --query-gpu=memory.used`（5 秒間隔、97.9 GB のカード）:

| 区間 | rank 0 | rank 1 | util | 何が乗っているか |
|---|---:|---:|---:|---|
| rollout（session open） | 86.9 → 88.5 GB | 87.4 → 91.4 GB | 50〜90% | vLLM KV pool（0.6 × 97.9 − 重み ≈ 55 GB）+ 重み 3.4 + actor/optimizer + teacher 3 体 + 窓内 forward |
| gen 直後（sleep 後） | 22.9 GB | 22.9 GB | 0% | actor param/optimizer/grad shard + teacher shard + teacher head 1.9 GB |
| old_log_prob / teacher | 36〜46 GB | 33〜52 GB | 0〜100% | 上 + no-grad forward の活性 |
| **update_actor** | **46.5 GB** | **60.9 GB** | 80〜100% | 上 + 学習の活性（checkpointing 有効）+ torch cache |

**帰結。**

* `perf/update_peak_allocated_gb`（87 GB）、`perf/max_memory_reserved_gb`（103 GB > 物理）は容量の読みではない。
  起動スクリプトのコメント（「vLLM's CuMemAllocator is a pluggable allocator, so its pool lands in torch's accounting」）が正しい。
  容量判断は `nvidia-smi` か、OOM メッセージの "this process has N GiB in use" でしか出来ない。
* **update 中の余裕: rank 0 で 51 GB、rank 1 で 37 GB。** rank 1 の +14 GB は step 1〜2 とも同じ値で、live なのか torch cache なのかは
  rank 別の `memory_allocated` を update 末尾で読まないと分からない（§3.0 の計測項目）。保守的に rank 1 の 37 GB を上限に取る。
* **rollout 窓の余裕は 6〜11 GB しかない。** 窓内に仕事を足す手（`ROLLOUT_WINDOW_FORWARD_TOKENS`、KV 予算 0.6→0.68）はここが縛る。
* `perf/max_alloc_retries` は 89 step で 31 回（累積）、`stall/cuda_mallocs` 240/step。retry は窓内で frozen forward が
  プールの隣に活性を置くときに起きるもので、費用は step あたり 1 秒未満。`expandable_segments` は vLLM の CuMemAllocator と
  共存できない（スクリプトのコメントどおり）ので選択肢に無い。

以前「checkpointing を切ると step 1 の最初の micro-batch で OOM（91.38 GiB in use）」と記録されたのは、
いまの構成の update 中の物理使用量（46〜61 GB）とは整合しない。当時は vLLM のプールが update 中も mapped だったか、
窓内の frozen forward で起きたかのどちらかで、**現構成では再測定が要る**。

---

## 3. 提案

### 3.0 先に取る計測（3 step の probe run、GPU が空いたとき）

```bash
ROLLOUT_TURN_TIMING=1 GPU_PROFILER=1 GPU_PROFILER_ROLLUP_EVERY=1 VLLM_LOGGING_LEVEL=INFO \
ARM=control bash examples/opd_grpo_trainer/run_multitask_opd_coef_qwen3.sh trainer.total_training_steps=3
```

| 指標 | 答えるもの |
|---|---|
| turn table の `preproc / gen / tchWait / decode / envstep` と `genGPU%` | gen 268 s のうち、デコード・env.step・driver glue・窓内 frozen forward（`tchWait`）がそれぞれ何秒か。§0 の「往復律速」の直接確認 |
| vLLM の KV usage / preemption 行 | KV 予算が足りているか（0.6→0.68 の判断） |
| update 末尾の rank 別 `memory_allocated`（vLLM プールを除いた値。`torch.cuda.memory_stats` の pool 別集計、または NVML の rank 別読み） | rank 1 の +14 GB の正体、micro-batch を何倍まで広げられるか |
| `actor.fwd / bwd / optim / opd_diag` の phase 別時間（`GPU_PROFILER`） | 3.1〜3.4 の見込みの分母 |

費用: 3 step ≈ 30 分。既存の計測基盤だけで足りる。

### 3.1 [update] `ppo_micro_batch_size_per_gpu` 5 → 10（設定 1 行）

**機構.** micro-batch あたりトークンが 4〜5 k → 9〜10 k になり、rank あたりの micro-batch 数が半分になる。
消えるのは micro-batch ごとの固定費: (i) `shard_grad_op` の全パラメータ all-gather（bf16 3.4 GB のうち rank あたり 1.7 GB 受信、
NVLink 無しの PCIe で 45〜70 ms）×440〜690 回 = **20〜40 s/step**、(ii) teacher lookup の collective 4 本 × micro-batch 数、
(iii) 4〜5 k トークンの GEMM の起動費（MFU 0.29 の主因）。

**メモリ.** 活性は 2 倍（checkpointing 有効で +5〜7 GB、worst case の 15 k → 30 k トークン micro-batch で +13 GB）。
rank 1 の余裕 37 GB の内側。`adjust_batch` の padding は `micro × world` = 20 行の倍数になり、捨てる行は ≤ 19（現行 ≤ 9）。

**数値クラス.** 加算順序（10 → 5 にしたときと同じクラス）。比較する全アームに同じ値。

**見込み.** update −12〜20%（−25〜40 s）。以前の見積り「−2〜3%」は update を演算律速と見ていた時のもので、
MFU 0.29・tensor pipe 28% の実測と合わない。

### 3.2 [update] dynamic batch size を task 重み付き損失に対応させる（コード 2 箇所 + 試験）

**現状.** `use_dynamic_bsz=False`。`ppo_max_token_len_per_gpu=9216` と `dynamic_bsz_token_scale=True` は設定済みだが、
`check_task_weighting_supported`（`dp_actor.py:307`）が `normalize_loss_by_task` との併用を assert で拒否している。理由は 2 つ。

1. **損失のスケール.** 重み付き経路は `row_kl * task_loss_weight` の和に `task_dp_world_size * gradient_accumulation` を掛け、
   micro-batch 末尾で `/ gradient_accumulation` するので正味 `world_size` 倍。dynamic bsz の分岐（`:3081`）は
   `self.gradient_accumulation` を更新せず、末尾では token 数比で割る（`:4695`）ため、重み付き和が壊れる。
   **直し方**: dynamic 分岐で `self.gradient_accumulation = len(micro_batches)` を設定し、重み付き経路では
   token 比ではなく `/ self.gradient_accumulation` を使う。重みは step 水準の token 総数から来る絶対値なので、
   行の分け方に依らず和は不変（`seqlen_balancing.py:296` の注記と同じ論理）。
2. **teacher exchange の形.** `exchange_teacher_logprobs_multi`（`teacher_cache.py:989`）は
   「全 rank が同じ数・同じ大きさの micro-batch を回す」ことを前提に loop 内で `all_gather` を呼ぶ。
   `rearrange_micro_batches` は個数は `all_reduce(MAX)` で揃える（`seqlen_balancing.py:453`）が、各 micro-batch の行数は rank ごとに違う。
   **直し方**: exchange の前に行数を `all_gather` して最大長に padding する（(n, P) の key と (n, resp, k) の id を
   max n に揃え、余りは −1 / 0 で埋め、答えを切り詰める）。値は不変。

**メモリ.** ピークはトークン予算で上から抑えられる。予算 12〜16 k なら現行の worst case（15 k）と同等以下で、
平均 micro-batch は 4〜5 k → 12〜16 k。**ピークを増やさずに micro-batch 数を 1/3 にできる**のが 3.1 との違い。

**数値クラス.** 加算順序。

**見込み.** update −25〜35%。3.1 の上位互換だが、実装とテスト（両分岐、exchange の ragged 対応、`opd_diag` の集計不変性）が要る。
先に 3.1 で効果の向きを確かめてから着手するのが安全。

### 3.3 [update] gradient checkpointing を切る（3.1 と組で）

**機構.** backward が forward を再計算する分（backward の約 1/3）を消す。sg1 run の実測で `actor.bwd` は step の 35%。
この arm では update 191 s のうち bwd ≈ 90〜100 s と見て、**−25〜35 s**。

**メモリ.** micro 10 で活性 +13 GB（スクリプトの見積り）。rank 0 は余裕 51 GB、rank 1 は 37 GB → 収まるが、
3.1 の +7〜13 GB と足すと rank 1 で 37 GB に対し 20〜26 GB。**3.0 の rank 別計測が先。**

**数値クラス.** 再計算と保存で値は同じ（flash-attn の再計算は決定的）なので原理的にビット同一。lock に固定されているので lock 側の変更が要る。

### 3.4 [update] mini-batch 内で FSDP の re-gather を止める（FSDP2、大）

`shard_grad_op` は micro-batch ごとに全パラメータを gather し backward 後に reshard する。optimizer step が無い間はパラメータは変わらないので、
mini-batch 内（micro 5 で 6 回、micro 10 で 3 回）の gather は 1 回でよい。FSDP1 にその knob は無く、FSDP2（`fully_shard`）の
`reshard_after_backward=False` が要る。この fork は `fsdp2` 戦略を持つ（`fsdp_workers.py:443`）が、この arm では未使用。
**見込み −20〜30 s、メモリ +0**（gather 済みパラメータ 3.4 GB は計算中どのみち常駐）。FSDP1→2 の切替は数値がビット同一にならず、
checkpoint の互換・`no_sync_grad_accum` の書き換えも要る。3.1〜3.3 の後、別実験として。

### 3.5 [update] 診断の頻度（設定のみ、ビット同一）

`opd_attribution.every=5` と `token_stats.every=5` で 5 の倍数 step の update が +20 s。解析に使っていないアーム
（control / uniform / pushback）では `every=25`（checkpoint と同期）にして −3〜4 s/step。損失に触れない。

### 3.6 [gen] 投機的デコード（n-gram）— tail を直接狙う、メモリ中立

**根拠.** engine は **V1**（起動ログ `[rollout-engine] vllm 0.9.2, core=v1`）。以前 spec decode を撤回した理由（V0 の `SpecDecodeWorker` が
`sleep()` 未実装）は消えている。V1 の `vllm/v1/spec_decode/ngram_proposer.py` は draft モデル不要で、prompt/履歴の n-gram から
k トークンを提案し、rejection sampling で**元の分布を厳密に保つ**。alfworld の行動文（"go to cabinet 1" 等）は反復が多く、採択率は高いと見込める。

**なぜ tail に効くか.** 実効同時本数が 7〜15 本の区間では 1 デコード step は重み 3.4 GB の読み出し（≈2 ms）+ 起動費で、
演算器は遊んでいる。k=3〜4 の提案を 1 forward で検証すれば、採択率 60〜70% で **デコード step 数 −40〜55%**。
バルク区間（同時本数が多い）では逆に損なので、`speculative_config.disable_by_batch_size`（V1 で効くかは要確認）で
「running が N 本以下のときだけ投機」にする。

**入れ方.** `engine_kwargs.vllm` は既に `LLM(**engine_kwargs)` に渡っている（`vllm_rollout_spmd.py:205,250`）:
`+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={method:ngram,num_speculative_tokens:4,prompt_lookup_max:4,prompt_lookup_min:2}`。

**未確認（3 step probe で潰す）.** sleep mode・`external_launcher`・pump・prefix caching との共存、`disable_by_batch_size` が V1 で効くか、
採択率（vLLM の spec decode metrics）。**メモリ**: draft トークン分の KV slot のみ。**数値クラス**: 同一分布（③ prefix caching と同じ扱いにできる）。

**見込み.** gen −15〜30%（tail が gen の 70% で、その半分弱を削る）。

### 3.7 [gen] ターンの往復を短くする（設計、要測定）

実効同時本数が 1 桁ということは、各系列が engine の外（env.step、driver の decode/tokenize、pump の round 待ち）にいる時間が
engine の中にいる時間と同程度以上ある。turn table の `envstep / preproc / decode` 列と `[rollout-pump]` の round 周期がそれを分ける。

候補（測ってから）: (i) alfworld の `env.step` の実時間（TextWorld は Python、`num_cpus=0.1` の Ray actor）、
(ii) 終わった系列を round の末尾まで待たせず即再投入する粒度、(iii) search の retrieval（remote、10 ms コアレッサ）の往復。
**メモリ中立、数値クラスは pump と同じ（同一分布）。** 見込みは測定前には出せない。

### 3.8 [gen] 窓内 frozen forward の見直し（turn table の `tchWait` を見てから）

teacher / old_log_prob の prefetch は pump の間に actor の call slot で走る。同時本数が少ない tail では、4 行 chunk の forward
（50〜100 ms）が pump の round を遅らせ、gen を延ばす（スクリプトのコメント: 「gen grew by very nearly the same amount」）。
`tchWait` が大きければ、(a) chunk をトークン予算（`ROLLOUT_WINDOW_FORWARD_TOKENS=8192`、窓の余裕 6〜11 GB の内側で）、
(b) old_log_prob prefetch は窓から外し update 前にトークン予算で回す（現在 12 s なので失うものは小さい）、のどちらかを選ぶ。

### 3.9 [gen] KV 予算 0.6 → 0.68（前回提案 #8、未実施）

窓の物理余裕が 6〜11 GB なので、+8 GB の KV は rank 1 で収まらない可能性が高い。3.0 の KV usage / preemption 行で
「足りていない」と出た場合だけ、窓内 forward を縮めた分（3.8）と引き換えに入れる。

### 3.10 [memory] fp8 KV cache

KV の読み出しとプールを半分にする。だが同時本数 1 桁の tail では KV 帯域は律速ではなく、バルク区間でも
1 系列 0.5〜1 k トークン × 180 本 ≈ 10〜18 GB/step の読みで 5〜10 ms、重み読みと同程度。効果は限定的で、
数値は変わる（lossy）。Blackwell（sm_120）+ vLLM 0.9.2 の attention backend で fp8 KV が動くかも未確認。優先度低。

### 3.11 [構造] 次 step の rollout と update を重ねる（1-step off-policy）

gen の tail（GPU は 7〜15 本のデコードしかしていない）に update（191 s）を重ねれば step ≈ max(gen, update) + α ≈ 280〜300 s（−40%）。
これが唯一 100 s 級の手だが、生成に使う方策が 1 step 古くなる。PPO の比は生成時方策の old_log_prob に対して取るので
更新自体は整合するが、**on-policy という実験の定義が変わる**（`gpu_profiling_report_opd.md` §2.6 の判断）。
現在の比較系列には入れない。系列が終わった後の別実験として記録する。

---

## 4. 適用の順序と、走行中の比較との整合

| 順 | 手 | 数値クラス | メモリ | 前提 | 見込み |
|---|---|---|---|---|---:|
| 0 | 3.0 計測 probe（3 step） | — | — | GPU が空く | 判断材料 |
| 1 | 3.5 診断頻度 | ビット同一 | 0 | なし | −3〜4 s |
| 2 | 3.1 micro 5→10 | 加算順序 | update +7〜13 GB | 3.0 の rank 別ピーク | −25〜40 s |
| 3 | 3.6 n-gram spec decode | 同一分布 | ≈0 | probe で共存確認 | −40〜80 s |
| 4 | 3.3 checkpointing off | ビット同一（原理） | update +13 GB | 2 の後の再計測 | −25〜35 s |
| 5 | 3.8 窓内 forward の整理 | 同一分布 | 窓内 ±5 GB | turn table | 0〜30 s |
| 6 | 3.2 dynamic bsz | 加算順序 | ピーク低下 | 2 の効果確認、実装 | 2 に対し追加 −20〜40 s |
| 7 | 3.4 FSDP2 re-gather 抑止 | 非ビット同一 | 0 | 別実験 | −20〜30 s |
| — | 3.11 1-step off-policy | 定義変更 | 0 | 系列終了後 | −180 s |

**比較との整合.** 2・4・6 は勾配の加算順序を動かす。10→5 のときと同じく「比較する全アームに同じ値」が条件で、
片方だけ変えると redistribute / pushback / control / uniform の比較が壊れる。いま走っている pushback は step 2、
control と uniform は未着手なので、**knob の集合をここで確定して pushback を再起動するのが最も安い**。redistribute（step 89 まで）は
旧 knob のままなので、比較に使うなら旧 knob の control が要る。これはユーザーの判断。

**検証手順（各手ごと）.** 3 step の probe で (i) phase 別時間、(ii) rank 別 NVML の update ピーク、(iii) ビット同一を主張する手は
step 1〜3 の `actor/pg_loss`・`actor/teacher_kl_loss` と `[val-hash]` の一致、を取ってから本番に入れる。

---

## 5. 見積りの根拠と限界

* update の内訳（fwd / bwd / optim）はこの arm では未計測。3.1・3.3 の見込みは sg1 run の profile（bwd 35%）と MFU 0.29 からの推定。
* gen の「往復律速」は生成トークン数 ÷ gen 時間 ÷ デコード step 時間の推定で、turn table による直接測定ではない。
  デコード 1 step の時間（5〜10 ms）は V0 時代の実測（6〜7 ms）の転用。
* rank 1 の +14 GB は正体未特定。live なら余裕は 37 GB、cache なら 51 GB。
* n-gram の採択率は未測定。反復の多い alfworld で高く、webshop の商品説明で低いと予想するが、数値は probe 待ち。
* spec decode と sleep mode / `external_launcher` / pump の共存は未確認。V1 ではどれも engine 内部の機構なので原理的な衝突は無いが、
  「原理的に無い」で走らせた機構が 2 度静かに無効化された経緯（pump、search batch）があるので、probe の `[rollout-engine]` 行で確認する。
* 物理メモリのサンプルは pushback run の step 1〜2（2 step）。step が進むと行数が増える（alfworld の成功率上昇でターンが増減）ので、
  update のピークは ±5 GB 動きうる。
