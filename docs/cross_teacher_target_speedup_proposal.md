# `claude/cross-teacher-target` の高速化提案 — sg1 run (`n9zfny6m`) の実測から

対象 run: `asayu255-/verl_agent_opd_grpo_cross_teacher_klw_sg1/runs/n9zfny6m`
（commit `d85d5c7`、`claude/source-gate-teacher-similarity` の基点、148 step、2× RTX PRO 6000 Blackwell Max-Q、host tamago）。

`claude/cross-teacher-target` は HEAD ではまだ走っていない。この run を材料にする理由は、
両アームが**同じ4モデルを同じ student top-k support で読み、同じ hidden-state cache を
同じ経路（`_all_teacher_planes` → `exchange_teacher_logprobs_multi`）で引く**からで、
target アームが足すのは (bs, resp, 20) 上の elementwise 演算と統計だけである。
下の phase 内訳は target アームでもそのまま再現すると見てよい。

集計は `scripts/wandb_phase_util.py`（本文書と同時に追加）で再現できる。wandb の system
指標を `timing_s/*` の順序で step 末尾から逆算して phase に帰属させ、util の**不足積分**
（`Σ (100−util)/100·dt`）で「カードが何も走らせていなかった秒数」を出す。

---

## 0.0 実装済み (1〜7) — 実装して分かった訂正が2件

本文書の §4 の 1〜7 は実装した。**8（KV 0.6→0.68）は未実施。** 実装中に §1.3 の記述が2箇所誤っていたことが分かったので、先に訂正する。

### 訂正1:pump は validation でも使われていなかった

§1.3 で「pump は validation のみが使う」と書いたが、**この run では1回も使われていない。** `rollout.return_rollout_log_probs=True` の間、worker の `_pump_refuse`（`vllm_rollout_spmd.py`）が「rollout log-probs are requested」でパス全体を拒否する — pool が返すのは token id であって log-prob ではないため。handshake が失敗し、`ROLLOUT_ASYNC_GENERATE=1` は**全 step の全 call で inert**だった。証拠は数時間前に1行出る `[rollout-pump] staying on the blocking path: ...` だけである。

これは `sign_prefetch/hit_rate = 0.000` と**同じ形の失敗**である:機構は実装され、既定で on で、script も export しているのに、別の設定が静かに殺している。

したがって #6 を効かせるには `return_rollout_log_probs=False` が必要で、3スクリプトともそう変更した。**代償は rollout-vs-actor のドリフト検査(`training/rollout_probs_diff_*`)を失うこと**で、これは実在する診断である。intent lock には入っていない(「diagnostic であって loss に届く値は同じ」と明記されている)ので変更自体は自由だが、失うことは失う。

### 訂正2:既存の「sync-free」パスは sync-free ではなかった

§3.C で `.item()` を 16回/micro-batch と数えたが、per-task ループは `x[rows]`(bool 索引)も使っていた。**bool 索引は出力サイズを host に読み戻すので、それ自体が sync である。** pure-OPD の `sync_free_task_metrics` パスも `response_mask[rows]` と `teacher_kld[rows]` で 2回/task/micro-batch 払っていた。

token-mean では行索引と行マスクは同じ数になる(除外行は分子にも分母にも 0 を足す)ので、ループ全体を `response_mask * rows` に置き換えた。seq-mean-* では等価でないので、そちらは索引のまま残してある。

### 実装の要約

| # | 手 | 実装 | ビット同一 |
|---|---|---|:---:|
| 1 | `perf/update_peak_{allocated,reserved}_gb` | `reset_phase_peak` / `phase_peak_metrics`(`verl/utils/metric/memory.py`)、`update_policy` の前後で開閉 | ○ |
| 2 | sign prefetch の辞退理由 | `_decline_sign_prefetch` が1回だけ理由を print、`sign_prefetch/enabled` を metric に | ○ |
| 3 | per-micro-batch host sync | pooled 4 + per-task 12 を `_defer`/`_defer_present` に。bool 索引を行マスクに。`sync_free_task_metrics` から `pg_loss_coef == 0` を削除 | ○(値)/×(per-task 診断の最下位ビット) |
| 4 | post-rollout のトークン予算 | `_teacher_call(..., budget=True)` → worker 側は `meta_info.setdefault` で caller override を許可。窓側は 4 行のまま | × |
| 5 | cache を put 時に pinned host へ | `TeacherHiddenCache._to_host`、`teacher_cache/device_gb` を追加 | ○ |
| 6 | 学習 rollout を pump へ + join を後ろへ | `ROLLOUT_PUMP_TRAINING`、handshake が rank の `n` を報告、`_pump_will_serve` が join の位置を決める | × |
| 7 | `ROLLOUT_PREFETCH_LOGPROB=1` | 3スクリプトで export | × |

**#3 の per-task 診断について。** 行マスクは token-mean で行索引と同じ値だが、加算順序が変わるので最下位ビットは動く。損失には一切触れないので勾配は不変である。

**新しい env フラグは3アームすべてに同じ値で入れた**(`ROLLOUT_PUMP_TRAINING`、`ROLLOUT_PREFETCH_LOGPROB`)。どれも非ビット同一なので、片方だけ立てると比較が壊れる。

### 走らせる前に読む指標

1. **`sign_prefetch/enabled`** — 0 なら #6/#7 以前に窓が動いていない。理由は起動ログに1行出る
2. **`teacher_cache/device_gb`** — #5 が効いていれば ~0。ここが `teacher_cache/gb` に近ければ put の host コピーが動いていない
3. **`perf/update_peak_allocated_gb`** — #8(KV 予算)と gradient checkpointing の判断はこれが出てから。§2.2 の「不明」がこれで潰れる
4. 起動ログの `[rollout-pump] driver: ROLLOUT_PUMP_TRAINING=...` と、`staying on the blocking path` が**出ていないこと**

---

## 0. 結論を先に

| 事実 | 数値 |
|---|---|
| step | **676 s**（148 step 平均、checkpoint 5回込み） |
| 内訳 | gen **243 s (36%)** / update_actor **241 s (36%)** / sign_weight_forward **143 s (21%)** / old_log_prob 44 s (6.5%) / それ以外 <1% |
| GPU-idle（不足積分） | **153 s/step = 23%**。gen 75 / update 51 / sign_wf 22 / old_lp 5 |
| メモリが張り付いているのは | **gen の間だけ**（NVML 96.5%）。old_log_prob 38–44%、sign_wf 45–52%、update 63–69% |
| 最大の単一損失 | **`sign_prefetch/hit_rate = 0.000`（全 148 step）**。3つの frozen forward が rollout の外で 143 s 走っている。機構は存在し（`ce34fbf`）、script は `ROLLOUT_PREFETCH_SIGN=1` を export している |

「GPU util が 100 に張り付いていないのに、メモリは 100」という前提は、**gen にしか当てはまらない**。
gen 以外の 3 phase（step の 64%）では NVML メモリは 40–70% で、そこに GPU-idle が 77 s ある。
ただし update の**本当の live peak は wandb からは読めない**（§2.2）。

提案は3段に分ける。
* **今すぐ（設定 / 数十行、ビット同一）** — 計測の穴を塞ぐ（§3.E）、hit_rate 0 の原因特定（§3.A1）、update の host sync 除去（§3.C）
* **中（1〜2日、非ビット同一だが同分布）** — serial path の frozen forward をトークン予算に（§3.B）、cache の put を host 直行に（§3.A3）
* **大（数日、3 arm 同時）** — pump を学習 rollout に開放して frozen forward を decode tail に隠す（§3.A4）。**これだけが 100 s 級**

---

## 1. 実測

### 1.1 phase 別（148 step 平均）

| phase | 秒 | share | GPU-idle 秒 | util% | NVML mem% | smActive | tensor pipe | util p10/p50/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gen | 243.5 | 36.0 | **75.3** | 69.2 | **94.0** | 48.3 | 17.6 | 38 / 72 / 88 |
| old_log_prob | 43.7 | 6.5 | 4.6 | 89.5 | 42.7 | 77.3 | 48.3 | 50 / 98 / 98 |
| sign_weight_forward | 143.1 | 21.2 | 22.1 | 84.5 | 51.7 | 65.8 | 41.0 | 58 / 90 / 96 |
| update_actor | 241.2 | 35.7 | **50.6** | 79.1 | 67.1 | 53.6 | 27.7 | 52 / 84 / 99 |
| save_checkpoint | 33.8 (5回) | 5.0 | 32.2 | 2.4 | 72.9 | 0 | 0 | — |

読み方の注意（`docs/gpu_profiling_report_opd.md` §2.1, §2.5）: `util%` は「カーネルが常駐していた時間割合」であって効率ではない。効率に近いのは `tensor pipe`。
**gen は tensor pipe 17.6%**、update は 27.7% — 両方とも「動いてはいるが薄い」。

### 1.2 1 step の時系列（step 107、15 s 刻み、抜粋）

```
   t(s)  phase                 mem%   util%   smAct  tensor
     15  gen                   90.7    29.5    31.6   13.6      ← vLLM wake + prefill
     45  gen                   96.5    74.0    73.5   45.5
     90  gen                   96.5    96.0    45.8   28.4
    135  gen                   96.5    74.0    47.0    2.8      ← decode tail: util 74 でも tensor 3%
    235  old_log_prob          38.6   100.0     2.1    nan      ← vLLM が寝てメモリが 96→39
    280  sign_weight_forward   45.4     0.0    46.4   31.6      ← RPC 境界
    340  sign_weight_forward   50.2    96.5    84.9   53.2
    430  update_actor          52.5     0.0     nan    nan
    475  update_actor          63.5    46.5    60.8   33.0      ← 周期的な落ち込み（§3.C）
    535  update_actor          65.7    45.0    78.1   43.7
    625  update_actor          69.0    31.5    60.6   30.5
```

### 1.3 この run の設定で速度に効くもの

| knob | 値 | 備考 |
|---|---|---|
| `ppo_micro_batch_size_per_gpu` / `ppo_mini_batch_size` | 5 / 60 | 5,261 行/step → **526 micro-batch/GPU、88 optimizer step**/step |
| `use_dynamic_bsz` | False | `ppo_max_token_len_per_gpu=9216` は設定されているが未使用 |
| `ref.log_prob_micro_batch_size_per_gpu` | **4 行固定** | `speedup_mechanisms.md` §7.2（窓内 OOM 対策で 16→4）。`ref.log_prob_use_dynamic_bsz=False` |
| `gpu_memory_utilization` | 0.6 | KV pool 57 GB。gen 中の 96.5% の主因 |
| `ROLLOUT_KEEP_VLLM_AWAKE` | 1 | rollout の間だけ。step 末で `end_rollout_session` → sleep(level=1)。**update 中に vLLM の pool は無い**（1.2 の 235 s で 96→39% に落ちるのがその証拠） |
| `ROLLOUT_PREFETCH_TEACHER` / `_SIGN` | 1 / 1 | on-task は hit 0.989、**sign planes は 0.000** |
| `ROLLOUT_ASYNC_GENERATE` | 1 | pump は **validation のみ**が使う（§3.A4） |
| `ROLLOUT_PREFETCH_LOGPROB` | 未 export | consumer は `opd_grpo_ray_trainer.py:81` に既にある |
| `TEACHER_CACHE_OFFLOAD` | 既定 1 | ただし **rollout 中は GPU に溜まり**、update の最初の read で host へ（§2.1） |
| `enable_gradient_checkpointing` | True | lock 固定。「切ると OOM」の根拠は §2.2 参照 |
| `BALANCE_MINIBATCH_COLUMNS` | 1 | `global_seqlen/microbatch_wait_frac_columns = 0.000` — DP 不均衡は解決済み |

---

## 2. メモリの実態 — 「張り付いている」の中身

### 2.1 gen 中の 96.5% は何か

vLLM の KV pool（0.6 × 96 GB ≈ 57 GB）＋ vLLM 重み 3.4 GB ＋ actor（bf16 param + fp32 master + Adam、`optimizer_offload=False`、shard_grad_op で ~10 GB/GPU）＋ frozen 4 モデル（sharded、6.8 GB/GPU）＋ ref 1.7 GB ＋ **teacher hidden cache**。

最後の項が見落とされやすい。`TeacherHiddenCache.put()` は `h.device`（GPU）に置き、`_finalize()` が**update の最初の read で**初めて pinned host にまとめて移す（`teacher_cache.py:563-650`）。つまり rollout 中、prefetch が積む cache は GPU に溜まる。on-task だけなら ~2 GB/GPU だが、**sign planes を prefetch すると 4 plane 分（`teacher_cache/gb` 15.6 → ~8 GB/GPU）が vLLM の隣に積まれる**。§3.A3 はこれを消す。

### 2.2 update 中の headroom は「不明」であって「無い」ではない

NVML 63–69% は torch の *reserved* を含む（`stall/cuda_mallocs` 259/step で segment が育ち、`empty_cache` されない）。live の peak を知る指標が無い: `perf/max_memory_allocated_gb` は `max_memory_allocated()` を **reset せずに読むプロセス生涯の高水位**（`verl/utils/metric/memory.py:69-`）なので、97.6 GB は gen 中の vLLM を含む値である。

intent lock が「gradient checkpointing を切ると step 1 の最初の micro-batch で OOM した（allocated 93.9 GB）」と記録しているが、その 93.9 も同じ生涯高水位を読んだ可能性が高い。**update の live peak を測るまで、checkpointing / micro-batch 5→10 の可否は決められない**（§3.E）。

### 2.3 allocator と GC は問題ではない（誤った方向へ行かないために）

* `perf/max_alloc_retries` は**累積**（`memory.py` の docstring）。148 step で 115 回 = 0.8 回/step。`expandable_segments` を入れる理由は無い（script のコメントも「入れない」と言っている）
* `stall/cuda_mallocs` 259/step ≈ 0.1–0.3 s/step。無視できる
* `stall/gc_gen2` 97/step は `host_gc`（既定 ON）が freeze 後に安価に回している回収で、設計どおり

---

## 3. 提案

### A. `sign_weight_forward` 143 s — prefetch が効いていない

#### A1. まず原因（診断 10 行、ビット同一）

`_prefetch_sign_planes` は `cross_teacher_enabled and _ROLLOUT_PREFETCH_SIGN` で走り、`_ROLLOUT_PREFETCH_SIGN` は**未設定でも True**（`opd_ray_trainer.py:62`）。on-task の hit が 0.989 なので chunk 関数自体は成功して返っている。それで sign 側が 0 になる経路は、静的には

1. 起動時の環境で `ROLLOUT_PREFETCH_SIGN=0` が明示されていた（メモリ §2.1 の理由で切った可能性がある）
2. `_prefetch_sign_planes` が `rows` 空で `None` を返した（`task_name` の正規化が `teacher_wg` のキーと食い違う）

のどちらかしか無い。run ログの `[rollout][teacher-prefetch] ... failed` 行と起動時 env を見れば決まる。
再発防止として、`_prefetch_sign_planes` が `None` を返す**理由**を1回ログに出し、`sign_prefetch/enabled` を metric に足す。
「hit_rate が 0」だけでは、切ったのか壊れたのか分からない — それが今回起きた。

#### A2. 効かせても glue には入り切らない（見積り）

`ce34fbf` の実測: glue は 34% busy、on-task prefetch（≈57 s 相当）で tchWait spill 21.6 s。つまり **glue の空きは ~35–40 s/step**。
そこへ 3 モデル 143 s を流せば ~100 s は spill として gen に載る。差し引き **−40 s 程度**。A1 単独の価値はこの程度で、A4 と組んで初めて 143 s が消える。

#### A3. `put()` を pinned host に直行させる（60 行、ビット同一）

`_finalize` の遅延コピーをやめ、`put` 時点で packed rows を pinned host へ side stream で DMA する（1 chunk = 512 行 × ~180 位置 × 2048 × 2 B ≈ 377 MB/plane、~40 ms、copy engine で重なる）。
効果は「gen 中の GPU メモリが cache サイズに依存しなくなる」こと — 4 plane prefetch（A1/A4）の前提であり、**空いた ~8 GB/GPU で `gpu_memory_utilization` 0.6→0.68 が可能になる**（既知候補、gen −3〜8%）。単独の速度効果は無い。

#### A4. pump を学習 rollout に開放し、frozen forward を decode tail に隠す（100–200 行、非ビット同一）

これが唯一 100 s 級の手で、根拠は2つある。

**(1) `TokenPump` は worker 内の背景スレッドで engine を回す**（`token_pump.py:80-85`）。RPC `pump_step` は submit/collect だけで即返る。したがって pump 経路では Ray actor の本スレッドが generation 中も空いていて、**teacher forward の RPC を decode と同時に実行できる**。`gpu_profiling_report_opd.md` §2.4「colocated worker は GPU 呼び出しを直列化する」を破る既存機構はこれだけである。

**(2) 学習 rollout は pump を使っていない。** `_pump_pins_one_sample`（`rollout_loop.py:151`）が `do_sample=True` を拒否するので、学習の generate は blocking path に落ちる。この run の `rollout.n=1`（`env.rollout.n=8` は driver 側 `repeat` で実現）なので、拒否の理由「n を複数欲しがるかもしれない」は形式的で、`config.rollout.n == 1` なら通してよい。

そのうえで `_join_teacher_prefetch()` を「次の generate の前」ではなく **rollout 末尾**に移す（現状 `rollout_loop.py:1563` は同一 WorkerDict を理由に generate 前に join している — pump ではその理由が消える）。

隠せる窓: decode tail は gen 壁時計の **70.5%**（`gpu_profiling_report_opd.md` §10.3、active ≤100 の 39 ターン）≈ 170 s で、そこは smActive 48% / tensor 17%。3 forward 143 s + old_log_prob 44 s（§3.D）をここに重ねる。
2 スレッドが GIL を取り合うので完全には重ならない。見込みは **−100 s 前後、上限 −180 s**（step −15〜27%）。

代償と条件:
* pump は request ごとに `(prompt, row)` から seed する（`seed_for_prompt`）ので、**同分布・非ビット同一**。3 arm 同時に入れる
* on-policy 性は保たれる: session 中は重みが凍っていて、teacher forward を挟んでも policy は変わらない。`docs/async_rollout_project.md` が退けたのは「訓練と生成の重畳」であって、これは違う
* decode tail に teacher forward が割り込むぶん per-turn latency は伸びる（4 行 forward ≈ 70 ms が decode step 20–30 ms の間に挟まる）。tail の critical path は latency 律速なので、gen が +20–40 s 伸びる可能性がある。上の見込みはそれを差し引いている
* 窓内のメモリは今の on-task prefetch と同じ形（`ref` 4 行 micro-batch、vLLM 隣）。A3 を先に入れて cache を GPU から外す

### B. serial path の frozen forward をトークン予算に（30 行、非ビット同一）

3 モデル × 4.2 M token = 4.3e16 FLOP を 143 s → **MFU ~30%**。`ref.log_prob_micro_batch_size_per_gpu=4` は 1 forward ≈ 3.2k token で、1.7B ではカーネル起動が支配的（tensor pipe 41%）。4 行は**窓内**（vLLM 隣）で OOM を避けるために §7.2 が決めた値で、**vLLM が寝ている post-rollout path に同じ上限を課す理由は無い**。

`compute_ref_topk_log_prob`（`fsdp_workers.py:1213-1225`）は `micro_batch_size` / `use_dynamic_bsz` を config から読む。`data.meta_info` に override があればそれを優先させ、driver の `compute_sign_weight_cache._cache` だけ `use_dynamic_bsz=True, max_token_len=18432` で呼ぶ（`speedup_mechanisms.md` §7.4 が「未着手」として設計済みの手）。prefetch 経路は 4 行のまま。

見込み: **−45〜55 s**（143 s の 30–40%）。A4 が入れば残る miss 分にしか効かないが、A4 の前に単独で入れられる。packed GEMM の長さが変わるので非ビット同一（同期待値）。

### C. update_actor の per-micro-batch host sync を消す（30 行、ビット同一）

`dp_actor.py:4047-4050` と `4112-4115`: GRPO 経路では `pg_loss / pg_clipfrac / ppo_kl / pg_clipfrac_lower` を pooled で 4 回、task 別に 4×3 回、**micro-batch ごとに 16 回 `.item()`** している。526 micro-batch/GPU/step なので **8.4k sync/step**。pure-OPD 用の `_defer`/`_defer_task` 機構が同じ関数の隣にあるのに、`pg_loss_coef != 0` の枝だけ延長されていない。

各 sync は backward の queue を drain させ、次の micro-batch の CPU 側準備（DataProto slice、4 plane の host packing loop、`exchange_teacher_logprobs_multi` の 4 collective）の間 GPU が空く。1.2 の update 中の周期的な 45% への落ち込みの候補。update の idle 51 s のうちどれだけかは **`ACTOR_TORCH_MICRO=20` で測ってから**（§4）。見込み −10〜30 s。

### D. old_log_prob prefetch（env 1 行、非ビット同一）

`ROLLOUT_PREFETCH_LOGPROB=1`。consumer は `opd_grpo_ray_trainer.py:81` にあり、pure-OPD の doc が「消費側が無い」と言ったのはこの GRPO アームには当たらない。ただし glue（35–40 s）を on-task teacher と取り合うので、**A4 なしでは差し引きほぼゼロ**。A4 と組で −30〜40 s。

### E. 先に入れる計測（15 行、ビット同一）

1. **`perf/update_peak_allocated_gb`**: `update_policy` の入口で `reset_peak_memory_stats()`、出口で per-rank max。§2.2 の「不明」を潰す。これが出るまで checkpointing / micro-batch 5→10 は判断しない
2. **`sign_prefetch/enabled`** と、`_prefetch_sign_planes` が None を返した理由の1回ログ（A1）
3. `timing_s/gen` の内訳は `ROLLOUT_TURN_TIMING=1` で既に出る（tchWait 列）。A1 の確認にそのまま使う

### F. やらない / 効かない（wandb が否定したもの）

| 手 | 理由 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | retries 115/run 累積。fragmentation は起きていない |
| GC チューニング | `host_gc` 既定 ON。gen2 97/step は freeze 後の安価な回収 |
| DP 不均衡 | `microbatch_wait_frac_columns = 0.000`。`BALANCE_MINIBATCH_COLUMNS` で解決済み |
| actor の `use_dynamic_bsz` | `exchange_teacher_logprobs_multi` が **micro-batch の形が全 rank で一致すること**を前提に loop 内で collective を呼ぶ（`teacher_cache.py:895-`）。ragged にすると hang する。5→10 のような固定形の変更は可（E1 の後） |
| gradient checkpointing off | E1 の実測が先。lock の OOM 記録は gen 混入の生涯高水位を読んでいる可能性がある（§2.2） |
| chunked prefill / KV 0.6→0.7 | 既知候補。A3 で 8 GB/GPU 空けてから、3 arm 同時 |
| `reward` / `adv` の overlap | 各 <1 s |

---

## 4. 優先順位と見込み

| # | 手 | 規模 | 見込み | ビット同一 | 前提 |
|---|---|---|---:|:---:|---|
| 1 | E 計測 | 15 行 | 0（判断材料） | ○ | — |
| 2 | A1 診断 | 10 行 | 0（原因特定） | ○ | — |
| 3 | C `.item()` 除去 | 30 行 | −10〜30 s | ○ | §4 の計測で確定 |
| 4 | B serial path token budget | 30 行 | −45〜55 s | × | — |
| 5 | A3 put→host | 60 行 | 0（A4/KV の前提） | ○ | — |
| 6 | A4 pump for training + late join | 100–200 行 | **−100 s 前後** | × | 3, 5。3 arm 同時 |
| 7 | D logprob prefetch | env 1 行 | −30〜40 s | × | 6 |
| 8 | KV 0.6→0.68 | config | gen −3〜8% | × | 5。3 arm 同時 |

3+4 だけで **676 → ~600 s（−11%）**、6+7 まで入れば **~450–500 s（−25〜35%）**。
評価は `gpu_profiling_report_opd.md` §10.4 の手順どおり — `s/step` ではなく同一 checkpoint・同一 step の n=1 制御 A/B で `perf/throughput` と phase 内訳、そのうえで 30 step の throughput。

## 5. profiler を走らせる場合

既存の計測基盤で足りる。再起動が要る。

```bash
GPU_PROFILER=1 GPU_PROFILER_ROLLUP_EVERY=1 \
ROLLOUT_TURN_TIMING=1 \
ACTOR_TORCH_MICRO=20 ACTOR_TORCH_SKIP=40 ACTOR_TORCH_DIR=$HOME/actor_trace \
bash examples/opd_grpo_trainer/run_multitask_cross_teacher_target_qwen3.sh trainer.total_training_steps=3
```

* `GPU_PROFILER`（phase タグつき NVML、0.3 s）— sign_weight_forward の `/base` `/alfworld` … 分割と、update 内の `actor.fwd / teacher_lookup / cross_teacher_target / bwd / task_metrics / optim` 分割。wandb の 15 s では見えない
* `ROLLOUT_TURN_TIMING`（per-turn `preproc / gen / tchWait / decode / envstep`）— **A1 の答え**（tchWait と `[rollout][teacher-prefetch]` 行）と、A4 が狙う tail の長さ
* `ACTOR_TORCH_MICRO`（`torch.profiler` の Chrome trace、rank ごと）— **C の答え**。`scripts/actor_trace_summary.py` で読む。update 内の 45% 落ち込みが `.item()` の sync なのか、`exchange` の collective 待ちなのか、cache の host pull なのかが、kernel 名で決まる
* `GPU_PROFILER_TRACE` は分散で多重 open する（未修正、§2.7）ので使わない。`GPU_PROFILER_SYNC_PHASES` は遅くなるので立てない
