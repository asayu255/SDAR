# 6 項目の改良案に対する実測判定(2026-08-27)

外部から提示された改良案 6 項目に、**このリポジトリで実際に測った結果**で答える。
各項目の事実記述は**ほぼすべて正しい**。分かれたのは**効果の見積もりと優先順**である。

対象は `trainer.val_only=True` の multitask 評価(Qwen3-1.7B、A6000 × 3)。
一次資料は `docs/eval_gpu_util_status.md`、時系列は `docs/eval_gpu_utilization.md`。

---

## 0. まず、天井の表を差し替える

提示された表:

| 段階 | GPU util |
| --- | ---: |
| 現在 | 79.90% |
| EMPTY を完全に除去 | 87.96% |
| PARTIAL も完全に除去 | 91.07% |
| vLLM 内部 gap も完全に隠蔽 | 理論上 100% |

**最後の行が測定で否定された。** `num_scheduler_steps=4` で duty は
91.07% → **92.55%**。gap を scheduler 由来 S とそれ以外 R に分けると:

```
S + R   = 8.93   (steps=1)
S/4 + R = 7.45   (steps=4)
→ S = 1.97,  R = 6.96
```

**engine 設定が触れるのは 9 pt のうち 2 pt だけ。** `steps=8` は +0.24 pt。
残る 7 pt は step ごとの output 処理、CUDA graph replay、forward 内部の隙間で、
`engine_kwargs` から触れるつまみは無い。

そして**測って分かった最も重要なこと**は、この表の縦軸そのものである:

| 完走 run | 構成 | node util | duty | **wall** | success |
| --- | --- | ---: | ---: | ---: | ---: |
| `...-092241` | V1, depth 3, **pump** | 79.90% | 91.07% | **0.96 h** | 0.3875 |
| `...-124410` | **V0 + ms4**, depth 3 | **83.21%** | **92.55%** | 1.24 h | 0.3865 |
| `...-201115` | V1, depth 2 | 81.2% | — | 1.26 h | 0.3869 |

**util が最も高い構成が、最も速い構成より 29% 遅い。** 同じ checkpoint、
同じ 3 枚、スコアは 3 本とも 0.001 以内。**util で選ぶと 29% 遅い設定を選ぶ。**

「実用上の目標は 95〜99% かつ ms/row 最小」という指摘は正しいが、
**この 2 つは並立しない。** wall が決める。

---

## 1. trajectory 単位の連続実行 —— **不要になった**

### 事実記述は正しい

`responses = [future.result(timeout=timeout) for future in futures]` が
batch 全体を待つ —— **確認した**(`rollout_loop.py`)。

### だが、それが直すはずだった tail は pump が既に直していた

item 1 を作りかけて、根拠にした turn table が **pump OFF の run** だったことに
気づいた。pump ON で測り直すと:

| turn | active | pump OFF | **pump ON** |
| ---: | ---: | ---: | ---: |
| 2 | 58 → 40 | 13.70 s | **5.58 s** |
| 3 | 12 → 7 | **10.82 s** | **3.17 s** |

slot A の turn 3(7 行)と slot B の turn 1(252 行)が**同じプールに入る**ので、
engine は 7 行ではなく 259 行を見る。tail は生成 wall の 48% → **32%**。

### そして真の per-trajectory は環境層で塞がれている

`MultiTaskEnvironmentManager.step` は `managers[task].step(actions)` に
**そのタスクの全行**を渡す。下位マネージャは行の部分集合を進める術を持たず、
部分ステップは `_task_steps` を破綻させる。**環境マネージャの改造が前提**になる。

**判定: 生成側は完了、環境側は測定された的が残っていない。着手しない。**

---

## 2. `async_scheduling` —— **落とす**

### バージョン不一致の指摘は正しい

`environment.yml` は `vllm==0.11.0` を固定、実機は **0.8.5**。確認した。
そして 0.8.5 の **V0** には `num_scheduler_steps` がある —— これも正しく、
実際に動いた(`[rollout-engine] vllm 0.8.5, core=v0`)。

### だが的は 8.93 pt ではなく約 2 pt である

§0 の分解のとおり **S = 1.97 pt**。`async_scheduling` が隠すのは**同じ S**。

0.8.5 → 0.11.0 は kernel と reduction 順が変わるので、**過去のスコアが
全部比較対象でなくなる。** 約 2 pt にその代償は見合わない。

**判定: 保留。** やるなら独立実験として、スコア基準の作り直しを込みで。

### 付随して直したもの

V0 は組めたが `AttributeError: 'MultiStepModelRunner' object has no attribute
'model'` で落ちた。multi-step が model runner を包み、FSDP → vLLM の重み同期が
届かなくなる。`unwrap_model_runner()` で解決(`b2adf4d`)。

`VLLM_USE_V1=0` は**要求であって結果ではない**(vLLM は serve できない構成では
黙って V1 に戻り、`num_scheduler_steps` は受理されて無視される)ので、
`ROLLOUT_REQUIRE_CORE=v0` で build 時に落とすようにした(`f7198a6`)。

---

## 3. preprocessing の batch 化 —— **保留**

### 事実記述は正しい

`_run_full_preprocess` が行ごとに `preprocess_single_sample` を回している。確認した。

### だが preproc は支配していない

turn table の `preproc` は **turn あたり 0.14〜1.9 秒**、batch 合計 32 秒に対して。
そして activity census で EMPTY のとき **preproc は 0.2〜0.6**(3 slot 中)。
**一度も過半を取っていない。**

見積もられた「EMPTY/driver glue の 2〜5 pt」を支持する測定が無い。

**判定: 保留。** BPE 境界問題が無く検証しやすいという理由は正しいので、
**census が preproc を名指ししたら**着手する。

---

## 4. Search の `step_batch` —— **落とす**

### 事実記述は正しい

252 個の `search()` を投げて `_Coalescer` が 100 ms 待って束ね直している。確認した。

### だが的が消えた

activity census で **`envstep` が 0.05 未満**、つまり出てこない。
そして DEEP EMPTY の判定は **`EMPTY is the DRIVER RUNNING PYTHON`**
(driver CPU 80% of one core while EMPTY vs 11% while busy。別 run で 82%/15%)。

**retriever は原因ではない。** 提示された分岐の
「`EMPTY is a wait OFF the box`」側(retriever replica、IVF/HNSW)は
**この証拠では正当化されない。**

**判定: 落とす。**

---

## 5. Pump の RPC —— **指摘は正しく、処方が違った**

### 回数の指摘は正しい

request が残る間 20 ms ごとに 3 worker への collective RPC。確認した。

### だが 2 点で処方が誤り

1. **worker 側は既に long-poll である。** `pump_done.get(timeout=timeout_s)` は
   完了があれば即返る。`round_s` を上げても**待ちは増えない** ——
   空ポーリングが減るだけで、私が説明した「latency との引き換え」も存在しない。
2. **空ポーリングは主コストではない。**

### 主コストは `.tolist()` だった

```python
def _as_id_list(prompt_token_ids):
    return prompt_token_ids.tolist() if hasattr(...) else list(...)
```

`raw_prompt_ids` は numpy 配列で来るのに、Ray に渡す前に Python list へ展開して
いた —— **1 token につき 1 個の Python int オブジェクト**。
252 request × 約 1,300 token = **1 ターンあたり約 33 万オブジェクト**を
生成・pickle・送信・unpickle。**driver スレッドで、census が `gen` とタグし
カードが空と読む窓の中で。**

int32 配列は pickle protocol 5 の 1 バッファ(memcpy)。変換は **worker の
vLLM 境界**でやる —— list が必要なのはそこだけ。**待ちは増えない。**

戻り側は反転だけでは駄目で、`pad_2d_list_to_length` が `tuple(sub_list)` を
作るため同じコストが下流に移る。`_pad_rows` に配列専用経路を足し、
**2 経路が同じテンソルを出すことをテストで固定**した(`1b889d8`)。

**判定: 実装済み。** `round_s=0.1` は無害な追加だが、**単独で測るのは
`.tolist()` の効果を確定させてから。**

---

## 6. rank 割当を予測 token 時間で —— **保留**

### 事実記述は正しい

`rank = min(range(world_size), key=lambda r: placed[r])` が件数で分けている。確認した。

### だが測定された的が小さい

PARTIAL は pump ON で **6.6%**、per-GPU 平均は **80.2 / 78.6 / 78.2**
(spread 1〜2 pt)。**特定 rank が遅いのではなく、毎回ちがう rank が
順番に待っている** —— routing の選択ではなく collective の構造である。

そして pump がその構造の尾を大部分消している(§1)。

**判定: 保留。** LPT と sticky routing の設計自体は妥当なので、
PARTIAL が再び大きくなったら戻る。

---

## 7. `cpu_pct >= 60` への批判 —— **正しい。撤回して作り直した**

「driver が CPU を使っている」までしか証明できない、という指摘はそのとおり。
Rust の tokenizer は GIL を離すし、native BLAS は複数コアぶん出る。**撤回した。**

同じ timestamp で phase を記録すべき、という提案も採った。ただし
`push_phase` のスタックは**スレッドローカルではない**ので slot 3 本では壊れる。
**数える方式**にした(`activity()` / `activity_snapshot()`)。

そして判定は **2 つの読みが一致したものだけ**にした:

| 計器 | 答えられること |
| --- | --- |
| `cpu_pct` | **働いていたか**(ソケット待ちは CPU を焼かない) |
| activity census | **どこで**(1 phase が **slot 数の**過半を占めるとき) |

食い違えば `these disagree`、CPU が 20〜60% なら `UNRESOLVED`。
さらに pump ON では **二峰分布**(待ちと実行が混ざる)なので、平均をやめて
**BLOCKED / RUNNING の割合**で出す。

> **この判定は 2 回振った。** 1 回目は cpu 単独、2 回目は census 単独。
> **どちらも単独では答えられない質問に、単独で答えさせていた。**

---

## 8. 実験順への回答

| 提示された順 | 判定 |
| --- | --- |
| 1. 現コードで 1 本、DEEP EMPTY と version/core を確定 | **完了。** `EMPTY is the DRIVER RUNNING PYTHON`、`vllm 0.8.5, core=v0` |
| 2. `ROLLOUT_PUMP_ROUND_S=0.1` だけ | **処方が違う。** `.tolist()` を先に直した(実装済み) |
| 3. `async_scheduling` だけ | **落とす**(的 ~2 pt、スコア基準が全部リセット) |
| 4. batch tokenizer | **保留**(census が preproc を名指ししていない) |
| 5. Search `step_batch` | **落とす**(`envstep` が 0.05 未満) |
| 6. trajectory 単位 scheduler | **不要**(生成側は pump が完了、環境側は塞がれている) |

### 判定に使う計器はすべて実装済み

```bash
bash examples/sft_trainer/judge_eval.sh <control.log> <candidate.log>
```

`[rollout-pump]` / `[rollout-engine]` が噛んだか、`[gpu-residency]` の
EMPTY / PARTIAL / duty、`ms/row all`(batch 構成の違いを割り算で落とした唯一の
速度軸)、`val/*/test_score`、`[val-hash]` を一度に出す。

**完走が要るのはスコアだけ。** しかも `[val-hash]` が対照と一致していれば
生成は変わっておらず、スコアは動きようがない。末尾の VERDICT が
「まだ早い／いま判定可能／スコアのために完走が要る」のどれかを言う。

---

## 9. いまの構成と、いまの数字

```bash
ray stop --force
bash examples/sft_trainer/eval_checkpoints.sh <step>
```

既定で入るもの: **pump + REQUIRE**、`VAL_PIPELINE_DEPTH=3`、
`ROLLOUT_GPU_MEM_UTIL=0.75`、`val_per_task_batch_size={alfworld:126,search:252,webshop:126}`、
`SEARCH_URL=:8000`、`SEARCH_TCP_USER_TIMEOUT_S=10`、`GPU_PROFILER`、`ROLLOUT_TURN_TIMING`。

| | wall | node util |
| --- | ---: | ---: |
| retriever が壊れていた時 | 2.85 h | 57.9% |
| retriever 修正(唯一の大きな勝ち) | 1.26 h | 79.0% |
| **+ depth 3 + KV 0.75 + pump** | **0.96 h** | 79.90% |

**wall は 2.97 倍。util は 79.9%。**

---

## 9bis. 配列転送は Ray 境界の手前で無効化されていた(訂正)

外部レビューの指摘。`_generate_via_pump` が `np.int32` 配列にした直後、
**`PumpClient.submit` が 1 関数先で `list()` を掛けていた:**

```python
self._inbox.append((request_id, list(prompt_token_ids), carried))
```

`list()` を配列に掛けると **`np.int32` スカラーの list** になる ——
置き換えたはずの Python int **より重い**。

この機械で測り直した(252 request × 1,300 token、pickle protocol 5):

| 転送形式 | pickle | unpickle | サイズ |
| --- | ---: | ---: | ---: |
| `list[ndarray[int32]]`(修正後) | **3.3 ms** | **2.0 ms** | 1.32 MB |
| `list[list[np.int32]]`(修正前) | **922.1 ms** | **116.0 ms** | 4.92 MB |

**1 ターンあたり約 1 秒、driver スレッドで。**
census が `gen` とタグし、カードが空と読む窓の中で。

### テストが通り続けた理由

`_as_id_list` **単体**のテストを書いていた。**helper は正しく配列を返していた。**
壊れていたのはその先で、**payload を見ていなかったから見えなかった。**

`client._inbox` の実際の payload が `ndarray` であることを確認するテストに
差し替えた。**修正前のコードで落ちることを確認済み。**

> **「GPU の空きを付け替えるのではなく、driver の仕事そのものを消す」**
> —— §10 で自分が書いた基準に、自分の実装が届いていなかった。

---

## 10. 4 回観測された保存則

| 変更 | 下がったもの | 上がったもの |
| --- | --- | --- |
| depth 3 | EMPTY 8.9 → 3.8% | PARTIAL 7.7 → 12.7% |
| pump | PARTIAL 6.79 → 3.11 pt | EMPTY 3.29 → 8.06 pt |
| V0 + multi-step | duty 91.07 → 92.55% | wall 0.96 → 1.24 h |
| pump の tail 解消 | tail 48% → 32% | cpu-glue 6.9 → 15.1% |

**4 回とも、狙った側は機構どおりに動き、総和は動かなかった。**

これが「95〜99% を狙う」に対する最も重要な留保である。**GPU 側の並べ替えでは
総量が変わらない。** 変わったのは 2 回だけ —— retriever の修正(外部の直列資源を
消した)と `.tolist()`(driver の Python 仕事そのものを消した)。

**次に効くものがあるとすれば、同じ形をしている:**
GPU の空きを付け替えるのではなく、**driver の Python 仕事か、外部の待ちを
実際に消すもの。**

---

## 11. 配列転送を完走させた後の測定 —— **wall は動かない。案2 も同じ理由で効かない**

### 測った対

`eval_log_inventory.sh` で 11 本の eval ログの設定を読み、4 列(PUMP / CORE /
STEPS / DEPTH)が一致する対を選んだ。**ファイル名では選べない** ——
`eval_pump_ms4.log` は名前に反して V0 + `num_scheduler_steps=4` であり、
これを V1 の候補と比べた最初の判定は 2 つの変更を同時に測って片方の手柄に
していた。

| | PUMP | CORE | STEPS | DEPTH | batches | wall | ms/row |
|---|---|---|---|---|---:|---:|---:|
| `eval_pump.log`(修正前) | ON | v1 | 1 | 3 | 55(打切) | 726.9s @25 | 77 @55 |
| `eval_arraywire.log`(修正後) | ON | v1 | 1 | 3 | 180+ | 742.3s @25 | 82 @55 |

**同一 prefix での比較 —— ms/row +6.5%、wall +2.1%。速くなっていない。**

### 単体では 926 ms/turn を消している

| | 時間 |
|---|---:|
| `list(ndarray)` の構築(driver、修正前) | 15.8 ms |
| その list の pickle(修正前) | 922.1 ms |
| 配列の pickle(修正後) | 3.3 ms |
| `tolist()`(worker、vLLM 境界、修正後に追加) | 8.8 ms |

**1 turn あたり約 938 ms → 約 12 ms。** search バッチは 4 turn なので
バッチあたり約 3.7 s。バッチの実測は約 17 s だから、**もしこれが critical path
上にあれば wall は 2 割落ちるはずだった。1 秒も落ちていない。**

### 従って: あの 926 ms は他スロットの生成の陰に完全に隠れていた

これは pump が設計どおり働いている証拠であり、同時に **次の判断を確定させる**:

- **案2(batch pump protocol: flat int32 + offsets + 1 metadata dict)は、
  同じ隠れたコストを狙っている。同じく wall を動かさない。** —— 実装しない。
- 配列転送そのものは残す。wall には効かないが、Ray object store に載る量が
  1 round あたり 4.92 MB → 1.32 MB になる。これは速度ではなく圧の問題。

### 計器は容疑を晴らした

対の control は profiler 拡張より前の run で、duty cycle 行も verdict 行も
持たない。つまり候補側だけが activity census を積んでいる。**census の実測は
enter+exit で 1.85 µs、208 バッチの run 全体で 0.028 s** —— 6.5% を説明できない。

残る +2〜6% は 25〜55 バッチという薄さと、共有機の run 間変動の範囲。
**この対が否定できるのは「大きな勝ち」であって、小さな差の符号ではない。**

### 訂正: `cpu-glue` を見ろと言ったのは誤り

`cpu_glue = preproc + decode + envstep`。**submit の経路は入っていない。**
pickle は `gen` の中で起きるので、この変更で `cpu-glue` は(比率として)
むしろ上がる。判定に使えない数字を 3 回勧めた。

### 保存則、5 回目

| 変更 | 動いた数字 | 動かなかった数字 |
|---|---|---|
| 配列転送(1b889d8 + ca4a26b) | driver 938 → 12 ms/turn | wall、EMPTY、PARTIAL |

**総和を動かした変更は依然として 2 つだけ** —— retriever の一括化と
`.tolist()` の除去。どちらも「driver の Python 仕事か外部の待ちを実際に消した」。
今回の配列化は driver の仕事を消したが、**その仕事はすでに隠れていた** ——
消しても総和は動かない。隠れていないものを探す必要がある。

### 同じ表から出たもう 1 つの結果: pump 単独の効果(初測定)

| | PUMP | CORE | STEPS | DEPTH | batches | wall | ms/row | score |
|---|---|---|---|---|---:|---:|---:|---:|
| `eval_v0_ms4.log` | **off** | v0 | 4 | 3 | 208 | 4447.5s | 84 | — |
| `eval_pump_ms4.log` | **ON** | v0 | 4 | 3 | 208 | 3847.2s | 73 | 0.3874 |

エンジン固定・両方完走・同一 208 バッチ。**wall −13.5%、ms/row −13.1%。**
これまでの 0.96 h 対 1.24 h は pump とエンジンを同時に変えていたので、
pump 単独の値はこれが初めて。

---

## 12. env reset は EMPTY の正体ではなかった(仮説の棄却)

`_reset_envs` は毎バッチの先頭、generate の前に走り、検証パスでは prefetch が
効かない。EMPTY はバッチあたり 1.50s / 1.40s で、`ENV_RESET_REPORT_S` の既定は
2 秒 —— **ちょうど探している大きさの reset は 1 行も出ない**。形が合いすぎていた。

累計を取った結果:

```
[env-reset] cumulative: 20 resets, 69.4s (3.47s each), 0 overlapped
[env-reset] cumulative: 25 resets, 69.5s (2.78s each), 0 overlapped
```

**reset 21〜25 の 5 回で 0.1 秒。1 回あたり 0.02 秒。**
69.4 秒は最初の数回 —— スロットごとの env manager 構築 —— の**一度きりの費用**で、
バッチあたりの費用ではない。208 バッチなら約 73 秒、run 全体の **1.9%**。
EMPTY は約 9%。**中身ではない。**

同時に、これは案3(幅 252→378)の機構の一つを否定する: widening は
バッチ数を 2/3 にするが、スロット数は変えないので、この 69 秒は消えない。

## 13. 当てずっぽうをやめて、インタプリタに訊く

census の残差は減らなかった:

```
while EMPTY the slots were in: gen 0.8, envstep 0.6, preproc 0.4
                               (of 4 slots; 2.1 in no tagged phase)
```

(`4 slots` は 3 スロット + 呼び出しスレッド。`val_pipeline` の
`dataload`/`prepare`/`scoring` は呼び出しスレッドで数えられるため。)

**2 回続けて外した。** 1 回目は record/assemble、2 回目は envreset ——
どちらも実測で 0.05 スロット未満だった。2 つの処方を生き延びた残差は、
3 つ目の当てずっぽうでは名前がつかない。

`sys._current_frames()` は当てずっぽうをしない。全スレッドが**いま実行している
フレーム**を返す。スレッドごとに 2 つ保持する:

- **repo 内の最深フレーム** —— 我々のどのコードの責任か
- **最内フレーム(全体)** —— そこで何をしているのか

この 2 つは**まさに興味深い場合に食い違う** —— 我々のコードが他人のコードの中で
待っているとき。実測で出た例:

```
val_pipeline.py:181 retire <- threading.py:327 wait
```

これは呼び出しスレッドが future を待って**ブロックしている** —— census には
原理的に見えないもので、しかも「driver が Python を回している」でもない。

コストは **14 スレッドで 1 回 49.8 µs、0.3 秒間隔・3800 秒の run 全体で 0.63 秒**
(profiler スレッド上、スロット上ではない)。`GPU_PROFILER_STACKS=0` で止まる ——
**止められない計器は、見ているものの原因として除外できない。**

出力は census が説明できていないとき(無タグ ≥ 0.33 スレッド)だけ出る:

```
[gpu-residency]    2.1 of those threads are in NO tagged phase.
                   Where they actually were, asked of the interpreter:
[gpu-residency]      1.20  rollout_loop.py:1408 _scatter_active_to_full
[gpu-residency]      0.60  val_pipeline.py:181 retire <- threading.py:327 wait
```

## 13bis. その計器の最初の実測 —— と、そこにあった単位のバグ

```
2.8 of those threads are in NO tagged phase. Where they actually were:
  126.58  (no repo frame) <- thread.py:81 _worker
   10.78  search.py:343 call <- threading.py:320 wait
    9.00  (no repo frame) <- threading.py:320 wait
    2.00  (no repo frame) <- threading.py:324 wait
    1.00  (no repo frame) <- runners.py:44 run
    0.98  val_pipeline.py:181 retire <- threading.py:320 wait
-> EMPTY is the DRIVER RUNNING PYTHON: 65% of one core while EMPTY vs 20% while busy
```

**`2.8` と `126.58` は単位が違う。** census はタグ付きスレッド(この run で 4 本)を
数え、frame は**プロセス内の全生存スレッド**を数える。pump・Ray・retriever pool・
3 スロットを動かしている機械には 140 本近いスレッドがあり、その大半は駐車中。
`thread.py:81 _worker` は `concurrent.futures` の**待機中ワーカー**で、来ない仕事を
待っているだけ。**pct をサンプル数で割っていたのと同じ種類の誤り**を、
新しい計器で作った。

そして実害が出た: 上位 6 件が駐車スレッドで埋まり、**1 コアの 65% を焼いている
当のスレッドがリストから溢れた。** 探しているものだけが表示されない。

### 直した形

- **repo 外のスレッドは 1 つの数に畳む** —— ここから手が出せないので、
  1 行ずつ並べる価値がない
- **repo 内のスレッドは RUNNING と PARKED に分けて並べる** —— 処方が正反対
- **打ち切りは分類の後** —— 生の件数で切ると、切った先に駐車スレッドしか残らない

```
2.0 of the tagged slots are in NO tagged phase. Mean live threads per EMPTY
sample, from the interpreter (138 outside this repo, parked or otherwise):
  -- RUNNING Python --
   0.67  rollout_loop.py:1408 _scatter_active_to_full
   0.33  rollout_loop.py:872 preprocess_single_sample
  -- PARKED, burning nothing --
   7.33  search.py:343 call <- threading.py:320 wait
   0.33  val_pipeline.py:181 retire <- threading.py:320 wait
```

### 最初の実測から、それでも読み取れたこと

`search.py:343` は `slot.done.wait()` —— **バッチ合体の follower 待ち**で、
10.78 本は 1 本の実 HTTP にぶら下がっている設計どおりの姿。CPU は焼いていない。
ただし **EMPTY の最中に retriever が in-flight である**ことは言っている。

そして census が retriever を「0.05 未満」として除外していた理由も分かった:
`envstep` タグは**呼び出しスレッド**に付いており、検索ツールが自前の pool
スレッドで実際の HTTP を行う分は、そのタグの外側にある。**census は構造的に
retriever のスレッドを数えていなかった。**
