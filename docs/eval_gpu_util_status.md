# 評価の GPU 利用率 —— 現状(2026-08-27)

`docs/eval_gpu_utilization.md` は時系列の実験ノート、
`docs/eval_performance_summary.md` は打ち手の結論集である。
**この文書は「いまどこに居るか」だけを、外れた仮説も含めて 1 枚にしたもの。**

対象: `trainer.val_only=True` の multitask 評価
(`examples/sft_trainer/eval_checkpoints.sh`、Qwen3-1.7B、A6000 × 3)。

---

## 1 行

**util 79.9%、到達可能な上限 91.1%、差 11.2 pt。そのうち銀行に入った分は 0。**

---

## 2. 数字

同じ推定器(wandb events stream、15 秒サンプル)、**同じ起動後 24 分窓**で揃えた。

| | depth 2 | depth 3 | + pump |
| --- | ---: | ---: | ---: |
| node util | 78.19% | 79.58% | **79.90%** |
| スパイク(node < 60%) | 18.4% | 16.9% | **15.8%** |
| ノード停止(< 5%) | 6.6% | 2.6% | 6.6% |
| **3 枚とも忙しいときの util** | — | 89.66% | **91.07%** |

**スパイクは消えていない。** 3 構成で 18.4% → 15.8%、util +1.7 pt。

---

## 3. 空きの内訳と、それぞれの状態

pump run、起動後 210 サンプル:

| 種類 | 割合 | util 換算 | 正体 | 状態 |
| --- | ---: | ---: | --- | :---: |
| **PARTIAL**(1〜2 枚) | 7.5% | 3.11 pt | collective generate の尾。rank が自分の chunk を終えて最遅 rank を待つ | **確定** |
| **EMPTY / DEEP** | 5.2% | 8.06 pt | **箱の外を待っている**(retriever か driver の Python か未確定) | **未確定** |
| **EMPTY / SHALLOW** | 3.3% | (上に含む) | 15 秒サンプルのエイリアシング。GPU は窓の大半で動いている | 確定 |
| **DUTY**(3 枚とも忙しい) | — | 約 9 pt | vLLM の 1 step ごとのホスト処理。4.5ms の GPU に 0.5〜0.8ms が露出 | 確定 |

### PARTIAL が確定している理由

**機構が予測した動きを 2 回とも見せた。** depth を上げても動かず(collective の
中なので届かない)、pump を入れると半減した(6.79 pt → 3.11 pt)。

### DEEP EMPTY で分かっていること

power < 130W が続く 11 サンプル(165 秒)を busy と突き合わせた:

| | DEEP EMPTY | busy |
| --- | ---: | ---: |
| GPU sm util | **1.5%** | 91.7% |
| GPU power | **87.9 W** | 288.4 W |
| **GPU メモリコントローラ** | **0.8%** | 78.4% |
| host CPU | **0.6%** | **0.6%** |
| スレッド数 | 868 | 856 |
| disk read | 0 | 0 |

**GPU は計算も転送もしていない。ホストも動いていない。スレッドは全部生きている。**
計算待ちでも帯域待ちでも I/O 待ちでもなく、**プロセス全体が箱の外を待っている形**。

**断定しない理由:** wandb の `system.cpu` は busy でも DEEP EMPTY でも 0.6% で、
**GIL を握った Python 1 本とソケット待ちを見分ける分解能が無い。**

---

## 4. 効いたもの、効かなかったもの

| 変更 | util | 判定 |
| --- | ---: | --- |
| **retriever の修正**(`:8000` / 窓 100ms / connect 5s) | **57.9% → 79.0%** | **唯一の大きな勝ち** |
| socket 上限(`TCP_USER_TIMEOUT`) | 79.0 → 79.1% | 効果なし(裾しか消さない) |
| `VAL_PIPELINE_DEPTH=3` | 78.2 → 79.6% | EMPTY は消えた、util は動かず |
| `ROLLOUT_ASYNC_GENERATE=1`(pump) | 79.6 → 79.9% | PARTIAL は半減、util は動かず |
| `ROLLOUT_GPU_MEM_UTIL=0.75` | — | 余白のみ(今のバッチでは効かない) |

**retriever を直して以降、util は 0.9 pt しか動いていない。**

---

## 5. 空きが保存される —— 3 回とも

| 変更 | 下がったもの | 上がったもの |
| --- | --- | --- |
| depth 3 | EMPTY 8.9 → 3.8% | PARTIAL 7.7 → 12.7% |
| pump | PARTIAL 6.79 → 3.11 pt | EMPTY 3.29 → **8.06 pt** |

**pump は確定した原因を機構どおり半減させ、3.68 pt を実際に取り戻し、
4.77 pt を EMPTY に渡した。** バケツ収支 −1.09 pt。
util が +0.32 に見えるのは基準値(3 枚とも忙しいときの util)が
89.66 → 91.07 に上がったぶんだけである。

**GPU 側を速くしても GPU 外の仕事は減らない。露出が増えるだけである。**

---

## 6. util 以外で得たもの

| | depth 2 完走 | pump run 完走 |
| --- | ---: | ---: |
| **wall** | 1.26 h | **0.96 h** |
| success_rate | 0.3869 | 0.3875 |
| search | 0.3523 | 0.3571 |
| alfworld | 3.649 | 3.634 |
| webshop | 5.938 | 5.696 |

**24% 短い。スコアはほぼ動いていない。**
ただし depth・`gpu_memory_utilization`・timeout・pump が同時に変わっているので、
**pump 単体の効果ではない**(単一変数になっていない)。

**util を目的にすると「効果なし」、wall を目的にすると「今回いちばん効いた」。
同じ run が両方の答えを出している。**

---

## 7. 計器 —— 3 つあり、3 つとも別の盲点を持っていた

| 計器 | 見えるもの | 見えないもの |
| --- | --- | --- |
| `[val-pipeline]` の `NOTHING running` | slot が batch の中にいるか | **env.step で止まった slot を「実行中」と数える。** 285 秒のノード停止を 0.1% と報告した |
| `genGPU%` | `generate` の中の util | generate と generate の間 |
| wandb のチャート | スパイクが有ること | 15 秒グリッドが PARTIAL を水増し(8.0% を 14.6% と読む)。0 枚と 1 枚を区別しない。token を見ない |
| **`[gpu-residency]`**(今回追加) | 0.3 秒間隔で「何枚に仕事があったか」、EMPTY と PARTIAL の分離、per-GPU の偏り、**EMPTY の理由** | — |

**この盲点が判断を狂わせた実例:** async(pump)の的を「0.4%」と見積もったのは
`[val-pipeline]` の数字から。device に聞けば PARTIAL だけで 6.79 pt だった。

---

## 7bis. 外部レビューで確認できた 3 点(2026-08-27)

| 主張 | 確認 |
| --- | :---: |
| `environment.yml` は **vllm 0.11.0** を固定しているが、動いているのは **0.8.5** | ✅ ただし下記 |
| pump も最後に `[future.result(...) for future in futures]` で **batch 全体を待つ** | **✅ 事実**(`rollout_loop.py:440`) |
| `_run_full_preprocess` は active row を **Python で 1 件ずつ** | **✅ 事実**(dict 内包) |

### 1 つめの訂正 —— `environment.yml` はこの機械の記録ではない

「`environment.yml` が指している版を入れるだけ」と一度書いたが、**言い過ぎである。**
中身を読むと:

```
name: verl-agent                          ← 動いているのは sdar-multitask
channels: ... sankuai.com ...             ← 社内ミラー
torch==2.8.0  transformers==4.57.3  vllm==0.11.0  xformers==0.0.32.post1
```

**別名の環境を、別のチャンネルから作る vendored な成果物**であって、
「本来こうだった環境がずれた記録」ではない。**ここから復元してはいけない。**

### それでも 0.11.0 に上げる価値はある(そして代償がある)

`async_scheduling`(V1 が scheduler を forward に被せる)は **0.10.2 より前に存在しない**。
3 枚とも忙しいのに 91% しか出ない残り約 9 pt に対する、**唯一残った打ち手**である。

**代償は 2 つ:**

1. **torch 2.6 → 2.8。** vLLM 0.8.5 は torch 2.6、0.11.0 は 2.8 を要求する。
   2.6 に対してビルドされたもの(flash-attn、flashinfer、xformers、apex、
   自前の CUDA 拡張)は**インストール時には落ちず、import か最初の kernel で落ちる。**
2. **kernel と reduction 順が変わる。** 0.8.5 で測ったスコアは**全部**比較対象でなくなる。

だから**現環境は触らず、クローンに入れる**:

```bash
bash examples/sft_trainer/clone_env_vllm011.sh sdar-multitask sdar-vllm011
```

clone → `pip install vllm==0.11.0` → **pip が何を変えたかの diff** →
torch / vllm / flash_attn / flashinfer / xformers の import 確認 →
`LLM()` に `async_scheduling` があるかの確認、まで一度に出る。
落ちたら元の環境は無傷なので、クローンを捨てるだけで済む。

### 私の cpu_pct 判定は弱かった —— レビューの指摘が正しい

`cpu_pct >= 60` が証明するのは「driver が CPU を使っている」までで、
**GIL を握っていることの証明にはならない**(Rust の tokenizer は GIL を離すし、
native BLAS は複数コアぶん出る)。**判定を撤回した。**

代わりに **activity census** を入れた。`push_phase` のスタックは
スレッドローカルでないので slot 3 本では壊れる —— **数える**方式にした:

```
[gpu-residency] while EMPTY the slots were in: envstep 2.7, preproc 0.2
                -> EMPTY is the ENVIRONMENT (retriever round trip) -- off the box
[gpu-residency] driver CPU 4% of one core while EMPTY vs 160% while busy
                (blocked during EMPTY)
```

**cpu は報告するが、原因は名指ししない。** 原因は census が言う ——
推論ではなく直接観測だからである。どれも支配的でなければ
`no single activity dominates` と言って**名指しを拒む**。

---

## 7ter. 0.8.5 のまま engine 内部に手がある —— V0 の multi-step

「0.8.5 には手が無い」も**不正確だった**。正しくは
**「0.8.5 の V1 経路に `async_scheduling` が無い」**である。
同じパッケージに V0 が同梱されていて、`VLLM_USE_V1=0` で切り替わり、
V0 には `num_scheduler_steps`(multi-step scheduling)がある。

`num_scheduler_steps=4` なら 1 回の scheduling で最大 4 回 forward を進めるので、
**decode step ごとに露出している host scheduling が約 1/4 になる。**
狙うのは 3 枚とも忙しいときの 91.07% より上、約 8.9 pt である。

### コード側の前提は満たされている(確認済み)

| | |
| --- | :---: |
| `engine_kwargs.vllm` がそのまま `LLM(...)` へ届く | ✅ |
| pump の sampling params に `RequestOutputKind.FINAL_ONLY` が載っている | ✅ `vllm_rollout_spmd.py:110` |
| この経路が `VLLM_USE_V1` を強制していない | ✅ |
| `enable_chunked_prefill=False` / PP なし / spec decoding なし | ✅ multi-step の非互換条件に当たらない |

### 確認できない前提が 2 つある —— だから guard を入れた

この rollout は V0 が serve できるか分からないものを 2 つ渡している:

- `distributed_executor_backend="external_launcher"`
- `enable_sleep_mode=True`

**`VLLM_USE_V1=0` は要求であって結果ではない。** vLLM は serve できない構成では
**落ちずに V1 へ戻る。** そのとき `num_scheduler_steps` は
**受け取られて無視される** —— エラーより悪い。

これは既に 2 回まるごと run を潰した失敗と同じ形である
(pool が断って blocking path に落ちた、retriever が 422 で 1 クエリずつに落ちた)。
どちらも完走し、どちらも普通に見え、どちらも**対照を 2 回測っていた**。

```bash
ROLLOUT_REQUIRE_CORE=v0   # 違う core が組まれたら build で落ちる
```

### A → B → C → D(core の変更を効果から分離する)

| | core | steps | 目的 |
| --- | --- | ---: | --- |
| A | V1 | 1 | いまの基準 |
| B | V0 | 1 | **V1→V0 それ自体の影響**(V0 が遅い可能性がある) |
| C | V0 | 4 | multi-step の効果 |
| D | V0 | 8 | 追加効果と latency |

```bash
# B
VLLM_USE_V1=0 ROLLOUT_REQUIRE_CORE=v0 \
bash examples/sft_trainer/eval_checkpoints.sh <step>

# C
VLLM_USE_V1=0 ROLLOUT_REQUIRE_CORE=v0 \
bash examples/sft_trainer/eval_checkpoints.sh <step> -- \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.num_scheduler_steps=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.multi_step_stream_outputs=false
```

**B を飛ばして C から測ってはいけない。** V0 が V1 より遅ければ、
multi-step の利得と core の損失が同じ数字の中で相殺される。

### multi-step で必ず確かめること

driver が EOS と完了を知るのが最大数 step 遅れる。**latency なら問題ないが、
生成が変わるなら別である。**

- `[val-hash]` が A と一致するか(greedy なので一致すべき)
- 各 task の score
- pump の Future が全部完了するか(`[pump] N requests, N finished, 0 timed out`)
- `max_tokens` の超過。verl は `response_length` に切り詰めるので、
  **超過しても黙って消える** —— hash で見るしかない

pump は vLLM の内部 API(`add_request` / `step` / `abort_request`)を直接叩いている。
**`LLM.generate()` が動くことは、この経路が動く証明にならない。**

### 実際に走らせた結果(2026-08-27 12:27) —— V0 は組めた、weight sync が落ちた

```
[rollout-engine] vllm 0.8.5, core=v0; overlap knobs: async_scheduling=absent,
  num_scheduler_steps=4 (set here), disable_async_output_proc=<default> (available),
  enable_chunked_prefill=False (set here), max_num_seqs=1024 (set here),
  enforce_eager=False (set here)
[rollout-engine] ROLLOUT_REQUIRE_CORE=v0: confirmed.
```

**懸念していた 2 つは杞憂だった。** `external_launcher` と `enable_sleep_mode` は
V0 でも通り、multi-step も受け付けられた。guard も期待どおり確認を出した。

落ちたのは**別の本物のバグ**である:

```
AttributeError: 'MultiStepModelRunner' object has no attribute 'model'
  verl/workers/sharding_manager/fsdp_vllm.py:263 in update_params
```

`num_scheduler_steps>1` は model runner を **`MultiStepModelRunner` で包む**。
包みは `.model` を転送せず、実物は `_base_model_runner` にいる。
FSDP → vLLM の重み同期がそこで死ぬ。

**engine は組め、config は通り、checkpoint も読み終えてから落ちる** ——
数分入ってから、multi-step とも weight sync とも書かれていない例外で。
`unwrap_model_runner()` で包みを剥がすようにした。

---

## 7quater. DEEP EMPTY に名前が付いた —— **driver の Python** だった

V0 + `num_scheduler_steps=4` の run のログ:

```
[gpu-residency] while EMPTY the slots were in: envstep 1.1, preproc 0.6, gen 0.5
                -> EMPTY is the ENVIRONMENT (retriever round trip) -- off the box
[gpu-residency] driver CPU 82% of one core while EMPTY vs 15% while busy
                (running during EMPTY)
```

**この 2 行は矛盾していて、判定ロジックのほうが間違っていた。**

- **ソケットを待つプロセスは CPU を焼かない。** EMPTY で 82%、busy で 15% ——
  **カードが空のときのほうが driver は忙しい。** retriever ではない。
- `envstep 1.1` は **3 slot 中の 1.1**、つまり 1/3 である。閾値を
  「タグ済み合計に対する割合」で見ていたので、支配していないものを名指しした。
- 合計 2.2 なので、**0.8 slot 分がどのタグにも入っていない。**

### 直したもの —— 2 つの読みが一致したものだけを判定にする

| 計器 | 答えられること | 答えられないこと |
| --- | --- | --- |
| `cpu_pct` | **働いていたか**(待ちは CPU を焼かない) | どこで |
| activity census | **どこで**(1 phase が slot の過半を占めるとき) | 待ちか実行か |

```
[gpu-residency] while EMPTY the slots were in: envstep 1.1, preproc 0.6, gen 0.5
                (of 3 slots; 0.8 in no tagged phase)
[gpu-residency] -> EMPTY is the DRIVER RUNNING PYTHON: 82% of one core while
                EMPTY against 15% while busy, and no single phase dominates
```

支配の判定は **slot 数に対して**行う。両者が食い違えば **`these disagree`** と言い、
CPU が 20〜60% の中間帯なら **`UNRESOLVED`** と言って**決めない**。

> **判定ロジックを 2 回振った。** 1 回目は `cpu >= 60` を「GIL を握っている証明」に
> した(撤回)。2 回目は census だけで原因を名指しした(撤回)。
> **どちらも単独では答えられない質問に、単独で答えさせていた。**

### だから分岐は「driver の Python」側である

§8 の分岐表のうち、**`EMPTY is the DRIVER's own Python` の側**が現実である:

1. **batch tokenizer**(全 prompt を fast tokenizer へ 1 回)
2. **Search の `step_batch()`**(252 スレッドへの分解自体をやめる)
3. **pump coordinator の別プロセス化**
4. **タグの無い 0.8 slot 分を特定する** —— scatter/union、val-pipeline の
   採点とデータロード、DataProto の結合。**まだどこにも計上されていない**

retriever の複製や Flat index の置き換えは、**この証拠からは正当化されない。**

---

## 7quinquies. multi-step の取り分は出し切った —— そして計画の 1 項が落ちる

| | duty cycle | gap |
| --- | ---: | ---: |
| V1 + pump | 91.07% | 8.93 pt |
| **V0 + multi-step 4** | **92.55%** | **7.45 pt** |

**回収 1.48 pt、gap の 17%。** 隠蔽できていれば duty は 100% 近くになる。

gap を scheduler 由来 S とそれ以外 R に分ける。`num_scheduler_steps=4` は S を 1/4 にする:

```
S + R    = 8.93     (steps=1)
S/4 + R  = 7.45     (steps=4)
------------------------------
S = 1.97,  R = 6.96
```

**scheduler は 9 pt のうち 2 pt しかない。** `steps=8` は `S/8 + R = 7.21`、
**+0.24 pt** —— 走らせる価値はない。

> **これは計画の 1 項を落とす。** `async_scheduling`(vLLM 0.11)が隠すのは
> **同じ S** である。**したがって 0.11 への更新も ~2 pt** であって、
> かつて書いた ~8.93 pt ではない。kernel と reduction 順が変わって
> スコアの基準が全部リセットされる代償に見合わない。**保留。**

**留保 2 つ。** (1) B(V0 単体)が無いので S=1.97 は「V0 == V1」を仮定している。
(2) NVML の 1 サンプルが turn の境界をまたぐと busy サンプルの読みも下がるので、
**7.45 pt は engine 内部 gap の上限**である。

`disable_async_output_proc` は既定 False(= async output 処理は既に ON)なので、
そこは取ってある。残る R は step ごとの output 処理、CUDA graph の replay、
model forward 内部の隙間 —— **`engine_kwargs` から触れるつまみは無い。**

---

## 7sexies. 「どのタグにも入っていない 0.8」を塞いだ

EMPTY のとき `envstep 1.1, preproc 0.6, gen 0.5` = 2.2、**3 slot に対して 0.8 が
どこにも計上されていなかった。** batch tokenizer を作る前にここを見ないと、
preproc(0.6)を最適化してより大きい塊を外す。

**正体は「slot スレッドしかタグしていなかった」こと。** 中身:

| 追加したタグ | どこ |
| --- | --- |
| `dataload` / `prepare` / `scoring` | **呼び出しスレッド**(val_pipeline)。ロード・準備・採点の間ずっと GIL を握る |
| `glue` | pad / unpad / `DataProto.union` —— phase の**あいだ** |
| `preproc`(範囲拡大) | `batch.pop` と key の帳簿。タグが `preprocess_batch` で閉じていた |

判定行も phase を名前で言うようにした:

```
-> EMPTY is the DRIVER RUNNING PYTHON: 82% of one core while EMPTY against
   15% while busy, mostly in reward accumulation on the calling thread
```

**次の run が 0.8 の中身を名指しする。** そこで初めて 4(batch tokenizer)と
5(`step_batch`)のどちらが本命か決まる。

---

## 7septies. util を上げた run が、いちばん遅い run だった

`sft-multitask-eval-20260827-124410`(V0 + multi-step 4)が完走した。

| 完走 run | 構成 | node util | duty | **wall** |
| --- | --- | ---: | ---: | ---: |
| `...-201115` | V1, depth 2, pump なし | 81.2% | — | 1.26 h |
| `...-092241` | V1, depth 3, **pump** | 79.90% | 91.07% | **0.96 h** |
| `...-124410` | **V0 + ms4**, depth 3, pump なし | **83.21%** | **92.55%** | 1.24 h |

**util が最も高い run が、pump run より 29% 遅い。**
そして **util が最も低い run が、最も速い。**

スコアは動いていない(V0 は kernel が変わるので確認が要った):

| | pump (V1) | V0 + ms4 |
| --- | ---: | ---: |
| success_rate | 0.3875 | 0.3865 |
| search(greedy) | 0.3571 | 0.3560 |
| alfworld(temp 0.4) | 3.634 | 3.770 |
| webshop(temp 0.4) | 5.696 | 5.528 |

greedy の search が 0.3571 → 0.3560 とほぼ不動なので、**V0 への切替は生成を
壊していない。** alfworld と webshop は temperature 0.4 のサンプリングで、
この幅は run 間のばらつきの範囲である。

### 交絡が 2 つある

124410 と 092241 の差は **(V0 + multi-step)** と **(pump の有無)** の 2 つ。
124410 は pump 無しである。pump 無し同士で比べると:

```
201115 (V1, depth 2, mem 0.6) 1.26 h
124410 (V0 + ms4, depth 3, mem 0.75) 1.24 h
```

**ほぼ同じ。** つまり **0.96 h を買ったのは pump** であって、
multi-step でも V0 でも depth でもない。

### これが「util を目的関数にするな」の最も明確な実例である

| 目的 | 勝者 |
| --- | --- |
| **node util** | V0 + multi-step(83.2%) |
| **wall** | **pump(0.96 h)** |

同じ 3 台、同じ checkpoint、同じスコア。**util で選ぶと 29% 遅い設定を選ぶ。**

§5 罠 1「占有率を目的関数にした」は、この文書に 3 回踏んだと書いてある。
**これはその 4 回目を防ぐための実測である。**

### 次に測るべきは pump + multi-step である

両者は別の層を触る(pump は PARTIAL、multi-step は duty)。まだ**一度も
一緒に走っていない**:

```bash
VLLM_USE_V1=0 ROLLOUT_REQUIRE_CORE=v0 \
ROLLOUT_ASYNC_GENERATE=1 ROLLOUT_ASYNC_REQUIRE=1 \
bash examples/sft_trainer/eval_checkpoints.sh <step> -- \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.num_scheduler_steps=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.multi_step_stream_outputs=false
```

**判定は wall で。** util は上がって当然で、それでは何も決まらない。

---

## 7octies. EMPTY の半分は、まだどこにも計上されていなかった

タグを 3 つ足したあとの実測(V0 + ms4、pump なし):

```
[gpu-residency] while EMPTY the slots were in: gen 0.8, preproc 0.5, scoring 0.1
                (of 3 slots; 1.5 in no tagged phase)
[gpu-residency] -> EMPTY is the DRIVER RUNNING PYTHON: 80% of one core while
                EMPTY against 11% while busy, and no single phase dominates
```

| 読めること | |
| --- | --- |
| **`envstep` が消えた**(0.05 未満) | **retriever は原因ではない。確定。** |
| 80%/11% (前回 82%/15%) | **driver が Python を回している。2 run で一致。** |
| **`1.5 in no tagged phase`** | **半分がまだ計器の外** |
| `gen 0.8` | generate 呼び出しの中にいるのに**カードは空** —— Ray RPC と DataProto の直列化 |

### 塞いだもの

| タグ | 何 |
| --- | --- |
| **`record`** | turn の後半すべて。`to_list_of_dict(batch)` が**毎ターン 252 行の DataProto を dict に展開**している |
| **`assemble`** | `gather_rollout_data` —— 最後の generate が返ったあとに全 turn × 全 trajectory を pad する。**その間カードは全部空** |

### `no single phase dominates` が示唆すること

タグを 6 つに増やしても過半を取る phase が無い。**driver は 1 箇所で詰まっているのではなく、
ほぼ 1 コア飽和(= GIL 律速)で全体的に忙しい。**

**単一の phase を速くしても、別の phase が同じだけ露出する** —— §5 の「空きの保存」と同じ形。
`record` と `assemble` が名指しされれば話は変わるが、**それは次の run が言う。**

---

## 7novies. tail turn —— 生成 wall の半分が、token の 2 割に使われている

同じログの turn table。search batch の 1 本(batch#9):

| turn | active | gen | genTok |
| ---: | ---: | ---: | ---: |
| 0 | 252 | 1.26 s | 4,879 |
| 1 | 252 | 25.54 s | 52,401 |
| **2** | **58** | **13.70 s** | 11,273 |
| **3** | **12** | **10.82 s** | 2,514 |

**turn 2+3 = 生成 wall の 48%、token の 19%。** batch#8 では **59% / 25%**。

turn 3 は **12 行で 2,514 token に 10.8 秒** —— 1 step あたり 53 ms。
`genGPU%` は 78〜94 なので、**GPU は「忙しい」と読まれる。忙しく、かつ無駄である。**

**これは util では見えない。** duty cycle も EMPTY も PARTIAL も、この 10 秒を正常と報告する。

これを直すのは **trajectory 単位の進行**(計画 6)である。turn 1 で終わった 240 行が
turn 3 の 12 行を待つ理由はない。**preproc は turn あたり 0.14〜1.3 秒**なので、
batch tokenizer(計画 4)はここには桁が届かない。

---

## 8. 次の 1 手 —— これだけ

**残り 5.2% の DEEP EMPTY に名前が付くまで、GPU 側の変更はしない。**
名前の付いていない相手に対して既に 3 回同じことをやった。

`6f563cb` 以降、ログが名指しする:

```bash
grep 'EMPTY is' /tmp/<log>
```

| 出力 | 意味 | 打ち手 |
| --- | --- | --- |
| `EMPTY is the ENVIRONMENT` | retriever 往復 | server 側の dynamic batching、replica、`step_batch` で 252 スレッドの分解自体をやめる |
| `EMPTY is the DRIVER's own Python` | tokenize / detokenize | batch tokenizer(全 prompt を fast tokenizer へ 1 回)、pump coordinator の別プロセス化 |
| `no single activity dominates` | slot の**間**にある | preproc と envstep の境目。turn barrier そのもの |

あわせて `[rollout-engine] vllm ..., core=...` も出る。**0.11.0/V1 なら
`async_scheduling` がその場で使える**(§7bis)。

**いまの証拠(DEEP EMPTY で host CPU が busy と同じ 0.6%、disk 0、
メモリコントローラ 0.8%)は後者に傾いている。** もしそうなら、
**このリポジトリのコードでは util はこれ以上上がらない。**

---

## 9. 天井

```
到達可能な上限(3 枚とも忙しいときの util)   91.07%
いま                                        79.90%
差                                          11.2 pt   ← 全部が賞金
そのうち銀行に入った分                       0 pt
```

91.07% より上は vLLM の 1 step ごとのホスト処理で、**vllm 0.8.5 には回せる
つまみが無い**(`async_scheduling` は 0.10.2 以降。`num_scheduler_steps` と
`disable_async_output_proc` は V0 専用)。更新は可能だが別の実験である。

---

## 10. 自分が間違えた点(記録)

| | 間違い | 訂正 |
| --- | --- | --- |
| 1 | 40 秒の停止を「connect の中」と帰属 | ログの「2 回」が否定。詰まった socket、connect の先 |
| 2 | async の的を「0.4%」と見積もり | 壊れた計器の数字。実測 6.79 pt |
| 3 | 「88〜90% が天井」 | 物理ではない。engine のホスト処理で、隠す機構は存在する |
| 4 | `grep TOTAL \| tail -3` を切り分けの根拠に | 別の batch を比べていた(promptTok が 6 倍違う)。§5 罠 3 と同じ |
| 5 | util を目的関数に最適化(3 回目) | 効いたかは wall と ms/row でしか言えない |
| 6 | `cpu_pct >= 60` を「GIL を握っている」の証明に | 証明できるのは「CPU を使っている」まで。Rust tokenizer は GIL を離す。**activity census に置き換えた**(§7bis) |
| 7 | 動いている vLLM の版を確かめずに「0.8.5 には手が無い」 | 版は確かめるべきだった。ただし §7bis のとおり `environment.yml` は**別環境の vendored な成果物**で、「入っていないだけ」も言い過ぎだった —— **2 段階で間違えた** |
