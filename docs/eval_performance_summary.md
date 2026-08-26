# 評価の速度と GPU 利用率 —— 現状・打ち手・計測の作法

`docs/eval_gpu_utilization.md` は時系列の実験ノートで、外れた仮説と訂正が
そのまま残してある。**この文書はそれを読まなくても済むように、結論だけを
現在形で書いたもの**である。数字はすべて実測で、推測には「推測」と書いてある。

対象は `trainer.val_only=True` の multitask 評価
(`examples/sft_trainer/eval_checkpoints.sh`、Qwen3-1.7B、A6000 × 3)。

---

## 1. いまの数字

| | 出発点 | いま |
| --- | ---: | ---: |
| 評価の wall | **2.85 h** | **~0.85 h**(見込み、`ms/row` からの外挿) |
| ms/row | 118 | **57** |
| 機械の GPU util(起動後) | — | **77.8%** |
| generate 中の util(`genGPU%`) | — | 85〜90% |

**3.4 倍。** ただし util は 65〜78% で、100% には遠い。その理由は §4。

---

## 2. 採用したもの

| 打ち手 | 効果(実測) | 既定 |
| --- | --- | --- |
| **rollout session**(vLLM を turn ごとに寝起きさせない) | 13% の wall | ON |
| **session の hoist**(batch ごとの wake/sleep も止める) | 10.4% の wall | ON |
| **retriever のバッチ化**(1 turn 1 リクエスト) | envstep **10.60 → 1.00 s/batch** | ON —— ただし**サーバがリストを受けなければ黙って無効になる**(§2.1) |
| **retriever のバージョンずれ検出 + 自動再試行** | envstep 10.8 → 1.0 s(事故復旧) | ON |
| **`VAL_PIPELINE_DEPTH=2`** | s/batch **17.0 → 14.2** | ON |
| **search の val batch 252 行** | ms/row **94.8 → 78.0** | OFF(§6) |
| **`VAL_PIPELINE_DEPTH=3` + generate 合流** | ms/row **74 → 57** | OFF(§6) |
| 採点時の無駄な detokenize 削除 | 数%(504 回 → 1 回/batch) | ON |
| **ログ用テーブルの全行 decode 停止** | `log_val_generations=0` なのに全 52k 行の prompt+response を decode していた | ON |
| **raw_prompt の二重 tokenize 停止** | 同一文字列を 2 回 encode(turn ごと 252 回)。最初の 8 呼び出しで新旧一致を自己検証し、不一致なら永久フォールバック | ON |


### 2.1 バッチ化は、サーバ次第で黙って無効になる

`retrieval_server.py` の `QueryRequest.query` は `Union[str, List[str]]` だが、
**それが入る前の版が動いている URL に当たると 422 が返る。** クライアントは
1 クエリずつに永久フォールバックし、300 秒ごとに再プローブしてまた断られる
(`search.py:308-319`)。

**何も失敗しない。** run は完走し、スコアも普通に見え、envstep だけが
70 秒の batch のうち 4 秒ではなく **40 秒**になる。

実際に起きた:

| run | retriever | util | wall |
| --- | --- | ---: | ---: |
| `...-230457` | **:8000**(新しい) | **77.88%** | **1.20 h** |
| `...-102303` | :8001(37 日前のプロセス) | 57.91% | 2.23 h |
| `...-131419` | :8001 | 55.70% | 2.11 h |
| `...-174359` | :8001 | 51.12% | — |

**77.88% を出した唯一の run が、唯一 :8000 を使っていた run である。** この差は
最初「幅 252 のおかげ」と読み、次に「checkpoint の違い」と読んだ。どちらでもなく
**ポート**だった。`API Request Error: 422` はログに最初から出ていて、
「件数が増えていないので一過性」と判断したのが誤り —— 件数ではなく、
**その 422 が何を無効化しているか**が問題だった。

run script が起動前に probe して**落ちる**ようにした(`SEARCH_PREFLIGHT=0` で無効化、
URL は `SEARCH_URL` で差し替え)。**URL を変えるのではなく、そのポートの
retriever を今の `retrieval_server.py` で立て直すのが本筋** —— 古い版が
動いている限り、どのポートを指しても同じことが起きる。`ROLLOUT_ASYNC_REQUIRE` と同じ理屈で、黙って
遅い経路に落ちるのを構造的に止める。

### 効いた理由が直感と違うもの

* **depth 3 単独は効果ゼロ**(75 ms/row、depth 2 と同じ)。slot を足しても
  generate 呼び出しは worker group 上で直列のままである。**depth 3 は合流に
  相手を供給するためだけの前提条件**で、23% は全部合流の効果である。
* **合流は depth 2 では一度も発火しない。** 待ち行列が 1 を超えないので相手が
  いない。「効かなかった」ではなく「実行されなかった」。
* **幅を 126 → 252 にしても util は動かない**(86.3% → 86.0%)。decode は帯域
  律速で、席を増やしても 1 step の時間は変わらない。**動いたのは throughput
  だけ**で、これが「util は目的関数ではない」の一例目である。

---

## 3. 試して捨てたもの

| | 結果 |
| --- | --- |
| **FlashInfer** | **4.3% 遅い。** util は 77 → 88% に上がった(§5 の罠) |
| **`gpu_memory_utilization=0.85`** | **±0%。** KV cache を 1.7 倍にしても preemption の損失は無かった。34.64x という同時実行上限は 4,608 tokens/request の話で、実際の系列長がそこに遠く及ばない |
| **`enable_chunked_prefill=True` + `max_num_batched_tokens=32768`** | **±0%。** 減るのは prefill のステップ境界 6 個ぶん(≈60 ms / 16 s)で、走らせる前に計算すれば分かった |
| `data.val_batch_size` を上げて幅を広げる | **不可。** env manager が同じ値で size され、alfworld の worker 数が変わって**採点が動く**。task ごとの size が必要だった |
| `raw_prompt_ids` だけ送る / 出力組み立ての最適化 / `to_cpu` の非同期化 | **全部的外れ。** verl が engine の周りでしていることの合計は **14 ミリ秒**(§4) |

---

## 4. util が 100% にならない理由 —— 22% の分解

起動後の util 77.8%、空き 22.2%。GPU が空くのは 2 通りしかない。

**これは retriever のバッチ化が効いている前提の分解である**(§2.1)。効いて
いない run では envstep だけで batch の 57% を占め、この表は当てはまらない。
使う前に `[rollout-turn-timing]` の SHARE 行で確かめること。

```
util = (誰かが generate に入っている割合) × genGPU%
0.778 = G/W × 0.87   →   G/W = 0.894
```

| 原因 | wall に占める空き | 中身 |
| --- | ---: | --- |
| **(a) 全 slot が GPU の外** | **10.6%** | preproc 3.9 s + envstep 3.0 s + decode(batch あたり)。**preproc がその 58%** |
| **(b) `vllm.generate` の中** | **11.6%** | 呼び出し固定費 ≈0.6 s/回 + decode 1 step ごとの host 処理 |
| (c) pipeline が空 | 0.4% | 無視できる |

### (a) はなぜ「全 slot 同時」になるのか

slot を増やすと **GPU の外の処理そのものが遅くなる**:

| | preproc | envstep |
| --- | ---: | ---: |
| depth 2(252 行) | 2.7 s | 2.0 s |
| depth 3(252 行、46 batch 平均) | **3.9 s** | **3.0 s** |

**1.5 倍。** 独立に走っているなら伸びない。伸びるのは取り合っているからで、
preproc は Python の tokenize なので **GIL**、envstep は 3 倍のクエリを同じ
retriever に投げるので**そちら**である(どちらも推測、プロファイル未取得)。
取り合って一緒に遅くなるので位相がずれず、揃って GPU の外にいる。

これは **depth 3 単独が効果ゼロだった理由**でもある。

### (b) は verl の外にある

worker 側で脚を分解した結果(300 呼び出しの平均):

```
[rollout-phases] build_inputs 0.000  engine 4.374  assemble 0.005  total 4.379
[gen-phases]     to_device 0.002  preprocess 0.002  generate 4.381
                 postprocess 0.001  to_cpu 0.001  total 4.388
```

**verl が engine の周りでしていることは 14 ミリ秒。** sharding manager も
Ray の往復も detokenize も全部ゼロ。**0.6 s の固定費は `vllm.generate()` の
内側にあり、verl 側からはこれ以上割れない。**

step ごとの host 処理は **1.7B というモデルの小ささ**が直接の原因である
(GPU 1 step が重み 3.44 GB の読み出しで 4〜5 ms、そこに host 0.5〜1 ms)。
7B なら同じ host 処理が 2% に埋もれる。

---

## 4bis. 「学習は 100% なのに検証は 87%」は比較になっていない

この arm の**学習は 1 トークンも生成しない。**

```bash
# run_multitask_sft_qwen3.sh:504
# This arm never generates (test_freq=-1; validation is a separate process), so
# the vLLM rollout is a passenger that costs startup time ...
export SKIP_ROLLOUT_BUILD=${SKIP_ROLLOUT_BUILD:-1}
```

`SKIP_ROLLOUT_BUILD=1` が既定で、**学習プロセスには vLLM が建たない**。教師軌跡を
プールから読んで forward/backward するだけの offline SFT で、`test_freq=-1` なので
途中の検証もしない。

| | 学習 | 検証 |
| --- | --- | --- |
| decode | **ゼロ** | 100% |
| 環境(retriever / シミュレータ) | **無し** | あり |
| vLLM | **プロセスに存在しない** | 本体 |

**同じ仕事を効率違いでやっているのではなく、別のコードを走らせている。**
100% と 87% を並べるのは、forward/backward だけの run と生成だけの run を
並べているだけである。

GRPO や PPO のように学習が rollout を持つ場合、**その rollout 区間の util は
検証と同じになる** —— `multi_turn_loop` という同一のコードだからである。学習の
100% は「rollout フェーズが存在しない」ことの帰結であって、生成が速いことの
証明ではない。

したがって **「検証の util を学習に合わせる」は目標として成立しない。**
意味のある比較は、生成という処理形態の天井(§8 の 95% 前後)に対して検証が
どこにいるか、だけである。

### 生成の 1 step が短い理由(天井そのものの説明)

| | 学習の 1 micro-batch | decode の 1 step |
| --- | ---: | ---: |
| 処理トークン(GPU あたり) | ~2,600 | **84**(252 ÷ 3) |
| **重みの読み出し** | **3.44 GB** | **3.44 GB(同じ)** |
| 演算強度 | 7,700 FLOPs/byte | **83 FLOPs/byte** |
| 律速 | 計算 | **帯域** |
| 時間 | ~1,010 ms(stall watch の median) | ~4.5 ms |

同じ重み読み出しを、学習は 2,600 トークンで償却し decode は 84 で償却する。
1 step で出せるのが「同時に走る系列の数」だけなのは**因果の制約**で、実装では
消せない。1 トークンあたりでは decode の方が 7 倍安い(0.054 ms 対 0.39 ms)
—— 遅いのではなく、**step が短いせいで step ごとの host 処理が相対的に大きい。**

A6000 の計算/帯域比は約 **100 FLOPs/byte**、decode の演算強度は**系列数そのもの**
(FLOPs = 2·P·B、bytes = 2·P)。**GPU あたり 100 系列が境目**で、いまは 84。
126 → 252 が大きく効いて 378 では効きが落ちる見込みなのは、この境界による。

## 5. 三度踏んだ罠 —— 読む前にこれを読む

### 罠 1:占有率を目的関数にした(2 回)

| | 占有率 | 速度 |
| --- | --- | --- |
| `slots-busy` 1.82x | 上がった | **変わらない** |
| `genGPU%` 77 → 88(FlashInfer) | 上がった | **4.3% 悪化** |
| 学習の NVML 100% | 100% | MFU は **0.34** |

**NVML が測るのは「kernel が載っているか」であって「有用な仕事が進んでいるか」
ではない。** スピンループも、帯域待ちの遅い kernel も、完璧な kernel も等しく
100% を返す。

### 罠 2:単発の値で run 間比較をした

`s/batch last20` は**同一 run の中で 11.8〜16.3 まで振れる**(search の batch
ごとに質問も応答長も違う)。20 batch の窓でも足りない。n=1 で比べて
「depth 2 は効果ゼロ」と結論し、あとで 16.5% だったと分かった。

### 罠 3:span と s/batch を突き合わせた

**span は batch の中身、s/batch は batch の間隔。** その差(2.9 s)こそが
pipeline の埋める対象で、両者を比べるのは軸を間違えている。

### 罠 4:2 つの変更を 1 run に混ぜた(2 回)

chunked prefill と合流、depth 3 と合流。前者は合算がゼロだったので分離せず、
後者は 1 run 追加で切り分けて **23% は全部合流**と判明した。

### 罠 5:計器を作って出力を忘れた

`GenerateMerger` は合流回数を数えていたのに印字していなかった。「一度も合流
しなかった」と「合流したが利得がなかった」が外から同じに見える —— この arm が
既に 3 回直した構図を、その修正の中で自分で作っていた。

### 正しい判定の型

**決定的な task(search は `do_sample=False`)の区間で、同じ batch 番号の
累積 `wall=` の差分を取る。** 完走は要らない。batch 幅が変わる比較では
`ms/row` を使う(batch 番号が同じ行を指さなくなるため)。

---

## 6. 計測器の一覧 —— 何が見えて何が見えないか

| 計器 | 出どころ | 見えるもの | **見えないもの** |
| --- | --- | --- | --- |
| `WALL ... ms/row` | `rollout_loop.py` | run 間で比較できる唯一の速度指標 | 止まっている場所 |
| `WALL ... slots-busy` | 同 | slot の平均占有 | **「全 slot が空」** |
| turn table `genGPU%` | 同(`GPU_PROFILER=1` が要る) | **generate 呼び出しの中**の GPU 占有 | **呼び出しの外**。機械 util ではない |
| turn table `promptTok`/`genTok` | 同 | prefill と decode の量 | prefix cache hit(promptTok は**上界**) |
| `[gen-phases]` | `fsdp_workers.py` | worker 内の 5 脚 | engine の内側 |
| `[rollout-phases]` | `vllm_rollout_spmd.py` | engine 境界の 3 脚 | engine の内側 |
| `[rollout-merge]` | `generate_merge.py` | 合流率と相乗り行数 | — |
| `[rollout-pump]` | `pump_client.py` | 投入・完了・round 数・rank ごとの in-flight | engine の内側 |
| `[rollout-advance]` | `rollout_loop.py` | turn が lock-step か task ごとか | どちらが速いか |
| turn table `slot=<slot>/<task>` | 同(§7.6 が ON のとき) | **task ごとの内訳** | task をまたぐ相互作用 |
| `[val-pipeline]` 被覆 | `val_pipeline.py` | **どの slot も走っていない**時間、呼び出しスレッドの内訳 | slot が走っていて GPU が空いている状態 |
| wandb system stream | wandb | **機械の util(唯一の真値)** | 15 秒点サンプル。原因は言わない |

**罠:`genGPU%` を「util」と呼んではいけない。** 本文書 §1 の 77.8% は wandb の
値で、`genGPU%` の 87% とは別物である。

### 環境変数

```
GPU_PROFILER=1              genGPU%/perGPU% を埋める。これが無いと列は "-" のまま
ROLLOUT_TURN_TIMING=1       turn table と WALL 行
ROLLOUT_GEN_PHASE_EVERY=50  [gen-phases] / [rollout-phases] の周期
VAL_PIPELINE_REPORT_EVERY=25 [val-pipeline] の周期(0 で最後だけ)
ROLLOUT_MERGE_GENERATES=1   generate の合流(既定 OFF)
VAL_PIPELINE_DEPTH=3        slot 数(合流には 3 以上が要る)
ROLLOUT_ASYNC_GENERATE=1    engine をプールとして回す(既定 OFF、合流の上位互換)
ROLLOUT_PUMP_ROUND_S=0.02   1 round の待ち。長いほど RPC は減り、完了通知は遅れる
ROLLOUT_PUMP_REPORT_EVERY=0 [rollout-pump] の周期(0 で session 終了時だけ)
ROLLOUT_PUMP_MAX_IN_FLIGHT=0 engine に同時に預ける上限(0 で無制限。既定でよい)
ROLLOUT_PER_TASK_ADVANCE=1 task ごとに turn を進める(既定 OFF、§7.6)
ROLLOUT_ASYNC_REQUIRE=1    pool に載らない呼び出しを fallback させず落とす(測定時は必須)
ROLLOUT_PUMP_REQUEST_TIMEOUT_S=900  どの rank も finished/failed に載せなかった request を諦める
ROLLOUT_PUMP_RESULT_TIMEOUT_S=1800  driver 側の保険。0 で無効
ROLLOUT_PUMP_SEED=1        request ごとに (prompt, row) から seed を作る(既定 ON)
```

---

## 7. いま最も速い設定

```bash
ray stop --force
ROLLOUT_MERGE_GENERATES=1 VAL_PIPELINE_DEPTH=3 \
EXPECTED_CONFIG_WAIVE=env.multitask.val_per_task_batch_size \
bash examples/sft_trainer/eval_checkpoints.sh \
  -- env.multitask.val_per_task_batch_size='{alfworld:126,search:252,webshop:126}' \
```

`ROLLOUT_KEEP_VLLM_AWAKE`、`SKIP_ROLLOUT_BUILD=0`、`GPU_PROFILER`、
`ROLLOUT_TURN_TIMING`、`VAL_PIPELINE_DEPTH=2` はスクリプトの既定で入る。

### 恒久化するには 2 つの宿題がある

1. **`EXPECTED_CONFIG_WAIVE` は一時回避である。** `val_per_task_batch_size` は
   `expected_multitask_sft_config.yaml` で固定されていて、それは正しい
   —— この値が alfworld の episode を決めるからである。効果が確認できたら
   マッピング形に書き換えて、決定をファイルに残すこと。
2. **合流は生成を変える(実測)。既定に昇格させてはいけない。**
   同一 checkpoint・同一 retriever・同一 batching で、合流あり/なしの
   `[val-hash]` を search 30 batch ぶん比べた結果:

   | | |
   | --- | ---: |
   | 一致しなかった batch | **28 / 30** |
   | 1 batch あたりの行数差 | 591 対 587 など、**数行(~0.7%)** |

   `rows=` は「軌跡 × 有効 turn」の総数なので、**行数が違う = 一部の軌跡が
   違う turn 数で終わった** —— token が変わり、検索するか答えるかの分岐が
   変わったということである。greedy でも batch 形状が reduction 順序を変える、
   という機構どおりの結果。

   **したがって採否は score でしか決められない。** 判定は
   `val/search/test_score` を合流あり/なしで完走比較すること。
   **その前に対照が要る:合流なしを 2 回走らせて `[val-hash]` が一致するか。**
   一致すれば非決定性は合流由来と確定し、しなければ depth 3 自体が
   非決定的ということになって話が変わる。

---

## 7.5 async(engine をプールとして回す)

`ROLLOUT_ASYNC_GENERATE=1`。**合流の上位互換**であって、別の打ち手ではない。

### 何をしているか

`LLM.generate(prompts)` は「252 行を全部 add して、最後の 1 行が終わるまで
step して、まとめて返す」1 本の同期呼び出しである。中の batching は
もともと continuous なのに、**呼び出しが同期**なので、別の batch の行は
途中から入れない。だから毎回の呼び出しの尻で、数本の長い軌跡だけが
ほぼ空の GPU を占有して decode する(12 軌跡で per-GPU 61/89/56 を実測)。

async はその engine を、**1 本のスレッドが `step()` を回し続けるプール**に
する。pipeline の各 slot は自分の行を投げて自分の答えを待つだけになるので、
slot A の行と slot B の行が**常に**同じ decode step に同居する。合流が
「たまたま queue が重なった 21.5%」で拾っていたものを、100% にする。

| | 合流 | async |
| --- | --- | --- |
| 同居する条件 | 呼び出しが偶然重なったとき | 常に |
| 実測の発火率 | 21.5% | ― |
| 効果(実測/見込み) | −19〜24% (ms/row 75→57) | 未測定 |

### 実装(4 つの部品)

| ファイル | 役割 |
| --- | --- |
| `verl/workers/rollout/token_pump.py` | 1 スレッドが engine を占有し `add_request`/`step` を回す。**他スレッドは queue 経由**でしか触らない |
| `verl/workers/rollout/generation_output.py` | token id → DataProto の組み立て。**blocking 経路と共通**。2 本あれば必ずずれる場所なので 1 本にした |
| `vllm_rollout_spmd.py: pump_step` | rank 側の 1 round。投入と回収を**同じ呼び出しで**行う(Ray actor はメソッドを 1 本ずつしか実行しないので、分けると回収が投入の後ろに並ぶ) |
| `verl/workers/rollout/pump_client.py` | driver 側。全 slot が 1 つの client を共有し、**1 スレッドだけ**が round を回す。rank 割り当ては least-in-flight |

### 成り立つ前提と、断る条件

**`tensor_model_parallel_size=1`** が効いている。各 rank が丸ごとモデルを
持つので、どの rank がどの request を捌いてもよく、driver が負荷で選べる。
TP>1 なら TP group の全 rank が同じ request を同じ順で add しないと次の
collective で固まるので、この設計は成立しない —— worker 側で断る。

断る条件(いずれも **blocking 経路に静かに落ちる**、近似で処理しない):

| 条件 | 理由 |
| --- | --- |
| `tensor_model_parallel_size > 1` | 上記 |
| LoRA | adapter は呼び出し単位で選ばれる |
| `return_rollout_log_probs=True` | pump が返すのは token id だけ |
| multimodal な行がある | prompt が token id だけではない |
| `n > 1` になりうる呼び出し | request 1 本 = 1 系列。黙って 1 本目を採るのは採点の変更 |
| `raw_prompt_ids` が無い | blocking 側はそこで input_ids から作り直す。その処理をもう 1 本持たない |
| rollout session が開いていない | pump は round の合間も engine を step する。session 外では sharding manager が寝かせた engine を叩くことになる |

session を閉じるより**先に** pump を閉じる。pool の寿命は最も外側の
session と同じで、重みが変わる境界をまたがない。

### 再現性 —— merge と同じ判断である

async は merge より**徹底的に**非決定的である。merge は「呼び出しが重なったか」
で決まるが、async は request の到着タイミングで decode step の中身が決まる。
**設定を固定しても走行ごとに変わる。**

したがって §7 の宿題 2 と同じ 1 つの判断になる:

* 判定を「`[val-hash]` 一致」に置くなら、async も merge も入らない。
  ついでに val batch 126→252 のような**採用済み**の変更も同じ理由で落ちる
  (batch の形が変われば reduction 順序が変わる、が機構だから)。
* 判定を「score が動かない」に置くなら両方入る。そのとき必要な数字は
  **同一設定を 2 回走らせた `val/*/test_score` の差**(評価器のノイズ下限)で、
  以後の比較はすべてこれを超えていなければ意味がない。

**測っていないものは既定にしない。** `ROLLOUT_ASYNC_GENERATE` は OFF。

### 試すコマンド

合流と**入れ替え**る(重ねない)。比較対象は §7 の設定、指標は `WALL ... ms/row`:

```bash
ray stop --force
ROLLOUT_ASYNC_GENERATE=1 VAL_PIPELINE_DEPTH=3 ROLLOUT_PUMP_REPORT_EVERY=200 \
EXPECTED_CONFIG_WAIVE=env.multitask.val_per_task_batch_size \
bash examples/sft_trainer/eval_checkpoints.sh \
  -- env.multitask.val_per_task_batch_size='{alfworld:126,search:252,webshop:126}' \
```

起動直後に出るべき 2 行:

```
[rollout-pump] driver: ROLLOUT_ASYNC_GENERATE='1' -> generate calls go through one continuously stepped engine per rank
[rollout-pump] driving 3 ranks as a pool; {'pad_token_id': ..., 'response_length': ..., 'eos_token_id': ...}
```

**2 行目が出ず `staying on the blocking path: ...` が出たら、その理由が答え**で、
測っているのは blocking 経路である(§7.5「断る条件」の表)。合流のときに
「一度も合流していない」と「合流したが効果ゼロ」を取り違えたのと同じ穴なので、
ここは必ず確認すること。

`ROLLOUT_PUMP_ROUND_S` は RPC 回数と完了通知の遅れのトレードオフ。既定 0.02 で
30 秒の batch あたり 1500 round。`[rollout-pump]` の `rounds` が多すぎるようなら
0.05 まで上げてよい(尻の遅れは 1 turn あたり最大 1 round ぶん)。


### 外部レビューで出た欠陥と、その修正

構造の話ではなく、**動く／動かないの話**が5件出た。うち3件は本物で、2件は前提が
違っていた。修正済みで、いずれも回帰テストが元のコードで落ちることを確認してある。

| | 何が起きるか | |
| --- | --- | --- |
| **abort が exception 経路で一度も走らない** | `_fail_all` が `_pending` を空にしてから `_abort_outstanding` が読むので、engine 例外のあと request が**engine に居座り KV block を握ったまま**次の rollout に持ち越される。まさに abort が存在する理由の場面で、abort だけが起きない | 直した |
| **submit と生存確認が別ロック** | 確認と挿入の間に pump が死ぬと、`_fail_all` が済んだ後の `_pending` に future が入る。誰も解決せず、**待ち手は永久に待つ** | 直した(確認を挿入と同じロックの中へ) |
| **どの rank も返さない request で永久停止** | `_pending` が空にならないので round も idle にならず、client は回り続け caller は止まったまま。timeout も watchdog も無かった | 直した(request ごとの deadline + driver 側 timeout) |
| **request id の使い回し** | レビュー外。id は `{name}-{n}` で `_next_id` は client ごとに 0 から。rank 側の done-queue は client より長生きするので、**前 session の完了が次 session の別 request を間違った token で解決しうる**。例外は出ない | 直した(client ごとの epoch + stop 時に queue を drain) |
| **n サンプル検証の潰れ** | 私が入れた seed のバグ。`val_kwargs.n>1` は DataProto 側で行を複製するので n 個のプロンプトが**バイト単位で同一** → 同じ seed → 同じ軌跡。n 個の標本が 1 個の n 複製になり、**分散が下がるので改善に見える** | 直した(seed に行番号を混ぜる) |

前提が違っていたもの:

* **「silent fallback で blocking を測ってしまう」** —— 起きうる。`ROLLOUT_ASYNC_REQUIRE=1`
  を足した。ただし `ROLLOUT_ASYNC_GENERATE` を忘れた REQUIRE 単独が同じ穴になるので、
  その組み合わせは import 時に落とす。
* **「least-in-flight が prefix cache の sticky を壊す」** —— **blocking 経路が
  そもそも sticky ではない。** `dispatch_dp_compute_data_proto` は
  `chunk(world_size)` で**圧縮済み active 行**を等分するので、軌跡が 1 本終わる
  たびに以降の行の rank がずれる。pump が壊したのではなく、最初から無い。
  sticky routing は**新しい最適化**であって回帰修正ではない(§8)。
* **「per-trajectory ではない」** —— そのとおり。§7.6 参照。task 単位まで。

---

## 7.6 per-task advance(task ごとに turn を進める)

`ROLLOUT_PER_TASK_ADVANCE=1`。**§4 の (a) 10.6% を狙う唯一の打ち手**で、async
(§7.5)が狙った (c) 0.4% とは的が違う。

### 何が問題だったか

turn ループが 1 本しかないので、全 task が毎 turn の境界で最も遅い task を待つ。
alfworld は 50 turn、search は 4 turn、webshop は 15 turn なので、search が
4 turn で終わった後も**その行は alfworld の 50 turn 目まで同じループに縛られる**。
そして §4 の測定どおり、preproc も envstep も互いに取り合って一緒に遅くなるので
位相がずれず、**3 task が揃って GPU の外にいる時間**ができる。

「揃って」いるのは物理ではなく、lock-step ループがそう並べているからである。

### 何をしたか

| | |
| --- | --- |
| `MultiTaskEnvironmentManager.step(text_actions, tasks=[t])` | t だけを step する。`_task_steps` / `_task_done` / `_last_obs_by_task` は元から task キーなので、別 task を持つ呼び出し同士は同じキーに触れない |
| `task_row_indices()` | task → 行番号。reset が決めた分割そのもので、**互いに素**である |
| `_advance_turn(rows=..., task=...)` | turn 1 回ぶん。**唯一の実装**で、batch 全体のループも per-task のループも同じものを呼ぶ |
| `_advance_per_task` | task ごとに 1 スレッド。各スレッドは自分の行だけを書く |

**ロックは無い。** 行集合が互いに素なので、共有配列の各要素に書き手は 1 人しか
いない。危険なのは「同じ task を 2 スレッドが名指す」ことで、task ごとに 1 本
という fan-out がそれを構造的に禁じている。

### 断る条件(`_per_task_groups`)

| 条件 | 理由 |
| --- | --- |
| flag が OFF / task が 1 つ | 分ける相手がいない |
| `obs['image']` がある | batch 全体で 1 枚の配列で、行ごとに書き込めるリストではない |
| `_logprob_prefetch_enabled` | 共有リストに append し、env.step 中に GPU へ 2 本目を投げる。学習側の機構で、検証では立たない |

**trajectory 単位ではなく task 単位である。** manager が持つ turn カウンタが
task ごとだからで、行ごとにするには alfworld / webshop / search の
env package を de-vectorize する必要がある(`docs/optimization_report.md` が
「multi-week」と書いているのはこれ)。**task 単位は (a) の位相ずれを作るには
足りる**というのがここの賭けで、足りるかどうかは測って決める。

### 生成は変わる

merge と同じ理由で変わる。各 task が自分の行だけで generate するので batch の
形が変わり、reduction 順序が変わる。**採否は score でしか決められない。**
`tests/ray_cpu/test_per_task_advance.py` が保証するのは記帳の方 ——
生成を行ごとの純関数にした条件で、lock-step と per-task が同じ
`total_batch_list` / rewards / episode_lengths を作ること。

### 走らせ方

```bash
ray stop --force
ROLLOUT_PER_TASK_ADVANCE=1 ROLLOUT_ASYNC_GENERATE=1 ROLLOUT_ASYNC_REQUIRE=1 \
EXPECTED_CONFIG_WAIVE=env.multitask.val_per_task_batch_size \
bash examples/sft_trainer/eval_checkpoints.sh 100 \
  -- env.multitask.val_per_task_batch_size='{alfworld:126,search:252,webshop:126}' \
  2>&1 | tee /tmp/eval_pta.log
```

async とは独立に効くが、**組み合わせる意味はある** —— per-task advance は
3 本の generate 呼び出しを同時に立てるので、pump が無ければそれらは worker
group 上で直列に並ぶ。§7.5 の pump がそこを埋める。

turn table が **task ごとに 1 枚**出るようになる(`slot=primary/search` など)。
「タスクごとの内訳が出せない」と書いてあった §6 の穴はここで埋まる。WALL 行は
batch ごとに 1 本のままで、3 枚の表が 3 本の WALL を出すことはない。

---

## 8. 残っている打ち手

| | 見込み | コスト |
| --- | ---: | --- |
| **preproc の増分トークナイズ** | util +6 pt、wall −8% | 毎 turn 履歴を丸ごと tokenize し直しているのを、token id を持ち回って差分だけにする。**BPE の結合が turn 境界をまたぐと採点が変わる**ので、全 turn で「全履歴 tokenize」と「差分連結」が token 単位で一致することを実軌跡で確認するテストが先 |
| **trajectory ごとの env step** | util +6 pt(§4a の残り) | **task 単位までは実装した(§7.6、既定 OFF、未測定)。** 残るのは行単位で、そこは env package の de-vectorize が要る。retriever のバッチ化(envstep 10.60→1.00 s/batch)と両立させるため「窓で束ねる」形が必要になるのも行単位から |
| **sticky routing** | 不明 | 同じ軌跡を毎 turn 同じ rank に送れば prefix cache が効く。**blocking 経路も sticky ではない**ので回帰修正ではなく新規の打ち手。軌跡の識別子が generate 呼び出しまで来ていないので、そこの配線が要る |
| `VAL_PIPELINE_DEPTH=4` | 不明 | 合流率が上がる可能性。ただし CPU 競合(§4a)が先に律速する見込み |
| retriever の GPU 専有 | +2 pt | `CUDA_VISIBLE_DEVICES` で 1 枚ずつ。第三者(`100.86.45.34`)との調整が要る |

**天井は 95% 前後**で、100% には届かない。(b) の残余は 1.7B というモデルサイズ
の物理で、モデルを大きくする以外に手が無い。

---

## 9. 評価そのものの設計について(性能の外)

評価時間の **99.5% は search** である。`prepare_sdar_multitask.py` が
alfworld と webshop を 126 行に絞る一方、**search は test parquet の全行
(51,713)を採点する**(`search: None`)。これは意図的で、単一タスクの search
baseline と `val/search/test_score` を比較可能にするためである。

結果として誤差の釣り合いは取れていない:

| task | n | 標準誤差(p≈0.4) |
| --- | ---: | ---: |
| alfworld | 126 | ±4.4% |
| webshop | 126 | ±4.4% |
| **search** | **51,713** | **±0.22%** |

**search だけ 20 倍細かい。** 固定の部分集合(seed 固定 2,000 行)にすれば
誤差 ±1.1% で **評価は 25 分の 1**になる。ただし baseline を同じ集合で
採り直さないと比較可能性を失うので、**これは性能ではなく科学的な決定**である。

**性能側の 3.4 倍より、こちらの方が桁で大きい。** 反復を速くしたいなら、
まずここを検討する価値がある。
