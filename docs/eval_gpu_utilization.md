# 評価が GPU を 47% しか使わない —— 計測と、どこまで直したか

学習は 99.9%(`docs/spike_investigation.md` 11 節)。同じコード・同じカードで
評価は **46.7%** である。この文書は評価側だけを扱う。

計測対象: run `sft-multitask-eval-20260824-141141`
(`global_step_300`、`val_only=True`、vLLM、3×A6000)。

---

## 1. 二つの計測器が同じ答えを出した

wandb のシステムストリーム(15 秒の点サンプル、warm-up 後 618 点)と、
走行中に取った `nvidia-smi -l 2`(60 秒、30 点)。

| 区分 | wandb | nvidia-smi | GPU | 電力 |
| --- | ---: | ---: | ---: | ---: |
| 生成(vLLM 常駐、busy) | 46.6% | 37% | 78% | 257 W |
| **env.step 待ち(vLLM 常駐、カーネル 0 本)** | **42.6%** | **50%** | 0% | 135 W |
| **vLLM の 21 GB unmap/remap** | **10.8%** | **13%** | 97% | 153 W |
| ノード平均 | **46.7%** | — | | |

2 秒サンプルの内訳が粗いのは n=30 だからで、形は一致している。

**15 秒の点サンプルから回数や周期は出せない。** 最初にこのデータから
「66 回、1〜3 分間隔」と書いたのは読み過ぎで、周期を確定したのは 2 秒側である。

---

## 2. 13% —— vLLM が毎 turn 寝起きしていた(解消済み)

2 秒トレースのメモリが 34 秒周期でこう動く:

```
29176 → 7694 → 16448 → 28984  (所要 ~4 秒)
```

これは `FSDPVLLMShardingManager.__enter__` の三段構造と一致する
(`verl/workers/sharding_manager/fsdp_vllm.py`):

| 観測値 | コード |
| --- | --- |
| 29176 → 7694 | `__exit__` の `sleep(level=1)` + `empty_cache()` |
| 7694 → 16448 | `wake_up(tags=["weights"])` + `update_params` |
| 16448 → 28984 | `wake_up(tags=["kv_cache"])` |

`ROLLOUT_KEEP_VLLM_AWAKE=1` の session mode は、これを rollout 1 回につき
1 度に畳むためのものである。34 秒周期は turn のオーダーで、
**session が効いていない**。

### なぜ気付かなかったか —— 両方の状態が無言だった

`__enter__` / `__exit__` のログは `log_gpu_memory_usage(..., logger=logger)`
経由で、**既定 `level=logging.DEBUG`**、そのロガーは
`logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))`。**どちらの状態でも
一行も出ない。** 外から見ると「driver が session を要求しなかった」と
「worker が開くのを断った」が区別できない。

さらに、フラグは **rollout loop を回すプロセス(trainer の Ray actor)の
import 時**に解決される。launcher でも rollout worker でもない。
worker の `/proc/*/environ` を grep するのは自然な確認だが、**別の問いに
答えていて、しかも安心させる答えを返す**。

### 入れたもの

* driver が**自分で解決した値**を 1 回 print(`_say_rollout_env`)
* worker が**どの分岐を取ったか**を print。早期 return は理由を名指しする
  (manager が無い / vLLM の manager でない)
* session を閉じるとき、**1 回の wake が何回の generate を賄ったか**。
  1 回しか賄っていない session は何も買っていない
* `eval_checkpoints.sh` が `ROLLOUT_KEEP_VLLM_AWAKE=1` を強制

### 結果 —— turn 内では解消、batch 間には残っていた

次の eval のログ:

```
[rollout-session] driver: ROLLOUT_KEEP_VLLM_AWAKE='1' -> session mode ON
[rollout-session] rank 0: opened -- vLLM stays awake until the outermost scope closes
[rollout-session] rank 0: closed after 50 generate calls on one wake
```

**50 turn を 1 回の wake で賄っている** —— turn ごとの寝起きは消えた。

**しかし batch ごとには残っていた。** session は `multi_turn_loop` 単位で開閉し
(`rollout_loop.py`)、`_validate` は `val_dataloader` を回すので **413 回**開閉
していた。旧 run で NVML が測った低メモリ状態 **wall の 10.4%** と、2 秒トレースの
**周期 34 秒 ≒ search batch 1 本の長さ**は、どちらもこの per-batch churn である。
alfworld batch(230 秒)なら償却できるが、search batch は 24 秒しかない。

### hoist —— 413 回を 1 回に

評価中は重みが一度も変わらないので、session は `_validate` 全体を 1 回で包める。
入れたもの:

* `rollout_session(actor_rollout_wg)` —— 両方の呼び出し側が共有する contextmanager
  (`rollout_loop.py`)。`finally` で必ず閉じるので、途中で例外が出ても vLLM は
  眠って戻る
* `_validate` が **batch ループの外側**でそれを開く
* worker 側の session を **深さで数える**ようにした(`fsdp_workers.py`)。
  bool のままだと**内側の 413 回目の close が外側の session を閉じてしまい**、
  hoist が黙って無効になる —— 最初の batch 以降は元通り毎 batch 寝起きし、
  しかもログは「1 回開いた」と言い続ける

**期待効果: wall の 10.4%。** 3 節の (a) と合わせて 88.2% → 98.5%(見積もり)。

## 3. 42.6% —— retriever 待ち。二つの計測器が同じ数字に着地した

turn timing を batch ごとに読むと、**評価は 2 種類の batch でできている**:

```
batch 0  alfworld  TOTAL  pre 11.2  gen 201.0  dec 1.4  env  17.0  total 230.6   gen 87.1%
batch 1  webshop   TOTAL  pre  5.4  gen  78.5  dec 0.4  env   3.5  total  87.8   gen 89.4%
batch 2+ search    TOTAL  pre  1.3  gen  12.2  dec 0.1  env  10.6  total  24.2   gen 48-56%
```

search batch が **411 本**(413 − alfworld − webshop)ある。重み付けすると:

| | 秒 | 割合 |
| --- | ---: | ---: |
| generate | 5,316 | 51.8% |
| **envs.step** | **4,377** | **42.6%** |
| preproc | 535 | 5.2% |
| decode | 43 | 0.4% |
| 合計 | 10,268 s = 2.85 h | |

**envs.step の 99.5% は search batch にある。**

そして 1 節の NVML —— 前 run の 694 窓で「engine 常駐・カーネル 0 本」が
**42.9%** —— と、turn timing の **42.6%** が一致する。**独立な二つの計測器が
同じ量を測っていた。**

### 経緯としての訂正

この 42〜50% を最初に NVML から出し、そのあと alfworld batch 1 本の
turn timing(envstep 7.4%)を見て「外挿の誤りだった」と撤回した。
**撤回の方が誤りだった。** alfworld batch は run の 2.2% で、そこでの
envstep は本当に 7.4% だが、残り 97.8% を占める search batch では 44% である。
1 本から一般化して外し、次は別の 1 本から一般化して戻し過ぎた。
**batch ごとに形が違う workload では、加重するまで何も言えない。**

### 中身 —— retriever への HTTP

search の `envs.step` は `http://100.86.45.30:8001/retrieve` への呼び出しである。
1 batch 4 turn なので **1 turn あたり 2.65 秒**、同じ turn の generate が
3.06 秒。既に 126 リクエストは並行で飛んでいる
(`envs.py:68` の `ThreadPoolExecutor(max_workers=min(batch_size, 256))`)ので、
2.65 秒は**126 並行下での retriever の応答時間**であって直列化ではない。

### 打ち手 —— サーバ側は塞がっていて、バッチだけが残った

`100.86.45.30` を見て、当初の想定が 2 つ潰れた。

```
retrieval_server.py --index_path .../e5_Flat.index --retriever_model e5-base-v2 --faiss_gpu --port 8000
retrieval_server.py --index_path .../e5_Flat.index --retriever_model e5-base-v2 --faiss_gpu --port 8001
GPU0: 36599/49140 MiB   GPU1: 34509/49140 MiB   (両プロセスが両 GPU に分割して載っている)
112 cores / 754 GB
```

* **worker 数を増やす → 該当しない。** `--faiss_gpu` なので仕事は GPU にある。
  112 コアは無関係。
* **レプリカを増やす → 塞がっている。** 8000 と 8001 は**同じ 2 枚**を共有していて、
  空きは 12.5 / 14.6 GB。index 1 部が約 35 GB なので 3 台目は入らない。

そして 80 ms の正体が分かった。**`e5_Flat` は全探索**である。wiki-18 は約 2,100 万
パッセージ、768 次元 fp16 で **約 32 GB**。1 クエリごとにこれを全部読む:

```
32 GB ÷ A6000 の帯域 768 GB/s ≒ 42 ms(2 枚分割で ~21 ms)+ エンコード → 実測 80 ms
```

**93 倍の膨張はここから出る。1 クエリで既に帯域を使い切るので、126 並行は
並列化されず順番待ちになる。** サーバを増やしても帯域は増えない。

#### バッチだけが桁で効く

Flat index では、126 クエリを 1 回の `index.search()` に渡せば **32 GB を 1 回しか
読まない**:

| | 読む量 | 時間 |
| --- | --- | ---: |
| 126 クエリを個別に | 32 GB × 126 | 10.1 s |
| 126 クエリを 1 バッチで | 32 GB × 1 + 演算 | ~0.1 s |

そして **サーバには既にその実装がある** —— `DenseRetriever._batch_search` は
`encoder.encode(query_batch)` と `index.search(batch_emb)` を 1 回ずつ呼び、
`retrieval_batch_size=512` で刻む。HTTP から届いていなかっただけで、
`retrieval_batch_size` のコメントは「単一クエリしか対応していないので未使用」と
書かれていた。

#### 入れたもの

**サーバ**(`examples/search/retriever/retrieval_server.py`): `QueryRequest.query` を
`Union[str, List[str]]` に。文字列の挙動もレスポンスの形も変えていないので、
**古いクライアントはそのまま動く** —— 再起動を誰かと調整する必要がない。

**クライアント**(`.../skyrl_gym/tools/search.py`): `call_search_api` の内側に
**コアレッサ**を置いた。rollout は全 env の action を ThreadPoolExecutor から
同時に投げるので、呼び出しは数ミリ秒以内に揃う。最初の呼び出しが 10 ms 待って、
溜まったぶんを 1 リクエストで送る。

**env の意味論には一切触れていない。** 各呼び出し元は
`{"result": [documents]}` という**単一クエリと同じ形**を受け取るので、
下流はバッチ化されたことを知りようがない —— 同じクエリ、同じ順序、同じ文書。

* クエリ 1 本のときは**素の文字列**で送る(未対応サーバを突かない)
* バッチが失敗したら**1 本ずつ再送**し、それで通れば「リストを受けないサーバ」と
  学習して以後バッチをやめる。エラー文字列を解釈するのではなく、**試して判る**
* 本当に retriever が落ちているときは、単発でも失敗するのでエラーが呼び出し元まで
  届く(ここを握り潰すと、エラー文字列が `<information>` として学習データに入る)

`SEARCH_BATCH_REQUESTS=0` で無効化、`SEARCH_BATCH_WINDOW_MS` で窓幅。

### 実測 —— 9 倍

バッチ対応の retriever(port 8000)に向けた run。search batch 3 本の平均:

| | before | after | |
| --- | ---: | ---: | ---: |
| **envstep** | **10.60 s** | **1.17 s** | **0.11x** |
| env / turn | 2.65 s | 0.29 s | |
| gen | 12.25 s | 11.87 s | 変わらず |
| preproc | 1.26 s | 1.20 s | 変わらず |
| batch 合計 | 24.21 s | **14.33 s** | 0.59x |

```
SHARE  gen(GPU-busy)=80.1%  cpu-glue=19.9%
SHARE  gen(GPU-busy)=84.8%  cpu-glue=15.2%
SHARE  gen(GPU-busy)=83.1%  cpu-glue=16.9%
```

**評価全体(turn 時間)2.85 h → 1.73 h、GPU util 51.8% → 83.0%。**
モデルの見積もりは 1.70 h / 87.0% だったので、ほぼ当たった —— 外れたぶんは
1 turn の retrieval が想定 100 ms に対し実測 292 ms だったこと(`load_docs` が
132 パッセージを HF dataset から引く分と JSON 化)。

### 残ったもの

| search batch 1 本 | 秒 | 割合 |
| --- | ---: | ---: |
| generate(GPU busy) | 11.87 | 82.8% |
| **preproc(CPU tokenize)** | 1.20 | **8.4%** |
| **envstep(retriever)** | 1.17 | **8.1%** |
| decode | 0.10 | 0.7% |

**preproc と retriever が同じ大きさになった。** どちらも generate の裏に
隠す以外に消しようがなく、それは cohort / async pipeline の話になる(5 節)。
retriever 側にはまだ 2〜3 倍ありそうだが(292 ms 対 理論下限 ~100 ms、差は
`load_docs` と JSON)、8.1% の 2/3 なので 5 pt 程度である。

### サーバ側のバージョンずれ —— 実際に踏んだ

2026-08-25 の eval で envstep が **1.17 s → 10.8 s** に戻った。原因は
retriever のバージョンずれで、`8000` は新コード、eval が向いていた
**`8001` は旧コードのまま**だった。リストのクエリが 422 で弾かれ、
クライアントが単発送信に落ちていた:

```
batched search of 38 queries failed (... 422 Client Error: Unprocessable Entity
for url: http://100.86.45.30:8001/retrieve); retrying singly
```

**この状態は turn table の `TOTAL` にしか出ない。** util も wall も「なんとなく
遅い」としか言わないので、`envstep` を見るまで retriever を疑えなかった。

無効化はプロセス内で永続だったため、retriever を直しても**走っている run は
単発のまま**という二段目の罠があった。`SEARCH_BATCH_RETRY_S`(既定 300 s)で
**定期的に再試行する**ようにした —— コストは 1 周期あたり 1 回の失敗リクエスト
だけ(クエリはどのみち単発で送られる)で、retriever を再起動すれば run を
落とさずに戻る。`0` で従来どおり永続。

## 4. 生成中の rank 不均衡 —— 帰属を間違えた(撤回)

2 秒トレースの生成中サンプル:

```
0, 87   1,  0   2,  0
0, 100  1, 31   2, 53
0, 87   1, 25   2, 88
```

これを「batch が task ごとに固まっていて、rank ごとに 1 task が当たるため」と
読み、round-robin する `VAL_TASK_INTERLEAVE` を入れた。**前提が偽だった。**

`examples/data_preprocess/prepare_sdar_multitask.py` の `_build_split` は
test セットをこう組む:

| batch | 中身 | max_steps |
| ---: | --- | ---: |
| 0 | alfworld 126 行 | 50 |
| 1 | webshop 126 行 | 15 |
| 2〜412 | search(test parquet 全行 ≈ 51,713) | 4 |

**batch は task ごとに単一**である。しかも `_validation_task_name`
(`ray_trainer.py:810`)は**混在 batch で ValueError を投げる** ——
`val_kwargs_by_task` が batch 単位で解決されるので、混ぜてはいけない。
search が最後に置かれているのも、その端数 batch が次の task と混ざらない
ようにするためである(同ファイルのコメント)。

したがって round-robin は恒等写像を返す no-op で、仮に効けば例外だった。
**revert 済み。** 観測された偏りは task 混在ではなく、**単一 task 内の
生成長のばらつき**である —— 早く終わった rank が、他 rank の最長系列を待つ。

**教訓としてここに残す**: `87/0/0` という署名から機構を推定して実装し、
データ生成側を読んでいなかった。この arm で 4 回目の同じ失敗である。

## 5. batch を 2 本並走させる —— 残った 16.5% への手

3 節のあと、search batch 1 本は **generate 11.87 s に対し preproc 1.20 s +
envstep 1.17 s**。この 2.37 s は「環境の答えを受けてから次の生成まで」に
挟まっていて、**先読みできない** —— 次の turn のプロンプトは今の turn への
環境の答えそのものだからである。

1 本の batch の中では埋められない。**別の batch なら埋められる。** batch は
互いに独立なので、片方が環境を待つ間にもう片方が生成できる。worker group は
Ray actor で自分の呼び出しを直列化するので、**2 本の generate は勝手に順番待ち
になり、重なるのは「一方の環境待ち」と「他方の生成」だけ** —— つまり狙った隙間
そのものである。

### 採点を変えないための 3 条件

`verl/utils/val_pipeline.py`。

1. **順序。** 結果は投入順に retire し、集計は呼び出しスレッドで行う。
   集計先は平坦なリストで、あとから `data_source` や `task_name` と**位置で**
   突き合わせるので、順序が崩れれば全行が別の行のメタデータで採点される
   —— しかも例外は出ない
2. **隔離。** slot は env manager と TrajectoryCollector を**専有**する。
   どちらも rollout ごとの状態(観測履歴、env ごとの step カウンタ)を持つので、
   2 本が同じ manager に入れば履歴が混ざる
3. **適格性。** extra slot は**2 個目の manager を作っても同じ episode を採点
   できる task に限る**(`PIPELINEABLE_VAL_TASKS`)。alfworld は入っていない
   —— `AlfworldEnvs` は worker i を `seed + i // group_n` で seed するので、
   どの game を引くかが**その manager 内の位置**の関数になる。2 つに分ければ
   全行が別の game を引き、これも例外は出ない。search は各行が自分の
   question と ground_truth を reset で受け取り、`_rng` は作られて一度も
   使われないので、2 つの manager は交換可能である

**`VAL_PIPELINE_DEPTH=1`(既定)では thread に投げすらしない** —— 呼び出しは
inline で、従来の逐次ループと同一である。

### 見込み

generate が 2.97 s/turn、preproc + env が 0.59 s/turn なので、pipeline は
generate 律速になる。batch 14.33 s → ~12.5 s、評価全体 **1.73 h → ~1.52 h、
util 83% → ~94%**。**未実測。**

### 効いたかどうかの判定 —— turn table では答えが出ない

pipeline は **1 本の batch のコストを変えない**。変えるのは「2 本目がいつ走るか」
だけなので、turn timing の `SHARE gen(GPU-busy)` は depth 1 でも depth 2 でも
同じ値を出す。実際 depth 2 の 1 batch 目(alfworld)は
`SHARE gen(GPU-busy)=87.1%` で、逐次実行と一致した。**ここを見て「効いていない」
と読むのは、この arm が 4 節でやったのと同じ種類の帰属ミスである。**

そこで、batch ごとの表の末尾に **WALL 行**を出すようにした
(`ROLLOUT_TURN_TIMING=1` のとき)。

```
WALL   slot=extra-1  batch#412  span=11.9s  s/batch last20=14.2s all=14.9s  wall=6145.0s  slots-busy=1.82x
```

* **`s/batch` —— run 間で比べる数字はこれだけである。** `last20` は直近 20 本の
  完了間隔、`all` は先頭からの累積
* `span` —— その batch 自身の秒数。**pipeline 下では膨らむ**(後述)
* `slots-busy` —— **占有率であって速度向上ではない**

```bash
grep 'WALL   slot=' eval.log | tail -1
```

**行頭アンカー(`^WALL`)は使えない。** rollout loop は Ray actor の中で走るので、
stdout の各行に `(SFTMultiTaskTaskRunner pid=…) ` が前置される。

#### `slots-busy` を speedup と読んではいけない —— 最初にそう設計して間違えた

当初この行は `serial/wall` という名前で、「逐次なら sum-of-spans、実際は wall、
だから比が速度向上」と説明していた。**前提が誤っている。** pipeline 下では
batch 自身の span が膨らむ —— その batch が座っている generate 呼び出しが、
他方の batch の generate の後ろに並ぶからである。実測の `TOTAL` でも gen が
**28.4 s と 10.3 s** に割れており、前者は待ち時間を含んだ値である。

span が 2 倍になった slot が 2 本あれば、**何も得ていなくても比は 2.00x を指す。**
実際 depth 2 の完走 run は **1.82x** を出しながら、s/batch は 1.5% しか動かなかった。
占有率としては正しく、速度向上としては無意味である。

#### depth 2 の実測 —— 効果はゼロだった

同条件(retriever は 8000、envstep 0.7〜1.1 s)で depth 1 と depth 2 を測った。

| | s/batch | span | slots-busy | 出典 |
| --- | ---: | ---: | ---: | --- |
| depth 1 | **14.2** | 13.8 s | 0.99x | 完了間隔 574.0 − 559.8 |
| depth 2 | **14.2** | 27〜38 s | 1.82x | `s/batch last20`、413 batch 完走 |

**同一である。** 見込みの 15.1 → 12.5 s/batch は出ず、**1 秒も縮まなかった。**

意味するところは明確で、`slots-busy=1.82x` が示すとおり **2 slot は確かに埋まって
いた**。埋まっていて throughput が変わらないのは、**両者が奪い合う資源が既に
飽和していた**ということである。したがって turn table の
`cpu-glue(preproc+decode+envstep, GPU-idle)=17.2%` を「2 本目の batch で埋められる
空き」と読んだ 5 節冒頭の前提が誤っていた。あの 17% は、少なくとも
「別の batch を並べる」という手では埋まらない。

**未解決:** NVML 平均 79.9% に対し同区間の gen share が 50.7% だった食い違い
(3 節の劣化中の測定)は、glue 中も GPU が動いていることを示す。wasabi の GPU を
eval 以外のプロセスが使っていないかは**未確認**であり、確認されるまでこの arm の
util の数字はすべて汚染の可能性を抱えている:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**`VAL_PIPELINE_DEPTH` は既定の 1 のままにする。** depth 1 では thread に投げすら
せず inline で走るので、コードが残っていても実行時のコストはない。

depth は毎回 `[val-pipeline] VAL_PIPELINE_DEPTH=N: ...` として**必ず出す**
(depth 1 でも)。出ないことが「depth 1」と「pipeline の無いビルド」の両方を
意味してしまうのは、2 節で session を 13% 見落とした構図そのものだからである。

`reset_batch_wall()` を `_validate()` の先頭で呼ぶので、学習中に評価を挟む場合も
評価ごとに 0 から数え直す。

### 使っていないもの

`agent_system/multi_turn_rollout/async_rollout_core.py` の軌跡単位スケジューラは
使っていない。あれが要るのは「1 本の batch の中で軌跡ごとに非同期化する」場合で、
それには vLLM の AsyncLLM 経路と env の部分 step が要る。**batch を 2 本並べる
だけで同じ隙間が埋まり、env にも rollout loop にも触らずに済む。**

## 6. 現状の台帳

| 項目 | before | after | 状態 |
| --- | ---: | ---: | --- |
| turn ごとの vLLM wake/sleep | — | 0 | **解消**(2 節、session) |
| batch ごとの vLLM wake/sleep | 10.4% | 0 | **解消**(2 節、hoist) |
| **search の retriever 待ち** | **42.6%** | **8.1%** | **解消**(3 節、バッチ化。10.60 → 1.17 s/batch) |
| preproc(CPU tokenize) | 5.2% | 8.4% | 残(割合は分母が縮んで上がった) |
| generate | 51.8% | 82.8% | GPU が働いている区間 |

**評価の turn 時間 2.85 h → 1.73 h、GPU util 51.8% → 83.0%(実測)。**

### ここから先

| | wall | util | 状態 |
| --- | ---: | ---: | --- |
| いま | 1.73 h | 83.0% | 実測 |
| `VAL_PIPELINE_DEPTH=2` | 変化なし | 変化なし | **実測、効果ゼロ**(5 節)。同条件で depth 1 も depth 2 も 14.2 s/batch |
| retriever の GPU 専有 | −0.04 h | +2 pt | 未着手。`CUDA_VISIBLE_DEVICES` で 1 枚ずつ。ただし 8001 は第三者(`100.86.45.34`)が使っており調整が要る |

retriever の内訳は実測済みで、**`load_docs` は 250 ms → 2.8 ms(total の 1.2%)**
で解決、残りは `encode`(61%)と `faiss`。`encode` がクエリ数に比例しない
(3 本 134 ms、21 本 27 ms、50 本 253 ms)ことから、計算ではなく **8000 と 8001 が
同じ 2 枚の GPU を取り合っている待ち**である。

**理論上限は 99.9% で、隠せないのは pipeline の fill と drain だけ。**
depth 2 の残り 6 pt はその fill/drain と、generate 律速からのわずかなずれである。
