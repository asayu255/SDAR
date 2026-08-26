# 評価が GPU を 47% しか使わない —— 計測と、どこまで直したか

> **結論だけ読むなら `docs/eval_performance_summary.md`。**
> こちらは時系列の実験ノートで、外れた仮説と訂正をそのまま残してある。
> 同じ罠を踏まないための記録であって、現状の要約ではない。

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

#### depth 2 の実測 —— 16.5% 短縮した

同一 checkpoint、同一 retriever(8000、envstep 1.0 s)、同じ推定器
(`WALL` 行の `s/batch last20`、n=20)で:

| | s/batch | span | slots-busy |
| --- | ---: | ---: | ---: |
| depth 1 | **17.0** | 14.1 s | 0.98x |
| **depth 2** | **14.2** | 27〜38 s | 1.82x |

**17.0 → 14.2、16.5% 短縮。** 見込み(15.1 → 12.5)とほぼ同じ比率である。

回収した中身は **turn table には出ない**。depth 1 の span は 14.1 s なので、
**17.0 s のうち 2.9 s は batch と batch の間**にある —— 次の batch の 126 本の
プロンプトを decode し、前の batch の 126 本の応答を decode し、reward を計算する
main thread の時間で、その間 GPU には何もない。turn table は batch の**中**しか
測らないので、この 2.9 s はどの行にも現れない。depth 2 が埋めたのは主にここである。

##### 一度これを「効果ゼロ」と誤判定した

depth 1 の s/batch を**完了間隔 1 個(14.2 s)**から取り、depth 2 の
`s/batch last20`(14.2 s)と並べて「同一、効果ゼロ」と結論した。誤りは二重だった。

1. n=1 だった。たまたま軽い batch が 2 本続いた区間を引いていた
2. **span と s/batch を突き合わせた。** span は batch の中身、s/batch は batch の
   間隔で、両者の差(2.9 s)こそが pipeline の埋める対象である。同じ推定器で
   両方を取れば 17.0 対 14.2 とはっきり離れる

**比較は必ず `s/batch last20` 同士で行うこと。** `slots-busy` は占有率、`span` は
batch の中身であって、どちらも run 間の比較に使ってはならない。

#### 残っているもの —— 0.9 s/batch

depth 2 の 14.2 s/batch に対し、generate 自身は 12.8〜13.8 s。**差は約 0.9 s、
6%** である。これが「GPU の裏に完全に隠す」で取れる残り全部であり、それには
trajectory 単位の連続バッチ(vLLM AsyncLLM + env の部分 step)が要る。
rollout loop と env manager の両方を書き換えて 6% なので、**割に合わない。**

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

最終測定は 2026-08-25、retriever 健全(8000)、`VAL_PIPELINE_DEPTH=1`。
search batch 1 本の内訳:

| | 秒 | 状態 |
| --- | ---: | --- |
| **generate(GPU busy)** | 12.8〜13.8 | 働いている区間 |
| preproc(CPU tokenize) | 1.15 | batch 内。depth 2 が隠す |
| envstep(retriever) | 1.00 | batch 内。depth 2 が隠す |
| decode | 0.10 | batch 内 |
| batch 間(126 本の decode ×2 + reward) | **2.9** | **turn table に出ない。** depth 2 が隠す |
| **depth 1 の合計** | **17.0** | |
| **depth 2 の実測** | **14.2** | 残る GPU 待ちは約 0.9 s |

| 項目 | before | after | 状態 |
| --- | ---: | ---: | --- |
| turn ごとの vLLM wake/sleep | — | 0 | **解消**(2 節、session) |
| batch ごとの vLLM wake/sleep | 10.4% | 0 | **解消**(2 節、hoist) |
| **search の retriever 待ち** | **42.6%** | **6.4%** | **解消**(3 節、バッチ化。10.60 → 1.00 s/batch) |
| retriever のバージョンずれ | 10.8 s | 1.0 s | **解消**(3 節、8000 へ張り替え + `SEARCH_BATCH_RETRY_S`) |
| preproc(CPU tokenize) | 5.2% | 7.4% | 残。**envstep を抜いて glue の最大項になった** |
| generate | 51.8% | **85.4%** | GPU が働いている区間 |

**評価の turn 時間 2.85 h → 1.71 h(413 batch 完走の実測)、s/batch は
depth 1 の 17.0 に対し depth 2 で 14.2。generate 律速の下限 12.8〜13.8 に対し
残る GPU 待ちは 0.9 s/batch。**

### ここから先

残っているのは **0.9 s/batch(6%)** だけである。generate 自身が 12.8〜13.8 s で、
これが下限。

| | 見込み | 状態 |
| --- | ---: | --- |
| **`VAL_PIPELINE_DEPTH=2`** | **−16.5%** | **実測・既定 ON**(5 節)。17.0 → 14.2 s/batch。`eval_checkpoints.sh` の既定を 2 にした |
| trajectory 単位の連続バッチ | −6% | **割に合わない。** vLLM AsyncLLM + env の部分 step が要り、rollout loop と env manager の両方を書き換えることになる |
| `VAL_PIPELINE_DEPTH=3` | 不明 | 未測定。埋める対象が 0.9 s しか残っていないので、slot を増やす余地はほぼない |

**GPU を完全に埋めきる形は trajectory 単位の連続バッチだが、残りが 6% なので
着手していない。** batch 単位で足並みが揃う限り「全員が同時に env を待つ瞬間」は
必ず生まれ、それを消すには 126 本を独立にスケジュールするしかない。depth 2 は
その谷を「もう 1 つの塊」で埋める手で、谷の大半(2.9 + α のうち約 2.8 s)を
実際に埋めた。

## 7. 次の的 —— 尻尾の turn が generate 時間の半分を食っている

depth 2 まで来て残るのは占有率で 6% だけである。**ここから先は「谷を消す」話では
なく、94% を占める generate 自身が**どれだけ**埋まっているか**の話になる。

search batch 1 本の turn 内訳(2026-08-25、depth 1):

| turn | active | gen (s) |
| ---: | ---: | ---: |
| 0 | 126 | 1.05 |
| 1 | 126 | 5.47 |
| **2** | **27** | **4.49** |
| **3** | **11** | **3.19** |
| | | **14.2** |

**後半 2 turn は仕事の 13%(38 / 290 軌跡×turn)に、generate 時間の 54%
(7.68 / 14.2 s)を使っている。**

decode の wall は「何本流すか」ではなく「何ステップ回すか」で決まるので、
11 本でも 126 本でもほぼ同じ時間がかかる。search の episode は終わる turn が
まちまちで、後半には数本しか残らない —— **126 本ぶんの席に 11 本を乗せて
3.19 秒走らせている。** NVML はこれを「busy」と呼ぶので、占有率をいくら見ても
出てこない。

### 先にやること —— `genGPU%` を埋める

turn table には `genGPU%` と `perGPU%` の列が最初からあるが、**評価では一度も
埋まったことがなかった。** 理由は二段階ある。

1. NVML サンプラは `GPU_PROFILER=1` でなければ no-op —— `eval_checkpoints.sh` の
   既定を 1 にした
2. **それだけでは足りなかった。** サンプラを生成するのは `push_phase()` だけで、
   それを呼ぶのは trainer の fit ループである。**評価パスは一度も呼ばない**ので、
   `GPU_PROFILER=1` を立てても `_sampler` は None のまま、`mean_util_between` は
   全ての窓に None を返し、列は `-` を出し続けた

列があって中身が無い状態は 2 節の session と同じ「無言の穴」で、しかも今回は
**フラグを立てた側から見て正しく設定できているように見える**ぶん質が悪い。
`gpu_profiler.ensure_started()` を公開し、turn timing が有効な rollout の先頭で
呼ぶようにした。あわせて `[rollout-session]` の行が
`GPU_PROFILER=... -> genGPU%/perGPU% columns will be filled / EMPTY (-)` を
名指しで出すので、次からは起動時に分かる。

これで尻尾の turn の SM 使用率が直接読める。**満員の turn が 95%、尻尾が 20% なら
上の読みは確定**で、打ち手の見込みも計算できる。両方とも高いなら
(decode がメモリ帯域律速で、本数を増やしても埋まらないなら)、**幅を広げても
無駄**ということになり、その場合に残るのはモデル側の高速化だけである。

### 見込まれる打ち手

| | 見込み | コスト |
| --- | --- | --- |
| **task ごとの val batch size** | 尻尾を 4 倍に薄める。gen 14.2 → ~8 s/126 行 | `val_per_task_batch_size` を task ごとの dict にする(`history_length` に前例あり)。search だけ 504 にすれば alfworld と webshop の manager は 126 のままで、**両者の採点は 1 行も動かない** |
| batch をまたぐ軌跡の詰め替え | 同上かそれ以上 | rollout loop に軌跡プールのスケジューラが要る。大きい |
| `data.val_batch_size` だけ上げる | 同上 | **不可。** env manager が同じ値で size されるので alfworld の worker 数が変わり、`seed + i // group_n` で引く game が変わって**採点が変わる** |

**まだどれも着手していない。`genGPU%` の実測が先である。**

## 8. `genGPU%` の実測 —— 尻尾ではなく、呼び出し固定費だった

サンプラを起動して初めて読めた値(2026-08-25、depth 2、10 turn の batch):

| turn | active | gen (s) | genGPU% | perGPU% | **GPU が空いた秒** |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 126 | 2.08 | 72 | 72/72/71 | 0.58 |
| 1 | 126 | 3.43 | 79 | 79/79/79 | 0.72 |
| 2 | 126 | 3.25 | 80 | 80/79/80 | 0.65 |
| 5 | 110 | 4.02 | 69 | 60/79/68 | 1.25 |
| 9 | 89 | 2.85 | 77 | 77/77/78 | 0.66 |

**7 節の読みは外れた。** 満員(active=126)でも SM は 72〜80% しかなく、
**active が減っても util はほとんど下がらない**(126 で 79%、89 で 77%)。
尻尾の席が空いているという話ではなかった。

`perGPU%` は 3 枚とも揃っているので **DP の偏りでもない**(この分岐は閉じた)。

代わりに出たのは **呼び出し 1 回あたり約 0.65 s の固定した GPU 空き**である。
turn の長さにも active 数にも比例していない。データ量に比例する転送コストなら、
長い履歴を送る後半 turn ほど util が下がるはずだが、**逆に上がっている**。

`_gw0.._gw1` は `actor_rollout_wg.generate_sequences()` 全体を囲っているので、
この 0.65 s には driver→Ray→worker の往復、sharding manager の per-call
前後処理、vLLM 入口、出力の detokenize が入っている。**turn ごとに 1 回払う** ——
search batch(4 turn)で 2.6 s、10 turn の batch で 6.5 s。

設定ミスではない。`enforce_eager=False`(CUDA graph 有効)、
`enable_prefix_caching` 既定 True、`perGPU%` 均等。

### 中身を割る —— worker 側の per-call 計測

driver からは 1 つの不透明な span なので、worker 側で
`to_device / preprocess / generate / postprocess / to_cpu` を積算し、
rank 0 が N 回ごとに平均を出すようにした(`ROLLOUT_TURN_TIMING=1` で有効、
`ROLLOUT_GEN_PHASE_EVERY` で周期、既定 50)。

```
[gen-phases] rank 0, mean over 50 calls (s): to_device 0.02  preprocess 0.31  generate 2.55  postprocess 0.09  to_cpu 0.12  worker-total 3.09  (driver's gen column minus this = the Ray round trip)
```

**driver の `gen` 列からこの `worker-total` を引いた差が Ray の往復**で、
これは片側だけでは絶対に測れない唯一の脚である。0.65 s がどの脚にあるかで
打ち手が変わる:

| 大きい脚 | 意味 | 打ち手 |
| --- | --- | --- |
| Ray の往復(差分) | turn ごとに全履歴を padded tensor で送り直している | `raw_prompt_ids` だけ送る。attention_mask と position_ids は vLLM に不要 |
| `preprocess` / `postprocess` | sharding manager の per-call データ整形 | 整形自体を削るか、GPU 上で行う |
| `to_cpu` | 出力の転送 | 非同期化 |
| `generate` の中 | vLLM 自身の per-step CPU | vLLM 側の設定・版 |

### 実測 —— 0.65 s は vLLM の内側だった

```
[gen-phases] rank 0, mean over 150 calls (s): to_device 0.00  preprocess 0.00
  generate 3.23  postprocess 0.00  to_cpu 0.00  worker-total 3.24
```

**4 つの脚が同時に消えた。** sharding manager の per-call 整形も、device への
転送も、detokenize も、CPU への戻しも **0.00 s**。さらに worker-total 3.24〜3.52 は
driver 側の `gen` 列(3.25〜3.43)とほぼ一致するので、**Ray の往復も実質ゼロ**である。

上の表に挙げた打ち手のうち「`raw_prompt_ids` だけ送る」「整形を削る」「to_cpu を
非同期化」は**すべて的外れだった**。padded tensor を毎 turn 送り直しているのは
事実だが、それは 1 秒も食っていない。

空いている 20% は **`vllm.generate()` の内側**、decode ステップの隙間にある。
起動ログが候補を名指ししている:

```
WARNING [topk_topp_sampler.py:69] FlashInfer is not available.
Falling back to the PyTorch-native implementation of top-p & top-k sampling.
```

Qwen3-1.7B の decode 1 ステップの GPU 仕事は重み 3.4 GB の読み出しで、A6000 の
帯域からおよそ 4〜5 ms。そこにステップごとの CPU 処理(vLLM のスケジューラ +
PyTorch-native サンプラ)が乗れば 20% の泡は説明がつく。**小さいモデル ×
GPU あたり 42 系列という、CPU 律速に落ちる典型的な条件である。**

### 打ち手

| | 見込み | コスト |
| --- | --- | --- |
| **FlashInfer を入れる** | ログが名指ししている fallback が消える。ステップごとの CPU が減る | `pip install flashinfer-python`。**コード変更なし** |
| **GPU あたりの系列数を増やす** | decode は帯域律速で幅に対して**劣線形**(11 系列 3.19 s 対 126 系列 5.47 s = 11.5 倍の仕事が 1.7 倍の時間)。3 倍幅にすれば throughput が 1.5〜2 倍 | task ごとの val batch size(search だけ)。ただし **KV cache の余裕を先に確認**すること —— `gpu_memory_utilization=0.6`、`max_model_len=4608`、Qwen3-1.7B は 1 token 約 0.115 MB なので、満長なら 1 系列 530 MB、GPU あたり 47 系列で頭打ちになる |
| `enable_chunked_prefill=True` | prefill と decode を同じステップに混ぜられる | 現在明示的に False。理由の確認が要る |

**util を上げるのではなく throughput を上げる**のが 2 つ目の要点である。帯域律速の
decode では幅を増やしても util はほぼ動かないが、同じ時間で処理する行数が増える。

## 9. FlashInfer —— util は上がり、遅くなった

`pip install flashinfer-python` で、起動時の
`FlashInfer is not available. Falling back to the PyTorch-native implementation`
は消えた。同じ checkpoint、同じ retriever、**同じ batch 番号**での比較:

| | genGPU% | batch#171 までの wall |
| --- | ---: | ---: |
| なし | 72〜80 | **2552.6 s** |
| あり | **86〜90** | **2661.6 s(+4.3%)** |

**GPU 占有率は 10 pt 上がり、スループットは 4.3% 落ちた。** 8 節で
「サンプラの fallback が泡の原因では」と読んだのは外れである。

これで **`genGPU%` を目的関数にしてはならない**ことが二度実証された(一度目は
5 節の `slots-busy`)。NVML が測るのは「kernel が走っているか」であって
「有用な仕事が進んでいるか」ではない。**判定は常に同じ batch 番号での累積 wall で
行うこと。**

なお この run は `VLLM_LOGGING_LEVEL=INFO` も同時に変えている(KV cache の値を
読むため)。4.3% がどちらの寄与かは切り分けていないが、**どちらも利得が無い以上
両方戻すのが結論**なので切り分けていない。

### 比較の作法 —— 単発の `s/batch last20` では足りない

同一 run の中で `s/batch last20` は **11.8〜16.3** まで振れる(search の batch ごとに
質問も応答長も違う)。20 batch の窓でも run 間比較には足りない。
**同じ batch 番号での累積 `wall=` だけが振れない** —— dataloader の順序は
決定的なので、`batch#171` はどの run でも同じ 126 行だからである。

```bash
grep 'batch#171 ' run_a.log run_b.log
```

完走を待つ必要もない。

## 10. 残っている最後の 1 手 —— KV cache が足りていない

```
GPU KV cache size: 159,600 tokens
Maximum concurrency for 4,608 tokens per request: 34.64x
```

**GPU あたり 42 系列(126 ÷ 3)を流しているのに、上限が 34.64 系列である。**
満長の系列が揃うと vLLM は preemption と再計算を起こす。8 節で見た decode の泡は
これかもしれない。

`gpu_memory_utilization` を 0.6 → 0.85 にすれば KV cache はおよそ 1.7 倍
(同時 ~57 系列)。**`val_only` は optimizer state を使わないので余地はあるはず**である。
`expected_multitask_sft_config.yaml` はこのキーを固定していないので、上書きは通る。

### 実測 —— 変わらなかった

`gpu_memory_utilization=0.85` で走らせ、**確率的な alfworld / webshop を挟まない
search 区間だけ**を切り出して比較した(`wall@40 − wall@5` = search batch 35 本):

| | search 35 本 |
| --- | ---: |
| 0.6 | **418.1 s** |
| 0.85 | **420.6 s(+0.6%)** |

**誤差である。** batch#40 の `span=22.3s` と `s/batch last20=12.3s` は両 run で
完全に一致した —— search は `do_sample=False` で決定的なので、同じ batch は同じ
時間で終わる。**KV cache を 1.7 倍にしても preemption の損失は測れなかった。**
34.64x という上限は、実際の系列長が 4,608 に遠く及ばないため効いていない。

設定は 0.6 のまま。

## 11. 締め

| 打ち手 | 結果 |
| --- | --- |
| turn ごとの vLLM wake/sleep を止める(session) | **採用** |
| batch ごとの wake/sleep を止める(hoist) | **採用**(10.4%) |
| retriever のバッチ化 | **採用**(envstep 10.60 → 1.00 s) |
| retriever のバージョンずれ + 自動再試行 | **採用** |
| `VAL_PIPELINE_DEPTH=2` | **採用**(17.0 → 14.2 s/batch) |
| FlashInfer | **不採用**(+4.3%) |
| `gpu_memory_utilization=0.85` | **不採用**(±0%) |
| task ごとの val batch size | 未着手(8 節の前提が崩れたため見送り) |
| trajectory 単位の連続バッチ | 未着手(残り 6% に対して rollout 中核の書き換え) |

**評価の wall 2.85 h → 1.71 h。**

残る非効率は `vllm.generate()` の内側の約 20% で、**設定では動かないことが
2 つの実測で確認された**(FlashInfer、KV cache)。ここから先はモデル側
(量子化、投機デコード)か、rollout の作り替えになる。

### この節で三度踏んだ穴

1. **占有率を目的関数にした。** `slots-busy=1.82x`(速度は同じ)、
   `genGPU%` 77 → 88(速度は 4.3% 低下)。NVML は kernel が載っているかしか見ない
2. **単発の値で run 間比較をした。** n=1 の完了間隔、20 batch の窓 —— どちらも
   同一 run 内で 11.8〜16.3 まで振れる
3. **span と s/batch を突き合わせた。** batch の中身と batch の間隔は別物で、
   その差(2.9 s)こそが pipeline の埋める対象だった

**正しい判定は一つだけ:同じ batch 番号での累積 `wall=` の差分を、決定的な
task の区間で取る。**

## 12. prefill と decode の比 —— まだ測っていなかった

11 節まで「検証は decode 中心」という前提で説明してきたが、**その比率は一度も
測っていない。** 指摘されて気付いた:検証も毎 turn prefill をしており、しかも
小さくない。

| | 1 回の forward のトークン数(GPU あたり) |
| --- | ---: |
| 学習(micro batch 10) | 10 × 最大 4,608 ≈ **最大 4.6 万** |
| 検証の prefill(42 系列) | 42 × プロンプト長 → 1,500 なら **6.3 万** |

**検証の prefill のほうが大きいことすらある。** 「学習は大きい 1 パス、検証は
小さいステップ」という整理は誤りだった。

### vLLM の統計ロガーは使えなかった

`disable_log_stats=False` は `vllm_rollout.py:133` でエンジンまで渡っており、
`VLLM_LOGGING_LEVEL=INFO` も効いている(`kv_cache_utils` の行は出る)。それでも
**`Avg prompt throughput` の行は 1 行も出ない**(21 batch 進んだ時点で 0 件)。
この経路は諦めた。

### 自分で数える

turn table に `promptTok` と `genTok` の 2 列を足した。どちらも
`attention_mask` の総和 —— エンジンに渡したトークンそのもの、
`pad_dataproto_to_divisor` の padding 行も含む(エンジンはそれも処理する)。
生成側は出力の総和から入力の総和を引くので、pad id の仮定が要らない。

**`promptTok` は prefill 仕事の上界であって、実測ではない。** prefix caching が
有効なので、共有接頭辞は再計算されずキャッシュから返る。渡したトークン数は
数えられるが、再計算されたトークン数は数えられない。したがって:

* `genTok` が `promptTok` を上回っていれば、**その turn は確実に decode 律速**
* 逆は言えない —— `promptTok` が大きくても、その大半がキャッシュ hit かもしれない

上界でも判定に足りるのは、打ち手が「幅を広げる」だからである。decode 律速なら
幅はほぼ無料、prefill 律速なら比例して高くつく。**上界で decode 律速と出れば、
それは確定である。**

**未測定。次の run で `TOKENS` の行を読むこと。**

## 13. 実測 —— 尻尾は decode 律速だった

`promptTok` / `genTok` を入れて、search batch 1 本の turn 別内訳:

| turn | active | gen | genGPU% | promptTok/行 | genTok/行 | **新規 prefill/行** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 126 | 0.91 | **44%** | 167 | 20 | 167 |
| 1 | 126 | 10.29 | 90% | 730 | 208 | ~563 |
| 2 | 32 | 5.53 | 90% | 1,330 | 198 | ~600 |
| 3 | 10 | 3.99 | 81% | 2,214 | 239 | ~884 |

search は履歴窓が滑らない(`history_length=4` = `max_steps=4`)ので、turn N の
プロンプトは turn N−1 の接頭辞を含む。**実 prefill は promptTok の差分**である。

まず、8 節以降で置いていた「プロンプト 1,500 トークン」は**外れていた**
(turn 0 で 167、9 倍違う)。

読めること:

1. **turn 0 の 44% は呼び出し固定費。** 0.91 s のうち 0.51 s が空き —— 実測の
   0.6 s とほぼ一致する。呼び出しが短いほど固定費の比率が上がる、という予測どおり
2. **turn 1〜3 は 81〜90%。** 呼び出しが長ければ固定費は薄まる
3. **尻尾は decode 律速で確定。** turn 3 は新規 prefill が 8,840 トークンしか
   ないのに 3.99 秒。prefill でこの時間は説明できない。**240〜500 回の decode
   ステップを、126 席のうち 10 席だけ埋めて回している**
4. **尻尾は generate 時間の 46%(9.52 / 20.7 s)を使い、仕事は 14%(42 / 294)**

7 節でこの可能性を「util が active に反応しないから」と棄却したのは誤りだった。
反応しないのは帯域律速だからで、**席が空いていないことの証明にはならない。**

## 14. task ごとの validation batch size

尻尾の席を埋める最小の手は、**search の batch を広げること**である。decode の
1 ステップは席の数にほぼ依存しないので、252 行にすれば同じステップ数で倍の行を
処理する。

### 実装

* `env.multitask.val_per_task_batch_size` が **数値 1 つ、または task ごとの
  マッピング**を取る(`max_steps` と `history_length` に前例がある)。
  名前のない task は `data.val_batch_size` のまま
* `verl/utils/val_batching.py` の **`TaskBatchSampler`** —— 行を task で
  グループ化し、その task の size で刻む。**全 task が同じ size なら sampler を
  作らず、従来の loader のまま**(他の arm は無影響)
* 環境 manager は task ごとに違う worker 数で建つ
* `WALL` 行に **`rows=` と `ms/row`** を追加

### なぜ sampler が要るのか

**いま single-task batch になっているのは規則ではなく偶然である。** loader は
`batch_size=126` で機械的に切っているだけで、alfworld と webshop がちょうど
126 行だから揃っている。**`val_batch_size` を 252 にすると最初の batch が
alfworld 126 + webshop 126 の混合になり、`get_task_names` が例外を投げる。**
sampler は task 境界で切ることでこれを規則にする。

### 採点は動かない

| task | 変わるもの | 採点 |
| --- | --- | --- |
| alfworld | 何も(manager 126、seed、game cycle そのまま) | **不変** |
| webshop | 何も | **不変** |
| search | まとめ方だけ(同じ行、同じ順序) | **不変** |

### 有効化には pinned config の編集が要る

`expected_multitask_sft_config.yaml` が
`"env.multitask.val_per_task_batch_size": 126` を固定している。これは正しい
——この値が alfworld の episode を決めるからである。**弱めない。** 有効にするには
そちらもマッピング形に書き換える。決定がコマンドラインではなくファイルに残る、
というのがこの pin の目的である。

### 比較の作法(更新)

widening は **413 batch を 208 batch に変える**ので、
**`s/batch` も batch 番号も同じ量ではなくなる。** 比較は `ms/row` で行う。

### 実測 —— 18% 速い

`Size of val dataloader: 208`、`rows=252`。search 区間だけを切り出した増分:

| | ms/row | 出典 |
| --- | ---: | --- |
| 126 行(depth 2) | **94.8** | `(793.0 − 374.9) s ÷ 4,410 行` |
| **252 行** | **78.0** | `(1218.4 − 413.0) s ÷ 10,332 行` |

`ms/row last20` も 72〜78 で安定し、126 行時の同区間 97.6(batch#40)を一貫して
下回る。**−18%。**

完走見込みは `(231 + 90 + 51,713×0.078) ÷ 3600` ≈ **1.21 h**(現行 1.71 h、−29%)。
差が増分の 18% より大きいのは、alfworld の 231 s が倍の行数に薄まるためである。

`slots-busy` は 1.77 → 1.60 に下がった。想定どおりで、batch が倍になれば
batch 間の隙間は相対的に薄まり、depth 2 が埋めるものが減る。

**採点は動いていない**(alfworld と webshop は 126 行のまま、search は同じ行を
同じ順序で、まとめ方だけ変えている)。

## 15. util を上げる 4 手

13〜14 節の 252 行 batch で、generate 中の空きは **4.06 s / 28.8 s = 14%**。

| turn | gen | util | 空き | 正体 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1.98 | 69% | 0.61 s | 呼び出し固定費 |
| 1 | 15.67 | 91% | 1.41 s | 固定費 + prefill のステップ境界 |
| 2 | 6.81 | 90% | 0.68 s | 固定費 |
| 3 | 4.38 | 69% | 1.36 s | 固定費 + **GPU 間の応答長不均衡** |

126 行では turn 0 が 44%、空き 0.51 s。**呼び出しが 2 倍長くなっても空きは
0.5〜0.6 s のまま**で、固定費の読みがここでも一致した。**固定費 0.6 s × 4 turn が
空きの約 6 割**である。

### ① chunked prefill と token 予算(設定のみ、未測定)

turn 1 は 186,764 トークンを投げており、GPU あたり 62,255。
`max_num_batched_tokens=8192` の予算では **prefill が 8 ステップに分割され**、
境界ごとにホスト処理が挟まる。`enable_chunked_prefill=True` は prefill と decode を
同じステップに混ぜるので、この隙間を埋める設定である。

```bash
-- actor_rollout_ref.rollout.enable_chunked_prefill=True \
   actor_rollout_ref.rollout.max_num_batched_tokens=32768
```

`enable_chunked_prefill=False` は誰かが明示的に選んだ値で、vLLM の版によっては
prefix caching と併用できない。起動時に落ちたらそれが理由である。

### ② 378 行(設定のみ、未測定)

turn 0 の 44% → 69% は固定費が薄まった結果なので、もう一段ある。ただし KV cache が
turn 1 で 156,500 / 159,600 と上限すれすれで、**preemption が起きれば逆効果**。

### ③ engine 境界での分割(実装済み、未測定)

固定費 0.6 s は `generate_sequences` の内側までは絞れたが、その先が
**「vllm.generate の中」なのか「その周りの Python」なのか**は分けていなかった。
verl の rollout wrapper が入力リストの構築・engine 呼び出し・出力の組み立てを
別々に計測する。出力の組み立ては応答ごとの Python ループと pad + concat で、
252 行 × 最大 512 トークンに対して自明に安いとは言えない。

```
[rollout-phases] rank 0, mean over 50 calls (s): build_inputs 0.031  engine 3.402  assemble 0.118  total 3.551
```

計測は `verl/utils/phase_timing.py` に切り出した。worker 側と重複しており、
**どちらのコピーもテストできなかった**(片方は vllm、片方は worker が要る)。

### ④ 順番待ちの generate を合流させる(実装済み・既定 OFF、未測定)

turn 3 の `perGPU% 61/89/56` —— 12 本を 3 rank に割れば本数は 4/4/4 で揃うが、
**揃っていないのは応答長**である。呼び出しは 3 rank 全部が終わるまで返らないので、
短い方の rank は待つ。空いた席は**別の batch からしか埋められない**。

pipeline には別の batch があるが、**わざと位相をずらして走らせている**(その
ずらしが 16.5% を生んだ)。だから**待たない**:`verl/utils/generate_merge.py` は
**既に順番待ちしている呼び出しだけ**を合流させる。worker group は Ray actor で
呼び出しを直列化するので、飛び込んできた 2 本目はどのみち待つ —— 合流させても
何も失わず、1 本目の空き rank にその行が乗る。

合流してよいのは「行以外が同一の呼び出し」だけなので、**sampling パラメータと
テンソル幅**を鍵にする。search の greedy と alfworld の temperature 0.4 は
決して同じ呼び出しに入らない。

**既定 OFF。** 行も sampling も同一とはいえ、採点経路で仮定によって有効化するもの
ではない。`ROLLOUT_MERGE_GENERATES=1` で入り、起動時に状態を必ず出す。

### ③ の実測 —— 0.6 s は vllm.generate の中で確定

```
[rollout-phases] rank 0, mean over 200 calls (s):
  build_inputs 0.000  engine 3.888  assemble 0.004  total 3.893
```

**engine の周りの Python は 4 ミリ秒である。** 出力の組み立て(応答ごとの
Python ループ + pad + concat)が固定費の正体ではないかという読みは外れた。

これで消去は完了した。sharding manager 0.00、device 転送 0.00、detokenize 0.00、
CPU 戻し 0.00、Ray 往復 ≒ 0、入力構築 0.000、出力組み立て 0.004。
**固定費 0.6 s は `vllm.generate()` の内側にある**、これ以上は verl 側からは
割れない。次に進むなら vLLM 内部のプロファイルである。

### ①+④ の実測 —— 効果なし

同じ batch 番号での累積 wall(#36):

| | wall | 由来 |
| --- | ---: | --- |
| 252 行のみ | ~1,004 s | #31 の 903.5 と #45 の 1,184.7 から内挿 |
| **+ chunked prefill + merge** | **1,018.4 s** | **+1.4%** |

`ms/row last20` は 74 で、252 行のみの run の 72〜78 の帯の中。
gen 加重の util は **86.0% → 84.8%** とわずかに低下。**どちらも利得なし。**

**そして ①(chunked prefill)と ④(merge)を同じ run で変えてしまった。**
このドキュメントで 2 度目の同じ失敗である。合算がゼロなので分離せずに両方
落とすが、片方が効いて片方が損をしていた可能性は排除できていない。

### ④ が本当に合流したのかは分からなかった

`GenerateMerger` は合流回数を数えていたのに、**それを出していなかった。**
「一度も合流しなかった」と「合流したが利得がなかった」が外から同じに見える
—— この arm が既に 3 回直した構図そのものを、自分で作っていた。
`[rollout-merge] N generate calls, M of them merged (X%), carrying R extra rows`
を 200 呼び出しごとに出すようにした。**次に ④ を試すときは、まずこの行を見ること。**

**② は未測定。判定は `ms/row` と `genGPU%` の両方で行うこと** ——
FlashInfer は util +10 pt で速度 −4.3% だった。

## 16. depth 3 + 合流 —— 23% 短縮、合流率 24.5%

④ が depth 2 で空振りしたのは「効かなかった」のではなく **待ち行列が 1 を超えず、
合流の相手が存在しなかった**からである。depth 3 にして初めて機会が生まれる。

| | ms/row | batch#45 の累積 wall | gen 加重 util | turn 0 util |
| --- | ---: | ---: | ---: | ---: |
| 252 行・depth 2 | 74 | 1,184.7 s | 84.8% | 79% |
| **252 行・depth 3 + 合流** | **57** | **1,020.6 s** | **86.7%** | **89%** |

**ms/row −23%、同 batch の累積 wall −13.9%。**

```
[rollout-merge] 200 generate calls, 49 of them merged (24.5%), carrying 6327 extra rows
```

**1 合流あたり平均 129 行。** 尻尾の 10〜20 行が別 batch の 252 行と同じ呼び出しに
乗っている、という狙いどおりの姿である。`slots-busy=2.32x`。

turn 0 の util は 126 行で 44%、252 行で 79%、ここで 89% —— 呼び出し固定費が
「幅」と「合流」の両方で薄まった結果で、固定費の読みがまた一致した。

### 副作用 —— driver 側の CPU 競合

| | depth 2 | depth 3 |
| --- | ---: | ---: |
| preproc | 2.7 s | **4.6 s** |
| envstep | 2.0 s | **3.1 s** |

3 slot 分の tokenize と env step が競合して 1.7 倍になっている。いまは generate の
裏に隠れているが、**depth 4 ではここが先に律速する**可能性が高い。

### 未決着 —— depth 3 と合流のどちらが効いたか

2 つを同じ run で入れた。合流率が 0 でないので**両方が寄与している可能性が高い**が、
比率は分けていない。分ける価値はある:**合流は採点経路で呼び出し内の行の組み方を
変える手**で、depth 3 は単に slot を増やすだけである。合流の寄与が小さいなら、
リスクの小さい方だけ残せる。

```bash
VAL_PIPELINE_DEPTH=3 ...（ROLLOUT_MERGE_GENERATES なし）
```

### 切り分けの実測 —— 23% は全部 merge、depth 3 単独はゼロ

| | ms/row | batch#25 の累積 wall |
| --- | ---: | ---: |
| 252 行・depth 2 | 74 | 794.0 s |
| **252 行・depth 3(merge なし)** | **75** | **788.7 s** |
| 252 行・depth 3 + merge | **57** | — |

**depth 3 単独は depth 2 と区別がつかない。** slot を増やしても generate 呼び出しは
worker group 上で直列のままなので、当然である。**depth 3 は throughput のレバーでは
なく、merger に相手を供給するためだけの前提条件**である。

したがって **load-bearing な部品は merger** であり、それは
**採点経路で呼び出し内の行の組み方を変える方**である。「リスクの小さい depth 3 だけ
残す」という逃げ道は無い。

### 採点の確認が要る

merge は greedy であっても、呼び出しごとの batch 形状を変える。reduction の順序が
変われば token が稀に変わりうる —— 126 → 252 の幅変更で既に踏んだのと同じ種類の
変化である。**merge あり / なしの run で `val/search/test_score` を wandb で
突き合わせること。** 一致すれば以後は気にしなくてよく、ずれるならその幅を
記録した上で採用可否を決める。

### depth 4 は試す価値がある

merge 率は depth 3 で 24.5%。**depth を上げれば待ち行列が長くなり、合流率は
上がるはず**である。ただし driver 側の CPU(preproc 4.6 s、envstep 3.1 s)が
depth 4 で律速に回る可能性がある。**判定は同じく ms/row と、`[rollout-merge]` の
合流率の両方で。**

## 17. 訂正 —— 機械の util は 86% ではなく 65.6% だった

wandb の system stream(15 秒点サンプル、108 点、26.8 分)を数え直した:

| | |
| --- | ---: |
| **3 枚の平均 util** | **65.6%** |
| 中央値 | 89% |
| **3 枚とも exact-0 のサンプル** | **23〜24%** |

分布は 90–100% が 43.5%、70–90% が 26.5%、そして **0–10% が 24.7%**。
起動の 1.3 分を除いても定常状態で 18% 前後が 0 である。

**これまで本文で `genGPU%` を「util」と呼んできたのは誤りである。** あれは
`generate_sequences()` 呼び出しの**中だけ**を測った値で、呼び出しの外側は 1 秒も
見ていない。「動いているときは 89%、wall の 1/4 は 3 枚とも完全停止」が実態で、
8 節以降の「util 86%」はすべてこの読み替えが要る。

15 秒サンプルが **exact-0** ということは、**15 秒以上まるごと止まっている**という
ことでもある。turn 間の小さな隙間ではない。

### 既存の計器では見えない状態

* per-turn table —— rollout の**中**しか測らない
* `slots-busy` —— slot の**平均**占有。「全 slot が空」を捉えない
* `ms/row` —— 速さは分かるが、止まっている場所は言わない

止まっている先を名指しできる計器が無かった。

### 入れたもの —— launch 区間の和集合

`run_pipelined` が各 launch の `(開始, 終了)` を記録し、最後に**和集合**を出す。
和ではなく和集合なので、2 slot が同じ秒を走っても 1 回しか数えない。
**どの launch にも覆われていない時間が、確実に「何も走っていない」時間**である。

```
[val-pipeline] 208 batches over 3010.5s: at least one slot running 2610.2s (86.7%),
  NOTHING running 400.3s (13.3%). Calling thread: prepare 180.1s, scoring 210.4s,
  waiting on a slot 2200.1s.
```

`prepare` は呼び出しスレッドの tokenize、`scoring` は retire 後に呼び出し側が
使った時間(252 行の decode + reward)。**この 2 つが長ければ、全 slot が
終わったまま次が投入されない**という、いま疑っている状態の直接の証拠になる。

**未測定。** 次の run でこの 1 行を読むこと。

## 18. 止まっているのは pipeline の外ではなく、slot の中でもなかった

同じ run(`sft-multitask-eval-20260825-211138`)を両方の計器で測った:

| 計器 | 値 |
| --- | ---: |
| wandb system stream(起動後) | 平均 **77.8%**、**exact-0 が 11.9%** |
| `run_pipelined` の被覆(和集合) | **何も走っていない 0.4%** |

**slot は走っているのに GPU が止まっている。** 被覆率は「slot が走っている」しか
見ないので、3 slot が同時に GPU の外(envstep / preproc)にいる状態を
「走っている」と数えてしまう。

数字で言うと、batch 1 本は preproc 3.9 + gen 44.4 + envstep 3.8 ≈ 52.4 s、
**GPU の外は 15.3%**。3 slot が独立なら「全部同時に外」は 0.153³ = **0.36%**。
観測は **11.9%、33 倍**である。偶然の重なりではない。

### retriever ではなかった

0 の塊が 8〜10 batch ごとに 15〜45 秒であることから、3 slot が共有する唯一の
外部資源 —— retriever —— を疑った。**外れ。** 46 batch の envstep は
**平均 3.8 s、最大 8.6 s**(17.5 は alfworld batch)で、15〜40 s の詰まりは 1 本も無い。

### 計測外だったもの —— loader

```python
for item in items:          # ← next(dataloader) はここ。計測外だった
    prepared = prepare(item)   # ← ここから測っていた
```

`num_workers=8` で、torch は worker から**厳密なラウンドロビン順**に batch を返す。
**8 batch ごとに一番遅い worker を待つ**構造で、観測された「8〜10 batch ごと」と
一致する。1 batch は 252 行の tokenize(最大 4096 トークン)である。

`dataload` と **その最悪値**を計測に追加した(平均だけでは周期的な待ちが
薄まって見えなくなる)。

### 実測 —— loader も外れ

```
Calling thread: dataload 1.7s (worst single wait 0.4s), prepare 5.8s,
  scoring 19.2s, waiting on a slot 734.7s.
```

**dataload は 25 batch で 1.7 秒、最悪 0.4 秒。** loader 仮説は死んだ。
(この節の草稿には説明用に 31.2s という架空の例が書いてあり、それが
実測値として引用される事故が起きた。**例示の数字を実測と同じ形式で書いては
ならない。**)

呼び出しスレッドの合計は 26.7s / 760.3s = **3.5%**。残る説明は一つ:
**GPU の外の処理が slot 間で取り合いになり、揃って遅くなる**。実測で
preproc は depth 2 → 3 で 2.7 → 3.9 s、envstep は 2.0 → 3.0 s(1.5 倍)。
独立なら伸びない。取り合って一緒に遅くなるから位相が揃い、3 slot 同時に
GPU の外にいる時間(wall の 10.6%)ができる。preproc は Python の tokenize
なので GIL、envstep は 3 倍のクエリを受ける retriever(どちらも推測、
プロファイル未取得)。まとめは `eval_performance_summary.md` §4。

## 19. 採点が 1 batch につき 504 回 detokenize して 503 回捨てていた

`agent_system/reward_manager/episode.py` が、**行ごとに prompt と response を
decode してから**、印字するかどうかを判定していた:

```python
prompt_str = self.tokenizer.decode(valid_prompt_ids, ...)     # 252 行すべて
response_str = self.tokenizer.decode(valid_response_ids, ...) # 252 行すべて
...
if already_print[...] < num_examine and np.random.random() < 0.1:   # 1 batch に 1 回
    print(f"[{data_source}][prompt]", prompt_str)
```

**採点自体は文字列を一切使わない** —— `episode_rewards` と `episode_lengths` を
読むだけである。文字列は印字のためだけに存在していた。252 行の batch なら
**504 回 decode して 503 回捨てる**、しかも **pipeline が batch 間に待つ
呼び出しスレッド上**で。

判定の中に移した。

これは 15〜45 秒の停止の説明にはならない(`scoring` は 25 batch で 17.5 s、
1 batch あたり 0.7 s)。**純粋な無駄で、直しても数%である。**

### 同時に確認できたこと

```
[rollout-phases] 300 calls: build_inputs 0.000  engine 4.374  assemble 0.005  total 4.379
[gen-phases]     300 calls: to_device 0.002  preprocess 0.002  generate 4.381
                            postprocess 0.001  to_cpu 0.001  total 4.388
```

**verl が engine の周りでしていることの合計は 14 ミリ秒。** 300 呼び出しの平均で、
8 節の 200 呼び出しと同じ結論である。

---

## 20. async —— engine をプールとして回す(実装、未計測)

`ROLLOUT_ASYNC_GENERATE=1`。**詳細は
[`eval_performance_summary.md` §7.5](eval_performance_summary.md) に書いた。**
ここには経緯だけ残す。

きっかけは 19 節までの分解のうち **(b) generate 呼び出しの中で GPU が空いている
11.6%** である。12 軌跡が 3 rank に散って per-GPU 61/89/56 という測定がその姿で、
座席は空いているのに、それを埋められる行は**別の batch にある**。

合流(`ROLLOUT_MERGE_GENERATES`)はこれを「呼び出しが偶然重なったとき」だけ
埋める仕掛けで、実測の発火率は 21.5%、効果は ms/row 75→57(−24%)だった。
async はその「偶然」を外す —— engine を 1 スレッドが回し続けるプールにして、
どの slot の行も常に同居させる。

**上位互換なので、両方を同時に入れる意味はない。** `ROLLOUT_ASYNC_GENERATE=1`
のときは合流は経路に入らない(起動時の `[rollout-pump]` 行がそう言う)。

### 詰まった点、と外し方

* **SPMD で per-request の投入ができるのか。** `distributed_executor_backend=
  "external_launcher"` かつ `tensor_model_parallel_size=1` なので、各 rank が
  丸ごとモデルを持つ独立した engine である。rank 間に collective が無いから、
  driver がどの rank に何を投げるか自由に決められる。**TP>1 ならこの設計は
  成立しない**ので worker 側で断る。
* **Ray actor はメソッドを 1 本ずつしか実行しない。** 投入と回収を別の呼び出しに
  分けると、回収が投入の後ろに並んで自分の待っているものを塞ぐ。だから
  `rollout_pump_step` は **1 呼び出しで投入と回収の両方**をする。
* **組み立てが 2 本になる危険。** token id → DataProto の算術(response の
  padding、position_ids の続き、eos での mask 打ち切り)は worker と driver の
  両方が要る。2 本持てば必ずずれ、しかも**ずれても例外は出ない** —— position_id
  が 1 ずれた batch で学習/採点が進むだけである。`generation_output.py` に
  1 本だけ置いて両方から呼ぶ形にした。

### 状態

**既定 OFF、効果は未計測。** merge と同じ理由で、`[val-hash]` は一致しない
(むしろ merge より徹底的に一致しない —— request の到着タイミングで decode step
の中身が決まるので、設定を固定しても走行ごとに変わる)。

採否の判定は §7.5 に書いたとおり **score でしか決められない**。そのために要る
数字は「同一設定 2 回の `val/*/test_score` の差」で、これは merge の判定と
**同じ 1 つの数字**である。

---

## 21. async の的は 0.4% だった —— 測ってから (a) に向き直る

### 実測 —— async は util を動かさなかった

step 100 の checkpoint を同じ構成で 2 本。wandb の system stream から:

| run | util | wall | util<10% のサンプル |
| --- | ---: | ---: | ---: |
| `...-131419`(async ON) | **55.70%** | 2.11 h | **32.7%** |
| `...-102303`(async OFF) | 57.91% | 2.23 h | 25.2% |
| `...-230457`(幅 252) | **77.88%** | **1.20 h** | 9.7% |

**async は util を上げていない。** wall は 2.23 → 2.11 h だが、この 2 本は
retriever の 4xx 件数も違う統制されていない比較で、noise と区別できない。

### なぜ動かなかったか —— §4 に書いてあった

`docs/eval_performance_summary.md` §4 の分解:

| 原因 | wall に占める空き |
| --- | ---: |
| (a) 全 slot が GPU の外(preproc + envstep) | 10.6% |
| (b) `vllm.generate` の中 | 11.6% |
| **(c) pipeline が空** | **0.4%** |

**async が消しに行ったのは (c) の 0.4% である。** この表は async を実装する前に
測って書いたもので、読み返さずに作った。実装は動いている(`[rollout-pump]` は
engaged、`slots-busy` も `ms/row` も期待どおり)。的が小さかっただけである。

`slots-busy=1.86x`(上限 2.00)が同じことを言っていた —— **slot は既に埋まって
いた**ので、消せる待ち行列が無かった。

### 教訓としては 3 度目

§17 と同じ形である。占有率を目的関数にすると、上がっても速くならない。
そして今回は**上がりすらしなかった**。順序は「分解を読む → 一番大きい項を狙う」で、
「作れるものを作る」ではない。

### 幅 252 が (a) を隠していた

`230457` の config を引くと
`val_per_task_batch_size = {search: 252, webshop: 126, alfworld: 126}` で、
これは §15 で測って **§7 の「いま最も速い設定」に書いたまま既定 OFF になっていた**
ものである。util<10% が 32.7% → 9.7% に落ちている。1 回の generate が長くなり、
その裏で env が回るので、(a) の空白が埋まる。**消してはいない、隠している。**

### (a) を消す方へ —— per-task advance

隠すのではなく消すには、GPU の外の処理を GPU の中の処理と重ねる必要がある。
lock-step の turn ループがそれを禁じていた(§7.6)。task ごとに turn を進める形に
して、alfworld の 50 turn 目と search の 4 turn 目を別スレッドの別の点にした。

**実装済み、既定 OFF、未計測。** 測る順序は:

1. 幅 252 だけ(async / per-task なし) —— `230457` の再現。基準
2. 幅 252 + per-task advance —— (a) がどれだけ埋まるか
3. 3 に async を足す —— per-task が立てる 3 本の generate を pump が並列に流す

1 を飛ばすと、また「どれが効いたか分からない」に戻る。

---

## 22. 外部レビュー —— 構造は合っていたが、落ちる経路が3本あった

async(§20)に外部レビューを受け、指摘をコードに当てた。**5件中3件が本物**で、
検証の過程で**レビューに無い4件目**と、**私が入れた修正自身のバグ**が出た。

### 本物だったもの

| | |
| --- | --- |
| **abort が例外経路で走らない** | `_fail_all` が `_pending` を空にしてから `_abort_outstanding` が読む。engine 例外のあと request が居座り、KV block を握ったまま次の rollout に持ち越す。**abort が要る唯一の場面で abort だけが起きない** |
| **submit と生存確認が別ロック** | 確認と挿入の間に pump が死ぬと `_fail_all` 済みの `_pending` に future が入り、誰も解決しない。待ち手は永久に待つ |
| **返らない request で永久停止** | `_pending` が空にならない → round も idle にならない → client は回り続け caller は止まったまま。timeout も watchdog も無かった |

### レビューに無かったもの —— request id の使い回し

id は `{name}-{n}`、`_next_id` は client ごとに 0 から。rank 側の `_pump_done` は
client より長生きする。**前 session の完了が次 session の別 request を、別の
token で解決しうる。** 例外は出ない。client ごとの epoch を id に混ぜ、stop 時に
queue を drain するようにした。

### 私の修正自身のバグ —— n サンプル検証の潰れ

レビューの「per-request seed が無い」に応えて prompt の token id から seed を
作った。ところが `val_kwargs.n>1` の検証は **DataProto 側で行を複製する**
(`test_batch.repeat(...)`)ので、n 個のプロンプトはバイト単位で同一 —— 同じ
seed、同じ draw、同じ action、同じ観測、同じ次プロンプト。**n 個の標本が 1 個の
n 複製に潰れる。** しかも標本分散が下がるので「安定した」ように見える。

seed に行番号を混ぜて直した。**「速くする変更」ではなく「正しさを守る変更」を
入れて壊した**ので、これは記録に残す価値がある。

### 前提が違っていたもの

* **「least-in-flight が prefix cache の sticky を壊す」** ——
  `dispatch_dp_compute_data_proto` は `chunk(world_size)` で**圧縮済み active 行**を
  等分する。軌跡が 1 本終わるたび以降の行の rank がずれるので、**blocking 経路も
  sticky ではない。** pump が壊したのではなく最初から無い。sticky routing は
  新規の打ち手であって回帰修正ではない。
* **「per-trajectory ではない」** —— そのとおり。§7.6 の per-task advance で
  task 単位までは進めたが、行単位ではない。

### 測定に足したもの

`ROLLOUT_ASYNC_REQUIRE=1`。pool に載らない呼び出しで落ちる。§17 と同じ罠
——「効かなかった」と「走らなかった」を取り違える —— を構造的に塞ぐ。
`ROLLOUT_ASYNC_GENERATE` を忘れた REQUIRE 単独は import 時に落とす(それ自体が
同じ穴になるので)。pool を一度も使わずに終わった run は、終了時にそう言う。

---

## 23. 原因は 37 日前から動いていた retriever だった

### 分解を取ったら、私の仮説が 3 つとも外れていた

`[rollout-turn-timing]` の TOTAL を初めて読んだ(rows=252、1 batch ≈ 70 秒):

| | 秒 | 割合 |
| --- | ---: | ---: |
| **envstep** | **~40** | **~57%** |
| gen(GPU busy) | ~27 | ~39% |
| preproc | ~2.0 | 2.9% |
| decode | ~0.15 | 0.2% |

```
SHARE  gen(GPU-busy)=30〜42%   cpu-glue=58〜70%
```

**preproc は 2.0 秒**である。§22 で「(a) の 58% が preproc だから preproc を
バッチ化する」と書いたが、それは §4 —— つまり **retriever が効いている run** の
分解であって、この run には当てはまらない。的は 3% しかなかった。

空きはほぼ全部 **envstep** だった。

### 原因はログに最初から書いてあった

```
batched search of 37 queries failed (... 422 Unprocessable Entity ...); retrying singly
http://100.86.45.30:8001/retrieve rejected a batched request but served the queries
  individually; batching disabled for it.
Connection pool is full, discarding connection: 100.86.45.30. Connection pool size: 512
```

`curl` で確定した:

```
{"query": ["a","b"]}   -> 422  {"type":"string_type","loc":["body","query"]}
{"queries": ["a","b"]} -> 422  {"type":"missing","loc":["body","query"]}
```

サーバのモデルが `query: str` 固定。**複数形も受けないので、クライアント側で
方言を合わせて回避する道も無い。**

`ps` を見ると答えが同じホストに立っていた:

| pid | port | 経過 |
| --- | --- | --- |
| 3080012 | **8001**(run が使っていた) | **37 日 17 時間** |
| 364474 | 8000 | 1 日 20 時間 |

`examples/search/retriever/retrieval_server.py:350` は
`query: Union[str, List[str]]` を持っている。8001 はそれが入る前のコピーだった。

### そして 77.88% の run だけが 8000 を使っていた

wandb の config を引いた:

| run | search_url | util | wall |
| --- | --- | ---: | ---: |
| `...-230457` | **:8000** | **77.88%** | **1.20 h** |
| `...-102303` | :8001 | 57.91% | 2.23 h |
| `...-131419` | :8001 | 55.70% | 2.11 h |
| `...-174359` | :8001 | 51.12% | — |

**一発で説明がつく。** §21 で「幅 252 のおかげ」と書き、次に「checkpoint の
違い」と書いた 22 ポイントの差は、**ポート**だった。

### 何を間違えたか

`API Request Error: 422` は**最初に貼られたログから出ていた**。私は
`grep -c` して「56 件、増えていないので一過性、続行」と判断した。
**件数だけを見て、その 422 が何を無効化しているかを読まなかった。**

その誤りの上に、async(§20)を作り、per-task advance(§7.6)を作り、
preproc のバッチ化を提案した。**どれも分解を取る前の推測に基づいていた。**
§21 で「順序は分解を読む → 一番大きい項を狙う」と書いた直後に、
その分解を取らずに 2 本実装している。

### 直した

* run script が `SEARCH_URL`(既定 `:8000`)を使うようにした
* **起動前に 2 クエリのリストを POST して、422 なら落とす**
  (`SEARCH_PREFLIGHT=0` で無効化)。`ROLLOUT_ASYNC_REQUIRE` と同じ理屈で、
  黙って 10 倍遅い経路に落ちるのを構造的に止める。3 分岐とも偽サーバで検証済み

見込みは envstep 40 → 約 4 秒、batch 70 → 約 34 秒、util 51 → 77% 前後。
そこまで戻して初めて §4 の分解が当てはまる状態になり、async と
per-task advance を測る意味が出る。

---

## 24. 窓が 10 ms では、252 クエリは 30 本に割れる

§23 で retriever を直したあと、search batch の turn 1 だけが **28.3 秒**残った。
turn 0 は 1.17 秒、turn 2/3 は 0.1〜0.8 秒なので、1 ターンだけが batch の 55% を
食っている形である。

retriever のログを見ると、届いていたのは **1〜14 クエリ**の細切れリクエストだった。
そして `encode` がクエリ数と相関していない:

| クエリ数 | encode |
| ---: | ---: |
| 3 | **42 ms** |
| 3 | **432 ms** |
| 9 | 39 ms |
| 9 | 283 ms |

同じ 3 クエリで 10 倍。処理量ではなく**待ち**である。

### 原因

`search/envs.py:69` は `max_workers = min(batch_size, 256)` で 252 行を同時に
投げる。だが GIL の下で `search()` に到達するのは 1 本ずつで、出揃うのに
**約 300 ms**。既定の窓 10 ms はその間に 30 回開閉し、1 本あたり 8 クエリしか
集めない。3 slot 同時なので最大 90 本が retriever の単一 GPU エンコーダを
取り合う。

### 実測 —— 窓だけ 10 → 100 ms

| | 窓 10 ms | 窓 100 ms |
| --- | ---: | ---: |
| turn 1 envstep | **28.32 s** | **0.31〜0.37 s** |
| `SHARE gen(GPU-busy)` | 37〜46% | **89.9〜94.0%** |
| cpu-glue | 55〜63% | **6.0〜10.1%** |

**80 倍。** 変えたのは環境変数 1 つで、生成には触れていない。既定を 100 ms にした。

### 残った空き

`genGPU%` は 87〜92%。cpu-glue が 6〜10% まで落ちた以上、残りは §4 の (b) ——
`vllm.generate` の内側である。`[gen-phases]` で verl 側は 14 ミリ秒と確定済みで、
**コードでは触れない。**

`gen` 列が同じトークン数に対して 7.9 / 15.5 / 23.9 秒と 3 倍振れるのは、3 slot の
generate が worker group 上で直列に並ぶためだが、**これは損失ではない** —— slot 3 が
待つ間 GPU は slot 1 と 2 を走らせており、`genGPU%` 90% がそれを示している。
`rollout_loop.py:250` の legend の通り、pipeline 下で span が膨らむのは占有であって
速度ではない。**したがって async を入れてもここは動かない。**

### 2 つの失敗は別物である

§23 は「機構が無効になっていた」、§24 は「有効だが足りていなかった」。
前者はログに 422 として出た。**後者は何も言わない** —— retriever のログに
`8 queries` と出ているのを読むまで、バッチ化は「効いている」ように見えていた。

---

## 25. どの計器にも映らない区間があった —— env reset

§24 のあと、wandb の分布はこうなっていた(起動 5 分以降、183 サンプル):

```
median 90.0%   mean 76.72%
   0-9  %:  9.8% のサンプル   ← 欠損の 41.9%
  10-49 %:  7.0%              ← 欠損の 21.3%
  80-89 %: 22.4%
  90-95 %: 53.8%
```

**定常状態は 90% に乗っている。** 平均を引き下げているのは、深いゼロの塊である:

| 時刻 | util | 電力 | メモリ |
| --- | ---: | ---: | ---: |
| 0.0〜1.0 分 | 0% ×5 | 94→**11 W** | 58.1% |
| 20.3〜20.8 分 | 0% ×3 | 100→**13 W** | 59.2% |
| 28.5〜29.0 分 | 0% ×3 | 101→**13 W** | 59.2% |
| 37.0〜37.3 分 | 0% ×2 | 99→**14 W** | 59.2% |

電力 12〜15 W はアイドル下限で、DMA も走っていない。メモリは動いていないので
モデルも vLLM も生きている。**30〜45 秒、GPU の外で何かが起きている。**

### どの計器も見ていなかった

* `[val-pipeline]`: 「どの slot も走っていない」は **3.9 秒 / 2639.7 秒 (0.1%)**。
  slot は走っていることになっている
* `WALL ... span`: `_batch_started` は `_reset_envs` の **後**で取られる
* turn table: turn 0 の preproc から始まる

**`envs.reset()` は 3 つのどれにも入らない。** 唯一の証拠が wandb の
system stream で、そこでは「機械が一瞬死ぬ」ように見えるだけだった。

一度 `span ≈ Σturns` を根拠に「reset は無い」と結論しかけたが、これは
`span` の定義からして常に成り立つ関係で、**何の検証にもなっていない。**

### 検証パスは prefetch を使っていない

`prefetch_env_reset` の呼び出し元は `rlsd_ray_trainer.py:303` と
`skillsd_ray_trainer.py:206` だけ。**`RayPPOTrainer._validate` は一度も呼ばない。**
検証は毎 batch の reset を同期で払っている。

### 計器を足した

`_reset_envs` を計時して、2 秒以上なら出す:

```
[env-reset] slot=primary  41.3s  SYNCHRONOUS (nothing overlapped it)
```

prefetch を消費できたのかどうかも言う。**打ち手を決めるのはこの数字を見てから。**
この session で 4 回、分解を取る前に実装して外している。

---

## 26. 止まっていたのは retriever ではなく、そこへの経路だった

§25 で `[env-reset]` を足したが、**空振りだった。** 答えは turn table にあり、
計器を足す前に読めば済んだ。5 回目の外れである。

### wandb の全系列で dip を挟む

| | dip(util<10) | busy(util≥85) |
| --- | ---: | ---: |
| GPU util | 0.0% | 90.3% |
| **GPU メモリコントローラ** | **0.0%** | 81.3% |
| GPU smClock | **1800**(ブースト) | 1740 |
| GPU memAlloc | 59.2% | 59.2% |
| host CPU | 0.3% | 0.3% |
| host スレッド / RSS | 868 / 4851 MB | 868 / 4824 MB |
| disk read | 0 | 0 |
| network | 1.14 MB/s | 1.12 MB/s |

CPU も disk も network も busy と変わらない。**ノード全体が待っている。**

### span が 3 連続で膨らむ

```
batch#49, 50, 51     93.1 /  93.9 / 100.3 s
batch#73, 74, 75     86.7 / 105.7 / 110.2 s
batch#101,102,103    77.8 /  78.8 /  83.6 s
```

3 連続 = 3 slot が同時。共有されているものが原因である。

### turn table が名指した

```
batch#74  turn 2:  gen 13.89   envstep 68.55     (他のターンは 0.08〜0.31)
batch#50  turn 0:  gen  1.76   envstep 46.01     (他のターンは 0.09〜0.31)
```

そしてログ:

```
Connection Error: [Errno 113] No route to host
Connection Error: ('Connection aborted.', ConnectionResetError(104, ...))
search recovered after 2 attempts (41s)   after 4 attempts (33s)   after 2 attempts (40s)
```

**41/33/40 秒 = dip の長さ。** バックオフ 1 秒で 2 試行なら、40 秒は connect の中。
`100.86.x.x` は CGNAT レンジなので、トンネル(Tailscale/WireGuard)の瞬断に見える。

`Connection pool is full` は **0 件** —— §24 の窓修正で枯渇は解消済み。別の失敗である。

### 1 行の 40 秒ではない

合流窓の follower は leader を**無制限に**待つ(`_Coalescer.call` の意図的な設計)。
詰まった connect 1 本が窓の全行を止め、3 slot が同じ retriever を叩くので 3 つとも
止まる。だから「ノード全体が 40 秒死ぬ」ように見える。

### 直したもの

`timeout=600` はスカラーで、requests はスカラーを connect と read の両方に使う。
**connect だけを 5 秒で切る**(`SEARCH_CONNECT_TIMEOUT_S`、0 で従来どおり)。
read はそのまま。

**根治ではない。** 落ちているのは `wasabi`→`wakaba` の経路で、5 秒の上限は
40 秒の停止を 6 秒に縮めるだけである。

---

## 27. 完走 —— 37 日ぶりのまともな評価と、残った 21 pt の所在

`sft-multitask-eval-20260826-201115` が完走した。§23〜§26 の 3 つの修正
(`:8000` への切替 / 窓 100 ms / connect 5 秒)がすべて入った最初の run である。

### 速度

```
[val-pipeline] final: 208 batches over 4534.0s:
  at least one slot running 4528.3s (99.9%), NOTHING running 5.7s (0.1%).
  dataload 10.7s, prepare 7.1s, scoring 81.2s, waiting on a slot 4435.5s.
```

**4534 s = 1.26 h。** 壊れた retriever の run は 2.23 h だったので **1.77 倍**。
出発点 2.85 h からは 2.3 倍。

### スコア(3 タスクすべて出た)

途中で `grep -c 'val/alfworld\|val/webshop'` が 0 だったのは、**タスクの順番**
であって欠落ではなかった。完走時の `Initial validation metrics`:

| task | test_score | success_rate |
| --- | ---: | ---: |
| alfworld | 3.649 | 0.643 |
| webshop | 5.938 | 0.659(task_score 0.823) |
| search | 0.352 | 0.384 |
| **全体** | | **0.387** |

search の内訳: nq 0.331 / triviaqa 0.508 / popqa 0.380 / hotpotqa 0.302 /
2wikimultihopqa 0.299 / musique 0.088 / bamboogle 0.278。

**これが 37 日ぶりに正しい構成で取れたスコアである。** §23 より前の run は
retriever が 1 クエリずつしか答えていないので、速度だけでなく検索結果の
質も違う。**比較の基準はこの run で、それ以前の値ではない。**

### util —— 同じ推定器で前後を並べる

wandb の events stream、15 秒サンプル全部の平均(単発値ではない):

| run | span | node util | 空き |
| --- | ---: | ---: | ---: |
| `...-102303`(`:8001`、窓 10 ms) | 2.23 h | 57.9% | 42.1 pt |
| `...-201115`(修正後) | 1.26 h | **79.0%** | **21.0 pt** |

**空きが半分になった。** GPU 別は 80.2 / 78.6 / 78.2% で、偏りはほぼない。

### 残り 21 pt の分解

| バケツ | wall | util | 空き | 割合 |
| --- | ---: | ---: | ---: | ---: |
| 起動(最初の 5 分) | 300 s | 48.4% | 3.4 pt | 16.3% |
| **ノード全停止**(3 枚とも 0%) | 285 s | 0.0% | **6.3 pt** | 29.9% |
| 走行中 | 3960 s | 87.0% | 11.3 pt | 53.8% |

### ノード全停止 19 サンプルの正体

§26 で「経路断」と当てた仮説を、完走 run 自身で検算した。停止サンプルと
走行サンプルを同じ時刻で突き合わせる:

| | ノード全停止 | 走行中 |
| --- | ---: | ---: |
| GPU power | **99 W** | 282 W |
| host CPU | 0.3% | 0.3% |
| プロセスのスレッド数 | 856 | 861 |
| network recv rate | 1.09 MB/s | 1.12 MB/s |
| disk read | 0 | 0 |
| RSS | 4908 MB | 4857 MB |

**GPU の消費電力だけが落ちて、他は全部同じ。** CPU も回っていない、
ディスクも読んでいない、ネットワークも流れていない、スレッドは全部生きている。
計算待ちでも I/O 待ちでもない。**小さな要求を出して答えを待っている**形である。
§26 の診断がそのまま確認された。

出現は 20.3〜21.8 分、27.0〜29.0 分、37.0〜37.8 分、50.2 / 53.5 / 57.8 / 69.5 /
71.3 分 —— **run 全体に散っていて、特定のバッチに寄っていない。**
これも経路の瞬断らしい形である。

### 計器の限界を 1 つ記録する

`[val-pipeline]` は `NOTHING running 5.7s (0.1%)` と報告するが、
NVML は同じ run で **285 秒**のノード全停止を見ている。

**矛盾ではない。** search の中で止まっている slot は pipeline から見れば
「slot 実行中」であり、GPU は空である。**pipeline 計器はこの種の停止を
原理的に見られない。** §18 で「止まっているのは pipeline の外ではなく、
slot の中でもなかった」と書いたのは、この盲点のことだった。
**外部の待ちを疑うときは NVML を見る。**

### 打ち手として残っているもの

| 空き | 相手 | このリポジトリで直せるか |
| ---: | --- | --- |
| 6.3 pt | `wasabi`→`wakaba` の経路断 | **いいえ**(ネットワーク側) |
| 3.4 pt | 起動(model load + engine init) | 長く回せば薄まるだけ |
| 11.3 pt | 1.7B の decode 1 step が短すぎる | いいえ(§4 (b)) |

**最大の 1 手はコードではない。** 経路が直れば 79.0% → 85% 前後になる。
async(§20〜§21)も per-task advance(§22)も、この 6.3 pt には触れない。

---

## 28. 残り 21 pt に手を入れる —— 40 秒はどこで消えていたのか

§27 で分けた 3 つのバケツに、それぞれ何ができるかを決める。

| 空き | 相手 | 打ち手 |
| ---: | --- | --- |
| 6.3 pt | ノード全停止 | **socket を縛る**(下記) |
| 3.4 pt | 起動 5 分 | **なし**(下記) |
| 11.3 pt | 走行中 87.0% | KV の余白を作る + 幅を広げる |

### 6.3 pt —— 「2 回」が犯人を特定した

§26 では 40 秒を **connect の中**と書いた。**これは間違いである。**

ログは `search recovered after 2 attempts (41s)`。**2 回**である。

経路が本当に 40 秒落ちていたなら、`EHOSTUNREACH` は**待たずに即座に返る**ので、
このバックオフ(1, 2, 3, … 秒)で 40 秒を埋めるには **9 回**ほど要る。
2 回で済んだということは:

- 1 回目が **1 本の socket の中で 40 秒すべてを使った**
- 2 回目は新しい接続で**即座に成功した**

つまり**経路の断ではなく、詰まった 1 本の接続**である。

### なぜ既存の 2 つの上限が効かなかったか

| 上限 | 守る範囲 | この故障に効くか |
| --- | --- | --- |
| `SEARCH_CONNECT_TIMEOUT_S=5`(§26) | connect が完了するまで | **いいえ** —— connect は成功済み |
| TCP keepalive(idle 30s) | **idle** な socket | **いいえ** —— 要求が出ている socket は idle ではない |

要求を送って ACK が返らない socket は、どちらの管轄でもない。Linux は
`tcp_retries2` に従い、**約 15 分**。今回は tunnel の瞬断が先に終わったので
40 秒で済んだにすぎない。

### 入れたもの —— `TCP_USER_TIMEOUT`

**未応答のデータを socket が抱えていられる上限。** これがちょうどこの穴である。

```
SEARCH_TCP_USER_TIMEOUT_S=10   (0 でカーネル既定)
```

40 秒 → 約 11 秒(10 秒 + バックオフ 1 秒で再試行、新しい接続で成功)。
**6.3 pt のうちおよそ 4.6 pt。**

あわせて `env.search.timeout` を **600 → 120** にした。600 は「永遠に待つ」の
綴りであって上限ではない。socket 層が 5 秒/10 秒で縛られたいま、この値が守るのは
**「接続は生きていて、サーバが考え込んでいる」**場合だけである。実測の最悪値は
バッチ検索 7.5 秒なので、120 はその 16 倍。`max_retries=null` は据え置き ——
諦めないことと、詰まりに気づくのが速いことは両立する。

### 3.4 pt —— これは手を入れない

最初の 5 分は model load + vLLM engine init(CUDA graph capture 含む)。
削るには「checkpoint ごとに新しいプロセス」をやめるしかないが、それは
alfworld の game cycle が seed 位置から再構築されるための条件で、
**評価の正しさそのもの**である(`eval_checkpoints.sh` の冒頭)。
1 checkpoint 4534 秒のうち 300 秒。**払う。**

### 11.3 pt —— カードを評価に渡す

`gpu_memory_utilization=0.6` は**学習アームの数字**であって、慎重さではない。
学習は FSDP パラメータ・勾配・optimizer state・136.5 GiB の teacher pool を
同じカードに載せている。**評価プロセスはそのどれも持たない。**

同じ 0.6 のまま走っていたので、評価はカードの約 1/3 を空けたまま走っていた。

```
ROLLOUT_GPU_MEM_UTIL=0.75   (eval_checkpoints.sh で既定、学習は 0.6 のまま)
```

実測の錨: 0.6 で KV 予算は **159,600 token/GPU**。予算 28.8 GiB のうち
KV 以外が 10.9 GiB(重み + profiling ピーク)。0.75 なら 36 GiB で
KV は **約 224,000 token**、カードには 12 GiB 残る。

**これは今のバッチを速くしない。** 最も重い search turn は 159,600 のうち
118,000 しか使っていない(74%)。**次の幅のための余白**である。

### 幅 252 を既定に —— コマンドラインから設定ファイルへ

`val_per_task_batch_size` は 413 batch → 208 batch、ms/row 不変、と**測って**
あるのに、`EXPECTED_CONFIG_WAIVE` の裏でコマンドラインに住んでいた。
**pinned config が防ぐためにある配置そのもの**なので、両方に書き下ろした。

これは今の運用の変更ではない —— **いま走っているのがこれである**。

### 次の 1 変数は 378

252 で 118,000/159,600 = 74%。378 なら約 156,500 で、0.6 の 159,600 に対しては
**危険なほど近かった**から見送った。0.75 の 224,000 に対しては **70%** ——
252 と同じ安全域である。**他を何も変えずに**これだけを動かすこと(罠 4)。

### waiver の副作用を先に潰した

pinned config は mapping を**平坦化して**保存する
(`val_per_task_batch_size.alfworld` など 3 キー)。
そのため scalar → mapping にした瞬間、親名の waiver が
「pin されていないキーを名指ししている」として**run を落とす**。
向きが完全に逆なので、waiver が**ドット境界で子を覆う**ようにした
(`a.b` は `a.b.c` を覆い、`a.bc` は覆わない)。

### この run が測るもの

**socket の上限だけ。** KV の余白は余白であって、今のバッチでは何も動かない。
幅 252 は既に走っている値。**node 全停止のサンプル数**が減れば当たり、
減らなければ外れ —— §27 と同じ推定器で数えられる。

---

## 29. 計器が2つとも見ていなかったもの —— 「何枚のカードに仕事があったか」

§28 の socket 上限を入れた run を 16 分で測ったら、**停止は減っていなかった**。
同じ窓で比べる:

| | 前回 `...-201115` | 今回 `...-074421` |
|---|---:|---:|
| ノード全停止サンプル | 1 | **3** |
| 停止イベント | 1 | **2** |
| 走行中 util | 81.5% | 84.3% |

イベント 1 対 2 では何も言えない。**だが、そこが問題ではなかった。**

### 3 枚のうち何枚が働いていたか

起動後の全サンプルを「50% 以上のカードが何枚あったか」で分けた:

| 忙しいカード | `...-201115` | `...-074421` | 失われる util |
| ---: | ---: | ---: | ---: |
| 3 枚 | 85.9% | 81.4% | 0 pt |
| 2 枚 | 3.2% | 4.3% | 1.1〜1.4 pt |
| 1 枚 | 3.5% | 5.7% | 2.4〜3.8 pt |
| **0 枚** | **7.4%** | **8.6%** | **7.4〜8.6 pt** |

**そして 3 枚とも忙しい 85.9% の間ですら util は 87〜90% である。**

つまり空きは2種類に分かれる:

- **約 11 pt** —— カードが 3 枚とも埋まっていない(0/1/2 枚)。**別のバッチで埋まる**
- **約 9 pt** —— 3 枚とも埋まっていて 87〜90%。**decode step の duty cycle**

### 87〜90% は物理である

1.7B の decode 1 step が約 4.5 ms、その間に host 側処理が約 0.6 ms。
`4.5 / (4.5 + 0.6) = 88%`。**観測値とそのまま一致する。**
NVML の `utilization.gpu` は内部窓に対する時間割合なので、この duty cycle が
そのまま出る。**1.7B の decode で 100% は物理的に出ない。**
埋めるには step を長くするしかなく、それはモデルサイズの話である。

### 2 つの計器が両方ともこれを見られない

| 計器 | 何を見る | この空きが見えるか |
| --- | --- | --- |
| `[val-pipeline]` | slot が batch の中にいるか | **×** env.step で止まった slot は「実行中」 |
| `genGPU%` | `generate` の中の util | **×** generate と generate の間は範囲外 |

`[val-pipeline]` は前回 run を `NOTHING running 5.7s (0.1%)` と報告したが、
NVML は同じ run で **285 秒**のノード全停止を見ている。

> **この盲点が判断を1つ狂わせた。** async(§20〜§21)の的を「0.4%」と
> 見積もったのは `[val-pipeline]` の数字からである。**device に聞けば 7.4% だった。**
> 「async は割に合わない」という結論は、**壊れた計器の上に立っていた。**

### 入れた計器 —— `[gpu-residency]`

既存の NVML sampler(0.3 秒間隔、wandb の 15 秒よりはるかに細かい)に
`residency_between()` を足し、`[val-pipeline]` の**隣に必ず出す**:

```
[gpu-residency] 3612s sampled: 3gpu 85.9% (3103s), 2gpu 3.2% (116s),
                1gpu 3.5% (126s), 0gpu 7.4% (267s) | EMPTY 7.4% -- this is
                what more batches in flight can recover
```

**0 枚と 1 枚を分けているのが肝。** 1 枚だけ働いているのは負荷の偏りで、
バッチを増やしても直らない。混ぜると処方を間違える。

### 機構 —— depth は幾何的に効く

slot は generate(GPU)と env.step(GPU外)を交互にやる。GPU外の割合を p とすると、
**全カードが空になるのは全 slot が同時に GPU 外にいるとき**で、slot は独立なので
確率は **p^depth**。

実測 depth 2 で 0 枚 7.4〜9.3%(2 run)⇒ **p ≈ 0.27〜0.31**:

| depth | 全カード空(p=0.27) | (p=0.31) |
| ---: | ---: | ---: |
| 2 | 7.4% | 9.3% |
| **3** | **2.0%** | **2.8%** |
| 4 | 0.5% | 0.9% |

**socket 上限を入れた run でも 0 枚は 9.3% で、減っていない。**
それが p の本体は病的な停止ではなく**普通の env.step** だという証拠である
—— 上限は裾しか消さない。だから depth なのである。

**depth 2 → 3 で約 5.4 pt。** このリストのどの打ち手より大きい。
socket 上限が消すのは p の病的な裾で、depth が消すのは p の本体である。

**生成は変わらない。** batch は自分の行を持ち、worker group は generate を
1 本ずつ走らせ、退役は投入順、集計は呼び出しスレッド —— depth が変えるのは
「どの batch が env を待っている間にどれが生成しているか」だけである
(生成を変えるのは `ROLLOUT_MERGE_GENERATES` で、だから既定にしていない)。

`VAL_PIPELINE_DEPTH` の既定を **3** にした。追加コストは search 環境 1 組と
その KV cache で、それは §28 の `ROLLOUT_GPU_MEM_UTIL=0.75` が払っている。

### 到達可能な天井

| | |
| --- | ---: |
| いま(depth 2) | 79〜81% |
| depth 3 で 0 枚を 7.4% → 2.0% | **約 86%** |
| depth 4 でさらに | 約 87% |
| **decode duty cycle の天井** | **約 88〜90%** |

**100% は 1.7B では出ない。** 出せるのはここまでである。

---

## 30. 「88〜90% が天井」は間違い —— 空きは3層で、層ごとに相手が違う

§29 で「1.7B の decode duty cycle だから 88〜90% が天井」と書いた。
**天井ではない。** duty cycle は物理ではなく、**engine のホスト処理が
kernel と kernel の間に露出している**というだけで、隠す機構は存在する。

空きは3層あり、**層ごとに効く打ち手が違う**。混ぜていたのが間違いだった。

| 層 | 実測(depth 2) | 何が起きているか | 消すもの |
| --- | ---: | --- | --- |
| **EMPTY** 0 枚 | 7.4〜9.3% | 全 slot が env.step | **depth**(p^depth) |
| **PARTIAL** 1〜2 枚 | 6.7〜8.4% | rank が自分の chunk を終えて、**collective 呼び出しの中で最遅 rank を待っている** | **pump のみ** |
| **DUTY** 3 枚で 87〜90% | 約 10 pt | engine の 1 step ごとのホスト処理 | **engine 側の overlap** |

### PARTIAL は depth では消えない —— これが重要な訂正

`generate_sequences` は **worker group の collective 呼び出し**で、
`Dispatch.DP_COMPUTE` が行を 3 rank に分ける。各 rank の `vllm.generate` は
**自分の chunk が全部終わってから**返り、呼び出し自体は**最遅 rank を待つ**。
その間、先に終わった rank は空である。

**worker group は一度に 1 呼び出ししか走らせない。** だから slot B の generate は
slot A の generate が返るまで始まらない —— **depth を上げても、空いた rank に
次の仕事を渡せない。**

per-GPU 平均は 80.2 / 78.6 / 78.2(spread 2 pt)で、**特定の rank が遅いのではない。**
毎回ちがう rank が順番に待っている。collective の構造そのものである。

**これを消せるのは pump だけである。** engine をプールとして回し、request を
個別に投げるので、chunk を終えた rank が次の slot の request をすぐ受け取れる。

> **§21 で async の的を「0.4%」と見積もったのは撤回する。**
> あの数字は `[val-pipeline]` の `NOTHING running` から出したもので、
> あの計器は env.step で止まった slot を「実行中」と数える。
> **device に聞けば PARTIAL だけで 6.7〜8.4% ある。**
> async は「割に合わない」のではなく、**割に合うかどうかを測れていなかった。**

### 計器に両方を出させる

`[gpu-residency]` は EMPTY と PARTIAL を**分けて**印字する。処方が違うからである。

```
[gpu-residency] 3612s sampled: 3gpu 85.9%, 2gpu 3.2%, 1gpu 3.5%, 0gpu 7.4%
                | per-gpu 80 79 78 (spread 2 pt)
[gpu-residency] EMPTY 7.4% (more batches in flight fill this),
                PARTIAL 6.7% (a rank idle inside a collective call -- only the
                pump fills this), rest is the engine's own duty cycle
```

per-GPU の spread が大きければ**特定 rank の遅れ**、小さければ**collective の構造**。
prescription が変わるので、同じ行に出す。

### DUTY 層 —— vLLM 側で隠せるものが残っている

1 step の内訳は概ね: schedule(Python)→ input 構築 → **GPU 実行** →
output 処理(sampling / stop 判定 / 追記)。GPU 実行が約 4.5 ms、
残りが約 0.5〜0.8 ms 露出して 87〜90% になる。

**すでに取ってあるもの:**

| | 状態 |
| --- | --- |
| `enforce_eager=False`(CUDA graph) | **ON** |
| `detokenize=False` | **ON**(step ごとの逐次 detokenize を止めてある) |
| `logprobs=None` | **ON**(`return_rollout_log_probs=False`) |

**まだ見ていないもの:** vLLM の **scheduler と GPU 実行の overlap**
(V1 の `async_scheduling`)、および `enable_chunked_prefill=False` の是非。
`vllm_rollout_spmd.py` の `engine_kwargs.vllm` は**任意の engine 引数を通す
素通しの口**なので、コード変更なしに渡せる。存在しない引数は `LLM()` が
`TypeError` で落とすので、黙って無視されることはない。

**入っている vLLM が何を持っているかを先に見ること:**

```bash
python - <<'PY'
import inspect, vllm
from vllm import LLM
print("vllm", vllm.__version__)
sig = set(inspect.signature(LLM.__init__).parameters)
from vllm.config import SchedulerConfig
sched = set(getattr(SchedulerConfig, "__dataclass_fields__", {}))
for name in ("async_scheduling", "num_scheduler_steps", "disable_async_output_proc",
             "enable_chunked_prefill", "max_num_seqs"):
    where = "LLM()" if name in sig else ("SchedulerConfig" if name in sched else "ABSENT")
    print(f"  {name:28s} {where}")
PY
```

**推測で渡さない。** バージョンで消えた引数(V0 の `num_scheduler_steps` は
V1 で削除)を渡すと落ちるだけで、時間を失う。

### 手で叩かなくても、run が自分で答えるようにした

engine を組んだ直後に `[rollout-engine]` を印字する。**次の run のログに
そのまま出る**ので、上のコマンドを叩く必要はない:

```
[rollout-engine] vllm 0.11.0; overlap knobs: async_scheduling=<default> (available),
  num_scheduler_steps=absent, disable_async_output_proc=absent,
  enable_chunked_prefill=False (set here), max_num_seqs=1024 (set here),
  enforce_eager=False (set here)
```

3 状態を区別する: **`(set here)`** = 我々が選んだ値、
**`<default> (available)`** = この vLLM にはあるが我々は触っていない、
**`absent`** = このバージョンには存在しない。

`enable_chunked_prefill` などは `engine_kwargs` ではなく `LLM(...)` に直接
渡っているので、それも `explicit` として渡してある —— **選んだ値を
「default」と印字するのは、この行が防ごうとしている嘘そのものである。**

診断は `verl/utils/engine_overlap.py` に置いた。`vllm_rollout` パッケージは
vLLM が入っていないと import 自体が失敗するので、そこに置いた診断は
GPU イメージなしにテストできない —— つまり腐る。

### 実測(2026-08-27、この機械)—— **DUTY 層に回せるつまみは無かった**

```
vllm 0.8.5
  async_scheduling             ABSENT
  num_scheduler_steps          SchedulerConfig
  disable_async_output_proc    LLM()
  enable_chunked_prefill       SchedulerConfig
  max_num_seqs                 SchedulerConfig
```

| つまみ | この版での状態 | 使えるか |
| --- | --- | --- |
| `async_scheduling` | **無い**(0.10.2 以降で追加) | **×** |
| `num_scheduler_steps` | config には在るが **V0 専用** | V1 core なら無視される |
| `disable_async_output_proc` | `LLM()` に在るが **V0 専用**、既定で既に有効 | 触っても変わらない |

**つまり 0.8.5 で DUTY 層に効かせる手は無い。** 残る道は vLLM の更新で、
`setup.py` は `<=0.11.0` を許しているので**可能ではある**が、それは
**別の実験**である。今回の run に混ぜてはいけない(罠 4)。

**V0 か V1 かで意味が変わる**ので、`[rollout-engine]` に core を出すようにした:

```
[rollout-engine] vllm 0.8.5, core=v1; overlap knobs: async_scheduling=absent, ...
```

`VLLM_USE_V1` は**要求であって結果ではない** —— V1 が扱えない構成では V0 に
落ちるのに変数は 1 のままなので、**env var を読んではいけない。**
実際に組まれた engine のクラスだけが嘘をつけない。

**DUTY 層の結論は「まだ測っていない」から「この版には手が無い」に変わった。**
EMPTY(depth)と PARTIAL(pump)は残っており、そちらが先である。

### 到達可能な数字を並べ直す

| | 全カード空 | 部分空 | duty | node util |
| --- | ---: | ---: | ---: | ---: |
| いま(depth 2) | 7〜9% | 7〜8% | 10 pt | **78〜81%** |
| depth 3 | 2〜3% | 7〜8% | 10 pt | 約 86% |
| depth 3 + pump | 2〜3% | ~0% | 10 pt | 約 90% |
| + engine overlap | 2〜3% | ~0% | ? | **測ってから言う** |

**「88〜90% が天井」は取り消す。** depth と pump で 90% 前後までは構造で取れる。
その先は engine の中の話で、**まだ測っていない。**
