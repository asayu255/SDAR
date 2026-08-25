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
