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

## 5. 実装していないもの —— episode 単位の async loop

GPU が「どれかの episode が生成中なら常に busy」になる唯一の形。
`agent_system/multi_turn_rollout/async_rollout_core.py` に、同期 loop と
同一軌跡を collect することを CPU で証明する制御フロー核が既にあり
(`tests/ray_cpu/test_async_rollout_equivalence.py` が 6 件で固めている)、
docstring は「次に `async_rollout_loop.py` を足す」と書いている。

その統合ファイルは**存在しない**。そして繋ぐ相手の `vLLMAsyncRollout`
(`vllm_rollout_spmd.py:410`)は verl の **agent-loop / async server 経路**用の
SPMD ラッパであって、この rollout loop に差せる AsyncLLM ではない。
worker group の駆動方法ごと変わる。

**3 節の cohort より大きい変更で、3 節の測定結果が前提になる。**
順序としては後である。

---

## 6. 現状の台帳

| 項目 | 大きさ | 状態 |
| --- | --- | --- |
| **search の retriever 待ち** | **42.6%** | **未着手。最大項。** 打ち手は 3 節 —— 実測 80 ms/クエリが 126 並行で 7.5 秒(93 倍)、実効並列度 1.35、16.8 クエリ/秒。retriever は事実上シングルスレッドで、**別マシン**(100.86.45.30、wasabi は .24)なので worker を増やしても eval と競合しない |
| batch ごとの vLLM wake/sleep | 10.4% | **解消**(2 節の hoist、413 回 → 1 回) |
| turn ごとの vLLM wake/sleep | — | **解消**(2 節の session) |
| preproc(CPU tokenize) | 5.2% | overlap しない限り露出する |
| generate | 51.8% | GPU が実際に働いている区間 |

### 到達点(モデル、未実測)

| | wall | GPU util |
| --- | ---: | ---: |
| 現状 | 2.85 h | 51.8% |
| retriever を並列化 | 1.67 h | 88.2% |
| + session hoist(**実装済み**) | 1.50 h | 98.5% |
| batch をまたぐ async pipeline | 1.48 h | 99.9% |

**評価の理論上限は 99.9% である。** 「環境との往復だから 90% が限界」と一度書いたが、
それは誤りだった —— 短くできないもの(retriever の 80 ms、decode、preproc)と
**隠せないもの**は別で、裏で generate が回っていれば全部隠れる。本当に隠せないのは
**pipeline の fill と drain だけ**で、それも batch 境界をまたげば run 全体で 1 回に
なる(いまは 413 回払っている)。

ただしその道には条件がある: **retriever が generate に追いつくこと。**
評価全体の retrieval は 73,123 クエリで、generate の総時間 5,316 秒に収めるには
**13.8 クエリ/秒**が要る。実測 16.8 なので追いつくが、稼働率 82% は薄い。
retriever を直せばここに余裕が生まれ、pipeline が安定する。

**実務上は上の 2 つ(retriever + hoist)で 98.5%。** 3 つ目が買うのは残り 1.4 pt で、
`async_rollout_loop.py` の新規実装(vLLM AsyncLLM 経路、env の部分 step、
`_validate` が batch 境界をまたいで結果を集める形への再設計)が要る。
