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

### 打ち手は 2 つ。片方はコード変更ゼロ

**(a) retriever を増やす。** `env_config.search.search_url` は **リストを受け、
env を round-robin で振り分ける**(`envs.py:54-66`)。レプリカを 2〜3 本立てて
リストで渡すだけで、126 並行の負荷が分散する。**コード変更なし。**
2.65 秒がサーバ飽和由来なら、ここがいちばん安い。

**(b) cohort overlap。** search は `gen 3.06 / envstep 2.65` と**ほぼ釣り合って
いる**ので、overlap の理想形に最も近い。2 cohort に割って
`gen(A) ‖ step(B)` を組むと 1 batch 24.2 → 16.2 秒(**−33%**)、
評価全体で **2.85 h → 1.93 h**。

(b) には依然として subset step が要る(`SimpleMemory` と manager の positional
state を index 対応にする)。ただし search の leaf は `self.envs` が独立な
`SearchEnv` のリストで executor から index 指定で叩かれているだけなので、
**alfworld より素直である**。そして 42.6% の 99.5% が search にある以上、
**手を入れる先は search だけでよい** —— 当初「全 manager を index 対応に」と
見積もったのは、ここでも加重する前の話だった。

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
