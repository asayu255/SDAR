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

## 2. 13% —— vLLM が毎 turn 寝起きしている

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

**原因そのものはまだ特定できていない。** 次の eval が
`[rollout-session]` の 2 行で確定させる。

---

## 3. env.step の比率 —— 最初の見積もりは 1 batch からの外挿だった

`vanilla_multi_turn_loop` の 1 turn が完全直列であることは変わらない:

```
preprocess → generate(GPU) → decode → envs.step(CPU/IPC/HTTP)
```

`envs.step` の間、vLLM は 28.5 GB 常駐したままカーネルが 1 本も走らない。
既存の overlap 機構(`_env_step_executor` による logprob prefetch)は
`is_train` 必須なので評価では死んでいる。

**ただしその大きさを 42〜50% としたのは誤りだった。** turn timing を入れて
測った実測(alfworld batch):

```
TOTAL             11.2    201.0      1.4     17.0    230.6
SHARE  gen(GPU-busy)=87.1%  cpu-glue(preproc+decode+envstep, GPU-idle)=12.9%
```

**envstep は 17.0 / 230.6 = 7.4%。** そして gen が wall の 87.1% を占める
一方で、NVML は同じ run の GPU busy を 46.6% と読む —— つまり
**`generate_sequences` の中で GPU が半分空いている**。

**この表は 413 batch のうちの batch 0 だけ**、つまり run の 0.24% である。
4 節の通り 411/413 は search batch で、その `envs.step` は retriever への
HTTP 呼び出しであり、alfworld の表には原理的に写らない。1 節の
「engine 常駐 + util 0 が 42.9%」はほぼ全部そちらのはずだが、**まだ
測っていない**。

判断に必要なものは走行中のログに既にある。turn timing は batch ごとに出るので:

```bash
grep -rhE "TOTAL +[0-9]|SHARE  gen" /tmp/ray/session_latest/logs/*.out | head -30
```

先頭 2 本が alfworld と webshop、以降が search。search の `SHARE` 行が
出た時点で、標的が **generate の中**なのか **retriever 待ち**なのかが決まる。
前者なら vLLM 側の話(`enable_chunked_prefill=False`、
`max_num_batched_tokens=8192`、毎 turn 全履歴を再 prefill)で env とは無関係、
後者なら `search_url` の fan-out(`envs.py:54` が複数 client に対応済み)が
最も安い。**cohort 分割はどちらでもない** —— search は 4 turn しかないので
overlap の余地自体が小さい。

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
| vLLM 毎 turn 寝起き | 10.4〜13% | **解消**。`closed after 50 generate calls on one wake` で確認 |
| `generate_sequences` の中の空き | alfworld batch で gen が wall の 87.1%、同 run の GPU busy が 46.6% | **未特定**。`GPU_PROFILER=1` が turn timing の `genGPU%` 列を埋める |
| search batch の retriever 待ち | **未測定**。411/413 の batch がこれ | 走行中のログの `SHARE` 行が出す |
| env.step(alfworld batch 実測) | 7.4% | 当初 42〜50% としたのは 1 batch からの外挿の誤り |
| 生成中の rank 不均衡 | — | 帰属を誤り、修正を revert(4 節) |

**評価の天井は「wall のうち GPU に仕事がある割合」で決まる。** alfworld batch
ではそれが 87.1% あり、うち半分が gen の中で空いている。search batch では
まだ分かっていない。**どちらを直すかを決める数字は、いま走っている run が
書き出している。**
