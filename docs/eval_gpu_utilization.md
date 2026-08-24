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

## 3. 43〜50% —— env.step が生成と直列

`vanilla_multi_turn_loop`(`agent_system/multi_turn_rollout/rollout_loop.py`)の
1 turn は完全直列である:

```
preprocess → generate(GPU) → decode → envs.step(CPU/IPC/HTTP)
```

GPU が働くのは generate だけ。`envs.step` の間、**vLLM は 28.5 GB 常駐した
まま カーネルが 1 本も走らない**。既存の overlap 機構(`_env_step_executor`
による logprob prefetch)は `is_train` 必須なので、**評価では死んでいる**。

これが最大の項で、**学習並みに持っていくには必ずこれを潰す必要がある**。
そして評価の天井はここで決まる: wall の 46% しか GPU の仕事が存在しない以上、
env を生成の裏に入れない限り util は 46% を超えない。

### なぜまだ実装していないか

隠すには batch を cohort に割り、片方が env.step 中に他方を生成する必要が
ある。alfworld の seeded game cycle を保つには**単一の envs オブジェクトを
保ったまま部分集合を step する**しかなく、それは:

* `SimpleMemory.store/fetch` を index 対応にする(全 manager 共有)
* 各 manager の `step` が持つ positional state
  (`self.pre_text_obs`、`self.memory`、`prev_admissible_commands`)を index 対応にする
* 各 leaf の `envs.step` を部分ディスパッチにする
  (alfworld は env ごとに Ray actor なので leaf は容易)
* rollout loop を cohort パイプラインに書き換える

を全部要求する。**評価の正確性クリティカルパスへの侵襲的変更**であり、
しかも「env 時間が turn のどこにどう分布しているか」を測る前に設計すると、
この調査で 3 回やった失敗(測らずに原因を言う)の 4 回目になる。

`ROLLOUT_TURN_TIMING=1` を `eval_checkpoints.sh` の既定にした。次の eval が
turn ごとの `preproc / gen / decode / envstep / genGPU%` を出す。決めるのは:

* 末尾が alfworld 単独なら cohort は **alfworld の中**を割る必要がある
* task 混在が続くなら cohort は **task** でよく、`_task_indices` が既にあるので
  はるかに安全

**この 2 つは実装も risk も別物である。**

---

## 4. 生成中の rank 不均衡 —— これは直した

2 秒トレースの生成中サンプル:

```
0, 87   1,  0   2,  0
0, 100  1, 31   2, 53
0, 87   1, 25   2, 88
```

生成 batch は**連続チャンクで rank に配られ**、評価セットは**task ごとに
固めて**保存されている。つまり rank 0 = alfworld、rank 1 = search、
rank 2 = webshop。3 task は生成長も終了 turn も違う。しかも search が数 turn で
終わると、**生き残った task の行が 1 枚に集中し、他の 2 枚には何も渡らない**。

`TaskBalancedSampler` の `TASK_BALANCE_INTERLEAVE` は *sampler* なので効かない
—— 検証 dataloader は sampler を取らず、データセット順に読む。

`verl/utils/task_interleave.py` を足した。round-robin で並べ替えるだけで、
**task 内の相対順序は保つ**。これが seeding 不変条件そのもので、alfworld は
TextWorld の seeded game cycle を**位置で**引くため、task 内で行を動かすと
別の episode を採点することになる —— per-checkpoint プロセス設計が防いで
いる当のものである。

`VAL_TASK_INTERLEAVE=1` で有効。**既定は OFF**: rank が変われば温度>0 で
サンプルされる token が変わる。batch 構成変更と同じ精度クラスだが、ここは
**採点経路**であり、pull しただけで報告する数字が変わってはいけない。
sweep 単位で on か off を決めること。

---

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
| env.step 直列 | **43〜50%** | **未着手。最大項。** 次の eval の turn timing が cohort の粒度を決める |
| vLLM 毎 turn 寝起き | 10.8〜13% | 観測可能にした。原因は次の run が確定させる |
| 生成中の rank 不均衡 | 生成 37〜47% のうち不明分 | `VAL_TASK_INTERLEAVE=1` で解消。既定 OFF |
| env 自体の遅延 | 未測定 | search は `search_url` の fan-out に対応済み。turn timing が最遅 task を出す |

**この 4 つを全部潰した場合の上限は、env を完全に隠せたとして ~90%。**
学習の 99.9% には届かない —— 評価は生成と環境の往復であり、環境が CPU に
いる限り、隠せるのは重ねられるぶんだけである。
