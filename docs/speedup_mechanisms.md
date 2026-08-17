# 高速化手法の稼働状態と Phase 1 詳解

この文書は 2 つの役割を持つ。

1. **稼働状態の切り分け** —— 実装されている高速化機構を、フェーズをまたいで 1 か所に並べ、
   **今この構成で走っているもの / 走っていないもの**に分ける。既存の 3 文書はそれぞれ別の
   作業単位の記録で、しかも実装だけあって発火していない機構・撤回した機構・削除した機構が
   混ざっているため、稼働状態を通しで読む場所が無かった（1 節・4 節）。
2. **Phase 1 の詳解** —— 5 機構（①〜⑤）が具体的にどのコードの何を消しているのか、
   なぜ精度が変わらないのか、そして**どこには効かないのか**を、コード位置つきで書く
   （2 節・3 節）。`docs/optimization_report.md` は結果の表であって、機構の中身は書いていない。

1〜6 節に新しい計測や新しい主張は含まない。数値はすべて下記の 3 文書からの引用である。
**7 節だけが例外で、そこには新しい計測がある** ―― step 18 の障害対応で入れた 2 機構と、
その実測コストである。速度を上げた機構ではないが、rollout の内訳を実際に動かしたので、
稼働中の機構として同じ場所に置く。

| 文書 | 範囲 |
|---|---|
| `docs/optimization_report.md` | Phase 1（rollout）＋ Phase 3（actor update）。sdar arm |
| `docs/optimization_phase2.md` | Phase 2（rollout の隙間埋め A–E） |
| `docs/gpu_profiling_report_opd.md` | Phase 4（OPD teacher）。誤判断の記録つき。pure OPD arm |
| `docs/webshop_worker_memory.md` | ホスト RAM（webshop worker の JVM） |

---

## 1. 稼働状態

基準は **pure OPD multitask arm の本番構成**（`examples/opd_trainer/run_multitask_qwen3.sh`
＋ `docs/gpu_profiling_report_opd.md` 7 節の再現手順）。arm による差は 1.4 節にまとめる。

**「実装されている」と「走っている」は別である。** 実装だけあって一度も発火していない機構
（B）、撤回した機構（spec decode）、削除した機構（Fix2 / query cache）が混ざっているので、
まず稼働状態で分ける。フェーズの帰属は列として残す。

### 1.1 稼働中

**rollout（`gen`）** —— ①〜⑤ の詳解は 2 節。

| 機構 | 有効化 | Phase | 精度 |
|---|---|---|---|
| ① vLLM セッション | `ROLLOUT_KEEP_VLLM_AWAKE=1`（script が export） | 1 | ビット同一 |
| ② active-only preprocess | 既定 on | 1 | ビット同一 |
| ③ prefix caching | `+rollout.enable_prefix_caching=True`（script 内） | 1 | ロスレス |
| ④ タスク interleave 配置 | `TASK_BALANCE_INTERLEAVE=1`（script が export） | 1 | ビット同一 |
| E2 active-only decode | 既定 on | 2 | ビット同一 |
| E3 compact per-turn record | 既定 on | 2 | ビット同一 |
| C env reset prefetch | `ENV_RESET_PREFETCH=1`（script が export） | 2 | ビット同一 |
| `max_model_len=4608` | script 内 | 1 | KV 予算を必要ちょうどに絞る |
| rollout log-prob を作らない | `rollout.return_rollout_log_probs=False` | 5 | 生成トークン不変 |
| session 中の `empty_cache` 抑止 | コード（常時、session 中のみ） | 5 | ビット同一 |

**5 期の 2 件は「読まれない結果を作るのをやめた」もの。** `rollout_log_probs` の消費者は
`RayPPOTrainer.fit` の drift 検査（`rollout_probs_diff`）だけで、比較対象の `old_log_prob`
フェーズを持たない pure OPD では一度も読まれないまま、毎ターン全生成トークンを Python で
走査して組み立てていた。`empty_cache` は `generate_sequences` の末尾で session 判定の外に
あり、vLLM を起こしたままにする ① を入れてもなお毎ターン走って同期を強制していた。
どちらも drift 検査を回す arm では従来どおりにしておくこと。

**teacher**

| 機構 | 有効化 | Phase | 精度 |
|---|---|---|---|
| teacher の CPUOffload 解除 | `ref.fsdp_config.param_offload=False` | 4 | 配置のみ |
| teacher の ZeRO-2 化 | `ref.fsdp_config.sharding_strategy=shard_grad_op` | 4 | ビット同一 |
| response-only lse/topk/gather | コード（常時） | 4 | ビット同一 |
| chunked teacher overlap（ターン単位） | `ROLLOUT_PREFETCH_TEACHER=1`（script が export） | 4 | 期待値同一 |
| chunk サイズの glue 追従 | `ROLLOUT_PREFETCH_TEACHER_ADAPTIVE=1`（既定 on） | 5 | 値に触れない |
| **response-only `lm_head`** | `actor.response_only_logits=True` / `ref.response_only_logits=True` | 5 | **ビット非同一**（GEMM 形状） |
| 死んだ sampled-token log-prob の削除 | コード（`pg_loss_coef=0` ＋ topk_kl のときのみ） | 5 | 値不変（未使用値の削除） |
| prefetch chunk の worker 側失敗を致命化 | コード（常時） | 6 | 値に触れない（失敗時のみ） |
| teacher forward の行数上限 | `ref.log_prob_micro_batch_size_per_gpu=4` | 6 | ビット同一 |

**teacher の下 2 行は速度機構ではない**（**遅くする**）。step 18 で run を殺した 2 つの欠陥の
修正で、rollout の内訳を実際に動かしたので同じ表に置いてある。機構と実測コストは 7 節。

**actor update**

| 機構 | 有効化 | Phase | 精度 |
|---|---|---|---|
| ZeRO-2（actor） | `actor.fsdp_config.sharding_strategy=shard_grad_op` | 3 | 算術中立 |
| `no_sync` 勾配蓄積 | `actor.no_sync_grad_accum=True` | 3 | **ビット非同一**（加算順序） |
| FSDP forward prefetch | `actor.fsdp_config.forward_prefetch=True` | 3 | スケジューリングのみ |
| metric 読み出しの遅延化 | コード（常時） | 3 | 値同一 |
| param-offload 解除（actor） | `actor.fsdp_config.param_offload=False` | 1 | ビット同一 |
| optimizer-offload 解除 | `actor.fsdp_config.optimizer_offload=False` | — | 同アルゴリズム |

**ホスト RAM / 起動系**（速度指標には出ないが、無いと 300 step 完走しない）

| 機構 | 有効化 |
|---|---|
| webshop worker の JVM heap 制限（`-Xmx512m -Xms64m -XX:+UseSerialGC`） | コード（常時、`SDAR_WEBSHOP_JVM_OPTIONS` で上書き可） |
| `SimServer.user_sessions` の刈り取り | コード（常時） |
| `LazyEnvManager`（env の遅延生成） | コード（常時） |
| `val_only` 時に train env を作らない | コード（常時） |
| retriever の無限リトライ + TCP keepalive | `env.search.max_retries=null`（速度ではなくデータ品質） |

**env var 系は run script が `${VAR:-1}` で export する。** `ROLLOUT_KEEP_VLLM_AWAKE` /
`TASK_BALANCE_INTERLEAVE` / `ENV_RESET_PREFETCH` / `ROLLOUT_PREFETCH_TEACHER` の 4 つで、
呼び出し側が 0 を渡せば従来どおり無効化できる。手で export する運用をやめたのは、
**立て忘れがエラーにならない**ためである ―― config 側の機構は効いたまま rollout 系だけが
全部 off になり、300 step の run が再起動されるたびにその危険がある。

### 1.2 稼働していない

| 機構 | 状態 | 理由 |
|---|---|---|
| A 完了軌跡の log-prob prefetch | **この arm では立てない** | pure OPD の薄いループに `old_log_prob` フェーズが無く、prefetch した値を消費する側がいない |
| B CUDA-graph capture sizes | **未発火** | `CompilationConfig` は V1 の機能。この環境は実際には V0 で動いている。ノブは sdar script のみ露出 |
| ngram speculative decoding | **撤回** | `SpecDecodeWorker` が `sleep()` 未実装で `init_workers` で落ちる |
| Fix2 per-task generation | **削除済み** | Fix1 の利得を相殺、peak mem 51→55.6 GB |
| async / continuous-batching rollout | **使えない** | on-policy 定義に反する。scaffolding のみ残る |
| retriever query cache / 並列プロンプト tokenizer | **削除済み** | Phase 2 で試作後に撤去 |
| chunked prefill | off | 次 run 候補（gen −5〜10% 見込み） |
| KV 予算 `gpu_memory_utilization` 0.6→0.7 | 0.6 のまま | 次 run 候補（gen −3〜8%） |
| `ppo_micro_batch_size_per_gpu` 5→10 | 5 のまま | 次 run 候補（`update_actor` −2〜3%） |
| `reward` の overlap | 見送り | 2.3〜3.1 s/step で割に合わない |
| `GPU_PROFILER_SYNC_PHASES` | 立てない | phase 境界で `synchronize()` するので実際に遅くなる |
| `GPU_PROFILER_TRACE` | 使えない | 分散で多重 open する（未修正） |

理由の詳細は 4 節。「次 run 候補」の 3 つは **3 arm 同時に入れる**前提である ―― 片方だけ
有効にすると実験条件が揃わない。

### 1.3 計測インフラ（常時利用可、既定 off）

- `verl/utils/gpu_profiler.py`（`GPU_PROFILER=1`）—— phase タグつき NVML サンプラ。
  per-phase の SM% / memBW% / idle% / mem / power / clock / PCIe / **per-GPU SM**
- per-turn rollout timing（`ROLLOUT_TURN_TIMING=1`）—— `preproc / gen / tchWait / decode /
  envstep` と `genGPU%` / `perGPU%`、`cpu-glue` / `teacher-spill` / `DP-IMBALANCE` の集計行
- worker 側 stage phase（`actor.fwd` / `bwd` / `task_metrics` / `optim`）—— ドライバの
  `_timer` は worker の中を見られないので、rank 0 が自分で phase を push する
- `timing_s/update_actor_worker` —— ドライバの blocking `ray.get` に含まれる転送を分離
- `flops_counter` への RTX A6000 / RTX PRO 6000 Blackwell 追加 —— 無いと
  `perf/mfu/actor` が `inf` で割って毎 step ちょうど `0.000` を出す

### 1.4 arm による差

| | pure OPD | sdar (GRPO) |
|---|---|---|
| A log-prob prefetch | 使わない | **使える**（`old_log_prob` フェーズがある） |
| teacher 系 4 機構 | 使う | 該当なし（teacher が無い） |
| B のノブ露出 | 無し | `CUDAGRAPH_CAPTURE_SIZES` env で渡せる（V0 なので無効） |
| Phase 1 ①〜⑤ | 使う | 使う |
| Phase 3（actor update） | 使う | 使う |

### 1.5 既存文書との食い違い 1 件

`docs/optimization_report.md` 4 節 / 7 節（Phase 1、sdar arm）は `OPTIMIZER_OFFLOAD` を
「既定 True のまま（＝offload 有効）」と記録しているが、**現在の run script は両 arm とも
`optimizer_offload=False`**（Adam を GPU 常駐）である。Phase 1 当時の判断が後で覆ったのか、
script 更新時に doc が追従しなかったのかは記録から追えない。速度の主張には影響しないが、
あちらの 7 節「Final production configuration」をそのまま再現手順として使うと現行 script と
食い違う。この文書の 5 節が現行の構成である。

---

## 2. Phase 1 詳解

### 2.1 ① vLLM セッション（`ROLLOUT_KEEP_VLLM_AWAKE`）

**変更前。** `generate_sequences` は毎ターン sharding manager を `with` で入り直していた
（`fsdp_workers.py:740`）。`FSDPVLLMShardingManager.__enter__`（`fsdp_vllm.py:114`）と
`__exit__`（`:216`）の中身:

| `__enter__` | `__exit__` |
|---|---|
| `empty_cache()` | `inference_engine.sleep(level=1)` ← **KV キャッシュ破棄** |
| （offload 時）`load_fsdp_model_to_gpu`（`:171`） | `module.train()` |
| `self.module.state_dict()` ← **FSDP 全 state_dict を gather** | `empty_cache()` |
| `wake_up(tags=["weights"])` | RNG を学習側ストリームに戻す |
| `update_params(params)` ← **全重みを vLLM に転送** | |
| `wake_up(tags=["kv_cache"])` | |
| RNG を生成用ストリームに差し替え | |

alfworld は最大 50 ターン回るので、これが 1 rollout に約 50 回。rollout 中の actor の
重みは 1 バイトも変わらないので、2 回目以降の gather と転送は**同じ値を書き直しているだけ**。
さらに `sleep(level=1)` が毎ターン KV を捨てるため、prefix cache もターンを跨げなかった。

**変更後。** worker に 2 メソッドを追加（`fsdp_workers.py:757` / `:780`、`Dispatch.ONE_TO_ALL`）:

```python
def begin_rollout_session(self):
    if not isinstance(self.rollout_sharding_manager, FSDPVLLMShardingManager):
        return                      # vLLM 以外のバックエンドでは no-op
    if getattr(self, "_rollout_session_active", False):
        return                      # 二重 open の防止
    self.rollout_sharding_manager.__enter__()
    self._rollout_session_active = True
```

`generate_sequences` はセッション中なら `with` を飛ばし、**データ側の sharding だけ**を
実行する（`fsdp_workers.py:735-748`）:

```python
if getattr(self, "_rollout_session_active", False):
    prompts = self.rollout_sharding_manager.preprocess_data(prompts)   # TP all-gather は毎ターン必要
    output  = self.rollout.generate_sequences(prompts=prompts)
    output  = self.rollout_sharding_manager.postprocess_data(output)
else:
    with self.rollout_sharding_manager:      # 従来経路をそのまま保存
        ...
```

呼び出しは rollout ループの最外周、**`finally` つき**（`rollout_loop.py:1125-1146`）。これに
より rollout を抜けた時点のエンジン状態（sleep 済み・offload 済み・`module.train()` 済み）が
非セッション経路と完全に一致する。後続の `teacher_forward` / `update_actor` から見ると
何も変わらない。

**精度。** 3 点そろっている。

1. rollout 中 `update_actor` は走らないので、セッション開始時に 1 回同期した重みは
   最終ターンまで同じ値。50 回同期しても 1 回でも vLLM が見る重みは同一
2. 非セッション経路も `__exit__` で生成側 RNG を `gen_random_states` に退避し次ターンの
   `__enter__` で復元しているので、**生成用 RNG は元々ターンを跨いで連続**だった。
   セッションはその退避/復元を省くだけ。prefetch で走る FSDP forward は eval モードで
   RNG を消費しないので、割り込んでも乱数列は動かない
3. `preprocess_data` の TP all-gather と `postprocess_data` の chunk は毎ターン実行される

**実測。** gen 738→562 s、gen/tok 0.708→0.541（−24%）、throughput 877→932（+6.3%)、
`gen` の sm 60→**51**。Phase 1 で単独最大の壁時計短縮。

**代償。** rollout の間ずっと KV キャッシュが常駐する。これが Phase 4 で効いてくる ——
`ROLLOUT_PREFETCH_TEACHER` は teacher forward を rollout の**中**へ移す機構なので、
そこは vLLM が起きている狭い環境である。ターン単位化で teacher 仕事のほぼ全部
（hit_rate 0.99）がその狭い方に移った結果、step 136 で 10.47 GiB の確保が OOM した
（`gpu_profiling_report_opd.md` 9.2）。セッション自体の欠陥ではなく、**セッションが作る
実行環境を前提に他機構をサイジングする必要がある**という話である。

### 2.2 ② active-only preprocess（`ROLLOUT_SKIP_DONE_PREPROC`、既定 on）

**変更前。** 各ターンの頭で `preprocess_batch` がバッチ**全行**に
`preprocess_single_sample`（`rollout_loop.py:275`）を掛けていた。1 行あたり:

```python
prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, ...)
input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(   # max_prompt_length=4096 まで左パディング
    prompt=..., max_length=self.config.data.max_prompt_length, left_pad=True, truncation=...)
position_ids   = compute_position_id_with_mask(attention_mask)
raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)   # ← vLLM 用に 2 回目のトークナイズ
```

ドライバ 1 プロセスの逐次 Python ループ（`_run_full_preprocess`、`:268`）で回り、その間
**GPU は完全に idle**。そして終了済み軌跡の行についてはこの結果が**どこからも読まれない**:

1. **生成に渡らない** —— `active_batch_input = batch_input[active_idx]`（`:851`）
2. **学習にも入らない** —— `gather_rollout_data` は `if data['active_masks']:` の行だけを
   `effective_batch` に積む（`:566`）

エピソード上限は search 4 / webshop 15 / alfworld 50 なので、**ターン 16 以降はバッチの
2/3 が終了済み**。そこでフルトークナイズを走らせ続けていた。

**変更後。** `preprocess_batch`（`:463`）に `active_mask` を追加:

```python
skip_done = (
    active_mask is not None
    and not bool(active_mask.all())          # 全員アクティブなら従来経路そのまま
    and obs.get('image', None) is None       # マルチモーダルは常にフル処理
)
```

3 条件のどれかが外れると元のコードパスに完全にフォールバックする。有効時、アクティブ行は
**変更なしの** `preprocess_single_sample` を通り、終了済み行は `_placeholder_single_sample`
（`:426`）が処理する:

```python
row_dict = {
    'input_ids':      torch.full_like(template['input_ids'], pad_token_id),
    'attention_mask': torch.zeros_like(template['attention_mask']),
    'position_ids':   torch.zeros_like(template['position_ids']),
    'raw_prompt_ids': [pad_token_id],
    'anchor_obs': _obs_anchor, 'index': item, 'data_source': ...,   # dict 引きだけの安いメタデータ
}
```

`template` は「その回で最初にフル処理されたアクティブ行」で、そこから shape と dtype を
コピーする。**行を削除せず placeholder で埋めるのが要点**である —— `collate_fn` は全行が
同じ shape であることを要求し、`active_idx` も `_scatter_active_to_full` も
「フルバッチのインデックス空間」を前提に書かれている。行を落とすとその対応が全部崩れる。

**精度。** アクティブ行は一文字も変わらない経路を通るので、vLLM に渡る入力はバイト単位で
従来と同一。placeholder 行のトークン列は上の 2 段のフィルタで誰にも読まれない。

**実測。** preproc 95.5 → 37 s/step（−61%）。0 にはならない。残る 37 s は主に
ターン 1〜15 の**全行がアクティブな区間**で、そこは `skip_done=False` なので従来経路その
ものである。この残りが Phase 2 A / Phase 4 の prefetch が埋めにいく CPU glue の一部になる。

**記録された測定ミス。** コード冒頭（`:46-48`）に残してある ——
「an earlier "measured neutral" conclusion was invalid — that run had not pulled this code」。
一度「効果なし」と結論した測定があり、その run はこのコードを取り込んでいなかった。
既定 on にしたうえで `=0` で従来挙動に戻せるのは、この A/B を正しくやり直せるようにするため。

### 2.3 ③ prefix caching の config 化

**変更前の実態。** エンジン構築側は既に有効だった（`vllm_rollout_spmd.py:201`）:

```python
enable_prefix_caching=config.get("enable_prefix_caching", True),   # ← 既定 True
```

一方 `verl/trainer/config/ppo_trainer.yaml` の `rollout` ブロックに
**`enable_prefix_caching` というキーは存在しない**。つまり:

- prefix caching は**すでに on** だったが、それはコード内の `.get(..., True)` の
  第 2 引数によるものだった
- 合成後の config にも wandb の config dump にも**現れない**＝実験条件として記録されない
- off にする手段が無い（A/B が取れない）
- Hydra は未宣言キーへの代入を拒否するので、渡すには `+` 接頭辞が要る

verl 内でも経路ごとに既定が揃っていない。`vllm_rollout.py:136` は同じ `.get(..., True)`、
`vllm_async_server.py:181` は `enable_prefix_caching=True` の決め打ちである。

**変更。** 両 run script が明示的に渡す。sdar 側は `ENABLE_PREFIX_CACHING` で上書き可能:

```bash
+actor_rollout_ref.rollout.enable_prefix_caching=True \
```

速度の話ではなく再現性の話である。だから報告書の効果欄は「marginal」になっている。

**効いている場所を正確に。** 「マルチターンだから会話が伸びるほど効く」という直感は
この実装には当てはまらない。

- **(a) ターン内・バッチ内の共有プレフィクス —— これが主。** 1 回の `generate_sequences` は
  数百行を同時に投げ、同一タスクの行はチャットテンプレート＋タスク指示＋`skills/` の
  スキル文という同じヘッダを持つ。この共通部分の prefill が 1 回で済む
- **(b) ターン跨ぎ —— ①と組み合わせて初めて成立し、範囲は限定的。** `sleep(level=1)` は
  KV を破棄するので、セッション無しでは毎ターン全部 cold。ただし観測テキストは
  `agent_system/memory/memory.py:84` の `self._data[env_idx][-history_length:]` ——
  **直近 N ステップのスライディングウィンドウ**で組み立てられ、`env.history_length` は
  alfworld/webshop=2、search=4。プロンプトは単調に伸びる prefix ではなく、履歴部分は
  ターンごとに窓がずれる。ターンを跨いで再利用されるのは**窓より前の静的ヘッダまで**

**精度クラス。** ロスレス／分布保存であって、ビット同一とは言っていない
（`optimization_report.md` 6 節）。そもそもこの構成は `logprobs=0` で生成し、学習側の
log-prob / teacher-KL は FSDP actor がサンプルされたトークンに対して再計算するので、
vLLM の数値が損失に入る経路自体が無い。

**周辺ノブ。**

| ノブ | 現在値 | 関係 |
|---|---|---|
| `free_cache_engine` | **False** | KV エンジンを解放しない。①の前提でもある |
| `gpu_memory_utilization` | 0.6 | KV 予算。**次 run 候補**（gen −3〜8% 見込み） |
| `max_model_len` | 4608（prompt 4096 + response 512） | KV 予算を必要ちょうどに絞る。有効な系列を 1 つも切らない |
| `enable_chunked_prefill` | **False**（yaml の既定は True） | **次 run 候補**（gen −5〜10%、分布不変だがビット同一ではない） |

### 2.4 ④ タスク interleave 配置（`TASK_BALANCE_INTERLEAVE`、Fix1）

**前提。** `tensor_model_parallel_size=1` の 2 GPU なので **DP=2**。各 GPU が自分の vLLM
エンジンを持ち、バッチの自分の取り分だけを生成する。分割は `DataProto.chunk()`
（`verl/protocol.py:652`、中身は `torch.chunk(dim=0)`）で、**連続ブロック**である。
1 ターンの壁時計は 2 ランクを `ray.get` で待ち合わせるので `max(rank0, rank1)`。

**変更前。** `TaskBalancedSampler.__iter__`（`main_ppo.py:281`）の従来経路:

```python
for task in self.tasks:
    yield from task_indices[task][start:end]      # alfworld を全部、次に search、次に webshop
```

そのあと GRPO のグループ展開が入る（`rollout_loop.py:1097`、`interleave=True` はグループ内の
n 行を隣接させる指定）。結果:

```
[alfworld ×120][search ×120][webshop ×120]        (15 prompts/task × n=8 = 360 行)
      ↓ chunk(2)
rank0 = alfworld 120 + search 60
rank1 = search 60 + webshop 120
```

**rank 0 は alfworld 専用機、rank 1 は webshop 専用機**になっていた。タスクごとにプロンプト長
（webshop は約 2.3k トークン）も応答の出方も EOS の来るターンも系統的に違うので、
2 ランクの仕事量が構造的にずれる。片方が先に終わって実際に遊ぶ。

終了行を落としても直らない。`active_idx` による圧縮は順序を保つので、search が終わった
turn 5 以降は `[alfworld 120][webshop 120]` → rank0 が全部 alfworld、rank1 が全部 webshop と
さらにきれいに分離する。

**変更後**（`main_ppo.py:293`）:

```python
if self._interleave:
    # Round-robin task layout: alf0, search0, webshop0, alf1, ...
    for i in range(self.per_task_batch_size):
        for task in self.tasks:
            yield task_indices[task][start + i]
```

グループ展開後は「n 行 × 3 タスク」のブロックが並ぶ形になり、どこで連続分割しても各ランクが
ほぼ同じタスク構成を受け取る。

**精度。** サンプル集合が完全に同一であることが要点。`_indices_for_required_size(indices,
required, rng)` は `_interleave` と無関係に**先に**タスクごとの index 列を作り、`rng.shuffle`
もそこで済んでいる。変わるのは `yield` の順番だけ。GRPO グループは後段で `uid` から作られ、
`repeat(interleave=True)` によってグループ内の行は隣接したままなので、advantage の正規化
母集団は不変。

**実測。** DP imbalance 約 20 pp（40/60）→ **約 8 pp**、`gen` の sm 51→57%、
gen/tok 0.541→0.496（**−8%**）。turn table に専用行が出る（`rollout_loop.py:226-231`）:

```
DP-IMBALANCE  mean |maxGPU-minGPU| during gen = 8.2 pp
              (lower=better; TASK_BALANCE_INTERLEAVE shrinks this on mixed turns)
```

**①との対比。** ① は sm 60→51 と**下がって** 24% 速くなり、④ は sm 51→57 と**上がって**
8% 速くなった。① で消えたのは冗長な重み同期という偽の仕事、④ で埋めたのは本物の idle。
util を上げるも下げるも目標にはならない、という `optimization_report.md` 5 節の結論は
この 2 例の対比から出ている。

**却下された代案 —— Fix2（per-task generation）。** 同じ不均衡に対して「タスクごとに別々の
`generate` を呼ぶ」案も実装して測った。狙いどおりタスク内は均質になったが、呼び出し回数が
3 倍になるオーバヘッドが Fix1 の利得を打ち消した —— gen/tok は Fix1 のみ 0.496 に対し
Fix1+Fix2 が 0.53〜0.57、peak mem 51→55.6 GB。スループット改善が無いのでブランチから削除済み。

**適用条件。** `data.task_balance.enable=True` のマルチタスク run 専用（`TaskBalancedSampler`
が使われるとき）。単一タスクの run では sampler 自体が別なので無効。

### 2.5 ⑤ FSDP param-offload 解除（`PARAM_OFFLOAD=False`）

**まず用語の分離。** 「param_offload」は 2 か所で意味が違う。

| | actor（`role == "actor"`） | ref / teacher（`role == "ref"`） |
|---|---|---|
| FSDP の `CPUOffload` | **強制 off**（grad accumulation で誤った結果になるため） | 従来**強制 on** ← Phase 4 3.1 節で解除 |
| 手動 offload（`load_fsdp_model_to_gpu` / `offload_fsdp_model_to_cpu`） | `_is_offload_param` で制御 ← **⑤はこれ** | 呼び出し自体が存在しない |

`fsdp_workers.py:372` の 1 行がこの分岐:

```python
cpu_offload = None if role == "actor" else (
    CPUOffload(offload_params=True) if fsdp_config.get("param_offload", False) else None)
```

ref 側は手動 offload の呼び出しが 1 つも無い（`compute_ref_log_prob:834` /
`compute_ref_topk_log_prob:869` のどちらにも `_is_offload_param` が出てこない）ため、
キーが読まれてもどこにも効かない **dead key** だった —— これが Phase 4 で発見された方の話。
⑤ は actor 側の手動 offload だけを扱う。

**変更前。** `_is_offload_param=True` のとき、actor の重みは各 phase の入口で GPU に載り、
出口で CPU に送り返される:

| 場所 | 動作 |
|---|---|
| `fsdp_vllm.py:171` / `:202`（rollout sharding manager の enter/exit） | load → state_dict → vLLM 同期 → offload |
| `fsdp_workers.py:796` / `:827`（`compute_log_prob`） | load → forward → offload |
| `fsdp_workers.py:669` / `:706`（`update_actor`） | load → fwd/bwd/optim → offload |

Qwen3-1.7B bf16 = 3.4 GB、2 GPU なので shard は 1.7 GB/GPU。これが 1 step の中で何往復もする。
さらに `offload_fsdp_model_to_cpu`（`verl/utils/fsdp_utils.py`）は最後に `empty_cache()` を
呼ぶので、**PyTorch の caching allocator を毎回捨てて**いる。NVLink の無いホストでは
この転送はすべて PCIe を通り、同じバスを使う FSDP の collectives と競合する。

**変更後。** `_is_offload_param` が False になると上の `if` がすべて素通りになり、shard は
GPU に常駐する。配置が変わるだけで、どのカーネルも同じ入力に対して同じ計算をする。
前提条件は GPU 空きメモリで、報告書の記述は「≥64 GB headroom」。

**効果が「small」な理由。** ⑤ 単体の効き幅は**①が先に食っている**。セッションが入ると
rollout 中の enter/exit が 50 回から 1 回に減るので、そこに乗っていた offload 往復も同時に
減る。⑤ が残りで削れるのは `compute_log_prob` / `update_actor` の phase 境界と、rollout
1 回ぶんの往復だけである。**機構の効果は適用順に依存する。**

**対になる却下 —— `OPTIMIZER_OFFLOAD=False`。**

| | `PARAM_OFFLOAD=False`（採用） | `OPTIMIZER_OFFLOAD=False`（却下） |
|---|---|---|
| 常駐する物 | 重み shard 1.7 GB/GPU | Adam の m, v を fp32 で ≈13.6 GB |
| 対象 phase の性質 | phase 境界の転送（純粋な待ち時間） | `update_actor` は SM 97% |
| 結果 | 小さいが正の効果 | **速度改善ゼロ**。peak mem 51→60 GB、KV preemption が発生 |
| 判断 | 採用 | env knob としては残し、**既定 True のまま** |

「offload を外せば速い」は一般には成り立たない。

**ZeRO-2 との関係。** `get_sharding_strategy` は矛盾する組み合わせを警告する
（`fsdp_workers.py:126-128`）。ZeRO-2 は「gather した parameters を forward〜backward の間
保持する」機構、param_offload は「使い終わったら CPU に送り返す」機構で、目的が正反対である。
Phase 3 で ZeRO-2 を採用した時点で ⑤ は選択肢ではなく**前提条件**になった。

---

## 3. ④ の適用範囲 —— alfworld テールには効かない

「interleave を入れれば alfworld の 50 ターンに律速される問題も緩和されるのでは」という
読み方が成り立ちうるので、明示的に否定しておく。**効かない。設計上そこを狙っていない。**

### 3.1 テールで no-op になる理由

turn 16 以降はアクティブ行が alfworld だけになる。interleave はサンプラの yield 順を
変えるだけなので、**alfworld 行どうしの相対順序は両レイアウトで同一**である
（`task_indices['alfworld'][start + i]` を i 昇順で吐くため）。したがって `active_idx` 圧縮後の
行列は 2 つのレイアウトで完全に一致し、`chunk(2)` の分割も一致する。差分がゼロ。

| ターン | アクティブなタスク | interleave |
|---|---|---|
| 1–4 | 3 タスク（フルバッチ） | **効く**。contiguous では rank0=alfworld / rank1=webshop に完全分離 |
| 5–15 | alfworld + webshop | **効く**。同上 |
| 16–50 | alfworld のみ | **no-op** |

turn table の文言もそう書いてある —— `shrinks this on **mixed turns**`。テールに効かない
機構としては gen −8% が上限で、実際その通りの数字が出た、という関係である。

### 3.2 テールは 3 つの別々の穴に分かれる

`gpu_profiling_report_opd.md` 2.3 節が (a)(b) を混同するなと繰り返し書いている部分。
④ はそのどちらでもない第三の穴を埋めている。

| 穴 | 正体 | 担当機構 | 状態 |
|---|---|---|---|
| DP 間の待ち合わせ | 混在ターンでランク間の仕事量が違う | **④ interleave** | 20→8 pp、混在ターンのみ |
| (a) driver の CPU glue | decode / `envs.step` / 次ターンの tokenize。rollout の 18%、GPU は 0 | ②、Phase 2 A、Phase 4 teacher prefetch | **埋め切った**。hit_rate 0.991 |
| (b) `generate_sequences` の内側 | decode テール。数本まで減った系列の帯域律速（6〜7 ms/decode-step） | — | **ドライバから触れない**。`gen_util 65.3` |

**「50 ターン回る」という長さそのものを縮めた機構は 1 つも無い。** テールに対して実際に
やったのは、その間 GPU が遊んでいる (a) の窓へ別の仕事を運び込むことだけである。しかも
それも 2.4 節の制約下でしか成立しない —— rollout と teacher は同じ `WorkerDict` に同居し
Ray actor は 1 呼び出しずつしか走らないので、`generate_sequences` の実行中に teacher を
投げてもキューに並ぶだけ。埋められるのは (a) であって (b) ではない。

### 3.3 (b) を縮める手が残っていない

| 手 | 状態 |
|---|---|
| ngram speculative decoding | **撤回**。`SpecDecodeWorker` が `sleep()` 未実装で `init_workers` で落ちる。V1 エンジンが前提 |
| chunked prefill / KV 予算 0.6→0.7 | **次 run**（gen −5〜10% / −3〜8% 見込み）。3 arm 同時に入れる |
| CUDA graph capture sizes | V1 限定。現状 V0 なので無効 |
| async / disaggregated RL | **選べない**。生成と訓練を重ねる＝訓練中のポリシーで生成していない＝ off-policy 化で、pure OPD arm の存在意義が壊れる |

見積りは 2.6 節にあり、入れられる手を全部入れても sm 82–85% / 〜490 s/it、async まで行けば
90+ / 〜380 s/it。最終行は実験定義上選べないので、**テール由来の idle の大部分は
「最適化で消せる idle」ではなく「実験の定義が要求している idle」**である。

### 3.4 ただし「alfworld 律速＝損」ではない

3 タスクは 1 本の alfworld-gated な rollout wall を**共有**している。つまりこの multitask run
は 1 タスク分の壁時計で 3 タスク分の軌跡を出している。search が turn 4 で、webshop が
turn 15 で終わったあと GPU の一部が空くのは事実だが、その空きは「alfworld だけを単独で
回した場合」と比べて増えたものではない。テールの低 util は主に見かけ上のもので、単位仕事
あたりのスループットは天井に近い。

そして現時点でその空きに入れる仕事はもう残っていない（9.3 節）—— teacher の hit_rate が
0.991 に達して teacher 仕事は尽き、pure OPD の薄いループには `old_log_prob` フェーズが
無いので prefetch する対象も無い。GPU-idle な glue が 46.9 s 残っているが、**この arm には
埋める仕事が存在しない。**

---

## 4. 稼働していない機構 —— 状態と理由

1.2 節の一覧の根拠。4 つの状態を区別する ―― **この arm では立てない**（他の arm では動く）、
**未発火**（設定はあるがエンジンが無視する）、**撤回 / 削除**（コードから外した、または
動かないと判明した）、**次 run 候補**（入れる予定があるが今は入っていない）。

### 4.1 この arm では立てない

**A 完了軌跡の log-prob prefetch**（`ROLLOUT_PREFETCH_LOGPROB`）。
`envs.step()` をバックグラウンドスレッドに出し、その裏で記録済み行の `compute_log_prob` を
先取りする機構。pure OPD の薄いループには `old_log_prob` フェーズが無いので、prefetch した
値を消費する側がいない ―― 計算しても捨てるだけになる。sdar arm では有効。

同じ glue 窓を pure OPD で埋めているのは teacher prefetch の方である。実装は共有されていて、
`_queue_row_for_prefetch` が両方の pending リストに同じ entry を積み、有効な機構の側だけが
ドレインする。

### 4.2 未発火

**B CUDA-graph capture sizes**（`CUDAGRAPH_CAPTURE_SIZES`）。
passthrough は `vllm_rollout_spmd.py:170-183` に元からあり、Phase 2 が足したのは run script の
ノブだけ。`enforce_eager=False` は両 arm とも設定済みなので **CUDA graph 自体は既に有効**で、
B が制御するのは「どのバッチサイズを capture するか」である。

しかし `CompilationConfig` は V1 エンジンの機能で、この環境は実際には **V0 で動いている**。
渡しても無視される。

ここには判断の誤りが記録されている（`gpu_profiling_report_opd.md` 5 節②→⑧）。一度
「`environment.yml` は `vllm==0.11.0` を pin しており V0 は削除済み」と述べたが、spec decode の
トレースバックが `vllm/engine/llm_engine.py` / `vllm/executor/uniproc_executor.py` /
`vllm.spec_decode.*` ―― すべて V0 の経路を示した。

> 5 節②で「pin なので V0 は無い」と書いたのは *`VLLM_USE_V1` の export が no-op である*
> ことの根拠であって、**実行中のエンジンが V1 である証明ではなかった**。

`VLLM_USE_V1=1` は全フェーズのエンジンを変えるので、3 arm 同時の別実験として残っている。
B と spec decode はその 1 つの実験で一緒に検証すべき項目である。

### 4.3 撤回 / 削除

| 機構 | 経緯 |
|---|---|
| **ngram speculative decoding** | `engine_kwargs.vllm.speculative_config` を渡すと vLLM は V0 の `spec_decode.SpecDecodeWorker`（内側に `NGramWorker`）に差し替わる。このラッパーは `sleep()` を実装していない一方、`vllm_rollout_spmd` はエンジンを `enable_sleep_mode=True` で作り直後に `sleep(level=1)` を呼ぶので、**step 1 に入る前に `init_workers` で落ちる**。`free_cache_engine=False` と `ROLLOUT_KEEP_VLLM_AWAKE` が依存する wake/sleep サイクル全体がこのメソッドを要求するため、引数の不足ではなく構造的非互換 |
| **Fix2 per-task generation** | 目的（タスク内の util 均質化）は達成したが、呼び出し回数が 3 倍になるオーバヘッドが Fix1 の利得を打ち消した ―― gen/tok は Fix1 のみ 0.496 に対し Fix1+Fix2 が 0.53〜0.57、peak mem 51→55.6 GB。ブランチから削除 |
| **`OPTIMIZER_OFFLOAD=False`（Phase 1 の判定）** | sdar arm では「`update_actor` が compute-bound で速度改善ゼロ、peak mem 51→60 GB、KV preemption」として却下された。ただし**現行の run script は両 arm とも `optimizer_offload=False`** である（1.5 節） |
| **retriever query cache** | Phase 2 で試作、その後削除 |
| **並列プロンプト tokenizer** | 同上 |
| **async / continuous-batching rollout** | sdar arm では「de-vectorize + token-level async engine で多週」として保留。pure OPD arm では**不可** ―― 生成と訓練を重ねるとは訓練中のポリシーで生成していないということで、この arm の存在意義そのものを壊す。実装面でも `AsyncActorRolloutRefWorker.generate_sequences` が `NotImplementedError` を投げる |

### 4.4 次 run 候補

| 手 | 見込み | 備考 |
|---|---|---|
| chunked prefill 有効化 | `gen` −5〜10% | サンプリング分布は不変。ビット同一ではない |
| KV 予算 `gpu_memory_utilization` 0.6 → 0.7 | `gen` −3〜8% | OOM 判定は `max_memory_reserved 145.3 GB` から逆算 |
| `ppo_micro_batch_size_per_gpu` 5 → 10 | `update_actor` −2〜3% | `adjust_batch` の lcm が 160 → 320 になり padding が増える |

**3 つとも 3 arm 全部に同時に入れること。** いずれも損失の計算式を変えないが、片方の arm
だけ有効にすると「実験条件が揃っていない」と言われる余地を作る。

到達点の見積りは `gpu_profiling_report_opd.md` 2.6 節 ―― sm 82〜85%、〜490 s/it。

### 4.5 計測側で立てないもの

- `GPU_PROFILER_SYNC_PHASES` —— phase 境界ごとに `device.synchronize()` を入れるので、本来
  重なる処理を直列化して**実際に遅くなる**。帰属としては読めても速度としては読めない
- `GPU_PROFILER_TRACE` —— 分散では多重 open する（パスに rank を混ぜれば直るが未修正）

---

## 5. 本番構成（pure OPD multitask）

**プロセス env var**（run script が `${VAR:-1}` で export する。手で立てる必要はない）:

```bash
export ROLLOUT_KEEP_VLLM_AWAKE=${ROLLOUT_KEEP_VLLM_AWAKE:-1}
export ENV_RESET_PREFETCH=${ENV_RESET_PREFETCH:-1}
export TASK_BALANCE_INTERLEAVE=${TASK_BALANCE_INTERLEAVE:-1}
export ROLLOUT_PREFETCH_TEACHER=${ROLLOUT_PREFETCH_TEACHER:-1}
# 既定 on: ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY / ROLLOUT_COMPACT_RECORD
#          ROLLOUT_PREFETCH_TEACHER_ADAPTIVE（chunk を glue に追従。128 が下限 / 512 が上限）
# 立てない: ROLLOUT_PREFETCH_LOGPROB（消費側が無い）、GPU_PROFILER_SYNC_PHASES（遅くなる）
# 計測（任意）: GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1 GPU_PROFILER_ROLLUP_EVERY=1
bash examples/opd_trainer/run_multitask_qwen3.sh env.search.search_url=http://<host>:8000/retrieve
```

**config**（すべて run script 内のリテラル引数）:

```
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
+actor_rollout_ref.actor.fsdp_config.sharding_strategy=shard_grad_op
+actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
+actor_rollout_ref.actor.no_sync_grad_accum=True
+actor_rollout_ref.actor.response_only_logits=True
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.ref.fsdp_config.sharding_strategy=shard_grad_op
+actor_rollout_ref.ref.response_only_logits=True
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4      # メモリ上限（7 節）。速度は下がる
+actor_rollout_ref.rollout.enable_prefix_caching=True
+actor_rollout_ref.rollout.return_rollout_log_probs=False       # drift 検査を回す arm では True
actor_rollout_ref.rollout.disable_log_stats=False              # 計測。vLLM 内部統計を出す
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.max_model_len=4608
```

`sharding_strategy` と `no_sync_grad_accum`（actor 側）は勾配経路上にあるため、性能ノブは
intent lock に入れないという規則の例外として
`examples/opd_trainer/expected_multitask_config.yaml` に pin してある。ref 側の 2 つは
ビット同一なので pin していない。

---
## 6. 数値の読み方

この一連の作業で 4 回同じ形の誤りを踏んだので、結論だけ書いておく。詳細は
`gpu_profiling_report_opd.md` 5 節。

1. **`s/step` は指標にならない。** run ごとに step あたりのトークン量が最大 43% 違う。
   3 構成の 517.2 → 548.3 → 496.3 が非単調であること自体がその証拠である。
   速度の議論は必ず `perf/throughput`（per-GPU、定義は全 run で同一）で行う
2. **NVML の util は目的関数として壊れている。** PCIe 転送中も NCCL collectives 中も
   busy と数える。util を上げて遅くする変更は簡単に作れる（`enforce_eager=True`、
   `enable_prefix_caching=False`、teacher の CPUOffload を戻す）。util は「どこを見るか」を
   決める中間指標であって、最適化対象ではない
3. **代理指標は「効いている」を示すが「効果量」は示さない。** `pcieRX` / `hit_rate` /
   `tchWait` は機構が動いている証拠にはなるが、どれだけ速くなったかではない。両方要る。
   そして少数 step の測定を「速くなった量」として報告してはいけない（2 点で +20% と述べ、
   49 点で +8.2% に落ち着いた例がある）
4. **phase 列は、機構を 1 つ入れるだけで意味が変わる。** `timing_s/teacher_forward` は
   prefetch を入れる前は teacher のコスト全部だったが、入れた後は **glue に隠しきれなかった
   残りしか持っていない**。この列だけ見て teacher の micro-batch を「ほぼ無料」と値付けし、
   実測すると 2.0% だった（7.3 節）。ノブを値付けする前に、そのコストが今どの列に入るのかを
   確かめること ―― 名前は変わらないまま中身が移る

---

## 7. 追加機構 2 件 —— teacher prefetch の失敗経路と teacher forward の行数上限

**両方とも速度機構ではない。片方は速度に中立、もう片方は測って 2.0% 遅い。**
それでもここに書くのは、(a) rollout の内訳（`tchWait`）を実際に動かしたので 2 節・6 節の
数字の読み方に直接関わること、(b) 「teacher の micro-batch を細かくしてもコストはループ
overhead だけ」という**この文書の前提だった見積りが、実測で外れた**ことによる。

経緯は `gpu_profiling_report_opd.md` 5 節⑩。300 step の run が step 18 で死んだとき、
独立した欠陥が 2 つ同時に露出した ―― teacher prefetch chunk の OOM と、その OOM を
握り潰した設計である。以下はその 2 つに対応する。

### 7.1 worker 側で起きた prefetch 失敗を致命化する

`_join_teacher_prefetch`（`agent_system/multi_turn_rollout/rollout_loop.py`）は、chunk が
上げた例外を握り潰して rollout を継続していた。設計根拠は「取りこぼした行は
`compute_teacher_log_probs` が serial 経路で再計算し、completeness guard が残りを検出する。
**値は同一**だから、step を殺す方が高くつく」。

値の議論としては正しい。**collective の議論としては誤り**だった。この経路の worker 呼び出しは
すべて FSDP forward を通るので、worker 内の例外は「ある rank が all-gather から抜けた」を
意味する。他の rank は abort しない ―― NCCL watchdog の 30 分を待ってから死ぬ。実際に
step 18 で `SeqNum` が rank 間で 1 ずれ、1,800,001 ms 後に run ごと落ちた。

修正は判定軸を 1 本入れただけである。

| 例外の出どころ | 扱い | 根拠 |
|---|---|---|
| driver 側 | 従来どおり印字して drop | プロセスグループは無傷。serial 経路が同じ値を再計算する |
| worker 内（`RayTaskError` / `ActorDiedError` / `ActorUnavailableError`） | **即 raise** | rank が collective を抜けている。再計算では直らない |

型は `ray.exceptions` から**名前で遅延解決**する（版によって定義が違い、ray 非依存の CPU
テストに import を持ち込まないため）。解決できなければ空タプルになり、`except ()` は何にも
マッチしないので従来の挙動に落ちる。

**速度への影響はゼロ**（正常系では 1 度も通らない）。回帰テストは
`tests/ray_cpu/test_teacher_chunk_adaptive.py` に 3 件追加してある（worker 側 OOM /
worker 死亡 / 型解決が plain な例外を拾わないこと）。新しい `except` 節を `except ()` に
差し替える mutation を当てると、**その致命化 2 件だけが落ちて他は通る** ―― テストが
実際にこの分岐を見ていることの確認である。

### 7.2 teacher forward の行数上限を 16 → 4

`compute_ref_topk_log_prob` の lm_head が作る `(n_resp, 151936)` fp32 が step 中で最も
広い割り当てで、micro-batch は**行数**で切られている（`log_prob_use_dynamic_bsz=False`）。
16 行は step 1 の応答長 139.1 を前提にした値で、学習が進んで応答が伸びると上界も伸びた
――step 18 で mean 257.0 / clip 22.2%、16 × ~417 token → 3.77 GiB を 3.49 GiB free に
要求して OOM した。

4 にすると上界が**構造的**になる。`data.max_response_length=512` は pin 済みなので
4 行は最大 2,048 response token = 1.16 GiB、logsumexp の一時バッファと合わせて約 2.3 GiB を
超えられない。以後の応答長の伸びで壊れない。`adjust_batch` の除数にも入らないので
（`size_divisor_ref` は `use_kl_in_reward` / `actor.use_kl_loss` が両方 False のとき rollout
側にフォールバックする）、padding も落とす行も動かない ―― **ビット同一**。

### 7.3 実測コスト —— 2.0%、全額が `tchWait`

**16 行の run（step 18 で死亡）と 4 行の run の、step 1〜18 の平均**。同一 seed・同一データで、
step あたりトークン量も 4,597,659 → 4,589,489（−0.2%）とほぼ揃っているので、6 節 1 の
「`s/step` は指標にならない」に抵触せずに両方を並べられる区間である。

| 指標（step 1〜18 平均） | 16 行 | 4 行 | 差 |
|---|---|---|---|
| `perf/throughput` | 4565 | 4474 | **−2.0%** |
| `timing_s/step` | 502.8 | 512.5 | +9.6 s |
| rollout TOTAL | 243.7 | 253.7 | +10.0 s |
| ├ `preproc` | 22.2 | 22.1 | −0.1 s |
| ├ `gen` | 179.9 | 178.9 | −1.0 s |
| ├ **`tchWait`** | **15.6** | **26.4** | **+10.8 s** |
| ├ `decode` | 1.8 | 1.8 | ±0 |
| └ `envstep` | 24.3 | 24.5 | +0.2 s |
| `timing_s/update_actor` | 249.7 | 248.7 | −1.0 s |
| `timing_s/teacher_forward` | 1.10 | 1.81 | +0.7 s |
| `teacher_prefetch/hit_rate` | 0.978 | 0.974 | ±0 |
| `genGPU%`（全ターン平均） | 72.7 | 71.2 | −1.5 pt |
| `perf/mfu/actor` | 0.27 | 0.27 | ±0 |

**増分は 1 か所に集中している。** rollout の他 4 列も `update_actor` も動いていない
（勾配経路に触れていないので当然）。`hit_rate` が変わらないことが効いていて、prefetch で
採点する行数は同じまま、その採点が遅くなった分がそのまま `tchWait`（glue に隠しきれなかった
teacher 時間）に出た、と読める。glue 窓（`preproc + decode + envstep` ≒ 48 s、両 run とも
同じ）は 16 行の時点で既に 15.6 s の spill を出していた ―― つまり隠しきれる量を超えていた
――ので、増えた分に逃げ場が無い。

`timing_s/teacher_forward` が +0.7 s しか動いていないことに注意。**この列だけ見ると
「ほぼ無料」に見える** ―― run script のコメントに実際そう書いてあり、それが誤りだった。
teacher のコストは prefetch 導入以降、大半が `gen` の中に移っている。

### 7.4 取り戻す手（未着手）

`ref.log_prob_use_dynamic_bsz=True` ＋ `ref.log_prob_max_token_len_per_gpu` に切り替えると、
上限が**行数ではなくトークン数**になる。4 行固定は「常に最悪ケースで割る」ので、応答が短い
step でも 16 行相当をまとめられない分を毎 step 払っている。トークンで切れば OOM に対する
構造的保証は同等のまま、短い step では大きくまとめられる。

4.4 節の 3 候補と同じ扱いで、**3 arm 同時に入れること**。ビット同一ではない（micro-batch の
分割が変わる）ので、片方の arm だけ変えると比較が壊れる。
