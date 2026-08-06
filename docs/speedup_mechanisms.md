# 実装済み高速化手法の総覧と Phase 1 詳解

この文書は 2 つの役割を持つ。

1. **総覧** —— 現在このリポジトリに実装されている高速化機構を、フェーズをまたいで
   1 か所に並べる。既存の 3 文書はそれぞれ別の作業単位の記録なので、「今どの機構が
   入っているのか」を通しで読む場所が無かった。
2. **Phase 1 の詳解** —— 5 機構（①〜⑤）が具体的にどのコードの何を消しているのか、
   なぜ精度が変わらないのか、そして**どこには効かないのか**を、コード位置つきで書く。
   `docs/optimization_report.md` は結果の表であって、機構の中身は書いていない。

新しい計測や新しい主張は含まない。数値はすべて下記の 3 文書からの引用である。

| 文書 | 範囲 |
|---|---|
| `docs/optimization_report.md` | Phase 1（rollout）＋ Phase 3（actor update）。sdar arm |
| `docs/optimization_phase2.md` | Phase 2（rollout の隙間埋め A–E） |
| `docs/gpu_profiling_report_opd.md` | Phase 4（OPD teacher）。誤判断の記録つき。pure OPD arm |
| `docs/webshop_worker_memory.md` | ホスト RAM（webshop worker の JVM） |

---

## 1. 全機構一覧

### 1.1 Phase 1 —— rollout（`gen`）

| # | 機構 | ノブ | 実測 | 精度 |
|---|---|---|---|---|
| ① | vLLM セッション | `ROLLOUT_KEEP_VLLM_AWAKE=1` | gen 738→562 s、gen/tok −24% | ビット同一 |
| ② | active-only preprocess | `ROLLOUT_SKIP_DONE_PREPROC`（既定 on） | preproc 95.5→37 s/step（−61%） | ビット同一 |
| ③ | prefix caching の config 化 | `+rollout.enable_prefix_caching=True` | marginal（既に on だった） | ロスレス |
| ④ | タスク interleave 配置 | `TASK_BALANCE_INTERLEAVE=1` | DP 不均衡 20→8 pp、gen/tok −8% | ビット同一 |
| ⑤ | FSDP param-offload 解除 | `PARAM_OFFLOAD=False` | small | ビット同一 |

合成で **gen wall 977→~520 s（−47%）、step −13%、throughput +7%**。詳解は 2 節。

### 1.2 Phase 2 —— rollout の隙間埋め

| # | 機構 | ノブ（既定） | 内容 |
|---|---|---|---|
| A | 完了軌跡の log-prob prefetch | `ROLLOUT_PREFETCH_LOGPROB`（off）、chunk 64 | `envs.step()` 中の GPU idle 窓で `old_log_prob` を先取り。**pure OPD では立てない**（薄いループに `old_log_prob` phase が無く、prefetch した値が消費されない） |
| B | CUDA graph decode ノブ | `CUDAGRAPH_CAPTURE_SIZES`（unset） | V1 エンジン必須。現状 V0 なので**未発火** |
| C | env reset prefetch | `ENV_RESET_PREFETCH=1`（off） | 次バッチの `envs.reset()` を学習フェーズと重畳。ビット同一 |
| E2 | active-only decode | `ROLLOUT_DECODE_ACTIVE_ONLY`（on） | pad 行の decode をやめて `''` を直接埋める。ビット同一 |
| E3 | compact per-turn record | `ROLLOUT_COMPACT_RECORD`（on） | `active_masks=False` 行を記録しない。ビット同一 |

試作後に削除：retriever query cache、並列プロンプト tokenizer。

### 1.3 Phase 3 —— actor update / PCIe collectives

NVLink の無い 2 GPU ホストでは `update_actor` が全 phase 中で最大の PCIe 帯域
（TX 10.5 GB/s、`gen` の 2.0 に対して）を出す。NCCL collectives は NVML から
SM-busy に見えるので、sm% では「計算している」と「通信している」を区別できない。

| 機構 | ノブ | 削減対象 | 精度 |
|---|---|---|---|
| ZeRO-2 | `actor.fsdp_config.sharding_strategy=shard_grad_op` | gradient checkpointing 下の layer あたり all-gather 3 回→1 回 | 算術中立 |
| `no_sync` 勾配蓄積 | `actor.no_sync_grad_accum=True` | mini-batch あたり reduce-scatter 12→1（60/5 時）。ZeRO-2 下では再 gather も消える | **ビット非同一**（加算順序、期待値同一） |
| FSDP forward prefetch | `actor.fsdp_config.forward_prefetch=True` | 次ユニット all-gather の直列化 | スケジューリングのみ |
| metric 読み出しの遅延化 | 常時 on | logger 専用スカラを 0-d GPU tensor のまま保持。step あたり約 450 回の host 同期を削除 | 値同一 |

最初の 2 つは勾配経路上にあるため、性能ノブは intent lock に入れないという通常の
規則に反して `examples/opd_trainer/expected_multitask_config.yaml` に pin してある。

### 1.4 Phase 4 —— OPD teacher

| 機構 | 実装 | 実測 | 精度 |
|---|---|---|---|
| teacher の CPUOffload 解除 | `fsdp_workers.py:372` | `teacher_forward` 67.4→40.1 s/step、pcieRX 8,204→4,307 | 配置のみ |
| response-only lse/topk/gather | `dp_actor.py:114 response_row_selection` | 対象行 約 1/9、**ピークメモリ 121.2→93.9 GB** | ビット同一 |
| chunked teacher overlap | `rollout_loop.py` + `opd_ray_trainer.py:254` | `gen` sm 56→65.3、hit_rate 0.28–0.46 | 期待値同一 |
| prefetch のターン単位化 | `_queue_row_for_prefetch` | hit_rate 0.531→**0.991**、`teacher_forward` 33.7→**1.30 s**、**step −29.8 s** | 同上 |
| teacher の ZeRO-2 化 | `ref.fsdp_config.sharding_strategy=shard_grad_op` | `gen` の pcieRX 768→208、メモリ +8.2 GB | ビット同一 |

累積 throughput **3,321 → 3,594 → 3,782 tok/s（+13.9%）**。

付随して `ref.log_prob_micro_batch_size_per_gpu` 16→8。これは高速化ではなく
**OOM 対策**である（`lm_head` を通す teacher ロジットが 10.47 GiB の一括確保になった）。

### 1.5 ホスト RAM / 起動系

run を完走させるための変更。速度指標には現れないが、これが無いと 300 step 走らない。

- **webshop worker の JVM heap 制限** —— pyserini が worker ごとに JVM を起動し、
  既定 max heap が物理 RAM の 1/4（256 GB ホストで 62 GB）。120 worker で 1.8 GB/step の
  ドリフト。`_JAVA_OPTIONS=-Xmx512m -Xms64m -XX:+UseSerialGC`（`SDAR_WEBSHOP_JVM_OPTIONS`
  で上書き可）
- **`SimServer.user_sessions` の刈り取り** —— reset 時にクリア
- **`LazyEnvManager`**（`env_manager.py:956`）—— env の遅延生成。multitask では 492 actor
  のうち 252 が使われないまま常駐していた
- **`val_only` 時に train env を作らない** —— `_fast_forward_env_schedules` をガード
- **retriever の無限リトライ + TCP keepalive** —— 速度ではなくデータ品質（リトライ切れの
  エラー文字列がそのまま `<information>` ブロックとして学習に入るのを防ぐ）

### 1.6 計測インフラ

- `verl/utils/gpu_profiler.py`（`GPU_PROFILER=1`）—— phase タグつき NVML サンプラ。
  per-phase の SM% / memBW% / idle% / mem / power / clock / PCIe / **per-GPU SM**
- per-turn rollout timing（`ROLLOUT_TURN_TIMING=1`）—— `preproc / gen / tchWait / decode /
  envstep` と `genGPU%` / `perGPU%`、`cpu-glue` / `teacher-spill` / `DP-IMBALANCE` の集計行
- worker 側 stage phase（`actor.fwd` / `bwd` / `task_metrics` / `optim`）—— ドライバの
  `_timer` は worker の中を見られないので、rank 0 が自分で phase を push する
- `timing_s/update_actor_worker` —— ドライバの blocking `ray.get` に含まれる転送を分離
- `flops_counter` への RTX A6000 / RTX PRO 6000 Blackwell 追加 —— 無いと
  `perf/mfu/actor` が `inf` で割って毎 step ちょうど `0.000` を出す

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
util を上げるも下げるも目標にはならない、という 5 節の結論はこの 2 例の対比から出ている。

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

## 4. 採用しなかった / 撤回した手法

| 手法 | 判定 | 理由 |
|---|---|---|
| Fix2 per-task generation | **削除** | 目的は達成したが呼び出しオーバヘッドで Fix1 の利得を相殺。peak mem 51→55.6 GB |
| `OPTIMIZER_OFFLOAD=False` | 既定 True のまま | `update_actor` は compute-bound で無効。peak mem 51→60 GB |
| ngram speculative decoding | **撤回** | `speculative_config` を渡すと V0 の `SpecDecodeWorker` に差し替わり `sleep()` 未実装。wake/sleep に依存する rollout 設計との構造的非互換 |
| async / continuous-batching rollout | sdar arm は保留 / pure OPD arm は**不可** | 前者は de-vectorize + token-level async engine で多週。後者は on-policy 定義違反 |
| `reward` の overlap | 見送り | 2.3〜3.1 s/step で割に合わない |
| retriever query cache、並列プロンプト tokenizer | 試作後**削除** | Phase 2 で入れたが外した |
| chunked prefill / KV 予算 0.7 / `ppo_micro_batch` 5→10 | **次 run** | 3 arm 同時に入れる（片方だけだと実験条件が揃わない） |

---

## 5. 本番構成（OPD multitask）

```bash
export ROLLOUT_KEEP_VLLM_AWAKE=1
export ENV_RESET_PREFETCH=1
export TASK_BALANCE_INTERLEAVE=1
export ROLLOUT_PREFETCH_TEACHER=1        # chunk は既定 128（hit_rate 0.99 で足りている）
# ROLLOUT_SKIP_DONE_PREPROC / ROLLOUT_DECODE_ACTIVE_ONLY / ROLLOUT_COMPACT_RECORD は既定 on
# ROLLOUT_PREFETCH_LOGPROB は pure OPD では立てない
# 計測（任意）: GPU_PROFILER=1 ROLLOUT_TURN_TIMING=1
```

config 側は `sharding_strategy=shard_grad_op`（actor / ref 両方）、`no_sync_grad_accum=true`、
`forward_prefetch=true`、`ref.param_offload=false`。最初の 2 つは勾配経路上にあるため
`examples/opd_trainer/expected_multitask_config.yaml` の intent lock に pin されている。

---

## 6. 数値の読み方

この一連の作業で 3 回同じ形の誤りを踏んだので、結論だけ書いておく。詳細は
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
