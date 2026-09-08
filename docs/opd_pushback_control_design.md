# OPD 押し戻し制御: タスク別・step 別の保持率を、対立する token にだけ適用する

状態: **実装済み、未実行。** ブランチ `claude/opd-per-task-coef`。`ARM=pushback` で起動する。
静的係数アーム（`docs/opd_coefficient_arm_design.md`）を**置き換える**機構であり、その上に重ねるものではない。

---

## 0. なぜ静的係数を捨てるか

静的アームは step 300 の checkpoint で一度測った $b$ を step 0 から適用した。走らせて分かった弱点は 2 つで、どちらも実測に基づく。

| 弱点 | 根拠 |
|---|---|
| **古い** | 校正時の `adv_zero_frac` は (0.215, 0.761, 0.623)、走行中 step 73 では (0.125, 0.784, 0.209)。webshop の対角の符号も step 1–6 と 73 で反転している |
| **無差別** | webshop の OPD には「webshop 自身を助ける部分」と「現在の報酬方向と対立する部分」が混在する。両方まとめて半減していた |

行列 $C_{ij}$ をパラメータ空間で作り直すには最低 6 本（検算・再現性込みで 15 本）の backward が要る。計算量は 2 倍で済むが、FSDP の collective は 6 倍、勾配バッファは 41 GB。それを毎 step に入れる代わりに、**出力空間で分離できる量だけで組む**規則にした。

## 1. 機構

同じ token のロジット $z$ 上で、降下方向を

$$u_R = -\frac{\partial L_{\rm GRPO}}{\partial z}, \qquad u_D = -\frac{\partial(\beta L_{\rm OPD})}{\partial z}$$

と置く（$\beta$ は基準係数。動的係数は**含めない**）。両方とも既存 forward の値からの閉形式で、追加 forward・backward は無い。

token $t$ に適用する OPD の重み:

$$w_{i,t} = \begin{cases} a_i & \text{control 母集団内 かつ } u_R\ne 0 \text{ かつ } u_R^\top u_D < 0 \\ 1 & \text{それ以外} \end{cases}$$

$$L = L_{\rm GRPO} + \beta\,\mathrm{Agg}\big[w_{i,t}\,L_{{\rm OPD},i,t}\big]$$

タスク $i$ の保持率 $a_i \in [0,1]$ は、**押し戻しを報酬自身の降下量の $\varepsilon$ 倍以内に抑える最小の減衰**として決める。

$$R_i = \sum_t \|u_{R,t}\|^2, \qquad C_i^- = \sum_t\big[-u_{R,t}^\top u_{D,t}\big]_+$$

$$\min_{0\le a\le 1}\tfrac12(a-1)^2 \ \text{s.t.}\ a\,C_i^- \le \varepsilon R_i
\quad\Longrightarrow\quad
a_i^* = \begin{cases}1 & C_i^-=0\\ \min\!\left(1, \dfrac{\varepsilon R_i}{C_i^-}\right) & C_i^->0\end{cases}$$

| 押し戻し率 $C^-/R$ | $a$（$\varepsilon=0.1$） |
|---:|---:|
| 0.02 | 1.0 |
| 0.10 | 1.0 |
| 0.50 | 0.2 |

**0.5 倍を先に決めるのではなく、観測された押し戻しの大きさから保持率が決まる。**$\varepsilon$ は許容する介入強度を表す hyperparameter で、最適と分かっているわけではない。

### 1.1 設計上の要点

* **$C^-$ は token ごとに負部分を取ってから合算する。** $\sum_t[-X_t]_+ \ne [-\sum_t X_t]_+$。符号付きの和では、ある token の有益な蒸留が別 token の強い対立を隠す。実装上、符号付き和 `ctl_X` からは復元できないので別カラム `ctl_C_neg` を持つ。
* **母集団・重み・$\beta$ を揃える。**$R$, $C^-$ は同じ token 集合（control 母集団 = response token かつ生成 token が support 内）、同じ行重み（`basis²`、両方向が線形に持つため）、$\beta$ 込み・$a$ 抜きで作る。行重みはタスク内で一定でない（重複行が重み 0）ので約分されない。
* **clip された token は母集団に残す。**$u_R=0$ は定義された値で、$R$, $C^-$ に 0、$D$ に実量を寄与する。除外すると 3 量の母集団が再び分かれる。
* **ゲートは二値。**soft gate $w = 1-(1-a)q_t$ では制御後の押し戻しが $C^- - (1-a)C_q^-$ となり、閾値は $a \le 1 - (C^--\varepsilon R)/C_q^-$ に変わって $a=0$ でも達成不能になり得る。二値なら $C_q^-=C^-$ で元の閉形式に戻る。soft 化は閉形式の変更を伴う**別条件**として評価する。
* **信頼度は「制御に寄与した観測」で数える。**token 数は損失が実際に重み付ける行（`basis > 0`）に限る —— 重複行は重み 0 で $R, C^-$ に寄与しないので、証拠として数えない。群数は**タスク別の群ビットマップ**で数える: driver が行ごとの密な群 ID を付け、actor が「loss mask・clip・top-k を通って $R$ に届いた行」のビットを立て、all-reduce してから非ゼロを数える。rank ごとの群数を単純加算すると、1 群を 2 rank が持つと 2 と数えてしまう。
* **1 step 遅れ、step 内固定。**$a_i$ は step $s-1$ までの統計から決め、step $s$ の全 micro-batch に同じ値を適用する。token の対立判定だけは step $s$ の forward で行う。
* **EMA は $R$, $C^-$ の和に掛け、比はその後に取る。**比の EMA ではない。欠測タスクは EMA に 0 を投入しない。
* **増幅しない。**v1 は $a\le1$。
* **タスク間の予算移動はしない。**alfworld の OPD を減らしても search・webshop は増えない。

### 1.2 信号の 4 状態

| 状態 | 判定 | $a$ | EMA |
|---|---|---|---|
| `absent` | そのタスクの行が無い | 変更なし | 更新しない |
| `no_pg` | $R_i \le$ 数値定数 | **1** | 更新しない |
| `few_groups` | 独立プロンプト群 $<$ `min_live_groups` または control token $<$ `min_ctl_tokens` | **1** | 更新する |
| `ok` | 十分 | 閉形式 | 更新する |

「不足」を $R$ の絶対値では決めない。$R$ は損失の正規化・群数・行重みに依存するので、それらを変えると挙動が変わる。群の被覆は driver が uid を見て数え（`live_groups_by_task`）、`meta_info["pushback_live_groups"]` で actor に渡す。

## 2. 実装

| 部品 | 場所 |
|---|---|
| 制御器（閉形式・ゲート・状態機械・checkpoint state） | `verl/trainer/ppo/opd_pushback.py` |
| 制御入力 $R, C^-, D$ と保持率の会計 | `verl/trainer/ppo/opd_task_diag.py`（`ctl_*`, `pb_*` カラム） |
| 配線（loss 前に terms→gate、両集約経路に $w$、step 末に $a_{\rm next}$） | `verl/workers/actor/dp_actor.py` |
| 群被覆 | `verl/trainer/ppo/opd_ray_trainer.py`（`_attach_task_ids` 直後） |
| checkpoint への保存・復元 | `fsdp_checkpoint_manager.py` の `actor_extra` hook、`fsdp_workers.py` |
| 設定注入・排他検証 | `verl/trainer/main_opd.py`（`validate_pushback_exclusivity`） |
| 起動 | `run_multitask_opd_coef_qwen3.sh` の `ARM=pushback`、lock `expected_multitask_opd_coef_pushback_config.yaml` |

**排他。**静的 `kl_loss_coef_by_task` は未設定または全タスク 1 のみ許可。`sign_weight` / `cross_teacher_kl_weight` / `cross_teacher_target` との併用は禁止。`task_diag=True` が必須（制御器は readout の reduced table を読み、自前の collective を持たない）。全部 1 なら既存 control と同じコード経路。

**resume。**$a_i$・EMA・観測数・step を `actor_extra.pushback` として保存。`eps` 等の設定が変わった checkpoint からの復帰は拒否する（別規則で計算した $a$ を混ぜない）。

**メモリ。**terms の $(bs,T,k)$ 中間は関数内で消え、残るのは $(bs,T)$ 数本。ただし loss の前に移したので学習グラフのピークと重なる。「小さく抑えられる見込み」であり、**実装後にピークを測る**。

## 3. 記録する指標

| 指標 | 何を答えるか |
|---|---|
| `pushback/a_applied/{task}`、`a_next` | この step に適用した $a$ と、次 step 用に決めた $a$ |
| `pushback/state/{task}`、`n_obs` | 4 状態と EMA の観測数 |
| `pushback/ema_R`、`ema_C_neg`、`ema_ratio`、`bound` | 制御入力と閾値 |
| `pushback/constraint_met/{task}` | 床 $a_{\min}$ に当たって制約未達なら 0 |
| `pushback/conflict_frac/{task}` | 教師が生きた報酬に対立している token の割合。**$a$ に依存しない事実** |
| `pushback/gated_frac/{task}` | 実際に減衰された token の割合。$a=1$ なら対立があっても 0 |
| `pushback/live_groups/{task}` | $R$ に寄与した独立プロンプト群の数（rank 横断の和集合） |
| `pushback/w_mean/{task}` | 集約重み基準の平均保持率 |
| `pushback/kl_retained/{task}` | **損失自身の** KL 保持率 $\sum b\,wKL/\sum b\,KL$（分子・分母とも行重み込み） |
| `pushback/kl_retained_unweighted/{task}` | 行重み無しの同比。別名で残す |
| `pushback/strength_retained/{task}` | 出力空間 OPD 強度の保持率 $\sum w^2\|u_D\|^2/\sum\|u_D\|^2$ |
| `pushback/ratio_after/{task}` | ゲート後の押し戻し率（その step の token 上） |
| `opd_diag/ctl_pushback_neg_frac/{task}` | ゲート前の $C^-/R$ |
| `opd_diag/ctl_cover`、`ctl_tokens_weighted`、`ctl_live_rows`、`tail_mass_mean` | 測定被覆（token・重み付き token・行・確率質量） |

保持率を 3 種に分けるのは、$\bar w$ を揃えても $\sum_t w_t L_{{\rm OPD},t}$ は揃わない（減衰対象と KL 量が相関する）ため。

## 4. 比較設計

| アーム | 役割 |
|---|---|
| `control` | $b=1$、ゲートなし |
| **`pushback`** | 処置 |
| （事後）一様 replay | `pushback` が実現した保持率スケジュールを一様に適用。「選択的」と「単に少ない」を分ける **近似的**対照。訪問状態が変わるので総量一致でも因果分離でもない |

採用判断は診断指標ではなく **alfworld・search・webshop それぞれの評価精度**で行う。

## 4.1 実行前に潰した不具合

初回実装をレビューで指摘され、CPU で再現・修正した 4 件。

| | 症状 | 原因 | 対処 |
|---|---|---|---|
| **P1** | 初回 update の末尾で `AttributeError` | `update_policy` 内で `data` が micro-batch dict → TensorDict → メトリクス dict と **3 回再束縛**され、末尾で `data.meta_info` が存在しない | 群数は reduced ビットマップから取る。`data` を末尾で読まないことを AST テストで固定 |
| **P2** | 実効 token 1 個でも重み 0 の重複行で `ctl_n=300` になり `state=ok` | 被覆カウントに行重みが入っていなかった／群数が advantage 非ゼロだけで判定していた | 重み付き token 数と群ビットマップ |
| **P3** | タスクが batch から欠けると `AssertionError` | 制御器が初回の `task_id_names` を固定していたが、driver は**存在するタスクだけ**から一覧を作る | 状態をタスク名で保持。欠測は `STATE_ABSENT`、新規は $a=1$ で参入 |
| **P4** | `kl_retained` が 0.2421、実損失は 0.2800 | 行重みが分子・分母に入っていなかった | 重み込みを `kl_retained`、無しを `kl_retained_unweighted` に分離 |

いずれも変異検査つき: 元の実装に戻すと対応するテストだけが落ちる。

## 5. 言えないこと

$$r_i^\top P d_j = u_{R,i}^\top J_i P J_j^\top u_{D,j}$$

token ごとの量は Jacobian を運べないので、$i\ne j$ は測れない。この機構は**各タスク自身の報酬と自身の教師の釣り合い**を調整する。共有パラメータを介したタスク間の犠牲を出力空間だけで排除することはできず、「タスク間干渉を解消する機構」とは呼ばない。局所的な対立を教師が間違っている証拠とも扱わない —— 全遮断ではなく強度に応じた減衰にしている理由がそれである。

主目的との勾配類似度で補助損失を制御する考え方（Du et al. 2018）に先行研究はあるが、出力空間近似・選択的減衰・GRPO+Adam にその保証は移らない。
