# OPD 係数の再配分アーム: 確定した機構

状態: **機構は確定。実装は途中（§4）。GPU 未実行。** 判断が残るのは §7 の 1 点だけ。

このブランチ（`claude/opd-per-task-coef`）は `claude/privileged-notice` を base にした実装用である。
関連文書は別 worktree（`claude/opd-coefficient-cross-effect-cd4c38`）にあり、この文書はそこでの
検討と改訂を経て**確定した形だけ**を、実装側から書いたものである。

* 元の提案への検討: `docs/opd_coefficient_cross_effect_review.md`
* 設計の改訂履歴: `docs/opd_coefficient_cross_effect_design.md`
* 測定の出所: `docs/cross_teacher_theory.md` §4.14（$N=8$、OPD+GRPO control step 300、勾配取得の修正後）
* $b$ の算出: `scripts/opd_cross_effect_qp.py --redistribute`

測定データは `~/grad_probe/terms_n8_fixed.json`。**この文書の数値はすべてそこから独立に再計算して
一致を確認したものである**（プローブは `claude/cross-teacher-target-6b2123` にある）。

---

## 0. 何を検定するのか

MOPD の損失は $L=\sum_i w_i L_i^{RL} + \beta\sum_j w_j L_j^{OPD}$ で、$\beta$（`opd.kl_loss_coef`、現行 0.01）は
**3 タスク共通の 1 スカラー**である。したがって「webshop の教師だけ弱く蒸留する」は表現できなかった。

§4.14 の測定は**教師ごとの量**を出した —— タスク $j$ の教師が押す方向が、**他タスクの報酬勾配**と揃っているか。
このアームが検定するのは:

> **その順位で 3 本の教師の重みを配り直すと、@150 の到達点が変わるか。**

一次近似の正しさは検定しない（一次寄与は RL 自身の 0.25〜0.99% で、成功率に届く経路は固定点の移動である）。
RL–RL の衝突も対象外（cosine 0.002〜0.016 で、解消すべきものが無い）。

---

## 1. 機構（確定）

### 1.1 係数の決め方

$$
\tilde C_{ij}=\frac1N\sum_{n=1}^{N}\frac{\langle r_i^{(n)},\,d_j^{(n)}\rangle}{\|r_i^{(n)}\|\,\|d_j^{(n)}\|},
\qquad
c_j=\sum_{i}\tilde C_{ij}
$$

$$
w_j=\frac{\overline{\|d_j\|}}{\sum_k\overline{\|d_k\|}},\qquad
\delta_j=c_j-\sum_k w_k c_k
\quad\Longrightarrow\quad \sum_j w_j\delta_j=0
$$

$$
\kappa=\min\!\left(\frac{1-b_{\min}}{-\min_j\delta_j},\ \frac{b_{\max}-1}{\max_j\delta_j}\right),
\qquad
b_j=\mathrm{clip}\big(1+\kappa\,\delta_j,\ b_{\min},\ b_{\max}\big)
$$

$b_{\min}=0.5$、$b_{\max}=1.5$。実効係数は $\beta\,b_j$。$r_i,d_j$ は**係数込み**（`task_loss_weight` を含む）。

**付則（タスクを区別しない）**: あるバッチでタスク $i$ の RL 勾配が厳密に 0 なら、その行の平均から
そのバッチを除く。定義されたバッチが $N/2$ 未満の行があれば go しない。
step 300 の 24 セルはすべて非ゼロで、**この付則は発動していない**。

### 1.2 3 つの設計判断と、その理由

| 判断 | 採用 | 理由 |
|---|---|---|
| $C$ が決めるのは**配分だけ**、総量は決めない | $\sum_j w_j\delta_j=0$ | 停留点では任意の正定値 $P$ で $\sum_{ij}C_{ij}=-\|\sum_i r_i\|_P^2\le 0$。総量を $C$ に委ねると**必ず縮む**（= $\beta(t)$ 減衰の再発明）。実測でも 8 バッチ中 7 で負 |
| 硬い制約 QP ではなく**比例則** | $b_j=1+\kappa\delta_j$ | 絶対水準（$C_ib\ge0$ が成り立つか）は雑音。bootstrap で解が全停止と部分減衰の間を 6:4 で行き来した。**順位だけが再現する**（96.4%） |
| 行の重みは**等しい票** | $\alpha_i=1$ | cosine 空間では信頼度の低い行は**既に減衰している**（Spearman の減衰）。実測で search 行の $\overline{\lvert\tilde C\rvert}$ は 0.023 に対し alfworld 0.047・webshop 0.039 —— **既に半分の票**。被覆率（0.237）で更に割ると実効 0.23 倍の二重割引になる |

**予算は係数の和ではなく $\|d_j\|$ 加重和で取る。** $\|d_j\|$ は 0.408〜0.886（2.17 倍）違うので、
$\sum_j b_j$ を固定しても蒸留量は保存されない（初版 $\sum b=3$ では線形和 +10.0%、合成ノルム +24.4%）。

**捨てた案とその理由**（同じ失敗を繰り返さないための記録）:

* **精度重み（逆分散）** —— search 行を**最大**（1.859）にして順位再現を 97.1% → **65.1%** に落とした。
  cosine は大きさを正規化するので、減衰して小さくなった値のバッチ間分散は小さく、「安定」に見える。
* **被覆率重み** —— §1.2 のとおり二重割引。加えて (i) 「8 本の rollout がたまたま揃った標本の人工物」と
  「その状態で真に方策勾配が 0」を同じ割引で扱う、(ii) `rollout.n` と成功率の regime に依るので
  学習段階で $b$ が整列と無関係に動く、(iii) 人選 $S$ との一致を根拠にするのは循環。
* **偏差を $w$ で割る形** —— $\|d\|$ の小さい alfworld を 1.16 まで動かす。alfworld の列和はほぼ 0
  （$c=-0.019$、bootstrap で $P(b_{\text{alf}}>1)=0.76$）で、測定は alfworld を動かす根拠を持たない。
* **人が選ぶ $S=\{\text{alfworld},\text{webshop}\}$** —— 機構が入力の関数として閉じない。新しい run、
  4 タスク目、生存群の増減のたびに人が再判断することになり、再現性の向きと逆。

### 1.3 何が正則化の量を決めるか

$\langle r_i,d_j\rangle$ が負に揃うのは**正則化が効いている印**であって、切る理由ではない（§1.2 の第 1 行）。
したがって正則化の量は $\beta$（現行 0.01）と床 $b_{\min}=0.5$ が担い、**$C$ からは決めない**。
$\beta$ の時間変化を入れるなら $\beta(t)$ 減衰として $C$ とは独立に決める。

---

## 2. 現時点の値（step 300、$N=8$）

| | alfworld | search | webshop |
|---|---:|---:|---:|
| $c_j$ | −0.0188 | **+0.0258** | **−0.2432** |
| $\overline{\|d_j\|}$ | 0.4563 | 0.8859 | 0.4084 |
| $w_j$ | 0.2607 | 0.5061 | 0.2333 |
| **$b_j$** | **1.076** | **1.191** | **0.500** |
| 実効 $\beta b_j$ | 0.0108 | 0.0119 | 0.0050 |

**不変量**:

| | $\sum_j b_j\|d_j\|$ | $\|\sum_j b_j d_j\|$ |
|---|---:|---:|
| control (1,1,1) | 1.7506 | 1.0700 |
| 再配分 (1.076, 1.191, 0.500) | **1.7506（+0.00%）** | 1.1885（**+11.1%**） |
| 一様 1.111 | 1.945 | 1.1885 |

$d_j$ はほぼ直交（非対角 cosine −0.114 〜 +0.086）なので、線形和と合成ノルムは**一様倍率では同時に揃わない**。
規則は線形和を厳密に保存し、合成ノルムの +11.1% は一様アームが引き受ける（§3）。

**bootstrap（8 バッチ再抽出 ×2000）**:

| | |
|---|---|
| 順位 search > alfworld > webshop | **96.4%** |
| $P(b_{\text{search}}>1)$ | 0.999 |
| $b_{\text{webshop}}$ の 5–95% | [0.50, 0.50]（box 下端に張り付く） |
| $b_{\text{alfworld}}$ の 5–95% | [0.99, 1.15]、$P(>1)=0.76$ |

**go/no-go**: 順位の bootstrap 再現率が 90% を下回れば $\kappa=0$（= control）とし、走らせない。

**測定が支持しているのは 1 つの情報**である —— webshop の教師が押す向きが他タスクの報酬勾配と食い違う。
3 つの独立な測定が同じ向きを指す（§4.14.3 の OPD–OPD が 8/8 負、§3.6 の出力空間、この $C$ の webshop 列）。
残りは $\sum w\delta=0$ の帰結であり、**実際に検定する自由度は「webshop を半分にし、その分を主に search に回す」の 1 つ**。

---

## 3. 実験設計（3 アーム、確定）

0 → 150 step。OPD が効く区間を含めるため（150→300 では理論から「3 アームとも区別できない」が事前予測）。

| アーム | $b$ | 実効 $\beta b_j$ | 役割 |
|---|---|---|---|
| **control** | (1, 1, 1) | 0.01 / 0.01 / 0.01 | 基準 |
| **一様** | (1.111, 1.111, 1.111) | 0.0111 ×3 | **合成ノルムを再配分に合わせる**。+11.1% だけの効果を分離 |
| **再配分** | (1.076, 1.191, 0.500) | 0.0108 / 0.0119 / 0.0050 | 処置 |

一様アームは**線形和を合わせるためではない**（それは規則が既に保存している）。$d_j$ が直交なので
合成ノルムだけがずれ、それを引き受けるのがこのアームである。3 本あれば
「再配分が control とも一様とも違えば配分の効果」と読み分けられる。

**検証**: step 150 の checkpoint を同一プロトコルで 3 回（`RUN_TAG` を変える。同じ step を再検証すると
per-instance dump が上書きされる）。タスク別に報告し、pooled は出さない。
反復雑音は alfworld SD 1.39pp、webshop acc 0.46pp（$n=3$ の推定）。

### 3.1 測定点（確定: B1 本線、B2 で符号確認）

| | checkpoint | 状態 |
|---|---|---|
| **B1（本線）** | 8/5 control の step 50<br>`~/checkpoints/verl_agent_opd_grpo_multitask/global_step_50` | **実在確認済み。world size 3**、`optim_*.pt` あり。HF にマージして `PROBE_HF_PATH` で読む（Adam 状態は落ちるが、規則が使うのは $P=I$ なので不足なし） |
| **B2（確認）** | 8/13 アームの step 50<br>`~/checkpoints/verl_agent_opd_grpo_tmp_multitask/global_step_50` | **実在確認済み。world size 2**、`optim_*.pt` あり。そのまま読め、Adam 幾何の $C$ も取れる。ただし signweight target アームで control ではない |

**$C$ と一緒に $\|d_j\|$ も取り直す。** $\|d_j\|$ の比は学習段階で動く —— 純 OPD step 150 で
search ×4.47、OPD+GRPO step 300 で ×1.94。$w_j$ はその値で計算する。

**go の追加条件**: 早期 checkpoint では生存群が少ない可能性がある（base 近傍は成功率が低く、
全失敗群が退化する）。プローブの `advantages` 報告で alfworld / webshop の生存群が
8 バッチとも 3/15 以上あることを確認する。

### 3.2 事前予測

| タスク | $b$ | 予測 | 外れたら |
|---|---|---|---|
| **webshop** | 0.50 | **2 つの仮説が逆を予測する。** H1（干渉）: acc ≥ control。H2（早期は教師が必要）: acc < control | §6 の交絡に注意。`acc < control` は H2 と「62% のトークンが唯一の勾配を半分失う」の両方から出る |
| search | 1.19 | OPD が主信号（生存群 1〜5/15）。≥ control、差は小さい | < control なら教師寄りが強すぎる（$\tau$ の増加が固定点を悪化） |
| alfworld | 1.08 | ≈ control（±2pp 以内）。測定は動かす根拠を持たない | 差が出れば共有パラメータ経由の効果 |

**3 タスクとも null** なら、$\beta=0.01$ でのタスク別係数には梃子が無い、で機構族を閉じる。

---

## 4. 実装状況

### 4.1 完了（`7a82ed7`、このブランチ）

| 項目 | 内容 |
|---|---|
| 設定 | `algorithm.opd.kl_loss_coef_by_task` → `actor.teacher_kl_loss_coef_by_task`（`main_opd.py`。`main_opd_grpo` は同じ関数を再利用するので両エントリを覆う） |
| 損失 | `dp_actor.teacher_kl_row_coef()` が行ごとの $b$ を作り、**両方の集約経路**に適用。per-task 重み付き経路は `row_kl * task_loss_weight * b`、素の token-mean 経路は `agg_loss(teacher_kld * b)`（分母は不変） |
| 不適用時に落ちる | `by_task` が設定されているのに `task_ids` / `task_id_names` が無ければ AssertionError。タスク名の打ち間違いも拒否 |
| PG 項に触れない | 変更なし |
| 計測 | `actor/teacher_kl_coef_effective/{task}` |
| lock | `expected_multitask_config.yaml` 2 ファイルに `teacher_kl_loss_coef_by_task: null` を固定（control が「使っていない」ことを検査可能にする） |
| 試験 | `tests/trainer/test_teacher_kl_coef_by_task.py` 14 件。$b$ 未設定で従来式に戻ることを AST で検証 |

### 4.2 残り

| 項目 | 内容 |
|---|---|
| box 検証 | 起動時に $b_j\in[0.5,1.5]$ を検査。**「平均 1」は検査しない** —— 勾配予算では $\sum b_j=2.767$ になり、学習側は $\|d_j\|$ を知らないので予算の成立は検証できない |
| per-task KL 計測 | `actor/teacher_kl_loss_weighted_{task}`。配線検証（最初の 2 step で比が $b$ に一致するか）に必要 |
| スクリプト | `examples/opd_grpo_trainer/run_multitask_opd_coef_qwen3.sh`。**`run_multitask_cross_teacher_klw_control_qwen3.sh`（$\beta$=0.01）から派生**。`run_multitask_qwen3.sh` は **$\beta$=1.0 なので使えない**（確認済み） |
| アーム lock | `expected_multitask_opd_coef_config.yaml` + `tests/trainer/test_opd_coef_arm.py`。control との差が `kl_loss_coef_by_task` だけであることを固定 |
| $b$ の算出 | `scripts/opd_cross_effect_qp.py --redistribute` をこのブランチへ移送。$C$・$\|d\|$・$w$・両不変量・bootstrap の順位再現率を JSON に残す |
| 文書 | 別 worktree の設計書と検討をこのブランチへ移送（または相互参照を維持） |

`teacher_kl_loss_coef` を受け取る他の 6 箇所（`xt_position_terms`、`opd_logit_push`、channel-loss の列）は
klw / signweight アームの**診断**で、control 系のスクリプトでは無効。このアームでも無効のままにし、
「診断はスカラー $\beta$ を報告する」を lock に書く。

### 4.3 起動前の確認（4 本の rollout が 1 行の問題で死んだ記録に従う）

1. dry-run（`--cfg job`）で config dump に `kl_loss_coef_by_task` と `kl_loss_coef=0.01` が**両方**出ること
2. `teacher_kl_loss_coef` の全消費箇所を grep し、損失経路が 1 つだけであること
3. CPU 試験: $b=\mathbf1$ で損失が control と bit-identical、$b$ 設定時にタスク $j$ の KL 項だけが $b_j$ 倍、
   PG 項が不変、box 外・名前不一致で起動拒否
4. 最初の 2 step で `actor/teacher_kl_loss_weighted_{task}` の比が $b$ に一致すること
5. インテントロックがアーム専用の期待値ファイルで通り、control との差が `kl_loss_coef_by_task` だけであること

---

## 5. 理論的な位置づけ

* 命題 2: $b_j$ はタスク $j$ の全位置で $\tau=\beta W^{kl}/W^{pg}$ を $b_j$ 倍する。固定点は
  $\pi^\star\propto\pi_d\exp(\tilde Q/(\beta b_j))$ の側へ動く。webshop は教師寄りが弱まり、search は強まる。
  **これが OPD の主効果であり、一次の $C$ には現れない。**
* 一次の効果は小さい（OPD の一次寄与は RL 自身の 0.25〜0.99%）。この run が検定するのは
  一次の量ではなく、順位で配り直したときの到達点である。
* 信頼を**教師と独立な信号**（報酬勾配）で決めているので、cross-teacher の識別不能性定理に触れない。
* 同一タスク対 $\tilde C_{ii}$ には共分散が混入する（$r_i$ と $d_i$ は同じ軌跡から）。
  対角を除いても順位は変わらないので、この run の $b$ には効かない。

---

## 6. 限界（走らせる前から分かっていること）

* **$C$ は一次・局所・$P=I$ の量。** Adam 幾何の $C$ は control では本ホストで測れない
  （HF マージで `exp_avg_sq` を失う）。B2 で符号一致の確認だけ行う。
* **合成ノルムが +11.1% ずれる。** 一様アームがこれを引き受けるが、$d_j$ が直交なので
  線形和と合成ノルムを 1 本の一様アームで同時に揃えることはできない。
* **box $[0.5,1.5]$ は選択であり、測定から出ていない。** $b_{\text{webshop}}$ は常に下端に張り付くので、
  実際に検定する自由度は 1 つ。
* **順位は checkpoint 1 点・プロンプト列 1 本から出ている。** 8 バッチは同じデータローダから続けて
  引いた 8 回のロールアウトで、シードを変えた反復ではない。
* **【解釈の交絡】タスクごとに OPD 依存度が違う。** $A=0$ のトークン（そこでは OPD が唯一の勾配）は
  alfworld 21.5%・**webshop 62.3%**・**search 76.1%** で、しかも $A=0$ は群まるごとの単位（3 タスクとも 100%）。
  **これは機構の非対称ではない** —— 規則は 3 タスクに同一の形で、結果が違うのはデータが違うから。
  ただし webshop の `acc < control` は「早期は教師が必要」（H2）と「62% のトークンが唯一の勾配を
  半分失う」の**両方から出る**ので、§3.2 の予測は交絡している。
  区別は per-instance dump で付けられる（退化群の instance と生存群の instance の成功率を分けて見る。GPU 不要）。
* **判定の根拠は交差効果（他タスクへの影響）だが、係数が効くのはそのタスク自身の学習である。**
  タスク $j$ の OPD 項はタスク $j$ の行にしか触らないので、「他タスクへの害を減らす」ための唯一の操作が
  「そのタスク自身の蒸留を減らす」になる。$b_j$ 1 つで両方を動かすので分離できない。
  (source teacher, target task) の重みが要るが、この損失にその自由度は無い。
* **$n=1$ の学習 run では、alfworld で 5pp 未満の差は決められない。**
* **適応版（学習中に $C$ を測って $b$ を更新する）は採らない。** backward が 6 倍になる測定 step が要り、
  FSDP の勾配取得は 2 度の取り違え（mini-batch 1 つ分、gathered buffer）を起こした経路である。
  静的な $b$ で先に答えを出す。

---

## 7. 判断が残る 1 点

**事前登録する $b$ を A（step 300 の値）にするか B（早期で測り直した値）にするか。**

| | 事前登録 | 意味 |
|---|---|---|
| **A** | $b=(1.076, 1.191, 0.500)$ を今固定 | 「step 300 の順位が早期にも効くか」を検定する。B1 は確認用。null が出ても「早期の順位が違った」と「順位が効かない」を区別できない |
| **B** | B1 を測ってから $b$ を確定 | 適用点と測定点が一致する。**null が「順位が効かない」を意味する**。$b$ の値は測るまで未定 |

**A で登録して B を確認に使う**のと**B で登録する**のは別の実験である。
B1 の順位が step 300 と同じなら両者は同じ $b$ になるが、それは測ってから分かる。
