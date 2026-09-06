# 蒸留目標の 2 つのスカラー — 縮小 $\lambda'$ と外挿 $\lambda$ の導出・訂正・次の測定

状態: **解析とセッション内の数値検証のみ。実装なし・走行なし(2026-09-06)。**
本文書は既存の走行結果を新しく解釈し、既存文書の誤りを 1 つ・欠落を 1 つ特定し、
次に測るべきものを費用順に並べたものである。**新しい測定は含まない。**

前提文書:
[cross_teacher_theory.md](cross_teacher_theory.md)(命題 1–3、階層モデル、識別不能性、§5 の設計、§6 の事前登録)、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)(監査。$g$ ≈ 書式、$\phi$ 表、§10-4 の未解決)、
[cross_teacher_target_design.md](cross_teacher_target_design.md) / [cross_teacher_curriculum_design.md](cross_teacher_curriculum_design.md)(2 モードの設計)、
[cross_teacher_curriculum_related_work.md](cross_teacher_curriculum_related_work.md)(MOPD 周辺のサーベイ)、
[multitask_signweight_150step_handover.md](multitask_signweight_150step_handover.md)(純 OPD 150 step の実測、`agree_rate` 行列)。

---

## 0. 結論(先に)

1. **蒸留目標のスカラーは 1 つではなく 2 つあり、直交する**(§2)。
   $\lambda'$ =「教師群のどちらを向くか」(箱の中の内分)、$\lambda$ =「その方向にどこまで行くか」(箱の外への外挿)。
   一般形は $\log\tilde\pi_d = \log\pi_0 + \lambda\,\sigma_d[\lambda'\hat h_d + (1-\lambda')\overline{\hat h}_{-d}]$ で、
   $(\lambda',\lambda) = (1,1)$ が control。**リポジトリの 2 モードはどちらも $\lambda \le 1$ 側にあり、$\lambda > 1$ は実装が無い。**

2. **$\lambda'$ 縮小の利得は閉形式で出る: MSE 比がちょうど $\lambda'$**(§3、モンテカルロで検証済み)。
   削減率は $1-\lambda' = (K-1)(1-\lambda)/K$ で、$K=3$・$\lambda=0.9$ なら 6.7%、$\lambda=0.5$ なら 33%。
   **向きは $\lambda$ に依らず常に改善側**(Bayes 最適なので当然)で、$\lambda$ が決めるのは大きさだけである。

3. **理論 §3.4 の事後分散の式は誤り**(§4)。$(1-\lambda)^2$ が抜けている。
   現行の式のままだと縮小推定量が素の $h_d$ より悪く見え、Bayes 最適性と矛盾する。**§3.4 の結論(等分散なら位置に依らない)は補正後も変わらない。**

4. **理論 §3.3 の $\lambda'$ 導出は報酬を含んでいない**(§5)。純 OPD では厳密だが、
   GRPO+OPD では固定点が $\pi_d\exp(\tilde Q/\tau)$ で教師分布ではなく、**報酬が $\varepsilon_d$ を既に部分的に補正している**。
   したがって **$\lambda'$ の解析値は GRPO 下では利得の上界**である。null の結論には安全、正の結論には危険。

5. **同タスクの第 2 シード平均が、$\lambda>0.25$ の全域で cross-teacher 縮小に勝つ**(§6)。
   MSE 比 0.5(λ に依らず)対 $\lambda'$。しかもバイアスの危険が無い。**E4 はその第 2 シードを作る run なので、測定と代替案を同時に生む。**

6. **alfworld–webshop のペア類似性が本物なら、$\lambda'$ はスカラーでは足りず教師別ベクトルになる**(§7)。
   内容位置での最適重みは $w_S = 0$ **ちょうど**、$w_W \propto \sigma_p^2\cdot\sigma_\varepsilon^2$。
   **類似性だけでは重みは出ない** — $\sigma_\varepsilon^2 = 0$ なら $\sigma_p^2$ がいくら大きくても $w_W = 0$。

7. **その前提であるペア類似性について、リポジトリの 2 つの測定が正反対を指しており、片方は中心化されていない疑いが強い**(§8)。
   `agree_rate` 行列(webshop↔alfworld 0.761/0.679、search 0.35–0.49)は**周辺分布で中心化されていない生の一致率**で、
   監査が「中心化しない一致率は主に周辺傾向を測る」と警告している当のもの。中心化済みの $\phi$ は A→W に **−0.045**(負)を出す。
   **この矛盾は走行ゼロ・GPU ゼロで決着しうる**(§11 の第 0 手)。

8. **$\lambda>1$(外挿)の余地は純 OPD 側にある**(§9)。命題 1 で到達点 = 目標が天井であり、
   実測で生徒はそこに着いている(`teacher_kl` 0.571 → 0.005 nats)。GRPO+OPD 側では
   固定点 $\pi_d\exp(\tilde Q/\tau)$ が**既に外挿と同じ形**なので、$\lambda>1$ は報酬がやっている仕事の劣化版になる。

9. **優先順位の訂正: E−1 は §6.1 の「最も安く識別力が高い」から、§6.4 で「限界的」に降格済み**(§10)。
   本文書の途中まで §6.1 の順序を引いていたが、後の版が正しい。

---

## 1. 何を扱っているか

対象は `cross_teacher_target.py` の目標構成 $\log\tilde p$ と、その中のスカラー係数である。
機構の**重み**(`sign_weights.py` の $\tilde W$)は本文書の対象ではない — 命題 1・2 により
純 OPD では固定点を動かさず、OPD+PG では比 $\tau$ でしか効かないことが既に決着しているため。

記法は理論文書に合わせる: $h_m = \log\pi_m - \log\pi_0$、$\hat h_m = h_m/\sigma_m$(RMS 標準化)、
$K$ = 教師数(= 3)、$d$ = 行のタスク、$\bar h_{-d}$ = off-task 教師の平均。

---

## 2. $\lambda$ と $\lambda'$ は直交する 2 つのスカラー

目標の一般形:

$$\log\tilde\pi_d = \log\pi_0 + \underbrace{\lambda}_{\text{どこまで}}\cdot\sigma_d\Big[\underbrace{\lambda'\hat h_d + (1-\lambda')\,\mathrm{mean}_{m\ne d}\hat h_m}_{\text{どっち向き}}\Big]$$

| | 意味 | 範囲 | 前提している欠陥 |
|---|---|---|---|
| $\lambda'$ | 教師群のどちらを向くか(内分) | $[1/K, 1]$ | $\pi_d$ は正しい目標の**雑音込みの推定**($h_d = h^\star_d + \varepsilon_d$) |
| $\lambda$ | その方向にどこまで行くか(スカラー倍) | $(0,\infty)$ | $\pi_d$ は**向きは正しいが踏み込みが足りない**(有限 RL・KL 制約) |

**2 つは正反対の診断である。** $\lambda' < 1$ は「教師は行き過ぎて雑音を拾った、引き戻せ」、
$\lambda > 1$ は「教師は足りない、踏み増せ」。単一のグローバルなスカラーとしてはどちらかに賭けることになる。
両方を決めるのは同じ量 $\sigma_\varepsilon^2$ である(§6、§11 の E4)。

### 2.1 現行実装の位置

| モード | 位置 | 根拠 |
|---|---|---|
| `curriculum` | $\lambda \le 1$、終点が $(\lambda',\lambda) = (1,1)$ に**設計で固定** | `log p_tilde = log p_0 + shared + rho_pair(pair−shared) + rho_own(own−pair)`、$\rho=(1,1)$ で $p_{on}$ に一致。設計 P3(到達点不変)・P4(注入ゼロ) |
| `tilt` | 箱を**出られる**が、出る向きが off-task 合意 $s\cdot L$ | モジュール docstring: "The target can leave the interval between base and the on-task teacher … and also what carries other tasks' task-specific components into this task's target"。理論 §3.3 が誤りと判定した方向 |
| **$\lambda > 1$ に沿って $h_d$ を伸ばす** | **未実装** | 文献で唯一天井を破った方向(ExOPD $\lambda=1.25$、math 48.0 対 46.0) |

**@300 の null は発見ではなく設計通りである。** curriculum アームは到達点不変を宣言しているので、
その null を「目標側の操作は効かない」の証拠に数えるのは二重計上になる。この点は本文書の主要な訂正の 1 つ。

---

## 3. $\lambda'$ の Bayes 導出と、MSE 比 = $\lambda'$

理論 §3.3 の推定量:

$$\mathbb E[h^\star_d \mid h] = \bar h + \lambda(h_d - \bar h) = \lambda' h_d + (1-\lambda')\bar h_{-d},
\qquad \lambda' = \frac{1+(K-1)\lambda}{K},\quad \lambda = \frac{\sigma_s^2}{\sigma_s^2+\sigma_\varepsilon^2}$$

**新しい結果(本セッションで導出、モンテカルロで検証):** この推定量の MSE と素の $h_d$ の MSE の比は**ちょうど $\lambda'$** である。

$$\frac{\mathrm{MSE}(\text{縮小})}{\mathrm{MSE}(h_d)} = \frac{\lambda\sigma_\varepsilon^2 + (1-\lambda)^2 T/K}{\sigma_\varepsilon^2} = \lambda + \frac{1-\lambda}{K} = \lambda',
\qquad T := \sigma_s^2+\sigma_\varepsilon^2$$

($\sigma_\varepsilon^2/T = 1-\lambda$ を使う。)

| $\lambda$ | 意味 | $\lambda'$ ($K{=}3$) | MSE 削減 |
|---|---|---|---|
| 0 | 教師の出力は task 知識ゼロ | 0.333 | 67% |
| 0.5 | 知識と雑音が同程度 | 0.667 | 33% |
| 0.9 | 知識 ≫ 雑音 | 0.933 | 6.7% |
| 1.0 | 教師に推定誤差なし | 1.0 | **0%** |

**読み方の要点。** 「$\lambda$ の値によって精度が上がったり上がらなかったりする」のではない。
$\lambda' \le 1$ は常に成り立つので**向きは保証されており、$\lambda$ が決めるのは大きさだけ**である。
問いは「改善するか」ではなく「改善幅が測定雑音(§6.3: alfworld SD 1.39pp)を超えるか」。

### 3.1 MSE 改善は精度改善ではない(未確立の鎖)

**MSE(nats²)→ 成功率(pp) の写像はリポジトリのどこにも確立されていない。** そして反例側の実測がある:

* signweight target アームは目標を書き換えた(位置平均 `target_tv` = 0.0072、発火位置では TV 8% 規模)のに、
  @300 は control と一致(alfworld control 0.754、arm 3 回で 0.754 / 0.778 / 0.754、平均 0.762)。
* $\beta=0.01$ では固定点が $\pi_d\exp(100\tilde Q/W)$ で、目標の中身が固定点を決めるのは $A=0$ の位置だけ(理論 §4.5)。

この鎖が欠けている限り、$\lambda'$ の MSE 削減率から pp を予測することはできない。

---

## 4. 訂正 1 — 理論 §3.4 の事後分散に $(1-\lambda)^2$ が抜けている

`docs/cross_teacher_theory.md:237` の現行:

> 縮小後の事後分散 $\mathrm{Var}[h^\star_d \mid h] = (\sigma_s^2+\sigma_\varepsilon^2)/K + \lambda\sigma_\varepsilon^2$ は
> 等分散の下で**位置に依らない**。

正しくは:

$$\mathrm{Var}[h^\star_d \mid h] = (1-\lambda)^2\,\frac{\sigma_s^2+\sigma_\varepsilon^2}{K} + \lambda\,\sigma_\varepsilon^2$$

**導出**(全分散の法則)。flat prior の下で $g \mid h \sim \mathcal N(\bar h,\ T/K)$、
$s_d \mid g, h_d \sim \mathcal N(\lambda(h_d-g),\ \lambda\sigma_\varepsilon^2)$。$h^\star_d = g + s_d$ なので
$\mathbb E[h^\star_d \mid g,h] = (1-\lambda)g + \lambda h_d$、$\mathrm{Var}(h^\star_d \mid g,h) = \lambda\sigma_\varepsilon^2$。
$g\mid h$ について取り直すと $(1-\lambda)^2 T/K$ が加わる。

**なぜ直す必要があるか。** 現行の式は $\sigma_s^2=\sigma_\varepsilon^2=1$、$K=3$ で 1.167 を与えるが、
素の $h_d$ の MSE は 1.000 である。**Bayes 最適推定量が素の教師より悪いことになり、矛盾する。**
補正後は 0.667 で、モンテカルロの実測 0.6678 と一致する(§12)。

**この訂正が変えないもの。** §3.4 の主張(等分散なら事後分散は位置に依らない / 異分散なら $\tau(x)\propto 1/\sigma_\varepsilon^2(x)$)は
補正後も成り立つ。$\tilde W$ の向きが逆という §3.4 の表も変わらない。転記レベルの誤りである。

**この訂正が可能にするもの。** §3 の「MSE 比 = $\lambda'$」は補正後の式からしか出ない。

---

## 5. 欠落 1 — 理論 §3.3 は報酬を含んでいない

$\lambda'$ の導出は「$h_1,\dots,h_K$ から $h^\star_d$ を推定する」二乗誤差の問題として書かれており、
報酬は式のどこにも入っていない。しかし:

* 命題 2 より、GRPO+OPD の固定点は $\pi^\star \propto \pi_d\exp(\tilde Q/\tau)$ で、**教師分布ではない。**
* $\varepsilon_d$(教師の推定誤差)は定義上タスク成功に寄与しないので、**報酬項は $\varepsilon_d$ のうち害のある部分を自力で押し返す。**

したがって GRPO 下では $\lambda'$ の利得が二重に減る:

1. 目標の重みが小さい($\beta=0.01$、$A\ne0$ の位置では報酬支配 — §4.5)
2. **目標を綺麗にする仕事を、報酬が既に一部やっている**

2 のほうが本質的である。1 だけなら $\beta$ を上げれば取り戻せるが、2 は $\lambda'$ の存在理由そのものを
独立で質の良い信号が横取りしている構図である。

**帰結:**

$$\text{純 OPD: } \lambda' \text{ の解析は厳密} \qquad\Longrightarrow\qquad \text{GRPO+OPD: } \lambda' \text{ の解析は利得の}\textbf{上界}$$

* **null の結論には安全。** 「$\lambda'$ で利得なし」は報酬を入れればさらに強まる向き。
* **正の結論には危険。** 純 OPD で $\lambda'$ が効いても、GRPO に移すと目減りする。移植性は別途測る必要がある。

**文書内の接続不良として記録する。** 同じ理論文書の §3.4 / §4.5 は KL 正則化 RL を近似 Bayes 推論として読み、
教師 = 事前分布・報酬 = データと明示している。§5.1 が設計変数を「目標」と「信頼 $\tau$」に分け、
報酬を $\tau$ 側の話にしたため、目標側の導出から報酬が落ちたと思われる。
**$\lambda'$ は「$h$ の二乗誤差を最小化する目標」であって「GRPO+OPD の目的関数を最大化する目標」ではない。**

### 5.1 推奨する編集(未適用)

`cross_teacher_theory.md` に 2 箇所:

* **`:237`** — 式を $(1-\lambda)^2(\sigma_s^2+\sigma_\varepsilon^2)/K + \lambda\sigma_\varepsilon^2$ に差し替え。
* **§3.3 の末尾(`:228` の「ただし期待利得は測定から小さい」段落の直前か直後)** — 次の趣旨を 2–3 行:
  この導出は報酬を含まない二乗誤差問題であり、純 OPD では到達点そのものだが、
  GRPO+OPD では固定点が $\pi_d\exp(\tilde Q/\tau)$ なので、ここで出る利得は**上界**である。

本文書の作成時点では**適用していない**(本セッションの依頼は文書化のみ)。適用は次の担当の判断で。

---

## 6. 同タスクの第 2 シード平均が $\lambda'$ を支配する条件

$\lambda' < 1$ の利得の源は $\sigma_\varepsilon^2$ である。$\sigma_\varepsilon^2$ を減らす方法は 2 つあり、
**バイアスの危険が無いのは後者だけ**である。

| 方法 | MSE 比 | バイアス危険 | コスト |
|---|---|---|---|
| 他タスク教師 $K-1$ 本で縮小 | $\lambda' = \frac{1+(K-1)\lambda}{K}$ | **あり**(相関した誤りならバイアス注入) | ゼロ(教師は既にある) |
| **同タスク教師の第 2 シードで平均** | **0.5**($\lambda$ に依らず) | **なし** | 教師 1 本の学習 run |

交叉点は $\lambda' = 0.5$、すなわち $K=3$ で $\lambda = 0.25$ ちょうど(モンテカルロで確認、§12):

```
  lam   lam'    cross-teacher    2-seed-on-task      winner
 0.20  0.467   0.468 (pred .467)  0.500 (pred .5)   cross-teacher
 0.25  0.500   0.500 (pred .500)  0.500 (pred .5)   tie  <- crossover
 0.30  0.533   0.535 (pred .533)  0.500 (pred .5)   2-seed on-task
 0.50  0.667   0.669 (pred .667)  0.500 (pred .5)   2-seed on-task
 0.90  0.933   0.934 (pred .933)  0.498 (pred .5)   2-seed on-task
```
(MSE は素の $h_d$ に対する比。全分散 $T = \sigma_s^2+\sigma_\varepsilon^2 = 1$ に固定して $\lambda$ を振ったもの。§12.1 で再現できる。)

**$\lambda > 0.25$ では第 2 シード平均が勝つ。** cross-teacher が勝つのは $\lambda < 0.25$、
すなわち $\sigma_\varepsilon^2 > 3\sigma_s^2$ — 教師の出力の 3/4 以上が RL の推定雑音、という場合だけである。
それが本当なら、それ自体が教師の作り方についての重大な発見になる。

**E4(alfworld 教師の第 2 シード、150 step、理論 §6.1)は $\sigma_\varepsilon^2$ の測定と、
バイアスの無い代替目標の生成を同時に行う。** これが $\lambda'$・$\lambda$ 双方の上流である理由。

---

## 7. ペア成分がある場合の一般化(alfworld–webshop)

### 7.1 モデルの変更

理論 §3.1 は $s_m$ がタスク間で iid(交換可能)と仮定している。
alfworld–webshop が特別に似ているなら交換可能性が壊れ、**ペア成分**が入る:

$$h_A = g + p + u_A + \varepsilon_A,\qquad h_W = g + p + u_W + \varepsilon_W,\qquad h_S = g \phantom{{}+p} + u_S + \varepsilon_S$$

alfworld の目標は $h^\star_A = g + p + u_A$。**$h_W$ は $p$ を含むので目標の一部を直接持つ** —
これまで off-task 教師は $g$ しか運んでいなかったのが変わる。
なお、この 3 層構造(全体共有 / ペア共有 / 固有)は curriculum 設計の P7
($h_m = g + \sum_j s_{mj} + s_m + \varepsilon_m$)と同じもので、理論 §3.1 がペア項を落として単純化したものである。

### 7.2 最適重み(Bayes 線形、数値解)

$\log\tilde\pi_A = \log\pi_0 + \sigma_A[w_A\hat h_A + w_W\hat h_W + w_S\hat h_S]$ として、
$\sigma_u^2 = \sigma_\varepsilon^2 = 1$、flat prior on $g$:

| $\sigma_p^2$ | $w_A$ | $w_W$ | $w_S$ | |
|---|---|---|---|---|
| 0 | 0.667 | 0.167 | 0.167 | **従来の $\lambda'=2/3$ に厳密一致**(検算) |
| 1 | 0.687 | 0.187 | 0.125 | ペア成分で非対称化 |
| 3 | 0.708 | 0.208 | 0.083 | 似ているほど search が削られる |

内容位置($g \approx 0$、監査の「$g$ ≈ 書式」が正しければここが本番)に限ると、和を 1 に正規化して:

| $\sigma_p^2$ | $\sigma_\varepsilon^2$ | $w_A$ | $w_W$ | $w_S$ |
|---|---|---|---|---|
| 1 | 0.25 | 0.933 | 0.067 | **0.000** |
| 1 | 1.0 | 0.833 | 0.167 | **0.000** |
| 3 | 1.0 | 0.786 | 0.214 | **0.000** |
| 1 | **0** | **1.000** | **0.000** | 0.000 |
| 0.3 | 1.0 | 0.906 | 0.094 | **0.000** |

**3 つの読み:**

1. **$w_S = 0$ ちょうど。** $g$ が書式にしか無いなら、内容位置で search は alfworld の目標に一切入れてはならない。
   **現行の一様 $\lambda'$ は $w_W = w_S$ を強制するので、ペア成分が実在するなら構造的に誤りである。**
2. **$w_W \propto \sigma_p^2\cdot\sigma_\varepsilon^2$。** 最終行がそれで、$\sigma_\varepsilon^2 = 0$ なら $\sigma_p^2 = 1$ でも $w_W = 0$。
   **類似性は上限を上げるだけで、利得を作るのは雑音である。**
3. **和は 1 のまま**(flat prior)。これは「search の取り分を webshop に付け替える」操作であって、
   base 方向への追加の縮小ではない。

### 7.3 配線 — `pair_source` が既にある

`nested_layers`(`cross_teacher_target.py:431`)は既に `pair_source` を返している。docstring より:

> ``pair_source`` is the index of the teacher that SET the pair layer -- the loudest agreeing one --
> or -1 where the layer is zero. … the search teacher was trained at a 10x smaller KL coefficient,
> and the target design's D4 asked whether it carries any signal at all.

**「どの教師が pair 層を立てたか」は既に計算され、診断として出力されているだけである。**
必要な変更は $\rho_{pair}$ を `pair_source` ごとに分けること(webshop が立てた層と search が立てた層で別係数)。
curriculum モードの式そのままで、$\rho$ を時間 ramp ではなく**静的な縮小係数**として使う形になる。

---

## 8. §7 の前提は成立するか — 2 つの測定が正反対を指している

**これが §7 全体の生死を決める。**

| 測定 | A–W | A–S | 出典 |
|---|---|---|---|
| **`agree_rate` 行列**(訓練時ログ) | **0.761 / 0.679** | 0.392 / 0.485 | `multitask_signweight_150step_handover.md:379-389`、`sign_weights.py:696` |
| **符号一致 $\phi$**(監査、base 統制後、$\min\lvert\delta\rvert>0.3$ RMS、n=752) | W→A **+0.023** / A→W **−0.045** | S→A **+0.148** / A→S **+0.140** | 監査 §4.1 |

監査 §10-4 はこれを未解決として記録している:

> $\phi$ が最も高いのは search→alfworld(+0.148)と alfworld→search(+0.140)で、
> 旧アームの転移行列(webshop↔alfworld 0.761/0.679、search↔他 0.35–0.49)と一致しない。
> **2つの推定量のどちらが何を測っているかは未解決。**

### 8.1 解決の仮説(本文書の主張、未検証)

**`agree_rate` は周辺分布で中心化されていない。** 定義(handover §12.5、`sign_weights.py:542-543`)は
「両教師がデッドゾーン外だった候補のうち符号が一致した割合」で、**独立期待値を引いていない**。
そして監査 §4.1 が当のものについて警告している:

> $P(\text{on}\uparrow)$ が 0.24〜0.89 と偏っているので、**中心化しない一致率は主に周辺傾向を測る。**

$\phi$ 表の周辺分布を見ると webshop 教師は $P(\text{src}\uparrow) = 0.896$、
つまり **9 割の候補で上げ向き**である。そのような教師は何と比べても高い生の一致率を出す。

**補強**: handover の著者自身が同じページで
「`webshop__on__alfworld` は 0.732 → 0.787 に上がる。**生徒が動くと行列も動く**ので、
この行列を『タスク間類似度』として引用するときは step を明記すること」と書いている。
凍結モデル同士の類似度が生徒とともに動くなら、それは支持集合(生徒 top-20)経由の効果であって、
純粋な教師間類似度ではない。中心化すればこの依存も部分的に落ちる。

### 8.2 未解決のまま残ること

**現時点でどちらが正しいかは言えない。** 監査の周辺分布(P(src↑)=0.896 等)は
**別 run・別支持・別フィルタ**(klw アーム、$\lvert\delta\rvert>0.3$ RMS、n=752)なので、
`agree_rate` の母集団(生徒 top-20、$\epsilon = 0.1$ nats デッドゾーン、signweight アーム)にそのまま転用できない。
判定は二択で、どちらでも次が決まる:

* **chance が 0.75 前後** → 0.761 は関連ほぼゼロ。出力空間に $\sigma_p^2$ は無く、§7 の非対称設計は根拠を失う。§10-4 も同時に片付く。
* **chance が 0.5 前後** → 0.761 は本物の超過。$w_S = 0$ / $w_W > 0$ の非対称設計に測定上の根拠が立つ。

**なお、パラメータ空間での A≈W が正しいとしても §7 の根拠にはならない。**
損失が作用するのは出力空間の、生徒が訪れた位置の、生徒 top-20 候補の $h$ 値である。
パラメータ空間の類似方向が訪問状態の候補に発現しなければ $\sigma_p^2 = 0$ である
(理論 §3.5 の測度の議論と同じ論点)。

---

## 9. $\lambda > 1$(外挿)の余地の所在

### 9.1 regime — 純 OPD にあり、GRPO+OPD にほぼ無い

**純 OPD(`pg_loss_coef=0`)に余地がある:**

* 命題 1 により到達点は $\pi_d$ ちょうど。**目標を動かす以外に上がる道が原理的に存在しない。**
* 生徒は既にそこに着いている: `teacher_kl` $= \mathrm{KL}(p_s\Vert\tilde p)$ が **0.571 → 0.005 nats**(150 step)、
  $\text{on\_travel} = 1 - 0.005/0.571 \approx 0.99$。handover §3.2 が「**残り伸びしろがほとんど無い状態に入っている**」と明記。

**GRPO+OPD($\beta=0.01$)には無い:**

* 固定点 $\pi^\star \propto \pi_d\exp(\tilde Q/\tau)$ を base 相対で書くと $h^\star = h_d + \tilde Q/\tau$ — **外挿と同じ形**である。
  つまり **GRPO は既に報酬に基づいた外挿をやっており**、$h_d$ 方向への盲目的な倍化より根拠のある方向で進んでいる。
* 実測でも control@300 の `off_travel` が 6 ペアすべてで **1.07–2.11**(生徒は自分の教師より遠くまで進んでいる)。
  ただしこの量は off-task 教師からの距離軸であって base→on-task 教師軸ではないので、**示唆であって証明ではない**。
  コード側の但し書き(`sign_weights.py:1367`)も「`on_travel` と一緒に読め、単独で読むな」と言っている。

### 9.2 タスク — 前提が成立するのは 3 つ中 2 つ

外挿には「生徒が既に教師に着いている」前提が要る。@150 の内訳(handover §3.2):

| task | `teacher_kl` | `target_kl_ratio` | 前提 |
|---|---|---|---|
| alfworld | 0.004–0.006 | 35.4% | **成立** |
| webshop | 0.004–0.006 | 22.3% | 成立(ただし klw 系で一貫して悪化した実績あり) |
| search | **0.025**(他の 4–6 倍) | 3.5% | **不成立** — まだ着いていない |

**search を外挿アームに入れるのは筋が悪い。**

### 9.3 レバーの大きさと実装コスト

局所的に $\mathrm{KL}\approx\frac12\chi^2$ なので $\sqrt{\text{ratio}}$ が距離比になり、
$\sqrt{0.252}\approx0.50$ — **生徒の残距離は現行の書き換え量の約 2 倍**(alfworld は $\sqrt{0.354}\approx0.60$)。
handover §10.3 の予測「**150→300 の区間こそが、target の固定点移動が目的関数を支配する区間である**」がここから出ており、
**その区間はまだ走っていない。**

実装は `build_target` に $c \mathrel{+}= (\lambda-1)\sigma_d\hat h_d$ を足すだけ。
1 RMS = 2.148 nats なので $\lambda=1.25$ の典型変位は約 0.54 nats、`_EXPONENT_CLAMP = 5.0` に対して
中央値では余裕があるが**裾では当たる**ので、clamp 発火率を診断に出すこと。

---

## 10. 優先順位の訂正 — E−1 は降格済み

理論 §6.1 は E−1(既存 checkpoint 12 点の検証、学習なし)を「最も安く、最も識別力が高い」としていたが、
**後から入った §6.4(2026-09-03)がこれを「限界的」に落としている**:

> 形の問題は学習曲線が既に 300 点で答えている(§4.3)。各 step の alfworld は 15 プロンプト × `env.rollout.n=8` = 120 本で
> 検証の 126 問と標本サイズがほぼ同じ、点数は 25 倍。E−1 が足すのは「held-out・T=0.4 でも同じ形か」だけ

24 run / 約 15 時間に対して得られるものが小さい。**§6.4 が後の版であり、そちらが正しい。**
§6.4 自身の結論は、残る測定可能な問いは 2 つ:
(a) @300 の相殺が control の 2 本目でも残るか(**ctl@300、38 分**)、
(b) 早期優位と相殺がどのトークンから来るか(dump 付き再現、数十時間)。

---

## 11. 次に何をするか(費用順)

| # | 何を | コスト | 何が決まる |
|---|---|---|---|
| **0** | `agree_rate` の chance baseline と $\phi$ を、同じ母集団で出す | **ほぼゼロ**(数行 + 既存ダンプ/次回 run) | 出力空間に $\sigma_p^2$ があるか。監査 §10-4 の未解決。**§7 の非対称 $\lambda'$ の生死** |
| **2** | `ctl@300` × 1 | **38 分 GPU**、第 0 手と**並行可** | @300 の相殺が本物か(control の n=1 を外す)。control の per-instance dump が初めて残る |
| 1 | 同じ母集団で共分散($\sigma_p^2$ の大きさ版) | 安い、オフライン | $w_W$ の値そのもの。第 0 手が「生きている」と出た場合のみ |
| 3 | E4(alfworld 教師の第 2 シード、150 step) | 学習 run | $\sigma_\varepsilon^2$ → $\lambda'$ と $\lambda$ の両方、第 2 シード平均の可否 |
| 4 | dump 付きの 8/13 設定の再現(seed 2、`token_stats` / `event_dump`) | 数十時間 | 早期優位(@150 の alfworld +11.1pp、3.3–5.7σ)と相殺が**どのトークンから**来るか。§6.4 曰く「唯一『なぜ』に届く」 |

**第 0 手と第 2 手を先に、並行で。** 第 0 手は GPU を使わずに §7 全体が成立するかを決める。

### 11.1 第 0 手のパッチ仕様

`verl/trainer/ppo/sign_weights.py`。既存の `pair_both` / `pair_same`(`:444-445`、`:542-543`、`:696`)と
**同じ分母 `both` の上で**カウンタを 3 つ足す:

```python
# :542-543 の隣に
self._add(self.pair_on_pos,   (t, m_task), int(((on[r] > 0) & both).sum()))
self._add(self.pair_off_pos,  (t, m_task), int(((so[r] > 0) & both).sum()))
self._add(self.pair_both_pos, (t, m_task), int(((on[r] > 0) & (so[r] > 0) & both).sum()))
```

`:696` の隣で、$N$ = `both`、$n_{1\cdot}$ = `pair_on_pos`、$n_{\cdot1}$ = `pair_off_pos`、$n_{11}$ = `pair_both_pos` として:

$$q_{on} = \frac{n_{1\cdot}}{N},\quad q_{off} = \frac{n_{\cdot1}}{N},\qquad
\text{chance} = q_{on}q_{off} + (1-q_{on})(1-q_{off})$$

$$\phi = \frac{N\,n_{11} - n_{1\cdot}n_{\cdot1}}{\sqrt{n_{1\cdot}(N-n_{1\cdot})\,n_{\cdot1}(N-n_{\cdot1})}},
\qquad \text{excess} = \frac{\text{observed} - \text{chance}}{1 - \text{chance}}$$

を `agree_chance` / `agree_phi` / `agree_excess` として `agree_rate` の隣に出す。

**注意 2 点:**

* **rank 集約。** `agree_rate` は rank-0 の量である旨がコードにある(`:1954` 付近の但し書き)。
  新しい指標も同じ扱いにするか、カウンタを全 rank で集約するかを明示すること。
  比の平均は比ではないので、**カウンタを出して比は後で作る**のが安全(既存の `pair_both` がそうしている理由と同じ)。
* **母集団。** この $\phi$ は生徒 top-20 / $\epsilon=0.1$ nats デッドゾーンの上の量で、
  監査 §4.1 の $\phi$($\lvert\delta\rvert>0.3$ RMS、n=752)とは母集団が違う。**直接比較しないこと。**
  比較したいなら監査側の母集団を再現するか、両方を並べて報告する。

### 11.2 第 0 手をオフラインで回す場合

`scripts/cross_teacher_offline_audit.py` が既存ダンプを読む経路を持っている。
ただし監査 §2 の通り、**本物の control のダンプは別ホスト**(`/opt/home/ohara/sign_tokens/..._klw_control_qwen3_1.7b_xt1`)にあり、
tamago 上のものは treatment run(RUN_TAG=xt1)である。**どのダンプを読んだかを必ず記録すること** —
監査の初版はこれを取り違えている。

---

## 12. 再現用の計算

本文書の数値は次の 2 つの自己完結したスクリプトで再現できる(numpy 不要、標準ライブラリのみ)。

### 12.1 MSE 比 = $\lambda'$、第 2 シード平均、および §4 の訂正

```python
import random, math
random.seed(0); N = 300_000; K = 3
print("  lam   lam'   cross  (pred)   2seed  (0.5) | doc-var  corrected  empirical")
for lam in [0.20, 0.25, 0.30, 0.50, 0.90]:
    ss, se = lam, 1.0-lam                      # total variance T = 1
    lamp = (1+(K-1)*lam)/K; T = ss+se
    sp, sq = math.sqrt(ss), math.sqrt(se)
    a = b = c = 0.0
    for _ in range(N):
        g = random.gauss(0,3.0)
        s = [random.gauss(0,sp) for _ in range(K)]
        e = [random.gauss(0,sq) for _ in range(K)]
        h = [g+s[m]+e[m] for m in range(K)]
        star = g + s[0]; hbar = sum(h)/K
        shrunk = hbar + lam*(h[0]-hbar)                    # theory 3.3 posterior mean
        e2 = random.gauss(0,sq); seed2 = g + s[0] + (e[0]+e2)/2   # 2nd seed of the SAME teacher
        a += (h[0]-star)**2; b += (shrunk-star)**2; c += (seed2-star)**2
    print(f" {lam:5.2f}  {lamp:5.3f}  {b/a:6.3f} ({lamp:.3f})  {c/a:6.3f} (0.500) |"
          f" {T/K+lam*se:7.3f}  {(1-lam)**2*T/K+lam*se:9.3f}  {b/N:9.3f}")
```

出力:

```
  lam   lam'   cross  (pred)   2seed  (0.5) | doc-var  corrected  empirical
  0.20  0.467   0.468 (0.467)   0.500 (0.500) |   0.493      0.373      0.374
  0.25  0.500   0.500 (0.500)   0.500 (0.500) |   0.521      0.375      0.375
  0.30  0.533   0.535 (0.533)   0.500 (0.500) |   0.543      0.373      0.374
  0.50  0.667   0.669 (0.667)   0.500 (0.500) |   0.583      0.333      0.335
  0.90  0.933   0.934 (0.933)   0.498 (0.500) |   0.423      0.093      0.093
```

読み方: `cross` 列が予測 $\lambda'$ と一致すること(§3 の主結果)、`2seed` 列が $\lambda$ に依らず 0.5 であること(§6)、
そして **`corrected` 列だけが `empirical` と一致し `doc-var` 列は一致しない**こと(§4 の訂正)。

### 12.2 §7 のペア成分つき最適重み

```python
def solve(gg, pp, uu, ee):
    """h_A=g+p+u_A+e_A, h_W=g+p+u_W+e_W, h_S=g+u_S+e_S; target T=g+p+u_A."""
    V, VS = gg+pp+uu+ee, gg+uu+ee
    M = [[V, gg+pp, gg], [gg+pp, V, gg], [gg, gg, VS]]
    c = [gg+pp+uu, gg+pp, gg]
    A = [row[:] + [c[i]] for i, row in enumerate(M)]
    for i in range(3):
        piv = max(range(i,3), key=lambda r: abs(A[r][i])); A[i], A[piv] = A[piv], A[i]
        for r in range(3):
            if r == i: continue
            f = A[r][i]/A[i][i]
            for k in range(i,4): A[r][k] -= f*A[i][k]
    return [A[i][3]/A[i][i] for i in range(3)]

print(solve(1e7, 0.0, 1.0, 1.0))   # (0.667, 0.167, 0.167) == uniform lambda' = 2/3
print(solve(1e7, 1.0, 1.0, 1.0))   # (0.687, 0.187, 0.125)
print(solve(0.0,  1.0, 1.0, 1.0))  # (0.625, 0.125, 0.000) -> renormalised (0.833, 0.167, 0)
print(solve(0.0,  1.0, 1.0, 0.0))  # (1.000, 0.000, 0.000)  <- noiseless teacher
```

`solve(1e7, 0.0, ...)` が理論 §3.3 の一様 $\lambda'$ に厳密一致することが、
§7 の一般化が既存の導出の上位互換であることの検算になっている。

---

## 13. 限界・未検証

* **本文書に新しい測定は無い。** §3・§6・§7 は解析とモンテカルロ、§8・§9 は既存文書からの再読である。
  「$\sigma_p^2$ が出力空間にあるか」「$\sigma_\varepsilon^2$ の大きさ」はどちらも**未測定**で、
  本文書の設計提案はすべてその 2 つに条件付きである。
* **§8.1 の解決仮説(中心化の欠如)は仮説である。** 監査 §4.1 の警告と handover の step 依存性という
  2 つの状況証拠に基づくが、`agree_rate` の母集団での chance baseline は計算されていない。
  **反対の結果(chance ≈ 0.5)も十分ありうる。**
* **§7 のモデル(g / p / u / ε の 4 層、ガウス、$u$ と $\varepsilon$ が iid)は仮定である。**
  off-task 教師が task d の内容位置で**相関した誤り**を持つ場合、混入は平均 0 雑音ではなくバイアスで、
  MSE は上がりうる。klw 系が webshop を一貫して悪化させた実績(理論 §4.6)がその方向の状況証拠。
* **§9.1 の `off_travel` 1.07–2.11 は off-task 教師からの距離軸**であって、
  base→on-task 教師軸ではない。「生徒が既に外挿している」の**示唆であって証明ではない**。
* **MSE → pp の写像は無い**(§3.1)。本文書の削減率(%)を成功率(pp)に翻訳してはならない。
* **反復雑音の推定は n=3** で、$\sigma$ の 95% CI は $[0.52\hat\sigma, 6.3\hat\sigma]$(理論 §6.3)。
  上限側を取っても変わらない主張だけを述べること。
* **`agree_rate` の rank 集約の扱いを確認していない。** `:1954` 付近に rank-0 である旨の但し書きがあるが、
  第 0 手の新指標にどう効くかは未検討(§11.1 の注意 1)。
* **`_EXPONENT_CLAMP` が $\lambda>1$ でどれだけ発火するかは未測定**(§9.3)。中央値の見積りのみ。
