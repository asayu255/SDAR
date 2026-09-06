# 測定精度で決める蒸留目標(measured-precision target)— 提案手法と事前登録

状態: **提案。解析とセッション内の数値検証のみ、実装なし・走行なし(2026-09-06)。**
走らせる前に §7 の事前登録を固定すること。

前提文書:
[cross_teacher_theory.md](cross_teacher_theory.md)(命題 1–3、階層モデル、識別不能性、原則 1/2)、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)(監査。$g$ ≈ 書式、$\phi$ 表、§10-4 の未解決)、
[cross_teacher_target_design.md](cross_teacher_target_design.md) / [cross_teacher_curriculum_design.md](cross_teacher_curriculum_design.md)(既存 2 モード)、
[multitask_signweight_150step_handover.md](multitask_signweight_150step_handover.md)(純 OPD 150 step の実測、`agree_rate` 行列)、
[cross_teacher_curriculum_related_work.md](cross_teacher_curriculum_related_work.md)(MOPD 周辺)。

**機構**: 蒸留目標を $\log\pi_0 + \lambda\sum_m w_m\sigma_d\hat h_m$ の形に置き、
**$w$(向き)を探索ではなく教師集合の分散分解の測定から同定する**。
$w$ の同定に必要な量は 2 つだけで、どちらも既存の配管で測れる。

---

## 0. 結論(先に)

1. **提案は「新しい重み付け」ではなく「係数を測定で同定する目標族」である。**
   既存の全アーム(klw の $\tilde W$、tilt の $c$、curriculum の $\rho$)は係数を設計者が決めていた。
   本提案の係数は、教師間共分散 $\hat\Sigma$ と教師の推定誤差 $\sigma_\varepsilon^2$ の関数である(§3)。
   **ハイパラは $\lambda$(外挿量)1 つだけに減る。**

2. **3 段階で、各段の測定が次段を走らせるかを決める**(§2)。
   **S1 自己縮小**(同タスク教師の第 2 シード平均)→ **S2 非対称 cross-teacher 縮小** → **S3 外挿**。
   S1 は cross-teacher の仮定を一切使わず、バイアスの危険もなく、MSE を確実に半分にする。
   S2 は S1 が副産物として与える $\sigma_\varepsilon^2$ なしには係数が決まらない。S3 は S1/S2 と直交する別軸。

3. **S1 が本命である。** MSE 比は $\lambda$ に依らず 0.5、対して cross-teacher 縮小は $\lambda'$。
   **$\lambda > 0.25$ の全域で S1 が S2 に勝つ**(§4.3)。S2 が勝つのは $\sigma_\varepsilon^2 > 3\sigma_s^2$、
   すなわち教師の出力の 3/4 以上が RL の推定雑音という場合だけである。

4. **S2 の価値は「勝つこと」ではなく「cross-teacher 系全体を畳めること」にある**(§7.4)。
   理論 §5.2 が書いている通り、これは cross-teacher 目標の**最も弱い仮定で最も強い形**で、
   ここで null なら符号ゲート付きの変種を走らせる理由が無くなる。**事前予測は null 側**である。

5. **S2 の前提(alfworld–webshop のペア成分)は、走行ゼロで決着しうる**(§6)。
   リポジトリの 2 つの測定が正反対を指しており、片方は周辺分布で中心化されていない疑いが強い。
   **カウンタ 3 つで判定できる**(§8.1)。

6. **S3(外挿)の家は純 OPD である**(§5)。命題 1 で到達点 = 目標が天井であり、実測で生徒はそこに着いている
   (`teacher_kl` 0.571 → 0.005 nats)。GRPO+OPD 側は固定点 $\pi_d\exp(\tilde Q/\tau)$ が既に外挿と同じ形なので、
   $\lambda>1$ は報酬がやっている仕事の劣化版になる。

7. **本提案は理論文書の 2 点の修正の上に立つ**(§3.4、§3.5)。
   (i) §3.4 の事後分散に $(1-\lambda)^2$ が抜けている。(ii) §3.3 の導出に報酬が入っていないので、
   GRPO 下では利得の**上界**として読む必要がある。**どちらも本提案の主張の向きを変えないが、
   (i) を直さないと S1 と S2 の比較そのものが成立しない。**

---

## 1. 現行機構からの差分

| | 目標の形 | 係数の決め方 | 到達点 |
|---|---|---|---|
| control(MOPD の routing) | $\log\pi_d$ | — | 教師 |
| `tilt` | $\log\pi_{on} + c$、$c = s\cdot L$ 系 | `exponent_scale` を設計者が選ぶ | 教師の外(off-task 方向) |
| `curriculum` | $\log\pi_0 + \text{shared} + \rho_{pair}\Delta + \rho_{own}\Delta$ | $\rho$ を step の関数として設計者が選ぶ | **設計で control に固定** |
| **本提案** | $\log\pi_0 + \lambda\sum_m w_m\sigma_d\hat h_m$ | **$w$ は $\hat\Sigma$ と $\sigma_\varepsilon^2$ から計算**。$\lambda$ のみハイパラ | $\lambda=1$ で教師群の張る範囲内、$\lambda>1$ で外 |

差分を 1 行で言うと、**「どの位置を強く当てるか」(重み)から「どこを目標に置くか」(目標)へ、
そして「目標を設計者が選ぶ」から「目標を測定が決める」へ**の 2 段の移動である。
前段は `tilt` / `curriculum` で既に済んでおり(監査 P1 と命題 1 が強制した)、本提案は後段だけを足す。

### 1.1 文献に対する位置

| 先行 | 何をしているか | 本提案との差 |
|---|---|---|
| **MOPD**(Ma+ 2026) | routing。$w = e_d$ を**天下り**で置く | 同じ族の $w$ を測定で決める。$w = e_d$ は $\sigma_\varepsilon^2 = 0$ の特別な場合として出てくる |
| **MOD**(Shi+ 2024 NeurIPS) | $f$-divergence で整合した複数モデルの**線形結合を閉形式で** | **最も近い競合。** 理論 §3.3 の幾何混合そのもの。差は (a) 同タスク第 2 シードによる $\sigma_\varepsilon^2$ の直接同定、(b) ペア成分を含む非対称 $w$ |
| **ExOPD / G-OPD**(Yang+ 2026) | $\log\pi_\theta = \lambda\log\pi^\star + (1-\lambda)\log\pi_{ref}$、$\lambda = 1.25$ 固定 | S3 と同じ軸。$\lambda$ を探索で決めている点は同じで、本提案も $\lambda$ は同定できない(§5.4 で明示) |
| **TAID**(ICLR 2025) | 目標を時間で補間 | 係数が時間の関数。本提案は静的で、時間軸とは直交(併用可) |
| **Open-MOPD**(Gao+ 2026) | 多教師 OPD の失敗は**予算配分**であって干渉ではない | **直交する対立仮説。** 本提案が null でも Open-MOPD の主張は生き残る。§9 の限界に明記 |

**新規性の主張は 1 点だけ**: 蒸留目標の係数を、教師集合の分散分解の**測定**から同定すること。
特に $\sigma_\varepsilon^2$(教師の推定誤差)を同タスクの第 2 シードで直接測る手続きは、上表のどれにも無い。

---

## 2. 提案手法

### 2.1 目標族

行のタスク $d$、候補 $(x,v)$ について、支持は生徒 top-20 + tail(現行と同じ):

$$\boxed{\ \log\tilde\pi_d(v\mid x) = \log\pi_0(v\mid x) + \lambda\,\sigma_d\sum_{m=1}^{K} w^{(d)}_m\,\hat h_m(v\mid x) - \log Z(x),\qquad \sum_m w^{(d)}_m = 1\ }$$

* $\hat h_m = h_m/\sigma_m$ は RMS 標準化した shift、$\sigma_d$ で宛先の単位に戻す(理論 §3.3 の「RMS 単位の扱い」と同じ)。
* $(w, \lambda) = (e_d, 1)$ で **control と bit-identical**。
* tail は `tail_on` / `tail_off` を同じ式に入れるので $Z$ は支持と tail の和で閉じる(現行と同じ)。
* **符号ゲート・全会一致・min・deadzone は無い。** 理論 §3.3 の通り、Bayes 最適な形にはどれも現れない。

### 2.2 3 段階と、その間のゲート

```
          [測定 M1: 中心化した教師間一致]        [測定 M2: sigma_eps^2]
                     |                                   |
                     v                                   v
  S1 自己縮小 ---------------------------------> S2 非対称 cross-teacher 縮小
  (w = e_d, teacher = 2 seed 平均)                (w = Bayes 線形解)
       |                                               |
       +--------------------> S3 外挿 (lambda > 1, 純 OPD のみ) <---+
```

| 段 | 何を変える | 必要な測定 | 走らせる条件 |
|---|---|---|---|
| **S1** 自己縮小 | $\pi_d$ を**同タスク教師 2 シードの幾何平均**に置き換える。$w = e_d$ のまま | なし(S1 自身が M2 を生む) | 無条件。cross-teacher の仮定を一切使わない |
| **S2** 非対称縮小 | $w$ を Bayes 線形解に置き換える | **M1**(ペア成分の実在)+ **M2**($\sigma_\varepsilon^2$) | M1 が「ペア成分あり」かつ M2 が $\lambda < 0.25$ |
| **S3** 外挿 | $\lambda > 1$ | なし($\lambda$ は同定できない、§5.4) | 純 OPD、alfworld / webshop のみ |

**S1 → S2 のゲートが厳しいのは意図的である。** §4.3 の通り $\lambda > 0.25$ では S1 が S2 に勝つので、
S2 を走らせる意味があるのは M2 が $\lambda < 0.25$ を出したときだけである。

### 2.3 S1 — 自己縮小(同タスク第 2 シード)

$$\hat h_d \ \longleftarrow\ \tfrac12\big(\hat h_d^{(1)} + \hat h_d^{(2)}\big)$$

対数空間の平均なので、分布としては 2 シードの**幾何平均**である。

* **バイアスがゼロ。** 2 つのシードは同じ $h^\star_d$ の推定で、混ざるのは $\varepsilon$ だけ。
  cross-teacher 縮小が抱える「他タスクの固有成分の混入」という危険が構造的に無い。
* **MSE 比は 0.5、$\lambda$ に依らない**(§4.2)。
* **副産物として $\sigma_\varepsilon^2$ が測れる**: $\mathrm{Var}(h_d^{(1)} - h_d^{(2)}) = 2\sigma_\varepsilon^2$(理論 §3.2)。
  これが M2 で、S2 の係数も S3 の判断もこれに依存する。
* コストは教師 1 本の学習 run(理論 §6.1 の E4 と同じもの)。

### 2.4 S2 — 非対称 cross-teacher 縮小

理論 §3.1 は $s_m$ がタスク間で iid(交換可能)と仮定しており、その下では $w$ は
$(\lambda', \frac{1-\lambda'}{K-1}, \dots)$ という**一様**な形になる。
alfworld–webshop に固有のペア成分があるなら交換可能性は壊れ、$w$ は非対称になる:

$$h_A = g + p + u_A + \varepsilon_A,\qquad h_W = g + p + u_W + \varepsilon_W,\qquad h_S = g\phantom{{}+p} + u_S + \varepsilon_S$$

($g$ = 全体共有、$p$ = A–W のペア共有、$u$ = 固有、$\varepsilon$ = 推定誤差。
この 3 層は curriculum 設計の P7 $h_m = g+\sum_j s_{mj}+s_m+\varepsilon_m$ と同じもので、
理論 §3.1 がペア項を落として単純化していた。)

alfworld の目標は $h^\star_A = g+p+u_A$。$w$ は Bayes 線形解として閉形式で出る(§4.4 に数値)。

**この段の予測は 2 つとも強い**:

1. **$w_S = 0$ ちょうど**(内容位置、$g\approx0$)。search は alfworld の目標に一切入らない。
2. **$w_W \propto \sigma_p^2\cdot\sigma_\varepsilon^2$。$\sigma_\varepsilon^2 = 0$ なら $\sigma_p^2$ がいくら大きくても $w_W = 0$。**
   **類似性は上限を上げるだけで、利得を作るのは雑音である。**

### 2.5 S3 — 外挿

$\lambda > 1$。$w$ が決めた方向の延長線上に目標を置く。**純 OPD、alfworld / webshop のみ**(理由は §5)。

---

## 3. 理論: 言えること・言えないこと

### 3.1 言えること — この族が正しい形であること

* **重みではなく目標である必要がある。** 命題 1 より、純 OPD の位置スカラー重みは
  $\arg\min$ を動かさない。目標を動かす以外に到達点を変える経路が無い。
* **教師統計を入れてよいのは目標だけである。** 理論 §5.1 の原則 2。信頼 $\tau$ に教師統計を流すのは
  識別不能性(§3.2)に反する。本提案は $\tau$ に触らない。
* **Bayes 最適な形に符号ゲートは現れない。** 出てくるのは平均と一様/非一様な縮小だけ(理論 §3.3)。
* **共有成分の除去(contrastive 形)は誤り。** $g$ は目標 $h^\star_d$ の一部である。
  監査が同一標本で検定して悪化側だった(sg1 −0.042、xt1 −0.035)ことと整合。

### 3.2 言えること — $\lambda'$ の利得は閉形式で出る

縮小推定量の MSE と素の $h_d$ の MSE の比は**ちょうど $\lambda'$** である(§4.2 で数値検証):

$$\frac{\mathrm{MSE}(\text{縮小})}{\mathrm{MSE}(h_d)} = \frac{\lambda\sigma_\varepsilon^2 + (1-\lambda)^2 T/K}{\sigma_\varepsilon^2} = \lambda + \frac{1-\lambda}{K} = \lambda',\qquad T := \sigma_s^2+\sigma_\varepsilon^2$$

**向きは $\lambda$ に依らず常に改善側**(Bayes 最適なので当然)で、$\lambda$ が決めるのは大きさだけ。
問いは「改善するか」ではなく「改善幅が測定雑音を超えるか」である。

### 3.3 言えないこと — MSE 改善は精度改善ではない

**MSE(nats²)→ 成功率(pp)の写像はリポジトリのどこにも無い。** そして反例側の実測がある:

* signweight target アームは目標を書き換えた(位置平均 `target_tv` = 0.0072、発火位置では TV 8% 規模)のに、
  @300 は control と一致(alfworld control 0.754、arm 3 回で 0.754 / 0.778 / 0.754、平均 0.762)。
* $\beta=0.01$ では固定点が $\pi_d\exp(100\tilde Q/W)$ で、目標の中身が固定点を決めるのは $A=0$ の位置だけ(理論 §4.5)。

**この鎖が欠けている限り、§4 の削減率(%)を pp に翻訳してはならない。** §7.2 の予測はすべて pp で書く。

### 3.4 前提の修正 (i) — 理論 §3.4 の事後分散

`cross_teacher_theory.md:237` の現行は

$$\mathrm{Var}[h^\star_d \mid h] = (\sigma_s^2+\sigma_\varepsilon^2)/K + \lambda\sigma_\varepsilon^2$$

だが、正しくは $(1-\lambda)^2$ が掛かる:

$$\mathrm{Var}[h^\star_d \mid h] = (1-\lambda)^2\,\frac{\sigma_s^2+\sigma_\varepsilon^2}{K} + \lambda\,\sigma_\varepsilon^2$$

**導出**(全分散の法則)。flat prior の下で $g\mid h \sim \mathcal N(\bar h, T/K)$、
$s_d \mid g,h_d \sim \mathcal N(\lambda(h_d-g), \lambda\sigma_\varepsilon^2)$。
$\mathbb E[h^\star_d\mid g,h] = (1-\lambda)g+\lambda h_d$、$\mathrm{Var}(h^\star_d\mid g,h) = \lambda\sigma_\varepsilon^2$。
$g\mid h$ について取り直すと $(1-\lambda)^2 T/K$ が加わる。

**なぜ本提案に効くか。** 現行の式は $\sigma_s^2=\sigma_\varepsilon^2=1$、$K=3$ で 1.167 を与えるが、
素の $h_d$ の MSE は 1.000 である。Bayes 最適推定量が素の教師より悪いことになり矛盾する。
補正後は 0.667 で、実測 0.666 と一致する(§4.2)。**この修正なしには S1 と S2 の比較(§4.3)が成立しない。**
§3.4 の結論(等分散なら位置に依らない / 異分散なら $\tau(x)\propto1/\sigma_\varepsilon^2(x)$)は補正後も変わらない。

### 3.5 前提の修正 (ii) — 理論 §3.3 は報酬を含まない

$\lambda'$ の導出は「$h_1,\dots,h_K$ から $h^\star_d$ を推定する」二乗誤差の問題で、報酬が入っていない。しかし
命題 2 より GRPO+OPD の固定点は $\pi^\star\propto\pi_d\exp(\tilde Q/\tau)$ で**教師分布ではない**。
そして $\varepsilon_d$ は定義上タスク成功に寄与しないので、**報酬項は $\varepsilon_d$ の害のある部分を自力で押し返す**。

したがって GRPO 下では本提案の利得が二重に減る:

1. 目標の重みが小さい($\beta=0.01$、$A\ne0$ の位置では報酬支配)
2. **目標を綺麗にする仕事を報酬が既に一部やっている**

$$\text{純 OPD: 解析は厳密}\qquad\Longrightarrow\qquad\text{GRPO+OPD: 解析は利得の}\textbf{上界}$$

* **null の結論には安全。** 「効かない」は報酬を入れればさらに強まる向き。
* **正の結論には危険。** 純 OPD で効いても GRPO に移すと目減りする。**移植性は別途測る**(§7.3)。

**推奨する編集(本提案では未適用)**: `cross_teacher_theory.md:237` の式の差し替えと、
§3.3 末尾(`:228` 付近)に「この導出は報酬を含まない二乗誤差問題であり、GRPO 下では上界」の 2–3 行。

---

## 4. 係数の同定

### 4.1 必要な測定は 2 つだけ

| | 記号 | どう測るか | コスト |
|---|---|---|---|
| **M1** | $\sigma_p^2$(ペア成分) | 中心化した教師間一致 / 共分散。既存の `agree_rate` の分母の上でカウンタ 3 つ(§8.1) | **ほぼゼロ** |
| **M2** | $\sigma_\varepsilon^2$ | $\mathrm{Var}(h_d^{(1)}-h_d^{(2)}) = 2\sigma_\varepsilon^2$。**S1 の副産物** | 教師 1 本の学習 run |

既存ダンプの 2 次モーメントは $\mathrm{Var}(g)$ と**和** $\sigma_s^2+\sigma_\varepsilon^2$ を識別する(理論 §3.2)。
M2 が $\sigma_\varepsilon^2$ を与えれば差し引きで $\sigma_s^2$ が出て、$\lambda$、$\lambda'$、$w$ がすべて計算で確定する。
**$w$ はハイパラではなくなる。**

### 4.2 $\lambda'$ と S1 の MSE(数値検証済み)

$K=3$、全分散 $T=1$ に固定して $\lambda$ を振った 300k サンプルのモンテカルロ(再現は §10.1):

```
  lam   lam'   cross  (pred)   2seed  (0.5) | doc-var  corrected  empirical
  0.20  0.467   0.468 (0.467)   0.500 (0.500) |   0.493      0.373      0.374
  0.25  0.500   0.500 (0.500)   0.500 (0.500) |   0.521      0.375      0.375
  0.30  0.533   0.535 (0.533)   0.500 (0.500) |   0.543      0.373      0.374
  0.50  0.667   0.669 (0.667)   0.500 (0.500) |   0.583      0.333      0.335
  0.90  0.933   0.934 (0.933)   0.498 (0.500) |   0.423      0.093      0.093
```

`cross` 列が予測 $\lambda'$ と一致(§3.2 の主結果)、`2seed` 列が $\lambda$ に依らず 0.5、
そして **`corrected` 列だけが `empirical` と一致し `doc-var` 列は一致しない**(§3.4 の修正)。

### 4.3 S1 が S2 を支配する条件 — 交叉点は $\lambda = 0.25$

| 方法 | MSE 比 | バイアス危険 | コスト |
|---|---|---|---|
| S2(他タスク教師 $K-1$ 本で縮小) | $\lambda' = \frac{1+(K-1)\lambda}{K}$ | **あり**(相関した誤りならバイアス注入) | ゼロ(教師は既にある) |
| **S1(同タスク第 2 シード平均)** | **0.5**($\lambda$ に依らず) | **なし** | 教師 1 本の学習 run |

$\lambda' = 0.5$ すなわち $K=3$ で $\lambda = 0.25$ が交叉点。
**$\lambda > 0.25$ では S1 が勝つ。** S2 が勝つのは $\sigma_\varepsilon^2 > 3\sigma_s^2$ のときだけで、
それが本当なら教師の作り方そのものについての重大な発見になる。

### 4.4 S2 の重み(ペア成分つき Bayes 線形解)

$\sigma_u^2 = \sigma_\varepsilon^2 = 1$、flat prior on $g$(再現は §10.2):

| $\sigma_p^2$ | $w_A$ | $w_W$ | $w_S$ | |
|---|---|---|---|---|
| 0 | 0.667 | 0.167 | 0.167 | **従来の一様 $\lambda'=2/3$ に厳密一致**(検算) |
| 1 | 0.687 | 0.187 | 0.125 | ペア成分で非対称化 |
| 3 | 0.708 | 0.208 | 0.083 | 似ているほど search が削られる |

内容位置($g\approx0$)に限り、和を 1 に正規化して:

| $\sigma_p^2$ | $\sigma_\varepsilon^2$ | $w_A$ | $w_W$ | $w_S$ |
|---|---|---|---|---|
| 1 | 0.25 | 0.933 | 0.067 | **0.000** |
| 1 | 1.0 | 0.833 | 0.167 | **0.000** |
| 3 | 1.0 | 0.786 | 0.214 | **0.000** |
| 1 | **0** | **1.000** | **0.000** | 0.000 |
| 0.3 | 1.0 | 0.906 | 0.094 | **0.000** |

$\sigma_p^2=0$ の行が理論 §3.3 の一様 $\lambda'$ に厳密一致することが、**本提案が既存導出の上位互換**である検算になっている。

---

## 5. S3(外挿)を純 OPD に限る理由

### 5.1 純 OPD には余地がある

* 命題 1 により到達点は $\pi_d$ ちょうど。**目標を動かす以外に上がる道が原理的に存在しない。**
* 生徒は既にそこに着いている: `teacher_kl` $=\mathrm{KL}(p_s\Vert\tilde p)$ が **0.571 → 0.005 nats**(150 step)、
  $\text{on\_travel} = 1-0.005/0.571 \approx 0.99$。handover §3.2 が「**残り伸びしろがほとんど無い状態に入っている**」と明記。

### 5.2 GRPO+OPD には無い

固定点 $\pi^\star\propto\pi_d\exp(\tilde Q/\tau)$ を base 相対で書くと $h^\star = h_d + \tilde Q/\tau$ — **外挿と同じ形**である。
つまり **GRPO は既に報酬に基づいた外挿をやっており**、$h_d$ 方向への盲目的な倍化より根拠のある方向で進んでいる。
実測でも control@300 の `off_travel` が 6 ペアすべてで **1.07–2.11**。
ただしこの量は off-task 教師からの距離軸であって base→on-task 教師軸ではないので、
**示唆であって証明ではない**(コード側の但し書き `sign_weights.py:1367` も「`on_travel` と一緒に読め」と言っている)。

### 5.3 タスク — 前提が成立するのは 3 つ中 2 つ

@150 の内訳(handover §3.2):

| task | `teacher_kl` | `target_kl_ratio` | 前提(教師に着いたか) |
|---|---|---|---|
| alfworld | 0.004–0.006 | 35.4% | **成立** |
| webshop | 0.004–0.006 | 22.3% | 成立(ただし klw 系で一貫して悪化した実績) |
| search | **0.025**(他の 4–6 倍) | 3.5% | **不成立** — まだ着いていない |

**search は S3 から外す。**

### 5.4 $\lambda$ は同定できない — ここだけハイパラが残る

$\lambda'$/$w$ が「$\pi_d$ は雑音込みの推定」という診断に対応するのに対し、$\lambda>1$ は
「$\pi_d$ は向きは正しいが踏み込みが足りない」という**別の診断**で、これは推定問題の外にある。
ExOPD の読み(「$h_d$ は教師の RL が誘導した暗黙報酬の方向」)を採っても、
どこまで踏み込むかは測定から出ない。**$\lambda \in \{1, 1.25\}$ の 2 水準を事前登録する**(ExOPD の値を借りる)。

レバーの大きさは足りている: 局所的に $\mathrm{KL}\approx\frac12\chi^2$ なので $\sqrt{0.252}\approx0.50$ —
**生徒の残距離は現行の書き換え量の約 2 倍**(alfworld は $\sqrt{0.354}\approx0.60$)。
handover §10.3 の「**150→300 の区間こそが target の固定点移動が目的関数を支配する区間**」がここから出ており、
**その区間はまだ走っていない。**

---

## 6. S2 の前提は成立するか — 走行ゼロで決着する

**これが S2 全体の生死を決める。**

| 測定 | A–W | A–S | 出典 |
|---|---|---|---|
| **`agree_rate` 行列**(訓練時ログ) | **0.761 / 0.679** | 0.392 / 0.485 | `multitask_signweight_150step_handover.md:379-389`、`sign_weights.py:696` |
| **符号一致 $\phi$**(監査、base 統制後、$\min\lvert\delta\rvert>0.3$ RMS、n=752) | W→A **+0.023** / A→W **−0.045** | S→A **+0.148** / A→S **+0.140** | 監査 §4.1 |

監査 §10-4 が未解決として記録している:

> $\phi$ が最も高いのは search→alfworld(+0.148)と alfworld→search(+0.140)で、
> 旧アームの転移行列(webshop↔alfworld 0.761/0.679、search↔他 0.35–0.49)と一致しない。
> **2つの推定量のどちらが何を測っているかは未解決。**

### 6.1 解決の仮説(本文書の主張、未検証)

**`agree_rate` は周辺分布で中心化されていない。** 定義(handover §12.5、`sign_weights.py:542-543`)は
「両教師がデッドゾーン外だった候補のうち符号が一致した割合」で、独立期待値を引いていない。
監査 §4.1 が当のものについて警告している:

> $P(\text{on}\uparrow)$ が 0.24〜0.89 と偏っているので、**中心化しない一致率は主に周辺傾向を測る。**

$\phi$ 表の周辺分布では webshop 教師が $P(\text{src}\uparrow) = 0.896$、つまり **9 割の候補で上げ向き**である。
そのような教師は何と比べても高い生の一致率を出す。

**補強**: handover の著者自身が「`webshop__on__alfworld` は 0.732 → 0.787 に上がる。**生徒が動くと行列も動く**ので、
この行列を『タスク間類似度』として引用するときは step を明記すること」と書いている。
凍結モデル同士の類似度が生徒とともに動くなら、それは支持集合経由の効果であって純粋な教師間類似度ではない。

### 6.2 判定は二択で、どちらでも次が決まる

* **chance が 0.75 前後** → 0.761 は関連ほぼゼロ。**出力空間に $\sigma_p^2$ は無く、S2 の非対称化は根拠を失う**(S1 と S3 は残る)。監査 §10-4 も同時に片付く。
* **chance が 0.5 前後** → 0.761 は本物の超過。**$w_S=0$ / $w_W>0$ に測定上の根拠が立つ。**

**現時点でどちらかは言えない。** 監査の周辺分布は別 run・別支持・別フィルタなので `agree_rate` の母集団に転用できない。

### 6.3 パラメータ空間の類似性は根拠にならない

損失が作用するのは出力空間の、生徒が訪れた位置の、生徒 top-20 候補の $h$ 値である。
パラメータ空間の類似方向が訪問状態の候補に発現しなければ $\sigma_p^2 = 0$ である(理論 §3.5 の測度の議論と同じ論点)。
**M1 は必ず出力空間で、損失と同じ支持・同じ位置で測ること。**

---

## 7. 事前登録

### 7.1 配線の検証(結果ではない)

走行開始時に確認し、外れたら止める:

| # | 量 | 期待 | 意味 |
|---|---|---|---|
| V1 | $(w,\lambda) = (e_d, 1)$ での目標 | control と**bit 一致** | 族が control を含むこと |
| V2 | $\sum_m w_m$ | 1.000 ± 1e-6 | base 方向への意図しない縮小が無いこと |
| V3 | S1 の $\mathrm{KL}(\pi_d^{(1)}\Vert\pi_d^{(2)})$ | > 0 | 2 シードが実際に違うこと(同一なら $\sigma_\varepsilon^2$ が測れない) |
| V4 | `_EXPONENT_CLAMP` 発火率(S3) | 報告する | $\lambda>1$ で裾が clamp に当たる率。1 RMS = 2.148 nats、$\lambda=1.25$ の典型変位 0.54 nats に対し clamp 5.0 |
| V5 | `target_tv` / `abs_dkl_mean` | 報告する | 書き換え量が現行アーム(TV 0.72%)とどう違うか |

### 7.2 結果の事前予測

**検出力の基準**(理論 §6.3、arm@300 の 3 回の実測): alfworld SD **1.39pp**、webshop acc 0.46pp、
webshop score 1.42pp、search 0.06pp。差の 2 SE は alfworld **3.9pp**、webshop acc 1.3pp、webshop score 4.0pp、search 0.6pp。
($n=3$ の SD で、$\sigma$ の 95% CI は $[0.52\hat\sigma, 6.3\hat\sigma]$。上限側でも変わらない主張だけを述べる。)

| 段 | 予測 | 根拠 |
|---|---|---|
| **S1**(純 OPD) | alfworld @300 で control 以上。**差は 2 SE に届かない可能性が高い**(3.9pp) | MSE は確実に半減するが、pp への写像が無い(§3.3) |
| **S1**(GRPO+OPD) | S1(純 OPD)より小さい | §3.5 の上界性 |
| **S2** | **alfworld ≈ 0、webshop ≤ 0**(理論 §5.2 の事前予測を継承) | $g$ ≈ 書式なら書式位置で縮小は無効、内容位置では他タスク固有成分の混入 |
| **S3**($\lambda=1.25$、純 OPD) | alfworld / webshop @300 で control 以上。**唯一、教師を超えうる段** | 命題 1 の天井を外に出る唯一の操作。ExOPD の +2.0pt が唯一の先例 |

**主要な事前予測は S2 の null である。** それを承知で走らせる設計になっている(理由は §7.4)。

### 7.3 反証条件

| 段 | 反証条件 | 意味 |
|---|---|---|
| S1 | いずれかのタスクが control より **2 SE 以上下** | バイアスゼロのはずの操作が害を出した ⇒ モデルか配線が誤り。**S2・S3 の前提も崩れる** |
| S2 | 同上、または `agree` 系の診断が M1 と食い違う | 混入が平均 0 雑音ではなくバイアスだった(§9) |
| S3 | webshop が 2 SE 以上下、または clamp 発火率 > 5% | 外挿が裾を壊している。$\lambda$ を下げるか撤回 |
| 全体 | 純 OPD で正だが GRPO+OPD で 0 | §3.5 の上界性が実測で確認された。**主張を純 OPD に限定して書く** |

### 7.4 S2 の存在意義は null にある

理論 §5.2:

> これは cross-teacher 目標の**最も弱い仮定で最も強い形**であり、この形で null なら、
> 符号ゲート付きの変種を走らせる理由は無くなる。

S2 は勝ちに行くアームではなく、**cross-teacher 系全体の停止規則**である。
符号ゲート・全会一致・min・deadzone を全部剥がした Bayes 最適な素の形がここにあり、
これが効かないならそれより仮定の強い変種(リポジトリのアームのほとんど)を走らせる理由が消える。
逆に正が出れば「この教師集合に書式以外の共有すべきタスク知識がある」ことの、これまでで最も直接的な証拠になる。

---

## 8. 実装

### 8.1 M1(§6 の判定)— カウンタ 3 つ

`verl/trainer/ppo/sign_weights.py`。既存の `pair_both` / `pair_same`(`:444-445`、`:542-543`、`:696`)と
**同じ分母 `both` の上で**:

```python
# :542-543 の隣に
self._add(self.pair_on_pos,   (t, m_task), int(((on[r] > 0) & both).sum()))
self._add(self.pair_off_pos,  (t, m_task), int(((so[r] > 0) & both).sum()))
self._add(self.pair_both_pos, (t, m_task), int(((on[r] > 0) & (so[r] > 0) & both).sum()))
```

`:696` の隣で、$N$ = `both`、$n_{1\cdot}$ = `pair_on_pos`、$n_{\cdot1}$ = `pair_off_pos`、$n_{11}$ = `pair_both_pos` として:

$$q_{on}=\frac{n_{1\cdot}}{N},\quad q_{off}=\frac{n_{\cdot1}}{N},\qquad \text{chance}=q_{on}q_{off}+(1-q_{on})(1-q_{off})$$

$$\phi=\frac{N\,n_{11}-n_{1\cdot}n_{\cdot1}}{\sqrt{n_{1\cdot}(N-n_{1\cdot})\,n_{\cdot1}(N-n_{\cdot1})}},\qquad
\text{excess}=\frac{\text{observed}-\text{chance}}{1-\text{chance}}$$

を `agree_chance` / `agree_phi` / `agree_excess` として `agree_rate` の隣に出す。

**注意 2 点。** (a) **rank 集約**: `agree_rate` は rank-0 の量である旨がコードにある(`:1954` 付近)。
比の平均は比ではないので**カウンタを出して比は後で作る**(既存の `pair_both` がそうしている理由と同じ)。
(b) **母集団**: この $\phi$ は生徒 top-20 / $\epsilon=0.1$ nats デッドゾーンの上の量で、
監査 §4.1 の $\phi$($\lvert\delta\rvert>0.3$ RMS、n=752)とは母集団が違う。**直接比較しないこと。**

オフラインで回すなら `scripts/cross_teacher_offline_audit.py` が既存ダンプを読む経路を持つ。
ただし監査 §2 の通り**本物の control のダンプは別ホスト**にあり、tamago 上のものは treatment run(RUN_TAG=xt1)である。
**どのダンプを読んだかを必ず記録すること** — 監査の初版はこれを取り違えている。

### 8.2 目標族 — `build_target` の 1 スカラーと 1 ベクトル

理論 §5.2 の指摘通り、既存配管は $\log\tilde p = \log p_{on} + (c - \log Z)$ を作っているので、
$c$ を差し替えるだけで済む:

$$c = \sigma_d\Big[\lambda\sum_m w_m\hat h_m - \hat h_d\Big]$$

$(w,\lambda)=(e_d,1)$ で $c \equiv 0$、すなわち control と bit 一致(V1)。
符号ゲート・幾何平均・clamp は目標の構成からは不要になる(clamp は S3 の rate limit としてのみ残す)。

### 8.3 S2 の非対称化 — `pair_source` が既にある

`nested_layers`(`cross_teacher_target.py:431`)は既に `pair_source` を返している:

> ``pair_source`` is the index of the teacher that SET the pair layer -- the loudest agreeing one --
> or -1 where the layer is zero. … the search teacher was trained at a 10x smaller KL coefficient,
> and the target design's D4 asked whether it carries any signal at all.

**「どの教師が pair 層を立てたか」は既に計算され、診断として出力されているだけである。**
S2 の非対称化は $\rho_{pair}$ を `pair_source` ごとに分けることで書ける
(curriculum モードの式そのままで、$\rho$ を時間 ramp ではなく静的な縮小係数として使う)。

### 8.4 S1 — 教師キャッシュはタスク鍵なので、そのままでは第 2 シードが載らない

`teacher_cache.py` は hidden state と lse をキャッシュして gather で復元する(`teacher_logprobs_from_hidden`)。
**ただしキャッシュは task 名で引かれている** — `self._task: Dict[int, str]`、`self._weights[task]`、
`register_lm_head(task, weight, slot, n_tasks)`、そして `(n_tasks * vocab, hidden)` の stacked projection。
したがって**同じ task の第 2 シードは、素直に足すと 1 本目と鍵が衝突する。**

選択肢は 2 つで、走らせる前にどちらかに決めること:

* **(a) 別の task 鍵を与える**(例 `alfworld@seed2`)。`n_tasks` が 1 増え、stacked projection も 1 スロット伸びる。
  そのうえで **routing に「この 2 本は同じタスクの教師である」と教える**必要がある —
  行のタスク → on-task 平面の対応(`off_plane_tasks`)が 1 対 1 を前提にしているため。
* **(b) キャッシュの鍵を (task, seed) にする。** 影響範囲は広いが、3 タスク × 2 シードに拡張するなら結局こちらになる。

D2 で alfworld のみと決めるなら **(a) が安い**。いずれにせよ `n_off = 2` の想定はコード側では
ハードコードされていない(`cross_teacher_target.py:339` に "so this generalises past n_off = 2" とある)ので、
平面数そのものは制約ではない。

$\hat h$ の平均は標準化後に取り、宛先の単位に戻す(§2.1)。**本節は読みだけで書いており、実装は試していない。**

---

## 9. 限界(正直な見通し)

* **本提案に新しい測定は無い。** §3・§4 は解析とモンテカルロ、§5・§6 は既存文書からの再読である。
  **M1 と M2 はどちらも未測定**で、S2 の係数も S1 と S2 の優劣も、その 2 つに条件付きである。
* **§6.1 の解決仮説(中心化の欠如)は仮説である。** 監査 §4.1 の警告と handover の step 依存性という
  2 つの状況証拠に基づくが、`agree_rate` の母集団での chance baseline は計算されていない。
  **反対の結果(chance ≈ 0.5)も十分ありうる。**
* **§2.4 のモデル(g / p / u / ε の 4 層、ガウス、$u$ と $\varepsilon$ が iid)は仮定である。**
  off-task 教師が task d の内容位置で**相関した誤り**を持つ場合、混入は平均 0 雑音ではなく**バイアス**で、
  MSE は上がりうる。klw 系が webshop を一貫して悪化させた実績(理論 §4.6)がその方向の状況証拠であり、
  §7.2 の「webshop ≤ 0」はここから来ている。
* **効果量は文献的にも理論的にも小さいと予測される。** §3.3 の写像の欠如に加え、
  $\lambda$ が大きい(教師が正確)なら削減率は $\lambda\to1$ で 0 に収束する。
  **2 シード・中間 checkpoint・AUC / steps-to-threshold で読む設計が要る**(理論 §6.3、関連研究 R4)。
* **Open-MOPD の対立仮説を潰していない。** 多教師 OPD の失敗が干渉でも推定誤差でもなく
  **token 別最適化予算の誤配分**(長さ差・収束 drift・報酬 staleness)なら、本提案の効果はそれに埋もれる。
  リポジトリは `normalize_loss_by_task` で部分対処しているが**未計測**(関連研究 R10)。
  Open-MOPD は headroom 回収を 35.6% → 83.4% にしており、本提案が期待する効果量より 1 桁大きい。
* **Wu+ 2026 は Merge / Mix-RL / MOPD がドメイン平均 1.4pt しか違わないと報告している。**
  「教師の混ぜ方」という軸自体が薄いという外部証拠であり、本提案はその軸の上にある。
  S3(外挿)だけがこの制約の外にある。
* **`_EXPONENT_CLAMP` が $\lambda>1$ でどれだけ発火するかは未測定**(§5.4 の見積りは中央値のみ、V4 で測る)。
* **`agree_rate` の rank 集約の扱いを確認していない**(§8.1 の注意 a)。

---

## 10. 再現用の計算

標準ライブラリのみ、numpy 不要。

### 10.1 §4.2 の表($\lambda'$、S1、§3.4 の修正)

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
        shrunk = hbar + lam*(h[0]-hbar)                          # S2: theory 3.3 posterior mean
        e2 = random.gauss(0,sq); seed2 = g + s[0] + (e[0]+e2)/2  # S1: 2nd seed, same teacher
        a += (h[0]-star)**2; b += (shrunk-star)**2; c += (seed2-star)**2
    print(f" {lam:5.2f}  {lamp:5.3f}  {b/a:6.3f} ({lamp:.3f})  {c/a:6.3f} (0.500) |"
          f" {T/K+lam*se:7.3f}  {(1-lam)**2*T/K+lam*se:9.3f}  {b/N:9.3f}")
```

### 10.2 §4.4 の重み(ペア成分つき Bayes 線形解)

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

print(solve(1e7, 0.0, 1.0, 1.0))   # (0.667, 0.167, 0.167) == uniform lambda' = 2/3  <- 検算
print(solve(1e7, 1.0, 1.0, 1.0))   # (0.687, 0.187, 0.125)
print(solve(0.0,  1.0, 1.0, 1.0))  # (0.625, 0.125, 0.000) -> renormalised (0.833, 0.167, 0)
print(solve(0.0,  1.0, 1.0, 0.0))  # (1.000, 0.000, 0.000)  <- noiseless teacher: w_W = 0
```

---

## 11. 決めること(未確定)

| # | 決定事項 | 選択肢 | 推奨 |
|---|---|---|---|
| D1 | M1 を訓練時指標で取るかオフラインで取るか | 次回 run に載せる / 既存ダンプ | **両方**(母集団が違うので突き合わせる価値がある) |
| D2 | S1 の第 2 シードをどのタスクで作るか | alfworld のみ / 3 タスク | **alfworld のみ**(理論 §6.1 E4 と同じ。3 本は費用が 3 倍で、$\lambda$ のタスク差は二次) |
| D3 | S1 を純 OPD と GRPO+OPD のどちらで走らせるか | 片方 / 両方 | **純 OPD 先**(§3.5 の上界性を実測で確かめる意味がある) |
| D4 | S3 の $\lambda$ | 1.25 固定 / 探索 | **1.25 固定**(ExOPD から借りる。探索するなら別 run として分ける) |
| D5 | S2 を走らせるか | M1・M2 次第 | §2.2 のゲート通り。**M1 が null なら走らせない** |
| D6 | 理論文書 §3.4 / §3.3 の編集を先に入れるか | 入れる / 本提案の中に留める | **入れる**(§3.4 の式は本提案の比較の前提) |
