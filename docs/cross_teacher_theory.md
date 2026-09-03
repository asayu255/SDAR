# 教師間一致による重み付けの理論解析と、複数教師蒸留の整合的な設計

状態: **解析と提案のみ。実装なし。** 走らせる前に §6 の事前登録を固定すること。

対象: `verl/trainer/ppo/cross_teacher_kl_weight.py`（klw、位置スカラー重み）、
`verl/trainer/ppo/sign_weights.py`（signweight の position / target）、
`verl/trainer/ppo/cross_teacher_target.py`（target mode）、および
`claude/cross-teacher-target` 上の `docs/cross_teacher_pg_weight_design.md`（67b1249、PG 項への重み）。
先行測定: [cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)（監査）、
[cross_teacher_role_mask_arms.md](cross_teacher_role_mask_arms.md)、
[cross_teacher_target_design.md](cross_teacher_target_design.md)。

問い: **教師間の一致（corroboration）で蒸留を重み付けることは、OPD 項・PG 項のどちらに掛けるかに依らず、
理論的に整合した機構か。** 整合しないなら、証明が指す形に機構を切り替える。

---

## 0. 結論（先に）

1. **一致は「共有成分の大きさ」を測る統計量であり、「on-task 教師の信頼性」を測る統計量ではない。**
   教師 3 本と base の 4 モデルだけからは、タスク固有の知識と on-task 教師の推定誤差を分離できない（§3.2、識別不能性）。
   一致で KL 項を重み付けることは「全教師が共有する成分を優先して学ぶ」効果しか持たず、
   「一致した位置では教師を信じてよい」という読みは導けない。
2. **位置スカラー重みは OPD 単独では固定点を動かさない**（既知、監査 P1）。**OPD+PG では、位置ごとの固定点も
   各 step の更新方向も、比 $\tau(x) = \beta\,W^{kl}(x)/W^{pg}(x)$ だけで決まる**（§2.2）。
   したがって 67b1249 の 3 アームは、`both` = control と同じ信頼プロファイル（違いは位置別の学習率のみ）、
   `pg` 単独 = `kl` 単独の**逆**の信頼プロファイル、である。「飽和しない項に同じ証拠を届ける」という設計書の読みは、
   固定点の観点では成立しない。
3. **階層モデル下で Bayes 最適な使い方は、目標を一様に縮小した幾何混合**
   $\lambda' \log\pi_d + (1-\lambda')\,\mathrm{mean}_{m\ne d}\log\pi_m$ **であり、位置依存の重みではない**（§3.3）。
   共有成分を「差し引く」contrastive 形は同じモデルで誤りと出る。監査がその形を検定して悪化側だったこと（§4.6 末尾）と整合する。
4. **信頼を位置依存にする正当な根拠は教師側に無い。** 教師と独立な信号（報酬、または同タスクの第 2 シード）が要る。
   理論的に整合し、かつ実行可能なのは (a) 時間で減衰する信頼 $\beta(t)$、(b) reverse KL の unlearning 裾を切るゲート、
   (c) 目標の一様縮小 $\lambda'$ の 3 つ（§5）。**PG 項を教師統計で重み付けることは、教師と独立な唯一の信号に
   教師を混ぜることであり、推奨しない**（§5.4）。
5. **経験的記録の照合**（§4）: 67b1249 §0.1 の「signweight target（1.25/0.75、target mode、OPD+GRPO）」という
   アームの同定は**正しい**。この文書の初版はそれを content マスク klw アームと読み違え、「係数が 100 倍違う」とも書いたが、
   **両方とも撤回する**（§4.0）。検証ログの `experiment_name` は流用したスクリプトの名で、`resume_from_path` は 8/13 に起動した
   signweight target GRPO run の checkpoint を指していた。そのアームも、その対照（8/5 の GRPO run）も、klw 系列も、
   すべて `teacher_kl_loss_coef = 0.01` である。**残る訂正は 3 つ。** (i) 67b1249 の control 列は klw 系列の control
   （別 run）で、このアーム自身の control@300 は alfworld **0.754** — アームの @300（0.754 / 0.778）と一致し、
   「control が追い付いた」が観測になる（§4.2）。学習中の T=1.0 プロファイルもそれを支持する: 優位は step 31–60 で立ち、
   200 step 以降は全指標が一致する（§4.3）。(ii) §0.3 の 2×2 の「重みづけあり」2 セルは重み表も support も異なる（§4.4）。
   (iii) 現在のスクリプトと lock（β=1.0、`4d4c681` 以降の重み表、生徒 top-k）は 8/13 のアームを再現せず、本日再起動された
   run もそうである（§4.5、§5.6）。

---

## 1. 記法と対象

| 記号 | 意味 |
|---|---|
| $k \in \{1..K\}$, $K=3$ | タスク。文脈 $x$ の属するタスクを $d = d(x)$ と書く |
| $\pi_0$ | base（Qwen3-1.7B） |
| $\pi_k$ | タスク $k$ の教師 = $\pi_0$ を単タスク RL で 300 step 調整したもの（`teachers/{k}_step300`） |
| $\pi_\theta$ | 生徒。3 タスク混合で学習 |
| $h_m(v\mid x) = \log\pi_m(v\mid x) - \log\pi_0(v\mid x)$ | 教師 $m$ の policy shift [nats] |
| $\hat h_m = h_m / \sigma_m$ | 自 domain の RMS で標準化した shift（`CumulativePolicyShiftRMS`） |
| $D(x) = \mathrm{KL}(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_d(\cdot\mid x))$ | reverse KL。生徒 top-20 + tail（`topk_kl_per_token`） |
| $\mathrm{pg}(x) = -A\cdot\mathrm{clip}(r)$ | GRPO の per-token 損失。$A$ は群内正規化した軌跡 advantage（全トークンへ broadcast） |
| $\beta$ | `teacher_kl_loss_coef`。klw 系列 0.01、signweight 系列 1.0 |

損失は task 別 token-mean の和 $\;L = \sum_x \big[\mathrm{pg}(x) + \beta\,D(x)\big]$（`normalize_loss_by_task=True`）。

検討対象の 3 機構:

* **(M1) 位置スカラー重み**（klw / signweight position）: $D(x) \to W(x)\,D(x)$。klw では
  $\tilde W = 1 + \sum_v p_\theta(v)\,e(v)$、$e = |c| + q\sum_m \mathrm{relu}(|\hat h_m| - |\hat h_{on}|)$、
  $W = \tilde W/\mu_d$、$\mu_d$ は前 step の KL 加重 task 平均で $\sum W D = \sum D$。
* **(M2) 目標の変更**（signweight target / cross_teacher_target）: $\pi_d \to \tilde\pi_d \propto \pi_d\, e^{c}$。
* **(M3) PG 項への重み**（67b1249 の提案）: $\mathrm{pg}(x) \to W^{pg}(x)\,\mathrm{pg}(x)$、$W^{pg} = \tilde W/\mu^{pg}_d$。

---

## 2. 重みが動かせるもの（構造的結果）

### 2.1 命題 1（OPD 単独。既知）

任意の $W(x) > 0$ について、$\arg\min_\pi \sum_x W(x)\,\mathrm{KL}(\pi(\cdot\mid x)\Vert\pi_d(\cdot\mid x))$ は
各 $x$ で $\pi = \pi_d$。**目標も固定点も動かない。** 監査 P1 / 目標設計 C1 と同じ。
この時点で、OPD 単独の位置スカラー重みが結果を変えうる経路は「有限の学習予算・共有パラメータの下で、
どの位置を先に・強く当てるか」（位置別学習率）だけである。

### 2.2 命題 2（OPD + PG。位置ごとの固定点と更新方向は比で決まる）

位置 $x$ での目的関数を、advantage を条件付き期待値 $\tilde Q(x,a) = \mathbb E[A \mid x, a]$ で書くと

$$
J_x(\pi) = -\,\mathbb E_{a\sim\pi}\big[\tilde Q(x,a)\big] \cdot W^{pg}(x) \;+\; \beta\, W^{kl}(x)\,\mathrm{KL}\big(\pi\,\Vert\,\pi_d\big).
$$

$W^{pg} > 0$ で割ると、$\tau(x) := \beta\,W^{kl}(x)/W^{pg}(x)$ として

$$
\frac{J_x}{W^{pg}} = -\,\mathbb E_\pi[\tilde Q] + \tau(x)\,\mathrm{KL}(\pi\Vert\pi_d),
\qquad
\pi^\star(a\mid x) \;\propto\; \pi_d(a\mid x)\,\exp\!\big(\tilde Q(x,a)/\tau(x)\big).
$$

**証明.** $\sum_a \pi(a) = 1$ の Lagrangian の停留条件 $-\tilde Q(a) + \tau(\log\pi(a) - \log\pi_d(a) + 1) + \nu = 0$ を
解けばよい。PG 項が $-\mathbb E_\pi[\tilde Q]$ の勾配であることは、PPO surrogate の $r=1$ における勾配
$-\mathbb E_{a\sim\pi}[A\,\nabla\log\pi(a)] = -\sum_a A(a)\nabla\pi(a)$ と、
$\sum_a \nabla\pi(a) = 0$ による baseline の消去から従う。$\tilde Q$ は現在の方策の関数なので、これは静的な最適解ではなく
**固定点条件**である。$\square$

**更新方向についても同じことが言える。** 位置 $x$ の logit に対する勾配は
$W^{pg}\big[\nabla\mathrm{pg}(x) + \tau(x)\,\nabla D(x)\big]$ であり、括弧の中（方向）は $\tau(x)$ だけで決まり、
$W^{pg}$ は歩幅だけを変える。固定点に到達していなくても、**各位置・各 step で「教師と報酬のどちらへ寄るか」は
比 $\tau(x)$ が決める。**

**67b1249 の 3 アームへの帰結**（$\tilde W$ は共通、正規化器だけ別）:

| アーム | $\tau(x)$ | 固定点・更新方向 |
|---|---|---|
| `kl`（現行 klw） | $\beta\,\tilde W(x)/\mu^{kl}$ | 一致が大きい位置で**教師を強く**信じる |
| `pg` 単独 | $\beta\,\mu^{pg}/\tilde W(x)$ | 一致が大きい位置で**報酬を強く**信じる — `kl` の**逆** |
| `both` | $\beta\,\mu^{pg}/\mu^{kl}$ = 位置に依らない定数 | **control と同じ**（$\beta$ の task 定数倍を除く）。位置別の歩幅のみ違う |

設計書は `pg` 単独を「飽和しない項に同じ証拠を届ける」実験と位置づけているが、上の表の通り
`pg` は `kl` と**反対向き**の介入であり、`both` は control の信頼プロファイルに一致カリキュラムを重ねたものである。
設計書 §6.2 の予測（「alfworld @300 で control 比伸び続ける」）は、この 2 アームのどちらについても
理論から導かれない。§6 で予測を書き直す。

**共有パラメータの下での注意.** 命題 2 は位置ごとに独立に最適化できる理想化（tabular）で厳密。
実際は 1 つのネットワークが全位置を担うので、$W^{pg}$ が決める歩幅は「どの位置がパラメータ空間で勝つか」を左右する。
つまり `both` は固定点では control と同じだが、有限 step では**一致した位置を優先して当てる**カリキュラムとして働く。
命題 1 の帰結（純 OPD で位置重みが結果を変えない）は、このカリキュラム経路が少なくとも 150 step では効かなかったことを
示唆する（§4.2 参考: signweight position 純 OPD は +4pp だが 1 run の揺れ 2–3pp の内側）。

### 2.3 命題 3（PG 項の状態依存重みは、状態分布の再重み付け）

軌跡水準の advantage $A$ を全トークンに broadcast する GRPO で、トークン $t$ の PG 項に状態のみの関数 $W(s_t) > 0$ を掛けると

$$
\mathbb E\Big[\sum_t W(s_t)\,A\,\nabla\log\pi(a_t\mid s_t)\Big]
= \sum_s d^\pi(s)\,W(s)\sum_a \tilde Q(s,a)\,\nabla\pi(a\mid s).
$$

**証明.** $\mathbb E[A\,\nabla\log\pi(a_t\mid s_t)\mid s_t] = \sum_a \nabla\pi(a\mid s_t)\,\tilde Q(s_t,a)$
（$s_t$ は接頭辞全体なので $t$ 以前の報酬は $s_t$ の関数で、$\sum_a\nabla\pi = 0$ により消える）。
$t$ について足し、$d^\pi(s) = \sum_t P(s_t = s)$ とおく。$\square$

これは訪問分布 $d^\pi(s)W(s)$ の下での policy gradient であり、**$J$ の不偏勾配ではなく別の目的関数 $J_W$ の勾配**である。
設計書の不変量 $\sum W|\mathrm{pg}| = \sum|\mathrm{pg}|$ はスケールの保存であって、不偏性とは無関係。
tabular では $J_W$ と $J$ の最適解は一致するが、共有パラメータの下では $W$ の大きい状態が優先される。
ここに教師統計を入れる理由は、モデルのどこにも出てこない（§5.4）。

---

## 3. 一致は何を測るか（階層モデル）

### 3.1 モデル

候補 $(x, v)$、タスク $d = d(x)$ について、各教師の shift を

$$
h_m = g + s_m + \varepsilon_m, \qquad m = 1..K,
$$

* $g$: **共有成分**。全タスクの RL が同じ方向に書いたもの（ハーネス書式・タグ構文・一人称など）。
* $s_m$: **タスク固有成分**。タスク $m$ の知識。$m$ について iid、分散 $\sigma_s^2$。
* $\varepsilon_m$: 教師 $m$ の**推定誤差**（有限 RL、シード、報酬ハック）。iid、分散 $\sigma_\varepsilon^2$。

タスク $d$ での**目標**は $h^\star_d = g + s_d$。on-task 教師は $h_d = h^\star_d + \varepsilon_d$ でこれを推定している。
教師ごとに KL 係数が違う（search 0.001、他 0.01）ので、実際は $h_m = a_m(\cdot)$ と利得 $a_m$ が掛かる。
repo の RMS 標準化 $\hat h_m = h_m/\sigma_m$ はこの $a_m$ の推定・除去に相当し、以下は標準化後の量で読む。

### 3.2 定理（識別不能性）

$(g, \{s_m\}, \{\varepsilon_m\})$ が互いに独立なら、観測 $(h_1,\dots,h_K)$ の 2 次モーメントが識別するのは
$\mathrm{Var}(g)$ と **和 $\sigma_s^2 + \sigma_\varepsilon^2$** だけである。
実際、偏差 $h_m - \bar h$ の分散は $(1 - 1/K)(\sigma_s^2 + \sigma_\varepsilon^2)$ で、$\sigma_s^2$ と $\sigma_\varepsilon^2$ は
和の形でしか現れない。

**帰結.** on-task 教師が合意から外れているとき、それが「知識 $s_d$」なのか「誤差 $\varepsilon_d$」なのかを、
教師と base の 4 モデルの**どんな関数でも**判別できない。一致統計 $c$（`common_soft`、$|c| \le |\hat h_{on}|$）が測るのは
$g$ の大きさであり、$\varepsilon_d$ とは独立である（$h_m$, $m\ne d$ は $\varepsilon_d$ と独立）。

したがって「一致した位置では on-task 教師が信頼できる」は、モデルのどの読みでも出てこない。
出てくるのは「一致した位置には $g$ がある」だけで、$g$ は on-task 教師が**単独で既に教えている**成分である
（監査 §11.6: 一致しているのは書式で、それは on-task 教師が既に教えている）。
$\sigma_\varepsilon^2$ を識別するには**同じ $s_d$ に対する $\varepsilon$ の 2 回目の抽出**、すなわち
on-task 教師の第 2 シードが要る: $\mathrm{Var}(h_d^{(1)} - h_d^{(2)}) = 2\sigma_\varepsilon^2$。
これが監査の「$(p_{on}, p_{off}, p_{base})$ の三つ組が持つ情報の上限」の形式的な言い換えである。

### 3.3 Bayes 最適な目標は一様縮小（幾何混合）

$g$ に平坦事前分布を置くと、$g \mid h \sim \mathcal N\big(\bar h,\ (\sigma_s^2+\sigma_\varepsilon^2)/K\big)$。
$u_d := h_d - g = s_d + \varepsilon_d$ に対し $\mathbb E[s_d \mid u_d] = \lambda\,u_d$、$\lambda = \sigma_s^2/(\sigma_s^2+\sigma_\varepsilon^2)$。よって

$$
\mathbb E[h^\star_d \mid h] = \bar h + \lambda\,(h_d - \bar h)
= \lambda' h_d + (1-\lambda')\,\bar h_{-d},
\qquad \lambda' = \frac{1 + (K-1)\lambda}{K} \in \Big[\tfrac1K, 1\Big],
$$

$\bar h_{-d}$ は off-task 教師の平均。base は打ち消えて、目標分布は

$$
\log\tilde\pi_d = \lambda'\,\log\pi_d + (1-\lambda')\,\mathrm{mean}_{m\ne d}\log\pi_m + \text{const}
$$

**on-task 教師と off-task 教師の幾何混合（product of experts）**である。この結果が言うこと:

* **位置に依らない一様な $\lambda'$。** 位置依存性は $\sigma_s^2$ か $\sigma_\varepsilon^2$ が位置で変わるとき（異分散）にだけ入り、
  それでも「一致の大きさ」の関数にはならない。
* **符号ゲート・全会一致・min・deadzone はどれも出てこない。** 出てくるのは平均と一様な縮小だけ。
  監査 §4.6 が唯一の正の手掛かりとして挙げた構造（「一致ではなく平均、重みではなく方向、position ではなく target」）は、
  この推定量の構造そのものである。
* **共有成分の除去は誤り。** contrastive 形 $\log p_{on} - \mathrm{mean}_m\log p_m$ は $g$ を目標から引き去るが、
  $g$ は目標 $h^\star_d$ の一部である。監査が同一標本でこの形を検定して差が悪化側だったこと（sg1 −0.042、xt1 −0.035）は
  モデルの予測と一致する。
* **RMS 単位の扱い。** 教師の利得が違うので、混合は標準化した $\hat h$ で取り、宛先の単位に戻す:
  $\bar h_{-d} := \sigma_d\cdot\mathrm{mean}_{m\ne d}\hat h_m$。

**ただし期待利得は測定から小さい。** 監査は base 統制後の水準相関を −0.003〜+0.043、符号一致の $\phi$ を 0.066、
shuffled 比を 0.82 と測っている。$g$ がほぼ書式に限られ、内容位置では $s_m$ 同士が無相関なら、
縮小は書式位置で何も変えず（そこでは $h_d \approx \bar h$）、内容位置では**他タスクの固有成分を混入させる**。
モデルは $\lambda' \to 1$ を選ぶことになり、利得は $\sigma_\varepsilon^2$ が $\sigma_s^2$ と同程度に大きい場合にしか出ない。
これが「理論的に正しい形でも、この教師集合では効かない可能性が高い」という判断の根拠である（§5.2 の事前予測）。

### 3.4 信頼 $\tau(x)$ の最適形は事後精度であり、一致ではない

KL 正則化 RL を近似 Bayes 推論として読むと、位置 $x$ の係数 $\tau(x)$ は事前分布（教師）の**精度**に比例するのが正しい。
縮小後の事後分散 $\mathrm{Var}[h^\star_d \mid h] = (\sigma_s^2+\sigma_\varepsilon^2)/K + \lambda\sigma_\varepsilon^2$ は
等分散の下で**位置に依らない**。異分散なら $\tau(x) \propto 1/\sigma_\varepsilon^2(x)$。

現行の $\tilde W = 1 + \sum_v p_\theta(v)|c(v)|$ をこの尺度で読むと、形が逆向きになる箇所がある:

| 位置の種類 | 現行 $\tilde W$ | 精度からの信頼 |
|---|---|---|
| 誰も動かしていない（証拠ゼロ、位置の 24.5%） | **最小**（$W = 1/\mu < 1$） | **最大**（教師 = base、RL 雑音なし） |
| 全員が同方向に大きく動かした | 大 | $g$ が大きいだけで精度は不変。目標を $\bar h$ に置き換えてよい位置 |
| on-task だけが大きく動かした | ≈ 最小 | $s_d + \varepsilon_d$ が大きい。雑音が更新量に比例するなら**低**（ここは一致する） |

一致が唯一「正しい向き」に出る 3 行目も、根拠は「on-task の更新が大きいほど雑音も大きい」という異分散仮定であって、
一致そのものではない。信頼を位置依存にする正当化があるとしても、それは $|\hat h_{on}|$ の減少関数であり、$|c|$ の増加関数ではない。

### 3.5 測度が生徒質量であることが、機構を entropy 正則化器にする

$\tilde W = 1 + \sum_v p_\theta(v)\,e(v)$ は生徒が $e$ の大きい候補に質量を集めるほど大きい。
一方 reverse KL の logit $j$ に対する勾配は $p_\theta(j)\,[f_j - \mathbb E_\theta f]$、$f = \log p_\theta - \log p_d$ で、
生徒が確信し教師がほぼゼロを与える候補 $v^\star$（$f_{v^\star} \gg 0$）で最大かつ**下げ向き**である。
$\tilde W$ はこの位置で最大になるので、機構は「生徒の確信を剥がす」勾配を選択的に増幅する。
方向を捨てた $|c|$ を使うため、教師が揃って**上げた**候補（`agree_pos`。監査 P4: その $p_{on}$ 中央値は $6\times10^{-7}$）でも、
$p_d(v^\star) \ll p_\theta(v^\star)$ である限り介入の向きは**下げ**である。
つまり作用しているのは reverse KL の unlearning 勾配であって、一致の内容ではない。

これが監査 §15 の帰属（伸びは entropy 上昇 = 探索の副作用、Spearman(伸び, entropy 比) = +1.00、介入量とは逆順）の
機構的な裏付けである。加えて、$p \sim 10^{-7}$ の領域で読む対数比は softmax 裾の推定雑音に支配され、
RMS 標準化はそれを「大きな shift」に見せる。証拠を読む測度は目標の質量（$p_d$、あるいは $\tilde\pi_d$）であるべきで、
生徒の質量は reverse KL の**コスト**の重みであって証拠の重みではない（監査 P9 の理論的な言い換え）。

---

## 4. 経験的記録の再読

### 4.0 同定の手順と、この文書の初版が踏んだ罠

検証ログと per-instance jsonl の `experiment_name` は**流用したスクリプトの名**であり、検証した checkpoint の名ではない。
`~/logs/opd_grpo_klw_content_sg1fast_val{150,300}.log` は名前が content klw アームだが、`resume_from_path` は
`~/checkpoints/verl_agent_opd_grpo_tmp_multitask/global_step_{150,300}`、すなわち **8/13 に起動した signweight target GRPO run**
（wandb `t6vg80ut`、`sdar_multitask_opd_grpo_signweight_target_qwen3_1.7b`）の checkpoint である。content klw アームの学習
（9/02）は Ray のノードメモリで OOM し、checkpoint を 1 つも残していない。この文書の初版はここで誤った。
**同定は `resume_from_path` と wandb の config dump（`wandb/run-*/logs/debug.log`）で行うこと。** 以下はそれで取り直した。

### 4.1 run の対応と、走った設定（config dump より）

| 役割 | run | 起動 | $\beta$ | 機構 | top-k support | GPU |
|---|---|---|---:|---|---|---:|
| **アーム** | `t6vg80ut`（+resume `rnmmiwc7`, `i1nbo7wz`）→ `tmp_multitask` | 8/13 | **0.01** | target、`agree_weight=1.25`、**`disagree_weight=0.75`**、deadzone 0.1 | 教師 | 2 |
| **その control** | `ktrcnege` → `verl_agent_opd_grpo_multitask` | 8/5 | **0.01** | なし | 教師 | 2 |
| klw control（67b1249 が引いた control） | `91v55ri7`（xt1） | 8/29 | 0.01 | なし | 生徒 | 3 |
| klw sg1（新機構） | `n9zfny6m` | 8/31 | 0.01 | klw | 生徒 | 2 |
| klw xt1（旧機構） | — | 8/28 | 0.01 | klw（旧） | 生徒 | — |
| 純 OPD signweight target | `owv67d3y` / `h2ihajq2` | 8/20–22 | 1.0（pg=0、無関係） | target、`agree_neg_weight=0.75`、`disagree_weight=1.0` | 生徒 | — |

**8/13 のアームが実行した重み表**は起動時のコード（`e75051f` の `sign_weights.py`）で決まる。`candidate_weights(agree_weight,
disagree_weight)` に `agree_neg_weight` は**存在せず**、target mode でも `agree → ×1.25`（上げ一致・下げ一致とも）、
`conflict → ×0.75`、他は 1.0。目標は `reweight_teacher_logprobs` による $\tilde p \propto w\,p_{on}$（tail 固定、再正規化）。
**したがって論文の表（負/負 → 増強、対立 → ×0.75）は、論文の数値を出した run に対して正しい。** 8/26 の `4d4c681` が表を
`agree_neg×0.75 / conflict×1.0` に変え、target mode で `disagree_weight ≠ 1` を assert で拒否するようにした。現在のコードとの
食い違いは論文の誤りではなく、**論文の run とその後のコードの違い**である。なお現在の module docstring が「下げ一致を増強すると、
全教師が抑制に同意したトークンの目標確率を上げてしまう」と書く批判は、まさに 8/13 のアームがしたことへの批判であり、
論文のアームが「一致の内容」で効いたという読みには不利な材料になる。

### 4.2 検証値（一致した control で）

control は `~/logs/opd_grpo_val_{150,300}.log`（8/7–8/8 の val_only、`verl_agent_opd_grpo_multitask/global_step_{150,300}`）。
アームは `tmp_multitask` の checkpoint を 9/02–03 に検証したもので、@300 は**同一 checkpoint の 2 回の検証**
（67b1249 §0.1 の値と、per-instance jsonl の再計算値）。

| | control @150 → @300 | アーム @150 → @300 | 差 @150 → @300 |
|---|---|---|---|
| alfworld | 0.651 → **0.754** | **0.762** → 0.754 / 0.778 | **+11.1pp** → **0.0 / +2.4pp** |
| webshop acc | 0.635 → 0.738 | 0.667 → 0.746 / 0.754 | +3.2 → +0.8 / +1.6 |
| webshop score | 0.760 → 0.869 | 0.783 → 0.849 | +2.3 → −2.0 |
| search | 0.384 → 0.395 | 0.382 → 0.396 | −0.2 → +0.1 |

run 間 SD（alfworld 2.1–2.6pp、webshop acc 0.7–3.1pp）で読むと: **alfworld の +11pp は @300 で消え、control が同じ水準に到達している。**
webshop・search はどちらの step でも雑音の内側。**符号反転はどこにも無い。**

67b1249 §0.1 が control 列に置いた 0.651 → 0.738 は klw 系列の control（`91v55ri7`: 別 run、3 GPU、生徒 top-k、8/29 のコード）で、
このアームの対照ではない。@150 が偶然同じ 0.651（82/126）だったため差は @300 にだけ現れ、「アームは飽和し control は 0.738 まで」
という読みになった。正しい対照では**「control は 0.754 まで伸び、アームに追い付く」**である。

参考（同じ手順で取り直した他アーム、per-instance jsonl）: klw sg1@150 0.714 / 0.667 / 0.394（`..._klw_multitask_sg1/global_step_150`）、
klw xt1@300 0.746 / 0.683 / 0.399（`..._klw_multitask_xt1/global_step_300`、`9dfp1kav`）。純 OPD 系列（control / signweight position /
signweight target / teachertopk）@150: alfworld 0.651 / 0.690 / 0.651 / 0.667、@300: 0.675 / 0.714 / 0.690 / —。

### 4.3 学習中の時間プロファイル（T=1.0 rollout、`wandb/*/files/output.log`、10-step 窓平均）

| 窓 | alfworld 成功 CTL | ARM | 差 | 正規形式採用（alfworld）CTL / ARM | entropy CTL / ARM |
|---|---|---|---|---|---|
| 1–30 | 0.10–0.29 | 0.10–0.30 | ≈ 0 | 0 / 0 | ≈ 等 |
| **31–60** | 0.38–0.49 | 0.46–0.58 | **+0.08 〜 +0.09**（3 窓連続） | 0 / 0 | **≈ 等**（0.19–0.20 / 0.17–0.21） |
| 61–100 | 0.59–0.70 | 0.57–0.72 | −0.01 〜 +0.07 | 0 / 0 | 0.16–0.35 / 0.22–0.31 |
| 101–150 | 0.62–0.68 | 0.69–0.79 | +0.06 〜 **+0.11** | CTL が **step 135** で切替 / ARM 0 | 0.48–0.64 / 0.35–0.72（ARM が低い窓が多い） |
| 151–200 | 0.65–0.73 | 0.73–0.78 | +0.03 〜 +0.13 | 0.99 / ARM が **step 162** で切替 | 0.26–0.42 / 0.23–0.49 |
| **201–300** | 0.69–0.83 | 0.72–0.83 | **−0.02 〜 +0.03**（10 窓連続） | 0.99 / 0.99 | ≈ 等 |

各窓 n=10 step。1 step の成功率 SD は ≈ 0.10–0.15 なので窓平均の差の SE ≈ 0.05。単一の窓では決まらないが、31–60 の 3 窓連続の
+0.08〜0.09 と、201–300 の 10 窓連続の ≈ 0 は、それぞれ一貫している。応答長も 291–300 で 197 対 196。

読み:

* **優位は step 31–60 で既に立っている。** そこでは両 run とも正規形式を採用しておらず、entropy も等しい。klw アームで監査が
  伸びの主因とした entropy 上昇（§3.5）は、このアームの早期優位の説明にならない。
* **正規形式の切替は control が 135、アームが 162。** アームは切替を 27 step 遅らせた（klw アームでも同じ方向: role_mask 文書 §1）。
  ただし優位は切替の前後を通じて存在し、切替時期の差は優位の起源ではない。
* **200 step 以降、T=1.0 成功率・形式採用・entropy・応答長・検証値がすべて一致する。到達点は同じで、到達が早かった。**
  命題 2 の言葉では、$\beta = 0.01$ と目標の TV 0.7%（signweight 引継ぎ文書 §1）の下で固定点は control と実質同一で、
  差は動学だけである。この文書の初版が §4.4 に「予測」として書いた「十分な step で control に追い付かれる」は、
  一致した control ではこの run で**観測**になる。
* **早い理由は未測定。** 8/13 の run には token 表・event dump が無い（dump 機構は 8/26 以降）。step ≤ 60 で 1.25/0.75 の
  tilt が何を動かしたかは、dump を付けた再現 run でしか分からない。「到達順序が効いた」は言えるが、「何の順序か」は言えない。

### 4.4 2×2 は一因子の階乗ではない

67b1249 §0.3 の「重みづけあり・純 OPD」= 8/20 の純 OPD signweight target（`agree_neg×0.75, conflict×1.0`、生徒 top-k）、
「重みづけあり・OPD+GRPO」= 8/13 のアーム（`agree×1.25` 両符号、`conflict×0.75`、教師 top-k）。**重み表も support も異なる。**
「重みづけ単独 +0.0、GRPO 単独 +0.0、両方で +11.1」という交互作用の読みは、2 つの「重みづけ」が別物なので成立しない。
「GRPO 単独 +0.0」は 8/5 control@150（0.651）と純 OPD control@150（0.651）が偶然同値だったことに依る。

### 4.5 $\beta = 0.01$ がどの regime か

$A$ は群内標準化されているので $|\tilde Q| \sim 1$。命題 2 の固定点 $\pi_d\exp(\tilde Q/\tau)$ は $\tau = 0.01\,W$ では
$\exp(100\,\tilde Q/W)$ で、**advantage が非ゼロな位置では報酬がほぼ単独で固定点を決める**。教師が効くのは $A = 0$ の位置
（群内の報酬が揃った場合。search の群の約 7 割）と、動学の初期である。「OPD は 150 step で頭打ち、以後の伸びは GRPO」は
弱い事前分布として当然の振る舞いで、重み付けを増やして直すべき欠陥ではない。§4.3 の「到達点は同じ」もこれと整合する。
**係数の選択が一次、位置別の重みは二次**である。

現在の GRPO signweight スクリプトと intent lock は `4d4c681`（8/26）以降 $\beta = 1.0$ を書いており、本日再起動された
`opd_grpo_multitask_signweight_target_qwen3_1.7b_sg1fast` もそれで走っている。$\beta = 1.0$ は教師支配の別 regime で、
8/13 のアームの再現にも、その対照にもならない（§5.6）。

### 4.6 理論が説明するもの・しないもの

* **説明する:** klw の entropy 上昇と裾集中（§3.5）、純 OPD で位置重みが効かないこと（命題 1）、contrastive 除去の悪化（§3.3）、
  webshop の一貫した悪化の方向（klw 系列。内容位置に他タスクの固有成分を混入させる介入）、そして signweight target アームの
  **到達点が control と一致すること**（命題 2、$\beta=0.01$、TV 0.7%）。
* **説明しない:** signweight target アームの step 31–60 の早期優位。entropy でも形式切替の時期でもない（§4.3）。
  「tilt された目標は早期に学びやすい」以上のことは、dump のある再現 run なしには言えない。klw の alfworld 早期優位
  （監査 §15.5）も同じく未説明で、2 つの別機構が同じ形の早期優位を出していることは記録に残す価値がある。
* **予測（未検証）:** 一致した control で 8/13 アームの全 checkpoint（25 step 刻み）を検証すれば、検証値の優位は
  @50–75 で立ち、@200 以降で消える（§6.1 E−1）。

---

## 5. 理論に整合する設計

### 5.1 原則: 目標と信頼を分け、教師統計は目標にだけ使う

複数教師蒸留の設計変数は 2 つしかない。

1. **目標** $\tilde\pi_d(\cdot\mid x)$ — 何に向けて蒸留するか。教師群の統計はここに入れてよい（§3.3）。
2. **信頼** $\tau(x, t)$ — 目標と報酬のどちらをどれだけ信じるか。**教師群の統計では決められない**（§3.2、§3.4）。
   教師と独立な情報（報酬、時間、同タスクの第 2 シード）だけが根拠になる。

現行の全機構（M1〜M3）は、一致統計を信頼に流し込んでいる点で原則 2 に反する。

### 5.2 目標: 一様縮小（幾何混合）

$$
\log\tilde\pi_d(v\mid x) = \log\pi_0(v\mid x) + \sigma_d\Big[\lambda'\,\hat h_d(v\mid x) + (1-\lambda')\,\mathrm{mean}_{m\ne d}\hat h_m(v\mid x)\Big] - \log Z(x)
$$

* support は生徒 top-20 + tail（現行と同じ）。tail は $\hat h$ の tail 版（`tail_on` / `tail_off`）で同じ式に入るので、
  $Z$ は support と tail の和で閉じる。
* $\lambda' = 1$ で control と bit-identical。**唯一のハイパラは $\lambda'$** で、意味は「on-task 教師の合意からの偏差のうち保つ割合」。
* $\lambda'$ の原理的な決め方は第 2 シード（§3.2）。それが無い間は $\lambda' \in \{1, 0.8\}$ の 2 水準。
* 既存配管: `cross_teacher_target.py` の `build_target` は $\log\tilde p = \log p_{on} + (c - \log Z)$ を作っている。
  $c$ を $\sigma_d(1-\lambda')(\mathrm{mean}_m\hat h_m - \hat h_d)$ に置き換えるだけで、符号ゲート・幾何平均・clamp は不要になる。

**事前予測.** 監査の測定（$g$ ≈ 書式、内容位置の $s_m$ 同士は無相関）が正しければ、
alfworld（書式 role のシェア 0.87）では **≈ 0**、webshop（内容 role のシェア 0.55）では **≤ 0**。
正の結果が出れば「この教師集合に共有すべきタスク知識がある」ことの、これまでで最も直接的な証拠になる。
これは cross-teacher 目標の**最も弱い仮定で最も強い形**であり、この形で null なら、符号ゲート付きの変種を走らせる理由は無くなる。

### 5.3 信頼: 教師と独立な信号だけで

**(a) 時間減衰 $\beta(t)$.** 事後分布が報酬データで締まるにつれ事前分布の重みを落とす、という Bayes の標準形。
`teacher_kl_loss_coef` を step の関数にする（例: 0.01 → 0.001 を 300 step で線形）。
cross-teacher 機構を一切持たず、「OPD は前半、GRPO は後半」という観測を目的関数に直接書き込む。

**(b) unlearning 裾のゲート（位置スカラー）.** §3.5 の裾は reverse KL の勾配が最大で、教師の「ほぼゼロ」が
「見ていない」のか「抑制した」のかを区別できない位置である。抽出トークンの gap
$\delta_t = \log\pi_d(a_t\mid x) - \log\pi_\theta(a_t\mid x)$ で $g_t = \sigma(\kappa\,\delta_t)$ を作り、
**位置全体の** $D(x)$ に掛ける（`sdar_utils.compute_sdar_loss` と同じ形。候補ごとの項に掛けると divergence 性が
壊れる — `sign_weights.py` 冒頭の導出）。klw が増幅していた位置を、この形は**ゼロに近づける**（監査 §13.4 が指摘した符号の逆転）。
$\kappa$ は 1 つのハイパラ。教師と生徒の on-task の gap しか読まないので、cross-teacher の識別不能性に触れない。

**(c) 位置依存の信頼を入れるなら $|\hat h_{on}|$ の減少関数**（§3.4 の異分散仮定）で、一致の関数ではない。
ただし (a)(b) より仮定が強いので、第 2 シードで $\sigma_\varepsilon^2(x)$ の位置依存性を測るまでは走らせない。

### 5.4 PG 項には重みを入れない

命題 2 により、PG 側の重みは KL 側の重みと**比でしか効かない**。同じ $\tilde W$ を両方に掛ける `both` は control の信頼に戻り、
PG だけに掛ける `pg` は `kl` の逆を走る。どちらも「証拠を第 2 の項へ届ける」ことにはならない。
命題 3 により、PG の状態依存重みは訪問分布の変更である。報酬は教師と独立な唯一の信号なので、
それを教師統計で再重み付けすることは、原則 2 が要求する独立性を自ら壊す。67b1249 の M3 は**採用しない**。

### 5.5 一致統計の残る使い道

$c$、`shared_share`、shuffled 比は**共有成分 $g$ の診断**として価値がある（どの role に $g$ が載っているか、
教師集合に書式以外の共有があるか）。損失には入れない。

### 5.6 係数と重み表を明示する

比較は同じ $\beta$・同じ重み表・同じ support の中でしか成立しない（§4.1、§4.4）。8/13 のアームを再現するには
$\beta = 0.01$、`agree×1.25`（両符号）、`conflict×0.75`、教師 top-k が要り、**現在のコードは target mode の `disagree_weight ≠ 1`
を assert で拒否する**。本日再起動された run（$\beta = 1.0$、`agree_neg×0.75 / conflict×1.0`、生徒 top-k）は 3 点で異なり、
その対照は `run_multitask_qwen3.sh`（$\beta = 1.0$）であって 8/5 の control でも klw control でもない。再現に価値があるかは
§6.1 E−1 の結果で決めるべきで、それまでは走らせる理由が無い。

---

## 6. 事前登録の骨子

### 6.1 走らせる順（安い・識別力が高い順）

| # | アーム | 何を答えるか | 事前予測 |
|---|---|---|---|
| **E−1** | **既存 checkpoint の全点検証**: 8/5 control と 8/13 アームの `global_step_{25,…,300}`（各 12 点、全て `~/checkpoints/` に存在）を標準プロトコルで検証。**学習なし** | 検証値での優位の時間プロファイル。§4.3 の T=1.0 プロファイルが T=0.4 の検証でも成り立つか | **@50–75 で優位が立ち、@200 以降で消える。** @150 の +11pp が孤立点なら §4.3 の読みは誤り。最も安く、最も識別力が高い |
| E0 | control（$\beta=0.01$）を 300 → 450 step に延長 | control の固定点は 300 で到達しているか | alfworld はさらに伸びる。E−1 で到達点一致が確認されれば優先度は下がる |
| E1 | $\beta(t)$ 減衰（§5.3a）、cross-teacher なし | 「OPD は前半」を目的関数に書くだけで klw 系の早期優位が出るか | @150 で control 以上、@300 で control 以上。entropy は control 同等 |
| E2 | 裾ゲート（§5.3b）、cross-teacher なし | klw の伸びが裾の副作用なら、裾を切る方向で webshop の悪化が消えるか | webshop が control 以上に戻る。entropy は上がらない。alfworld は不明（残差） |
| E3 | 一様縮小 $\lambda' = 0.8$（§5.2） | この教師集合に、書式以外の共有知識があるか | alfworld ≈ 0、webshop ≤ 0（§5.2）。正なら重要 |
| E4 | alfworld 教師の第 2 シード（学習のみ、150 step） | $\sigma_\varepsilon^2$ の直接測定。$\lambda'$ と §5.3c の根拠 | $\sigma_\varepsilon^2/(\sigma_s^2+\sigma_\varepsilon^2)$ を報告。これが小さければ E3 は原理的に効かない |

67b1249 の `both` / `pg` を走らせる場合の**書き直した予測**: `both` は @150・@300 とも control と区別できない
（信頼プロファイルが同じ）。`pg` は一致位置で教師を軽視するので、alfworld @300 で **control 以下**。
どちらかが外れたら命題 2 の tabular 近似（位置別の歩幅が結果を変えない）が破れている、と読む。

**anneal（前半だけ 1.25/0.75、以後 1.0/1.0）について。** 到達点が control と同じなら（§4.3）、anneal は full arm と区別できない結果を出す。目標変更と到達順序を分けるという目的には、到達点が違う場合にしか効かない。E−1 で到達点の一致が確認された時点で、anneal は走らせる理由を失う。順序の**中身**を問うなら、dump を付けた 8/13 設定の再現（seed 2）の方が答えに近い。

### 6.2 配線の検証（結果ではない）

| アーム | 指標 | 期待値 |
|---|---|---|
| E1 | `actor/kl_coef` の軌跡 | 指定したスケジュールと一致 |
| E2 | `sdar/gate_mean`, 裾（$p_\theta > 0.5$, $p_{on} < 10^{-3}$）での gate 平均 | 裾で < 0.1 |
| E3 | $\lambda' = 1$ で control と bit-identical（テスト） | `target/tv` = 0 |
| E3 | `target/tv`, `target/entropy_delta`, role 別の介入シェア | 内容 role に介入が載ることを**確認**する（それが仮説の検定対象） |

### 6.3 検出力

1 run の run 間 SD は alfworld 2.1–2.6pp、webshop acc 0.7–3.1pp（role_mask 文書 §1、67b1249 §6.3）。
**数 pp の差は 1 run では決まらない。** E1〜E3 は同一 checkpoint の検証を 3 回、可能なら学習も 2 シード。
評価は task 別 + 一致 `traj_uid` 上の McNemar（`scripts/val_paired.py`）。pooled は報告しない。

---

## 7. 検証できなかったこと・限界

* **control の per-instance ログは本ホストに無い。** 8/5 control の値は `~/logs/opd_grpo_val_{150,300}.log` の集計値、klw control の値は
  role_mask 文書からの転記で、どちらも `traj_uid` 単位では再計算していない。
* **アーム @300 は同一 checkpoint の 2 回の検証で 0.754 / 0.778**（SD 2.1pp の範囲）。3 回目（temperature 0、`val300_det1.log`）は
  本日実行中で未完。
* **§4.3 の時間プロファイルは n=1 対 n=1 の run** で、窓平均の差の SE ≈ 0.05。31–60 の連続 3 窓と 201–300 の連続 10 窓の一貫性に
  依っており、シード反復は無い。
* **早期優位の機構は未測定**（§4.3、§4.6）。8/13 の run に dump は無い。
* **階層モデルはガウス・等分散・独立。** 定理（§3.2）の識別不能性は 2 次モーメントの話なので分布形に依らないが、縮小推定量の**形**
  （一様 $\lambda'$）は等分散に依る。異分散なら $\lambda'(x)$ になるが、それでも一致の関数にはならない（§3.4）。
* **命題 2 の tabular 近似。** 共有パラメータの下では位置別の歩幅が結果を変えうる。それが 150 step で効いていないことは
  純 OPD の記録が示唆するが、証明ではない。`both` アームがこの近似の直接の検定になる。
* **単タスク参照（67b1249 が引く alfworld 0.849）との差を cross-teacher 情報で埋める根拠は理論からは出ない。**
  理論が指す一次の変数は $\beta$ のスケジュールと裾の扱いであり、8/5 control が 300 step で 0.754 に達していることから、
  step 数そのものも候補である。
* **記録の罠（再発防止）。** 検証ログ・jsonl の `experiment_name` は流用スクリプトの名で checkpoint の名ではない（§4.0）。
  この文書の初版はそれで run を取り違えた。同定は `resume_from_path` と wandb の config dump で行い、`teacher_kl_loss_coef`・
  重み表・`student_indexed_topk` を run ごとに読むこと。スクリプトの現在値から過去の run の設定を推定してはいけない。
