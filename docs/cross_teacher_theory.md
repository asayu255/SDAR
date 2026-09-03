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
5. **経験的記録の訂正**（§4）: 67b1249 §0.1 の「signweight target（1.25/0.75）」アームの @150 値
   （alfworld 0.762、webshop acc 0.667）は、**content マスク付き klw アーム**（`sg1fast`、$\beta=0.01$）の
   per-instance 再計算値と 2 タスクとも一致する。signweight target の OPD+GRPO run は本日 $\beta=1.0$ で
   起動されたばかりで検証値を持たない。signweight 系列（$\beta=1.0$）と klw 系列（$\beta=0.01$）は係数が
   100 倍違う（commit `ed8994f` が明記）。よって §0.3 の 2×2 は機構も係数も混ざっており、
   「重みづけ × GRPO の交互作用 +11.1pp」の根拠にならない。

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
示唆する（§4.1 表: signweight position 純 OPD は +4pp だが 1 run の揺れ 2–3pp の内側）。

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

### 4.1 per-instance ログからの再計算（本ホスト、`~/val_instances/*/val_step*.jsonl`）

`traj_uid` で畳んで max、score > 0 を成功とする（role_mask 文書 §0.1 と同じ手順。klw sg1@150 の 0.7143 / 0.6667 で照合済み）。

| run | 損失 | $\beta$ | step | alfworld | webshop acc | search |
|---|---|---:|---:|---:|---:|---:|
| `opd_multitask`（純 OPD control） | OPD | — | 150 | 0.6508 | 0.6270 | 0.3906 |
| | | | 300 | 0.6746 | 0.6667 | 0.3886 |
| `opd_multitask_signweight_position` | OPD | — | 150 | 0.6905 | 0.6746 | 0.3858 |
| | | | 300 | 0.7143 | 0.6349 | 0.3851 |
| `opd_multitask_signweight_target` | OPD | — | 150 | **0.6508** | 0.6429 | 0.3859 |
| | | | 300 | 0.6905 | 0.6667 | 0.3884 |
| `opd_multitask_signweight_target_teachertopk` | OPD | — | 150 | 0.6667 | 0.6667 | 0.3903 |
| `opd_grpo_..._klw_sg1`（新機構） | OPD+GRPO | 0.01 | 150 | 0.7143 | 0.6667 | 0.3938 |
| `opd_grpo_..._klw_xt1`（旧機構） | OPD+GRPO | 0.01 | 300 | 0.7460 | 0.6825 | 0.3993 |
| `opd_grpo_..._klw_content_sg1fast`（content マスク） | OPD+GRPO | 0.01 | 150 | **0.7619** | **0.6667** | 0.3825 |
| | | | 300 | 0.7778 | 0.7540 | 0.3963 |
| control（`xt1/885xeeru`, `bvl7inr6`。本ホストに無し、role_mask 文書 §0/§5 より） | OPD+GRPO | 0.01 | 150 / 300 | 0.651 / 0.738 | 0.722 / — | 0.390 / — |

純 OPD では係数は Adam のスケール不変性により実質無関係。

### 4.2 67b1249 が並べた数値の出所と、係数の混同

* 67b1249 §0.1 の「signweight target（1.25/0.75、target mode、OPD+GRPO）」の @150 値 alfworld 0.762 / webshop acc 0.667 は、
  上表の **content マスク klw アーム（sg1fast）@150 と 2 タスクとも一致**する（126 問中 96 / 84）。
  signweight target の OPD+GRPO run（`opd_grpo_multitask_signweight_target_qwen3_1.7b_sg1fast`）は
  **本日 2026-09-03 に 4 回起動され、`teacher_kl_loss_coef: 1.0`、検証はまだ無い**（`wandb/run-20260903_*/logs/debug.log`）。
  @300 値（alfworld 0.754 / webshop 0.746）は上表の content@300（0.7778 / 0.7540）と一致しないので、別の出所である。
* commit `ed8994f`（2026-08-27）は「OPD 係数を 1.0 → 0.01 に動かしたので、**それ以前の全アームと 150-step sign-weight 報告は、
  機構より先に係数で違う**」と明記している。signweight 系列（`examples/opd_grpo_trainer/run_multitask_signweight_*.sh`、$\beta=1.0$）と
  klw 系列（$\beta=0.01$）の control は別物で、前者の control は `run_multitask_qwen3.sh`（`expected_multitask_config.yaml`、$\beta=1.0$）である。
* よって 67b1249 §0.3 の 2×2 は、「重みづけあり・純 OPD」= signweight **target**（機構 M2）、「重みづけあり・OPD+GRPO」=
  content マスク **klw**（機構 M1、別の証拠・別の正規化）で、**別機構の 2 セルを 1 つの因子として読んでいる**。
  さらに純 OPD の 2 セルと OPD+GRPO control が 3 つとも 0.6508（82/126）で並ぶのは偶然であり、「+0.0pp」の読みは
  1 run の揺れ（alfworld 2.1–2.6pp）の内側の話である。交互作用の主張は現時点で支持されない。

### 4.3 $\beta = 0.01$ がどの regime か

$A$ は群内標準化されているので $|\tilde Q| \sim 1$。命題 2 の固定点 $\pi_d\exp(\tilde Q/\tau)$ は $\tau = 0.01\,W$ では
$\exp(100\,\tilde Q/W)$ で、**advantage が非ゼロな位置では報酬がほぼ単独で固定点を決める**。
教師が効くのは $A = 0$ の位置（群内の報酬が揃った場合。search の群の約 7 割）と、動学の初期である。
これは「OPD は 150 step で頭打ち、以後の伸びは GRPO」という観測の、弱い事前分布として当然の振る舞いであって、
重み付けを増やして直すべき欠陥ではない。$\beta = 1.0$ の signweight 系列は逆に教師支配の regime にあり、
2 系列の「飽和」を同じ物語で読むことはできない。**係数の選択が一次、位置別の重みは二次**である。

### 4.4 理論が説明するもの・しないもの

* **説明する:** klw の entropy 上昇と裾集中（§3.5）、純 OPD で位置重みが効かないこと（命題 1）、
  contrastive 除去の悪化（§3.3）、webshop の一貫した悪化の方向（内容位置に他タスクの固有成分を混入させる介入。
  webshop は内容 role のシェアが最大）。
* **説明しない:** alfworld の早期優位（監査 §15.5 の残差、entropy 同等の区間で +4.7σ）。
  理論は「一致の内容が運んだ」を否定するが、代わりの機構を特定しない。シード反復なしにこれ以上は言えない。
* **予測（未検証）:** control は 300 step でまだ伸びており（0.651 → 0.738）、klw 系アームの固定点は control と違う（§2.2）。
  一致位置で教師を過信し・非一致位置で教師を軽視する $\tau(x)$ が誤っているなら、**十分な step で control に追い付かれる**。
  content@300 0.778 > control@300 0.738 はこれと矛盾するが、1 run 差 4pp は揺れの 1.5–2σ である。

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

### 5.6 係数を明示する

比較は同じ $\beta$ の中でしか成立しない（§4.2）。本日起動した signweight target（$\beta = 1.0$）には
`run_multitask_qwen3.sh`（`opd_grpo_trainer`、$\beta=1.0$）が対応する control であり、klw control ではない。

---

## 6. 事前登録の骨子

### 6.1 走らせる順（安い・識別力が高い順）

| # | アーム | 何を答えるか | 事前予測 |
|---|---|---|---|
| E0 | control（$\beta=0.01$）を 300 → 450 step に延長 | control の固定点は 300 で到達しているか。§4.4 の「追い付かれる」予測の前提 | alfworld はさらに伸びる |
| E1 | $\beta(t)$ 減衰（§5.3a）、cross-teacher なし | 「OPD は前半」を目的関数に書くだけで klw 系の早期優位が出るか | @150 で control 以上、@300 で control 以上。entropy は control 同等 |
| E2 | 裾ゲート（§5.3b）、cross-teacher なし | klw の伸びが裾の副作用なら、裾を切る方向で webshop の悪化が消えるか | webshop が control 以上に戻る。entropy は上がらない。alfworld は不明（残差） |
| E3 | 一様縮小 $\lambda' = 0.8$（§5.2） | この教師集合に、書式以外の共有知識があるか | alfworld ≈ 0、webshop ≤ 0（§5.2）。正なら重要 |
| E4 | alfworld 教師の第 2 シード（学習のみ、150 step） | $\sigma_\varepsilon^2$ の直接測定。$\lambda'$ と §5.3c の根拠 | $\sigma_\varepsilon^2/(\sigma_s^2+\sigma_\varepsilon^2)$ を報告。これが小さければ E3 は原理的に効かない |

67b1249 の `both` / `pg` を走らせる場合の**書き直した予測**: `both` は @150・@300 とも control と区別できない
（信頼プロファイルが同じ）。`pg` は一致位置で教師を軽視するので、alfworld @300 で **control 以下**。
どちらかが外れたら命題 2 の tabular 近似（位置別の歩幅が結果を変えない）が破れている、と読む。

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

* **control の per-instance ログは本ホストに無い。** §4.1 の control 行は role_mask 文書からの転記で、再計算していない。
* **67b1249 の @300 値（0.754 / 0.746）の出所は特定できなかった。** @150 値は content klw アームと一致するが、@300 は一致しない。
* **階層モデルはガウス・等分散・独立。** 定理（§3.2）の識別不能性は 2 次モーメントの話なので分布形に依らないが、
  縮小推定量の**形**（一様 $\lambda'$）は等分散に依る。異分散なら $\lambda'(x)$ になるが、それでも一致の関数にはならない（§3.4）。
* **命題 2 の tabular 近似。** 共有パラメータの下では位置別の歩幅が結果を変えうる。それが 150 step で効いていないことは
  純 OPD の 2×2 行が示唆するが、証明ではない。`both` アームがこの近似の直接の検定になる。
* **alfworld の早期優位は未説明のまま**（§4.4）。本文書はそれを一致の内容に帰属させる読みを否定するが、代替を特定していない。
* **単タスク参照（67b1249 が引く alfworld 0.849）との差 7–10pp を、cross-teacher 情報で埋める根拠は理論からは出ない。**
  理論が指す一次の変数は $\beta$ のスケジュールと裾の扱いであり、control が 300 step でまだ伸びていることから、
  step 数そのものも候補である。
