# MOPD v3: RL 方向を多タスクで統合し、その結果を目標分布として蒸留する

状態: **提案。** 素の OPD+GRPO を基準に、**学習損失を蒸留 1 本にする。** GRPO 損失は足さない。
RL は目標分布を作る側にだけ入る。

先行: [opd_output_space_cross_gate_design.md](opd_output_space_cross_gate_design.md)（v1/v2 の乗算ゲート）、
[opd_pushback_control_review.md](opd_pushback_control_review.md)（self gate の実測と撤回）、
理論文書 §2.2（Prop. 2）、`cross_teacher_target.py`（目標書き換えの既存機構）、
[RLSD](https://arxiv.org/html/2604.03128v1)、`rlsd_utils.compute_rlsd_token_advantage`。

**なぜ乗算をやめるのか。** `cross_teacher_target.py` の監査結果:

> a positive scalar on `KL(p_student || p_on)` cannot inject anything,
> since the minimiser is `p_student = p_on` whatever the scalar is.

係数（self gate、cross gate v1/v2、係数 QP）は**教師へ引かれる速さ**しか変えられず、行き先は常に $p_{\rm on}$。
「他タスクの報酬に逆らう成分を消す」を係数で実現することは原理的にできない。**目標そのものを書き換える**のが唯一の道。

---

## 0. 機構

$$
L=\beta\,\mathrm{KL}\big(p_\theta\,\Vert\,\mathrm{sg}(q^\star_{i,t})\big),
\qquad
q^\star_{i,t}=\mathrm{softmax}\big(\log q_{i,t}+c_{i,t}\big)
$$

$$
c_{i,t}=\eta\;\alpha_{i,c(t)}\;\underbrace{f_{i,t}\,e_{i,t}}_{\text{OPD 強度 × 教師の支持}}\;r_{i,t}
$$

$q_{i,t}$ は自タスク教師、$r_{i,t}$ は clip 後の PG 降下方向、$\alpha$ はタスク間統合係数。
**GRPO 損失は無い。** 報酬は $r$ を通じて目標生成にだけ入る。

**実際に注入される logit 降下方向は $c$ ではない。**

$$
\boxed{\;-\nabla_z\big[\beta\mathrm{KL}(p_\theta\Vert q^\star)\big]\Big|_{p_0}=d+\beta F_{p_0}c,
\qquad F_p=\mathrm{diag}(p)-pp^\top\;}
$$

成分では $\beta\,p\odot(c-\langle p,c\rangle)$。autograd 一致 5.4e-19。**平均を引く項を落とすと相対誤差 24.4%** になるので、
統合も記録もこの $F_pc$ の方向で行う（§3、§7）。

**$\mathbb E_p[\cdot]$ ではなく $\langle p,\cdot\rangle$ と書く理由（訂正）。** 損失の支持は top-k で、$p$ は
その上の**非正規化**確率（$\sum_S p=m\le1$）である。したがって

$$F_p\mathbf 1=p\,(1-m)\neq 0$$

で、**$F_p$ は厳密には定数を消さない。** 初版の docstring とテストは「消す」と書いていたが、それは $m=1$ の場合だけ成り立つ主張で、
そのテストは $p$ を正規化していたために通っていた。

**ただし、この効果の大きさは pin した支持で決まる（二重訂正）。** このアームは `student_indexed_topk=True` を pin しており、
$S$ は生徒自身の top-k なので実測 tail mass は **0.000–0.002**（`actor/opd_diag/tail_mass_mean`、cross2 run）。
$m=0.999$ では再中心化が注入方向を動かす量は **0.02%** で、実質無視できる。$m=0.5$ なら 23% になるが、
それは**教師 index の支持**が与える値であり lock が禁じている。

したがって $c$ の再中心化は「$F_p$ が消すから無害」でも「大きな補正」でもない。
**clamp を任意のオフセットに依存させないため**に置く操作であり、根拠はそちら一本である。

**$c$ の構成順序（clamp と $\alpha$ の順序は仕様である）。**

$$
c^{\text{base}}_{i,t}=\mathrm{clip}\Big(\eta\,\mathrm{center}\big(f_{i,t}e_{i,t}r_{i,t}\big),\ \pm c_{\max}\Big),
\qquad c_{i,t}=\alpha_{i,c(t)}\;c^{\text{base}}_{i,t}
$$

**$\alpha$ は clamp の後に掛ける。** clamp は斉次でないので $\mathrm{clip}(\alpha x)\neq\alpha\,\mathrm{clip}(x)$ であり、
先に $\alpha$ を掛けて clamp すると、統合の統計が積み上げる $F_pc^{\text{base}}$ と損失が実際に取った方向が
clamp が効く token でずれる。この順序なら注入は $\alpha$ について厳密に線形:
$\;\beta F_pc=\alpha\cdot\beta F_pc^{\text{base}}$（テストで float32 相対 3e-8 まで確認）。

**目標を書き換える役割と、統合係数を解く役割は別である。**

| 集合 | 中身 | 意味 |
|---|---|---|
| `target_roles` | format, reasoning, env_action, tool_call, tag | **目標をそもそも書き換える範囲。** `pg_loss_coef=0` なので、ここから外れた役割は報酬経路が一つも無く純 OPD になる。学生が生成する役割は全て入れる（`env_obs` は生成物ではないので入らない） |
| `roles` | format, env_action | **$\alpha$ を解く範囲。** 共通参照を作れる役割だけ。`target_roles` の部分集合であることを `validate()` が要求する |

$\alpha$ が定義されない役割は $\alpha=1$、すなわち**自タスクの RL をそのまま残す**。0 ではない。
`tool_call` は search にしか無く交差相手が存在しないので、ここを 0 にすると
search の `<search>` / `<answer>` から報酬が消える — search の課題そのものが消える。

---

## 1. RL 信号の重み付け（token ごと）

### 1.1 OPD 強度 $f$

$$
s_{i,t}=\|d_{i,t}\|_2,\qquad
f_{i,t}=\frac{2\,s_{i,t}}{s_{i,t}+\bar s_{i,c}+\delta}\in[0,2)
$$

$\bar s_{i,c}$ は前 step までのタスク・役割別平均（EMA）。平均的な強度で 1、OPD ゼロで 0、上限 2。

### 1.2 教師の支持 $e$（RLSD）

$$
\Delta_t=\log q_t(a_t)-\log p_t(a_t),\qquad
e_{i,t}=\mathrm{clip}\big(e^{\mathrm{sign}(A_t)\Delta_t},\,1-\epsilon_w,\,1+\epsilon_w\big)
$$

失敗軌跡（$A_t<0$）で教師がその token を支持していれば $e_t<1$ となり、**罰を弱める。**
[opd_pushback_control_review.md](opd_pushback_control_review.md) §9.3 で特定した「失敗軌跡内の正しい中間手順まで罰する」危険への直接の対処。
**教師が正しいことを保証するものではない。**

これは `rlsd_utils.compute_rlsd_token_advantage` の `w_t` と同一式。**再実装せず同じ関数を使う。**
違いは注入点だけ —— ローカルは advantage に掛け、ここは logit 空間の $r$ に掛ける。

### 1.3 二段にする理由

強度だけで重み付けると、大きな OPD 信号に含まれる「教師の有用な知識」と「教師と生徒の不一致」を区別できない。
$f$ が前者を、$e$ が後者を担う。**$f\cdot e\in[0,2.4]$ で、RLSD の clip 範囲には収まらない**（強度係数を掛けるため）。

$$
\tilde r_{i,t}=\mathrm{sg}(f_{i,t}e_{i,t})\,r_{i,t}
$$

---

## 2. タスク間統合

### 2.1 何を統合し、何を統合しないか

**異なる状態の token ベクトルを足して全タスク共通の目標分布を作らない。** search の検索語と webshop の商品操作を混ぜた分布に意味は無い。
**タスク間では統合係数 $\alpha$ だけを決め、それを各タスク自身の token ベクトルへ戻す。**

役割 $c$ ごとに、損失の正規化に合わせた平均:

$$
v_{i,c}^{(k)}=\mathbb E_{i,c,k}\big[F_pr_{i,t}\big]\quad(\text{保護対象の元 RL 方向}),\qquad
g_{i,c}^{(k)}=\mathbb E_{i,c,k}\big[F_pc^{\text{base}}_{i,t}\big]\quad(\text{候補方向})
$$

$k\in\{1,2\}$ は**プロンプトを 2 分割した半分**。分割鍵は v1/v2 と同じ（turn-0 anchor をタスクとともに hash）。
$g$ は $\tilde r$ ではなく $c^{\text{base}}$ の平均である — 再中心化と clamp を通した後の、
$\alpha=1$ で損失が実際に注入する方向そのもの。

### 2.2 制約付き最適化

$$
\min_{0\le\alpha_{\cdot,c}\le1}\ \tfrac12\sum_j K_{j,c}(1-\alpha_{j,c})^2
\qquad\text{s.t.}\qquad
\big(\beta F\,v_{i,c}\big)^\top\Big(\sum_j\alpha_{j,c}\,\beta F g_{j,c}\Big)\ \ge\ 0\quad\forall i\ \text{valid}
$$

$K_{j,c}=\beta^2\,\mathbb E_{j,c}\|F_pc^{\text{base}}_{j,t}\|^2$。目的は**重み付け後の RL 信号を削る量**を小さくすること。
削られる量は $(1-\alpha_j)\,\beta F_pc^{\text{base}}_j$ なので、$K$ はその二乗ノルムに一致させる。

**制約は $F$ を通した方向で評価する**（§0）。$v$ と $g$ の生の内積ではない。

**Gram は必ず異なる半分どうしで組む。**

$$
G_{ij,c}=\tfrac12\Big[\big(v_{i,c}^{(1)}\big)^\top g_{j,c}^{(2)}+\big(v_{i,c}^{(2)}\big)^\top g_{j,c}^{(1)}\Big]
$$

問題になるのは対角 $G_{ii}$ で、同じ token を内積の両側に置けば条件は自分自身を確認するだけになる。
非対角は元々プロンプトが交わらないが、規則は一つにして全体へ適用する。

**QP は自分のスケールで正規化して解く。** 実測の $G$ は $10^{-24}$ 以下で、絶対量のまま扱うと
(a) 受け手の行に対する実行可能性判定が全ての点を受理し、(b) $K$ への ridge が $K$ 自体を上書きし、
(c) 目的値の改善判定（絶対 $10^{-15}$）が効かず**列挙が最初に見つけた実行可能点を返す**。
(c) は真の最適が $\alpha=0.5$ の場合に $\alpha=0$ を返す — 蒸留のみでは「RL 半分」と「RL 皆無」の差である。
したがって受け手の行は単位ノルムへ正規化して残差を `solver_tol` と比べ、ridge は $\max|K|$ に対する比、
目的は $\max|K|$ で割る（正の定数倍なので argmin は動かない）。$10^{3}$ から $10^{-30}$ まで同一解を返すことをテストで固定。

性質:

* 全タスクに整合するなら $\alpha=1$ をそのまま採る
* 衝突がある場合だけ、必要な候補成分を減らす
* あるタスクで減らした分を別タスクへ移す制約は置かない
* $\alpha=0$ が常に実行可能

**保証の範囲。** これが保護するのは、同じ役割の全状態に共通の logit bias を加えたと考えたときの、
**局所的な出力空間の代理目的**である。共有パラメータの Adam 更新や 3 タスクの成功率を保証する条件ではない。

### 2.3 統合不能時のフォールバック（v1/v2 から変わる）

**蒸留のみでは $h=0\Rightarrow q^\star=q$ となり、その部分は純 OPD に戻って RL 信号が消える。**
したがって v1/v2 の「共通方向が作れなければ係数をゼロ」は引き継げない。

$$
\text{統合不能・参照不足}\ \Longrightarrow\ \alpha_{i,c}=1\ (\text{自タスクの重み付け RL 方向})
$$

**保護条件を満たしたとは扱わず、フォールバックとして記録する。**
取る行動は同じでも、**run について言っていることが違う**ので理由は分けて記録する:

| code | 意味 |
|---|---|
| `no_receiver` | 有効な受け手が一つも無い |
| `nonfinite` | $K$ か $G$ に非有限値 |
| `no_signal` | 全ての $K_j=0$。目的が実行可能点を順序づけられず、答えは ridge の産物にしかならない |
| `unsolved` | 数値が破綻し、実行可能な KKT 点が一つも得られなかった（**答えは不明**） |
| `no_common_dir` | 解けた上で答えが「全送り手を切れ」だった。**非零の共通方向が存在しない**という所見であり、`unsolved` とは別の事実 |

`no_common_dir` の検出は厳密である: $\alpha=0$ は常に実行可能で、$K>0$ なら他に実行可能点があれば必ず目的が下がるので、
最適解が $0$ であることと「$0$ 以外に実行可能点が無い」ことは同値。

### 2.4 既存の MTL 統合手法を採らない理由

実測の RL Gram（N=8 probe、step 300）:

```
<r_i,r_j> = [[ 17.02   0.14   0.08]      norms 4.13 / 10.00 / 2.50
             [  0.14 100.02   0.26]      off-diagonal cos = 0.007
             [  0.08   0.26   6.22]]     |off| / |diag| = 0.4%
```

| 手法 | この Gram で何が起きるか |
|---|---|
| PCGrad | 発火は 8 batch 中 3〜4 回（コイン投げ）、発火しても射影が除くのは 0.7% |
| MGDA（生） | $\alpha=(0.256,0.042,0.703)$。直交勾配では $\alpha_i\propto1/\|g_i\|^2$ に退化し、**ノルムの逆数**で決まる |
| MGDA（単位化） | $(0.334,0.333,0.332)$ ——一様と区別がつかない |
| CAGrad / Nash-MTL | 衝突が無ければ平均へ帰着 |

**衝突解消系は「勾配が衝突している」を前提とし、その構造がここには無い。**
加えて、これらの保証（Pareto 性・収束）は**パラメータ更新**に対するもので、$\exp$ と再正規化を通した目標分布には引き継げない。
文献側でも [In Defense of the Unitary Scalarization](https://arxiv.org/abs/2201.04122)、
[Uniform Loss vs. Specialized Optimization (2025)](https://arxiv.org/abs/2505.10347v2) が
「専用 MTL 最適化の優位の多くは調整不足由来の見かけ」と報告している。

§2.2 の制約付き統合は、保護条件・除去コスト・介入しない場合を明示できる点で初版に採る。

---

## 3. 有界 log 空間補正を採り、厳密一致は主張しない

$c=\eta h/(\beta p_0)$（$1/p_0$ 形）なら $\sum h=0$ の下で $\beta F c=\eta h$ が厳密に成り立つ。
しかし実測規模で指数は $[-37.5,+21.7]$（$p_0^{\min}=3.6\times10^{-3}$）に達する。
log 空間で計算自体は安定だが、**目標が極端に集中する**。clamp を入れると:

| clamp | clip された座標 | $d+\eta h$ からの偏差 | $\mathrm{TV}(q^\star,q)$ |
|---:|---:|---:|---:|
| $\lvert e\rvert\le5$ | 14/20 | 69.6% | 0.744 |
| $\lvert e\rvert\le2$ | 17/20 | 85.0% | 0.550 |
| $\lvert e\rvert\le1$ | 19/20 | 90.9% | 0.427 |

**恒等式が成り立つ領域と、使える領域が重ならない。** したがって初版は

$$
c_{i,t}=\eta\,\alpha_{i,c}f_{i,t}e_{i,t}\,r_{i,t}\qquad(\text{$1/p_0$ を掛けない})
$$

を採り、注入方向は $\beta F_{p_0}c$ と**認める**。素の OPD+GRPO との厳密一致は主張しない。

**$\eta=1$ を「対照」と呼ばない。** 厳密一致には (a) forward ごとに現在の $p_\theta$ と clip 枝から作り直す、
(b) clamp 無し、(c) 支持集合の近似無し、の 3 条件が要る。(a) は実装で満たすが (b) を捨てるため、
**生成点でも一致しない。** 対照は素の OPD+GRPO アームそのものが担う。

**零和。** $r=c_t(e_a-p)$ の支持上の和は $c_t\times(\text{tail 質量})\ne0$。再中心化しないと $F$ 形からの偏差 14%。
$c$ は支持集合上で再中心化する。

---

## 4. 固定点と PPO clip について、言えることの範囲

**言えない。** $q^\star$ 自体が生徒・rollout・advantage に依存するので、**学習全体の固定点を事前に 1 つの $q^\star$ として指定したことにはならない。**
自己無撞着な式である。理論文書 Prop. 2 と対応させるなら、指数傾斜のスコアは $h$ ではなく **$h/p_0$**（$1/p_0$ 形の場合）。
$\eta$ だけを温度と同一視しない。

**教師目標への KL は信頼領域ではない。** PPO clip は**旧方策**との比を縛り、$\mathrm{KL}(p\Vert q^\star)$ は**目標**への距離。別物。

**ただし clip 情報は失われない。** $r$ は `policy_loss_gradient_coef`（clip 枝込み）から作るので、
clip でゼロになった位置では $r=0\Rightarrow c=0\Rightarrow q^\star=q$。検算済み（追加方向のノルム 1.1e-18）。
**forward ごとに目標を作り直す限り、その step の clip 枝がそのまま目標に反映される。**

---

## 5. 二重計上を隠さない

損失は蒸留 1 本なので、生成点での更新信号は

$$
d+\beta F_{p_0}c
$$

であり、**$r$ が二度入ることはない。** ただし:

* **RL を使わなくなるわけではない。** RL は目標生成側に入る
* **強度比の設計は消えない。** 設計場所が損失係数から目標生成（$\eta$、$f$、$e$）へ移っただけ
* **新しい報酬情報も教師知識も生まれない。** 目標分布に変換しただけ
* 元の OPD 項が残っても、追加 RL が教師の有益な作用を打ち消す可能性は残る

---

## 6. アーム構成

| アーム | $\alpha$ | $f,e$ | 損失 | 問い |
|---|---|---|---|---|
| **A. control** | — | — | $L_{\rm GRPO}+\beta\mathrm{KL}(p\Vert q)$ | 素の OPD+GRPO |
| **B. target, no integration** | $\equiv1$ | 有効 | $\beta\mathrm{KL}(p\Vert q^\star)$ | RL を目標経由で入れると変わるか（**単タスクの問い**） |
| **C. target + integration** | §2.2 の解 | 有効 | $\beta\mathrm{KL}(p\Vert q^\star)$ | **タスク間統合が効くか** |

**B と C の差だけが多タスク機構の効果。** A と B の差は単タスクの問いで、そう明記する。

---

## 7. 記録する指標

**注入されたもの**

| 指標 | 何を答えるか |
|---|---|
| `target/inject_norm/{task}/{role}` | $\|\beta F_{p_0}c\|$。**実際に注入された方向**の大きさ |
| `target/inject_over_d/{task}/{role}` | $\|\beta Fc\|/\|d\|$。OPD 項に対する比 |
| `target/inject_over_r/{task}/{role}` | $\|\beta Fc\|/\|r\|$。素の GRPO 項に対する比 |
| `target/cos_inject_r/{task}/{role}` | $\cos(\beta Fc,\ r)$。注入方向が元の RL 方向をどれだけ保つか |
| `target/tv_qstar_q/{task}/{role}` | $\mathrm{TV}(q^\star,q)$。目標が教師からどれだけ離れたか |
| `target/clamped_frac/{task}/{role}`、`target/c_absmean` | clamp の発火率と補正の大きさ |
| `target/recenter_residual/{task}/{role}` | 再中心化前の $\lvert\sum_v c_v\rvert$。零和条件の破れ |

**重み付けの内訳**

| 指標 | 何を答えるか |
|---|---|
| `target/f_mean`、`target/f_p10`、`target/f_p90`（task/role 別） | OPD 強度係数の分布 |
| `target/e_mean`、`target/e_clip_frac` | RLSD 支持係数と clip 発火率 |
| `target/fe_mean` | 積。1 からの乖離が機構の中身 |
| `target/e_mean_adv_neg`、`target/e_mean_adv_pos` | **失敗軌跡で罰が弱まっているか**（§1.2 の狙い） |

**タスク間統合**

**規約 1: 大きさの指標は $\beta$ 込みで出す。** 更新が見るのは $d+\beta F_pc$ なので介入は $\beta F_pc$ であり、
$\|F_pc\|$ 単体は $\beta=0.01$ で 100 倍ずれる。$\beta$ 抜きの値は別名（`*_nobeta`）でのみ残す。

**規約 2: 比の分子と分母は同じ母集団で取る。** 分子を live token 平均、分母を全 token 平均にすると何の比でもない。
既定は**全損失 token 基準**（書き換えなかった token は分子に 0、分母に自分の $\|d\|$ を出す — 更新のどれだけに触れたかの素直な言明）。
live token だけに限った版は `_live` を付けて補助として併記する。

| 指標 | 何を答えるか |
|---|---|
| `target/alpha/{task}/{role}` | 統合係数。$\equiv1$ なら不活性 |
| `target/alpha_at_bound_frac/{role}` | $\alpha$ が 0 か 1 に張り付いた割合 |
| `target/constraint_slack/{i}/{role}` | $\big(\beta Fv_i\big)^\top\sum_j\alpha_j\beta Fg_j$。**$\alpha=1$ での値も記録**（縛るかどうか） |
| `target/gram_vg/{i}/{j}/{role}` | $G_{ij}$（§2.2、異なる半分どうし）。**$v$ の Gram ではなく $g$ の Gram** |
| `target/cos_vg/{i}/{j}/{role}` | 同じ Gram を角度で。$10^{-12}$ の値が「揃っているが小さい」のか「直交」なのかを分ける |
| `target/alpha_fallback/{role}`、`target/alpha_converged/{role}` | §2.3 の code（0=solved）と収束 |
| `target/K_side{k}/{task}/{role}` | 除去コスト係数 $\beta^2\mathbb E\|F_pc^{\text{base}}\|^2$ |
| `target/removed_by_integration/{task}/{role}` | $\beta(1-\alpha)\|F_pc^{\text{base}}\|$ の live 平均。統合が実際に削った量 |
| `target/removed_frac/{task}/{role}` | 同じものを $\|F_pc^{\text{base}}\|$ 比で |
| `target/inject_norm`、`target/inject_norm_nobeta` | $\beta\|F_pc\|$（既定）と $\beta$ 抜き |
| `target/inject_over_d`、`target/inject_over_d_live` | 全 token 基準（既定）と live 基準（補助） |
| `target/inject_over_r`、`target/cos_inject_r` | 注入が元 RL 方向に対してどれだけ／どちら向きか |
| `target/f_mean`、`f_p10`、`f_p50`、`f_p90` | $f$ の中心と裾。平均だけでは重みか switch かが分からない |
| `target/e_mean`、`e_mean_adv_pos/neg`、`e_clip_frac` | $e$ の水準と、clip に張り付いて定数化した割合 |
| `target/clamped_frac`、`recenter_residual`、`tv_qstar_q` | clamp が効いた割合、再中心化の大きさ、$q^\star$ と $q$ の TV |
| `target/sbar`、`sbar_ready` | $f$ の基準スケールと、それが立ち上がったか |

**参照の健全性**（v2 から流用、両半分について）

`ref_cos_sides_fv` / `ref_cos_sides_fg`（**分割半分間の再現性**）、`n_eff_side{k}`（参加率であってサンプル数ではない）、
`n_distinct_side{k}`、`ref_norm_fv_side{k}` / `ref_norm_fg_side{k}`、`prompts_side{k}`、`pg_prompts_side{k}`、
`tokens_window_side{k}`、`staleness_side{k}`、`valid`、`invalid_reason`、`invalid_steps`、`unkeyed_rows`

**精度**: 共通条件の評価で採否を決める。`episode/*_success_rate` は推移把握用。

---

## 8. 事前固定条件

| 条件 | 初版 | 根拠 |
|---|---:|---|
| $\eta$ | **1.0** | 対照ではないが、$c$ の自然な単位。実験条件として固定 |
| $\epsilon_w$（RLSD clip） | **0.2** | `rlsd_utils` の既定 |
| $c$ の clamp | **$\lvert c\rvert\le2$** | §3。恒等式は既に捨てているので、目標の集中を縛る側で決める |
| $\bar s$ の EMA | **0.8** | v2 と同じ |
| 役割 | **format, env_action** | v2 と同じ |
| 参照の有効性 | v2 と同じ（8 step 窓、`min_prompts`=4 / `min_pg_prompts`=2 / `min_tokens`=64、鮮度 2）。**両半分が満たすこと**を要求する | 片側だけで通ると再現性が測れない |
| `target_roles` | **format, reasoning, env_action, tool_call, tag** | §0。`pg_loss_coef=0` なので、外れた役割は報酬が消える |
| 分割鍵 | `split_seed`=1、v1/v2 と同じ turn-0 anchor hash | |
| $\beta$ | **0.01** | 変えない |

---

## 9. 費用

追加 forward / backward は**無い**。$q$ は pure OPD の on-task pass が既に埋め、$r$ は `policy_loss_gradient_coef` の閉形式、
$c$ は $(bs,T,k)$ のテンソル演算、$\mathrm{KL}(p\Vert q^\star)$ は `topk_kl_per_token` の第 2 引数差し替え。

**既存の target アーム（`cross_teacher_target`）が追加 forward を要するのは off-task 教師を読むからで、この機構は読まない。**

---

## 10. 限界

* $\beta F_pc$ は出力空間の量で、パラメータへの Jacobian を含まない。共有パラメータの更新を保証しない
* §2.2 の制約は**局所的な代理目的**。3 タスクの成功率の条件ではない
* 役割別平均が相殺でほぼゼロなら、条件を満たしたこと自体に意味が無い。参照の再現性を併記する
* v1/v2 の実測では $v_i^\top v_j$ の非対角が自タスク項の −0.47〜+0.18 で、制約は 10 セル中 10 セルで不活性だった。
  **ただしそれは $v$ の Gram であり、制約が使う $g$ の Gram は未測定。**$fe\in[0,2.4]$ の再重み付けが入る
* 採否は 3 タスクの評価精度で決める。ここで記録する量はどれも代理

---

## 11. 実装（2026-09-09）

* 機構: [`verl/trainer/ppo/opd_target_distill.py`](../verl/trainer/ppo/opd_target_distill.py)
  —— 設定、`rl_support_direction`（clip 後の $r$）、`opd_strength_ratio`（$f$）、`rlsd_support_factor`（$e$）、
  `fisher_apply`（$F_pc$）、`build_target`（$q^\star$ と診断の材料、すべて detached）、
  `TargetDistillStats`（step 内累積、all-reduce 1 回/バッファ）、`solve_alpha`（§2.2、**厳密な active-set 列挙**）、
  `TargetDistillController`（$\bar s$・参照・$\alpha$・checkpoint）。
* 配線: [`dp_actor.py`](../verl/workers/actor/dp_actor.py) —— step 頭で参照を 1 回読み、
  **未 tilt の KL → `build_target` → `teacher_topk_lp` を差し替え → KL** の順、backward 後に累積、末尾で解く。
  [`main_opd.py`](../verl/trainer/main_opd.py) —— 注入と `validate_target_distill_exclusivity`。
* アーム: `ARM=tdist`（$\alpha\equiv1$）と `ARM=tdist_int`（統合あり）。
  lock は `expected_multitask_opd_coef_tdist{,_int}_config.yaml`。**両者の pin 差は `integrate` の 1 値のみ。**
* **`pg_loss_coef=0` は起動時検証で強制**。1 のままだと更新が $r+d+\beta Fc$ になり報酬が二度入るが、
  run は正常に学習して見えるので、run script を信頼せず注入側で拒否する。

### 11.1 実装で分かったこと

**`build_target` は $\|d\|$ を自分で導出する。** `opd_task_diag` から取ると循環する —— あちらは教師 KL から $g_{\rm opd}$ を作り、
その KL こそこの関数が書き換える対象だから。未 tilt の KL を先に 1 回計算して渡す（テンソル演算 1 回、追加 forward なし）。

**ソルバは厳密解でなければならない。** 罰則法を先に書いたところ、粗いグリッドより 60% 悪い点を返した
（制約を満たすために $\alpha$ を下げ、戻る経路が無い）。$n\le4$ なので**全 KKT 候補の列挙**が可能で、そちらに置き換えた。

**フォールバックは必ず $\alpha=1$。** §2.3 のとおり、蒸留のみでは $\alpha=0$ が RL 信号を消す。
`solve_alpha` の全経路（非有限、受け手なし、信号なし、未解決、共通方向なし）が 1 を返すことをテストで固定した。

### 11.2 検証（CPU、初版）

* `tests/trainer/test_opd_target_distill.py`（24）: **注入方向 $d+\beta F_pc$ を autograd と照合**（残差 1e-12 以下、
  かつ平均を引かない形が実際に誤りであることも主張）、clip 済み token が目標を動かさないこと、
  $e$ が `rlsd_utils` の $w_t$ と一致すること、$\eta=0$ / $\alpha=0$ が厳密に純 OPD であること、
  $c$ の零和と clamp、統合の解がグリッド以下であること、欠測・鮮度、resume
* `tests/trainer/test_opd_target_distill_arm.py`（11）: 順序、未 tilt KL が先、backward 後の累積、
  **`pg_loss_coef` が 0 でなければ拒否**、他機構との排他（双方向）、挿入ブロックの名前解決、実 Hydra 注入
* `tests/trainer/test_opd_coef_arm.py`（62）: 8 アームの lock と shell 起動。
  **`tdist` と `tdist_int` の pin 差が `integrate` だけであることを固定**

**未修正の既存失敗が 1 件**: `test_cross_teacher_kl_weight.py::test_the_reliability_pass_...`。
`claude/cross-gate-v2` の時点で失敗しており、この変更とは無関係（クリーンな作業ツリーで確認）。

### 11.3 修正: `pg_loss_coef = 0` が機構を無効化していた

初版の実装は **GPU 上で一切発火しなかった。** このアームが必須とする `pg_loss_coef = 0` を、
actor の既存 3 経路が「方策勾配の信号が一切ない」と解釈するため。

| 経路 | 条件 | 帰結 |
|---|---|---|
| `select_keys` | `if pg_loss_coef != 0` | `advantages` が micro-batch に入らず **$e\equiv1$** |
| `need_log_prob` | `pg_loss_coef == 0 and teacher_topk_kl and ...` | **`log_prob = None`** → `build_target` のガードが通らない |
| `xt_pg_grad_coef` | `if pg_loss_coef != 0:` の内側 | **`None`** → $r=0$ |

3 つ揃うと $c=0$、$q^\star=q$。**アームは純 OPD として 300 step 回り、しかも `alpha`・`sbar`・`usable` などの
メトリクスは出続ける** —— commit が `pg_loss_coef` について警告したのと同じ「正常に学習して見える」失敗形である。

**AST 検査では捕まらなかった。** `assert "pg_grad_coef=xt_pg_grad_coef" in call` は名前が書かれていることしか見ず、
実行時にそれが `None` であることを検出しない。

修正は 3 経路すべてを `needs_policy_gradient_inputs(cfg)` という**単一の述語**に依らせた（3 箇所が独立に drift しないため）。
併せて `xt_pg_grad_coef` を要求する機構のリストに `target_distill` を加えたので、
**`task_diag` が偶然有効であることに依存しなくなった**（lock は pin しているが、検証は要求していなかった）。

回帰テストは**条件式を実際に評価する**形にした（`need_log_prob` と select ガードを AST から取り出して eval）。
3 つの修正それぞれを元に戻す変異試験で、3 つとも失敗することを確認済み。

### 11.4 修正: 2 回目のレビューが挙げた 8 件（機構 4・診断 5・記述 1）

`pg_loss_coef` の件とは別に、外部レビューが以下を指摘した。**全て修正済み**、各項目に回帰テストを 1 本ずつ付けた。

**機構**

1. **`tool_call` / `tag` から報酬が消えていた。** 目標を書き換える範囲と $\alpha$ を解く範囲が同じ集合
   （`roles`）だったので、`format` / `env_action` 以外は $\alpha=0$ 扱い、すなわち
   `pg_loss_coef=0` の下で報酬経路が一つも無い純 OPD になっていた。search の `<search>` / `<answer>` は
   `tool_call` なので、**search の課題そのものから報酬が消える**。§0 のとおり 2 集合に分離し、
   `target_roles`∖`roles` は $\alpha=1$（自タスクの RL を残す）とした。
2. **clamp と $\alpha$ の順序が統計と食い違っていた。** $\mathrm{clip}(\alpha x)\neq\alpha\,\mathrm{clip}(x)$ なので、
   統合の統計が積み上げる方向を損失が取っていなかった。$c^{\text{base}}$ を先に中心化・clamp し、
   $\alpha$ は後から掛ける（§0）。参照も $F_p\tilde r$ ではなく $F_pc^{\text{base}}$ を積む。**仕様変更である。**
3. **QP のスケール。** §2.2 に記述を追加。実測スケールでは真の最適 $\alpha=0.5$ に対して $\alpha=0$ を返していた。
4. **「解けなかった」と「非零の共通方向が存在しない」を区別。** §2.3 の表。行動は同じ $\alpha=1$ でも所見が違う。

**診断**

5. $\beta$ 込みを既定に（§7 規約 1）。`inject_over_r` も $\beta$ を含める。
6. 比の母集団を明示（§7 規約 2）。全 token 基準を既定、live 基準を `_live` で併記。
7. $\bar s$ が立つまで $f=1$。$\bar s=0$ では比が全 token で 2（上限）になり、初回 step が倍の強さで注入されていた。
   EMA の初期化は**step 集計**から行う（micro-batch 平均だと rank 配置と行順に依存する）。順序不変性をテストで固定。
8. `v_sum` / `g_sum` を削除。各 10.9 MiB を device に確保し毎 step all-reduce していたが、**誰も読んでいなかった**。
9. 不足していた指標を追加: `ref_cos_sides_*`、`prompts_side{k}` / `pg_prompts_side{k}`、
   `removed_by_integration` / `removed_frac`、`f_p10` / `f_p50` / `f_p90`、`e_clip_frac`、`cos_vg`、`unkeyed_rows`。
   併せて**有効性条件を文書（4/2/64）に合わせて実装**した —— 参照を 2 分割し、プロンプト数を数え、両半分に条件を課す。
   分割列はドライバが v1/v2 と同じ鍵で付ける（`cross_side` / `cross_prompt_idx` を v3 でも要求するようにした）。

**記述**

10. **「$F_p$ は定数を消す」を撤回。** §0 のとおり支持上では $F_p\mathbf 1=p(1-m)\neq0$。
    docstring と、それを「確認」していたテスト（$p$ を正規化していた）を直し、
    正規化した場合との差が自分のノルムを超えることを主張するテストに置き換えた。

**副次的な配線**: $\beta$ は `teacher_kl_loss_coef` から controller へ渡す（yaml で別に置くことは拒否する）。
checkpoint は参照の意味が変わったので **version 2**、v1 の state は読まずに拒否する。

### 11.5 検証（CPU）

`test_opd_target_distill.py` 43 + `test_opd_target_distill_arm.py` 17 + 近傍アーム 208 = **268 件 pass**。
新規の固定: $\alpha$ 線形性（clamp が 96% の token で効く条件下で相対 3e-8）、QP のスケール不変性（$10^3$〜$10^{-30}$）、
フォールバック 3 種の区別、$\bar s$ の順序不変性、初回 step の $f=1$、削除したバッファの不在、
$\beta$ に比例する大きさ指標、Gram が異なる半分どうしであること、片側が枯れた参照が無効になること。

### 11.6 GPU 上で未実行

比較は §6 の 3 アーム。**A vs B は単タスクの問い、B vs C だけが多タスク機構の効果。**
最初に読むのは精度ではなく `inject_over_r`、`tv_qstar_q`、`fe_mean`、そして `alpha` が 1 から動くかどうか。
