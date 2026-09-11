# 案: ロジット空間の精度重み付き和による RL–OPD 信号統合

状態: **提案。未実装。事前実験の前に固定するための文書。**

これまでの提案（静的 redistribute、pushback、cross gate v1/v2、
[受け手別一次制約 S-LP](opd_receiver_constraint_design.md)）を**置き換える**。
それらはすべて「対立の符号を見てゲートを閉じる」形だった。本案の主張は、
**ゲートは形が違う**というものである。§9 に各機構が本案の特殊例として位置づく表を置く。

数値の出所は 3 つ。(a) `~/grad_probe/terms_n8_fixed.json`（OPD+GRPO control、step 300、$N=8$）、
(b) `~/grad_probe/halves_step150_fixed.json`（純 OPD run、step 150）、
(c) `~/sign_tokens/opd_grpo_multitask_opd_coef_cross2k100_qwen3_1.7b/`（k100 run、5 step ごと、52 本）。
(c) は本案のために新たに読んだもので、§6 に結果を置く。

---

## 1. 問題の所在

1 step の中で、共有パラメータは 6 本の信号を受ける。RL 3 本（タスクごと）と OPD 3 本（教師ごと）。
これらが**同じ座標の上で出会う場所**でなければ「対立」は定義できない。

位置（トークンの出現箇所）では出会わない。タスク $i$ の位置には RL$_i$ と OPD$_i$ しか乗らず、
タスク $j$ の信号は乗らない。出会うのは**共有された語彙 id** である。id $v$ のロジットは
すべての位置で同じ出力行 $w_v$ から作られるので、全タスクの全位置が同じ座標 $v$ を押す。

$$
a_i[v] = \sum_{t\in T_i} \rho_t\, c_t\,\big(\mathbb 1[y_t=v]-\pi_t[v]\big),
\qquad
d_j[v] = \sum_{t\in T_j} \rho_t\,\big(\tau_t[v]-\pi_t[v]\big)
$$

$c_t$ は dp_actor が既に持つトークン係数（advantage・ratio・clip 込み）、
$\rho_t$ は `task_loss_weight`、$\tau_t$ は教師の top-k 分布。
これは「共有出力バイアス $b_v$ が発信源別に受け取る勾配」そのものである。

**恒等式。** $\sum_v(\mathbb 1[y=v]-\pi[v])=0$ かつ $\sum_v(\tau[v]-\pi[v])=0$ なので、
6 本すべてが語彙について**厳密に総和ゼロ**。以下の分散推定で中心化は不要である。

---

## 2. モデルと仮定

座標 $v$ ごとに独立に扱う。

| 記号 | 意味 |
|---|---|
| $\theta_i[v]=\partial J_i/\partial b_v$ | タスク $i$ の**真の**目的勾配（未知） |
| $\Theta[v]=\sum_i\theta_i[v]$ | 結合目的 $J=\sum_i J_i$ の真の勾配 |
| $a_i[v]=\theta_i[v]+\varepsilon_i[v]$ | RL 観測。不偏、分散 $\sigma_i^2[v]$ |
| $d_j[v]=\lambda_j\theta_j[v]+\zeta_j[v]$ | 教師観測。未知の尺度 $\lambda_j$、誤差分散 $\varsigma_j^2$ |

仮定は 4 つで、すべて明示的に外せる。

1. **加法性**: 共有座標への押しは発信源について和。共有出力バイアスのモデル。
2. **等重み目的**: $J=\sum_i J_i$。これは `normalize_loss_by_task=True` が既に宣言している。
3. **教師はタスク $j$ の目的についての証拠**であって独立した目的ではない。
   $\lambda_j$ が大きいほど有用、$\lambda_j\approx0$ なら無情報、$\lambda_j<0$ なら有害。
   **これは仮定ではなく推定する量**である（§4.4）。
4. **最適化器は Adam 型**で座標ごとに大きさを正規化する。

仮定 4 が最適性の形を決める。Adam の実効ステップは $\text{lr}\cdot\hat m/\sqrt{\hat v}$ で、
座標ごとに大きさが割られる。残るのは**向きと信号雑音比**だけである。

---

## 3. 最適な結合

適用する押しを $P[v]=\sum_i w_i a_i[v]+\sum_j u_j d_j[v]$ とする。

**命題（Gauss–Markov）。** $P$ が $\Theta$ について不偏、すなわち各 $i$ で $w_i+u_i\lambda_i=1$
という制約の下で $\mathrm{Var}[P]$ を最小にする係数は一意に

$$
w_i=\frac{1/\sigma_i^2}{1/\sigma_i^2+\lambda_i^2/\varsigma_i^2},
\qquad
u_i=\frac{\lambda_i/\varsigma_i^2}{1/\sigma_i^2+\lambda_i^2/\varsigma_i^2}
$$

*証明.* 制約付き最小化の Lagrange 条件は $2w_i\sigma_i^2=\mu_i$, $2u_i\varsigma_i^2=\mu_i\lambda_i$。
比を取ると $u_i/w_i=\lambda_i\sigma_i^2/\varsigma_i^2$、制約に代入して上式。∎

Gauss 性は要らない。不偏性と分散最小だけである。仮定 4 により、
座標ごとの大きさは Adam が割るので、**分散最小化がそのまま期待進行の最大化になる**。

**縮小（第 2 層、任意だが推奨）。** BLUE は不偏だが、二乗誤差では 0 へ縮めた方が良い。
$s_i^2[v]=\big(1/\sigma_i^2[v]+\lambda_i^2/\varsigma_i^2\big)^{-1}$ を BLUE の分散、
$\hat\tau_i^2$ を id をまたいだ真の信号の分散とすると

$$
\hat\theta_i[v]=\underbrace{\frac{\hat\tau_i^2}{\hat\tau_i^2+s_i^2[v]}}_{k_i[v]}\cdot
\underbrace{s_i^2[v]\Big(\frac{a_i[v]}{\sigma_i^2[v]}+\frac{\lambda_i d_i[v]}{\varsigma_i^2}\Big)}_{\text{BLUE}},
\qquad
P[v]=\sum_i\hat\theta_i[v]
$$

James–Stein の正部分推定量に対応する。雑音が支配する座標が自動的に抑えられる。

**タスク間に門は無い。** 教師 $j$ が語るのはタスク $j$ の $\theta_j$ についてだけなので、
$i\ne j$ の交差項は存在しない。「教師 $j$ の押しがタスク $i$ の報酬に逆らう」ことは、
RL$_j$ が RL$_i$ に逆らうのと同じ資格の出来事であり、等重み目的の下では**和が正解**である。
S-LP がタスク間制約を置いたのは、教師を目的の証拠ではなく無内容な補助損失と見たとき
（Du et al. 2018 の枝）の扱いで、蒸留には当てはまらない。

---

## 4. 推定量

すべて backward の**前**に、forward にある量だけから求まる。

### 4.1 集約
§1 の 2 式。6 本 × 語彙 15 万 = 3.6 MB。

### 4.2 RL の雑音床（順列、閉形式）

GRPO はグループ内で advantage を中心化しているので $\bar A_g=0$。
グループ $g$ のロールアウト $k$ の id プロファイルを
$m_{g,k}[v]=\sum_{t\in(g,k)}\rho_t r_t(\mathbb 1[y_t=v]-\pi_t[v])$ とすると
$a_i[v]=\sum_g\sum_k A_{g,k}m_{g,k}[v]$。グループ内で $A$ を入れ替える帰無分布は

$$
\mathbb E_{\rm perm}[a_i[v]]=0,\qquad
\sigma_i^2[v]=\sum_{g\in i}\frac{\big(\sum_k A_{g,k}^2\big)\big(\sum_k (m_{g,k}[v]-\bar m_g[v])^2\big)}{n-1}
$$

$n=8$。**標本を取る必要はない**（閉形式）。グループごとに $8\times V$ の一時領域（5 MB）で回る。

### 4.3 事前分散

$a_i$ は語彙について総和ゼロなので中心化不要。

$$\hat\tau_i^2=\max\Big(0,\ \overline{a_i[v]^2}-\overline{\sigma_i^2[v]}\Big)$$

**id の活動量（$\sum_t\pi_t[v]$）の十分位ごとに別々に推定する。**
よく出る id は信号も雑音も大きく、1 個の事前分散では合わない。

### 4.4 教師の重み（id をまたいだ回帰）

$d_j$ を $a_j$ に回帰する。$a_j$ 自身に測定誤差があるので減衰補正を入れる。

$$
\hat\lambda_j=\frac{\overline{a_j d_j}}{\overline{a_j^2}-\overline{\sigma_j^2}}
=\frac{\overline{a_j d_j}}{\hat\tau_j^2},
\qquad
\hat\varsigma_j^2=\overline{d_j^2}-\hat\lambda_j^2\,\hat\tau_j^2
$$

$\hat\tau_j^2$ が §4.3 と同一の量であることに注意。1 回の推定が両方に効く。

**役割別に持つ**（format / env_action / tag / reasoning / tool_call / env_obs）。
tag は再現するが format は再現しないという既知の非対称がここに入る。

これが**信頼の文書が求めていた「勾配の符号ではない信頼の証拠」**である。
対角の整合をトークンごとの符号門としてではなく、id をまたいだ回帰係数として使う。

### 4.5 結合
§3 の式。等価な per-term 重みは

$$
\omega^R_i[v]=k_i[v]\,\frac{s_i^2[v]}{\sigma_i^2[v]},\qquad
\omega^D_i[v]=k_i[v]\,\frac{s_i^2[v]\,\hat\lambda_i}{\hat\varsigma_i^2}
$$

健全性: 教師が無情報（$\hat\lambda_i=0$）なら $\omega^D=0$, $\omega^R=k_i\le1$、RL を縮めるだけ。
RL が純雑音（$\sigma_i^2\to\infty$）なら $\omega^R\to0$, $\omega^D\to k_i/\hat\lambda_i$、
教師の推定値を尺度直しして使う。

### 4.6 二重利用の回避

$a_i$ から作った重みを同じ $a_i$ に掛けると縮小が二重にかかる。
prompt を交わらない 2 群に分け、**A 群の集約から作った重みを B 群に適用**し、逆も行う。
step 間 EMA（decay 0.8）も併用する。§6 の測定では 5 step 離れた押しの相関が 0.970 なので、
長い窓が使える。

---

## 5. 実装

### 5.1 損失を組まず、cotangent を 1 回

重みはロジット勾配への座標ごとの掛け算なので、**スカラー損失を作らずに
$\partial L/\partial z$ を直接組み立て、1 回の backward に渡す**。

```python
# forward 内。両方すでに materialise されている
g_rl  = c_t.unsqueeze(-1) * (onehot_y - pi)          # (bs, resp, V)
g_opd = (tau - pi)                                    # top-k 支持上
g_z   = -(wR[task].gather(...) * g_rl + wD[task].gather(...) * g_opd)
torch.autograd.grad(logits, params, grad_outputs=g_z)  # backward 1 回
```

**これが「追加 forward/backward なし」の根拠である。** 2 つの信号はロジット空間では
最初から別々に存在していて、足すのは我々である。足す前に重みを掛ければよい。
パラメータ空間で $r_i$ と $d_j$ を分離する必要は無い。

（[S-LP 文書 §4.2](opd_receiver_constraint_design.md) の「1 backward = 1 VJP だから厳密な分離は不可能」は
パラメータ空間で分離しようとしたときの話で、分離すべき場所を間違えていた。）

### 5.2 コスト

| 項目 | 量 |
|---|---|
| 追加 forward / backward | **なし** |
| scatter-add | 6 本 × 約 7.5 万トークン/タスク |
| 一時領域 | $8\times V$（グループ順列）= 5 MB、集約 3.6 MB |
| 回帰 | $O(V)$ × 3 タスク |
| ソルバ・$\lambda$・KKT 証明書 | **不要**（現行 cross gate より単純） |

### 5.3 何を消すか

現行 cross gate の `solve_role`、$\lambda$ の状態、証明書、`q_scale`、役割マスク、
半分割参照 $v_i^s$、$\gamma$。すべて本案には現れない。

---

## 6. 既存 dump からわかっていること（GPU 不要）

k100 run は 5 step ごとに語彙 id 別の OPD 押しを吐いている
（`token_stats.logit_push`、`push_top_n=32`、タスク別 scope）。
`base_logit_push` が本案の $d_j[v]$ **そのもの**である。step 256 の dump で:

| 所見 | 数値 | 含意 |
|---|---|---|
| OPD 押しの集中 | webshop は上位 8 id で 79%、上位 32 で 94%。alfworld は上位 32 で 82% | **id 座標には内容がある。** 座標ごとに重みを変える意味がある |
| 押しの正体 | `<th`, `<`, `.`, `>ĊĊ`, `.ĊĊ` | 書式トークン。役割別の $\hat\lambda$ が要る |
| 教師同士の不一致 | 3 タスク共通 id で 3 者の符号一致は 44%。対ごとに 0.50〜0.66、相関 +0.22〜+0.69 | 教師は id 上で実際に割れている |
| 時間安定性 | 5 step 離れた押しの相関 **0.970** | EMA の窓は長く取れる |
| **search の押しの総量** | alfworld 200、webshop 247 に対し **search は 11** | 教師 search はロジットをほとんど押していない。$\hat\lambda_{\rm search}/\hat\varsigma^2_{\rm search}$ は小さく出ると予想され、**全タスク一律 $\beta=0.01$ が最も外れている場所** |

**足りないのは RL 側 $a_i[v]$** である。dump には `sampled_count` と `p_student_mean` はあるが
advantage で重み付けした押しが無い。回帰にはこれが要る。§7 手順 1 がそれを足す。

---

## 7. 事前実験

| 手順 | 内容 | GPU | 時間 |
|---|---|---|---|
| 1 | 計測コード。6 本の集約、順列分散、回帰、dump。**学習には触れない** | 不要 | 数時間 |
| 2 | 観測 run。checkpoint 1 つで 10 step。$\hat\lambda_j,\hat\varsigma_j^2,\sigma_i^2[v],\hat\tau_i^2$ を出す | 要 | 1.5 h |
| 2' | 同じものを early（step 25）と late（step 150）の 2 点で | 要 | +1.5 h |
| 3 | 判定 | — | — |

**手順 3 の判定基準**（事前に固定する）:

* $\hat\lambda_i\sigma_i^2/\hat\varsigma_i^2$（= 現行 $\beta$ が暗黙に主張している比）が
  タスク間で**桁で違う**、または 0.01 と桁で違う → 本 run へ進む価値がある。
* 3 タスクで同じで 0.01 に近い → 本案の取り分は小さい。$\omega$ の id 依存だけが残る。
* $\hat\lambda_j\le0$ のタスクがある → その教師は現在の報酬にとって逆向き。
  $\beta_j$ を下げる根拠が初めて測定から出る。
* 2' で early と late の $\hat\lambda$ が大きく違う → 静的較正はすべて無効。
  これは step 300 の 1 点しか測っていないという積年の穴を同時に塞ぐ。

観測 run は学習を変えないので control としても使える（checkpoint も val も control と比較可能）。

---

## 8. 限界

1. **id 座標は文脈を潰す。** 実際に共有されているのは lm_head の行 $w_v$（2048 次元）で、
   文脈の違う位置は $w_v$ の中で直交しうる。同じ id を文脈 A で上げ B で下げる信号は和の中で消える。
   細分（id × 役割、id × 隠れクラスタ）で改善する。極限の id × 位置では共有が消え、対立もゼロになる。
   **粗さは 1 つの設計変数**で、現行の役割分割はその一点である。
2. **head 経路のみ。** 埋め込みの似た id が trunk を通じて結合する経路は捉えない。
   根拠になる測定: プローブの `root` unit（勾配エネルギーの 3.1%）の交差効果行列は、
   全体との**符号が 9 セルすべて一致、相関 0.785、絶対値は 3〜7 倍ずれる**。
   → **向きは捉え、大きさは外す**と読むべき。
3. **等重み目的を仮定する。** 「どのタスクも悪化させない」が欲しいなら別の基準であり、
   争われた id を $(1-c)$ 倍する CAGrad 型の項が要る。search の $-0.70$pp が許容できないなら、
   それは目的の変更として宣言する。
4. **教師のゼロ次の価値は $\hat\lambda$ に映らない。** 報酬に効くまで時間がかかる知識は
   当該 step の回帰に現れない。$\lambda_j$ に正の事前分布を置き EMA で更新する。
5. **PPO clip は $a_i$ に偏りを入れる。** 実測 `n_clip/clipfrac_den` = 524/618075 = **0.085%** なので
   この設定では無視できる。clip が増える設定では不偏性の仮定が崩れる。
6. **縮小は id をまたいだ Gauss 事前を仮定する。** 実際は重い裾。活動量十分位別の $\hat\tau^2$ は緩和だが完全ではない。
7. **RL–RL の対立は解消しない。設計上そうする。** 等重み目的の下では和が正しい勾配であり、
   有意な RL 同士の対立は誤りではなく「この座標が 2 つのタスクに取り合われている」という情報である。
   消すなら §8-3 として宣言する。

---

## 9. 既存機構との関係

いずれも本案の特殊例、または本案が否定する形である。

| 機構 | 本案での位置 |
|---|---|
| 単純和 $\sum a_i+\beta\sum d_j$ | $\sigma_i^2$ を全 id・全タスクで定数、$\hat\lambda_j^2/\hat\varsigma_j^2$ を定数 $\beta$ に固定。**$\beta=0.01$ は「教師は 1 step の RL より 100 倍精度が低い」という未測定の主張** |
| 静的 redistribute $(1.076,1.191,0.5)$ | $\beta$ をタスク別にした。id 依存と縮小は無い。step 300 の 1 点で較正 |
| pushback（符号門 + $\varepsilon$） | $\omega^D$ を $\{a_i,1\}$ の 2 値にした粗い近似。符号は雑音の多い推定量の**最も当てにならない部分** |
| cross gate v1/v2（$w=1-\lambda h$） | 同上。加えて交差の核を「役割一致」で置換していた。本案では核は不要（id が共有座標そのもの） |
| S-LP（受け手別一次制約） | タスク間に門を置いた。§3 の通り、等重み目的の下でその門に根拠は無い |
| PCGrad / MGDA | 争われた座標を凍結。目的を変える（Pareto 基準）。本案は和の目的を保つ |
| CAGrad($c$) | 争われた座標を $(1-c)$ 倍。和の停留点に収束するので目的は保たれる。§8-3 の選択肢 |
| Du et al. 2018（補助損失の cos 門） | 「補助損失に固有の価値は無い」前提。蒸留では半分しか成り立たない（§8-4） |

**対立統計の測定は無駄にならない。** 使い道が、トークンごとの門から、
重みを校正する回帰へ変わるだけである。
