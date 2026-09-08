# 押し戻し制御の多タスク拡張（改訂版）: 項別勾配のスケッチによる、タスクごとの保護制約付き cross 係数

状態: **提案のみ。実装なし。** 先行: [opd_pushback_cross_task_extension.md](opd_pushback_cross_task_extension.md)（初版。§3 の測定設計はレビューで棄却）、
レビュー（9 点、日本語訳は会話に掲載）。本稿はレビューの「制御構造は残し、測定を書き直し、有益 OPD の保護を足す」に従って書き直したもの。

前提となる現行機構（変えない）: `ARM=pushback`（`f7333cb`）の self gate
$w^{\rm self}_{j,t}=a_j$（自タスクの RL と対立する token）/ $1$（他）、$a_j=\min(1,\varepsilon R_j/C_j^-)$、ε=0.003、EMA、1 step 遅れ。

---

## 0. 機構の要約

1. **損失**: $L=L_{\rm GRPO}+\beta\sum_j b_j\,\mathrm{Agg}_t\big[w^{\rm self}_{j,t}L^{\rm OPD}_{j,t}\big]$。
   追加は task ごとのスカラー $b_j\in[b_{\min},1]$ だけ。$b\equiv1$ で現行と一致する。
2. **測る量**（パラメータ空間、同じモデル状態 $\theta_k$、rank 間を集約した後に内積）:
   $r_i=\nabla L^{\rm RL}_i$（RL 単独）、$\tilde d_j=\nabla[\beta\sum_t w^{\rm self}_{j,t}L^{\rm OPD}_{j,t}]$（self gate 込み、$b$ 抜き）、
   $\bar C_{ij}=\langle r_i,\tilde d_j\rangle$、$\bar R_i=\|r_i\|^2$。
3. **取り方**（§8 で改訂: masked backward 6 本 → OPD 項のみの backward **1 本** + 学習 backward への hook。その前に出力層 proxy の直接検定を置く）: $K$ 個に 1 個の mini-batch で、1 本の micro-batch について、
   勾配を**線形スケッチ**（count-sketch、$k=2^{20}$）に落として rank 間で all-reduce し、スケッチ同士の内積を取る。
   学習の勾配には触れない（§2）。
4. **偏りの扱い**: 内積は符号付きのまま EMA し、**平均してから負部分を取る**。$[-\,\cdot\,]_+$ を先に取る現行の C⁻ とは違う定義（§3）。
5. **係数**: 3 変数の小さな QP。目的 $\min\sum_j(1-b_j)^2$、制約は (a) 保護対象タスクごとの押し戻し上限、
   (b) cross 制御によって失う有益な降下の下限（レビュー 7）。実行不能なら $b=1$（§4）。
6. **数値クラス**: $b\equiv1$ なら学習の勾配はビット同一（行の並べ替えを伴わない。測定は `.grad` を汚さない）。

---

## 1. 制御構造（レビュー §9・7 の形）

保護対象タスク $i$ ごとに

$$
\text{(a)}\quad \sum_{j\ne i} b_j\,\big[-\bar C_{ij}\big]_+ \;\le\; \varepsilon_\times\,\bar R_i
\qquad\qquad
\text{(b)}\quad \sum_{j}(b_j-1)\,\bar C_{ij} \;\ge\; -\,\delta_i\,\bar R_i
$$

* (a): 他教師がタスク $i$ の報酬降下を打ち消す量を $\varepsilon_\times$ 倍以内に抑える。分母はタスク $i$ 自身の $\bar R_i$（全タスクの和ではない）。
* (b): $b<1$ にした結果、タスク $i$ が失う降下 $\langle r_i,\Delta g(b)\rangle$、$\Delta g(b)=\sum_j(b_j-1)\tilde d_j$ を $\delta_i\bar R_i$ 以内に抑える。
  $j=i$ の項（自教師の有益な寄与）も含む。webshop の OPD を縮めて alfworld が得をするとき、webshop が失う有益な寄与も検査される。
* 目的関数は現状維持（$b=1$）からの最小の乖離。$b_{\min}$ は事前に選ぶ実験条件（例 0.5）。
* $b=1$ は (b) を常に満たす。(a) が $b=1$ で破れているときだけ縮め、(a)(b) が両立しなければ**縮めない**（$b=1$ を保ち `cross_infeasible` を記録）。
  制御器を作動させるためにタスクを犠牲にしない。
* 1 step 遅れ、step 内固定、EMA、状態機械（§3）、`actor_extra` への保存と設定変更時の resume 拒否は現行と同じ。

---

## 2. 測定: 項別勾配のスケッチ

### 2.1 何を避けるか（レビューで落ちた点）

| 初版 | 問題 | 本稿 |
|---|---|---|
| $g_i=r_i+d_i\approx r_i$ | 汚染項が中央値 38〜64%、符号反転 3/8 | RL 単独を masked backward で取る |
| rank ごとに内積→スカラー reduce | rank 間の項が落ちる | **線形スケッチを all-reduce してから内積** |
| `flat_param.grad` の差分（bf16） | 小さい増分が消える。最後の micro-batch は `no_sync` 外 | 差分を取らない。`.grad` を空にしてから 1 項だけ backward し、スケッチして戻す |
| `autograd.grad(flat_params)` | bf16 混合精度でグラフに無い | 通常の backward + `zero_grad` |
| task-pure な並べ替え | rank あたり 3 本で少数タスクが消える。加算順序も変わる | 並べ替えない。混在 micro-batch で mask により分離 |

### 2.2 手順（$K$ 個に 1 個の mini-batch、その中の 1 本の micro-batch）

1. その micro-batch の `.grad`（unsharded、`no_sync` 中）を bf16 で snapshot する（3.4 GB/rank）。**mini-batch の先頭 micro-batch を選ぶなら snapshot は不要**
   （`zero_grad` 直後で空）だが、`deal_by_length` は先頭に最長行を置くので長い行（webshop）に偏る。無作為な位置を選び snapshot を払う方を採る。
2. forward を 1 回（`retain_graph=True`）。
3. 存在するタスク $i$ ごとに、`.grad` を 0 にし、RL 項をタスク $i$ の行に mask した損失で backward → `.grad` に $r_i^{(m)}$ が立つ → スケッチ $s(r_i^{(m)})$ を取る。
   同様に OPD 項（self gate 込み、$b$ 抜き、係数 β）をタスク $j$ の行に mask して backward → $s(\tilde d_j^{(m)})$。
   backward は通常経路なので FSDP の hook は普段どおり動き、`no_sync` 中は reduce しない。
4. `.grad` に snapshot を書き戻す（bf16 の copy なので厳密）。
5. 学習の backward（全項、全行）を普段どおり走らせる。**学習の勾配は測定が無いときと同じ**。
6. スケッチを rank 間で all-reduce（SUM）。線形性により $\sum_q s(v^{(q)})=s(\sum_q v^{(q)})$ で、これが「先に集約してから内積」になる。
   通信は 4 MB × ベクトル数。

### 2.3 スケッチ

count-sketch: 座標 $c$ を bucket $h(c)\in\{1..k\}$ に符号 $\sigma(c)\in\{\pm1\}$ で足す。$h,\sigma$ は座標の**大域**インデックスから決定的に作る
（unit と shard の offset から計算。rank 間で同じ写像）。

* $\mathbb E\langle s(u),s(v)\rangle=\langle u,v\rangle$、$\mathbb E\|s(v)\|^2=\|v\|^2$（不偏）。
* 標準誤差 $\approx\|u\|\|v\|/\sqrt k$。$k=2^{20}$ で $|\cos|\approx0.1$ の内積に対し相対 1%。
* 計算は 1.7B 座標の chunk ごとの `scatter_add_`（インデックスは chunk ごとに生成、保持しない）。1 ベクトル ≈ 50 ms。
* 蓄積は fp32（bucket 和）、内積は fp64。bf16 の `.grad` を読むが、1 項だけの新しい累積なので差分の桁落ちは無い。

### 2.4 費用（正直に）

micro-batch 10、rank あたり mini-batch に 3 本、$K$ 個に 1 個の mini-batch で 1 本を測る。

| 項目 | 量 |
|---|---|
| 追加 backward | 存在タスク数 × 2（最大 6）。gradient checkpointing により各 backward は forward の再計算を含む（≈ forward+backward） |
| 追加 forward | 1（retain）。学習 forward と共有できるなら 0 |
| 時間 | 学習 micro-batch ≈ 4 単位（fwd 1 + 再計算 1 + bwd 2）に対し、測定 micro-batch は +18 単位（6 × 3）程度。mini-batch 12 単位 → $K=8$ で **+19%**、$K=16$ で **+9%** の update |
| メモリ | snapshot 3.4 GB + retain した活性（checkpointing 下では層入力 + logits ≈ 2〜3 GB）+ スケッチ 6 × 4 MB → **+6 GB 程度**。初版の +7〜14 GB より小さいが、余裕（少数 step の実測 37〜50 GB）は保証値ではないので 3 step probe で測る |
| 標本 | $K=8$ で step あたり ≈ 9 micro-batch（両 rank で 20 行）。1 step では雑音が大きく、EMA（0.9 → 約 10 step、≈ 90 標本）で読む |

行の抽出は「無作為な mini-batch × 無作為な micro-batch 位置」で、行順の偏りを避ける。被覆（各タスクの行数、生存群数、各対の観測数）を毎回記録する。

---

## 3. 推定量と偏り

* 推定するのは **mini-batch 水準・$\theta_k$ での局所的な交差効果**。N=8 プローブ（固定 checkpoint、batch 全体を累積）とは別の推定量で、
  そう命名する。同じ checkpoint で lr=0 の 1 step を回せば $\theta_k$ が固定され、和の違い（mini-batch 間の交差項）だけが残るので、較正の比較はそこで行う。
* **Jensen の偏り**: $\mathbb E[(-\hat C)_+]\ge(-\mathbb E\hat C)_+$。負部分を先に取ると雑音が押し戻しに化ける
  （N=8 で webshop の保護比率が 1.38% → 平均後に切ると 0.000%）。本稿は **符号付き $\hat C_{ij}$ と $\hat R_i$ を EMA し、制約の評価時に負部分を取る**。
  作動するには一貫した負の平均が要る。現行 self gate の C⁻ は token ごとの負部分で偏りを持つが、それは現行の振る舞いなので変えない（記録のみ）。
* **状態機械**（対 $(i,j)$ ごと）: `absent`（観測なし）/ `few`（観測数 < `min_cross_obs`、または生存群数 < `min_live_groups`）/ `ok`。
  制約 (a)(b) は `ok` の対だけで評価する。保護対象 $i$ の必要な対に `few`/`absent` があれば、**その $i$ の制約は評価しない**（欠測を 0 干渉で埋めない）
  → その step は $b$ を前の値のまま保つ。
* self gate の $a_j$ は従来どおりロジット空間で決める。$\tilde d_j$ は $a_j$ を含み $b_j$ を含まない（制御器が自分の出力を読まない）。

---

## 4. 係数の決定

各 step の末尾で、EMA 後の $\bar C_{ij}$（3×3）、$\bar R_i$ から

$$
\min_{b_{\min}\le b\le 1}\ \sum_j(1-b_j)^2
\quad\text{s.t.}\quad
\text{(a)}\ \sum_{j\ne i}b_j[-\bar C_{ij}]_+\le\varepsilon_\times\bar R_i,\qquad
\text{(b)}\ \sum_j(b_j-1)\bar C_{ij}\ge-\delta_i\bar R_i
\qquad(i\in\text{ok})
$$

3 変数・6 制約なので格子か active set で解く。実行不能なら $b=1$ と `cross_infeasible=1`。次 step に適用、step 内固定。

指標: `cross/b_applied/{task}`、`cross/C/{i}/{j}`（EMA、符号付き）、`cross/R/{task}`、`cross/ratio_a/{task}`（(a) の左辺/右辺）、
`cross/loss_b/{task}`（(b) の左辺/右辺）、`cross/state/{i}/{j}`、`cross/obs/{i}/{j}`、`cross/infeasible`、`cross/sketch_k`。

---

## 5. 作動点の予測（N=8、平均後に切る定義）

| 保護対象 $i$ | $\sum_{j\ne i}[-\bar C_{ij}]_+/\bar R_i$ | 主な押し戻し元 |
|---|---:|---|
| alfworld | **1.1%** | webshop の教師（$\bar C_{\rm alf,web}=-0.19$） |
| search | 0.2% | webshop |
| webshop | **0.0%** | （平均は正。1.38% は負部分先取りの偏り） |

* self gate の ε=0.003 と同じく作動点に相対的に置くなら、$\varepsilon_\times$ を 0.5% 前後にすると alfworld の (a) が $b=1$ で破れ、$b_{\rm web}$ が下がる。
* そのとき (b) for webshop: $\bar C_{\rm web,web}=-0.10<0$ なので $b_{\rm web}<1$ は webshop の降下を**増やす**方向で、(b) は拘束しない。
  step 300 の数値では「webshop の教師を縮めて alfworld を保護し、webshop も損をしない」が解になる。早期は未測定で、これは予測であって結果ではない。
* $\varepsilon_\times$、$\delta_i$、$b_{\min}$、$K$ は実験条件で、測定から最適値は出ない。lock に理由を書く。

---

## 6. 検証の順序

1. **2 rank CPU 小モデル**（レビューと同じ枠）: bf16 混合精度・`SHARD_GRAD_OP`・`no_sync`・non-reentrant checkpoint の下で、
   (i) snapshot → zero → masked backward → restore → 学習 backward の勾配が、測定無しの経路とビット同一、
   (ii) スケッチ内積が厳密内積と誤差 $O(1/\sqrt k)$ で一致、(iii) 2 rank の all-reduce 後のスケッチ内積が「全行を 1 rank で流した」内積と一致
   （rank 間の項の回復）、(iv) 欠測時に制約が評価されないこと、(v) QP の実行不能で $b=1$。
2. **同一 checkpoint での較正**: `klw_control_step300_hf` を lr=0 で 1 step 回し、online の $\bar C$ を N=8 プローブと比べる。
   符号と桁が合うことを確認する（推定量は別なので一致は求めない）。
3. **3 step の GPU probe**: update の時間増と rank 別の物理ピーク（`nvidia-smi`）、被覆率、`cross/state` の分布。
4. **Tier 1**: $b\equiv1$ で本番に入れ、3×3 の時系列だけ残す。**Tier 2**: Tier 1 の分位で $\varepsilon_\times$ を決めて有効化。

---

## 7. 言えないこと

* token 単位の交差制御は行わない（測定が無い）。$b_j$ はタスク一様で、失う有益 OPD は (b) で上限を置くだけ。
* Adam の前処理は入らない（$P=I$ のスケッチ）。
* RL–RL の衝突は対象外。
* 局所（mini-batch・$\theta_k$）の一次量で、次 step の実際の損失変化を保証しない。採否は 3 タスクの評価精度で決める。
* 費用とメモリは probe で測るまで見積りである。

---

## 8. 改訂: 追加 backward は 6 本ではなく 1 本、そして先に出力層 proxy を正しく検定する

「現行機構も別案も出力空間の微分で判定しているのに、なぜ追加 backward を要求するのか」という問いへの答えと、その帰結。

### 8.1 なぜ対角は出力空間で足りて、非対角は足りないか

対角: $u_{R,t}$ と $u_{D,t}$ は**同じ token の同じロジット** $z_t$ に対する微分である。両者は同じ点に住むので、
その内積 $u_R^\top u_D$ はそれ自体で「この位置で報酬と教師が押し合っているか」という意味を持ち、Jacobian は要らない。

非対角: $r_i$ はタスク $i$ の token 上、$\tilde d_j$ はタスク $j$ の token 上にある。**共通の出力は存在しない。**

$$
\langle r_i,\tilde d_j\rangle=\sum_{s\in i}\sum_{t\in j} u_{R,s}^\top J_s J_t^\top u_{D,t}
$$

2 つを結びつけているのは共有パラメータ、すなわち $J_sJ_t^\top$ そのものである。だから「出力空間で足りない」のではなく、
**非対角では出力空間に測るべき共通の場所が無い**。これが現行機構が単タスクに見える理由でもある。

### 8.2 別案の proxy は、この $J_sJ_t^\top$ を最終層 1 枚で近似する案だった

$G_t=u_th_t^\top$ は出力層の勾配そのもので、$\langle G_s,G_t\rangle_F=(u_s^\top u_t)(h_s^\top h_t)$。
最終層に限れば Jacobian を明示的に運んでいる。追加 backward が要らないのは、$u$ と $h$ が forward の値だからである。
**筋は正しい。** 問題は「最終層 1 枚が全体を代表するか」だけで、これは検定できる。

### 8.3 私の検定は決定的ではなかった（レビュー 8 の指摘は正しい）

§9 で使った `root` は `tie_word_embeddings=true` の下で**入力埋め込みと出力 head が同一パラメータ**であり、
保存された勾配は両経路の和である。head 単独の量ではない。したがって「root が全体を追わない」から
「head proxy が使えない」は導けない。

**決定的な検定は安く、backward を 1 本も要らない。** $u_t$ は既に閉形式で計算済み（`opd_pg_alignment_terms`）、
$h_t$ は `_capture_last_hidden` が既に取れる。よって

$$
\langle G_i^{RL},G_j^{OPD}\rangle=\Big(\sum_{s\in i}u_{R,s}\Big)\text{型の二重和}=\sum_{s\in i}\sum_{t\in j}(u_{R,s}^\top u_{D,t})(h_s^\top h_t)
$$

を head 単独で計算し、同じ batch のモデル全体の交差内積（$N=8$ プローブ、または §8.4 の 1 本 backward）と符号を比べればよい。
二重和は $(\sum_s u_{R,s}h_s^\top)$ と $(\sum_t u_{D,t}h_t^\top)$ の Frobenius 内積なので、
$O\times H$ の行列 2 つ（Qwen3-1.7B で 151936×2048 は大きすぎるので、$u$ 側は support 上の疎な形か、
`(n_tok × k)` の係数と `(n_tok × H)` の $h$ の積として保持する）で済む。

**これを先に走らせる。** 一致すれば別案が正しく、追加 backward は不要で、しかも token 単位の $H_j$ が作れる。
一致しなければ §8.4 に進む。probe 1 回、GPU 追加 backward 無し。

### 8.4 backward が要る場合でも 1 本で足りる（6 本は過剰設計だった）

本稿 §2.2 は「存在タスク数 × 2 = 最大 6 本の masked backward」と書いた。これは 2 つの構造を使っていない。

1. **タスクの分離は無料。** タスクは**行が互いに素**なので、backward の hook で層ごとに
   $G_i^\ell=\delta^\ell[\text{rows}\in i]^\top a^\ell[\text{rows}\in i]$ をタスク別に足せば、1 本の backward から 3 タスク分が出る。
   masked backward を 3 回走らせる必要は無い。
2. **項の分離は backward の seed に対する線形性で足りる。** backprop は seed $u$ について線形なので、
   OPD 項だけを seed にした backward 1 本で $\tilde d_j$（3 タスク分）が出る。学習の backward は seed $u_R+u_D$ なので
   そこから $g_i$（3 タスク分）が出て、**$r_i = g_i-\tilde d_i$ が厳密に得られる**（近似ではない。レビュー 1 が棄却したのは
   $g_i\approx r_i$ という近似であって、この差分ではない）。スケッチは線形なので $s(r_i)=s(g_i)-s(\tilde d_i)$。

したがって **追加は「OPD 項だけを seed にした backward 1 本」と、学習 backward への hook** で足りる。
費用は 6 本 → 1 本で、gradient checkpointing の再計算を含めても測定 micro-batch あたり概ね
「学習 backward の +1 本分 + タスク別の weight-grad matmul の増分」になる。$K$ 個に 1 個で $K=8$ なら update +5〜8% 程度と見込むが、
probe で測る。メモリは snapshot 3.4 GB + スケッチで、§2.4 より小さい。

**変わらない点**: rank 間はスケッチを all-reduce してから内積（レビュー 2）、差分は bf16 の勾配同士ではなく
**同じ backward 内の項別**なので桁落ちの経路が違う（ただし $g_i-\tilde d_i$ は近い量同士の差になりうるので、
スケッチを fp32 で持ち、差分の相対誤差を `cross/subtract_rel_err` として記録する）、
負部分は平均後に取る（レビュー 5）、欠測は欠測のまま（レビュー 4）。

### 8.5 順序

1. **head proxy の直接検定**（§8.3、backward 0 本）。これで別案が生き返る可能性がある。
2. 生き返らなければ §8.4 の 1 本 backward + hook。
3. どちらでも制御構造（§1 の (a)(b) 制約と QP）は変わらない。**測定の選択は制御の形に影響しない。**
