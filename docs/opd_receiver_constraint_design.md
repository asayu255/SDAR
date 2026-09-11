# 案: 受け手別の一次制約で OPD 量を決める機構（S-LP）

状態: **提案。未実装。** 現行の cross gate（MOPD v2）を引き継がない独立の機構。
数値は `~/grad_probe/terms_n8_fixed.json`（OPD+GRPO control、step 300、$N=8$ バッチ、
`pg_loss_coef=1.0`, `teacher_kl_loss_coef=0.01`）から再計算した。再現は
`/tmp/.../lp_Si.py`, `lp_v2.py`。関連: [opd_coefficient_cross_effect_review.md](opd_coefficient_cross_effect_review.md)（以下「検討」）、
[opd_coefficient_trust_vs_spillover.md](opd_coefficient_trust_vs_spillover.md)。

---

## 0. 前案からの訂正

会話中の前案は制約を $R_i + \sum_j \beta_j C_{ij} \ge \tau[R_i]_+$ と置いていた。
これは「**OPD が RL の一次進行を反転させない上限**」であって、指示された「正の貢献」ではない。
正しい量は受け手 $i$ が受け取る**交差効果そのもの**

$$S_i = \sum_j \beta_j C_{ij} \ \ge\ 0 .$$

この 1 項の差で機構の性質が正反対になる。

| | 前案（$R_i$ 込み） | 本案（$S_i$ のみ） |
|---|---|---|
| step 300 データでの拘束 | **一度も拘束しない**（余裕 12〜27 倍） | **ほぼ常に拘束**。$\varepsilon=0$ では 8 バッチ中 6 で $\beta=0$ |
| 実体 | 上限だけを決める安全装置（= control と同一） | OPD 量そのものを決める。緩和項が要る |

---

## 1. 量の定義

1 step（= 45 prompt × 8 rollout = 360 行、optimizer step 6 回）について

| 記号 | 定義 |
|---|---|
| $r_i$ | タスク $i$ の GRPO 損失の勾配（`pg_loss_coef`・`task_loss_weight` 込み。`use_kl_loss=False` なので ref-KL は無い） |
| $d_j$ | タスク $j$ の OPD（teacher$_j$ top-20 KL）勾配、**係数 1** のもの。現行は $\beta_0 = 0.01$ を掛けて足している |
| $\beta_j$ | 決めたい量。$\beta_0$ の**倍数**で表す。control は $\beta = (1,1,1)$ |
| $C_{ij}$ | $\langle r_i,\ \beta_0 d_j\rangle$。送り手 $j$ の OPD が受け手 $i$ の RL 目的を一次で動かす量 |
| $S_i$ | $\sum_j \beta_j C_{ij}$。受け手 $i$ が受けている交差効果の総量（**自タスク $j=i$ を含む 3 項**） |
| $R_i$ | $\langle r_i,\ r_{\rm tot}\rangle$、$r_{\rm tot}=\sum_k r_k$。受け手 $i$ 自身の RL 一次進行 |

対角 $C_{ii}$ は同一の行から計算すると $\mathrm{Cov}(r_i,d_i)$ が混入する（検討 §4.3）。
prompt 単位で交わらない半分割 $A,B$ を取り $C_{ii} = \tfrac12(\langle r_i^A, d_i^B\rangle + \langle r_i^B, d_i^A\rangle)$ とする。

計量。素の内積（Euclid）を本線とし、Adam 計量 $\langle a,b\rangle_P=\sum a_kb_k/(\sqrt{\hat v_k}+\epsilon)$ は併記のみ。
理由は §4.4 の信頼度: Adam 計量では search の同一タスク半分割信頼度が **0.001**（素の内積では 0.279）で、
$C$ の推定に 1000 step 規模の平均が要る。実際に重みへ届くのは Adam 側なので、読み値としては残す。

---

## 2. 決定則

$$
\max_{\beta}\ \sum_j \|d_j\|\,\beta_j \;-\; \frac{\mu}{2}\big\|\beta-\beta^{(s-1)}\big\|^2
\qquad\text{s.t.}\qquad
\sum_j \bar C_{ij}\,\beta_j \ \ge\ -\varepsilon\,\bar R_i \quad(\forall i),\qquad
0 \le \beta_j \le \beta_{\max}
$$

* $\bar C,\bar R$ は EMA（decay 0.8）。生の 1 step 値で解くと bang-bang になる（バッチ間の cos の標準偏差が平均と同程度）。
* $\varepsilon$ が**唯一の意味のあるつまみ**。$\varepsilon=0$ が指示どおりの「正の貢献」で、$\varepsilon>0$ は
  「3 本の OPD 合計で受け手自身の一次進行の $\varepsilon$ 倍までは食ってよい」。pushback アームの `eps=0.003` と同じ形・同じ尺度。
* 目的は**線形和** $\sum_j\|d_j\|\beta_j$（設計文書 §1.2 の 2 つの不変量のうち、任意の $L_k$ への一次効果の上限を決める方）。
* **平滑化は近接罰則ではなく硬いトラスト領域** $\|\beta-\beta^{(s-1)}\|_\infty \le \delta$ で行う。
  罰則 $\tfrac\mu2\|\beta-\beta^{(s-1)}\|^2$ は尺度が合わない: 目的の値域が $10^{-2}$ 台なのに
  $\|\beta-\mathbf 1\|^2$ は $O(1)$ なので、$\mu=0.3$ で解はほぼ control、$\mu\ge3$ で厳密に
  control に潰れ、$\mu\to0$ では平滑化しない（実測、§3.2 の行列）。中間が無い。
  トラスト領域なら単位が「1 step で $\beta$ が動ける量」で解釈でき、雑音の 1 step が与えうる
  被害も $\delta$ で抑えられる。$\delta=0.05$ で最適点まで 25 step、$\delta=0.2$ で 6 step。
* 3 変数なので頂点列挙（$2^3$ 個の有効集合）で厳密に解け、KKT 残差で証明書を出せる。
  現行 `opd_cross_gate.solve_role` と同じ作り。証明できない解は $\beta=(1,1,1)$（= control）に落とす。
* 適用は 1 step 遅れ。`teacher_kl_loss_coef_by_task` にそのまま入る（既存経路、実装済み）。

---

## 3. 実データが言うこと

$C_{ij}/R_i$（バッチ平均、行 = 受け手、列 = OPD 送り手）:

| 受け手 \ 送り手 | alfworld | search | webshop |
|---|---:|---:|---:|
| alfworld | −0.0001 | **+0.0083** | −0.0111 |
| search | −0.0006 | −0.0004 | −0.0015 |
| webshop | +0.0011 | +0.0050 | **−0.0154** |

**列の符号がこの機構の全内容である。** webshop の教師は 3 つの受け手すべてに負、
alfworld の教師はほぼ 0、search の教師は 3 つ中 2 つに正。

### 3.1 $\varepsilon=0$ は退化する（構造的）

$\beta\ge 0$ で全行 $S_i\ge0$ を満たすのは、$C$ のある行が全成分負なら $\beta=0$ しかない。
平均行列では search 行がそれ（−0.0006, −0.0004, −0.0015）。バッチ単位でも 8 中 5 で
全成分負の行が 1 本出る。結果:

| $\varepsilon$ | EMA 行列での $\beta$ | バッチ平均 $\beta$ | $\beta=0$ の割合 |
|---:|---|---|---:|
| 0 | **(0, 0, 0)** | (0.75, 0.38, 0.22) | **0.75** |
| 0.001 | (0, 2.72, 0) | (0.84, 0.59, 0.26) | 0.00 |
| 0.003 | (3, 3, 0.10) | (1.04, 0.96, 0.36) | 0.00 |
| 0.010 | (3, 3, 1.83) | (1.50, 1.52, 0.75) | 0.00 |
| 0.030 | (3, 3, 3) | (2.21, 2.65, 1.63) | 0.00 |

（$\beta_{\max}=3$。$\beta$ は $\beta_0=0.01$ の倍数。）

$\varepsilon=0$ の機構は「**OPD を全部切る**」であり、これは偶然ではない。検討 §4.1 の命題により、
結合目的の停留点では $\sum_{ij}C_{ij} = -\|\sum_i r_i\|^2 \le 0$ で、学習が進むほど確実に成り立つ。
正則化項が「データ項の行きたい方向に逆らう」のは正則化の定義であって、切る理由ではない。
**したがって $\varepsilon>0$ は緩和ではなく、機構が内容を持つための必要条件である。**

### 3.2 予算固定版（交絡の少ない対照）

$\varepsilon$ を選ぶ代わりに OPD の**総量を control と同じに固定**し、配分だけを決める:

$$\max_\beta\ \min_i \frac{S_i}{R_i}\qquad\text{s.t.}\qquad \sum_j\|d_j\|\beta_j=\sum_j\|d_j\|,\quad 0\le\beta_j\le\beta_{\max}$$

EMA 行列で $\beta=(0,\ 1.98,\ 0)$、最悪の受け手の余裕が $-0.94\% \to -0.073\%$（**13 倍改善**）。
バッチ平均 $\beta=(1.10,\ 1.23,\ 0.33)$。総正則化量が control と等しいので、
差が出たときに「OPD を増減した効果」と「配分を変えた効果」が混ざらない。

**この 2 つを取り違えてはならない。** 機構が実際に出すのは EMA 行列を解いた $(0.01,\ 1.97,\ 0)$ で、
alfworld と webshop の OPD を**完全に切って** search に全量を回す。静的 redistribute アーム
$(1.076,\ 1.191,\ 0.50)$ よりはるかに攻撃的である。バッチごとの解の平均 $(1.10, 1.23, 0.33)$ が
redistribute に似て見えるのは、頂点解を平均したからであって、制御器が適用する値ではない。
共通するのは順序（webshop が最小、search が最大）だけで、量は別物である。

トラスト領域 $\delta$ はこの攻撃性を時間方向に薄める。$\delta=0.05$ なら 25 step かけて
$(1,1,1)$ から $(0.01,1.97,0)$ へ動き、途中で行列が変われば向きも変わる。

### 3.3 bootstrap: 再現するのは 1 つの主張だけ

8 バッチの bootstrap（2000 回、行列を再標本化してから解く）。既存設計文書の go/no-go は
「順位が 90% 再現しなければ走らせない」である。

| 規則 | 点推定 $\beta$ | alfworld 5–95% | search 5–95% | webshop 5–95% | $P(\beta_{\rm web}<1)$ |
|---|---|---|---|---|---:|
| $\varepsilon=0.003$ | (3, 3, 0.10) | [0.00, 3.00] | [0.18, 3.00] | [0.00, 1.83] | 0.772 |
| $\varepsilon=0.010$ | (3, 3, 1.82) | [0.80, 3.00] | [1.41, 3.00] | [0.00, 2.49] | 0.329 |
| **予算固定** | (0, 1.98, 0) | [0.00, 3.00] | [0.34, 2.00] | **[0.00, 0.42]** | **0.972** |

読み方は 3 つ。

1. **$\beta$ の値そのものは再現しない。** どの規則でも 5–95% 帯が box をほぼ覆う。
   LP は頂点解なので、行列がわずかに動くだけで有効制約集合が入れ替わる。
   これは $\mu$（近接項）と長い EMA が任意の飾りではなく**必須**であることの根拠である。
2. **90% の基準を通るのは予算固定版の 1 主張だけ**: 「webshop の OPD は control より小さい」が 97.2%。
   $\varepsilon$ 版の同じ主張は 77.2% / 32.9% で基準未満。$\varepsilon=0.01$ に至っては
   webshop がむしろ control 以上になる方が多い。
3. **したがって $\varepsilon$ 版を単独で走らせる根拠は現時点では無い。** §3.2 の予算固定版が
   唯一 go/no-go を通る形であり、その内容は「webshop の教師への配分を下げ、search へ回す」
   という静的 redistribute アームと同じ結論である。

これは step 300・$N=8$ の 1 点での話で、適用区間（step $\le$ 60）の測定ではない。
$N$ を増やすか早期 checkpoint で測り直せば帯は縮む。§5 手順 4 の観測専用モードが
その両方を学習を変えずに与える。

---

## 4. 追加 forward / backward なしの測定

### 4.1 無料で取れるもの

micro-batch ごとの勾配は、**accumulate される `.grad` を micro-batch 境界で差分するだけ**で取れる。
追加 FLOP はゼロ。バッチ幾何がちょうど合う:

| | 値 |
|---|---|
| 1 step の行数 | 45 prompt × 8 = 360 |
| `ppo_mini_batch_size` 60、`rollout.n`=1、2 rank | rank あたり 30 行 = optimizer step 6 回/step |
| `ppo_micro_batch_size_per_gpu` 10 | optimizer step あたり rank あたり **3 micro-batch** |
| タスクあたりの行数 | 120/step、rank あたり 60、optimizer step あたり **ちょうど 10 行 = 1 micro-batch** |

つまり **micro-batch をタスク純粋にすると 3 タスクが 3 スロットに過不足なく収まる**。
`normalize_loss_by_task=True` の下では損失集約が step 全体の重みによる純粋な加重和なので、
**並べ替えても適用される勾配は変わらない**。確認済み:
`agg_loss_by_task_weights` は `((loss_mat*loss_mask).sum(-1)*weights).sum()`（`core_algos.py:497`）、
teacher-KL 側も `(row_kl*row_w).sum()` で、どちらも行の和である。
`use_dynamic_bsz=False` なので最後の `/gradient_accumulation` は定数、`entropy_coeff=0`、
`use_kl_loss=False`、invalid-action penalty は actor の集約経路に入らない。

**ただしこれは既存の長さ均衡と競合する。** `_balance_batch` は
`rebalance_minibatch_columns` で「mini-batch $k$ と micro-batch 位置を**rank 間で揃えたまま**
長さを均す」ことをしている（`ray_trainer.py:1597`、理由は teacher lookup と勾配 reduce で
rank が出会う位置がそこだから）。効果は測定済みで `global_seqlen/microbatch_wait_frac` は 0.006。
タスクをスロットに固定すると、この均衡器が使っていた自由度が消える。

回避は単純で、**タスクを外側の鍵にし、長さ均衡を各タスクの行の中で rank 間に掛ける**。
両ランクが同じスロットで同じタスクを処理するので同期は保たれ、
タスク内の行は長さが揃っている（prompt 上限が alfworld 2048 / search・webshop 4096 と別）ので
micro-batch 内の padding はむしろ減る可能性がある。実装の主リスクはここで、
`microbatch_wait_frac` を回帰指標として見る。

### 4.2 無料にできない部分

1 回の backward は 1 個の VJP しか計算しない。損失は logits の位置で
$g_z = g_z^{\rm RL} + \beta g_z^{\rm OPD}$ と**足された後**に幹へ入るので、
$J^\top g_z^{\rm RL}$ と $J^\top g_z^{\rm OPD}$ を分けるには幹を 2 回通す以外にない。
gradient checkpointing 有りで 2 本目は再計算 1 + backward 2 = 初回の 4 に対して +3、**+75%**。

**この +75% は FLOP 比からの理論値で、この workload では未測定である。** worker 側の
段階プロファイラ（`_actor_phase`、`actor.fwd` / `actor.bwd`）は実装済みだが
k100 run では有効になっておらず（`GPU_PROFILER_SYNC_PHASES` と rank 0 の profiler が必要）、
ログに forward/backward の内訳が無い。§5 手順 4 の観測 run ではこれを有効にして
実測値に置き換える。teacher lookup・sign_weight・cross_teacher_target など
backward 以外の段階が `update_actor` の 155 s のうち相当を占めている可能性があり、
その場合 2 本目の相対コストは 75% より小さい。

$\Rightarrow$ **「厳密 かつ 無料」は両立しない。** 選べるのは「不偏 かつ 無料」か「厳密 かつ 有料」。

### 4.3 行分割による回避（無料側）

タスク $i$ の prompt を交わらない 2 群 $A,B$ に分け、**項を群に割り当てる**:

* 群 $A$: $\dfrac{|{\rm all}|}{|A|}\cdot \text{pg}_A$ のみ（RL 項だけ）
* 群 $B$: $\dfrac{|{\rm all}|}{|B|}\cdot \beta_0 D_B$ のみ（OPD 項だけ）

和の期待値は $r_i+\beta_0 d_i$ で**不偏**。各群は項純粋なので、1 回ずつの backward（元から必要な回数）と
`.grad` 差分だけで $\hat r_i,\hat d_i$ が**厳密に分離して**取れる。$A,B$ が prompt で交わらないので
対角の共分散混入も同時に消える。追加 FLOP ゼロ。

**唯一のコストは適用される勾配の分散**である。50/50 分割で各項の分散は 2 倍。
これは不偏だが無害ではない（§4.4）。probe 比率 $f$（分割する prompt の割合）と
周期 $N$（何 step に 1 回測るか）で連続的に落とせる。

### 4.4 コストの大きさ

**clip 圧が主因である。** `grad_clip=1.0` に対して k100 run（step 1–150）の `grad_norm` は

| | 値 |
|---|---:|
| 平均 / 中央値 | 1.459 / 1.242 |
| step 間の変動係数 CV | **0.646** |
| clip される step の割合 | **0.787** |
| clip された step の平均縮小率 | 0.720 |
| step 101–150 に限ると | 平均 0.989、clip 割合 0.500 |

CV 0.646 は勾配が step ごとに大きく揺れていることの**直接の観測**である。
分割で分散を足すとノルムが上がり、上がった分は clip で削られて実効学習率がそのまま下がる。
clip 圧は学習が進むと下がる（step 150 で半減）ので、コストは序盤に集中する。

**信頼度の直接測定は無い。** 手元の同一タスク半分割信頼度は
`~/grad_probe/halves_step150_fixed.json` にあるが、これは
`verl_agent_opd_multitask/global_step_150`（recipe `examples/opd_trainer/run_multitask_qwen3.sh`、
`pg_loss_coef=0` の**純 OPD** run）で取ったものである。したがってこの $\rho$ は
**OPD 勾配の信頼度であって RL 勾配のものではない**。分割の代償は $\|r\|\approx 9\|\beta_0 d\|$ より
RL 項が支配するので、必要なのは測っていない方である。参考値として:

| 計量 | alfworld | search | webshop |
|---|---:|---:|---:|
| 素の内積（**OPD 勾配**、純 OPD run） | 0.940 | 0.279 | 0.885 |
| Adam 前処理後（同上） | 0.502 | 0.001 | 0.348 |

この $\rho$ をそのまま当てはめれば、雑音分散 2 倍で勾配ノルムは素で ×1.03〜1.31、
Adam 側で ×1.22〜1.41、$f=1,N=1$ で実効学習率 −18〜−29%、$f=0.5,N=3$ で平均 −2% となる。
**ただしこの見積もりは借り物の $\rho$ に依存しており、上にも下にも外れうる。**
§5 手順 4 の観測専用モードは、$C$ と同時にこの $\rho$ を正しい checkpoint で測る。
$f$ と $N$ はその測定の後に決めるべきで、事前に固定してはならない。

Adam 側で search の $\rho$ が 0.001 なのは別の含意を持つ。Adam 計量での $C_{ij}$ は
search 行について 1000 step 規模の平均を要するということで、§1 が本線を素の内積に置く根拠である。

### 4.5 選択肢

| | 追加 FLOP | 適用勾配への影響 | $C_{ij}$ の質 |
|---|---|---|---|
| **A. 行分割**（§4.3、$f=0.5,N=3$） | **0** | 実効 LR −約 2% | 不偏・厳密（分散は EMA で潰す） |
| B. head NTK（`root` unit のみ、token 部分抽出） | ~0 | **なし** | proxy。9 セルの符号は全一致、相関 0.785、絶対値は 3〜7 倍ずれる |
| C. 2 本目の backward を probe 部分にだけ | $+0.75f/N$（$f{=}0.5,N{=}3$ で step +4.5%） | **なし** | 厳密。分散も最小 |
| D. タスク純粋 micro-batch のみ（項分離しない） | **0** | **なし** | $G_{ij}=\langle g_i,g_j\rangle$ のみ。$\langle r_i,r_j\rangle$ が 81 倍大きく交差項を取り出せない |

指示（追加 forward/backward なし）に沿うのは **A**。C は指示を 1 つ外す代わりに A の唯一の欠点を消す。
コストの大きさは A の −2%（実効 LR）と C の +4.5%（wall clock）で同程度。

### 4.6 通信とメモリ

* 差分を取るには micro-batch ごとに勾配を reduce する必要があるので、測定 step では `no_sync_grad_accum` を切る。
  reduce-scatter が optimizer step あたり 1 → 3 回、all-gather が 6 → 18 回/step。合計 +約 2.4 s/step（430 s に対して 0.6%）。
* shard された勾配は rank あたり 3.4 GB (fp32)。6 群で 20 GB を CPU の pinned buffer に置く（空き 173 GB）。
* Gram は 21 個の内積。shard が rank 間で交わらないので rank 内で計算してスカラー 21 個だけ all-reduce。
* D2H 転送は 6 群 × 6 optimizer step × 3.4 GB = 122 GB/step $\approx$ 5 s。測定 step だけなので $N=3$ で 1.7 s/step。

---

## 5. 実装計画

### 5.0 判断は 1 つに収束している

§3.3 の bootstrap の後、選ぶべきものは 3 つではなく 1 つになった。

| 元の問い | 現時点の答え | 根拠 |
|---|---|---|
| 予算固定版か $\varepsilon$ 版か | **予算固定版** | go/no-go（90% 再現）を通るのは予算固定版の 1 主張だけ（97.2%）。$\varepsilon$ 版は 77.2% / 32.9%（§3.3） |
| $\varepsilon$ の値 | 不要 | 予算固定版に $\varepsilon$ は無い |
| 測定は行分割か 2 本目の backward か | **今決めない** | $f,N$ を決めるのに必要な RL 勾配の信頼度が未測定（§4.4）。手順 4 の観測で測ってから決める |

したがって**次の一手は手順 4（観測専用モード）だけ**であり、それは学習を一切変えない。

**GPU の空きが律速。** tamago は k100 の step 300 まで塞がっている（本日 19 時頃まで）。
wasabi は別セッションのジョブが 3 枚を占有中。観測 run はそれ以降。

### 5.1 手順

再利用: ブランチ `claude/cross-teacher-target-6b2123` の
`verl/trainer/ppo/task_gradient_conflict.py`（FSDP unit 別の勾配捕捉、Adam 前処理、
pairwise moments の all-reduce、prompt 群単位の半分割）と `grad_probe_driver.py` の
`partition_by_task_and_half`（等サイズ・`world_size × micro_batch_size` 割り切れ・
`task_loss_weight` 保持）。測定側の大部分がすでにある。

新規:

1. `verl/trainer/ppo/opd_receiver_lp.py` — $\bar C,\bar R$ の EMA、頂点列挙 LP、KKT 証明書、
   検証ゲート（タスクあたり最小 prompt 数 / 最小 token 数 / staleness）、`state_dict`。
2. `dp_actor.update_policy` に測定モード: 測定 step のみ micro-batch を (task, term) 純粋に並べ替え、
   `no_sync` を切り、境界で `.grad` を差分して CPU バッファに加算。step 末に Gram → コントローラ。
3. 出力 $\beta_j$ を既存の `teacher_kl_loss_coef_by_task` 経路へ（実装済み経路、変更不要）。
4. **観測専用モード**（$\beta$ 固定、記録のみ）。**最初に走らせるのはこれだけ。**
   control の設定のまま、毎 step $C_{ij}$・$R_i$・反実仮想の $\beta$ 軌跡・
   **RL 勾配の同一タスク半分割信頼度**を記録する。学習は一切変わらないので
   control run そのものとして使える（checkpoint も val も control と比較可能）。
   これが決めるもの: (a) §3 の表が早期 step でも成り立つか、(b) $f$ と $N$、
   (c) $\beta$ が本当に動くのか（動かないなら本 run は不要）。
5. intent lock に新キーを固定、アーム名 `slp`、`expected_multitask_opd_coef_slp_config.yaml`。
6. テスト: 頂点列挙 vs scipy `linprog`、分解恒等式 $\sum_j g_j = r_{\rm tot}+\beta_0\sum_j d_j$、
   半分割の prompt 独立性、$\varepsilon=0$ で全成分負の行があれば $\beta=0$ を返すこと。

ログ: $C_{ij}$（生・Adam・cos）、$R_i$、$S_i$、$\beta_j$、拘束した受け手、
$\|r_i\|$、$\|d_j\|$、半分割信頼度、有効制約集合、KKT 残差、clip 前後の勾配ノルム。

---

## 6. 限界（走らせる前から分かっていること）

1. **一次の交差効果は OPD の主効果を測らない**（検討 §4.2）。本機構が決めるのは
   「RL を害さない安全上限」であって「蒸留として有益な量」ではない。$C_{ij}$ が 0 でも
   教師に寄せること自体の価値（ゼロ次）は別の量である。
2. **$\varepsilon$ は測定から出ない選択である。** §3.1 の通り $\varepsilon=0$ は退化するので、
   $\varepsilon$ が実質的に OPD 総量を決める。「$C$ が総量を決める」とは言えない。
3. **測定点が step 300 のみ。** OPD が効くのは step $\le$ 60 とされる（検討 §4.5）ので、
   §3 の表が早期でも成り立つ保証はない。手順 4 の観測専用モードはこれを潰すためにある。
4. **LP は頂点解**なので、$\varepsilon>0$ では多くのタスクが $\beta_{\max}$ に張り付く。
   そのとき機構の実体は $\beta_{\max}$ であり、k100 で $\lambda_{\max}$ が実体だったのと同じ形になる。
   §3.2 の予算固定版はこれを構造的に避ける。トラスト領域 $\delta$ は張り付きを防がないが、
   張り付くまでの時間を与えるので、その間の val で異常に気づける。
5. **$\beta$ の上げ方向は未検証。** control は $\beta=1$ で、$\beta=3$ まで上げた run は存在しない。
