# 押し戻し制御へのタスク間干渉の組み込み: 現行の振る舞いを保つ最小拡張

状態: **提案のみ。実装なし。** 対象は `claude/opd-per-task-coef` の押し戻し制御（`f7333cb`、
[opd_pushback_control_review.md](opd_pushback_control_review.md) で検討済み）。
数値は `~/grad_probe/terms_n8_fixed.json`（OPD+GRPO control step 300、$N=8$、パラメータ空間）と、
pushback run の物理メモリ実測（[opd_grpo_speedup_proposal_2026_09.md](opd_grpo_speedup_proposal_2026_09.md) §2）。

問い: 現行の押し戻し制御は各タスク自身の報酬と自身の教師の釣り合いしか見ておらず、単タスク学習でも成り立つ。
静的係数版が持っていたタスク間の情報（$C_{ij},\ i\ne j$）を、現行の振る舞いを大きく変えずに戻せるか。

---

## 0. 結論

1. **構造的にはできる。** 現行の制約「教師 $i$ の押し戻し $\le \varepsilon\times$ タスク $i$ の報酬降下」を
   「教師 $j$ の押し戻し（**全タスク**の報酬に対する）$\le \varepsilon\times$ **全タスク**の報酬降下」に広げる。
   対角（自タスク）は現行の token 単位ゲート（ロジット空間）のまま、非対角（他タスク）は step 単位でパラメータ空間から測り、
   同じ閉形式で保持率 $a_j^{\times}$ を作ってタスク $j$ の OPD 項に一様に掛ける。閉形式・EMA・1 step 遅れ・4 状態・checkpoint 保存は現行のまま（§2、§4）。
2. **非対角は token 単位では測れない**（Jacobian を運べない。設計書 §5 のとおり）。update の backward から取る:
   task-pure な micro-batch と勾配の差分で $g_i\approx r_i$、$K$ 本に 1 本の micro-batch で OPD 項だけの `autograd.grad` を取って $d_j$。
   追加費用は backward の $1/K$ とメモリ +7〜14 GB で、update 中の物理余裕（37〜50 GB）の内側（§3）。
3. **振る舞いは cap が拘束するまで変わらない。そして step 300 の作動点では拘束しない。** 非対角の押し戻し率（教師 $j$ が他タスクの報酬降下を打ち消す割合）は
   webshop で 0.3%（search の雑音の多い $R$ を分母から外しても 1.3%）、alfworld 0.08%、search 0.03%。$\varepsilon_\times=0.1$ の 30 倍下にある（§5）。
   **多タスク性は機構の構造として入るが、この regime での効果は測定上ほぼゼロである。** 早期（step ≤ 60）は未測定で、
   §3 の測定はそのまま「学習中に干渉が伸びるか」の時系列になる。**先に測定だけ入れる（Tier 1）ことを勧める。**
4. **token 単位の非対角ゲートは作れない。** 必要な量 $u_{D,j,s}\cdot J_s r_i$ は Jacobian-vector 積で、
   この repo の帰属機構（$\varepsilon$ 摂動）は線形域に入らなかった（理論文書 §4.16）。非対角の粒度はタスクが正直な上限（§6）。
5. **代替の多タスク結合が 2 つある**が、どちらも「干渉制御」ではない（§6）: (a) 保持率の予算結合（配分）、
   (b) OPD–OPD 衝突（search↔webshop、cos² 1〜3% で最大の交差構造）への cap — 信頼信号無しには根拠が無い。

---

## 1. 現行の構造のどこが単タスクか

| 量 | 現行 | 何を見ているか |
|---|---|---|
| $u_{R,t},\ u_{D,t}$ | 同じ token のロジット | タスク $i$ の報酬 vs タスク $i$ の教師、同じ状態で |
| $R_i=\sum_t\|u_{R,t}\|^2$、$C_i^-=\sum_t[-u_R^\top u_D]_+$ | タスク $i$ の token だけ | 対角 $C_{ii}$ のロジット空間版 |
| $a_i=\min(1,\varepsilon R_i/C_i^-)$ | タスクごとに独立 | 他タスクは式に出てこない |

タスクが 1 つでも式は同じ形で動く。静的係数版が使っていた非対角 $C_{ij}=\langle r_i,d_j\rangle$（教師 $j$ の引きがタスク $i$ の報酬を通じて害するか）は、
現行では捨てられている。理由は測定粒度で、$C_{ij}$ はパラメータ空間の量であり、token ごとのロジット量からは作れない。

---

## 2. 拡張: 制約の一般化（§9 の追記により、分母は「全タスクの和」ではなくタスクごとの保護制約に置き換える）

教師 $j$ の押し戻しを、自タスクの token 上（ロジット空間）と他タスク（パラメータ空間）に分ける。

$$
\text{自タスク（現行）}:\quad a_j = \min\!\Big(1,\ \frac{\varepsilon\,R_j}{C_j^-}\Big)
\qquad
\text{他タスク（追加）}:\quad a_j^{\times} = \min\!\Big(1,\ \frac{\varepsilon_\times\,\sum_{i}R_i^{(P)}}{\sum_{i\ne j}\big[-\langle g_i,\,d_j\rangle\big]_+}\Big)
$$

* $R_i^{(P)}=\|g_i\|^2$、$g_i$ はタスク $i$ の行が作る mini-batch 勾配（RL+OPD。$\|r\|/\|d\|\approx 6$〜11 なので $g_i\approx r_i$）。
* $d_j$ はタスク $j$ の行の OPD 項だけの勾配。$\langle g_i,d_j\rangle$ は同じ $\theta_k$（同じ mini-batch）で取る。
* 分母は $\sum_{i\ne j}[-\cdot]_+$ で、静的版と同じく token（ここでは mini-batch）ごとに負部分を取ってから足す。
* 適用: $w_{j,t}\leftarrow w_{j,t}\cdot a_j^{\times}$ をタスク $j$ の**全 token** に掛ける。他タスクへの波及は $d_j$ 全体の性質で token に局在化できないので、一様が正直な粒度。
* 単位: 分子・分母ともパラメータ空間の $\|\cdot\|^2$ 単位で揃う。ロジット空間の $C_j^-/R_j$ と混ぜない。
* $\varepsilon_\times$ は自タスクの $\varepsilon$ と別の knob。意味は「他タスクの報酬降下のうち教師 $j$ が打ち消してよい割合」。

$a_j^{\times}=1$ のとき現行と完全に同じ。cap が拘束したときだけタスク $j$ の OPD 項が縮む。増幅はしない。

---

## 3. 非対角の測定: update の backward から取る

### 3.1 task-pure な micro-batch

`rebalance_minibatch_columns` は各列（60 行 = 1 optimizer step）を token 数で rank に等分し、rank 内は長い順に配る（micro-batch $k$ の形を rank 間で揃えるため。
teacher exchange の collective が micro-batch ごとに走る）。ここを **「列内で task ごとに rank へ等分し、rank 内は task 順に並べ、task 内で長い順」** に変える。

* optimizer step が見る 60 行は変わらない（列の所属は不変）。変わるのは rank 内の順序だけで、重み付き行和は順序不変。
  **数値クラスは加算順序**（`BALANCE_MINIBATCH_COLUMNS` と同じ）。比較する全アームに同じ値。
* micro-batch $k$ の task は rank 間で一致し、token 数も task 内で長い順に配るので近い。境界の micro-batch（task が変わる所）だけ混在し、
  そこは統計から除く（列あたり ≤ 2 本）。
* teacher exchange の形の前提（同じ行数）は固定 micro-batch なので保たれる。

### 3.2 $g_i$: 累積勾配の差分

`no_sync_grad_accum=True` の下で、mini-batch 内の勾配は各 FSDP unit の `flat_param.grad`（unsharded）に累積する。
task 境界で差分を取れば task-pure な区間の勾配和 $g_i^{(k)}$ が出る。必要なのは直前の snapshot 1 本（bf16 3.4 GB）と、
unit ごとの内積を fp64 で累積する集計（`task_gradient_conflict.pairwise_moments` と同じ形）。

読み取りは micro-batch 境界、reduce の前。「`.grad` は最後の mini-batch しか残らない」という取り違えは、mini-batch を跨がないここでは起きない。

### 3.3 $d_j$: 間引いた OPD 項だけの backward

$K$ 本に 1 本の task-pure micro-batch で、学習の `loss.backward()` の前に `torch.autograd.grad(β·L_OPD, flat_params, retain_graph=True)` を取る。
`inputs=` を指定した `autograd.grad` は AccumulateGrad を通らないので FSDP の post-backward hook が発火せず、`.grad` にも足されない
（学習側の勾配を汚さない）。その micro-batch では $r_j = g_j - d_j$ も出るので対角 $\langle r_j,d_j\rangle$ もパラメータ空間で取れる。

費用: backward $1/K$ 本分 + その micro-batch の活性を retain する分。$K=4$ で update +5% 程度、メモリ +3.4〜6.8 GB（$d_j$ の一時保持）。
$K=4$・micro 10 なら step あたり task ごとに 15〜25 本の micro-batch から $d_j$ が出る。1 step では雑音が大きいが、EMA（0.9）で 10 step 分に均す。

### 3.4 統計

step ごとに、mini-batch 内で同じ $\theta_k$ の組だけを足す:

| 量 | 出所 |
|---|---|
| $\langle g_i, d_j\rangle$、$i\ne j$ | $g_i$（3.2、その mini-batch の全 task-$i$ micro-batch）× $d_j$（3.3） |
| $\langle r_j, d_j\rangle$ | 3.3 の micro-batch |
| $\|g_i\|^2$ | 3.2 |
| $\langle d_i, d_j\rangle$ | 3.3 が同じ mini-batch で 2 task 以上に当たったとき |

これは **§4.14 の $N=8$ プローブの online 版**である。6 本の backward と 41 GB の勾配バッファを毎 step 払わずに、
学習の backward から同じ行列を（間引きと EMA の精度で）毎 step 出す。

### 3.5 費用（micro 10、checkpointing on の上に）

| 項目 | 時間 | メモリ（rank あたり） |
|---|---|---|
| task-pure 順序 | 0（rank 間の token 差による待ちが増えうる。列内 task 別等分で抑える） | 0 |
| snapshot と差分・内積 | ≈ 1 s / step | +3.4 GB |
| OPD-only `autograd.grad`（$K=4$） | +5% / update | +3.4〜6.8 GB + retain 分 |
| 合計 | **+5〜7% / update** | **+7〜14 GB**（余裕 37〜50 GB の内側） |

---

## 4. 制御器の拡張

`PushbackController` に task ごとの第 2 状態（`ema_R_cross`、`ema_C_cross`、`a_cross`、`n_obs_cross`、`state_cross`）を足す。

* 4 状態はそのまま: `absent`（その task の行が無い）/ `no_pg`（$\sum_i R_i^{(P)}\le$ tiny）/ `few_groups`（$d_j$ の標本数 < `min_cross_samples`）/ `ok`。
* EMA は和に掛けて比は後（現行と同じ）。1 step 遅れ、step 内固定。
* 保持率の合成: $w_{j,t} = \big(a_j\ \text{if conflict}\ \text{else}\ 1\big)\cdot a_j^{\times}$。
  対角の token ゲートは触らない。$a_j^{\times}$ は `teacher_kl_row_coef` と同じ経路（行ごとの係数）で入るので、損失側の変更は係数の掛け算 1 箇所。
* resume: 第 2 状態も `actor_extra.pushback` に入れ、`eps_cross` の変更は拒否。
* 指標: `pushback/a_cross/{task}`、`pushback/ratio_cross/{task}`、`opd_diag/cross_dot/{i}/{j}`（3×3）、`opd_diag/cross_norm/{task}`、
  `pushback/cross_samples/{task}`。**`a_cross` が 1 のままでも 3×3 の時系列が残る**のがこの拡張の最低限の価値。

---

## 5. 現在の作動点での予測

step 300、$N=8$、パラメータ空間、mean over batches:

| 教師 $j$ | 他タスクへの押し戻し率 $\sum_{i\ne j}[-C_{ij}]_+/\sum_i R_i$（中央値 / 最大） | search の $R$ を分母から除く | 自タスク $[-C_{jj}]_+/R_j$（現行が見る量の対応物） |
|---|---:|---:|---:|
| alfworld | 0.0008 / 0.0018 | 0.0035 / 0.0116 | 0.0038 / 0.0112 |
| search | 0.0003 / 0.0014 | 0.0011 / 0.0097 | 0.0015 / 0.0142 |
| webshop | **0.0034 / 0.0076** | **0.0133 / 0.0312** | 0.0193 / 0.0723 |

* $\varepsilon_\times=0.1$ では**どの教師も拘束しない**（最大でも 30 倍下）。現行の対角 cap が拘束しないのと同じ構造で、
  β=0.01 では教師は報酬の降下の 1〜2% しか打ち消せず、他タスクへは更にその 1/6 である。
* webshop 教師だけは他の 2 教師より 4〜10 倍大きい（静的版・§4.14.3・§3.6 と同じ「webshop 教師が外れる」構造）。
  cap を拘束させるなら $\varepsilon_\times$ を観測比の分位で置く（例: 直近 20 step の webshop の比の 1/2）が、
  それは「他タスクの報酬降下の 10% まで許容」という意味を捨てて「現状の半分にする」という相対強度になる。
* **したがってこの拡張の第一の価値は制御ではなく測定である。** 3×3 の時系列が 300 step 分残れば、
  「早期（OPD が効く区間）に干渉が伸びるか」「webshop 教師の外れ方が学習で変わるか」が初めて分かる。
  拘束する規模の干渉が出たときに、既に cap が入っている、というのが多タスク版の正しい位置づけである。

---

## 6. できないこと・代替

* **token 単位の非対角ゲート。** 必要な量は $u_{D,j,s}\cdot(J_s r_i)$、すなわちタスク $j$ の token $s$ のロジットがタスク $i$ の報酬方向にどう動くか。
  Jacobian-vector 積 1 本で取れるが、この repo の帰属機構は $\varepsilon$ 摂動が bf16 の雑音に埋もれて線形域に入らなかった（理論文書 §4.16）。
  forward-mode の JVP を FSDP 上で通す実装は別件で、しかもロジットは response 部分だけなので prompt 側の Jacobian は落ちる。取らない。
* **OPD–OPD 衝突の cap。** 最大の交差構造は search↔webshop の蒸留どうし（8/8 負、cos² 1〜3%）だが、2 教師が共有パラメータで逆を引くのは
  MOPD の目的関数が要求する妥協で、どちらを縮めるかの根拠は報酬水準の信頼（trust_vs_spillover §2.2）無しには無い。§3.4 の $\langle d_i,d_j\rangle$ で**測るだけ**にする。
* **予算結合。** $\sum_j q_j a_j=$ 一定の再配分を online にすれば保持率がタスク間で結合し、機構は多タスクになるが、干渉制御ではなく配分である。
  現行が「増幅しない」を設計判断にしているのとも衝突する。採らない。
* **finite-difference による因果推定。** 係数を ±10% 揺らして各タスクの return を回帰する（SPSA 型）は唯一の因果的な交差効果推定だが、
  return の step 雑音（SD 0.10〜0.15）に対して 300 step では検出力が無い。

---

## 7. 実装の触る場所（go の後）

| 箇所 | 変更 |
|---|---|
| `verl/utils/seqlen_balancing.py::rebalance_minibatch_columns` | 列内を task 別に rank へ等分し、rank 内を task 順・task 内長い順に。境界 micro-batch の task 混在を `meta_info` で印 |
| `verl/workers/actor/dp_actor.py` | micro-batch 境界で `flat_param.grad` の差分と内積（unit ごと fp64）。$K$ 本に 1 本で OPD 項の `autograd.grad`。step 末に controller へ |
| `verl/trainer/ppo/opd_pushback.py` | 第 2 状態、`a_cross` の閉形式、EMA、4 状態、state_dict |
| `verl/trainer/ppo/opd_task_diag.py` | 3×3 の列を追加（reduce は既存の 1 回に相乗り） |
| `main_opd.py` | `pushback_control.eps_cross`、`cross_every`（$K$）、`min_cross_samples` の注入と検証 |
| 試験 | (i) task-pure 順序で列の所属と 60 行が不変、(ii) 差分が micro-batch 勾配と一致（小モデル）、(iii) `autograd.grad` が `.grad` を汚さない、(iv) $a^\times=1$ で現行と bit-identical、(v) 閉形式と 4 状態 |

段階: **Tier 1** = 測定のみ（$a^\times\equiv1$ を固定、3×3 を毎 step 出す）。**Tier 2** = cap を有効化。Tier 1 の数値を見てから $\varepsilon_\times$ を決める。

---

## 8. 限界

* $g_i\approx r_i$ の近似は $\|r\|/\|d\|\ge 6$ に依る。$\langle g_i,d_j\rangle=\langle r_i,d_j\rangle+\langle d_i,d_j\rangle$ で、第 2 項は step 300 で第 1 項の 10〜15%。
  厳密にしたければ $d_i$ も同じ mini-batch で取る（費用 2 倍）。
* $d_j$ は $K$ 本に 1 本の間引き。1 step の推定は雑音が大きく、cap は EMA 後の値で動く（現行と同じ 1 step 遅れの上に EMA 10 step）。
* task-pure 順序は rank 間の token 差を増やしうる。列内 task 別等分で抑えるが、`microbatch_wait_frac` で確認する（現行 0.011）。
* 非対角の粒度はタスク。教師 $j$ の OPD 項を一様に縮めるので、効率は静的版と同じ（波及 1% を消すのに蒸留を大きく削る）。
  この拡張が「多タスク用の機構」として機能するのは、干渉が cap に届く regime に限られ、step 300 の測定はそこにいない。

---

## 9. 追記: 出力層（lm_head）勾配を交差効果のプロキシにする案の検証

別案: token $t$ の出力層勾配 $G_t=u_t h_t^\top$ を使い、$c_{i,j,t}=\langle G_i^{RL},G_{j,t}^{OPD}\rangle_F=\sum_{s\in i}(u_{R,s}^\top u_{D,t})(h_s^\top h_t)$ で
**token 単位**の交差効果を追加 forward/backward 無しで取り、他タスクを害する token だけに $b_j$ を掛け、タスクごとの保護制約
$\sum_{j\ne i}b_jC^-_{ij}\le\varepsilon_{\rm cross}R_i^{\rm head}$ の QP で $b$ を決める。

設計としては本稿 §2 より細かく（token 選択）、安く（追加 backward 無し）、振る舞いも保つ。**問題は測定の妥当性**で、
出力層は共有パラメータの一部に過ぎない。N=8 プローブは unit 別（`root` = 埋め込み + lm_head、tied、および 28 層）の内積を持っているので、
`root` の交差内積がモデル全体の交差内積と同じ符号を持つかを直接検定できる。

| 対（RL $i$ \| OPD $j$） | 符号一致 root vs 全体（8 バッチ） | root/全体（中央値） | cos root | cos 全体 |
|---|:---:|---:|---:|---:|
| alfworld \| webshop（最強の交差対、全体 7/8 負） | **4/8** | −0.003 | −0.020 | −0.090 |
| search \| webshop | 4/8 | −0.002 | −0.011 | −0.029 |
| alfworld \| search | 5/8 | 0.015 | +0.021 | +0.039 |
| search \| alfworld | 6/8 | 0.035 | −0.003 | −0.009 |
| webshop \| alfworld | 4/8 | −0.001 | +0.001 | −0.012 |
| webshop \| search | **2/8** | −0.004 | +0.002 | −0.004 |
| （対角）alfworld \| alfworld | 8/8 | 0.021 | −0.027 | −0.036 |
| （対角）webshop \| webshop | 5/8 | 0.007 | −0.069 | −0.112 |

* `root` の $\|g\|^2$ シェアは 2.5〜7%。交差対では **root は全体の 0.1〜3.5% しか運ばず、符号は偶然の水準（4/8 前後）**。
  最強の対（alfworld:RL \| webshop:OPD）で root は 4/8、大きさは全体の 0.3% で符号が逆。
* 層別に見ても、同じ対で 28 層のどれも全体の符号を安定に追わない（一致 5〜8/8、最良 3 層が 8/8 だが 29 unit の多重比較）。
  交差内積は多数の層の小さい寄与の和で、**どの 1 ブロックも代理にならない**。
* `root` には tied 埋め込みの入力側勾配も混ざる。別案の $G=u h^\top$ は出力側だけなので厳密には同じ量ではないが、
  出力側は root の一部であり、しかも $h_s^\top h_t$ は書式 token 対（位置の 84〜92%、理論文書 §4.8）に支配されるので、
  出力側だけが全体を追うと期待する理由は無い。
* 対角（自タスク）では root の一致は良い（alfworld 8/8）が、それは現行ゲートが既にロジット空間で扱っている量である。
* OPD–OPD では alfworld\|webshop 8/8、search\|webshop 7/8 と一致する（交差効果の対象ではない）。

**結論。** 出力層プロキシで選ぶ token 集合 $H_j$ は、真の交差干渉に対して偶然と区別できない。
token 単位の交差ゲートは「Jacobian を運べない」という設計書の限界を出力層 1 ブロックで回避しようとしたものだが、
データはそのブロックが代表的でないと言っている。**採らない。**

**採るべき組み合わせ。** 別案の**制御構造**は本稿 §2 より良い: 保護制約をタスクごとに置く
（$\sum_{j\ne i}b_jC^-_{ij}\le\varepsilon_{\times}R_i$、各 $i$）ことで、alfworld の改善で webshop の悪化を埋めるような平均化を避けられる。
本稿 §2 の分母 $\sum_iR_i$ をこの形に置き換える。測定は §3 のパラメータ空間（task-pure micro-batch + 勾配差分 + 間引いた OPD-only
`autograd.grad`）で行い、$b_j$ はタスク $j$ の OPD 項に一様に掛ける。token 選択は現状では成立しない。
どちらの案でも step 300 の作動点では拘束しない（§5）。

---

## 10. 追記: 走行中の run（ε=0.003）が示す、この拡張の実効性

自タスク cap は ε=0.003 で作動しており（[opd_pushback_control_review.md](opd_pushback_control_review.md) §8）、
step 1〜8 で alfworld と search の対立 token を a ≈ 0.70〜0.81 に落としている。**webshop だけは C⁻/R = 0.002 < ε で無傷である。**

§5 の「ε_×=0.1 では拘束しない」は、自タスク側と同じく ε_× を作動点に相対的に置けば成り立たない。パラメータ空間の交差押し戻し率は
webshop 0.34%（search の R を除いて 1.3%）、alfworld 0.08%、search 0.03% で、**同じ相対水準の ε_× なら webshop の教師だけが cap にかかる**。
つまりこの拡張は、自タスク基準では無傷の教師を、他タスクの報酬を根拠に縮める。cross-task の 3 つの独立な測定（§4.14.3、§3.6、C の webshop 列）が
指していた教師と一致しており、「多タスク性が構造だけ」ではなく振る舞いに出る。

留保: 自タスク cap の比はロジット空間、交差 cap の比はパラメータ空間で、単位も checkpoint も違う。ε_× を「自タスクの ε と同じ数」にはできない
（別案の指摘どおり）。決め方は Tier 1 の 3×3 時系列の分位で、拘束の強さは事前に選ぶ実験条件になる。
