# Cross-teacher target injection:設計案

前提となる測定は `docs/cross_teacher_kl_weight_offline_audit.md`。本文書はそこで確定した制約の上に、2つの意図を1つの機構として書き下す。

* **意図1** 全教師の policy shift が一致したとき、強く蒸留する。
* **意図2** on-task 教師の shift が小さく、off-task 教師の shift が共通して大きいとき、off-task 教師から学習信号を出す。

---

## 0.0 改訂 (2026-09-02):3点の撤回

**§2(`c` の構成)と §6・§7 は無効化されていない。§3・§3.1〜§3.3・§4 の全数値は無効である。**
以下は初版の走行前に、初版の設計が生む介入量を実測した結果からの撤回である。撤回の理由を残すため、初版の記述は削除せず「(初版)」として残す。

### 撤回1:質量保存の交換 → 単純正規化

$$\tilde p(v) = \frac{p_{on}(v)\,e^{c(v)}}{Z}, \qquad Z = \sum_{v \in S} p_{on}(v)e^{c(v)} + p_{\text{tail}}$$

§3 の容量制限交換($R_D, A_U, T=\min$)を丸ごと削除する。理由は §4 の事前予測そのものである:**上げ側の throttle が $T/A_U = 0.026$** で、意図1のチャネルは要求した介入の 2.6% しか実行しない。実測で top 層の効き平均は **0.058 nats**、旧 KL 加重アームの 58.3 に対し **1/1000** だった。

* **得たもの:** throttle が消え **38倍**。
* **失ったもの:** §3.2 の性質2(頭が動かない)と性質3(符号忠実性)。頭は $1/Z$ で課税され、符号忠実性は「$c>0 \Rightarrow w>1$」から「$c > \log Z \Rightarrow w>1$」に弱まる。$Z=1$ は恒等式ではなくなり、`mass_error` が測るのは「$\tilde p$ が分布であること」に変わる。

**正規化だけでは足りない、という測定も同時に記録する。** 単純正規化での損失の変化は

$$\Delta\mathrm{KL} = \mathrm{KL}(p_s\Vert\tilde p) - \mathrm{KL}(p_s\Vert p_{on}) = \log Z - \langle c\rangle_{p_s}, \qquad |\Delta\mathrm{KL}| \le 2\max|c|$$

で、**正規化の書き方に依らず $c$ の大きさだけで上限が決まる。** 旧アームが $(W-1)\cdot\mathrm{KL}$ という無制限な2量の積だったのとは構造が違う。正規化後の top 層は 2.24 nats(旧アーム比 **1/26**)。

### 撤回2:support を student top-k に戻す(C7 の撤回)

C7 は「目標を生徒非依存の固定点にする」ために teacher top-k を要求した。撤回する:**教師 top-20 の下限は 1e-2〜1e-4** なので、$p_{student}$ が高く $p_{on}$ が極小(中央 2.3e-06)の候補を**構造的に含み得ない**。旧アームで効果の **68.9%** はそこにあった。

* **代償:** C7 が名指しした通りのフィードバックループ。旧 student-indexed アームの `frac_agree_pos` は 0.238→0.193 と漂流した。
* **副次的な利点:** intent lock の差分が**機構キーのみ**になる。ペアリングテストの `student_indexed_topk` 例外を削除した。キャッシュは行あたり 4 モデル読みに戻る(`ppo_micro_batch_size_per_gpu=5` は既にその前提で決まっている)。

### 撤回3:`exponent_scale` 1.0 → 2.148、`_EXPONENT_CLAMP` 30.0 → 3.0

§3.1 で「保守側を採る」とした判断を撤回し、実測の RMS→nats 換算 **2.148** を採る。正規化の 38倍に単位換算の 2.1倍を重ねて top 層 **4.82 nats**、旧アーム比 **1/12**。

クランプは**設計パラメータに変わった**。交換が介入量を自律的に絞っていた間は 30.0 の overflow ガードでよかったが、正規化では他に上限が無い。3.0 は「1候補あたり $e^3\approx20$、1位置あたり $|\Delta\mathrm{KL}| \le 6$ nats」という律速である。`exponent_scale=2.148` では作用候補の $|c|$ 中央が 2.15(クランプの 72%)なので**これは実際に binding する**。`target/clamped_per_step` と `target/clamped_frac_of_acted` は病理の指標ではなく**レート**として読む。

### この改訂で入れた指標

| キー | 意味 |
|---|---|
| `target/abs_dkl_mean`, `target/dkl_mean`, `target/abs_dkl_live_mean` | $\Delta\mathrm{KL} = \log Z - \langle c\rangle_{p_s}$、nats/位置。**旧アームと同じ単位**で、スケールが届いたかを決める数字 |

> **比較の注意。** 上に並べた 58.3 / 0.058 / 2.24 / 4.82 nats は **top 層**(効果上位で選ばれた偏った標本)の数字である。偏りのない spread 層では旧アームの位置あたり平均が 0.274 nats、正規化前の新機構が候補あたり 0.0039 nats で、位置あたりに直すと(20候補、符号相殺込み)0.02〜0.08 と**幅のある見積もり**になる。典型的な位置での差は 4〜14倍に留まる可能性が高い。`abs_dkl_mean` は全位置平均なので **top 層の 58.3 と直接比べてはいけない** — この指標を入れたのは、その見積もりの幅を実測で潰すためである。
| `target/log_z_mean`, `target/max_abs_log_z` | 頭の課税。交換では恒等的に 0 だった |
| `target/clamped_frac_of_acted` | クランプの発火率 |

**削除した指標:** `target/throttle_up`, `target/throttle_down`(交換に属する量)。

### まだ実行していない選択肢

局所 KL に比例させる tilt($c'(v) = c(v)\cdot \mathrm{KL}_{pos}/\langle\mathrm{KL}\rangle$)は**採用していない**。旧アームの $W$ は KL と無相関(Spearman +0.003、監査 §15)なので、これは旧アームの再現ではなく**新しい性質(focal 化)の追加**であり、そう名付けて別に走らせるべきものである。`target/abs_dkl_mean` の実測を見てから判断する。

---

## 0. 設計を縛る、測定済みの制約

| # | 制約 | 出典 |
|---|---|---|
| C1 | **位置スカラー重みでは off-task の知識を注入できない。** $W\cdot \mathrm{KL}(p_s\Vert p_{on})$ の最小解は $W$ に依らず $p_s=p_{on}$ | 監査 P1 |
| C2 | **候補ごとの証拠を位置スカラーに畳むと「一致した候補を強める」が表現できない** | 監査 P2 |
| C3 | **KL 加重 mean-1 正規化はゼロサムを強制し、増幅の原資の半分が教師の最自信トークンから出る**(`norm_offset` が gross shift の 50%) | 監査 P3、旧 target アームの 46% 税 |
| C4 | **deadzone は不要かつ有害。** $\epsilon=0.1$ RMS は到達質量 5.1%・純度 0.16 で試した中で最悪。`min` 型の振幅は自己減衰するので閾値が要らない | 監査 §3 |
| C5 | **`q` 類似度ゲートは実際の一致を予測しない**(四分位で 0.41–0.48 とフラット)。**捨てる** | 監査 P7 |
| C6 | **測度を student mass にするとアンラーニング裾を増幅する**(効果の 69–88%) | 監査 P9 |
| C7 | **support は teacher top-k。** student top-k は目標が生徒に依存しフィードバックループになる | 旧アーム `frac_agree_pos` 0.238→0.193 |
| C8 | **`shuffled` 反実仮想が 0.82。** 位置対応を壊しても8割のゲートが開く。これを一級の指標にする | 監査 P7 |
| C9 | **タグトークン `{13708, 766}` が webshop の全 teacher-KL の 21.7%** を占める。診断で分離しないと機構の効果と混ざる | 監査 §13.5 |

**C1 と C2 から、両チャネルは target mode でなければならない。** 意図1を「per-token KL への重み」として実装すると C2 で表現できず、意図2は C1 で原理的に不可能。以下はすべて目標分布 $\tilde p$ の設計として書く。

---

## 1. 記号

位置 $t$、行のタスク $d$。support $S$ = **on-task 教師の top-k**($k=20$、C7)。$M$ = off-task 教師の集合($|M|=2$)。

| 量 | 定義 |
|---|---|
| $p_0(v)$ | base の確率 |
| $p_{on}(v)$ | on-task 教師($d$)の確率 |
| $p_m(v)$ | off-task 教師 $m$ の確率 |
| $\delta_{on} = \log p_{on} - \log p_0$ | on-task の policy shift [nats] |
| $\delta_m = \log p_m - \log p_0$ | off-task $m$ の policy shift [nats] |
| $\sigma_{on}, \sigma_m$ | 各教師の累積 RMS(対角のみ) |
| $\hat h_{on} = \delta_{on}/\sigma_{on}$, $\hat h_m = \delta_m/\sigma_m$ | RMS 標準化 shift |

**RMS 標準化は必須。** 生の nats では教師間で桁が違い(alfworld の push は ±12、search は ±0.33)、`min` 型の振幅は常に search に支配される。標準化後は全教師が $O(1)$ になる。

---

## 2. 中核:2つのチャネルは1つの式に telescoping する

off-task 教師の**符号一致**を両チャネル共通の前提にする:

$$s(v) = \mathrm{sign}(\hat h_1(v)) \quad \text{if } \mathrm{sign}(\hat h_1) = \mathrm{sign}(\hat h_2) \neq 0, \text{ else } 0$$

そして

$$L(v) = \Big(\textstyle\prod_{m \in M} |\hat h_m(v)|\Big)^{1/|M|} \quad (\text{off-task の合意音量、幾何平均}), \qquad O(v) = |\hat h_{on}(v)|$$

**幾何平均を採る**($\min$ ではない)。$\min$ は最も弱い声に支配されて消極的すぎ、実測で幾何平均の **1/2.19**(中央)・**1/6.1**(p90)しか出ない。幾何平均はどれかが 0 に近づけば 0 に向かうので**自己減衰(deadzone 不要)は維持**され、$L = \min(L,O) + \mathrm{relu}(L-O)$ は $L$ の作り方に依らない恒等式なので **telescoping も維持**、$\mathrm{geomean}(x,x)=x$ なので**教師複製への鈍感さも維持**される。

### 2.1 各チャネル

**A(意図1:裏付け、主チャネル)** — on-task が off-task 合意と同符号のときだけ発火:

$$a(v) = s \cdot \min\big(O(v),\, L(v)\big) \quad \text{if } \mathrm{sign}(\hat h_{on}) = s, \text{ else } 0$$

**B(意図2:代替信号)** — off-task が on-task を**超えた分**だけを使う:

$$b(v) = s \cdot \mathrm{relu}\big(L(v) - O(v)\big)$$

$b$ は $O$ が小さいほど大きく、$O \ge L$ で厳密に 0 になる。**「on-task が自信を持って別のことを言っているなら上書きしない」が閾値なしで成立する。**

**厳密な分割。** 恒等式 $|\hat h_m| = \min(|\hat h_m|, O) + \mathrm{relu}(|\hat h_m| - O)$ により、A は on-task の包絡内、B はその超過分を使う。**同じ証拠を二重計上しない。**

### 2.2 合成すると1つの式になる

$$c(v) = a(v) + b(v) = \begin{cases}
s \cdot L(v) & \mathrm{sign}(\hat h_{on}) = s \quad \text{(全教師一致 → off-task の声を満額)}\\
s \cdot \mathrm{relu}\big(L(v) - O(v)\big) & \text{otherwise} \quad \text{(on-task が沈黙 or 反対)}\\
0 & s = 0 \quad \text{(off-task が割れている)}
\end{cases}$$

一致枝で $a+b$ が $s\cdot L$ に潰れることの確認: $O < L$ なら $a = sO$、$b = s(L-O)$、和は $sL$。$O \ge L$ なら $a = sL$、$b = 0$、和は $sL$。**どちらでも $sL$。**

### 2.3 この式が連続であること(deadzone が不要な理由)

3つの境界すべてで跳びが消える:

1. **$\hat h_{on} \to 0$**(沈黙の境界)。$\hat h_{on} = +\epsilon$(一致)→ $c = sL$。$\hat h_{on} = -\epsilon$(反対)→ $c = s\,\mathrm{relu}(L-\epsilon) \to sL$。**一致**。
2. **$O \to L$**(B の消滅点)。$\mathrm{relu}$ の折れ点で、値は 0 に連続。
3. **off-task の符号が割れる境界**。片方が $\pm\epsilon$ に近づくと $L = \min_m|\hat h_m| \to 0$ なので、$s$ が反転する瞬間の跳びの大きさも $\to 0$。**自己減衰**。

したがって **$\epsilon$ / deadzone / 倍率 / ゲート閾値はどこにも要らない**(C4, C5)。

### 2.4 決定:conflict 枝を許す

$b = s\cdot\mathrm{relu}(L-O)$ は $O<L$ なら**符号が逆でも**正になる。すなわち on-task 教師が弱く反対している位置でも off-task 方向に目標が動く。実測(質量重み):

| | 候補 | 質量 |
|---|---:|---:|
| off-task が一致 | 0.431 | 0.308 |
| ┗ on-task も同符号(A枝) | 0.272 | 0.186 |
| ┗ on-task が反対(conflict) | 0.157 | 0.115 |
| ┗ **on-task が厳密に 0** | **0.002** | **0.007** |

**意図2を文字通り「on-task が沈黙」と読むと、対象は候補の 0.2% しかない。** 監査でも `neutral_on_task_silent` の実体は「$p_{on}=0.935$ で確信しつつ base と同じ」だった。したがって連続形 $\mathrm{relu}(L-O)$ が実際に拾うのは「沈黙」ではなく**「相対的に弱い」**位置である。

**決定 (a): conflict 枝を許す。** 意図2を「on-task の shift が**相対的に**小さい」と読む。硬く切ると $\hat h_{on}=0$ で不連続になり deadzone が復活するため。ただし §4.1 の通り**枝別に分離計測し、事後に ablation 可能**にする(B枝は $|c|$ 質量の 12.9%)。

### 2.5 意図との対応

| 意図 | 実現 |
|---|---|
| 全教師一致時に強く蒸留 | 一致枝で $c = sL$(満額)。$O$ にキャップされない — 一致しているなら on-task の音量で制限する理由がない |
| on-task 小・off-task 共通に大 → off-task から信号 | $O$ 小 ⇒ $\mathrm{relu}(L-O) \approx L$。**target mode なので固定点が動き、on-task 教師が表現していないものを学べる**(C1 の解除) |
| on-task が自信を持って反対 | $O \ge L$ で $c = 0$。上書きしない |
| off-task が互いに矛盾 | $s = 0$ で $c = 0$(C5 の代わりに一次の要求で濾す) |

---

## 3. (初版・撤回)目標分布:質量保存を構成上成立させる

> **撤回済み。** §0.0 撤回1 を参照。この節の交換($R_D, A_U, T=\min$)は実装から削除され、単純正規化に置き換わった。以下は撤回の理由を読むために残す。



C3(税)を消すため、**再正規化を使わず、作用集合の内部だけで質量を交換する**。交換量は**両側の容量の小さい方**で決める — これが唯一の上限になり、ハイパラを増やさずに $w$ が有界になる。

位置ごとに $U = \{v \in S: c(v)>0\}$、$D = \{v \in S: c(v)<0\}$ とする。

$$
\begin{aligned}
R_D &= \sum_{v \in D}\big(1 - e^{c(v)}\big)\,p_{on}(v) && \text{下げ側の供給}\\
A_U &= \sum_{v \in U}\big(e^{c(v)} - 1\big)\,p_{on}(v) && \text{上げ側の需要}\\
T &= \min(R_D,\; A_U) && \text{実際に動かす質量}\\[4pt]
w(v) &= 1 - \tfrac{T}{R_D}\big(1 - e^{c(v)}\big) & v &\in D\\
w(v) &= 1 + \tfrac{T}{A_U}\big(e^{c(v)} - 1\big) & v &\in U\\
w(v) &= 1 & v &\in S\setminus(U\cup D),\ \text{tail}\\[4pt]
\tilde p(v) &= w(v)\,p_{on}(v)
\end{aligned}
$$

$U$ か $D$ が空、または $T \le 0$ の位置は $w \equiv 1$(no-op)。

### 3.1 (初版)スケール:これは明示的な設計判断であり、唯一残ったハイパラである

> **決定を撤回。** 「初回は保守側」は撤回し、`exponent_scale = 2.148`(nats 換算)を採る。§0.0 撤回3。



$e^{c}$ が確率比なので $c$ は nats でなければならないが、$c$ は RMS 単位(無次元)である。実測で **1 RMS = 2.148 nats**(on-task、中央値)。したがって単位の選択で介入量が変わる:

| | RMS 単位のまま | on-task nats へ換算($c\cdot\sigma_{on}$) |
|---|---:|---:|
| $L=\min$ | $e^c$ = 1.52 | 2.45 |
| $L=$ geomean | $e^c$ = **2.90** | 9.82 |

**4通りで 1.52〜9.82 倍、6.5倍の幅がある。** 質量保存の容量制約は大きい $c$ で飽和するが、飽和点が「作用集合間で質量を総取り替え」なので自己制限は不十分。

**決定: 初回は保守側 — geomean × RMS 単位($e^c \approx 2.9$)。** 「本機構はハイパラを持たない」という主張は**撤回する**。単位換算という形で1つ持っている。nats 換算版は第2実験として残す。

### 3.2 この形が持つ4つの性質(すべて構成上、数値検証済み)

1. **質量保存が厳密。** $\sum_v w p = \sum_v p - T + T = \sum_v p$。tail と $c=0$ の候補は係数 1 のままなので $\sum \tilde p = 1$。**$Z=1$ は測定量ではなく恒等式**であり、`inv_z` は assertion になる。
2. **頭が動かない。** $c=0$ の候補は $w=1$。旧 target アームの「増幅の原資の 46–55% が教師の最自信トークン」が構造的に消える(C3)。
3. **符号に忠実。** $T/R_D,\,T/A_U \in (0,1]$ なので $w \le 1$ on $D$、$w \ge 1$ on $U$ が**常に**成立。「一致して負なら減衰」が破れない。
4. **$w$ が $e^{c}$ の内側に収まる。** $D$ 上で $e^{c} \le w \le 1$、$U$ 上で $1 \le w \le e^{c}$。**上限は教師の声の大きさ自身が決める** — 別途のクリップが要らない。

さらに **自己制限的**である: $U$ の質量が薄い位置では $A_U$ が小さく $T$ もそれに従うので、$\tilde p/p$ が爆発しない。

### 3.3 3案の比較(乱数 20,000 位置、$k=20$、Dirichlet(0.3) の $p$、$\mathcal N(0,1.5)$ の $\hat h$)

当初案(解放質量を $g\cdot p$ 比例で配る)は**病理を起こす**ことが数値検証で判明したため棄却した。

| 案 | $\max w$ の p99 | 符号忠実度 | 頭不動 | ハイパラ |
|---|---:|---:|:---:|:---:|
| (i) 解放 → $g\cdot p$ 比例配分 | **2,331**(最大 2.4e10) | 0.947 | ○ | なし |
| (ii) 作用集合内で再正規化 | 14.6 | **0.789** | ○ | なし |
| **(v) 両側傾斜 + 容量 $\min$**(採用) | **9.30**(最大 37.3) | **1.0000** | ○ | なし |

(i) の破綻原因: $D$ の解放質量が $U$ の容量を超える位置で、微小確率の候補に大量の質量を押し込む。配分の重みを変えても直らない(容量の問題)。(ii) は有界だが上げ下げの境界が $c=0$ ではなく作用集合の $p$ 加重平均 $\langle e^c\rangle$ になるため、$c<0$ の候補が質量を得る場合が 21% 生じる。

**検証済みの恒等式・性質**(乱数テスト):

| 検査 | 結果 |
|---|---|
| $a + b = c$ の telescoping | $\max\|a+b-c\| = 4.4\times10^{-16}$ |
| $\hat h_{on} \to 0$ での連続性 | $\hat h_{on}=\pm10^{-6}$ で $c$ が一致 |
| off-task 符号反転境界での連続性 | $\hat h_1 = \pm10^{-6}$ で $c \to 0$(自己減衰) |
| 質量保存 | $\max\|\sum wp + p_{tail} - 1\| = 6.7\times10^{-16}$ |
| $c \equiv 0$ で恒等写像 | $\max\|wp - p\| = 0$(ビット同一) |
| $w$ が $e^c$ の内側 | 100% |
| 教師複製に対する $L$ の不変性 | 複製 (1.3,1.3) と (1.3,4.0) で $L$ 同一 |

**損失:** $\mathrm{KL}(p_{student}\Vert\tilde p)$ を support + tail 上で。適用点は `dp_actor.py:3688` の `teacher_kld *= W` を**削除**し、KL 計算の前に教師 logprob を $\log\tilde p$ に差し替える。

---

## 4. (初版・無効)走行前に確定している予測値(事前登録)

> **この節の数値はすべて無効。** 質量保存交換 × teacher-indexed support の上で計算されており、機構が両方とも変わった。`target_tv = 0.019` と枝別シェア 0.871/0.129 は run ログからも削除した。**stale な事前登録は無いより悪い** — 機構が単に違うだけの状態を「違反」として読ませてしまうため。
>
> なお §4.1 の 0.8705/0.1293 は**枝別**の $|c|$ 質量シェアであり、実装が出す `target/channel/{a,b}_share` は**チャネル別**($\Sigma|a|p$ 対 $\Sigma|b|p$)である。一致枝でも $L>O$ なら $b$ が発火するので両者は一致しない。この対応付けの誤りは初版から存在したが、事前登録が無効になったことで実害は消えた。チャネル分解は §6 診断1 の量として引き続き有効である。



実イベント 10,624 件(両アームの spread、$\hat h$ は既存ダンプの標準化 shift)に**最終設計をそのまま通した**値。手計算ではなく実データからの推定である。

| 量 | 予測 | 備考 |
|---|---|---|
| $c \neq 0$ の教師質量 | **0.2545** (up 0.0806 / down **0.1739**) | 作用集合は下げ側が 2.2 倍重い |
| $\|c\|$ 中央($c\neq0$) | **1.002 RMS** → $e^c$ = 2.73 | geomean。$\min$ なら 0.42 |
| $\|c\|$ 質量加重平均 | 0.371 RMS | 中央より小さい = 高質量候補ほど $c$ が小さい(頭の効果) |
| 下げ側供給 $R_D$ | **0.0189** | 高質量候補の $\|c\|$ が小さいので供給は薄い |
| 上げ側需要 $A_U$ | 0.7288 | 低質量・高 $c$ の候補が名目需要を膨らませる |
| **交換量 $T$ = `target_tv`** | **0.0189** | **下げ側が律速**($T/R_D=1.00$、$T/A_U=0.026$) |
| 上げ側の相対質量変化 | **+23.4%** | $T / \sum_U p$ |
| 下げ側の相対質量変化 | **−10.9%** | $T / \sum_D p$ |
| `inv_z` | **1.0000**(厳密) | 構成上 |

旧 teachertopk アームの実測 `target_tv` = 0.0142 に対し **1.3 倍**。$\min$ 版なら 0.0107(旧アーム以下)。**幾何平均にした効果は $L$ で 2.19 倍だが、容量制約が throttle するので TV では 1.8 倍に留まる。**

**上げ側が強く throttle される($T/A_U = 0.026$)** ことは設計上の帰結として認識しておく。ただし絶対質量では上げ側が +23.4% を得るので、介入としては十分な大きさである。~~`target/throttle_up` = $T/A_U$、`target/throttle_down` = $T/R_D$ を必ず記録すること。~~ **この 0.026 が §0.0 撤回1 の直接の原因であり、両指標は交換ごと削除された。**

### 4.1 枝別のシェア(主チャネルが A であることの確認)

| 枝 | $\|c\|$ 質量シェア |
|---|---:|
| **A(全教師一致)** | **0.8705** ← 主チャネル、決定どおり |
| B(on-task が弱く反対、$O<L$) | 0.1293 ← §2.4 (a) で許可した分 |
| B(on-task が厳密に 0) | 0.0002 |

### 4.2 タスク別(G5 の事前予測)

| task | $c\neq0$ 質量 | $\|c\|$ 中央 | A枝シェア |
|---|---:|---:|---:|
| alfworld | 0.2556 | 1.072 | 0.904 |
| **search** | **0.3710** | **1.399** | 0.973 |
| **webshop** | 0.2358 | 0.837 | **0.659** |

**2つの警告がここに出ている。**

1. **search が最大の介入を受ける**(質量 0.371、$|c|$ 1.399)。しかし search は監査の全測定で外れ値で、転移も 0.35–0.49 と偶然水準。「最も信頼できないタスクに最も強く介入する」構図になる。
2. **webshop の A枝シェアが 0.659** — 介入の **34% が conflict 枝**から来る。webshop は ARM で唯一悪化したタスクである。

どちらも per-task 診断で分離できるようにしておくこと。

**この表からのズレは、設計かデータのどちらかの理解が違うことを意味する。** 特に「下げ側が律速」は質量の非対称から出ているので、逆転したら前提が崩れている。

## 5. 監視する指標(自動中止はしない — 判断は人間が行う)

監査は「この教師集合では信号が弱い」と繰り返し示している($\phi = 0.066$、新規性のある一致質量 0.3%)。当初は自動中止のゲートとして設計したが、**2026-09-02 に「中止はせず、指標の表示のみ」に変更した。** 走行は 150 step まで進み、下の値は判断材料として提示されるだけである。

実装は自動停止を一切持たない。毎 step、rank 0 が1行を run ログに出す(`[cross_teacher_target] gates (advisory): ...`)。全 rank で all_reduce 済みの同一値なので、rank 0 の1行が全体を代表する。

**改訂 (§0.0) 後の表。** G1〜G3 は機構を機構自身と比べる比なので、$w$ の作り方が変わっても**そのまま有効**である。事前登録に依っていた行(`tv`、`A/B`)は水準を外した。

| 表示名 | wandb キー | 気にすべき水準 | 何を意味するか |
|---|---|---|---|
| `\|dKL\|` | `target/abs_dkl_mean` | **旧アームとの比**が判断材料 | $\log Z - \langle c\rangle_{p_s}$ の絶対値、nats/位置。**この改訂が届いたかを決める数字。** 旧アームの top 層は 58.3 |
| `log_z` | `target/log_z_mean` | 発散、または `tv` に対して支配的 | 頭の課税。正規化で復活した項(§0.0 撤回1) |
| `clamped` | `target/clamped_frac_of_acted` | レートとして読む | クランプ 3.0 は binding する設計。0 なら `exponent_scale` が効いていない疑い |
| `tv` | `target/tv` | 水準なし(事前登録は無効) | $\mathrm{TV}(\tilde p, p_{on})$、tail 込み |
| `shuffled/live` | `target/shuffled_tv_ratio` | **> 0.6 で懸念** | off-task shift を行内で roll して $c$ を再計算した TV 比。旧機構のゲートはここで 0.82 だった = 文法で同じ質量を動かしていた |
| `novelty` | `target/acted_novelty` | **< 0.1 で懸念** | 作用先の $1 - \sum D_{on}/\sum D_{base}$。低ければ base 相続の合意に作用しており、on-task 教師が既に教えている |
| `tag_share` | `target/tag_share` | **> 0.3 で懸念** | タグ語彙への介入シェア。超えたら測っているのはタグのトークン化(§13.5、`'<th'` 単独で webshop の 21.7%) |
| `max\|log w\|` | `target/max_abs_log_w` | $> 2\times$ クランプ | 構成上 $|c_{\text{clamped}}| + |\log Z| \le 6$。超えたら実装のバグ |
| `mass_err` | `target/mass_error_max` | $> 10^{-12}$ | $\sum \tilde p = 1$($Z=1$ **ではない**)。指標ではなく assertion |
| `A/B` | `target/channel/{a,b}_share` | 水準なし(§4 参照) | チャネル分解。**枝別シェアとは別量**なので、§4.1 の 0.871/0.129 と直接比べてはいけない |

**読む時期。** `shuffled/live` と `novelty` は step 1 から計算できるので、150 step の完了を待つ必要はない。早期に懸念水準へ振れていたら、それは「機構の実装が悪かった」ではなく「**この教師集合に共有すべきタスク知識が無い**」ことの直接の証拠になり、negative result として §11.6 と合わせて報告できる形になる — その判断をいつ下すかは、走行を止めるかどうかとは別の問題である。

## 6. 診断(監査で「無いと読めなかった」ものだけ)

1. **チャネル分解。** $c = a + b$ の厳密な分割で、各チャネルの質量・TV・$\sum |c|$ シェア。`a` が支配なら意図1、`b` が支配なら意図2 が効いている。両方報告し、`allocation_cosine`(2チャネルが同じ候補を触っているか)も。
2. **状態別。** 一致枝 / B枝(on-task 沈黙) / B枝(on-task 反対だが $O<L$) / $s=0$ の4状態で、候補数・**教師質量**・TV。監査の「frac と mass_frac を必ず対で読む」を踏襲(候補数 4.3% が質量 64.7% だった)。
3. **反実仮想を3本。** `shuffled`(§5 の `shuffled/live`)、`a` のみ、`b` のみ。それぞれ TV と状態分布。
4. **タグトークンの分離**(G3)。`{13708, 766, 27, 29, 522, 1311, 151667, 151668}` の介入シェア。
5. **`top64_share` を必ず出す。** 監査で control のトークン表の絶対量を出せなかった原因(`opd/*` にはあった)。新アームでは `target/*` に同名で出す。
6. **event dump は全ターン。** 監査 §9-0 で判明した通り、既存ダンプは `turn=0` しか含まない。ターン層別サンプリングにすること。
7. **σ の軌跡。** $e^c$ は RMS のスケールを継承するので、$\sigma$ が漂うと介入量が漂う。`rms/sigma/<teacher>` を出す。

---

## 7. 対照の組み方(監査 §0.2 の再発防止)

前回の監査は、treatment の別 run を control と誤認して結論の頑健性を失った。**今回は次を守る。**

1. **同一ホスト・同一 GPU 枚数・同一コミット。** 前回は 2 対 3 枚、かつ control が `BALANCE_MINIBATCH_COLUMNS` 導入前のコードだった。
2. **`enable` 以外の差分ゼロ。** intent lock(`expected_*_config.yaml`)で機械検証。差分キーは `enable` 1個のみであるべき。
3. **ダンプ先の命名を treatment と control で明確に分ける**(前回の混同はここが原因)。起動時に `applied_weight` が control で恒等 1.0 であることを assert し、ログに出す。
4. **wandb の `opd/*` / `target/*` の列数を起動時にログ**。control で `target/*` が 0 列であることが機構オフの証拠になる。
5. **val instance log に生成文を保存する**(監査 §12.1)。現在は score のみで、文章の対比が原理的にできない。同一 `traj_uid` で両アームの生成文を残せば、初めて「何が変わったか」を見られる。

---

## 8. 実装

**新モジュール** `verl/trainer/ppo/cross_teacher_target.py`。既存の `cross_teacher_kl_weight.py` は position mode 専用で、$\mu$・$q$・student mass 測度がすべて不要になるため、拡張より新設が明快。

**実装済み(2026-09-02、branch `claude/cross-teacher-target`)。** 中核 `verl/trainer/ppo/cross_teacher_target.py`(§2 の式 + §3 の交換 + `TargetStepStats`)、actor 配線(`dp_actor.py`: 教師 logprob を $\log\tilde p$ に差し替え、3アーム相互排他 assert)、driver 配線(`opd_ray_trainer.py` の第3消費者)、config 伝播(`main_opd.py`)、run script(`run_multitask_cross_teacher_target_qwen3.sh`)、intent lock(`expected_multitask_cross_teacher_target_config.yaml`、機構キーは enable / base_path / exponent_scale の3つ)、対 control・対 klw のペアリングテスト(`test_cross_teacher_target_arm.py`: 差分が機構キーと識別子のみであることを合成 config で機械検証)。

**指標(すべて `target/*`、control には0列):** `tv`、`abs_dkl_mean`/`dkl_mean`/`abs_dkl_live_mean`、`log_z_mean`/`max_abs_log_z`、`shuffled_tv_ratio`・`acted_novelty`・`tag_share`・`max_abs_log_w`(§5、毎 step 1行で run ログにも出る)、`mass_error_max`($\sum\tilde p = 1$ の実測)、`live_frac`、`branch/<name>/{cand_frac,mass_frac}`(frac と mass_frac を対で)、`channel/{a,b}_share`、`channel/{a,b}_only_tv`(チャネル単独の反実仮想)、per-task 変種、`clamped_per_step`/`clamped_frac_of_acted`。σ の軌跡は既存の `kl_weight/rms/*` を同名で再利用(3 run が1軸に載る)。トークン表は `TokenStateCounts(mode="target")` を (branch,sign)→旧 state 名の完全対応で再利用し(dq = p_on(w−1)、w は正規化後なので候補ごとの変化として厳密。ただし **support 上でゼロ和ではない** — tail も $p_{tail}(1/Z-1)$ だけ動く)、イベントは `SignEventSamples` を再利用(`norm` 列は本物の $Z$ になった) — **ダンプファイル・scan script・読み手の語彙が3アームで共通**。D8 も実装: `trainer.val_instance_log_text=True` で検証 jsonl に生成文が載り、3スクリプトすべてに付与(既存2アームは次回の val_only から効く)。

**対 control の差分は機構キーのみである。** 初版では `student_indexed_topk`(control: true / target: false、C7)という第2の差分があったが、§0.0 撤回2 で C7 ごと撤回した。ペアリングテストの例外は削除し、代わりに3アームすべてが `student_indexed_topk=true` であることを検査する(`test_the_support_matches_the_comparators`)。キャッシュは行あたり 4 モデル読みに戻る。

**resume の扱い:** target アームは sidecar を持たない(μ も α も無いため)。resume 時は RMS が生きたバッチから再ウォームされ、最初のスナップショットまで $c\equiv0$(= no-op step が1回入るだけ)。

**再利用するもの:** `CumulativePolicyShiftRMS`、`standardize_policy_shifts`、`decorrelated_off_shifts`(§5 の反実仮想)、`TokenStateCounts` / `LogitPushTokens` / `SignEventSamples`(§6)、teacher hidden-state cache(teacher-indexed なので 3 モデル読みで済む)。

**捨てるもの:** `PreviousStepTaskKLWeightedMean`、cold start、`teacher_similarity`($q$)、`candidate_mass`(student 測度)、`report_epsilon`、sidecar のバージョン管理。

**適用点:** `dp_actor.py` の teacher KL 計算で、`sign_on_task_logprobs` を $\log \tilde p$ に差し替える。`teacher_kld *= W` の行は**削除**する(重みは存在しない)。$\log \tilde p = \log p_{on} + (c - \log Z)$ を float64 で作って1回だけキャストするので、作用していない位置では on-task 教師の bit がそのまま渡る。

**テスト(監査の教訓から必須):**

* $\sum_v \tilde p(v) = 1$ を乱数で 1e-12 以内。
* $c$ の連続性: $\hat h_{on}$ を $-\epsilon \to +\epsilon$ に掃いて $c$ が跳ばない。off-task の符号反転境界でも同様。
* $a + b = s\cdot L$ の telescoping を両枝で。
* $c \equiv 0$ のとき $\tilde p \equiv p_{on}$ が**ビット同一**(恒等写像であることの確認 — 旧 target アームでは再正規化のせいで成立しなかった)。
* 教師を複製したとき $L$ が不変(`min` なので複製に鈍感 — $q$ ゲートが持っていた「複製で source が倍になる」欠陥が無いことの確認)。
* per-task で $\sigma$ が独立に効くこと。

---

## 9. 残る不確実性(正直な見通し)

**この設計は監査が指摘した機構側の欠陥をすべて潰すが、信号が無いという測定結果は変わらない。** $\phi = 0.066$、base 統制後の偏相関 0.00–0.04、新規性のある一致質量 0.3%、advantage 予測 ≤ 0、action 粒度でも回復せず。

設計が正しくても効かない可能性が高い、という前提で走らせるべきである。だからこそ §5 の指標を step 1 から見る(中止はしない)。**`shuffled/live` と `novelty` が予測通りなら、それは「機構の実装が悪かったのではなく、この教師集合に共有すべきタスク知識が無い」ことの、これまでで最も直接的な証拠になる** — negative result として §11.6 と合わせて報告できる形になる。

そのうえで、監査が指摘した唯一の正方向の手掛かりは別にある: $\Delta_{on}$ の action 粒度(base 統制後 +0.128、§4.4/§5.2)。本機構の指標が懸念水準に振れた場合の次の候補はそちらである。

---

## 10. 残る決定事項

### 10.1 測定で解決したもの(決定不要)

| 論点 | 結論 | 根拠 |
|---|---|---|
| support は on-task 教師 top-k で足りるか(off-task の関心が外にあると B が盲目になる) | **足りる。** off-task 教師の確率質量のうち support 内は **98.4–99.3%**(全6ペア) | `kl_weight/evidence/<src>__on__<dst>/support_mass` |
| $\sigma$ は対角のみ($\sigma_{m,m}$)か、宛先条件付き($\sigma_{d,m}$)か | **対角のみが正しい。** $\sigma_{d,m}$ で割ると off-task 教師の domain 外の動きを domain 外の尺度で正規化するため、**雑音が $O(1)$ に引き上げられて「大声」に見える**。対角なら「自分の domain より小さく動いた」という情報が保たれる | 既存コード `CT:495–506` の設計意図と一致 |
| $\sigma$ をいつ読むか | **前 step のスナップショット。** micro/mini-batch 分割に objective が依存しないための要件 | commit `6eac8d7` の不変性議論 |
| tail の扱い | $w=1$ で固定。support カバレッジが 98%+ なので影響は小さい | 上記 support_mass |

### 10.2 決めるべきもの(影響の大きい順)

**(D1) シード数 1 か 2 か。** 監査の中心的な失敗は「どの比較も結論に至らなかった」ことである。ノイズ実測では全体成功率の 20-step 窓差の 2SE ≈ **0.032**。単一シード対では**それより小さい効果は原理的に検出できない**。2シードにすればコストは倍(150 step × 2 アーム × 2 シード)。
→ **決定: 1 シード。** §5 の指標で判断する前提(自動中止はしない)。**結果水準の差は原理的に 0.032 以下を検出できない**ことを、報告時に必ず添えること。

**(D2) 対照を新規に走らせるか、既存を流用するか。** §7 は同一ホスト・同一 GPU 枚数・同一コミットを要求する。既存 control(`91v55ri7`)は 3 GPU・別ホスト・`BALANCE_MINIBATCH_COLUMNS` 導入前で、**流用すると §0.2 の失敗を再現する**。新規は 150 step で約 27–29 時間。
→ **決定: 既存(`91v55ri7`)を流用。** よって §0.2 の交絡がこのアームにも引き継がれる: **GPU 2 対 3、別ホスト、`BALANCE_MINIBATCH_COLUMNS` 導入前のコード**。成功率の差は機構の効果として解釈できない。**§5 の機構診断が事実上の唯一の判断材料になる**ので、`shuffled/live` と `novelty` を早期から必ず見ること。

**(D3) `pg_loss_coef` を 1(OPD+GRPO)か 0(純 OPD)か。** 1 なら現行アーム群と比較可能だが結果が GRPO と混ざる(`grpo/grad_cosine ≈ 0.01` でほぼ直交はしている)。0 なら機構の効果が分離できるが、比較相手は旧 signweight アーム群になる。
→ 推奨: 1(走行中の control と揃える)。

**(D4) off-task 教師集合。** 全て使うか、非 search 宛先で search を落とすか。幾何平均は自己減衰するので、**情報を持たない教師が 1 本混じると $L$ が引き下げられる**。監査 §6 (A) が提案した最安の交絡切り分けでもある。ただし $|M|=1$ にすると符号一致 $s$ が自明になり(単一教師は常に自分と一致)、機構はペア裏付けに退化する。
→ 推奨: 初回は全て。§4.2 の警告(search が最大の介入を受ける)を診断で追跡し、必要なら第2実験で落とす。

**(D5) $\sigma$ の cold start。** $\sigma$ が有効になるまでどうするか。
→ 推奨: $c \equiv 0$(no-op)。旧 `cold_start_state` と同じ挙動で、step 1–2 が不作為になるだけ。

**(D6) タグトークンを機構から除外するか、まず測るか。** 監査 §13.5 の通り `{13708, 766}` が webshop の全 teacher-KL の 21.7%。機構の作用集合がタグに支配されると、測っているものが「タグのトークン化」になる。
→ 推奨: **まず測る**(§5 G3 が 0.3 超で警告)。事前除外すると旧アームとの比較可能性が落ちる。

**(D7) 主要評価指標の事前登録。** pooled `val/success_rate` は search が母数の 99.5% を占めるので3タスク要約に使えない(監査 §11.2 と旧レポート)。
→ 推奨: **タスク別成功率 + 一致した `traj_uid` 上の McNemar** を主要評価とし、事前に固定する(`scripts/val_paired.py`)。pooled は報告しない。

**(D8) val instance log に生成文を保存するか。** 現在は score のみで、**文章の対比が原理的にできない**(監査 §12.1)。writer の小変更で済む。
→ 推奨: 入れる。この機構は目標分布を動かすので、「何が変わったか」を文章で見られることの価値が position mode より高い。

**(D9) `teacher_kl_loss_coef` を 0.01 のままにするか。** 目標が動くので `actor/teacher_kl_loss` は上がる(生徒は $p_{on}$ より $\tilde p$ から遠い)。
→ 推奨: 0.01 のまま(比較可能性)。ただし `actor/teacher_kl_loss` の絶対値が control と乖離することを予期し、乖離量を記録する。

