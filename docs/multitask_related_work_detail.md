# 関連研究 詳解 — 各論文の設定・機構・数式・数値・本研究との距離

原稿の Related Work 節を書くための一次資料。
2026-08-10 セッションで arxiv HTML 原文を照合した内容(`docs/multitask_transfer_related_work_review.md`)を、
**論文ごとのカード形式**に展開したもの。各カードは以下の構成:

- **設定** — 何タスク / 教師は何本 / どのモデル
- **機構** — 損失と転移経路(数式を原文の形で)
- **数値** — 引用可能な具体値
- **本研究との距離** — 何が同じで何が違うか
- **原稿での書き方** — そのまま使える一文

信頼度の注記は末尾 §8 にまとめた。**投稿前に PDF で最終確認すべき箇所は ⚠️ を付けてある。**

---

## 0. 全体地図 — 2軸で並べると空白が見える

既存研究は「転移の型」と「タスク数」の2軸に並ぶ。

|  | **単一タスク** | **マルチタスク** |
|---|---|---|
| **coverage 型**<br>(共有表現で自然に混ざる) | — | **PromptSD** ← 唯一の正の転移報告 |
| **override 型**<br>(他教師の意見を位置ごとに注入) | **MAD-OPD**, W2S-OPD, OPD² | **← 空白。本研究がここ** |
| **転移を避ける**<br>(ルーティングで遮断) | — | **MOPD, Two-Phase, CaMOPD, UI-MOPD** |

**この表が Related Work 節の骨格になる。**
「マルチタスク OPD は4本あるが全部が転移を遮断する設計」
「転移を起こすと主張するのは PromptSD 1本だけで、その機構は共有表現」
「多教師で上界を破るのは MAD-OPD だが単一タスク」
— この3文で空白象限が定義できる。

---

## 1. 最近接競合(新規性判定に直接効く4本)

### 1.1 PromptSD (2607.18293) — **最近接競合。原稿の扱いが最も危険**

**正式タイトル**: *One Student, Many Teachers: Multi-Task On-Policy Distillation via Soft-Prompt Privileged Context*

**設定**: 4タスク(Math / Science / Tooluse / Biology)。生徒1本に対しタスク別教師4本。

**機構**:
教師は「**生徒と同一の凍結バックボーン + タスク別の学習可能ソフトプロンプト**」である。
タスク $k$ の教師は $\pi_{\theta + P_k^\star}$ と書かれ、$\theta$ は生徒と共有・凍結、
$P_k^\star$ だけが訓練される。原文の主張は
"preserves the student's exact representational geometry"。
FFT 教師・LoRA 教師・別モデル教師はいずれも劣る対照条件として提示されている。

つまり**転移の源泉は「教師と生徒が同じ表現空間を共有していること」= coverage 型**である。
明示的な転移注入機構(重み・ゲート・合成)は持たない。

**数値**:
- Math: 単一タスク OPD 51.0% → マルチタスク 67.2%(**+16.2pp**)。ベースは 16.4%
- **leave-one-teacher-out**: Math の教師とデータを**両方**除外しても Math 61.0%
  → 他タスクの教師だけで対象タスクが大幅向上する直接証拠
- ただし転移は一様に正ではない: Tooluse **−3.1**、Biology **−8.0**(マルチ vs 単一タスク OPD)

**逐語引用**(原稿で使える):
> "Multi-task exposure thus provides **positive cross-task transfer**: jointly training on
> Science, Tooluse, and Biology rollouts reinforces the reasoning and instruction-following
> Math latently requires"

**本研究との距離**:

| | PromptSD | 本研究 |
|---|---|---|
| 教師の作り方 | 同一凍結バックボーン + ソフトプロンプト | **独立に RL 訓練した単一タスク ckpt** |
| 転移経路 | 共有表現(暗黙) | **教師間 policy shift の符号一致(明示)** |
| 転移の制御 | 不可(混ぜれば起きる/起きない) | **候補ごとに増幅/減衰を選べる** |
| 主張 | 「転移が起きた」 | **「どのタスク対で起き、どこで起きないかを事前に測れる量で予測する」** |

**原稿での書き方**:
> PromptSD は多タスク OPD で正の跨タスク転移を報告した唯一の研究だが、
> その教師は生徒と同一の凍結バックボーンにソフトプロンプトを付したものであり、
> 転移は共有表現に由来する。独立に訓練された異種教師間で転移を起こす機構ではない。

⚠️ **原稿の現行記述「単OPSD手法でMOPD手法ではない」は誤り。**タイトルに Many Teachers と
Multi-Task が入っている。ここを間違えると新規性主張ごと崩れる。

---

### 1.2 MAD-OPD (2605.01347) — **上界突破そのものが主題**

**正式タイトル**: *Breaking the Ceiling in On-Policy Distillation*(Multi-Agent Debate)

**設定**: **単一タスク**。汎用教師の**3ペア**で討論させる:
Qwen3 14B+8B / Qwen3 32B+30B-A3B / Qwen3.5 27B+9B。タスク特化の概念はない。

**機構**:
複数教師に討論(debate)させ、各ステップの自己申告 confidence $w_k$ で重み付けした
**ダイバージェンスの加重和**を最小化する:

$$\mathcal{L} = \sum_k w_k\, D(p_{T_k} \| p_S) \qquad \text{(Eq. 8)}$$

**重要**: 混ぜているのは分布ではなく**ダイバージェンス**である。
forward KL なら勾配的に算術混合と等価だが、JSD / reverse KL では等価にならない。
confidence は **per-step**(トークン毎ではない)。

**Prop 1**: reverse KL は $p \to 0$, $q > 0$ で勾配が非有界になるため避ける、と論じている。
(ただし著者らのコードは実際には reverse KL を使っている。)

**数値**: 4B 生徒が 14B 教師を上回る事例あり。結論部は
「OPD のボトルネックは**教師プールの多様性**であって単一教師の能力ではない」。

**本研究との距離**: override 型多教師蒸留の先行だが**単一タスク**。
教師はサイズ違いの汎用モデルで、「タスク特化教師どうしの意見の食い違い」という
本研究の問題設定そのものが存在しない。

**原稿での書き方**:
> MAD-OPD は多教師の討論で教師上界を破ることを示したが、教師は同族の汎用モデルであり
> 単一タスク設定である。タスク特化教師間の跨タスク転移は扱われていない。

⚠️ **原稿の「上界突破を狙った研究ではない」は誤り。**タイトルが Breaking the Ceiling である。
生き残る差別化は **single-task か multi-task か**の一点。

---

### 1.3 MOPD (2606.30406) — 干渉回避型の代表

**正式タイトル**: *MOPD: **Multi-Teacher** On-Policy Distillation for Capability Integration in LLM Post-Training*
(原稿が「多ドメイン」と書いているのは正確でない。Multi-Teacher が正式名)

**設定**: 3ドメイン(数学 / 指示追従 / SWE)、多教師 → 生徒1本。

**機構**: **per-prompt routing**。プロンプトがどのドメインか判定し、対応する教師だけで
reverse KL を取る。したがって**他ドメインの教師の意見は当該位置に一切入らない**
= 転移経路が構造的に遮断されている。

目的は転移ではなく **"see-saw"(あるドメインを上げると別が下がる)の回避**である。
この語は MOPD が使う(Two-Phase には出てこない)。

**数値**: MiMo-V2-Flash で生徒が教師を上回る事例あり(HMMT25 **+1.8** ほか)。
著者はこれを**反復ラウンドの効果**に帰属させており、跨タスク転移とは主張していない。

**本研究との距離**: 問題設定(多教師 OPD の能力統合)は最も近いが、
機構は真逆(遮断 vs 注入)。

⚠️ **名前衝突に注意**: 「MOPD」は 2606.30406 と 2605.12652(Multi-Rollout OPD)の
2本が名乗っている。引用時に必ず番号で区別すること。

---

### 1.4 Two-Phase (2606.30044) — 外部手法として最も再現しやすい

**設定**: 4タスク agentic(airline / telecom / sudoku / mastermind)。
タスク別 RL 専門家 → 生徒1本。生徒 8B。

**機構**: 2段階。
- **Phase 1**: 教師軌跡で offline SFT
- **Phase 2**: reverse KL を **RL の報酬に変換**し、per-token advantage
  $\hat A_t = -(\log \pi_\theta - \log \pi_T)$ で on-policy 蒸留

**新規に確認した重要事実**: Phase 2 **にもタスク別ルーティングがある**。逐語:
> "For a prompt from task $i$, we apply the on-policy distillation objective using the
> corresponding teacher model $\pi_T^{(i)}$"

問題意識の逐語(引用価値が高い):
> "aggregating data from multiple tasks introduces a large number of behavioral modes that
> can exceed the student's capacity, forcing it to average across behaviors"

**数値**: 8B 生徒 vs 各専門教師 — airline **−2.1** / telecom **±0** / sudoku **+2.0** /
mastermind **+1.3**。つまり主張は「**劣化なしの統合**」であり正の転移主張ではない。
Mix-RL 比較は Appendix B.1("multi-task RL underperforms distillation-based methods"、
telecom が 85.9→80.6 に劣化)。

**Limitations / Future Work 節は存在しない。跨タスクシナジーへの言及はゼロ。**
→ 「彼らが open problem として認めたギャップ」と書くのは不可。
「単に扱われていないギャップ」と書くこと。

**本研究との距離**: 本リポジトリの `offline-KD → 純OPD` 連結でほぼ再現できる。
**ほぼ無料で追加できる外部比較アーム**として価値が高い。

---

## 2. 同一ベンチマーク圏(数値の外部参照点)

### 2.1 ATOD (2606.27814) — **本プロジェクトと完全に同じ3ベンチ**

**設定**: ALFWorld / WebShop / Search-QA、Qwen3 生徒(0.6B / 1.7B / 4B)。
ただし**単一タスク**(3ベンチそれぞれで別々に訓練)。マルチタスク実験はない。

**機構**: annealed OPD → RL。序盤は OPD 主体、後半は RL 主体へ係数を焼きなます。
重み **T-DUR** は **per-turn**(トークン毎ではない):

$$w_k = 1 - (1-\tilde d)(1-\tilde h)$$

$\tilde d$ = 教師-生徒の不一致、$\tilde h$ = 不確実性。soft-OR で結合。**OPD 項のみに適用**。

**数値**(1.7B の到達点): ALFWorld **80.47** / WebShop **89.06** / Search-QA **45.21**。
本プロジェクトの 300step 値(AlfWorld 0.849 / WebShop score 0.869)と同じ帯域にある
→ **外部参照点として使える**。

**Search-QA の改善幅はどの構成でも最小**: OPD→ATOD で **+0.2〜+0.9**、
GRPO→ATOD でも **+2.9〜+3.2**。
→ 本研究の「Search 完全帰無(p=0.85)」への**外部傍証**。原稿で必ず引く。

**本研究との距離**:
- 重み付け粒度: ATOD は per-turn、本研究は**候補トークンごと**
- 重みの信号源: ATOD は**生徒-教師ギャップと不確実性**(単一教師内で閉じる)、
  本研究は**別タスク教師の符号一致**(教師間)
- ATOD の annealing は本機構と**直交**する。実測で本機構の利得が後半消えている以上、
  annealing は素直な次の一手になる(`docs/multitask_next_experiments.md` 案1)

---

### 2.2 Revisiting OPD (2603.25562) — top-K 実装の規範

**設定**: OPD の推定量分析。評価は数学 + ALFWorld / WebShop。

**機構**: 推奨推定量は **teacher top-K local support matching**。決定的なのは
**教師・生徒とも top-K 集合 $S$ の内側で再正規化する**点(Eq. 7):
> "Renormalization is necessary because the objective is evaluated on a truncated support"

**本研究との距離 — ここは実装差がある**:
本リポジトリの `dp_actor.py:431` は
```python
topk_out = (logits.gather(-1, topk_ids) - lse)
```
で **full-vocab の log-softmax を抜き出しているだけ**であり、$S$ 上の総和は1にならない。
つまり文献の推奨推定量とは**異なる推定量**を使っている。

本機構の target モードでは、この残差質量を **tail バケット**として明示的に扱っている:
$$Z = \sum_{v \in S} w(v)p_t(v) + \underbrace{\Big(1 - \sum_{v\in S} p_t(v)\Big)}_{\text{tail}}$$
これにより $w \equiv 1$ で厳密に恒等写像になる(テストで $\max|out-in| = 0$ を確認済み)。
**原稿では「$S$ 内再正規化ではなく tail アンカー方式を採った」と明記すべき。**
バイアスの性格が違うので、黙って書くとレビューで刺される。

---

### 2.3 SDAR (2605.15155) — 上流。`festive-gates` アーム

**設定**: 3タスク(本プロジェクトと同一ベンチ圏)、**教師なし**。

**機構**: 純粋な蒸留ではなく **RL 主軸 + sigmoid ゲート付き OPSD 補助**。
スキル(自然言語)を**特権文脈**として教師ブランチに与え、自己蒸留する。
`algorithm.sdar.skills_dirs.{alfworld,search,webshop}` でタスク別スキルを読む。

**本研究との距離**: 転移経路が**根本的に別チャネル**である。
- 本機構: **ロジット**を通る → 共有できるのは出力トークンで表現できるものだけ
  → 出力形式が違えば原理的に何も通らない(Search 帰無の説明)
- SDAR: **自然言語の特権文脈**を通る → 出力形式に依存しない

この解離が `docs/multitask_two_channel_proposal.md` の骨子。
**WebShop と Search は "Iterative Query Refinement" という同名スキルを共有している**一方、
形式の Jaccard は 0.30 しかない。「チャネルごとに支配する距離が違う」という主張の材料になる。

⚠️ 引用時の注意: "teacher-ceiling" という複合語は原文にない。abstract の
"gains saturate once the student approaches the teacher, limiting the final performance ceiling"
を使うのが安全。

---

## 3. 干渉回避型(残り2本)

### 3.1 CaMOPD (2605.27115)

**設定**: 一般能力の回復 + ドメイン能力の保存。

**機構**: 勾配対立を**交互スケジュール(3:1)で時間分離**する。
選択の粒度は**サンプルレベル**。トークンレベルのゲートは持たない。

**本研究との距離**: 問題設定(多教師 OPD の干渉)は同一だが、
機構は勾配空間のドット積診断 + スケジューリングで完全に別。
「干渉を時間で分ける」vs「候補ごとに空間で分ける」という対比で書ける。

### 3.2 UI-MOPD (2607.04425)

**設定**: デスクトップ / モバイル GUI エージェント。

**機構**: **プラットフォーム条件付き routing** で遮断。

**本研究との距離**: 共有するのは問題意識(「相互作用規約の差」が干渉を生む)だけ。
本研究の Search の帰無 —「出力形式が違うと転移しない」— と同じ現象を
別ドメインで観測しているとも読めるので、**形式距離仮説の傍証**として引ける。

---

## 4. δ(policy shift)を扱う系統 — 機構K側の包囲網

本研究の中核概念 $\delta_m(v) = \log \pi_m(v|s) - \log \pi_0(v|s)$ は、
**「RL で獲得した能力は logit の差分に載る」**という系譜の上にある。
ここは先行が厚いので、**δ を使うこと自体では新規性を主張できない**。

### 4.1 Weak-to-Strong OPD (2607.26246, Microsoft Research) — **機構Kの直接競合**

合成教師を
$$\pi_{T,\alpha} \propto \mathrm{softmax}\Big(z_{\text{base}} + \sum_k \alpha_k (z_k^+ - z_k^-)\Big)$$
と定義し、**生徒ロールアウト上の reverse KL で OPD ターゲットにする**。
数学 + コード、$\alpha$ は固定、衝突解消機構はない。

logit 空間で $\sum_m w_m \delta_m$ を足すことは幾何平均 $\prod_m \pi_m^{w_m}$ と**恒等**なので、
**「δ 合成教師を蒸留ターゲットにする」自体は既発表**である。

→ 機構Kの新規性主張は「合成教師蒸留の発明」ではなく
**「合意ゲート付き off-task consensus 正則化」**に絞る必要がある。残る差分は
(1) on-task 教師を除いた綱引き構成($\gamma$ が純粋な転移ツマミになる)、
(2) $\kappa_t$ によるトークン毎の合意ゲート(W2S-OPD に衝突解消はない)、
(3) $\epsilon \pi_0$ 平滑化、(4) マルチタスク agentic 干渉の緩和という目的設定。

**副次的な効用**: W2S-OPD は「δ 合成ターゲットへの reverse KL」が実際に機能することを示した
実績なので、機構Kが reverse KL を採る根拠として**引用できる**(§5.1 の修正の裏書き)。

### 4.2 Direct-OPD (2607.05394)

単一タスク(数学)、weak-to-strong。**policy shift δ を暗黙報酬として扱う**。
top-k overlap を必要としない定式化。
→ 「δ を転移媒体にする」発想の先行(ただし**同一タスク内**)。

### 4.3 ExOPD / G-OPD (2602.12125)

報酬外挿 $\lambda > 1$ で
$$\text{target} \propto \pi_{\text{ref}} \left(\frac{\pi_T}{\pi_{\text{ref}}}\right)^{\lambda}$$
すなわち δ を**増幅**した教師へ蒸留する。教師超えの経験的根拠。

⚠️ **形式的な定理・命題は存在しない。** informal な Remark と経験的結果のみ。
原稿で「理論」と書くと過大。「定式化と経験的根拠」と書くこと。

**本研究との距離**: 本機構の増幅係数 1.25 は、**候補ごとに $\lambda$ を変える**ものと読める。
ExOPD は全候補一律。この対比は原稿で使うと分かりやすい。

### 4.4 推論時制御の系譜(δ 合成の祖先)

| 手法 | 合成式 | 位置づけ |
|---|---|---|
| **DExperts** (2105.03023) | $z_{\text{base}} + \alpha(z^+ - z^-)$ | 元祖 |
| **proxy-tuning** (2401.08565) | 小モデルの δ を大モデルに足す | 訓練不要の能力移植 |
| **EFT** (2310.12962) | 報酬と方策の分解 | δ の解釈枠組み |
| **Logit Composition** (2605.28304) | $\hat p \propto p_{\text{base}} \prod_i (p_i/p_{\text{base}})$ | factorized conditionals の理論付き |
| **ThinkLogit** (2507.12759) | — | **RL 獲得能力が δ に載ることの実証** |

ThinkLogit は本研究の前提そのものを支える実証なので、**Introduction で引くべき**。

### 4.5 OPD² (2607.15161)

δ を per-token 報酬にし、**符号一致ゲート**を掛ける。単一教師。
→ 「δ の符号でゲートする」という発想の最接近。**本研究との差は教師が1本か複数タスクかの一点**
なので、related work で明示的に差別化しないと危ない。

---

## 5. トークン重み付け系統 — 機構W側の包囲網

2026年の激戦区で **20本超**ある。掃引した範囲:
EOPD (2603.07079), TIP (2604.14084), SCOPE (2604.10688), PACED (2603.11178),
2605.26844, FiRe-OPD (2606.02684), SEAD (2606.28562), 2606.22600,
CADENCE (2607.16955), DASH (2608.06243) ほか。

**重要な発見: 重み信号はすべて次の5種のいずれかに帰着する。**

1. エントロピー
2. 正誤(verifier)
3. 生徒-教師ギャップ
4. 生徒自身のドリフト
5. verifier 合意

**「off-task 教師群の policy shift の方向合意 $\kappa_t$ × on-task shift との整合 $a_t$」という
教師間・shift 空間・方向性の信号は見つからなかった。**
OPD サーベイ2本(2604.00626, 2606.22793)のトークン重み分類にも該当項目がない。

### 5.1 最接近4本(必ず差別化を書くこと)

**UniSD (2605.06597) — 機構面での単一最接近**
「複数ビューの合意 → トークン毎信頼度重み」というテンプレートが**同一**。
ただしビューは**単一教師の文脈摂動**であり、信号は log-prob の**分散**(スカラー)。
本研究は独立 RL 教師の base 相対 shift の **cosine(方向)**。マルチタスクでもない。

**GOVERN (2405.03764)**
教師ごとの**勾配降下方向の多数決投票**でサンプル単位に教師を選択。
「方向合意で裁く」発想の先行。ただし**勾配空間・サンプル単位・BERT 系**。

**SG-OPD (2606.09304)**
verifier と教師の**符号合意ゲート**。単一教師。

**Blockwise Policy-Drift Gating (2606.24084)**
log-prob シフトでトークン重み。ただし信号源は**生徒自身の**ドリフト。

### 5.2 位置づけの一文(そのまま使える)

> PCGrad / CAGrad / Gradient Vaccine が確立した「勾配空間の cosine 合意で干渉を裁く」という
> 発想を、**logit shift(policy shift)空間へ移し、蒸留損失のゲートとして用いた初の研究**である。

補強材料: **2607.16062** は、エージェント RL のタスクベクトルが重み空間で cosine 0.06〜0.10 と
**ほぼ直交しているのに機能干渉は残る**ことを示した。
→ 「重み空間の cosine は機能を写さない。だから shift 空間で測る」という動機づけになる。

引用すべき勾配衝突 MTL: PCGrad (2001.06782), CAGrad (2110.14048),
Gradient Vaccine (2010.05874), TIES-Merging (2306.01708)。

### 5.3 Distral (1707.04175) — 思想的先祖

全タスクの方策を、**学習される共有セントロイド** $\pi_0$ への KL で正則化する
マルチタスク RL。機構Kの遠い祖先。差は3点:

1. Distral のセントロイドは**全タスクの算術的重心を学習**する /
   機構Kは **on-task 教師を除いた off-task 教師の凍結幾何平均**
2. Distral の KL は**一様・訓練全体** / 機構Kは $\kappa_t$ によるトークン毎ゲート付き補助項
3. **Distral に方向合意の概念はない**

---

## 6. リスク文献(先回りして引いておくべき3本)

### 6.1 Top-K サポート欠落 (2607.07050, "When Top-K Misses the Decision")

多教師 OPD で **top-K 打ち切りサポートが挙動決定的トークンを欠落させる** failure mode の実証。
決定的な数値: **top-32 が確率質量の 99.99% を保持していても、決定トークンの 0.4% しか含まない**事例。

**本研究への影響**: 共通サポート $S_t = \text{top-}K(q) \cup \{\hat y_t\}$ は**生徒基準**なので、
off-task 教師が推す決定トークンが落ちうる。本機構は on-task 教師の top-20 で候補を決めているため、
**off-task 教師の意見は「on-task 教師が見ている候補の上でしか」聞けない**。
これは仕様であって不具合ではないが、**limitation として明記すべき**。
対処するなら $S_t$ に各教師の top-$k'$ を合併する(予算 $K = 64$〜$128$ 内で配分)。

### 6.2 SFT Conflicts, RL Coexists (2608.03573)

マルチタスク訓練 / マージングで「**KL と性能変化の相関が有意でない**」との報告。
→ 「$\kappa$(shift 空間の合意)が転移を予測する」という本研究の前提への**潜在的反証**。
先回りして引用し、$\kappa$ と実際の転移利得の相関を測定項目に含めて直接答えるべき。

### 6.3 ⚠️ H-OPD (2607.02592) — **未確認**

confidence-aware 多教師マルチモーダル OPD。**本文の機械抽出に失敗しており教師結合式が未確認**。
多教師 × confidence 重みという組合せが本研究に近い可能性があるため、
**投稿前に PDF 精読が必須**。

---

## 7. 同一ベンチ圏の単一教師 agentic OPD(related work に列挙するだけでよい)

- **FutureBridge-OPD (2608.01953)**: ALFWorld / WebShop / ScienceWorld で +16.6pt
- **SKILL-KD (2607.28048)**
- **Skill-SD (2604.10674)**
- **Structured Agent Distillation (2505.13820)**

いずれも単一教師・単一タスクなので競合しない。「agentic OPD は活発だが、
いずれも単一教師である」という一文で束ねられる。

---

## 8. 信頼度の注記

| 対象 | 方法 | 信頼度 |
|---|---|---|
| §1〜§4 の主要11本 | arxiv HTML 原文を複数回照会、争点は逐語文字列チェック | **高**(逐語引用は原文英語で保持) |
| §5 の20本超の掃引 | WebSearch + abstract 確認、重み信号の分類のみ | 中(**「見つからなかった」の主張は掃引範囲に依存**) |
| §6.3 H-OPD | 抽出失敗 | **低。PDF 必須** |

**投稿前チェックリスト**:

1. ⚠️ H-OPD (2607.02592) の教師結合式を PDF で確認
2. ⚠️ ATOD の "teacher-ceiling" 複合語 / UI-MOPD の SFT ベースライン数値を PDF で確認
3. ⚠️ MOPD の名前衝突(2606.30406 vs 2605.12652)を引用で区別
4. 逐語引用(PromptSD / Two-Phase / Revisiting OPD)は PDF で最終照合
5. §5 の「該当なし」主張は、投稿直前にもう一度サーベイを更新
   (2026年のこの領域は月単位で増えている)

---

## 9. 原稿の Related Work 節の推奨構成

1. **OPD の基本**(Agarwal et al. → Revisiting OPD の推定量)— 2〜3文
2. **多教師 / 多タスク OPD** — MOPD / Two-Phase / CaMOPD / UI-MOPD を
   「**全て per-prompt / per-platform ルーティングで転移経路を遮断している**」で束ねる
3. **正の跨タスク転移** — PromptSD **のみ**。共有表現型であることを明記(最近接競合)
4. **教師上界の突破** — MAD-OPD(多教師だが単一タスク)、ExOPD(δ 増幅、定理なし)、
   W2S-OPD(δ 合成、衝突解消なし)
5. **トークン重み付き蒸留** — 5種の信号に分類し、**教師間の方向合意が空白**であることを示す。
   UniSD / GOVERN / SG-OPD / OPD² を名指しで差別化
6. **勾配衝突 MTL** — PCGrad 系。「shift 空間へ移す」という一文で本研究に接続
7. **同一ベンチ圏** — ATOD(数値の外部参照点、Search が動きにくいことの傍証)、SDAR(上流)

この順序なら、§0 の 2×2 表の空白象限が自然に浮かび上がる。
