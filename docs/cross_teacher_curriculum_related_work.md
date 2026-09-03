# 裏付け限定カリキュラム蒸留の関連研究サーベイと位置づけ

状態: **サーベイと分析のみ。実装・走行なし(2026-09-03)。**
対象機構: [cross_teacher_curriculum_design.md](cross_teacher_curriculum_design.md)(`cross_teacher_target.py` の `mode="curriculum"`)。
前提: [cross_teacher_theory.md](cross_teacher_theory.md)(階層モデル・識別不能性・命題 1–3)、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)($g$ ≈ 書式)。

問い: (1) この機構は既存研究のどれと**類似・競合・対立**するか。(2) 既存研究の**傾向**はこの機構を支持するか。

文献は 2026-09-03 時点で arXiv / 会議録ページを取得して同定したもの(著者・年・arXiv 番号)。
本文書の引用は要点の要約であり、数値は各論文の abstract / 本文からの転記。「未検証」と記した項目は listing でしか確認できていない。

---

## 0. 結論(先に)

1. **機構は 3 つの既知の系譜の交点にあり、交点そのものは先行例が無い。**
   (a) 蒸留**目標**を時間で変えるカリキュラム(Annealing-KD、CTKD、TAID、progressive distillation)、
   (b) 複数の fine-tune 済みモデルの**符号一致で共有成分を切り出す**合意機構(TIES、Consensus/TALL mask、AND-mask、NegMerge)、
   (c) 同一 base から作った**ドメイン別 RL 教師 → 単一生徒**の on-policy 蒸留(MOPD、ExOPD、Open-MOPD、CaMOPD、LS-MOPD、Two-Phase Distillation)。
   「base 相対の logit shift $h = \log\pi_T - \log\pi_0$ を、他教師の符号一致本数で入れ子分解し、段階解放する蒸留目標」は (a)(b)(c) のどれにも無い(§3.1)。
   ただし各部品には明確な系譜があり、新規性は**組合せ**にある。査読では「TIES / Consensus を logit 空間に持ち込み、TAID のように時間解放した」と要約される。
2. **最も近い構造的類似は、順に** Consensus/TALL mask(k-of-T の数え上げ)、AND-mask(符号一致率の閾値)、NegMerge(全会一致の保持)、TIES(符号選挙)、
   AE-KD(全教師が同意する降下方向)、tri-training(2-of-3 の合意)、SG-OPD(同じ GRPO+OPD 設定での符号ゲートと段階スケジュール。ただし教師×検証器の符号で、
   多教師ゲートは future work と明記)、TAID / TOP-D(生徒側から教師側へ動く目標)、Distral(共有方策 × 固有 advantage の乗法形)。
3. **競合は 4 系統**(§3.2): (i) 目標ではなく**係数**を時間で変える $\beta(t)$ anneal / warm-up-then-anneal(KDRL、SAF-OPD、ATOD。ATOD は本機構と同じ
   ALFWorld / WebShop / Search-QA で利得を報告)、(ii) 教師の**助けの量**を前半に厚く後半に消す誘導減衰(UFT、CBRL、Prefix-RFT、CHORD、Guided-OPD)、
   (iii) 目標を**スカラー**で補間する TAID 型 $\pi_0 e^{\alpha_t h_d}$、(iv) 多教師情報の別の使い方 — 幾何混合(MOD / 理論 §5.2 の E3)、マージ(TIES 等)、
   ルーティングのみ(MOPD = 本アームの control)、バグ修正済みの SFT→RL(Limozin+ 2026)。**査読者が最初に要求するのは (i)(iii) と「KL-to-base ramp」との対照**である。
4. **対立する証拠・理論は 6 種類**(§3.3): 「不一致こそ情報」(QBC、Decoupling、Co-teaching+、EnD²、TA-OPD)、「一致は共有バイアスと真実を混同する」(Eisenstein+、
   Spurious Rewards、SA-OPD)、「全会一致は保守的すぎる」(AE-KD、CAGrad、Twin-Merging)、「自然な学習順序と冗長」(Saxe、Kalimeris、Rubruck の OCS、Rethinking-OPD)、
   「カリキュラムの効果は小さく一過性」(Wu+ 2021、Saglietti+ 2022、Mannelli+ 2024、Elgaar & Amiri 2026)、そして**「教師の影響は初期に最大であるべき」**という
   2026 年の主流スケジュール(KDRL、SAF-OPD、ATOD、Guided-OPD、UFT、CBRL)。本機構は教師の**内容**を初期に最小(base + 共有成分)にして広げるので、
   教師影響の時間軸では主流の**逆**を走る(§2.5.2)。
5. **傾向は「形」を支持し、「中身」は条件付きでしか支持しない**(§4)。
   支持: 到達点不変・早期のみ・難タスクで大きいという**予測の形**は Hacohen & Weinshall 2019 の定理と経験則そのもの。Wu+ 2021 が「カリキュラムが効く」とした
   2 条件(短い予算・雑音ラベル)は本設定の両方に当たる。目標側カリキュラムの加速は Panigrahi+ (ICLR 2025 oral) が証明付きで示し、OPD の**時期**が結果を
   変えることは同設定の SAF-OPD / ATOD が示す。注入ゼロの設計判断は Twin-Merging / CaMOPD / MOPD / Two-Phase Distillation と整合。書式を**柔らかい分布目標**で
   先に与えることは、R1 の cold start・R1-Searcher の 2 段報酬・SimpleRL-Zoo / Rock Tokens の注意(報酬でも全教師 KL でもなく目標で)と整合。
   不支持: 第 1 段の目標が**書式**であることは、その成分を「spurious」と分類する SA-OPD と、低 entropy トークンは RL の利得を殆ど担わないという Beyond-80/20 に
   逆行する。多教師 OPD の失敗は勾配衝突ではなく予算配分だと Open-MOPD は主張する。教師影響の向きは主流と逆で、同じ向きの先行例(TAID / TOP-D)の根拠は
   同一 base の教師に当てはまりにくい。**結論: 機構が予測する信号は文献が予測する信号と一致するが、文献はその信号が「小さく、$\beta(t)$ や線形補間や
   KL-to-base ramp と区別しにくい」ことも予測する。** 走らせる価値は、設計 §6 の反証条件に加えて §5 の対照(R1–R3)を並べたときにだけ確定する。
6. **文献が変える優先順位**(§5): 設計 §6.4(2026-09-03 追記)は「順序と量が分離できていない」と自ら認め、$\beta$ warm-up(前半 0、step 90 で 0.01、
   cross-teacher 無し)を対照に置いた。warm-up は本機構と同じ向き(教師を後から増やす)なので**量**の対照である。文献の側からはさらに 3 つが要る:
   stage 1 の目標を $\pi_0$ に置き換えた **KL-to-base ramp**(R1、$a_3$ の**内容**を分ける)、スカラー補間(R2、合意**構造**を分ける)、
   そして主流と同じ向きの E1($\beta$ 減衰、R3)。これらを並べない限り「合意構造が順序を作る」という本機構固有の主張は検定されない。

---

## 1. 本機構をサーベイ用に抽象化する

`claude/cross-teacher-curriculum` の機構は、既存研究と突き合わせるために次の 7 性質に分解できる。以後 P1–P7 で参照する。

| # | 性質 | 出典 |
|---|---|---|
| P1 | **目標のカリキュラム。** データ順序でも損失係数でもなく、蒸留**目標分布の内容**を時間で変える | 設計 §3.3 |
| P2 | **合意 = 符号一致 + 集合 min。** 候補 $(x,v)$ ごとに on-task 教師の shift $h_d$ を「同じ符号を主張する教師の本数」で $a_3 \subseteq a_2 \subseteq a_1$ と入れ子に分解。集約は平均でなく min(3 本)・max-of-min(2 本)。閾値・deadzone 無し | 設計 §2.2、`nested_layers` |
| P3 | **到達点不変。** 最終 stage は control(on-task 教師)と bit 同一。主張は順序と速度のみ | 設計 §4.1、`curriculum_exponent` の差分形 |
| P4 | **注入ゼロ。** 全 stage で目標は候補ごとに $[\min(p_0,p_d), \max(p_0,p_d)]$ の内側。off-task 固有成分は入らない | 設計 §3.2 |
| P5 | **時間ベースの解放。** step の純関数($\rho_{pair}, \rho_{own}$ の線形 ramp、40/80、ramp 10)。習熟トリガは第 2 run 以降 | 設計 §5、`curriculum_rho` |
| P6 | **shift 空間での操作。** 対象は $h_m = \log\pi_m - \log\pi_0$、すなわち出力空間の task vector。パラメータではなく分布に対して合意を取る | 理論 §1 |
| P7 | **階層モデルによる動機。** $h_m = g + \sum_j s_{mj} + s_m + \varepsilon_m$。4 モデルから $s_d$ と $\varepsilon_d$ は識別不能なので、合意は「共有成分の大きさ」しか測れない。解放順 = 推定精度順 | 理論 §3、設計 §4.2 |

機構が**自分に課している検定**(設計 §6.2)は、(i) @300 で control と一致、(ii) 差は中間 checkpoint と学習曲線にだけ、(iii) stage 1 で webshop が保留コストを払い、
alfworld の step 41–100 の早期優位が順序仮説の証拠、の 3 つ。§4 の「支持されるか」はこの 3 つに対して判定する。
なお設計 §5.1(2026-09-03 追記)によれば stage 2 は step 50 から、全額解放は step 90 からで、150 step で走らせる場合は制限区間(89 step)が
run の 59.3% を占める(300 step なら 29.7%)。この配分は §4.3 の判定に効く。

設定そのもの(同一 base の単タスク RL 教師 ×3 → 多タスク生徒、GRPO + $\beta$·token 別 reverse KL、$\beta = 0.01$、生徒 top-20 + tail)は、
本文書ではサーベイ対象ではなく前提として扱う。ただし §2.4 で、この設定が 2026 年時点で標準化していること、および control がその標準の既定形(MOPD の routing)と
同型であることを確認する。

---

## 2. 領域別サーベイ

各表の「関係」列: **類似**(構造的アナログ)/ **競合**(同じ目的の別機構)/ **対立**(前提に反する証拠・理論)/ **支持**(前提を支える証拠・理論)/ 中立。

### 2.1 カリキュラム学習の理論 — 「順序は変わるが到達点は変わらない」の系譜(P1, P3)

| 文献 | 要点 | 関係 |
|---|---|---|
| Bengio+ 2009 ICML "Curriculum learning" | カリキュラムは収束速度と(非凸なら)局所解の質に効く。**continuation method**(平滑化した目的関数列の最後が真の目的)として読める | 支持(速度・continuation の枠組み)。ただし Bengio らは到達点が変わることを期待し、本機構は不変を主張 |
| Kumar, Packer, Koller 2010 NIPS 自己ペース学習 | 「易しい標本」を現在の仮説に相対的に選び、包含量を anneal。**学習者主導**の解放 | 類似 / 競合(固定 clock の P5 に対し学習者ペース) |
| Weinshall, Cohen, Amir 2018 ICML arXiv:1802.03796 | 凸線形回帰で、理想カリキュラムは**学習初期の収束を加速**し、最適解は不変。難しい課題ほど汎化にも効く | 支持 |
| **Hacohen & Weinshall 2019 ICML arXiv:1904.03626** | 「理想カリキュラムは緩い条件の下で**大域最適を変えない**」を定理として示し、経験的に「CL の力の大部分は学習の**初期**にある」「課題が難しいほど利得が大きい」 | **支持(本機構の予測の形そのもの)** |
| **Wu, Dyer, Neyshabur 2021 ICLR arXiv:2012.03107 "When do curricula work?"** | 数千の順序を試し、標準ベンチマークではカリキュラムの利得は**周辺的**(利得は動的な訓練集合サイズによる)。ただし**訓練予算が限られる場合と雑音ラベルがある場合**にはカリキュラム(反カリキュラムではなく)が効く。暗黙の学習順序は run 間で高度に一貫 | 一般には対立 / **本設定では条件付き支持**(300 step 予算、単 run 教師の推定雑音 $\varepsilon$ = 雑音ラベル regime) |
| Saglietti, Mannelli, Saxe 2022 NeurIPS arXiv:2106.08068 | teacher-student 解析理論。online ではカリキュラムは「**控えめに**学習を速める」。batch では標準損失で汎化利得なし、段階間を Gaussian 事前で**接続(consolidation)**したときだけ大きな利得 | 支持(控えめな加速 = 予測)/ リスク(consolidation 無しでは残らない。本機構は入れ子で $a_3$ を最後まで目標に残す点で部分的に対処) |
| Mannelli, Ivashynka, Saxe, Saglietti 2024 ICML arXiv:2406.01589 | 過剰パラメータ化はカリキュラムの利得を**制限**する(deep learning でカリキュラムが効かないことの理論的説明) | 対立(リスク) |
| Xu & Tewari 2022 ICML arXiv:2111.07126 | 多タスク線形回帰でカリキュラムの minimax 統計利得。「局所的な予測利得が最大の課題を選ぶ」正当化 | 支持(分散に基づくカリキュラムの統計的根拠) |
| Elgaar & Amiri 2026 arXiv:2601.21698 | Pythia 14M–1B の事前学習でカリキュラムは「共有された潜在フェーズ列の**各フェーズに費やす時間**を変えるだけ」で、規模が大きいほど差は小さい | 中立 / 対立(効果は「時間配分」に限られ規模で縮む) |

**読み.** 本機構の主張の形(到達点不変・初期のみ・難タスクで大きい)は、この系譜の定理と経験則に**正確に**対応する。同時にこの系譜は、その効果が
「小さく、一過性で、予算依存」であることも予測する。Wu+ の 2 条件がどちらも本設定に当てはまることは、本機構にとって最も強い「傾向からの支持」である。
設計 §6.2 が「差が出るなら control が最も弱いサブタスクに局在する」と予測しているのは、Hacohen & Weinshall の「難しいほど効く」と同じ向きである。

### 2.2 蒸留**目標**のカリキュラム — 中間目標で加速する系譜(P1, P3)

| 文献 | 要点 | 関係 |
|---|---|---|
| **Panigrahi, Liu, Malladi, Risteski, Goel 2025 ICLR (oral) arXiv:2410.05464 "Progressive distillation induces an implicit curriculum"** | 教師の**中間 checkpoint** から蒸留すると、最終教師には無い「低次成分(sparse parity の degree-1 単項式 = support)」の強い信号が暗黙のカリキュラムとして供給され、経験的加速と**証明付きの標本複雑度利得**(2 checkpoint で $\tilde\Theta(2^k d^2)$ 対 one-shot の $\Omega(d^{k-1})$ 下界)。**checkpoint の選択が決定的**で、教師の相転移中の checkpoint だけが役に立つ。Transformer/PCFG/Wikipedia でも「文脈長を段階的に捉える特徴」を確認 | **支持(最強の理論的先行例)** — ただし有効な中間目標は**課題に関係する低次構造**を担っており、書式ではない。$a_2$ 層に書式以外の共有技能が載っているかが本機構の急所(設計 §9 の「pair 層の内容 role シェアは初測定」と一致) |
| Gupta & Karmalkar 2025 arXiv:2503.17494 (ICLR 2025 WS) | **最終教師だけ**から中間目標を合成(隠れ表現のランダム射影で段階訓練)し progressive distillation に匹敵 | 類似(本機構も最終教師群から $a_k$ を合成する) |
| Jin+ 2019 ICCV RCO arXiv:1904.09149 | 教師の**学習経路上の anchor** を順に目標にする(route constrained optimization) | 類似 |
| Cho & Hariharan 2019 ICCV arXiv:1910.01348 | 早期停止した(弱い)教師の方が良い教師になる。ただし**多段の逐次蒸留は効かなかった** | 類似 + 注意 |
| Mirzadeh+ 2020 AAAI TAKD arXiv:1902.03393 | 中間サイズの teacher assistant で容量ギャップを埋める | 類似(形のみ。容量ギャップの論理は同一 base の本設定に当たらない) |
| Jafari+ 2021 EACL Annealing-KD arXiv:2104.07163 | 温度で annealed した教師出力を段階的に追う | 類似(スカラー anneal。本機構は構造化された分解に沿って anneal) |
| Li+ 2023 AAAI CTKD arXiv:2211.16231 | 学習可能な温度で easy→hard | 類似 |
| **Shing, Misaki, Bao, Yokoi, Akiba 2025 ICLR (spotlight) TAID arXiv:2501.16937** | 目標 $p_t = \mathrm{softmax}((1-t)\,\mathrm{logit}_{student} + t\,\mathrm{logit}_{teacher})$、$t$ を目的関数の改善量で**適応的に**上げる。生徒の初期分布から教師へ徐々に移す; mode collapse 回避の定理 | **類似 / 競合(LLM 規模で最も近い「時間で動く目標」の先行例)**。スカラー補間・学習者適応ペース。**本機構の自然な対照** |
| Ko+ 2024 ICML DistiLLM arXiv:2402.03898 | skew KL(生徒・教師の混合を KL 内部に) | 中立 |
| Xie+ 2026 arXiv:2607.04751 TOP-D | 「大きな目標は一度に達成しにくい」— **近接教師**を動的に構成して OPD を安定化 | 類似(中間目標を近接性で作る。本機構は合意で作る) |
| Cao, Zeng, Liu, Mokhtari 2026 arXiv:2605.11260 CLPD | データ easy→hard + 教師を容量順に切替 | 類似 |
| Jiang+ 2025 EMNLP "Teach small models to reason by curriculum distillation" | **能力を先に、明示的推論(書式)を後に**の 2 段階 | 類似だが**順序が逆**(書式先行への注意) |
| Liu & Zhang 2025 arXiv:2506.05695 POCL | データ部分集合 + 温度上昇の LLM KD カリキュラム | 類似 |
| Guo+ 2026 arXiv:2606.03532 CGTR | 自己 OPD で教師をいつ更新するか。**clock 駆動の更新は state-oblivious collapse**、報酬ゲート更新を提案 | 類似(目標のスケジュール)+ 注意(固定 step 切替の脆さ、P5) |
| Han+ 2026 arXiv:2605.11458 ATESD; Yang+ 2026 arXiv:2608.08176 USD | 特権情報の**露出量**を制御器で調節 / 生徒が吸収できる難易度予算で監督を整合 | 類似(教師信号の**内容**のカリキュラム。OPSD 系で最も近い) |

**読み.** 「同じ到達点に向かう時間変化する目標」は確立した設計軸で、LLM 規模でも TAID(ICLR 2025)と Panigrahi+(ICLR 2025 oral)が主流会議で認められている。
本機構の差分は、補間の**軸**がスカラー(温度・$t$)ではなく「何本の教師が裏付けるか」という**構造**である点にある。Panigrahi+ が同時に与える警告は重要で、
効く中間目標は「課題に関係する低次の構造」を運んでいた。監査(§11.6)が $g$ ≈ 書式と測っている以上、stage 1 は Panigrahi+ の意味での「役に立つ中間 checkpoint」
ではない可能性がある。設計が pair 層の role 別シェアを「初測定」として事前登録しているのは、この点で正しい置き方である。

### 2.3 合意・符号一致・共有成分の切り出し(P2, P4, P6, P7)

#### 2.3.1 パラメータ / 勾配空間の符号一致

| 文献 | 要点 | 関係 |
|---|---|---|
| **Parascandolo+ 2021 ICLR AND-mask arXiv:2009.00329** | 環境間で勾配の**符号が一致する割合**が閾値 $\tau$ 以上の座標だけ更新(平均 = 論理 OR、幾何平均 = 論理 AND)。合成 spiral、CIFAR 雑音ラベル、CoinRun で検証 | **類似(勾配空間で最も近い)**。2 値のマスク(min ではない)、1 課題の環境群(異なる課題ではない)、入れ子・段階なし |
| Shahtalebi+ 2021 SAND-mask arXiv:2106.02266 | AND-mask の連続緩和(方向と大きさ) | 類似(符号ゲートのソフト版) |
| Arjovsky+ 2019 IRM arXiv:1907.02893 | 環境間で不変な予測器を信頼 | 中立(精神は同じ) |
| Bernstein+ 2019 ICLR signSGD majority vote arXiv:1810.05291 | 符号だけを多数決 | 類似 |
| Chen+ 2020 NeurIPS GradDrop arXiv:2010.06808 | 課題間の**符号純度**で確率的にマスク | 類似(多数派・確率的。少数派は捨てる) |
| Chaubard, Eddy, Kochenderfer 2024 arXiv:2412.18052 Gradient Agreement Filtering | micro-batch 勾配の cosine 一致でフィルタ | 類似 |
| Yu+ 2020 NeurIPS PCGrad arXiv:2001.06782; Liu+ 2021 NeurIPS CAGrad arXiv:2110.14048; Navon+ 2022 ICML Nash-MTL arXiv:2202.01017 | 勾配衝突の除去 / 最悪課題の改善で軌道を正則化(**最適解は不変**) | 競合(勾配空間での干渉対策)。CAGrad の「軌道は変え固定点は変えない」は P3 と同型 |
| Xin+ 2022 NeurIPS arXiv:2209.11379; Kurin+ 2022 NeurIPS arXiv:2201.04122 | 多タスク最適化手法は調整済みの単純和に勝てず、多くは**正則化**として説明できる | 対立(干渉低減の利得は消えやすい。stage 1 は文字通り参照 KL 正則化) |
| Wang, Tsvetkov, Firat, Cao 2021 ICLR Gradient Vaccine arXiv:2010.05874 | 軌道上の勾配類似度は性能と相関 | 支持 |

#### 2.3.2 モデルマージにおける符号合意(重み空間の task vector)

| 文献 | 要点 | 関係 |
|---|---|---|
| Ilharco+ 2023 ICLR Task Arithmetic arXiv:2212.04089 | task vector = fine-tuned − pre-trained | 類似(P6 の定義上の祖先。$h_m$ は出力空間の task vector) |
| **Yadav+ 2023 NeurIPS TIES arXiv:2306.01708** | trim → **符号を質量で選挙** → 選挙された符号に一致するモデルだけの平均(disjoint mean が最重要と ablation)。干渉の源として「符号の不一致」を名指し | 類似(符号選挙)/ 競合(多数派・平均・全課題を注入するのが目的) |
| **Wang, Dimitriadis, Ortiz-Jiménez, Fleuret, Frossard 2024 ICML TALL masks / Consensus merging arXiv:2405.07813** | $m_{consensus} = \mathbf 1\{\sum_t m_t \ge k\}$($k=2$)。1 課題だけが使う「selfish weights」は他課題への干渉源、どの課題も使わない「catastrophic weights」を削除 | **類似(k-of-T の数え上げとして最も近い)**。ただし 2 値の support マスクで、課題固有の重みを**削除**する — 本機構は最終 stage で全額**復元**する(逆向き) |
| Kim, Han, Choe 2025 ICML NegMerge arXiv:2410.05583 | **全 task vector で符号が一致**する要素だけ保持(平均)。一致 = forget set 由来、不一致 = 雑音と読む。ただし task vector は**同一課題のハイパラ違い**の複製 | 類似($a_3$ 型の全会一致)。同一課題の複製だから全会一致が信号/雑音を分離できる — 異課題教師ではそれが使えないと理論 §3.2 が示している点で示唆的 |
| Lu+ 2024 NeurIPS Twin-Merging arXiv:2406.15479 | 共有知識 + 排他的知識に分解。「**排他的知識を直接マージすると性能を損なう**」 | 支持(P4 の注入ゼロと webshop の記録に整合) |
| Yan+ 2025 ICML CALM arXiv:2506.13406 | consensus-aware な局所マスク | 類似 |
| Gargiulo+ 2025 CVPR TSV arXiv:2412.00081; Marczak+ 2025 ICML Iso-CTS arXiv:2502.04959 | 共通部分空間と課題固有部分空間の分離 | 類似(符号ではなく部分空間) |
| Yu+ 2023 DARE arXiv:2311.03099; Davari & Belilovsky 2024 ECCV Breadcrumbs arXiv:2312.06795; Du+ 2024 NeurIPS PCB arXiv:2410.02396 | 合意を使わないマージ | 中立 |

#### 2.3.3 logit shift の算術(出力空間の task vector)

| 文献 | 要点 | 関係 |
|---|---|---|
| Liu+ 2021 ACL DExperts | expert / anti-expert の logit を加算(product of experts) | 類似 / 競合(加法混合) |
| Liu+ 2024 COLM Proxy-tuning arXiv:2401.08565 | $\log\pi_{tuned} - \log\pi_{base}$ を大きな base に加える | 類似(P6 の primitive) |
| Mitchell+ 2023 EFT arXiv:2310.12962 | log-prob 代数 base + (chat − base)。KL 正則化 RL の帰結として fine-tune は暗黙の報酬 $\propto \log\pi_{ft}/\pi_{base}$ を学ぶ | 類似(目標 $\pi_0 e^{a_k}$ は EFT 代数のゲート付き shift) |
| Shi+ 2024 NeurIPS MOD arXiv:2406.18853 | f-divergence で整合した複数モデルの**線形結合**を閉形式で | 競合(理論 §3.3 の Bayes 最適な幾何混合そのもの。合意ゲート無し) |
| Fan+ 2024 NeurIPS Dynamic Logits Fusion arXiv:2406.15480 | 複数の小型 expert の logit 差を step ごとの KL 制約重みで大型 base に融合 | 類似(複数 expert の delta 合成。訓練目標ではなく decode 時) |
| Zhou+ 2024 NeurIPS Weak-to-Strong Search arXiv:2405.19262 | tuned/untuned の log 比を報酬に探索 | 類似 |
| Feng+ 2026 arXiv:2607.05394 Direct-OPD | 「post-RL 教師と pre-RL 参照の log 比を dense な暗黙報酬として生徒に」 | 類似(同じ primitive、単一教師) |
| Li+ 2026 arXiv:2606.04378 DLLG | 複数 expert の logit を学習ゲートで融合 | 競合 |
| Zeng+ 2025 arXiv:2510.13855 CoRE | test-time ensemble を一貫性で重み付け | 類似(合意ゲート。base 相対 shift ではない) |

**判定(2.3.3).** base 相対 logit shift を**複数 expert の符号一致でゲート**する先行例は見つからなかった。多 expert の logit 合成は加法(DExperts、MOD、proxy-tuning)、
学習(DLLG)、perplexity(PackLLM arXiv:2404.11531)、一貫性(CoRE)のいずれかで、符号ゲート付き OPD は SG-OPD(§2.4)が教師×検証器で行い、多教師は future work としている。

#### 2.3.4 「一致を信じる」対「不一致から学ぶ」

| 文献 | 要点 | 関係 |
|---|---|---|
| Zhou & Li 2005 TKDE Tri-training | 他の 2 分類器が**一致**したらラベル付け(2-of-3)。一致しても誤るケースを**ラベル雑音として陽に扱い**(Angluin–Laird)、追加標本で補償できる条件を与える | 類似($a_2$ 型の合意を信頼、雑音の会計付き) |
| Wei+ 2020 CVPR JoCoR arXiv:2003.02752 | 2 網の**一致**を清浄ラベルの信号として co-regularize | 類似 |
| Seung, Opper, Sompolinsky 1992 COLT Query by Committee | **最大不一致**の点を問い合わせる | **対立** |
| Malach & Shalev-Shwartz 2017 NeurIPS Decoupling arXiv:1706.02613 | 2 予測器が**不一致のときだけ**更新。同一初期化なら一度も更新しない | **対立**(一致 = 学ぶものが無い) |
| Yu+ 2019 ICML Co-teaching+ arXiv:1901.04215 | Co-teaching は合意に収束して自己訓練に退化するので、**不一致データだけ**を残す | **対立** |
| Malinin+ 2020 ICLR EnD² arXiv:1905.00076 | ensemble の多様性(不一致)は情報であり、点推定に潰すと失われる | 対立 |
| Wang+ 2026 arXiv:2605.26844 TA-OPD | OPD の不一致を「学べる不一致」と「両立しない不一致」に分け、**teachability の高いトークンだけ**で学ぶ | 対立(一致でなく両立性でゲートせよ) |

**読み.** この対立は文献内で明示的で、Co-teaching+ と JoCoR は同じベンチマークで正面からぶつかっている。本機構は「一致 = 信頼」側ではなく
「一致 = **共有**」側に立ち(P7、理論 §3.2 は「信頼」の読みを自ら否定している)、不一致トークンを捨てるのではなく**後回し**にする。
したがって対立の実体は「不一致トークン(= on-task 固有分 $\hat s_d$)を stage 1–2 で遅らせるコストが、順序の利得を上回るか」であり、
これは設計 §6.2 の反証条件(webshop の 2 SE 低下)そのものである。

#### 2.3.5 多教師蒸留における合意

| 文献 | 要点 | 関係 |
|---|---|---|
| You+ 2017 KDD Learning from multiple teacher networks | 教師出力の平均 + 中間層の相対非類似度 | 中立 |
| **Du+ 2020 NeurIPS AE-KD "Agree to Disagree"** | ensemble KD を多目的最適化とし、MGDA で**全教師が同意する降下方向**(Pareto)を取る。許容度 $C \in [1/M, 1]$: $C=1$(全会一致、不一致を許さない)は「弱い・雑音の多い教師の方向も下げねばならず不必要」で、「**全教師の方向を保守的に受け入れるのも良い選択ではない**」。$C \in (1/M, 1)$ が最良 | 類似(全会一致の降下方向)+ **対立(純粋な全会一致は準最適)** |
| Zhang+ 2022 ICASSP CA-MKD arXiv:2201.00007; Liu+ 2020 AMTML-KD arXiv:2103.04062; Yuan+ 2021 AAAI RL 教師選択 arXiv:2012.06048 | ラベル・学習表現・RL による教師重み | 中立 |
| Zhou+ 2024 EMNLP Industry GOVERN arXiv:2405.03764 | 教師 logit と**生徒** logit の勾配方向の**多数決**で一致する教師だけ使う。ラベル不要 | 類似(教師間の符号投票。多数派・生徒相対・少数派は捨てる) |
| Li+ 2025 ACL Findings FAIR arXiv:2410.03663 | 教師 LLM 間の**査読**で採択された rationale だけ蒸留 | 類似(教師間の裏付けフィルタ) |
| Sumit+ 2026 arXiv:2604.03192 EWAD | token ごとの gate $1 - \mathrm{JSD}(p_{T1}\Vert p_{T2})/\log 2$。一致→KD、対立→正解 CE にフォールバック。結果は弱い | 類似(token 別の教師間一致ゲート) |
| Wang+ 2026 arXiv:2605.01347 MAD-OPD | 教師群の**討論**による合意を目標、事後信頼で重み付け。agentic にも拡張 | 類似(討論で作る合意) |
| Jin+ 2026 arXiv:2602.01064 | 多教師 KD の knowledge purification 5 法。合意分解は無し | 中立 |

#### 2.3.6 ensemble と悲観、共有 vs 固有の分解

| 文献 | 要点 | 関係 |
|---|---|---|
| Coste+ 2024 ICLR arXiv:2310.02743 | 報酬モデル ensemble の **min**(WCO)/ mean − λ·var(UWO)で overoptimization を抑制。「1 つでも過大評価しなければ」の保証は**同じ推定対象**だから成り立つ | 類似(min = 悲観的集約)。異課題教師の min は別の推定対象(共有成分)なので保証は移らない |
| **Eisenstein+ 2023 arXiv:2312.09244 "Helping or Herding"** | 同じ base を共有する報酬モデル群は**似た誤りパターン**を示し、ensemble でも reward hacking は消えない | **対立**(一致は共有バイアスと真実を混同する) |
| Mukherjee+ 2025 arXiv:2505.11711 | RL は 5–30% の小さな部分網だけを更新し、**seed・データ・アルゴリズムが違っても部分網の重なりは偶然より大幅に大きい** | 支持(全会一致成分が存在する機構的根拠)/ 注意(重なりの一部は base の事前傾向) |
| Shao+ 2025 arXiv:2506.10947 Spurious Rewards | Qwen2.5-Math ではランダム報酬でも正解報酬に迫る(MATH-500 で +21.4 対 +29.1)。RL の shift の大部分は事前学習から浮上した**共有スタイル** | 支持($a_3$ ≈ 書式)/ 対立(書式は信頼性でも課題知識でもない) |
| Shenfeld, Pari, Agrawal 2025 arXiv:2509.04259 RL's Razor | on-policy RL は KL 最小解に偏る | 中立 / 支持(教師 shift は小さい編集) |
| Evgeniou & Pontil 2004 KDD; Jalali+ 2010 NeurIPS dirty model | $w_t = w_0 + v_t$ の共有+固有分解 / 「一部の課題だけが共有する」成分 | 類似(P7 の古典形。dirty model の「一部で共有」= $a_2$ 層) |
| **Teh+ 2017 NeurIPS Distral arXiv:1707.04175** | 「全課題に共通する振る舞いを捉える蒸留方策」を共有し、各課題方策は $\pi_i \propto \pi_0^\alpha e^{\beta A_i}$ の形で KL 正則化。共有方策は課題方策の**重心(centroid)**として蒸留学習。動機は「課題間の勾配干渉と報酬の衝突」 | 類似 / 支持(stage 1 = 固定重心版 Distral。共有×固有の乗法形が同じ)。差: 重心(平均)ではなく交差(min)、事前分布ではなく目標そのもの |

**読み.** 本機構の合意 primitive は、重み空間ではマージ研究(TIES → Consensus → NegMerge)が 2023–2025 年に確立したものと同型で、それを出力空間の shift(P6)に移した形になる。
マージ研究の到達点は「共有成分と排他的成分を分け、排他的成分を他課題に混ぜない」(Twin-Merging、Consensus)であり、P4 の注入ゼロはその結論と一致する。
一方で AE-KD と Eisenstein+ は、全会一致を**そのまま**目標にすることの 2 つの限界(保守的すぎる、共有バイアスを拾う)を示しており、本機構がそれを
「一時的にしか使わない」ことで回避しているかどうかは、スケジュール(P5)の妥当性に帰着する。

### 2.4 LLM の on-policy 蒸留、多教師・専門家→汎用、多タスク RL(設定と control の位置)

#### 2.4.1 設定の標準化(2025–2026)

| 文献 | 要点 | 関係 |
|---|---|---|
| Agarwal+ 2024 ICLR GKD arXiv:2306.13649; Gu+ 2024 ICLR MiniLLM arXiv:2306.08543 | 生徒生成系列上の教師フィードバック / reverse KL の on-policy 最適化。GKD は RL との統合を明示 | 支持(目的関数族) |
| Thinking Machines OPD blog(Lu, 2025-10-27) | token 別 reverse KL を advantage に。RL の 7–10 倍速。「token 別蒸留報酬と系列水準の環境報酬の組合せは future research」。多教師の議論無し | 支持 |
| Qwen3 tech report arXiv:2505.09388 | off-policy → on-policy の strong-to-weak 蒸留、RL の 1/10 GPU 時間 | 支持 |
| Xu+ 2025 KDRL arXiv:2506.02208 | $J_{GRPO} - \beta\,\mathrm{KL}^{k2}$。top-k 近似は不安定。**$\beta$ を線形減衰(5e-3→1e-3)させたものが最良**:「強い KD は初期に有効だが、高い $\beta$ を維持すると継続的改善を妨げる」 | 支持(設定)/ **競合($\beta(t)$ anneal がまず効く)** |
| **Ding+ 2026 arXiv:2607.29209 SAF-OPD(Meituan)** | 固定係数の GRPO+OPD は entropy collapse。原因は magnitude mismatch と **temporal mismatch(全強度の OPD が生徒を教師に引き続ける)**。sparsify-then-compress + **warm-up-then-anneal**。Qwen3-1.7B/4B/8B | **競合(自然な対照)**+ 支持(時期が結果を変える) |
| **Tan+ 2026 arXiv:2606.27814 ATOD** | **本機構と同じ ALFWorld / WebShop / Search-QA** で、「OPD が初期を支配し RL を徐々に強める」anneal + turn 別の不一致・不確実性重み。OPD +4.16、GRPO +23.62、教師 +2.16 | **競合(同一ベンチマークで係数 anneal が既に順序の利得を出している)**+ 支持(早期 OPD / 後期 RL の粗い順序) |
| Yang+ 2026 arXiv:2602.12125 G-OPD / ExOPD | OPD = 参照が任意の dense KL 制約 RL。**同一 base のドメイン別 RL 専門家を base に戻す**設定で、$\lambda = 1.25$ の外挿で教師を超える(math 48.0 対 46.0)。参照に教師の pre-RL base を使う「reward correction」= 教師の RL が誘導した暗黙報酬 | 類似(同じ primitive・同じ設定)/ 競合(合意分解ではなく shift 全体をスカラー倍。「外挿より段階解放が良い理由は」と問われる) |
| Li+ 2026 arXiv:2604.13016 Rethinking OPD | OPD 成功の条件は思考パターンの両立と教師の新規能力。機構は「生徒訪問状態での**高確率トークン上の漸進的整合**」で、共有トークン集合が 97–99% の質量を持ち、重なりは 72%→91% に上がる | 支持(OPD 初期 = 共有高確率集合の整合 → $a_3$ ≈ 書式と整合)/ **対立(明示的な段階付け無しでも OPD は共有集合から揃う → 冗長の可能性)** |
| Liu+ 2026 arXiv:2605.13230 TGPO | 教師・生徒の乖離が大きいと RL 探索が教師分布外に出て負のフィードバックが無情報 | 支持(off-task 教師を他課題の状態で読むことへの注意。ただし本機構は off-task をゲートにしか使わない) |
| Song & Zheng 2026 arXiv:2604.00626; Zhang 2026 arXiv:2606.22793(OPD サーベイ) | 設計軸の分類に**多教師も目標カリキュラムも項目が無い** | 中立(不在が情報) |
| Lin+ 2026 arXiv:2608.24696 OPDVR; Akhondzadeh+ 2026 arXiv:2607.04037 RG-OPD | 正しさで OPD 報酬を ReLU ゲート / 検証器が教師を信じる時を決める。「重み付き結合や heuristic な切替は余分なハイパラ」 | 競合(外部基準でゲート。合意ではない) |
| Zhao+ 2026 arXiv:2601.18734 OPSD | 特権文脈の自己教師。forward KL が最良。**token 別 KL clipping は「文体トークンが学習信号を支配するのを防ぐ」** | 中立 / 注意(文体トークン支配の観察) |
| Wang+ 2025 arXiv:2506.01939 Beyond the 80/20 Rule | RLVR の利得は高 entropy の少数「分岐」トークン(≈20%)が担い、低 entropy の 80% だけで訓練すると劣化 | 中立 / 対立($a_3$ ≈ 低 entropy 書式なら stage 1 の利得は小さい) |
| Jiang+ 2026 arXiv:2608.03632 SA-OPD | 「入力非依存の言語事前・**書式慣習**・定型推論テンプレート」による教師判断を spurious とし、大きな勾配を出すが task 改善方向を殆ど持たないとしてフィルタ | **対立**(stage 1 の目標は経験的に SA-OPD が spurious と呼ぶ成分) |

#### 2.4.2 多教師 OPD(本アームの直近の隣人)

| 文献 | 要点 | 関係 |
|---|---|---|
| **Ma+ 2026 arXiv:2606.30406 MOPD(Xiaomi MiMo-V2-Flash arXiv:2601.02780)** | ドメイン別 RL 教師を作り、**各軌跡をそのドメインの教師に routing**、token 別 reverse KL(top-k 64)。Mix-RL 0.882、Param-Merge 0.857 に対し 0.937。「**同一 base の教師が蒸留を安定化**」(Qwen3-235B 教師は KL 5 倍で悪化)。カリキュラム無し | 類似(教師の作り方が同一)/ **競合(routing のみ = 本アームの control)** |
| Gao+ 2026 arXiv:2608.19098 Open-MOPD | oracle routing の M-OPD は headroom の 35.6% しか回収しない。原因は「**勾配衝突ではなく token 別最適化予算の深刻な誤配分**」(長さ差・収束 drift・報酬 staleness)。修正で 83.4% | **対立(干渉説)**/ 支持(多教師 OPD の動学には大きな改善余地) |
| Chen+ 2026 arXiv:2605.27115 CaMOPD | 汎用教師とドメイン教師で「回復と保持の勾配が逆向きに打ち消し合う」(**cross-domain 勾配内積が持続的に負**)。交互訓練で解消 | 支持(教師間の衝突は実在し、教師を**時間で分ける**のが解) |
| Lu+ 2026 arXiv:2607.27770 Beyond the Best Teacher | 同一 base の複数 RL 教師を作り「信頼できる教師だけ寄与」で圧縮 | 類似 |
| Xie+ 2026 arXiv:2608.03610 LS-MOPD | 言語別 RL 教師の routing + token 別多教師蒸留。生徒が最良教師の envelope を超える | 類似 |
| Shen+ 2026 arXiv:2607.07050 | routing 2 教師の OPD で教師 top-32 が決定的トークンを 0.4% しか含まない。student-aware support で解決 | 中立 / 注意(生徒 top-k + tail 推定器の既知の穴。C7 撤回で部分的に対処済み) |
| **Wu+ 2026 arXiv:2608.27409(Tencent/Fudan)fusion paradigms** | Merge / Mix-RL / MOPD はドメイン平均で**約 1.4pt** しか違わない(単一ベンチで最大 8.6)。MOPD は教師性能に制約、Merge は圧縮効果 | 競合(「マージか Mix-RL でよい」)。速度だけを主張する機構は、この研究が測っていない軸で勝つ必要がある |
| Yuan+ 2025 arXiv:2510.02227 AMPO | on-policy が失敗した時だけ複数教師の off-policy 誘導 | 中立 |

#### 2.4.3 ALFWorld / WebShop / Search 上の agent OPD とカリキュラム

| 文献 | 要点 | 関係 |
|---|---|---|
| Li+ 2026 arXiv:2606.15912 Guided-OPD | 教師の turn 介入確率を**0 まで減衰**する curriculum。ALFWorld/ScienceWorld/WebShop、+21.1% | 類似(同ベンチ。誰が生成するかのカリキュラム) |
| Wang+ 2026 arXiv:2604.24005 TCOD | 生徒に見せる軌跡**深さ**を短→長に。+18、教師超え | 類似(深さのカリキュラム) |
| Chen+ 2026 arXiv:2608.01953 FutureBridge-OPD | 高不一致状態での教師「橋」を将来の正信号密度で検証。Qwen3-32B→1.7B | 類似 |
| Feng+ 2025 NeurIPS GiGPO arXiv:2505.10978; verl-agent; Xi+ 2025 AgentGym-RL arXiv:2509.08755(interaction horizon の curriculum); Jin+ 2025 Search-R1 arXiv:2503.09516 | ベンチと backbone の定義 | 中立。この系列のカリキュラムは horizon / 深さ / 介入で、**目標の内容**には無い |
| Lu+ 2026 arXiv:2605.15155 SDAR(本 repo) | 特権文脈(skill)を与えた**同一モデル**を教師とする OPSD を、sigmoid ゲート付き補助目的として GRPO に足す($\lambda = 0.01$、$\beta_{gate} = 5$)。Qwen2.5 3B/7B・Qwen3-1.7B、同 3 ベンチ。多教師は無し | 類似(同 backbone・同係数。監査 §13.4 の通り書式交絡が構造的に無い) |

#### 2.4.4 古典的な多タスク RL 蒸留と、多ドメイン LLM RL の干渉

| 文献 | 要点 | 関係 |
|---|---|---|
| Rusu+ 2016 ICLR Policy Distillation arXiv:1511.06295; Parisotto+ 2016 ICLR Actor-Mimic arXiv:1511.06342 | 単タスク教師群 → 多タスク生徒。蒸留した生徒は教師と同時訓練 DQN を上回る | 支持(専門家→汎用の原型) |
| Ghosh+ 2018 ICLR Divide-and-Conquer RL arXiv:1711.09874 | 局所方策を KL 制約で中心方策に統合 | 支持 |
| Li+ 2025 arXiv:2507.17512 "Can One Domain Help Others?" | Qwen2.5-7B の GRPO で Math+Puzzle は math を改善するが code を**両単独以下に**落とす。難易度カリキュラム + policy refresh(参照を最新 actor に)が混合訓練に勝つ | 支持(多ドメイン GRPO の衝突は実在。段階 + 参照更新が混合に勝つ) |
| Li+ 2025 arXiv:2507.14783 Omni-Thinker | 課題を backward transfer で順序付け、**structured → open-ended** の順で同時訓練比 +6.2、マージ比 +12.4 | 支持(順序が効く。「構造的なものが先」は書式先行と同じ向き) |
| Yang+ 2026 arXiv:2606.25178 TAC | 「他ドメインに広く利益をもたらすドメイン」を**勾配整合**で優先する bandit カリキュラム。Qwen3-1.7B | 類似(クロスドメインの勾配一致をカリキュラム信号に。勾配空間版の P2) |
| Yang, Ding, Xiong 2026 arXiv:2606.02398 | 全モデル勾配が直交でも「低次元の**共有衝突部分空間**」で干渉。逐次訓練 + 短い refresh で回復 | 支持(干渉は実在し共有部分空間に局在) |
| Cheng+ 2025 arXiv:2506.14965 Guru; Akter+ 2025 arXiv:2504.13941 Nemotron-CrossThink; Ramesh+ 2026 arXiv:2602.05547 MT-GRPO | 転移はドメイン依存 / 相乗 / 不均衡 | 中立 |

#### 2.4.5 KL の参照を動かす

| 文献 | 要点 | 関係 |
|---|---|---|
| Liu+ 2025 arXiv:2505.24864 ProRL; arXiv:2507.12507 | 参照方策を最新 snapshot に**ハードリセット** | 支持(KL の参照は訓練中の制御変数) |
| Gorbatovski+ 2024 arXiv:2404.09656 TR-DPO | 参照を soft/hard に更新 | 類似 |
| **Ackermann+ 2026 arXiv:2602.18037** | 参照リセットは勾配正則化として働き、「**$\beta$ を減衰させても参照リセットには及ばない**」 | 支持(KL が**何を指すか**は $\beta$ のスケールと別の軸 → P1 は $\beta(t)$ と別の実験軸である) |
| Liu+ 2025 arXiv:2510.01555 | k1 報酬 / k2 損失が原理的、k3 損失(GRPO 流)は 1 次の偏り近似 | 中立 |

### 2.5 LLM-RL のカリキュラム — データ難易度・誘導減衰・KL スケジュール・書式先行(P1, P3, P5)

#### 2.5.1 データ / 難易度カリキュラム(RLVR)

| 文献 | 要点 | 関係 |
|---|---|---|
| Chen+ 2025 SEC arXiv:2505.14970; Wang, Cui+ 2025 DUMP arXiv:2504.09710 | 問題カテゴリ / 分布に対する非定常 bandit。**絶対 advantage** を学習可能性の代理に | 中立(データ側。$\lvert A\rvert$ を学習可能性信号として確立) |
| **Parashar+ 2025 E2H Reasoner arXiv:2506.06632(ICLR 2026)** | **step ベース**の確率スケジューラ(cosine/Gaussian)で易→難へ。1.5–3B の素の RL は難問で失敗、カリキュラムで学ぶ。理論: CRL は直接学習より少ない標本で済む。「易しい段階を**適切に減衰させること**が過学習防止に不可欠」 | 支持(小モデル・時間ベース・早期段階の消滅、標本複雑度の論理) |
| Bae+ 2026 EACL Online Difficulty Filtering arXiv:2504.03380 | 期待方策改善は課題成功確率の**分散**で下から抑えられる。均衡フィルタで「半分の step で +12%」 | 中立(pass rate ≈ 0.5 の理論) |
| Foster+ 2025 NeurIPS LILO arXiv:2502.12272; Shi+ AdaRFT arXiv:2504.05520; Jiang+ VCRL arXiv:2509.19803; Gao+ PCL arXiv:2510.01135; Kong+ CDAS arXiv:2505.17652 | 成功分散最大 / 「難しいが解ける」/ 群報酬分散 / 価値モデルで中難度 / 難易度と能力の固定点整合。いずれも**習熟トリガ** | 中立 |
| Yu+ 2025 DAPO arXiv:2503.14476 | 全正・全誤のプロンプトを除く動的サンプリング | 中立(暗黙の $p \in (0,1)$ カリキュラム) |
| Gu+ 2026 Actor-Curator arXiv:2602.20532; Zheng+ 2026 METIS arXiv:2605.11235; Sundaram+ 2026 SOAR arXiv:2601.18778 | 学習した curator / 方策自身が難易度を予測 / 教師が学習可能性の縁で問題生成 | 中立 |
| Cai+ 2026 Boundary-aware CRL arXiv:2606.22317 | pass@k で能力境界を測り、境界付近に**教師誘導**、最後に RL で定着 | 類似(習熟トリガ、教師誘導 → 純 RL) |
| Liu+ 2026 SC-SDPO arXiv:2605.27765 | GRPO の群相対 advantage は中難度に自然に集中するが、**KL ベース(自己蒸留)の advantage には難易度の概念が無い** | 支持 / 中立(token 別 KL-to-teacher にも難易度順序は内在しない → 外から順序を与える理由) |
| Xi+ 2024 ICML R3 arXiv:2402.05808 | demonstration の開始点を末尾から先頭へ滑らせる逆カリキュラム | 類似(与える demonstration の量を段階的に減らす) |
| Zhang+ SPEED-RL arXiv:2506.09016 | **著者により撤回(2026-03、実験にバグ)** | 引用不可(効率主張の脆さの例) |
| Yu+ 2026 ACL survey arXiv:2604.17312 | データ稀少下の RL。カリキュラムと「一貫性・合意機構」の節 | 中立 |

**読み.** RLVR のデータカリキュラムは「成功分散 $p(1-p)$ が高い所を出す」で収束しており、検証済みの利得は**標本効率と小モデルの救済**で、最終精度の利得は控えめ。
本機構と直交する(どのプロンプトを出すかではなく目標のどの成分を出すか)が、3 点を借りられる: (a) カリキュラムは RLVR で**速度機構**として受け入れられている、
(b) step ベースで早期段階が消えるスケジュールは正当(E2H)、(c) KL 項には難易度順序が内在しない(SC-SDPO)ので、順序を外から与えることに理由がある。

#### 2.5.2 誘導減衰カリキュラム — 教師の助けを前半に厚く、後半に消す

| 文献 | スケジュール | 到達点 | 関係 |
|---|---|---|---|
| Yan+ 2025 NeurIPS LUFFY arXiv:2504.14945 | off-policy 軌跡 1 本 + on-policy 7 本を**全 500 step 一定**。「表面的パターンに固着する」危険を警告 | 誘導は消えない | 競合(減衰しない教師誘導の標準 baseline) |
| **Liu, Farina, Ozdaglar 2025 UFT arXiv:2505.16984** | hint 割合 $p(t)$ を cosine で **step 300 までに 0** へ。理論: 長 horizon 推論で指数的加速 | 純 RL | 類似(hint 系で最も近い時間ベース・消滅型) |
| **Agashe+ 2026 CBRL arXiv:2603.18953** | few-shot demonstration の注入確率を高く始めて **0 へ anneal** | 純 RLVR | 類似(最も素直な時間ベース・消滅型) |
| Prefix-RFT arXiv:2507.01679 | prefix 比の下限を cosine で 0.95→0.05(step 500) | ほぼ純 RL | 類似 |
| **Zhang+ 2025 CHORD arXiv:2508.11408** | SFT 係数 $\mu$ を 0.9→0.05(最初の 200 step)で減衰。token 別重みがあれば減衰スケジュールは必須でない | ≈ 純 RL | 類似 / 競合 |
| Zhang+ 2025 NeurIPS BREAD arXiv:2506.17211; Amani+ AdaBack arXiv:2506.18110; Li+ SEELE arXiv:2509.06923; Ma+ ReLIFT arXiv:2506.07527; Lv+ HPT arXiv:2509.04419 | 失敗時だけ短い expert prefix / 標本別に監督長を報酬で調整 / IRT で hint 長 / 未解決問題だけ SFT、進むにつれ RL / 成績ゲートで SFT↔RL | 純 RL(漸近) | 類似(**習熟トリガ**) |
| **Fu+ 2026 ICLR SRFT arXiv:2506.19767** | entropy 適応の SFT/RL 重み。知見: 「**SFT は方策分布に粗い大域的変化を、RL は細かい選択的最適化を**」 | — | 支持(教師 = 粗い共有 shift、報酬 = 細かい内容、という分業) |
| Setlur+ 2026 PrefixRL arXiv:2601.18795 | 成功した very-off-policy 軌跡の prefix に条件付け、prefix 無しへ「back-generalization」 | 純 RL | 類似 |
| Xu+ 2025 KDRL arXiv:2506.02208 | $\beta$ 線形減衰(5e-3→1e-3)が最良 | KL は残る | 競合(§2.4.1) |
| **Limozin, Durech, Hoefler, Schlag, Pyatkin 2026 arXiv:2604.23747 "SFT-then-RL Outperforms Mixed-Policy Methods"** | 混合方策論文の SFT baseline を弱めていた 2 つのバグ(optimizer の micro-batch 脱落、損失集計)を修正すると、**SFT→RL が評価した全混合方策法に勝つ**(Qwen2.5-Math-7B +3.8、Llama-3.1-8B +22.2) | — | **対立 / 注意**(「前半に混ぜて後で解放」型の利得は、バグの無い段階型 baseline と比べて初めて主張できる) |

**読み.** 2026 年の主流は「**教師の助けは最初に最大、時間とともに 0 へ**」で、固定 step のスケジュール(UFT、CBRL、Prefix-RFT、CHORD、§2.4.3 の Guided-OPD)と
習熟トリガ(BREAD、AdaBack、SEELE、ReLIFT、HPT)の 2 系統がある。「到達点 = 純 RL(= 誘導無しの目的関数)」は時間ベース系の標準なので、P3(到達点 = control)と
P5(step の純関数)は**形式として慣例的**である。

ただし**向きが逆**である点は明示しておく必要がある。誘導減衰系は教師の影響を**最大から減らす**。本機構は $\beta$ 一定のまま、目標の内容を
「base + 共有成分」(生徒の初期分布に最も近い)から on-task 教師へ**広げる**ので、教師の**内容**の影響は時間とともに**増える**。生徒が base 近傍にいる stage 1 では
KL 項の勾配はほぼゼロで、実質的に「GRPO + 書式の参照 KL」(設計 §3.2-4)である。つまり教師影響の時間軸では、本機構は KDRL / SAF-OPD / ATOD / Guided-OPD /
UFT / CBRL の**逆**を走る。同じ向きの先行例は TAID / TOP-D(目標が生徒側から教師側へ動く)だが、その根拠(容量ギャップ・mode collapse・大きな乖離下の安定化)は
同一 base の教師には当てはまりにくい。本機構が根拠にするのは P7 の**精度順序**(共有成分は低分散、固有成分は $\varepsilon$ を含む)であり、これを
**時間**スケジュールとして使った先行例は無い(pool による低分散推定の論理としては Du+ 2021 / Tripuraneni+ 2021、構造としては Distral の共有方策が最も近い)。
理論文書 §5.3a の E1($\beta(t)$ 減衰)は主流と同じ向きの機構で、設計 §6.4 が言う通り両者は「時間で何を変えるか」が違う。**主流の傾向は E1 を、本機構をではなく、まず支持する。**
設計 §6.4 の追記(2026-09-03)は同じ論点を「順序と量が分離できていない」として自ら認め、$\beta$ warm-up アーム(前半 0 → step 90 で 0.01)を対照に置いた。
warm-up は本機構と**同じ向き**(教師を後から増やす)なので「量」の対照であり、主流の向きの E1 とは別物である。両方があって初めて「向き」と「内容」が分かれる。

#### 2.5.3 KL の係数・参照のスケジュール

| 文献 | 要点 | 関係 |
|---|---|---|
| Liu+ 2025 NeurIPS ProRL arXiv:2505.24864; Noukhovitch+ 2023 NeurIPS Elastic Reset arXiv:2312.07551; Gorbatovski+ 2024 TR-DPO arXiv:2404.09656 | 参照方策のハードリセット(検証停滞で)/ EMA へのリセット / soft・hard 更新($\alpha = 0.9$ や $\tau < 16$ は不安定) | 類似(動く参照。習熟 / 時間トリガ) |
| **Liu, Liu, Cohan 2024 arXiv:2407.13709** | 「強い参照方策は性能を上げるが、**fine-tune するモデルに似ているときだけ**」 | 支持(生徒から遠い目標は害。base + 共有成分から始める根拠) |
| Aminian, Asadi, Shenfeld, Mroueh 2025 NeurIPS arXiv:2502.01203 | **複数の参照モデル**への reverse KL 正則化の厳密解と標本複雑度 | 類似(複数参照。集約は重みで、合意ではない) |
| Liu+ 2025 arXiv:2510.01555; He+ 2026 arXiv:2602.11523 | k1 報酬 / k2 損失が原理的、k3 は偏り近似 / $\pi_0$ と $\pi_t$ への正則化のトレードオフ | 中立 |
| Zhang+ 2026 ICLR Co-rewarding-II arXiv:2508.00410 | ゆっくり更新される参照教師で自己蒸留 | 類似(動く教師目標) |
| Yang+ 2026 G-OPD arXiv:2602.12125 | 最適解は $\log\pi_\theta = \lambda\log\pi^* + (1-\lambda)\log\pi_{ref}$、$\lambda = 1.25$ 固定 | 類似 / 競合(§2.4.1) |

「書式だけの参照に強く正則化してから解放する」ことの直接の証拠は**見つからなかった**。最も近いのは §2.5.4 の cold-start / 書式の知見である。

#### 2.5.4 書式先行(format-first)の証拠

| 文献 | 要点 | 関係 |
|---|---|---|
| DeepSeek-AI 2025 R1 arXiv:2501.12948 | R1-Zero は書式報酬でタグ内推論を誘導。可読性・言語混在の問題を **cold-start SFT** で先に解決してから RL | 支持(慣行として書式 / 文体を RL の前か並行に監督で固める) |
| **Liu+ 2025 "Understanding R1-Zero-Like Training" arXiv:2503.20783** | Qwen2.5 はテンプレート無しが最良。「**テンプレートを課すと能力が壊れ、RL がそれを再構築する**」。テンプレートは初期方策の性能を決めるが RL は全方策を同程度まで上げる | 支持(書式不整合は RL が初期に予算を割く相。書式を先に与えればその相が消える) |
| **Zeng+ 2025 SimpleRL-Zoo arXiv:2503.18892** | 厳格な書式**報酬**は探索を妨げ性能を落とす。base モデルは初期に書式に従えず、書式報酬は正しい探索も罰する。弱いモデルは早期に応答長が伸びるだけで性能は上がらない | 両義(書式は早期のボトルネック / **報酬**で教えると害 → 柔らかい分布目標で教える stage 1 の**形**を支持) |
| **Song+ 2025 R1-Searcher arXiv:2503.05592** | stage 1 の報酬 = 検索呼び出し + **書式のみ**(正解性を見ない)、stage 2 = 回答 F1 + 書式罰。search-QA で | **支持(同ドメインで明示的な「書式 → 正解性」2 段カリキュラム)** |
| He+ 2025 SAKE arXiv:2505.15062 | 固定の加法的書式報酬。3B は 7B より重み大(「構造遵守の強い anchor が要る」)。スケジュール無し | 中立(小モデルには書式 anchor が要る) |
| Shao+ 2025 Spurious Rewards arXiv:2506.10947 | ランダム報酬でも Qwen2.5-Math で正解報酬に迫る。モデル族依存 | 支持 / 対立(§2.3.6) |
| Akgül+ 2026 arXiv:2605.06241 | 「RL は新しい戦略を教えず質量を再配分する」。影響は token 位置の 1–3%、昇格 token は base の top-5 内 | 支持(教師 shift は疎で構造的 → 符号一致に意味がある) |
| Zhao+ 2025 Echo Chamber arXiv:2504.07912; Wang+ 2026 Linear Dynamics arXiv:2601.04537 | RL は事前学習の支配的パターンを増幅 / RLVR は重みと出力 log-prob で線形 regime に入る | 中立(後者: 早期の方向変化は検出可能) |
| **Jiang+ 2026 Rock Tokens in OPD arXiv:2605.09253** | 飽和後も最大 18% の token が高い KL を保ち、機能的寄与は無視できる — 「生徒が内在化できない、または**する必要の無い**構造・談話の残差」 | 両義(全教師 KL は構造 token に勾配を浪費 → 目標で書式と内容を**分ける**動機 / 書式だけの stage が低機能 token に予算を使う注意) |
| Zhao+ 2026 OPSD arXiv:2601.18734 | token 別 KL clipping は「文体トークンが学習信号を支配するのを防ぐ」 | 両義(同上) |
| Thinking Machines 2025 OPD blog | off-policy 蒸留は「教師の文体と自信は真似できるが事実の正確さは必ずしも」 | 中立 / 支持(文体は最も伝わりやすい成分) |

**読み.** コミュニティの**慣行**は書式を先に固める(cold-start SFT、stage 1 の書式・ツール報酬、テンプレート不整合の再構築相、小モデルの書式 anchor)。
検証済みの注意は 2 つ — 書式を**報酬**で教えると探索を抑える(SimpleRL-Zoo)、教師 KL は構造 token に勾配を浪費する(Rock Tokens、OPSD) — で、
どちらも「書式は報酬でも全教師 KL でもなく、**柔らかい分布目標**で与えよ」を指す。stage 1 はまさにその形である。ただし
**「書式だけの目標を先に、内容を後に」を統制実験として分離した論文は見つからなかった**。設計 §4.5 の「正規形式切替が control(135)より早まる」予測は、
この慣行が正しければ当たる。一方 §2.4.1 の SA-OPD / Beyond-80/20 は「書式は spurious で利得を担わない」側で、慣行と 2026 年の OPD 研究は
ここで割れている。

#### 2.5.5 agent RL(ALFWorld / WebShop / Search)のカリキュラム

| 文献 | 要点 | 関係 |
|---|---|---|
| Qi+ 2025 ICLR WebRL arXiv:2411.02337 | 失敗から新課題を生成する self-evolving curriculum、KL 制約更新 | 中立 / 類似(データ側) |
| Feng+ 2025 NeurIPS GiGPO arXiv:2505.10978 | 1.5B で ALFWorld 86% 超 | 中立(小モデルの天井。中間 checkpoint 比較で天井効果に注意) |
| Xi+ 2025 AgentGym-RL arXiv:2509.08755 | 相互作用 horizon を短→長(ScalingInter) | 類似(時間ベース。予算のカリキュラム) |
| Li+ 2026 Guided-OPD arXiv:2606.15912 | 教師介入確率を cosine で 1→0(250 step の 80%)。**環境報酬無し**。0.6B で最大の利得 | 競合 / 類似(§2.4.3) |
| **Wang, Xu, Wu, Lyu 2026 Two-Phase Distillation arXiv:2606.30044** | 課題別 RL expert → off-policy + on-policy 蒸留で多課題 agent(τ²-bench、GEM、Qwen3-8B/30B-A3B)。off-policy 多課題蒸留は「振る舞いモードが生徒容量を超えて」失敗、多課題 RL は不均衡、**パラメータマージは 1 課題以外で劇的に低下** | 支持(課題別 RL expert + OPD の設定を検証)/ 競合。**agent 設定ではマージが弱い**(§2.4.2 の推論ドメイン fusion 研究と対照的) |
| Wang+ 2026 "To Mix or To Merge" arXiv:2602.12566 | 「ドメイン横断の RLVR は相互干渉が少なく、推論ドメインは相乗的」 | 中立 / 対立(干渉説。推論ドメインの話で agent ドメインは未検証) |
| Lu+ 2026 SDAR arXiv:2605.15155 | 同 3 ベンチ、Qwen3-1.7B、GRPO + ゲート付き自己蒸留 | 競合 / 類似(§2.4.3) |
| Xu+ 2026 Sparse-to-Dense arXiv:2605.12483 | 「教師で RL、生徒へは dense に転移」が生徒の直接 RL に勝つ | 競合(RL 済み教師からの純蒸留が baseline) |
| Zhou+ 2025 SWEET-RL arXiv:2503.15478; Wang+ 2025 RAGEN arXiv:2504.20073; AgentRL arXiv:2510.04206 | step 別 critic / Echo Trap / 多課題安定化 | 中立 |

ALFWorld / WebShop に特化した易→難の**課題インスタンス**カリキュラムは見つからなかった。この系列のカリキュラムは horizon・深さ・介入・目標の**量**に対するもので、
目標の**内容**に対するものは無い。

#### 2.5.6 合意を訓練信号にする(TTRL 系)と本機構の違い

| 文献 | 要点 | 関係 |
|---|---|---|
| TTRL arXiv:2504.16084; Yuan+ 2025 RLCCF arXiv:2508.12338; Zhang+ 2026 ICLR Co-rewarding arXiv:2508.00410 | 多数決 / 自己一貫性で重み付けた多モデル投票 / 類似問題間の一致を**報酬**に | 類似(一致を信号に。目標ではなく報酬) |
| **Shafayat+ 2025 SRT arXiv:2505.21444** | 多数決の自己訓練は早期に改善するが、長く回すと「reward hacking … 突然の完全な崩壊」 | 注意(**共進化する当事者間の合意はハックされる**。本機構の教師は凍結されており、残る フィードバックループは生徒 top-k support(C7)だけ) |
| Yu+ 2025 RESTRAIN arXiv:2510.02172; Chen+ 2026 JURY-RL arXiv:2604.25419 | 過信した多数決を罰する / 投票 + 形式検証器 | 中立 |

### 2.6 自然な学習順序 — 明示的カリキュラムは冗長か(P1, P3 への最大のリスク)

| 文献 | 要点 | 関係 |
|---|---|---|
| Saxe, McClelland, Ganguli 2019 PNAS arXiv:1810.10531 | 深い線形網は特異モードを特異値の順に $t \propto (1/s_\alpha)\ln(s_\alpha/\epsilon)$ で学ぶ。粗い(共有)構造が先、細かい構造が後。浅い網では全モードが同時 | 支持(前提)/ **冗長リスク**(共有幹が支配モードなら素の GD が先に学ぶ) |
| Lampinen & Ganguli 2019 ICLR arXiv:1809.10374 | 「最も重要な課題構造を先に」。転移は課題対の SNR と整合に依存 | 支持(前提) |
| Arpit+ 2017 ICML arXiv:1706.05394; Kalimeris/Nakkiran+ 2019 NeurIPS arXiv:1905.11604; Refinetti, Ingrosso, Goldt 2023 ICML arXiv:2211.11567 | 単純パターン先行 / 複雑さの増す関数列 / 低次統計から高次へ。初期の線形分類器は最後まで**保持**される | 中立(暗黙の順序)。「保持」は早期に学んだ共有成分が上書きされないことを支持 |
| Rubruck, Bauer, Saxe, Summerfield 2024 arXiv:2406.17467 | 入力非依存の**最適定数解(OCS)**をまず学ぶ初期相 | 類似 / 冗長リスク(書式 ≈ OCS 相の自然な対応物) |
| Gissin, Shalev-Shwartz, Daniely 2020 ICLR arXiv:1909.12051 | 漸進的(単純→複雑)学習は rich regime の暗黙性質で、lazy regime では消える | 中立 |
| Pezeshki+ 2021 NeurIPS Gradient Starvation arXiv:2011.09468 | 先に損失を下げた特徴が残りの特徴の勾配を**飢えさせる** | 両義(干渉の説明を支持 / 書式を先に当てると内容成分が飢えるリスク。stage 1 では内容成分が目標に無いので競合していない) |
| Lee, Goldt, Saxe 2021 ICML arXiv:2107.04384 | 逐次学習で**中間的な課題類似度**が最も忘却を生む | リスク(入れ子目標間の遷移は「よく似た逐次目標」の regime) |
| Chizat, Oyallon, Bach 2019 NeurIPS arXiv:1812.07956 | lazy training では線形化モデルと等価 | 中立(設計 §4.3 の NTK 注記の根拠) |

**読み.** この系譜は「網は共有・低複雑度の成分を**放っておいても先に**学ぶ」と言い、Rethinking-OPD(§2.4.1)は OPD について同じことを経験的に言う
(共有高確率集合から揃う)。明示的な目標カリキュラムが何かを足すのは、(a) 暗黙の順序が意図した順序と違う場合(課題固有成分の勾配が共有成分より大きく先に学ばれる、
または干渉する)、(b) 共有成分が教師側で**雑音を含み**、カリキュラムの分散低減が効く場合(Wu+ の雑音ラベル regime)、(c) run が短い場合(Wu+ の予算 regime)、
のいずれかである。(b)(c) は本設定に当たる。(a) は未測定で、設計 §4.3 の勾配整合仮説はまさに (a) を主張している。

### 2.7 feature learning regime での「段階解放が速い」理論(P1, P3 の理論的根拠)

| 文献 | 要点 | 関係 |
|---|---|---|
| Abbe, Boix-Adserà, Brennan, Bresler, Nagaraj 2021 NeurIPS arXiv:2108.10573 staircase | 高次 Fourier 係数が低次から**鎖状に到達可能**なら GD は低次を貪欲に組み合わせて学ぶ | 支持(入れ子 $a_3 \subset a_2 \subset a_1$ は構成された staircase) |
| Abbe, Boix-Adserà, Misiakiewicz 2023 COLT arXiv:2302.11055 leap complexity | 学習時間は成分間の最大の「跳び」で決まり、saddle-to-saddle で support を逐次学ぶ | 支持(中間成分を露出すると実効 leap が下がる。kernel regime では利得なし) |
| Cornacchia & Mossel 2023 ICML arXiv:2301.13833 | parity で、低次成分と相関する分布を先に見せるカリキュラムは計算量を大幅に減らし到達点は同じ。ただし階層分解できない目標(Hamming 混合)には効かない | 支持(条件付き) |
| Abbe, Cornacchia, Lotfi 2023 NeurIPS arXiv:2306.16921 | カリキュラム付き noisy-GD は、無し(任意の幅・深さの FCN)では学べない parity を学ぶ、の形式的分離 | 支持(feature learning regime) |
| Du, Hu, Kakade, Lee, Lei 2021 ICLR arXiv:2002.09434; Tripuraneni, Jin, Jordan 2021 ICML arXiv:2002.11684 | 共有表現は全課題の標本を**プールして**低分散で推定できる | 支持(P7 の「精度順」の統計的根拠) |

**読み.** 「同じ到達点で、低次成分を先に露出すると速い」は parity/staircase の系譜で**証明**されており、しかもその利得が feature learning regime にしか無いことも
証明されている。設計 §4.3 が「線形化では効かず、非線形・共有パラメータ regime でだけ効きうる」と書いた注記は、この系譜の結論と正確に一致する。
残る問いは、$a_3 \to a_2 \to a_1$ が parity の低次→高次のような**課題に関係する階層**なのか、それとも書式→内容という**無関係な分割**なのか、である。

---

## 3. 類似・競合・対立・支持の整理

### 3.1 新規性の判定

| 部品 | 先行例 | 本機構の差分 |
|---|---|---|
| base 相対 shift $h = \log\pi_T - \log\pi_0$ を蒸留の対象にする | proxy-tuning、EFT、Direct-OPD、ExOPD の reward correction、G-OPD の最適解 $\pi_{ref}(\pi^*/\pi_{ref})^\lambda$ | 同じ primitive。差分なし |
| 複数 fine-tune モデルの**符号一致**で共有成分を切り出す | TIES(質量で符号選挙)、AND-mask(一致率閾値)、NegMerge(全会一致)、GradDrop(符号純度)、GOVERN(多数決) | **min による入れ子**(集合が大きいほど主張は小さい)と**教師本数での層化**。多数決でも 2 値でもない |
| k-of-T の数え上げ | Consensus/TALL mask($k = 2$)、tri-training(2-of-3) | Consensus は課題固有の重みを削除、本機構は最後に復元。tri-training は合意した対が第 3 者にラベル付け |
| 全教師が同意する方向だけを使う | AE-KD(MGDA、許容度 $C$)、Distral(重心) | 一時的にしか使わない(stage 1)。AE-KD の「$C=1$ は準最適」は stage 1 を恒久化しない理由 |
| 時間で動く蒸留目標(到達点は教師) | Annealing-KD、CTKD、TAID、RCO、progressive distillation、TOP-D、CLPD | 補間軸がスカラーでなく**合意構造**。TAID は学習者適応ペース、本機構は step 固定 |
| 教師の助けを時間で消す | UFT、CBRL、Prefix-RFT、CHORD、Guided-OPD、KDRL の $\beta(t)$ | **向きが逆**(教師の内容は増える)。到達点 = 誘導無しの目的関数、という形だけ共通 |
| 同一 base のドメイン RL 教師 → 単一生徒の OPD | MOPD、ExOPD、Open-MOPD、LS-MOPD、CaMOPD、Two-Phase Distillation、fusion 研究 | control と同型。off-task 教師を**ゲートとしてだけ**使う点は無い |
| GRPO + 符号ゲート OPD + 段階スケジュール | SG-OPD(教師×検証器の符号、cosine 減衰の教師混合) | 符号の相手が「他教師」。SG-OPD は多教師ゲートを future work と明記 |
| agent OPD のカリキュラム(同ベンチ) | Guided-OPD(介入確率)、TCOD(深さ)、ATOD(OPD/RL 比)、AgentGym-RL(horizon) | どれも目標の**内容**は変えない |
| 複数参照への KL | Aminian+ 2025(複数参照の厳密解)、Distral | 参照の集約が合意で、しかも時間で変わる |

**判定: 組合せとしての新規性は成立する。** 一方で部品ごとの系譜が明確なため、各系譜の既知の限界(AE-KD の保守性、Consensus の削除、TAID のスカラー補間との比較、
誘導減衰系との向きの違い)を問われる。

### 3.2 競合(同じ目的の別機構)と、それぞれが本機構に課す対照

| 競合 | 何を共有するか | 本機構が示すべきこと |
|---|---|---|
| **$\beta(t)$ スケジュール**: 減衰(KDRL、SAF-OPD の warm-up-then-anneal、ATOD。理論文書 E1)と warm-up(設計 §6.4 追記) | 減衰は「教師は前半、報酬は後半」の時間構造で、ATOD は**同一 3 ベンチ**で利得を報告。warm-up は本機構と同じ向きで「前半に蒸留量を減らす」効果だけを持つ | 目標の**内容**を変えることが係数を変えることと区別できる(Ackermann+ 2026 が KL の**参照**と $\beta$ は別軸だと示している点は追い風)。warm-up で「量」を、E1 で「向き」を分ける — 両方要る |
| **KL-to-base ramp**(stage 1 の目標を $\pi_0$ に置き換え、同じ ramp で $\pi_d$ へ) | 設計 §3.2-4 の恒等「stage 1 + GRPO ≡ GRPO + 参照 KL」 | $a_3$ の**内容**(書式)を先に教えることが、単に base に留めることより速い |
| **TAID 型スカラー補間** $\pi_0 e^{\alpha_t h_d}$(同じスケジュール) | 「base と教師の間の目標を時間で動かす」 | **合意構造**が、単なる制限(スカラー)より速い。本機構固有の主張を分離する唯一の対照 |
| **誘導減衰**(UFT、CBRL、Prefix-RFT、CHORD、Guided-OPD) | 到達点 = 誘導無し | 教師の内容を**増やす**向きが、減らす向きより良い理由(P7 の精度順序) |
| **SFT → RL**(Limozin+ 2026。repo の `claude/sft-multitask` / `claude/offline-kd-multitask` 系のアームが該当しうる) | 段階型の教師利用 | バグの無い段階型 baseline との差。混合方策の利得は baseline の欠陥だった例がある |
| **一様幾何混合 / MOD**(理論 §5.2 E3)、**討論合意**(MAD-OPD) | 多教師情報を目標に入れる | 注入ゼロ(min)が平均・討論より良い(webshop で検定可能) |
| **マージ / Mix-RL**(TIES、Consensus、fusion 研究、Two-Phase) | 専門家を 1 つにする | fusion 研究(推論ドメイン)は Merge ≈ Mix-RL ≈ MOPD だが、Two-Phase(agent)はマージが弱い。速度という別軸を主張していると明示する |
| **routing のみ(MOPD)** / **報酬無しの純多教師 OPD** / **ExOPD の $\lambda > 1$** | = control / 教師を超える外挿 | 既に control。外挿と段階解放は直交(併用可) |
| **勾配空間の干渉対策**(PCGrad、CAGrad、GradDrop、TAC) | 干渉低減 | 目標空間で干渉を減らすことの利点(GRPO 項に触れない = 理論 §5.4 原則) |
| **両立性 / 正しさでゲート**(TA-OPD、SA-OPD、RG-OPD、OPDVR) | token 選択 | 合意ゲートはこれらと**直交**(教師間 vs 教師–生徒 / 検証器)で併用可能 |
| **学習者ペースの解放**(自己ペース学習、TAID の適応 $t$、HPT の成績ゲート、CGTR の報酬ゲート更新、設計 §5.2 の習熟トリガ) | いつ解放するか | 固定 clock の第 1 run は事前登録として正当。CGTR の「clock 駆動は state-oblivious」は第 2 run の論点 |

### 3.3 対立(前提に反する証拠・理論)

| 対立の種類 | 文献 | 本機構の前提のどこに当たるか | 機構側の応答(設計・理論文書に既にあるもの) |
|---|---|---|---|
| **教師の影響は初期に最大であるべき** | KDRL、SAF-OPD、ATOD(同ベンチ)、Guided-OPD(同ベンチ)、UFT、CBRL、Prefix-RFT、CHORD | P1・P5 の向き(教師の内容を初期に絞る) | 理論 §4.5: $\beta = 0.01$ では $A \ne 0$ の位置は報酬支配なので教師が効くのは $A = 0$ と初期動学だけ。本機構は「量」でなく「内容」を制御し、絞る部分は $\varepsilon$ を含む固有分。ただし設計 §3.2-4 追記の通り、未裏付け候補を base へ引き戻すのは報酬と教師の向きに対する**抗力**であって「遅れだけ」ではない。**直接の反証条件は webshop の保留コスト(設計 §6.2)** |
| 不一致こそ情報 | QBC、Decoupling、Co-teaching+、EnD²、TA-OPD | stage 1–2 は不一致トークン($\hat s_d$)を後回しにする | 捨てるのではなく遅らせる。コストは反証条件(webshop 2 SE)で測る |
| 一致は共有バイアスと真実を混同する | Eisenstein+ 2023、Spurious Rewards、SA-OPD、AND-mask(全環境で一貫する spurious は除去不能) | $a_3$ が「正しい」成分だという読み | 理論 §3.2 が「一致 = 信頼」を**自ら否定**。設計 §9 追記も「裏付けは一致を証明し、正しさを証明しない — 同じ base・ハーネス・報酬構造の系統誤差が最初に最も高い確信で蒸留される」と記録。主張は「共有」のみ。「共有だが spurious な成分を先に教える利得」は未証明 |
| 全会一致は保守的すぎる | AE-KD($C=1$ は準最適)、CAGrad / Twin-Merging(純共有は不足) | stage 1 の目標 | 一時的。設計 §2.3 の漏れ表(帰無で 10.5%)は保守性の定量。スケジュールが急所 |
| 自然な順序と冗長 | Saxe、Kalimeris、Refinetti、Rubruck(OCS)、Rethinking-OPD(OPD は共有高確率集合から揃う) | P1 が何かを足すか | 設計 §4.3 が仮説と明記。Wu+ の 2 条件(予算・雑音)が当てはまる点で「冗長でない」余地 |
| 効果は小さく一過性 | Wu+ 2021、Saglietti+ 2022、Mannelli+ 2024、Elgaar & Amiri 2026、SPEED-RL の撤回 | 検出力 | 設計 §6.3 が反復 SD で下限を書き、中間 checkpoint と学習曲線を主要評価にしている。**2 シード**が要る |
| 混合方策の利得は baseline の欠陥だった | Limozin+ 2026 | 「前半に教師、後で解放」型の利得一般 | 本機構の control は同一コードの GRPO+OPD で、SFT baseline ではない。段階型(SFT→RL)との比較は別途 |
| 干渉説は疑わしい | Xin+ 2022、Kurin+ 2022、Open-MOPD(予算配分が原因)、To Mix or To Merge(推論ドメインは干渉が少ない) | 設計 §4.3 の勾配整合仮説 | 仮説として事前登録。Open-MOPD の「長さ差・drift」は `normalize_loss_by_task` で一部対処済みだが未検証。agent ドメインの干渉は Two-Phase / CaMOPD が実在を示す |
| 早期優位は entropy 上昇(探索)の副作用かもしれない | 監査 §15(klw アームの利得は entropy に帰属)、設計 §6.2 訂正(stage 1 の目標は base 寄りで平坦 → entropy は step 90 まで control より高い)、SAF-OPD(OPD の時期と entropy) | 早期優位を「順序」に帰属すること | 設計 §6.2・§9 追記が交絡を排除できないと明記。R1(KL-to-base ramp)は同じ entropy 挙動を持つので、R1 との比較は交絡を共有した上で $a_3$ の内容の効果だけを分離する |
| 順序と量が分離できていない | 設計 §6.4・§9 追記(本文書と独立に到達した同じ結論)。KDRL / SAF-OPD の「量の時期」の効果 | 早期優位を「順序」に帰属すること | 設計 §6.4 追記の $\beta$ warm-up 対照。§5 R1–R3 |
| 固定 clock は脆い | CGTR、自己ペース学習、TAID、HPT | P5 | §5.2 で習熟トリガを第 2 run に留保 |
| 中間目標の「種類」が違う | Panigrahi+(相転移中の checkpoint だけ有効)、Cho & Hariharan(逐次 KD 無効)、Beyond-80/20、Rock Tokens、Jiang+ 2025(能力→書式の逆順) | stage 1 = 書式 | pair 層の内容 role シェアを初測定として事前登録(設計 §6.1)。これが低ければ Panigrahi+ の意味での有効な中間目標ではない |
| off-task 教師を他課題の状態で読む | TGPO、MOPD の 5× KL ablation、Shen+ の top-k の穴 | P2 の符号は off-task 教師の他課題状態での出力 | off-task はゲートにしか使わず目標には入らない(P4)。符号の雑音は $a_3$ の過小評価(設計 §2.3)として現れ、致命的ではない |

### 3.4 支持(前提を支える証拠・理論)

| 支持の種類 | 文献 | 支える前提 |
|---|---|---|
| 到達点不変・早期のみ・難タスクで大 | Weinshall+ 2018、**Hacohen & Weinshall 2019**、Saglietti+ 2022(控えめな加速) | P3、設計 §6.2 の予測の形(局在は control の弱いサブタスク) |
| カリキュラムが効く 2 条件 | **Wu+ 2021**(短い予算、雑音ラベル)、E2H(小モデル・時間ベース・早期段階の消滅) | 300 step 予算、単 run 教師の $\varepsilon$、P5 の形 |
| 目標側カリキュラムの加速(証明付き) | **Panigrahi+ 2025**、Gupta & Karmalkar 2025、TAID、Annealing-KD、CTKD、RCO、CLPD | P1 |
| feature learning regime の段階解放 | staircase、leap complexity、Cornacchia & Mossel、Abbe-Cornacchia-Lotfi | P1、P3、設計 §4.3 の NTK 注記 |
| 共有成分の低分散推定 | Du+ 2021、Tripuraneni+ 2021、Xu & Tewari 2022 | P7 の精度順 |
| 目標は生徒に近い所から | Liu, Liu, Cohan 2024(参照は似ているときだけ有効)、TAID、TOP-D | stage 1 が base 近傍から始まること |
| OPD の**時期**が結果を変える(同設定) | KDRL、**SAF-OPD**、**ATOD**(同 3 ベンチ)、CaMOPD(交互訓練) | P5 の存在意義(向きは別) |
| KL の参照は $\beta$ と別の軸 | **Ackermann+ 2026**、ProRL、TR-DPO、Aminian+ 2025 | P1 が $\beta(t)$ と別の実験軸であること |
| 排他的成分を混ぜると損なう | **Twin-Merging**、Consensus(selfish weights)、CaMOPD(負の勾配内積)、MOPD(同一 base 教師が必要)、Two-Phase(agent でマージが劇的に低下) | P4 の注入ゼロ |
| 全会一致成分の実在と shift の疎性 | Mukherjee+ 2025(RL 部分網の重なり)、Akgül+ 2026(1–3% の位置、base top-5 内)、Spurious Rewards(共有スタイル)、監査 §11.6($g$ ≈ 書式) | P7 の $g \ne 0$、P2 の符号一致に意味があること |
| 多ドメイン LLM RL の干渉は実在し順序が効く | Li+ 2025(Math+Puzzle が code を落とす、段階 + refresh が勝つ)、Omni-Thinker(structured → open-ended)、Yang-Ding-Xiong 2026(共有衝突部分空間)、TAC(勾配整合カリキュラム、Qwen3-1.7B) | 設計 §4.3 の勾配整合仮説 |
| 書式は先に、柔らかい目標で | R1(cold start)、**R1-Searcher**(search-QA で書式 → 正解性)、Understanding R1-Zero(書式再構築相)、SimpleRL-Zoo(書式**報酬**は害)、Rock Tokens / OPSD(全教師 KL は構造 token に浪費)、SAKE(小モデルの anchor)、SRFT(SFT = 粗い大域、RL = 細かい選択) | stage 1 の**形**(書式を報酬でも全教師 KL でもなく分布目標で)。設計 §4.5 の切替予測 |
| 共有方策への KL は多タスク RL の安定化 | Distral、Divide-and-Conquer、Policy Distillation | stage 1 の regime |
| 早期 OPD = 共有高確率集合の整合 | Rethinking-OPD | $a_3$ が先に揃うという記述(冗長リスクと表裏) |
| 凍結教師なら合意はハックされない | SRT(共進化の多数決は崩壊) | P2 が凍結教師の上にあること |

---

## 4. 既存研究の傾向は本機構を支持するか

### 4.1 形(設定・設計軸)についての傾向 — 支持する

* **設定は 2026 年の主流である。** 同一 base のドメイン別 RL 教師 → 単一生徒の on-policy 蒸留は MOPD(製品投入)、ExOPD、Open-MOPD、LS-MOPD、CaMOPD、
  Two-Phase Distillation(agent)、Tencent の fusion 研究が揃って採用し、MOPD は同一 base であることを安定化の要件として示した。本アームの control(on-task routing)はその既定形である。
* **OPD の時間構造は活発な設計軸である。** 同じ 3 ベンチで ATOD、agent OPD で Guided-OPD / TCOD / FutureBridge、一般 OPD で SAF-OPD / CGTR / ATESD / USD が 2026 年に集中している。
  RLVR 側でも誘導減衰(UFT、CBRL、CHORD、Prefix-RFT)が標準化した。「いつ・何を教師から受け取るか」は問いとして認められており、「到達点 = 誘導無しの目的関数」は慣例である。
* **合意分解はマージ研究で確立した。** TIES(2023)→ Consensus(2024)→ NegMerge / CALM(2025)と、「符号一致で共有成分を切り出し、排他的成分を混ぜない」は標準的な発想になった。
  出力空間への移植は自然な拡張として読まれる。
* **目標側カリキュラムは主流会議で認められている。** TAID(ICLR 2025 spotlight)、Panigrahi+(ICLR 2025 oral)、E2H(ICLR 2026)。

### 4.2 中身(仮説の内容)についての傾向 — 条件付きでしか支持しない

* **予測の形は一致するが、大きさは小さいと予測される。** Hacohen & Weinshall・Saglietti・Wu+ は「早期のみ・控えめ」を予測し、Mannelli+ 2024 と Elgaar & Amiri 2026 は
  規模と過剰パラメータ化で消えると予測する。1.7B・300 step(150 step ならなお短い)・3 タスクは Wu+ の「効く条件」の側にあるが、設計 §6.3 の反復 SD(alfworld 1.39pp、差の 2 SE 3.9pp)に対して、
  文献が予測する効果量が上回る保証は無い。8/13 アームの +11.1pp(理論 §4.2)は、もし順序機構で再現するなら文献の予測より大きい。
* **教師影響の向きは主流と逆である。** 2026 年の誘導減衰・$\beta$ anneal・SAF/ATOD は「教師は初期に最大」で一致し、ATOD は同じ 3 ベンチでそれを示した。
  本機構は教師の**内容**を初期に最小にする。同じ向きの先行例(TAID / TOP-D)の根拠は同一 base 教師には薄く、本機構の根拠(P7 の精度順序)を時間スケジュールとして
  使った先行例は無い。理論 §4.5 の「$\beta = 0.01$ では教師は $A = 0$ の位置と初期動学にしか効かない」が正しければ、絞られる固有分のコストは小さいはずだが、
  それは設計 §6.2 の webshop 反証条件で測る事柄で、文献からは出ない。**E1($\beta(t)$)は主流と同じ向きなので、本機構と E1 は同時に走らせる対照でなければならない。**
* **stage 1 の目標が「書式」であることは、慣行と 2026 年の OPD 研究の間で割れる。** 慣行(R1 の cold start、R1-Searcher の 2 段報酬、テンプレート再構築相、
  小モデルの書式 anchor)は書式を先に固めることを支持し、SimpleRL-Zoo / Rock Tokens / OPSD の注意は「報酬でも全教師 KL でもなく柔らかい目標で」を指す —
  これは stage 1 の形そのものである。一方 SA-OPD は書式慣習を spurious としてフィルタし、Beyond-80/20 は低 entropy トークンが RL の利得を担わないと示し、
  Rethinking-OPD は OPD が放っておいても共有高確率集合から揃うと言う。**「書式だけを先に」を統制した実験は無い。** 設計 §4.5 の「切替を早める」予測が外れたら後者が正しい。
* **干渉説は分かれている。** CaMOPD・Li+ 2025・Yang-Ding-Xiong 2026・TAC・Two-Phase は多ドメイン LLM RL / agent の干渉を実在とし、Open-MOPD は多教師 OPD の失敗を
  予算配分だと言い、To Mix or To Merge は推論ドメインでは干渉が少ないと言う。Xin+ / Kurin+ は一般の多タスク最適化手法を正則化として説明する。
  設計 §4.3 が干渉低減を**仮説**として事前登録しているのは、この分裂に対して正しい姿勢である。
* **「一致 = 共有であって信頼ではない」は文献と一致し、しかも文献はその先を言っている。** Eisenstein+ と AND-mask の限界は「共有バイアスは合意で除けない」であり、
  理論 §3.2 の識別不能性の経験的な裏付けになる。NegMerge が示すように、全会一致が雑音と信号を分離できるのは**同一課題の複製**のときで、異課題教師の全会一致が
  分離するのは「共有か固有か」だけである。本機構はそれを前提に設計されているので矛盾はないが、「共有成分を先に教えると速い」という**利得**の主張は
  文献のどこにも直接の先例が無い(Distral の重心、Omni-Thinker の順序、SRFT の粗→細の分業が最も近い間接証拠)。

### 4.3 総合判定

**傾向は「この機構を走らせること」を支持し、「この機構が効くこと」は支持も否定もしない。** より正確には:

1. 文献が予測する信号(早期・一過性・難サブタスクに局在・到達点一致)と、設計 §6.2 が事前登録した信号は**同じ**である。設計の予測は文献的に妥当。
2. 文献は同時に、その信号が $\beta(t)$ anneal(ATOD が同ベンチで示した)、KL-to-base ramp、TAID 型スカラー補間でも出うると予測する。
   **本機構固有の主張(合意構造が順序を作る)は、これら 3 対照を並べない限り検定されない。**
3. 教師影響の向きは主流の逆であり、本機構が効くとすれば「初期に絞った固有分 $\hat s_d$ のコスト < 精度順序の利得」という、文献に先例の無い不等式が成り立つときである。
   設計 §6.2 の反証条件(webshop @150 で 2 SE 低下)はこの不等式の左辺を直接測る。
4. stage 1 = 書式という経験的事実は、慣行と 2026 年の OPD 研究で評価が割れる。本機構が効くなら、$a_2$ 層(2 教師一致)に書式以外の共有技能が載っている場合か、
   「書式を早く固定すること」自体が探索を助ける場合に限られる。前者は設計 §6.1 の初測定で、後者は正規形式切替の時期(設計 §6.2)で読める。
5. 効果量は文献的に小さいと予測されるので、**2 シード・中間 checkpoint・AUC / steps-to-threshold** で読む設計が要る。設計 §6.3 の中間 checkpoint 保存はこれに当たる。
6. **150 step で走らせるなら「到達点 = control」(P3)は検定できない。** 設計 §5.1 追記の通り制限区間が run の 59.3% を占め、Hacohen & Weinshall / Wu+ の
   「効くのは初期だけ」の regime が run の大半を覆う。文献の予測は「@150 の差はどちらの向きにも出うる(順序の利得 − 保留の抗力 − 量の減少)」に変わり、
   @150 は到達点ではなく中間点として読むか、@300 まで走らせる必要がある。反証条件(webshop @150 で 2 SE 低下)はこの配分でも有効だが、
   「control に追い付く」側の予測は @300 でしか読めない。
7. **設計 §9 の追記(順序と量の未分離、entropy 交絡、base への抗力、系統誤差の共有、反証可能性の非対称)は、本文書が文献から独立に導いた
   主要な対立点(§3.3)と一致する。** 文献はこれらに名前と先例を与える(KDRL / SAF-OPD = 量の時期、監査 §15 = entropy、Eisenstein+ = 系統誤差)が、
   新しい対立点は加えない。逆に文献だけが加えるのは「向きが主流と逆」(§2.5.2)と「書式は spurious か anchor か」(§2.5.4)の 2 点である。

---

## 5. 文献から出る追加の対照と改良案

設計 §6・§8 に**無い**もの、または文献が優先順位を変えるものだけを挙げる。

| # | 提案 | 根拠 | 設計との関係 |
|---|---|---|---|
| R1 | **KL-to-base ramp 対照**: stage 1 目標を $\pi_0$($a_3 \equiv 0$)、以後同じ ramp で $\pi_d$ へ | 設計 §3.2-4 の恒等「stage 1 + GRPO ≡ GRPO + 参照 KL」。Xin+ / Kurin+ の「多タスク手法は正則化」。設計 §6.4 追記の $\beta$ warm-up は前半に KL 項を持たない「量」の対照で、R1 は前半に KL-to-base を持つ。3 者で読むと、warm-up と R1 の差 = base への抗力(設計 §3.2-4 追記)、R1 と本機構の差 = $a_3$ の内容 | §6・§8 に無い。**warm-up と並べて最優先** |
| R2 | **スカラー補間対照**: $\pi_0 e^{\alpha_t h_d}$、$\alpha_t$ を $\rho$ と同じ ramp で 0→1 | TAID(ICLR 2025)。合意**構造**と単なる制限を分離する唯一の対照 | §6・§8 に無い。R1 の次 |
| R3 | **設計 §6.4 追記の $\beta$ warm-up と、E1($\beta$ 減衰)の両方を本機構と同時に走らせる** | warm-up は同じ向きで「量」を分ける(設計 §6.4 追記の 4 通りの読み)。E1 は主流と同じ向きで、KDRL・SAF-OPD・ATOD が効くと示した形。両方あって初めて「向き」と「内容」が分かれる。設計 §6.4 は「本案 → warm-up の順」としたが、向きが主流と逆である以上、本案単独の結果は E1 との差としても読めない | 設計 §6.4 追記(warm-up)、理論 §6.1 E1。R1・R2 と合わせて 4 対照 |
| R4 | 評価に **AUC / steps-to-threshold** と **2 シード** を加える | Hacohen & Weinshall、Wu+、Saglietti(効果は早期・控えめ)。SPEED-RL の撤回と Limozin+ の baseline 欠陥は効率主張の脆さの例 | 設計 §6.3 の中間 checkpoint に追加 |
| R5 | **pair 層の role 別シェア**を主要診断に格上げ | Panigrahi+: 有効な中間目標は課題に関係する低次構造を運ぶ。書式だけなら R1 と区別できない | 設計 §6.1 に「初測定」として存在。判定基準に格上げ |
| R6 | 段階型 baseline(**SFT → GRPO**、教師出力での SFT)をバグの無い形で 1 本置く | Limozin+ 2026。repo の `claude/sft-multitask` / `claude/offline-kd-multitask` 系が該当しうる(本文書では内容未確認) | §6・§8 に無い |
| R7 | stage 2 の集約に**許容度**(AE-KD の $C$、SAND-mask の連続化)を検討 | AE-KD: 純粋な全会一致は準最適 | 第 2 run 以降。設計 D2 の max(和集合)は既に $C < 1$ 方向 |
| R8 | 解放を**習熟トリガ**にする(設計 §5.2)際は HPT の成績ゲート・CGTR の報酬ゲート更新・TAID の適応 $t$ を参照 | clock 駆動の脆さ(CGTR)。2026 年の主流は習熟トリガ | §5.2 に既にある。文献的裏付けの追加 |
| R9 | 段階境界での **consolidation**(前段階の解に対する弱い anchor)の有無を第 2 run で比較 | Saglietti+ 2022: 段階間の接続無しでは利得が残らない。入れ子目標は部分的にこれを満たす | §6・§8 に無い |
| R10 | 多教師 OPD の既知の落とし穴(Open-MOPD の長さ差・drift、Shen+ の top-k の穴)を診断列に加える | 干渉ではなく予算配分が原因なら、順序機構の効果はそれに埋もれる | `normalize_loss_by_task` と生徒 top-k で部分対処。未計測 |
| R11 | 正規形式切替の step を主要読み出しに残し、**書式報酬**との比較は行わない | 設計 §4.5 の切替予測は慣行(R1-Searcher、Understanding R1-Zero)の検定になる。SimpleRL-Zoo が示す通り書式報酬は別の害を持つので対照にならない | 設計 §6.2 に既にある |

---

## 6. 限界

* サーベイは 2026-09-03 時点の web 検索で、arXiv 番号と abstract を取得して同定した(4 領域に分けた調査で、各領域の検索予算は上限に達した)。
  **本文を精読した論文は一部**で、要点は abstract と該当節の転記に依る。
* 「未検証」として除外した候補: GATES(consensus-gated KL 自己蒸留、OPD サーベイ内の言及のみ)、Nemotron-Cascade 2、DeepSeek-V4 の OPD 主張(arXiv:2606.19348)、
  Lucy(arXiv:2508.00360)/ PageLLM(arXiv:2506.09084)の「書式は後」「warm-up 無しは schema 学習に予算」という snippet、Spurious Rewards の書式報酬 +13.8 の数字、
  "Prefix Teach, Suffix Fade"(arXiv:2605.13643)ほか awesome-list 上の 2026 年 OPD 論文約 20 件。LPPO・"Curriculum-guided RLVR" は該当論文が見つからなかった。
* 2026 年の OPD 論文の多くは arXiv preprint で査読を経ていない。「傾向」はそれらの**数と向き**についての言明で、個々の結果の再現性は担保しない。
  SPEED-RL の撤回と Limozin+ 2026 の baseline 欠陥は、この分野の効率主張が動きうることの例である。
* 本文書は機構の**設計**を文献に照らしたもので、機構の**結果**は無い(未走行)。§4 の判定は走行後に書き換わる。
* 「類似・競合・対立」の分類は本文書の判断で、特に競合と対立の境界(例: TA-OPD、誘導減衰系)は恣意的である。
* repo 内の他アーム(`claude/sft-multitask` 等)の内容は本文書では確認していない。R6 はブランチ名からの推測である。
* 設計文書の 2026-09-03 の改訂(境界 50 / 90、150 step の配分、$\beta$ warm-up 対照、§6.2 の entropy 訂正、§9 の 5 つの限界)は本文書の執筆中に入った。
  本文書はそれに合わせて §1・§2.5.2・§3・§4.3・§5 を更新したが、設計文書の新しい診断列(`target/control/grad_cosine` 等)は文献と突き合わせていない。
