# 裏付け限定・段階解放カリキュラム蒸留(corroboration-only curriculum)

状態: **設計 + 実装済み(2026-09-03)。未走行。** §6 の事前登録は intent lock
(`examples/opd_grpo_trainer/expected_multitask_cross_teacher_curriculum_config.yaml`)で機械的に固定されている。
実装は §7、走らせる前に決まっていること・残っていることは §8。

前提文書: [cross_teacher_theory.md](cross_teacher_theory.md)(理論、命題 1–3、階層モデル、識別不能性)、
[cross_teacher_target_design.md](cross_teacher_target_design.md)(現行 target 機構と 3 撤回)、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)(監査: $g$ ≈ 書式、shuffled 比 0.82)。
対象コード: `verl/trainer/ppo/cross_teacher_target.py`(`mode="curriculum"`: `curriculum_rho` /
`nested_layers` / `curriculum_exponent` / `build_target` / `TargetStepStats`)、
適用点 `verl/workers/actor/dp_actor.py` の `xtt_build_target` 呼び出し、
スケジュールの供給元 `verl/trainer/ppo/opd_ray_trainer.py`。
アーム: `examples/opd_grpo_trainer/run_multitask_cross_teacher_curriculum_qwen3.sh`。

要求(2026-09-03): 現行機構を **corroboration(裏付け)機能のみ**に変え、
(1) 前半は全教師の policy shift 一致信号 $g$ だけを蒸留、
(2) 中盤は on-task 教師と**いずれか 1 本**の off-task 教師の 2 教師一致信号 $s_{dj}$ を加えて蒸留、
(3) 後半は不一致でも on-task 教師の全 shift $s_d$ を蒸留する、という段階解放カリキュラムにする。

---

## 0. 結論(先に)

1. **機構は「on-task shift の入れ子分解 + 時間解放」として書ける。** on-task 教師の shift $h_d$ を、主張する教師の本数で
   $h_d = \underbrace{a_3}_{3 本一致} + \underbrace{(a_2 - a_3)}_{2 本一致の超過} + \underbrace{(h_d - a_2)}_{on\text{-}task 単独}$
   と分け、$|a_3| \le |a_2| \le |h_d|$、符号は全て $\mathrm{sign}(h_d)$(§2)。stage $k$ の目標は
   $\tilde\pi^{(k)} \propto \pi_0\, e^{a_k}$、すなわち **base に「$k$ 本以上の教師が主張する成分」だけを載せた分布**。
   stage 3 で $\tilde\pi^{(1)} = \pi_d$、control と bit-identical。
2. **off-task 知識の注入は構成上ゼロ。** 全 stage で目標は base と on-task 教師の**間**にある
   ($p_0 e^{a_k}$ は候補ごとに $[\min(p_0,p_d), \max(p_0,p_d)]$ の内側、§3.2)。現行アームのチャネル B(fallback)と、
   幾何平均 $L$ が $O$ を超える経路は削除。監査が webshop 悪化の機構と読んだ「内容位置に他タスクの固有成分を混入させる介入」は
   起こり得ない。**exponent_scale と `_EXPONENT_CLAMP` も不要になる**(自然な上界 $p_0$ があるため、§3.2)。
3. **到達点は control と同一**(理論文書 命題 1・2 の帰結、§4.1)。この機構は**順序**の機構であり、
   効果があるなら中間 step にだけ現れる。主要評価は @300 ではなく、T=1.0 窓平均の時間プロファイルと @75/@150 の検証値(§6)。
   「単タスク参照 0.849 との差を埋める」ことは理論から出ない。
4. **理論が与えるもの**: (a) 分解の粒度の上限 — 識別不能性(理論 §3.2)により、4 モデルから引ける最も細かい分解は
   「何本の教師がその方向を主張するか」であり、$s_d$ と $\varepsilon_d$ はどの stage でも分けられない。
   (b) 精度順序 — 3 本一致 → 2 本一致 → 単独は、共有成分の推定分散が小さい順(§4.2)。
   (c) 漏れの定量 — 共有成分が無い($g=0$)候補で stage 1 の目標に漏れる on-task shift は期待値で **10.5%**、
   stage 2 は **48%**(§2.3 のシミュレーション)。stage 2 の選択性は低く、stage 1 と stage 3 の中間ではなく stage 3 寄り。
   (d) 固定点不変。
5. **理論が与えないもの**: 「共有成分から先に学ぶと速い」ことの証明。共有パラメータ下の勾配整合の**仮説**として事前登録し(§4.3)、
   反証条件を書く(§6.2)。理論文書 §4.3 が「到達点は同じで到達が早かった」と観測し「何の順序か」を未測定とした問いに、
   この機構は「順序 = 裏付けの本数」という 1 つの答えを**検定可能な形で**置く。

---

## 1. 現行機構からの差分

| | 現行 target アーム(`cross_teacher_target.py`) | 本案 |
|---|---|---|
| 意図 | 1: 一致で強く蒸留、2: off-task から代替信号 | **1 のみ**(裏付け限定)。2 は削除 |
| $c$ の向き | $\log\tilde p = \log p_{on} + c$、$c = s\cdot L$ で **on-task を超えて増幅** | $c_k = a_k - h_d$ で **on-task から未裏付け分を差し引く**。$|c_k| \le |h_d|$ |
| off-task の集約 | 幾何平均 $L$(自己減衰・複製不変) | 集合ごとの **min**(入れ子性が要る、§2.2)。複製不変は max/min なので維持 |
| 時間 | 定常 | **$t$ で解放**($\rho_2(t), \rho_1(t)$、§3.3) |
| 自由パラメータ | `exponent_scale`(1 個)+ clamp 5.0 | **スケジュール**(境界 2 点 + ramp 幅、§5)。scale・clamp は消える |
| control との差 | 機構キー 3 個 | 機構キー(`mode`, `stage_steps`, `ramp_steps`)。$\rho \equiv 1$ で bit-identical |
| 固定点 | 動く($c \ne 0$ が定常) | **動かない**(stage 3 で $c \equiv 0$) |
| G1(shuffled) | `shuffled_tv_ratio` ≪ 1 を要求 | 意味が反転する。**保持質量比**に置き換える(§6.1) |
| G3(tag_share) | 0.3 超で警告 | stage 1 では**高いことが予測**($g$ ≈ 書式)。警告ではなく確認項目 |

チャネル A の `min(O, L)` はもともと $|a| \le O$ を満たしていたので、本案の stage 3 直前の姿(全 stage の合成)は
「チャネル A だけを残し、増幅ではなく制限として使い、時間で緩める」と読める。

---

## 2. 階層モデルと入れ子分解

### 2.1 モデル(理論文書 §3.1 の拡張)

候補 $(x, v)$、タスク $d = d(x)$、標準化 shift $\hat h_m = h_m/\sigma_m$ について

$$
h_m = g + \sum_{j \ne m} s_{mj} + s_m + \varepsilon_m, \qquad m = 1..3.
$$

* $g$: 全教師共有(ハーネス書式・タグ構文・一人称。監査 §11.6)。
* $s_{mj} = s_{jm}$: **2 教師だけが共有**する成分(例: alfworld と webshop が共有する「品定めをやめて動け」型の傾向)。
* $s_m$: タスク固有。$\varepsilon_m$: 教師 $m$ の RL 推定誤差(有限 step、シード、報酬ハック)。

タスク $d$ の目標は $h^\star_d = g + \sum_{j\ne d} s_{dj} + s_d$。**$s_{jk}$($d$ を含まないペア、$d=1$ なら $s_{23}$)は目標に含まれない。**
要求文の「s12, s23, s13」のうち、タスク 1 の stage 2 に入るのは $s_{12}, s_{13}$ だけで、$s_{23}$ を入れることは
チャネル B と同じ off-task 注入になる。裏付け限定の規則(目標は $h_d$ の部分集合)から自動的に除外される。

### 2.2 推定量: 集合ごとの符号ゲート付き min

off-task shift を on-task 教師の nats に直す: $\tilde h_j := \sigma_d\,\hat h_j = h_j\,\sigma_d/\sigma_j$
(理論 §3.3「RMS 単位の扱い」と同じ。$\sigma$ は対角、前 step スナップショット — target 設計 §10.1 の決定を踏襲)。
$s_d := \mathrm{sign}(h_d)$ として

$$
\begin{aligned}
a_1 &:= h_d \\
a_2 &:= s_d \cdot \max_{j \ne d}\ \mathbf 1[\mathrm{sign}(\tilde h_j) = s_d]\ \min\big(|h_d|, |\tilde h_j|\big) \\
a_3 &:= s_d \cdot \mathbf 1[\forall j\ne d:\ \mathrm{sign}(\tilde h_j) = s_d]\ \min\big(|h_d|, \min_{j\ne d}|\tilde h_j|\big)
\end{aligned}
$$

**補題(入れ子).** $|a_3| \le |a_2| \le |a_1|$、かつ非零なら全て符号 $s_d$。
*証明.* 3 本一致なら $\min_{\text{3 本}} \le \min_{\text{任意のペア}} \le \max_j \min_{\text{ペア } j}$。3 本一致でなければ $a_3 = 0$。
$|a_2| \le |h_d|$ は各 min が $|h_d|$ で頭打ちだから。$\square$

層: $\hat g := a_3$、$\hat s_{d\cdot} := a_2 - a_3$(「on-task と少なくとも 1 本が主張する超過分」= $s_{dj} \cup s_{dk}$ の推定)、
$\hat s_d := a_1 - a_2$(単独分)。3 層の和は恒等的に $h_d$。

**なぜ min であって幾何平均でないか。** 現行の $L$ は 2 本の off-task の幾何平均で、複製不変・自己減衰のために選ばれた。
本案では「集合 $S$ の全員が主張する量」を集合の大きさ順に**入れ子**にする必要があり、$\min_S$ はそれを満たす
($S \subset S' \Rightarrow \min_{S'} \le \min_S$)が幾何平均は満たさない。複製不変は min・max とも成り立つ。
自己減衰(1 本が沈黙すれば 0)も min の方が強い(幾何平均は「小さくなる」、min は「その値以下」)。

**なぜ max(和集合)であって平均でないか(stage 2).** 要求は「on-task と**何か 1 つ**の off-task」であり、これは $j$ についての和集合。
平均を取ると、片方だけが一致するペア成分が半減し、「2 本一致」の意味から外れる。

**閾値・deadzone は無い。** 符号ゲートは $\hat h \to 0$ で $\min \to 0$ と連続に閉じる(現行 §2.3 と同じ論法)。
`report_epsilon` の失敗(監査: 到達 5.1%・純度 0.16)を再現しない。

### 2.3 漏れの定量(共有成分が無いときに何が stage 1 に入るか)

$g = s_{dj} = 0$、残差が iid $\mathcal N(0,1)$(標準化単位)の候補で、各層に漏れる on-task shift の期待値。
純 Python、$N = 4\times10^5$(付録 A):

| $g$ | $\mathbb E|a_1|$ | $\mathbb E|a_2|$ | $\mathbb E|a_3|$ | $|a_2|/|a_1|$ | $|a_3|/|a_1|$ | $P(a_2 \ne 0)$ | $P(a_3 \ne 0)$ | $\mathbb E[a_3]$(バイアス) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.797 | 0.384 | 0.084 | 0.48 | **0.105** | 0.750 | 0.251 | 0 |
| 0.5 | 0.896 | 0.505 | 0.159 | | | 0.788 | 0.361 | 0.144(−0.36) |
| 1.0 | 1.167 | 0.822 | 0.384 | | | 0.867 | 0.600 | 0.383(−0.62) |
| 2.0 | 2.015 | 1.723 | 1.176 | | | 0.978 | 0.933 | 1.176(−0.82) |
| 3.0 | 3.000 | 2.718 | 2.153 | | | 0.999 | 0.996 | 2.153(−0.85) |

読み:

* **stage 1 は選択的だが保守的。** 共有成分が無い候補では on-task shift の 10.5% しか通さない(単独教師の 1/10)。
  一方、真の $g$ を 0.6–0.85σ 過小評価する。この過小評価は一時的で、stage 3 で全額解放されるので固定点には効かない。
* **stage 2 の選択性は低い。** 帰無下で 75% の候補が非零、大きさで 48% を通す。stage 2 は「stage 1 と 3 の中点」ではなく
  stage 3 寄りで、$g$ の回復は良い($g=1$ で 0.80)。スケジュールは stage 2 を短めに取る(§5)。
* **stage 1 の推定分散は単独の 1/4.6**(sd 0.219 対 0.999)。これが「精度順序」の中身(§4.2)。
* この表は等分散ガウスの数字。実際の $\hat h$ は softmax 裾の推定雑音で重い尾を持つ(理論 §3.5)。
  **実測は shuffled 反実仮想で行う**(§6.1 の保持質量比)。表は「何を測るべきか」の目盛りにすぎない。

---

## 3. 目標分布の構成

### 3.1 tilt 形で書く(既存の正規化経路をそのまま使う)

stage $k$ の tilt を $c_k := a_k - h_d$(nats、support 上)とし、現行の `normalized_weight` に渡す:

$$
\tilde p^{(k)}(v) = \frac{p_{on}(v)\,e^{c_k(v)}}{Z_k} = \frac{p_0(v)\,e^{a_k(v)}}{Z_k}\ (v \in S), \qquad
Z_k = \sum_{v\in S} p_0(v)\,e^{a_k(v)} + p_{on,\text{tail}}.
$$

* $c_k$ は $h_d$ と**逆符号**で、大きさ $|h_d| - |a_k|$。「増幅」ではなく「未裏付け分の差し引き」。
* **tail は on-task 教師の形のまま**($c = 0$、$Z$ にだけ入る)。現行と同じ扱い。support カバレッジは 98%+(target 設計 §10.1)
  なので tail を base に戻す/戻さないの差は小さいが、on-task 側に置く方が stage 3 の bit-identity が自明になる。
* **stage 3 で $c_1 \equiv 0$ → `live = False` → $\log\tilde p$ は on-task 教師の bit そのまま**(既存の恒等経路)。
  これが「control と同一の到達点」の実装上の根拠で、テストで検証する(§7)。

### 3.2 性質(構成上)

1. **有界・clamp 不要.** $p_{on}e^{c_k} = p_0 e^{a_k}$ で、$a_k$ は $h_d$ と同符号・$|a_k| \le |h_d|$ だから
   $p_0 e^{a_k} \in [\min(p_0, p_{on}), \max(p_0, p_{on})]$。1 候補が $e^{5}$ を超えて動くことは、
   その候補で on-task 教師自身がそれ以上動いている場合に限られ、そのときの到達点は $p_0$ 以下。`_EXPONENT_CLAMP` は
   **この mode では外す**(残すと、教師が強く抑制した候補で base 質量の復元が $148\times$ で止まり、stage 1 の目標が
   「base」でなく「教師寄りの中間」になる)。
2. **$\Delta\mathrm{KL}$ の上界.** 現行と同じ $|\mathrm{KL}(p_s\|\tilde p) - \mathrm{KL}(p_s\|p_{on})| \le 2\max|c_k| \le 2\max|h_d|$。
   これは on-task 蒸留自体の規模で、新しい大きさは持ち込まない。
3. **注入ゼロ.** 1. の帰結。どの候補も base と on-task 教師の間から出ない。webshop 型の害(他タスク固有成分の混入)は
   機構的に不可能で、残る害の経路は「教えるのを遅らせること」のみ(§4.4)。
4. **stage 1 の未裏付け候補では目標 = base.** 生徒が base から出発する限りそこでは KL の勾配はゼロ(生徒 = 目標)。
   生徒が報酬で動いた後は **KL-to-base の正則化**(RLHF の参照 KL と同形)として働く。すなわち
   stage 1 + GRPO ≡ 「GRPO + 参照 KL(参照 = base + 共有書式)」。既知の安定 regime で、stage 1 の下振れリスクは小さい。
5. **裾ゲートとしての作用(理論 §3.5・§5.3b との関係).** reverse KL の勾配が最大なのは「生徒が確信し教師がほぼ 0」の候補で、
   klw アームはそこを増幅して entropy を上げた。stage 1 でそこが蒸留されるのは **3 本全員が抑制した候補だけ**。
   裾の推定雑音 $\varepsilon_m$ は教師間で独立なので 3 本同符号かつ大きいことは稀(§2.3: 帰無で 25%、大きさで 10%)。
   理論 §5.3b の裾ゲートと同じ向き(裾を**切る**)に、追加パラメータ $\kappa$ なしで働く。

### 3.3 連続解放

硬い切替は目標の跳びを作る(損失の段差、`actor/teacher_kl_loss` の不連続)。層が入れ子なので連続形は自然:

$$
\log\tilde p_t = \log p_0 + a_3 + \rho_2(t)\,(a_2 - a_3) + \rho_1(t)\,(a_1 - a_2) - \log Z_t,
\qquad
c_t = (\rho_2 - 1)(a_2 - a_3) + (\rho_1 - 1)(a_1 - a_2),
$$

$\rho_2, \rho_1 \in [0,1]$、$\rho_2$ が先に 0→1、次に $\rho_1$。$(\rho_2,\rho_1) = (0,0)$ が stage 1、$(1,0)$ が stage 2、$(1,1)$ が stage 3 = control。
$\rho$ は step の関数(§5)。$c_t$ は $\rho$ について線形なので、ramp 中も §3.2 の 1.–3. は保たれる
(凸結合は区間 $[\min(p_0,p_{on}), \max(p_0,p_{on})]$ を出ない)。

### 3.4 却下した代替: 位置マスク

「未裏付けの位置は KL を 0 にする」(位置スカラー $W \in \{0,1\}$、目標は $p_{on}$ のまま)も同じ意図を持つが、
(i) 位置粒度なので、書式候補と内容候補が同居する位置で内容の未裏付け分が漏れる、
(ii) 理論 §3.5 の通り位置重みは reverse KL の unlearning 裾をそのまま通す(本案は候補粒度でそれを base に戻す)、
(iii) stage 1 で「参照 KL」の正則化が消え、純 GRPO になる。3 点で候補粒度の目標変更が勝る。

---

## 4. 理論: この機構について言えること・言えないこと

### 4.1 固定点は control と同一(命題 1・2 の帰結)

stage 3 の目標は $\pi_d$。純 OPD なら固定点 $\pi_\theta = \pi_d$、OPD+GRPO なら $\pi_d\exp(\tilde Q/\tau)$、
どちらも control と一致。$\beta = 0.01$ では $A \ne 0$ の位置は報酬支配(理論 §4.5)なので、
**教師側の機構が効く場所は $A = 0$ の位置と動学の初期に限られる**。本案が変えるのは後者。
理論 §6.1 の anneal への注記「到達点が同じなら anneal は full arm と区別できない」は、本案では
**対照が control**(full arm ではない)なので当たらない。予測は「@300 で control と一致、差は中間にだけ」。

**理論 §4.3.1(2026-09-03 追加)がこれを測定で裏付ける。** 8/13 アームの発火率は全区間で平坦
(AGREE は step 25–75 で 0.256、225–300 で 0.237、相対 −7%)で、優位が消える 200 step 以降も初期の 9 割超の率で
当たり続けていた。「押され続けたまま到達点が並んだ」が実際の形である。帰結は本案にとって 2 つ。
(i) 到達点の一致は機構の消耗ではなく $\beta = 0.01$ の regime そのものなので、§4.1 の予測は測定に支えられている。
(ii) **step 91 で全額解放するスケジュールは、効く介入を捨てていない** — 捨てるのは「当て続けても優位を生まないと
測定された区間」である。

### 4.2 精度順序(モデルから出ること)

階層モデル下で、$n$ 本の教師が共有する成分の推定は $n$ 個の独立な観測に基づく。平均推定量なら分散 $1/n$、
min 推定量は §2.3 の通り $n=3$ で sd 0.22、$n=2$(和集合)で 0.56、$n=1$ で 1.0。
**カリキュラムは目標の成分を推定精度の高い順に解放する。** これは理論 §3.4「信頼は事後精度に比例するのが正しい」を、
位置重み(理論が否定した形)ではなく**目標成分の解放順**として実装したもの。$\beta$ は動かさないので、
理論 §5.1 原則 2(信頼は教師統計で決めない)に触れない。原則 1(教師統計は目標にだけ使う)には従う。
時間 $t$ は教師と独立な信号で、理論 §5.3a の $\beta(t)$ と同じ資格を持つ。

理論 §3.3 の Bayes 最適形(一様縮小・平均)を採らない理由は 1 つ: **漏れゼロ制約**。平均は $s_j/K$ を目標に混入させる
(理論 §3.3 末尾の「内容位置で他タスクの固有成分を混入させる」)。min は混入ゼロで、代償は $g$ の過小評価(一時的)。
webshop の一貫した悪化方向を止めることを優先する設計判断。

### 4.3 順序仮説(モデルから出ないこと — 事前登録する仮説)

線形化(NTK 近似)の下では、目標を時間で変えても最終到達点も到達速度もほぼ変わらない(勾配が成分ごとに加法的)。
本案が効くとすれば非線形・共有パラメータの regime で、機構は次の通り:

* **勾配整合.** $g$ は 3 タスク全ての行が同じ方向に押す成分。stage 1 ではミニバッチ内の 3 タスクの KL 勾配がその成分で揃い、
  タスク固有成分同士の干渉(無相関な方向への同時更新)が無い。
* **初期化の質.** base から「共有幹」を先に学んだ点を初期化として、タスク固有分を小さな偏差として学ぶ。

**介入「率」は説明の候補から消えている。** 理論 §4.3.1 が 8/13 アームの `frac_*` を全点引き、早期と後期で
相対 −7% しか違わないことを示した。早期優位が「早期に強く当たっていたから」では説明できないなら、
残る候補は介入の**内容と順序**である。本案はその残りを直接操作する唯一の機構になる。

これは仮説であり、証明ではない。純 OPD で位置重みが 150 step で効かなかった記録(理論 §2.2 末尾)は、
「位置別の歩幅」では順序が作れなかったことを示すが、本案は歩幅ではなく**目標の内容**を変えるので、同じ記録では反証されない。
検定は §6.2。

### 4.4 コストの理論的な見積り

stage 1–2 の間、on-task 単独分 $\hat s_d$ は蒸留されない。監査によれば内容語は OPD 予算の 24%(pooled)、
webshop は内容 role のシェア 0.55。**webshop は stage 1 で最も多くを保留する**。
到達点は同じなので、保留のコストは「webshop の立ち上がりが遅れる」形で現れ、stage 3 以降に回収されるはず。
回収されなければ(§6.2 の反証条件)、「順序の利得 < 保留の損失」で、この線は閉じる。

### 4.5 stage 1 の目標が「書式」であること

監査(§11.6、`opd/role/format/kl_share = 0.86`)によれば $g$ はほぼ書式。したがって stage 1 は事実上
「書式だけを教師から、内容は報酬から」。control の正規形式切替は step 135、8/13 のアームは 162 に**遅らせた**(理論 §4.3)。
本案は書式成分を単独で先に蒸留するので、**切替を早める方向**を予測する(§6.2 の識別予測)。
8/13 のアームと逆向きの予測なので、両機構の早期優位が同じ原因かどうかをここで分けられる。

---

## 5. スケジュール

### 5.1 推奨: step で事前登録、線形 ramp

| 区間 | $(\rho_2, \rho_1)$ | 目標 | 根拠 |
|---|---|---|---|
| step 1–40 | (0, 0) | $\pi_0 e^{a_3}$($g$ のみ) | 8/13 アームの早期優位は 31–60 で立った(理論 §4.3)。その窓の前半を $g$ だけで走る |
| 41–50 | ($\uparrow$, 0) | ramp | 10 step。1 step の成功率 SD 0.10–0.15 より短い窓で跳びを作らない |
| 51–80 | (1, 0) | $\pi_0 e^{a_2}$ | §2.3 の通り stage 2 の選択性は低く stage 3 寄りなので、stage 1 より短く |
| 81–90 | (1, $\uparrow$) | ramp | |
| 91–300 | (1, 1) | $\pi_d$ = control | OPD の効きは 150 step で頭打ち(理論 §4.5)。書式切替(control 135)より前に全額解放しておく |

91 step 以降を control と同一にすることの代償は、**測定上ほぼゼロ**である(理論 §4.3.1: 200 step 以降も満額で
当たり続けたアームが、その区間で優位を生んでいない)。前半に予算を寄せることは、後半を諦める取引ではない。

3 タスク共通のスケジュール(タスク別にすると比較が壊れる)。境界と ramp 幅が**この案の自由パラメータの全て**で、
現行の `exponent_scale`(1 個、単位換算)と交換した形になる。2 個増えたことは正直に記す。

### 5.2 代替: 習熟トリガ(第 2 run 以降)

stage $k$ の KL $D_k = \mathrm{KL}(\pi_\theta\|\tilde\pi^{(k)})$ のタスク平均が、10 step 窓で相対減少 < 5% になったら解放を始める。
第 1 run で採らない理由: (i) 第 2 のフィードバックループ(生徒 → 目標 → 生徒)が乗り、C7 撤回で既に 1 つある
(生徒 top-k support)、(ii) タスク間で非同期になり比較が壊れる、(iii) 「トリガが発火しない」失敗様式が増える。
第 1 run で $D_k$ をタスク別に**記録だけ**し(§6.1)、境界の妥当性を事後に読む。

### 5.3 cold start

$\sigma$ が有効になるまでの 1–2 step は現行と同じく $c \equiv 0$(= 全額 on-task、stage 3 の挙動)。
「stage 1 で始めるべき機構が最初の 2 step だけ stage 3」という矛盾はあるが、代替($\sigma$ 比を 1 として raw nats で min を取る)
は別の量を 2 step だけ混ぜることになる。2 step は雑音の内側なので現行踏襲を推奨。resume 時も同じ(sidecar 無し)。

---

## 6. 指標と事前登録

### 6.1 配線の検証と機構の診断(結果ではない)

| 指標 | 期待 |
|---|---|
| $\rho \equiv 1$ で `target/tv` = 0、`target/live_frac` = 0 | control と bit-identical(テスト + step 91 以降の run ログ) |
| `target/rho/{2,1}` | 指定スケジュールと一致 |
| `target/layer/{shared,pair,own}/backed_frac` = $\sum p_{on}\lvert a_k\rvert / \sum p_{on}\lvert h_d\rvert$ | 入れ子 $0 \le$ shared $\le$ pair $\le$ own $= 1$。**§2.3 の帰無表(0.105 / 0.48)の実データ版。** 保留率は $1 - [\text{shared} + \rho_{pair}(\text{pair}-\text{shared}) + \rho_{own}(1-\text{pair})]$ で厳密に導出できる(層が同符号で入れ子だから)ので、別指標にしない |
| `target/layer/{g,pair,own}/role/{format,tag,env_action,tool_call,reasoning}` | **予測**: $g$ 層の 80% 以上が format+tag。pair 層の内容 role シェアはこれが初測定(監査の base 統制後相関 −0.003〜+0.043 から、小さいと予測) |
| **保持質量比**(G1 の置換)`target/retained_shuffled_ratio` = $\sum p_{on}\lvert a_3^{sh}\rvert / \sum p_{on}\lvert a_3\rvert$ | shuffled off-task(`decorrelated_off_shifts`)で $a_3$ を作り直す。**1 に近いなら $a_3$ は雑音**、§2.3 の目盛りで 0.1–0.2 なら $g$ は実在。現行の `shuffled_tv_ratio` は本 mode では ≥ 1 になり(裏付けが減るほど差し引きが増える)意味が反転するので使わない |
| `target/tag_share`(G3) | stage 1 で**高い**ことを確認(警告ではない)。stage 3 で消える |
| `target/stage_kl/{1,2,3}` = $D_k$ タスク別 | §5.2 の事後読み。stage 1 の $D_1$ が早期に飽和するか |
| `target/log_z_mean`, `mass_error_max`, `max_abs_log_w` | 現行同様。`max_abs_log_w` は $\max\lvert h_d\rvert$ 級まで許容(clamp 無し) |
| `actor/teacher_kl_loss` | stage 1 で control より**小さい**(目標が base 寄りで生徒が base 近傍)。stage 3 で control と同水準に**上がる** |
| **`target/control/grad_cosine`**、`target/control/grad_norm_ratio` | live 目標の OPD 勾配と、同じ位置・同じ生徒での control(素の on-task)の OPD 勾配の余弦と norm 比(logit 空間、解析的、`opd/grpo/*` と同じ構成)。**「制限であって方向転換ではない」の直接測定。** 全額解放で恒等的に 1 / 1。stage 1 で余弦が**負**に振れる位置は、報酬が未裏付け成分の方向に生徒を動かした所を stage 1 が base へ引き戻している所 = §3.2-4 の「参照 KL」の読みの実測。role 別(`target/role/<role>/control/*`)で**内容 role に負が集中する**と予測 |
| `target/grpo/grad_cosine` と `target/control/grpo_grad_cosine` | live 目標の OPD 勾配 vs 報酬勾配、control の OPD 勾配 vs 報酬勾配。同一の位置・行重みで対にしたもの(klw アームの `kl_weight/grpo` 対 `opd/grpo` と同じ対)。カリキュラムが蒸留項と報酬の整合を変えるか |
| `target/kl_to_base` | KL(生徒‖base) を全位置で。`stage_kl/own`(KL(生徒‖教師))と並べると、生徒が base と教師の間のどこに居るかが読める。stage 1 では control より base 寄りに留まる(参照 KL 効果)と予測 |
| `target/tail/{cand_frac,student_mass_frac,three_frac,withheld_share}` | 裾領域($p_\theta > 0.5$ かつ $p_{on} < 10^{-3}$、理論 §3.5 の unlearning 裾)。`three_frac` = 裾候補のうち 3 教師全員が抑制したもの(stage 1 でも蒸留される割合)、`withheld_share` = 生徒質量で重み付けた保留量のうち裾に載る割合。**§3.2-5 の裾ゲート主張の直接測定**: `three_frac` が小さければ stage 1 は裾をほぼ base に戻している |
| `target/<dst>/pair_source/<src>/share` | pair 層 ($\lvert pair\rvert - \lvert shared\rvert$) の質量を、それを決めた off-task 教師のタスク名で分けたもの。**pair 層が常に同じ 1 本(例: search)で決まっているなら、stage 2 は 2 教師一致ではなくその教師の写し**。target 設計 D4 の問いの実測 |
| トークン表 `target/…/token/*`(`sign_tokens_step*.jsonl`) | **層で分類する**(`three / pair / own`、`state` 列)。「stage 1 は何のトークンを保留しているか」は層の分割でしか読めない。corroboration state での分類は tilt アームのまま; イベント dump の `state` は corroboration のまま(行は 4 モデルの確率を持つので層は事後に導出できる) |
| `target/withheld_smass_mean` | $\sum p_\theta \lvert c\rvert$ / 位置数。生徒の質量で重み付けた保留量 = reverse KL の勾配が感じる保留の大きさ。$p_{on}$ 重みの層質量(教師が何を書いたか)と対で読む |

**測らないと決めたもの(理由つき)。** (i) **パラメータ空間でのタスク間勾配整合**(§4.3 の順序仮説の文字通りの機構)。
タスク別の backward が要り(3 倍、または task 別 hook)、`ppo_micro_batch_size_per_gpu=5` のメモリ上限で step 時間と常駐量を変える。
§6.2 の事前登録した検定は結果水準(窓プロファイル・書式切替 step・サブタスク内訳)で、この量を要求しない。
logit 空間の `target/control/grad_cosine` が手の届く代理である。(ii) 保留率(上記の通り厳密に導出できる)。
(iii) イベント dump への層・ρ 列の追加(行の 4 確率と wandb の σ から層は導出でき、ρ は step の関数)。

### 6.2 結果の事前予測と反証条件

対照: 同一コミット・同一ホスト・同一 GPU 枚数・$\beta = 0.01$・生徒 top-k の control(target 設計 §7)。
評価: **タスク別成功率を、同一 checkpoint の反復雑音を尺度に読む。** 理論 §4.2.2 が 3 回の反復で
タスク別に直接測っており、桁が違う(復号設定の違いによる)。McNemar は主要評価から降ろす(§6.3)。
pooled は報告しない。
検証点: **@50, @100, @150, @300**(現行の 150 刻みでは中間が見えない。@50/@100 は val_only で後から取れる)。
alfworld は**サブタスク別内訳を必ず読む** — 検証ログに率が出ており分母(26 / 13 / 25 / 32 / 19 / 11)は率から
復元できるので、追加 run はいらない(理論 §4.2.3)。

| | 予測 | 外れたときの読み |
|---|---|---|
| @300 全タスク | control ± 2 SE(§6.3 の表) | 有意に上: 固定点近似が破れている(共有パラメータ効果が残る)。有意に下: 保留の損失が回収されていない |
| T=1.0 窓 1–40 | webshop ≤ control(保留コスト)、差 ≤ 0.05。alfworld ≈ control | webshop が control 以上なら保留コストは存在しない(良い外れ) |
| T=1.0 窓 41–100 | **順序仮説が正しければ** alfworld が 3 窓連続で control + 0.05 以上 | 1 窓も無ければ順序仮説は null。この線を閉じる |
| 正規形式切替 step(alfworld) | control(135)**より早い**(§4.5) | 遅いなら「書式を先に教えると書式採用が早い」が偽。8/13 アームと同じ向き |
| alfworld サブタスク内訳 @150 | 差が出るなら **control が最も弱い 2 つ**(cool→place 0.600、two obj 0.316)に局在する。8/13 アームの +14 問中 10 問がそこだった(理論 §4.2.3) | 局在しないなら「control が苦手な所で勝つ」形は本案では再現していない。別の局在なら、その形自体が新しい所見 |
| **反証条件** | @150 でいずれかのタスクが control より **2 SE 以上下**(alfworld 3.9pp、webshop acc 1.3pp、webshop score 4.0pp、search 0.6pp。§6.3) | 保留の損失 > 順序の利得。カリキュラムは採らない |
| webshop @150 以降 | control − 0.05 を下回らない(注入ゼロなので klw 型の悪化は無い) | 下回れば「害は注入ではなく別経路」。理論 §4.6 の webshop 読みを改める |
| entropy | control と同等(klw 型の上昇は無い。§3.2-5) | 上がれば裾ゲート解釈が誤り |

### 6.3 検出力(理論 §4.2.2 の実測値で書き直した)

* **検証値の尺度はタスク別で、桁が違う**(理論 §4.2.2、arm@300 の 3 回、同一 checkpoint・同一プロトコル)。
  1 draw 対 1 draw の差の SE は $\mathrm{SD}\sqrt2$:

  | metric | 検証時の復号 | 反復 SD | 差の SE | 読める下限(2 SE) |
  |---|---|---:|---:|---:|
  | alfworld (n=126) | T=0.4 標本抽出 | 1.39pp | 1.96pp | **3.9pp** |
  | webshop acc (n=126) | T=0.4 標本抽出 | 0.46pp | 0.65pp | **1.3pp** |
  | webshop score (n=126) | T=0.4 標本抽出 | 1.42pp | 2.01pp | **4.0pp** |
  | search (n=51713) | **T=0 貪欲** | (0.06pp は過小) | 0.30pp | **0.6pp** |

  **search の SD は使わない。** 3 桁丸めの値から出ており、$n = 51713$ の二項 SE 0.215pp を下回る。
  上の 0.30pp は二項 SE から取った($0.215\sqrt2$)。
  **$n = 3$ の SD は弱い推定**で、$\sigma$ の 95% CI は概ね $[0.52\hat\sigma, 6.3\hat\sigma]$(理論 §4.2.2)。
  webshop acc の 0.46pp は 3 値のぶれが 1 問しかないので特に運が良い数字で、**単独で 1.3pp を主張してはいけない**。
  堅く述べるなら range 由来の見積り(alfworld 2.40pp → 3.4pp)も併記する。8/13 アームの +11.1pp は
  この尺度で 3.3σ(range 由来)〜5.7σ(SD 由来)だった。
* **McNemar は主要評価から降ろす.** 両アームは同一の 126 問(同じ val set、`data.seed=1`)を採点するので、
  問題サンプリング分散は差の中で既に相殺されている。McNemar の z は不一致対 $b+c$ に対し最良でも 3.74σ、
  現実的な $b+c = 20$–30 では 2.6–3.1σ で、反復 SD 経由の読みを上回らない(理論 §4.2.2)。
  **`scripts/val_paired.py` は「どの問題が反転したか」を見るために使う** — 検定ではなく記述として。
* **サブタスク分解は局在の記述で、検定ではない.** n は 11–32 なので個々の差は二項雑音の内側。
  担保するのは合計の問題数。
* **学習中の窓平均.** 差の SE ≈ 0.05。順序仮説の検定は「連続窓」の一貫性に依る(§6.2)。可能なら 2 シード。
  **本案の主要な識別はここで起きる**: 検証値は @50/@100 の 2 点しか中間に無いが、T=1.0 の学習曲線は
  300 点あり、理論 §6.4 が E−1 を「限界的」に落とした理由もそれである(各 step の alfworld は
  15 プロンプト × `env.rollout.n=8` = 120 本で、検証の 126 問と標本サイズがほぼ同じ)。
* **中間 checkpoint(25 step 刻み)を必ず保存する.** この機構は中間にしか差が出ない予測なので、
  @150/@300 だけでは検定不能。

### 6.4 位置づけ(理論 §6.4 の判定が変わった後で)

理論 §6.4(2026-09-03)は **E−1 を「限界的」に落とした**: 形の問題は学習曲線が既に 300 点で答えており、
追加の検証 run から得られるものは小さい。残る問いは「なぜ step ≤ 60 でアームが先行したか」で、
**答えに届くのは学習 run だけ**である。初版の本節は E−1 を前提に本案の優先度を条件づけていたが、その前提は消えた。

候補は 2 本あり、目的が違う。

| 候補 | 何に答えるか | 対照 |
|---|---|---|
| 8/13 設定の dump 付き再現(seed 2、$\beta=0.01$、`agree×1.25` 両符号、`conflict×0.75`、教師 top-k) | **既存アームの**早期優位の中身。理論 §4.3 / §4.6 が「未測定」と書いた箇所 | 8/5 control(既存)か新規 |
| **本案** | 「順序 = 裏付けの本数」という**特定の答え**が早期優位を生むか | 新規 control(D5) |

**推奨: 本案を先に走らせる。** 理由は 2 つで、どちらも解釈のコストの話である。(i) 本案は固定点を動かさないので
@300 の一致が予測であり、外れたときの読みが一意(§6.2)。再現 run は目標を動かし続けるので @300 の解釈が要る。
(ii) 本案は現行コードの assert(target mode で `disagree_weight != 1` を拒否)に触らない。再現は 8/26 以降の
コード変更を巻き戻す必要がある(理論 §5.6)。

ただし**本案が null でも 8/13 アームの早期優位は未説明のまま残る**ので、再現の価値は独立に保たれる。
E1($\beta(t)$)との関係は初版のまま: 「時間で何を変えるか」が違う(E1: 信頼 $\beta$、本案: 目標の成分)。
両方走れば 2×2 になるが、まず本案単独。

---

## 7. 実装(2026-09-03 実装済み)

**変更点は 1 モジュール + 配管 3 箇所。** 新しいキャッシュ・新しい教師読みは無い(行あたり 4 モデル読みは現行通り)。
実装は `claude/cross-teacher-target` の tip を基点にしている。

1. `verl/trainer/ppo/cross_teacher_target.py`
   * `MODES = ("tilt", "curriculum")`、`LAYERS`、`LAYER_BRANCHES`。**新モジュールではなく mode キー**にした:
     正規化経路・`TargetStepStats`・トークン表・イベント dump・3 アーム相互排他 assert・ペアリングテストが
     そのまま再利用でき、control との差分が機構キーだけで済む。
   * `curriculum_rho(step, stage_steps, ramp_steps) -> {"pair", "own"}`: §5.1 の線形 ramp。**純関数**で、
     ramp が重なる設定を assert で拒否する。
   * `nested_layers(shift_on, hat_off, sigma_on) -> {shared, pair, own, branch}`: §2.2。
     **`own` は生の shift** で、$\sigma\cdot\hat h$ で作り直さない(そうすると全額解放が bit 一致しない)。
     off-task 側を on-task nats に変換して比較する。
   * `curriculum_exponent(layers, rho_pair, rho_own) -> c`: §3.3。**差分形で書く** —
     $(\rho_{pair}-1)(pair-shared) + (\rho_{own}-1)(own-pair)$。代数的に等しい $a - own$ は
     $\rho=(1,1)$ で厳密に 0 にならない(`shared + (pair-shared)` は `pair` と bit 一致しない)。
   * `normalized_weight(..., clamp=None)`: clamp を引数化し、curriculum では外す(§3.2-1)。
   * `build_target(mode=..., rho=..., curriculum_counterfactuals=...)`。`exponent_scale` は curriculum では
     受け取らず、config に残っていれば actor が assert で落とす。
   * `standardize_policy_shifts` が `sigma_on` を返すようになった(index の clamp と mask を呼び出し側で
     再実装しないため)。既存 2 アームには後方互換。
   * `TargetStepStats(mode=...)`: 列を mode ごとに切り替える。curriculum では層別 `backed_frac`・層×role・
     保持質量比・stage 別 KL・層 branch・`kl_to_base`・裾領域・pair 層の供給元・生徒質量重みの保留量を足し、
     チャネル分割と shuffled TV と clamp を**落とす**(構造的にゼロな列は測定に見えるので、0 を出すのではなく列を作らない)。
   * `curriculum_gradient_terms` / `curriculum_gradient_metrics`: live 目標・control 目標・報酬の 3 方向の
     logit 空間勾配の全ペア積(6 項)。`opd_logit_push` と `logit_gradient_terms` を再利用し、報酬勾配の式は 1 箇所のまま。
     pooled・task 別・role 別(curated 3 種)。
   * `TokenStateCounts(state_names=..., acted_states=...)`(`sign_weights.py`): 状態語彙をインスタンス単位に。
     既定は従来の 7 状態で後方互換。curriculum のトークン表は層で分類し、`scripts/sign_token_scan.py` は層名の表も出す。
2. `verl/trainer/ppo/opd_ray_trainer.py`: `__init__` で mode とスケジュールを読み、
   ramp の重なりと `total_training_steps` との整合を assert。`fit()` で `update_actor` の直前に
   `batch.meta_info["cross_teacher_curriculum_rho"] = (ρ_pair, ρ_own)`。
3. `verl/workers/actor/dp_actor.py`: mode と ρ を読み(`update_policy` の先頭で 1 回、micro-batch 分割で
   `meta_info` が消える経路があるため)、`build_target` に渡し、`roles` を stats に渡す。
   curriculum 用の 1 行ゲート出力を追加(tilt のゲート行とは別 — A/B シェアも clamp も
   shuffled/live も、この mode では意味が無い)。
4. `examples/opd_grpo_trainer/run_multitask_cross_teacher_curriculum_qwen3.sh` と
   `expected_multitask_cross_teacher_curriculum_config.yaml`(機構キーは
   `enable / mode / stage_steps / ramp_steps / base_path`、`exponent_scale` は**無い**)。

**テスト(`tests/trainer/test_cross_teacher_curriculum{,_arm}.py`、46 件)。** 設計の主張と 1 対 1:

* 入れ子 $|shared| \le |pair| \le |own|$、非零なら符号一致、3 層の和が $h_d$ に 1e-12 以内。
* **注入ゼロ**: 全 stage・ramp 中も $p_{on}e^c$ が候補ごとに $[\min(p_0,p_{on}), \max(p_0,p_{on})]$ 内。
* **全額解放が control と bit 同一**($c$ が厳密に 0、`live` が空、`target_logprob` が on-task と `torch.equal`)。
* clamp 無しで 12 nats の抑制が base まで戻ること(tilt の clamp 5.0 を超える)。
* 複製不変、符号反転境界と $\rho$ についての連続性、cold start が no-op、per-task $\sigma$ が独立。
* スケジュールが step の純関数であること(resume 正当性)、境界が事前登録値と一致、単調・有界、
  ramp 重なりを拒否。
* stage 別 KL が**損失自身の reverse KL 関数**で計算した値と一致すること、全額解放後も測り続けること。
* 保持質量比が shared 層自身の比であること、curriculum の列が出て tilt の列が出ないこと。
* 対 control・対 tilt の差分が機構キーのみ、ディレクトリ非共有、$\beta$ と support が 3 アームで一致、
  スケジュールが run に収まること、中間 checkpoint が各 stage に入ること、
  **composed config + `inject_distillation_config` + intent lock が通ること**(= この設定で実際に起動する)、
  driver が ρ を送り actor が `curriculum_rho` を import していないこと(resume 正当性をコードの性質として)。

**既知の無関係な失敗**: `tests/trainer/test_cross_teacher_kl_weight.py::test_the_reliability_pass_runs_on_realistic_shapes_and_files_the_right_cells`
はこの実装の前から失敗している(`sign_cache_ids` の KeyError)。基点コミットで stash して確認した。

---

## 8. 決めたこと(2026-09-03、A–G は推奨で確定)

| # | 論点 | 推奨 | 理由 |
|---|---|---|---|
| D1 | stage 境界と ramp | 40 / 80、ramp 10(§5.1) | 早期優位の窓(31–60)、書式切替(135)、OPD 頭打ち(150)から。**唯一の自由パラメータ**なので事前登録が要る |
| D2 | stage 2 の集約 | max(和集合) | 要求「何か 1 つ」に一致。平均は「2 本一致」の意味を薄める(§2.2) |
| D3 | tail | on-task 形のまま | stage 3 の恒等が自明。カバレッジ 98%+ で差は小さい |
| D4 | cold start | $c \equiv 0$(現行踏襲) | 2 step、雑音の内側(§5.3) |
| D5 | 対照 | 新規 control(同一コミット・ホスト・GPU) | 予測「@300 一致」は同条件の control でしか検定できない。既存 control 流用は target 設計 §0.2 の失敗を再現する |
| D6 | 中間検証 | @50, @100 を追加、25 step 刻みで checkpoint 保存 | 差は中間にしか出ない予測(§6.3) |
| — | **未決 (D)** | control を並列に走らせるか逐次か | GPU の空きで決まる。実装には影響しない |
| D7 | ablation「stage 1 のまま解放しない」 | 任意(第 2 run) | 「$g$ + GRPO だけで control に届くか」= 書式以外を教師から学ぶ必要があるかの直接検定。理論 §3.3 は届かないと予測 |
| D8 | $\beta$ | 0.01 のまま | 比較可能性。stage 1 で `teacher_kl_loss` が小さくなるのは目標が近いからで、$\beta$ を上げる理由にはならない |

---

## 9. 限界(正直な見通し)

* **動学の利得に証明は無い**(§4.3)。到達点不変は証明できるが、それは「効かない可能性が高い」ことの別の言い方でもある。
  理論文書 §5.2 と同じ立場: 理論的に整合した形で null なら、裏付け系の順序機構を走らせる理由は無くなる。
* **識別不能性はそのまま。** $\hat s_d$ の中の $s_d$ と $\varepsilon_d$ は分けられず、stage 3 は control と同じ雑音を蒸留する。
  本案が減らすのは「雑音を**早く**蒸留すること」だけ。
* **§2.3 の数字は等分散ガウスの目盛り。** 実際の重い尾での漏れは保持質量比で測る(§6.1)。
* **生徒 top-k support のフィードバックループ**(C7 撤回)は残る。
* **stage 2 の選択性は低い**(帰無で 48% 通過)。stage 2 が「効く」ことを主張するには、pair 層の role 別シェアが
  内容 role に載っていることの確認が要る(初測定)。
* **スケジュールは事前登録した推測**で、原理的な決め方(習熟トリガ)は第 2 run 以降。
* **単タスク参照との差**は理論から埋まらない(理論 §7)。一次の変数は $\beta$ のスケジュールと step 数のまま。
* **順序の説明が単一機構で済まない可能性がある。** 理論 §4.2.3 は 8/13 アームの内訳が step で入れ替わることを示した
  (@150 の主役 two obj は @300 で control に逆転され、cool→place と heat→place はアームが保つ)。
  「早く効く所」と「最後まで保つ所」が別なら、裏付けの本数という単一の順序づけでは両方を説明できない。
  本案が @150 で局在を再現しても、@300 の保持まで説明できるとは限らない。

---

## 付録 A: §2.3 のシミュレーション

純 Python、seed 0、$N = 4\times10^5$。on-task を添字 0 とし、$h_m = g + \mathcal N(0,1)$ iid。

```python
import random, math
random.seed(0); N = 400_000
def nested(h):
    sd = 1.0 if h[0] > 0 else -1.0
    ab = [abs(x) for x in h]
    a1 = h[0]
    a3 = sd * min(ab) if (h[1]*sd > 0 and h[2]*sd > 0) else 0.0
    p1 = min(ab[0], ab[1]) if h[1]*sd > 0 else 0.0
    p2 = min(ab[0], ab[2]) if h[2]*sd > 0 else 0.0
    a2 = sd * max(p1, p2)
    return a1, a2, a3
for g in [0.0, 0.5, 1.0, 2.0, 3.0]:
    acc = [0.0]*9; n2 = n3 = 0
    for _ in range(N):
        a1, a2, a3 = nested([g + random.gauss(0, 1) for _ in range(3)])
        for i, a in enumerate((a1, a2, a3)):
            acc[i] += abs(a); acc[3+i] += a; acc[6+i] += a*a
        n2 += a2 != 0; n3 += a3 != 0
    E = lambda s: s / N
    print(g, [round(E(acc[i]), 3) for i in range(3)], [round(E(acc[3+i]), 3) for i in range(3)],
          [round(math.sqrt(E(acc[6+i]) - E(acc[3+i])**2), 3) for i in range(3)], round(n2/N, 3), round(n3/N, 3))
```
