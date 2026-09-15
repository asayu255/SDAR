# 一様縮小による cross-teacher 目標(uniform shrinkage / 幾何混合)

状態: **設計 + 実装済み(2026-09-14)。未走行。** 事前登録は intent lock
(`examples/opd_grpo_trainer/expected_multitask_cross_teacher_shrink_config.yaml`)で機械的に固定。
実装は §7、限界は §9。

前提文書: [cross_teacher_theory.md](cross_teacher_theory.md)(理論。命題 1–3、階層モデル、識別不能性、§3.3 と §5.2 = 本案の導出、§6.1 の E3)、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)(監査。$g$ ≈ 書式、$s_{gen}$ の +0.092)、
[cross_teacher_curriculum_design.md](cross_teacher_curriculum_design.md)(**符号が逆の姉妹アーム**: 注入ゼロ・固定点不変)。
対象コード: `verl/trainer/ppo/cross_teacher_target.py`(`mode="shrink"`: `shrink_exponent` / `build_target` / `TargetStepStats`)、
適用点 `verl/workers/actor/dp_actor.py`。アーム: `examples/opd_grpo_trainer/run_multitask_cross_teacher_shrink_qwen3.sh`。

要求(2026-09-14): **on-task の蒸留に off-task 教師の KL を混ぜる。**

---

## 0. 結論(先に)

1. **逆 KL の下では「KL を混ぜる」と「目標を幾何混合に置き換える」は同一の機構である。** 重みの和が 1 なら
   $\sum_m w_m \mathrm{KL}(\pi_\theta\Vert\pi'_m) = \mathrm{KL}(\pi_\theta\Vert\tilde\pi) - \log Z$ で、$\log Z$ は
   生徒に依らないので勾配が一致する(§1、テスト `test_mixing_the_kls_is_the_mixed_target`)。設計判断ではなく表記の違いなので、
   正規化・tail・診断の配管がある目標側に書く。
2. **自由パラメータは $\lambda'$ 1 個で、範囲は理論が決める。** $\lambda' \in [1/K, 1] = [1/3, 1]$。$\lambda'=1$ は control と
   bit-identical、$\lambda'=1/3$ は 3 教師の等重み幾何平均 = **目標から routing が消えた形**(= 純多教師 OPD / MOD)。
   **1 つのスカラーが MOPD の routing とその最も自明な ablation を両端に持つ。**
3. **本アームは固定点を動かす。** これが curriculum アームとの決定的な差である(あちらは注入ゼロ・到達点は control と同一)。
   したがって @300 が有効な読み取り点になる代わりに、注入が誤っていれば恒久的なバイアスになる。
4. **事前予測は null で、それはオフラインで測ってある(§4)。** 監査 §4.6 と同じ推定量・同じダンプ・同じ cluster bootstrap で
   $\lambda'$ を掃引すると、**許容範囲 [1/3, 1] の全域で control との対応差がゼロ**。動くのは $\lambda'=0$ だけで、そこは階層モデルの範囲外。
5. **それでも走らせる価値は baseline としてある。** 「教師を平均するだけではどうか」は curriculum アームの査読で必ず出る問いで、
   関連研究 §3.2 の対抗表に既に立っている(MOD / 報酬無しの純多教師 OPD)。第 1 run は $\lambda'=0.6$ で走らせる(§5 の決定経緯)。

---

## 1. 機構

生徒 top-20 の support $S(x)$ 上、タスク $d$ の位置ごとに

$$\log\tilde\pi_d(v) = \log\pi_0(v) + \sigma_d\Big[\lambda'\,\hat h_d(v) + (1-\lambda')\,\underset{m\ne d}{\mathrm{mean}}\;\hat h_m(v)\Big] - \log Z(x)$$

既存の tilt パラメタ化 $\log\tilde p = \log p_{on} + c - \log Z$ に落とすと

$$c(v) = (1-\lambda')\big(\sigma_d\,\underset{m\ne d}{\mathrm{mean}}\;\hat h_m(v) - h_d(v)\big)$$

正規化は既存経路のまま: $\tilde p = p_{on}e^c/Z$、$Z = \sum_{v\in S}p_{on}e^c + p_{\text{tail}}$、**tail は tilt せず $Z$ にだけ入る**。

### 1.1 「KL を混ぜる」との同一性

重み $w_d = \lambda'$、$w_{m\ne d} = (1-\lambda')/(K-1)$、$\sum_m w_m = 1$ に対し

$$\sum_m w_m \mathrm{KL}(\pi_\theta\Vert\pi'_m) = \sum_v \pi_\theta(v)\Big[\log\pi_\theta(v) - \sum_m w_m\log\pi'_m(v)\Big] = \mathrm{KL}(\pi_\theta\Vert\tilde\pi_d) - \log Z$$

**含意 3 つ。**

* 損失側に項を足す実装は、各教師の正規化分布を support 上で別々に持つ必要がある分だけ高コストで、得られる勾配は同じ。
* **重みの和を 1 にしないと $\beta \to \beta(1+w)$ になる。** 理論 §4.5 の通り係数は一次・混合は二次なので、正規化しない版は
  「混合の効果」と「教師を強めた効果」を分離できない。しかも後者の方が大きい。
* 逆 KL を混ぜると**幾何混合**(product of experts)、順 KL を混ぜると**算術混合**になる。MOPD は token 別 reverse KL なので前者。

### 1.2 $\sigma_d$ による単位変換

教師ごとに KL 係数が違う(search 0.001、他 0.01)ので、off-task の声を**宛先タスクの nats に変換してから**混ぜる。
これは利得を揃えた教師 $\pi'_m \propto \pi_0(\pi_m/\pi_0)^{\sigma_d/\sigma_m}$ に対する混合と厳密に等価で、$\pi'_d = \pi_d$ である。
変換しないと「base から遠くへ動いた教師の声が大きい」だけの機構になる。

### 1.3 $\lambda'=1$ が bit-identical であること

$c$ を**生の** $h_d$ からの引き算として書いてあるので、$\lambda'=1$ で係数 $(1-\lambda')$ が厳密に $0.0$ になり
$c \equiv 0$ → `live=False` → 損失は on-task 教師の bit をそのまま受け取る。$\sigma_d[\lambda'\hat h_d + \dots]$ と書くと
$\sigma\cdot(h/\sigma) \ne h$ の丸めが残る。`nested_layers` が同じ理由で同じ書き方をしている。

---

## 2. 理論(なぜこの形か)

階層モデル $h_m = g + s_m + \varepsilon_m$(理論 §3.1)で目標は $h^\star_d = g + s_d$。平坦事前の下で

$$\mathbb E[h^\star_d\mid h] = \lambda' h_d + (1-\lambda')\bar h_{-d},\qquad
\lambda' = \frac{1+(K-1)\lambda}{K},\qquad \lambda = \frac{\sigma_s^2}{\sigma_s^2+\sigma_\varepsilon^2}$$

* **位置に依らない一様な $\lambda'$。** 符号ゲート・全会一致・min・deadzone はこの式から出てこない。tilt アームの
  チャネル・q ゲート・deadzone がすべて消えるのはそのため。
* **共有成分の「除去」は誤り。** contrastive 形は $g$ を目標から引き去るが $g$ は目標の一部で、監査は同一標本で悪化側を測っている
  (sg1 −0.042、xt1 −0.035)。
* **範囲 $[1/K, 1]$。** $\lambda'=1/K$ は $\sigma_s^2=0$、$\lambda'=1$ は $\sigma_\varepsilon^2=0$。
  **$1/K$ 未満は on-task 教師を off-task より軽く扱うことで、どの事後分布もそうしない** — $s_d$ を観測した唯一の教師だから。

### 2.1 $\lambda'$ は本来「測る」量である

識別には $\sigma_s^2$ と $\sigma_\varepsilon^2$ の分離が要るが、教師 3 本 + base の 2 次モーメントが識別するのは和だけ(理論 §3.2)。
分離には **on-task 教師の第 2 シード**が要る:

$$\mathrm{Var}(\hat h_m - \bar{\hat h}) = (1-\tfrac1K)(\sigma_s^2+\sigma_\varepsilon^2),\qquad
\mathrm{Var}(\hat h_d^{(1)} - \hat h_d^{(2)}) = 2\sigma_\varepsilon^2$$

第 1 run はそれを持たないので $\lambda'=0.6$ は**事前**である(§5)。理論 §6.1 の E4 がこの測定にあたる。

---

## 3. curriculum アームとの対比(姉妹アーム、符号が逆)

| | curriculum | **shrink(本案)** |
|---|---|---|
| $c$ の向き | on-task から未裏付け分を**引く** | off-task の声を**足す** |
| 上下界 | $p_0$ と $p_{on}$ の間(構成上) | **無界** |
| 注入 | **ゼロ** | あり(`target/shrink/beyond_*` で計数) |
| clamp | 不要(外してある) | **必要**(`_EXPONENT_CLAMP=5.0` を残す) |
| 固定点 | control と同一 | **動く** |
| 主張 | 順序(中間 step でしか読めない) | 目標そのもの(@300 が読み取り点) |
| 自由度 | スケジュール 3 個 | $\lambda'$ 1 個 |

この 2 本で「off-task 教師をどう使うか」に **制限する / 無視する(control) / 注入する** の 3 点が揃う。

---

## 4. 事前に測ったこと(オフライン $\lambda'$ 掃引、GPU 不要)

監査 §4.6 の推定量そのまま — 発火トークンでの目標 shift と advantage の partial Spearman($\log p_{base}$ 統制、
(step, dst) クラスタブートストラップ 3000)。$\lambda'=0$ が監査の公表値を再現することを確認済み
(xt1 +0.0922 [+0.0055, +0.1767]、sd 比 2.33、タスク別 alfworld +0.152 / webshop +0.118 / search −0.103)。

**xt1(n=602、clusters=156)、対応差は同一クラスタでのブートストラップ**

| $\lambda'$ | score [95% CI] | control($\lambda'=1$)との対応差 | $\lvert c\rvert$ 中央 [nats] | clamp 超 |
|---:|---|---|---:|---:|
| 1.000 | −0.030 [−0.123, +0.059] | — (定義) | 0.00 | 0.000 |
| 0.900 | −0.027 [−0.122, +0.064] | +0.003 [−0.022, +0.029] | 0.44 | 0.000 |
| **0.600** | **−0.026 [−0.116, +0.072]** | **+0.004 [−0.035, +0.047]** | **1.76** | **0.151** |
| 1/3 | −0.004 [−0.104, +0.093] | +0.025 [−0.035, +0.085] | 2.94 | 0.331 |
| 0.000 | +0.092 [+0.006, +0.177] | +0.122 [+0.004, +0.241] | 4.41 | 0.455 |

sg1(n=662、clusters=86)も同形で、$\lambda'$ = 0.9/0.8/0.6/0.5/(1/3) の対応差は +0.017/+0.013/+0.019/+0.023/+0.023、すべて 0 を跨ぐ。

**読み。**

1. **理論が許す範囲 [1/3, 1] は、off-task の信号が既に壊れている領域の内側に丸ごと入っている。** 監査が「3 教師平均にすると
   信号が消える(−0.004)」と書いたのは knife-edge ではなく**内部全域**である。
2. 壊れ方は分散支配。sd(ĥ_on)/sd(ĥ_off) = 2.33、corr +0.72 で、$\lambda'=0.9$ で既に混合の sd の 96% が on-task 由来。
3. **順序は粒度で反転する。** 同じダンプを span 単位で読むと xt1 では $\lambda'=1$ が最高(+0.052)、$\lambda'=0$ が最低(−0.109、
   対応差 −0.161)。sg1 は逆向き。**「$\lambda'=0$ が最良」はデータに安定して存在しない。**
4. 予測力は format トークンに載る($\lambda'=0$、xt1 format n=533 で +0.096 [+0.008, +0.182]、非 format n=69 は null)。
   format は on-task 教師が単独で既に教えている成分(監査 §11.6)なので、混合で新しく入るものは無い。

**この検定の弱さ(明記する)。** 推定量は control の目標自身に −0.03 を与える。標本はターン 0 の発火トークンのみ、
2 本とも機構 ON の run、ダンプは極値 3 層(top_shift / top_push / spread)からの抽出。**必要条件のゲート**であって
「効かないことの証明」ではなく、特に**動学(早期の順序)には原理的に触れていない**。

---

## 5. $\lambda'$ の決定(第 1 run = 0.6)

掃引が平坦なので、$\lambda'$ は測定からは決まらない。候補は 3 つだった。

| 値 | 根拠 | 性質 |
|---|---|---|
| 1/3 | 許容範囲の端点 | 単調性より**族全体を bound** できる。null の解釈が一意。ただし clamp が候補の 1/3 で binding |
| **0.6** | **利用者判断(2026-09-14)** | $\lambda = 0.4$ の事前。内部点なので null は「効かない」「dose 不足」の両方と整合する |
| 0.8 | 理論 §6.1 の E3 | 事前登録との継続性。dose 最小 |

**0.6 を選んだのは利用者の判断であり、測定の帰結ではない。** これを書いておくのは、null が出たときに
「dose が足りなかった」という対抗説明を**排除できない**ことを事前に確定させるためである。排除したければ第 2 run を 1/3 で走らせる。

---

## 6. 事前登録

### 6.1 配線の検証(結果ではない。最初の数 step で読む)

| 指標 | 期待値 |
|---|---|
| $\lambda'=1$ で control と bit-identical | `target/tv = 0`、`live_frac = 0`(テストで固定済み) |
| `target/shrink/c_to_on_absmass` | $\lambda'=0.6$ なので $(1-\lambda')\lvert\text{off}-h_d\rvert/\lvert h_d\rvert$。**構成上決まる量**で、run はサイズを確認するだけ |
| `target/shrink/beyond_cand_frac` | **> 0**。ゼロなら注入が起きておらず、curriculum アームを別名で走らせている |
| `target/mass_error_max` | ~1e-6。$\sum\tilde p = 1$ は構成なので、これは assertion |
| `target/clamped_mass_frac` | 読む(count 比は 1.9–3.1 倍過小評価する) |
| `target/shuffled_tv_ratio` | **G1 は本モードでも反転する**(§6.4 の訂正)。1 を超えるのが既定で、1 に近い方が「2 つの shift が無相関」を意味する |
| `target/shrink/role/content_share` | **確認項目**。内容 role に載ることが仮説の検定対象であり、警告ではない |

### 6.2 結果の予測と反証条件

* **予測: @75 と @150 のどのタスクも control と検証雑音の内側で一致する。** 根拠は §4 の掃引(許容範囲全域で対応差ゼロ)。
* **反証条件: いずれかのタスクが @75 と @150 の両方で、検証床の 2 倍を超えて control を上回る。**
* **null は「dose 不足」を排除しない**(§5)。$\lambda'=0.6$ の $\lvert c\rvert$ 中央値は発火トークンで約 1.8 nats で小さくはないが、
  端点で測っていない以上、bound にはならない。
* **entropy の向きは初版の予測と逆だった**(§6.4)。幾何混合は product of experts で連言的なので、目標は on-task 教師より
  **鋭く**なる(`target/entropy_delta` = −0.039 を実測)。「目標が base 寄りになるので entropy が上がる」という初版の理由づけは
  誤りで、撤回する。生徒側の entropy がどちらへ動くかは目標の entropy から直接は出ないので、`actor/entropy` を実測して報告する。
  klw の利得が監査 §15 で entropy 上昇(探索)に帰属されている以上、**どちらへ動いてもその交絡は残る**。

### 6.3 検出力と採点

* **決定的検証で採点する**(prefix caching / CUDA graph / async・merge・pump 生成をすべて off)。同一 checkpoint の反復が
  byte-identical になるので、測定雑音はゼロにできる。1.6 pp の床は infra 由来だった。
* **fast mode の数字と混ぜない**(同一 checkpoint で 0.754 対 0.774、約 2 pp ずれる)。arm と control を同じ設定で採点する。
* 残る不確実性は**学習シード**で未測定。n=1 対 n=1 で言えるのは「この 2 本の重みは違う結果を出す」までで、
  「機構の効果」と言うには第 2 シードが要る。
* **ALFWorld のゲーム順はホスト依存。** control を別ホストで採点した数字と直接比較しない(sort 修正 5a62ed9 以降の checkout のみ一致)。

---

## 7. 実装(2026-09-14 実装済み)

| 場所 | 内容 |
|---|---|
| `cross_teacher_target.py` | `MODES += ("shrink",)`、`shrink_exponent()`、`build_target` の分岐、`TargetStepStats._SHRINK_ONLY` と描画 |
| `dp_actor.py` | `lambda_prime` の読み取りと範囲 assert、`exponent_scale` の**不在** assert、`build_target` への引き渡し、shuffled 反実仮想を shrink でも有効化 |
| driver | **変更なし**。$\lambda'$ は静的な config で、curriculum の $\rho$ と違い step の関数ではない |
| `tests/trainer/test_cross_teacher_shrink.py` | 13 件。bit-identity、幾何混合との一致、**KL 混合との同一性**、単位変換、注入の計数、clamp、step 表 |

## 8. 走らせ方

```bash
SEARCH_URL=http://100.86.45.30:8000/retrieve \
STOP_AFTER_STEPS=150 RUN_TAG=sh1 \
bash examples/opd_grpo_trainer/run_multitask_cross_teacher_shrink_qwen3.sh
```

`total_training_steps` は 300 のまま(LR スケジュール・データ順・step 索引を比較対象と揃えるため)で、
`stop_after_steps` が 150 で止める。resume すれば 300 まで続く。

## 9. 限界(正直な見通し)

* **事前予測が null で、その null から学べることが少ない。** 内部点なので dose 不足と区別できない(§5)。
* **$\lambda'$ が測定ではなく事前である。** 原理的に決めるには第 2 シード(E4)が要る。
* **固定点が動く**ので、注入が誤っていれば恒久的なバイアスになる。curriculum アームの安全性(到達点不変)は捨てている。
* **150 step は transient しか見せない。** 本アームの自然な読み取り点は漸近側で、@150 の null は機構の否定ではない。
* **$\beta = 0.01$ では $A\ne0$ の位置は報酬支配**(理論 §4.5)。目標変更が効く窓は $A=0$ の位置と初期動学に限られる。
* **タスク間で dose が違う。** off-task の声を $\sigma_d$ で宛先の nats に直すので、教師の shift が小さいタスクほど
  off-task が小さく変換される。`target/<task>/shrink/*` で読めるが、$\lambda'$ は 3 タスク共通で補正されない。
* **新規性は低い。** 幾何混合は MOD(Shi+ 2024)そのもので、routing + token 別多教師蒸留は LS-MOPD(Xie+ 2026)が既にやっている。
  本アームの価値は機構の新しさではなく、**この教師集合に書式以外の共有知識があるかを、最も弱い仮定で測ること**にある。
* **オフライン掃引は動学に触れていない**(§4 の弱さ)。

---

## 6.4 初回起動で観測された値と、それによる 2 つの訂正(2026-09-14/15)

第 1 回の起動は tamago で step 13 まで走り、§7 の role 列が埋まらない実装不具合(下記)のために停止した。
checkpoint は save_freq=25 に届いておらず、残っているのは dump 5 file と下の観測だけである。

**機構は設計通りに発火した**(step 3、コンソールは小数第 3 位丸め):

| 指標 | 実測 | 事前の記述との関係 |
|---|---:|---|
| `live_frac` | 1.000 | step 1 は cold start で 0(`row_available=False`)、step 2 から全位置 |
| `tv` | 0.068 | 旧 teachertopk アームの実測 1.42%、再設計見積り 3.9% より**大きい** |
| `abs_dkl_mean` | 0.233 nats | — |
| **`shrink/beyond_cand_frac`** | **0.226** | §6.1 の「> 0」を満たす。質量では 0.309。**注入は実在する** |
| `shrink/c_to_on_absmass` | 0.277 | $\lambda'=0.6$ が構成上決める dose |
| `shrink/off_to_on_absmass` | 0.746 | off-task の声は on-task shift の 3/4 の大きさ |
| `shrink/agree_cand_frac` | 0.702 | off-task 平均が on-task と同符号なのは 7 割 |
| `clamped_mass_frac` | 0.255 | **acted 量の 1/4 が ±5 に潰れている**。count 比(0.108)ではなくこちらを読むこと |
| `mass_error_max` | 0.000 | 構成通り |

**訂正 1: G1 は本モードでも反転する。** `shuffled_tv_ratio = 1.620` を実測した。off-task を位置方向に decorrelate すると
on-task shift との相関(監査実測 $\mathrm{corr}(\delta_{on}, s_{gen}) = +0.72$)が切れ、$c$ はその**差**に比例するので
$|c|$ は構造的に大きくなる。監査の spread から $\sqrt{138.3+25.4}/\sqrt{138.3+25.4-2(0.72)(11.76)(5.04)} = 1.45$ と見積もられ、
実測 1.62 と同じ向き・同程度である。初版が「shrink では G1 は元の意味を保つ」と書いたのは**誤りで、撤回する**。
1 に近い方が「2 つの shift が無相関」を意味する。

**訂正 2: 目標の entropy は下がる。** `entropy_delta = −0.039`。幾何混合は product of experts で連言的なので、
目標は on-task 教師より鋭くなる。初版の「目標が base 寄りになるので entropy は上がる」は理由づけから誤りで、撤回する。

**実装不具合(修正済み)。** `dp_actor.py` の `xtt_stats.update(roles=...)` が `xtt_mode == "curriculum"` に gate されていたため、
shrink では `roles=None` が渡り、`target/shrink/role/{structural,content}_share` が両方**きっかり 0.000** になっていた。
`_SUMS` に列があるのでゼロとして描画され、`TargetStepStats` 自身の注記が警告する「構造的にゼロの列が測定値として読める」状態だった。
gate を `in ("curriculum", "shrink")` に広げ、`tests/trainer/test_cross_teacher_shrink_arm.py` に呼び出し側の gate を読む回帰テストを追加した。

**所要時間の実測**: 起動 13 分 + 約 12.3 分/step。150 step は**約 31 時間**で、設計時の 27 時間見積りより長い。

## 6.5 3 回目の死と、rollout knob の切り替え(2026-09-15)

supervisor 下で 2 回落ちた(attempt 1: step 34、attempt 2: step 34。どちらも checkpoint 30 から復旧)。
flight recorder(§6.4 で有効化)が捕まえた内容:

* 両 rank とも `nccl:_all_gather_base`(input 25,168,000)が **20 個 `scheduled` のまま開始されない**。rank 1 が rank 0 より 4 collective 先行。
* 位置は step 34 の指標行の直後、step 35 の rollout フェーズ。最後のマイクロバッチ活動(19:04:42)から 35 分後に watchdog。
* OOM ではない(初報の "oom 103 件" は `dim_room` の部分一致で誤り)。

**同じ署名がこのホストで 4 回**: `opd_coef_redistribute` step 16(09-07)、本アーム step 19 / 34 / 34。全部 2 GPU、
`ROLLOUT_PUMP_TRAINING=1 ROLLOUT_PREFETCH_LOGPROB=1 ROLLOUT_PREFETCH_TEACHER=1`。fuji(3 GPU)には一度も無い。
alfworld-only の supervisor ログは同じ死を **`_prefetch_pending_log_probs` の中**と特定し、`PUMP=0 LOGPROB=0 TEACHER=1`
で 300 step 完走している(TEACHER=0 は post-rollout の一括 teacher 呼び出しで OOM)。

**判断**: step 41(checkpoint 40)で attempt 3 を止め、launcher に上の 3 つの export を入れて supervisor に再開させた。
損失は 1 step + 起動 13 分。放置した場合の期待損失は残り 110 step で約 5 回 × 2 時間。pump が買っていた時間
(logprob prefetch ≈ 44 s/step、pump 本体は未計測)は失う。**intent lock には触れていない**(性能経路であって科学的 knob ではない)。
同じ数学が rollout の後に走るだけで、目標・損失・データ順は同一。

**記録上の含意**: step 41 以降の学習曲線は knob が違う。速度の比較(s/step)は 41 を境に分けて読むこと。精度への影響は無いはずだが、
「はず」であって測定ではない — 41 前後で `episode/*` の窓平均が跳ねないことを確認項目にする。
