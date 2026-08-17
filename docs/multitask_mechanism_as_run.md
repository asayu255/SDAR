# 実際に走らせた機構の詳細 — target モード符号重み付け

原稿の Method 節を書くための一次資料。**実験に使われたのは `mode=target` である**
(`examples/opd_grpo_trainer/run_multitask_signweight_target_qwen3.sh`)。
`mode=position` は実装済みだが本実験では走らせていない。
この2つは**数学的に全く違う操作**なので、原稿で取り違えると Method 節が実験と一致しなくなる。

参照実装: `verl/trainer/ppo/sign_weights.py`,
`verl/trainer/ppo/opd_ray_trainer.py:461-574`,
`verl/workers/actor/dp_actor.py:1148-1198`,
`verl/trainer/ppo/core_algos.py:753-790`

---

## 0. 一段落での要約

3タスク同時学習の OPD+GRPO では、各サンプルは**自タスクの教師とだけ** reverse KL を取る。
他タスクの教師は同じバッチにいるのに一切参照されない。本機構は、
**on-task 教師が挙げた上位20候補のそれぞれについて、off-task 教師2本が
「その候補を上げたか下げたか」を全会一致で表明したとき**、その候補の目標確率を
1.25倍(一致)/0.75倍(対立)して再正規化し、生徒をその**書き換えた分布**へ蒸留する。
重みは凍結モデルだけの関数なので生徒に依存せず、backward では定数である。

---

## 1. ベースライン(何に対する差分か)

`run_multitask_qwen3.sh`(= `claude/opd-grpo-multitask` アーム)の損失は

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{GRPO}}}_{\text{方策勾配}} \;+\; \beta \sum_{t} \mathrm{KL}_{\text{top-}k}\big(\pi_\theta(\cdot|s_t) \,\|\, \pi_{T(i)}(\cdot|s_t)\big)$$

- $\beta = $ `algorithm.opd.kl_loss_coef = 0.01`
- $T(i)$ = サンプル $i$ のタスクの教師(**per-task routing**。これが転移を遮断している箇所)
- 教師3本 = `alfworld_step300` / `search_step300` / `webshop_step300`、いずれも
  同一ベース `Qwen/Qwen3-1.7B` からの単一タスク RL fine-tune
- `normalize_loss_by_task=True` — 各タスクが損失のちょうど 1/3 を持つ

top-$k$ reverse KL の定義(`core_algos.py:753`、$k=20$):

$$\mathrm{KL}_{\text{top-}k} = \sum_{v \in S} p_s(v)\big(\log p_s(v) - \log p_t(v)\big)
\;+\; \tau_s\big(\log \tau_s - \log \tau_t\big)$$

$S$ = 教師の top-20 id、$\tau = 1 - \sum_{v\in S} p(v)$ は **tail(top-k 外の全質量を1つの
バケットに束ねたもの)**。生徒・教師とも **full-vocab の log-softmax を $S$ で gather した値**で、
$S$ 内で再正規化はしていない(`dp_actor.py:431` の `logits.gather(-1, topk_ids) - lse`)。

> ⚠️ Revisiting OPD (2603.25562) の推奨推定量は $S$ 内再正規化なので、**本実装は別の推定量**である。
> 原稿に明記が必要(`docs/multitask_related_work_detail.md` §2.2)。

---

## 2. 転移の媒体 — なぜ「符号」なのか

3教師は**同一ベース $\pi_0$ からの独立な RL fine-tune** である。したがって
「タスク $m$ の RL がこの状態でモデルに書き込んだもの」は policy shift

$$\delta_m(v) \;=\; \log \pi_m(v\mid s) - \log \pi_0(v\mid s)$$

に完全に含まれる。$\delta_m > 0$ は「タスク $m$ の RL がこのトークンを**上げた**」、
$\delta_m < 0$ は「**下げた**」、$|\delta_m| \approx 0$ は「**触っていない**」。

**大きさではなく符号を使う理由は交絡である。**$|\delta_m|$ には各教師の
KL 係数と訓練ステップ数が乗っている — 実際 search 教師は `kl_loss_coef=0.001`、
他の2本は `0.01` で **10倍差**がある。したがって $|\delta_{\text{search}}|$ と
$|\delta_{\text{alfworld}}|$ は共通の物差しに載っていない。
一方「上げたか下げたか」は係数に依らない。

これが本機構の設計上の中心的な判断である。

---

## 3. 符号の抽出 — デッドゾーン $\epsilon$

$$\mathrm{sgn}_\epsilon(\delta) = \begin{cases}
+1 & \delta > \epsilon \\
-1 & \delta < -\epsilon \\
0 & |\delta| \le \epsilon
\end{cases} \qquad \epsilon = 0.1 \text{ nats}$$

**デッドゾーンは細部ではなく、これが無いと機構が壊れる。**
候補トークンの大半は教師の RL が一度も動かしていないもので、その $\delta$ はゼロ近傍の
ドリフトノイズである。素の `sign()` はそのノイズを自信満々の $\pm1$ に変え、
独立な2教師はそれに半分の確率で「一致」し、**コイン投げで損失が重み付けされる**。

$\epsilon = 0.1$ nats は「トークンの確率が約10%未満しか変わっていないものは無視する」に相当する
($e^{0.1} \approx 1.105$)。事前実験なしで設定した値。

> 原稿で $\delta$ を確率差として定義しているなら誤り。**log 差**であり、
> $\epsilon$ の単位は nats である。

---

## 4. off-task 合意 — 全会一致要求

off-task 教師は $n_{\text{off}} = 2$ 本(3タスク中、自タスクを除く)。
**合意符号は「全員がデッドゾーンの外にいて、かつ全員の符号が一致する」ときだけ定義する。**

実装(`sign_weights.py:170-172`)は和の絶対値で判定する:

```python
off_sum = sign_off.sum(dim=-1)          # 各要素は {-1, 0, +1}
unanimous = off_sum.abs() == n_off      # ±n_off に届くのは全員非ゼロかつ同符号のときだけ
consensus = torch.where(unanimous, torch.sign(off_sum), 0)
```

割れた場合(片方が上げ片方が下げ / どちらかが沈黙)は**重み付けしない**。
根拠は「他タスクが意見を異にする」ことは、どちらの方向にも共有構造の証拠にならないから。

---

## 5. 候補ごとの重み

on-task 教師の符号 $\mathrm{sgn}_\epsilon(\delta_{i})$ と off-task 合意 $c$ の組で決まる:

| on-task | off-task 合意 | 状態 | 重み $w(v)$ |
|---|---|---|---|
| $+$ | $+$ | `agree_pos` | **1.25** |
| $-$ | $-$ | `agree_neg` | **1.25** |
| $+$ | $-$ | `conflict_on_pos` | **0.75** |
| $-$ | $+$ | `conflict_on_neg` | **0.75** |
| $0$(デッドゾーン内) | — | `neutral_on_task_silent` | 1.0 |
| $\pm$ | 割れ / 沈黙 | `neutral_off_task_split` | 1.0 |

**重要: 中立の2状態が実測では多数派である。**step1 時点で
`neutral_off_task_split` が **0.561**、`neutral_on_task_silent` が 0.040 —
**候補の約60%は重み1.0のまま**である。4パターンの表だけを原稿に載せると、
機構が常時全候補に作用しているような印象を与えるので**網羅的でない**。

実測の推移(`docs/multitask_sign_weighting_results_150.md`):

| step | 1 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| `agree_pos` | 0.261 | 0.218 | 0.181 | 0.170 | 0.187 |
| `conflict` 合計 | 0.108 | 0.158 | 0.156 | 0.115 | 0.121 |
| `neutral_off_split` | 0.561 | 0.537 | 0.571 | 0.626 | 0.605 |

一致は前半で痩せるが step150 以降は下げ止まり、**機構は最後まで動作している**
(帰無結果が「機構が止まっていたから」ではないことの確認)。

**対称性について**: 1.25 と 0.75 は $1 \pm 0.25$ で対称に選んだ。SDAR のような
非対称性(増幅と減衰で強度を変える)は導入していない。ベースラインとの比較を
単純に保つため。

---

## 6. target モード — 実際に走った操作

### 6.1 定義

$$\tilde p(v) \;=\; \frac{w(v)\,p_t(v)}{Z}, \qquad
Z \;=\; \sum_{v \in S} w(v)\,p_t(v) \;+\; \underbrace{\Big(1 - \sum_{v\in S} p_t(v)\Big)}_{\text{tail、重み 1.0 固定}}$$

log 空間の実装(`sign_weights.py:268-299`):

```python
p    = on_task_logprob.exp()                                  # 教師の top-k 確率
tail = (1.0 - p.sum(-1, keepdim=True)).clamp(0.0, 1.0)
z    = (p * candidate_weight).sum(-1, keepdim=True) + tail
return on_task_logprob + log(candidate_weight) - log(z)
```

出力は入力と同じ「full-vocab log-softmax を $S$ で gather した形」($\exp(\cdot).\text{sum} \le 1$)なので、
**`teacher_topk_logprobs` をそのまま差し替えるだけでよく、actor 側は一切変更が要らない。**
損失は正規の reverse KL のまま(非負・目標一致でのみゼロ)で、
**目標だけが意図的に別のものに置き換わる。**

### 6.2 tail が「スケールの錨」である

$w$ を全候補一律に $c$ 倍しても、$Z$ の第1項だけが $c$ 倍され tail は変わらないので、
**$\tilde p$ は不変にならない**。つまり tail が $w$ の絶対スケールを固定している。
中立重みを tail と同じ 1.0 に取っているのはこのためで、その結果:

$$w \equiv 1 \;\Longrightarrow\; Z = \sum_{v\in S} p_t(v) + \tau_t = 1 \;\Longrightarrow\; \tilde p = p_t \quad(\textbf{厳密な恒等写像})$$

数値確認済み: $\max|\text{out} - \text{in}| = 0.000\mathrm{e}{+}00$。
テストで固定: `test_target_shift_is_exactly_zero_when_nothing_is_flagged`。

**この性質が「再正規化そのものが効いたのでは」という交絡を潰す。**
重みが一様なら再正規化は何もしない以上、独立した介入たりえない。
(Revisiting OPD 方式の「$S$ 内で再正規化して推定量を変える」には**なっていない**。)

### 6.3 平均1正規化の扱い — position と target で違う

- **position モード**: `normalize_per_task` でタスクごとに masked mean を 1.0 に揃える。
  これが無いと「一致が多いタスクは実質 $\beta$ が大きい」ことになり、
  機構ではなく `teacher_kl_loss_coef` の変更と区別がつかなくなる。
- **target モード(実走)**: `normalize_per_task` は**呼ばれない**。
  $Z$ による再正規化と tail アンカーがその役割を果たすため。
  代わりに「系統的に鋭く/平坦にしていないか」を `target_entropy_delta` で監視する。

実測(step151 の状態比率に基づくシミュレーション):
エントロピー変化 **−0.003〜−0.005 nats**、**等価温度 $T \approx 0.998$**。
温度換算で 0.2% であり、AlfWorld の +0.111 を説明できる量ではない。
目標移動量は **0.004〜0.008 nats** で、`actor/teacher_kl_loss = 0.222` に対して **2〜4%**。

> ⚠️ これはシミュレーション推定。実測用のメトリクス
> `sign_weight/target_kl`(= $\mathrm{KL}(p_t \| \tilde p)$、`actor/teacher_kl_loss` と同単位)と
> `sign_weight/target_entropy_delta` は実装済みなので、次回走行で実測値に置き換えること。

### 6.4 不動点がどこへ動くか

reverse KL $\mathrm{KL}(\pi_\theta \| \tilde p)$ の最小解は $\pi_\theta = \tilde p \propto w(v)p_t(v)$。
つまり**両教師が推すトークンは、on-task 教師が単独で割り当てたより多くの質量を得る**。
これが「実際に何かを注入できる」変種であり、同時に**間違ったものを注入しうる**変種でもある。

---

## 7. 【重要】KL の各項に掛けてはいけない理由

原稿が「本来の学習信号に増強時1.25倍」と書いているなら、それは**別の操作**であり、
**意図と逆向きに効く**。

蒸留損失は reverse KL $\sum_v p_s(v)(\log p_s(v) - \log p_t(v))$ で、
候補 $v$ の項は**生徒自身が $v$ に置いた質量が払うコスト**である。
そのコストを 1.25 倍すれば、生徒は $v$ から質量を**逃がす**。
「両教師が推すトークンだからもっと学べ」の正反対になる。

数値確認: 教師 $(0.5, 0.5)$ に $w = (1.25, 0.75)$ を項ごとに掛けると、
勾配降下で生徒は $q_1 = 0.381$ に**収束する**(1.25 を掛けた側が下がる)。
テストで固定: `test_multiplying_the_kl_terms_would_move_the_student_the_wrong_way`。

加えて、項ごとの積は**正規のダイバージェンスではなくなる**(目標一致でもゼロにならず、
負にもなりうる)。目標分布を書き換える方式はこの性質を保つ。

---

## 8. position モード(実装済み・未実行)との対比

$$w_{\text{pos}}(t) \;=\; \sum_{v\in S} p_t(v)\,w(v) \;+\; \tau_t
\;=\; \mathbb{E}_{p_t}[w] \quad(\text{tail は重み 1.0})$$

これを per-token KL に掛ける(`dp_actor.py:1176`)。

$w_{\text{pos}}$ は**生徒に依存しない正の定数**なので、重み付き損失の最小解は依然として
$p_t$ である — **目標は動かない**。したがって:

| | position | target(実走) |
|---|---|---|
| 不動点 | $p_t$(不変) | $\tilde p \propto w\,p_t$(**移動**) |
| 何を変える | どの位置をどれだけ強く学ぶか | 何になれと言われるか |
| 知識転移 | **できない**(サンプル効率のみ) | **できる**(誤ったものも入りうる) |
| 安全性 | 構成上安全 | 検証が必要 |
| 平均1正規化 | 明示的に実施 | $Z$ と tail が代行 |

候補平均を教師確率で取っているのは、19個の見捨てられたトークンが
1個の確信あるトークンを覆さないようにするため。

---

## 9. 計算経路と追加コスト

訓練ループ内の順序(`opd_grpo_ray_trainer.py:305-325`):

```
rollout → compute_teacher_log_probs → _attach_task_ids → compute_sign_weights → update_actor
```

`compute_sign_weights`(`opd_ray_trainer.py:476`)がやること:

1. **base 方策**(`Qwen/Qwen3-1.7B`、`sign_weight.base_path`)を**全行**について、
   on-task 教師の top-20 id の上でスコアリング
2. **各教師**を、**自タスク以外の行**についてスコアリング(各教師がバッチの 2/3)
3. 符号を取り、全会一致を判定し、重みを作る
4. target モードなら `batch.batch["teacher_topk_logprobs"]` を書き換える

**追加コスト = 凍結モデルの forward 3回分**(base 全行 + 教師2本分相当)。
on-task 教師の pass は `compute_teacher_log_probs` で既に済んでいるので再利用する。

実装上の細工が1つ: 教師スコアの3次元スタックを**ゼロではなく base の値で seed** している
(`opd_ray_trainer.py:520`)。未書き込みセルは「その教師自身のタスクの行」で、
`plane_idx` が除外するので本来読まれない。仮にその除外が誤っていても、
ゼロは $\delta = -\log \pi_0$ という**大きく確信に満ちた架空の意見**として読まれるのに対し、
base 値なら $\delta = 0$ =「この教師は何も動かしていない」と読まれ、重み付けが自分で切れる。

新規のワーカー配線:
- `fsdp_workers.py`: `compute_ref_topk_log_prob_at_ids`(`Dispatch.DP_COMPUTE_PROTO`)
- `dp_actor.py`: `compute_topk_log_prob_at_ids` — **外部から与えられた id** で
  log-prob を取る(既存の `compute_ref_topk_log_prob` は自分で top-k を選ぶので使えない)
- `role="ref"` の base worker を1本追加

---

## 10. 変更していないもの(交絡の監査結果)

`enable=false` は**ビット等価**である(`SIGN_WEIGHT_KEY` が無ければ actor は素通り、
target モードなら `teacher_topk_logprobs` が書き換わらない)。

対照アーム(`run_multitask_qwen3.sh`)との config 差分は **11キー**で、内訳は:
- 機構そのもの 6キー(`sign_weight.*`)
- 同一性キー 4キー(`experiment_name`, `default_local_dir` ほか)
- `actor_rollout_ref.ref.fsdp_config.sharding_strategy` — **配置のみ**で学習信号に影響しない

損失・データ・シード・教師・$\beta$・top-$k$・タスク正規化はすべて同一。
`data.seed=1` なので、**step $k$ では両アームが同じプロンプトを見ている**
(統計をペア検定でやるべき理由)。

> `default_local_dir` はアームごとに分けてある。共有したままだと `resume_mode: auto` で
> **後発の対照ランが処理ランのチェックポイントから再開する**事故が起きる。
> テスト `test_run_script_matches_lock.py` で異なることを要求している。

---

## 11. 実際に走った設定(そのまま原稿に書ける)

| 項目 | 値 |
|---|---|
| 生徒 | Qwen3-1.7B |
| ベース $\pi_0$ | Qwen/Qwen3-1.7B(教師と同一の出発点) |
| 教師 | alfworld / search / webshop の単一タスク RL step300 |
| タスク | ALFWorld / WebShop / Search-QA 同時、15 prompt/task × 8 rollout |
| 損失 | GRPO + $\beta\,\mathrm{KL}_{\text{top-20}}(\pi_\theta \| \tilde p_{T(i)})$ |
| $\beta$ | 0.01 |
| top-$k$ | 20(on-task 教師の上位20候補) |
| モード | **target** |
| $w_{\text{agree}}$ / $w_{\text{conflict}}$ | **1.25 / 0.75** |
| デッドゾーン $\epsilon$ | **0.1 nats** |
| off-task 合意 | **全会一致のみ**(2本とも同符号かつ非沈黙) |
| タスク正規化 | `normalize_loss_by_task=True`(各タスク 1/3) |
| GPU | 2 |

---

## 12. 原稿に必ず書くべき limitation 2点

1. **候補集合が on-task 教師に支配されている。**off-task 教師は
   「on-task 教師が top-20 に入れたトークンについてしか」発言できない。
   off-task 教師だけが強く推すトークンは構造的に届かない。
   2607.07050 は「top-32 が確率質量 99.99% を保っても決定トークンの 0.4% しか含まない」
   事例を報告しており、この制約は実害がありうる。仕様であって不具合ではないが明記が必要。

2. **符号は共有ベースからの相対量である。**3教師が同一 $\pi_0$ の fine-tune であることに
   依存しており、異なるベースの教師には**そのままでは適用できない**。
   本機構の適用条件として明記すること。
