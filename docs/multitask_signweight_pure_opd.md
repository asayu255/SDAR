# 符号一致重み付け(純OPD・生徒インデックス) — 実装と読み方

対象アーム: `examples/opd_trainer/run_multitask_{qwen3,signweight_position_qwen3,signweight_target_qwen3}.sh`
参照実装: `verl/trainer/ppo/sign_weights.py`, `verl/trainer/ppo/opd_ray_trainer.py`,
`verl/workers/actor/dp_actor.py`, `verl/workers/teacher_cache.py`

---

## 0. 一段落での要約

3タスク同時の純OPDでは、各サンプルは**自タスクの教師とだけ** reverse KL を取る。他タスクの
教師は同じバッチにいるのに一切参照されない。本機構は、**生徒が選んだ上位20候補**のそれぞれに
ついて、base 方策からの policy shift の**符号**を4モデル(自タスク教師・off-task 教師2本・base)
で読み、off-task 2本が全会一致でかつ自タスク教師と一致した候補に重みを付ける。
`pg_loss_coef=0` なので蒸留項が損失の全部であり、重みが動かすものは損失の全部である。

---

## 1. 支持集合は生徒の top-20

$$S = \operatorname{Top20}(p_s), \qquad
p_m(v) = \exp\big(h_m \cdot W_m[v] - \mathrm{lse}_m\big),\ v \in S$$

4モデルとも **full-vocab log-softmax を $S$ で gather した値**で、$S$ 内で再正規化はしない。
教師の値は `teacher_cache.py` にキャッシュした hidden state と lse から復元する — id に依存する
のは最後の gather だけなので、教師は生徒より先に走れる。

**この選択の帰結を2つ、原稿に書くこと。**

1. **重みは凍結モデルだけの関数ではなくなった。** backward では依然として定数だが、同じ状態でも
   生徒が漂えば候補集合が変わる。固定の注釈ではなくフィードバックループである。
2. **教師インデックス版の構造的制約が消えた。** off-task 教師が「自タスク教師が top-20 に入れた
   トークンについてしか発言できない」制約が無くなった。

---

## 2. 符号とデッドゾーン

$$\delta_m(v) = \log p_m(v) - \log p_0(v), \qquad
\mathrm{sgn}_\epsilon(\delta) = \begin{cases}
+1 & \delta > \epsilon \\ -1 & \delta < -\epsilon \\ 0 & |\delta| \le \epsilon
\end{cases}$$

$\epsilon = 0.1$ nats。大きさではなく符号を使うのは交絡のため — $|\delta_m|$ には各教師の
KL 係数(search は 0.001、他は 0.01 の**10倍差**)とステップ数が乗っており、共通の物差しに載って
いない。デッドゾーンが無いと、教師が一度も動かしていない候補のドリフトノイズが確信ある $\pm1$ に
なり、独立な2教師が半分の確率で「一致」する。

off-task 合意は**全会一致のときだけ**定義する(2本とも非沈黙かつ同符号)。割れた場合と沈黙した
場合は**別の状態として記録する** — 前者は「他タスクが実際に食い違う」、後者は「デッドゾーンが
証拠を飲み込んだ」で、診断が違う。

---

## 3. 重み表 — 2モードは表を共有しない

| on-task | off-task 合意 | position | target |
|---|---|---|---|
| ＋ | ＋ | **1.25** | **1.25** |
| − | − | **1.25** | **0.75** |
| ＋ | − | 1.0 | 1.0 |
| − | ＋ | 1.0 | 1.0 |
| 0 / 割れ / 沈黙 | — | 1.0 | 1.0 |

**position** は $w$ が per-token KL に掛かる。KL 項に向きは無いので、上げる方向で一致していようが
下げる方向で一致していようが「共有構造がある」という同じ意味になる。最小解は自タスク教師のままで、
**固定点は動かない**。

**target** は $w$ が**確率**に掛かる。したがって向きが内容そのものになる。両方の一致を増強すると、
全教師が一致して抑制したトークンに質量を足すことになり、一致が証拠であるはずの編集を打ち消す。

対立は**どちらのアームでも重み付けしない**(`disagree_weight=1.0`)。target モードは 1.0 以外を
**拒否する** — 異議を唱えた教師側へ引き戻すとは、自タスク教師が上げたトークンを下げ、下げた
トークンを**上げる**ことであり、1未満の単一係数は後者を逆向きにやる。

---

## 4. target の再正規化

$$\tilde p(v) = \frac{w(v)\,p_i(v)}{Z},\quad
Z = \sum_{v\in S} w(v)p_i(v) + \tau_i = 1 + \sum_{v\in S}\big(w(v)-1\big)p_i(v)$$

tail は重み 1.0 固定。これが $w$ の**絶対スケールの錨**であり、同時に

$$w \equiv 1 \;\Longrightarrow\; Z = 1 \;\Longrightarrow\; \tilde p = p_i \quad(\textbf{厳密な恒等写像})$$

を与える。**再正規化そのものが効いたのでは、という交絡はこれで潰れる** — 重みが一様なら何も
起きない以上、独立した介入たりえない。

$\tilde\tau_i = \tau_i / Z$ は返さない。$k{+}1$ カテゴリの KL が `1 - sum` から厳密にそれを復元
するため。**tail の重みが 1.0 でも tail の「割合」は変わる**ことに注意。

---

## 5. 【重要】$Z > 1$ は系統的である

$$Z - 1 = 0.25\!\!\sum_{v\in\text{agree\_pos}}\!\!p_i(v)\;-\;0.25\!\!\sum_{v\in\text{agree\_neg}}\!\!p_i(v)$$

**この2項は相殺しない。** agree_pos は教師が上げたトークンなので定義上 $p_i$ が高く、agree_neg は
下げたトークンなので低い。よって $Z>1$、$\tilde\tau_i < \tau_i$ となり、目標が**系統的に
シャープ化する**。これは「機構の名を借りた温度変更」という交絡そのもので、
`sign_weight/inv_z` と `sign_weight/target_entropy_delta` はこれを直接測るために入っている。

**符号シャッフル対照はこれを代替できない。** シャッフルは符号情報とシャープ化の両方を同時に
壊すので、シャープ化が効いていたならシャッフルでの利得消失は符号情報の証拠にならない。
`target_entropy_delta` がゼロ近傍であることが、シャッフル対照を符号情報の検定として読む前提。

---

## 6. position の平均1正規化

新しい表には 1.0 未満の値が無いので、**生の平均は必ず1を超える**。正規化しないと
「一致が多い＝実質 $\beta$ が大きい」となり、機構ではなく `teacher_kl_loss_coef` を上げただけの
アームと区別がつかない。

必要な平均はステップ全体のものだが、重みは forward の中(マイクロバッチ1個分)にしか存在しない。
そこで**前回呼び出しの per-task 平均**で割る。マイクロバッチ自身の平均で割るとバッチ分割の仕方が
目的関数を動かす。平均は遅い量(一致率は150stepで 0.26→0.17)なので前回値で十分であり、定数なので
勾配を歪めることはなく、実効係数を1%未満動かすだけ。ランク間で合算する — ランクごとに違う数で
割ると正規化ではなく目的関数の変更になる。**step1 は正規化されずに走る**。その分は
`sign_weight/*/w_mean_pre_norm` に出る。

---

## 7. 計算経路と追加コスト

```
rollout → compute_teacher_log_probs → _attach_task_ids
        → compute_sign_weight_cache → check_teacher_hidden_cache → update_actor
```

`compute_sign_weight_cache`(ドライバ)は base を**全行**、各教師を**自タスク以外の行**
(各2/3)について走らせ、hidden state をキャッシュする。**追加コストは凍結モデルの forward 3回分**。
自タスク教師のパスは `compute_teacher_log_probs` で済んでおり、同じキャッシュから読み直す。

書き出す列は2つ:

- `sign_cache_ids` (bs, 1+n_off) — 列0が base、列1..がその行の off-task 教師(タスク名のソート順)。
  **アクターは位置で読む**ので、列の意味は行のタスクだけの関数でなければならない。
- `sign_off_tasks` (bs, n_off) — 各列のタスク id。診断(タスク対ごとの一致率)専用。

アクターは生徒の top-k が確定した直後に4モデルを同じ id で解決し、重みを作り、モードに応じて
適用する。base は `role="ref"` のワーカーとして1本追加され、その lm_head は
**タスク名ではないラベル**(`__sign_base__`)で登録される — キャッシュは projection をこの文字列で
選び、ルーティングはタスク名で教師を選ぶので、衝突すれば黙って別モデルで採点される。

---

## 8. wandb 指標の読み方

### 機構が動いているか
| 指標 | 読み方 |
|---|---|
| `sign_weight/frac_<state>` (7状態) | 候補数の比率。`neutral_*` が支配的なら機構はほぼ何もしていない |
| `sign_weight/<task>/frac_<state>` | タスク別。「AlfWorld だけ効いた」を機構側の数字で裏付ける |
| `frac_neutral_off_task_split` vs `_silent` | 前者は他タスクが食い違う、後者は $\epsilon$ が証拠を飲んだ。**別物** |
| `sign_weight/mass_frac_<state>` | **確率質量**加重。目標を動かすのは $w\cdot p_i$ なので、実効的な大きさはこちら |
| `sign_weight/teacher_coverage` (+タスク別) | 生徒 top-20 上の教師質量 $\sum_{v\in S}p_i(v)$。**target の梃子の天井**。`mass_frac_*` はこの中の構成比なので、これ無しでは過大に読める |

### 介入の大きさ(target)
| 指標 | 読み方 |
|---|---|
| `sign_weight/target_kl` | $\mathrm{KL}(p_i \Vert \tilde p)$。`actor/teacher_kl_loss` と同単位 |
| `sign_weight/target_kl_ratio` | 上を `teacher_kl` で割った比。生徒の残り距離のうち書き換えが担う割合 |
| `sign_weight/inv_z` | $=\tilde\tau_i/\tau_i$。1未満 = tail が縮んで top-k が鋭くなっている(§5) |
| `sign_weight/target_entropy_delta` | 同じことを nats で。温度交絡の直接測定 |
| `sign_weight/target_tv` | 全変動距離。$[0,1]$ なのでステップ間・アーム間で比較しやすい |

### 転移の中身(訓練コストゼロで取れる転移可能性行列)
| 指標 | 読み方 |
|---|---|
| `sign_weight/agree_rate/<off>__on__<on>` | タスク対ごとの符号一致率。**3×3 行列がこれで埋まる** |
| `sign_weight/abs_delta_mean/<task>` | 教師ごとの $\overline{\lvert\delta_m\rvert}$。教師間で桁が違えば $\epsilon$ が片方にだけ効いている |
| `sign_weight/deadzone_frac/<task>` | 教師ごとの沈黙率。1本が9割沈黙なら全会一致要件は実質その1本が決めている |

### position 固有
| 指標 | 読み方 |
|---|---|
| `sign_weight/w_mean_pre_norm` (+タスク別) | 正規化**前**の平均。1.15 なら正規化が実効 $\beta$ の15%増を取り除いた |

### 資源
| 指標 | 読み方 |
|---|---|
| `teacher_cache/gb`, `/rows` | キャッシュが保持している量。**重み付けアームは行あたり4モデル**なので最初に見る数字 |
| `teacher_cache/witness_max_err` | エントリが別の行に紐付いていないかの検証。上がったら止める |

---

## 9. 検証のインスタンス単位ログ

`trainer.val_instance_log_dir` を設定すると、検証1回につき
`val_step<N>.jsonl` に1インスタンス1行を書く(`val_index` / `task` / `score` / `traj_uid`)。

**前回の解析が2回死んだ箇所である。** WebShop の連続スコアはインスタンス分散が保存されておらず
区間も z も後から計算できなかった。また2アームは**同じ126インスタンスを同じ順に**評価しており、
ペア検定(成功への McNemar、スコアへの対応あり t)が使えるのに、どれがどれか残っていなかった。

ペアリングのキーは `val_index` = 検証パス内での行位置。検証ローダは `shuffle=False` で固定
ファイルを読み、ファイル・件数・シードは intent lock で固定されているので、index $i$ は
全アーム・全ステップで同じ問題である。`traj_uid` は1ロールアウトの識別子であって**このキーでは
ない**。成功判定はファイルに焼き込んでいない — alfworld と search は score==1.0 だが webshop は
閾値であり、焼き込むとその推測の分だけログの価値が下がる。

---

## 10. 走行の組み方

| アーム | スクリプト | 役割 |
|---|---|---|
| 対照 | `run_multitask_qwen3.sh` | 純OPD |
| position | `run_multitask_signweight_position_qwen3.sh` | 固定点を動かさない = **target の対照** |
| target | `run_multitask_signweight_target_qwen3.sh` | 固定点を動かす |

走行は**150 → 評価 → 300 の順**（`run_signweight_sequence.sh`）。150 で止めるのは
`trainer.stop_after_steps` であって `total_training_steps` ではない — 後者は lock 固定で、
かつ warmup(総step の10%)を変えて LR 軌道が対照とずれる。前者はプロセスの寿命だけを決め、
再開後はクラッシュ復帰と同一の継続になる。

3本すべて同一コード・`data.seed=1` なので、step $k$ では同じプロンプトを見る(統計をペア検定で
やるべき理由)。position が target の対照になるのが本構成の要点:

- position ≈ target がともに改善 → 利得は再配分(サンプル効率)であって知識注入ではない
- target > position → 固定点移動が効いている

`student_indexed_topk=True` は3本とも同じでなければならない。サポートが違う run 同士は比較でき
ない(`ppo_trainer.yaml` の当該コメント)。**8/16 より前の純OPD 150step 値は教師インデックスで
出ているので、この3本の対照には使えない。**
