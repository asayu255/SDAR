# 教師間符号一致による蒸留重み付け — 実装仕様

対象ブランチ: `claude/opd-grpo-signweight-multitask`(`claude/opd-grpo-multitask` から分岐)

`docs/multitask_mechanism_v2_proposal.md` の機構W′/K′ を、既存の OPD+GRPO アームに
最小の変更で載せた簡易版の仕様である。連続量(κ, cos)を使う v2 提案に対し、
本実装は**符号の一致/不一致だけ**を見る。

---

## 1. 何をしているか

3教師はいずれも同一のベース方策 π_0 を単一タスクRLで微調整したものなので、
候補トークン v についての **policy shift**

```
δ_m(v) = log π_m(v|s) − log π_0(v|s)
```

の符号が「タスクmのRLがこのトークンを上げたか下げたか」を表す。
**符号だけを使うのは意図的**で、大きさは各教師の KL 係数と学習ステップ数を
引きずっている(search 教師は `kl_loss_coef=0.001`、他2つは 0.01 で10倍差)ため
共通の土俵に乗らないが、「上げたか下げたか」は乗る。

処理中のタスクの教師を **on-task 教師**、残り2つを **off-task 教師**と呼ぶ。

| on-task | off-task(2教師とも一致) | 判断 | 重み |
|---|---|---|---|
| 正 | 正 | 全ドメイン共通の知識 | **1.25** |
| 負 | 負 | 同上(**抑制の一致も一致**) | **1.25** |
| 正 | 負 | タスク固有 | **0.75** |
| 負 | 正 | 同上 | **0.75** |
| いずれかが沈黙、または off-task が割れる | | 判断材料なし | **1.00** |

重みは対称(1 ± 0.25)。SDAR の非対称性(正の支持は積極的に、負の拒絶は慎重に)は
**導入していない**。元の OPD+GRPO との比較を単純に保つためで、非対称にすると
平均重みがどの状態が多いかに依存し、効果と切り分けるべきものが1つ増える。

**off-task は全員一致のときだけ発言できる。**一方が上げ他方が下げる、あるいは
どちらかがデッドゾーンに入っている場合は「他タスクはこのトークンについて
意見が割れている」であって、どちら向きの共有構造の証拠でもないため重み付けしない。

---

## 2. 2つのモード

`algorithm.opd.sign_weight.mode` で選ぶ。**性質が根本的に異なる。**

### `position` — 位置ごとに1つの重み(既定)

20個の候補の重みを on-task 教師の確率質量で加重平均し、位置ごとの1スカラーにして
per-token KL 全体に掛ける。

```
w_t = Σ_v p_i(v)·w(v) + (1 − Σ_v p_i(v))·1.0        ← 末尾(top-k外)は中立
```

重みは生徒に依存しない正のスカラーなので、**損失の最小解は on-task 教師のまま動かない。**
学習の配分だけが変わる。安全である代わりに、**原理的に知識は転移しない**
(運べるのはサンプル効率だけ)。

### `target` — 目標分布そのものを書き換える

on-task 教師の top-k 分布を候補ごとに重み付けして再正規化し、その分布に蒸留する。

```
p̃(v) ∝ w(v)·p_i(v)        (末尾は重み 1.0 のまま = スケールの錨)
```

**固定点が `p̃` に移る。**両教師が推すトークンは、on-task 教師が置いた以上の
確率を持つ。これが実際に何かを注入できる唯一のモードであり、
同時に誤ったものを注入しうる唯一のモードでもある。

### なぜ「KLの各項に重みを掛ける」ではないのか

蒸留損失 `topk_kl_per_token` は **reverse KL** である。

```
D = Σ_v p_student(v)·(log p_student(v) − log p_teacher(v))
```

候補 v の項は**生徒自身の質量が払うコスト**なので、そこに大きな重みを掛けると
生徒はその候補から確率を逃がす。「両教師が推すトークンを強く学べ」の**真逆**になる。

素直な実装がなぜ罠かは数値でも確認できる(`tests/trainer/test_sign_weights.py::
test_multiplying_the_kl_terms_would_move_the_student_the_wrong_way`):
教師が (0.5, 0.5) のとき、候補1に 1.25・候補2に 0.75 を掛けて最小化すると
生徒は候補1を **0.38 まで下げる**。

目標分布を重み付けする形なら向きが正しく、かつ損失が真のダイバージェンス
(非負・目標でのみ0)のまま保たれる。

---

## 3. デッドゾーンと正規化

**デッドゾーン ε = 0.1 nats。**|δ| がこれ未満の候補は「そのタスクのRLは
このトークンに触っていない」とみなす。0.1 はトークン確率の約10%の変化に相当する。
これは細かい調整ではなく**機構の成否を分ける設定**で、無いと
「RLが動かさなかった大量のトークンのドリフトノイズ」が確信ある ±1 に化け、
独立な2教師がその半分で偶然「一致」して、コイン投げで損失が重み付けされる。

**タスクごとに平均1へ正規化(`position` のみ)。**しないと、一致が多いバッチでは
平均重みが1を超え、「単に蒸留を強くした」ことと区別できなくなる
(= `teacher_kl_loss_coef` を上げただけ)。タスクごとなのは、このアームが
`normalize_loss_by_task=true` で各タスク1/3の取り分を保証しているため。

正規化の帰結として、**`position` モードはタスク内のトークン間の差にしか反応しない。**
あるタスクの全トークンが一様に一致していても何も起きない。これは設計どおりだが
知っておくべき性質で、テストで明示的に固定してある。

`target` モードは末尾の重み1.0がスケールの錨になるため正規化しない。

---

## 4. 計算コスト

| | on-task top-20 KL | off-task 符号 | base 符号 | 合計 |
|---|---|---|---|---|
| 元の OPD+GRPO | 1 | — | — | 1.0× |
| 本アーム | 1(再利用) | 2 | 1 | **4.0×** |

各教師は**自分のタスクでない行だけ**(バッチの2/3)を採点する。自タスク行は
`compute_teacher_log_probs` が既に採点済みで、再計算は同じ forward の二度手間になる。
base は全行。教師と同じ `role="ref"` で作られるため CPU オフロードされ、
GPU ピークは変わらない(4モデルは逐次実行)。

---

## 5. 実装の所在

| ファイル | 役割 |
|---|---|
| `verl/trainer/ppo/sign_weights.py` | 符号判定・重み・正規化・目標書き換え・メトリクス(純関数) |
| `verl/trainer/ppo/opd_ray_trainer.py` | base worker の生成、`compute_sign_weights`(3モデルの forward とプレーン組み立て) |
| `verl/trainer/ppo/opd_grpo_ray_trainer.py` | `fit()` からの呼び出し(`_attach_task_ids` の後 = `task_ids` が揃ってから) |
| `verl/workers/fsdp_workers.py` | `compute_ref_topk_log_prob_at_ids`(外部指定 ids での評価入口) |
| `verl/workers/actor/dp_actor.py` | `compute_topk_log_prob_at_ids`、および `position` モードの per-token 乗算 |
| `examples/opd_grpo_trainer/run_multitask_signweight_qwen3.sh` | 実行スクリプト |
| `examples/opd_grpo_trainer/expected_multitask_signweight_config.yaml` | 意図ロック |

**全モデルは on-task 教師の top-20 ids 上で評価される。**信号は複数モデルの差なので
共通のサポートが要る。各モデルが自分の top-k を返すと、もっともらしい数値のまま
符号が無意味になる。`dp_actor.py` は既に指定 ids での full-vocab log-softmax gather を
持っていたので、worker API に入口を1つ足しただけで済んでいる。

---

## 6. 退化点(元アームが対照として成立する条件)

- `enable=false` → base worker を作らず、追加 forward も走らず、バッチに何も書かない
- `agree_weight = disagree_weight = 1.0` → 両モードとも恒等変換
- デッドゾーンが全シフトを飲み込む → 重みが全て1.0

いずれもテストで固定してある。加えて
`tests/trainer/test_run_script_matches_lock.py::test_the_two_arms_differ_only_in_the_sign_weighting`
が、2つの実行スクリプトの実効設定を差分して
**`algorithm.opd.sign_weight.*` と run identity 以外が一致すること**を検査する。
片方のスクリプトだけを調整した瞬間に落ちる。

---

## 7. 走らせたら最初に読むメトリクス

| メトリクス | 見るべきこと |
|---|---|
| `sign_weight/frac_neutral_*` | **これが9割超なら機構は何もしていない。**デッドゾーンを下げる |
| `sign_weight/frac_agree_*` vs `frac_conflict_*` | 一致と対立の比。タスク対ごとの構造の証拠 |
| `sign_weight/w_std` | 0 に近ければ、正規化後に配分が動いていない |
| `actor/pg_loss_weighted` vs `actor/teacher_kl_loss_weighted` | **KL項が勾配のごく一部なら、その内部で配分を変えてもできることは限られる。**比が極端に小さければ η か `teacher_kl_loss_coef` を見直す判断材料 |

最後の項目は expected_config 自身が「step 1 でこの比を読め」と注記している箇所と同じで、
平均1正規化をする以上、この機構が動かせるのは KL 項の内部だけであることに由来する。

---

## 8. 実行

```bash
bash examples/opd_grpo_trainer/run_multitask_signweight_qwen3.sh
```

`target` モードで走らせるには、スクリプトとロックの
`algorithm.opd.sign_weight.mode` を**同一コミットで** `target` に変更する
(スクリプトだけ直すと `verl/utils/expected_config.py` が起動時に弾く)。
`position` と `target` は別の科学的主張であって調整ノブではないため、
実験名も分けること。
