# 特権的な複数タスク通知（privileged multitask notice）— 設計と事前登録

状態: **設計のみ。実装なし。** 走らせる前に §5 の事前登録を固定すること。

前提文書: [cross_teacher_theory.md](cross_teacher_theory.md)（命題 1–3、階層モデル、識別不能性、原則 1/2）、
[cross_teacher_kl_weight_offline_audit.md](cross_teacher_kl_weight_offline_audit.md)（監査: 共有成分 ≈ 書式、
§13.4 の OPSD 評価）、[cross_teacher_curriculum_design.md](cross_teacher_curriculum_design.md)（直前のアームと
その ladder 測定）。

**機構**: 学習中のみ、system ロールで「1 つの重みが 3 環境を同時に学習していること」と「他タスクの語彙・知識を
持ち込まないこと」を明示した短い文書を見せる。教師側・生徒側のどちらにも適用でき、パラメータで切り替える。

---

## 0. 前提が偽であることは、走らせる前に測ってある

**この機構が除去しようとしている汚染は、存在しない。** 保存済みの検証生成文
（`~/val_instances/opd_grpo_multitask_cross_teacher_klw_content_qwen3_1.7b_sg1fast/val_step{150,300}.jsonl`、
@150 で 138,109 応答、@300 で 129,618 応答）に対する測定:

| task | 応答数 (@300) | 他タスクの**行動構文**を含む | 他タスク固有**語彙**を含む |
|---|---:|---:|---:|
| alfworld | 2,826 | **0** (0.00000) | **0** (0.0000) |
| webshop | 767 | **0** (0.00000) | **0** (0.0000) |
| search | 126,025 | **0** (0.00000) | 230 (0.0018) |

@150 でも同じく全タスク 0。行動構文は `<action>` / `<search>` / `<answer>` / `search[` / `click[` の 5 種を
曖昧さなく判定した。alfworld の 2,826 応答は**すべて** `<action>` のみを使い、`<search>` も `search[` も 1 件も
出さない。search の語彙側 0.18% は中身を確認すると `go to` が通常の英語表現として出ているだけで、alfworld 語彙
ではない。

**独立な第 2 の測定も同じ向きを指す。** 同じ日に取った off-task ladder（control@300、その方策自身の rollout 位置、
support カバレッジ 0.98+）は 6 ペアすべてで

$$\mathrm{off\_travel} = \frac{\mathrm{KL}(\pi_\theta\Vert\pi_{src}) - \mathrm{KL}(\pi_0\Vert\pi_{src})}{\mathrm{KL}(\pi_d\Vert\pi_{src}) - \mathrm{KL}(\pi_0\Vert\pi_{src})} \in [1.07,\ 2.11]$$

すなわち**生徒は自分の教師よりさらに他タスク教師から離れている**。汚染どころか過剰に分離している。

**したがって主要な事前予測は null である**（§5.2）。それを承知の上で走らせる、というのが 2026-09-05 の決定である。
本文書はその決定を記録し、null が出たときに「予測どおり」と読めるようにするために、走行前に書かれている。

**2026-09-05 追記（実装時）。** 操作者の読みは「この機構は特殊トークン（行動構文）の抑制を意図したものではない」であり、
上の測定は無視して進める、と決まった。測定自体は撤回しない — 行動構文の混入が 0 であることは事実である — が、
機構の狙いが行動構文以外の何か（語彙の選び方、reasoning の文体）にあるなら、上の 0 はその狙いの反証にはならない。
§4 の診断 1（文書の効果量）が、狙いが実際に方策を動かしたかを直接測る。

---

## 1. 決定事項（2026-09-05 確定）

| # | 論点 | 決定 |
|---|---|---|
| A | 誰に見せるか | **教師・生徒の両方に対応**し、`apply_to` で切り替える。**初回は生徒モード** |
| B | 「干渉するな」の行動的定義 | **他タスクの語彙・知識を輸入しない。現タスクが確立した語彙のみを使う** |
| C | 過去 rollout を含めるか | **含めない**（文書は静的） |
| C' | コンテキスト予算 | 教師モードでは生徒のプロンプトに入らないので無関係。生徒モードでは入るが、rollout の上限はグローバル 4096/left で余裕 1300+ なので**上限は変えない**（§3.2、実装時に訂正） |
| D | どの教師に足すか | **OPD 教師（$\pi_d$）のみ**。対照が同じ教師なので、差は文書だけに帰属する |
| E | 対照 | **プラセボ B のみ**（同じ指示から複数タスクの明示だけを抜いたもの）。プラセボ A（長さ対照）は置かない |
| F | 事前検定 | **行わない**。代わりに run 内診断を必須にする（§4） |
| G | 生徒モードの評価時 | **訓練時のみ**。評価は control と完全に同一（実装: `TrajectoryCollector._notice_active` を `multi_turn_loop(is_train=)` から設定。検証は専用の collector インスタンスで走るため、config だけでは訓練と区別できない） |
| H | $\beta$ | 0.01 のまま（比較可能性） |

---

## 2. 文書

### 2.1 本アーム（named）— ALFWorld 版

> One set of weights is being trained to act in three environments at once: ALFWorld (this one: locating and
> manipulating household objects), WebShop (finding and buying a specified product), and Search (answering a
> question with a retriever). You are acting in ALFWorld now.
>
> Because the same parameters serve all three, words and assumptions that belong to WebShop or Search can
> surface here, where they are simply wrong.
>
> Use only what this environment has established: the receptacles, objects and movements ALFWorld names, and
> the action forms it accepts. Nothing here has a price, a rating, a listing, a search result or a citation,
> and no question here is answered from a retrieved passage.
>
> Before you act, check that every term you used came from this room's observations or from ALFWorld's own
> action set. If a phrasing would fit WebShop or Search but not here, drop it.

**4 段の役割。** 第 1 段が特権情報そのもの（1 つの重みが 3 環境を同時に、いまはこれ）。第 2 段が汚染が起きる理由。
第 3 段が語彙の限定と、他タスク由来で使ってはならない具体物の列挙。第 4 段が行動前の自己点検で、これが
トークンレベルで実行可能な唯一の部分である。**勾配・パラメータ・干渉には一切言及しない** — 方策が作用できないため。

WebShop 版・Search 版は名指しと列挙を入れ替える。WebShop 版は「the product attributes, search terms, buttons and
page elements WebShop names」を確立語彙とし「Nothing here is a room, a receptacle or an object you can pick up,
heat or cool」を除外。Search 版は「the question, the passages the retriever returns, and the query and answer
forms」を確立語彙とし「Nothing here is a room, a receptacle or a movable object, and nothing here has a price,
a rating, a listing or a cart」を除外。

### 2.2 プラセボ B — 何を切り分けるか

プラセボは本アームから**第 1 段の「3 環境が同時」と他タスクの名指しだけを抜き**、避けるべき具体物の列挙は残す。
ALFWorld 版なら「One set of weights is being trained to act in ALFWorld: ...」で始まり、「Words and assumptions
that do not belong to this environment can surface here」と続く。第 3・4 段は名指し部分を除いて同一。

**両アームは「何を避けるか」を同じ具体度で指示する。違うのは「なぜか」を知らせるかどうかだけである。**
したがってこの対照は「良い禁止リストを書いた効果」を除去し、**複数タスクであることを周知する効果だけ**を残す。
差が出なければ、機構はプロンプト工学であって MOPD の主張にならない。これが 2 アーム構成で最も情報量の多い切り方である。

### 2.3 トークン数とハッシュ（intent lock に固定する）

| variant / task | tokens (system msg) | sha256[:12] |
|---|---:|---|
| named / alfworld | 193 | `46f6f162d9fb` |
| named / search | 196 | `35500e7a3bec` |
| named / webshop | 199 | `c58107a11ff3` |
| placebo / alfworld | 149 | `53b879e2adb6` |
| placebo / search | 149 | `a382c97fdebf` |
| placebo / webshop | 152 | `a4774a75f179` |

（初版にあった「引き上げ後の上限」列は §3.2 の訂正により削除。）

**文書は設定の細部ではなく機構そのものなので、本文のハッシュを lock に固定する。** run ごとに書き換えられては
比較が壊れる。

---

## 3. 配線

### 3.1 挿入点

**system ブロックは chat template 上で厳密な接頭辞である**（実測: `template([system, user]) == "<|im_start|>system\n{text}<|im_end|>\n" + template([user])`、
文字列でもトークン列でも成立、8 トークン + 本文）。したがって生徒モードの system メッセージと教師モードのトークン前置は
**同一のトークン列**を生み、1 つの実装で両モードを扱える。起動時にこの性質を assert する。

**生徒モードでは教師も文書を見る。** on-task 教師は生徒の (prompt, response) をそのまま採点するので、生徒のプロンプトに
入った文書は教師の文脈にも入る。これは「共有文脈」の意味論であり、KL は同じ文脈上の 2 分布の間で取られる。
base と off-task 教師（ladder 用）は文書を見ない。

教師モードでは、教師の forward にだけ system メッセージとして前置する。生徒の `input_ids` は変わらないので
`max_prompt_length` も `truncation` も無関係。教師の系列が 193–199 トークン伸びるが、`response_only_logits=True`
で lm_head は応答行にしか掛からず、`timing_s/teacher_forward` は 0.47–0.85 秒（`update_actor` 226–306 秒に対し
無視できる）なので、前置 KV の再利用は不要。

生徒モードでは、**生成時のプロンプトに入る**。生徒の log-prob は生成した方策のものでなければならないため、
update 時だけ足すことはできない。したがって**軌跡そのものが変わり、報酬も変わる**。教師モードが KL の目標だけを
動かすのに対し、生徒モードははるかに大きな介入である。

### 3.2 コンテキスト予算（生徒モードのみ）— 実装時に訂正

**初版の前提が誤っていた。** per-task の上限（`data.task_overrides.<task>.max_prompt_length` と `truncation=error`）は
**データセットの初期プロンプトのフィルタにしか効かない**（`rl_dataset.py`）。rollout の各ターンでプロンプトを
トークン化する `TrajectoryCollector.preprocess_single_sample` は**グローバルの** `data.max_prompt_length=4096` と
`data.truncation=left` を使う。したがって「alfworld で 2048 を超えると run が死ぬ」は成立せず、超過時は
**左から黙って切り詰められる**。文書は system メッセージとして先頭にあるので、切り詰めが起きれば**文書が最初に消える**。

実測プロンプト長（system 文書を除く）と、文書を足した最大:

| task | 平均 | 最大 | 文書後の最大 | 4096 までの余裕 |
|---|---:|---:|---:|---:|
| alfworld | 521 | 1249 | 1442 | 2654 |
| search | 699 | 2214 | 2410 | 1686 |
| webshop | 1265 | 2554 | 2753 | 1343 |

どのタスクも 4096 に対して 1300 トークン以上の余裕があるので、**上限は変えない**（初版の 2241 / 2197 は撤回）。
上限を変えないことで、両アームと control の実効予算は自動的に同一になる。代わりに、文書が切り詰められた行の割合を
`notice/truncated_frac/<task>` として毎 step 出し、0 であることを確認する（§4）。

**教師モードでは**文書は生徒の `input_ids` に入らず、教師 forward の系列だけが伸びる。そちらの上限は
`log_prob_max_token_len_per_gpu=18432` で、無関係。

### 3.3 config

```
algorithm.opd.privileged_notice.enable: true
algorithm.opd.privileged_notice.apply_to: [student]      # [teacher] / [student] / 両方
algorithm.opd.privileged_notice.variant: named           # named / placebo
algorithm.opd.privileged_notice.doc_sha256: {alfworld: ..., search: ..., webshop: ...}
algorithm.opd.privileged_notice.effect_probe_every: 5    # §4 診断 1 の頻度
algorithm.opd.transfer_ladder.enable: true               # §4 の ladder（重み付けなし）
algorithm.opd.transfer_ladder.base_path: Qwen/Qwen3-1.7B
```

既定は `enable: false` で、そのとき control と bit 一致。**本文はファイルではなくコード**
（`verl/trainer/ppo/privileged_notice.py`）に置く — ファイルは編集される物だから。`apply_to` が空の `enable: true` は
起動時に拒否する（誰にも見せない通知は、アームの名で control を走らせることになる）。

**実装（2026-09-05）。** 生徒モードの挿入点は `TrajectoryCollector.preprocess_single_sample`（全ターンがここを通る）で、
行ごとに `notice_len`（通知のトークン数、無ければ 0）と `notice_truncated`（プロンプトが上限に達した = 通知が
左から切られた）の 2 列を載せる。教師モードは全教師 forward の漏斗 `_teacher_call` の**呼び出し側 2 箇所**（rollout 後の
on-task 呼び出しと prefetch の on-task 呼び出し）で on-task 教師にだけ前置し、worker 側で `fingerprint_adjust` を
引いて生徒の行として cache に登録する。ladder は `transfer_ladder.enable` で sign_weight なしに planes を読む
第 4 の消費者として driver/actor 両方にゲートを足した。

---

## 4. run 内診断（事前検定を行わないので必須）

**事前検定を外した以上、null が出たときに「文書が効かなかった」のか「効いたが性能に繋がらなかった」のかを
run 内で分けられなければならない。** 最低限、次の 3 つを毎 step 出す。

1. **文書の効果量**。同じトークン列に対し文書ありと無しで生徒の分布を 2 回読み、per-token の KL とその per-task /
   per-role 平均。forward 1 回の追加で、`update_actor` に対して無視できる。**これが 0 に近ければ機構は空振りで、
   それ以上の解釈は不要になる。**
2. **語彙漏れ率**。§0 と同じ判定（他タスクの行動構文 5 種 + 固有語彙）を学習中の応答に対して毎 step。
   ベースラインが 0 であることは測ってあるので、これは**床の確認**であって改善の余地の測定ではない。
   本アームで 0 のままなら、機構は「既に無いものを除去した」ことになる。
3. **応答長と entropy**。文書が語彙ではなく文体を変えている場合、そこに出る。

加えて、既存の off-task ladder（`transfer/off_travel/<dst>__on__<src>`）を両アームで有効にする。
文書が「輸入するな」として効いているなら off_travel は**上がる**はずで、これは方向の確認になる。

**指標名（実装）。** 診断 1 = `notice/effect_kl`, `notice/effect_kl/<task>`（`effect_probe_every` step ごと、
epoch 0 の最初の micro-batch で、通知を剥がした同じ生徒を同じ top-k で読み直した reverse KL）。
診断 2 = `notice/leak_rate`, `notice/leak_rate/<task>`（他タスクの行動構文 5 種; 語彙リストは §0 で
偽陽性だったので使わない）。床 = `notice/truncated_frac`, `notice/truncated_frac/<task>`, `notice/prefix_tokens/<task>`。
ladder = `transfer/off_travel/<dst>__on__<src>` と `transfer/kl_to_off/{base,on,stu}/<pair>`。

---

## 5. 事前登録

### 5.1 対照の組み方

同一コミット・同一ホスト・同一 GPU 枚数・$\beta = 0.01$。**両アームとも文書を持つ**ので、どちらも既存の
control run とは比較できない。2 本とも新規に走らせる。150 step で 1 本あたり約 30 時間、**合計約 60 時間**。

評価はタスク別成功率のみ。pooled は報告しない。反復雑音は理論文書 §4.2.2 の直接測定を使う:
1 draw 対 1 draw の差の SE は alfworld 1.96pp、webshop acc 0.65pp、webshop score 2.01pp、search 0.30pp（二項）。
**2 SE がそれぞれ 3.9 / 1.3 / 4.0 / 0.6pp** で、これを下回る差は「観測されなかった」と同義とする。

### 5.2 予測

| | 予測 | 外れたときの読み |
|---|---|---|
| **主要** | **本アーム ≈ プラセボ、全タスクで 2 SE 以内** | §0 の測定（漏れ率 0）から導かれる既定の予測 |
| 語彙漏れ率 | 両アームとも 0 のまま | 0 でなければ §0 の測定が検証標本に依存していたことになる |
| 文書の効果量（診断 1） | **不明。ここが唯一の未知である** | 0 に近ければ機構は空振り。大きければ「効いたが性能に繋がらない」 |
| off_travel | 両アームとも control@300 の 1.07–2.11 と同等かやや上 | 下がれば文書が逆向き（素直化）に効いている |
| 本アーム > プラセボ（2 SE 超） | **これが出たら重要。** 複数タスクの明示だけが差を生んだことになる | — |
| 本アーム < プラセボ（2 SE 超） | 名指しが 1.7B の容量を食っている | — |

### 5.3 この run で答えられないこと

* **文書が効かない理由**が「1.7B が指示に従えない」のか「従ったが除去対象が無い」のかは、診断 1 で分かれる。
  効果量が 0 なら前者、非 0 で漏れ率が 0 のままなら後者。
* **教師モードの結果**。生徒モードを先に走らせるので、教師モードは別途 2 本（さらに 60 時間）。
* **train/test 不一致の寄与**。生徒は毎 rollout で文書を見て学習し、評価では見ない。差が出た場合、
  それが文書の内容によるのか不一致によるのかは、この設計では分けられない。

---

## 6. 限界

* **主要な前提が偽であることを測ってある**（§0）。null が最も可能性が高く、その場合に得られるのは
  「複数タスクの明示は、汚染が無い状況では効かない」という限定的な否定的結果である。
* **生徒モードは軌跡を変える**ので、差が出ても「文書が方策を変えた」のか「違う軌跡を学習した」のかが分かれない。
  帰属の一意性は教師モードの方が高い。それを承知で生徒モードを先にする、という決定である。
* **train/test 不一致**が構造的に入る（§5.3）。
* **長さが 44–47 トークンずれる**（named 193–199 対 placebo 149–152）。プラセボ A を置かないので、
  長さの効果は残留交絡として報告に添える。
* **ホスト RAM が未解決**。`used` が 197–243 GB で天井（251 GB、Ray 閾値 246 GB）に近く、
  カリキュラム run はこれで 2 度落ちて 27 step を失った。環境 worker が anonymous 122 GB（webshop 121 個で
  75.3 GB、alfworld 120 個で 47.1 GB）を占める。**60 時間を投じる前に潰す価値があるが、2026-09-05 の決定で保留。**
* **監査の予測と整合的**: 教師間で共有されるのは書式で、書式は advantage と 0 か負の関係（§4.3、§4.6）。
  この文書が文体しか変えないなら、既存の測定から null が予測される。
