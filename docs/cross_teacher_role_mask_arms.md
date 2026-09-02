# 役割マスク付き cross-teacher 重み付けアーム — 事前登録

実装: `verl/trainer/ppo/cross_teacher_kl_weight.py` の `ROLE_GROUPS` / `role_keep_mask`、
`build_position_weight(role_keep=...)`、`PreviousStepTaskKLWeightedMean.update(role_keep=...)`。
アーム: `examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_content_qwen3.sh`
（`role_mask=content`）。対照は無マスクアームが使ったものと**同一の** control
（`run_multitask_cross_teacher_klw_control_qwen3.sh`、project `..._cross_teacher_klw_xt1`、既に 298 step 完了）。

このファイルは走らせる**前**に書かれている。閾値を後から動かせないようにするためであり、
それがこのアームの唯一の存在理由（下記 H1/H2 の切り分け）を成立させる条件である。

## 0. 何を切り分けるのか

訂正後の 150step 検証（control 比）:

| task | control | 無マスクアーム | 差 |
|---|---:|---:|---:|
| alfworld | 0.651 | 0.714 | **+6.3pp** |
| search | 0.390 | 0.393 | +0.3pp |
| webshop score | 0.872 | 0.768 | **−10.4pp** |
| webshop acc | 0.722 | 0.667 | **−5.5pp** |

無マスクアームの**絶対介入量のうち内容 role に落ちた割合**（steps 121-148、role 別 KL シェア ×
pooled の role 別介入率）と、厳密な role 別 KL シェア:

| task | 内容介入シェア | tag KL シェア | 構造 KL シェア | entropy 比 | 差 |
|---|---:|---:|---:|---:|---:|
| alfworld | 0.165 | 0.766 | 0.874 | 2.45 | +6.3pp |
| search | 0.456 | 0.247 | 0.695 | 0.96 | +0.3pp |
| webshop | 0.661 | 0.195 | 0.449 | 1.74 | −5.5pp |

- Spearman(内容介入シェア, 差) = **−1.00**
- Spearman(構造 KL シェア, 差) = **+1.00**（近似を含まない列だけでも同じ順序）
- Spearman(entropy 比, 差) = +0.50 ← 以前の候補説明

n=3 で完全順序、偶然の確率 1/6 = 0.167。**検出力は不足しており、これは所見ではなく仮説である。**
生き残る読みは2つで、予測が逆になる。

| | 主張 | このアームへの予測 |
|---|---|---|
| **H1** | 内容半分が害の所在、構造半分が利得の所在 | webshop の損失を**再現**し、alfworld の利得は**再現しない** |
| **H2** | 介入量に比例して害があり、どんな制限でも最も制限された task が助かる | 両 task が control 方向に動く |

structural-only アーム単独ではこの2つを区別できない（「構造こそ信号」と「内容への介入が害/介入が
少ない方が良い」の両方と整合する）。content-only を先に走らせるのは、その予測が反証可能な向きに
明確だからである。

## 0.1 検証値の一次記録（2026-09-02 追記）

**検証は学習 run に載っていない。** 学習 run `xt1/91v55ri7`（name=`control`、
`cross_teacher_kl_weight.enable=False`、298 step）は `val/*` を1つも持たない。`test_freq=150` の
検証は独立した `val_only` run として別に記録される。§0 の表の control 値の出所は:

| run | ckpt | スクリプト | 位置づけ |
|---|---|---|---|
| `xt1/885xeeru` | `global_step_150` | control | **本物の control@150。§0 の表の control 列** |
| `xt1/bvl7inr6` | `global_step_300` | control | control@300 |
| `sg1/r09dbydu` | `global_step_150` | treatment | **新機構アーム@150。§0 の表の ARM 列** |
| `xt1/hdzm8277` | `global_step_150` | treatment | **旧機構アーム@150**（§0.2） |
| `xt1/9dfp1kav` | `global_step_300` | treatment | 旧機構アーム@300 |

`885xeeru` と `r09dbydu` の値は §0 の表と 4指標すべてで一致する
（0.6508 / 0.3897 / 0.8719 / 0.7222 と 0.7143 / 0.3933 / 0.7676 / 0.6667）。

**取り違えの再発を1件記録する。** この照合を per-instance ログで先にやろうとして
`~/val_instances/opd_grpo_multitask_cross_teacher_klw_qwen3_1.7b_xt1/` を control として読んだが、
`_control_` を含まないこのディレクトリは**処置（旧機構）**のものだった。offline audit §0.2 に
記録された取り違えと同じ罠である。集計手順そのものは正しく（`traj_uid` で畳んで max、ARM が
4指標一致）、読む対象を間違えていた。control 側の per-instance ログは
`..._klw_control_...` 名のディレクトリにあり、step 150 のファイルは残っていない。

## 0.2 旧機構との比較 — H1 に対する反証材料（2026-09-02 追記）

`xt1/hdzm8277` は**同じ control@150 に対する第2の処置測定**であり、2026-08-31 の3コミット
（`e6ef6a5` 教師類似度ゲート、`0b4b630` corroboration の連続化と生徒質量重み付け、
`4e2797e` チャネル分解）より**前**の機構である（作成 2026-08-30、`shuffled_to_live_gate_ratio` を
持たない）。sg1 は 08-31 15:53 作成で変更後。**両者は複製ではなく、別機構である。**

| 検証差（control@150 比） | 旧機構 `hdzm8277` | 新機構 `r09dbydu` |
|---|---:|---:|
| alfworld | **+0.0pp** | **+6.3pp** |
| search | −1.6pp | +0.4pp |
| webshop score | −16.0pp | −10.4pp |
| webshop acc | −12.7pp | −5.5pp |

**しかし両者の役割分布はほぼ同じである**（steps 121-148、`xt1/a7x8ko1r` 対 sg1）:

| | 旧機構 | 新機構 |
|---|---:|---:|
| tag KL シェア | 0.724 | 0.753 |
| **tag が絶対介入に占める割合** | **0.607** | **0.716** |

**したがって「tag に重み付けされたこと」だけでは alfworld の +6.3pp を説明できない。** 旧機構も
介入の 60.7% を tag に置いて +0.0pp だった。差は「どこ」ではなく「どれだけ・どれだけ選択的に」:

| | 旧機構 | 新機構 | 比 |
|---|---:|---:|---:|
| `kl_weight/effect/kl_shift_gross_frac` | 0.124 | 0.309 | 2.5× |
| `kl_weight/position/w_cv` | 0.187 | 0.402 | 2.2× |
| `kl_weight/effect/weight_kl_corr` | 0.054 | 0.129 | 2.4× |
| `kl_weight/evidence/shared_share` | 0.874 | 0.680 | 0.78× |
| `actor/entropy_loss` | 0.418 | 0.590 | 1.41× |

両方の観測を同時に満たす読みは **「tag は当てるべき場所だが、当てるだけでは足りず、十分な強さと
選択性が必要」** である。タスク間では tag シェアが利得を完全順序で並べる（+1.00）が、機構バージョン
間では tag シェアがほぼ同じで結果が分かれる。

なお webshop の悪化は**両バージョンで持続する**（−12.7pp → −5.5pp）。3コミットで半減したが消えて
おらず、klw 機構の性質と見るべきである。

### これが事前登録に足す条件

**tag-only アームは「場所」と「強さ」を同時に動かす。** マスクは総介入量 `gross_frac` を保つよう
正規化されるので（§2）、tag に落ちる介入は 0.716 倍から 1.0 倍へ**上がる**。上の表が示すとおり
強さは結果を分ける変数なので、tag-only の結果を「場所の効果」と読んではいけない。読むには
`kl_weight/role/tag/effect/kl_shift_gross_frac` を無マスクアームの値と並べ、強さの変化分を明示する
必要がある。

**webshop での再現は再マスクだけでは届かない。** webshop の tag KL シェアは 0.195（alfworld は
0.766）で、tag-only にしても webshop が tag に置ける絶対介入量は alfworld の 4分の1 のままである。
「alfworld の利得を webshop / search でも再現する」という目標は、総量を上げる変更を伴わない限り
この経路では達成できない。

## 1. 事前登録した閾値

σ は 4本の準複製 pure-OPD run から得た経験的 replicate SD:
alfworld 2.64pp / search 0.35pp / webshop acc 3.11pp。

| 指標 | H1 の予測 | 外れた場合 |
|---|---|---|
| webshop acc | control 比 **≤ −5pp** | H1 死亡 |
| alfworld | control 比 **< +3.7pp**（1σ） | H1 死亡 |
| search | 不変（±1σ = ±0.35pp 程度） | — |
| `episode/valid_action_ratio` | step 133 までに control の ~0.99 に**回復** | 形式破壊はタグ介入由来ではない |

`valid_action_ratio` の項の根拠: 無マスクアームは step 148 まで 0.00 のままで、control は
step 119 の 0.00 から step 133 の 0.99 まで滑らかに立ち上がった（alfworld −0.876、webshop −0.668、
search +0.001）。この指標は「行動できるか」ではなく**正規形式を採用したか**を測っている
（両 run とも step 1-119 は 0.00 のまま成功率が 0.62 まで上がる）。構造トークンに触らない
content-only で回復すれば、形式破壊がタグ介入由来だと確定し、**structural-only は走らせるべきでない**
と分かる。回復しなければ原因は別にある。

## 2. 結果ではなく配線の検証

走り始めた最初の数 step で読む。ここが崩れていれば結果は何も意味しない。

| 指標 | 期待値 | 崩れたときの意味 |
|---|---|---|
| `kl_weight/effect/kl_scale` | **1**（他の全 klw アームと同じ） | H2 を偶然測っている。結果は無効 |
| `kl_weight/role/tag/effect/kl_shift_gross_frac` | **厳密に 0** | マスクが weight に届いていない |
| `kl_weight/role/format/effect/kl_shift_gross_frac` | **厳密に 0** | 同上 |
| `kl_weight/role/{reasoning,tool_call,env_action,env_obs}/effect/kl_shift_gross_frac` | > 0 | マスクが反転している |

`kl_scale = 1` が保たれる理由は正規化器の `mu` を**保持 role のみ**で取っているためである
（`PreviousStepTaskKLWeightedMean.update` の docstring）。マスクされた半分は自分自身の
非重み付き KL をそのまま持つので、総蒸留量は変わらず、アームは control と
**「どこに再配分するか」でのみ**異なる。`mu` を全位置で取ると総蒸留量が変わり、アームは
「介入が少ない」ことを測ってしまう — それは切り分けようとしている H2 そのものである。

これは `tests/trainer/test_cross_teacher_role_mask.py` の
`test_kl_scale_stays_one_when_the_weight_and_the_normaliser_share_the_mask` と、
その失敗形を固定した `test_kl_scale_leaves_one_if_only_the_weight_carries_the_mask` で
テストされている。

## 3. 役割の定義

`verl/trainer/ppo/sign_weights.py` の `token_roles()` が per-token に付ける。

| group | role | 中身 |
|---|---|---|
| **structural** | `ROLE_TAG` | タグトークン自体（`<think>`, `</action>` …）。純粋な構文 |
| | `ROLE_FORMAT` | span の外側。空白、chat scaffolding、閉じタグ直後 |
| **content** | `ROLE_REASONING` | `<think>` の中 |
| | `ROLE_ENV_ACTION` | `<action>` の中 — env が実行する手 |
| | `ROLE_TOOL_CALL` | `<search>` / `<answer>` の中 — 呼び出しと最終回答 |
| | `ROLE_ENV_OBS` | `<information>` の中 — env が返した文 |

2 group は6 role を**排他的に分割する**（テスト済み）。未知の role コードは
どちらの group にも入らず**マスクされる**。包含で作っているためで、prompt に span 型が
追加されたときに黙って作用集合に入らないようにしてある。

## 4. 走らせる順序

1. **content-only**（このアーム）。上の閾値で H1 が生きるか死ぬかが決まる。
2. H1 が生きたら **tag-only**（`role_mask=tag`、`ROLE_GROUPS` に1行足すだけ）。**structural-only
   ではない** — §4.1 の通り tag と format は逆を向いている。予測は逆で、alfworld の利得を再現し
   webshop を悪化させない。読むときは §0.2 の強さの交絡を明示する。
3. H1 が死んだら、両アームとも止める。cross-teacher 系列に実証された価値が無いことになる。

### 4.1 structural を tag と format に割ると逆を向く（2026-09-02 追記）

| task | tag | format | 内容 | 利得 |
|---|---:|---:|---:|---:|
| alfworld | 0.766 | 0.107 | 0.126 | +6.3pp |
| search | 0.247 | 0.448 | 0.305 | +0.3pp |
| webshop | 0.195 | 0.254 | 0.551 | −5.5pp |

| | Spearman(シェア, 利得) |
|---|---:|
| **tag** | **+1.00** |
| format | **−0.50** |
| tag+format | +1.00 |
| 内容 | −1.00 |

structural（tag+format）は **+1.00 の予測子と −0.50 の予測子を混ぜる**ので、群として一貫しない。
ステップ2は tag-only にする。介入量も足りる: tag は pooled で絶対介入シェア 0.716 を占めるので、
tag-only でも介入の7割強が残る（structural の 0.831 と大差ない）。

**形式破壊は tag シェアに追随しない。** webshop は tag シェア最小（0.195）で
`valid_action_ratio` −0.668、search は 0.247 で無傷（+0.001）、alfworld は 0.766 で −0.876。
分けているのは介入量ではなくタグ語彙（`<action>` 対 `<search>`/`<answer>`）である。よって
tag-only が自動的に形式に悪いとは言えないが、監視項目として §1 の表に残す。

## 5. 未解決のまま残す点

- **交絡が残っている。** 内容シェアは task 型と絡んでいる（webshop は段階評価 score で内容が重い、
  alfworld は二値でタグが重い）。n=3 では分離できない。
- **shuffled プラセボは別の問いである。** 無マスクアームでは shuffled 教師が介入量の 88〜93% を
  再現した（alfworld 0.278 / live 0.312）。「信号は位置固有か」はこのアームでは答えられない。
- **role 別 `shared_share` は逆向きだった。** 内容 role の証拠はほぼ純粋な共有チャネル由来
  （reasoning 1.000、tool_call 0.816）で、構造 role が最も固有証拠を持つ（tag 0.599、search の tag
  は 0.263）。これは content-only を支持する材料だが、`shared_share` は共有 vs 固有を測るもので
  文法 vs 知識ではない。タグ上の固有成分は各 off-task 教師が**自分のタスクのタグ語彙を押し込んで
  いる**可能性が高く、それは上の形式破壊の最有力な機構である。
- **alfworld の +6.3pp は未説明のままである。** 介入の 83.5% がタグ+format 上にあり、生徒は形式を
  既に正しく出している。知識転移ではありえず、正則化か entropy チャネルのいずれかだが、
  このアームはどちらかを決めない。§0.2 はさらに「tag に当てただけでは足りない」ことを示したので、
  未説明の範囲は広がっている。
- **3タスクでは完全順序が安すぎる（2026-09-02 追記）。** 構造的に無関係な3つの変数が同時に
  完全順序を達成する:

  | 候補 | alfworld | search | webshop | Spearman |
  |---|---:|---:|---:|---:|
  | tag KL シェア | 0.766 | 0.247 | 0.195 | +1.00 |
  | 検証−rollout 差（control@150） | −0.072 | −0.023 | +0.075 | 完全（−1.00） |
  | サブタスク数 | 6 | 2 | 0 | +1.00 |
  | entropy 比 | 2.45 | 0.96 | 1.74 | +0.50 |
  | 教師KL損失 | 0.207 | 0.115 | 0.185 | +0.50 |
  | token シェア | 0.703 | 0.042 | 0.255 | +0.50 |
  | 応答長 | 214 | 132 | 286 | −0.50 |
  | control 精度 | 0.651 | 0.390 | 0.722 | −0.50 |

  「alfworld > search > webshop」の順に並ぶ任意の変数が +1.00 を得る。**+1.00 は証拠ではない。**
- **正則化仮説は目標の可否を左右する（2026-09-02 追記）。** 2番目の候補は種類の違う説明である:
  「機構は正則化器で、alfworld は単にそれを最も必要としていた」。alfworld の検証は自身の T=1.0
  rollout を 7.2pp 下回る（検証は低温で有利なはずなのに下回る = 本物の汎化不足）。webshop は逆に
  7.5pp 上回る。tag 説なら webshop の tag シェアを上げれば利得が出るが、正則化説なら webshop は
  正則化を必要としていないので**何をどう重み付けしても利得は出ない**。切り分けは学術的な問題では
  なく、目標が達成可能かを決める。content-only が alfworld を再現すれば正則化説、しなければ
  tag 説が残る。正則化説を完全に潰すには**機構の外側の対照**（entropy bonus 等を同じ entropy
  軌跡に合わせる）が必要で、それは再現されれば cross-teacher 系列がこの効果に不要という結論になる。
- **control@150 → @300 で alfworld の汎化ギャップは自力で閉じている（2026-09-02 追記）。**
  control の alfworld は検証 0.6508（@150）→ 0.7381（@300）、webshop score は 0.8719 → 0.8238。
  つまり control@300 の 0.738 は **ARM@150 の 0.714 を上回る**。ARM の +6.3pp は「その予算での
  速さ」であって天井の優位とは限らない。sg1 は 148 step で止まっているので ARM@300 は不明。
- **search の検証単位は他タスクと異なる。** 7データセット（2wikimultihopqa / bamboogle / hotpotqa
  / musique / nq / popqa / triviaqa）の集計で、per-instance ログの `traj_uid` は 51,713 個
  （alfworld / webshop は 126）。集計値 +0.4pp の内側は ±3pp の振れを含む（bamboogle −2.8pp、
  2wikimultihopqa +1.6pp、nq −1.1pp）。「search は不変」はこの粒度での話である。
