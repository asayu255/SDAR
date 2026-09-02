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
2. H1 が生きたら **structural-only**（`role_mask=structural`、同じ実装、script と lock を
   同じやり方で派生させる）。予測は逆で、alfworld の利得を再現し webshop を悪化させない。
3. H1 が死んだら、両アームとも止める。cross-teacher 系列に実証された価値が無いことになる。

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
  このアームはどちらかを決めない。
