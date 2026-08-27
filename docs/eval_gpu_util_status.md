# 評価の GPU 利用率 —— 現状(2026-08-27)

`docs/eval_gpu_utilization.md` は時系列の実験ノート、
`docs/eval_performance_summary.md` は打ち手の結論集である。
**この文書は「いまどこに居るか」だけを、外れた仮説も含めて 1 枚にしたもの。**

対象: `trainer.val_only=True` の multitask 評価
(`examples/sft_trainer/eval_checkpoints.sh`、Qwen3-1.7B、A6000 × 3)。

---

## 1 行

**util 79.9%、到達可能な上限 91.1%、差 11.2 pt。そのうち銀行に入った分は 0。**

---

## 2. 数字

同じ推定器(wandb events stream、15 秒サンプル)、**同じ起動後 24 分窓**で揃えた。

| | depth 2 | depth 3 | + pump |
| --- | ---: | ---: | ---: |
| node util | 78.19% | 79.58% | **79.90%** |
| スパイク(node < 60%) | 18.4% | 16.9% | **15.8%** |
| ノード停止(< 5%) | 6.6% | 2.6% | 6.6% |
| **3 枚とも忙しいときの util** | — | 89.66% | **91.07%** |

**スパイクは消えていない。** 3 構成で 18.4% → 15.8%、util +1.7 pt。

---

## 3. 空きの内訳と、それぞれの状態

pump run、起動後 210 サンプル:

| 種類 | 割合 | util 換算 | 正体 | 状態 |
| --- | ---: | ---: | --- | :---: |
| **PARTIAL**(1〜2 枚) | 7.5% | 3.11 pt | collective generate の尾。rank が自分の chunk を終えて最遅 rank を待つ | **確定** |
| **EMPTY / DEEP** | 5.2% | 8.06 pt | **箱の外を待っている**(retriever か driver の Python か未確定) | **未確定** |
| **EMPTY / SHALLOW** | 3.3% | (上に含む) | 15 秒サンプルのエイリアシング。GPU は窓の大半で動いている | 確定 |
| **DUTY**(3 枚とも忙しい) | — | 約 9 pt | vLLM の 1 step ごとのホスト処理。4.5ms の GPU に 0.5〜0.8ms が露出 | 確定 |

### PARTIAL が確定している理由

**機構が予測した動きを 2 回とも見せた。** depth を上げても動かず(collective の
中なので届かない)、pump を入れると半減した(6.79 pt → 3.11 pt)。

### DEEP EMPTY で分かっていること

power < 130W が続く 11 サンプル(165 秒)を busy と突き合わせた:

| | DEEP EMPTY | busy |
| --- | ---: | ---: |
| GPU sm util | **1.5%** | 91.7% |
| GPU power | **87.9 W** | 288.4 W |
| **GPU メモリコントローラ** | **0.8%** | 78.4% |
| host CPU | **0.6%** | **0.6%** |
| スレッド数 | 868 | 856 |
| disk read | 0 | 0 |

**GPU は計算も転送もしていない。ホストも動いていない。スレッドは全部生きている。**
計算待ちでも帯域待ちでも I/O 待ちでもなく、**プロセス全体が箱の外を待っている形**。

**断定しない理由:** wandb の `system.cpu` は busy でも DEEP EMPTY でも 0.6% で、
**GIL を握った Python 1 本とソケット待ちを見分ける分解能が無い。**

---

## 4. 効いたもの、効かなかったもの

| 変更 | util | 判定 |
| --- | ---: | --- |
| **retriever の修正**(`:8000` / 窓 100ms / connect 5s) | **57.9% → 79.0%** | **唯一の大きな勝ち** |
| socket 上限(`TCP_USER_TIMEOUT`) | 79.0 → 79.1% | 効果なし(裾しか消さない) |
| `VAL_PIPELINE_DEPTH=3` | 78.2 → 79.6% | EMPTY は消えた、util は動かず |
| `ROLLOUT_ASYNC_GENERATE=1`(pump) | 79.6 → 79.9% | PARTIAL は半減、util は動かず |
| `ROLLOUT_GPU_MEM_UTIL=0.75` | — | 余白のみ(今のバッチでは効かない) |

**retriever を直して以降、util は 0.9 pt しか動いていない。**

---

## 5. 空きが保存される —— 3 回とも

| 変更 | 下がったもの | 上がったもの |
| --- | --- | --- |
| depth 3 | EMPTY 8.9 → 3.8% | PARTIAL 7.7 → 12.7% |
| pump | PARTIAL 6.79 → 3.11 pt | EMPTY 3.29 → **8.06 pt** |

**pump は確定した原因を機構どおり半減させ、3.68 pt を実際に取り戻し、
4.77 pt を EMPTY に渡した。** バケツ収支 −1.09 pt。
util が +0.32 に見えるのは基準値(3 枚とも忙しいときの util)が
89.66 → 91.07 に上がったぶんだけである。

**GPU 側を速くしても GPU 外の仕事は減らない。露出が増えるだけである。**

---

## 6. util 以外で得たもの

| | depth 2 完走 | pump run 完走 |
| --- | ---: | ---: |
| **wall** | 1.26 h | **0.96 h** |
| success_rate | 0.3869 | 0.3875 |
| search | 0.3523 | 0.3571 |
| alfworld | 3.649 | 3.634 |
| webshop | 5.938 | 5.696 |

**24% 短い。スコアはほぼ動いていない。**
ただし depth・`gpu_memory_utilization`・timeout・pump が同時に変わっているので、
**pump 単体の効果ではない**(単一変数になっていない)。

**util を目的にすると「効果なし」、wall を目的にすると「今回いちばん効いた」。
同じ run が両方の答えを出している。**

---

## 7. 計器 —— 3 つあり、3 つとも別の盲点を持っていた

| 計器 | 見えるもの | 見えないもの |
| --- | --- | --- |
| `[val-pipeline]` の `NOTHING running` | slot が batch の中にいるか | **env.step で止まった slot を「実行中」と数える。** 285 秒のノード停止を 0.1% と報告した |
| `genGPU%` | `generate` の中の util | generate と generate の間 |
| wandb のチャート | スパイクが有ること | 15 秒グリッドが PARTIAL を水増し(8.0% を 14.6% と読む)。0 枚と 1 枚を区別しない。token を見ない |
| **`[gpu-residency]`**(今回追加) | 0.3 秒間隔で「何枚に仕事があったか」、EMPTY と PARTIAL の分離、per-GPU の偏り、**EMPTY の理由** | — |

**この盲点が判断を狂わせた実例:** async(pump)の的を「0.4%」と見積もったのは
`[val-pipeline]` の数字から。device に聞けば PARTIAL だけで 6.79 pt だった。

---

## 8. 次の 1 手 —— これだけ

**残り 5.2% の DEEP EMPTY に名前が付くまで、GPU 側の変更はしない。**
名前の付いていない相手に対して既に 3 回同じことをやった。

`6f563cb` 以降、ログが名指しする:

```bash
grep 'EMPTY is' /tmp/<log>
```

| 出力 | 意味 | 打ち手 |
| --- | --- | --- |
| `EMPTY is Python on the driver` | GIL | **GPU 側の変更は全部同じ結果になる。** driver の Python 仕事そのものを減らす |
| `EMPTY is a wait OFF the box (retriever, RPC)` | retriever | 複製、または Flat index の置き換え —— **どちらもこのリポジトリの外** |

**いまの証拠(DEEP EMPTY で host CPU が busy と同じ 0.6%、disk 0、
メモリコントローラ 0.8%)は後者に傾いている。** もしそうなら、
**このリポジトリのコードでは util はこれ以上上がらない。**

---

## 9. 天井

```
到達可能な上限(3 枚とも忙しいときの util)   91.07%
いま                                        79.90%
差                                          11.2 pt   ← 全部が賞金
そのうち銀行に入った分                       0 pt
```

91.07% より上は vLLM の 1 step ごとのホスト処理で、**vllm 0.8.5 には回せる
つまみが無い**(`async_scheduling` は 0.10.2 以降。`num_scheduler_steps` と
`disable_async_output_proc` は V0 専用)。更新は可能だが別の実験である。

---

## 10. 自分が間違えた点(記録)

| | 間違い | 訂正 |
| --- | --- | --- |
| 1 | 40 秒の停止を「connect の中」と帰属 | ログの「2 回」が否定。詰まった socket、connect の先 |
| 2 | async の的を「0.4%」と見積もり | 壊れた計器の数字。実測 6.79 pt |
| 3 | 「88〜90% が天井」 | 物理ではない。engine のホスト処理で、隠す機構は存在する |
| 4 | `grep TOTAL \| tail -3` を切り分けの根拠に | 別の batch を比べていた(promptTok が 6 倍違う)。§5 罠 3 と同じ |
| 5 | util を目的関数に最適化(3 回目) | 効いたかは wall と ms/row でしか言えない |
