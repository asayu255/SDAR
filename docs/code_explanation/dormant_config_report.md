# Dormant / 非実効設定レポート

「設定キーが存在する」「共有コードが読む」「Pure OPDの実効scalar lossへ届く」を分離した。ここでdormantは削除候補という意味ではなく、この固定Pure OPD実験で非実効という分類である。

| setting / path | configured state | classification | evidence-based effect |
|---|---|---|---|
| `use_invalid_action_penalty` / task別係数 | run scriptで有効 | **monitoring-only** | reward tensorとmetric/dumpを変えるが、thin loopはadvantage/returnsを作らずOPD KLへ渡さない |
| `algorithm.use_kl_in_reward` | `main_opd.py`がfalseを注入 | **disabled** | reference-policy reward KL phaseを実行しない |
| `actor_rollout_ref.actor.pg_loss_coef` | `main_opd.py`が0.0を注入 | **disabled** | GRPO/PPO policy-gradientをscalar lossへ加算しない |
| `actor_rollout_ref.actor.entropy_coeff` | `main_opd.py`が0.0を注入 | **disabled** | entropyは観測可能でもscalar loss係数は0 |
| `actor_rollout_ref.actor.use_kl_loss` | `main_opd.py`がfalseを注入 | **disabled** | 通常actor KL regularizerではなくteacher top-k KLだけを使う |
| `use_sdl_loss` / `use_sdar_loss` | `main_opd.py`がfalseを注入 | **disabled** | 共有actorに実装はあるがPure OPD scalar lossへ到達しない |
| `ROLLOUT_PREFETCH_LOGPROB` | 共有rolloutに実装、Pure OPD runは無効化 | **dormant-in-pure-opd** | old-log-probを前倒しするがthin loopに消費phaseがない |
| critic / value micro-batch設定 | base PPO configに存在 | **dormant-in-pure-opd** | Pure OPD thin loopはcritic value・returnsを計算しない |
| dynamic batch size / token budget | 共有workerに実装、現在runはfalse | **disabled-performance-path** | 現在は固定actor/teacher micro-batchを使用 |
| `async_rollout_core`関連構想 | coreとCPU testのみ存在 | **experimental-only** | 固定sourceではproduction multi-turn loopへの完全接続がない |

## 実効loss

Pure OPDのgradientへ寄与する中心項は、studentが生成したresponse上でtask-specific teacher signalから計算するtop-k teacher KLである。monitoring rewardはログ・dumpには残るが、その事実だけではloss signalにならない。

## 注意

`compute_teacher_log_probs()`のsource APIは入力batch mutationである一方、routing testは戻り値をTensorとして扱う不一致がある。これは設定休眠ではなくsource/test ambiguityとして`ambiguity_report.md`に分離した。
