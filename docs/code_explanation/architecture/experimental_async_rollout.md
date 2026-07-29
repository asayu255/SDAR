# 実験的async rollout

`agent_system/multi_turn_rollout/async_rollout_core.py` はtorch、vLLM、具体envに依存しない
control-flow coreである。trajectoryごとに coroutine を作り、各trajectory内部では
`generate_action → env_step` を逐次実行しつつ、trajectory間を並行化する。

## 保持する不変条件

- trajectory `i` の次actionは同じtrajectoryの直前observationだけに依存する。
- `done` または `max_steps` で終了し、終了後のdummy recordを残さない。
- 完了時刻ではなく `results[i]` に格納し、入力trajectory順を保つ。
- GRPO用 `uid` は連続する `rollout_n` trajectoryで共有する。
- `traj_uid` はtrajectoryごとに一意で、turn row再集約に使う。

`max_in_flight` semaphoreは同時generate数だけを制限する。収集内容を変えずengine pressureを
調整する設計であり、`finally` で必ずcounterとpermitを戻す。

## 現在位置

このcoreとCPU等価性testは分岐に存在するが、module docstringが述べる
`async_rollout_loop.py` の本統合はこの固定sourceには存在しない。したがって本番Pure OPD
経路は既存のmulti-turn collectorであり、この機構を有効済みと解釈してはならない。
文書・テストは実験的設計と将来integrationの契約を示す。
