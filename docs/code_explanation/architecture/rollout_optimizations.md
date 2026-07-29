# rollout最適化と意味論

このブランチの最適化は、env reset prefetch、old-log-prob prefetch、並行 env step、
rollout engine の batching を含む。最適化の可否は速度ではなく、生成 token・mask・UID・
row ordering が基準経路と同じ契約を満たすかで判断する。

## reset prefetch

次 step の reset を現在 step の後段処理と重ねる。reset 回数を増やさず時刻だけを前倒し
するため、stateful env の schedule は維持される。checkpoint resume 側は「global step
あたり一回 reset」という不変条件を前提に fast-forward する。

## old-log-prob prefetch

rollout 後に必要な old log-prob 計算を他処理と重ねる候補である。ただし Pure OPD は
PPO ratio を実効 loss に使わないため、値を計算・保持していても scalar loss への寄与はない。
性能機構の存在と Pure OPD での効果を区別する必要がある。

## 等価性の観測点

- trajectoryごとの action/observation/reward/done 列
- `uid` と `traj_uid` の grouping
- response token、attention/response mask
- active trajectoryだけから作る turn row
- global batchへ戻したときの安定した順序

`tests/ray_cpu/test_async_rollout_equivalence.py` は engine非依存 core を決定的 callback で比較し、
並行実行が実際に発生したことと収集結果の一致を同時に検査する。
